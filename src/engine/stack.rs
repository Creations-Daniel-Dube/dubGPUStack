// SPDX-License-Identifier: GPL-3.0-or-later
//! Empilement par bandes : drizzle ou translation, puis rejet sur le GPU.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use super::fits::{self, ImageInfo};
use super::lights::Light;
use super::register::Transform;
use crate::gpu::Gpu;

pub const NAN_BITS: u32 = 0x7fc0_0000;

#[derive(Clone, Debug)]
pub struct SampleFrame {
    pub x0: i32,
    pub y0: i32,
    pub w: i32,
    pub h: i32,
    pub a: [f64; 3],
    pub off: [f64; 3],
    pub imgw: [f64; 3],
    pub mask: Option<Vec<f32>>,
    pub mask_w: i32,
    pub mask_h: i32,
    pub source: SampleSource,
}

#[derive(Clone, Debug)]
pub enum SampleSource {
    Shift { path: PathBuf, weight: Option<PathBuf> },
    Raw { light: Light, transform: [f64; 6], lut: Option<[i32; 4]>, pixfrac: f32 },
}

#[derive(Clone, Debug)]
pub struct StackParams {
    pub channels: usize,
    pub rejection: i32,
    pub sigma_low: f32,
    pub sigma_high: f32,
    pub feather: f32,
    pub plain: bool,
    pub strip: usize,
    pub tile: usize,
    pub want_coverage: bool,
}

pub struct Stacked {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub rejected_low: f64,
    pub rejected_high: f64,
    pub coverage: Option<Vec<u16>>,
}

pub fn tune(frames: usize, width: usize, limit: usize, mut strip: usize, mut tile: usize) -> (usize, usize) {
    tile = tile.min(width.max(32));
    let k = frames.max(1);
    let budget = (limit as f64 * 0.55) as usize;
    let bytes = |strip: usize, tile: usize| k.saturating_mul(strip).saturating_mul(tile).saturating_mul(20).saturating_add(strip.saturating_mul(tile).saturating_mul(32));
    while bytes(strip, tile) > budget && (strip > 8 || tile > 128) {
        if tile > 256 {
            tile /= 2;
        } else if strip > 8 {
            strip /= 2;
        } else {
            tile = (tile / 2).max(32);
            break;
        }
    }
    (strip.max(1), tile.max(32))
}

pub fn stack_region(
    gpu: &Gpu,
    frames: &[SampleFrame],
    region: (i32, i32, i32, i32),
    params: &StackParams,
    cancel: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<Stacked, String> {
    let (x_lo, y_lo, x_hi, y_hi) = region;
    let width = (x_hi - x_lo).max(0) as usize;
    let height = (y_hi - y_lo).max(0) as usize;
    let channels = params.channels.max(1);
    if width == 0 || height == 0 {
        return Err("la zone à empiler est vide".to_string());
    }
    if frames.len() < 2 {
        return Err("il faut au moins deux images".to_string());
    }
    let mut data = vec![0.0f32; channels * height * width];
    let mut coverage = if params.want_coverage { Some(vec![0u16; height * width]) } else { None };
    let mut nlo = 0i64;
    let mut nhi = 0i64;
    let mut nvalid = 0i64;
    let strips = height.div_ceil(params.strip.max(1));
    let mut done_strips = 0usize;
    let mut y = 0usize;
    while y < height {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let rows = params.strip.min(height - y);
        let mut x = 0usize;
        while x < width {
            let cols = params.tile.min(width - x);
            let tile = stack_tile(gpu, frames, x_lo + x as i32, y_lo + y as i32, cols, rows, channels, params)?;
            for channel in 0..channels {
                for row in 0..rows {
                    let src = (channel * rows + row) * cols;
                    let dst = (channel * height + y + row) * width + x;
                    data[dst..dst + cols].copy_from_slice(&tile.values[src..src + cols]);
                }
            }
            if let Some(map) = coverage.as_mut() {
                for row in 0..rows {
                    for col in 0..cols {
                        map[(y + row) * width + x + col] = tile.kept[row * cols + col].min(u16::MAX as i32) as u16;
                    }
                }
            }
            nlo += tile.nlo as i64;
            nhi += tile.nhi as i64;
            nvalid += tile.nvalid as i64;
            x += cols;
        }
        y += rows;
        done_strips += 1;
        progress(done_strips, strips);
    }
    let rejected_low = if nvalid > 0 { nlo as f64 / nvalid as f64 } else { 0.0 };
    let rejected_high = if nvalid > 0 { nhi as f64 / nvalid as f64 } else { 0.0 };
    Ok(Stacked { data, width, height, channels, rejected_low, rejected_high, coverage })
}

struct TileOut {
    values: Vec<f32>,
    kept: Vec<i32>,
    nlo: i32,
    nhi: i32,
    nvalid: i32,
}

fn stack_tile(gpu: &Gpu, frames: &[SampleFrame], origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize, params: &StackParams) -> Result<TileOut, String> {
    let pixels = rows * cols;
    let k = frames.len() as i32;
    let mut values = vec![0.0f32; channels * pixels];
    let mut kept = vec![0i32; pixels];
    let mut nlo = 0i32;
    let mut nhi = 0i32;
    let mut nvalid = 0i32;
    let raw = matches!(frames.first().map(|frame| &frame.source), Some(SampleSource::Raw { .. }));
    if raw {
        for channel in 0..channels {
            let mut dst_v = gpu.zeros(pixels * frames.len())?;
            let mut dst_w = gpu.zeros(pixels * frames.len())?;
            gpu.fill(&mut dst_v, NAN_BITS)?;
            gpu.fill(&mut dst_w, 0)?;
            let mut num = gpu.zeros(pixels)?;
            let mut den = gpu.zeros(pixels)?;
            for (slot, frame) in frames.iter().enumerate() {
                let SampleSource::Raw { light, transform, lut, pixfrac } = &frame.source else { continue };
                if !overlaps(frame, origin_x, origin_y, cols as i32, rows as i32) {
                    continue;
                }
                gpu.fill(&mut num, 0)?;
                gpu.fill(&mut den, 0)?;
                let (y0, y1) = input_rows(transform, light.height, *pixfrac, origin_x, origin_y, cols as i32, rows as i32);
                if y1 <= y0 {
                    continue;
                }
                let info = ImageInfo::open(&light.path)?;
                let plane = if lut.is_some() { 0 } else { channel.min(info.channels - 1) };
                let image = fits::read_rows(&info, plane, y0 as usize, y1 as usize)?;
                gpu.drizzle_into(&image, light.width as i32, (y1 - y0) as i32, y0, *lut, if lut.is_some() { channel as i32 } else { -1 }, *transform, (*pixfrac as f64) * 0.5, origin_x, origin_y, cols as i32, rows as i32, &mut num, &mut den)?;
                let mask = upload_mask(gpu, frame)?;
                gpu.finish_into(&num, &den, pixels as i32, cols as i32, frame.a[channel], frame.off[channel], frame.imgw[channel], params.feather, mask.as_ref(), frame.mask_w, frame.mask_h, frame.w, frame.h, origin_x, origin_y, frame.x0, frame.y0, &mut dst_v, &mut dst_w, slot as i32, k)?;
            }
            if params.want_coverage && channel == 0 {
                let weights = gpu.download(&dst_w)?;
                for pixel in 0..pixels {
                    let mut n = 0i32;
                    for slot in 0..frames.len() {
                        if weights[pixel * frames.len() + slot] > 0.0 {
                            n += 1;
                        }
                    }
                    kept[pixel] = n;
                }
            }
            let rejected = gpu.reject(&dst_v, &dst_w, k, pixels as i32, params.rejection, params.sigma_low, params.sigma_high, params.plain)?;
            values[channel * pixels..(channel + 1) * pixels].copy_from_slice(&rejected.values);
            nlo = nlo.saturating_add(rejected.nlo);
            nhi = nhi.saturating_add(rejected.nhi);
            nvalid = nvalid.saturating_add(rejected.nvalid);
        }
    } else {
        for channel in 0..channels {
            let mut host_v = vec![f32::from_bits(NAN_BITS); pixels * frames.len()];
            let mut host_w = vec![0.0f32; pixels * frames.len()];
            for (slot, frame) in frames.iter().enumerate() {
                let SampleSource::Shift { path, weight } = &frame.source else { continue };
                if !overlaps(frame, origin_x, origin_y, cols as i32, rows as i32) {
                    continue;
                }
                let info = ImageInfo::open(path)?;
                let y0 = (origin_y - frame.y0).max(0).min(info.height as i32);
                let y1 = (origin_y + rows as i32 - frame.y0).clamp(0, info.height as i32);
                if y1 <= y0 {
                    continue;
                }
                let image = fits::read_rows(&info, channel.min(info.channels.saturating_sub(1)), y0 as usize, y1 as usize)?;
                let weights = read_weight_rows(weight.as_deref(), y0 as usize, y1 as usize, info.width)?;
                for row in 0..(y1 - y0) as usize {
                    let canvas_y = frame.y0 + y0 + row as i32;
                    let local_row = (canvas_y - origin_y) as usize;
                    if local_row >= rows {
                        continue;
                    }
                    for col in 0..cols {
                        let canvas_x = origin_x + col as i32;
                        let local_x = canvas_x - frame.x0;
                        if local_x < 0 || local_x >= info.width as i32 {
                            continue;
                        }
                        let v = image[row * info.width + local_x as usize];
                        let d = weights.as_ref().map(|w| w[row * info.width + local_x as usize]).unwrap_or(1.0);
                        let nv = (v as f64 * frame.a[channel] - frame.off[channel]) as f32;
                        if v == 0.0 || d == 0.0 || !v.is_finite() || nv == 0.0 {
                            continue;
                        }
                        let lx = local_x as f32;
                        let ly = (y0 as usize + row) as f32;
                        let fw = feather_of(frame, lx, ly, params.feather);
                        let index = (local_row * cols + col) * frames.len() + slot;
                        host_v[index] = nv;
                        host_w[index] = d * fw * frame.imgw[channel] as f32;
                    }
                }
            }
            if params.want_coverage && channel == 0 {
                for pixel in 0..pixels {
                    let mut n = 0i32;
                    for slot in 0..frames.len() {
                        if host_v[pixel * frames.len() + slot].is_finite() {
                            n += 1;
                        }
                    }
                    kept[pixel] = n;
                }
            }
            let dst_v = gpu.upload(&host_v)?;
            let dst_w = gpu.upload(&host_w)?;
            let rejected = gpu.reject(&dst_v, &dst_w, k, pixels as i32, params.rejection, params.sigma_low, params.sigma_high, params.plain)?;
            values[channel * pixels..(channel + 1) * pixels].copy_from_slice(&rejected.values);
            nlo = nlo.saturating_add(rejected.nlo);
            nhi = nhi.saturating_add(rejected.nhi);
            nvalid = nvalid.saturating_add(rejected.nvalid);
        }
    }
    Ok(TileOut { values, kept, nlo, nhi, nvalid })
}

fn overlaps(frame: &SampleFrame, x: i32, y: i32, w: i32, h: i32) -> bool {
    frame.x0 < x + w && frame.x0 + frame.w > x && frame.y0 < y + h && frame.y0 + frame.h > y
}

fn input_rows(transform: &[f64; 6], height: usize, pixfrac: f32, x: i32, y: i32, w: i32, h: i32) -> (i32, i32) {
    let t = Transform { m: *transform };
    let Some(inv) = t.inverse() else { return (0, height as i32) };
    let pad = 2.0 + pixfrac as f64;
    let corners = [(x as f64, y as f64), ((x + w) as f64, y as f64), (x as f64, (y + h) as f64), ((x + w) as f64, (y + h) as f64)];
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for (cx, cy) in corners {
        let (_, iy) = inv.apply(cx, cy);
        min_y = min_y.min(iy);
        max_y = max_y.max(iy);
    }
    let y0 = (min_y - pad).floor().max(0.0) as i32;
    let y1 = ((max_y + pad).ceil() as i32).clamp(0, height as i32).max(y0);
    (y0, y1.min(height as i32))
}

fn feather_of(frame: &SampleFrame, lx: f32, ly: f32, feather: f32) -> f32 {
    if !(feather > 0.0) {
        return 1.0;
    }
    let Some(mask) = frame.mask.as_ref() else { return 1.0 };
    if frame.mask_w <= 0 || frame.mask_h <= 0 || frame.w <= 0 || frame.h <= 0 {
        return 1.0;
    }
    let fx = frame.mask_w as f32 / frame.w as f32;
    let fy = frame.mask_h as f32 / frame.h as f32;
    let sx = ((lx + 0.5) * fx - 0.5).clamp(0.0, (frame.mask_w - 1) as f32);
    let sy = ((ly + 0.5) * fy - 0.5).clamp(0.0, (frame.mask_h - 1) as f32);
    let x0 = sx as i32;
    let y0 = sy as i32;
    let x1 = (x0 + 1).min(frame.mask_w - 1);
    let y1 = (y0 + 1).min(frame.mask_h - 1);
    let tx = sx - x0 as f32;
    let ty = sy - y0 as f32;
    let at = |x: i32, y: i32| mask[(y * frame.mask_w + x) as usize];
    let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
    let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
    dubgpustack_kernels::fmath::feather_weight(top * (1.0 - ty) + bot * ty, feather, &dubgpustack_kernels::fmath::ramp_table())
}

fn read_weight_rows(path: Option<&std::path::Path>, y0: usize, y1: usize, width: usize) -> Result<Option<Vec<f32>>, String> {
    let Some(path) = path else { return Ok(None) };
    if !path.is_file() {
        return Ok(None);
    }
    let info = ImageInfo::open(path)?;
    let mut out = vec![0.0f32; (y1 - y0) * width];
    let item = info.header.item_size()?;
    let mut file = std::fs::File::open(path).map_err(|err| err.to_string())?;
    use std::io::{Read, Seek, SeekFrom};
    let mut raw = vec![0u8; width * item];
    for (slot, y) in (y0..y1).enumerate() {
        let file_row = if info.bottom_up { info.height - 1 - y } else { y };
        let offset = info.header.data_offset + (file_row * info.width * item) as u64;
        file.seek(SeekFrom::Start(offset)).map_err(|err| err.to_string())?;
        file.read_exact(&mut raw).map_err(|err| err.to_string())?;
        for x in 0..width.min(info.width) {
            let bytes = &raw[x * item..(x + 1) * item];
            out[slot * width + x] = match info.bitpix {
                8 => bytes[0] as f32 * info.bscale + info.bzero,
                16 => i16::from_be_bytes([bytes[0], bytes[1]]) as f32 * info.bscale + info.bzero,
                _ => fits::decode_sample(info.bitpix, bytes, info.bscale, info.bzero),
            };
        }
    }
    Ok(Some(out))
}

fn upload_mask(gpu: &Gpu, frame: &SampleFrame) -> Result<Option<cudarc::driver::CudaSlice<f32>>, String> {
    match &frame.mask {
        Some(mask) if !mask.is_empty() => Ok(Some(gpu.upload(mask)?)),
        _ => Ok(None),
    }
}

pub fn normalize_output(data: &mut [f32], channels: usize, height: usize, width: usize) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                if c == 0 && y == height - 1 && x == 0 {
                    continue;
                }
                let v = data[(c * height + y) * width + x];
                if v != 0.0 && v.is_finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
    }
    if !lo.is_finite() || hi <= lo {
        return;
    }
    let lo32 = lo;
    let range = hi - lo32;
    for value in data.iter_mut() {
        if *value != 0.0 && value.is_finite() {
            *value = (*value - lo32) / range;
        }
    }
}

pub fn bin_preview(data: &[f32], channels: usize, height: usize, width: usize, factor: usize) -> (Vec<f32>, usize, usize) {
    let factor = factor.max(1);
    if factor == 1 {
        return (data.to_vec(), height, width);
    }
    let bh = height.div_ceil(factor);
    let bw = width.div_ceil(factor);
    let mut out = vec![0.0f32; channels * bh * bw];
    for c in 0..channels {
        for by in 0..bh {
            for bx in 0..bw {
                let mut sum = 0.0f64;
                let mut n = 0.0f64;
                for y in 0..factor {
                    for x in 0..factor {
                        let iy = by * factor + y;
                        let ix = bx * factor + x;
                        if iy < height && ix < width {
                            let v = data[(c * height + iy) * width + ix];
                            if v != 0.0 && v.is_finite() {
                                sum += v as f64;
                                n += 1.0;
                            }
                        }
                    }
                }
                if n > 0.0 {
                    out[(c * bh + by) * bw + bx] = (sum / n) as f32;
                }
            }
        }
    }
    (out, bh, bw)
}
