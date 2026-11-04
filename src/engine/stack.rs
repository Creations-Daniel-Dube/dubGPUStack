// SPDX-License-Identifier: GPL-3.0-or-later
//! Empilement par bandes : drizzle ou translation, puis rejet sur le GPU.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use super::fits::FrameFile;
use super::lights::Light;
use super::register::Transform;
use crate::gpu::{Gpu, Work};

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
    pub workers: usize,
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

/// Octets des valeurs et des poids d'un échantillon. Le grattage du rejet tient sur quelques lignes, pas sur tout le cube.
pub const SAMPLE_BYTES: usize = 8;
/// Lignes traitées d'un coup par le rejet. Le cube des poses, lui, couvre toute la bande.
pub const SCRATCH_ROWS: usize = 64;

/// Mémoire GPU utilisable. `manual_gib` est un plafond ; 0 prend la mémoire libre moins 1 Gio.
pub fn memory_budget(manual_gib: f64, free: usize, _total: usize) -> usize {
    let gib = 1024usize.saturating_mul(1024).saturating_mul(1024);
    let available = if free > gib { free - gib } else { (free / 2).max(1) };
    if manual_gib > 0.0 {
        let manual = (manual_gib * gib as f64) as usize;
        manual.min(available).max(1)
    } else {
        available.max(1)
    }
}

/// Moitié de la RAM libre, en laissant 2 Gio au système, pour les cubes hôte d'une bande.
pub fn host_budget() -> usize {
    let avail = available_ram().unwrap_or(8 * 1024 * 1024 * 1024);
    let reserve = 2 * 1024 * 1024 * 1024;
    avail.saturating_sub(reserve).max(512 * 1024 * 1024)
}

#[cfg(windows)]
fn available_ram() -> Option<usize> {
    #[repr(C)]
    struct MemStatus {
        length: u32,
        load: u32,
        total: u64,
        avail: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended: u64,
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemStatus) -> i32;
    }
    let mut status = MemStatus {
        length: std::mem::size_of::<MemStatus>() as u32,
        load: 0,
        total: 0,
        avail: 0,
        total_page: 0,
        avail_page: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended: 0,
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 { None } else { Some(status.avail as usize) }
}

#[cfg(not(windows))]
fn available_ram() -> Option<usize> {
    None
}

pub fn scratch_pixels(rows: usize, cols: usize) -> usize {
    rows.min(SCRATCH_ROWS).max(1).saturating_mul(cols.max(1))
}

pub fn peak_bytes(frames: usize, rows: usize, cols: usize, channels: usize, accum_planes: usize) -> usize {
    let pixels = rows.saturating_mul(cols);
    let frames = frames.max(1);
    let samples = pixels.saturating_mul(frames);
    let scratch = scratch_pixels(rows, cols).saturating_mul(frames).saturating_mul(20);
    samples
        .saturating_mul(SAMPLE_BYTES)
        .saturating_add(scratch)
        .saturating_add(pixels.saturating_mul(4))
        .saturating_add(pixels.saturating_mul(accum_planes).saturating_mul(8))
        .saturating_add(pixels.saturating_mul(channels.max(1)).saturating_mul(8))
}

pub fn host_bytes(frames: usize, rows: usize, cols: usize, channels: usize) -> usize {
    if channels == 0 || rows == 0 || cols == 0 {
        return 0;
    }
    rows.saturating_mul(cols).saturating_mul(frames.max(1)).saturating_mul(channels).saturating_mul(8)
}

/// Bande la plus haute qui tient en pleine largeur. `strip_cap` ou `tile_cap` à 0 lève le plafond.
/// `host_channels` compte les canaux gardés en RAM. Le premier canal brut reste dans le cube GPU.
pub fn geometry(
    frames: usize,
    width: usize,
    height: usize,
    channels: usize,
    accum_planes: usize,
    gpu_budget: usize,
    host_budget: usize,
    extra_gpu: usize,
    strip_cap: usize,
    tile_cap: usize,
    host_channels: usize,
) -> (usize, usize) {
    let width = width.max(1);
    let height = height.max(1);
    let frames = frames.max(1);
    let usable = {
        let left = gpu_budget.saturating_sub(extra_gpu);
        if left >= 1024 * 1024 { left } else { (gpu_budget / 2).max(1) }
    };
    let host_budget = host_budget.max(1);
    let tile_limit = if tile_cap == 0 { width } else { tile_cap.clamp(1, width) };
    let strip_limit = if strip_cap == 0 { height } else { strip_cap.clamp(1, height) };
    let floor_tile = 32.min(width).max(1);
    let fits = |rows: usize, cols: usize| {
        rows > 0 && cols > 0 && peak_bytes(frames, rows, cols, channels, accum_planes) <= usable && host_bytes(frames, rows, cols, host_channels) <= host_budget
    };
    let max_strip = |cols: usize| -> usize {
        if !fits(1, cols) {
            return 0;
        }
        let mut best = 1usize;
        let mut lo = 1usize;
        let mut hi = strip_limit;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid, cols) {
                best = mid;
                lo = mid.saturating_add(1);
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }
        best
    };
    let mut tile = tile_limit;
    loop {
        let strip = max_strip(tile);
        if (strip >= 16.min(strip_limit) && strip > 0) || tile <= floor_tile {
            return (strip.max(1), tile.max(1));
        }
        let next = (tile / 2).max(floor_tile);
        if next == tile {
            return (strip.max(1), tile.max(1));
        }
        tile = next;
    }
}

pub fn geometry_line(strip: usize, tile: usize, free: usize, total: usize) -> String {
    let gib = 1024.0 * 1024.0 * 1024.0;
    format!("bande {strip}×{tile} · {:.1} Gio libres sur {:.1}", free as f64 / gib, total as f64 / gib)
}

pub fn allowance(frames: &[SampleFrame]) -> usize {
    let image = frames
        .iter()
        .map(|frame| match &frame.source {
            SampleSource::Raw { light, .. } => light.width.saturating_mul(light.height).saturating_mul(4),
            SampleSource::Shift { .. } => 0,
        })
        .max()
        .unwrap_or(0);
    let masks: usize = frames.iter().map(|frame| frame.mask.as_ref().map(|mask| mask.len().saturating_mul(4)).unwrap_or(0)).sum();
    image.saturating_add(masks).saturating_add(32 * 1024 * 1024)
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
    let raw = matches!(frames.first().map(|frame| &frame.source), Some(SampleSource::Raw { .. }));
    let bayer = matches!(frames.first().map(|frame| &frame.source), Some(SampleSource::Raw { lut: Some(_), .. }));
    let mut holds = open_holds(frames)?;
    let device_masks = frames.iter().map(|frame| upload_mask(gpu, frame)).collect::<Result<Vec<_>, _>>()?;
    let max_pixels = params.strip.max(1).saturating_mul(params.tile.max(1));
    let image_len = frames
        .iter()
        .map(|frame| match &frame.source {
            SampleSource::Raw { light, .. } => light.width.saturating_mul(light.height),
            SampleSource::Shift { .. } => 1,
        })
        .max()
        .unwrap_or(1);
    let accum = if raw && bayer { channels } else { 1 };
    let scratch = scratch_pixels(params.strip.max(1), params.tile.max(1));
    let mut work = gpu.prepare(max_pixels, frames.len(), accum, channels, image_len, scratch)?;
    let nan = f32::from_bits(NAN_BITS);
    let host_channels = if raw { channels.saturating_sub(1) } else { channels };
    let mut host_v = vec![nan; host_channels.saturating_mul(max_pixels).saturating_mul(frames.len())];
    let mut host_w = vec![0.0f32; host_v.len()];
    let mut got_v = vec![0.0f32; channels.saturating_mul(max_pixels)];
    let mut got_w = vec![0.0f32; got_v.len()];
    let mut kept_buf = vec![0i32; max_pixels];
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
            let tile = fill_tile(&mut TileCtx {
                gpu,
                work: &mut work,
                holds: &mut holds,
                masks: &device_masks,
                frames,
                params,
                raw,
                bayer,
                host_v: &mut host_v,
                host_w: &mut host_w,
                got_v: &mut got_v,
                got_w: &mut got_w,
                kept: &mut kept_buf,
            }, x_lo + x as i32, y_lo + y as i32, cols, rows, channels)?;
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

fn fill_tile(ctx: &mut TileCtx<'_>, origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize) -> Result<TileOut, String> {
    let pixels = rows * cols;
    let count = ctx.frames.len();
    ctx.kept[..pixels].fill(0);
    if ctx.raw {
        pull_raw(ctx, origin_x, origin_y, cols, rows, channels, pixels)?;
        if ctx.params.want_coverage {
            ctx.gpu.coverage_slots(&ctx.work.dst_w, count as i32, pixels as i32, &mut ctx.kept[..pixels])?;
        }
        return reject_cube(ctx, pixels, channels);
    }
    let used = channels.saturating_mul(pixels).saturating_mul(count);
    ctx.host_v[..used].fill(f32::from_bits(NAN_BITS));
    ctx.host_w[..used].fill(0.0);
    let v_ptr = ctx.host_v.as_mut_ptr();
    let w_ptr = ctx.host_w.as_mut_ptr();
    let feather = ctx.params.feather;
    let workers = ctx.params.workers;
    pull_shift(ctx.holds, ctx.frames, origin_x, origin_y, cols, rows, channels, feather, workers, v_ptr, w_ptr)?;
    if ctx.params.want_coverage {
        for pixel in 0..pixels {
            let mut n = 0i32;
            for slot in 0..count {
                if ctx.host_v[pixel * count + slot].is_finite() {
                    n += 1;
                }
            }
            ctx.kept[pixel] = n;
        }
    }
    reject_host(ctx, pixels, channels)
}

struct TileCtx<'a> {
    gpu: &'a Gpu,
    work: &'a mut Work,
    holds: &'a mut [Hold],
    masks: &'a [Option<cudarc::driver::CudaSlice<f32>>],
    frames: &'a [SampleFrame],
    params: &'a StackParams,
    raw: bool,
    bayer: bool,
    host_v: &'a mut [f32],
    host_w: &'a mut [f32],
    got_v: &'a mut [f32],
    got_w: &'a mut [f32],
    kept: &'a mut [i32],
}

enum Hold {
    Raw(FrameFile),
    Shift(ShiftHold),
}

struct ShiftHold {
    image: FrameFile,
    weight: Option<FrameFile>,
    index: usize,
}

struct Loaded {
    y0: i32,
    width: i32,
    nrows: i32,
    planes: usize,
    data: Vec<f32>,
}

fn open_holds(frames: &[SampleFrame]) -> Result<Vec<Hold>, String> {
    let mut holds = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        match &frame.source {
            SampleSource::Raw { light, .. } => holds.push(Hold::Raw(FrameFile::open(&light.path)?)),
            SampleSource::Shift { path, weight } => {
                let image = FrameFile::open(path)?;
                let weight = match weight {
                    Some(path) if path.is_file() => Some(FrameFile::open(path)?),
                    _ => None,
                };
                holds.push(Hold::Shift(ShiftHold { image, weight, index }));
            }
        }
    }
    Ok(holds)
}

fn pull_raw(ctx: &mut TileCtx<'_>, origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize, pixels: usize) -> Result<(), String> {
    let n = ctx.frames.len();
    let samples = pixels.saturating_mul(n);
    ctx.gpu.fill_prefix(&mut ctx.work.dst_v, samples, NAN_BITS)?;
    ctx.gpu.fill_prefix(&mut ctx.work.dst_w, samples, 0)?;
    if channels > 1 {
        let used = (channels - 1) * samples;
        ctx.host_v[..used].fill(f32::from_bits(NAN_BITS));
        ctx.host_w[..used].fill(0.0);
    }
    let mut upcoming = load_raw(ctx, 0, origin_x, origin_y, cols, rows, channels)?;
    for slot in 0..n {
        let current = upcoming;
        if let Some(loaded) = current.as_ref() {
            queue_frame(ctx, loaded, slot, origin_x, origin_y, cols, rows, channels)?;
        }
        upcoming = if slot + 1 < n { load_raw(ctx, slot + 1, origin_x, origin_y, cols, rows, channels)? } else { None };
        if current.is_some() {
            if channels > 1 {
                let count = channels * pixels;
                ctx.gpu.copy_dtoh(&ctx.work.frame_v, count, &mut ctx.got_v[..count])?;
                ctx.gpu.copy_dtoh(&ctx.work.frame_w, count, &mut ctx.got_w[..count])?;
                for channel in 1..channels {
                    let src = channel * pixels;
                    let dst = (channel - 1) * samples + slot * pixels;
                    ctx.host_v[dst..dst + pixels].copy_from_slice(&ctx.got_v[src..src + pixels]);
                    ctx.host_w[dst..dst + pixels].copy_from_slice(&ctx.got_w[src..src + pixels]);
                }
            } else {
                ctx.gpu.sync()?;
            }
        }
    }
    Ok(())
}

fn load_raw(ctx: &mut TileCtx<'_>, slot: usize, origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize) -> Result<Option<Loaded>, String> {
    let frame = &ctx.frames[slot];
    if !overlaps(frame, origin_x, origin_y, cols as i32, rows as i32) {
        return Ok(None);
    }
    let SampleSource::Raw { transform, pixfrac, .. } = &frame.source else { return Ok(None) };
    let transform = *transform;
    let pixfrac = *pixfrac;
    let bayer = ctx.bayer;
    let Hold::Raw(file) = &mut ctx.holds[slot] else { return Ok(None) };
    let height = file.info.height;
    let width = file.info.width;
    let info_channels = file.info.channels;
    let (y0, y1) = input_rows(&transform, height, pixfrac, origin_x, origin_y, cols as i32, rows as i32);
    if y1 <= y0 {
        return Ok(None);
    }
    let nrows = (y1 - y0) as usize;
    let planes = if bayer { 1 } else { channels.max(1) };
    let mut data = vec![0.0f32; planes * nrows * width];
    for plane in 0..planes {
        let src = if bayer { 0 } else { plane.min(info_channels.saturating_sub(1)) };
        let begin = plane * nrows * width;
        file.read_rows_into(src, y0 as usize, y1 as usize, &mut data[begin..begin + nrows * width])?;
    }
    Ok(Some(Loaded { y0, width: width as i32, nrows: nrows as i32, planes, data }))
}

fn queue_frame(ctx: &mut TileCtx<'_>, loaded: &Loaded, slot: usize, origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize) -> Result<(), String> {
    let frame = &ctx.frames[slot];
    let SampleSource::Raw { transform, lut, pixfrac, .. } = &frame.source else { return Ok(()) };
    let transform = *transform;
    let lut = *lut;
    let half = *pixfrac as f64 * 0.5;
    let coeff_a = frame.a;
    let coeff_off = frame.off;
    let coeff_w = frame.imgw;
    let mask_w = frame.mask_w;
    let mask_h = frame.mask_h;
    let frame_w = frame.w;
    let frame_h = frame.h;
    let frame_x0 = frame.x0;
    let frame_y0 = frame.y0;
    let feather = ctx.params.feather;
    let bayer = ctx.bayer;
    let pixels = cols * rows;
    if bayer {
        ctx.gpu.fill_prefix(&mut ctx.work.num, pixels * channels, 0)?;
        ctx.gpu.fill_prefix(&mut ctx.work.den, pixels * channels, 0)?;
        ctx.gpu.copy_htod(&loaded.data, &mut ctx.work.image)?;
        ctx.gpu.drizzle_loaded(ctx.work, loaded.width, loaded.nrows, loaded.y0, lut, -1, transform, half, origin_x, origin_y, cols as i32, rows as i32)?;
        for channel in 0..channels {
            ctx.gpu.finish_plane(
                ctx.work,
                pixels,
                cols as i32,
                channel as i32,
                channel as i32,
                coeff_a[channel],
                coeff_off[channel],
                coeff_w[channel],
                feather,
                ctx.masks[slot].as_ref(),
                mask_w,
                mask_h,
                frame_w,
                frame_h,
                origin_x,
                origin_y,
                frame_x0,
                frame_y0,
                if channel == 0 { slot as i32 } else { -1 },
            )?;
        }
    } else {
        let stride = (loaded.nrows as usize).saturating_mul(loaded.width.max(0) as usize);
        for channel in 0..channels {
            let plane = channel.min(loaded.planes.saturating_sub(1));
            let start = plane * stride;
            ctx.gpu.fill_prefix(&mut ctx.work.num, pixels, 0)?;
            ctx.gpu.fill_prefix(&mut ctx.work.den, pixels, 0)?;
            ctx.gpu.copy_htod(&loaded.data[start..start + stride], &mut ctx.work.image)?;
            ctx.gpu.drizzle_loaded(ctx.work, loaded.width, loaded.nrows, loaded.y0, None, -1, transform, half, origin_x, origin_y, cols as i32, rows as i32)?;
            ctx.gpu.finish_plane(
                ctx.work,
                pixels,
                cols as i32,
                0,
                channel as i32,
                coeff_a[channel],
                coeff_off[channel],
                coeff_w[channel],
                feather,
                ctx.masks[slot].as_ref(),
                mask_w,
                mask_h,
                frame_w,
                frame_h,
                origin_x,
                origin_y,
                frame_x0,
                frame_y0,
                if channel == 0 { slot as i32 } else { -1 },
            )?;
        }
    }
    Ok(())
}

fn pull_shift(holds: &mut [Hold], frames: &[SampleFrame], origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize, feather: f32, workers: usize, values: *mut f32, weights: *mut f32) -> Result<(), String> {
    let mut jobs: Vec<&mut ShiftHold> = holds.iter_mut().filter_map(|hold| match hold { Hold::Shift(job) => Some(job), Hold::Raw(_) => None }).collect();
    if jobs.is_empty() {
        return Ok(());
    }
    let workers = workers.max(1).min(jobs.len());
    let chunk = jobs.len().div_ceil(workers);
    let pixels = rows * cols;
    let k = frames.len();
    let values = values as usize;
    let weights = weights as usize;
    std::thread::scope(|scope| {
        let mut guards = Vec::new();
        for group in jobs.chunks_mut(chunk) {
            guards.push(scope.spawn(move || -> Result<(), String> {
                for job in group.iter_mut() {
                    gather_shift(job, &frames[job.index], origin_x, origin_y, cols, rows, channels, pixels, k, feather, values, weights)?;
                }
                Ok(())
            }));
        }
        for guard in guards {
            guard.join().map_err(|_| "un fil d'empilement s'est arrêté".to_string())??;
        }
        Ok(())
    })
}

fn gather_shift(job: &mut ShiftHold, frame: &SampleFrame, origin_x: i32, origin_y: i32, cols: usize, rows: usize, channels: usize, pixels: usize, k: usize, feather: f32, values: usize, weights: usize) -> Result<(), String> {
    if !overlaps(frame, origin_x, origin_y, cols as i32, rows as i32) {
        return Ok(());
    }
    let height = job.image.info.height;
    let width = job.image.info.width;
    let info_channels = job.image.info.channels;
    let y0 = (origin_y - frame.y0).max(0).min(height as i32);
    let y1 = (origin_y + rows as i32 - frame.y0).clamp(0, height as i32);
    if y1 <= y0 {
        return Ok(());
    }
    let nrows = (y1 - y0) as usize;
    let mut planes: Vec<Option<Vec<f32>>> = vec![None; info_channels.max(1)];
    let mut weight_rows: Option<Vec<f32>> = None;
    if let Some(weight) = job.weight.as_mut() {
        let wy1 = y1.min(weight.info.height as i32);
        if wy1 > y0 {
            weight_rows = Some(weight.read_rows_vec(0, y0 as usize, wy1 as usize)?);
        }
    }
    let weight_width = job.weight.as_ref().map(|weight| weight.info.width).unwrap_or(0);
    for channel in 0..channels {
        let plane = channel.min(info_channels.saturating_sub(1));
        if planes[plane].is_none() {
            planes[plane] = Some(job.image.read_rows_vec(plane, y0 as usize, y1 as usize)?);
        }
        let image = planes[plane].as_ref().unwrap();
        for row in 0..nrows {
            let canvas_y = frame.y0 + y0 + row as i32;
            let local_row = (canvas_y - origin_y) as usize;
            if local_row >= rows {
                continue;
            }
            for col in 0..cols {
                let local_x = origin_x + col as i32 - frame.x0;
                if local_x < 0 || local_x >= width as i32 {
                    continue;
                }
                let v = image[row * width + local_x as usize];
                let d = match weight_rows.as_ref() {
                    Some(map) if (local_x as usize) < weight_width && row < map.len() / weight_width.max(1) => map[row * weight_width + local_x as usize],
                    Some(_) => 0.0,
                    None => 1.0,
                };
                let nv = (v as f64 * frame.a[channel] - frame.off[channel]) as f32;
                if v == 0.0 || d == 0.0 || !v.is_finite() || nv == 0.0 {
                    continue;
                }
                let fw = feather_of(frame, local_x as f32, (y0 as usize + row) as f32, feather);
                let index = channel * pixels * k + (local_row * cols + col) * k + job.index;
                // Chaque fil possède des poses distinctes : les cases (pixel, pose) ne se recouvrent pas.
                unsafe {
                    *(values as *mut f32).add(index) = nv;
                    *(weights as *mut f32).add(index) = d * fw * frame.imgw[channel] as f32;
                }
            }
        }
    }
    Ok(())
}

/// Le premier canal est déjà dans le cube. Les suivants y sont copiés, pose par pose, sans dispersion.
fn reject_cube(ctx: &mut TileCtx<'_>, pixels: usize, channels: usize) -> Result<TileOut, String> {
    let k = ctx.frames.len();
    let mut values = vec![0.0f32; channels * pixels];
    let mut nlo = 0i32;
    let mut nhi = 0i32;
    let mut nvalid = 0i32;
    for channel in 0..channels {
        if channel > 0 {
            let start = (channel - 1) * pixels * k;
            let count = pixels * k;
            ctx.gpu.copy_htod(&ctx.host_v[start..start + count], &mut ctx.work.dst_v)?;
            ctx.gpu.copy_htod(&ctx.host_w[start..start + count], &mut ctx.work.dst_w)?;
        }
        let rejected = ctx.gpu.reject_loaded(ctx.work, k as i32, pixels as i32, ctx.params.rejection, ctx.params.sigma_low, ctx.params.sigma_high, ctx.params.plain, true)?;
        values[channel * pixels..(channel + 1) * pixels].copy_from_slice(&rejected.values);
        nlo = nlo.saturating_add(rejected.nlo);
        nhi = nhi.saturating_add(rejected.nhi);
        nvalid = nvalid.saturating_add(rejected.nvalid);
    }
    Ok(TileOut { values, kept: ctx.kept[..pixels].to_vec(), nlo, nhi, nvalid })
}

fn reject_host(ctx: &mut TileCtx<'_>, pixels: usize, channels: usize) -> Result<TileOut, String> {
    let k = ctx.frames.len();
    let mut values = vec![0.0f32; channels * pixels];
    let mut nlo = 0i32;
    let mut nhi = 0i32;
    let mut nvalid = 0i32;
    for channel in 0..channels {
        let start = channel * pixels * k;
        let count = pixels * k;
        ctx.gpu.copy_htod(&ctx.host_v[start..start + count], &mut ctx.work.dst_v)?;
        ctx.gpu.copy_htod(&ctx.host_w[start..start + count], &mut ctx.work.dst_w)?;
        let rejected = ctx.gpu.reject_loaded(ctx.work, k as i32, pixels as i32, ctx.params.rejection, ctx.params.sigma_low, ctx.params.sigma_high, ctx.params.plain, false)?;
        values[channel * pixels..(channel + 1) * pixels].copy_from_slice(&rejected.values);
        nlo = nlo.saturating_add(rejected.nlo);
        nhi = nhi.saturating_add(rejected.nhi);
        nvalid = nvalid.saturating_add(rejected.nvalid);
    }
    Ok(TileOut { values, kept: ctx.kept[..pixels].to_vec(), nlo, nhi, nvalid })
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

#[cfg(test)]
mod tests {
    use super::{geometry, host_bytes, memory_budget, peak_bytes, SAMPLE_BYTES};

    #[test]
    fn full_width_strip_grows_until_the_budget() {
        let frames = 141;
        let width = 6000;
        let height = 200;
        let channels = 3;
        let accum = 3;
        let peak = peak_bytes(frames, height, width, channels, accum);
        let host = host_bytes(frames, height, width, channels);
        let (strip, tile) = geometry(frames, width, height, channels, accum, peak + 8 * 1024 * 1024, host, 0, 0, 0, channels);
        assert_eq!(tile, width);
        assert_eq!(strip, height);
    }

    #[test]
    fn a_tight_budget_splits_the_width() {
        let (strip, tile) = geometry(200, 4096, 2000, 3, 3, 64 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 0, 0, 0, 3);
        assert!(tile < 4096, "{tile}");
        assert!(strip >= 16, "{strip}");
        assert!(peak_bytes(200, strip, tile, 3, 3) <= 64 * 1024 * 1024);
    }

    #[test]
    fn caps_limit_the_band() {
        let (strip, tile) = geometry(8, 3000, 2000, 1, 0, 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 0, 32, 512, 1);
        assert!(strip <= 32, "{strip}");
        assert!(tile <= 512, "{tile}");
    }

    #[test]
    fn scratch_stays_a_strip_of_rows() {
        let frames = 141;
        let rows = 800;
        let cols = 2160;
        let cube = rows * cols * frames * SAMPLE_BYTES;
        let peak = peak_bytes(frames, rows, cols, 3, 3);
        assert!(peak < cube * 2, "pic {peak} cube {cube}");
    }

    #[test]
    fn the_raw_cube_is_not_limited_by_a_host_copy_of_the_first_channel() {
        let gpu = 10 * 1024 * 1024 * 1024;
        let (strip, tile) = geometry(141, 2160, 3840, 3, 3, gpu, 1024, 0, 0, 0, 0);
        let (limited, _) = geometry(141, 2160, 3840, 3, 3, gpu, 1024, 0, 0, 0, 3);
        assert_eq!(tile, 2160);
        assert!(strip > limited, "{strip} contre {limited}");
    }

    #[test]
    fn manual_memory_is_a_ceiling_on_free_memory() {
        let gib = 1024usize * 1024 * 1024;
        assert_eq!(memory_budget(0.0, 10 * gib, 12 * gib), 9 * gib);
        assert_eq!(memory_budget(4.0, 10 * gib, 12 * gib), 4 * gib);
        assert_eq!(memory_budget(20.0, 10 * gib, 12 * gib), 9 * gib);
        assert_eq!(memory_budget(0.0, gib / 2, gib), gib / 4);
    }
}
