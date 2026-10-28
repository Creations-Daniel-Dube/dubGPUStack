// SPDX-License-Identifier: GPL-3.0-or-later
//! Masques de fondu : fermeture 7×7, réduction bilinéaire 11 bits, chamfer Q16.

use std::path::Path;

use super::fits;

pub const MASK_SCALE: f64 = 0.1;
const SHIFT: i64 = 16;
const HV: i64 = 62587;
const DIAG: i64 = 89738;
const INIT: i64 = (1i64 << 31) - 1;
const DIST_MAX: i64 = INIT >> 2;

pub fn mask_size(width: usize, height: usize) -> (usize, usize) {
    ((width as f64 * MASK_SCALE) as usize, (height as f64 * MASK_SCALE) as usize)
}

pub fn chamfer_distance(foreground: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut tmp = vec![INIT; (height + 2) * (width + 2)];
    let stride = width + 2;
    let left: Vec<i64> = (0..width).map(|j| INIT + HV + j as i64 * HV).collect();
    let right: Vec<i64> = (0..width).map(|j| INIT + width as i64 * HV - j as i64 * HV).collect();
    for i in 0..height {
        let mut c = vec![0i64; width];
        for j in 0..width {
            let up = i * stride;
            let a = tmp[up + j] + DIAG;
            let b = tmp[up + j + 1] + HV;
            let d = tmp[up + j + 2] + DIAG;
            c[j] = a.min(b).min(d);
            if foreground[i * width + j] == 0 {
                c[j] = 0;
            }
        }
        let mut acc = i64::MAX;
        let mut row = vec![0i64; width];
        for j in 0..width {
            acc = acc.min(c[j] - j as i64 * HV);
            row[j] = (acc + j as i64 * HV).min(left[j]);
            if foreground[i * width + j] == 0 {
                row[j] = 0;
            }
        }
        for j in 0..width {
            tmp[(i + 1) * stride + 1 + j] = row[j];
        }
    }
    let mut out = vec![0.0f32; height * width];
    for i in (0..height).rev() {
        let mut e = vec![0i64; width];
        for j in 0..width {
            let down = (i + 2) * stride;
            let current = tmp[(i + 1) * stride + 1 + j];
            let a = tmp[down + j + 2] + DIAG;
            let b = tmp[down + 1 + j] + HV;
            let d = tmp[down + j] + DIAG;
            e[j] = current.min(a.min(b).min(d));
        }
        let mut acc = i64::MAX;
        let mut row = vec![0i64; width];
        for k in 0..width {
            let j = width - 1 - k;
            acc = acc.min(e[j] + j as i64 * HV);
            row[j] = (acc - j as i64 * HV).min(right[j]);
        }
        for j in 0..width {
            tmp[(i + 1) * stride + 1 + j] = row[j];
            out[i * width + j] = row[j].min(DIST_MAX) as f32 / 65536.0;
        }
    }
    let _ = SHIFT;
    out
}

fn linear_coeffs(src: i32, dst: i32) -> (Vec<i32>, Vec<i16>) {
    let scale = 1.0 / (dst as f64 / src as f64);
    let mut offsets = Vec::with_capacity(dst as usize);
    let mut coef = Vec::with_capacity(dst as usize * 2);
    for i in 0..dst {
        let mut f = ((i as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = f.floor() as i64;
        f -= s as f32;
        if s < 0 {
            f = 0.0;
            s = 0;
        }
        if s >= src as i64 - 1 {
            f = 0.0;
            s = src as i64 - 1;
        }
        offsets.push(s as i32);
        coef.push(((1.0 - f) * 2048.0).round() as i16);
        coef.push((f * 2048.0).round() as i16);
    }
    (offsets, coef)
}

fn resize_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return vec![0; dw * dh];
    }
    let (xofs, ax) = linear_coeffs(sw as i32, dw as i32);
    let (yofs, by) = linear_coeffs(sh as i32, dh as i32);
    let mut dst = vec![0u8; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let sx = xofs[x] as usize;
            let sy = yofs[y] as usize;
            let sx1 = (sx + 1).min(sw - 1);
            let sy1 = (sy + 1).min(sh - 1);
            let a0 = ax[2 * x] as i32;
            let a1 = ax[2 * x + 1] as i32;
            let b0 = by[2 * y] as i32;
            let b1 = by[2 * y + 1] as i32;
            let h0 = src[sy * sw + sx] as i32 * a0 + src[sy * sw + sx1] as i32 * a1;
            let h1 = src[sy1 * sw + sx] as i32 * a0 + src[sy1 * sw + sx1] as i32 * a1;
            let v = h0 as i64 * b0 as i64 + h1 as i64 * b1 as i64 + (1 << 21);
            let o = (v >> 22) as i32;
            dst[y * dw + x] = o.clamp(0, 255) as u8;
        }
    }
    dst
}

fn clamp_index(v: i32, n: i32) -> usize {
    v.clamp(0, n - 1) as usize
}

fn maximum_filter7(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut dst = vec![0u8; width * height];
    let w = width as i32;
    let h = height as i32;
    for y in 0..height {
        for x in 0..width {
            let mut m = 0u8;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let sx = clamp_index(x as i32 + dx, w);
                    let sy = clamp_index(y as i32 + dy, h);
                    m = m.max(src[sy * width + sx]);
                }
            }
            dst[y * width + x] = m;
        }
    }
    dst
}

fn minimum_filter7(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut dst = vec![255u8; width * height];
    let w = width as i32;
    let h = height as i32;
    for y in 0..height {
        for x in 0..width {
            let mut m = 255u8;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let sx = clamp_index(x as i32 + dx, w);
                    let sy = clamp_index(y as i32 + dy, h);
                    m = m.min(src[sy * width + sx]);
                }
            }
            dst[y * width + x] = m;
        }
    }
    dst
}

/// Masque de distance top-down, à partir d'un plan top-down.
pub fn make_mask(plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>, String> {
    let (mw, mh) = mask_size(width, height);
    if mw < 1 || mh < 1 {
        return Err(format!("image trop petite pour un masque ({width}×{height})"));
    }
    let mut binary = vec![0u8; width * height];
    for (out, value) in binary.iter_mut().zip(plane.iter()) {
        if *value != 0.0 {
            *out = 255;
        }
    }
    let closed = minimum_filter7(&maximum_filter7(&binary, width, height), width, height);
    let down = resize_u8(&closed, width, height, mw, mh);
    let mut padded = vec![0u8; (mh + 2) * (mw + 2)];
    for y in 0..mh {
        for x in 0..mw {
            padded[(y + 1) * (mw + 2) + x + 1] = down[y * mw + x];
        }
    }
    let dist = chamfer_distance(&padded, mw + 2, mh + 2);
    let mut mask = vec![0.0f32; mw * mh];
    for y in 0..mh {
        for x in 0..mw {
            mask[y * mw + x] = dist[(y + 1) * (mw + 2) + x + 1];
        }
    }
    Ok(mask)
}

pub fn read_mask(path: &Path) -> Result<Vec<f32>, String> {
    let (data, info) = fits::read_image(path)?;
    if info.width == 0 || info.height == 0 {
        return Err("masque vide".to_string());
    }
    Ok(data)
}

pub fn write_mask(path: &Path, mask: &[f32], width: usize, height: usize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fits::write_fits(path, mask, 1, height, width, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_pixel_has_zero_distance_and_neighbour_is_the_chamfer_step() {
        let mut fg = vec![255u8; 9];
        fg[4] = 0;
        let dist = chamfer_distance(&fg, 3, 3);
        assert_eq!(dist[4], 0.0);
        let step = dist[5];
        assert!((step - 62587.0 / 65536.0).abs() < 1e-4, "{step}");
    }
}
