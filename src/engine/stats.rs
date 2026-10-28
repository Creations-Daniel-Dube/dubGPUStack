// SPDX-License-Identifier: GPL-3.0-or-later
//! Statistiques Siril : médiane d'histogramme, IKSSlite, bruit de fond.

use std::path::Path;

use serde::{Deserialize, Serialize};

pub const CACHE_VERSION: i32 = 2;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameStats {
    #[serde(default)]
    pub loc: Vec<Option<f64>>,
    #[serde(default)]
    pub scale: Vec<Option<f64>>,
    #[serde(default)]
    pub noise: Vec<Option<f64>>,
    #[serde(default)]
    pub ngood: Vec<Option<i64>>,
    #[serde(default)]
    pub bad: String,
}

impl FrameStats {
    pub fn has_norm(&self, layers: usize) -> bool {
        self.scale.len() >= layers
            && (0..layers).all(|c| self.scale[c].is_some_and(|v| v > 0.0) && self.loc.get(c).and_then(|v| *v).is_some())
    }

    pub fn has_noise(&self, layers: usize) -> bool {
        self.noise.len() >= layers && (0..layers).all(|c| self.noise[c].is_some_and(|v| v > 0.0))
    }
}

pub fn file_signature(path: &Path) -> [i64; 2] {
    super::lights::file_signature(path)
}

pub fn histogram_median(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for value in values {
        lo = lo.min(*value);
        hi = hi.max(*value);
    }
    if lo == hi || !lo.is_finite() || !hi.is_finite() {
        return lo;
    }
    let nbins_m1 = (n.min(65536) - 1) as f32;
    let bins = nbins_m1 as usize + 1;
    let scale = nbins_m1 / (hi - lo);
    let mut hist = vec![0u32; bins];
    for value in values {
        let f = scale * (*value - lo);
        let bin = if (-2147483648.0..2147483648.0).contains(&f) { f as i32 } else { i32::MIN };
        let index = bin as u16 as usize;
        if index < hist.len() {
            hist[index] += 1;
        }
    }
    let thresh = 0.5 * n as f32;
    let mut count = 0u64;
    let mut k = 0usize;
    while (count as f32) < thresh && k < hist.len() {
        count += hist[k] as u64;
        k += 1;
    }
    if k == 0 {
        return lo;
    }
    let bin_count = hist[k - 1] as f32;
    let c0 = count as f32 - thresh;
    let c1 = thresh - (count as f32 - bin_count);
    let denom = c0 + c1;
    let mut m = if denom == 0.0 { k as f32 } else { (c1 * k as f32 + c0 * (k as f32 - 1.0)) / denom };
    m = m / scale + lo;
    m.clamp(lo, hi)
}

pub fn mad(values: &[f32], median: f32) -> f32 {
    let dev: Vec<f32> = values.iter().map(|value| (value - median).abs()).collect();
    histogram_median(&dev)
}

pub fn bwmv(values: &[f32], mad: f32, median: f32) -> f32 {
    if !(mad > 0.0) || values.is_empty() {
        return 0.0;
    }
    let factor = 1.0 / (9.0 * mad);
    let mut up = 0.0f64;
    let mut down = 0.0f64;
    for value in values {
        let i_med = *value - median;
        let yi = i_med * factor;
        let yi2 = if yi.abs() < 1.0 { yi * yi } else { 1.0 };
        let w = 1.0 - yi2;
        let a = i_med * (w * w);
        up += (a * a) as f64;
        down += (w * (1.0 - 5.0 * yi2)) as f64;
    }
    if down == 0.0 {
        return 0.0;
    }
    let n = values.len() as f64;
    (n * (up / (down * down))) as f32
}

pub fn ikss_lite(values: &[f32], median: f32, mad_value: f32) -> Option<(f32, f32)> {
    if !(mad_value > 0.0) {
        return None;
    }
    let lo = (median as f64 - 6.0 * mad_value as f64) as f32;
    let hi = (median as f64 + 6.0 * mad_value as f64) as f32;
    let kept: Vec<f32> = values.iter().copied().filter(|value| *value >= lo && *value <= hi).collect();
    if kept.is_empty() {
        return None;
    }
    let loc = histogram_median(&kept);
    let scale_mad = mad(&kept, loc);
    if scale_mad == 0.0 {
        return None;
    }
    Some((loc, bwmv(&kept, scale_mad, loc).sqrt() * 0.991))
}

pub fn norm_stats(values: &[f32]) -> Option<(f32, f32)> {
    if values.len() < 2 {
        return None;
    }
    let median = histogram_median(values);
    let mad_value = mad(values, median);
    ikss_lite(values, median, mad_value)
}

fn row_noise(row: &[f32]) -> Option<f64> {
    if row.len() < 3 {
        return None;
    }
    let moments = |ncrit: usize, cm: &[f32], cs: &[f64]| -> (i64, i64, f64, f64) {
        let mut nv = 0i64;
        let mut ng = 0i64;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut ii = 0;
        while ii < row.len() && (row[ii] == 0.0 || row[ii].is_nan()) {
            ii += 1;
        }
        if ii == row.len() {
            return (0, 0, 0.0, 0.0);
        }
        let mut v1 = row[ii];
        ii += 1;
        while ii < row.len() {
            let v = row[ii];
            ii += 1;
            if v == 0.0 || v.is_nan() {
                continue;
            }
            let d = v1 - v;
            v1 = v;
            let mut keep = true;
            for t in 0..ncrit {
                if !(((d - cm[t]).abs() as f64) < 5.0 * cs[t]) {
                    keep = false;
                    break;
                }
            }
            if !keep {
                continue;
            }
            nv += 1;
            if d != 0.0 {
                ng += 1;
                let x = d as f64;
                sum += x;
                sum2 += x * x;
            }
        }
        (nv, ng, sum, sum2)
    };
    let mean_sigma = |ng: i64, sum: f64, sum2: f64| -> (f64, f64) {
        if ng > 1 {
            let n = ng as f64;
            let mean = sum / n;
            let var = sum2 / n - mean * mean;
            (mean, if var > 0.0 { var.sqrt() } else { 0.0 })
        } else if ng == 1 {
            (sum, 0.0)
        } else {
            (0.0, 0.0)
        }
    };
    let (mut nv, ng, sum, sum2) = moments(0, &[], &[]);
    if nv < 2 {
        return None;
    }
    let (mut mean, mut sd) = mean_sigma(ng, sum, sum2);
    let mut cm = [0.0f32; 4];
    let mut cs = [0.0f64; 4];
    if sd > 0.0 {
        for iter in 0..3 {
            cm[iter] = mean as f32;
            cs[iter] = sd;
            let (kk, ng2, sum, sum2) = moments(iter + 1, &cm, &cs);
            if kk == nv {
                break;
            }
            nv = kk;
            (mean, sd) = mean_sigma(ng2, sum, sum2);
        }
    }
    Some(sd)
}

pub fn bgnoise(plane: &[f32], width: usize, height: usize) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let mut sigmas = Vec::new();
    for y in 0..height {
        if let Some(sd) = row_noise(&plane[y * width..(y + 1) * width]) {
            sigmas.push(sd);
        }
    }
    if sigmas.is_empty() {
        return 0.0;
    }
    sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sigmas.len();
    let mid = if n == 1 { sigmas[0] } else { (sigmas[(n - 1) / 2] + sigmas[n / 2]) / 2.0 };
    (0.70710678 * mid) as f32
}

/// `plane` est top-down. Les statistiques de normalisation parcourent les pixels valides en ordre FITS.
pub fn measure_plane(plane: &[f32], width: usize, height: usize, need_norm: bool, need_noise: bool) -> (Option<f64>, Option<f64>, Option<f64>, i64) {
    let mut valid = Vec::new();
    for y in (0..height).rev() {
        for x in 0..width {
            let value = plane[y * width + x];
            if value != 0.0 && value.is_finite() {
                valid.push(value);
            }
        }
    }
    let ngood = valid.len() as i64;
    let (loc, scale) = if need_norm && ngood > 1 { norm_stats(&valid).map(|(l, s)| (Some(l as f64), Some(s as f64))).unwrap_or((None, None)) } else { (None, None) };
    let noise = if need_noise && ngood > 2 { let n = bgnoise(plane, width, height); if n > 0.0 { Some(n as f64) } else { None } } else { None };
    (loc, scale, noise, ngood)
}

pub fn measure_image(data: &[f32], channels: usize, height: usize, width: usize, need_norm: bool, need_noise: bool) -> FrameStats {
    let mut stats = FrameStats::default();
    let plane = height * width;
    for channel in 0..channels {
        let (loc, scale, noise, ngood) = measure_plane(&data[channel * plane..(channel + 1) * plane], width, height, need_norm, need_noise);
        stats.loc.push(loc);
        stats.scale.push(scale);
        stats.noise.push(noise);
        stats.ngood.push(Some(ngood));
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_a_ramp_is_the_middle() {
        let values: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let median = histogram_median(&values);
        assert!((median - 4.5).abs() < 0.05, "{median}");
    }

    #[test]
    fn flat_row_has_no_noise() {
        assert!(row_noise(&[0.2, 0.2, 0.2, 0.2]).unwrap_or(1.0) == 0.0);
    }
}
