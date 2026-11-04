// SPDX-License-Identifier: GPL-3.0-or-later
//! Étirement d'affichage (STF) et couleurs de la carte de couverture.

const MAD_TO_SIGMA: f64 = 1.4826;

pub fn mtf(m: f32, x: f32) -> f32 {
    ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)
}

pub fn auto_stf(data: &[f32], channels: usize, height: usize, width: usize, linked: bool) -> Vec<(f32, f32, f32)> {
    let pixels = height.saturating_mul(width);
    if channels == 0 || pixels == 0 || data.len() < channels * pixels {
        return vec![(0.0, 0.5, 1.0); channels.max(1)];
    }
    let mut index = Vec::new();
    for pixel in 0..pixels {
        if (0..channels).any(|c| data[c * pixels + pixel] != 0.0) {
            index.push(pixel);
        }
    }
    if index.is_empty() {
        return vec![(0.0, 0.5, 1.0); channels];
    }
    if index.len() > 250_000 {
        let last = index.len() - 1;
        let step = index.len() as f64 / 250_000.0;
        index = (0..250_000).map(|i| index[((i as f64) * step).min(last as f64) as usize]).collect();
    }
    let mut samples = vec![Vec::with_capacity(index.len()); channels];
    for pixel in &index {
        for c in 0..channels {
            samples[c].push(data[c * pixels + *pixel]);
        }
    }
    let mut top = 0.0f32;
    for values in &samples {
        top = top.max(percentile(values, 99.995));
    }
    if top <= 0.0 {
        top = 1.0;
    }
    let mut stats: Vec<(f64, f64)> = samples.iter().map(|values| robust(values, top)).collect();
    if linked && !stats.is_empty() {
        let med = stats.iter().map(|s| s.0).sum::<f64>() / stats.len() as f64;
        let mad = stats.iter().map(|s| s.1).sum::<f64>() / stats.len() as f64;
        stats = vec![(med, mad); channels];
    }
    stats.into_iter().map(|(med, mad)| {
        let c0 = (med + -2.8 * mad).clamp(0.0, 1.0) as f32;
        let x = if c0 < 1.0 { ((med as f32 - c0) / (1.0 - c0)).clamp(0.0, 1.0) } else { 0.5 };
        let m = if x > 0.0 && x < 1.0 { mtf(0.25, x) } else { 0.5 };
        (c0 * top, m, top)
    }).collect()
}

pub fn apply_stf(data: &[f32], channels: usize, height: usize, width: usize, params: &[(f32, f32, f32)]) -> Vec<u8> {
    let pixels = height * width;
    let mut rgba = vec![0u8; pixels * 4];
    let map = if channels <= 1 { [0, 0, 0] } else { [0, 1, 2] };
    for pixel in 0..pixels {
        for (out_c, src_c) in map.iter().enumerate() {
            let (lo, mid, hi) = params.get(*src_c).copied().unwrap_or((0.0, 0.5, 1.0));
            let v = data.get(src_c * pixels + pixel).copied().unwrap_or(0.0);
            let x = if hi > lo { ((v - lo) / (hi - lo)).clamp(0.0, 1.0) } else { 0.0 };
            let y = mtf(mid, x);
            rgba[pixel * 4 + out_c] = (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        rgba[pixel * 4 + 3] = 255;
    }
    rgba
}

/// Image déjà étirée dans `[0, 1]`, quantifiée comme l’aperçu VeraLux.
pub fn unit_rgba(data: &[f32], channels: usize, height: usize, width: usize) -> Vec<u8> {
    let pixels = height * width;
    let mut rgba = vec![0u8; pixels * 4];
    let map = if channels <= 1 { [0, 0, 0] } else { [0, 1, 2] };
    for pixel in 0..pixels {
        for (out_c, src_c) in map.iter().enumerate() {
            let v = data.get(src_c * pixels + pixel).copied().unwrap_or(0.0);
            let y = (v.clamp(0.0, 1.0) * 255.0).min(255.0);
            rgba[pixel * 4 + out_c] = y as u8;
        }
        rgba[pixel * 4 + 3] = 255;
    }
    rgba
}

pub fn coverage_rgba(depth: &[i32], width: usize, height: usize) -> Vec<u8> {
    let vmax = depth.iter().copied().max().unwrap_or(1).max(1) as f32;
    let mut rgba = vec![0u8; width * height * 4];
    for (pixel, value) in depth.iter().copied().enumerate().take(width * height) {
        if value <= 0 {
            continue;
        }
        let t = (value as f32 / vmax).clamp(0.0, 1.0).sqrt();
        let (r, g, b) = coverage_color(t);
        let out = pixel * 4;
        rgba[out] = r;
        rgba[out + 1] = g;
        rgba[out + 2] = b;
        rgba[out + 3] = 255;
    }
    rgba
}

fn coverage_color(t: f32) -> (u8, u8, u8) {
    const STOPS: [(f32, [u8; 3]); 5] = [
        (0.0, [14, 18, 32]),
        (0.22, [38, 64, 140]),
        (0.45, [40, 150, 168]),
        (0.70, [226, 196, 84]),
        (1.0, [255, 248, 230]),
    ];
    let mut i = 0;
    while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
        i += 1;
    }
    let (p0, c0) = STOPS[i];
    let (p1, c1) = STOPS[(i + 1).min(STOPS.len() - 1)];
    let u = if p1 > p0 { ((t - p0) / (p1 - p0)).clamp(0.0, 1.0) } else { 0.0 };
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * u).round() as u8;
    (mix(c0[0], c1[0]), mix(c0[1], c1[1]), mix(c0[2], c1[2]))
}

fn robust(values: &[f32], top: f32) -> (f64, f64) {
    let scaled: Vec<f32> = values.iter().map(|v| v / top).collect();
    let med = percentile(&scaled, 50.0) as f64;
    let mut dev: Vec<f32> = scaled.iter().map(|v| (v - med as f32).abs()).collect();
    let mad = percentile(&mut dev, 50.0) as f64 * MAD_TO_SIGMA;
    (med, mad)
}

/// Résultat du stretch HyperMetric Ready-to-Use (boutons Auto-Calc puis PROCESS).
pub struct HypermetricResult {
    pub image: Vec<f32>,
    pub log_d: f64,
}

const SEESTAR_S30: [f64; 3] = [0.2928, 0.5053, 0.2019];
const TARGET_BG: f64 = 0.20;
const PROTECT_B: f64 = 6.0;
const CONVERGENCE: f64 = 3.5;
const RTU_PEDESTAL: f64 = 0.001;

/// Stretch photométrique calqué sur VeraLux HyperMetric Stretch 1.5.2
/// (Riccardo Paterniti, GPL-3.0-or-later) : Auto-Calc Log D, puis PROCESS
/// en mode Ready-to-Use, ancre adaptative, fond 0,20, Protect b 6,
/// vecteur pur, convergence 3,5, poids ZWO Seestar S30.
///
/// `data` est planaire, `c * h * w + y * w + x`, déjà dans `[0, 1]`.
pub fn auto_stretch_hypermetric(data: &[f32], channels: usize, height: usize, width: usize) -> HypermetricResult {
    let pixels = height.saturating_mul(width);
    if channels == 0 || pixels == 0 || data.len() < channels * pixels {
        return HypermetricResult { image: data.to_vec(), log_d: 2.0 };
    }
    let log_d = solve_log_d_ready(data, channels, height, width);
    let image = process_ready(data, channels, height, width, log_d);
    HypermetricResult { image, log_d }
}

fn finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn plane(data: &[f32], channel: usize, pixels: usize) -> &[f32] {
    let start = channel * pixels;
    &data[start..start + pixels]
}

fn hyperbolic(data: f64, d: f64, b: f64) -> f64 {
    let d = d.max(0.1);
    let b = b.max(0.1);
    let term1 = (d * data + b).asinh();
    let term2 = b.asinh();
    let mut norm = (d + b).asinh() - term2;
    if norm.abs() < 1e-12 {
        norm = 1e-6;
    }
    (term1 - term2) / norm
}

fn percentile_sorted(sorted: &[f32], percent: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0] as f64;
    }
    let rank = (percent / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (rank.ceil() as usize).min(n - 1);
    let t = rank - lo as f64;
    let a = sorted[lo.min(n - 1)] as f64;
    let b = sorted[hi] as f64;
    a * (1.0 - t) + b * t
}

fn percentile_linear(values: &[f32], percent: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&copy, percent)
}

fn median(values: &[f32]) -> f64 {
    percentile_linear(values, 50.0)
}

fn std_pop(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = values.iter().map(|v| {
        let d = *v as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    var.sqrt()
}

fn smooth_hist(hist: &[u64]) -> Vec<f64> {
    let n = hist.len();
    let kernel = 50usize;
    let start = (kernel - 1) / 2;
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        let center = i + start;
        for k in 0..kernel {
            if center >= k {
                let j = center - k;
                if j < n {
                    sum += hist[j] as f64;
                }
            }
        }
        out[i] = sum / kernel as f64;
    }
    out
}

fn argmax_f64(values: &[f64]) -> usize {
    let mut best_i = 0;
    let mut best = f64::NEG_INFINITY;
    for (i, value) in values.iter().enumerate() {
        if *value > best {
            best = *value;
            best_i = i;
        }
    }
    best_i
}

fn adaptive_anchor_from_sample(sample: &[f32]) -> f64 {
    if sample.is_empty() {
        return 0.0;
    }
    let mut hist = vec![0u64; 65536];
    for value in sample {
        let v = *value as f64;
        if !(0.0..=1.0).contains(&v) {
            continue;
        }
        let mut idx = (v * 65536.0).floor() as usize;
        if idx >= 65536 {
            idx = 65535;
        }
        hist[idx] += 1;
    }
    let hist_smooth = smooth_hist(&hist);
    let mut search_start = 100usize;
    let head_max = hist_smooth[..search_start.min(hist_smooth.len())].iter().copied().fold(0.0, f64::max);
    if head_max > 0.0 {
        search_start = 0;
    }
    if search_start >= hist_smooth.len() {
        search_start = 0;
    }
    let peak_idx = argmax_f64(&hist_smooth[search_start..]) + search_start;
    let target_val = hist_smooth[peak_idx] * 0.06;
    let mut anchor_idx = None;
    for (i, value) in hist_smooth[..peak_idx].iter().enumerate() {
        if *value < target_val {
            anchor_idx = Some(i);
        }
    }
    let anchor = if let Some(i) = anchor_idx {
        i as f64 / 65536.0
    } else {
        percentile_linear(sample, 0.5)
    };
    anchor.max(0.0)
}

fn luma_at(data: &[f32], pixels: usize, index: usize) -> f32 {
    let r = finite(data[index]) as f64;
    let g = finite(data[pixels + index]) as f64;
    let b = finite(data[2 * pixels + index]) as f64;
    (SEESTAR_S30[0] * r + SEESTAR_S30[1] * g + SEESTAR_S30[2] * b) as f32
}

fn adaptive_anchor(data: &[f32], channels: usize, height: usize, width: usize) -> f64 {
    let pixels = height * width;
    if pixels == 0 {
        return 0.0;
    }
    let stride = (pixels / 2_000_000).max(1);
    let mut sample = Vec::with_capacity(pixels / stride + 1);
    if channels >= 3 {
        for index in (0..pixels).step_by(stride) {
            sample.push(luma_at(data, pixels, index));
        }
    } else {
        for index in (0..pixels).step_by(stride) {
            sample.push(finite(data[index]));
        }
    }
    adaptive_anchor_from_sample(&sample)
}

fn star_pressure(luminance: &[f32]) -> f64 {
    if luminance.is_empty() {
        return 0.0;
    }
    let stride = (luminance.len() / 300_000).max(1);
    let mut sample: Vec<f32> = luminance.iter().step_by(stride).copied().filter(|v| *v > 1e-7).collect();
    if sample.len() < 100 {
        return 0.0;
    }
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p999 = percentile_sorted(&sample, 99.9);
    let p9999 = percentile_sorted(&sample, 99.99);
    let bright = sample.iter().filter(|v| **v as f64 > p999).count() as f64 / sample.len() as f64;
    let p_term = (p9999 / (p999 + 1e-9)).clamp(1.0, 5.0);
    let p_term = (p_term - 1.0) / 4.0;
    let f_term = (bright * 200.0).clamp(0.0, 1.0);
    (0.7 * p_term + 0.3 * f_term).clamp(0.0, 1.0)
}

fn solve_log_d(sample: &[f32], target: f64, b: f64) -> f64 {
    let median_in = median(sample);
    if median_in < 1e-9 {
        return 2.0;
    }
    let mut low = 0.0;
    let mut high = 7.0;
    let mut best = 2.0;
    for _ in 0..40 {
        let mid = (low + high) / 2.0;
        let test = hyperbolic(median_in, 10.0_f64.powf(mid), b);
        if (test - target).abs() < 0.0001 {
            best = mid;
            break;
        }
        if test < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    best
}

fn solve_log_d_ready(data: &[f32], channels: usize, height: usize, width: usize) -> f64 {
    let pixels = height * width;
    let step = (pixels / 100_000).max(1);
    let rgb = channels >= 3;
    let n = plane(data, 0, pixels).iter().step_by(step).count();
    if n == 0 {
        return 2.0;
    }
    let mut anchored = Vec::with_capacity(n);
    if rgb {
        let r: Vec<f32> = plane(data, 0, pixels).iter().step_by(step).copied().map(finite).collect();
        let g: Vec<f32> = plane(data, 1, pixels).iter().step_by(step).copied().map(finite).collect();
        let b: Vec<f32> = plane(data, 2, pixels).iter().step_by(step).copied().map(finite).collect();
        let luma: Vec<f32> = (0..n).map(|i| {
            (SEESTAR_S30[0] * r[i] as f64 + SEESTAR_S30[1] * g[i] as f64 + SEESTAR_S30[2] * b[i] as f64) as f32
        }).collect();
        let stride = (luma.len() / 2_000_000).max(1);
        let anchor = if stride == 1 {
            adaptive_anchor_from_sample(&luma)
        } else {
            let sample: Vec<f32> = luma.iter().step_by(stride).copied().collect();
            adaptive_anchor_from_sample(&sample)
        };
        for i in 0..n {
            let rv = (r[i] as f64 - anchor).max(0.0);
            let gv = (g[i] as f64 - anchor).max(0.0);
            let bv = (b[i] as f64 - anchor).max(0.0);
            anchored.push((SEESTAR_S30[0] * rv + SEESTAR_S30[1] * gv + SEESTAR_S30[2] * bv) as f32);
        }
    } else {
        let mono: Vec<f32> = plane(data, 0, pixels).iter().step_by(step).copied().map(finite).collect();
        let stride = (mono.len() / 2_000_000).max(1);
        let anchor = if stride == 1 {
            adaptive_anchor_from_sample(&mono)
        } else {
            let sample: Vec<f32> = mono.iter().step_by(stride).copied().collect();
            adaptive_anchor_from_sample(&sample)
        };
        for value in &mono {
            anchored.push((*value as f64 - anchor).max(0.0) as f32);
        }
    }
    let pressure = star_pressure(&anchored);
    let valid: Vec<f32> = anchored.into_iter().filter(|v| *v > 1e-7).collect();
    if valid.is_empty() {
        return 2.0;
    }
    let mut target = TARGET_BG;
    let mut best = 2.0;
    for _ in 0..15 {
        best = solve_log_d(&valid, target, PROTECT_B);
        if pressure > 0.6 {
            target *= 1.0 - 0.15 * pressure;
        }
        let d = 10.0_f64.powf(best);
        let stretched: Vec<f32> = valid.iter().map(|v| hyperbolic(*v as f64, d, PROTECT_B) as f32).collect();
        let med = median(&stretched);
        let dev = std_pop(&stretched);
        let min_v = stretched.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let floor = min_v.max(med - 2.7 * dev);
        if floor <= 0.001 {
            break;
        }
        target -= 0.015;
        if target < 0.05 {
            break;
        }
    }
    best
}

fn apply_mtf_value(x: f64, m: f64) -> f64 {
    let denom = (2.0 * m - 1.0) * x - m;
    if denom == 0.0 {
        return 0.0;
    }
    let value = ((m - 1.0) * x) / denom;
    if value.is_nan() {
        0.0
    } else if value.is_infinite() {
        if value.is_sign_positive() { 1.0 } else { 0.0 }
    } else {
        value
    }
}

fn luminance_plane(image: &[f32], channels: usize, pixels: usize) -> Vec<f32> {
    if channels >= 3 {
        (0..pixels).map(|i| luma_at(image, pixels, i)).collect()
    } else {
        plane(image, 0, pixels).iter().copied().map(finite).collect()
    }
}

fn percentile_strided(values: &[f32], stride: usize, percent: f64) -> f64 {
    let sample: Vec<f32> = values.iter().step_by(stride.max(1)).copied().collect();
    percentile_linear(&sample, percent)
}

fn adaptive_output_scaling(image: &mut [f32], channels: usize, height: usize, width: usize, target_bg: f64) {
    let pixels = height * width;
    if pixels == 0 || width == 0 {
        return;
    }
    let luma = luminance_plane(image, channels, pixels);
    let median_l = median(&luma);
    let std_l = std_pop(&luma);
    let min_l = luma.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let global_floor = min_l.max(median_l - 2.7 * std_l);
    let mut abs_max = f64::NEG_INFINITY;
    let mut idx_max = 0usize;
    for (i, value) in luma.iter().enumerate() {
        if *value as f64 > abs_max {
            abs_max = *value as f64;
            idx_max = i;
        }
    }
    let mut valid_physical_max = true;
    if abs_max > 0.001 {
        let y = idx_max / width;
        let x = idx_max % width;
        let y0 = y.saturating_sub(1);
        let y1 = (y + 2).min(height);
        let x0 = x.saturating_sub(1);
        let x1 = (x + 2).min(width);
        let mut max_neighbor = f64::NEG_INFINITY;
        let mut any = false;
        for yy in y0..y1 {
            for xx in x0..x1 {
                let v = luma[yy * width + xx] as f64;
                if v < abs_max {
                    any = true;
                    max_neighbor = max_neighbor.max(v);
                }
            }
        }
        if any && max_neighbor < abs_max * 0.20 {
            valid_physical_max = false;
        }
    }
    let mut soft_ceil = if channels >= 3 {
        let stride = (pixels / 500_000).max(1);
        let red = percentile_strided(plane(image, 0, pixels), stride, 99.0);
        let green = percentile_strided(plane(image, 1, pixels), stride, 99.0);
        let blue = percentile_strided(plane(image, 2, pixels), stride, 99.0);
        red.max(green).max(blue)
    } else {
        let stride = (pixels / 200_000).max(1);
        percentile_strided(&luma, stride, 99.0)
    };
    if soft_ceil <= global_floor {
        soft_ceil = global_floor + 1e-6;
    }
    if abs_max <= soft_ceil {
        abs_max = soft_ceil + 1e-6;
    }
    let scale_contrast = (0.98 - RTU_PEDESTAL) / (soft_ceil - global_floor + 1e-9);
    let final_scale = if valid_physical_max {
        let scale_limit = (1.0 - RTU_PEDESTAL) / (abs_max - global_floor + 1e-9);
        scale_contrast.min(scale_limit)
    } else {
        scale_contrast
    };
    let planes = if channels >= 3 { 3 } else { channels };
    for c in 0..planes {
        for i in 0..pixels {
            let v = image[c * pixels + i] as f64;
            image[c * pixels + i] = ((v - global_floor) * final_scale + RTU_PEDESTAL).clamp(0.0, 1.0) as f32;
        }
    }
    let luma = luminance_plane(image, channels, pixels);
    let current_bg = median(&luma);
    if current_bg > 0.0 && current_bg < 1.0 && (current_bg - target_bg).abs() > 1e-3 {
        let denom = current_bg * (2.0 * target_bg - 1.0) - target_bg;
        if denom != 0.0 {
            let m = (current_bg * (target_bg - 1.0)) / denom;
            for c in 0..planes {
                for i in 0..pixels {
                    image[c * pixels + i] = apply_mtf_value(image[c * pixels + i] as f64, m) as f32;
                }
            }
        }
    }
}

fn soft_clip(image: &mut [f32], channels: usize, pixels: usize, threshold: f64, rolloff: f64) {
    let planes = if channels >= 3 { 3 } else { channels };
    for c in 0..planes {
        for i in 0..pixels {
            let v = image[c * pixels + i] as f64;
            let y = if v > threshold {
                let t = ((v - threshold) / (1.0 - threshold + 1e-9)).clamp(0.0, 1.0);
                threshold + (1.0 - threshold) * (1.0 - (1.0 - t).powf(rolloff))
            } else {
                v
            };
            image[c * pixels + i] = y.clamp(0.0, 1.0) as f32;
        }
    }
}

fn process_ready(data: &[f32], channels: usize, height: usize, width: usize, log_d: f64) -> Vec<f32> {
    let pixels = height * width;
    let mut image = vec![0.0f32; channels * pixels];
    let anchor = adaptive_anchor(data, channels, height, width);
    let d = 10.0_f64.powf(log_d);
    if channels >= 3 {
        for i in 0..pixels {
            let r = (finite(data[i]) as f64 - anchor).max(0.0);
            let g = (finite(data[pixels + i]) as f64 - anchor).max(0.0);
            let b = (finite(data[2 * pixels + i]) as f64 - anchor).max(0.0);
            let l = SEESTAR_S30[0] * r + SEESTAR_S30[1] * g + SEESTAR_S30[2] * b;
            let l_str = hyperbolic(l, d, PROTECT_B).clamp(0.0, 1.0);
            let k = l_str.powf(CONVERGENCE);
            let safe = l + 1e-9;
            let lift = |value: f64| {
                let ratio = value / safe;
                let mixed = ratio * (1.0 - k) + k;
                (l_str * mixed * 0.995 + 0.005).clamp(0.0, 1.0) as f32
            };
            image[i] = lift(r);
            image[pixels + i] = lift(g);
            image[2 * pixels + i] = lift(b);
        }
        if channels > 3 {
            image[3 * pixels..].copy_from_slice(&data[3 * pixels..channels * pixels]);
        }
    } else {
        for c in 0..channels {
            for i in 0..pixels {
                let v = (finite(data[c * pixels + i]) as f64 - anchor).max(0.0);
                let stretched = hyperbolic(v, d, PROTECT_B).clamp(0.0, 1.0);
                image[c * pixels + i] = (stretched * 0.995 + 0.005).clamp(0.0, 1.0) as f32;
            }
        }
    }
    adaptive_output_scaling(&mut image, channels, height, width, TARGET_BG);
    soft_clip(&mut image, channels, pixels, 0.98, 2.0);
    image
}

fn percentile(values: &[f32], percent: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((percent / 100.0) * (copy.len() - 1) as f64).round() as usize;
    copy[rank.min(copy.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtf_matches_the_display_formula() {
        let y = mtf(0.25, 0.2);
        assert!((y - ((0.25 - 1.0) * 0.2) / ((0.5 - 1.0) * 0.2 - 0.25)).abs() < 1e-5);
    }

    #[test]
    fn hyperbolic_matches_the_inverse_sinh() {
        let data = 0.01;
        let d = 10.0;
        let b = 6.0;
        let asinh = |z: f64| (z + (z * z + 1.0).sqrt()).ln();
        let term1 = asinh(d * data + b);
        let term2 = asinh(b);
        let expected = (term1 - term2) / (asinh(d + b) - term2);
        assert!((hyperbolic(data, d, b) - expected).abs() < 1e-12);
    }

    #[test]
    fn percentile_linear_interpolates() {
        let values = [0.0f32, 10.0, 20.0, 30.0];
        assert!((percentile_linear(&values, 50.0) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn hypermetric_lifts_the_sky_toward_the_target() {
        let width = 48usize;
        let height = 48usize;
        let pixels = width * height;
        let mut data = vec![0.004f32; 3 * pixels];
        for c in 0..3 {
            for y in 20..23 {
                for x in 20..23 {
                    data[c * pixels + y * width + x] = 0.85;
                }
            }
        }
        let result = auto_stretch_hypermetric(&data, 3, height, width);
        assert!((0.0..=7.0).contains(&result.log_d));
        assert!(result.image.len() == 3 * pixels);
        assert!(result.image.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        let luma: Vec<f32> = (0..pixels).map(|i| {
            let r = result.image[i] as f64;
            let g = result.image[pixels + i] as f64;
            let b = result.image[2 * pixels + i] as f64;
            (SEESTAR_S30[0] * r + SEESTAR_S30[1] * g + SEESTAR_S30[2] * b) as f32
        }).collect();
        let med = median(&luma);
        assert!((med - 0.20).abs() < 0.05, "médiane du fond {med}");
    }
}
