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
}
