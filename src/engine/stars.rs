// SPDX-License-Identifier: GPL-3.0-or-later
//! Détection d'étoiles sur une luminance binée 2×2.

use dubgpustack_kernels::measure::measure_one;

pub const MESH: usize = 32;
pub const MAX_CANDIDATES: usize = 4000;
pub const KEEP: usize = 800;

#[derive(Clone, Debug, Default)]
pub struct StarList {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub flux: Vec<f64>,
    pub fwhm: Vec<f64>,
    pub roundness: Vec<f64>,
    pub saturated: Vec<bool>,
}

impl StarList {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn brightest(&self, n: usize) -> Self {
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_by(|&a, &b| self.flux[b].partial_cmp(&self.flux[a]).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(n.min(order.len()));
        self.pick(&order)
    }

    pub fn pick(&self, order: &[usize]) -> Self {
        Self {
            x: order.iter().map(|i| self.x[*i]).collect(),
            y: order.iter().map(|i| self.y[*i]).collect(),
            flux: order.iter().map(|i| self.flux[*i]).collect(),
            fwhm: order.iter().map(|i| self.fwhm[*i]).collect(),
            roundness: order.iter().map(|i| self.roundness[*i]).collect(),
            saturated: order.iter().map(|i| self.saturated[*i]).collect(),
        }
    }

    pub fn to_array(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len() * 6);
        for i in 0..self.len() {
            out.extend_from_slice(&[self.x[i] as f32, self.y[i] as f32, self.flux[i] as f32, self.fwhm[i] as f32, self.roundness[i] as f32, if self.saturated[i] { 1.0 } else { 0.0 }]);
        }
        out
    }

    pub fn from_array(data: &[f32]) -> Self {
        let mut stars = Self::default();
        for chunk in data.chunks_exact(6) {
            stars.x.push(chunk[0] as f64);
            stars.y.push(chunk[1] as f64);
            stars.flux.push(chunk[2] as f64);
            stars.fwhm.push(chunk[3] as f64);
            stars.roundness.push(chunk[4] as f64);
            stars.saturated.push(chunk[5] > 0.5);
        }
        stars
    }
}

#[derive(Clone, Debug)]
pub struct FrameMetrics {
    pub nstars: i64,
    pub fwhm: f64,
    pub roundness: f64,
    pub background: f64,
    pub noise: f64,
}

impl Default for FrameMetrics {
    fn default() -> Self {
        Self { nstars: 0, fwhm: f64::NAN, roundness: f64::NAN, background: 0.0, noise: 0.0 }
    }
}

pub fn bin2(image: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let w = width / 2 * 2;
    let h = height / 2 * 2;
    let bw = w / 2;
    let bh = h / 2;
    let mut out = vec![0.0f32; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            let i = (y * 2) * width + x * 2;
            out[y * bw + x] = (image[i] + image[i + 1] + image[i + width] + image[i + width + 1]) * 0.25;
        }
    }
    (out, bw, bh)
}

fn median_of(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { 0.5 * (values[n / 2 - 1] + values[n / 2]) }
}

fn background(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let ny = (height / MESH).max(1);
    let nx = (width / MESH).max(1);
    let mut mesh = vec![0.0f32; nx * ny];
    for cy in 0..ny {
        for cx in 0..nx {
            let mut cell = Vec::with_capacity(MESH * MESH);
            for y in 0..MESH {
                for x in 0..MESH {
                    cell.push(image[(cy * MESH + y) * width + cx * MESH + x]);
                }
            }
            mesh[cy * nx + cx] = median_of(&mut cell);
        }
    }
    if ny >= 3 && nx >= 3 {
        let source = mesh.clone();
        for cy in 0..ny {
            for cx in 0..nx {
                let mut win = Vec::with_capacity(9);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let yy = (cy as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                        let xx = (cx as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                        win.push(source[yy * nx + xx]);
                    }
                }
                mesh[cy * nx + cx] = median_of(&mut win);
            }
        }
    }
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        let fy = if height == 1 { 0.0 } else { y as f32 * (ny as f32 - 1.0) / (height as f32 - 1.0) };
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(ny - 1);
        let ty = fy - y0 as f32;
        for x in 0..width {
            let fx = if width == 1 { 0.0 } else { x as f32 * (nx as f32 - 1.0) / (width as f32 - 1.0) };
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(nx - 1);
            let tx = fx - x0 as f32;
            let a = mesh[y0 * nx + x0] * (1.0 - tx) + mesh[y0 * nx + x1] * tx;
            let b = mesh[y1 * nx + x0] * (1.0 - tx) + mesh[y1 * nx + x1] * tx;
            out[y * width + x] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

fn gaussian_kernel() -> Vec<f32> {
    let radius = 4i32;
    let mut kernel = Vec::new();
    let mut sum = 0.0f32;
    for i in -radius..=radius {
        let v = (-0.5 * (i as f32) * (i as f32)).exp();
        kernel.push(v);
        sum += v;
    }
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn reflect(index: i32, n: i32) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = index.rem_euclid(period);
    if m >= n {
        m = period - m;
    }
    m as usize
}

fn gaussian(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let kernel = gaussian_kernel();
    let radius = 4i32;
    let mut tmp = vec![0.0f32; width * height];
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width as i32 {
            let mut acc = 0.0f32;
            for k in 0..kernel.len() {
                let sx = reflect(x + k as i32 - radius, width as i32);
                acc += kernel[k] * image[y * width + sx];
            }
            tmp[y * width + x as usize] = acc;
        }
    }
    for y in 0..height as i32 {
        for x in 0..width {
            let mut acc = 0.0f32;
            for k in 0..kernel.len() {
                let sy = reflect(y + k as i32 - radius, height as i32);
                acc += kernel[k] * tmp[sy * width + x];
            }
            out[y as usize * width + x] = acc;
        }
    }
    out
}

fn subsample_median_sigma(image: &[f32], width: usize) -> (f32, f32) {
    let mut sample = Vec::new();
    let height = if width == 0 { 0 } else { image.len() / width };
    for y in (0..height).step_by(3) {
        for x in (0..width).step_by(3) {
            sample.push(image[y * width + x]);
        }
    }
    if sample.is_empty() {
        return (0.0, 1e-6);
    }
    let med = median_of(&mut sample.clone());
    let mut abs = sample.iter().map(|v| (v - med).abs()).collect::<Vec<_>>();
    let sigma = (median_of(&mut abs) * 1.4826).max(1e-6);
    (med, sigma)
}

fn local_max(image: &[f32], width: usize, height: usize, x: usize, y: usize) -> bool {
    let center = image[y * width + x];
    for dy in -2..=2 {
        for dx in -2..=2 {
            let xx = x as i32 + dx;
            let yy = y as i32 + dy;
            if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                continue;
            }
            if image[yy as usize * width + xx as usize] > center {
                return false;
            }
        }
    }
    true
}

pub fn detect_luma(luma: &[f32], width: usize, height: usize, saturation: f32) -> (StarList, FrameMetrics) {
    if width < 16 || height < 16 {
        return (StarList::default(), FrameMetrics { background: 0.0, noise: 1e-6, ..FrameMetrics::default() });
    }
    let bg = background(luma, width, height);
    let mut residual = vec![0.0f32; width * height];
    for i in 0..residual.len() {
        residual[i] = luma[i] - bg[i];
    }
    let (_, sigma) = subsample_median_sigma(&residual, width);
    let smooth = gaussian(&residual, width, height);
    let (s_med, s_sig) = subsample_median_sigma(&smooth, width);
    let thresh_peak = s_med + 5.0 * s_sig;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let margin = 6usize;
    for y in margin..height.saturating_sub(margin) {
        for x in margin..width.saturating_sub(margin) {
            let v = smooth[y * width + x];
            if v > thresh_peak && local_max(&smooth, width, height, x, y) {
                xs.push(x as i32);
                ys.push(y as i32);
            }
        }
    }
    let bg_med = {
        let mut copy = bg.clone();
        median_of(&mut copy) as f64
    };
    if xs.is_empty() {
        return (StarList::default(), FrameMetrics { background: bg_med, noise: sigma as f64, ..FrameMetrics::default() });
    }
    if xs.len() > MAX_CANDIDATES {
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| smooth[ys[b] as usize * width + xs[b] as usize].partial_cmp(&smooth[ys[a] as usize * width + xs[a] as usize]).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(MAX_CANDIDATES);
        xs = order.iter().map(|i| xs[*i]).collect();
        ys = order.iter().map(|i| ys[*i]).collect();
    }
    let mut measured = Vec::new();
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let row = measure_one(&residual, luma, width as i32, height as i32, x, y, 2.0 * sigma, saturation);
        if row[0] >= 0.0 {
            measured.push(row);
        }
    }
    let mut fwhm: Vec<f32> = measured.iter().map(|row| row[3] * 2.0).collect();
    let mut keep = Vec::new();
    for (index, row) in measured.iter().enumerate() {
        if row[3] > 0.55 && row[4] > 0.25 && fwhm[index].is_finite() {
            keep.push(index);
        }
    }
    if keep.is_empty() {
        return (StarList::default(), FrameMetrics { background: bg_med, noise: sigma as f64, ..FrameMetrics::default() });
    }
    let mut fwhm_kept: Vec<f32> = keep.iter().map(|i| fwhm[*i]).collect();
    let med_fwhm = median_of(&mut fwhm_kept);
    keep.retain(|i| fwhm[*i] < 3.0 * med_fwhm + 2.0);
    fwhm = keep.iter().map(|i| fwhm[*i]).collect();
    let total = keep.len() as i64;
    let mut order = keep;
    order.sort_by(|&a, &b| measured[b][2].partial_cmp(&measured[a][2]).unwrap_or(std::cmp::Ordering::Equal));
    order.truncate(KEEP.min(order.len()));
    let mut stars = StarList::default();
    for index in &order {
        let row = measured[*index];
        stars.x.push(row[0] as f64 * 2.0);
        stars.y.push(row[1] as f64 * 2.0);
        stars.flux.push(row[2] as f64);
        stars.fwhm.push((row[3] * 2.0) as f64);
        stars.roundness.push(row[4] as f64);
        stars.saturated.push(row[6] > 0.5);
    }
    let good: Vec<usize> = (0..stars.len()).filter(|&i| !stars.saturated[i] && stars.roundness[i] > 0.4).collect();
    let use_idx: Vec<usize> = if good.len() >= 5 { good } else { (0..stars.len()).collect() };
    let metric = |values: Vec<f64>| {
        if values.is_empty() { f64::NAN } else { median_of(&mut values.iter().map(|x| *x as f32).collect::<Vec<_>>()) as f64 }
    };
    let metrics = FrameMetrics {
        nstars: total,
        fwhm: metric(use_idx.iter().map(|i| stars.fwhm[*i]).collect()),
        roundness: metric(use_idx.iter().map(|i| stars.roundness[*i]).collect()),
        background: bg_med,
        noise: sigma as f64,
    };
    let _ = fwhm;
    (stars, metrics)
}

pub fn luminance(data: &[f32], channels: usize, width: usize, height: usize, cfa: bool) -> (Vec<f32>, usize, usize) {
    let plane = width * height;
    if cfa || channels == 1 {
        return bin2(&data[..plane.min(data.len())], width, height);
    }
    let mut mean = vec![0.0f32; plane];
    for i in 0..plane {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += data[c * plane + i];
        }
        mean[i] = sum / channels as f32;
    }
    bin2(&mean, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bright_spot_is_a_star() {
        let width = 64usize;
        let height = 64usize;
        let mut image = vec![0.05f32; width * height];
        for y in 28..36 {
            for x in 28..36 {
                let dx = x as f32 - 31.5;
                let dy = y as f32 - 31.5;
                image[y * width + x] += (-0.4 * (dx * dx + dy * dy)).exp();
            }
        }
        let (binned, bw, bh) = bin2(&image, width, height);
        let (stars, metrics) = detect_luma(&binned, bw, bh, 0.98);
        assert!(metrics.nstars >= 1, "étoiles {}", metrics.nstars);
        assert!(!stars.is_empty());
    }
}
