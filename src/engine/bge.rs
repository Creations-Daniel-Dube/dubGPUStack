// SPDX-License-Identifier: GPL-3.0-or-later
//! Extraction de fond automatique, d'après AutoBGE 2.0.2
//! (Adrian Knagg-Baugh, à partir du code SetiAstro / Franklin Marek, GPL-3.0-or-later).
//!
//! Réglages figés : 100 points intérieurs, polynôme degré 2, RBF multiquadric
//! lissée à 0,1, sous-échantillonnage 4, médiane de patch 15×15.
//! L'image est planaire, mono ou RVB, dans `[0, 1]`.

use std::sync::atomic::{AtomicBool, Ordering};

const SCALE: usize = 4;
const INTERIOR_POINTS: usize = 100;
const DEGREE: usize = 2;
const SMOOTH: f64 = 0.1;
const PATCH: i32 = 15;
const TARGET_MEDIAN: f64 = 0.25;
const BORDER: i32 = 10;
const DESCEND_STEPS: usize = 100;
const RNG_SEED: u64 = 0xA17B_6E01_D0B5;

struct StretchState {
    mins: Vec<f64>,
    medians: Vec<f64>,
}

pub fn apply(image: &mut [f32], channels: usize, height: usize, width: usize, cancel: &AtomicBool, mut progress: impl FnMut(f64, f64)) -> Result<(), String> {
    let pixels = height.saturating_mul(width);
    if channels != 1 && channels != 3 {
        return Err("l'extraction de fond attend une image mono ou couleur".to_string());
    }
    if height < 2 || width < 2 || image.len() < channels.saturating_mul(pixels) {
        return Err("image trop petite pour l'extraction de fond".to_string());
    }
    progress(0.0, 4.0);
    let state = stretch_image(image, channels, pixels);
    let anchor = median_of_copy(&image[..channels * pixels]);
    check(cancel)?;
    progress(1.0, 4.0);
    subtract_polynomial(image, channels, height, width, anchor, cancel)?;
    check(cancel)?;
    progress(2.0, 4.0);
    subtract_rbf(image, channels, height, width, anchor, cancel, &mut progress)?;
    check(cancel)?;
    progress(3.0, 4.0);
    unstretch_image(image, channels, pixels, &state);
    progress(4.0, 4.0);
    Ok(())
}

fn subtract_polynomial(image: &mut [f32], channels: usize, height: usize, width: usize, anchor: f64, cancel: &AtomicBool) -> Result<(), String> {
    let (small, sh, sw) = downsample(image, channels, height, width, SCALE);
    let points = sample_points(&small, channels, sh, sw);
    if points.is_empty() {
        return Err("aucun point d'échantillon pour le fond polynomial".to_string());
    }
    for channel in 0..channels {
        check(cancel)?;
        let plane = plane_of(&small, channel, sh * sw);
    let z = patch_values(&plane, sw, sh, &points);
    let (points, z) = unique_samples(&points, &z);
    let model = if points.len() < polynomial_terms(DEGREE).len() {
        let mean = z.iter().sum::<f64>() / z.len().max(1) as f64;
        vec![mean as f32; sw * sh]
    } else {
        let coeff = fit_polynomial(&points, &z, DEGREE)?;
        eval_polynomial(&coeff, DEGREE, sw, sh)
    };
        let full = resize_lanczos(&model, sh, sw, height, width);
        let offset = channel * height * width;
        for (pixel, background) in image[offset..offset + height * width].iter_mut().zip(&full) {
            *pixel -= background;
        }
    }
    shift_median(image, channels * height * width, anchor);
    clip01(image, channels * height * width);
    Ok(())
}

fn subtract_rbf(image: &mut [f32], channels: usize, height: usize, width: usize, anchor: f64, cancel: &AtomicBool, progress: &mut impl FnMut(f64, f64)) -> Result<(), String> {
    let (small, sh, sw) = downsample(image, channels, height, width, SCALE);
    let points = sample_points(&small, channels, sh, sw);
    if points.is_empty() {
        return Err("aucun point d'échantillon pour le fond RBF".to_string());
    }
    for channel in 0..channels {
        check(cancel)?;
        progress(2.0 + channel as f64 / channels as f64, 4.0);
        let plane = plane_of(&small, channel, sh * sw);
        let z = patch_values(&plane, sw, sh, &points);
        let (points, z) = unique_samples(&points, &z);
        let model = if points.len() < 6 {
            let mean = z.iter().sum::<f64>() / z.len().max(1) as f64;
            vec![mean as f32; sw * sh]
        } else {
            let nodes = fit_rbf(&points, &z, SMOOTH)?;
            eval_rbf(&points, &nodes, sw, sh, cancel)?
        };
        let full = resize_lanczos(&model, sh, sw, height, width);
        let offset = channel * height * width;
        for (pixel, background) in image[offset..offset + height * width].iter_mut().zip(&full) {
            *pixel -= background;
        }
    }
    shift_median(image, channels * height * width, anchor);
    clip01(image, channels * height * width);
    Ok(())
}

fn plane_of(image: &[f32], channel: usize, pixels: usize) -> Vec<f32> {
    image[channel * pixels..(channel + 1) * pixels].to_vec()
}

fn stretch_image(image: &mut [f32], channels: usize, pixels: usize) -> StretchState {
    let mut mins = Vec::with_capacity(channels);
    let mut medians = Vec::with_capacity(channels);
    for channel in 0..channels {
        let plane = &mut image[channel * pixels..(channel + 1) * pixels];
        let min = plane.iter().copied().fold(f32::MAX, f32::min) as f64;
        for value in plane.iter_mut() {
            *value -= min as f32;
        }
        let median = median_of_copy(plane);
        if median != 0.0 {
            for value in plane.iter_mut() {
                *value = curve(*value, median, TARGET_MEDIAN);
            }
        }
        mins.push(min);
        medians.push(median);
    }
    clip01(image, channels * pixels);
    StretchState { mins, medians }
}

fn unstretch_image(image: &mut [f32], channels: usize, pixels: usize, state: &StretchState) {
    for channel in 0..channels {
        let plane = &mut image[channel * pixels..(channel + 1) * pixels];
        let current = median_of_copy(plane);
        let original = state.medians[channel];
        let min = state.mins[channel];
        if current != 0.0 && original != 0.0 {
            for value in plane.iter_mut() {
                *value = curve(*value, current, original);
            }
        }
        for value in plane.iter_mut() {
            *value += min as f32;
        }
    }
    clip01(image, channels * pixels);
}

/// Courbe d'étirement du script : `y = (m - 1) t x / (m (t + x - 1) - t x)`.
fn curve(x: f32, median: f64, target: f64) -> f32 {
    let x = x as f64;
    let numerator = (median - 1.0) * target * x;
    let mut denominator = median * (target + x - 1.0) - target * x;
    if denominator == 0.0 {
        denominator = 1e-6;
    }
    (numerator / denominator) as f32
}

fn shift_median(image: &mut [f32], count: usize, target: f64) {
    let current = median_of_copy(&image[..count]);
    let delta = (target - current) as f32;
    for value in &mut image[..count] {
        *value += delta;
    }
}

fn clip01(image: &mut [f32], count: usize) {
    for value in &mut image[..count] {
        *value = value.clamp(0.0, 1.0);
    }
}

fn downsample(image: &[f32], channels: usize, height: usize, width: usize, scale: usize) -> (Vec<f32>, usize, usize) {
    let out_h = (height / scale).max(1);
    let out_w = (width / scale).max(1);
    let mut out = vec![0.0f32; channels * out_h * out_w];
    for channel in 0..channels {
        let src = &image[channel * height * width..(channel + 1) * height * width];
        let dst = &mut out[channel * out_h * out_w..(channel + 1) * out_h * out_w];
        for y in 0..out_h {
            let y0 = y * scale;
            let y1 = (y0 + scale).min(height);
            for x in 0..out_w {
                let x0 = x * scale;
                let x1 = (x0 + scale).min(width);
                let mut sum = 0.0f64;
                let mut count = 0.0f64;
                for row in y0..y1 {
                    for col in x0..x1 {
                        sum += src[row * width + col] as f64;
                        count += 1.0;
                    }
                }
                dst[y * out_w + x] = (sum / count) as f32;
            }
        }
    }
    (out, out_h, out_w)
}

fn sample_points(image: &[f32], channels: usize, height: usize, width: usize) -> Vec<(i32, i32)> {
    let luminance = luminance(image, channels, height, width);
    let mut points = Vec::new();
    let margin = border_margin(height, width);
    let right = width as i32 - margin - 1;
    let bottom = height as i32 - margin - 1;
    let corners = [(margin, margin), (right, margin), (margin, bottom), (right, bottom)];
    for (x, y) in corners {
        let (x, y) = descend(&luminance, width, height, x, y);
        points.push((x, y));
    }
    for x in linspace_trunc(margin, width as i32 - margin, 5) {
        let (x, y) = descend(&luminance, width, height, x, margin);
        points.push((x, y));
        let (x, y) = descend(&luminance, width, height, x, bottom);
        points.push((x, y));
    }
    for y in linspace_trunc(margin, height as i32 - margin, 5) {
        let (x, y) = descend(&luminance, width, height, margin, y);
        points.push((x, y));
        let (x, y) = descend(&luminance, width, height, right, y);
        points.push((x, y));
    }
    let mut rng = Rng(RNG_SEED);
    let half_h = height / 2;
    let half_w = width / 2;
    let quartiles = [
        (0, 0, half_h, half_w),
        (0, half_w, half_h, width - half_w),
        (half_h, 0, height - half_h, half_w),
        (half_h, half_w, height - half_h, width - half_w),
    ];
    let per_quartile = INTERIOR_POINTS / 4;
    for (y0, x0, qh, qw) in quartiles {
        if qh == 0 || qw == 0 || per_quartile == 0 {
            continue;
        }
        let mut values = Vec::with_capacity(qh * qw);
        for y in y0..y0 + qh {
            let row = y * width + x0;
            values.extend_from_slice(&luminance[row..row + qw]);
        }
        let threshold = median_of_copy(&values) as f32;
        let mut eligible = Vec::new();
        for (index, value) in values.iter().copied().enumerate() {
            if value < threshold {
                let y = y0 + index / qw;
                let x = x0 + index % qw;
                eligible.push((x as i32, y as i32));
            }
        }
        for (x, y) in choose(&mut rng, &eligible, per_quartile) {
            points.push(descend(&luminance, width, height, x, y));
        }
    }
    points
}

fn border_margin(height: usize, width: usize) -> i32 {
    let limit = height.min(width) as i32;
    if limit <= 1 {
        0
    } else {
        BORDER.min(limit / 2)
    }
}

fn linspace_trunc(start: i32, stop: i32, count: usize) -> Vec<i32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    (0..count)
        .map(|index| {
            let t = start as f64 + (stop - start) as f64 * index as f64 / (count - 1) as f64;
            t as i32
        })
        .collect()
}

fn descend(image: &[f32], width: usize, height: usize, x: i32, y: i32) -> (i32, i32) {
    let mut x = x.clamp(0, width as i32 - 1);
    let mut y = y.clamp(0, height as i32 - 1);
    let half = PATCH / 2;
    for _ in 0..DESCEND_STEPS {
        let current = patch_median(image, width, height, x, y, half);
        let mut best: Option<(i32, i32, f64)> = None;
        let y_from = (y - 1).max(0);
        let y_to = (y + 1).min(height as i32 - 1);
        let x_from = (x - 1).max(0);
        let x_to = (x + 1).min(width as i32 - 1);
        for nx in x_from..=x_to {
            for ny in y_from..=y_to {
                if nx == x && ny == y {
                    continue;
                }
                let value = patch_median(image, width, height, nx, ny, half);
                if best.is_none_or(|(_, _, best_value)| value < best_value) {
                    best = Some((nx, ny, value));
                }
            }
        }
        let Some((nx, ny, value)) = best else { break };
        if value >= current {
            break;
        }
        x = nx;
        y = ny;
    }
    (x, y)
}

fn patch_values(image: &[f32], width: usize, height: usize, points: &[(i32, i32)]) -> Vec<f64> {
    let half = PATCH / 2;
    points.iter().map(|(x, y)| patch_median(image, width, height, *x, *y, half)).collect()
}

fn patch_median(image: &[f32], width: usize, height: usize, x: i32, y: i32, half: i32) -> f64 {
    let x0 = (x - half).max(0) as usize;
    let x1 = ((x + half + 1).max(0) as usize).min(width);
    let y0 = (y - half).max(0) as usize;
    let y1 = ((y + half + 1).max(0) as usize).min(height);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let mut values = Vec::with_capacity((y1 - y0) * (x1 - x0));
    for row in y0..y1 {
        values.extend_from_slice(&image[row * width + x0..row * width + x1]);
    }
    median_of_copy(&values)
}

fn luminance(image: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let pixels = height * width;
    if channels == 1 {
        return image[..pixels].to_vec();
    }
    let mut out = vec![0.0f32; pixels];
    let green = pixels;
    let blue = 2 * pixels;
    for pixel in 0..pixels {
        out[pixel] = image[pixel] * 0.2989 + image[green + pixel] * 0.5870 + image[blue + pixel] * 0.1140;
    }
    out
}

fn fit_polynomial(points: &[(i32, i32)], values: &[f64], degree: usize) -> Result<Vec<f64>, String> {
    let terms = polynomial_terms(degree);
    let rows = points.len().min(values.len());
    if rows < terms.len() {
        return Err("pas assez de points pour le polynôme de fond".to_string());
    }
    let mut design = vec![0.0f64; rows * terms.len()];
    for (row, (x, y)) in points.iter().take(rows).enumerate() {
        for (col, (i, j)) in terms.iter().enumerate() {
            design[row * terms.len() + col] = powi(*x as f64, *i) * powi(*y as f64, *j);
        }
    }
    Ok(least_squares(&design, rows, terms.len(), &values[..rows]))
}

fn eval_polynomial(coeff: &[f64], degree: usize, width: usize, height: usize) -> Vec<f32> {
    let terms = polynomial_terms(degree);
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0f64;
            for (coeff, (i, j)) in coeff.iter().zip(&terms) {
                value += coeff * powi(x as f64, *i) * powi(y as f64, *j);
            }
            out[y * width + x] = value as f32;
        }
    }
    out
}

fn polynomial_terms(degree: usize) -> Vec<(u32, u32)> {
    let mut terms = Vec::new();
    for i in 0..=degree {
        for j in 0..=degree - i {
            terms.push((i as u32, j as u32));
        }
    }
    terms
}

fn fit_rbf(points: &[(i32, i32)], values: &[f64], smooth: f64) -> Result<Vec<f64>, String> {
    let rows = points.len().min(values.len());
    if rows == 0 {
        return Err("pas assez de points pour le fond RBF".to_string());
    }
    let mut matrix = vec![0.0f64; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            let (x0, y0) = points[i];
            let (x1, y1) = points[j];
            let dx = (x0 - x1) as f64;
            let dy = (y0 - y1) as f64;
            let mut kernel = multiquadric(dx, dy);
            if i == j {
                kernel -= smooth;
            }
            matrix[i * rows + j] = kernel;
        }
    }
    let mut nodes = values[..rows].to_vec();
    solve_linear(&mut matrix, rows, &mut nodes)?;
    Ok(nodes)
}

fn eval_rbf(points: &[(i32, i32)], nodes: &[f64], width: usize, height: usize, cancel: &AtomicBool) -> Result<Vec<f32>, String> {
    let nodes: Vec<f32> = nodes.iter().map(|value| *value as f32).collect();
    let xs: Vec<f32> = points.iter().take(nodes.len()).map(|(x, _)| *x as f32).collect();
    let ys: Vec<f32> = points.iter().take(nodes.len()).map(|(_, y)| *y as f32).collect();
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        if y % 32 == 0 {
            check(cancel)?;
        }
        let yf = y as f32;
        for x in 0..width {
            let xf = x as f32;
            let mut value = 0.0f32;
            for ((cx, cy), node) in xs.iter().zip(&ys).zip(&nodes) {
                let dx = xf - cx;
                let dy = yf - cy;
                value += node * (dx * dx + dy * dy + 1.0).sqrt();
            }
            out[y * width + x] = value;
        }
    }
    Ok(out)
}

fn multiquadric(dx: f64, dy: f64) -> f64 {
    (dx * dx + dy * dy + 1.0).sqrt()
}

fn resize_lanczos(src: &[f32], src_h: usize, src_w: usize, dst_h: usize, dst_w: usize) -> Vec<f32> {
    if src_h == dst_h && src_w == dst_w {
        return src.to_vec();
    }
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return vec![0.0; dst_h * dst_w];
    }
    let horizontal = resize_axis(src, src_h, src_w, src_h, dst_w, true);
    resize_axis(&horizontal, src_h, dst_w, dst_h, dst_w, false)
}

fn resize_axis(src: &[f32], src_h: usize, src_w: usize, dst_h: usize, dst_w: usize, horizontal: bool) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_h * dst_w];
    let (src_span, dst_span) = if horizontal { (src_w, dst_w) } else { (src_h, dst_h) };
    let scale = src_span as f64 / dst_span as f64;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
    let rows = dst_h.max(1);
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (band, dst_rows) in dst.chunks_mut(chunk * dst_w).enumerate() {
            let y0 = band * chunk;
            scope.spawn(move || {
                for (local_y, row) in dst_rows.chunks_mut(dst_w).enumerate() {
                    let y = y0 + local_y;
                    if horizontal {
                        let src_row = &src[y * src_w..(y + 1) * src_w];
                        for x in 0..dst_w {
                            let pos = (x as f64 + 0.5) * scale - 0.5;
                            row[x] = lanczos_sample(src_row, pos);
                        }
                    } else {
                        for x in 0..dst_w {
                            let pos = (y as f64 + 0.5) * scale - 0.5;
                            row[x] = lanczos_column(src, src_h, src_w, x, pos);
                        }
                    }
                }
            });
        }
    });
    dst
}

fn lanczos_sample(row: &[f32], pos: f64) -> f32 {
    let base = pos.floor() as isize;
    let mut acc = 0.0f64;
    for tap in -3..=4 {
        let index = base + tap;
        let weight = lanczos4(pos - index as f64);
        if weight == 0.0 {
            continue;
        }
        let clamped = index.clamp(0, row.len() as isize - 1) as usize;
        acc += weight * row[clamped] as f64;
    }
    acc as f32
}

fn lanczos_column(src: &[f32], height: usize, width: usize, x: usize, pos: f64) -> f32 {
    let base = pos.floor() as isize;
    let mut acc = 0.0f64;
    for tap in -3..=4 {
        let index = base + tap;
        let weight = lanczos4(pos - index as f64);
        if weight == 0.0 {
            continue;
        }
        let clamped = index.clamp(0, height as isize - 1) as usize;
        acc += weight * src[clamped * width + x] as f64;
    }
    acc as f32
}

fn lanczos4(distance: f64) -> f64 {
    let distance = distance.abs();
    if distance < 1e-8 {
        return 1.0;
    }
    if distance >= 4.0 {
        return 0.0;
    }
    let pi_d = std::f64::consts::PI * distance;
    (pi_d.sin() / pi_d) * ((pi_d / 4.0).sin() / (pi_d / 4.0))
}

fn least_squares(design: &[f64], rows: usize, cols: usize, values: &[f64]) -> Vec<f64> {
    let mut matrix = design.to_vec();
    let mut target = values.to_vec();
    for col in 0..cols {
        let mut norm_sq = 0.0;
        for row in col..rows {
            let value = matrix[row * cols + col];
            norm_sq += value * value;
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-14 {
            continue;
        }
        let sign = if matrix[col * cols + col] >= 0.0 { 1.0 } else { -1.0 };
        matrix[col * cols + col] += sign * norm;
        let mut reflector = 0.0;
        for row in col..rows {
            let value = matrix[row * cols + col];
            reflector += value * value;
        }
        if reflector < 1e-18 {
            matrix[col * cols + col] = -sign * norm;
            continue;
        }
        let beta = 2.0 / reflector;
        for next in col + 1..cols {
            let mut dot = 0.0;
            for row in col..rows {
                dot += matrix[row * cols + col] * matrix[row * cols + next];
            }
            for row in col..rows {
                matrix[row * cols + next] -= beta * matrix[row * cols + col] * dot;
            }
        }
        let mut dot = 0.0;
        for row in col..rows {
            dot += matrix[row * cols + col] * target[row];
        }
        for row in col..rows {
            target[row] -= beta * matrix[row * cols + col] * dot;
        }
        matrix[col * cols + col] = -sign * norm;
    }
    let mut solution = vec![0.0; cols];
    for col in (0..cols).rev() {
        let diag = matrix[col * cols + col];
        if diag.abs() < 1e-12 {
            continue;
        }
        let mut sum = target[col];
        for next in col + 1..cols {
            sum -= matrix[col * cols + next] * solution[next];
        }
        solution[col] = sum / diag;
    }
    solution
}

fn solve_linear(matrix: &mut [f64], n: usize, values: &mut [f64]) -> Result<(), String> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = matrix[col * n + col].abs();
        for row in col + 1..n {
            let value = matrix[row * n + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return Err("le système RBF du fond est singulier".to_string());
        }
        if pivot != col {
            for index in col..n {
                matrix.swap(col * n + index, pivot * n + index);
            }
            values.swap(col, pivot);
        }
        let diag = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diag;
            for index in col + 1..n {
                matrix[row * n + index] -= factor * matrix[col * n + index];
            }
            values[row] -= factor * values[col];
        }
    }
    for col in (0..n).rev() {
        let mut sum = values[col];
        for index in col + 1..n {
            sum -= matrix[col * n + index] * values[index];
        }
        values[col] = sum / matrix[col * n + col];
    }
    Ok(())
}

fn median_of_copy(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut copy = values.to_vec();
    median_mut(&mut copy)
}

fn median_mut(values: &mut [f32]) -> f64 {
    let count = values.len();
    if count == 0 {
        return 0.0;
    }
    if count % 2 == 1 {
        let mid = count / 2;
        values.select_nth_unstable_by(mid, f32::total_cmp);
        values[mid] as f64
    } else {
        let mid = count / 2;
        values.select_nth_unstable_by(mid, f32::total_cmp);
        let high = values[mid] as f64;
        let low = values[..mid].iter().copied().max_by(f32::total_cmp).unwrap_or(0.0) as f64;
        (low + high) * 0.5
    }
}

fn powi(value: f64, exp: u32) -> f64 {
    let mut result = 1.0;
    for _ in 0..exp {
        result *= value;
    }
    result
}

fn choose(rng: &mut Rng, items: &[(i32, i32)], count: usize) -> Vec<(i32, i32)> {
    let mut pool = items.to_vec();
    let count = count.min(pool.len());
    for index in 0..count {
        let pick = index + rng.index(pool.len() - index);
        pool.swap(index, pick);
    }
    pool.truncate(count);
    pool
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn index(&mut self, count: usize) -> usize {
        if count == 0 {
            0
        } else {
            (self.next() % count as u64) as usize
        }
    }
}

fn unique_samples(points: &[(i32, i32)], values: &[f64]) -> (Vec<(i32, i32)>, Vec<f64>) {
    let mut seen = std::collections::BTreeSet::new();
    let mut kept_points = Vec::new();
    let mut kept_values = Vec::new();
    for (point, value) in points.iter().zip(values) {
        if seen.insert(*point) {
            kept_points.push(*point);
            kept_values.push(*value);
        }
    }
    (kept_points, kept_values)
}

fn check(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) { Err("cancelled".to_string()) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unstretch_inverts_the_internal_stretch() {
        let mut image = vec![0.0f32; 48 * 40];
        for (index, value) in image.iter_mut().enumerate() {
            *value = 0.04 + (index % 17) as f32 * 0.02 + (index % 5) as f32 * 0.01;
        }
        let original = image.clone();
        let pixels = image.len();
        let state = stretch_image(&mut image, 1, pixels);
        unstretch_image(&mut image, 1, pixels, &state);
        for (got, expected) in image.iter().zip(&original) {
            assert!((got - expected).abs() < 2e-5, "{got} != {expected}");
        }
    }

    #[test]
    fn polynomial_fit_recovers_a_known_surface() {
        let mut points = Vec::new();
        let mut values = Vec::new();
        for y in (0..20).step_by(3) {
            for x in (0..24).step_by(3) {
                points.push((x, y));
                let xf = x as f64;
                let yf = y as f64;
                values.push(3.0 + 0.5 * yf + 0.25 * yf * yf + 1.5 * xf + 0.1 * xf * yf - 0.05 * xf * xf);
            }
        }
        let coeff = fit_polynomial(&points, &values, 2).unwrap();
        let expected = [3.0, 0.5, 0.25, 1.5, 0.1, -0.05];
        for (got, want) in coeff.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{coeff:?}");
        }
    }

    #[test]
    fn rbf_with_zero_smooth_passes_through_the_samples() {
        let points = vec![(0, 0), (10, 0), (0, 10), (10, 10), (5, 5), (2, 8)];
        let values = vec![0.2, 0.5, 0.3, 0.8, 0.4, 0.55];
        let nodes = fit_rbf(&points, &values, 0.0).unwrap();
        let cancel = AtomicBool::new(false);
        let model = eval_rbf(&points, &nodes, 11, 11, &cancel).unwrap();
        for ((x, y), value) in points.iter().zip(&values) {
            let got = model[*y as usize * 11 + *x as usize] as f64;
            assert!((got - value).abs() < 1e-3, "{got} != {value}");
        }
    }

    #[test]
    fn synthetic_polynomial_gradient_becomes_flat() {
        let height = 256;
        let width = 256;
        let mut image = vec![0.0f32; height * width];
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32 / (width - 1) as f32;
                let yf = y as f32 / (height - 1) as f32;
                let mut value = 0.28 + 0.22 * xf + 0.10 * yf + 0.06 * xf * xf;
                for cy in [48i32, 128, 208] {
                    for cx in [48i32, 128, 208] {
                        let dx = x as f32 - cx as f32;
                        let dy = y as f32 - cy as f32;
                        if dx * dx + dy * dy < 28.0 * 28.0 {
                            value -= 0.16;
                        }
                    }
                }
                image[y * width + x] = value.clamp(0.02, 0.95);
            }
        }
        let before = column_span(&image, height, width);
        let cancel = AtomicBool::new(false);
        apply(&mut image, 1, height, width, &cancel, |_, _| {}).unwrap();
        let after = column_span(&image, height, width);
        assert!(after < before * 0.45, "pente des colonnes {before} -> {after}");
    }

    fn column_span(image: &[f32], height: usize, width: usize) -> f64 {
        let mut low = f64::MAX;
        let mut high = f64::MIN;
        for x in 0..width {
            let mut sum = 0.0f64;
            for y in 0..height {
                sum += image[y * width + x] as f64;
            }
            let mean = sum / height as f64;
            low = low.min(mean);
            high = high.max(mean);
        }
        high - low
    }
}
