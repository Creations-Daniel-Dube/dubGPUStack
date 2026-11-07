// SPDX-License-Identifier: GPL-3.0-or-later
//! Correction d'aberrations par le modèle ONNX d'Aberration Remover 1.1.4
//! (Riccardo Alberghi, GPL-3.0-or-later).
//!
//! Tuiles 512, recouvrement 64, fenêtre de Hann, force 100 %. Le graphe est
//! exécuté par ONNX Runtime : CUDA s'il se charge, sinon le processeur.
//!
//! « Dark ringing reduction » et « Protect background » restent désactivées,
//! comme les cases du script 1.1.4. Les activer recolle le halo d'origine
//! autour du cœur corrigé par le modèle.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

const PATCH: usize = 512;
const OVERLAP: usize = 64;
const MODEL_NAME: &str = "model_v2_5_0.onnx";
const INPUT_NAME: &str = "gen_input_image";
const STRENGTH: f32 = 1.0;
const DARK_RINGING: bool = false;
const PROTECT_BACKGROUND: bool = false;
const MAD_FLOOR: f64 = 1e-8;
const HANN_FLOOR: f32 = 1e-3;

pub struct Report {
    pub tiles: usize,
    pub seconds: f64,
    pub device: String,
}

pub fn require_model() -> Result<PathBuf, String> {
    for candidate in model_candidates() {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err("Fichier model_v2_5_0.onnx introuvable. Placez-le dans le même dossier que dubGPUStack.exe, ou dans siril/.".to_string())
}

pub fn apply(image: &mut [f32], channels: usize, height: usize, width: usize, cancel: &AtomicBool, mut progress: impl FnMut(usize, usize)) -> Result<Report, String> {
    if channels != 1 && channels != 3 {
        return Err("la correction d'aberrations attend une image mono ou couleur".to_string());
    }
    let pixels = height.saturating_mul(width);
    if height == 0 || width == 0 || image.len() < channels.saturating_mul(pixels) {
        return Err("image vide".to_string());
    }
    let started = Instant::now();
    let path = require_model()?;
    let (padded, pad_h, pad_w) = reflect_pad(image, channels, height, width, PATCH, PATCH);
    let env = ort_env()?;
    let (mut session, mut device) = open_session(&env, &path)?;
    let window = hann2d(PATCH);
    let rows = patch_starts(pad_h, PATCH, OVERLAP);
    let cols = patch_starts(pad_w, PATCH, OVERLAP);
    let total = rows.len() * cols.len() * channels;
    let mut accum = vec![0.0f32; channels * pad_h * pad_w];
    // La fenêtre est la même pour chaque couche. Le poids ne doit être compté qu'une fois,
    // sinon une image couleur est divisée par le nombre de couches et les nébuleuses disparaissent.
    let mut weight = vec![0.0f32; pad_h * pad_w];
    let mut done = 0usize;
    progress(0, total);
    for channel in 0..channels {
        let plane = channel * pad_h * pad_w;
        for &y0 in &rows {
            for &x0 in &cols {
                check(cancel)?;
                let patch = extract(&padded, pad_w, plane, y0, x0);
                let mut predicted = match infer_patch(&mut session, &patch) {
                    Ok(values) => values,
                    Err(err) if device == "CUDA" => {
                        session = session_with(&env, &path, false)?;
                        device = "CPU".to_string();
                        infer_patch(&mut session, &patch).map_err(|retry| format!("{err} · repli processeur : {retry}"))?
                    }
                    Err(err) => return Err(err),
                };
                for value in &mut predicted {
                    if !value.is_finite() {
                        *value = 0.0;
                    }
                }
                match_statistics(&mut predicted, &patch);
                for value in &mut predicted {
                    if !value.is_finite() {
                        *value = 0.0;
                    }
                }
                blend_patch(&mut accum[plane..plane + pad_h * pad_w], &mut weight, &predicted, &window, pad_w, y0, x0, channel == 0);
                done += 1;
                progress(done, total);
            }
        }
    }
    for channel in 0..channels {
        let plane = channel * pad_h * pad_w;
        for pixel in 0..pad_h * pad_w {
            let w = if weight[pixel] == 0.0 { 1.0 } else { weight[pixel] };
            accum[plane + pixel] /= w;
        }
    }
    let mut result = crop(&accum, channels, pad_h, pad_w, height, width);
    let original = image[..channels * pixels].to_vec();
    apply_optional_filters(&mut result, &original, channels, height, width, DARK_RINGING, PROTECT_BACKGROUND);
    for (value, source) in result.iter_mut().zip(&original) {
        let mixed = (1.0 - STRENGTH) * *source + STRENGTH * *value;
        *value = if mixed.is_finite() { mixed.clamp(0.0, 1.0) } else { 0.0 };
    }
    image[..channels * pixels].copy_from_slice(&result);
    drop(session);
    Ok(Report { tiles: done, seconds: started.elapsed().as_secs_f64(), device })
}

fn open_session(env: &ort::environment::Environment, path: &Path) -> Result<(Session, String), String> {
    if let Ok(session) = session_with(env, path, true) {
        return Ok((session, "CUDA".to_string()));
    }
    let session = session_with(env, path, false)?;
    Ok((session, "CPU".to_string()))
}

fn ort_env() -> Result<ort::environment::Environment, String> {
    // ONNX Runtime n'accepte qu'un environnement par processus, même après sa destruction.
    // On le conserve pour que la correction suivante réutilise le même.
    static KEPT: std::sync::Mutex<Option<ort::environment::Environment>> = std::sync::Mutex::new(None);
    let mut slot = KEPT.lock().map_err(|_| "ONNX Runtime : verrou indisponible".to_string())?;
    if let Some(env) = slot.as_ref() {
        return Ok(env.clone());
    }
    let dll = require_runtime()?;
    let env = ort::init_from(&dll).map_err(|err| format!("ONNX Runtime : {err}"))?.build().map_err(|err| format!("ONNX Runtime : {err}"))?;
    *slot = Some(env.clone());
    Ok(env)
}

fn runtime_library_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["onnxruntime.dll"]
    } else {
        &["libonnxruntime.so", "libonnxruntime.so.1"]
    }
}

pub(crate) fn require_runtime() -> Result<PathBuf, String> {
    for candidate in runtime_candidates() {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let name = runtime_library_names()[0];
    if cfg!(windows) {
        Err(format!("{name} 1.31 introuvable. Placez-le dans le dossier onnxruntime, ou à côté de dubGPUStack.exe."))
    } else {
        Err(format!("{name} 1.31 introuvable. Placez-le dans le dossier onnxruntime, onnxruntime/lib, ou à côté de l'exécutable."))
    }
}

fn runtime_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_runtime(&mut candidates, &crate::runtime::install_dir());
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in runtime_library_names() {
                candidates.push(dir.join(name));
            }
            push_runtime(&mut candidates, dir);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_runtime(&mut candidates, &cwd);
    }
    let home = crate::runtime::home();
    for name in runtime_library_names() {
        candidates.push(home.join(name));
        candidates.push(home.join("onnxruntime").join(name));
        candidates.push(home.join("onnxruntime").join("lib").join(name));
    }
    candidates
}

fn push_runtime(candidates: &mut Vec<PathBuf>, start: &Path) {
    let mut current = Some(start);
    for _ in 0..8 {
        let Some(dir) = current else { break };
        for name in runtime_library_names() {
            candidates.push(dir.join("onnxruntime").join(name));
            candidates.push(dir.join("onnxruntime").join("lib").join(name));
        }
        current = dir.parent();
    }
}

fn session_with(env: &ort::environment::Environment, path: &Path, cuda: bool) -> Result<Session, String> {
    let mut builder = Session::builder(env).map_err(|err| format!("ONNX Runtime : {err}"))?;
    if cuda {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .map_err(|err| format!("CUDA : {err}"))?;
    }
    builder.commit_from_file(path).map_err(|err| format!("{} : {err}", path.display()))
}

fn infer_patch(session: &mut Session, patch: &[f32]) -> Result<Vec<f32>, String> {
    let tensor = Tensor::from_array(([1usize, 1, PATCH, PATCH], patch.to_vec().into_boxed_slice())).map_err(|err| format!("entrée ONNX : {err}"))?;
    let outputs = session.run(ort::inputs![INPUT_NAME => tensor]).map_err(|err| format!("inférence : {err}"))?;
    let view = outputs[0].try_extract_array::<f32>().map_err(|err| format!("sortie ONNX : {err}"))?;
    let shape = view.shape();
    let (out_h, out_w) = match shape {
        [h, w] => (*h, *w),
        [_, h, w] => (*h, *w),
        [_, _, h, w] => (*h, *w),
        other => return Err(format!("forme de sortie inattendue {other:?}")),
    };
    if out_h != PATCH || out_w != PATCH || view.len() != PATCH * PATCH {
        return Err(format!("la tuile renvoyée est {out_h}×{out_w}"));
    }
    Ok(view.iter().copied().collect())
}

fn match_statistics(output: &mut [f32], input: &[f32]) {
    let input_median = median_of(input);
    let input_mad = mad_of(input, input_median).max(MAD_FLOOR);
    let output_median = median_of(output);
    let output_mad = mad_of(output, output_median).max(MAD_FLOOR);
    for value in output {
        let scaled = (*value as f64 - output_median) / output_mad * input_mad + input_median;
        *value = scaled as f32;
    }
}

fn apply_optional_filters(image: &mut [f32], original: &[f32], channels: usize, height: usize, width: usize, dark_ringing: bool, protect_background: bool) {
    if dark_ringing {
        suppress_dark_rings(image, original, channels, height, width);
    }
    if protect_background {
        protect_background_pixels(image, original, channels, height, width);
    }
}

fn suppress_dark_rings(image: &mut [f32], original: &[f32], channels: usize, height: usize, width: usize) {
    for channel in 0..channels {
        let offset = channel * height * width;
        let plane = &original[offset..offset + height * width];
        let threshold = dark_threshold(plane, height, width);
        for (value, (source, limit)) in image[offset..offset + height * width].iter_mut().zip(plane.iter().zip(&threshold)) {
            if *value < *limit {
                *value = *source;
            }
        }
    }
}

fn dark_threshold(plane: &[f32], height: usize, width: usize) -> Vec<f32> {
    if height < 100 || width < 100 {
        return percentile_filter(plane, height, width, 15, 2.0);
    }
    let (small, sh, sw) = stride_view(plane, height, width, 2);
    let window = (15 / 2).max(3);
    let filtered = percentile_filter(&small, sh, sw, window, 2.0);
    zoom_bilinear(&filtered, sh, sw, height, width)
}

fn protect_background_pixels(image: &mut [f32], original: &[f32], channels: usize, height: usize, width: usize) {
    let pixels = height * width;
    for channel in 0..channels {
        let offset = channel * pixels;
        let plane = &original[offset..offset + pixels];
        let limit = background_threshold(plane, height, width);
        for (value, source) in image[offset..offset + pixels].iter_mut().zip(plane) {
            if *source < limit {
                *value = *source;
            }
        }
    }
}

fn background_threshold(plane: &[f32], height: usize, width: usize) -> f32 {
    let mut samples = Vec::new();
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            samples.push(plane[y * width + x]);
            x += 4;
        }
        y += 4;
    }
    let median = median_of(&samples);
    let mad = mad_of(&samples, median);
    (median + 1.4826 * mad) as f32
}

fn percentile_filter(image: &[f32], height: usize, width: usize, window: usize, percentile: f64) -> Vec<f32> {
    let count = window * window;
    let rank = ((count as f64) * percentile / 100.0) as usize;
    let half = window as isize / 2;
    let mut out = vec![0.0f32; height * width];
    let mut samples = Vec::with_capacity(count);
    for y in 0..height {
        for x in 0..width {
            samples.clear();
            for dy in -half..half + 1 {
                for dx in -half..half + 1 {
                    let yy = reflect_coord(y as isize + dy, height as isize) as usize;
                    let xx = reflect_coord(x as isize + dx, width as isize) as usize;
                    samples.push(image[yy * width + xx]);
                }
            }
            let index = rank.min(samples.len() - 1);
            samples.select_nth_unstable_by(index, f32::total_cmp);
            out[y * width + x] = samples[index];
        }
    }
    out
}

fn stride_view(image: &[f32], height: usize, width: usize, stride: usize) -> (Vec<f32>, usize, usize) {
    let out_h = if height == 0 { 0 } else { (height - 1) / stride + 1 };
    let out_w = if width == 0 { 0 } else { (width - 1) / stride + 1 };
    let mut out = Vec::with_capacity(out_h * out_w);
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            out.push(image[y * width + x]);
        }
    }
    (out, out_h, out_w)
}

fn zoom_bilinear(src: &[f32], src_h: usize, src_w: usize, dst_h: usize, dst_w: usize) -> Vec<f32> {
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return vec![0.0; dst_h * dst_w];
    }
    let mut out = vec![0.0f32; dst_h * dst_w];
    for y in 0..dst_h {
        let sy = (y as f64) * (src_h as f64) / (dst_h as f64);
        let y0 = (sy.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (sy - y0 as f64).clamp(0.0, 1.0) as f32;
        for x in 0..dst_w {
            let sx = (x as f64) * (src_w as f64) / (dst_w as f64);
            let x0 = (sx.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = (sx - x0 as f64).clamp(0.0, 1.0) as f32;
            let top = src[y0 * src_w + x0] + (src[y0 * src_w + x1] - src[y0 * src_w + x0]) * fx;
            let bottom = src[y1 * src_w + x0] + (src[y1 * src_w + x1] - src[y1 * src_w + x0]) * fx;
            out[y * dst_w + x] = top + (bottom - top) * fy;
        }
    }
    out
}

fn patch_starts(length: usize, patch: usize, overlap: usize) -> Vec<usize> {
    if length <= patch {
        return vec![0];
    }
    let stride = patch - overlap;
    let mut indices = Vec::new();
    let mut pos = 0usize;
    loop {
        if pos + patch >= length {
            indices.push(length - patch);
            break;
        }
        indices.push(pos);
        pos += stride;
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn hann2d(size: usize) -> Vec<f32> {
    let mut line = vec![0.0f32; size];
    if size == 1 {
        line[0] = 1.0;
    } else {
        for (index, value) in line.iter_mut().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * index as f64 / (size - 1) as f64;
            *value = (0.5 - 0.5 * phase.cos()) as f32;
        }
    }
    let peak = line.iter().copied().fold(0.0f32, f32::max).max(1e-12);
    let mut window = vec![0.0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            window[y * size + x] = (line[y] * line[x] / peak).max(HANN_FLOOR);
        }
    }
    window
}

fn extract(image: &[f32], width: usize, plane: usize, y0: usize, x0: usize) -> Vec<f32> {
    let mut patch = vec![0.0f32; PATCH * PATCH];
    for y in 0..PATCH {
        let src = plane + (y0 + y) * width + x0;
        patch[y * PATCH..(y + 1) * PATCH].copy_from_slice(&image[src..src + PATCH]);
    }
    patch
}

fn blend_patch(accum: &mut [f32], weight: &mut [f32], patch: &[f32], window: &[f32], width: usize, y0: usize, x0: usize, record_weight: bool) {
    for y in 0..PATCH {
        let dst = (y0 + y) * width + x0;
        let src = y * PATCH;
        for x in 0..PATCH {
            let w = window[src + x];
            accum[dst + x] += patch[src + x] * w;
            if record_weight {
                weight[dst + x] += w;
            }
        }
    }
}

fn reflect_pad(image: &[f32], channels: usize, height: usize, width: usize, min_h: usize, min_w: usize) -> (Vec<f32>, usize, usize) {
    let out_h = height.max(min_h);
    let out_w = width.max(min_w);
    if out_h == height && out_w == width {
        return (image[..channels * height * width].to_vec(), height, width);
    }
    let mut out = vec![0.0f32; channels * out_h * out_w];
    for channel in 0..channels {
        for y in 0..out_h {
            let sy = reflect_coord(y as isize, height as isize) as usize;
            for x in 0..out_w {
                let sx = reflect_coord(x as isize, width as isize) as usize;
                out[channel * out_h * out_w + y * out_w + x] = image[channel * height * width + sy * width + sx];
            }
        }
    }
    (out, out_h, out_w)
}

fn crop(image: &[f32], channels: usize, src_h: usize, src_w: usize, dst_h: usize, dst_w: usize) -> Vec<f32> {
    if src_h == dst_h && src_w == dst_w {
        return image[..channels * dst_h * dst_w].to_vec();
    }
    let mut out = vec![0.0f32; channels * dst_h * dst_w];
    for channel in 0..channels {
        for y in 0..dst_h {
            let src = channel * src_h * src_w + y * src_w;
            let dst = channel * dst_h * dst_w + y * dst_w;
            out[dst..dst + dst_w].copy_from_slice(&image[src..src + dst_w]);
        }
    }
    out
}

fn reflect_coord(index: isize, length: isize) -> isize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length - 1);
    let mut index = index.rem_euclid(period);
    if index >= length {
        index = period - index;
    }
    index
}

fn median_of(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut copy = values.to_vec();
    let count = copy.len();
    if count % 2 == 1 {
        let mid = count / 2;
        copy.select_nth_unstable_by(mid, f32::total_cmp);
        copy[mid] as f64
    } else {
        let mid = count / 2;
        copy.select_nth_unstable_by(mid, f32::total_cmp);
        let high = copy[mid] as f64;
        let low = copy[..mid].iter().copied().max_by(f32::total_cmp).unwrap_or(0.0) as f64;
        (low + high) * 0.5
    }
}

fn mad_of(values: &[f32], median: f64) -> f64 {
    let dev: Vec<f32> = values.iter().map(|value| (*value as f64 - median).abs() as f32).collect();
    median_of(&dev)
}

fn model_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let installed = crate::runtime::install_dir();
    candidates.push(installed.join(MODEL_NAME));
    candidates.push(installed.join("siril").join(MODEL_NAME));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(MODEL_NAME));
            push_project_model(&mut candidates, dir);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_project_model(&mut candidates, &cwd);
    }
    let home = crate::runtime::home();
    candidates.push(home.join(MODEL_NAME));
    candidates.push(home.join("siril").join(MODEL_NAME));
    candidates
}

fn push_project_model(candidates: &mut Vec<PathBuf>, start: &Path) {
    let mut current = Some(start);
    for _ in 0..8 {
        let Some(dir) = current else { break };
        candidates.push(dir.join("siril").join(MODEL_NAME));
        current = dir.parent();
    }
}

fn check(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) { Err("cancelled".to_string()) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_grid_covers_the_end() {
        let starts = patch_starts(1000, 512, 64);
        assert_eq!(*starts.first().unwrap(), 0);
        assert_eq!(*starts.last().unwrap(), 1000 - 512);
        for window in starts.windows(2) {
            assert!(window[1] > window[0]);
            assert!(window[1] - window[0] <= 512 - 64);
        }
    }

    #[test]
    fn statistics_are_copied_onto_the_prediction() {
        let input = vec![0.2f32; 16];
        let mut output = vec![5.0f32; 16];
        output[0] = 9.0;
        match_statistics(&mut output, &input);
        let median = median_of(&output);
        assert!((median - 0.2).abs() < 1e-5, "{median}");
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_second_environment_reuses_the_first() {
        if require_runtime().is_err() {
            return;
        }
        let first = ort_env().expect("premier environnement");
        drop(first);
        ort_env().expect("deuxième environnement");
    }

    fn ring_image() -> Vec<f32> {
        let mut image = vec![0.15f32; 512 * 512];
        for y in 0..512 {
            for x in 0..512 {
                let dx = x as f32 - 256.0;
                let dy = y as f32 - 256.0;
                let radius = (dx * dx + dy * dy).sqrt();
                if (40.0..80.0).contains(&radius) {
                    image[y * 512 + x] = 0.45;
                }
                if radius < 3.0 {
                    image[y * 512 + x] = 0.95;
                }
            }
        }
        image
    }

    fn ring_median(image: &[f32], plane: usize) -> f64 {
        let mut ring = Vec::new();
        for y in 0..512 {
            for x in 0..512 {
                let dx = x as f32 - 256.0;
                let dy = y as f32 - 256.0;
                let radius = (dx * dx + dy * dy).sqrt();
                if (50.0..70.0).contains(&radius) {
                    ring.push(image[plane + y * 512 + x]);
                }
            }
        }
        median_of(&ring)
    }

    #[test]
    fn an_extended_ring_survives_the_correction() {
        if require_model().is_err() {
            return;
        }
        let mut mono = ring_image();
        let cancel = AtomicBool::new(false);
        apply(&mut mono, 1, 512, 512, &cancel, |_, _| {}).expect("mono");
        let mono_ring = ring_median(&mono, 0);
        let mut color = vec![0.0f32; 3 * 512 * 512];
        let source = ring_image();
        for channel in 0..3 {
            color[channel * 512 * 512..(channel + 1) * 512 * 512].copy_from_slice(&source);
        }
        apply(&mut color, 3, 512, 512, &cancel, |_, _| {}).expect("couleur");
        let color_ring = ring_median(&color, 0);
        assert!(mono_ring > 0.30, "anneau mono effacé, médiane {mono_ring}");
        assert!(color_ring > 0.30, "anneau couleur effacé, médiane {color_ring}");
    }

    #[test]
    fn constant_tile_keeps_a_finite_renormalized_shape() {
        let Ok(path) = require_model() else { return };
        let env = ort_env().expect("ONNX Runtime");
        let (mut session, device) = open_session(&env, &path).expect("session");
        let input = vec![0.2f32; PATCH * PATCH];
        let mut output = match infer_patch(&mut session, &input) {
            Ok(values) => values,
            Err(err) if device == "CUDA" => {
                let mut cpu = session_with(&env, &path, false).expect("processeur");
                infer_patch(&mut cpu, &input).unwrap_or_else(|retry| panic!("{err} · {retry}"))
            }
            Err(err) => panic!("{err}"),
        };
        for value in &mut output {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        match_statistics(&mut output, &input);
        assert_eq!(output.len(), PATCH * PATCH);
        assert!(output.iter().all(|value| value.is_finite()));
        let median = median_of(&output);
        assert!((median - 0.2).abs() < 1e-3, "{median}");
    }
}
