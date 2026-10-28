// SPDX-License-Identifier: GPL-3.0-or-later
//! Pipelines séquence Siril et brutes : analyse, enregistrement, mesure, empilement.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::cache::{self, Analysis, FrameStatsLite, Measures, RegistrationCache, SeqStatsCache};
use super::fits::{self, FitsValue};
use super::lights::{self, Light};
use super::masks;
use super::register;
use super::sequence::{self, Frame};
use super::stack::{self, SampleFrame, SampleSource, StackParams};
use super::stars::{self, FrameMetrics, StarList};
use super::stats::{self, FrameStats};
use crate::gpu::Gpu;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct JobConfig {
    pub seq_path: String,
    pub lights_dir: String,
    pub output_path: String,
    pub include: Option<Vec<i64>>,
    pub include_files: Option<Vec<String>>,
    pub reference_file: String,
    pub rejection: String,
    pub sigma_low: f64,
    pub sigma_high: f64,
    pub normalization: String,
    pub rgb_equal: bool,
    pub weighting: String,
    pub feather: f64,
    pub output_norm: bool,
    pub crop: Option<[i32; 4]>,
    pub rows: Option<[i32; 2]>,
    pub save_coverage: bool,
    pub gpu_mem_gib: f64,
    pub strip: i32,
    pub tile: i32,
    pub overwrite: bool,
    pub drizzle_scale: f64,
    pub pixfrac: f64,
    pub transform: String,
    pub analyze_only: bool,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            seq_path: String::new(),
            lights_dir: String::new(),
            output_path: String::new(),
            include: None,
            include_files: None,
            reference_file: String::new(),
            rejection: "winsorized".to_string(),
            sigma_low: 3.0,
            sigma_high: 3.0,
            normalization: "addscale".to_string(),
            rgb_equal: true,
            weighting: "noise".to_string(),
            feather: 0.0,
            output_norm: true,
            crop: None,
            rows: None,
            save_coverage: false,
            gpu_mem_gib: 0.0,
            strip: 64,
            tile: 1024,
            overwrite: false,
            drizzle_scale: 1.0,
            pixfrac: 1.0,
            transform: "affine".to_string(),
            analyze_only: false,
        }
    }
}

impl JobConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.rejection.as_str(), "winsorized" | "sigma" | "none") {
            return Err("le rejet doit être winsorized, sigma ou none".to_string());
        }
        if !matches!(self.weighting.as_str(), "none" | "noise" | "wfwhm" | "nbstars") {
            return Err("la pondération n'est pas reconnue".to_string());
        }
        if !matches!(self.normalization.as_str(), "addscale" | "none") {
            return Err("la normalisation doit être addscale ou none".to_string());
        }
        if self.rejection != "none" && !(self.sigma_low > 0.0 && self.sigma_high > 0.0) {
            return Err("les sigmas doivent être positifs".to_string());
        }
        if self.feather < 0.0 {
            return Err("le fondu doit être positif ou nul".to_string());
        }
        if self.seq_path.is_empty() && self.lights_dir.is_empty() {
            return Err("indiquez une séquence ou un dossier de brutes".to_string());
        }
        if !(0.05..=1.0).contains(&self.pixfrac) {
            return Err("pixfrac doit être entre 0,05 et 1".to_string());
        }
        if ![1.0, 1.5, 2.0, 3.0].contains(&self.drizzle_scale) {
            return Err("l'échelle de drizzle doit être 1, 1,5, 2 ou 3".to_string());
        }
        if !matches!(self.transform.as_str(), "affine" | "similarity") {
            return Err("l'alignement doit être affine ou similarity".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Log { level: String, message: String },
    Progress { phase: String, done: f64, total: f64, detail: String },
}

#[derive(Clone, Debug)]
pub struct RunOutput {
    pub path: String,
    pub frames: usize,
    pub low: f64,
    pub high: f64,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub elapsed: f64,
    pub preview: Vec<f32>,
    pub preview_w: usize,
    pub preview_h: usize,
}

pub fn run(cfg: &JobConfig, cancel: &AtomicBool, mut emit: impl FnMut(Event)) -> Result<RunOutput, String> {
    let started = std::time::Instant::now();
    cfg.validate()?;
    emit(Event::Progress { phase: "Démarrage".into(), done: 0.0, total: 1.0, detail: "GPU".into() });
    let gpu = Gpu::new().map_err(|err| format!("GPU indisponible : {err}"))?;
    let info = gpu.info()?;
    emit(Event::Log { level: "info".into(), message: format!("GPU {} · {:.1} Gio", info.name, info.total as f64 / 1024.0 / 1024.0 / 1024.0) });
    let limit = if cfg.gpu_mem_gib > 0.0 { (cfg.gpu_mem_gib * 1024.0 * 1024.0 * 1024.0) as usize } else { (info.total as f64 * 0.75) as usize };
    let mut output = if cfg.lights_dir.is_empty() {
        run_sequence(cfg, &gpu, limit, cancel, &mut emit)?
    } else {
        run_lights(cfg, &gpu, limit, cancel, &mut emit)?
    };
    output.elapsed = started.elapsed().as_secs_f64();
    Ok(output)
}

fn run_sequence(cfg: &JobConfig, gpu: &Gpu, limit: usize, cancel: &AtomicBool, emit: &mut impl FnMut(Event)) -> Result<RunOutput, String> {
    let seq = sequence::parse_seq(Path::new(&cfg.seq_path))?;
    let problems = sequence::check_stackable(&seq);
    if !problems.is_empty() {
        return Err(problems.join(" "));
    }
    let mut frames: Vec<Frame> = seq.frames.iter().filter(|frame| frame.has_reg && (cfg.include.as_ref().map(|list| list.contains(&frame.filenum)).unwrap_or(frame.included))).cloned().collect();
    if frames.len() < 2 {
        return Err("sélectionnez au moins deux images enregistrées".to_string());
    }
    emit(Event::Progress { phase: "Mesure".into(), done: 0.0, total: frames.len() as f64, detail: "en-têtes".into() });
    let mut cache = SeqStatsCache::load(&seq.stats_cache_path());
    let need_norm = cfg.normalization == "addscale";
    let need_noise = cfg.weighting == "noise";
    let need_mask = cfg.feather > 0.0;
    let mut stats_map: HashMap<i64, FrameStats> = HashMap::new();
    let mut masks: HashMap<i64, (Vec<f32>, i32, i32)> = HashMap::new();
    for (index, frame) in frames.iter().enumerate() {
        check(cancel)?;
        let path = seq.frame_path(frame);
        let sig = stats::file_signature(&path);
        let mut st = cache.get(frame.filenum, sig).unwrap_or_default();
        if frame.siril.len() >= seq.layers && (0..seq.layers).all(|c| frame.siril.get(c).and_then(|s| s.as_ref()).is_some_and(|s| s.scale.is_some() && s.location.is_some())) {
            st.loc = (0..seq.layers).map(|c| frame.siril[c].as_ref().and_then(|s| s.location)).collect();
            st.scale = (0..seq.layers).map(|c| frame.siril[c].as_ref().and_then(|s| s.scale)).collect();
            st.ngood = (0..seq.layers).map(|c| frame.siril[c].as_ref().and_then(|s| s.ngood)).collect();
            let noise: Vec<Option<f64>> = (0..seq.layers).map(|c| frame.siril[c].as_ref().and_then(|s| s.bgnoise)).collect();
            if noise.iter().all(Option::is_some) {
                st.noise = noise;
            }
        }
        if (need_norm && !st.has_norm(seq.layers)) || (need_noise && !st.has_noise(seq.layers)) || (need_mask && !masks.contains_key(&frame.filenum)) {
            let (image, info) = fits::read_image(&path)?;
            let measured = stats::measure_image(&image, info.channels, info.height, info.width, true, true);
            if need_norm && !st.has_norm(seq.layers) {
                st.loc = measured.loc;
                st.scale = measured.scale;
            }
            if need_noise && !st.has_noise(seq.layers) {
                st.noise = measured.noise;
            }
            st.ngood = measured.ngood;
            let npix = (info.width * info.height) as f64;
            if st.ngood.iter().flatten().copied().min().unwrap_or(0) < (0.10 * npix) as i64 {
                st.bad = "image presque vide".to_string();
            }
            if need_mask && st.bad.is_empty() {
                let siril_path = seq.siril_mask_path(frame);
                if let Some((mask, mw, mh)) = read_ready_mask(&siril_path) {
                    masks.insert(frame.filenum, (mask, mw, mh));
                } else {
                    let plane = if info.channels >= 3 { 1 } else { 0 };
                    let mask = masks::make_mask(&image[plane * info.width * info.height..(plane + 1) * info.width * info.height], info.width, info.height)?;
                    let (mw, mh) = masks::mask_size(info.width, info.height);
                    let mask_path = seq.own_mask_path(frame);
                    masks::write_mask(&mask_path, &mask, mw, mh)?;
                    masks.insert(frame.filenum, (mask, mw as i32, mh as i32));
                }
            }
            cache.put(frame.filenum, sig, &st);
        }
        stats_map.insert(frame.filenum, st);
        if index % 8 == 0 {
            emit(Event::Progress { phase: "Mesure".into(), done: index as f64, total: frames.len() as f64, detail: frame.filenum.to_string() });
        }
    }
    cache.save();
    frames.retain(|frame| stats_map.get(&frame.filenum).map(|st| st.bad.is_empty()).unwrap_or(false));
    if cfg.analyze_only {
        emit(Event::Log { level: "ok".into(), message: format!("{} images mesurées", frames.len()) });
        return Ok(RunOutput { path: String::new(), frames: frames.len(), low: 0.0, high: 0.0, width: seq.frames.first().map(|frame| frame.width).unwrap_or(0), height: seq.frames.first().map(|frame| frame.height).unwrap_or(0), channels: seq.layers, elapsed: 0.0, preview: Vec::new(), preview_w: 0, preview_h: 0 });
    }
    if frames.len() < 2 {
        return Err("moins de deux images utilisables".to_string());
    }
    let canvas = sequence::max_framing(&frames)?;
    let (mut x0, mut y0, mut x1, mut y1) = (0i32, 0, canvas.width as i32, canvas.height as i32);
    if let Some([x, y, w, h]) = cfg.crop {
        x0 = x.max(0);
        y0 = y.max(0);
        x1 = (x + w).min(canvas.width as i32);
        y1 = (y + h).min(canvas.height as i32);
    }
    if let Some([r0, r1]) = cfg.rows {
        y0 = y0.max(r0);
        y1 = y1.min(r1);
    }
    let ref_frame = seq.reference_frame(&frames)?.clone();
    let (a, off, imgw) = coefficients(cfg, &frames, &ref_frame, seq.layers, seq.reg_layer, &stats_map, &frames)?;
    let use_drizzle = seq.drizzle && frames.iter().all(|frame| seq.drizzle_path(frame).is_file());
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let Some((ox, oy)) = canvas.origin_of(frame.filenum) else { continue };
        let (mask, mw, mh) = masks.get(&frame.filenum).cloned().unwrap_or((Vec::new(), 0, 0));
        samples.push(SampleFrame {
            x0: ox,
            y0: oy,
            w: frame.width as i32,
            h: frame.height as i32,
            a: a[index],
            off: off[index],
            imgw: imgw[index],
            mask: if mask.is_empty() { None } else { Some(mask) },
            mask_w: mw,
            mask_h: mh,
            source: SampleSource::Shift { path: seq.frame_path(frame), weight: if use_drizzle { Some(seq.drizzle_path(frame)) } else { None } },
        });
    }
    finish_stack(cfg, gpu, limit, cancel, emit, samples, seq.layers, (x0, y0, x1, y1), &seq.frame_path(&ref_frame), &frames.iter().map(|frame| sequence::read_frame_meta(&seq.frame_path(frame))).collect::<Vec<_>>())
}

fn run_lights(cfg: &JobConfig, gpu: &Gpu, limit: usize, cancel: &AtomicBool, emit: &mut impl FnMut(Event)) -> Result<RunOutput, String> {
    let folder = lights::find_lights_dir(Path::new(&cfg.lights_dir));
    let mut lights = lights::scan_lights(&folder);
    if let Some(names) = &cfg.include_files {
        lights.retain(|light| names.iter().any(|name| name == &light.name()));
    }
    let (width, height, channels, bayer) = lights::common_geometry(&lights);
    if width == 0 {
        return Err("aucune brute FITS utilisable dans ce dossier".to_string());
    }
    lights.retain(|light| light.error.is_empty() && light.width == width && light.height == height && light.channels == channels && light.bayer == bayer);
    if lights.len() < 2 {
        return Err("il faut au moins deux brutes de la même géométrie".to_string());
    }
    let cdir = cache::cache_dir(&folder);
    let mut analysis = Analysis::load(&cdir);
    let mut star_map: HashMap<usize, StarList> = HashMap::new();
    let mut metrics: HashMap<usize, FrameMetrics> = HashMap::new();
    emit(Event::Log { level: "info".into(), message: format!("{} brutes · {width}×{height} {}", lights.len(), if bayer.is_empty() { "sans Bayer".to_string() } else { bayer.clone() }) });
    for (n, light) in lights.iter().enumerate() {
        check(cancel)?;
        if let Some((stars, lite)) = analysis.get(light) {
            metrics.insert(light.index, FrameMetrics { nstars: lite.nstars, fwhm: lite.fwhm, roundness: lite.roundness, background: lite.background, noise: lite.noise });
            star_map.insert(light.index, stars);
        } else {
            let (image, info) = fits::read_image(&light.path)?;
            let (luma, bw, bh) = stars::luminance(&image, info.channels, info.width, info.height, !bayer.is_empty());
            let (stars, metric) = stars::detect_luma(&luma, bw, bh, 0.98);
            analysis.put(light, &stars, &FrameStatsLite { nstars: metric.nstars, fwhm: metric.fwhm, roundness: metric.roundness, background: metric.background, noise: metric.noise });
            star_map.insert(light.index, stars);
            metrics.insert(light.index, metric);
        }
        if n % 4 == 0 {
            emit(Event::Progress { phase: "Étoiles".into(), done: n as f64, total: lights.len() as f64, detail: light.name() });
        }
    }
    analysis.save();
    lights.retain(|light| metrics.get(&light.index).is_some_and(|m| m.nstars >= 15));
    star_map.retain(|index, _| lights.iter().any(|light| light.index == *index));
    if lights.len() < 2 {
        return Err("moins de deux images ont assez d'étoiles (15)".to_string());
    }
    let mut reg = RegistrationCache::load(&cdir);
    let forced = lights.iter().find(|light| !cfg.reference_file.is_empty() && light.name() == cfg.reference_file).map(|light| light.index);
    let cached = if reg.why_invalid(&reg.reference, &cfg.transform, &lights).is_empty() {
        lights.iter().find(|light| light.name() == reg.reference).map(|light| light.index)
    } else {
        None
    };
    let reference = forced.or(cached).or_else(|| register::pick_reference(&metrics)).ok_or_else(|| "aucune image de référence".to_string())?;
    let ref_name = lights.iter().find(|light| light.index == reference).map(|light| light.name()).unwrap_or_default();
    if !reg.why_invalid(&ref_name, &cfg.transform, &lights).is_empty() {
        emit(Event::Progress { phase: "Alignement".into(), done: 0.0, total: lights.len() as f64, detail: ref_name.clone() });
        emit(Event::Log { level: "info".into(), message: "enregistrement des images".into() });
        let result = register::register_all(&star_map, reference, &cfg.transform, &|| cancel.load(Ordering::Relaxed));
        reg.reference = ref_name.clone();
        reg.model = cfg.transform.clone();
        reg.frames.clear();
        reg.failed.clear();
        for light in &lights {
            let sig = lights::file_signature(&light.path);
            if let Some(transform) = result.transforms.get(&light.index) {
                let (inliers, rms, via) = result.info.get(&light.index).cloned().unwrap_or((0, 0.0, String::new()));
                reg.frames.insert(light.name(), serde_json::json!({"sig": sig, "t": transform.to_list(), "inliers": inliers, "rms": rms, "via": via}));
            } else {
                reg.failed.insert(light.name(), serde_json::json!({"sig": sig, "reason": result.failed.get(&light.index).cloned().unwrap_or_else(|| "échec".to_string())}));
            }
        }
        reg.save();
    }
    lights.retain(|light| reg.transform(&light.name()).is_some());
    if lights.len() < 2 {
        return Err("moins de deux images ont pu être alignées".to_string());
    }
    if cfg.analyze_only {
        emit(Event::Log { level: "ok".into(), message: format!("{} images alignées sur {ref_name}", lights.len()) });
        return Ok(RunOutput { path: String::new(), frames: lights.len(), low: 0.0, high: 0.0, width, height, channels: if bayer.is_empty() { channels } else { 3 }, elapsed: 0.0, preview: Vec::new(), preview_w: 0, preview_h: 0 });
    }
    let mut transforms = HashMap::new();
    for light in &lights {
        if let Some(m) = reg.transform(&light.name()) {
            transforms.insert(light.index, m);
        }
    }
    let (canvas_w, canvas_h, placed, boxes) = raw_canvas(&transforms, width, height, cfg.drizzle_scale, cfg.pixfrac);
    let layers = if bayer.is_empty() { channels } else { 3 };
    let lut = lights[0].cfa_lut();
    let mut measures = Measures::load(&cdir);
    let mut stats_map: HashMap<usize, FrameStats> = HashMap::new();
    let mut mask_map: HashMap<usize, (Vec<f32>, i32, i32)> = HashMap::new();
    let npix = (width as f64) * (height as f64) * cfg.drizzle_scale * cfg.drizzle_scale;
    let prepare_total = lights.len() as f64;
    for (index, light) in lights.iter().enumerate() {
        check(cancel)?;
        emit(Event::Progress { phase: "Préparation".into(), done: index as f64, total: prepare_total, detail: light.name() });
        let Some(&(_origin_x, _origin_y, bw, bh)) = boxes.get(&light.index) else { continue };
        let Some(transform) = placed.get(&light.index) else { continue };
        let key = cache::frame_key(transform, bw.max(0) as usize, bh.max(0) as usize, cfg.pixfrac);
        if let Some(st) = measures.get(light, &key) {
            if !st.bad.is_empty() {
                continue;
            }
            stats_map.insert(light.index, st);
            if cfg.feather > 0.0 {
                let path = measures.mask_path(light);
                if let Ok(mask) = masks::read_mask(&path) {
                    let (mw, mh) = masks::mask_size(bw.max(1) as usize, bh.max(1) as usize);
                    mask_map.insert(light.index, (mask, mw as i32, mh as i32));
                }
            }
            continue;
        }
        let warped = warp_box(gpu, light, transform, lut, cfg.pixfrac as f32, bw, bh, layers)?;
        let measured = stats::measure_image(&warped, layers, bh.max(1) as usize, bw.max(1) as usize, true, true);
        let mut st = measured;
        let good = st.ngood.iter().flatten().copied().max().unwrap_or(0);
        if (good as f64) < 0.02 * npix {
            st.bad = "image presque vide après drizzle".to_string();
        } else if !st.has_norm(layers) {
            st.bad = "statistiques inutilisables".to_string();
        } else if !st.has_noise(layers) {
            st.bad = "bruit de fond non mesurable".to_string();
        }
        if st.bad.is_empty() && cfg.feather > 0.0 {
            let plane = if layers == 3 { 1 } else { 0 };
            let pw = bw.max(1) as usize;
            let ph = bh.max(1) as usize;
            let mask = masks::make_mask(&warped[plane * pw * ph..(plane + 1) * pw * ph], pw, ph)?;
            let (mw, mh) = masks::mask_size(pw, ph);
            masks::write_mask(&measures.mask_path(light), &mask, mw, mh)?;
            mask_map.insert(light.index, (mask, mw as i32, mh as i32));
        }
        let mut entry = cache::stats_to_json(&st);
        if let Some(obj) = entry.as_object_mut() {
            obj.insert("sig".into(), serde_json::json!(lights::file_signature(&light.path)));
            obj.insert("key".into(), serde_json::json!(key));
        }
        measures.frames.insert(light.name(), entry);
        if st.bad.is_empty() {
            stats_map.insert(light.index, st);
        }
    }
    measures.save();
    lights.retain(|light| stats_map.contains_key(&light.index));
    if lights.len() < 2 {
        return Err("moins de deux images mesurables".to_string());
    }
    let n_ref = metrics.get(&reference).map(|m| m.nstars.max(1)).unwrap_or(1) as f64;
    let adapted: Vec<(Light, f64, i64)> = lights.iter().cloned().map(|light| {
        let metric = metrics.get(&light.index);
        let fwhm = metric.map(|m| m.fwhm).unwrap_or(1.0);
        let nstars = metric.map(|m| m.nstars).unwrap_or(0);
        let wfwhm = fwhm * (1.0 + 2.0 * (n_ref - nstars as f64) / n_ref);
        (light, wfwhm, nstars)
    }).collect();
    let ref_light = adapted.iter().find(|(light, _, _)| light.index == reference).map(|(light, _, _)| light.clone()).unwrap_or_else(|| adapted[0].0.clone());
    let fake_frames: Vec<Frame> = adapted.iter().enumerate().map(|(index, (light, wfwhm, nstars))| Frame {
        index,
        filenum: light.index as i64,
        included: true,
        width: light.width,
        height: light.height,
        wfwhm: *wfwhm,
        nstars: *nstars,
        has_reg: true,
        pure_translation: true,
        ..empty_frame(index, light.index as i64)
    }).collect();
    let ref_fake = fake_frames.iter().find(|frame| frame.filenum == ref_light.index as i64).cloned().unwrap_or_else(|| fake_frames[0].clone());
    let mut stat_by_file = HashMap::new();
    for frame in &fake_frames {
        if let Some(st) = stats_map.get(&(frame.filenum as usize)) {
            stat_by_file.insert(frame.filenum, st.clone());
        }
    }
    let (a, off, imgw) = coefficients(cfg, &fake_frames, &ref_fake, layers, Some(if layers == 3 { 1 } else { 0 }), &stat_by_file, &fake_frames)?;
    let mut samples = Vec::new();
    for (index, _) in fake_frames.iter().enumerate() {
        let light = &adapted[index].0;
        let Some(&(bx, by, bw, bh)) = boxes.get(&light.index) else { continue };
        let Some(transform) = placed.get(&light.index).copied() else { continue };
        let (mask, mw, mh) = mask_map.get(&light.index).cloned().unwrap_or((Vec::new(), 0, 0));
        samples.push(SampleFrame {
            x0: bx,
            y0: by,
            w: bw,
            h: bh,
            a: a[index],
            off: off[index],
            imgw: imgw[index],
            mask: if mask.is_empty() { None } else { Some(mask) },
            mask_w: mw,
            mask_h: mh,
            source: SampleSource::Raw { light: light.clone(), transform, lut, pixfrac: cfg.pixfrac as f32 },
        });
    }
    let mut x0 = 0i32;
    let mut y0 = 0i32;
    let mut x1 = canvas_w;
    let mut y1 = canvas_h;
    if let Some([x, y, w, h]) = cfg.crop {
        x0 = x.max(0);
        y0 = y.max(0);
        x1 = (x + w).min(canvas_w);
        y1 = (y + h).min(canvas_h);
    }
    if let Some([r0, r1]) = cfg.rows {
        y0 = y0.max(r0);
        y1 = y1.min(r1);
    }
    let metas: Vec<_> = samples.iter().filter_map(|sample| match &sample.source {
        SampleSource::Raw { light, .. } => Some(sequence::read_frame_meta(&light.path)),
        _ => None,
    }).collect();
    let _ = adapted;
    finish_stack(cfg, gpu, limit, cancel, emit, samples, layers, (x0, y0, x1, y1), &ref_light.path, &metas)
}

fn finish_stack(cfg: &JobConfig, gpu: &Gpu, limit: usize, cancel: &AtomicBool, emit: &mut impl FnMut(Event), samples: Vec<SampleFrame>, channels: usize, region: (i32, i32, i32, i32), reference: &Path, metas: &[sequence::FrameMeta]) -> Result<RunOutput, String> {
    if !cfg.output_path.is_empty() && Path::new(&cfg.output_path).exists() && !cfg.overwrite && cfg.rows.is_none() {
        return Err(format!("{} existe déjà", cfg.output_path));
    }
    let (strip, tile) = stack::tune(samples.len(), (region.2 - region.0).max(1) as usize, limit, cfg.strip.max(8) as usize, cfg.tile.max(128) as usize);
    let plain = cfg.feather <= 0.0 && cfg.weighting == "none" && samples.iter().all(|sample| matches!(sample.source, SampleSource::Shift { weight: None, .. }));
    let started = Instant::now();
    let total_rows = (region.3 - region.1).max(1) as f64;
    let stacked = stack::stack_region(gpu, &samples, region, &StackParams {
        channels,
        rejection: match cfg.rejection.as_str() { "sigma" => 1, "none" => 0, _ => 2 },
        sigma_low: cfg.sigma_low as f32,
        sigma_high: cfg.sigma_high as f32,
        feather: cfg.feather as f32,
        plain,
        strip,
        tile,
        want_coverage: cfg.save_coverage,
    }, cancel, |done, total| {
        emit(Event::Progress { phase: "Empilement".into(), done: done as f64, total: total as f64, detail: format!("{done}/{total} bandes") });
    })?;
    let mut data = stacked.data;
    if cfg.output_norm {
        stack::normalize_output(&mut data, stacked.channels, stacked.height, stacked.width);
    } else {
        for value in &mut data {
            *value = value.clamp(0.0, 1.0);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let factor = (stacked.width / 960).max(1);
    let (preview, preview_h, preview_w) = stack::bin_preview(&data, stacked.channels, stacked.height, stacked.width, factor);
    if cfg.rows.is_some() && cfg.output_path.is_empty() {
        emit(Event::Log { level: "ok".into(), message: format!("bande d'essai en {elapsed:.1} s") });
        return Ok(RunOutput { path: String::new(), frames: samples.len(), low: stacked.rejected_low, high: stacked.rejected_high, width: stacked.width, height: stacked.height, channels: stacked.channels, elapsed, preview, preview_w, preview_h });
    }
    let out = if cfg.output_path.is_empty() {
        let folder = reference.parent().unwrap_or(Path::new("."));
        auto_output_path(folder, &reference.file_stem().and_then(|s| s.to_str()).unwrap_or("stack"), samples.len(), metas.iter().map(|m| m.exptime).sum())
    } else {
        PathBuf::from(&cfg.output_path)
    };
    let cards = output_cards(cfg, reference, samples.len(), metas, stacked.width, stacked.height);
    emit(Event::Progress { phase: "Écriture FITS".into(), done: total_rows, total: total_rows, detail: out.display().to_string() });
    fits::write_fits(&out, &data, stacked.channels, stacked.height, stacked.width, &cards)?;
    if let Some(coverage) = &stacked.coverage {
        let cov: Vec<f32> = coverage.iter().map(|v| *v as f32).collect();
        let cov_path = out.with_file_name(format!("{}_coverage.fit", out.file_stem().and_then(|s| s.to_str()).unwrap_or("stack")));
        fits::write_fits(&cov_path, &cov, 1, stacked.height, stacked.width, &[])?;
    }
    emit(Event::Log { level: "ok".into(), message: format!("terminé en {elapsed:.1} s · {}", out.display()) });
    let _ = total_rows;
    Ok(RunOutput { path: out.display().to_string(), frames: samples.len(), low: stacked.rejected_low, high: stacked.rejected_high, width: stacked.width, height: stacked.height, channels: stacked.channels, elapsed, preview, preview_w, preview_h })
}

fn coefficients(cfg: &JobConfig, frames: &[Frame], reference: &Frame, layers: usize, reg_layer: Option<usize>, stats_map: &HashMap<i64, FrameStats>, all: &[Frame]) -> Result<(Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>), String> {
    let n = frames.len();
    let mut a = vec![[1.0; 3]; n];
    let mut off = vec![[0.0; 3]; n];
    let mut w = vec![[1.0; 3]; n];
    let refl = reg_layer.filter(|layer| *layer < layers).unwrap_or(if layers == 3 { 1 } else { 0 });
    if cfg.normalization == "addscale" {
        let rs = stats_map.get(&reference.filenum).ok_or_else(|| "statistiques de la référence manquantes".to_string())?;
        for (i, frame) in frames.iter().enumerate() {
            let st = stats_map.get(&frame.filenum).ok_or_else(|| format!("statistiques manquantes pour {}", frame.filenum))?;
            for c in 0..layers {
                let rl = if cfg.rgb_equal { refl } else { c };
                let scale = st.scale.get(c).copied().flatten().unwrap_or(0.0);
                let loc = st.loc.get(c).copied().flatten().unwrap_or(0.0);
                let scale_ref = rs.scale.get(rl).copied().flatten().unwrap_or(1.0);
                let loc_ref = rs.loc.get(rl).copied().flatten().unwrap_or(0.0);
                a[i][c] = if scale == 0.0 { 1.0 } else { scale_ref / scale };
                off[i][c] = a[i][c] * loc - loc_ref;
            }
        }
    }
    if cfg.weighting == "noise" {
        for (i, frame) in frames.iter().enumerate() {
            let st = stats_map.get(&frame.filenum).ok_or_else(|| "bruit manquant".to_string())?;
            for c in 0..layers {
                let noise = st.noise.get(c).copied().flatten().unwrap_or(0.0);
                w[i][c] = if noise > 0.0 { 1.0 / (a[i][c] * a[i][c] * noise * noise) } else { 0.0 };
            }
        }
    } else if cfg.weighting == "wfwhm" {
        let values: Vec<f64> = all.iter().map(|frame| frame.wfwhm).filter(|v| *v > 0.0).collect();
        if values.is_empty() {
            return Err("la pondération wFWHM a besoin des FWHM d'enregistrement".to_string());
        }
        let fmin = values.iter().copied().fold(f64::MAX, f64::min);
        let fmax = values.iter().copied().fold(0.0, f64::max);
        for (i, frame) in frames.iter().enumerate() {
            let weight = if fmax == fmin { 1.0 } else if frame.wfwhm > 0.0 { (1.0 / (frame.wfwhm * frame.wfwhm) - 1.0 / (fmax * fmax)) / (1.0 / (fmin * fmin) - 1.0 / (fmax * fmax)) } else { 0.0 };
            w[i] = [weight; 3];
        }
    } else if cfg.weighting == "nbstars" {
        let smin = all.iter().map(|frame| frame.nstars).min().unwrap_or(0) as f64;
        let smax = all.iter().map(|frame| frame.nstars).max().unwrap_or(0) as f64;
        for (i, frame) in frames.iter().enumerate() {
            let weight = if smax == smin { 1.0 } else { ((frame.nstars as f64 - smin) / (smax - smin)).powi(2) };
            w[i] = [weight; 3];
        }
    }
    if cfg.weighting != "none" {
        for c in 0..layers {
            let sum: f64 = w.iter().map(|row| row[c]).sum();
            let norm = sum / n as f64;
            if norm == 0.0 {
                return Err(format!("la pondération {} annule toutes les images", cfg.weighting));
            }
            for row in &mut w {
                row[c] /= norm;
            }
        }
    }
    Ok((a, off, w))
}

fn output_cards(cfg: &JobConfig, reference: &Path, frames: usize, metas: &[sequence::FrameMeta], width: usize, height: usize) -> Vec<String> {
    let mut cards = Vec::new();
    if let Ok(header) = fits::read_header(reference) {
        let drop = ["FILENAME", "DATE", "COMMENT", "HISTORY", "CRPIX1", "CRPIX2", "DATE-OBS", "PROGRAM", "STACKCNT", "LIVETIME", "DATAMIN", "DATAMAX", "CHECKSUM", "DATASUM"];
        for card in header.user_cards() {
            let key = card[..8.min(card.len())].trim();
            if !drop.contains(&key) && !key.starts_with("MIPS-") && !key.starts_with("BAYER") {
                cards.push(card);
            }
        }
    }
    let livetime: f64 = metas.iter().map(|meta| if meta.livetime > 0.0 { meta.livetime } else { meta.exptime }).sum();
    if let Some(date) = metas.iter().filter_map(|meta| if meta.date_obs.is_empty() { None } else { Some(meta.date_obs.as_str()) }).min() {
        cards.push(fits::card("DATE-OBS", FitsValue::Str(date.to_string()), "first frame"));
    }
    cards.push(fits::card("STACKCNT", FitsValue::Int(frames as i64), "Stack frames"));
    cards.push(fits::card("LIVETIME", FitsValue::Float(livetime), "[s] Exposure time"));
    cards.push(fits::card("PROGRAM", FitsValue::Str(format!("dubGPUStack {VERSION}")), "created by"));
    let rej = if cfg.rejection == "none" { "none".to_string() } else { format!("{} {} {}", cfg.rejection, cfg.sigma_low, cfg.sigma_high) };
    cards.extend(fits::history(&format!("dubGPUStack {VERSION}: {frames} frames, rejection {rej}")));
    cards.extend(fits::history(&format!("normalization {}, weighting {}, feather {}, {}x{}", cfg.normalization, cfg.weighting, cfg.feather, width, height)));
    cards
}

pub fn auto_output_path(folder: &Path, name: &str, frames: usize, seconds: f64) -> PathBuf {
    let name: String = name.chars().map(|ch| if ch.is_ascii_alphanumeric() || "_-+.".contains(ch) { ch } else { '_' }).collect();
    let name = if name.is_empty() { "stack".to_string() } else { name };
    let stamp = chrono_stamp();
    let base = folder.join(format!("{name}_{frames}frames_{:.1}h_gpu_{stamp}", seconds / 3600.0));
    let mut path = base.with_extension("fit");
    let mut k = 2;
    while path.exists() {
        path = PathBuf::from(format!("{}_{k}.fit", base.display()));
        k += 1;
    }
    path
}

fn chrono_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    if m <= 2 { y += 1; }
    let tod = secs % 86400;
    format!("{y:04}-{m:02}-{d:02}_{:02}{:02}", tod / 3600, (tod % 3600) / 60)
}

fn empty_frame(index: usize, filenum: i64) -> Frame {
    Frame {
        index,
        filenum,
        included: true,
        width: 0,
        height: 0,
        tx: 0.0,
        ty: 0.0,
        pure_translation: true,
        has_reg: false,
        fwhm: f64::NAN,
        wfwhm: f64::NAN,
        roundness: f64::NAN,
        quality: f64::NAN,
        background: f64::NAN,
        nstars: 0,
        siril: Vec::new(),
    }
}

fn warp_box(gpu: &Gpu, light: &Light, transform: &[f64; 6], lut: Option<[i32; 4]>, pixfrac: f32, bw: i32, bh: i32, layers: usize) -> Result<Vec<f32>, String> {
    let bw = bw.max(1) as usize;
    let bh = bh.max(1) as usize;
    let pixels = bw * bh;
    let mut out = vec![0.0f32; layers * pixels];
    let info = fits::ImageInfo::open(&light.path)?;
    for channel in 0..layers {
        let mut num = gpu.zeros(pixels)?;
        let mut den = gpu.zeros(pixels)?;
        let plane = if lut.is_some() { 0 } else { channel.min(info.channels - 1) };
        let image = fits::read_rows(&info, plane, 0, info.height)?;
        gpu.drizzle_into(&image, info.width as i32, info.height as i32, 0, lut, if lut.is_some() { channel as i32 } else { -1 }, *transform, pixfrac as f64 * 0.5, 0, 0, bw as i32, bh as i32, &mut num, &mut den)?;
        let nums = gpu.download(&num)?;
        let dens = gpu.download(&den)?;
        for i in 0..pixels {
            let q = (dens[i].min(1.0) * 255.0 + 0.5).floor();
            if q > 0.0 && dens[i] > 0.0 {
                out[channel * pixels + i] = nums[i] / dens[i];
            }
        }
    }
    Ok(out)
}

pub fn raw_canvas(transforms: &HashMap<usize, [f64; 6]>, width: usize, height: usize, scale: f64, pixfrac: f64) -> (i32, i32, HashMap<usize, [f64; 6]>, HashMap<usize, (i32, i32, i32, i32)>) {
    let mut scaled = HashMap::new();
    for (key, transform) in transforms {
        let mut m = *transform;
        for value in &mut m {
            *value *= scale;
        }
        scaled.insert(*key, m);
    }
    let mut minx = f64::MAX;
    let mut miny = f64::MAX;
    let mut maxx = f64::MIN;
    let mut maxy = f64::MIN;
    for transform in scaled.values() {
        for corner in footprint(transform, width, height, pixfrac) {
            minx = minx.min(corner[0]);
            miny = miny.min(corner[1]);
            maxx = maxx.max(corner[0]);
            maxy = maxy.max(corner[1]);
        }
    }
    if !minx.is_finite() {
        return (0, 0, HashMap::new(), HashMap::new());
    }
    let ox = (minx + 1e-9).floor();
    let oy = (miny + 1e-9).floor();
    let cw = ((maxx - 1e-9).ceil() - ox).max(1.0) as i32;
    let ch = ((maxy - 1e-9).ceil() - oy).max(1.0) as i32;
    let mut placed = HashMap::new();
    let mut boxes = HashMap::new();
    for (key, mut m) in scaled {
        m[2] -= ox;
        m[5] -= oy;
        let corners = footprint(&m, width, height, pixfrac);
        let x0 = (corners.iter().map(|p| p[0]).fold(f64::MAX, f64::min) + 1e-9).floor();
        let y0 = (corners.iter().map(|p| p[1]).fold(f64::MAX, f64::min) + 1e-9).floor();
        let x1 = (corners.iter().map(|p| p[0]).fold(f64::MIN, f64::max) - 1e-9).ceil();
        let y1 = (corners.iter().map(|p| p[1]).fold(f64::MIN, f64::max) - 1e-9).ceil();
        let bx = x0.max(0.0) as i32;
        let by = y0.max(0.0) as i32;
        let bw = (x1.min(cw as f64) as i32 - bx).max(1);
        let bh = (y1.min(ch as f64) as i32 - by).max(1);
        placed.insert(key, m);
        boxes.insert(key, (bx, by, bw, bh));
    }
    (cw, ch, placed, boxes)
}

fn footprint(transform: &[f64; 6], width: usize, height: usize, pixfrac: f64) -> [[f64; 2]; 4] {
    let pad = 0.5 - pixfrac / 2.0;
    let xs = [pad, width as f64 - pad, width as f64 - pad, pad];
    let ys = [pad, pad, height as f64 - pad, height as f64 - pad];
    let mut out = [[0.0; 2]; 4];
    for i in 0..4 {
        out[i][0] = transform[0] * xs[i] + transform[1] * ys[i] + transform[2];
        out[i][1] = transform[3] * xs[i] + transform[4] * ys[i] + transform[5];
    }
    out
}

pub fn coverage_quads(boxes: &HashMap<usize, (i32, i32, i32, i32)>, canvas_w: i32, canvas_h: i32, binning: usize) -> Vec<i32> {
    let binning = binning.max(1) as i32;
    let gw = (canvas_w.max(1) as usize).div_ceil(binning as usize);
    let gh = (canvas_h.max(1) as usize).div_ceil(binning as usize);
    let mut diff = vec![0i32; gh * (gw + 1)];
    for &(x0, y0, w, h) in boxes.values() {
        let xa = (x0.max(0) / binning).clamp(0, gw as i32) as usize;
        let xb = ((x0 + w).max(0) / binning).clamp(0, gw as i32) as usize;
        let ya = (y0.max(0) / binning).clamp(0, gh as i32) as usize;
        let yb = ((y0 + h).max(0) / binning).clamp(0, gh as i32) as usize;
        if xb <= xa || yb <= ya {
            continue;
        }
        for y in ya..yb {
            diff[y * (gw + 1) + xa] += 1;
            diff[y * (gw + 1) + xb] -= 1;
        }
    }
    let mut out = vec![0i32; gh * gw];
    for y in 0..gh {
        let mut run = 0;
        for x in 0..gw {
            run += diff[y * (gw + 1) + x];
            out[y * gw + x] = run;
        }
    }
    out
}

fn read_ready_mask(path: &Path) -> Option<(Vec<f32>, i32, i32)> {
    if !path.is_file() {
        return None;
    }
    let info = fits::ImageInfo::open(path).ok()?;
    let mask = masks::read_mask(path).ok()?;
    Some((mask, info.width as i32, info.height as i32))
}

fn check(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) { Err("cancelled".to_string()) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_a_bad_pixfrac() {
        let mut cfg = JobConfig { lights_dir: "x".into(), pixfrac: 2.0, ..JobConfig::default() };
        assert!(cfg.validate().is_err());
        cfg.pixfrac = 1.0;
        assert!(cfg.validate().is_ok());
    }
}
