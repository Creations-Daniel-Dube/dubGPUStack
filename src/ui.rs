// SPDX-License-Identifier: GPL-3.0-or-later
//! Interface egui : accueil, réglages, canevas, images, résultat et journal.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;

use eframe::egui::{self, Color32, RichText, Stroke};
use egui::{ComboBox, DragValue, Frame, TextureHandle};

use crate::engine::cache::{self, Analysis};
use crate::engine::job::{self, CropDecision, Event, JobConfig, RunOutput};
use crate::engine::lights;
use crate::engine::rotate_crop::{self, Interpolation, Spec};
use crate::engine::sequence;
use crate::engine::stretch;
use crate::gpu;
use crate::icon::app_icon;
use crate::runtime::{self, Update};
use crate::settings::Settings;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab { Canvas, Frames, Result, Log }

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobKind { None, Analyze, Test, Stack }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Metric { Fwhm, Roundness, Stars, Background, Noise }

struct Row {
    key: String,
    filenum: i64,
    label: String,
    included: bool,
    fwhm: f64,
    roundness: f64,
    nstars: i64,
    background: f64,
    noise: f64,
}

struct Loaded {
    lights: bool,
    path: String,
    folder: PathBuf,
    width: usize,
    height: usize,
    channels: usize,
    rows: Vec<Row>,
    reference: String,
    coverage: Vec<i32>,
    grid_w: usize,
    grid_h: usize,
    binning: usize,
    canvas_w: i32,
    canvas_h: i32,
}

enum Note {
    Log(String, String),
    Progress(String, f64, f64, String),
    Finished(Result<RunOutput, String>),
}

struct CropUi {
    full_w: f64,
    full_h: f64,
    preview_w: f32,
    preview_h: f32,
    rect: Option<rotate_crop::Rect>,
    interpolation: Interpolation,
    clamp: bool,
    drag: CropDrag,
}

enum CropDrag {
    None,
    Draw { start: [f64; 2] },
    Edge { edge: usize, origin: rotate_crop::Rect, press: [f64; 2] },
}

pub struct App {
    settings: Settings,
    loaded: Option<Loaded>,
    log: Vec<(String, String)>,
    rx: Option<Receiver<Note>>,
    cancel: Option<Arc<AtomicBool>>,
    busy: bool,
    job: JobKind,
    phase: String,
    fraction: f32,
    tab: Tab,
    metric: Metric,
    linked: bool,
    confirm: bool,
    result_path: String,
    summary: String,
    preview: Option<TextureHandle>,
    coverage_tex: Option<TextureHandle>,
    selected: Option<usize>,
    gpu_line: String,
    output_path: String,
    crop_x: i32,
    crop_y: i32,
    crop_w: i32,
    crop_h: i32,
    crop_rx: Option<Receiver<job::CropPause>>,
    crop_reply: Option<mpsc::Sender<CropDecision>>,
    crop_ui: Option<CropUi>,
    advanced_open: bool,
    open_at_start: Option<String>,
    runtime_rx: Option<Receiver<Update>>,
    runtime_fraction: f32,
    runtime_label: String,
    runtime_error: Option<String>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, open: Option<String>) -> Self {
        install_font(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(theme());
        let mut app = Self {
            settings: Settings::load(),
            loaded: None,
            log: Vec::new(),
            rx: None,
            cancel: None,
            busy: false,
            job: JobKind::None,
            phase: String::new(),
            fraction: 0.0,
            tab: Tab::Canvas,
            metric: Metric::Fwhm,
            linked: true,
            confirm: false,
            result_path: String::new(),
            summary: String::new(),
            preview: None,
            coverage_tex: None,
            selected: None,
            gpu_line: String::new(),
            output_path: String::new(),
            crop_x: 0,
            crop_y: 0,
            crop_w: 0,
            crop_h: 0,
            crop_rx: None,
            crop_reply: None,
            crop_ui: None,
            advanced_open: false,
            open_at_start: open,
            runtime_rx: None,
            runtime_fraction: 0.0,
            runtime_label: "Préparation des fichiers requis…".to_string(),
            runtime_error: None,
        };
        runtime::enable_dll_search();
        if !runtime::ready() {
            app.runtime_rx = Some(runtime::Fetch::spawn().rx);
        }
        match gpu::probe_gpu() {
            Ok(info) => app.gpu_line = info.summary(),
            Err(err) => {
                app.gpu_line = "aucun GPU NVIDIA".to_string();
                app.push("warn", &format!("GPU indisponible : {err}"));
            }
        }
        app
    }

    fn push(&mut self, level: &str, message: &str) {
        self.log.push((level.to_string(), message.to_string()));
        if self.log.len() > 500 {
            self.log.drain(0..self.log.len() - 500);
        }
        if level == "error" || level == "warn" || level == "ok" {
            self.summary = message.to_string();
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if self.open_at_start.take().is_some_and(|path| { self.open_path(Path::new(&path)); false }) {}
        let dropped: Vec<std::path::PathBuf> = ctx.input(|input| input.raw.dropped_files.iter().map(|file| file.path().to_path_buf()).collect());
        for path in dropped {
            self.open_path(&path);
        }
        let Some(rx) = self.rx.take() else { return };
        let mut again = true;
        while let Ok(note) = rx.try_recv() {
            match note {
                Note::Log(level, message) => {
                    if self.busy && level == "info" {
                        self.phase = message.clone();
                    }
                    self.push(&level, &message);
                },
                Note::Progress(phase, done, total, detail) => {
                    self.phase = format!("{phase} · {detail}");
                    self.fraction = if total > 0.0 { (done / total) as f32 } else { 0.0 };
                }
                Note::Finished(result) => {
                    again = false;
                    self.busy = false;
                    self.job = JobKind::None;
                    self.cancel = None;
                    self.crop_ui = None;
                    self.crop_rx = None;
                    self.crop_reply = None;
                    match result {
                        Ok(output) => self.finish_ok(ctx, output),
                        Err(err) if err == "cancelled" => self.push("warn", "annulé"),
                        Err(err) => {
                            self.push("error", &err);
                            self.tab = Tab::Log;
                        }
                    }
                    self.phase.clear();
                    self.fraction = 0.0;
                }
            }
        }
        if again {
            self.rx = Some(rx);
        }
        self.poll_crop(ctx);
        if self.busy {
            ctx.request_repaint();
        }
    }

    fn poll_crop(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.crop_rx.take() else { return };
        match rx.try_recv() {
            Ok(pause) => {
                self.crop_rx = Some(rx);
                self.begin_crop(ctx, pause);
            }
            Err(TryRecvError::Empty) => self.crop_rx = Some(rx),
            Err(TryRecvError::Disconnected) => {}
        }
    }

    fn begin_crop(&mut self, ctx: &egui::Context, pause: job::CropPause) {
        if pause.preview_w > 0 && pause.preview_h > 0 && pause.rgba.len() >= pause.preview_w * pause.preview_h * 4 {
            let image = egui::ColorImage::from_rgba_unmultiplied([pause.preview_w, pause.preview_h], &pause.rgba);
            self.preview = Some(ctx.load_texture("recadrage", image, egui::TextureOptions::LINEAR));
        }
        self.crop_ui = Some(CropUi {
            full_w: pause.width.max(1) as f64,
            full_h: pause.height.max(1) as f64,
            preview_w: pause.preview_w.max(1) as f32,
            preview_h: pause.preview_h.max(1) as f32,
            rect: None,
            interpolation: Interpolation::Lanczos4,
            clamp: true,
            drag: CropDrag::None,
        });
        self.tab = Tab::Result;
        self.phase = "Recadrage · dessinez la zone à garder".into();
        self.fraction = 0.0;
    }

    fn finish_crop(&mut self, decision: CropDecision) {
        let applying = matches!(decision, CropDecision::Apply(_));
        if let Some(tx) = self.crop_reply.take() {
            let _ = tx.send(decision);
        }
        self.crop_ui = None;
        self.phase = if applying { "Recadrage · application".into() } else { "Extraction de fond".into() };
    }

    fn finish_ok(&mut self, ctx: &egui::Context, output: RunOutput) {
        self.apply_analysis();
        self.coverage_tex = None;
        self.refresh_coverage();
        if !output.path.is_empty() {
            self.result_path = output.path.clone();
            self.tab = Tab::Result;
        } else if !output.preview.is_empty() {
            self.tab = Tab::Canvas;
        } else if self.loaded.as_ref().is_some_and(|loaded| !loaded.coverage.is_empty()) {
            self.tab = Tab::Canvas;
        } else {
            self.tab = Tab::Frames;
        }
        self.summary = if !output.path.is_empty() {
            format!("{} images · {}×{} · rejet bas {:.1} % · haut {:.1} % · {:.1} s", output.frames, output.width, output.height, output.low * 100.0, output.high * 100.0, output.elapsed)
        } else if output.preview.is_empty() {
            format!("{} images analysées · {}×{} · {:.1} s", output.frames, output.width, output.height, output.elapsed)
        } else {
            format!("bande test · {} images · {}×{} · {:.1} s", output.frames, output.preview_w, output.preview_h, output.elapsed)
        };
        let summary = self.summary.clone();
        self.push("ok", &summary);
        if !output.preview.is_empty() {
            let rgba = if output.display_ready {
                stretch::unit_rgba(&output.preview, output.channels, output.preview_h, output.preview_w)
            } else {
                let params = stretch::auto_stf(&output.preview, output.channels, output.preview_h, output.preview_w, self.linked);
                stretch::apply_stf(&output.preview, output.channels, output.preview_h, output.preview_w, &params)
            };
            let image = egui::ColorImage::from_rgba_unmultiplied([output.preview_w, output.preview_h], &rgba);
            self.preview = Some(ctx.load_texture("apercu", image, egui::TextureOptions::LINEAR));
        }
    }

    fn open_path(&mut self, path: &Path) {
        let previous = self.loaded.as_ref().map(|loaded| loaded.path.clone());
        let path = if path.is_file() && !path.extension().and_then(|ext| ext.to_str()).unwrap_or("").eq_ignore_ascii_case("seq") {
            path.parent().unwrap_or(path)
        } else {
            path
        };
        let result = if path.is_file() { self.load_seq(path) } else { self.load_lights(path) };
        match result {
            Ok(()) => {
                let current = self.loaded.as_ref().map(|loaded| loaded.path.clone());
                if current.as_deref() != previous.as_deref() {
                    self.output_path.clear();
                    self.confirm = false;
                }
                self.settings.remember(&path.display().to_string());
                self.push("info", &format!("ouvert {}", path.display()));
                let count = self.loaded.as_ref().map(|loaded| loaded.rows.len()).unwrap_or(0);
                self.summary = format!("{count} images chargées");
                self.refresh_coverage();
                self.ask_output_name();
            }
            Err(err) => {
                self.push("error", &err);
                self.tab = Tab::Log;
            }
        }
    }

    fn ask_output_name(&mut self) {
        let dialog = rfd::FileDialog::new()
            .set_title("Nom de sortie")
            .set_directory(self.output_directory())
            .set_file_name(self.suggest_output_file())
            .add_filter("FITS", &["fit", "fits"]);
        if let Some(path) = dialog.save_file() {
            self.output_path = path.display().to_string();
            self.confirm = false;
            self.push("info", &format!("nom de sortie {}", short_path(&self.output_path)));
        }
    }

    fn output_directory(&self) -> PathBuf {
        if !self.output_path.is_empty() {
            if let Some(parent) = Path::new(&self.output_path).parent() {
                if parent.is_dir() {
                    return parent.to_path_buf();
                }
            }
        }
        if let Some(loaded) = &self.loaded {
            if loaded.folder.is_dir() {
                return loaded.folder.clone();
            }
        }
        PathBuf::from(&self.settings.last_dir)
    }

    fn suggest_output_file(&self) -> String {
        if !self.output_path.is_empty() {
            if let Some(name) = Path::new(&self.output_path).file_name().and_then(|name| name.to_str()) {
                return name.to_string();
            }
        }
        let Some(loaded) = &self.loaded else { return "stack.fit".to_string() };
        let stem = if loaded.lights {
            loaded.folder.file_name().and_then(|name| name.to_str()).unwrap_or("stack")
        } else {
            Path::new(&loaded.path).file_stem().and_then(|name| name.to_str()).unwrap_or("stack")
        };
        let stem: String = stem.chars().map(|ch| if ch.is_ascii_alphanumeric() || "_-+.".contains(ch) { ch } else { '_' }).collect();
        let stem = if stem.is_empty() { "stack".to_string() } else { stem };
        format!("{stem}.fit")
    }

    fn output_clash(&self) -> Option<String> {
        if self.output_path.is_empty() {
            return None;
        }
        job::planned_outputs(Path::new(&self.output_path), self.settings.denoise, self.settings.save_coverage)
            .into_iter()
            .find(|path| path.exists())
            .map(|path| path.display().to_string())
    }

    fn apply_analysis(&mut self) {
        let Some(loaded) = self.loaded.as_mut() else { return };
        if !loaded.lights {
            return;
        }
        let dir = cache::cache_dir(&loaded.folder);
        let analysis = Analysis::load(&dir);
        let reg = cache::RegistrationCache::load(&dir);
        if !reg.reference.is_empty() {
            loaded.reference = reg.reference;
        }
        for row in &mut loaded.rows {
            let Some(entry) = analysis.frames.get(&row.key) else { continue };
            if let Some(value) = entry.get("nstars").and_then(|v| v.as_i64()) { row.nstars = value; }
            if let Some(value) = entry.get("fwhm").and_then(|v| v.as_f64()) { row.fwhm = value; }
            if let Some(value) = entry.get("roundness").and_then(|v| v.as_f64()) { row.roundness = value; }
            if let Some(value) = entry.get("background").and_then(|v| v.as_f64()) { row.background = value; }
            if let Some(value) = entry.get("noise").and_then(|v| v.as_f64()) { row.noise = value; }
        }
    }

    fn phase_text(&self) -> String {
        let body = if self.phase.is_empty() { "démarrage".to_string() } else { self.phase.clone() };
        let prefix = match self.job {
            JobKind::Stack => "Empilement",
            JobKind::Analyze => "Analyse",
            JobKind::Test => "Bande test",
            JobKind::None => return body,
        };
        if body.starts_with(prefix) { body } else { format!("{prefix} · {body}") }
    }

    fn cancel_job(&mut self) {
        if let Some(flag) = &self.cancel {
            flag.store(true, Ordering::Relaxed);
        }
    }

    fn load_seq(&mut self, path: &Path) -> Result<(), String> {
        let seq = sequence::parse_seq(path)?;
        let problems = sequence::check_stackable(&seq);
        if !problems.is_empty() {
            return Err(problems.join(" "));
        }
        let override_path = seq.cache_dir().join("selection.json");
        let included = included_override(&override_path);
        let rows = seq.frames.iter().map(|frame| Row {
            key: frame.filenum.to_string(),
            filenum: frame.filenum,
            label: format!("{} {}", seq.name, frame.number()),
            included: included.as_ref().map(|list| list.contains(&frame.filenum)).unwrap_or(frame.included),
            fwhm: frame.fwhm,
            roundness: frame.roundness,
            nstars: frame.nstars,
            background: frame.background,
            noise: frame.background,
        }).collect();
        let (width, height) = seq.frames.first().map(|frame| (frame.width, frame.height)).unwrap_or((0, 0));
        self.loaded = Some(Loaded {
            lights: false,
            path: path.display().to_string(),
            folder: seq.directory(),
            width,
            height,
            channels: seq.layers,
            rows,
            reference: seq.reference.to_string(),
            coverage: Vec::new(),
            grid_w: 0,
            grid_h: 0,
            binning: 1,
            canvas_w: width as i32,
            canvas_h: height as i32,
        });
        let _ = seq;
        Ok(())
    }

    fn load_lights(&mut self, path: &Path) -> Result<(), String> {
        let folder = lights::find_lights_dir(path);
        let found = lights::scan_lights(&folder);
        if found.is_empty() {
            return Err("aucune brute FITS dans ce dossier".to_string());
        }
        let (width, height, channels, _) = lights::common_geometry(&found);
        let (excluded, reference) = cache::load_selection_excluded(&cache::cache_dir(&folder).join("selection.json"));
        let analysis = Analysis::load(&cache::cache_dir(&folder));
        let rows = found.iter().filter(|light| light.error.is_empty()).map(|light| {
            let cached = analysis.get(light);
            Row {
                key: light.name(),
                filenum: light.index as i64,
                label: format!("{}  {}", light.number(), light.name()),
                included: !excluded.iter().any(|name| name == &light.name()),
                fwhm: cached.as_ref().map(|(_, m)| m.fwhm).unwrap_or(f64::NAN),
                roundness: cached.as_ref().map(|(_, m)| m.roundness).unwrap_or(f64::NAN),
                nstars: cached.as_ref().map(|(_, m)| m.nstars).unwrap_or(0),
                background: cached.as_ref().map(|(_, m)| m.background).unwrap_or(f64::NAN),
                noise: cached.as_ref().map(|(_, m)| m.noise).unwrap_or(f64::NAN),
            }
        }).collect();
        self.loaded = Some(Loaded {
            lights: true,
            path: folder.display().to_string(),
            folder: folder.clone(),
            width,
            height,
            channels,
            rows,
            reference,
            coverage: Vec::new(),
            grid_w: 0,
            grid_h: 0,
            binning: 1,
            canvas_w: width as i32,
            canvas_h: height as i32,
        });
        Ok(())
    }

    fn refresh_coverage(&mut self) {
        let Some(loaded) = self.loaded.as_mut() else { return };
        let binning = (loaded.width / 800).max(1);
        if loaded.lights {
            let reg = cache::RegistrationCache::load(&cache::cache_dir(&loaded.folder));
            let mut transforms = std::collections::HashMap::new();
            for row in &loaded.rows {
                if let Some(m) = reg.transform(&row.key) {
                    transforms.insert(row.filenum as usize, m);
                }
            }
            if transforms.len() >= 2 {
                let (cw, ch, _, boxes) = job::raw_canvas(&transforms, loaded.width, loaded.height, self.settings.drizzle_scale, self.settings.pixfrac);
                loaded.canvas_w = cw;
                loaded.canvas_h = ch;
                loaded.coverage = job::coverage_quads(&boxes, cw, ch, binning);
                loaded.grid_w = (cw.max(1) as usize).div_ceil(binning);
                loaded.grid_h = (ch.max(1) as usize).div_ceil(binning);
            }
        } else if let Ok(seq) = sequence::parse_seq(Path::new(&loaded.path)) {
            let frames: Vec<sequence::Frame> = seq.frames.iter().filter(|frame| frame.has_reg && loaded.rows.iter().any(|row| row.filenum == frame.filenum && row.included)).cloned().collect();
            if let Ok(canvas) = sequence::max_framing(&frames) {
                loaded.canvas_w = canvas.width as i32;
                loaded.canvas_h = canvas.height as i32;
                loaded.coverage = sequence::coverage_map(&frames, &canvas, binning);
                loaded.grid_w = canvas.width.div_ceil(binning);
                loaded.grid_h = canvas.height.div_ceil(binning);
            }
        }
        loaded.binning = binning;
        if self.settings.crop {
            self.apply_crop();
        }
    }

    fn apply_crop(&mut self) {
        let Some(loaded) = self.loaded.as_ref() else { return };
        if loaded.coverage.is_empty() || loaded.grid_w == 0 {
            return;
        }
        let vmax = loaded.coverage.iter().copied().max().unwrap_or(0);
        let need = self.settings.min_coverage.min(vmax).max(1);
        let mask: Vec<bool> = loaded.coverage.iter().map(|value| *value >= need).collect();
        let (top, left, height, width) = sequence::largest_rectangle(&mask, loaded.grid_w);
        let bin = loaded.binning as i32;
        self.crop_x = left as i32 * bin;
        self.crop_y = top as i32 * bin;
        self.crop_w = (width as i32 * bin).min(loaded.canvas_w - self.crop_x).max(0);
        self.crop_h = (height as i32 * bin).min(loaded.canvas_h - self.crop_y).max(0);
    }

    fn save_selection(&self) {
        let Some(loaded) = &self.loaded else { return };
        let path = if loaded.lights { cache::cache_dir(&loaded.folder).join("selection.json") } else { PathBuf::from(&loaded.path).parent().unwrap_or(Path::new(".")).join("gpustacker").join("selection.json") };
        if loaded.lights {
            let excluded: Vec<String> = loaded.rows.iter().filter(|row| !row.included).map(|row| row.key.clone()).collect();
            cache::save_selection(&path, &excluded, &loaded.reference);
        } else {
            if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
            let included: Vec<i64> = loaded.rows.iter().filter(|row| row.included).map(|row| row.filenum).collect();
            let _ = std::fs::write(&path, serde_json::json!({"included": included}).to_string());
        }
    }

    fn config(&self, analyze_only: bool, rows: Option<[i32; 2]>) -> Result<JobConfig, String> {
        let loaded = self.loaded.as_ref().ok_or_else(|| "ouvrez d'abord des images".to_string())?;
        let included_files: Vec<String> = loaded.rows.iter().filter(|row| row.included).map(|row| row.key.clone()).collect();
        let included_nums: Vec<i64> = loaded.rows.iter().filter(|row| row.included).map(|row| row.filenum).collect();
        if included_files.len() < 2 && !analyze_only {
            return Err("sélectionnez au moins deux images".to_string());
        }
        let cfg = JobConfig {
            seq_path: if loaded.lights { String::new() } else { loaded.path.clone() },
            lights_dir: if loaded.lights { loaded.path.clone() } else { String::new() },
            output_path: if rows.is_some() || analyze_only { String::new() } else { self.output_path.clone() },
            include: if loaded.lights { None } else { Some(included_nums) },
            include_files: if loaded.lights { Some(included_files) } else { None },
            reference_file: if loaded.lights { loaded.reference.clone() } else { String::new() },
            rejection: self.settings.rejection.clone(),
            sigma_low: self.settings.sigma_low,
            sigma_high: self.settings.sigma_high,
            normalization: self.settings.normalization.clone(),
            rgb_equal: self.settings.rgb_equal,
            weighting: self.settings.weighting.clone(),
            feather: self.settings.feather,
            output_norm: self.settings.output_norm,
            crop: if self.settings.crop && self.crop_w > 0 && self.crop_h > 0 { Some([self.crop_x, self.crop_y, self.crop_w, self.crop_h]) } else { None },
            rows,
            save_coverage: self.settings.save_coverage,
            gpu_mem_gib: self.settings.gpu_mem_gib,
            strip: self.settings.strip,
            tile: self.settings.tile,
            workers: self.settings.workers,
            overwrite: self.confirm,
            drizzle_scale: self.settings.drizzle_scale,
            pixfrac: self.settings.pixfrac,
            transform: self.settings.transform.clone(),
            denoise: self.settings.denoise && rows.is_none() && !analyze_only,
            analyze_only,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn start(&mut self, analyze_only: bool, test_strip: bool) {
        if self.busy {
            return;
        }
        if test_strip {
            self.refresh_coverage();
        }
        let rows = if test_strip { Some(self.test_rows()) } else { None };
        if !analyze_only && !test_strip {
            if !self.output_path.is_empty() {
                let path = self.output_path.clone();
                if let Some(found) = self.output_clash() {
                    if !self.confirm {
                        self.confirm = true;
                        self.push("warn", &format!("{found} existe. Confirmez l'écrasement."));
                        return;
                    }
                }
                if let Some(free) = free_bytes(Path::new(&path).parent().unwrap_or(Path::new("."))) {
                    let loaded = self.loaded.as_ref();
                    let bytes = loaded.map(|item| item.canvas_w as u64 * item.canvas_h as u64 * item.channels.max(1) as u64 * 4).unwrap_or(0);
                    if bytes > 0 && free < bytes.saturating_mul(2) {
                        self.push("error", "espace disque insuffisant pour l'image de sortie");
                        self.tab = Tab::Log;
                        return;
                    }
                }
            }
        }
        let cfg = match self.config(analyze_only, rows) {
            Ok(cfg) => cfg,
            Err(err) => {
                self.push("error", &err);
                self.tab = Tab::Log;
                return;
            }
        };
        self.push("info", &estimate_label(&cfg, self.loaded.as_ref()));
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        self.rx = Some(rx);
        self.cancel = Some(cancel);
        self.job = if analyze_only { JobKind::Analyze } else if test_strip { JobKind::Test } else { JobKind::Stack };
        self.phase = match self.job {
            JobKind::Analyze => "Analyse · démarrage".into(),
            JobKind::Test => "Bande test · démarrage".into(),
            JobKind::Stack => "Empilement · démarrage".into(),
            JobKind::None => String::new(),
        };
        self.busy = true;
        self.confirm = false;
        self.fraction = 0.0;
        let gate = if !analyze_only && !test_strip {
            let (pause_tx, pause_rx) = mpsc::channel();
            let (decision_tx, decision_rx) = mpsc::channel();
            self.crop_rx = Some(pause_rx);
            self.crop_reply = Some(decision_tx);
            self.crop_ui = None;
            Some(job::CropGate { pause_tx, decision_rx })
        } else {
            self.crop_rx = None;
            self.crop_reply = None;
            self.crop_ui = None;
            None
        };
        let tx_job = tx.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                job::run(&cfg, &flag, gate.as_ref(), |event| {
                    let note = match event {
                        Event::Log { level, message } => Note::Log(level, message),
                        Event::Progress { phase, done, total, detail } => Note::Progress(phase, done, total, detail),
                    };
                    let _ = tx_job.send(note);
                })
            }));
            let result = match result {
                Ok(value) => value,
                Err(payload) => Err(panic_text(payload)),
            };
            let _ = tx.send(Note::Finished(result));
        });
    }

    fn test_rows(&self) -> [i32; 2] {
        let Some(loaded) = &self.loaded else { return [0, 192] };
        let mut best_y = loaded.canvas_h / 2;
        if loaded.grid_w > 0 && !loaded.coverage.is_empty() {
            let mut best = -1i32;
            for y in 0..loaded.grid_h {
                let sum: i32 = loaded.coverage[y * loaded.grid_w..(y + 1) * loaded.grid_w].iter().sum();
                if sum > best {
                    best = sum;
                    best_y = (y as i32) * loaded.binning as i32 + loaded.binning as i32 / 2;
                }
            }
        }
        let y0 = (best_y - 96).max(if self.settings.crop { self.crop_y } else { 0 });
        let limit = if self.settings.crop && self.crop_h > 0 { self.crop_y + self.crop_h } else { loaded.canvas_h };
        let y1 = (y0 + 192).min(limit).max(y0 + 1);
        [y0, y1]
    }

    fn exclude_worst(&mut self) {
        let Some(loaded) = self.loaded.as_mut() else { return };
        let mut order: Vec<usize> = (0..loaded.rows.len()).filter(|index| loaded.rows[*index].fwhm.is_finite()).collect();
        order.sort_by(|&a, &b| loaded.rows[b].fwhm.partial_cmp(&loaded.rows[a].fwhm).unwrap_or(std::cmp::Ordering::Equal));
        let drop_n = (order.len() / 10).max(1).min(order.len().saturating_sub(2));
        for index in order.into_iter().take(drop_n) {
            loaded.rows[index].included = false;
        }
    }

    fn ui_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Fichier", |ui| {
                if ui.button("Ouvrir un dossier de brutes…").on_hover_text("Ouvre un dossier d'images brutes.").clicked() {
                    if let Some(path) = rfd::FileDialog::new().set_directory(&self.settings.last_dir).pick_folder() {
                        self.open_path(&path);
                    }
                    ui.close();
                }
                if ui.button("Ouvrir une séquence Siril…").on_hover_text("Ouvre un fichier de séquence Siril (.seq).").clicked() {
                    if let Some(path) = rfd::FileDialog::new().set_directory(&self.settings.last_dir).add_filter("Séquence", &["seq"]).pick_file() {
                        self.open_path(&path);
                    }
                    ui.close();
                }
                ui.menu_button("Récents", |ui| {
                    let recent = self.settings.recent.clone();
                    if recent.is_empty() { ui.label("aucun"); }
                    for path in recent {
                        if ui.button(&path).on_hover_text("Rouvre ce dossier ou cette séquence.").clicked() {
                            self.open_path(Path::new(&path));
                            ui.close();
                        }
                    }
                }).response.on_hover_text("Rouvre un dossier ou une séquence récents.");
                if ui.button("Recharger").on_hover_text("Relit le dossier ou la séquence déjà ouvert.").clicked() {
                    if let Some(path) = self.loaded.as_ref().map(|loaded| loaded.path.clone()) {
                        self.open_path(Path::new(&path));
                    }
                    ui.close();
                }
                if ui.button("Empiler").on_hover_text("Empile les images sélectionnées. Écrit le FITS _postStack, puis _stretch. Le calcul s'arrête pour recadrer et redresser, puis enchaîne _BGE, _AB et, si le débruitage est actif, _DN.").clicked() { self.start(false, false); ui.close(); }
                if ui.button("Quitter").on_hover_text("Ferme dubGPUStack.").clicked() { ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close); }
            }).response.on_hover_text("Ouvrir, recharger ou quitter.");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(&self.gpu_line).small());
            });
        });
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("actions").show(ui, |ui| self.action_bar(ui));
        egui::ScrollArea::vertical().show(ui, |ui| {
            group(ui, "Données", |ui| {
                let title = self.loaded.as_ref().map(|loaded| short_path(&loaded.path)).unwrap_or_else(|| "Aucune image".to_string());
                ui.label(title);
                if let Some(loaded) = &self.loaded {
                    ui.label(format!("{}×{} · {} images · {}", loaded.width, loaded.height, loaded.rows.len(), if loaded.lights { "brutes" } else { "séquence" }));
                }
                ui.horizontal(|ui| {
                    if ui.button("Brutes…").on_hover_text("Ouvre un dossier d'images brutes.").clicked() {
                        if let Some(path) = rfd::FileDialog::new().set_directory(&self.settings.last_dir).pick_folder() { self.open_path(&path); }
                    }
                    if ui.button("Séquence…").on_hover_text("Ouvre un fichier de séquence Siril (.seq).").clicked() {
                        if let Some(path) = rfd::FileDialog::new().set_directory(&self.settings.last_dir).add_filter("Séquence", &["seq"]).pick_file() { self.open_path(&path); }
                    }
                });
            });
            let lights = self.loaded.as_ref().is_some_and(|loaded| loaded.lights);
            if lights {
                group(ui, "Registration et drizzle", |ui| {
                    combo_str(ui, "alignement", &mut self.settings.transform, &[("affine", "Affine"), ("similarity", "Similitude")], "Affine autorise une déformation. Similitude se limite à la translation, la rotation et l'échelle.");
                    combo(ui, "échelle", &mut self.settings.drizzle_scale, &[(1.0, "1×"), (1.5, "1,5×"), (2.0, "2×"), (3.0, "3×")], "Agrandissement drizzle du canevas.");
                    let pixfrac = "Taille de la goutte drizzle. 1 couvre un pixel entier.";
                    ui.label("Pixfrac").on_hover_text(pixfrac);
                    ui.add(DragValue::new(&mut self.settings.pixfrac).range(0.05..=1.0).speed(0.01).fixed_decimals(2)).on_hover_text(pixfrac);
                });
            }
            group(ui, "Sortie", |ui| {
                ui.checkbox(&mut self.settings.crop, "Recadrer par couverture").on_hover_text("Garde la zone couverte par assez d'images.");
                if self.settings.crop {
                    let minimum = "Nombre d'images requis pour garder un pixel dans le cadre.";
                    ui.label("Images minimum").on_hover_text(minimum);
                    if ui.add(DragValue::new(&mut self.settings.min_coverage).range(1..=9999).speed(1.0)).on_hover_text(minimum).changed() { self.apply_crop(); }
                    ui.horizontal(|ui| {
                        ui.add(DragValue::new(&mut self.crop_x).range(0..=200000).prefix("x ")).on_hover_text("Bord gauche du recadrage, en pixels.");
                        ui.add(DragValue::new(&mut self.crop_y).range(0..=200000).prefix("y ")).on_hover_text("Bord haut du recadrage, en pixels.");
                    });
                    ui.horizontal(|ui| {
                        ui.add(DragValue::new(&mut self.crop_w).range(0..=200000).prefix("l ")).on_hover_text("Largeur du recadrage, en pixels.");
                        ui.add(DragValue::new(&mut self.crop_h).range(0..=200000).prefix("h ")).on_hover_text("Hauteur du recadrage, en pixels.");
                    });
                }
                if ui.button("Choisir le fichier de sortie…").on_hover_text("Change le nom de base. Chaque enregistrement ajoute le suffixe de l'étape : _postStack, _stretch, _BGE, _AB, _DN.").clicked() {
                    self.ask_output_name();
                }
                let name_tip = if self.output_path.is_empty() {
                    "Sans nom choisi, un nom automatique est créé à l'empilement. Les fichiers reçoivent _postStack, _stretch, _BGE, _AB et, si le débruitage est actif, _DN."
                } else {
                    "Nom de base, sans suffixe. L'empilement écrit ce nom avec _postStack, _stretch, _BGE, _AB et, si le débruitage est actif, _DN."
                };
                ui.label(if self.output_path.is_empty() { "Nom automatique".to_string() } else { short_path(&self.output_path) }).on_hover_text(name_tip);
                ui.checkbox(&mut self.settings.save_coverage, "Carte de couverture").on_hover_text("Écrit aussi la carte du nombre d'images par pixel, avec le suffixe _coverage.");
                ui.checkbox(&mut self.settings.denoise, "Débruitage dubNoise").on_hover_text("Après le FITS _AB, retire le bruit puis écrit le FITS _DN. Les FITS _postStack, _stretch, _BGE et _AB restent inchangés. Réglages figés : force 90 %, tuile 512, recouvrement 96.");
            });
            advanced_toggle(ui, &mut self.advanced_open);
            if self.advanced_open {
                group(ui, "Empilement", |ui| {
                    combo_str(ui, "rejet", &mut self.settings.rejection, &[("winsorized", "Winsorisé"), ("sigma", "Sigma"), ("none", "Aucun")], "Winsorisé adoucit les écarts, sigma les coupe, aucun garde tout.");
                    let low = "Seuil de rejet des pixels trop sombres.";
                    let high = "Seuil de rejet des pixels trop brillants.";
                    ui.horizontal(|ui| {
                        ui.label("Sigma bas").on_hover_text(low);
                        ui.add(DragValue::new(&mut self.settings.sigma_low).range(0.1..=10.0).speed(0.1).fixed_decimals(1)).on_hover_text(low);
                        ui.label("haut").on_hover_text(high);
                        ui.add(DragValue::new(&mut self.settings.sigma_high).range(0.1..=10.0).speed(0.1).fixed_decimals(1)).on_hover_text(high);
                    });
                    combo_str(ui, "pondération", &mut self.settings.weighting, &[("noise", "Bruit"), ("wfwhm", "wFWHM"), ("nbstars", "Étoiles"), ("none", "Aucune")], "Poids de chaque image : bruit, FWHM, nombre d'étoiles, ou poids égal.");
                    combo_str(ui, "normalisation", &mut self.settings.normalization, &[("addscale", "Additive + échelle"), ("none", "Aucune")], "Met le fond et l'échelle des images au même niveau, ou les laisse tels quels.");
                    ui.checkbox(&mut self.settings.rgb_equal, "Égaliser RVB").on_hover_text("Aligne le fond des trois couches couleur.");
                    let feather = "Lisse le bord des masques, en pixels.";
                    ui.label("Fondu (pixels de masque)").on_hover_text(feather);
                    ui.add(DragValue::new(&mut self.settings.feather).range(0.0..=500.0).speed(1.0).fixed_decimals(0)).on_hover_text(feather);
                    ui.checkbox(&mut self.settings.output_norm, "Normaliser la sortie 0–1").on_hover_text("Ramène le FITS écrit entre 0 et 1.");
                });
                group(ui, "Carte Vidéo / GPU", |ui| {
                    let memory = "Plafond de mémoire GPU, en Gio. 0 utilise 75 % de la carte.";
                    ui.label("Mémoire GPU (0 = 75 %)").on_hover_text(memory);
                    ui.add(DragValue::new(&mut self.settings.gpu_mem_gib).range(0.0..=128.0).speed(0.5).suffix(" Gio")).on_hover_text(memory);
                    let workers = "Nombre de fils qui lisent les images.";
                    ui.label("Fils CPU").on_hover_text(workers);
                    ui.add(DragValue::new(&mut self.settings.workers).range(1..=64)).on_hover_text(workers);
                    let strip = "Hauteur maximale d'une bande, en lignes. 0 laisse le logiciel choisir.";
                    ui.label("Bande max (0 = auto)").on_hover_text(strip);
                    ui.add(DragValue::new(&mut self.settings.strip).range(0..=8192)).on_hover_text(strip);
                    let tile = "Côté maximal d'une tuile, en pixels. 0 laisse le logiciel choisir.";
                    ui.label("Tuile max (0 = auto)").on_hover_text(tile);
                    ui.add(DragValue::new(&mut self.settings.tile).range(0..=16384)).on_hover_text(tile);
                });
            }
        });
    }

    fn action_bar(&mut self, ui: &mut egui::Ui) {
        let analyze = if self.busy && self.job == JobKind::Analyze { "Analyse…" } else { "Analyser les images" };
        let test = if self.busy && self.job == JobKind::Test { "Test…" } else { "Empiler une Bande test" };
        let stack = if self.busy && self.job == JobKind::Stack { "Empilement…" } else { "Empiler les images retenues" };
        let wide = egui::vec2(ui.available_width(), 36.0);
        if ui.add_enabled(!self.busy, egui::Button::new(RichText::new(analyze).size(16.0)).min_size(wide)).on_hover_text("Détecte les étoiles et aligne les images, sans écrire de FITS.").clicked() {
            self.start(true, false);
        }
        if ui.add_enabled(!self.busy, egui::Button::new(RichText::new(test).size(16.0)).min_size(wide)).on_hover_text("Empile une bande de 192 lignes au plus couvert, sans écrire de fichier.").clicked() {
            self.start(false, true);
        }
        if ui.add_enabled(!self.busy, egui::Button::new(RichText::new(stack).size(16.0)).min_size(wide)).on_hover_text("Empile les images sélectionnées. Écrit le FITS _postStack, puis _stretch. Le calcul s'arrête pour recadrer et redresser, puis enchaîne _BGE, _AB et, si le débruitage est actif, _DN.").clicked() {
            self.start(false, false);
        }
        if self.busy {
            ui.add(egui::ProgressBar::new(self.fraction.clamp(0.0, 1.0)).text(self.phase_text()));
            if ui.button("Annuler").on_hover_text("Interrompt le calcul en cours.").clicked() { self.cancel_job(); }
        }
        if self.confirm {
            ui.label("Un fichier de cette série existe déjà.");
            ui.horizontal(|ui| {
                if ui.button("Écraser").on_hover_text("Remplace le fichier de sortie existant.").clicked() { self.start(false, false); }
                if ui.button("Garder").on_hover_text("Annule l'écriture et conserve le fichier existant.").clicked() { self.confirm = false; }
            });
        }
        if !self.busy {
            ui.label(RichText::new(estimate_text(self.loaded.as_ref(), &self.settings)).small());
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        if self.busy {
            ui.label(RichText::new(self.phase_text()).color(Color32::from_rgb(0xff, 0xe1, 0x4d)).strong());
            ui.add(egui::ProgressBar::new(self.fraction.clamp(0.0, 1.0)));
            if ui.button("Annuler").on_hover_text("Interrompt le calcul en cours.").clicked() { self.cancel_job(); }
            ui.separator();
        } else if !self.summary.is_empty() {
            ui.label(RichText::new(&self.summary).color(self.status_color()).strong());
            ui.separator();
        }
        if self.coverage_tex.is_none() {
            if let Some(loaded) = &self.loaded {
                if loaded.grid_w > 0 && loaded.coverage.len() >= loaded.grid_w * loaded.grid_h && loaded.grid_h > 0 {
                    let rgba = stretch::coverage_rgba(&loaded.coverage, loaded.grid_w, loaded.grid_h);
                    let image = egui::ColorImage::from_rgba_unmultiplied([loaded.grid_w, loaded.grid_h], &rgba);
                    self.coverage_tex = Some(ui.ctx().load_texture("couverture", image, egui::TextureOptions::NEAREST));
                }
            }
        }
        if let Some(tex) = &self.preview {
            ui.label("Aperçu");
            show_fitted_image(ui, tex, 0.0);
        } else if let Some(tex) = &self.coverage_tex {
            ui.label("Couverture");
            let reserve = if self.settings.crop && self.crop_w > 0 { text_line(ui) } else { 0.0 };
            show_fitted_image(ui, tex, reserve);
            if self.settings.crop && self.crop_w > 0 {
                ui.label(format!("Recadrage x {} y {}  {}×{}", self.crop_x, self.crop_y, self.crop_w, self.crop_h));
            }
        } else if let Some(count) = self.loaded.as_ref().map(|loaded| loaded.rows.len()) {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.heading(format!("{count} images chargées"));
                ui.label("Analyser mesure les étoiles et aligne. Bande test montre une bande. Empiler écrit le FITS.");
            });
        } else {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.heading("Déposez un dossier de brutes ou un fichier .seq");
                ui.label("Les images déjà mesurées par GPU Stacker restent utilisables.");
            });
        }
    }

    fn status_color(&self) -> Color32 {
        match self.log.last() {
            Some((level, message)) if message == &self.summary && level == "error" => Color32::from_rgb(0xff, 0x6b, 0x4a),
            Some((level, message)) if message == &self.summary && level == "warn" => Color32::from_rgb(0xff, 0x93, 0x20),
            _ => Color32::from_rgb(0xff, 0xe1, 0x4d),
        }
    }

    fn frames(&mut self, ui: &mut egui::Ui) {
        if self.loaded.is_none() { ui.label("Aucune image"); return; }
        let mut changed = false;
        let mut set_ref = false;
        let mut reset = false;
        let mut worst = false;
        let mut include = false;
        let mut exclude = false;
        ui.horizontal(|ui| {
            ComboBox::from_id_salt("metric").selected_text(metric_name(self.metric)).show_ui(ui, |ui| {
                for metric in [Metric::Fwhm, Metric::Roundness, Metric::Stars, Metric::Background, Metric::Noise] {
                    ui.selectable_value(&mut self.metric, metric, format!("{}  —  {}", metric_name(metric), metric_gloss(metric)));
                }
            }).response.on_hover_text("Grandeur tracée pour comparer les images.");
            if ui.button("Inclure").on_hover_text("Garde l'image sélectionnée dans l'empilement.").clicked() { include = true; }
            if ui.button("Exclure").on_hover_text("Retire l'image sélectionnée de l'empilement.").clicked() { exclude = true; }
            if ui.button("Exclure les pires").on_hover_text("Retire environ 10 % des images les moins nettes.").clicked() { worst = true; }
            if ui.button("Réinitialiser").on_hover_text("Coche toutes les images et efface la référence.").clicked() { reset = true; }
            if ui.button("Référence").on_hover_text("Utilise l'image sélectionnée comme référence d'alignement.").clicked() { set_ref = true; }
        });
        let metric = self.metric;
        ui.horizontal(|ui| {
            let (swatch, _) = ui.allocate_exact_size(egui::vec2(22.0, 10.0), egui::Sense::hover());
            ui.painter().line_segment([swatch.left_center(), swatch.right_center()], Stroke::new(2.0, Color32::from_rgb(0xff, 0x93, 0x20)));
            ui.label(RichText::new(metric_legend(metric)).small());
        });
        let selected = self.selected;
        let reference = self.loaded.as_ref().map(|loaded| loaded.reference.clone()).unwrap_or_default();
        let values: Vec<f64> = self.loaded.as_ref().map(|loaded| loaded.rows.iter().map(|row| metric_of(row, metric)).collect()).unwrap_or_default();
        let plot = ui.available_width();
        let plot_h = (ui.available_height() - 150.0).clamp(220.0, 280.0);
        let (rect, response) = ui.allocate_exact_size(egui::vec2(plot, plot_h), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_rgb(0x0e, 0x03, 0x03));
        let inner = egui::Rect::from_min_max(egui::pos2(rect.left() + 72.0, rect.top() + 10.0), egui::pos2(rect.right() - 10.0, rect.bottom() - 22.0));
        let label_color = Color32::from_rgb(0xc8, 0xb4, 0x8a);
        let font = egui::FontId::proportional(12.0);
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() >= 2 {
            let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = (hi - lo).max(1e-9);
            let mid_y = inner.center().y;
            painter.hline(inner.x_range(), mid_y, Stroke::new(1.0, Color32::from_rgb(0x3d, 0x12, 0x10)));
            painter.text(egui::pos2(inner.left() - 6.0, inner.top()), egui::Align2::RIGHT_TOP, metric_value(metric, hi), font.clone(), label_color);
            painter.text(egui::pos2(inner.left() - 6.0, inner.bottom()), egui::Align2::RIGHT_BOTTOM, metric_value(metric, lo), font.clone(), label_color);
            painter.text(egui::pos2(inner.left(), rect.bottom() - 4.0), egui::Align2::LEFT_BOTTOM, "1", font.clone(), label_color);
            painter.text(egui::pos2(inner.right(), rect.bottom() - 4.0), egui::Align2::RIGHT_BOTTOM, values.len().to_string(), font, label_color);
            let step = inner.width() / (values.len().saturating_sub(1).max(1) as f32);
            let mut points = Vec::new();
            for (index, value) in values.iter().enumerate() {
                if !value.is_finite() { continue; }
                let x = inner.left() + index as f32 * step;
                let y = inner.bottom() - ((*value - lo) / span) as f32 * inner.height();
                points.push(egui::pos2(x, y));
            }
            if points.len() >= 2 {
                painter.line(points, Stroke::new(1.5, Color32::from_rgb(0xff, 0x93, 0x20)));
            }
            if let Some(pos) = response.hover_pos() {
                if inner.contains(pos) {
                    let index = ((pos.x - inner.left()) / step).round() as isize;
                    let index = index.clamp(0, values.len() as isize - 1) as usize;
                    if values[index].is_finite() {
                        let x = inner.left() + index as f32 * step;
                        painter.vline(x, inner.y_range(), Stroke::new(1.0, Color32::from_rgb(0xff, 0xe1, 0x4d)));
                        response.on_hover_text(format!("image {} · {}", index + 1, metric_value(metric, values[index])));
                    }
                }
            }
        } else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Analysez les images pour tracer cette courbe.", egui::FontId::proportional(14.0), label_color);
        }
        let mut selected_now = selected;
        let Some(loaded) = self.loaded.as_mut() else { return };
        egui::ScrollArea::vertical().max_height(ui.available_height()).show(ui, |ui| {
            for (index, row) in loaded.rows.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut row.included, "").on_hover_text("Inclut ou exclut cette image de l'empilement.").changed() { changed = true; }
                    let star = if row.key == reference || row.filenum.to_string() == reference { "★ " } else { "" };
                    let text = format!("{star}{}   FWHM {:.2}   rond. {:.2}   ét. {}   fond {:.4}   bruit {:.5}", row.label, row.fwhm, row.roundness, row.nstars, row.background, row.noise);
                    if ui.selectable_label(selected == Some(index), text).on_hover_text("Sélectionne cette image.").clicked() { selected_now = Some(index); }
                });
            }
        });
        self.selected = selected_now;
        if include || exclude {
            if let Some(index) = self.selected {
                if let Some(loaded) = self.loaded.as_mut() { loaded.rows[index].included = include; }
            }
            changed = true;
        }
        if worst { self.exclude_worst(); changed = true; }
        if reset {
            if let Some(loaded) = self.loaded.as_mut() {
                for row in &mut loaded.rows { row.included = true; }
                loaded.reference.clear();
            }
            changed = true;
        }
        if set_ref {
            if let Some(index) = self.selected {
                if let Some(loaded) = self.loaded.as_mut() {
                    loaded.reference = loaded.rows[index].key.clone();
                }
            }
            changed = true;
        }
        if changed { self.save_selection(); self.coverage_tex = None; }
    }
}

impl App {
    fn runtime_gate(&mut self, ui: &mut egui::Ui) -> bool {
        let mut disconnected = false;
        let updates: Vec<Update> = self.runtime_rx.as_ref().map(|rx| {
            let mut items = Vec::new();
            loop {
                match rx.try_recv() {
                    Ok(update) => items.push(update),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            items
        }).unwrap_or_default();
        if disconnected && !updates.iter().any(|update| matches!(update, Update::Finished(_))) {
            self.runtime_error = Some("Téléchargement interrompu.".to_string());
            self.runtime_rx = None;
        }
        for update in updates {
            match update {
                Update::Progress { fraction, label } => {
                    self.runtime_fraction = fraction;
                    self.runtime_label = label;
                }
                Update::Finished(Ok(())) => {
                    runtime::enable_dll_search();
                    self.runtime_rx = None;
                    self.runtime_error = None;
                    return false;
                }
                Update::Finished(Err(err)) => {
                    self.runtime_error = Some(err);
                    self.runtime_rx = None;
                }
            }
        }
        if self.runtime_rx.is_none() && self.runtime_error.is_none() {
            return false;
        }
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
        egui::CentralPanel::default().show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);
                ui.label(RichText::new("Fichiers requis").size(22.0).strong());
                ui.add_space(12.0);
                ui.label(&self.runtime_label);
                ui.add_space(8.0);
                if self.runtime_error.is_none() {
                    ui.add(egui::ProgressBar::new(self.runtime_fraction.clamp(0.0, 1.0)).animate(self.runtime_fraction <= 0.0).desired_width(420.0));
                }
                if let Some(err) = &self.runtime_error {
                    ui.add_space(12.0);
                    ui.colored_label(Color32::from_rgb(0xff, 0x55, 0x44), err);
                }
            });
        });
        true
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.runtime_gate(ui) {
            return;
        }
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        egui::Panel::top("menu").show(ui, |ui| self.ui_bar(ui));
        egui::Panel::left("reglages").resizable(true).exact_size(340.0).show(ui, |ui| self.sidebar(ui));
        egui::Panel::bottom("etat").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Canvas, "Canevas").on_hover_text("Montre la couverture ou l'aperçu.");
                ui.selectable_value(&mut self.tab, Tab::Frames, "Images").on_hover_text("Courbe de qualité et choix des images à empiler.");
                ui.selectable_value(&mut self.tab, Tab::Result, "Résultat").on_hover_text("Aperçu de l'image écrite.");
                ui.selectable_value(&mut self.tab, Tab::Log, "Journal").on_hover_text("Messages du calcul en cours ou terminé.");
                ui.separator();
                if self.busy {
                    ui.add(egui::ProgressBar::new(self.fraction.clamp(0.0, 1.0)).desired_width(160.0));
                    ui.label(RichText::new(self.phase_text()).color(Color32::from_rgb(0xff, 0xe1, 0x4d)));
                } else {
                    ui.label(RichText::new(&self.summary).color(self.status_color()));
                }
            });
        });
        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Canvas => self.canvas(ui),
            Tab::Frames => self.frames(ui),
            Tab::Result => self.result_tab(ui),
            Tab::Log => self.log_tab(ui),
        });
        self.crop_window(&ctx);
        if self.busy {
            ctx.request_repaint();
        } else {
            self.settings.save();
        }
    }
}

fn text_line(ui: &egui::Ui) -> f32 {
    ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y
}

/// La texture est affichée à sa taille en pixels par défaut. Une brute portrait
/// dépasse alors le bas du panneau. On la ramène dans le rectangle encore visible.
fn letterbox(area: egui::Rect, img_w: f32, img_h: f32) -> egui::Rect {
    let img_w = img_w.max(1.0);
    let img_h = img_h.max(1.0);
    let scale = (area.width() / img_w).min(area.height() / img_h);
    let size = egui::vec2(img_w * scale, img_h * scale);
    let origin = egui::pos2(area.left() + (area.width() - size.x) * 0.5, area.top() + (area.height() - size.y) * 0.5);
    egui::Rect::from_min_size(origin, size)
}

fn screen_to_full(pos: egui::Pos2, image: egui::Rect, full_w: f64, full_h: f64) -> [f64; 2] {
    let u = ((pos.x - image.left()) / image.width().max(1.0)) as f64;
    let v = ((pos.y - image.top()) / image.height().max(1.0)) as f64;
    [u * full_w, v * full_h]
}

fn full_to_screen(point: [f64; 2], image: egui::Rect, full_w: f64, full_h: f64) -> egui::Pos2 {
    egui::pos2(
        image.left() + (point[0] / full_w) as f32 * image.width(),
        image.top() + (point[1] / full_h) as f32 * image.height(),
    )
}

fn dist_to_segment(point: egui::Pos2, start: egui::Pos2, end: egui::Pos2) -> f32 {
    let side = end - start;
    let len2 = side.length_sq();
    if len2 <= f32::EPSILON {
        return (point - start).length();
    }
    let t = ((point - start).dot(side) / len2).clamp(0.0, 1.0);
    (point - (start + side * t)).length()
}

fn hit_edge(pos: egui::Pos2, rect: &rotate_crop::Rect, image: egui::Rect, full_w: f64, full_h: f64) -> Option<usize> {
    let pts: [egui::Pos2; 4] = rotate_crop::corners(rect).map(|point| full_to_screen(point, image, full_w, full_h));
    let mut best = None;
    let mut best_d = 8.0f32;
    for edge in 0..4 {
        let distance = dist_to_segment(pos, pts[edge], pts[(edge + 1) % 4]);
        if distance <= best_d {
            best_d = distance;
            best = Some(edge);
        }
    }
    best
}

fn cursor_for_normal(normal: [f64; 2]) -> egui::CursorIcon {
    let ax = normal[0].abs();
    let ay = normal[1].abs();
    if ax > ay * 2.0 {
        egui::CursorIcon::ResizeHorizontal
    } else if ay > ax * 2.0 {
        egui::CursorIcon::ResizeVertical
    } else if normal[0] * normal[1] >= 0.0 {
        egui::CursorIcon::ResizeNwSe
    } else {
        egui::CursorIcon::ResizeNeSw
    }
}

fn show_fitted_image(ui: &mut egui::Ui, tex: &TextureHandle, reserve_below: f32) {
    let width = ui.available_width().max(1.0);
    let height = (ui.clip_rect().bottom() - ui.cursor().top() - reserve_below).max(1.0);
    ui.add(egui::Image::from_texture(tex).shrink_to_fit().max_size(egui::vec2(width, height)));
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        format!("le calcul s'est interrompu : {text}")
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        format!("le calcul s'est interrompu : {text}")
    } else {
        "le calcul s'est interrompu de façon inattendue".to_string()
    }
}

impl App {
    fn result_tab(&mut self, ui: &mut egui::Ui) {
        if self.crop_ui.is_some() {
            ui.label("Tracez un rectangle, faites-le tourner, puis Appliquer.").on_hover_text("Le rectangle rouge est la zone gardée. L'image résultante est redressée, sans bandes noires.");
            self.crop_canvas(ui);
            return;
        }
        ui.checkbox(&mut self.linked, "Étirement lié").on_hover_text("Lie les trois couches pour l'étirement de l'aperçu.");
        if let Some(tex) = &self.preview {
            show_fitted_image(ui, tex, text_line(ui) + ui.spacing().interact_size.y + ui.spacing().item_spacing.y);
        }
        ui.label(if self.result_path.is_empty() { "Pas encore d'image écrite".to_string() } else { self.result_path.clone() });
        ui.horizontal(|ui| {
            if ui.button("Ouvrir").on_hover_text("Ouvre l'image écrite.").clicked() && !self.result_path.is_empty() {
                let _ = std::process::Command::new("explorer").arg(&self.result_path).spawn();
            }
            if ui.button("Dossier").on_hover_text("Montre le fichier dans son dossier.").clicked() && !self.result_path.is_empty() {
                let _ = std::process::Command::new("explorer").args(["/select,", &self.result_path]).spawn();
            }
            if ui.button("Copier le chemin").on_hover_text("Copie le chemin du fichier dans le presse-papiers.").clicked() && !self.result_path.is_empty() {
                ui.ctx().copy_text(self.result_path.clone());
            }
        });
    }

    fn crop_canvas(&mut self, ui: &mut egui::Ui) {
        let tex = self.preview.clone();
        let mut session = match self.crop_ui.take() {
            Some(session) => session,
            None => return,
        };
        let width = ui.available_width().max(1.0);
        let height = (ui.clip_rect().bottom() - ui.cursor().top()).max(1.0);
        let (area, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
        let image_rect = letterbox(area, session.preview_w, session.preview_h);
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                if image_rect.contains(pos) {
                    let full = screen_to_full(pos, image_rect, session.full_w, session.full_h);
                    let edge_hit = session.rect.and_then(|rect| hit_edge(pos, &rect, image_rect, session.full_w, session.full_h).map(|edge| (edge, rect)));
                    if let Some((edge, origin)) = edge_hit {
                        session.drag = CropDrag::Edge { edge, origin, press: full };
                    } else {
                        session.rect = None;
                        session.drag = CropDrag::Draw { start: full };
                    }
                }
            }
        }
        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let full = screen_to_full(pos, image_rect, session.full_w, session.full_h);
                match session.drag {
                    CropDrag::Draw { start } => {
                        let x0 = start[0].clamp(0.0, session.full_w);
                        let y0 = start[1].clamp(0.0, session.full_h);
                        let x1 = full[0].clamp(0.0, session.full_w);
                        let y1 = full[1].clamp(0.0, session.full_h);
                        session.rect = Some(rotate_crop::Rect {
                            center_x: (x0 + x1) * 0.5,
                            center_y: (y0 + y1) * 0.5,
                            width: (x1 - x0).abs(),
                            height: (y1 - y0).abs(),
                            angle_deg: 0.0,
                        });
                    }
                    CropDrag::Edge { edge, origin, press } => {
                        let normal = rotate_crop::outward_normal(&origin, edge);
                        let along = (full[0] - press[0]) * normal[0] + (full[1] - press[1]) * normal[1];
                        let mut moved = origin;
                        rotate_crop::move_edge(&mut moved, edge, along, 8.0);
                        session.rect = Some(moved);
                    }
                    CropDrag::None => {}
                }
            }
        }
        if response.drag_stopped() {
            if session.rect.is_some_and(|rect| rect.width < 2.0 && rect.height < 2.0) {
                session.rect = None;
            }
            session.drag = CropDrag::None;
        }
        if let Some(tex) = &tex {
            ui.painter().image(tex.id(), image_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        if let Some(rect) = session.rect {
            let mut pts: Vec<egui::Pos2> = rotate_crop::corners(&rect).into_iter().map(|point| full_to_screen(point, image_rect, session.full_w, session.full_h)).collect();
            if let Some(first) = pts.first().copied() {
                pts.push(first);
                ui.painter().line(pts, Stroke::new(2.0, Color32::from_rgb(255, 40, 40)));
            }
        }
        if let Some(pos) = response.hover_pos() {
            let cursor = if let Some(rect) = session.rect.as_ref() {
                if let Some(edge) = hit_edge(pos, rect, image_rect, session.full_w, session.full_h) {
                    Some(cursor_for_normal(rotate_crop::outward_normal(rect, edge)))
                } else if image_rect.contains(pos) {
                    Some(egui::CursorIcon::Crosshair)
                } else {
                    None
                }
            } else if image_rect.contains(pos) {
                Some(egui::CursorIcon::Crosshair)
            } else {
                None
            };
            if let Some(cursor) = cursor {
                response.clone().on_hover_cursor(cursor);
            }
        }
        self.crop_ui = Some(session);
    }

    fn crop_window(&mut self, ctx: &egui::Context) {
        let mut session = match self.crop_ui.take() {
            Some(session) => session,
            None => return,
        };
        let mut decision = None;
        egui::Window::new("Rotation et recadrage")
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(380.0, 48.0))
            .show(ctx, |ui| {
                ui.set_min_width(280.0);
                let has_rect = session.rect.is_some();
                let mut angle = session.rect.map(|rect| rect.angle_deg).unwrap_or(0.0);
                ui.horizontal(|ui| {
                    ui.label("Angle (°)");
                    ui.add_enabled(has_rect, DragValue::new(&mut angle).range(-180.0..=180.0).speed(0.05).max_decimals(2))
                        .on_hover_text("Angle du rectangle, en degrés. Une valeur positive tourne dans le sens horaire.");
                    if ui.add_enabled(has_rect, egui::Button::new("−")).on_hover_text("Tourne le rectangle de 0,1° dans le sens antihoraire.").clicked() {
                        angle = (angle - 0.1).clamp(-180.0, 180.0);
                    }
                    if ui.add_enabled(has_rect, egui::Button::new("+")).on_hover_text("Tourne le rectangle de 0,1° dans le sens horaire.").clicked() {
                        angle = (angle + 0.1).clamp(-180.0, 180.0);
                    }
                });
                ui.add_enabled(has_rect, egui::Slider::new(&mut angle, -180.0..=180.0).show_value(false));
                if let Some(rect) = session.rect.as_mut() {
                    rect.angle_deg = angle;
                }
                ComboBox::from_label("Interpolation")
                    .selected_text(session.interpolation.label())
                    .show_ui(ui, |ui| {
                        for mode in Interpolation::ALL {
                            ui.selectable_value(&mut session.interpolation, mode, mode.label()).on_hover_text("Méthode utilisée pour calculer les pixels de l'image redressée.");
                        }
                    });
                ui.checkbox(&mut session.clamp, "Bridage de l'interpolation")
                    .on_hover_text("Ramène chaque pixel interpolé entre le minimum et le maximum des pixels du noyau, pour limiter les anneaux.");
                ui.horizontal(|ui| {
                    if ui.button("Annuler").on_hover_text("Poursuit l'extraction de fond sur l'image entière, sans recadrage.").clicked() {
                        decision = Some(CropDecision::Skip);
                    }
                    let usable = session.rect.as_ref().is_some_and(rotate_crop::is_usable);
                    if ui.add_enabled(usable, egui::Button::new("Appliquer")).on_hover_text("Recadre, redresse l'image, puis lance l'extraction de fond.").clicked() {
                        if let Some(rect) = session.rect {
                            decision = Some(CropDecision::Apply(Spec { rect, interpolation: session.interpolation, clamp: session.clamp }));
                        }
                    }
                });
            });
        if let Some(decision) = decision {
            self.crop_ui = Some(session);
            self.finish_crop(decision);
        } else {
            self.crop_ui = Some(session);
        }
    }

    fn log_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Effacer").on_hover_text("Vide le journal.").clicked() { self.log.clear(); }
            if ui.button("Copier").on_hover_text("Copie le journal dans le presse-papiers.").clicked() {
                let text = self.log.iter().map(|(level, message)| format!("[{level}] {message}")).collect::<Vec<_>>().join("\n");
                ui.ctx().copy_text(text);
            }
            if ui.button("Enregistrer…").on_hover_text("Enregistre le journal dans un fichier texte.").clicked() {
                if let Some(path) = rfd::FileDialog::new().set_file_name("journal.txt").save_file() {
                    let text = self.log.iter().map(|(level, message)| format!("[{level}] {message}")).collect::<Vec<_>>().join("\n");
                    let _ = std::fs::write(path, text);
                }
            }
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (level, message) in &self.log {
                let color = match level.as_str() {
                    "error" => Color32::from_rgb(0xff, 0x55, 0x44),
                    "ok" => Color32::from_rgb(0xff, 0xe1, 0x4d),
                    "warn" => Color32::from_rgb(0xff, 0x93, 0x20),
                    _ => Color32::from_rgb(0xff, 0xe1, 0x4d),
                };
                ui.colored_label(color, format!("[{level}] {message}"));
            }
        });
    }
}

pub fn run(open: Option<String>) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 1080.0]).with_min_inner_size([980.0, 640.0]).with_title(format!("dubGPUStack v{}", job::VERSION)).with_icon(app_icon()),
        centered: true,
        ..Default::default()
    };
    eframe::run_native("dubGPUStack", options, Box::new(move |cc| Ok(Box::new(App::new(cc, open)))))
}

fn theme() -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();
    let label = Color32::from_rgb(0xff, 0xe1, 0x4d);
    let control = Color32::from_rgb(0xff, 0x93, 0x20);
    visuals.window_fill = Color32::from_rgb(0x16, 0x05, 0x05);
    visuals.panel_fill = Color32::from_rgb(0x16, 0x05, 0x05);
    visuals.extreme_bg_color = Color32::from_rgb(0x0e, 0x03, 0x03);
    visuals.faint_bg_color = Color32::from_rgb(0x3d, 0x12, 0x10);
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(0x3d, 0x12, 0x10);
    visuals.widgets.noninteractive.fg_stroke.color = label;
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(0x62, 0x20, 0x18);
    visuals.widgets.inactive.fg_stroke.color = control;
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x7a, 0x28, 0x20);
    visuals.widgets.hovered.fg_stroke.color = control;
    visuals.widgets.active.bg_fill = Color32::from_rgb(0x4a, 0x14, 0x14);
    visuals.widgets.active.fg_stroke.color = control;
    visuals.widgets.open.bg_fill = Color32::from_rgb(0x62, 0x20, 0x18);
    visuals.widgets.open.fg_stroke.color = control;
    visuals.selection.bg_fill = Color32::from_rgb(0xc4, 0x3a, 0x18);
    visuals.selection.stroke.color = control;
    visuals.hyperlink_color = control;
    visuals
}

fn install_font(ctx: &egui::Context) {
    let candidates = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"];
    let Some(bytes) = candidates.iter().find_map(|path| std::fs::read(path).ok()) else { return };
    ctx.add_font(egui::FontInsert::new("stack-bold", egui::FontData::from_owned(bytes), vec![egui::InsertFontFamily { family: egui::FontFamily::Proportional, priority: egui::FontPriority::Highest }]));
}

fn advanced_toggle(ui: &mut egui::Ui, open: &mut bool) {
    ui.add_space(6.0);
    let help = "Affiche ou cache Empilement et Carte Vidéo / GPU.";
    let clicked = ui.horizontal(|ui| {
        let title = ui.add(egui::Label::new(RichText::new("Fonctions avancées").strong()).sense(egui::Sense::click())).on_hover_text(help);
        let icon = if *open { "▲" } else { "▼" };
        let mark = ui.add(egui::Label::new(RichText::new(icon).color(Color32::from_rgb(0xff, 0xe1, 0x4d)).strong()).sense(egui::Sense::click())).on_hover_text(help);
        title.clicked() || mark.clicked()
    }).inner;
    if clicked {
        *open = !*open;
    }
}

fn group(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).strong());
    Frame::NONE.fill(Color32::from_rgb(0x3d, 0x12, 0x10)).stroke(Stroke::new(1.0, Color32::from_rgb(0x8e, 0x32, 0x24))).inner_margin(8.0).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        add(ui);
    });
}

fn short_path(path: &str) -> String {
    let path = Path::new(path);
    path.file_name().and_then(|name| name.to_str()).unwrap_or(path.to_str().unwrap_or("")).to_string()
}

fn combo_str(ui: &mut egui::Ui, salt: &str, value: &mut String, choices: &[(&str, &str)], help: &str) {
    let current = choices.iter().find(|(item, _)| item == value).map(|(_, label)| *label).unwrap_or(value.as_str());
    ComboBox::from_id_salt(salt).selected_text(current).show_ui(ui, |ui| {
        for (item, label) in choices {
            if ui.selectable_label(value == item, *label).clicked() {
                *value = (*item).to_string();
            }
        }
    }).response.on_hover_text(help);
}

fn combo<T: PartialEq + Copy + 'static>(ui: &mut egui::Ui, salt: &str, value: &mut T, choices: &[(T, &str)], help: &str) {
    let current = choices.iter().find(|(item, _)| item == value).map(|(_, label)| *label).unwrap_or("");
    ComboBox::from_id_salt(salt).selected_text(current).show_ui(ui, |ui| {
        for (item, label) in choices {
            ui.selectable_value(value, *item, *label);
        }
    }).response.on_hover_text(help);
}

fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::Fwhm => "FWHM",
        Metric::Roundness => "Rondeur",
        Metric::Stars => "Étoiles",
        Metric::Background => "Fond",
        Metric::Noise => "Bruit",
    }
}

fn metric_gloss(metric: Metric) -> &'static str {
    match metric {
        Metric::Fwhm => "netteté, en pixels",
        Metric::Roundness => "1 = étoile ronde",
        Metric::Stars => "nombre détecté",
        Metric::Background => "niveau du ciel",
        Metric::Noise => "écart-type du ciel",
    }
}

fn metric_legend(metric: Metric) -> &'static str {
    match metric {
        Metric::Fwhm => "Largeur à mi-hauteur des étoiles, en pixels. Plus la courbe monte, moins l'image est nette.",
        Metric::Roundness => "Forme des étoiles : 1 si elles sont rondes, plus bas si elles sont allongées. Une baisse signale souvent un défaut de suivi.",
        Metric::Stars => "Nombre d'étoiles détectées. Une chute vient souvent des nuages, d'une obstruction ou d'un suivi perdu.",
        Metric::Background => "Niveau du fond de ciel. Une montée vient de la Lune, de l'aube, de la pollution lumineuse ou des nuages.",
        Metric::Noise => "Écart-type du fond de ciel. Plus la courbe monte, plus l'image est bruitée.",
    }
}

fn metric_value(metric: Metric, value: f64) -> String {
    match metric {
        Metric::Fwhm => format!("{value:.2} px"),
        Metric::Roundness => format!("{value:.2}"),
        Metric::Stars => format!("{value:.0}"),
        Metric::Background => format!("{value:.4}"),
        Metric::Noise => format!("{value:.5}"),
    }
}

fn metric_of(row: &Row, metric: Metric) -> f64 {
    match metric {
        Metric::Fwhm => row.fwhm,
        Metric::Roundness => row.roundness,
        Metric::Stars => row.nstars as f64,
        Metric::Background => row.background,
        Metric::Noise => row.noise,
    }
}

fn included_override(path: &Path) -> Option<Vec<i64>> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value.get("included")?.as_array().map(|list| list.iter().filter_map(|v| v.as_i64()).collect())
}

fn estimate_text(loaded: Option<&Loaded>, settings: &Settings) -> String {
    let Some(loaded) = loaded else { return "Ouvrez des images pour estimer la durée.".to_string() };
    let n = loaded.rows.iter().filter(|row| row.included).count().max(1) as f64;
    let pixels = loaded.width as f64 * loaded.height as f64 * loaded.channels.max(1) as f64;
    let seconds = if loaded.lights {
        let scale = pixels / (2160.0 * 3840.0);
        n * (1.6 + 2.2 * settings.drizzle_scale) * scale.max(0.2)
    } else {
        let bytes = n * pixels * 2.0;
        bytes / (400.0 * 1024.0 * 1024.0) + bytes / (380.0 * 1024.0 * 1024.0)
    };
    format!("Durée estimée : environ {:.0} min", (seconds / 60.0).max(1.0))
}

fn estimate_label(cfg: &JobConfig, loaded: Option<&Loaded>) -> String {
    let _ = cfg;
    estimate_text(loaded, &Settings { drizzle_scale: cfg.drizzle_scale, ..Settings::default() }).replace("Durée estimée", "Estimation")
}

fn free_bytes(path: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        unsafe extern "system" {
            fn GetDiskFreeSpaceExW(dir: *const u16, free: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
        }
        let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        wide.push(0);
        let mut free = 0u64;
        let mut total = 0u64;
        let mut total_free = 0u64;
        let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, &mut total, &mut total_free) };
        if ok != 0 { Some(free) } else { None }
    }
    #[cfg(not(windows))]
    { let _ = path; None }
}
