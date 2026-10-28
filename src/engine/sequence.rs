// SPDX-License-Identifier: GPL-3.0-or-later
//! Séquences Siril `.seq` et cadrage `-maximize`.

use std::path::{Path, PathBuf};

use super::fits::{self, ImageInfo};

pub const NULL_STAT: f64 = -999_999.0;

#[derive(Clone, Debug)]
pub struct LayerStats {
    pub ngood: Option<i64>,
    /// Médiane d'histogramme lue dans le `.seq`. La normalisation utilise `location`.
    #[allow(dead_code)]
    pub median: Option<f64>,
    /// MAD lue dans le `.seq`. L'échelle utilisée est `scale`.
    #[allow(dead_code)]
    pub mad: Option<f64>,
    pub location: Option<f64>,
    pub scale: Option<f64>,
    pub bgnoise: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub index: usize,
    pub filenum: i64,
    pub included: bool,
    pub width: usize,
    pub height: usize,
    pub tx: f64,
    pub ty: f64,
    pub pure_translation: bool,
    pub has_reg: bool,
    pub fwhm: f64,
    pub wfwhm: f64,
    pub roundness: f64,
    pub quality: f64,
    pub background: f64,
    pub nstars: i64,
    pub siril: Vec<Option<LayerStats>>,
}

impl Frame {
    pub fn number(&self) -> usize {
        self.index + 1
    }
}

#[derive(Clone, Debug, Default)]
pub struct FrameMeta {
    pub exptime: f64,
    pub livetime: f64,
    pub stackcnt: i64,
    pub date_obs: String,
    pub filter: String,
    pub object: String,
    pub bitpix: i32,
    pub error: String,
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub path: PathBuf,
    pub name: String,
    pub start: i64,
    pub nb_images: i64,
    pub nb_selected: i64,
    pub fixed_len: usize,
    pub reference: i64,
    pub version: i64,
    pub variable_size: bool,
    pub fz: bool,
    pub drizzle: bool,
    pub layers: usize,
    pub reg_layer: Option<usize>,
    pub frames: Vec<Frame>,
    pub ext: String,
}

impl Sequence {
    pub fn directory(&self) -> PathBuf {
        self.path.parent().unwrap_or(Path::new(".")).to_path_buf()
    }

    fn number(&self, frame: &Frame) -> String {
        format!("{:0width$}", frame.filenum, width = self.fixed_len.max(1))
    }

    pub fn frame_path(&self, frame: &Frame) -> PathBuf {
        self.directory().join(format!("{}{}{}", self.name, self.number(frame), self.ext))
    }

    pub fn drizzle_path(&self, frame: &Frame) -> PathBuf {
        self.directory().join("drizztmp").join(format!("{}{}.fit", self.name, self.number(frame)))
    }

    pub fn siril_mask_path(&self, frame: &Frame) -> PathBuf {
        self.directory().join("cache").join(format!("{}{}.msk", self.name, self.number(frame)))
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.directory().join("gpustacker")
    }

    pub fn own_mask_path(&self, frame: &Frame) -> PathBuf {
        self.cache_dir().join("masks").join(format!("{}{}.msk", self.name, self.number(frame)))
    }

    pub fn stats_cache_path(&self) -> PathBuf {
        self.cache_dir().join(format!("{}stats.json", self.name))
    }

    pub fn reference_frame<'a>(&'a self, candidates: &'a [Frame]) -> Result<&'a Frame, String> {
        if let Some(frame) = candidates.iter().find(|frame| frame.index as i64 == self.reference) {
            return Ok(frame);
        }
        candidates.first().ok_or_else(|| "aucune image sélectionnée".to_string())
    }
}

pub fn parse_seq(path: &Path) -> Result<Sequence, String> {
    let text = std::fs::read_to_string(path).map_err(|_| format!("fichier de séquence introuvable : {}", path.display()))?;
    let mut head: Option<(String, Vec<i64>)> = None;
    let mut layers = 1usize;
    let mut images: Vec<(i64, bool, Option<usize>, Option<usize>)> = Vec::new();
    let mut regs: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut stats: Vec<(usize, usize, LayerStats)> = Vec::new();
    let mut reg_layer = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let tag = parts.next().unwrap_or("");
        if tag == "S" {
            let Some(start) = line.find('\'') else {
                return Err(format!("{} : ligne S illisible", path.display()));
            };
            let Some(end) = line[start + 1..].find('\'') else {
                return Err(format!("{} : ligne S illisible", path.display()));
            };
            let name = line[start + 1..start + 1 + end].to_string();
            let rest = line[start + 1 + end + 1..].trim();
            let mut nums: Vec<i64> = rest.split_whitespace().filter_map(|token| token.parse::<f64>().ok().map(|v| v as i64)).collect();
            nums.resize(9, 0);
            head = Some((name, nums));
        } else if tag == "L" {
            layers = parts.next().and_then(|v| v.parse().ok()).unwrap_or(1);
        } else if tag == "I" {
            let filenum = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let included = parts.next() == Some("1");
            let size = parts.next().unwrap_or("");
            let (w, h) = if size.contains(',') {
                let mut dims = size.split(',');
                (dims.next().and_then(|v| v.parse().ok()), dims.next().and_then(|v| v.parse().ok()))
            } else {
                (None, None)
            };
            images.push((filenum, included, w, h));
        } else if tag.starts_with('R') && tag[1..].chars().all(|ch| ch.is_ascii_digit()) {
            let layer: usize = tag[1..].parse().unwrap_or(0);
            if reg_layer.is_none() {
                reg_layer = Some(layer);
            }
            if Some(layer) != reg_layer {
                continue;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if let Some(hpos) = tokens.iter().position(|token| *token == "H") {
                let vals = tokens[1..hpos].iter().filter_map(|v| v.parse().ok()).collect();
                let matrix = tokens[hpos + 1..].iter().take(9).filter_map(|v| v.parse().ok()).collect();
                regs.push((vals, matrix));
            } else {
                let vals = tokens[1..].iter().filter_map(|v| v.parse().ok()).collect();
                regs.push((vals, Vec::new()));
            }
        } else if tag.starts_with('M') && tag.contains('-') {
            let body = &tag[1..];
            let Some((layer_s, idx_s)) = body.split_once('-') else { continue };
            let layer: usize = layer_s.parse().unwrap_or(0);
            let idx: usize = idx_s.parse().unwrap_or(0);
            let nums: Vec<f64> = line.split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect();
            if nums.len() >= 14 {
                let val = |index: usize, positive: bool| {
                    let value = nums[index];
                    if value == NULL_STAT || (positive && value <= 0.0) { None } else { Some(value) }
                };
                stats.push((idx, layer, LayerStats {
                    ngood: if nums[1] < 0.0 { None } else { Some(nums[1] as i64) },
                    median: val(3, false),
                    mad: val(6, false),
                    location: val(8, false),
                    scale: val(9, true),
                    bgnoise: val(13, true),
                }));
            }
        }
    }
    let Some((name, nums)) = head else {
        return Err(format!("{} n'est pas une séquence Siril", path.display()));
    };
    let mut seq = Sequence {
        path: path.to_path_buf(),
        name,
        start: nums[0],
        nb_images: nums[1],
        nb_selected: nums[2],
        fixed_len: nums[3] as usize,
        reference: nums[4],
        version: nums[5],
        variable_size: nums[6] != 0,
        fz: nums[7] != 0,
        drizzle: nums[8] != 0,
        layers,
        reg_layer,
        frames: Vec::new(),
        ext: ".fit".to_string(),
    };
    for (index, (filenum, included, w, h)) in images.iter().copied().enumerate() {
        seq.frames.push(Frame {
            index,
            filenum,
            included,
            width: w.unwrap_or(0),
            height: h.unwrap_or(0),
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
            siril: vec![None; layers.max(1)],
        });
    }
    if !regs.is_empty() && regs.len() != seq.frames.len() {
        return Err(format!("{} lignes d'enregistrement pour {} images", regs.len(), seq.frames.len()));
    }
    for (frame, (vals, matrix)) in seq.frames.iter_mut().zip(regs.iter()) {
        if vals.len() >= 6 {
            frame.fwhm = vals[0];
            frame.wfwhm = vals[1];
            frame.roundness = vals[2];
            frame.quality = vals[3];
            frame.background = vals[4];
            frame.nstars = vals[5] as i64;
        }
        if matrix.len() == 9 {
            frame.has_reg = true;
            frame.tx = matrix[2];
            frame.ty = matrix[5];
            frame.pure_translation = (matrix[0] - 1.0).abs() < 1e-9
                && (matrix[4] - 1.0).abs() < 1e-9
                && matrix[1].abs() < 1e-9
                && matrix[3].abs() < 1e-9
                && matrix[6].abs() < 1e-12
                && matrix[7].abs() < 1e-12;
        }
    }
    for (idx, layer, stat) in stats {
        if let Some(frame) = seq.frames.get_mut(idx) {
            if layer >= frame.siril.len() {
                frame.siril.resize(layer + 1, None);
            }
            frame.siril[layer] = Some(stat);
        }
    }
    seq.ext = detect_extension(&seq);
    if seq.frames.first().is_some_and(|frame| frame.width == 0) {
        if let Ok(info) = ImageInfo::open(&seq.frame_path(&seq.frames[0])) {
            for frame in &mut seq.frames {
                frame.width = info.width;
                frame.height = info.height;
            }
        }
    }
    Ok(seq)
}

fn detect_extension(seq: &Sequence) -> String {
    let Some(frame) = seq.frames.first() else { return ".fit".to_string() };
    let base = seq.directory().join(format!("{}{:0width$}", seq.name, frame.filenum, width = seq.fixed_len.max(1)));
    for ext in [".fit", ".fits", ".fts", ".FIT", ".FITS"] {
        if base.with_extension(ext.trim_start_matches('.').to_string()).exists() || path_with(&base, ext).exists() {
            return ext.to_string();
        }
    }
    for ext in [".fit.fz", ".fits.fz", ".fz"] {
        if path_with(&base, ext).exists() {
            return ext.to_string();
        }
    }
    ".fit".to_string()
}

fn path_with(base: &Path, ext: &str) -> PathBuf {
    PathBuf::from(format!("{}{ext}", base.display()))
}

pub fn check_stackable(seq: &Sequence) -> Vec<String> {
    let mut problems = Vec::new();
    if seq.fz || seq.ext.ends_with(".fz") {
        problems.push("Les images sont compressées (.fz). Désactivez la compression FITS dans Siril, puis relancez l'enregistrement.".to_string());
    }
    if seq.layers != 1 && seq.layers != 3 {
        problems.push(format!("{} couches par image : seul le mono ou le RVB est pris en charge.", seq.layers));
    }
    if seq.frames.is_empty() {
        problems.push(format!(
            "Séquence vide : {} images annoncées à partir de {}, version {}, {} sélectionnée(s){}.",
            seq.nb_images,
            seq.start,
            seq.version,
            seq.nb_selected,
            if seq.variable_size { ", taille variable" } else { "" }
        ));
    }
    if !seq.frames.iter().any(|frame| frame.has_reg) {
        problems.push("La séquence n'a pas d'enregistrement. Ouvrez la séquence r_ produite par « Apply registration », ou le dossier de brutes.".to_string());
    } else if seq.frames.iter().any(|frame| frame.has_reg && !frame.pure_translation) {
        problems.push("L'enregistrement n'est pas encore appliqué (il contient une rotation). Ouvrez la séquence r_.".to_string());
    }
    problems
}

pub fn read_frame_meta(path: &Path) -> FrameMeta {
    let mut meta = FrameMeta { stackcnt: 1, ..FrameMeta::default() };
    match fits::read_header(path) {
        Ok(header) => {
            meta.bitpix = header.bitpix().unwrap_or(0);
            meta.exptime = header.get("EXPTIME").and_then(|v| v.as_f64()).or_else(|| header.get("EXPOSURE").and_then(|v| v.as_f64())).unwrap_or(0.0);
            meta.livetime = header.get("LIVETIME").and_then(|v| v.as_f64()).unwrap_or(meta.exptime);
            meta.stackcnt = header.get("STACKCNT").and_then(|v| v.as_int()).filter(|v| *v > 0).unwrap_or(1);
            meta.date_obs = header.get_str("DATE-OBS");
            meta.filter = header.get_str("FILTER");
            meta.object = header.get_str("OBJECT").trim().to_string();
        }
        Err(err) => meta.error = err,
    }
    meta
}

pub fn round_to_int(x: f64) -> i32 {
    if x >= 0.0 { (x + 0.5).floor() as i32 } else { -((-x + 0.5).floor() as i32) }
}

#[derive(Clone, Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub x0: Vec<i32>,
    pub y0: Vec<i32>,
    pub filenums: Vec<i64>,
}

impl Canvas {
    pub fn origin_of(&self, filenum: i64) -> Option<(i32, i32)> {
        self.filenums.iter().position(|value| *value == filenum).map(|index| (self.x0[index], self.y0[index]))
    }
}

pub fn max_framing(frames: &[Frame]) -> Result<Canvas, String> {
    if frames.is_empty() {
        return Err("aucune image sélectionnée".to_string());
    }
    let xmin = frames.iter().map(|frame| frame.tx).fold(f64::INFINITY, f64::min);
    let ymin = frames.iter().map(|frame| frame.ty).fold(f64::INFINITY, f64::min);
    let xmax = frames.iter().map(|frame| frame.tx + frame.width as f64).fold(f64::NEG_INFINITY, f64::max);
    let ymax = frames.iter().map(|frame| frame.ty + frame.height as f64).fold(f64::NEG_INFINITY, f64::max);
    let width = (xmax as i32) - (xmin as i32) + 1;
    let height = (ymax as i32) - (ymin as i32) + 1;
    let off_x = xmin as i32;
    let off_y = -(ymin as i32);
    let mut canvas = Canvas { width: width.max(1) as usize, height: height.max(1) as usize, x0: Vec::new(), y0: Vec::new(), filenums: Vec::new() };
    for frame in frames {
        canvas.x0.push(round_to_int(frame.tx - off_x as f64));
        canvas.y0.push(-round_to_int(-frame.ty - off_y as f64));
        canvas.filenums.push(frame.filenum);
    }
    Ok(canvas)
}

pub fn coverage_map(frames: &[Frame], canvas: &Canvas, binning: usize) -> Vec<i32> {
    let binning = binning.max(1);
    let gh = canvas.height.div_ceil(binning);
    let gw = canvas.width.div_ceil(binning);
    let mut diff = vec![0i32; (gh + 1) * (gw + 1)];
    let stride = gw + 1;
    for frame in frames {
        let Some((x0, y0)) = canvas.origin_of(frame.filenum) else { continue };
        let xa = (x0.max(0) as usize) / binning;
        let ya = (y0.max(0) as usize) / binning;
        let xb = (canvas.width.min((x0 + frame.width as i32).max(0) as usize)) / binning;
        let yb = (canvas.height.min((y0 + frame.height as i32).max(0) as usize)) / binning;
        if xb <= xa || yb <= ya || ya > gh || xa > gw {
            continue;
        }
        let xb = xb.min(gw);
        let yb = yb.min(gh);
        diff[ya * stride + xa] += 1;
        diff[ya * stride + xb] -= 1;
        diff[yb * stride + xa] -= 1;
        diff[yb * stride + xb] += 1;
    }
    for x in 0..=gw {
        let mut acc = 0i32;
        for y in 0..=gh {
            acc += diff[y * stride + x];
            diff[y * stride + x] = acc;
        }
    }
    let mut out = vec![0i32; gh * gw];
    for y in 0..gh {
        let mut acc = 0i32;
        for x in 0..gw {
            acc += diff[y * stride + x];
            out[y * gw + x] = acc;
        }
    }
    out
}

pub fn largest_rectangle(mask: &[bool], cols: usize) -> (usize, usize, usize, usize) {
    if cols == 0 || mask.is_empty() {
        return (0, 0, 0, 0);
    }
    let rows = mask.len() / cols;
    let mut best = (0usize, 0usize, 0usize, 0usize);
    let mut heights = vec![0i64; cols + 1];
    for r in 0..rows {
        for c in 0..cols {
            heights[c] = if mask[r * cols + c] { heights[c] + 1 } else { 0 };
        }
        let mut stack: Vec<(usize, i64)> = Vec::new();
        for c in 0..=cols {
            let mut start = c;
            let h = heights[c];
            while stack.last().is_some_and(|(_, sh)| *sh >= h) {
                let (s, sh) = stack.pop().unwrap();
                let area_h = sh as usize;
                let area_w = c - s;
                if area_h * area_w > best.2 * best.3 {
                    best = (r + 1 - area_h, s, area_h, area_w);
                }
                start = s;
            }
            stack.push((start, h));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_rounds_away_from_zero() {
        assert_eq!(round_to_int(1.5), 2);
        assert_eq!(round_to_int(-1.5), -2);
        assert_eq!(round_to_int(2.2), 2);
    }

    #[test]
    fn rectangle_finds_the_full_block() {
        let mask = vec![true, true, false, true, true, false];
        let (top, left, height, width) = largest_rectangle(&mask, 3);
        assert_eq!((top, left, height, width), (0, 0, 2, 2));
    }
}
