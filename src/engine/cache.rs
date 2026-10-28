// SPDX-License-Identifier: GPL-3.0-or-later
//! Caches JSON et NPZ à côté des données, compatibles avec GPU Stacker.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::lights::{self, Light};
use super::stars::StarList;
use super::stats::{self, FrameStats};

const VERSION: i64 = 1;

pub fn cache_dir(lights_dir: &Path) -> PathBuf {
    lights::dataset_dir(lights_dir).join("gpustacker")
}

fn load_json(path: &Path) -> Value {
    let Ok(text) = std::fs::read_to_string(path) else { return Value::Null };
    let Ok(value) = serde_json::from_str::<Value>(&text) else { return Value::Null };
    if value.get("version").and_then(|v| v.as_i64()) == Some(VERSION) { value } else { Value::Null }
}

fn save_json(path: &Path, mut body: Map<String, Value>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    body.insert("version".to_string(), json!(VERSION));
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    if std::fs::write(&tmp, serde_json::to_string(&Value::Object(body)).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Analysis {
    pub frames: Map<String, Value>,
    pub stars: std::collections::HashMap<String, StarList>,
    pub json_path: PathBuf,
    pub npz_path: PathBuf,
}

impl Analysis {
    pub fn load(dir: &Path) -> Self {
        let json_path = dir.join("analysis.json");
        let npz_path = dir.join("analysis_stars.npz");
        let value = load_json(&json_path);
        let frames = value.get("frames").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let mut stars = std::collections::HashMap::new();
        if !frames.is_empty() {
            if let Some(loaded) = read_stars_npz(&npz_path) {
                stars = loaded;
            }
        }
        Self { frames, stars, json_path, npz_path }
    }

    pub fn get(&self, light: &Light) -> Option<(StarList, FrameStatsLite)> {
        let entry = self.frames.get(&light.name())?;
        let sig = entry.get("sig").and_then(json_i64_pair);
        if sig != Some(lights::file_signature(&light.path)) {
            return None;
        }
        let stars = self.stars.get(&light.name())?.clone();
        Some((stars, FrameStatsLite {
            nstars: entry.get("nstars").and_then(|v| v.as_i64()).unwrap_or(0),
            fwhm: entry.get("fwhm").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            roundness: entry.get("roundness").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            background: entry.get("background").and_then(|v| v.as_f64()).unwrap_or(0.0),
            noise: entry.get("noise").and_then(|v| v.as_f64()).unwrap_or(0.0),
        }))
    }

    pub fn put(&mut self, light: &Light, stars: &StarList, metrics: &FrameStatsLite) {
        self.frames.insert(light.name(), json!({
            "sig": lights::file_signature(&light.path),
            "nstars": metrics.nstars,
            "fwhm": metrics.fwhm,
            "roundness": metrics.roundness,
            "background": metrics.background,
            "noise": metrics.noise,
        }));
        self.stars.insert(light.name(), stars.clone());
    }

    pub fn save(&self) {
        let names: Vec<String> = self.frames.keys().filter(|name| self.stars.contains_key(*name)).cloned().collect();
        let _ = write_stars_npz(&self.npz_path, &names, &self.stars);
        let mut frames = Map::new();
        for name in &names {
            if let Some(entry) = self.frames.get(name) {
                frames.insert(name.clone(), entry.clone());
            }
        }
        let mut body = Map::new();
        body.insert("frames".to_string(), Value::Object(frames));
        save_json(&self.json_path, body);
    }
}

#[derive(Clone, Debug)]
pub struct FrameStatsLite {
    pub nstars: i64,
    pub fwhm: f64,
    pub roundness: f64,
    pub background: f64,
    pub noise: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RegistrationCache {
    pub path: PathBuf,
    pub reference: String,
    pub model: String,
    pub frames: Map<String, Value>,
    pub failed: Map<String, Value>,
}

impl RegistrationCache {
    pub fn load(dir: &Path) -> Self {
        let path = dir.join("registration.json");
        let value = load_json(&path);
        Self {
            path,
            reference: value.get("reference").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            model: value.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            frames: value.get("frames").and_then(|v| v.as_object()).cloned().unwrap_or_default(),
            failed: value.get("failed").and_then(|v| v.as_object()).cloned().unwrap_or_default(),
        }
    }

    pub fn why_invalid(&self, reference: &str, model: &str, lights: &[Light]) -> String {
        if self.frames.is_empty() {
            return "aucun enregistrement enregistré".to_string();
        }
        if reference != self.reference {
            return format!("la référence a changé ({} → {reference})", if self.reference.is_empty() { "aucune" } else { &self.reference });
        }
        if model != self.model {
            return format!("le modèle d'alignement a changé ({} → {model})", if self.model.is_empty() { "aucun" } else { &self.model });
        }
        let mut missing = 0usize;
        let mut changed = 0usize;
        let mut example = String::new();
        for light in lights {
            let sig = lights::file_signature(&light.path);
            if let Some(entry) = self.frames.get(&light.name()).or_else(|| self.failed.get(&light.name())) {
                if json_i64_pair(entry.get("sig").unwrap_or(&Value::Null)) != Some(sig) {
                    changed += 1;
                    if example.is_empty() { example = light.name(); }
                }
            } else {
                missing += 1;
                if example.is_empty() { example = light.name(); }
            }
        }
        if missing == 0 && changed == 0 {
            return String::new();
        }
        format!("{missing} nouvelle(s) image(s), {changed} fichier(s) modifié(s), par exemple {example}")
    }

    pub fn transform(&self, name: &str) -> Option<[f64; 6]> {
        let list = self.frames.get(name)?.get("t")?.as_array()?;
        let mut m = [0.0; 6];
        for (slot, value) in m.iter_mut().zip(list.iter()) {
            *slot = value.as_f64()?;
        }
        Some(m)
    }

    pub fn save(&self) {
        let mut body = Map::new();
        body.insert("reference".to_string(), json!(self.reference));
        body.insert("model".to_string(), json!(self.model));
        body.insert("frames".to_string(), Value::Object(self.frames.clone()));
        body.insert("failed".to_string(), Value::Object(self.failed.clone()));
        save_json(&self.path, body);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Measures {
    pub path: PathBuf,
    pub dir: PathBuf,
    pub frames: Map<String, Value>,
}

impl Measures {
    pub fn load(dir: &Path) -> Self {
        let path = dir.join("measures.json");
        let value = load_json(&path);
        Self { path, dir: dir.to_path_buf(), frames: value.get("frames").and_then(|v| v.as_object()).cloned().unwrap_or_default() }
    }

    pub fn get(&self, light: &Light, key: &str) -> Option<FrameStats> {
        let entry = self.frames.get(&light.name())?;
        if json_i64_pair(entry.get("sig").unwrap_or(&Value::Null)) != Some(lights::file_signature(&light.path)) {
            return None;
        }
        if entry.get("key").and_then(|v| v.as_str()) != Some(key) {
            return None;
        }
        Some(stats_from_json(entry))
    }

    pub fn mask_path(&self, light: &Light) -> PathBuf {
        let stem = light.path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        self.dir.join("masks").join(format!("{stem}.msk"))
    }

    pub fn save(&self) {
        let mut body = Map::new();
        body.insert("frames".to_string(), Value::Object(self.frames.clone()));
        save_json(&self.path, body);
    }
}

pub fn stats_from_json(entry: &Value) -> FrameStats {
    FrameStats {
        loc: json_opt_list(entry.get("loc")),
        scale: json_opt_list(entry.get("scale")),
        noise: json_opt_list(entry.get("noise")),
        ngood: entry.get("ngood").and_then(|v| v.as_array()).map(|list| list.iter().map(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))).collect()).unwrap_or_default(),
        bad: entry.get("bad").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    }
}

pub fn stats_to_json(stats: &FrameStats) -> Value {
    json!({
        "loc": stats.loc,
        "scale": stats.scale,
        "noise": stats.noise,
        "ngood": stats.ngood,
        "bad": stats.bad,
    })
}

fn json_opt_list(value: Option<&Value>) -> Vec<Option<f64>> {
    value.and_then(|v| v.as_array()).map(|list| list.iter().map(|v| if v.is_null() { None } else { v.as_f64() }).collect()).unwrap_or_default()
}

fn json_i64_pair(value: &Value) -> Option<[i64; 2]> {
    let list = value.as_array()?;
    Some([list.first()?.as_i64()?, list.get(1)?.as_i64()?])
}

pub fn frame_key(transform: &[f64], width: usize, height: usize, pixfrac: f64) -> String {
    let mut text = transform.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(",");
    text.push_str(&format!("|{width}x{height}|{pixfrac:.4}|stats{CACHE}", CACHE = stats::CACHE_VERSION));
    let digest = sha1(text.as_bytes());
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

fn sha1(data: &[u8]) -> [u8; 20] {
    // SHA-1 public domain compact implementation, enough to nommer une clé de cache.
    fn rol(x: u32, n: u32) -> u32 { x.rotate_left(n) }
    let mut h0 = 0x67452301u32;
    let mut h1 = 0xEFCDAB89u32;
    let mut h2 = 0x98BADCFEu32;
    let mut h3 = 0x10325476u32;
    let mut h4 = 0xC3D2E1F0u32;
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for i in 0..80 {
            let (f, k) = if i < 20 { ((b & c) | ((!b) & d), 0x5A827999) } else if i < 40 { (b ^ c ^ d, 0x6ED9EBA1) } else if i < 60 { ((b & c) | (b & d) | (c & d), 0x8F1BBCDC) } else { (b ^ c ^ d, 0xCA62C1D6) };
            let temp = rol(a, 5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(w[i]);
            e = d; d = c; c = rol(b, 30); b = a; a = temp;
        }
        h0 = h0.wrapping_add(a); h1 = h1.wrapping_add(b); h2 = h2.wrapping_add(c); h3 = h3.wrapping_add(d); h4 = h4.wrapping_add(e);
    }
    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

fn read_stars_npz(path: &Path) -> Option<std::collections::HashMap<String, StarList>> {
    let file = std::fs::File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let names = read_npy_strings(&mut archive, "names.npy")?;
    let offsets = read_npy_i64(&mut archive, "offsets.npy")?;
    let stars = read_npy_f32(&mut archive, "stars.npy")?;
    let mut map = std::collections::HashMap::new();
    for (index, name) in names.iter().enumerate() {
        let start = *offsets.get(index)? as usize;
        let end = *offsets.get(index + 1)? as usize;
        if end >= start && end * 6 <= stars.len() {
            map.insert(name.clone(), StarList::from_array(&stars[start * 6..end * 6]));
        }
    }
    Some(map)
}

fn write_stars_npz(path: &Path, names: &[String], stars: &std::collections::HashMap<String, StarList>) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let tmp = PathBuf::from(format!("{}.tmp.npz", path.display()));
    let file = std::fs::File::create(&tmp).map_err(|err| err.to_string())?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let arrays: Vec<&StarList> = names.iter().filter_map(|name| stars.get(name)).collect();
    let mut offsets = vec![0i64];
    let mut flat = Vec::new();
    for stars in &arrays {
        flat.extend(stars.to_array());
        offsets.push(offsets.last().copied().unwrap_or(0) + stars.len() as i64);
    }
    zip.start_file("names.npy", opts).map_err(|err| err.to_string())?;
    zip.write_all(&npy_unicode(names)).map_err(|err| err.to_string())?;
    zip.start_file("offsets.npy", opts).map_err(|err| err.to_string())?;
    zip.write_all(&npy_i64(&offsets)).map_err(|err| err.to_string())?;
    zip.start_file("stars.npy", opts).map_err(|err| err.to_string())?;
    let rows = if arrays.is_empty() { 0 } else { flat.len() / 6 };
    zip.write_all(&npy_f32(&flat, &[rows, 6])).map_err(|err| err.to_string())?;
    zip.finish().map_err(|err| err.to_string())?;
    std::fs::rename(&tmp, path).map_err(|err| err.to_string())
}

fn npy_header(descr: &str, shape: &str) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let mut header = dict.into_bytes();
    header.push(b'\n');
    let prefix = 10;
    let pad = (64 - ((prefix + header.len()) % 64)) % 64;
    header.extend(std::iter::repeat_n(b' ', pad));
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend(header);
    out
}

fn npy_f32(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let shape_txt = if shape.len() == 1 { format!("({},)", shape[0]) } else { format!("({}, {})", shape[0], shape[1]) };
    let mut out = npy_header("<f4", &shape_txt);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn npy_i64(data: &[i64]) -> Vec<u8> {
    let mut out = npy_header("<i8", &format!("({},)", data.len()));
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn npy_unicode(names: &[String]) -> Vec<u8> {
    let width = names.iter().map(|name| name.chars().count()).max().unwrap_or(1).max(1);
    let mut out = npy_header(&format!("<U{width}"), &format!("({},)", names.len()));
    for name in names {
        let chars: Vec<char> = name.chars().collect();
        for index in 0..width {
            let ch = chars.get(index).copied().unwrap_or('\0') as u32;
            out.extend_from_slice(&ch.to_le_bytes());
        }
    }
    out
}

fn read_zip_file(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Option<Vec<u8>> {
    let mut file = archive.by_name(name).ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

fn npy_payload(bytes: &[u8]) -> Option<(&[u8], String)> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let major = bytes[6];
    let header_len = if major == 1 { u16::from_le_bytes([bytes[8], bytes[9]]) as usize } else { u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize };
    let header_at = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(bytes.get(header_at..header_at + header_len)?).ok()?.to_string();
    Some((&bytes[header_at + header_len..], header))
}

fn read_npy_f32(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Option<Vec<f32>> {
    let bytes = read_zip_file(archive, name)?;
    let (payload, _) = npy_payload(&bytes)?;
    Some(payload.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_npy_i64(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Option<Vec<i64>> {
    let bytes = read_zip_file(archive, name)?;
    let (payload, _) = npy_payload(&bytes)?;
    Some(payload.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
}

fn read_npy_strings(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Option<Vec<String>> {
    let bytes = read_zip_file(archive, name)?;
    let (payload, header) = npy_payload(&bytes)?;
    let width = header.split("<U").nth(1)?.split(|c: char| !c.is_ascii_digit()).next()?.parse::<usize>().ok()?;
    let mut names = Vec::new();
    for chunk in payload.chunks(width * 4) {
        if chunk.len() < width * 4 {
            break;
        }
        let mut text = String::new();
        for char_bytes in chunk.chunks_exact(4) {
            let code = u32::from_le_bytes(char_bytes.try_into().ok()?);
            if code == 0 { break; }
            if let Some(ch) = char::from_u32(code) { text.push(ch); }
        }
        names.push(text);
    }
    Some(names)
}

pub fn load_selection_excluded(path: &Path) -> (Vec<String>, String) {
    let Ok(text) = std::fs::read_to_string(path) else { return (Vec::new(), String::new()) };
    let Ok(value) = serde_json::from_str::<Value>(&text) else { return (Vec::new(), String::new()) };
    let excluded = value.get("excluded").and_then(|v| v.as_array()).map(|list| list.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
    let reference = value.get("reference").and_then(|v| v.as_str()).unwrap_or("").to_string();
    (excluded, reference)
}

pub fn save_selection(path: &Path, excluded: &[String], reference: &str) {
    if excluded.is_empty() && reference.is_empty() {
        let _ = std::fs::remove_file(path);
        return;
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, serde_json::to_string(&json!({"excluded": excluded, "reference": reference})).unwrap_or_default());
}

pub struct SeqStatsCache {
    pub path: PathBuf,
    pub frames: Map<String, Value>,
}

impl SeqStatsCache {
    pub fn load(path: &Path) -> Self {
        let value = std::fs::read_to_string(path).ok().and_then(|text| serde_json::from_str::<Value>(&text).ok()).unwrap_or(Value::Null);
        let frames = if value.get("version").and_then(|v| v.as_i64()) == Some(stats::CACHE_VERSION as i64) {
            value.get("frames").and_then(|v| v.as_object()).cloned().unwrap_or_default()
        } else {
            Map::new()
        };
        Self { path: path.to_path_buf(), frames }
    }

    pub fn get(&self, filenum: i64, sig: [i64; 2]) -> Option<FrameStats> {
        let entry = self.frames.get(&filenum.to_string())?;
        if json_i64_pair(entry.get("sig").unwrap_or(&Value::Null)) != Some(sig) {
            return None;
        }
        Some(stats_from_json(entry))
    }

    pub fn put(&mut self, filenum: i64, sig: [i64; 2], stats: &FrameStats) {
        let mut value = stats_to_json(stats);
        if let Some(obj) = value.as_object_mut() {
            obj.insert("sig".to_string(), json!(sig));
        }
        self.frames.insert(filenum.to_string(), value);
    }

    pub fn save(&self) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let body = json!({"version": stats::CACHE_VERSION, "frames": self.frames});
        let tmp = PathBuf::from(format!("{}.tmp", self.path.display()));
        if std::fs::write(&tmp, body.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}
