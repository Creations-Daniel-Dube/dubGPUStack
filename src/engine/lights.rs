// SPDX-License-Identifier: GPL-3.0-or-later
//! Brutes FITS : inventaire, métadonnées et géométrie Bayer.

use std::path::{Path, PathBuf};

use super::fits::{self, ImageInfo};

pub const FITS_EXT: &[&str] = &[".fit", ".fits", ".fts"];

#[derive(Clone, Debug)]
pub struct Light {
    pub index: usize,
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub bitpix: i32,
    pub bzero: f32,
    pub bscale: f32,
    pub bayer: String,
    pub bottom_up: bool,
    pub exptime: f64,
    pub date_obs: String,
    pub filter: String,
    pub object: String,
    pub error: String,
    pub data_offset: u64,
    pub xbayroff: i32,
    pub ybayroff: i32,
}

impl Light {
    pub fn number(&self) -> usize {
        self.index + 1
    }

    pub fn name(&self) -> String {
        self.path.file_name().and_then(|name| name.to_str()).unwrap_or("").to_string()
    }

    pub fn cfa_lut(&self) -> Option<[i32; 4]> {
        cfa_lut(&self.bayer, self.xbayroff, self.ybayroff)
    }
}

pub fn cfa_lut(pattern: &str, xoff: i32, yoff: i32) -> Option<[i32; 4]> {
    let pattern = pattern.trim().to_ascii_uppercase();
    if pattern.len() != 4 {
        return None;
    }
    let mut lut = [0; 4];
    for y in 0..2 {
        for x in 0..2 {
            let index = ((y + yoff).rem_euclid(2) * 2 + (x + xoff).rem_euclid(2)) as usize;
            lut[(y * 2 + x) as usize] = match pattern.as_bytes().get(index).copied() {
                Some(b'R') => 0,
                Some(b'G') => 1,
                Some(b'B') => 2,
                _ => return None,
            };
        }
    }
    Some(lut)
}

pub fn find_lights_dir(folder: &Path) -> PathBuf {
    for name in ["lights", "Lights", "LIGHTS", "light", "Light"] {
        let sub = folder.join(name);
        if sub.is_dir() {
            return sub;
        }
    }
    folder.to_path_buf()
}

pub fn dataset_dir(lights_dir: &Path) -> PathBuf {
    let base = lights_dir.file_name().and_then(|name| name.to_str()).unwrap_or("").to_ascii_lowercase();
    if base == "lights" || base == "light" {
        lights_dir.parent().unwrap_or(lights_dir).to_path_buf()
    } else {
        lights_dir.to_path_buf()
    }
}

pub fn list_fits(folder: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(folder) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("").to_ascii_lowercase();
        if FITS_EXT.iter().any(|suffix| suffix.trim_start_matches('.') == ext) {
            out.push(path);
        }
    }
    out.sort();
    out
}

pub fn read_light(path: &Path, index: usize) -> Light {
    let mut light = Light {
        index,
        path: path.to_path_buf(),
        width: 0,
        height: 0,
        channels: 0,
        bitpix: 0,
        bzero: 0.0,
        bscale: 1.0,
        bayer: String::new(),
        bottom_up: false,
        exptime: 0.0,
        date_obs: String::new(),
        filter: String::new(),
        object: String::new(),
        error: String::new(),
        data_offset: 0,
        xbayroff: 0,
        ybayroff: 0,
    };
    match ImageInfo::open(path) {
        Ok(info) => {
            light.width = info.width;
            light.height = info.height;
            light.channels = info.channels;
            light.bitpix = info.bitpix;
            light.bzero = info.bzero;
            light.bscale = info.bscale;
            light.bottom_up = info.bottom_up;
            light.data_offset = info.header.data_offset;
            if info.channels == 1 {
                light.bayer = info.header.get_str("BAYERPAT").trim().to_ascii_uppercase();
            }
            light.exptime = info.header.get("EXPTIME").and_then(|v| v.as_f64()).or_else(|| info.header.get("EXPOSURE").and_then(|v| v.as_f64())).unwrap_or(0.0);
            light.date_obs = info.header.get_str("DATE-OBS");
            light.filter = info.header.get_str("FILTER").trim().to_string();
            light.object = info.header.get_str("OBJECT").trim().to_string();
            light.xbayroff = info.header.get("XBAYROFF").and_then(|v| v.as_int()).unwrap_or(0) as i32;
            light.ybayroff = info.header.get("YBAYROFF").and_then(|v| v.as_int()).unwrap_or(0) as i32;
            if !matches!(light.bitpix, 8 | 16 | -32) {
                light.error = format!("BITPIX {} non pris en charge", light.bitpix);
            } else if light.channels != 1 && light.channels != 3 {
                light.error = format!("{} canaux non pris en charge", light.channels);
            }
        }
        Err(err) => light.error = err,
    }
    let _ = fits::BLOCK;
    light
}

pub fn scan_lights(folder: &Path) -> Vec<Light> {
    let mut lights: Vec<Light> = list_fits(folder).iter().map(|path| read_light(path, 0)).collect();
    lights.sort_by(|a, b| (if a.date_obs.is_empty() { "~" } else { &a.date_obs }, &a.path).cmp(&(if b.date_obs.is_empty() { "~" } else { &b.date_obs }, &b.path)));
    for (index, light) in lights.iter_mut().enumerate() {
        light.index = index;
    }
    lights
}

pub fn common_geometry(lights: &[Light]) -> (usize, usize, usize, String) {
    let mut counts: Vec<((usize, usize, usize, String), usize)> = Vec::new();
    for light in lights.iter().filter(|light| light.error.is_empty()) {
        let key = (light.width, light.height, light.channels, light.bayer.clone());
        if let Some((_, count)) = counts.iter_mut().find(|(found, _)| *found == key) {
            *count += 1;
        } else {
            counts.push((key, 1));
        }
    }
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    counts.into_iter().next().map(|(key, _)| key).unwrap_or((0, 0, 0, String::new()))
}

pub fn file_signature(path: &Path) -> [i64; 2] {
    match std::fs::metadata(path) {
        Ok(meta) => [meta.len() as i64, meta.modified().ok().and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as i64).unwrap_or(0)],
        Err(_) => [0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grbg_lut_follows_the_pattern() {
        let lut = cfa_lut("GRBG", 0, 0).unwrap();
        assert_eq!(lut, [1, 0, 2, 1]);
    }
}
