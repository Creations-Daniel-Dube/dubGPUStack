// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub recent: Vec<String>,
    pub last_dir: String,
    pub rejection: String,
    pub sigma_low: f64,
    pub sigma_high: f64,
    pub weighting: String,
    pub normalization: String,
    pub rgb_equal: bool,
    pub feather: f64,
    pub output_norm: bool,
    pub pixfrac: f64,
    pub drizzle_scale: f64,
    pub transform: String,
    pub workers: i32,
    pub strip: i32,
    pub tile: i32,
    pub gpu_mem_gib: f64,
    pub save_coverage: bool,
    pub crop: bool,
    pub min_coverage: i32,
    pub output_dir: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            recent: Vec::new(),
            last_dir: String::new(),
            rejection: "winsorized".to_string(),
            sigma_low: 3.0,
            sigma_high: 3.0,
            weighting: "noise".to_string(),
            normalization: "addscale".to_string(),
            rgb_equal: true,
            feather: 20.0,
            output_norm: true,
            pixfrac: 1.0,
            drizzle_scale: 1.0,
            transform: "affine".to_string(),
            workers: 6,
            strip: 64,
            tile: 1024,
            gpu_mem_gib: 0.0,
            save_coverage: false,
            crop: false,
            min_coverage: 50,
            output_dir: String::new(),
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let Ok(text) = std::fs::read_to_string(settings_path()) else { return Self::default() };
        serde_json::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self) {
        let path = settings_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }

    pub fn remember(&mut self, path: &str) {
        self.recent.retain(|item| item != path);
        self.recent.insert(0, path.to_string());
        self.recent.truncate(8);
        if let Some(parent) = std::path::Path::new(path).parent() {
            self.last_dir = parent.display().to_string();
        }
        self.save();
    }
}

pub fn settings_path() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("APPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
        base.join("dubGPUStack").join("settings.json")
    } else {
        let base = std::env::var("XDG_CONFIG_HOME").ok().filter(|value| !value.is_empty()).map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|home| PathBuf::from(home).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("dubGPUStack").join("settings.json")
    }
}
