// SPDX-License-Identifier: GPL-3.0-or-later
//! Lecteur et écrivain FITS du HDU primaire.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const BLOCK: usize = 2880;

#[derive(Clone, Debug)]
pub enum FitsValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl FitsValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Self::Str(_) => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Str(v) => v.clone(),
            Self::Int(v) => v.to_string(),
            Self::Float(v) => v.to_string(),
            Self::Bool(v) => if *v { "T" } else { "F" }.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub cards: Vec<String>,
    pub values: Vec<(String, FitsValue)>,
    pub data_offset: u64,
    pub path: PathBuf,
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&FitsValue> {
        self.values.iter().rev().find(|(name, _)| name == key).map(|(_, value)| value)
    }

    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(FitsValue::as_f64).unwrap_or(default)
    }

    pub fn get_str(&self, key: &str) -> String {
        self.get(key).map(FitsValue::as_str).unwrap_or_default()
    }

    pub fn bitpix(&self) -> Result<i32, String> {
        self.get("BITPIX").and_then(FitsValue::as_int).map(|v| v as i32).ok_or_else(|| format!("{}: BITPIX manquant", self.path.display()))
    }

    pub fn shape(&self) -> Result<(usize, usize, usize), String> {
        let naxis = self.get("NAXIS").and_then(FitsValue::as_int).unwrap_or(0);
        if naxis != 2 && naxis != 3 {
            return Err(format!("{}: NAXIS={naxis}, image 2D ou 3D attendue", self.path.display()));
        }
        let w = self.get("NAXIS1").and_then(FitsValue::as_int).unwrap_or(0) as usize;
        let h = self.get("NAXIS2").and_then(FitsValue::as_int).unwrap_or(0) as usize;
        let c = if naxis == 3 { self.get("NAXIS3").and_then(FitsValue::as_int).unwrap_or(1) as usize } else { 1 };
        if w == 0 || h == 0 || c == 0 {
            return Err(format!("{}: dimensions nulles", self.path.display()));
        }
        Ok((c, h, w))
    }

    pub fn item_size(&self) -> Result<usize, String> {
        Ok(match self.bitpix()? {
            8 => 1,
            16 => 2,
            32 | -32 => 4,
            -64 => 8,
            other => return Err(format!("{}: BITPIX {other} non pris en charge", self.path.display())),
        })
    }

    pub fn user_cards(&self) -> Vec<String> {
        const STRUCTURAL: &[&str] = &["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "END"];
        self.cards.iter().filter(|card| !STRUCTURAL.contains(&card[..8.min(card.len())].trim())).cloned().collect()
    }
}

pub fn read_header(path: &Path) -> Result<Header, String> {
    let mut file = File::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut cards = Vec::new();
    let mut values = Vec::new();
    let mut block = [0u8; BLOCK];
    let mut nblk = 0u64;
    loop {
        file.read_exact(&mut block).map_err(|_| format!("{}: en-tête FITS tronqué", path.display()))?;
        nblk += 1;
        for index in (0..BLOCK).step_by(80) {
            let card = String::from_utf8_lossy(&block[index..index + 80]).replace('\0', " ");
            let key = card[..8].trim().to_string();
            if key == "END" {
                return Ok(Header { cards, values, data_offset: nblk * BLOCK as u64, path: path.to_path_buf() });
            }
            if card.len() >= 10 && &card[8..10] == "= " {
                values.push((key, parse_value(&card[10..])));
            }
            cards.push(card);
        }
        if nblk > 4000 {
            return Err(format!("{}: carte END introuvable", path.display()));
        }
    }
}

fn parse_value(raw: &str) -> FitsValue {
    let s = raw.trim();
    if let Some(rest) = s.strip_prefix('\'') {
        let mut out = String::new();
        let chars: Vec<char> = rest.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '\'' {
                if i + 1 < chars.len() && chars[i + 1] == '\'' {
                    out.push('\'');
                    i += 2;
                    continue;
                }
                break;
            }
            out.push(chars[i]);
            i += 1;
        }
        return FitsValue::Str(out.trim_end().to_string());
    }
    let v = s.split('/').next().unwrap_or("").trim();
    if v == "T" {
        return FitsValue::Bool(true);
    }
    if v == "F" {
        return FitsValue::Bool(false);
    }
    if let Ok(int) = v.parse::<i64>() {
        return FitsValue::Int(int);
    }
    if let Ok(float) = v.replace('D', "E").parse::<f64>() {
        return FitsValue::Float(float);
    }
    FitsValue::Str(v.to_string())
}

pub fn card(key: &str, value: FitsValue, comment: &str) -> String {
    let body = match &value {
        FitsValue::Bool(flag) => format!("{key:<8}= {:>20}", if *flag { "T" } else { "F" }),
        FitsValue::Str(text) => {
            let inner = format!("{:<8}", text.replace('\'', "''"));
            let quoted = format!("'{inner}'");
            format!("{key:<8}= {quoted:<20}")
        }
        FitsValue::Int(int) => format!("{key:<8}= {int:>20}"),
        FitsValue::Float(float) => {
            let mut text = if float.abs() < 1e15 { format!("{float}") } else { format!("{float:.15E}") };
            if text.len() > 20 {
                text = format!("{float:.12E}");
            }
            format!("{key:<8}= {text:>20}")
        }
    };
    let mut line = if comment.is_empty() { body } else { format!("{body} / {comment}") };
    if line.len() > 80 {
        line.truncate(80);
    }
    format!("{line:<80}")
}

pub fn history(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let (head, tail) = rest.split_at(rest.len().min(72));
        out.push(format!("{:<80}", format!("HISTORY {head}")));
        rest = tail;
    }
    out
}

/// Décode des octets FITS en valeurs 0..1 (16 bits) ou en flottants natifs.
pub fn decode_sample(bitpix: i32, bytes: &[u8], bscale: f32, bzero: f32) -> f32 {
    match bitpix {
        8 => bytes.first().copied().unwrap_or(0) as f32 / 255.0,
        16 => {
            let raw = i16::from_be_bytes([bytes[0], bytes[1]]) as f32;
            (raw * bscale + bzero) / 65535.0
        }
        32 => {
            let raw = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32;
            raw * bscale + bzero
        }
        -32 => {
            let raw = f32::from_bits(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            if (bscale - 1.0).abs() > f32::EPSILON || bzero != 0.0 { raw * bscale + bzero } else { raw }
        }
        -64 => {
            let raw = f64::from_bits(u64::from_be_bytes(bytes[..8].try_into().unwrap_or([0; 8]))) as f32;
            if (bscale - 1.0).abs() > f32::EPSILON || bzero != 0.0 { raw * bscale + bzero } else { raw }
        }
        _ => 0.0,
    }
}

pub struct ImageInfo {
    pub header: Header,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub bitpix: i32,
    pub bscale: f32,
    pub bzero: f32,
    pub bottom_up: bool,
}

impl ImageInfo {
    pub fn open(path: &Path) -> Result<Self, String> {
        let header = read_header(path)?;
        let (channels, height, width) = header.shape()?;
        let bitpix = header.bitpix()?;
        let bottom_up = header.get_str("ROWORDER").trim().eq_ignore_ascii_case("BOTTOM-UP");
        Ok(Self {
            bscale: header.get_f64("BSCALE", 1.0) as f32,
            bzero: header.get_f64("BZERO", 0.0) as f32,
            header,
            channels,
            height,
            width,
            bitpix,
            bottom_up,
        })
    }
}

/// Lignes top-down `[y0, y1)` d'un plan, en f32.
pub fn read_rows(info: &ImageInfo, plane: usize, y0: usize, y1: usize) -> Result<Vec<f32>, String> {
    if y1 < y0 || y1 > info.height || plane >= info.channels {
        return Err(format!("{}: lignes hors image", info.header.path.display()));
    }
    let item = info.header.item_size()?;
    let width = info.width;
    let height = info.height;
    let rows = y1 - y0;
    let mut file = File::open(&info.header.path).map_err(|err| err.to_string())?;
    let mut out = vec![0.0f32; rows * width];
    let mut raw = vec![0u8; width * item];
    for (slot, y) in (y0..y1).enumerate() {
        let file_row = if info.bottom_up { height - 1 - y } else { y };
        let offset = info.header.data_offset + ((plane * height + file_row) * width * item) as u64;
        file.seek(SeekFrom::Start(offset)).map_err(|err| err.to_string())?;
        file.read_exact(&mut raw).map_err(|_| format!("{}: lecture trop courte", info.header.path.display()))?;
        for x in 0..width {
            out[slot * width + x] = decode_sample(info.bitpix, &raw[x * item..(x + 1) * item], info.bscale, info.bzero);
        }
    }
    Ok(out)
}

pub fn read_image(path: &Path) -> Result<(Vec<f32>, ImageInfo), String> {
    let info = ImageInfo::open(path)?;
    let mut data = vec![0.0f32; info.channels * info.height * info.width];
    for plane in 0..info.channels {
        let rows = read_rows(&info, plane, 0, info.height)?;
        let offset = plane * info.height * info.width;
        data[offset..offset + rows.len()].copy_from_slice(&rows);
    }
    Ok((data, info))
}

pub fn write_fits(path: &Path, data: &[f32], channels: usize, height: usize, width: usize, cards: &[String]) -> Result<(), String> {
    if data.len() < channels * height * width || width == 0 || height == 0 {
        return Err("image de sortie vide".to_string());
    }
    let mut head = vec![
        card("SIMPLE", FitsValue::Bool(true), "conforms to FITS standard"),
        card("BITPIX", FitsValue::Int(-32), "32-bit float"),
        card("NAXIS", FitsValue::Int(if channels > 1 { 3 } else { 2 }), ""),
        card("NAXIS1", FitsValue::Int(width as i64), ""),
        card("NAXIS2", FitsValue::Int(height as i64), ""),
    ];
    if channels > 1 {
        head.push(card("NAXIS3", FitsValue::Int(channels as i64), ""));
    }
    for extra in cards {
        let mut line = extra.clone();
        if line.len() < 80 {
            line = format!("{line:<80}");
        } else if line.len() > 80 {
            line.truncate(80);
        }
        head.push(line);
    }
    head.push(card("ROWORDER", FitsValue::Str("BOTTOM-UP".to_string()), "first row is the bottom"));
    head.push(format!("{:<80}", "END"));
    let mut text = head.concat();
    let padded = text.len().div_ceil(BLOCK) * BLOCK;
    text.extend(std::iter::repeat_n(' ', padded - text.len()));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let tmp = PathBuf::from(format!("{}.partial", path.display()));
    let write_result = (|| {
        let mut file = File::create(&tmp).map_err(|err| err.to_string())?;
        file.write_all(text.as_bytes()).map_err(|err| err.to_string())?;
        let mut raw = vec![0u8; width * 4];
        for channel in 0..channels {
            for y in (0..height).rev() {
                for x in 0..width {
                    let value = data[channel * height * width + y * width + x];
                    raw[x * 4..(x + 1) * 4].copy_from_slice(&value.to_be_bytes());
                }
                file.write_all(&raw).map_err(|err| err.to_string())?;
            }
        }
        let nbytes = channels * height * width * 4;
        let pad = (BLOCK - (nbytes % BLOCK)) % BLOCK;
        file.write_all(&vec![0u8; pad]).map_err(|err| err.to_string())?;
        file.flush().map_err(|err| err.to_string())?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    std::fs::rename(&tmp, path).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_float_image() {
        let dir = std::env::temp_dir().join("dubgpustack-fits-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("round.fit");
        let data = vec![0.0, 0.25, 0.5, 1.0, 0.125, 0.75];
        write_fits(&path, &data, 1, 2, 3, &[]).unwrap();
        let (back, info) = read_image(&path).unwrap();
        assert_eq!((info.channels, info.height, info.width), (1, 2, 3));
        for (got, expect) in back.iter().zip(data.iter()) {
            assert!((got - expect).abs() < 1e-6, "{got} != {expect}");
        }
    }

    #[test]
    fn unsigned_16_uses_bzero() {
        let sample = decode_sample(16, &32767i16.to_be_bytes(), 1.0, 32768.0);
        assert!((sample - 1.0).abs() < 1e-5, "{sample}");
    }
}
