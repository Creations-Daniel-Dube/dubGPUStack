// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use eframe::egui::IconData;

const ICON_BYTES: &[u8] = include_bytes!("../images/dubGPUStack.ico");

pub fn app_icon() -> Arc<IconData> {
    let (width, height, rgba) = decode_largest_dib(ICON_BYTES).expect("images/dubGPUStack.ico");
    Arc::new(IconData { rgba, width, height })
}

fn decode_largest_dib(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    if bytes.len() < 6 || u16_at(bytes, 0)? != 0 || u16_at(bytes, 2)? != 1 {
        return None;
    }
    let count = u16_at(bytes, 4)? as usize;
    let mut best: Option<(u32, u32, Vec<u8>)> = None;
    for index in 0..count {
        let entry = 6 + index * 16;
        let offset = u32_at(bytes, entry + 12)? as usize;
        let Some((width, height, rgba)) = decode_dib(bytes, offset) else { continue };
        let replace = best.as_ref().is_none_or(|current| width * height > current.0 * current.1);
        if replace {
            best = Some((width, height, rgba));
        }
    }
    best
}

fn decode_dib(bytes: &[u8], offset: usize) -> Option<(u32, u32, Vec<u8>)> {
    if offset + 40 > bytes.len() || u32_at(bytes, offset)? != 40 {
        return None;
    }
    let width = i32_at(bytes, offset + 4)?;
    let stored_height = i32_at(bytes, offset + 8)?;
    let bit_count = u16_at(bytes, offset + 14)?;
    let compression = u32_at(bytes, offset + 16)?;
    if width <= 0 || stored_height <= 1 || bit_count != 24 || compression != 0 {
        return None;
    }
    let width = width as usize;
    let height = (stored_height as usize) / 2;
    let xor_stride = (width * 3).div_ceil(4) * 4;
    let and_stride = width.div_ceil(32) * 4;
    let xor_at = offset + 40;
    let and_at = xor_at + xor_stride * height;
    if and_at + and_stride * height > bytes.len() {
        return None;
    }
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        let file_row = height - 1 - y;
        let xor_row = xor_at + file_row * xor_stride;
        let and_row = and_at + file_row * and_stride;
        for x in 0..width {
            let pixel = xor_row + x * 3;
            let mask = bytes[and_row + x / 8];
            let transparent = (mask >> (7 - (x % 8))) & 1 == 1;
            let out = (y * width + x) * 4;
            rgba[out] = bytes[pixel + 2];
            rgba[out + 1] = bytes[pixel + 1];
            rgba[out + 2] = bytes[pixel];
            rgba[out + 3] = if transparent { 0 } else { 255 };
        }
    }
    Some((width as u32, height as u32, rgba))
}

fn u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(offset..offset + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?))
}

fn i32_at(bytes: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?))
}
