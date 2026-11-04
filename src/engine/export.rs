// SPDX-License-Identifier: GPL-3.0-or-later
//! Exports de l'image stretchée : PNG 8 bits, PNG 16 bits, TIFF 32 bits flottant.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub fn write_png(path: &Path, data: &[f32], channels: usize, height: usize, width: usize, sixteen: bool) -> Result<(), String> {
    let pixels = height.saturating_mul(width);
    let planes = export_planes(channels);
    if planes == 0 || pixels == 0 || width > u32::MAX as usize || height > u32::MAX as usize || data.len() < planes * pixels {
        return Err("image de sortie vide".to_string());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    let file = File::create(path).map_err(|err| err.to_string())?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(if planes == 3 { png::ColorType::Rgb } else { png::ColorType::Grayscale });
    encoder.set_depth(if sixteen { png::BitDepth::Sixteen } else { png::BitDepth::Eight });
    let mut writer = encoder.write_header().map_err(|err| err.to_string())?;
    let bytes = png_bytes(data, planes, pixels, sixteen);
    writer.write_image_data(&bytes).map_err(|err| err.to_string())?;
    writer.finish().map_err(|err| err.to_string())?;
    Ok(())
}

pub fn write_tiff32(path: &Path, data: &[f32], channels: usize, height: usize, width: usize) -> Result<(), String> {
    let pixels = height.saturating_mul(width);
    let planes = export_planes(channels);
    if planes == 0 || pixels == 0 || width > u32::MAX as usize || height > u32::MAX as usize || data.len() < planes * pixels {
        return Err("image de sortie vide".to_string());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    let file = File::create(path).map_err(|err| err.to_string())?;
    let mut encoder = tiff::encoder::TiffEncoder::new(BufWriter::new(file)).map_err(|err| err.to_string())?;
    let samples = interleave(data, planes, pixels);
    if planes == 3 {
        encoder.write_image::<tiff::encoder::colortype::RGB32Float>(width as u32, height as u32, &samples).map_err(|err| err.to_string())?;
    } else {
        encoder.write_image::<tiff::encoder::colortype::Gray32Float>(width as u32, height as u32, &samples).map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn export_planes(channels: usize) -> usize {
    if channels >= 3 { 3 } else if channels >= 1 { 1 } else { 0 }
}

fn interleave(data: &[f32], planes: usize, pixels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels * planes);
    for i in 0..pixels {
        for c in 0..planes {
            let v = data[c * pixels + i];
            out.push(if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
        }
    }
    out
}

fn png_bytes(data: &[f32], planes: usize, pixels: usize, sixteen: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels * planes * if sixteen { 2 } else { 1 });
    for i in 0..pixels {
        for c in 0..planes {
            let v = data[c * pixels + i];
            let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
            if sixteen {
                let sample = (v as f64 * 65535.0).round().clamp(0.0, 65535.0) as u16;
                out.extend_from_slice(&sample.to_be_bytes());
            } else {
                out.push((v * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample() -> Vec<f32> {
        // 2×2 RVB, planaire. Pixel (0,0) = (0, 0.25, 0.5).
        let mut data = vec![0.0f32; 12];
        data[0] = 0.0;
        data[1] = 1.0;
        data[2] = 0.0;
        data[3] = 0.0;
        data[4] = 0.25;
        data[5] = 0.0;
        data[6] = 1.0;
        data[7] = 0.0;
        data[8] = 0.5;
        data[9] = 0.0;
        data[10] = 0.0;
        data[11] = 1.0;
        data
    }

    fn dir() -> PathBuf {
        let path = std::env::temp_dir().join("dubgpustack-export-test");
        let _ = std::fs::create_dir_all(&path);
        path
    }

    #[test]
    fn png8_png16_and_tiff32_roundtrip() {
        let data = sample();
        let folder = dir();
        let png8 = folder.join("sample.png");
        let png16 = folder.join("sample_16.png");
        let tiff = folder.join("sample.tiff");
        write_png(&png8, &data, 3, 2, 2, false).unwrap();
        write_png(&png16, &data, 3, 2, 2, true).unwrap();
        write_tiff32(&tiff, &data, 3, 2, 2).unwrap();

        let eight = read_png(&png8);
        assert_eq!((eight.0, eight.1), (2, 2));
        assert_eq!(eight.2, png::ColorType::Rgb);
        assert_eq!(eight.3, png::BitDepth::Eight);
        assert_eq!(eight.4[..3], [0, 64, 128]);

        let sixteen = read_png(&png16);
        assert_eq!((sixteen.0, sixteen.1), (2, 2));
        assert_eq!(sixteen.3, png::BitDepth::Sixteen);
        assert_eq!(&sixteen.4[..6], &u16_be_bytes([0, 16384, 32768]));

        let file = File::open(&tiff).unwrap();
        let mut decoder = tiff::decoder::Decoder::new(file).unwrap();
        assert_eq!(decoder.dimensions().unwrap(), (2, 2));
        match decoder.read_image().unwrap() {
            tiff::decoder::DecodingResult::F32(values) => {
                assert!((values[0] - 0.0).abs() < 1e-6);
                assert!((values[1] - 0.25).abs() < 1e-6);
                assert!((values[2] - 0.5).abs() < 1e-6);
            }
            other => panic!("TIFF flottant attendu, obtenu {other:?}"),
        }
    }

    fn u16_be_bytes(samples: [u16; 3]) -> [u8; 6] {
        let mut out = [0u8; 6];
        for (i, sample) in samples.into_iter().enumerate() {
            out[i * 2..i * 2 + 2].copy_from_slice(&sample.to_be_bytes());
        }
        out
    }

    fn read_png(path: &Path) -> (u32, u32, png::ColorType, png::BitDepth, Vec<u8>) {
        let file = File::open(path).unwrap();
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        (info.width, info.height, info.color_type, info.bit_depth, buf)
    }
}
