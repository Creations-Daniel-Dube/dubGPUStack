// SPDX-License-Identifier: GPL-3.0-or-later
//! Correction du bruit vert (SCNR) masquée sur le fond.
//!
//! Neutre `(R + B) / 2`. Le masque se mesure sur ce neutre : le grain vert
//! du fond (G élevé, R et B bas) est corrigé ; les étoiles et le signal
//! (OIII, Hα, continuum) qui relèvent R ou B restent protégés. La luminance
//! Rec.709 d'avant correction est restaurée pour éviter un fond magenta.

use std::sync::atomic::{AtomicBool, Ordering};

const MAD_TO_SIGMA: f64 = 1.4826;
const SIGMA_CUT: f64 = 4.0;
const SAMPLE_CAP: usize = 250_000;
const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Corrige l'excès de vert près du fond. Image planaire, mono ou RVB, dans `[0, 1]`.
/// Mono : aucun changement.
pub fn apply(image: &mut [f32], channels: usize, height: usize, width: usize, cancel: &AtomicBool) -> Result<(), String> {
    let pixels = height.saturating_mul(width);
    if channels == 1 {
        return Ok(());
    }
    if channels != 3 {
        return Err("la correction du bruit vert attend une image mono ou couleur".to_string());
    }
    if height == 0 || width == 0 || image.len() < 3 * pixels {
        return Err("image trop petite pour la correction du bruit vert".to_string());
    }
    check(cancel)?;
    let (bg, sigma) = neutral_stats(image, pixels);
    // Fond plat (MAD nul) : tout le monde sous la médiane reste corrigé à plein.
    let lo = bg;
    let hi = bg + SIGMA_CUT * sigma.max(0.0);
    let span = hi - lo;
    for pixel in 0..pixels {
        if pixel & 0x3FFF == 0 {
            check(cancel)?;
        }
        let r = image[pixel] as f64;
        let g = image[pixels + pixel] as f64;
        let b = image[2 * pixels + pixel] as f64;
        let n = 0.5 * (r + b);
        if g <= n {
            continue;
        }
        let w = mask_weight(n, lo, hi, span);
        if w <= 0.0 {
            continue;
        }
        let luma_before = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b;
        let g_new = g * (1.0 - w) + n * w;
        let luma_after = LUMA[0] * r + LUMA[1] * g_new + LUMA[2] * b;
        let scale = if luma_after > 1e-12 { luma_before / luma_after } else { 1.0 };
        image[pixel] = (r * scale).clamp(0.0, 1.0) as f32;
        image[pixels + pixel] = (g_new * scale).clamp(0.0, 1.0) as f32;
        image[2 * pixels + pixel] = (b * scale).clamp(0.0, 1.0) as f32;
    }
    Ok(())
}

fn mask_weight(n: f64, lo: f64, hi: f64, span: f64) -> f64 {
    if n <= lo {
        return 1.0;
    }
    if n >= hi || span <= 0.0 {
        return 0.0;
    }
    let t = ((n - lo) / span).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

fn neutral_stats(image: &[f32], pixels: usize) -> (f64, f64) {
    let step = if pixels > SAMPLE_CAP { (pixels + SAMPLE_CAP - 1) / SAMPLE_CAP } else { 1 };
    let mut sample: Vec<f32> = (0..pixels)
        .step_by(step)
        .map(|i| 0.5 * (image[i] + image[2 * pixels + i]))
        .collect();
    if sample.is_empty() {
        return (0.0, 0.0);
    }
    let median = median_mut(&mut sample);
    let mut abs_dev: Vec<f32> = sample.iter().map(|v| (v - median as f32).abs()).collect();
    let mad = median_mut(&mut abs_dev);
    (median, mad * MAD_TO_SIGMA)
}

fn median_mut(values: &mut [f32]) -> f64 {
    let count = values.len();
    if count == 0 {
        return 0.0;
    }
    if count % 2 == 1 {
        let mid = count / 2;
        values.select_nth_unstable_by(mid, f32::total_cmp);
        values[mid] as f64
    } else {
        let mid = count / 2;
        values.select_nth_unstable_by(mid, f32::total_cmp);
        let high = values[mid] as f64;
        let low = values[..mid].iter().copied().max_by(f32::total_cmp).unwrap_or(0.0) as f64;
        (low + high) * 0.5
    }
}

fn check(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err("cancelled".to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn luma(r: f32, g: f32, b: f32) -> f64 {
        LUMA[0] * r as f64 + LUMA[1] * g as f64 + LUMA[2] * b as f64
    }

    #[test]
    fn background_green_becomes_neutral_and_keeps_luminance() {
        // Fond gris à 0,20 avec un pixel de grain vert au centre.
        let width = 8usize;
        let height = 8usize;
        let pixels = width * height;
        let mut image = vec![0.20f32; 3 * pixels];
        let center = 4 * width + 4;
        let r0 = 0.20f32;
        let g0 = 0.35f32;
        let b0 = 0.20f32;
        image[center] = r0;
        image[pixels + center] = g0;
        image[2 * pixels + center] = b0;
        let before = luma(r0, g0, b0);
        apply(&mut image, 3, height, width, &AtomicBool::new(false)).unwrap();
        let r = image[center];
        let g = image[pixels + center];
        let b = image[2 * pixels + center];
        assert!((r - b).abs() < 1e-4, "R et B restent égaux");
        assert!((g - r).abs() < 1e-3, "G rejoint le neutre");
        assert!((luma(r, g, b) - before).abs() < 1e-4, "luminance conservée");
    }

    #[test]
    fn bright_continuum_is_left_alone() {
        let width = 8usize;
        let height = 8usize;
        let pixels = width * height;
        let mut image = vec![0.20f32; 3 * pixels];
        let center = 4 * width + 4;
        // Continuum / OIII : (R+B)/2 nettement au-dessus du fond.
        image[center] = 0.70;
        image[pixels + center] = 0.85;
        image[2 * pixels + center] = 0.70;
        let r0 = image[center];
        let g0 = image[pixels + center];
        let b0 = image[2 * pixels + center];
        apply(&mut image, 3, height, width, &AtomicBool::new(false)).unwrap();
        assert!((image[center] - r0).abs() < 1e-6);
        assert!((image[pixels + center] - g0).abs() < 1e-6);
        assert!((image[2 * pixels + center] - b0).abs() < 1e-6);
    }

    #[test]
    fn mono_image_is_unchanged() {
        let mut image = vec![0.1f32, 0.2, 0.3, 0.4];
        let before = image.clone();
        apply(&mut image, 1, 2, 2, &AtomicBool::new(false)).unwrap();
        assert_eq!(image, before);
    }
}
