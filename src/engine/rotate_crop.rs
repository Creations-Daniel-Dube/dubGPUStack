// SPDX-License-Identifier: GPL-3.0-or-later
//! Recadrage rotatif : le rectangle tourne dans le sens horaire, l'image extraite
//! est échantillonnée dans le repère de ce rectangle et sort donc droite.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos4,
}

impl Interpolation {
    pub const ALL: [Interpolation; 4] = [Self::Nearest, Self::Bilinear, Self::Bicubic, Self::Lanczos4];

    pub fn label(self) -> &'static str {
        match self {
            Self::Nearest => "Plus proche",
            Self::Bilinear => "Bilinéaire",
            Self::Bicubic => "Bicubique",
            Self::Lanczos4 => "Lanczos-4",
        }
    }

    pub fn history_name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
            Self::Lanczos4 => "lanczos4",
        }
    }
}

/// Rectangle en coordonnées de bord : le pixel `(x, y)` couvre `[x, x+1) × [y, y+1)`.
/// L'angle est en degrés, positif dans le sens horaire, avec y vers le bas.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    pub angle_deg: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub rect: Rect,
    pub interpolation: Interpolation,
    pub clamp: bool,
}

pub fn is_usable(rect: &Rect) -> bool {
    rect.center_x.is_finite()
        && rect.center_y.is_finite()
        && rect.width.is_finite()
        && rect.height.is_finite()
        && rect.angle_deg.is_finite()
        && rect.width >= 8.0
        && rect.height >= 8.0
}

/// Coins dans l'ordre haut-gauche, haut-droite, bas-droite, bas-gauche du rectangle tourné.
pub fn corners(rect: &Rect) -> [[f64; 2]; 4] {
    let hw = rect.width / 2.0;
    let hh = rect.height / 2.0;
    let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
    let mut out = [[0.0; 2]; 4];
    for (index, (lx, ly)) in local.into_iter().enumerate() {
        let [x, y] = rotate(lx, ly, rect.angle_deg);
        out[index] = [rect.center_x + x, rect.center_y + y];
    }
    out
}

/// Normale sortante de l'arête, dans l'image. 0 haut, 1 droite, 2 bas, 3 gauche.
pub fn outward_normal(rect: &Rect, edge: usize) -> [f64; 2] {
    let (lx, ly) = match edge % 4 {
        0 => (0.0, -1.0),
        1 => (1.0, 0.0),
        2 => (0.0, 1.0),
        _ => (-1.0, 0.0),
    };
    rotate(lx, ly, rect.angle_deg)
}

/// Déplace une arête le long de sa normale. L'arête opposée reste en place.
pub fn move_edge(rect: &mut Rect, edge: usize, delta: f64, min_side: f64) {
    let changes_width = edge % 2 == 1;
    let side = if changes_width { rect.width } else { rect.height };
    let delta = if delta.is_finite() { delta.max(min_side - side) } else { 0.0 };
    let normal = outward_normal(rect, edge);
    if changes_width {
        rect.width = side + delta;
    } else {
        rect.height = side + delta;
    }
    rect.center_x += normal[0] * delta / 2.0;
    rect.center_y += normal[1] * delta / 2.0;
}

pub fn apply(image: &[f32], channels: usize, height: usize, width: usize, spec: &Spec, cancel: &AtomicBool) -> Result<(Vec<f32>, usize, usize), String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }
    if channels == 0 || height == 0 || width == 0 || image.len() < channels * height * width {
        return Err("image vide".to_string());
    }
    let out_w = spec.rect.width.round();
    let out_h = spec.rect.height.round();
    if !out_w.is_finite() || !out_h.is_finite() || out_w < 1.0 || out_h < 1.0 {
        return Err("le rectangle est trop petit".to_string());
    }
    let out_w = out_w as usize;
    let out_h = out_h as usize;
    let mut dst = vec![0.0f32; channels * out_h * out_w];
    for channel in 0..channels {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let src = &image[channel * height * width..(channel + 1) * height * width];
        let plane = &mut dst[channel * out_h * out_w..(channel + 1) * out_h * out_w];
        resample_plane(src, height, width, plane, out_h, out_w, &spec.rect, spec.interpolation, spec.clamp);
    }
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }
    Ok((dst, out_h, out_w))
}

fn rotate(lx: f64, ly: f64, angle_deg: f64) -> [f64; 2] {
    let radians = angle_deg.to_radians();
    let (sin, cos) = (radians.sin(), radians.cos());
    [lx * cos - ly * sin, lx * sin + ly * cos]
}

fn resample_plane(src: &[f32], src_h: usize, src_w: usize, dst: &mut [f32], dst_h: usize, dst_w: usize, rect: &Rect, interpolation: Interpolation, clamp: bool) {
    let radians = rect.angle_deg.to_radians();
    let (sin, cos) = (radians.sin(), radians.cos());
    let cx = rect.center_x - 0.5;
    let cy = rect.center_y - 0.5;
    let scale_x = rect.width / dst_w as f64;
    let scale_y = rect.height / dst_h as f64;
    let threads = std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1).clamp(1, 8).min(dst_h.max(1));
    let chunk = dst_h.div_ceil(threads);
    std::thread::scope(|scope| {
        for (band, rows) in dst.chunks_mut(chunk * dst_w).enumerate() {
            let y_start = band * chunk;
            scope.spawn(move || {
                for (local_y, row) in rows.chunks_mut(dst_w).enumerate() {
                    let y = y_start + local_y;
                    let ly = (y as f64 + 0.5) * scale_y - rect.height / 2.0;
                    for (x, pixel) in row.iter_mut().enumerate() {
                        let lx = (x as f64 + 0.5) * scale_x - rect.width / 2.0;
                        let sx = cx + lx * cos - ly * sin;
                        let sy = cy + lx * sin + ly * cos;
                        *pixel = sample(src, src_h, src_w, sx, sy, interpolation, clamp);
                    }
                }
            });
        }
    });
}

fn sample(src: &[f32], height: usize, width: usize, sx: f64, sy: f64, interpolation: Interpolation, clamp: bool) -> f32 {
    if !sx.is_finite() || !sy.is_finite() || sx < -0.5 || sy < -0.5 || sx > width as f64 - 0.5 || sy > height as f64 - 0.5 {
        return 0.0;
    }
    let (x0, x1) = tap_range(interpolation, sx);
    let (y0, y1) = tap_range(interpolation, sy);
    let mut acc = 0.0f64;
    let mut low = f32::MAX;
    let mut high = f32::MIN;
    let mut inside = false;
    for y in y0..=y1 {
        let wy = kernel(interpolation, sy - y as f64);
        if wy == 0.0 {
            continue;
        }
        for x in x0..=x1 {
            let wx = kernel(interpolation, sx - x as f64);
            if wx == 0.0 {
                continue;
            }
            let in_image = x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height;
            let value = if in_image { src[y as usize * width + x as usize] } else { 0.0 };
            acc += wx * wy * value as f64;
            if in_image {
                inside = true;
                low = low.min(value);
                high = high.max(value);
            }
        }
    }
    let mut value = acc as f32;
    if clamp && inside {
        value = value.clamp(low, high);
    }
    value
}

fn tap_range(interpolation: Interpolation, pos: f64) -> (isize, isize) {
    let base = pos.floor() as isize;
    match interpolation {
        Interpolation::Nearest => {
            let nearest = pos.round() as isize;
            (nearest, nearest)
        }
        Interpolation::Bilinear => (base, base + 1),
        Interpolation::Bicubic => (base - 1, base + 2),
        Interpolation::Lanczos4 => (base - 3, base + 4),
    }
}

fn kernel(interpolation: Interpolation, distance: f64) -> f64 {
    let distance = distance.abs();
    match interpolation {
        Interpolation::Nearest => {
            if distance <= 0.5 { 1.0 } else { 0.0 }
        }
        Interpolation::Bilinear => {
            if distance < 1.0 {
                1.0 - distance
            } else {
                0.0
            }
        }
        Interpolation::Bicubic => {
            if distance < 1.0 {
                (1.5 * distance - 2.5) * distance * distance + 1.0
            } else if distance < 2.0 {
                ((-0.5 * distance + 2.5) * distance - 4.0) * distance + 2.0
            } else {
                0.0
            }
        }
        Interpolation::Lanczos4 => lanczos4(distance),
    }
}

fn lanczos4(distance: f64) -> f64 {
    if distance < 1e-8 {
        return 1.0;
    }
    if distance >= 4.0 {
        return 0.0;
    }
    let pi_d = std::f64::consts::PI * distance;
    (pi_d.sin() / pi_d) * ((pi_d / 4.0).sin() / (pi_d / 4.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn full(width: usize, height: usize, angle: f64, interpolation: Interpolation, clamp: bool) -> Spec {
        Spec {
            rect: Rect { center_x: width as f64 / 2.0, center_y: height as f64 / 2.0, width: width as f64, height: height as f64, angle_deg: angle },
            interpolation,
            clamp,
        }
    }

    #[test]
    fn full_frame_keeps_every_pixel() {
        let width = 5;
        let height = 4;
        let mut image = vec![0.0f32; 2 * height * width];
        for index in 0..height * width {
            image[index] = index as f32 * 0.01;
            image[height * width + index] = 0.2 + index as f32 * 0.01;
        }
        for interpolation in Interpolation::ALL {
            let (out, out_h, out_w) = apply(&image, 2, height, width, &full(width, height, 0.0, interpolation, true), &idle()).unwrap();
            assert_eq!((out_h, out_w), (height, width));
            for (got, expected) in out.iter().zip(image.iter()) {
                assert!((got - expected).abs() < 1e-4, "{interpolation:?} {got} != {expected}");
            }
        }
    }

    #[test]
    fn clockwise_rectangle_straightens_the_opposite_way() {
        let size = 5;
        let mut image = vec![0.0f32; size * size];
        image[2 * size + 4] = 1.0;
        let spec = full(size, size, 90.0, Interpolation::Nearest, false);
        let (out, _, _) = apply(&image, 1, size, size, &spec, &idle()).unwrap();
        assert!((out[2] - 1.0).abs() < 1e-6, "le pixel de droite doit arriver en haut");
        let others: f32 = out.iter().enumerate().filter(|(index, _)| *index != 2).map(|(_, value)| value.abs()).sum();
        assert!(others < 1e-6);
    }

    #[test]
    fn bilinear_midpoint_is_the_average() {
        let image = [0.0f32, 1.0];
        let spec = Spec {
            rect: Rect { center_x: 1.0, center_y: 0.5, width: 1.0, height: 1.0, angle_deg: 0.0 },
            interpolation: Interpolation::Bilinear,
            clamp: false,
        };
        let (out, out_h, out_w) = apply(&image, 1, 1, 2, &spec, &idle()).unwrap();
        assert_eq!((out_h, out_w), (1, 1));
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sample_outside_the_image_is_zero() {
        let image = vec![1.0f32; 8 * 8];
        let spec = Spec {
            rect: Rect { center_x: -100.0, center_y: -100.0, width: 8.0, height: 8.0, angle_deg: 0.0 },
            interpolation: Interpolation::Nearest,
            clamp: false,
        };
        let (out, _, _) = apply(&image, 1, 8, 8, &spec, &idle()).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn lanczos_clamping_removes_the_ringing() {
        let size = 16;
        let mut image = vec![0.0f32; size * size];
        image[8 * size + 8] = 1.0;
        let mut spec = Spec {
            rect: Rect { center_x: 8.25, center_y: 8.0, width: size as f64, height: size as f64, angle_deg: 0.0 },
            interpolation: Interpolation::Lanczos4,
            clamp: false,
        };
        let (open, _, _) = apply(&image, 1, size, size, &spec, &idle()).unwrap();
        assert!(open.iter().any(|value| *value < -1e-4), "Lanczos-4 doit osciller sous zéro");
        spec.clamp = true;
        let (closed, _, _) = apply(&image, 1, size, size, &spec, &idle()).unwrap();
        assert!(closed.iter().all(|value| (-1e-5..=1.0 + 1e-5).contains(value)));
    }

    #[test]
    fn right_edge_grows_without_moving_the_left_side() {
        let mut rect = Rect { center_x: 10.0, center_y: 10.0, width: 20.0, height: 10.0, angle_deg: 0.0 };
        move_edge(&mut rect, 1, 4.0, 8.0);
        assert!((rect.width - 24.0).abs() < 1e-9);
        assert!((rect.center_x - 12.0).abs() < 1e-9);
        assert!((rect.center_x - rect.width / 2.0).abs() < 1e-9);
        assert!((rect.height - 10.0).abs() < 1e-9);
    }

    #[test]
    fn rotated_edge_moves_along_its_normal() {
        let mut rect = Rect { center_x: 10.0, center_y: 10.0, width: 20.0, height: 10.0, angle_deg: 90.0 };
        move_edge(&mut rect, 1, 4.0, 8.0);
        assert!((rect.width - 24.0).abs() < 1e-6);
        assert!((rect.center_x - 10.0).abs() < 1e-6);
        assert!((rect.center_y - 12.0).abs() < 1e-6);
    }

    #[test]
    fn square_corners_turn_clockwise() {
        let rect = Rect { center_x: 0.0, center_y: 0.0, width: 2.0, height: 2.0, angle_deg: 90.0 };
        let corners = corners(&rect);
        let expected = [[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]];
        for (got, want) in corners.iter().zip(expected) {
            assert!((got[0] - want[0]).abs() < 1e-9 && (got[1] - want[1]).abs() < 1e-9);
        }
    }
}
