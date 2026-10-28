// SPDX-License-Identifier: GPL-3.0-or-later
//! Arithmétique flottante disponible sur l'hôte et dans les noyaux PTX.

pub fn sqrt_f32(x: f32) -> f32 {
    if !(x > 0.0) {
        return 0.0;
    }
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        sqrt_f32_intrinsic(x)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        x.sqrt()
    }
}

pub fn sqrt_f64(x: f64) -> f64 {
    if !(x > 0.0) {
        return 0.0;
    }
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        sqrt_f64_intrinsic(x)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        x.sqrt()
    }
}

#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm.sqrt.f32"]
    fn sqrt_f32_intrinsic(x: f32) -> f32;
    #[link_name = "llvm.sqrt.f64"]
    fn sqrt_f64_intrinsic(x: f64) -> f64;
}

pub fn floor_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let truncated = x as i64 as f64;
    if x < 0.0 && truncated != x {
        truncated - 1.0
    } else {
        truncated
    }
}

pub fn ceil_f64(x: f64) -> f64 {
    let floored = floor_f64(x);
    if floored == x { floored } else { floored + 1.0 }
}

pub fn ramp_at(index: usize) -> f32 {
    let r = (index as f32 / 1000.0).clamp(0.0, 1.0);
    let rr = r * r;
    rr * r * (6.0 * rr - 15.0 * r + 10.0)
}

pub fn ramp_table() -> [f32; 1001] {
    let mut table = [0.0; 1001];
    let mut i = 0;
    while i < 1001 {
        table[i] = ramp_at(i);
        i += 1;
    }
    table
}

/// Poids de plume Siril : 0 au bord du masque, 1 au-delà de `feather` pixels de masque.
pub fn feather_weight(distance: f32, feather: f32, ramp: &[f32]) -> f32 {
    if !(feather > 0.0) {
        return 1.0;
    }
    if distance <= 0.0 {
        return 0.0;
    }
    if distance > feather {
        return 1.0;
    }
    let idx = (distance * (1.0 / feather) * 1000.0) as i32;
    let idx = idx.clamp(0, 1000) as usize;
    ramp.get(idx).copied().unwrap_or(1.0)
}
