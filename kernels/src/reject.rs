// SPDX-License-Identifier: GPL-3.0-or-later
//! Rejet et moyenne d'un pixel, port de Siril 1.4.4 (`reference.py`).
//!
//! `abort_zero_median` reproduit l'oracle CPU : en sigma, une première médiane
//! exactement nulle vide le pixel. Le chemin GPU de production le laisse à faux.

pub const REJECT_NONE: i32 = 0;
pub const REJECT_SIGMA: i32 = 1;
pub const REJECT_WINSORIZED: i32 = 2;

pub struct Combined {
    pub value: f32,
    pub nlo: i32,
    pub nhi: i32,
    pub nvalid: i32,
    pub kept: i32,
}

const NETS: [&[(usize, usize)]; 9] = [
    &[],
    &[],
    &[(0, 1)],
    &[(0, 1), (1, 2), (0, 1)],
    &[(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)],
    &[(0, 1), (2, 3), (1, 3), (2, 4), (0, 2), (1, 4), (1, 2), (3, 4), (2, 3)],
    &[(0, 1), (2, 3), (4, 5), (0, 2), (3, 5), (1, 4), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (2, 3)],
    &[
        (1, 2),
        (3, 4),
        (5, 6),
        (0, 2),
        (4, 6),
        (3, 5),
        (2, 6),
        (1, 5),
        (0, 4),
        (2, 5),
        (0, 3),
        (2, 4),
        (1, 3),
        (0, 1),
        (2, 3),
        (4, 5),
    ],
    &[
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (1, 2),
        (5, 6),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (2, 4),
        (3, 5),
        (1, 2),
        (3, 4),
        (5, 6),
    ],
];

pub fn quickmedian(a: &mut [f32], n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    if n < 9 {
        if n == 1 {
            return a[0];
        }
        for &(i, j) in NETS[n] {
            if a[i] > a[j] {
                a.swap(i, j);
            }
        }
        let k = n / 2;
        return if n % 2 == 0 { ((a[k - 1] + a[k]) as f64 / 2.0) as f32 } else { a[k] };
    }
    let k = n / 2;
    let mut left = 0usize;
    let mut right = n - 1;
    while left < right {
        let mut pindex = (left + right) / 2;
        let pivot = a[pindex];
        a[pindex] = a[right];
        a[right] = pivot;
        pindex = left;
        let mut i = left;
        while i < right {
            if a[i] < pivot {
                a.swap(pindex, i);
                pindex += 1;
            }
            i += 1;
        }
        a[right] = a[pindex];
        a[pindex] = pivot;
        if pindex < k {
            left = pindex + 1;
        } else {
            right = pindex;
        }
    }
    if n % 2 == 0 {
        ((a[k - 1] as f64 + a[k] as f64) / 2.0) as f32
    } else {
        a[k]
    }
}

pub fn sd(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for value in a {
        acc += *value as f64;
    }
    let mean = (acc / n as f64) as f32;
    acc = 0.0;
    for value in a {
        let d = *value - mean;
        acc += (d * d) as f64;
    }
    let var = (acc / (n as f64 - 1.0)) as f32;
    if var > 0.0 { crate::fmath::sqrt_f32(var) } else { 0.0 }
}

fn clip(pixel: f32, sigma: f32, siglow: f32, sighigh: f32, median: f32) -> i32 {
    if median - pixel > sigma * siglow {
        return -1;
    }
    if pixel - median > sigma * sighigh {
        return 1;
    }
    0
}

fn reject_in_place(stack: &mut [f32], mode: i32, siglow: f32, sighigh: f32, tmp: &mut [f32], out: &mut [f32], abort_zero_median: bool, max_winsor: u32) -> (usize, i32, i32) {
    let mut n = stack.len();
    if n <= 1 || mode == REJECT_NONE {
        return (n, 0, 0);
    }
    let mut nlo = 0i32;
    let mut nhi = 0i32;
    let mut r = 0i32;
    if mode == REJECT_SIGMA {
        let mut median = quickmedian(stack, n);
        if abort_zero_median && median == 0.0 {
            return (0, 0, 0);
        }
        let mut first = true;
        loop {
            let var = sd(&stack[..n]);
            if !first {
                median = quickmedian(&mut stack[..n], n);
            }
            first = false;
            let mut m = 0usize;
            let mut i = 0usize;
            while i < n {
                let mut rej = 0;
                if n as i32 - r > 4 {
                    rej = clip(stack[i], var, siglow, sighigh, median);
                    if rej != 0 {
                        r += 1;
                        if rej < 0 { nlo += 1; } else { nhi += 1; }
                    }
                }
                if rej == 0 {
                    out[m] = stack[i];
                    m += 1;
                }
                i += 1;
            }
            let changed = m != n;
            stack[..m].copy_from_slice(&out[..m]);
            n = m;
            if !(changed && n > 3) {
                break;
            }
        }
        return (n, nlo, nhi);
    }
    loop {
        let mut sigma = sd(&stack[..n]);
        let medianf = quickmedian(&mut stack[..n], n);
        tmp[..n].copy_from_slice(&stack[..n]);
        let mut guard = 0u32;
        loop {
            let m0 = medianf - 1.5 * sigma;
            let m1 = medianf + 1.5 * sigma;
            let mut i = 0;
            while i < n {
                let x = tmp[i];
                tmp[i] = if x < m0 { m0 } else if x > m1 { m1 } else { x };
                i += 1;
            }
            let sigma0 = sigma;
            sigma = 1.134 * sd(&tmp[..n]);
            guard += 1;
            if !((sigma - sigma0).abs() > sigma0 * 0.0005) || guard >= max_winsor {
                break;
            }
        }
        let mut m = 0usize;
        let mut i = 0usize;
        while i < n {
            let mut rej = 0;
            if n as i32 - r > 4 {
                rej = clip(stack[i], sigma, siglow, sighigh, medianf);
                if rej != 0 {
                    r += 1;
                    if rej < 0 { nlo += 1; } else { nhi += 1; }
                }
            }
            if rej == 0 {
                out[m] = stack[i];
                m += 1;
            }
            i += 1;
        }
        let changed = m != n;
        stack[..m].copy_from_slice(&out[..m]);
        n = m;
        if !(changed && n > 3) {
            break;
        }
    }
    (n, nlo, nhi)
}

#[allow(clippy::too_many_arguments)]
pub fn combine(values: &[f32], weights: &[f32], mode: i32, siglow: f32, sighigh: f32, work: &mut [f32], tmp: &mut [f32], out: &mut [f32], plain: bool, abort_zero_median: bool, max_winsor: u32) -> Combined {
    let n = values.len();
    let nvalid = n as i32;
    if n == 0 {
        return Combined { value: 0.0, nlo: 0, nhi: 0, nvalid: 0, kept: 0 };
    }
    work[..n].copy_from_slice(values);
    let (kept, nlo, nhi) = reject_in_place(&mut work[..n], mode, siglow, sighigh, tmp, out, abort_zero_median, max_winsor);
    if kept == 0 {
        return Combined { value: 0.0, nlo, nhi, nvalid, kept: 0 };
    }
    if plain {
        let mut sum = 0.0f64;
        let mut i = 0;
        while i < kept {
            sum += work[i] as f64;
            i += 1;
        }
        return Combined { value: (sum / kept as f64) as f32, nlo, nhi, nvalid, kept: kept as i32 };
    }
    let mut pmin = work[0];
    let mut pmax = work[0];
    let mut i = 1;
    while i < kept {
        pmin = pmin.min(work[i]);
        pmax = pmax.max(work[i]);
        i += 1;
    }
    let mut sum = 0.0f64;
    let mut norm = 0.0f64;
    i = 0;
    while i < n {
        let v = values[i];
        if v >= pmin && v <= pmax && v != 0.0 {
            let w = weights[i] as f64;
            sum += v as f64 * w;
            norm += w;
        }
        i += 1;
    }
    let value = if norm == 0.0 || sum == 0.0 {
        let mut positive = 0.0f64;
        i = 0;
        while i < n {
            let v = values[i];
            if v >= pmin && v <= pmax && v > 0.0 {
                positive += v as f64;
            }
            i += 1;
        }
        (positive / kept as f64) as f32
    } else {
        (sum / norm) as f32
    };
    Combined { value, nlo, nhi, nvalid, kept: kept as i32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quickmedian_matches_the_sorted_middle() {
        let mut even = [1.0, 4.0, 2.0, 3.0];
        assert!((quickmedian(&mut even, 4) - 2.5).abs() < 1e-6);
        let mut odd = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert!((quickmedian(&mut odd, 5) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn sigma_drops_a_hot_pixel_and_keeps_four() {
        let values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0];
        let weights = [1.0; 16];
        let mut work = [0.0; 16];
        let mut tmp = [0.0; 16];
        let mut out = [0.0; 16];
        let combined = combine(&values, &weights, REJECT_SIGMA, 3.0, 3.0, &mut work, &mut tmp, &mut out, true, true, 1000);
        assert!(combined.value < 2.0, "{}", combined.value);
        assert!(combined.kept >= 4);
    }
}
