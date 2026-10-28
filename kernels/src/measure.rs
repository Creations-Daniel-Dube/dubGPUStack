// SPDX-License-Identifier: GPL-3.0-or-later
//! Moments d'une étoile dans une fenêtre de rayon 4. Même calcul que le noyau `measure`.

pub fn measure_one(residual: &[f32], luma: &[f32], width: i32, height: i32, x0: i32, y0: i32, thresh: f32, satlevel: f32) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    out[0] = -1.0;
    let rad = 4i32;
    let mut s = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut peak = 0.0f32;
    let mut lmax = 0.0f32;
    let mut dy = -rad;
    while dy <= rad {
        let y = y0 + dy;
        if y >= 0 && y < height {
            let mut dx = -rad;
            while dx <= rad {
                let x = x0 + dx;
                if x >= 0 && x < width && dx * dx + dy * dy <= rad * rad + 1 {
                    let index = (y * width + x) as usize;
                    let v = residual[index];
                    let lv = luma[index];
                    if lv > lmax {
                        lmax = lv;
                    }
                    if v > peak {
                        peak = v;
                    }
                    if v > thresh {
                        let vf = v as f64;
                        s += vf;
                        sx += vf * (x as f64 + 0.5);
                        sy += vf * (y as f64 + 0.5);
                    }
                }
                dx += 1;
            }
        }
        dy += 1;
    }
    if s <= 0.0 {
        return out;
    }
    let mx = sx / s;
    let my = sy / s;
    let mut mxx = 0.0f64;
    let mut myy = 0.0f64;
    let mut mxy = 0.0f64;
    let mut w = 0.0f64;
    dy = -rad;
    while dy <= rad {
        let y = y0 + dy;
        if y >= 0 && y < height {
            let mut dx = -rad;
            while dx <= rad {
                let x = x0 + dx;
                if x >= 0 && x < width && dx * dx + dy * dy <= rad * rad + 1 {
                    let v = residual[(y * width + x) as usize];
                    if v > thresh {
                        let vf = v as f64;
                        let ddx = x as f64 + 0.5 - mx;
                        let ddy = y as f64 + 0.5 - my;
                        mxx += vf * ddx * ddx;
                        myy += vf * ddy * ddy;
                        mxy += vf * ddx * ddy;
                        w += vf;
                    }
                }
                dx += 1;
            }
        }
        dy += 1;
    }
    if w <= 0.0 {
        return out;
    }
    mxx /= w;
    myy /= w;
    mxy /= w;
    let tr = mxx + myy;
    let det = mxx * myy - mxy * mxy;
    let disc = crate::fmath::sqrt_f64((tr * tr / 4.0 - det).max(0.0));
    let l1 = tr / 2.0 + disc;
    let l2 = (tr / 2.0 - disc).max(1e-9);
    out[0] = mx as f32;
    out[1] = my as f32;
    out[2] = s as f32;
    out[3] = (2.3548 * crate::fmath::sqrt_f64((l1 + l2) / 2.0)) as f32;
    out[4] = crate::fmath::sqrt_f64(l2 / l1) as f32;
    out[5] = peak;
    out[6] = if lmax >= satlevel { 1.0 } else { 0.0 };
    out[7] = w as f32;
    out
}
