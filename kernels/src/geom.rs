// SPDX-License-Identifier: GPL-3.0-or-later
//! Aire signée du drizzle carré (boxer), identique au noyau CUDA de GPU Stacker.

pub fn sgarea(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    if dx == 0.0 {
        return 0.0;
    }
    let negdx = dx < 0.0;
    let mut xlo = if negdx { x2 } else { x1 };
    let mut xhi = if negdx { x1 } else { x2 };
    if xlo >= 1.0 || xhi <= 0.0 {
        return 0.0;
    }
    xlo = xlo.max(0.0);
    xhi = xhi.min(1.0);
    let m = (y2 - y1) / dx;
    let c = y1 - m * x1;
    let mut ylo = m * xlo + c;
    let mut yhi = m * xhi + c;
    if ylo <= 0.0 && yhi <= 0.0 {
        return 0.0;
    }
    if ylo < 0.0 {
        ylo = 0.0;
        xlo = -c / m;
    }
    if yhi < 0.0 {
        yhi = 0.0;
        xhi = -c / m;
    }
    let a = if ylo >= 1.0 && yhi >= 1.0 {
        xhi - xlo
    } else if ylo <= 1.0 && yhi <= 1.0 {
        0.5 * (xhi - xlo) * (yhi + ylo)
    } else {
        let xtop = (1.0 - c) / m;
        if ylo <= 1.0 {
            0.5 * (xtop - xlo) * (1.0 + ylo) + xhi - xtop
        } else {
            0.5 * (xhi - xtop) * (1.0 + yhi) + xtop - xlo
        }
    };
    if negdx { -a } else { a }
}
