// SPDX-License-Identifier: GPL-3.0-or-later
//! Enregistrement par triangles d'étoiles, ajustement robuste et chaînage mosaïque.

use std::collections::HashMap;

use super::stars::{FrameMetrics, StarList};

pub const TRI_STARS: usize = 45;
pub const TRI_NEIGHBOURS: usize = 6;
pub const INV_TOL: f64 = 0.006;
pub const MIN_INLIERS: usize = 12;

#[derive(Clone, Debug)]
pub struct Transform {
    pub m: [f64; 6],
}

impl Transform {
    pub fn identity() -> Self {
        Self { m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] }
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.m[0] * x + self.m[1] * y + self.m[2], self.m[3] * x + self.m[4] * y + self.m[5])
    }

    pub fn compose(&self, other: &Self) -> Self {
        let a = self.matrix3();
        let b = other.matrix3();
        let mut c = [0.0; 9];
        for r in 0..3 {
            for col in 0..3 {
                c[r * 3 + col] = a[r * 3] * b[col] + a[r * 3 + 1] * b[3 + col] + a[r * 3 + 2] * b[6 + col];
            }
        }
        Self { m: [c[0], c[1], c[2], c[3], c[4], c[5]] }
    }

    pub fn inverse(&self) -> Option<Self> {
        let a = self.matrix3();
        let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let mut o = [0.0; 9];
        o[0] = (a[4] * a[8] - a[5] * a[7]) * inv_det;
        o[1] = (a[2] * a[7] - a[1] * a[8]) * inv_det;
        o[2] = (a[1] * a[5] - a[2] * a[4]) * inv_det;
        o[3] = (a[5] * a[6] - a[3] * a[8]) * inv_det;
        o[4] = (a[0] * a[8] - a[2] * a[6]) * inv_det;
        o[5] = (a[2] * a[3] - a[0] * a[5]) * inv_det;
        o[6] = (a[3] * a[7] - a[4] * a[6]) * inv_det;
        o[7] = (a[1] * a[6] - a[0] * a[7]) * inv_det;
        o[8] = (a[0] * a[4] - a[1] * a[3]) * inv_det;
        Some(Self { m: [o[0], o[1], o[2], o[3], o[4], o[5]] })
    }

    pub fn rotation_deg(&self) -> f64 {
        self.m[3].atan2(self.m[0]).to_degrees()
    }

    pub fn scale(&self) -> f64 {
        (self.m[0] * self.m[4] - self.m[1] * self.m[3]).abs().sqrt()
    }

    fn matrix3(&self) -> [f64; 9] {
        [self.m[0], self.m[1], self.m[2], self.m[3], self.m[4], self.m[5], 0.0, 0.0, 1.0]
    }

    pub fn to_list(&self) -> Vec<f64> {
        self.m.to_vec()
    }

    pub fn from_list(v: &[f64]) -> Self {
        let mut m = [0.0; 6];
        for (slot, value) in m.iter_mut().zip(v.iter()) {
            *slot = *value;
        }
        Self { m }
    }
}

#[derive(Clone, Debug)]
pub struct Match {
    pub t: Transform,
    pub inliers: usize,
    pub rms: f64,
}

struct Tris {
    verts: Vec<[usize; 3]>,
    inv: Vec<[f64; 2]>,
    sign: Vec<f64>,
}

fn triangles(x: &[f64], y: &[f64]) -> Tris {
    let n = x.len();
    if n < 3 {
        return Tris { verts: Vec::new(), inv: Vec::new(), sign: Vec::new() };
    }
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            dist[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    let k = TRI_NEIGHBOURS.min(n - 1);
    let mut nn = vec![0usize; n * k];
    for i in 0..n {
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| dist[i * n + a].partial_cmp(&dist[i * n + b]).unwrap_or(std::cmp::Ordering::Equal));
        for (slot, index) in order.into_iter().take(k).enumerate() {
            nn[i * k + slot] = index;
        }
    }
    let mut set = std::collections::BTreeSet::new();
    for i in 0..n {
        for a in 0..k {
            for b in (a + 1)..k {
                let mut tri = [i, nn[i * k + a], nn[i * k + b]];
                tri.sort_unstable();
                set.insert(tri);
            }
        }
    }
    let mut verts = Vec::new();
    let mut inv = Vec::new();
    let mut sign = Vec::new();
    for tri in set {
        let sides = [
            dist[tri[1] * n + tri[2]],
            dist[tri[0] * n + tri[2]],
            dist[tri[0] * n + tri[1]],
        ];
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| sides[b].partial_cmp(&sides[a]).unwrap_or(std::cmp::Ordering::Equal));
        let ordered = [tri[order[0]], tri[order[1]], tri[order[2]]];
        let ss = [sides[order[0]], sides[order[1]], sides[order[2]]];
        if ss[2] > 8.0 && ss[0] - ss[1] > 0.01 * ss[0] && ss[1] - ss[2] > 0.01 * ss[0] {
            let (x0, y0) = (x[ordered[0]], y[ordered[0]]);
            let (x1, y1) = (x[ordered[1]], y[ordered[1]]);
            let (x2, y2) = (x[ordered[2]], y[ordered[2]]);
            let cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
            verts.push(ordered);
            inv.push([ss[1] / ss[0], ss[2] / ss[0]]);
            sign.push(cross.signum());
        }
    }
    Tris { verts, inv, sign }
}

fn similarity(src: &[[f64; 2]], dst: &[[f64; 2]]) -> [f64; 6] {
    if src.is_empty() {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    }
    let n = src.len() as f64;
    let ms = [src.iter().map(|p| p[0]).sum::<f64>() / n, src.iter().map(|p| p[1]).sum::<f64>() / n];
    let md = [dst.iter().map(|p| p[0]).sum::<f64>() / n, dst.iter().map(|p| p[1]).sum::<f64>() / n];
    let mut var = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (s, d) in src.iter().zip(dst.iter()) {
        let a = [s[0] - ms[0], s[1] - ms[1]];
        let b = [d[0] - md[0], d[1] - md[1]];
        var += a[0] * a[0] + a[1] * a[1];
        sxx += a[0] * b[0] + a[1] * b[1];
        sxy += a[0] * b[1] - a[1] * b[0];
    }
    if var <= 0.0 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    }
    let scale = (sxx * sxx + sxy * sxy).sqrt() / var;
    let th = sxy.atan2(sxx);
    let c = scale * th.cos();
    let s = scale * th.sin();
    let tx = md[0] - (c * ms[0] - s * ms[1]);
    let ty = md[1] - (s * ms[0] + c * ms[1]);
    [c, -s, tx, s, c, ty]
}

fn affine(src: &[[f64; 2]], dst: &[[f64; 2]]) -> [f64; 6] {
    if src.len() < 3 {
        return similarity(src, dst);
    }
    // Moindres carrés : [x y 1] * M = destination, résolu par les équations normales.
    let mut ata = [0.0f64; 9];
    let mut atb = [0.0f64; 6];
    for (s, d) in src.iter().zip(dst.iter()) {
        let row = [s[0], s[1], 1.0];
        for r in 0..3 {
            for c in 0..3 {
                ata[r * 3 + c] += row[r] * row[c];
            }
            atb[r] += row[r] * d[0];
            atb[3 + r] += row[r] * d[1];
        }
    }
    let Some(solx) = solve3(&ata, &[atb[0], atb[1], atb[2]]) else { return similarity(src, dst) };
    let Some(soly) = solve3(&ata, &[atb[3], atb[4], atb[5]]) else { return similarity(src, dst) };
    [solx[0], solx[1], solx[2], soly[0], soly[1], soly[2]]
}

fn solve3(a: &[f64; 9], b: &[f64; 3]) -> Option<[f64; 3]> {
    let mut m = [a[0], a[1], a[2], b[0], a[3], a[4], a[5], b[1], a[6], a[7], a[8], b[2]];
    for col in 0..3 {
        let mut pivot = col;
        for row in col + 1..3 {
            if m[row * 4 + col].abs() > m[pivot * 4 + col].abs() {
                pivot = row;
            }
        }
        if m[pivot * 4 + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..4 {
                m.swap(col * 4 + k, pivot * 4 + k);
            }
        }
        let div = m[col * 4 + col];
        for k in col..4 {
            m[col * 4 + k] /= div;
        }
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = m[row * 4 + col];
            for k in col..4 {
                m[row * 4 + k] -= factor * m[col * 4 + k];
            }
        }
    }
    Some([m[3], m[7], m[11]])
}

fn pairs(pa: &[[f64; 2]], pb: &[[f64; 2]], radius: f64) -> Vec<(usize, usize)> {
    if pa.is_empty() || pb.is_empty() {
        return Vec::new();
    }
    let mut best_b = vec![0usize; pa.len()];
    for (ia, a) in pa.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = f64::MAX;
        for (ib, b) in pb.iter().enumerate() {
            let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
            if d < best_d {
                best_d = d;
                best = ib;
            }
        }
        best_b[ia] = best;
    }
    let mut best_a = vec![0usize; pb.len()];
    for (ib, b) in pb.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = f64::MAX;
        for (ia, a) in pa.iter().enumerate() {
            let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
            if d < best_d {
                best_d = d;
                best = ia;
            }
        }
        best_a[ib] = best;
    }
    let mut out = Vec::new();
    let limit = radius * radius;
    for ia in 0..pa.len() {
        let ib = best_b[ia];
        let d = (pa[ia][0] - pb[ib][0]).powi(2) + (pa[ia][1] - pb[ib][1]).powi(2);
        if best_a[ib] == ia && d <= limit {
            out.push((ia, ib));
        }
    }
    out
}

fn points(stars: &StarList) -> Vec<[f64; 2]> {
    stars.x.iter().zip(stars.y.iter()).map(|(x, y)| [*x, *y]).collect()
}

pub fn refine(a: &StarList, b: &StarList, t0: &Transform, model: &str, radii: &[f64]) -> Option<Match> {
    let pa = points(a);
    let pb = points(b);
    let mut t = t0.clone();
    let mut last = Vec::new();
    for radius in radii {
        let moved: Vec<[f64; 2]> = pa.iter().map(|p| {
            let (x, y) = t.apply(p[0], p[1]);
            [x, y]
        }).collect();
        let matched = pairs(&moved, &pb, *radius);
        if matched.len() < MIN_INLIERS.max(4) / 2 {
            return None;
        }
        let src: Vec<[f64; 2]> = matched.iter().map(|(ia, _)| pa[*ia]).collect();
        let dst: Vec<[f64; 2]> = matched.iter().map(|(_, ib)| pb[*ib]).collect();
        let m = if model == "affine" && matched.len() >= 12 { affine(&src, &dst) } else { similarity(&src, &dst) };
        t = Transform { m };
        last = matched;
    }
    if last.is_empty() {
        return None;
    }
    let mut acc = 0.0;
    for (ia, ib) in &last {
        let (x, y) = t.apply(pa[*ia][0], pa[*ia][1]);
        let dx = x - pb[*ib][0];
        let dy = y - pb[*ib][1];
        acc += dx * dx + dy * dy;
    }
    Some(Match { inliers: last.len(), rms: (acc / last.len() as f64).sqrt(), t })
}

pub fn match_stars(a: &StarList, b: &StarList, model: &str, max_shift: Option<f64>) -> Option<Match> {
    if a.len() < 6 || b.len() < 6 {
        return None;
    }
    let sa = a.brightest(TRI_STARS);
    let sb = b.brightest(TRI_STARS);
    let ta = triangles(&sa.x, &sa.y);
    let tb = triangles(&sb.x, &sb.y);
    if ta.inv.is_empty() || tb.inv.is_empty() {
        return None;
    }
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for i in 0..ta.inv.len() {
        for j in 0..tb.inv.len() {
            let dx = ta.inv[i][0] - tb.inv[j][0];
            let dy = ta.inv[i][1] - tb.inv[j][1];
            if (dx * dx + dy * dy).sqrt() < INV_TOL && ta.sign[i] == tb.sign[j] {
                ia.push(i);
                ib.push(j);
            }
        }
    }
    if ia.is_empty() {
        return None;
    }
    let pa = points(&sa);
    let pb = points(&sb);
    let mut rot = Vec::new();
    let mut tx0 = Vec::new();
    let mut ty0 = Vec::new();
    let mut which = Vec::new();
    for (n, (&i, &j)) in ia.iter().zip(ib.iter()).enumerate() {
        let src = [pa[ta.verts[i][0]], pa[ta.verts[i][1]], pa[ta.verts[i][2]]];
        let dst = [pb[tb.verts[j][0]], pb[tb.verts[j][1]], pb[tb.verts[j][2]]];
        let found = Transform::from_list(&similarity(&src, &dst));
        let scale = found.scale();
        if !(0.9..1.1).contains(&scale) {
            continue;
        }
        let angle = found.rotation_deg();
        let tx = found.m[2];
        let ty = found.m[5];
        if let Some(limit) = max_shift {
            if tx.abs() >= limit || ty.abs() >= limit {
                continue;
            }
        }
        rot.push(angle);
        tx0.push(tx);
        ty0.push(ty);
        which.push(n);
    }
    if rot.is_empty() {
        return None;
    }
    let cx = median(&sa.x);
    let cy = median(&sa.y);
    let mut bins: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for i in 0..rot.len() {
        let th = rot[i].to_radians();
        let tx = tx0[i] + th.cos() * cx - th.sin() * cy;
        let ty = ty0[i] + th.sin() * cx + th.cos() * cy;
        let key = ((rot[i] / 2.0).round() as i64, (tx / 8.0).round() as i64, (ty / 8.0).round() as i64);
        bins.entry(key).or_default().push(i);
    }
    let mut ranked: Vec<Vec<usize>> = bins.into_values().collect();
    ranked.sort_by_key(|members| std::cmp::Reverse(members.len()));
    let a_c = a.brightest(250);
    let b_c = b.brightest(350);
    let mut best: Option<Match> = None;
    for members in ranked.into_iter().take(4) {
        if members.len() < 2 && best.is_some() {
            break;
        }
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for member in &members {
            let n = which[*member];
            let i = ia[n];
            let j = ib[n];
            for k in 0..3 {
                src.push(pa[ta.verts[i][k]]);
                dst.push(pb[tb.verts[j][k]]);
            }
        }
        let t0 = Transform { m: similarity(&src, &dst) };
        if let Some(found) = refine(&a_c, &b_c, &t0, model, &[5.0, 3.0, 2.0, 1.5]) {
            if found.inliers >= MIN_INLIERS && found.rms < 2.0 && best.as_ref().is_none_or(|current| found.inliers > current.inliers) {
                let enough = found.inliers as f64 >= 0.3 * a_c.len().min(b_c.len()) as f64;
                best = Some(found);
                if enough {
                    break;
                }
            }
        }
    }
    let best = best?;
    refine(&a.brightest(600), b, &best.t, model, &[2.5, 1.5]).filter(|found| found.inliers >= MIN_INLIERS).or(Some(best))
}

fn median(values: &[f64]) -> f64 {
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if copy.is_empty() {
        return 0.0;
    }
    let n = copy.len();
    if n % 2 == 1 { copy[n / 2] } else { 0.5 * (copy[n / 2 - 1] + copy[n / 2]) }
}

#[derive(Clone, Debug, Default)]
pub struct RegResult {
    pub transforms: HashMap<usize, Transform>,
    pub info: HashMap<usize, (usize, f64, String)>,
    pub failed: HashMap<usize, String>,
}

pub fn pick_reference(metrics: &HashMap<usize, FrameMetrics>) -> Option<usize> {
    let mut best = None;
    let mut score = f64::NEG_INFINITY;
    for (index, metric) in metrics {
        if metric.nstars >= 20 && metric.fwhm.is_finite() && metric.fwhm > 0.0 {
            let s = metric.nstars as f64 / (metric.fwhm * metric.fwhm) * (0.5 + 0.5 * metric.roundness);
            if s > score {
                score = s;
                best = Some(*index);
            }
        }
    }
    best.or_else(|| metrics.iter().max_by_key(|(_, metric)| metric.nstars).map(|(index, _)| *index))
}

pub fn register_all(stars: &HashMap<usize, StarList>, reference: usize, model: &str, cancel: &dyn Fn() -> bool) -> RegResult {
    let mut result = RegResult::default();
    let Some(reference_stars) = stars.get(&reference).filter(|list| !list.is_empty()) else { return result };
    result.transforms.insert(reference, Transform::identity());
    result.info.insert(reference, (reference_stars.len(), 0.0, "reference".to_string()));
    let mut pending = Vec::new();
    for index in stars.keys().copied().filter(|index| *index != reference) {
        if cancel() {
            return result;
        }
        if let Some(found) = match_stars(&stars[&index], reference_stars, model, None) {
            result.transforms.insert(index, found.t);
            result.info.insert(index, (found.inliers, found.rms, "reference".to_string()));
        } else {
            pending.push(index);
        }
    }
    let mut changed = true;
    while !pending.is_empty() && changed {
        changed = false;
        let mut still = Vec::new();
        for index in pending {
            if cancel() {
                return result;
            }
            let mut near: Vec<usize> = result.transforms.keys().copied().collect();
            near.sort_by_key(|other| (*other as i64 - index as i64).unsigned_abs());
            let mut ok = false;
            for other in near.into_iter().take(8) {
                if other == reference {
                    continue;
                }
                if let Some(found) = match_stars(&stars[&index], &stars[&other], model, None) {
                    let chained = result.transforms[&other].compose(&found.t);
                    result.transforms.insert(index, chained);
                    result.info.insert(index, (found.inliers, found.rms, format!("via image {}", other + 1)));
                    ok = true;
                    changed = true;
                    break;
                }
            }
            if !ok {
                still.push(index);
            }
        }
        pending = still;
    }
    for index in pending {
        result.failed.insert(index, "impossible d'apparier ses étoiles aux autres images".to_string());
    }
    if result.info.values().any(|(_, _, via)| via != "reference") {
        polish(stars, &mut result, model);
    }
    result
}

fn polish(stars: &HashMap<usize, StarList>, result: &mut RegResult, model: &str) {
    for _ in 0..2 {
        let catalog = catalog(stars, result);
        let chained: Vec<usize> = result.info.iter().filter(|(_, (_, _, via))| via != "reference").map(|(index, _)| *index).collect();
        for index in chained {
            let Some(current) = result.transforms.get(&index).cloned() else { continue };
            if let Some(found) = refine(&stars[&index], &catalog, &current, model, &[3.0, 2.0, 1.5]) {
                if found.inliers >= MIN_INLIERS {
                    let via = result.info.get(&index).map(|(_, _, via)| via.clone()).unwrap_or_default();
                    result.transforms.insert(index, found.t);
                    result.info.insert(index, (found.inliers, found.rms, via));
                }
            }
        }
    }
}

fn catalog(stars: &HashMap<usize, StarList>, result: &RegResult) -> StarList {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut flux = Vec::new();
    for index in result.transforms.keys() {
        let list = stars[index].brightest(200);
        for i in 0..list.len() {
            let (xx, yy) = result.transforms[index].apply(list.x[i], list.y[i]);
            x.push(xx);
            y.push(yy);
            flux.push(list.flux[i]);
        }
    }
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| flux[b].partial_cmp(&flux[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut seen = std::collections::BTreeSet::new();
    let mut keep = Vec::new();
    for index in order {
        let key = ((x[index] / 2.0).round() as i64, (y[index] / 2.0).round() as i64);
        if seen.insert(key) {
            keep.push(index);
        }
    }
    StarList {
        x: keep.iter().map(|i| x[*i]).collect(),
        y: keep.iter().map(|i| y[*i]).collect(),
        flux: keep.iter().map(|i| flux[*i]).collect(),
        fwhm: vec![0.0; keep.len()],
        roundness: vec![0.0; keep.len()],
        saturated: vec![false; keep.len()],
    }
}
