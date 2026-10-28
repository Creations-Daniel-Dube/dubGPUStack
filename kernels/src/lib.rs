// SPDX-License-Identifier: GPL-3.0-or-later
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", allow(internal_features))]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx, stdarch_nvptx, link_llvm_intrinsics, asm_experimental_arch))]

pub mod fmath;
pub mod geom;
pub mod measure;
pub mod reject;

#[cfg(target_arch = "nvptx64")]
mod device {
    use crate::fmath::{ceil_f64, feather_weight, floor_f64};
    use crate::geom::sgarea;
    use crate::measure::measure_one;
    use crate::reject::combine;
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};

    unsafe extern "C" {
        #[link_name = "llvm.nvvm.atomic.load.add.f32.p0"]
        fn atomic_add_f32(ptr: *mut f32, val: f32) -> f32;
    }

    fn atomic_add_i32(ptr: *mut i32, val: i32) {
        if val == 0 {
            return;
        }
        unsafe {
            core::arch::asm!(
                "atom.global.add.s32 {0}, [{1}], {2};",
                out(reg32) _,
                in(reg64) ptr,
                in(reg32) val,
                options(nostack)
            );
        }
    }

    fn global_x() -> u32 {
        let thread = unsafe { _thread_idx_x() };
        let block = unsafe { _block_idx_x() };
        let width = unsafe { _block_dim_x() };
        block.wrapping_mul(width).wrapping_add(thread)
    }

    fn floor_f32(x: f32) -> f32 {
        if !x.is_finite() {
            return x;
        }
        let truncated = x as i32 as f32;
        if x < 0.0 && truncated != x { truncated - 1.0 } else { truncated }
    }

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn fill_bits(dst: *mut u32, n: i32, bits: u32) {
        let gid = global_x() as i32;
        if gid >= n {
            return;
        }
        unsafe { *dst.add(gid as usize) = bits; }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn drizzle(
        img: *const f32,
        width: i32,
        nrows: i32,
        y0: i32,
        cfa: i32,
        l00: i32,
        l01: i32,
        l10: i32,
        l11: i32,
        only_ch: i32,
        coef: *const f64,
        out_w: i32,
        out_h: i32,
        origin_x: i32,
        origin_y: i32,
        num: *mut f32,
        den: *mut f32,
    ) {
        let gid = global_x() as i32;
        let n = width.wrapping_mul(nrows);
        if gid >= n || width <= 0 || out_w <= 0 || out_h <= 0 {
            return;
        }
        let x = gid % width;
        let row = gid / width;
        let y = y0 + row;
        let mut ch = 0i32;
        if cfa != 0 {
            let k = ((y & 1) << 1) | (x & 1);
            ch = if k == 0 { l00 } else if k == 1 { l01 } else if k == 2 { l10 } else { l11 };
            if only_ch >= 0 && ch != only_ch {
                return;
            }
        }
        let plane = if only_ch >= 0 || cfa == 0 { 0 } else { ch };
        let v = unsafe { *img.add(gid as usize) };
        let (a, b, c, d, e, f, half) = unsafe { (*coef, *coef.add(1), *coef.add(2), *coef.add(3), *coef.add(4), *coef.add(5), *coef.add(6)) };
        let cx = x as f64 + 0.5;
        let cy = y as f64 + 0.5;
        let px = [cx - half, cx + half, cx + half, cx - half];
        let py = [cy - half, cy - half, cy + half, cy + half];
        let mut qx = [0.0f64; 4];
        let mut qy = [0.0f64; 4];
        let mut minx = 1.0e300f64;
        let mut maxx = -1.0e300f64;
        let mut miny = 1.0e300f64;
        let mut maxy = -1.0e300f64;
        let mut k = 0;
        while k < 4 {
            qx[k] = a * px[k] + b * py[k] + c;
            qy[k] = d * px[k] + e * py[k] + f;
            minx = minx.min(qx[k]);
            maxx = maxx.max(qx[k]);
            miny = miny.min(qy[k]);
            maxy = maxy.max(qy[k]);
            k += 1;
        }
        let u0 = floor_f64(minx) as i32;
        let u1 = ceil_f64(maxx) as i32 - 1;
        let v0 = floor_f64(miny) as i32;
        let v1 = ceil_f64(maxy) as i32 - 1;
        let u0 = u0.max(origin_x);
        let v0 = v0.max(origin_y);
        let u1 = u1.min(origin_x + out_w - 1);
        let v1 = v1.min(origin_y + out_h - 1);
        if u0 > u1 || v0 > v1 {
            return;
        }
        let stride = (out_w as usize).wrapping_mul(out_h as usize);
        let nump = unsafe { num.add(plane as usize * stride) };
        let denp = unsafe { den.add(plane as usize * stride) };
        let mut vv = v0;
        while vv <= v1 {
            let mut uu = u0;
            while uu <= u1 {
                let mut s = 0.0f32;
                let mut edge = 0;
                while edge < 4 {
                    let next = (edge + 1) & 3;
                    s += sgarea((qx[edge] - uu as f64) as f32, (qy[edge] - vv as f64) as f32, (qx[next] - uu as f64) as f32, (qy[next] - vv as f64) as f32);
                    edge += 1;
                }
                let area = if s < 0.0 { -s } else { s };
                if area > 0.0 {
                    let local = ((vv - origin_y) * out_w + (uu - origin_x)) as usize;
                    unsafe {
                        atomic_add_f32(nump.add(local), area * v);
                        atomic_add_f32(denp.add(local), area);
                    }
                }
                uu += 1;
            }
            vv += 1;
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn finish_prep(
        num: *const f32,
        den: *const f32,
        pixels: i32,
        tile_w: i32,
        coef: *const f64,
        feather: f32,
        ramp: *const f32,
        mask: *const f32,
        mask_w: i32,
        mask_h: i32,
        frame_w: i32,
        frame_h: i32,
        origin_x: i32,
        origin_y: i32,
        frame_x0: i32,
        frame_y0: i32,
        dst_v: *mut f32,
        dst_w: *mut f32,
        slot: i32,
        k: i32,
        has_mask: i32,
    ) {
        let p = global_x() as i32;
        if p >= pixels || k <= 0 || tile_w <= 0 {
            return;
        }
        let wden = unsafe { *den.add(p as usize) };
        let q = floor_f32(wden.min(1.0) * 255.0 + 0.5) as i32;
        let base = (p as usize).wrapping_mul(k as usize).wrapping_add(slot as usize);
        if q <= 0 || !(wden > 0.0) {
            unsafe {
                *dst_v.add(base) = f32::from_bits(0x7fc0_0000);
                *dst_w.add(base) = 0.0;
            }
            return;
        }
        let v = unsafe { *num.add(p as usize) } / wden;
        let a = unsafe { *coef };
        let off = unsafe { *coef.add(1) };
        let imgw = unsafe { *coef.add(2) };
        let nv = (v as f64 * a - off) as f32;
        if v == 0.0 || !v.is_finite() || nv == 0.0 || !nv.is_finite() {
            unsafe {
                *dst_v.add(base) = f32::from_bits(0x7fc0_0000);
                *dst_w.add(base) = 0.0;
            }
            return;
        }
        let x = p % tile_w;
        let y = p / tile_w;
        let lx = (origin_x + x - frame_x0) as f32;
        let ly = (origin_y + y - frame_y0) as f32;
        let fw = if has_mask != 0 && feather > 0.0 && frame_w > 0 && frame_h > 0 && mask_w > 0 && mask_h > 0 {
            let fx = mask_w as f32 / frame_w as f32;
            let fy = mask_h as f32 / frame_h as f32;
            let sx = ((lx + 0.5) * fx - 0.5).clamp(0.0, (mask_w - 1) as f32);
            let sy = ((ly + 0.5) * fy - 0.5).clamp(0.0, (mask_h - 1) as f32);
            let x0 = sx as i32;
            let y0 = sy as i32;
            let x1 = (x0 + 1).min(mask_w - 1);
            let y1 = (y0 + 1).min(mask_h - 1);
            let tx = sx - x0 as f32;
            let ty = sy - y0 as f32;
            let m00 = unsafe { *mask.add((y0 * mask_w + x0) as usize) };
            let m10 = unsafe { *mask.add((y0 * mask_w + x1) as usize) };
            let m01 = unsafe { *mask.add((y1 * mask_w + x0) as usize) };
            let m11 = unsafe { *mask.add((y1 * mask_w + x1) as usize) };
            let top = m00 * (1.0 - tx) + m10 * tx;
            let bot = m01 * (1.0 - tx) + m11 * tx;
            let distance = top * (1.0 - ty) + bot * ty;
            let ramp_s = unsafe { core::slice::from_raw_parts(ramp, 1001) };
            feather_weight(distance, feather, ramp_s)
        } else if feather > 0.0 && has_mask == 0 {
            1.0
        } else {
            1.0
        };
        let weight = (q as f32) * fw * imgw as f32;
        unsafe {
            *dst_v.add(base) = nv;
            *dst_w.add(base) = weight;
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn reject_mean(
        values: *const f32,
        weights: *const f32,
        k: i32,
        pixels: i32,
        mode: i32,
        siglow: f32,
        sighigh: f32,
        plain: i32,
        outv: *mut f32,
        nlo: *mut i32,
        nhi: *mut i32,
        nvalid: *mut i32,
        scratch_v: *mut f32,
        scratch_w: *mut f32,
        scratch_a: *mut f32,
        scratch_b: *mut f32,
        scratch_c: *mut f32,
    ) {
        let p = global_x() as usize;
        if p >= pixels as usize || k <= 0 {
            return;
        }
        let kk = k as usize;
        let src = p.wrapping_mul(kk);
        let mut n = 0usize;
        let mut i = 0usize;
        while i < kk {
            let v = unsafe { *values.add(src + i) };
            if !v.is_nan() {
                unsafe {
                    *scratch_v.add(src + n) = v;
                    *scratch_w.add(src + n) = *weights.add(src + i);
                }
                n += 1;
            }
            i += 1;
        }
        let values_s = unsafe { core::slice::from_raw_parts(scratch_v.add(src), n) };
        let weights_s = unsafe { core::slice::from_raw_parts(scratch_w.add(src), n) };
        let work = unsafe { core::slice::from_raw_parts_mut(scratch_a.add(src), n) };
        let tmp = unsafe { core::slice::from_raw_parts_mut(scratch_b.add(src), n) };
        let kept = unsafe { core::slice::from_raw_parts_mut(scratch_c.add(src), n) };
        let result = combine(values_s, weights_s, mode, siglow, sighigh, work, tmp, kept, plain != 0, false, 1000);
        unsafe {
            *outv.add(p) = result.value;
            if result.nlo != 0 {
                atomic_add_i32(nlo, result.nlo);
            }
            if result.nhi != 0 {
                atomic_add_i32(nhi, result.nhi);
            }
            if result.nvalid != 0 {
                atomic_add_i32(nvalid, result.nvalid);
            }
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "ptx-kernel" fn measure(
        residual: *const f32,
        luma: *const f32,
        width: i32,
        height: i32,
        xs: *const i32,
        ys: *const i32,
        n: i32,
        thresh: f32,
        satlevel: f32,
        out: *mut f32,
    ) {
        let k = global_x() as i32;
        if k >= n || width <= 0 || height <= 0 {
            return;
        }
        let pixels = (width as usize).wrapping_mul(height as usize);
        let residual_s = unsafe { core::slice::from_raw_parts(residual, pixels) };
        let luma_s = unsafe { core::slice::from_raw_parts(luma, pixels) };
        let x = unsafe { *xs.add(k as usize) };
        let y = unsafe { *ys.add(k as usize) };
        let measured = measure_one(residual_s, luma_s, width, height, x, y, thresh, satlevel);
        let dst = unsafe { out.add((k as usize) * 8) };
        let mut i = 0;
        while i < 8 {
            unsafe { *dst.add(i) = measured[i]; }
            i += 1;
        }
    }
}
