// SPDX-License-Identifier: GPL-3.0-or-later
//! Session CUDA : PTX embarqué, chargé par le pilote NVIDIA déjà présent.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use dubgpustack_kernels::fmath::ramp_table;

macro_rules! klaunch {
    ($self:expr, $count:expr, $func:expr $(, $arg:expr)* $(,)?) => {{
        let count: u32 = $count;
        if count > 0 {
            let block = launch_threads($func);
            let config = LaunchConfig {
                grid_dim: (count.div_ceil(block), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                $self.stream.launch_builder($func) $(.arg($arg))* .launch(config).map_err(gpu_err)?;
            }
        }
    }};
}

/// 1024 fils dépassent les registres des noyaux drizzle et rejet sur une RTX 3060.
fn launch_threads(func: &CudaFunction) -> u32 {
    let reported = func.max_threads_per_block().unwrap_or(128).max(32) as u32;
    (reported.min(128) / 32 * 32).max(32)
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub total: usize,
    pub free: usize,
    pub major: i32,
    pub minor: i32,
}

impl GpuInfo {
    pub fn summary(&self) -> String {
        let arch = kernel_arch(self.major, self.minor).unwrap_or("non supporté");
        format!("{} · compute {}.{} · {arch} · {:.1} Gio", self.name, self.major, self.minor, self.total as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

/// `sm_61` à partir de la GTX 1060, `sm_70` à partir de la RTX 2060 (compute 7.0).
pub fn kernel_arch(major: i32, minor: i32) -> Option<&'static str> {
    if (major, minor) >= (7, 0) {
        Some("sm_70")
    } else if (major, minor) >= (6, 1) {
        Some("sm_61")
    } else {
        None
    }
}

fn ptx_source(major: i32, minor: i32) -> Result<&'static str, String> {
    match kernel_arch(major, minor) {
        Some("sm_70") => Ok(include_str!(env!("DUBGPUSTACK_PTX"))),
        Some("sm_61") => Ok(include_str!(env!("DUBGPUSTACK_PTX_SM61"))),
        _ => Err(format!(
            "cette carte est en compute {major}.{minor}. dubGPUStack prend en charge la GTX 1060 (6.1) et les cartes plus récentes"
        )),
    }
}

pub fn probe_gpu() -> Result<GpuInfo, String> {
    let ctx = CudaContext::new(0).map_err(|err| err.to_string())?;
    device_info(&ctx)
}

fn device_info(ctx: &CudaContext) -> Result<GpuInfo, String> {
    let name = ctx.name().map_err(|err| err.to_string())?;
    let total = ctx.total_mem().map_err(|err| err.to_string())?;
    let (free, _) = ctx.mem_get_info().map_err(|err| err.to_string())?;
    let (major, minor) = ctx.compute_capability().map_err(|err| err.to_string())?;
    Ok(GpuInfo { name, total, free, major, minor })
}

pub struct Gpu {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    fill_bits: CudaFunction,
    drizzle: CudaFunction,
    finish_prep: CudaFunction,
    reject_mean: CudaFunction,
    coverage_count: CudaFunction,
    ramp: CudaSlice<f32>,
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|err| err.to_string())?;
        let (major, minor) = ctx.compute_capability().map_err(|err| err.to_string())?;
        let _stamp = env!("DUBGPUSTACK_PTX_STAMP");
        let _stamp_pascal = env!("DUBGPUSTACK_PTX_SM61_STAMP");
        Self::from_context(ctx, ptx_source(major, minor)?)
    }

    fn from_context(ctx: Arc<CudaContext>, ptx_src: &str) -> Result<Self, String> {
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(ptx_src)).map_err(|err| err.to_string())?;
        let load = |name: &str| module.load_function(name).map_err(|err| format!("{name}: {err}"));
        let ramp = stream.clone_htod(&ramp_table()).map_err(|err| err.to_string())?;
        Ok(Self {
            fill_bits: load("fill_bits")?,
            drizzle: load("drizzle")?,
            finish_prep: load("finish_prep")?,
            reject_mean: load("reject_mean")?,
            coverage_count: load("coverage_count")?,
            ramp,
            stream,
            _module: module,
        })
    }

    pub fn info(&self) -> Result<GpuInfo, String> {
        device_info(self.stream.context())
    }

    pub fn zeros(&self, len: usize) -> Result<CudaSlice<f32>, String> {
        self.stream.alloc_zeros::<f32>(len).map_err(|err| err.to_string())
    }

    pub fn fill(&self, dst: &mut CudaSlice<f32>, bits: u32) -> Result<(), String> {
        let n = dst.len() as i32;
        klaunch!(self, dst.len() as u32, &self.fill_bits, dst, &n, &bits);
        Ok(())
    }

    pub fn upload(&self, data: &[f32]) -> Result<CudaSlice<f32>, String> {
        self.stream.clone_htod(data).map_err(|err| err.to_string())
    }

    pub fn download(&self, src: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
        let host = self.stream.clone_dtoh(src).map_err(|err| err.to_string())?;
        self.stream.synchronize().map_err(|err| err.to_string())?;
        Ok(host)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn drizzle_into(
        &self,
        image: &[f32],
        width: i32,
        nrows: i32,
        y0: i32,
        lut: Option<[i32; 4]>,
        only_ch: i32,
        transform: [f64; 6],
        half: f64,
        origin_x: i32,
        origin_y: i32,
        out_w: i32,
        out_h: i32,
        num: &mut CudaSlice<f32>,
        den: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        if nrows <= 0 || width <= 0 {
            return Ok(());
        }
        let device = self.upload(image)?;
        let mut coef = transform.to_vec();
        coef.push(half);
        let coef = self.stream.clone_htod(&coef).map_err(|err| err.to_string())?;
        let cfa = if lut.is_some() { 1i32 } else { 0 };
        let lut = lut.unwrap_or([0, 0, 0, 0]);
        let count = (width as u32).saturating_mul(nrows as u32);
        klaunch!(
            self,
            count,
            &self.drizzle,
            &device,
            &width,
            &nrows,
            &y0,
            &cfa,
            &lut[0],
            &lut[1],
            &lut[2],
            &lut[3],
            &only_ch,
            &coef,
            &out_w,
            &out_h,
            &origin_x,
            &origin_y,
            num,
            den
        );
        Ok(())
    }

    pub fn copy_htod(&self, src: &[f32], dst: &mut CudaSlice<f32>) -> Result<(), String> {
        self.stream.memcpy_htod(src, dst).map_err(|err| err.to_string())
    }

    pub fn copy_dtoh(&self, src: &CudaSlice<f32>, n: usize, dst: &mut [f32]) -> Result<(), String> {
        let n = n.min(src.len()).min(dst.len());
        if n == 0 {
            return Ok(());
        }
        self.stream.memcpy_dtoh(&src.slice(0..n), &mut dst[..n]).map_err(|err| err.to_string())?;
        self.stream.synchronize().map_err(|err| err.to_string())
    }

    pub fn sync(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(|err| err.to_string())
    }

    pub fn fill_prefix(&self, dst: &mut CudaSlice<f32>, n: usize, bits: u32) -> Result<(), String> {
        let n = n.min(dst.len());
        let mut offset = 0usize;
        while offset < n {
            let count = (n - offset).min(1 << 28);
            let mut view = dst.slice_mut(offset..offset + count);
            let count_i = count as i32;
            klaunch!(self, count as u32, &self.fill_bits, &mut view, &count_i, &bits);
            offset += count;
        }
        Ok(())
    }

    /// Tampons alloués une fois pour toutes les bandes : cube, grattage du rejet, accumulateurs.
    /// `accum_planes` dimensionne num/den. `frame_planes` dimensionne la sortie d'une pose.
    /// `scratch_pixels` borne le rejet à un bloc de lignes, au lieu de dupliquer tout le cube.
    pub fn prepare(&self, pixels: usize, frames: usize, accum_planes: usize, frame_planes: usize, image_len: usize, scratch_pixels: usize) -> Result<Work, String> {
        let pixels = pixels.max(1);
        let frames = frames.max(1);
        let accum_planes = accum_planes.max(1);
        let frame_planes = frame_planes.max(1);
        let samples = pixels.saturating_mul(frames);
        let scratch_pixels = scratch_pixels.max(1).min(pixels);
        let scratch_len = scratch_pixels.saturating_mul(frames);
        let image_len = image_len.max(1);
        Ok(Work {
            dst_v: self.zeros(samples)?,
            dst_w: self.zeros(samples)?,
            scratch_v: self.zeros(scratch_len)?,
            scratch_w: self.zeros(scratch_len)?,
            scratch_a: self.zeros(scratch_len)?,
            scratch_b: self.zeros(scratch_len)?,
            scratch_c: self.zeros(scratch_len)?,
            scratch_pixels,
            out: self.zeros(pixels)?,
            counters: self.stream.alloc_zeros::<i32>(3).map_err(|err| err.to_string())?,
            num: self.zeros(pixels.saturating_mul(accum_planes))?,
            den: self.zeros(pixels.saturating_mul(accum_planes))?,
            frame_v: self.zeros(pixels.saturating_mul(frame_planes))?,
            frame_w: self.zeros(pixels.saturating_mul(frame_planes))?,
            image: self.zeros(image_len)?,
        })
    }

}

pub struct RejectSums {
    pub values: Vec<f32>,
    pub nlo: i32,
    pub nhi: i32,
    pub nvalid: i32,
}

pub struct Work {
    pub dst_v: CudaSlice<f32>,
    pub dst_w: CudaSlice<f32>,
    scratch_v: CudaSlice<f32>,
    scratch_w: CudaSlice<f32>,
    scratch_a: CudaSlice<f32>,
    scratch_b: CudaSlice<f32>,
    scratch_c: CudaSlice<f32>,
    out: CudaSlice<f32>,
    scratch_pixels: usize,
    counters: CudaSlice<i32>,
    pub num: CudaSlice<f32>,
    pub den: CudaSlice<f32>,
    pub frame_v: CudaSlice<f32>,
    pub frame_w: CudaSlice<f32>,
    pub image: CudaSlice<f32>,
}

impl Gpu {
    /// Drizzle d'une image déjà copiée dans `work.image`. `rows` est le nombre de lignes utiles au début du tampon.
    #[allow(clippy::too_many_arguments)]
    pub fn drizzle_loaded(
        &self,
        work: &mut Work,
        width: i32,
        nrows: i32,
        y0: i32,
        lut: Option<[i32; 4]>,
        only_ch: i32,
        transform: [f64; 6],
        half: f64,
        origin_x: i32,
        origin_y: i32,
        out_w: i32,
        out_h: i32,
    ) -> Result<(), String> {
        if nrows <= 0 || width <= 0 {
            return Ok(());
        }
        let mut coef = transform.to_vec();
        coef.push(half);
        let coef = self.stream.clone_htod(&coef).map_err(|err| err.to_string())?;
        let cfa = if lut.is_some() { 1i32 } else { 0 };
        let lut = lut.unwrap_or([0, 0, 0, 0]);
        let count = (width as u32).saturating_mul(nrows as u32);
        klaunch!(
            self,
            count,
            &self.drizzle,
            &work.image,
            &width,
            &nrows,
            &y0,
            &cfa,
            &lut[0],
            &lut[1],
            &lut[2],
            &lut[3],
            &only_ch,
            &coef,
            &out_w,
            &out_h,
            &origin_x,
            &origin_y,
            &mut work.num,
            &mut work.den
        );
        Ok(())
    }

    /// Écrit le plan `plane` de num/den. `cube_slot` ≥ 0 range la pose dans le cube (`slot * pixels`).
    /// Sinon le plan part dans `frame_v` / `frame_w` à l'offset `dest * pixels`.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_plane(
        &self,
        work: &mut Work,
        pixels: usize,
        tile_w: i32,
        plane: i32,
        dest: i32,
        a: f64,
        off: f64,
        imgw: f64,
        feather: f32,
        mask: Option<&CudaSlice<f32>>,
        mask_w: i32,
        mask_h: i32,
        frame_w: i32,
        frame_h: i32,
        origin_x: i32,
        origin_y: i32,
        frame_x0: i32,
        frame_y0: i32,
        cube_slot: i32,
    ) -> Result<(), String> {
        if pixels == 0 {
            return Ok(());
        }
        let start = if cube_slot >= 0 { (cube_slot as usize).saturating_mul(pixels) } else { (dest.max(0) as usize).saturating_mul(pixels) };
        let end = start + pixels;
        let into_cube = cube_slot >= 0;
        if into_cube {
            if end > work.dst_v.len() || end > work.dst_w.len() {
                return Ok(());
            }
        } else if end > work.frame_v.len() || end > work.frame_w.len() {
            return Ok(());
        }
        let coef = self.stream.clone_htod(&[a, off, imgw]).map_err(|err| err.to_string())?;
        let has_mask = if mask.is_some() { 1i32 } else { 0 };
        let pixels_i = pixels as i32;
        let slot = 0i32;
        let k = 1i32;
        let mut dst_v = if into_cube { work.dst_v.slice_mut(start..end) } else { work.frame_v.slice_mut(start..end) };
        let mut dst_w = if into_cube { work.dst_w.slice_mut(start..end) } else { work.frame_w.slice_mut(start..end) };
        let mask_ref = match mask {
            Some(mask) => mask,
            None => &work.num,
        };
        klaunch!(
            self,
            pixels as u32,
            &self.finish_prep,
            &work.num,
            &work.den,
            &pixels_i,
            &tile_w,
            &coef,
            &feather,
            &self.ramp,
            mask_ref,
            &mask_w,
            &mask_h,
            &frame_w,
            &frame_h,
            &origin_x,
            &origin_y,
            &frame_x0,
            &frame_y0,
            &mut dst_v,
            &mut dst_w,
            &slot,
            &k,
            &has_mask,
            &plane
        );
        Ok(())
    }

    /// `slot_major` : le cube est rangé pose puis pixel (`slot * pixels + pixel`). Sinon pixel puis pose.
    #[allow(clippy::too_many_arguments)]
    pub fn reject_loaded(&self, work: &mut Work, k: i32, pixels: i32, mode: i32, siglow: f32, sighigh: f32, plain: bool, slot_major: bool) -> Result<RejectSums, String> {
        let full = pixels.max(0) as usize;
        if full == 0 || k <= 0 {
            return Ok(RejectSums { values: vec![0.0; full], nlo: 0, nhi: 0, nvalid: 0 });
        }
        let logical = full.saturating_mul(k as usize);
        if logical > work.dst_v.len() || logical > work.dst_w.len() || full > work.out.len() {
            return Err("le cube GPU est trop petit pour ce rejet".to_string());
        }
        self.stream.memset_zeros(&mut work.counters).map_err(|err| err.to_string())?;
        let plain_i = if plain { 1i32 } else { 0 };
        let slot_i = if slot_major { 1i32 } else { 0 };
        let stride = if slot_major { pixels } else { k };
        let chunk_cap = work.scratch_pixels.max(1);
        let mut start = 0usize;
        while start < full {
            let count = (full - start).min(chunk_cap);
            let count_i = count as i32;
            let (v0, v1) = if slot_major { (start, logical) } else { (start.saturating_mul(k as usize), (start + count).saturating_mul(k as usize)) };
            let scratch_n = count.saturating_mul(k as usize);
            if scratch_n > work.scratch_v.len() {
                return Err("le grattage GPU est trop petit pour ce rejet".to_string());
            }
            let values = work.dst_v.slice(v0..v1);
            let weights = work.dst_w.slice(v0..v1);
            let mut out = work.out.slice_mut(start..start + count);
            let mut scratch_v = work.scratch_v.slice_mut(0..scratch_n);
            let mut scratch_w = work.scratch_w.slice_mut(0..scratch_n);
            let mut scratch_a = work.scratch_a.slice_mut(0..scratch_n);
            let mut scratch_b = work.scratch_b.slice_mut(0..scratch_n);
            let mut scratch_c = work.scratch_c.slice_mut(0..scratch_n);
            klaunch!(
                self,
                count as u32,
                &self.reject_mean,
                &values,
                &weights,
                &k,
                &count_i,
                &stride,
                &slot_i,
                &mode,
                &siglow,
                &sighigh,
                &plain_i,
                &mut out,
                &work.counters.slice(0..1),
                &work.counters.slice(1..2),
                &work.counters.slice(2..3),
                &mut scratch_v,
                &mut scratch_w,
                &mut scratch_a,
                &mut scratch_b,
                &mut scratch_c
            );
            start += count;
        }
        let mut values = vec![0.0f32; full];
        self.copy_dtoh(&work.out, full, &mut values)?;
        let counts = self.stream.clone_dtoh(&work.counters).map_err(|err| err.to_string())?;
        self.stream.synchronize().map_err(|err| err.to_string())?;
        Ok(RejectSums { values, nlo: counts.first().copied().unwrap_or(0), nhi: counts.get(1).copied().unwrap_or(0), nvalid: counts.get(2).copied().unwrap_or(0) })
    }

    /// Couverture d'un cube rangé par pose : nombre de poids > 0 pour chaque pixel.
    pub fn coverage_slots(&self, weights: &CudaSlice<f32>, k: i32, pixels: i32, dst: &mut [i32]) -> Result<(), String> {
        let n = pixels.max(0) as usize;
        if n == 0 || k <= 0 {
            return Ok(());
        }
        let mut out = self.stream.alloc_zeros::<i32>(n).map_err(|err| err.to_string())?;
        klaunch!(self, pixels as u32, &self.coverage_count, weights, &k, &pixels, &mut out);
        let n = n.min(dst.len());
        self.stream.memcpy_dtoh(&out.slice(0..n), &mut dst[..n]).map_err(|err| err.to_string())?;
        self.stream.synchronize().map_err(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{kernel_arch, Gpu};

    #[test]
    fn kernel_arch_splits_pascal_and_turing() {
        assert_eq!(kernel_arch(6, 0), None);
        assert_eq!(kernel_arch(6, 1), Some("sm_61"));
        assert_eq!(kernel_arch(6, 2), Some("sm_61"));
        assert_eq!(kernel_arch(7, 0), Some("sm_70"));
        assert_eq!(kernel_arch(7, 5), Some("sm_70"));
        assert_eq!(kernel_arch(8, 6), Some("sm_70"));
    }

    #[test]
    fn embedded_ptx_has_no_unresolved_extern() {
        for ptx in [include_str!(env!("DUBGPUSTACK_PTX")), include_str!(env!("DUBGPUSTACK_PTX_SM61"))] {
            assert!(!ptx.contains(".extern .func"), "{ptx}");
        }
        assert!(include_str!(env!("DUBGPUSTACK_PTX")).contains(".target sm_70"));
        assert!(include_str!(env!("DUBGPUSTACK_PTX_SM61")).contains(".target sm_61"));
    }

    #[test]
    fn module_loads_on_gpu() {
        let gpu = Gpu::new().expect("le pilote doit accepter le PTX");
        let info = gpu.info().expect("info GPU");
        assert!(!info.name.is_empty());
        assert!(info.major > 0);
    }

    #[test]
    fn pascal_module_launches_on_newer_gpu() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("contexte");
        let (major, minor) = ctx.compute_capability().expect("compute");
        if (major, minor) < (6, 1) {
            return;
        }
        let gpu = Gpu::from_context(ctx, include_str!(env!("DUBGPUSTACK_PTX_SM61"))).expect("le pilote doit accepter le PTX sm_61");
        launch_sample(&gpu);
    }

    #[test]
    fn heavy_kernels_launch() {
        let gpu = Gpu::new().expect("gpu");
        launch_sample(&gpu);
    }

    fn launch_sample(gpu: &Gpu) {
        let image = vec![0.4f32; 64 * 48];
        let mut work = gpu.prepare(64 * 48, 2, 1, 1, 64 * 48, 64 * 48).expect("tampons");
        gpu.copy_htod(&image, &mut work.image).expect("copie");
        gpu.drizzle_loaded(&mut work, 64, 48, 0, Some([1, 0, 2, 1]), 0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.5, 0, 0, 64, 48).expect("drizzle");
        gpu.finish_plane(&mut work, 64 * 48, 64, 0, 0, 1.0, 0.0, 1.0, 0.0, None, 0, 0, 64, 48, 0, 0, 0, 0, -1).expect("finish");
        gpu.copy_htod(&[1.0, 1.0, 2.0, 1.0], &mut work.dst_v).expect("valeurs");
        gpu.copy_htod(&[1.0, 1.0, 1.0, 1.0], &mut work.dst_w).expect("poids");
        gpu.reject_loaded(&mut work, 2, 2, 2, 3.0, 3.0, false, false).expect("rejet");
    }
}

fn gpu_err(err: DriverError) -> String {
    let text = err.to_string();
    if text.contains("LAUNCH_OUT_OF_RESOURCES") {
        return format!("Le noyau GPU demande trop de ressources pour la taille du bloc. {text}");
    }
    if text.to_ascii_lowercase().contains("memory") {
        format!("La carte graphique n'a plus de mémoire. Baissez la limite dans les options avancées, ou réduisez la bande. {text}")
    } else {
        text
    }
}
