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

    #[allow(clippy::too_many_arguments)]
    pub fn finish_into(
        &self,
        num: &CudaSlice<f32>,
        den: &CudaSlice<f32>,
        pixels: i32,
        tile_w: i32,
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
        dst_v: &mut CudaSlice<f32>,
        dst_w: &mut CudaSlice<f32>,
        slot: i32,
        k: i32,
    ) -> Result<(), String> {
        let coef = self.stream.clone_htod(&[a, off, imgw]).map_err(|err| err.to_string())?;
        let dummy = self.zeros(1)?;
        let (mask_ref, has_mask): (&CudaSlice<f32>, i32) = match mask {
            Some(mask) => (mask, 1),
            None => (&dummy, 0),
        };
        klaunch!(
            self,
            pixels as u32,
            &self.finish_prep,
            num,
            den,
            &pixels,
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
            dst_v,
            dst_w,
            &slot,
            &k,
            &has_mask
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reject(
        &self,
        values: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        k: i32,
        pixels: i32,
        mode: i32,
        siglow: f32,
        sighigh: f32,
        plain: bool,
    ) -> Result<RejectSums, String> {
        let scratch_len = (pixels as usize).saturating_mul(k as usize).max(1);
        let mut scratch_v = self.zeros(scratch_len)?;
        let mut scratch_w = self.zeros(scratch_len)?;
        let mut scratch_a = self.zeros(scratch_len)?;
        let mut scratch_b = self.zeros(scratch_len)?;
        let mut scratch_c = self.zeros(scratch_len)?;
        let mut out = self.zeros(pixels.max(0) as usize)?;
        let counters = self.stream.alloc_zeros::<i32>(3).map_err(|err| err.to_string())?;
        let plain_i = if plain { 1i32 } else { 0 };
        klaunch!(
            self,
            pixels as u32,
            &self.reject_mean,
            values,
            weights,
            &k,
            &pixels,
            &mode,
            &siglow,
            &sighigh,
            &plain_i,
            &mut out,
            &counters.slice(0..1),
            &counters.slice(1..2),
            &counters.slice(2..3),
            &mut scratch_v,
            &mut scratch_w,
            &mut scratch_a,
            &mut scratch_b,
            &mut scratch_c
        );
        let values = self.download(&out)?;
        let counts = self.stream.clone_dtoh(&counters).map_err(|err| err.to_string())?;
        self.stream.synchronize().map_err(|err| err.to_string())?;
        Ok(RejectSums { values, nlo: counts.first().copied().unwrap_or(0), nhi: counts.get(1).copied().unwrap_or(0), nvalid: counts.get(2).copied().unwrap_or(0) })
    }
}

pub struct RejectSums {
    pub values: Vec<f32>,
    pub nlo: i32,
    pub nhi: i32,
    pub nvalid: i32,
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
        let mut num = gpu.zeros(64 * 48).expect("num");
        let mut den = gpu.zeros(64 * 48).expect("den");
        gpu.drizzle_into(&image, 64, 48, 0, Some([1, 0, 2, 1]), 0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.5, 0, 0, 64, 48, &mut num, &mut den).expect("drizzle");
        let mut dst_v = gpu.zeros(64 * 48).expect("valeurs");
        let mut dst_w = gpu.zeros(64 * 48).expect("poids");
        gpu.finish_into(&num, &den, 64 * 48, 64, 1.0, 0.0, 1.0, 0.0, None, 0, 0, 64, 48, 0, 0, 0, 0, &mut dst_v, &mut dst_w, 0, 1).expect("finish");
        let values = gpu.upload(&[1.0, 1.0, 2.0, 1.0]).expect("values");
        let weights = gpu.upload(&[1.0, 1.0, 1.0, 1.0]).expect("weights");
        gpu.reject(&values, &weights, 2, 2, 2, 3.0, 3.0, false).expect("rejet");
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
