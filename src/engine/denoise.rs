// SPDX-License-Identifier: GPL-3.0-or-later
//! Débruitage Prism Mini sur l'image déjà étirée par HyperMetric.
//! Réglages figés : force 90 %, tuile 512, recouvrement 96, bord réfléchi 96,
//! étirement statistique interne vers une médiane 0,25, calcul 32 bits.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::gpu::Gpu;

const TILE: usize = 512;
const OVERLAP: usize = 96;
const PAD: usize = 96;
const STRENGTH: f32 = 0.90;
const TARGET: f64 = 0.25;
const LN_EPS: f32 = 1e-6;
const PTX: &str = include_str!("../../kernels/denoise.ptx");

macro_rules! launch1 {
    ($stream:expr, $func:expr, $n:expr $(, $arg:expr)*) => {{
        let count: u32 = $n;
        if count > 0 {
            unsafe {
                $stream.launch_builder($func)
                    $(.arg($arg))*
                    .launch(LaunchConfig {
                        grid_dim: (count.div_ceil(256).max(1), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|err| err.to_string())?;
            }
        }
        Result::<(), String>::Ok(())
    }};
}

pub struct Report {
    pub tiles: usize,
    pub seconds: f64,
}

pub fn require_weights() -> Result<PathBuf, String> {
    weights_path()
}

pub fn apply(gpu: &Gpu, image: &mut [f32], channels: usize, height: usize, width: usize, cancel: &AtomicBool, mut progress: impl FnMut(usize, usize)) -> Result<Report, String> {
    if channels != 1 && channels != 3 {
        return Err("Prism Mini attend une image mono ou couleur".to_string());
    }
    let pixels = height.saturating_mul(width);
    if height == 0 || width == 0 || image.len() < channels.saturating_mul(pixels) {
        return Err("image vide".to_string());
    }
    if height <= PAD || width <= PAD {
        return Err("image trop petite pour le bord de 96 pixels".to_string());
    }
    let started = Instant::now();
    let rgb = to_rgb(image, channels, pixels);
    let (padded, ph, pw) = reflect_pad(&rgb, 3, height, width, PAD, PAD, PAD, PAD);
    let original = padded.clone();
    let mut stretched = padded;
    let params = statistical_params(&stretched, 3, ph, pw);
    apply_statistical(&mut stretched, 3, ph, pw, &params);
    let whole = ph.min(pw) <= TILE;
    let prep_h = if whole { ph.div_ceil(16) * 16 } else { TILE };
    let prep_w = if whole { pw.div_ceil(16) * 16 } else { TILE };
    let session = Session::open(gpu, &weights_path()?, prep_h, prep_w)?;
    session.self_test()?;
    let denoised = if whole {
        progress(0, 1);
        check(cancel)?;
        let out = session.forward_cpu(&stretched, ph, pw)?;
        progress(1, 1);
        out
    } else {
        session.tiled(&stretched, ph, pw, cancel, &mut progress)?
    };
    let mut linear = denoised;
    invert_statistical(&mut linear, 3, ph, pw, &params);
    for (value, source) in linear.iter_mut().zip(&original) {
        *value = *value * STRENGTH + *source * (1.0 - STRENGTH);
    }
    let cropped = crop(&linear, 3, ph, pw, PAD, PAD, height, width);
    if channels == 1 {
        for i in 0..pixels {
            image[i] = (cropped[i] + cropped[pixels + i] + cropped[2 * pixels + i]) / 3.0;
        }
    } else {
        image[..3 * pixels].copy_from_slice(&cropped);
    }
    let _ = rgb;
    Ok(Report { tiles: if ph.min(pw) <= TILE { 1 } else { tile_count(ph, pw) }, seconds: started.elapsed().as_secs_f64() })
}

pub fn run_check(gpu: &Gpu) -> Result<String, String> {
    let kernels = Kernels::load(gpu)?;
    kernels.self_test()?;
    let mut lines = vec!["noyaux Prism : ok".to_string()];
    let gemm_ms = kernels.bench_gemm()?;
    lines.push(format!("multiplication 512², 48 vers 96 : {gemm_ms:.1} ms"));
    let path = weights_path()?;
    let full = Session::open(gpu, &path, TILE, TILE)?;
    let mut input = vec![0.0f32; 3 * 512 * 512];
    let mut state = 1u32;
    for value in &mut input {
        *value = lcg(&mut state);
    }
    let _ = full.forward_cpu(&input, 512, 512)?;
    let started = Instant::now();
    let _ = full.forward_cpu(&input, 512, 512)?;
    let tile_ms = started.elapsed().as_secs_f64() * 1000.0;
    lines.push(format!("tuile 512, réseau complet : {tile_ms:.1} ms"));
    let input_ref = std::fs::read(ref_path(&path, "prism_ref_in.f32")?).map_err(|err| format!("référence : {err}"))?;
    let output_ref = std::fs::read(ref_path(&path, "prism_ref_out.f32")?).map_err(|err| format!("référence : {err}"))?;
    let input_ref = bytes_f32(&input_ref)?;
    let output_ref = bytes_f32(&output_ref)?;
    if input_ref.len() != 3 * 64 * 64 || output_ref.len() != 3 * 64 * 64 {
        return Err("référence 64×64 de taille inattendue".to_string());
    }
    let got = full.forward_cpu(&input_ref, 64, 64)?;
    let err = max_abs(&got, &output_ref);
    lines.push(format!("écart max sur la référence 64×64 : {err:.6}"));
    if err > 2e-3 {
        return Err(format!("{}\nécart trop grand", lines.join("\n")));
    }
    Ok(lines.join("\n"))
}

fn weights_path() -> Result<PathBuf, String> {
    for candidate in weight_candidates() {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err("Fichier dub_noise_rd.wts introuvable. Placez-le dans le même dossier que dubGPUStack.exe.".to_string())
}

fn weight_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let installed = crate::runtime::install_dir();
    candidates.push(installed.join("dub_noise_rd.wts"));
    candidates.push(installed.join("dubNoise").join("dub_noise_rd.wts"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("dub_noise_rd.wts"));
            push_project_weights(&mut candidates, dir);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_project_weights(&mut candidates, &cwd);
    }
    let home = crate::runtime::home();
    candidates.push(home.join("dub_noise_rd.wts"));
    candidates.push(home.join("dubNoise").join("dub_noise_rd.wts"));
    candidates
}

fn push_project_weights(candidates: &mut Vec<PathBuf>, start: &Path) {
    let mut current = Some(start);
    for _ in 0..8 {
        let Some(dir) = current else { break };
        candidates.push(dir.join("dubNoise").join("dub_noise_rd.wts"));
        current = dir.parent();
    }
}

fn ref_path(weights: &Path, name: &str) -> Result<PathBuf, String> {
    if let Some(dir) = weights.parent() {
        let beside = dir.join(name);
        if beside.is_file() {
            return Ok(beside);
        }
    }
    for weights in weight_candidates() {
        if let Some(dir) = weights.parent() {
            let beside = dir.join(name);
            if beside.is_file() {
                return Ok(beside);
            }
        }
    }
    Err(format!("{name} introuvable à côté de dub_noise_rd.wts"))
}

struct HostTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct Arch {
    base: usize,
    enc: Vec<usize>,
    middle: usize,
    dec: Vec<usize>,
}

struct Kernels {
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    conv1x1: CudaFunction,
    dwconv3: CudaFunction,
    conv3x3: CudaFunction,
    conv_k2s2: CudaFunction,
    layernorm: CudaFunction,
    gate: CudaFunction,
    ch_mul: CudaFunction,
    resid: CudaFunction,
    add_inplace: CudaFunction,
    copy_f32: CudaFunction,
    shuffle2: CudaFunction,
    pool_mean: CudaFunction,
    extract_tile: CudaFunction,
    accumulate: CudaFunction,
    div_clamp: CudaFunction,
    clamp01: CudaFunction,
}

struct Session {
    k: Kernels,
    dev: HashMap<String, CudaSlice<f32>>,
    shapes: HashMap<String, Vec<usize>>,
    arch: Arch,
    b0: CudaSlice<f32>,
    b1: CudaSlice<f32>,
    b2: CudaSlice<f32>,
    scale_a: CudaSlice<f32>,
    scale_b: CudaSlice<f32>,
    skips: Vec<CudaSlice<f32>>,
    forward_h: u32,
    forward_w: u32,
}

impl Session {
    fn open(gpu: &Gpu, path: &Path, max_h: usize, max_w: usize) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|err| format!("{} : {err}", path.display()))?;
        let host = parse_wts(&bytes)?;
        let arch = arch_of(&host)?;
        let k = Kernels::load(gpu)?;
        let mut dev = HashMap::with_capacity(host.len());
        let mut shapes = HashMap::with_capacity(host.len());
        for (name, tensor) in host {
            let slice = k.stream.clone_htod(&tensor.data).map_err(|err| err.to_string())?;
            shapes.insert(name.clone(), tensor.shape);
            dev.insert(name, slice);
        }
        let h = u32n(max_h)?;
        let w = u32n(max_w)?;
        let scratch = scratch_elems(arch.base, arch.enc.len(), h, w)?;
        let max_ch = arch.base << arch.enc.len();
        let skips = skip_buffers(&k, arch.base, arch.enc.len(), h, w)?;
        Ok(Self {
            b0: k.stream.alloc_zeros::<f32>(scratch).map_err(|err| err.to_string())?,
            b1: k.stream.alloc_zeros::<f32>(scratch).map_err(|err| err.to_string())?,
            b2: k.stream.alloc_zeros::<f32>(scratch).map_err(|err| err.to_string())?,
            scale_a: k.stream.alloc_zeros::<f32>(max_ch.max(8)).map_err(|err| err.to_string())?,
            scale_b: k.stream.alloc_zeros::<f32>(max_ch.max(8)).map_err(|err| err.to_string())?,
            skips,
            dev,
            shapes,
            arch,
            forward_h: h,
            forward_w: w,
            k,
        })
    }

    fn forward_cpu(&self, image: &[f32], height: usize, width: usize) -> Result<Vec<f32>, String> {
        let h = u32n(height)?;
        let w = u32n(width)?;
        let ah = h.div_ceil(16) * 16;
        let aw = w.div_ceil(16) * 16;
        if ah > self.forward_h || aw > self.forward_w {
            return Err("tuile plus grande que l'espace préparé".to_string());
        }
        let source = if ah == h && aw == w {
            image.to_vec()
        } else {
            reflect_pad(image, 3, height, width, 0, (ah - h) as usize, 0, (aw - w) as usize).0
        };
        let input = self.k.stream.clone_htod(&source).map_err(|err| err.to_string())?;
        let elems = (ah as usize) * (aw as usize) * 3;
        let output = self.k.stream.alloc_zeros::<f32>(elems).map_err(|err| err.to_string())?;
        self.forward(&input, ah, aw, &output)?;
        launch1!(&self.k.stream, &self.k.clamp01, elems as u32, &output, &(elems as u32))?;
        let mut got = download(&self.k.stream, &output)?;
        if ah != h || aw != w {
            got = crop(&got, 3, ah as usize, aw as usize, 0, 0, height, width);
        }
        Ok(got)
    }

    fn tiled(&self, image: &[f32], height: usize, width: usize, cancel: &AtomicBool, progress: &mut impl FnMut(usize, usize)) -> Result<Vec<f32>, String> {
        let stride = TILE - OVERLAP;
        let ys = tile_positions(height, TILE, stride);
        let xs = tile_positions(width, TILE, stride);
        let total = ys.len() * xs.len();
        progress(0, total);
        let h = u32n(height)?;
        let w = u32n(width)?;
        let input = self.k.stream.clone_htod(image).map_err(|err| err.to_string())?;
        let pixels = height * width;
        let output = self.k.stream.alloc_zeros::<f32>(pixels * 3).map_err(|err| err.to_string())?;
        let wsum = self.k.stream.alloc_zeros::<f32>(pixels).map_err(|err| err.to_string())?;
        let tile_in = self.k.stream.alloc_zeros::<f32>(3 * TILE * TILE).map_err(|err| err.to_string())?;
        let tile_out = self.k.stream.alloc_zeros::<f32>(3 * TILE * TILE).map_err(|err| err.to_string())?;
        let masks = upload_masks(&self.k)?;
        let mut done = 0usize;
        for y in ys {
            let ey = edge(y, height, TILE);
            for &x in &xs {
                check(cancel)?;
                let ex = edge(x, width, TILE);
                let mask = &masks[ey * 3 + ex];
                launch_extract(&self.k.stream, &self.k.extract_tile, &input, &tile_in, 3, h, w, y as u32, x as u32, TILE as u32, TILE as u32)?;
                self.forward(&tile_in, TILE as u32, TILE as u32, &tile_out)?;
                launch_accumulate(&self.k.stream, &self.k.accumulate, &tile_out, mask, &output, &wsum, 3, h, w, y as u32, x as u32, TILE as u32, TILE as u32)?;
                self.k.stream.synchronize().map_err(|err| err.to_string())?;
                done += 1;
                progress(done, total);
            }
        }
        launch1!(&self.k.stream, &self.k.div_clamp, (pixels * 3) as u32, &output, &wsum, &3u32, &(pixels as u32))?;
        download(&self.k.stream, &output)
    }

    fn forward(&self, input: &CudaSlice<f32>, h: u32, w: u32, output: &CudaSlice<f32>) -> Result<(), String> {
        if h % 16 != 0 || w % 16 != 0 {
            return Err("dimension non multiple de 16".to_string());
        }
        {
            let wgt = tensor(&self.dev, "intro.weight")?;
            let bias = tensor(&self.dev, "intro.bias")?;
            launch_conv3(&self.k.stream, &self.k.conv3x3, input, &self.b0, wgt, bias, 3, self.arch.base as u32, h, w)?;
        }
        let mut x = 0usize;
        let mut ch = self.arch.base as u32;
        let mut hh = h;
        let mut ww = w;
        let mut skip_meta: Vec<(usize, u32, u32, u32)> = Vec::new();
        for level in 0..self.arch.enc.len() {
            for block in 0..self.arch.enc[level] {
                x = self.block(&format!("encoders.{level}.{block}"), x, ch, hh, ww)?;
            }
            let count = ch * hh * ww;
            {
                let src = slot(&self.b0, &self.b1, &self.b2, x);
                launch1!(&self.k.stream, &self.k.copy_f32, count, &self.skips[level], src, &count)?;
            }
            skip_meta.push((level, ch, hh, ww));
            let y = other(x);
            let cout = self.dim(&format!("downs.{level}.weight"), 0)? as u32;
            {
                let src = slot(&self.b0, &self.b1, &self.b2, x);
                let dst = slot(&self.b0, &self.b1, &self.b2, y);
                let wgt = tensor(&self.dev, &format!("downs.{level}.weight"))?;
                let bias = tensor(&self.dev, &format!("downs.{level}.bias"))?;
                launch_k2(&self.k.stream, &self.k.conv_k2s2, src, dst, wgt, bias, ch, cout, hh, ww)?;
            }
            x = y;
            ch = cout;
            hh /= 2;
            ww /= 2;
        }
        for block in 0..self.arch.middle {
            x = self.block(&format!("middle.{block}"), x, ch, hh, ww)?;
        }
        for level in 0..self.arch.dec.len() {
            let cin = self.dim(&format!("ups.{level}.0.weight"), 1)? as u32;
            let cout = self.dim(&format!("ups.{level}.0.weight"), 0)? as u32;
            if cout % 4 != 0 {
                return Err("sous-échantillonnage Prism inattendu".to_string());
            }
            let oc = cout / 4;
            let y = other(x);
            let z = third(x, y);
            {
                let src = slot(&self.b0, &self.b1, &self.b2, x);
                let dst = slot(&self.b0, &self.b1, &self.b2, y);
                let wgt = tensor(&self.dev, &format!("ups.{level}.0.weight"))?;
                let bias = tensor(&self.dev, &format!("ups.{level}.0.bias"))?;
                launch_conv1x1(&self.k.stream, &self.k.conv1x1, src, dst, wgt, bias, cin, cout, hh * ww)?;
            }
            {
                let src = slot(&self.b0, &self.b1, &self.b2, y);
                let dst = slot(&self.b0, &self.b1, &self.b2, z);
                launch_shuffle(&self.k.stream, &self.k.shuffle2, src, dst, oc, hh, ww)?;
            }
            x = z;
            hh *= 2;
            ww *= 2;
            let (skip_i, sch, sh, sw) = skip_meta.pop().ok_or_else(|| "raccord Prism manquant".to_string())?;
            if sch != oc || sh != hh || sw != ww {
                return Err(format!("raccord {sch}×{sh}×{sw}, attendu {oc}×{hh}×{ww}"));
            }
            {
                let dst = slot(&self.b0, &self.b1, &self.b2, x);
                launch1!(&self.k.stream, &self.k.add_inplace, oc * hh * ww, dst, &self.skips[skip_i], &(oc * hh * ww))?;
            }
            for block in 0..self.arch.dec[level] {
                x = self.block(&format!("decoders.{level}.{block}"), x, oc, hh, ww)?;
            }
            ch = oc;
        }
        {
            let src = slot(&self.b0, &self.b1, &self.b2, x);
            let wgt = tensor(&self.dev, "ending.weight")?;
            let bias = tensor(&self.dev, "ending.bias")?;
            launch_conv3(&self.k.stream, &self.k.conv3x3, src, output, wgt, bias, ch, 3, hh, ww)?;
        }
        Ok(())
    }

    fn block(&self, prefix: &str, x: usize, ch: u32, h: u32, w: u32) -> Result<usize, String> {
        let spatial = h * w;
        let y = other(x);
        let z = third(x, y);
        self.ln(x, y, &format!("{prefix}.norm1"), ch, spatial)?;
        self.conv1(y, z, &format!("{prefix}.conv1"), ch, ch * 2, spatial)?;
        self.dw(z, y, &format!("{prefix}.dwconv"), ch * 2, h, w)?;
        {
            let src = slot(&self.b0, &self.b1, &self.b2, y);
            let dst = slot(&self.b0, &self.b1, &self.b2, z);
            launch_gate(&self.k.stream, &self.k.gate, src, dst, ch, spatial)?;
        }
        {
            let src = slot(&self.b0, &self.b1, &self.b2, z);
            launch_pool(&self.k.stream, &self.k.pool_mean, src, &self.scale_a, ch, spatial)?;
        }
        {
            let wgt = tensor(&self.dev, &format!("{prefix}.sca.1.weight"))?;
            let bias = tensor(&self.dev, &format!("{prefix}.sca.1.bias"))?;
            launch_conv1x1(&self.k.stream, &self.k.conv1x1, &self.scale_a, &self.scale_b, wgt, bias, ch, ch, 1)?;
        }
        {
            let src = slot(&self.b0, &self.b1, &self.b2, z);
            let dst = slot(&self.b0, &self.b1, &self.b2, y);
            launch_ch_mul(&self.k.stream, &self.k.ch_mul, src, &self.scale_b, dst, ch, spatial)?;
        }
        self.conv1(y, z, &format!("{prefix}.conv2"), ch, ch, spatial)?;
        self.resid(x, z, &format!("{prefix}.beta"), ch, spatial)?;
        self.ln(x, y, &format!("{prefix}.norm2"), ch, spatial)?;
        self.conv1(y, z, &format!("{prefix}.ffn1"), ch, ch * 2, spatial)?;
        {
            let src = slot(&self.b0, &self.b1, &self.b2, z);
            let dst = slot(&self.b0, &self.b1, &self.b2, y);
            launch_gate(&self.k.stream, &self.k.gate, src, dst, ch, spatial)?;
        }
        self.conv1(y, z, &format!("{prefix}.ffn2"), ch, ch, spatial)?;
        self.resid(x, z, &format!("{prefix}.gamma"), ch, spatial)?;
        Ok(x)
    }

    fn ln(&self, src: usize, dst: usize, name: &str, ch: u32, spatial: u32) -> Result<(), String> {
        let src = slot(&self.b0, &self.b1, &self.b2, src);
        let dst = slot(&self.b0, &self.b1, &self.b2, dst);
        let wgt = tensor(&self.dev, &format!("{name}.weight"))?;
        let bias = tensor(&self.dev, &format!("{name}.bias"))?;
        launch_ln(&self.k.stream, &self.k.layernorm, src, dst, wgt, bias, ch, spatial, LN_EPS)
    }

    fn conv1(&self, src: usize, dst: usize, name: &str, cin: u32, cout: u32, spatial: u32) -> Result<(), String> {
        let src = slot(&self.b0, &self.b1, &self.b2, src);
        let dst = slot(&self.b0, &self.b1, &self.b2, dst);
        let wgt = tensor(&self.dev, &format!("{name}.weight"))?;
        let bias = tensor(&self.dev, &format!("{name}.bias"))?;
        launch_conv1x1(&self.k.stream, &self.k.conv1x1, src, dst, wgt, bias, cin, cout, spatial)
    }

    fn dw(&self, src: usize, dst: usize, name: &str, channels: u32, h: u32, w: u32) -> Result<(), String> {
        let src = slot(&self.b0, &self.b1, &self.b2, src);
        let dst = slot(&self.b0, &self.b1, &self.b2, dst);
        let wgt = tensor(&self.dev, &format!("{name}.weight"))?;
        let bias = tensor(&self.dev, &format!("{name}.bias"))?;
        launch_dw(&self.k.stream, &self.k.dwconv3, src, dst, wgt, bias, channels, h, w)
    }

    fn resid(&self, x: usize, y: usize, name: &str, ch: u32, spatial: u32) -> Result<(), String> {
        let x = slot(&self.b0, &self.b1, &self.b2, x);
        let y = slot(&self.b0, &self.b1, &self.b2, y);
        let scale = tensor(&self.dev, name)?;
        launch_resid(&self.k.stream, &self.k.resid, x, y, scale, ch, spatial)
    }

    fn dim(&self, name: &str, axis: usize) -> Result<usize, String> {
        self.shapes.get(name).and_then(|shape| shape.get(axis).copied()).ok_or_else(|| format!("forme manquante : {name}"))
    }

    fn self_test(&self) -> Result<(), String> {
        self.k.self_test()
    }
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Self, String> {
        let stream = gpu.stream();
        let module = stream.context().load_module(Ptx::from_src(PTX)).map_err(|err| format!("noyau Prism : {err}"))?;
        let load = |name: &str| module.load_function(name).map_err(|err| format!("{name} : {err}"));
        Ok(Self {
            conv1x1: load("conv1x1")?,
            dwconv3: load("dwconv3")?,
            conv3x3: load("conv3x3")?,
            conv_k2s2: load("conv_k2s2")?,
            layernorm: load("layernorm")?,
            gate: load("gate")?,
            ch_mul: load("ch_mul")?,
            resid: load("resid")?,
            add_inplace: load("add_inplace")?,
            copy_f32: load("copy_f32")?,
            shuffle2: load("shuffle2")?,
            pool_mean: load("pool_mean")?,
            extract_tile: load("extract_tile")?,
            accumulate: load("accumulate")?,
            div_clamp: load("div_clamp")?,
            clamp01: load("clamp01")?,
            stream,
            _module: module,
        })
    }

    fn bench_gemm(&self) -> Result<f64, String> {
        let spatial = 512u32 * 512;
        let cin = 48u32;
        let cout = 96u32;
        let input = self.stream.clone_htod(&vec![0.1f32; spatial as usize * cin as usize]).map_err(|err| err.to_string())?;
        let weight = self.stream.clone_htod(&vec![0.01f32; cout as usize * cin as usize]).map_err(|err| err.to_string())?;
        let bias = self.stream.clone_htod(&vec![0.0f32; cout as usize]).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(spatial as usize * cout as usize).map_err(|err| err.to_string())?;
        launch_conv1x1(&self.stream, &self.conv1x1, &input, &output, &weight, &bias, cin, cout, spatial)?;
        self.stream.synchronize().map_err(|err| err.to_string())?;
        let started = Instant::now();
        launch_conv1x1(&self.stream, &self.conv1x1, &input, &output, &weight, &bias, cin, cout, spatial)?;
        self.stream.synchronize().map_err(|err| err.to_string())?;
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }

    fn self_test(&self) -> Result<(), String> {
        self.test_conv1x1(3, 5, 7)?;
        self.test_conv1x1(17, 70, 65)?;
        self.test_dw()?;
        self.test_conv3()?;
        self.test_k2()?;
        self.test_gate()?;
        self.test_ln()?;
        self.test_shuffle()?;
        self.test_pool()?;
        Ok(())
    }

    fn test_conv1x1(&self, cin: usize, cout: usize, spatial: usize) -> Result<(), String> {
        let mut state = 7u32;
        let input = randn(cin * spatial, &mut state);
        let weight = randn(cout * cin, &mut state);
        let bias = randn(cout, &mut state);
        let expect = cpu_conv1x1(&input, &weight, &bias, cin, cout, spatial);
        let got = self.gpu_conv1x1(&input, &weight, &bias, cin, cout, spatial)?;
        close("conv1x1", &got, &expect, 1e-3)
    }

    fn gpu_conv1x1(&self, input: &[f32], weight: &[f32], bias: &[f32], cin: usize, cout: usize, spatial: usize) -> Result<Vec<f32>, String> {
        let input_d = self.stream.clone_htod(input).map_err(|err| err.to_string())?;
        let weight_d = self.stream.clone_htod(weight).map_err(|err| err.to_string())?;
        let bias_d = self.stream.clone_htod(bias).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(cout * spatial).map_err(|err| err.to_string())?;
        launch_conv1x1(&self.stream, &self.conv1x1, &input_d, &output, &weight_d, &bias_d, cin as u32, cout as u32, spatial as u32)?;
        download(&self.stream, &output)
    }

    fn test_dw(&self) -> Result<(), String> {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let bias = vec![0.5];
        let expect = cpu_dw(&input, &weight, &bias, 1, 3, 3);
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let weight_d = self.stream.clone_htod(&weight).map_err(|err| err.to_string())?;
        let bias_d = self.stream.clone_htod(&bias).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(9).map_err(|err| err.to_string())?;
        launch_dw(&self.stream, &self.dwconv3, &input_d, &output, &weight_d, &bias_d, 1, 3, 3)?;
        close("dwconv", &download(&self.stream, &output)?, &expect, 1e-5)
    }

    fn test_conv3(&self) -> Result<(), String> {
        let input = vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let mut weight = vec![0.0; 9];
        weight[4] = 1.0;
        let bias = vec![0.25];
        let expect = cpu_conv3(&input, &weight, &bias, 1, 1, 3, 3);
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let weight_d = self.stream.clone_htod(&weight).map_err(|err| err.to_string())?;
        let bias_d = self.stream.clone_htod(&bias).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(9).map_err(|err| err.to_string())?;
        launch_conv3(&self.stream, &self.conv3x3, &input_d, &output, &weight_d, &bias_d, 1, 1, 3, 3)?;
        close("conv3x3", &download(&self.stream, &output)?, &expect, 1e-5)
    }

    fn test_k2(&self) -> Result<(), String> {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0];
        let expect = cpu_k2(&input, &weight, &bias, 1, 1, 2, 2);
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let weight_d = self.stream.clone_htod(&weight).map_err(|err| err.to_string())?;
        let bias_d = self.stream.clone_htod(&bias).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(1).map_err(|err| err.to_string())?;
        launch_k2(&self.stream, &self.conv_k2s2, &input_d, &output, &weight_d, &bias_d, 1, 1, 2, 2)?;
        close("conv2", &download(&self.stream, &output)?, &expect, 1e-5)
    }

    fn test_gate(&self) -> Result<(), String> {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expect = cpu_gate(&input, 2, 2);
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(4).map_err(|err| err.to_string())?;
        launch_gate(&self.stream, &self.gate, &input_d, &output, 2, 2)?;
        close("gate", &download(&self.stream, &output)?, &expect, 1e-5)
    }

    fn test_ln(&self) -> Result<(), String> {
        let input = vec![1.0, 3.0, 2.0, 4.0];
        let weight = vec![1.0, 2.0];
        let bias = vec![0.0, 0.5];
        let expect = cpu_ln(&input, &weight, &bias, 2, 2, LN_EPS);
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let weight_d = self.stream.clone_htod(&weight).map_err(|err| err.to_string())?;
        let bias_d = self.stream.clone_htod(&bias).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(4).map_err(|err| err.to_string())?;
        launch_ln(&self.stream, &self.layernorm, &input_d, &output, &weight_d, &bias_d, 2, 2, LN_EPS)?;
        close("layernorm", &download(&self.stream, &output)?, &expect, 1e-4)
    }

    fn test_shuffle(&self) -> Result<(), String> {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let expect = cpu_shuffle(&input, 1, 1, 1);
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(4).map_err(|err| err.to_string())?;
        launch_shuffle(&self.stream, &self.shuffle2, &input_d, &output, 1, 1, 1)?;
        close("shuffle", &download(&self.stream, &output)?, &expect, 1e-6)
    }

    fn test_pool(&self) -> Result<(), String> {
        let input = vec![1.0, 3.0, 2.0, 4.0];
        let expect = vec![2.5];
        let input_d = self.stream.clone_htod(&input).map_err(|err| err.to_string())?;
        let output = self.stream.alloc_zeros::<f32>(1).map_err(|err| err.to_string())?;
        launch_pool(&self.stream, &self.pool_mean, &input_d, &output, 1, 4)?;
        close("pool", &download(&self.stream, &output)?, &expect, 1e-5)
    }
}

fn tensor<'a>(dev: &'a HashMap<String, CudaSlice<f32>>, name: &str) -> Result<&'a CudaSlice<f32>, String> {
    dev.get(name).ok_or_else(|| format!("poids manquant : {name}"))
}

fn slot<'a>(b0: &'a CudaSlice<f32>, b1: &'a CudaSlice<f32>, b2: &'a CudaSlice<f32>, index: usize) -> &'a CudaSlice<f32> {
    match index {
        0 => b0,
        1 => b1,
        _ => b2,
    }
}

fn other(index: usize) -> usize {
    (index + 1) % 3
}

fn third(a: usize, b: usize) -> usize {
    (0..3).find(|i| *i != a && *i != b).unwrap_or(0)
}

fn launch_conv1x1(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, weight: &CudaSlice<f32>, bias: &CudaSlice<f32>, cin: u32, cout: u32, spatial: u32) -> Result<(), String> {
    if cin == 0 || cout == 0 || spatial == 0 {
        return Ok(());
    }
    unsafe {
        stream.launch_builder(func).arg(input).arg(weight).arg(bias).arg(output).arg(&cin).arg(&cout).arg(&spatial).launch(LaunchConfig {
            grid_dim: (spatial.div_ceil(64), cout.div_ceil(64), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        }).map_err(|err| format!("conv1x1 : {err}"))?;
    }
    Ok(())
}

fn launch_dw(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, weight: &CudaSlice<f32>, bias: &CudaSlice<f32>, channels: u32, h: u32, w: u32) -> Result<(), String> {
    let n = channels.saturating_mul(h).saturating_mul(w);
    launch1!(stream, func, n, input, weight, bias, output, &channels, &h, &w)
}

fn launch_conv3(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, weight: &CudaSlice<f32>, bias: &CudaSlice<f32>, cin: u32, cout: u32, h: u32, w: u32) -> Result<(), String> {
    let n = cout.saturating_mul(h).saturating_mul(w);
    launch1!(stream, func, n, input, weight, bias, output, &cin, &cout, &h, &w)
}

fn launch_k2(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, weight: &CudaSlice<f32>, bias: &CudaSlice<f32>, cin: u32, cout: u32, h: u32, w: u32) -> Result<(), String> {
    let n = cout.saturating_mul(h / 2).saturating_mul(w / 2);
    launch1!(stream, func, n, input, weight, bias, output, &cin, &cout, &h, &w)
}

fn launch_ln(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, weight: &CudaSlice<f32>, bias: &CudaSlice<f32>, channels: u32, spatial: u32, eps: f32) -> Result<(), String> {
    launch1!(stream, func, spatial, input, output, weight, bias, &channels, &spatial, &eps)
}

fn launch_gate(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, channels: u32, spatial: u32) -> Result<(), String> {
    launch1!(stream, func, channels.saturating_mul(spatial), input, output, &channels, &spatial)
}

fn launch_ch_mul(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, scale: &CudaSlice<f32>, output: &CudaSlice<f32>, channels: u32, spatial: u32) -> Result<(), String> {
    launch1!(stream, func, channels.saturating_mul(spatial), input, scale, output, &channels, &spatial)
}

fn launch_resid(stream: &CudaStream, func: &CudaFunction, x: &CudaSlice<f32>, y: &CudaSlice<f32>, scale: &CudaSlice<f32>, channels: u32, spatial: u32) -> Result<(), String> {
    launch1!(stream, func, channels.saturating_mul(spatial), x, y, scale, &channels, &spatial)
}

fn launch_shuffle(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, oc: u32, h: u32, w: u32) -> Result<(), String> {
    let n = oc.saturating_mul(h * 2).saturating_mul(w * 2);
    launch1!(stream, func, n, input, output, &oc, &h, &w)
}

fn launch_pool(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, channels: u32, spatial: u32) -> Result<(), String> {
    if channels == 0 || spatial == 0 {
        return Ok(());
    }
    let inv = (spatial as f64).recip() as f32;
    unsafe {
        stream.launch_builder(func).arg(input).arg(output).arg(&spatial).arg(&inv).launch(LaunchConfig {
            grid_dim: (channels, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }).map_err(|err| format!("pool : {err}"))?;
    }
    Ok(())
}

fn launch_extract(stream: &CudaStream, func: &CudaFunction, input: &CudaSlice<f32>, output: &CudaSlice<f32>, channels: u32, h: u32, w: u32, y0: u32, x0: u32, th: u32, tw: u32) -> Result<(), String> {
    let n = channels.saturating_mul(th).saturating_mul(tw);
    launch1!(stream, func, n, input, output, &channels, &h, &w, &y0, &x0, &th, &tw)
}

fn launch_accumulate(stream: &CudaStream, func: &CudaFunction, pred: &CudaSlice<f32>, mask: &CudaSlice<f32>, image: &CudaSlice<f32>, wsum: &CudaSlice<f32>, channels: u32, h: u32, w: u32, y0: u32, x0: u32, th: u32, tw: u32) -> Result<(), String> {
    launch1!(stream, func, th.saturating_mul(tw), pred, mask, image, wsum, &channels, &h, &w, &y0, &x0, &th, &tw)
}

fn download(stream: &Arc<CudaStream>, src: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
    let host = stream.clone_dtoh(src).map_err(|err| err.to_string())?;
    stream.synchronize().map_err(|err| err.to_string())?;
    Ok(host)
}

fn u32n(value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| "dimension trop grande".to_string())
}

fn check(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) { Err("cancelled".to_string()) } else { Ok(()) }
}

fn parse_wts(bytes: &[u8]) -> Result<HashMap<String, HostTensor>, String> {
    if bytes.len() < 12 || &bytes[..8] != b"DUBPRISM" {
        return Err("fichier de poids Prism illisible".to_string());
    }
    let mut cursor = 8usize;
    let version = read_u32(bytes, &mut cursor)?;
    if version != 1 {
        return Err(format!("version de poids {version}"));
    }
    let count = read_u32(bytes, &mut cursor)? as usize;
    let mut map = HashMap::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u16(bytes, &mut cursor)? as usize;
        let name_bytes = bytes.get(cursor..cursor + name_len).ok_or("poids tronqués")?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| "nom de poids illisible".to_string())?;
        cursor += name_len;
        let ndim = *bytes.get(cursor).ok_or("poids tronqués")? as usize;
        cursor += 1;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(bytes, &mut cursor)? as usize);
        }
        let n = shape.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim)).ok_or("poids trop gros")?;
        let byte_len = n.checked_mul(4).ok_or("poids trop gros")?;
        let raw = bytes.get(cursor..cursor + byte_len).ok_or("poids tronqués")?;
        let mut data = Vec::with_capacity(n);
        for chunk in raw.chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        cursor += byte_len;
        map.insert(name, HostTensor { shape, data });
    }
    Ok(map)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let raw = bytes.get(*cursor..*cursor + 4).ok_or("poids tronqués")?;
    *cursor += 4;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, String> {
    let raw = bytes.get(*cursor..*cursor + 2).ok_or("poids tronqués")?;
    *cursor += 2;
    Ok(u16::from_le_bytes(raw.try_into().unwrap()))
}

fn arch_of(host: &HashMap<String, HostTensor>) -> Result<Arch, String> {
    let intro = host.get("intro.weight").ok_or("intro.weight manquant")?;
    if intro.shape.len() != 4 || intro.shape[1] != 3 {
        return Err("le réseau Prism n'est pas un modèle couleur".to_string());
    }
    let base = intro.shape[0];
    let enc = block_counts(host, "encoders")?;
    let dec = block_counts(host, "decoders")?;
    let downs = index_count(host, "downs")?;
    let ups = index_count(host, "ups")?;
    let middle = index_count(host, "middle")?;
    if enc.len() != dec.len() || enc.len() != downs || enc.len() != ups || middle == 0 {
        return Err(format!("architecture Prism inattendue : {} niveaux", enc.len()));
    }
    Ok(Arch { base, enc, middle, dec })
}

fn block_counts(host: &HashMap<String, HostTensor>, prefix: &str) -> Result<Vec<usize>, String> {
    let mut groups: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let head = format!("{prefix}.");
    for name in host.keys() {
        let Some(rest) = name.strip_prefix(&head) else { continue };
        let mut parts = rest.split('.');
        let Some(group) = parts.next().and_then(|part| part.parse::<usize>().ok()) else { continue };
        let Some(block) = parts.next().and_then(|part| part.parse::<usize>().ok()) else { continue };
        groups.entry(group).or_default().insert(block);
    }
    if groups.is_empty() {
        return Err(format!("blocs {prefix} absents"));
    }
    let mut counts = Vec::new();
    for (index, (group, blocks)) in groups.iter().enumerate() {
        if *group != index {
            return Err(format!("trou dans {prefix}"));
        }
        counts.push(blocks.len());
    }
    Ok(counts)
}

fn index_count(host: &HashMap<String, HostTensor>, prefix: &str) -> Result<usize, String> {
    let mut found = BTreeSet::new();
    let head = format!("{prefix}.");
    for name in host.keys() {
        let Some(rest) = name.strip_prefix(&head) else { continue };
        if let Some(index) = rest.split('.').next().and_then(|part| part.parse::<usize>().ok()) {
            found.insert(index);
        }
    }
    if found.is_empty() {
        return Err(format!("{prefix} absent"));
    }
    Ok(found.len())
}

fn scratch_elems(base: usize, levels: usize, h: u32, w: u32) -> Result<usize, String> {
    let mut max_e = mul3(3, h as usize, w as usize)?;
    let mut ch = base;
    let mut hh = h as usize;
    let mut ww = w as usize;
    for _ in 0..levels {
        max_e = max_e.max(mul3(2 * ch, hh, ww)?);
        ch = ch.saturating_mul(2);
        hh /= 2;
        ww /= 2;
    }
    max_e = max_e.max(mul3(2 * ch, hh, ww)?);
    for _ in 0..levels {
        max_e = max_e.max(mul3(ch.saturating_mul(2), hh, ww)?);
        ch /= 2;
        hh = hh.saturating_mul(2);
        ww = ww.saturating_mul(2);
        max_e = max_e.max(mul3(2 * ch, hh, ww)?);
    }
    Ok(max_e.max(1))
}

fn mul3(a: usize, b: usize, c: usize) -> Result<usize, String> {
    a.checked_mul(b).and_then(|v| v.checked_mul(c)).ok_or_else(|| "image trop grande pour Prism".to_string())
}

fn skip_buffers(k: &Kernels, base: usize, levels: usize, h: u32, w: u32) -> Result<Vec<CudaSlice<f32>>, String> {
    let mut skips = Vec::with_capacity(levels);
    let mut ch = base;
    let mut hh = h as usize;
    let mut ww = w as usize;
    for _ in 0..levels {
        let n = mul3(ch, hh, ww)?;
        skips.push(k.stream.alloc_zeros::<f32>(n).map_err(|err| err.to_string())?);
        ch = ch.saturating_mul(2);
        hh /= 2;
        ww /= 2;
    }
    Ok(skips)
}

fn upload_masks(k: &Kernels) -> Result<Vec<CudaSlice<f32>>, String> {
    let (start_y, mid_y, end_y) = axis_weights(TILE, OVERLAP);
    let (start_x, mid_x, end_x) = axis_weights(TILE, OVERLAP);
    let axes_y = [&start_y, &mid_y, &end_y];
    let axes_x = [&start_x, &mid_x, &end_x];
    let mut masks = Vec::with_capacity(9);
    for wy in axes_y {
        for wx in axes_x {
            let mut mask = vec![0.0f32; TILE * TILE];
            for y in 0..TILE {
                for x in 0..TILE {
                    mask[y * TILE + x] = wy[y] * wx[x];
                }
            }
            masks.push(k.stream.clone_htod(&mask).map_err(|err| err.to_string())?);
        }
    }
    Ok(masks)
}

fn axis_weights(size: usize, overlap: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let base = weight_1d(size, overlap);
    let mut start = base.clone();
    let mut end = base.clone();
    let n = overlap.min(size);
    for i in 0..n {
        start[i] = 1.0;
        end[size - 1 - i] = 1.0;
    }
    (start, base, end)
}

fn weight_1d(size: usize, overlap: usize) -> Vec<f32> {
    let mut weights = vec![1.0f32; size];
    if overlap == 0 || size == 0 {
        return weights;
    }
    let n = overlap.min(size);
    let ramp = |i: usize| if n == 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
    for i in 0..n {
        weights[i] = ramp(i);
    }
    for i in 0..n {
        weights[size - n + i] = ramp(n - 1 - i);
    }
    weights
}

fn tile_positions(size: usize, tile: usize, stride: usize) -> Vec<usize> {
    if size <= tile {
        return vec![0];
    }
    let mut pos: Vec<usize> = (0..size - tile).step_by(stride).collect();
    let last = size - tile;
    if pos.last().copied() != Some(last) {
        pos.push(last);
    }
    pos
}

pub fn tile_count_hint(_len: usize, _channels: usize, height: usize, width: usize) -> usize {
    let ph = height.saturating_add(PAD * 2);
    let pw = width.saturating_add(PAD * 2);
    if ph.min(pw) <= TILE { 1 } else { tile_count(ph, pw) }
}

fn tile_count(height: usize, width: usize) -> usize {
    let stride = TILE - OVERLAP;
    tile_positions(height, TILE, stride).len() * tile_positions(width, TILE, stride).len()
}

fn edge(pos: usize, size: usize, tile: usize) -> usize {
    if pos == 0 { 0 } else if pos + tile >= size { 2 } else { 1 }
}

fn to_rgb(image: &[f32], channels: usize, pixels: usize) -> Vec<f32> {
    if channels == 1 {
        let mut rgb = Vec::with_capacity(pixels * 3);
        rgb.extend_from_slice(&image[..pixels]);
        rgb.extend_from_slice(&image[..pixels]);
        rgb.extend_from_slice(&image[..pixels]);
        rgb
    } else {
        image[..pixels * 3].to_vec()
    }
}

fn reflect_pad(src: &[f32], channels: usize, height: usize, width: usize, top: usize, bottom: usize, left: usize, right: usize) -> (Vec<f32>, usize, usize) {
    let oh = height + top + bottom;
    let ow = width + left + right;
    let mut dst = vec![0.0f32; channels * oh * ow];
    for c in 0..channels {
        for y in 0..oh {
            let sy = reflect_index(y as i32 - top as i32, height as i32) as usize;
            for x in 0..ow {
                let sx = reflect_index(x as i32 - left as i32, width as i32) as usize;
                dst[(c * oh + y) * ow + x] = src[(c * height + sy) * width + sx];
            }
        }
    }
    (dst, oh, ow)
}

fn reflect_index(mut index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }
    let period = 2 * length - 2;
    index = index.rem_euclid(period);
    if index >= length {
        index = period - index;
    }
    index
}

fn crop(src: &[f32], channels: usize, height: usize, width: usize, top: usize, left: usize, oh: usize, ow: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; channels * oh * ow];
    for c in 0..channels {
        for y in 0..oh {
            let row = (c * height + y + top) * width + left;
            let dst_row = (c * oh + y) * ow;
            dst[dst_row..dst_row + ow].copy_from_slice(&src[row..row + ow]);
        }
    }
    dst
}

struct StretchParams {
    bp: [f64; 3],
    scale: [f64; 3],
    med2: [f64; 3],
}

fn statistical_params(image: &[f32], channels: usize, height: usize, width: usize) -> StretchParams {
    let pixels = height * width;
    let mut params = StretchParams { bp: [0.0; 3], scale: [1.0; 3], med2: [1e-6; 3] };
    for c in 0..channels.min(3) {
        let plane = &image[c * pixels..(c + 1) * pixels];
        let stride = (plane.len() / 100_000).max(1);
        let sample: Vec<f32> = plane.iter().step_by(stride).copied().collect();
        if sample.is_empty() {
            continue;
        }
        let min_c = sample.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let med = median(&mut sample.clone()) as f64;
        let mut dev: Vec<f32> = sample.iter().map(|value| (*value as f64 - med).abs() as f32).collect();
        let mad = median(&mut dev) as f64;
        let sig = 1.4826 * mad;
        let bp = min_c.max(med - 5.0 * sig);
        let mut scale = 1.0 - bp;
        if scale.abs() < 1e-12 {
            scale = 1e-12;
        }
        let mut stretched: Vec<f32> = sample.iter().map(|value| ((*value as f64 - bp) / scale).clamp(0.0, 1.0) as f32).collect();
        let med2 = (median(&mut stretched) as f64).max(1e-6);
        params.bp[c] = bp;
        params.scale[c] = scale;
        params.med2[c] = med2;
    }
    params
}

fn apply_statistical(image: &mut [f32], channels: usize, height: usize, width: usize, params: &StretchParams) {
    let pixels = height * width;
    for c in 0..channels.min(3) {
        for value in &mut image[c * pixels..(c + 1) * pixels] {
            let x = ((*value as f64 - params.bp[c]) / params.scale[c]).clamp(0.0, 1.0);
            *value = pi_mtf(x, params.med2[c], TARGET).clamp(0.0, 1.0) as f32;
        }
    }
}

fn invert_statistical(image: &mut [f32], channels: usize, height: usize, width: usize, params: &StretchParams) {
    let pixels = height * width;
    for c in 0..channels.min(3) {
        for value in &mut image[c * pixels..(c + 1) * pixels] {
            let y = pi_mtf_inv(*value as f64, params.med2[c], TARGET).clamp(0.0, 1.0);
            *value = (y * params.scale[c] + params.bp[c]) as f32;
        }
    }
}

fn pi_mtf(x: f64, med: f64, target: f64) -> f64 {
    let num = (med - 1.0) * target * x;
    let denom = guard(med * (target + x - 1.0) - target * x);
    num / denom
}

fn pi_mtf_inv(y: f64, med: f64, target: f64) -> f64 {
    let num = y * med * (1.0 - target);
    let denom = guard(y * (med - target) - target * (med - 1.0));
    num / denom
}

fn guard(value: f64) -> f64 {
    if value.abs() < 1e-12 { if value < 0.0 { -1e-12 } else { 1e-12 } } else { value }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { 0.5 * (values[n / 2 - 1] + values[n / 2]) }
}

fn bytes_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("référence tronquée".to_string());
    }
    Ok(bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(left, right)| (left - right).abs()).fold(0.0, f32::max)
}

fn close(name: &str, got: &[f32], expect: &[f32], tol: f32) -> Result<(), String> {
    if got.len() != expect.len() {
        return Err(format!("{name} : longueur {} contre {}", got.len(), expect.len()));
    }
    let err = max_abs(got, expect);
    if err > tol {
        Err(format!("{name} : écart {err}"))
    } else {
        Ok(())
    }
}

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn randn(n: usize, state: &mut u32) -> Vec<f32> {
    (0..n).map(|_| lcg(state)).collect()
}

fn cpu_conv1x1(input: &[f32], weight: &[f32], bias: &[f32], cin: usize, cout: usize, spatial: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cout * spatial];
    for oc in 0..cout {
        for s in 0..spatial {
            let mut acc = bias[oc];
            for ic in 0..cin {
                acc += weight[oc * cin + ic] * input[ic * spatial + s];
            }
            out[oc * spatial + s] = acc;
        }
    }
    out
}

fn cpu_dw(input: &[f32], weight: &[f32], bias: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * h * w];
    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let mut acc = bias[c];
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = y as i32 + ky as i32 - 1;
                        let ix = x as i32 + kx as i32 - 1;
                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                            acc += input[(c * h + iy as usize) * w + ix as usize] * weight[c * 9 + ky * 3 + kx];
                        }
                    }
                }
                out[(c * h + y) * w + x] = acc;
            }
        }
    }
    out
}

fn cpu_conv3(input: &[f32], weight: &[f32], bias: &[f32], cin: usize, cout: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cout * h * w];
    for oc in 0..cout {
        for y in 0..h {
            for x in 0..w {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let iy = y as i32 + ky as i32 - 1;
                            let ix = x as i32 + kx as i32 - 1;
                            if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                let wv = weight[((oc * cin + ic) * 3 + ky) * 3 + kx];
                                acc += input[(ic * h + iy as usize) * w + ix as usize] * wv;
                            }
                        }
                    }
                }
                out[(oc * h + y) * w + x] = acc;
            }
        }
    }
    out
}

fn cpu_k2(input: &[f32], weight: &[f32], bias: &[f32], cin: usize, cout: usize, h: usize, w: usize) -> Vec<f32> {
    let oh = h / 2;
    let ow = w / 2;
    let mut out = vec![0.0f32; cout * oh * ow];
    for oc in 0..cout {
        for y in 0..oh {
            for x in 0..ow {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let iy = y * 2 + ky;
                            let ix = x * 2 + kx;
                            let wv = weight[((oc * cin + ic) * 2 + ky) * 2 + kx];
                            acc += input[(ic * h + iy) * w + ix] * wv;
                        }
                    }
                }
                out[(oc * oh + y) * ow + x] = acc;
            }
        }
    }
    out
}

fn cpu_gate(input: &[f32], channels: usize, spatial: usize) -> Vec<f32> {
    let n = channels * spatial;
    (0..n).map(|i| input[i] * input[i + n]).collect()
}

fn cpu_ln(input: &[f32], weight: &[f32], bias: &[f32], channels: usize, spatial: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    for s in 0..spatial {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += input[c * spatial + s];
        }
        let mean = sum / channels as f32;
        let mut var = 0.0f32;
        for c in 0..channels {
            let d = input[c * spatial + s] - mean;
            var += d * d;
        }
        var /= channels as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..channels {
            let d = input[c * spatial + s] - mean;
            out[c * spatial + s] = d * inv * weight[c] + bias[c];
        }
    }
    out
}

fn cpu_shuffle(input: &[f32], oc: usize, h: usize, w: usize) -> Vec<f32> {
    let oh = h * 2;
    let ow = w * 2;
    let mut out = vec![0.0f32; oc * oh * ow];
    for c in 0..oc {
        for oy in 0..oh {
            for ox in 0..ow {
                let ic = c * 4 + (oy % 2) * 2 + (ox % 2);
                let iy = oy / 2;
                let ix = ox / 2;
                out[(c * oh + oy) * ow + ox] = input[(ic * h + iy) * w + ix];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_matches_numpy_edges() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let (padded, h, w) = reflect_pad(&src, 1, 1, 4, 0, 0, 2, 2);
        assert_eq!((h, w), (1, 8));
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn median_matches_even_and_odd() {
        assert!((median(&mut [1.0, 3.0, 2.0, 4.0]) - 2.5).abs() < 1e-6);
        assert!((median(&mut [1.0, 2.0, 9.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn statistical_roundtrip_stays_close() {
        let mut image = vec![0.02, 0.08, 0.15, 0.4, 0.7, 0.05, 0.2, 0.9];
        let original = image.clone();
        let n = image.len();
        let params = statistical_params(&image, 1, 1, n);
        apply_statistical(&mut image, 1, 1, n, &params);
        invert_statistical(&mut image, 1, 1, n, &params);
        for (got, expect) in image.iter().zip(&original) {
            assert!((got - expect).abs() < 1e-4, "{got} contre {expect}");
        }
    }

    #[test]
    fn tiles_cover_the_end() {
        let pos = tile_positions(928, 512, 416);
        assert_eq!(pos, vec![0, 416]);
        assert_eq!(weight_1d(5, 2), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }
}
