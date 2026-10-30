// SPDX-License-Identifier: GPL-3.0-or-later
//! `dubGPUStack probe` et `dubGPUStack stack DOSSIER_OU_SEQ`.

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::engine::job::{self, Event, JobConfig};
use crate::gpu;

pub fn dispatch(args: &[String]) -> Option<i32> {
    let command = args.first().map(String::as_str)?;
    if command == "probe" {
        return Some(probe());
    }
    if command == "stack" {
        return Some(stack(&args[1..]));
    }
    if command == "--help" || command == "-h" || command == "help" {
        print_help();
        return Some(0);
    }
    None
}

fn probe() -> i32 {
    match gpu::probe_gpu() {
        Ok(info) => {
            let kernels = gpu::kernel_arch(info.major, info.minor).unwrap_or("unsupported");
            println(&format!("{{\"ev\":\"probe\",\"name\":\"{}\",\"total\":{},\"free\":{},\"major\":{},\"minor\":{},\"kernels\":\"{kernels}\"}}", info.name.replace('"', "'"), info.total, info.free, info.major, info.minor));
            0
        }
        Err(err) => {
            println(&format!("{{\"ev\":\"probe\",\"error\":\"{}\"}}", err.replace('"', "'")));
            1
        }
    }
}

fn take<'a>(args: &'a [String], index: &mut usize) -> Result<&'a str, i32> {
    let Some(value) = args.get(*index) else {
        println("valeur manquante");
        return Err(2);
    };
    *index += 1;
    Ok(value)
}

fn stack(args: &[String]) -> i32 {
    let cfg = match stack_config(args) {
        Ok(cfg) => cfg,
        Err(code) => return code,
    };
    run_stack(cfg)
}

fn stack_config(args: &[String]) -> Result<JobConfig, i32> {
    let mut cfg = JobConfig::default();
    let mut source = String::new();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].clone();
        index += 1;
        let parsed = match arg.as_str() {
            "--out" => { cfg.output_path = take(args, &mut index)?.to_string(); Ok(()) }
            "--rejection" => { cfg.rejection = take(args, &mut index)?.to_string(); Ok(()) }
            "--sigma" => {
                cfg.sigma_low = take(args, &mut index)?.parse().unwrap_or(-1.0);
                cfg.sigma_high = take(args, &mut index)?.parse().unwrap_or(-1.0);
                Ok(())
            }
            "--weight" => { cfg.weighting = take(args, &mut index)?.to_string(); Ok(()) }
            "--feather" => { cfg.feather = take(args, &mut index)?.parse().unwrap_or(-1.0); Ok(()) }
            "--no-norm" => { cfg.normalization = "none".into(); Ok(()) }
            "--no-rgb-equal" => { cfg.rgb_equal = false; Ok(()) }
            "--no-output-norm" => { cfg.output_norm = false; Ok(()) }
            "--rows" => {
                let a = take(args, &mut index)?.parse::<i32>().unwrap_or(0);
                let b = take(args, &mut index)?.parse::<i32>().unwrap_or(0);
                cfg.rows = Some([a, b]);
                Ok(())
            }
            "--crop" => {
                let mut v = [0i32; 4];
                for slot in &mut v {
                    *slot = take(args, &mut index)?.parse().unwrap_or(0);
                }
                cfg.crop = Some(v);
                Ok(())
            }
            "--overwrite" => { cfg.overwrite = true; Ok(()) }
            "--gpu-mem-gib" => { cfg.gpu_mem_gib = take(args, &mut index)?.parse().unwrap_or(0.0); Ok(()) }
            "--scale" => { cfg.drizzle_scale = take(args, &mut index)?.parse().unwrap_or(0.0); Ok(()) }
            "--pixfrac" => { cfg.pixfrac = take(args, &mut index)?.parse().unwrap_or(0.0); Ok(()) }
            "--transform" => { cfg.transform = take(args, &mut index)?.to_string(); Ok(()) }
            "--analyze-only" => { cfg.analyze_only = true; Ok(()) }
            "--strip" => { cfg.strip = take(args, &mut index)?.parse().unwrap_or(64); Ok(()) }
            "--tile" => { cfg.tile = take(args, &mut index)?.parse().unwrap_or(1024); Ok(()) }
            "--help" | "-h" => { print_help(); return Err(0); }
            other if other.starts_with('-') => {
                println(&format!("option inconnue : {other}"));
                return Err(2);
            }
            other if source.is_empty() => { source = other.to_string(); Ok(()) }
            other => {
                println(&format!("argument inattendu : {other}"));
                return Err(2);
            }
        };
        if let Err(code) = parsed {
            return Err(code);
        }
    }
    if source.is_empty() {
        println("indiquez un fichier .seq ou un dossier de brutes");
        return Err(2);
    }
    let path = Path::new(&source);
    if path.is_dir() {
        cfg.lights_dir = source;
    } else if path.is_file() {
        cfg.seq_path = source;
    } else {
        println(&format!("introuvable : {source}"));
        return Err(2);
    }
    Ok(cfg)
}

fn run_stack(cfg: JobConfig) -> i32 {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    let _ = ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed));
    match job::run(&cfg, &cancel, |event| match event {
        Event::Log { level, message } => println(&format!("[{level}] {message}")),
        Event::Progress { phase, done, total, detail } => println(&format!("{phase} {done:.0}/{total:.0} {detail}")),
    }) {
        Ok(output) => {
            if !output.path.is_empty() {
                println(&format!("écrit {}", output.path));
            }
            0
        }
        Err(err) if err == "cancelled" => {
            println("annulé");
            3
        }
        Err(err) if is_user(err.as_str()) => {
            println(&err);
            2
        }
        Err(err) => {
            println(&err);
            1
        }
    }
}

fn is_user(err: &str) -> bool {
    err.contains("indiquez") || err.contains("doit") || err.contains("sélectionnez") || err.contains("au moins") || err.contains("introuvable") || err.contains("existe déjà") || err.contains("reconnue") || err.contains("n'est pas")
}

fn print_help() {
    println("dubGPUStack probe");
    println("dubGPUStack stack SEQ_OU_DOSSIER [--out FICHIER] [--rejection winsorized|sigma|none]");
    println("  [--sigma BAS HAUT] [--weight none|noise|wfwhm|nbstars] [--feather PIXELS]");
    println("  [--no-norm] [--no-rgb-equal] [--no-output-norm] [--rows Y0 Y1] [--crop X Y W H]");
    println("  [--overwrite] [--gpu-mem-gib N] [--scale 1|1.5|2|3] [--pixfrac 0.05..1]");
    println("  [--transform affine|similarity] [--analyze-only]");
    println("Sans argument, ou avec un chemin, l'interface s'ouvre.");
}

fn println(text: &str) {
    let mut out = console();
    let _ = writeln!(out, "{text}");
    let _ = out.flush();
}

fn console() -> Box<dyn Write> {
    #[cfg(windows)]
    {
        attach_parent();
        if let Ok(file) = std::fs::OpenOptions::new().write(true).open("CONOUT$") {
            return Box::new(file);
        }
    }
    Box::new(io::stdout())
}

#[cfg(windows)]
fn attach_parent() {
    unsafe extern "system" {
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe { AttachConsole(u32::MAX); }
}
