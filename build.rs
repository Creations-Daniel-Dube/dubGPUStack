use std::path::{Path, PathBuf};
use std::process::Command;

/// Dernier nightly d'avant le relèvement du plancher NVPTX à `sm_70` (Rust 1.97, 29 avril 2026).
const PASCAL_TOOLCHAIN: &str = "nightly-2026-04-28";

fn main() {
    embed_windows_icon();

    let manifest = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let kernel_root = manifest.join("kernels").join("src");
    rerun_tree(&kernel_root);

    let out_dir = Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    let source = kernel_root.join("lib.rs");

    let modern = compile_ptx(&out_dir.join("kernels.ptx"), &source, "nightly", "sm_70");
    let pascal = compile_ptx(&out_dir.join("kernels-sm61.ptx"), &source, PASCAL_TOOLCHAIN, "sm_61");

    println!("cargo:rustc-env=DUBGPUSTACK_PTX={}", modern.display());
    println!("cargo:rustc-env=DUBGPUSTACK_PTX_STAMP={}", stamp_of(&modern));
    println!("cargo:rustc-env=DUBGPUSTACK_PTX_SM61={}", pascal.display());
    println!("cargo:rustc-env=DUBGPUSTACK_PTX_SM61_STAMP={}", stamp_of(&pascal));
}

fn compile_ptx(dest: &Path, source: &Path, toolchain: &str, cpu: &str) -> PathBuf {
    let emit = format!("asm={}", dest.display());
    let cpu_flag = format!("target-cpu={cpu}");
    let output = Command::new("rustup")
        .args([
            "run",
            toolchain,
            "rustc",
            "--edition",
            "2024",
            "--crate-name",
            "dubgpustack_kernels",
            "--crate-type",
            "rlib",
            "--target",
            "nvptx64-nvidia-cuda",
            "-C",
            "panic=abort",
            "-C",
            "opt-level=3",
            "-C",
            "overflow-checks=off",
            "-C",
            "debug-assertions=off",
            "-C",
            &cpu_flag,
            "--emit",
            &emit,
        ])
        .arg(source)
        .output()
        .unwrap_or_else(|err| panic!("impossible de lancer rustup {toolchain} : {err}"));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "la compilation PTX {cpu} ({toolchain}) a échoué.\n{stderr}\n\
             Le module Pascal exige :\n  \
             rustup toolchain install {PASCAL_TOOLCHAIN} --profile minimal\n  \
             rustup target add nvptx64-nvidia-cuda --toolchain {PASCAL_TOOLCHAIN}"
        );
    }

    let ptx = std::fs::read_to_string(dest).unwrap_or_else(|err| panic!("lecture du PTX {cpu} : {err}"));
    let marker = format!(".target {cpu}");
    if !ptx.lines().any(|line| line.trim() == marker) {
        panic!("le PTX {cpu} ne contient pas `{marker}`");
    }
    let ptx = stub_extern_funcs(&ptx);
    std::fs::write(dest, &ptx).unwrap_or_else(|err| panic!("écriture du PTX {cpu} : {err}"));
    println!("cargo:rerun-if-changed={}", dest.display());
    dest.to_path_buf()
}

fn stamp_of(path: &Path) -> u64 {
    std::fs::read(path)
        .unwrap_or_default()
        .into_iter()
        .fold(0u64, |acc, byte| acc.wrapping_mul(16777619).wrapping_add(byte as u64))
}

/// Le JIT du pilote refuse un module qui appelle `panic_bounds_check` sans la définir.
/// Ces chemins ne partent que si un indice est hors limites ; ils terminent le fil.
fn stub_extern_funcs(ptx: &str) -> String {
    let mut out = String::with_capacity(ptx.len() + 256);
    let mut lines = ptx.lines();
    while let Some(line) = lines.next() {
        if line.trim_start().starts_with(".extern .func") {
            out.push_str(&line.replacen(".extern .func", ".visible .func", 1));
            out.push('\n');
            let mut closed = false;
            for next in lines.by_ref() {
                match next.trim() {
                    ".noreturn;" => {
                        out.push_str(".noreturn\n{\n\ttrap;\n\texit;\n}\n");
                        closed = true;
                        break;
                    }
                    ";" => {
                        out.push_str("{\n\ttrap;\n\texit;\n}\n");
                        closed = true;
                        break;
                    }
                    ");" => {
                        out.push_str(")\n{\n\ttrap;\n\texit;\n}\n");
                        closed = true;
                        break;
                    }
                    _ => {
                        out.push_str(next);
                        out.push('\n');
                    }
                }
            }
            if !closed {
                panic!("PTX : fonction externe sans terminateur");
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn embed_windows_icon() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }
    println!("cargo:rerun-if-changed=images/dubGPUStack.ico");
    let mut res = winresource::WindowsResource::new();
    res.set_icon("images/dubGPUStack.ico");
    res.compile().expect("intégration de images/dubGPUStack.ico dans l'exécutable");
}

fn rerun_tree(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
