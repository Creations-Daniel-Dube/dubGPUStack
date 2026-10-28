use std::path::Path;
use std::process::Command;

fn main() {
    embed_windows_icon();

    let manifest = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).to_path_buf();
    let kernel_root = manifest.join("kernels").join("src");
    rerun_tree(&kernel_root);

    let out_dir = Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    let ptx_path = out_dir.join("kernels.ptx");
    let source = kernel_root.join("lib.rs");
    let emit = format!("asm={}", ptx_path.display());

    let status = Command::new("rustup")
        .args([
            "run",
            "nightly",
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
            "--emit",
            &emit,
        ])
        .arg(&source)
        .status()
        .expect("rustup nightly est requis pour compiler les noyaux GPU");

    if !status.success() {
        panic!("la compilation PTX des noyaux a échoué");
    }

    let ptx = std::fs::read_to_string(&ptx_path).expect("lecture du PTX");
    let ptx = stub_extern_funcs(&ptx);
    std::fs::write(&ptx_path, &ptx).expect("écriture du PTX");
    let stamp = ptx.bytes().fold(0u64, |acc, byte| acc.wrapping_mul(16777619).wrapping_add(byte as u64));
    println!("cargo:rerun-if-changed={}", ptx_path.display());
    println!("cargo:rustc-env=DUBGPUSTACK_PTX={}", ptx_path.display());
    println!("cargo:rustc-env=DUBGPUSTACK_PTX_STAMP={stamp}");
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
                if next.trim() == ".noreturn;" {
                    out.push_str(".noreturn\n{\n\ttrap;\n\texit;\n}\n");
                    closed = true;
                    break;
                }
                out.push_str(next);
                out.push('\n');
            }
            if !closed {
                panic!("PTX : fonction externe sans terminateur .noreturn;");
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
