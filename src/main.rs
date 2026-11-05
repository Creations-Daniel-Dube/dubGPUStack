// SPDX-License-Identifier: GPL-3.0-or-later
#![cfg_attr(windows, windows_subsystem = "windows")]

mod cli;
mod engine;
mod gpu;
mod icon;
mod runtime;
mod settings;
mod ui;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if needs_runtime(&args) {
        if let Err(err) = runtime::ensure() {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
    if let Some(code) = cli::dispatch(&args) {
        std::process::exit(code);
    }
    let open = args.first().filter(|arg| !arg.starts_with('-')).cloned();
    if let Err(err) = ui::run(open) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn needs_runtime(args: &[String]) -> bool {
    matches!(args.first().map(String::as_str), Some("stack") | Some("prism-check"))
}
