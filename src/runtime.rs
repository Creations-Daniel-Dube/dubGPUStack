// SPDX-License-Identifier: GPL-3.0-or-later
//! Fichiers requis installés hors du dossier que dubSuite remplace à chaque mise à jour.
//! Le zip est publié sur la Release stable `runtime`.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

fn package() -> (&'static str, &'static str) {
    if cfg!(windows) {
        (
            "https://github.com/Creations-Daniel-Dube/dubGPUStack/releases/download/runtime/dubGPUStack_Required.zip",
            "dubGPUStack_Required.zip.partial",
        )
    } else {
        (
            "https://github.com/Creations-Daniel-Dube/dubGPUStack/releases/download/runtime/dubGPUStack_Required_linux.zip",
            "dubGPUStack_Required_linux.zip.partial",
        )
    }
}

fn markers() -> &'static [&'static str] {
    if cfg!(windows) {
        &["dubNoise/dub_noise_rd.wts", "siril/model_v2_5_0.onnx", "onnxruntime/onnxruntime.dll"]
    } else {
        &["dubNoise/dub_noise_rd.wts", "siril/model_v2_5_0.onnx", "onnxruntime/lib/libonnxruntime.so"]
    }
}

pub enum Update {
    Progress { fraction: f32, label: String },
    Finished(Result<(), String>),
}

pub struct Fetch {
    pub rx: Receiver<Update>,
}

impl Fetch {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = download_and_install(|fraction, label| {
                let _ = tx.send(Update::Progress { fraction, label: label.to_string() });
            });
            let _ = tx.send(Update::Finished(result));
        });
        Self { rx }
    }
}

pub fn home() -> PathBuf {
    install_dir()
}

pub fn install_dir() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
        return base.join("dubGPUStack");
    }
    let base = std::env::var("XDG_DATA_HOME").ok().filter(|value| !value.is_empty()).map(PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(|home| PathBuf::from(home).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("dubGPUStack")
}

pub fn ready() -> bool {
    markers().iter().all(|marker| roots().iter().any(|root| root.join(marker).is_file()))
}

pub fn enable_dll_search() {
    #[cfg(windows)]
    {
        let dir = install_dir();
        if !dir.is_dir() {
            return;
        }
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        unsafe { SetDllDirectoryW(wide.as_ptr()); }
    }
}

pub fn ensure() -> Result<(), String> {
    ensure_blocking()
}

pub fn ensure_blocking() -> Result<(), String> {
    enable_dll_search();
    if ready() {
        return Ok(());
    }
    download_and_install(|fraction, label| eprintln!("{label} {:.0} %", fraction * 100.0))?;
    enable_dll_search();
    if ready() {
        Ok(())
    } else {
        Err("Les fichiers requis sont toujours introuvables après l'installation.".to_string())
    }
}

fn download_and_install(mut progress: impl FnMut(f32, &str)) -> Result<(), String> {
    let dir = install_dir();
    fs::create_dir_all(&dir).map_err(|err| format!("dossier des fichiers requis : {err}"))?;
    let (url, partial) = package();
    let archive = dir.join(partial);
    progress(0.0, "Téléchargement des fichiers requis");
    download(url, &archive, &mut progress)?;
    progress(1.0, "Installation des fichiers requis");
    extract(&archive, &dir)?;
    let _ = fs::remove_file(&archive);
    if ready() {
        Ok(())
    } else {
        Err("L'archive ne contient pas les fichiers requis.".to_string())
    }
}

fn download(url: &str, dest: &Path, progress: &mut impl FnMut(f32, &str)) -> Result<(), String> {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(2 * 60 * 60)).build();
    let response = agent.get(url).set("User-Agent", concat!("dubGPUStack/", env!("CARGO_PKG_VERSION"))).call()
        .map_err(|err| format!("téléchargement des fichiers requis : {err}"))?;
    let total = response.header("Content-Length").and_then(|value| value.parse::<u64>().ok()).filter(|value| *value > 0);
    let mut reader = response.into_reader();
    let mut file = File::create(dest).map_err(|err| format!("écriture de l'archive : {err}"))?;
    let mut got = 0u64;
    let mut buf = [0u8; 256 * 1024];
    let mut last = Instant::now() - Duration::from_secs(1);
    loop {
        let read = reader.read(&mut buf).map_err(|err| format!("téléchargement interrompu : {err}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read]).map_err(|err| format!("écriture de l'archive : {err}"))?;
        got += read as u64;
        if last.elapsed() >= Duration::from_millis(80) {
            last = Instant::now();
            let fraction = total.map(|total| (got as f32 / total as f32).clamp(0.0, 1.0)).unwrap_or(0.0);
            progress(fraction, "Téléchargement des fichiers requis");
        }
    }
    file.flush().map_err(|err| format!("écriture de l'archive : {err}"))?;
    Ok(())
}

fn extract(archive: &Path, dest: &Path) -> Result<(), String> {
    let file = File::open(archive).map_err(|err| format!("lecture de l'archive : {err}"))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|err| format!("archive illisible : {err}"))?;
    let mut names = Vec::new();
    for index in 0..zip.len() {
        let entry = zip.by_index(index).map_err(|err| format!("archive illisible : {err}"))?;
        if let Some(name) = entry.enclosed_name() {
            names.push(name.to_path_buf());
        }
    }
    let root = common_directory(&names);
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index).map_err(|err| format!("archive illisible : {err}"))?;
        let Some(name) = entry.enclosed_name().map(|path| path.to_path_buf()) else { continue };
        let relative = match &root {
            Some(root) => name.strip_prefix(root).map_err(|_| "chemin d'archive inattendu".to_string())?.to_path_buf(),
            None => name,
        };
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = dest.join(&relative);
        if entry.is_dir() {
            fs::create_dir_all(&target).map_err(|err| format!("{} : {err}", target.display()))?;
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|err| format!("{} : {err}", parent.display()))?;
        }
        if entry.is_symlink() {
            place_symlink(&mut entry, &target)?;
            continue;
        }
        let mut output = File::create(&target).map_err(|err| format!("{} : {err}", target.display()))?;
        io::copy(&mut entry, &mut output).map_err(|err| format!("{} : {err}", target.display()))?;
    }
    Ok(())
}

fn place_symlink(entry: &mut impl Read, target: &Path) -> Result<(), String> {
    let mut link = String::new();
    entry.read_to_string(&mut link).map_err(|err| format!("{} : {err}", target.display()))?;
    let link = link.trim_end_matches('\0');
    let path = Path::new(link);
    if link.is_empty() || path.is_absolute() || path.components().any(|part| matches!(part, std::path::Component::ParentDir)) {
        return Err(format!("{} : lien symbolique refusé", target.display()));
    }
    #[cfg(unix)]
    {
        if target.symlink_metadata().is_ok() {
            fs::remove_file(target).map_err(|err| format!("{} : {err}", target.display()))?;
        }
        std::os::unix::fs::symlink(link, target).map_err(|err| format!("{} : {err}", target.display()))?;
        return Ok(());
    }
    #[cfg(not(unix))]
    {
        let _ = target;
        Err("lien symbolique non pris en charge".to_string())
    }
}

fn roots() -> Vec<PathBuf> {
    let mut roots = vec![install_dir()];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            push_parents(&mut roots, dir);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_parents(&mut roots, &cwd);
    }
    roots
}

fn push_parents(roots: &mut Vec<PathBuf>, start: &Path) {
    let mut current = Some(start);
    for _ in 0..8 {
        let Some(dir) = current else { break };
        if !roots.iter().any(|root| root == dir) {
            roots.push(dir.to_path_buf());
        }
        current = dir.parent();
    }
}

fn common_directory(paths: &[PathBuf]) -> Option<PathBuf> {
    let mut shared: Option<std::ffi::OsString> = None;
    for path in paths {
        let mut parts = path.components();
        let head = parts.next()?.as_os_str().to_os_string();
        if parts.next().is_none() {
            return None;
        }
        match &shared {
            None => shared = Some(head),
            Some(current) if current != &head => return None,
            Some(_) => {}
        }
    }
    shared.map(PathBuf::from)
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetDllDirectoryW(path: *const u16) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_archive_folder_is_removed() {
        let paths = vec![PathBuf::from("pack/dubNoise/dub_noise_rd.wts"), PathBuf::from("pack/onnxruntime/onnxruntime.dll")];
        assert_eq!(common_directory(&paths), Some(PathBuf::from("pack")));
    }

    #[test]
    fn top_level_files_stay_in_place() {
        let paths = vec![PathBuf::from("cublas64_12.dll"), PathBuf::from("dubNoise/dub_noise_rd.wts")];
        assert_eq!(common_directory(&paths), None);
    }

    #[cfg(unix)]
    #[test]
    fn archive_symlink_is_restored() {
        let dir = std::env::temp_dir().join(format!("dubgpustack-zip-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("pack.zip");
        let mut file = File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(&mut file);
        let options = zip::write::SimpleFileOptions::default();
        writer.add_symlink("onnxruntime/lib/libonnxruntime.so", "libonnxruntime.so.1.31.0", options).unwrap();
        writer.start_file("onnxruntime/lib/libonnxruntime.so.1.31.0", options).unwrap();
        writer.write_all(b"runtime").unwrap();
        writer.start_file("dubNoise/marker.txt", options).unwrap();
        writer.write_all(b"wts").unwrap();
        writer.finish().unwrap();
        drop(file);
        let dest = dir.join("out");
        extract(&archive, &dest).unwrap();
        let link = dest.join("onnxruntime/lib/libonnxruntime.so");
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
        assert_eq!(fs::read_link(&link).unwrap(), PathBuf::from("libonnxruntime.so.1.31.0"));
        assert_eq!(fs::read(dest.join("onnxruntime/lib/libonnxruntime.so.1.31.0")).unwrap(), b"runtime");
        assert_eq!(fs::read(&link).unwrap(), b"runtime");
        let _ = fs::remove_dir_all(&dir);
    }
}
