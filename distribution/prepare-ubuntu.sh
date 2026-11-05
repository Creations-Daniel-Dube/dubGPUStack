#!/usr/bin/env bash
# Installe dubGPUStack dans distribution/Ubuntu.
# Chaque fichier est copié depuis le dépôt s'il est déjà là, sinon téléchargé
# depuis la Release GitHub. Les bibliothèques ONNX Runtime manquantes viennent
# de l'archive officielle 1.31. Les bibliothèques CUDA sont copiées si le
# toolkit est installé ; sans elles, les aberrations restent sur le processeur.
set -euo pipefail

DISTRIBUTION="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$DISTRIBUTION/.." && pwd)"
DEST="$DISTRIBUTION/Ubuntu"
MANIFEST="$DISTRIBUTION/manifest.json"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 est requis pour lire distribution/manifest.json"
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl est requis pour telecharger les fichiers"
    exit 1
fi

mapfile -t META < <(python3 - "$MANIFEST" << 'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print(m["repo"])
print(m["tag"])
print(m["ortLinux"]["url"])
for item in m["ubuntu"]:
    kind = "ort" if item.get("ort") else "file"
    local = item.get("local") or ""
    print("\t".join((kind, item["asset"], item["dest"], local)))
print("CUDA")
for item in m.get("ubuntuCuda", []):
    print("\t".join((item["name"], item["dest"])))
PY
)

REPO="${META[0]}"
TAG="${META[1]}"
ORT_URL="${META[2]}"

mkdir -p "$DEST"

same_length() {
    local src="$1" dest="$2"
    [[ -e "$dest" ]] || return 1
    local a b
    a=$(stat -L -c%s "$src")
    b=$(stat -L -c%s "$dest")
    [[ "$a" == "$b" ]]
}

download() {
    local url="$1" dest="$2"
    local remote="" have=""
    remote="$(curl -sI -L --fail "$url" 2>/dev/null | awk 'tolower($1)=="content-length:" {gsub(/\r/,"",$2); n=$2} END{if (n=="") n=0; print n}' || true)"
    if [[ -e "$dest" && -n "$remote" && "$remote" != "0" ]]; then
        have=$(stat -L -c%s "$dest")
        if [[ "$have" == "$remote" ]]; then
            echo "deja en place  $(basename "$dest")"
            return 0
        fi
    fi
    echo "telechargement  $url"
    local partial="${dest}.partial"
    if ! curl -L --fail --retry 3 --retry-delay 2 -o "$partial" "$url"; then
        rm -f "$partial"
        return 1
    fi
    mv -f "$partial" "$dest"
    return 0
}

install_asset() {
    local kind="$1" asset="$2" destrel="$3" localrel="$4"
    local dest="$DEST/$destrel"
    mkdir -p "$(dirname "$dest")"
    if [[ -n "$localrel" && -e "$ROOT/$localrel" ]]; then
        if same_length "$ROOT/$localrel" "$dest"; then
            echo "deja en place  $(basename "$dest")"
        else
            echo "copie  $ROOT/$localrel"
            cp -L "$ROOT/$localrel" "$dest"
        fi
        return 0
    fi
    local url="https://github.com/${REPO}/releases/download/${TAG}/${asset}"
    if download "$url" "$dest"; then
        return 0
    fi
    echo "introuvable  $asset"
    return 1
}

install_cuda_optional() {
    local name="$1" destrel="$2"
    local dest="$DEST/$destrel"
    local src="" dir="" found_dir=""
    for dir in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib /usr/lib/x86_64-linux-gnu; do
        if [[ -e "$dir/$name" ]]; then
            src="$dir/$name"
            found_dir="$dir"
            break
        fi
    done
    if [[ -z "$src" ]]; then
        echo "optionnel absent  $name"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    if same_length "$src" "$dest"; then
        echo "deja en place  $(basename "$dest")"
    else
        echo "copie  $src"
        cp -L "$src" "$dest"
    fi
    if [[ "$name" == "libnvrtc.so.12" ]]; then
        local extra
        for extra in "$found_dir"/libnvrtc-builtins.so*; do
            [[ -e "$extra" ]] || continue
            cp -L "$extra" "$DEST/onnxruntime/lib/$(basename "$extra")"
        done
    fi
}

fetch_ort_tarball() {
    local tmp libdir
    tmp="$(mktemp -d)"
    echo "telechargement  $ORT_URL"
    if ! curl -L --fail --retry 3 --retry-delay 2 -o "$tmp/ort.tgz" "$ORT_URL"; then
        rm -rf "$tmp"
        return 1
    fi
    tar -xzf "$tmp/ort.tgz" -C "$tmp"
    libdir="$(find "$tmp" -type d -name lib -print -quit)"
    if [[ -z "$libdir" ]]; then
        rm -rf "$tmp"
        return 1
    fi
    mkdir -p "$DEST/onnxruntime/lib"
    cp -a "$libdir"/. "$DEST/onnxruntime/lib"/
    rm -rf "$tmp"
    return 0
}

ort_ready() {
    [[ -e "$DEST/onnxruntime/lib/libonnxruntime.so" || -e "$DEST/onnxruntime/lib/libonnxruntime.so.1" ]] \
        && [[ -e "$DEST/onnxruntime/lib/libonnxruntime_providers_cuda.so" ]] \
        && [[ -e "$DEST/onnxruntime/lib/libonnxruntime_providers_shared.so" ]]
}

missing=()
ort_missing=0
mode="assets"
for ((i=3; i<${#META[@]}; i++)); do
    line="${META[$i]}"
    if [[ "$line" == "CUDA" ]]; then
        mode="cuda"
        continue
    fi
    if [[ "$mode" == "assets" ]]; then
        IFS=$'\t' read -r kind asset dest local <<< "$line"
        if ! install_asset "$kind" "$asset" "$dest" "$local"; then
            if [[ "$kind" == "ort" ]]; then
                ort_missing=1
            else
                missing+=("$asset")
            fi
        fi
    else
        IFS=$'\t' read -r name dest <<< "$line"
        install_cuda_optional "$name" "$dest"
    fi
done

if [[ "$ort_missing" -eq 1 ]]; then
    if fetch_ort_tarball && ort_ready; then
        ort_missing=0
    fi
fi

cat > "$DEST/dubGPUStack.sh" << 'EOF'
#!/bin/sh
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$DIR:$DIR/onnxruntime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DIR/dubGPUStack" "$@"
EOF
chmod +x "$DEST/dubGPUStack.sh"
if [[ -f "$DEST/dubGPUStack" ]]; then
    chmod +x "$DEST/dubGPUStack"
fi

if [[ "$ort_missing" -eq 1 ]]; then
    missing+=("libonnxruntime.so")
fi
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Fichiers introuvables : ${missing[*]}"
    exit 1
fi
echo "Installation Ubuntu prete : $DEST"
