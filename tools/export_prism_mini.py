# SPDX-License-Identifier: GPL-3.0-or-later
"""Convertit prism_mini.pt en dubNoise/dub_noise_rd.wts, lu par dubGPUStack.

Le réseau repris ici suit le script MIT SyQon Prism (Adrian Knagg-Baugh,
Franklin Marek). On n'en garde que l'architecture et les poids.

Depuis le dossier du projet, avec le Python de Siril :

    & "$env:LOCALAPPDATA\\siril\\venv\\Scripts\\python.exe" tools\\export_prism_mini.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "siril" / "syqon_prism" / "prism_mini.pt"
OUT = ROOT / "dubNoise" / "dub_noise_rd.wts"
REF_IN = ROOT / "dubNoise" / "prism_ref_in.f32"
REF_OUT = ROOT / "dubNoise" / "prism_ref_out.f32"


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlockMini(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.dwconv = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1, bias=True))
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.ffn2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sg(self.dwconv(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.sg(self.ffn1(self.norm2(x)))
        return x + self.ffn2(y) * self.gamma


class NAFNetMini(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=48, enc_blk_nums=(2, 4, 6, 8), dec_blk_nums=(2, 2, 2, 2), middle_blk_num=4):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlockMini(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2
        self.middle = nn.Sequential(*[NAFBlockMini(ch) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=True), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlockMini(ch) for _ in range(n)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, dec in zip(self.ups, self.decoders):
            x = up(x) + skips.pop()
            x = dec(x)
        return self.ending(x)


def schedule(depth: int):
    depth = max(1, min(8, depth))
    if depth >= 4:
        enc = (2, 4, 6, 8) + (8,) * (depth - 4)
        dec = (2,) * depth
        middle = 4
    elif depth == 3:
        enc, dec, middle = (2, 4, 6), (2, 2, 2), 3
    elif depth == 2:
        enc, dec, middle = (2, 4), (2, 2), 2
    else:
        enc, dec, middle = (2,), (2,), 2
    return enc, dec, middle


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"weights_only a échoué ({exc}), second essai")
        return torch.load(path, map_location="cpu", weights_only=False)


def write_wts(path: Path, state: dict[str, torch.Tensor]) -> None:
    items = sorted((name, tensor.detach().contiguous().float().cpu()) for name, tensor in state.items())
    with path.open("wb") as handle:
        handle.write(b"DUBPRISM")
        handle.write(struct.pack("<I", 1))
        handle.write(struct.pack("<I", len(items)))
        for name, tensor in items:
            raw = name.encode("utf-8")
            shape = tuple(int(v) for v in tensor.shape)
            handle.write(struct.pack("<H", len(raw)))
            handle.write(raw)
            handle.write(struct.pack("<B", len(shape)))
            for dim in shape:
                handle.write(struct.pack("<I", dim))
            array = tensor.numpy()
            handle.write(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def main() -> int:
    if not PT.exists():
        print(f"introuvable : {PT}")
        return 1
    print(f"lecture de {PT}")
    ckpt = load_checkpoint(PT)
    if not isinstance(ckpt, dict):
        print(f"checkpoint inattendu : {type(ckpt)}")
        return 1
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    state = ckpt["model"] if "model" in ckpt and isinstance(ckpt["model"], dict) else ckpt
    state = {str(k): v for k, v in state.items() if torch.is_tensor(v)}
    base = int(args.get("base_ch", 48))
    depth = int(args.get("depth", 4))
    enc, dec, middle = schedule(depth)
    print(f"args base_ch={base} depth={depth} enc={enc} dec={dec} middle={middle} tenseurs={len(state)}")
    model = NAFNetMini(width=base, enc_blk_nums=enc, dec_blk_nums=dec, middle_blk_num=middle)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        names = sorted(state)[:40]
        print(f"poids incompatibles : {exc}")
        print("premiers noms :", ", ".join(names))
        return 1
    model.eval()
    params = sum(int(p.numel()) for p in model.parameters())
    print(f"paramètres : {params}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_wts(OUT, model.state_dict())
    print(f"écrit {OUT} ({OUT.stat().st_size} octets)")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(0)
    sample = torch.rand(1, 3, 64, 64, generator=generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"référence 64×64 sur {device}")
    with torch.no_grad():
        result = model.to(device)(sample.to(device)).float().cpu()
    REF_IN.write_bytes(np.ascontiguousarray(sample.numpy(), dtype=np.float32).tobytes())
    REF_OUT.write_bytes(np.ascontiguousarray(result.numpy(), dtype=np.float32).tobytes())
    print(f"référence {REF_IN.name} {REF_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
