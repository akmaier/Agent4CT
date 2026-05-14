"""DL-Sparse-View CT solver — RESIDUAL-STACK denoiser variant.

This is the spawn agent's solver, exploring an alternative denoiser
architecture family: a stack of plain residual blocks (no down/up-sampling,
no skip-encoder-decoder), inspired by DnCNN (Zhang 2017) and the residual
networks in the Sidky 2022 DL-Sparse-View report (§ 5 mentions multiple
top teams using residual / variational networks rather than U-Nets).

Inductive bias contrast vs. the main agent's `pentathlon/dl_sparse_view`
solver:
    U-Net  : multi-scale, large receptive field via pooling, lots of skip
             routing. Good at large streak artefacts.
    ResNet : no down/up-sampling, smaller receptive field but very dense
             local processing at full resolution. Cheap per-pixel
             refinement, hopefully sharper edges and less smoothing.

The denoiser keeps the SmallUNet contract from `ddssl_ldct.models` (single
1-channel in / out, predicts the residual if `residual=True`) and plugs
into the same `DualDomainPipeline`.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose, split_projections
from ddssl_ldct.training import DualDomainPipeline
from ddssl_ldct.metrics import psnr, ssim


# ----------------------------------------------------------------------- #
#  CONFIG  —  the spawn agent edits this block + the model below.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (FIXED — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,

    # Training subset for the 5-minute iteration budget.
    "train_n":       400,
    "val_n":         100,

    # Editable: training schedule.
    "epochs":        8,
    "batch_size":    1,
    "lr":            1e-4,
    "optimizer":     "adamw",
    "weight_decay":  1e-4,

    # Editable: residual-stack model architecture.
    "res_blocks":    6,          # iter-2: 8 -> 6 to fit 5-min budget
    # iter-27: asymmetric depth REVERSED of iter-26 (which had proj=4, img=8
    # and was the most-balanced loss among 4 recent discards at -0.83pp).
    # Hypothesis: iter-26 noted "proj capacity matters more than expected",
    # so test the opposite asymmetry — heavier proj-side denoising (where
    # the noise originates per Poisson + Gaussian sinogram model) and
    # lighter image-side post-processing. Total blocks 6+6=12 = 8+4, so
    # total params stay essentially identical to iter-14 baseline. Single
    # knob test of the proj > img asymmetry direction.
    "res_blocks_proj": 8,        # iter-27: 6 -> 8 (heavier proj-side)
    "res_blocks_img":  4,        # iter-27: 6 -> 4 (lighter img-side)
    "res_channels":  32,         # iter-2: 48 -> 32 (saves ~2.2x flops)
    "res_norm":      "group",    # "group" | "none" | "batch"
    "res_act":       "relu",     # "relu" | "gelu" | "swish"
    "res_kernel":    3,
    "res_dropout":   0.0,
    "residual":      True,       # global residual (predict noise)
    "res_scale":     0.1,        # iter-14: EDSR-style learnable per-block residual scaling, init 0.1


    # Noise simulation — FIXED so headroom comparable across iter / runs.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,
}


# ----------------------------------------------------------------------- #
#  Residual-stack denoiser — agent edits HERE for architecture changes.
# ----------------------------------------------------------------------- #

def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


def _make_norm(name: str, c: int) -> nn.Module:
    if name == "group":
        return nn.GroupNorm(_pick_groups(c), c)
    if name == "batch":
        return nn.BatchNorm2d(c)
    return nn.Identity()


def _make_act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU(inplace=False)
    return nn.ReLU(inplace=True)


class ResBlock(nn.Module):
    """Standard residual block: conv -> norm -> act -> conv -> norm -> + x.

    The activation after the addition is omitted (pre-activation style with
    addition at the end works well in practice for low-level vision; cf.
    DnCNN / EDSR conventions). A dropout can sit between the two convs.

    iter-14: optional learnable residual-scaling scalar alpha (per block),
    so the block becomes x + alpha * h. Init alpha small (0.1) so the stack
    starts as a near-identity at random init (cf. EDSR Lim et al. 2017
    residual-scaling and ReZero Bachlechner et al. 2020 alpha=0 init).
    Combined with the zero-init tail conv this gives a very gentle info
    path early in training — the network must actively learn to use the
    residual blocks rather than rely on them by default.
    """
    def __init__(self, c: int, kernel: int = 3, norm: str = "group",
                 act: str = "relu", dropout: float = 0.0,
                 res_scale: float | None = None):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n1 = _make_norm(norm, c)
        self.act1 = _make_act(act)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n2 = _make_norm(norm, c)
        if res_scale is None:
            self.alpha = None
        else:
            self.alpha = nn.Parameter(torch.tensor(float(res_scale)))

    def forward(self, x):
        h = self.conv1(x)
        h = self.n1(h)
        h = self.act1(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.n2(h)
        if self.alpha is None:
            return x + h
        return x + self.alpha * h


class ResidualStack(nn.Module):
    """Stack of residual blocks at a single spatial resolution.

    Architecture:
        head conv (1 -> c)
        N x ResBlock(c, c)
        tail conv (c -> 1, zero-init so the network starts as identity)
        + (optional) global residual: y = x - tail(features)

    The zero-init of the tail matches `SmallUNet`'s behaviour: the network
    starts as the identity (or as `x` when residual=True), so optimisation
    starts at a sensible point even at random init.
    """
    def __init__(self, n_blocks: int = 8, c: int = 48,
                 kernel: int = 3, norm: str = "group", act: str = "relu",
                 dropout: float = 0.0, residual: bool = True,
                 res_scale: float | None = None):
        super().__init__()
        self.residual = residual
        pad = kernel // 2
        self.head = nn.Conv2d(1, c, kernel, padding=pad)
        self.head_act = _make_act(act)
        self.blocks = nn.Sequential(*[
            ResBlock(c, kernel=kernel, norm=norm, act=act, dropout=dropout,
                     res_scale=res_scale)
            for _ in range(n_blocks)
        ])
        self.tail = nn.Conv2d(c, 1, kernel, padding=pad)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        h = self.head(x)
        h = self.head_act(h)
        h = self.blocks(h)
        y = self.tail(h)
        return x - y if self.residual else y


def build_denoisers(cfg: dict) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser).

    iter-27 enables per-domain depth via cfg["res_blocks_proj"] /
    cfg["res_blocks_img"], falling back to cfg["res_blocks"] if absent so
    older callers / journal-baseline configs still resolve.
    """
    n_proj = int(cfg.get("res_blocks_proj", cfg["res_blocks"]))
    n_img  = int(cfg.get("res_blocks_img",  cfg["res_blocks"]))

    def make(n_blocks: int):
        return ResidualStack(
            n_blocks=n_blocks,
            c=cfg["res_channels"],
            kernel=cfg["res_kernel"],
            norm=cfg["res_norm"],
            act=cfg["res_act"],
            dropout=cfg["res_dropout"],
            residual=cfg["residual"],
            res_scale=cfg.get("res_scale", None),
        )
    return make(n_proj), make(n_img)


# ----------------------------------------------------------------------- #
#  Data + training harness — leave alone (changes here count as solver
#  changes; prefer model / schedule edits).
# ----------------------------------------------------------------------- #

def build_geometry(cfg: dict) -> FanBeamGeometry:
    return FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"],
        sod=cfg["sod"], sdd=cfg["sdd"],
    )


def build_dataset(geom, n, seed, i0, sigma_e, device):
    proj = PyronnFanBeamProjector(geom).to(device)
    phantoms = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)
    with torch.no_grad():
        clean = proj.forward_project(phantoms)
        noisy = simulate_low_dose(clean, i0=i0, sigma_e=sigma_e, seed=seed + 10_000)
    return phantoms, clean, noisy


def make_optimizer(params, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver-res] device={device}  config={json.dumps(cfg)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = build_geometry(cfg)
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = R_full.fbp(val_noisy)

    proj_dn, img_dn = build_denoisers(cfg)
    pipe = DualDomainPipeline(
        geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn,
    ).to(device)
    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver-res] params = {params_total/1e6:.3f} M", flush=True)

    opt = make_optimizer(pipe.parameters(), cfg)
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        pipe.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            batch = train_noisy[idx].to(device)
            losses = pipe.training_step(batch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            running += float(losses["loss"].detach().cpu())
        mean_loss = running / max(1, train_noisy.shape[0])
        print(f"[solver-res] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}",
              flush=True)
    train_time = time.time() - t0
    print(f"[solver-res] training took {train_time:.1f}s", flush=True)

    pipe.eval()
    # Chunked validation so we don't OOM on smaller VRAM (Q5000 16GB) when
    # the full-res residual stack is run on val_n samples at once.
    val_chunk = int(cfg.get("val_chunk", 8))
    pred_list = []
    with torch.no_grad():
        for s in range(0, val_noisy.shape[0], val_chunk):
            pred_list.append(pipe.predict(val_noisy[s:s + val_chunk]))
    pred = torch.cat(pred_list, dim=0)
    val_psnr = float(psnr(pred, val_ref).cpu())
    val_ssim = float(ssim(pred, val_ref).cpu())
    val_rmse = float(((pred - val_ref) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(ld_fbp, val_ref).cpu())
    baseline_rmse = float(((ld_fbp - val_ref) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        vmax = float(val_ref.max())
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=0, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray", vmin=0, vmax=vmax)
            ax[i, 1].set_title(f"sparse-view FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=0, vmax=vmax)
            ax[i, 2].set_title(f"dual-domain ResNet  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray")
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver-res] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver-res] figure failed: {e}", flush=True)

    result = {
        "val_score":      val_score,
        "val_psnr":       val_psnr,
        "val_ssim":       val_ssim,
        "val_rmse":       val_rmse,
        "baseline_psnr":  baseline_psnr,
        "baseline_rmse":  baseline_rmse,
        "headroom":       headroom,
        "params_M":       params_total / 1e6,
        "train_n":        cfg["train_n"],
        "val_n":          cfg["val_n"],
        "train_time_s":   train_time,
        "config":         cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver-res] result: val_score={val_score:.4f}  headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="directory to write result.json + comparison.png")
    args = p.parse_args()
    main(Path(args.out_dir))
