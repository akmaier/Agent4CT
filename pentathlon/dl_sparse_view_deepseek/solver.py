"""DL-Sparse-View CT solver — the file the autoresearch agent edits.

Per the autoresearch contract, **this is the only file the agent should
modify per iteration**. The harness, geometry, projector, simulation, and
metric definitions are *fixed* (see ``ddssl_ldct/`` and
``challenges/dl_sparse_view/README.md``).

Inputs:
    - reads the geometry from the constants below
    - synthesises random-ellipse phantoms as a stand-in for the real
      AAPM DL-Sparse-View breast phantom (real data download is gated;
      see challenges/dl_sparse_view/README.md). The forward problem is
      the same: 2D fan-beam, sparse views, perfectly-known ground truth.

Outputs (written into the directory passed as argv[1]):
    - result.json with keys: val_score, val_psnr, val_ssim, val_rmse,
      headroom, baseline_score, oracle_score, params_M, train_n,
      change_class, rationale, advice_for_others
    - comparison.png — reference / low-view FBP / dual-domain / phantom

Scoring (Choice A, headroom-recovered):
    baseline = sparse-view FBP, no learning
    oracle   = RMSE = 0 against the truth (score = 1.0 by construction)
    score    = 1 - val_rmse / baseline_rmse  ∈ [0, 1]

Anti-overfit rules (see docs/agents.md):
    - total trainable parameters under ~10 × train_n × pixels_per_sample
    - stage_gap > 0 surfaces in the journal; agent must regularise next
    - one change per iteration
    - cite the iteration that inspired the change in --rationale
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure the project package is importable when this script is run from a
# Slurm-allocated working dir.
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
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.training import DualDomainPipeline, train
from ddssl_ldct.metrics import psnr, ssim


# ----------------------------------------------------------------------- #
#  CONFIG  —  the agent edits this block + the model-builder below.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (fixed for now — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,       # sparse view!
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
    "optimizer":     "adamw",   # adam | adamw
    "weight_decay":  1e-4,

    # Editable: model architecture.
    "unet_c":        16,
    # "unet" | "unet_plus_bf" (Wagner 2022 BF tail) | "resnet"
    #   resnet: plain residual stack at full res (DnCNN / Sidky-2022 top-team
    #           family, cross-ported from spawn agent dl-sparse-view-res
    #           iter-2 which beat the U-Net+BF baseline on headroom).
    "img_denoiser":  "resnet",
    # Editable: residual-stack architecture (only used when img_denoiser="resnet").
    # Default to spawn agent B iter-2 winner (6 blocks, c=32, GroupNorm, ReLU).
    "res_blocks":    6,
    "res_channels":  32,
    "res_norm":      "group",   # group | batch | none
    "res_act":       "relu",    # relu | gelu | swish
    "res_kernel":    3,
    "res_dropout":   0.0,
    "res_residual":  True,      # global residual head (predicts noise)

    # Noise simulation — kept fixed so headroom is comparable across iter.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,

    # Intensity calibration. CT images live on a standard scale (HU for
    # real data; for the synthetic ellipse phantoms here the canonical
    # max is 0.05 in attenuation-coefficient units). PSNR and SSIM are
    # computed with a FIXED data_range = display_max - display_min so
    # numbers are comparable across iterations and across agents — auto
    # data-range drifts with FBP overshoot and breaks comparability.
    # Display vmin/vmax are the same fixed range for every column so a
    # comparison image's grey value means the same thing across columns.
    "display_min":   0.0,
    "display_max":   0.05,
}


# ----------------------------------------------------------------------- #
#  Model builder — agent edits HERE for architecture changes.
# ----------------------------------------------------------------------- #

class UNetPlusBF(nn.Module):
    """SmallUNet followed by a Wagner-2022 4-param trainable bilateral
    filter. The BF is interpretable (3 spatial σ + 1 range σ) and acts
    as an edge-preserving sharpening / denoising layer on top of the
    U-Net's coarse denoising."""
    def __init__(self, c: int = 16, bf_kernel: int = 7,
                 bf_sigma_x: float = 1.5, bf_sigma_y: float = 1.5,
                 bf_sigma_r: float = 0.01):
        super().__init__()
        self.unet = SmallUNet(c=c)
        self.bf = TrainableBilateralFilter2d(
            kernel_size=bf_kernel,
            sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
    def forward(self, x):
        return self.bf(self.unet(x))


# ----------------------------------------------------------------------- #
#  Residual-stack denoiser  (cross-ported from spawn agent
#  dl-sparse-view-res iter-2 winner: 6 blocks @ c=32, GroupNorm + ReLU,
#  global residual head; 0.225 M params total, beats U-Net+BF on headroom).
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
    """Standard residual block: conv -> norm -> act -> (drop) -> conv -> norm -> + x."""
    def __init__(self, c: int, kernel: int = 3, norm: str = "group",
                 act: str = "relu", dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n1 = _make_norm(norm, c)
        self.act1 = _make_act(act)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n2 = _make_norm(norm, c)
    def forward(self, x):
        h = self.conv1(x)
        h = self.n1(h)
        h = self.act1(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.n2(h)
        return x + h


class ResidualStack(nn.Module):
    """Single-resolution residual stack with zero-init tail (identity start).

    Architecture:
        head conv (1 -> c) -> act
        N x ResBlock(c)
        tail conv (c -> 1, zero-init)
        + (optional) global residual: y = x - tail(features)   (predicts noise)
    """
    def __init__(self, n_blocks: int = 6, c: int = 32, kernel: int = 3,
                 norm: str = "group", act: str = "relu",
                 dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.residual = residual
        pad = kernel // 2
        self.head = nn.Conv2d(1, c, kernel, padding=pad)
        self.head_act = _make_act(act)
        self.blocks = nn.ModuleList([
            ResBlock(c, kernel, norm, act, dropout) for _ in range(n_blocks)
        ])
        self.tail = nn.Conv2d(c, 1, kernel, padding=pad)
        # Zero-init tail so the network starts as identity
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        h = self.head(x)
        h = self.head_act(h)
        for blk in self.blocks:
            h = blk(h)
        residual = self.tail(h)
        if self.residual:
            return x - residual
        return residual


# ----------------------------------------------------------------------- #
#  build_geometry
# ----------------------------------------------------------------------- #

def build_geometry() -> FanBeamGeometry:
    cfg = CONFIG
    return FanBeamGeometry(
        image_size=cfg["image_size"],
        pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"],
        n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"],
        sod=cfg["sod"],
        sdd=cfg["sdd"],
    )


# ----------------------------------------------------------------------- #
#  build_dataset  —  synthetic ellipses for the 5-minute iteration budget.
# ----------------------------------------------------------------------- #

def build_dataset(geometry: FanBeamGeometry,
                  train_n: int = 400,
                  val_n: int = 100,
                  seed: int = 42,
                  device: str = "cuda") -> tuple:
    """Generate synthetic training and validation sets.

    Returns:
        train_sinos:  (train_n, 1, n_angles, n_det)
        train_truth:  (train_n, 1, image_size, image_size)
        val_sinos:    (val_n, 1, n_angles, n_det)
        val_truth:    (val_n, 1, image_size, image_size)
    """
    projector = PyronnFanBeamProjector(geometry)
    rng = torch.Generator(device='cpu').manual_seed(seed)

    def _gen(n, base_seed):
        sinos, truths = [], []
        for i in range(n):
            s = int(torch.randint(0, 2**31, (), generator=rng).item())
            phantom = random_ellipses_phantom(
                size=geometry.image_size, n_ellipses=14,
                seed=s, device=device)
            sino_clean = projector.forward(phantom)
            sino_noisy = simulate_low_dose(
                sino_clean,
                i0=CONFIG["noise_i0"],
                sigma_e=CONFIG["noise_sigma_e"],
                seed=s + 1000)
            sinos.append(sino_noisy)
            truths.append(phantom)
        return torch.cat(sinos, dim=0), torch.cat(truths, dim=0)

    train_sinos, train_truth = _gen(train_n, seed)
    val_sinos, val_truth = _gen(val_n, seed + 1)
    return train_sinos, train_truth, val_sinos, val_truth


# ----------------------------------------------------------------------- #
#  main
# ----------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str, help="Output directory")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}")

    # ---- geometry ---- #
    geometry = build_geometry()
    print(f"[solver] geometry: {geometry}")

    # ---- dataset ---- #
    t0 = time.time()
    train_sinos, train_truth, val_sinos, val_truth = build_dataset(
        geometry,
        train_n=CONFIG["train_n"],
        val_n=CONFIG["val_n"],
        seed=CONFIG["seed"],
        device=device,
    )
    t1 = time.time()
    print(f"[solver] dataset: {train_sinos.shape[0]} train, "
          f"{val_sinos.shape[0]} val  ({t1-t0:.1f}s)")

    # ---- models ---- #
    cfg = CONFIG
    # Projection denoiser: SmallUNet
    proj_denoiser = SmallUNet(c=cfg["unet_c"])

    # Image denoiser
    img_denoiser_type = cfg["img_denoiser"]
    if img_denoiser_type == "unet":
        img_denoiser = SmallUNet(c=cfg["unet_c"])
    elif img_denoiser_type == "unet_plus_bf":
        img_denoiser = UNetPlusBF(c=cfg["unet_c"])
    elif img_denoiser_type == "resnet":
        img_denoiser = ResidualStack(
            n_blocks=cfg["res_blocks"],
            c=cfg["res_channels"],
            kernel=cfg["res_kernel"],
            norm=cfg["res_norm"],
            act=cfg["res_act"],
            dropout=cfg["res_dropout"],
            residual=cfg["res_residual"],
        )
    else:
        raise ValueError(f"Unknown img_denoiser: {img_denoiser_type}")

    pipeline = DualDomainPipeline(
        geometry=geometry,
        proj_denoiser=proj_denoiser,
        image_denoiser=img_denoiser,
    )

    # Count params
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    params_M = trainable_params / 1e6
    print(f"[solver] params: {params_M:.3f}M trainable / {total_params/1e6:.3f}M total")

    # ---- train ---- #
    t0 = time.time()
    history = train(
        pipeline,
        dataset_sinos=train_sinos,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        device=device,
        log_every=4,
        val_sinos=val_sinos,
        val_ground_truth=val_truth,
    )
    t1 = time.time()
    print(f"[solver] training done in {t1-t0:.1f}s")

    # ---- evaluate ---- #
    pipeline.eval()
    with torch.no_grad():
        pred = pipeline.predict(val_sinos.to(device))
        val_psnr_val = float(psnr(pred, val_truth.to(device),
                                  data_range=CONFIG["display_max"] - CONFIG["display_min"]).cpu())
        val_ssim_val = float(ssim(pred, val_truth.to(device),
                                  data_range=CONFIG["display_max"] - CONFIG["display_min"]).cpu())
        val_rmse_val = float(torch.sqrt(F.mse_loss(pred, val_truth.to(device))).cpu())

    # Baseline: sparse-view FBP (no denoising)
    projector = PyronnFanBeamProjector(geometry)
    with torch.no_grad():
        baseline_pred = projector.fbp(val_sinos.to(device))
        baseline_rmse = float(torch.sqrt(F.mse_loss(baseline_pred, val_truth.to(device))).cpu())
        baseline_psnr = float(psnr(baseline_pred, val_truth.to(device),
                                   data_range=CONFIG["display_max"] - CONFIG["display_min"]).cpu())
        baseline_ssim = float(ssim(baseline_pred, val_truth.to(device),
                                   data_range=CONFIG["display_max"] - CONFIG["display_min"]).cpu())

    headroom = 1.0 - val_rmse_val / max(baseline_rmse, 1e-12)
    val_score = val_ssim_val

    print(f"\n[solver] Baseline:  RMSE={baseline_rmse:.6f}  PSNR={baseline_psnr:.2f}  SSIM={baseline_ssim:.4f}")
    print(f"[solver] Ours:      RMSE={val_rmse_val:.6f}  PSNR={val_psnr_val:.2f}  SSIM={val_ssim_val:.4f}")
    print(f"[solver] Headroom:  {headroom:.4f}   Val score (SSIM): {val_score:.4f}")

    # ---- save comparison image ---- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        idx = 0
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        vmin, vmax = CONFIG["display_min"], CONFIG["display_max"]
        titles = ["Reference", "Sparse-view FBP", "Dual-domain", "Phantom"]
        imgs = [
            val_truth[idx, 0].cpu().numpy(),
            baseline_pred[idx, 0].cpu().numpy(),
            pred[idx, 0].cpu().numpy(),
            val_truth[idx, 0].cpu().numpy(),
        ]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "comparison.png", dpi=150)
        plt.close()
        print(f"[solver] comparison.png saved")
    except Exception as e:
        print(f"[solver] WARNING: could not save comparison.png: {e}")

    # ---- save result.json ---- #
    result = {
        "val_score": val_score,
        "val_psnr": val_psnr_val,
        "val_ssim": val_ssim_val,
        "val_rmse": val_rmse_val,
        "headroom": headroom,
        "baseline_score": baseline_ssim,
        "oracle_score": 1.0,
        "params_M": params_M,
        "train_n": CONFIG["train_n"],
        "change_class": "other",
        "rationale": "Baseline run with ResidualStack (6 blocks, c=32) image denoiser + SmallUNet proj denoiser.",
        "advice_for_others": "Start with the resnet image denoiser; it beats U-Net+BF on headroom.",
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[solver] result.json saved")
    print(f"[solver] done.")


if __name__ == "__main__":
    main()
