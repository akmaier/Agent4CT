"""Reference: ItNet-style iterative reconstruction with data consistency.

Based on the winning approach from Sidky 2022 DL-Sparse-View challenge.

Architecture:
  1. Pre-train a U-Net on (FBP, truth) pairs
  2. Iterative refinement loop (K=5):
     x_{k+1} = CNN_denoise(x_k) - alpha * R^T(R*x_k - sinogram)
  3. Compare sinogram of prediction against measured sinogram

This implementation uses an estimated geometry and explicit data consistency.
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

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.models import SmallUNet
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # Pre-training
    "pretrain_epochs": 20,
    "pretrain_lr":   1e-3,
    # ItNet iterations
    "itnet_k":       5,
    "itnet_alpha":   0.1,
    # Fine-tuning
    "finetune_epochs": 10,
    "finetune_lr":   1e-4,
    "unet_c":        16,
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    # Dispatches on AGENT4CT_DATASET / cfg["dataset_kind"]. Phantom path
    # is backwards-compatible; staged paths load from disk. Split is picked
    # from the existing seed convention (train: seed=cfg["seed"]; val:
    # seed=cfg["seed"]+1000).
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom)


class ItNet(nn.Module):
    """Iterative network with data consistency."""
    def __init__(self, geometry, denoiser, k=5, alpha=0.1):
        super().__init__()
        self.geometry = geometry
        self.denoiser = denoiser
        self.k = k
        self.alpha = alpha
        self.projector = PyronnFanBeamProjector(geometry)

    def forward(self, fbp_init, sinogram):
        """
        fbp_init: (B, 1, H, W) initial guess
        sinogram: (B, 1, A, D) measured sinogram
        """
        x = fbp_init
        for _ in range(self.k):
            # CNN denoising step
            x_denoised = self.denoiser(x)
            # Data consistency: gradient of ||R*x - g||^2
            R_x = self.projector.forward_project(x_denoised)
            residual = R_x - sinogram
            R_T_residual = self.projector.fbp(residual)  # approximate gradient
            x = x_denoised - self.alpha * R_T_residual
        return x


def pretrain_denoiser(denoiser, fbp_images, truth_images, epochs, lr, device):
    """Pre-train U-Net on (FBP, truth) pairs."""
    denoiser.to(device)
    opt = torch.optim.Adam(denoiser.parameters(), lr=lr)
    n = fbp_images.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(n):
            idx = perm[i:i+1]
            fbp = fbp_images[idx].to(device)
            truth = truth_images[idx].to(device)
            pred = denoiser(fbp)
            loss = supervised_recon_loss(pred, truth, lambda_neg=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
        print(f"[pretrain] epoch {ep+1}/{epochs}  loss={running/n:.5f}", flush=True)


def finetune_itnet(itnet, train_sinos, train_fbps, epochs, lr, device):
    """Fine-tune ItNet end-to-end on (sinogram, FBP) pairs."""
    itnet.to(device)
    opt = torch.optim.Adam(itnet.parameters(), lr=lr)
    n = train_sinos.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(n):
            idx = perm[i:i+1]
            sino = train_sinos[idx].to(device)
            fbp = train_fbps[idx].to(device)
            pred = itnet(fbp, sino)
            # Target: FBP of clean sinogram (proxy for truth)
            # In practice, we'd use truth images
            target = fbp  # simplified: we want pred close to clean recon
            loss = supervised_recon_loss(pred, target, lambda_neg=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
        print(f"[finetune] epoch {ep+1}/{epochs}  loss={running/n:.5f}", flush=True)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg, default=str)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    # Build datasets
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # Compute FBPs
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        train_fbp = torch.clamp(proj.fbp(train_noisy), min=0.0)
        train_ref = proj.fbp(train_clean)
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)
        val_ref = proj.fbp(val_clean)

    t0 = time.time()

    # 1. Pre-train denoiser
    denoiser = SmallUNet(c=cfg["unet_c"])
    print(f"[solver] pre-training denoiser...", flush=True)
    pretrain_denoiser(
        denoiser, train_fbp, train_ph,  # use truth as target
        epochs=cfg["pretrain_epochs"], lr=cfg["pretrain_lr"], device=device)

    # 2. Build ItNet
    itnet = ItNet(
        geom, denoiser,
        k=cfg["itnet_k"], alpha=cfg["itnet_alpha"])

    # 3. Fine-tune (simplified: just evaluate on validation)
    itnet.eval()
    with torch.no_grad():
        chunk = 10
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            preds.append(itnet(val_fbp[i:i+chunk], val_noisy[i:i+chunk]))
        pred = torch.cat(preds, dim=0)

    train_time = time.time() - t0

    params_total = sum(p.numel() for p in itnet.parameters() if p.requires_grad)

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    val_fbp = val_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=val_fbp,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    val_score = val_ssim

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] ItNet: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f}  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="ItNet", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
