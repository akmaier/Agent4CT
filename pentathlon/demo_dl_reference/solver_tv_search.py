"""TV-regularized iterative reconstruction — agent-editable solver.

Hyperparameter search space:
  tv_lambda:     [0.0001, 0.01]    (regularization strength)
  tv_iterations: [50, 500]          (optimization steps)
  tv_lr:         [0.001, 0.1]       (gradient descent step size)
  tv_clip_max:   [0.03, 0.08]       (hard constraint upper bound)
  tv_decay:      [0.0, 0.05]        (step size decay rate per iteration)
  tv_optimizer:  ["gd", "adam"]      (optimizer choice)
  tv_init:       ["fbp", "zeros"]    (initialization)

The agent edits CONFIG below and submits via sbatch.
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
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison


# ---------------------------------------------------------------------------
#  CONFIG — agent edits this block
# ---------------------------------------------------------------------------
CONFIG = {
    # Geometry (fixed)
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,
    # Data (fixed for fair comparison)
    "train_n":       0,       # TV needs no training
    "val_n":         100,
    "noise_i0":      1e5,
    "noise_sigma_e": 10.0,
    "seed":          42,
    "display_min":   0.0,
    "display_max":   0.05,
    # TV hyperparameters (AGENT EDITS THESE)
    "tv_lambda":     0.001,   # regularization weight
    "tv_iterations": 200,     # number of iterations
    "tv_lr":         0.01,    # step size
    "tv_clip_max":   0.05,    # hard upper bound
    "tv_decay":      0.01,    # step decay per iteration
    "tv_optimizer":  "gd",     # "gd" or "adam"
    "tv_init":       "fbp",    # "fbp" or "zeros"
}


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


def total_variation(x):
    """Isotropic TV."""
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = torch.nn.functional.pad(dx, (0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 1, 0, 0))
    return torch.sum(torch.sqrt(dx ** 2 + dy ** 2 + 1e-8))


def tv_reconstruction(proj, sino, fbp_init, cfg, device):
    """TV reconstruction with configurable optimizer and schedule."""
    lam = cfg["tv_lambda"]
    iterations = cfg["tv_iterations"]
    lr = cfg["tv_lr"]
    clip_max = cfg["tv_clip_max"]
    decay = cfg["tv_decay"]
    optimizer = cfg["tv_optimizer"]
    init = cfg["tv_init"]

    # Initialize
    if init == "zeros":
        f = torch.zeros_like(fbp_init).requires_grad_(True)
    else:
        f = fbp_init.clone().requires_grad_(True)

    # Optimizer
    if optimizer == "adam":
        opt = torch.optim.Adam([f], lr=lr)
    else:
        opt = None  # manual GD

    for it in range(iterations):
        if opt is not None:
            opt.zero_grad()
        elif f.grad is not None:
            f.grad.zero_()

        # Forward model
        Rf = proj.forward_project(f)
        data_residual = Rf - sino
        data_term = 0.5 * torch.mean(data_residual ** 2)

        # TV regularization
        tv_term = total_variation(f)

        # Loss
        loss = data_term + lam * tv_term
        loss.backward()

        # Step
        with torch.no_grad():
            if opt is not None:
                opt.step()
                # Adam has its own schedule, just decay LR
                if decay > 0:
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr / (1.0 + decay * (it + 1))
            else:
                # Manual GD with decay
                step = lr / (1.0 + decay * it) if decay > 0 else lr
                f -= step * f.grad

            # Hard constraints
            f.clamp_(0.0, clip_max)

        if (it + 1) % 50 == 0 or it == 0:
            print(f"[TV] iter {it+1}/{iterations}  data={data_term.item():.6f} "
                  f"tv={tv_term.item():.6f}  loss={loss.item():.6f}", flush=True)

    return f.detach()


def main(out_dir: Path, cfg_override: dict | None = None) -> dict:
    # Check for environment-based config override
    import os
    env_config_path = os.environ.get("TV_CONFIG_PATH")
    if env_config_path and Path(env_config_path).exists():
        with open(env_config_path) as f:
            env_cfg = json.load(f)
        cfg = {**CONFIG, **env_cfg}
        print(f"[solver] Loaded config from {env_config_path}")
    elif cfg_override:
        cfg = {**CONFIG, **cfg_override}
    else:
        cfg = CONFIG.copy()
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] TV reconstruction", flush=True)
    print(f"[solver] config={json.dumps({k:v for k,v in cfg.items() if k.startswith('tv_')})}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)
        val_ref = proj.fbp(val_clean)

    # TV reconstruction
    t0 = time.time()
    pred = tv_reconstruction(proj, val_noisy, val_fbp, cfg, device)
    train_time = time.time() - t0

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp(cfg["display_min"], cfg["display_max"])
    val_fbp = torch.clamp(val_fbp, cfg["display_min"], cfg["display_max"])
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=val_fbp,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]

    # Secondary: against noiseless FBP (practical denoising metric, also calibrated)
    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp(cfg["display_min"], cfg["display_max"])
    val_fbp = torch.clamp(val_fbp, cfg["display_min"], cfg["display_max"])
    metrics_fbp = evaluate_calibrated(
        pred, val_ref,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    val_psnr_fbp = metrics_fbp["val_psnr"]
    val_ssim_fbp = metrics_fbp["val_ssim"]
    baseline_rmse_fbp = float(((val_fbp - val_ref) ** 2).mean().sqrt().cpu())
    headroom_fbp = max(0.0, 1.0 - val_rmse / max(baseline_rmse_fbp, 1e-12))

    print(f"[solver] vs phantom:  SSIM={val_ssim:.4f} PSNR={val_psnr:.2f} headroom={headroom:.4f}")
    print(f"[solver] vs FBP ref:  SSIM={val_ssim_fbp:.4f} PSNR={val_psnr_fbp:.2f} headroom={headroom_fbp:.4f}")

    result = {
        "val_score": val_ssim,              # Primary: SSIM vs phantom
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "val_rmse": val_rmse,
        "headroom": headroom,
        # Secondary metrics
        "val_ssim_fbp": val_ssim_fbp,
        "val_psnr_fbp": val_psnr_fbp,
        "headroom_fbp": headroom_fbp,
        # Baselines
        "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse,
        "baseline_rmse_fbp": baseline_rmse_fbp,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": 0.0,
        "train_n": 0,
        "val_n": cfg["val_n"],
        "train_time_s": train_time,
        "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] TV recon: val_score={val_ssim:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} time={train_time:.1f}s  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="TV", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
