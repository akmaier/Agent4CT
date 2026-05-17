"""Reference: Pure Pyro-NN FBP baseline reconstruction.

This is the simplest possible reconstruction: project the noisy sparse-view
sinogram through PYRO-NN's ramp-filtered FBP. No learning. This establishes
the lower bound (score = 0, headroom = 0 by definition).

Outputs (written into the directory passed as argv[1]):
    - result.json with keys: val_score, val_psnr, val_ssim, val_rmse,
      headroom, baseline_score, oracle_score, params_M, train_n,
      change_class, rationale, advice_for_others
    - comparison.png — reference / FBP / phantom
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim


CONFIG = {
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,
    "train_n":       400,
    "val_n":         100,
    "noise_i0":      1e5,
    "noise_sigma_e": 10.0,
    "seed":          42,
    "display_min":   0.0,
    "display_max":   0.05,
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


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
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

    t0 = time.time()
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    R_full = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        val_ref = R_full.fbp(val_clean)   # reference (noiseless FBP)
        pred = torch.clamp(R_full.fbp(val_noisy), min=0.0)  # our "recon" = standard FBP
        pred = pred.clamp(0.0, cfg["display_max"])
    train_time = time.time() - t0

    # CORRECT: Compare against ground truth phantom (not noiseless FBP)
    data_range = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ph, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
    # Baseline: noisy FBP vs phantom (actual challenge baseline)
    baseline_psnr = float(psnr(ld_fbp, val_ph, data_range=data_range).cpu())
    baseline_rmse = float(((ld_fbp - val_ph) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim

    # Comparison figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("truth" if i == 0 else "")
            ax[i, 1].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"FBP recon  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            residual = (pred[i, 0] - val_ph[i, 0]).cpu()
            vmax_res = max(abs(residual.min()), abs(residual.max()))
            ax[i, 3].imshow(residual, cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res)
            ax[i, 3].set_title("residual" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "params_M": 0.0, "train_n": 0, "val_n": cfg["val_n"],
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] BASELINE FBP: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
