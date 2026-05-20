"""Reference: Iterative TV-regularized reconstruction (non-learned).

Solves:  min_f  ||Rf - g||_2^2 + lambda_TV * TV(f)
using gradient descent with explicit forward/back-projector from PYRO-NN.

This is a classical model-based iterative reconstruction (MBIR) baseline.
No neural network is used.

Outputs same format as the autoresearch solvers.
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
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # TV-specific
    "tv_lambda":     0.001,
    "tv_iterations": 200,
    "tv_lr":         0.01,
    "tv_clip_max":   0.05,
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    # Dispatches on AGENT4CT_DATASET / cfg["dataset_kind"]. Phantom path
    # is backwards-compatible; staged paths load from disk.
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom)


def total_variation(x):
    """Isotropic TV: sum of sqrt((dx)^2 + (dy)^2)."""
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    # Pad to match original shape
    dx = torch.nn.functional.pad(dx, (0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 1, 0, 0))
    return torch.sum(torch.sqrt(dx ** 2 + dy ** 2 + 1e-8))


def tv_reconstruction(proj, sino, fbp_init, lam, iterations, lr, clip_max, device):
    """
    Gradient descent on: 0.5 * ||R*f - g||^2 + lambda * TV(f)
    Using vanilla gradient descent with line-search-like decay.
    """
    f = fbp_init.clone().requires_grad_(True)
    
    for it in range(iterations):
        if f.grad is not None:
            f.grad.zero_()
        
        # Forward projection
        Rf = proj.forward_project(f)
        data_residual = Rf - sino
        data_term = 0.5 * torch.mean(data_residual ** 2)
        
        # TV regularization
        tv_term = total_variation(f)
        
        loss = data_term + lam * tv_term
        loss.backward()
        
        with torch.no_grad():
            # Adaptive step size that decays
            step = lr / (1.0 + 0.01 * it)
            f -= step * f.grad
            # Hard constraints
            f.clamp_(0.0, clip_max)
        
        if (it + 1) % 50 == 0:
            print(f"[TV] iter {it+1}/{iterations}  data={data_term.item():.6f} "
                  f"tv={tv_term.item():.6f}  step={step:.4f}", flush=True)

    return f.detach()


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

    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        val_ref = proj.fbp(val_clean)
        fbp_init = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # Run TV reconstruction
    t0 = time.time()
    pred = tv_reconstruction(
        proj, val_noisy, fbp_init,
        lam=cfg["tv_lambda"],
        iterations=cfg["tv_iterations"],
        lr=cfg["tv_lr"],
        clip_max=cfg["tv_clip_max"],
        device=device,
    )
    train_time = time.time() - t0

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    fbp_init = fbp_init.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=fbp_init,
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
        "params_M": 0.0, "train_n": 0, "val_n": cfg["val_n"],
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] TV recon: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f}  (intensity-calibrated)", flush=True)

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
