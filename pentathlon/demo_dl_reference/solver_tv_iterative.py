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
    # TV-specific
    "tv_lambda":     0.001,
    "tv_iterations": 200,
    "tv_lr":         0.01,
    "tv_clip_max":   0.05,
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

    data_range = cfg["display_max"] - cfg["display_min"]
    # Compare against ground truth phantom (not noiseless FBP reference)
    val_psnr = float(psnr(pred, val_ph, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(fbp_init, val_ph, data_range=data_range).cpu())
    baseline_rmse = float(((fbp_init - val_ph) ** 2).mean().sqrt().cpu())
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
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(fbp_init[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"TV recon  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 3].set_title("phantom" if i == 0 else "")
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
    print(f"[solver] TV recon: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
