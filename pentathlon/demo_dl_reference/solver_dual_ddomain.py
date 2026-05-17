"""Reference: Dual-Domain Denoising with learned denoisers (Wagner et al. 2023).

This is the "standard" learned approach used by the autoresearch agents.
Two learned denoisers (projection-domain + image-domain) trained end-to-end
via Noise2Inverse self-supervision.

Architecture: SmallUNet (c=16) in both domains.
Adapted from the main dl_sparse_view solver for the demo track.
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
from ddssl_ldct.simulate import simulate_low_dose, split_projections
from ddssl_ldct.models import SmallUNet
from ddssl_ldct.training import DualDomainPipeline
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
    # Training
    "epochs":        8,
    "batch_size":    1,
    "lr":            1e-3,
    "optimizer":     "adam",
    "unet_c":        16,
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
    # Check for environment-based config override
    import os
    env_config_path = os.environ.get("DD_CONFIG_PATH")
    if env_config_path and Path(env_config_path).exists():
        with open(env_config_path) as f:
            env_cfg = json.load(f)
        cfg = {**CONFIG, **env_cfg}
        print(f"[solver] Loaded config from {env_config_path}")
    elif cfg is not None:
        cfg = {**CONFIG, **cfg}
    else:
        cfg = CONFIG.copy()
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

    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = torch.clamp(R_full.fbp(val_noisy), min=0.0)

    # Build denoisers
    c = cfg["unet_c"]
    proj_dn = SmallUNet(c=c)
    img_dn = SmallUNet(c=c)
    pipe = DualDomainPipeline(geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn).to(device)

    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] params = {params_total/1e6:.3f} M", flush=True)

    opt = torch.optim.Adam(pipe.parameters(), lr=cfg["lr"])

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
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}", flush=True)
    train_time = time.time() - t0

    # Validation
    pipe.eval()
    with torch.no_grad():
        chunk = cfg.get("val_chunk", 10)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            preds.append(pipe.predict(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)
        pred = pred.clamp(0.0, cfg["display_max"])

    data_range = cfg["display_max"] - cfg["display_min"]
    # Compare against ground truth phantom (not noiseless FBP reference)
    val_psnr = float(psnr(pred, val_ph, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
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
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"dual-domain  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
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
        "params_M": params_total / 1e6, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Dual-domain: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
