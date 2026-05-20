"""Reference: Dual-Domain Denoising with trainable bilateral filters (Wagner et al. 2022).

This is the ultra-low-parameter alternative to the U-Net dual-domain approach.
Each denoiser has only 4 trainable parameters (sigma_x, sigma_y, sigma_r) plus
kernel size, compared to 230K+ for SmallUNet.

Wagner showed dual bilateral filters achieve 97% of dual U-Net SSIM with
only 8 total parameters (4 for projection + 4 for image domain) on abdomen CT.

Architecture: TrainableBilateralFilter2d in both projection and image domain.
Training: Same Noise2Inverse self-supervision as the U-Net variant.
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
from ddssl_ldct.models import TrainableBilateralFilter2d
from ddssl_ldct.training import DualDomainPipeline
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # Bilateral filter initial parameters
    "proj_kernel":   5,       # kernel size for projection denoiser
    "proj_sx":       1.0,     # initial sigma_x (angular smoothness)
    "proj_sy":       2.0,     # initial sigma_y (detector smoothness)
    "proj_sr":       0.02,    # initial sigma_r (range/attenuation)
    "img_kernel":    7,       # kernel size for image denoiser
    "img_sx":        1.5,     # initial sigma_x (spatial)
    "img_sy":        1.5,     # initial sigma_y (spatial)
    "img_sr":        0.02,    # initial sigma_r (intensity range)
    # Training
    "epochs":        20,
    "batch_size":    1,
    "lr":            5e-3,     # Wagner used 5e-3 for BFs (vs 5e-5 for U-Nets)
    "optimizer":     "adam",
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


def main(out_dir: Path, cfg: dict | None = None) -> dict:
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

    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = torch.clamp(R_full.fbp(val_noisy), min=0.0)

    # Build bilateral filter denoisers
    proj_dn = TrainableBilateralFilter2d(
        kernel_size=cfg["proj_kernel"],
        sigma_x=cfg["proj_sx"],
        sigma_y=cfg["proj_sy"],
        sigma_r=cfg["proj_sr"],
    )
    img_dn = TrainableBilateralFilter2d(
        kernel_size=cfg["img_kernel"],
        sigma_x=cfg["img_sx"],
        sigma_y=cfg["img_sy"],
        sigma_r=cfg["img_sr"],
    )
    pipe = DualDomainPipeline(geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn).to(device)

    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] Bilateral dual-domain params: {params_total} (proj={sum(p.numel() for p in proj_dn.parameters())}, img={sum(p.numel() for p in img_dn.parameters())})", flush=True)

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

        # Log learned sigmas
        with torch.no_grad():
            psx = float(torch.exp(proj_dn.log_sx).cpu())
            psy = float(torch.exp(proj_dn.log_sy).cpu())
            psr = float(torch.exp(proj_dn.log_sr).cpu())
            isx = float(torch.exp(img_dn.log_sx).cpu())
            isy = float(torch.exp(img_dn.log_sy).cpu())
            isr = float(torch.exp(img_dn.log_sr).cpu())
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  "
              f"proj(σx={psx:.3f} σy={psy:.3f} σr={psr:.4f})  "
              f"img(σx={isx:.3f} σy={isy:.3f} σr={isr:.4f})", flush=True)

    train_time = time.time() - t0

    # Validation
    pipe.eval()
    with torch.no_grad():
        chunk = cfg.get("val_chunk", 10)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            preds.append(pipe.predict(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    ld_fbp = ld_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=ld_fbp,
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
        "params_total": params_total, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time,
        "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Dual-domain BF: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} params={params_total}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="DD-BF", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
