"""Reference: R2Gaussian-lite — Radiative Gaussian Splatting for sparse-view CT.

Per-scene optimisation of N anisotropic 2-D Gaussian primitives whose
attenuation contributions sum into an μ image, which is then projected
through PyronnFanBeamProjector and matched against the noisy sinogram.

Adapted from:
  Zha R., Cheng L., Han L., Gao C., Zhang Y.
  "R²-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic
  Reconstruction." NeurIPS 2024. arXiv:2405.20693
  (code: github.com/Ruyi-Zha/r2_gaussian)

"Lite" variant: rasterises Gaussians directly into the (H, W) μ image
via per-pixel evaluation of the analytic 2-D Gaussian (no tiled
splatting kernels needed at 512²). Forward projection then runs through
the existing PyronnFanBeamProjector for differentiable data fidelity.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
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
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "val_n": 20,
    "gs_n_gaussians": 1024,
    "gs_n_iter": 600,
    "gs_lr_pos": 5e-3,
    "gs_lr_scale": 1e-2,
    "gs_lr_amp": 1e-2,
    "gs_lr_rot": 1e-2,
    "gs_amp_init": 0.01,
    "gs_scale_init": 0.04,    # in normalised [-1,1] coords; ~10 px at 512
    "gs_tv_weight": 1e-4,
    "gs_n_clip": 0.05,
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


class GS2D(nn.Module):
    """N anisotropic 2-D Gaussians rasterised to an (H, W) image."""
    def __init__(self, n, image_size, amp_init, scale_init):
        super().__init__()
        self.n = n
        self.H = image_size
        # Positions in normalised [-1, 1].
        self.pos = nn.Parameter(torch.empty(n, 2).uniform_(-0.6, 0.6))
        # Log-scales for x, y (positive via exp).
        self.log_scale = nn.Parameter(torch.full((n, 2), math.log(scale_init)))
        # Rotation angle.
        self.rot = nn.Parameter(torch.empty(n).uniform_(0.0, math.pi))
        # Amplitude (positive via softplus).
        inv_sp = math.log(math.expm1(amp_init))
        self.amp_raw = nn.Parameter(torch.full((n,), inv_sp))

    def forward(self):
        device = self.pos.device
        H = self.H
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, H, device=device),
            indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)               # (H, H, 2)
        sx = torch.exp(self.log_scale[:, 0]).clamp(min=1e-3)
        sy = torch.exp(self.log_scale[:, 1]).clamp(min=1e-3)
        amp = F.softplus(self.amp_raw)
        cos, sin = torch.cos(self.rot), torch.sin(self.rot)
        # Accumulate Gaussian contributions in chunks to limit memory.
        out = torch.zeros(H, H, device=device)
        chunk = 64
        for s in range(0, self.n, chunk):
            e = min(s + chunk, self.n)
            d = coords.unsqueeze(2) - self.pos[s:e].view(1, 1, e - s, 2)
            # Rotate into Gaussian axes.
            dx = d[..., 0] * cos[s:e] + d[..., 1] * sin[s:e]
            dy = -d[..., 0] * sin[s:e] + d[..., 1] * cos[s:e]
            quad = (dx / sx[s:e]) ** 2 + (dy / sy[s:e]) ** 2
            gauss = torch.exp(-0.5 * quad)
            out = out + (gauss * amp[s:e]).sum(dim=-1)
        return out


def _tv(img):
    return (img[..., 1:, :] - img[..., :-1, :]).abs().mean() + \
           (img[..., :, 1:] - img[..., :, :-1]).abs().mean()


def fit_one_scene(noisy_sino, geom, cfg, device):
    proj = PyronnFanBeamProjector(geom).to(device)
    model = GS2D(cfg["gs_n_gaussians"], cfg["image_size"],
                 cfg["gs_amp_init"], cfg["gs_scale_init"]).to(device)
    opt = torch.optim.Adam([
        {"params": [model.pos],       "lr": cfg["gs_lr_pos"]},
        {"params": [model.log_scale], "lr": cfg["gs_lr_scale"]},
        {"params": [model.amp_raw],   "lr": cfg["gs_lr_amp"]},
        {"params": [model.rot],       "lr": cfg["gs_lr_rot"]},
    ])
    for it in range(cfg["gs_n_iter"]):
        mu = model().clamp(0.0, cfg["gs_n_clip"]).unsqueeze(0).unsqueeze(0)
        sino_pred = proj.forward_project(mu)
        loss = F.mse_loss(sino_pred, noisy_sino) + cfg["gs_tv_weight"] * _tv(mu) + 1.0 * negativity_penalty(mu)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mu = model().clamp(0.0, cfg["gs_n_clip"]).unsqueeze(0).unsqueeze(0)
    return mu


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("R2G_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] device={device}  cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('gs_')}, default=str)}", flush=True)

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
    proj_full = PyronnFanBeamProjector(geom).to(device)
    phs, _, noisys = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    with torch.no_grad():
        fbps = torch.clamp(proj_full.fbp(noisys), min=0.0)

    outer_wall = float(cfg.get("gs_outer_wall_s", 600))
    t0 = time.time(); preds = []
    for i in range(cfg["val_n"]):
        s = noisys[i:i+1]
        pred_i = fit_one_scene(s, geom, cfg, device)
        preds.append(pred_i.detach())
        if (i + 1) % 5 == 0:
            print(f"[fit] {i+1}/{cfg['val_n']}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > outer_wall:
            print(f"[fit] outer wall {outer_wall:.0f}s hit at sample {i+1}",
                  flush=True); break
    train_time = time.time() - t0
    pred = torch.cat(preds, 0)
    val_ph = phs[:pred.shape[0]]; val_fbp = fbps[:pred.shape[0]]

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
    n_params = cfg["gs_n_gaussians"] * 6   # pos(2) + scale(2) + rot(1) + amp(1)
    params_M = n_params / 1e6

    result = {
        "val_score": val_ssim, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_M,
        "train_n": 0, "val_n": int(pred.shape[0]),
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] R2Gauss: hr={headroom:.4f}  SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="R2GS", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
