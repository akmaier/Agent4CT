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
    # Agentic iter (2026-06-02): when True, initialise Gaussian positions
    # from the FBP-of-noisy image instead of uniform random in [-0.6, 0.6].
    # Hypothesis: R²-G's hr=0 on dense-view breast comes from the cold
    # uniform init — many Gaussians waste capacity on the background. With
    # FBP-warm-start, Gaussians anchor on actual anatomy from iter 0 and
    # only have to refine local detail. Sampling uses FBP-intensity as a
    # per-pixel weight; positions are drawn from `multinomial(p ∝ fbp²)`
    # so brighter regions get more Gaussians.
    "gs_init_from_fbp": False,
    # When gs_init_from_fbp=True, initialise amp_raw from the FBP intensity
    # at each Gaussian's position (clamped to gs_amp_init_max), so the
    # warm start matches the truth scale on iter 0.
    "gs_amp_init_max": 0.05,
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


def _fbp_init_positions(fbp_image: torch.Tensor, n: int,
                          amp_init_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample Gaussian positions from an FBP image.

    Returns (pos, amp_init):
      pos:       (n, 2) in normalised [-1, +1] coords.
      amp_init:  (n,) — FBP intensity at each sampled position,
                 clamped to ``amp_init_max`` so a noisy FBP peak
                 doesn't drive the amp parameter to inf.

    Probability is proportional to clamp(fbp, 0, inf)**2 — squaring
    biases the sampling toward bright anatomy and away from the
    diffuse background. Falls back to uniform sampling if the FBP is
    all zero.
    """
    H, W = fbp_image.shape
    weights = fbp_image.clamp(min=0.0).reshape(-1) ** 2
    s = float(weights.sum())
    if not (s > 0 and math.isfinite(s)):
        # Degenerate FBP — fall back to uniform random.
        device = fbp_image.device
        pos = torch.empty(n, 2, device=device).uniform_(-0.6, 0.6)
        amp = torch.full((n,), 1e-3, device=device)
        return pos, amp
    idx_flat = torch.multinomial(weights / s, num_samples=n, replacement=True)
    yi = (idx_flat // W).to(torch.float32)
    xi = (idx_flat %  W).to(torch.float32)
    # Map (y, x) ∈ [0, H-1] to normalised [-1, +1].
    pos = torch.stack([
        (xi / (W - 1)) * 2.0 - 1.0,
        (yi / (H - 1)) * 2.0 - 1.0,
    ], dim=-1)
    # Jitter by ±0.5 pix to avoid duplicate positions at the same peak.
    pos = pos + torch.randn_like(pos) * (1.0 / max(H, W))
    amp_at_pos = fbp_image[yi.long(), xi.long()].clamp(min=1e-4, max=amp_init_max)
    return pos.to(fbp_image.device), amp_at_pos.to(fbp_image.device)


class GS2D(nn.Module):
    """N anisotropic 2-D Gaussians rasterised to an (H, W) image."""
    def __init__(self, n, image_size, amp_init, scale_init,
                  init_fbp: torch.Tensor | None = None,
                  amp_init_max: float = 0.05):
        super().__init__()
        self.n = n
        self.H = image_size
        if init_fbp is not None:
            # FBP-warm-start: sample positions ∝ fbp² and seed amp from
            # the FBP intensity at each picked location.
            pos0, amp_at_pos = _fbp_init_positions(init_fbp, n, amp_init_max)
            self.pos = nn.Parameter(pos0)
            # Per-Gaussian inverse-softplus(amp) so softplus(amp_raw) = amp.
            amp_at_pos = amp_at_pos.clamp(min=1e-5)
            self.amp_raw = nn.Parameter(torch.log(torch.expm1(amp_at_pos)))
        else:
            # Cold init (original behaviour).
            self.pos = nn.Parameter(torch.empty(n, 2).uniform_(-0.6, 0.6))
            inv_sp = math.log(math.expm1(amp_init))
            self.amp_raw = nn.Parameter(torch.full((n,), inv_sp))
        # Log-scales for x, y (positive via exp).
        self.log_scale = nn.Parameter(torch.full((n, 2), math.log(scale_init)))
        # Rotation angle.
        self.rot = nn.Parameter(torch.empty(n).uniform_(0.0, math.pi))

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


def fit_one_scene(noisy_sino, geom, cfg, device, fbp_init: torch.Tensor | None = None):
    proj = PyronnFanBeamProjector(geom).to(device)
    init_fbp = fbp_init if cfg.get("gs_init_from_fbp", False) else None
    model = GS2D(cfg["gs_n_gaussians"], cfg["image_size"],
                 cfg["gs_amp_init"], cfg["gs_scale_init"],
                 init_fbp=init_fbp,
                 amp_init_max=cfg.get("gs_amp_init_max", 0.05)).to(device)
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

    # Mayo: val split is a single patient (L277) -> reconstruct at its native
    # pixel-spacing (default ps mis-scales L277 ~5% and reads as broken).
    if cfg.get("dataset_kind") == "mayo_ldct_2d":
        from ddssl_ldct.staged_dataset import load_val_split as _lvs
        _g0 = FanBeamGeometry(
            image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
            n_angles=cfg["n_angles"], n_det=cfg["n_det"],
            det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
        try:
            _vps = _lvs("mayo_ldct_2d", "val", cfg["val_n"], device=device,
                        seed=cfg["seed"] + 1000, noise_i0=cfg["noise_i0"],
                        noise_sigma_e=cfg["noise_sigma_e"], geom=_g0,
                        return_ps=True)[-1]
            if _vps is not None:
                import numpy as _np
                cfg["pixel_spacing"] = round(float(_np.median(_np.asarray(_vps, float))), 5)
                print(f"[solver] Mayo val ps -> pixel_spacing={cfg['pixel_spacing']}", flush=True)
        except Exception as _e:
            print(f"[solver] val-ps probe failed ({_e}); using default ps", flush=True)

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
        # When gs_init_from_fbp=True, pass the per-slice FBP to seed positions.
        fbp_init_i = fbps[i, 0] if cfg.get("gs_init_from_fbp", False) else None
        pred_i = fit_one_scene(s, geom, cfg, device, fbp_init=fbp_init_i)
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
