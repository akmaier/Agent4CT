"""Manhart 2013 — Statistically-Ray-Weighted (PWLS) TV iterative reconstruction.

Reference: M. T. Manhart, A. Fieselmann, Y. Deuerling-Zheng, A. K. Maier,
M. Kowarschik, "Dynamic Reconstruction with Statistical Ray Weighting for
C-Arm CT Perfusion Imaging" (papers/Manhart13-DRW.pdf).

Carry-over for STATIC reconstruction
-------------------------------------
The paper's contribution that we port here is the **statistical ray weighting**
(PWLS) of the data-consistency term (eqs. 3-8). Manhart models the line-integral
noise as Gaussian with variance sigma_L^2 = 1/i, where the detected photon count
is  i = i^S * exp(-p)  (i^S = incident flux, p = measured line integral, eq. 4).
The inverse-variance weight is therefore proportional to the photon count:

        w_i  ∝  1 / sigma_L^2  =  i  =  i^S * exp(-p_i).

In the PWLS / penalized-weighted-least-squares objective (eq. 6) this appears
as a diagonal weighting matrix D = diag{1/sigma_i^2}, and Manhart's eq. 8 uses
the approximation (omitting constants)

        D = diag{ exp(-p_1), ..., exp(-p_N) },

so the weighted gradient update is

        x  <-  x - step * [ A^T ( D (A x - p) ) + lambda * grad_TV(x) ].

High-attenuation rays (large p) carry few photons -> high noise -> are
down-weighted; low-attenuation rays dominate the data term. This is exactly the
classical TV recon (½‖Ax-p‖² + λ·TV) with the diagonal statistical weight W
inserted INTO the data residual before the back-projection A^T.

        x  <-  x - step * [ A^T( W ⊙ (Ax - p) ) + λ·grad_TV(x) ],
        W_i = exp( -gamma * p_i )      (gamma = 1 is Manhart; gamma = 0 ⇒ plain TV)

W is normalized (by its max or mean) for step-size stability so the effective
step magnitude matches the plain-TV base solver. With gamma = 0 this solver is
numerically identical to solver_tv_search (W ≡ 1 / norm = 1).

Implementation note
-------------------
We realise the eq.-8 weighted back-projection through autograd: the data term is
written as  0.5 * mean( W ⊙ (A x - p)^2 ). Differentiating w.r.t. x gives
A^T( W ⊙ (A x - p) ) — exactly the PWLS gradient — so we get the weighted
back-projection for free from PYRO-NN's differentiable forward projector, with
no extra projector call.

This is a CLASSICAL model-based iterative solver (like TV_iterative); the
auto-research loop tunes its hyperparameters. No neural network is trained.

Everything non-algorithmic (data loading, projector access, Mayo per-sample-ps
probe, evaluate_calibrated scoring, return dict, comparison figure, the
TV_CONFIG_PATH env override) mirrors solver_tv_search.py EXACTLY.

Hyperparameter search space (agentic-search knobs):
  PWLS ray-weighting
    pwls_gamma:       [0.0, 2.0]   ray-weight exponent in W=exp(-gamma*p);
                                   1.0 = Manhart, 0.0 = plain TV (default 1.0)
    pwls_N0:          incident flux i^S for the weight; default = dataset
                      noise_i0. The line-integral weight exp(-gamma*p) does NOT
                      depend on N0 (the i^S factor and constants are absorbed by
                      pwls_weight_norm), so this is kept only as an explicit,
                      documented lever and a record of the assumed incident flux.
    pwls_weight_norm: "mean" | "max"  how W is normalized before entering the
                      residual (default "mean"; divides W by its mean/max so the
                      data-term scale — and thus the usable step size — matches
                      plain TV).
  TV (inherited from solver_tv_search)
    tv_lambda:        [0.0001, 0.01]  regularization strength
    tv_iterations:    [50, 500]       optimization steps (per val slice)
    tv_lr:            [0.001, 0.1]     gradient-descent step size
    tv_clip_max:      [0.03, 0.08]     hard upper bound (lower clip 0)
    tv_decay:         [0.0, 0.05]      step-size decay per iteration
    tv_optimizer:     "gd" | "adam"    optimizer choice
    tv_init:          "fbp" | "zeros"  initialization
    seed:             RNG seed

The agent edits CONFIG below (or supplies it via TV_CONFIG_PATH) and submits
via sbatch.
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
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


# ---------------------------------------------------------------------------
#  CONFIG — agent edits this block
# ---------------------------------------------------------------------------
CONFIG = {
    **DEMO_DL_DEFAULTS,
    "train_n":       0,        # PWLS-TV needs no training
    # --- PWLS statistical ray-weighting (Manhart 2013) -------------------
    "pwls_gamma":       1.0,   # ray-weight exponent; 1=Manhart, 0=plain TV
    "pwls_N0":          None,  # incident flux i^S; None -> dataset noise_i0
    "pwls_weight_norm": "mean",  # "mean" | "max" normalization of W
    # --- TV hyperparameters (mirror solver_tv_search) --------------------
    "tv_lambda":     0.001,   # regularization weight
    "tv_iterations": 200,     # number of iterations
    "tv_lr":         0.01,    # step size
    "tv_clip_max":   0.09,    # hard upper bound (2026-06-29: was 0.05 display window; raised to physical mu)
    "tv_decay":      0.01,    # step decay per iteration
    "tv_optimizer":  "gd",     # "gd" or "adam"
    "tv_init":       "fbp",    # "fbp" or "zeros"
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
    """Isotropic TV."""
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = torch.nn.functional.pad(dx, (0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 1, 0, 0))
    return torch.sum(torch.sqrt(dx ** 2 + dy ** 2 + 1e-8))


def pwls_ray_weights(sino, gamma, weight_norm):
    """Manhart statistical ray weight W_i = exp(-gamma * p_i), normalized.

    sino       : measured line integrals p (= A x_true), shape [B,1,A,Det]
    gamma      : ray-weight exponent (0 -> uniform weights == plain TV)
    weight_norm: "mean" | "max" — divides W so its central tendency is ~1, so
                 the data-term magnitude (hence the usable step size) matches
                 plain TV.

    Returns a detached weight tensor broadcastable against the sinogram.
    Line integrals are non-negative after -log(I/I0); we clamp at 0 for safety
    so W in (0, 1]. With gamma == 0, W == 1 everywhere and this reduces exactly
    to the plain-TV data residual.
    """
    if gamma == 0.0:
        return torch.ones_like(sino)
    p = sino.clamp_min(0.0)
    W = torch.exp(-float(gamma) * p)
    if weight_norm == "max":
        denom = W.amax()
    else:  # "mean" (default)
        denom = W.mean()
    denom = torch.clamp(denom, min=1e-12)
    return (W / denom).detach()


def tv_reconstruction(proj, sino, fbp_init, cfg, device):
    """PWLS-weighted TV reconstruction with configurable optimizer/schedule.

    Identical to solver_tv_search.tv_reconstruction except the data term is
    weighted by the Manhart statistical ray weight W = exp(-gamma*p):

        data_term = 0.5 * mean( W ⊙ (R f - p)^2 )

    Differentiating w.r.t. f yields the eq.-8 weighted back-projection
    A^T( W ⊙ (R f - p) ) automatically through autograd.
    """
    lam = cfg["tv_lambda"]
    iterations = cfg["tv_iterations"]
    lr = cfg["tv_lr"]
    clip_max = max(cfg["tv_clip_max"], cfg["display_max"])   # 2026-06-29: box >= physical mu (display_max=0.09); 0.05 truncated bone
    decay = cfg["tv_decay"]
    optimizer = cfg["tv_optimizer"]
    init = cfg["tv_init"]
    gamma = cfg.get("pwls_gamma", 1.0)
    weight_norm = cfg.get("pwls_weight_norm", "mean")

    # PWLS statistical ray weight from the measured line integrals (eq. 8).
    # Constant across iterations -> compute once. Detached: it is a fixed
    # diagonal preconditioner D, not a variable being optimized.
    W = pwls_ray_weights(sino, gamma, weight_norm)
    print(f"[PWLS] gamma={gamma} norm={weight_norm}  "
          f"W: min={W.min().item():.4f} mean={W.mean().item():.4f} "
          f"max={W.max().item():.4f}", flush=True)

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
        # PWLS statistical ray weighting (Manhart 2013, eqs. 6-8): weight the
        # squared residual by W so the autograd gradient is A^T(W ⊙ (Rf - p)).
        data_term = 0.5 * torch.mean(W * data_residual ** 2)

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

            # μ≥0 floor only. Upper clamp REMOVED 2026-06-30: clip_max truncated
            # bone (μ up to 0.0814) — same bug class as the metric clamp. Vestigial.
            f.clamp_min_(0.0)

        if (it + 1) % 50 == 0 or it == 0:
            print(f"[PWLS-TV] iter {it+1}/{iterations}  data={data_term.item():.6f} "
                  f"tv={tv_term.item():.6f}  loss={loss.item():.6f}", flush=True)

    return f.detach()


def main(out_dir: Path, cfg_override: dict | None = None) -> dict:
    # Check for environment-based config override (mirror solver_tv_search)
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

    # Resolve PWLS incident flux: default to the dataset's noise_i0. Documented
    # explicit lever; the line-integral weight exp(-gamma*p) is independent of
    # N0 (constants absorbed by pwls_weight_norm), so this is recorded, not used
    # to scale W. Kept so the assumed flux is auditable in the config.
    if cfg.get("pwls_N0") is None:
        cfg["pwls_N0"] = cfg.get("noise_i0")

    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] PWLS-TV (Manhart 2013) reconstruction", flush=True)
    print(f"[solver] config={json.dumps({k:v for k,v in cfg.items() if k.startswith('tv_') or k.startswith('pwls_')})}", flush=True)
    torch.manual_seed(cfg["seed"])

    # Mayo: the val split is a single patient (L277) -> reconstruct at its
    # native pixel-spacing so the FBP / TV recon lands on the un-resampled
    # truth grid (the default ps mis-scales L277 by ~5% and reads as broken).
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
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)
        val_ref = proj.fbp(val_clean)

    # PWLS-TV reconstruction
    t0 = time.time()
    pred = tv_reconstruction(proj, val_noisy, val_fbp, cfg, device)
    train_time = time.time() - t0

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

    # Secondary vs-noiseless-FBP diagnostics. Computed DIRECTLY (psnr/ssim on the
    # FBP-calibrated pred) rather than via a SECOND evaluate_calibrated call, so
    # there is NO second AGENT4CT_SAVE_RECON dump that could overwrite the primary
    # pred/phantom/baseline recon. The earlier env-pop guard proved unreliable on
    # the test-scoring path, leaving recon_raw.npz with truth=val_ref(FBP) and no
    # baseline -> test_hr uncomputable. One evaluate_calibrated call = one clean save.
    from ddssl_ldct.metrics import intensity_calibrate as _ical
    pred = pred.clamp_min(0.0)
    val_fbp = val_fbp.clamp_min(0.0)
    _dr_fbp = float(val_ref.max() - val_ref.min())
    _dr_fbp = _dr_fbp if _dr_fbp > 1e-6 else 1e-6
    _pred_fbpcal = _ical(pred, val_ref, display_max=cfg["display_max"]).clamp_min(0.0)
    val_psnr_fbp = float(psnr(_pred_fbpcal, val_ref, data_range=_dr_fbp).cpu())
    val_ssim_fbp = float(ssim(_pred_fbpcal, val_ref, data_range=_dr_fbp).cpu())
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
    print(f"[solver] PWLS-TV recon: val_score={val_ssim:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} time={train_time:.1f}s  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="PWLS-TV", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
