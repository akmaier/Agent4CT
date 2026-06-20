"""Reference: few-step DC-guided reconstruction from a fast-diffusion prior.

Inference-time twin of ``solver_fast_diffusion.py``. It loads a frozen
flow-matching checkpoint (pixel OR wavelet domain, auto-detected from the
ckpt) and reconstructs an image from a measured sinogram with a handful of
Euler steps plus data-consistency guidance:

  * **DPS** (Chung et al. 2023, arXiv:2209.14687) — at each Euler step,
    steer x_t down the gradient of the sinogram residual through the clean
    estimate (and, for the wavelet prior, through the differentiable IDWT).
  * **optional hard CG DC-step** — a REAL conjugate-gradient solve of
    ``min_x ||A x - y||^2`` warm-started at the current clean estimate, then
    a DETERMINISTIC re-embed at the next flow-time level that reuses the
    SAME noise direction (no fresh randn, no stale eps).

This is the corrected sampler: the audit flagged ``solver_diffusion_recon``'s
``dc_step_cg`` as a mislabeled steepest-descent "CG" plus a stale-eps +
fresh-randn re-noise; both bugs are fixed here (true CG with residual /
conjugate direction / beta; deterministic same-noise re-embed).

A single ckpt path (``fd_recon_ckpt``, env-overridable ``FAST_RECON_CKPT``)
selects which of the four trained priors to reconstruct with, so this one
file serves all four recon experiments.

Citations:
  Liu X. et al. "Flow Straight and Fast." 2022. arXiv:2209.03003
  Lipman Y. et al. "Flow Matching for Generative Modeling." 2022. arXiv:2210.02747
  Friedrich P. et al. "WDM." 2024. arXiv:2402.19043
  Chung H. et al. "Diffusion Posterior Sampling." ICLR 2023. arXiv:2209.14687
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated, make_4panel_comparison
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS
# Reuse the model + transforms from the trainer so a single definition is the
# source of truth across both solvers.
from pentathlon.demo_dl_reference.solver_fast_diffusion import (
    FlowUNet, haar_dwt, haar_idwt, from_model_domain, to_model_domain,
)


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "val_n": 20,
    # Path to a checkpoint produced by solver_fast_diffusion.py. Env override.
    "fd_recon_ckpt":      "/cluster/maier/Agent4CT/checkpoints/fast_diffusion_unconstrained.pt",
    # Sampler hyperparameters (what the search varies).
    "fd_recon_steps":     8,        # number of Euler steps from init_t down to 0
    "fd_recon_init":      "fbp",    # "fbp" | "noise"
    "fd_recon_init_t":    0.7,      # for fbp init, embed FBP at this flow time
    "fd_recon_eta":       1.0,      # DPS guidance scale (adaptive, Chung eq.12)
    # Optional hard CG data-consistency step.
    "fd_recon_dc":        0,        # 0 = off | 1 = hard CG DC-step each Euler step
    "fd_recon_dc_n_cg":   5,        # inner CG iterations
    "fd_recon_dc_relax":  0.5,      # blend of the CG-projected image (0..1)
}


# ---------------------------------------------------------------------------
def cg_data_consistency(x0_img, sino, proj, n_cg, relax):
    """REAL conjugate-gradient solve of ``min_x ||A x - y||^2`` warm-started
    at ``x0_img`` (image domain, μ units). ``relax`` blends the CG solution
    with the warm start: 1 = full project, 0 = no change.

    Normal-equation CG on ``A^T A x = A^T y`` (A^T = back_project). Maintains a
    residual ``r`` (in image domain = A^T(y - A x)), a conjugate direction
    ``p``, and the Fletcher-Reeves ``beta`` — this is genuine CG, NOT the
    steepest-descent the audit flagged in solver_diffusion_recon.dc_step_cg.
    Runs under no_grad (a hard projection, not part of the DPS graph).
    """
    with torch.no_grad():
        x = x0_img.clone()
        # r0 = A^T (y - A x)
        r = proj.back_project(sino - proj.forward_project(x))
        p = r.clone()
        rs_old = (r * r).sum(dim=(1, 2, 3), keepdim=True)
        for _ in range(n_cg):
            Ap = proj.forward_project(p)                 # A p           (B,1,A,D)
            AtAp = proj.back_project(Ap)                 # A^T A p       (B,1,H,W)
            denom = (p * AtAp).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * AtAp
            rs_new = (r * r).sum(dim=(1, 2, 3), keepdim=True)
            # Stop early on numerically-converged residual to avoid NaNs.
            if float(rs_new.sqrt().max()) < 1e-12:
                break
            beta = rs_new / rs_old.clamp(min=1e-12)
            p = r + beta * p
            rs_old = rs_new
        return relax * x + (1.0 - relax) * x0_img


def sample_recon(model, proj, sino, fbp_init, *, domain, n_steps, init_kind,
                 init_t, eta, out_scale, device,
                 dc=0, dc_n_cg=5, dc_relax=0.5):
    """Few-step DC-guided flow reconstruction for ONE scene.

    Integrate the flow ODE backwards from ``init_t`` to 0 with Euler steps,
    adding DPS guidance (and optionally a hard CG DC re-embed) at each step.

    Conventions:
      * ``x_t`` lives in the MODEL domain (pixel = image-norm, wavelet = DWT).
      * clean estimate at time t:  x0_hat_dom = x_t - t * v   (rectified-flow
        relation x_t = (1-t) x0 + t x1 with v = x1 - x0  =>  x0 = x_t - t v).
      * mapped to the image:  pixel -> identity, wavelet -> haar_idwt.
      * Euler update toward t=0:  x_t <- x_t - dt*v - dps_step.

    Returns a denormalised, clamp_min(0) image (1,1,H,W) in μ units.
    """
    eps = 1e-6
    fbp_norm = (fbp_init / out_scale).clamp(0.0, 1.0)         # (1,1,H,W)
    fbp_dom = to_model_domain(fbp_norm, domain)              # model domain

    if init_kind == "fbp":
        t = float(init_t)
        noise = torch.randn_like(fbp_dom)
        x_t = (1.0 - t) * fbp_dom + t * noise               # embed FBP at init_t
    else:
        t = 1.0
        x_t = torch.randn_like(fbp_dom)

    dt = t / n_steps                                         # uniform Euler step

    for k in range(n_steps):
        t_cur = t - k * dt
        t_next = t - (k + 1) * dt
        t_b = torch.full((x_t.shape[0],), t_cur, device=device)

        x_req = x_t.detach().requires_grad_(True)
        with torch.enable_grad():
            v = model(x_req, t_b * 1000.0)
            x0_hat_dom = x_req - t_cur * v                   # clean estimate (dom)
            x0_hat_img = from_model_domain(x0_hat_dom, domain)
            x0_img = x0_hat_img.clamp(0.0, 1.0) * out_scale  # μ units
            r = proj.forward_project(x0_img) - sino          # residual
            sse = (r ** 2).sum()
            resid_norm = sse.detach().sqrt().clamp(min=1e-8)
            grad = torch.autograd.grad(sse, x_req)[0]
        # DPS adaptive scale (Chung 2023, eq. 12): step = eta/||r|| * grad.
        dps_step = (eta / resid_norm) * grad.detach()
        v_det = v.detach()

        with torch.no_grad():
            if dc and t_next > eps:
                # ----- hard CG data-consistency + deterministic re-embed -----
                # Project the clean image estimate onto the data manifold.
                x0_proj_img = cg_data_consistency(
                    x0_img.detach(), sino, proj, n_cg=dc_n_cg, relax=dc_relax)
                x0_proj_dom = to_model_domain(
                    (x0_proj_img / out_scale).clamp(0.0, 1.0), domain)
                # Recover THIS step's noise direction from the current state
                # WITHOUT a fresh randn (x_t = (1-t)x0 + t*noise  =>
                # noise = (x_t - (1-t) x0_hat_dom) / t). Re-embed the PROJECTED
                # x0 at t_next reusing that same noise direction.
                noise_dir = (x_t - (1.0 - t_cur) * x0_hat_dom.detach()) / max(t_cur, eps)
                x_t = (1.0 - t_next) * x0_proj_dom + t_next * noise_dir
                # Still apply the (soft) DPS nudge so guidance + hard DC compose.
                x_t = x_t - dps_step
            else:
                # ----- plain DPS-guided Euler update -----
                x_t = x_t - dt * v_det - dps_step

    # Final clean estimate at t -> 0 is x_t itself (t=0 => x0 = x_t).
    img = from_model_domain(x_t.detach(), domain)
    return (img * out_scale).clamp_min(0.0)


# ---------------------------------------------------------------------------
def build_dataset(geom, n, seed, i0, sigma_e, device):
    """Dispatches on AGENT4CT_DATASET / cfg. Phantom path is backwards-
    compatible; staged paths load truth + real sinos from disk."""
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("FAST_RECON_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    env_ckpt = os.environ.get("FAST_RECON_CKPT")
    if env_ckpt:
        cfg["fd_recon_ckpt"] = env_ckpt

    # Dataset dispatch. When dataset_kind != "phantoms" override geometry.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] device={device}  init={cfg['fd_recon_init']}  "
          f"steps={cfg['fd_recon_steps']}  dc={cfg['fd_recon_dc']}", flush=True)
    print(f"[solver] cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('fd_recon_')}, default=str)}",
          flush=True)

    # ---- load checkpoint (auto-detect domain) --------------------------------
    ckpt_path = Path(cfg["fd_recon_ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FastDiff ckpt missing: {ckpt_path}. "
                                "Run solver_fast_diffusion.py first.")
    state = torch.load(ckpt_path, map_location=device)
    domain = state.get("fd_domain", "pixel")
    fd_ch = state["fd_ch"]
    out_scale = state.get("fd_out_scale", cfg["display_max"])
    fd_mode = state.get("fd_mode", "unknown")
    in_ch = out_ch = (4 if domain == "wavelet" else 1)
    print(f"[solver] loaded FastDiff ckpt: domain={domain}  mode={fd_mode}  "
          f"ch={fd_ch}  out_scale={out_scale}  "
          f"val_flow_loss={state.get('final_val_loss')}", flush=True)
    model = FlowUNet(in_ch=in_ch, out_ch=out_ch, ch=fd_ch).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Mayo: the val split is a single patient (L277) -> reconstruct at its
    # native pixel-spacing. The default ps (0.700857) mis-scales L277's native
    # ps (~0.74) by ~5%, which tanks BOTH the FBP baseline (PSNR ~20 vs ~36)
    # AND the DPS/DC recon physics (this `proj` is used inside sample_recon).
    # Single-ps probe (val is one patient) mirrors solver_ram.py lines ~288-311.
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

    # ---- build val set -------------------------------------------------------
    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
    proj = PyronnFanBeamProjector(geom).to(device)
    val_ph, _, val_noisy = build_dataset(geom, cfg["val_n"],
                                         cfg["seed"] + 1000,
                                         cfg["noise_i0"], cfg["noise_sigma_e"],
                                         device)
    with torch.no_grad():
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # ---- reconstruct each val scene ------------------------------------------
    t0 = time.time()
    preds = []
    for i in range(cfg["val_n"]):
        pred_i = sample_recon(model, proj,
                              val_noisy[i:i + 1], val_fbp[i:i + 1],
                              domain=domain,
                              n_steps=cfg["fd_recon_steps"],
                              init_kind=cfg["fd_recon_init"],
                              init_t=cfg["fd_recon_init_t"],
                              eta=cfg["fd_recon_eta"],
                              out_scale=out_scale, device=device,
                              dc=int(cfg.get("fd_recon_dc", 0)),
                              dc_n_cg=cfg.get("fd_recon_dc_n_cg", 5),
                              dc_relax=cfg.get("fd_recon_dc_relax", 0.5))
        preds.append(pred_i)
        if (i + 1) % 5 == 0:
            print(f"[sample] {i+1}/{cfg['val_n']}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > 1800:
            print(f"[sample] 30-min wall at {i+1}", flush=True); break
    sample_time = time.time() - t0
    pred = torch.cat(preds, 0)
    val_ph = val_ph[:pred.shape[0]]; val_fbp = val_fbp[:pred.shape[0]]

    # CONVENTIONS rule: clamp_min(0) before scoring so negative outliers don't
    # bias the linear calibration's background mean.
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
    params_total = sum(p.numel() for p in model.parameters())

    result = {
        "val_score": val_ssim, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6,
        "train_n": state.get("n_train", 0), "val_n": int(pred.shape[0]),
        "train_time_s": sample_time, "config": cfg,
        "fd_domain": domain,
        "fd_mode": fd_mode,
        "fd_train_seed": state.get("train_seed"),
        "fd_final_val_flow_loss": state.get("final_val_loss"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] FastRecon[{domain}/{fd_mode}]: hr={headroom:.4f}  "
          f"SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}  RMSE={val_rmse:.5f}  "
          f"baseline_PSNR={baseline_psnr:.2f}  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="FastDiff", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
