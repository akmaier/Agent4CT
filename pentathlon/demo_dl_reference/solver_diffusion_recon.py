"""Reference: Diffusion-prior CT reconstruction from a *pre-trained* DDPM.

This is the inference-time twin of ``solver_ddpm.py``. It loads a frozen
DDPM checkpoint produced by that training solver, then runs DDIM
sampling with data-consistency guidance against the measured sparse-view
sinogram. Two guidance modes (Chung et al. 2023 / 2022):

  * ``"dps"`` — Diffusion Posterior Sampling: gradient steering
    ``x_t ← x_t − η · ∇_x ‖A x̂₀(x_t) − y‖²``
  * ``"mcg"`` — Manifold-Constrained Gradient via pseudo-inverse:
    ``g_t := ∇_x ‖A†(A x̂₀ − y)‖²``  with ``A†`` ≈ FBP

The DDPM checkpoint is selected by the path in ``recon_ckpt`` (env-overridable),
so a single solver file serves two recon experiments:
  - ``demo-dl-diffusion-recon-unconstrained-*``  uses ddpm_unconstrained_final.pt
  - ``demo-dl-diffusion-recon-constrained-*``    uses ddpm_constrained_final.pt

The autoresearch agent varies sampling hyperparameters
(``recon_mode, recon_sample_steps, recon_eta, recon_init``) only.
The DDPM weights stay frozen across all search iters.

Citations:
  Chung H. et al. "Diffusion Posterior Sampling for General Noisy Inverse
    Problems." ICLR 2023. arXiv:2209.14687
  Chung H. et al. "Improving Diffusion Models for Inverse Problems
    Using Manifold Constraints." NeurIPS 2022. arXiv:2206.00941
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS
# Reuse the model + schedule definitions from the training file so a single
# class definition is the source of truth across both solvers.
from pentathlon.demo_dl_reference.solver_ddpm import (
    SmallDDPM, NoiseSchedule, build_phantoms,
)


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "val_n": 20,
    # Path to a checkpoint produced by solver_ddpm.py. Override via env.
    "recon_ckpt":          "/cluster/maier/Agent4CT/checkpoints/ddpm_unconstrained_final.pt",
    # Sampling hyperparameters (what the search varies).
    "recon_mode":          "dps",
    "recon_sample_steps":  50,
    "recon_eta":           1.0,
    "recon_init":          "fbp",
    "recon_eta_clamp":     False,    # if True, clamp x_t after each step into [0, 1]
    # ---- DC-step (Resample-style hard projection) --------------------------
    "recon_dcstep_every":  0,        # 0 = off; otherwise apply DC-step every K reverse steps
    "recon_dcstep_n_cg":   5,        # CG inner iterations per DC-step
    "recon_dcstep_warmup": 5,        # skip the first N reverse steps (too noisy for projection)
    "recon_dcstep_relax":  0.5,      # 0=no relax, 1=full hard projection
    # ---- DC-step SOLVER select (audit fix, 2026-06-20) ---------------------
    # "legacy" = the ORIGINAL (buggy) path: dc_step_cg() is mislabeled
    #            steepest-descent, and the re-injection forms x0 from a STALE
    #            eps then re-noises with a FRESH randn. DEFAULT — preserved
    #            byte-for-byte for reproducibility of all existing cfgs.
    # "cg"     = the CORRECTED path ported from solver_fast_recon.py: a genuine
    #            Fletcher-Reeves CG solve on the normal equations AᵀA x = Aᵀy,
    #            plus a DETERMINISTIC re-embed that reuses the CURRENT step's
    #            eps_pred (no stale eps, no fresh randn).
    "recon_dc_solver":     "legacy",
}


# ---------------------------------------------------------------------------
def x0_from_eps(xt, eps, ab):
    return (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-6)


def ddim_step(xt, eps_pred, sched, t_now, t_next):
    ab_now = sched.alpha_bar[t_now]
    ab_next = sched.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0,
                                                                       device=xt.device)
    x0_hat = x0_from_eps(xt, eps_pred, ab_now)
    return ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps_pred


def dc_step_cg(x0_mu, sino, proj, n_cg, relax):
    """Resample-style hard projection toward data-fidelity: a few steps of
    Gauss-Newton (~CG) on `min_x ‖A x − sino‖²` starting from x0_mu. Returns
    the projected mu-image. `relax` blends with the input: 1 = full project,
    0 = no change. Operates in the noise-free image domain (mu units).
    """
    x = x0_mu.clone()
    for _ in range(n_cg):
        with torch.no_grad():
            r = proj.forward_project(x) - sino                  # (B, 1, A, D)
            g = proj.back_project(r)                              # (B, 1, H, W)
            # CG step size via Polak-Ribiere-ish: use g^T A^T A g  /  ‖A g‖²
            Ag = proj.forward_project(g)
            num = (g * g).sum(dim=(1, 2, 3), keepdim=True)
            den = (Ag * Ag).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
            alpha = num / den
            x = x - alpha * g
    return relax * x + (1.0 - relax) * x0_mu


def cg_data_consistency(x0_img, sino, proj, n_cg, relax):
    """CORRECTED DC solver (audit fix, ported verbatim in spirit from
    ``solver_fast_recon.cg_data_consistency``). REAL conjugate-gradient solve
    of ``min_x ||A x - y||^2`` warm-started at ``x0_img`` (image domain, μ
    units). ``relax`` blends the CG solution with the warm start: 1 = full
    project, 0 = no change.

    Normal-equation CG on ``A^T A x = A^T y`` (A^T = back_project). Maintains a
    residual ``r`` (in image domain = A^T(y - A x)), a conjugate direction
    ``p``, and the Fletcher-Reeves ``beta`` — this is genuine CG, NOT the
    steepest-descent that ``dc_step_cg`` (the legacy path) mislabels as CG.
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
            # Stop early on numerically-converged residual to avoid NaNs. Guard
            # the empty-batch edge case: a patient whose slices chunk to a 0-size
            # tail hands CG a 0-element tensor, and .max() on numel()==0 raises
            # (crashed diff-recon on L056, the 93-slice patient). Nothing to
            # solve -> stop.
            if rs_new.numel() == 0 or float(rs_new.sqrt().max()) < 1e-12:
                break
            beta = rs_new / rs_old.clamp(min=1e-12)
            p = r + beta * p
            rs_old = rs_new
        return relax * x + (1.0 - relax) * x0_img


def sample_guided(model, sched, proj, sino, fbp_init, *, mode, n_steps,
                  eta, init_kind, out_scale, device, eta_clamp,
                  dcstep_every=0, dcstep_n_cg=5, dcstep_warmup=5,
                  dcstep_relax=0.5, dc_solver="legacy"):
    T = sched.T
    times = torch.linspace(T, 1, n_steps + 1).long().tolist()
    fbp_norm = (fbp_init / out_scale).clamp(0.0, 1.0)
    if init_kind == "fbp":
        ab_T = sched.alpha_bar[T]
        x = ab_T.sqrt() * fbp_norm + (1 - ab_T).sqrt() * torch.randn_like(fbp_norm)
    else:
        x = torch.randn_like(fbp_norm)

    for k in range(n_steps):
        t_now = times[k]; t_next = times[k + 1] if k + 1 < len(times) else 0
        t_tensor = torch.tensor([t_now], device=device)
        x_req = x.detach().requires_grad_(True)
        with torch.enable_grad():
            eps = model(x_req, t_tensor)
            ab_now = sched.alpha_bar[t_now]
            x0_hat = x0_from_eps(x_req, eps, ab_now)
            # Clamp x0_hat to [0,1] BEFORE projecting — keeps the projector
            # operating on valid mu values. Crucially we do NOT clamp x_t.
            x0_mu = x0_hat.clamp(0.0, 1.0) * out_scale
            sino_pred = proj.forward_project(x0_mu)
            if mode == "dps":
                # Use sum-of-squares (NOT mean) and DPS's canonical adaptive
                # weight ζ_t = eta / ‖residual‖_2 (Chung 2023, eq. 12). Without
                # this scaling, no fixed eta works across noise levels.
                sse = ((sino_pred - sino) ** 2).sum()
                resid_norm = sse.detach().sqrt().clamp(min=1e-8)
                grad = torch.autograd.grad(sse, x_req)[0]
                step = (eta / resid_norm) * grad.detach()
            elif mode == "mcg":
                # MCG: pseudo-inverse via FBP, same adaptive scaling.
                resid = sino_pred - sino
                back = proj.fbp(resid)
                sse = (back ** 2).sum()
                resid_norm = sse.detach().sqrt().clamp(min=1e-8)
                grad = torch.autograd.grad(sse, x_req)[0]
                step = (eta / resid_norm) * grad.detach()
            else:
                raise ValueError(mode)
        with torch.no_grad():
            x_clean = ddim_step(x.detach(), eps.detach(), sched, t_now, t_next)
            x = x_clean - step
            if eta_clamp:
                cap = 0.5  # max per-step pixel displacement in [0,1] space
                x = x_clean + (x - x_clean).clamp(-cap, cap)

            # ---- DC-step refinement (Resample-style) ---------------------
            # Periodically project x toward the data-fidelity manifold by
            # solving min ‖A x_mu − y‖² for a few CG steps, then re-noising
            # back to the current t-level. Off by default (dcstep_every=0);
            # set 5-10 to enable.
            if (dcstep_every > 0
                    and k >= dcstep_warmup
                    and (k - dcstep_warmup) % dcstep_every == 0
                    and t_next > 0):
                ab_next = sched.alpha_bar[t_next]
                if dc_solver == "cg":
                    # ----- CORRECTED DC-step (audit fix) ----------------------
                    # Form x0 from the CURRENT step's eps (x0_hat, computed at
                    # t_now above) — NOT a stale eps re-evaluated at the wrong
                    # alpha_bar. Project it onto the data manifold with a REAL
                    # CG solve, then re-embed at t_next DETERMINISTICALLY by
                    # reusing the SAME noise direction implied by the current
                    # eps (eps-prediction => the noise at level t IS eps_pred).
                    # No stale eps, NO fresh randn.
                    x0_cur_mu = x0_hat.detach().clamp(0.0, 1.0) * out_scale
                    x0_proj = cg_data_consistency(x0_cur_mu, sino, proj,
                                                  n_cg=dcstep_n_cg,
                                                  relax=dcstep_relax)
                    x0_proj_norm = (x0_proj / out_scale).clamp(0.0, 1.0)
                    x = (ab_next.sqrt() * x0_proj_norm
                         + (1 - ab_next).sqrt() * eps.detach())
                else:
                    # ----- LEGACY DC-step (preserved byte-for-byte) -----------
                    # Move to t_next noise level: form x̂₀ from current x
                    # Convert current x → predicted clean image in mu units
                    # (use the just-computed eps as the noise estimate)
                    x0_est = x0_from_eps(x, eps.detach(), ab_next).clamp(0.0, 1.0) * out_scale
                    # Hard project via CG against the sinogram
                    x0_proj = dc_step_cg(x0_est, sino, proj,
                                          n_cg=dcstep_n_cg, relax=dcstep_relax)
                    # Re-noise back to the t_next level so the DDIM trajectory
                    # can resume.
                    x0_proj_norm = (x0_proj / out_scale).clamp(0.0, 1.0)
                    noise = torch.randn_like(x0_proj_norm)
                    x = ab_next.sqrt() * x0_proj_norm + (1 - ab_next).sqrt() * noise
    # Final denorm + clamp; only here do we crop to display range.
    return (x.detach() * out_scale).clamp(0.0, out_scale)


# ---------------------------------------------------------------------------
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


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("DIFFUSION_RECON_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}
    # Allow env override of the checkpoint path independently of the JSON cfg.
    env_ckpt = os.environ.get("DIFFUSION_RECON_CKPT")
    if env_ckpt: cfg["recon_ckpt"] = env_ckpt

    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))
    # The DDPM prior must match the image distribution. The search-space
    # `recon_ckpt` is hardcoded to the demo-dl ellipse-phantom DDPM; for a
    # real staged dataset, swap to the dataset-specific checkpoint. Without
    # this the diffusion prior hallucinates ellipse structure into breast
    # images (breast-ct diff_recon scored SSIM 0.936 < FBP128's 0.955 — see
    # SLURM 761515/761516, fixed 2026-05-21).
    if cfg["dataset_kind"] == "breast_ct" and not os.environ.get("DIFFUSION_RECON_CKPT"):
        _ck = Path(cfg["recon_ckpt"])
        _name = _ck.name
        if _name.startswith("ddpm_") and "breast" not in _name:
            _breast = _ck.with_name("ddpm_breast_" + _name[len("ddpm_"):])
            if _breast.exists():
                print(f"[solver] breast_ct: swapping DDPM prior "
                      f"{_ck.name} -> {_breast.name}", flush=True)
                cfg["recon_ckpt"] = str(_breast)
            else:
                print(f"[solver] WARNING: {_breast} not found; "
                      f"keeping {_ck.name} (prior mismatch likely).", flush=True)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] device={device}  mode={cfg['recon_mode']}", flush=True)
    print(f"[solver] cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('recon_')}, default=str)}",
          flush=True)

    # Load DDPM checkpoint.
    ckpt_path = Path(cfg["recon_ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DDPM ckpt missing: {ckpt_path}. "
                                "Run solver_ddpm.py first.")
    state = torch.load(ckpt_path, map_location=device)
    ddpm_ch = state["ddpm_ch"]; T = state["ddpm_n_steps"]
    out_scale = state.get("ddpm_out_scale", cfg["display_max"])
    ddpm_mode = state.get("ddpm_mode", "unknown")
    print(f"[solver] loaded DDPM ckpt: mode={ddpm_mode}  ch={ddpm_ch}  T={T}  "
          f"out_scale={out_scale}  train_loss={state.get('final_train_loss')}  "
          f"val_eps={state.get('final_val_loss')}", flush=True)
    sched = NoiseSchedule(T=T, device=device)
    model = SmallDDPM(ch=ddpm_ch).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Mayo: the val split is a single patient (L277) -> reconstruct at its
    # native pixel-spacing. The default ps (0.700857) mis-scales L277's native
    # ps (~0.74) by ~5%, which tanks BOTH the FBP baseline (PSNR ~20 vs ~36)
    # AND the DPS/DC recon physics (this `proj` is used inside sample_guided).
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

    # Build val set.
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

    # Guided sample each val scene.
    t0 = time.time()
    preds = []
    for i in range(cfg["val_n"]):
        pred_i = sample_guided(model, sched, proj,
                                val_noisy[i:i + 1], val_fbp[i:i + 1],
                                mode=cfg["recon_mode"],
                                n_steps=cfg["recon_sample_steps"],
                                eta=cfg["recon_eta"],
                                init_kind=cfg["recon_init"],
                                out_scale=out_scale, device=device,
                                eta_clamp=cfg["recon_eta_clamp"],
                                dcstep_every=cfg.get("recon_dcstep_every", 0),
                                dcstep_n_cg=cfg.get("recon_dcstep_n_cg", 5),
                                dcstep_warmup=cfg.get("recon_dcstep_warmup", 5),
                                dcstep_relax=cfg.get("recon_dcstep_relax", 0.5),
                                dc_solver=cfg.get("recon_dc_solver", "legacy"))
        preds.append(pred_i)
        if (i + 1) % 5 == 0:
            print(f"[sample] {i+1}/{cfg['val_n']}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
        if time.time() - t0 > 1800:
            print(f"[sample] 30-min wall at {i+1}", flush=True); break
    sample_time = time.time() - t0
    pred = torch.cat(preds, 0)
    val_ph = val_ph[:pred.shape[0]]; val_fbp = val_fbp[:pred.shape[0]]

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
        "ddpm_mode": ddpm_mode,
        "ddpm_train_seed": state.get("train_seed"),
        "ddpm_final_val_eps_loss": state.get("final_val_loss"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] DiffRecon[{ddpm_mode}/{cfg['recon_mode']}]: "
          f"hr={headroom:.4f}  SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="Diffusion", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out_dir")
    args = p.parse_args(); main(Path(args.out_dir))
