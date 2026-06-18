"""Reference: Dual-Domain Denoising with learned denoisers (Wagner et al. 2023)
trained with **Noise2Inverse self-supervision** on 2×half-view sub-sets.

This is the original Wagner self-supervised setup: two learned U-Net
denoisers (one in projection domain, one in image domain) optimised
end-to-end with the split-view MSE described in
`ddssl_ldct/training.py::DualDomainPipeline.training_step`. No clean
target is used at train time.

**Per-image protocol (2026-06-18 fix).** N2I is inherently a *per-image*
self-supervised method (Hendriksen et al. 2020, Wagner et al. 2023): each
scan is reconstructed by optimising on ITS OWN low-dose measurements
(angularly split into two half-view subsets), with no clean ground truth
and no cross-scan amortisation. The earlier implementation here WRONGLY
trained one amortised model on the 4 train patients and forward-applied it
to val/test — that is not N2I. The recon is now:

  1. **Warm-start pre-pass (cfg["warm_start"], default True):** a SHORT
     amortised N2I pre-train on the TRAIN split (bounded by
     ``pretrain_epochs`` / ``pretrain_steps``) using the SAME GT-free
     N2I self-supervised loss. Captures ``warm_state = pipe.state_dict()``
     as a generic initialisation. Skipped if ``warm_start`` is False
     (then the train split is never loaded).
  2. **Per-image fine-tune (the actual recon):** for EACH val/showcase
     scan, build a FRESH ``DualDomainPipeline``, load ``warm_state`` if
     present, and run ``cfg["n_iter"]`` Adam steps of the N2I
     self-supervised loss computed on THAT ONE scan's sinogram only
     (grad-clipped; nonfinite steps skipped), guarded by a per-scene
     wall-time limit. Then ``predict`` under no_grad.

For the **supervised** (full-view, MSE-vs-clean) variant see
`solver_dual_ddomain_supervised.py` — built 2026-05-22 after finding
that N2I systematically over-smooths on dense breast-CT scans (see
`docs/findings.md`).

Architecture: SmallUNet (c=16) in both domains.

Mayo per-sample-ps (2026-06-16 onboarding): the canonical Mayo val split
is a single patient (L277, native ps≈0.74) and the 4 train patients span 4
recon pixel-spacings. `DualDomainPipeline.training_step`/`predict` only ever
use ``self.R_half`` (the half-angle projector for the N2I view-split) — so we
swap a per-ps HALF-ANGLE projector into ``pipe.R_half`` per sample for both
the warm-start pre-pass and each per-image fine-tune (batch_size=1 train /
one scan at a time val). The full-view LD-FBP baseline (the headroom
denominator) still comes from the full-angle per-ps cache, matching every
other Mayo solver. ps=None (non-Mayo) -> single fixed geometry.
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
import torch.nn as nn

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.models import SmallUNet
from ddssl_ldct.training import DualDomainPipeline
from ddssl_ldct.metrics import evaluate_calibrated, make_4panel_comparison
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # --- Warm-start pre-pass (short amortised N2I on the train split) ---
    "warm_start":      True,   # if False, the train split is never loaded
    "pretrain_epochs": 3,      # bound for the warm-start pre-train
    "pretrain_steps":  0,      # alt cap: max total opt.steps (0 = use epochs)
    # --- Per-image fine-tune (the actual recon) ---
    "n_iter":          400,    # per-scan Adam steps of the N2I self-sup loss
    "grad_clip":       1.0,    # mandatory >0 for Mayo's 2304-view FBP adjoint
    "outer_wall_s":    3600,   # whole val-loop wall; per_scene = outer / val_n
    "per_scene_s":     None,   # optional per-scene override (s)
    # --- Optimiser / capacity (shared by pre-pass + per-image fit) ---
    "epochs":        8,        # legacy; superseded by pretrain_epochs in warm-start
    "batch_size":    1,
    "lr":            1e-3,
    "optimizer":     "adam",
    "unet_c":        16,
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)   # 4-tuple; ps=None for non-mayo


def _fit_one_scene(scan_sino, geom, cfg, device, warm_state, c,
                   r_half=None, t_limit=None):
    """Per-image N2I reconstruction of ONE scan.

    Builds a FRESH DualDomainPipeline (fresh SmallUNet denoisers), loads
    ``warm_state`` if present, then runs ``cfg["n_iter"]`` Adam steps of the
    GT-free N2I self-supervised loss (``pipe.training_step(sino)["loss"]``)
    on THIS scan's sinogram only — grad-clipped, nonfinite steps skipped —
    bounded by ``t_limit`` seconds. Returns ``(pred (1,1,H,W), last_loss,
    n_done)``.

    ``r_half``: per-slice HALF-ANGLE projector (Mayo per-ps). When None the
    pipeline keeps the default half-projector built from ``geom``.
    """
    proj_dn = SmallUNet(c=c)
    img_dn = SmallUNet(c=c)
    pipe = DualDomainPipeline(geometry=geom, proj_denoiser=proj_dn,
                              image_denoiser=img_dn).to(device)
    if warm_state is not None:
        pipe.load_state_dict(warm_state)
    if r_half is not None:   # Mayo: per-ps half-angle projector for the view-split
        pipe.R_half = r_half
    opt = torch.optim.Adam(pipe.parameters(), lr=cfg["lr"])
    gc = float(cfg.get("grad_clip", 0.0))
    pipe.train()
    t0 = time.time()
    last_loss = float("inf")
    n_done = 0
    for it in range(int(cfg["n_iter"])):
        losses = pipe.training_step(scan_sino)   # self-supervised; no clean target
        loss = losses["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Mayo's many-view FBP adjoint amplifies gradients; clip the grad NORM
        # and skip the step on ANY nonfinite loss OR gradient. grad_clip<=0 ->
        # off (demo_dl/breast unchanged) but the norm is still computed so a
        # nonfinite-grad step is still skipped.
        gnorm = torch.nn.utils.clip_grad_norm_(
            pipe.parameters(), gc if gc > 0 else float("inf"))
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            opt.step()
            last_loss = float(loss.detach().cpu())
        else:
            opt.zero_grad(set_to_none=True)
        n_done = it + 1
        if t_limit is not None and (time.time() - t0) > t_limit:
            break
    pipe.eval()
    with torch.no_grad():
        pred = pipe.predict(scan_sino)
    return pred.detach(), last_loss, n_done


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

    import numpy as _np
    warm_start = bool(cfg.get("warm_start", True))
    # The TRAIN split is ONLY consumed by the warm-start pre-pass; with
    # warm_start=False we skip loading it entirely (pure per-image recon).
    if warm_start:
        train_ph, train_clean, train_noisy, train_ps = build_dataset(
            geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    else:
        train_ph = train_clean = train_noisy = train_ps = None
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # PER-SAMPLE geometry (Mayo canonical). N2I training_step/predict only use
    # the HALF-ANGLE projector, so build a per-ps half-angle cache from the
    # full-angle cache (g.split_angles()[0]) and swap pipe.R_half per sample.
    # The full cache feeds the full-view LD-FBP baseline (headroom denominator).
    _per_ps = val_ps is not None
    _projs_half = None
    if _per_ps:
        from ddssl_ldct.staged_dataset import mayo_proj_cache
        _all_ps = _np.concatenate([train_ps, val_ps]) if train_ps is not None else val_ps
        _projs = mayo_proj_cache(_all_ps, cfg["n_angles"], cfg["n_det"], device)
        _projs_half = {k: PyronnFanBeamProjector(v.geom.split_angles()[0]).to(device)
                       for k, v in _projs.items()}
        _trk = _np.round(_np.asarray(train_ps, float), 5) if train_ps is not None else None
        _vrk = _np.round(_np.asarray(val_ps, float), 5)

    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        if _per_ps:
            ld_fbp = torch.empty(val_noisy.shape[0], 1, 512, 512, device=device)
            for u in _np.unique(_vrk):
                ii = _np.where(_vrk == u)[0]
                ld_fbp[ii] = _projs[float(u)].fbp(val_noisy[ii]).clamp(min=0.0)
        else:
            ld_fbp = torch.clamp(R_full.fbp(val_noisy), min=0.0)

    c = cfg["unet_c"]
    _p0 = sum(p.numel() for p in SmallUNet(c=c).parameters())
    params_total = 2 * _p0   # proj + img denoiser
    print(f"[solver] N2I DD-UNet (per-image): c={c} params={params_total/1e6:.3f} M "
          f"(proj={_p0/1e6:.3f} M, img={_p0/1e6:.3f} M)  warm_start={warm_start}",
          flush=True)

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1) Warm-start pre-pass: SHORT amortised N2I pre-train on the train
    #    split (GT-free self-supervised loss). Capture warm_state as a
    #    generic init for the per-image fits. Bounded by pretrain_epochs or
    #    (if >0) a hard pretrain_steps cap.
    # ------------------------------------------------------------------ #
    warm_state = None
    if warm_start:
        proj_dn = SmallUNet(c=c)
        img_dn = SmallUNet(c=c)
        pipe = DualDomainPipeline(geometry=geom, proj_denoiser=proj_dn,
                                  image_denoiser=img_dn).to(device)
        opt = torch.optim.Adam(pipe.parameters(), lr=cfg["lr"])
        gc = float(cfg.get("grad_clip", 0.0))
        max_steps = int(cfg.get("pretrain_steps", 0) or 0)
        n_steps = 0
        for ep in range(int(cfg.get("pretrain_epochs", 3))):
            pipe.train()
            perm = torch.randperm(train_noisy.shape[0])
            running = 0.0
            n_seen = 0
            n_skip = 0
            for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                if _per_ps:   # batch_size=1 for Mayo -> one ps per step
                    pipe.R_half = _projs_half[float(_trk[int(idx[0])])]
                sino = train_noisy[idx].to(device)
                losses = pipe.training_step(sino)   # self-supervised; no clean target
                loss = losses["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Mayo's many-view FBP adjoint amplifies gradients vs demo_dl's
                # 128 views, so the backward can overflow to a nonfinite
                # gradient. Clip the grad NORM and skip the step on ANY
                # nonfinite loss OR gradient. grad_clip<=0 -> off
                # (demo_dl/breast unchanged); Mayo injects 1.0.
                gnorm = torch.nn.utils.clip_grad_norm_(
                    pipe.parameters(), gc if gc > 0 else float("inf"))
                if torch.isfinite(loss) and torch.isfinite(gnorm):
                    opt.step()
                    running += float(loss.detach().cpu()) * idx.numel()
                    n_seen += idx.numel()
                else:
                    opt.zero_grad(set_to_none=True)
                    n_skip += 1
                n_steps += 1
                if max_steps and n_steps >= max_steps:
                    break
            mean_loss = running / max(1, n_seen)
            print(f"[solver] warm-start epoch {ep+1:3d}/{cfg.get('pretrain_epochs', 3)}"
                  f"  loss={mean_loss:.6f}  (skipped {n_skip}/{n_seen + n_skip} "
                  f"nonfinite-grad batches)", flush=True)
            if max_steps and n_steps >= max_steps:
                print(f"[solver] warm-start pretrain_steps cap {max_steps} hit", flush=True)
                break
        warm_state = {k: v.detach().clone() for k, v in pipe.state_dict().items()}
    pretrain_time = time.time() - t0

    # ------------------------------------------------------------------ #
    # 2) Per-image fine-tune: for EACH val/showcase scan, fit a FRESH
    #    pipeline (warm_state init) on THAT scan's sinogram only, then
    #    predict. This is the actual N2I reconstruction. AGENT4CT_SHOWCASE=
    #    valtest is handled transparently by build_dataset -> the showcase
    #    scans already arrive in val_noisy / val_ps.
    # ------------------------------------------------------------------ #
    val_n = val_noisy.shape[0]
    outer_wall = float(cfg.get("outer_wall_s", 3600))
    per_scene_s = cfg.get("per_scene_s") or (outer_wall / max(1, val_n))
    t1 = time.time()
    preds = []
    for i in range(val_n):
        r_half = _projs_half[float(_vrk[i])] if _per_ps else None
        pred_i, last_loss, n_done = _fit_one_scene(
            val_noisy[i:i + 1], geom, cfg, device, warm_state, c,
            r_half=r_half, t_limit=per_scene_s)
        preds.append(pred_i)
        if (i + 1) % 2 == 0 or i == val_n - 1:
            print(f"[fit] {i+1}/{val_n}  inner_iters={n_done}  "
                  f"last_loss={last_loss:.4g}  elapsed={time.time()-t1:.1f}s",
                  flush=True)
        if time.time() - t1 > outer_wall:
            print(f"[fit] outer wall {outer_wall}s hit at sample {i+1}", flush=True)
            break
    pred = torch.cat(preds, dim=0)
    train_time = time.time() - t0   # warm-start + per-image total
    # Trim labels / baseline to the scans we actually reconstructed (in case
    # the outer wall cut the loop short), mirroring solver_naf.py.
    val_ph = val_ph[:pred.shape[0]]
    ld_fbp = ld_fbp[:pred.shape[0]]

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
        "params_M": params_total / 1e6,
        "train_n": cfg["train_n"] if warm_start else 0,
        "val_n": int(pred.shape[0]), "train_time_s": train_time,
        "pretrain_time_s": pretrain_time,
        "config": cfg,
        "training_scheme": "noise2inverse_per_image_warmstart_selfsup",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] N2I DD-UNet: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f}  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="DD-N2I", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
