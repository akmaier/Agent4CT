"""Reference: Dual-Domain Denoising with learned U-Net denoisers,
trained with **supervised L2** on **all 128 projections**
(no Noise2Inverse split).

Mirror of `solver_dual_ddomain_bilateral_supervised.py` (2026-05-22) but
using ``SmallUNet`` denoisers instead of bilateral filters. The N2I
self-supervised variant (`solver_dual_ddomain_n2i.py`) was stuck at
hr=0 on breast-CT for c=4 / c=8 / c=16 — capacity wasn't the lever, the
loss was. This solver removes the N2I noise floor by training MSE-vs-
clean-phantom on the full 128-view forward pass.

Pipeline:
    pred = img_dn( R_full.fbp( proj_dn( sino_full ) ) )
    loss = mse(pred, clean_phantom) + lambda_neg * negativity_penalty(pred)

Trade-off: needs the clean phantom at train time, so this variant is
only fair against other supervised baselines (it cannot play the
"self-supervised" tag).
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
from ddssl_ldct.metrics import (
    evaluate_calibrated, make_4panel_comparison,
    supervised_recon_loss,
)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "unet_c":        16,
    # Training
    "epochs":        10,
    "batch_size":    1,
    "lr":            5e-4,
    "optimizer":     "adam",
    "lambda_neg":    1.0,
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)   # 4-tuple; ps=None for non-mayo


class FullViewUNetPipeline(nn.Module):
    """Single-pass dual-domain U-Net: proj_dn -> FBP(128 views) -> img_dn."""

    def __init__(self, geometry: FanBeamGeometry,
                 proj_dn: nn.Module, img_dn: nn.Module):
        super().__init__()
        self.geometry = geometry
        self.proj_dn = proj_dn
        self.img_dn = img_dn
        self.R_full = PyronnFanBeamProjector(geometry)

    def forward(self, sino_full: torch.Tensor) -> torch.Tensor:
        s = self.proj_dn(sino_full)
        r = self.R_full.fbp(s)
        return self.img_dn(r)


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

    train_ph, train_clean, train_noisy, train_ps = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # PER-SAMPLE geometry (Mayo canonical): each slice reconstructs at its own
    # ps_eff (only ps varies; angle uniform). Build a per-ps projector cache and
    # swap pipe.R_full per sample (batch_size=1). ps=None -> single fixed geom.
    import numpy as _np
    _per_ps = train_ps is not None
    if _per_ps:
        from ddssl_ldct.staged_dataset import mayo_proj_cache
        _projs = mayo_proj_cache(_np.concatenate([train_ps, val_ps]),
                                 cfg["n_angles"], cfg["n_det"], device)
        _trk = _np.round(_np.asarray(train_ps, float), 5)
        _vrk = _np.round(_np.asarray(val_ps, float), 5)

    def _proj_for(idx_arr, key_arr):
        return _projs[float(key_arr[int(idx_arr[0])])] if _per_ps else None

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
    proj_dn = SmallUNet(c=c)
    img_dn = SmallUNet(c=c)
    pipe = FullViewUNetPipeline(geom, proj_dn, img_dn).to(device)

    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] Supervised-L2 DD-UNet: c={c} params={params_total/1e6:.3f} M "
          f"(proj={sum(p.numel() for p in proj_dn.parameters())/1e6:.3f} M, "
          f"img={sum(p.numel() for p in img_dn.parameters())/1e6:.3f} M)", flush=True)

    opt = torch.optim.Adam(pipe.parameters(), lr=cfg["lr"])

    # Opt-in "train once, save checkpoint, reuse" hook. Gated ENTIRELY on the
    # AGENT4CT_MODEL_CKPT env var, which is set ONLY by the testset sweep worker
    # (scripts/score_mayo_testset.py). Unset in normal runs -> no-op, byte-
    # identical behaviour. When set: the FIRST patient's process trains + saves
    # the ckpt; the other four LOAD it and skip the training loop. Training is
    # deterministic + patient-independent (train set is always the 4 fixed train
    # patients; only the EVAL set changes per patient), so 5 trainings/iter -> 1.
    _CKPT = os.environ.get("AGENT4CT_MODEL_CKPT")
    _CKPT_EXISTS = bool(_CKPT) and Path(_CKPT).exists()

    t0 = time.time()
    if _CKPT_EXISTS:
        # Skip-branch: load the trained weights, do NOT run the training loop.
        print(f"[solver] loading checkpoint {_CKPT} (skip training)", flush=True)
        pipe.load_state_dict(torch.load(_CKPT, map_location=device))
    for ep in ([] if _CKPT_EXISTS else range(cfg["epochs"])):
        pipe.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        n_seen = 0
        n_skip = 0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if _per_ps:   # batch_size=1 for Mayo -> one ps per step
                pipe.R_full = _projs[float(_trk[int(idx[0])])]
            sino = train_noisy[idx].to(device)
            truth = train_ph[idx].to(device)
            pred = pipe(sino)
            loss = supervised_recon_loss(pred, truth,
                                          lambda_neg=cfg["lambda_neg"], base="mse")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Mayo's 2304-view FBP adjoint amplifies gradients ~18x vs demo_dl's
            # 128 views (ramp |freq| weighting summed over many views), so the
            # backward occasionally overflows to a nonfinite gradient and the run
            # NaNs within 1-2 epochs (standalone: lr=5e-4 NaNs ep1, lr=1e-4 ep2).
            # Clip the grad NORM and skip the step on ANY nonfinite loss OR
            # gradient -- a FINITE loss can still carry an Inf grad whose clip
            # yields NaN, so guarding the loss alone is not enough. grad_clip=0 ->
            # off (demo_dl/breast unchanged); Mayo injects grad_clip=1.0 via clamp.
            gc = float(cfg.get("grad_clip", 0.0))
            gnorm = torch.nn.utils.clip_grad_norm_(
                pipe.parameters(), gc if gc > 0 else float("inf"))
            if torch.isfinite(loss) and torch.isfinite(gnorm):
                opt.step()
                running += float(loss.detach().cpu()) * idx.numel()
                n_seen += idx.numel()
            else:
                opt.zero_grad(set_to_none=True)
                n_skip += 1
        mean_loss = running / max(1, n_seen)
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.6f}"
              f"  (skipped {n_skip}/{n_seen + n_skip} nonfinite-grad batches)",
              flush=True)

    if _CKPT and not _CKPT_EXISTS:
        # First patient: persist the trained model for the other four to reuse.
        Path(_CKPT).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pipe.state_dict(), _CKPT)
        print(f"[solver] saved checkpoint {_CKPT}", flush=True)

    train_time = time.time() - t0

    pipe.eval()
    with torch.no_grad():
        chunk = cfg.get("val_chunk", 10)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            if _per_ps:   # val_chunk=1 for Mayo -> one ps per slice
                pipe.R_full = _projs[float(_vrk[i])]
            preds.append(pipe(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)

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
        "params_M": params_total / 1e6, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time,
        "config": cfg,
        "training_scheme": "supervised_l2_full_views_128",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Supervised-L2 DD-UNet: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f}  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="DD-UNet-L2", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
