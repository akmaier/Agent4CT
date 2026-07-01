"""Reference: Unrolled TV-Gradient-Descent with per-iter learnable step + lambda,
trained supervised-L2 vs the clean phantom on the full 128-view forward pass.

Mirror of `solver_dual_ddomain_supervised.py` (2026-05-22) but applied to a
classical TV minimisation rather than a U-Net dual-domain. The TV iteration

    f_{k+1} = clamp( f_k - step_k * (R^T (R f_k - g) + lambda_k * grad_TV(f_k)),
                     0, tv_clip_max )

is unrolled for K=10 iterations and each `step_k`, `lambda_k` is an
`nn.Parameter` (parametrised via log to stay positive). The unrolled
network is initialised from FBP(noisy) and trained end-to-end with MSE
against the clean phantom plus a non-negativity penalty.

Built 2026-05-23 against `docs/findings.md` 2026-05-22 entry:
"The supervised-L2 dual-domain DD-BF went from hr=0 (N2I) to hr=0.21 just
by switching the loss. The N2I TV-iterative (`solver_tv_search.py`) is
likely at the same loss-bottleneck on breast-CT. This is the same recipe
applied to TV iterations with two learnable scalars per iter."

Trainable parameters (defaults K=10): 20 scalars total. Tiny.

Trade-off: needs the clean phantom at train time, so this variant is
only fair against other supervised baselines.
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
import torch.nn.functional as F

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import (
    evaluate_calibrated, make_4panel_comparison, supervised_recon_loss,
)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # TV unrolled
    "tv_K":            10,          # unrolled iterations
    "tv_step_init":    1.0e-2,      # per-iter GD step (log-parametrised)
    "tv_lambda_init":  1.0e-3,      # per-iter TV weight (log-parametrised)
    "tv_share_steps":  False,       # if True, single scalar shared across K
    "tv_clip_max":     0.09,        # μ clamp upper bound (2026-06-29: was 0.05 display window; raised to physical mu)
    "tv_eps":          1.0e-6,      # smooth-TV epsilon
    # Training
    "epochs":          10,
    "batch_size":      1,
    "lr":              5e-3,        # outer optimiser lr (over the 20 scalars)
    "lambda_neg":      1.0,
    "loss_base":       "mse",       # or "l1"
    "grad_clip":       1.0,
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom)


def _smooth_tv_grad(f: torch.Tensor, eps: float) -> torch.Tensor:
    """Closed-form gradient of isotropic smooth TV
       TV(f) = sum sqrt(eps + (∇x f)^2 + (∇y f)^2)
    Returns ∂TV/∂f with the same shape as `f` ((B, 1, H, W)).
    """
    # Forward differences with replicate padding so the divergence at the
    # boundary stays finite.
    dx = f[..., :, 1:] - f[..., :, :-1]    # (B,1,H, W-1)
    dy = f[..., 1:, :] - f[..., :-1, :]    # (B,1,H-1,W)
    # Pad to (B,1,H,W) so the per-pixel magnitude is well-defined.
    dx_full = F.pad(dx, (0, 1, 0, 0))      # pad right
    dy_full = F.pad(dy, (0, 0, 0, 1))      # pad bottom
    mag = torch.sqrt(dx_full ** 2 + dy_full ** 2 + eps)  # (B,1,H,W)
    # ∂/∂f_ij of TV ≈ -divergence(grad / mag).
    nx = dx_full / mag                     # x-component of unit normal
    ny = dy_full / mag                     # y-component
    # Backward divergence (matches the forward-diff above):
    # div_x(nx) = nx[..., :, i] - nx[..., :, i-1]
    div_x = nx - F.pad(nx[..., :, :-1], (1, 0, 0, 0))
    div_y = ny - F.pad(ny[..., :-1, :], (0, 0, 1, 0))
    return -(div_x + div_y)


class UnrolledTV(nn.Module):
    """K-step unrolled TV-GD with learnable per-iter step and lambda.

    Initialise from `fbp_init`; output is the K-th iterate after clamping
    to `[0, tv_clip_max]`.
    """

    def __init__(self, geometry: FanBeamGeometry, K: int,
                 step_init: float, lambda_init: float,
                 clip_max: float, eps: float, share_steps: bool):
        super().__init__()
        self.K = K
        self.clip_max = clip_max
        self.eps = eps
        self.proj = PyronnFanBeamProjector(geometry)
        n = 1 if share_steps else K
        # log-parametrise to keep positive.
        self.log_step = nn.Parameter(torch.full((n,), math.log(step_init)))
        self.log_lambda = nn.Parameter(torch.full((n,), math.log(lambda_init)))

    def _step_for(self, k: int):
        i = 0 if self.log_step.numel() == 1 else k
        return torch.exp(self.log_step[i]), torch.exp(self.log_lambda[i])

    def forward(self, sino: torch.Tensor, fbp_init: torch.Tensor) -> torch.Tensor:
        f = fbp_init
        for k in range(self.K):
            step, lam = self._step_for(k)
            Rf = self.proj.forward_project(f)
            data_grad = self.proj.back_project(Rf - sino)
            tv_grad = _smooth_tv_grad(f, self.eps)
            f = f - step * (data_grad + lam * tv_grad)
            # μ≥0 floor only. Upper clamp REMOVED 2026-06-30: clip_max truncated
            # bone (μ up to 0.0814) — same bug class as the metric clamp. Vestigial.
            f = f.clamp_min(0.0)
        return f


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    import os
    env_config_path = os.environ.get("TV_SUP_CONFIG_PATH")
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

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg, default=str)}", flush=True)
    torch.manual_seed(cfg["seed"])

    # Mayo: the val split is a single patient (L277); probe its native ps and
    # build the projector at it (canonical ps mis-scales L277 ~5%). Training
    # reuses the same ps -- the 20 TV scalars are ps-robust. Mirrors tv_search.
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

    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    proj_full = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        ld_fbp_train = torch.clamp(proj_full.fbp(train_noisy), min=0.0)
        ld_fbp_val   = torch.clamp(proj_full.fbp(val_noisy),   min=0.0)
    per_ps = False  # single-ps projector (probe); loops below skip per-sample swaps

    model = UnrolledTV(
        geom, K=cfg["tv_K"],
        step_init=cfg["tv_step_init"], lambda_init=cfg["tv_lambda_init"],
        clip_max=max(cfg["tv_clip_max"], cfg["display_max"]), eps=cfg["tv_eps"],  # 2026-06-29: box >= display_max (0.09); 0.05 truncated bone
        share_steps=cfg["tv_share_steps"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] Unrolled-TV-L2: K={cfg['tv_K']} share={cfg['tv_share_steps']} "
          f"params={n_params} (clip_max={cfg['tv_clip_max']}, eps={cfg['tv_eps']})", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    t0 = time.time()
    bs = 1 if per_ps else cfg["batch_size"]
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0; n_seen = 0
        for i in range(0, train_noisy.shape[0], bs):
            idx = perm[i:i + bs]
            if per_ps:
                model.proj = _projs[float(_trk[int(idx[0])])]
            sino = train_noisy[idx].to(device)
            truth = train_ph[idx].to(device)
            fbp0 = ld_fbp_train[idx].to(device)
            pred = model(sino, fbp0)
            loss = supervised_recon_loss(pred, truth,
                                         lambda_neg=cfg["lambda_neg"],
                                         base=cfg["loss_base"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += float(loss.detach().cpu()) * idx.numel()
            n_seen += idx.numel()
        mean_loss = running / max(1, n_seen)
        with torch.no_grad():
            steps = torch.exp(model.log_step).detach().cpu().numpy().tolist()
            lams  = torch.exp(model.log_lambda).detach().cpu().numpy().tolist()
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.6f}  "
              f"step[0..K-1]={[f'{s:.4g}' for s in steps[:5]]}{'...' if len(steps)>5 else ''}  "
              f"lambda[0..K-1]={[f'{l:.4g}' for l in lams[:5]]}{'...' if len(lams)>5 else ''}",
              flush=True)
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        chunk = 1 if per_ps else cfg.get("val_chunk", 5)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            if per_ps:
                model.proj = _projs[float(_vrk[i])]
            preds.append(model(val_noisy[i:i+chunk], ld_fbp_val[i:i+chunk]))
        pred = torch.cat(preds, dim=0)

    pred = pred.clamp_min(0.0)
    ld_fbp_val = ld_fbp_val.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=ld_fbp_val,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    val_score = val_ssim

    # Final learned scalars for the autoresearch journal.
    with torch.no_grad():
        learned_steps = [float(s) for s in torch.exp(model.log_step).cpu().tolist()]
        learned_lambdas = [float(l) for l in torch.exp(model.log_lambda).cpu().tolist()]

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": n_params / 1e6,
        "train_n": cfg["train_n"], "val_n": cfg["val_n"],
        "train_time_s": train_time,
        "learned_steps": learned_steps,
        "learned_lambdas": learned_lambdas,
        "config": cfg,
        "training_scheme": "supervised_l2_unrolled_TV_K{}".format(cfg["tv_K"]),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Unrolled-TV-L2: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f}  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="TV-L2-unrolled", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
