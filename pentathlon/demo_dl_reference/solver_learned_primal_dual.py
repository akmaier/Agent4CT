"""Learned Primal-Dual reconstruction (Adler & Öktem, IEEE TMI 2018).

Reference paper: `literature/1707.06474_Adler_LearnedPrimalDual_TMI2018.md`,
PDF at `papers/1707.06474_Adler_LearnedPrimalDual_TMI2018.pdf`.

Unrolls a primal-dual hybrid gradient (PDHG / Chambolle-Pock) iteration
into a deep network where the proximal operators are replaced by small
3-layer residual CNNs. The forward operator T (ray transform) and its
adjoint T* (back-projection) are kept as differentiable PYRO-NN layers
inside the network, so the network learns the *update rule* for an
iterative solver — not a one-shot post-processing.

Generalises (per Adler §III-D):
  - Classical PDHG / Chambolle-Pock (special case at N_primal=2, N_dual=1,
    fixed combining ops)
  - ADMM-Net (Sun, Li, Xu 2016 — special case of the dual update)
  - Learned-gradient schemes (Adler-Öktem 2017)
  - Gradient descent with step-length α

Default configuration follows the paper:
  I = 10 unrolled iterations
  N_primal = N_dual = 5 channels of inter-iteration "memory"
  3 conv layers per proximal CNN, 3×3 kernels, 32 hidden channels, PReLU
  Iteration-specific weights (no sharing across iterations)
  Total parameters at defaults: ≈ 2.4 × 10⁵

Training: supervised L2 against clean phantom + non-negativity penalty,
full 128 views (consistent with `solver_dual_ddomain_supervised.py`).
Zero-init for f_0 and h_0 (Adler reports FBP-init gives only marginal
training-speed improvement, no quality difference).
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
    clip_and_step,
)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # Architecture (Adler & Öktem 2018 defaults)
    "lpd_iters":      10,    # I — number of unrolled primal-dual iterations
    "lpd_n_primal":   5,     # primal state channels (inter-iter memory)
    "lpd_n_dual":     5,     # dual state channels
    "lpd_hidden":    32,     # hidden channels in each proximal CNN
    "lpd_share_weights": False,  # if True, share proximal CNNs across iters
    # Training
    "epochs":        10,
    "batch_size":     1,
    "lr":            1e-3,   # Adam default; Adler used cosine annealing 1e-3 → small
    "lr_schedule":   "cosine",  # "constant" or "cosine"
    "grad_clip":      1.0,   # global-norm gradient clip (Adler used 1.0)
    "lambda_neg":     1.0,   # non-negativity penalty weight
    "loss_base":     "mse",  # "mse" or "l1"
}


def build_dataset(geom, n, seed, i0, sigma_e, device):
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)   # 4-tuple; ps=None for non-mayo


# ---------------------------------------------------------------------------
class ProximalBlock(nn.Module):
    """3-layer residual conv block per Adler §IV-B-1.

    Output = leading `out_channels` of the input + Δ, where Δ is produced
    by the conv stack. Equivalent to ``Id + W3 ∘ PReLU ∘ W2 ∘ PReLU ∘ W1``
    applied to the input, with the identity skipping only the leading
    state channels (the operator-evaluation channels are concatenated
    as inputs and don't appear in the skip).
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.act1  = nn.PReLU(num_parameters=hidden, init=0.0)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.act2  = nn.PReLU(num_parameters=hidden, init=0.0)
        self.conv3 = nn.Conv2d(hidden, out_channels, 3, padding=1)
        # Xavier init (Adler §IV-B-1), zero bias.
        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """x: full input (skip + operator evaluations concatenated).
           skip: the leading state channels (out_channels wide) for the
                 identity skip path."""
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))
        h = self.conv3(h)
        return skip + h


class LearnedPrimalDual(nn.Module):
    """Unrolled Learned Primal-Dual algorithm (Adler & Öktem 2018, Algorithm 3).

    forward(sino_full) executes I primal-dual iterations:

        h_i ← Γ_θᵢᵈ( h_{i-1}, K(f_{i-1}^(2)), g )      # dual update
        f_i ← Λ_θᵢᵖ( f_{i-1}, [∂K(f_{i-1}^(1))]*(h_i^(1)) )  # primal update

    K = ray transform (forward_project). ∂K* = back_project (linear case).
    """

    def __init__(self, geometry: FanBeamGeometry,
                 iters: int = 10, n_primal: int = 5, n_dual: int = 5,
                 hidden: int = 32, share_weights: bool = False):
        super().__init__()
        self.geometry = geometry
        self.R = PyronnFanBeamProjector(geometry)
        self.I = iters
        self.n_primal = n_primal
        self.n_dual = n_dual

        # Dual block input: [h (n_dual), K(f^(2)) (1), g (1)] → n_dual + 2
        # Primal block input: [f (n_primal), T*(h^(1)) (1)] → n_primal + 1
        dual_in_c = n_dual + 2
        prim_in_c = n_primal + 1

        if share_weights:
            shared_dual = ProximalBlock(dual_in_c, hidden, n_dual)
            shared_prim = ProximalBlock(prim_in_c, hidden, n_primal)
            self.dual_blocks  = nn.ModuleList([shared_dual] * iters)
            self.primal_blocks = nn.ModuleList([shared_prim] * iters)
        else:
            self.dual_blocks = nn.ModuleList([
                ProximalBlock(dual_in_c, hidden, n_dual) for _ in range(iters)
            ])
            self.primal_blocks = nn.ModuleList([
                ProximalBlock(prim_in_c, hidden, n_primal) for _ in range(iters)
            ])

    def forward(self, sino_full: torch.Tensor) -> torch.Tensor:
        """sino_full: (B, 1, A, D) low-dose measurement.
        Returns: (B, 1, H, W) reconstruction (primary primal channel).
        """
        B, _, A, D = sino_full.shape
        H = W = self.geometry.image_size
        device = sino_full.device

        # Zero initial primal and dual states (Adler Eq. 9).
        f = torch.zeros(B, self.n_primal, H, W, device=device, dtype=sino_full.dtype)
        h = torch.zeros(B, self.n_dual, A, D, device=device, dtype=sino_full.dtype)

        for i in range(self.I):
            # Dual update: evaluate forward operator on the 2nd primal channel,
            # concat with current dual + measurement, run the dual proximal.
            Kf2 = self.R.forward_project(f[:, 1:2])           # (B, 1, A, D)
            dual_in = torch.cat([h, Kf2, sino_full], dim=1)   # (B, n_dual+2, A, D)
            h = self.dual_blocks[i](dual_in, skip=h)

            # Primal update: back-project the 1st dual channel, concat with
            # current primal, run the primal proximal.
            Adjh = self.R.back_project(h[:, 0:1])              # (B, 1, H, W)
            prim_in = torch.cat([f, Adjh], dim=1)              # (B, n_primal+1, H, W)
            f = self.primal_blocks[i](prim_in, skip=f)

        return f[:, 0:1]  # primary primal channel as the reconstruction


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    import os
    env_config_path = os.environ.get("LPD_CONFIG_PATH")
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

    # PER-SAMPLE geometry (Mayo canonical): reconstruct each slice at its own
    # ps_eff via a per-ps projector cache; swap model.R per sample (bs=1). The
    # canonical sino is angle-uniform so only the recon pixel-spacing varies.
    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)

    with torch.no_grad():
        if per_ps:
            ld_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            R_full = PyronnFanBeamProjector(geom).to(device)
            ld_fbp = torch.clamp(R_full.fbp(val_noisy), min=0.0)

    model = LearnedPrimalDual(
        geometry=geom,
        iters=cfg["lpd_iters"], n_primal=cfg["lpd_n_primal"],
        n_dual=cfg["lpd_n_dual"], hidden=cfg["lpd_hidden"],
        share_weights=cfg["lpd_share_weights"],
    ).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] Learned Primal-Dual: I={cfg['lpd_iters']} "
          f"N_primal={cfg['lpd_n_primal']} N_dual={cfg['lpd_n_dual']} "
          f"hidden={cfg['lpd_hidden']} share_weights={cfg['lpd_share_weights']} "
          f"params={params_total/1e3:.1f} k", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99))
    if cfg["lr_schedule"] == "cosine":
        total_steps = cfg["epochs"] * max(1, cfg["train_n"] // cfg["batch_size"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    else:
        sched = None

    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        n_seen = 0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if per_ps:                      # swap to this slice's ps projector
                model.R = _projs[float(_trk[int(idx[0])])]
            sino = train_noisy[idx].to(device)
            truth = train_ph[idx].to(device)
            pred = model(sino)
            loss = supervised_recon_loss(pred, truth,
                                          lambda_neg=cfg["lambda_neg"],
                                          base=cfg.get("loss_base", "mse"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip_and_step adds the nonfinite-grad skip the Mayo 2304-view FBP
            # needs (a finite loss can still carry an Inf grad). Only advance the
            # LR schedule when an actual step was taken.
            if clip_and_step(opt, loss, cfg.get("grad_clip", 0.0)) and sched is not None:
                sched.step()
            running += float(loss.detach().cpu()) * idx.numel()
            n_seen += idx.numel()
        mean_loss = running / max(1, n_seen)
        lr_now = opt.param_groups[0]["lr"]
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  lr={lr_now:.2e}", flush=True)

    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        chunk = cfg.get("val_chunk", 4)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            if per_ps:                      # val_chunk=1 for Mayo -> one ps/slice
                model.R = _projs[float(_vrk[i])]
            preds.append(model(val_noisy[i:i + chunk]))
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
        "architecture": "learned_primal_dual_adler_2018",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Learned Primal-Dual: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} params={params_total/1e3:.1f}k "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="LPD", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
