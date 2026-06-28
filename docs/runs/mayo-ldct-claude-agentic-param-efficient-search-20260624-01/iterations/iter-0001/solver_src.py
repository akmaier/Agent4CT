"""Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-1 SEED).

A weight-TIED unrolled proximal-gradient reconstruction with explicit
data-consistency against the measured sinogram. The SAME regulariser
module and the SAME step-size scalar `alpha` are reused at every unrolled
step, so the trainable parameter budget is set by ONE small regulariser
(hundreds to low-thousands of params) regardless of `n_iter` — in sharp
contrast to the 233k-param ITNet champion whose denoiser is a full SmallUNet.

Architecture (per step k, all weights tied across k):

    x_0 = LD-FBP(sino)
    for k in range(K):
        dc = R^T( R x  -  sino ) / dc_norm          # data-consistency grad
        x  = clamp( x - alpha * ( dc + reg(x) ),     # proximal-gradient step
                    0.0, clip_max )

`dc_norm` is a power-iteration estimate of ‖R^T R‖ so `alpha` lives in O(1)
regardless of geometry (mirrors solver_hammernik_vn.py). `alpha` is a
learnable softplus scalar (init from `alpha_init`) when `learnable_alpha`.

The learned regulariser `reg(x)` is selected by `reg_type`:
  - "cnn"       (DEFAULT — most likely to train stably): a tiny residual CNN,
                `cnn_layers` 3x3 convs (`cnn_channels` channels, GroupNorm+ReLU
                between), zero-init final 1x1 head so reg ≈ 0 at init.
  - "foe"       a single Fields-of-Experts / VN filter bank: conv2d analysis
                (`foe_n_filters` filters, `foe_kernel`x`foe_kernel`) → RBF
                activation (`foe_n_bumps` bumps) → tied conv_transpose2d
                synthesis (mirrors solver_hammernik_vn.py VNStep, but tied).
  - "bilateral" a cascade of `n_bf` TrainableBilateralFilter2d (3 params each).

Trained end-to-end supervised against the HD truth image
(`supervised_recon_loss`, Adam). The DC term + a modest learnable `alpha`
keep the recon data-consistent so it beats the LD-FBP RMSE floor (the
headroom gate is RMSE-vs-LD-FBP, not SSIM).

Citation context: this is the parameter-tied limit of the unrolled
proximal-gradient family (Hammernik 2018 MRM variational network; Adler &
Öktem 2018 learned primal-dual). See literature/ for the lineage.
"""
from __future__ import annotations
import argparse
import json
import math
import os
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
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.metrics import (psnr, ssim, evaluate_calibrated,
                                make_4panel_comparison, supervised_recon_loss,
                                negativity_penalty, clip_and_step)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # ---- architecture (the SEED) ----
    "reg_type":        "cnn",      # "cnn" (default) | "foe" | "bilateral"
    "n_iter":          5,          # unrolled proximal-gradient steps (weight-tied)
    "learnable_alpha": True,       # alpha = softplus(param), init from alpha_init
    "alpha_init":      0.1,        # step size (O(1) thanks to dc_norm scaling)
    "clip_max":        0.05,       # per-step clamp upper bound (= display_max μ)
    "dc_norm":         True,       # divide R^T(R x - g) by power-iter ‖R^T R‖
    "checkpoint":      True,       # gradient-checkpoint each unrolled step
    # ---- "cnn" regulariser ----
    "cnn_channels":    12,    # 2.8k params at 3 layers (low-thousands target)
    "cnn_layers":      3,
    # ---- "foe" regulariser ----
    "foe_n_filters":   8,
    "foe_kernel":      5,
    "foe_n_bumps":     15,
    "foe_x_range":     1.0,
    "foe_filter_init_std": 0.05,
    "foe_rbf_init_std":    0.01,
    # ---- "bilateral" regulariser ----
    "n_bf":            4,
    "bf_kernel":       7,
    # ---- training ----
    "train_n":   200,
    "val_n":     214,
    "epochs":    8,
    "lr":        5e-3,
    "batch_size": 1,               # per-sample-ps geometry (Mayo): keep at 1
    "lambda_neg": 1.0,
    "grad_clip": 1.0,
    "seed":      42,
}


# ---------------------------------------------------------------------------
# Learned regularisers. Each module maps (B,1,H,W) -> (B,1,H,W); the value is
# the regulariser GRADIENT contribution `reg(x)` added inside the prox step.
# All are weight-tied (one instance reused across every unrolled iteration).
# ---------------------------------------------------------------------------
class CNNReg(nn.Module):
    """Tiny residual CNN regulariser. `layers` 3x3 convs with GroupNorm+ReLU
    between them and a zero-initialised 1x1 head, so reg(x) ≈ 0 at init (the
    seed therefore starts as a clean gradient-descent-with-DC scheme and
    learns a correction)."""

    def __init__(self, channels: int = 16, layers: int = 3):
        super().__init__()
        channels = int(channels)
        layers = max(1, int(layers))
        body: list[nn.Module] = [nn.Conv2d(1, channels, 3, padding=1)]
        for _ in range(layers - 1):
            body += [nn.GroupNorm(_pick_groups(channels), channels),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(channels, channels, 3, padding=1)]
        self.body = nn.Sequential(*body)
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(self.body(x)))


def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class FoEReg(nn.Module):
    """Single tied Fields-of-Experts / VN filter bank.

    reg(x) = K^T ρ'(K x), with K an analysis conv2d bank (n_filters,
    kernel x kernel), ρ' a per-filter RBF mixture (n_bumps bumps), and K^T
    the tied conv_transpose2d synthesis — exactly the regulariser-gradient
    of one solver_hammernik_vn.py VNStep, but ONE bank reused at every
    unrolled step (weight-tied) instead of T untied banks."""

    def __init__(self, n_filters: int = 8, kernel_size: int = 5,
                 n_bumps: int = 15, x_range: float = 1.0,
                 filter_init_std: float = 0.05, rbf_init_std: float = 0.01):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.n_bumps = int(n_bumps)
        self.weight = nn.Parameter(
            torch.randn(self.n_filters, 1, self.kernel_size, self.kernel_size)
            * filter_init_std)
        centres = torch.linspace(-x_range, x_range, self.n_bumps)
        sigma = 2.0 * x_range / max(1, self.n_bumps - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("inv_sigma_sq", torch.tensor(1.0 / (sigma ** 2)))
        self.rbf_weights = nn.Parameter(
            torch.randn(self.n_filters, self.n_bumps) * rbf_init_std)

    def _rho_prime(self, Kx: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(Kx)
        for j in range(self.n_bumps):
            mu_j = self.centres[j]
            bump = torch.exp(-0.5 * (Kx - mu_j) ** 2 * self.inv_sigma_sq)
            w_j = self.rbf_weights[:, j].view(1, self.n_filters, 1, 1)
            out = out + bump * w_j
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        Kx = F.conv2d(x, self.weight, padding=pad)
        rho_Kx = self._rho_prime(Kx)
        return F.conv_transpose2d(rho_Kx, self.weight, padding=pad)


class BilateralReg(nn.Module):
    """Cascade of `n_bf` TrainableBilateralFilter2d (3 params each). The
    regulariser gradient is the residual `x - cascade(x)` (a denoise-then-
    subtract proximal correction), keeping it ≈0 for an already-smooth x."""

    def __init__(self, n_bf: int = 4, kernel_size: int = 7):
        super().__init__()
        self.filters = nn.ModuleList(
            [TrainableBilateralFilter2d(kernel_size=int(kernel_size))
             for _ in range(max(1, int(n_bf)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for f in self.filters:
            y = f(y)
        return x - y


def build_reg(cfg: dict) -> nn.Module:
    rt = cfg["reg_type"]
    if rt == "cnn":
        return CNNReg(channels=cfg["cnn_channels"], layers=cfg["cnn_layers"])
    if rt == "foe":
        return FoEReg(n_filters=cfg["foe_n_filters"], kernel_size=cfg["foe_kernel"],
                      n_bumps=cfg["foe_n_bumps"], x_range=cfg["foe_x_range"],
                      filter_init_std=cfg["foe_filter_init_std"],
                      rbf_init_std=cfg["foe_rbf_init_std"])
    if rt == "bilateral":
        return BilateralReg(n_bf=cfg["n_bf"], kernel_size=cfg["bf_kernel"])
    raise ValueError(f"unknown reg_type={rt!r} (expected cnn|foe|bilateral)")


# ---------------------------------------------------------------------------
class ParamEfficientUnrolled(nn.Module):
    """Weight-tied unrolled proximal-gradient with data consistency.

    x_0 = u0 (LD-FBP). For k in range(n_iter):
        dc = R^T(R x - g) / dc_norm
        x  = clamp(x - alpha * (dc + reg(x)), 0.0, clip_max)
    `reg` and `alpha` are SHARED across all steps (weight-tied)."""

    def __init__(self, projector: PyronnFanBeamProjector, cfg: dict,
                 dc_norm: float = 1.0):
        super().__init__()
        self.projector = projector             # shared single instance, not a sub-module
        self.n_iter = int(cfg["n_iter"])
        self.clip_max = float(cfg["clip_max"])
        self.checkpoint = bool(cfg.get("checkpoint", True))
        self.register_buffer("dc_norm", torch.tensor(float(dc_norm)))
        self.reg = build_reg(cfg)              # ONE tied regulariser
        # ONE tied step size.
        if cfg["learnable_alpha"]:
            inv_softplus = math.log(math.expm1(max(float(cfg["alpha_init"]), 1e-6)))
            self.log_alpha = nn.Parameter(torch.tensor(float(inv_softplus)))
            self._alpha_const = None
        else:
            self.log_alpha = None
            self._alpha_const = float(cfg["alpha_init"])

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is not None:
            return F.softplus(self.log_alpha)
        return torch.as_tensor(self._alpha_const, device=self.dc_norm.device,
                               dtype=self.dc_norm.dtype)

    def _step(self, x: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        R_x = self.projector.forward_project(x)
        dc = self.projector.back_project(R_x - sino) / self.dc_norm
        return torch.clamp(x - self.alpha * (dc + self.reg(x)),
                           0.0, self.clip_max)

    def forward(self, u0: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        x = u0
        for _ in range(self.n_iter):
            if self.checkpoint and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(self._step, x, sino,
                                                      use_reentrant=False)
            else:
                x = self._step(x, sino)
        return x


# ---------------------------------------------------------------------------
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
                          geom=geom, return_ps=True)


def _count_reg_params(cfg: dict) -> int:
    """Trainable param count of ONE regulariser at the given cfg (for the
    start-of-run print + sanity check; the tied model reuses this once)."""
    reg = build_reg(cfg)
    return sum(p.numel() for p in reg.parameters() if p.requires_grad)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("PARAM_EFFICIENT_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        env_cfg = json.loads(Path(env_path).read_text())
        cfg = {**CONFIG, **env_cfg, **(cfg or {})}
        print(f"[solver] Loaded config from {env_path}", flush=True)
    else:
        cfg = {**CONFIG, **(cfg or {})}

    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}", flush=True)
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k in ('reg_type','n_iter','learnable_alpha','alpha_init','clip_max','cnn_channels','cnn_layers','foe_n_filters','foe_kernel','foe_n_bumps','n_bf','bf_kernel','epochs','batch_size','lr','train_n','val_n')}, default=str)}",
          flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    train_ph, train_clean, train_noisy, train_ps = build_dataset(
        geom, cfg["train_n"], cfg["seed"],
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # PER-SAMPLE geometry (Mayo canonical): swap model.projector per slice +
    # build the per-ps FBP init/baseline (falls back to single proj non-mayo).
    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        if per_ps:
            train_u0 = mayo_per_sample_fbp(_projs, _trk, train_noisy, cfg["image_size"])
            val_u0   = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            train_u0 = torch.clamp(proj.fbp(train_noisy), min=0.0)
            val_u0   = torch.clamp(proj.fbp(val_noisy),   min=0.0)

    # Power-iteration estimate of ‖R^T R‖ so alpha stays in O(1).
    norm_val = 1.0
    if cfg.get("dc_norm", True):
        with torch.no_grad():
            v = torch.randn(1, 1, cfg["image_size"], cfg["image_size"], device=device)
            v = v / v.norm()
            for _ in range(8):
                Av = proj.forward_project(v)
                v = proj.back_project(Av)
                n = v.norm().clamp(min=1e-12)
                v = v / n
            norm_val = float(n.item())
            print(f"[solver] dc_norm power-iter ≈ {norm_val:.3g}", flush=True)

    model = ParamEfficientUnrolled(proj, cfg, dc_norm=norm_val).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_params = _count_reg_params(cfg)
    print(f"[solver] ParamEfficient seed: reg_type={cfg['reg_type']!r}  "
          f"n_iter={cfg['n_iter']} (weight-TIED)  "
          f"trainable params={params_total} "
          f"(reg={reg_params} + alpha={params_total - reg_params})  "
          f"= {params_total/1e6:.6f} M  vs 233k ITNet", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    train_start = time.time()
    bs = max(1, int(cfg["batch_size"]))
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0
        for i in range(0, cfg["train_n"], bs):
            idx = perm[i:i + bs]
            if per_ps:
                model.projector = _projs[float(_trk[int(idx[0])])]
            u0 = train_u0[idx]
            sino = train_noisy[idx]
            truth = train_ph[idx]
            pred = model(u0, sino)
            loss = supervised_recon_loss(pred, truth,
                                         lambda_neg=cfg["lambda_neg"], base="mse")
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
            running += float(loss.detach().cpu())
            n_batches += 1
        avg_loss = running / max(1, n_batches)
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"alpha={float(model.alpha.detach().cpu()):.4g}", flush=True)
        if time.time() - train_start > cfg.get("max_train_s", 1800):
            print(f"[train] wall ({cfg.get('max_train_s', 1800)}s) reached at epoch {ep+1}",
                  flush=True)
            break
    train_time = time.time() - train_start

    model.eval()
    preds = []
    with torch.no_grad():
        chunk = 1 if per_ps else max(1, bs)
        for i in range(0, val_u0.shape[0], chunk):
            if per_ps:
                model.projector = _projs[float(_vrk[i])]
            preds.append(model(val_u0[i:i + chunk], val_noisy[i:i + chunk]))
    pred = torch.cat(preds, dim=0)

    # baseline = the LD-FBP starting point (the headroom anchor).
    with torch.no_grad():
        if per_ps:
            val_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

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
    val_score = val_ssim

    result = {
        "val_score": val_score, "headroom": headroom,
        "val_ssim": val_ssim, "val_psnr": val_psnr, "val_rmse": val_rmse,
        "val_ssim_std": metrics["val_ssim_std"],
        "val_psnr_std": metrics["val_psnr_std"],
        "val_rmse_std": metrics["val_rmse_std"],
        "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6,
        "reg_type": cfg["reg_type"],
        "reg_params": reg_params,
        "n_iter": cfg["n_iter"],
        "alpha_learned": float(model.alpha.detach().cpu()),
        "train_n": cfg["train_n"], "val_n": cfg["val_n"],
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] ParamEfficient: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"params={params_total}  alpha={result['alpha_learned']:.4g}  "
          f"time={train_time:.1f}s  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label=f"ParamEff[{cfg['reg_type']}]",
            headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
