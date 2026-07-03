"""Reference: Hammernik 2017 — Variational Network for limited-angle CT.

Adapted to our sparse-view setup. We skip the Würfl 2016 compensation-weights
step (Step 1 of the paper) because sparse-view CT has full angular range —
the standard FBP + non-negativity is good enough as the variational network's
starting iterate ``y_NN``.

The variational network (Step 2) is the paper's main contribution:

    y^t = y^{t-1}
            - Σ_i K^T_{i,t} · ρ'_{i,t}( K_{i,t} · y^{t-1} )    (gradient of regulariser)
            - λ_t · ( y^{t-1} - y_NN )                          (gradient of data term)

Each unrolled step has its own bank of N_k learned 2-D filters K_{i,t},
learned activation derivatives ρ'_{i,t} (parameterised as a weighted sum
of Gaussian RBFs on a fixed grid — Chen-Yu-Pock 2015), and a learned
data-fidelity weight λ_t. End-to-end MSE training.

Citation: Hammernik K., Würfl T., Pock T., Maier A., "A deep learning
architecture for limited-angle computed tomography reconstruction",
BVM 2017. See literature/hammernik_2017_bvm_limited_angle.md.
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
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty, clip_and_step
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    "train_n":       200,
    # Variational-network architecture (paper defaults: T=5, N_k=24, k=13)
    "vn_T":          5,
    "vn_n_filters":  24,
    "vn_kernel":     11,         # k=13 best per paper, k=11 cheaper at -0.003 SSIM
    "vn_n_bumps":    31,         # RBF activation grid
    "vn_x_range":    2.0,        # RBF centres span [-x_range, +x_range]
    "vn_filter_init_std": 0.05,
    "vn_rbf_init_std":    0.01,
    "vn_lambda_init":     1.0e-3,
    # Training
    "epochs":        20,
    "batch_size":    4,
    "lr":            5e-4,
}


# ---------------------------------------------------------------------------
class GDStep(nn.Module):
    """One unrolled gradient-descent step of Hammernik's variational network.

    Parameters
    ----------
    n_filters : int
        Number of 2-D analysis filters K_i in this step's filter bank.
    kernel_size : int
        Spatial size of each filter (square).
    n_bumps : int
        Number of Gaussian RBFs parameterising ρ'_i (per filter).
    x_range : float
        RBF centres span [-x_range, +x_range].
    lambda_init : float
        Initial data-fidelity weight (softplus-parameterised so it stays >0).
    """

    def __init__(self, n_filters=24, kernel_size=11, n_bumps=31,
                 x_range=2.0, filter_init_std=0.05, rbf_init_std=0.01,
                 lambda_init=1e-3):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.n_bumps = int(n_bumps)
        # Filter weights K_i: (n_filters, 1, k, k).
        self.weight = nn.Parameter(
            torch.randn(n_filters, 1, kernel_size, kernel_size) * filter_init_std
        )
        # RBF centres on fixed grid.
        centres = torch.linspace(-x_range, x_range, n_bumps)
        # Width set so adjacent RBFs overlap at ~e^{-1/2} (i.e. σ = grid spacing).
        sigma = 2.0 * x_range / max(1, n_bumps - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("inv_sigma_sq", torch.tensor(1.0 / (sigma ** 2)))
        # Per-filter, per-bump weight for ρ'_i.
        self.rbf_weights = nn.Parameter(
            torch.randn(n_filters, n_bumps) * rbf_init_std
        )
        # Learnable data-fidelity weight λ_t > 0 via softplus.
        # log_lambda chosen so softplus(log_lambda) ≈ lambda_init.
        # softplus^{-1}(y) = log(exp(y) - 1) for y > 0.
        inv_softplus = math.log(math.expm1(max(lambda_init, 1e-6)))
        self.log_lambda = nn.Parameter(torch.tensor(float(inv_softplus)))

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def _rho_prime(self, Kx: torch.Tensor) -> torch.Tensor:
        """Apply ρ'_i pointwise. Kx shape (B, F, H, W) → (B, F, H, W).

        Implemented as a chunked sum over RBF bumps so we never
        materialise the (B, F, H, W, n_bumps) tensor.
        """
        out = torch.zeros_like(Kx)
        for j in range(self.n_bumps):
            mu_j = self.centres[j]
            diff_j = Kx - mu_j
            bump = torch.exp(-0.5 * (diff_j ** 2) * self.inv_sigma_sq)
            w_j = self.rbf_weights[:, j].view(1, self.n_filters, 1, 1)
            out = out + bump * w_j
        return out

    def forward(self, y: torch.Tensor, y_NN: torch.Tensor) -> torch.Tensor:
        # y, y_NN: (B, 1, H, W).
        pad = self.kernel_size // 2
        # K_i · y for all filters: (B, F, H, W)
        Kx = F.conv2d(y, self.weight, padding=pad)
        # ρ'_i(K_i y): (B, F, H, W)
        rho_Kx = self._rho_prime(Kx)
        # K^T_i · ρ'_i(...). Pad symmetric padding so conv_transpose gives same H,W.
        KT_rho = F.conv_transpose2d(rho_Kx, self.weight, padding=pad)
        # Update.
        return y - KT_rho - self.lam * (y - y_NN)


class HammernikVN(nn.Module):
    """T unrolled GD steps with untied weights — Hammernik 2017 Step 2."""

    def __init__(self, T=5, n_filters=24, kernel_size=11, n_bumps=31,
                 x_range=2.0, filter_init_std=0.05, rbf_init_std=0.01,
                 lambda_init=1e-3):
        super().__init__()
        self.steps = nn.ModuleList([
            GDStep(n_filters=n_filters, kernel_size=kernel_size,
                   n_bumps=n_bumps, x_range=x_range,
                   filter_init_std=filter_init_std,
                   rbf_init_std=rbf_init_std,
                   lambda_init=lambda_init)
            for _ in range(T)
        ])

    def forward(self, y_NN: torch.Tensor) -> torch.Tensor:
        y = y_NN
        for step in self.steps:
            y = step(y, y_NN)
        return y


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
                          geom=geom, return_ps=True)   # 4-tuple; ps=None for non-mayo


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("HAMMERNIK_CONFIG_PATH")
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}", flush=True)
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k.startswith('vn_') or k in ('epochs','batch_size','lr','train_n','val_n')}, default=str)}",
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

    # PER-SAMPLE geometry (Mayo canonical): this VN denoises an image-space FBP,
    # so only the FBP (+ baseline) needs to be per-ps; no in-model projector.
    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)
    with torch.no_grad():
        if per_ps:
            train_fbp = mayo_per_sample_fbp(_projs, _trk, train_noisy, cfg["image_size"])
            val_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            proj = PyronnFanBeamProjector(geom).to(device)
            train_fbp = torch.clamp(proj.fbp(train_noisy), min=0.0)    # Ψ ≡ ReLU
            val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    model = HammernikVN(
        T=cfg["vn_T"],
        n_filters=cfg["vn_n_filters"],
        kernel_size=cfg["vn_kernel"],
        n_bumps=cfg["vn_n_bumps"],
        x_range=cfg["vn_x_range"],
        filter_init_std=cfg["vn_filter_init_std"],
        rbf_init_std=cfg["vn_rbf_init_std"],
        lambda_init=cfg["vn_lambda_init"],
    ).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] Hammernik VN params: {params_total/1e3:.2f} k "
          f"(T={cfg['vn_T']}, N_k={cfg['vn_n_filters']}, k={cfg['vn_kernel']}, "
          f"n_bumps={cfg['vn_n_bumps']})", flush=True)

    # Opt-in "train once, save checkpoint, reuse" hook. Gated ENTIRELY on the
    # AGENT4CT_MODEL_CKPT env var, which is set ONLY by the testset sweep worker
    # (scripts/score_mayo_testset.py). Unset in normal runs -> no-op, byte-
    # identical behaviour. When set: the FIRST patient's process trains + saves
    # the ckpt; the other four LOAD it and skip the training loop. Training is
    # deterministic + patient-independent (train set is always the 4 fixed train
    # patients; only the EVAL set changes per patient), so 5 trainings/iter -> 1.
    _CKPT = os.environ.get("AGENT4CT_MODEL_CKPT")
    _CKPT_EXISTS = bool(_CKPT) and Path(_CKPT).exists()

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    train_start = time.time()
    if _CKPT_EXISTS:
        # Skip-branch: load the trained weights, do NOT run the training loop.
        print(f"[solver] loading checkpoint {_CKPT} (skip training)", flush=True)
        model.load_state_dict(torch.load(_CKPT, map_location=device))
    for ep in ([] if _CKPT_EXISTS else range(cfg["epochs"])):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0
        for i in range(0, cfg["train_n"], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            x0 = train_fbp[idx]
            truth = train_ph[idx]
            pred = model(x0)
            loss = supervised_recon_loss(pred, truth, lambda_neg=1.0)
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
            running += float(loss.detach().cpu())
            n_batches += 1
        lambdas = [float(s.lam.detach().cpu()) for s in model.steps]
        avg_loss = running / max(1, n_batches)
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"λ_t={[f'{l:.3g}' for l in lambdas]}", flush=True)
        if time.time() - train_start > cfg.get("max_train_s", 1800):
            print(f"[train] wall ({cfg.get('max_train_s', 1800)}s) reached at epoch {ep+1}", flush=True)
            break
    train_time = time.time() - train_start

    if _CKPT and not _CKPT_EXISTS:
        # First patient: persist the trained model for the other four to reuse.
        Path(_CKPT).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), _CKPT)
        print(f"[solver] saved checkpoint {_CKPT}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        chunk = max(1, cfg["batch_size"])
        for i in range(0, val_fbp.shape[0], chunk):
            preds.append(model(val_fbp[i:i + chunk]))
    pred = torch.cat(preds, dim=0)

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

    print(f"[solver] λ_t per step: "
          f"{[f'{float(s.lam.detach().cpu()):.4g}' for s in model.steps]}",
          flush=True)

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time,
        "lambdas_learned": [float(s.lam.detach().cpu()) for s in model.steps],
        "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Hammernik VN: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"time={train_time:.1f}s  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="Hammernik2017", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
