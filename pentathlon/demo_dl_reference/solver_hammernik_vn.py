"""Reference: Hammernik-VN — MRI Variational Network (Hammernik 2018 MRM, arXiv 1704.00447)
adapted to sparse-view CT.

Each unrolled gradient-descent step combines a learned regulariser
gradient with a data-consistency step against the measured sinogram:

    u^{t+1} = u^t
              - Σ_i (K_i^t)^T Φ'_i^t( K_i^t · u^t )      (regulariser grad)
              - λ^t · R^T( R · u^t  -  g )               (data fidelity grad)

The regulariser parameterisation (per-step learned filter banks + per-filter
learned RBF activations) is identical to solver_hammernik_2017.py. The
new ingredient versus that solver is the **data-consistency term using
the actual forward projector R**, i.e. the architectural twin of
ItNet v3 but with a tiny, interpretable per-step regulariser instead of
a 5-level U-Net.

Citation: Hammernik K., Klatzer T., Kobler E., Recht M., Sodickson D.,
Pock T., Knoll F., "Learning a Variational Network for Reconstruction
of Accelerated MRI Data", Magnetic Resonance in Medicine, 79(6), 2018.
See literature/hammernik_2018_mri_variational_network.md.
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
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, negativity_penalty


CONFIG = {
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,
    "train_n":       200,
    "val_n":         100,
    "noise_i0":      1e5,
    "noise_sigma_e": 10.0,
    "seed":          42,
    "display_min":   0.0,
    "display_max":   0.05,
    # Hammernik-VN architecture. Paper used (T=10, N_k=48, k=11) on 320x288 MRI;
    # at our 512x512 fan-beam this OOMs a 24 GB Q8000. Memory-fitted defaults
    # (T=5, N_k=24) match the BVM 2017 budget while keeping the projector-DC step.
    "vn_T":             5,
    "vn_n_filters":     24,
    "vn_kernel":        11,
    "vn_n_bumps":       31,
    "vn_x_range":       1.0,
    "vn_filter_init_std": 0.05,
    "vn_rbf_init_std":  0.01,
    "vn_lambda_init":   1.0e-3,
    "vn_init":          "fbp",     # "fbp" (default) or "backproj" (paper-faithful)
    "vn_normalize_filters": False, # paper constrains filters zero-mean unit-norm
    "vn_dc_norm":       True,      # divide R^T(R x - g) by an estimate of ‖R^T R‖
                                   # so λ_t lives in O(1); prevents divergence
    "vn_checkpoint":    True,      # gradient-checkpoint each unrolled step
    # Training
    "epochs":     12,
    "batch_size": 2,
    "lr":         2e-4,
}


# ---------------------------------------------------------------------------
class VNStep(nn.Module):
    """One unrolled gradient-descent step of the variational network.

    Combines the regulariser-gradient `Σ_i K_i^T ρ'_i(K_i u)` with the
    data-fidelity gradient `λ · R^T(R u - g)` through the actual forward
    projector. Untied weights across steps, like the paper.
    """

    def __init__(self, projector: PyronnFanBeamProjector,
                 n_filters=24, kernel_size=11, n_bumps=31,
                 x_range=1.0, filter_init_std=0.05, rbf_init_std=0.01,
                 lambda_init=1e-3, normalize_filters=False,
                 dc_norm=1.0):
        super().__init__()
        self.projector = projector              # shared single instance, not a sub-module
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.n_bumps = int(n_bumps)
        self.normalize_filters = bool(normalize_filters)
        # Scalar dividing R^T(R x - g) so λ_t stays in O(1) regardless of
        # the projector's spectral norm. Estimated from a power iteration
        # at HammernikVN init.
        self.register_buffer("dc_norm", torch.tensor(float(dc_norm)))
        # Filter weights K_i: (n_filters, 1, k, k).
        self.weight = nn.Parameter(
            torch.randn(n_filters, 1, kernel_size, kernel_size) * filter_init_std
        )
        # RBF centres + width.
        centres = torch.linspace(-x_range, x_range, n_bumps)
        sigma = 2.0 * x_range / max(1, n_bumps - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("inv_sigma_sq", torch.tensor(1.0 / (sigma ** 2)))
        # Per-filter, per-bump RBF weights.
        self.rbf_weights = nn.Parameter(
            torch.randn(n_filters, n_bumps) * rbf_init_std
        )
        # Learnable λ_t via softplus to enforce positivity (paper Eq. 6).
        inv_softplus = math.log(math.expm1(max(lambda_init, 1e-6)))
        self.log_lambda = nn.Parameter(torch.tensor(float(inv_softplus)))

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def _effective_weight(self) -> torch.Tensor:
        """Optionally enforce paper's zero-mean unit-norm filter constraint."""
        if not self.normalize_filters:
            return self.weight
        w = self.weight
        flat = w.view(self.n_filters, -1)
        flat = flat - flat.mean(dim=1, keepdim=True)             # zero-mean
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        flat = flat / norms                                       # unit-norm
        return flat.view_as(w)

    def _rho_prime(self, Kx: torch.Tensor) -> torch.Tensor:
        """Pointwise RBF mixture per filter; sum over bumps in a loop to avoid
        materialising the (B, F, H, W, n_bumps) tensor."""
        out = torch.zeros_like(Kx)
        for j in range(self.n_bumps):
            mu_j = self.centres[j]
            bump = torch.exp(-0.5 * (Kx - mu_j) ** 2 * self.inv_sigma_sq)
            w_j = self.rbf_weights[:, j].view(1, self.n_filters, 1, 1)
            out = out + bump * w_j
        return out

    def forward(self, u: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        # u: (B,1,H,W); sino: (B,1,A,D).
        pad = self.kernel_size // 2
        w_eff = self._effective_weight()

        # Regulariser-gradient term.
        Kx = F.conv2d(u, w_eff, padding=pad)
        rho_Kx = self._rho_prime(Kx)
        KT_rho = F.conv_transpose2d(rho_Kx, w_eff, padding=pad)

        # Data-consistency term via the actual projector. Divide by the
        # power-iteration estimate of ‖R^T R‖ so λ_t stays in O(1).
        R_u = self.projector.forward_project(u)               # (B,1,A,D)
        sino_residual = R_u - sino
        R_T_residual = self.projector.back_project(sino_residual) / self.dc_norm

        return u - KT_rho - self.lam * R_T_residual


class HammernikVN(nn.Module):
    """T unrolled VN steps with untied weights and projection-domain DC."""

    def __init__(self, projector: PyronnFanBeamProjector,
                 T=5, n_filters=24, kernel_size=11, n_bumps=31,
                 x_range=1.0, filter_init_std=0.05, rbf_init_std=0.01,
                 lambda_init=1e-3, normalize_filters=False,
                 dc_norm=True, checkpoint=True):
        super().__init__()
        self.projector = projector
        self.checkpoint = bool(checkpoint)
        # Power-iteration estimate of ‖R^T R‖.  Used to scale the DC step
        # so λ_t stays in O(1) regardless of geometry.
        norm_val = 1.0
        if dc_norm:
            with torch.no_grad():
                device = next(projector.parameters(), torch.zeros(1)).device \
                    if any(True for _ in projector.parameters()) else "cpu"
                v = torch.randn(1, 1, projector.geom.image_size,
                                projector.geom.image_size, device=device)
                v = v / v.norm()
                for _ in range(8):
                    Av = projector.forward_project(v)
                    v = projector.back_project(Av)
                    n = v.norm().clamp(min=1e-12)
                    v = v / n
                norm_val = float(n.item())
                print(f"[HammernikVN] dc_norm power-iter ≈ {norm_val:.3g}",
                      flush=True)
        self.steps = nn.ModuleList([
            VNStep(projector, n_filters=n_filters, kernel_size=kernel_size,
                   n_bumps=n_bumps, x_range=x_range,
                   filter_init_std=filter_init_std,
                   rbf_init_std=rbf_init_std,
                   lambda_init=lambda_init,
                   normalize_filters=normalize_filters,
                   dc_norm=norm_val)
            for _ in range(T)
        ])

    def forward(self, u0: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        u = u0
        for step in self.steps:
            if self.checkpoint and u.requires_grad:
                u = torch.utils.checkpoint.checkpoint(step, u, sino,
                                                       use_reentrant=False)
            else:
                u = step(u, sino)
        return u


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
                          geom=geom)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("HAMMERNIK_VN_CONFIG_PATH")
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

    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"],
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)

    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        if cfg["vn_init"] == "fbp":
            train_u0 = torch.clamp(proj.fbp(train_noisy), min=0.0)
            val_u0   = torch.clamp(proj.fbp(val_noisy),   min=0.0)
        elif cfg["vn_init"] == "backproj":
            train_u0 = proj.back_project(train_noisy)
            val_u0   = proj.back_project(val_noisy)
        else:
            raise ValueError(f"unknown vn_init={cfg['vn_init']!r}")

    model = HammernikVN(
        projector=proj,
        T=cfg["vn_T"],
        n_filters=cfg["vn_n_filters"],
        kernel_size=cfg["vn_kernel"],
        n_bumps=cfg["vn_n_bumps"],
        x_range=cfg["vn_x_range"],
        filter_init_std=cfg["vn_filter_init_std"],
        rbf_init_std=cfg["vn_rbf_init_std"],
        lambda_init=cfg["vn_lambda_init"],
        normalize_filters=cfg["vn_normalize_filters"],
        dc_norm=cfg.get("vn_dc_norm", True),
        checkpoint=cfg.get("vn_checkpoint", True),
    ).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] Hammernik-VN params: {params_total/1e3:.2f} k "
          f"(T={cfg['vn_T']}, N_k={cfg['vn_n_filters']}, k={cfg['vn_kernel']}, "
          f"init={cfg['vn_init']}, norm={cfg['vn_normalize_filters']})",
          flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    train_start = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0
        for i in range(0, cfg["train_n"], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            u0 = train_u0[idx]
            sino = train_noisy[idx]
            truth = train_ph[idx]
            pred = model(u0, sino)
            loss = supervised_recon_loss(pred, truth, lambda_neg=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            n_batches += 1
        avg_loss = running / max(1, n_batches)
        lambdas = [float(s.lam.detach().cpu()) for s in model.steps]
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"λ_t={[f'{l:.3g}' for l in lambdas]}", flush=True)
        if time.time() - train_start > 480:        # 8-min wall
            print(f"[train] 8-min wall reached at epoch {ep+1}", flush=True)
            break
    train_time = time.time() - train_start

    model.eval()
    preds = []
    with torch.no_grad():
        chunk = max(1, cfg["batch_size"])
        for i in range(0, val_u0.shape[0], chunk):
            preds.append(model(val_u0[i:i + chunk], val_noisy[i:i + chunk]))
    pred = torch.cat(preds, dim=0)

    # baseline = the FBP starting point (regardless of vn_init choice — use FBP
    # for fair comparison against the rest of the demo_dl_reference table)
    with torch.no_grad():
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
    print(f"[solver] Hammernik-VN: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"time={train_time:.1f}s  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="HammernikVN", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
