"""Single demo-dl phantom, four reconstructions side-by-side.

User wanted images, not 100-case numerical sweeps. Builds one demo-dl
phantom + the matching Poisson(I0)+Gaussian(σ_e) noisy sino, then runs:

  1. FBP                              (`PyronnFanBeamProjector.fbp`)
  2. TV-iterative                     (`solver_tv_iterative.tv_reconstruction`)
  3. Wu 2015 ADMM                     (`solver_wu_2015.wu_2015_reconstruct`)
  4. Dual-domain bilateral (init)     (sino-domain BF + image-domain BF
                                       at the solver's default initial σ's,
                                       no training)

For each: calibrated SSIM / PSNR / RMSE vs truth (FOV-masked, double-clamp
intensity_calibrate — matches every other comparison in this debug stream).

Output: a single PNG with 2 rows × 5 cols (row 0 = recons, row 1 = diffs).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import intensity_calibrate, fov_mask, psnr, ssim

# Solver-core functions we want to call
sys.path.insert(0, str(REPO / "pentathlon" / "demo_dl_reference"))
from solver_tv_iterative import tv_reconstruction
from solver_wu_2015 import wu_2015_reconstruct, _build_dense_projector

OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


def cal_metrics(pred, truth, *, dmax, fov):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    pc = pc * fov
    truth_m = truth * fov
    return (
        float(ssim(pc, truth_m, data_range=dmax).cpu()),
        float(psnr(pc, truth_m, data_range=dmax).cpu()),
        float(((pc - truth_m) ** 2).mean().sqrt().cpu()),
        pc,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = dict(DEMO_DL_DEFAULTS)
    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )
    proj = PyronnFanBeamProjector(geom).to(device)
    dmax = float(cfg["display_max"])

    # ── one phantom + simulated noisy sino ────────────────────────────────
    seed = int(cfg["seed"]) + 1000
    phantom = random_ellipses_phantom(size=geom.image_size, n_ellipses=10,
                                       seed=seed, device=device)   # (1,1,H,W)
    with torch.no_grad():
        clean_sino = proj.forward_project(phantom)
        noisy_sino = simulate_low_dose(clean_sino, i0=cfg["noise_i0"],
                                         sigma_e=cfg["noise_sigma_e"], seed=seed)
    fov = fov_mask(geom.image_size, device=device, dtype=torch.float32)

    # ── recon 1: FBP ──────────────────────────────────────────────────────
    with torch.no_grad():
        rec_fbp = proj.fbp(noisy_sino).clamp_min(0.0)

    # ── recon 2: TV-iterative ─────────────────────────────────────────────
    # Defaults aligned with solver_tv_iterative's typical search range.
    rec_tv = tv_reconstruction(proj, noisy_sino, rec_fbp,
                                lam=1e-3, iterations=80, lr=5e-3,
                                clip_max=dmax, device=device).detach()
    rec_tv = rec_tv.clamp_min(0.0)

    # ── recon 3: Wu 2015 ADMM ─────────────────────────────────────────────
    dense_proj = _build_dense_projector(geom, factor=2, device=device)
    with torch.no_grad():
        rec_wu = wu_2015_reconstruct(
            proj=proj, dense_proj=dense_proj, sino=noisy_sino,
            n_bands=4, n_outer=3, motion_range=2, motion_window=5,
            soft_thresh=2e-4,
        )

    # ── recon 4: dual-domain bilateral (untrained, init σ's) ─────────────
    from ddssl_ldct.models import TrainableBilateralFilter2d
    proj_dn = TrainableBilateralFilter2d(
        kernel_size=5, sigma_x=1.0, sigma_y=2.0, sigma_r=0.02).to(device)
    img_dn = TrainableBilateralFilter2d(
        kernel_size=7, sigma_x=1.5, sigma_y=1.5, sigma_r=0.02).to(device)
    with torch.no_grad():
        sino_dn = proj_dn(noisy_sino)
        rec_bf_init = proj.fbp(sino_dn).clamp_min(0.0)
        rec_bf = img_dn(rec_bf_init).clamp_min(0.0)

    # ── metrics vs truth (FOV-masked, double-clamp cal) ──────────────────
    panels = [
        ("FBP",                       rec_fbp),
        ("TV iterative",              rec_tv),
        ("Wu 2015 ADMM",              rec_wu),
        ("Dual-D bilateral (init)",   rec_bf),
    ]
    print(f"\n{'solver':<28} {'SSIM':>8} {'PSNR':>8} {'RMSE':>10}")
    cals = []
    for name, rec in panels:
        s, p, r, pc = cal_metrics(rec, phantom, dmax=dmax, fov=fov)
        cals.append((name, pc))
        print(f"{name:<28} {s:8.4f} {p:8.2f} {r:10.4e}")

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    truth_np = (phantom * fov)[0, 0].cpu().numpy()
    axes[0, 0].imshow(truth_np, cmap="gray", vmin=0, vmax=dmax)
    axes[0, 0].set_title(f"truth\nrange=[{truth_np.min():.4f}, {truth_np.max():.4f}]",
                          fontsize=10); axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "(no diff for truth)", ha="center", va="center",
                     transform=axes[1, 0].transAxes, fontsize=10)
    diff_lim = dmax / 4.0
    for ci, (name, pc) in enumerate(cals, start=1):
        rec_np = pc[0, 0].cpu().numpy()
        s, p, r, _ = cal_metrics(panels[ci-1][1], phantom, dmax=dmax, fov=fov)
        axes[0, ci].imshow(rec_np, cmap="gray", vmin=0, vmax=dmax)
        axes[0, ci].set_title(f"{name}\nSSIM={s:.3f}  PSNR={p:.2f} dB\nRMSE={r:.3e}",
                               fontsize=10); axes[0, ci].axis("off")
        diff_np = rec_np - truth_np
        axes[1, ci].imshow(diff_np, cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[1, ci].set_title(f"{name} − truth\n|err|max={float(np.abs(diff_np).max()):.4f}",
                               fontsize=10); axes[1, ci].axis("off")
    plt.suptitle(f"Demo-DL single-phantom 4-solver comparison. "
                 f"Display [0, {dmax}]. Diffs ±{diff_lim:.4f}. "
                 f"FOV-masked, intensity-calibrated.", fontsize=11, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "demo_dl_4solvers.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
