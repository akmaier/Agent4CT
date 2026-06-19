"""Diagnose the breast_ct fan-beam geometry by comparing our FBP against
Sidky's reference FBP128 (which IS the dataset's own FBP of the staged
sinograms). If our geometry is correct, our FBP should match FBP128
pixel-for-pixel (up to filter/clipping differences). If our FBP is
rotated / zoomed / shifted, the diff reveals exactly which geometry
parameter is off.

Outputs:
  /tmp/_breast_debug/comparison.png — 4 cases × 4 cols:
    truth / FBP128 (reference) / OUR FBP / OUR FBP - FBP128 (diff)

Usage on cluster (CPU/GPU node):
  AGENT4CT_DATASET=breast_ct python scripts/debug_breast_ct_geometry.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import json
import numpy as np
import torch
import hdf5plugin   # register lz4 filter BEFORE h5py opens files
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector


# Current breast_ct geometry (as set in ddssl_ldct/staged_dataset.py).
CURRENT = dict(
    image_size=512, pixel_spacing=0.7,
    n_angles=128, n_det=1024, det_spacing=1.2858,
    sod=595.0, sdd=1085.6,
)


def fbp_one(geom_kw, sino_t: torch.Tensor, device="cuda") -> torch.Tensor:
    """Run our PyronnFanBeamProjector.fbp on a single (1, 1, A, D) sinogram."""
    geom = FanBeamGeometry(**geom_kw)
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        recon = proj.fbp(sino_t).clamp_min(0.0)
    return recon


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("/tmp/_breast_debug")
    out_dir.mkdir(exist_ok=True)

    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f:
        truth_np = f["image"][:4]                         # (4, 512, 512)
    with h5py.File(data / "val_sinograms.h5", "r") as f:
        sino_np = f["sino"][:4]                            # (4, 128, 1024)
    with h5py.File(data / "val_fbp128.h5", "r") as f:
        fbp_ref_np = f["image"][:4]                        # (4, 512, 512) — Sidky's own FBP

    print(f"truth   shape={truth_np.shape} range=[{truth_np.min():.4f},{truth_np.max():.4f}]")
    print(f"sino    shape={sino_np.shape} range=[{sino_np.min():.4f},{sino_np.max():.4f}]")
    print(f"FBP128  shape={fbp_ref_np.shape} range=[{fbp_ref_np.min():.4f},{fbp_ref_np.max():.4f}]")

    sino_t = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)   # (4, 1, 128, 1024)

    # ============================================================
    # Sweep candidate geometries to find the one that matches FBP128.
    # ============================================================
    # The user reported: 90° CW rotation + ~2× zoom-in. The 2× zoom
    # suggests our image FOV is half what it should be — either
    # pixel_spacing is 2× too small for the actual chord, OR the chord
    # is 2× too wide because n_det/det_spacing/sod/sdd ratios are off.
    #
    # Sidky 2022 Med Phys lays out the breast challenge scan geometry:
    # in the paper Table 1 the values typically reported are different
    # from the Mayo SDD/SOD we copied by mistake. Without the paper at
    # hand, sweep a few plausible candidates and pick whichever matches
    # FBP128 best in SSIM / pixel difference.
    candidates = {
        "current (Mayo-copied)": dict(CURRENT),
        "px=1.4 (2x larger)":    {**CURRENT, "pixel_spacing": 1.4},
        "px=0.35 (2x smaller)":  {**CURRENT, "pixel_spacing": 0.35},
        "det_spacing=0.6429 (half)": {**CURRENT, "det_spacing": 0.6429},
        "det_spacing=2.5716 (2x)":   {**CURRENT, "det_spacing": 2.5716},
        # Geometry from Sidky's challenge code (best-guess values):
        # det chord at iso = n_det * det_spacing * (sod/sdd) should equal
        # image FOV = image_size * pixel_spacing for the recon to fit.
        # current: 1024 * 1.2858 * (595/1085.6) = 721 mm; image FOV = 358 mm
        # ratio 2.0× -> matches user's "zoomed 2x"
        "sod/sdd 2x ratio fix":  {**CURRENT, "sod": 542.8, "sdd": 1085.6},   # changes ratio
        "matching FOV (det_spacing=0.6429)": {
            **CURRENT, "det_spacing": 0.6429,
        },
    }

    fig, axes = plt.subplots(len(candidates) + 1, 4, figsize=(14, 3.0 * (len(candidates)+1)))
    # Reference row 0: truth / FBP128 / blank / blank
    vmin, vmax = 0.0, float(np.percentile(truth_np, 99.5))
    diff_lim = vmax / 2
    for j in range(4):
        axes[0, j].imshow(truth_np[j], cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"truth #{j}", fontsize=9); axes[0, j].axis("off")

    for i, (label, geom_kw) in enumerate(candidates.items(), start=1):
        try:
            recon = fbp_one(geom_kw, sino_t, device=device)
            recon_np = recon[:, 0].detach().cpu().numpy()
        except Exception as e:
            print(f"FAILED {label}: {e}")
            recon_np = np.zeros_like(truth_np)
        # Compute scalar fit metrics against FBP128
        diffs = (recon_np - fbp_ref_np).reshape(4, -1)
        rmse_per = np.sqrt((diffs**2).mean(axis=1))
        corr_per = []
        for j in range(4):
            a = recon_np[j].ravel(); b = fbp_ref_np[j].ravel()
            corr_per.append(np.corrcoef(a, b)[0, 1])
        print(f"\n=== {label} ===")
        print(f"  geom = {geom_kw}")
        print(f"  RMSE vs FBP128 = {rmse_per.mean():.4f} ± {rmse_per.std():.4f}")
        print(f"  CORR vs FBP128 = {np.mean(corr_per):.4f}")
        for j in range(4):
            axes[i, j].imshow(recon_np[j], cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, j].set_title(
                f"{label}\n"
                f"#{j}: corr={corr_per[j]:.2f} rmse={rmse_per[j]:.3f}",
                fontsize=8)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
