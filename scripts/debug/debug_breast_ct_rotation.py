"""Sweep angle/detector conventions to fix the breast-CT 90° rotation.
det_spacing=0.6429 already nailed the scale (corr 0.265 -> 0.889 vs FBP128).
This script holds that geometry and tries:
  - Reversing the sino along the angle axis (CW vs CCW gantry)
  - Reversing the sino along the detector axis (channel mirroring)
  - 90° / 180° / 270° image-rotation of the OUTPUT (purely cosmetic
    relabeling, but indicates which angle origin Sidky chose)
The goal is corr >= 0.99 against FBP128 on at least one combination.
Saves the comparison figure to a shared /cluster path so we can pull it.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin   # noqa: F401
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector


GEOM = dict(image_size=512, pixel_spacing=0.7,
            n_angles=128, n_det=1024, det_spacing=0.6429,
            sod=595.0, sdd=1085.6)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f:
        truth_np = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f:
        sino_np = f["sino"][:4]
    with h5py.File(data / "val_fbp128.h5", "r") as f:
        fbp_ref_np = f["image"][:4]

    sino_t = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)   # (4,1,128,1024)
    geom = FanBeamGeometry(**GEOM)
    proj = PyronnFanBeamProjector(geom).to(device)

    # Each variant: (sino transform, post-recon image transform).
    # sino_xform takes (B,1,A,D); recon_xform takes (B,1,H,W) tensor.
    def _flip(t, dim):
        return torch.flip(t, dims=(dim,))

    variants = {
        "baseline (det_spacing=0.6429)":
            (lambda s: s, lambda r: r),
        "flip angle axis (CW)":
            (lambda s: _flip(s, -2), lambda r: r),
        "flip detector axis":
            (lambda s: _flip(s, -1), lambda r: r),
        "flip both axes":
            (lambda s: _flip(_flip(s, -2), -1), lambda r: r),
        "image rot +90":
            (lambda s: s, lambda r: torch.rot90(r, k=1, dims=(-2, -1))),
        "image rot -90":
            (lambda s: s, lambda r: torch.rot90(r, k=-1, dims=(-2, -1))),
        "image rot 180":
            (lambda s: s, lambda r: torch.rot90(r, k=2, dims=(-2, -1))),
        "flip angle + rot +90":
            (lambda s: _flip(s, -2), lambda r: torch.rot90(r, k=1, dims=(-2, -1))),
        "flip det + rot -90":
            (lambda s: _flip(s, -1), lambda r: torch.rot90(r, k=-1, dims=(-2, -1))),
        "flip det + rot +90":
            (lambda s: _flip(s, -1), lambda r: torch.rot90(r, k=1, dims=(-2, -1))),
        "transpose image":
            (lambda s: s, lambda r: r.transpose(-2, -1)),
        "transpose then flip H":
            (lambda s: s, lambda r: _flip(r.transpose(-2, -1), -2)),
    }

    n = len(variants) + 1
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.0 * n))
    vmin, vmax = 0.0, float(np.percentile(truth_np, 99.5))
    diff_lim = vmax / 2

    # Row 0: truth and FBP128 reference (cases 0..3 + a blank)
    for j in range(4):
        axes[0, j].imshow(truth_np[j], cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"truth #{j}", fontsize=9); axes[0, j].axis("off")
    axes[0, 4].imshow(fbp_ref_np[0], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 4].set_title("FBP128 #0 (ref)", fontsize=9); axes[0, 4].axis("off")

    best = (None, -np.inf)
    for i, (label, (sx, rx)) in enumerate(variants.items(), start=1):
        s = sx(sino_t)
        with torch.no_grad():
            recon = rx(proj.fbp(s).clamp_min(0.0))
        recon_np = recon[:, 0].detach().cpu().numpy()
        corrs = [np.corrcoef(recon_np[j].ravel(), fbp_ref_np[j].ravel())[0, 1]
                 for j in range(4)]
        rmses = [float(np.sqrt(((recon_np[j] - fbp_ref_np[j]) ** 2).mean()))
                 for j in range(4)]
        mean_corr = float(np.mean(corrs))
        print(f"{label:45}  corr={mean_corr:.4f}  rmse={np.mean(rmses):.4f}")
        if mean_corr > best[1]:
            best = (label, mean_corr)
        for j in range(4):
            axes[i, j].imshow(recon_np[j], cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, j].set_title(f"{label}\n#{j} corr={corrs[j]:.2f}", fontsize=7)
            axes[i, j].axis("off")
        # Last column: diff (recon - FBP128) for case 0
        diff = recon_np[0] - fbp_ref_np[0]
        axes[i, 4].imshow(diff, cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[i, 4].set_title(f"diff #0", fontsize=8); axes[i, 4].axis("off")

    plt.tight_layout()
    out_path = out_dir / "rotation_sweep.png"
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[best] {best[0]}  corr={best[1]:.4f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
