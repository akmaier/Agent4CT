"""Visualize FBP with the corrected Sidky geometry (sod=500, sdd=1000,
pixel_spacing=det_spacing=180/512 ≈ 0.3516). Sweep sinogram/image axis
flips at this geometry to find the angle-direction convention.

Saves a single side-by-side figure for visual inspection.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin  # noqa
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector

# Sidky 2022 paper values (Section II.B)
GEOM = dict(
    image_size=512,
    pixel_spacing=180.0/512.0,    # 0.3516 mm/pixel; FOV 180mm
    n_angles=128,
    n_det=1024,
    det_spacing=360.0/1024.0,     # 0.3516 mm; chord at iso = 180mm = FOV ✓
    sod=500.0,
    sdd=1000.0,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:4]
    with h5py.File(data / "val_fbp128.h5", "r") as f: ref = f["image"][:4]

    s = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    geom = FanBeamGeometry(**GEOM); proj = PyronnFanBeamProjector(geom).to(device)

    def fbp(sino_in):
        with torch.no_grad():
            return proj.fbp(sino_in)[:, 0].cpu().numpy()

    # 5 sino variants × 4 cases.
    def flp(t, dim): return torch.flip(t, dims=(dim,))
    variants = {
        "baseline (Sidky geom)":           (lambda x: x, lambda r: r),
        "flip angle axis":                  (lambda x: flp(x, -2), lambda r: r),
        "flip detector axis":               (lambda x: flp(x, -1), lambda r: r),
        "rot +90 (CCW)":                    (lambda x: x, lambda r: np.rot90(r, k=1, axes=(-2,-1))),
        "rot -90 (CW)":                     (lambda x: x, lambda r: np.rot90(r, k=-1, axes=(-2,-1))),
        "transpose":                        (lambda x: x, lambda r: r.swapaxes(-2,-1)),
        "transpose+fliph":                  (lambda x: x, lambda r: np.flip(r.swapaxes(-2,-1), axis=-1)),
        "flip ang + rot -90":               (lambda x: flp(x, -2), lambda r: np.rot90(r, k=-1, axes=(-2,-1))),
    }

    fig, axes = plt.subplots(len(variants) + 1, 5, figsize=(20, 3.0*(len(variants)+1)))
    # Row 0: 4× truth + 1× FBP128
    for j in range(4):
        axes[0, j].imshow(truth[j], cmap="gray", vmin=0, vmax=float(np.percentile(truth[j], 99.5)))
        axes[0, j].set_title(f"truth #{j}", fontsize=10); axes[0, j].axis("off")
    axes[0, 4].imshow(ref[0], cmap="gray", vmin=0, vmax=float(np.percentile(ref[0], 99.5)))
    axes[0, 4].set_title("FBP128 (Sidky ref) #0", fontsize=10); axes[0, 4].axis("off")

    best = (None, -np.inf)
    for i, (label, (sxf, ixf)) in enumerate(variants.items(), start=1):
        s_v = sxf(s)
        recon = fbp(s_v)
        recon = ixf(recon) if not isinstance(ixf(recon), torch.Tensor) else ixf(recon)
        corrs = []
        for j in range(4):
            corrs.append(np.corrcoef(recon[j].ravel(), ref[j].ravel())[0, 1])
        mc = float(np.mean(corrs))
        print(f"{label:35} corr={mc:.4f}  range=[{recon.min():.4f},{recon.max():.4f}]")
        if mc > best[1]: best = (label, mc)
        for j in range(4):
            lo = float(np.percentile(recon[j], 1)); hi = float(np.percentile(recon[j], 99.5))
            axes[i, j].imshow(recon[j], cmap="gray", vmin=lo, vmax=hi)
            axes[i, j].set_title(f"{label}\n#{j} corr={corrs[j]:.2f}\n[{recon[j].min():.3f},{recon[j].max():.3f}]", fontsize=8)
            axes[i, j].axis("off")
        # 5th column: diff against FBP128 case 0 (auto-scaled)
        diff = recon[0] - ref[0]
        dl = float(np.percentile(np.abs(diff), 99))
        axes[i, 4].imshow(diff, cmap="bwr", vmin=-dl, vmax=dl)
        axes[i, 4].set_title("(recon - FBP128) #0", fontsize=9); axes[i, 4].axis("off")

    plt.tight_layout()
    out = Path("/cluster/maier/Agent4CT/results/breast_debug/sidky_geom.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[best] {best[0]}  corr={best[1]:.4f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
