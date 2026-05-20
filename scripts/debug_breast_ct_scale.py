"""Why is our FBP value range so tiny? Print everything we have:
  - sino value range, sum, mean, std
  - our FBP raw range (before clamp)
  - FBP128 raw range
  - Ratio between our FBP and FBP128 in matching ROI (foreground only)
This tells us whether the scale-off is a constant factor (filter
normalisation), a det_spacing factor (geometry), or something else.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin   # noqa
import h5py
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:4]
    with h5py.File(data / "val_fbp128.h5", "r") as f: ref = f["image"][:4]
    print(f"truth   range=[{truth.min():.5f}, {truth.max():.5f}] mean={truth.mean():.5f}")
    print(f"sino    range=[{sino.min():.5f}, {sino.max():.5f}] mean={sino.mean():.5f}  sum/N={sino.sum()/sino.size:.5f}")
    print(f"FBP128  range=[{ref.min():.5f}, {ref.max():.5f}] mean={ref.mean():.5f}")
    print()

    s = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    # Test a range of det_spacing values and report our FBP scale.
    for dsp in [0.6429, 1.0, 1.2858, 2.0, 0.32, 0.5]:
        geom_kw = dict(image_size=512, pixel_spacing=0.7, n_angles=128,
                        n_det=1024, det_spacing=dsp,
                        sod=595.0, sdd=1085.6)
        g = FanBeamGeometry(**geom_kw); p = PyronnFanBeamProjector(g).to(device)
        with torch.no_grad():
            r = p.fbp(s)[:, 0].cpu().numpy()
        # In the foreground (truth > 0.1), compute mean ratio = ref/our
        for j in [0]:
            mask = truth[j] > 0.1
            if mask.any():
                ref_fg = ref[j][mask].mean()
                our_fg = r[j][mask].mean()
                ratio = ref_fg / max(our_fg, 1e-12)
            else:
                ratio = float("nan"); our_fg = 0; ref_fg = 0
        print(f"det_spacing={dsp:>7.4f}: our_range=[{r.min():.5f}, {r.max():.5f}] our_fg_mean={our_fg:.5f}  ref_fg_mean={ref_fg:.5f}  ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
