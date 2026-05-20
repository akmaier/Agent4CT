"""Sweep det_spacing to find the value that makes Siddon's forward projection
of val_truth best match the released val_sinograms. The Sidky paper specifies
image dims, n_det, SOD, SDD but NOT the detector pitch — our 0.3516 mm value
is a (possibly wrong) FOV-coverage assumption.

Sweep det_spacing ∈ {0.3516 × c} for c in a fine grid bracketing ±5%.

For each candidate:
  - build SiddonFanBeamProjector with that det_spacing.
  - forward-project val_truth[0:1].
  - compare to val_sinograms[0:1]: raw rel-L2 + best linear scale k + rel-L2(k).
  - also FBP val_sinograms[0:1] and compare to val_fbp128[0:1].

Best forward-match c is Sidky's true det_spacing (forward is the most direct
test — pixel-position drift in the back-projection just smears slightly, but
the forward fingerprint is sharp).

Prints numerical tables only — no rotation sweep, no image conclusions.
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
import h5py

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.siddon_projector import SiddonFanBeamProjector
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate

BASE_DET_SPACING = 360.0 / 1024.0                 # 0.3516 mm, current value
PIX = 180.0 / 512.0                                # 0.3516 mm, fixed (= paper FOV/512)
GEOM_BASE = dict(image_size=512, pixel_spacing=PIX, n_angles=128,
                 n_det=1024, sod=500.0, sdd=1000.0)
DISPLAY_MAX = 0.5


def cal_metrics(pred, truth, dmax=DISPLAY_MAX):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    return (
        float(ssim(pc, truth, data_range=dmax).cpu()),
        float(psnr(pc, truth, data_range=dmax).cpu()),
        float(((pc - truth) ** 2).mean().sqrt().cpu()),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth_np = f["image"][:1]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_np = f["sino"][:1]
    with h5py.File(data / "val_fbp128.h5", "r") as f: fbp_np = f["image"][:1]

    truth = torch.from_numpy(truth_np).float().to(device).unsqueeze(1)
    sino_sky = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)
    fbp_sky = torch.from_numpy(fbp_np).float().to(device).unsqueeze(1)
    s_k = sino_sky[0, 0].cpu().numpy()
    f_k = fbp_sky[0, 0].cpu().numpy()

    scale_factors = [1.014, 1.016, 1.017, 1.018, 1.019, 1.020, 1.021, 1.022, 1.023, 1.024]

    print(f"\nbase det_spacing = {BASE_DET_SPACING:.6f} mm (= pix_spacing = {PIX:.6f})")
    print(f"\nSweep: det_spacing = base · c\n")
    print(f"{'c':>6} {'det_sp(mm)':>11} | {'L2(fwd)raw':>11} {'k_fwd':>7} {'L2_k(fwd)':>11} |"
          f" {'L2(fbp)raw':>11} {'k_fbp':>7} {'L2_k(fbp)':>11} {'calSSIMvFBP':>12}")
    print("-" * 130)
    fwd_l2k, fwd_l2_raw = [], []
    for c in scale_factors:
        det_sp = BASE_DET_SPACING * c
        geom = FanBeamGeometry(**{**GEOM_BASE, "det_spacing": det_sp})
        proj = SiddonFanBeamProjector(geom, ray_batch=8192, length_unit_scale=0.1).to(device)

        # Forward
        with torch.no_grad():
            sino_ours = proj.forward_project(truth)
        s_o = sino_ours[0, 0].cpu().numpy()
        L2_raw = float(np.linalg.norm(s_o - s_k) / np.linalg.norm(s_k))
        k_fwd = float((s_o * s_k).sum() / max((s_o * s_o).sum(), 1e-12))
        L2_k = float(np.linalg.norm(k_fwd * s_o - s_k) / np.linalg.norm(s_k))
        fwd_l2_raw.append(L2_raw); fwd_l2k.append(L2_k)

        # FBP
        with torch.no_grad():
            fbp_ours = proj.fbp(sino_sky)
        f_o = fbp_ours[0, 0].cpu().numpy()
        L2_fbp_raw = float(np.linalg.norm(f_o - f_k) / np.linalg.norm(f_k))
        k_fbp = float((f_o * f_k).sum() / max((f_o * f_o).sum(), 1e-12))
        L2_fbp_k = float(np.linalg.norm(k_fbp * f_o - f_k) / np.linalg.norm(f_k))
        ss_fbp, _, _ = cal_metrics(fbp_ours, fbp_sky)

        print(f"{c:>6.3f} {det_sp:>11.5f} | {L2_raw:>11.4e} {k_fwd:>7.4f} {L2_k:>11.4e} |"
              f" {L2_fbp_raw:>11.4e} {k_fbp:>7.4f} {L2_fbp_k:>11.4e} {ss_fbp:>12.4f}")

    best_idx = int(np.argmin(fwd_l2k))
    print(f"\nbest forward L2(k):  c = {scale_factors[best_idx]:.3f}  →  det_spacing = "
          f"{BASE_DET_SPACING * scale_factors[best_idx]:.5f} mm  →  L2_k = {fwd_l2k[best_idx]:.4e}")


if __name__ == "__main__":
    main()
