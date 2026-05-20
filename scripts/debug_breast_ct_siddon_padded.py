"""Image test of SiddonFanBeamProjector with the *proper* zero-padded ram-lak:
ramp built at length M = 2·N_det, sinogram zero-padded N → 2N along the
detector axis, FFT-multiply-IFFT, then truncate back to N.

Produces one PNG, no sweeps, no rotation changes, no automated conclusions.
Layout — 4 rows, one per breast case, 5 columns:

  truth | sidky_fbp128 | OUR FBP (intensity-cal'd) | OUR − sidky_fbp128 | cal_OUR − truth

For the user to look at.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.siddon_projector import SiddonFanBeamProjector
from ddssl_ldct.metrics import intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=360/1024, sod=500.0, sdd=1000.0)
DISPLAY_MAX = 0.5
N_CASES = 4
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth_np = f["image"][:N_CASES]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_np = f["sino"][:N_CASES]
    with h5py.File(data / "val_fbp128.h5", "r") as f: fbp_np = f["image"][:N_CASES]

    truth = torch.from_numpy(truth_np).float().to(device).unsqueeze(1)
    sino = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)

    geom = FanBeamGeometry(**GEOM)
    proj = SiddonFanBeamProjector(geom, ray_batch=8192, length_unit_scale=0.1).to(device)
    with torch.no_grad():
        fbp_ours = proj.fbp(sino)
        fbp_ours_cal = intensity_calibrate(fbp_ours.clamp_min(0.0), truth, display_max=DISPLAY_MAX)

    # ── ramp-filter "by the book" sanity check ─────────────────────────────
    # Kak-Slaney (1988) §3.3.3 textbook values, at det_spacing_scaled τ=0.03516 cm:
    #   spatial: h(0) = 1/(4τ²) = 202.27   (cm⁻²)
    #            h(τ·n odd) = -1/(π²·n²·τ²)  →  h(τ) = -81.94, h(3τ) = -9.10, h(5τ) = -3.28
    #   freq:    H(ω) = |ω| sampled at DFT bins ω_k = 2π·k/(M·τ) for k=0..M-1
    #            with H[0] ≈ 0 (truncation residual 2/(π²·M·τ²))
    #            and peak at k=M/2 of |ω|_max = 1/(2τ)  →  ~14.22 cm⁻¹ (in ω/2π = cycles/cm)
    d = proj._det_spacing_scaled
    M = 2 * geom.n_det
    f = proj._ramlak(M, device, torch.float32).cpu().numpy()
    print(f"\nramlak by-the-book check (Kak-Slaney 1988, eq 3.30, M={M}, τ={d:.5f} cm):")
    print(f"  h(0) expected = {1.0/(4*d*d):.4f} cm⁻²   (our DFT-back-derived: h(0) = {f.mean():.4f})")
    print(f"  H[0]  = {f[0]:+.4e}   (should be ~0; residual = 2/(π²·M·τ²) = {2.0/(math.pi**2 * M * d * d):+.4e})")
    print(f"  H[M/4] = {f[M//4]:+.4e}   (≈ 1/(4τ) = {1.0/(4*d):.4f})")
    print(f"  H[M/2] = {f[M//2]:+.4e}   (≈ 1/(2τ) = {1.0/(2*d):.4f})  [Nyquist peak]")
    print(f"  H[M-1] = {f[M-1]:+.4e}   (= H[1] by symmetry: {f[1]:+.4e})")
    print(f"  |H| peak = {float(np.max(np.abs(f))):.4f}")

    # ── per-case raw scale comparison ───────────────────────────────────────
    print(f"\nraw scale comparison (OUR FBP vs sidky FBP128):")
    print(f"{'case':>4}  {'OUR min':>10}  {'OUR max':>10}  {'OUR mean':>10}  "
          f"{'sky min':>10}  {'sky max':>10}  {'sky mean':>10}  {'mean ratio':>11}  {'max ratio':>10}")
    for i in range(N_CASES):
        o = fbp_ours[i, 0].cpu().numpy()
        s = fbp_np[i]
        print(f"{i:>4}  {o.min():10.4f}  {o.max():10.4f}  {o.mean():10.4f}  "
              f"{s.min():10.4f}  {s.max():10.4f}  {s.mean():10.4f}  "
              f"{o.mean()/s.mean():11.4f}  {o.max()/s.max():10.4f}")

    fig, axes = plt.subplots(N_CASES, 5, figsize=(22, 4.6 * N_CASES))
    diff_lim = DISPLAY_MAX / 4.0
    for r in range(N_CASES):
        t = truth_np[r]
        sky = fbp_np[r]
        ours = fbp_ours_cal[r, 0].cpu().numpy()
        ours_raw = fbp_ours[r, 0].cpu().numpy()

        axes[r, 0].imshow(t, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[r, 0].set_title(f"truth #{r}\nrange=[{t.min():.3f}, {t.max():.3f}]", fontsize=10)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(sky, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[r, 1].set_title(f"sidky FBP128\nrange=[{sky.min():.3f}, {sky.max():.3f}]", fontsize=10)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(ours, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[r, 2].set_title(f"OUR Siddon FBP (cal, zero-pad 2N)\nraw range=[{ours_raw.min():.3f}, {ours_raw.max():.3f}]",
                              fontsize=10)
        axes[r, 2].axis("off")

        diff_sky = ours - sky
        axes[r, 3].imshow(diff_sky, cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[r, 3].set_title(f"cal_OUR − sidky_fbp128\n|err|max={float(np.abs(diff_sky).max()):.3f}", fontsize=10)
        axes[r, 3].axis("off")

        diff_t = ours - t
        axes[r, 4].imshow(diff_t, cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[r, 4].set_title(f"cal_OUR − truth\n|err|max={float(np.abs(diff_t).max()):.3f}", fontsize=10)
        axes[r, 4].axis("off")

    plt.suptitle(f"SiddonFanBeam zero-padded (N→2N) ram-lak. All gray panels at vmin=0, vmax={DISPLAY_MAX}. "
                 f"All diffs at ±{diff_lim:.3f}.",
                 fontsize=11, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "siddon_padded.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
