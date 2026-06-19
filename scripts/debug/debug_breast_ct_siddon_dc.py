"""Test the user's hypothesis: the cupping in our Siddon FBP comes from the
residual non-zero DC bin of the Kak-Slaney ramp filter when the sinogram is
not zero-padded before the FFT.

Ramp variants tested:
  (a) Kak-Slaney spatial, length N, no pad         (current)
  (b) Kak-Slaney spatial, length N, no pad + H[0]:=0 (force DC=0)
  (c) Kak-Slaney spatial, length 2N (zero-pad sino) (proper linear conv)
  (d) Kak-Slaney spatial, length 2N + H[0]:=0       (both fixes)

For each, run SiddonFanBeam.fbp on `val_sinograms[0]` and compare to
`val_fbp128[0]` (raw rel-L2, scale-matched rel-L2, calibrated SSIM/PSNR/RMSE
vs Sidky's FBP128 and vs truth).

If (b) closes most of the gap → DC truncation IS the cupping cause.

Outputs:
  /cluster/maier/Agent4CT/results/breast_debug/siddon_dc.png
"""
from __future__ import annotations
import sys, math
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
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=360/1024, sod=500.0, sdd=1000.0)
DISPLAY_MAX = 0.5
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


def kak_slaney_h(N: int, d: float) -> np.ndarray:
    h = np.zeros(N, dtype=np.float64)
    h[0] = 0.25 / (d * d)
    odd_v = -1.0 / (math.pi * math.pi * d * d)
    for i in range(1, N):
        if i < N / 2 and (i % 2) == 1:
            h[i] = odd_v / (i * i)
        elif i >= N / 2:
            tmp = N - i
            if (tmp % 2) == 1:
                h[i] = odd_v / (tmp * tmp)
    return h


def filter_sino_variant(sino: torch.Tensor, det_spacing: float, *,
                         zero_pad: bool, dc_zero: bool) -> torch.Tensor:
    N = sino.shape[-1]
    device, dtype = sino.device, sino.dtype
    if zero_pad:
        h = kak_slaney_h(2 * N, det_spacing)
        pad = torch.zeros(sino.shape[:-1] + (N,), device=device, dtype=dtype)
        x = torch.cat([sino, pad], dim=-1)
    else:
        h = kak_slaney_h(N, det_spacing)
        x = sino
    f_np = np.real(np.fft.fft(h)).astype(np.float64)
    print(f"    ramp N={len(h)}  DFT[0]={f_np[0]:+.4e}  peak={float(np.max(np.abs(f_np))):.4f}  "
          f"DC/peak={f_np[0]/np.max(np.abs(f_np)):+.4e}")
    if dc_zero:
        f_np[0] = 0.0
        print(f"    after dc_zero: DFT[0]={f_np[0]:+.4e}")
    f = torch.as_tensor(f_np.astype(np.float32), device=device, dtype=dtype)
    spec = torch.fft.fft(x, dim=-1, norm="ortho")
    spec = spec * f
    y = torch.fft.ifft(spec, dim=-1, norm="ortho").real
    if zero_pad:
        y = y[..., :N]
    return y


def cal_metrics(pred, truth, dmax=DISPLAY_MAX):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    return (
        float(ssim(pc, truth, data_range=dmax).cpu()),
        float(psnr(pc, truth, data_range=dmax).cpu()),
        float(((pc - truth) ** 2).mean().sqrt().cpu()),
        pc,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth_np = f["image"][:1]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_np = f["sino"][:1]
    with h5py.File(data / "val_fbp128.h5", "r") as f: fbp_np = f["image"][:1]

    truth = torch.from_numpy(truth_np).float().to(device).unsqueeze(1)
    sino_sidky = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)
    fbp_sidky = torch.from_numpy(fbp_np).float().to(device).unsqueeze(1)
    f_k = fbp_sidky[0, 0].cpu().numpy()
    t_np = truth_np[0]

    geom = FanBeamGeometry(**GEOM)
    proj = SiddonFanBeamProjector(geom, ray_batch=8192, length_unit_scale=0.1).to(device)
    # scaled det_spacing matches the projector's
    d_scaled = proj._det_spacing_scaled
    print(f"d_scaled (cm) = {d_scaled:.5f}")

    # Also need the FBP angle weight applied (Δβ/2 for 2π scan)
    ang_range = float(geom.angle_end - geom.angle_start)
    fbp_weight = (ang_range / geom.n_angles) / 2.0 if abs(ang_range - 2 * math.pi) < 1e-3 \
                 else (ang_range / geom.n_angles)
    print(f"fbp angle weight = {fbp_weight:.6e}")

    variants = [
        ("(a) plain Kak-Slaney N=1024",  dict(zero_pad=False, dc_zero=False)),
        ("(b) Kak-Slaney N=1024 + H[0]=0", dict(zero_pad=False, dc_zero=True)),
        ("(c) Kak-Slaney pad to 2N",     dict(zero_pad=True, dc_zero=False)),
        ("(d) Kak-Slaney pad to 2N + H[0]=0", dict(zero_pad=True, dc_zero=True)),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    print(f"\n{'variant':<40} {'L2 raw':>10} {'L2_k':>10} {'k':>6}  {'SSIMvFBP':>9}  {'SSIMvTruth':>10}")
    for ci, (label, kw) in enumerate(variants):
        print(f"\n  {label}:")
        with torch.no_grad():
            filt = filter_sino_variant(sino_sidky, d_scaled, **kw)
            recon = proj.back_project(filt) * fbp_weight
        f_o = recon[0, 0].cpu().numpy()
        raw_l2 = float(np.linalg.norm(f_o - f_k) / np.linalg.norm(f_k))
        k = float((f_o * f_k).sum() / max((f_o * f_o).sum(), 1e-12))
        k_l2 = float(np.linalg.norm(k * f_o - f_k) / np.linalg.norm(f_k))
        ss_f, _, _, fbp_cal = cal_metrics(recon, fbp_sidky)
        ss_t, ps_t, rm_t, fbp_cal_t = cal_metrics(recon, truth)
        print(f"  {label:<40} {raw_l2:10.4e} {k_l2:10.4e} {k:6.3f}  {ss_f:9.4f}  {ss_t:10.4f}")

        # Plot grayscale recon, diff vs FBP128, and cal-recon vs truth
        axes[0, ci].imshow(f_o, cmap="gray", vmin=0, vmax=DISPLAY_MAX)
        axes[0, ci].set_title(f"{label}\nrange=[{f_o.min():.3f}, {f_o.max():.3f}]\n"
                              f"calSSIMvFBP={ss_f:.4f}", fontsize=10)
        axes[0, ci].axis("off")
        diff = f_o - f_k
        lim = float(np.percentile(np.abs(diff), 99))
        axes[1, ci].imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
        axes[1, ci].set_title(f"OURS - sidky_fbp128\nrelL2={raw_l2:.3e}  |err|99={lim:.3f}", fontsize=10)
        axes[1, ci].axis("off")
        cal_diff = fbp_cal_t[0, 0].cpu().numpy() - t_np
        lim_t = DISPLAY_MAX / 4
        axes[2, ci].imshow(cal_diff, cmap="bwr", vmin=-lim_t, vmax=lim_t)
        axes[2, ci].set_title(f"cal-OURS - truth\ncalSSIM={ss_t:.4f}", fontsize=10)
        axes[2, ci].axis("off")

    plt.suptitle("Ramp-filter DC variants. (b) and (d) force H[0]=0 explicitly. "
                 "If the cupping is from the Kak-Slaney DC truncation, (b) and (d) should be visibly cleaner.",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "siddon_dc.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
