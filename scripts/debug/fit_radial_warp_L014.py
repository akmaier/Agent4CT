#!/usr/bin/env python -u
"""Test the curved→flat detector rebinning for a radial 'fisheye' effect
by fitting a polynomial radial scale on the L014 peak FBP that best
matches the truth.

The reconstructed image is warped around iso (which sits at the image
centre for our PYRO-NN FBP) via:

    r_new = r · (1 + c1·(r/rmax)² + c2·(r/rmax)⁴ + c3·(r/rmax)⁶)

with rmax = half the image diagonal (so r/rmax ∈ [0, 1]). The warp is
applied to FBP_cal (intensity-calibrated FBP); the calibration is
re-run inside the loss so the comparison stays apples-to-apples with
the validator metric. L2 + small smoothness regularisation on the
warp displacement field (so that the polynomial isn't pushed to
extremes by image noise).

If the fitted c1, c2, c3 are not all ≈ 0 AND the table-banding pattern
is meaningfully reduced in the post-warp diff, then the
curved-to-flat rebin has a residual fisheye that we should fix in
ddssl_ldct/helix2fan.py.

Outputs:
  L014_radial_warp_fit.png     — 4-panel (truth, FBP before, FBP warped, diff after)
  L014_radial_warp_diff.png    — diff before vs diff after, same colour scale
  L014_radial_warp_profile.png — implied radial displacement curve (mm vs r)

Usage:  python -u scripts/fit_radial_warp_L014.py
"""
from __future__ import annotations

import math
import sys
import json
from pathlib import Path

import h5py
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import (
    evaluate_calibrated, intensity_calibrate,
    ssim as ssim_fn, psnr as psnr_fn,
)


def _list_truth(raw_dir: Path):
    SOP_CT = "1.2.840.10008.5.1.4.1.1.2"
    truth_files = []
    for series_dir in sorted(raw_dir.iterdir()):
        sample = next(series_dir.iterdir(), None)
        if sample is None: continue
        try:
            head = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception: continue
        if getattr(head, 'SOPClassUID', '') != SOP_CT: continue
        desc = getattr(head, 'SeriesDescription', '').lower()
        if 'full' not in desc or 'image' not in desc: continue
        for fp in series_dir.iterdir():
            try:
                m = pydicom.dcmread(str(fp), stop_before_pixels=True)
                z = float(m.ImagePositionPatient[2])
                truth_files.append((z, fp))
            except Exception:
                continue
        break
    truth_files.sort()
    return truth_files


def _mu(fp: Path):
    ds = pydicom.dcmread(str(fp))
    hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
          + float(ds.RescaleIntercept))
    return 0.02 * (1.0 + hu / 1000.0), ds


def build_fbp_and_truth(target_pZ: float = -254.50,
                         filter_name: str = "hann"):
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    nu, rotview, nz = int(geom_json['nu']), int(geom_json['rotview']), int(geom_json['nz_rebinned'])
    du = float(geom_json['du'])
    dv = float(geom_json.get('dv_rebinned', 1.0))
    z_start_src = float(geom_json['z_start'])
    angle_start = float(geom_json['angle_start_corrected'])
    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu, ds = _mu(fp)
    pixel_sp = float(ds.PixelSpacing[0])
    slice_thk = float(ds.SliceThickness)

    slab_lo, slab_hi = -pZ - slice_thk/2.0, -pZ + slice_thk/2.0
    j_lo = max(0, int(math.floor((slab_lo - z_start_src) / dv - 0.5)))
    j_hi = min(nz - 1, int(math.ceil((slab_hi - z_start_src) / dv + 0.5)))
    weights = {}
    for j in range(j_lo, j_hi + 1):
        z_j = z_start_src + j * dv
        bin_lo, bin_hi = z_j - dv/2.0, z_j + dv/2.0
        ov = max(0.0, min(bin_hi, slab_hi) - max(bin_lo, slab_lo))
        if ov > 0:
            weights[j] = ov / slice_thk

    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start + 2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")
    fbp_slab = np.zeros_like(truth_mu, dtype=np.float64)
    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        for j, w in weights.items():
            s = np.ascontiguousarray(np.flip(np.asarray(f["sino"][:, :, j],
                                                         dtype=np.float32), axis=-1))
            t = torch.from_numpy(s).to("cuda").float()[None, None]
            out = proj.fbp(t, filter_name=filter_name).detach()[0, 0].cpu().numpy()
            fbp_slab += w * np.fliplr(np.flipud(out))
    fbp_slab = np.clip(fbp_slab.astype(np.float32), 0.0, None)

    fbp_t = torch.from_numpy(fbp_slab).to("cuda").float()[None, None]
    truth_t = torch.from_numpy(truth_mu).to("cuda").float()[None, None]
    m = evaluate_calibrated(fbp_t, truth_t, baseline=fbp_t,
                             display_min=0.0, display_max=0.05, fov=False)
    fbp_cal = m['pred_cal'][0, 0].cpu().numpy()
    return truth_mu, fbp_cal, pixel_sp


def main() -> int:
    truth_np, fbp_np, pixel_sp = build_fbp_and_truth(target_pZ=-254.50,
                                                        filter_name="hann")
    H, W = truth_np.shape
    assert H == W == 512
    dr = 0.05

    # Set up coordinate grids. Image centre = iso (for our PYRO-NN FBP).
    truth = torch.from_numpy(truth_np).to("cuda").float()
    fbp = torch.from_numpy(fbp_np).to("cuda").float()
    yy, xx = torch.meshgrid(
        torch.arange(H, device="cuda").float(),
        torch.arange(W, device="cuda").float(),
        indexing="ij",
    )
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    r = torch.sqrt(dx**2 + dy**2)
    r_max = math.sqrt(cx**2 + cy**2)        # diagonal half-length
    r_norm = (r / r_max).clamp_max(1.0)     # ∈ [0, 1]

    print(f"[warp] cx={cx} cy={cy} r_max={r_max:.2f} px = "
          f"{r_max * pixel_sp:.2f} mm", flush=True)
    print(f"[warp] truth/fbp shape {H}×{W}, pixel_sp={pixel_sp:.4f} mm", flush=True)

    # Coordinate normalisation for grid_sample (range [-1, +1])
    # grid_sample expects: input[b, c, h_in, w_in],
    #                     grid[b, h_out, w_out, 2]  where 2 = (x, y) in [-1, 1]
    # x_norm = 2 * x / (W - 1) - 1     (and similarly y_norm)
    def warp(c1, c2, c3):
        scale = 1.0 + c1 * r_norm**2 + c2 * r_norm**4 + c3 * r_norm**6
        # New sample coordinates (in pixels) at which to sample FBP:
        # warped image at (x, y) = FBP at (cx + dx*scale, cy + dy*scale).
        # So grid_sample input grid = (cx + dx*scale, cy + dy*scale) normalised.
        x_src = cx + dx * scale
        y_src = cy + dy * scale
        x_norm_g = 2 * x_src / (W - 1) - 1
        y_norm_g = 2 * y_src / (H - 1) - 1
        grid = torch.stack([x_norm_g, y_norm_g], dim=-1)[None]    # (1, H, W, 2)
        warped = F.grid_sample(fbp[None, None], grid,
                                mode="bilinear", padding_mode="border",
                                align_corners=True)[0, 0]
        return warped

    # Calibrated baseline (no warp)
    with torch.no_grad():
        m_base = evaluate_calibrated(
            fbp[None, None], truth[None, None], baseline=fbp[None, None],
            display_min=0.0, display_max=dr, fov=False,
        )
        ssim_base = float(m_base["val_ssim"])
        psnr_base = float(m_base["val_psnr"])
        rmse_base = float(m_base["val_rmse"])
        diff_base = (m_base["pred_cal"][0, 0] - truth).cpu().numpy()
        fbp_cal_base = m_base["pred_cal"][0, 0].cpu().numpy()
    print(f"[warp] BEFORE  SSIM={ssim_base:.4f}  PSNR={psnr_base:.2f} dB  "
          f"RMSE={rmse_base:.5f}  diff_max={np.abs(diff_base).max():.4f}",
          flush=True)

    # Parameters: c1, c2, c3 — polynomial radial scale coefficients
    c1 = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    c2 = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    c3 = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    opt = torch.optim.Adam([c1, c2, c3], lr=5e-3)

    n_iters = 1500
    log_every = max(1, n_iters // 30)
    for it in range(n_iters):
        opt.zero_grad()
        warped = warp(c1, c2, c3)
        # Calibrate inside the loss (truth thresholds are fixed; gradient
        # flows through pred via intensity_calibrate which is
        # differentiable in pred).
        warped_cal = intensity_calibrate(warped, truth, display_max=dr)
        data_loss = ((warped_cal - truth) ** 2).mean()
        # Tikhonov on coefficient magnitudes — prefer small warp
        reg = 1e-3 * (c1**2 + c2**2 + c3**2)
        total = data_loss + reg
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                m = evaluate_calibrated(
                    warped[None, None], truth[None, None], baseline=fbp[None, None],
                    display_min=0.0, display_max=dr, fov=False,
                )
                ssim_i = float(m["val_ssim"])
            print(f"[warp] iter {it:4d}  data_loss={data_loss.item():.3e}  "
                  f"c1={c1.item():+.5f}  c2={c2.item():+.5f}  c3={c3.item():+.5f}  "
                  f"SSIM={ssim_i:.4f}", flush=True)

    # Final metrics with calibration
    with torch.no_grad():
        warped = warp(c1, c2, c3)
        m_after = evaluate_calibrated(
            warped[None, None], truth[None, None], baseline=fbp[None, None],
            display_min=0.0, display_max=dr, fov=False,
        )
        ssim_after = float(m_after["val_ssim"])
        psnr_after = float(m_after["val_psnr"])
        rmse_after = float(m_after["val_rmse"])
        warped_cal_np = m_after["pred_cal"][0, 0].cpu().numpy()
        diff_after = warped_cal_np - truth_np

    print()
    print("=== SUMMARY ===")
    print(f"BEFORE  SSIM={ssim_base:.4f}  PSNR={psnr_base:.2f} dB  RMSE={rmse_base:.5f}  "
          f"diff_max={np.abs(diff_base).max():.4f}")
    print(f"AFTER   SSIM={ssim_after:.4f}  PSNR={psnr_after:.2f} dB  RMSE={rmse_after:.5f}  "
          f"diff_max={np.abs(diff_after).max():.4f}")
    print(f"Δ       ΔSSIM={ssim_after-ssim_base:+.4f}  ΔPSNR={psnr_after-psnr_base:+.2f} dB  "
          f"ΔRMSE={(rmse_after-rmse_base)/rmse_base*100:+.1f}%")
    print(f"FITTED  c1={c1.item():+.5f}  c2={c2.item():+.5f}  c3={c3.item():+.5f}")
    print(f"        scale(r=rmax/2) = {1 + c1.item()*0.25 + c2.item()*0.0625 + c3.item()*0.015625:.5f}")
    print(f"        scale(r=rmax)   = {1 + c1.item() + c2.item() + c3.item():.5f}")

    # Plot ------------------------------------------------------------------
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4-panel comparison
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(truth_np, cmap="gray", vmin=0, vmax=dr)
    ax[0].set_title("truth (GT#76, pZ=−254.50 mm)\nB30f / 5-mm slice", fontsize=10)
    ax[1].imshow(np.clip(fbp_cal_base, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[1].set_title(f"FBP_cal (before warp)\n"
                    f"SSIM={ssim_base:.4f}  PSNR={psnr_base:.2f} dB  RMSE={rmse_base:.5f}",
                    fontsize=10)
    ax[2].imshow(np.clip(warped_cal_np, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[2].set_title(f"FBP_cal (radial polynomial warp)\n"
                    f"c1={c1.item():+.4f}, c2={c2.item():+.4f}, c3={c3.item():+.4f}\n"
                    f"SSIM={ssim_after:.4f}  PSNR={psnr_after:.2f} dB  RMSE={rmse_after:.5f}",
                    fontsize=9)
    ax[3].imshow(diff_after, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax[3].set_title(f"diff after warp\nmax|·|={np.abs(diff_after).max():.4f}",
                    fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("L014 fulldose: radial polynomial warp (curved→flat fisheye test)",
                 fontsize=11)
    fig.tight_layout()
    out_main = out_dir / "L014_radial_warp_fit.png"
    fig.savefig(out_main, dpi=120)
    print(f"[warp] wrote {out_main}", flush=True)

    # Before/after diff
    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.5))
    ax2[0].imshow(diff_base, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[0].set_title(f"diff BEFORE (FBP_cal − truth)\nmax|·|={np.abs(diff_base).max():.4f}",
                     fontsize=10)
    ax2[1].imshow(diff_after, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[1].set_title(f"diff AFTER warp\nmax|·|={np.abs(diff_after).max():.4f}",
                     fontsize=10)
    for a in ax2: a.set_xticks([]); a.set_yticks([])
    fig2.tight_layout()
    out_diff = out_dir / "L014_radial_warp_diff.png"
    fig2.savefig(out_diff, dpi=120)
    print(f"[warp] wrote {out_diff}", flush=True)

    # Radial displacement curve: (r_orig - r_new) vs r, in mm
    c1_, c2_, c3_ = c1.item(), c2.item(), c3.item()
    r_grid_mm = np.linspace(0, r_max * pixel_sp, 200)
    r_grid_norm = r_grid_mm / (r_max * pixel_sp)
    scale_grid = 1.0 + c1_ * r_grid_norm**2 + c2_ * r_grid_norm**4 + c3_ * r_grid_norm**6
    displacement_mm = (scale_grid - 1.0) * r_grid_mm

    fig3, ax3 = plt.subplots(1, 2, figsize=(13, 4.5))
    ax3[0].plot(r_grid_mm, scale_grid, lw=2)
    ax3[0].axhline(1.0, color="gray", ls=":", lw=0.8, label="identity")
    ax3[0].set_xlabel("radius from iso (mm)"); ax3[0].set_ylabel("scale factor")
    ax3[0].set_title("Radial scale factor s(r) — fitted polynomial warp")
    ax3[0].grid(alpha=0.3); ax3[0].legend()
    ax3[1].plot(r_grid_mm, displacement_mm, lw=2, color="C3")
    ax3[1].axhline(0, color="gray", ls=":", lw=0.8)
    ax3[1].set_xlabel("radius from iso (mm)")
    ax3[1].set_ylabel("radial pixel displacement (mm)\n(positive ⇒ FBP feature moves outward to match truth)")
    ax3[1].set_title("Radial displacement (r·s(r) − r) vs r")
    ax3[1].grid(alpha=0.3)
    fig3.tight_layout()
    out_prof = out_dir / "L014_radial_warp_profile.png"
    fig3.savefig(out_prof, dpi=120)
    print(f"[warp] wrote {out_prof}", flush=True)

    np.save(out_dir / "L014_radial_warp_coeffs.npy",
            np.array([c1_, c2_, c3_], dtype=np.float64))
    return 0


if __name__ == "__main__":
    sys.exit(main())
