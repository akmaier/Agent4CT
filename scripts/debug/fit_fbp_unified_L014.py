#!/usr/bin/env python -u
"""Unified gradient-descent fit: radial warp + frequency filter +
intensity scaling + ReLU-clip — all simultaneously via Adam.

Replaces the previous staged pipeline:
  (1) scipy Powell on 5 geometry scalars                    → +3.26 dB
  (2) Adam on a 96-bin radial filter H(ρ)                   → +0.78 dB
  (3) Post-hoc intensity_calibrate (2-point linear)          → ≈ +0 dB
  (4) Post-hoc clamp(0, display_max)                         → noop

with a single Adam loop over 4 parameter groups, all differentiable:

  Pipeline
  --------
  1. fbp_nom = back_project(sino, NOMINAL geometry)         [fixed input]
  2. warped  = grid_sample(fbp_nom, warp(c_warp))           # geometry
     where  s(r) = 1 + c1·(r/rmax)² + c2·(r/rmax)⁴ + c3·(r/rmax)⁶
  3. filt   = IFFT(H_2d(h_radial) · FFT(warped)).real       # kernel
  4. scaled = a · (filt − bg)                               # intensity
  5. clipped = relu(scaled)             [optionally: min(clipped, hi)]

  Loss
  ----
  L = ‖clipped − truth‖²  +  λ_H · ‖Δ² h_radial‖²  +  λ_w · ‖c_warp‖²

Adam updates (c_warp, h_radial, a, bg, [hi]) jointly.

The PYRO-NN FBP runs ONCE (at nominal geometry); subsequent
optimization is pure tensor math on GPU — fast and gradient-clean.

Usage:  python -u scripts/fit_fbp_unified_L014.py
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
    intensity_calibrate, ssim as ssim_fn, psnr as psnr_fn,
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


# ----------------------------------------------------------------------
# Pipeline forward.
# ----------------------------------------------------------------------

def warp_grid(c_warp: torch.Tensor, H: int, W: int,
               cx: float, cy: float, device: str):
    """Return a (1, H, W, 2) grid for F.grid_sample, normalised to [-1, 1].
    Applies a polynomial radial scale around (cx, cy)."""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - cx
    dy = yy - cy
    r = torch.sqrt(dx * dx + dy * dy)
    r_max = math.sqrt(cx * cx + cy * cy)
    r_norm = (r / r_max).clamp_max(1.0)
    scale = (1.0 + c_warp[0] * r_norm**2
                  + c_warp[1] * r_norm**4
                  + c_warp[2] * r_norm**6)
    x_src = cx + dx * scale
    y_src = cy + dy * scale
    gx = 2.0 * x_src / (W - 1) - 1.0
    gy = 2.0 * y_src / (H - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)[None]


def radial_filter_2d(h_radial: torch.Tensor, rho: torch.Tensor,
                      n_bins: int) -> torch.Tensor:
    """Map 1-D radial-frequency vector h_radial[k] → 2-D filter H(ρ) on
    the FFT2 grid via linear interpolation between bins."""
    rho_max = float(rho.max())
    bin_pos = (rho / rho_max) * (n_bins - 1)
    bin_lo = bin_pos.floor().long().clamp(0, n_bins - 1)
    bin_hi = (bin_lo + 1).clamp(0, n_bins - 1)
    bin_frac = (bin_pos - bin_lo.float())
    return h_radial[bin_lo] * (1 - bin_frac) + h_radial[bin_hi] * bin_frac


def forward_pipeline(fbp_nom: torch.Tensor, params: dict,
                      precomp: dict) -> torch.Tensor:
    H, W = fbp_nom.shape
    # 1. radial warp
    grid = warp_grid(params["c_warp"], H, W,
                      precomp["cx"], precomp["cy"], precomp["device"])
    warped = F.grid_sample(fbp_nom[None, None], grid,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)[0, 0]
    # 2. radial frequency filter
    fbp_fft = torch.fft.fft2(warped)
    h_2d = radial_filter_2d(params["h_radial"], precomp["rho"],
                              params["h_radial"].shape[0])
    filt_fft = torch.complex(h_2d * fbp_fft.real, h_2d * fbp_fft.imag)
    filt = torch.fft.ifft2(filt_fft).real
    # 3. intensity scale
    scaled = params["a"] * (filt - params["bg"])
    # 4. ReLU-like clip
    clipped = F.relu(scaled)
    # 5. optional upper clip (display range)
    hi = params.get("hi", None)
    if hi is not None:
        # soft upper bound: clamp(0, hi) via min
        clipped = torch.minimum(clipped, hi)
    return clipped


def calc_metrics(pred_np: np.ndarray, truth_np: np.ndarray, dr: float = 0.05):
    pred_t = torch.from_numpy(np.clip(pred_np, 0, None)).to("cuda").float()[None, None]
    truth_t = torch.from_numpy(truth_np).to("cuda").float()[None, None]
    return {
        "ssim": float(ssim_fn(pred_t, truth_t, data_range=dr).cpu()),
        "psnr": float(psnr_fn(pred_t, truth_t, data_range=dr).cpu()),
        "rmse": float(((pred_t - truth_t) ** 2).mean().sqrt().cpu()),
        "diff_max": float(np.abs(pred_np - truth_np).max()),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    nu, rotview, nz = int(geom_json['nu']), int(geom_json['rotview']), int(geom_json['nz_rebinned'])
    du_nom = float(geom_json['du'])
    sod_nom = float(geom_json.get('sod', 595.0))
    sdd_nom = float(geom_json.get('sdd', 1085.6))
    dv = float(geom_json.get('dv_rebinned', 1.0))
    z_start_src = float(geom_json['z_start'])
    angle_start = float(geom_json['angle_start_corrected'])

    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    target_pZ = -254.50
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    pixel_sp_truth = float(ds.PixelSpacing[0])
    slice_thk = float(ds.SliceThickness)
    print(f"[uni] truth #{ti} pZ={pZ:.2f}  PixelSpacing(DICOM)={pixel_sp_truth:.6f}", flush=True)

    # Build the physical-overlap 5-mm slab weights for the truth slice
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

    # FBP at NOMINAL geometry (one-time)
    geom_nom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp_truth,
        n_angles=rotview, n_det=nu, det_spacing=du_nom,
        sod=sod_nom, sdd=sdd_nom,
        angle_start=angle_start, angle_end=angle_start + 2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom_nom).to("cuda")
    fbp_slab = np.zeros_like(truth_mu_np, dtype=np.float64)
    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        for j, w in weights.items():
            s = np.ascontiguousarray(np.flip(
                np.asarray(f["sino"][:, :, j], dtype=np.float32),
                axis=-1,
            ))
            t = torch.from_numpy(s).to("cuda").float()[None, None]
            out = proj.fbp(t).detach()[0, 0].cpu().numpy()
            fbp_slab += w * np.fliplr(np.flipud(out))
    fbp_nom_np = np.clip(fbp_slab.astype(np.float32), 0.0, None)

    # Tensors on CUDA
    fbp_nom = torch.from_numpy(fbp_nom_np).to("cuda").float()
    truth = torch.from_numpy(truth_mu_np).to("cuda").float()

    # Precomputed frequency grid + image centre
    H, W = truth.shape
    fy = torch.fft.fftfreq(H, device="cuda").float()
    fx = torch.fft.fftfreq(W, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy**2 + fxx**2)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    precomp = {"rho": rho, "cx": cx, "cy": cy, "device": "cuda"}

    dr = 0.05
    n_bins = 64

    # ----- Parameter init -----
    # c_warp: polynomial radial scale coefficients
    c_warp = torch.nn.Parameter(torch.zeros(3, device="cuda", dtype=torch.float32))
    # h_radial: radial freq filter, initialised to identity
    h_radial = torch.nn.Parameter(torch.ones(n_bins, device="cuda", dtype=torch.float32))
    # Intensity scaling: a ≈ 1, bg ≈ 0 (with calibration the data lives in
    # the same μ-units as truth, so this is roughly correct already).
    a = torch.nn.Parameter(torch.tensor(1.0, device="cuda", dtype=torch.float32))
    bg = torch.nn.Parameter(torch.tensor(0.0, device="cuda", dtype=torch.float32))
    # Optional upper-clip parameter (initialised at dr = display_max)
    hi = torch.nn.Parameter(torch.tensor(dr, device="cuda", dtype=torch.float32))

    # Baseline metrics
    m_base_uncal = calc_metrics(fbp_nom_np, truth_mu_np, dr=dr)
    print(f"[uni] BASELINE (nominal FBP, uncalibrated): "
          f"SSIM={m_base_uncal['ssim']:.4f}  PSNR={m_base_uncal['psnr']:.2f} dB  "
          f"RMSE={m_base_uncal['rmse']:.5f}", flush=True)

    m_base_cal_np = intensity_calibrate(
        torch.from_numpy(fbp_nom_np).to("cuda").float()[None, None],
        truth[None, None], display_max=dr,
    )[0, 0].cpu().numpy()
    m_base_cal = calc_metrics(m_base_cal_np, truth_mu_np, dr=dr)
    print(f"[uni] BASELINE (nominal FBP, intensity_calibrate): "
          f"SSIM={m_base_cal['ssim']:.4f}  PSNR={m_base_cal['psnr']:.2f} dB  "
          f"RMSE={m_base_cal['rmse']:.5f}", flush=True)

    # ----- Optimisation -----
    params = {"c_warp": c_warp, "h_radial": h_radial,
              "a": a, "bg": bg, "hi": hi}
    opt = torch.optim.Adam([c_warp, h_radial, a, bg, hi], lr=2e-3)
    n_iters = 3000
    log_every = max(1, n_iters // 30)
    lam_h = 1e-4    # smoothness on H
    lam_w = 1e-3    # Tikhonov on c_warp

    print(f"[uni] Adam fit, {n_iters} iters, n_bins(H)={n_bins}, "
          f"lr=2e-3, λ_H={lam_h}, λ_w={lam_w}", flush=True)

    for it in range(n_iters):
        opt.zero_grad()
        pred = forward_pipeline(fbp_nom, params, precomp)
        data_loss = ((pred - truth) ** 2).mean()
        smooth_loss = ((h_radial[2:] - 2 * h_radial[1:-1] + h_radial[:-2]) ** 2).mean()
        warp_reg = (c_warp ** 2).sum()
        total = data_loss + lam_h * smooth_loss + lam_w * warp_reg
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                m_iter = calc_metrics(pred.detach().cpu().numpy(),
                                        truth_mu_np, dr=dr)
            print(f"[uni] iter {it:4d}/{n_iters}  data_loss={data_loss.item():.3e}  "
                  f"c_warp=[{c_warp[0].item():+.5f},{c_warp[1].item():+.5f},{c_warp[2].item():+.5f}]  "
                  f"a={a.item():.3f} bg={bg.item():+.4f} hi={hi.item():.3f}  "
                  f"H_range=[{h_radial.min().item():.3f},{h_radial.max().item():.3f}]  "
                  f"SSIM={m_iter['ssim']:.4f} PSNR={m_iter['psnr']:.2f}",
                  flush=True)

    # Final
    with torch.no_grad():
        pred_final = forward_pipeline(fbp_nom, params, precomp)
        pred_final_np = pred_final.detach().cpu().numpy()
    m_fit = calc_metrics(pred_final_np, truth_mu_np, dr=dr)
    print()
    print("=== SUMMARY ===")
    print(f"BASELINE (nominal FBP, intensity_calibrate)")
    print(f"   SSIM={m_base_cal['ssim']:.4f}  PSNR={m_base_cal['psnr']:.2f} dB  "
          f"RMSE={m_base_cal['rmse']:.5f}  diff_max={m_base_cal['diff_max']:.4f}")
    print(f"FITTED (unified gradient descent)")
    print(f"   SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
          f"RMSE={m_fit['rmse']:.5f}  diff_max={m_fit['diff_max']:.4f}")
    print(f"Δ  ΔSSIM={m_fit['ssim']-m_base_cal['ssim']:+.4f}  "
          f"ΔPSNR={m_fit['psnr']-m_base_cal['psnr']:+.2f} dB  "
          f"ΔRMSE={(m_fit['rmse']-m_base_cal['rmse'])/m_base_cal['rmse']*100:+.1f}%")
    print()
    print(f"LEARNED  c_warp = [{c_warp[0].item():+.5f}, {c_warp[1].item():+.5f}, {c_warp[2].item():+.5f}]")
    print(f"         a={a.item():.4f}  bg={bg.item():+.5f}  hi={hi.item():.4f}")
    print(f"         H(ρ) range = [{h_radial.min().item():.3f}, {h_radial.max().item():.3f}]")
    print(f"         H(0) = {h_radial[0].item():.3f}  H(rho_max/2) ≈ {h_radial[n_bins//2].item():.3f}  H(rho_max) ≈ {h_radial[-1].item():.3f}")

    # ----- Plots -----
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_base = m_base_cal_np - truth_mu_np
    diff_fit = pred_final_np - truth_mu_np

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(truth_mu_np, cmap="gray", vmin=0, vmax=dr)
    ax[0].set_title("truth (GT#76, pZ=−254.50 mm)\nB30f / 5-mm slice", fontsize=10)
    ax[1].imshow(np.clip(m_base_cal_np, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[1].set_title(f"FBP (nominal + intensity_calibrate)\n"
                    f"SSIM={m_base_cal['ssim']:.4f}  PSNR={m_base_cal['psnr']:.2f} dB  "
                    f"RMSE={m_base_cal['rmse']:.5f}", fontsize=9)
    ax[2].imshow(np.clip(pred_final_np, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[2].set_title(f"FBP (unified gradient-descent fit)\n"
                    f"SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
                    f"RMSE={m_fit['rmse']:.5f}", fontsize=9)
    ax[3].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax[3].set_title(f"diff after unified fit\n"
                    f"max|·|={np.abs(diff_fit).max():.4f}", fontsize=10)
    for a_ax in ax: a_ax.set_xticks([]); a_ax.set_yticks([])
    fig.suptitle("L014 fulldose: unified gradient-descent fit "
                 "(warp + freq filter + scale + ReLU)", fontsize=11)
    fig.tight_layout()
    out_main = out_dir / "L014_unified_fit.png"
    fig.savefig(out_main, dpi=120)
    print(f"[uni] wrote {out_main}", flush=True)

    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.5))
    ax2[0].imshow(diff_base, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[0].set_title(f"diff BEFORE (intensity_calibrate)\n"
                     f"max|·|={np.abs(diff_base).max():.4f}", fontsize=10)
    ax2[1].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[1].set_title(f"diff AFTER (unified fit)\n"
                     f"max|·|={np.abs(diff_fit).max():.4f}", fontsize=10)
    for a_ax in ax2: a_ax.set_xticks([]); a_ax.set_yticks([])
    fig2.tight_layout()
    out_diff = out_dir / "L014_unified_fit_diff.png"
    fig2.savefig(out_diff, dpi=120)
    print(f"[uni] wrote {out_diff}", flush=True)

    # Learned radial profiles
    r_max_px = math.sqrt(cx**2 + cy**2)
    r_axis_mm = np.linspace(0, r_max_px * pixel_sp_truth, 200)
    r_norm = r_axis_mm / (r_max_px * pixel_sp_truth)
    c1_, c2_, c3_ = [v.item() for v in c_warp]
    scale_curve = 1 + c1_ * r_norm**2 + c2_ * r_norm**4 + c3_ * r_norm**6
    rho_axis = np.linspace(0, float(rho.max()), n_bins)
    h_radial_np = h_radial.detach().cpu().numpy()

    fig3, ax3 = plt.subplots(1, 2, figsize=(13, 4.5))
    ax3[0].plot(r_axis_mm, scale_curve, lw=2, color="C0")
    ax3[0].axhline(1.0, color="gray", ls=":", lw=0.6)
    ax3[0].set_xlabel("radius from iso (mm)")
    ax3[0].set_ylabel("radial scale factor s(r)")
    ax3[0].set_title(f"Geometry warp (joint fit)\n"
                     f"c1={c1_:+.5f}  c2={c2_:+.5f}  c3={c3_:+.5f}")
    ax3[0].grid(alpha=0.3)
    ax3[1].plot(rho_axis, h_radial_np, lw=2, color="C1")
    ax3[1].axhline(1.0, color="gray", ls=":", lw=0.6, label="identity")
    ax3[1].set_xlabel("radial frequency ρ (cycles / pixel)")
    ax3[1].set_ylabel("filter response H(ρ)")
    ax3[1].set_title(f"Kernel filter (joint fit)\n"
                     f"a={a.item():.4f}  bg={bg.item():+.5f}  hi={hi.item():.4f}")
    ax3[1].grid(alpha=0.3); ax3[1].legend()
    fig3.tight_layout()
    out_curves = out_dir / "L014_unified_fit_curves.png"
    fig3.savefig(out_curves, dpi=120)
    print(f"[uni] wrote {out_curves}", flush=True)

    # JSON dump
    out_json = out_dir / "L014_unified_fit.json"
    out_json.write_text(json.dumps({
        "c_warp": [c1_, c2_, c3_],
        "a": float(a.item()),
        "bg": float(bg.item()),
        "hi": float(hi.item()),
        "h_radial": h_radial_np.tolist(),
        "rho_axis": rho_axis.tolist(),
        "metrics_baseline_calibrated": m_base_cal,
        "metrics_fitted": m_fit,
        "n_iters": n_iters,
        "n_bins": n_bins,
    }, indent=2))
    print(f"[uni] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
