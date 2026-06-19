#!/usr/bin/env python -u
"""Data-driven L2 fit of 5 FBP geometry parameters on the L014 peak GT slice.

We hold the rebinned sinogram fixed (= the helix2fan output) and vary
PYRO-NN's FBP back-projection geometry:

    1. pixel_spacing       — image grid mm/voxel
    2. det_spacing  (du)   — detector mm/channel
    3. sod                 — source-isocentre distance (mm)
    4. sdd                 — source-detector distance (mm)
    5. det_origin_offset   — sub-mm shift of the central-ray detector
                              column (added to the symmetric default)

For each parameter set we rebuild PyronnFanBeamProjector, run the
physical-overlap 5-mm slab FBP at GT#76 (pZ = −254.50 mm), intensity-
calibrate, and compute L2 to the truth DICOM. scipy Nelder-Mead handles
the 5-D scalar search (no gradients needed — each eval is one FBP).

If the optimum is non-trivial (≥ 0.5% relative shift in any param) AND
metrics improve substantially, those are the corrected values to use
in the next bulk rebin.

Usage:  python -u scripts/fit_fbp_geometry_L014.py
"""
from __future__ import annotations

import math
import sys
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pydicom
import torch
import scipy.optimize
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


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    nu, rotview, nz = int(geom_json['nu']), int(geom_json['rotview']), int(geom_json['nz_rebinned'])
    z_start_src = float(geom_json['z_start'])
    dv = float(geom_json.get('dv_rebinned', 1.0))
    angle_start = float(geom_json['angle_start_corrected'])

    # Nominal DICOM values
    du_nom = float(geom_json['du'])
    sod_nom = float(geom_json.get('sod', 595.0))
    sdd_nom = float(geom_json.get('sdd', 1085.6))

    truth_files = _list_truth(raw_dir)
    target_pZ = -254.50
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    pixel_sp_nom = float(ds.PixelSpacing[0])
    slice_thk = float(ds.SliceThickness)
    print(f"[geofit] truth #{ti} pZ={pZ:.2f} mm  PixelSpacing(DICOM)={pixel_sp_nom:.6f}  thk={slice_thk}",
          flush=True)
    print(f"[geofit] DICOM nominal du={du_nom:.6f} mm/ch  sod={sod_nom:.3f} mm  sdd={sdd_nom:.3f} mm",
          flush=True)

    # Build the physical-overlap slab indices + weights for GT#76 (cached).
    slab_lo, slab_hi = -pZ - slice_thk / 2.0, -pZ + slice_thk / 2.0
    j_lo = max(0, int(math.floor((slab_lo - z_start_src) / dv - 0.5)))
    j_hi = min(nz - 1, int(math.ceil((slab_hi - z_start_src) / dv + 0.5)))
    weights = {}
    for j in range(j_lo, j_hi + 1):
        z_j = z_start_src + j * dv
        bin_lo, bin_hi = z_j - dv / 2.0, z_j + dv / 2.0
        ov = max(0.0, min(bin_hi, slab_hi) - max(bin_lo, slab_lo))
        if ov > 0:
            weights[j] = ov / slice_thk
    print(f"[geofit] slab weights = {[f'{j}:{w:.3f}' for j, w in weights.items()]}",
          flush=True)

    # Pre-load the sino slices (they don't change with geometry).
    sino_slices = {}
    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        for j in weights:
            s = np.asarray(f["sino"][:, :, j], dtype=np.float32)
            sino_slices[j] = np.ascontiguousarray(np.flip(s, axis=-1))
    print(f"[geofit] loaded {len(sino_slices)} sino slices", flush=True)

    truth = torch.from_numpy(truth_mu_np).to("cuda").float()
    dr = 0.05

    # Eval cache (geometry → loss) so the optimiser can't ask the same point twice.
    cache: dict = {}
    n_evals = [0]

    def loss_fn(params):
        px, du_fit, sod_fit, sdd_fit, det_off = [float(p) for p in params]
        key = (round(px, 6), round(du_fit, 6), round(sod_fit, 3),
               round(sdd_fit, 3), round(det_off, 4))
        if key in cache:
            return cache[key]
        # Build the projector
        geom = FanBeamGeometry(
            image_size=512, pixel_spacing=px,
            n_angles=rotview, n_det=nu, det_spacing=du_fit,
            sod=sod_fit, sdd=sdd_fit,
            angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
        )
        proj = PyronnFanBeamProjector(geom).to("cuda")
        # Add sub-mm detector offset
        if abs(det_off) > 1e-9:
            new_origin = proj._tensor_geom["detector_origin"] + det_off
            proj._tensor_geom["detector_origin"] = new_origin

        # Run the slab FBP
        fbp_slab = np.zeros((512, 512), dtype=np.float64)
        for j, w in weights.items():
            t = torch.from_numpy(sino_slices[j]).to("cuda").float()[None, None]
            out = proj.fbp(t).detach()[0, 0].cpu().numpy()
            fbp_slab += w * np.fliplr(np.flipud(out))
        fbp_slab = np.clip(fbp_slab.astype(np.float32), 0.0, None)

        # Calibrate + L2
        fbp_t = torch.from_numpy(fbp_slab).to("cuda").float()[None, None]
        truth_t = truth[None, None]
        cal = intensity_calibrate(fbp_t, truth_t, display_max=dr)
        loss = float(((cal - truth_t) ** 2).mean().cpu())

        cache[key] = loss
        n_evals[0] += 1
        if n_evals[0] % 5 == 0 or n_evals[0] <= 5:
            print(f"[geofit] eval {n_evals[0]:3d}  px={px:.5f} du={du_fit:.5f} "
                  f"sod={sod_fit:.2f} sdd={sdd_fit:.2f} det_off={det_off:+.4f}  "
                  f"loss={loss:.3e}", flush=True)
        return loss

    # Baseline (nominal) loss
    t0 = time.time()
    init = np.array([pixel_sp_nom, du_nom, sod_nom, sdd_nom, 0.0])
    loss_baseline = loss_fn(init)
    print(f"[geofit] BASELINE loss = {loss_baseline:.5e}  "
          f"(eval took {time.time()-t0:.1f}s)", flush=True)

    # Bounds: ±5% on geometry distances; ±1 mm on detector offset
    bounds = [
        (pixel_sp_nom * 0.95,  pixel_sp_nom * 1.05),    # pixel_spacing
        (du_nom       * 0.95,  du_nom       * 1.05),    # det_spacing
        (sod_nom      * 0.95,  sod_nom      * 1.05),    # sod
        (sdd_nom      * 0.95,  sdd_nom      * 1.05),    # sdd
        (-2.0, 2.0),                                     # det_origin_offset (mm)
    ]
    # Nelder-Mead doesn't honour bounds natively; use 'L-BFGS-B' via finite
    # difference, or use Powell (also bound-aware in scipy ≥ 1.5).
    # Powell works well for noisy 5-D scalar problems with bounds.
    print(f"[geofit] starting Powell over 5-D, bounds=±5% (det_off ±2 mm)…",
          flush=True)
    result = scipy.optimize.minimize(
        loss_fn, init, method="Powell",
        bounds=bounds,
        options={"xtol": 1e-5, "ftol": 1e-8, "maxiter": 250, "disp": True},
    )

    px_f, du_f, sod_f, sdd_f, det_off_f = [float(x) for x in result.x]
    loss_fit = float(result.fun)
    print()
    print("=== SUMMARY ===")
    print(f"BASELINE  pixel_sp={pixel_sp_nom:.6f}  du={du_nom:.6f}  "
          f"sod={sod_nom:.3f}  sdd={sdd_nom:.3f}  det_off=+0.0000")
    print(f"          loss={loss_baseline:.5e}  PSNR={10*math.log10(dr**2 / loss_baseline):.2f} dB")
    print()
    print(f"FITTED    pixel_sp={px_f:.6f}  du={du_f:.6f}  "
          f"sod={sod_f:.3f}  sdd={sdd_f:.3f}  det_off={det_off_f:+.4f}")
    print(f"          loss={loss_fit:.5e}  PSNR={10*math.log10(dr**2 / loss_fit):.2f} dB")
    print()
    print(f"DELTAS    Δpx ={(px_f-pixel_sp_nom)*1e3:+.3f} µm")
    print(f"          Δdu ={(du_f-du_nom)*1e3:+.3f} µm")
    print(f"          Δsod={sod_f-sod_nom:+.3f} mm = {(sod_f/sod_nom-1)*100:+.3f} %")
    print(f"          Δsdd={sdd_f-sdd_nom:+.3f} mm = {(sdd_f/sdd_nom-1)*100:+.3f} %")
    print(f"          M_nom = sdd/sod = {sdd_nom/sod_nom:.5f}")
    print(f"          M_fit = sdd/sod = {sdd_f/sdd_f:.5f}  (Δ={sdd_f/sod_f-sdd_nom/sod_nom:+.5f})")
    print(f"          det_off = {det_off_f:+.4f} mm = {det_off_f/du_f:+.4f} channels")
    print(f"          n_evals = {n_evals[0]}")

    # Visualise the converged FBP (vs nominal vs truth)
    def run_one(params):
        px, du_fit, sod_fit, sdd_fit, det_off = params
        geom = FanBeamGeometry(
            image_size=512, pixel_spacing=px,
            n_angles=rotview, n_det=nu, det_spacing=du_fit,
            sod=sod_fit, sdd=sdd_fit,
            angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
        )
        proj = PyronnFanBeamProjector(geom).to("cuda")
        if abs(det_off) > 1e-9:
            proj._tensor_geom["detector_origin"] = (
                proj._tensor_geom["detector_origin"] + det_off
            )
        fbp_slab = np.zeros((512, 512), dtype=np.float64)
        for j, w in weights.items():
            t = torch.from_numpy(sino_slices[j]).to("cuda").float()[None, None]
            out = proj.fbp(t).detach()[0, 0].cpu().numpy()
            fbp_slab += w * np.fliplr(np.flipud(out))
        fbp_slab = np.clip(fbp_slab.astype(np.float32), 0.0, None)
        fbp_t = torch.from_numpy(fbp_slab).to("cuda").float()[None, None]
        cal = intensity_calibrate(fbp_t, truth[None, None], display_max=dr)
        return cal[0, 0].cpu().numpy(), fbp_slab

    fbp_nom_cal, _ = run_one(init.tolist())
    fbp_fit_cal, _ = run_one(result.x.tolist())

    def metrics(pred_np):
        pred_t = torch.from_numpy(np.clip(pred_np, 0, None)).to("cuda").float()[None, None]
        truth_t = truth[None, None]
        return {
            "ssim": float(ssim_fn(pred_t, truth_t, data_range=dr).cpu()),
            "psnr": float(psnr_fn(pred_t, truth_t, data_range=dr).cpu()),
            "rmse": float(((pred_t - truth_t) ** 2).mean().sqrt().cpu()),
            "diff_max": float(np.abs(pred_np - truth_mu_np).max()),
        }

    m_nom = metrics(fbp_nom_cal)
    m_fit = metrics(fbp_fit_cal)
    print()
    print(f"NOMINAL geom  SSIM={m_nom['ssim']:.4f}  PSNR={m_nom['psnr']:.2f} dB  "
          f"RMSE={m_nom['rmse']:.5f}  diff_max={m_nom['diff_max']:.4f}")
    print(f"FITTED  geom  SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
          f"RMSE={m_fit['rmse']:.5f}  diff_max={m_fit['diff_max']:.4f}")
    print(f"Δ             ΔSSIM={m_fit['ssim']-m_nom['ssim']:+.4f}  "
          f"ΔPSNR={m_fit['psnr']-m_nom['psnr']:+.2f} dB  "
          f"ΔRMSE={(m_fit['rmse']-m_nom['rmse'])/m_nom['rmse']*100:+.1f}%")

    # Plot
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_nom = fbp_nom_cal - truth_mu_np
    diff_fit = fbp_fit_cal - truth_mu_np
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(truth_mu_np, cmap="gray", vmin=0, vmax=dr)
    ax[0].set_title("truth", fontsize=10)
    ax[1].imshow(np.clip(fbp_nom_cal, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[1].set_title(f"FBP_cal nominal geom\n"
                    f"SSIM={m_nom['ssim']:.4f}  PSNR={m_nom['psnr']:.2f} dB\n"
                    f"RMSE={m_nom['rmse']:.5f}", fontsize=9)
    ax[2].imshow(np.clip(fbp_fit_cal, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[2].set_title(f"FBP_cal fitted geom\n"
                    f"SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB\n"
                    f"RMSE={m_fit['rmse']:.5f}", fontsize=9)
    ax[3].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax[3].set_title(f"diff fitted vs truth\nmax|·|={np.abs(diff_fit).max():.4f}",
                    fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"L014 fulldose: data-driven FBP geometry fit "
                 f"(Δsod={sod_f-sod_nom:+.2f} mm  Δsdd={sdd_f-sdd_nom:+.2f} mm  "
                 f"Δdu={(du_f-du_nom)*1e3:+.2f} µm  det_off={det_off_f:+.3f} mm)",
                 fontsize=10)
    fig.tight_layout()
    out_main = out_dir / "L014_fbp_geometry_fit.png"
    fig.savefig(out_main, dpi=120)
    print(f"[geofit] wrote {out_main}", flush=True)

    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.5))
    ax2[0].imshow(diff_nom, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[0].set_title(f"diff BEFORE (nominal geom)\nmax|·|={np.abs(diff_nom).max():.4f}",
                     fontsize=10)
    ax2[1].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[1].set_title(f"diff AFTER (fitted geom)\nmax|·|={np.abs(diff_fit).max():.4f}",
                     fontsize=10)
    for a in ax2: a.set_xticks([]); a.set_yticks([])
    fig2.tight_layout()
    out_diff = out_dir / "L014_fbp_geometry_diff.png"
    fig2.savefig(out_diff, dpi=120)
    print(f"[geofit] wrote {out_diff}", flush=True)

    # Save the fitted geometry as a JSON
    out_json = out_dir / "L014_fbp_geometry_fitted.json"
    out_json.write_text(json.dumps({
        "nominal": {
            "pixel_spacing": pixel_sp_nom, "du": du_nom,
            "sod": sod_nom, "sdd": sdd_nom, "det_origin_offset": 0.0,
        },
        "fitted": {
            "pixel_spacing": px_f, "du": du_f,
            "sod": sod_f, "sdd": sdd_f, "det_origin_offset": det_off_f,
        },
        "metrics_nominal": m_nom,
        "metrics_fitted": m_fit,
        "n_evals": n_evals[0],
        "scipy_message": result.message,
        "scipy_status": int(result.status),
    }, indent=2))
    print(f"[geofit] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
