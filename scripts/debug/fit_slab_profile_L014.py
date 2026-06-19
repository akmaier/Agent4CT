#!/usr/bin/env python -u
"""Fit Mayo Siemens's effective slice-sensitivity profile S(z) on L014
fulldose by minimising L2 between a weighted sum of 1-mm-rebinned FBP
slices and the corresponding raw GT DICOM (no z-interpolation):

  slab_n(x, y) = Σ_i w_i · FBP_{j_center_n + i}(x, y)
  loss        = Σ_n ‖slab_n − truth_n‖²
              + λ_smooth · ‖Δ²w‖²

w is parametrised as `softmax(θ)` ⇒ Σw = 1 and w ≥ 0 automatically.
9 weights on integer-mm offsets {−4, −3, …, +3, +4} cover ±4 mm
around the closest-to-truth-z sino slice. The fit is shared across
11 consecutive GT slices around the cone-beam centre, so the
recovered S is anatomy-independent.

Baseline for comparison: uniform overlap-weighted 5-mm slab
(scripts/z_aligned_validation.py uses w ≈ {0.072, 0.2, 0.2, 0.2,
0.2, 0.128} for L014's 0.31 mm misalignment).

Outputs:
  L014_slab_profile_fit.png     — S_fit(z) curve vs S_uniform(z)
  L014_slab_profile_montage.png — per-GT 5-column montage:
                                   truth | uniform | fitted | diff_unif | diff_fit
  L014_slab_profile_metrics.csv — per-GT SSIM/PSNR/RMSE before/after
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


def calc_metrics(pred_np: np.ndarray, truth_np: np.ndarray, dr: float = 0.05):
    pred_clip = np.clip(pred_np, 0.0, None)
    pred_t = torch.from_numpy(pred_clip).to("cuda").float()[None, None]
    truth_t = torch.from_numpy(truth_np).to("cuda").float()[None, None]
    return {
        "ssim": float(ssim_fn(pred_t, truth_t, data_range=dr).cpu()),
        "psnr": float(psnr_fn(pred_t, truth_t, data_range=dr).cpu()),
        "rmse": float(((pred_t - truth_t) ** 2).mean().sqrt().cpu()),
        "diff_max": float(np.abs(pred_clip - truth_np).max()),
    }


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    nu, rotview, nz = int(geom_json['nu']), int(geom_json['rotview']), int(geom_json['nz_rebinned'])
    du = float(geom_json['du'])
    dv = float(geom_json.get('dv_rebinned', 1.0))
    z_start = float(geom_json['z_start'])
    angle_start = float(geom_json['angle_start_corrected'])

    truth_files = _list_truth(raw_dir)
    truth_zs = np.array([t[0] for t in truth_files])
    pixel_sp = float(pydicom.dcmread(str(truth_files[0][1]),
                                       stop_before_pixels=True).PixelSpacing[0])
    slice_thk = float(pydicom.dcmread(str(truth_files[0][1]),
                                        stop_before_pixels=True).SliceThickness)
    print(f"[slab] truth PixelSpacing={pixel_sp:.4f}  SliceThickness={slice_thk}",
          flush=True)

    # Centre of the rebinned sino in source frame
    sino_centre_src = z_start + (nz / 2) * dv
    target_pZ_centre = -sino_centre_src
    centre_idx = int(np.argmin(np.abs(truth_zs - target_pZ_centre)))
    span = 5
    gt_idxs = list(range(max(0, centre_idx - span),
                          min(len(truth_files), centre_idx + span + 1)))
    print(f"[slab] sweep GT indices {gt_idxs[0]}..{gt_idxs[-1]} = {len(gt_idxs)} slices",
          flush=True)

    # Slab support: ±N_HALF integer-mm offsets around the closest-to-truth-z sino slice.
    # N_HALF = 4 covers ±4 mm = 9 bins. Should be enough for any reasonable
    # 5-mm slice profile.
    N_HALF = 4
    n_bins = 2 * N_HALF + 1   # 9

    # Build (truth_n, j_center_n) for each GT slice
    gt_data = []
    for ti in gt_idxs:
        pZ, fp = truth_files[ti]
        source_z = -pZ
        j_center = int(round((source_z - z_start) / dv))
        if j_center - N_HALF < 0 or j_center + N_HALF >= nz:
            print(f"[slab] WARN GT #{ti} out of range, skipping", flush=True)
            continue
        truth_mu, _ = _mu(fp)
        gt_data.append({"ti": ti, "pZ": pZ, "source_z": source_z,
                         "j_center": j_center, "truth": truth_mu})
        # Note offset between closest sino z and truth source_z
        sino_z = z_start + j_center * dv
        print(f"[slab] GT #{ti:3d} pZ={pZ:7.2f}  source_z={source_z:7.2f}  "
              f"j_center={j_center}  sino_z={sino_z:7.2f}  "
              f"Δ={sino_z - source_z:+.3f} mm", flush=True)

    # Collect all unique sino j indices we need
    j_set = set()
    for g in gt_data:
        for i in range(-N_HALF, N_HALF + 1):
            j_set.add(g["j_center"] + i)
    j_sorted = sorted(j_set)
    print(f"[slab] need {len(j_sorted)} unique sino slices (j_range "
          f"{j_sorted[0]}..{j_sorted[-1]})", flush=True)

    # FBP each unique slice
    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start + 2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")
    fbp_dict: dict[int, np.ndarray] = {}
    print(f"[slab] running FBP on {len(j_sorted)} sino slices…", flush=True)
    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        for k, j in enumerate(j_sorted):
            s = np.asarray(f["sino"][:, :, j], dtype=np.float32)
            s = np.ascontiguousarray(np.flip(s, axis=-1))
            t = torch.from_numpy(s).to("cuda").float()[None, None]
            out = proj.fbp(t).detach()[0, 0].cpu().numpy()
            fbp_dict[j] = np.fliplr(np.flipud(out)).astype(np.float32)
            if (k + 1) % 10 == 0:
                print(f"[slab]   {k+1}/{len(j_sorted)} done", flush=True)
    print(f"[slab] FBP done.", flush=True)

    # Stack FBPs into a tensor per GT: shape (n_bins, H, W)
    H, W = gt_data[0]["truth"].shape
    fbp_stack = torch.zeros((len(gt_data), n_bins, H, W),
                              device="cuda", dtype=torch.float32)
    truth_stack = torch.zeros((len(gt_data), H, W),
                               device="cuda", dtype=torch.float32)
    for n, g in enumerate(gt_data):
        for i, off in enumerate(range(-N_HALF, N_HALF + 1)):
            fbp_stack[n, i] = torch.from_numpy(fbp_dict[g["j_center"] + off]).to("cuda")
        truth_stack[n] = torch.from_numpy(g["truth"]).to("cuda")
    fbp_stack = fbp_stack.clamp_min(0.0)

    # ---- Optimisation ----------------------------------------------------
    # w = softmax(theta) ⇒ Σw=1, w≥0
    theta = torch.nn.Parameter(torch.zeros(n_bins, device="cuda"))
    lam_smooth = 1e-3
    n_iters = 1500
    log_every = max(1, n_iters // 25)

    def make_w():
        return F.softmax(theta, dim=0)

    def slab(w):
        # w: (n_bins,);  fbp_stack: (N_gt, n_bins, H, W)
        return (w[None, :, None, None] * fbp_stack).sum(dim=1)   # (N_gt, H, W)

    # First, evaluate the BASELINE: physical-overlap weights (= what
    # z_aligned_validation.py uses). For an unaligned rebin (Δ=+0.31 mm),
    # the centred 5-mm slab gives w ≈ {0, …, 0.072, 0.2, 0.2, 0.2, 0.2,
    # 0.128, 0, …, 0}. The exact values depend on j_center but are the
    # same for all GTs in our sweep (the offset is constant patient-wide).
    w_base_np = np.zeros(n_bins, dtype=np.float32)
    g0 = gt_data[0]
    slab_lo_src = g0["source_z"] - slice_thk / 2.0
    slab_hi_src = g0["source_z"] + slice_thk / 2.0
    for i, off in enumerate(range(-N_HALF, N_HALF + 1)):
        z_j = z_start + (g0["j_center"] + off) * dv
        bin_lo, bin_hi = z_j - dv / 2.0, z_j + dv / 2.0
        ov = max(0.0, min(bin_hi, slab_hi_src) - max(bin_lo, slab_lo_src))
        w_base_np[i] = ov / slice_thk
    w_base = torch.from_numpy(w_base_np).to("cuda")
    print(f"[slab] baseline w (uniform 5-mm overlap) = "
          f"{[f'{x:.3f}' for x in w_base_np]}  Σ={w_base_np.sum():.4f}", flush=True)

    def calibrate_stack(slab_t: torch.Tensor) -> torch.Tensor:
        """Per-GT intensity_calibrate (differentiable: thresholds come from
        truth, which is a fixed tensor, so all gradients flow back to slab)."""
        out = []
        for n in range(slab_t.shape[0]):
            out.append(intensity_calibrate(slab_t[n], truth_stack[n],
                                            display_max=0.05))
        return torch.stack(out, dim=0)

    # Baseline metrics on the CALIBRATED slab (apples-to-apples with z_aligned)
    with torch.no_grad():
        slab_base = slab(w_base)
        slab_base_cal = calibrate_stack(slab_base)
        data_base = ((slab_base_cal - truth_stack) ** 2).mean().item()
        print(f"[slab] baseline data_loss (after cal) = {data_base:.3e}", flush=True)

    opt = torch.optim.Adam([theta], lr=5e-2)
    print(f"[slab] Adam fit, {n_iters} iters, λ_smooth={lam_smooth}", flush=True)
    for it in range(n_iters):
        opt.zero_grad()
        w = make_w()
        slab_pred = slab(w)
        slab_pred_cal = calibrate_stack(slab_pred)
        data_loss = ((slab_pred_cal - truth_stack) ** 2).mean()
        smooth_loss = ((w[2:] - 2 * w[1:-1] + w[:-2]) ** 2).mean()
        total = data_loss + lam_smooth * smooth_loss
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            print(f"[slab] iter {it:4d}/{n_iters}  data_loss={data_loss.item():.3e}  "
                  f"smooth_loss={smooth_loss.item():.3e}  "
                  f"w_max={w.max().item():.3f}  w_min={w.min().item():.4f}",
                  flush=True)

    with torch.no_grad():
        w_fit = make_w().detach().cpu().numpy()
        slab_fit_t = slab(make_w())
        slab_fit_cal = calibrate_stack(slab_fit_t)
        slab_fit_np = slab_fit_cal.detach().cpu().numpy()
        slab_base_cal_np = calibrate_stack(slab_base).detach().cpu().numpy()
    slab_base_np = slab_base_cal_np   # report calibrated baseline
    truth_np = truth_stack.detach().cpu().numpy()

    print(f"[slab] fitted w = {[f'{x:.3f}' for x in w_fit]}  Σ={w_fit.sum():.4f}",
          flush=True)

    # Per-GT metrics: BEFORE (uniform/baseline) vs AFTER (fit)
    dr = 0.05
    rows = []
    for n, g in enumerate(gt_data):
        m_base = calc_metrics(slab_base_np[n], truth_np[n], dr=dr)
        m_fit = calc_metrics(slab_fit_np[n], truth_np[n], dr=dr)
        rows.append({**g, "m_base": m_base, "m_fit": m_fit,
                     "slab_base": slab_base_np[n], "slab_fit": slab_fit_np[n],
                     "truth_np": truth_np[n]})
        print(f"[slab] GT #{g['ti']:3d}  pZ={g['pZ']:+7.2f}  "
              f"BASE SSIM={m_base['ssim']:.4f} PSNR={m_base['psnr']:.2f}dB RMSE={m_base['rmse']:.5f}  "
              f"FIT  SSIM={m_fit['ssim']:.4f} PSNR={m_fit['psnr']:.2f}dB RMSE={m_fit['rmse']:.5f}  "
              f"Δ={m_fit['ssim']-m_base['ssim']:+.4f}/{m_fit['psnr']-m_base['psnr']:+.2f}",
              flush=True)

    # Save CSV
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "L014_slab_profile_metrics.csv"
    with open(csv, "w") as f:
        f.write("gt_idx,pZ,base_ssim,base_psnr,base_rmse,base_diffmax,"
                "fit_ssim,fit_psnr,fit_rmse,fit_diffmax\n")
        for r in rows:
            f.write(f"{r['ti']},{r['pZ']:.2f},"
                    f"{r['m_base']['ssim']:.4f},{r['m_base']['psnr']:.2f},"
                    f"{r['m_base']['rmse']:.5f},{r['m_base']['diff_max']:.4f},"
                    f"{r['m_fit']['ssim']:.4f},{r['m_fit']['psnr']:.2f},"
                    f"{r['m_fit']['rmse']:.5f},{r['m_fit']['diff_max']:.4f}\n")
    print(f"[slab] wrote {csv}", flush=True)

    # Aggregate metrics
    base_ssims = np.array([r["m_base"]["ssim"] for r in rows])
    fit_ssims = np.array([r["m_fit"]["ssim"] for r in rows])
    base_psnrs = np.array([r["m_base"]["psnr"] for r in rows])
    fit_psnrs = np.array([r["m_fit"]["psnr"] for r in rows])
    base_rmses = np.array([r["m_base"]["rmse"] for r in rows])
    fit_rmses = np.array([r["m_fit"]["rmse"] for r in rows])
    print()
    print(f"=== AGGREGATE ({len(rows)} GT slices) ===")
    print(f"BASE  SSIM mean={base_ssims.mean():.4f} (range [{base_ssims.min():.4f}, {base_ssims.max():.4f}])  "
          f"PSNR mean={base_psnrs.mean():.2f} dB  RMSE mean={base_rmses.mean():.5f}")
    print(f"FIT   SSIM mean={fit_ssims.mean():.4f} (range [{fit_ssims.min():.4f}, {fit_ssims.max():.4f}])  "
          f"PSNR mean={fit_psnrs.mean():.2f} dB  RMSE mean={fit_rmses.mean():.5f}")
    print(f"Δ     ΔSSIM mean={(fit_ssims-base_ssims).mean():+.4f}  "
          f"ΔPSNR mean={(fit_psnrs-base_psnrs).mean():+.2f} dB  "
          f"ΔRMSE mean={(fit_rmses-base_rmses).mean()/base_rmses.mean()*100:+.1f}%")

    # ---- Plots -----------------------------------------------------------
    # 1. Slab profile curve
    offsets_mm = np.arange(-N_HALF, N_HALF + 1) * dv
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(offsets_mm, w_base_np, "o-", color="C0", lw=2,
            label=f"baseline (physical 5-mm overlap)  Σ={w_base_np.sum():.4f}")
    ax.plot(offsets_mm, w_fit, "s-", color="C1", lw=2,
            label=f"fitted (Adam L2+smooth)  Σ={w_fit.sum():.4f}")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("z offset from closest-to-truth sino slice (mm)", fontsize=11)
    ax.set_ylabel("slab weight w_i", fontsize=11)
    ax.set_title(f"L014 fitted slice-sensitivity profile (5-mm B30f truth)\n"
                  f"shared across {len(rows)} GT slices around the cone-beam centre",
                  fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout()
    out_profile = out_dir / "L014_slab_profile_fit.png"
    fig.savefig(out_profile, dpi=120)
    print(f"[slab] wrote {out_profile}", flush=True)

    # 2. Per-GT montage: 5 columns × len(rows) rows  (= 11 × 5 = 55 panels)
    fig2, ax2 = plt.subplots(len(rows), 5, figsize=(20, 3.8 * len(rows)))
    if len(rows) == 1:
        ax2 = ax2[None, :]
    for n, r in enumerate(rows):
        diff_base = r["slab_base"] - r["truth_np"]
        diff_fit = r["slab_fit"] - r["truth_np"]
        ax2[n, 0].imshow(r["truth_np"], cmap="gray", vmin=0, vmax=dr)
        ax2[n, 0].set_title(f"truth #{r['ti']}  pZ={r['pZ']:.2f}", fontsize=9)
        ax2[n, 1].imshow(np.clip(r["slab_base"], 0, None), cmap="gray", vmin=0, vmax=dr)
        ax2[n, 1].set_title(f"baseline slab (5-mm overlap)\n"
                              f"SSIM={r['m_base']['ssim']:.4f}  "
                              f"PSNR={r['m_base']['psnr']:.2f}dB  "
                              f"RMSE={r['m_base']['rmse']:.5f}", fontsize=9)
        ax2[n, 2].imshow(np.clip(r["slab_fit"], 0, None), cmap="gray", vmin=0, vmax=dr)
        ax2[n, 2].set_title(f"fitted slab\n"
                              f"SSIM={r['m_fit']['ssim']:.4f}  "
                              f"PSNR={r['m_fit']['psnr']:.2f}dB  "
                              f"RMSE={r['m_fit']['rmse']:.5f}", fontsize=9)
        ax2[n, 3].imshow(diff_base, cmap="seismic", vmin=-0.02, vmax=0.02)
        ax2[n, 3].set_title(f"diff baseline\nmax|·|={np.abs(diff_base).max():.4f}",
                             fontsize=9)
        ax2[n, 4].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
        ax2[n, 4].set_title(f"diff fitted\nmax|·|={np.abs(diff_fit).max():.4f}",
                             fontsize=9)
        for a in ax2[n]:
            a.set_xticks([]); a.set_yticks([])
    fig2.suptitle("L014 fulldose: per-GT comparison, baseline (5-mm overlap) vs fitted slab profile",
                  fontsize=11)
    fig2.tight_layout()
    out_montage = out_dir / "L014_slab_profile_montage.png"
    fig2.savefig(out_montage, dpi=110)
    print(f"[slab] wrote {out_montage}", flush=True)

    np.save(out_dir / "L014_slab_profile_w_fit.npy", w_fit)
    np.save(out_dir / "L014_slab_profile_w_base.npy", w_base_np)
    return 0


if __name__ == "__main__":
    sys.exit(main())
