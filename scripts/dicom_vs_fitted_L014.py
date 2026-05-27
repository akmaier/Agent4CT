#!/usr/bin/env python -u
"""Side-by-side: DICOM-nominal vs multi-GT-fitted recon on L014 central GT.

Runs two pipelines on the SAME central GT slice and reports SSIM /
PSNR / RMSE / diff_max for both:

  (1) DICOM-NOMINAL:
       - FBP geometry  = DICOM tags  (sod=595.0, sdd=1085.6, ps=0.703125, ds=1.285839)
       - SSR geometry  = DICOM tags  (same)
       - No Δz, no slab averaging (pick-closest single-z), no FFS-z
         correction (α_dz = 0), no fitted radial filter, no post-FBP
         scaling beyond two-point linear intensity_calibrate.

  (2) FITTED:
       - FBP geometry  = `FanBeamGeometry.mayo_ldct_fitted()` (Powell)
       - SSR geometry  = `MAYO_LDCT_SSR_DEFAULTS`              (multi-GT)
       - All multi-GT learned knobs: Δz, 7-tap slab, α_dz=+1, H(ρ),
         and post-FBP (a, bg, hi).

Output: side-by-side comparison PNG + JSON with metric numbers.

This is the empirical answer to "how much of the SSIM gap is explained
by trusting DICOM tags vs the data-driven fit?"
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

from ddssl_ldct.geometry import (
    FanBeamGeometry, MAYO_LDCT_DET_OFFSET, MAYO_LDCT_SSR_DEFAULTS,
)
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
            except Exception: continue
        break
    truth_files.sort()
    return truth_files


def _mu(fp: Path):
    ds = pydicom.dcmread(str(fp))
    hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
          + float(ds.RescaleIntercept))
    return 0.02 * (1.0 + hu / 1000.0), ds


def precompute_picks(z_pos: torch.Tensor, original_indices: torch.Tensor,
                      rotview: int, z_target: float):
    n_sub = z_pos.shape[0]
    device = z_pos.device
    s_angles = (original_indices % rotview).long()
    picked = torch.full((rotview,), -1, dtype=torch.long, device=device)
    z_dist = torch.full((rotview,), float("inf"), device=device, dtype=z_pos.dtype)
    for k in range(n_sub):
        s = int(s_angles[k].item())
        d = abs(float(z_pos[k].item()) - z_target)
        if d < float(z_dist[s].item()):
            z_dist[s] = d
            picked[s] = k
    return picked


def helical_ssr(proj_flat, z_pos, picked, z_target, sod, sdd, du, dv,
                 u_centre, v_centre):
    n_sub, nv, nu = proj_flat.shape
    rotview = picked.shape[0]
    device = proj_flat.device
    proj_picked = proj_flat[picked]
    i_u = torch.arange(nu, device=device, dtype=torch.float32)
    u_mm = (i_u - u_centre) * du
    z_src = z_pos[picked].to(torch.float32)
    dZ = z_src - float(z_target)
    v_precise = dZ[:, None] * (u_mm[None, :] ** 2 + sdd ** 2) / (sod * sdd)
    v_idx = v_precise / dv + v_centre
    v_floor = v_idx.floor().long().clamp(0, nv - 2)
    v_frac = (v_idx - v_floor.to(v_idx.dtype)).clamp(0.0, 1.0)
    in_range = (v_idx >= 0) & (v_idx <= (nv - 1))
    idx_s = torch.arange(rotview, device=device).view(rotview, 1).expand(rotview, nu)
    idx_u = torch.arange(nu,      device=device).view(1, nu).expand(rotview, nu)
    val_lo = proj_picked[idx_s, v_floor,     idx_u]
    val_hi = proj_picked[idx_s, v_floor + 1, idx_u]
    sample = val_lo * (1.0 - v_frac) + val_hi * v_frac
    sample = torch.where(in_range, sample, torch.zeros_like(sample))
    w_cos = sdd / torch.sqrt(u_mm[None, :] ** 2 + v_precise ** 2 + sdd ** 2)
    return sample * w_cos


def radial_filter_2d(h_radial, rho, n_bins):
    rho_max = float(rho.max())
    bin_pos = (rho / rho_max) * (n_bins - 1)
    bin_lo = bin_pos.floor().long().clamp(0, n_bins - 1)
    bin_hi = (bin_lo + 1).clamp(0, n_bins - 1)
    bin_frac = (bin_pos - bin_lo.float())
    return h_radial[bin_lo] * (1 - bin_frac) + h_radial[bin_hi] * bin_frac


def calc_metrics(pred_np, truth_np, dr=0.05):
    pred_t = torch.from_numpy(np.clip(pred_np, 0, None)).to("cuda").float()[None, None]
    truth_t = torch.from_numpy(truth_np).to("cuda").float()[None, None]
    return {
        "ssim": float(ssim_fn(pred_t, truth_t, data_range=dr).cpu()),
        "psnr": float(psnr_fn(pred_t, truth_t, data_range=dr).cpu()),
        "rmse": float(((pred_t - truth_t) ** 2).mean().sqrt().cpu()),
        "diff_max": float(np.abs(pred_np - truth_np).max()),
    }


def build_dicom_nominal_pred(proj_flat, z_pos_eff, orig_idx, rotview, nu, nv,
                              du_dicom, dv_dicom, target_source_z, target_pZ,
                              truth, truth_mu_np, angle_start):
    """DICOM-tag-only pipeline. Single-z pick (no slab), no Δz, no
    fitted filter, intensity_calibrate only."""
    # DICOM-nominal geometry (sod, sdd, pixel_spacing, det_spacing).
    geom_dicom = FanBeamGeometry.mayo_ldct_nominal(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    sod = geom_dicom.sod
    sdd = geom_dicom.sdd
    pixel_sp = geom_dicom.pixel_spacing
    proj_fbp = PyronnFanBeamProjector(geom_dicom).to("cuda")

    # No FFS-z correction (DICOM-naive view ignores the FFS tags).
    picks = precompute_picks(z_pos_eff, orig_idx, rotview, target_source_z)
    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0
    sino = helical_ssr(
        proj_flat, z_pos_eff, picks,
        target_source_z, sod, sdd, du_dicom, dv_dicom,
        u_centre_nom, v_centre_nom,
    )

    sino_input = torch.flip(sino, dims=[-1])[None, None]
    fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
    fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])
    fbp_np = fbp_2d.cpu().numpy()

    # Standard two-point linear intensity calibration (used everywhere else
    # in the project for the "calibrated" metric).
    pred_t = torch.from_numpy(np.clip(fbp_np, 0, None)).to("cuda").float()[None, None]
    truth_t = truth[None, None]
    cal_t = intensity_calibrate(pred_t, truth_t).cpu().numpy()[0, 0]
    return cal_t, pixel_sp


def build_fitted_pred(proj_flat, z_pos, orig_idx, ffs_dz, rotview, nu, nv,
                       target_source_z, target_pZ, truth, truth_mu_np,
                       angle_start, fit_json: Path):
    """Powell FBP + multi-GT SSR + all multi-GT learned knobs."""
    # ---- FBP geometry: Powell defaults ----
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )

    # ---- SSR geometry: multi-GT defaults ----
    sod_global = MAYO_LDCT_SSR_DEFAULTS["sod"]
    sdd_global = MAYO_LDCT_SSR_DEFAULTS["sdd"]
    du_ssr = MAYO_LDCT_SSR_DEFAULTS["du"]
    dv_ssr = MAYO_LDCT_SSR_DEFAULTS["dv"]
    delta_z = MAYO_LDCT_SSR_DEFAULTS["delta_z_mm"]
    alpha_dz = MAYO_LDCT_SSR_DEFAULTS["alpha_dz"]
    w_slab_np = np.asarray(MAYO_LDCT_SSR_DEFAULTS["w_slab"], dtype=np.float32)
    slab_offsets_mm = np.asarray(MAYO_LDCT_SSR_DEFAULTS["slab_offsets_mm"],
                                  dtype=np.float32)
    a_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_a"]
    bg_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_bg"]
    hi_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_hi"]

    # Override with the latest multi-GT fit JSON if present (job 762369
    # writes this file; it is the source of truth for the learned knobs).
    h_radial_np = None
    if fit_json.exists():
        try:
            blob_j = json.loads(fit_json.read_text())
            pf = blob_j.get("post_fbp_fitted", {})
            if "h_radial" in pf:
                h_radial_np = np.asarray(pf["h_radial"], dtype=np.float32)
            if "a" in pf:  a_val  = float(pf["a"])
            if "bg" in pf: bg_val = float(pf["bg"])
            if "hi" in pf: hi_val = float(pf["hi"])
            rf = blob_j.get("rebin_fitted", {})
            if "sod" in rf: sod_global = float(rf["sod"])
            if "sdd" in rf: sdd_global = float(rf["sdd"])
            sf = blob_j.get("slab_fitted", {})
            if "delta_z_mm" in sf:
                delta_z = float(sf["delta_z_mm"])
            if "w_slab" in sf:
                w_slab_np = np.asarray(sf["w_slab"], dtype=np.float32)
        except Exception as e:
            print(f"[cmp] WARN reading fit JSON: {e}", flush=True)
    if h_radial_np is None:
        h_radial_np = np.ones(64, dtype=np.float32)
    n_bins = len(h_radial_np)

    w_slab = torch.from_numpy(w_slab_np).to("cuda")
    h_radial = torch.from_numpy(h_radial_np).to("cuda")
    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0

    # FFS-z correction (DICOM tag 0x7033, 0x100C; α_dz = +1 winner)
    z_pos_eff = z_pos + float(alpha_dz) * ffs_dz

    # Precompute picks for every slab offset (with the multi-GT Δz)
    picks_per_slab = []
    for off in slab_offsets_mm:
        z_target = target_source_z + delta_z + float(off)
        picks = precompute_picks(z_pos_eff, orig_idx, rotview, z_target)
        picks_per_slab.append(picks)

    Himg, Wimg = truth.shape
    yy_pix = torch.arange(Himg, device="cuda", dtype=torch.float32)
    xx_pix = torch.arange(Wimg, device="cuda", dtype=torch.float32)
    yy_grid, xx_grid = torch.meshgrid(yy_pix, xx_pix, indexing="ij")
    cy_img = (Himg - 1) / 2.0
    cx_img = (Wimg - 1) / 2.0
    pixel_sp = fbp_geom.pixel_spacing
    r_img_mm = torch.sqrt((yy_grid - cy_img) ** 2 + (xx_grid - cx_img) ** 2) * pixel_sp

    half_det = (nu / 2.0) * du_ssr
    r_fov_mm = sod_global * math.sin(math.atan(half_det / sdd_global))
    fov_mask = torch.sigmoid((r_fov_mm - r_img_mm) / 1.0)

    fy = torch.fft.fftfreq(Wimg, device="cuda").float()
    fx = torch.fft.fftfreq(Himg, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    sino_slab = None
    for k_slab, off in enumerate(slab_offsets_mm):
        z_target = target_source_z + delta_z + float(off)
        sino_k = helical_ssr(
            proj_flat, z_pos_eff, picks_per_slab[k_slab],
            z_target, sod_global, sdd_global,
            du_ssr, dv_ssr, u_centre_nom, v_centre_nom,
        )
        if sino_slab is None:
            sino_slab = w_slab[k_slab] * sino_k
        else:
            sino_slab = sino_slab + w_slab[k_slab] * sino_k

    sino_input = torch.flip(sino_slab, dims=[-1])[None, None]
    fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
    fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])

    fft_fbp = torch.fft.fft2(fbp_2d)
    h_2d = radial_filter_2d(h_radial, rho, n_bins)
    filt_fft = torch.complex(h_2d * fft_fbp.real, h_2d * fft_fbp.imag)
    filt = torch.fft.ifft2(filt_fft).real

    scaled = a_val * (filt - bg_val)
    clipped = F.relu(scaled)
    clipped = torch.minimum(clipped, torch.tensor(hi_val, device="cuda"))
    clipped = clipped * fov_mask
    return clipped.cpu().numpy(), pixel_sp


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    fit_json = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_rebin_end2end_fit.json")

    blob_path = sino_dir / "L014_proj_flat_peak.pt"
    print(f"[cmp] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")
    proj_flat = blob["proj_flat"].to("cuda")
    z_pos = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz = blob["ffs_dz"].to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    du_dicom = float(blob["du"])      # 1.285839
    dv_dicom = float(blob["dv"])      # 1.094723
    target_source_z = float(blob["target_source_z"])
    target_pZ = -target_source_z
    angle_start = float(blob["angle_start_corrected"])

    # Truth
    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    truth = torch.from_numpy(truth_mu_np).to("cuda").float()
    print(f"[cmp] central GT #{ti}  pZ={pZ:+.2f} mm  "
          f"truth PixelSpacing={float(ds.PixelSpacing[0]):.6f} mm", flush=True)
    print(f"[cmp] DICOM du={du_dicom:.6f}  dv={dv_dicom:.6f}  "
          f"(0x7029, 0x1002/0x1006)", flush=True)
    print()

    # (1) DICOM-nominal config
    print("[cmp] === (1) DICOM-NOMINAL CONFIG ===", flush=True)
    print("  FBP geom = DICOM tags (sod=595.0, sdd=1085.6, ps=0.703125, ds=1.285839)")
    print("  SSR geom = DICOM tags (same)")
    print("  Δz = 0  •  no slab averaging  •  α_dz = 0 (no FFS-z correction)")
    print("  No fitted H(ρ), no post-FBP scaling (intensity_calibrate only)")
    pred_dicom, ps_dicom = build_dicom_nominal_pred(
        proj_flat, z_pos, orig_idx, rotview, nu, nv,
        du_dicom, dv_dicom, target_source_z, target_pZ,
        truth, truth_mu_np, angle_start,
    )
    m_dicom = calc_metrics(pred_dicom, truth_mu_np, dr=0.05)
    print(f"  → SSIM={m_dicom['ssim']:.4f}  PSNR={m_dicom['psnr']:.2f} dB  "
          f"RMSE={m_dicom['rmse']:.5f}  diff_max={m_dicom['diff_max']:.4f}",
          flush=True)
    print()

    # (2) Fitted config
    print("[cmp] === (2) FITTED CONFIG (Powell FBP + multi-GT SSR) ===", flush=True)
    print("  FBP geom = mayo_ldct_fitted() Powell  (ps=0.700857, ds=1.285044,")
    print("                                          sod=595.362, sdd=1086.803)")
    print("  SSR geom = MAYO_LDCT_SSR_DEFAULTS     (sod=593.461, sdd=1086.831,")
    print("                                          du=1.285839, dv=1.094723)")
    print("  Δz=-0.578 mm  •  7-tap slab  •  α_dz=+1 FFS-z  •  H(ρ) filter")
    print("  Post-FBP a,bg,hi from multi-GT fit JSON")
    pred_fit, ps_fit = build_fitted_pred(
        proj_flat, z_pos, orig_idx, ffs_dz, rotview, nu, nv,
        target_source_z, target_pZ, truth, truth_mu_np, angle_start,
        fit_json,
    )
    m_fit = calc_metrics(pred_fit, truth_mu_np, dr=0.05)
    print(f"  → SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
          f"RMSE={m_fit['rmse']:.5f}  diff_max={m_fit['diff_max']:.4f}",
          flush=True)
    print()

    # ---- Summary ----
    print("=== SIDE-BY-SIDE ===", flush=True)
    print(f"{'config':<22s}  {'SSIM':>6s}  {'PSNR(dB)':>8s}  {'RMSE':>8s}  {'|diff|max':>9s}")
    print(f"{'DICOM-nominal':<22s}  "
          f"{m_dicom['ssim']:.4f}  {m_dicom['psnr']:6.2f}    "
          f"{m_dicom['rmse']:.5f}  {m_dicom['diff_max']:.4f}")
    print(f"{'Fitted (Powell+multi-GT)':<22s}  "
          f"{m_fit['ssim']:.4f}  {m_fit['psnr']:6.2f}    "
          f"{m_fit['rmse']:.5f}  {m_fit['diff_max']:.4f}")
    print(f"{'Δ (fitted - DICOM)':<22s}  "
          f"+{m_fit['ssim']-m_dicom['ssim']:.4f}  "
          f"+{m_fit['psnr']-m_dicom['psnr']:.2f} dB    "
          f"{(m_fit['rmse']-m_dicom['rmse'])/m_dicom['rmse']*100:+.1f}%  "
          f"{m_fit['diff_max']-m_dicom['diff_max']:+.4f}")

    # ---- Plot ----
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    dr = 0.05
    ax[0, 0].imshow(truth_mu_np, cmap="gray", vmin=0, vmax=dr)
    ax[0, 0].set_title(f"truth (B30f)  pZ={pZ:+.2f} mm", fontsize=10)
    ax[0, 1].imshow(np.clip(pred_dicom, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[0, 1].set_title(f"(1) DICOM-nominal\n"
                       f"SSIM={m_dicom['ssim']:.4f}  PSNR={m_dicom['psnr']:.2f} dB",
                       fontsize=9)
    ax[0, 2].imshow(np.clip(pred_fit, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[0, 2].set_title(f"(2) Fitted (Powell + multi-GT)\n"
                       f"SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB",
                       fontsize=9)
    diff_dicom = pred_dicom - truth_mu_np
    diff_fit   = pred_fit   - truth_mu_np
    vmax_diff = max(abs(diff_dicom).max(), abs(diff_fit).max(), 0.005) * 0.6
    ax[1, 0].imshow(np.zeros_like(truth_mu_np), cmap="gray")
    ax[1, 0].set_title(" ", fontsize=10)
    ax[1, 1].imshow(diff_dicom, cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff)
    ax[1, 1].set_title(f"diff (1) — max|·|={abs(diff_dicom).max():.4f}",
                       fontsize=9)
    ax[1, 2].imshow(diff_fit, cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff)
    ax[1, 2].set_title(f"diff (2) — max|·|={abs(diff_fit).max():.4f}",
                       fontsize=9)
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        "L014 central GT — DICOM-nominal vs multi-GT-fitted recon\n"
        "(same proj data, same evaluate_calibrated; only the recon pipeline differs)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = out_dir / "L014_dicom_vs_fitted.png"
    fig.savefig(out_png, dpi=130)
    print(f"\n[cmp] wrote {out_png}", flush=True)

    out_json = out_dir / "L014_dicom_vs_fitted.json"
    out_json.write_text(json.dumps({
        "gt_idx": ti, "pZ_mm": pZ,
        "dicom_nominal": m_dicom,
        "fitted": m_fit,
        "delta_ssim": m_fit["ssim"] - m_dicom["ssim"],
        "delta_psnr_dB": m_fit["psnr"] - m_dicom["psnr"],
        "delta_rmse_pct": (m_fit["rmse"] - m_dicom["rmse"]) / m_dicom["rmse"] * 100,
    }, indent=2))
    print(f"[cmp] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
