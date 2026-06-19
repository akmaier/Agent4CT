#!/usr/bin/env python -u
"""Pixel-spacing ablation on L014 single-GT (central) reconstruction.

The earlier Powell scipy fit (FanBeamGeometry.mayo_ldct_fitted) picked
pixel_spacing = 0.700857 mm. The Mayo truth DICOM PixelSpacing tag
is 0.703125 mm. The original Wagner default was 0.700000 mm.

This ablation locks ALL other parameters at the multi-GT fitted
values (job 762296) and sweeps pixel_spacing over a range
{0.695, 0.698, 0.700, 0.700857, 0.703125, 0.705, 0.708} mm. For
each, report SSIM / PSNR / RMSE on the central GT (no z-interp).

Goal: settle whether the 0.700857 fitted value is genuinely better
than the truth 0.703125 (= "Powell absorbed some other residual into
pixel_spacing") or whether the metric peaks at the physically
correct 0.703125.

Also keeps pixel_sp CONSISTENT between FBP geometry and r_img_mm
(unlike the inconsistent setup in fit_rebin_end2end_L014.py).
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


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"

    blob_path = sino_dir / "L014_proj_flat_peak.pt"
    print(f"[abl] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")

    proj_flat = blob["proj_flat"].to("cuda")
    z_pos = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz = blob["ffs_dz"].to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    du = float(blob["du"])
    dv = float(blob["dv"])
    target_source_z = float(blob["target_source_z"])
    target_pZ = -target_source_z
    angle_start = float(blob["angle_start_corrected"])

    # ---- Multi-GT fitted SSR params (SLURM 762369, α_dz = +1 FFS-z) ----
    # SSR-step (sod, sdd) come from MAYO_LDCT_SSR_DEFAULTS — these are
    # the multi-GT joint Adam fit values. They are SEPARATE from the
    # FBP-step (sod, sdd) in `FanBeamGeometry.mayo_ldct_fitted()` (which
    # holds the Powell-fit values; do not collapse them — see the
    # SSR-vs-FBP-sod note in ddssl_ldct/geometry.py).
    # The Δz / slab / post-FBP knobs come from L014_rebin_end2end_fit.json
    # if available — that file is the multi-GT fit's persistent output.
    sod_global = MAYO_LDCT_SSR_DEFAULTS["sod"]    # 593.461 mm (SSR step)
    sdd_global = MAYO_LDCT_SSR_DEFAULTS["sdd"]    # 1086.831 mm (SSR step)
    delta_z = MAYO_LDCT_SSR_DEFAULTS["delta_z_mm"]
    w_slab_np = np.asarray(MAYO_LDCT_SSR_DEFAULTS["w_slab"], dtype=np.float32)
    a_val  = MAYO_LDCT_SSR_DEFAULTS["post_fbp_a"]
    bg_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_bg"]
    hi_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_hi"]
    fit_json = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_rebin_end2end_fit.json")
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
            print(f"[abl] WARN: {e}", flush=True)
    if h_radial_np is None:
        h_radial_np = np.ones(64, dtype=np.float32)

    n_bins = len(h_radial_np)
    print(f"[abl] locked params: sod={sod_global:.3f}  sdd={sdd_global:.3f}  "
          f"Δz={delta_z:+.4f}  a={a_val:.4f}", flush=True)

    # Use α_dz = +1 (the FFS-z winner)
    z_pos_eff = z_pos + 1.0 * ffs_dz

    # ---- Truth ----
    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    truth = torch.from_numpy(truth_mu_np).to("cuda").float()
    pixel_sp_truth = float(ds.PixelSpacing[0])
    print(f"[abl] central GT #{ti}  pZ={pZ:.2f}  truth PixelSpacing={pixel_sp_truth:.6f} mm",
          flush=True)

    # Tensors
    w_slab = torch.from_numpy(w_slab_np).to("cuda")
    h_radial = torch.from_numpy(h_radial_np).to("cuda")
    slab_offsets_mm = np.arange(-3, 4, dtype=np.float32)
    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0
    dr = 0.05

    # Precompute picks (done once, depends only on z_pos_eff and target_source_z)
    picks_per_slab = []
    for off in slab_offsets_mm:
        z_target = target_source_z + delta_z + float(off)
        picks = precompute_picks(z_pos_eff, orig_idx, rotview, z_target)
        picks_per_slab.append(picks)

    def forward_at_pixel_spacing(pixel_sp: float):
        """Build FBP geometry using `pixel_sp` everywhere consistently
        (FBP, FoV mask, image grid). Returns metrics + pred.

        The other FBP-geometry knobs (det_spacing, sod, sdd) come from
        `FanBeamGeometry.mayo_ldct_fitted()` — see ddssl_ldct/geometry.py
        for the current production defaults."""
        _base = FanBeamGeometry.mayo_ldct_fitted(
            n_angles=rotview, n_det=nu,
            angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
        )
        fbp_geom = FanBeamGeometry(
            image_size=_base.image_size,
            pixel_spacing=pixel_sp,            # the swept knob
            n_angles=_base.n_angles, n_det=_base.n_det,
            det_spacing=_base.det_spacing,
            sod=_base.sod, sdd=_base.sdd,
            angle_start=_base.angle_start, angle_end=_base.angle_end,
        )
        proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
        proj_fbp._tensor_geom["detector_origin"] = (
            proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
        )

        # r_img_mm and FoV at the SAME pixel_spacing (consistent)
        Himg, Wimg = truth.shape
        yy_pix = torch.arange(Himg, device="cuda", dtype=torch.float32)
        xx_pix = torch.arange(Wimg, device="cuda", dtype=torch.float32)
        yy_grid, xx_grid = torch.meshgrid(yy_pix, xx_pix, indexing="ij")
        cy_img = (Himg - 1) / 2.0
        cx_img = (Wimg - 1) / 2.0
        r_img_mm_local = torch.sqrt((yy_grid - cy_img) ** 2 + (xx_grid - cx_img) ** 2) * pixel_sp

        # Geometric FoV radius
        half_det = (nu / 2.0) * du
        r_fov_mm = sod_global * math.sin(math.atan(half_det / sdd_global))
        fov_mask = torch.sigmoid((r_fov_mm - r_img_mm_local) / 1.0)

        fy = torch.fft.fftfreq(Wimg, device="cuda").float()
        fx = torch.fft.fftfreq(Himg, device="cuda").float()
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
        rho = torch.sqrt(fyy ** 2 + fxx ** 2)

        # Slab
        sino_slab = None
        for k_slab, off in enumerate(slab_offsets_mm):
            z_target = target_source_z + delta_z + float(off)
            sino_k = helical_ssr(
                proj_flat, z_pos_eff, picks_per_slab[k_slab],
                z_target, sod_global, sdd_global,
                du, dv, u_centre_nom, v_centre_nom,
            )
            if sino_slab is None:
                sino_slab = w_slab[k_slab] * sino_k
            else:
                sino_slab = sino_slab + w_slab[k_slab] * sino_k

        # FBP
        sino_input = torch.flip(sino_slab, dims=[-1])[None, None]
        fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
        fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])

        # Filter
        fft_fbp = torch.fft.fft2(fbp_2d)
        h_2d = radial_filter_2d(h_radial, rho, n_bins)
        filt_fft = torch.complex(h_2d * fft_fbp.real, h_2d * fft_fbp.imag)
        filt = torch.fft.ifft2(filt_fft).real

        scaled = a_val * (filt - bg_val)
        clipped = F.relu(scaled)
        clipped = torch.minimum(clipped, torch.tensor(hi_val, device="cuda"))
        clipped = clipped * fov_mask
        return clipped, fov_mask

    # ---- Sweep ----
    sweep_values = [0.695, 0.698, 0.700, 0.700857, 0.703125, 0.705, 0.708]
    rows = []
    print(f"\n=== Pixel-spacing ablation (FBP geom + r_img_mm CONSISTENT) ===",
          flush=True)
    print(f"  All other params LOCKED at multi-GT fitted values "
          f"(+ α_dz = +1 FFS-z applied)", flush=True)
    print(f"  Comparing central GT #{ti}, pZ={pZ:.2f} (NO z-interp on GT)\n",
          flush=True)
    with torch.no_grad():
        for ps in sweep_values:
            pred, mask = forward_at_pixel_spacing(ps)
            pred_np = pred.cpu().numpy()
            m = calc_metrics(pred_np, truth_mu_np, dr=dr)
            note = ""
            if abs(ps - 0.700857) < 1e-6:
                note = "  ← Powell fitted (mayo_ldct_fitted)"
            elif abs(ps - 0.703125) < 1e-6:
                note = "  ← Mayo truth PixelSpacing"
            elif abs(ps - 0.700) < 1e-6:
                note = "  ← Wagner default"
            rows.append({"pixel_sp": ps, **m, "pred": pred_np})
            print(f"  pixel_sp = {ps:.6f} mm  "
                  f"SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f} dB  "
                  f"RMSE={m['rmse']:.5f}  diff_max={m['diff_max']:.4f}{note}",
                  flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r["psnr"])
    print(f"\n=== RANKED BY PSNR ===", flush=True)
    for k, r in enumerate(rows_sorted):
        note = ""
        if abs(r["pixel_sp"] - 0.700857) < 1e-6:
            note = "  (Powell fitted)"
        elif abs(r["pixel_sp"] - 0.703125) < 1e-6:
            note = "  (Mayo truth)"
        elif abs(r["pixel_sp"] - 0.700) < 1e-6:
            note = "  (Wagner default)"
        marker = "★" if k == 0 else " "
        print(f"  {marker} #{k+1}: pixel_sp = {r['pixel_sp']:.6f} mm  "
              f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB  "
              f"RMSE={r['rmse']:.5f}{note}",
              flush=True)

    # Plot — SSIM/PSNR curves
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    pss = np.array([r["pixel_sp"] for r in rows])
    ssims = np.array([r["ssim"] for r in rows])
    psnrs = np.array([r["psnr"] for r in rows])
    rmses = np.array([r["rmse"] for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(pss, ssims, "o-", lw=1.5)
    ax[0].axvline(0.700, color="C1", ls=":", label="Wagner 0.700")
    ax[0].axvline(0.700857, color="C2", ls=":", label="Powell 0.700857")
    ax[0].axvline(0.703125, color="C3", ls=":", label="Truth 0.703125")
    ax[0].set_xlabel("pixel_spacing (mm)"); ax[0].set_ylabel("SSIM")
    ax[0].set_title("SSIM vs pixel_spacing"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(pss, psnrs, "o-", lw=1.5, color="C1")
    ax[1].axvline(0.700, color="C1", ls=":")
    ax[1].axvline(0.700857, color="C2", ls=":")
    ax[1].axvline(0.703125, color="C3", ls=":")
    ax[1].set_xlabel("pixel_spacing (mm)"); ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("PSNR vs pixel_spacing"); ax[1].grid(alpha=0.3)
    ax[2].plot(pss, rmses * 1e4, "o-", lw=1.5, color="C3")
    ax[2].axvline(0.700, color="C1", ls=":")
    ax[2].axvline(0.700857, color="C2", ls=":")
    ax[2].axvline(0.703125, color="C3", ls=":")
    ax[2].set_xlabel("pixel_spacing (mm)"); ax[2].set_ylabel("RMSE × 10⁴")
    ax[2].set_title("RMSE vs pixel_spacing"); ax[2].grid(alpha=0.3)
    fig.suptitle(f"L014 pixel-spacing ablation, central GT #{ti}", fontsize=11)
    fig.tight_layout()
    out_curve = out_dir / "L014_pixel_spacing_ablation_curve.png"
    fig.savefig(out_curve, dpi=120)
    print(f"\n[abl] wrote {out_curve}", flush=True)

    # Image montage: diff per pixel_spacing
    n = len(rows)
    cols = min(n, 4)
    rows_grid = (n + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows_grid, cols, figsize=(4.5 * cols, 4.5 * rows_grid))
    if rows_grid == 1: axes2 = axes2[None, :]
    for k, r in enumerate(rows):
        i, j = divmod(k, cols)
        ax2 = axes2[i, j]
        diff = r["pred"] - truth_mu_np
        ax2.imshow(diff, cmap="seismic", vmin=-0.02, vmax=0.02)
        note = ""
        if abs(r["pixel_sp"] - 0.700857) < 1e-6: note = " (Powell)"
        elif abs(r["pixel_sp"] - 0.703125) < 1e-6: note = " (truth)"
        elif abs(r["pixel_sp"] - 0.700) < 1e-6: note = " (Wagner)"
        ax2.set_title(f"px={r['pixel_sp']:.6f}{note}\n"
                       f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB",
                       fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
    for k in range(n, rows_grid * cols):
        i, j = divmod(k, cols)
        axes2[i, j].axis("off")
    fig2.tight_layout()
    out_montage = out_dir / "L014_pixel_spacing_ablation_diffs.png"
    fig2.savefig(out_montage, dpi=120)
    print(f"[abl] wrote {out_montage}", flush=True)

    out_json = out_dir / "L014_pixel_spacing_ablation.json"
    out_json.write_text(json.dumps({
        "results": [{k: v for k, v in r.items() if k != "pred"} for r in rows],
        "ranked": [(r["pixel_sp"], r["ssim"], r["psnr"], r["rmse"]) for r in rows_sorted],
    }, indent=2))
    print(f"[abl] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
