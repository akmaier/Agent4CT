#!/usr/bin/env python -u
"""Rigid 2D alignment of fitted L014 recon to truth.

Hypothesis: after all geometry corrections (pixel_spacing, sod, sdd,
det_spacing, Δz, slab, post-FBP, H(ρ), FFS-z), there may still be a
small residual rigid 2D misregistration (rotation + translation in
the image plane) between the fitted reconstruction and Mayo's B30f
truth — sub-pixel or sub-degree quantities that the radial-only
geometry fit cannot capture because they break rotational symmetry.

Method:
  1. Build the fitted prediction at the central L014 GT using the
     current production pipeline (Powell FBP + MAYO_LDCT_SSR_DEFAULTS
     + Δz/slab/post-FBP/H(ρ) from the multi-GT fit JSON).
  2. Optimize a 3-parameter rigid transform (θ, tx, ty) applied via
     `affine_grid` + `grid_sample` to the prediction. Loss = MSE
     against truth on FoV-masked region (smoother gradient than 1-SSIM
     for sub-pixel alignment).
  3. Report SSIM/PSNR/RMSE before and after the rigid transform, and
     write a side-by-side comparison + diff.

Output: console table, comparison PNG, JSON dump of (θ, tx, ty) +
metrics.
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


def precompute_picks(z_pos, original_indices, rotview, z_target):
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
    }


def build_fitted_pred(blob, raw_dir: Path, fit_json: Path):
    """Returns: pred (HxW tensor), truth (HxW tensor), truth_np, fov_mask, central GT info."""
    proj_flat = blob["proj_flat"].to("cuda")
    z_pos = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz = blob["ffs_dz"].to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    target_source_z = float(blob["target_source_z"])
    target_pZ = -target_source_z
    angle_start = float(blob["angle_start_corrected"])

    # FBP: Powell
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )

    # SSR: multi-GT
    sod_g = MAYO_LDCT_SSR_DEFAULTS["sod"]
    sdd_g = MAYO_LDCT_SSR_DEFAULTS["sdd"]
    du = MAYO_LDCT_SSR_DEFAULTS["du"]
    dv = MAYO_LDCT_SSR_DEFAULTS["dv"]
    delta_z = MAYO_LDCT_SSR_DEFAULTS["delta_z_mm"]
    alpha_dz = MAYO_LDCT_SSR_DEFAULTS["alpha_dz"]
    w_slab_np = np.asarray(MAYO_LDCT_SSR_DEFAULTS["w_slab"], dtype=np.float32)
    slab_offsets_mm = np.asarray(MAYO_LDCT_SSR_DEFAULTS["slab_offsets_mm"],
                                  dtype=np.float32)
    a_val  = MAYO_LDCT_SSR_DEFAULTS["post_fbp_a"]
    bg_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_bg"]
    hi_val = MAYO_LDCT_SSR_DEFAULTS["post_fbp_hi"]

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
            if "sod" in rf: sod_g = float(rf["sod"])
            if "sdd" in rf: sdd_g = float(rf["sdd"])
            sf = blob_j.get("slab_fitted", {})
            if "delta_z_mm" in sf: delta_z = float(sf["delta_z_mm"])
            if "w_slab" in sf:
                w_slab_np = np.asarray(sf["w_slab"], dtype=np.float32)
        except Exception as e:
            print(f"[align] WARN: {e}", flush=True)
    if h_radial_np is None:
        h_radial_np = np.ones(64, dtype=np.float32)
    n_bins = len(h_radial_np)

    # Truth
    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    truth = torch.from_numpy(truth_mu_np).to("cuda").float()

    # FoV mask at the CONSISTENT pixel_spacing = FBP pixel_spacing
    pixel_sp = fbp_geom.pixel_spacing
    Himg, Wimg = truth.shape
    yy_pix = torch.arange(Himg, device="cuda", dtype=torch.float32)
    xx_pix = torch.arange(Wimg, device="cuda", dtype=torch.float32)
    yy_grid, xx_grid = torch.meshgrid(yy_pix, xx_pix, indexing="ij")
    cy_img = (Himg - 1) / 2.0
    cx_img = (Wimg - 1) / 2.0
    r_img_mm = torch.sqrt((yy_grid - cy_img) ** 2 + (xx_grid - cx_img) ** 2) * pixel_sp
    half_det = (nu / 2.0) * du
    r_fov_mm = sod_g * math.sin(math.atan(half_det / sdd_g))
    fov_mask = torch.sigmoid((r_fov_mm - r_img_mm) / 1.0)

    fy = torch.fft.fftfreq(Wimg, device="cuda").float()
    fx = torch.fft.fftfreq(Himg, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    # FFS-z and picks
    z_pos_eff = z_pos + float(alpha_dz) * ffs_dz
    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0
    w_slab = torch.from_numpy(w_slab_np).to("cuda")
    h_radial = torch.from_numpy(h_radial_np).to("cuda")

    sino_slab = None
    for k_slab, off in enumerate(slab_offsets_mm):
        z_target = target_source_z + delta_z + float(off)
        picks = precompute_picks(z_pos_eff, orig_idx, rotview, z_target)
        sino_k = helical_ssr(
            proj_flat, z_pos_eff, picks, z_target,
            sod_g, sdd_g, du, dv, u_centre_nom, v_centre_nom,
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
    pred = clipped * fov_mask
    return pred, truth, truth_mu_np, fov_mask, ti, pZ, pixel_sp


def rigid_warp(img: torch.Tensor, theta_rad: torch.Tensor,
                tx_px: torch.Tensor, ty_px: torch.Tensor) -> torch.Tensor:
    """Apply rigid 2D transform (rotation θ + translation in pixels).

    Uses affine_grid + grid_sample with normalised [-1, +1] coordinates.
    img: (H, W) tensor.
    Returns: (H, W) warped tensor.
    """
    H, W = img.shape
    cos_t = torch.cos(theta_rad)
    sin_t = torch.sin(theta_rad)
    # Normalise pixel translation to [-1, +1] range of grid_sample.
    # PyTorch convention: align_corners=True → step = 2/(N-1)
    nx = 2.0 * tx_px / (W - 1)
    ny = 2.0 * ty_px / (H - 1)
    # affine_grid theta matrix is the INVERSE map from output → input.
    # We want "warp the image by (θ, tx, ty) when sampling output", so
    # output_coords = R · input_coords + t, equivalently
    # input_coords = R^T · (output_coords - t).
    # affine_grid wants `theta` such that grid = θ · [x, y, 1]^T at each
    # output pixel — this is the inverse of the transform we want.
    theta_mat = torch.stack([
        torch.stack([cos_t, -sin_t, -nx]),
        torch.stack([sin_t,  cos_t, -ny]),
    ]).unsqueeze(0)   # (1, 2, 3)
    grid = F.affine_grid(theta_mat, (1, 1, H, W), align_corners=True)
    out = F.grid_sample(
        img[None, None], grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )[0, 0]
    return out


def main() -> int:
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    fit_json = Path("/cluster/maier/Agent4CT/results/breast_debug/L014_rebin_end2end_fit.json")

    blob_path = sino_dir / "L014_proj_flat_peak.pt"
    print(f"[align] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")

    pred, truth, truth_mu_np, fov_mask, ti, pZ, pixel_sp = build_fitted_pred(
        blob, raw_dir, fit_json,
    )
    pred_np_before = pred.detach().cpu().numpy()
    print(f"[align] central GT #{ti}  pZ={pZ:+.2f}  pixel_sp={pixel_sp:.6f} mm",
          flush=True)
    m_before = calc_metrics(pred_np_before, truth_mu_np, dr=0.05)
    print(f"[align] BEFORE rigid align:  SSIM={m_before['ssim']:.4f}  "
          f"PSNR={m_before['psnr']:.2f} dB  RMSE={m_before['rmse']:.5f}",
          flush=True)

    # ---- Rigid 2D Adam fit ----
    pred_const = pred.detach()
    theta_rad = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    tx_px    = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    ty_px    = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))

    optimizer = torch.optim.Adam([theta_rad, tx_px, ty_px], lr=5e-3)
    n_iter = 800

    # Use a smoother loss for the rigid fit (MSE inside FoV).
    truth_inside = truth * fov_mask
    print(f"[align] starting Adam: {n_iter} iters, lr=5e-3, loss=MSE inside FoV",
          flush=True)
    for it in range(n_iter):
        optimizer.zero_grad(set_to_none=True)
        warped = rigid_warp(pred_const, theta_rad, tx_px, ty_px)
        warped_in = warped * fov_mask
        loss = F.mse_loss(warped_in, truth_inside)
        loss.backward()
        optimizer.step()
        if it % 100 == 0 or it == n_iter - 1:
            m_now = calc_metrics(warped.detach().cpu().numpy(), truth_mu_np, dr=0.05)
            theta_deg = float(theta_rad.detach().cpu()) * 180.0 / math.pi
            print(f"  iter {it:4d}/{n_iter}  loss={loss.item():.3e}  "
                  f"θ={theta_deg:+.4f}°  tx={tx_px.item():+.4f}  ty={ty_px.item():+.4f}  "
                  f"SSIM={m_now['ssim']:.4f}  PSNR={m_now['psnr']:.2f} dB",
                  flush=True)

    with torch.no_grad():
        warped_final = rigid_warp(pred_const, theta_rad, tx_px, ty_px)
    pred_np_after = warped_final.detach().cpu().numpy()
    m_after = calc_metrics(pred_np_after, truth_mu_np, dr=0.05)

    theta_deg = float(theta_rad.detach().cpu()) * 180.0 / math.pi
    tx_val = float(tx_px.detach().cpu())
    ty_val = float(ty_px.detach().cpu())
    # In mm for interpretability
    tx_mm = tx_val * pixel_sp
    ty_mm = ty_val * pixel_sp
    print(f"\n=== RIGID FIT RESULT ===", flush=True)
    print(f"  θ  = {theta_deg:+.4f}°  ({float(theta_rad):+.5e} rad)")
    print(f"  tx = {tx_val:+.4f} pixels  = {tx_mm:+.4f} mm")
    print(f"  ty = {ty_val:+.4f} pixels  = {ty_mm:+.4f} mm")
    print()
    print(f"=== METRICS BEFORE / AFTER ===", flush=True)
    print(f"  {'metric':<8s}  {'before':>10s}  {'after':>10s}  {'Δ':>10s}")
    print(f"  {'SSIM':<8s}  {m_before['ssim']:>10.4f}  {m_after['ssim']:>10.4f}  "
          f"{m_after['ssim']-m_before['ssim']:>+10.4f}")
    print(f"  {'PSNR(dB)':<8s}  {m_before['psnr']:>10.2f}  {m_after['psnr']:>10.2f}  "
          f"{m_after['psnr']-m_before['psnr']:>+10.2f}")
    print(f"  {'RMSE':<8s}  {m_before['rmse']:>10.5f}  {m_after['rmse']:>10.5f}  "
          f"{m_after['rmse']-m_before['rmse']:>+10.5f}")

    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    dr = 0.05
    ax[0, 0].imshow(truth_mu_np, cmap="gray", vmin=0, vmax=dr)
    ax[0, 0].set_title(f"truth (B30f)  pZ={pZ:+.2f} mm", fontsize=10)
    ax[0, 1].imshow(np.clip(pred_np_before, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[0, 1].set_title(f"fitted (before rigid)\n"
                       f"SSIM={m_before['ssim']:.4f}  PSNR={m_before['psnr']:.2f} dB",
                       fontsize=9)
    ax[0, 2].imshow(np.clip(pred_np_after, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[0, 2].set_title(f"fitted (after rigid: θ={theta_deg:+.3f}°,\n"
                       f"tx={tx_mm:+.3f} mm, ty={ty_mm:+.3f} mm)\n"
                       f"SSIM={m_after['ssim']:.4f}  PSNR={m_after['psnr']:.2f} dB",
                       fontsize=9)
    diff_before = pred_np_before - truth_mu_np
    diff_after  = pred_np_after  - truth_mu_np
    vmax_diff = max(abs(diff_before).max(), abs(diff_after).max(), 0.005) * 0.6
    ax[1, 0].imshow(np.zeros_like(truth_mu_np), cmap="gray")
    ax[1, 0].set_title(" ")
    ax[1, 1].imshow(diff_before, cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff)
    ax[1, 1].set_title(f"diff (before)  max|·|={abs(diff_before).max():.4f}",
                       fontsize=9)
    ax[1, 2].imshow(diff_after, cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff)
    ax[1, 2].set_title(f"diff (after)  max|·|={abs(diff_after).max():.4f}",
                       fontsize=9)
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        f"L014 central GT — rigid 2D alignment of fitted recon to truth\n"
        f"fit: θ={theta_deg:+.4f}°, tx={tx_mm:+.4f} mm, ty={ty_mm:+.4f} mm",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = out_dir / "L014_rigid_align.png"
    fig.savefig(out_png, dpi=130)
    print(f"\n[align] wrote {out_png}", flush=True)

    out_json = out_dir / "L014_rigid_align.json"
    out_json.write_text(json.dumps({
        "gt_idx": ti, "pZ_mm": pZ, "pixel_sp_mm": pixel_sp,
        "theta_deg": theta_deg, "theta_rad": float(theta_rad),
        "tx_px": tx_val, "ty_px": ty_val,
        "tx_mm": tx_mm, "ty_mm": ty_mm,
        "before": m_before, "after": m_after,
        "delta_ssim": m_after["ssim"] - m_before["ssim"],
        "delta_psnr_dB": m_after["psnr"] - m_before["psnr"],
    }, indent=2))
    print(f"[align] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
