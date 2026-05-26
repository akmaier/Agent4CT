#!/usr/bin/env python -u
"""End-to-end gradient-descent fit of the helical-to-fan REBIN geometry.

Replaces the previous polynomial-warp post-hoc correction with a
torch-differentiable helical-to-fan SSR rebin. The Noo-1999 SSR for a
single output (s_angle, z_target) is:

    z_src = z_positions[s_angle + k·rotview]      # closest k to z_target
    dZ    = z_src - z_target
    u_mm  = (i_u - u_centre) · du                 # detector coord, mm
    v_precise = dZ · (u² + sdd²) / (sod · sdd)    # Noo Eq. (1)
    sample = bilinear_v(proj_flat[idx, :, i_u], v_precise / dv + v_centre)
    w_cos  = sdd / sqrt(u² + v_precise² + sdd²)   # Noo Eq. (2)
    sino[s_angle, i_u] = sample · w_cos

Learnable parameters (all torch.Parameters, jointly optimised by Adam):

  Rebin geometry:
    sod, sdd, du, dv

  FBP geometry (fed into PYRO-NN; back-projection is differentiable in
  the sino input but the FBP geometry constants stay fixed at the
  data-driven optimum from job 762284):
    [held at mayo_ldct_fitted values]

  Post-FBP correction:
    h_radial : (n_bins,) radial frequency-filter response
    a, bg    : intensity-scale + background
    hi       : upper clip (after ReLU lower clip)

Loss:
    L = ‖clipped(scaled(filt(FBP(rebin(params))))) − truth‖²
      + λ_H ‖Δ²H‖²
      + λ_g (rebin_geom_penalty)

Usage:  python -u scripts/fit_rebin_end2end_L014.py
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

from ddssl_ldct.geometry import FanBeamGeometry, MAYO_LDCT_DET_OFFSET
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


# ---------------------------------------------------------------------------
# Torch-differentiable helical→fan SSR (single z_target).
# ---------------------------------------------------------------------------

def precompute_picks(z_positions_sub: torch.Tensor,
                      original_indices: torch.Tensor,
                      rotview: int, z_target: float):
    """For each s_angle 0..rotview-1, pick the subset index whose
    z_src is closest to z_target (within the helix readouts that have
    original_idx % rotview == s_angle).

    This is a non-differentiable lookup (depends only on z_positions
    and z_target, both fixed during fit). Done once, cached.
    """
    n_sub = z_positions_sub.shape[0]
    device = z_positions_sub.device
    s_angles_per_sub = (original_indices % rotview).long()
    picked = torch.full((rotview,), -1, dtype=torch.long, device=device)
    z_dist = torch.full((rotview,), float("inf"), device=device, dtype=z_positions_sub.dtype)
    for k in range(n_sub):
        s = int(s_angles_per_sub[k].item())
        d = abs(float(z_positions_sub[k].item()) - z_target)
        if d < float(z_dist[s].item()):
            z_dist[s] = d
            picked[s] = k
    if (picked < 0).any():
        n_miss = int((picked < 0).sum().item())
        print(f"[rebin] WARN: {n_miss} s_angles have no helix readout in window — "
              f"increase half_window_mm in caching script", flush=True)
    return picked


def helical_ssr_torch(proj_flat: torch.Tensor,        # (n_sub, nv, nu)
                       z_positions_sub: torch.Tensor,  # (n_sub,)
                       picked_idx: torch.Tensor,       # (rotview,)
                       z_target,                       # float OR 0-dim tensor
                       sod: torch.Tensor, sdd: torch.Tensor,
                       du: torch.Tensor, dv: torch.Tensor,
                       u_centre: float, v_centre: float) -> torch.Tensor:
    """Vectorised SSR. Returns sino[rotview, nu] for one z_target.
    If z_target is a tensor, dZ becomes differentiable in it — gradient
    flows from the SSR output through to z_target."""
    n_sub, nv, nu = proj_flat.shape
    rotview = picked_idx.shape[0]
    device = proj_flat.device

    # Gather the picked readouts: (rotview, nv, nu)
    proj_picked = proj_flat[picked_idx]                              # gather

    # u-coordinate in mm: (i_u - u_centre) · du
    i_u = torch.arange(nu, device=device, dtype=torch.float32)
    u_mm = (i_u - u_centre) * du                                     # (nu,)

    # z_src per s_angle: (rotview,)
    z_src = z_positions_sub[picked_idx].to(torch.float32)
    # z_target may be a Python float or a tensor; subtraction broadcasts.
    dZ = z_src - z_target                                            # (rotview,)

    # v_precise: (rotview, nu)
    v_precise = dZ[:, None] * (u_mm[None, :] ** 2 + sdd ** 2) / (sod * sdd)

    # v-index in proj_picked: bilinear in v
    v_idx = v_precise / dv + v_centre                                # (rotview, nu)

    v_floor = v_idx.floor().long().clamp(0, nv - 2)
    v_frac = (v_idx - v_floor.to(v_idx.dtype)).clamp(0.0, 1.0)       # (rotview, nu)

    # Mask: only sample where v_idx is in bounds [0, nv-1]
    in_range = (v_idx >= 0) & (v_idx <= (nv - 1))                    # (rotview, nu)

    idx_s = torch.arange(rotview, device=device).view(rotview, 1).expand(rotview, nu)
    idx_u = torch.arange(nu,      device=device).view(1, nu).expand(rotview, nu)
    val_lo = proj_picked[idx_s, v_floor,        idx_u]
    val_hi = proj_picked[idx_s, v_floor + 1,    idx_u]
    sample = val_lo * (1.0 - v_frac) + val_hi * v_frac               # (rotview, nu)
    sample = torch.where(in_range, sample, torch.zeros_like(sample))

    # Cosine weight (Noo Eq. 2): sdd / sqrt(u² + v² + sdd²)
    w_cos = sdd / torch.sqrt(u_mm[None, :] ** 2 + v_precise ** 2 + sdd ** 2)

    return sample * w_cos                                            # (rotview, nu)


def radial_filter_2d(h_radial: torch.Tensor, rho: torch.Tensor,
                      n_bins: int) -> torch.Tensor:
    rho_max = float(rho.max())
    bin_pos = (rho / rho_max) * (n_bins - 1)
    bin_lo = bin_pos.floor().long().clamp(0, n_bins - 1)
    bin_hi = (bin_lo + 1).clamp(0, n_bins - 1)
    bin_frac = (bin_pos - bin_lo.float())
    return h_radial[bin_lo] * (1 - bin_frac) + h_radial[bin_hi] * bin_frac


def calc_metrics(pred_np: np.ndarray, truth_np: np.ndarray, dr: float = 0.05):
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
    if not blob_path.exists():
        print(f"[fit] missing {blob_path} — run cache_proj_flat_L014.py first",
              file=sys.stderr)
        return 2
    print(f"[fit] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")

    proj_flat = blob["proj_flat"].to("cuda")                  # (n_sub, nv, nu)
    z_pos_sub = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    pitch_mm = float(blob["pitch_mm"])
    angle_start = float(blob["angle_start_corrected"])
    target_source_z = float(blob["target_source_z"])
    target_pZ = -target_source_z         # sign-flip to patient frame

    # Truth slice (single GT DICOM at -254.5 — note: target_source_z=256.86
    # corresponds to patient_z=-256.86, which sits between two GT slices.
    # For end-to-end fitting, just use the nearest GT — no interp here.)
    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti_center = int(np.argmin(np.abs(zs - target_pZ)))

    # Multi-GT: load N_GT consecutive truth slices around the cone-beam
    # centre. They share ALL learnable parameters (geometry, z-shift,
    # slab profile, kernel filter, intensity, FoV).
    N_GT = 10
    half_gt = N_GT // 2
    gt_indices = list(range(max(0, ti_center - half_gt),
                              min(len(truth_files), ti_center + half_gt)))
    if len(gt_indices) < N_GT:
        # Pad on whichever side has room
        if gt_indices[0] > 0:
            gt_indices = list(range(gt_indices[0] - (N_GT - len(gt_indices)),
                                      gt_indices[0])) + gt_indices
        else:
            gt_indices = gt_indices + list(range(gt_indices[-1] + 1,
                                                   gt_indices[-1] + 1 + (N_GT - len(gt_indices))))

    truth_list_np = []
    truth_pZ_list = []
    pixel_sp = None
    for ti in gt_indices:
        pZ_i, fp_i = truth_files[ti]
        mu_i, ds_i = _mu(fp_i)
        truth_list_np.append(mu_i)
        truth_pZ_list.append(pZ_i)
        if pixel_sp is None:
            pixel_sp = float(ds_i.PixelSpacing[0])
    truth_stack = torch.stack([torch.from_numpy(x).to("cuda").float() for x in truth_list_np], dim=0)  # (N_GT, H, W)
    # The central one (closest to target_pZ) is the diagnostic plot anchor.
    central_gt_idx = int(np.argmin([abs(z - target_pZ) for z in truth_pZ_list]))
    truth = truth_stack[central_gt_idx]
    truth_mu_np = truth_list_np[central_gt_idx]
    print(f"[fit] Multi-GT fit, N_GT={N_GT}", flush=True)
    for k, (idx, z) in enumerate(zip(gt_indices, truth_pZ_list)):
        marker = " ★" if k == central_gt_idx else ""
        print(f"[fit]    GT #{idx}: pZ={z:+.2f} mm{marker}", flush=True)
    print(f"[fit] pixel_sp={pixel_sp:.6f}  central GT for plots = #{gt_indices[central_gt_idx]}",
          flush=True)
    pZ = truth_pZ_list[central_gt_idx]
    ti = gt_indices[central_gt_idx]

    # ---- Geometry centres (kept fixed; mismatch is sub-pixel) ----
    u_centre_nom = (nu - 1) / 2.0     # PYRO-NN convention; the (nu - u0) flip is baked into proj_flat
    v_centre_nom = (nv - 1) / 2.0

    # ---- Learnable rebin parameters ----
    # Initialise at the values that the curved-to-flat step was done with
    # (= DICOM nominal).
    #
    # Physical rationale for which params are learnable:
    #   * sod, sdd: scanner alignment constants, manufacturer tolerance
    #     ±0.5 mm — Adam can find sub-mm corrections.
    #   * du, dv: detector-channel pitch is HARDWARE — cannot change at
    #     run time. They were also baked into the curved-to-flat step,
    #     so changing them retrospectively in the SSR (without redoing
    #     curved-to-flat) gives a degenerate compensation. Treat as
    #     hardware constants.
    # If you need to vary du/dv too, redo the curved-to-flat rebin in
    # torch and chain everything through one autograd graph. Out of
    # scope for this script.
    sod = torch.nn.Parameter(torch.tensor(float(blob["sod"]), device="cuda"))
    sdd = torch.nn.Parameter(torch.tensor(float(blob["sdd"]), device="cuda"))
    du  = torch.tensor(float(blob["du"]),  device="cuda")   # FIXED (hardware)
    dv  = torch.tensor(float(blob["dv"]),  device="cuda")   # FIXED (hardware)

    # FBP geometry kept fixed at the data-driven fitted values (job 762284)
    # — PYRO-NN's back-projection isn't differentiable in geometry, so we
    # treat it as a known operator.
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )

    n_bins = 64
    h_radial = torch.nn.Parameter(torch.ones(n_bins, device="cuda"))
    a   = torch.nn.Parameter(torch.tensor(1.0,  device="cuda"))
    bg  = torch.nn.Parameter(torch.tensor(0.0,  device="cuda"))
    hi  = torch.nn.Parameter(torch.tensor(0.05, device="cuda"))

    # ---- Image-space "geometric FoV" mask (new) ----
    # The fan-beam FBP can reconstruct pixels at radius up to
    #     r_max_fov = sod · sin(half_fan_angle)
    #               = sod · sin(atan(n_det/2 · du / sdd))
    # For Mayo L014: ≈ 237 mm. Beyond that the FBP cannot back-project
    # (the source's fan-beam doesn't cover those pixels), and Mayo's
    # truth is also 0 / background there. The corners of the 512²
    # image (radius up to 254 mm) include thin triangular wedges
    # outside the 237 mm circle — those are physically unreachable.
    #
    # The fan-beam FoV is DERIVED FROM the scanner geometry (sod, sdd,
    # du, n_det), NOT a free parameter. We use it as a soft sigmoid
    # loss WEIGHT (not a hard image mask) so the prediction image
    # stays natural (FBP corner noise still visible for diagnostic).
    fov_transition_mm = 1.0   # sigmoid steepness for the loss weight

    # Precompute per-pixel radius (in mm, image-frame) using truth's
    # PixelSpacing. Centered on the image centre (= where the body is in
    # both truth and FBP, regardless of scanner-frame iso offset).
    Himg, Wimg = truth.shape
    yy_pix = torch.arange(Himg, device="cuda", dtype=torch.float32)
    xx_pix = torch.arange(Wimg, device="cuda", dtype=torch.float32)
    yy_grid, xx_grid = torch.meshgrid(yy_pix, xx_pix, indexing="ij")
    cy_img = (Himg - 1) / 2.0
    cx_img = (Wimg - 1) / 2.0
    r_img_mm = torch.sqrt((yy_grid - cy_img) ** 2 + (xx_grid - cx_img) ** 2) * pixel_sp

    # n_det · du from the rebin geometry (n_det is fixed = 736; du is
    # fixed hardware = 1.28584).
    n_det_eff = float(nu)
    du_eff = float(du.item())

    def compute_fov_mask(sod_t: torch.Tensor, sdd_t: torch.Tensor) -> torch.Tensor:
        """Returns (H, W) soft sigmoid mask, 1 inside the fan-beam FoV,
        0 outside. Differentiable in (sod, sdd) via the geometric reach
        formula: r_max = sod · sin(atan(n_det/2 · du / sdd))."""
        half_fan = torch.atan(torch.tensor(n_det_eff / 2.0 * du_eff,
                                            device=sod_t.device) / sdd_t)
        r_max = sod_t * torch.sin(half_fan)
        return torch.sigmoid((r_max - r_img_mm) / fov_transition_mm)

    # ---- z-shift + slab profile (new) ----
    # Δz: sub-mm shift of the slab anchor (compensates the misalignment
    # between sino-z grid and GT-z grid; ~ 0.31 mm for Mayo L014).
    delta_z = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    # Slab: 7 z slices at offsets ±3 mm around the anchor, integrated
    # via learnable softmax weights w_slab (= effective slice profile).
    slab_offsets_mm = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    n_slab = len(slab_offsets_mm)
    slab_offsets = torch.tensor(slab_offsets_mm, device="cuda", dtype=torch.float32)
    # Initialise weights as uniform-over-central-5-mm (mimics what we
    # had before with the 5-mm physical-overlap baseline): bins -2..+2
    # at 0.2 each, -3 and +3 at 0.
    init_logits = torch.tensor([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0],
                                 device="cuda", dtype=torch.float32)
    w_slab_logits = torch.nn.Parameter(init_logits.clone())

    # ---- Pre-compute closest helix indices per (GT slice × slab bin) ----
    # For each of N_GT × n_slab combinations of (z_target_i, slab_offset_k),
    # find the helix readout closest to (target + slab + GT_offset). Δz
    # shifts during the fit are sub-mm, so picks stay valid.
    # Each GT slice has a source_z = -pZ_i; we map per-GT target source z's.
    target_source_z_per_gt = [-z for z in truth_pZ_list]   # source frame
    print(f"[fit] precomputing picked helix indices for {N_GT} GT × {n_slab} slab = "
          f"{N_GT * n_slab} combos …", flush=True)
    picked_per_gt_per_slab = []   # [N_GT][n_slab] → (rotview,) tensor
    for i_gt, z_tgt_i in enumerate(target_source_z_per_gt):
        per_slab = []
        for off in slab_offsets_mm:
            picks = precompute_picks(z_pos_sub, orig_idx, rotview, z_tgt_i + off)
            per_slab.append(picks)
        picked_per_gt_per_slab.append(per_slab)
    # Diagnostic for the central GT
    picked_per_slab = picked_per_gt_per_slab[central_gt_idx]
    picked_idx = picked_per_slab[n_slab // 2]
    print(f"[fit]   central GT #{ti}: |Δz| at slab=0 = "
          f"[{(z_pos_sub[picked_idx] - target_source_z_per_gt[central_gt_idx]).abs().min().item():.4f}, "
          f"{(z_pos_sub[picked_idx] - target_source_z_per_gt[central_gt_idx]).abs().max().item():.4f}] mm",
          flush=True)

    # Precompute FFT2 radial grid for the filter
    fy = torch.fft.fftfreq(512, device="cuda").float()
    fx = torch.fft.fftfreq(512, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    dr = 0.05

    # ---- Baseline: nominal geometry, no fit; central GT only for plots ----
    with torch.no_grad():
        sino_nom = helical_ssr_torch(
            proj_flat, z_pos_sub, picked_idx,
            target_source_z_per_gt[central_gt_idx],
            sod, sdd, du, dv, u_centre_nom, v_centre_nom,
        )
        # FBP via PYRO-NN (input is differentiable; output is image)
        # PYRO-NN's fbp expects (B, 1, A, D); sino_nom is (rotview, nu).
        # Also: flip the u-axis to match the validator's "Siemens flip".
        sino_input = torch.flip(sino_nom, dims=[-1])[None, None]
        fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
        # flipud + fliplr to match Mayo DICOM display orientation
        fbp_nom_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1]).clamp_min(0.0)
        fbp_nom_cal = intensity_calibrate(fbp_nom_2d, truth, display_max=dr)
    m_base = calc_metrics(fbp_nom_cal.cpu().numpy(), truth_mu_np, dr=dr)
    print(f"[fit] BASELINE (nominal rebin geometry + intensity_calibrate):  "
          f"SSIM={m_base['ssim']:.4f}  PSNR={m_base['psnr']:.2f} dB  "
          f"RMSE={m_base['rmse']:.5f}  diff_max={m_base['diff_max']:.4f}",
          flush=True)
    fbp_nom_cal_np = fbp_nom_cal.cpu().numpy()

    # ---- Forward pipeline (used in optimisation) ----
    def _forward_one_gt(i_gt: int):
        """Forward pass for one GT slice; uses the picked indices for
        that GT × each slab bin. Shared (sod, sdd, w_slab, h_radial,
        a, bg, hi) across all GTs."""
        w_slab = F.softmax(w_slab_logits, dim=0)
        sino_slab = None
        target_z_i = target_source_z_per_gt[i_gt]
        picks_per_slab_i = picked_per_gt_per_slab[i_gt]
        for k, off in enumerate(slab_offsets_mm):
            z_eff = target_z_i + off + delta_z
            sino_k = helical_ssr_torch(
                proj_flat, z_pos_sub, picks_per_slab_i[k], z_eff,
                sod, sdd, du, dv, u_centre_nom, v_centre_nom,
            )
            sino_slab = (w_slab[k] * sino_k) if sino_slab is None else (sino_slab + w_slab[k] * sino_k)
        sino_input = torch.flip(sino_slab, dims=[-1])[None, None]
        fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
        fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])
        fft_fbp = torch.fft.fft2(fbp_2d)
        h_2d = radial_filter_2d(h_radial, rho, n_bins)
        filt_fft = torch.complex(h_2d * fft_fbp.real, h_2d * fft_fbp.imag)
        filt = torch.fft.ifft2(filt_fft).real
        scaled = a * (filt - bg)
        clipped = F.relu(scaled)
        clipped = torch.minimum(clipped, hi)
        fov_mask_img = compute_fov_mask(sod, sdd)
        clipped = clipped * fov_mask_img
        return clipped, sino_slab, fbp_2d

    def forward():
        """Multi-GT forward. Returns (pred_stack[N_GT, H, W], sino[central], fbp[central])."""
        preds = []
        sino_central = None
        fbp_central = None
        for i_gt in range(N_GT):
            clipped_i, sino_i, fbp_i = _forward_one_gt(i_gt)
            preds.append(clipped_i)
            if i_gt == central_gt_idx:
                sino_central = sino_i
                fbp_central = fbp_i
        return torch.stack(preds, dim=0), sino_central, fbp_central

    # ---- Adam loop ----
    # Learnable: rebin geometry (sod, sdd), z-shift (delta_z), slab
    # profile (w_slab_logits), radial filter (h_radial), intensity
    # (a, bg, hi). du/dv stay fixed at hardware. The FoV-loss-mask
    # radius is geometry-derived (NOT a free param) — see compute_fov_mask.
    opt = torch.optim.Adam(
        [sod, sdd, delta_z, w_slab_logits, h_radial, a, bg, hi],
        lr=2e-3,
    )
    n_iters = 1500
    log_every = max(1, n_iters // 30)
    lam_h = 1e-4

    sod0, sdd0 = sod.item(), sdd.item()
    du0,  dv0  = float(du.item()), float(dv.item())   # fixed
    print(f"[fit] starting Adam, {n_iters} iters, lr=2e-3", flush=True)
    print(f"[fit] init  sod={sod0:.3f}  sdd={sdd0:.3f}  du={du0:.5f}(fixed)  dv={dv0:.5f}(fixed)",
          flush=True)
    for it in range(n_iters):
        opt.zero_grad()
        clipped, _, _ = forward()
        # FoV-weighted L2 — the mask is derived from the (learnable)
        # scanner geometry: r_max = sod·sin(atan(n_det/2·du/sdd)) ≈ 237 mm
        # for Mayo L014. Only the tiny triangles in the corners get
        # down-weighted; the table at r≈144 mm and body at r≈150 mm
        # are fully inside the mask (loss weight ≈ 1).
        fov_mask = compute_fov_mask(sod, sdd)
        # Sum L2 over ALL N_GT slices, weighted by FoV mask. clipped is
        # (N_GT, H, W); truth_stack is (N_GT, H, W); fov_mask is (H, W).
        sq_err = (clipped - truth_stack) ** 2                 # (N_GT, H, W)
        # Each GT contributes equally to the mean; mask weights pixels.
        w_eff = fov_mask[None].expand_as(sq_err)              # (N_GT, H, W)
        data_loss = (sq_err * w_eff).sum() / w_eff.sum().clamp_min(1.0)
        smooth_loss = ((h_radial[2:] - 2 * h_radial[1:-1] + h_radial[:-2]) ** 2).mean()
        total = data_loss + lam_h * smooth_loss
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                # Metric reported on the central GT (consistent w/ single-GT runs)
                m_it = calc_metrics(clipped[central_gt_idx].detach().cpu().numpy(),
                                      truth_mu_np, dr=dr)
            with torch.no_grad():
                w_show = F.softmax(w_slab_logits, dim=0).cpu().numpy()
            # Diagnostic: current geometric FoV radius (function of sod, sdd)
            with torch.no_grad():
                r_fov_now = sod.item() * math.sin(math.atan(n_det_eff/2 * du_eff / sdd.item()))
            print(f"[fit] iter {it:4d}/{n_iters}  data_loss={data_loss.item():.3e}  "
                  f"sod={sod.item():.3f}  sdd={sdd.item():.3f}  Δz={delta_z.item():+.4f}  "
                  f"a={a.item():.3f} bg={bg.item():+.4f} hi={hi.item():.3f}  "
                  f"r_fov_geom={r_fov_now:.2f}mm  "
                  f"w_slab=[{','.join(f'{x:.2f}' for x in w_show)}]  "
                  f"SSIM={m_it['ssim']:.4f} PSNR={m_it['psnr']:.2f} dB",
                  flush=True)

    # ---- Final ----
    with torch.no_grad():
        clipped_stack_f, sino_f, fbp_2d_f = forward()    # (N_GT, H, W)
        pred_stack_np = clipped_stack_f.cpu().numpy()
        pred_np = pred_stack_np[central_gt_idx]
        fov_mask_final = compute_fov_mask(sod, sdd).cpu().numpy()
    # Per-GT metrics (the FoV-mask is already applied inside `forward`).
    per_gt_metrics = []
    for i in range(N_GT):
        m_i = calc_metrics(pred_stack_np[i], truth_list_np[i], dr=dr)
        per_gt_metrics.append(m_i)
    # Central-GT metric for headline comparison
    pred_inside = pred_np * fov_mask_final
    truth_inside = truth_mu_np * fov_mask_final
    m_fit = calc_metrics(pred_inside, truth_inside, dr=dr)
    m_fit_full = calc_metrics(pred_np, truth_mu_np, dr=dr)
    # Aggregate across all GTs
    ssims = np.array([m["ssim"] for m in per_gt_metrics])
    psnrs = np.array([m["psnr"] for m in per_gt_metrics])
    rmses = np.array([m["rmse"] for m in per_gt_metrics])
    diff_maxs = np.array([m["diff_max"] for m in per_gt_metrics])
    print(f"\n=== PER-GT METRICS ({N_GT} slices) ===")
    print(f"{'idx':>4s}  {'pZ':>8s}  {'SSIM':>6s}  {'PSNR(dB)':>8s}  {'RMSE':>7s}  {'diff_max':>8s}")
    for k, (gi, pZ_k, m_k) in enumerate(zip(gt_indices, truth_pZ_list, per_gt_metrics)):
        marker = " ★" if k == central_gt_idx else "  "
        print(f"{gi:4d}{marker} {pZ_k:+8.2f}  {m_k['ssim']:.4f}  "
              f"{m_k['psnr']:6.2f}    {m_k['rmse']:.5f}  {m_k['diff_max']:.4f}",
              flush=True)
    print(f"  mean         {ssims.mean():.4f}  {psnrs.mean():6.2f}    "
          f"{rmses.mean():.5f}  {diff_maxs.mean():.4f}")
    print(f"  range  [{ssims.min():.4f}, {ssims.max():.4f}]  "
          f"[{psnrs.min():.2f}, {psnrs.max():.2f}]  "
          f"[{rmses.min():.5f}, {rmses.max():.5f}]")
    print()
    print("=== SUMMARY ===")
    print(f"BASELINE (nominal rebin + intensity_calibrate)")
    print(f"   SSIM={m_base['ssim']:.4f}  PSNR={m_base['psnr']:.2f} dB  "
          f"RMSE={m_base['rmse']:.5f}  diff_max={m_base['diff_max']:.4f}")
    print(f"FITTED (end-to-end gradient descent, on FOV-masked region)")
    print(f"   SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
          f"RMSE={m_fit['rmse']:.5f}  diff_max={m_fit['diff_max']:.4f}")
    print(f"FITTED (full 512² image, including FOV corners — diagnostic)")
    print(f"   SSIM={m_fit_full['ssim']:.4f}  PSNR={m_fit_full['psnr']:.2f} dB  "
          f"RMSE={m_fit_full['rmse']:.5f}  diff_max={m_fit_full['diff_max']:.4f}")
    print(f"Δ  ΔSSIM={m_fit['ssim']-m_base['ssim']:+.4f}  "
          f"ΔPSNR={m_fit['psnr']-m_base['psnr']:+.2f} dB  "
          f"ΔRMSE={(m_fit['rmse']-m_base['rmse'])/m_base['rmse']*100:+.1f}%")
    print()
    print(f"LEARNED REBIN GEOMETRY:")
    print(f"   sod  {sod0:.3f} → {sod.item():.3f}  (Δ={sod.item()-sod0:+.4f} mm = "
          f"{(sod.item()/sod0-1)*100:+.4f} %)")
    print(f"   sdd  {sdd0:.3f} → {sdd.item():.3f}  (Δ={sdd.item()-sdd0:+.4f} mm = "
          f"{(sdd.item()/sdd0-1)*100:+.4f} %)")
    print(f"   du   {du0:.5f}  (FIXED hardware)")
    print(f"   dv   {dv0:.5f}  (FIXED hardware)")
    print(f"LEARNED Z-SHIFT + SLAB PROFILE:")
    print(f"   Δz = {delta_z.item():+.4f} mm")
    with torch.no_grad():
        w_final = F.softmax(w_slab_logits, dim=0).cpu().numpy()
    print(f"   w_slab (offsets {slab_offsets_mm} mm):")
    for off, w in zip(slab_offsets_mm, w_final):
        print(f"     {off:+5.1f} mm:  {w:.4f}")
    print(f"   Σw = {w_final.sum():.4f}")
    r_fov_final = sod.item() * math.sin(math.atan(n_det_eff/2 * du_eff / sdd.item()))
    print(f"LEARNED POST-FBP:")
    print(f"   a={a.item():.4f}  bg={bg.item():+.5f}  hi={hi.item():.4f}")
    print(f"   geometric FoV radius (derived) = {r_fov_final:.2f} mm  "
          f"= sod·sin(atan(n_det/2·du/sdd))")
    print(f"   (image corner radius = {255.5 * pixel_sp:.2f} mm, so loss "
          f"down-weighted in the corner triangles outside r={r_fov_final:.0f} mm)")
    print(f"   H(ρ) range = [{h_radial.min().item():.3f}, {h_radial.max().item():.3f}]")

    # ---- Plots ----
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_base = fbp_nom_cal_np - truth_mu_np
    diff_fit  = (pred_np - truth_mu_np) * fov_mask_final

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(truth_mu_np, cmap="gray", vmin=0, vmax=dr)
    ax[0].set_title(f"truth GT#{ti}  pZ={pZ:.2f} mm", fontsize=10)
    ax[1].imshow(np.clip(fbp_nom_cal_np, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[1].set_title(f"FBP (nominal rebin + intensity_calibrate)\n"
                    f"SSIM={m_base['ssim']:.4f}  PSNR={m_base['psnr']:.2f} dB  "
                    f"RMSE={m_base['rmse']:.5f}", fontsize=9)
    ax[2].imshow(np.clip(pred_np, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[2].set_title(f"FBP (end-to-end fit: rebin geom + filter + scale + clip)\n"
                    f"SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
                    f"RMSE={m_fit['rmse']:.5f}", fontsize=9)
    ax[3].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax[3].set_title(f"diff after end-to-end fit\nmax|·|={np.abs(diff_fit).max():.4f}",
                    fontsize=10)
    for a_ax in ax: a_ax.set_xticks([]); a_ax.set_yticks([])
    fig.suptitle("L014 fulldose: end-to-end gradient-descent fit of helical→fan rebin geometry",
                 fontsize=10)
    fig.tight_layout()
    out_main = out_dir / "L014_rebin_end2end_fit.png"
    fig.savefig(out_main, dpi=120)
    print(f"[fit] wrote {out_main}", flush=True)

    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.5))
    ax2[0].imshow(diff_base, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[0].set_title(f"diff BEFORE (nominal rebin)\n"
                     f"max|·|={np.abs(diff_base).max():.4f}", fontsize=10)
    ax2[1].imshow(diff_fit, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[1].set_title(f"diff AFTER (end-to-end fit)\n"
                     f"max|·|={np.abs(diff_fit).max():.4f}", fontsize=10)
    for a_ax in ax2: a_ax.set_xticks([]); a_ax.set_yticks([])
    fig2.tight_layout()
    out_diff = out_dir / "L014_rebin_end2end_diff.png"
    fig2.savefig(out_diff, dpi=120)
    print(f"[fit] wrote {out_diff}", flush=True)

    # Multi-GT montage: 3 columns (truth | pred | diff) × N_GT rows
    fig3, ax3 = plt.subplots(N_GT, 3, figsize=(11, 3.5 * N_GT))
    if N_GT == 1:
        ax3 = ax3[None, :]
    for k, (gi, pZ_k, m_k) in enumerate(zip(gt_indices, truth_pZ_list, per_gt_metrics)):
        diff_k = (pred_stack_np[k] - truth_list_np[k]) * fov_mask_final
        is_centre = (k == central_gt_idx)
        tag = " ★" if is_centre else ""
        ax3[k, 0].imshow(truth_list_np[k], cmap="gray", vmin=0, vmax=dr)
        ax3[k, 0].set_title(f"truth GT#{gi}  pZ={pZ_k:+.2f}{tag}", fontsize=9)
        ax3[k, 1].imshow(np.clip(pred_stack_np[k], 0, None),
                          cmap="gray", vmin=0, vmax=dr)
        ax3[k, 1].set_title(f"end-to-end fit  "
                              f"SSIM={m_k['ssim']:.4f}  PSNR={m_k['psnr']:.2f} dB",
                              fontsize=9)
        ax3[k, 2].imshow(diff_k, cmap="seismic", vmin=-0.02, vmax=0.02)
        ax3[k, 2].set_title(f"diff  max|·|={np.abs(diff_k).max():.4f}", fontsize=9)
        for a_ax in ax3[k]:
            a_ax.set_xticks([]); a_ax.set_yticks([])
    fig3.suptitle(f"L014 end-to-end multi-GT fit  ({N_GT} slices share all params)",
                  fontsize=11)
    fig3.tight_layout()
    out_montage = out_dir / "L014_rebin_end2end_montage.png"
    fig3.savefig(out_montage, dpi=110)
    print(f"[fit] wrote {out_montage}", flush=True)

    out_json = out_dir / "L014_rebin_end2end_fit.json"
    out_json.write_text(json.dumps({
        "rebin_init": {"sod": sod0, "sdd": sdd0, "du": du0, "dv": dv0,
                        "du_status": "fixed", "dv_status": "fixed"},
        "rebin_fitted": {"sod": sod.item(), "sdd": sdd.item(),
                          "du": du0, "dv": dv0},
        "slab_fitted": {
            "delta_z_mm": delta_z.item(),
            "offsets_mm": slab_offsets_mm,
            "w_slab": w_final.tolist(),
        },
        "post_fbp_fitted": {
            "a": a.item(), "bg": bg.item(), "hi": hi.item(),
            "fov_radius_mm_derived": r_fov_final,
            "fov_radius_formula": "sod * sin(atan(n_det/2 * du / sdd))",
            "h_radial": h_radial.detach().cpu().numpy().tolist(),
        },
        "metrics_baseline": m_base,
        "metrics_fitted_central": m_fit,
        "metrics_fitted_full_central": m_fit_full,
        "per_gt": [
            {"gt_index": gt_indices[k], "pZ": truth_pZ_list[k],
              "ssim": per_gt_metrics[k]["ssim"],
              "psnr": per_gt_metrics[k]["psnr"],
              "rmse": per_gt_metrics[k]["rmse"],
              "diff_max": per_gt_metrics[k]["diff_max"],
              "is_central": (k == central_gt_idx)}
            for k in range(N_GT)
        ],
        "aggregate": {
            "ssim_mean": float(ssims.mean()), "ssim_min": float(ssims.min()),
            "ssim_max": float(ssims.max()),
            "psnr_mean": float(psnrs.mean()), "psnr_min": float(psnrs.min()),
            "psnr_max": float(psnrs.max()),
            "rmse_mean": float(rmses.mean()),
            "diff_max_mean": float(diff_maxs.mean()),
        },
        "target_pZ": target_pZ, "nearest_gt_pZ": pZ, "gt_index": ti,
    }, indent=2))
    print(f"[fit] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
