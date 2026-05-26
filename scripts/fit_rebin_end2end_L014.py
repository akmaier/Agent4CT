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
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    truth = torch.from_numpy(truth_mu_np).to("cuda").float()
    pixel_sp = float(ds.PixelSpacing[0])
    print(f"[fit] nearest GT #{ti}  pZ={pZ:.2f}  target_pZ={target_pZ:.2f}  "
          f"(Δ={target_pZ - pZ:+.2f} mm)  pixel_sp={pixel_sp:.6f}", flush=True)

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

    # ---- Pre-compute closest helix indices per s_angle, PER SLAB Z SLICE ----
    # The picked-helix-index set changes per z slice (different "closest
    # readout per s_angle"). Compute once at init for each slab offset and
    # cache; Δz shifts (sub-mm) don't change which helix readout is
    # closest, so these caches stay valid during the fit.
    print(f"[fit] precomputing picked helix indices for {n_slab} slab slices …",
          flush=True)
    picked_per_slab = []
    for off in slab_offsets_mm:
        z_eff = target_source_z + off
        picks = precompute_picks(z_pos_sub, orig_idx, rotview, z_eff)
        picked_per_slab.append(picks)
        n_picked = (picks >= 0).sum().item()
        print(f"[fit]   slab off={off:+.1f} mm: {n_picked}/{rotview} picks, "
              f"|Δz| range [{(z_pos_sub[picks] - z_eff).abs().min().item():.4f}, "
              f"{(z_pos_sub[picks] - z_eff).abs().max().item():.4f}] mm",
              flush=True)
    # For backwards-compatible baseline (single-z), keep the central pick.
    picked_idx = picked_per_slab[n_slab // 2]

    # Precompute FFT2 radial grid for the filter
    fy = torch.fft.fftfreq(512, device="cuda").float()
    fx = torch.fft.fftfreq(512, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    dr = 0.05

    # ---- Baseline: nominal geometry, no fit ----
    with torch.no_grad():
        sino_nom = helical_ssr_torch(
            proj_flat, z_pos_sub, picked_idx, target_source_z,
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
    def forward():
        # SLAB integration in sino domain: compute n_slab z slices,
        # softmax-weight, sum. The FBP is linear so sino-domain slab
        # average ≡ image-domain slab average (we save N - 1 FBPs).
        w_slab = F.softmax(w_slab_logits, dim=0)                         # (n_slab,)
        sino_slab = None
        for k, off in enumerate(slab_offsets_mm):
            # z_eff is target_source_z + off + delta_z. Δz is the
            # learnable sub-mm shift; the per-slab pick set was computed
            # at z = target + off (no Δz), so we keep using that pick
            # but include Δz in dZ via the z_target argument.
            z_eff = target_source_z + off + delta_z
            sino_k = helical_ssr_torch(
                proj_flat, z_pos_sub, picked_per_slab[k], z_eff,
                sod, sdd, du, dv, u_centre_nom, v_centre_nom,
            )
            sino_slab = (w_slab[k] * sino_k) if sino_slab is None else (sino_slab + w_slab[k] * sino_k)
        # SLAB→ FBP
        sino_input = torch.flip(sino_slab, dims=[-1])[None, None]
        fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
        fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])
        # Radial frequency filter
        fft_fbp = torch.fft.fft2(fbp_2d)
        h_2d = radial_filter_2d(h_radial, rho, n_bins)
        filt_fft = torch.complex(h_2d * fft_fbp.real, h_2d * fft_fbp.imag)
        filt = torch.fft.ifft2(filt_fft).real
        # Intensity scale
        scaled = a * (filt - bg)
        # ReLU lower clip + soft upper clip at hi
        clipped = F.relu(scaled)
        clipped = torch.minimum(clipped, hi)
        return clipped, sino_slab, fbp_2d

    # ---- Adam loop ----
    # Learnable: rebin geometry (sod, sdd), z-shift (delta_z), slab
    # profile (w_slab_logits), radial filter (h_radial), intensity
    # (a, bg, hi). du/dv stay fixed at hardware.
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
        data_loss = ((clipped - truth) ** 2).mean()
        smooth_loss = ((h_radial[2:] - 2 * h_radial[1:-1] + h_radial[:-2]) ** 2).mean()
        total = data_loss + lam_h * smooth_loss
        total.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                m_it = calc_metrics(clipped.detach().cpu().numpy(),
                                      truth_mu_np, dr=dr)
            with torch.no_grad():
                w_show = F.softmax(w_slab_logits, dim=0).cpu().numpy()
            print(f"[fit] iter {it:4d}/{n_iters}  data_loss={data_loss.item():.3e}  "
                  f"sod={sod.item():.3f}  sdd={sdd.item():.3f}  Δz={delta_z.item():+.4f}  "
                  f"a={a.item():.3f} bg={bg.item():+.4f} hi={hi.item():.3f}  "
                  f"w_slab=[{','.join(f'{x:.2f}' for x in w_show)}]  "
                  f"SSIM={m_it['ssim']:.4f} PSNR={m_it['psnr']:.2f} dB",
                  flush=True)

    # ---- Final ----
    with torch.no_grad():
        clipped_f, sino_f, fbp_2d_f = forward()
        pred_np = clipped_f.cpu().numpy()
    m_fit = calc_metrics(pred_np, truth_mu_np, dr=dr)
    print()
    print("=== SUMMARY ===")
    print(f"BASELINE (nominal rebin + intensity_calibrate)")
    print(f"   SSIM={m_base['ssim']:.4f}  PSNR={m_base['psnr']:.2f} dB  "
          f"RMSE={m_base['rmse']:.5f}  diff_max={m_base['diff_max']:.4f}")
    print(f"FITTED (end-to-end gradient descent on rebin geometry)")
    print(f"   SSIM={m_fit['ssim']:.4f}  PSNR={m_fit['psnr']:.2f} dB  "
          f"RMSE={m_fit['rmse']:.5f}  diff_max={m_fit['diff_max']:.4f}")
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
    print(f"LEARNED POST-FBP:")
    print(f"   a={a.item():.4f}  bg={bg.item():+.5f}  hi={hi.item():.4f}")
    print(f"   H(ρ) range = [{h_radial.min().item():.3f}, {h_radial.max().item():.3f}]")

    # ---- Plots ----
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_base = fbp_nom_cal_np - truth_mu_np
    diff_fit  = pred_np - truth_mu_np

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
            "h_radial": h_radial.detach().cpu().numpy().tolist(),
        },
        "metrics_baseline": m_base, "metrics_fitted": m_fit,
        "target_pZ": target_pZ, "nearest_gt_pZ": pZ, "gt_index": ti,
    }, indent=2))
    print(f"[fit] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
