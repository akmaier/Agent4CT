#!/usr/bin/env python -u
"""FFS sign-flip ablation on the L014 single-GT (central) reconstruction.

Locks ALL parameters at the multi-GT fitted values (from job 762296
JSON) and sweeps the 9 combinations of (alpha_dz, alpha_drho) ∈
{-1, 0, +1}², applying:

    z_src_eff(idx) = z_positions[idx] + alpha_dz   · ffs_dz[idx]
    sod_eff(idx)   = sod_global       + alpha_drho · ffs_drho[idx]
    sdd_eff(idx)   = sdd_global       + alpha_drho · ffs_drho[idx]

Per combination, report SSIM/PSNR/RMSE/diff_max on the central GT
slice. Output as a 3×3 grid table + a 9-panel image montage so the
user can visually compare.

Rationale: the multi-GT fit already absorbed any AVERAGE geometric
bias into (sod, sdd). The FFS-dz / FFS-drho effects are PERIOD-2
oscillating biases — if they're not modelled, the residual contains
alternating-readout streaks that the rebin can't compensate. The
sign tells us which direction the source moves (or which DICOM tag
convention Mayo uses).
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
            except Exception: continue
        break
    truth_files.sort()
    return truth_files


def _mu(fp: Path):
    ds = pydicom.dcmread(str(fp))
    hu = (ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope)
          + float(ds.RescaleIntercept))
    return 0.02 * (1.0 + hu / 1000.0), ds


def helical_ssr_torch_ffs(proj_flat: torch.Tensor,
                            z_pos_eff: torch.Tensor,
                            sod_per_readout: torch.Tensor,
                            sdd_per_readout: torch.Tensor,
                            picked_idx: torch.Tensor,
                            z_target: float,
                            du: float, dv: float,
                            u_centre: float, v_centre: float) -> torch.Tensor:
    """SSR with per-readout sod/sdd (for FFS-drho) and effective z."""
    n_sub, nv, nu = proj_flat.shape
    rotview = picked_idx.shape[0]
    device = proj_flat.device

    proj_picked = proj_flat[picked_idx]                          # (rotview, nv, nu)
    i_u = torch.arange(nu, device=device, dtype=torch.float32)
    u_mm = (i_u - u_centre) * du                                  # (nu,)

    z_src_picked = z_pos_eff[picked_idx].to(torch.float32)
    sod_picked = sod_per_readout[picked_idx].to(torch.float32)
    sdd_picked = sdd_per_readout[picked_idx].to(torch.float32)
    dZ = z_src_picked - float(z_target)                           # (rotview,)

    # v_precise per (rotview, nu) — uses per-readout sod and sdd
    v_precise = dZ[:, None] * (u_mm[None, :] ** 2 + sdd_picked[:, None] ** 2) \
                / (sod_picked[:, None] * sdd_picked[:, None])

    v_idx = v_precise / dv + v_centre                             # (rotview, nu)
    v_floor = v_idx.floor().long().clamp(0, nv - 2)
    v_frac = (v_idx - v_floor.to(v_idx.dtype)).clamp(0.0, 1.0)
    in_range = (v_idx >= 0) & (v_idx <= (nv - 1))

    idx_s = torch.arange(rotview, device=device).view(rotview, 1).expand(rotview, nu)
    idx_u = torch.arange(nu,      device=device).view(1, nu).expand(rotview, nu)
    val_lo = proj_picked[idx_s, v_floor,     idx_u]
    val_hi = proj_picked[idx_s, v_floor + 1, idx_u]
    sample = val_lo * (1.0 - v_frac) + val_hi * v_frac
    sample = torch.where(in_range, sample, torch.zeros_like(sample))

    w_cos = sdd_picked[:, None] / torch.sqrt(
        u_mm[None, :] ** 2 + v_precise ** 2 + sdd_picked[:, None] ** 2
    )
    return sample * w_cos


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


def radial_filter_2d(h_radial: torch.Tensor, rho: torch.Tensor,
                      n_bins: int) -> torch.Tensor:
    rho_max = float(rho.max())
    bin_pos = (rho / rho_max) * (n_bins - 1)
    bin_lo = bin_pos.floor().long().clamp(0, n_bins - 1)
    bin_hi = (bin_lo + 1).clamp(0, n_bins - 1)
    bin_frac = (bin_pos - bin_lo.float())
    return h_radial[bin_lo] * (1 - bin_frac) + h_radial[bin_hi] * bin_frac


def compute_fov_mask(sod_g, sdd_g, du_g, n_det_g, r_img_mm, transition=1.0):
    """Geometric FoV radius: sod·sin(atan(n_det/2·du / sdd))."""
    half_det = (n_det_g / 2.0) * du_g
    fan_half = math.atan(half_det / sdd_g)
    r_fov_mm = sod_g * math.sin(fan_half)
    return torch.sigmoid((r_fov_mm - r_img_mm) / transition), r_fov_mm


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
    print(f"[abl] loading {blob_path} …", flush=True)
    blob = torch.load(blob_path, weights_only=False, map_location="cpu")

    # Verify FFS arrays are present
    assert "ffs_dz" in blob and "ffs_drho" in blob, "re-run cache with FFS arrays"

    proj_flat = blob["proj_flat"].to("cuda")
    z_pos = blob["z_positions"].to("cuda")
    orig_idx = blob["original_indices"].to("cuda")
    ffs_dz = blob["ffs_dz"].to("cuda")
    ffs_drho = blob["ffs_drho"].to("cuda")
    rotview = int(blob["rotview"])
    nu, nv = int(blob["nu"]), int(blob["nv"])
    du = float(blob["du"])
    dv = float(blob["dv"])
    target_source_z = float(blob["target_source_z"])
    target_pZ = -target_source_z
    angle_start = float(blob["angle_start_corrected"])

    print(f"[abl] ffs_dz   range [{float(ffs_dz.min()):.4f}, {float(ffs_dz.max()):.4f}] mm",
          flush=True)
    print(f"[abl] ffs_drho range [{float(ffs_drho.min()):.4f}, {float(ffs_drho.max()):.4f}] mm",
          flush=True)

    # ---- Load multi-GT-fitted params (job 762296) ----
    # Hard-code from the SUMMARY block:
    sod_global = 593.677
    sdd_global = 1086.801
    delta_z = -0.2685
    w_slab_np = np.array([0.0301, 0.2329, 0.1440, 0.1811, 0.1547, 0.2080, 0.0491],
                          dtype=np.float32)
    a_val = 0.8071
    bg_val = -0.00030
    hi_val = 0.0437
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
            if "delta_z" in rf: delta_z = float(rf["delta_z"])
            if "w_slab" in rf:
                w_slab_np = np.asarray(rf["w_slab"], dtype=np.float32)
            print(f"[abl] loaded multi-GT-fitted params from {fit_json}", flush=True)
        except Exception as e:
            print(f"[abl] WARN: could not parse {fit_json}: {e}", flush=True)
    else:
        print(f"[abl] WARN: {fit_json} missing — using hardcoded multi-GT values", flush=True)

    if h_radial_np is None:
        h_radial_np = np.ones(64, dtype=np.float32)
        print(f"[abl] WARN: no H(ρ) in JSON — using identity filter", flush=True)

    n_bins = len(h_radial_np)
    print(f"[abl] params: sod={sod_global:.3f}  sdd={sdd_global:.3f}  Δz={delta_z:+.4f}  "
          f"a={a_val:.4f}  bg={bg_val:+.5f}  hi={hi_val:.4f}  "
          f"H(ρ) n_bins={n_bins}  range=[{h_radial_np.min():.3f}, {h_radial_np.max():.3f}]",
          flush=True)
    print(f"[abl] w_slab = {[f'{x:.3f}' for x in w_slab_np]}", flush=True)

    # Truth (central GT only, no z-interp)
    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu_np, ds = _mu(fp)
    truth = torch.from_numpy(truth_mu_np).to("cuda").float()
    pixel_sp = float(ds.PixelSpacing[0])
    print(f"[abl] central GT #{ti}  pZ={pZ:.2f}  target_pZ={target_pZ:.2f}  "
          f"pixel_sp={pixel_sp:.6f}", flush=True)

    # FBP via PYRO-NN at fitted geometry
    fbp_geom = FanBeamGeometry.mayo_ldct_fitted(
        n_angles=rotview, n_det=nu,
        angle_start=angle_start, angle_end=angle_start + 2 * math.pi,
    )
    proj_fbp = PyronnFanBeamProjector(fbp_geom).to("cuda")
    proj_fbp._tensor_geom["detector_origin"] = (
        proj_fbp._tensor_geom["detector_origin"] + MAYO_LDCT_DET_OFFSET
    )

    # Tensors
    w_slab = torch.from_numpy(w_slab_np).to("cuda")
    h_radial = torch.from_numpy(h_radial_np).to("cuda")
    slab_offsets_mm = np.arange(-3, 4, dtype=np.float32)         # 7 bins, ±3 mm

    Himg, Wimg = truth.shape
    yy_pix = torch.arange(Himg, device="cuda", dtype=torch.float32)
    xx_pix = torch.arange(Wimg, device="cuda", dtype=torch.float32)
    yy_grid, xx_grid = torch.meshgrid(yy_pix, xx_pix, indexing="ij")
    cy_img = (Himg - 1) / 2.0
    cx_img = (Wimg - 1) / 2.0
    r_img_mm = torch.sqrt((yy_grid - cy_img) ** 2 + (xx_grid - cx_img) ** 2) * pixel_sp

    fy = torch.fft.fftfreq(Wimg, device="cuda").float()
    fx = torch.fft.fftfreq(Himg, device="cuda").float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(fyy ** 2 + fxx ** 2)

    u_centre_nom = (nu - 1) / 2.0
    v_centre_nom = (nv - 1) / 2.0
    dr = 0.05

    # ---- Forward function with FFS sign flags ----
    def forward(alpha_dz: float, alpha_drho: float):
        # Effective per-readout z and sod/sdd
        z_pos_eff = z_pos + alpha_dz * ffs_dz
        sod_per_readout = torch.full_like(ffs_drho, sod_global) + alpha_drho * ffs_drho
        sdd_per_readout = torch.full_like(ffs_drho, sdd_global) + alpha_drho * ffs_drho

        # Build the slab: for each slab offset, compute SSR and weight
        sino_slab = None
        for k_slab, off in enumerate(slab_offsets_mm):
            z_target = target_source_z + delta_z + float(off)
            picked = precompute_picks(z_pos_eff, orig_idx, rotview, z_target)
            sino_k = helical_ssr_torch_ffs(
                proj_flat, z_pos_eff, sod_per_readout, sdd_per_readout,
                picked, z_target, du, dv, u_centre_nom, v_centre_nom,
            )
            if sino_slab is None:
                sino_slab = w_slab[k_slab] * sino_k
            else:
                sino_slab = sino_slab + w_slab[k_slab] * sino_k

        # FBP
        sino_input = torch.flip(sino_slab, dims=[-1])[None, None]
        fbp_out = proj_fbp.fbp(sino_input, filter_name="ramlak")[0, 0]
        fbp_2d = torch.flip(torch.flip(fbp_out, dims=[0]), dims=[1])

        # Frequency filter
        fft_fbp = torch.fft.fft2(fbp_2d)
        h_2d = radial_filter_2d(h_radial, rho, n_bins)
        filt_fft = torch.complex(h_2d * fft_fbp.real, h_2d * fft_fbp.imag)
        filt = torch.fft.ifft2(filt_fft).real

        scaled = a_val * (filt - bg_val)
        clipped = F.relu(scaled)
        clipped = torch.minimum(clipped, torch.tensor(hi_val, device="cuda"))
        fov_mask, r_fov = compute_fov_mask(sod_global, sdd_global, du, nu, r_img_mm)
        clipped = clipped * fov_mask
        return clipped, fov_mask, r_fov

    # ---- Run 3×3 sweep ----
    sweep = [-1.0, 0.0, +1.0]
    rows = []
    print(f"\n=== FFS sign ablation (α_dz, α_drho) ∈ {{-1, 0, +1}}² ===", flush=True)
    print(f"  All other params LOCKED at multi-GT fitted values", flush=True)
    print(f"  Comparing central GT #{ti}, pZ={pZ:.2f} (NO z-interp on GT)\n", flush=True)
    with torch.no_grad():
        for s_dz in sweep:
            for s_drho in sweep:
                pred, mask, rfov = forward(s_dz, s_drho)
                pred_np = pred.cpu().numpy()
                m = calc_metrics(pred_np, truth_mu_np, dr=dr)
                rows.append({"alpha_dz": s_dz, "alpha_drho": s_drho,
                             "ssim": m["ssim"], "psnr": m["psnr"],
                             "rmse": m["rmse"], "diff_max": m["diff_max"],
                             "pred": pred_np})
                tag = "★" if (s_dz == 0 and s_drho == 0) else " "
                print(f"  α_dz={s_dz:+.0f}  α_drho={s_drho:+.0f}  {tag}  "
                      f"SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f} dB  "
                      f"RMSE={m['rmse']:.5f}  diff_max={m['diff_max']:.4f}",
                      flush=True)

    # Sort by PSNR descending
    rows_sorted = sorted(rows, key=lambda r: -r["psnr"])
    print(f"\n=== RANKED BY PSNR ===", flush=True)
    for k, r in enumerate(rows_sorted):
        baseline = (r["alpha_dz"] == 0 and r["alpha_drho"] == 0)
        tag = "  ← (no FFS correction baseline)" if baseline else ""
        marker = "★" if k == 0 else " "
        print(f"  {marker} #{k+1}: α_dz={r['alpha_dz']:+.0f}  α_drho={r['alpha_drho']:+.0f}  "
              f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB  "
              f"RMSE={r['rmse']:.5f}  diff_max={r['diff_max']:.4f}{tag}",
              flush=True)

    # ---- Plot 3×3 montage of pred diffs ----
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    rows_by_grid = {(int(r["alpha_dz"]), int(r["alpha_drho"])): r for r in rows}
    for i_dz, s_dz in enumerate(sweep):
        for j_drho, s_drho in enumerate(sweep):
            r = rows_by_grid[(int(s_dz), int(s_drho))]
            diff = r["pred"] - truth_mu_np
            ax = axes[i_dz, j_drho]
            ax.imshow(diff, cmap="seismic", vmin=-0.02, vmax=0.02)
            baseline = (s_dz == 0 and s_drho == 0)
            tag = "  (no FFS)" if baseline else ""
            ax.set_title(f"α_dz={int(s_dz):+d}  α_drho={int(s_drho):+d}{tag}\n"
                          f"SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f} dB  "
                          f"RMSE={r['rmse']:.5f}",
                          fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"L014 FFS sign ablation — central GT #{ti} (pZ={pZ:.2f} mm)\n"
                 f"all other params locked at multi-GT fitted values",
                 fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "L014_ffs_sign_ablation.png"
    fig.savefig(out_png, dpi=110)
    print(f"\n[abl] wrote {out_png}", flush=True)

    # JSON dump
    out_json = out_dir / "L014_ffs_sign_ablation.json"
    out_json.write_text(json.dumps({
        "results": [{k: v for k, v in r.items() if k != "pred"} for r in rows],
        "ranked": [(r["alpha_dz"], r["alpha_drho"], r["ssim"], r["psnr"], r["rmse"]) for r in rows_sorted],
        "central_gt": {"index": ti, "pZ": pZ},
        "fitted_params": {"sod": sod_global, "sdd": sdd_global, "delta_z": delta_z,
                          "a": a_val, "bg": bg_val, "hi": hi_val,
                          "w_slab": w_slab_np.tolist()},
    }, indent=2))
    print(f"[abl] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
