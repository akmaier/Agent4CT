"""Sweep image-axis and view-sweep conventions to find which matches Sidky's
val_fbp128.h5 exactly. After v2 of SiddonFanBeamProjector the scale is
matched (k≈0.989), but rel-L2 of FBP vs val_fbp128 is still ~25%. That
residual is structural — likely a row-convention or rotation-direction
mismatch.

Variants tested:
  (a) y_up vs y_down         (image row 0 = top  vs  image row 0 = bottom)
  (b) CCW   vs CW            (view β increasing CCW  vs  CW about iso)
  (c) det+  vs det-          (detector u = (+cos β, +sin β)  vs  (−cos β, −sin β))

8 combinations. Also runs a self-round-trip
  rt(truth) := SiddonFBP(SiddonForward(truth))
to confirm the projector is internally consistent for each convention.

Reports table of rel-L2 (vs val_fbp128, raw) and SSIM (vs truth, cal'd) for
all 8 combos + the self-round-trip SSIM/rel-L2 of cal-rt vs truth.

The best combo (closest to val_fbp128) is the one matching Sidky's convention.
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin  # noqa
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.siddon_projector import SiddonFanBeamProjector
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=360/1024, sod=500.0, sdd=1000.0)
DISPLAY_MAX = 0.5
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


def patch_projector(proj: SiddonFanBeamProjector, *, y_up: bool, ccw: bool, det_pos: bool):
    """Reinitialise projector buffers for a given convention triple.

    Implementation: we keep the rest of the class intact and just rewrite
    the cached source / detector positions and re-monkey-patch
    ``_siddon_segments`` to flip the row convention. ``ccw`` controls the
    sign of the angle sweep; ``det_pos`` controls the sign of the u-axis.
    """
    g = proj.geom
    s = proj.length_unit_scale
    sod, sdd = float(g.sod) * s, float(g.sdd) * s
    d_spacing = float(g.det_spacing) * s
    A, N_det = g.n_angles, g.n_det
    betas = torch.linspace(float(g.angle_start), float(g.angle_end), A + 1,
                            dtype=torch.float64)[:-1]
    if not ccw:
        betas = -betas
    sin_b, cos_b = torch.sin(betas), torch.cos(betas)
    sx = -sin_b * sod
    sy =  cos_b * sod
    cx =  sin_b * (sdd - sod)
    cy = -cos_b * (sdd - sod)
    sign = 1.0 if det_pos else -1.0
    ux = sign * cos_b
    uy = sign * sin_b
    d_offsets = (torch.arange(N_det, dtype=torch.float64) - (N_det - 1) / 2.0) * d_spacing
    dx_full = cx[:, None] + d_offsets[None, :] * ux[:, None]
    dy_full = cy[:, None] + d_offsets[None, :] * uy[:, None]
    sx_full = sx[:, None].expand_as(dx_full)
    sy_full = sy[:, None].expand_as(dy_full)
    device = proj._sx.device
    for name, t in (("sx", sx_full), ("sy", sy_full), ("dx", dx_full), ("dy", dy_full)):
        getattr(proj, f"_{name}")[:] = t.reshape(-1).contiguous().to(torch.float32).to(device)
    proj._y_up = y_up        # consumed by patched _siddon_segments below


# Patch row computation: capture original and bind ``y_up`` from the projector.
_orig_segments = SiddonFanBeamProjector._siddon_segments


def _siddon_segments_y_aware(self, ray_slice: slice):
    """Override row index computation: row 0 may be top (y_up=True, default)
    or bottom (y_up=False)."""
    geom = self.geom
    N = int(geom.image_size)
    delta = self._pixel_spacing_scaled
    device = self._sx.device
    dtype = self._sx.dtype
    sx = self._sx[ray_slice]; sy = self._sy[ray_slice]
    dx = self._dx[ray_slice]; dy = self._dy[ray_slice]
    vx = dx - sx; vy = dy - sy
    eps = torch.full_like(vx, 1e-9)
    vx_safe = torch.where(torch.abs(vx) < 1e-12, eps, vx)
    vy_safe = torch.where(torch.abs(vy) < 1e-12, eps, vy)
    k = torch.arange(N + 1, device=device, dtype=dtype)
    grid = -N / 2.0 * delta + k * delta
    alpha_x = (grid[None, :] - sx[:, None]) / vx_safe[:, None]
    alpha_y = (grid[None, :] - sy[:, None]) / vy_safe[:, None]
    alphas = torch.cat([alpha_x, alpha_y], dim=-1)
    a_x_min = torch.minimum(alpha_x[:, 0], alpha_x[:, -1])
    a_x_max = torch.maximum(alpha_x[:, 0], alpha_x[:, -1])
    a_y_min = torch.minimum(alpha_y[:, 0], alpha_y[:, -1])
    a_y_max = torch.maximum(alpha_y[:, 0], alpha_y[:, -1])
    a_min = torch.maximum(a_x_min, a_y_min)
    a_max = torch.minimum(a_x_max, a_y_max)
    alphas_sorted, _ = torch.sort(alphas, dim=-1)
    alpha_mid = 0.5 * (alphas_sorted[:, 1:] + alphas_sorted[:, :-1])
    d_alpha = alphas_sorted[:, 1:] - alphas_sorted[:, :-1]
    valid = (alpha_mid >= a_min[:, None]) & (alpha_mid <= a_max[:, None]) & (d_alpha > 0)
    px = sx[:, None] + alpha_mid * vx[:, None]
    py = sy[:, None] + alpha_mid * vy[:, None]
    i_col = torch.floor((px + (N * delta) / 2.0) / delta).to(torch.int64)
    y_up = getattr(self, "_y_up", True)
    if y_up:
        i_row = torch.floor(((N * delta) / 2.0 - py) / delta).to(torch.int64)
    else:
        i_row = torch.floor((py + (N * delta) / 2.0) / delta).to(torch.int64)
    in_bounds = (i_col >= 0) & (i_col < N) & (i_row >= 0) & (i_row < N)
    valid = valid & in_bounds
    i_col = i_col.clamp(0, N - 1)
    i_row = i_row.clamp(0, N - 1)
    ray_len = torch.sqrt(vx * vx + vy * vy)
    weights = d_alpha * ray_len[:, None] * valid.to(dtype)
    return weights, i_row, i_col


SiddonFanBeamProjector._siddon_segments = _siddon_segments_y_aware


def cal_metrics(pred, truth, dmax=DISPLAY_MAX):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    return (
        float(ssim(pc, truth, data_range=dmax).cpu()),
        float(psnr(pc, truth, data_range=dmax).cpu()),
        float(((pc - truth) ** 2).mean().sqrt().cpu()),
        pc,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth_np = f["image"][:1]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino_np = f["sino"][:1]
    with h5py.File(data / "val_fbp128.h5", "r") as f: fbp_np = f["image"][:1]

    truth = torch.from_numpy(truth_np).float().to(device).unsqueeze(1)
    sino_sidky = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)
    fbp_sidky = torch.from_numpy(fbp_np).float().to(device).unsqueeze(1)
    f_k = fbp_sidky[0, 0].cpu().numpy()
    t_np = truth_np[0]

    geom = FanBeamGeometry(**GEOM)
    proj = SiddonFanBeamProjector(geom, ray_batch=8192, length_unit_scale=0.1).to(device)

    rows = []  # (y_up, ccw, det_pos, raw_l2_vs_fbp128, best_k_l2, cal_ssim_vs_truth, cal_ssim_vs_fbp, ssim_rt_vs_truth, raw_l2_rt_vs_truth)
    cal_recons = {}
    cal_rts = {}

    for y_up in (True, False):
        for ccw in (True, False):
            for det_pos in (True, False):
                patch_projector(proj, y_up=y_up, ccw=ccw, det_pos=det_pos)

                # FBP of Sidky's sinogram with this convention
                with torch.no_grad():
                    fbp_ours = proj.fbp(sino_sidky)
                f_o = fbp_ours[0, 0].cpu().numpy()
                raw_l2 = float(np.linalg.norm(f_o - f_k) / np.linalg.norm(f_k))
                k = float((f_o * f_k).sum() / max((f_o * f_o).sum(), 1e-12))
                k_l2 = float(np.linalg.norm(k * f_o - f_k) / np.linalg.norm(f_k))
                ss_f, _, _, fbp_cal = cal_metrics(fbp_ours, fbp_sidky)
                ss_t, ps_t, rm_t, fbp_cal_t = cal_metrics(fbp_ours, truth)

                # Self-round-trip: SiddonFBP(SiddonForward(truth))
                with torch.no_grad():
                    sino_rt = proj.forward_project(truth)
                    fbp_rt = proj.fbp(sino_rt)
                rt_np = fbp_rt[0, 0].cpu().numpy()
                raw_l2_rt = float(np.linalg.norm(rt_np - t_np) / np.linalg.norm(t_np))
                ss_rt, ps_rt, rm_rt, rt_cal = cal_metrics(fbp_rt, truth)

                tag = f"y_up={int(y_up)} ccw={int(ccw)} det+={int(det_pos)}"
                rows.append((y_up, ccw, det_pos, raw_l2, k_l2, ss_f, ss_t, ss_rt, raw_l2_rt))
                cal_recons[tag] = (fbp_cal_t[0, 0].cpu().numpy(), ss_f, ss_t)
                cal_rts[tag] = (rt_cal[0, 0].cpu().numpy(), ss_rt, raw_l2_rt)
                print(f"  {tag:<28}  L2vs_fbp128={raw_l2:.3e}  L2_k={k_l2:.3e}  k={k:.4f}  "
                      f"calSSIMvFBP={ss_f:.4f}  calSSIMvTruth={ss_t:.4f}  "
                      f"calSSIM_rt_vs_truth={ss_rt:.4f}  rt_L2_vs_truth={raw_l2_rt:.3e}")

    # Sort by best agreement to FBP128 (smallest raw L2)
    rows_sorted = sorted(rows, key=lambda r: r[3])
    print("\n=== BEST CONVENTIONS BY rel-L2 vs val_fbp128 ===")
    for i, r in enumerate(rows_sorted[:4]):
        y_up, ccw, det_pos, raw_l2, k_l2, ss_f, ss_t, ss_rt, raw_l2_rt = r
        print(f"  #{i+1}  y_up={int(y_up)} ccw={int(ccw)} det+={int(det_pos)}  "
              f"L2vsFBP={raw_l2:.3e}  calSSIMvFBP={ss_f:.4f}  ssim_rt_vs_truth={ss_rt:.4f}")

    print("\n=== BEST CONVENTIONS BY SELF-ROUND-TRIP SSIM vs TRUTH ===")
    rt_sorted = sorted(rows, key=lambda r: -r[7])
    for i, r in enumerate(rt_sorted[:4]):
        y_up, ccw, det_pos, raw_l2, k_l2, ss_f, ss_t, ss_rt, raw_l2_rt = r
        print(f"  #{i+1}  y_up={int(y_up)} ccw={int(ccw)} det+={int(det_pos)}  "
              f"ssim_rt_vs_truth={ss_rt:.4f}  rt_L2={raw_l2_rt:.3e}  L2vsFBP={raw_l2:.3e}")

    # Figure: grid of cal-OURS for all 8 conventions, plus diff to fbp128
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 8, hspace=0.30, wspace=0.10)
    for i, (tag, (im, ss_f, ss_t)) in enumerate(cal_recons.items()):
        c = i % 4
        rgroup = 0 if i < 4 else 2
        ax = fig.add_subplot(gs[rgroup, c]); ax.imshow(im, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title(f"{tag}\ncal SSIM vsFBP={ss_f:.4f}", fontsize=9); ax.axis("off")
        diff = im - f_k
        lim = float(np.percentile(np.abs(diff), 99))
        ax = fig.add_subplot(gs[rgroup + 1, c]); ax.imshow(diff, cmap="bwr", vmin=-lim, vmax=lim); ax.set_title(f"cal_OURS − sidky_fbp128\n|err|99={lim:.3f}", fontsize=9); ax.axis("off")
    # Row 2 reserved for tags 4..7; row 3 for their diffs (done above via rgroup)
    # Also: bottom-right column show truth and Sidky FBP128 for reference
    ax = fig.add_subplot(gs[0, 4]); ax.imshow(t_np, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title("truth", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 5]); ax.imshow(f_k, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title("sidky FBP128", fontsize=9); ax.axis("off")
    # Best round-trip reconstruction
    best_rt_tag = max(cal_rts.items(), key=lambda x: x[1][1])[0]
    rt_im, rt_ss, rt_l2 = cal_rts[best_rt_tag]
    ax = fig.add_subplot(gs[0, 6]); ax.imshow(rt_im, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title(f"BEST RT (cal)\n{best_rt_tag}\nSSIM={rt_ss:.4f}", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 7]); ax.imshow(rt_im - t_np, cmap="bwr", vmin=-DISPLAY_MAX/4, vmax=DISPLAY_MAX/4); ax.set_title(f"BEST RT − truth\nL2={rt_l2:.3e}", fontsize=9); ax.axis("off")

    plt.suptitle("Sweep of image y-axis × view direction × detector u-sign. "
                 "Best convention has smallest rel-L2 vs Sidky's val_fbp128 (and clean self-round-trip).",
                 fontsize=12, y=1.001)
    out = OUT_DIR / "siddon_conv_sweep.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
