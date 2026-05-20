"""SiddonFanBeamProjector test against Sidky's released breast_ct reference.

Predictions from the paper (§II.B, lines 96-110, 150):
  - val_sinograms.h5 was generated as g = X · f where X is Siddon line-int.
  - val_fbp128.h5    was generated as fbp = X^T · (ramp · g).

So with our SiddonFanBeamProjector at the verified geometry:
  - forward_project(val_truth[0]) should reproduce val_sinograms[0].
  - fbp(val_sinograms[0])         should reproduce val_fbp128[0].

This script tests ONE case (case #0) and reports:

  Test 1 (forward agreement):
    rel-L2 ||X·truth - sidky_sino|| / ||sidky_sino||
    max-abs error
  Test 2 (FBP agreement vs val_fbp128.h5):
    SSIM / PSNR / max-abs against val_fbp128[0] (no intensity calibration —
    if matched, scales agree to a constant; report calibrated metrics too).
  Test 3 (FBP agreement vs val_truth[0]):
    SSIM / PSNR (with intensity calibration; display_max=0.5).

Outputs:
  /cluster/maier/Agent4CT/results/breast_debug/siddon_match.png

If forward L2 ≪ 1% and FBP SSIM vs val_fbp128 > 0.99, the matched-pair
hypothesis is confirmed and we can use SiddonFanBeamProjector for any
solver / training step that needs a Sidky-matched forward / back-projector.
"""
from __future__ import annotations
import sys, time
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
SINO_SHIFT = 32                  # +90° CW sino advance, verified empirically
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


def metrics_cal(pred, truth, dmax=DISPLAY_MAX):
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

    truth = torch.from_numpy(truth_np).float().to(device).unsqueeze(1)  # (1,1,H,W)
    sino_sidky = torch.from_numpy(sino_np).float().to(device).unsqueeze(1)  # (1,1,A,D)
    fbp_sidky = torch.from_numpy(fbp_np).float().to(device).unsqueeze(1)  # (1,1,H,W)
    print(f"truth        shape={tuple(truth.shape)}  range=[{truth.min():.4f}, {truth.max():.4f}]")
    print(f"sidky sino   shape={tuple(sino_sidky.shape)}  range=[{sino_sidky.min():.4f}, {sino_sidky.max():.4f}]")
    print(f"sidky FBP128 shape={tuple(fbp_sidky.shape)}  range=[{fbp_sidky.min():.4f}, {fbp_sidky.max():.4f}]")

    geom = FanBeamGeometry(**GEOM)
    # breast_ct: geometry in mm, truth attenuation in 1/cm → scale=0.1.
    proj = SiddonFanBeamProjector(geom, ray_batch=8192, length_unit_scale=0.1).to(device)
    print(f"\ndevice={device}, ray_batch={proj.ray_batch}, "
          f"length_unit_scale={proj.length_unit_scale}, "
          f"A·N_det={geom.n_angles * geom.n_det}")

    # ── Test 1: forward agreement ───────────────────────────────────────────
    print("\n=== TEST 1: forward agreement (X · truth vs sidky_sino) ===")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        sino_ours = proj.forward_project(truth)
    if device == "cuda":
        torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000
    print(f"  forward took {fwd_ms:.1f} ms")
    print(f"  ours: range=[{sino_ours.min():.4f}, {sino_ours.max():.4f}], mean={sino_ours.mean():.4f}")
    print(f"  sidky: range=[{sino_sidky.min():.4f}, {sino_sidky.max():.4f}], mean={sino_sidky.mean():.4f}")
    # Try direct match
    # Apply the same SINO_SHIFT we use in geometry to align angle indexing
    sino_sidky_aligned = torch.roll(sino_sidky, shifts=SINO_SHIFT, dims=-2)
    s_o = sino_ours[0, 0].cpu().numpy()
    s_k_aligned = sino_sidky_aligned[0, 0].cpu().numpy()
    s_k_unaligned = sino_sidky[0, 0].cpu().numpy()

    def report(s_o, s_k, tag):
        l2 = float(np.linalg.norm(s_o - s_k) / np.linalg.norm(s_k))
        amax = float(np.abs(s_o - s_k).max())
        # Best linear scale-match k* = (s_o · s_k) / (s_o · s_o)
        k = float((s_o * s_k).sum() / max((s_o * s_o).sum(), 1e-12))
        l2k = float(np.linalg.norm(k * s_o - s_k) / np.linalg.norm(s_k))
        print(f"  {tag:<30}  rel-L2={l2:.4e}  max-abs={amax:.4e}  best k={k:.4g}  rel-L2(k)={l2k:.4e}")
        return k, l2k

    print("  alignment sweep:")
    _, l2_a = report(s_o, s_k_aligned,    f"shift={+SINO_SHIFT}")
    _, l2_u = report(s_o, s_k_unaligned,  "shift=0")
    # Also try -SINO_SHIFT
    s_k_neg = torch.roll(sino_sidky, shifts=-SINO_SHIFT, dims=-2)[0, 0].cpu().numpy()
    _, l2_n = report(s_o, s_k_neg, f"shift={-SINO_SHIFT}")
    best_shift = [0, +SINO_SHIFT, -SINO_SHIFT][int(np.argmin([l2_u, l2_a, l2_n]))]
    print(f"  best angular shift = {best_shift}")
    sino_sidky_best = torch.roll(sino_sidky, shifts=best_shift, dims=-2)

    # ── Test 2: FBP(sidky_sino) vs val_fbp128 ───────────────────────────────
    print("\n=== TEST 2: FBP(sidky_sino) vs val_fbp128 ===")
    t0 = time.time()
    with torch.no_grad():
        fbp_ours = proj.fbp(sino_sidky_best)
    if device == "cuda":
        torch.cuda.synchronize()
    fbp_ms = (time.time() - t0) * 1000
    print(f"  FBP took {fbp_ms:.1f} ms")
    f_o = fbp_ours[0, 0].cpu().numpy()
    f_k = fbp_sidky[0, 0].cpu().numpy()
    k = float((f_o * f_k).sum() / max((f_o * f_o).sum(), 1e-12))
    f_o_k = k * f_o
    l2_raw = float(np.linalg.norm(f_o - f_k) / np.linalg.norm(f_k))
    l2_scaled = float(np.linalg.norm(f_o_k - f_k) / np.linalg.norm(f_k))
    print(f"  ours range  =[{f_o.min():.4g}, {f_o.max():.4g}]   mean={f_o.mean():.4g}")
    print(f"  sidky range =[{f_k.min():.4g}, {f_k.max():.4g}]   mean={f_k.mean():.4g}")
    print(f"  rel-L2 raw      ={l2_raw:.4e}")
    print(f"  rel-L2 best-k   ={l2_scaled:.4e}   k={k:.4g}")
    # Also compute SSIM / PSNR via intensity-calibration vs val_fbp128
    ss_f, ps_f, rm_f, _ = metrics_cal(fbp_ours, fbp_sidky)
    print(f"  cal SSIM vs FBP128={ss_f:.4f}  PSNR={ps_f:.2f} dB  RMSE={rm_f:.4f}")

    # ── Test 3: FBP(sidky_sino) vs val_truth ────────────────────────────────
    print("\n=== TEST 3: FBP(sidky_sino) vs val_truth (after intensity-calibration) ===")
    ss_t, ps_t, rm_t, fbp_ours_cal = metrics_cal(fbp_ours, truth)
    print(f"  cal SSIM vs truth = {ss_t:.4f}  PSNR={ps_t:.2f} dB  RMSE={rm_t:.4f}")

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.30, wspace=0.20)

    # Row 0: sino panel (truth fwd, sidky-aligned, sino diff, sino diff zoomed)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(s_o, cmap="gray", aspect="auto"); ax.set_title(f"X·truth (Siddon)\nrange=[{s_o.min():.2f}, {s_o.max():.2f}]", fontsize=10); ax.axis("off")
    s_k_b = sino_sidky_best[0, 0].cpu().numpy()
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(s_k_b, cmap="gray", aspect="auto"); ax.set_title(f"sidky sino (shift={best_shift})\nrange=[{s_k_b.min():.2f}, {s_k_b.max():.2f}]", fontsize=10); ax.axis("off")
    diff_s = s_o - s_k_b
    lim_s = float(np.percentile(np.abs(diff_s), 99))
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(diff_s, cmap="bwr", aspect="auto", vmin=-lim_s, vmax=lim_s); ax.set_title(f"sino diff (X·truth − sidky)\nrel-L2={float(np.linalg.norm(diff_s)/np.linalg.norm(s_k_b)):.3f}", fontsize=10); ax.axis("off")
    # Per-angle mean residual (helps spot angular misalignment)
    ax = fig.add_subplot(gs[0, 3])
    per_view_l2 = np.linalg.norm(diff_s, axis=-1) / max(np.linalg.norm(s_k_b, axis=-1).mean(), 1e-9)
    ax.plot(per_view_l2, "o-", markersize=3); ax.set_xlabel("view index"); ax.set_ylabel("per-view rel-L2"); ax.grid(alpha=0.3); ax.set_title("residual per view")

    # Row 1: truth | sidky_fbp128 | ours_fbp | ours - sidky_fbp128
    ax = fig.add_subplot(gs[1, 0]); ax.imshow(truth_np[0], cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title("truth (val_truth[0])", fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[1, 1]); ax.imshow(f_k, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title(f"sidky FBP128\nrange=[{f_k.min():.3f}, {f_k.max():.3f}]", fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[1, 2]); ax.imshow(f_o, cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title(f"OUR FBP (Siddon X^T)\nrange=[{f_o.min():.3f}, {f_o.max():.3f}]\ncal SSIM vs FBP128={ss_f:.4f}", fontsize=10); ax.axis("off")
    diff_fbp = f_o - f_k
    lim_f = float(np.percentile(np.abs(diff_fbp), 99)) if np.any(diff_fbp) else 1e-6
    ax = fig.add_subplot(gs[1, 3]); ax.imshow(diff_fbp, cmap="bwr", vmin=-lim_f, vmax=lim_f); ax.set_title(f"OURS − sidky_fbp128\nrel-L2={l2_raw:.3e}", fontsize=10); ax.axis("off")

    # Row 2: cal-OURS | truth | OURS_cal - truth | scale-matched OURS - sidky
    ax = fig.add_subplot(gs[2, 0]); ax.imshow(fbp_ours_cal[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title(f"OUR FBP (intensity-cal)\ncal SSIM vs truth={ss_t:.4f}", fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[2, 1]); ax.imshow(truth_np[0], cmap="gray", vmin=0, vmax=DISPLAY_MAX); ax.set_title("truth", fontsize=10); ax.axis("off")
    diff_t = fbp_ours_cal[0, 0].cpu().numpy() - truth_np[0]
    lim_t = DISPLAY_MAX / 4
    ax = fig.add_subplot(gs[2, 2]); ax.imshow(diff_t, cmap="bwr", vmin=-lim_t, vmax=lim_t); ax.set_title(f"cal OURS − truth\n|err|max={float(np.abs(diff_t).max()):.3f}", fontsize=10); ax.axis("off")
    diff_scaled = f_o_k - f_k
    ax = fig.add_subplot(gs[2, 3]); ax.imshow(diff_scaled, cmap="bwr", vmin=-lim_f, vmax=lim_f); ax.set_title(f"{k:.3g}·OURS − sidky_fbp128\nrel-L2={l2_scaled:.3e}", fontsize=10); ax.axis("off")

    plt.suptitle(f"SiddonFanBeamProjector vs Sidky reference (case #0). "
                 f"Forward agreement should be tight; FBP via X^T should reproduce val_fbp128.",
                 fontsize=11, y=1.001)
    out = OUT_DIR / "siddon_match.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
