"""Disc-phantom + breast-data diagnostic for FBP cupping in pyronn_projector.

For Sidky breast_ct geometry (sod=500, sdd=1000, ps=ds=0.3516, 128 views), test
four variants of the FBP filter step:

  V1  baseline     : current pyronn FBP (no padding, no fan-beam cosine pre-weight)
  V2  +pad         : zero-pad sinogram to 2N along detector before ramp FFT
  V3  +cos         : pre-multiply sinogram by sdd / sqrt(sdd² + s²)
  V4  +pad +cos    : both fixes together

Test A (uniform disc): forward-project a constant-mu disc through the same
geometry, then FBP each variant. A correct FBP gives a flat plateau across
the disc. The deviation from flat (= cupping profile) isolates which fix matters.

Test B (real breast): apply each variant to 4 val_sinograms cases. Reports SSIM/
PSNR/RMSE vs truth (after intensity calibration, display_max=0.5) + diff images.

Outputs:
  /cluster/maier/Agent4CT/results/breast_debug/padcos_disc.png
  /cluster/maier/Agent4CT/results/breast_debug/padcos_breast.png
"""
from __future__ import annotations
import sys
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
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate

GEOM = dict(image_size=512, pixel_spacing=180/512, n_angles=128,
            n_det=1024, det_spacing=360/1024, sod=500.0, sdd=1000.0)

DISPLAY_MAX = 0.5
SINO_SHIFT = 32        # Sidky breast_ct -90° CW alignment
OUT_DIR = Path("/cluster/maier/Agent4CT/results/breast_debug")


# ─── filter variants ──────────────────────────────────────────────────────────

def _fan_cosine(geom: FanBeamGeometry, device, dtype):
    """w[d] = sdd / sqrt(sdd² + s²), s = (d - (N-1)/2) · det_spacing (flat det)."""
    n = geom.n_det
    s = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0) * geom.det_spacing
    return geom.sdd / torch.sqrt(geom.sdd ** 2 + s ** 2)


def _ramlak_spec(N: int, det_spacing: float, device, dtype, pad: bool):
    """Return the frequency-domain ram-lak filter for length M (= N or 2N)."""
    M = 2 * N if pad else N
    h = np.zeros(M, dtype=np.float64)
    h[0] = 0.25 / (det_spacing * det_spacing)
    odd = -1.0 / (np.pi * np.pi * det_spacing * det_spacing)
    for i in range(1, M):
        # exact pyronn convention: split kernel about M/2, mirror odd-n on the
        # negative side via tmp = M - i.
        if i < M / 2 and (i % 2) == 1:
            h[i] = odd / (i * i)
        elif i >= M / 2:
            tmp = M - i
            if (tmp % 2) == 1:
                h[i] = odd / (tmp * tmp)
    f = np.real(np.fft.fft(h)).astype(np.float32)
    return torch.as_tensor(f, dtype=dtype, device=device)


def filter_sino_variant(sino: torch.Tensor, geom: FanBeamGeometry,
                         pad: bool, cos_pre: bool) -> torch.Tensor:
    """Diagnostic filter_sino with optional zero-pad and optional fan cosine pre-weight."""
    N = sino.shape[-1]
    device, dtype = sino.device, sino.dtype
    x = sino
    if cos_pre:
        w = _fan_cosine(geom, device, dtype)        # shape (N,)
        x = x * w                                   # broadcast along last axis
    if pad:
        # zero-pad along detector to 2N for linear convolution
        pad_tail = torch.zeros(x.shape[:-1] + (N,), device=device, dtype=dtype)
        xp = torch.cat([x, pad_tail], dim=-1)        # (..., 2N)
    else:
        xp = x
    f = _ramlak_spec(N, geom.det_spacing, device, dtype, pad=pad)
    spec = torch.fft.fft(xp, dim=-1, norm="ortho")
    spec = spec * f
    y = torch.fft.ifft(spec, dim=-1, norm="ortho").real
    if pad:
        y = y[..., :N]                                # discard padded tail
    return y


def fbp_variant(proj: PyronnFanBeamProjector, sino: torch.Tensor, *,
                pad: bool, cos_pre: bool) -> torch.Tensor:
    """FBP path with the same redundancy weighting as pyronn_projector.fbp(),
    but the *filter step* swapped for our diagnostic version."""
    A = sino.shape[-2]
    if A == proj._redundancy_weights.shape[0]:
        rw = proj._redundancy_weights
    else:
        raise NotImplementedError("diagnostic: use full-set sinograms only")
    sino_w = sino * rw
    filt = filter_sino_variant(sino_w, proj.geom, pad=pad, cos_pre=cos_pre)
    return proj.back_project(filt)


# ─── disc phantom ─────────────────────────────────────────────────────────────

def make_disc(N: int, radius_pix: int, mu: float, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(N, device=device, dtype=dtype) - (N - 1) / 2.0,
        torch.arange(N, device=device, dtype=dtype) - (N - 1) / 2.0,
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy)
    img = torch.where(r <= radius_pix, torch.tensor(mu, device=device, dtype=dtype),
                      torch.tensor(0.0, device=device, dtype=dtype))
    return img.view(1, 1, N, N)


def metrics_v(pred, truth, dmax=DISPLAY_MAX):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=dmax)
    ss = float(ssim(pc, truth, data_range=dmax).cpu())
    ps = float(psnr(pc, truth, data_range=dmax).cpu())
    rm = float(((pc - truth) ** 2).mean().sqrt().cpu())
    return ss, ps, rm, pc


# ─── tests ────────────────────────────────────────────────────────────────────

def run_disc_test(proj, device):
    """Forward-project disc → FBP four ways → radial profile + recon panel."""
    N = proj.geom.image_size
    radius_pix = N // 3        # ~170 px ≈ 60 mm — well inside the FOV
    disc = make_disc(N, radius_pix, mu=0.20, device=device, dtype=torch.float32)
    with torch.no_grad():
        sino = proj.forward_project(disc)            # (1, 1, A, D)

    variants = [
        ("baseline",   dict(pad=False, cos_pre=False)),
        ("+pad",       dict(pad=True,  cos_pre=False)),
        ("+cos",       dict(pad=False, cos_pre=True)),
        ("+pad+cos",   dict(pad=True,  cos_pre=True)),
    ]
    recs = []
    for name, kw in variants:
        with torch.no_grad():
            rec = fbp_variant(proj, sino, **kw)        # (1, 1, N, N)
        rec_cal = intensity_calibrate(rec.clamp_min(0.0), disc, display_max=DISPLAY_MAX)
        recs.append((name, rec_cal[0, 0].cpu().numpy()))

    # Layout: 2 rows × 4 cols. Row 0: recon images. Row 1: radial profile.
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    mid = N // 2
    disc_np = disc[0, 0].cpu().numpy()
    truth_profile = disc_np[mid, :]
    for ci, (name, im) in enumerate(recs):
        axes[0, ci].imshow(im, cmap="gray", vmin=0.0, vmax=DISPLAY_MAX)
        axes[0, ci].set_title(f"disc recon — {name}\nrange=[{im.min():.3f}, {im.max():.3f}]",
                              fontsize=10)
        axes[0, ci].axis("off")
        # horizontal radial profile through centre
        prof = im[mid, :]
        axes[1, ci].plot(truth_profile, "k--", linewidth=0.8, label="truth")
        axes[1, ci].plot(prof, "r", linewidth=1.0, label=name)
        axes[1, ci].set_ylim(-0.02, DISPLAY_MAX * 1.1)
        axes[1, ci].set_title(f"horizontal profile  {name}", fontsize=10)
        axes[1, ci].legend(loc="lower center", fontsize=8)
        axes[1, ci].grid(alpha=0.3)
    plt.suptitle(f"Disc phantom (r={radius_pix}px, μ=0.20) — FBP variants. "
                 f"Flat plateau = correct. Cup = bad.", fontsize=12, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "padcos_disc.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return recs, truth_profile


def run_breast_test(proj, device):
    """Apply each variant to 4 real breast cases, vs ground truth."""
    data = Path("/cluster/maier/Agent4CT/data/dl_sparse_view/staged")
    with h5py.File(data / "val_truth.h5", "r") as f: truth = f["image"][:4]
    with h5py.File(data / "val_sinograms.h5", "r") as f: sino = f["sino"][:4]

    s_all = torch.from_numpy(sino).float().to(device).unsqueeze(1)
    t_all = torch.from_numpy(truth).float().to(device).unsqueeze(1)
    # Apply -90° CW alignment via sino shift (Sidky convention)
    s_all = torch.roll(s_all, shifts=SINO_SHIFT, dims=-2)

    variants = [
        ("baseline",   dict(pad=False, cos_pre=False)),
        ("+pad",       dict(pad=True,  cos_pre=False)),
        ("+cos",       dict(pad=False, cos_pre=True)),
        ("+pad+cos",   dict(pad=True,  cos_pre=True)),
    ]

    fig, axes = plt.subplots(4, 6, figsize=(24, 14))
    print(f"\n{'case':<4} {'variant':<12} {'SSIM':>7} {'PSNR':>7} {'RMSE':>8}")
    summary = {name: {"ssim": [], "psnr": [], "rmse": []} for name, _ in variants}
    for r in range(4):
        gt = t_all[r:r+1]
        gt_np = truth[r]
        axes[r, 0].imshow(gt_np, cmap="gray", vmin=0.0, vmax=DISPLAY_MAX)
        axes[r, 0].set_title(f"truth #{r}", fontsize=10); axes[r, 0].axis("off")
        for ci, (name, kw) in enumerate(variants, start=1):
            with torch.no_grad():
                rec = fbp_variant(proj, s_all[r:r+1], **kw)
            ss, ps, rm, rec_cal = metrics_v(rec, gt)
            summary[name]["ssim"].append(ss); summary[name]["psnr"].append(ps); summary[name]["rmse"].append(rm)
            rec_np = rec_cal[0, 0].cpu().numpy()
            axes[r, ci].imshow(rec_np, cmap="gray", vmin=0.0, vmax=DISPLAY_MAX)
            axes[r, ci].set_title(f"OUR {name}\nSSIM={ss:.3f} PSNR={ps:.1f}\nRMSE={rm:.4f}",
                                   fontsize=9)
            axes[r, ci].axis("off")
            print(f"{r:<4} {name:<12} {ss:7.4f} {ps:7.2f} {rm:8.4f}")
        # Diff column: best-SSIM variant - truth
        best_name = max(summary, key=lambda n: summary[n]["ssim"][r])
        idx = [n for n, _ in variants].index(best_name)
        # Re-run best variant to get the calibrated recon
        kw = variants[idx][1]
        with torch.no_grad():
            rec = fbp_variant(proj, s_all[r:r+1], **kw)
        _, _, _, rec_cal = metrics_v(rec, gt)
        diff = (rec_cal - gt)[0, 0].cpu().numpy()
        lim = DISPLAY_MAX / 4
        axes[r, 5].imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
        axes[r, 5].set_title(f"diff: best ({best_name}) - truth\n|err|max={float(np.abs(diff).max()):.3f}",
                              fontsize=9); axes[r, 5].axis("off")

    print(f"\n--- mean across 4 cases ---")
    for name, _ in variants:
        ss = np.mean(summary[name]["ssim"]); ps = np.mean(summary[name]["psnr"]); rm = np.mean(summary[name]["rmse"])
        print(f"{name:<12} SSIM={ss:.4f}  PSNR={ps:5.2f}  RMSE={rm:.4f}")

    plt.suptitle(f"Real breast FBP — 4 variants (intensity-calibrated, vmin=0 vmax={DISPLAY_MAX}). "
                 f"diff at ±{DISPLAY_MAX/4:.3f}.", fontsize=12, y=1.001)
    plt.tight_layout()
    out = OUT_DIR / "padcos_breast.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geom = FanBeamGeometry(**GEOM)
    proj = PyronnFanBeamProjector(geom).to(device)

    print("=== TEST A: uniform-disc phantom (cupping isolation) ===")
    run_disc_test(proj, device)
    print("\n=== TEST B: real breast (4 cases) ===")
    run_breast_test(proj, device)


if __name__ == "__main__":
    main()
