#!/usr/bin/env python -u
"""Fit a radially-symmetric 2D image-space filter K such that

    IFFT(  H(ρ) · FFT( FBP_cal )  )  ≈  truth

for L014/fulldose at the SSIM-peak GT slice (GT#76, patient_z = −254.50 mm).

We parametrise H by its values on `n_bins` radial-frequency bins (real-
valued; rotationally symmetric kernels are zero-phase). The fit is L2
with a second-difference smoothness penalty on H:

    minimise  || IFFT(H_2D · FFT_FBP) − truth ||² + λ · || D²H_radial ||²

Optimised with LBFGS in torch on CUDA. Outputs:
  1. Before-fit vs after-fit calibrated SSIM/PSNR/RMSE/diff_max
  2. Plot of H(ρ) — the learned filter response
  3. 4-panel image: truth | FBP_cal (before) | filtered FBP (after) | diff after
  4. Side-by-side diff (before vs after) at the same colour scale

Usage:  python -u scripts/fit_radial_kernel_L014.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import evaluate_calibrated, ssim as ssim_fn, psnr as psnr_fn


# ---------------------------------------------------------------------------
# 1. Compute the FBP for the SSIM-peak slice the same way job 762230 did.
# ---------------------------------------------------------------------------

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


def build_fbp_and_truth(target_pZ: float = -254.50,
                         filter_name: str = "ramlak"):
    """Returns (truth, fbp_cal) both as numpy float32 (512, 512) in mu units.

    ``filter_name`` is the PYRO-NN FBP ramp-filter window. Default is
    ``ramlak`` (un-windowed) so the radial-kernel-fit downstream has a
    clean baseline — Hann pre-blurs higher frequencies which the fit
    would then have to undo, biasing the L2 solution.
    """
    import os
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    sino_dir = root / "staged_helix2fan"
    raw_dir = root / "raw" / "L014"
    geom_json = json.loads((sino_dir / "L014_sino_fulldose_geometry.json").read_text())
    nu, rotview, nz = int(geom_json['nu']), int(geom_json['rotview']), int(geom_json['nz_rebinned'])
    du = float(geom_json['du'])
    dv = float(geom_json.get('dv_rebinned', 1.0))
    z_start_src = float(geom_json['z_start'])
    angle_start = float(geom_json['angle_start_corrected'])

    truth_files = _list_truth(raw_dir)
    zs = np.array([t[0] for t in truth_files])
    ti = int(np.argmin(np.abs(zs - target_pZ)))
    pZ, fp = truth_files[ti]
    truth_mu, ds = _mu(fp)
    pixel_sp = float(ds.PixelSpacing[0])
    slice_thk = float(ds.SliceThickness)
    print(f"[fit] truth #{ti} pZ={pZ:.2f} mm  PixelSpacing={pixel_sp:.4f}  thk={slice_thk}")

    # Physical-overlap slab integral (same as job 762230)
    slab_lo, slab_hi = -pZ - slice_thk/2.0, -pZ + slice_thk/2.0
    j_lo = max(0, int(math.floor((slab_lo - z_start_src) / dv - 0.5)))
    j_hi = min(nz - 1, int(math.ceil((slab_hi - z_start_src) / dv + 0.5)))
    weights = {}
    for j in range(j_lo, j_hi + 1):
        z_j = z_start_src + j * dv
        bin_lo, bin_hi = z_j - dv/2.0, z_j + dv/2.0
        ov = max(0.0, min(bin_hi, slab_hi) - max(bin_lo, slab_lo))
        if ov > 0:
            weights[j] = ov / slice_thk
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    geom = FanBeamGeometry(
        image_size=512, pixel_spacing=pixel_sp,
        n_angles=rotview, n_det=nu, det_spacing=du,
        sod=595.0, sdd=1085.6,
        angle_start=angle_start, angle_end=angle_start + 2*math.pi,
    )
    proj = PyronnFanBeamProjector(geom).to("cuda")
    print(f"[fit] FBP filter = {filter_name!r}", flush=True)
    fbp_slab = np.zeros_like(truth_mu, dtype=np.float64)
    with h5py.File(sino_dir / "L014_sino_fulldose.h5", "r") as f:
        for j, w in weights.items():
            s = np.ascontiguousarray(np.flip(np.asarray(f["sino"][:, :, j],
                                                         dtype=np.float32), axis=-1))
            t = torch.from_numpy(s).to("cuda").float()[None, None]
            out = proj.fbp(t, filter_name=filter_name).detach()[0, 0].cpu().numpy()
            fbp_slab += w * np.fliplr(np.flipud(out))
    fbp_slab = np.clip(fbp_slab.astype(np.float32), 0.0, None)

    # Intensity-calibrate (same as validator)
    fbp_t = torch.from_numpy(fbp_slab).to("cuda").float()[None, None]
    truth_t = torch.from_numpy(truth_mu).to("cuda").float()[None, None]
    m = evaluate_calibrated(fbp_t, truth_t, baseline=fbp_t,
                             display_min=0.0, display_max=0.05, fov=False)
    fbp_cal = m['pred_cal'][0, 0].cpu().numpy()
    return truth_mu, fbp_cal


# ---------------------------------------------------------------------------
# 2. Fit a radially-symmetric L2-smooth image-space filter.
# ---------------------------------------------------------------------------

def fit_radial_filter(truth_np: np.ndarray, fbp_np: np.ndarray, *,
                       n_bins: int = 96, lam_smooth: float = 1e-4,
                       n_iters: int = 80, device: str = "cuda"):
    """Returns (h_radial, filtered_fbp).

    h_radial: float32 array of length n_bins giving H(ρ_k) on a uniform
              ρ grid from 0 to ρ_max = sqrt(2)/2 cycles/pixel.
    filtered_fbp: float32 (H, W) = IFFT( H_2D · FFT( fbp ) ).real
    """
    H, W = fbp_np.shape
    fbp = torch.from_numpy(fbp_np).to(device).float()
    truth = torch.from_numpy(truth_np).to(device).float()

    # Radial frequency grid
    fy = torch.fft.fftfreq(H, device=device).float()
    fx = torch.fft.fftfreq(W, device=device).float()
    fyy, fxx = torch.meshgrid(fy, fx, indexing='ij')
    rho = torch.sqrt(fyy**2 + fxx**2)               # 0 ≤ rho ≤ sqrt(2)/2
    rho_max = float(rho.max())
    bin_pos = (rho / rho_max) * (n_bins - 1)
    bin_lo = bin_pos.floor().long().clamp(0, n_bins - 1)
    bin_hi = (bin_lo + 1).clamp(0, n_bins - 1)
    bin_frac = (bin_pos - bin_lo.float()).float()

    # H(ρ) — initialised at 1.0 (identity filter)
    h_radial = torch.nn.Parameter(torch.ones(n_bins, device=device, dtype=torch.float32))

    fbp_fft = torch.fft.fft2(fbp)
    # Split into real/imag parts (and stack along last dim) so the
    # multiplication chain stays in real-valued autograd land. Using
    # torch.complex on a casted-to-complex tensor lost the gradient
    # through .to(complex64) in the previous attempt.
    fbp_fft_r = fbp_fft.real
    fbp_fft_i = fbp_fft.imag

    def loss_fn():
        h_2d = h_radial[bin_lo] * (1 - bin_frac) + h_radial[bin_hi] * bin_frac  # (H, W) real
        # Multiply real-valued filter with complex FFT, component-wise.
        filt_fft = torch.complex(h_2d * fbp_fft_r, h_2d * fbp_fft_i)
        filt = torch.fft.ifft2(filt_fft).real
        data_loss = ((filt - truth) ** 2).mean()
        # Second-difference smoothness on h_radial
        smooth_loss = ((h_radial[2:] - 2 * h_radial[1:-1] + h_radial[:-2]) ** 2).mean()
        return data_loss + lam_smooth * smooth_loss, data_loss.item(), smooth_loss.item()

    # Quick gradient sanity check
    h_radial.grad = None
    total0, d0, s0 = loss_fn()
    total0.backward()
    grad_norm0 = float(h_radial.grad.norm())
    print(f"[fit] initial: data_loss={d0:.3e}  smooth_loss={s0:.3e}  "
          f"|grad|={grad_norm0:.3e}", flush=True)

    # Adam — robust to small gradients. LBFGS terminated immediately because
    # |grad| (1.85e-6) was below tolerance_grad (1e-5). The data_loss is
    # naturally tiny because both images are normalised to mu-units ~ 0.02.
    # A multi-thousand iter Adam loop reliably finds the radial filter.
    opt = torch.optim.Adam([h_radial], lr=1e-3)
    n_iters_adam = max(n_iters * 25, 500)  # at least 500
    log_every = max(1, n_iters_adam // 20)

    for i in range(n_iters_adam):
        opt.zero_grad()
        total, d_i, s_i = loss_fn()
        total.backward()
        opt.step()
        if i % log_every == 0 or i == n_iters_adam - 1:
            print(f"[fit] adam {i:5d}/{n_iters_adam}  data_loss={d_i:.3e}  "
                  f"smooth_loss={s_i:.3e}  "
                  f"h_radial range=[{h_radial.min().item():.4f}, "
                  f"{h_radial.max().item():.4f}]", flush=True)

    # Apply (same component-wise multiply as in loss_fn)
    with torch.no_grad():
        h_2d = h_radial[bin_lo] * (1 - bin_frac) + h_radial[bin_hi] * bin_frac
        filt_fft = torch.complex(h_2d * fbp_fft_r, h_2d * fbp_fft_i)
        filt = torch.fft.ifft2(filt_fft).real
    return h_radial.detach().cpu().numpy(), filt.cpu().numpy(), rho_max


# ---------------------------------------------------------------------------
# 3. Metrics helpers
# ---------------------------------------------------------------------------

def calc_metrics(pred_np: np.ndarray, truth_np: np.ndarray, dr: float = 0.05) -> dict:
    pred_clip = np.clip(pred_np, 0.0, None)
    pred_t = torch.from_numpy(pred_clip).to("cuda").float()[None, None]
    truth_t = torch.from_numpy(truth_np).to("cuda").float()[None, None]
    return {
        "ssim": float(ssim_fn(pred_t, truth_t, data_range=dr).cpu()),
        "psnr": float(psnr_fn(pred_t, truth_t, data_range=dr).cpu()),
        "rmse": float(((pred_t - truth_t) ** 2).mean().sqrt().cpu()),
        "diff_max": float(np.abs(pred_clip - truth_np).max()),
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Filter to use for the FBP baseline ("hann" or "ramlak"). The output
    # PNG/NPY filenames carry the filter name so multiple runs can coexist.
    import os
    filter_name = os.environ.get("FBP_FILTER", "ramlak")
    print(f"[fit] using FBP filter = {filter_name!r} (override via FBP_FILTER env)",
          flush=True)
    truth, fbp_cal = build_fbp_and_truth(target_pZ=-254.50,
                                           filter_name=filter_name)
    dr = 0.05
    metrics_before = calc_metrics(fbp_cal, truth, dr=dr)
    print(f"\n[fit] BEFORE  SSIM={metrics_before['ssim']:.4f}  "
          f"PSNR={metrics_before['psnr']:.2f} dB  RMSE={metrics_before['rmse']:.5f}  "
          f"diff_max={metrics_before['diff_max']:.4f}", flush=True)

    h_radial, filtered, rho_max = fit_radial_filter(
        truth, fbp_cal, n_bins=96, lam_smooth=1e-4, n_iters=60, device="cuda"
    )
    metrics_after = calc_metrics(filtered, truth, dr=dr)
    print(f"[fit] AFTER   SSIM={metrics_after['ssim']:.4f}  "
          f"PSNR={metrics_after['psnr']:.2f} dB  RMSE={metrics_after['rmse']:.5f}  "
          f"diff_max={metrics_after['diff_max']:.4f}", flush=True)

    diff_before = np.clip(fbp_cal, 0, None) - truth
    diff_after = np.clip(filtered, 0, None) - truth

    # --- Plot --------------------------------------------------------------
    out_dir = Path("/cluster/maier/Agent4CT/results/breast_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = filter_name  # filename suffix to keep multiple baselines side-by-side

    # Main 4-panel comparison
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(truth, cmap="gray", vmin=0, vmax=dr)
    ax[0].set_title("truth (GT#76, pZ=−254.50 mm)\nB30f / 5-mm slice", fontsize=10)
    ax[1].imshow(np.clip(fbp_cal, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[1].set_title(f"FBP_cal ({filter_name}, before filter)\n"
                    f"SSIM={metrics_before['ssim']:.4f}  "
                    f"PSNR={metrics_before['psnr']:.2f} dB  "
                    f"RMSE={metrics_before['rmse']:.5f}",
                    fontsize=10)
    ax[2].imshow(np.clip(filtered, 0, None), cmap="gray", vmin=0, vmax=dr)
    ax[2].set_title(f"filtered FBP (L2-fit radial K)\n"
                    f"SSIM={metrics_after['ssim']:.4f}  "
                    f"PSNR={metrics_after['psnr']:.2f} dB  "
                    f"RMSE={metrics_after['rmse']:.5f}",
                    fontsize=10)
    ax[3].imshow(diff_after, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax[3].set_title(f"diff (filtered − truth)\nmax|·|={np.abs(diff_after).max():.4f}",
                    fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("L014 fulldose: radial L2-smooth filter fit (B30f-vs-Hann + slab residual)",
                 fontsize=11)
    fig.tight_layout()
    out_main = out_dir / f"L014_radial_kernel_fit_{suf}.png"
    fig.savefig(out_main, dpi=120)
    print(f"[fit] wrote {out_main}", flush=True)

    # Before/after diff panel
    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.5))
    ax2[0].imshow(diff_before, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[0].set_title(f"diff BEFORE (FBP_cal − truth)\n"
                     f"max|·|={np.abs(diff_before).max():.4f}",
                     fontsize=10)
    ax2[1].imshow(diff_after, cmap="seismic", vmin=-0.02, vmax=0.02)
    ax2[1].set_title(f"diff AFTER (filtered − truth)\n"
                     f"max|·|={np.abs(diff_after).max():.4f}",
                     fontsize=10)
    for a in ax2: a.set_xticks([]); a.set_yticks([])
    fig2.tight_layout()
    out_diff = out_dir / f"L014_radial_kernel_diff_beforeafter_{suf}.png"
    fig2.savefig(out_diff, dpi=120)
    print(f"[fit] wrote {out_diff}", flush=True)

    # Radial filter response curve
    rho_axis = np.linspace(0, rho_max, len(h_radial))
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4.5))
    ax3.plot(rho_axis, h_radial, lw=1.5)
    ax3.axhline(1.0, color="gray", ls=":", lw=0.8, label="identity")
    ax3.set_xlabel("radial frequency ρ (cycles / pixel)")
    ax3.set_ylabel("H(ρ)  —  multiplicative filter response")
    ax3.set_title("Estimated radial L2-smooth filter response (image-domain rotational kernel)\n"
                  "ρ > 1 ⇒ truth has more energy than our Hann FBP at that radial frequency",
                  fontsize=10)
    ax3.grid(alpha=0.3); ax3.legend()
    fig3.tight_layout()
    out_curve = out_dir / f"L014_radial_kernel_response_{suf}.png"
    fig3.savefig(out_curve, dpi=120)
    print(f"[fit] wrote {out_curve}", flush=True)

    # Save the fitted filter
    np.save(out_dir / f"L014_radial_kernel_h_radial_{suf}.npy", h_radial)
    np.save(out_dir / f"L014_radial_kernel_rho_axis_{suf}.npy", rho_axis)
    print()
    print(f"=== SUMMARY ===")
    print(f"BEFORE  SSIM={metrics_before['ssim']:.4f}  PSNR={metrics_before['psnr']:.2f} dB  "
          f"RMSE={metrics_before['rmse']:.5f}  diff_max={metrics_before['diff_max']:.4f}")
    print(f"AFTER   SSIM={metrics_after['ssim']:.4f}  PSNR={metrics_after['psnr']:.2f} dB  "
          f"RMSE={metrics_after['rmse']:.5f}  diff_max={metrics_after['diff_max']:.4f}")
    print(f"Δ       ΔSSIM={metrics_after['ssim']-metrics_before['ssim']:+.4f}  "
          f"ΔPSNR={metrics_after['psnr']-metrics_before['psnr']:+.2f} dB  "
          f"ΔRMSE={(metrics_after['rmse']-metrics_before['rmse'])/metrics_before['rmse']*100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
