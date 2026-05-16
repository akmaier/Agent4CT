"""Reference: Wu et al. 2015 — Novel FBP for Sparse-View CT Reconstruction.

Citation: Wu M., Maier A., Yang Q., Fahrig R. "A Novel Filtered Backprojection-
Based Algorithm for Sparse View CT Image Reconstruction." Fully3D 2015.

The algorithm (no learning):
  1. Build a view-aliasing-free reconstruction via a radial-position-dependent
     ramp filter. The ramp is split into K frequency bands; each band is
     back-projected separately and combined per-pixel with a sigmoid weight
     that suppresses bands which exceed the local Nyquist sampling Δ̃_R = s·Δβ.
  2. Forward-project the aliasing-free image and subtract from the measured
     sinogram to get a residual that holds the high-frequency structures.
  3. Feature-preserving interpolation: for each pair of adjacent views,
     synthesise a midpoint view by symmetric motion-compensated averaging,
     where the per-detector shift is the L1-minimising local patch match.
  4. FBP the densified residual; soft-threshold; add back to the aliasing-free
     image. Iterate steps 2–4 a few times.

This is the classical non-learned baseline that any learned sparse-view
method should be measured against. Implementation in PyTorch on top of
PYRO-NN, using 4 frequency bands (paper uses 8) and 2 outer iterations.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim


CONFIG = {
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,
    "train_n":       400,
    "val_n":         100,
    "noise_i0":      1e5,
    "noise_sigma_e": 10.0,
    "seed":          42,
    "display_min":   0.0,
    "display_max":   0.05,
    # Wu 2015 hyperparameters
    "wu_n_bands":      4,        # paper uses 8 triangular bands; 4 keeps cost down
    "wu_n_outer":      2,        # restoration iterations (paper recommends 2-3)
    "wu_motion_range": 5,        # ±pixels for symmetric motion search
    "wu_motion_window": 2,       # ±pixels for L1 windowed patch
    "wu_soft_thresh":  0.0015,   # soft threshold on the residual reco (mu mm^-1)
}


# ---------------------------------------------------------------------------
def build_dataset(geom, n, seed, i0, sigma_e, device):
    proj = PyronnFanBeamProjector(geom).to(device)
    phantoms = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)
    with torch.no_grad():
        clean = proj.forward_project(phantoms)
        noisy = simulate_low_dose(clean, i0=i0, sigma_e=sigma_e, seed=seed + 10_000)
    return phantoms, clean, noisy


# ---------------------------------------------------------------------------
def _ramp_freq(n_det: int, det_spacing: float, device):
    """1D ramp |f| at the FFT bin frequencies for a detector row."""
    freqs = torch.fft.fftfreq(n_det, d=det_spacing, device=device)  # cycles / mm
    return freqs.abs()                                              # (n_det,)


def _band_filters(n_det: int, det_spacing: float, n_bands: int, device):
    """Split |f| into n_bands triangular sub-filters whose sum is |f|.

    Returns:
        band_filters: (n_bands, n_det) — per-band frequency response
        band_centers: (n_bands,)       — centre frequency of each band (mm^-1)
    """
    freqs = torch.fft.fftfreq(n_det, d=det_spacing, device=device)  # signed
    abs_f = freqs.abs()
    f_max = abs_f.max()
    edges = torch.linspace(0.0, float(f_max), n_bands + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])                        # (n_bands,)
    band_filters = torch.zeros(n_bands, n_det, device=device, dtype=torch.float32)
    for i in range(n_bands):
        lo, hi = float(edges[i]), float(edges[i + 1])
        width = max(hi - lo, 1e-12)
        # Triangular weight inside [lo, hi]
        rel = (abs_f - lo) / width
        tri = torch.clamp(1.0 - 2.0 * torch.abs(rel - 0.5), min=0.0)
        band_filters[i] = abs_f * tri                                # ramp · triangle
    return band_filters, centers


def _radial_distance_map(image_size: int, pixel_spacing: float, device):
    """Per-pixel s = √(x²+y²) · pixel_spacing in mm. Origin at image centre."""
    yy, xx = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=torch.float32),
        torch.arange(image_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (image_size - 1) / 2
    cx = (image_size - 1) / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * pixel_spacing
    return r                                                         # (H, W)


def aliasing_free_fbp(proj: PyronnFanBeamProjector, sino: torch.Tensor,
                     n_bands: int) -> torch.Tensor:
    """View-aliasing-free reconstruction via radius-dependent ramp filter."""
    device = sino.device
    g = proj.geom
    image_size = g.image_size
    pixel_spacing = g.pixel_spacing
    n_det = g.n_det
    det_spacing = g.det_spacing
    A_sino = sino.shape[-2]

    # Build band filters and per-band sigmoid weight maps.
    band_filters, band_centers = _band_filters(n_det, det_spacing, n_bands, device)
    r_map = _radial_distance_map(image_size, pixel_spacing, device)  # (H, W) in mm
    delta_beta = 2.0 * math.pi / float(A_sino)
    delta_R = r_map * delta_beta                                     # (H, W) in mm
    # c_i(x) = 1 / (1 + exp(10 (f_i · ΔR − 1))). Shape (n_bands, H, W).
    fR = band_centers.view(-1, 1, 1) * delta_R.unsqueeze(0)          # (B, H, W)
    weights = torch.sigmoid(-10.0 * (fR - 1.0))                      # (B, H, W)
    # Normalise across bands so the sum equals 1 wherever any band is "supported".
    wsum = weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    weights = weights / wsum                                         # (B, H, W)

    # Apply each band filter to the sino, back-project, weight-sum.
    # sino: (N, 1, A, D) or (N, A, D)
    sino4 = sino if sino.dim() == 4 else sino.unsqueeze(1)
    spec = torch.fft.fft(sino4, dim=-1, norm="ortho")                # (N, 1, A, D)
    out = torch.zeros(sino4.shape[0], 1, image_size, image_size,
                      device=device, dtype=torch.float32)
    for i in range(n_bands):
        h = band_filters[i]                                          # (D,)
        # Reshape for broadcasting along detector axis
        filtered_spec = spec * h.view(1, 1, 1, -1)
        filtered_sino = torch.fft.ifft(filtered_spec, dim=-1, norm="ortho").real
        # PyronnFanBeamProjector.fbp does its own filtering — we want only
        # the back-projection of an already-filtered sinogram, plus the
        # redundancy weight. Replicate the fbp() recipe minus the ramp.
        rw = proj._redundancy_weights.to(device)
        if rw.shape[0] == filtered_sino.shape[-2]:
            weighted = filtered_sino * rw.view(1, 1, *rw.shape)
        else:
            weighted = filtered_sino
        bp = proj.back_project(weighted)
        out = out + bp * weights[i].view(1, 1, image_size, image_size)
    return out


# ---------------------------------------------------------------------------
def feature_preserving_interp(residual_sino: torch.Tensor,
                               max_shift: int, window: int) -> torch.Tensor:
    """Double the angular sampling rate of `residual_sino` via symmetric
    motion-compensated interpolation (Eq. 6–7 of Wu 2015).

    residual_sino: (N, 1, A, D) or (N, A, D), float32.
    Returns: (N, 1, 2A, D)  — interleaved [orig view k, interp midpoint, ...].
    """
    had_channel = residual_sino.dim() == 4
    sino = residual_sino if had_channel else residual_sino.unsqueeze(1)
    N, _, A, D = sino.shape
    device = sino.device

    # Generate adjacent-pair tensors y1 = sino[..., :-1, :], y2 = sino[..., 1:, :]
    y1 = sino[..., :-1, :]                                            # (N,1,A-1,D)
    y2 = sino[..., 1:, :]
    # Wrap-around: also handle the (A-1, 0) pair so we end up with A new views.
    y1_wrap = sino[..., -1:, :]                                       # (N,1,1,D)
    y2_wrap = sino[..., :1, :]
    y1 = torch.cat([y1, y1_wrap], dim=-2)                             # (N,1,A,D)
    y2 = torch.cat([y2, y2_wrap], dim=-2)                             # (N,1,A,D)

    # For each candidate shift t, compute windowed-L1 cost and the candidate
    # midpoint view. Pick t per (pair_idx, detector_idx) that minimises cost.
    best_cost = None
    best_mid = None
    pad = window
    kernel = torch.ones(1, 1, 2 * window + 1, device=device,
                        dtype=torch.float32) / (2 * window + 1)
    for t in range(-max_shift, max_shift + 1):
        y1s = torch.roll(y1, shifts=-t, dims=-1)   # y1[..., u-t] at position u
        y2s = torch.roll(y2, shifts=+t, dims=-1)   # y2[..., u+t] at position u
        diff = (y1s - y2s).abs()                   # (N,1,A,D)
        # Windowed sum along detector axis: collapse N*1*A into batch for conv1d
        flat = diff.reshape(N * A, 1, D)
        cost = F.conv1d(flat, kernel, padding=pad).reshape(N, 1, A, D)
        mid = 0.5 * (y1s + y2s)
        if best_cost is None:
            best_cost = cost
            best_mid = mid
        else:
            take = cost < best_cost
            best_cost = torch.where(take, cost, best_cost)
            best_mid = torch.where(take, mid, best_mid)

    # Interleave original and midpoint views: (k, mid_k, k+1, mid_{k+1}, ...).
    interleaved = torch.stack([sino, best_mid], dim=-2)               # (N,1,A,2,D)
    interleaved = interleaved.reshape(N, 1, 2 * A, D)
    return interleaved if had_channel else interleaved.squeeze(1)


# ---------------------------------------------------------------------------
def _build_dense_projector(geom: FanBeamGeometry, factor: int, device):
    """Make a fresh projector at `factor*` angular sampling for FBP-ing the
    densified residual sinogram."""
    dense_geom = FanBeamGeometry(
        image_size=geom.image_size, pixel_spacing=geom.pixel_spacing,
        n_angles=geom.n_angles * factor, n_det=geom.n_det,
        det_spacing=geom.det_spacing, sod=geom.sod, sdd=geom.sdd,
    )
    return PyronnFanBeamProjector(dense_geom).to(device)


def wu_2015_reconstruct(proj: PyronnFanBeamProjector,
                         dense_proj: PyronnFanBeamProjector,
                         sino: torch.Tensor,
                         n_bands: int, n_outer: int,
                         motion_range: int, motion_window: int,
                         soft_thresh: float) -> torch.Tensor:
    """End-to-end Wu 2015 reconstruction. `sino` shape `(N, 1, A, D)`."""
    # 1. Aliasing-free FBP.
    g = aliasing_free_fbp(proj, sino, n_bands=n_bands)                # (N,1,H,W)

    for it in range(n_outer):
        # 2. Residual sinogram = measured − forward(g)
        with torch.no_grad():
            fp = proj.forward_project(g)                              # (N,1,A,D)
            residual_sino = sino - fp
        # 3. Feature-preserving interpolation → 2x angles
        residual_dense = feature_preserving_interp(
            residual_sino, max_shift=motion_range, window=motion_window)
        # 4. FBP of densified residual through the matching-angle projector
        residual_img = dense_proj.fbp(residual_dense, filter_name="hann")
        # Soft threshold (suppress streaks from un-interpolated micro-structures)
        residual_img = torch.sign(residual_img) * torch.clamp(
            residual_img.abs() - soft_thresh, min=0.0)
        # Merge
        g = g + residual_img
    return g


# ---------------------------------------------------------------------------
def main(out_dir: Path, cfg: dict | None = None) -> dict:
    # Allow env override (set by random-search agent — see
    # scripts/wu_search_agent_standalone.py).
    import os
    env_path = os.environ.get("WU_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        env_cfg = json.loads(Path(env_path).read_text())
        cfg = {**CONFIG, **env_cfg, **(cfg or {})}
        print(f"[solver] Loaded config from {env_path}", flush=True)
    else:
        cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg, default=str)}",
          flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)

    proj = PyronnFanBeamProjector(geom).to(device)
    dense_proj = _build_dense_projector(geom, factor=2, device=device)
    with torch.no_grad():
        val_ref = proj.fbp(val_clean)
        fbp_init = proj.fbp(val_noisy)

    t0 = time.time()
    pred = wu_2015_reconstruct(
        proj, dense_proj, val_noisy,
        n_bands=cfg["wu_n_bands"], n_outer=cfg["wu_n_outer"],
        motion_range=cfg["wu_motion_range"],
        motion_window=cfg["wu_motion_window"],
        soft_thresh=cfg["wu_soft_thresh"],
    )
    train_time = time.time() - t0

    data_range = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ph, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ph, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(fbp_init, val_ph, data_range=data_range).cpu())
    baseline_rmse = float(((fbp_init - val_ph) ** 2).mean().sqrt().cpu())
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(fbp_init[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(
                f"Wu 2015  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})"
                if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "params_M": 0.0, "train_n": 0, "val_n": cfg["val_n"],
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Wu 2015 recon: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"time={train_time:.1f}s", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
