"""Impact of the pyronn filter-padding fix on demo-dl FBP-type solvers.

Demo-DL is the synthetic Wagner-LDCT-style fan-beam phantom test used by
`pentathlon/demo_dl_reference/solver_fbp_baseline.py` and friends. The
FBP filter in `PyronnFanBeamProjector.filter_sino` was patched today to
(i) zero-pad the sinogram to ``M = 2 · next_pow2(N_det)`` along the
detector axis (was previously: no pad, length-N circular convolution)
and (ii) halve the Kak-Slaney truncation DC residual.

For n_det=736 (demo-dl), the variants are:
  - V0 baseline (no pad, no DC fix)
  - V1 pad to 2N=1472 + H[0]/2
  - V2 pad to 2·next_pow2(N)=2048 + H[0]/2   (current production)

This script reproduces the demo-dl simulation (same phantoms, same noise),
runs the FBP through each variant, calibrates with `evaluate_calibrated`
(FOV-masked, same as a real solver_fbp_baseline run would now), and
prints a table over `val_n` cases. The headroom relative to the unpatched
V0 quantifies the padding fix's direct impact on demo-intensity scores.

Also runs a second pipeline using the no-noise (clean) sinogram so we can
separate "filter improvement when input is perfect" from "filter
improvement when input is noisy" (the actual solver_fbp_baseline metric).
"""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch
import hdf5plugin  # noqa: F401

from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS
from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim, intensity_calibrate, fov_mask


# ───────────────────────── three filter variants ─────────────────────────

def _kak_slaney_h(M: int, det_spacing: float) -> np.ndarray:
    h = np.zeros(M, dtype=np.float64)
    h[0] = 0.25 / (det_spacing ** 2)
    odd = -1.0 / (math.pi * math.pi * det_spacing ** 2)
    for i in range(1, M):
        if i < M / 2 and (i % 2) == 1:
            h[i] = odd / (i * i)
        elif i >= M / 2:
            tmp = M - i
            if (tmp % 2) == 1:
                h[i] = odd / (tmp * tmp)
    return h


def filter_variant(sino: torch.Tensor, det_spacing: float, *,
                    pad_kind: str, halve_h0: bool) -> torch.Tensor:
    """``pad_kind`` ∈ {"none", "2N", "2*pow2"}."""
    N = sino.shape[-1]
    if pad_kind == "none":
        M = N
        x = sino
    elif pad_kind == "2N":
        M = 2 * N
        pad = torch.zeros(sino.shape[:-1] + (N,), device=sino.device, dtype=sino.dtype)
        x = torch.cat([sino, pad], dim=-1)
    elif pad_kind == "2*pow2":
        next_pow2 = 1 << (int(N - 1).bit_length())
        M = 2 * next_pow2
        n_pad = M - N
        pad = torch.zeros(sino.shape[:-1] + (n_pad,), device=sino.device, dtype=sino.dtype)
        x = torch.cat([sino, pad], dim=-1)
    else:
        raise ValueError(pad_kind)
    h = _kak_slaney_h(M, det_spacing)
    f_np = np.real(np.fft.fft(h)).astype(np.float32)
    if halve_h0:
        f_np[0] *= 0.5
    f = torch.as_tensor(f_np, device=sino.device, dtype=sino.dtype)
    spec = torch.fft.fft(x, dim=-1, norm="ortho")
    y_full = torch.fft.ifft(spec * f, dim=-1, norm="ortho").real
    return y_full[..., :N]


def fbp_with(proj: PyronnFanBeamProjector, sino, pad_kind, halve_h0):
    """Run the rest of pyronn.fbp() but with our diagnostic filter."""
    rw = proj._redundancy_weights
    A = sino.shape[-2]
    if A != rw.shape[0]:
        raise NotImplementedError("only full-set sinograms in this diagnostic")
    sino_w = sino * rw
    filt = filter_variant(sino_w, proj.geom.det_spacing,
                            pad_kind=pad_kind, halve_h0=halve_h0)
    return proj.back_project(filt)


# ─────────────────────────── eval pipeline ────────────────────────────

def cal_eval(pred, truth, *, display_max, fov):
    pc = intensity_calibrate(pred.clamp_min(0.0), truth, display_max=display_max)
    if fov is not None:
        pc = pc * fov
        truth = truth * fov
    dr = float(display_max)
    return (
        float(ssim(pc, truth, data_range=dr).cpu()),
        float(psnr(pc, truth, data_range=dr).cpu()),
        float(((pc - truth) ** 2).mean().sqrt().cpu()),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(DEMO_DL_DEFAULTS)
    print(f"\ndemo-dl DEFAULTS:")
    for k in ("image_size", "pixel_spacing", "n_angles", "n_det",
              "det_spacing", "sod", "sdd", "noise_i0", "noise_sigma_e",
              "val_n", "display_max"):
        print(f"  {k:<15} = {cfg[k]}")

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )
    proj = PyronnFanBeamProjector(geom).to(device)

    # demo-dl phantoms + Poisson(I0)+AWGN(σ_e) noisy sino — same as the solver.
    val_n = int(cfg["val_n"])
    seed = int(cfg["seed"]) + 1000
    # random_ellipses_phantom returns (1, 1, H, W); concatenate along batch dim.
    phs = torch.cat([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10,
                                 seed=seed + i, device=device)
        for i in range(val_n)
    ], dim=0)
    with torch.no_grad():
        clean_sino = proj.forward_project(phs)
        noisy_sino = simulate_low_dose(clean_sino, i0=cfg["noise_i0"],
                                         sigma_e=cfg["noise_sigma_e"])

    fov = fov_mask(geom.image_size, device=device, dtype=torch.float32)
    dmax = float(cfg["display_max"])

    variants = [
        ("V0 baseline    (no pad, full H[0])",  dict(pad_kind="none",   halve_h0=False)),
        ("V1 +2N pad     + H[0]/2",             dict(pad_kind="2N",     halve_h0=True)),
        ("V2 +2*pow2 pad + H[0]/2  (NEW)",      dict(pad_kind="2*pow2", halve_h0=True)),
    ]

    print(f"\nshape debug: phs {tuple(phs.shape)}  clean_sino {tuple(clean_sino.shape)}  "
          f"noisy_sino {tuple(noisy_sino.shape)}")
    for label, kw in variants:
        with torch.no_grad():
            rec_clean = fbp_with(proj, clean_sino, **kw)
            rec_noisy = fbp_with(proj, noisy_sino, **kw)
        print(f"  variant={label!r}: rec_clean shape={tuple(rec_clean.shape)}")
        # Per-case calibrated metrics
        ss_c, ps_c, rm_c = [], [], []
        ss_n, ps_n, rm_n = [], [], []
        for i in range(val_n):
            s, p, r = cal_eval(rec_clean[i:i+1], phs[i:i+1], display_max=dmax, fov=fov)
            ss_c.append(s); ps_c.append(p); rm_c.append(r)
            s, p, r = cal_eval(rec_noisy[i:i+1], phs[i:i+1], display_max=dmax, fov=fov)
            ss_n.append(s); ps_n.append(p); rm_n.append(r)
        print(f"\n{label}")
        print(f"  CLEAN sino  (filter quality alone): "
              f"SSIM={np.mean(ss_c):.4f}  PSNR={np.mean(ps_c):5.2f}  RMSE={np.mean(rm_c):.4e}")
        print(f"  NOISY sino  (solver_fbp_baseline metric): "
              f"SSIM={np.mean(ss_n):.4f}  PSNR={np.mean(ps_n):5.2f}  RMSE={np.mean(rm_n):.4e}")


if __name__ == "__main__":
    main()
