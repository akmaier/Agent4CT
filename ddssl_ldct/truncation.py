"""Sinogram truncation correction (water-cylinder edge extrapolation).

When the scanned object extends past the field-of-measurement, the fan
projections are cut off at the detector edges and FBP throws a bright rim
+ cupping. The classic fix (Hsieh et al., Med. Phys. 2004, "A novel
reconstruction algorithm to extend the CT scan field-of-view") extends
each view past the truncation boundary by a water-cylinder profile

    f(t) = 2 * mu_w * sqrt(R^2 - (t - t_c)^2)

matched in value AND outward slope to the measured boundary, decaying
smoothly to zero. The FBP then runs on a widened detector so the ramp
filter never sees the discontinuity.

Validated on the 10 Wagner Mayo patients (SLURM 763608, 2026-06-13):
HD SSIM_cal 0.915 -> 0.943, biggest on the 400 mm-FOV (largest) patients,
near-no-op (self-gating) on the 340 mm patients. See docs/findings.md.

The torch implementation here is the production path used by
`PyronnFanBeamProjector(truncation=...)`. `scripts/
compare_gt_hd_ld_fbp_wagner_trunc.py` carries the reference numpy version.
"""
from __future__ import annotations

import torch


def water_cylinder_extrapolate(sino: torch.Tensor, du_iso: float, pad: int,
                                mu_water: float = 0.02, edge_k: int = 7
                                ) -> torch.Tensor:
    """Extend a fan sinogram on both detector edges by a water cylinder.

    Args:
        sino: ``(..., D)`` line-integral sinogram; extrapolation is along
            the last (detector-channel) axis, independent per leading view.
        du_iso: detector-channel sampling at isocentre (mm) =
            ``det_spacing * sod / sdd``. Sets the physical length unit.
        pad: channels added per side (output width ``D + 2*pad``).
        mu_water: water linear attenuation (mm^-1); sets the decay width.
        edge_k: number of boundary channels used to estimate value + slope.

    Returns:
        ``(..., D + 2*pad)`` sinogram with the measured data centred and the
        pads filled by the value+slope-matched water cylinder decaying to 0.
        Self-gating: a near-zero edge yields a tiny R, hence a ~0 pad.
    """
    if pad <= 0:
        return sino
    # edge_k must be >= 2 (need a slope from >=2 samples). Guard the nasty
    # `sino[..., -edge_k:]` footgun: edge_k=0 silently slices the WHOLE axis
    # (since `-0 == 0`), producing a "(D) vs (0)" broadcast error downstream.
    edge_k = int(edge_k)
    if edge_k < 2:
        raise ValueError(f"edge_k must be >= 2, got {edge_k!r} "
                         f"(check the caller's argument order)")
    *lead, D = sino.shape
    dev, dt = sino.device, sino.dtype
    eps = 1e-4
    two_mu = 2.0 * mu_water
    four_mu2 = 4.0 * mu_water ** 2
    Rmax = pad * du_iso

    # least-squares slope weights for k equally-spaced points (per channel step)
    x = torch.arange(edge_k, device=dev, dtype=dt)
    xb = x.mean()
    sxx = ((x - xb) ** 2).sum()
    w = (x - xb) / sxx                                   # (edge_k,)
    ch = torch.arange(pad, device=dev, dtype=dt)          # (pad,)

    out = sino.new_zeros(*lead, D + 2 * pad)
    out[..., pad:pad + D] = sino

    def _cylinder(p_b, s_in, dist):
        # p_b: (...,) boundary value; s_in: (...,) inward d p / d channel;
        # dist: (pad,) outward distance (channels). Returns (..., pad).
        g = (-s_in / du_iso).clamp(max=0.0)               # outward grad, non-+
        tau = -g * p_b / four_mu2                          # mm, >= 0
        R = torch.sqrt((p_b / two_mu) ** 2 + tau ** 2).clamp(max=Rmax)
        t = tau[..., None] + dist * du_iso                 # (..., pad)
        inside = t < R[..., None]
        val = two_mu * torch.sqrt((R[..., None] ** 2 - t ** 2).clamp(min=0.0))
        return torch.where(inside, val, torch.zeros_like(val))

    # left edge (low u): channel j in [0,pad) is (pad - j) channels outward
    left = sino[..., :edge_k]
    p_b = left.mean(dim=-1).clamp(min=eps)
    s_in = (left * w).sum(dim=-1)
    out[..., :pad] = _cylinder(p_b, s_in, (pad - ch))

    # right edge (high u): inward = decreasing channel -> flip the window;
    # channel j in [0,pad) of the right pad is (j + 1) channels outward
    right = sino[..., -edge_k:]
    p_b = right.mean(dim=-1).clamp(min=eps)
    s_in = (right.flip(-1) * w).sum(dim=-1)
    out[..., pad + D:] = _cylinder(p_b, s_in, (ch + 1.0))
    return out
