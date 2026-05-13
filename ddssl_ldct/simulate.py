"""Low-dose CT noise simulation following the Yu/Wu/Fletcher Mayo model:

    I = Poisson(I0 * exp(-p)) + N(0, sigma_e^2)
    p_noisy = -log(max(I, 1) / I0)

where p are the (noise-free) line integrals produced by the forward projector,
I0 is the incident photon count per detector pixel at the reference dose, and
sigma_e is electronic read-out noise.

Reducing I0 corresponds to reducing dose. Compared to clinical scanners this is
a simplified model (no bowtie, no scatter, monochromatic), but matches the noise
model used in Wagner et al. 2022 (and Yu et al. 2012) for low-dose simulation.
"""
from __future__ import annotations
import torch


def simulate_low_dose(sino_clean: torch.Tensor,
                      i0: float = 1e4,
                      sigma_e: float = 5.0,
                      seed: int | None = None) -> torch.Tensor:
    """Apply Poisson photon + Gaussian electronic noise to a clean sinogram.

    sino_clean: (B,1,A,D) of line integrals (≥ 0).
    Returns: noisy line integrals of the same shape.
    """
    if seed is not None:
        g = torch.Generator(device=sino_clean.device).manual_seed(seed)
    else:
        g = None
    expected = i0 * torch.exp(-sino_clean.clamp_min(0.0))
    photons = torch.poisson(expected, generator=g) if g is not None else torch.poisson(expected)
    if sigma_e > 0:
        if g is not None:
            photons = photons + torch.randn(photons.shape, generator=g,
                                            device=sino_clean.device,
                                            dtype=sino_clean.dtype) * sigma_e
        else:
            photons = photons + torch.randn_like(photons) * sigma_e
    photons = photons.clamp_min(1.0)
    return -torch.log(photons / i0)


def split_projections(sino: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sinogram views into odd/even subsets along the angle axis.

    sino: (B,1,A,D) → two (B,1,A/2,D) tensors. The two subsets are independent
    realizations of the same scan at half the angular sampling — the foundation
    of the Noise2Inverse-style self-supervised loss used in 2211.01111.
    """
    even = sino[..., 0::2, :]
    odd = sino[..., 1::2, :]
    return even, odd
