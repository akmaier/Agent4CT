"""Synthetic phantoms (random ellipses + Shepp-Logan) for sanity testing.

Used in lieu of the Mayo LDCT-and-Projection-data set. The intensities are
roughly in units of linear attenuation (1/mm) for soft-tissue ~0.02; the
exact scale is not important as long as it is consistent between simulation
and reconstruction.
"""
from __future__ import annotations
import math
import torch


def shepp_logan(size: int = 256, device=None, dtype=torch.float32) -> torch.Tensor:
    """Modified Shepp-Logan phantom, returned as (1, 1, size, size)."""
    ellipses = [
        # x0, y0, a, b, theta(rad), intensity
        (0.0,    0.0,    0.69,   0.92,   0.0,           1.00),
        (0.0,   -0.0184, 0.6624, 0.874,  0.0,          -0.80),
        (0.22,   0.0,    0.11,   0.31,  -math.pi*0.1,  -0.20),
        (-0.22,  0.0,    0.16,   0.41,   math.pi*0.1,  -0.20),
        (0.0,    0.35,   0.21,   0.25,   0.0,           0.10),
        (0.0,    0.1,    0.046,  0.046,  0.0,           0.10),
        (0.0,   -0.1,    0.046,  0.046,  0.0,           0.10),
        (-0.08, -0.605,  0.046,  0.023,  0.0,           0.10),
        (0.0,   -0.605,  0.023,  0.023,  0.0,           0.10),
        (0.06,  -0.605,  0.023,  0.046,  0.0,           0.10),
    ]
    return _draw_ellipses(ellipses, size, device, dtype) * 0.05  # scale to ~0.05 mm⁻¹


def random_ellipses_phantom(size: int = 256, n_ellipses: int = 12,
                            seed: int | None = None,
                            device=None, dtype=torch.float32) -> torch.Tensor:
    """A random-disc/ellipse phantom; coarse stand-in for abdominal CT."""
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(seed)
    ellipses = []
    # Body outline
    ellipses.append((0.0, 0.0, 0.85, 0.65, 0.0, 0.04))
    # Cavity
    ellipses.append((0.0, 0.0, 0.78, 0.55, 0.0, -0.02))
    rnd = lambda lo, hi: (torch.rand((), generator=g).item() * (hi - lo) + lo)
    for _ in range(n_ellipses):
        x0 = rnd(-0.55, 0.55)
        y0 = rnd(-0.35, 0.35)
        a = rnd(0.05, 0.25)
        b = rnd(0.05, 0.25)
        theta = rnd(0.0, math.pi)
        inten = rnd(-0.025, 0.05)
        ellipses.append((x0, y0, a, b, theta, inten))
    return _draw_ellipses(ellipses, size, device, dtype)


def _draw_ellipses(ellipses, size, device, dtype):
    ys = torch.linspace(1.0, -1.0, size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    img = torch.zeros((size, size), device=device, dtype=dtype)
    for x0, y0, a, b, theta, val in ellipses:
        ct, st = math.cos(theta), math.sin(theta)
        Xp = (X - x0) * ct + (Y - y0) * st
        Yp = -(X - x0) * st + (Y - y0) * ct
        mask = (Xp / a) ** 2 + (Yp / b) ** 2 <= 1.0
        img = img + mask.to(dtype) * val
    return img.clamp_min(0.0)[None, None]
