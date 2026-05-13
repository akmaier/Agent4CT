"""Denoising operators used in the dual-domain pipeline.

Two variants are exposed so the same pipeline can use either:
  - SmallUNet:        the CNN baseline in 2211.01111. Used at both proj/img domain.
  - TrainableBilateralFilter2d: the 4-parameter bilateral filter from
    2201.10345 (Wagner et al., Med. Phys. 2022). The bilateral kernel is
    evaluated explicitly so the layer is differentiable in σx, σy, σr and the
    input image; it stays small enough to remain interpretable.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Small U-Net (3 levels). Channels are kept low so a smoke test fits on CPU.
# --------------------------------------------------------------------------

def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class _DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(_pick_groups(c_out), c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(_pick_groups(c_out), c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """3-level U-Net with residual prediction. Predicts the noise component."""

    def __init__(self, c: int = 16, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.enc1 = _DoubleConv(1, c)
        self.enc2 = _DoubleConv(c, c * 2)
        self.enc3 = _DoubleConv(c * 2, c * 4)
        self.bot = _DoubleConv(c * 4, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = _DoubleConv(c * 8, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = _DoubleConv(c * 4, c)
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = _DoubleConv(c * 2, c)
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # Pad to multiple of 8 so three pools fit cleanly.
        h, w = x.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            x_in = F.pad(x, (0, pw, 0, ph), mode='reflect')
        else:
            x_in = x

        e1 = self.enc1(x_in)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bot(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        y = self.head(d1)
        if ph or pw:
            y = y[..., :h, :w]
        return x - y if self.residual else y


# --------------------------------------------------------------------------
# Trainable bilateral filter (4 params): σx, σy, σr; weights via Gaussians.
# Differentiable wrt σ via the explicit kernel computation.
# --------------------------------------------------------------------------

class TrainableBilateralFilter2d(nn.Module):
    def __init__(self, kernel_size: int = 7,
                 sigma_x: float = 1.0, sigma_y: float = 1.0, sigma_r: float = 0.05):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = kernel_size
        self.log_sx = nn.Parameter(torch.tensor(math.log(sigma_x)))
        self.log_sy = nn.Parameter(torch.tensor(math.log(sigma_y)))
        self.log_sr = nn.Parameter(torch.tensor(math.log(sigma_r)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 1, "single-channel"
        k = self.k
        r = k // 2
        sx = torch.exp(self.log_sx)
        sy = torch.exp(self.log_sy)
        sr = torch.exp(self.log_sr)

        ys = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        xs = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        spatial = torch.exp(-0.5 * ((gx / sx) ** 2 + (gy / sy) ** 2))         # (k,k)

        x_pad = F.pad(x, (r, r, r, r), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k)                              # (B, k*k, H*W)
        patches = patches.reshape(B, k * k, H, W)
        center = x                                                            # (B,1,H,W)
        range_w = torch.exp(-0.5 * ((patches - center) / sr) ** 2)            # (B,k*k,H,W)
        spatial_flat = spatial.reshape(k * k, 1, 1)                           # (k*k,1,1)
        w = range_w * spatial_flat
        num = (w * patches).sum(dim=1, keepdim=True)
        den = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return num / den
