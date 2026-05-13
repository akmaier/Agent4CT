from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> torch.Tensor:
    if data_range is None:
        data_range = float(target.amax() - target.amin())
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    k = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    w = torch.exp(-0.5 * (k / sigma) ** 2)
    w = w / w.sum()
    return w.outer(w)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         data_range: float | None = None,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if data_range is None:
        data_range = float(target.amax() - target.amin())
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype)[None, None]
    pad = window_size // 2
    mu_x = F.conv2d(pred, w, padding=pad)
    mu_y = F.conv2d(target, w, padding=pad)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    sigma_xx = F.conv2d(pred * pred, w, padding=pad) - mu_xx
    sigma_yy = F.conv2d(target * target, w, padding=pad) - mu_yy
    sigma_xy = F.conv2d(pred * target, w, padding=pad) - mu_xy
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    return (num / den).mean()
