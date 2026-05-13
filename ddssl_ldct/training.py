"""End-to-end self-supervised dual-domain training (arXiv:2211.01111).

Pipeline per training step (Fig. 2 / Eq. 3-5 of the paper):

    x = noisy low-dose sinogram (full set of A views, A even)
    x_a, x_b = split_projections(x)            # odd / even views
    y_hat_a = D_img( R_half( D_proj(x_a) ) )   # full pipeline on split A
    y_tgt_b = R_half( x_b )                    # plain FBP on split B
    loss    = MSE(y_hat_a, y_tgt_b)

Then symmetric pass with a/b swapped; gradients backpropagate through every
trainable parameter via the differentiable FBP. No high-dose target is used.

At inference (Eq. 6 of the paper) the prediction is the average of the two
half-set dual-domain predictions.

The reconstruction backbone is PYRO-NN — see `pyronn_projector.py`. PYRO-NN
requires CUDA; this module deliberately does *not* offer a CPU fallback.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import FanBeamGeometry
from .pyronn_projector import PyronnFanBeamProjector
from .simulate import split_projections


class DualDomainPipeline(nn.Module):
    def __init__(self,
                 geometry: FanBeamGeometry,
                 proj_denoiser: nn.Module,
                 image_denoiser: nn.Module):
        super().__init__()
        assert geometry.n_angles % 2 == 0, "n_angles must be even for split"
        self.geometry = geometry
        self.proj_denoiser = proj_denoiser
        self.image_denoiser = image_denoiser
        self.R_full = PyronnFanBeamProjector(geometry)
        self.R_half = PyronnFanBeamProjector(geometry.split_angles()[0])

    # ------------------------------------------------------------------ #

    def _half_pipeline(self, sino_half: torch.Tensor) -> torch.Tensor:
        x = self.proj_denoiser(sino_half)
        r = self.R_half.fbp(x)
        return self.image_denoiser(r)

    def training_step(self, sino_full: torch.Tensor) -> dict[str, torch.Tensor]:
        x_a, x_b = split_projections(sino_full)
        y_hat_a = self._half_pipeline(x_a)
        with torch.no_grad():
            y_tgt_b = self.R_half.fbp(x_b)
        loss_ab = F.mse_loss(y_hat_a, y_tgt_b)

        y_hat_b = self._half_pipeline(x_b)
        with torch.no_grad():
            y_tgt_a = self.R_half.fbp(x_a)
        loss_ba = F.mse_loss(y_hat_b, y_tgt_a)

        loss = 0.5 * (loss_ab + loss_ba)
        return {"loss": loss, "loss_ab": loss_ab.detach(), "loss_ba": loss_ba.detach()}

    @torch.no_grad()
    def predict(self, sino_full: torch.Tensor) -> torch.Tensor:
        """Inference (Eq. 6): average of the two half-set dual-domain predictions."""
        x_a, x_b = split_projections(sino_full)
        return 0.5 * (self._half_pipeline(x_a) + self._half_pipeline(x_b))


def train(pipeline: DualDomainPipeline,
          dataset_sinos: torch.Tensor,
          epochs: int = 50,
          batch_size: int = 1,
          lr: float = 1e-3,
          device: str = "cuda",
          log_every: int = 10,
          val_sinos: torch.Tensor | None = None,
          val_ground_truth: torch.Tensor | None = None) -> dict:
    """Minimal training loop. dataset_sinos: (N,1,A,D)."""
    from .metrics import psnr, ssim
    pipeline.to(device)
    opt = torch.optim.Adam(pipeline.parameters(), lr=lr)
    history = {"loss": [], "val_psnr": [], "val_ssim": []}
    N = dataset_sinos.shape[0]
    for ep in range(epochs):
        pipeline.train()
        perm = torch.randperm(N)
        running, steps = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch = dataset_sinos[idx].to(device)
            losses = pipeline.training_step(batch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            running += float(losses["loss"].detach().cpu()); steps += 1
        mean_loss = running / max(steps, 1)
        history["loss"].append(mean_loss)
        line = f"[epoch {ep+1:3d}/{epochs}] loss={mean_loss:.5f}"
        if val_sinos is not None and val_ground_truth is not None and (
                (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1):
            pipeline.eval()
            with torch.no_grad():
                pred = pipeline.predict(val_sinos.to(device))
                p = float(psnr(pred, val_ground_truth.to(device)).cpu())
                s = float(ssim(pred, val_ground_truth.to(device)).cpu())
            history["val_psnr"].append((ep + 1, p))
            history["val_ssim"].append((ep + 1, s))
            line += f"   val PSNR={p:.2f} dB  SSIM={s:.3f}"
        print(line, flush=True)
    return history
