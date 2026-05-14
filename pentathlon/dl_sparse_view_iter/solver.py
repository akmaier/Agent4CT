"""DL-Sparse-View CT solver — UNROLLED ITERATIVE RECON branch (Spawn-agent A).

This solver replaces the image_denoiser of the dual-domain pipeline with an
**unrolled iterative reconstruction** module inspired by the top-five teams in
the AAPM DL-Sparse-View challenge (Sidky & Pan 2022, §III):

    1. Robust-and-stable / ItNet:       U-Net + data-consistency layer (LS step)
    2. YM&RH:                           variational network with proximal U-Net
    3. DEEP UL:                         TV-LS first stage + HF U-Net cleanup
    4. deepx:                           scale-attention image-only
    5. HBB / JSR-Net:                   ADMM, image + sino regularisation

Concretely: the image_denoiser receives the half-set FBP image r and runs K
unrolled steps of denoise -> re-project -> data-fidelity gradient correct.
The data target g = R(r) is the projection of the input image (the only
half-set sinogram structurally available inside image_denoiser without
modifying DualDomainPipeline); the iterate is pulled back toward that
projection with a learnable step-size alpha.

This file MUST stay out of the main agent's way — the main agent edits
pentathlon/dl_sparse_view/solver.py; we are pentathlon/dl_sparse_view_iter/.

The harness (DualDomainPipeline, R_full, R_half, FBP, training loop, metrics)
is identical to the main solver — only build_denoisers() differs.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose, split_projections
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.training import DualDomainPipeline, train
from ddssl_ldct.metrics import psnr, ssim


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3) -> torch.Tensor:
    """Smooth-L1 / Charbonnier loss: mean(sqrt((pred - target)^2 + eps^2)).
    Robust to outliers vs MSE while staying differentiable everywhere.
    """
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def charbonnier_training_step(pipe: DualDomainPipeline,
                              sino_full: torch.Tensor,
                              eps: float = 1e-3) -> dict[str, torch.Tensor]:
    """Drop-in replacement for ``pipe.training_step`` that uses Charbonnier
    instead of MSE. Mirrors the dual-domain N2I loss in ddssl_ldct/training.py:
    half-set A -> half-set B FBP target (and vice versa)."""
    x_a, x_b = split_projections(sino_full)
    y_hat_a = pipe._half_pipeline(x_a)
    with torch.no_grad():
        y_tgt_b = pipe.R_half.fbp(x_b)
    loss_ab = charbonnier_loss(y_hat_a, y_tgt_b, eps=eps)

    y_hat_b = pipe._half_pipeline(x_b)
    with torch.no_grad():
        y_tgt_a = pipe.R_half.fbp(x_a)
    loss_ba = charbonnier_loss(y_hat_b, y_tgt_a, eps=eps)

    loss = 0.5 * (loss_ab + loss_ba)
    return {"loss": loss, "loss_ab": loss_ab.detach(), "loss_ba": loss_ba.detach()}


# ----------------------------------------------------------------------- #
#  CONFIG  —  one change per iteration on this branch.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (fixed — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,

    # Training subset for the 5-min iteration budget.
    "train_n":       400,
    "val_n":         100,

    # Training schedule. Iter-21 reverts epochs 4 -> 6 because the LPD
    # projector overhead is gone and NAFNet's per-pass cost is in the
    # ResStack ballpark (full-resolution conv but no down/up).
    "epochs":        6,
    "batch_size":    1,
    "lr":            1e-4,
    "optimizer":     "adamw",
    "weight_decay":  1e-4,

    # Iter-21: switch architecture family from iterated-denoiser (iter-7
    # ceiling 0.5745) to NAFNet (Chen et al. 2022, "Simple Baselines for
    # Image Restoration"). NAFNet is the SOTA simple-baseline for image
    # restoration: a plain stack of blocks at full resolution, each block
    # is LN -> 1x1 expand -> 3x3 depthwise -> {SimpleGate or ReLU} -> 1x1
    # squeeze -> +x. No BatchNorm. Heavy use of 1x1 channel-mixing.
    # Spawn-B's iter-13 explicit advice: "channel-mixing inside the
    # spatial pathway (e.g. NAFNet 1x1 MLP)" is more promising than
    # SE-gating on a c=32 residual stack. We try that now.
    "unet_c":        32,               # NAFNet uses wider channels than UNet
    "img_denoiser":  "nafnet",         # NEW: NAFNet stack as image_denoiser
    "proj_denoiser": "nafnet",         # NEW: also use NAFNet for proj domain
    # NAFNet block hyperparameters (iter-21 baseline: safe / no SimpleGate).
    "naf_blocks":    6,                # 6 NAF blocks at c=32, full res
    "naf_expand":    2,                # expand 1x1 (c -> 2c), squeeze (2c -> c)
    "naf_dw":        True,             # depthwise 3x3 mid-conv
    "naf_gate":      "gelu",           # iter-38: GELU on iter-36 KEEP substrate (smoother than ReLU)
    "naf_alpha":     0.1,              # EDSR-style learnable residual scaling
                                         #  (spawn-B iter-14 advice: cheap +0.15pp)
    "n_unroll":      1,                # NAFNet is single-pass (not iterated)
    "share_weights": False,            # n/a at K=1
    "n_bf":          0,                # iter-21: no tail BF on legacy paths
    # Iter-29: Build on iter-28 KEEP (NAFNet+SWA+1BF, hr=0.5805, +0.60pp
    # over iter-7). Push the BF tail count to 2 — iter-7's recipe used a
    # variable n_bf, and the iter-7 best from the K=2 UNet family used n_bf=1.
    # On NAFNet, the noise floor after 6 NAF blocks may need more aggressive
    # smoothing; 2 BFs gives independent learnable sigmas at different
    # noise/edge regimes. Cost: +3 params (negligible) and +1 BF unfold per
    # forward pass. If +0.1pp/BF holds, we land at ~0.5905. If diminishing
    # to +0.05pp, we land at ~0.5810 — still well above iter-7 and within
    # noise of iter-28.
    "ema":           True,
    # Iter-40: switch from SWA (uniform mean) to EMA decay=0.999 on iter-38
    # substrate. Full SWA (last-6-of-6) gave hr=0.5985. EMA weighs later
    # steps more heavily — better signal/noise if late epochs are clearly
    # superior. decay=0.999 with ~400 steps/epoch * 6 epochs = 2400 steps
    # gives an effective window of ~1000 steps (half-life ~693), which
    # roughly emphasizes the last 2.5 epochs.
    "ema_decay":     0.999,            # iter-40: EMA decay=0.999 (was 0.0 = SWA)
    # Iter-36: widen SWA window from last-5-of-6 (start_ep=1) to last-6-of-6
    # (start_ep=0). iter-35 KEEP at last-5 (hr=0.5942, +0.22pp over iter-34)
    # confirmed monotonic widening; test if including epoch 0 (the noisiest)
    # still helps or if early-epoch noise finally re-enters the average.
    # If keep: SWA covers full training => may want to test EMA decay > 0
    #   for higher emphasis on later epochs.
    # If discard: last-5 is the sweet spot, move to architecture knobs
    #   (c=40, SimpleGate).
    "ema_start_ep":  0,                # iter-36: last 6 of 6 (was 1 = last-5)
    "ema_every":     1,
    # Iter-31: push BF tail to 3. Saturation test. Curve so far on NAFNet:
    #   n_bf=0 (iter-27, SWA-only):     hr=0.5777
    #   n_bf=1 (iter-28):               hr=0.5805 (+0.28pp over 0)
    #   n_bf=2 (iter-29 KEEP):          hr=0.5868 (+0.63pp over 1, +0.91pp over 0)
    #   n_bf=3 (iter-31, this try):     hr=??? — does the non-linear lift continue?
    # Cost: +3 params, +1 BF unfold pass (~5-10s wall). Each BF can learn its
    # own sigma_x/y/r so the third tail can specialise (e.g. very low sigma_r
    # for hard-edge preservation, or very high sigma_x/y for residual streak
    # smoothing). If +0.6pp holds we land near 0.5928 — would beat team-wide
    # best iter-32 main (0.5906). If saturating at +0.1pp we land near 0.5878.
    # Cross-agent intel says iter-30 wd=0 didn't help here, so this is the
    # next most-likely lift on the existing substrate.
    "naf_n_bf":      3,                # iter-31: 3 trainable BF tails (iter-29 used 2)
    "loss_type":     "mse",
    "lr_schedule":   "constant",
    "lr_min":        1e-5,
    "bf_in_loop":     False,
    "n_bf_tail":      0,
    "alpha_init":    1e-4,             # unused on NAFNet path
    "dc_precondition": "adjoint",       # unused on NAFNet path

    # Noise simulation — fixed so headroom is comparable.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,

    # Intensity calibration — IDENTICAL to main agent's iter-24 fix
    # (data_range = display_max - display_min = 0.05). Without this, the
    # auto-data-range drifts with FBP overshoot iteration-to-iteration and
    # val_score isn't comparable across agents. Ported here so the iter
    # branch competes against the SAME calibration as the main run.
    "display_min":   0.0,
    "display_max":   0.05,
    "val_chunk":     16,           # chunked validation to bound peak memory
}


# ----------------------------------------------------------------------- #
#  NAFNet stack — Chen et al. 2022, "Simple Baselines for Image Restoration".
# ----------------------------------------------------------------------- #

class _LayerNorm2d(nn.Module):
    """LayerNorm over channels, applied per (B, H, W). NAFNet's choice over
    BatchNorm and GroupNorm — invariant to batch size, full-channel mixing
    via learnable affine.
    """

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W).  Normalise over the channel dim.
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _SimpleGate(nn.Module):
    """NAFNet's SimpleGate activation: split tensor in half along channel
    dim, multiply the two halves element-wise. Replaces both Swish/GELU
    AND the multiplicative channel attention in a single op — cheap and
    proven SOTA for image restoration (Chen et al. 2022, §4).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NafBlock(nn.Module):
    """A single NAFNet block.

        x ---> LayerNorm
              -> 1x1 conv  (c -> expand*c)
              -> 3x3 dwconv (groups=expand*c)   (depthwise if dw=True)
              -> gate (SimpleGate halves channels OR ReLU keeps them)
              -> 1x1 conv  (c_after_gate -> c)
              -> + learnable_alpha * residual
              -> +x

    iter-21 baseline:  gate="relu", dw=True, expand=2  (no channel halving)
    iter-22 plan:      gate="simple", dw=True, expand=2  (true NAFNet)
    """

    def __init__(self, c: int, expand: int = 2,
                 dw: bool = True, gate: str = "simple",
                 alpha_init: float = 0.1):
        super().__init__()
        assert gate in ("simple", "relu", "gelu"), gate
        self.gate_kind = gate
        c_mid = c * expand
        self.norm = _LayerNorm2d(c)
        self.pw_in = nn.Conv2d(c, c_mid, kernel_size=1)
        if dw:
            # Depthwise 3x3 — groups=c_mid so each channel has its own kernel.
            self.dw = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1,
                                groups=c_mid)
        else:
            # Plain 3x3 if user wants the ablation.
            self.dw = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        if gate == "simple":
            self.gate = _SimpleGate()
            c_after = c_mid // 2
            assert c_mid % 2 == 0, "SimpleGate needs even c_mid"
        elif gate == "gelu":
            # iter-38: GELU (modern default for image restoration networks,
            # NAFNet baseline before SimpleGate). Same param count as ReLU
            # but smoother gradient near 0, preserves negative info.
            self.gate = nn.GELU()
            c_after = c_mid
        else:
            self.gate = nn.ReLU(inplace=True)
            c_after = c_mid
        self.pw_out = nn.Conv2d(c_after, c, kernel_size=1)
        # EDSR-style learnable residual scale (spawn-B iter-14: free +0.15pp).
        # Initialised at 0.1 — gentle near-identity start.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # Zero-init the final 1x1 so the block starts as identity.
        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.pw_in(h)
        h = self.dw(h)
        h = self.gate(h)
        h = self.pw_out(h)
        return x + self.alpha * h


class NafNetStack(nn.Module):
    """Plain stack of NAFNet blocks at full resolution.

    A *single-pass* image (or projection) denoiser — no iteration, no
    down/up-sampling, no skip connections beyond the per-block residual.
    Stem: 3x3 conv (1 -> c).  Body: n blocks.  Head: 3x3 conv (c -> 1).
    Predicts the noise residual (residual=True): output = x - tail(blocks(stem(x))).
    Iter-28 adds an OPTIONAL bilateral-filter tail (n_bf>=1) that applies
    iter-7's Wagner-style edge-preserving smoothing after the CNN body —
    the trick that gave the K=2 UNet family +0.15pp on top of its 0.5730
    baseline. We test if it composes with NAFNet+SWA (iter-27 0.5777).

    iter-21 baseline: 6 blocks @ c=32, gate=ReLU, dw=True, n_bf=0.
    """

    def __init__(self, c: int = 32, n_blocks: int = 6,
                 expand: int = 2, dw: bool = True,
                 gate: str = "simple", alpha_init: float = 0.1,
                 residual: bool = True,
                 n_bf: int = 0, bf_kernel: int = 7,
                 bf_sigma_x: float = 1.5, bf_sigma_y: float = 1.5,
                 bf_sigma_r: float = 0.01):
        super().__init__()
        self.residual = residual
        self.stem = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            NafBlock(c=c, expand=expand, dw=dw, gate=gate,
                     alpha_init=alpha_init)
            for _ in range(n_blocks)
        ])
        self.head = nn.Conv2d(c, 1, kernel_size=3, padding=1)
        # Zero-init head so the network starts as the identity (predict 0
        # residual <=> output equals input). Same convention as SmallUNet.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Iter-28: optional trainable BF tail (Wagner's iter-7 trick).
        self.bfs = nn.ModuleList([
            TrainableBilateralFilter2d(
                kernel_size=bf_kernel,
                sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
            for _ in range(n_bf)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        y = self.head(h)
        out = x - y if self.residual else y
        for bf in self.bfs:
            out = bf(out)
        return out


# ----------------------------------------------------------------------- #
#  Unrolled-iterative image denoiser.
# ----------------------------------------------------------------------- #

class IteratedUNetBF(nn.Module):
    """Shared-weight iterated U-Net + bilateral-filter tail.

    A *recurrent* denoiser: the same SmallUNet is applied K times in sequence
    to the half-set FBP, then K trainable bilateral filters refine the
    output. With shared weights, the parameter count stays at the iter-16
    main-agent baseline (~0.466 M) while the *effective depth* doubles —
    a cheap structural prior for iterative refinement.

    Inspired by Sidky 2022 §III: ItNet (Robust-and-stable, K=4 unrolled
    U-Net + DC) and DEEP UL's HF U-Net cleanup. We drop the explicit
    data-consistency (R^T R) step (too expensive at our 5-min budget and
    diverges with FBP-preconditioning per iter-1/2) and rely solely on
    iterated learned denoising.

    Iter-14 of the main agent showed *unshared* K=2 stacked U-Nets overfit
    (0.93 M params); shared K=2 keeps the param budget tight while still
    iterating the denoiser.
    """

    def __init__(self, c: int = 16, K: int = 2, n_bf: int = 1,
                 bf_kernel: int = 7, bf_sigma_x: float = 1.5,
                 bf_sigma_y: float = 1.5, bf_sigma_r: float = 0.01,
                 bf_in_loop: bool = False, n_bf_tail: int | None = None):
        """Iterated denoiser with optional in-loop bilateral filter (iter-19).

        - bf_in_loop=False (legacy, iter-7 best): K applications of U-Net,
          then `n_bf` tail BFs in series. Default for backward compat.
        - bf_in_loop=True (iter-19 candidate A): each of the K steps is
          (unet -> BF), with per-step trainable BFs. `n_bf_tail` extra BFs
          can still be applied after the loop (set to 0 to disable tail).
          Inspired by Sidky 2022 §III ItNet: edge-preserving regularisation
          interleaved with learned denoising at every iteration.
        """
        super().__init__()
        assert K >= 1
        self.K = K
        self.bf_in_loop = bool(bf_in_loop)
        self.unet = SmallUNet(c=c, residual=True)
        if self.bf_in_loop:
            # One BF per unroll step (K total) — each step has its own
            # learnable sigma_x/y/r so the smoothing strength can adapt
            # to the iteration index (early step: stronger smoothing on
            # noisy FBP; late step: gentle to preserve sharpness).
            self.bfs_loop = nn.ModuleList([
                TrainableBilateralFilter2d(
                    kernel_size=bf_kernel,
                    sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
                for _ in range(K)
            ])
            # Optional tail BFs (default 0 when in-loop is active).
            n_tail = n_bf if n_bf_tail is None else int(n_bf_tail)
            self.bfs = nn.ModuleList([
                TrainableBilateralFilter2d(
                    kernel_size=bf_kernel,
                    sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
                for _ in range(n_tail)
            ])
        else:
            self.bfs_loop = nn.ModuleList()
            self.bfs = nn.ModuleList([
                TrainableBilateralFilter2d(
                    kernel_size=bf_kernel,
                    sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
                for _ in range(n_bf)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bf_in_loop:
            # iter-19: edge-preserving smoothing INSIDE the unroll
            for k in range(self.K):
                x = self.unet(x)
                x = self.bfs_loop[k](x)
        else:
            # iter-7 legacy: K unets first, then tail BFs.
            for _ in range(self.K):
                x = self.unet(x)
        for bf in self.bfs:
            x = bf(x)
        return x


class UnrolledLPDImageDenoiser(nn.Module):
    """Learned Primal-Dual-style unrolled iterative refinement, in the image
    domain only. K unrolled steps of:

        z_k     = U(x_{k-1})                          # learned denoiser
        x_k     = z_k - alpha_k * R_half^T (R_half z_k - g)

    where g = R_half(input) is the projection of the half-set FBP input image
    — the only data tensor structurally available inside an image_denoiser
    without changing DualDomainPipeline. This is an LPD-lite: the regulariser
    is a CNN, the data-fidelity step is a single gradient on
    ||R z - g||^2. After K steps a 4-param trainable bilateral filter
    sharpens edges (cheap, +3 params, well-tested on iter-16 of the main
    run).

    Parameters: K * (U-Net params)  +  K alphas  +  3 BF params per BF.
    Shared-weight K=2 with c=16:  ~0.23 M params (same order as iter-16).
    """

    def __init__(self, geometry: FanBeamGeometry,
                 c: int = 16, K: int = 2, alpha_init: float = 0.1,
                 n_bf: int = 1, share_weights: bool = False,
                 dc_precondition: str = "fbp"):
        super().__init__()
        assert K >= 1
        assert dc_precondition in ("fbp", "adjoint"), dc_precondition
        self.K = K
        self.share_weights = share_weights
        self.dc_precondition = dc_precondition
        # Per-step or shared U-Net regulariser.
        if share_weights:
            self.unet = SmallUNet(c=c, residual=True)
        else:
            self.unets = nn.ModuleList(
                [SmallUNet(c=c, residual=True) for _ in range(K)]
            )
        # Per-step learnable log-alpha (so it stays positive after exp).
        self.log_alpha = nn.Parameter(
            torch.full((K,), math.log(max(alpha_init, 1e-6)))
        )
        # Half-set projector — re-uses the same PYRO-NN geometry as the
        # outer pipeline's R_half (split_angles()[0]).
        self.R_half = PyronnFanBeamProjector(geometry.split_angles()[0])
        # Optional bilateral tail (Wagner 2022).
        self.bfs = nn.ModuleList([
            TrainableBilateralFilter2d(kernel_size=7,
                                       sigma_x=1.5, sigma_y=1.5, sigma_r=0.01)
            for _ in range(n_bf)
        ])

    def _denoiser(self, k: int, x: torch.Tensor) -> torch.Tensor:
        if self.share_weights:
            return self.unet(x)
        return self.unets[k](x)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # The half-set FBP image is the *data anchor* for our DC term.
        # g acts like the "measurement we trust" inside the image domain.
        # Computed once and held fixed across the K unrolled steps.
        with torch.no_grad():
            g = self.R_half.forward_project(x_in)
        x = x_in
        for k in range(self.K):
            z = self._denoiser(k, x)                              # learned regulariser
            r_z = self.R_half.forward_project(z)                  # forward project
            residual = r_z - g
            if self.dc_precondition == "fbp":
                # FBP-preconditioned descent: the residual is filtered + back-
                # projected, putting `grad` in image-space coordinates with
                # magnitude comparable to z. Standard trick for iterative
                # learned recon to avoid R^T R's huge spectral norm. The
                # `alpha` then sits near 0.1-1.
                grad = self.R_half.fbp(residual)
            else:
                grad = self.R_half.back_project(residual)         # raw adjoint
            alpha = torch.exp(self.log_alpha[k])
            x = z - alpha * grad
        for bf in self.bfs:
            x = bf(x)
        return x


def _build_one_denoiser(kind: str, c: int, cfg: dict,
                        geometry: FanBeamGeometry) -> nn.Module:
    """Construct a single denoiser by kind. Shared between proj+image slots."""
    if kind == "nafnet":
        # Iter-28: only enable BF tail on the IMAGE slot via cfg["naf_n_bf_img"];
        # the projection slot stays BF-free (BF is image-domain edge smoothing).
        # cfg["naf_n_bf"] is the catch-all default and applies to both slots.
        n_bf_naf = int(cfg.get("naf_n_bf", 0))
        return NafNetStack(
            c=c,
            n_blocks=int(cfg.get("naf_blocks", 6)),
            expand=int(cfg.get("naf_expand", 2)),
            dw=bool(cfg.get("naf_dw", True)),
            gate=str(cfg.get("naf_gate", "simple")),
            alpha_init=float(cfg.get("naf_alpha", 0.1)),
            residual=True,
            n_bf=n_bf_naf,
        )
    if kind == "iter_unet_bf":
        return IteratedUNetBF(
            c=c,
            K=cfg.get("n_unroll", 2),
            n_bf=cfg.get("n_bf", 1),
            bf_in_loop=cfg.get("bf_in_loop", False),
            n_bf_tail=cfg.get("n_bf_tail", None),
        )
    if kind == "unrolled_lpd":
        return UnrolledLPDImageDenoiser(
            geometry=geometry,
            c=c,
            K=cfg.get("n_unroll", 2),
            alpha_init=cfg.get("alpha_init", 0.1),
            n_bf=cfg.get("n_bf", 1),
            share_weights=cfg.get("share_weights", False),
            dc_precondition=cfg.get("dc_precondition", "fbp"),
        )
    if kind == "unet_plus_bf":
        from pentathlon.dl_sparse_view.solver import UNetPlusBF
        return UNetPlusBF(c=c, n_bf=cfg.get("n_bf", 1))
    # default: small U-Net (matches main-agent iter-16 best)
    return SmallUNet(c=c, residual=True)


def build_denoisers(cfg: dict, geometry: FanBeamGeometry
                    ) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser).

    Both slots are individually configurable via cfg["proj_denoiser"] and
    cfg["img_denoiser"]; default proj is SmallUNet to match main-agent
    iter-16. iter-21 introduces "nafnet" which can be plugged into either
    slot. Iter-28: BF tail only applies to the image slot — the proj slot
    builds with naf_n_bf=0 forced, since bilateral filtering belongs in
    the image domain (edge smoothing post-FBP).
    """
    c = cfg["unet_c"]
    proj_type = cfg.get("proj_denoiser", "smallunet")
    img_type = cfg.get("img_denoiser", "unrolled_lpd")
    # Image slot uses the full BF count.
    img = _build_one_denoiser(img_type, c, cfg, geometry)
    # Proj slot: force BF count to 0 (don't bilateral-smooth sinograms).
    proj_cfg = dict(cfg, naf_n_bf=0)
    proj = _build_one_denoiser(proj_type, c, proj_cfg, geometry)
    return proj, img


# ----------------------------------------------------------------------- #
#  Data + training harness — identical to the main solver.
# ----------------------------------------------------------------------- #

def build_geometry(cfg: dict) -> FanBeamGeometry:
    return FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"],
        sod=cfg["sod"], sdd=cfg["sdd"],
    )


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


def make_optimizer(params, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver-iter] device={device}  config={json.dumps(cfg)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = build_geometry(cfg)
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = R_full.fbp(val_noisy)

    proj_dn, img_dn = build_denoisers(cfg, geom)
    pipe = DualDomainPipeline(
        geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn,
    ).to(device)
    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver-iter] params = {params_total/1e6:.3f} M", flush=True)

    opt = make_optimizer(pipe.parameters(), cfg)

    # ----- LR scheduler (iter-18) -----
    lr_schedule = str(cfg.get("lr_schedule", "constant")).lower()
    if lr_schedule == "cosine":
        # Cosine anneal across the WHOLE training (epochs * steps_per_epoch).
        steps_per_epoch = max(1, train_noisy.shape[0] // cfg["batch_size"])
        T_max = cfg["epochs"] * steps_per_epoch
        eta_min = float(cfg.get("lr_min", 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max, eta_min=eta_min)
        print(f"[solver-iter] LR schedule = cosine  T_max={T_max}  "
              f"lr_init={cfg['lr']}  lr_min={eta_min}",
              flush=True)
    else:
        scheduler = None

    # ----- EMA / SWA setup (iter-14/15/16; iter-26: vectorised foreach) -----
    use_ema = bool(cfg.get("ema", False))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    ema_start_ep = int(cfg.get("ema_start_ep", 0))  # >0 => start averaging late (SWA-style)
    ema_every = int(cfg.get("ema_every", 1))       # update shadow every N steps
    swa_mode = (ema_decay <= 0.0)                   # decay=0 => uniform mean (SWA)
    swa_count = 0                                   # running count of SWA samples
    if use_ema:
        # Pre-collect (param, shadow) pairs ONCE outside the training loop so
        # we don't pay the named_parameters() traversal cost on every step.
        # Iter-25 lesson: per-step dict lookup over 0.060M params via a Python
        # loop with mul_/add_ was 3-4x slower than baseline (timeout at ep2).
        # Solution: keep two flat lists of tensors and use torch._foreach_mul_
        # / torch._foreach_add_ which fuse the kernel-launch overhead.
        ema_params = [p for p in pipe.parameters() if p.requires_grad]
        if swa_mode:
            ema_shadows = [torch.zeros_like(p.detach()) for p in ema_params]
        else:
            ema_shadows = [p.detach().clone() for p in ema_params]
        mode = "SWA" if swa_mode else "EMA"
        n_shadow = sum(s.numel() for s in ema_shadows) / 1e6
        print(f"[solver-iter] {mode} enabled  decay={ema_decay}  start_ep={ema_start_ep}  "
              f"every={ema_every}  shadow_params={n_shadow:.3f}M  n_tensors={len(ema_shadows)}",
              flush=True)
        ema_state = True  # truthy sentinel for legacy branches below
    else:
        ema_state = None
        ema_params, ema_shadows = [], []

    # Iter-17: pluggable loss — keep MSE (default) or use Charbonnier.
    loss_type = str(cfg.get("loss_type", "mse")).lower()
    char_eps = float(cfg.get("char_eps", 1e-3))
    if loss_type == "charbonnier":
        def step_fn(batch):
            return charbonnier_training_step(pipe, batch, eps=char_eps)
        print(f"[solver-iter] training loss = Charbonnier (eps={char_eps})", flush=True)
    else:
        step_fn = pipe.training_step
        print(f"[solver-iter] training loss = MSE (default)", flush=True)

    t0 = time.time()
    for ep in range(cfg["epochs"]):
        pipe.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            batch = train_noisy[idx].to(device)
            losses = step_fn(batch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            running += float(losses["loss"].detach().cpu())
            # EMA / SWA update — vectorised via torch._foreach_* (iter-26).
            # In SWA mode (decay=0.0): shadow = ((n-1)*shadow + p) / n.
            # In EMA mode: shadow = decay*shadow + (1-decay)*p.
            # Skip until ep>=ema_start_ep and only every ema_every steps.
            if ema_state is not None and ep >= ema_start_ep:
                step_idx = ep * (train_noisy.shape[0] // cfg["batch_size"]) + (i // cfg["batch_size"])
                if step_idx % ema_every == 0:
                    with torch.no_grad():
                        # Snapshot detached param values into a flat list.
                        cur = [p.detach() for p in ema_params]
                        if swa_mode:
                            swa_count += 1
                            swa_w = 1.0 / swa_count
                            torch._foreach_mul_(ema_shadows, 1.0 - swa_w)
                            torch._foreach_add_(ema_shadows, cur, alpha=swa_w)
                        else:
                            torch._foreach_mul_(ema_shadows, ema_decay)
                            torch._foreach_add_(ema_shadows, cur,
                                                alpha=1.0 - ema_decay)
        mean_loss = running / max(1, train_noisy.shape[0])
        # Log the current alpha(s) so we see what the iterative DC step
        # learned to weight.
        try:
            alphas = torch.exp(img_dn.log_alpha).detach().cpu().tolist()
        except AttributeError:
            alphas = []
        cur_lr = opt.param_groups[0]['lr']
        print(f"[solver-iter] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  "
              f"lr={cur_lr:.2e}  alphas={['%.4f' % a for a in alphas]}",
              flush=True)
    train_time = time.time() - t0
    print(f"[solver-iter] training took {train_time:.1f}s", flush=True)

    # Swap in EMA / SWA weights for validation.  Refuse to swap if SWA never
    # accumulated a sample (start_ep too late) — fall back to final weights.
    swap_ok = ema_state is not None and (not swa_mode or swa_count > 0)
    if swap_ok:
        # Iter-26: vectorised swap via the same flat (param, shadow) lists.
        with torch.no_grad():
            for p, s in zip(ema_params, ema_shadows):
                p.data.copy_(s)
        mode = "SWA" if swa_mode else "EMA"
        extra = f" (n={swa_count})" if swa_mode else ""
        print(f"[solver-iter] swapped {mode}{extra} weights into model for validation",
              flush=True)

    pipe.eval()
    # Chunked validation — iter-4 OOM'd at val_n=100 with the iterated
    # denoiser + BF unfold (4.79 GiB allocation request). Splitting val
    # into chunks of `val_chunk` keeps peak memory bounded.
    val_chunk = cfg.get("val_chunk", 16)
    with torch.no_grad():
        chunks = []
        for j in range(0, val_noisy.shape[0], val_chunk):
            chunks.append(pipe.predict(val_noisy[j:j + val_chunk]))
        pred = torch.cat(chunks, dim=0)
    # FIXED-CALIBRATION metrics (mirrors main solver post iter-24): SSIM /
    # PSNR data_range = display_max - display_min so the score numerics
    # are stable & comparable across iterations.
    data_range = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ref, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ref, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ref) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(ld_fbp, val_ref, data_range=data_range).cpu())
    baseline_rmse = float(((ld_fbp - val_ref) ** 2).mean().sqrt().cpu())
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
        # Fixed display range matches main solver (clipped to canonical
        # phantom intensities) so comparisons are visually comparable.
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"sparse-view FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"unrolled-iter  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver-iter] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver-iter] figure failed: {e}", flush=True)

    # Capture learned alphas in result.json for the audit trail.
    try:
        learned_alphas = torch.exp(img_dn.log_alpha).detach().cpu().tolist()
    except AttributeError:
        learned_alphas = []
    result = {
        "val_score":      val_score,
        "val_psnr":       val_psnr,
        "val_ssim":       val_ssim,
        "val_rmse":       val_rmse,
        "baseline_psnr":  baseline_psnr,
        "baseline_rmse":  baseline_rmse,
        "headroom":       headroom,
        "params_M":       params_total / 1e6,
        "train_n":        cfg["train_n"],
        "val_n":          cfg["val_n"],
        "train_time_s":   train_time,
        "learned_alphas": learned_alphas,
        "config":         cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver-iter] result: val_score={val_score:.4f}  headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}  "
          f"alphas={learned_alphas}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="directory to write result.json + comparison.png")
    args = p.parse_args()
    main(Path(args.out_dir))
