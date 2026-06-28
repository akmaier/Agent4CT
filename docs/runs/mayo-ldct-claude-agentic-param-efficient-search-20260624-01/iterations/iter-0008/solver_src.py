"""Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-8).

A weight-TIED unrolled proximal-gradient reconstruction with explicit
data-consistency against the measured sinogram. The SAME regulariser
module and the SAME step-size scalar `alpha` are reused at every unrolled
step, so the trainable parameter budget is set by ONE small regulariser
(hundreds-to-low-thousands of params) regardless of `n_iter` — in sharp
contrast to the 233k-param ITNet champion whose denoiser is a full SmallUNet.

Architecture (per step k, all weights TIED across k — iter-4's exact recipe):

    x_0 = LD-FBP(sino)
    for k in range(K):
        dc = R^T( R x  -  sino ) / dc_norm          # data-consistency grad
        x  = clamp( x - alpha * ( dc + reg(x) ),     # proximal-gradient step
                    0.0, clip_max )

`dc_norm` is a power-iteration estimate of ‖R^T R‖ so `alpha` lives in O(1)
regardless of geometry (mirrors solver_hammernik_vn.py). `alpha` is a single
learnable softplus SCALAR (init from `alpha_init`) when `learnable_alpha` —
shared across all K steps. NO per-step alpha, NO momentum (both shown to
destabilise the recon in the 20-min budget; see iter-3 below).

iter-8 SIX-BOX (CODE-EVOLVING: SCALE THE WINNING FoE BANK at FEW params)
-----------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = 34.08 dB):
  - iter-1: cnn reg (2,798 params), K=5 tied prox+DC, learnable SCALAR
    alpha, 8 epochs, ~12 min wall -> hr 0.0871, ssim 0.846, psnr 34.87.
  - iter-2 (FAIL @ 8 ep): micro-UNet reg (25,890 params), K=5, 8 ep ->
    hr 0 (psnr 32.41). a 9x-wider POOLED reg below floor.
  - iter-3 (FAIL worse): tiny cnn + K=8 + per-step alpha + Nesterov
    MOMENTUM (2,806 params), 6 ep -> hr 0 (psnr 28.08, UNSTABLE).
    LESSON: deep-unroll (K>6) + momentum is UNSTABLE in-budget. AVOID.
  - iter-4: single-scale tiny cnn (2,798p, 3-layer 12ch, K=5, scalar alpha,
    NO momentum) + cosine LR + 16 epochs -> hr 0.2378, ssim 0.886,
    psnr 36.44 (~20 min wall, clear of the floor). BREAKTHROUGH: the solver
    was TRAINING-limited, not capacity-limited — proper training (16 ep
    cosine) TRIPLED hr at the SAME 2,798 params.
  - iter-5 (FAIL, the CLEAN CONTROL): micro-UNet reg (25,890p) under the
    EXACT iter-4 trainer -> hr 0 (psnr 32.41). VERDICT: POOLING in the reg
    is ARCHITECTURALLY BAD here. Do NOT pool.
  - iter-6 (FAIL, FLAT-DILATED A/B): cnn reg GROWN flat+dilated (37,601p)
    under the iter-4 trainer -> underperformed. SETTLED that SCALING the CNN
    reg fails BOTH ways (pooled AND flat-dilated); 2,798p is the CNN-family
    sweet spot for this tied-prox-DC + 20-min scheme.
  - iter-7 (PREV BEST, BASE): switched reg FAMILY cnn -> "foe" (TIED
    Fields-of-Experts / VN learned filter bank, reg(x)=K^T ρ'(Kx)). Sized
    nf=24 / k=7 / nb=31 -> 1,920 reg params, total 1,921 (1 scalar alpha),
    zero-init ρ'-weights, K=5 tied prox+DC, 16-ep cosine ->
    hr 0.2515, ssim 0.906, psnr 36.59 (~22 min wall, 1331 s).
    KEY RESULT: the FoE BEAT the tiny CNN (0.2515 vs 0.2378) at FEWER params
    (1,921 vs 2,798). The learned-filter-bank prior (linear analysis filters
    K + per-filter RBF activation + tied synthesis K^T) is a DIFFERENT, more
    apt structure than a black-box CNN for this tied-prox-DC recon, and it WON.

FAILURE MODE / OPPORTUNITY addressed (iter-8): the black-box CNN reg DID NOT
scale (iter-2/5/6 all regressed past 2,798p). But the FoE that beat it is the
NATURAL axis to push — its structure (edge-aware analysis bank + flexible
per-filter nonlinearity) matches the recon's needs, so a RICHER filter prior
may keep climbing where the CNN couldn't. iter-7 is the only FoE point we have;
we have NOT yet probed whether the FoE bank's CAPACITY (vs the CNN's) scales.

CHANGE (iter-8, CFG only — the FoEReg code already parametrises the bank):
GROW the FoE bank's ANALYSIS WIDTH while keeping its per-step WALL flat so the
proven 16-ep cosine schedule still fits the 18-min train backstop.
  foe_n_filters 24 -> 40   (wider learned edge-aware analysis bank — MORE
                            distinct filters K, the FoE's discriminative axis;
                            iter-2/5/6 showed channel/depth growth fails for a
                            CNN, but the FoE's per-filter structure is the part
                            iter-7 proved apt, so widen THAT)
  foe_n_bumps   31 -> 23   (TRIM the per-filter RBF activation resolution to
                            OFFSET the wider conv's cost — see BUDGET below)
  foe_kernel     7  (UNCHANGED — keep iter-7's eff RF 7px/step, ~19px over K=5)
reg = 40*7*7 + 40*23 = 1960 + 920 = 2880 params, total 2,881 incl. the 1
scalar alpha. Still « 233k ITNet (and below iter-4's 2,798 in spirit: the FoE
remains the more param-efficient FAMILY). ONE change in INTENT: scale the FoE
bank capacity along its proven-apt axis (filter diversity), budget-neutral.
EVERYTHING ELSE from iter-7 byte-for-byte: reg_type="foe", K=5, single tied
scalar alpha, NO momentum, NO per-step alpha, plain prox step, cosine LR
5e-3 -> 1e-5, 16 epochs, max_train_s=1080 18-min backstop, train_n=200,
grad_clip=1.0, batch_size=1, val_n=214, zero-init ρ'-weights (rbf_init_std=0).
STABILITY (UNCHANGED from iter-7): zero-init ρ'-weights => reg(x) ≡ 0 at init,
so the seed IS the clean GD+DC iter-4 scheme and learns the wider bank as a
correction. Recon DYNAMICS identical to iter-7 (same K, same scalar alpha,
same plain prox step, same cosine LR that only DECREASES the step) — nothing
in iter-3's UNSTABLE failure mode (K>6, per-step alpha, momentum) is touched.
HYPOTHESIS: because the FoE structure (not the CNN) is the apt prior for this
recon (iter-7 proved it BEAT the CNN), a WIDER analysis bank (40 vs 24
filters) gives the learned-filter prior more edge/texture selectivity and
climbs PAST iter-7's 0.2515 — i.e. the FoE SCALES where the CNN did not.
If it MATCHES iter-7 (~0.25), the FoE has hit its own capacity sweet spot at
nf~24 for this scheme (still a clean frontier result: the FoE family ceiling
is mapped). If it REGRESSES, the FoE shares the CNN's "tiny is best" property
on dense full-view Mayo. The budget-neutral bump-trim controls for "did it
regress because of FEWER epochs?" — per-step wall is held ~constant.
BUDGET (must train in ~18 min; iter-7 ran 1331 s, ~1080 s train + val, so it
was wall-bounded and the marginal cost is the per-step Python RBF bump loop):
the RBF loop runs foe_n_bumps iters PER step PER sample, so 31->23 CUTS it
~26%; the analysis/synthesis conv widens 24->40 (~1.67x its FLOPs), but the
conv is GPU-parallel and the per-step cost is DOMINATED by the DC term (one
forward + one back project, UNCHANGED). Net per-step wall ≈ iter-7's, so the
16-ep cosine schedule fits the SAME max_train_s=1080 backstop. PREDICTED hr
~0.24-0.27 (a modest BEAT of iter-7's 0.2515 if the FoE scales; a clean climb
past the 0.2515 BEST is plausible but the iter-6 CNN ceiling warns capacity
gains may saturate, so a MATCH at ~0.25 is the conservative read).
EXPECT THE FoE TO SCALE PAST 0.2515? Tentatively YES, modestly — the FoE
already BEAT the CNN at FEWER params (a different, apt structure), and a wider
analysis bank pushes its proven-apt axis. But the gain is likely SMALL
(0.25 -> ~0.26) and a flat MATCH is a real outcome; this iter MAPS whether the
FoE bank has its own capacity sweet spot, the question iter-7 left open.

The learned regulariser `reg(x)` is selected by `reg_type`:
  - "foe"       (DEFAULT — iter-7/8: TIED Fields-of-Experts / VN filter bank,
                a DIFFERENT, single-scale, param-EFFICIENT reg family that
                BEAT the CNN at fewer params in iter-7 — hr 0.2515 vs 0.2378):
                analysis conv2d (`foe_n_filters` filters, `foe_kernel`x
                `foe_kernel`) -> per-filter RBF activation (`foe_n_bumps`
                bumps) -> TIED conv_transpose2d synthesis. reg(x)=K^T ρ'(Kx),
                one VNStep's reg-gradient reused at every unrolled step.
                ZERO-INIT ρ'-weights => reg ≈ 0 at init (stability). iter-7
                nf=24/k=7/nb=31 = 1,920p (BEST so far). iter-8 SCALES the bank
                nf=24->40 / nb=31->23 (budget-neutral) = 2,880p (total 2,881)
                to probe whether the FoE capacity scales where the CNN didn't.
  - "cnn"       (iter-1/iter-4 BEST, iter-6 FLAT-DILATED — kept selectable):
                `cnn_layers` 3x3 convs at `cnn_channels` channels with a
                per-layer `cnn_dilations` ladder (reflect-padded), GroupNorm+
                ReLU between, zero-init 1x1 head so reg ≈ 0 at init.
                Single-scale, NO pooling. iter-4 BEST: c=12/3-layers/dil=1
                (2,797p, eff RF ~7px, hr 0.2378). iter-6 GROWN: c=32/5-layers/
                dil[1,2,4,2,1] (37,601p) FAILED to beat iter-4 -> 2,798p is
                the CNN-family sweet spot.
  - "microunet" (iter-2/iter-5, REGRESSED — kept selectable): a 2-level
                (one-downsample) micro-UNet denoiser, ~25.9k params at c=16.
                ARCHITECTURALLY BAD here: pooling caps psnr at ~32.4 (hr 0)
                at BOTH 8 ep (iter-2) and 16 ep (iter-5). Do NOT use.
  - "bilateral" a cascade of `n_bf` TrainableBilateralFilter2d (3 params each).

Trained end-to-end supervised against the HD truth image
(`supervised_recon_loss`, Adam + cosine LR). The DC term + a modest learnable
`alpha` keep the recon data-consistent so it beats the LD-FBP RMSE floor (the
headroom gate is RMSE-vs-LD-FBP, not SSIM).

Citation context: this is the parameter-tied limit of the unrolled
proximal-gradient family (Hammernik 2018 MRM variational network; Adler &
Öktem 2018 learned primal-dual). See literature/ for the lineage.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: not always auto-loaded by `import torch`

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.metrics import (psnr, ssim, evaluate_calibrated,
                                make_4panel_comparison, supervised_recon_loss,
                                negativity_penalty, clip_and_step)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # ---- architecture (iter-7: iter-4's stable trainer + a DIFFERENT reg FAMILY: FoE/VN bank) ----
    "reg_type":        "foe",      # iter-8: KEEP "foe" (TIED Fields-of-Experts / VN bank); iter-7 BEAT the CNN at fewer params (hr 0.2515 vs 0.2378). "foe" (iter-7 BEST / iter-8 SCALED) | "cnn" (iter-4 / iter-6) | "microunet" (iter-2/iter-5, REGRESSED, pooling caps psnr 32.4) | "bilateral"
    "n_iter":          5,          # unrolled prox-gradient steps, weight-TIED (iter-1/iter-4 stable K; K>6 + momentum was UNSTABLE in iter-3)
    "learnable_alpha": True,       # ONE tied alpha = softplus(param); init from alpha_init
    "per_step_alpha":  False,      # iter-4: REVERT iter-3's per-step alpha (it destabilised) -> single shared scalar (iter-1)
    "momentum":        False,      # iter-4: REVERT iter-3's Nesterov momentum (it destabilised) -> plain prox step (iter-1)
    "beta_init":       0.5,        # unused when momentum=False (kept for backward-compat selectability)
    "alpha_init":      0.1,        # step size (O(1) thanks to dc_norm scaling)
    "clip_max":        0.05,       # per-step clamp upper bound (= display_max μ)
    "dc_norm":         True,       # divide R^T(R x - g) by power-iter ‖R^T R‖
    "checkpoint":      True,       # gradient-checkpoint each unrolled step
    # ---- "microunet" regulariser (iter-2/iter-5, REGRESSED — pooling caps psnr 32.4, do NOT use; kept selectable) ----
    "mu_channels":     16,    # base width; 2-level micro-UNet = 25,889 reg params at c=16 (exact iter-2/iter-5 arch)
    # ---- "cnn" regulariser (iter-4 BEST = c12/3-layers/dil1 = 2,797p; iter-6 GROWN = c32/5-layers/dilladder = 37,601p, FAILED) ----
    "cnn_channels":    12,    # iter-7: REVERT to iter-4 BEST (12) — the CNN-family sweet spot (iter-6's 32 failed). 2,797 reg params at 3 layers.
    "cnn_layers":      3,     # iter-7: REVERT to iter-4 BEST (3) — kept selectable, not used at reg_type=foe.
    "cnn_dilations":   None,  # iter-7: REVERT to iter-4 BEST (None = all dilation 1, eff RF ~7px). iter-6's [1,2,4,2,1] failed.
    # ---- "foe" regulariser (iter-8 DEFAULT — SCALED TIED FoE/VN filter bank, the winner; iter-7 BEAT the CNN at FEWER params) ----
    "foe_n_filters":   40,    # iter-8: 24 -> 40 analysis filters (WIDER learned edge-aware bank — the FoE's discriminative axis; the CNN didn't scale but the FoE's filter structure is the apt part)
    "foe_kernel":      7,     # iter-7/8: UNCHANGED at 7 (eff RF 7px/step, ~19px over K=5 tied steps, no pooling)
    "foe_n_bumps":     23,    # iter-8: 31 -> 23 RBF bumps (TRIM the per-step Python RBF loop to OFFSET the wider conv -> per-step wall stays flat so 16-ep cosine still fits the 18-min backstop)
    "foe_x_range":     1.0,
    "foe_filter_init_std": 0.05,
    "foe_rbf_init_std":    0.0,   # iter-7: ZERO-INIT synthesis -> ρ'≡0 => reg(x)≡0 at init (stability; mirrors CNNReg zero-init head)
    # ---- "bilateral" regulariser ----
    "n_bf":            4,
    "bf_kernel":       7,
    # ---- training (iter-4: spend the unused budget on TRAINING) ----
    "train_n":   200,
    "val_n":     214,
    "epochs":    16,          # iter-4: 8 -> 16 (iter-1 was undertrained: 8 ep in ~12 min; the 20-min budget has slack)
    "cosine_lr": True,        # iter-4: cosine-anneal lr 5e-3 -> cosine_lr_min over the run (gentler late-training updates than iter-1's constant lr)
    "cosine_lr_min": 1e-5,    # eta_min for CosineAnnealingLR
    "max_train_s": 1080,      # iter-4: hard 18-min train backstop so 16 ep never overruns the 20-min budget / 30-min slurm wall
    "lr":        5e-3,
    "batch_size": 1,               # per-sample-ps geometry (Mayo): keep at 1
    "lambda_neg": 1.0,
    "grad_clip": 1.0,
    "seed":      42,
}


# ---------------------------------------------------------------------------
# Learned regularisers. Each module maps (B,1,H,W) -> (B,1,H,W); the value is
# the regulariser GRADIENT contribution `reg(x)` added inside the prox step.
# All are weight-tied (one instance reused across every unrolled iteration).
# ---------------------------------------------------------------------------
class CNNReg(nn.Module):
    """FLAT (single-scale, NO pooling) residual CNN regulariser (iter-6).

    `layers` 3x3 convs at `channels` channels with GroupNorm+ReLU between
    them and a zero-initialised 1x1 head, so reg(x) ≈ 0 at init (the seed
    therefore starts as a clean gradient-descent-with-DC scheme and learns a
    correction).

    iter-6 grows the iter-4 module WITHOUT any downsampling: it accepts a
    per-layer DILATION ladder (`dilations`) so the effective receptive field
    expands at FULL resolution. Each 3x3 conv is reflect-padded by its own
    dilation so spatial size is preserved exactly. With c=32, layers=5,
    dilations=[1,2,4,2,1] the eff RF is ~21px (vs ~7px at the iter-4
    c=12/3-layers/dil=1 default) — the long-range context that the pooled
    micro-UNet bought (iter-2/iter-5), but with ZERO pooling, which iter-5
    proved to be the architectural failure mode here.

    `dilations` may be None (all dilation 1, the iter-4 behaviour), a single
    int (applied to every conv), or a list/tuple — recycled/truncated to
    `layers` entries so the cfg ladder stays robust to off-by-one."""

    def __init__(self, channels: int = 16, layers: int = 3,
                 dilations=None):
        super().__init__()
        channels = int(channels)
        layers = max(1, int(layers))
        # Normalise dilations -> a length-`layers` list of positive ints.
        if dilations is None:
            dil = [1] * layers
        elif isinstance(dilations, (int, float)):
            dil = [max(1, int(dilations))] * layers
        else:
            seq = [max(1, int(d)) for d in dilations] or [1]
            dil = [seq[i % len(seq)] for i in range(layers)]

        def _conv(ci, co, d):
            # reflect-pad by `d` so a dilated 3x3 conv keeps H,W exactly.
            return nn.Conv2d(ci, co, 3, padding=d, dilation=d,
                             padding_mode="reflect")

        body: list[nn.Module] = [_conv(1, channels, dil[0])]
        for li in range(1, layers):
            body += [nn.GroupNorm(_pick_groups(channels), channels),
                     nn.ReLU(inplace=True),
                     _conv(channels, channels, dil[li])]
        self.body = nn.Sequential(*body)
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(self.body(x)))


def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class MicroUNetReg(nn.Module):
    """Multi-scale 2-level (one-downsample) micro-UNet regulariser (iter-2).

    Why: the iter-1 single-scale CNNReg had an effective receptive field of
    only ~7px and saturated at hr=0.087 — too small to model Mayo's
    spatially-correlated streak/low-dose noise. This adds ONE coarse branch
    (avg-pool/2 -> double-conv -> bilinear-up, concat skip) so the tied prox
    step sees both fine (3x3 @ full res) and coarse (effective ~14px @
    half-res) structure, at ~16k params (c=16) — one order under the 233k
    SmallUNet champion. Still ONE instance reused at every unrolled step
    (weight-tied). Zero-init 1x1 head ⇒ reg(x) ≈ 0 at init, so the seed
    starts as clean GD+DC and learns a correction (mirrors CNNReg/SmallUNet).
    REGRESSED in iter-2 (below LD-FBP floor): undertrainable in 20-min budget.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        c = int(channels)
        g = _pick_groups(c)
        g2 = _pick_groups(2 * c)

        def dconv(ci, co, gn):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1),
                nn.GroupNorm(gn, co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1),
                nn.GroupNorm(gn, co), nn.ReLU(inplace=True),
            )

        self.enc = dconv(1, c, g)               # full-res encoder
        self.down = dconv(c, 2 * c, g2)         # half-res branch (coarse)
        self.dec = dconv(c + 2 * c, c, g)       # fuse coarse(up) + fine skip
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # Pad to even dims so the single pool/upsample round-trips exactly.
        ph = h % 2
        pw = w % 2
        x_in = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        e = self.enc(x_in)                                  # (B,c,H,W)
        d = self.down(F.avg_pool2d(e, 2))                  # (B,2c,H/2,W/2)
        u = F.interpolate(d, size=e.shape[-2:], mode="bilinear",
                          align_corners=False)             # (B,2c,H,W)
        y = self.dec(torch.cat([u, e], dim=1))             # (B,c,H,W)
        out = self.head(y)
        if ph or pw:
            out = out[..., :h, :w]
        return out


class FoEReg(nn.Module):
    """Single tied Fields-of-Experts / VN filter bank (iter-7 DEFAULT).

    reg(x) = K^T ρ'(K x), with K an analysis conv2d bank (n_filters,
    kernel x kernel), ρ' a per-filter RBF mixture (n_bumps bumps), and K^T
    the tied conv_transpose2d synthesis — exactly the regulariser-gradient
    of one solver_hammernik_vn.py VNStep, but ONE bank reused at every
    unrolled step (weight-tied) instead of T untied banks.

    ZERO-INIT SYNTHESIS (iter-7 stability fix): with `rbf_init_std=0.0` the
    per-filter RBF mixture weights start at 0, so ρ'(·) ≡ 0 and therefore
    reg(x) ≡ 0 at init. The seed thus starts as the EXACT clean GD+DC scheme
    (the reg contributes nothing) and LEARNS the filter bank as a correction
    — the same proven zero-init-output pattern as CNNReg/MicroUNet's zero-init
    1x1 head, ported to the FoE family so this very-different reg can be
    dropped into the known-stable iter-4 trainer without destabilising. The
    analysis filters K remain randomly initialised (filter_init_std) and fully
    learnable; only the OUTPUT magnitude is gated to 0 by the zero ρ'-weights
    at step 0."""

    def __init__(self, n_filters: int = 8, kernel_size: int = 5,
                 n_bumps: int = 15, x_range: float = 1.0,
                 filter_init_std: float = 0.05, rbf_init_std: float = 0.01):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.n_bumps = int(n_bumps)
        self.weight = nn.Parameter(
            torch.randn(self.n_filters, 1, self.kernel_size, self.kernel_size)
            * filter_init_std)
        centres = torch.linspace(-x_range, x_range, self.n_bumps)
        sigma = 2.0 * x_range / max(1, self.n_bumps - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("inv_sigma_sq", torch.tensor(1.0 / (sigma ** 2)))
        self.rbf_weights = nn.Parameter(
            torch.randn(self.n_filters, self.n_bumps) * rbf_init_std)

    def _rho_prime(self, Kx: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(Kx)
        for j in range(self.n_bumps):
            mu_j = self.centres[j]
            bump = torch.exp(-0.5 * (Kx - mu_j) ** 2 * self.inv_sigma_sq)
            w_j = self.rbf_weights[:, j].view(1, self.n_filters, 1, 1)
            out = out + bump * w_j
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        Kx = F.conv2d(x, self.weight, padding=pad)
        rho_Kx = self._rho_prime(Kx)
        return F.conv_transpose2d(rho_Kx, self.weight, padding=pad)


class BilateralReg(nn.Module):
    """Cascade of `n_bf` TrainableBilateralFilter2d (3 params each). The
    regulariser gradient is the residual `x - cascade(x)` (a denoise-then-
    subtract proximal correction), keeping it ≈0 for an already-smooth x."""

    def __init__(self, n_bf: int = 4, kernel_size: int = 7):
        super().__init__()
        self.filters = nn.ModuleList(
            [TrainableBilateralFilter2d(kernel_size=int(kernel_size))
             for _ in range(max(1, int(n_bf)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for f in self.filters:
            y = f(y)
        return x - y


def build_reg(cfg: dict) -> nn.Module:
    rt = cfg["reg_type"]
    if rt == "microunet":
        return MicroUNetReg(channels=cfg.get("mu_channels", 16))
    if rt == "cnn":
        return CNNReg(channels=cfg["cnn_channels"], layers=cfg["cnn_layers"],
                      dilations=cfg.get("cnn_dilations"))
    if rt == "foe":
        return FoEReg(n_filters=cfg["foe_n_filters"], kernel_size=cfg["foe_kernel"],
                      n_bumps=cfg["foe_n_bumps"], x_range=cfg["foe_x_range"],
                      filter_init_std=cfg["foe_filter_init_std"],
                      rbf_init_std=cfg["foe_rbf_init_std"])
    if rt == "bilateral":
        return BilateralReg(n_bf=cfg["n_bf"], kernel_size=cfg["bf_kernel"])
    raise ValueError(f"unknown reg_type={rt!r} (expected microunet|cnn|foe|bilateral)")


# ---------------------------------------------------------------------------
class ParamEfficientUnrolled(nn.Module):
    """Weight-tied unrolled proximal-gradient with data consistency (iter-4:
    iter-1's EXACT stable recipe).

    x_0 = u0 (LD-FBP). For k in range(n_iter):
        dc = R^T(R x - g) / dc_norm
        x  = clamp(x - alpha * (dc + reg(x)), 0.0, clip_max)
    `reg` AND `alpha` are SHARED across all steps (weight-tied). NO per-step
    alpha, NO momentum — iter-3 proved both destabilise the recon in-budget.

    For backward-compatibility the per_step_alpha / momentum config knobs are
    still honoured (so the iter-3 variant remains selectable), but the iter-4
    DEFAULTS turn BOTH off, collapsing the iteration to the plain tied
    prox-gradient that iter-1 ran (and the ONLY config that cleared the floor).
    With per_step_alpha=False and momentum=False:
        y == x at every step (no look-ahead), v unused, beta == 0.
    """

    def __init__(self, projector: PyronnFanBeamProjector, cfg: dict,
                 dc_norm: float = 1.0):
        super().__init__()
        self.projector = projector             # shared single instance, not a sub-module
        self.n_iter = int(cfg["n_iter"])
        self.clip_max = float(cfg["clip_max"])
        self.checkpoint = bool(cfg.get("checkpoint", True))
        self.per_step_alpha = bool(cfg.get("per_step_alpha", False))
        self.use_momentum = bool(cfg.get("momentum", False))
        self.register_buffer("dc_norm", torch.tensor(float(dc_norm)))
        self.reg = build_reg(cfg)              # ONE tied regulariser
        # Step size(s): one tied scalar (iter-4 default), OR a length-n_iter
        # vector (per_step, iter-3 variant — off by default).
        if cfg["learnable_alpha"]:
            inv_softplus = math.log(math.expm1(max(float(cfg["alpha_init"]), 1e-6)))
            n = self.n_iter if self.per_step_alpha else 1
            self.log_alpha = nn.Parameter(torch.full((n,), float(inv_softplus)))
            self._alpha_const = None
        else:
            self.log_alpha = None
            self._alpha_const = float(cfg["alpha_init"])
        # Single tied momentum coefficient beta in (0,1) via sigmoid (off by
        # default in iter-4; iter-3 variant only).
        if self.use_momentum:
            b0 = min(max(float(cfg.get("beta_init", 0.5)), 1e-4), 1 - 1e-4)
            inv_sig = math.log(b0 / (1.0 - b0))
            self.beta_raw = nn.Parameter(torch.tensor(float(inv_sig)))
        else:
            self.beta_raw = None

    @property
    def alpha(self) -> torch.Tensor:
        """Back-compat scalar view (mean over steps) for logging."""
        if self.log_alpha is not None:
            return F.softplus(self.log_alpha).mean()
        return torch.as_tensor(self._alpha_const, device=self.dc_norm.device,
                               dtype=self.dc_norm.dtype)

    def _alpha_k(self, k: int) -> torch.Tensor:
        if self.log_alpha is not None:
            idx = k if self.per_step_alpha else 0
            return F.softplus(self.log_alpha[idx])
        return torch.as_tensor(self._alpha_const, device=self.dc_norm.device,
                               dtype=self.dc_norm.dtype)

    @property
    def beta(self) -> torch.Tensor:
        if self.beta_raw is not None:
            return torch.sigmoid(self.beta_raw)
        return torch.zeros((), device=self.dc_norm.device, dtype=self.dc_norm.dtype)

    def _grad(self, y: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        """Prox-gradient direction at the (look-ahead) point y."""
        R_y = self.projector.forward_project(y)
        dc = self.projector.back_project(R_y - sino) / self.dc_norm
        return dc + self.reg(y)

    def _step(self, x: torch.Tensor, y: torch.Tensor, sino: torch.Tensor,
              alpha_k: torch.Tensor, beta: torch.Tensor):
        """One (optionally accelerated) prox-gradient step. Returns (x_new,
        y_new). With beta==0 (iter-4 default) y_new == x_new, i.e. the plain
        tied prox step of iter-1. Checkpointed: all tensor args/returns so
        grads flow through any carried velocity/look-ahead state."""
        x_new = torch.clamp(y - alpha_k * self._grad(y, sino), 0.0, self.clip_max)
        v = x_new - x
        y_new = x_new + beta * v
        return x_new, y_new

    def forward(self, u0: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        x = u0
        y = u0                                   # look-ahead == x at k=0 (v=0)
        beta = self.beta                         # == 0 when momentum off
        for k in range(self.n_iter):
            alpha_k = self._alpha_k(k)
            if self.checkpoint and y.requires_grad:
                x, y = torch.utils.checkpoint.checkpoint(
                    self._step, x, y, sino, alpha_k, beta, use_reentrant=False)
            else:
                x, y = self._step(x, y, sino, alpha_k, beta)
        return x


# ---------------------------------------------------------------------------
def build_dataset(geom, n, seed, i0, sigma_e, device):
    # Dispatches on AGENT4CT_DATASET / cfg["dataset_kind"]. Phantom path
    # is backwards-compatible; staged paths load from disk. Split is picked
    # from the existing seed convention (train: seed=cfg["seed"]; val:
    # seed=cfg["seed"]+1000).
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)


def _count_reg_params(cfg: dict) -> int:
    """Trainable param count of ONE regulariser at the given cfg (for the
    start-of-run print + sanity check; the tied model reuses this once)."""
    reg = build_reg(cfg)
    return sum(p.numel() for p in reg.parameters() if p.requires_grad)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("PARAM_EFFICIENT_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        env_cfg = json.loads(Path(env_path).read_text())
        cfg = {**CONFIG, **env_cfg, **(cfg or {})}
        print(f"[solver] Loaded config from {env_path}", flush=True)
    else:
        cfg = {**CONFIG, **(cfg or {})}

    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}", flush=True)
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k in ('reg_type','n_iter','learnable_alpha','per_step_alpha','momentum','beta_init','alpha_init','clip_max','mu_channels','cnn_channels','cnn_layers','cnn_dilations','foe_n_filters','foe_kernel','foe_n_bumps','n_bf','bf_kernel','epochs','cosine_lr','cosine_lr_min','max_train_s','batch_size','lr','train_n','val_n')}, default=str)}",
          flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    train_ph, train_clean, train_noisy, train_ps = build_dataset(
        geom, cfg["train_n"], cfg["seed"],
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # PER-SAMPLE geometry (Mayo canonical): swap model.projector per slice +
    # build the per-ps FBP init/baseline (falls back to single proj non-mayo).
    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        if per_ps:
            train_u0 = mayo_per_sample_fbp(_projs, _trk, train_noisy, cfg["image_size"])
            val_u0   = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            train_u0 = torch.clamp(proj.fbp(train_noisy), min=0.0)
            val_u0   = torch.clamp(proj.fbp(val_noisy),   min=0.0)

    # Power-iteration estimate of ‖R^T R‖ so alpha stays in O(1).
    norm_val = 1.0
    if cfg.get("dc_norm", True):
        with torch.no_grad():
            v = torch.randn(1, 1, cfg["image_size"], cfg["image_size"], device=device)
            v = v / v.norm()
            for _ in range(8):
                Av = proj.forward_project(v)
                v = proj.back_project(Av)
                n = v.norm().clamp(min=1e-12)
                v = v / n
            norm_val = float(n.item())
            print(f"[solver] dc_norm power-iter ≈ {norm_val:.3g}", flush=True)

    model = ParamEfficientUnrolled(proj, cfg, dc_norm=norm_val).to(device)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_params = _count_reg_params(cfg)
    n_alpha = (model.n_iter if model.per_step_alpha else 1) if model.log_alpha is not None else 0
    n_beta = 1 if model.beta_raw is not None else 0
    print(f"[solver] ParamEfficient iter-8: reg_type={cfg['reg_type']!r}  "
          f"n_iter={cfg['n_iter']} (weight-TIED reg)  "
          f"per_step_alpha={model.per_step_alpha} momentum={model.use_momentum}  "
          f"epochs={cfg['epochs']} cosine_lr={cfg.get('cosine_lr', False)}  "
          f"trainable params={params_total} "
          f"(reg={reg_params} + alpha={n_alpha} + beta={n_beta})  "
          f"= {params_total/1e6:.6f} M  vs 233k ITNet", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # iter-4: cosine LR anneal over the planned epochs. clip_and_step() calls
    # opt.step() internally; we step the scheduler ONCE per epoch afterwards,
    # so the two compose cleanly. T_max = planned epochs (the wall backstop may
    # cut the run short, in which case lr simply never reaches eta_min — fine).
    sched = None
    if cfg.get("cosine_lr", False):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(cfg["epochs"])),
            eta_min=float(cfg.get("cosine_lr_min", 1e-5)))
    train_start = time.time()
    bs = max(1, int(cfg["batch_size"]))
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0
        for i in range(0, cfg["train_n"], bs):
            idx = perm[i:i + bs]
            if per_ps:
                model.projector = _projs[float(_trk[int(idx[0])])]
            u0 = train_u0[idx]
            sino = train_noisy[idx]
            truth = train_ph[idx]
            pred = model(u0, sino)
            loss = supervised_recon_loss(pred, truth,
                                         lambda_neg=cfg["lambda_neg"], base="mse")
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
            running += float(loss.detach().cpu())
            n_batches += 1
        cur_lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step()
        avg_loss = running / max(1, n_batches)
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"lr={cur_lr:.3g}  alpha={float(model.alpha.detach().cpu()):.4g}",
              flush=True)
        if time.time() - train_start > cfg.get("max_train_s", 1800):
            print(f"[train] wall ({cfg.get('max_train_s', 1800)}s) reached at epoch {ep+1}",
                  flush=True)
            break
    train_time = time.time() - train_start

    model.eval()
    preds = []
    with torch.no_grad():
        chunk = 1 if per_ps else max(1, bs)
        for i in range(0, val_u0.shape[0], chunk):
            if per_ps:
                model.projector = _projs[float(_vrk[i])]
            preds.append(model(val_u0[i:i + chunk], val_noisy[i:i + chunk]))
    pred = torch.cat(preds, dim=0)

    # baseline = the LD-FBP starting point (the headroom anchor).
    with torch.no_grad():
        if per_ps:
            val_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    val_fbp = val_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=val_fbp,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    val_score = val_ssim

    result = {
        "val_score": val_score, "headroom": headroom,
        "val_ssim": val_ssim, "val_psnr": val_psnr, "val_rmse": val_rmse,
        "val_ssim_std": metrics["val_ssim_std"],
        "val_psnr_std": metrics["val_psnr_std"],
        "val_rmse_std": metrics["val_rmse_std"],
        "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6,
        "reg_type": cfg["reg_type"],
        "reg_params": reg_params,
        "n_iter": cfg["n_iter"],
        "epochs": cfg["epochs"],
        "cosine_lr": bool(cfg.get("cosine_lr", False)),
        "per_step_alpha": bool(model.per_step_alpha),
        "momentum": bool(model.use_momentum),
        "alpha_learned": float(model.alpha.detach().cpu()),  # mean over steps
        "alpha_per_step": ([float(model._alpha_k(k).detach().cpu())
                            for k in range(model.n_iter)]
                           if model.log_alpha is not None else None),
        "beta_learned": (float(model.beta.detach().cpu())
                         if model.beta_raw is not None else None),
        "train_n": cfg["train_n"], "val_n": cfg["val_n"],
        "train_time_s": train_time, "config": cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    _beta_s = f"{result['beta_learned']:.3g}" if result['beta_learned'] is not None else "off"
    print(f"[solver] ParamEfficient: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"params={params_total}  alpha_mean={result['alpha_learned']:.4g}  "
          f"beta={_beta_s}  time={train_time:.1f}s  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label=f"ParamEff[{cfg['reg_type']}]",
            headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
