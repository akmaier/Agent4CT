"""Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-26).

iter-26 DUAL-DOMAIN (idea G): the FIRST iter to touch the PROJECTION domain. Up
to iter-25 every lever lived purely in the image-domain image-unroll. iter-26
composes a NOISE-ADAPTIVE PROJECTION-DOMAIN filter (Manduca et al., Med. Phys.
2009 -- the photon/Anscombe variance-stabilising bilateral) IN FRONT OF the
iter-7 FoE image-domain unroll. The proj filter denoises the LD log-sinogram
BEFORE it enters the recon; that denoised sino feeds BOTH the LD-FBP init AND the
DC data term g. The iter-7 image-domain FoE reg is UNCHANGED.

LESSON carried from iters 21-25 (4/5 extension failures): only changes that START
byte-for-byte iter-7 (the added stage contributes EXACTLY ZERO at init) stay in
iter-7's stable basin. iter-26 honours this LITERALLY in the projection domain:
the proj filter is a learnable RESIDUAL with a ZERO-INIT scalar gain --
    sino_filtered = sino + proj_gain * ( Manduca(sino) - sino )
so proj_gain=0 => sino_filtered == sino BYTE-FOR-BYTE => the SAME LD-FBP init and
the SAME DC g as iter-7 => the WHOLE dual-domain pipeline IS iter-7 at init.

iter-26 SIX-BOX (NUMBERS) -- dual-domain = zero-init Manduca proj filter ∘ iter-7 FoE
------------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = psnr 34.08 dB):
  - iter-7 (CHAMPION, BASE): FoE nf24/k7/nb31 = 24*49 + 24*31 = 1,176 + 744 =
    1,920 reg + 1 scalar alpha = 1,921 params. K=5 tied prox+DC, single-phase
    PARTIAL cosine peak 5e-3 (budget-cut @~ep8, loss STILL falling MONOTONICALLY
    => UNDER-TRAINED) -> val hr 0.2515, ssim 0.9058, psnr 36.59, val_rmse 7.40e-4.
    test hr 0.1852. THE per-param champion on this board.
  - iter-8..16: capacity/depth/stage/LR/kernel-geometry knobs all mapped at the
    fixed model + single-phase trainer -> 0.2515 is the ceiling; nf24/k7 is the
    FoE geometry optimum (iter-16 k9/nf17 iso-param REGRESSED to 0.2404).
  - iter-17 (bilateral ALONE image-domain, ~17p): hr ~0 BELOW the FBP floor.
  - iter-18/19: HALVING the bank (nf 12/6) walks the frontier DOWN (capacity helps).
  - iter-21/22 (denoiser-pretrain): FROZE x2, hr 0 -> the reg only learns COUPLED.
  - iter-23 (ordered-subsets DC): NULL 0.2401 -> view-count not the training wall.
  - iter-24 (steerable/equivariant FoE): DIVERGED (non-identity analysis reparam).
  - iter-25 (image-domain bilateral ON TOP of FoE, +4p, zero-init gain): the
    SAFE-superset image-domain control. (Result pending / ~iter-7; the gain
    magnitude IS the answer to "does an image edge term help on top of FoE".)
  - STANDALONE proj-domain on dense Mayo (the HONEST CAVEAT, see HYPOTHESIS):
    Manduca proj-bilateral ALONE reached test hr ~0.069 and PWLS ray-weighting
    logged NEGATIVE -- at 2,304 views the FBP ALREADY averages the projection
    noise, so there is LITTLE projection-domain headroom on dense full-view Mayo.
FAILURE MODE addressed (iter-26): the campaign has explored the IMAGE domain
  exhaustively (FoE geometry, capacity, depth, LR, an image-bilateral-on-top) and
  hit a 0.2515 ceiling. It has NEVER composed an UPSTREAM projection-domain
  denoiser with the FoE unroll -- a genuinely ORTHOGONAL lever (the proj filter
  cleans the sino BEFORE the recon; the FoE reg cleans the image AFTER). Prior
  proj-domain solvers were tested STANDALONE (proj-BF -> plain FBP); none was
  COMPOSED with a learned image unroll end-to-end.
CHANGE (iter-26, ONE knob -- PREPEND a zero-init Manduca proj filter; everything
  else iter-7 byte-for-byte): config flag `proj_filter_on=True`. A ManducaProjFilter
  module runs INSIDE the model at forward entry:
    1. N   = N0 * exp(-P)                photon counts (N0 = blank-scan flux, learnable log_N0)
    2. Q   = sqrt(N)                     Anscombe/sqrt VARIANCE STABILISATION
    3. Q̂   = BF_proj(Q)                  ONE TrainableBilateralFilter2d, fixed range σ on Q
    4. N̂   = Q̂²  ;  P̂ = -ln(max(N̂,ε)/N0)    denoised log-sinogram
    5. sino_filtered = P + proj_gain * (P̂ - P)   ZERO-INIT-gain RESIDUAL
  NOISE-ADAPTIVITY (the Manduca mechanism): the bilateral runs on the sqrt-count
  image Q where, under Poisson statistics, the noise std is ~CONSTANT (Anscombe)
  REGARDLESS of the local count level. A single fixed range σ on Q is therefore
  IMPLICITLY noise-adaptive in the line-integral domain: a high-attenuation (dense,
  low-count, NOISY) ray P has small N=N0·exp(-P) -> small Q -> the SAME range σ
  spans MORE of its noise -> MORE smoothing; an air ray (high count, clean) gets
  proportionally LESS. The strength scale is N0 (the photon count: noisier =
  high-attenuation rays smoothed more), parameterised as a learnable log_N0.
  WHERE IT INSERTS: sino_filtered replaces the raw sino in BOTH the LD-FBP init
  (recomputed inside forward so grads flow to the proj-filter params) AND the DC
  term g = R^T(R x - sino_filtered). The FoE image reg is byte-for-byte iter-7.
SPEND CHOICE -- proj_n_bf=1 (single proj-bilateral): 3 sigmas (log_sx,log_sy,
  log_sr) + 1 learnable log_N0 (the noise-adaptivity scale) + 1 scalar proj_gain
  = 5 extra trainable params. The MINIMAL noise-adaptive proj filter (the prompt's
  ~tens-of-params range). NO trainable bowtie (that adds n_det params and is not
  needed to test the COMBINATION). ONE gain keeps the proj lever a clean on/off
  knob the optimiser can drive to ~0 if the dense-view FBP already averages the
  projection noise (the predicted dense-view-ceiling outcome).
EXACT PARAM COUNT (iter-26 vs iter-7):
  FoE analysis 24*7*7 = 1,176  +  FoE RBF 24*31 = 744                = 1,920 (iter-7)
  + 1 scalar alpha                                                  =     1
  + proj-bilateral 3 sigmas                                         =     3 (NEW)
  + proj log_N0 (noise-adaptivity scale)                            =     1 (NEW)
  + proj_gain (zero-init residual gain)                             =     1 (NEW)
  = 1,926 TOTAL  (iter-7 was 1,921; +5 params, +0.26%).
  (NB the image-domain bilateral_on_top of iter-25 is OFF in iter-26 -- this iter
   tests the PROJECTION-domain lever in isolation, not stacked with the iter-25
   image lever.)
VERIFY-AT-INIT (mandatory): a runtime self-check builds the model and on ONE val
  sino asserts proj_gain==0.0 AND ‖sino_filtered - sino‖ == 0 EXACTLY (the proj
  filter is the identity at init REGARDLESS of the Manduca σ / N0). It then also
  asserts the FoE image reg(x)==0 at init (zero-init RBF) so the recon seed is
  iter-7's clean GD+DC scheme byte-for-byte. The check PRINTS proj_gain,
  ‖sino_filtered - sino‖, ‖reg(x)‖ and ABORTS LOUDLY before the 20-min run if any
  is non-zero.
STABILITY (why the dual-domain pipeline stays in the iter-7 basin):
  (1) sino_filtered == sino EXACTLY at init (proj_gain=0) => the LD-FBP init AND
      the DC g are iter-7's byte-for-byte. With zero-init RBF the FoE reg(x)==0
      too => the recon seed IS iter-7's clean GD+DC scheme. The proj filter is a
      STRICTLY ADDITIVE residual lifted off zero only by the supervised gradient.
      SAME zero-init-OUTPUT stability pattern as iter-7's zero-init rho' /
      iter-25's zero-init image gain, now applied to the projection stage.
  (2) NO change to the image unroll: FoE nf24/k7/nb31 UNCHANGED (dodges iter-8's
      nf40 divergence / iter-24's steerable divergence), K=5 single tied scalar
      alpha, NO momentum/per-step-alpha (dodges iter-3), NO extra unroll stage
      (dodges iter-10/11), NO learned-init forward stage (dodges that dead end).
      The proj filter is a fixed-form Manduca bilateral (no dense channel mixing),
      not a learned-init CNN.
  (3) TRAINER is iter-7 EXACTLY: plain Adam (NO weight-decay), peak lr 5e-3
      (NOT iter-12's diverged 1e-2), PARTIAL cosine T_max=16 (NOT iter-13/14's
      full anneal that overfit, NOT iter-15's diverged constant LR), grad_clip=1.0,
      bs=1, per-sample-ps, full-view DC, max_train_s=1080, train_n=200, val_n=214.
HYPOTHESIS (with the DENSE-VIEW CAVEAT): an UPSTREAM noise-adaptive proj denoiser
  cleans the LD-FBP init + DC g and lifts val-RMSE below iter-7's 7.40e-4 at +5
  params -> hr > 0.2515. HONEST CAVEAT: projection-domain filtering was found
  NEAR-COSMETIC / NEGATIVE *standalone* on this DENSE full-view Mayo (2,304 views
  -> the FBP already averages the projection noise) -- Manduca proj-BF standalone
  test hr ~0.069, PWLS ray-weighting logged NEGATIVE. At 2,304 views there is
  little projection-domain headroom; the big win for noise-adaptive proj filtering
  is SPARSE-view, not dense. iter-26 tests the COMBINATION on the FoE baseline.
  REALISTIC OUTCOME: modest / NULL (proj_gain trains toward ~0, hr ≈ 0.2515)
  confirming the dense-view ceiling -- a clean, valuable negative that says the
  dense FBP already subsumes the projection-domain denoise. SAFE: zero-init gain
  => seed = iter-7 => NO divergence/regression risk (worst case == iter-7).
  PREDICTED hr ~0.25-0.26 (gain -> ~0; the dense-view ceiling). REPORT proj_gain,
  the learned proj σ stack + log_N0 (their drift IS the answer: gain ~0 =
  dense-view ceiling confirmed; meaningfully nonzero = the proj denoise helps).
DEAD ENDS (do NOT reintroduce): denoiser-pretrain (froze x2); ordered-subsets
  (null); steerable/equivariant reparam (diverged); capacity^; K>5;
  momentum/per-step-alpha; separate learned-init forward stage; peak LR>5e-3;
  constant LR; full anneal to 1e-5; image-bilateral-ALONE (below floor); proj-
  bilateral-ALONE (below floor -- this iter is proj filter COMPOSED with FoE).

The learned regulariser `reg(x)` is selected by `reg_type`:
  - "foe"       (iter-7 CHAMPION base): TIED Fields-of-Experts / VN filter bank
                reg(x)=K^T rho'(Kx). analysis conv2d (`foe_n_filters` filters,
                `foe_kernel`x`foe_kernel`) -> per-filter RBF activation
                (`foe_n_bumps` bumps) -> TIED conv_transpose2d synthesis.
                ZERO-INIT rho'-weights => reg ~ 0 at init. nf24/k7/nb31 = 1,920p.
                When `bilateral_on_top=True` the FoE is WRAPPED in
                ComposedFoEBilateralReg (iter-25 DEFAULT): reg = FoE + gain*(x-BF).
  - "cnn"       (iter-1/iter-4 BEST, iter-6 FLAT-DILATED -- kept selectable).
  - "microunet" (iter-2/iter-5, REGRESSED -- pooling caps psnr 32.4; do NOT use).
  - "bilateral" (iter-17, bilateral ALONE ~17p -- BELOW the FBP floor; do NOT use
                ALONE. iter-25 uses it ON TOP via bilateral_on_top instead).

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
    # ---- architecture (iter-26: a NOISE-ADAPTIVE PROJECTION-domain Manduca filter PREPENDED to the iter-7 CHAMPION FoE image-domain unroll) ----
    "reg_type":        "foe",      # iter-26: KEEP "foe" (iter-7 CHAMPION image-domain bank, UNCHANGED). "foe" (iter-7 CHAMPION 1921p, hr 0.2515) | "cnn" (iter-4 / iter-6) | "microunet" (iter-2/iter-5, REGRESSED) | "bilateral" (iter-17 ALONE, below floor, do NOT use alone)
    "bilateral_on_top": False,     # iter-26: OFF. iter-25's IMAGE-domain bilateral-on-top is OFF here so iter-26 tests the PROJECTION-domain lever in ISOLATION (not stacked). True => the iter-25 image edge term.
    # ---- iter-26 THE LEVER: zero-init noise-adaptive PROJECTION-domain Manduca filter (Manduca et al. Med. Phys. 2009) PREPENDED to the unroll ----
    "proj_filter_on":  True,       # iter-26 THE LEVER: denoise the sino BEFORE the recon. sino_filtered = sino + proj_gain*(Manduca(sino) - sino); feeds BOTH the LD-FBP init AND the DC g. proj_gain ZERO-INIT (proj_gain_init=0.0) => sino_filtered == sino byte-for-byte at init => whole pipeline == iter-7. False => plain iter-7.
    "proj_n_bf":       1,          # iter-26: ONE proj-bilateral on the sqrt-count image Q (the MINIMAL noise-adaptive proj filter). 3 sigmas / filter.
    "proj_kernel":     5,          # iter-26: 5x5 proj-bilateral window (Manduca paper w~5; spatial weights computed explicitly, only the 3 sigmas + log_N0 + gain are learnable).
    "proj_sigma_x":    0.83,       # iter-26: spatial-x σ init (px) ~ Manduca d/w≈1/6 * w=5. Learnable via log_sx.
    "proj_sigma_y":    0.83,       # iter-26: spatial-y σ init (px). Learnable via log_sy.
    "proj_sigma_r":    1.5,        # iter-26: range σ init on the sqrt-count image Q in sqrt-count units (Manduca 0.7-2.8). Learnable via log_sr; this is what is noise-adaptive after the Anscombe transform.
    "proj_N0":         None,       # iter-26: incident blank-scan flux N0 for N=N0*exp(-P); None -> default to cfg["noise_i0"]. The NOISE-ADAPTIVITY scale (photon count). Learnable via log_N0.
    "proj_eps":        1.0,        # iter-26: floor on N̂ before the -ln (fp stability; never bites on real O(1-10) log-sinos).
    "proj_gain_init":  0.0,        # iter-26 THE STABILITY FIX: ZERO-INIT scalar gain => sino_filtered == sino at init REGARDLESS of the Manduca σ / N0 => LD-FBP init + DC g are iter-7's byte-for-byte. Training lifts the gain off 0 only if the proj denoise lowers val-RMSE.
    "n_iter":          5,          # unrolled prox-gradient steps, weight-TIED (iter-7 CHAMPION K; K>6 + momentum was UNSTABLE in iter-3, K=7 DIVERGED in iter-9)
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
    # ---- "foe" regulariser (iter-7 CHAMPION — TIED FoE/VN filter bank, the historical hr/param leader) ----
    "foe_n_filters":   24,    # iter-7 CHAMPION: 24 analysis filters (Hammernik-VN default bank width). iter-8's nf40 DIVERGED; iter-18/19's nf12/6 walked DOWN the frontier — 24 is the optimum.
    "foe_kernel":      7,     # iter-7 CHAMPION: 7 (eff RF 7px/step, ~19px over K=5 tied steps, no pooling). iter-16's k9 iso-param REGRESSED — 7 is the optimum.
    "foe_n_bumps":     31,    # iter-7 CHAMPION: 31 RBF bumps (Hammernik-VN default activation resolution)
    "foe_x_range":     1.0,
    "foe_filter_init_std": 0.05,
    "foe_rbf_init_std":    0.0,   # iter-7: ZERO-INIT synthesis -> ρ'≡0 => FoE(x)≡0 at init (stability; mirrors CNNReg zero-init head)
    # ---- "bilateral" residual term (iter-25: ON TOP of the FoE via bilateral_on_top; ZERO-INIT gain) ----
    "n_bf":            1,         # iter-25: ONE bilateral filter ON TOP of the FoE. reg += gain*(x - BF(x)). The MINIMAL complementary edge term (3 sigmas + 1 gain = 4 extra params).
    "bf_kernel":       7,         # iter-25: 7x7 bilateral window (eff RF 7px ~ matches iter-7's k7 FoE; spatial weights computed explicitly so only the 3 sigmas are learnable, NO trainable kernel weights).
    "bf_sigma_x":      1.5,       # iter-25: spatial-x bandwidth init (px). Learnable via log_sx.
    "bf_sigma_y":      1.5,       # iter-25: spatial-y bandwidth init (px). Learnable via log_sy.
    "bf_sigma_r":      0.02,      # iter-25: range (intensity) bandwidth init in mu units (clip_max=0.05). Learnable via log_sr; controls edge preservation.
    "bf_gain_init":    0.0,       # iter-25 THE STABILITY FIX: ZERO-INIT scalar gain => bilateral term = 0 at init => reg == FoE == iter-7 byte-for-byte. Training lifts the gain off 0 only if the edge term lowers val-RMSE.
    # ---- training (iter-7 CHAMPION: the proven-stable PARTIAL-cosine regime, BYTE-FOR-BYTE) ----
    "train_n":   200,
    "val_n":     214,
    "epochs":    16,          # iter-7: 16. With cosine_lr=True & T_max=16 the partial cosine starts 5e-3 and the hard max_train_s=1080 wall budget-cuts the run at ~ep8 (final LR ~2.5e-3) — iter-7's accidental-early-stop partial anneal, the ONLY stable+optimal regime.
    "cosine_lr": True,        # iter-7: partial cosine from peak 5e-3. iter-15's CONSTANT 5e-3 DIVERGED; iter-13/14's FULL anneal OVERFIT. Partial anneal is the optimum.
    "cosine_lr_min": 1e-5,    # iter-7 eta_min (the run never reaches it — budget-cut at ~ep8).
    "max_train_s": 1080,      # iter-7: hard 18-min train backstop so the run never overruns the 30-min slurm wall.
    "lr":        5e-3,        # iter-7 PEAK lr (NOT raised — iter-12's 2x peak 1e-2 DIVERGED; 5e-3 is the stability edge).
    "batch_size": 1,          # per-sample-ps geometry (Mayo): keep at 1
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
    correction). `dilations` may be None (all dilation 1, the iter-4 behaviour),
    a single int, or a list/tuple — recycled/truncated to `layers` entries."""

    def __init__(self, channels: int = 16, layers: int = 3,
                 dilations=None):
        super().__init__()
        channels = int(channels)
        layers = max(1, int(layers))
        if dilations is None:
            dil = [1] * layers
        elif isinstance(dilations, (int, float)):
            dil = [max(1, int(dilations))] * layers
        else:
            seq = [max(1, int(d)) for d in dilations] or [1]
            dil = [seq[i % len(seq)] for i in range(layers)]

        def _conv(ci, co, d):
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
    REGRESSED in iter-2/iter-5 (pooling caps psnr ~32.4): do NOT use."""

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
    """Single tied Fields-of-Experts / VN filter bank (iter-7 CHAMPION).

    reg(x) = K^T ρ'(K x), with K an analysis conv2d bank (n_filters,
    kernel x kernel), ρ' a per-filter RBF mixture (n_bumps bumps), and K^T
    the tied conv_transpose2d synthesis — exactly the regulariser-gradient
    of one solver_hammernik_vn.py VNStep, but ONE bank reused at every
    unrolled step (weight-tied).

    ZERO-INIT SYNTHESIS (iter-7 stability fix): with `rbf_init_std=0.0` the
    per-filter RBF mixture weights start at 0, so ρ'(·) ≡ 0 and therefore
    reg(x) ≡ 0 at init. The seed thus starts as the EXACT clean GD+DC scheme."""

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
    """Cascade/sum of `n_bf` TrainableBilateralFilter2d residuals (iter-17).

    reg(x) = sum_i gain_i * (x - BF_i(x)), per-filter learnable gain (zero-init).
    Used ALONE in iter-17 (below the FBP floor — do NOT use alone). iter-25 uses
    bilateral ON TOP of the FoE via ComposedFoEBilateralReg instead."""

    def __init__(self, n_bf: int = 4, kernel_size: int = 7,
                 sigma_x: float = 1.5, sigma_y: float = 1.5,
                 sigma_r: float = 0.02, gain_init: float = 0.0):
        super().__init__()
        n_bf = max(1, int(n_bf))
        self.filters = nn.ModuleList(
            [TrainableBilateralFilter2d(kernel_size=int(kernel_size),
                                        sigma_x=float(sigma_x),
                                        sigma_y=float(sigma_y),
                                        sigma_r=float(sigma_r))
             for _ in range(n_bf)])
        self.gains = nn.Parameter(torch.full((n_bf,), float(gain_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for i, f in enumerate(self.filters):
            out = out + self.gains[i] * (x - f(x))
        return out


class ComposedFoEBilateralReg(nn.Module):
    """iter-25 DEFAULT — composed regulariser = FoE + zero-init-gain bilateral.

        reg(x) = FoE(x)  +  gain * sum_i ( x - BF_i(x) )

    The iter-7 CHAMPION FoE bank (zero-init RBF => FoE(x)=0 at init) PLUS an
    edge-preserving bilateral RESIDUAL term, scaled by ONE learnable scalar
    `gain` that is ZERO-INIT. Both terms are EXACTLY 0 at init, so reg(x)==0 ==
    iter-7's clean GD+DC seed byte-for-byte. The optimiser lifts `gain` off zero
    ONLY if the complementary bilateral edge term lowers val-RMSE; if the FoE
    bank already subsumes an edge-preserving prior the gain trains toward ~0 and
    the run reproduces iter-7 (a clean NULL).

    A SINGLE shared scalar gain (NOT per-filter) keeps the added lever a clean
    on/off knob the optimiser can drive to ~0; the gain magnitude IS the answer
    to "does an edge term help on top of the FoE". Params: FoE (1,920 at iter-7
    geometry) + n_bf*3 bilateral sigmas + 1 gain. At n_bf=1: 1,920 + 3 + 1 =
    1,924 (total 1,925 incl. the 1 scalar alpha; iter-7 was 1,921, +4)."""

    def __init__(self, foe: nn.Module, n_bf: int = 1, kernel_size: int = 7,
                 sigma_x: float = 1.5, sigma_y: float = 1.5,
                 sigma_r: float = 0.02, gain_init: float = 0.0):
        super().__init__()
        self.foe = foe                          # the iter-7 FoE bank (zero-init RBF)
        n_bf = max(1, int(n_bf))
        self.filters = nn.ModuleList(
            [TrainableBilateralFilter2d(kernel_size=int(kernel_size),
                                        sigma_x=float(sigma_x),
                                        sigma_y=float(sigma_y),
                                        sigma_r=float(sigma_r))
             for _ in range(n_bf)])
        # ONE shared scalar gain, ZERO-INIT => bilateral term = 0 at init =>
        # reg == FoE (== iter-7) at init REGARDLESS of the sigma values.
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))

    def bilateral_term(self, x: torch.Tensor) -> torch.Tensor:
        """The (un-gained) summed bilateral residual sum_i (x - BF_i(x)).
        Exposed for the runtime zero-at-init self-check."""
        res = torch.zeros_like(x)
        for f in self.filters:
            res = res + (x - f(x))
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.foe(x) + self.gain * self.bilateral_term(x)


class ManducaProjFilter(nn.Module):
    """iter-26 — zero-init noise-adaptive PROJECTION-domain Manduca filter.

    A learnable RESIDUAL projection-domain denoiser (Manduca et al., Med. Phys.
    2009) PREPENDED to the image-domain unroll. On a log-sinogram / line-integral
    tensor P (B,1,A,D):

        N  = N0 * exp(-P)            photon counts (N0 = blank-scan flux, learnable)
        Q  = sqrt(N)                 Anscombe/sqrt VARIANCE STABILISATION
        Q̂  = BF_proj(Q)              ONE TrainableBilateralFilter2d, fixed range σ on Q
        N̂  = Q̂²                      (Q̂ clamped >= 0)
        P̂  = -ln(max(N̂, ε) / N0)     denoised log-sinogram
        out = P + gain * (P̂ - P)     ZERO-INIT-gain residual

    NOISE-ADAPTIVITY (Manduca §II.B-E): the bilateral runs on the sqrt-count image
    Q where, under Poisson statistics, the noise std is ~CONSTANT (Anscombe)
    regardless of the local count level, so a SINGLE fixed range σ on Q is
    IMPLICITLY noise-adaptive in the line-integral domain -- a high-attenuation
    (dense, low-count, NOISY) ray P -> small N=N0·exp(-P) -> small Q -> the same
    range σ spans more of its noise -> MORE smoothing; an air ray (high count,
    clean) gets proportionally less. The strength scale is the photon count N0
    (parameterised in log-space so it stays strictly positive and -ln(N̂/N0) is
    well defined).

    ZERO-INIT GAIN (the stability fix): with ``gain_init=0.0`` the scalar gain is
    0 at init, so out == P EXACTLY at init REGARDLESS of the σ / N0 values -- the
    LD-FBP init AND the DC term g downstream are iter-7's byte-for-byte. The
    optimiser lifts the gain off zero only if the proj denoise lowers val-RMSE.

    Trainable params: 3 bilateral σ (log_sx, log_sy, log_sr) + 1 log_N0 (the
    noise-adaptivity scale) + 1 scalar gain = 5 at proj_n_bf=1 (one σ-triple per
    extra stacked filter). NO trainable bowtie (kept a clean scalar-N0 noise
    model for the in-isolation projection lever)."""

    def __init__(self, n_bf: int = 1, kernel_size: int = 5,
                 sigma_x: float = 0.83, sigma_y: float = 0.83,
                 sigma_r: float = 1.5, N0: float = 1.0e5,
                 eps: float = 1.0, gain_init: float = 0.0):
        super().__init__()
        n_bf = max(1, int(n_bf))
        k = int(kernel_size)
        if k % 2 == 0:                       # bilateral needs an odd kernel
            k += 1
        self.filters = nn.ModuleList(
            [TrainableBilateralFilter2d(kernel_size=k,
                                        sigma_x=float(sigma_x),
                                        sigma_y=float(sigma_y),
                                        sigma_r=float(sigma_r))
             for _ in range(n_bf)])
        self.log_N0 = nn.Parameter(torch.tensor(math.log(max(float(N0), 1.0))))
        self.eps = float(eps)
        # ONE shared scalar gain, ZERO-INIT => residual == 0 at init => out == P
        # at init REGARDLESS of the σ / N0 values.
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))

    def _manduca(self, P: torch.Tensor) -> torch.Tensor:
        """The denoised log-sinogram P̂ (un-gained)."""
        logN0 = self.log_N0
        # N = N0*exp(-P) = exp(logN0 - P); clamp the exponent for fp stability
        # (staged log-sinos are O(1-10) so this never bites on real data).
        N = torch.exp((logN0 - P).clamp(max=30.0))
        Q = torch.sqrt(N.clamp_min(0.0) + 1e-8)      # Anscombe/sqrt stabilisation
        for f in self.filters:
            Q = f(Q)
        Nhat = Q.clamp_min(0.0) ** 2
        return logN0 - torch.log(Nhat.clamp_min(self.eps))   # -ln(N̂/N0)

    def residual(self, P: torch.Tensor) -> torch.Tensor:
        """The (un-gained) residual P̂ - P. Exposed for the zero-at-init check."""
        return self._manduca(P) - P

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        return P + self.gain * self.residual(P)

    @torch.no_grad()
    def sigmas(self):
        return [(float(torch.exp(f.log_sx).cpu()),
                 float(torch.exp(f.log_sy).cpu()),
                 float(torch.exp(f.log_sr).cpu())) for f in self.filters]

    @torch.no_grad()
    def N0_value(self) -> float:
        return float(torch.exp(self.log_N0).cpu())


def build_reg(cfg: dict) -> nn.Module:
    rt = cfg["reg_type"]
    if rt == "microunet":
        return MicroUNetReg(channels=cfg.get("mu_channels", 16))
    if rt == "cnn":
        return CNNReg(channels=cfg["cnn_channels"], layers=cfg["cnn_layers"],
                      dilations=cfg.get("cnn_dilations"))
    if rt == "foe":
        foe = FoEReg(n_filters=cfg["foe_n_filters"], kernel_size=cfg["foe_kernel"],
                     n_bumps=cfg["foe_n_bumps"], x_range=cfg["foe_x_range"],
                     filter_init_std=cfg["foe_filter_init_std"],
                     rbf_init_std=cfg["foe_rbf_init_std"])
        # iter-25: optionally compose a zero-init-gain bilateral residual ON TOP.
        if bool(cfg.get("bilateral_on_top", False)):
            return ComposedFoEBilateralReg(
                foe, n_bf=cfg.get("n_bf", 1), kernel_size=cfg.get("bf_kernel", 7),
                sigma_x=cfg.get("bf_sigma_x", 1.5),
                sigma_y=cfg.get("bf_sigma_y", 1.5),
                sigma_r=cfg.get("bf_sigma_r", 0.02),
                gain_init=cfg.get("bf_gain_init", 0.0))
        return foe
    if rt == "bilateral":
        return BilateralReg(n_bf=cfg["n_bf"], kernel_size=cfg["bf_kernel"],
                            sigma_x=cfg.get("bf_sigma_x", 1.5),
                            sigma_y=cfg.get("bf_sigma_y", 1.5),
                            sigma_r=cfg.get("bf_sigma_r", 0.02),
                            gain_init=cfg.get("bf_gain_init", 0.0))
    raise ValueError(f"unknown reg_type={rt!r} (expected microunet|cnn|foe|bilateral)")


# ---------------------------------------------------------------------------
class ParamEfficientUnrolled(nn.Module):
    """Weight-tied unrolled proximal-gradient with data consistency (iter-7).

    x_0 = u0 (LD-FBP). For k in range(n_iter):
        dc = R^T(R x - g) / dc_norm
        x  = clamp(x - alpha * (dc + reg(x)), 0.0, clip_max)
    `reg` AND `alpha` are SHARED across all steps (weight-tied). NO per-step
    alpha, NO momentum — iter-3 proved both destabilise the recon in-budget.

    For backward-compatibility the per_step_alpha / momentum config knobs are
    still honoured (so the iter-3 variant remains selectable), but the iter-7
    DEFAULTS turn BOTH off, collapsing the iteration to the plain tied
    prox-gradient. With per_step_alpha=False and momentum=False:
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
        # iter-26: optional zero-init noise-adaptive PROJECTION-domain Manduca
        # filter, PREPENDED to the image unroll. When active, the model filters
        # the sino at forward entry; the filtered sino feeds BOTH the LD-FBP init
        # (recomputed inside forward so grads flow to the proj-filter params) AND
        # the DC term g. proj_gain=0 at init => sino_filtered == sino byte-for-byte.
        self.proj_filter_on = bool(cfg.get("proj_filter_on", False))
        if self.proj_filter_on:
            N0 = cfg.get("proj_N0")
            if N0 is None:
                N0 = cfg.get("noise_i0", 1.0e5)
            self.proj_filter = ManducaProjFilter(
                n_bf=cfg.get("proj_n_bf", 1),
                kernel_size=cfg.get("proj_kernel", 5),
                sigma_x=cfg.get("proj_sigma_x", 0.83),
                sigma_y=cfg.get("proj_sigma_y", 0.83),
                sigma_r=cfg.get("proj_sigma_r", 1.5),
                N0=float(N0), eps=cfg.get("proj_eps", 1.0),
                gain_init=cfg.get("proj_gain_init", 0.0))
        else:
            self.proj_filter = None
        if cfg["learnable_alpha"]:
            inv_softplus = math.log(math.expm1(max(float(cfg["alpha_init"]), 1e-6)))
            n = self.n_iter if self.per_step_alpha else 1
            self.log_alpha = nn.Parameter(torch.full((n,), float(inv_softplus)))
            self._alpha_const = None
        else:
            self.log_alpha = None
            self._alpha_const = float(cfg["alpha_init"])
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
        """One (optionally accelerated) prox-gradient step. With beta==0 (iter-7
        default) y_new == x_new, i.e. the plain tied prox step of iter-1."""
        x_new = torch.clamp(y - alpha_k * self._grad(y, sino), 0.0, self.clip_max)
        v = x_new - x
        y_new = x_new + beta * v
        return x_new, y_new

    def forward(self, u0: torch.Tensor, sino: torch.Tensor) -> torch.Tensor:
        # iter-26: when the projection-domain filter is active, denoise the sino
        # FIRST, then recompute the LD-FBP init from the FILTERED sino (so grads
        # flow into the proj-filter params through the init too) and use the
        # filtered sino for the DC term g. With proj_gain=0 the filter is the
        # identity (sino_filtered == sino) so x0 == u0 == iter-7's init exactly.
        if self.proj_filter is not None:
            sino = self.proj_filter(sino)
            u0 = torch.clamp(self.projector.fbp(sino), min=0.0)
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
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)


def _count_reg_params(cfg: dict) -> int:
    """Trainable param count of ONE regulariser at the given cfg."""
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

    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}", flush=True)
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k in ('reg_type','bilateral_on_top','proj_filter_on','proj_n_bf','proj_kernel','proj_sigma_x','proj_sigma_y','proj_sigma_r','proj_N0','proj_eps','proj_gain_init','n_iter','learnable_alpha','per_step_alpha','momentum','beta_init','alpha_init','clip_max','mu_channels','cnn_channels','cnn_layers','cnn_dilations','foe_n_filters','foe_kernel','foe_n_bumps','foe_rbf_init_std','n_bf','bf_kernel','bf_sigma_x','bf_sigma_y','bf_sigma_r','bf_gain_init','epochs','cosine_lr','cosine_lr_min','max_train_s','batch_size','lr','train_n','val_n')}, default=str)}",
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

    # Pin a per-ps projector for the runtime self-checks (Mayo swaps per slice).
    if per_ps and (model.proj_filter is not None or bool(cfg.get("bilateral_on_top", False))):
        model.projector = _projs[float(_vrk[0])]

    # iter-26 RUNTIME SELF-CHECK (the verify-at-init PROJECTION-domain guard): on
    # ONE val sino, confirm sino_filtered == sino at init by asserting proj_gain
    # ==0.0 AND ‖proj_filter(sino) - sino‖ == 0 EXACTLY (the proj filter is the
    # identity at init REGARDLESS of the Manduca σ / N0). => the LD-FBP init AND
    # the DC g downstream are iter-7's byte-for-byte. A non-zero gain (a broken
    # zero-init) FAILS LOUDLY here, BEFORE the 20-min run is spent.
    if model.proj_filter is not None and val_noisy.shape[0] > 0:
        with torch.no_grad():
            p_chk = val_noisy[0:1]
            pgain = float(model.proj_filter.gain.detach().cpu())
            sino_f = model.proj_filter(p_chk)
            sino_diff_n = float((sino_f - p_chk).norm())
            resid_n = float(model.proj_filter.residual(p_chk).norm())
            print(f"[selfcheck] proj_filter==identity at init: proj_gain={pgain:.3e}  "
                  f"‖sino_filtered - sino‖={sino_diff_n:.3e}  "
                  f"‖(un-gained) residual P̂-P‖={resid_n:.3e}  N0_init="
                  f"{model.proj_filter.N0_value():.4g}  "
                  f"(proj_gain MUST be 0.0, ‖sino_filtered - sino‖ MUST be 0)",
                  flush=True)
            assert pgain == 0.0, (
                f"proj_gain NOT zero at init (proj_gain={pgain:.3e}): the proj "
                "zero-init gain is broken -> sino_filtered != sino. ABORTING.")
            assert sino_diff_n == 0.0, (
                f"sino_filtered != sino at init (‖diff‖={sino_diff_n:.3e}): the "
                "proj filter is NOT exactly the identity at init. ABORTING.")

    # iter-25 RUNTIME SELF-CHECK (kept; only fires when bilateral_on_top=True):
    # confirm the IMAGE reg(x) == FoE(x) at init via the zero-init bilateral gain.
    if bool(cfg.get("bilateral_on_top", False)) and isinstance(
            model.reg, ComposedFoEBilateralReg) and val_u0.shape[0] > 0:
        with torch.no_grad():
            x_chk = val_u0[0:1]
            gain_val = float(model.reg.gain.detach().cpu())
            foe_out = model.reg.foe(x_chk)
            bil_unscaled = model.reg.bilateral_term(x_chk)
            bil_term = gain_val * bil_unscaled        # the actual added term
            reg_out = model.reg(x_chk)
            foe_n = float(foe_out.norm())
            bil_n = float(bil_term.norm())
            reg_n = float(reg_out.norm())
            diff_n = float((reg_out - foe_out).norm())
            print(f"[selfcheck] composed reg==FoE at init: gain={gain_val:.3e}  "
                  f"‖FoE(x)‖={foe_n:.3e}  ‖gain*(x-BF)‖={bil_n:.3e}  "
                  f"‖reg(x)‖={reg_n:.3e}  ‖reg-FoE‖={diff_n:.3e}  "
                  f"(gain MUST be 0.0, ‖reg-FoE‖ MUST be 0)", flush=True)
            assert gain_val == 0.0, (
                f"bilateral gain NOT zero at init (gain={gain_val:.3e}): the "
                "zero-init gain is broken -> seed is NOT iter-7. ABORTING.")
            assert diff_n == 0.0, (
                f"reg(x) != FoE(x) at init (‖reg-FoE‖={diff_n:.3e}): the "
                "bilateral term is NOT exactly 0 at init. ABORTING.")

    # iter-7 FoE zero-init guard (fires whenever foe_rbf_init_std==0.0): the image
    # reg(x) == 0 at init so the recon seed is iter-7's clean GD+DC scheme.
    if cfg.get("reg_type") == "foe" and float(cfg.get("foe_rbf_init_std", 0.0)) == 0.0 \
            and val_u0.shape[0] > 0:
        with torch.no_grad():
            foe_mod = model.reg.foe if isinstance(model.reg, ComposedFoEBilateralReg) else model.reg
            foe_chk_n = float(foe_mod(val_u0[0:1]).norm())
            print(f"[selfcheck] FoE(x)==0 at init: ‖FoE(x)‖={foe_chk_n:.3e}  "
                  f"(MUST be 0 with zero-init RBF => recon seed == iter-7)",
                  flush=True)
            assert foe_chk_n == 0.0, (
                f"FoE(x) != 0 at init (‖FoE‖={foe_chk_n:.3e}) despite zero-init "
                "RBF: the FoE zero-init is broken. ABORTING.")

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_params = _count_reg_params(cfg)
    n_alpha = (model.n_iter if model.per_step_alpha else 1) if model.log_alpha is not None else 0
    n_beta = 1 if model.beta_raw is not None else 0
    n_gain = (1 if (bool(cfg.get("bilateral_on_top", False))
                    and isinstance(model.reg, ComposedFoEBilateralReg)) else 0)
    # iter-26: the PROJECTION-domain Manduca filter params (3 σ/filter + log_N0 +
    # gain), counted separately from the image-domain reg.
    proj_params = (sum(p.numel() for p in model.proj_filter.parameters()
                       if p.requires_grad) if model.proj_filter is not None else 0)
    print(f"[solver] ParamEfficient iter-26: reg_type={cfg['reg_type']!r}  "
          f"proj_filter_on={bool(cfg.get('proj_filter_on', False))} "
          f"(proj_n_bf={cfg.get('proj_n_bf')}, proj_kernel={cfg.get('proj_kernel')}, "
          f"proj_gain_init={cfg.get('proj_gain_init')})  "
          f"bilateral_on_top={bool(cfg.get('bilateral_on_top', False))}  "
          f"n_iter={cfg['n_iter']} (weight-TIED reg)  "
          f"per_step_alpha={model.per_step_alpha} momentum={model.use_momentum}  "
          f"epochs={cfg['epochs']} cosine_lr={cfg.get('cosine_lr', False)} "
          f"peak_lr={cfg.get('lr')}  "
          f"trainable params={params_total} "
          f"(reg={reg_params} + alpha={n_alpha} + beta={n_beta}; gain∈reg={n_gain}; "
          f"proj_filter={proj_params})  "
          f"= {params_total/1e6:.6f} M  vs 233k ITNet", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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
        _gain_s = (f" gain={float(model.reg.gain.detach().cpu()):.4g}"
                   if isinstance(model.reg, ComposedFoEBilateralReg) else "")
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"lr={cur_lr:.3g}  alpha={float(model.alpha.detach().cpu()):.4g}"
              f"{_gain_s}", flush=True)
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

    with torch.no_grad():
        if per_ps:
            val_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

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

    # iter-25: report the learned IMAGE bilateral gain (off when bilateral_on_top
    # is False, the iter-26 default).
    gain_learned = (float(model.reg.gain.detach().cpu())
                    if isinstance(model.reg, ComposedFoEBilateralReg) else None)

    # iter-26: report the learned PROJECTION-domain Manduca filter. proj_gain's
    # magnitude IS the answer: ~0 = the dense-view FBP already subsumes the proj
    # denoise (the predicted dense-view ceiling, a clean NULL); meaningfully
    # nonzero = the upstream noise-adaptive proj filter contributes on top of FoE.
    if model.proj_filter is not None:
        proj_gain_learned = float(model.proj_filter.gain.detach().cpu())
        proj_sigmas_learned = model.proj_filter.sigmas()
        proj_N0_learned = model.proj_filter.N0_value()
        proj_params_n = sum(p.numel() for p in model.proj_filter.parameters()
                            if p.requires_grad)
    else:
        proj_gain_learned = None
        proj_sigmas_learned = None
        proj_N0_learned = None
        proj_params_n = 0

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
        "bilateral_on_top": bool(cfg.get("bilateral_on_top", False)),
        "n_bf": int(cfg.get("n_bf", 1)),
        "bf_gain_init": float(cfg.get("bf_gain_init", 0.0)),
        "gain_learned": gain_learned,
        # iter-26: projection-domain Manduca filter
        "proj_filter_on": bool(cfg.get("proj_filter_on", False)),
        "proj_n_bf": int(cfg.get("proj_n_bf", 1)),
        "proj_kernel": int(cfg.get("proj_kernel", 5)),
        "proj_gain_init": float(cfg.get("proj_gain_init", 0.0)),
        "proj_params": proj_params_n,
        "proj_gain_learned": proj_gain_learned,
        "proj_sigmas_learned": proj_sigmas_learned,
        "proj_N0_learned": proj_N0_learned,
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
    _gain_s = f"{gain_learned:.4g}" if gain_learned is not None else "off"
    if proj_gain_learned is not None:
        _ps = "; ".join(f"σx={sx:.3f} σy={sy:.3f} σr={sr:.4f}"
                        for (sx, sy, sr) in proj_sigmas_learned)
        _proj_s = f"proj_gain={proj_gain_learned:.4g} N0={proj_N0_learned:.4g} proj[{_ps}]"
    else:
        _proj_s = "proj_filter=off"
    print(f"[solver] ParamEfficient: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"params={params_total}  alpha_mean={result['alpha_learned']:.4g}  "
          f"gain={_gain_s}  beta={_beta_s}  {_proj_s}  time={train_time:.1f}s  "
          f"(intensity-calibrated)", flush=True)

    try:
        _lbl = (f"ParamEff[{cfg['reg_type']}+projBF]"
                if model.proj_filter is not None
                else f"ParamEff[{cfg['reg_type']}+bil]")
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label=_lbl,
            headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
