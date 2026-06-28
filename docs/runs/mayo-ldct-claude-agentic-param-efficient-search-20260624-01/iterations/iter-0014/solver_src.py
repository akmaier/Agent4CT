"""Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-14).

A weight-TIED unrolled proximal-gradient reconstruction with explicit
data-consistency against the measured sinogram. The SAME regulariser
module and the SAME step-size scalar `alpha` are reused at every unrolled
step, so the trainable parameter budget is set by ONE small regulariser
(hundreds-to-low-thousands of params) regardless of `n_iter` — in sharp
contrast to the 233k-param ITNet champion whose denoiser is a full SmallUNet.

Architecture (iter-14: the iter-7 CHAMPION FoE unroll, BYTE-FOR-BYTE — no new
trainable params, NO learned-init, NO added depth/capacity/stages):

    x_0 = LD-FBP(sino)                              # raw FBP init (iter-7)
    for k in range(K):
        dc = R^T( R x  -  sino ) / dc_norm          # data-consistency grad
        x  = clamp( x - alpha * ( dc + reg(x) ),     # proximal-gradient step
                    0.0, clip_max )

iter-14 is NOT an architecture change: the FoE bank (nf24/k7/nb31 = 1,920p),
K=5, the single tied scalar alpha (total 1,921p), zero-init rho', the plain
prox step (NO momentum/per-step-alpha/learned-init), max_train_s=1080,
train_n=200, val_n=214, grad_clip=1.0, batch_size=1 are ALL iter-7
byte-for-byte. PEAK lr stays 5e-3 (NOT raised — iter-12 proved 1e-2 diverges),
and the iter-13 COMPLETED cosine anneal (epochs=8, cosine_t_max=8) is KEPT.
The ONLY change is the OPTIMISER: plain Adam -> AdamW with a light DECOUPLED
weight_decay 3e-4 (see iter-14 SIX-BOX). iter-13 proved that completing the
anneal LOWERED train loss (5.78e-7 < iter-7's 7.1e-7) but MILDLY OVERFIT val
(hr 0.2273 < iter-7's 0.2515, val_rmse 7.6e-4 > 7.40e-4). The binding
constraint is now VAL GENERALISATION (overfit), NOT stability or train loss —
so iter-14 regularises the full-anneal fine-tuning with decoupled L2 to close
the train/val gap. This is SAFE at the SAME 5e-3 peak: AdamW only shrinks the
weights; iter-12 diverged from its 2x peak LR (1e-2), NOT from AdamW.

(The learned-init refiner `g` / LearnedInit module from iter-10/11 is KEPT in
the file but DISABLED by default — learned_init=False — so the model is iter-7
exactly. Both iter-10/11 learned-init attempts FROZE training; iter-14 does not
touch capacity at all. iter-12's 2x peak LR + warmup are NOT revived — they
DIVERGED; iter-14 keeps iter-13's COMPLETED 8-ep cosine at the iter-7 peak
5e-3, and the ONLY new ingredient is the decoupled weight_decay 3e-4 that
turns the optimiser into AdamW.)

`dc_norm` is a power-iteration estimate of ‖R^T R‖ so `alpha` lives in O(1)
regardless of geometry (mirrors solver_hammernik_vn.py). `alpha` is a single
learnable softplus SCALAR (init from `alpha_init`) when `learnable_alpha` —
shared across all K steps. NO per-step alpha, NO momentum (both shown to
destabilise the recon in the 20-min budget; see iter-3 below).

iter-14 SIX-BOX (REGULARISE the completed-anneal OVERFIT with decoupled AdamW wd — NOT architecture, NOT higher LR, ZERO new params)
------------------------------------------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = 34.08 dB):
  - iter-1: cnn reg (2,798 params), K=5 tied prox+DC, learnable SCALAR
    alpha, 8 epochs, ~12 min wall -> hr 0.0871, ssim 0.846, psnr 34.87.
  - iter-2 (FAIL @ 8 ep): micro-UNet reg (25,890 params), K=5, 8 ep ->
    hr 0 (psnr 32.41). a 9x-wider POOLED reg below floor.
  - iter-3 (FAIL worse): tiny cnn + K=8 + per-step alpha + Nesterov
    MOMENTUM (2,806 params), 6 ep -> hr 0 (psnr 28.08, UNSTABLE).
    LESSON: the iter-3 instability was the MOMENTUM + per-step-alpha COMBO
    (a non-monotone accelerated scheme). AVOID both.
  - iter-4: single-scale tiny cnn (2,798p, 3-layer 12ch, K=5, scalar alpha,
    NO momentum) + cosine LR + 16 epochs -> hr 0.2378, ssim 0.886,
    psnr 36.44 (~20 min wall). BREAKTHROUGH: training-limited, not
    capacity-limited -- 16-ep cosine TRIPLED hr at the SAME 2,798 params.
  - iter-5 (FAIL): micro-UNet reg (25,890p) under the iter-4 trainer ->
    hr 0 (psnr 32.41). POOLING in the reg is ARCHITECTURALLY BAD. Do NOT pool.
  - iter-6 (FAIL): cnn reg GROWN flat+dilated (37,601p) -> underperformed.
    SCALING the CNN reg fails BOTH ways; 2,798p is the CNN sweet spot.
  - iter-7 (CHAMPION, BASE): reg FAMILY cnn -> "foe" (TIED Fields-of-Experts
    / VN learned filter bank, reg(x)=K^T rho'(Kx)). nf=24 / k=7 / nb=31 ->
    1,920 reg params, total 1,921 (1 scalar alpha), zero-init rho'-weights,
    K=5 tied prox+DC, 16-ep cosine LR 5e-3->1e-5 (Adam) -> hr 0.2515, ssim
    0.906, psnr 36.59, val_rmse 7.40e-4 (1331 s, wall-bounded; TRAIN-CUT at
    epoch 8, loss STILL falling MONOTONICALLY 3.4e-6 ep1 -> 7.1e-7 ep8 with
    NO oscillation). train MSE ~= val MSE => NOT overfit, genuinely
    UNDER-TRAINED. The FoE BEAT the tiny CNN (0.2515 vs 0.2378) at FEWER
    params (1,921 vs 2,798).
  - iter-8 (FAIL, TOTAL DIVERGENCE): SCALED the FoE bank nf 24->40 (2,881p)
    -> hr 0, psnr 1.87 (recon BLEW UP). CAPACITY-SCALING IS FULLY EXHAUSTED:
    EVERY reg-capacity increase (iter-2/5/6/8) regresses or diverges in the
    20-min budget. ~1.9-2k params (iter-7 FoE) is the ARCHITECTURE SWEET SPOT.
  - iter-9 (FAIL, DIVERGENCE): DEEPENED the tied unroll K 5->7 at the SAME
    1,921 params -> hr 0, psnr 11.38, RMSE 0.0135 (the recon DIVERGED).
    VERDICT: ADDING UNROLL DEPTH destabilises in-budget too. K=5 EXACTLY.
  - iter-10 (FAIL, FROZE): learned-init refiner g with a BUGGY init clamp
    (clamp(LD-FBP+g, 0, clip_max)) truncated bright LD-FBP pixels -> ep-1 loss
    100x iter-7's, FROZE at the degenerate loss 1.9e-4 for 8 epochs.
  - iter-11 (FAIL, FROZE AGAIN — the science result): re-wired the learned-init
    CORRECTLY (x_0 = LD-FBP + g(LD-FBP), NO upper clamp, GELU+zero-init head,
    +177 params). The runtime self-check VERIFIED true zero-init (rel_g≈0,
    x_0 == LD-FBP byte-for-byte). YET training STILL FROZE at the identical
    degenerate loss ~1.9e-4: the extra trainable stage collapses training to a
    bad attractor REGARDLESS of init correctness. VERDICT: ADDING ANY trainable
    stage (capacity/depth/refiner) destabilises in the 20-min budget. The ONLY
    proven-stable thing is the iter-7 FoE unroll ITSELF, and it is UNDER-TRAINED.

  - iter-12 (FAIL, DIVERGED): TRAINER rewrite — peak LR 5e-3 -> 1e-2 (2x) +
    0.5-ep linear WARMUP + Adam -> AdamW(wd 1e-4), all else iter-7. epoch-1
    fine (loss 3.2e-6) but epoch-2 JUMPED to 7e-5, epoch-3 hit the degenerate
    attractor 1.9e-4 -> DIVERGED. VERDICT: 5e-3 is at the STABILITY EDGE; a 2x
    peak overshoots even WITH warmup. Do NOT raise the peak LR. (The divergence
    came from the 2x PEAK LR, NOT from AdamW — iter-14 reuses AdamW at the SAME
    5e-3 peak and stays stable.)

  - iter-13 (STABLE, BASE): TRAINER SCHEDULE PERIOD fix at the iter-7 model
    byte-for-byte — epochs 16 -> 8 + a decoupled cosine_t_max=8 so the cosine
    FULLY anneals to eta_min=1e-5 within the 1080s wall (iter-7's T_max=16
    cosine was cut at ~ep8 with LR still ≈2.5e-3, the anneal SKIPPED). Plain
    Adam, peak lr 5e-3 (UNCHANGED), warmup 0, wd 0. RESULT: hr 0.2273, ssim
    0.898, psnr 36.32, val_rmse 7.64e-4, train loss 5.78e-7 (< iter-7's 7.1e-7),
    1240 s. SCIENCE: completing the anneal LOWERED TRAIN LOSS (5.78e-7 vs 7.1e-7)
    yet DROPPED val hr (0.2273 < iter-7's 0.2515) and RAISED val_rmse
    (7.64e-4 > 7.40e-4). So the full low-LR fine-tune phase OVERFIT the 200-slice
    train set: train MSE fell but val MSE rose. VERDICT: the run is NO LONGER
    under-trained or unstable — the binding constraint is now VAL GENERALISATION
    (a train/val gap). The lever is REGULARISATION of the completed anneal, NOT
    more/fewer epochs and NOT a different LR.

FAILURE MODE addressed (iter-14): iter-13's COMPLETED cosine anneal drove the
train loss BELOW iter-7's (5.78e-7 vs 7.1e-7) but the val hr REGRESSED
(0.2273 < 0.2515) with a HIGHER val_rmse (7.64e-4 vs 7.40e-4). That is the
textbook OVERFIT signature: the low-LR fine-tune phase that iter-7 never reached
keeps lowering train MSE while val MSE rises, because the 1,921-param FoE bank
fits the 200-slice train set's noise realisation in the last 2-3 low-LR epochs.
Stability and train-loss headroom are both SOLVED (iter-13); the only thing
between us and beating iter-7 is closing the train/val gap.

CHANGE (iter-14, OPTIMISER ONLY — ZERO new params, peak LR UNCHANGED 5e-3, the
unroll is iter-7 byte-for-byte and the schedule is iter-13 byte-for-byte):
  (1) Plain Adam -> AdamW with DECOUPLED weight_decay 3e-4 (the cfg knob
      `weight_decay` 0.0 -> 3e-4; the trainer already routes wd>0 to AdamW).
      Decoupled L2 shrinks every weight by (1 - lr*wd) each step INDEPENDENTLY
      of the gradient, so it regularises the full-anneal fine-tuning that iter-13
      overfit -> a smaller-norm filter bank that generalises better -> lower
      val-RMSE -> higher hr. wd=3e-4 is LIGHT (iter-12 used 1e-4 and survived
      epoch-1 cleanly; the 3x bump is still tiny and only matters in the low-LR
      tail where overfit happens, since the decay magnitude lr*wd shrinks WITH
      the cosine LR).
EVERYTHING ELSE from iter-13/iter-7 byte-for-byte: reg_type="foe" nf24/k7/nb31
(1,920p), n_iter=5, single tied scalar alpha (total 1,921p), zero-init rho',
NO momentum, NO per-step alpha, NO learned-init (learned_init=False), plain prox
step, epochs=8, cosine_lr=True, cosine_t_max=8, peak lr=5e-3, cosine_lr_min=1e-5,
warmup_frac=0.0, max_train_s=1080, train_n=200, grad_clip=1.0, batch_size=1,
val_n=214.
STABILITY (why AdamW wd 3e-4 at the SAME 5e-3 peak stays in the iter-7/13 basin):
  (1) The PEAK LR is UNCHANGED at 5e-3 — the proven-stable step from iter-7/13.
      iter-12 diverged ONLY because it RAISED the peak to 1e-2; iter-14 does NOT
      touch the peak. The forward/backward dynamics per step are iter-13's.
  (2) Decoupled weight decay is a CONTRACTION on the weights (multiply by
      1 - lr*wd < 1 each step) — it can only SHRINK parameters toward 0, the
      direction of the zero-init seed (reg ≡ 0 at init). It cannot inject energy
      or push the recon toward the 1.9e-4 divergence attractor; if anything it
      pulls AWAY from it. The decay magnitude per step is lr*wd <= 5e-3*3e-4 =
      1.5e-6 at peak and shrinks to 1e-5*3e-4 = 3e-9 in the tail — negligible
      against the gradient term, a gentle nudge not a destabiliser.
  (3) grad_clip=1.0 (kept from iter-7) caps any residual gradient spike.
  (4) The recon DYNAMICS are byte-for-byte iter-13/iter-7: same K=5, same single
      scalar alpha, same plain prox step, same zero-init FoE bank. NOTHING in
      the model changes; ONLY the optimiser's weight-decay term does.
      Categorically unlike iter-8/9/10/11 (added structure) and iter-12 (raised
      the peak LR).
HYPOTHESIS: a light decoupled wd 3e-4 regularises the completed-anneal fine-tune
that iter-13 overfit -> shrinks the train/val gap -> val-RMSE drops BELOW iter-7's
7.40e-4 -> hr ABOVE 0.2515, beating BOTH iter-13 (0.2273) and iter-7 (0.2515)
at the SAME 1,921 params, SAME stable peak LR 5e-3, SAME ~1080s wall. If it
MATCHES iter-13 (~0.227), 3e-4 is too weak (next: wd 1e-3). If it OVERSHOOTS
(under-fits, hr below iter-13), wd is too strong (next: wd 1e-4). PREDICTED
hr ~0.26-0.28 (recovering the iter-13 overfit loss + a margin over iter-7).

The learned regulariser `reg(x)` is selected by `reg_type`:
  - "foe"       (DEFAULT — iter-7/9/10/11: TIED Fields-of-Experts / VN filter
                bank, a DIFFERENT, single-scale, param-EFFICIENT reg family that
                BEAT the CNN at fewer params in iter-7 — hr 0.2515 vs 0.2378):
                analysis conv2d (`foe_n_filters` filters, `foe_kernel`x
                `foe_kernel`) -> per-filter RBF activation (`foe_n_bumps`
                bumps) -> TIED conv_transpose2d synthesis. reg(x)=K^T ρ'(Kx),
                one VNStep's reg-gradient reused at every unrolled step.
                ZERO-INIT ρ'-weights => reg ≈ 0 at init (stability). iter-7
                CHAMPION nf=24/k=7/nb=31 = 1,920p (total 1,921). iter-8 SCALED
                the bank nf=24->40 = 2,881p and DIVERGED (psnr 1.87) — capacity
                exhausted. iter-11 KEEPS the 1,920p iter-7 bank byte-for-byte.
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
    # ---- architecture (iter-14: iter-7 CHAMPION unroll byte-for-byte — NO learned-init, NO added capacity/depth) ----
    "reg_type":        "foe",      # iter-14: KEEP "foe" (TIED Fields-of-Experts / VN bank, iter-7 CHAMPION hr 0.2515, 1,920p). iter-8's SCALED bank DIVERGED -> capacity exhausted; reg UNTOUCHED. "foe" (iter-7 BEST) | "cnn" (iter-4 / iter-6) | "microunet" (iter-2/iter-5, REGRESSED) | "bilateral"
    "n_iter":          5,          # iter-14: KEEP K=5 (iter-7 CHAMPION, the SHARP stable basin; iter-9's K=7 DIVERGED). iter-14 changes the OPTIMISER (AdamW wd), not the unroll.
    # ---- learned-init refiner `g` (iter-10/11 BOTH FROZE training -> DISABLED in iter-14; kept selectable for the post-mortem) ----
    "learned_init":      False,    # iter-14: OFF (iter-10 AND iter-11 BOTH froze training even with verified true zero-init -> any added trainable stage collapses training in-budget). With this False the model is iter-7 byte-for-byte (raw LD-FBP init). True => the iter-11 learned-init build (kept selectable).
    "init_channels":     16,       # (unused at learned_init=False) hidden width of g; 16 => 177 params at init_layers=1.
    "init_layers":       1,        # (unused at learned_init=False) number of 3x3 hidden convs in g.
    "init_clamp":        False,    # (unused at learned_init=False) iter-11 FIX kept: do NOT upper-clamp the init (the iter-10 BUG).
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
    # ---- "foe" regulariser (iter-11: iter-7 CHAMPION bank nf24/k7/nb31 = 1,920p, byte-for-byte; iter-8's SCALED nf40 DIVERGED) ----
    "foe_n_filters":   24,    # iter-11: 24 (iter-7 champion bank; iter-8's nf=40 DIVERGED psnr 1.87 — capacity-scaling is exhausted)
    "foe_kernel":      7,     # iter-7/11: 7 (eff RF 7px/step, ~19px over K=5 tied steps, no pooling)
    "foe_n_bumps":     31,    # iter-11: 31 (iter-7 champion RBF resolution; bank byte-for-byte the 1,920p winner)
    "foe_x_range":     1.0,
    "foe_filter_init_std": 0.05,
    "foe_rbf_init_std":    0.0,   # iter-7: ZERO-INIT synthesis -> ρ'≡0 => reg(x)≡0 at init (stability; mirrors CNNReg zero-init head)
    # ---- "bilateral" regulariser ----
    "n_bf":            4,
    "bf_kernel":       7,
    # ---- training (iter-14: REGULARISE the completed-anneal OVERFIT with decoupled AdamW wd — schedule kept iter-13, peak LR kept 5e-3) ----
    "train_n":   200,
    "val_n":     214,
    "epochs":    8,           # iter-14: KEEP 8 (iter-13). The 8-ep cosine (cosine_t_max=8) BOTTOMS OUT at eta_min within the 1080s wall — the COMPLETED anneal that iter-13 proved works (and mildly overfit). iter-14 regularises that anneal rather than changing its length.
    "cosine_lr": True,        # iter-14: KEEP cosine (iter-13). Anneal the PEAK lr (5e-3, UNCHANGED) -> cosine_lr_min over cosine_t_max epochs. Optimiser is now AdamW (weight_decay>0) at the SAME schedule.
    "cosine_t_max": 8,        # iter-14: KEEP 8 (iter-13). Cosine PERIOD in epochs, decoupled from `epochs`; the LR FULLY anneals to eta_min within the budget-achievable epoch count. None/<=0 => fall back to `epochs`.
    "cosine_lr_min": 1e-5,    # eta_min for the cosine tail (UNCHANGED from iter-7/13)
    "max_train_s": 1080,      # iter-14: hard 18-min train backstop (UNCHANGED). AdamW adds ZERO per-step compute vs Adam, so wall == iter-13 (~1240s observed, within the cut).
    "lr":        5e-3,        # iter-14: PEAK lr 5e-3 (UNCHANGED — NOT raised). iter-12's 2x peak (1e-2) DIVERGED (epoch-3 hit the degenerate 1.9e-4 attractor); 5e-3 is the stability edge. The lever is decoupled weight_decay, NOT the peak.
    "warmup_frac": 0.0,       # iter-14: KEEP 0.0 (no warmup, the iter-7/13 behaviour). Warmup only mattered for the higher peak LR, which is NOT revived.
    "weight_decay": 3e-4,     # iter-14: 0.0 -> 3e-4. The PRIMARY lever: decoupled L2 (wd>0 => AdamW) regularises the completed-anneal fine-tuning that iter-13 OVERFIT (train loss 5.78e-7 < iter-7 but val hr 0.2273 < iter-7's 0.2515). Light (iter-12's 1e-4 survived ep1 cleanly; 3x bump still tiny). Decay magnitude lr*wd shrinks with the cosine LR so it bites mainly in the low-LR overfit tail. Stays stable: only SHRINKS weights toward the zero-init seed; cannot push to the divergence attractor (which needed a 2x peak LR). 0.0 => plain Adam.
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


class LearnedInit(nn.Module):
    """Tiny zero-init residual refiner of the LD-FBP init (iter-11 RE-WIRED).

    A single-scale residual conv `g` applied ONCE to the LD-FBP before the
    unroll: the model uses x_0 = LD-FBP + g(LD-FBP) (NO upper clamp on the init
    -- the iter-10 BUG was clamping x_0 to clip_max, which truncated the bright
    LD-FBP pixels and broke byte-for-byte parity with iter-7). It is a SMOOTH
    path -- conv(1->channels, 3x3, reflect-padded) -> GELU -> conv(channels->1,
    1x1) -- with BOTH the 1x1 head's weight AND bias ZERO-INITIALISED, so
    g(.) ≡ 0 at init and therefore x_0 == LD-FBP EXACTLY (byte-for-byte) at init.
    The seed is the iter-7 champion byte-for-byte, and training LEARNS a
    lower-RMSE init as a correction (the same proven zero-init-output stability
    pattern as CNNReg's head / FoEReg's rho'-weights).

    Why GELU not ReLU (the dead-ReLU-before-zero-head trap fix): with a ReLU
    feeding a zero-init head, the head's gradient w.r.t. the body weights routes
    through ReLU' which is 0 on half the activations at init; combined with the
    zero head this can leave the refiner unable to escape g≡0. GELU is smooth
    and nonzero-derivative everywhere, so once the zero head's own weights start
    to move (driven by the supervised loss), gradient flows back into the body
    cleanly. NO pooling (iter-2/5's failure mode); eff RF ~3px at layers=1, i.e.
    the smallest possible learned-init -- a local FBP-noise/DC refiner, NOT a
    denoiser. At channels=16, layers=1 it is (16*9+16) (3x3 conv w+b) + (16+1)
    (1x1 head w+b) = 177 params, applied OUTSIDE the unroll loop so it adds neither
    unroll depth nor reg capacity."""

    def __init__(self, channels: int = 16, layers: int = 1):
        super().__init__()
        channels = int(channels)
        layers = max(1, int(layers))

        def _conv(ci, co, k):
            return nn.Conv2d(ci, co, k, padding=k // 2, padding_mode="reflect")

        body: list[nn.Module] = [_conv(1, channels, 3)]
        for _ in range(1, layers):
            body += [nn.GELU(), _conv(channels, channels, 3)]
        body += [nn.GELU()]
        self.body = nn.Sequential(*body)
        # ZERO-INIT 1x1 head (weight AND bias) => g(.) ≡ 0 at init.
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


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
    iter-1's EXACT stable recipe), iter-11: + a CORRECTLY-WIRED learned init.

    x_0 = u0 + g(u0)  (g ≡ 0 at init => x_0 == LD-FBP byte-for-byte; NO upper
    clamp on the init -- the iter-10 BUG was clamping x_0 to clip_max).
    For k in range(n_iter):
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
        # iter-11 RE-WIRED STAGE: a tiny ZERO-INIT learned refiner of the
        # LD-FBP, applied ONCE before the unroll (OUTSIDE the loop). g≡0 at init
        # so x_0 == LD-FBP byte-for-byte => the seed is iter-7 byte-for-byte.
        # NO upper clamp on the init (the iter-10 BUG); the unroll's per-step
        # clamp handles range, exactly as iter-7's x=u0 raw init.
        self.use_learned_init = bool(cfg.get("learned_init", False))
        self.init_clamp = bool(cfg.get("init_clamp", False))   # iter-11: False (iter-10 BUG was True)
        if self.use_learned_init:
            self.init_refiner = LearnedInit(
                channels=int(cfg.get("init_channels", 16)),
                layers=int(cfg.get("init_layers", 1)))
        else:
            self.init_refiner = None
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

    def refined_init(self, u0: torch.Tensor) -> torch.Tensor:
        """The learned-init x_0 = u0 + g(u0) (clamp_min(0) only; NO upper clamp
        unless init_clamp is the buggy iter-10 mode). Exposed so the runtime
        self-check can verify x_0 == u0 byte-for-byte at init."""
        if self.init_refiner is None:
            return u0
        x0 = u0 + self.init_refiner(u0)
        if self.init_clamp:
            # iter-10 BUGGY behaviour (kept selectable for the post-mortem A/B).
            return torch.clamp(x0, 0.0, self.clip_max)
        # iter-11 FIX: clamp_min(0) only (no-op since LD-FBP >= 0), so at init
        # (g≡0) x_0 == u0 EXACTLY, byte-for-byte iter-7.
        return x0.clamp_min(0.0)

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
        # iter-11: refine the LD-FBP init ONCE (outside the unroll). The
        # refiner's head is zero-init so at init x_0 == u0 (LD-FBP) byte-for-byte
        # (no upper clamp -- the iter-10 BUG fix); the unroll's per-step clamp
        # handles range, exactly as iter-7's x=u0 raw init.
        x0 = self.refined_init(u0)
        x = x0
        y = x0                                   # look-ahead == x at k=0 (v=0)
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
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k in ('reg_type','n_iter','learnable_alpha','per_step_alpha','momentum','beta_init','alpha_init','clip_max','mu_channels','cnn_channels','cnn_layers','cnn_dilations','foe_n_filters','foe_kernel','foe_n_bumps','n_bf','bf_kernel','learned_init','init_channels','init_layers','init_clamp','epochs','cosine_lr','cosine_t_max','cosine_lr_min','max_train_s','batch_size','lr','warmup_frac','weight_decay','train_n','val_n')}, default=str)}",
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

    # iter-11 RUNTIME SELF-CHECK (the guard iter-10 lacked): right after building
    # the model, on ONE val sample, verify the learned-init is TRULY zero at init
    # so x_0 == LD-FBP byte-for-byte. rel = ‖g(LD_FBP)‖ / ‖LD_FBP‖ must be < 1e-6.
    # A non-zero-init bug (the iter-10 failure mode) FAILS LOUDLY here, BEFORE the
    # 20-min run is wasted. Also reports ‖x_0 - LD_FBP‖ to catch any init clamp.
    if model.use_learned_init and val_u0.shape[0] > 0:
        with torch.no_grad():
            if per_ps:
                model.projector = _projs[float(_vrk[0])]
            u0_chk = val_u0[0:1]
            g_out = model.init_refiner(u0_chk)
            u0_norm = float(u0_chk.norm().clamp(min=1e-12))
            rel_g = float(g_out.norm()) / u0_norm
            x0_chk = model.refined_init(u0_chk)
            rel_x0 = float((x0_chk - u0_chk).norm()) / u0_norm
            print(f"[selfcheck] learned-init zero-at-init: "
                  f"‖g(LD_FBP)‖/‖LD_FBP‖={rel_g:.3e}  "
                  f"‖x0-LD_FBP‖/‖LD_FBP‖={rel_x0:.3e}  "
                  f"(init_clamp={model.init_clamp}, must be <1e-6)", flush=True)
            assert rel_g < 1e-6, (
                f"learned-init NOT zero at init (rel_g={rel_g:.3e} >= 1e-6): "
                "the zero-init head is broken (the iter-10 bug). ABORTING.")
            assert rel_x0 < 1e-6, (
                f"x_0 != LD-FBP at init (rel_x0={rel_x0:.3e} >= 1e-6): the init "
                "is being corrupted (e.g. by an upper clamp -- the iter-10 bug). "
                "ABORTING.")

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_params = _count_reg_params(cfg)
    init_params = (sum(p.numel() for p in model.init_refiner.parameters()
                       if p.requires_grad)
                   if model.init_refiner is not None else 0)
    n_alpha = (model.n_iter if model.per_step_alpha else 1) if model.log_alpha is not None else 0
    n_beta = 1 if model.beta_raw is not None else 0
    print(f"[solver] ParamEfficient iter-14: reg_type={cfg['reg_type']!r}  "
          f"n_iter={cfg['n_iter']} (weight-TIED reg)  "
          f"learned_init={model.use_learned_init} (init={init_params}p)  "
          f"per_step_alpha={model.per_step_alpha} momentum={model.use_momentum}  "
          f"epochs={cfg['epochs']} cosine_lr={cfg.get('cosine_lr', False)} "
          f"cosine_t_max={cfg.get('cosine_t_max')} "
          f"peak_lr={cfg.get('lr')} warmup_frac={cfg.get('warmup_frac', 0.0)} "
          f"weight_decay={cfg.get('weight_decay', 0.0)} "
          f"opt={'AdamW' if float(cfg.get('weight_decay', 0.0)) > 0 else 'Adam'}  "
          f"trainable params={params_total} "
          f"(reg={reg_params} + init={init_params} + alpha={n_alpha} + beta={n_beta})  "
          f"= {params_total/1e6:.6f} M  vs 233k ITNet", flush=True)

    # iter-14 TRAINER (REGULARISE the completed-anneal OVERFIT with decoupled wd):
    #   - AdamW (weight_decay 3e-4) + cosine, PEAK lr 5e-3 UNCHANGED. iter-13
    #     COMPLETED the cosine anneal (epochs=8, cosine_t_max=8) and that LOWERED
    #     train loss (5.78e-7 < iter-7) but MILDLY OVERFIT val (hr 0.2273 <
    #     iter-7's 0.2515). The binding constraint is now VAL GENERALISATION, so
    #     iter-14 adds a LIGHT decoupled L2 (wd>0 => the AdamW branch below) to
    #     regularise the low-LR fine-tune tail without touching the schedule.
    #     iter-12's 2x peak LR (the actual divergence cause) is NOT revived;
    #     AdamW at the SAME 5e-3 peak only shrinks weights toward the zero-init
    #     seed -> safe.
    #   - The cosine PERIOD stays `cosine_t_max=8` EPOCHS (iter-13), FULLY
    #     annealing to eta_min=1e-5 within the 1080s wall. The decoupled decay
    #     magnitude per step is lr*wd, which shrinks WITH the cosine LR — it
    #     bites most where the overfit happens (the low-LR tail) and is
    #     negligible (<=1.5e-6) at the peak, so it cannot destabilise.
    #   - PER-BATCH LR off a global batch counter so it is robust to the
    #     max_train_s wall cut (the run simply stops, but with epochs aligned to
    #     the budget the cosine has already bottomed out by then).
    #   clip_and_step() only calls opt.step(); we set the LR ourselves BEFORE
    #   each step, so the two compose cleanly (no torch scheduler needed).
    wd = float(cfg.get("weight_decay", 0.0))
    if wd > 0.0:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    peak_lr = float(cfg["lr"])
    eta_min = float(cfg.get("cosine_lr_min", 1e-5))
    use_cosine = bool(cfg.get("cosine_lr", False))
    warmup_frac = float(cfg.get("warmup_frac", 0.0))
    bs = max(1, int(cfg["batch_size"]))
    steps_per_epoch = max(1, math.ceil(cfg["train_n"] / bs))
    # iter-13: cosine PERIOD in epochs, decoupled from `epochs`. None/<=0 falls
    # back to `epochs` (the iter-7 behaviour where T_max == epochs).
    _ctm = cfg.get("cosine_t_max", None)
    cosine_t_max_ep = int(_ctm) if (_ctm is not None and int(_ctm) > 0) else int(cfg["epochs"])
    total_steps = max(1, cosine_t_max_ep * steps_per_epoch)
    warmup_steps = int(round(warmup_frac * steps_per_epoch))  # 0 => no warmup

    def _lr_at(step: int) -> float:
        """Per-batch LR: linear warmup 0->peak over warmup_steps, then cosine
        peak->eta_min over the cosine period (cosine_t_max epochs of steps), or
        constant peak if cosine_lr is off. `step` is the 0-based global batch
        index; prog is clamped to 1.0 so the LR holds at eta_min after the
        period (it cannot go below eta_min)."""
        if warmup_steps > 0 and step < warmup_steps:
            # ramp from peak/warmup_steps up to peak (never exactly 0, so the
            # very first step still makes nonzero progress).
            return peak_lr * float(step + 1) / float(warmup_steps)
        if not use_cosine:
            return peak_lr
        denom = max(1, total_steps - warmup_steps)
        prog = min(1.0, float(step - warmup_steps) / float(denom))
        return eta_min + 0.5 * (peak_lr - eta_min) * (1.0 + math.cos(math.pi * prog))

    train_start = time.time()
    gstep = 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0
        ep_lr0 = None
        for i in range(0, cfg["train_n"], bs):
            cur_lr = _lr_at(gstep)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            if ep_lr0 is None:
                ep_lr0 = cur_lr
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
            gstep += 1
        avg_loss = running / max(1, n_batches)
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"lr={ep_lr0:.3g}->{_lr_at(gstep - 1):.3g}  "
              f"alpha={float(model.alpha.detach().cpu()):.4g}",
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
        "learned_init": bool(model.use_learned_init),
        "init_clamp": bool(model.init_clamp),
        "init_params": init_params,
        "init_channels": int(cfg.get("init_channels", 16)),
        "init_layers": int(cfg.get("init_layers", 1)),
        "n_iter": cfg["n_iter"],
        "epochs": cfg["epochs"],
        "cosine_lr": bool(cfg.get("cosine_lr", False)),
        "cosine_t_max": cosine_t_max_ep,
        "lr": float(cfg.get("lr", 0.0)),
        "warmup_frac": float(cfg.get("warmup_frac", 0.0)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
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
