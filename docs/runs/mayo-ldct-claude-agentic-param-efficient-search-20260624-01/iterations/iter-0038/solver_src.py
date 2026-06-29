"""Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-32).

iter-32 PUSHES THE DOMINANT LEVER (at display_max=0.09, recon mu>=0 only). The 0.09
CHAMPION is iter-31 = FoE + a GENTLE Manduca PROJECTION filter (proj_n_bf=1, single
zero-init gain on the Anscombe sqrt-count) + an anisotropic ORIENTED image filter, val
hr 0.3968 @ 1,929 params. The recombination map at 0.09: FoE+bilateral 0.346;
FoE+Manduca 0.394 (the Manduca PROJECTION filter is the DOMINANT complementary bias,
+0.048 over bilateral); +anisotropic 0.3968; an ISOTROPIC image bilateral on top of
Manduca is REDUNDANT (iter-30 regressed to 0.381). iter-32 extends the DOMINANT lever:
the Manduca PROJECTION filter currently has ONE spatial scale -> a MULTI-SCALE PARALLEL
projection bank (proj_multiscale=True, proj_n_bf=3, per-filter init σ {0.83,1.66,3.32}
px ×2 octave ladder), each parallel projection bilateral with its OWN ZERO-INIT gain:
    sino_out = P + Σ_{i=1..M} gain_i * ( P̂_i - P ),  P̂_i = -ln(BF_i(sqrt(N0·e^-P))²/N0)
FoE + anisotropic + the single existing Manduca scale all stay intact; the extra
projection scale(s) are ADDED with zero-init gain so the seed == iter-31 byte-for-byte
(all proj gains=0 ⇒ ‖sino_filtered − sino‖=0 ⇒ pipeline == iter-31; worst case == 0.3968,
no regression). RESPECT GENTLENESS (Maier): each added proj scale is gentle (zero-init
gain, soft range σ); too MUCH projection denoising -> over-smooth, too-HARD edge
preservation -> view inconsistency. The learned per-scale gain VECTOR is the experiment.
DENSE-VIEW CAVEAT (honest EV): at 2,304 views the FBP already averages much projection
noise, so projection-domain headroom is limited -- a small gain or a clean null, either
way it maps the Manduca lever's ceiling. EXACT PARAM COUNT: FoE 1920 + alpha 1 + aniso 3
+ proj(M=3: 3*3 σ + 1 log_N0 + 3 gains = 13) = 1,937 (+8 over iter-31, <<233k).
PREDICTED hr ~0.397-0.41. DEAD ENDS (do NOT reintroduce): steerable/equivariant FoE
reparam (diverged); pooling/U-Net; capacity-up FoE width; K>5; momentum; denoiser-
pretrain; peak LR>5e-3; constant LR; full anneal to 1e-5; recon upper clamp.


iter-27 (the prior IMAGE-domain lever) docstring below, preserved for lineage:
Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-27).

iter-27 SCALES THE iter-25 BREAKTHROUGH: iter-25 became the NEW CHAMPION (val hr
0.2930, ssim 0.923, psnr 37.1 @ 1,925 params, beating iter-7's 0.2515 by +0.041)
by adding ONE image-domain bilateral edge-preserving term IN PARALLEL with the
iter-7 FoE bank -- reg(x) = FoE(x) + gain*(x - BF(x)), n_bf=1, ZERO-init gain that
TRAINED TO A NON-ZERO VALUE. So a complementary bilateral term in parallel with the
FoE convs genuinely helps. iter-27 SCALES that winning branch to a MULTI-RF
PARALLEL BANK (the user's "multi-scale + parallel bilateral", done WITHOUT pooling):
    reg(x) = FoE(x) + Σ_{i=1..M} gain_i * ( x - BF_i(x) )
with M=3 bilateral kernels at DIFFERENT spatial sigmas (small / mid / large) so the
bank captures edge-preserving structure at MULTIPLE SCALES in parallel -- the
multi-scale receptive field the U-Net was meant to provide, but WITHOUT the pooling
that diverged in iters 2/5/6. Each bilateral keeps its OWN learnable 3 sigmas + its
OWN ZERO-INIT gain, so the bank starts == iter-7's clean seed and training engages
whichever scales help.

LESSON carried from iters 21-26 (extension failures except iter-25): only changes
that START byte-for-byte iter-7 (the added term contributes EXACTLY ZERO at init)
stay in iter-7's stable basin. iter-27 honours this: every gain_i is ZERO-init, so
Σ_i gain_i*(x-BF_i(x)) = 0 at init REGARDLESS of the σ_i, and with zero-init FoE RBF
reg(x) == 0 == iter-7's clean GD+DC seed BYTE-FOR-BYTE. iter-25 already PROVED this
pattern lifts off zero productively (its single gain trained non-zero -> +0.041 hr).

iter-27 SIX-BOX (NUMBERS) -- multi-RF parallel bilateral bank ON TOP of iter-7 FoE
----------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = psnr 34.08 dB):
  - iter-7 (FoE-only BASE): FoE nf24/k7/nb31 = 24*49 + 24*31 = 1,176 + 744 = 1,920
    reg + 1 scalar alpha = 1,921 params. K=5 tied prox+DC, single-phase PARTIAL
    cosine peak 5e-3 -> val hr 0.2515, ssim 0.9058, psnr 36.59, val_rmse 7.40e-4.
  - iter-8..16: capacity/depth/stage/LR/kernel-geometry all mapped -> 0.2515 was
    the FoE-ONLY ceiling; nf24/k7 is the FoE geometry optimum (iter-16 REGRESSED).
  - iter-17 (bilateral ALONE image-domain, ~17p): hr ~0 BELOW the FBP floor -- a
    bilateral prior REPLACING the FoE is too weak. iter-25/27 put it ON TOP.
  - iter-18/19: HALVING the bank (nf 12/6) walked the frontier DOWN (capacity helps).
  - iter-21/22 (denoiser-pretrain): FROZE x2, hr 0 -> the reg only learns COUPLED.
  - iter-23 (ordered-subsets DC): NULL 0.2401 -> view-count not the training wall.
  - iter-24 (steerable/equivariant FoE): DIVERGED (non-identity analysis reparam).
  - iter-25 (NEW CHAMPION, image-domain SINGLE bilateral ON TOP of FoE, +4p,
    zero-init gain): val hr 0.2930, ssim 0.9227, psnr 37.09, val_rmse 6.99e-4 @
    1,925 params. The zero-init gain TRAINED NON-ZERO => a complementary edge term
    in parallel with the FoE genuinely helps. +0.041 hr over iter-7.
  - iter-26 (PROJECTION-domain Manduca filter): dense-view ceiling (2,304 views ->
    the FBP already averages the projection noise; proj-domain headroom is for
    SPARSE-view, not dense). The image domain is the live lever; iter-25 proved it.
FAILURE MODE addressed (iter-27): iter-25 proved ONE parallel bilateral at a SINGLE
  fixed init σ=(1.5,1.5) helps, but its receptive field is single-scale -- one
  spatial bandwidth cannot simultaneously preserve fine edges AND smooth broad
  low-frequency noise. The campaign's multi-scale attempts ALL used POOLING
  (micro-UNet iters 2/5/6) and REGRESSED (pooling caps psnr 32.4). iter-27 delivers
  the missing multi-scale receptive field WITHOUT pooling: M parallel bilaterals at
  DISTINCT spatial σ, each a zero-init-gain residual, so the bank is multi-RF yet
  stays in iter-7's basin.
CHANGE (iter-27, ONE knob -- scale iter-25's bilateral branch from M=1 to a M=3
  MULTI-SCALE PARALLEL BANK; everything else iter-25/iter-7 byte-for-byte): config
  flag `bf_multiscale=True` + per-filter init sigmas `bf_sigmas`. ComposedFoEBilateralReg
  becomes a PER-FILTER-GAIN, PER-FILTER-σ bank:
    reg(x) = FoE(x) + Σ_{i=1..M} gain_i * ( x - BF_i(x) )
  - FoE(x): the iter-7 bank EXACTLY (nf24/k7/nb31, zero-init RBF => FoE(x)=0 at
    init), 1,920 params, fully shared/tied across the K=5 unrolled steps.
  - BF_i: M=3 TrainableBilateralFilter2d (Wagner et al. Med. Phys. 2022,
    ddssl_ldct/models.py), each with 3 learnable σ (log sx, log sy, log sr) at a
    DISTINCT spatial init: σ_x,y ∈ {0.8, 1.6, 3.2} px (geometric ×2 ladder: small
    fine-edge / mid / large broad-noise scales). Range σ_r = 0.02 μ (iter-25's) for
    all three -- the spatial bandwidth is what scales, not the intensity edge gate.
  - gain_i: M=3 INDEPENDENT learnable scalars, each ZERO-INIT (bf_gain_init=0.0) =>
    Σ_i gain_i*(x-BF_i(x)) = 0 at init REGARDLESS of the σ_i. Per-filter (NOT a
    single shared scalar as in iter-25) so each scale engages independently; the
    learned gain VECTOR is the answer (which scales the optimiser kept).
SPEND CHOICE -- M=3 bilateral kernels (bf_multiscale, bf_sigmas=[0.8,1.6,3.2]): the
  MINIMAL multi-scale bank that brackets fine/mid/coarse with a clean ×2 octave
  ladder. M=1 is iter-25 (single-scale, proven +0.041); M=3 tests whether MORE
  spatial scales add headroom. Each BF_i = 3 σ + 1 gain = 4 params => +8 over
  iter-25's single bilateral (+4 over its already-counted 4).
EXACT PARAM COUNT (iter-27 vs iter-7 / iter-25):
  FoE analysis 24*7*7 = 1,176  +  FoE RBF 24*31 = 744                = 1,920 (iter-7)
  + 1 scalar alpha                                                  =     1
  + M bilaterals × (3 σ + 1 gain), M=3                              =    12 (NEW)
  = 1,933 TOTAL  (iter-7 was 1,921; iter-25 was 1,925; +12 over iter-7, +8 over
  iter-25, +0.42% over iter-7). CONFIRMED == prompt's 1,921 + 4M = 1,933 at M=3.
VERIFY-AT-INIT (mandatory): a runtime self-check builds the model and on ONE val
  sample asserts EVERY gain_i==0.0 AND ‖reg(x) - FoE(x)‖ == 0 EXACTLY (the whole
  bilateral bank is 0 at init REGARDLESS of the σ_i). With zero-init RBF FoE(x)==0
  too, so reg(x)==0 == iter-7's seed byte-for-byte. The check PRINTS the gain
  vector, ‖FoE(x)‖, ‖bank‖, ‖reg-FoE‖ and ABORTS LOUDLY before the 20-min run if
  any gain is non-zero or the bank is not exactly 0.
STABILITY (why the multi-RF parallel bank stays in the iter-7 basin):
  (1) reg(x) == FoE(x) == iter-7's reg EXACTLY at init (all gain_i=0 AND zero-init
      RBF). The seed IS iter-7's clean GD+DC scheme byte-for-byte. The bilateral
      bank is a STRICTLY ADDITIVE correction lifted off zero only by the supervised
      gradient -- the SAME zero-init-OUTPUT pattern iter-25 already proved engages
      productively (its gain trained non-zero -> +0.041 hr).
  (2) NO pooling (dodges the micro-UNet iters 2/5/6 that capped psnr 32.4 -- the
      multi-scale RF is delivered by parallel bilaterals at distinct σ, NOT a
      down/up sample), NO FoE capacity scale-up (nf24/k7 unchanged -- dodges
      iter-8's nf40 divergence), NO analysis reparam (dodges iter-24's steerable
      divergence), NO extra unroll stage/depth (dodges iter-9/10/11), K=5 single
      tied scalar alpha, NO momentum/per-step-alpha (dodges iter-3).
  (3) TRAINER is iter-7/iter-25 EXACTLY: plain Adam (NO weight-decay), peak lr 5e-3
      (NOT iter-12's diverged 1e-2), PARTIAL cosine T_max=16 (NOT iter-13/14's full
      anneal that overfit, NOT iter-15's diverged constant LR), grad_clip=1.0,
      bs=1, per-sample-ps, full-view DC, max_train_s=1080, train_n=200, val_n=214.
HYPOTHESIS: MULTIPLE spatial scales of bilateral add MORE headroom than iter-25's
  single bilateral -> val hr > 0.2930. A NULL (only one gain stays non-zero,
  hr ≈ 0.2930) cleanly says ONE scale suffices (the FoE + one bilateral already
  spans the useful edge structure). SAFE: all gains zero-init => worst case ≈
  iter-25/iter-7 -- NO divergence/regression risk. PREDICTED hr ~0.293-0.31.
  REPORT the per-filter learned gain VECTOR (which scales engaged) + the per-filter
  learned σ.
DEAD ENDS (do NOT reintroduce): pooling/U-Net multi-scale (diverged iters 2/5/6);
  denoiser-pretrain (froze x2); ordered-subsets (null); steerable/equivariant
  reparam (diverged); capacity↑ via FoE width; K>5; momentum/per-step-alpha; peak
  LR>5e-3; constant LR; full anneal to 1e-5; bilateral-ALONE (no FoE, below floor).

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
    # ---- architecture (iter-27: iter-25 CHAMPION FoE unroll + a MULTI-RF PARALLEL bilateral bank ON TOP, M=3 scales, each zero-init gain) ----
    "reg_type":        "foe",      # iter-27: KEEP "foe" (iter-7 CHAMPION bank). The composed multi-RF reg is layered on via bilateral_on_top + bf_multiscale below. "foe" (iter-7 CHAMPION 1921p, hr 0.2515) | "cnn" (iter-4 / iter-6) | "microunet" (iter-2/iter-5, REGRESSED) | "bilateral" (iter-17 ALONE, below floor, do NOT use alone)
    "bilateral_on_top": True,      # iter-27 THE LEVER (KEPT ON from iter-25 CHAMPION): wrap the FoE in reg(x) = FoE(x) + Σ_i gain_i*(x - BF_i(x)). All gain_i ZERO-INIT => bilateral bank = 0 at init => reg == FoE == iter-7 byte-for-byte. False => plain iter-7 FoE.
    "proj_filter_on":  False,      # iter-32: the CHAMPION (iter-31) sets this True via cfg_full. iter-26's PROJECTION-domain Manduca filter (the DOMINANT complementary bias at 0.09: FoE+Manduca 0.394). True => the Manduca proj filter is PREPENDED to the unroll.
    # ---- iter-32: multi-scale PARALLEL projection bank (extends ManducaProjFilter) ----
    "proj_n_bf":       1,          # iter-32 THE LEVER: number of projection bilaterals. iter-26/31 used 1 (single-scale). 3 = a small/mid/large multi-scale PARALLEL projection bank (each filter on the SAME sqrt-count Q, its OWN zero-init gain). Only multi-scale when proj_multiscale=True.
    "proj_multiscale": False,      # iter-32 THE SCALING LEVER: True => the M proj bilaterals get DISTINCT per-filter init sigmas (proj_sigmas) + PER-FILTER zero-init gains, run IN PARALLEL on Q => out = P + Σ_i gain_i*(P̂_i - P). False => iter-26/31 behaviour (sequential cascade of M filters + ONE shared scalar gain). The CHAMPION (iter-31) is multiscale=False, proj_n_bf=1.
    "proj_sigmas":     [0.83, 1.66, 3.32],  # iter-32: per-filter SPATIAL σ init (sqrt-count px), small/mid/large (geometric ×2 octave ladder; first == iter-31's 0.83). Length must == proj_n_bf. Each filter's σ_x AND σ_y init to this; range σ_r = proj_sigma_r for all. Learnable per-filter. Only used when proj_multiscale=True.
    "n_iter":          5,          # unrolled prox-gradient steps, weight-TIED (iter-7 CHAMPION K; K>6 + momentum was UNSTABLE in iter-3, K=7 DIVERGED in iter-9)
    "learnable_alpha": True,       # ONE tied alpha = softplus(param); init from alpha_init
    "per_step_alpha":  False,      # iter-4: REVERT iter-3's per-step alpha (it destabilised) -> single shared scalar (iter-1)
    "momentum":        False,      # iter-4: REVERT iter-3's Nesterov momentum (it destabilised) -> plain prox step (iter-1)
    "beta_init":       0.5,        # unused when momentum=False (kept for backward-compat selectability)
    "alpha_init":      0.1,        # step size (O(1) thanks to dc_norm scaling)
    "clip_max":        0.05,       # UNUSED since 2026-06-29: in-loop upper clamp removed (it truncated bone, true mu up to 0.0814); recon is mu>=0 only now
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
    # ---- "bilateral" residual bank (iter-27: MULTI-RF PARALLEL, ON TOP of the FoE via bilateral_on_top; per-filter ZERO-INIT gain) ----
    "n_bf":            3,         # iter-27: M=3 bilateral filters IN PARALLEL ON TOP of the FoE. reg += Σ_i gain_i*(x - BF_i(x)). iter-25 used 1 (single-scale, hr 0.2930). 3 = small/mid/large multi-scale bank (3 sigmas + 1 gain per filter = 4 params/filter).
    "bf_multiscale":   True,      # iter-27 THE SCALING LEVER: True => the M bilaterals get DISTINCT per-filter init sigmas (bf_sigmas) + PER-FILTER zero-init gains (a multi-RF bank). False => iter-25 behaviour (all filters share bf_sigma_x/y + ONE shared scalar gain).
    "bf_sigmas":       [0.8, 1.6, 3.2],  # iter-27: per-filter SPATIAL σ init (px), small/mid/large (geometric ×2 octave ladder). Length must == n_bf (3). Each filter's σ_x AND σ_y init to this value; range σ_r = bf_sigma_r for all. Learnable per-filter via log_sx/log_sy.
    "bf_kernel":       7,         # iter-27: 7x7 bilateral window for ALL filters (contains σ up to ~3.2 at the edges; eff RF 7px ~ matches iter-7's k7 FoE; spatial weights computed explicitly so only the 3 sigmas are learnable, NO trainable kernel weights).
    "bf_sigma_x":      1.5,       # iter-25/single-scale fallback: spatial-x bandwidth init (px) when bf_multiscale=False. Learnable via log_sx. (Multi-scale uses bf_sigmas instead.)
    "bf_sigma_y":      1.5,       # iter-25/single-scale fallback: spatial-y bandwidth init (px) when bf_multiscale=False. Learnable via log_sy.
    "bf_sigma_r":      0.02,      # iter-25/iter-27: range (intensity) bandwidth init in mu units (clip_max=0.05), SHARED across all M filters. Learnable per-filter via log_sr; controls edge preservation.
    "bf_gain_init":    0.0,       # iter-27 THE STABILITY FIX: per-filter ZERO-INIT gains => the WHOLE bilateral bank = 0 at init => reg == FoE == iter-7 byte-for-byte. Training lifts each gain off 0 only if its scale lowers val-RMSE.
    # ---- "anisotropic" oriented term (iter-31: ON TOP of FoE(+Manduca) via anisotropic_on_top; direction-aware streak cleanup; zero-init gain) ----
    "anisotropic_on_top": False,  # iter-31 THE LEVER: when True wrap the FoE in reg(x) = FoE(x) + gain*Σ_θ(x - G_θ(x)), G_θ an oriented anisotropic Gaussian blur at fixed orientations θ. gain ZERO-INIT => term=0 at init => reg==FoE byte-for-byte. Composed on iter-28 (FoE+Manduca) => pipeline==iter-28 at init. False => no oriented term.
    "aniso_n_orient":   4,        # iter-31: M FIXED orientations on [0,pi): {0,45,90,135}deg (NOT learnable -> a plain oriented filter bank, NOT the iter-24 steerable reparam that DIVERGED). 4 brackets the principal streak directions.
    "aniso_kernel":     7,        # iter-31: 7x7 oriented-Gaussian window (eff RF 7px ~ matches iter-7's k7 FoE; weights analytic from the 2 sigmas + angle, NO trainable kernel weights).
    "aniso_sigma_along": 2.0,     # iter-31: blur extent ALONG the streak (init large) -> smooth along the oriented streak. Learnable via log_s_along, SHARED across orientations.
    "aniso_sigma_across": 0.6,    # iter-31: blur extent ACROSS the streak (init small) -> preserve structure across the edge. Learnable via log_s_across, SHARED across orientations.
    "aniso_gain_init":  0.0,      # iter-31 THE STABILITY FIX: ZERO-INIT scalar gain => the WHOLE oriented term = 0 at init REGARDLESS of the sigmas/orientations => reg==FoE => pipeline==iter-28 byte-for-byte. Training lifts it off 0 only if direction-aware cleanup lowers val-RMSE.
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
    """iter-27 DEFAULT — composed reg = FoE + a MULTI-RF PARALLEL bilateral bank.

        reg(x) = FoE(x)  +  Σ_{i=1..M} gain_i * ( x - BF_i(x) )

    The iter-7 CHAMPION FoE bank (zero-init RBF => FoE(x)=0 at init) PLUS M
    edge-preserving bilateral RESIDUAL terms IN PARALLEL, each at a DISTINCT
    spatial scale and each scaled by its OWN learnable scalar `gain_i` that is
    ZERO-INIT. Every term is EXACTLY 0 at init, so reg(x)==0 == iter-7's clean
    GD+DC seed byte-for-byte. The optimiser lifts each gain_i off zero ONLY if
    that scale's complementary edge term lowers val-RMSE.

    TWO MODES (gated by `multiscale`):
      * multiscale=True (iter-27 DEFAULT): each BF_i gets a DISTINCT spatial-σ
        init from `sigmas` (small/mid/large, e.g. [0.8,1.6,3.2] px) and its OWN
        zero-init gain -> a multi-RF parallel bank (the U-Net's multi-scale RF
        WITHOUT pooling). The learned gain VECTOR is the answer (which scales the
        optimiser kept). Params: FoE (1,920) + M*(3 σ + 1 gain).
      * multiscale=False (iter-25 CHAMPION behaviour): all M filters share the
        single (sigma_x, sigma_y) init and ONE shared scalar gain. At n_bf=1 this
        is iter-25 EXACTLY (1,920 FoE + 3 σ + 1 gain). Kept selectable.

    iter-25 (M=1, single-scale, ONE gain) was the NEW CHAMPION (hr 0.2930, the
    zero-init gain trained NON-ZERO => a parallel edge term helps). iter-27
    (M=3, multi-scale, per-filter gains) tests whether MULTIPLE scales add more
    headroom. At M=3 multiscale: 1,920 + 3*4 = 1,932 (total 1,933 incl. the 1
    scalar alpha; iter-7 was 1,921, +12; iter-25 was 1,925, +8)."""

    def __init__(self, foe: nn.Module, n_bf: int = 1, kernel_size: int = 7,
                 sigma_x: float = 1.5, sigma_y: float = 1.5,
                 sigma_r: float = 0.02, gain_init: float = 0.0,
                 multiscale: bool = False, sigmas=None):
        super().__init__()
        self.foe = foe                          # the iter-7 FoE bank (zero-init RBF)
        n_bf = max(1, int(n_bf))
        self.multiscale = bool(multiscale)
        if self.multiscale:
            # iter-27: per-filter DISTINCT spatial-σ init from `sigmas`; each
            # filter's σ_x AND σ_y init to that value; range σ_r shared. If the
            # list is short/None it falls back to sigma_x/sigma_y for the rest.
            seq = list(sigmas) if sigmas else [sigma_x]
            per = [float(seq[i]) if i < len(seq) else float(sigma_x)
                   for i in range(n_bf)]
            self.filters = nn.ModuleList(
                [TrainableBilateralFilter2d(kernel_size=int(kernel_size),
                                            sigma_x=per[i], sigma_y=per[i],
                                            sigma_r=float(sigma_r))
                 for i in range(n_bf)])
            self.sigmas_init = per
        else:
            # iter-25: all filters share one (sigma_x, sigma_y) init.
            self.filters = nn.ModuleList(
                [TrainableBilateralFilter2d(kernel_size=int(kernel_size),
                                            sigma_x=float(sigma_x),
                                            sigma_y=float(sigma_y),
                                            sigma_r=float(sigma_r))
                 for _ in range(n_bf)])
            self.sigmas_init = [float(sigma_x)] * n_bf
        # PER-FILTER gains (multiscale) or ONE shared scalar (iter-25). Either
        # way ALL ZERO-INIT => the bilateral bank = 0 at init REGARDLESS of the σ.
        if self.multiscale:
            self.gains = nn.Parameter(torch.full((n_bf,), float(gain_init)))
            self.gain = None
        else:
            self.gain = nn.Parameter(torch.tensor(float(gain_init)))
            self.gains = None

    def bilateral_term(self, x: torch.Tensor) -> torch.Tensor:
        """The (un-gained) summed bilateral residual Σ_i (x - BF_i(x)).
        Exposed for the runtime zero-at-init self-check (iter-25 single-gain path)."""
        res = torch.zeros_like(x)
        for f in self.filters:
            res = res + (x - f(x))
        return res

    def gain_values(self):
        """The learned gain(s) as a python list (per-filter in multiscale)."""
        if self.gains is not None:
            return [float(g) for g in self.gains.detach().cpu()]
        return [float(self.gain.detach().cpu())]

    def sigmas_learned(self):
        """Per-filter learned (σ_x, σ_y, σ_r)."""
        return [(float(torch.exp(f.log_sx).detach().cpu()),
                 float(torch.exp(f.log_sy).detach().cpu()),
                 float(torch.exp(f.log_sr).detach().cpu()))
                for f in self.filters]

    def added_term(self, x: torch.Tensor) -> torch.Tensor:
        """The FULL gained bilateral bank Σ_i gain_i*(x - BF_i(x)) (the actual
        term added to FoE). Exposed for the runtime zero-at-init self-check so it
        is correct in BOTH the per-filter-gain and shared-gain modes."""
        if self.gains is not None:                  # multiscale: per-filter gains
            out = torch.zeros_like(x)
            for i, f in enumerate(self.filters):
                out = out + self.gains[i] * (x - f(x))
            return out
        return self.gain * self.bilateral_term(x)   # iter-25: one shared gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.foe(x) + self.added_term(x)


class ManducaProjFilter(nn.Module):
    """iter-32 — zero-init noise-adaptive PROJECTION-domain Manduca filter, now a
    MULTI-SCALE PARALLEL bank (iter-26 single-scale is the multiscale=False case).

    A learnable RESIDUAL projection-domain denoiser (Manduca et al., Med. Phys.
    2009) PREPENDED to the image-domain unroll. On a log-sinogram / line-integral
    tensor P (B,1,A,D):

        N  = N0 * exp(-P)            photon counts (N0 = blank-scan flux, learnable)
        Q  = sqrt(N)                 Anscombe/sqrt VARIANCE STABILISATION

    TWO MODES (gated by `multiscale`):
      * multiscale=False (iter-26/28/31 CHAMPION behaviour, BYTE-FOR-BYTE): the M
        bilaterals are a SEQUENTIAL CASCADE applied to Q, scaled by ONE shared
        scalar gain ->  Q̂ = BF_M(...BF_1(Q));  N̂ = Q̂²;  P̂ = -ln(N̂/N0);
        out = P + gain*(P̂ - P). At proj_n_bf=1 this is iter-31 EXACTLY (3 σ +
        log_N0 + 1 gain = 5 params).
      * multiscale=True (iter-32 THE LEVER): the M bilaterals run IN PARALLEL on
        the SAME Q at DISTINCT spatial σ (a multi-RF projection bank, analogous to
        the image multi-RF bilateral idea). Each produces its OWN denoised log-sino
        P̂_i = -ln(BF_i(Q)²/N0) and contributes gain_i*(P̂_i - P) with its OWN
        ZERO-INIT gain_i:
            out = P + Σ_{i=1..M} gain_i * ( P̂_i - P )
        Params: M*(3 σ) + 1 log_N0 + M gains. At M=3: 9+1+3 = 13.

    NOISE-ADAPTIVITY (Manduca §II.B-E): the bilateral runs on the sqrt-count image
    Q where, under Poisson statistics, the noise std is ~CONSTANT (Anscombe)
    regardless of the local count level, so a SINGLE fixed range σ on Q is
    IMPLICITLY noise-adaptive in the line-integral domain -- a high-attenuation
    (dense, low-count, NOISY) ray P -> small N=N0·exp(-P) -> small Q -> the same
    range σ spans more of its noise -> MORE smoothing; an air ray (high count,
    clean) gets proportionally less. The strength scale is the photon count N0
    (parameterised in log-space so it stays strictly positive and -ln(N̂/N0) is
    well defined). The multi-scale bank gives DISTINCT spatial bandwidths so the
    projection denoise can smooth broad low-frequency streaks AND preserve fine
    high-frequency ray structure in parallel rather than at one fixed σ.

    ZERO-INIT GAIN (the stability fix): with ``gain_init=0.0`` EVERY gain is 0 at
    init, so out == P EXACTLY at init REGARDLESS of the σ / N0 values -- the
    LD-FBP init AND the DC term g downstream are iter-31's byte-for-byte. The
    optimiser lifts each gain off zero only if that scale's proj denoise lowers
    val-RMSE; the learned per-scale gain VECTOR is the experiment.

    RESPECTS GENTLENESS (Maier methodology): each added projection scale is GENTLE
    (zero-init gain, soft range σ); too MUCH projection denoising -> over-smooth,
    too-HARD edge preservation -> view inconsistency. The lever is STRENGTH (the
    learned gains), not edge-hardness."""

    def __init__(self, n_bf: int = 1, kernel_size: int = 5,
                 sigma_x: float = 0.83, sigma_y: float = 0.83,
                 sigma_r: float = 1.5, N0: float = 1.0e5,
                 eps: float = 1.0, gain_init: float = 0.0,
                 multiscale: bool = False, sigmas=None):
        super().__init__()
        n_bf = max(1, int(n_bf))
        k = int(kernel_size)
        if k % 2 == 0:                       # bilateral needs an odd kernel
            k += 1
        self.multiscale = bool(multiscale)
        if self.multiscale:
            # iter-32: per-filter DISTINCT spatial-σ init from `sigmas`; each
            # filter's σ_x AND σ_y init to that value; range σ_r shared. Short/
            # None list falls back to sigma_x for the rest.
            seq = list(sigmas) if sigmas else [sigma_x]
            per = [float(seq[i]) if i < len(seq) else float(sigma_x)
                   for i in range(n_bf)]
            self.filters = nn.ModuleList(
                [TrainableBilateralFilter2d(kernel_size=k,
                                            sigma_x=per[i], sigma_y=per[i],
                                            sigma_r=float(sigma_r))
                 for i in range(n_bf)])
            self.sigmas_init = per
        else:
            # iter-26/31: all M filters share one (sigma_x, sigma_y) init.
            self.filters = nn.ModuleList(
                [TrainableBilateralFilter2d(kernel_size=k,
                                            sigma_x=float(sigma_x),
                                            sigma_y=float(sigma_y),
                                            sigma_r=float(sigma_r))
                 for _ in range(n_bf)])
            self.sigmas_init = [float(sigma_x)] * n_bf
        self.log_N0 = nn.Parameter(torch.tensor(math.log(max(float(N0), 1.0))))
        self.eps = float(eps)
        # multiscale: PER-FILTER gains (M); single/cascade: ONE shared scalar.
        # Either way ALL ZERO-INIT => residual == 0 at init REGARDLESS of σ / N0.
        if self.multiscale:
            self.gains = nn.Parameter(torch.full((n_bf,), float(gain_init)))
            self.gain = None
        else:
            self.gain = nn.Parameter(torch.tensor(float(gain_init)))
            self.gains = None

    def _Q(self, P: torch.Tensor):
        """Anscombe/sqrt variance-stabilised photon counts Q = sqrt(N0*exp(-P))."""
        logN0 = self.log_N0
        # N = N0*exp(-P) = exp(logN0 - P); clamp the exponent for fp stability
        # (staged log-sinos are O(1-10) so this never bites on real data).
        N = torch.exp((logN0 - P).clamp(max=30.0))
        return torch.sqrt(N.clamp_min(0.0) + 1e-8), logN0

    def _phat_from_Q(self, Qhat: torch.Tensor, logN0) -> torch.Tensor:
        """Denoised log-sino P̂ = -ln(Q̂²/N0) from a (denoised) sqrt-count Q̂."""
        Nhat = Qhat.clamp_min(0.0) ** 2
        return logN0 - torch.log(Nhat.clamp_min(self.eps))   # -ln(N̂/N0)

    def _manduca(self, P: torch.Tensor) -> torch.Tensor:
        """The denoised log-sinogram P̂ (un-gained), single/cascade mode (the
        sequential cascade BF_M(...BF_1(Q))). Used by `residual`/`forward` when
        multiscale=False; preserved byte-for-byte for the iter-31 path."""
        Q, logN0 = self._Q(P)
        for f in self.filters:
            Q = f(Q)
        return self._phat_from_Q(Q, logN0)

    def residual(self, P: torch.Tensor) -> torch.Tensor:
        """The (un-gained) residual.
        - single/cascade: P̂ - P (one scalar gain multiplies it downstream).
        - multiscale: Σ_i (P̂_i - P), the un-gained summed multi-scale residual
          (exposed for the zero-at-init self-check; the gained term uses
          `added_term` with the per-filter gains)."""
        if self.multiscale:
            Q, logN0 = self._Q(P)
            res = torch.zeros_like(P)
            for f in self.filters:
                phat_i = self._phat_from_Q(f(Q), logN0)   # parallel: each on Q
                res = res + (phat_i - P)
            return res
        return self._manduca(P) - P

    def added_term(self, P: torch.Tensor) -> torch.Tensor:
        """The FULL gained projection residual actually added to P.
        - multiscale: Σ_i gain_i*(P̂_i - P) (per-filter gains).
        - single/cascade: gain*(P̂ - P) (one shared gain)."""
        if self.multiscale:
            Q, logN0 = self._Q(P)
            out = torch.zeros_like(P)
            for i, f in enumerate(self.filters):
                phat_i = self._phat_from_Q(f(Q), logN0)   # parallel: each on Q
                out = out + self.gains[i] * (phat_i - P)
            return out
        return self.gain * self.residual(P)

    def gain_values(self):
        """The learned gain(s) as a python list (per-filter in multiscale)."""
        if self.gains is not None:
            return [float(g) for g in self.gains.detach().cpu()]
        return [float(self.gain.detach().cpu())]

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        return P + self.added_term(P)

    @torch.no_grad()
    def sigmas(self):
        return [(float(torch.exp(f.log_sx).cpu()),
                 float(torch.exp(f.log_sy).cpu()),
                 float(torch.exp(f.log_sr).cpu())) for f in self.filters]

    @torch.no_grad()
    def N0_value(self) -> float:
        return float(torch.exp(self.log_N0).cpu())


class AnisotropicOrientedReg(nn.Module):
    """iter-31 — additive zero-init-gain ORIENTED anisotropic-smoothing residual.

    LD-CT streaks are DIRECTIONAL (photon-starved high-attenuation rays backproject
    as oriented streaks). The gentle Manduca projection filter (iter-26/28 champion)
    deliberately leaves SOME streak residual behind (it must stay soft to preserve
    view consistency). This module adds a direction-AWARE image-domain cleanup that
    the ISOTROPIC bilateral (iter-30, redundant on Manduca) could not provide:

        reg(x) += gain * Σ_{θ∈Θ} ( x - G_θ(x) )

    where G_θ is an ORIENTED ANISOTROPIC GAUSSIAN blur — elongated ALONG orientation
    θ (sigma_along, large) and narrow ACROSS it (sigma_across, small). Subtracting
    (x - G_θ(x)) inside the prox step smooths ALONG the streak direction (suppressing
    the oriented streak) while preserving structure ACROSS the edge. Summing over a
    small fixed set of orientations Θ catches streaks at any direction.

    PARAMETERIZATION (few-param + STABLE, NOT a steerable/equivariant reparam):
      * Θ = M FIXED orientations on [0, pi) (e.g. {0, 45, 90, 135} deg). FIXED, not
        learnable -> a plain oriented filter BANK, NOT the iter-24 coeff->basis
        reparam that DIVERGED. The kernel weights are computed ANALYTICALLY from
        two sigmas + the rotation angle (no trainable conv weights).
      * sigma_along (log_s_along): blur extent ALONG the streak (init large ~2.0 px).
      * sigma_across (log_s_across): blur extent ACROSS the streak (init small
        ~0.6 px). SHARED across all M orientations (each orientation just rotates
        the same anisotropic kernel). 2 learnable bandwidth params total.
      * gain: ONE shared scalar, ZERO-INIT => the WHOLE oriented term = 0 at init
        REGARDLESS of the sigmas / orientations. Lifted off 0 only if direction-
        aware smoothing lowers val-RMSE.

    Trainable params: 2 sigmas + 1 gain = 3 (the M orientations are fixed buffers).
    """

    def __init__(self, n_orient: int = 4, kernel_size: int = 7,
                 sigma_along: float = 2.0, sigma_across: float = 0.6,
                 gain_init: float = 0.0):
        super().__init__()
        k = int(kernel_size)
        if k % 2 == 0:                       # odd kernel for a centred window
            k += 1
        self.k = k
        self.n_orient = max(1, int(n_orient))
        # FIXED orientations on [0, pi): {0, pi/M, 2pi/M, ...}. Buffer, NOT a param.
        thetas = torch.tensor(
            [math.pi * i / self.n_orient for i in range(self.n_orient)])
        self.register_buffer("thetas", thetas)
        # learnable log-bandwidths (along/across the orientation), shared over Θ.
        self.log_s_along = nn.Parameter(torch.tensor(math.log(float(sigma_along))))
        self.log_s_across = nn.Parameter(torch.tensor(math.log(float(sigma_across))))
        # ONE shared scalar gain, ZERO-INIT => oriented term == 0 at init.
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))

    def _kernels(self, device, dtype) -> torch.Tensor:
        """The M analytic oriented anisotropic-Gaussian kernels (M,1,k,k),
        each normalised to sum 1 so G_θ is a smoothing (low-pass) operator."""
        r = self.k // 2
        s_along = torch.exp(self.log_s_along)
        s_across = torch.exp(self.log_s_across)
        ys = torch.arange(-r, r + 1, device=device, dtype=dtype)
        xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")               # (k,k)
        ker = []
        for i in range(self.n_orient):
            th = self.thetas[i].to(device=device, dtype=dtype)
            ct, st = torch.cos(th), torch.sin(th)
            # rotate the grid into (along, across) the orientation th
            u = gx * ct + gy * st         # along the streak direction
            v = -gx * st + gy * ct        # across it
            g = torch.exp(-0.5 * ((u / s_along) ** 2 + (v / s_across) ** 2))
            g = g / g.sum().clamp_min(1e-12)
            ker.append(g)
        return torch.stack(ker, dim=0).unsqueeze(1)                  # (M,1,k,k)

    def oriented_term(self, x: torch.Tensor) -> torch.Tensor:
        """The (un-gained) summed oriented residual Σ_θ (x - G_θ(x))."""
        ker = self._kernels(x.device, x.dtype)                       # (M,1,k,k)
        r = self.k // 2
        x_pad = F.pad(x, (r, r, r, r), mode="reflect")
        blurred = F.conv2d(x_pad, ker)                               # (B,M,H,W)
        # Σ_θ (x - G_θ(x)) = M*x - Σ_θ G_θ(x)
        return self.n_orient * x - blurred.sum(dim=1, keepdim=True)

    def added_term(self, x: torch.Tensor) -> torch.Tensor:
        """The FULL gained oriented term gain * Σ_θ (x - G_θ(x))."""
        return self.gain * self.oriented_term(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.added_term(x)

    @torch.no_grad()
    def sigmas(self):
        return (float(torch.exp(self.log_s_along).cpu()),
                float(torch.exp(self.log_s_across).cpu()))

    @torch.no_grad()
    def orientations_deg(self):
        return [float(t.cpu()) * 180.0 / math.pi for t in self.thetas]


class ComposedFoEAnisotropicReg(nn.Module):
    """iter-31 — composed image reg = FoE  +  zero-init-gain ORIENTED anisotropic term.

        reg(x) = FoE(x)  +  gain * Σ_{θ} ( x - G_θ(x) )

    The iter-7 CHAMPION FoE bank (zero-init RBF => FoE(x)=0 at init) PLUS the
    AnisotropicOrientedReg directional-smoothing residual (zero-init gain => term=0
    at init). Both terms are EXACTLY 0 at init, so reg(x)==0 at init. Composed on
    the iter-28 champion (FoE + Manduca projection filter) the WHOLE pipeline ==
    iter-28 byte-for-byte at init; the optimiser lifts the gain off zero ONLY if the
    direction-aware image cleanup adds headroom on top of the projection-domain
    streak fix. The learned gain (and along/across sigmas) ARE the experiment.

    Params: FoE (1,920) + anisotropic (2 sigmas + 1 gain = 3)."""

    def __init__(self, foe: nn.Module, n_orient: int = 4, kernel_size: int = 7,
                 sigma_along: float = 2.0, sigma_across: float = 0.6,
                 gain_init: float = 0.0):
        super().__init__()
        self.foe = foe                      # the iter-7 FoE bank (zero-init RBF)
        self.aniso = AnisotropicOrientedReg(
            n_orient=n_orient, kernel_size=kernel_size,
            sigma_along=sigma_along, sigma_across=sigma_across,
            gain_init=gain_init)

    def gain_value(self) -> float:
        return float(self.aniso.gain.detach().cpu())

    def added_term(self, x: torch.Tensor) -> torch.Tensor:
        """The full gained oriented term (exposed for the zero-at-init self-check)."""
        return self.aniso.added_term(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.foe(x) + self.aniso(x)


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
        # iter-31 THE LEVER: optionally compose a zero-init-gain ORIENTED
        # ANISOTROPIC smoothing residual ON TOP (a DIRECTION-aware image cleanup
        # for the residual directional streaks the gentle Manduca projection filter
        # leaves behind). Takes precedence over the bilateral branch. gain=0 at init
        # => the term is exactly 0 => reg == FoE byte-for-byte at init.
        if bool(cfg.get("anisotropic_on_top", False)):
            return ComposedFoEAnisotropicReg(
                foe, n_orient=cfg.get("aniso_n_orient", 4),
                kernel_size=cfg.get("aniso_kernel", 7),
                sigma_along=cfg.get("aniso_sigma_along", 2.0),
                sigma_across=cfg.get("aniso_sigma_across", 0.6),
                gain_init=cfg.get("aniso_gain_init", 0.0))
        # iter-25/iter-27: optionally compose a zero-init-gain bilateral residual
        # bank ON TOP. iter-27 multiscale=True => M parallel filters at distinct
        # init sigmas, each its OWN zero-init gain (a multi-RF bank).
        if bool(cfg.get("bilateral_on_top", False)):
            return ComposedFoEBilateralReg(
                foe, n_bf=cfg.get("n_bf", 1), kernel_size=cfg.get("bf_kernel", 7),
                sigma_x=cfg.get("bf_sigma_x", 1.5),
                sigma_y=cfg.get("bf_sigma_y", 1.5),
                sigma_r=cfg.get("bf_sigma_r", 0.02),
                gain_init=cfg.get("bf_gain_init", 0.0),
                multiscale=bool(cfg.get("bf_multiscale", False)),
                sigmas=cfg.get("bf_sigmas"))
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
        x  = clamp_min(x - alpha * (dc + reg(x)), 0.0)   # mu>=0 only (no upper clamp since 2026-06-29)
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
                gain_init=cfg.get("proj_gain_init", 0.0),
                multiscale=bool(cfg.get("proj_multiscale", False)),
                sigmas=cfg.get("proj_sigmas"))
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
        # 2026-06-29: upper clamp REMOVED (was clamp(...,0,clip_max=0.05)). The
        # 0.05 ceiling = the FBP *display* window, not physical mu: truth bone/
        # contrast reaches 0.0814 (19% of val slices >0.05), so the in-loop upper
        # clamp truncated bone. Now mu>=0 only, matching ITNet/Hammernik (no upper clamp).
        x_new = torch.clamp_min(y - alpha_k * self._grad(y, sino), 0.0)
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
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k in ('reg_type','bilateral_on_top','proj_filter_on','proj_n_bf','proj_multiscale','proj_sigmas','proj_kernel','proj_sigma_x','proj_sigma_y','proj_sigma_r','proj_N0','proj_eps','proj_gain_init','n_iter','learnable_alpha','per_step_alpha','momentum','beta_init','alpha_init','clip_max','mu_channels','cnn_channels','cnn_layers','cnn_dilations','foe_n_filters','foe_kernel','foe_n_bumps','foe_rbf_init_std','n_bf','bf_multiscale','bf_sigmas','bf_kernel','bf_sigma_x','bf_sigma_y','bf_sigma_r','bf_gain_init','anisotropic_on_top','aniso_n_orient','aniso_kernel','aniso_sigma_along','aniso_sigma_across','aniso_gain_init','epochs','cosine_lr','cosine_lr_min','max_train_s','batch_size','lr','train_n','val_n')}, default=str)}",
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
    if per_ps and (model.proj_filter is not None
                   or bool(cfg.get("bilateral_on_top", False))
                   or bool(cfg.get("anisotropic_on_top", False))):
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
            # iter-32: multiscale-aware -> read EVERY per-filter gain (single mode
            # returns the one shared gain as a 1-element list).
            pgain_vec = model.proj_filter.gain_values()
            sino_f = model.proj_filter(p_chk)
            sino_diff_n = float((sino_f - p_chk).norm())
            resid_n = float(model.proj_filter.residual(p_chk).norm())
            _pg = ", ".join(f"{g:.3e}" for g in pgain_vec)
            print(f"[selfcheck] proj_filter==identity at init: multiscale="
                  f"{model.proj_filter.multiscale} proj_gains=[{_pg}]  "
                  f"‖sino_filtered - sino‖={sino_diff_n:.3e}  "
                  f"‖(un-gained) residual Σ(P̂_i-P)‖={resid_n:.3e}  N0_init="
                  f"{model.proj_filter.N0_value():.4g}  "
                  f"(ALL proj_gains MUST be 0.0, ‖sino_filtered - sino‖ MUST be 0)",
                  flush=True)
            assert all(g == 0.0 for g in pgain_vec), (
                f"a proj_gain NOT zero at init (proj_gains={pgain_vec}): the proj "
                "zero-init gain is broken -> sino_filtered != sino. ABORTING.")
            assert sino_diff_n == 0.0, (
                f"sino_filtered != sino at init (‖diff‖={sino_diff_n:.3e}): the "
                "proj filter is NOT exactly the identity at init. ABORTING.")

    # iter-27 RUNTIME SELF-CHECK (kept; only fires when bilateral_on_top=True):
    # confirm the IMAGE reg(x) == FoE(x) at init via the zero-init bilateral gains.
    # Multi-scale-aware: asserts EVERY per-filter gain is 0 AND the WHOLE gained
    # bank Σ_i gain_i*(x-BF_i(x)) == 0 EXACTLY (REGARDLESS of the distinct σ_i).
    if bool(cfg.get("bilateral_on_top", False)) and isinstance(
            model.reg, ComposedFoEBilateralReg) and val_u0.shape[0] > 0:
        with torch.no_grad():
            x_chk = val_u0[0:1]
            gain_vec = model.reg.gain_values()        # list (per-filter in MS)
            foe_out = model.reg.foe(x_chk)
            bil_term = model.reg.added_term(x_chk)    # the ACTUAL gained bank
            reg_out = model.reg(x_chk)
            foe_n = float(foe_out.norm())
            bil_n = float(bil_term.norm())
            reg_n = float(reg_out.norm())
            diff_n = float((reg_out - foe_out).norm())
            _gv = ", ".join(f"{g:.3e}" for g in gain_vec)
            print(f"[selfcheck] composed reg==FoE at init: multiscale="
                  f"{model.reg.multiscale} gains=[{_gv}]  ‖FoE(x)‖={foe_n:.3e}  "
                  f"‖Σ gain_i*(x-BF_i)‖={bil_n:.3e}  ‖reg(x)‖={reg_n:.3e}  "
                  f"‖reg-FoE‖={diff_n:.3e}  (ALL gains MUST be 0.0, ‖reg-FoE‖ MUST be 0)",
                  flush=True)
            assert all(g == 0.0 for g in gain_vec), (
                f"a bilateral gain NOT zero at init (gains={gain_vec}): the "
                "zero-init gain is broken -> seed is NOT iter-7. ABORTING.")
            assert diff_n == 0.0, (
                f"reg(x) != FoE(x) at init (‖reg-FoE‖={diff_n:.3e}): the "
                "bilateral bank is NOT exactly 0 at init. ABORTING.")

    # iter-31 RUNTIME SELF-CHECK (only fires when anisotropic_on_top=True): confirm
    # the IMAGE reg(x) == FoE(x) at init via the zero-init oriented gain. Asserts the
    # gain is 0 AND the WHOLE gained oriented term gain*Σ_θ(x-G_θ(x)) == 0 EXACTLY
    # (REGARDLESS of the along/across sigmas / the fixed orientations). With zero-init
    # RBF FoE(x)==0 too, so reg(x)==0 == iter-7's seed; composed on the iter-28 base
    # (FoE+Manduca) the WHOLE pipeline == iter-28 byte-for-byte at init.
    if bool(cfg.get("anisotropic_on_top", False)) and isinstance(
            model.reg, ComposedFoEAnisotropicReg) and val_u0.shape[0] > 0:
        with torch.no_grad():
            x_chk = val_u0[0:1]
            a_gain = model.reg.gain_value()
            foe_out = model.reg.foe(x_chk)
            ani_term = model.reg.added_term(x_chk)        # the ACTUAL gained term
            reg_out = model.reg(x_chk)
            foe_n = float(foe_out.norm())
            ani_n = float(ani_term.norm())
            reg_n = float(reg_out.norm())
            diff_n = float((reg_out - foe_out).norm())
            s_along, s_across = model.reg.aniso.sigmas()
            _or = ", ".join(f"{d:.0f}" for d in model.reg.aniso.orientations_deg())
            print(f"[selfcheck] composed reg==FoE at init: aniso_gain={a_gain:.3e}  "
                  f"orientations=[{_or}]deg  σ_along={s_along:.3f} σ_across={s_across:.3f}  "
                  f"‖FoE(x)‖={foe_n:.3e}  ‖gain*Σ_θ(x-G_θ)‖={ani_n:.3e}  "
                  f"‖reg(x)‖={reg_n:.3e}  ‖reg-FoE‖={diff_n:.3e}  "
                  f"(aniso_gain MUST be 0.0, ‖reg-FoE‖ MUST be 0)", flush=True)
            assert a_gain == 0.0, (
                f"aniso_gain NOT zero at init (gain={a_gain:.3e}): the oriented "
                "zero-init gain is broken -> seed is NOT iter-28. ABORTING.")
            assert diff_n == 0.0, (
                f"reg(x) != FoE(x) at init (‖reg-FoE‖={diff_n:.3e}): the oriented "
                "anisotropic term is NOT exactly 0 at init. ABORTING.")

    # iter-7 FoE zero-init guard (fires whenever foe_rbf_init_std==0.0): the image
    # reg(x) == 0 at init so the recon seed is iter-7's clean GD+DC scheme.
    if cfg.get("reg_type") == "foe" and float(cfg.get("foe_rbf_init_std", 0.0)) == 0.0 \
            and val_u0.shape[0] > 0:
        with torch.no_grad():
            foe_mod = (model.reg.foe if isinstance(
                model.reg, (ComposedFoEBilateralReg, ComposedFoEAnisotropicReg))
                else model.reg)
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
    # iter-27: per-filter gains (multiscale) or ONE shared gain (iter-25), all in
    # the reg, so n_gain counts them; the bilateral σ + gains are part of reg_params.
    n_gain = (len(model.reg.gain_values())
              if (bool(cfg.get("bilateral_on_top", False))
                  and isinstance(model.reg, ComposedFoEBilateralReg)) else 0)
    # iter-26: the PROJECTION-domain Manduca filter params, counted separately
    # (OFF in iter-27 -> proj_params=0).
    proj_params = (sum(p.numel() for p in model.proj_filter.parameters()
                       if p.requires_grad) if model.proj_filter is not None else 0)
    _ms = (bool(model.reg.multiscale)
           if isinstance(model.reg, ComposedFoEBilateralReg) else False)
    print(f"[solver] ParamEfficient iter-27: reg_type={cfg['reg_type']!r}  "
          f"bilateral_on_top={bool(cfg.get('bilateral_on_top', False))} "
          f"(n_bf={cfg.get('n_bf')}, multiscale={_ms}, bf_sigmas={cfg.get('bf_sigmas')}, "
          f"gain_init={cfg.get('bf_gain_init')})  "
          f"proj_filter_on={bool(cfg.get('proj_filter_on', False))}  "
          f"n_iter={cfg['n_iter']} (weight-TIED reg)  "
          f"per_step_alpha={model.per_step_alpha} momentum={model.use_momentum}  "
          f"epochs={cfg['epochs']} cosine_lr={cfg.get('cosine_lr', False)} "
          f"peak_lr={cfg.get('lr')}  "
          f"trainable params={params_total} "
          f"(reg={reg_params} + alpha={n_alpha} + beta={n_beta}; gains∈reg={n_gain}; "
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
        # iter-27/iter-31: print the learned reg gain(s) each epoch (which term the
        # optimiser is engaging): the per-filter bilateral VECTOR (iter-27) or the
        # single oriented anisotropic gain (iter-31).
        if isinstance(model.reg, ComposedFoEBilateralReg):
            _gain_s = " gains=[" + ", ".join(f"{g:.4g}"
                      for g in model.reg.gain_values()) + "]"
        elif isinstance(model.reg, ComposedFoEAnisotropicReg):
            _gain_s = f" aniso_gain={model.reg.gain_value():.4g}"
        else:
            _gain_s = ""
        # iter-32: print the per-scale PROJECTION gain VECTOR each epoch (the
        # experiment -> which projection spatial scale(s) the optimiser engages).
        if model.proj_filter is not None:
            _gain_s += " proj_gains=[" + ", ".join(
                f"{g:.4g}" for g in model.proj_filter.gain_values()) + "]"
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

    # iter-27: report the learned IMAGE bilateral bank. The per-filter gain VECTOR
    # is the answer (which scales the optimiser kept); off when bilateral_on_top
    # is False. `gain_learned` keeps a scalar (mean gain) for back-compat with the
    # iter-25 single-gain convention; `bf_gains_learned` is the full per-filter
    # vector and `bf_sigmas_learned` the per-filter learned (σx,σy,σr).
    if isinstance(model.reg, ComposedFoEBilateralReg):
        bf_gains_learned = model.reg.gain_values()
        bf_sigmas_learned = model.reg.sigmas_learned()
        bf_multiscale_used = bool(model.reg.multiscale)
        gain_learned = float(sum(bf_gains_learned) / max(1, len(bf_gains_learned)))
    else:
        bf_gains_learned = None
        bf_sigmas_learned = None
        bf_multiscale_used = False
        gain_learned = None

    # iter-31: report the learned ORIENTED ANISOTROPIC term. The gain's magnitude
    # IS the answer: ~0 = the gentle Manduca projection filter already subsumes the
    # residual directional streaks (a clean NULL -> Manduca subsumes image-domain
    # streak cleanup); meaningfully nonzero = direction-aware image smoothing adds
    # headroom on top of FoE+Manduca. Also report the learned along/across sigmas.
    if isinstance(model.reg, ComposedFoEAnisotropicReg):
        aniso_gain_learned = model.reg.gain_value()
        aniso_sigmas_learned = list(model.reg.aniso.sigmas())          # (along, across)
        aniso_orient_deg = model.reg.aniso.orientations_deg()
        aniso_n_orient_used = int(model.reg.aniso.n_orient)
    else:
        aniso_gain_learned = None
        aniso_sigmas_learned = None
        aniso_orient_deg = None
        aniso_n_orient_used = 0

    # iter-26: report the learned PROJECTION-domain Manduca filter. proj_gain's
    # magnitude IS the answer: ~0 = the dense-view FBP already subsumes the proj
    # denoise (the predicted dense-view ceiling, a clean NULL); meaningfully
    # nonzero = the upstream noise-adaptive proj filter contributes on top of FoE.
    if model.proj_filter is not None:
        # iter-32: per-filter proj gain VECTOR is the answer (which projection
        # scales the optimiser kept). `proj_gain_learned` keeps a scalar (mean)
        # for back-compat with the iter-26/31 single-gain convention.
        proj_gains_learned = model.proj_filter.gain_values()
        proj_gain_learned = float(sum(proj_gains_learned)
                                  / max(1, len(proj_gains_learned)))
        proj_multiscale_used = bool(model.proj_filter.multiscale)
        proj_sigmas_learned = model.proj_filter.sigmas()
        proj_N0_learned = model.proj_filter.N0_value()
        proj_params_n = sum(p.numel() for p in model.proj_filter.parameters()
                            if p.requires_grad)
    else:
        proj_gains_learned = None
        proj_gain_learned = None
        proj_multiscale_used = False
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
        "gain_learned": gain_learned,                 # mean over filters (back-compat)
        # iter-27: multi-RF parallel bilateral bank
        "bf_multiscale": bf_multiscale_used,
        "bf_sigmas_init": cfg.get("bf_sigmas"),
        "bf_gains_learned": bf_gains_learned,         # per-filter gain VECTOR (the answer)
        "bf_sigmas_learned": bf_sigmas_learned,       # per-filter learned (σx,σy,σr)
        # iter-31: oriented anisotropic image term (direction-aware streak cleanup)
        "anisotropic_on_top": bool(cfg.get("anisotropic_on_top", False)),
        "aniso_n_orient": aniso_n_orient_used,
        "aniso_kernel": int(cfg.get("aniso_kernel", 7)),
        "aniso_gain_init": float(cfg.get("aniso_gain_init", 0.0)),
        "aniso_gain_learned": aniso_gain_learned,     # the answer (0 = Manduca subsumes it)
        "aniso_sigmas_learned": aniso_sigmas_learned, # learned (σ_along, σ_across)
        "aniso_orient_deg": aniso_orient_deg,         # the fixed orientations (deg)
        # iter-26: projection-domain Manduca filter
        "proj_filter_on": bool(cfg.get("proj_filter_on", False)),
        "proj_n_bf": int(cfg.get("proj_n_bf", 1)),
        "proj_kernel": int(cfg.get("proj_kernel", 5)),
        "proj_gain_init": float(cfg.get("proj_gain_init", 0.0)),
        "proj_params": proj_params_n,
        "proj_gain_learned": proj_gain_learned,                 # mean over scales (back-compat)
        # iter-32: multi-scale PARALLEL projection bank
        "proj_multiscale": proj_multiscale_used,
        "proj_sigmas_init": cfg.get("proj_sigmas"),
        "proj_gains_learned": proj_gains_learned,               # per-scale gain VECTOR (the answer)
        "proj_sigmas_learned": proj_sigmas_learned,             # per-scale learned (σx,σy,σr)
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
    # iter-27: print the per-filter gain VECTOR (which scales engaged) + learned σ.
    if bf_gains_learned is not None:
        _gv = ", ".join(f"{g:.4g}" for g in bf_gains_learned)
        _sv = "; ".join(f"σx={sx:.3f} σy={sy:.3f} σr={sr:.4f}"
                        for (sx, sy, sr) in bf_sigmas_learned)
        _gain_s = f"[{_gv}] (ms={bf_multiscale_used}) bf_sigmas[{_sv}]"
    elif aniso_gain_learned is not None:
        _or = ", ".join(f"{d:.0f}" for d in aniso_orient_deg)
        _gain_s = (f"aniso_gain={aniso_gain_learned:.4g} "
                   f"σ_along={aniso_sigmas_learned[0]:.3f} "
                   f"σ_across={aniso_sigmas_learned[1]:.3f} "
                   f"orient=[{_or}]deg")
    else:
        _gain_s = "off"
    if proj_gain_learned is not None:
        _ps = "; ".join(f"σx={sx:.3f} σy={sy:.3f} σr={sr:.4f}"
                        for (sx, sy, sr) in proj_sigmas_learned)
        _pgv = ", ".join(f"{g:.4g}" for g in proj_gains_learned)
        _proj_s = (f"proj_gains=[{_pgv}] (ms={proj_multiscale_used}) "
                   f"N0={proj_N0_learned:.4g} proj[{_ps}]")
    else:
        _proj_s = "proj_filter=off"
    print(f"[solver] ParamEfficient: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"params={params_total}  alpha_mean={result['alpha_learned']:.4g}  "
          f"gain={_gain_s}  beta={_beta_s}  {_proj_s}  time={train_time:.1f}s  "
          f"(intensity-calibrated)", flush=True)

    try:
        _aniso_on = bool(cfg.get("anisotropic_on_top", False))
        if model.proj_filter is not None and _aniso_on:
            _lbl = f"ParamEff[{cfg['reg_type']}+projBF+aniso]"
        elif model.proj_filter is not None:
            _lbl = f"ParamEff[{cfg['reg_type']}+projBF]"
        elif _aniso_on:
            _lbl = f"ParamEff[{cfg['reg_type']}+aniso]"
        else:
            _lbl = f"ParamEff[{cfg['reg_type']}+bil]"
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
