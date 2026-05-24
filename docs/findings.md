---
title: Cross-cutting findings
description: Substantive learnings from the autoresearch loop that span multiple iterations, agents, or sessions. Newest first.
---

This file is the cross-agent handoff log. Per-iteration rationale belongs in
`docs/runs/<slug>/iterations/iter-NNNN/observation.json`. Stage-check
verdicts belong in `docs/runs/<slug>/stages.tsv`. Things that belong here:
**facts about the substrate or methodology that the next agent should not
have to re-discover.**

📋 **Method**: see [`solver_plan.md`](../solver_plan.md) (the canonical
recipe for adapting solvers to a new dataset — FBP investigation,
agentic autoresearch, TPE refinement, DDPM constrained+unconstrained,
leaderboard + per-solver cross-dataset insights).

## 2026-05-24 — `mayo_ldct` Wagner split is the on-disk convention

Mirrors the Wagner et al. 2023 ISBI paper's experimental setup; defined
once in `data/fetch_mayo_ldct.py` (`WAGNER_SPLITS`) and consumed by every
Mayo-touching script:

```
Train: L145, L186, L209, L219      (4 patients, used to train supervised solvers)
Val:   L277                         (1 patient, used for early-stopping / hyperparam selection)
Test:  L014, L056, L058, L075, L123 (5 patients, only touched at final eval)
```

Every Mayo helix-2-fan rebinning and validator pass operates on
patients sourced from this split. For the DDPM constrained/unconstrained
distinction (see `solver_plan.md` Step 4), **constrained = train on
L145/186/209/219 labels only**, **unconstrained = train on all 10
patients** — the latter measures how much the diffusion prior benefits
from having seen test-set anatomy.

## 2026-05-23 — Mayo helix2fan DOMINANT BUG FOUND: alphabetic `sorted(files)` ≠ acquisition order

The "featureless disc" FBP from the previous-agent's rebin + this session's
re-rebin had a much simpler root cause than the Bug 1–6 list in
`literature/wagner_helix2fan_algorithm.md` suggested. **`read_dicom_ctpd`
in `ddssl_ldct/helix2fan.py` was reading the per-readout `.dcm` files in
alphabetic order**, which on Mayo DICOM-CT-PD data does not match
acquisition order at all.

Concrete evidence (`results/breast_debug/L014_file_order_check.png`,
job 762035 — sample of 1500 alphabetic-sorted L014 fulldose files):

| alphabetic filename | InstanceNumber | gantry angle | z (mm) |
|---|---|---|---|
| 00000001.dcm | 10298 | −1.18 | 137.7 |
| 00000002.dcm | 23913 | −0.61 | 318.8 |
| 00000003.dcm | 26560 | −1.55 | 354.0 |
| 00000004.dcm | 37666 | −0.42 | 501.8 |
| 00000005.dcm | 2811 | +0.39 | 38.0 |
| 00000006.dcm | 35550 | −0.93 | 473.7 |
| 00000007.dcm | 26625 | +4.56 | 354.9 |

`InstanceNumber` is the time index (Mayo writes 1..n_proj sequentially
during the helix sweep). Over 1500 alphabetic-sorted files, `InstanceNumber`
spans 48..37 945 essentially uniformly random. Sorting by alphabetic
filename therefore produces a sinogram whose source-z and gantry-angle
oscillate chaotically across the full scan extent (visible in
`L014_unwrap_check.png` and `L014_fulldose_raw_sino_inspect.png`).

The SSR step in `rebin_helical_to_fan` indexes helical readouts as
`idx_helix = arange(s_angle, n_proj, rotview)` — i.e. it assumes
consecutive file indices are consecutive in the helix. With files
shuffled, each rebinned `s_angle` bin gathers 60 readouts at wildly
different physical angles and z-positions — explaining why the
rebinned sino has no patient sinusoids and the FBP has no anatomy
(SSIM 0.19 against an *also-misaligned* truth slice).

**Patch applied 2026-05-23** (`ddssl_ldct/helix2fan.py`,
`read_dicom_ctpd`): added a pre-pass that reads each file's
`InstanceNumber` header and re-sorts the file list by it before
the main pixel-read loop. ~1 extra second per 1000 files (~40 s for
L014's 37 982 fulldose projections — a once-per-rebin overhead).

L014 fulldose is being re-rebinned with this fix (job 762036). The
old buggy artefact is preserved under
`*_alphabetic_sort_buggy.{h5,json,npy}` for diagnostic comparison.

Pending: re-validate FBP after the rebin. If the FBP now shows real
anatomy (lungs / ribs / spine), the file-order patch is the dominant
fix and Bug 1–6 may or may not still apply (most likely the rebin code
was structurally correct all along; we'd only need to verify against
torch-radon angle convention in the validator before scaling out to
the other 9 patients).

### Wagner papers do NOT validate helix2fan against image truth

While diagnosing this, established that **no Wagner paper actually
tests FBP-of-rebinned-sino against the Mayo reconstructed-image truth**:

- **Wagner 2022 (Med. Phys., trainable BF)**: uses Mayo's reconstructed
  CT image DICOMs (3411 slices) + Yu et al. noise simulation. **No
  helix2fan at all.**
- **Wagner 2023 (ISBI, dual-domain)**: uses helix2fan but reports only
  LD-vs-HD PSNR within its own pipeline. If both LD and HD go through
  the same broken pipeline, that PSNR can be 41–46 dB while the
  recon is geometrically wrong.

Wagner explicitly notes in the 2023 paper that of 36 Mayo LDCT papers,
**only 4 use the projection data**; the other 32 work on the
reconstructed images. Implication for Track A: the pragmatic LDCT
pipeline is **reconstructed image DICOMs + Yu et al. noise simulation
+ forward project for the dual-domain training**, not helix2fan
rebinning. We can keep helix2fan as a research curiosity once the
file-order fix is verified; the Pentathlon's `mayo_ldct` track should
default to the image-domain path.

## 2026-05-23 — DD-UNet supervised L2 capacity saturates at c=32; c=64 overfits

Extended the supervised-L2 dual-domain U-Net capacity sweep:

| iter | unet_c | params | val_psnr | hr | notes |
|---:|---:|---:|---:|---:|---|
| 2 | 8  | 120 k  | 52.53 | 0.771 | undertrained |
| 1 | 16 | 466 k  | 54.25 | 0.812 | sweet (prev top) |
| 3 | 32 | 1.86 M | 54.94 | **0.826** | new top |
| 4 | 64 | 7.41 M | 53.82 | 0.802 | **overfits 400-phantom train** |

c=32 is the optimum. c=64 drops -0.024 hr — the 400-phantom train set
can't support 7.4 M params. Diminishing returns curve (+0.041 from
8→16, +0.014 from 16→32, **−0.024 from 32→64**).

**For the autoresearch agent**: do not search above unet_c=32 on the
400-phantom train set. To push past hr=0.826, the next lever is
**more training data** (dispatched as iter-5: c=32 + train_n=1600 =
4× more data, job 762044). If even that fails to break 0.826, the
val_n=20 metric noise is likely the floor.

## 2026-05-23 — Optuna TPE for LPD: subprocess timeout was 3600 s; bumped to 5400 s

First attempt at LPD TPE (job 762038, slug
`breast-ct-calibrated-tpe-lpd-search-20260523-01`) FAILED after trial 1:

- Trial 1 (seed = iter-3 winner): `I=10, hidden=64, ep=20, lr=5e-4,
  grad_clip=1.0` → hr=0.8204 on Q5000 (40 min wall — close to but
  inside the 3600 s subprocess cap).
- Trial 2 (TPE's first non-seed proposal): `I=12, hidden=64, ep=28,
  lr=8e-4, grad_clip=0.5` → killed by the `subprocess.run(timeout=3600)`
  in `learned_solver_search_agent.run_solver` at exactly 3600 s. Whole
  TPE study died (raises `TimeoutExpired`).

Two patches in `scripts/learned_solver_search_agent.py`:

1. Bumped subprocess timeout 3600 → 5400 s (env-overridable via
   `SEARCH_AGENT_SUBPROC_TIMEOUT_S`). Original cap was sized for
   ITNet/USwin/NAF/R2G; LPD's bigger trials need more room on Q5000.
2. Tightened LPD search bounds:
   - `lpd_iters` choices `[8, 10, 12]` → `[8, 10]` (I=12 is over budget)
   - `epochs` int `[15, 30]` → `[15, 25]` (ep=28 was over budget)

Resubmitted as job 762043 (slug becomes
`breast-ct-calibrated-tpe-lpd-search-20260523-02`). Note: Optuna
study is keyed on slug, so the new sbatch starts fresh; the −01 study
keeps trial 1 as a curiosity in `/cluster/maier/Agent4CT/optuna/*.db`
but is otherwise orphaned.

Also a minor observation: trial 1 reproduced iter-3 at hr=0.820 (vs
iter-3's recorded hr=0.829). The 0.01 gap is within val_n=20 noise +
Q5000 vs Q6000 cudnn nondeterminism. Not concerning.

## 2026-05-23 — Breast-trained DDPM checkpoints EXIST but the diff-recon TPE-seed-config gives hr=0

The hand-off correctly diagnosed that the earlier diff-recon TPE runs
on breast-CT used the wrong (demo-phantom) DDPM checkpoint. The
breast-trained checkpoints
(`/cluster/maier/Agent4CT/checkpoints/ddpm_breast_{,un}constrained_final.pt`,
3.86 MB each, dated 2026-05-20) DO exist; they were just not wired
into the SOLVERS dict.

Patched `learned_solver_search_agent.py` with two new entries
(`diffusion_recon_dcstep_{,un}constrained_breast`) and dispatched
20-iter TPE searches as jobs 762041 / 762042 with slugs

```
breast-ct-calibrated-tpe-diff-recon-dcstep-unconstrained-breast-search-20260523-01
breast-ct-calibrated-tpe-diff-recon-dcstep-constrained-breast-search-20260523-01
```

Both are running.  Early results (each first 2 trials):

| variant | trial 1 (seed) | trial 2 |
|---|---|---|
| unconstrained | hr=0.000  SSIM=0.431 | hr=0.000  SSIM=0.411 |
| constrained   | hr=0.000  SSIM=0.470 | hr=0.000  SSIM=0.350 |

The seed config is the **same** that won the demo-phantom TPE search
(DPS, 500 sample steps, eta=30, every=3, n_cg=20, warmup=25,
relax=1.0). Reuse-it-here gives **SSIM ≈ 0.4** — *far below baseline
FBP (0.957)*. The recon is structurally failing.

Possible causes (in order of probability):

1. **Breast DDPM checkpoint quality**. Same architecture & file size
   as the demo checkpoints; maybe trained on too little breast data /
   too few epochs to learn a useful prior. The recon's pixel scale
   may be off if the checkpoint expects μ-normalisation different from
   the breast data's intensity range.
2. **DPS-style guidance hyperparams don't transfer**. The demo phantoms
   are simple ellipses; the breast phantoms have ~ 100× more fine
   structure. The DPS step size that worked on ellipses may now be
   too large and overshoot.
3. **Initialisation issue**. `recon_init="fbp"` for the breast model
   may produce out-of-distribution inputs.

The TPE will explore the bounds and may find a non-seed config that
works. Updates to come. **If both TPEs end at hr=0**, the next move
is **retrain the breast DDPM** with more data / epochs — the current
checkpoint is the bottleneck, not the search space.

## 2026-05-23 — TV-iter L2 (unrolled-TV-GD supervised) is structurally bounded by FBP quality

Built `solver_tv_iterative_supervised.py` — K unrolled gradient-descent
steps on `½‖Rf − g‖² + λ TV(f)`, with per-iter learnable scalar `step_k`
and `λ_k`, initialised from FBP(noisy), trained MSE vs clean phantom on
breast-CT (400 train, 20 val). 20 trainable scalars total at K=10.

Three iters (`breast-ct-claude-agentic-tv-iterative-supervised-search-20260523-01`):

| iter | K  | step_init | λ_init  | val_psnr | hr   | training loss → |
|---:|---:|---:|---:|---:|---:|---|
| 1  | 10 | 1e-2      | 1e-3    | 30.07    | 0.00 | 0.0147 ⇨ 0.0145 |
| 2  | 10 | **1e-4**  | **1e-5**| 32.05    | 0.00 | 0.0146 ⇨ 0.0145 |
| 3  | 30 | 1e-4      | 1e-5    | 32.05    | 0.00 | 0.0146 ⇨ 0.0145 |

iter-1's `step=1e-2` was 100× too large for stable GD: the data-grad
overshoots, the optimiser compensated by growing late-iter λ's to 0.025
(over-smoothing), recon ended **9.7 dB worse than baseline FBP**.

iter-2's `step=1e-4` was conservative enough that the network learned to
do *almost nothing*: per-iter steps stayed near the init for k≥2, only
`step_0` drifted to ~2e-3. The recon ends *close to* FBP-init (val_ssim
0.954 vs baseline 0.957 — 0.3 % below). PSNR 32.05 dB still well below
baseline 39.74 dB.

iter-3's `K=30` (3× more iters) made **zero difference**: same val_score,
same loss trajectory, same learned scalar pattern. K≥2 iters are
effectively no-ops because the data-fidelity gradient at f≈FBP(g) is
≈0 (FBP is approximately R⁺, so R^T(R·FBP(g) − g) ≈ 0).

**Verdict**: with FBP init and hand-crafted smooth-TV gradient, this
architecture cannot exceed FBP quality on dense-view (128-view)
breast-CT. The data-fidelity term is saturated; the only signal a
supervised-L2 loss can give is "stay where you are". To beat FBP under
this skeleton one must either:

1. Replace `∇TV(f)` with a *learnable* prior (small CNN) → that's
   ITNet / Learned Primal-Dual, which we already have as
   `solver_learned_primal_dual.py` (hr=0.83 at I=10).
2. Drop the FBP init and start from zero — risky and slow, K iters
   must rediscover FBP first.
3. Add a learnable additive bias (per-iter offset) → effectively turns
   the algorithm into "FBP + light CNN", duplicating
   `solver_dual_ddomain_supervised.py` (hr=0.81) without its U-Net
   structure.

**TV-iter L2 deprioritised for breast-CT.** Worth keeping as a
sparse-view baseline where FBP itself is the bottleneck (the K iters
have actual work to do), not for dense-view.

## 2026-05-23 — Learned Primal-Dual saturates at I=10; pushing to I=15 hurts

Following the LPD iter-3 result (hr 0.829, I=10, hidden=64 — top of
breast-CT leaderboard), iter-5 tested whether more unrolled iterations
help. Config: I=15, hidden=64, lr=2e-4 (lowered from 5e-4 in iter-4 to
avoid the NaN that killed iter-4), grad_clip=0.5, 20 epochs, cosine
schedule. Job 761921.

| iter | I  | hidden | params  | epochs | lr     | val_psnr | hr     | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 10 | 64 | 875 k   | 20 | 5e-4 | **55.08** | **0.829** | breast-CT leaderboard top |
| 4 | 15 | 64 | 1312 k  | 20 | 5e-4 | — | — | **NaN** from epoch 1; cancelled |
| 5 | 15 | 64 | 1312 k  | 20 | 2e-4 | 52.48 | 0.769 | **train loss → 0 by ep 13; classic overfit** |

The drop from hr=0.829 → 0.769 with 1.5× more iterations is a clear
saturation signal: the I=10, hidden=64 model is already at its
expressivity sweet spot for breast-CT, and adding more capacity
(50% more parameters) overfits the 400-phantom train set rather than
extracting more signal. The epoch-1 loss spike to 831 M is a known
LPD initialisation transient (CNN proximal weights produce wild
divergence on the first batch); the cosine LR schedule rescues it,
but the recovered model still ends up worse than iter-3 because the
deeper unrolling has more parameters to overfit per training pair.

**For the autoresearch agent that picks up LPD**: do not search past
I=10 on the breast-CT 400-phantom set. Iter-3 (`I=10, hidden=64,
lr=5e-4, ep=20, cosine`) is the converged optimum. If the next move
is more capacity, the right axis is `hidden=128` *with* the existing
I=10 (modest +30% params), not more iterations. The other axis is
**more training data** — the staged breast-CT set has 3600 train
phantoms; iter-3 used 400.

## 2026-05-23 — NAF on dense-view breast-CT: structurally hr=0, even at 5× n_iter

Following the hand-off hypothesis "TPE n_iter cap (600) was compute
starvation", agentic NAF iter-1 (job 761922) tested `naf_n_iter=12000`
(20× the TPE cap) with `val_n=5`. Result: hr=0, SSIM=0.755,
PSNR=15.78 dB — **24 dB *below* baseline FBP** (39.61 dB). The
per-scene fit ran to completion in ~730 s with `last_loss ≈ 0.87`
(should be much closer to 0 for a converged fit).

The compute-starvation hypothesis is refuted: at 12 000 inner iters
NAF still can't beat FBP on dense-view (128-angle) breast-CT. The
issue is structural — NAF's coordinate-MLP with sin/cos positional
encoding has the wrong inductive bias for a dense-view fan-beam
reconstruction. The MLP's frequency-band partitioning is a useful
inductive bias when *views are missing* (NAF's intended sparse-CBCT
setting), but on a 128-view dense scan the FBP is already a strong
reconstruction and a coordinate MLP can't outperform a properly-tuned
back-projector + denoising chain.

NAF iter-2 (job 762023, in flight at hand-off time) tests `lr=1e-3`
(5× lower than iter-1) to check whether the high lr was causing
optimizer bouncing around the minimum. If hr stays at 0, **NAF should
be deprioritised on breast-CT** — it's the wrong architectural family
for this challenge. Worth keeping as a sparse-view baseline.

## 2026-05-22 — Breast-CT DD-UNet supervised L2 reaches PSNR 54.25 dB (hr 0.81)

Following the DD-BF supervised-L2 finding (next entry), ported the same
loss change to the U-Net dual-domain solver
(`solver_dual_ddomain_supervised.py`, 466 k params at c=16):

| width | params | val_ssim | val_psnr | hr |
|---|---:|---:|---:|---:|
| c=8  | ≈120 k | 0.9979 | 52.53 dB | 0.771 |
| c=16 | 466 k  | 0.9986 | 54.25 dB | 0.812 |

This is 14.5 dB above baseline FBP — saturation level for the val set.
The previous N2I-trained version of the same architecture at c=16
peaked at val_ssim 0.967 / val_psnr 38.0 dB / hr=0. **Loss is the
dominant lever; doubling U-Net width gains only +1.7 dB on top of the
supervised-L2 change.** Mark this for the pentathlon all-rounder: any
dual-domain solver competing on breast-CT or other dense-scan
challenges should default to MSE-vs-clean if clean targets are
available; reserve N2I for sparse-view or no-truth regimes.

### Naming convention (2026-05-22)

The N2I solvers now carry the training scheme explicitly in their
filename:

| Old name | New name | Loss |
|---|---|---|
| `solver_dual_ddomain.py`           | `solver_dual_ddomain_n2i.py`           | Noise2Inverse (self-sup) |
| `solver_dual_ddomain_bilateral.py` | `solver_dual_ddomain_bilateral_n2i.py` | Noise2Inverse (self-sup) |
| —                                  | `solver_dual_ddomain_supervised.py`           | Supervised L2 vs clean phantom |
| —                                  | `solver_dual_ddomain_bilateral_supervised.py` | Supervised L2 vs clean phantom |

The bilateral solvers also gained `proj_n_bf` / `img_n_bf` config keys
(Wagner §3.2 BF chain). On the supervised-L2 BF variant, n=3 BFs per
domain pushed hr from 0.214 (n=1) → 0.230 (n=1, bigger kernel) →
0.248 (n=3). The image-domain cascade differentiated into a
multi-scale stack (BF1 σ_x≈1.10, BF2 σ_x≈1.18, BF3 σ_x≈1.33); the
projection-domain stack stayed locked-symmetric (gradients too small
to break the identical init).

### Trainable Wu 2015 (`solver_wu_2015_trainable.py`) — 10-iter autoresearch loop

Built the variant 2026-05-22 and ran a 10-iteration agentic search
(slug `breast-ct-claude-agentic-wu-2015-l2-search-20260522-01`):

| iter | change vs prior | val_psnr | hr | observation |
|---|---|---:|---:|---|
| 1  | baseline cfg (lr=1e-2, ep=5)        | 36.65 | 0.000 | lr too high, killed high-freq bands |
| 2  | **lr=1e-3, ep=10**                   | **41.74** | **0.219** | **sweet spot — best of the search** |
| 3  | ep=20                                | 39.72 | 0.015 | overfit; train loss 0.016→1e-4 but val drops |
| 4  | ep=8                                 | 40.64 | 0.114 | undertrained |
| 5  | n_outer=3                            | 25.14 | 0.000 | collapse — blend went [1.87, 2.04, 1.03] |
| 6  | wd=1e-3 (AdamW)                      | 40.81 | 0.131 | wd too weak; trajectory ≈ no-wd |
| 7  | wd=1e-2                              | 40.65 | 0.114 | still too weak |
| 8  | L1 loss (`loss_base="l1"`)           | 40.81 | 0.131 | same band-runaway pattern |
| 9  | train_n=2000 (5× more data)          | 38.63 | 0.000 | data made overfit *worse* — band_scale[0]→53× |
| 10 | hard clamps on band_scale / blend / soft_thresh | 36.55 | 0.000 | optimizer pinned to clamp corners — new pathology |

**Ceiling**: hr ≈ 0.22 (iter-2). No further config or
architectural-clamp move broke past it. The pattern across iters:
**the optimisation landscape consistently pushes band_scale[1]
upward and bands 3-4 toward zero** — a textbook noise-suppression
solution that does well on the 400-phantom train set but fails on val
because the per-band amplification it learns is dataset-specific
overfit. Hard clamps don't help (corners become the new attractor);
more data makes it worse (the optimizer becomes more confident);
weight decay at reasonable magnitudes is too weak; L1 doesn't change
the trajectory.

What would likely break past hr=0.22:
- **Validation-based early stopping** with best-checkpoint restoration.
  Iter-2 happened to land in the sweet spot because epochs=10 was
  exactly right; a proper early-stop would generalise that.
- **Softmax-parametrised band weights** (sum constrained to a fixed
  total) instead of independent log-scales — removes the band-runaway
  degree of freedom by construction.
- **Hybrid Wu + BF tail** — Wu does aliasing-free + residual cleanup,
  a 6-param image-domain BF tail (supervised L2) cleans up remaining
  streaks. Likely lands near hr ≈ 0.30 based on the 6-param BF
  reference (hr=0.21).
- **Bigger algorithmic capacity** — make `wu_n_bands` a config knob
  that goes to 8 (the paper's value); learn per-pixel soft thresholds
  via a tiny CNN-prior; etc.

Even with the ceiling, the trainable Wu (10 params, hr=0.22) matches
the 6-param DD-BF supervised (hr=0.23) and the 18-param 3×3 BF stack
(hr=0.25). For a *classical* CT algorithm with end-to-end trainable
knobs and no neural net, that's a respectable result and proves the
machinery works.

## 2026-05-22 — Breast-CT DD-BF: Noise2Inverse is the bottleneck; supervised L2 + full 128 views unlocks hr ≈ 0.21

The dual-domain bilateral-filter (DD-BF) solver
(`solver_dual_ddomain_bilateral_n2i.py`, formerly
`solver_dual_ddomain_bilateral.py` — renamed 2026-05-22 to make the
training scheme explicit) had been stuck at hr=0 across all TPE and
agentic iterations on breast-CT (best val_ssim ≈ 0.957 ≤ baseline FBP
0.957). The U-Net dual-domain variant (`solver_dual_ddomain_n2i.py`,
formerly `solver_dual_ddomain.py`) had the same problem at every U-Net
width
tested (c=4, c=8, c=16): SSIM stuck just above baseline FBP, PSNR
*below* baseline by ~1.7 dB.

Root cause confirmed empirically (iters 1–5 on
`breast-ct-claude-agentic-dual-domain-{,bf-}search-20260521-01`): the
**Noise2Inverse split-view MSE** (`DualDomainPipeline.training_step`,
`ddssl_ldct/training.py`) feeds the model the FBP of one 64-view
half-set as a "target" for the other half-set. On dense breast scans
the half-view FBP has an irreducible noise floor, so minimising MSE
against it *encourages over-smoothing*. The bilateral filter's image-
domain spatial sigma grew 0.5 → 1.08 in 2 epochs (iter-1) and 0.3 →
0.87 in 5 epochs (iter-2), saturating the kernel and over-blurring
fibroglandular detail. Increasing U-Net width didn't help — the loss
was pulling all architectures the same way.

Fix: new `solver_dual_ddomain_bilateral_supervised.py` — full 128-view
forward pass, plain MSE against the *clean* phantom + non-negativity
penalty (using the existing `supervised_recon_loss`). Same 6-parameter
bilateral filter, same intensity calibration, same comparison panel.

Iter-1 result (slug
`breast-ct-claude-agentic-dual-domain-bf-l2-search-20260522-01`,
job 761594, 70 s, lme221):

| solver | params | val_ssim | val_psnr | headroom |
|---|---:|---:|---:|---:|
| baseline FBP | 0 | 0.957 | 39.74 dB | 0 |
| N2I DD-BF (best of 2 iters) | 6 | 0.957 | 37.50 dB | 0 |
| N2I DD-UNet (c=16, 5 ep) | 466 k | 0.967 | 38.01 dB | 0 |
| **Supervised-L2 DD-BF (this iter)** | **6** | **0.986** | **41.83 dB** | **0.21** |

A 6-parameter bilateral filter trained with the right loss beats a
466 k-parameter U-Net trained with the wrong loss by 3.8 dB PSNR on
the same dataset. Lesson: **loss formulation, not model capacity, was
the bottleneck for breast-CT dense-view denoising.** Mark this for the
all-rounder phase — the N2I assumption is good for sparse-view (where
the half-view FBP is so degraded the noise floor is below the signal)
but a poor fit for dense scans where FBP itself is already strong.

Caveat: supervised-L2 needs the clean phantom at train time, so this
variant is only fair against other supervised baselines (it cannot
play the "self-supervised" tag). Track separately in the dashboard.

## 2026-05-22 — Random / TPE search is NOT autoresearch (terminology fix)

Previous Agent4CT iterations conflated three distinct things by writing all
of them into `docs/runs/<slug>/...` with the same on-disk shape:

1. **Autoresearch (Karpathy-style)** — an *LLM* reads prior observations
   + literature, forms a hypothesis about *why* the last iter
   under/over-performed, proposes a *qualitatively different* next config
   (architecture / loss / optimiser / preprocessing) and writes a
   `rationale` that names the hypothesis. The slug carries
   `claude-agentic` (or another LLM family tag).
2. **TPE / Bayesian hyperparameter optimisation** — Optuna or similar
   samples the *same* parametric solver inside a *fixed* numeric bounding
   box, learning a surrogate over the box. No hypothesis, no architecture
   change, no qualitative jump. Slug should carry `tpe`, model field
   should be `optuna-tpe-*`.
3. **Random search** — uniform / log-uniform sampling inside the same
   box. Slug should carry `random-search`.

All three are *hyperparameter-tuning aids* for the agentic loop, but
**only #1 is the autoresearch loop itself**. Earlier scripts
(`scripts/tv_search_agent.py`, `scripts/learned_solver_search_agent.py`,
`scripts/tv_search_agent_standalone.py`, `scripts/record_demo_refs.py`)
called themselves "agentic" or "autoresearch" because they wrote into the
same `docs/runs/` schema; that wording is wrong and has been corrected.
Rule of thumb when reading any prior run: **if `agent` ends in
`-search` and `model` is `random-search` or starts with `optuna-`, the
run was a parameter sampler, not an autoresearch iteration.** The
dashboard groups them next to each other for comparison, which is fine
— just don't claim that "TPE found x" was an "agentic finding".

What an autoresearch iteration *must* include:
- A specific testable hypothesis in the `rationale` field, naming the
  expected mechanism (e.g. "img-domain BF σ ran away from 0.5 → 1.08 in
  iter 1, causing over-smoothing; cap with smaller kernel + lower lr").
- A change that the sampler could not have proposed by itself (a new
  loss term, a new architecture, an asymmetric hyperparameter constraint,
  a literature-cited prior).
- Inspection of the previous iter's `comparison.png` *by the agent*
  before the proposal (the agent should be able to name the artefact:
  radial blur, ringing, oversmoothing, streak, etc.).

If the proposal is "the same solver with the next sample in a box", it
should be filed as a TPE / random-search iteration, not an autoresearch
one.

## 2026-05-16 — B epochs=10 confirmed at stage: new high-water mark 0.6248

Followed up on B's "underfits at stage" finding by raising iter-base
epochs 8 → 10 (iter-147 at epochs=12 dropped to 0.5864 at iter scale,
so 10 is at or above the iter sweet spot). Ran a fresh B stage on
iter-146 (epochs=10). Stage hr=**0.6248**, up +0.46pp from the previous
B stage (0.6202 at epochs=8). Highest stage headroom on any agent.

Caveat: the stage sbatch auto-scales epochs to `max(base*2, 16)`, so
base=10 → 20 stage epochs vs base=8 → 16 stage epochs. The gain mixes
"iter-base config matters" and "more stage epochs". Operational
takeaway is clear: keep `epochs=10` for B going forward.

## 2026-05-16 — A and main iter scores are insensitive to weight_decay

After the iter-95 capacity-down failure, the next hypothesis was that
regularisation (wd-up) might close A and main's iter→stage gap. Tested:

| Agent | wd values tested | Samples | Iter hr range |
|---|---|---|---|
| A | 3e-5, 1e-4, 2e-4 | 5 | 0.6162–0.6167 (0.05pp) |
| main | 1e-5, 5e-5, 1e-4 | 5 | 0.6001–0.6005 (0.04pp) |

**Iter is essentially insensitive to wd** on both substrates. Doesn't
mean wd is neutral at stage scale — only iter. main stage v2 (wd=1e-4)
and a future A stage will test whether wd helps at scale.

## 2026-05-15 — capacity-down does NOT close A's iter-stage gap

Hypothesis from the first round of stages: A's -6.56pp gap was overfit due
to capacity (BF tail + 5 NAFNet blocks memorise the 400-case iter subset).
Tested by cutting capacity: `naf_n_bf` 10→6 + `naf_blocks` 5→4 (model
shrinks 0.050M → 0.040M params). Iter score dropped -1.39pp (expected if
overfit shrunk). **Then ran a second stage check on the capacity-down
config: stage hr DROPPED 0.5506 → 0.4991, gap GREW from -6.56pp to
-10.32pp.** Hypothesis falsified.

What we know now:
- The iter-stage gap on A is real but it is **not** "overfit due to
  capacity". Cutting capacity hurts both iter and stage.
- The gap must come from something else: maybe training-dynamics
  mismatch (12 epochs vs 6, larger val set), maybe optimiser/LR not
  tuned for longer training, maybe the BF tail's α parameters benefit
  disproportionately from longer training, maybe the iter-phase val
  set is genuinely easier than the stage val set.

Reverted A's solver to iter-86 KEEP base. Next probes for closing the
gap should look like **regularisation** (dropout, weight decay, schedule)
or **training-time fixes** (longer LR warmup, lower lr), not capacity.

## 2026-05-15 — iter vs stage gap signs differ by architecture

First four stage checks ran on DL-Sparse-View (with synthetic phantom data,
since real data isn't staged yet). Results:

| Agent | Architecture | Iter best | Stage hr (1h, 3× data, 2× epochs) | Gap | Direction |
|---|---|---:|---:|---:|---|
| main | NAFNet + SWA + BF tail | 0.6144 | 0.5928 | -2.16pp | overfits |
| A | NAFNet + BF tail | 0.6162 | 0.5506 | -6.56pp | severely overfits |
| **B** | resnet + AdamW + wd=5e-5 + batchnorm | 0.6120 | **0.6202** | **+0.82pp** | **underfits in iter, scales positively** |
| C | resnet + Adam + wd=0 + batchnorm + aug | 0.6102 | 0.5787 | -3.15pp | overfits |

**Real capability ranking at stage scale: B > main > C > A** (opposite of
the iter-best ranking). The iter phase's 400-case subset is small enough
to be memorised by NAFNet+BF families; the narrower resnet is undertrained
at iter scale.

**B vs C divergence is informative** — both resnet+batchnorm, but B
(AdamW + wd=5e-5) underfits while C (Adam + wd=0 + augs) overfits. The
augmentations in C don't add data diversity for sparse-view CT in a useful
way; Adam wd=0 removes the only weight regularisation. Cross-port `B`'s
optimiser to C (iter-98 in flight as of writing).

**Strategy implications:**
- Stop probing sub-pp KEEPs on B/C's 5-min iters — that's under-fitting
  noise, not signal.
- For A/main: iter probes should *reduce* capacity or add regularisation,
  not push capacity up.
- For B/C: iter probes should test things that benefit from more data
  (augmentations are not those things on this dataset).
- Stages are the only reliable signal until staged HDF5 data exists.

## 2026-05-15 — same-config variance on the 5-min iter substrate

Repeated same-config runs to characterise noise:

- **B (resnet)**: 6 same-config samples spanned 0.5774–0.6120, mean ≈0.600,
  3.5pp range. The iter-135 +0.14pp "KEEP" and iter-88 +0.22pp "KEEP" both
  fall inside this noise window.
- **main (NAFNet)**: 3 same-config samples spanned 0.6001–0.6144, mean ≈0.605,
  1.4pp range. The iter-102 best 0.6144 is the upper tail.

Likely cause: BatchNorm with `batch_size=1` + CUDA non-determinism.
`torch.manual_seed(42)` is fixed but BN running stats accumulate
differently across reorderings. Adding `torch.use_deterministic_algorithms(True)`
+ `CUBLAS_WORKSPACE_CONFIG=:4096:8` would reduce this; not enabled yet.

**Action:** treat any iter-phase KEEP < 0.5pp on B as variance. On
main/A/C the noise is tighter (probably <0.5pp) but still relevant.

## 2026-05-15 — silent substrate drift from un-reverted DISCARDs

When a DISCARD iter changes a CONFIG knob but the next iter doesn't
explicitly revert it, the substrate drifts. The "best so far" baseline is
only the headroom at the recorded KEEP iter, but the **current solver
state** can quietly be worse, masking opportunities.

Confirmed case: C iter-76 found a +0.29pp KEEP just by reverting
lr 8.5e-5→8.0e-5 — a change introduced in iter-70's DISCARD that was
never reverted. The solver had silently run below its iter-46 KEEP base
for ~30 iters.

**Action:** every ~15–20 iters without a KEEP, `diff` the current CONFIG
against the last KEEP iter's solver snapshot in
`docs/runs/<slug>/iterations/iter-NNNN/solver.py.txt`. Any knob that
drifted across a DISCARD without explicit revert is a candidate for
restoration. The current iter loop does not auto-revert.

## 2026-05-15 — data download status (see `data/README.md` for current state)

Live state of which challenge data is on the cluster lives in
[`data/README.md`](../data/README.md) §"Where the raw data lives on the
cluster". Summary at time of writing: CT-MAR + DL-Spectral are on disk,
Mayo Wagner subset is downloading slowly, DL-Sparse-View and TrueCT are
CodaLab-gated. **`stage_h5()` is not implemented for any fetcher**, so
the harness still trains on synthetic phantoms (`ddssl_ldct/phantoms.py`).
