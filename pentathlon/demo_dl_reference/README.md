# Demo Reference Implementations — Results Summary

**Run**: `demo-dl-reference-20260515-01`
**Dashboard**: https://akmaier.github.io/Agent4CT/dashboard.html

> ⚖️ **Comparison conventions** — every solver in this directory must
> follow the rules in [CONVENTIONS.md](CONVENTIONS.md): same val-set
> seeds (`cfg["seed"]+1000`), prediction clamped to `[0, display_max]`
> before metric computation, `data_range = display_max-display_min`
> passed explicitly to PSNR/SSIM. Past `demo-dl-<solver>-search-*`
> entries that pre-date this enforcement are kept for historical
> visibility but are superseded by the **`demo-fair-*` slugs**
> launched 2026-05-17 (jobs 761199-761203) under the corrected
> convention.

## Random vs Optuna-TPE search (2026-05-17)

Every fair re-run is now duplicated under two samplers so the dashboard
can compare them directly:

| Slug prefix | Sampler | Description |
|---|---|---|
| `demo-fair-<solver>-search-*` | `random` | Independent uniform/log samples; embarrassingly parallel; ~64 % chance of landing a top-5 % config in 20 iters (Bergstra & Bengio 2012). |
| `demo-fair-tpe-<solver>-search-*` | `optuna-tpe` | Tree-structured Parzen Estimator: 5 random startup trials + 15 acquisition trials guided by prior outcomes. SQLite-backed study at `/cluster/maier/Agent4CT/optuna/<slug>.db` so Slurm restarts resume. ~3-5× more sample efficient than random for shallow-optima spaces. |

Both samplers draw from **the identical search-space** (`SOLVERS[<name>]
["space"]` in `scripts/learned_solver_search_agent.py`). The dashboard
chart-group key `demo-fair` collects both into the same chart so the
random-vs-TPE delta is visually obvious.

**Why 20 iterations?** Hits a sweet spot:
- TPE at 20 iters reaches ~95 % of best-attainable headroom on a 5-8 dim
  search space (rule of thumb: ~10-15 for 90 %, ~20-25 for 95 %).
- Random at 20 iters has ~64 % chance to land a top-5 % config (only
  ~30-50 random trials needed for 90 %); we accept that lower sample
  efficiency in exchange for full parallelism.
- Cluster wall budget: solvers take 5-40 min per iter; 20 × 40 min ≈
  13 h fits inside the largest sbatch wall (NAF's 14 h). Doubling to
  40 iters would push some solvers past their sbatch wall.

The TPE jobs are submitted with Slurm `--dependency=afterany:<random-job-id>`
so they queue but do not start until the matching random run finishes,
freeing the GPU for the same solver and letting the per-study SQLite
file's prior history (if any) accumulate cleanly.

## 2026-05-17 fair-run results

All 11 solvers re-evaluated under the strict
[CONVENTIONS.md](CONVENTIONS.md) (same val seeds, output clamped to
`[0, display_max]` before metrics, `data_range = display_max −
display_min`) under both random search and Optuna-TPE. Live on the
dashboard under the `demo-fair` chart group.

### Random vs TPE delta (best headroom across 20 iters)

| Solver | Random | **TPE** | Δ (TPE − random) |
|---|---:|---:|---:|
| ItNet v3 (end-to-end) | 0.8187 | **0.8378** 🏆 | +0.0191 |
| U-Swin (transformer) | 0.8090 | 0.8180 | +0.0090 |
| TV (tuned) | 0.6180 | 0.6793 | **+0.0613** |
| ItNet v2 (pre-train only) | **0.6330** | 0.5761 | −0.0569 |
| DD Bilateral (Wagner 2022) | 0.6089 | 0.6106 | +0.0017 |
| DD U-Net (Wagner 2023) | 0.6068 | 0.6019 | −0.0049 |
| Hammernik 2017 (variational net) | 0.5924† | 0.6007 | +0.0083 |
| Hammernik-VN (MRI VN port) | **0.6061** | 0.5961 | −0.0100 |
| NAF (per-scene INR) | **0.5419** | 0.5354 | −0.0065 |
| Wu 2015 (classical) | 0.4109 | 0.4123 | +0.0014 |
| R2Gaussian (per-scene GS) | 0.3589 | 0.4014 | +0.0425 |
| Diffusion recon (random / TPE) | 0.0000 | 0.0000 | broken without DC-step — see "Outstanding" below |
| **Diffusion recon (Claude agentic, 20 iters)** | — | **0.5712** | post-DC-step fix; see [agentic search README](../../docs/runs/demo-fair-claude-diffusion-recon-search-20260518-01/README.md) |
| **RAM zero-shot (Claude agentic, 20 iters)** | — | **0.5938** | Terris 2025 foundational model + PyroNN deepinv adapter; SSIM = 0.934 (best in table); see [agentic search README](../../docs/runs/demo-fair-claude-ram-search-20260518-01/README.md) |

† Hammernik 2017 random only had 4/20 iters survive even at
`batch_size=2`; TPE got 20/20 by side-stepping OOM-prone configs.

**Take-aways from the random-vs-TPE comparison:**
1. TPE wins on **7 / 11** solvers, with the biggest gain on TV (+0.06)
   and R2Gaussian (+0.04).
2. Random wins on 4 solvers, with the biggest drop on ItNet v2
   (−0.057). The ItNet v2 random search happened to hit a lucky
   `residual=False, alpha=0.038` config in 1 of 20 trials that TPE's
   first 5 startup-random trials missed; with 15 acquisition trials
   TPE couldn't find that exact config back.
3. TPE most useful for **noisy / sparse landscapes** (per-scene fits
   like R2Gaussian where most random trials fail; TV where the
   `lambda × lr × clip` interactions matter).
4. TPE less useful for **bimodal / shallow landscapes** (ItNet v2's
   residual-learning toggle creates two near-equivalent basins; TPE
   commits to one early).
5. New leader: **ItNet v3 tuned with TPE at hr = 0.8378**, params
   2.5 M.

### Fair vs legacy buggy runs

| Solver | Old (buggy) | Fair (random) | Fair (TPE) | Δ ssim under fix |
|---|---:|---:|---:|---:|
| U-Swin | hr 0.8103, SSIM 0.5433 | hr 0.8090, SSIM 0.7817 | hr 0.8180 | **SSIM +0.238** (clamp fix) |
| ItNet v2 | hr 0.4806 | hr 0.6330 | hr 0.5761 | hr **+0.152** (clamp fix) |
| ItNet v3 | hr 0.8215 | hr 0.8187 | hr 0.8378 | hr ≈ flat then +0.016 with TPE |
| TV (tuned) | hr 0.6033 | hr 0.6180 | hr 0.6793 | hr **+0.076** (TPE big win) |
| DD U-Net | hr 0.5868 | hr 0.6068 | hr 0.6019 | hr +0.020 |
| DD Bilateral | hr 0.6095 (fixed-config) | hr 0.6089 | hr 0.6106 | tuned ≈ fixed |
| Hammernik 2017 | hr 0.6113 | hr 0.5924 | hr 0.6007 | hr −0.011 (the un-clamped run was overfitting overshoot to look better) |
| Hammernik-VN | hr 0.6113 | hr 0.6061 | hr 0.5961 | flat |
| Wu 2015 | hr 0.4101 | hr 0.4109 | hr 0.4123 | flat |
| R2Gaussian | hr 0.5053 (partial) | hr 0.3589 | hr 0.4014 | hr **−0.10** under fair; investigating |
| NAF | hr 0.0000 (broken) | hr 0.5419 | hr 0.5354 | now actually works |

The biggest clamp-fix beneficiaries: **ItNet v2 (+0.15 hr)** and
**U-Swin (+0.24 SSIM)** — both produced predictions that exceeded
`display_max` in the old runs and got dinged by the data-range-mismatch
SSIM. NAF and Hammernik 2017 each regressed slightly; the new fair
numbers reflect the actually-bounded prediction, which is what the
challenge metric expects.

## Diffusion-recon: unblocked by DC-step + Claude-agentic search

### The collapse (random / TPE attempts)

8 separate sampling-search attempts (random + TPE × constrained +
unconstrained DDPM × v1 + v2 grad-normalised samplers) all collapsed to
hr=0.0 with the output saturating to uniform display_max white. The
DDPM training itself looked healthy (final val ε-loss ≈ 0.0028 on
held-out phantoms — within standard DDPM ranges). The unconditional
DDIM samples (job 761258) confirmed both checkpoints produced
phantom-like images, isolating the bug to the DPS guidance trajectory.

### The fix: Resample-style DC-step

A periodic hard projection toward the data-fidelity manifold (a few CG
steps of `min ‖A·x_μ − y‖²` on the predicted-clean image, then re-noise
back to the current diffusion time) unblocked the collapse on the very
first attempt (iter 1: hr=0.4689). See `solver_diffusion_recon.py`'s
`dc_step_cg` and the `sample_guided` loop.

### Claude-driven agentic search (20 iters)

A separate 20-iteration search where Claude proposed each next
configuration after reading the prior iter's `result.json` and
`comparison.png`. The trajectory and final winning config live in
[`docs/runs/demo-fair-claude-diffusion-recon-search-20260518-01/`](../../docs/runs/demo-fair-claude-diffusion-recon-search-20260518-01/README.md).

**Final: iter 16, headroom = 0.5712, SSIM = 0.6495.**

| axis | won at | ceiling | note |
|---|---|---|---|
| `recon_dcstep_n_cg`  | **20** | 40 | over-projects against noisy sinogram |
| `recon_eta`          | **30** | 100 | over-pulls DPS step when projection is strong |
| `recon_dcstep_every` | **3**  | 2 | too-frequent ≈ same failure mode as high n_cg |
| `recon_sample_steps` | **500** | (200 worse) | longer trajectory wins |
| `recon_dcstep_relax` | **1.0** | (lower worse) | full hard projection wins given enough CG |
| `recon_dcstep_warmup`| **25** | 5 worse, 10 ok | DC needs the image to be partly denoised first |
| `recon_mode`         | **dps** | mcg worse | MCG's FBP-pseudoinverse blurs the gradient |
| DDPM ckpt            | **unconstrained (2000 phantoms)** | constrained (200) worse | constrained train set too narrow |

Diffusion-recon now sits between Hammernik 2017 (hr 0.6007) and NAF (hr
0.5354). Still well behind the supervised end-to-end leaders (ItNet v3
hr 0.8378, U-Swin hr 0.8180) but no longer broken.

## Fair re-run plan (2026-05-17)

A bug audit found two sources of incomparability across older runs:

1. Some solvers used different seed offsets for the val phantoms
   (mostly `cfg["seed"]+1000` but a few drifted to `+10000` or used
   the global `torch.manual_seed`).
2. Some solvers did not clamp negative or `>display_max` pixel values
   in the prediction before metric computation, biasing SSIM/PSNR/RMSE
   either way.

Both are now codified in [CONVENTIONS.md](CONVENTIONS.md) and every
solver has been patched. The dashboard now hosts both:

| Old slug (buggy) | Fair re-run slug (2026-05-17) | Job |
|---|---|---|
| `demo-dl-uswin-search-20260516-01` | `demo-fair-uswin-search-20260517-01` | 761199 |
| `demo-dl-naf-search-20260516-02` | `demo-fair-naf-search-20260517-01` | 761200 |
| `demo-dl-r2gaussian-search-20260516-01` | `demo-fair-r2gaussian-search-20260517-01` | 761201 |
| `demo-dl-hammernik-search-20260516-02` | `demo-fair-hammernik-search-20260517-01` | 761202 |
| `demo-dl-hammernik-vn-search-20260516-02` | `demo-fair-hammernik-vn-search-20260517-01` | 761203 |
| `demo-dl-itnet-v2-search-20260516-01` | `demo-fair-itnet-v2-search-20260517-01` | 761204 |
| `demo-dl-itnet-v3-search-20260516-02` | `demo-fair-itnet-v3-search-20260517-01` | 761205 |
| `demo-dl-dual_domain-20260515-01` | `demo-fair-dual-domain-search-20260517-01` | 761206 |
| (DD-Bilateral: no prior search slug; first official run) | `demo-fair-dual-domain-bf-search-20260517-01` | 761207 |
| `demo-dl-tv-search-20260515-01` | `demo-fair-tv-search-20260517-01` | 761208 |
| `demo-dl-wu-search-20260516-01` | `demo-fair-wu-search-20260517-01` | 761209 |

Unaffected (their old slugs remain the official leaderboard):

- FBP baseline (`solver_fbp_baseline.py` — no learnable knobs; convention-compliant by default)
- TV iterative reference (`solver_tv_iterative.py` — single-config, no search)
- ItNet v1 (deprecated, kept for the failed-baseline narrative only)

The original-vs-fair deltas will be summarised in section "Fair
re-runs vs old runs" once the 761199-761203 jobs finish.

---

## Implemented Reference Methods

All methods run on **synthetic random-ellipse phantoms** (stand-in for real AAPM DL-Sparse-View data).

### 1. FBP Baseline (`solver_fbp_baseline.py`)
- **Method**: Pure PYRO-NN ramp-filtered FBP, no learning
- **Purpose**: Establishes headroom=0 reference point
- **Result**: 
  - SSIM: 0.4454
  - PSNR: 11.14 dB
  - RMSE: 0.01387
  - **Headroom: 0.0000** (by definition)
- **Time**: ~3 seconds (no training)

### 2. TV-Regularized Iterative (`solver_tv_iterative.py`)
- **Method**: Gradient descent on `0.5*||Rf-g||² + λ*TV(f)`
- **Purpose**: Classical model-based iterative reconstruction (MBIR)
- **Status**: First run diverged (LR=0.5 too high → RMSE=0.20, negative PSNR). Fixed with LR=0.01, λ=0.001, 200 iters, adaptive decay.
- **Result** (Job 760895):
  - SSIM: 0.1281
  - PSNR: 13.71 dB
  - RMSE: 0.01032
  - **Headroom: 0.2562** ← Solid improvement over FBP!
- **Time**: ~57 seconds

### 3. Dual-Domain Denoising with U-Nets (`solver_dual_ddomain.py`)
- **Method**: Wagner et al. 2023 — learned denoisers in projection + image domain, Noise2Inverse self-supervision
- **Architecture**: SmallUNet(c=16) in both domains
- **Training**: 8 epochs, Adam, lr=1e-3, 400 samples
- **Result**: 
  - SSIM: 0.3055
  - PSNR: 18.74 dB
  - RMSE: 0.00578
  - **Headroom: 0.5831** ← Best so far
  - Params: 0.47M
- **Time**: ~175 seconds

### 4. Dual-Domain Denoising with Bilateral Filters (`solver_dual_ddomain_bilateral.py`)
- **Method**: Wagner et al. 2022 — trainable bilateral filters (4 params each) in both domains
- **Architecture**: `TrainableBilateralFilter2d` (σx, σy, σr learnable) in projection + image domain
- **Training**: Same Noise2Inverse self-supervision as U-Net variant
- **Purpose**: Ultra-low-parameter alternative to U-Nets. Wagner showed bilateral filters achieve **97% of U-Net SSIM with only 8 parameters** (4 + 4) on abdomen CT.
- **Result** (Job 761109):
  - SSIM: **0.3071**
  - PSNR: 18.78 dB
  - RMSE: 0.00576
  - **Headroom: 0.6095** ← **NEW BEST!**
  - Params: **6** (4 proj + 2? Let me check... actually 3+3=6)
- **Time**: ~262 seconds (train_n=400, epochs=20)
- **Why bilateral filters?**:
  - Only **6 parameters** vs 470K for U-Nets
  - Trains in similar time but is far more interpretable
  - Physically meaningful parameters (σr = edge preservation, σx/y = spatial smoothing)
  - Achieves competitive or better results than U-Net variant

### 5. ItNet-Style Iterative (`solver_itnet.py`) — v1 FAILED
- **Method**: Sidky 2022 winner approach — pretrained U-Net + iterative data consistency
- **Architecture**: SmallUNet(c=16), 5 iterations, fixed α=0.1
- **Training**: 20 epochs pre-training on (FBP, truth)
- **Result**:
  - SSIM: 0.1369, PSNR: 10.70 dB, RMSE: 0.0146
  - **Headroom: 0.0000** ← WORSE than FBP!
- **Root causes**:
  1. DC step α=0.1 is **way too large** — causes divergence (>10x too big)
  2. No learnable step size — cannot adapt to data
  3. Pre-training loss→0 means denoiser learned **identity mapping**
  4. Evaluated against FBP reference, not truth phantom
- **Time**: ~129 seconds

### 6. ItNet-Style v2 (`solver_itnet_v2.py`) — FIXES APPLIED
- **Fixes**:
  1. α=0.01 (10x smaller), **learnable** via softplus parameterization
  2. Residual learning: predict (truth - fbp) instead of full image
  3. Early stopping (patience=3) prevents identity collapse
  4. Evaluate against **truth phantom** (not FBP)
  5. Show residual error map in comparison figure

---

### 7. Wu 2015 Classical FBP (`solver_wu_2015.py`)
- **Method**: Wu, Maier, Yang, Fahrig 2015 — radius-dependent frequency-split
  ramp filter (aliasing-free FBP) + feature-preserving symmetric
  motion-compensated sinogram interpolation, 2 outer residual-restoration
  iterations with soft thresholding. No learning. See
  [literature/wu_2015_sparse_view_fbp.md](../../literature/wu_2015_sparse_view_fbp.md).
- **Default architecture**: 4 triangular frequency bands (paper uses 8), ±5-pixel
  symmetric motion search with ±2-pixel L1 window, soft threshold 0.0015 μ.
- **Result** (Job 761107):
  - SSIM: 0.1285
  - PSNR: 14.74 dB
  - RMSE: 0.00916
  - **Headroom: 0.3786** ← strongest classical baseline at the default config
  - Params: 0 (no learning)
- **Time**: ~1.1 seconds — by far the cheapest non-trivial method.
- **Default hyperparameters** (`solver_wu_2015.py`):
  ```python
  wu_n_bands       = 4        # paper uses 8; 4 keeps cost down with ~no quality drop
  wu_n_outer       = 2        # restoration iterations (paper: 2–3)
  wu_motion_range  = 5        # ±pixels for symmetric motion search
  wu_motion_window = 2        # ±pixels for L1 windowed patch
  wu_soft_thresh   = 0.0015   # soft threshold on residual reco (μ mm⁻¹)
  ```

**Take-aways**
- Cheapest non-trivial method by a wide margin (1.1 s vs 24–462 s).
- Beats TV iterative *at fixed hyperparams* by +0.12 headroom; TV reaches
  0.60 only after random search.
- Useful diagnostic: any learned method that beats FBP (0.0) but not Wu
  2015 (0.38) is mostly re-discovering sinogram-interpolation tricks the
  classical algorithm gets for free.

---

## Parameter Search Results (20 iterations each)

### TV Hyperparameter Search (`demo-dl-tv-search-20260515-01`)

20-iteration random search over:
- `tv_lambda` [1e-4, 1e-2]
- `tv_iterations` [50, 500]
- `tv_lr` [1e-3, 1e-1]
- `tv_clip_max` [0.03, 0.08]
- `tv_decay` [0.0, 0.05]

#### Best Results

| Metric | Iter-0009 (best headroom) | Iter-0017 (best SSIM) |
|--------|---------------------------|----------------------|
| **SSIM** | **0.4691** | **0.4804** |
| **PSNR** | 18.64 dB | 15.38 dB |
| **Headroom** | **0.6033** | 0.4223 |

#### Best Parameters (headroom)

```python
tv_lambda     = 0.003723
tv_iterations = 408
tv_lr         = 0.041828
tv_clip_max   = 0.071431
tv_decay      = 0.046289
```

Training budget: **~3 min** (fits 5-min wall-clock limit comfortably).

### Dual-Domain Hyperparameter Search (`demo-dl-dual_domain-20260515-01`)

20-iteration random search over:
- `epochs` [3, 8]
- `lr` [1e-4, 5e-3]
- `batch_size` [1, 4]
- `unet_c` [8, 24]

#### Best Results

| Metric | Value |
|--------|-------|
| **SSIM** | **0.2812** |
| **PSNR** | 18.78 dB |
| **Headroom** | **0.5868** |

#### Best Parameters

```python
epochs      = 6
lr          = 0.000407
batch_size  = 2
unet_c      = 19
train_n     = 200
```

Architecture: Two **SmallUNet(c=19)** denoisers (projection + image domain),
trained end-to-end via Noise2Inverse self-supervision.
Parameters: **0.66 M** (×2 denoisers). Training time: **~101 s**.

### ItNet Hyperparameter Search (`demo-dl-itnet-20260515-01`)

20-iteration random search over:
- `pretrain_epochs` [3, 8]
- `pretrain_lr` [1e-4, 5e-3]
- `itnet_k` [3, 8]
- `unet_c` [8, 24]
- `itnet_alpha_init` [1e-3, 5e-2]
- `residual_learning` {True, False}

#### Best Results

| Metric | Value |
|--------|-------|
| **SSIM** | **0.3296** |
| **PSNR** | — |
| **Headroom** | **0.5432** |

#### Architecture (v2)

| Component | Details |
|-----------|---------|
| Denoiser | SmallUNet(c=11) — 3 levels, **0.11 M params** |
| Unrolled iterations | k = 6 |
| DC weight | α_init = 0.027 (learnable via softplus) |
| Residual learning | **False** (performed better than True) |

Training paradigm: **Pre-train denoiser only** on (FBP, truth) pairs, no
end-to-end unrolled training.

### ItNet v2 Hyperparameter Search (`demo-dl-itnet-v2-search-20260516-01`)

20-iteration random search over:
- `pretrain_epochs` [3, 8]
- `pretrain_lr` log[1e-4, 5e-3]
- `itnet_k` [3, 8]
- `itnet_alpha_init` log[1e-3, 5e-2]
- `residual_learning` {True, False}

#### Best Results (iter-0012)

| Metric | Value |
|--------|-------|
| **SSIM** | 0.3583 |
| **Headroom** | **0.4806** |

#### Best Parameters

```python
pretrain_epochs   = 6
pretrain_lr       = 0.000292
itnet_k           = 3
itnet_alpha_init  = 0.0378
residual_learning = False
```

Half the iters in this v2 sweep collapsed to hr=0 (divergent α / non-residual
configs), so the effective search budget was ~10 working points — the
earlier `demo-dl-itnet-20260515-01` run found a better hr=0.5432 at a
similar config; v2 is not a robust architecture without tight α and k tuning.

### ItNet v3 Hyperparameter Search (`demo-dl-itnet-v3-search-20260516-02`)

20-iteration random search over:
- `epochs` [5, 15]
- `lr` log[1e-4, 2e-3]
- `batch_size` {10, 20, 40}
- `unet_c` {8, 12, 16}
- `itnet_k` {2, 3, 4}
- `alpha_init` log[1e-3, 1e-2]

#### Best Results (iter-0009)

| Metric | Value |
|--------|-------|
| **SSIM** | **0.6933** |
| **Headroom** | **0.8215** 🏆 |

#### Best Parameters

```python
epochs       = 10
lr           = 0.000421
batch_size   = 10
unet_c       = 12        # 5-level U-Net, 2.5 M params
itnet_k      = 3         # unrolled iterations
alpha_init   = 0.00910   # learnable via softplus
```

Top 3 iters: hr ∈ [0.7961, 0.8087, **0.8215**] — all `unet_c=12, k∈{3,4},
α≈3e-3 → 9e-3, lr≈4e-4 → 12e-4`. **End-to-end training with TIED weights
is decisive** — v3 jumps +0.34 headroom over v2's tuned best (0.4806) by
gradient-flow through the full unrolled loop instead of pre-training the
denoiser only. 2 of 20 iters failed (OOM at `batch=40, unet_c=16, k=4`).

### Hammernik 2017 Variational Network Hyperparameter Search (`demo-dl-hammernik-search-20260516-02`)

20-iteration random search over:
- `epochs` [10, 30]
- `lr` log[1e-4, 2e-3]
- `vn_T` {3, 5, 7}
- `vn_n_filters` {16, 24, 32}
- `vn_kernel` {7, 9, 11, 13}
- `vn_lambda_init` log[1e-4, 1e-2]

#### Best Results (iter-0018)

| Metric | Value |
|--------|-------|
| **SSIM** | 0.3313 |
| **Headroom** | **0.5263** |

#### Best Parameters

```python
epochs           = 15
lr               = 0.000121
vn_T             = 5         # paper's default
vn_n_filters     = 16        # paper used 24
vn_kernel        = 13        # matches the paper's optimum
vn_lambda_init   = 0.00213
```

Headroom-wise the search picked `kernel=13` (the paper's empirical best
filter size), but the overall best (0.5263) sits ~0.001 below the
unsearched reference run (0.5278) — so the default `(T=5, N_k=24, k=11,
λ=1e-3, lr=5e-4, epochs=20)` is already near-optimal on this synthetic
phantom geometry. 1 of 20 iters failed (OOM at `vn_T=7, vn_n_filters=32`).

### Wu 2015 Hyperparameter Search (`demo-dl-wu-search-20260516-01`)

20-iteration random search over:
- `wu_n_bands` ∈ {4, 6, 8, 12}
- `wu_n_outer` ∈ {1, 2, 3}
- `wu_motion_range` ∈ {3, 5, 8, 12}
- `wu_motion_window` ∈ {1, 2, 4}
- `wu_soft_thresh` log[5e-4, 5e-3]

#### Best Results (iter-0013)

| Metric | Value |
|--------|-------|
| **SSIM** | 0.1797 |
| **Headroom** | **0.4101** |

#### Best Parameters

```python
wu_n_bands       = 8        # matches the paper
wu_n_outer       = 2        # paper's lower bound
wu_motion_range  = 5        # back-of-envelope geometry guess held up
wu_motion_window = 1        # tighter than the ±2 default
wu_soft_thresh   = 0.00438  # ~3× the 0.0015 default
```

Headroom spread across 20 iters: **0.341 → 0.410** — algorithm is robust
to hyperparameters. Improvement over fixed defaults: **+0.032 headroom
(8.4 %)** at ~33 s of total cluster wall time. Patterns: high
`soft_thresh` (≈4e-3) consistently beats the default; `n_outer = 3`
clustered in the worst iters (over-iteration injects more noise than
detail); `motion_window = 1` appears in 4 of the top 5; `n_bands` is the
least sensitive knob.

---

## Head-to-Head Comparison

| Solver | Best SSIM | Best Headroom | Params | Train Time | vs TV Δ HR |
|--------|-----------|---------------|--------|------------|-----------|
| **ItNet v3 (tuned, end-to-end)** | **0.6933** | **0.8215** 🏆 | 2.5 M | ~30 s | +0.2182 |
| Dual-Domain Bilateral | 0.3071 | 0.6095 | **6** | ~262 s | +0.0062 |
| TV (tuned) | 0.4804 | 0.6033 | 0 | ~3 min | — |
| Dual-Domain U-Net (tuned) | 0.2812 | 0.5868 | 0.66 M | ~101 s | −0.0165 |
| Dual-Domain baseline | 0.3055 | 0.5831 | 0.47 M | ~462 s | — |
| ItNet v2 (tuned, 20260515) | 0.3296 | 0.5432 | 0.11 M | ~24 s | −0.0601 |
| Hammernik VN (default) | 0.2563 | 0.5278 | **18 k** | ~278 s | −0.0755 |
| Hammernik VN (tuned) | 0.3313 | 0.5263 | 12 k | ~280 s | −0.0770 |
| ItNet v2 (tuned, 20260516) | 0.3583 | 0.4806 | 0.23 M | ~25 s | −0.1227 |
| **Wu 2015 (tuned)** | 0.1797 | 0.4101 | **0** | **~1.5 s** | −0.1932 |
| Wu 2015 (default) | 0.1285 | 0.3786 | 0 | ~1.1 s | −0.2247 |
| TV iterative baseline | 0.4454 | 0.2562 | 0 | ~57 s | — |
| FBP baseline | 0.4454 | 0.0000 | 0 | ~3 s | — |

---

## Key Findings

### On Synthetic Phantoms:
| Method | Headroom | SSIM | Notes |
|--------|---------:|-----:|-------|
| FBP | 0.0000 | 0.4454 | Baseline (defines headroom=0) |
| TV iterative baseline | 0.2562 | 0.1281 | Classical MBIR |
| Wu 2015 (default) | 0.3786 | 0.1285 | Strongest classical, ~1 s wall |
| Wu 2015 (tuned) | 0.4101 | 0.1797 | After 20-iter random search |
| ItNet v2 (tuned) | 0.5432 | 0.3296 | Pre-train only, not end-to-end |
| Hammernik VN (default) | 0.5278 | 0.2563 | 18 k params, no pre-training |
| Hammernik VN (tuned) | 0.5263 | 0.3313 | Search confirmed paper's k=13 |
| Dual-Domain baseline | 0.5831 | 0.3055 | Fixed hyperparameters |
| Dual-Domain (tuned) | 0.5868 | 0.2812 | Searched hyperparameters |
| **TV (tuned)** | 0.6033 | **0.4691** | Best classical |
| **Dual-Domain Bilateral** | 0.6095 | 0.3071 | **6 params** |
| **ItNet v3 (tuned, end-to-end)** | **0.8215** 🏆 | **0.6933** 🏆 | 2.5 M params, k=3, tied weights |

### Lessons:
1. **End-to-end unrolling is the decisive factor** — ItNet v3 (tied-weight 5-level U-Net, k=3 unrolled iterations, trained MSE on truth phantom) jumps to hr=0.8215, **+0.21 over the next best** (Dual-Domain Bilateral). The gradient flowing through the full unroll matters far more than denoiser depth.
2. **Pre-training is fragile** — ItNet v2 (pre-train then frozen) sits at hr=0.48–0.54 depending on random search, ~0.27–0.34 below v3's end-to-end variant with the same backbone.
3. **TV (tuned) still wins on SSIM** (0.4691) — the learned methods sacrifice perceptual sharpness for low RMSE.
4. **Dual-Domain Bilateral** (Wagner 2022) reaches hr=0.6095 with only **6 parameters** — a useful sweet spot.
5. **Hammernik VN** (Hammernik 2017) is competitive at hr≈0.527 with only 18 k params and **no pre-training step** — the search confirmed the paper's `kernel=13` optimum but didn't move the headroom much above the default config.
6. **Wu 2015** is the strongest classical (no-learning) baseline at hr=0.38–0.41 and serves as the diagnostic floor.
7. All methods need validation on **real AAPM challenge data**.

---

## Next Steps

1. **Run Dual-Domain Bilateral Filter** — compare 4-parameter vs 0.5M-parameter denoising
2. **ItNet v3** — 5-level U-Net, end-to-end training, k=3, α = TV lambda (0.0037), tied weights
3. **Add Real Data** — Port all solvers to actual AAPM DL-Sparse-View breast phantoms
4. **Add Ensemble Methods** — Top challenge teams used ensembles of 5-10 networks

---

## Files

| Solver | Description | Parameters |
|--------|-------------|------------|
| `solver_fbp_baseline.py` | Pure FBP | 0 |
| `solver_tv_iterative.py` | TV-regularized iterative | 0 |
| `solver_tv_search.py` | TV with hyperparameter search | 0 |
| `solver_dual_ddomain.py` | Dual-domain learned denoising (U-Net) | ~0.5M |
| `solver_dual_ddomain_bilateral.py` | Dual-domain learned denoising (bilateral) | **8** |
| `solver_itnet.py` | ItNet-style v1 (broken) | ~0.2M |
| `solver_itnet_v2.py` | ItNet-style v2 (pre-train only) | ~0.2M |
| `solver_itnet_v3.py` | ItNet-style v3 (end-to-end, 5-level) | ~2.5M |
| `solver_wu_2015.py` | Classical aliasing-free FBP | 0 |

Sbatch scripts: `cluster/slurm/demo_ref_*.sbatch`, `cluster/slurm/demo_*_search.sbatch`
