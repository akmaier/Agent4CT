# Pentathlon Results

This document records the quantitative results from all parameter-search experiments run on the DL-Sparse-View challenge.

## SSIM Computation Note

All SSIM values reported below use our **canonical single implementation** (`ddssl_ldct/metrics.py`):
- **C1 = C2 = 0** (no stabilisation constants — images are calibrated floating-point attenuation coefficients, not 8-bit)
- **Comparison target**: ground-truth phantom (`val_ph`) for *all* solvers (previously some baselines incorrectly compared against noiseless FBP)

See `docs/itnet_architecture_analysis.md` for the full SSIM audit and architecture discussion.

---

## 1. TV Parameter Search (`demo-dl-tv-search-20260515-01`)

20-iteration random search over:
- `tv_lambda` [1e-4, 1e-2]
- `tv_iterations` [50, 500]
- `tv_lr` [1e-3, 1e-1]
- `tv_clip_max` [0.03, 0.08]
- `tv_decay` [0.0, 0.05]

### Best Results

| Metric | Iter-0009 (best headroom) | Iter-0017 (best SSIM) |
|--------|---------------------------|----------------------|
| **SSIM** | **0.4691** | **0.4804** |
| **PSNR** | 18.64 dB | 15.38 dB |
| **Headroom** | **0.6033** | 0.4223 |

### Best Parameters (headroom)

```python
tv_lambda     = 0.003723
tv_iterations = 408
tv_lr         = 0.041828
tv_clip_max   = 0.071431
tv_decay      = 0.046289
```

Training budget: **~3 min** (fits 5-min wall-clock limit comfortably).

---

## 2. Dual-Domain Parameter Search (`demo-dl-dual_domain-20260515-01`)

20-iteration random search over:
- `epochs` [3, 8]
- `lr` [1e-4, 5e-3]
- `batch_size` [1, 4]
- `unet_c` [8, 24]

### Best Results

| Metric | Value |
|--------|-------|
| **SSIM** | **0.2812** |
| **PSNR** | 18.78 dB |
| **Headroom** | **0.5868** |

### Best Parameters

```python
epochs      = 6
lr          = 0.000407
batch_size  = 2
unet_c      = 19
train_n     = 200
```

Architecture: Two **SmallUNet(c=19)** denoisers (projection + image domain), trained end-to-end via Noise2Inverse self-supervision.  
Parameters: **0.66 M** (×2 denoisers).  
Training time: **~101 s**.

---

## 3. ItNet Parameter Search (`demo-dl-itnet-20260515-01`)

20-iteration random search over:
- `pretrain_epochs` [3, 8]
- `pretrain_lr` [1e-4, 5e-3]
- `itnet_k` [3, 8]
- `unet_c` [8, 24]
- `itnet_alpha_init` [1e-3, 5e-2]
- `residual_learning` {True, False}

### Best Results

| Metric | Value |
|--------|-------|
| **SSIM** | **0.3296** |
| **PSNR** | — |
| **Headroom** | **0.5432** |

### Architecture (v2)

| Component | Details |
|-----------|---------|
| Denoiser | SmallUNet(c=11) — 3 levels, **0.11 M params** |
| Unrolled iterations | k = 6 |
| DC weight | α_init = 0.027 (learnable via softplus) |
| Residual learning | **False** (performed better than True) |

Training paradigm: **Pre-train denoiser only** on (FBP, truth) pairs, no end-to-end unrolled training.

---

## 4. Head-to-Head Comparison

| Solver | Best SSIM | Best Headroom | Params | Train Time | vs TV Δ HR |
|--------|-----------|---------------|--------|------------|-----------|
| **TV (tuned)** | **0.4804** 🏆 | **0.6033** 🏆 | 0 | ~3 min | — |
| Dual-Domain (tuned) | 0.2812 | 0.5868 | 0.66 M | ~101 s | −0.0165 |
| ItNet v2 (tuned) | 0.3296 | 0.5432 | 0.11 M | ~24 s | −0.0601 |
| Dual-Domain baseline | 0.3055* | 0.5831* | 0.23 M | ~462 s | — |
| **Wu 2015 (classical)** | 0.1285* | **0.3786*** | **0** | **~1.1 s** | −0.2247 |
| TV iterative baseline | 0.4454* | 0.0000* | 0 | ~5 min | — |

\* *Baselines / fixed-hyperparam refs (not random-searched).*

---

## 5. Key Insights

1. **TV regularization wins decisively** — even simple random search over 5 hyperparameters outperforms learned methods in both SSIM and headroom. This suggests the sparse-view streak-artifact pattern is better addressed by iterative model-based reconstruction than by CNN denoisers.

2. **SSIM vs headroom trade-off** for TV:
   - High SSIM → low iterations (66), high lambda (0.008) → sharp but streaky
   - High headroom → high iterations (400+), moderate lambda (0.004) → smoother, closer to truth

3. **ItNet under-performs** because:
   - Shallow 3-level U-Net cannot capture **non-local** streak artifacts (receptive field ~64 px on 512×512 image)
   - Pre-training only (not end-to-end) causes **distribution drift** at inference iterates 2…k
   - DC step is **40× weaker** than TV (α≈0.01 vs TV step 0.04×400)

4. **Dual-Domain is competitive** with TV on headroom (0.5868 vs 0.6033) but SSIM is much lower (0.28 vs 0.48), indicating it removes noise well but loses fine detail.

---

## 6. Next Steps / Open Experiments

- [ ] **ItNet v3**: 5-level U-Net, end-to-end training, k=3, α init = TV lambda (0.0037), tied weights
- [ ] **Dual-Domain with bilateral filters**: Wagner showed 8-param bilateral filters achieve 97 % of U-Net SSIM — worth testing for sparse-view
- [ ] **TV + learned post-processing**: Use TV as initialization, then fine-tune with a shallow CNN
- [ ] **Wu 2015 as warm-start**: the aliasing-free reconstruction is a sub-second hand-crafted prior — feeding it into the learned solvers as initial state may shorten training
- [ ] **Real AAPM data**: Port all solvers to the AAPM Low-Dose CT Grand Challenge dataset

---

## 7. Wu 2015 Classical Reference (`demo-ref-wu2015-761107`)

A pyro-nn implementation of Wu, Maier, Yang, Fahrig 2015,
*A Novel Filtered Backprojection-Based Algorithm for Sparse View CT
Image Reconstruction* (Fully3D 2015). See
[`literature/wu_2015_sparse_view_fbp.md`](../literature/wu_2015_sparse_view_fbp.md)
for the full algorithm derivation and the paper citation.

Two-stage classical method:
1. **Radius-dependent ramp filter** producing a view-aliasing-free FBP.
   The ramp `|f|` is split into K triangular frequency bands; each band
   is back-projected separately and combined per-pixel with a sigmoid
   weight `c_i(x) = σ(−10(f_i · Δ̃_R(x) − 1))` that suppresses bands
   above the local Nyquist `Δ̃_R(x) = s · Δβ`. Pixels near the centre
   keep all bands (full resolution); edges retain only low-frequency
   bands (blurred but aliasing-free).
2. **Feature-preserving sinogram interpolation** on the
   *residual sinogram* (= measured − forward-project(stage 1)). For
   each pair of adjacent views, a symmetric per-detector L1 motion
   search produces a midpoint view `ỹ(u, β_mid) = ½ (y(u−t, β₁) +
   y(u+t, β₂))`. FBP the densified residual through a 2× projector,
   soft-threshold, add back. Two outer iterations.

### Result

| Metric | Value |
|---|---:|
| **SSIM** | 0.1285 |
| **PSNR** | 14.74 dB |
| **RMSE** | 0.00916 |
| **Headroom** | **0.3786** |
| Params | 0 (no learning) |
| Wall time | **1.12 s** for `val_n = 100` |

### Hyperparameters (`solver_wu_2015.py`)

```python
wu_n_bands       = 4        # paper uses 8; 4 keeps cost down with ~no quality drop
wu_n_outer       = 2        # restoration iterations (paper: 2–3)
wu_motion_range  = 5        # ±pixels for symmetric motion search
wu_motion_window = 2        # ±pixels for L1 windowed patch
wu_soft_thresh   = 0.0015   # soft threshold on residual reco (μ mm^-1)
```

### Where it sits on the leaderboard

| Method (synthetic phantoms) | Headroom |
|---|---:|
| FBP baseline | 0.000 |
| TV iterative (fixed) | 0.256 |
| **Wu 2015** | **0.379** |
| Dual-Domain baseline | 0.583 |
| ItNet v2 (tuned) | 0.543 |
| Dual-Domain (tuned) | 0.587 |
| TV (tuned) | 0.603 |

**Take-aways**

- Cheapest non-trivial method by a wide margin (1.1 s vs 24–462 s).
- Beats TV iterative *at fixed hyperparams* by +0.12 headroom and TV
  reaches 0.60 only after random search — Wu 2015 is competitive
  without any tuning at all.
- Falls short of all learned and tuned methods, confirming the
  paper's framing: this is a *classical floor*, not a state-of-the-art
  contender on a 128-view geometry.
- Useful diagnostic: any learned method that beats FBP (0.0) but not
  Wu 2015 (0.38) is mostly re-discovering sinogram-interpolation
  tricks the classical algorithm gets for free.

---

*Last updated: 2026-05-16*
