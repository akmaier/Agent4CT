# Demo Reference Implementations — Results Summary

**Run**: `demo-dl-reference-20260515-01`  
**Dashboard**: https://akmaier.github.io/Agent4CT/dashboard.html

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

### 3. Dual-Domain Denoising (`solver_dual_ddomain.py`)
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

### 4. ItNet-Style Iterative (`solver_itnet.py`) — v1 FAILED
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

### 5. ItNet-Style v2 (`solver_itnet_v2.py`) — FIXES APPLIED
- **Fixes**:
  1. α=0.01 (10x smaller), **learnable** via softplus parameterization
  2. Residual learning: predict (truth - fbp) instead of full image
  3. Early stopping (patience=3) prevents identity collapse
  4. Evaluate against **truth phantom** (not FBP)
  5. Show residual error map in comparison figure
- **Result**: Pending (Job 760896)
- **Expected**: Headroom > 0, should beat FBP baseline

---

### 6. Wu 2015 Classical FBP (`solver_wu_2015.py`)
- **Method**: Wu, Maier, Yang, Fahrig 2015 — radius-dependent frequency-split
  ramp filter (aliasing-free FBP) + feature-preserving symmetric
  motion-compensated sinogram interpolation, 2 outer residual-restoration
  iterations with soft thresholding. No learning. See
  [literature/wu_2015_sparse_view_fbp.md](../../literature/wu_2015_sparse_view_fbp.md).
- **Architecture**: 4 triangular frequency bands (paper uses 8), ±5-pixel
  symmetric motion search with ±2-pixel L1 window, soft threshold 0.0015 μ.
- **Result** (Job 761107):
  - SSIM: 0.1285
  - PSNR: 14.74 dB
  - RMSE: 0.00916
  - **Headroom: 0.3786** ← strongest classical baseline on this geometry
  - Params: 0 (no learning)
- **Time**: ~1.1 seconds — by far the cheapest non-trivial method.

---

## Key Findings

### On Synthetic Phantoms:
| Method | Headroom | SSIM | Notes |
|--------|---------:|-----:|-------|
| FBP | 0.0000 | 0.4454 | Baseline (defines headroom=0) |
| TV iterative | 0.2562 | 0.1281 | Classical MBIR |
| **Wu 2015** | **0.3786** | **0.1285** | Strongest classical, ~1 s wall |
| Dual-Domain | 0.5831 | 0.3055 | Best learned method (Wagner 2023) |
| ItNet v1 | 0.0000 | 0.1369 | Broken; needs debugging |

### Lessons:
1. **Dual-Domain denoising is strongest** on synthetic data (0.58 headroom)
2. **ItNet is fragile** — requires careful tuning of DC step size and pre-training
3. **TV is an important baseline** — should improve over FBP but less than learned methods
4. All methods need validation on **real AAPM challenge data** for meaningful comparison

---

## Next Steps

1. **Fix TV reconstruction**: Already submitted with better hyperparameters (Job 760889)
2. **Debug ItNet**: 
   - Try smaller DC step (α=0.01 instead of 0.1)
   - Pre-train on clean FBP vs. truth pairs
   - Add momentum to iterations
3. **Add Real Data**: Port all solvers to use actual AAPM DL-Sparse-View breast phantoms
4. **Add Ensemble Methods**: Top challenge teams used ensembles of 5-10 networks

---

## Files

- `solver_fbp_baseline.py` — Pure FBP
- `solver_tv_iterative.py` — TV-regularized iterative
- `solver_dual_ddomain.py` — Dual-domain learned denoising
- `solver_itnet.py` — ItNet-style with data consistency

Sbatch scripts: `cluster/slurm/demo_ref_*.sbatch`
