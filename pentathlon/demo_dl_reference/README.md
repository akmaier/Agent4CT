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
- **Status**: First run diverged (LR too high). Fixed with LR=0.01, λ=0.001, 200 iters.
- **Result**: Pending re-run (Job 760889)
- **Expected**: Should improve over FBP but may underperform learned methods

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

### 4. ItNet-Style Iterative (`solver_itnet.py`)
- **Method**: Sidky 2022 winner — pretrained U-Net + iterative data consistency
- **Architecture**: SmallUNet(c=16), 5 iterations, α=0.1
- **Training**: 20 epochs pre-training on (FBP, truth)
- **Result**:
  - SSIM: 0.1369
  - PSNR: 10.70 dB
  - RMSE: 0.0146
  - **Headroom: 0.0000** ← Underperforms (worse than FBP!)
  - Params: 0.23M
- **Issues**: 
  - DC gradient step likely too aggressive
  - Pre-training on noisy FBP may be poor
  - Needs better initialization and step-size scheduling
  - Real challenge data may behave differently
- **Time**: ~129 seconds

---

## Key Findings

### On Synthetic Phantoms:
| Method | Headroom | Notes |
|--------|----------|-------|
| FBP | 0.0000 | Baseline |
| Dual-Domain | **0.5831** | Best performance |
| TV | TBD | Classical approach |
| ItNet | 0.0000 | Needs debugging |

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
