# ItNet v2 Architecture Analysis: Why It Performs Poorly

## 1. Network Architecture

### SmallUNet (the denoiser)

```
Input: 1×512×512 (single-channel CT slice)

Encoder:
  enc1: Conv2d(1→c, 3×3) + GroupNorm + ReLU
        Conv2d(c→c, 3×3) + GroupNorm + ReLU
        → c×512×512
  pool → c×256×256
  
  enc2: Conv2d(c→2c, 3×3) + GroupNorm + ReLU
        Conv2d(2c→2c, 3×3) + GroupNorm + ReLU
        → 2c×256×256
  pool → 2c×128×128
  
  enc3: Conv2d(2c→4c, 3×3) + GroupNorm + ReLU
        Conv2d(4c→4c, 3×3) + GroupNorm + ReLU
        → 4c×128×128
  pool → 4c×64×64

Bottleneck:
  bot:  Conv2d(4c→4c, 3×3) + GroupNorm + ReLU
        Conv2d(4c→4c, 3×3) + GroupNorm + ReLU
        → 4c×64×64

Decoder:
  up3: ConvTranspose2d(4c→4c, 2×2, stride=2)
  dec3: Conv2d(8c→2c, 3×3) + GroupNorm + ReLU  [skip connection from enc3]
        Conv2d(2c→2c, 3×3) + GroupNorm + ReLU
        → 2c×128×128
        
  up2: ConvTranspose2d(2c→2c, 2×2, stride=2)
  dec2: Conv2d(4c→c, 3×3) + GroupNorm + ReLU    [skip from enc2]
        Conv2d(c→c, 3×3) + GroupNorm + ReLU
        → c×256×256
        
  up1: ConvTranspose2d(c→c, 2×2, stride=2)
  dec1: Conv2d(2c→c, 3×3) + GroupNorm + ReLU    [skip from enc1]
        Conv2d(c→c, 3×3) + GroupNorm + ReLU
        → c×512×512

Head:
  head: Conv2d(c→1, 1×1)  [zero-initialized]
        → 1×512×512
        
Output: x - head(x) if residual else head(x)
```

### Parameter Count

| Component | Formula | c=16 | c=19 |
|-----------|---------|------|------|
| enc1 | (1·c·9+c) + (c·c·9+c) | 2,480 | 3,457 |
| enc2 | (c·2c·9+2c) + (2c·2c·9+2c) | 13,888 | 20,854 |
| enc3 | (2c·4c·9+4c) + (4c·4c·9+4c) | 55,424 | 83,008 |
| bottleneck | (4c·4c·9+4c) × 2 | 73,856 | 110,672 |
| up3 + dec3 | trans + (8c·2c·9+2c) + (2c·2c·9+2c) | 62,448 | 93,504 |
| up2 + dec2 | trans + (4c·c·9+c) + (c·c·9+c) | 15,600 | 23,257 |
| up1 + dec1 | trans + (2c·c·9+c) + (c·c·9+c) | 7,025 | 10,498 |
| head | c·1 + 1 | 17 | 20 |
| **Total U-Net** | | **231,921** | **345,757** |
| **+ alpha param** | | **231,922** | **345,758** |
| **In Megabytes** | | **0.93 MB** | **1.38 MB** |

*All layers use Kaiming (He) initialization except head which is zero-initialized.*

### Activation Functions
- **ReLU** (inplace) after every conv
- **Softplus** for alpha constraint: α = softplus(log_α) > 0

### Normalization
- **GroupNorm** with adaptive group count (divides channels by largest factor ≤ 8)

## 2. Training Configuration

| Parameter | Search Range | Best Value |
|-----------|-------------|------------|
| Training samples | 200 | 200 |
| Pre-training epochs | 3-8 | 3-4 |
| Learning rate | 1e-4 - 5e-3 | 0.00076 - 0.002 |
| Optimizer | Adam | Adam |
| Batch size | 1 (sample-by-sample) | 1 |
| Early stopping patience | 3 epochs @ 1% threshold | 3 |
| Unrolled iterations (k) | 3-8 | 4-6 |
| Alpha initialization | 0.001-0.05 | 0.001-0.027 |
| Residual learning | True / False | **False** |

### Actual Training Times

| Iteration | unet_c | pretrain_ep | k | Time | Headroom |
|-----------|--------|------------|---|------|----------|
| 5 (best HR) | 11 | 4 | 6 | 23.9s | 0.5432 |
| 15 (best SSIM) | 17 | 3 | 4 | 26.4s | 0.4093 |
| 20 | 11 | 3 | 8 | 20.7s | 0.0000 |

## 3. Why ItNet Fails on This Task

### A. Shallow Architecture (Only 3 Levels)

**Receptive field calculation:**
- 3 encoder levels × 2 convs × 3×3 kernels ≈ **~37×37 pixel RF**
- With skip connections: effective RF ≈ **~64×64 pixels**

**Problem:**
- Phantom ellipses span 50-200 pixels
- Sparse-view streaks span the **entire 512×512 image** (global)
- The network cannot capture non-local streak patterns
- Compare to Dual-Domain which operates in sinogram space where streaks are local

### B. Wrong Learning Paradigm

**ItNet assumes:**
```
Input:  noisy image with additive Gaussian noise
Target: clean image
Task:   denoising (local pixel regression)
```

**Our task is:**
```
Input:  FBP with sparse-view streak artifacts  
Target: ground truth phantom
Task:   reconstruction (global artifact removal)
```

**Key difference:**
- Denoising: noise is pixel-wise independent, locally stationary
- Reconstruction: streaks are highly structured, non-local, deterministic

The U-Net's convolutional prior assumes local correlations, which is wrong for streak artifacts.

### C. Training/Inference Distribution Mismatch

**Training:** Denoiser sees FBP images
```
x_0 = FBP(noisy_sino)  ← training distribution
```

**Inference (unrolled):**
```
x_0 = FBP(noisy_sino)       ← matches training ✓
x_1 = DC(denoiser(x_0))     ← different distribution ✗
x_2 = DC(denoiser(x_1))     ← very different ✗✗
x_3 = DC(denoiser(x_2))     ← out of distribution ✗✗✗
```

The denoiser is never trained on intermediate iterates, causing distribution drift.

### D. Weak Data Consistency Step

**TV regularization:**
```
Gradient: ∇_f [½||Rf - g||² + λ·TV(f)]
Step size: lr ≈ 0.04, iterations: 400
Effective: strong data fidelity enforcement
```

**ItNet DC step:**
```
x ← x_denoised - α·R^T(R·x - g)
where α = softplus(log_α_init) ≈ 0.01
```

With α ≈ 0.01, the data consistency correction is **40× smaller** than TV's effective step. The denoiser dominates, pushing the solution away from data consistency.

### E. No End-to-End Training of the Unrolled Loop

**How ItNet should be trained:**
```python
for epoch in range(N):
    for batch in data:
        pred = ItNet(fbp_batch, sino_batch)  # full unrolled forward
        loss = MSE(pred, truth_batch)
        loss.backward()  # gradients flow through all k iterations
        optimizer.step()
```

**How we actually train:**
```python
# Phase 1: Pre-train denoiser only
for epoch in range(pretrain_epochs):
    pred = Denoiser(fbp_batch)
    loss = MSE(pred, truth_batch)
    ...

# Phase 2: No end-to-end training!
# The unrolled loop is never trained jointly
```

The denoiser is trained in isolation, then plugged into the unrolled loop without adaptation.

## 4. Comparison with Dual-Domain

| Aspect | ItNet | Dual-Domain |
|--------|-------|-------------|
| **Parameters** | 0.1-0.4M | 0.15-0.7M |
| **Training time** | 20-60s | 40-100s |
| **Architecture** | 3-level U-Net | Two 3-level U-Nets |
| **Domain** | Image-only | Projection + Image |
| **Data consistency** | Weak (α≈0.01) | Strong (end-to-end trained) |
| **Training** | Denoiser pre-training only | End-to-end dual-domain joint |
| **Best headroom** | 0.54 | 0.60 |

**Dual-Domain wins because:**
1. Operates in sinogram space where artifacts are local
2. Trained end-to-end (both denoisers jointly optimize)
3. Stronger data fidelity through projection-domain processing

## 5. Recommendations to Fix ItNet

1. **Deeper network**: 4-5 U-Net levels for larger receptive field
2. **End-to-end training**: Train the full unrolled loop, not just denoiser
3. **Larger DC step**: Initialize α ≈ 0.1-0.5, not 0.01
4. **Multi-scale training**: Train on patches at multiple resolutions
5. **Architecture change**: Use attention or transformers for non-local streaks
6. **Hybrid approach**: Combine with TV regularization as a hard constraint
