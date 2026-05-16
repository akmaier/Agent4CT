# Liu 2025 — A Review on 3D Gaussian Splatting for Sparse-View Reconstruction

40-page open-access review (AI Review, Springer, 2025) of sparse-view
3D reconstruction methods built on **3D Gaussian Splatting** (Kerbl
2023). Organises ~30 recent papers into a unified four-stage pipeline:
**Sparse Inputs → Multi-dimensional Feature Extract → 3D Gaussian
Generation → Outputs**. Catalogues the loss-function and module
choices of each method, then runs side-by-side numerical comparisons
on five standard sparse-view benchmarks.

The survey is the natural counterpart to the diffusion-CT benchmark
(Shi 2026) — same "sparse-view inverse problem" framing, but in 3D
graphics rather than tomography.

## Citation

```bibtex
@article{liu2025review,
  author    = {Liu, Haitian and Liu, Binglin and Hu, Qianchao and Du, Peilun
               and Li, Jing and Bao, Yang and Wang, Feng},
  title     = {A review on 3D {G}aussian splatting for sparse view reconstruction},
  journal   = {Artificial Intelligence Review},
  volume    = {58},
  pages     = {215},
  year      = {2025},
  doi       = {10.1007/s10462-025-11171-4}
}
```

PDF: [papers/gaussian_splatting_survey_2025.pdf](../papers/gaussian_splatting_survey_2025.pdf)

## 3D Gaussian Splatting in one paragraph

3D GS (Kerbl 2023) represents a scene as N anisotropic 3D Gaussian
ellipsoids `{(μᵢ, Σᵢ, αᵢ, cᵢ)}` — centre, covariance, opacity,
view-dependent colour (SH coefficients). Rendering is differentiable
α-blending of projected 2D Gaussians along the ray:

```
C(pixel) = Σᵢ cᵢ · αᵢ' · Tᵢ       Tᵢ = Π_{j<i} (1 − αⱼ')
αᵢ' = αᵢ · exp(−½ (x' − μᵢ')ᵀ Σᵢ'⁻¹ (x' − μᵢ'))
```

Training uses RGB + D-SSIM loss with adaptive *densification* (clone or
split Gaussians) and *pruning* (remove low-α). The explicit point-cloud
form gives near-real-time rendering — the main reason 3D GS displaced
NeRF for live applications.

In sparse-view (few-shot or single-image) the standard 3D GS pipeline
under-determines the scene: missing input views → floaters, scale
ambiguity, multi-view inconsistency. The survey is about how recent
methods plug those holes.

## The survey's four-stage framework (paper Sect. 3)

```
Sparse inputs ──► Multi-dim feature extract ──► 3D Gaussian generation ──► Outputs
  (1–few imgs)     (point cloud, depth maps,    (end-to-end inference,    (mesh, novel-
                    semantic / spatial /         diffusion priors, …)      view images)
                    geometric / cross-view
                    features)
```

Each downstream stage can rely on multiple feature types from the
preceding stage; the survey maps every method into Table 9 by which
ingredients it uses (depth estimation, feature extract, text prompt,
diffusion model) and its distinctive module.

## Quantitative leaderboards (paper Sect. 7)

The survey aggregates results from each method's own paper on five
benchmarks. **GS-based methods are in bold below** (the survey
highlights them with ✓ in the "3D GS" column).

### MipNeRF360, 4-views (Table 4 — large outdoor scenes)
| Method | 3D GS | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|:-:|---:|---:|---:|
| FreeNeRF | | 13.71 | 0.853 | 16.83 |
| DietNeRF | | 18.90 | 0.897 | 11.17 |
| **3D GS (baseline)** | ✓ | 20.31 | 0.899 | 10.80 |
| **FSGS** | ✓ | 21.07 | 0.910 | 9.51 |
| **GSObject** | ✓ | **24.81** | **0.935** | **4.98** |

### NeRF-LLFF, 3-views (Table 5 — forward-facing scenes)
| Method | 3D GS | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time | FPS |
|---|:-:|---:|---:|---:|---:|---:|
| FreeNeRF | | **19.63** | **0.612** | 0.308 | 2.3 h | 0.21 |
| 3D GS (baseline) | ✓ | 14.51 | 0.398 | 0.406 | 2.7 m | 280 |
| Chung et al. | ✓ | 17.17 | 0.497 | 0.337 | — | — |
| **DNGaussian** | ✓ | 19.12 | 0.591 | **0.294** | **3.5 m** | **300** |

### GSO (Google Scanned Objects), single view (Table 6)
| Method | 3D GS | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time |
|---|:-:|---:|---:|---:|---:|
| Realfusion | | 15.26 | 0.722 | 0.283 | 20 m |
| One-2-3-4-5 | | 17.84 | 0.800 | 0.199 | 45 s |
| Shap-e | | 15.45 | 0.772 | 0.297 | 9 s |
| **DreamGaussian** | ✓ | 19.56 | 0.853 | 0.174 | 2 m |
| **TriplaneGaussian** | ✓ | 16.81 | 0.797 | 0.257 | **0.2 s** |
| **LGM** | ✓ | 16.90 | 0.819 | 0.235 | 5 s |
| **GRM** | ✓ | 20.10 | 0.826 | 0.136 | 5 s |
| **FDGaussian** | ✓ | **22.98** | **0.899** | **0.146** | 70 s |

### D-NeRF, monocular dynamic (Table 7)
| Method | 3D GS | PSNR ↑ | SSIM ↑ | FPS | Time |
|---|:-:|---:|---:|---:|---:|
| TiNeuVox-S | | 30.75 | **0.96** | 0.32 | **8 m** |
| TiNeuVox-B | | 32.67 | **0.97** | 0.13 | 28 m |
| K-Planes | | 31.61 | **0.97** | 0.54 | 52 m |
| V4D | | **33.72** | 0.98 | 1.47 | 6.9 h |
| **3D GS (baseline)** | ✓ | 20.51 | 0.89 | 170 | 6 m |
| **D-3D GS** | ✓ | 17.22 | 0.81 | 173 | 15 m |
| **Katsumata et al.** | ✓ | 32.07 | **0.96** | **150** | 8 m |

### RealEstate10K, 2-views (Table 8)
| Method | 3D GS | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time |
|---|:-:|---:|---:|---:|---:|
| PixelNeRF | | 20.43 | 0.589 | 0.550 | 5.30 s |
| muRF | | 26.10 | 0.858 | 0.143 | 0.186 s |
| **pixelSplat** | ✓ | 25.89 | 0.858 | 0.142 | 0.104 s |
| **MVSplat** | ✓ | **26.39** | **0.869** | **0.128** | **0.044 s** |

## Most successful approaches by application regime

### Static scene reconstruction (multi-view, few-shot)
1. **GSObject** (Yang et al. 2024b) — wins MipNeRF360 4-view by ~4 PSNR
   over plain 3D GS.
   - Uses a **visual hull** for initial Gaussian sampling and a
     **Gaussian repair model** (ControlNet + Stable Diffusion fine-tuned
     in a leave-one-out scheme) to fix floaters and rendering defects.
   - Adds depth, mask, and LPIPS losses on top of standard RGB + SSIM.
2. **FDGaussian** (Feng et al. 2024) — wins GSO single-view by ~3 PSNR.
   - Two-stage: orthogonal plane decomposition (3 view planes →
     geometry) followed by a diffusion-guided Gaussian generator.
3. **DNGaussian** (Li et al. 2024) — best speed + accuracy combo on
   NeRF-LLFF 3-view (PSNR 19.1, 300 FPS, 3.5 min training).
   - **Hard + soft depth regularisation**: hard regulariser snaps
     Gaussian positions onto depth surface, soft regulariser modulates
     opacities to remove floaters near surfaces.
   - **Global-local depth normalisation** to escape monocular-depth
     relative-scale issues.
4. **FSGS** (Zhu et al. 2023) — second on MipNeRF360 4-view; introduces
   **proximity-guided Gaussian unpooling** (densify near triangulated
   sparse points first).
5. **SparseGS** (Xiong et al. 2023) — patch-based depth correlation loss
   + **floater elimination** via dip-test on bivariate depth
   discrepancies.

### Single-image inference (end-to-end)
1. **MVSplat** (Chen et al. 2024) — wins RealEstate10K 2-view at the
   fastest inference (44 ms).
   - Cost-volume + CNN/Transformer hybrid for cross-view feature
     extraction.
2. **pixelSplat** (Charatan et al. 2024) — close second (104 ms).
   - **Epipolar transformer** to infer per-scene scale factors;
     probabilistic Gaussian-position sampling avoids local minima.
3. **GRM** (Xu et al. 2024c) — pure-transformer single-view 3D GS;
   pixel-aligned Gaussian prediction.
4. **Gamba** (Shen et al. 2024) — *Mamba*-based long-sequence model that
   reframes Gaussian densification as sequence processing.
5. **Splatter Image** (Szymanowicz et al. 2024) — U-Net predicts a
   pixel-wise "Splatter Image" of 3D GS parameters; full end-to-end.
6. **TriplaneGaussian** (Zou et al. 2024) — hybrid triplane + 3D GS,
   real-time inference at 0.2 s/image.

### Dynamic scenes
1. **Katsumata et al. 2023** — only GS method competitive with V4D on
   D-NeRF (PSNR 32 vs 33.7) at 150 FPS (vs 1.5 FPS).
2. **Efficient4D** (Pan et al. 2024) — 4D extension via SyncDreamer-T
   spatiotemporal video aggregation.
3. **BAGS** (Zhang et al. 2024) — *Building Animatable Gaussian
   Splatting*; rigid-regularisation + SDS loss on diffusion-generated
   invisible views.

### Diffusion-guided generation (text- or image-to-3D)
1. **DreamGaussian** (Tang et al. 2023) — first 3D GS + SDS pipeline;
   mesh extraction + UV-space texture refinement; ~2 min single-image
   inference at PSNR 19.6 on GSO.
2. **LGM** (Tang et al. 2024) — four orthogonal-azimuth + fixed-elevation
   image generator → asymmetric U-Net → 3D GS; uses Plücker ray
   embeddings for camera pose.
3. **FDGaussian** (Feng et al. 2024) — also fits here; diffusion-guided
   geometry-aware multi-view generation with **Gaussian Divergence
   Saliency (GDS)** to suppress redundant splits.

## How sparse-view artefacts are addressed (paper Table 3 + Sect. 6)

| Artefact | Approach | Methods |
|---|---|---|
| **Floaters / wrong-depth Gaussians** | Adaptive pruning, deep regularisation, mask-based exclusion | SparseGS, FDGaussian (GDS), DNGaussian, pixelSplat, GSObject (visual hull), BAGS |
| **View inconsistency** | Diffusion-supplied novel views, epipolar attention | pixelSplat, FDGaussian, Lee et al. (text-inverted SDS) |
| **Scale ambiguity** | Depth scale alignment with sparse SfM points, multi-view epipolar transformer | Chung et al., SparseGS (Pearson depth correlation), pixelSplat |
| **Overfitting** | Early-stopping when depth-guided loss plateaus, regularisation | Chung et al., DreamGaussian (texture-stage MSE) |
| **High-resolution detail** | Total Variation in spatiotemporal dimension | EndoGS |

## Loss-function recipes that recur

`L_RGB` (L1) + `L_SSIM` is universal. Extensions:

- **Depth losses** `L_Depth`: present in 11 of 17 methods (Chung et al.,
  DNGaussian, FSGS, SparseGS, GSObject, Lee et al., pixelSplat,
  MVSplat, GRM, DIG3D, …) — almost always alongside an explicit
  monocular-depth estimator.
- **Score Distillation Sampling** `L_SDS`: 7 methods (DreamGaussian,
  GSObject, Lee et al., LGM, BAGS, Efficient4D, FDGaussian) — pairs a
  pretrained 2D diffusion model as a regulariser.
- **LPIPS** `L_LPIPS`: 9 methods — adds perceptual-feature alignment.
- **Smoothness / TV / Rigid**: scene-class-specific extras (FSGS, EndoGS,
  BAGS).

The survey explicitly highlights the consistency of `L_SSIM + L_Depth +
L_LPIPS` as the **modern recipe** for sparse-view 3D GS — and the
*absence* of `L_Depth` in plain 3D GS as the main culprit for
sparse-view failures.

## Open challenges (paper Sect. 9, the only section worth quoting)

| # | Challenge | Promising direction |
|---|---|---|
| 1 | Discrete Gaussian representation lacks temporal continuity | Add survival-time parameters; hybrid NeRF + GS |
| 2 | Dependency on auxiliary models (depth, pose) | Better COLMAP / DPT, joint training |
| 3 | Special scenes (monochrome, specular, sky) | Edge/normal priors, depth normalisation |
| 4 | Resource consumption | Distributed compute, sparse data structures |
| 5 | Insufficient 3D dataset diversity | More datasets, augmentation |
| 6 | Overfitting under sparse views | New geometric regularisers |
| 7 | Complex texture generation | Higher-resolution input, image enhancers |

## Relevance to Agent4CT

3D GS is **not directly applicable to sparse-view 2D CT** — it's a 3D
scene rendering model, not a tomographic reconstruction algorithm.
However, two specific lines of work in the survey *do* port:

1. **R2Gaussian** (Zha et al. 2024) — mentioned in Shi 2026 / DM4CT as
   a baseline; explicitly represents objects with Gaussian primitives
   and optimises their parameters directly to fit CT measurements.
   This is the closest cross-over and would be the right cite if we
   added a GS baseline to `demo_dl_reference/`.
2. **Diffusion-guided sparse-view priors** (DreamGaussian, FDGaussian,
   GSObject, LGM) — share the same "use a 2D diffusion model to
   hallucinate missing views" pattern that Shi 2026 evaluates for CT.
   If we ever extend to true 3D cone-beam reconstruction (e.g. the
   CT-MAR challenge in `data/ct_mar/`), the SDS loss + diffusion
   prior pattern is the obvious extension.

The pragmatic take: **for our 2D fan-beam pentathlon, the
ItNet-v3-style unrolled networks and Hammernik-VN dominate; 3D GS
becomes relevant only if/when we tackle full-volume cone-beam CT.**
