# Zha 2024 — R²-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction

Adapts 3D Gaussian Splatting (Kerbl 2023) to CT by replacing the
visible-light α-blending renderer with a **radiative** rasteriser:
each Gaussian primitive contributes its analytic line integral to the
sinogram, the sum-image is forward-projected, and the L2 gap against
the measured sinogram is back-propagated through the Gaussian
parameters. Picked here as the GS-style reference for the sparse-view
pentathlon — it is the only published 3D GS variant explicitly for CT
and is the same baseline used by the DM4CT benchmark (Shi 2026).

## Citation

```bibtex
@inproceedings{zha2024r2gaussian,
  title     = {{R$^2$-Gaussian}: Rectifying Radiative Gaussian Splatting
               for Tomographic Reconstruction},
  author    = {Zha, Ruyi and Cheng, Lin and Han, Lujia and Gao, Chen and
               Zhang, Yanhao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2405.20693; code: github.com/Ruyi-Zha/r2\_gaussian}
}
```

## In one paragraph

A scan is represented as N anisotropic Gaussian primitives
`{(μᵢ, Σᵢ, αᵢ)}` directly in the patient frame. Unlike visible-light
3D GS, the renderer does *not* α-blend along the ray — it
*integrates* (sums) the analytic Gaussian densities along each
projection ray, giving an attenuation sinogram. The Gaussian
parameters are optimised by L2 fit to the measured sinogram, with
adaptive cloning/splitting/pruning of low-α Gaussians the same way 3D
GS does. No training set; one optimisation per scan.

## Architecture (2D adaptation in our pentathlon)

```
N anisotropic 2-D Gaussians {(μᵢ, Σᵢ, αᵢ, θᵢ)}
       │
       ▼  Rasterise → analytic 2-D Gaussian sum into (H, W) μ-image
       │
       ▼  ForwardProject (PyronnFanBeamProjector) → sinogram
       │
       ▼  ‖sino_pred − sino_measured‖²  +  λ_TV · TV(μ_image)
       │
       ▼  back-prop into μᵢ (positions), log Σᵢ (scales/rotation), softplus(αᵢ) (amplitudes)
```

The original paper uses tiled CUDA splatting for cone-beam 3D
rendering. For 2D fan-beam at 512² we rasterise the Gaussian sum
directly with a per-pixel evaluation in chunked batches — slower
in absolute terms but trivial to implement and adequate for the
pentathlon's 100-scan validation budget.

## In this repo

[`pentathlon/demo_dl_reference/solver_r2gaussian.py`](../pentathlon/demo_dl_reference/solver_r2gaussian.py)
implements the 2-D variant:

| Default | Value | Search range |
|---|---|---|
| N (Gaussians) | 1024 | {512, 1024, 2048} |
| Inner Adam iters | 600 | [400, 1000] |
| `lr_pos` | 5e-3 | log[1e-3, 2e-2] |
| `lr_scale` | 1e-2 | log[5e-3, 5e-2] |
| `lr_amp` | 1e-2 | log[5e-3, 5e-2] |
| Initial amplitude | 0.01 | log[5e-3, 5e-2] |
| Initial scale (norm coords) | 0.04 | log[0.02, 0.10] |
| TV weight | 1e-4 | log[1e-5, 1e-3] |

Val set: 20 random-ellipse phantoms (each fit independently), 10-min
wall clamp per outer iter. Parameter count = 6 × N (position 2 +
scale 2 + rotation 1 + amplitude 1) — at N=1024 that is ~6 k params
per scene, smaller than Hammernik-VN's 73 k.
