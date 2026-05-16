# Zha 2022 — NAF: Neural Attenuation Fields for Sparse-View CBCT

Per-scene neural-implicit-representation reconstruction for sparse-view
cone-beam CT. A small MLP with positional encoding maps 3-D voxel
coordinates → linear attenuation μ; the MLP weights are optimised
against the measured sinogram via the standard radiative loss
`‖R(μ_θ) − g‖²`. The conceptual ancestor of every "NeRF for CT" paper
that followed (Tomo-INR, SAX-NeRF, IntraTomo, R2Gaussian's INR
baseline). Picked here as the NeRF-style reference for the sparse-view
pentathlon.

## Citation

```bibtex
@inproceedings{zha2022naf,
  title     = {{NAF}: Neural Attenuation Fields for Sparse-View {CBCT} Reconstruction},
  author    = {Zha, Ruyi and Zhang, Yanhao and Li, Hongdong},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention
               -- MICCAI 2022},
  pages     = {442--452},
  year      = {2022},
  doi       = {10.1007/978-3-031-16446-0_42},
  note      = {arXiv:2209.14540; code: github.com/Ruyi-Zha/naf\_cbct}
}
```

## In one paragraph

For each scan, NAF trains a coordinate MLP `μ_θ(x, y, z)` from scratch
purely against that scan's sinogram. The MLP has a positional-encoding
front-end (sin/cos at log-spaced frequencies, NeRF-style) so a small
network can fit high-frequency anatomical detail. Forward projection
through the scanner's geometry yields a predicted sinogram; the L2
mismatch against the measurement is back-propagated into the MLP. No
training set, no neural prior — the implicit-network architecture
itself is the regulariser.

## Architecture (2D adaptation in our pentathlon)

```
(x, y) ∈ [-1, 1]² ──► PosEnc ──► MLP_θ ──► softplus ──► μ
                       L freqs   4 layers ×128 hidden
```

Then for each training step:

```
μ_image = μ_θ(grid)                            # (H, W)
sino_pred = ForwardProject(μ_image)            # (n_angles, n_det)
loss = ‖sino_pred − sino_measured‖² + λ_TV · TV(μ_image)
```

Adam (lr ≈ 5e-3), 500–2000 inner iterations, no batch. The TV term
helps in extremely sparse-view regimes (< 64 views) but the original
paper relies on the implicit network's smoothness prior at moderate
sparsity.

## In this repo

[`pentathlon/demo_dl_reference/solver_naf.py`](../pentathlon/demo_dl_reference/solver_naf.py)
implements the 2-D variant for fan-beam sparse-view CT:

| Default | Value | Search range |
|---|---|---|
| Positional-encoding frequencies | 10 | {6, 10, 14} |
| MLP hidden | 128 | {96, 128, 192} |
| MLP layers | 4 | {3, 4, 5} |
| Inner Adam iters per scene | 600 | [300, 1200] |
| `lr` | 5e-3 | log[1e-3, 2e-2] |
| TV weight | 1e-4 | log[1e-5, 1e-3] |

Val set: 20 random-ellipse phantoms (each fit independently), 10-min
wall clamp per outer iter to stay inside the cluster's 1-h sbatch
budget.
