# Terris 2025 — RAM: Reconstruct Anything Model

Matthieu Terris, Samuel Hurault, Maxime Song, Julián Tachella.
**"Reconstruct Anything Model: a lightweight general model for computational imaging."**
arXiv:2503.08915 (March 2025; v2 March 2026). BSD-3-Clause.

- Paper PDF: [`papers/2503.08915_Terris_ReconstructAnythingModel.pdf`](../papers/2503.08915_Terris_ReconstructAnythingModel.pdf)
- Code: <https://github.com/matthieutrs/ram>
- Pretrained checkpoint: <https://huggingface.co/mterris/ram> (`ram.pth.tar`, 143 MB)
- Built on `deepinv >= 0.3.0`.

## Problem setup

Standard linear inverse problem `y = H x + n`, where `H` is the forward operator
(here: fan-beam projection) and `n` is Poisson-Gaussian noise. RAM is trained as
a single **non-iterative** mapping `y → x̂` that nonetheless consumes the forward
operator at inference time and adapts to its acquisition physics.

The key claim is that one model trained across deblurring, MRI, CT, inpainting,
and super-resolution generalises to **unseen** inverse problems (and to unseen
datasets via 1–50-shot self-supervised fine-tuning).

## Architecture

- **Backbone**: a DRUNet-style multi-scale encoder/decoder (4 scales, channels
  `[64, 128, 256, 512]`, 4 residual blocks per scale, c_mult=2).
- **Modality heads & tails**: routed by input channel count
  (`in_channels=[1, 2, 3]`) so grayscale, complex, and color share most weights.
- **Multi-scale physics conditioning**: at every residual block, a
  `MeasCondBlock` runs `N=2` Krylov iterations
  `[A^T y, (A^T A) A^T y, (A^T A)^2 A^T y, …]` of the operator on the current
  feature stream. Deeper U-Net scales operate on downsampled images, so the
  forward operator is composed with sinc downsampling
  (`MultiScaleLinearPhysics`).
- **Noise conditioning**: the network reads `physics.noise_model.sigma`
  (Gaussian) and `physics.noise_model.gain` (Poisson) and concatenates them as
  constant channels — that's how it adapts to per-input noise levels.
- **Operator-gain conditioning** (`physics.factor`): used by an optional
  `prox_l2` realignment of the adjoint input. Skipped if not set.

End-to-end this looks like a single forward pass, but adds up to ~50 calls of
`A` + `A^T` because of the in-block Krylov embeddings — effectively a
"few-iteration unrolled" network packaged as one inference.

## Inputs / outputs

```python
model(y, physics=physics)   # y: (B, C, *meas_shape) → x̂: (B, C, *img_shape)
```

For CT with our geometry:

- `physics.A(x)` must return our 128-angle / 1024-detector sinogram for a μ-image `x` of shape `(B, 1, 512, 512)`.
- `physics.A_adjoint(y)` must be the **mathematical adjoint** (un-filtered backprojection), **NOT FBP**. Adjointness `⟨A x, y⟩ = ⟨x, A^T y⟩` is required for the Krylov embeddings to be meaningful.
- `physics.noise_model.sigma` should be set to the expected post-log sinogram noise std (rough order: 1e-3 to 1e-2 in our μ-units).

Images are normalised to `[0, 1]` at training time. RAM is dimension-agnostic
(input only needs `H, W` divisible by 8; 512 fits).

## Self-supervised fine-tuning

`ram.finetune(model, y, physics, noise_loss="SURE", transform="shift", lr=1e-4)`
does ~50 Equivariant Imaging + SURE epochs on a single measurement. Costs
seconds per slice and frequently beats zero-shot inference on out-of-distribution
operators. Good axis to explore in the agentic search.

## How we plan to use it for the DL-Sparse-View challenge

Zero-shot baseline as a strong "foundation-model" reference for the dashboard,
then optional self-supervised fine-tune. Hyperparameter search axes (driven by
Claude over 20 iters):

| axis | candidates |
|---|---|
| `ram_sigma` (noise conditioning) | log [1e-3 .. 5e-2] |
| `ram_input_norm` | divide y by `display_max`, OR by FBP-init max, OR none |
| `ram_clamp_output` | True / False (clamp to [0, display_max]) |
| `ram_finetune` | True / False |
| `ram_finetune_epochs` | 0, 20, 50, 100 |
| `ram_finetune_lr` | log [1e-5 .. 1e-3] |
| `ram_factor` | 1.0, power-iteration estimate, or 0 (skip realign) |
| `ram_post_fbp_blend` | 0 (RAM only) … 0.5 … 1.0 (FBP only) |

Solver lives at `pentathlon/demo_dl_reference/solver_ram.py`; agentic harness
mirrors the Claude-driven diffusion-recon search (oneshot sbatch reads
`RAM_CONFIG_PATH`, writes a per-iter `result.json` + `comparison.png`).

## Risks / caveats

- Trained on natural images via deepinv's physics zoo. CT performance is
  evaluated in the paper but zero-shot on our specific 128-angle geometry may
  lag a CT-trained baseline like ItNet v3.
- The `MultiScaleLinearPhysics` wrapper composes `A` with sinc downsampling for
  the deeper U-Net scales. PyroNN's projector is configured for 512² → if we
  feed it 256² / 128² / 64² inputs without re-configuring the geometry, it will
  either error or silently project through the wrong geometry. Two paths:
  - **Quick**: monkey-patch RAM to `scales=[1]` only.
  - **Proper**: instantiate scaled `PyronnFanBeamProjector`s at 256/128/64 with
    proportionally scaled `image_spacing`, then dispatch by input shape.
- `prox_l2` realignment falls back to conjugate gradient, which calls `A` and
  `A^T` many times. If too slow, override with closed-form Tikhonov or skip via
  `factor=0`.

## Citation

```bibtex
@article{terris2025ram,
  title={Reconstruct Anything Model: a lightweight foundational model for
         computational imaging},
  author={Terris, Matthieu and Hurault, Samuel and Song, Maxime and
          Tachella, Julián},
  journal={arXiv preprint arXiv:2503.08915},
  year={2025}
}
```
