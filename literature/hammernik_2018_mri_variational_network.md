# Hammernik 2018 — MRI Variational Network (with data consistency)

The MRI counterpart of Hammernik's CT variational network (BVM 2017).
Embeds a generalised compressed-sensing reconstruction in an unrolled
gradient-descent scheme where **each step combines a learned-regulariser
gradient with a closed-form data-consistency step against the measured
k-space data**. End-to-end trained on clinical knee MRI; reconstruction
takes ~193 ms per case on a single GPU.

Adapted in this repo as the **Hammernik-VN** sparse-view CT solver: the
MRI forward operator `A` (Fourier + coil sensitivities) is swapped for
our PyRoNN fan-beam projector, but every other piece of the
architecture — the per-step learned filter banks `K_i^t`, the per-filter
learned RBF activations `Φ'_i^t`, and the per-step data weight `λ^t` —
carries over verbatim.

## Citation

```bibtex
@article{hammernik2018learning,
  author    = {Hammernik, Kerstin and Klatzer, Teresa and Kobler, Erich and
               Recht, Michael P. and Sodickson, Daniel K. and Pock, Thomas and
               Knoll, Florian},
  title     = {Learning a variational network for reconstruction of accelerated {MRI} data},
  journal   = {Magnetic Resonance in Medicine},
  volume    = {79},
  number    = {6},
  pages     = {3055--3071},
  year      = {2018},
  doi       = {10.1002/mrm.26977},
  note      = {arXiv:1704.00447}
}
```

PDF: [papers/Hammernik_2018_MRM_variational_network_1704.00447.pdf](../papers/Hammernik_2018_MRM_variational_network_1704.00447.pdf)

## Theory in three equations

1. **Linear inverse problem** with under-sampled measurements `f` and
   linear sampling operator `A` (MRI: Fourier·SensitivityMaps; our CT
   adaptation: fan-beam forward projector):

   `min_u  (1/2)‖A u − f‖²`

2. **Landweber iteration** (paper Eq. 3) — the data-only update:

   `u^{t+1} = u^t − α^t · A*(A u^t − f)`

3. **Add Fields-of-Experts regulariser** (paper Eq. 4) and plug into Eq. 3
   — the form the network unrolls (paper Eq. 6):

   `u^{t+1}  =  u^t  −  Σ_{i=1..N_k} (K_i^t)^T · Φ'_i^t( K_i^t · u^t )  −  λ^t · A*( A · u^t − f )`
   `              └────── gradient of learned regulariser ──────┘     └── gradient of data term ──┘`

The key difference vs the BVM 2017 (CT) paper is that the **data term
gradient lives in the measurement domain**: it back-projects the
current sinogram residual through `A*`. Both Hammernik papers share the
same regulariser parameterisation (learned filters + learned RBF
activations).

The step size `α^t` of Eq. 5 is absorbed into the activations and `λ^t`
in Eq. 6 — the paper omits it explicitly.

## What each step learns

| Parameter | Shape | Constraints |
|---|---|---|
| Filter bank `k_i^t` | `(N_k, 1, k, k)` per step | zero-mean, unit-norm (projected gradient) |
| RBF weights `w_{ij}^t` | `(N_k, N_w)` per step | free |
| Data weight `λ^t` | scalar per step | non-negative (`softplus`) |

The activation is a Gaussian-RBF mixture (Chen-Yu-Pock 2015):

`Φ'_i(z) = Σ_{j=1..N_w} w_{ij} · exp(−(z − μ_j)² / (2σ²))`

with `N_w = 31` equidistant RBF centres on `[−I_max, I_max]` and
`σ = 2 I_max / (N_w − 1)`.

## Paper configuration (clinical knee MRI)

| Hyperparameter | Value |
|---|---|
| `T` (unrolled steps) | 10 |
| `N_k` (filters per step) | 48 |
| Filter size `k` | 11×11 |
| `N_w` (RBF bumps) | 31 |
| `I_max` (RBF range) | 150 |
| `λ^t` init | 0.001, non-negative |
| Optimizer | IIPG (Inertial Incremental Proximal Gradient) |
| Step size | η = 10⁻³ |
| Training | 1000 epochs, batch=10, 200 training images |
| Loss | ε-smoothed MSE on magnitude images |

Total parameters: ~131 050 — about 7× the BVM 2017 variational network
(which used T=5, N_k=24) but four orders of magnitude smaller than a
typical U-Net solver.

## Adaptation to sparse-view CT in this repo

`pentathlon/demo_dl_reference/solver_hammernik_vn.py` makes the
following changes:

| MRI paper | Sparse-view CT adaptation |
|---|---|
| Complex k-space `f` | Real-valued sinogram `g` (128 views × 736 dets) |
| Sampling operator `A` = `M·F·C` | `PyronnFanBeamProjector.forward_project` |
| Adjoint `A*` = `C*·F*·M*` | `PyronnFanBeamProjector.back_project` |
| Coil sensitivities `C` | dropped (single-channel CT) |
| Real/imag filter pairs `(k_re, k_im)` | single real filters |
| Zero-filled init `u^0 = A*f` | choice of `fbp` (default — better start at 128 views) or `backproj` (paper-faithful) |
| IIPG optimiser | Adam (matches the rest of the demo_dl_reference suite) |
| ε-smoothed magnitude MSE | plain MSE on real-valued images |
| Filter normalisation | optional (`vn_normalize_filters`); off by default |

Everything else — per-step untied weights, RBF activation, learned
`λ^t`, MSE loss against truth — is unchanged.

The data-consistency step is the key architectural feature that
distinguishes Hammernik-VN from `solver_hammernik_2017.py` (which only
pulls toward the fixed initial FBP, never re-uses the sinogram). It
puts Hammernik-VN architecturally in the same family as
`solver_itnet_v3.py` — both unroll Landweber-with-learned-regulariser
gradient descent through the projector. The differences:

| | `solver_hammernik_vn.py` (this) | `solver_itnet_v3.py` |
|---|---|---|
| Regulariser per step | Linear filter bank + RBF activations | Full 5-level U-Net |
| Per-step weights | Untied (paper-faithful) | Tied (same U-Net every iter) |
| Params at `T=10, N_k=48, k=11` | ~73 k | ~2.5 M |
| Data fidelity gradient | `λ^t · R^T(R u^t − g)` | `α · R^T(R u^t − g)` via FBP-like step |
| End-to-end MSE training | yes | yes |

So Hammernik-VN is the **structurally minimal** unrolled-with-DC
solver: same skeleton as ItNet v3 but with a tiny, interpretable
per-step regulariser instead of a U-Net.
