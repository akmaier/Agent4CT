# Hammernik 2017 — Variational Network for Limited-Angle CT

Two-step deep learning architecture for limited-angle CT
reconstruction. **Step 1** is the Würfl 2016 fan-beam-as-NN
reconstruction: learned per-detector compensation weights `W_comp` plus
fixed cosine weights, ramp filter, and back-projection produce an
intensity-corrected image `y_NN`. **Step 2** — the focus of this paper —
is a *variational network*: `T` unrolled gradient-descent steps on a
parameterised image-restoration energy whose regulariser is a learned
filter bank acting through learned activation functions. End-to-end
trained with MSE.

The architecture is the CT counterpart of Hammernik's 2016 MR
variational network (ISMRM). It precedes — and conceptually anticipates
— the ItNet family of unrolled CT solvers (Sidky 2022).

## Citation

```bibtex
@inproceedings{hammernik2017deep,
  title     = {A deep learning architecture for limited-angle computed tomography reconstruction},
  author    = {Hammernik, Kerstin and W{\"u}rfl, Tobias and Pock, Thomas and Maier, Andreas},
  booktitle = {Bildverarbeitung f{\"u}r die Medizin 2017: Algorithmen-Systeme-Anwendungen. Proceedings des Workshops vom 12. bis 14. M{\"a}rz 2017 in Heidelberg},
  pages     = {92--97},
  year      = {2017},
  organization = {Springer}
}
```

PDF: [papers/Hammernik_2017_bvm_limited_angle.pdf](../papers/Hammernik_2017_bvm_limited_angle.pdf)

## Two-step architecture

```
sinogram  ──►  Step 1  ──►  y_NN  ──►  Step 2  ──►  y_VN^T
   x       NN reconstruction         T unrolled GD steps
```

### Step 1 — Würfl-style NN reconstruction

For limited-angle data the back-projection picks up intensity
inhomogeneities. The paper inherits Würfl et al. 2016's recipe:

```
y_NN = Ψ( B · C · W_comp · W_cos · x )
```

| Symbol | Meaning | Learned? |
|---|---|---|
| `W_cos` | Cosine weights (geometry) | fixed |
| `W_comp` | Compensation weights (per detector × per angle) | **learned** |
| `C` | 1D detector-wise filter (ramp) | fixed |
| `B` | Back-projection operator | fixed |
| `Ψ` | Non-negativity (ReLU) | fixed |

The output `y_NN` is intensity-corrected but still has the streaks that
analytical FBP can't remove on its own.

### Step 2 — Variational network (the main contribution)

A regularised image-restoration objective

```
E(y) = (λ/2) · ‖y − y_NN‖²    +    Σ_{i=1..N_k}  ρ_i( K_i · y )
       └──── data fidelity ────┘   └──── learned regulariser ────┘
```

— with `N_k` learned 2-D analysis filters `K_i` and learned scalar
potentials `ρ_i` — is unrolled into `T` gradient-descent steps. Each
step uses its **own** parameters (untied weights):

```
y^t = y^{t−1}  −  Σ_{i=1..N_k} K^T_{i,t} · ρ'_{i,t}( K_{i,t} · y^{t−1} )  −  λ_t · ( y^{t−1} − y_NN )
       └───────── gradient of regulariser at step t ─────────┘       └── gradient of data term at step t ──┘
```

The transpose convolution `K^T` is the kernel flipped along both
spatial axes (standard valid-convolution adjoint). The potential
derivatives `ρ'_{i,t}` are parameterised as a weighted sum of
Gaussian RBFs on a fixed grid of centres (Chen-Yu-Pock 2015) — so each
activation is a *learned* shape, not a fixed nonlinearity.

**Learned per step:** `{K_{i,t}}, {ρ'_{i,t}}, λ_t`. Step count `T = 5`
fixed empirically; filter count `N_k = 24`; kernel sizes tested
`k ∈ {5, 7, 9, 11, 13}` — k=13 best.

## Training

- **Loss**: MSE between `y^T_VN` and full-scan reference `z`.
- **Optimiser**: L-BFGS-B (paper). 5×100 iters of per-step pre-training
  followed by 700 iters of joint end-to-end training.
- **Data**: 450 fan-beam projections at 512×512 from 10 patients,
  5-fold CV with 80/20 split.

## Reported results (limited-angle CT, paper Table 1)

| Method | PSNR | SSIM |
|---|---:|---:|
| Neural network (Step 1 only) | 34.66 ± 2.07 | 0.908 ± 0.015 |
| Bilateral filter | 29.93 ± 3.61 | 0.907 ± 0.021 |
| BM3D | 34.75 ± 2.09 | 0.911 ± 0.015 |
| TV | 34.82 ± 2.10 | 0.914 ± 0.014 |
| TGV | 34.80 ± 2.09 | 0.914 ± 0.014 |
| Variational Network k=5 | 36.16 ± 2.13 | 0.930 ± 0.010 |
| Variational Network k=7 | 36.86 ± 2.01 | 0.938 ± 0.010 |
| Variational Network k=9 | 38.14 ± 2.27 | 0.947 ± 0.009 |
| Variational Network k=11 | 37.87 ± 1.98 | 0.949 ± 0.009 |
| Variational Network k=13 | **38.23 ± 2.06** | **0.952 ± 0.010** |

The variational network beats every reference filter (BM3D, TV, TGV,
bilateral). At kernel size 13 the SSIM improvement over TV is +0.038.

## Relation to ItNet v3 in this repo

`pentathlon/demo_dl_reference/solver_itnet_v3.py` is a *spiritual* descendant of
the Hammernik 2017 variational network — same unrolled-GD skeleton —
but differs in three key parameterisation choices:

| | Hammernik 2017 VN | ItNet v3 |
|---|---|---|
| Regulariser at each step | Linear filter bank `K_i` + learned RBF activations `ρ'_i` (~3.6 k params/step) | Full 5-level U-Net (~2.5 M params total) |
| Per-step weights | **Untied** (`T=5` separate filter banks) | **Tied** (same U-Net at every iter) |
| Data fidelity inside the unroll | Pulls toward `y_NN` (image domain) | Pulls toward measured sinogram (projection domain, via `R^T(R·x − g)`) |
| Total params | ~18 k for `T=5, N_k=24, k=11` | ~2.5 M |
| Origin | Variational image restoration (Chen-Yu-Pock 2015) | Sidky 2022 ItNet |

The variational-network framing is **much smaller, much more
interpretable** (filters and activations can be visualised), and
trains end-to-end like ItNet v3. The trade-off: each filter has limited
receptive field, so deep U-Net features that ItNet can build are out
of reach for a 24-filter bank.

## Reference implementation in this repo

[`pentathlon/demo_dl_reference/solver_hammernik_2017.py`](../pentathlon/demo_dl_reference/solver_hammernik_2017.py)
adapts the Step-2 variational network to the **sparse-view** setup:

- **No Step 1 compensation weights** — sparse-view has full angular
  coverage, just under-sampled, so no Würfl-style per-angle weight
  correction is needed; we use the standard FBP + non-negativity as
  `y_NN`.
- `T = 5`, `N_k = 24`, `k = 11` (paper's penultimate kernel size — k=13
  costs more memory at marginal SSIM gain).
- 31-RBF activation grid on `[−2, 2]`, σ = grid spacing.
- Adam optimiser (paper used L-BFGS-B; Adam works fine for the smaller
  end-to-end unroll and matches the rest of our demo solvers).
