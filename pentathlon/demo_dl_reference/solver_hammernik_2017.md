# `solver_hammernik_2017.py` — Hammernik 2017 (limited-angle CT variational network)

Companion design doc. For the MRI VN ported to CT (more capacity) see
`solver_hammernik_vn.py`.

Hammernik et al. 2017 "Learning a Variational Network for
Reconstruction of Accelerated MRI Data" was originally an MRI paper,
but the variational-network skeleton applies to any inverse problem.
This solver is the original (2017) recipe ported to CT:

```
x_0 = init (FBP-of-noisy)
for t in range(T):
    grad_data = R^T (R x_t - y)
    R_reg = filter_bank(x_t) → activation(RBF) → filter_bank^T
    x_{t+1} = x_t − α * grad_data − λ_t * R_reg
return x_T
```

Notable knobs vs the later VN port:
- **RBF-parametrised activation** (`vn_n_bumps=31`, `vn_x_range=2.0`)
  — learned per-pixel non-linearity between filter passes.
- **Per-step λ** but shared filter bank across steps.
- `vn_T` typically 3-5 (paper uses T=10; expensive).

## Cross-dataset record

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | 0.3622 | TPE `demo-intensity-calibrated-tpe-hammernik-search-20260520-01` (ep=30, lr=1.5e-3, T=3, filters=16, kernel=7, train_n=200) | rank 14 on demo-DL. Behind every other learned solver — RBF activation + shared filter bank is dated. |
| `breast_ct` | **0.4549** | TPE `breast-ct-calibrated-tpe-hammernik-search-20260521-01` (ep=30, lr=9.6e-4, T=3, filters=16, kernel=9, λ=9.7e-3, train_n=200) | rank 9 on breast-CT. Strong baseline at only 5 k params. Beaten by the VN port (rank 8, hr=0.4883) which adds learned filter banks. |
| `mayo_ldct` | **0** | Mayo Step-2 iter-1: hr=0 SSIM 0.27, PSNR 11.55 < baseline 12.59 (λ_t stuck within 0.8 % of init) | **STOP** — Per-step λ regulariser barely budges over 3 epochs. Same Hammernik family failure as the 2017 VN port saw initially. Sino-complexity ceiling, not a config knob away from working. |

## 2026-06-08 — Mayo verdict

The Mayo Step-2 agentic iter-1 dispatched Hammernik 2017 at default
(T=3, filters=16, ep=3, lr=5e-4, train_n=50). Result: hr=0 SSIM 0.27,
PSNR 11.55 dB < baseline 12.59. Logged λ_t learning curves show the
per-step λ scalars moved within 0.8% of init — the optimiser found no
useful gradient.

Unlike the VN port (which Step-3 TPE overturned to hr=0.0551 by
finding a working corner), **Hammernik 2017 was NOT retested via
TPE on Mayo**. The 2017 architecture has fewer capacity knobs than
the VN port (RBF activation grid + shared filter bank) — the
expected TPE finding would be the same hr=0 verdict, just slower.

Cross-dataset family pattern:

| Dataset | Hammernik 2017 | Hammernik VN | Δ (VN − 2017) |
|---|---:|---:|---:|
| `demo_dl` | 0.3622 | 0.3621 | −0.0001 (tied) |
| `breast_ct` | 0.4549 | 0.4883 | +0.0334 |
| `mayo_ldct` | 0 | 0.0551 | +0.0551 (VN's overturn) |

The VN port's **learned filter bank per step** (Hammernik 2017 uses
a shared filter bank across all steps) is the key architectural
difference. On harder datasets, having per-step learned filters lets
the VN port absorb data complexity that the 2017 variant cannot.

## CONFIG defaults

```python
CONFIG = {
    "train_n":            200,
    "vn_T":               3,                # unroll depth (paper: T=10; we use 3)
    "vn_n_filters":       24,
    "vn_kernel":          11,               # k=13 best per paper, k=11 cheaper at -0.003 SSIM
    "vn_n_bumps":         31,               # RBF activation grid
    "vn_x_range":         2.0,              # RBF centres span [-x_range, +x_range]
    "vn_filter_init_std": 0.05,
    "vn_rbf_init_std":    0.01,
    "vn_lambda_init":     1.0e-3,
    "epochs":             20,
    "batch_size":         4,
    "lr":                 5e-4,
}
```

## Hints for the next autoresearch agent

- Use Hammernik 2017 as a **minimal-capacity baseline** for the VN
  family — ~5 k trainable params is the smallest learned solver
  that clears baseline on synthetic data.
- Don't waste agentic iters on Hammernik 2017 for Mayo or anything
  helical — the structural ceiling is below baseline. If you need
  a Hammernik-family solver on Mayo, use the VN port (which Step-3
  TPE pushed to hr=0.0551 with vn_T=5, vn_n_filters=16, vn_kernel=11,
  vn_λ_init=2.3e-3).
- The RBF activation grid (`vn_n_bumps=31`, `vn_x_range=2.0`) is
  expensive at backward pass — if memory-pressured, prefer the VN
  port which uses plain ReLU/PReLU activations.
