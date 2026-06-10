# `solver_hammernik_vn.py` — Hammernik MRI Variational Network (2018, ported to CT)

Hammernik's MRI VN architecture ported to the CT inverse problem:
`vn_T` shared-weight CG-like unrolled steps, each with a learnable
gradient-descent step + image-domain regulariser via learned filter
bank. Companion design doc.

## Architecture summary

```
x_0  = init (FBP-of-noisy or zeros)
for t in range(vn_T):
    grad_data = R^T (R x_t - y)            # data-fidelity gradient
    R_reg = filter_bank(x_t)               # learned filter response
    x_{t+1} = x_t − α * grad_data − λ_t * R_reg
return x_T
```

Knobs:
- `vn_T`: number of unrolled steps (typically 3-7).
- `vn_n_filters`: filter-bank width (16-32 typical).
- `vn_kernel`: filter kernel size (7-11).
- `vn_lambda_init`: initial λ regulariser strength.
- `vn_init`: "fbp" or "zeros" for x_0.

## When to use VN vs Hammernik 2017

Hammernik 2017 has 1 trainable λ per unroll step (~3 parameters total)
— minimal-capacity baseline. VN adds learned filter banks (~10-30 k
parameters with vn_n_filters=16, vn_kernel=11) — more capacity, more
flexible.

## Cross-dataset record (filled in 2026-06-08 via Mayo Step-3 TPE)

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | 0.3621 | TPE `demo-intensity-calibrated-tpe-hammernik-vn-search-20260520-01` (ep=17, lr=2.0e-4, T=3, filters=16, kernel=9, init=fbp, train_n=200) | rank 15 on demo-DL. |
| `breast_ct` | **0.4883** | TPE `breast-ct-calibrated-tpe-hammernik-vn-search-20260521-01` (ep=18, lr=3.0e-4, T=3, filters=32, kernel=7, init=fbp, train_n=200) | rank 8 on breast-CT. Beats Hammernik 2017 (rank 9, hr=0.4549) — the learned filter bank earns its capacity. |
| `mayo_ldct` | **0.0551** | TPE 762926 iter-6 (vn_T=5, vn_n_filters=16, vn_kernel=11, vn_λ_init=2.3e-3, ep=12, lr=2.6e-4) — **OVERTURNED Step-2 STOP** | rank 10 on Mayo. Step-2 agentic loop filed 2-consecutive-hr=0 plateau ("STOP"); Step-3 TPE found a working corner the agentic random walk missed. |

## 2026-06-08 — Mayo: Step-2 STOP overturned by Step-3 TPE

**This is the headline finding for Hammernik VN.**

The Mayo Step-2 agentic loop dispatched Hammernik VN at iter-2/3
with hr=0 SSIM 0.27→0.29 — the agentic protocol filed a "2 consecutive
hr=0 plateau" STOP verdict. Same Hammernik family failure pattern as
the 2017 variant (per-step λ regulariser barely budges).

Step-3 TPE (job 762926, `mayo-ldct-2d-calibrated-tpe-hammernik-vn-search-20260608-01`)
ran 20-trial Optuna TPE with Mayo clamps. **iter-6 found a working
corner at hr=0.0551** (vn_T=5, vn_n_filters=16, vn_kernel=11,
vn_λ_init=2.3e-3, ep=12, lr=2.6e-4). TPE then explored the eta-corner
neighborhood across iters 13/17 (similar vn_T=5/n_filters=16/kernel=11
families) finding hr=0.0290-0.0338 — close but not beating iter-6.

Final TPE 20/20: best hr=0.0551, rank 10 on Mayo (between TV-iter at
0.0557 and NAF at 0.0202).

### Why the agentic loop missed it

The Mayo Step-2 agentic random walk explored vn_n_filters from the
solver's CONFIG default (16) but only varied vn_T (3 → 5) and
epochs. The agentic loop's neighbourhood exploration of `vn_λ_init`
and `vn_kernel` was too narrow to find the iter-6 corner.

**Lesson for the autoresearch protocol:** when the agentic loop files
a 2-consecutive-hr=0 plateau on a low-complexity learned solver (here
12 k trainable params), TPE on a wider search space still has a
chance to escape. The agentic loop's neighbourhood random walk can
miss isolated working corners in low-dimensional but multi-axis
spaces.

This is the FIRST and ONLY case in the Step-3 TPE phases where TPE
overturned an agentic STOP verdict on Mayo. NAF's Mayo Step-2 result
went the other way (Step-2 iter-1 0.0202 stayed; TPE found 0.0131
worse).

### Cross-dataset pattern

Hammernik VN works on every dataset tested, but requires different
capacity profiles per dataset:

| Dataset | vn_T | vn_n_filters | vn_kernel | Optimum reason |
|---|---:|---:|---:|---|
| `demo_dl`   | 3 | 16 | 9 | Simpler substrate, fewer steps needed. |
| `breast_ct` | 3 | 32 | 7 | Wider filter bank to handle anatomy details. |
| `mayo_ldct` | 5 | 16 | 11 | Deeper unroll + bigger kernel for 2304-angle sino complexity. |

The VN's "shared-weight per-step learned filter bank" lets it absorb
data complexity by adding unroll steps rather than expanding the
single-step model — the right inductive bias for CT.
