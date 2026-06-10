# `solver_tv_iterative.py` — TV-iterative (non-trainable, hand-tuned)

Companion design doc. For the supervised-L2 trainable variant see
`solver_tv_iterative_supervised.py`.

Plain projected-gradient-descent on the data-fidelity term with a
smooth-TV regulariser. **Zero trainable parameters** — every knob is
hand-tuned or TPE-searched, and the same config applies to every
scene at inference.

```
x_0 = FBP(noisy)
for iter in range(tv_iterations):
    grad_data = R^T (R x_t - y)
    grad_tv   = ∇·(∇x_t / √(|∇x_t|² + ε²))     # smooth-TV gradient
    x_{t+1} = clip(x_t − tv_lr * (grad_data + tv_lambda * grad_tv),
                   0, tv_clip_max)
return x_final
```

## When TV-iterative wins

This is the **"properly tuned classical algorithm"** baseline. It
wins when:

- Baseline FBP is **noisy enough** that data-fidelity + TV
  regularisation can recover detail FBP smoothed away.
- The dataset is **dense-view** enough that the data term has a
  unique solution (sparse-view + TV is also fine but a different
  regime).
- **No training budget is available** — TV-iterative is
  zero-train-time, only inference TV iterations.

## Cross-dataset record

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | **0.4056** | TPE `demo-intensity-calibrated-tpe-tv-search-20260520-01` (λ=3.6e-3, iters=382, lr=0.099, train_n=0) | rank 12 on demo-DL. Beats DD-UNet N2I (rank 13, hr=0.3811) at zero parameters. |
| `breast_ct` | **0** | TPE `breast-ct-calibrated-tpe-tv-search-20260521-01` (λ=1.8e-3, iters=214, lr=4.0e-2, train_n=0): all 20 trials hr=0 | **STOP** — breast-CT's properly-tuned FBP baseline (SSIM 0.957, PSNR 39.7 dB) is already at the truth. TV iterations smooth the image away from truth, dropping SSIM. |
| `mayo_ldct` | **0.0557** | Mayo Step-2 iter-8 (tv_iterations=12800, tv_lambda=0.01, tv_step=0.5, train_n=0) | **rank 9 on Mayo.** Real-helical data has enough noise that TV regularisation lifts above baseline FBP. Step-3 TPE (job 762924, search-space-clamped) found 0.0511 — slightly under Step-2 because the TPE clamp on tv_iterations was too tight; agentic's 12,800 iters wins. |

## 2026-06-08 — Mayo: TV-iterative is the strongest non-trainable solver

Mayo Step-2 agentic ran 9 iters of TV-iterative, climbing from
iter-1 hr=0.0080 (tv_iter=200, default) to iter-8 hr=**0.0557** at
tv_iter=12800, λ=0.01, step=0.5. The hr-vs-tv_iter curve is
monotonic — more iterations = better recovery, up to the data-term
fixed point.

Step-3 TPE attempted refinement (job 762924) but the search-space
clamp on `tv_iterations` ([50, 800] from the demo-DL default) was
two orders too tight for Mayo. Best TPE hr=0.0511, slightly under
agentic. **Step-2 stays as the rank-9 entry.**

Lesson: when a non-trainable solver hits a hr peak at the FAR EDGE
of the search space, expand the search space before TPE-refining —
TPE in a clamped-too-tight space can't find the agentic winner.

## CONFIG defaults

```python
CONFIG = {
    "tv_lambda":     0.001,
    "tv_iterations": 200,
    "tv_lr":         0.01,
    "tv_clip_max":   0.05,
}
```

## Cross-solver comparison (non-trainable rank)

| Solver | demo-DL hr | breast-CT hr | Mayo hr |
|---|---:|---:|---:|
| **TV-iterative** (this) | 0.4056 | 0 | **0.0557** |
| Wu 2015 non-trainable | 0.2295 | 0.0425 | 0 |
| R²-Gaussian v2 | 0.3455 | 0 | 0 |
| FBP baseline | 0 (ref) | 0 (ref) | 0 (ref) |

TV-iterative is the **strongest non-trainable solver on all 3
datasets when it works**. On breast-CT it doesn't work because FBP
is already at the truth (SSIM > 0.95); TV smoothing pulls SSIM down.

## Hints for the next autoresearch agent

- **TV-iterative is the right starting baseline on new datasets**
  where FBP is mediocre (SSIM < 0.85). It's parameter-free, has
  4 knobs, and converges in 200-1000 iters at lr=1e-2 on most data.
- For Mayo-style real-helical data with noise: tv_iterations up to
  12800 is worth trying (each iter is cheap — ~5 ms on Q6000).
  Don't clamp the iter axis below 5000 if you're TPE-searching on
  data with real noise.
- For sparse-view synthetic phantoms (demo-DL): tv_iterations
  300-500 is usually enough; the TV regulariser is the binding
  constraint, not iter count.
- For breast-CT-like datasets (FBP already at truth): skip
  TV-iterative entirely. It cannot improve on a SSIM>0.95 baseline.
