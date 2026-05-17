# `demo_dl_reference/` — Strict conventions for fair comparison

Every solver in this directory MUST follow the rules below so that all
results in `docs/runs/demo-dl-*` are directly comparable. Past runs were
contaminated by two recurring bugs (per-solver seed drift and missing
clamps on negative output values); this document codifies the fix so it
doesn't happen again.

## The four rules

### 1. Fixed seeds, fixed val set

All solvers must use **the same validation phantoms**. Concretely:

```python
val_ph, val_clean, val_noisy = build_dataset(
    geom,
    cfg["val_n"],            # = 100 in the canonical CONFIG
    cfg["seed"] + 1000,      # val phantoms = cfg["seed"] + 1000 + i for i in 0..val_n-1
    cfg["noise_i0"],         # = 1e5
    cfg["noise_sigma_e"],    # = 10.0
    device,
)
```

`build_dataset` (the helper defined in every solver) calls
`random_ellipses_phantom(seed=seed + i)` for each sample, so the val
seed *offset* must always be `+1000` and `val_n` must be the canonical
100 unless the solver explicitly documents a smaller budget for
compute reasons (e.g. NAF / R2Gaussian use `val_n=20` because they fit
per scene).

Per-scene solvers that build the val set without `build_dataset` MUST
use the equivalent per-image seed: `cfg["seed"] + 1000 + i`.

`simulate_low_dose` must always be called with `seed=cfg["seed"] +
10_000` so the simulated Poisson + Gaussian noise realisation is
identical across solvers.

### 2. Clamp predictions to `[display_min, display_max]` before metrics

```python
pred = pred.clamp(cfg["display_min"], cfg["display_max"])     # [0, 0.05] in mu mm^-1
```

Reasons:
- The truth phantoms live in `[0, 0.05]`. Negative or `>0.05` pixels in
  the prediction give misleading SSIM/PSNR/RMSE numbers and inflate
  headroom artificially in either direction.
- The FBP baseline reference is computed against the same clamp:
  `val_fbp = torch.clamp(proj.fbp(val_noisy), min=0)`. If the candidate
  recon doesn't clamp, the baseline_rmse / candidate_rmse ratio that
  defines `headroom` is biased.

The clamp goes **after the network output**, just before metric
computation. Don't insert it inside the optimisation loop (kills
gradients — see the NAF v1 / Diffusion v1 collapse bugs).

### 3. Canonical reference baselines

```python
data_range = cfg["display_max"] - cfg["display_min"]    # 0.05
val_psnr = float(psnr(pred, val_ph, data_range=data_range).cpu())
val_ssim = float(ssim(pred, val_ph, data_range=data_range).cpu())
val_rmse = float(((pred - val_ph) ** 2).mean().sqrt().cpu())

baseline_psnr = float(psnr(val_fbp, val_ph, data_range=data_range).cpu())
baseline_rmse = float(((val_fbp - val_ph) ** 2).mean().sqrt().cpu())
headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
```

- `data_range` is `display_max - display_min`, not `target.amax() -
  target.amin()`. Always pass it explicitly to `psnr` and `ssim` from
  `ddssl_ldct.metrics`.
- `val_score` reported in `result.json` must equal `val_ssim` (the
  autoresearch agent ranks by `val_score` for tie-breaking).
- `val_fbp` for the baseline is the noisy-sinogram FBP clamped to
  `[0, +∞)` (we don't upper-clamp the baseline — the headroom denominator
  is more conservative that way and easier to beat fairly).

### 4. Standardised `result.json` keys

Every solver writes `result.json` with at least:

```python
{
    "val_score":      float,    # = val_ssim
    "val_psnr":       float,
    "val_ssim":       float,
    "val_rmse":       float,
    "baseline_psnr":  float,
    "baseline_rmse":  float,
    "headroom":       float,    # = max(0, 1 - val_rmse / baseline_rmse)
    "params_M":       float,    # trainable parameters in millions
    "train_n":        int,
    "val_n":          int,
    "train_time_s":   float,
    "config":         dict,
}
```

The autoresearch agent (`scripts/learned_solver_search_agent.py`) reads
`headroom` to rank iterations. Missing/non-numeric `headroom` is
treated as 0.

## Past bugs this prevents

| Bug | Symptom in old runs | Fix in convention |
|---|---|---|
| Solver evaluated against val phantoms with different seeds than the FBP baseline | SSIM/PSNR depend on val choice; cross-solver comparison meaningless | Rule 1: `cfg["seed"]+1000` mandatory |
| Solver output left un-clamped (e.g. negative pixels from residual prediction) | RMSE artificially low in dark regions, headroom inflated | Rule 2: `pred = pred.clamp(0, display_max)` mandatory |
| Solver used `data_range = pred.amax() - pred.amin()` for SSIM | Solvers that produce wider dynamic range get higher SSIM | Rule 3: `data_range = display_max - display_min` fixed |
| Inner-loop clamp killed gradients (NAF v1, Diffusion v1) | hr = 0 across all 20 search iters; constant SSIM ≈ 0.25 | Clamp is applied AFTER optimisation, not inside |

## Backwards-incompatibility note

Old `docs/runs/demo-dl-*` entries (dated 2026-05-15 / 16) may reflect
results computed BEFORE these conventions were enforced. They are kept
for historical visibility but **must not be cited as the official
leaderboard** going forward. The official leaderboard lives in the
`demo-dl-fair-*` slugs (date `2026-05-17` onwards) which were generated
under this convention.
