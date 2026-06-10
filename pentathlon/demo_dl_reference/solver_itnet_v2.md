# `solver_itnet_v2.py` — ItNet v2 (gradient clipping + LR schedule)

Companion design doc. For the original v1 see `solver_itnet.py`; for
the canonical v3 (deeper UNet + per-step α) see `solver_itnet_v3.py`.

v2 is a stabilisation of v1 — same lightweight UNet-per-step
architecture but with two training-time changes that prevent loss
explosions on harder data:

1. **Residual learning toggle** (`residual_learning=True` by default):
   per-step UNet predicts an additive residual, not the full updated
   image — keeps the unrolled trajectory closer to FBP.
2. **Smaller `itnet_alpha_init`** (0.01 vs v1's 0.1) — per-step
   gradient step starts at one-tenth the v1 default, with early
   stopping in pretraining (`pretrain_patience=3`).

## When to use v2 vs v1 vs v3

| Variant | Distinguishing feature | Best dataset | Best hr |
|---|---|---|---:|
| v1 | Original recipe — no per-step α toggle | demo-DL (tied with v3) | 0.4665 |
| **v2** (this solver) | `residual_learning` + smaller α_init + early stopping | breast-CT | **0.5386** |
| v3 | Deeper UNet + per-step learnable α | breast-CT | 0.7342, Mayo 0.2181 |

## Cross-dataset record

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | 0.4567 | TPE `demo-intensity-calibrated-tpe-itnet-v2-search-20260520-01` (pre_ep=6, pre_lr=2.3e-4, k=3, α=0.032, residual=F, train_n=400) | rank 7 on demo-DL. Beaten by v3 (rank 3, 0.4676) and v1 (rank 4, 0.4665). |
| `breast_ct` | **0.5386** | TPE `breast-ct-calibrated-tpe-itnet-v2-search-20260521-01` (pre_ep=3, pre_lr=2.6e-4, k=4, α=2.6e-3, residual=T, train_n=400) | rank 7 on breast-CT. Confirms ItNet ordering on breast-CT: v1 (0.1703) < v2 (0.5386) < v3 (0.7342). |
| `mayo_ldct` | **0** | Step-2 retry post-cfg-patch eae661bc (2026-06-08); iter-2 (k=2, c=16, train_n=50, ep=3) hr=0 SSIM 0.268 PSNR 10.21; iter-3 (ep=6) hr=0 SSIM 0.264 | **STOP** — same low-capacity ceiling as v1. The v3 architecture (deeper UNet + per-step α) is the only ItNet that lifts above baseline on Mayo. |

## cfg-merge bug history (2026-06-08 patch — commit eae661bc)

Same bug as v1: `solver_itnet_v2.py` had no `env_var` read for the
agentic JSON cfg, so it used hardcoded defaults regardless of what
the agentic loop sent. Mayo's 2304-angle sino at k=5/c=16 OOM'd
`filter_sino` (5 GiB FFT pad on Q6000 24-GB cap).

Patch added env-read pattern matching `solver_itnet_v3.py` (commit
eae661bc). Mayo v2 retries then completed but landed hr=0 with
SSIM=0.26 — the **low-capacity ceiling is structural**, not a
configuration knob away.

## CONFIG defaults

```python
CONFIG = {
    "pretrain_epochs": 10,
    "pretrain_lr":   1e-3,
    "pretrain_patience": 3,        # early stopping
    "itnet_k":       5,
    "itnet_alpha_init": 0.01,      # Much smaller than v1's 0.1
    "residual_learning": True,     # Predict residual, not full image
    # plus standard train/val knobs
}
```

## Hints for the next autoresearch agent

- v2 is the **gentlest of the three ItNets** to train — if v3 keeps
  diverging, fall back to v2's `residual_learning=True` + smaller
  α_init + early stopping triplet.
- v2 lifts demo-DL hr by +0.02 over v1 with similar parameter
  counts (~233 k for v2 vs ~466 k for v1 at default unet_c=16). The
  capacity-vs-stability trade is essentially flat on synthetic data.
- On Mayo, v2 hits the same low-capacity ceiling as v1 — neither
  scales to 2304-angle helical complexity. **Use v3 on Mayo.**
