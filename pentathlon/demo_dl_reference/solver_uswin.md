# `solver_uswin.py` — UNet-Swin hybrid

Companion design doc. A U-shaped backbone with Swin-Transformer
windowed-attention blocks instead of plain convolutions at the
bottleneck. Combines CNN's translation equivariance (encoder/decoder)
with self-attention's long-range modelling (bottleneck).

Knobs:
- `uswin_c`: base channel width (16-32).
- `uswin_win`: Swin window size (4, 8, 16).
- `uswin_heads`: number of attention heads (2-8).
- `epochs`, `lr`, `train_n`, `batch_size`: standard supervised-L2 knobs.

## When to use USwin vs DD-UNet sup L2 vs LPD

USwin sits between DD-UNet supervised L2 (plain UNet) and LPD (unrolled
physics) on the parameter spectrum. With Swin attention at the
bottleneck, USwin can model long-range dependencies a plain UNet
misses — useful for streak artifacts that propagate across the image.

## Cross-dataset record (filled in 2026-06-08 via Mayo Step-3 TPE)

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl`   | 0.4655 | TPE `demo-intensity-calibrated-tpe-uswin-search-20260520-01` (ep=14, lr=4.9e-4, c=24, win=8, heads=2, train_n=200) | rank 5 on demo-DL. Behind ItNet v3 (0.4676) and v1 (0.4665). |
| `breast_ct` | 0.7174 | TPE `breast-ct-calibrated-tpe-uswin-search-20260521-01` (ep=14, lr=1.8e-4, c=24, win=16, heads=8, train_n=200) | rank 6 on breast-CT. Bigger win + more heads on the denser anatomy substrate. |
| `mayo_ldct` | **0.2492** | TPE iter-11 winner (search-space-clamped, Mayo Step-3 phase 1) | **rank 3 on Mayo.** +75% over Step-2 agentic 0.1425. |

## 2026-06-08 — Mayo Step-3 TPE: +75% lift to rank 3

USwin was a Mayo Step-2 plateau positive at hr=0.1425 (iter-2 winner:
c=16, win=8, heads=8, ep=3, train_n=50). iter-3/4 OOM'd at FBP scratch;
iter-5 (ep=6) regressed to 0.107. Step-2 verdict was "plateaued".

Mayo Step-3 TPE (job 762xxx-uswin-search-20260608-03) ran 20-trial
Optuna TPE with Mayo-specific search-space clamps. **TPE iter-11
landed hr=0.2492** — +75% over the Step-2 agentic plateau.

Mayo USwin is now the **rank 3 entry on Mayo** behind DD-UNet sup TPE
(0.3890) and LPD TPE (0.3063).

### Mayo search-space clamp lessons

Mayo USwin TPE had a rocky start. The first dispatch (job 762897)
COMPLETED but **all trials hit hr=0** — TPE's default search-space
inherited from breast/demo had `train_n=200` which OOMed in
`filter_sino` (5 GiB FFT pad on Mayo's 2304-angle sino at Q6000
24-GB cap). This led to the Mayo-specific clamp work:

- `MAYO_CLAMPS` in `scripts/learned_solver_search_agent.py` clamps
  `train_n` to [50] when `--dataset=mayo_ldct_2d` (must auto-insert
  even if not in original space, via `ALWAYS_INSERT`).
- Also clamps `uswin_c` to [16] (Mayo Step-2 winner; the default
  search space allowed c=32 which OOMs gradient memory at Mayo's
  512² scenes).
- Clamps `epochs` to (5, 12, int) — Step-2's ep=6 LR-scheduled
  config TIMEOUTed at 90 min subprocess wall on Mayo; cap epochs
  ≤ 12.

Re-dispatched as job `mayo-ldct-2d-calibrated-tpe-uswin-search-20260608-03`
with the clamp; landed iter-11 winner at hr=0.2492.

### Cross-dataset USwin pattern

| Dataset | uswin_c | uswin_win | uswin_heads | epochs | Pattern |
|---|---:|---:|---:|---:|---|
| `demo_dl`   | 24 | 8  | 2 | 14 | Simpler substrate → narrower attention. |
| `breast_ct` | 24 | 16 | 8 | 14 | Denser anatomy → wider window + more heads. |
| `mayo_ldct` | 16 (clamped) | tuned | tuned | tuned | Memory cap forces narrow channel; TPE explores the rest within Mayo bounds. |

USwin's "Swin attention bottleneck on a UNet skeleton" works on every
dataset but has a steep memory cost — Mayo's 512² + 2304-angle
forces c=16 and bounded train_n. Within those caps, TPE still finds
+75% over the agentic plateau, putting USwin at Mayo rank 3.
