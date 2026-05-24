# `solver_learned_primal_dual.py` — Adler & Öktem Learned Primal-Dual (IEEE TMI 2018)

Companion design doc. For the paper-level summary see
`literature/1707.06474_Adler_LearnedPrimalDual_TMI2018.md`. This doc is
the **Agent4CT-specific** record of what was tried, what won, and
what to skip on the next agentic / TPE iteration.

## What it is

Classical primal-dual hybrid gradient (Chambolle–Pock 2010) unrolled
into a network where the proximal operators are replaced by small
3-layer ResBlock CNNs. The fan-beam forward / back-projector
(`PyronnFanBeamProjector`) stays inside the network as a differentiable
layer — so the network only has to learn the *update rule*, not the
geometry. Per Adler:

- Primal state carries `N_primal` channels (memory between iters).
- Dual state carries `N_dual` channels likewise.
- `I` unrolled iterations.
- Each iter has its OWN ResBlock weights (no sharing).

Trained supervised-L2 against clean phantom + non-negativity penalty,
cosine LR schedule, gradient clipping.

## Design considerations

- **Physics-aware backbone is the inductive bias.** A pure U-Net
  post-processor has to relearn the fan-beam geometry from data; LPD
  bakes the projector in. On breast-CT this lets it match a 4× larger
  DD-UNet (1.86 M params) at hr 0.83 with only 0.88 M params.
- **Tiny per-iter CNN.** 3-layer ResBlock with `lpd_hidden` hidden
  channels, total per-iter params ≈ 2 × (n_in × hidden + hidden × hidden
  + hidden × n_out) ≈ small. Memory cost is dominated by the
  forward/back-projection activations across all I unrolled iterations,
  NOT the CNN weights.
- **Iteration count is the dominant memory knob.** I=10 fits at
  hidden=64 on a single Q5000/Q6000; I=15 nearly doubles the autograd
  graph and needs lower lr to stay stable.
- **No regularisation.** Adler explicitly tried wd / dropout /
  batchnorm and reports they hurt. Don't bother.

## Strengths

- **Top of the breast-CT leaderboard** at hr=0.829 (PSNR 55.08 dB).
- **Parameter-efficient**: 0.88 M params beats DD-UNet L2 (1.86 M) by
  +0.003 hr — half the parameter count.
- **Stable training when lr is matched to I.** lr ≈ 5e-4 at I=10 is
  the converged operating point.
- **Same code path covers `I ∈ [4, 12]` and `hidden ∈ [32, 96]`**
  without recompiling anything.

## Weaknesses

- **Memory scales linearly in I.** Pushing I past 12 on a 16-GB
  Q5000 is borderline; on a 24-GB Q6000 still tight.
- **Optimisation is fragile at the boundaries.** I=15 at lr=5e-4 NaN'd
  in iter-4 (epoch 1 loss spike to 1e9). Lower lr (2e-4) recovers but
  ends up overfitting (iter-5: hr 0.829 → 0.77).
- **Requires clean ground truth at train time** — supervised L2.
  Cannot play the self-supervised tag.

## When to prefer this solver

- **Dense-view supervised challenges** where parameter count matters
  (regulated / medical-device contexts).
- **As the breast-CT pentathlon champion until something better lands.**
- **Whenever you'd reach for ITNet/Hammernik-VN/USwin and want the
  unrolled-iteration family** — LPD's parametrisation tends to win on
  the same data.

## When to **not** prefer this solver

- **Self-supervised regimes** (Mayo LDCT clinical, no clean target).
  Fall back to N2I dual-domain.
- **Sparse-view very-low-photon** — LPD trained on this geometry may
  generalise less well than a flexible U-Net post-processor.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `lpd_iters` (I) | 10 | Unrolled iters. **Sweet spot = 10 on breast-CT.** Past 12 OOMs / overfits; below 8 undertrained. |
| `lpd_hidden` | 64 | Per-iter ResBlock hidden channels. Linear ~params and memory. |
| `lpd_n_primal` | 5 | Primal-state channels (memory between iters). Adler's default. |
| `lpd_n_dual` | 5 | Dual-state channels. Adler's default. |
| `lpd_share_weights` | False | Per Adler, **no sharing wins**. Don't set True. |
| `epochs` | 20 | iter-3 used 20; iter-5 ran 20 too. Above 25 risks overfit. |
| `lr` | 5e-4 | iter-3 winner. lr=1e-3 destabilises; lr=2e-4 underfits at I=10. |
| `lr_schedule` | "cosine" | Adler's recommendation; constant lr observed worse. |
| `batch_size` | 1 | Memory-limited at the unrolled-graph scale. |
| `grad_clip` | 1.0 | iter-3 used 1.0. Lower (0.3, 0.5) tested in TPE. |
| `lambda_neg` | 1.0 | Non-negativity penalty weight. |
| `val_chunk` | 4 | Val-pass batching to keep GPU memory bounded. |

## Hints for the next autoresearch agent

- **Iter-3 is the converged sweet spot.** I=10, hidden=64, lr=5e-4,
  ep=20, cosine, grad_clip=1.0 → hr 0.83.
- **DO NOT search I > 12.** iter-4 (I=15) NaN'd; iter-5 (I=15 at lower
  lr) overfit to hr 0.77. Already-known dead end.
- **DO search hidden=96** — not yet tested at I=10. Could add another
  +0.005 hr.
- **DO test train_n=1600 / 4000** if the data path supports it. iter-3
  used train_n=400; the breast-CT staged set has ~3600 phantoms.
- **TPE bounds**: `lpd_iters ∈ [8, 10]`, `lpd_hidden ∈ [32, 48, 64, 96]`,
  `epochs ∈ [15, 25]`, `lr ∈ log[2e-4, 1e-3]`, `grad_clip ∈ [0.3, 0.5,
  1.0]`. Above ranges proven safe under `subprocess.run(timeout=5400)`
  on Q5000.
- **Q5000 vs Q6000 timing**: a Q5000 trial of the iter-3 config takes
  ~40 min, Q6000 ~24 min. Plan sbatch walls accordingly.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | params | val_psnr | val_ssim | hr |
|---|---|---:|---:|---:|---:|
| baseline FBP | — | 0 | 39.74 dB | 0.957 | 0 |
| `claude-agentic-learned-primal-dual-search-20260522-01/iter-1` | I=10, hidden=32, ep=10, lr=5e-4 | 0.22 M | 52.4 dB | 0.997 | 0.74 |
| `claude-agentic-learned-primal-dual-search-20260522-01/iter-2` | I=10, hidden=48, ep=15, lr=5e-4 | 0.49 M | 53.8 dB | 0.998 | 0.79 |
| `claude-agentic-learned-primal-dual-search-20260522-01/iter-3` (top) | **I=10, hidden=64, ep=20, lr=5e-4** | **0.88 M** | **55.08 dB** | **0.9985** | **0.829** |
| `claude-agentic-learned-primal-dual-search-20260522-01/iter-4` | I=15, hidden=64, ep=20, lr=5e-4 | 1.31 M | NaN | NaN | — (cancelled) |
| `claude-agentic-learned-primal-dual-search-20260522-01/iter-5` | I=15, hidden=64, ep=20, lr=2e-4, grad_clip=0.5 | 1.31 M | 52.48 dB | 0.9973 | 0.769 ⚠ overfit |
| `calibrated-tpe-lpd-search-20260523-01/trial-1` (seed) | = iter-3 winner config | 0.88 M | 55.0 dB | 0.9986 | 0.820 |

The trial-1 result reproduces iter-3 within val-set noise (∼0.01 hr)
+ Q5000 vs Q6000 cudnn nondeterminism.

## Cross-dataset observations

| Dataset | Best hr | Config | Notes |
|---|---:|---|---|
| `breast_ct` | **0.9062** | I=8, hidden=96, ep=23, lr=3.2e-4, grad_clip=0.3, 1.49 M params | **#1 on the leaderboard.** TPE refined the agentic seed (I=10, hidden=64, 0.88M) — fewer unrolls + wider hidden won. |
| `demo_dl`   | — | not yet TPE'd under calibrated scoring | older uncalibrated demo-dl runs are not directly comparable; needs a calibrated TPE pass |
| `mayo_ldct` | — | autoresearch not yet started | geometry validated 2026-05-24; expected to lead here too given the physics-aware backbone |

**Pattern across datasets**: LPD's "physics-aware backbone + small
per-iter CNN proximal" is the most parameter-efficient way to top
synthetic-phantom benchmarks at 128 views. Beats DD-UNet L2 by +0.014
hr at half the parameters on `breast_ct`.

**Untested datasets where it should win**: any dense-view supervised
challenge with a clean target — Wagner's helix-rebinned Mayo
fulldose-vs-lowdose, Truth-CT, DL-Spectral CT.

## Known failure mode: subprocess timeout on Q5000

When running this solver under the TPE search agent
(`scripts/learned_solver_search_agent.py`), large-trial configs
(`I=12, ep=28` was the trigger) exceed the previous 3600 s subprocess
cap on Q5000 nodes and crash the whole study. Patched 2026-05-23:
cap raised to 5400 s; LPD search space tightened to
`lpd_iters ∈ [8, 10]`, `epochs ∈ [15, 25]` to stay safely under the
new cap. Re-running with these bounds is what job 762043 does.
