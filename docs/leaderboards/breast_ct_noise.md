---
title: BreastCT-Noise leaderboard
description: No-retrain robustness re-evaluation of every Breast-CT solver on Poisson-noised (high-dose, I0=100k) test inputs. The ideal-data ranking nearly inverts.
---

# BreastCT-Noise leaderboard — noise-robustness re-evaluation

The [Breast-CT board](breast_ct.html) uses the Sidky DL-Sparse-View challenge
data, which is **ideal noiseless projection data by design** (the RMSE floor is
zero — a pure sparse-view inverse problem). This second board asks a different
question: **how robust is each solver to a little input noise it never trained on?**

**Setup (no retraining).** Each solver's *noiseless* best iter is re-evaluated on
the **same 200 held-out test cases**, but with the sinograms corrupted by
photon-counting (Poisson) noise at a high dose, **I0 = 100 000 photons/pixel**
(≈1–2 % noise at the thickest ray — mild). Supervised solvers **load their
noiseless-trained checkpoint and skip training**; per-scene / classical solvers
re-fit on the noisy input (that *is* their inference). The ground truth stays the
clean phantom, and the FBP baseline is recomputed from the *noisy* sinogram, so
`hr = max(0, 1 − rmse / rmse_noisy-FBP)` measures improvement over the noisy FBP.
This is exactly the realistic `M_phys = M + ε` extension the Sidky challenge flags
but does not study.

## Leaderboard — every solver (rendered from the registry)

Rendered live from `docs/runs/index/leaderboard.json`
([`build_registry.py`](../../scripts/build_registry.py)), same run-ids as Breast-CT
but scored on the noised inputs. `hr` is fair to rank *within* this board (all
solvers share the same noisy FBP baseline); the **absolute SSIM / PSNR** columns
(vs the clean truth) distinguish genuine robustness from baseline-gaming.

<div data-leaderboard="breast_ct_noise">loading leaderboard…</div>

<script src="../assets/table.js"></script>
<script src="../assets/leaderboard.js"></script>

## The finding — ideal-data performance does not predict noise robustness

The ranking **nearly inverts** relative to the [noiseless board](breast_ct.html):

- **`learned-primal-dual` becomes champion** (hr 0.72 → **0.93**, SSIM 0.9905) — its
  primal-dual data-consistency + learned regularization is genuinely noise-robust.
- **Bilateral / self-supervised / classical methods rise:** `manduca-bilateral`
  (21 params) 0.28 → **0.84**, `dual-domain-bilateral-n2i` → 0.79, `manhart-PWLS-TV`
  and `tv-iterative` 0.36 → 0.51 — explicit or implicit smoothing suppresses noise.
- **The noiseless champion collapses to last.** `dual-domain-supervised`, a pure
  supervised streak-remover, drops **0.89 → 0.00** (SSIM 0.9992 → 0.35): trained only
  on noiseless FBP, it is brittle to a distribution it never saw. The `fastdiff`
  constrained variants collapse likewise.
- **DC-unrolled networks (ItNet family) degrade moderately** (0.89 → 0.55–0.70): the
  data-consistency step re-injects some measurement noise, but the learned prior
  partly holds. `param-efficient` (195 params) holds relatively well (0.62 → 0.52) —
  its output bilateral filter helps.

**Take-away for the paper:** a leaderboard won under *ideal* conditions says little
about deployment robustness. The methods that top the noiseless board (supervised
denoisers optimizing exact inversion) are the *least* robust; the physically
regularized methods (primal-dual, bilateral, TV) that sit mid-pack on ideal data
are the most robust. Report both boards.

**DNF.** `naf` and `r2gaussian` (per-scene neural-implicit / Gaussian-splatting) are
included but do not complete — per-scene optimization scores ~20/200 cases even with
a 5-hour wall and `hr ≈ 0` on those, on noisy as on noiseless data.

## Methodology

Noise staging: [`data/stage_breast_noise.py`](../../data/stage_breast_noise.py)
(deterministic, seed 20260709). No-retrain eval:
[`scripts/score_breast_noise.py`](../../scripts/score_breast_noise.py) →
`score_breast_testset.py --noise-i0`. See paper plan §5.6.7.
