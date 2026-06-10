# `solver_diffusion_recon.py` — DPS / MCG / DC-step posterior sampling on a frozen DDPM

DDPM-prior CT reconstruction: load a pre-trained unconditional DDPM as
the image prior, sample via DPS (Diffusion Posterior Sampling, Chung
et al. 2023) or MCG (Manifold-Constrained Gradient), with an optional
hard DC-step (Conjugate Gradient projection onto the data-fidelity
manifold every K steps).

The DDPM itself is trained by `solver_ddpm.py` and stored under
`/cluster/maier/Agent4CT/checkpoints/`. This solver only does sampling
+ data-fidelity correction; it never updates the DDPM weights.

## What it is

For each val sample:

1. Initialise `x_T` from `recon_init = "fbp" | "noise"`.
2. Reverse-diffuse `T = recon_sample_steps` steps, at each step
   computing the score `∇log p(x)` from the DDPM and adding a
   data-fidelity guidance term scaled by `recon_eta`.
3. Every `recon_dcstep_every` steps after the `recon_dcstep_warmup`
   timestep, do a hard projection onto the data-fidelity manifold via
   `recon_dcstep_n_cg` CG iterations of the normal equations
   `R^T R x = R^T g`, weighted by `recon_dcstep_relax`.
4. Return `x_0` clamped to `[0, tv_clip_max]`.

## Design considerations

- **Checkpoint = the prior.** Whatever distribution the DDPM was
  trained on is the prior; if your test data is from a different
  distribution, the sampling can fail catastrophically (recons look
  like the training distribution, not the truth).
- **DPS vs MCG**: DPS uses an isotropic Gaussian likelihood guidance
  (`∝ ‖R x − g‖²`); MCG uses a manifold-constrained gradient (more
  conservative). DPS won the demo-phantom TPE search.
- **DC-step (Resample-style hard projection)** trades a higher-quality
  recon for a slower one — each DC step is `n_cg` CG iterations of
  R^T R = R^T g.

## Strengths (when prior matches data)

- **Strong on out-of-distribution noise** because the prior fills in
  what the data doesn't say.
- **No re-training required** — only the sampling is per-sample.

## Weaknesses

- **Catastrophic failure when checkpoint distribution ≠ test
  distribution.** See "Empirical results" below.
- **Slow per-sample.** 500 reverse steps with CG every 3 steps is
  ~15–25 min per sample on a Q5000.

## Knobs (in `CONFIG`)

| Knob | Effect |
|---|---|
| `recon_ckpt` | **Path to the DDPM checkpoint**. THE critical choice — see "Empirical results". |
| `recon_mode` | "dps" \| "mcg". DPS won the demo TPE. |
| `recon_sample_steps` | Reverse-diffusion steps. 200 / 500 / 800 tested. 500 is the demo winner. |
| `recon_eta` | Data-fidelity guidance strength (adaptive scaling). |
| `recon_init` | "fbp" \| "noise". FBP-init won the demo TPE. |
| `recon_eta_clamp` | Optional displacement cap. False won the demo TPE. |
| `recon_dcstep_every` | DC-step cadence. 3 won the demo TPE (more frequent over-projects). |
| `recon_dcstep_n_cg` | CG iters per DC step. 20 won the demo TPE. |
| `recon_dcstep_warmup` | First timestep at which DC step is applied. 25 won the demo TPE. |
| `recon_dcstep_relax` | DC step relaxation factor. 1.0 won the demo TPE. |

## Hints for the next autoresearch agent

- **`recon_ckpt` is THE bottleneck.** The other knobs only tune the
  guidance and CG behaviour; if the DDPM checkpoint doesn't model the
  test data distribution, no hyperparam can rescue it.
- **Demo TPE winner** is the right seed-trial for any new
  checkpoint-data pairing:
  ```
  recon_mode=dps, recon_sample_steps=500, recon_eta=30,
  recon_init=fbp, recon_eta_clamp=false, recon_dcstep_every=3,
  recon_dcstep_n_cg=20, recon_dcstep_warmup=25, recon_dcstep_relax=1.0
  ```
- **Available checkpoints (2026-05-23)**:
  - `ddpm_unconstrained_final.pt` (3.86 MB, 2026-05-17) — **demo
    ellipses**, won the demo-phantom TPE.
  - `ddpm_constrained_final.pt` (2026-05-17) — demo ellipses, narrower
    distribution; loses to unconstrained on demo.
  - `ddpm_breast_unconstrained_final.pt` (2026-05-20) — breast tissue,
    **broken / under-trained** (see Empirical results).
  - `ddpm_breast_constrained_final.pt` (2026-05-20) — breast tissue,
    same issue.
- **If a TPE search on breast-CT returns all hr=0**, retrain the breast
  DDPM with more epochs / data before further TPE search — checkpoint
  quality is the problem.

## Cross-dataset observations

| Dataset | DDPM variant | hr | Best SSIM | Notes |
|---|---|---:|---:|---|
| `demo_dl` | unconstrained (train_n=2000, disjoint seeds) | 0.4530 | 0.825 | TPE iter-17; eta=30 DPS+DC-step seed config wins |
| `demo_dl` | constrained (train_n=200, same seeds as supervised) | 0.4418 | 0.809 | TPE iter-18; **+0.011 gap** when DDPM sees more training distribution |
| `breast_ct` | unconstrained breast (train_n=3600) | **0.000** | 0.463 | All 20 TPE trials hr=0 — checkpoint visibly under-trained, recons SSIM<0.5 vs FBP 0.957 |
| `breast_ct` | constrained breast (train_n=200) | **0.000** | 0.470 | Same — checkpoint also weak |
| `mayo_ldct` | — | — | — | DDPM not yet trained on Mayo |

**Pattern: this solver lives or dies by the checkpoint.** Two
qualitatively different outcomes here:

1. **`demo_dl`**: DDPM trained on simple Sidky ellipse phantoms
   converged cleanly; diff-recon hits hr≈0.45, competitive with
   ITNet v3 / USwin / RAM. The unconstrained-vs-constrained gap
   (+0.011 hr) is the empirical answer to "how much does more
   training distribution help?" — modest but positive.

2. **`breast_ct`**: same architecture, but the breast-DDPM checkpoint
   produces SSIM ~0.46 reconstructions across all 40 TPE trials of
   (constrained, unconstrained). Even though the metadata says the
   training went well (`val_loss=0.0013`, comparable to demo-DDPM's),
   the resulting prior cannot guide reconstruction. **The breast DDPM
   needs retraining** — likely a normalization, σ-schedule, or
   data-pipeline issue we haven't yet pinned down.

**Take-away for new datasets**: before running diff-recon TPE,
**validate the trained DDPM checkpoint** by sampling unconditional
images and comparing to truth-distribution. If samples look anatomically
plausible, run the diff-recon TPE. If they look like blurry mush (the
breast case), retrain the DDPM with different hyperparams before
spending TPE budget on a broken prior.

## Empirical results on breast-CT (128 views, intensity-calibrated)

`breast-ct-calibrated-tpe-diff-recon-dcstep-{,un}constrained-search-*`
(using the WRONG demo checkpoint, 2026-05-21/22): **all 20 trials hit
hr=0**, confirming the demo checkpoint doesn't model breast tissue.

`breast-ct-calibrated-tpe-diff-recon-dcstep-{,un}constrained-breast-search-20260523-01`
(using the correct breast checkpoint, dispatched 2026-05-23):

| variant | trial | config snippet | val_ssim | hr |
|---|---:|---|---:|---:|
| unconstrained breast | 1 (seed = demo winner) | dps, 500 steps, eta=30, every=3, n_cg=20 | 0.431 | 0 |
| unconstrained breast | 2 | dps, 800 steps, eta=4.3, init=noise, clamp=true | 0.411 | 0 |
| constrained breast   | 1 (seed = demo winner) | dps, 500 steps, eta=30, every=3, n_cg=20 | 0.470 | 0 |
| constrained breast   | 2 | dps, 800 steps, eta=4.3, init=noise, clamp=true | 0.350 | 0 |

**Baseline FBP is 0.957 SSIM / 39.6 dB.** All breast-DDPM trials are
0.43–0.47 SSIM (≈ 0.5× baseline) and PSNR 3–5 dB. The recons visually
look like blurry low-frequency artefacts on a uniform disc (see
`docs/runs/<slug>/iterations/iter-0001/comparison.png`). **This is
not a hyperparam problem — the breast DDPM checkpoint itself is
under-trained or has the wrong intensity normalisation.**

**Next move**: retrain the breast DDPM. The `ddpm` SOLVERS entry in
`scripts/learned_solver_search_agent.py` provides a TPE search over
DDPM training hyperparams (n_train, ch, n_steps, epochs, batch, lr,
weight_decay). Dispatch with `--solver ddpm --calibrated --dataset
breast_ct`. Once a usable checkpoint exists, re-run the
`diffusion_recon_dcstep_*_breast` TPEs with the new path.

## 2026-06-09 — Mayo Step-3 TPE phase 3 results (4 jobs COMPLETE)

Four Mayo TPE jobs (762933 CON v2, 762934 UNCON v2, 762935 UNCON v4,
762936 CON v4) closed 2026-06-09 with substantial above-Step-2 lifts.
All four converged on a **previously-unexplored very-low-eta corner**
that the agentic loop had clamped out (agentic eta range 1-30
inherited from breast-CT; Mayo TPE space extended to 0.3-30 log).

### Mode × prior eta-corner matrix (Mayo, val_n=5)

| Mode × DDPM | Optimum cfg | Step-2 (val_n=3) | TPE (val_n=5) | Lift |
|---|---|---:|---:|---:|
| UNCON v2 | eta=**0.31**, **noise** init, clamp=False, sample_steps=500, every=5, warmup=10, relax=0.95 | 0.2095 | **0.2352** | +12% |
| UNCON v4 | eta=**0.30**, **fbp** init, clamp=True, sample_steps=200, every=3, warmup=25, relax=1.0 | 0.1736 | **0.2377** | +37% |
| CON v2   | eta=**7.21**, fbp init, clamp=True, warmup=40, sample_steps=200, every=3, n_cg=10 | 0.0847 | **0.1071** | +26% |
| CON v4   | eta=**1.52**, fbp init, clamp=True, sample_steps=200, every=3, n_cg=20, warmup=25, relax=0.85 | 0.0981 | **0.1632** | +66% |

### Key cross-cutting findings

1. **UNCON modes converge at very-low eta (~0.3)**, CON modes prefer
   mid-eta (1.5-7). At eta<0.5 with `eta_clamp=True`, the DPS noise
   injection becomes essentially deterministic — DPS reduces to mostly
   CG-based data consistency with mild diffusion prior. The agentic
   loop's eta≥1 clamp (inherited from breast-CT priors) missed this
   regime entirely.

2. **Init/clamp split between DDPM priors:** v4 (ch=96, batch=2, 120 ep,
   8.594 M params) prefers `fbp init + clamp=True` in both modes; v2
   (ch=64, batch=2, 60 ep, 3.823 M params) prefers `noise init +
   clamp=False` in UNCON but `fbp + clamp=True` in CON. Higher-capacity
   prior wants a deterministic anchor; lower-capacity has more noise
   tolerance in UNCON.

3. **DDPM training quality is NOT predictive of DPS performance.** Mayo
   v4 had the best DDPM training (val ε-loss 0.0025 vs v2's 0.0049),
   but the initial diff_recon iter-1 ranked v4 LOWER than v2 (0.1466
   vs 0.2095). TPE then found the eta=0.3 corner where v4's higher
   capacity pays off (final v4 0.2377 > v2 0.2352).

4. **TPE reproducibility check**: UNCON v2 hit hr=0.2352 at iter-12
   AND iter-16 (same eta=0.31 corner) — TPE rediscovered the global
   optimum on independent prior-conditioned trials, strong signal that
   it's the true val_n=5 optimum.

5. **CON v4 had the broadest robust corner**: eta=0.55-2.0 with
   fbp+clamp=True all produce hr=0.1597-0.1632 (±0.002). UNCON v4's
   eta=0.30-0.39 corner was tighter (hr=0.2373-0.2377 across 3 iters).

### Mayo DDPM checkpoint history

Three Mayo DDPM training rounds:
- **v2** (ch=64, batch=2, 60 ep): val ε-loss 0.0049. DPS-best at val_n=5
  for both UNCON (0.2352) and CON (0.1071).
- **v3** (ch=96, batch=1, 60 ep): val ε-loss 0.0061. **Deprioritised** —
  UNCON best 0.0641, CON best 0.0686, ~⅓ of v2. Root cause: ¼ effective
  training (half batch size at 2.25× params).
- **v4** (ch=96, batch=2, 120 ep): val ε-loss 0.0025. Final DPS-best
  UNCON 0.2377 (rank 4), CON 0.1632 (rank 7). The v4 fix (pair ch=96
  with batch=2 AND double epochs to 120) proposed in the v3 verdict
  worked.

Search-space scaffolding added to `scripts/learned_solver_search_agent.py`
(commit 6fa14e1b): `diffusion_recon_dcstep_{,un}constrained_mayo_v{2,4}`
SOLVERS entries with `recon_eta ∈ (0.3, 30) log` (narrowed from breast's
3-60) and `tpe_seed_trial` matching the Step-2 agentic winner per
mode×prior. Mayo-specific `MAYO_CLAMPS` auto-injects val_n=5,
val_chunk=1, train_n=50 (unused — DPS doesn't train), batch_size=1.

### Mayo leaderboard impact

Diff_recon DCstep entries now occupy 4 Mayo ranks (out of top 12):
rank 4 UNCON v4 0.2377, rank 5 UNCON v2 0.2352, rank 7 CON v4 0.1632,
rank 8 CON v2 0.1071. Together with LPD TPE (rank 2, 0.3063) and
DD-UNet sup TPE (rank 1, 0.3890), the diff-recon family is the
**largest single-architecture cluster** in the Mayo top 12.
