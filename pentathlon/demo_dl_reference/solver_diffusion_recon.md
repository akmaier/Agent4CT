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
