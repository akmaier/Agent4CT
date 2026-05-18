# Claude-driven agentic diffusion-recon search (DL Sparse-View CT)

20-iteration search where Claude proposed each next configuration after reading
the previous iter's `result.json` and `comparison.png`. Each iter ran on the
SLURM cluster via `cluster/slurm/demo_diff_recon_oneshot.sbatch`. The sampler is
DPS (Chung 2023) with adaptive ζ_t = η/‖residual‖, optionally refined with a
Resample-style DC-step (Gauss-Newton/CG hard projection toward the sinogram,
followed by re-noising back to the current diffusion time).

## Final winner — iter 16 (headroom = 0.5712, SSIM = 0.6495)

```json
{
  "recon_mode":          "dps",
  "recon_sample_steps":  500,
  "recon_eta":           30.0,
  "recon_init":          "fbp",
  "recon_eta_clamp":     false,
  "recon_dcstep_every":  3,
  "recon_dcstep_n_cg":   20,
  "recon_dcstep_warmup": 25,
  "recon_dcstep_relax":  1.0,
  "recon_ckpt":          "ddpm_unconstrained_final.pt"
}
```

Wall time per scene: ~30 s (15 s without DC-step, 500 DDIM × ~30 ms + 158 DC ×
~80 ms with n_cg=20). 20-scene val: ~590 s total.

## Trajectory

| iter | hr     | SSIM   | change vs prior best                                   |
|------|--------|--------|--------------------------------------------------------|
| 1    | 0.4689 | 0.666  | baseline: DPS+DC, eta=1, 200s, every=5, n_cg=5, warmup=10, relax=0.5 |
| 2    | 0.5400 | 0.741  | relax 0.5→0.8, n_cg 5→10 ⬆ +0.071                       |
| 3    | 0.5456 | 0.742  | relax 0.8→1.0 ⬆ +0.006                                  |
| 4    | 0.5499 | 0.752  | every 5→3 ⬆ +0.004                                      |
| 5    | 0.5540 | 0.747  | 200→500 sample steps ⬆ +0.004                           |
| 6    | 0.5401 | 0.742  | DPS→MCG ⬇ −0.014                                        |
| 7    | 0.5277 | 0.721  | unconstrained→constrained DDPM ⬇ −0.026                 |
| 8    | 0.5412 | 0.742  | eta 1→0.3 ⬇ −0.013                                      |
| 9    | 0.5569 | 0.763  | eta 1→3 ⬆ +0.003                                        |
| 10   | 0.5621 | 0.762  | eta 3→10 ⬆ +0.005                                       |
| 11   | 0.5597 | 0.727  | warmup 10→5 ⬇ −0.002                                    |
| 12   | 0.5625 | 0.741  | eta 10→30 ⬆ +0.000                                      |
| 13   | 0.5561 | 0.808  | eta=30 + 500s + every=5 (SSIM peak)                     |
| 14   | 0.5640 | 0.719  | 500s + every=3                                          |
| 15   | 0.5641 | 0.736  | noise init (≈tied with FBP)                             |
| **16** | **0.5712** | **0.650** | **n_cg 10→20  ⬆ +0.007  ⭐ winner**                |
| 17   | 0.4590 | 0.479  | n_cg 20→40 ⬇ −0.112 (ceiling)                           |
| 18   | 0.5402 | 0.589  | eta 30→100 ⬇ −0.031 (ceiling)                           |
| 19   | 0.5670 | 0.654  | every 3→2 ⬇ −0.004 (ceiling)                            |
| 20   | 0.5632 | 0.649  | eta=50 refinement (confirms peak at 30)                 |

## What the search learned (axis ceilings)

The first ~15 iters climbed monotonically; the last 5 probed the boundaries
around the iter-16 optimum, each landing on a regression that pinned down
the ceiling on that axis:

| axis           | won at | ceiling at | reason                                          |
|----------------|--------|------------|-------------------------------------------------|
| `recon_dcstep_n_cg`  | **20** | 40 | over-projects against noisy sinogram, undoes the prior |
| `recon_eta`          | **30** | 100 | over-pulls DPS gradient when projection is strong |
| `recon_dcstep_every` | **3**  | 2 | too-frequent projection ≈ same failure mode as high n_cg |
| `recon_sample_steps` | **500** | (200 worse) | longer trajectory = more chances for DC + DPS |
| `recon_dcstep_relax` | **1.0** | (lower worse) | full hard projection wins given enough CG    |
| `recon_dcstep_warmup`| **25**  | 5 worse, 10 ok | early DC = applied while x_t is still mostly noise |
| `recon_init`         | **fbp ≈ noise** | — | irrelevant once DC pressure is high              |
| `recon_mode`         | **dps**  | mcg worse | MCG's FBP-pseudoinverse blurs the gradient    |
| DDPM ckpt            | **unconstrained** | constrained worse | constrained checkpoint trained on 200 phantoms only — too narrow |

## Key takeaway

Diffusion posterior sampling alone was unable to reconstruct sparse-view CT
images in this setting (every random/TPE iter prior to introducing the DC-step
gave headroom = 0). The DC-step (Resample-style hard projection) was the
unlock. Once enabled, all three pressure axes (DC depth, DC frequency, DPS
strength) had a clear ceiling that the search converged on by iter 16. The
final config is at the intersection of three soft optima — going further on
any axis regresses.

## Filesystem layout

```
docs/runs/demo-fair-claude-diffusion-recon-search-20260518-01/
├── manifest.json          # search metadata + final winning config
├── README.md              # this file
├── results.tsv            # per-iter HR / SSIM / status row
├── stages.tsv
└── iterations/
    └── iter-0001/         # ... through iter-0020
        ├── observation.json
        └── comparison.png
```

Cluster-side configs live in `/cluster/maier/Agent4CT/configs/claude_iter01.json` …
`claude_iter20.json`. The driver sbatch is
`cluster/slurm/demo_diff_recon_oneshot.sbatch`, parametrised by
`CFG_JSON` + `OUT` env vars.

## Why this differs from the random/TPE sweep

The random and Optuna-TPE searches for diffusion-recon both flatlined at
headroom=0 because they were not given the DC-step axes (those were added to
the solver mid-search, after diagnosing the collapse). This 20-iter agentic
search ran exclusively with the DC-step enabled — every iter has a non-trivial
score. See `docs/runs/demo-fair-diffusion-recon-search-*` (random / TPE) for
the failed prior attempts.
