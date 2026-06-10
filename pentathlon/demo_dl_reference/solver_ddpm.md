# `solver_ddpm.py` — DDPM prior training (companion to `solver_diffusion_recon.py`)

Companion design doc. This solver **does not reconstruct CT scans**;
it trains a DDPM prior on the dataset's μ-distribution, then saves
the checkpoint for `solver_diffusion_recon.py` to use as a frozen
posterior-sampling prior.

The DDPM training script is structured as a "solver" so the
autoresearch + TPE infrastructure can search over training
hyperparameters (channels, batch size, epochs, schedule length)
the same way they search over reconstruction hyperparameters.

## Architecture

`SmallDDPM` — a 3-level UNet with `ddpm_ch` base channels,
sin-positional time embedding, and standard DDPM forward/reverse
schedule. T = `ddpm_n_steps` noise steps with cosine β-schedule.
Output normalisation: μ-values → [0, 1] via dataset-fixed
`ddpm_out_scale` (typically 0.05 for breast-CT, ~0.03 for Mayo).

Two training modes:
- **`unconstrained`**: trained on `ddpm_n_train` images sampled
  freely from the full pool (typically 2000-3600 random seeds).
  Larger prior; richer distribution coverage.
- **`constrained`**: trained on `ddpm_n_train_constrained` images
  drawn from the SAME split as the supervised solvers' train set
  (typically 200 images). Smaller prior; no test-distribution
  leakage when paired with diff-recon TPE.

## Per-dataset training history

### demo-DL DDPM

Trained 2026-05-20. Two ckpts:
- `ddpm_constrained_final.pt` (train_n=200, ch=32)
- `ddpm_unconstrained_final.pt` (train_n=2000, ch=32)

Best val ε-loss: ~0.0013. Both ckpts produce visually plausible
unconditional samples (Sidky-style ellipses). Used by
`solver_diffusion_recon.py` to reach demo-DL rank 7/8 (UNCON 0.4530,
CON 0.4418).

### breast-CT DDPM — multiple retraining rounds

| Version | Date | ch | epochs | batch | val ε-loss | Diff-recon hr | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| v1 | 2026-05-22 | 32 | 25 | — | 0.0066 | 0 | underfit — produces SSIM 0.46 blurry mush |
| v2 | 2026-06-02 | 64 | 80 | 2 | 0.0050 | 0 | better training but DPS-recon still hr=0 |
| v3 | 2026-06-03 | 128 | 60 | 1 | 0.0020 | 0 | bigger UNet, lower val loss, BUT DPS-recon STILL hr=0 |

**3 ckpt arches, all yield hr=0 on breast-CT diff-recon.** Logged in
`solver_diffusion_recon.md` and findings.md (2026-06-03 entry). The
SmallDDPM architecture cannot be the breast-CT prior — **the failure
is the prior class, not capacity or training duration**. Future work
needs a structurally different prior (score-SDE / EDM / U-ViT).

### Mayo DDPM — successful

| Version | Date | ch | batch | epochs | val ε-loss | Best diff-recon hr |
|---|---|---:|---:|---:|---:|---:|
| v2 | 2026-06-04 | 64 | 2 | 60 | 0.0049 | UNCON 0.2352 / CON 0.1071 (Step-3 TPE) |
| v3 | 2026-06-05 | 96 | 1 | 60 | 0.0061 | UNCON 0.0641 / CON 0.0686 — **deprioritised** (¼ effective training) |
| v4 | 2026-06-08 | 96 | 2 | 120 | 0.0025 | UNCON 0.2377 / CON 0.1632 (Step-3 TPE) |

**Lesson: pair channel-count with batch-size + double epochs.** The
v3 round used `batch=1` at 2.25× the v2 params, resulting in ¼ the
effective gradient updates per parameter. v4 fixed this with
`batch=2, ep=120`. Final result: v4 leads UNCON, v2 leads CON
narrowly (both at val_n=5).

**DDPM training quality is NOT predictive of DPS performance.** v4's
val ε-loss of 0.0025 (2× better than v2's 0.0049) does not directly
translate to better DPS-reconstruction. The Step-3 diff-recon TPE
had to discover the eta=0.3 corner for v4 to outperform v2 in
UNCON mode (final 0.2377 vs 0.2352 — close).

## CONFIG defaults

```python
CONFIG = {
    # DATA
    "ddpm_mode":              "unconstrained",  # or "constrained"
    "ddpm_n_train":           3000,             # used only when unconstrained
    "ddpm_n_train_constrained": 200,            # matches other dl_ref solvers' train_n
    "ddpm_n_val":             100,              # held-out eps-loss val set
    "ddpm_out_scale":         0.05,             # normalise μ → [0,1] for DDPM
    # ARCH
    "ddpm_ch":                32,               # base channels of the 3-level UNet
    "ddpm_n_steps":           1000,             # noise schedule length T
    # OPTIMIZER
    "ddpm_epochs":            30,
    "ddpm_batch":             8,
    "ddpm_lr":                2e-4,
    "ddpm_weight_decay":      0.0,
    # WALL
    "ddpm_train_wall_s":      3600,             # 1 hour cap for one training
    # OUTPUT
    "ddpm_ckpt":              "/cluster/maier/Agent4CT/checkpoints/ddpm_search.pt",
    "ddpm_keep_search_ckpts": False,
}
```

## Hints for the next autoresearch agent

- **Before dispatching diff-recon TPE on a new dataset, validate
  the DDPM ckpt**. Render 10-20 unconditional samples and visually
  inspect — if they look like blurry mush, the training is broken
  and no amount of diff-recon hyperparam search will help.
- For training: **pair `ddpm_ch` with `ddpm_batch` + scale
  `ddpm_epochs`**. v3's mistake was ch=96 + batch=1 at the same
  epoch count as v2 (ch=64 + batch=2) — half the gradient updates
  per parameter.
- The `constrained` mode is the right ckpt for diff-recon TPE on a
  benchmark where you want no test-distribution leakage. The
  `unconstrained` mode is the right ckpt when you want the strongest
  prior possible.
- Mayo trains DDPM in ~13 min per epoch at ch=96, batch=2. v4's
  120-epoch run took ~26h. For new datasets, budget similar wall
  per architecture choice.
- The DDPM checkpoint footprint at ch=96 is ~33 MB. Don't keep all
  search ckpts (`ddpm_keep_search_ckpts=False`) unless explicitly
  needed — a 20-trial TPE produces 600+ MB of useless intermediate
  ckpts otherwise.
