# `solver_dual_ddomain_supervised.py` — Dual-domain U-Net, supervised L2 (full 128 views)

Companion design doc. For the self-supervised version of the same
architecture see `solver_dual_ddomain_n2i.md`. For the
parameter-light bilateral version see
`solver_dual_ddomain_bilateral_supervised.md`.

## What it is

The supervised-loss twin of `solver_dual_ddomain_n2i.py`. Same
`SmallUNet` denoisers in projection and image domain, same fan-beam
projector — but trained on the **full 128-view forward pass** with
**MSE against the clean phantom + non-negativity penalty**, no
Noise2Inverse half-set split.

```
pred = img_dn( R_full.fbp( proj_dn(sino_full) ) )
loss = MSE(pred, clean_phantom) + 1.0 · negativity_penalty(pred)
```

The pipeline class is `FullViewUNetPipeline` (defined inline; mirrors
`FullViewBilateralPipeline` in the BF supervised solver).

## Design considerations

- **Built 2026-05-22 after finding** that N2I systematically
  over-smoothed dual-domain dense-view reconstructions on breast-CT
  (see `docs/findings.md`). Same dataset, same U-Net width, same lr
  schedule — only the loss changes — and the headroom went 0 → 0.81.
- **Needs the clean phantom at train time**, so this variant is
  **only fair against other supervised baselines**. Stage / score it
  alongside RAM zero-shot (uses pretrained checkpoint),
  hammernik / itnet-v3 (also supervised), not against pure N2I
  baselines.
- **Same SmallUNet architecture** as the N2I variant — keeping it
  fixed isolates the loss as the lever (this was the autoresearch
  experiment that lit the path: c=16 c=8, both with supervised L2,
  both clearly above all N2I variants).
- **Full 128 views, no half-set split** — the dual-domain
  interpretation no longer needs the noise-as-target trick.

## Strengths

- **Saturation-level results on dense-view breast-CT.** At c=16, hr ≈
  0.81 (PSNR 54.25 dB vs FBP 39.74 dB, +14.5 dB). Visually
  indistinguishable from truth.
- **Capacity scales clean**: c=8 (≈120 k params) → hr 0.77, c=16
  (466 k) → hr 0.81. Diminishing returns set in fast — the loss is
  doing most of the work, not the capacity.
- **No extra knobs**. Drop-in replacement for the N2I solver via
  `claude_agentic_one_iter.py` SOLVER_MAP key `dual_domain_supervised`.

## Weaknesses

- **Requires clean ground truth** at train time. Cannot be used in
  truly unsupervised regimes (clinical low-dose where there's no
  paired clean acquisition, real XRM with only one dose level, etc.).
- **Will saturate fast on tasks with available headroom.** Once hr >
  0.7, val PSNR is in the 50+ dB range; further gains are perceptual
  / SSIM-only and likely noise-floor limited.
- **Black-box failure modes.** If supervised L2 doesn't help (e.g.,
  challenge where FBP already exhausts available info), you won't see
  *why* in the per-epoch log — the SmallUNet has no interpretable
  intermediate state.

## When to prefer this solver

- **Dense-view supervised challenges** where a clean phantom or
  paired clean acquisition is available. Breast-CT (DL-Sparse-View),
  any simulation track.
- **Pentathlon all-rounder phase** where multiple challenges share
  paired clean data and a unified architecture wins on
  cross-challenge averaging.
- **Whenever the N2I variant is stuck at hr ≈ 0** — first move
  should be "switch to this loss".

## When to **not** prefer this solver

- **No clean target at train time** (Mayo LDCT clinical, real XRM).
  Fall back to `solver_dual_ddomain_n2i.py` and accept the N2I
  smoothing.
- **You care about parameter count**. Use
  `solver_dual_ddomain_bilateral_supervised.py` (6–18 params) — same
  loss, hr ≈ 0.25 on breast-CT instead of 0.81 but at < 0.01% of
  the parameters.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `unet_c` | 16 | Channel width. c=8 → hr 0.77, c=16 → hr 0.81; 2× cost. |
| `epochs` | 10 | More likely helps here than in N2I (no noise-floor over-smoothing pressure). |
| `lr` | 5e-4 | Conservative; 1e-3 also works on smaller c. |
| `lambda_neg` | 1.0 | Weight on the non-negativity penalty. |

## Hints for the next autoresearch agent

- **Capacity sweep**: c=4 / c=8 / c=16 / c=32. The c=8 → c=16
  improvement was small (hr 0.77 → 0.81); c=4 and c=32 weren't
  tested. There may be a diminishing-returns curve worth
  characterising.
- **Epoch sweep**: only 10 epochs run so far. Train loss may still be
  dropping — try 20-30 epochs for a saturating curve.
- **Data sweep**: train_n=400 used; the breast-CT staged dataset has
  3600 train phantoms available. More data plausibly pushes hr past
  0.81 since this model is supervised (no overfitting collapse seen
  yet).
- **Stage check the iter-1 winner** on the 3× larger val set
  (existing infrastructure) to confirm hr=0.81 is real and not
  iter-set-specific.
- For the **all-rounder** phase, this is the new default dual-domain
  reference architecture — the N2I variant should be reserved for
  challenges that genuinely lack clean targets.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | val_psnr | val_ssim | hr |
|---|---|---:|---:|---:|
| baseline FBP | — | 39.74 dB | 0.957 | 0 |
| `claude-agentic-dual-domain-unet-l2-search-20260522-01/iter-1` | c=16, ep=10, lr=5e-4 | **54.25 dB** | **0.9986** | **0.812** |
| `claude-agentic-dual-domain-unet-l2-search-20260522-01/iter-2` | c=8, ep=10, lr=5e-4 | 52.53 dB | 0.9979 | 0.771 |

For reference, same architecture with N2I loss
(`solver_dual_ddomain_n2i.py`): val_psnr ≈ 38 dB, hr=0.
