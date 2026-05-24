# `solver_dual_ddomain_n2i.py` — Dual-domain U-Net, Noise2Inverse self-supervised

Companion design doc. For the supervised-L2 variant of the same
architecture see `solver_dual_ddomain_supervised.md`. For the
parameter-light bilateral version see `solver_dual_ddomain_bilateral_n2i.md`.

## What it is

Wagner-style dual-domain learned reconstruction: two `SmallUNet(c)`
denoisers — one in projection (sinogram) domain, one in image domain —
trained jointly via Noise2Inverse self-supervision (`DualDomainPipeline`
in `ddssl_ldct/training.py`).

```
sino_full ─┬→ split into half-sets a,b (64 angles each) ─┐
           │                                             │
           │  pipeline(x) = img_dn( R_half.fbp( proj_dn(x) ) )
           │                                             │
           ├→ y_hat_a = pipeline(x_a)   y_tgt_b = R_half.fbp(x_b)
           └→ y_hat_b = pipeline(x_b)   y_tgt_a = R_half.fbp(x_a)
                                                         │
                          loss = ½(MSE(y_hat_a, y_tgt_b) + MSE(y_hat_b, y_tgt_a))
                                + 1.0 · ½(neg_pen(y_hat_a) + neg_pen(y_hat_b))
```

No clean target is used at training time. At inference, the prediction
is the symmetric average of the two half-set pipelines.

## Design considerations

- **Self-supervision** — the only useful target available when the
  ground truth is unobservable (Mayo LDCT-style clinical settings,
  low-dose XRM, any case where you have noisy measurements but not
  clean ones).
- **The half-view FBP target carries an irreducible noise floor.**
  Training MSE against a noisy reference *rewards* over-smoothing
  because over-smoothed predictions reduce variance against random
  noise. The smoother the pipeline, the lower the train loss.
- **Architecture choice (SmallUNet c=16)** — modest capacity that
  scales the same way in both domains. Default 0.47 M params total.
- **Reconstruction backbone (PYRO-NN PyTorch backend)** is fully
  differentiable, so the dual-domain joint optimisation is
  end-to-end with no detached operators.

## Strengths

- **Honest self-supervision** — no clean labels needed.
- **Inductive bias from the dual-domain split** — the model can't
  cheat by collapsing to identity in either domain; it has to do
  meaningful work in both.
- **Capacity is tuneable** via the `unet_c` config knob alone.
- **Generalises well to sparse-view regimes** — the half-view FBP
  target's noise floor is small relative to sparse-view streak
  artefacts, so MSE against it provides a useful gradient.

## Weaknesses

- **Over-smooths on dense-view scans.** On breast-CT (128 views,
  high-quality FBP baseline at 39.74 dB / 0.957 SSIM), every U-Net
  width tested (c=4, c=8, c=16) plateaued at val_ssim 0.967 / val_psnr
  ≈ 38 dB / headroom = 0 — *below baseline FBP in PSNR*. Same N2I
  loss that wins on sparse-view loses on dense-view.
- **Loss-as-bottleneck**: doubling capacity (c=8 → c=16) gave no
  measurable gain on breast-CT — *the loss formulation, not the
  network, was the limit.*
- **Black-box failure mode**: when the BF variant overshoots in N2I
  you can see σ_x growing across epochs; with the U-Net, the
  over-smoothing happens silently inside the conv weights.

## When to prefer this solver

- The challenge has **no clean ground-truth** at train time and the
  measurements are **sparse-view** (so the half-view FBP carries
  real signal above its noise floor).
- The task is **DL-Sparse-View** (where N2I-trained DD-UNet is the
  standard reference) or **Mayo LDCT** (where the clinical low-dose
  has structure even in half-view splits).

## When to **not** prefer this solver

- **Dense-view scans** like breast-CT 128 views — N2I systematically
  over-smooths here. Use `solver_dual_ddomain_supervised.py` if a
  clean target is available.
- When **PSNR is the headline metric** and the baseline FBP is
  already > 38 dB — N2I MSE will likely drop below baseline PSNR.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `unet_c` | 16 | Channel width. Capacity scales O(c²). |
| `epochs` | 8 | More epochs → more over-smoothing on dense scans. |
| `lr` | 1e-3 | Standard Adam. lower for c≥32. |
| `batch_size` | 1 | Larger only if memory allows; affects N2I noise estimate. |

## Hints for the next autoresearch agent

- If you're targeting breast-CT or any dense-view challenge, **switch
  to `solver_dual_ddomain_supervised.py` first** — it's the single
  biggest lever (this solver: hr 0 → supervised variant: hr 0.81 at
  c=16, +16 dB PSNR on the same network).
- If you must stay self-supervised, don't waste iters on U-Net width:
  capacity is not the lever for this loss on this dataset.
- Wagner's paper used different `tv_lambda` schedules and longer
  training (≥ 50 epochs) on sparse-view abdomen CT — those land in
  a different operating point than 8-epoch breast-CT runs.

## Cross-dataset observations

| Dataset | Best hr | Notes |
|---|---:|---|
| `demo_dl` | 0.3811 | TPE iter-17. N2I works OK on simple sparse-view phantoms — the half-set FBP target has real signal there. |
| `breast_ct` | **0.000** | Loss-bottleneck — N2I rewards over-smoothing on dense scans. Switching to the supervised-L2 twin gets hr=0.83. |
| `mayo_ldct` | — | Not yet run. Mayo is dense-view (~2304 angles) — N2I unlikely to help here either. |

**Pattern**: use N2I when (a) no clean target is available AND (b)
views are sparse enough that half-view FBP carries genuine signal.
For everything else, use `solver_dual_ddomain_supervised.py`.

## Empirical results on breast-CT (128 views, intensity-calibrated)

| Source | Config | val_psnr | val_ssim | hr |
|---|---|---:|---:|---:|
| baseline FBP | — | 39.74 dB | 0.957 | 0 |
| `claude-agentic-dual-domain-search-20260521-01/iter-1` | c=4, ep=2, lr=1e-3 | 38.03 dB | 0.969 | 0 |
| `claude-agentic-dual-domain-search-20260521-01/iter-5` | c=16, ep=5, lr=5e-4 | 38.01 dB | 0.967 | 0 |
| 20-iter TPE search | c ∈ [4,32], ep ∈ [2,8] | best 38 dB | 0.967 | 0 |

Same architecture trained with supervised L2 (see `solver_dual_ddomain_supervised.py`):
hr=0.812 (PSNR 54.25 dB at c=16, PSNR 52.53 dB at c=8).
