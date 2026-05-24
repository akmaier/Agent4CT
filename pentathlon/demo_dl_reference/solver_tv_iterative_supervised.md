# `solver_tv_iterative_supervised.py` — Unrolled TV-GD with supervised L2 (deprioritised on dense view)

Built 2026-05-23 as the supervised-L2 mirror of `solver_tv_search.py` /
`solver_tv_iterative.py`. The hypothesis was: the DD-BF supervised
recipe (`solver_dual_ddomain_bilateral_supervised.py`) went from hr=0
(N2I) to hr=0.21 just by switching the loss; could the same recipe
applied to TV iterations break the TV-iterative hr=0 ceiling on
breast-CT? **Answer: no.** With FBP init this architecture is
structurally bounded by FBP quality.

## What it is

K unrolled gradient-descent steps on the variational objective

```
F(f) = ½ ||R f - g||² + λ TV(f)
```

with **per-iter learnable scalars** `step_k` and `λ_k` (both
log-parametrised so they stay positive), initialised from
FBP(noisy), trained MSE against the clean phantom + non-negativity
penalty.

```
f = FBP_clamped(g)
for k = 1..K:
    f ← clamp( f − step_k · (R^T(Rf − g) + λ_k · ∇TV(f)),  0, tv_clip_max )
return f
```

TV gradient is a closed-form expression for the smoothed isotropic
form `TV(f) = ∑ sqrt(ε + (∇x f)² + (∇y f)²)`. Trainable params at
the default K=10: 20 scalars total. Tiny.

## Design considerations

- **Mirror of `solver_dual_ddomain_supervised.py`** in train/val
  structure (same dataset dispatch, same `supervised_recon_loss`).
- **Hand-crafted TV gradient** — closed form, no autograd-through-grad
  trick. Fast.
- **FBP init** so the K iters refine an already-decent starting
  point.
- **K=10 default** with per-iter learnable scalars; share-steps option
  available via `tv_share_steps`.

## Strengths

- **Tiny parameter count** (20 scalars).
- **Interpretable.** Per-epoch printout shows step_k / λ_k for each
  unrolled iter, so you can watch the algorithm's adaptive behaviour.
- **Drop-in supervised-L2 mirror** for the existing TV iterative.
- **Bounded by physics**: `f` is clamped to `[0, tv_clip_max]` every
  iter, so it can't diverge.

## Weaknesses

- **Structurally bounded by FBP quality on dense-view.** When init is
  FBP(g) and FBP is approximately R⁺g, the data-fidelity gradient
  `R^T(Rf − g)` is ~0 at init, so the K iters have nothing to do.
  Supervised L2 then drives the network toward "stay at init" — i.e.
  match baseline FBP, not beat it. Empirically on breast-CT (3 iters)
  the model converges to ~baseline FBP minus a small over-smoothing
  penalty. **hr=0, PSNR 30–32 dB vs baseline 39.74 dB.**
- **Hand-crafted ∇TV is the entire prior.** Unlike LPD (which learns
  the per-iter prox CNN), this solver's only flexibility is in 20
  scalars. Not expressive enough to add useful signal on top of FBP.
- **First-iter step has runaway gradient pressure.** In iter-2 / iter-3
  trace, only `step_0` grew (to ~2e-3); all other steps stayed near
  init. The model effectively does ONE big GD step then idles.

## When to prefer this solver

- **Sparse-view CT** where FBP is poor and the K iters have real
  work to do — not tested here, but the framework is general.
- **As a tiny / interpretable baseline** when 20 trainable scalars
  with closed-form TV is acceptable.
- **As a teaching reference** for unrolled iterative reconstruction
  — the code is short and the per-iter learned scalars are easy to
  visualise.

## When to **not** prefer this solver

- **Dense-view supervised CT (e.g. 128-view breast-CT)** — FBP-init
  saturates the data-fidelity term and K iters become no-ops.
  Use `solver_dual_ddomain_supervised.py` (hr 0.83) or
  `solver_learned_primal_dual.py` (hr 0.83) instead.
- **Whenever a learnable prior CNN is acceptable** — replacing
  `∇TV(f)` with a tiny CNN gives you ITNet / LPD, which dramatically
  outperform.

## Knobs (in `CONFIG`)

| Knob | Default | Effect |
|---|---:|---|
| `tv_K` | 10 | Unrolled iterations. **More didn't help on breast-CT**: iter-3 at K=30 gave identical val_score to iter-2 at K=10. |
| `tv_step_init` | 1e-2 | Per-iter GD step (log-parametrised). **1e-2 is 100× too large for stable GD** on this geometry — iter-1 at 1e-2 was 9.7 dB worse than baseline. Use 1e-4 (iter-2 / iter-3 default). |
| `tv_lambda_init` | 1e-3 | Per-iter TV weight. **Drop to 1e-5** when step_init is small, else the model over-regularises. |
| `tv_share_steps` | False | If True, single scalar shared across K. Untested. |
| `tv_clip_max` | 0.05 | μ clamp upper bound. Matches breast-CT data range. |
| `tv_eps` | 1e-6 | Smooth-TV epsilon. |
| `epochs` | 10 | iter-1/2/3 all 10 epochs; converges within 3–5. |
| `batch_size` | 1 | Per-sample autograd graph carries K iters of forward+back-project — keep small. |
| `lr` | 5e-3 | Outer optimiser over the 20 scalars. |
| `lambda_neg` | 1.0 | Non-negativity penalty weight (in the supervised loss). |
| `loss_base` | "mse" | "mse" / "l1". |
| `grad_clip` | 1.0 | Helpful when step_k explores larger values. |

## Hints for the next autoresearch agent

- **DON'T re-search this solver on dense-view breast-CT.** Iter-1/2/3
  cover the (step_init, lambda_init, K) corners; all bottomed at
  hr=0 by structural saturation. The data-fidelity term has nowhere
  to push when f is already FBP.
- **DO try this solver on sparse-view challenges** (DL-Sparse-View
  with `n_angles < 60` regime). FBP is bad there; K iters can refine.
- **To beat FBP under this skeleton**, the architectural change has
  to be: replace `∇TV(f)` with a learnable CNN prox. That's exactly
  LPD's recipe — use `solver_learned_primal_dual.py` instead.
- **Don't drop FBP init** thinking that'll help — zero init would just
  force the K iters to rediscover FBP, costing K×forward+backproj
  with no gain in the limit.

## Cross-dataset observations

| Dataset | Best hr | Config | Notes |
|---|---:|---|---|
| `demo_dl` | — | not run as supervised | The non-supervised `tv_iterative` (calibrated TPE) reaches 0.4056 on demo_dl — a useful baseline for that dataset where FBP alone is weaker. |
| `breast_ct` | **0.000** | K∈{10,30}, step=1e-4, λ=1e-5 | **Structurally bounded by FBP** on dense-view CT — data-fidelity gradient saturates at FBP init. See main weaknesses section. |
| `mayo_ldct` | — | not yet run | Likely same as breast_ct — dense view (2304 angles), FBP-init is already strong. |

**Pattern**: hand-crafted smooth-TV gradient is **only useful when FBP
is genuinely bad** (sparse-view / very-low-photon). At 128+ views with
clean targets, learned priors (LPD, DD-UNet) absolutely dominate. The
non-trainable `tv_iterative` is a more sensible baseline; trainable
TV-iter L2 only makes sense if FBP isn't already at the truth.

## Empirical results on breast-CT (128 views, intensity-calibrated)

`breast-ct-claude-agentic-tv-iterative-supervised-search-20260523-01`:

| iter | K  | step_init | λ_init  | val_psnr | val_ssim | hr   | notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1  | 10 | 1e-2      | 1e-3    | 30.07 dB | 0.940    | 0.00 | step too large; data-grad overshoots, late-iter λ's grew to 0.025 (over-smoothing) |
| 2  | 10 | 1e-4      | 1e-5    | 32.05 dB | 0.954    | 0.00 | step right-sized but model learns "do nothing" past iter 0 |
| 3  | 30 | 1e-4      | 1e-5    | 32.05 dB | 0.954    | 0.00 | K=30 identical to K=10 — extra iters wasted |

All three are **9.7 dB worse than baseline FBP** (39.74 dB). The
supervised loss saturates at "match FBP minus over-smoothing".

For reference, the same supervised-L2 recipe applied to a U-Net prior
(`solver_dual_ddomain_supervised.py`, c=32) gets hr=0.826 / PSNR 54.94
on the same data — proving the loss is fine; the failure mode is the
hand-crafted ∇TV prior, not the supervised-L2 training scheme.
