# `solver_itnet_v3.py` — ItNet v3 (canonical ItNet — deeper UNet + per-step α)

Companion design doc. For v1 see `solver_itnet.py`; for v2 see
`solver_itnet_v2.py`.

**v3 is the canonical ItNet variant** for the Agent4CT pentathlon —
the only one that lifts above baseline on Mayo (where v1 and v2 hit
hr=0). Two architectural changes vs v1/v2:

1. **Deeper UNet per step** — `unet_c` controls a 4-level UNet (vs
   v1/v2's shallow 2-level conv head). Default `unet_c=12` gives
   ~250 k params per step; v3 with k=3 has ~750 k total.
2. **Per-step learnable α** — instead of a single shared α (v1) or
   shared α-init (v2), v3 has a `nn.Parameter alpha` that the
   optimiser updates end-to-end. The unrolled trajectory learns its
   own per-iter step size.

## Cross-dataset record

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | **0.4676** | TPE `demo-intensity-calibrated-tpe-itnet-v3-search-20260520-01` (ep=13, lr=7.5e-4, unet_c=16, k=3, α=9.1e-3, train_n=200) | **#3 on demo-DL.** v3 leads the ItNet family but by tiny margin: v3 0.4676 ≈ v1 0.4665 > v2 0.4567. The deeper UNet's capacity doesn't pay off as much on simpler ellipse phantoms. |
| `breast_ct` | **0.7342** | TPE `breast-ct-calibrated-tpe-itnet-v3-search-20260521-01` (ep=15, lr=2.2e-4, c=16, k=2, α=2.6e-3, train_n=200) | rank 5 on breast-CT. Beats v2 (rank 7) by +0.20 hr at the same param count — deeper UNet earns its capacity here. |
| `mayo_ldct` | **0.2181** | TPE 762900 iter-9 (search-space-clamped, post-cfg-patch eae661bc), Mayo Step-3 phase 1 | **rank 6 on Mayo.** +63% over Step-2 agentic 0.1336. The ONLY ItNet variant that clears Mayo baseline. |

## 2026-06-08 — Mayo Step-3 TPE: +63% to rank 6

Mayo Step-2 iter-5 winner was c=16, k=3, ep=3, lr=5e-4, train_n=50,
hr=0.1336 (after cfg-patch eae661bc that fixed the silent-cfg-drop
bug). Step-2 verdict was "plateaued".

Mayo Step-3 TPE (job 762900, `mayo-ldct-2d-calibrated-tpe-itnet-v3-search-20260608-04`)
ran 20-trial Optuna TPE with Mayo clamps. **TPE iter-9 found
hr=0.2181** — +63% over Step-2.

Mayo ItNet v3 is now rank 6 (between USwin TPE 0.2492 and diff_recon
CON v4 TPE 0.1632). The deeper UNet + per-step α lets v3 absorb
Mayo's 2304-angle complexity in a way v1 and v2 cannot.

## cfg-merge bug history (2026-06-08 patch — commit eae661bc)

v3 was the first ItNet variant to have a proper `env_var` read for
the agentic JSON cfg (the original behaviour). v1 and v2 lacked it
and used hardcoded defaults — Mayo agentic loops OOM'd because of it.

Patch (commit eae661bc): unified v1/v2 to match v3's env-read
pattern. After patch, v1/v2 retries on Mayo cleanly received the
clamped configs but still landed hr=0 (capacity ceiling).

## CONFIG defaults

```python
CONFIG = {
    "train_n": 200,
    # ARCHITECTURE
    "unet_c": 12,                # 4-level UNet base channels
    "itnet_k": 3,                # unroll depth
    "alpha_init": 0.0037,        # initial per-step α (TV-lambda equivalent)
    # OPTIMIZER
    "epochs": 10,
    "batch_size": 20,
    "lr": 5e-4,
}
```

## Mayo-specific search-space clamp

When `--dataset=mayo_ldct_2d`, `MAYO_CLAMPS` in
`scripts/learned_solver_search_agent.py` injects:
- `unet_c` ∈ [16, 24] (Mayo Step-2 winner was c=16; lock here)
- `itnet_k` ∈ [2, 3] (Step-2 found k=3; k=4+ OOMs gradient memory)
- `epochs` ∈ [5, 12] int (Step-2 ep=3 was the winner; cap at 12 to
  fit the 90-min subprocess timeout)
- `train_n` = 50 (Mayo Q6000 24-GB cap on filter_sino FFT pad)

## Hints for the next autoresearch agent

- **v3 is the default ItNet for any new dataset.** Don't waste time
  on v1/v2 unless you have a specific reason (e.g., parameter budget
  ≤ 100 k makes v1's lighter heads attractive).
- On Mayo, the cfg-merge patch eae661bc is REQUIRED for v3 to honor
  agentic configs. If you see Mayo ItNet v3 OOMing at filter_sino
  despite a small `train_n` in your cfg, check `git log
  pentathlon/demo_dl_reference/solver_itnet_v3.py` for the env-read
  block — it should be at the top of `main()`.
- v3's `per-step α` learns to taper through the trajectory: typical
  end-of-training α schedule across k=3 steps is [larger, mid,
  smaller]. This is what makes v3 absorb data complexity that v1's
  shared-α and v2's no-α-update can't.
