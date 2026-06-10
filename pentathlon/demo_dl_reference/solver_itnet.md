# `solver_itnet.py` — ItNet v1 (original Genzel-Hauptmann-Schaller 2022)

Companion design doc. For the v2 (gradient clipping + LR schedule
stable training) see `solver_itnet_v2.py`; for the canonical v3
(deeper UNet + per-step α) see `solver_itnet_v3.py`.

Original "Solving inverse problems with deep learning" style: a stack
of `itnet_k` lightweight residual UNet blocks unrolled through `k`
gradient-update steps, sino-domain pre-train followed by image-domain
end-to-end finetune.

## When ItNet v1 vs v2 vs v3

| Variant | Distinguishing feature | Best dataset | Best hr |
|---|---|---|---:|
| **v1** (this solver) | Original recipe — no per-step α, no residual_learning toggle | demo-DL | 0.4665 |
| v2 | Adds `residual_learning` toggle + LR schedule fixes | breast-CT | 0.5386 |
| v3 | Deeper UNet (vs shallow proj/img heads) + per-step learnable α | breast-CT | 0.7342, Mayo 0.2181 |

## Cross-dataset record (filled in 2026-06-09 after inventory gap closure)

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | **0.4665** | TPE 762957 (2026-06-09) | **Rank 4 — beats v2 (0.4567) and ties v3 (0.4676)!** k=2, c=16, pretrain_ep=6, lr=5e-4, α=5e-3, finetune_ep=10, finetune_lr=1e-4. On demo-DL's broader synthetic substrate, v1 with low-k + smaller-α is competitive with v3. |
| `breast_ct` | **0.1703** | TPE 762956 (2026-06-09) | Rank 15. k=8, c=8, pretrain_ep=5, lr=1e-3, α=0.015, finetune_ep=11, finetune_lr=5e-4. Confirms ItNet family transfer pattern: v1 < v2 < v3 on breast-CT (0.1703 → 0.5386 → 0.7342). |
| `mayo_ldct` | **0** | Step-2 retry post-cfg-patch eae661bc (2026-06-08) | **STOP** — iter-2 (k=2, c=16, train_n=50, ep=3) hr=0 SSIM 0.256; iter-3 (ep=6) hr=0 SSIM 0.249. Same low-capacity ceiling as v2. The v3 architecture (deeper UNet + per-step α) is the only ItNet that lifts above baseline on Mayo. |

## Cross-dataset pattern

ItNet family on Mayo: all three variants land hr=0 except v3 (post
cfg-merge patch). The deeper UNet + per-step α of v3 is necessary to
match Mayo's 2304-angle helical complexity.

On demo-DL the pattern reverses: v1 with the right hyperparams (k=2,
c=16) is competitive with v3. demo-DL's broader synthetic substrate
doesn't reward extra capacity in the same way.

## cfg-merge bug history (2026-06-08 patch — commit eae661bc)

ItNet v1 and v2 both had a silent-cfg-drop bug that surfaced when
the Mayo agentic loop dispatched them: `solver_itnet.py` and
`solver_itnet_v2.py` had no `env_var` read for the agentic JSON cfg,
so they used hardcoded `train_n=400, val_n=100, itnet_k=5` defaults
regardless of what the agentic loop sent. On Mayo's 2304-angle sino
those defaults OOM'd `filter_sino` at 5 GiB FFT pad on Q6000 24-GB.

Patch (commit eae661bc): added env-read pattern mirroring
`solver_itnet_v3.py`:
```python
def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_config_path = os.environ.get("ITNET_CONFIG_PATH")
    if env_config_path and Path(env_config_path).exists():
        with open(env_config_path) as f:
            env_cfg = json.load(f)
        cfg = {**CONFIG, **env_cfg}
```
+ unified SOLVER_MAP in `scripts/claude_agentic_one_iter.py` to use
the single `ITNET_CONFIG_PATH` env var name across v1/v2/v3 (was
inconsistent before).

After patch: Mayo v1 and v2 retries both landed hr=0 SSIM 0.25 — the
low-capacity ceiling is structural, not a configuration knob away.
v3 retry landed hr=0.1036 (rank 5 at the time, since lifted to 0.2181
via Step-3 TPE).

## Inventory-gap closure log (2026-06-09)

Job 762956 (breast-CT TPE) and 762957 (demo-DL TPE) closed the
inventory gap noted in the leaderboards. Both surprised the Mayo
hr=0 verdict by clearing baseline on the easier datasets:

- 762957 demo-DL: hr=0.4665 (rank 4)
- 762956 breast-CT: hr=0.1703 (rank 15)

Lesson: Mayo's data complexity sets a higher capacity floor than
breast-CT/demo-DL. A solver that fails on Mayo doesn't necessarily
fail on the synthetic datasets — and vice versa.

## SOLVERS entry (added 2026-06-09, commit 4b80357a)

```python
"itnet": {
    "solver": "pentathlon/demo_dl_reference/solver_itnet.py",
    "env_var": "ITNET_CONFIG_PATH",
    "slug_prefix": "demo-fair-itnet-v1-search",
    "agent_name": "itnet-v1-search",
    "space": {
        "pretrain_epochs":     (3, 8, "int"),
        "pretrain_lr":         (1e-4, 5e-3, "log"),
        "itnet_k":             (3, 8, "int"),
        "itnet_alpha":         (1e-3, 5e-2, "log"),
        "finetune_epochs":     (5, 15, "int"),
        "finetune_lr":         (1e-5, 1e-3, "log"),
        "unet_c":              ([8, 16, 24], "choice"),
    },
    "tpe_seed_trial": {
        # Seed from Mayo v1 iter-3 (hr=0):
        "pretrain_epochs": 6, "pretrain_lr": 5e-4,
        "itnet_k": 2, "itnet_alpha": 5e-3,
        "finetune_epochs": 10, "finetune_lr": 1e-4,
        "unet_c": 16,
    },
},
```
