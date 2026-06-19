#!/usr/bin/env python3
"""Complete the breast-CT LPD TPE study to 20 trials.

The original search (job 762081, slug
`breast-ct-calibrated-tpe-lpd-search-20260524-01`) hit the 14 h wall at 17/20
trials. This resumes the EXISTING Optuna study and writes iter-0018..0020 into
the EXISTING run dir, so the search becomes a clean 20/20 for the paper.

It reuses `learned_solver_search_agent` internals (the LPD search space,
`run_solver`, `_optuna_suggest`, `record_iteration`) so the 3 new trials are
IDENTICAL in search space + scoring to trials 1-17:
  - same Optuna study DB  -> TPE resumes from the 17 completed trials
  - same SOLVERS["lpd"] space (lpd_iters [8,10], epochs 15-25, ...)
  - AGENT4CT_DATASET=breast_ct -> the same calibrated-SSIM-headroom scoring
  - same agent string in results.tsv / observation.json as rows 1-17

Champion is expected to stay iter-11 (hr 0.9062); trials 12-17 all scored below
it, so the TPE had already plateaued. This run just fills the 3 missing slots.
"""
import os
import sys
import json
from pathlib import Path

REPO = Path("/cluster/maier/Agent4CT")
SLUG = "breast-ct-calibrated-tpe-lpd-search-20260524-01"
# Verbatim from results.tsv rows 1-17 so the new rows match exactly.
AGENT_NAME = "lpd-search-tpe-calibrated-breast-ct"
TARGETS = [18, 19, 20]
MAX_ATTEMPTS = 9  # allow a couple retries if a trial fails/times out

# Calibrated breast scoring is implicit in the dataset env (the --calibrated CLI
# flag only renamed the slug). Set it BEFORE importing/using the solver.
os.environ["AGENT4CT_DATASET"] = "breast_ct"
os.environ.setdefault("SEARCH_AGENT_SUBPROC_TIMEOUT_S", "5400")

sys.path.insert(0, str(REPO / "scripts"))
import optuna  # noqa: E402
import learned_solver_search_agent as A  # noqa: E402

spec = A.SOLVERS["lpd"]
run_dir = REPO / "docs" / "runs" / SLUG
out_base = REPO / "runs"
storage = f"sqlite:///{REPO}/optuna/{SLUG}.db"

assert run_dir.exists(), f"run dir missing: {run_dir}"
print(f"[resume] lpd space: {json.dumps({k: v[0] for k, v in spec['space'].items()})}",
      flush=True)

study = optuna.create_study(
    study_name=SLUG, storage=storage, direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=20260516, n_startup_trials=5),
    load_if_exists=True,
)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"[resume] study loaded: {len(study.trials)} trials, {len(done)} complete", flush=True)
best = max((t.value for t in done if t.value is not None), default=0.0)
print(f"[resume] best complete hr so far = {best:.4f}", flush=True)

attempts = 0
recorded = 0
for target in TARGETS:
    ok = False
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        trial = study.ask()
        params = A._optuna_suggest(trial, spec["space"])
        print(f"\n[resume] iter {target} (attempt {attempts}): {json.dumps(params)}", flush=True)
        out_dir = out_base / f"{SLUG}-iter-{target:04d}"
        result = A.run_solver(spec["solver"], spec["env_var"], params, out_dir)
        hr = (result or {}).get("headroom", 0.0)
        study.tell(trial, hr)
        if result is not None:
            A.record_iteration(run_dir, target, params, result, AGENT_NAME, out_dir)
            recorded += 1
            ok = True
            print(f"[resume] iter {target} RECORDED  hr={hr:.4f}  "
                  f"ssim={result.get('val_score', 0):.4f}", flush=True)
            break
        print(f"[resume] iter {target} attempt failed (rc/timeout) — retrying", flush=True)
    if not ok:
        print(f"[resume] GAVE UP on iter {target} after {MAX_ATTEMPTS} attempts", flush=True)
        break

print(f"\n[resume] done — recorded {recorded}/3 new iters "
      f"(run now {17 + recorded}/20).", flush=True)
