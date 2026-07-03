"""score_mayo_alliters.py — the Mayo TEST-selection sweep dispatcher/collector.

Test-scores EVERY iteration of EVERY Mayo solver so each solver's leaderboard
iter can be chosen by TEST mean hr (not val). Reuses the proven per-patient
worker in score_mayo_testset.py (one patient per eval pass; train once per iter +
reuse the checkpoint for the other 4 patients). param-efficient is EXCLUDED here
— it is already fully test-scored in docs/runs/pe-iter-testeval/ — but --collect
folds it back in.

Two roles:

  --build   (laptop): for each Mayo run-id (minus param-efficient), for each iter
            with a cfg_full, write a per-iter cfg JSON + append a manifest line
            "<slug>\t<solver_key>\t<cfg_relpath>\t<solver_src_relpath>". Skips iters
            whose final.json already has all 5 patients (idempotent; --force to
            redo). Prints the array-submit command + per-solver line ranges (so a
            single solver can be piloted via --array=<lo>-<hi>%8).

  --collect (laptop): for each solver, pick the iter with max test_hr_mean (finite,
            all 5 patients; test_ssim_mean tiebreak) = its test-best iter. Prints a
            table (val-selected vs test-selected) + writes a selection summary JSON.

Output namespace per solver: docs/runs/<run_id>-itertest/iter-NNNN/final.json
(the worker writes final.json there; recons are skipped in sweep mode — see
AGENT4CT_SWEEP_NORECON in score_mayo_testset.py). Run --build, rsync the cfg dir +
manifest to the cluster, then submit cluster/slurm/mayo_alliters_array.sbatch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.score_mayo_testset import (  # noqa: E402
    DOCS_RUNS, TEST_PATIENTS, solver_map_key, load_allow_runids, _final_complete,
)

CFG_DIR = REPO / "agentic_cfgs" / "alliters"
MANIFEST = CFG_DIR / "manifest.tsv"
PE_SUBSTR = "param-efficient"          # excluded from the sweep (already test-scored)
PE_TESTEVAL = DOCS_RUNS / "pe-iter-testeval"


def _iter_dirs(runid: str):
    d = DOCS_RUNS / runid / "iterations"
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("iter-*") if (p / "observation.json").exists())


def _finite(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != math.inf


# --------------------------------------------------------------------------
# BUILD
# --------------------------------------------------------------------------
def build(force: bool) -> int:
    runids = [r for r in load_allow_runids() if PE_SUBSTR not in r]
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    lines, ranges, skipped = [], [], []
    for runid in runids:
        smk = solver_map_key(runid)
        if smk is None:
            skipped.append((runid, "no SOLVER_MAP entry"))
            continue
        lo = len(lines) + 1
        n_added = n_done = n_nocfg = 0
        for itdir in _iter_dirs(runid):
            name = itdir.name                         # iter-0007
            slug = f"{runid}-itertest/{name}"
            if not force and _final_complete(slug):
                n_done += 1
                continue
            try:
                obs = json.loads((itdir / "observation.json").read_text())
            except Exception:
                n_nocfg += 1
                continue
            cfg = obs.get("cfg_full")
            if not cfg:
                n_nocfg += 1
                continue
            cfg_dir = CFG_DIR / runid
            cfg_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = cfg_dir / f"{name}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            snap = itdir / "solver_src.py"            # only param-efficient has these
            src_rel = str(snap.relative_to(REPO)) if snap.exists() else ""
            cfg_rel = str(cfg_path.relative_to(REPO))
            lines.append(f"{slug}\t{smk}\t{cfg_rel}\t{src_rel}")
            n_added += 1
        hi = len(lines)
        if n_added:
            ranges.append((runid, lo, hi, n_added, n_done, n_nocfg))
        else:
            ranges.append((runid, None, None, 0, n_done, n_nocfg))

    MANIFEST.write_text("\n".join(lines) + ("\n" if lines else ""))
    N = len(lines)
    print(f"[alliters] wrote {N} work items -> {MANIFEST.relative_to(REPO)}")
    print(f"{'solver (run-id short)':45s} {'lines':>11s} {'todo':>5s} {'done':>5s} {'nocfg':>5s}")
    print("-" * 80)
    for runid, lo, hi, todo, done, nocfg in ranges:
        short = runid.replace("mayo-ldct-claude-agentic-", "").rsplit("-search-", 1)[0]
        rng = f"{lo}-{hi}" if lo else "—"
        print(f"{short:45s} {rng:>11s} {todo:>5d} {done:>5d} {nocfg:>5d}")
    for runid, why in skipped:
        print(f"[alliters] SKIP {runid}: {why}")
    if N:
        print(f"\nPILOT one solver:  sbatch --array=<lo>-<hi>%8 "
              f"--export=ALL,MANIFEST=/cluster/maier/Agent4CT/{MANIFEST.relative_to(REPO)} "
              f"cluster/slurm/mayo_alliters_array.sbatch")
        print(f"FULL sweep:        sbatch --array=1-{N}%8 "
              f"--export=ALL,MANIFEST=/cluster/maier/Agent4CT/{MANIFEST.relative_to(REPO)} "
              f"cluster/slurm/mayo_alliters_array.sbatch")
        print("\nBEFORE submitting: rsync the cfg dir + manifest + sbatch to the cluster:")
        print("  rsync -az agentic_cfgs/alliters/ lme-bastion:/cluster/maier/Agent4CT/agentic_cfgs/alliters/")
        print("  rsync -az cluster/slurm/mayo_alliters_array.sbatch scripts/score_mayo_testset.py "
              "lme-bastion:/cluster/maier/Agent4CT/<same paths>")
    return 0


# --------------------------------------------------------------------------
# COLLECT
# --------------------------------------------------------------------------
def _best_test_iter(iter_final_paths) -> dict | None:
    """Pick the iter with max test_hr_mean (all 5 patients, finite; test_ssim_mean
    tiebreak) from a list of (iter_int, final.json path)."""
    best = None
    for it, fp in iter_final_paths:
        try:
            o = json.loads(fp.read_text())
        except Exception:
            continue
        pats = o.get("patients") or {}
        if not all(pats.get(p) is not None for p in TEST_PATIENTS):
            continue
        hr, ss = o.get("test_hr_mean"), o.get("test_ssim_mean")
        if not _finite(hr):
            continue
        key = (hr, ss if _finite(ss) else -math.inf)
        if best is None or key > best["key"]:
            best = {"key": key, "iter": it, "test_hr_mean": hr,
                    "test_hr_std": o.get("test_hr_std"),
                    "test_ssim_mean": ss, "n_complete": True}
    return best


def collect() -> int:
    runids = load_allow_runids()
    rows = []
    for runid in runids:
        short = runid.replace("mayo-ldct-claude-agentic-", "").rsplit("-search-", 1)[0]
        if PE_SUBSTR in runid:
            base = PE_TESTEVAL                      # already-scored per-iter finals
            finals = [(int(p.parent.name.split("-")[-1]), p)
                      for p in base.glob("iter-*/final.json")]
        else:
            base = DOCS_RUNS / f"{runid}-itertest"
            finals = [(int(p.parent.name.split("-")[-1]), p)
                      for p in base.glob("iter-*/final.json")]
        n_iters = len(_iter_dirs(runid))
        best = _best_test_iter(finals)
        # current (val-selected) board number, for the before/after diff
        cur = DOCS_RUNS / runid / "final.json"
        cur_hr = None
        if cur.exists():
            try:
                cur_hr = json.loads(cur.read_text()).get("test_hr_mean")
            except Exception:
                pass
        rows.append({"solver": short, "run_id": runid, "n_iters": n_iters,
                     "n_scored": len(finals),
                     "val_selected_test_hr": cur_hr,
                     "test_best_iter": best["iter"] if best else None,
                     "test_best_hr_mean": best["test_hr_mean"] if best else None,
                     "test_best_hr_std": best["test_hr_std"] if best else None,
                     "test_best_ssim_mean": best["test_ssim_mean"] if best else None})
    rows.sort(key=lambda r: (r["test_best_hr_mean"] is None,
                             -(r["test_best_hr_mean"] or -1)))
    out = DOCS_RUNS / "mayo_testsweep_selection.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"{'solver':40s} {'scored':>7s} {'valSel_hr':>10s} "
          f"{'testBest_iter':>13s} {'testBest_hr':>12s}")
    print("-" * 90)
    for r in rows:
        def f(x):
            return f"{x:.4f}" if _finite(x) else "—"
        print(f"{r['solver']:40s} {r['n_scored']:>3d}/{r['n_iters']:<3d} "
              f"{f(r['val_selected_test_hr']):>10s} "
              f"{str(r['test_best_iter']):>13s} {f(r['test_best_hr_mean']):>12s}")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


def _gave_up_lines() -> set:
    """1-based manifest line numbers the driver has permanently GIVEN UP on
    (retry count >= MAX_RETRY in the driver state file). Excluded from `pending`
    so the sweep can terminate when the only remaining incomplete iters are
    genuinely unscorable — e.g. a GT-leak config the solver refuses on scored runs
    (ram_use_deepinv_tomo=True), or an iter that keeps failing after retries. These
    are transparently listed by --collect as excluded. Read defensively (the driver
    rewrites the file concurrently)."""
    state = REPO / "results" / "alliters_driver_state.json"
    MAX_RETRY = 3
    try:
        st = json.loads(state.read_text())
        return {int(k) for k, v in st.items()
                if isinstance(v, (int, float)) and v >= MAX_RETRY}
    except Exception:
        return set()


def pending() -> int:
    """Print a comma-separated list of 1-based manifest line numbers whose
    final.json is NOT yet complete (all 5 patients) AND that the driver has not
    permanently given up on. Used by the cluster-side driver to top up the queue
    under the QOS submit cap. Idempotent: a line drops off as soon as its
    final.json lands (or it is given up), so the driver survives restarts and the
    sweep terminates even if a few iters are unscorable."""
    if not MANIFEST.exists():
        print("")
        return 0
    lines = MANIFEST.read_text().splitlines()
    gave_up = _gave_up_lines()
    todo = []
    for i, l in enumerate(lines, 1):
        if i in gave_up:
            continue
        slug = l.split("\t")[0]
        if not _final_complete(slug):
            todo.append(str(i))
    print(",".join(todo))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="write cfgs + manifest")
    g.add_argument("--collect", action="store_true", help="pick test-best iter/solver")
    g.add_argument("--pending", action="store_true",
                   help="print CSV of incomplete manifest line numbers (for the driver)")
    ap.add_argument("--force", action="store_true",
                    help="(build) include iters even if final.json is already complete")
    args = ap.parse_args()
    if args.build:
        return build(args.force)
    if args.pending:
        return pending()
    return collect()


if __name__ == "__main__":
    raise SystemExit(main())
