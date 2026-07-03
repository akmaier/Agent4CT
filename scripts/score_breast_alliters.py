"""score_breast_alliters.py — the Breast-CT TEST-selection sweep dispatcher/collector.

Test-scores EVERY iteration of EVERY Breast-CT solver so each solver's leaderboard
iter can be chosen by TEST mean hr (over the 200 held-out test cases), NOT val —
exactly the Mayo rule (scripts/score_mayo_alliters.py), simplified because breast
cases are i.i.d. (no patients: mean±std is a per-case spread over the 200 test
cases). Reuses the per-case worker in score_breast_testset.py (one solver run per
iter on the test split, recon persisted, re-scored per case).

Two roles:

  --build   (laptop): for each breast run-id, for each iter with a cfg_full, write
            a per-iter cfg JSON + append a manifest line
            "<slug-itertest/iter>\t<solver_key>\t<cfg_relpath>\t<solver_src_relpath>".
            Skips iters whose final.json is already complete (idempotent; --force
            to redo). Prints the array-submit command (%8 throttle, ≤60 cap) +
            per-solver line ranges (so one solver can be piloted via --array=lo-hi%8).

  --collect / --best (laptop): for each solver, pick the iter with MAX test_hr_mean
            (finite, complete; test_ssim_mean tiebreak) = its test-best iter. Prints
            a table (val-selected vs test-selected) + writes a selection summary JSON.

Output namespace per iter: docs/runs/<run_id>-itertest/iter-NNNN/final.json — the
SAME namespace build_registry._itertest_base() reads for breast. Run --build, rsync
the cfg dir + manifest to the cluster, then submit
cluster/slurm/breast_alliters_array.sbatch. DO NOT run cluster commands from here;
this only EMITS them.
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

from scripts.score_breast_testset import (  # noqa: E402
    DOCS_RUNS, BREAST_TEST_N, solver_map_key, load_allow_runids, _final_complete,
)

CFG_DIR = REPO / "agentic_cfgs" / "breast_alliters"
MANIFEST = CFG_DIR / "manifest.tsv"
ARRAY_SBATCH = "cluster/slurm/breast_alliters_array.sbatch"
QOS_CAP = 60          # ≤60 concurrent array tasks (QOS submit cap), %8 running throttle
THROTTLE = 8


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
    runids = load_allow_runids()
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
            name = itdir.name                          # iter-0007
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
            snap = itdir / "solver_src.py"             # only code-evolving solvers have these
            src_rel = str(snap.relative_to(REPO)) if snap.exists() else ""
            cfg_rel = str(cfg_path.relative_to(REPO))
            lines.append(f"{slug}\t{smk}\t{cfg_rel}\t{src_rel}")
            n_added += 1
        hi = len(lines)
        ranges.append((runid, lo if n_added else None, hi if n_added else None,
                       n_added, n_done, n_nocfg))

    MANIFEST.write_text("\n".join(lines) + ("\n" if lines else ""))
    N = len(lines)
    print(f"[breast-alliters] wrote {N} work items -> {MANIFEST.relative_to(REPO)}")
    print(f"{'solver (run-id short)':45s} {'lines':>11s} {'todo':>5s} {'done':>5s} {'nocfg':>5s}")
    print("-" * 80)
    for runid, lo, hi, todo, done, nocfg in ranges:
        short = runid.replace("breast-ct-claude-agentic-", "").rsplit("-search-", 1)[0]
        rng = f"{lo}-{hi}" if lo else "—"
        print(f"{short:45s} {rng:>11s} {todo:>5d} {done:>5d} {nocfg:>5d}")
    for runid, why in skipped:
        print(f"[breast-alliters] SKIP {runid}: {why}")
    if N:
        cap = min(N, QOS_CAP)
        mani_cluster = f"/cluster/maier/Agent4CT/{MANIFEST.relative_to(REPO)}"
        print(f"\nPILOT one solver:  sbatch --array=<lo>-<hi>%{THROTTLE} "
              f"--time=00:30:00 --export=ALL,MANIFEST={mani_cluster} {ARRAY_SBATCH}")
        print(f"FULL sweep (≤{QOS_CAP} cap): sbatch --array=1-{N}%{THROTTLE} "
              f"--time=00:30:00 --export=ALL,MANIFEST={mani_cluster} {ARRAY_SBATCH}")
        if N > QOS_CAP:
            print(f"  (N={N} > QOS cap {QOS_CAP}: submit in chunks of ≤{QOS_CAP}, "
                  f"e.g. --array=1-{cap}%{THROTTLE} then --array={cap+1}-{N}%{THROTTLE})")
        print("\nBEFORE submitting: rsync the cfg dir + manifest + sbatch + scorer to the cluster:")
        print(f"  rsync -az agentic_cfgs/breast_alliters/ "
              f"lme-bastion:/cluster/maier/Agent4CT/agentic_cfgs/breast_alliters/")
        print(f"  rsync -az {ARRAY_SBATCH} scripts/score_breast_testset.py "
              f"scripts/score_breast_alliters.py ddssl_ldct/staged_dataset.py "
              f"lme-bastion:/cluster/maier/Agent4CT/<same paths>")
    return 0


# --------------------------------------------------------------------------
# COLLECT / BEST
# --------------------------------------------------------------------------
_TESTSET_KEYS = ("test_hr_mean", "test_hr_std", "test_ssim_mean", "test_ssim_std",
                 "test_psnr_mean", "test_psnr_std", "test_rmse_mean", "test_rmse_std")


def _best_test_iter(iter_final_paths) -> dict | None:
    """Pick the iter with MAX test_hr_mean (complete, finite; test_ssim_mean
    tiebreak) from a list of (iter_int, final.json path)."""
    best = None
    for it, fp in iter_final_paths:
        try:
            o = json.loads(fp.read_text())
        except Exception:
            continue
        if not o.get("complete"):
            continue
        hr, ss = o.get("test_hr_mean"), o.get("test_ssim_mean")
        if not _finite(hr):
            continue
        key = (hr, ss if _finite(ss) else -math.inf)
        if best is None or key > best["key"]:
            best = {"key": key, "iter": it,
                    **{k: o.get(k) for k in _TESTSET_KEYS}}
    return best


def collect() -> int:
    runids = load_allow_runids()
    rows = []
    for runid in runids:
        short = runid.replace("breast-ct-claude-agentic-", "").rsplit("-search-", 1)[0]
        base = DOCS_RUNS / f"{runid}-itertest"
        finals = [(int(p.parent.name.split("-")[-1]), p)
                  for p in base.glob("iter-*/final.json")]
        n_iters = len(_iter_dirs(runid))
        best = _best_test_iter(finals)
        rows.append({"solver": short, "run_id": runid, "n_iters": n_iters,
                     "n_scored": len(finals),
                     "test_best_iter": best["iter"] if best else None,
                     "test_best_hr_mean": best["test_hr_mean"] if best else None,
                     "test_best_hr_std": best["test_hr_std"] if best else None,
                     "test_best_ssim_mean": best["test_ssim_mean"] if best else None})
    rows.sort(key=lambda r: (r["test_best_hr_mean"] is None,
                             -(r["test_best_hr_mean"] or -1)))
    out = DOCS_RUNS / "breast_testsweep_selection.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"{'solver':40s} {'scored':>7s} {'testBest_iter':>13s} "
          f"{'testBest_hr':>14s}")
    print("-" * 80)
    for r in rows:
        def f(x):
            return f"{x:.4f}" if _finite(x) else "—"
        hr = (f"{f(r['test_best_hr_mean'])} ± {f(r['test_best_hr_std'])}"
              if r["test_best_hr_mean"] is not None else "—")
        print(f"{r['solver']:40s} {r['n_scored']:>3d}/{r['n_iters']:<3d} "
              f"{str(r['test_best_iter']):>13s} {hr:>14s}")
    print(f"\nwrote {out.relative_to(REPO)}   (n_test={BREAST_TEST_N} per-case mean±std)")
    print("Next: build_registry.py picks each solver's max-test_hr_mean iter for the board.")
    return 0


def pending() -> int:
    """CSV of 1-based manifest line numbers whose final.json is NOT yet complete
    (for a cluster-side top-up driver). Idempotent: a line drops off once its
    final.json lands."""
    if not MANIFEST.exists():
        print("")
        return 0
    todo = []
    for i, l in enumerate(MANIFEST.read_text().splitlines(), 1):
        slug = l.split("\t")[0]
        if not _final_complete(slug):
            todo.append(str(i))
    print(",".join(todo))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="write cfgs + manifest, emit sbatch")
    g.add_argument("--collect", action="store_true", help="pick test-best iter/solver")
    g.add_argument("--best", action="store_true", help="alias of --collect")
    g.add_argument("--pending", action="store_true",
                   help="print CSV of incomplete manifest line numbers (for a driver)")
    ap.add_argument("--force", action="store_true",
                    help="(build) include iters even if final.json is already complete")
    args = ap.parse_args()
    if args.build:
        return build(args.force)
    if args.pending:
        return pending()
    return collect()          # --collect or --best


if __name__ == "__main__":
    raise SystemExit(main())
