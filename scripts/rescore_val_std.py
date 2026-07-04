"""rescore_val_std.py — populate per-CASE std for Breast-CT VAL iters (no retrain).

The Breast-CT LIVE (val-ranked) leaderboard reports single-patient val metrics.
Each per-iter `observation.json` carries the val MEANS but its std fields are
`None` (`val_ssim_std`/`val_psnr_std`/`val_rmse_std`) and it has NO `headroom_std`
at all: `ddssl_ldct.metrics.evaluate_calibrated` DOES compute per-slice std for
ssim/psnr/rmse but the harness dropped those keys, and headroom is only a single
aggregate. So the board shows means only.

This script fixes that WITHOUT retraining anything. Every scored iter already
persisted its RAW (pre-calibration) pred/truth/baseline arrays over the ~200 val
cases to `iterations/iter-NNNN/recon_raw.npz` (via AGENT4CT_SAVE_RECON, see
metrics.py ~line 309). We reload those arrays and recompute PER-CASE calibrated,
FOV-masked, frozen-metric values with the SAME `evaluate_calibrated` invoked
one case at a time — so its returned scalars ARE the per-case values (with the
per-case baseline). Then we write mean AND std back into that iter's
observation.json:

    val_ssim_std, val_psnr_std, val_rmse_std  (were None)  -> per-case std
    headroom_std                              (was absent)  -> ADDED

The MEANS (`val_ssim`/`val_psnr`/`val_rmse`/`headroom`) are left as the harness
wrote them (a recomputed mean is logged as a cross-check but NOT overwritten, so
the number the search kept never shifts by a rounding epsilon).

Scoring window: breast/demo use display_min=0.0, display_max=0.05 (the frozen
DEMO_DL_DEFAULTS the live solvers pass to evaluate_calibrated). Only fg_threshold
(= display_min + 0.05*(display_max-display_min)) depends on it; display_max is no
longer used for clamping and SSIM/PSNR data_range = truth's own range — so this
reproduces the live metric. cfg_full's display_* override if ever present.

Idempotent: an iter that already has a finite numeric `headroom_std` is skipped
unless --force. Iters with no recon_raw.npz are left as-is (means only). Writes
observation.json ATOMICALLY (tmp + os.replace). Reuses the per-case scorer from
scripts/score_breast_testset.py so val + test std are computed identically.

Usage (run ON the cluster, from the repo root, venv active):
  python scripts/rescore_val_std.py --all                 # every breast run
  python scripts/rescore_val_std.py --run <run-id>        # one run
  python scripts/rescore_val_std.py --all --dry-run       # report, write nothing
  python scripts/rescore_val_std.py --all --force         # redo even if std present

Pure numpy/torch on saved arrays: cheap, no GPU training. SSIM runs on GPU if
CUDA is available, else CPU (a few seconds per iter either way).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DOCS_RUNS = REPO / "docs" / "runs"
ALLOWLIST = DOCS_RUNS / "CURRENT_RUNIDS.json"

# Frozen breast/demo scoring window (challenges/demo_dl/geometry.py DEFAULTS, the
# values the live solvers pass to evaluate_calibrated). Only fg_threshold uses it.
BREAST_DISPLAY_MIN = 0.0
BREAST_DISPLAY_MAX = 0.05

# Max tolerated |recomputed_val_ssim_mean - stored val_ssim| for a rescore to be
# trusted. Below this the recon_raw.npz faithfully reproduces the live metric (all
# 11 well-behaved 20260703 solvers land < 0.01); above it the persisted recon does
# NOT match what was scored (manhart-pwls-tv: 0.08-0.20) so we skip writing std.
SSIM_DRIFT_MAX = 0.02

# Reuse the exact per-case scorer built for the test-set path so val + test std
# are byte-identical in method (calibrate + FOV per case, headroom_case =
# max(0, 1 - rmse_case/baseline_rmse_case), mean±std over the cases).
from scripts.score_breast_testset import _score_recon_per_case  # noqa: E402


def _finite_num(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_breast_runids() -> list[str]:
    allow = json.loads(ALLOWLIST.read_text())
    return list(allow["datasets"].get("breast_ct", {}).get("run_ids", []))


def _display_window(obs: dict) -> tuple[float, float]:
    """cfg_full display_* if present + finite, else the frozen breast window."""
    cfg = obs.get("cfg_full") or {}
    dmin = cfg.get("display_min")
    dmax = cfg.get("display_max")
    if not _finite_num(dmin):
        dmin = BREAST_DISPLAY_MIN
    if not _finite_num(dmax):
        dmax = BREAST_DISPLAY_MAX
    return float(dmin), float(dmax)


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON atomically (tmp in the same dir + os.replace) so a reader — or
    a racing publish rsync — never sees a half-written observation.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".obs-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def rescore_iter(iter_dir: Path, *, force: bool, dry_run: bool) -> str:
    """Rescore ONE iter dir. Returns a status token:
      'populated' | 'skip-has-std' | 'skip-no-recon' | 'skip-no-obs' |
      'skip-bad-obs' | 'error'."""
    op = iter_dir / "observation.json"
    npz = iter_dir / "recon_raw.npz"
    if not op.exists():
        return "skip-no-obs"
    try:
        obs = json.loads(op.read_text())
    except Exception as e:
        print(f"    [{iter_dir.name}] bad observation.json: {e}", flush=True)
        return "skip-bad-obs"
    if not force and _finite_num(obs.get("headroom_std")):
        return "skip-has-std"
    if not npz.exists():
        return "skip-no-recon"

    dmin, dmax = _display_window(obs)
    try:
        aggr = _score_recon_per_case(npz, {"display_min": dmin, "display_max": dmax})
    except Exception as e:
        print(f"    [{iter_dir.name}] scoring FAILED: {e}", flush=True)
        return "error"

    # _score_recon_per_case returns test_*_mean / test_*_std keys (its shared name);
    # map them onto the val_* / headroom fields the val board reads.
    hr_m, hr_s = aggr["test_hr_mean"], aggr["test_hr_std"]
    ss_m, ss_s = aggr["test_ssim_mean"], aggr["test_ssim_std"]
    ps_m, ps_s = aggr["test_psnr_mean"], aggr["test_psnr_std"]
    rm_m, rm_s = aggr["test_rmse_mean"], aggr["test_rmse_std"]

    # Cross-check the recomputed means against the harness-written means (must
    # match to a rounding epsilon — proves the display window / metric are right).
    def _drift(new, old):
        return abs(new - old) if (_finite_num(new) and _finite_num(old)) else None
    d_hr = _drift(hr_m, obs.get("headroom"))
    d_ss = _drift(ss_m, obs.get("val_ssim"))
    tag = ""
    for nm, d, tol in (("hr", d_hr, 5e-3), ("ssim", d_ss, 5e-3)):
        if d is not None and d > tol:
            tag += f"  WARN {nm}_mean drift={d:.4g}"
    # headroom is only computable when the recon_raw.npz persisted a baseline
    # array. A few solvers (e.g. manhart-pwls-tv) saved pred/truth only, so
    # per-case headroom (and its std) is unrecoverable here — SSIM/PSNR/RMSE std
    # still are. Populate what we can and flag the row rather than crash.
    hr_ok = _finite_num(hr_s)
    if not hr_ok:
        tag += "  [no baseline in npz -> headroom_std unavailable]"

    def _f(x, spec):
        return format(x, spec) if _finite_num(x) else "—"
    line = (f"    [{iter_dir.name}] n={aggr['n_cases']}  "
            f"hr={_f(hr_m, '.4f')}±{_f(hr_s, '.4f')}  ssim={_f(ss_m, '.4f')}±{_f(ss_s, '.4f')}  "
            f"psnr={_f(ps_m, '.2f')}±{_f(ps_s, '.2f')}  "
            f"rmse={_f(rm_m, '.3e')}±{_f(rm_s, '.3e')}{tag}")

    # Faithfulness guard: the std we write must correspond to the SAME recon the
    # harness scored. If the recomputed SSIM mean drifts far from the stored mean
    # (> SSIM_DRIFT_MAX), the recon_raw.npz does NOT match what was scored live
    # (e.g. manhart-pwls-tv persisted a raw pred that differs from its scored
    # output). Writing a std from a mismatched recon would sit next to an
    # inconsistent mean, so we SKIP the write and leave the iter means-only.
    if d_ss is not None and d_ss > SSIM_DRIFT_MAX:
        print(line + f"  -> SKIP-WRITE (ssim mean drift {d_ss:.3f} > {SSIM_DRIFT_MAX}: "
              f"npz recon != scored recon)", flush=True)
        # If a prior (pre-guard) run wrote std from this mismatched recon, revert
        # those fields to None so the board never shows a std inconsistent with the
        # displayed mean. No-op when nothing was written.
        if not dry_run and (obs.get("_val_std_rescored") is not None
                            or _finite_num(obs.get("headroom_std"))
                            or _finite_num(obs.get("val_ssim_std"))):
            for _k in ("val_ssim_std", "val_psnr_std", "val_rmse_std", "headroom_std"):
                obs[_k] = None
            obs.pop("_val_std_rescored", None)
            obs["_val_std_rescore_skipped"] = {
                "reason": "ssim_mean_drift", "drift": d_ss,
                "recomputed_val_ssim_mean": ss_m, "stored_val_ssim": obs.get("val_ssim"),
            }
            _atomic_write_json(op, obs)
            print(f"      reverted stale std fields on {iter_dir.name}", flush=True)
        return "skip-drift"

    if dry_run:
        print(line + "  (DRY)", flush=True)
        return "populated" if hr_ok else "populated-no-hr"

    # Write std back; ADD headroom_std. Leave the MEANS the harness kept intact —
    # only fill the std fields (and headroom_std) so the ranked number never moves.
    obs["val_ssim_std"] = ss_s
    obs["val_psnr_std"] = ps_s
    obs["val_rmse_std"] = rm_s
    obs["headroom_std"] = hr_s   # None when no baseline was persisted (rare)
    obs["_val_std_rescored"] = {
        "n_cases": aggr["n_cases"],
        "display_min": dmin, "display_max": dmax,
        "headroom_std_available": hr_ok,
        "recomputed_headroom_mean": hr_m, "recomputed_val_ssim_mean": ss_m,
        "recomputed_val_psnr_mean": ps_m, "recomputed_val_rmse_mean": rm_m,
    }
    _atomic_write_json(op, obs)
    print(line, flush=True)
    return "populated" if hr_ok else "populated-no-hr"


def rescore_run(slug: str, *, force: bool, dry_run: bool) -> dict:
    iters_dir = DOCS_RUNS / slug / "iterations"
    counts: dict[str, int] = {}
    if not iters_dir.is_dir():
        print(f"[rescore] {slug}: no iterations dir", flush=True)
        return counts
    print(f"[rescore] {slug}", flush=True)
    for d in sorted(iters_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("iter-"):
            continue
        st = rescore_iter(d, force=force, dry_run=dry_run)
        counts[st] = counts.get(st, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="rescore every breast run in the allowlist")
    g.add_argument("--run", help="rescore ONE breast run-id")
    ap.add_argument("--force", action="store_true",
                    help="rescore even iters that already have a numeric headroom_std")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    args = ap.parse_args()

    if args.all:
        runids = load_breast_runids()
    else:
        runids = [args.run]
    if not runids:
        print("[rescore] no breast run-ids found", flush=True)
        return 1

    total: dict[str, int] = {}
    per_run: dict[str, int] = {}
    for slug in runids:
        c = rescore_run(slug, force=args.force, dry_run=args.dry_run)
        per_run[slug] = c.get("populated", 0) + c.get("populated-no-hr", 0)
        for k, v in c.items():
            total[k] = total.get(k, 0) + v

    print("\n[rescore] ===== SUMMARY =====", flush=True)
    for slug in runids:
        print(f"  {slug:70s} populated={per_run.get(slug, 0)}", flush=True)
    print(f"[rescore] totals: {json.dumps(total, sort_keys=True)}", flush=True)
    pop = total.get("populated", 0)
    norecon = total.get("skip-no-recon", 0)
    hasstd = total.get("skip-has-std", 0)
    print(f"[rescore] {pop} iters std-populated, {hasstd} already had std, "
          f"{norecon} skipped (no recon_raw.npz){' [DRY-RUN]' if args.dry_run else ''}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
