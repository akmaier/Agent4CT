"""score_mayo_testset.py — Phase 1B: per-patient TEST-set std for ALL Mayo solvers.

The live Mayo metric scores val = L277 (1 patient). The user wants a std "over
patients" = the 5 held-out Wagner TEST patients (L014, L056, L058, L075, L123).
No solver scores test today. This harness gives every existing solver a
transparent per-patient test score with ZERO solver changes:

  AGENT4CT_EVAL_PATIENT=<Lxxx>  (loader override in ddssl_ldct/staged_dataset.py)
    -> load_val_split(..., split="val") returns THAT single test patient's whole
       volume (truth + lowdose sino + per-slice ps) in the SAME tuple shape val
       returns, so the solver's existing main() scores it as if it were val.

Two roles in one file (so all logic lives in Python, not a big bash loop):

  DISPATCH (default, run from the laptop): for each Mayo run-id, find its best
    iter (max headroom) + that iter's cfg_full; write a tmp cfg JSON; submit one
    sbatch job per solver that re-invokes THIS script in --worker mode. Prints
    the job ids. Heterogeneous solvers (supervised + per-image NAF / R2G /
    diffusion / RAM) need no special-casing because every job just runs the
    solver's own entry point (SOLVER_MAP path) once per patient.

  WORKER (--worker, runs inside the sbatch on a GPU node): loop the 5 test
    patients, run the solver on each (set AGENT4CT_DATASET=mayo_ldct_2d +
    AGENT4CT_EVAL_PATIENT=<Lxxx> + the solver's *_CONFIG_PATH=<cfg>, then
    `python <solver_path> <tmp_out>`), read result.json, collect the
    whole-volume-mean headroom/ssim/psnr/rmse. Aggregate -> mean ± std over the
    5 patients -> docs/runs/<slug>/final.json.

Usage:
  python scripts/score_mayo_testset.py --all                 # dispatch every solver
  python scripts/score_mayo_testset.py --solver itnet        # dispatch one (SOLVER_MAP key OR run-id)
  python scripts/score_mayo_testset.py --all --dry-run       # print what it WOULD dispatch
  python scripts/score_mayo_testset.py --all --force         # re-dispatch even if final.json is complete
  python scripts/score_mayo_testset.py --worker --slug <run-id> --solver <key> --cfg <json>   # cluster-side

Idempotent: a solver whose docs/runs/<slug>/final.json already has all 5 patients
is skipped (unless --force).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DOCS_RUNS = REPO / "docs" / "runs"
ALLOWLIST = DOCS_RUNS / "CURRENT_RUNIDS.json"
RUNS_BASE = Path(os.environ.get("AGENT4CT_RUNS", "/cluster/maier/Agent4CT/runs"))
CFG_DIR = Path(os.environ.get("AGENT4CT_CFGS", str(REPO / "agentic_cfgs")))
SBATCH = REPO / "cluster" / "slurm" / "mayo_testset_score.sbatch"

# The 5 held-out Wagner TEST patients, in the order the test h5 is packed.
TEST_PATIENTS = ["L014", "L056", "L058", "L075", "L123"]
FINAL_SCHEMA = "mayo_testset_final_v1"

from scripts.claude_agentic_one_iter import SOLVER_MAP  # noqa: E402
import scripts.registry_lib as R  # noqa: E402

# --------------------------------------------------------------------------
# run-id solver_key (dashed, from registry_lib.solver_key) -> SOLVER_MAP key
# (underscore). Most are a plain dash->underscore swap; a few run-ids use a
# short family name (ram) or strip the ckpt-version suffix (diff-recon-* -> the
# v4 ckpt variant the search-20260619-01 run-ids actually ran).
# --------------------------------------------------------------------------
_KEY_OVERRIDES = {
    "ram": "ram_zeroshot",
    "diff-recon-dcstep-constrained": "diffusion_recon_dcstep_constrained_mayo_v4",
    "diff-recon-dcstep-unconstrained": "diffusion_recon_dcstep_unconstrained_mayo_v4",
}


def solver_map_key(run_or_key: str) -> str | None:
    """Resolve a run-id, dashed solver_key, or underscore SOLVER_MAP key to the
    SOLVER_MAP key. Returns None if it cannot be resolved to a known solver."""
    if run_or_key in SOLVER_MAP:
        return run_or_key
    # A full run-id slug -> dashed solver_key via registry_lib.
    dashed = R.solver_key(run_or_key) if run_or_key.startswith("mayo-ldct") else run_or_key
    if dashed in _KEY_OVERRIDES:
        return _KEY_OVERRIDES[dashed]
    under = dashed.replace("-", "_")
    if under in SOLVER_MAP:
        return under
    if dashed in SOLVER_MAP:
        return dashed
    return None


def load_allow_runids() -> list[str]:
    allow = json.loads(ALLOWLIST.read_text())
    return list(allow["datasets"]["mayo_ldct"]["run_ids"])


def _best_iter(slug: str) -> tuple[int | None, dict | None]:
    """The best iter for a run = max headroom over its observation.json records
    (the immutable source of truth; results.tsv is a fallback). Returns
    (iter_int, cfg_full_dict) or (None, None) if the run has no usable iter."""
    iters_dir = DOCS_RUNS / slug / "iterations"
    best_it, best_hr, best_cfg = None, -1e30, None
    if iters_dir.is_dir():
        for d in sorted(iters_dir.iterdir()):
            op = d / "observation.json"
            if not op.exists():
                continue
            try:
                obs = json.loads(op.read_text())
            except Exception:
                continue
            hr = obs.get("headroom")
            if not isinstance(hr, (int, float)):
                continue
            if hr > best_hr:
                cfg = obs.get("cfg_full")
                try:
                    it = int(obs.get("iter", int(d.name.split("-")[-1])))
                except Exception:
                    continue
                best_it, best_hr, best_cfg = it, float(hr), cfg
    if best_it is not None and best_cfg is not None:
        return best_it, best_cfg
    # Fallback: results.tsv max-headroom row (no cfg_full there, so only used to
    # confirm an iter exists; cfg_full is required, so this stays None-cfg).
    return best_it, best_cfg


def _final_complete(slug: str) -> bool:
    fp = DOCS_RUNS / slug / "final.json"
    if not fp.exists():
        return False
    try:
        obj = json.loads(fp.read_text())
    except Exception:
        return False
    pats = obj.get("patients") or {}
    return all(p in pats and pats[p] is not None for p in TEST_PATIENTS)


# ==========================================================================
# WORKER role (cluster-side): loop 5 patients, run the solver, write final.json
# ==========================================================================
def run_worker(slug: str, solver_key: str, cfg_json: Path,
               solver_src: str | None = None) -> int:
    if solver_key not in SOLVER_MAP:
        print(f"[testset] unknown solver {solver_key!r}; choices: {list(SOLVER_MAP)}",
              flush=True)
        return 2
    default_path, env_var = SOLVER_MAP[solver_key]
    # Code-evolving solvers (param_efficient) snapshot their source per iter; the
    # SOLVER_MAP path is the LATEST iter, not the best one. `solver_src` (the
    # best-iter's solver_src.py) overrides it so we re-score the EXACT architecture
    # the best-iter cfg belongs to. Non-evolving solvers pass solver_src=None.
    solver_path = solver_src if solver_src else default_path
    base_out = RUNS_BASE / f"{slug}-testset"
    base_out.mkdir(parents=True, exist_ok=True)
    # ONE checkpoint path shared across all 5 patient subprocesses this iter:
    # patient-1 (L014) trains + saves it, L056..L123 load it and skip training
    # (training is patient-independent). NOT deleted between patients.
    model_ckpt = base_out / "model_ckpt.pt"

    def _run_eval(label: str, eval_patient: str | None) -> dict | None:
        """One held-out eval (a test patient, or val L277 when eval_patient is
        None). Persists the raw recon (AGENT4CT_SAVE_RECON) for auditability +
        future re-scoring; reads back the FIXED-metric scalars from result.json."""
        out_dir = base_out / label
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["AGENT4CT_DATASET"] = "mayo_ldct_2d"
        if eval_patient is not None:
            env["AGENT4CT_EVAL_PATIENT"] = eval_patient
        else:
            env.pop("AGENT4CT_EVAL_PATIENT", None)        # normal val (L277)
        # Persist the raw recon for auditability / future re-scoring — UNLESS
        # this is the big all-iters sweep (AGENT4CT_SWEEP_NORECON=1): ~500 iters
        # x 5 patients x ~300 MB ≈ 700 GB won't fit (3.3 TB free, 90% full). In
        # sweep mode we keep only the (tiny) per-iter checkpoint; recons for the
        # SELECTED board iter are regenerated from that ckpt after collection.
        if os.environ.get("AGENT4CT_SWEEP_NORECON"):
            env.pop("AGENT4CT_SAVE_RECON", None)
        else:
            env["AGENT4CT_SAVE_RECON"] = str(out_dir)     # persist raw recon
        # Shared per-iter checkpoint (train-once, reuse): the first patient's
        # process trains + saves it; the rest load it and skip training.
        env["AGENT4CT_MODEL_CKPT"] = str(model_ckpt)
        env[env_var] = str(cfg_json)
        print(f"[testset] {slug} {label}: python {solver_path} {out_dir}", flush=True)
        t0 = time.time()
        res = subprocess.run([sys.executable, str(REPO / solver_path), str(out_dir)],
                             env=env)
        elapsed = time.time() - t0
        rj = out_dir / "result.json"
        if res.returncode != 0 or not rj.exists():
            print(f"[testset] {label}: FAILED (rc={res.returncode}, "
                  f"result.json={'present' if rj.exists() else 'missing'})", flush=True)
            return None
        try:
            r = json.loads(rj.read_text())
        except Exception as e:
            print(f"[testset] {label}: bad result.json: {e}", flush=True)
            return None
        rec = {
            "headroom": r.get("headroom"), "ssim": r.get("val_ssim"),
            "psnr": r.get("val_psnr"), "rmse": r.get("val_rmse"),
            "ssim_std": r.get("val_ssim_std"), "psnr_std": r.get("val_psnr_std"),
            "rmse_std": r.get("val_rmse_std"),
            "n_slices": r.get("val_n"), "elapsed_s": round(elapsed, 1),
            "recon_saved": (out_dir / "recon_raw.npz").exists(),
        }
        print(f"[testset] {label}: hr={rec['headroom']} ssim={rec['ssim']} "
              f"({elapsed:.0f}s, recon_saved={rec['recon_saved']})", flush=True)
        return rec

    per_patient: dict[str, dict | None] = {}
    for patient in TEST_PATIENTS:
        per_patient[patient] = _run_eval(patient, patient)
    # NO validation pass. L277 val is a training-loop signal, NEVER a reported
    # result (see README "Evaluation paradigm"). Mayo results are TEST-only:
    # mean ± std over the 5 held-out patients. (Removed 2026-06-30.)

    final = _aggregate(slug, solver_key, per_patient)
    final["solver_src"] = solver_path
    final["recons_saved"] = all((per_patient.get(p) or {}).get("recon_saved")
                                for p in TEST_PATIENTS)
    out_path = DOCS_RUNS / slug / "final.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2))
    n_ok = sum(1 for v in per_patient.values() if v is not None)
    print(f"[testset] wrote {out_path}  ({n_ok}/{len(TEST_PATIENTS)} test patients ok)",
          flush=True)
    return 0 if n_ok == len(TEST_PATIENTS) else 1


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    """Population-ish mean/std over the finite patient values (std over patients
    = the requested 'std over patients'). Sample std (ddof=1) so n=5 reports the
    unbiased spread; falls back gracefully for n<2."""
    finite = [float(v) for v in vals
              if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")]
    if not finite:
        return None, None
    m = sum(finite) / len(finite)
    if len(finite) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in finite) / (len(finite) - 1)
    return m, var ** 0.5


def _aggregate(slug: str, solver_key: str, per_patient: dict) -> dict:
    def col(metric):
        return [(per_patient[p] or {}).get(metric) for p in TEST_PATIENTS]
    hr_m, hr_s = _mean_std(col("headroom"))
    ss_m, ss_s = _mean_std(col("ssim"))
    ps_m, ps_s = _mean_std(col("psnr"))
    rm_m, rm_s = _mean_std(col("rmse"))
    return {
        "schema": FINAL_SCHEMA,
        "run_id": slug,
        "solver_key": solver_key,
        "test_n_patients": len(TEST_PATIENTS),
        "patients": per_patient,
        "test_hr_mean": hr_m, "test_hr_std": hr_s,
        "test_ssim_mean": ss_m, "test_ssim_std": ss_s,
        "test_psnr_mean": ps_m, "test_psnr_std": ps_s,
        "test_rmse_mean": rm_m, "test_rmse_std": rm_s,
    }


# ==========================================================================
# DISPATCH role (laptop): write cfg JSON, submit one sbatch per solver
# ==========================================================================
def dispatch_one(slug: str, *, dry_run: bool, force: bool) -> str | None:
    smk = solver_map_key(slug)
    if smk is None:
        print(f"[testset] SKIP {slug}: no SOLVER_MAP entry for "
              f"solver_key={R.solver_key(slug)!r}")
        return None
    if not force and _final_complete(slug):
        print(f"[testset] SKIP {slug}: final.json already has all "
              f"{len(TEST_PATIENTS)} patients (use --force to redo)")
        return None
    best_it, cfg = _best_iter(slug)
    if best_it is None or cfg is None:
        print(f"[testset] SKIP {slug}: no best iter with cfg_full found")
        return None

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_DIR / f"{slug}_testset_best_iter_{best_it:04d}.json"
    # Code-evolving solvers snapshot their source per iter; re-score the EXACT
    # best-iter architecture, not the latest SOLVER_MAP file. Empty for the
    # (majority) non-evolving solvers -> worker uses the SOLVER_MAP path.
    snap = DOCS_RUNS / slug / "iterations" / f"iter-{best_it:04d}" / "solver_src.py"
    solver_src_rel = str(snap.relative_to(REPO)) if snap.exists() else ""
    if dry_run:
        print(f"[testset] DRY-RUN dispatch {slug}: solver={smk} best_iter={best_it} "
              f"cfg->{cfg_path.name} solver_src={solver_src_rel or '(SOLVER_MAP)'} "
              f"sbatch={SBATCH.name}")
        return None
    cfg_path.write_text(json.dumps(cfg, indent=2))
    cmd = [
        "sbatch",
        f"--job-name=mayo-test-{smk}",
        f"--export=ALL,SLUG={slug},SOLVER={smk},CFG_JSON={cfg_path},SOLVER_SRC={solver_src_rel}",
        str(SBATCH),
    ]
    print(f"[testset] dispatch {slug}: solver={smk} best_iter={best_it}")
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        print(f"           {out}")
        # "Submitted batch job 12345" -> 12345
        jid = out.split()[-1] if out else ""
        return jid
    except FileNotFoundError:
        print("[testset] ERROR: sbatch not found (run this from the cluster login node)")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[testset] ERROR: sbatch failed: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="dispatch every Mayo solver in the allowlist")
    g.add_argument("--solver", help="dispatch ONE solver (SOLVER_MAP key OR run-id)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be dispatched, submit nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-dispatch even if final.json already has all 5 patients")
    # worker role (cluster-side, invoked by the sbatch)
    ap.add_argument("--worker", action="store_true",
                    help="cluster-side: loop the 5 patients + write final.json")
    ap.add_argument("--slug", help="(worker) the run-id slug")
    ap.add_argument("--cfg", help="(worker) path to the best-iter cfg JSON")
    ap.add_argument("--solver-src", default=None,
                    help="(worker) explicit solver .py to run (code-evolving "
                         "iter snapshot); empty/unset -> SOLVER_MAP path")
    args = ap.parse_args()

    if args.worker:
        if not (args.slug and args.solver and args.cfg):
            ap.error("--worker needs --slug, --solver and --cfg")
        smk = solver_map_key(args.solver)
        if smk is None:
            ap.error(f"--worker: unknown solver {args.solver!r}")
        return run_worker(args.slug, smk, Path(args.cfg),
                          solver_src=(args.solver_src or None))

    runids = load_allow_runids()
    if args.solver:
        # accept a SOLVER_MAP key, a dashed solver_key, or a full run-id
        smk = solver_map_key(args.solver)
        targets = [s for s in runids if solver_map_key(s) == smk] if smk else []
        if not targets:
            # maybe the user passed a full run-id not (yet) in the allowlist
            if args.solver in runids:
                targets = [args.solver]
            else:
                ap.error(f"--solver {args.solver!r} matched no Mayo run-id "
                         f"(resolved key={smk})")
    elif args.all:
        targets = runids
    else:
        ap.error("pass --all or --solver <key> (or --worker for cluster-side)")

    jids = []
    for slug in targets:
        jid = dispatch_one(slug, dry_run=args.dry_run, force=args.force)
        if jid:
            jids.append(jid)
    if jids:
        print(f"[testset] submitted {len(jids)} job(s): {' '.join(jids)}")
    elif not args.dry_run:
        print("[testset] no jobs submitted (all skipped or dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
