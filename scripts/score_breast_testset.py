"""score_breast_testset.py — per-CASE TEST-set mean±std for Breast-CT solvers.

The Breast-CT redo (paper_CTautoresearch.md §5.0) gave breast a held-out TEST
split (train 3600 / val 200 / test 200, seed 20260703, staged as test_*.h5).
Breast cases are i.i.d. synthetic phantom realizations — NO patient grouping —
so Mayo's "mean±std over 5 patients" becomes "mean±std over the 200 test cases"
(a clean per-case spread). This is the direct analogue of scripts/
score_mayo_testset.py, simplified because there are no per-patient ranges.

Mechanism (ZERO solver changes, mirrors the Mayo AGENT4CT_EVAL_PATIENT trick):

  AGENT4CT_EVAL_SPLIT=test  (loader override in ddssl_ldct/staged_dataset.py)
    -> load_val_split(kind="breast_ct", split="val", ...) transparently loads the
       TEST split instead, so the solver's existing main() reconstructs the 200
       test cases as if they were val.

  AGENT4CT_SAVE_RECON=<dir>  (ddssl_ldct/metrics.py)
    -> evaluate_calibrated dumps the RAW (pre-calibration) pred/truth/baseline
       arrays to recon_raw.npz. We reload these and compute PER-CASE calibrated,
       FOV-masked, frozen-metric values ourselves (calibrate + FOV per case,
       headroom_case = max(0, 1 - rmse_case / baseline_rmse_case)), then report
       mean AND std over the 200 cases for all four metrics — INCLUDING
       headroom_std, which evaluate_calibrated's batch aggregate does not give.

Two roles in one file (same shape as score_mayo_testset.py):

  DISPATCH (default, laptop): for a breast run-id, find its best-by-val-hr iter +
    cfg_full, write a tmp cfg JSON (val_n forced to the test-set size so the whole
    200 loads), submit ONE sbatch that re-invokes this script in --worker mode.

  WORKER (--worker, cluster GPU node): run the solver ONCE with the loader
    redirected to the test split + recon persisted, reload recon_raw.npz, compute
    the per-case metrics -> mean±std -> docs/runs/<slug>-itertest/iter-NNNN/final.json
    (or docs/runs/<slug>/final.json for the single-run mode). Retrain-once-per-
    config (like Mayo), honoring the 20-min budget knobs already in the cfg.

Usage:
  python scripts/score_breast_testset.py --all                  # dispatch every breast solver
  python scripts/score_breast_testset.py --solver itnet-v2      # dispatch one (run-id OR key)
  python scripts/score_breast_testset.py --all --dry-run        # print, submit nothing
  python scripts/score_breast_testset.py --worker --slug <slug> --solver <key> \
         --cfg <json> [--split test] [--solver-src <py>]        # cluster-side

Idempotent: a slug whose final.json is already complete is skipped (unless --force).
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
SBATCH = REPO / "cluster" / "slurm" / "breast_testset_score.sbatch"

# Breast test-set size (paper §5.0: train 3600 / val 200 / test 200, seed 20260703).
# The reported spread is a per-CASE std over these; there is no patient grouping.
BREAST_TEST_N = 200
FINAL_SCHEMA = "breast_testset_final_v1"

from scripts.claude_agentic_one_iter import SOLVER_MAP  # noqa: E402
import scripts.registry_lib as R  # noqa: E402

# run-id dashed solver_key -> SOLVER_MAP key (underscore). Same override table as
# Mayo for the few families whose run-id name differs from the SOLVER_MAP key.
_KEY_OVERRIDES = {
    "ram": "ram_zeroshot",
}


def solver_map_key(run_or_key: str) -> str | None:
    """Resolve a run-id, dashed solver_key, or underscore SOLVER_MAP key to the
    SOLVER_MAP key. Returns None if unresolvable."""
    if run_or_key in SOLVER_MAP:
        return run_or_key
    dashed = R.solver_key(run_or_key) if run_or_key.startswith("breast-ct") else run_or_key
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
    return list(allow["datasets"].get("breast_ct", {}).get("run_ids", []))


def _best_iter(slug: str) -> tuple[int | None, dict | None]:
    """Best-by-val-headroom iter for a run over its observation.json records (the
    immutable source of truth). Returns (iter_int, cfg_full_dict) or (None, None).
    The SEARCH is val-selected (breast val=200); the leaderboard iter is then
    re-picked by TEST hr by score_breast_alliters.py --collect / build_registry.
    This single-run dispatch scores just the val-best iter as a fast smoke."""
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
    return best_it, best_cfg


def _final_complete(slug: str) -> bool:
    """A breast final.json is complete when it has the schema marker + a finite
    test_hr_mean over the full test set (n_test == BREAST_TEST_N cases scored)."""
    fp = DOCS_RUNS / slug / "final.json"
    if not fp.exists():
        return False
    try:
        obj = json.loads(fp.read_text())
    except Exception:
        return False
    if not obj.get("complete"):
        return False
    hr = obj.get("test_hr_mean")
    return isinstance(hr, (int, float)) and hr == hr


# ==========================================================================
# Per-case metric re-scoring (frozen metric, calibrated + FOV-masked, per case)
# ==========================================================================
def _mean_std(vals) -> tuple[float | None, float | None]:
    """Sample mean/std (ddof=1) over finite values; std=0.0 for n<2, (None,None)
    if empty. 'std over the 200 test cases' = the requested per-case spread."""
    finite = [float(v) for v in vals
              if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")]
    if not finite:
        return None, None
    m = sum(finite) / len(finite)
    if len(finite) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in finite) / (len(finite) - 1)
    return m, var ** 0.5


def _score_recon_per_case(npz_path: Path, cfg: dict) -> dict:
    """Reload the persisted RAW recon and compute PER-CASE calibrated, FOV-masked,
    frozen-metric values (one (pred, truth, baseline) triple per test case), then
    aggregate mean±std over all cases for hr/ssim/psnr/rmse.

    Uses the SAME evaluate_calibrated as the live metric — invoked per single case
    so its returned scalars ARE the per-case values (with the per-case baseline).
    The batch aggregate cannot give headroom_std; scoring case-by-case can."""
    import numpy as np
    import torch
    from ddssl_ldct.metrics import evaluate_calibrated, ssim as _ssim_fn, psnr as _psnr_fn

    d = np.load(str(npz_path))
    # Most solvers save the raw recon under "pred" (metrics.py AGENT4CT_SAVE_RECON),
    # but a few solver-side save blocks (dual-domain-bilateral-n2i, wu-2015-trainable)
    # use "recon". Accept either so those solvers score instead of KeyError'ing.
    _pred_key = "pred" if "pred" in d.files else ("recon" if "recon" in d.files else "pred")
    pred = torch.from_numpy(np.ascontiguousarray(d[_pred_key])).float()
    truth = torch.from_numpy(np.ascontiguousarray(d["truth"])).float()
    baseline = (torch.from_numpy(np.ascontiguousarray(d["baseline"])).float()
                if "baseline" in d.files else None)

    def _as_nchw(t):
        if t.dim() == 2:
            return t[None, None]
        if t.dim() == 3:
            return t[:, None]
        return t

    pred, truth = _as_nchw(pred), _as_nchw(truth)
    if baseline is not None:
        baseline = _as_nchw(baseline)
    n = pred.shape[0]
    dmin = float(cfg.get("display_min", 0.0))
    dmax = float(cfg.get("display_max", 0.5))

    # BATCH-WIDE SSIM/PSNR data_range — matches the frozen live board metric,
    # which computes dr = max(truth.max()-truth.min(), 1e-6) over the WHOLE batch
    # (ddssl_ldct/metrics.py:228). Calling evaluate_calibrated per single case
    # (n=1) would use that one case's own range as dr, so mean(per-case SSIM/PSNR)
    # would NOT match the board's val_ssim/val_psnr. RMSE and headroom are
    # data_range-independent, so we keep those straight from evaluate_calibrated
    # (with the correct per-case baseline); only SSIM/PSNR are recomputed here
    # against batch_dr on the calibrated, FOV-masked tensors it returns.
    batch_dr = max(float(truth.max() - truth.min()), 1e-6)

    hr, ss, ps, rm = [], [], [], []
    for i in range(n):
        b_i = baseline[i:i + 1] if baseline is not None else None
        # NOTE: evaluate_calibrated reads AGENT4CT_SAVE_RECON; unset it so the
        # per-case re-score does NOT re-dump a recon (we already have it).
        env_bak = os.environ.pop("AGENT4CT_SAVE_RECON", None)
        try:
            m = evaluate_calibrated(pred[i:i + 1], truth[i:i + 1], baseline=b_i,
                                    display_min=dmin, display_max=dmax)
        finally:
            if env_bak is not None:
                os.environ["AGENT4CT_SAVE_RECON"] = env_bak
        # Recompute SSIM/PSNR with the BATCH data_range on the calibrated,
        # FOV-masked case (evaluate_calibrated returns pred_cal UNMASKED plus the
        # fov_mask it used). This is the only line that differs from the board's
        # own per-case values; RMSE/headroom below are taken as-is.
        _pcal = m["pred_cal"]
        _fmask = m.get("fov_mask")
        _tcase = truth[i:i + 1].to(_pcal.device)
        if _fmask is not None:
            _pcal = _pcal * _fmask
            _tcase = _tcase * _fmask
        ss.append(float(_ssim_fn(_pcal, _tcase, data_range=batch_dr).cpu()))
        ps.append(float(_psnr_fn(_pcal, _tcase, data_range=batch_dr).cpu()))
        rm.append(m.get("val_rmse"))
        # headroom_case is present only when a baseline was supplied; recompute
        # explicitly against the per-case baseline_rmse so it is always defined.
        if b_i is not None and m.get("baseline_rmse"):
            hr.append(max(0.0, 1.0 - m["val_rmse"] / max(m["baseline_rmse"], 1e-12)))
        else:
            hr.append(m.get("headroom"))

    hr_m, hr_s = _mean_std(hr)
    ss_m, ss_s = _mean_std(ss)
    ps_m, ps_s = _mean_std(ps)
    rm_m, rm_s = _mean_std(rm)
    return {
        "n_cases": n,
        "test_hr_mean": hr_m, "test_hr_std": hr_s,
        "test_ssim_mean": ss_m, "test_ssim_std": ss_s,
        "test_psnr_mean": ps_m, "test_psnr_std": ps_s,
        "test_rmse_mean": rm_m, "test_rmse_std": rm_s,
    }


# ==========================================================================
# WORKER role (cluster-side): run solver once on the test split, re-score per case
# ==========================================================================
def run_worker(slug: str, solver_key: str, cfg_json: Path, split: str = "test",
               solver_src: str | None = None, noise_i0: float | None = None,
               retrain: bool = False) -> int:
    if solver_key not in SOLVER_MAP:
        print(f"[breast-test] unknown solver {solver_key!r}; choices: {list(SOLVER_MAP)}",
              flush=True)
        return 2
    default_path, env_var = SOLVER_MAP[solver_key]
    solver_path = solver_src if solver_src else default_path

    # BreastCT_Noise board (paper §5.6.7): a NO-RETRAIN robustness eval. Recon scratch +
    # docs namespace get a -noise<I0> suffix so the noisy run never clobbers the noiseless
    # one, and the loader is told to feed the Poisson-noised sino (AGENT4CT_TEST_NOISE_I0).
    noise_tag = (f"-noise{int(noise_i0)}" + ("-retrain" if retrain else "")) if noise_i0 else ""
    base_out = RUNS_BASE / f"{slug.replace('/', '__')}-breasttest{noise_tag}"
    out_dir = base_out / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force the solver to load the WHOLE held-out split (val_n = test-set size) so
    # every one of the 200 cases is reconstructed, regardless of the search val_n.
    cfg = json.loads(cfg_json.read_text())
    cfg["val_n"] = BREAST_TEST_N
    cfg_eff = out_dir / "cfg_effective.json"
    cfg_eff.write_text(json.dumps(cfg, indent=2))

    # Persist the TRAINED model checkpoint alongside the raw recon so a re-score
    # never has to retrain. Every breast solver honours AGENT4CT_MODEL_CKPT the
    # same opt-in way (save state_dict here if the file is ABSENT, load + skip
    # training if it EXISTS) — so setting it makes the trained weights inspectable
    # + reusable. Unlike Mayo (5 patients share one ckpt) the breast worker runs
    # the solver ONCE, so this is a single train -> save. Backward-compatible:
    # unset AGENT4CT_MODEL_CKPT (the prior behaviour) simply skipped this save.
    if noise_i0 and not retrain:
        # NO-RETRAIN: load the NOISELESS-trained checkpoint (the solver's opt-in
        # AGENT4CT_MODEL_CKPT loads + skips training when the file EXISTS), then infer on
        # the noisy sino. Per-scene / classical solvers have no such ckpt -> they simply
        # re-fit on the noisy input, which IS their inference (still no supervised retrain).
        model_ckpt = (RUNS_BASE / f"{slug.replace('/', '__')}-breasttest"
                      / split / "model_ckpt.pt")
    else:
        # Noiseless board OR retrain-on-noise: this ckpt path is ABSENT on the first
        # run, so the solver TRAINS from scratch and saves it here. For retrain the
        # training data is Poisson-noised (AGENT4CT_TRAIN_NOISE_I0 set below).
        model_ckpt = out_dir / "model_ckpt.pt"

    env = os.environ.copy()
    env["AGENT4CT_DATASET"] = "breast_ct"
    env["AGENT4CT_EVAL_SPLIT"] = split               # loader redirect val -> test
    env["AGENT4CT_SAVE_RECON"] = str(out_dir)        # persist raw recon for re-scoring
    env["AGENT4CT_MODEL_CKPT"] = str(model_ckpt)      # persist (or, in noise mode, LOAD) ckpt
    if noise_i0:
        if retrain:
            # RETRAIN-on-noise: noise train+val+test so the solver TRAINS on the noisy
            # sinograms (retrained-noisy board), then is scored on the noisy test split.
            env["AGENT4CT_TRAIN_NOISE_I0"] = str(int(noise_i0))
        else:
            env["AGENT4CT_TEST_NOISE_I0"] = str(int(noise_i0))  # no-retrain: noisy test/val only
    env[env_var] = str(cfg_eff)
    env.pop("AGENT4CT_EVAL_PATIENT", None)            # never mixed with Mayo path
    env.pop("AGENT4CT_SHOWCASE", None)

    print(f"[breast-test] {slug} [{split}]: python {solver_path} {out_dir}", flush=True)
    t0 = time.time()
    res = subprocess.run([sys.executable, str(REPO / solver_path), str(out_dir)], env=env)
    elapsed = time.time() - t0

    npz = out_dir / "recon_raw.npz"
    rj = out_dir / "result.json"
    if res.returncode != 0 or not npz.exists():
        print(f"[breast-test] FAILED (rc={res.returncode}, "
              f"recon_raw.npz={'present' if npz.exists() else 'MISSING'})", flush=True)
        return 1

    aggr = _score_recon_per_case(npz, cfg)
    # Sanity: solver's own whole-batch headroom (for a cross-check log only).
    solver_hr = None
    if rj.exists():
        try:
            solver_hr = json.loads(rj.read_text()).get("headroom")
        except Exception:
            pass

    ckpt_saved = model_ckpt.exists()
    if not ckpt_saved:
        print(f"[breast-test] WARN: model_ckpt not saved at {model_ckpt} "
              f"(solver may not honour AGENT4CT_MODEL_CKPT)", flush=True)

    # In noise mode the result lives in a parallel docs namespace so it feeds the
    # BreastCT_Noise board, not the noiseless breast_ct board.
    out_slug = slug.replace("-itertest/", f"-itertest{noise_tag}/") if noise_i0 else slug
    final = {
        "schema": FINAL_SCHEMA,
        "run_id": out_slug,
        "solver_key": solver_key,
        "split": split,
        "noise_i0": int(noise_i0) if noise_i0 else None,
        "retrain": bool(retrain),
        "test_n_cases": aggr["n_cases"],
        "n_test_expected": BREAST_TEST_N,
        "complete": aggr["n_cases"] == BREAST_TEST_N and aggr["test_hr_mean"] is not None,
        "elapsed_s": round(elapsed, 1),
        "solver_src": solver_path,
        "recon_npz": str(npz),
        "recon_saved": npz.exists(),
        "model_ckpt": str(model_ckpt),
        "checkpoint_saved": ckpt_saved,
        "solver_batch_headroom": solver_hr,
        "test_hr_mean": aggr["test_hr_mean"], "test_hr_std": aggr["test_hr_std"],
        "test_ssim_mean": aggr["test_ssim_mean"], "test_ssim_std": aggr["test_ssim_std"],
        "test_psnr_mean": aggr["test_psnr_mean"], "test_psnr_std": aggr["test_psnr_std"],
        "test_rmse_mean": aggr["test_rmse_mean"], "test_rmse_std": aggr["test_rmse_std"],
    }
    out_path = DOCS_RUNS / out_slug / "final.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2))
    print(f"[breast-test] wrote {out_path}  n={aggr['n_cases']}/{BREAST_TEST_N}  "
          f"hr={aggr['test_hr_mean']} ± {aggr['test_hr_std']}  "
          f"(solver batch hr={solver_hr})  "
          f"recon_saved={npz.exists()} ckpt_saved={ckpt_saved}", flush=True)
    return 0 if final["complete"] else 1


# ==========================================================================
# DISPATCH role (laptop): write cfg JSON, submit one sbatch
# ==========================================================================
def dispatch_one(slug: str, *, split: str, dry_run: bool, force: bool) -> str | None:
    smk = solver_map_key(slug)
    if smk is None:
        print(f"[breast-test] SKIP {slug}: no SOLVER_MAP entry for "
              f"solver_key={R.solver_key(slug)!r}")
        return None
    if not force and _final_complete(slug):
        print(f"[breast-test] SKIP {slug}: final.json already complete "
              f"(use --force to redo)")
        return None
    best_it, cfg = _best_iter(slug)
    if best_it is None or cfg is None:
        print(f"[breast-test] SKIP {slug}: no best iter with cfg_full found")
        return None

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_DIR / f"{slug}_breasttest_best_iter_{best_it:04d}.json"
    snap = DOCS_RUNS / slug / "iterations" / f"iter-{best_it:04d}" / "solver_src.py"
    solver_src_rel = str(snap.relative_to(REPO)) if snap.exists() else ""
    if dry_run:
        print(f"[breast-test] DRY-RUN dispatch {slug}: solver={smk} best_iter={best_it} "
              f"split={split} cfg->{cfg_path.name} "
              f"solver_src={solver_src_rel or '(SOLVER_MAP)'} sbatch={SBATCH.name}")
        return None
    cfg_path.write_text(json.dumps(cfg, indent=2))
    cmd = [
        "sbatch",
        f"--job-name=breast-test-{smk}",
        f"--export=ALL,SLUG={slug},SOLVER={smk},CFG_JSON={cfg_path},"
        f"SPLIT={split},SOLVER_SRC={solver_src_rel}",
        str(SBATCH),
    ]
    print(f"[breast-test] dispatch {slug}: solver={smk} best_iter={best_it} split={split}")
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        print(f"           {out}")
        return out.split()[-1] if out else ""
    except FileNotFoundError:
        print("[breast-test] ERROR: sbatch not found (run from the cluster login node)")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[breast-test] ERROR: sbatch failed: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="dispatch every breast solver in the allowlist")
    g.add_argument("--solver", help="dispatch ONE solver (SOLVER_MAP key OR run-id)")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="held-out split to score (default test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be dispatched, submit nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-dispatch even if final.json is already complete")
    # worker role (cluster-side, invoked by the sbatch)
    ap.add_argument("--worker", action="store_true",
                    help="cluster-side: run the solver on the split + write final.json")
    ap.add_argument("--slug", help="(worker) the run-id slug (may be <run>-itertest/iter-NNNN)")
    ap.add_argument("--cfg", help="(worker) path to the best-iter cfg JSON")
    ap.add_argument("--solver-src", default=None,
                    help="(worker) explicit solver .py to run; empty -> SOLVER_MAP path")
    ap.add_argument("--noise-i0", type=float, default=None,
                    help="(worker) BreastCT_Noise mode: feed the Poisson-noised sino at this "
                         "I0 (e.g. 100000), load the noiseless ckpt (skip retrain), write to "
                         "the -noise<I0> namespace. Requires data/stage_breast_noise.py --i0 <I0>.")
    ap.add_argument("--retrain", action="store_true",
                    help="(worker) retrained-noisy board: with --noise-i0, TRAIN the solver "
                         "from scratch on Poisson-noised train+val sino (AGENT4CT_TRAIN_NOISE_I0) "
                         "and score on the noisy test split; writes to the -noise<I0>-retrain "
                         "namespace. Requires noisy train+val+test staged.")
    args = ap.parse_args()

    if args.worker:
        if not (args.slug and args.solver and args.cfg):
            ap.error("--worker needs --slug, --solver and --cfg")
        smk = solver_map_key(args.solver)
        if smk is None:
            ap.error(f"--worker: unknown solver {args.solver!r}")
        return run_worker(args.slug, smk, Path(args.cfg), split=args.split,
                          solver_src=(args.solver_src or None), noise_i0=args.noise_i0,
                          retrain=args.retrain)

    runids = load_allow_runids()
    if args.solver:
        smk = solver_map_key(args.solver)
        targets = [s for s in runids if solver_map_key(s) == smk] if smk else []
        if not targets:
            if args.solver in runids:
                targets = [args.solver]
            else:
                ap.error(f"--solver {args.solver!r} matched no breast run-id "
                         f"(resolved key={smk})")
    elif args.all:
        targets = runids
    else:
        ap.error("pass --all or --solver <key> (or --worker for cluster-side)")

    jids = []
    for slug in targets:
        jid = dispatch_one(slug, split=args.split, dry_run=args.dry_run, force=args.force)
        if jid:
            jids.append(jid)
    if jids:
        print(f"[breast-test] submitted {len(jids)} job(s): {' '.join(jids)}")
    elif not args.dry_run:
        print("[breast-test] no jobs submitted (all skipped or dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
