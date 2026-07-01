"""rescore_pe_iters.py — re-score EVERY param-efficient iteration under the FROZEN
corrected metric, so the campaign's idea-by-idea trajectory becomes comparable on
one metric (iters 1-27 ran at the old 0.05-clamp metric; 28-40 at a 0.09 variant —
never one metric). Each iter is code-evolving, so we run its EXACT solver_src.py
snapshot + its cfg_full, persisting the recon.

Two evaluation targets:

VAL (default, L277) — the campaign basis, one eval pass:
  --dispatch : write each iter's cfg + submit one mayo_iter_rescore.sbatch job
  --collect  : read each iter's NEW result.json (corrected val) + the OLD
               observation.json, print the trajectory table (iter, old hr, NEW hr,
               NEW ssim, rationale) and write it to the rescore dir.

TEST (--test) — the HONEST Mayo metric: train once per held-out patient, mean ± std
over the 5 TEST patients (L014 L056 L058 L075 L123). Reuses the proven per-patient
worker loop in score_mayo_testset.py (one patient per eval pass — NEVER batch them,
or PYRO-NN crashes with "invalid resource handle" on mixed pixel-spacings):
  --test --dispatch : write each iter's cfg + submit ONE mayo_iter_testscore.sbatch
                      per iter (each loops the 5 patients ~= 5x train ~ 1-2h wall).
  --test --collect  : read each iter's final.json (mean ± std) + the OLD
                      observation.json; print + write trajectory_test.json with
                      columns iter, rationale, params, test_hr_mean±std,
                      test_ssim_mean±std (+ psnr/rmse/val_L277 for completeness).

Run on the cluster (sbatch). Idempotent dispatch: VAL skips an iter whose corrected
result.json already exists; TEST skips an iter whose final.json already has all 5
patients. --force re-dispatches regardless.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "mayo-ldct-claude-agentic-param-efficient-search-20260624-01"
ITERS = REPO / "docs" / "runs" / SLUG / "iterations"
RUNS = Path(os.environ.get("AGENT4CT_RUNS", "/cluster/maier/Agent4CT/runs"))
SBATCH = REPO / "cluster" / "slurm" / "mayo_iter_rescore.sbatch"
CLIP_FREE_VALUE = 1000.0   # clip_max -> 1e3 == effectively no upper clamp (μ ≪ 1)

# --- TEST-mode constants (honest 5-patient mean±std metric) -----------------
DOCS_RUNS = REPO / "docs" / "runs"
TEST_SBATCH = REPO / "cluster" / "slurm" / "mayo_iter_testscore.sbatch"
TEST_NS = "pe-iter-testeval"          # output namespace (NOT a real run-id)
TEST_CFG_DIR = REPO / "agentic_cfgs" / "pe_testeval"
TEST_PATIENTS = ["L014", "L056", "L058", "L075", "L123"]


def _paths(clamp_free: bool):
    """Separate output + cfg dirs for the as-built vs clamp-free trajectories so
    they never collide. iters 1-27 clamp internally (clip_max used); 28-40 are
    already clamp-free, so the clamp-free run only changes 1-27 (28-40 reproduce)."""
    tag = "-clampfree" if clamp_free else ""
    return (RUNS / f"pe-iter-rescore{tag}",
            REPO / "agentic_cfgs" / f"pe_rescore{tag}")


def iter_dirs():
    return sorted(d for d in ITERS.glob("iter-*") if (d / "solver_src.py").exists())


def dispatch(force: bool, clamp_free: bool):
    outbase, cfgdir = _paths(clamp_free)
    cfgdir.mkdir(parents=True, exist_ok=True)
    tag = "cf" if clamp_free else "as"
    jids = []
    for d in iter_dirs():
        name = d.name                       # iter-0007
        outdir = outbase / name
        if not force and (outdir / "result.json").exists():
            print(f"[pe-rescore] skip {name}: result.json exists")
            continue
        try:
            obs = json.loads((d / "observation.json").read_text())
        except Exception as e:
            print(f"[pe-rescore] skip {name}: bad observation.json ({e})")
            continue
        cfg = obs.get("cfg_full")
        if not cfg:
            print(f"[pe-rescore] skip {name}: no cfg_full")
            continue
        if clamp_free:
            # Lift the in-loop μ upper clamp (iters 1-27 use clip_max; 28-40 ignore
            # it — already clamp-free). 1e3 ≫ μ, so clamp(0,1e3) == μ≥0 floor only.
            cfg = dict(cfg); cfg["clip_max"] = CLIP_FREE_VALUE
        cfgp = cfgdir / f"{name}.json"
        cfgp.write_text(json.dumps(cfg, indent=2))
        src_rel = str((d / "solver_src.py").relative_to(REPO))
        cmd = ["sbatch", f"--job-name=pe-rescore-{tag}-{name}",
               f"--export=ALL,SOLVER_SRC={src_rel},CFG_JSON={cfgp},OUTDIR={outdir}",
               str(SBATCH)]
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            jid = out.split()[-1]
            jids.append(jid)
            print(f"[pe-rescore] {name} -> job {jid}")
        except Exception as e:
            print(f"[pe-rescore] {name}: sbatch FAILED ({e})")
    print(f"[pe-rescore] submitted {len(jids)} jobs")


def collect(clamp_free: bool):
    outbase, _ = _paths(clamp_free)
    rows = []
    for d in iter_dirs():
        name = d.name
        it = int(name.split("-")[-1])
        obs = json.loads((d / "observation.json").read_text())
        old_hr = obs.get("headroom")
        rat = (obs.get("cfg_full") or {}).get("rationale") or ""
        params = (obs.get("cfg_full") or {}).get("params") or obs.get("params_M")
        rp = outbase / name / "result.json"
        new_hr = new_ss = None
        if rp.exists():
            try:
                r = json.loads(rp.read_text())
                new_hr, new_ss = r.get("headroom"), r.get("val_ssim")
            except Exception:
                pass
        rows.append({"iter": it, "old_hr": old_hr, "corr_hr": new_hr,
                     "corr_ssim": new_ss, "params": params, "rationale": rat})
    rows.sort(key=lambda r: r["iter"])
    outbase.mkdir(parents=True, exist_ok=True)
    (outbase / "trajectory.json").write_text(json.dumps(rows, indent=2))
    print(f"{'it':>3} {'old_hr':>8} {'CORR_hr':>8} {'CORR_ssim':>9} {'params':>7}  rationale")
    print("-" * 100)
    for r in rows:
        def f(x, w, p=4):
            return (f"%{w}.{p}f" % x) if isinstance(x, (int, float)) else str(x).rjust(w)
        print(f"{r['iter']:>3} {f(r['old_hr'],8)} {f(r['corr_hr'],8)} {f(r['corr_ssim'],9)} "
              f"{str(r['params']):>7}  {r['rationale'][:64]}")
    n = sum(1 for r in rows if isinstance(r["corr_hr"], (int, float)))
    print(f"\n{n}/{len(rows)} iters have a corrected score; wrote {outbase/'trajectory.json'}")


# ==========================================================================
# TEST mode: per-iter, 5 held-out patients, mean ± std (the HONEST Mayo metric)
# ==========================================================================
def _test_slug(name: str) -> str:
    """Output namespace for an iter's test aggregate: pe-iter-testeval/iter-NNNN."""
    return f"{TEST_NS}/{name}"


def _test_final_complete(name: str) -> bool:
    fp = DOCS_RUNS / _test_slug(name) / "final.json"
    if not fp.exists():
        return False
    try:
        obj = json.loads(fp.read_text())
    except Exception:
        return False
    pats = obj.get("patients") or {}
    return all(p in pats and pats[p] is not None for p in TEST_PATIENTS)


def dispatch_test(force: bool):
    """One sbatch per iter; each loops the 5 TEST patients (one eval pass each)."""
    TEST_CFG_DIR.mkdir(parents=True, exist_ok=True)
    jids = []
    for d in iter_dirs():
        name = d.name                                  # iter-0007
        if not force and _test_final_complete(name):
            print(f"[pe-testscore] skip {name}: final.json has all "
                  f"{len(TEST_PATIENTS)} patients")
            continue
        try:
            obs = json.loads((d / "observation.json").read_text())
        except Exception as e:
            print(f"[pe-testscore] skip {name}: bad observation.json ({e})")
            continue
        cfg = obs.get("cfg_full")
        if not cfg:
            print(f"[pe-testscore] skip {name}: no cfg_full")
            continue
        cfgp = TEST_CFG_DIR / f"{name}.json"
        cfgp.write_text(json.dumps(cfg, indent=2))
        src_rel = str((d / "solver_src.py").relative_to(REPO))
        slug = _test_slug(name)
        cmd = ["sbatch", f"--job-name=pe-testscore-{name}",
               f"--export=ALL,SLUG={slug},CFG_JSON={cfgp},SOLVER_SRC={src_rel}",
               str(TEST_SBATCH)]
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            jid = out.split()[-1]
            jids.append(jid)
            print(f"[pe-testscore] {name} -> job {jid}")
        except Exception as e:
            print(f"[pe-testscore] {name}: sbatch FAILED ({e})")
    print(f"[pe-testscore] submitted {len(jids)} jobs: {' '.join(jids)}")


def collect_test():
    """Read each iter's TEST final.json (mean±std over the 5 patients) + the OLD
    observation.json; print + write trajectory_test.json."""
    rows = []
    for d in iter_dirs():
        name = d.name
        it = int(name.split("-")[-1])
        obs = json.loads((d / "observation.json").read_text())
        cfg = obs.get("cfg_full") or {}
        rat = obs.get("rationale") or cfg.get("rationale") or ""
        params = obs.get("params_M") or cfg.get("params")
        row = {"iter": it, "params": params, "rationale": rat,
               "old_val_hr": obs.get("headroom"),
               "test_hr_mean": None, "test_hr_std": None,
               "test_ssim_mean": None, "test_ssim_std": None,
               "test_psnr_mean": None, "test_psnr_std": None,
               "test_rmse_mean": None, "test_rmse_std": None,
               "val_L277_hr": None, "n_patients_ok": 0}
        fp = DOCS_RUNS / _test_slug(name) / "final.json"
        if fp.exists():
            try:
                f = json.loads(fp.read_text())
                for k in ("test_hr_mean", "test_hr_std", "test_ssim_mean",
                          "test_ssim_std", "test_psnr_mean", "test_psnr_std",
                          "test_rmse_mean", "test_rmse_std"):
                    row[k] = f.get(k)
                row["val_L277_hr"] = (f.get("val_L277") or {}).get("headroom")
                pats = f.get("patients") or {}
                row["n_patients_ok"] = sum(1 for p in TEST_PATIENTS
                                           if pats.get(p) is not None)
            except Exception:
                pass
        rows.append(row)
    rows.sort(key=lambda r: r["iter"])
    outdir = DOCS_RUNS / TEST_NS
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectory_test.json").write_text(json.dumps(rows, indent=2))

    def fmt(m, s):
        if not isinstance(m, (int, float)):
            return f"{'--':>15}"
        ss = f"{s:.4f}" if isinstance(s, (int, float)) else "?"
        return f"{m:.4f}±{ss:<6}"

    print(f"{'it':>3} {'params':>8} {'TEST_hr (m±s)':>16} {'TEST_ssim (m±s)':>16} "
          f"{'n':>2}  rationale")
    print("-" * 110)
    for r in rows:
        p = (f"{r['params']:.6g}" if isinstance(r['params'], (int, float))
             else str(r['params']))
        print(f"{r['iter']:>3} {p:>8} {fmt(r['test_hr_mean'], r['test_hr_std']):>16} "
              f"{fmt(r['test_ssim_mean'], r['test_ssim_std']):>16} "
              f"{r['n_patients_ok']:>2}  {r['rationale'][:56]}")
    n = sum(1 for r in rows if isinstance(r["test_hr_mean"], (int, float)))
    print(f"\n{n}/{len(rows)} iters have a TEST score "
          f"(all 5 patients: {sum(1 for r in rows if r['n_patients_ok'] == 5)}); "
          f"wrote {outdir/'trajectory_test.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="evaluate on the 5 held-out TEST patients (honest "
                         "mean±std metric) instead of val L277")
    ap.add_argument("--clamp-free", dest="clamp_free", action="store_true",
                    help="lift the in-loop μ upper clamp (clip_max->1e3) so iters "
                         "1-27 are scored clamp-free; separate output dir "
                         "(VAL mode only)")
    a = ap.parse_args()
    if a.test:
        if a.clamp_free:
            ap.error("--clamp-free is VAL-mode only (TEST snapshots already "
                     "clamp-free / cfg_full carried as-is)")
        if a.dispatch:
            dispatch_test(a.force)
        elif a.collect:
            collect_test()
        else:
            ap.error("pass --dispatch or --collect with --test")
    elif a.dispatch:
        dispatch(a.force, a.clamp_free)
    elif a.collect:
        collect(a.clamp_free)
    else:
        ap.error("pass --dispatch or --collect")
