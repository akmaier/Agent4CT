"""score_breast_noise.py — BreastCT_Noise board driver (paper §5.6.7).

For each breast solver, take its NOISELESS test-best iter (from
breast_testsweep_selection.json), and re-evaluate that SAME iter on the Poisson-noised
test set at a given dose I0 — WITHOUT retraining: supervised solvers load the noiseless
`model_ckpt.pt` (skip training); per-scene/classical solvers re-fit on the noisy sino
(their inference). Writes final.json into the `<run>-itertest-noise<I0>/` namespace via
score_breast_testset.run_worker(..., noise_i0=I0). Robustness-to-input-noise board.

Submits ONE sbatch per solver (inference-only -> fast). naf/r2gaussian (DNF on the
noiseless board) are skipped. Requires: data/stage_breast_noise.py --i0 <I0> already run.

Usage (cluster login node):
  python scripts/score_breast_noise.py --i0 100000                 # all non-DNF solvers
  python scripts/score_breast_noise.py --i0 100000 --solvers tv-iterative,manhart-pwls-tv
  python scripts/score_breast_noise.py --i0 100000 --dry-run
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "runs"
SEL = DOCS / "breast_testsweep_selection.json"
CFG_DIR = REPO / "agentic_cfgs" / "breast_noise"
DNF = {"naf", "r2gaussian"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i0", type=float, required=True)
    ap.add_argument("--solvers", default=None, help="comma list (default: all non-DNF)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude-small", action="store_true",
                    help="also exclude the 11GB nodes (for heavy solvers)")
    a = ap.parse_args()
    i0 = int(a.i0)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    sel = json.loads(SEL.read_text())
    want = set(a.solvers.split(",")) if a.solvers else None
    subs, skipped = [], []
    for s in sel:
        solver = s["solver"]
        if solver in DNF or s.get("test_best_hr_mean") is None:
            skipped.append((solver, "DNF/no noiseless score")); continue
        if want and solver not in want:
            continue
        rid = s["run_id"]; it = s["test_best_iter"]; itn = f"iter-{it:04d}"
        obs_p = DOCS / rid / "iterations" / itn / "observation.json"
        if not obs_p.exists():
            skipped.append((solver, f"no observation {itn}")); continue
        cfg = json.loads(obs_p.read_text()).get("cfg_full")
        if not cfg:
            skipped.append((solver, "no cfg_full")); continue
        cfg_p = CFG_DIR / f"{rid}__{itn}.json"
        cfg_p.write_text(json.dumps(cfg))
        snap = DOCS / rid / "iterations" / itn / "solver_src.py"
        src = f"--solver-src {snap.relative_to(REPO)}" if snap.exists() else ""
        slug = f"{rid}-itertest/{itn}"
        excl = "lme49,lme50,lme51,lme52,lme170" if a.exclude_small else "lme49,lme52,lme170"
        wrap = (f". {REPO}/.venv/bin/activate && cd {REPO} && "
                f"export HDF5_USE_FILE_LOCKING=FALSE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                f"PYTHONPATH={REPO} && python scripts/score_breast_testset.py --worker "
                f"--slug {slug} --solver {solver} --cfg {cfg_p.relative_to(REPO)} {src} --noise-i0 {i0}")
        cmd = ["sbatch", "--job-name=breast-noise", "--partition=main", "--qos=turbo",
               "--gres=gpu:1", "--cpus-per-task=4", "--mem=32G", "--time=00:30:00",
               f"--exclude={excl}", "-o", f"results/slurm/breast-noise-{solver}-%j.out",
               "-e", f"results/slurm/breast-noise-{solver}-%j.err", f"--wrap={wrap}"]
        if a.dry_run:
            print(f"[dry] {solver} {itn} (noiseless hr={s['test_best_hr_mean']:.4f})")
            subs.append(solver); continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = "Submitted batch job" in r.stdout
        print(f"  {solver} {itn}: {r.stdout.strip() or r.stderr.strip()}")
        if ok:
            subs.append(solver)
    print(f"\n[breast-noise] I0={i0}: submitted {len(subs)} solvers"
          + (f", skipped {len(skipped)}" if skipped else ""))
    for sv, why in skipped:
        print(f"  skip {sv}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
