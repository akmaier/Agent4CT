"""score_breast_noise_retrain.py — BreastCT_Noise RETRAINED board driver.

Companion to score_breast_noise.py. Where that board RE-EVALUATES the noiseless
checkpoints on the Poisson-noised test set (no retraining), THIS board takes each
*trainable* solver's noiseless test-best iter (from breast_testsweep_selection.json)
and RETRAINS that SAME configuration from scratch on the Poisson-noised TRAIN set,
then scores it on the noisy TEST set. It answers the reviewer question: if you train
for the noise, does the clean-optimal ranking come back?

Only the fixed clean-best config is retrained — no hyperparameter re-search. Classical /
per-scene / zero-shot solvers are NOT retrained (they have no supervised weights; their
retrained-noisy score is identical to their no-retrain noisy score, carried over when the
board is built). naf/r2gaussian (DNF on the noiseless board) are skipped.

Submits ONE sbatch per trainable solver, calling
  score_breast_testset.py --worker ... --noise-i0 <I0> --retrain
which sets AGENT4CT_TRAIN_NOISE_I0 (noise train+val+test) and trains a fresh ckpt.
Results land in the `<run>-itertest-noise<I0>-retrain/` docs namespace.

Requires (cluster): data/stage_breast_noise.py --i0 <I0> already run for splits
train, val AND test.

Usage (cluster login node):
  python scripts/score_breast_noise_retrain.py --i0 100000            # all trainable
  python scripts/score_breast_noise_retrain.py --i0 100000 --solvers itnet,uswin
  python scripts/score_breast_noise_retrain.py --i0 100000 --dry-run
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "runs"
SEL = DOCS / "breast_testsweep_selection.json"
CFG_DIR = REPO / "agentic_cfgs" / "breast_noise_retrain"

# Solvers with supervised / self-supervised / generative weights fit on the TRAIN set:
# these are the ones for which "retrain on noisy data" is a distinct experiment. Names
# are the dashed solver keys as they appear in breast_testsweep_selection.json.
TRAINABLE = {
    "dual-domain-supervised", "dual-domain-bilateral-supervised",
    "dual-domain-n2i", "dual-domain-bilateral-n2i",
    "itnet", "itnet-v2", "itnet-v3", "uswin", "learned-primal-dual",
    "hammernik-2017", "hammernik-vn", "wu-2015-trainable",
    "tv-iterative-supervised", "param-efficient",
    "fastdiff-flow-pixel-constrained", "fastdiff-flow-pixel-unconstrained",
    "fastdiff-wdm-wavelet-constrained", "fastdiff-wdm-wavelet-unconstrained",
}
# Classical / per-scene / zero-shot solvers: NOT retrained (no supervised weights).
# Their retrained-noisy score == their no-retrain noisy score, carried over at build time.
CARRY_OVER = {
    "tv-iterative", "manhart-pwls-tv", "manduca-bilateral", "wu-2015", "ram-zeroshot",
}
DNF = {"naf", "r2gaussian"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i0", type=float, required=True)
    ap.add_argument("--solvers", default=None, help="comma list (default: all trainable)")
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
        if solver not in TRAINABLE:
            skipped.append((solver, "carry-over/DNF (not retrained)")); continue
        if s.get("test_best_hr_mean") is None or s.get("test_best_iter") is None:
            skipped.append((solver, "no noiseless best iter")); continue
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
                f"--slug {slug} --solver {solver} --cfg {cfg_p.relative_to(REPO)} {src} "
                f"--noise-i0 {i0} --retrain")
        cmd = ["sbatch", "--job-name=breast-retrain", "--partition=main", "--qos=turbo",
               "--gres=gpu:1", "--cpus-per-task=8", "--mem=48G", "--time=00:50:00",
               f"--exclude={excl}", "-o", f"results/slurm/breast-retrain-{solver}-%j.out",
               "-e", f"results/slurm/breast-retrain-{solver}-%j.err", f"--wrap={wrap}"]
        if a.dry_run:
            print(f"[dry] {solver} {itn} (noiseless hr={s['test_best_hr_mean']:.4f}) -> retrain@I0={i0}")
            subs.append(solver); continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = "Submitted batch job" in r.stdout
        print(f"  {solver} {itn}: {r.stdout.strip() or r.stderr.strip()}")
        if ok:
            subs.append(solver)
    print(f"\n[breast-retrain] I0={i0}: submitted {len(subs)} trainable solvers"
          + (f", skipped {len(skipped)}" if skipped else ""))
    for sv, why in skipped:
        print(f"  skip {sv}: {why}")
    print(f"[breast-retrain] carry-over (not retrained): {sorted(CARRY_OVER)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
