"""Thin CLI wrapper around ``ddssl_ldct.harness`` for the agent.

Examples:

    # Start a new run targeting one challenge.
    python scripts/agent4ct_record.py new-run \
        --challenge dl_sparse_view --slug-prefix dl-sparse-view \
        --notes "first run, baseline U-Net"

    # Record one iteration's result.
    python scripts/agent4ct_record.py record \
        --slug dl-sparse-view-20260513-01 --iter 1 \
        --val-score 0.58 --headroom 0.37 \
        --change-class architecture --rationale "baseline U-Net c=24" \
        --kept true --commit $(git rev-parse --short HEAD) \
        --comparison runs/sanity.png \
        --solver pentathlon/dl_sparse_view/solver.py

    # Append a stage check (every 30 iterations).
    python scripts/agent4ct_record.py stage \
        --slug dl-sparse-view-20260513-01 --iter 30 \
        --stage-val-score 0.55 --stage-headroom 0.30 \
        --iter-val-score 0.62 --verdict overfit \
        --notes "iter val rising but stage flat -> reduce params"

    # Mark the run finished.
    python scripts/agent4ct_record.py finalize \
        --slug dl-sparse-view-20260513-01 --status done
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddssl_ldct.harness import (
    Run, Observation, FinalResult,
    utc_now_iso, write_final,
    git_commit_and_push, build_iter_commit_message,
    REPO_ROOT,
)


def cmd_new_run(args):
    run = Run.create(
        challenge=args.challenge,
        slug_prefix=args.slug_prefix,
        agent=args.agent,
        model=args.model,
        notes=args.notes,
    )
    print(run.slug)


def cmd_record(args):
    run = Run.load(args.slug)
    obs = Observation(
        ts=utc_now_iso(),
        run_id=run.slug,
        iter=args.iter,
        challenge=args.challenge or run.manifest["challenge"],
        change_class=args.change_class,
        rationale=args.rationale,
        val_score=args.val_score,
        headroom=args.headroom,
        delta_vs_best=args.delta_vs_best,
        kept=args.kept,
        status=args.status or ("keep" if args.kept else "discard"),
        params_M=args.params_M,
        train_n=args.train_n,
        agent=args.agent,
        model=args.model,
        advice_for_others=args.advice,
        commit=args.commit,
    )
    d = run.record_iteration(
        iter_n=args.iter,
        observation=obs,
        comparison_png=args.comparison,
        solver_src=args.solver,
        stdout_log=args.stdout_log,
    )
    print(d)
    if args.commit_git:
        msg = build_iter_commit_message(run.slug, args.iter, obs)
        sha = git_commit_and_push(REPO_ROOT, msg, push=args.push)
        if sha:
            print(f"committed {sha}{' & pushed' if args.push else ''}")
        else:
            print("nothing to commit")


def cmd_stage(args):
    run = Run.load(args.slug)
    run.record_stage(
        iter_n=args.iter,
        stage_val_score=args.stage_val_score,
        stage_headroom=args.stage_headroom,
        iter_val_score=args.iter_val_score,
        verdict=args.verdict,
        notes=args.notes,
    )


def cmd_finalize(args):
    run = Run.load(args.slug)
    # Discover best iteration so far from results.tsv.
    results_path = run.dir / "results.tsv"
    best_iter, best_val, best_hr = None, None, None
    if results_path.exists():
        for line in results_path.read_text().splitlines()[1:]:
            cells = line.split("\t")
            try:
                it = int(cells[0]); v = float(cells[2])
                if best_val is None or v > best_val:
                    best_val = v; best_iter = it
                    try:
                        best_hr = float(cells[3])
                    except (ValueError, IndexError):
                        pass
            except (ValueError, IndexError):
                continue
    n_iter = 0
    if results_path.exists():
        n_iter = sum(1 for _ in results_path.read_text().splitlines()[1:])

    final = FinalResult(
        slug=run.slug,
        ended=utc_now_iso(),
        n_iterations=n_iter,
        stop_reason=args.stop_reason,
        best_iter=best_iter,
        best_val_score=best_val,
        best_headroom=best_hr,
        final_test_score=args.final_test_score,
        final_test_headroom=args.final_test_headroom,
        final_test_comparison=args.final_test_comparison,
        notes=args.notes or "",
    )
    write_final(run, final)
    if args.commit_git:
        msg = (f"finalize {run.slug}: "
               f"test_score={args.final_test_score} "
               f"test_hr={args.final_test_headroom} "
               f"stop={args.stop_reason}")
        if args.notes:
            msg += "\n\n" + args.notes
        sha = git_commit_and_push(REPO_ROOT, msg, push=args.push)
        if sha:
            print(f"committed {sha}{' & pushed' if args.push else ''}")


def _parse_bool(s: str | None) -> bool | None:
    if s is None:
        return None
    return s.lower() in ("1", "true", "yes", "y", "keep")


def main():
    p = argparse.ArgumentParser(prog="agent4ct-record",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new-run")
    p_new.add_argument("--challenge", required=True)
    p_new.add_argument("--slug-prefix", required=True)
    p_new.add_argument("--agent", default="claude")
    p_new.add_argument("--model", default="claude-sonnet-4.5")
    p_new.add_argument("--notes", default="")
    p_new.set_defaults(func=cmd_new_run)

    p_rec = sub.add_parser("record")
    p_rec.add_argument("--slug", required=True)
    p_rec.add_argument("--iter", type=int, required=True)
    p_rec.add_argument("--challenge", default="")
    p_rec.add_argument("--val-score", dest="val_score", type=float, default=None)
    p_rec.add_argument("--headroom", type=float, default=None)
    p_rec.add_argument("--delta-vs-best", dest="delta_vs_best", type=float, default=None)
    p_rec.add_argument("--change-class", dest="change_class", default="other")
    p_rec.add_argument("--rationale", required=True)
    p_rec.add_argument("--kept", type=_parse_bool, default=None)
    p_rec.add_argument("--status", default=None,
                       choices=["keep", "discard", "crash", "timeout", None])
    p_rec.add_argument("--params-M", dest="params_M", type=float, default=None)
    p_rec.add_argument("--train-n", dest="train_n", type=int, default=None)
    p_rec.add_argument("--agent", default="claude")
    p_rec.add_argument("--model", default="claude-sonnet-4.5")
    p_rec.add_argument("--advice", default=None,
                       help="single-sentence generalisable advice for other agents")
    p_rec.add_argument("--commit", default="")
    p_rec.add_argument("--comparison", default=None,
                       help="path to comparison PNG; copied into the iteration dir")
    p_rec.add_argument("--solver", default=None,
                       help="path to solver.py snapshot to capture")
    p_rec.add_argument("--stdout-log", dest="stdout_log", default=None,
                       help="contents of a stdout log to save (string, not path)")
    p_rec.add_argument("--no-commit", dest="commit_git", action="store_false",
                       help="skip auto git commit (default: commit after recording)")
    p_rec.add_argument("--no-push", dest="push", action="store_false",
                       help="skip auto git push (default: push after commit)")
    p_rec.set_defaults(func=cmd_record, commit_git=True, push=True)

    p_st = sub.add_parser("stage")
    p_st.add_argument("--slug", required=True)
    p_st.add_argument("--iter", type=int, required=True)
    p_st.add_argument("--stage-val-score", dest="stage_val_score", type=float, required=True)
    p_st.add_argument("--stage-headroom", dest="stage_headroom", type=float, required=True)
    p_st.add_argument("--iter-val-score", dest="iter_val_score", type=float, required=True)
    p_st.add_argument("--verdict", choices=["ok", "overfit"], required=True)
    p_st.add_argument("--notes", default="")
    p_st.set_defaults(func=cmd_stage)

    p_fin = sub.add_parser("finalize",
        description="Run-level finalize: writes final.json with test scores "
                    "+ stop reason. Run this once per run, after the "
                    "iteration phase is over and you've evaluated the best "
                    "iteration's solver on the held test set.")
    p_fin.add_argument("--slug", required=True)
    p_fin.add_argument("--stop-reason", dest="stop_reason",
                       choices=["budget", "no_improvement", "overfit_stage",
                                "manual", "crashed"],
                       required=True,
                       help="why iteration phase ended")
    p_fin.add_argument("--final-test-score", dest="final_test_score", type=float,
                       default=None, help="score on the held test set")
    p_fin.add_argument("--final-test-headroom", dest="final_test_headroom",
                       type=float, default=None, help="headroom on held test set")
    p_fin.add_argument("--final-test-comparison", dest="final_test_comparison",
                       default=None, help="path (relative to docs/) to a comparison PNG")
    p_fin.add_argument("--notes", default="")
    p_fin.add_argument("--no-commit", dest="commit_git", action="store_false")
    p_fin.add_argument("--no-push", dest="push", action="store_false")
    p_fin.set_defaults(func=cmd_finalize, commit_git=True, push=True)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
