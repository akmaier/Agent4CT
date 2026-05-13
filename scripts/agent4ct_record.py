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

from ddssl_ldct.harness import Run, Observation, utc_now_iso


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
    run.finalize(status=args.status, notes=args.notes or "")


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
    p_rec.set_defaults(func=cmd_record)

    p_st = sub.add_parser("stage")
    p_st.add_argument("--slug", required=True)
    p_st.add_argument("--iter", type=int, required=True)
    p_st.add_argument("--stage-val-score", dest="stage_val_score", type=float, required=True)
    p_st.add_argument("--stage-headroom", dest="stage_headroom", type=float, required=True)
    p_st.add_argument("--iter-val-score", dest="iter_val_score", type=float, required=True)
    p_st.add_argument("--verdict", choices=["ok", "overfit"], required=True)
    p_st.add_argument("--notes", default="")
    p_st.set_defaults(func=cmd_stage)

    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--slug", required=True)
    p_fin.add_argument("--status", default="done",
                       choices=["done", "abandoned", "crashed"])
    p_fin.add_argument("--notes", default="")
    p_fin.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
