#!/usr/bin/env python3
"""Record demo reference results to the autoresearch dashboard.

Run this after the Slurm jobs complete to import results into docs/runs/.
Usage:
    python scripts/record_demo_refs.py --fbp <jobid> --tv <jobid> --dd <jobid> --itnet <jobid>
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddssl_ldct.harness import Run, Observation, utc_now_iso, git_commit_and_push, REPO_ROOT


def fetch_result(job_id: int, kind: str) -> dict:
    """Read result.json from cluster runs directory."""
    # Try local first (if already synced), then cluster
    local_pattern = f"runs/demo-ref-{kind}-{job_id}/result.json"
    local_path = REPO_ROOT / local_pattern
    if local_path.exists():
        return json.loads(local_path.read_text())

    # Fetch from cluster via ssh
    cmd = [
        "ssh", "-o", "BatchMode=yes",
        "maier@cluster.i5.informatik.uni-erlangen.de",
        f"cat /cluster/maier/Agent4CT/{local_pattern}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FileNotFoundError(f"Cannot find result for {kind} job {job_id}")
    return json.loads(result.stdout)


def create_demo_run() -> Run:
    """Create or load the demo-dl-reference run."""
    try:
        return Run.load("demo-dl-reference-20260515-01")
    except FileNotFoundError:
        return Run.create(
            challenge="dl_sparse_view",
            slug_prefix="demo-dl-reference",
            agent="reference",
            model="demo-implementations",
            notes="Reference reconstructions: FBP, TV, Dual-Domain, ItNet on synthetic phantoms",
        )


def record_iteration(run: Run, iter_n: int, kind: str, result: dict, job_id: int) -> None:
    """Record one reference result as an iteration."""
    obs = Observation(
        ts=utc_now_iso(),
        run_id=run.slug,
        iter=iter_n,
        challenge="dl_sparse_view",
        change_class="reference",
        rationale=f"{kind} reconstruction on synthetic random-ellipse phantoms. Job {job_id}.",
        val_score=result.get("val_score"),
        headroom=result.get("headroom"),
        kept=True,
        status="keep",
        params_M=result.get("params_M"),
        train_n=result.get("train_n", 0),
        agent="reference",
        model=kind,
        advice_for_others=f"Reference baseline: {kind} achieves headroom={result.get('headroom', 0):.4f}",
        commit="",
    )

    # Try to find comparison.png
    comp_path = REPO_ROOT / f"runs/demo-ref-{kind.lower()}-{job_id}/comparison.png"
    if not comp_path.exists():
        # Try cluster
        comp_path = None

    run.record_iteration(
        iter_n=iter_n,
        observation=obs,
        comparison_png=comp_path,
        solver_src=None,
    )
    print(f"Recorded {kind} as iter {iter_n}: headroom={result.get('headroom', 0):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fbp", type=int, required=True, help="Slurm job ID for FBP")
    p.add_argument("--tv", type=int, required=True, help="Slurm job ID for TV")
    p.add_argument("--dd", type=int, required=True, help="Slurm job ID for Dual-Domain")
    p.add_argument("--itnet", type=int, required=True, help="Slurm job ID for ItNet")
    p.add_argument("--no-commit", action="store_true", help="Skip git commit")
    args = p.parse_args()

    run = create_demo_run()
    print(f"Using run: {run.slug}")

    results = {
        "fbp": fetch_result(args.fbp, "fbp"),
        "tv": fetch_result(args.tv, "tv"),
        "dd": fetch_result(args.dd, "dd"),
        "itnet": fetch_result(args.itnet, "itnet"),
    }

    for i, (kind, result) in enumerate(results.items(), start=1):
        job_id = getattr(args, kind)
        record_iteration(run, i, kind.upper(), result, job_id)

    # Commit
    if not args.no_commit:
        msg = f"demo-ref: record {len(results)} reference implementations (headrooms: " + \
              ", ".join(f"{k}={v.get('headroom', 0):.4f}" for k, v in results.items()) + ")"
        sha = git_commit_and_push(REPO_ROOT, msg, push=True)
        print(f"Committed {sha}")

    print("Done!")


if __name__ == "__main__":
    main()
