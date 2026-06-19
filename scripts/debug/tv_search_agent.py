"""TV hyperparameter random-search harness (NOT an autoresearch agent).

Samples `tv_lambda`, `tv_iterations`, `tv_lr`, `tv_clip_max`, `tv_decay`
uniformly / log-uniformly inside a fixed box and records the resulting
runs in the `docs/runs/` schema so the dashboard can render them. This
is a hyperparameter sampler — *not* the Karpathy-style autoresearch
loop. The file name `tv_search_agent.py` predates the terminology
clean-up (2026-05-22); see `docs/findings.md` for the distinction
between sampler-driven and LLM-driven iterations.

Usage:
    # Start a new search
    python scripts/tv_search_agent.py new-run --iterations 20

    # Continue a search (reads last iteration to pick next params)
    python scripts/tv_search_agent.py continue --slug demo-dl-tv-search-20260515-01
"""
from __future__ import annotations
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddssl_ldct.harness import Run, Observation, utc_now_iso, REPO_ROOT


# TV hyperparameter search space
SEARCH_SPACE = {
    "tv_lambda":     (0.0001, 0.01, "log"),      # regularization
    "tv_iterations": (50, 500, "int"),            # number of iterations
    "tv_lr":         (0.001, 0.1, "log"),         # learning rate
    "tv_clip_max":   (0.03, 0.08, "linear"),      # clipping bound
    "tv_decay":      (0.0, 0.05, "linear"),       # step decay
}


def sample_params() -> dict:
    """Sample random hyperparameters from search space."""
    params = {}
    for key, (lo, hi, mode) in SEARCH_SPACE.items():
        if mode == "log":
            val = math.exp(random.uniform(math.log(lo), math.log(hi)))
        elif mode == "int":
            val = random.randint(lo, hi)
        else:  # linear
            val = random.uniform(lo, hi)
        params[key] = val
    return params


def run_tv_solver(params: dict, out_dir: Path) -> dict:
    """Run the TV solver with given params and return result."""
    # Write temporary config
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(params))
    
    # Run solver
    cmd = [
        sys.executable,
        str(REPO_ROOT / "pentathlon" / "demo_dl_reference" / "solver_tv_search.py"),
        str(out_dir),
    ]
    # Set params via environment override
    env = {
        **dict(subprocess.os.environ),
        "TV_CONFIG": json.dumps(params),
    }
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[agent] TV solver failed: {result.stderr}", file=sys.stderr)
        return None
    
    # Read result
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[agent] No result.json found", file=sys.stderr)
        return None
    
    return json.loads(result_path.read_text())


def record_iteration(run: Run, iter_n: int, params: dict, result: dict, solver_out_dir: Path | None = None) -> None:
    """Record one search iteration."""
    headroom = result.get("headroom", 0)
    val_score = result.get("val_score", 0)
    
    obs = Observation(
        ts=utc_now_iso(),
        run_id=run.slug,
        iter=iter_n,
        challenge="dl_sparse_view",
        change_class="optimizer",
        rationale=f"TV hyperparameter search: lambda={params['tv_lambda']:.5f}, "
                  f"iters={params['tv_iterations']}, lr={params['tv_lr']:.4f}, "
                  f"clip={params['tv_clip_max']:.3f}, decay={params['tv_decay']:.4f}",
        val_score=val_score,
        headroom=headroom,
        kept=headroom > 0,  # keep if positive headroom
        status="keep" if headroom > 0 else "discard",
        params_M=0.0,
        train_n=0,
        agent="tv-search",
        model="random-search",
        advice_for_others=f"TV params: lambda={params['tv_lambda']:.5f}, lr={params['tv_lr']:.4f} "
                          f"-> headroom={headroom:.4f}",
    )
    
    comparison_png = None
    if solver_out_dir is not None:
        img_path = solver_out_dir / "comparison.png"
        if img_path.exists():
            comparison_png = img_path
            print(f"[agent] Found comparison image at {img_path}")
    
    run.record_iteration(
        iter_n=iter_n,
        observation=obs,
        comparison_png=comparison_png,
    )
    print(f"[agent] Recorded iter {iter_n}: headroom={headroom:.4f} "
          f"(lambda={params['tv_lambda']:.5f}, lr={params['tv_lr']:.4f})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    
    p_new = sub.add_parser("new-run")
    p_new.add_argument("--iterations", type=int, default=20)
    p_new.add_argument("--notes", default="TV hyperparameter search")
    
    p_cont = sub.add_parser("continue")
    p_cont.add_argument("--slug", required=True)
    p_cont.add_argument("--iterations", type=int, default=20)
    
    args = p.parse_args()
    
    if args.command == "new-run":
        run = Run.create(
            challenge="dl_sparse_view",
            slug_prefix="demo-dl-tv-search",
            agent="tv-search",
            model="random-search",
            notes=args.notes,
        )
        start_iter = 1
    elif args.command == "continue":
        run = Run.load(args.slug)
        # Count existing iterations
        results = run.dir / "results.tsv"
        start_iter = 1
        if results.exists():
            start_iter = len(results.read_text().splitlines())  # header + rows
    else:
        p.print_help()
        return
    
    print(f"[agent] Starting TV search: {run.slug}")
    print(f"[agent] Iterations {start_iter} to {start_iter + args.iterations - 1}")
    
    best_headroom = 0
    best_params = None
    
    for i in range(start_iter, start_iter + args.iterations):
        print(f"\n[agent] === Iteration {i} ===")
        
        # Sample params
        params = sample_params()
        print(f"[agent] Params: {json.dumps(params)}")
        
        # Run solver
        out_dir = Path(f"/tmp/tv-search-{run.slug}-{i:04d}")
        out_dir.mkdir(exist_ok=True)
        
        result = run_tv_solver(params, out_dir)
        if result is None:
            print(f"[agent] Iteration {i} failed, skipping")
            continue
        
        headroom = result.get("headroom", 0)
        print(f"[agent] Result: headroom={headroom:.4f}, SSIM={result.get('val_score', 0):.4f}")
        
        # Track best
        if headroom > best_headroom:
            best_headroom = headroom
            best_params = params.copy()
            print(f"[agent] *** NEW BEST: headroom={best_headroom:.4f} ***")
        
        # Record
        record_iteration(run, i, params, result)
    
    print(f"\n[agent] Search complete!")
    print(f"[agent] Best headroom: {best_headroom:.4f}")
    if best_params:
        print(f"[agent] Best params: {json.dumps(best_params)}")


if __name__ == "__main__":
    main()
