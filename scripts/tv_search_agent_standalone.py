"""TV hyperparameter search agent (standalone, no ddssl_ldct imports).

This script runs the TV solver with different hyperparameters and records
results directly to docs/runs/ without importing ddssl_ldct.

Usage:
    python scripts/tv_search_agent_standalone.py new-run --iterations 20
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must NOT import ddssl_ldct (needs torch which may not be available in harness)
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO_ROOT / "docs" / "runs"


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_slug(prefix):
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = 1
    if DOCS_RUNS.exists():
        for d in DOCS_RUNS.iterdir():
            if d.is_dir() and d.name.startswith(f"{prefix}-{today}-"):
                try:
                    tail = int(d.name.split("-")[-1])
                    seq = max(seq, tail + 1)
                except ValueError:
                    pass
    return f"{prefix}-{today}-{seq:02d}"


def create_run(challenge, slug_prefix, agent, model, notes=""):
    DOCS_RUNS.mkdir(parents=True, exist_ok=True)
    slug = make_slug(slug_prefix)
    run_dir = DOCS_RUNS / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    
    manifest = {
        "slug": slug, "challenge": challenge, "slug_prefix": slug_prefix,
        "started": utc_now_iso(), "agent": agent, "model": model,
        "status": "running", "notes": notes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "results.tsv").write_text(
        "iter\tcommit\tval_score\theadroom\tstatus\tchange_class\tagent\tmodel\trationale\n"
    )
    (run_dir / "stages.tsv").write_text(
        "iter\tstage_val_score\tstage_headroom\tgap\tverdict\tnotes\n"
    )
    print(f"[agent] Created run: {slug}")
    return slug, run_dir


def record_iteration(run_dir, iter_n, params, result):
    """Record one iteration."""
    headroom = result.get("headroom", 0)
    val_score = result.get("val_score", 0)
    
    iter_dir = run_dir / "iterations" / f"iter-{iter_n:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    obs = {
        "ts": utc_now_iso(),
        "run_id": run_dir.name,
        "iter": iter_n,
        "challenge": "dl_sparse_view",
        "change_class": "optimizer",
        "rationale": (f"TV search: lambda={params['tv_lambda']:.5f}, "
                      f"iters={params['tv_iterations']}, lr={params['tv_lr']:.4f}, "
                      f"clip={params['tv_clip_max']:.3f}, decay={params['tv_decay']:.4f}"),
        "val_score": val_score,
        "headroom": headroom,
        "kept": headroom > 0,
        "status": "keep" if headroom > 0 else "discard",
        "params_M": 0.0,
        "train_n": 0,
        "agent": "tv-search",
        "model": "random-search",
        "advice_for_others": (f"TV lambda={params['tv_lambda']:.5f}, lr={params['tv_lr']:.4f}, "
                              f"iters={params['tv_iterations']} -> headroom={headroom:.4f}"),
    }
    
    (iter_dir / "observation.json").write_text(json.dumps(obs, indent=2))
    
    # Append to results
    with (run_dir / "results.tsv").open("a") as f:
        f.write(f"{iter_n}\t\t{val_score:.6g}\t{headroom:.6g}\t"
                f"{'keep' if headroom > 0 else 'discard'}\toptimizer\t"
                f"tv-search\trandom-search\t{obs['rationale'].replace(chr(9), ' ')}\n")
    
    # Append to scratch pad
    scratch = DOCS_RUNS / "observations.jsonl"
    with scratch.open("a") as f:
        f.write(json.dumps(obs) + "\n")
    
    print(f"[agent] Recorded iter {iter_n}: headroom={headroom:.4f}")


def update_index():
    """Update runs-index.json."""
    idx_path = DOCS_RUNS / "runs-index.json"
    runs = []
    for run_dir in sorted(DOCS_RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        m_path = run_dir / "manifest.json"
        if not m_path.exists():
            continue
        manifest = json.loads(m_path.read_text())
        results = run_dir / "results.tsv"
        n_iter = 0
        best_score = None
        best_hr = None
        if results.exists():
            rows = [r for r in results.read_text().splitlines()[1:] if r]
            n_iter = len(rows)
            for r in rows:
                cells = r.split("\t")
                try:
                    v = float(cells[2])
                    if best_score is None or v > best_score:
                        best_score = v
                    h = float(cells[3])
                    if best_hr is None or h > best_hr:
                        best_hr = h
                except (ValueError, IndexError):
                    pass
        
        m_short = run_dir.name.split("-")[-2] + "-" + run_dir.name.split("-")[-1] if "-" in run_dir.name else run_dir.name
        runs.append({
            "slug": run_dir.name, "short_id": m_short,
            "challenge": manifest.get("challenge"),
            "started": manifest.get("started"),
            "status": manifest.get("status", "running"),
            "n_iterations": n_iter,
            "best_score": best_score,
            "best_headroom": best_hr,
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
        })
    
    index = {"schema_version": 1, "updated": utc_now_iso(), "runs": runs}
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"[agent] Updated index with {len(runs)} runs")


# ---------------------------------------------------------------------------
#  Search logic
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "tv_lambda":     (0.0001, 0.01, "log"),
    "tv_iterations": (50, 500, "int"),
    "tv_lr":         (0.001, 0.1, "log"),
    "tv_clip_max":   (0.03, 0.08, "linear"),
    "tv_decay":      (0.0, 0.05, "linear"),
}


def sample_params():
    params = {}
    for key, (lo, hi, mode) in SEARCH_SPACE.items():
        if mode == "log":
            val = math.exp(random.uniform(math.log(lo), math.log(hi)))
        elif mode == "int":
            val = random.randint(lo, hi)
        else:
            val = random.uniform(lo, hi)
        params[key] = val
    return params


def run_tv_solver(params, out_dir):
    """Run TV solver on cluster via python."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config to temp file that solver will read
    config_file = out_dir / "tv_config.json"
    config_file.write_text(json.dumps(params))
    
    cmd = [
        sys.executable,
        str(REPO_ROOT / "pentathlon" / "demo_dl_reference" / "solver_tv_search.py"),
        str(out_dir),
    ]
    
    env = dict(os.environ)
    env["TV_CONFIG_PATH"] = str(config_file)
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[agent] Solver stderr: {result.stderr}", file=sys.stderr)
        return None
    
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[agent] No result.json in {out_dir}")
        return None
    
    return json.loads(result_path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--notes", default="TV hyperparameter random search")
    args = p.parse_args()
    
    slug, run_dir = create_run(
        challenge="dl_sparse_view",
        slug_prefix="demo-dl-tv-search",
        agent="tv-search",
        model="random-search",
        notes=args.notes,
    )
    
    print(f"[agent] Starting TV search: {slug}")
    print(f"[agent] Running {args.iterations} iterations")
    
    best_headroom = 0
    best_params = None
    
    for i in range(1, args.iterations + 1):
        print(f"\n[agent] === Iteration {i}/{args.iterations} ===")
        
        params = sample_params()
        print(f"[agent] Params: {json.dumps(params)}")
        
        out_dir = Path(f"/tmp/tv-search-{slug}-{i:04d}")
        result = run_tv_solver(params, out_dir)
        
        if result is None:
            print(f"[agent] Iteration {i} failed, skipping")
            continue
        
        headroom = result.get("headroom", 0)
        print(f"[agent] Result: headroom={headroom:.4f}, SSIM={result.get('val_score', 0):.4f}")
        
        if headroom > best_headroom:
            best_headroom = headroom
            best_params = params.copy()
            print(f"[agent] *** NEW BEST: headroom={best_headroom:.4f} ***")
        
        record_iteration(run_dir, i, params, result)
    
    update_index()
    
    print(f"\n{'='*60}")
    print(f"[agent] SEARCH COMPLETE")
    print(f"[agent] Best headroom: {best_headroom:.4f}")
    if best_params:
        print(f"[agent] Best params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
