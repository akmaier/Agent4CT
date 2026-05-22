"""Unified parameter search for Dual-Domain and ItNet solvers.

Usage:
    # Dual-domain search (20 iterations)
    python scripts/dd_and_itnet_search.py --solver dual_domain --iterations 20 --train-n 200
    
    # ItNet search (20 iterations)
    python scripts/dd_and_itnet_search.py --solver itnet --iterations 20 --train-n 200
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

def sample_params(search_space):
    params = {}
    for key, spec in search_space.items():
        if isinstance(spec, tuple):
            lo, hi, mode = spec
            if mode == "log":
                val = math.exp(random.uniform(math.log(lo), math.log(hi)))
            elif mode == "int":
                val = random.randint(lo, hi)
            elif mode == "choice":
                val = random.choice(hi)
            else:  # linear
                val = random.uniform(lo, hi)
        elif isinstance(spec, list):
            val = random.choice(spec)
        else:
            val = spec
        params[key] = val
    return params

def run_solver(solver_name, params, out_dir, train_n=200):
    """Run solver with given params."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = out_dir / "config.json"
    config_file.write_text(json.dumps(params))
    
    if solver_name == "dual_domain":
        solver_path = REPO_ROOT / "pentathlon" / "demo_dl_reference" / "solver_dual_domain_denoising.py"
        env_var = "DD_CONFIG_PATH"
        if not solver_path.exists():
            # Try alternative name
            solver_path = REPO_ROOT / "pentathlon" / "demo_dl_reference" / "solver_dual_ddomain_n2i.py"
    elif solver_name == "itnet":
        solver_path = REPO_ROOT / "pentathlon" / "demo_dl_reference" / "solver_itnet_v2.py"
        env_var = "ITNET_CONFIG_PATH"
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    
    cmd = [
        sys.executable,
        str(solver_path),
        str(out_dir),
    ]
    
    env = dict(os.environ)
    env[env_var] = str(config_file)
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=360)
    if result.returncode != 0:
        print(f"[agent] Solver stderr: {result.stderr}", file=sys.stderr)
        return None
    
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[agent] No result.json in {out_dir}")
        return None
    
    return json.loads(result_path.read_text())

def record_iteration(run_dir, iter_n, params, result, solver_name, out_dir=None):
    """Record one iteration."""
    headroom = result.get("headroom", 0)
    val_score = result.get("val_score", 0)
    
    iter_dir = run_dir / "iterations" / f"iter-{iter_n:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_image = None
    if out_dir is not None:
        src_img = Path(out_dir) / "comparison.png"
        if src_img.exists():
            dst_img = iter_dir / "comparison.png"
            shutil.copy2(src_img, dst_img)
            comparison_image = f"runs/{run_dir.name}/iterations/iter-{iter_n:04d}/comparison.png"
            print(f"[agent] Copied comparison image to {dst_img}")
    
    param_str = ", ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
    
    obs = {
        "ts": utc_now_iso(),
        "run_id": run_dir.name,
        "iter": iter_n,
        "challenge": "dl_sparse_view",
        "change_class": "architecture",
        "rationale": f"{solver_name} search: {param_str}",
        "val_score": val_score,
        "headroom": headroom,
        "kept": headroom > 0,
        "status": "keep" if headroom > 0 else "discard",
        "params_M": result.get("params_M", 0),
        "train_n": result.get("train_n", 0),
        "agent": f"{solver_name}-search",
        "model": "random-search",
        "advice_for_others": f"{solver_name} params: {param_str} -> headroom={headroom:.4f}",
    }
    if comparison_image:
        obs["comparison_image"] = comparison_image
    
    (iter_dir / "observation.json").write_text(json.dumps(obs, indent=2))
    
    with (run_dir / "results.tsv").open("a") as f:
        f.write(f"{iter_n}\t\t{val_score:.6g}\t{headroom:.6g}\t"
                f"{'keep' if headroom > 0 else 'discard'}\tarchitecture\t"
                f"{solver_name}-search\trandom-search\t{obs['rationale'].replace(chr(9), ' ')}\n")
    
    scratch = DOCS_RUNS / "observations.jsonl"
    with scratch.open("a") as f:
        f.write(json.dumps(obs) + "\n")
    
    print(f"[agent] Recorded iter {iter_n}: headroom={headroom:.4f}")

def update_index():
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

# Search spaces
DD_SEARCH_SPACE = {
    "epochs": (3, 8, "int"),
    "lr": (1e-4, 5e-3, "log"),
    "batch_size": (1, 4, "int"),
    "unet_c": (8, 24, "int"),
}

ITNET_SEARCH_SPACE = {
    "pretrain_epochs": (3, 8, "int"),
    "pretrain_lr": (1e-4, 5e-3, "log"),
    "itnet_k": (3, 8, "int"),
    "unet_c": (8, 24, "int"),
    "itnet_alpha_init": (0.001, 0.05, "log"),
    "residual_learning": [True, False],
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", required=True, choices=["dual_domain", "itnet"])
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--train-n", type=int, default=200)
    p.add_argument("--notes", default="")
    p.add_argument("--out-base", default="/cluster/maier/Agent4CT/runs")
    args = p.parse_args()
    
    solver_name = args.solver
    search_space = DD_SEARCH_SPACE if solver_name == "dual_domain" else ITNET_SEARCH_SPACE
    
    slug_prefix = f"demo-dl-{solver_name}"
    _, run_dir = create_run(
        challenge="dl_sparse_view",
        slug_prefix=slug_prefix,
        agent=f"{solver_name}-search",
        model="random-search",
        notes=args.notes or f"{solver_name} hyperparameter search, train_n={args.train_n}",
    )
    
    # If running on cluster, create output dir there too
    cluster_out = Path(args.out_base) / run_dir.name
    cluster_out.mkdir(parents=True, exist_ok=True)
    
    print(f"[agent] Starting {solver_name} search: {run_dir.name}")
    print(f"[agent] Running {args.iterations} iterations")
    print(f"[agent] Search space: {list(search_space.keys())}")
    
    best_headroom = 0
    best_params = None
    
    for i in range(1, args.iterations + 1):
        print(f"\n[agent] === Iteration {i}/{args.iterations} ===")
        
        params = sample_params(search_space)
        # Ensure train_n is set
        params["train_n"] = args.train_n
        print(f"[agent] Params: {json.dumps(params, default=str)}")
        
        out_dir = cluster_out / f"iter-{i:04d}"
        out_dir.mkdir(exist_ok=True)
        
        try:
            result = run_solver(solver_name, params, out_dir)
        except subprocess.TimeoutExpired:
            print(f"[agent] Iteration {i} timed out (>6min), skipping")
            continue
        
        if result is None:
            print(f"[agent] Iteration {i} failed, skipping")
            continue
        
        headroom = result.get("headroom", 0)
        print(f"[agent] Result: headroom={headroom:.4f}, SSIM={result.get('val_score', 0):.4f}")
        
        if headroom > best_headroom:
            best_headroom = headroom
            best_params = params.copy()
            print(f"[agent] *** NEW BEST: headroom={best_headroom:.4f} ***")
        
        record_iteration(run_dir, i, params, result, solver_name, out_dir=out_dir)
    
    update_index()
    
    # Update manifest to done
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["status"] = "done"
    manifest["ended"] = utc_now_iso()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    print(f"\n{'='*60}")
    print(f"[agent] SEARCH COMPLETE: {run_dir.name}")
    print(f"[agent] Best headroom: {best_headroom:.4f}")
    if best_params:
        print(f"[agent] Best params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
