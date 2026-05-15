#!/usr/bin/env python3
"""Parse TV search stdout and create dashboard entries."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO_ROOT / "docs" / "runs"

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def create_run():
    slug = "demo-dl-tv-search-20260515-01"
    run_dir = DOCS_RUNS / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir(exist_ok=True)
    
    manifest = {
        "slug": slug, "challenge": "dl_sparse_view",
        "slug_prefix": "demo-dl-tv-search",
        "started": "2026-05-15T21:48:00Z",
        "agent": "tv-search", "model": "random-search",
        "status": "done",
        "notes": "20-iteration random search over TV hyperparameters",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "results.tsv").write_text(
        "iter\tcommit\tval_score\theadroom\tstatus\tchange_class\tagent\tmodel\trationale\n"
    )
    return run_dir

def parse_stdout(log_path):
    """Parse the TV search stdout to extract iteration data."""
    text = log_path.read_text()
    
    iterations = []
    i = 1
    for block in text.split("[agent] === Iteration")[1:]:
        # Extract params
        params_match = re.search(r'Params: ({.*?})', block)
        if not params_match:
            continue
        params = json.loads(params_match.group(1))
        
        # Extract result
        result_match = re.search(r'Result: headroom=([\d.]+), SSIM=([\d.]+)', block)
        if not result_match:
            continue
        headroom = float(result_match.group(1))
        ssim = float(result_match.group(2))
        
        iterations.append({
            "iter": i,
            "params": params,
            "headroom": headroom,
            "ssim": ssim,
        })
        i += 1
    
    return iterations

def record_iteration(run_dir, data):
    iter_dir = run_dir / "iterations" / f"iter-{data['iter']:04d}"
    iter_dir.mkdir(exist_ok=True)
    
    obs = {
        "ts": utc_now(),
        "run_id": run_dir.name,
        "iter": data["iter"],
        "challenge": "dl_sparse_view",
        "change_class": "optimizer",
        "rationale": (f"TV search iter {data['iter']}: lambda={data['params']['tv_lambda']:.6f}, "
                      f"iters={data['params']['tv_iterations']}, lr={data['params']['tv_lr']:.6f}"
                      f", clip={data['params']['tv_clip_max']:.4f}, decay={data['params']['tv_decay']:.4f}"),
        "val_score": data["ssim"],
        "headroom": data["headroom"],
        "kept": data["headroom"] > 0,
        "status": "keep" if data["headroom"] > 0 else "discard",
        "params_M": 0.0,
        "train_n": 0,
        "agent": "tv-search",
        "model": "random-search",
        "advice_for_others": f"TV lambda={data['params']['tv_lambda']:.6f}, lr={data['params']['tv_lr']:.6f} -> headroom={data['headroom']:.4f}",
    }
    
    (iter_dir / "observation.json").write_text(json.dumps(obs, indent=2))
    
    with (run_dir / "results.tsv").open("a") as f:
        f.write(f"{data['iter']}\t\t{data['ssim']:.6g}\t{data['headroom']:.6g}\t"
                f"keep\toptimizer\ttv-search\trandom-search\t{obs['rationale'].replace(chr(9), ' ')}\n")
    
    scratch = DOCS_RUNS / "observations.jsonl"
    with scratch.open("a") as f:
        f.write(json.dumps(obs) + "\n")
    
    print(f"Recorded iter {data['iter']}: headroom={data['headroom']:.4f}")

def update_index():
    idx_path = DOCS_RUNS / "runs-index.json"
    index = json.loads(idx_path.read_text())
    
    # Add or update TV search run
    found = False
    for run in index["runs"]:
        if run["slug"] == "demo-dl-tv-search-20260515-01":
            found = True
            break
    
    if not found:
        index["runs"].append({
            "slug": "demo-dl-tv-search-20260515-01",
            "short_id": "20260515-01",
            "challenge": "dl_sparse_view",
            "started": "2026-05-15T21:48:00Z",
            "status": "done",
            "n_iterations": 20,
            "best_score": 0.4691,
            "best_headroom": 0.6033,
            "agent": "tv-search",
            "model": "random-search",
        })
    
    index["updated"] = utc_now()
    idx_path.write_text(json.dumps(index, indent=2))

def main():
    log_path = Path("/tmp/tv-search-760946.log")
    
    # Try to fetch from cluster
    import subprocess
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "maier@cluster.i5.informatik.uni-erlangen.de",
         "cat /cluster/maier/Agent4CT/results/slurm/demo-dl-tv-search-760946.out"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log_path.write_text(result.stdout)
        print("Fetched stdout from cluster")
    else:
        print("Could not fetch stdout")
        return
    
    run_dir = create_run()
    iterations = parse_stdout(log_path)
    
    print(f"Parsed {len(iterations)} iterations")
    
    best = max(iterations, key=lambda x: x["headroom"])
    print(f"\nBest iteration: {best['iter']}")
    print(f"  Headroom: {best['headroom']:.4f}")
    print(f"  SSIM: {best['ssim']:.4f}")
    print(f"  Params: {json.dumps(best['params'])}")
    
    for data in iterations:
        record_iteration(run_dir, data)
    
    update_index()
    print(f"\nDone! Run: demo-dl-tv-search-20260515-01")

if __name__ == "__main__":
    main()
