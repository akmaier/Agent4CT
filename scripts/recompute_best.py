#!/usr/bin/env python3
"""Recompute final (best) iterations with corrected SSIM + image."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO / "docs" / "runs"
CLUSTER_RUNS = Path("/cluster/maier/Agent4CT/runs")


def recompute_best(run_name: str, solver_name: str, iter_n: int, **override):
    """Run one solver with existing config + corrected C1=C2=0 SSIM."""
    iter_str = f"iter-{iter_n:04d}"
    run_dir_local = DOCS_RUNS / run_name
    run_dir_cluster = CLUSTER_RUNS / f"{run_name}-recompute-{iter_str}"
    run_dir_cluster.mkdir(parents=True, exist_ok=True)

    # Load existing config from saved result.json
    result_file = run_dir_local / "iterations" / iter_str / "result.json"
    if not result_file.exists():
        print(f"[recompute] MISSING result.json for {run_name}/{iter_str}")
        return False

    # Read existing result to get original config + params
    data = json.loads(result_file.read_text())
    cfg = data.get("config", {})
    cfg["seed"] = 42  # ensure deterministic
    cfg["val_n"] = 100

    # Determine solver path
    solvers = {
        "tv": REPO / "pentathlon" / "demo_dl_reference" / "solver_tv_search.py",
        "dual_domain": REPO / "pentathlon" / "demo_dl_reference" / "solver_dual_ddomain_n2i.py",
        "itnet": REPO / "pentathlon" / "demo_dl_reference" / "solver_itnet_v2.py",
    }
    solver_path = solvers.get(solver_name)
    if solver_path is None:
        print(f"[recompute] Unknown solver '{solver_name}'")
        return False

    # Convert solver path to cluster path
    cluster_repo = Path("/cluster/maier/Agent4CT")
    solver_path_cluster = cluster_repo / solver_path.relative_to(REPO)

    # Pick the right env var + write config into run_dir_cluster
    if solver_name == "tv":
        env_key = "TV_CONFIG_PATH"
    elif solver_name == "dual_domain":
        env_key = "DD_CONFIG_PATH"
    else:
        env_key = "ITNET_CONFIG_PATH"

    config_file = run_dir_cluster / "config.json"
    config_file.write_text(json.dumps(cfg))
    out_dir = run_dir_cluster

    print(f"[recompute] Recomputing {run_name}/{iter_str} on cluster ...")

    # Build sbatch script that runs the solver with the new code
    sbatch = out_dir / "recompute.sbatch"
    sbatch.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=ssim-recomp-{run_name}-{iter_str}
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output={out_dir}/slurm.out

set -euo pipefail
source /cluster/maier/Agent4CT/.venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT:$PYTHONPATH"

export {env_key}={config_file}
cd /cluster/maier/Agent4CT
python3 {solver_path_cluster} {out_dir}
echo "[recompute] Done."
"""
    )

    # Submit
    cmd = ["ssh", "-o", "BatchMode=yes", "maier@cluster.i5.informatik.uni-erlangen.de",
           f"cd /cluster/maier/Agent4CT && sbatch {sbatch}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[recompute] sbatch failed: {result.stderr}")
        return False
    job_id = result.stdout.strip().split()[-1] if "Submitted batch job" in result.stdout else "?"
    print(f"[recompute] Submitted job {job_id}")
    return job_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: recompute_best.py <run> <solver>")
        sys.exit(1)
    run = sys.argv[1]
    solver = sys.argv[2]
    # Determine best iteration from local results
    res_tsv = DOCS_RUNS / run / "results.tsv"
    best_ssim = -1
    best_iter = None
    if res_tsv.exists():
        for line in res_tsv.read_text().splitlines()[1:]:
            if not line:
                continue
            parts = line.split("\t")
            try:
                ssim = float(parts[2])
                if ssim > best_ssim:
                    best_ssim = ssim
                    best_iter = int(parts[0])
            except ValueError:
                continue
    if best_iter is None:
        print("[recompute] Could not determine best iteration")
        sys.exit(1)
    print(f"[recompute] Best iteration for {run}: {best_iter}")
    recompute_best(run, solver, best_iter)
