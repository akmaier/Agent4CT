"""Generate comparison images for TV search iterations retroactively.

Since the temp dirs were cleaned up, we re-run the TV solver for the top iterations
with the exact same parameters to regenerate comparison.png files.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO / "docs" / "runs"

# Top iterations to regenerate images for (we'll do all 5 that had good results)
TOP_ITERS = [1, 5, 8, 9, 16, 17]  # These had headroom > 0.35

def get_iter_params(run_dir, iter_n):
    """Extract params from observation.json."""
    obs_path = run_dir / "iterations" / f"iter-{iter_n:04d}" / "observation.json"
    obs = json.loads(obs_path.read_text())
    
    # Parse params from rationale
    rationale = obs["rationale"]
    params = {}
    for part in rationale.split(", "):
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip().replace("tv_", "", 1)
            if key in ["lambda", "lr", "clip", "decay"]:
                params[f"tv_{key}"] = float(val)
            elif key == "iters":
                params["tv_iterations"] = int(float(val))
    
    return params, obs["headroom"], obs["val_score"]


def submit_image_job(iter_n, params, headroom):
    """Submit a quick Slurm job to generate comparison image."""
    # Create temp sbatch
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=tv-img-{iter_n:02d}
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out

set -euo pipefail
cd /cluster/maier/Agent4CT
source .venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT${{PYTHONPATH:+:$PYTHONPATH}}"

OUT="/cluster/maier/Agent4CT/runs/tv-img-{iter_n:02d}-${{SLURM_JOB_ID}}"
mkdir -p "$OUT"

cat > "$OUT/tv_config.json" <<'EOF'
{json.dumps(params)}
EOF

python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT"

echo "Image generated at: $OUT/comparison.png"
"""
    
    script_path = f"/tmp/tv_img_{iter_n}.sbatch"
    with open(script_path, "w") as f:
        f.write(sbatch_content)
    
    # Submit
    result = subprocess.run(
        ["scp", script_path, "maier@cluster.i5.informatik.uni-erlangen.de:/tmp/"],
        capture_output=True, text=True
    )
    
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "maier@cluster.i5.informatik.uni-erlangen.de",
         f"cd /cluster/maier/Agent4CT && sbatch /tmp/tv_img_{iter_n}.sbatch | awk '{{print $4}}'"],
        capture_output=True, text=True
    )
    
    job_id = result.stdout.strip()
    print(f"Submitted iter {iter_n} (headroom={headroom:.4f}) -> Job {job_id}")
    return job_id


def main():
    run_dir = DOCS_RUNS / "demo-dl-tv-search-20260515-01"
    
    print("TV Image Regeneration")
    print(f"Will regenerate images for top iterations")
    print("")
    
    jobs = []
    for iter_n in TOP_ITERS:
        params, headroom, ssim = get_iter_params(run_dir, iter_n)
        print(f"Iter {iter_n}: headroom={headroom:.4f}, SSIM={ssim:.4f}")
        print(f"  Params: {json.dumps(params)}")
        job_id = submit_image_job(iter_n, params, headroom)
        jobs.append((iter_n, job_id))
    
    print(f"\nSubmitted {len(jobs)} jobs:")
    for iter_n, job_id in jobs:
        print(f"  Iter {iter_n}: Job {job_id}")
    
    print(f"\nCopy images back with:")
    print(f"  scp maier@cluster:/cluster/maier/Agent4CT/runs/tv-img-XX-JOBID/comparison.png \\")
    print(f"    docs/runs/demo-dl-tv-search-20260515-01/iterations/iter-XXXX/comparison.png")


if __name__ == "__main__":
    main()
