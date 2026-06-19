#!/bin/bash
#SBATCH --job-name=tv-best-image
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
export PYTHONPATH="/cluster/maier/Agent4CT${PYTHONPATH:+:$PYTHONPATH}"

# Run TV solver with BEST parameters to generate comparison image
OUT="$REPO/runs/tv-best-image-${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Write best config from iteration 9
cat > "$OUT/tv_config.json" <<'EOF'
{"tv_lambda": 0.0037234522067927157, "tv_iterations": 408, "tv_lr": 0.04182758725252923, "tv_clip_max": 0.07143070975406873, "tv_decay": 0.0462893960159932, "tv_optimizer": "gd", "tv_init": "fbp"}
EOF

python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT"
echo "[tv-best] done."
