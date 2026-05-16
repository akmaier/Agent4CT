#!/bin/bash
#SBATCH --job-name=dd-search
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=02:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out

set -euo pipefail
cd /cluster/maier/Agent4CT
source .venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT${PYTHONPATH:+:$PYTHONPATH}"

# Dual-domain parameter search (20 iterations, ~5 min each = ~100 min total)
python scripts/dd_and_itnet_search.py \
    --solver dual_domain \
    --iterations 20 \
    --train-n 200 \
    --notes "Dual-domain denoising parameter search, train_n=200, 5min budget"

echo "[dd-search] done."
