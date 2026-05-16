#!/bin/bash
#SBATCH --job-name=bf-search
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out

set -euo pipefail
cd /cluster/maier/Agent4CT
source .venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT${PYTHONPATH:+:$PYTHONPATH}"

# Bilateral filter parameter search (20 iterations, ~5 min each = ~100 min total)
python scripts/bf_search_agent.py \
    --iterations 20 \
    --notes "Dual-domain bilateral filter hyperparameter search, 20 iterations"

echo "[bf-search] done."
