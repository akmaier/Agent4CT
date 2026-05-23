#!/bin/bash
#SBATCH --job-name=demo-lpd-search-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Optuna-TPE search over Learned Primal-Dual hyperparameters on
# breast-CT (intensity-calibrated). Slug will be
# `breast-ct-calibrated-tpe-lpd-search-YYYYMMDD-NN`. Seeded with the
# iter-3 winner from the claude-agentic run (I=10, hidden=64, lr=5e-4,
# ep=20, cosine, grad_clip=1.0, hr=0.829).
#
# Per-iter wall ≈ 25-60 min depending on (I, epochs). 20 iters × ~40 min
# average ≈ 13 h; 14h sbatch wall has 1h margin.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[lpd-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver lpd \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "LPD TPE search on breast-CT, seeded with claude-agentic iter-3 winner (I=10/hidden=64/lr=5e-4/ep=20/cosine, hr=0.83)"

echo "[lpd-tpe] done."
