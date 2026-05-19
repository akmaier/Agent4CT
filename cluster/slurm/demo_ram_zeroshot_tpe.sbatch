#!/bin/bash
#SBATCH --job-name=demo-ram-zeroshot-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# 20-iter Optuna TPE search around the Claude-agentic winner
# (iter 20: sigma=0.10 factor=0.5 hr=0.5938 SSIM=0.93). RAM is feed-forward
# so each iter is ~15 s; 20 iters ~ 5 min. 1 h walltime is generous.
# Seed trial is the iter-20 winner so TPE starts with a strong prior.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[ram-zeroshot-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python scripts/learned_solver_search_agent.py \
    --solver ram_zeroshot \
    --sampler tpe \
    --iterations 20 \
    --notes "TPE around Claude RAM iter-20 winner (sigma=0.10 factor=0.5)"
echo "[ram-zeroshot-tpe] done."
