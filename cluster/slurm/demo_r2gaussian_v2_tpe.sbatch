#!/bin/bash
#SBATCH --job-name=demo-r2g-v2-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52,lme170
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-run R²-Gaussian TPE on demo-DL with the paper's reported iteration
# budget. The 2026-05-21 search capped gs_n_iter at 800 — ~50× short of
# the 20k–40k the R²-Gaussian paper uses for X3D convergence. Updated
# bounds in learned_solver_search_agent.py: gs_n_iter ∈ [10000, 40000].
# Slug: `demo-intensity-calibrated-tpe-r2gaussian-v2-search-…`.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[demo-r2g-v2-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver r2gaussian \
    --sampler tpe \
    --calibrated \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "R²-Gaussian TPE on demo-DL, extended n_iter bound to [10000, 40000] (paper-reported convergence window)"

echo "[demo-r2g-v2-tpe] done."
