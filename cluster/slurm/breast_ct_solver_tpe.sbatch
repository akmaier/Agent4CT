#!/bin/bash
#SBATCH --job-name=breast-ct-solver-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Generic Optuna-TPE search wrapper for a breast-CT (intensity-calibrated)
# solver TPE pass. Pass the SOLVERS-dict key in `SOLVER`:
#
#   sbatch --export=ALL,SOLVER=dual_domain_supervised cluster/slurm/breast_ct_solver_tpe.sbatch
#   sbatch --export=ALL,SOLVER=dual_domain_bilateral_supervised cluster/slurm/breast_ct_solver_tpe.sbatch
#
# Slug will be `breast-ct-calibrated-tpe-<solver>-search-YYYYMMDD-NN`,
# joining the existing dashboard chart group.

set -euo pipefail
: "${SOLVER:?need SOLVER (key from scripts/learned_solver_search_agent.py SOLVERS dict)}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[breast-ct-tpe] host=$(hostname) job=$SLURM_JOB_ID solver=$SOLVER"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver "$SOLVER" \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "breast-CT TPE search for $SOLVER, seeded with agentic winner"

echo "[breast-ct-tpe] done."
