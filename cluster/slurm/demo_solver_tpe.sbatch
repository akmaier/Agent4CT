#!/bin/bash
#SBATCH --job-name=demo-dl-solver-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Generic Optuna-TPE wrapper for demo-DL (calibrated metric). Pass the
# SOLVERS-dict key in `SOLVER`:
#
#   sbatch --export=ALL,SOLVER=itnet                     cluster/slurm/demo_solver_tpe.sbatch
#   sbatch --export=ALL,SOLVER=tv_iterative_supervised   cluster/slurm/demo_solver_tpe.sbatch
#
# Slug will be `demo-intensity-calibrated-tpe-<solver>-search-YYYYMMDD-NN`,
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

echo "[demo-dl-tpe] host=$(hostname) job=$SLURM_JOB_ID solver=$SOLVER"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver "$SOLVER" \
    --sampler tpe \
    --calibrated \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "demo-DL TPE search for $SOLVER, seeded with best-known config"

echo "[demo-dl-tpe] done."
