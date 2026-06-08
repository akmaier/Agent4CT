#!/bin/bash
#SBATCH --job-name=mayo-ldct-tpe-v2
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Mayo Step-3 TPE v2 (2026-06-08) — two fixes over v1:
#
#   1. Local /tmp Optuna storage to dodge the NFS-SQLite-lock issue that
#      crashed every v1 dispatch.
#   2. Mayo-specific search-space clamps live in
#      scripts/learned_solver_search_agent.py (commit bf62480a). They
#      activate automatically with --dataset=mayo_ldct_2d.
#
# Usage:
#   sbatch --export=ALL,SOLVER=lpd cluster/slurm/mayo_ldct_solver_tpe_v2.sbatch
#   sbatch --export=ALL,SOLVER=uswin cluster/slurm/mayo_ldct_solver_tpe_v2.sbatch
#   sbatch --export=ALL,SOLVER=dual_domain_supervised cluster/slurm/mayo_ldct_solver_tpe_v2.sbatch
#   sbatch --export=ALL,SOLVER=itnet_v3 cluster/slurm/mayo_ldct_solver_tpe_v2.sbatch

set -euo pipefail
: "${SOLVER:?need SOLVER}"
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
TPE_STORE=/tmp/optuna-$SLURM_JOB_ID
mkdir -p "$TPE_STORE"
echo "[mayo-tpe-v2] storage=$TPE_STORE  solver=$SOLVER"
nvidia-smi -L
python scripts/learned_solver_search_agent.py \
    --solver "$SOLVER" --sampler tpe --calibrated \
    --dataset mayo_ldct_2d --iterations 20 --tpe-startup 5 \
    --tpe-storage "$TPE_STORE" \
    --notes "Mayo Step-3 TPE for $SOLVER (clamped search space + local /tmp storage)"
mkdir -p "$REPO/optuna-local-backups"
cp "$TPE_STORE"/*.db "$REPO/optuna-local-backups/" 2>/dev/null || true
