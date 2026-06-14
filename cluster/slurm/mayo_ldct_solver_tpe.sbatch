#!/bin/bash
#SBATCH --job-name=mayo-ldct-solver-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Mayo Step-3 TPE template. Pass SOLVER key (e.g. learned_primal_dual,
# uswin, dual_domain_supervised, itnet_v3, diffusion_recon_dcstep_unconstrained_mayo_v2).
# Sweeps with full training set (train_n=200, val_n=10 — the entire Wagner-train
# pool) instead of the Step-2 short-budget train_n=50/val_n=5.
#
#   sbatch --export=ALL,SOLVER=learned_primal_dual cluster/slurm/mayo_ldct_solver_tpe.sbatch

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

echo "[mayo-tpe] host=$(hostname) job=$SLURM_JOB_ID solver=$SOLVER"
nvidia-smi -L

# Optuna SQLite on NODE-LOCAL disk: sqlite file-locking over the cluster NFS
# throws "database is locked" intermittently (and leaves 0-byte stale .db).
# The study DB is only needed within this single job; results go to docs/runs
# via the harness (NFS) separately. Copy the DB back to the NFS optuna/ dir at
# the end for the record (best-effort).
OPTUNA_DIR="/tmp/optuna_${SLURM_JOB_ID:-$$}"
mkdir -p "$OPTUNA_DIR"
trap 'cp -f "$OPTUNA_DIR"/*.db "$REPO/optuna/" 2>/dev/null || true; rm -rf "$OPTUNA_DIR"' EXIT

# ITERATIONS=1 => corrected re-score (run only the seeded old-best config on
# the all-slices corrected path: bg_target=truth + det_offset + truncation,
# train_n=579 / val_n=214 via MAYO_CLAMPS). ITERATIONS=20 => full TPE refinement.
python scripts/learned_solver_search_agent.py \
    --solver "$SOLVER" \
    --sampler tpe \
    --calibrated \
    --dataset mayo_ldct_2d \
    --iterations "${ITERATIONS:-20}" \
    --tpe-startup "${TPE_STARTUP:-5}" \
    --tpe-storage "$OPTUNA_DIR" \
    --notes "Mayo-LDCT REBUILD ($SOLVER) corrected all-slices path, iterations=${ITERATIONS:-20}"

echo "[mayo-tpe] done."
