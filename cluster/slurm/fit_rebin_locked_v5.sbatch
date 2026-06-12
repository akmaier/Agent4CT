#!/bin/bash
#SBATCH --job-name=fit-locked-v5
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# v5 — transfer test of the v4b projection-domain geometry through the
# full rebin+FBP stack with only nuisances free. See
# scripts/fit_rebin_locked_v5.py and findings.md 2026-06-12 v4 entry.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[v5] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -u scripts/fit_rebin_locked_v5.py
echo "[v5] done."
