#!/bin/bash
#SBATCH --job-name=fit-fwd-geom-v4
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# v4 — single-physical-geometry identification by projection-domain
# forward matching on L014. See scripts/fit_geometry_forward_L014_v4.py
# and the 2026-06-12 v4 entry in docs/findings.md.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[v4] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -u scripts/fit_geometry_forward_L014_v4.py
echo "[v4] done."
