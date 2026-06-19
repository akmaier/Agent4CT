#!/bin/bash
#SBATCH --job-name=n2i-smoke
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme221
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Smoke test for the per-image N2I rewrite. Tiny budget; verifies the path
# runs end-to-end and produces a finite val_score + comparison.png before any
# destructive cleanup / full restart.
set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export AGENT4CT_DATASET=mayo_ldct_2d

echo "[n2i-smoke] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python cluster/slurm/_n2i_smoke_driver.py
echo "[n2i-smoke] done."
