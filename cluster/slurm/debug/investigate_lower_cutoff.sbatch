#!/bin/bash
#SBATCH --job-name=cutoff-sweep
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Numerical SSIM investigation of the lower intensity cutoff (Wagner HD,
# truncation-corrected). See scripts/investigate_lower_cutoff_wagner.py.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm results/mayo_debug
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export STAGED_HELIX2FAN_SUBDIR=staged_helix2fan_v3

echo "[cutoff] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -u scripts/investigate_lower_cutoff_wagner.py
echo "[cutoff] done."
