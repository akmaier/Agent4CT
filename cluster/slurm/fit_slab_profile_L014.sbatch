#!/bin/bash
#SBATCH --job-name=fit-slab-profile-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Fit Mayo Siemens slice-sensitivity profile S(z) on L014 by L2-fitting
# 9 slab weights {w_-4, ..., w_+4} shared across 11 consecutive GT
# slices around the cone-beam centre. Softmax param ⇒ Σw=1, w≥0.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[slab] host=$(hostname) job=$SLURM_JOB_ID"
python -u scripts/fit_slab_profile_L014.py
echo "[slab] done."
