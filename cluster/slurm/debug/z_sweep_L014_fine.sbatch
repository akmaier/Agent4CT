#!/bin/bash
#SBATCH --job-name=z-sweep-L014-fine
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# 0.5-mm z-sweep with sub-pixel slab interpolation. Caches integer-slab
# FBPs so each fractional z_off only triggers a single new FBP call.
# Saves a per-z-step image grid (truth | FBP_cal | diff | overlay) into
# results/breast_debug/L014_z_sweep_fine.png.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[zfine] host=$(hostname) job=$SLURM_JOB_ID"
python -u scripts/z_sweep_L014_fine.py
echo "[zfine] done."
