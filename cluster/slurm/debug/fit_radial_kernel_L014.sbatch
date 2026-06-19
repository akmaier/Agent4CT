#!/bin/bash
#SBATCH --job-name=fit-radial-kernel-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Fit a rotationally-symmetric, L2-smooth image-space filter that maps
# our Hann FBP onto the Mayo B30f truth at the SSIM-peak GT slice.
# Output: 3 PNGs (4-panel comparison, before/after diff, filter response)
# plus npy of the fitted radial filter H(ρ).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[fit] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true
python -u scripts/fit_radial_kernel_L014.py
echo "[fit] done."
