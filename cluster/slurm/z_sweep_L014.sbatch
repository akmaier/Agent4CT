#!/bin/bash
#SBATCH --job-name=z-sweep-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Fine z-sweep on L014 to find the SSIM peak with the corrected validator
# (grid-aligned, no-FOV, sign-flip clean). Also reports body centroid row
# in truth vs FBP to detect any consistent vertical shift.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[z-sweep] host=$(hostname) job=$SLURM_JOB_ID"
# `-u` disables stdout buffering so we see progress live in the SLURM
# out file. The previous run buffered everything and was cancelled at
# the 25-min time limit without showing any intermediate values.
python -u scripts/z_sweep_L014.py
echo "[z-sweep] done."
