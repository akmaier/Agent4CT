#!/bin/bash
#SBATCH --job-name=fit-radial-warp-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Test the curved→flat detector rebinning for a radial fisheye effect.
# Fit a polynomial radial scale s(r) = 1 + c1·(r/rmax)² + c2·(r/rmax)⁴
# + c3·(r/rmax)⁶ on FBP_cal by L2 minimisation to truth (peak GT slice).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[warp] host=$(hostname) job=$SLURM_JOB_ID"
python -u scripts/fit_radial_warp_L014.py
echo "[warp] done."
