#!/bin/bash
#SBATCH --job-name=ablation-pixel-spacing
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Apples-to-apples ablation: same L014 sino, same +3.5 mm slab anchor,
# same truth slice (z-interpolated to -256.86 mm), but vary the FBP
# FanBeamGeometry pixel_spacing between 0.7 (Wagner default) and
# 0.703125 (Mayo's actual). Produces a 2-row figure with SSIM/PSNR/RMSE
# headlines plus an overlay panel showing pixel-level alignment.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[ablation] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true
python scripts/compare_pixel_spacing_ablation.py
echo "[ablation] done."
