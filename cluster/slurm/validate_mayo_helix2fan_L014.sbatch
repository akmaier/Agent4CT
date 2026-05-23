#!/bin/bash
#SBATCH --job-name=validate-mayo-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Track A.3 of docs/workplan_real_datasets.md.
# FBP-vs-truth validation of L014 (the cleanest fully-rebinned patient).
# Wagner geometry already baked into FanBeamGeometry defaults. Target
# SSIM >= 0.85; if it lands below, iterate on geometry knobs in helix2fan.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[validate] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true
mkdir -p "$REPO/results/breast_debug"
python scripts/validate_mayo_helix2fan.py --patient L014 --dose fulldose \
    --out-png "$REPO/results/breast_debug/validate_L014_fulldose.png"
ls -la "$REPO/results/breast_debug/validate_L014_fulldose.png"
echo "[validate] done."
