#!/bin/bash
#SBATCH --job-name=z-aligned-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme171,lme222
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Z-aligned validation: NO truth interpolation.
# - Pick one GT slice at a time (single DICOM, no z-mixing).
# - Build the FBP slab by physical-overlap weighting of the 1-mm sino
#   slices that intersect the GT's 5-mm physical z range.
# - Compare directly. Sweep 11 consecutive truth slices around the
#   cone-beam centre.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[zalign] host=$(hostname) job=$SLURM_JOB_ID"
python -u scripts/z_aligned_validation.py
echo "[zalign] done."
