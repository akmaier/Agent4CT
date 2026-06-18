#!/bin/bash
#SBATCH --job-name=restage-verify
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme221
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# FBP(HD) + FBP(LD) vs GT verification of the re-staged staged_canonical, at each
# volume's reconstruction FOV. Renders docs/_restage_verify/<patient>_fbp_vs_gt.png.
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

echo "[verify] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python scripts/verify_restage_fbp.py
echo "[verify] done."
