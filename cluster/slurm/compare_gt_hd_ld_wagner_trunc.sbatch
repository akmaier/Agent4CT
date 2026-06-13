#!/bin/bash
#SBATCH --job-name=cmp-wagner-trunc
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# GT vs HD/LD FBP across all 10 Wagner patients WITH water-cylinder
# truncation correction + difference images. Reconstructs raw + trunc-
# corrected arms per dose -> ~2x the FBP of the plain compare, hence the
# 2h wall. Reads per-patient v3 sinos from staged_helix2fan_v3/.
# See scripts/compare_gt_hd_ld_fbp_wagner_trunc.py.

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

echo "[cmp-trunc] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -u scripts/compare_gt_hd_ld_fbp_wagner_trunc.py --tag "${TRUNC_TAG:-v3trunc}" --pad 384
echo "[cmp-trunc] done."
