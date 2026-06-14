#!/bin/bash
#SBATCH --job-name=cmp-allslices
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# All-slices HD vs LD FBP baseline across all 10 Wagner patients.
# ~155 full-dose truth-image slices/patient x 2 doses, single trunc-
# corrected FBP per slab (sino-domain slab average; FBP is linear).
# Establishes oracle (HD FBP) + baseline (LD FBP) for the rebuilt Mayo
# leaderboard. v3 SSR sinos + Powell FBP geom + det_offset + truncation
# + bg_target=truth. See scripts/compare_hd_ld_fbp_allslices.py.

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

echo "[allslices] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
EXTRA=""
[ -n "${PATIENTS:-}" ] && EXTRA="--patients ${PATIENTS}"
[ -n "${OUT_DIR:-}" ] && EXTRA="$EXTRA --out-dir ${OUT_DIR}"
python -u scripts/compare_hd_ld_fbp_allslices.py --tag "${TAG:-v3allslices}" --stride "${STRIDE:-1}" $EXTRA
echo "[allslices] done."
