#!/bin/bash
#SBATCH --job-name=stage-mayo-v3
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Aggregate per-patient v3-rebinned sinos into per-split train/val/test
# h5s consumed by solvers. Pulls from `staged_helix2fan_v3/` (written by
# rebin_mayo_helix2fan_v3.sbatch) and writes the canonical solver-facing
# h5s at `data/mayo_ldct/staged/{train,val,test}_sino_{fulldose,lowdose}.h5`.
#
# WARNING: this OVERWRITES the previous staged/ h5s (which were aggregated
# from the v2 staged_helix2fan_ssr_fitted/). After this completes, every
# solver that reads `data/mayo_ldct/staged/` automatically uses v3
# geometry on the next epoch. The per-patient v2 inputs survive
# untouched at staged_helix2fan_ssr_fitted/.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
source "$VENV/bin/activate"

export HDF5_USE_FILE_LOCKING=FALSE
export STAGED_HELIX2FAN_SUBDIR=staged_helix2fan_v3

echo "[stage-v3] host=$(hostname) job=$SLURM_JOB_ID"
echo "[stage-v3] STAGED_HELIX2FAN_SUBDIR=$STAGED_HELIX2FAN_SUBDIR"

python data/stage_mayo_sinos.py

echo "[stage-v3] done."
ls -la "$REPO/data/mayo_ldct/staged" | head -40
