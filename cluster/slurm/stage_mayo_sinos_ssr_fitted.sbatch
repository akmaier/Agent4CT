#!/bin/bash
#SBATCH --job-name=stage-mayo-sinos-ssr
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Aggregate per-patient SSR-fitted sinos into per-split H5s for the
# autoresearch dispatchers. Reads from staged_helix2fan_ssr_fitted/
# (bulk rebin SLURM 762413 + L123 lowdose continuation 762630).
# Writes data/mayo_ldct/staged/{train,val,test}_sino_{full,low}dose.h5
# (alongside the existing *_truth.h5 from stage_h5).
#
# This is CPU-only (HDF5 I/O + slice index lookups via pydicom);
# no GPU needed.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

# Use the SSR-fitted bulk rebin output as input (multi-GT-fitted SSR
# sod/sdd in the helical→fan step). Default is "staged_helix2fan/"
# (legacy DICOM-nominal).
export STAGED_HELIX2FAN_SUBDIR=staged_helix2fan_ssr_fitted

echo "[stage-sino] host=$(hostname) job=$SLURM_JOB_ID"
echo "[stage-sino] STAGED_HELIX2FAN_SUBDIR=$STAGED_HELIX2FAN_SUBDIR"

python data/stage_mayo_sinos.py --dose both --force

STAGED=/cluster/maier/Agent4CT/data/mayo_ldct/staged
echo "[stage-sino] done."
ls -la "$STAGED" | head -20
