#!/bin/bash
#SBATCH --job-name=rebin-mayo-ssr-fit
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Bulk helical→fan rebin for the Mayo LDCT Wagner subset, this time
# using the multi-GT-fitted SSR sod/sdd (MAYO_LDCT_SSR_DEFAULTS in
# ddssl_ldct/geometry.py — from SLURM 762369) instead of the DICOM-
# nominal values. Writes to a SIBLING output dir
# `staged_helix2fan_ssr_fitted/` so the existing legacy 86 GB of
# DICOM-nominal staged data stays available for A/B comparison.
#
# Runtime budget: ~10 patients × 2 dose series at ~1 h/series on 16
# cores = ~20 h. The 24h wall is the safety margin.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
source "$VENV/bin/activate"

# Disable HDF5 file locking — NFS share returns errno=37 on h5py.File("w")
# sync. Single-writer; no locks needed.
export HDF5_USE_FILE_LOCKING=FALSE

# Switch the bulk rebin to the multi-GT-fitted SSR sod/sdd
# (sod=593.461, sdd=1086.831). Override is applied per-series in
# data/fetch_mayo_ldct.py:_rebin_patient_series.
export HELIX2FAN_SSR_FITTED=1
# Write to a sibling directory so the legacy DICOM-nominal staged
# data stays intact for comparison.
export STAGED_HELIX2FAN_SUBDIR=staged_helix2fan_ssr_fitted

echo "[rebin] host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
echo "[rebin] HELIX2FAN_SSR_FITTED=$HELIX2FAN_SSR_FITTED"
echo "[rebin] STAGED_HELIX2FAN_SUBDIR=$STAGED_HELIX2FAN_SUBDIR"
df -h /cluster/maier | tail -1

python data/fetch_mayo_ldct.py --rebin-only \
    --dv-rebinned 1.0 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}"

STAGED=/cluster/maier/Agent4CT/data/mayo_ldct/${STAGED_HELIX2FAN_SUBDIR}
echo "[rebin] done."
ls -la "$STAGED" | head -40
du -sh "$STAGED" || true
