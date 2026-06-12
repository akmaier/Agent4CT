#!/bin/bash
#SBATCH --job-name=rebin-mayo-v3
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Bulk helical→fan rebin for the Mayo LDCT Wagner subset using the
# v3-fitted SSR geometry (SLURM 763384, 2026-06-12) — adds `s_z`
# (z-axis scaling = 1.001665) on top of the prior multi-GT fit and
# corrects sod/sdd/Δz/post-FBP accordingly. See
# MAYO_LDCT_SSR_DEFAULTS in ddssl_ldct/geometry.py and the 2026-06-12
# entry in docs/findings.md.
#
# Output dir: `staged_helix2fan_v3/`. Sibling to the legacy
# `staged_helix2fan/` (DICOM-nominal) and `staged_helix2fan_ssr_fitted/`
# (v2 multi-GT 762369) directories — those stay intact for rollback /
# A-B comparison.
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

# Switch the bulk rebin to the v3-fitted SSR sod / sdd / s_z. Override
# is applied per-series in data/fetch_mayo_ldct.py:_rebin_patient_series.
export HELIX2FAN_SSR_FITTED=1
# Write to a v3-versioned sibling directory.
export STAGED_HELIX2FAN_SUBDIR=staged_helix2fan_v3

echo "[rebin-v3] host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
echo "[rebin-v3] HELIX2FAN_SSR_FITTED=$HELIX2FAN_SSR_FITTED"
echo "[rebin-v3] STAGED_HELIX2FAN_SUBDIR=$STAGED_HELIX2FAN_SUBDIR"
df -h /cluster/maier | tail -1

python data/fetch_mayo_ldct.py --rebin-only \
    --dv-rebinned 1.0 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}"

STAGED=/cluster/maier/Agent4CT/data/mayo_ldct/${STAGED_HELIX2FAN_SUBDIR}
echo "[rebin-v3] done."
ls -la "$STAGED" | head -40
du -sh "$STAGED" || true
