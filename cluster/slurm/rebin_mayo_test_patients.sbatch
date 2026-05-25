#!/bin/bash
#SBATCH --job-name=rebin-mayo-test-patients
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-rebin the four Wagner-test-split patients whose per-patient
# helix2fan H5s on disk are stale (file-order + pitch bugs from May 20):
#
#   L056 fulldose ~48 MB   L056 lowdose ~56 MB    ← both undersized
#   L058 fulldose ~225 KB  L058 lowdose ~54 MB    ← fulldose failed
#   L075 fulldose ~107 KB  L075 lowdose ~294 KB   ← both failed
#   L123 fulldose ~114 KB  L123 lowdose ~127 MB   ← fulldose failed
#
# Pre-step: rename the stale files to `*_pre_geomfix.h5` so the rebin
# loop sees them as missing (the loop's `if out_h5.exists(): skip`
# guard would otherwise leave them untouched).
#
# Run order: this complements the bulk rebin (SLURM 762097) which is
# only covering L145/186/209/219/277 + L014 lowdose. After both
# complete, run the aggregator (stage_mayo_sinos.sbatch).
#
# Set HELIX2FAN_FFS_DRHO=1 in the env before sbatch if FFS-drho
# correction is desired (consistent with the bulk rebin's choice).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE

echo "[rebin-test] host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
echo "[rebin-test] HELIX2FAN_FFS_DRHO=${HELIX2FAN_FFS_DRHO:-0}"
df -h /cluster/maier | tail -1

STAGED=$REPO/data/mayo_ldct/staged_helix2fan

# Rename stale per-patient files so the rebin loop redoes them.
# Three companion files per (pid, dose): the .h5, the *_z_grid.npy,
# and the *_geometry.json. All get the same `_pre_geomfix` suffix.
rename_companions () {
  local stem="$1"          # e.g. L056_sino_fulldose
  for src in "$STAGED/$stem.h5" \
             "$STAGED/${stem}_z_grid.npy" \
             "$STAGED/${stem}_geometry.json"; do
    [ -f "$src" ] || continue
    local fname; fname=$(basename "$src")
    local dst_fname=${fname/$stem/${stem}_pre_geomfix}
    local dst="$STAGED/$dst_fname"
    if [ ! -f "$dst" ]; then
      mv "$src" "$dst"
      echo "[rebin-test] renamed $fname -> $dst_fname"
    else
      echo "[rebin-test] backup $(basename "$dst") already exists; "\
           "removing redundant stale $fname"
      rm -f "$src"
    fi
  done
}

for pid in L056 L058 L075 L123; do
  for dose in fulldose lowdose; do
    rename_companions "${pid}_sino_${dose}"
  done
done

# The bulk rebin loop iterates ALL Wagner patients; train+val are
# already done, L014 lowdose handled by 762097, and the renames above
# make L056/058/075/123 look missing. The loop will redo them.
python data/fetch_mayo_ldct.py --rebin-only \
    --dv-rebinned 1.0 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}"

echo "[rebin-test] done."
ls -la "$STAGED" | head -40
