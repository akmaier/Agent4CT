#!/bin/bash
#SBATCH --job-name=rebin-mayo-z-aligned
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Bulk re-rebin with z_start aligned to the Mayo truth DICOM z grid.
#
# Why: Mayo truth DICOMs sit at 3 mm spacing in patient frame with a
# fractional z offset that's tied to the table position at acquisition
# (e.g. L014 truth source-frame z's are 23.55 + 3k mm). Our default
# rebin uses z_start = z_min + 0.5*pitch which yields a 1 mm sino grid
# whose fractional part doesn't match the truth grid (off by ~0.31 mm
# for L014). That 0.31 mm mismatch forced sub-pixel interpolation in
# the validator on BOTH the truth side and the FBP side — a double
# blur the user (rightly) flagged as nonsensical.
#
# With HELIX2FAN_Z_ALIGN=1, the rebin shifts z_start so that every
# dv_rebinned-th (= every 1 mm) sino slice lands exactly on a truth
# slice. Output filenames carry the `_aligned` suffix so the unaligned
# legacy outputs stay around for side-by-side comparison.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# NFS at /cluster/maier rejects fcntl locks (errno 37); disable HDF5 locking.
export HDF5_USE_FILE_LOCKING=FALSE

# Toggle truth-grid alignment.
export HELIX2FAN_Z_ALIGN=1

echo "[rebin-aligned] host=$(hostname) job=$SLURM_JOB_ID"
echo "[rebin-aligned] HELIX2FAN_Z_ALIGN=$HELIX2FAN_Z_ALIGN"
df -h /cluster/maier | tail -1

python -u data/fetch_mayo_ldct.py --rebin-only --dv-rebinned 1.0 --n-jobs 16

STAGED=$REPO/data/mayo_ldct/staged_helix2fan
ls -la "$STAGED"/*_aligned.* | head -40
echo "[rebin-aligned] done."
