#!/bin/bash
#SBATCH --job-name=rebin-L014-only
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Force re-rebin of L014 fulldose with the post-2026-05-21 nu/nv fix in
# read_dicom_ctpd (Rows = u/fan = 736; Columns = v/cone = 64).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# Disable HDF5 file locking. The cluster's NFS share at /cluster/maier
# returns errno=37 ("No locks available") on h5py.File("w") sync, which
# crashes the final H5 write after a 40-minute rebin. The data is
# written by a single process so no locking is needed.
export HDF5_USE_FILE_LOCKING=FALSE

echo "[rebin-L014] host=$(hostname) job=$SLURM_JOB_ID"
df -h /cluster/maier | tail -1

# Use the existing --rebin-only path; the L014 fulldose H5 was already
# deleted, so only that one will be rebinned (the other 9 patients still
# have their previous-bad H5s and will be skipped).
python data/fetch_mayo_ldct.py --rebin-only --dv-rebinned 1.0 --n-jobs 16

STAGED=$REPO/data/mayo_ldct/staged_helix2fan
ls -la "$STAGED"/L014_sino_fulldose* | head
echo "[rebin-L014] done."
