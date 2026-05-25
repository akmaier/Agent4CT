#!/bin/bash
#SBATCH --job-name=rebin-L014-ffs-drho
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-rebin L014 fulldose with the FFS-drho (radial flying-focal-spot)
# correction enabled. Outputs to L014_sino_fulldose_ffs_drho.h5 so the
# baseline (no-FFS-drho) L014_sino_fulldose.h5 stays available for
# side-by-side comparison via scripts/validate_mayo_helix2fan.py.
#
# Background: the FFS pattern inspector confirmed Mayo SOMATOM AS+
# uses period-2 alternation with drho oscillating 0 ↔ +5.45 mm. The
# correction maps each readout's effective sod/sdd via
#   sod_eff = sod + ffs_drho ; sdd_eff = sdd + ffs_drho
# in scripts/../ddssl_ldct/helix2fan.py:_rebin_one_sangle.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# NFS at /cluster/maier rejects fcntl locks (errno 37); disable HDF5 locking.
export HDF5_USE_FILE_LOCKING=FALSE

# Toggle FFS-drho correction in helix2fan rebinner.
export HELIX2FAN_FFS_DRHO=1
export N_JOBS=16

echo "[rebin-L014-ffs-drho] host=$(hostname) job=$SLURM_JOB_ID"
echo "[rebin-L014-ffs-drho] HELIX2FAN_FFS_DRHO=$HELIX2FAN_FFS_DRHO"
df -h /cluster/maier | tail -1

python scripts/rebin_l014_fulldose_ffs_drho.py

STAGED=$REPO/data/mayo_ldct/staged_helix2fan
ls -la "$STAGED"/L014_sino_fulldose_ffs_drho* | head
echo "[rebin-L014-ffs-drho] done."
