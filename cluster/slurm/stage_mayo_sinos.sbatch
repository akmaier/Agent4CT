#!/bin/bash
#SBATCH --job-name=stage-mayo-sinos
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Pack the helix2fan per-patient sinos into per-split aggregated H5s
# aligned to the existing per-split truth H5s. Run AFTER
# `rebin_mayo_helix2fan.sbatch` finishes — needs the per-patient sino
# files in data/mayo_ldct/staged_helix2fan/.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[stage-mayo-sinos] host=$(hostname) job=$SLURM_JOB_ID"
python data/stage_mayo_sinos.py --dose both
echo "[stage-mayo-sinos] done."
ls -la /cluster/maier/Agent4CT/data/mayo_ldct/staged/ | tail
