#!/bin/bash
#SBATCH --job-name=cache-proj-flat-L014
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE

echo "[cache] host=$(hostname) job=$SLURM_JOB_ID"
python -u scripts/cache_proj_flat_L014.py
echo "[cache] done."
