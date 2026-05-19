#!/bin/bash
#SBATCH --job-name=rebin-mayo-helix2fan
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Track A2: helical -> 2D fan-beam rebinning for the Mayo LDCT Wagner subset.
# Reads raw projection DICOMs from data/mayo_ldct/raw/L<NNN>/<series>/, runs
# the two-step curved->flat->fan pipeline in ddssl_ldct/helix2fan.py, and
# writes per-patient staged HDF5s to data/mayo_ldct/staged_helix2fan/.
#
# CPU-bound (joblib over s_angle and per-view); ~10 patients × 2 dose series
# at ~1 h/series on 16 cores is the rough budget.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
source "$VENV/bin/activate"

echo "[rebin] host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
df -h /cluster/maier | tail -1

python data/fetch_mayo_ldct.py --rebin-only \
    --dv-rebinned 1.0 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}"

STAGED=/cluster/maier/Agent4CT/data/mayo_ldct/staged_helix2fan
echo "[rebin] done."
ls -la "$STAGED" | head -40
du -sh "$STAGED" || true
