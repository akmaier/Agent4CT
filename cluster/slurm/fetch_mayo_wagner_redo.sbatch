#!/bin/bash
#SBATCH --job-name=fetch-mayo-wagner-redo
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-stage Mayo LDCT with the user-pinned Wagner patient split:
#   train: L145, L186, L209, L219
#   val:   L277
#   test:  L014, L056, L058, L075, L123
# Only L186 overlaps with the previous set, so 9 new patients need
# downloading from TCIA (LDCT-and-Projection-data). Existing patient
# downloads are reused.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
source "$VENV/bin/activate"

echo "[mayo-redo] host=$(hostname) job=$SLURM_JOB_ID"
df -h /cluster/maier | tail -1

# Wipe the old staged dir so the new Wagner split is clean.
STAGED=/cluster/maier/Agent4CT/data/mayo_ldct/staged
if [ -d "$STAGED" ]; then
  echo "[mayo-redo] removing old staged/ (re-stageable from raw/)"
  rm -rf "$STAGED"
fi
mkdir -p "$STAGED"

python data/fetch_mayo_ldct.py --force
echo "[mayo-redo] done."
ls -la "$STAGED"
du -sh "$STAGED"
