#!/bin/bash
#SBATCH --job-name=fetch-ct-mar-restage
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-stage CT-MAR from the existing raw/body*.tar.gz tars into the full
# 5-tensor MAR-ready layout: truth + clean sino + corrupted img + corrupted
# sino + metal mask + metalinfo.json. Replaces the earlier truth-only
# staging which threw away exactly the inputs MAR methods consume.
#
# Wall: 4h is generous; the bottleneck is HDF5 write speed across
# ~30-40 GB of float32 with lz4. Sino files are ~3.6 MB each, packing
# 4000 of them is ~14 GB train sino * 2 (clean+corrupt) = ~28 GB plus
# ~5 GB images + masks = ~35 GB on disk.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
source "$VENV/bin/activate"

echo "[ct-mar restage] host=$(hostname) job=$SLURM_JOB_ID"
df -h /cluster/maier | tail -1

# Wipe the old truth-only staging (user explicitly authorized destruction;
# the raw/ tars remain so this can be re-staged any time).
STAGED=/cluster/maier/Agent4CT/data/ct_mar/staged
if [ -d "$STAGED" ]; then
  echo "[ct-mar restage] removing old truth-only staged/ (re-stageable from raw/)"
  rm -rf "$STAGED"
fi
mkdir -p "$STAGED"

python data/fetch_ct_mar.py --force
echo "[ct-mar restage] done."
ls -la "$STAGED"
du -sh "$STAGED"
