#!/bin/bash
#SBATCH --job-name=fetch-dl-sparse-view
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Download + stage the AAPM DL-Sparse-View CT 2021 challenge data from
# Zenodo 14173522 (Sidky 2024 public release). ~5 GB raw / ~10 GB staged.
# Network is the long pole; conversion runs single-threaded but is bounded
# by HDF5 write speed (~50-100 MB/s).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
source "$VENV/bin/activate"

echo "[fetch] host=$(hostname) job=$SLURM_JOB_ID"
df -h /cluster/maier | tail -1

python data/fetch_dl_sparse_view.py
echo "[fetch] done."
ls -la /cluster/maier/Agent4CT/data/dl_sparse_view/staged/
