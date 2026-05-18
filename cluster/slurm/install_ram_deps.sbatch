#!/bin/bash
#SBATCH --job-name=install-ram-deps
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Install deepinv + RAM (pulled from upstream pip) and pre-stage the
# pretrained RAM checkpoint (mterris/ram, ~143 MB) into the cluster's
# /cluster/maier/Agent4CT/checkpoints/ directory so the per-iter solver
# doesn't have to re-download every time.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
VENV="$REPO/.venv"
source "$VENV/bin/activate"

echo "[install] host=$(hostname) job=$SLURM_JOB_ID"
which pip; pip --version

# 1. Install RAM (pulls in deepinv + huggingface_hub as transitive deps).
pip install --quiet --upgrade huggingface_hub deepinv
pip install --quiet 'git+https://github.com/matthieutrs/ram@main'
python -c "import ram, deepinv; print('[install] ram ok, deepinv', deepinv.__version__)"

# 2. Fetch the pretrained checkpoint from HuggingFace mterris/ram.
mkdir -p "$REPO/checkpoints"
python - << 'PY'
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(repo_id="mterris/ram", filename="ram.pth.tar")
dst = "/cluster/maier/Agent4CT/checkpoints/ram.pth.tar"
if os.path.realpath(src) != os.path.realpath(dst):
    shutil.copyfile(src, dst)
print(f"[install] checkpoint -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
PY

ls -la /cluster/maier/Agent4CT/checkpoints/ram.pth.tar
echo "[install] done."
