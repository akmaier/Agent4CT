#!/bin/bash
#SBATCH --job-name=demo-ram-oneshot
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Single-iter RAM zero-shot recon, config picked by the Claude agent.
# Required env: RAM_CONFIG_PATH  OUT  (driver supplies both)

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

: "${RAM_CONFIG_PATH:?need RAM_CONFIG_PATH}"
: "${OUT:?need OUT}"
mkdir -p "$OUT"

echo "[ram-oneshot] host=$(hostname) job=$SLURM_JOB_ID  cfg=$RAM_CONFIG_PATH  out=$OUT"
nvidia-smi -L

export RAM_CONFIG_PATH
python pentathlon/demo_dl_reference/solver_ram.py "$OUT"
cat "$OUT/result.json" 2>/dev/null | python3 -c "import json,sys; r=json.load(sys.stdin); print('HR=%.4f SSIM=%.4f PSNR=%.2f params=%.2fM' % (r['headroom'], r['val_ssim'], r['val_psnr'], r['params_M']))" || true
echo "[ram-oneshot] done."
