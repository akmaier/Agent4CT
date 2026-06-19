#!/bin/bash
#SBATCH --job-name=smoke-fbp-breast-ct
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Track B smoke test: run FBP baseline against breast_ct staged data
# (Sidky 2021) via AGENT4CT_DATASET dispatch. Should produce
# /tmp/_smoke_fbp_breast/result.json with finite headroom + a
# comparison.png in display range [0, 0.5].

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export AGENT4CT_DATASET=breast_ct

OUT=/cluster/maier/Agent4CT/runs/smoke-fbp-breast-ct-${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "[smoke] host=$(hostname) job=$SLURM_JOB_ID  AGENT4CT_DATASET=$AGENT4CT_DATASET"
nvidia-smi -L
python pentathlon/demo_dl_reference/solver_fbp_baseline.py "$OUT"
echo "---result---"
cat "$OUT/result.json" | python3 -c "
import json,sys
r = json.load(sys.stdin)
keys = ('val_score','val_psnr','val_ssim','val_rmse','baseline_psnr','baseline_rmse','headroom','calibration','fg_threshold')
for k in keys:
    if k in r: print(f'{k} = {r[k]}')"
ls "$OUT"
