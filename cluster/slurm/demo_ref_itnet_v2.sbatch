#!/bin/bash
#SBATCH --job-name=demo-ref-itnet-v2
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q8000:1
#SBATCH --exclude=lme171,lme222,lme223,lme49,lme50,lme51,lme52
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="$REPO/runs/demo-ref-itnet-v2-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[demo-ref-itnet-v2] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

python pentathlon/demo_dl_reference/solver_itnet_v2.py "$OUT"

cat "$OUT/result.json"
echo "[demo-ref-itnet-v2] done."
