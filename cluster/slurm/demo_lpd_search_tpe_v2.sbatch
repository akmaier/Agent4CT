#!/bin/bash
#SBATCH --job-name=demo-lpd-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Demo-DL LPD TPE search to fill the gap vs breast-ct (where LPD is the
# leader at hr=0.9062). Seeded with the breast-CT iter-3 winner
# (I=10/hidden=64/lr=5e-4/ep=20). 20 trials × ~40 min average ≈ 13 h.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[demo-lpd-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

python scripts/learned_solver_search_agent.py \
    --solver lpd \
    --sampler tpe \
    --calibrated \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "LPD TPE search on demo-DL (intensity-calibrated); fills the gap vs breast-ct where LPD is leader"

echo "[demo-lpd-tpe] done."
