#!/bin/bash
#SBATCH --job-name=demo-pack2-hammernik
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# v2 Hammernik 2017 retry — random + TPE in parallel under the
# tightened search space (vn_T={3,5}, N_k={16,24}, kernel={7,9,11})
# that fits 16 GB GPUs without OOM.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[pack2-hammernik] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

LOG=results/slurm/pack2-hammernik-${SLURM_JOB_ID}
mkdir -p "$LOG"

CUDA_VISIBLE_DEVICES=0 \
    python scripts/learned_solver_search_agent.py --solver hammernik \
        --iterations 20 --sampler random \
        --notes "Hammernik 2017 fair v3 (tightened search space)" \
        > "$LOG/hammernik-random.out" 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
    python scripts/learned_solver_search_agent.py --solver hammernik \
        --iterations 20 --sampler tpe \
        --notes "Hammernik 2017 fair TPE v3 (tightened search space)" \
        > "$LOG/hammernik-tpe.out" 2>&1 &

wait
echo "[pack2-hammernik] both children done."
