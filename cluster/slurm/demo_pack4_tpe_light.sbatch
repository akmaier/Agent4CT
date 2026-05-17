#!/bin/bash
#SBATCH --job-name=demo-pack4-tpe-light
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# 4 lightweight TPE searches in parallel: TV, Wu, ItNet v2, DD U-Net.
# Pairs the 4 random siblings that already finished in pack4-random
# (job 761233).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[pack-tpe-light] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

LOG=results/slurm/pack-tpe-light-${SLURM_JOB_ID}
mkdir -p "$LOG"

run_tpe () {
    local gpu="$1"; local solver="$2"; local notes="$3"
    local out="$LOG/${solver}.out"
    echo "[pack-tpe-light] CUDA_VISIBLE_DEVICES=$gpu solver=$solver TPE -> $out"
    CUDA_VISIBLE_DEVICES=$gpu \
        python scripts/learned_solver_search_agent.py \
            --solver "$solver" --iterations 20 --sampler tpe \
            --notes "$notes" \
            > "$out" 2>&1 &
}

run_tpe 0 tv_iterative   "packed-tpe: TV (Optuna TPE)"
run_tpe 1 wu_2015        "packed-tpe: Wu 2015 (Optuna TPE)"
run_tpe 2 itnet_v2       "packed-tpe: ItNet v2 (Optuna TPE)"
run_tpe 3 dual_domain    "packed-tpe: DD U-Net (Optuna TPE)"

wait
echo "[pack-tpe-light] all children done."
