#!/bin/bash
#SBATCH --job-name=demo-pack3-tpe-heavy
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# 3 heavier TPE searches in parallel: DD Bilateral, R2Gaussian, Hammernik-VN.
# Excludes small-GPU + driver-bad nodes (lme50-53) so the bigger models fit.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[pack-tpe-heavy] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

LOG=results/slurm/pack-tpe-heavy-${SLURM_JOB_ID}
mkdir -p "$LOG"

run_tpe () {
    local gpu="$1"; local solver="$2"; local notes="$3"
    local out="$LOG/${solver}.out"
    echo "[pack-tpe-heavy] CUDA_VISIBLE_DEVICES=$gpu solver=$solver TPE -> $out"
    CUDA_VISIBLE_DEVICES=$gpu \
        python scripts/learned_solver_search_agent.py \
            --solver "$solver" --iterations 20 --sampler tpe \
            --notes "$notes" \
            > "$out" 2>&1 &
}

run_tpe 0 dual_domain_bilateral  "packed-tpe: DD Bilateral (Optuna TPE)"
run_tpe 1 r2gaussian             "packed-tpe: R2Gaussian (Optuna TPE)"
run_tpe 2 hammernik_vn           "packed-tpe: Hammernik-VN (Optuna TPE)"

wait
echo "[pack-tpe-heavy] all children done."
