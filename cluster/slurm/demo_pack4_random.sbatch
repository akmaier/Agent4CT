#!/bin/bash
#SBATCH --job-name=demo-pack4-random
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclude=lme49
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Pack 4 lightweight solver searches into one Slurm allocation so we
# use one of our 4 MaxJobsPU slots to do 4 simultaneous searches —
# crucial workaround for the miti-QoS hard limit on concurrent jobs.
# Each child runs on its own GPU via CUDA_VISIBLE_DEVICES.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[pack4-random] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

LOG=results/slurm/pack4-random-${SLURM_JOB_ID}
mkdir -p "$LOG"

run_search () {
    local gpu="$1"; local solver="$2"; local notes="$3"
    local out="$LOG/${solver}.out"
    echo "[pack4] CUDA_VISIBLE_DEVICES=$gpu solver=$solver -> $out"
    CUDA_VISIBLE_DEVICES=$gpu \
        python scripts/learned_solver_search_agent.py \
            --solver "$solver" --iterations 20 \
            --notes "$notes" \
            > "$out" 2>&1 &
}

# 4 cheap solvers in parallel. Each uses ~16 GB GPU mem so any GPU type fits.
run_search 0 tv_iterative          "packed-random: TV under CONVENTIONS"
run_search 1 wu_2015               "packed-random: Wu 2015 under CONVENTIONS"
run_search 2 itnet_v2              "packed-random: ItNet v2 under CONVENTIONS"
run_search 3 dual_domain           "packed-random: DD U-Net under CONVENTIONS"

wait
echo "[pack4-random] all children done."
