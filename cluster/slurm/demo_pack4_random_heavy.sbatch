#!/bin/bash
#SBATCH --job-name=demo-pack4-heavy
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclude=lme49
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Round-2 pack: the 4 heavier random searches (r2g/hammernik/
# hammernik_vn/itnet_v3/dual_domain_bilateral — pick 4 with similar
# wall budgets).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[pack4-heavy] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

LOG=results/slurm/pack4-heavy-${SLURM_JOB_ID}
mkdir -p "$LOG"

run_search () {
    local gpu="$1"; local solver="$2"; local notes="$3"
    local out="$LOG/${solver}.out"
    echo "[pack4-heavy] CUDA_VISIBLE_DEVICES=$gpu solver=$solver -> $out"
    CUDA_VISIBLE_DEVICES=$gpu \
        python scripts/learned_solver_search_agent.py \
            --solver "$solver" --iterations 20 \
            --notes "$notes" \
            > "$out" 2>&1 &
}

run_search 0 r2gaussian             "packed-random: R2Gaussian under CONVENTIONS"
run_search 1 hammernik              "packed-random: Hammernik 2017 under CONVENTIONS"
run_search 2 hammernik_vn           "packed-random: Hammernik-VN under CONVENTIONS"
run_search 3 dual_domain_bilateral  "packed-random: DD-Bilateral under CONVENTIONS"

wait
echo "[pack4-heavy] all children done."
