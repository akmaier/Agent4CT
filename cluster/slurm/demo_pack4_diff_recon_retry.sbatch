#!/bin/bash
#SBATCH --job-name=demo-pack4-diff-recon
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# v2 diffusion-recon retry: 4 in parallel (uncon + con × random + TPE).
# Fixed sample_guided: gradient now normalised to unit-length so eta is a
# meaningful step size; clamp inside the loop kept tight.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[pack-diff-recon] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

LOG=results/slurm/pack-diff-recon-${SLURM_JOB_ID}
mkdir -p "$LOG"

run () {
    local gpu="$1"; local solver="$2"; local sampler="$3"; local notes="$4"
    local out="$LOG/${solver}-${sampler}.out"
    echo "[pack-diff] CUDA_VISIBLE_DEVICES=$gpu solver=$solver sampler=$sampler -> $out"
    CUDA_VISIBLE_DEVICES=$gpu \
        python scripts/learned_solver_search_agent.py \
            --solver "$solver" --iterations 20 --sampler "$sampler" \
            --notes "$notes" \
            > "$out" 2>&1 &
}

run 0 diffusion_recon_unconstrained random  "v2 diffusion-recon unconstrained (random) — grad-normalised"
run 1 diffusion_recon_constrained   random  "v2 diffusion-recon constrained   (random) — grad-normalised"
run 2 diffusion_recon_unconstrained tpe     "v2 diffusion-recon unconstrained (TPE)    — grad-normalised"
run 3 diffusion_recon_constrained   tpe     "v2 diffusion-recon constrained   (TPE)    — grad-normalised"

wait
echo "[pack-diff-recon] all children done."
