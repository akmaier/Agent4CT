#!/bin/bash
#SBATCH --job-name=demo-diff-recon-dcstep-uncon-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# 20-iter Optuna TPE search around the Claude-agentic winner (iter 16:
# hr=0.5712), using the UNCONSTRAINED DDPM checkpoint. First trial is
# seeded via enqueue_trial with the iter-16 config so TPE starts with a
# strong prior. Wall budget: 6h covers 20 iters × ~15 min each (some
# configs explore 800 sample steps × n_cg=25).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[diff-recon-dcstep-unconstrained-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python scripts/learned_solver_search_agent.py \
    --solver diffusion_recon_dcstep_unconstrained \
    --sampler tpe \
    --iterations 20 \
    --notes "TPE search around Claude iter-16 winner; unconstrained DDPM ckpt"
echo "[diff-recon-dcstep-unconstrained-tpe] done."
