#!/bin/bash
#SBATCH --job-name=demo-diff-recon-dcstep-con-tpe
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
# hr=0.5712), using the CONSTRAINED DDPM checkpoint (200-phantom trained
# prior). Same enqueue_trial seed as the unconstrained search. The
# Claude search found constrained worse by 0.026 hr (iter 7), but the
# broader cfg-space exploration here may favour the narrower prior.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[diff-recon-dcstep-constrained-tpe] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python scripts/learned_solver_search_agent.py \
    --solver diffusion_recon_dcstep_constrained \
    --sampler tpe \
    --iterations 20 \
    --notes "TPE search around Claude iter-16 winner; constrained DDPM ckpt"
echo "[diff-recon-dcstep-constrained-tpe] done."
