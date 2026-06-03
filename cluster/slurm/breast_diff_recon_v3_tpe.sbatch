#!/bin/bash
#SBATCH --job-name=breast-diff-recon-v3-tpe
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52,lme170
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Autoresearch iter-1 of the breast-DDPM ARCHITECTURE LOOP, using the
# new v3 ckpts (ch=128, 60 ep, val_eps_loss=0.0020 — 2.5× v2).
#
# History so far (refuted hypotheses):
#   v1 ckpts (ch=32, 25 ep): diff-recon all hr=0, SSIM ~0.46
#   v2 ckpts (ch=64, 80 ep): diff-recon all hr=0, SSIM ~0.45 (SLURM 762636)
#   → "more training" is REFUTED. v2 trained cleanly but samples still bad.
#
# Iter-1 hypothesis: v3 (ch=128) finally has enough capacity to model
# breast-phantom complexity. If diff-recon-v3 STILL gives hr=0 across
# 40 trials of DPS+DC sampling, the SmallDDPM architecture is genuinely
# too constrained for this dataset and we'd need a different prior
# class (score-SDE, U-ViT, EDM with adaptive variance).
#
# Slug prefixes:
#   breast-ct-calibrated-tpe-diff-recon-dcstep-{,un}constrained-breast-v3-search-…
# Both variants run sequentially (20 trials each = 40 total).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[diff-recon-v3] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L

# Variant 1: UNCONSTRAINED (all 3600 breast train + 40 val + ?)
python scripts/learned_solver_search_agent.py \
    --solver diffusion_recon_dcstep_unconstrained_breast_v3 \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "Diff-recon TPE on breast-CT against ddpm_breast_unconstrained_v3.pt (ch=128/60ep, v2 was still hr=0)"

# Variant 2: CONSTRAINED (200 breast train only)
python scripts/learned_solver_search_agent.py \
    --solver diffusion_recon_dcstep_constrained_breast_v3 \
    --sampler tpe \
    --calibrated \
    --dataset breast_ct \
    --iterations 20 \
    --tpe-startup 5 \
    --notes "Diff-recon TPE on breast-CT against ddpm_breast_constrained_v3.pt (ch=128/60ep, v2 was still hr=0)"

echo "[diff-recon-v3] done."
