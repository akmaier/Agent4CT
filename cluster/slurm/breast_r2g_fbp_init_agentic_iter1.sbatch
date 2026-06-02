#!/bin/bash
#SBATCH --job-name=breast-r2g-fbp-init-iter1
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme50,lme51,lme52,lme170
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Agentic-iter HYPOTHESIS for breast-CT R²-Gaussian (currently hr=0
# across all configs incl. the v2 retry at 40k iters / 1024 gauss):
# the model isn't structurally wrong, the COLD INIT is. R²-G with
# random uniform positions in [-0.6, 0.6] spends most of its
# capacity on background empty space; for breast-CT phantoms with
# densely-distributed soft tissue, this never converges to baseline.
#
# Iter-1 test: set gs_init_from_fbp=True. Positions sampled ∝ FBP²,
# amps seeded from FBP intensity. If R²-G can converge close to FBP
# on iter 0 and only refine from there, it has a fighting chance of
# clearing baseline. If THIS still gives hr=0, the verdict is
# definitively "wrong inductive bias on dense view" — no further
# R²-G iters needed.
#
# Single-iter dispatch (not TPE) — recorded as a claude-agentic
# observation. Slug:
#   breast-ct-claude-agentic-r2g-fbp-init-search-20260602-01
# CFG below is the post-740-iter sweet spot from breast-CT TPE
# 762629 / 762635 (1024 gauss, mid range of n_iter).

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

SLUG=breast-ct-claude-agentic-r2g-fbp-init-search-20260602-01
ITER=1
SOLVER=r2gaussian

mkdir -p /cluster/maier/Agent4CT/agentic_cfgs
CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/breast_r2g_fbp_init_iter_01.json
cat > "$CFG_JSON" <<'JEOF'
{
  "gs_n_gaussians": 1024,
  "gs_n_iter":      20000,
  "gs_lr_pos":      2.5e-3,
  "gs_lr_scale":    1.2e-2,
  "gs_lr_amp":      6.5e-3,
  "gs_lr_rot":      1.0e-2,
  "gs_amp_init":    0.01,
  "gs_scale_init":  0.04,
  "gs_tv_weight":   1e-4,
  "gs_n_clip":      0.05,
  "gs_init_from_fbp": true,
  "gs_amp_init_max":  0.05,
  "val_n":          20
}
JEOF

echo "[breast-r2g-fbp-init-iter1] host=$(hostname) job=$SLURM_JOB_ID"
echo "[breast-r2g-fbp-init-iter1] slug=$SLUG iter=$ITER solver=$SOLVER"
nvidia-smi -L
cat "$CFG_JSON"
echo

SLUG="$SLUG" ITER_N="$ITER" SOLVER="$SOLVER" CFG_JSON="$CFG_JSON" \
  AGENT4CT_DATASET=breast_ct \
  python scripts/claude_agentic_one_iter.py

echo "[breast-r2g-fbp-init-iter1] done."
