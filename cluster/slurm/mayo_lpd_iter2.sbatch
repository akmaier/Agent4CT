#!/bin/bash
#SBATCH --job-name=mayo-lpd-iter2
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Mayo-LDCT LPD autoresearch iter-2.
#
# Iter-1 (SLURM 762665) crashed at 48s with CUDA OOM on Q6000 (24 GB):
#     Tried to allocate 622.00 MiB while in dual_blocks[i].conv1
#
# Root-cause diagnosis: the breast-CT champion config (I=8, hidden=96)
# was tuned for 128-angle sinograms (DL-Sparse-View). Mayo has 2304
# angles — ~18× more sino pixels per channel. The LPD dual blocks
# materialise (B, hidden, A=2304, U=736) intermediates; at I=8 ×
# hidden=96 the unrolled forward/backward graph needs ~20 GB,
# exceeding Q6000 capacity.
#
# Iter-2 hypothesis: shrink LPD's unrolled depth and channel width by
# the same factor we scaled up angles — roughly 1/6 of v1. New
# config:
#     I=4 (was 8)              — half the unrolled iterations
#     hidden=32 (was 96)        — third the channel width
#     val_chunk=1 (was 4)       — chunk val one-at-a-time to keep
#                                 the eval pass small
#     epochs=15 (was 23)        — first iter is a feasibility test
# Param count drops 1.49 M → ≈ 80 k. If iter-2 fits AND gives any hr,
# iter-3 can scale back up (hidden=48 or 64) to find the Mayo sweet
# spot.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

SLUG=mayo-ldct-claude-agentic-learned-primal-dual-search-20260603-01
ITER=2
SOLVER=learned_primal_dual

mkdir -p /cluster/maier/Agent4CT/agentic_cfgs
CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_lpd_iter_02.json
cat > "$CFG_JSON" <<'JEOF'
{
  "lpd_iters": 4,
  "lpd_hidden": 32,
  "lpd_n_primal": 3,
  "lpd_n_dual": 3,
  "lpd_share_weights": false,
  "epochs": 15,
  "lr": 3.2e-4,
  "lr_schedule": "cosine",
  "batch_size": 1,
  "grad_clip": 0.3,
  "lambda_neg": 1.0,
  "loss_base": "mse",
  "val_n": 20,
  "val_chunk": 1
}
JEOF

echo "[mayo-lpd-iter2] host=$(hostname) job=$SLURM_JOB_ID slug=$SLUG iter=$ITER"
nvidia-smi -L
cat "$CFG_JSON"; echo

SLUG="$SLUG" ITER_N="$ITER" SOLVER="$SOLVER" CFG_JSON="$CFG_JSON" \
  AGENT4CT_DATASET=mayo_ldct_2d \
  python scripts/claude_agentic_one_iter.py

echo "[mayo-lpd-iter2] done."
