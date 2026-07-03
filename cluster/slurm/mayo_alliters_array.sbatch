#!/bin/bash
#SBATCH --job-name=mayo-alliter
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170   # same exclude as mayo_testset_score.sbatch (too-small / down / blacklisted nodes)
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%A_%a.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%A_%a.err

# All-iters TEST-scoring sweep (the test-selection sweep). ONE array task per
# (solver, iter). Each task loops the 5 held-out Wagner test patients via the
# proven score_mayo_testset.py --worker path, training ONCE per iter and reusing
# the checkpoint for the other 4 patients (AGENT4CT_MODEL_CKPT, set by the worker).
#
# Recons are NOT saved during the sweep (AGENT4CT_SWEEP_NORECON=1): ~500 iters x
# 5 patients x ~300 MB won't fit (3.3 TB free). Only the per-iter checkpoint is
# kept; board-iter recons are regenerated from it after collection.
#
# Submit with a concurrency cap (SLURM max 10 running / user):
#   sbatch --array=1-<N>%8 --export=ALL,MANIFEST=<path> cluster/slurm/mayo_alliters_array.sbatch
# The manifest is a TSV written by scripts/score_mayo_alliters.py --build; line
# <SLURM_ARRAY_TASK_ID> = "<slug>\t<solver_key>\t<cfg_relpath>\t<solver_src_relpath>".

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
MANIFEST="${MANIFEST:-$REPO/agentic_cfgs/alliters/manifest.tsv}"
[ -f "$MANIFEST" ] || { echo "[alliter] MANIFEST not found: $MANIFEST" >&2; exit 1; }

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export AGENT4CT_SWEEP_NORECON=1     # keep per-iter ckpt only; skip recons (disk)

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -n "$LINE" ] || { echo "[alliter] empty manifest line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
SLUG=$(printf '%s' "$LINE" | cut -f1)
SOLVER=$(printf '%s' "$LINE" | cut -f2)
CFG_JSON=$(printf '%s' "$LINE" | cut -f3)
SOLVER_SRC=$(printf '%s' "$LINE" | cut -f4)

echo "[alliter] host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[alliter] slug=$SLUG  solver=$SOLVER  cfg=$CFG_JSON  src=${SOLVER_SRC:-none}"
nvidia-smi -L || true

python scripts/score_mayo_testset.py --worker \
    --slug "$SLUG" --solver "$SOLVER" --cfg "$CFG_JSON" \
    --solver-src "${SOLVER_SRC:-}"

echo "[alliter] task=${SLURM_ARRAY_TASK_ID} done."
