#!/bin/bash
#SBATCH --job-name=breast-alliter
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme170   # only genuinely down nodes (lme49 unk, lme170 drain). The 11GB gtx1080ti nodes lme50/51/52 are ALLOWED for test-scoring: node choice does not affect the deterministic metric, and the otherwise-idle cluster is wasted. Heavy-solver OOM on 11GB is handled by a final resubmit onto big nodes.
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%A_%a.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%A_%a.err

# Breast-CT all-iters TEST-scoring sweep (the test-selection sweep). ONE array
# task per (solver, iter). Each task runs the solver ONCE on the 200 held-out
# test cases via the loader redirect (AGENT4CT_EVAL_SPLIT=test, set by the
# worker), persists the raw recon, and re-scores per case -> per-case mean±std
# incl. headroom_std. Breast cases are i.i.d. (no patients).
#
# --time=00:30:00 = the strict 20-min budget + headroom (breast 128-view fan is
# far cheaper than Mayo). Submit with a %8 running throttle under the ≤60 QOS cap:
#   sbatch --array=1-<N>%8 --export=ALL,MANIFEST=<path> cluster/slurm/breast_alliters_array.sbatch
# The manifest is a TSV written by scripts/score_breast_alliters.py --build; line
# <SLURM_ARRAY_TASK_ID> = "<slug-itertest/iter>\t<solver_key>\t<cfg_relpath>\t<solver_src_relpath>".

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
MANIFEST="${MANIFEST:-$REPO/agentic_cfgs/breast_alliters/manifest.tsv}"
[ -f "$MANIFEST" ] || { echo "[breast-alliter] MANIFEST not found: $MANIFEST" >&2; exit 1; }

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
python -c "import hdf5plugin" 2>/dev/null || true   # register blosc/zstd filter for staged breast h5

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -n "$LINE" ] || { echo "[breast-alliter] empty manifest line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
SLUG=$(printf '%s' "$LINE" | cut -f1)
SOLVER=$(printf '%s' "$LINE" | cut -f2)
CFG_JSON=$(printf '%s' "$LINE" | cut -f3)
SOLVER_SRC=$(printf '%s' "$LINE" | cut -f4)

echo "[breast-alliter] host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[breast-alliter] slug=$SLUG  solver=$SOLVER  cfg=$CFG_JSON  src=${SOLVER_SRC:-none}"
nvidia-smi -L || true

python scripts/score_breast_testset.py --worker \
    --slug "$SLUG" --solver "$SOLVER" --cfg "$CFG_JSON" \
    --split test --solver-src "${SOLVER_SRC:-}"

echo "[breast-alliter] task=${SLURM_ARRAY_TASK_ID} done."
