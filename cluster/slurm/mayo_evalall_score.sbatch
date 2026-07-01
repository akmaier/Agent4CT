#!/bin/bash
#SBATCH --job-name=mayo-evalall
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# TRAIN-ONCE / EVAL-ALL + persisted-recon scorer (one job per Mayo solver).
# Runs the solver ONCE with AGENT4CT_EVAL_PATIENT=all (val L277 ++ 5 test patients
# = 959 slices) and AGENT4CT_SAVE_RECON, so a SUPERVISED solver trains once and
# reconstructs every held-out slice (no per-patient retraining); the raw recon is
# persisted; then scripts/score_mayo_from_recons.py re-scores EACH patient under
# the FIXED metric (per-patient calibration) -> final.json (mean ± std over 5).
#
# Required exported variables (set by the dispatcher's --export=ALL,...):
#   SLUG        — Mayo run-id slug
#   SOLVER      — SOLVER_MAP key (metadata + offline scorer)
#   SOLVER_SRC  — path to the solver .py to RUN (the iter snapshot for code-evolving
#                 solvers, else the SOLVER_MAP path), relative to REPO
#   ENVVAR      — the solver's config env var (e.g. PARAM_EFFICIENT_CONFIG_PATH)
#   CFG_JSON    — path to the best-iter cfg JSON
#   OUTDIR      — output dir (recon_raw.npz + final.json + per-patient figures)

set -euo pipefail
: "${SLUG:?need SLUG}"; : "${SOLVER:?need SOLVER}"; : "${SOLVER_SRC:?need SOLVER_SRC}"
: "${ENVVAR:?need ENVVAR}"; : "${CFG_JSON:?need CFG_JSON}"; : "${OUTDIR:?need OUTDIR}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

RUNDIR="$OUTDIR/run"
mkdir -p "$RUNDIR"

export AGENT4CT_DATASET=mayo_ldct_2d
export AGENT4CT_EVAL_PATIENT=all
export AGENT4CT_SAVE_RECON="$RUNDIR"
export "$ENVVAR=$CFG_JSON"

echo "[evalall] host=$(hostname) job=${SLURM_JOB_ID:-NA}"
echo "[evalall] slug=$SLUG solver=$SOLVER src=$SOLVER_SRC envvar=$ENVVAR"
echo "[evalall] cfg=$CFG_JSON outdir=$OUTDIR"
nvidia-smi -L; echo

# 1) train once + reconstruct ALL held-out slices + persist raw recon
python "$SOLVER_SRC" "$RUNDIR"

if [ ! -f "$RUNDIR/recon_raw.npz" ]; then
  echo "[evalall] ERROR: $RUNDIR/recon_raw.npz not produced — solver failed or SAVE_RECON not honored" >&2
  exit 3
fi

# 2) per-patient re-score under the FIXED metric (zero retraining)
python scripts/score_mayo_from_recons.py \
    --recon "$RUNDIR/recon_raw.npz" --slug "$SLUG" \
    --outdir "$OUTDIR" --solver-key "$SOLVER"

echo "[evalall] done -> $OUTDIR/final.json"
