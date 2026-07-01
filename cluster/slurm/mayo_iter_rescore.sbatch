#!/bin/bash
#SBATCH --job-name=pe-rescore
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Re-score ONE param-efficient iteration's EXACT architecture (its solver_src.py
# snapshot + its cfg_full) on val L277 under the FROZEN corrected metric, so the
# whole campaign trajectory becomes comparable on one metric. Persists the raw
# recon (AGENT4CT_SAVE_RECON) for auditability. The solver's own evaluate_calibrated
# writes result.json with the corrected val headroom/ssim/psnr/rmse (+ stds).
#
# Required: SOLVER_SRC (iter snapshot, repo-relative), CFG_JSON, OUTDIR.

set -euo pipefail
: "${SOLVER_SRC:?need SOLVER_SRC}"; : "${CFG_JSON:?need CFG_JSON}"; : "${OUTDIR:?need OUTDIR}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$OUTDIR"
export AGENT4CT_DATASET=mayo_ldct_2d           # hard-wires bg_target=truth + Mayo FOV
export AGENT4CT_SAVE_RECON="$OUTDIR"           # persist raw recon
export PARAM_EFFICIENT_CONFIG_PATH="$CFG_JSON"
# NB: AGENT4CT_EVAL_PATIENT intentionally UNSET -> normal val (L277).

echo "[pe-rescore] host=$(hostname) job=${SLURM_JOB_ID:-NA} src=$SOLVER_SRC outdir=$OUTDIR"
nvidia-smi -L; echo
python "$SOLVER_SRC" "$OUTDIR"
echo "[pe-rescore] done -> $OUTDIR/result.json"
