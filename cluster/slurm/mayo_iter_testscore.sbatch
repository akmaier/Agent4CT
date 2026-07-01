#!/bin/bash
#SBATCH --job-name=pe-testscore
#SBATCH --partition=main
#SBATCH --qos=turbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme170
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# TEST-set re-score of ONE param-efficient iteration's EXACT architecture (its
# solver_src.py snapshot + its cfg_full) on the 5 held-out Wagner TEST patients
# (L014 L056 L058 L075 L123) — the HONEST Mayo metric (train once per patient,
# mean ± std over the 5). Reuses the proven per-patient worker loop in
# scripts/score_mayo_testset.py (--worker): it sets AGENT4CT_EVAL_PATIENT=<Lxxx>
# one patient per eval pass (the PYRO-NN "invalid resource handle" crash happens
# if a single pass cycles multiple pixel-spacings, so NEVER batch patients),
# runs the snapshot once per patient, and aggregates -> final.json.
#
# Output is keyed by SLUG, so passing SLUG=pe-iter-testeval/iter-NNNN lands the
# aggregate at docs/runs/pe-iter-testeval/iter-NNNN/final.json and recons under
# runs/pe-iter-testeval/iter-NNNN-testset/. This does NOT touch the 40 snapshots,
# ddssl_ldct/, or other solvers' final.json (different SLUG namespace).
#
# Required exported variables (set by scripts/rescore_pe_iters.py --test --dispatch):
#   SLUG       — pe-iter-testeval/iter-NNNN  (output namespace; not a real run-id)
#   CFG_JSON   — path to this iter's cfg_full JSON the dispatcher wrote
#   SOLVER_SRC — repo-relative path to this iter's solver_src.py snapshot

set -euo pipefail
: "${SLUG:?need SLUG (pe-iter-testeval/iter-NNNN)}"
: "${CFG_JSON:?need CFG_JSON path}"
: "${SOLVER_SRC:?need SOLVER_SRC (iter snapshot, repo-relative)}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "[pe-testscore] host=$(hostname) job=${SLURM_JOB_ID:-NA}"
echo "[pe-testscore] slug=$SLUG  src=$SOLVER_SRC  cfg=$CFG_JSON"
nvidia-smi -L
echo
cat "$CFG_JSON"; echo

# param_efficient is the supervised code-evolving solver; --solver-src points the
# worker at THIS iter's snapshot instead of the latest SOLVER_MAP file.
python scripts/score_mayo_testset.py --worker \
    --slug "$SLUG" --solver param_efficient --cfg "$CFG_JSON" \
    --solver-src "$SOLVER_SRC"

echo "[pe-testscore] done -> docs/runs/$SLUG/final.json"
