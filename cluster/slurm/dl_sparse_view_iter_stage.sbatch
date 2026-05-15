#!/bin/bash
#SBATCH --job-name=ddssl-dlsv-iter-stage
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:10:00            # 1-h stage + 10-min slack
#SBATCH --exclude=lme171,lme221,lme222,lme223,lme49,lme50,lme51,lme52,lme53
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# DL-Sparse-View ITER stage check: same solver, 3x data, longer training.
# Runs every 30 iterations to catch overfitting on the iteration subset.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="$REPO/runs/stage-iter-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[stage-iter] host=$(hostname)  job=$SLURM_JOB_ID  out=$OUT"
nvidia-smi -L

python - "$OUT" <<'PY'
import sys
sys.path.insert(0, ".")
import pentathlon.dl_sparse_view_iter.solver as solver
cfg = dict(solver.CONFIG)
cfg.update({
    "train_n": cfg["train_n"] * 3,
    "val_n":   cfg["val_n"]   * 3,
    "epochs":  max(cfg["epochs"] * 4, 32),
})
print(f"[stage-iter] config override: train_n={cfg['train_n']}  "
      f"val_n={cfg['val_n']}  epochs={cfg['epochs']}", flush=True)
from pathlib import Path
solver.main(Path(sys.argv[1]), cfg=cfg)
PY

echo
echo "==== STAGE METRICS ===="
cat "$OUT/result.json"
echo
echo "[stage-iter] done."
