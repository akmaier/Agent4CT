#!/bin/bash
#SBATCH --job-name=ddssl-dlsv-loss-stage
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:55:00            # 1.5h stage budget at 2x epochs
#SBATCH --exclude=lme171,lme221,lme222,lme223,lme49,lme50,lme51,lme52,lme53
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err
# DL-Sparse-View LOSS-FUNCTION stage check: same solver, 3x data, longer training.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
cd "$REPO"
mkdir -p results/slurm

VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

OUT="$REPO/runs/stage-loss-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[stage-loss] host=$(hostname)  job=$SLURM_JOB_ID  out=$OUT"
nvidia-smi -L

python - "$OUT" <<'PY'
import sys
sys.path.insert(0, ".")
import pentathlon.dl_sparse_view_loss.solver as solver
cfg = dict(solver.CONFIG)
cfg.update({
    "train_n": cfg["train_n"] * 3,
    "val_n":   cfg["val_n"]   * 3,
    "epochs":  max(cfg["epochs"] * 2, 16),
})
print(f"[stage-loss] config override: train_n={cfg['train_n']}  "
      f"val_n={cfg['val_n']}  epochs={cfg['epochs']}", flush=True)
from pathlib import Path
solver.main(Path(sys.argv[1]), cfg=cfg)
PY

echo
echo "==== STAGE METRICS ===="
cat "$OUT/result.json"
echo
echo "[stage-loss] done."
