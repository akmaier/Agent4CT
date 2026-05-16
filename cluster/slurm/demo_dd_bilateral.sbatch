#!/bin/bash
#SBATCH --job-name=dd-bf-test
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out

set -euo pipefail
cd /cluster/maier/Agent4CT
source .venv/bin/activate
export PYTHONPATH="/cluster/maier/Agent4CT${PYTHONPATH:+:$PYTHONPATH}"

OUT="/cluster/maier/Agent4CT/runs/dd-bilateral-test-${SLURM_JOB_ID}"
mkdir -p "$OUT"

python pentathlon/demo_dl_reference/solver_dual_ddomain_bilateral.py "$OUT"
echo "[dd-bf] Done. Output: $OUT"
