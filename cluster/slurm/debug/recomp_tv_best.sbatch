#!/bin/bash
# Recompute best TV iteration (iter-0017) with corrected SSIM
#SBATCH --job-name=tv-recomp-best
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

# Best TV params from search (iter-0017: best SSIM)
# lambda=0.008074, iters=66, lr=0.039511, clip=0.0370, decay=0.0273
OUT="/cluster/maier/Agent4CT/runs/tv-recomp-best-${SLURM_JOB_ID}"
mkdir -p "$OUT"

cat > "$OUT/tv_config.json" <<'JSONEOF'
{"tv_lambda": 0.008074, "tv_iterations": 66, "tv_lr": 0.039511, "tv_clip_max": 0.0370, "tv_decay": 0.0273, "tv_optimizer": "gd", "tv_init": "fbp"}
JSONEOF

TV_CONFIG_PATH="$OUT/tv_config.json" python pentathlon/demo_dl_reference/solver_tv_search.py "$OUT"
echo "[tv-recomp] Done. Output: $OUT"
