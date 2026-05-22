#!/bin/bash
# Recompute best Dual-Domain iteration (iter-0017) with corrected SSIM
#SBATCH --job-name=dd-recomp-best
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

# Best Dual-Domain params from search (iter-0017: best headroom)
# epochs=6, lr=0.000407, batch_size=2, unet_c=19
OUT="/cluster/maier/Agent4CT/runs/dd-recomp-best-${SLURM_JOB_ID}"
mkdir -p "$OUT"

cat > "$OUT/dd_config.json" <<'JSONEOF'
{"epochs": 6, "lr": 0.00040713221339467714, "batch_size": 2, "unet_c": 19, "train_n": 200}
JSONEOF

DD_CONFIG_PATH="$OUT/dd_config.json" python pentathlon/demo_dl_reference/solver_dual_ddomain_n2i.py "$OUT"
echo "[dd-recomp] Done. Output: $OUT"
