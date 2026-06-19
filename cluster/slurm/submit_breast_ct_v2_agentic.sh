#!/bin/bash
# Agentic v2 re-runs for the 4 hr=0 breast-ct solvers (tv, dual_domain,
# dual_domain_bilateral, ram_zeroshot). New search spaces are narrowed
# per user direction (2026-05-21):
#   - tv: lower lambda, more iters (more data fidelity)
#   - dual_domain: less training + smaller net (less oversmoothing)
#   - dual_domain_bilateral: smaller proj sigmas (less projection blur)
#   - ram_zeroshot: lower sigma + higher post_fbp_blend (more data fidelity)
# Each runs --calibrated --dataset breast_ct --iterations 20.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
mkdir -p $REPO/results/slurm

JOBS=(
  "tv_iterative_v2:01:00:00:8G:0"
  "dual_domain_v2:03:00:00:16G:1"
  "dual_domain_bilateral_v2:03:00:00:16G:1"
  "ram_zeroshot_v2:01:30:00:24G:1"
)

for entry in "${JOBS[@]}"; do
  IFS=':' read -r SOLVER H M S MEM GPU <<<"$entry"
  WALL="${H}:${M}:${S}"
  JOBNAME="breast-ct-v2-${SOLVER}"
  if [ "$GPU" = "1" ]; then
    GRES_LINE="#SBATCH --gres=gpu:1"
  else
    GRES_LINE="#SBATCH --comment=cpu-only"
  fi
  cat > /tmp/_v2_$$.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
$GRES_LINE
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=$MEM
#SBATCH --time=$WALL
#SBATCH --output=$REPO/results/slurm/%x-%j.out
#SBATCH --error=$REPO/results/slurm/%x-%j.err

set -euo pipefail
cd $REPO
VENV=$REPO/.venv
export CUDA_HOME=\$VENV/cuda-pip-bundle
export PATH=\$CUDA_HOME/bin:\$PATH
source \$VENV/bin/activate
export PYTHONPATH=$REPO:\${PYTHONPATH:-}

echo "[v2] solver=$SOLVER dataset=breast_ct host=\$(hostname) job=\$SLURM_JOB_ID"
if [ "$GPU" = "1" ]; then nvidia-smi -L || true; fi
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    --dataset breast_ct \\
    --notes "agentic-guided v2 (less smoothing / more data fidelity per user hints)"
echo "[v2] done."
EOF
  sbatch /tmp/_v2_$$.sbatch
done
rm -f /tmp/_v2_$$.sbatch
