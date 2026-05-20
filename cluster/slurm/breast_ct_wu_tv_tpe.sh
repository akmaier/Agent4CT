#!/bin/bash
# Track B.4 (partial): submit Wu and TV calibrated TPE searches on the
# Sidky 2021 breast_ct staged dataset. Both are CPU-only and don't need
# the breast DDPM, so they can run while the DDPM training (761428)
# proceeds. Slug-prefix becomes `breast-ct-calibrated-tpe-{tv,wu}-...`,
# giving its own `breast-ct` chart group on the dashboard.

set -euo pipefail
REPO=/cluster/maier/Agent4CT

for entry in "tv_iterative:01:30:00:8G" "wu_2015:01:00:00:8G"; do
  IFS=':' read -r SOLVER H M S MEM <<<"$entry"
  WALL="${H}:${M}:${S}"
  JOBNAME="bct-tpe-${SOLVER}"
  cat > /tmp/_bct_$$.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --comment=cpu-only
#SBATCH --exclude=lme49,lme53
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
export AGENT4CT_DATASET=breast_ct

echo "[bct-tpe] solver=$SOLVER host=\$(hostname) job=\$SLURM_JOB_ID  AGENT4CT_DATASET=\$AGENT4CT_DATASET"
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    --dataset breast_ct \\
    --notes "breast-ct calibrated TPE (Sidky 2021 real sino + intensity calibration)"
echo "[bct-tpe] done."
EOF
  sbatch /tmp/_bct_$$.sbatch
done
rm -f /tmp/_bct_$$.sbatch
