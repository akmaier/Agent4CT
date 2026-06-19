#!/bin/bash
# Fill in the r2gaussian TPE searches I forgot in submit_post_geomfix_tpe_batch.sh
# (also missing from the original submit_calibrated_tpe_batch.sh). Two jobs:
# one demo-intensity (phantoms) + one breast-ct.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
mkdir -p $REPO/results/slurm

DATASETS=("phantoms" "breast_ct")
WALL="03:00:00"
MEM="24G"
SOLVER="r2gaussian"

for ds in "${DATASETS[@]}"; do
  if [ "$ds" = "phantoms" ]; then
    DS_FLAG=""
    JOBNAME="cal-tpe-${SOLVER}"
  else
    DS_FLAG="--dataset $ds"
    JOBNAME="${ds//_/-}-tpe-${SOLVER}"
  fi
  cat > /tmp/_r2g_$$.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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

echo "[r2g] solver=$SOLVER dataset=$ds host=\$(hostname) job=\$SLURM_JOB_ID"
nvidia-smi -L || true
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    $DS_FLAG \\
    --notes "post-geomfix TPE r2gaussian backfill"
echo "[r2g] done."
EOF
  sbatch /tmp/_r2g_$$.sbatch
done
rm -f /tmp/_r2g_$$.sbatch
