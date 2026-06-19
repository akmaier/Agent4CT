#!/bin/bash
# Resubmit the 4 breast-ct TPE jobs that died on lme170's NVML mismatch
# (761515 / diff_recon_unconstrained, 761516 / diff_recon_constrained,
#  761517 / ram_zeroshot, 761518 / naf). Now excludes lme170 + lme171.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
mkdir -p $REPO/results/slurm

JOBS=(
  "diffusion_recon_dcstep_unconstrained:06:00:00:24G:1"
  "diffusion_recon_dcstep_constrained:06:00:00:24G:1"
  "ram_zeroshot:01:30:00:24G:1"
  "naf:16:00:00:24G:1"
)

for entry in "${JOBS[@]}"; do
  IFS=':' read -r SOLVER H M S MEM GPU <<<"$entry"
  WALL="${H}:${M}:${S}"
  JOBNAME="breast-ct-tpe-${SOLVER}"
  GRES_LINE="#SBATCH --gres=gpu:1"
  cat > /tmp/_resub_$$.sbatch <<EOF
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

echo "[resub] solver=$SOLVER dataset=breast_ct host=\$(hostname) job=\$SLURM_JOB_ID"
nvidia-smi -L || true
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    --dataset breast_ct \\
    --notes "post-geomfix TPE resubmit (lme170 NVML fix)"
echo "[resub] done."
EOF
  sbatch /tmp/_resub_$$.sbatch
done
rm -f /tmp/_resub_$$.sbatch
