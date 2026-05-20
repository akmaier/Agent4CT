#!/bin/bash
# Submit a batch of intensity-calibrated TPE searches (CONVENTIONS.md rule 4).
# All runs use slug-prefix demo-intensity-calibrated-* so they group as a
# separate dashboard chart from demo-fair-*.
#
# Designed for the lme/miti partition where MaxJobsPU caps user concurrency.
# We rely on Slurm's queue to serialize.
#
# Wraps each solver in a small sbatch that pins the GPU and walltime.

set -euo pipefail
REPO=/cluster/maier/Agent4CT
SLURM_DIR=$REPO/cluster/slurm

# (solver, walltime, mem, gpu?) — gpu? = 1 if needs a GPU, 0 if CPU-only.
# Format: "solver:walltime:mem:gpu"
JOBS=(
  "itnet_v3:04:00:00:24G:1"
  "uswin:04:00:00:24G:1"
  "hammernik_vn:04:00:00:24G:1"
  "hammernik:03:00:00:16G:1"
  "itnet_v2:03:00:00:16G:1"
  "dual_domain:03:00:00:16G:1"
  "dual_domain_bilateral:03:00:00:16G:1"
  "tv_iterative:01:00:00:8G:0"
  "wu_2015:01:00:00:8G:0"
  "diffusion_recon_dcstep_unconstrained:06:00:00:24G:1"
  "diffusion_recon_dcstep_constrained:06:00:00:24G:1"
  "ram_zeroshot:01:30:00:24G:1"
  "naf:16:00:00:24G:1"   # 8h TIMEOUT on previous run; bump to 16h
)

mkdir -p $REPO/results/slurm

for entry in "${JOBS[@]}"; do
  IFS=':' read -r SOLVER H M S MEM GPU <<<"$entry"
  WALL="${H}:${M}:${S}"
  JOBNAME="cal-tpe-${SOLVER}"
  # IMPORTANT: must include the "#SBATCH " prefix here. A bare
  # `--gres=gpu:1` line is a non-comment, non-blank line and Slurm
  # treats it as the start of the script body, silently dropping
  # every following #SBATCH directive (mem, time, output, error).
  # That was the bug behind the calibrated-TPE batch logs landing
  # in /home/maier/ and the itnet_v3 OOM (no --mem=24G honored).
  if [ "$GPU" = "1" ]; then
    GRES_LINE="#SBATCH --gres=gpu:1"
  else
    GRES_LINE="#SBATCH --comment=cpu-only"
  fi
  cat > /tmp/_cal_$$.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
$GRES_LINE
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

echo "[cal-tpe] solver=$SOLVER host=\$(hostname) job=\$SLURM_JOB_ID"
if [ "$GPU" = "1" ]; then nvidia-smi -L; fi
python scripts/learned_solver_search_agent.py \\
    --solver $SOLVER \\
    --sampler tpe \\
    --iterations 20 \\
    --calibrated \\
    --notes "intensity-calibrated TPE (CONVENTIONS rule 4)"
echo "[cal-tpe] done."
EOF
  sbatch /tmp/_cal_$$.sbatch
done
rm -f /tmp/_cal_$$.sbatch
