#!/bin/bash
#SBATCH --job-name=mayo-first-iter
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lme49,lme53,lme170,lme171
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/maier/Agent4CT/results/slurm/%x-%j.out
#SBATCH --error=/cluster/maier/Agent4CT/results/slurm/%x-%j.err

# Kick off the first Mayo-LDCT autoresearch iter for a given solver.
# Run AFTER `stage_mayo_sinos.sbatch` (job 762100) finishes.
#
#   sbatch --export=ALL,SOLVER=learned_primal_dual,SLUG=mayo-ldct-claude-agentic-learned-primal-dual-search-20260524-01,ITER=1 \
#          cluster/slurm/mayo_first_iter.sbatch
#
# Seeds the iter-1 config from the breast-CT champion for that solver
# (per solver_plan.md Step 2: start agentic from the cross-dataset
# leader, then let the loop adapt).

set -euo pipefail
: "${SOLVER:?need SOLVER (e.g. learned_primal_dual, dual_domain_supervised)}"
: "${SLUG:?need SLUG (e.g. mayo-ldct-claude-agentic-<solver>-search-YYYYMMDD-NN)}"
: "${ITER:?need ITER (typically 1)}"

REPO=/cluster/maier/Agent4CT
cd "$REPO"
VENV="$REPO/.venv"
export CUDA_HOME="$VENV/cuda-pip-bundle"
export PATH="$CUDA_HOME/bin:$PATH"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

# Choose the seed config from the breast-CT champion of this solver.
mkdir -p /cluster/maier/Agent4CT/agentic_cfgs
case "$SOLVER" in
  learned_primal_dual)
    # Breast-CT TPE trial-11 winner (hr 0.9062):
    CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_lpd_iter_01.json
    cat > "$CFG_JSON" << 'EOF'
{
  "lpd_iters": 8, "lpd_hidden": 96,
  "lpd_n_primal": 5, "lpd_n_dual": 5,
  "lpd_share_weights": false,
  "epochs": 23, "lr": 3.2e-4, "lr_schedule": "cosine",
  "batch_size": 1, "grad_clip": 0.3,
  "lambda_neg": 1.0, "loss_base": "mse",
  "val_n": 20, "val_chunk": 4
}
EOF
    ;;
  dual_domain_supervised)
    CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_dd_unet_l2_iter_01.json
    cat > "$CFG_JSON" << 'EOF'
{ "epochs": 10, "lr": 5.0e-4, "batch_size": 1, "unet_c": 32,
  "lambda_neg": 1.0, "val_n": 20 }
EOF
    ;;
  dual_domain_bilateral_supervised)
    CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_dd_bf_l2_iter_01.json
    cat > "$CFG_JSON" << 'EOF'
{ "proj_n_bf": 3, "img_n_bf": 3, "proj_kernel": 3, "img_kernel": 9,
  "proj_sx": 0.01, "proj_sy": 0.3, "proj_sr": 0.0005,
  "img_sx": 0.5, "img_sr": 0.02,
  "epochs": 10, "lr": 5.0e-3, "val_n": 20 }
EOF
    ;;
  ram_zeroshot)
    # Frozen RAM (Terris 2025) — no training, just inference. Seed
    # from breast-CT iter-7 winner (hr 0.3077): adjoint_max + sigma=0.008
    # + post-FBP blend=0.42. Same network ckpt used across all datasets.
    CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_ram_iter_01.json
    cat > "$CFG_JSON" << 'EOF'
{ "ram_ckpt_path": "/cluster/maier/Agent4CT/checkpoints/ram.pth.tar",
  "ram_input_norm": "adjoint_max", "ram_clamp_output": true,
  "ram_finetune": false, "ram_finetune_epochs": 0,
  "ram_disable_multiscale": false, "ram_disable_cudnn": false,
  "ram_sigma": 0.008, "ram_factor": 0.42, "ram_post_fbp_blend": 0.5,
  "val_n": 20 }
EOF
    ;;
  uswin)
    # Breast-CT TPE (iter-7) winner: ep=11, lr=7.3e-4, c=24, win=16,
    # heads=8 (hr=0.7174). Image-domain Swin — should fit on Mayo's
    # 512² recon without sino-side issues.
    CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/mayo_uswin_iter_01.json
    cat > "$CFG_JSON" << 'EOF'
{ "epochs": 11, "lr": 7.3e-4, "batch_size": 1, "uswin_c": 24,
  "swin_window": 16, "swin_heads": 8, "val_n": 20 }
EOF
    ;;
  *)
    echo "Unknown SOLVER=$SOLVER; add a seed config above." >&2
    exit 1
    ;;
esac

echo "[mayo-first-iter] host=$(hostname) job=$SLURM_JOB_ID"
echo "[mayo-first-iter] solver=$SOLVER  slug=$SLUG  iter=$ITER"
nvidia-smi -L
cat "$CFG_JSON"
echo

SLUG="$SLUG" ITER_N="$ITER" SOLVER="$SOLVER" CFG_JSON="$CFG_JSON" \
  AGENT4CT_DATASET=mayo_ldct_2d \
  python scripts/claude_agentic_one_iter.py

echo "[mayo-first-iter] done."
