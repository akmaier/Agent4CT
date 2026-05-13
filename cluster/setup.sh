#!/usr/bin/env bash
# Idempotent setup script for Agent4CT on the LME / i5 Slurm cluster.
#
# Builds (or refreshes) the project virtual environment at
#   /cluster/<user>/Agent4CT/.venv
# with:
#   - python 3.10 (system) + pip / wheel / setuptools
#   - PyTorch 2.3.0 (cu118 wheel, compatible with NVIDIA driver 535.x)
#   - PYRO-NN built from source against the locally available CUDA toolkit
#   - Everything in requirements.txt
#
# Verified bring-up: 2026-05-13, node lme170 (Quadro RTX 8000),
# driver 535.288.01, CUDA 12.2 driver / CUDA 11.2 toolkit on disk,
# torch wheel cu118 (works since cu118 only needs a >=520 driver).
#
# Run inside a Slurm job with a GPU allocated, e.g.
#
#     sbatch cluster/slurm/build.sbatch
#
# or interactively:
#
#     srun --gres=gpu:1 --time=01:00:00 --mem=24G --pty bash
#     bash cluster/setup.sh

set -euo pipefail
shopt -s nullglob

REPO=${REPO:-/cluster/$(whoami)/Agent4CT}
VENV=${VENV:-$REPO/.venv}
PYRONN_SRC=${PYRONN_SRC:-$REPO/vendor/PYRO-NN}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"6.1;7.0;7.5;8.0;8.6"}

echo "[setup] host=$(hostname) repo=$REPO venv=$VENV"
nvidia-smi | head -5 || { echo "FATAL: this must run on a node with a GPU"; exit 1; }

# ---------- 1. clone PYRO-NN if missing ------------------------------------
mkdir -p "$REPO/vendor"
if [ ! -d "$PYRONN_SRC" ]; then
    git clone --depth 1 https://github.com/csyben/PYRO-NN.git "$PYRONN_SRC"
fi

# ---------- 2. create / refresh venv ---------------------------------------
if [ ! -d "$VENV" ]; then
    /usr/bin/python3.10 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel "setuptools==69.5.1"

# ---------- 3. torch + cuda 11.8 wheel -------------------------------------
# Driver 535.x supports cu118 fine. (cu121 also works, but PYRO-NN
# upstream pins to cu118 and the cu118 wheel ships a complete bundled CUDA
# runtime.)
python -c "import torch" 2>/dev/null || \
    pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# Sanity (will not be able to verify cuda available on head node).
python - <<'PY'
import torch
print(f"[setup] torch={torch.__version__}  cuda_ok={torch.cuda.is_available()}  bundled_cuda={torch.version.cuda}")
PY

# ---------- 4. CUDA 11.8 toolkit (userspace, via .deb files) ---------------
# The compute nodes have only CUDA 11.2 in /usr/local and the system gcc is
# 11.4 (Ubuntu 22.04). CUDA 11.2 only supports gcc ≤ 10 but g++-9 is not
# installed (only the C frontend is) — a dead end. CUDA 11.8 accepts gcc 11
# fine, so we install it once into /cluster/$(whoami)/cuda-11.8.
#
# We avoid NVIDIA's runfile installer (segfaults under Slurm without sudo)
# and instead pull the individual component .debs from NVIDIA's Ubuntu 22.04
# apt repository and `dpkg-deb -x` them into the userspace dir. ~300 MB total.
CUDA_INSTALL="/cluster/$(whoami)/cuda-11.8"
if [ ! -x "$CUDA_INSTALL/bin/nvcc" ]; then
    REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
    DEBS=(
      cuda-nvcc-11-8_11.8.89-1_amd64.deb
      cuda-cudart-11-8_11.8.89-1_amd64.deb
      cuda-cudart-dev-11-8_11.8.89-1_amd64.deb
      cuda-driver-dev-11-8_11.8.89-1_amd64.deb
      cuda-cccl-11-8_11.8.89-1_amd64.deb
      cuda-nvtx-11-8_11.8.86-1_amd64.deb
      cuda-nvrtc-11-8_11.8.89-1_amd64.deb
      cuda-nvrtc-dev-11-8_11.8.89-1_amd64.deb
      cuda-toolkit-11-8-config-common_11.8.89-1_all.deb
      cuda-toolkit-11-config-common_11.8.89-1_all.deb
      cuda-toolkit-config-common_11.8.89-1_all.deb
    )
    DEBDIR="$REPO/vendor/cuda_debs"
    mkdir -p "$DEBDIR"
    cd "$DEBDIR"
    for d in "${DEBS[@]}"; do
        if [ ! -f "$d" ]; then
            echo "[setup] downloading $d"
            wget -q "$REPO_BASE/$d" -O "$d.partial"
            mv "$d.partial" "$d"
        fi
    done
    STAGE="$DEBDIR/stage"
    rm -rf "$STAGE"
    mkdir -p "$STAGE"
    for d in "${DEBS[@]}"; do
        dpkg-deb -x "$d" "$STAGE"
    done
    mkdir -p "$CUDA_INSTALL"
    rsync -a --no-perms --omit-dir-times \
        "$STAGE/usr/local/cuda-11.8/" "$CUDA_INSTALL/"
    [ -d "$CUDA_INSTALL/lib64" ] || ln -sfn lib "$CUDA_INSTALL/lib64"
    cd "$REPO"
fi

export CUDA_HOME="$CUDA_INSTALL"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[setup] CUDA_HOME=$CUDA_HOME"
which gcc g++ nvcc
gcc --version | head -1
nvcc --version | tail -2

# ---------- 5. project requirements ----------------------------------------
pip install --no-cache-dir -r "$REPO/requirements.txt"
pip install --no-cache-dir ninja build

# ---------- 6. build PYRO-NN torch backend ---------------------------------
# Upstream's pyproject.toml pins torch==2.3.0+cu118 as a build dep which
# pip-isolation can't resolve cleanly inside an installed venv (PyPI extra
# index handling). Swap it for a build-isolation-friendly file just for the
# install, then restore.
cd "$PYRONN_SRC"
[ -f pyproject.toml.orig ] || cp pyproject.toml pyproject.toml.orig
cat > pyproject.toml <<'TOML'
[build-system]
requires = ["setuptools==69.5.1", "ninja"]
build-backend = "setuptools.build_meta"

[project]
name = "pyronn"
version = "1.1.0"
authors = [{name = "Christopher Syben"}]
description = "PYRO-NN: high level Python API for known operators."
readme = "README.rst"
requires-python = ">=3.9"
dependencies = ["numpy"]

[tool.setuptools.packages.find]
where = ["src"]
include = ["pyronn*"]

[tool.setuptools.package-data]
pyronn_layers = ["*.so"]
TOML
echo '{"backend": "torch"}' > src/pyronn/CONFIG.json

# --no-build-isolation: re-use the venv's torch / setuptools / ninja so the
# CUDAExtension build sees the same torch we'll use at runtime.
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    pip install -v --no-build-isolation --no-cache-dir .

# Restore original pyproject so the vendored tree stays clean.
mv pyproject.toml.orig pyproject.toml
cd "$REPO"

# ---------- 7. smoke import ------------------------------------------------
python - <<'PY'
import torch, pyronn, pyronn_layers
print(f"[setup] torch={torch.__version__}  cuda_ok={torch.cuda.is_available()}")
print(f"[setup] pyronn backend = {pyronn.read_backend()}")
print(f"[setup] pyronn_layers ops = {[s for s in dir(pyronn_layers) if not s.startswith('_')][:8]}...")
from pyronn.ct_reconstruction.geometry.geometry_base import GeometryFan2D
from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D
from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D
print("[setup] PYRO-NN torch fan-beam classes resolve OK.")
PY
echo "[setup] DONE."
