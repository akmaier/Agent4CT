---
title: Setup
description: Cluster bring-up, environment, and data.
---

## What you need

1. SSH-key access to a Slurm cluster with at least one ≥ 24 GB-VRAM GPU node.
2. Python 3.10+.
3. CUDA-capable hardware (PYRO-NN's projection kernels are CUDA-only).
4. (Optional) Mayo LDCT or another public CT dataset — synthetic phantoms also work for testing.

## Local sanity check (without a GPU)

The package imports cleanly without CUDA — you can iterate on geometry, simulation, and U-Net code on a laptop. The PYRO-NN backend is only loaded when a projector is constructed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from ddssl_ldct.geometry import FanBeamGeometry; print(FanBeamGeometry())"
```

## Cluster bring-up

Copy the cluster operating-guide template:

```bash
cp config/cluster_agent_guide.template.md config/<your-site>_cluster_agent_guide.md
# fill in your site's hostnames, GPU inventory, etc. (this file is gitignored)
```

Then on the cluster (inside a GPU Slurm allocation):

```bash
sbatch cluster/slurm/ddssl_smoke.sbatch
# or interactively:
srun --gres=gpu:1 --time=01:00:00 --mem=24G --pty bash
bash cluster/setup.sh
```

`cluster/setup.sh` is idempotent: it clones [csyben/PYRO-NN](https://github.com/csyben/PYRO-NN), builds the CUDA extensions against the locally-available toolkit (it provisions CUDA 11.8 to a user-writable path if the node only has older), and verifies the torch-backend fan-beam projector imports cleanly.

## Smoke test

```bash
sbatch cluster/slurm/ddssl_smoke.sbatch
```

Watches: `nvidia-smi -L`, torch version + CUDA availability, a Shepp-Logan forward + FBP at the Wagner geometry (512 × 512 / 1152 views / 736 dets), then a 5-epoch run on 8 synthetic phantoms. The output PNGs land in `runs/exp_pyronn/`.

## Data

Each Pentathlon challenge has its own download instructions in `challenges/<name>/README.md`. None of them ship inside the repo — pull what you need from the cited source. Storage layout convention:

```
<cluster_root>/<user>/Agent4CT/data/<challenge>/
  raw/        # untouched source files
  staged/     # H5 / Zarr containers packed for fast streaming (see docs/performance.md)
```

See [Performance](performance.html) for the staging pattern that keeps the GPU busy on a contended NFS.

## Configuration

| File | Purpose |
|---|---|
| `config/llm_api.example.toml` | Copy to `config/llm_api.toml`, fill in your LLM endpoint + key. Real file is **gitignored**. |
| `config/cluster_agent_guide.template.md` | Copy to `config/<site>_cluster_agent_guide.md`. Real file is **gitignored**. |
| `requirements.txt` | Project dependencies (excluding PYRO-NN, which is built from source). |
