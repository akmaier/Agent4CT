# Runtime and I/O — Agent4CT on the LME cluster

This document covers two intertwined questions:

1. **How does PYRO-NN process data?** (Spoiler: it doesn't load it. We do.)
2. **How do we keep training fast on the LME cluster's shared filesystem?**

The short answer: pack the per-slice CT data into a handful of large container
files, stage them onto each compute node's `/scratch` SSD at job start, and
let PyTorch's `DataLoader` stream from there. NFS, small files, and Python
overhead are the three things that wreck wall-clock time if you don't.

---

## 1. PYRO-NN: what it actually consumes

PYRO-NN is a **CUDA kernel** wrapped in a PyTorch autograd `Function`. It
does **not**:

- read files,
- batch slices,
- dispatch work across GPUs,
- manage host ⇄ device transfers beyond a single forward call.

It does:

- Take a CUDA tensor of shape `(B, H, W)` for the image (no channel dim),
- Take a small dictionary of CUDA geometry tensors (built once),
- Return a CUDA tensor of shape `(B, A, D)` (sinogram), or the inverse for
  back-projection.

That means **all the throughput engineering happens around PYRO-NN, not
inside it**. Our wrapper (`ddssl_ldct/pyronn_projector.py`) is a thin
adapter: insert/remove the singleton channel axis, keep the geometry on the
GPU, and call `apply()`. Nothing more.

### The implications

- **Geometry tensors must be built once.** The high-level
  `FanProjectionFor2D` helper in upstream PYRO-NN rebuilds them on every
  call (~5–20 ms per call, negligible on a 2-second forward but real on a
  short pass). Our wrapper builds them in `__init__` and stores them as
  module buffers.
- **Inputs must be `.contiguous()` CUDA float32.** Otherwise the CUDA
  extension raises. The wrapper handles this defensively.
- **Mixed precision works for the U-Nets, not for the projector.** PYRO-NN
  compiles its CUDA kernels in float32; pass `.float()` into `apply()` even
  when you train the rest of the pipeline in bf16. Cast back afterwards.
- **Batching scales linearly until you run out of GPU memory.** A 512² FBP
  with 1152 views takes ~80 MB per image (sinogram + intermediates) — you
  can comfortably batch 8–16 on a 24 GB GPU, 32+ on a 48 GB A6000.

---

## 2. The small-files problem on `/cluster`

`/cluster` is an NFS mount of the lab's shared NAS. Two performance facts
matter:

1. **Latency dominates throughput.** Each `open()` is a network round-trip
   of ~1–5 ms. 14 000 cases × 5 tensors × 5 ms = **6 minutes of latency
   alone per epoch on CT-MAR**, before reading a byte. With 4 dataloader
   workers it's "only" 90 s, but the pattern is fragile and shows up as
   GPU-stall bubbles in nvidia-smi.
2. **Throughput is OK once a file is open.** Sequential reads from `/cluster`
   sit at 100–300 MB/s once the cache is warm.

The fix is to **never touch many small files inside the training loop**.

### What to do at data staging (one-time)

Pack each split into one or two container files:

```
data/<challenge>/staged/
    train_sinograms.h5      # one HDF5 dataset of shape (N, 1, A, D)
    train_images.h5         # one HDF5 dataset of shape (N, 1, 512, 512)
    val_*.h5
    test_*.h5
    manifest.json
```

HDF5 with `chunks=(1, 1, A, D)` and gzip-1 / lz4 compression gets us:

- Single `open()` per epoch (instead of N).
- Random-access into the chunked dataset is O(1) on the page cache.
- ~2–3× compression on sinograms, often more on images with low entropy.

Zarr is an equally good choice and writes directly to a directory tree.
`numpy.memmap` over one giant `.npy` works if compression isn't needed and
the data is float32.

### What to do at job start

`/scratch` is a per-node local SSD (~1 GB/s sequential, microsecond latency)
and is the right home for hot data. Stage there at job start:

```bash
#!/bin/bash
#SBATCH ...
set -euo pipefail
SRC=/cluster/maier/Agent4CT/data/mayo_ldct/staged
DST=/scratch/$USER/$SLURM_JOB_ID/mayo_ldct
mkdir -p "$DST"
trap 'rm -rf "$DST"' EXIT       # auto-clean on job end

# Lazy mirror — only copies what isn't there. Bandwidth is gated by the
# NFS link (~300 MB/s); 150 GB takes ~10 minutes.
rsync -aP "$SRC/" "$DST/"

source /cluster/maier/Agent4CT/.venv/bin/activate
export AGENT4CT_DATA=$DST
python -m scripts.run_experiment ...
```

The `trap` line is important — it removes the scratch copy when the job
exits so the node doesn't fill up across hundreds of jobs.

### What to do in the `DataLoader`

```python
from torch.utils.data import Dataset, DataLoader
import h5py

class StagedH5(Dataset):
    """Opens the file in each worker, not in the parent process."""
    def __init__(self, path):
        self._path = path
        self._f = None
    def _open(self):
        if self._f is None:                       # lazy in each worker
            self._f = h5py.File(self._path, "r", libver="latest", swmr=True)
        return self._f
    def __len__(self):
        return self._open()["sino"].shape[0]
    def __getitem__(self, i):
        ds = self._open()
        return (
            torch.from_numpy(ds["sino"][i]).float(),
            torch.from_numpy(ds["image"][i]).float(),
        )

loader = DataLoader(
    StagedH5("/scratch/.../mayo_ldct/staged/train.h5"),
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```

Why those flags:

- `num_workers=4` — one HDF5 reader per CPU core dedicated to the job
  (`--cpus-per-task=4` in the sbatch). More workers hit diminishing returns
  on a single GPU.
- `persistent_workers=True` — keep HDF5 file handles open across epochs;
  saves the per-epoch reopen.
- `pin_memory=True` — pinned host memory pages overlap with GPU DMA.
- `prefetch_factor=2` — two batches in flight per worker.

### Don't keep raw DICOM in the loop

A common bug pattern: code that opens hundreds of DICOM `.dcm` files per
training step using `pydicom`. Every parse pays the per-file open cost
**plus** pydicom's Python-side overhead. Convert once at staging time, never
again. For the Mayo data this means: rebin helical → fan-beam **once**,
write the fan-beam sinograms to a single `.h5`, throw away the DICOM in the
training loop.

---

## 3. Slurm flags that matter

```bash
#SBATCH --partition=main
#SBATCH --gres=gpu:a6000:1          # 48 GB; spec the type so you don't get a 1080 Ti
#SBATCH --cpus-per-task=4           # = num_workers
#SBATCH --mem=24G                   # host RAM; HDF5 page cache helps a lot
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --signal=B:USR1@180         # 3-min warning before kill → checkpoint
```

`SIGUSR1` 3 minutes before the wall-time kill is the cleanest way to
checkpoint and resubmit. Wrap training in a handler that saves and exits 0
on `USR1`, then have the sbatch resubmit itself with
`--dependency=afterok:$SLURM_JOB_ID`.

The cluster guide § 4.4 has the full chained-resubmit pattern; the
`gpu_experiments/cluster/slurm/submit_chain.sh` we *had* in the
`known_operator_ct_release` repo (now retired) was a working reference.

---

## 4. Concrete checklist for each challenge

When adding a new challenge folder:

- [ ] Pack the split into one or two HDF5 files under `data/<challenge>/staged/`.
- [ ] Write a `geometries.json` once and load it in the dataset constructor.
- [ ] Add a `Dataset` class that opens HDF5 lazily in each worker.
- [ ] In the sbatch wrapper: stage `/cluster/.../staged/` → `/scratch/...` at
      job start; `trap rm` at exit.
- [ ] Pin `--cpus-per-task` to match `num_workers`.
- [ ] Run a 1-epoch profile (`torch.profiler` or just `nvidia-smi dmon -s
      pucvmet -d 1`) — utilisation should stay > 80 % between FBP calls.
      If it dips below 50 % you're I/O-bound; widen the prefetch or shrink
      the model.

---

## 5. What to *not* do

- **Don't `.cpu()` mid-pipeline.** Every host transfer kills overlap. Keep
  the whole training step on the GPU; convert to CPU only for logging.
- **Don't write checkpoints under `/home`.** Cluster guide § 2: home is
  congested and slow. Checkpoints belong on `/cluster/maier/Agent4CT/runs`.
- **Don't pre-rebin Mayo on every job.** Once into the staged HDF5, never
  re-rebin.
- **Don't request `--gres=gpu:1` without a type pin** if you care about
  speed. You'll occasionally get a GTX 1080 with 8 GB and OOM at 256×256.
- **Don't use `pickle`-backed `.pt` per-slice files.** That's the worst of
  both worlds: small files **and** Python overhead.
