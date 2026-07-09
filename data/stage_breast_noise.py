"""stage_breast_noise.py — generate a DETERMINISTIC Poisson-noised copy of the
Breast-CT (Sidky DL-Sparse-View) TEST sinograms, for the BreastCT_Noise second
leaderboard (paper §5.6.7).

The Sidky challenge data is NOISELESS ideal line-integral projection data. This adds
photon-counting (Poisson) noise at a fixed dose I0 to the 200 TEST sinograms ONLY, with
a FIXED seed so every solver sees byte-identical noisy input (fair, reproducible board).
Truth is left clean — the noisy board still measures recovery of the clean phantom, so a
solver's robustness to input noise is exactly what it rewards.

Photon model on a line-integral sinogram p (p = -log(transmission)):
    N_clean = I0 * exp(-p)                 # expected transmitted photons per detector bin
    N_noisy ~ Poisson(N_clean)             # counting noise
    N_noisy = max(N_noisy, 1)              # floor (avoid log 0 on fully-blocked rays)
    p_noisy = log(I0) - log(N_noisy) = -log(N_noisy / I0)
    p_noisy = clamp(p_noisy, min=0)        # non-negative line integral

At I0=100000 the thickest breast ray (p~3.45 -> ~3170 photons) sees ~1.8% noise; air is
noiseless — i.e. a mild, high-dose perturbation. Writes test_sinograms_noise_i0_<I0>.h5
(dataset key "sino", float32, same shape) next to the clean staged files, plus a small
manifest recording I0 + seed.

Usage (cluster):
    python data/stage_breast_noise.py --i0 100000 --seed 20260709
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

STAGED = Path("data/dl_sparse_view/staged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i0", type=float, default=100000.0, help="incident photons/bin (dose)")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--staged", default=str(STAGED))
    args = ap.parse_args()

    import hdf5plugin  # noqa: F401  (register blosc filter for compressed h5 IO)
    import h5py
    import numpy as np

    sd = Path(args.staged)
    src = sd / f"{args.split}_sinograms.h5"
    assert src.exists(), f"missing {src}"
    i0_tag = int(args.i0)
    dst = sd / f"{args.split}_sinograms_noise_i0_{i0_tag}.h5"

    rng = np.random.default_rng(args.seed)
    with h5py.File(src, "r") as f:
        sino = f["sino"]
        n, H, W = sino.shape
        print(f"[noise] {src.name}: {n} cases {H}x{W}  I0={i0_tag}  seed={args.seed}", flush=True)
        # write incrementally to bound memory (200 * 128 * 1024 * 4 ~ 100MB is fine, but
        # stream anyway so this scales to the 3600 train set if ever needed).
        comp = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        with h5py.File(dst, "w") as g:
            out = g.create_dataset("sino", shape=(n, H, W), dtype="float32", **comp)
            eps_stats = {"max_noise_p": 0.0, "mean_abs_dp": 0.0}
            for i in range(n):
                p = np.asarray(sino[i], dtype=np.float64)
                n_clean = args.i0 * np.exp(-p)
                n_noisy = rng.poisson(n_clean).astype(np.float64)
                n_noisy = np.maximum(n_noisy, 1.0)
                p_noisy = np.log(args.i0) - np.log(n_noisy)
                p_noisy = np.clip(p_noisy, 0.0, None).astype(np.float32)
                out[i] = p_noisy
                dp = np.abs(p_noisy - p.astype(np.float32))
                eps_stats["max_noise_p"] = max(eps_stats["max_noise_p"], float(dp.max()))
                eps_stats["mean_abs_dp"] += float(dp.mean())
            eps_stats["mean_abs_dp"] /= n
    manifest = sd / f"{args.split}_sinograms_noise_i0_{i0_tag}.manifest.json"
    manifest.write_text(json.dumps({
        "source": src.name, "output": dst.name, "i0": args.i0, "seed": args.seed,
        "split": args.split, "n": n, "model": "poisson_transmission",
        **eps_stats}, indent=2))
    print(f"[noise] wrote {dst}  (max Δp={eps_stats['max_noise_p']:.4f}, "
          f"mean|Δp|={eps_stats['mean_abs_dp']:.5f})", flush=True)
    print(f"[noise] manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
