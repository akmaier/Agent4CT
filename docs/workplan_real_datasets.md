# Workplan — real datasets (breast-CT, Mayo-LDCT-2D) + matching DDPMs

Multi-day effort to extend the `demo-intensity-calibrated-*` battery from
synthetic ellipses to two real datasets that are already on the cluster.
**This document is the source-of-truth task list across context compressions
and session restarts** — update it incrementally as items complete.

Last updated: 2026-05-19

---

## Track A — Mayo helix→fan rebinning *(background)*

The Mayo Wagner-redo fetch (job 761363) completed truth-only staging
(579 train / 214 val / 745 test slices, all 10 patients downloaded).
The full ~6.4 GB/patient of helical projection-data DICOMs sits in
`raw/L<NNN>/<series-uid>/` ready for rebinning to 2D fan-beam.

- [ ] **A1.** Implement helix2fan-style rebinning in `data/fetch_mayo_ldct.py`:
  - read DICOM-CT-PD tags (0x70291002 du, 0x70291006 dv, 0x70311003 sod,
    0x70311031 sdd, 0x70311033 [u₀,v₀], 0x70311002 z, 0x70311001 φ,
    Rows/Columns/RescaleSlope/RescaleIntercept)
  - per-readout curved→flat detector remap (bilinear in (φ,v))
  - helical→fan SSR (Noo 1999), full 2π scan, no Parker weighting
  - output `(rotview≈2304, nu=736, nz_per_patient)` per series
  - keep per-projection z-positions in a sidecar so downstream code can
    align truth slice → sino angles for the 2D extract (Track C)
- [ ] **A2.** Slurm wrapper `cluster/slurm/rebin_mayo_helix2fan.sbatch`:
  - 24h walltime (CPU-only, joblib parallelism over patients)
  - one job per patient OR a single 10-patient job
  - writes `data/mayo_ldct/staged_helix2fan/L<NNN>_sino_clean.h5`
    + `L<NNN>_sino_lowdose.h5` (TCIA provides both dose series)
- [ ] **A3.** Validate via FBP on one patient: rebinned sino → fbp → compare
  to staged truth slice at matching z. Use existing PyronnFanBeamProjector.
  Iterate on geometry until SSIM ≥ 0.85.
- [ ] **A4.** Submit + run rebinning batch.

## Track B — Breast-CT experiment (`breast-ct-*`)

Reuse the existing `data/dl_sparse_view/staged/` (Sidky 2021 challenge,
4000 cases of breast phantoms with real 128-view sinograms + ground truth
images at 512² in μ-units). The solvers currently use synthetic ellipses;
swap in this dataset and re-run the calibrated TPE battery under a new
slug-prefix so it groups as its own dashboard chart.

### B1. Data loader

- [ ] **B1.1.** Add `ddssl_ldct/staged_dataset.py` with `load_split(kind, split,
    n, device)` that returns `(phantoms, clean, noisy_or_sino)`.
  Kinds: `"phantoms"` (current default), `"breast_ct"`, `"mayo_ldct_2d"`.
  For `breast_ct`: open `data/dl_sparse_view/staged/<split>_truth.h5` and
  `<split>_sinograms.h5`, return (truth, None, sino) — sino is real, not
  forward-projected.
- [ ] **B1.2.** Geometry: re-confirm `n_angles=128, n_det=1024, sod=595,
    sdd=1085.6, det_spacing=1.2858, pixel_spacing=0.7` matches what the
    Zenodo manifest documents (already pinned in `fetch_dl_sparse_view.py`
    GEOMETRY dict).
- [ ] **B1.3.** Modify each of the 17 solvers' `build_dataset()` to dispatch
    on `cfg["dataset_kind"]`. Default `"phantoms"` keeps current behaviour.

### B2. New DDPM checkpoints for breast phantoms

Diffusion-recon needs a DDPM trained on the target dataset's image
distribution. Breast phantoms differ from ellipse phantoms (different
texture, different tissue density distribution), so the existing
`ddpm_unconstrained_final.pt` / `ddpm_constrained_final.pt` are unusable.

- [ ] **B2.1.** Add `cfg["dataset_kind"]=breast_ct` support to
    `solver_ddpm.py` (load images from `data/dl_sparse_view/staged/train_truth.h5`).
- [ ] **B2.2.** Train `ddpm_breast_unconstrained_final.pt` (all 3600 train
    images) — 1–2 h on Q8000.
- [ ] **B2.3.** Train `ddpm_breast_constrained_final.pt` (subset of 200 train
    images, matching the constrained convention).
- [ ] **B2.4.** Update `solver_diffusion_recon.py` to accept a ckpt path so
    the calibrated TPE can target the breast DDPMs.

### B3. Search-agent prefix

- [ ] **B3.1.** Add a `--dataset breast_ct` flag (or `--slug-prefix-override`)
    to `learned_solver_search_agent.py` that rewrites `demo-intensity-
    calibrated-tpe-*` → `breast-ct-calibrated-tpe-*` and sets
    `dataset_kind=breast_ct` in every cfg.
- [ ] **B3.2.** Submit batch sbatch wrapping the 13 solvers (mirror of
    `submit_calibrated_tpe_batch.sh`, plus the breast DDPM ckpts).

### B4. Run + dashboard

- [ ] **B4.1.** Run TPE battery (~10 h serialised on 4-slot concurrency).
- [ ] **B4.2.** Sync to dashboard; the `breast-ct` chart-group key (first
    two hyphen segments of slug-prefix) gives a separate chart from
    `demo-intensity`.

## Track C — Mayo-LDCT-2D experiment (`mayo-ldct-2d-*`)

Take the **center axial slice** per patient + the corresponding helical
**projection angles that physically illuminated that slice** to build a
small per-slice 2D dataset of real sinograms aligned with real truth.
Critical: real sinos must be z-aligned to truth via the per-readout
DetectorFocalCenterAxialPosition tag — this is exactly what Track A is
producing as a side product.

### C1. Build the 2D extract

- [ ] **C1.1.** Pick the center axial slice z_c per patient from the
    existing truth staging (the helix2fan output's nz axis maps directly
    to physical z; choose nz_c = nz//2).
- [ ] **C1.2.** Extract the 2D fan-beam sinogram at z_c from the rebinned
    helical data (Track A): each of the 2304 output views has a per-row
    z-grid; interpolate to z_c per view to get a `(2304, 736)` 2D sino.
    `helix2fan` already does this z-interpolation as part of SSR — just
    grab the central z output of each (rotview, nu, nz) tensor.
- [ ] **C1.3.** Validate ONE patient end-to-end: FBP(extract_sino) vs
    truth slice at z_c. Need SSIM ≥ 0.85 and visual alignment (no
    rotation, no shift). Iterate until correct.
  - Debug knobs: angle direction (CW vs CCW), detector indexing order,
    SDD/SOD sign convention, axial offset z₀.
- [ ] **C1.4.** Once one patient is verified, batch the 10 patients into
    `data/mayo_ldct_2d/staged/{train,val,test}_truth.h5` (1 slice per
    patient, so 4/1/5 cases at first; can later use multiple slices per
    patient if 10 is too few for TPE).
- [ ] **C1.5.** Stage matching sinograms: `<split>_sino_fulldose.h5` and
    `<split>_sino_lowdose.h5` shaped `(N, 2304, 736)` float32.

### C2. New DDPM checkpoints for Mayo-2D

- [ ] **C2.1.** Train `ddpm_mayo2d_unconstrained_final.pt` on the per-patient
    center slices (very small dataset — may need to augment by including
    nearby z-slices too).
- [ ] **C2.2.** Train `ddpm_mayo2d_constrained_final.pt` similarly.

### C3. Search-agent prefix + run

- [ ] **C3.1.** Same as B3 but with `--dataset mayo_ldct_2d` → slug-prefix
    `mayo-ldct-2d-calibrated-tpe-*`.
- [ ] **C3.2.** Submit TPE battery; sync; commit + push.

---

## Dependencies / ordering

```
A1-A4 ──────────── (Mayo helix→fan rebinning, hours-long)
                      │
                      └─── C1.2 (z-aligned extract for 2D)
                                  │
                                  └─── C1.3 (FBP validate)
                                                │
                                                └─── C1.4, C1.5, C2, C3

B1-B4 (Breast-CT) — independent of A, can start immediately
```

## Estimated effort

| Track | Code LOC | Compute (cluster) | Calendar |
|---|---|---|---|
| A | ~400 (rebinning) | ~10 h (10 patients × ~1 h CPU) | 1 day |
| B | ~200 (loader + cfg) | ~12 h (DDPM train ~2-3 h + 13 TPE searches ~10 h) | 1 day |
| C | ~100 (extract + validate) | ~12 h (DDPM small ~30 min + 13 TPE searches ~10 h) | 1-2 days |

Total ~3–4 days to land all three tracks. Tracks A and B can run in
parallel; C waits on A's rebinning output.

## Checklist of artefacts produced

When complete, the dashboard will have THREE intensity-calibrated chart
groups visible:

- `demo-intensity-*` (synthetic phantoms, current state)
- `breast-ct-*` (Sidky 2021 breast challenge, real sinograms)
- `mayo-ldct-2d-*` (Mayo abdomen 2D extract, real helical-rebinned sinograms)

Each with the full 13-solver leaderboard. Diffusion-recon will use a
dataset-matched DDPM in each chart group.
