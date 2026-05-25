---
title: Mayo-LDCT leaderboard
description: Real-helical-data leaderboard. See solver_plan.md methodology.
---

# Mayo-LDCT leaderboard (Wagner split)

AAPM 2016 Low-Dose CT challenge data, helical Siemens SOMATOM AS+,
rebinned to 2-D fan-beam via the `helix2fan` SSR pipeline. Wagner split:

```
Train: L145, L186, L209, L219      (4 patients)
Val:   L277                         (1 patient)
Test:  L014, L056, L058, L075, L123 (5 patients)
```

## Status

🚧 **Geometry validated 2026-05-24; autoresearch not yet started.**

The rebin pipeline has been brought to a working state through this session:
- File-order bug fixed (sort by InstanceNumber, not alphabetic). +0.47 SSIM.
- Half-pitch SSR window enlarged from 23 → 30.6 mm (median-pitch → mean-pitch). +0.14 SSIM.
- FBP up-down + left-right orientation matched to DICOM convention.
- Truth-z mapping detected automatically (patient_z = −source_z + 0 for L014; head-first supine DICOM).
- Z-interpolated truth (eliminates 1.5 mm slice-quantisation error vs FBP).
- Intensity calibration via `evaluate_calibrated` (matches other-dataset convention).

L014 calibrated FBP-vs-truth — **best with 5-mm slab averaging**
(matching truth SliceThickness=5 mm):
- At z-centre slab anchor: **SSIM = 0.9436, PSNR = 37.35 dB,
  RMSE = 0.00068** (SLURM 762112).
- At z=+3.5 mm slab anchor (sign-flip truth mapping): **SSIM =
  0.9445, PSNR = 37.23 dB, RMSE = 0.00069** (SLURM 762115; the
  +3.5 mm shift is essentially noise — within ±0.001 SSIM of the
  centre anchor, confirming the geometry is anchor-insensitive).

Without slab averaging (1-mm thin FBP vs 5-mm thick truth):
SSIM = 0.9105, PSNR = 35.40 dB.

Slab averaging adds +0.033 SSIM and +1.95 dB PSNR — it's the right
apples-to-apples comparison for this dataset. Comparable to Wagner's
reported ≥ 0.85 target. **The geometry is fully validated on the test
patient L014.**

**Reference recon details** (from L014 DICOMs):
- Mayo "Full Dose Images" series uses kernel **B30f** (Siemens
  medium-soft body-imaging kernel; PYRO-NN's `hann` is the closest
  PYRO-NN filter approximation but not identical MTF).
- SliceThickness = 5 mm at 3 mm centre spacing (overlapping slabs).
- 154 truth slices per dose, spanning patient_z ∈ [-482.5, -23.6] mm.

**Remaining residuals** (sub-threshold):
- FFS-`drho` (radial flying focal spot): correction code landed in
  `helix2fan.py:rebin_helical_to_fan(ffs_correct_drho=True)`,
  toggled via env `HELIX2FAN_FFS_DRHO=1`. L014 test rebin in
  flight (SLURM 762117 + 762118 validator). Without correction,
  SSR averages two magnifications (sdd/sod ≈ 1.8245 vs. 1.8170),
  producing faint shadow / ghost edges. Pattern verified in
  `results/breast_debug/L014_ffs_pattern.png`: period-2,
  drho ∈ {0, +5.45 mm} every readout.
- Kernel MTF mismatch (Hann ≠ B30f) shows as faint smoothing
  differences in the diff panel.

Bulk rebin (SLURM 762097, started 2026-05-24) is producing the
fixed-pipeline H5s WITHOUT FFS-drho correction (so the in-flight
agentic seeds remain comparable to the existing TPE numbers). If the
FFS-drho test (762117/762118) shows a clear gain, the bulk rebin
will need to be redone.

## Plan (per solver_plan.md)

Once the rebin is fully blessed:

1. **Re-rebin the remaining 9 patients** (lowdose + 9 missing fulldose)
   with the fixed pipeline.
2. **Per-solver autoresearch** + TPE refinement on the Wagner train/val
   split. Solvers in order of expected promise (from breast-CT
   leaderboard):
   - Learned Primal-Dual (current breast-CT champ at hr=0.91)
   - DD-UNet supervised L2 (hr=0.83)
   - ITNet v3, USwin
   - RAM zero-shot (pretrained — distribution match on Mayo TBD)
   - Hammernik VN
   - DD-BF supervised L2
3. **DDPM training**: two variants per
   `solver_plan.md` Step 4. Constrained uses Train: L145/186/209/219
   labels; unconstrained uses all 10 patients.
4. Diff-recon TPE on both DDPM variants.

| Rank | Solver | Best config | SSIM | PSNR | hr | Source slug |
|---:|---|---|---:|---:|---:|---|
| _TBD_ | _autoresearch + TPE not yet run on Mayo-LDCT_ | | | | | |

## Methodology

See [`/solver_plan.md`](../../solver_plan.md).
