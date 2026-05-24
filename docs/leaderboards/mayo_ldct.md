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

L014 calibrated FBP-vs-truth: **SSIM = 0.9105, PSNR = 35.40 dB,
RMSE = 0.00085** (job 762096). Comparable to Wagner's reported ≥ 0.85
target. The geometry is fully validated on the test patient L014. Now
need to re-rebin the remaining 9 patients (8 fulldose + 10 lowdose)
with the fixed pipeline before starting autoresearch.

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
