---
title: Mayo-LDCT rebuild plan (2026-06-14)
description: How the Mayo leaderboard + dashboard get repopulated after the reset — geometry/calibration hard-wired, all-slices Wagner, images everywhere.
---

# Mayo-LDCT rebuild plan (2026-06-14)

The old Mayo leaderboard was discarded (scored with the `bg→0` calibration
bug, on a slice subset, before the v3 geometry + truncation-corrected FBP
path was hard-wired). This is the plan to rebuild it correctly. **Nothing
below is executed until the user approves the plan.**

Guiding constraints (from the user):
- Geometry **and** calibration must be **hard-wired** into every Mayo
  experiment — no path may silently use the old geometry or the `bg→0`
  metric.
- LD (and HD) FBP, and every solver, are evaluated on **all slices**
  (≈155 full-dose truth-image slices/patient), not a subset.
- Train/val/test follow the **Wagner split** (patient-level) **on all
  slices**. The **agentic auto-research loop is adjusted** for this; the
  **TPE experiments fully honour it**.
- Leaderboard **and** dashboard report **image results, not just numbers**.

---

## The frozen production path (what "hard-wired" means)

| Stage | Setting | Source of truth |
|---|---|---|
| SSR rebin geometry | v3 + `s_z=1.001665` | `MAYO_LDCT_SSR_DEFAULTS` |
| staged sinos | `staged_helix2fan_v3` | `STAGED_HELIX2FAN_SUBDIR` |
| FBP geometry | Powell sod=595.362 / sdd=1086.803 / ps=0.700857 / du=1.285044 | `FanBeamGeometry.mayo_ldct_fitted()` |
| detector offset | −0.0397 mm | `MAYO_LDCT_DET_OFFSET` |
| truncation | water-cylinder, pad=384 | `MAYO_LDCT_TRUNCATION` |
| per-patient FOV | `ps_eff = 0.700857·truth_ps/0.703125` | truth DICOM PixelSpacing |
| calibration | two-point affine, `bg_target="truth"` | `evaluate_calibrated(..., bg_target="truth")` |

---

## Phase 0 — HD vs LD FBP baseline (all slices) — *in progress*

`scripts/compare_hd_ld_fbp_allslices.py` reconstructs **every** truth slice
of all 10 Wagner patients, both doses, on the frozen path, and scores
calibrated SSIM/PSNR. This defines the **headroom endpoints**:
`score = (metric − LD_FBP) / (HD_FBP − LD_FBP)` per split.

Outputs (→ `results/mayo_debug/allslices_hd_ld/`): per-split oracle/baseline
table, per-patient SSIM-vs-z curves, per-patient representative-slice
montages (GT | HD | LD | diff), and a summary box/bar figure. These become
the baseline rows + first images of the new leaderboard.

## Phase 1 — Hard-wire the path into the training/eval pipeline

Audit (2026-06-14) found two gaps that made old results spurious:
1. `scripts/learned_solver_search_agent.py` scores Mayo with the default
   `bg_target=None` (`bg→0`). **Fix:** route all Mayo calibrated scoring
   through `bg_target="truth"`.
2. Solver FBP projectors don't carry `det_offset_mm` + `truncation`.
   **Fix:** build the Mayo solver projector with `MAYO_LDCT_DET_OFFSET` +
   `MAYO_LDCT_TRUNCATION` (already productionised in `PyronnFanBeamProjector`).

`ddssl_ldct/staged_dataset.py` already uses the Powell FBP geometry +
`staged_helix2fan_v3` (verified). Add a single assertion/log at startup that
prints the active geometry + calibration so no run can drift silently.

## Phase 2 — All-slices Wagner dataset + evaluator

- Stage **all** ≈155 slices/patient per split (train: L145/L186/L209/L219 ≈
  620 slices; val: L277 ≈ 155; test: L014/L056/L058/L075/L123 ≈ 775).
- A shared evaluator scores a solver on **all val slices** → calibrated
  SSIM + headroom vs Phase-0 endpoints, and emits a comparison image
  (GT | recon | diff) at a fixed reference slice + a multi-slice montage.

## Phase 3 — Adjust agentic loop; TPE honours all-slices

- **Agentic 5-min loop (adjusted):** train each iteration on a fixed,
  stratified **train subset** (sampled across z and all 4 train patients)
  so the iteration fits the 5-min budget — but the keep/discard decision
  **always scores all ≈155 val slices**. Document this in the Mayo
  `program.md` so the constraint is explicit and auditable.
- **TPE (fully considered):** each trial trains on the **full Wagner train
  (all slices)** and scores **all val slices**. Longer per-trial budget;
  this is the authoritative search.
- **Final:** best config retrained on full train, evaluated **once** on
  **all test slices**.

## Phase 4 — Repopulate leaderboard + dashboard *with images*

- **Leaderboard** (`docs/leaderboards/mayo_ldct.md`): baseline rows
  (HD/LD FBP) + one row per solver (`Rank | Solver | Variant | params |
  SSIM | headroom | Source | Comparison`), each linking an inline
  comparison image. Per-patient baseline montages embedded.
- **Dashboard:** every rebuild run is registered via the harness
  (`scripts/agent4ct_record.py new-run` / `record`), which writes
  `comparison.png` per iteration and appends to `docs/runs/runs-index.json`
  + `observations.jsonl` — so the live dashboard shows Mayo cards with
  images (Mayo was previously absent from the index).

## Phase 5 — Re-run the solver inventory (the science)

Re-run solvers on the corrected all-slices path via TPE, in priority
order (former front-runners first: LPD, DD-UNet supervised, USwin, ItNet,
diff_recon DCstep; then the rest). Populate leaderboard rows + dashboard
cards as each lands. Absolute SSIM rises vs the old numbers (corrected
calibration); **headroom is recomputed against the new HD/LD endpoints**,
so it is NOT comparable to the discarded headroom values.

---

## Open decisions for the user

1. **Leaderboard reference image:** fixed slice (e.g. L277 central) + a
   per-patient montage? Or a different reference?
2. **Solver scope first pass:** re-run the full 19-solver inventory, or a
   prioritised shortlist (top ~6) to repopulate quickly, then backfill?
3. **Agentic train-subset size:** how many slices per fast iteration
   (e.g. 200 stratified) — trade iteration speed vs. signal.
