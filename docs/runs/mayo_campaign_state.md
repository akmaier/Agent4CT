# Mayo-LDCT agentic campaign — live state & resume handoff

**Run-id (NEVER widen the glob):** `search-20260614-01`
**Mandate:** every one of the **19 solvers** reaches **iter-20** of genuine
six-box agentic autoresearch (read prior result+image → name failure mode →
change ONE knob → named hypothesis → dispatch). The "2×hr=0 → STOP" rule is
**retired** — do not bail capped/structural-negative solvers; drive them to 20
with genuine ablations (capacity / data / loss / seed / grad_clip probes that
document *why* the ceiling holds).

## Data provenance — `staged_canonical` (READ BEFORE touching Mayo data)

The `mayo_ldct_2d` training loader (`ddssl_ldct/staged_dataset.py`) reads
**`data/mayo_ldct/staged_canonical/`**: `{split}_truth.h5` (dataset key
`"truth"` + a per-slice `"ps"` array = `ps_eff = 0.700857·native_ps/0.703125`)
and `{split}_sino_{lowdose,fulldose}.h5` in the **canonical frame**
(`roll + u-flip + slab`, per patient, so a uniform angle_start=0 FBP lands on
truth). It is built **only** by **`data/stage_mayo_canonical.py`** from the
surviving `raw/` (truth) + `staged_helix2fan_v3/` (per-patient v3 sinos):
```
python data/stage_mayo_canonical.py --force --validate --subdir staged_helix2fan_v3
```
(sbatch wrapper `cluster/slurm/restage_canonical_v3.sbatch`; `--validate` FBPs
the val split per-sample and prints LD-FBP SSIM, expect **~0.81**). Do **NOT**
rebuild it with `fetch_mayo_ldct.py` (writes key `"image"`, no `"ps"`, shuffled)
or `stage_mayo_sinos.py` (older non-canonical packing) — both silently produce
data the loader mis-reads. Geometry is **v3** (Powell-fitted `mayo_ldct_fitted`:
sod 595.362 / sdd 1086.803 / det_spacing 1.285044 / ps 0.700857).

**FOV / masking (corrected 2026-06-18, audited):**
- **METRIC** (SSIM/RMSE/headroom in `evaluate_calibrated`): keeps its DEFAULT
  inscribed-256px FOV mask, applied **symmetrically to BOTH pred and truth**
  (`metrics.py`). This is INTENTIONAL and fine per the user ("the metric is ok").
  Solvers call `evaluate_calibrated` with NO `fov=` arg → default mask. Do NOT
  pass `fov=False`.
- **DISPLAY / figures**: **UNMASKED**. `evaluate_calibrated` now returns the
  UNMASKED calibrated `pred_cal`/`baseline_cal` (metric uses local masked copies),
  so figures show the FULL 512² recon + GT — no circular mask. ("mask=False for
  display.")
- **Result figure** = `valtest_showcase.png` (L277 val + 5 test patients, central
  slice each, `GT|FBP|recon|diff`, full view) via `make_test_showcase.py`
  (`AGENT4CT_SHOWCASE=valtest`); the dashboard builder prefers it as the result image.
- The detector-geometry measurement FOV `R = SOD·sin(atan(0.5·n_det·det_spacing/SDD))`
  ≈237.5 mm is the eventual "correct" FOV but is **DEFERRED**. Never derive the FOV
  from `ReconstructionDiameter/PixelSpacing` (the 256px inscribed circle is a recon
  property, not the scanner FOV).

## Dispatch protocol (cluster: `ssh lme-bastion`, `cd /cluster/maier/Agent4CT`)
- **Helper** (hardened: rm-first so a failed write can't leak a stale cfg, no
  f-string backslashes, post-write assert):
  ```
  python3 scripts/agentic_redispatch.py SOLVER DASH ITER BASE_ITER PATCH_JSON RATIONALE
  ```
  `SOLVER` = SOLVER_MAP key, `DASH` = key.replace("_","-"), `BASE_ITER` = iter
  whose cfg_full to copy (or `default` for solver CONFIG defaults). Writes
  `agentic_cfgs/mayo_<SOLVER>_iter_<NN>.json`, sbatches `mayo_agentic_iter.sbatch`.
- **4-slot QOS.** Keep all 4 slots saturated with the **slowest** solvers
  (they're the bottleneck); squeeze fast ones (tv, ram) into brief gaps.
- **Round-robin the most-behind** solver not currently running — do NOT
  over-feed the leaders (all must reach 20).
- **Driver = the 20-min loop cron `16f09972`** (session-only, `*/20`). Each tick:
  check the queue + per-solver iters, PUBLISH newly-completed iters to the
  dashboard (rsync docs/runs cluster→laptop → rebuild_runs_index.py → commit/push),
  and spawn ONE general-purpose subagent per FREE QOS slot (most-behind solver) to
  six-box + dispatch the next iter. (The old event-poll loop + cron `f2142633` are
  RETIRED/DELETED — ignore any lingering references to them.)
- Live per-solver state (max iter + best SSIM):
  ```
  for d in docs/runs/mayo-ldct-claude-agentic-*-search-20260614-01; do
    awk -F'\t' 'NR>1{if($3+0>b){b=$3+0;bi=$1}m=$1}END{print FILENAME,m,b,bi}' "$d/results.tsv"; done
  ```

## The 19 solvers
**Resumes (10):** uswin, itnet, itnet_v2, itnet_v3, dual_domain_supervised,
dual_domain_bilateral_supervised, learned_primal_dual, hammernik_2017,
hammernik_vn, wu_2015_trainable.
**Onboards (9) — ALL WIRED + driving (2026-06-16):** `ram`✓, `tv_iterative`✓,
`naf`✓, `r2gaussian`✓, `tv_iterative_supervised`✓, `dual_domain_n2i`✓,
`dual_domain_bilateral_n2i`✓, `diffusion_recon`✓ (con+uncon variants; "ddpm" =
the reused DDPM prior, no separate recon solver). **ALL 19 solvers now have
search-20260614-01 iters.**
- **N2I pair** (`dual_domain_n2i`, `dual_domain_bilateral_n2i`): onboarded by
  writing the per-sample HALF-ANGLE projector swap. `DualDomainPipeline`'s
  `training_step`/`predict` use ONLY `self.R_half`, so swap a per-ps half-angle
  cache `{k: PyronnFanBeamProjector(v.geom.split_angles()[0])}` into `pipe.R_half`
  per sample (bs=1 train, chunk=1 val); full cache feeds the LD-FBP baseline.
  Both structural-negative (hr0): UNet-N2I caps **0.9501** (ep32; ep64 overfits),
  bilateral-N2I **0.9507** (6 params; epochs flat). Half-view info loss < full-view FBP.
- **diffusion_recon** (DPS+DC-step vs REUSED Mayo DDPM v4 ckpts — NO retrain;
  `checkpoints/ddpm_mayo_{constrained,unconstrained}_v4.pt`): **had a per-sample-ps
  BUG** — built a single canonical-ps (0.700857) projector, mis-scaling L277's
  native 0.74 by ~5% → baseline_PSNR 19.65 (vs ~36) AND corrupt DPS physics. FIXED
  2026-06-16 with the single-val-ps probe (val is one patient). Pre-fix: con hr0.06,
  uncon hr0 (steps 100→200 didn't matter). Post-fix validation in flight.
- **structural-negatives confirmed <LD-FBP:** ram 0.9393 (blend0.3/factor0.7),
  tv 0.9528 (clip0.08 CLEARS FBP hr0.20!), naf 0.911, r2g 0.916, tv_sup 0.863
  (lambda_init0.02), N2I pair ~0.95. Per-scene/foreign/self-sup don't beat FBP.

### Onboard wiring recipe
- **Val-only solvers** (no training; ram, tv, naf, r2g): val split = single
  patient **L277**, native ps ≈ **0.73979**. `geometry_overrides` sets the
  canonical 0.700857 → mis-scales L277 ~5%. Fix = inline the **single-val-ps
  probe** before building geom (copy verbatim from `solver_ram.py` lines ~288-311
  / `solver_tv_search.py`). Probe prints `Mayo val ps -> pixel_spacing=0.73979`.
  Keep the probe **inlined per-solver** (no shared staged_dataset.py edit — that
  module is imported by every running job; a bad edit kills the whole queue).
- Edit solver locally → `python3 -m py_compile` → `scp` to cluster (cluster is
  **not** a git checkout, it's rsync-synced).
- **Trainers** that span the 4 train patients (4 ps) need real per-sample-ps.
  **Pattern (from `solver_itnet.py` lines 177-216, the template):**
  ```python
  # train_ps/val_ps come from load_val_split(..., return_ps=True)
  from ddssl_ldct.staged_dataset import mayo_per_sample_setup, mayo_per_sample_fbp
  per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)
  # _projs = {ps_float: PyronnFanBeamProjector}; _trk[i]/_vrk[i] = ps of sample i
  if per_ps:
      train_fbp = mayo_per_sample_fbp(_projs, _trk, train_noisy, cfg["image_size"])
      val_fbp   = mayo_per_sample_fbp(_projs, _vrk, val_noisy,   cfg["image_size"])
  # in train AND val loops: chunk=1, and before each sample i swap the model's
  # projector: model.proj = _projs[float(_trk[i])]  (itnet does itnet.projector=...)
  ```
  So `tv_iterative_supervised` (UnrolledTV uses self.proj in forward at lines
  130-133) needs: build_dataset must return ps (call load_val_split w/ return_ps),
  then per-sample setup + swap `model.proj` per sample, chunk=1, in both train and
  val. The single-val-ps PROBE does NOT work for trainers (Mayo train sinos are
  REAL at mixed native ps; a single-ps projector mismatches the DC term).
- **N2I** (dual_domain_n2i, dual_domain_bilateral_n2i) needs HALF-ANGLE projectors
  for the view-split (read `DualDomainPipeline.training_step` in
  `ddssl_ldct/training.py` first) ON TOP of per-sample-ps.
- **ddpm/diffusion** need training (solver_plan Step 4), constrained+unconstrained.

## Per-solver knob insights / ceilings (updated 2026-06-16)
- **itnet 0.9729 = CHAMPION, iter-20 DONE.** k=1 (fewer DC steps win on noisy
  LDCT); epochs dominant (ep104 peak); itnet_alpha + unet_c neutral at k=1;
  **train_n 200→300 lifted 0.97256→0.97294 (mildly data-limited).**
- **itnet_v3 0.9683 = iter-20 DONE** (ep40 peak; ep48 & lr3e-4 collapse; v3 is
  unstable, grad_clip=0.5 no help; val60 honest 0.9638).
- **itnet_v2 0.9714 ceiling** (iter-19; ep72 optimal, k=1; ep88/seed/lr-0.6×/
  unet_c=32/patience/**train_n=300 all worse** — NOT data-limited unlike v1).
- **uswin 0.9709** (iter-14; epochs lever, still the path to 20).
- **dual_domain_supervised 0.9626** (lambda_neg=0.1 best; capacity/lr/seed flat).
- **dd-bf (bilateral-sup) capped 0.9502** — every scalar flat (lr/epochs/kernels/
  proj_n_bf/seed/loss/grad_clip); 6-param structural ceiling.
- **hammernik_2017 0.9484**; **hammernik_vn 0.9203** — grad_clip 1.0→0.5 BROKE the
  0.9158 early-stop plateau; lr5e-4 stays optimal (8e-4 worse even at gc0.5).
- **wu_2015_trainable 0.9135** (n_outer=1 mandatory).
- **learned_primal_dual 0.9641** (lpd_iters=7, lr=5e-4 PEAK; 3e-4=0.959, 6e-4=0.65
  anomalous collapse, 7e-4=0.9638; huge arc from 0.836). ⚠️ **iter-12 ep45 TIMED OUT
  at BOTH the 1h (764506) and 2h (764562) walls** — ep45 @ lpd_iters=7 is too slow.
  Re-dispatch with **ep≤35** (iter-9 ep30 ran in <1h) OR pass a **4h wall**
  (agentic_redispatch.py TIME_OVERRIDE `04:00:00`). Do NOT retry ep45 at ≤2h.
- **STRUCTURAL-NEGATIVES (hr0, <LD-FBP), driving to 20 to document:** ram 0.9393
  (blend0.3, factor0.7; finetune hurts), tv_iterative **0.9528 CLEARS FBP hr0.20**
  (clip_max=0.08 preserves bone — the lever; 200 iters optimal), tv_sup 0.863
  (lambda_init0.02 peak), naf 0.911, r2g 0.916, N2I-UNet 0.9501 (ep32),
  N2I-bilateral 0.9507.
- **diffusion_recon:** reused v4 DDPM prior; PER-SAMPLE-PS BUG FIXED 2026-06-16
  (see onboards above). Constrained (hard-DC) clears FBP, unconstrained doesn't;
  post-fix numbers pending.

## Publish trigger
When the top-3 (itnet/uswin/itnet_v2) all reach iter-20, do ONE comprehensive
publish: regenerate test-showcase figures (`make_test_showcase.py`, idempotent —
`SHOWCASE_FORCE=1` to refresh changed best-iters) then
`bash scripts/publish_mayo_wave.sh "<msg>"`. Current published leaderboard still
shows USwin champion — **stale**, itnet 0.9726 is the real champion.
