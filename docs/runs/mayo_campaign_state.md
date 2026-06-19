# Mayo-LDCT agentic campaign — RESET 2026-06-19 — START HERE

> 🛑 **The previous campaign (`search-20260614-01`, 19 solvers) was DISCARDED and
> purged on 2026-06-19.** It was scored on an **invalid validation metric**, so the
> dashboard, leaderboard, and the search trajectory itself were all built on a bad
> signal. **Start a FRESH search from iter-1** under the corrected protocol below.
>
> **What was wrong (all three fixed in code now):**
> 1. Val scored the FIRST `val_n` (4–30) slices of L277 — top-of-volume boundary
>    slices, near-empty / near-identical — not representative anatomy.
> 2. The metric used the **256px Sidky inscribed-circle FOV** instead of the
>    detector-geometry measurement FOV (237.54 mm → **321 px** for L277), discarding
>    the valid 256→321 px annulus (commit `5ced1ec9`).
> 3. Figures rendered **upside-down** (default imshow origin) and repeated the same
>    boundary slice.

## Corrected protocol for the NEW search (`search-20260619-01`)

**Run-id (NEVER widen the glob):** `search-20260619-01`
**Slug:** `mayo-ldct-claude-agentic-<solver-dash>-search-20260619-01`
**Mandate:** drive all **19 solvers** from **iter-1 → iter-20** of genuine six-box
autoresearch. "2×hr=0 → STOP" is retired; document *why* a ceiling holds.

1. **Val metric (default now — do NOT revert):** score **ALL 214 L277 slices**
   (Wagner val patient), evenly-spaced across the volume if `val_n<214`, with the
   **detector-geometry FOV mask (321 px)**. Implemented in `evaluate_calibrated`
   (gated on `AGENT4CT_DATASET=mayo_ldct_2d`) + `staged_dataset.py`. **No test
   patients in the scored metric** (Wagner split — test leakage forbidden).
2. **HARD 20-minute compute budget** per iter for **train + val-score**. Size
   epochs / n_iter / per-image steps to fit. `mayo_agentic_iter.sbatch
   --time=00:30:00` (20-min budget + headroom). **No per-solver exceptions** (LPD
   too — the old 5h `--time` override is gone). See solver_plan.md.
3. **The val FIGURE is OUTSIDE the budget.** Per-iter `comparison.png` is cheap
   (renders the already-scored recon). The 6-patient leaderboard montage is a
   SEPARATE unbudgeted job: `mayo_showcase.sbatch` → `make_test_showcase.py`
   (covers all 19 solvers; L277-central + 5 test centrals, full-512, row-flipped).
   Test patients appear ONLY in the figure (presentation-only, no leakage).
   **Montage policy (user directive, 2026-06-19): regenerate the valtest montage
   for EVERY solver that records a new iter — INCLUDING regressed / low-scoring /
   below-baseline results. ALL montages go on the dashboard. Do NOT defer or skip
   montages to save QOS throughput (the user accepts the 4-slot contention).** The
   only montages to skip are ones with NO valid recon to render (e.g. a 0.0-SSIM
   OOM/crash iter) — dispatch those once the solver produces a finite recon.
4. **Figures:** display full-512 UNMASKED + row-flipped (Mayo truth is stored
   z-flipped). `make_4panel` picks evenly-spaced rows (never first-n).
5. **Subagents: Opus 4.8.** Every Agent-tool spawn + Workflow `agent()` uses
   `model:"opus"`. Never downgrade.
6. **CT-vision caveat:** do NOT draw conclusions from CT pixels (vision is
   unreliable on CT/sinogram imagery). The metric (val_ssim/headroom on the
   214-slice, geometry-FOV-masked L277) is the source of truth.

### The 19 solvers (SOLVER key → slug-dash)
**10 trainers** (get a valtest montage): `uswin`, `itnet`, `itnet_v2`, `itnet_v3`,
`dual_domain_supervised`, `dual_domain_bilateral_supervised`, `learned_primal_dual`,
`hammernik_2017`, `hammernik_vn`, `wu_2015_trainable`.
**9 inference / per-image:** `naf`, `ram_zeroshot`(→`ram`), `r2gaussian`,
`tv_iterative`, `tv_iterative_supervised`, `dual_domain_n2i` (PER-IMAGE),
`dual_domain_bilateral_n2i` (PER-IMAGE),
`diffusion_recon_dcstep_constrained_mayo_v4`(→`diff-recon-dcstep-constrained-mayo-v4`),
`diffusion_recon_dcstep_unconstrained_mayo_v4`(→`diff-recon-dcstep-unconstrained-mayo-v4`).
N2I are PER-IMAGE (section below — do NOT revert to amortized).

### Dispatch recipe (env `ITER`; `AGENT4CT_DATASET=mayo_ldct_2d` hard-wired in sbatch)
```
ssh lme-bastion 'cd /cluster/maier/Agent4CT && sbatch \
  --export=ALL,SOLVER=<key>,SLUG=mayo-ldct-claude-agentic-<dash>-search-20260619-01,ITER=<n>,CFG_JSON=/cluster/maier/Agent4CT/agentic_cfgs/<cfg>.json \
  cluster/slurm/mayo_agentic_iter.sbatch'
```

### Ready-to-run loop-tick — paste into `/loop` in the NEW session
```
20m Mayo-LDCT agentic campaign tick (run-id search-20260619-01; NEVER widen the glob). GOAL: drive all 19 solvers from iter-1 to iter-20 under the CORRECTED protocol (val = ALL 214 L277 slices + 321px detector-geometry FOV; HARD 20-min train+score budget; val figure OUTSIDE the budget; figures full-512 unmasked + row-flipped). Each tick: (1) ensure README.md + solver_plan.md Step 2 + docs/runs/mayo_campaign_state.md are read this turn; (2) check the 4-slot QOS queue (ssh lme-bastion 'squeue -u maier') + each solver's max iter + best SSIM from docs/runs/mayo-ldct-claude-agentic-*-search-20260619-01/results.tsv; (3) PUBLISH newly-completed iters — targeted per-slug rsync ONLY the campaign dirs (NEVER blanket rsync docs/runs — it re-bloats the index), python3 scripts/rebuild_runs_index.py, git commit + push only if the run count stays sane; (4) for each TRAINER solver (uswin/itnet/itnet_v2/itnet_v3/dual_domain_supervised/dual_domain_bilateral_supervised/learned_primal_dual/hammernik_2017/hammernik_vn/wu_2015_trainable) with a NEW iter, regen its valtest montage via mayo_showcase.sbatch (SEPARATE job — NOT in the 20-min budget); (5) for EVERY free QOS slot, spawn ONE general-purpose subagent (model: opus — Opus 4.8, never a cheaper tier) to set up the next iter for the most-behind solver not yet at iter-20 and not running — the subagent reads README + solver_plan.md + this file + the solver's results.tsv, does the six-box (NUMBERS not CT pixels), SIZES the config so train+score fits the 20-min budget, and dispatches via mayo_agentic_iter.sbatch (env ITER, AGENT4CT_DATASET=mayo_ldct_2d hard-wired); the hard-wired config is staged_canonical data, v3 mayo_ldct_fitted geometry, per-sample-ps, bg_target="truth", val=ALL 214 L277 slices, metric=321px geometry FOV (DEFAULT — do not pass fov=False); NEVER edit shared ddssl_ldct/ modules; dual_domain_n2i / dual_domain_bilateral_n2i stay PER-IMAGE (warm-start pre-pass + per-scan fit; NEVER revert to amortized); (6) when all 19 solvers reach iter-20, STOP the loop (CronDelete) + final dashboard publish. Report one line: solvers-at-20/19, jobs running, figures published this tick.
```

> **NOTE:** the per-solver knob insights further down this file are from the
> DISCARDED `search-20260614-01` run (boundary-slice metric). Treat them as loose
> hints about solver *behavior*, **NOT** as validated results — re-establish
> everything under the corrected metric.

### iter-1 findings under the CORRECTED metric (`search-20260619-01`, live)
These are validated under the corrected 214-slice / 321px-FOV metric. iter-2+
subagents MUST apply them:
- **hammernik_2017 iter-1 "0.0" was a CUDA OOM, NOT divergence.** SLURM err log:
  `torch.cuda.OutOfMemoryError` in `_rho_prime` (RBF-bump loop) at 72 s, `batch_size=4`
  on a 16 GB RTX 5000. **Fix for its iter-2:** `batch_size=2` + `vn_checkpoint=true`
  (gradient-checkpoint each unrolled step) + `grad_clip=1.0` + `vn_dc_norm=true`,
  exactly like the hammernik_vn iter-1 config that ran fine.
- **itnet v1 (0.441) and v2 (0.455) are stuck far below itnet_v3 (0.896).** The
  *pretrain-denoiser-then-DC* path under-converges in the 20-min budget; v3's
  END-TO-END unrolled training is what works. itnet/itnet_v2 iter-2 should move
  toward end-to-end training or accept they are budget-limited.
- **dual_domain_bilateral_supervised: raw SSIM 0.919 (highest) but hr only 0.028.**
  Not a bug — SSIM rewards the bilateral filter's smoothing; the ranked headroom
  metric (calibrated, RMSE-sensitive) does not credit it. Rank by hr.
- **iter-1 headroom leaderboard:** dual_domain_supervised 0.276 > itnet_v3 0.268 >
  uswin 0.173 > tv_iterative 0.133 > wu_2015_trainable 0.051 > dd_bilat_sup 0.028 >
  {LPD, itnet, itnet_v2} 0.000. (ram/naf/hammernik_vn iter-1 pending.)

---

## N2I solvers are PER-IMAGE self-supervised (2026-06-18 fix — do NOT revert)

`dual_domain_n2i` and `dual_domain_bilateral_n2i` were re-implemented from
AMORTIZED (one model trained on the 4 train patients, then forward-applied to
val/test — **wrong** for N2I) to genuine **per-image** Noise2Inverse
(commit `1f4c3394`, solver files only — `ddssl_ldct/training.py` untouched, its
`training_step`/`predict` were already per-sinogram-pure):
  1. short **warm-start pre-pass** (amortized N2I on the train split, GT-free) → U-Net init;
  2. **per-scan fit** — each val/test image is optimized on its OWN measurements
     (angular-subset split), no clean GT, then scored on the SAME L277 val + 5 test
     scenes via `evaluate_calibrated` (scoring/loading contract unchanged).
Per-image cfg knobs (the search tunes ONE per iter): `warm_start` (default True),
`pretrain_epochs`, `n_iter` (per-scan Adam steps), `grad_clip=1.0` (mandatory —
the 2304-view FBP adjoint overflows otherwise), `val_n`, `outer_wall_s`/`per_scene_s`.
`training_scheme = noise2inverse_per_image_warmstart_selfsup`. GPU-smoke verified
(both solvers, finite val_score + figures). **NEVER revert to amortized
train-then-forward.** The 16 prior amortized-N2I runs (mayo 7 + breast 6 + demo 3)
were purged 2026-06-18; supervised dual-domain runs were kept. N2I was restarted
per-image on all 3 datasets: mayo under `search-20260614-01` (driven by this loop);
breast_ct + demo_dl under fresh `search-20260618-01` via the generic
`claude_agentic_one_iter.sbatch` (env `ITER_N`, `AGENT4CT_DATASET=breast_ct`/`phantoms`,
config env `DD_CONFIG_PATH`). N2I is structurally FBP-bounded on dense-view
breast/demo — expect hr near the LD-FBP baseline there.

**Per-image N2I knob insights (confirmed across runs, 2026-06-18/19):**
- **U-Net N2I OVERFITS each scan** — the per-scan `_fit_one_scene` loop has no
  early-stop, so raising `n_iter` does NOT help and often HURTS: demo-n2i
  0.4785→0.4293 @n_iter1200, breast-n2i 0.9553→0.9407 @n_iter600, mayo-n2i flat.
  The lever is per-scan **lr↓** (anchor near the warm-start init): demo 0.4785→0.4880
  (lr1e-4), mayo 0.9508→0.9541 (lr3e-4), breast 0.9553→0.9593 (lr1e-4) — confirmed
  on all 3 datasets. `unet_c↑` is NOT the lever (mayo c16→c32 flat at 0.9508).
- **`warm_start` is LOAD-BEARING-POSITIVE for the U-Net N2I (do NOT turn it off).**
  Confirmed cross-substrate (2026-06-19): demo-n2i `warm_start`False 0.5081→0.4486
  (−0.06), mayo-n2i False 0.9562→0.9414 (−0.015). The amortized warm-start init is
  the anchor the per-scan lr↓ regularization relies on; without it each per-scan fit
  lands in a worse basin. (demo-n2i also has a large init-driven noise floor: a seed
  flip alone swung 0.5081→0.4203, so demo-n2i SSIM reads are noisy.) Don't spend
  future iters re-testing `warm_start`=off — it's settled on all tested datasets.
- **Bilateral (6-param) BEATS the U-Net** on these substrates: it physically can't
  overfit one scan (demo-bilat 0.77 hr+0.33 vs demo-n2i 0.49). For demo bilateral
  `img_n_bf↑` (image-domain depth) is the working lever (1→3→5: 0.67→0.75→0.77).
- **Bilateral over-smoothing lever is substrate-specific (and `img_sr` DIRECTION
  flips between substrates):** Mayo (2304-view) — the IMAGE-domain σ is inert and
  `proj_sr↓` is the lever. 128-view (breast/demo) — the IMAGE-domain `img_sr` is the
  lever (not proj; proj-tuning HURTS, breast PSNR 38.17→37.90), BUT its sign is
  substrate-specific: **breast wants `img_sr`→0.0005** (tight = edge-preserving on
  real-μ tissue; 0.9723→0.9732 with img_n_bf=3), while **demo (synthetic phantoms)
  wants `img_sr` LOOSE at 0.02** — tightening to 0.0005 OVER-sharpens and DROPS hr
  (0.3335→0.1998, PSNR 20.18→18.60). For demo bilateral the lever is `img_n_bf`↑ at
  loose img_sr (1→3→5: hr 0.28→0.33→0.33). Do NOT blindly tighten img_sr on demo.

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
- **itnet_v2 RESET 2026-06-18** (prior run overfit: val 0.9714 but valtest6 only
  0.88). The fresh restart re-hit a **CONFIG-DEFAULT BUG**: `residual_learning=True`
  double-applies the residual — `SmallUNet.forward` already returns `x−body(x)`
  (identity at init), then v2 re-adds `denoiser(x)` at eval → ≈2x, which the k=3 DC
  can't recover → stuck ~0.34, and lr changes make it WORSE (broken-target signature,
  not under-training). **FIX: `residual_learning=False`** (matches the working v1/v3
  recipe) — confirmed **0.3429→0.5373 @iter-3**. Now under-trained at ep8; raise
  pretrain_epochs to climb back toward ~0.97. KEEP residual_learning=False.
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
**Rsync trap (2026-06-18):** NEVER blanket `rsync docs/runs cluster→laptop`. The
cluster holds ~100 historical/scratch run dirs that are deliberately pruned from
the dashboard; a blanket rsync re-imports them all and bloats the index 78→197.
Publish by rsyncing ONLY the campaign slugs:
`for s in $(ssh lme-bastion 'cd /cluster/maier/Agent4CT && ls -d docs/runs/mayo-ldct-claude-agentic-*-search-20260614-01'); do rsync -az lme-bastion:/cluster/maier/Agent4CT/$s/ $s/; done`
then `rebuild_runs_index.py` — sanity-check the count stays ~78 before commit
(`git checkout -- docs/runs` if it jumped). Breast/demo N2I (search-20260618-01)
publish the same way, per-slug.

When the top-3 (itnet/uswin/itnet_v2) all reach iter-20, do ONE comprehensive
publish: regenerate test-showcase figures (`make_test_showcase.py`, idempotent —
`SHOWCASE_FORCE=1` to refresh changed best-iters) then
`bash scripts/publish_mayo_wave.sh "<msg>"`. Current published leaderboard still
shows USwin champion — **stale**, itnet 0.9726 is the real champion.
