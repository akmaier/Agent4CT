# Mayo-LDCT agentic campaign — live state & resume handoff

**Run-id (NEVER widen the glob):** `search-20260614-01`
**Mandate:** every one of the **19 solvers** reaches **iter-20** of genuine
six-box agentic autoresearch (read prior result+image → name failure mode →
change ONE knob → named hypothesis → dispatch). The "2×hr=0 → STOP" rule is
**retired** — do not bail capped/structural-negative solvers; drive them to 20
with genuine ablations (capacity / data / loss / seed / grad_clip probes that
document *why* the ceiling holds).

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
- **Event-poll** (background bash, baseline 4, wakes on slot free) + **cron
  `f2142633`** (`7,22,37,52`, agentic, fills idle slots — only acts when I leave
  slots open, so it's a safety net while I drive).
- Live per-solver state (max iter + best SSIM):
  ```
  for d in docs/runs/mayo-ldct-claude-agentic-*-search-20260614-01; do
    awk -F'\t' 'NR>1{if($3+0>b){b=$3+0;bi=$1}m=$1}END{print FILENAME,m,b,bi}' "$d/results.tsv"; done
  ```

## The 19 solvers
**Resumes (10):** uswin, itnet, itnet_v2, itnet_v3, dual_domain_supervised,
dual_domain_bilateral_supervised, learned_primal_dual, hammernik_2017,
hammernik_vn, wu_2015_trainable.
**Onboards (9), do in this order, diffusion LAST:** `ram`✓, `tv_iterative`✓,
`naf`✓, `r2gaussian`✓(wired, scp'd; iter-1 pending slot), tv_iterative_supervised,
dual_domain_n2i, dual_domain_bilateral_n2i, ddpm, diffusion_recon.
**Wired: 5/9** (ram/tv/naf/r2g done w/ results; tv_iterative_supervised wired
per-sample-ps + scp'd, iter-1 pending). Remaining 4: dual_domain_n2i,
dual_domain_bilateral_n2i (N2I half-angle + per-sample-ps), ddpm, diffusion_recon. ram/tv/naf all
confirmed structural-negative (<LD-FBP): ram 0.9329, tv 0.9497, naf 0.864 — foreign
/per-scene methods don't beat FBP on Mayo LDCT. Remaining 5: tv_iterative_supervised
(trains, per-sample-ps), N2I pair (HALF-ANGLE projectors), ddpm+diffusion_recon (train).

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

## Per-solver knob insights / ceilings (as of this writing)
- **itnet 0.9726 = CHAMPION, plateaued** (ep104≈ep120). k=1 (fewer DC steps win
  on noisy LDCT); epochs dominant lever, now saturated.
- **uswin 0.9709, still climbing** with epochs (36→48 lifted 0.9689→0.9709).
- **itnet_v2 0.9714 ceiling** (ep72 optimal, k=1; ep88/seed/lr-0.6× all worse).
- **itnet_v3 0.9657**, **dual_domain_supervised 0.9626**.
- **dd-bf (bilateral) capped ~0.950** — lr/epochs/img_n_bf/img_kernel/proj_n_bf
  all fail to beat iter-1; testing train_n×2 then structural ablations.
- **hammernik_2017 0.9484**, **hammernik_vn 0.909** (ep8 best; testing ep16).
- **wu_2015_trainable ~0.910** (n_outer=1 mandatory: 0→0.50 collapse,
  2→oversmooth; testing ep24→40).
- **learned_primal_dual climbing** 0.836→0.885 (lr=3e-4); epochs 10→30 in flight.
- **RAM = STRUCTURAL NEGATIVE** — caps 0.9329 (blend=0.5), hr=0, below LD-FBP;
  foundation model doesn't transfer zero-shot. `ram_finetune` is a no-op without
  `ram_finetune_epochs>0`. File the negative verdict once it completes 20.
- **tv_iterative** onboard baseline 0.9497 hr=0 (default TV over-smooths; tuning
  lambda down).

## Publish trigger
When the top-3 (itnet/uswin/itnet_v2) all reach iter-20, do ONE comprehensive
publish: regenerate test-showcase figures (`make_test_showcase.py`, idempotent —
`SHOWCASE_FORCE=1` to refresh changed best-iters) then
`bash scripts/publish_mayo_wave.sh "<msg>"`. Current published leaderboard still
shows USwin champion — **stale**, itnet 0.9726 is the real champion.
