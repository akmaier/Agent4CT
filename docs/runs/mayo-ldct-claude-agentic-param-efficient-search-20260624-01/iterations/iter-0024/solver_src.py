"""Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-24).

iter-24 PIVOTS AWAY FROM TRAINING-EFFICIENCY (iters 21-23 settled it is NOT
cheaply fixable: denoiser-pretrain FROZE x2; ordered-subsets gave NO per-epoch
speedup -- iter-23 still ~8 epochs at the wall, hr 0.2401 < iter-7's 0.2515) AND
ATTACKS HEADROOM-PER-PARAM at the SAME ~8-epoch COUPLED training: a ROTATION-
EQUIVARIANT / STEERABLE FoE analysis bank. CT noise/structure has NO preferred
orientation, so a steerable filter bank should denoise more efficiently per
trainable weight than iter-7's 24 FREE unstructured 7x7 kernels.

iter-24 SIX-BOX (NUMBERS) -- steerable / rotation-equivariant FoE analysis bank
------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = psnr 34.08 dB):
  - iter-7 (CHAMPION, BASE): FoE 1,921p (24 FREE 7x7 analysis kernels = 24*49 =
    1,176 + 24*31 = 744 RBF synth + 1 alpha), K=5 single-phase PARTIAL cosine 5e-3
    -> val hr 0.2515, ssim 0.9058, psnr 36.59, val_rmse 7.40e-4. test 0.1852.
  - iter-8..16: capacity/depth/stage/LR/kernel-geometry knobs all mapped at the
    fixed model + single-phase trainer -> 0.2515 is the ceiling; nf24/k7 is the
    FoE geometry optimum (iter-16 k9/nf17 iso-param REGRESSED to 0.2404).
  - iter-18/19: HALVING the bank (nf 12/6) walks the param/headroom frontier DOWN
    (0.2334 @961p, 0.1690 @481p) -> capacity helps at the top.
  - iter-21/22 (denoiser-pretrain): FROZE x2, hr 0 -> the reg only learns COUPLED.
  - iter-23 (ordered-subsets S=576 views): NO per-epoch speedup (still ~8 epochs),
    hr 0.2401 < iter-7 -> view-count is NOT the training-wall bottleneck (per-step
    cost is the FoE conv + K=5 unrolled autograd graph). Training-efficiency is
    NOT cheaply fixable; the ceiling is closer to CAPACITY-bound than starvation.
FAILURE MODE addressed (iter-24): iter-7's 24 FREE 7x7 analysis kernels are an
  UNSTRUCTURED basis -- the bank has NO inductive bias that CT noise/edges are
  orientation-agnostic, so to cover edges at many angles it must spend separate
  weights learning each rotated copy. The per-param denoising power is capped by
  this orientation-blind parameterization, NOT by the param COUNT (iter-16/18/19
  confirmed nf24/k7 is the geometry/capacity optimum at this budget).
CHANGE (iter-24, ONE knob -- the ANALYSIS-FILTER PARAMETERIZATION; everything
  else iter-7 byte-for-byte): reg_type "foe" -> "steerable_foe". The nf=24
  analysis filters are no longer 24 FREE 49-vectors; each is SYNTHESIZED as a
  learned linear combination of a SHARED, FIXED STEERABLE BASIS:
    K_f = sum_b  c_{f,b} * B_b          (c learned; B_b fixed 7x7 buffers)
  The basis B is a separable RADIAL x ANGULAR (circular-harmonic) frame:
  n_rad=7 Gaussian-windowed radial profiles x angular orders m in {0..M=3}
  (order m=0: 1 radial-symmetric atom; m>0: cos(m.theta) AND sin(m.theta) phase
  pair) => Nb = n_rad*(1 + 2*M) = 7*(1+6) = 49 steerable atoms. Because every
  filter lives in the span of COMPLETE circular-harmonic orders, an in-plane
  rotation of the input maps the filter responses among themselves (each order m
  rotates by phase m.theta) -- the bank is ROTATION-STEERABLE by construction:
  ONE learned filter implicitly represents its whole orientation orbit.
SPEND CHOICE -- option (a) ISO-PARAM, rotation-equivariant (NOT fewer-param):
  size the basis so analysis params match iter-7 EXACTLY. n_rad=7, M=3 => Nb=49
  (a COMPLETE steerable 7x7 frame; 49 = the full 7x7 DOF, so the span is the whole
  kernel space but expressed in the orientation-organized harmonic basis). At
  nf=24: analysis = 24*49 = 1,176 params -- BYTE-IDENTICAL to iter-7's analysis
  budget. The win is NOT param count but CONDITIONING / INDUCTIVE BIAS: training
  optimizes coefficients in an orientation-organized frame where the gradient
  shares signal across orientations, vs 24 unstructured 49-vectors. Chosen over
  (b) fewer-param because iso-param is the CLEAN test of "does equivariance help
  per param" with NO param confound (a fewer-param point would conflate
  equivariance-helps with fewer-params-hurts -- and iter-18/19 already showed
  fewer params hurts).
EXACT PARAM COUNT (iter-24 vs iter-7): analysis 24*49=1,176 (steerable coeffs c)
  + RBF synthesis 24*31=744 + 1 scalar alpha = 1,921 TOTAL -- EXACTLY iter-7.
  The 49 steerable basis atoms B_b are FIXED (registered buffers), 0 trainable.
STABILITY (why steerable-FoE stays in the iter-7 basin):
  (1) PARAMS UNCHANGED (1,921). Same nf=24, same RBF nb=31, same K=5, same single
      tied scalar alpha, same trainer (plain Adam, peak lr 5e-3, PARTIAL cosine
      T_max=16, grad_clip=1.0, bs=1, per-sample-ps, full-view DC). ONLY the
      analysis filters' parameterization changes (free 49-vector -> 49 steerable
      coeffs over a fixed complete frame).
  (2) ZERO-INIT RBF SYNTHESIS (foe_rbf_init_std=0.0, byte-for-byte iter-7) =>
      rho'(.)==0 => reg(x)==0 at init REGARDLESS of the analysis coeffs, so the
      seed is the EXACT clean GD+DC scheme of iter-7. Training lifts reg off zero.
  (3) The steerable analysis coeffs are randomly initialised at a std chosen so
      the EFFECTIVE 7x7 kernels have the SAME element-wise std as iter-7's
      filter_init_std=0.05 free kernels (the basis atoms are L2-normalised, so
      coeff_std = 0.05 reproduces iter-7's per-element analysis-filter scale) --
      the analysis bank starts statistically identical to iter-7, only organized
      in the steerable frame. NO pooling, NO extra stage, NO added depth.
HYPOTHESIS: rotation-equivariant filters give more denoising power per param ->
  val-RMSE below iter-7's 7.40e-4 at the SAME 1,921 params, SAME ~8-epoch coupled
  training -> hr > 0.2515. A NULL result (hr ~= 0.2515) cleanly says the free 7x7
  FoE bank was ALREADY near-optimal per param (the orientation orbit was already
  being learned implicitly by the free kernels), decisively settling the
  per-param-efficiency question. A regression would say the steerable frame
  mis-conditions the coupled optimisation. PREDICTED val hr ~0.25-0.28.
DEAD ENDS (do NOT reintroduce): projection-free/denoiser pretrain (FROZE x2);
  ordered-subsets/view-subsample (NO speedup); capacity^ via raw width (diverges);
  K>5; momentum/per-step-alpha; separate learned-init stage; peak LR>5e-3;
  constant LR; full anneal to 1e-5; bilateral-alone.

----------------------------------------------------------------------------
iter-23 (the prior BASE) header is preserved below for the campaign record.
----------------------------------------------------------------------------
Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-23).

iter-23 ABANDONS THE TWO-PHASE / DENOISER-PRETRAIN AXIS (idea A is DEAD) AND
ATTACKS THE TRAINING-COST OF THE *COUPLED* DC UNROLL DIRECTLY via STOCHASTIC-
VIEW-SUBSET (ORDERED-SUBSETS) DC TRAINING. The MODEL is iter-7 byte-for-byte
(FoE nf24/k7/nb31 = 1,920 reg + 1 scalar alpha = 1,921 params, K=5 tied prox+DC,
no momentum, no per-step-alpha, no learned-init); the trainer is iter-7's exact
SINGLE-PHASE partial-cosine (two_phase=False -- the pretrain is dead). The ONLY
change is: DURING TRAINING the DC term subsamples the projection VIEWS.

WHY THE PRETRAIN AXIS IS DEAD (iters 21-22, do NOT retry):
  The FoE reg CANNOT be trained as a standalone projection-free denoiser. The
  Phase-A objective D(x)=clamp(x-alpha*reg(x),0,clip_max) supervised LD-FBP->truth
  FROZE the denoise_loss at the identity-floor (~1.9e-4) with BOTH zero-init RBF
  (iter-21) AND a non-zero re-seeded RBF (iter-22) -- hr 0 both times. The reg
  only learns COUPLED through the DC unroll (the projection RESIDUAL Rᵀ(Rx-g) is
  what drives the gradient that shapes the bank). So PnP / denoiser-pretrain is
  off the table here; the training-efficiency fix MUST make the COUPLED
  end-to-end DC training cheaper WITHOUT decoupling the reg from the physics.

ROOT PROBLEM (recap): under the hard 1080s wall the end-to-end K=5 prox+DC does
  only ~8 epochs x 200 samples because each training forward costs 5 full-view
  projection/back-projection PAIRS over 2,304 views -- the expensive physics. The
  233k-param ITNet champion got ~24x more gradient exposure via a cheaper
  per-step paradigm (k=1 + image-domain denoiser). iter-7's loss was STILL FALLING
  monotonically when the wall cut it -> genuinely UNDER-TRAINED, partly starvation.

iter-23 SIX-BOX (NUMBERS) -- ordered-subsets DC: cut the VIEW COUNT in training
------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = psnr 34.08 dB):
  - iter-7 (CHAMPION, BASE): FoE 1,921p, K=5, single-phase PARTIAL cosine 5e-3
    (budget-cut @~ep8) -> val hr 0.2515, ssim 0.9058, psnr 36.59, val_rmse 7.40e-4
    (TRAIN-CUT @ep8, loss still falling MONOTONICALLY => UNDER-TRAINED). test 0.1852.
  - iter-8..20: EVERY capacity/depth/stage/LR/kernel/half-bank knob mapped at the
    fixed model + SINGLE-PHASE trainer -> 0.2515 is the single-phase ceiling.
  - iter-21/22 (TWO-PHASE denoiser-pretrain): hr 0 BOTH (Phase-A froze at the
    identity floor with zero AND non-zero RBF init). idea A (PnP) is DEAD.
FAILURE MODE addressed (iter-23): TRAINING-STARVATION of the coupled FoE+DC
  unroll. The binding cost is the FULL 2,304-view projection/back-projection done
  K=5 times per training forward; iter-7 fit only ~8 epochs in-budget with the
  loss still falling. The reg never saw enough gradient steps.
CHANGE (iter-23, ONE knob -- the TRAINING DC view count; model byte-for-byte):
  add `train_view_subset` (= 4): during TRAINING the data-consistency term uses a
  RANDOM SUBSET of S = 2304 / train_view_subset = 576 projection views per step.
  Build a reduced-angle fan-beam projector R_S over S uniform views and use the
  matching strided sinogram g_S = g[:, off::r, :] (the S measured columns):
    DC_S(x) = R_Sᵀ(R_S x - g_S) / ‖R_SᵀR_S‖
  The view subset is RE-RANDOMIZED each step via a random angular offset
  off in {0..r-1} (ordered-subsets / SGD-over-views); the reduced projector for
  each (ps, off) is built once and CACHED with angle_start = 2π·off/2304 so its
  S views land EXACTLY on the strided columns g[:, off::r] -- geometrically exact,
  not an approximation. At r=4 the projection physics is ~4x cheaper/step ->
  ESTIMATED ~3-4x MORE epochs fit in the 1080s budget (iter-7's ~8 -> ~24-30) ->
  the FoE finally gets ITNet-like gradient exposure on the COUPLED objective.
  Each reduced projector gets its OWN power-iteration ‖R_SᵀR_S‖ (cached per
  (ps, r)) so the per-step alpha scale stays O(1) on the subsampled operator
  (RᵀR magnitude scales with the view count).
INFERENCE / VAL SCORING uses the FULL 2,304-view DC: the eval loop runs the
  model's normal forward with the FULL per-sample projector (_projs[vrk[i]]), the
  FULL val sinogram, and the build-time FULL dc_norm. The scored recon is
  full-view -- ONLY training subsamples; NO train/test recon mismatch in the
  scored output (verified by a runtime self-check that the eval projector has
  n_angles == 2304).
STABILITY (why ordered-subsets stays in the iter-7 basin):
  (1) PARAMS UNCHANGED (1,921). NO capacity/depth/stage added. The model IS iter-7
      byte-for-byte; only the TRAINING DC operator's angular sampling changes.
  (2) DC_S is a STRICT SUBSET of the full DC gradient -- the SAME RᵀR physics at
      fewer angular samples -- with dc_norm rescaled to ‖R_SᵀR_S‖ so the step
      magnitude matches the full operator. This is the standard ordered-subsets /
      stochastic-gradient view of the DC term; the recon DYNAMICS (K=5, single
      tied scalar alpha, plain prox step, per-step clamp) are byte-for-byte iter-7.
  (3) TRAINER is iter-7 EXACTLY: single-phase, plain Adam, peak lr 5e-3, PARTIAL
      cosine (cosine_t_max=16), grad_clip=1.0, bs=1, per-sample-ps. NOT iter-12's
      diverged 1e-2 peak, NOT iter-15's diverged constant LR, NOT the dead
      two-phase pretrain.
HYPOTHESIS: ~4x cheaper subsampled-view TRAINING buys ~3-4x more epochs in-budget
  -> the FoE reg reaches lower val-RMSE on the COUPLED objective -> val hr > 0.2515
  => TRAINING-BOUND confirmed (the starvation diagnosis is right). If it SATURATES
  ~0.25 (or below) the ceiling is CAPACITY not training (clean, decisive -- the
  FoE bank is saturated even with ITNet-like exposure). If the reduced-view DC
  mis-shapes the reg vs the full-view eval operator it may REGRESS (also clean: the
  subset gradient and the full operator disagree). PREDICTED val hr ~0.27-0.30.
  REPORT epochs_done vs iter-7's ~8 + n_train_views (576) vs eval 2304.
DEAD ENDS (do NOT reintroduce): projection-free / denoiser pretrain (frozen x2,
  hr 0); two-phase; capacity^; K>5; momentum/per-step-alpha; separate learned-init
  stage; peak LR>5e-3; constant LR; full anneal to 1e-5; bilateral-alone.

----------------------------------------------------------------------------
iter-22 (the prior BASE) header is preserved below for the campaign record.
----------------------------------------------------------------------------
Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-22).

iter-22 FIXES THE iter-21 TWO-PHASE BOOTSTRAP. iter-21 introduced the two-phase
trainer (Phase A = projection-free FoE denoiser pretrain; Phase B = end-to-end
K=5 DC finetune) to test the campaign's headline hypothesis: "does MORE EFFECTIVE
TRAINING beat iter-7's 0.2515?". But Phase A FROZE -- a ~452s no-op -- so the
hypothesis was never actually tested. iter-22 makes the pretrain LEARN and adds a
mandatory decrease guard. MODEL is still iter-7 byte-for-byte (FoE nf24/k7/nb31
1920p + 1 scalar alpha = 1,921 params, K=5 tied prox+DC, no momentum, no
learned-init); the ONLY change vs iter-21 is the Phase-A BOOTSTRAP.

THE iter-21 BUG (the freeze, diagnosed):
  Phase-A denoise_loss stuck at EXACTLY 1.9e-4 (=~100x iter-7's healthy end-to-end
  ~3e-6) across ALL epochs; alpha frozen at 0.0862. ROOT CAUSE: the model is built
  with foe_rbf_init_std=0.0 (iter-7's zero-init synthesis), so the FoE RBF mixture
  is ZERO => rho'(.)==0 => reg(x)==0 at init. The standalone Phase-A denoiser
  D(x)=clamp(x - alpha*reg(x), 0, clip_max) is therefore the IDENTITY (D(x)=x for
  the non-negative LD-FBP in range). The supervised loss MSE(D(LD-FBP), truth) =
  MSE(LD-FBP, truth) is a CONSTANT, and the gradient that should lift the RBF off
  zero self-vanishes at the zero mixture (unlike the end-to-end scheme, where the
  projection residual / DC term drives learning even at reg==0). So Phase A could
  not move and Phase B started from a DEAD reg.

iter-22 SIX-BOX (NUMBERS) -- make the denoiser pretrain ACTUALLY LEARN
---------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = psnr 34.08 dB):
  - iter-7 (CHAMPION, BASE): FoE 1,921p, K=5, single-phase PARTIAL cosine 5e-3
    (budget-cut @~ep8) -> val hr 0.2515, ssim 0.9058, psnr 36.59, val_rmse 7.40e-4
    (TRAIN-CUT @ep8, loss still falling MONOTONICALLY => UNDER-TRAINED). test 0.1852.
  - iter-8..20: EVERY capacity/depth/stage/LR/kernel/half-bank knob mapped at the
    fixed model + SINGLE-PHASE trainer -> 0.2515 is the single-phase ceiling.
  - iter-21 (TWO-PHASE, BROKEN): Phase-A denoise_loss FROZE at 1.9e-4, alpha
    frozen 0.0862, ~452s wasted no-op; the two-phase hypothesis was NOT tested.
FAILURE MODE addressed (iter-22): Phase-A bootstrap is DEAD at zero-RBF-init
(D(x)=identity => constant loss => self-vanishing gradient). The two-phase
training-efficiency hypothesis cannot be tested until the pretrain learns.
CHANGE (iter-22, ONE knob -- the Phase-A bootstrap; model+Phase-B byte-for-byte):
  add foe_rbf_init_std_pretrain (=0.05): a NON-ZERO std used to RE-SEED ONLY the
  FoE synthesis (RBF) weights at the START of Phase A, so reg(x)!=0 from step 0
  => D(x) actually denoises => gradients flow => the FoE LEARNS a real
  image-domain denoiser. The model is still BUILT with foe_rbf_init_std=0.0
  (iter-7 byte-for-byte); only the synthesis weights are re-seeded in-place at the
  Phase-A boundary. The analysis filters K and the single tied scalar alpha are
  UNTOUCHED -> params stay EXACTLY 1,921. Phase B then fine-tunes end-to-end from
  the TRAINED (non-degenerate) reg.
VERIFICATION (mandatory -- iter-21 lacked it): a Phase-A decrease guard captures
  epoch-1 vs the last Phase-A denoise_loss and ASSERTS a >=1% relative drop;
  a flat loss (the iter-21 1.9e-4 identity-floor freeze) FAILS LOUDLY before
  Phase B + the eval are wasted. result.json carries pretrain_first_loss,
  pretrain_last_loss, pretrain_rel_drop, foe_rbf_init_std_pretrain.
STABILITY (why a non-zero pretrain init stays safe): the zero-init-for-stability
  argument from the end-to-end-ONLY iters (iter-7..20) applies to the COLD
  end-to-end start. iter-22 does NOT cold-start end-to-end from a non-zero reg --
  Phase A first TRAINS the reg to a sensible non-degenerate denoiser, then Phase B
  fine-tunes end-to-end from THAT trained reg with iter-7's exact partial cosine
  (peak 5e-3, NOT iter-12's diverged 1e-2, NOT iter-15's diverged constant LR),
  grad_clip=1.0. So Phase-B stability rests on the TRAINED bank, not the raw init.
HYPOTHESIS: a WORKING denoiser pretrain (Phase-A loss drops WELL BELOW the 1.9e-4
  identity floor) gives the FoE reg ITNet-like cheap image-domain data exposure
  (many epochs over the full 579-slice pool) -> the warm end-to-end finetune
  reaches lower val-RMSE in-budget -> val hr > 0.2515. PREDICTED ~0.26-0.30. If
  Phase A learns but final hr stays ~0.25, training-efficiency was NOT the
  bottleneck (capacity is) -- a clean, important campaign result. If it REGRESSES,
  the image-domain denoiser optimum and the DC-unroll optimum disagree (the warm
  start mis-seeds Phase B) -- also clean.
DEAD ENDS (NOT reintroduced): capacity^, K>5, momentum/per-step-alpha, separate
  learned-init forward stage, peak LR>5e-3, constant LR, full anneal to 1e-5,
  bilateral-alone. EVERYTHING ELSE byte-for-byte iter-21/iter-7: K=5, single
  scalar alpha, plain Adam, peak lr 5e-3, PARTIAL cosine, grad_clip=1.0, bs=1,
  per-sample-ps, clip_max=0.05, dc_norm power-iter, pretrain_frac 0.40,
  pretrain over the full pool, HARD max_train_s=1080, val_n=214.

----------------------------------------------------------------------------
iter-21 (the prior BASE) header is preserved below for the campaign record.
----------------------------------------------------------------------------
Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-21).

iter-21 OPENS THE 20-ITER EXTENSION (iters 21-40). The goal returns to the
campaign's headline question: "does MORE EFFECTIVE TRAINING beat iter-7's
0.2515?". The MODEL is iter-7 byte-for-byte (FoE nf24/k7/nb31 1920p, K=5 tied
prox+DC, ONE scalar alpha, no momentum, no learned-init -> 1,921 params); the
ONLY change is a TWO-PHASE TRAINER that fixes the TRAINING-STARVATION the prior
20 iters never addressed.

THE BINDING-CONSTRAINT DIAGNOSIS (what iter-8..20 all missed):
  under the hard 1080s wall the fully-unrolled K=5 prox+DC is TRAINING-STARVED.
  Each forward does ~5 projection/backprojection PAIRS per sample (the DC term),
  so iter-7 completed only ~8 epochs x 200 samples (~1.6k sample-passes) before
  the wall budget-cut it. The 233k-param ITNet champion (hr 0.4398) ran the SAME
  ~20-min wall but at k=1 + an IMAGE-DOMAIN denoiser pretrain -> ~66 epochs x 579
  samples (~38k sample-passes, ~24x more gradient exposure). So iter-7's 0.2515
  ceiling is PARTLY training-starvation, NOT pure capacity. The LR regime
  (iter-7/13/14/15) and the reg geometry/capacity (iter-8/9/16/18/19/20) are both
  fully mapped and EXHAUSTED at fixed model + single-phase training; the ONE
  untried axis is HOW MUCH effective gradient exposure the FoE reg gets in-budget.

iter-21 SIX-BOX (NUMBERS) — fix the STARVATION without touching the architecture
-------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = psnr 34.08 dB):
  - iter-7 (CHAMPION, BASE): FoE nf24/k7/nb31 = 1,920 reg + 1 alpha = 1,921p,
    K=5 tied prox+DC, single-phase 16-ep PARTIAL cosine 5e-3 (budget-cut @~ep8,
    final LR ~2.5e-3, plain Adam) -> val hr 0.2515, ssim 0.9058, psnr 36.59,
    val_rmse 7.40e-4 (1331s wall; TRAIN-CUT @ep8 with loss STILL falling
    MONOTONICALLY 3.4e-6->7.1e-7, train MSE ~= val MSE => NOT overfit, genuinely
    UNDER-TRAINED). test hr 0.1852. THE STARVATION SIGNATURE.
  - iter-8 (FAIL): SCALE FoE nf24->40 (2,881p) -> hr 0, psnr 1.87 (BLEW UP). capacity exhausted.
  - iter-9 (FAIL): K 5->7 -> hr 0, psnr 11.38 (DIVERGED). depth exhausted.
  - iter-10/11 (FROZE): learned-init refiner -> any added trainable stage collapses training.
  - iter-12 (DIVERGED): peak LR 5e-3->1e-2 + warmup -> 5e-3 is the stability EDGE.
  - iter-13 (0.2273): FULL anneal -> 1e-5 OVERFITS the low-LR tail.
  - iter-14 (0.2281): AdamW wd 3e-4 does NOT recover the anneal-tail overfit.
  - iter-15 (FAIL): constant 5e-3, NO anneal -> hr 0, psnr 11.38 (the KNEE; partial anneal load-bearing).
  - iter-16 (0.2404): FoE k7->k9/nf24->17 iso-param REGRESSED -> nf24/k7 is the FoE geometry optimum.
  - iter-18 (0.2334 @961p), iter-19 (0.1690 @481p): HALVING the bank walks the
    param/headroom frontier DOWN monotonically -> capacity helps, but the
    SINGLE-PHASE trainer is the bottleneck at the top.
  CLEAN VERDICT after 20 iters: at iter-7's fixed model + SINGLE-PHASE training,
  EVERY capacity/depth/stage/LR knob is mapped and 0.2515 is the ceiling. The
  reg is UNDER-TRAINED (iter-7's own loss-still-falling signature). The ONLY
  untried axis is the TRAINING PROTOCOL: give the FoE reg ITNet-like gradient
  exposure WITHOUT the expensive DC unroll on every step.

FAILURE MODE addressed (iter-21): TRAINING-STARVATION of the FoE reg. iter-7's
loss was still falling monotonically when the 1080s wall cut it at ~ep8 (~1.6k
sample-passes) -- the reg never saw enough gradient steps. The DC term (1 fwd +
1 back-project x K=5 per forward) is what makes each unrolled step ~5x more
expensive than a pure image-domain pass, throttling the epoch count.

CHANGE (iter-21, TRAINER ONLY -- model byte-for-byte iter-7, 1,921 params):
  add a config-gated TWO-PHASE trainer (`two_phase`):
  (A) PROJECTION-FREE DENOISER PRETRAIN (cheap, many epochs):
      train ONE prox step WITH THE DC TERM ZEROED as an image denoiser:
        D(x) = clamp(x - alpha * reg(x), 0, clip_max)
      supervised D(LD-FBP) -> truth, pure image-domain MSE, NO forward/back-
      project in the loop. SAME reg + SAME alpha that Phase B fine-tunes, so the
      pretrained bank drops in WARM. ~5x cheaper/step than the K=5 DC unroll ->
      MANY epochs over the FULL train pool (pretrain_train_n up to 579) in a
      small time slice. zero-init RBF (iter-7) => D(x)=x at init (clean seed).
      cosine LR peak 5e-3 (the proven-stable peak; NEVER exceeded).
  (B) END-TO-END DC FINE-TUNE (the iter-7 objective):
      then fine-tune the FULL K=5 tied prox+DC unroll end-to-end (iter-7's exact
      supervised MSE) starting from the pretrained reg+alpha, for the REMAINING
      budget. cosine LR peak 5e-3 (do NOT exceed -- iter-12's 1e-2 DIVERGED).
  BUDGET SPLIT of the 1080s wall: pretrain_frac=0.40 (~432s Phase A, ~648s
  Phase B). Phase A is ~5x cheaper/step so ~432s buys MANY image-domain epochs
  over 579 samples; Phase B then spends the bulk on the expensive DC unroll from
  a warm start. Each phase wall-caps itself; total train <= 1080s. val_n=214.
EVERYTHING ELSE from iter-7 byte-for-byte: reg_type="foe" nf24/k7/nb31 (1,920p),
n_iter=5, single tied scalar alpha (1p, total 1,921), zero-init rho', NO
momentum, NO per-step alpha, NO learned-init, plain prox step, plain Adam (NO
weight-decay), peak lr=5e-3, PARTIAL cosine (cosine_lr=True, cosine_t_max=16),
grad_clip=1.0, batch_size=1, per-sample-ps, clip_max=0.05, dc_norm power-iter.
STABILITY (why two-phase stays in the iter-7 basin):
  (1) PARAMS UNCHANGED (1,921). NO capacity/depth/stage added (dodges iter-8/9/
      10/11). The model IS iter-7; only the optimisation trajectory changes.
  (2) Phase A is the iter-7 reg's OWN prox step with DC=0 -- a STRICT SUBSET of
      the iter-7 forward (DC term dropped), pure image-domain MSE -> cannot blow
      up the recon (no projection feedback loop). zero-init rho' => D(x)=x at
      init, so Phase A starts as the identity and learns the denoiser.
  (3) Phase B is iter-7 EXACTLY, just from a warm reg -- same K=5, same scalar
      alpha, same partial cosine from peak 5e-3 (NOT iter-12's diverged 1e-2,
      NOT iter-15's diverged constant LR), grad_clip=1.0 caps spikes.
HYPOTHESIS: the cheap projection-free denoiser pretrain gives the FoE reg
ITNet-like gradient exposure (many image-domain epochs over the full pool), so
the expensive end-to-end DC finetune starts WARM -> reaches lower val-RMSE
in-budget than iter-7's cold under-trained run -> val hr ABOVE 0.2515. If it
MATCHES 0.2515, the ceiling is capacity not training (the FoE bank is saturated
even when well-trained). If it REGRESSES, the image-domain denoiser optimum and
the DC-unroll optimum disagree and the warm start mis-seeds Phase B (a clean
negative -- the two objectives are not aligned). PREDICTED val hr ~0.26-0.30
(beat 0.2515 by closing part of the ITNet-vs-FoE training-exposure gap).

----------------------------------------------------------------------------
iter-17 (a prior BASE) header is preserved below for the campaign record.
----------------------------------------------------------------------------
Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-17).

iter-17 PROBES THE EXTREME PARAM-EFFICIENCY FRONTIER (a DIFFERENT goal than
"beat iter-7's 0.2515"). The FoE FAMILY is SETTLED: iter-7 (FoE nf24/k7/nb31,
1921p) is a razor-sharp optimum at hr 0.2515 / test 0.1852 (best hr/param on
the board), and 9/9 perturbations + iter-16's iso-param kernel widening (k9/nf17
-> hr 0.2404, a regression) all confirmed nf24/k7 is the FoE geometry optimum.
iter-17 instead answers the user's explicit interest in "methods that work well
with 0 or only 12 trainable parameters": it SWAPS the 1920-param FoE reg for a
trainable EDGE-PRESERVING BILATERAL reg (~16 reg params + 1 alpha = ~17 total)
INSIDE iter-7's EXACT stable unrolled prox+DC. This will VERY LIKELY score BELOW
FoE's 0.2515 — that is FINE and EXPECTED. The value is a clean hr-at-~17-params
datapoint on the param/headroom Pareto frontier: does an edge-preserving
bilateral prior in a DC unroll beat classical TV's 0.2085 at ~tens of params?

THE BILATERAL reg (iter-17, the swapped FAMILY):
  reg(x) = sum_i  gain_i * (x - BF_i(x))
a SUM of n_bf=4 TrainableBilateralFilter2d (Wagner et al. Med. Phys. 2022,
ddssl_ldct/models.py:92) denoise-then-subtract RESIDUALS, each scaled by its own
learnable scalar gain_i. Each BF has 3 learnable params (log σx, log σy, log σr
— spatial-x, spatial-y, range bandwidths); the per-filter gain adds 1 => 4
trainable params PER FILTER. EXACT PARAM COUNT at n_bf=4: 4*(3+1) = 16 reg
params + 1 scalar alpha = 17 TOTAL (0.000017 M) — two orders below iter-7's
1,921p FoE.
ZERO-INIT GAIN (the iter-17 stability fix, mirroring iter-7's zero-init ρ'):
gain_init=0.0 => every gain starts at 0 => reg(x) ≡ 0 at init REGARDLESS of the
σ values, so the seed is the EXACT clean GD+DC scheme of iter-7 and training
LEARNS the bilateral correction by lifting the gains off zero. A bilateral
filter is a single EDGE-AWARE smoothing — NO pooling (dodges iter-2/5), NO extra
unroll stage (dodges iter-10/11), NO added depth (dodges iter-9). The recon
DYNAMICS (K=5, single tied scalar alpha, plain prox step) and the TRAINER (plain
Adam NO weight-decay, peak lr 5e-3, cosine_t_max=16 PARTIAL anneal, epochs=16,
max_train_s=1080, train_n=200, val_n=214, grad_clip=1.0, bs=1, per-sample-ps)
are ALL byte-for-byte iter-7. The ONLY change is the reg FAMILY (FoE -> bilateral)
and the matching tiny param budget.
PREDICTED hr ~0.10-0.21: very likely BELOW FoE's 0.2515; a competitive read is
landing near or above classical TV's 0.2085 at ~17 params (a strong extreme-
param-efficiency frontier point); the LD-FBP floor is psnr 34.08 dB. A regression
to hr 0 would mean the bilateral prior is too weak to beat the FBP RMSE floor in
this DC unroll at this scale — still a clean negative datapoint.

----------------------------------------------------------------------------
iter-16 (the prior BASE) header is preserved below for the campaign record.
----------------------------------------------------------------------------
Reference: Parameter-Efficient unrolled learned-proximal gradient (iter-16).

A weight-TIED unrolled proximal-gradient reconstruction with explicit
data-consistency against the measured sinogram. The SAME regulariser
module and the SAME step-size scalar `alpha` are reused at every unrolled
step, so the trainable parameter budget is set by ONE small regulariser
(hundreds-to-low-thousands of params) regardless of `n_iter` — in sharp
contrast to the 233k-param ITNet champion whose denoiser is a full SmallUNet.

Architecture (iter-16: the iter-7 CHAMPION FoE unroll + iter-7's EXACT stable
partial-anneal training, with ONLY the FoE kernel geometry WIDENED at iso-param
— no learned-init, no added depth/capacity/stages, no new params):

    x_0 = LD-FBP(sino)                              # raw FBP init (iter-7)
    for k in range(K):
        dc = R^T( R x  -  sino ) / dc_norm          # data-consistency grad
        x  = clamp( x - alpha * ( dc + reg(x) ),     # proximal-gradient step
                    0.0, clip_max )

iter-16 KEEPS the unroll DYNAMICS (K=5, single tied scalar alpha, zero-init
rho', plain prox step, NO momentum/per-step-alpha/learned-init) and the TRAINER
(plain Adam NO weight-decay, peak lr 5e-3, cosine_t_max=16 PARTIAL anneal,
epochs=16, max_train_s=1080, train_n=200, val_n=214, grad_clip=1.0, bs=1) ALL
byte-for-byte iter-7. The ONLY change is the reg RECEPTIVE FIELD at ISO-PARAM:
foe_kernel 7 -> 9 with foe_n_filters 24 -> 17 to hold params ~1900.
  iter-7  bank: 24*7*7 + 24*31 = 1176 + 744 = 1,920p (+1 alpha = 1,921 total)
  iter-16 bank: 17*9*9 + 17*31 = 1377 + 527 = 1,904p (+1 alpha = 1,905 total)
=> -16 reg params (-0.83%), essentially iso-param. eff RF widens 7px/step
(~19px over K=5) -> 9px/step (~25px over K=5).

CRITICAL — iter-16 REVERTS iter-15's trainer: iter-15 tested the constant-LR
trend endpoint (cosine_lr False, flat 5e-3 the whole run) and it DIVERGED
(hr 0, psnr 11.38, val_rmse 0.0135 — same signature as iter-9's K=7). The
trend-endpoint hypothesis is FALSIFIED: the partial-anneal dip toward 2.5e-3
is LOAD-BEARING for stability, NOT an overfitting tail. So iter-16 restores
iter-7's partial cosine (cosine_lr=True, cosine_t_max=16) byte-for-byte and the
ONLY scientific lever this iter is the FoE kernel widening.

(The learned-init refiner `g` / LearnedInit module from iter-10/11 is KEPT in
the file but DISABLED by default — learned_init=False — so the model is iter-7
exactly. Both iter-10/11 learned-init attempts FROZE training; iter-14 does not
touch capacity at all. iter-12's 2x peak LR + warmup are NOT revived — they
DIVERGED; iter-14 keeps iter-13's COMPLETED 8-ep cosine at the iter-7 peak
5e-3, and the ONLY new ingredient is the decoupled weight_decay 3e-4 that
turns the optimiser into AdamW.)

`dc_norm` is a power-iteration estimate of ‖R^T R‖ so `alpha` lives in O(1)
regardless of geometry (mirrors solver_hammernik_vn.py). `alpha` is a single
learnable softplus SCALAR (init from `alpha_init`) when `learnable_alpha` —
shared across all K steps. NO per-step alpha, NO momentum (both shown to
destabilise the recon in the 20-min budget; see iter-3 below).

iter-16 SIX-BOX (the ONE UNTRIED axis inside the proven-stable FoE family: widen the reg RECEPTIVE FIELD at ISO-PARAM — kernel 7->9, nf 24->17; iter-7 training restored)
------------------------------------------------------------------------------------------------------------------
PRIOR RESULTS on Mayo-LDCT (search-20260624-01, LD-FBP floor = 34.08 dB):
  - iter-1: cnn reg (2,798 params), K=5 tied prox+DC, learnable SCALAR
    alpha, 8 epochs, ~12 min wall -> hr 0.0871, ssim 0.846, psnr 34.87.
  - iter-2 (FAIL @ 8 ep): micro-UNet reg (25,890 params), K=5, 8 ep ->
    hr 0 (psnr 32.41). a 9x-wider POOLED reg below floor.
  - iter-3 (FAIL worse): tiny cnn + K=8 + per-step alpha + Nesterov
    MOMENTUM (2,806 params), 6 ep -> hr 0 (psnr 28.08, UNSTABLE).
    LESSON: the iter-3 instability was the MOMENTUM + per-step-alpha COMBO
    (a non-monotone accelerated scheme). AVOID both.
  - iter-4: single-scale tiny cnn (2,798p, 3-layer 12ch, K=5, scalar alpha,
    NO momentum) + cosine LR + 16 epochs -> hr 0.2378, ssim 0.886,
    psnr 36.44 (~20 min wall). BREAKTHROUGH: training-limited, not
    capacity-limited -- 16-ep cosine TRIPLED hr at the SAME 2,798 params.
  - iter-5 (FAIL): micro-UNet reg (25,890p) under the iter-4 trainer ->
    hr 0 (psnr 32.41). POOLING in the reg is ARCHITECTURALLY BAD. Do NOT pool.
  - iter-6 (FAIL): cnn reg GROWN flat+dilated (37,601p) -> underperformed.
    SCALING the CNN reg fails BOTH ways; 2,798p is the CNN sweet spot.
  - iter-7 (CHAMPION, BASE): reg FAMILY cnn -> "foe" (TIED Fields-of-Experts
    / VN learned filter bank, reg(x)=K^T rho'(Kx)). nf=24 / k=7 / nb=31 ->
    1,920 reg params, total 1,921 (1 scalar alpha), zero-init rho'-weights,
    K=5 tied prox+DC, 16-ep cosine LR 5e-3->1e-5 (Adam) -> hr 0.2515, ssim
    0.906, psnr 36.59, val_rmse 7.40e-4 (1331 s, wall-bounded; TRAIN-CUT at
    epoch 8, loss STILL falling MONOTONICALLY 3.4e-6 ep1 -> 7.1e-7 ep8 with
    NO oscillation). train MSE ~= val MSE => NOT overfit, genuinely
    UNDER-TRAINED. The FoE BEAT the tiny CNN (0.2515 vs 0.2378) at FEWER
    params (1,921 vs 2,798).
  - iter-8 (FAIL, TOTAL DIVERGENCE): SCALED the FoE bank nf 24->40 (2,881p)
    -> hr 0, psnr 1.87 (recon BLEW UP). CAPACITY-SCALING IS FULLY EXHAUSTED:
    EVERY reg-capacity increase (iter-2/5/6/8) regresses or diverges in the
    20-min budget. ~1.9-2k params (iter-7 FoE) is the ARCHITECTURE SWEET SPOT.
  - iter-9 (FAIL, DIVERGENCE): DEEPENED the tied unroll K 5->7 at the SAME
    1,921 params -> hr 0, psnr 11.38, RMSE 0.0135 (the recon DIVERGED).
    VERDICT: ADDING UNROLL DEPTH destabilises in-budget too. K=5 EXACTLY.
  - iter-10 (FAIL, FROZE): learned-init refiner g with a BUGGY init clamp
    (clamp(LD-FBP+g, 0, clip_max)) truncated bright LD-FBP pixels -> ep-1 loss
    100x iter-7's, FROZE at the degenerate loss 1.9e-4 for 8 epochs.
  - iter-11 (FAIL, FROZE AGAIN — the science result): re-wired the learned-init
    CORRECTLY (x_0 = LD-FBP + g(LD-FBP), NO upper clamp, GELU+zero-init head,
    +177 params). The runtime self-check VERIFIED true zero-init (rel_g≈0,
    x_0 == LD-FBP byte-for-byte). YET training STILL FROZE at the identical
    degenerate loss ~1.9e-4: the extra trainable stage collapses training to a
    bad attractor REGARDLESS of init correctness. VERDICT: ADDING ANY trainable
    stage (capacity/depth/refiner) destabilises in the 20-min budget. The ONLY
    proven-stable thing is the iter-7 FoE unroll ITSELF, and it is UNDER-TRAINED.

  - iter-12 (FAIL, DIVERGED): TRAINER rewrite — peak LR 5e-3 -> 1e-2 (2x) +
    0.5-ep linear WARMUP + Adam -> AdamW(wd 1e-4), all else iter-7. epoch-1
    fine (loss 3.2e-6) but epoch-2 JUMPED to 7e-5, epoch-3 hit the degenerate
    attractor 1.9e-4 -> DIVERGED. VERDICT: 5e-3 is at the STABILITY EDGE; a 2x
    peak overshoots even WITH warmup. Do NOT raise the peak LR. (The divergence
    came from the 2x PEAK LR, NOT from AdamW — iter-14 reuses AdamW at the SAME
    5e-3 peak and stays stable.)

  - iter-13 (STABLE, BASE): TRAINER SCHEDULE PERIOD fix at the iter-7 model
    byte-for-byte — epochs 16 -> 8 + a decoupled cosine_t_max=8 so the cosine
    FULLY anneals to eta_min=1e-5 within the 1080s wall (iter-7's T_max=16
    cosine was cut at ~ep8 with LR still ≈2.5e-3, the anneal SKIPPED). Plain
    Adam, peak lr 5e-3 (UNCHANGED), warmup 0, wd 0. RESULT: hr 0.2273, ssim
    0.898, psnr 36.32, val_rmse 7.64e-4, train loss 5.78e-7 (< iter-7's 7.1e-7),
    1240 s. SCIENCE: completing the anneal LOWERED TRAIN LOSS (5.78e-7 vs 7.1e-7)
    yet DROPPED val hr (0.2273 < iter-7's 0.2515) and RAISED val_rmse
    (7.64e-4 > 7.40e-4). So the full low-LR fine-tune phase OVERFIT the 200-slice
    train set: train MSE fell but val MSE rose. VERDICT: the run is NO LONGER
    under-trained or unstable — the binding constraint is now VAL GENERALISATION
    (a train/val gap). The lever is REGULARISATION of the completed anneal, NOT
    more/fewer epochs and NOT a different LR.

  - iter-14 (STABLE, the TREND-COMPLETING CONTROL): iter-13's COMPLETED-anneal
    schedule byte-for-byte (epochs=8, cosine_t_max=8, peak lr 5e-3, anneal ->
    eta_min 1e-5) + the ONE change plain Adam -> AdamW decoupled weight_decay
    3e-4. RESULT: hr 0.2281, ~= iter-13's 0.2273. VERDICT: decoupled L2 did NOT
    recover the anneal-tail overfit — wd is the WRONG lever for it. The overfit
    is NOT a weight-NORM problem that L2 shrinkage fixes; it is the low-LR tail
    itself fitting the 200-slice train noise realisation. So the only thing that
    helped iter-7 was that its 16-ep cosine got BUDGET-CUT at ~ep8 (LR still
    ≈2.5e-3): the SKIPPED low-LR tail acted as an EARLY-STOP regulariser.

  - iter-15 (FAIL, DIVERGED — the science result that REFUTES the trend endpoint):
    the unroll iter-7 byte-for-byte, peak lr 5e-3 UNCHANGED, plain Adam, but
    cosine_lr True -> False (CONSTANT 5e-3 held the WHOLE run, NO anneal tail).
    RESULT: hr 0, ssim 0.292, psnr 11.38, val_rmse 0.0135 — the recon DIVERGED
    (same signature as iter-9's K=7). VERDICT: the iter-7/13/14 "less anneal =
    better val" trend is NOT a monotone line with a flat-LR endpoint — it has a
    KNEE. The partial-anneal dip toward 2.5e-3 is LOAD-BEARING for STABILITY (it
    pulls the late-training step below the divergence threshold), NOT an
    overfitting tail to be removed. A constant 5e-3 held to the end is the
    STABILITY EDGE without the late dip and the recon blows up. iter-7's PARTIAL
    cosine (decaying toward 2.5e-3 by ep8) is the TRUE optimum; the LR REGIME is
    now fully mapped — partial anneal from 5e-3 is the ONLY stable+optimal trainer.

CLEAN TREND, now with the iter-15 KNEE (the LR regime is fully mapped):
  iter-7  PARTIAL anneal (cosine T_max=16 budget-cut @ep8, final LR ~2.5e-3) -> hr 0.2515  (CHAMPION)
  iter-13 FULL    anneal (epochs=8, cosine_t_max=8, LR -> 1e-5)              -> hr 0.2273  (low-LR tail OVERFITS)
  iter-14 FULL    anneal + AdamW wd 3e-4                                     -> hr 0.2281  (wd does NOT recover it)
  iter-15 NO      anneal (constant 5e-3, the trend ENDPOINT)                 -> hr 0       (DIVERGED — the knee)
  => the trend has a KNEE at iter-7's partial anneal: too much low-LR tail OVERFITS
     (iter-13/14), NO tail at all DIVERGES (iter-15). iter-7's partial cosine is the
     optimum. The LR regime is EXHAUSTED; the remaining untried axis is the reg GEOMETRY.

FAILURE MODE addressed (iter-16): the LR-schedule axis is now fully mapped (iter-15's
divergence closed it — partial anneal from 5e-3 is the only stable+optimal trainer).
Inside the proven-stable FoE family + iter-7's exact training, EVERY capacity/depth/
stage/LR change has been mapped and fails. The ONE untried axis is the reg RECEPTIVE
FIELD (kernel size) at ISO-PARAM: does a wider 9x9 analysis filter capture longer-range
LDCT noise correlations the 7x7 bank misses, lowering val-RMSE at the SAME param budget?

CHANGE (iter-16, FoE GEOMETRY at ISO-PARAM — ZERO net new params, iter-7 training
restored byte-for-byte):
  (1) foe_kernel 7 -> 9 (THE LEVER): widen the analysis kernel to eff RF 9px/step
      (~25px over K=5 vs ~19px at k7). A larger reg receptive field may model
      longer-range low-dose noise correlations the 7x7 bank cannot reach.
  (2) foe_n_filters 24 -> 17 (ISO-PARAM HOLD): reduce the filter COUNT so the wider
      kernel does not GROW the bank. nf17/k9/nb31 = 17*81 + 17*31 = 1,904p (+1 alpha
      = 1,905), iso-param to iter-7's 1,920p/1,921. NB filter COUNT goes DOWN (away
      from iter-8's divergent nf=40), only the receptive FIELD widens.
  (3) cosine_lr False -> True, cosine_t_max 8 -> 16 (REVERT iter-15's diverged
      trainer to iter-7's partial-anneal byte-for-byte). weight_decay stays 0.0.
EVERYTHING ELSE from iter-7 byte-for-byte: reg_type="foe", foe_n_bumps=31, n_iter=5,
single tied scalar alpha, zero-init rho', NO momentum, NO per-step alpha, NO
learned-init (learned_init=False), plain prox step, plain Adam (NO weight-decay),
epochs=16, peak lr=5e-3, cosine_lr_min=1e-5, warmup_frac=0.0, max_train_s=1080,
train_n=200, grad_clip=1.0, batch_size=1, val_n=214.
STABILITY (why a 9x9 iso-param FoE stays in the iter-7 basin):
  (1) SAME param count (~1,900) — no capacity increase. iter-8 diverged by GROWING
      the bank (nf 24->40 = 2,881p); iter-16 holds ~1,900 and even DROPS the filter
      count (24->17). A wider kernel at FEWER filters is not a capacity scale-up.
  (2) SAME FoE family, SAME zero-init rho' => reg(x) ≡ 0 at init, so the seed is the
      EXACT clean GD+DC scheme of iter-7. A wider kernel is still a single-scale
      LINEAR analysis filter — NO pooling (dodges iter-2/5), NO extra stage (dodges
      iter-10/11), NO added depth (dodges iter-9). The recon DYNAMICS (K=5, single
      scalar alpha, plain prox step) are byte-for-byte iter-7.
  (3) SAME training regime restored: iter-7's stable PARTIAL cosine from peak 5e-3
      (NOT iter-15's divergent constant LR, NOT iter-12's divergent 1e-2 peak),
      grad_clip=1.0 caps spikes. The trainer is the proven-stable iter-7 base.
HYPOTHESIS: a larger 9x9 reg receptive field captures longer-range LDCT noise
correlations the 7x7 bank misses -> val-RMSE drops BELOW iter-7's 7.40e-4 at the
SAME ~1,900 params, SAME stable partial-anneal training -> hr ABOVE 0.2515, a new
FoE-geometry frontier point. If it MATCHES iter-7 (~0.251), 7px is already the
sufficient receptive field and the extra reach buys nothing. If it REGRESSES, the
nf24/k7 geometry is confirmed optimal — trading filter count (24->17) for kernel
width (7->9) hurts more than the reach helps, cleanly settling the FoE geometry.
PREDICTED hr ~0.24-0.27 (recover iter-7's level; modest up/down around it).

The learned regulariser `reg(x)` is selected by `reg_type`:
  - "foe"       (DEFAULT — iter-7/9/10/11: TIED Fields-of-Experts / VN filter
                bank, a DIFFERENT, single-scale, param-EFFICIENT reg family that
                BEAT the CNN at fewer params in iter-7 — hr 0.2515 vs 0.2378):
                analysis conv2d (`foe_n_filters` filters, `foe_kernel`x
                `foe_kernel`) -> per-filter RBF activation (`foe_n_bumps`
                bumps) -> TIED conv_transpose2d synthesis. reg(x)=K^T ρ'(Kx),
                one VNStep's reg-gradient reused at every unrolled step.
                ZERO-INIT ρ'-weights => reg ≈ 0 at init (stability). iter-7
                CHAMPION nf=24/k=7/nb=31 = 1,920p (total 1,921). iter-8 SCALED
                the bank nf=24->40 = 2,881p and DIVERGED (psnr 1.87) — capacity
                exhausted. iter-16 WIDENS the kernel at ISO-PARAM: nf=17/k=9/nb=31
                = 1,904p (total 1,905, -16p vs iter-7), eff RF 9px/step (~25px over
                K=5 vs ~19px at k7) — filter COUNT down, receptive FIELD up.
  - "cnn"       (iter-1/iter-4 BEST, iter-6 FLAT-DILATED — kept selectable):
                `cnn_layers` 3x3 convs at `cnn_channels` channels with a
                per-layer `cnn_dilations` ladder (reflect-padded), GroupNorm+
                ReLU between, zero-init 1x1 head so reg ≈ 0 at init.
                Single-scale, NO pooling. iter-4 BEST: c=12/3-layers/dil=1
                (2,797p, eff RF ~7px, hr 0.2378). iter-6 GROWN: c=32/5-layers/
                dil[1,2,4,2,1] (37,601p) FAILED to beat iter-4 -> 2,798p is
                the CNN-family sweet spot.
  - "microunet" (iter-2/iter-5, REGRESSED — kept selectable): a 2-level
                (one-downsample) micro-UNet denoiser, ~25.9k params at c=16.
                ARCHITECTURALLY BAD here: pooling caps psnr at ~32.4 (hr 0)
                at BOTH 8 ep (iter-2) and 16 ep (iter-5). Do NOT use.
  - "bilateral" a cascade of `n_bf` TrainableBilateralFilter2d (3 params each).

Trained end-to-end supervised against the HD truth image
(`supervised_recon_loss`, Adam + cosine LR). The DC term + a modest learnable
`alpha` keep the recon data-consistent so it beats the LD-FBP RMSE floor (the
headroom gate is RMSE-vs-LD-FBP, not SSIM).

Citation context: this is the parameter-tied limit of the unrolled
proximal-gradient family (Hammernik 2018 MRM variational network; Adler &
Öktem 2018 learned primal-dual). See literature/ for the lineage.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: not always auto-loaded by `import torch`

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.metrics import (psnr, ssim, evaluate_calibrated,
                                make_4panel_comparison, supervised_recon_loss,
                                negativity_penalty, clip_and_step)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # ---- architecture (iter-17: iter-7 CHAMPION unroll + EXTREME-param-efficiency BILATERAL reg ~17p — NO learned-init, NO added capacity/depth) ----
    "reg_type":        "steerable_foe",  # iter-24: ROTATION-EQUIVARIANT FoE (analysis kernels synthesized from a SHARED FIXED steerable basis; ISO-PARAM 1,921 to iter-7). "steerable_foe" (iter-24) | "foe" (iter-7 CHAMPION 1921p, hr 0.2515) | "bilateral" (iter-17, ~17p, hr 0 below floor) | "cnn" (iter-4 / iter-6) | "microunet" (iter-2/iter-5, REGRESSED)
    "n_iter":          5,          # iter-21: KEEP K=5 (iter-7 CHAMPION, the SHARP stable basin; iter-9's K=7 DIVERGED). iter-21 changes ONLY the TRAINER; the unroll DYNAMICS are byte-for-byte iter-7.
    # ---- learned-init refiner `g` (iter-10/11 BOTH FROZE training -> DISABLED; kept selectable for the post-mortem) ----
    "learned_init":      False,    # iter-16: OFF (iter-10 AND iter-11 BOTH froze training even with verified true zero-init -> any added trainable stage collapses training in-budget). With this False the model is iter-7 byte-for-byte (raw LD-FBP init). True => the iter-11 learned-init build (kept selectable).
    "init_channels":     16,       # (unused at learned_init=False) hidden width of g; 16 => 177 params at init_layers=1.
    "init_layers":       1,        # (unused at learned_init=False) number of 3x3 hidden convs in g.
    "init_clamp":        False,    # (unused at learned_init=False) iter-11 FIX kept: do NOT upper-clamp the init (the iter-10 BUG).
    "learnable_alpha": True,       # ONE tied alpha = softplus(param); init from alpha_init
    "per_step_alpha":  False,      # iter-4: REVERT iter-3's per-step alpha (it destabilised) -> single shared scalar (iter-1)
    "momentum":        False,      # iter-4: REVERT iter-3's Nesterov momentum (it destabilised) -> plain prox step (iter-1)
    "beta_init":       0.5,        # unused when momentum=False (kept for backward-compat selectability)
    "alpha_init":      0.1,        # step size (O(1) thanks to dc_norm scaling)
    "clip_max":        0.05,       # per-step clamp upper bound (= display_max μ)
    "dc_norm":         True,       # divide R^T(R x - g) by power-iter ‖R^T R‖
    "checkpoint":      True,       # gradient-checkpoint each unrolled step
    # ---- "microunet" regulariser (iter-2/iter-5, REGRESSED — pooling caps psnr 32.4, do NOT use; kept selectable) ----
    "mu_channels":     16,    # base width; 2-level micro-UNet = 25,889 reg params at c=16 (exact iter-2/iter-5 arch)
    # ---- "cnn" regulariser (iter-4 BEST = c12/3-layers/dil1 = 2,797p; iter-6 GROWN = c32/5-layers/dilladder = 37,601p, FAILED) ----
    "cnn_channels":    12,    # iter-7: REVERT to iter-4 BEST (12) — the CNN-family sweet spot (iter-6's 32 failed). 2,797 reg params at 3 layers.
    "cnn_layers":      3,     # iter-7: REVERT to iter-4 BEST (3) — kept selectable, not used at reg_type=foe.
    "cnn_dilations":   None,  # iter-7: REVERT to iter-4 BEST (None = all dilation 1, eff RF ~7px). iter-6's [1,2,4,2,1] failed.
    # ---- "foe" regulariser (iter-16: WIDEN the reg receptive field at ISO-PARAM — nf24/k7 -> nf17/k9, holding params ~1900) ----
    "foe_n_filters":   24,    # iter-21: 24 (RESTORE iter-7 CHAMPION). nf24/k7/nb31 = 24*49 + 24*31 = 1176 + 744 = 1,920 reg params (+1 alpha = 1,921 total). The two-phase trainer fine-tunes THIS bank byte-for-byte; the frontier-shrinking iter-18/19/20 (nf 12/6/3) confirmed capacity helps at the top, so iter-21 returns to the full champion bank and attacks the TRAINING bottleneck instead.
    "foe_kernel":      7,     # iter-21: 7 (iter-7 CHAMPION byte-for-byte; the FoE GEOMETRY is SETTLED at nf24/k7 — iter-16's k9/nf17 REGRESSED to hr 0.2404). eff RF 7px/step (~19px over K=5 tied steps).
    "foe_n_bumps":     31,    # iter-16: 31 (UNCHANGED — iter-7 champion RBF activation resolution; only the kernel geometry/filter count change, the RBF non-linearity is byte-for-byte iter-7)
    "foe_x_range":     1.0,
    "foe_filter_init_std": 0.05,
    "foe_rbf_init_std":    0.0,   # iter-7: ZERO-INIT synthesis -> ρ'≡0 => reg(x)≡0 at init (stability; mirrors CNNReg zero-init head). This is the END-TO-END (Phase B / single-phase) init — UNCHANGED so the model is iter-7 byte-for-byte at build time. iter-22 re-seeds ONLY the pretrain (see foe_rbf_init_std_pretrain).
    "foe_rbf_init_std_pretrain": 0.05,  # iter-22 FIX: NON-ZERO RBF std used to RE-SEED the FoE synthesis weights at the START of Phase A ONLY (two_phase=True). At zero-init (iter-21) reg(x)≡0 => the projection-free denoiser D(x)=clamp(x-alpha*reg(x))=x is the IDENTITY => denoise_loss is a CONSTANT (frozen at the 1.9e-4 LD-FBP-vs-truth MSE floor) and the gradient that should lift the RBF self-vanishes. A small non-zero std makes reg(x)!=0 at the start of Phase A => D(x) actually denoises => gradients flow => the FoE LEARNS a real image-domain denoiser. Phase B then fine-tunes end-to-end from the TRAINED (non-degenerate) reg, so end-to-end stability rests on the trained bank, NOT the raw init (the zero-init-for-stability concern from the end-to-end-only iters does not apply once a pretrain provides the init). 0.0 => fall back to iter-21's broken zero-init (kept selectable for the post-mortem A/B).
    # ---- "steerable_foe" regulariser (iter-24 DEFAULT -- ROTATION-EQUIVARIANT FoE: analysis kernels synthesized from a SHARED FIXED steerable basis, ISO-PARAM to iter-7) ----
    "steer_n_rad":     7,     # iter-24: number of Gaussian-windowed RADIAL profiles in the steerable basis. With foe_kernel=7 and steer_max_order=3, Nb = n_rad*(1+2*M) = 7*7 = 49 = the COMPLETE 7x7 DOF, so analysis params = nf*Nb = 24*49 = 1,176 (BYTE-IDENTICAL to iter-7's free-kernel analysis budget).
    "steer_max_order": 3,     # iter-24: max angular circular-harmonic order M. order m=0 -> 1 radial-symmetric atom; each m in {1..M} -> a (cos m.theta, sin m.theta) PHASE PAIR. Nb = n_rad*(1 + 2*M). M=3 with n_rad=7 -> Nb=49 (complete 7x7 frame). The basis spans complete harmonic orders => an in-plane rotation maps filter responses among themselves (rotation-steerable by construction).
    "steer_coeff_init_std": 0.05,  # iter-24: std of the learned steerable coefficients c_{f,b}. The basis atoms are L2-normalised, so coeff_std=0.05 reproduces iter-7's free-kernel filter_init_std=0.05 per-element analysis scale -> the analysis bank starts statistically identical to iter-7, only organized in the steerable frame. The RBF synthesis is still ZERO-INIT (foe_rbf_init_std=0.0) so reg(x)==0 at init regardless (iter-7 byte-for-byte stability).
    # ---- "bilateral" regulariser (iter-17 DEFAULT — EXTREME param-efficiency: reg(x)=sum_i gain_i*(x-BF_i(x)), 4 trainable params/filter) ----
    "n_bf":            4,        # iter-17: 4 bilateral filters. reg = n_bf*(3 sigmas + 1 gain) trainable params = 4*4 = 16 (+1 scalar alpha = 17 total, 0.000017 M). Tiny but >1 so the bank can learn a few complementary edge-preserving bandwidths.
    "bf_kernel":       7,        # iter-17: 7x7 bilateral window (eff RF 7px/step ~ matches iter-7's k7 FoE; spatial weights are computed explicitly so NO trainable kernel weights — only the 3 sigmas are learnable).
    "bf_sigma_x":      1.5,      # iter-17: spatial-x bandwidth init (px). Learnable via log_sx.
    "bf_sigma_y":      1.5,      # iter-17: spatial-y bandwidth init (px). Learnable via log_sy.
    "bf_sigma_r":      0.02,     # iter-17: range (intensity) bandwidth init in mu units (clip_max=0.05). Learnable via log_sr; controls edge preservation.
    "bf_gain_init":    0.0,      # iter-17: ZERO-INIT per-filter gain => reg(x)=0 at init (stability, mirrors FoE zero-init rho'). Training lifts the gains off 0 to learn the bilateral correction; seed is iter-7's clean GD+DC byte-for-byte.
    # ---- ORDERED-SUBSETS DC training (iter-23: cut the TRAINING view count to buy more epochs; INFERENCE stays full-view) ----
    "train_view_subset": 4,     # iter-23: r => during TRAINING the DC term uses S = n_angles / r = 2304/4 = 576 random views per step (a reduced-angle projector + the strided sinogram g[:, off::r]). r=1 disables (full-view training = iter-7). r>1 makes the projection physics ~r x cheaper/step => more epochs in-budget. INFERENCE always uses the FULL n_angles=2304 DC.
    "train_view_random_offset": True,  # iter-23: re-randomize the angular offset off in {0..r-1} each step (ordered-subsets / SGD-over-views). False => fixed off=0 (deterministic decimation). The reduced projector for each (ps, off) is cached so the random offset costs only r distinct projectors per ps.
    # ---- TWO-PHASE trainer (iter-21/22: DEAD -- the FoE reg froze as a standalone denoiser with zero AND non-zero RBF init, hr 0 both times. OFF in iter-23.) ----
    "two_phase":        False,  # iter-23: OFF (DEAD). iter-21/22 proved the projection-free denoiser pretrain FREEZES (the reg only learns COUPLED through the DC unroll). True => the iter-21/22 two-phase path (kept selectable for the post-mortem). iter-23 uses the SINGLE-PHASE iter-7 trainer byte-for-byte + ordered-subsets DC.
    "pretrain_frac":    0.40,   # iter-21: fraction of max_train_s spent in Phase A (~432s of 1080s). Phase A is ~5x cheaper/step (no DC fwd/back-project) so this buys MANY image-domain epochs; Phase B gets the remaining ~648s for the expensive DC unroll from a warm reg.
    "pretrain_epochs":  60,     # iter-21: Phase-A epoch CAP (the cheap image-domain pass over pretrain_train_n; the pretrain_frac wall budget-cuts it first in practice). Generous so the wall, not the epoch count, is the binding limit.
    "pretrain_train_n": 0,      # iter-21: Phase-A train pool size; 0 => use train_n (200). Set up to 579 (full Mayo train pool) to give the reg ITNet-like data coverage in Phase A. Falls back to the available staged pool if smaller.
    "pretrain_cosine_t_max": 0, # iter-21: Phase-A cosine period in epochs; 0 => use pretrain_epochs (anneal over the planned A epochs). Phase A peak LR == lr (5e-3), eta_min == cosine_lr_min.
    # ---- training (iter-16: iter-7's EXACT stable partial-anneal regime, BYTE-FOR-BYTE — REVERTING iter-15's constant LR that DIVERGED) ----
    "train_n":   200,
    "val_n":     214,
    "epochs":    16,          # iter-16: 16 (iter-7's value, UNCHANGED). With cosine_lr=True and cosine_t_max=16, the partial cosine starts at 5e-3 and the hard max_train_s=1080 wall budget-cuts the run at ~ep8 (final LR ~2.5e-3) — iter-7's accidental-early-stop partial anneal that is the ONLY stable+optimal regime.
    "cosine_lr": True,        # iter-16: False -> True (REVERT iter-15). iter-15's CONSTANT 5e-3 with NO anneal DIVERGED (hr 0, psnr 11.38, val_rmse 0.0135 — same signature as iter-9's K=7). The trend-endpoint hypothesis is FALSIFIED: the partial-anneal dip toward 2.5e-3 was LOAD-BEARING for stability, not an overfitting tail. iter-7's partial cosine is restored byte-for-byte.
    "cosine_t_max": 16,       # iter-16: 8 -> 16 (iter-7's value: T_max == epochs == 16). This is the PARTIAL-anneal regime — the 16-ep cosine is BUDGET-CUT at ~ep8 by the 1080s wall (LR still ~2.5e-3, the anneal tail to 1e-5 is SKIPPED). NOT cosine_t_max=8 (iter-13's COMPLETED anneal -> 1e-5 which OVERFIT to 0.2273).
    "cosine_lr_min": 1e-5,    # iter-16: 1e-5 (iter-7 eta_min, UNCHANGED). With cosine_t_max=16 budget-cut at ~ep8 the run never reaches eta_min — exactly iter-7's behaviour.
    "max_train_s": 1080,      # iter-16: hard 18-min train backstop (UNCHANGED). A 9x9 analysis conv is marginally heavier per step than 7x7 but the per-step cost is DC-dominated (forward+back-project, unchanged); wall ~= iter-7's ~1331s observed, train-cut at ~ep8.
    "lr":        5e-3,        # iter-16: PEAK lr 5e-3 (UNCHANGED — NOT raised). iter-12's 2x peak (1e-2) DIVERGED; 5e-3 is the stability edge and the value iter-7 ran MONOTONE with no oscillation. Cosine anneals from this peak.
    "warmup_frac": 0.0,       # iter-16: KEEP 0.0 (no warmup, iter-7 behaviour). Warmup only mattered for the higher peak LR (NOT revived).
    "weight_decay": 0.0,      # iter-16: 0.0 (plain Adam, iter-7's optimiser exactly). iter-14 proved wd 3e-4 does NOT help. 0.0 => plain Adam (>0 => AdamW).
    "batch_size": 1,               # per-sample-ps geometry (Mayo): keep at 1
    "lambda_neg": 1.0,
    "grad_clip": 1.0,
    "seed":      42,
}


# ---------------------------------------------------------------------------
# Learned regularisers. Each module maps (B,1,H,W) -> (B,1,H,W); the value is
# the regulariser GRADIENT contribution `reg(x)` added inside the prox step.
# All are weight-tied (one instance reused across every unrolled iteration).
# ---------------------------------------------------------------------------
class CNNReg(nn.Module):
    """FLAT (single-scale, NO pooling) residual CNN regulariser (iter-6).

    `layers` 3x3 convs at `channels` channels with GroupNorm+ReLU between
    them and a zero-initialised 1x1 head, so reg(x) ≈ 0 at init (the seed
    therefore starts as a clean gradient-descent-with-DC scheme and learns a
    correction).

    iter-6 grows the iter-4 module WITHOUT any downsampling: it accepts a
    per-layer DILATION ladder (`dilations`) so the effective receptive field
    expands at FULL resolution. Each 3x3 conv is reflect-padded by its own
    dilation so spatial size is preserved exactly. With c=32, layers=5,
    dilations=[1,2,4,2,1] the eff RF is ~21px (vs ~7px at the iter-4
    c=12/3-layers/dil=1 default) — the long-range context that the pooled
    micro-UNet bought (iter-2/iter-5), but with ZERO pooling, which iter-5
    proved to be the architectural failure mode here.

    `dilations` may be None (all dilation 1, the iter-4 behaviour), a single
    int (applied to every conv), or a list/tuple — recycled/truncated to
    `layers` entries so the cfg ladder stays robust to off-by-one."""

    def __init__(self, channels: int = 16, layers: int = 3,
                 dilations=None):
        super().__init__()
        channels = int(channels)
        layers = max(1, int(layers))
        # Normalise dilations -> a length-`layers` list of positive ints.
        if dilations is None:
            dil = [1] * layers
        elif isinstance(dilations, (int, float)):
            dil = [max(1, int(dilations))] * layers
        else:
            seq = [max(1, int(d)) for d in dilations] or [1]
            dil = [seq[i % len(seq)] for i in range(layers)]

        def _conv(ci, co, d):
            # reflect-pad by `d` so a dilated 3x3 conv keeps H,W exactly.
            return nn.Conv2d(ci, co, 3, padding=d, dilation=d,
                             padding_mode="reflect")

        body: list[nn.Module] = [_conv(1, channels, dil[0])]
        for li in range(1, layers):
            body += [nn.GroupNorm(_pick_groups(channels), channels),
                     nn.ReLU(inplace=True),
                     _conv(channels, channels, dil[li])]
        self.body = nn.Sequential(*body)
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(self.body(x)))


def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


class MicroUNetReg(nn.Module):
    """Multi-scale 2-level (one-downsample) micro-UNet regulariser (iter-2).

    Why: the iter-1 single-scale CNNReg had an effective receptive field of
    only ~7px and saturated at hr=0.087 — too small to model Mayo's
    spatially-correlated streak/low-dose noise. This adds ONE coarse branch
    (avg-pool/2 -> double-conv -> bilinear-up, concat skip) so the tied prox
    step sees both fine (3x3 @ full res) and coarse (effective ~14px @
    half-res) structure, at ~16k params (c=16) — one order under the 233k
    SmallUNet champion. Still ONE instance reused at every unrolled step
    (weight-tied). Zero-init 1x1 head ⇒ reg(x) ≈ 0 at init, so the seed
    starts as clean GD+DC and learns a correction (mirrors CNNReg/SmallUNet).
    REGRESSED in iter-2 (below LD-FBP floor): undertrainable in 20-min budget.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        c = int(channels)
        g = _pick_groups(c)
        g2 = _pick_groups(2 * c)

        def dconv(ci, co, gn):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1),
                nn.GroupNorm(gn, co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1),
                nn.GroupNorm(gn, co), nn.ReLU(inplace=True),
            )

        self.enc = dconv(1, c, g)               # full-res encoder
        self.down = dconv(c, 2 * c, g2)         # half-res branch (coarse)
        self.dec = dconv(c + 2 * c, c, g)       # fuse coarse(up) + fine skip
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # Pad to even dims so the single pool/upsample round-trips exactly.
        ph = h % 2
        pw = w % 2
        x_in = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        e = self.enc(x_in)                                  # (B,c,H,W)
        d = self.down(F.avg_pool2d(e, 2))                  # (B,2c,H/2,W/2)
        u = F.interpolate(d, size=e.shape[-2:], mode="bilinear",
                          align_corners=False)             # (B,2c,H,W)
        y = self.dec(torch.cat([u, e], dim=1))             # (B,c,H,W)
        out = self.head(y)
        if ph or pw:
            out = out[..., :h, :w]
        return out


class FoEReg(nn.Module):
    """Single tied Fields-of-Experts / VN filter bank (iter-7 DEFAULT).

    reg(x) = K^T ρ'(K x), with K an analysis conv2d bank (n_filters,
    kernel x kernel), ρ' a per-filter RBF mixture (n_bumps bumps), and K^T
    the tied conv_transpose2d synthesis — exactly the regulariser-gradient
    of one solver_hammernik_vn.py VNStep, but ONE bank reused at every
    unrolled step (weight-tied) instead of T untied banks.

    ZERO-INIT SYNTHESIS (iter-7 stability fix): with `rbf_init_std=0.0` the
    per-filter RBF mixture weights start at 0, so ρ'(·) ≡ 0 and therefore
    reg(x) ≡ 0 at init. The seed thus starts as the EXACT clean GD+DC scheme
    (the reg contributes nothing) and LEARNS the filter bank as a correction
    — the same proven zero-init-output pattern as CNNReg/MicroUNet's zero-init
    1x1 head, ported to the FoE family so this very-different reg can be
    dropped into the known-stable iter-4 trainer without destabilising. The
    analysis filters K remain randomly initialised (filter_init_std) and fully
    learnable; only the OUTPUT magnitude is gated to 0 by the zero ρ'-weights
    at step 0."""

    def __init__(self, n_filters: int = 8, kernel_size: int = 5,
                 n_bumps: int = 15, x_range: float = 1.0,
                 filter_init_std: float = 0.05, rbf_init_std: float = 0.01):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.n_bumps = int(n_bumps)
        self.weight = nn.Parameter(
            torch.randn(self.n_filters, 1, self.kernel_size, self.kernel_size)
            * filter_init_std)
        centres = torch.linspace(-x_range, x_range, self.n_bumps)
        sigma = 2.0 * x_range / max(1, self.n_bumps - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("inv_sigma_sq", torch.tensor(1.0 / (sigma ** 2)))
        self.rbf_weights = nn.Parameter(
            torch.randn(self.n_filters, self.n_bumps) * rbf_init_std)

    def _rho_prime(self, Kx: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(Kx)
        for j in range(self.n_bumps):
            mu_j = self.centres[j]
            bump = torch.exp(-0.5 * (Kx - mu_j) ** 2 * self.inv_sigma_sq)
            w_j = self.rbf_weights[:, j].view(1, self.n_filters, 1, 1)
            out = out + bump * w_j
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        Kx = F.conv2d(x, self.weight, padding=pad)
        rho_Kx = self._rho_prime(Kx)
        return F.conv_transpose2d(rho_Kx, self.weight, padding=pad)


def _build_steerable_basis(kernel_size: int, n_rad: int, max_order: int):
    """Construct a FIXED steerable (separable RADIAL x ANGULAR) basis for a
    kernel_size x kernel_size filter, as an (Nb, k, k) tensor of L2-normalised
    real atoms (Nb = n_rad * (1 + 2*max_order)).

    The frame is a circular-harmonic basis: n_rad Gaussian-windowed radial
    profiles (rings at evenly spaced radii) crossed with angular orders
    m in {0..max_order}. Order m=0 is radially symmetric (1 atom per radial
    profile); each m>0 contributes a PHASE PAIR cos(m.theta), sin(m.theta).
    Because the basis spans COMPLETE harmonic orders, the span is closed under
    in-plane rotation (a rotation maps each order's (cos, sin) pair into a linear
    combination of itself), so any filter synthesized as a linear combination of
    these atoms is ROTATION-STEERABLE: its whole orientation orbit lives in the
    same span. With n_rad=7, max_order=3, k=7 -> Nb = 7*7 = 49 = the complete
    7x7 degrees-of-freedom, so the span is the entire kernel space but expressed
    in the orientation-organized harmonic frame (iso-DOF to a free 7x7 kernel).

    Atoms are L2-normalised so a coefficient std reproduces a free-kernel
    element-wise std of the same value (the analysis bank starts statistically
    identical to a free-kernel FoE bank, only organized in the steerable frame).
    Returned as a plain tensor (registered as a non-trainable buffer by the
    caller); 0 trainable params.
    """
    k = int(kernel_size)
    c = (k - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(k, dtype=torch.float32) - c,
        torch.arange(k, dtype=torch.float32) - c,
        indexing="ij")
    rr = torch.sqrt(xs ** 2 + ys ** 2)
    theta = torch.atan2(ys, xs)
    r_max = float(rr.max().clamp(min=1e-6))
    # n_rad ring centres from 0..r_max; sigma ~ inter-ring spacing.
    n_rad = max(1, int(n_rad))
    if n_rad == 1:
        centres = [0.0]
        sigma = max(r_max, 1.0)
    else:
        centres = [r_max * i / (n_rad - 1) for i in range(n_rad)]
        sigma = r_max / (n_rad - 1)
    sigma = max(float(sigma), 1e-3)
    atoms = []
    for rc in centres:
        radial = torch.exp(-0.5 * ((rr - rc) / sigma) ** 2)
        for m in range(int(max_order) + 1):
            if m == 0:
                atoms.append(radial.clone())
            else:
                atoms.append(radial * torch.cos(m * theta))
                atoms.append(radial * torch.sin(m * theta))
    B = torch.stack(atoms, dim=0)                       # (Nb, k, k)
    # L2-normalise each atom (so coeff_std maps to free-kernel element std).
    flat = B.view(B.shape[0], -1)
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    B = (flat / norms).view_as(B)
    return B


class SteerableFoEReg(nn.Module):
    """ROTATION-EQUIVARIANT / STEERABLE Fields-of-Experts bank (iter-24 DEFAULT).

    Identical to FoEReg EXCEPT the analysis filters K are not free k x k kernels;
    each filter is SYNTHESIZED as a learned linear combination of a SHARED, FIXED
    steerable basis B (n_rad radial profiles x circular-harmonic orders 0..M):

        K_f = sum_b  coeff_{f,b} * B_b      (coeff learned; B_b fixed buffers)

    Because B spans COMPLETE harmonic orders, the bank is rotation-steerable by
    construction: one learned filter implicitly represents its whole orientation
    orbit (CT noise/structure has no preferred orientation), so the analysis bank
    should denoise more efficiently per trainable weight than a free-kernel bank.

    ISO-PARAM to iter-7's FoE (option (a)): at n_rad=7, M=3, k=7 the basis has
    Nb = 49 atoms (the COMPLETE 7x7 DOF), so analysis = n_filters*Nb = 24*49 =
    1,176 trainable coeffs -- BYTE-IDENTICAL to iter-7's 24 free 7x7 kernels
    (24*49=1,176). The 49 basis atoms are FIXED buffers (0 trainable). With the
    RBF synthesis still ZERO-INIT (rbf_init_std=0.0) rho'(.)==0 => reg(x)==0 at
    init regardless of the coeffs (iter-7 byte-for-byte stability). The coeffs
    are init at coeff_init_std (=0.05) and, since the atoms are L2-normalised,
    the effective 7x7 kernels start with the SAME element-wise std as iter-7's
    filter_init_std=0.05 free kernels.

    The forward is the FoE regulariser-gradient reg(x) = K^T rho'(K x), with K
    materialised once per call from (coeff @ B) and the SAME tied analysis/
    synthesis kernel (conv2d then conv_transpose2d) as FoEReg -- byte-for-byte
    the iter-7 unroll dynamics; only the analysis kernels' parameterization
    changes (free 49-vector -> 49 steerable coeffs over a fixed complete frame).
    """

    def __init__(self, n_filters: int = 24, kernel_size: int = 7,
                 n_bumps: int = 31, x_range: float = 1.0,
                 n_rad: int = 7, max_order: int = 3,
                 coeff_init_std: float = 0.05, rbf_init_std: float = 0.0):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.n_bumps = int(n_bumps)
        B = _build_steerable_basis(self.kernel_size, n_rad, max_order)  # (Nb,k,k)
        self.register_buffer("basis", B)                # FIXED, 0 trainable params
        self.n_basis = int(B.shape[0])
        # Learned steerable coefficients: (n_filters, Nb). These ARE the analysis
        # params (1,176 at nf=24/Nb=49), replacing FoEReg's 24 free 7x7 kernels.
        self.coeff = nn.Parameter(
            torch.randn(self.n_filters, self.n_basis) * float(coeff_init_std))
        centres = torch.linspace(-x_range, x_range, self.n_bumps)
        sigma = 2.0 * x_range / max(1, self.n_bumps - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("inv_sigma_sq", torch.tensor(1.0 / (sigma ** 2)))
        # ZERO-INIT RBF synthesis (iter-7 byte-for-byte): rho'==0 => reg(x)==0 at init.
        self.rbf_weights = nn.Parameter(
            torch.randn(self.n_filters, self.n_bumps) * float(rbf_init_std))

    def _effective_weight(self) -> torch.Tensor:
        """Synthesize the (n_filters, 1, k, k) analysis kernels from the learned
        steerable coeffs over the fixed basis: K_f = sum_b coeff_{f,b} * B_b."""
        # (nf, Nb) @ (Nb, k*k) -> (nf, k*k) -> (nf, 1, k, k)
        k = self.kernel_size
        K = self.coeff @ self.basis.view(self.n_basis, -1)
        return K.view(self.n_filters, 1, k, k)

    def _rho_prime(self, Kx: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(Kx)
        for j in range(self.n_bumps):
            mu_j = self.centres[j]
            bump = torch.exp(-0.5 * (Kx - mu_j) ** 2 * self.inv_sigma_sq)
            w_j = self.rbf_weights[:, j].view(1, self.n_filters, 1, 1)
            out = out + bump * w_j
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        w_eff = self._effective_weight()
        Kx = F.conv2d(x, w_eff, padding=pad)
        rho_Kx = self._rho_prime(Kx)
        return F.conv_transpose2d(rho_Kx, w_eff, padding=pad)


class BilateralReg(nn.Module):
    """Trainable EDGE-PRESERVING bilateral regulariser (iter-17 DEFAULT — the
    EXTREME param-efficiency frontier: ~tens of trainable params).

    reg(x) = sum_i  gain_i * (x - BF_i(x))

    a sum of `n_bf` TrainableBilateralFilter2d (Wagner et al. Med. Phys. 2022,
    ddssl_ldct/models.py:92) denoise-then-subtract RESIDUALS, each scaled by its
    own learnable scalar `gain_i`. Each BF has 3 learnable params (log σx, log
    σy, log σr — spatial-x, spatial-y, range bandwidths); the per-filter gain
    adds 1 more => 4 trainable params PER FILTER. At n_bf=4 the reg has
    4*(3+1) = 16 trainable params (total 17 incl. the 1 scalar alpha) — TWO
    orders of magnitude below iter-7's 1,921-param FoE, and a clean
    "hr-at-~tens-of-params" datapoint on the param/headroom Pareto frontier.

    Why a SUM of residuals (not the iter-pre-17 sequential CASCADE): a parallel
    bank of scaled residuals (i) keeps reg(x) a simple superposition of
    edge-preserving smoothing corrections — exactly the gradient of a sum of
    bilateral-prior energy terms — and (ii) makes the per-filter gain a clean
    on/off knob so a filter that is not helping can be zeroed out by training.

    ZERO-INIT GAIN (the iter-17 stability fix, mirroring iter-7's zero-init ρ'):
    with `gain_init=0.0` every gain_i starts at 0, so reg(x) ≡ 0 at init
    REGARDLESS of the bilateral σ values. The seed is therefore the EXACT clean
    GD+DC scheme of iter-7 (the reg contributes nothing) and training LEARNS the
    bilateral correction by lifting the gains off zero — the same proven
    zero-init-OUTPUT pattern as CNNReg's zero head / FoEReg's zero ρ'-weights,
    ported to the bilateral family so this very-different (and far tinier) reg
    drops into the known-stable iter-7 trainer without destabilising. The σ
    bandwidths stay at their (interpretable, nonzero) init and are fully
    learnable; only the OUTPUT magnitude is gated to 0 by the zero gains at
    step 0. NO pooling (dodges iter-2/5), NO extra unroll stage (dodges
    iter-10/11), NO added depth (dodges iter-9); a bilateral filter is a single
    EDGE-AWARE smoothing — the recon DYNAMICS (K=5, single scalar alpha, plain
    prox step) are byte-for-byte iter-7."""

    def __init__(self, n_bf: int = 4, kernel_size: int = 7,
                 sigma_x: float = 1.5, sigma_y: float = 1.5,
                 sigma_r: float = 0.02, gain_init: float = 0.0):
        super().__init__()
        n_bf = max(1, int(n_bf))
        self.filters = nn.ModuleList(
            [TrainableBilateralFilter2d(kernel_size=int(kernel_size),
                                        sigma_x=float(sigma_x),
                                        sigma_y=float(sigma_y),
                                        sigma_r=float(sigma_r))
             for _ in range(n_bf)])
        # ONE learnable scalar gain per filter, ZERO-INIT => reg(x) ≡ 0 at init
        # (stability; mirrors FoEReg's zero-init ρ'-weights / CNNReg's zero head).
        self.gains = nn.Parameter(torch.full((n_bf,), float(gain_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for i, f in enumerate(self.filters):
            out = out + self.gains[i] * (x - f(x))
        return out


class LearnedInit(nn.Module):
    """Tiny zero-init residual refiner of the LD-FBP init (iter-11 RE-WIRED).

    A single-scale residual conv `g` applied ONCE to the LD-FBP before the
    unroll: the model uses x_0 = LD-FBP + g(LD-FBP) (NO upper clamp on the init
    -- the iter-10 BUG was clamping x_0 to clip_max, which truncated the bright
    LD-FBP pixels and broke byte-for-byte parity with iter-7). It is a SMOOTH
    path -- conv(1->channels, 3x3, reflect-padded) -> GELU -> conv(channels->1,
    1x1) -- with BOTH the 1x1 head's weight AND bias ZERO-INITIALISED, so
    g(.) ≡ 0 at init and therefore x_0 == LD-FBP EXACTLY (byte-for-byte) at init.
    The seed is the iter-7 champion byte-for-byte, and training LEARNS a
    lower-RMSE init as a correction (the same proven zero-init-output stability
    pattern as CNNReg's head / FoEReg's rho'-weights).

    Why GELU not ReLU (the dead-ReLU-before-zero-head trap fix): with a ReLU
    feeding a zero-init head, the head's gradient w.r.t. the body weights routes
    through ReLU' which is 0 on half the activations at init; combined with the
    zero head this can leave the refiner unable to escape g≡0. GELU is smooth
    and nonzero-derivative everywhere, so once the zero head's own weights start
    to move (driven by the supervised loss), gradient flows back into the body
    cleanly. NO pooling (iter-2/5's failure mode); eff RF ~3px at layers=1, i.e.
    the smallest possible learned-init -- a local FBP-noise/DC refiner, NOT a
    denoiser. At channels=16, layers=1 it is (16*9+16) (3x3 conv w+b) + (16+1)
    (1x1 head w+b) = 177 params, applied OUTSIDE the unroll loop so it adds neither
    unroll depth nor reg capacity."""

    def __init__(self, channels: int = 16, layers: int = 1):
        super().__init__()
        channels = int(channels)
        layers = max(1, int(layers))

        def _conv(ci, co, k):
            return nn.Conv2d(ci, co, k, padding=k // 2, padding_mode="reflect")

        body: list[nn.Module] = [_conv(1, channels, 3)]
        for _ in range(1, layers):
            body += [nn.GELU(), _conv(channels, channels, 3)]
        body += [nn.GELU()]
        self.body = nn.Sequential(*body)
        # ZERO-INIT 1x1 head (weight AND bias) => g(.) ≡ 0 at init.
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def build_reg(cfg: dict) -> nn.Module:
    rt = cfg["reg_type"]
    if rt == "microunet":
        return MicroUNetReg(channels=cfg.get("mu_channels", 16))
    if rt == "cnn":
        return CNNReg(channels=cfg["cnn_channels"], layers=cfg["cnn_layers"],
                      dilations=cfg.get("cnn_dilations"))
    if rt == "foe":
        return FoEReg(n_filters=cfg["foe_n_filters"], kernel_size=cfg["foe_kernel"],
                      n_bumps=cfg["foe_n_bumps"], x_range=cfg["foe_x_range"],
                      filter_init_std=cfg["foe_filter_init_std"],
                      rbf_init_std=cfg["foe_rbf_init_std"])
    if rt == "steerable_foe":
        return SteerableFoEReg(
            n_filters=cfg["foe_n_filters"], kernel_size=cfg["foe_kernel"],
            n_bumps=cfg["foe_n_bumps"], x_range=cfg["foe_x_range"],
            n_rad=cfg.get("steer_n_rad", 7),
            max_order=cfg.get("steer_max_order", 3),
            coeff_init_std=cfg.get("steer_coeff_init_std", 0.05),
            rbf_init_std=cfg["foe_rbf_init_std"])
    if rt == "bilateral":
        return BilateralReg(n_bf=cfg["n_bf"], kernel_size=cfg["bf_kernel"],
                            sigma_x=cfg.get("bf_sigma_x", 1.5),
                            sigma_y=cfg.get("bf_sigma_y", 1.5),
                            sigma_r=cfg.get("bf_sigma_r", 0.02),
                            gain_init=cfg.get("bf_gain_init", 0.0))
    raise ValueError(f"unknown reg_type={rt!r} (expected microunet|cnn|foe|steerable_foe|bilateral)")


# ---------------------------------------------------------------------------
class ParamEfficientUnrolled(nn.Module):
    """Weight-tied unrolled proximal-gradient with data consistency (iter-4:
    iter-1's EXACT stable recipe), iter-11: + a CORRECTLY-WIRED learned init.

    x_0 = u0 + g(u0)  (g ≡ 0 at init => x_0 == LD-FBP byte-for-byte; NO upper
    clamp on the init -- the iter-10 BUG was clamping x_0 to clip_max).
    For k in range(n_iter):
        dc = R^T(R x - g) / dc_norm
        x  = clamp(x - alpha * (dc + reg(x)), 0.0, clip_max)
    `reg` AND `alpha` are SHARED across all steps (weight-tied). NO per-step
    alpha, NO momentum — iter-3 proved both destabilise the recon in-budget.

    For backward-compatibility the per_step_alpha / momentum config knobs are
    still honoured (so the iter-3 variant remains selectable), but the iter-4
    DEFAULTS turn BOTH off, collapsing the iteration to the plain tied
    prox-gradient that iter-1 ran (and the ONLY config that cleared the floor).
    With per_step_alpha=False and momentum=False:
        y == x at every step (no look-ahead), v unused, beta == 0.
    """

    def __init__(self, projector: PyronnFanBeamProjector, cfg: dict,
                 dc_norm: float = 1.0):
        super().__init__()
        self.projector = projector             # shared single instance, not a sub-module
        self.n_iter = int(cfg["n_iter"])
        self.clip_max = float(cfg["clip_max"])
        self.checkpoint = bool(cfg.get("checkpoint", True))
        self.per_step_alpha = bool(cfg.get("per_step_alpha", False))
        self.use_momentum = bool(cfg.get("momentum", False))
        self.register_buffer("dc_norm", torch.tensor(float(dc_norm)))
        self.reg = build_reg(cfg)              # ONE tied regulariser
        # iter-11 RE-WIRED STAGE: a tiny ZERO-INIT learned refiner of the
        # LD-FBP, applied ONCE before the unroll (OUTSIDE the loop). g≡0 at init
        # so x_0 == LD-FBP byte-for-byte => the seed is iter-7 byte-for-byte.
        # NO upper clamp on the init (the iter-10 BUG); the unroll's per-step
        # clamp handles range, exactly as iter-7's x=u0 raw init.
        self.use_learned_init = bool(cfg.get("learned_init", False))
        self.init_clamp = bool(cfg.get("init_clamp", False))   # iter-11: False (iter-10 BUG was True)
        if self.use_learned_init:
            self.init_refiner = LearnedInit(
                channels=int(cfg.get("init_channels", 16)),
                layers=int(cfg.get("init_layers", 1)))
        else:
            self.init_refiner = None
        # Step size(s): one tied scalar (iter-4 default), OR a length-n_iter
        # vector (per_step, iter-3 variant — off by default).
        if cfg["learnable_alpha"]:
            inv_softplus = math.log(math.expm1(max(float(cfg["alpha_init"]), 1e-6)))
            n = self.n_iter if self.per_step_alpha else 1
            self.log_alpha = nn.Parameter(torch.full((n,), float(inv_softplus)))
            self._alpha_const = None
        else:
            self.log_alpha = None
            self._alpha_const = float(cfg["alpha_init"])
        # Single tied momentum coefficient beta in (0,1) via sigmoid (off by
        # default in iter-4; iter-3 variant only).
        if self.use_momentum:
            b0 = min(max(float(cfg.get("beta_init", 0.5)), 1e-4), 1 - 1e-4)
            inv_sig = math.log(b0 / (1.0 - b0))
            self.beta_raw = nn.Parameter(torch.tensor(float(inv_sig)))
        else:
            self.beta_raw = None

    @property
    def alpha(self) -> torch.Tensor:
        """Back-compat scalar view (mean over steps) for logging."""
        if self.log_alpha is not None:
            return F.softplus(self.log_alpha).mean()
        return torch.as_tensor(self._alpha_const, device=self.dc_norm.device,
                               dtype=self.dc_norm.dtype)

    def _alpha_k(self, k: int) -> torch.Tensor:
        if self.log_alpha is not None:
            idx = k if self.per_step_alpha else 0
            return F.softplus(self.log_alpha[idx])
        return torch.as_tensor(self._alpha_const, device=self.dc_norm.device,
                               dtype=self.dc_norm.dtype)

    @property
    def beta(self) -> torch.Tensor:
        if self.beta_raw is not None:
            return torch.sigmoid(self.beta_raw)
        return torch.zeros((), device=self.dc_norm.device, dtype=self.dc_norm.dtype)

    def refined_init(self, u0: torch.Tensor) -> torch.Tensor:
        """The learned-init x_0 = u0 + g(u0) (clamp_min(0) only; NO upper clamp
        unless init_clamp is the buggy iter-10 mode). Exposed so the runtime
        self-check can verify x_0 == u0 byte-for-byte at init."""
        if self.init_refiner is None:
            return u0
        x0 = u0 + self.init_refiner(u0)
        if self.init_clamp:
            # iter-10 BUGGY behaviour (kept selectable for the post-mortem A/B).
            return torch.clamp(x0, 0.0, self.clip_max)
        # iter-11 FIX: clamp_min(0) only (no-op since LD-FBP >= 0), so at init
        # (g≡0) x_0 == u0 EXACTLY, byte-for-byte iter-7.
        return x0.clamp_min(0.0)

    def _grad(self, y: torch.Tensor, sino: torch.Tensor,
              dc_norm: torch.Tensor) -> torch.Tensor:
        """Prox-gradient direction at the (look-ahead) point y.

        iter-23: `dc_norm` is passed IN (not read from self.dc_norm) so the
        ordered-subsets TRAINING path can supply the reduced operator's
        ‖R_SᵀR_S‖ while the model's `self.projector` is the reduced R_S; at
        INFERENCE the caller passes self.dc_norm (the full ‖RᵀR‖) and the full
        projector, byte-for-byte iter-7."""
        R_y = self.projector.forward_project(y)
        dc = self.projector.back_project(R_y - sino) / dc_norm
        return dc + self.reg(y)

    def _step(self, x: torch.Tensor, y: torch.Tensor, sino: torch.Tensor,
              alpha_k: torch.Tensor, beta: torch.Tensor,
              dc_norm: torch.Tensor):
        """One (optionally accelerated) prox-gradient step. Returns (x_new,
        y_new). With beta==0 (iter-4 default) y_new == x_new, i.e. the plain
        tied prox step of iter-1. Checkpointed: all tensor args/returns so
        grads flow through any carried velocity/look-ahead state.
        iter-23: `dc_norm` is threaded through so the checkpointed step uses the
        reduced operator's norm during ordered-subsets training."""
        x_new = torch.clamp(y - alpha_k * self._grad(y, sino, dc_norm),
                            0.0, self.clip_max)
        v = x_new - x
        y_new = x_new + beta * v
        return x_new, y_new

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Phase-A (iter-21) PROJECTION-FREE denoiser: ONE prox step with the
        DATA-CONSISTENCY term ZEROED. D(x) = clamp(x - alpha * reg(x), 0, clip_max).

        This is a STRICT SUBSET of the iter-7 unrolled step (the DC fwd/back-
        project dropped), so it trains EXACTLY the FoE reg + the tied scalar
        alpha that the end-to-end Phase B fine-tunes — the pretrained bank drops
        in WARM. With zero-init rho' (reg(x)=0 at init) D(x)=x at init, so Phase
        A starts as the identity and learns the image-domain denoiser. NO
        projection in the loop => ~5x cheaper/step than the K=5 DC unroll, so
        Phase A fits MANY epochs over the full train pool in a small time slice.
        Uses alpha at step 0 (the single tied scalar when per_step_alpha=False)."""
        alpha0 = self._alpha_k(0)
        return torch.clamp(x - alpha0 * self.reg(x), 0.0, self.clip_max)

    def forward(self, u0: torch.Tensor, sino: torch.Tensor,
                dc_norm: torch.Tensor | None = None) -> torch.Tensor:
        # iter-11: refine the LD-FBP init ONCE (outside the unroll). The
        # refiner's head is zero-init so at init x_0 == u0 (LD-FBP) byte-for-byte
        # (no upper clamp -- the iter-10 BUG fix); the unroll's per-step clamp
        # handles range, exactly as iter-7's x=u0 raw init.
        # iter-23: `dc_norm` defaults to the build-time FULL ‖RᵀR‖ (self.dc_norm,
        # used at INFERENCE and full-view training). The ordered-subsets TRAINING
        # path passes the REDUCED operator's ‖R_SᵀR_S‖ together with the reduced
        # projector swapped into self.projector and the strided sinogram in `sino`.
        if dc_norm is None:
            dc_norm = self.dc_norm
        x0 = self.refined_init(u0)
        x = x0
        y = x0                                   # look-ahead == x at k=0 (v=0)
        beta = self.beta                         # == 0 when momentum off
        for k in range(self.n_iter):
            alpha_k = self._alpha_k(k)
            if self.checkpoint and y.requires_grad:
                x, y = torch.utils.checkpoint.checkpoint(
                    self._step, x, y, sino, alpha_k, beta, dc_norm,
                    use_reentrant=False)
            else:
                x, y = self._step(x, y, sino, alpha_k, beta, dc_norm)
        return x


# ---------------------------------------------------------------------------
def build_dataset(geom, n, seed, i0, sigma_e, device):
    # Dispatches on AGENT4CT_DATASET / cfg["dataset_kind"]. Phantom path
    # is backwards-compatible; staged paths load from disk. Split is picked
    # from the existing seed convention (train: seed=cfg["seed"]; val:
    # seed=cfg["seed"]+1000).
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)


def _count_reg_params(cfg: dict) -> int:
    """Trainable param count of ONE regulariser at the given cfg (for the
    start-of-run print + sanity check; the tied model reuses this once)."""
    reg = build_reg(cfg)
    return sum(p.numel() for p in reg.parameters() if p.requires_grad)


def _power_iter_norm(projector, image_size: int, device, iters: int = 8) -> float:
    """Power-iteration estimate of ‖RᵀR‖ for a given projector (so alpha lives
    in O(1) regardless of the angular sampling). Mirrors the full-operator
    estimate done once in main(); reused here for the reduced subset operators
    so the per-step alpha scale tracks the (smaller) ‖R_SᵀR_S‖."""
    with torch.no_grad():
        v = torch.randn(1, 1, image_size, image_size, device=device)
        v = v / v.norm()
        n = torch.tensor(1.0, device=device)
        for _ in range(iters):
            Av = projector.forward_project(v)
            v = projector.back_project(Av)
            n = v.norm().clamp(min=1e-12)
            v = v / n
        return float(n.item())


def build_subset_projectors(ps_values, r: int, n_angles: int, n_det: int,
                            image_size: int, device):
    """ORDERED-SUBSETS reduced-angle projectors for iter-23 TRAINING.

    For each distinct recon pixel-spacing `ps` and each angular offset
    `off in {0..r-1}`, build a fan-beam projector over S = n_angles // r UNIFORM
    views whose trajectory starts at angle 2π·off/n_angles and spans a full 2π,
    so its S sample angles land EXACTLY on the strided sinogram columns
    g[:, off::r] (geometrically exact decimation, not an approximation). Each
    reduced projector also gets its OWN power-iteration ‖R_SᵀR_S‖.

    Returns ``(subset_projs, subset_norms, S)`` where
      subset_projs[(round(ps,5), off)] -> PyronnFanBeamProjector (S views)
      subset_norms[(round(ps,5), off)] -> float ‖R_SᵀR_S‖
    so the training loop looks up the (ps, off) pair per step.
    """
    import math as _math
    r = max(1, int(r))
    S = int(n_angles) // r
    subset_projs: dict = {}
    subset_norms: dict = {}
    uniq = sorted({round(float(p), 5) for p in np.asarray(ps_values, float)})
    for ps in uniq:
        for off in range(r):
            a0 = 2.0 * _math.pi * float(off) / float(n_angles)
            # S uniform views over a FULL 2π starting at the offset angle; the
            # m-th view is at a0 + 2π·m/S = 2π·(off + m·r)/n_angles -> exactly
            # the angles of g[:, off::r] (the m-th strided column). Canonical
            # Mayo detector/source geometry (matches mayo_proj_cache) but with
            # the offset angular range + S views.
            geom = FanBeamGeometry(
                image_size=int(image_size), pixel_spacing=float(ps),
                n_angles=int(S), n_det=int(n_det),
                det_spacing=1.285044, sod=595.362, sdd=1086.803,
                angle_start=a0, angle_end=a0 + 2.0 * _math.pi)
            pr = PyronnFanBeamProjector(geom).to(device)
            subset_projs[(ps, off)] = pr
            subset_norms[(ps, off)] = _power_iter_norm(pr, int(image_size), device)
    print(f"[iter-23] ordered-subsets: r={r} S={S} views/step over "
          f"{len(uniq)} ps x {r} offsets = {len(subset_projs)} reduced projectors; "
          f"norms[min,max]=[{min(subset_norms.values()):.3g},"
          f"{max(subset_norms.values()):.3g}]", flush=True)
    return subset_projs, subset_norms, S


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("PARAM_EFFICIENT_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        env_cfg = json.loads(Path(env_path).read_text())
        cfg = {**CONFIG, **env_cfg, **(cfg or {})}
        print(f"[solver] Loaded config from {env_path}", flush=True)
    else:
        cfg = {**CONFIG, **(cfg or {})}

    # Dataset dispatch (Track B/C of workplan). When dataset_kind != "phantoms"
    # we override the geometry to match the staged data.
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}", flush=True)
    print(f"[solver] config={json.dumps({k: v for k, v in cfg.items() if k in ('reg_type','n_iter','learnable_alpha','per_step_alpha','momentum','beta_init','alpha_init','clip_max','mu_channels','cnn_channels','cnn_layers','cnn_dilations','foe_n_filters','foe_kernel','foe_n_bumps','foe_rbf_init_std','foe_rbf_init_std_pretrain','steer_n_rad','steer_max_order','steer_coeff_init_std','n_bf','bf_kernel','bf_sigma_x','bf_sigma_y','bf_sigma_r','bf_gain_init','learned_init','init_channels','init_layers','init_clamp','two_phase','pretrain_frac','pretrain_epochs','pretrain_train_n','pretrain_cosine_t_max','train_view_subset','train_view_random_offset','epochs','cosine_lr','cosine_t_max','cosine_lr_min','max_train_s','batch_size','lr','warmup_frac','weight_decay','train_n','val_n')}, default=str)}",
          flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    train_ph, train_clean, train_noisy, train_ps = build_dataset(
        geom, cfg["train_n"], cfg["seed"],
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # PER-SAMPLE geometry (Mayo canonical): swap model.projector per slice +
    # build the per-ps FBP init/baseline (falls back to single proj non-mayo).
    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        if per_ps:
            train_u0 = mayo_per_sample_fbp(_projs, _trk, train_noisy, cfg["image_size"])
            val_u0   = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            train_u0 = torch.clamp(proj.fbp(train_noisy), min=0.0)
            val_u0   = torch.clamp(proj.fbp(val_noisy),   min=0.0)

    # Power-iteration estimate of ‖R^T R‖ so alpha stays in O(1).
    norm_val = 1.0
    if cfg.get("dc_norm", True):
        with torch.no_grad():
            v = torch.randn(1, 1, cfg["image_size"], cfg["image_size"], device=device)
            v = v / v.norm()
            for _ in range(8):
                Av = proj.forward_project(v)
                v = proj.back_project(Av)
                n = v.norm().clamp(min=1e-12)
                v = v / n
            norm_val = float(n.item())
            print(f"[solver] dc_norm power-iter ≈ {norm_val:.3g}", flush=True)

    model = ParamEfficientUnrolled(proj, cfg, dc_norm=norm_val).to(device)

    # iter-11 RUNTIME SELF-CHECK (the guard iter-10 lacked): right after building
    # the model, on ONE val sample, verify the learned-init is TRULY zero at init
    # so x_0 == LD-FBP byte-for-byte. rel = ‖g(LD_FBP)‖ / ‖LD_FBP‖ must be < 1e-6.
    # A non-zero-init bug (the iter-10 failure mode) FAILS LOUDLY here, BEFORE the
    # 20-min run is wasted. Also reports ‖x_0 - LD_FBP‖ to catch any init clamp.
    if model.use_learned_init and val_u0.shape[0] > 0:
        with torch.no_grad():
            if per_ps:
                model.projector = _projs[float(_vrk[0])]
            u0_chk = val_u0[0:1]
            g_out = model.init_refiner(u0_chk)
            u0_norm = float(u0_chk.norm().clamp(min=1e-12))
            rel_g = float(g_out.norm()) / u0_norm
            x0_chk = model.refined_init(u0_chk)
            rel_x0 = float((x0_chk - u0_chk).norm()) / u0_norm
            print(f"[selfcheck] learned-init zero-at-init: "
                  f"‖g(LD_FBP)‖/‖LD_FBP‖={rel_g:.3e}  "
                  f"‖x0-LD_FBP‖/‖LD_FBP‖={rel_x0:.3e}  "
                  f"(init_clamp={model.init_clamp}, must be <1e-6)", flush=True)
            assert rel_g < 1e-6, (
                f"learned-init NOT zero at init (rel_g={rel_g:.3e} >= 1e-6): "
                "the zero-init head is broken (the iter-10 bug). ABORTING.")
            assert rel_x0 < 1e-6, (
                f"x_0 != LD-FBP at init (rel_x0={rel_x0:.3e} >= 1e-6): the init "
                "is being corrupted (e.g. by an upper clamp -- the iter-10 bug). "
                "ABORTING.")

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reg_params = _count_reg_params(cfg)
    init_params = (sum(p.numel() for p in model.init_refiner.parameters()
                       if p.requires_grad)
                   if model.init_refiner is not None else 0)
    n_alpha = (model.n_iter if model.per_step_alpha else 1) if model.log_alpha is not None else 0
    n_beta = 1 if model.beta_raw is not None else 0
    print(f"[solver] ParamEfficient iter-23: reg_type={cfg['reg_type']!r}  "
          f"n_iter={cfg['n_iter']} (weight-TIED reg)  "
          f"train_view_subset(r)={cfg.get('train_view_subset', 1)} "
          f"random_offset={cfg.get('train_view_random_offset', True)} "
          f"two_phase={bool(cfg.get('two_phase', False))}  "
          f"learned_init={model.use_learned_init} (init={init_params}p)  "
          f"per_step_alpha={model.per_step_alpha} momentum={model.use_momentum}  "
          f"epochs={cfg['epochs']} cosine_lr={cfg.get('cosine_lr', False)} "
          f"cosine_t_max={cfg.get('cosine_t_max')} "
          f"peak_lr={cfg.get('lr')} warmup_frac={cfg.get('warmup_frac', 0.0)} "
          f"weight_decay={cfg.get('weight_decay', 0.0)} "
          f"opt={'AdamW' if float(cfg.get('weight_decay', 0.0)) > 0 else 'Adam'}  "
          f"trainable params={params_total} "
          f"(reg={reg_params} + init={init_params} + alpha={n_alpha} + beta={n_beta})  "
          f"= {params_total/1e6:.6f} M  vs 233k ITNet", flush=True)

    # ------------------------------------------------------------------
    # iter-23 ORDERED-SUBSETS DC TRAINING setup. Build the reduced-angle
    # projectors (per ps x offset) + their power-iter ‖R_SᵀR_S‖ ONCE, so the
    # training loop only does cheap dict look-ups + the strided-sino slice per
    # step. This is TRAINING-ONLY; INFERENCE uses the full per-sample projectors.
    # r=1 (or non-Mayo / dc_norm off) disables it -> full-view iter-7 training.
    # ------------------------------------------------------------------
    r_train = max(1, int(cfg.get("train_view_subset", 1)))
    use_subset = (r_train > 1 and per_ps and bool(cfg.get("dc_norm", True))
                  and int(cfg["n_angles"]) % r_train == 0)
    subset_projs = subset_norms = None
    S_views = int(cfg["n_angles"])
    rand_offset = bool(cfg.get("train_view_random_offset", True))
    if r_train > 1 and not use_subset:
        print(f"[iter-23] ordered-subsets DISABLED (r={r_train}, per_ps={per_ps}, "
              f"dc_norm={cfg.get('dc_norm', True)}, n_angles%r="
              f"{int(cfg['n_angles']) % r_train}) -> full-view training", flush=True)
    if use_subset:
        subset_projs, subset_norms, S_views = build_subset_projectors(
            _trk, r_train, int(cfg["n_angles"]), int(cfg["n_det"]),
            int(cfg["image_size"]), device)
        # VERIFY full-view inference: the model's build-time projector + the
        # per-sample eval projectors carry the FULL n_angles; only the training
        # subset projectors are reduced. Assert the eval projectors are full-view.
        _eval_pr = _projs[float(_vrk[0])]
        _eval_A = int(_eval_pr.geom.n_angles)
        print(f"[iter-23] full-view INFERENCE check: eval projector n_angles="
              f"{_eval_A} (must == {int(cfg['n_angles'])}); training uses "
              f"S={S_views} views/step.", flush=True)
        assert _eval_A == int(cfg["n_angles"]), (
            f"eval projector n_angles={_eval_A} != full {cfg['n_angles']}: the "
            "subsampling leaked into inference. ABORTING.")

    # iter-15 TRAINER (test the TREND ENDPOINT: CONSTANT lr 5e-3, NO anneal):
    #   - cosine_lr=False, so _lr_at() returns the CONSTANT peak lr 5e-3 for
    #     EVERY step (the `if not use_cosine: return peak_lr` branch below) — the
    #     LR is HELD at the proven-stable 5e-3 for the whole run, NO anneal tail.
    #     This is the pure endpoint of the CLEAN MONOTONE TREND: iter-7 PARTIAL
    #     anneal (budget-cut @ep8, final LR ~2.5e-3) hr 0.2515 > iter-13 FULL
    #     anneal (-> 1e-5) 0.2273 > iter-14 full anneal + AdamW wd 0.2281. The
    #     low-LR cosine tail OVERFITS the 200-slice train set; iter-15 removes it.
    #   - PLAIN Adam (weight_decay 0.0). iter-14 proved wd 3e-4 did NOT recover
    #     the anneal-tail overfit; the optimiser is iter-7's exactly.
    #   - epochs=16 (iter-7's value) but the hard max_train_s=1080 wall budget-cuts
    #     the run at ~ep8 — exactly iter-7's behaviour — with every reached epoch
    #     trained at the flat 5e-3.
    #   - PER-BATCH LR off a global batch counter (harmless at constant LR; kept so
    #     the cosine path stays selectable for the post-mortem A/B).
    #   STABILITY: a constant 5e-3 is everywhere <= iter-7's early-epoch LR (its
    #   cosine STARTS at the 5e-3 peak), iter-7 ran 5e-3 MONOTONE with no
    #   oscillation, and grad_clip=1.0 caps any spike. iter-12 diverged only by
    #   RAISING the peak to 1e-2 (NOT revived here).
    #   clip_and_step() only calls opt.step(); we set the LR ourselves BEFORE
    #   each step, so the two compose cleanly (no torch scheduler needed).
    wd = float(cfg.get("weight_decay", 0.0))
    if wd > 0.0:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    peak_lr = float(cfg["lr"])
    eta_min = float(cfg.get("cosine_lr_min", 1e-5))
    use_cosine = bool(cfg.get("cosine_lr", False))
    warmup_frac = float(cfg.get("warmup_frac", 0.0))
    bs = max(1, int(cfg["batch_size"]))
    steps_per_epoch = max(1, math.ceil(cfg["train_n"] / bs))
    # iter-13: cosine PERIOD in epochs, decoupled from `epochs`. None/<=0 falls
    # back to `epochs` (the iter-7 behaviour where T_max == epochs).
    _ctm = cfg.get("cosine_t_max", None)
    cosine_t_max_ep = int(_ctm) if (_ctm is not None and int(_ctm) > 0) else int(cfg["epochs"])
    total_steps = max(1, cosine_t_max_ep * steps_per_epoch)
    warmup_steps = int(round(warmup_frac * steps_per_epoch))  # 0 => no warmup

    def _lr_at(step: int) -> float:
        """Per-batch LR: linear warmup 0->peak over warmup_steps, then cosine
        peak->eta_min over the cosine period (cosine_t_max epochs of steps), or
        constant peak if cosine_lr is off. `step` is the 0-based global batch
        index; prog is clamped to 1.0 so the LR holds at eta_min after the
        period (it cannot go below eta_min)."""
        if warmup_steps > 0 and step < warmup_steps:
            # ramp from peak/warmup_steps up to peak (never exactly 0, so the
            # very first step still makes nonzero progress).
            return peak_lr * float(step + 1) / float(warmup_steps)
        if not use_cosine:
            return peak_lr
        denom = max(1, total_steps - warmup_steps)
        prog = min(1.0, float(step - warmup_steps) / float(denom))
        return eta_min + 0.5 * (peak_lr - eta_min) * (1.0 + math.cos(math.pi * prog))

    # ------------------------------------------------------------------
    # iter-21 TWO-PHASE TRAINER. train_start clocks the WHOLE training run so
    # the Phase-B wall check (time.time() - train_start > max_train_s) caps the
    # TOTAL train time at max_train_s (1080s) across BOTH phases. Phase A
    # additionally self-caps at pretrain_frac * max_train_s; Phase B then gets
    # the remaining budget. Phase A is the SAME optimiser + LR schedule as
    # Phase B (peak lr 5e-3, partial cosine) but its objective is the cheap
    # projection-free denoiser D(LD-FBP)->truth.
    # ------------------------------------------------------------------
    max_train_s = float(cfg.get("max_train_s", 1800))
    two_phase = bool(cfg.get("two_phase", False))
    train_start = time.time()
    pretrain_time = 0.0
    pretrain_epochs_done = 0
    pretrain_last_loss = None
    pretrain_first_loss = None   # iter-22: defined even when two_phase=False (result.json)
    pretrain_rel_drop = None     # iter-22: Phase-A ep1->last relative decrease
    pretrain_n_used = 0

    if two_phase:
        pretrain_frac = float(cfg.get("pretrain_frac", 0.40))
        pretrain_s = max(0.0, pretrain_frac * max_train_s)
        # Phase-A train pool: reuse the loaded train tensors when the requested
        # pretrain_train_n <= train_n; else load a dedicated larger split (same
        # train seed convention) and build its per-ps FBP init. 0 => use train_n.
        _ptn = int(cfg.get("pretrain_train_n", 0) or 0)
        pre_target_n = _ptn if _ptn > 0 else int(cfg["train_n"])
        if pre_target_n <= int(cfg["train_n"]):
            pre_ph = train_ph
            pre_u0 = train_u0
            pre_n = int(cfg["train_n"])
        else:
            # Load a larger train split through the SAME staged pipeline. The
            # loader caps n at the available staged pool, so an over-large
            # request just returns the full pool (robust to <579 staged slices).
            pre_ph2, _pclean, pre_noisy2, pre_ps2 = build_dataset(
                geom, pre_target_n, cfg["seed"],
                cfg["noise_i0"], cfg["noise_sigma_e"], device)
            pre_n = int(pre_ph2.shape[0])
            if per_ps and pre_ps2 is not None:
                # The larger pretrain pool draws from the SAME train patients
                # (seed=cfg["seed"]) -> its <=4 distinct ps values are already in
                # the _projs cache (built from train+val ps). Build any missing
                # key defensively, then per-ps FBP. NO projector swap is needed
                # in the Phase-A loop itself (denoise() uses no projection).
                from ddssl_ldct.staged_dataset import mayo_proj_cache
                pre_prk = np.round(np.asarray(pre_ps2, float), 5)
                missing = [u for u in np.unique(pre_prk) if float(u) not in _projs]
                if missing:
                    _projs.update(mayo_proj_cache(np.asarray(missing, float),
                                                  cfg["n_angles"], cfg["n_det"], device))
                with torch.no_grad():
                    pre_u0 = mayo_per_sample_fbp(_projs, pre_prk, pre_noisy2,
                                                 cfg["image_size"])
            else:
                with torch.no_grad():
                    pre_u0 = torch.clamp(proj.fbp(pre_noisy2), min=0.0)
            pre_ph = pre_ph2
        pretrain_n_used = pre_n

        # iter-22 BOOTSTRAP FIX: re-seed the FoE synthesis (RBF) weights NON-ZERO
        # at the START of Phase A so reg(x) != 0 from step 0. The model was BUILT
        # with foe_rbf_init_std=0.0 (iter-7 byte-for-byte, the END-TO-END init),
        # which makes the projection-free denoiser D(x)=clamp(x-alpha*reg(x)) the
        # IDENTITY (reg(x)=0 => D(x)=x): the supervised denoise_loss is a CONSTANT
        # (the LD-FBP-vs-truth MSE), so Phase A FROZE at ~1.9e-4 across ALL epochs
        # in iter-21 (the gradient that should lift the RBF off zero self-vanishes
        # at the zero mixture). With a small non-zero std the RBF mixture is
        # non-degenerate, D(x) actually denoises, gradients flow, and the FoE bank
        # LEARNS a real image-domain denoiser. Phase B then fine-tunes end-to-end
        # from the TRAINED (non-degenerate) reg -- so end-to-end stability rests on
        # the trained bank, not the raw init (the zero-init-for-stability concern
        # from the end-to-end-only iters does not apply once a pretrain seeds it).
        # ONLY the synthesis rbf_weights are re-seeded; the analysis filters K and
        # the single tied scalar alpha are UNTOUCHED (params stay exactly 1,921).
        pre_rbf_std = float(cfg.get("foe_rbf_init_std_pretrain", 0.0))
        if cfg["reg_type"] == "foe" and pre_rbf_std > 0.0 and hasattr(model.reg, "rbf_weights"):
            with torch.no_grad():
                g = torch.Generator(device=model.reg.rbf_weights.device)
                g.manual_seed(int(cfg["seed"]) + 7)  # decoupled from the global seed
                model.reg.rbf_weights.copy_(
                    torch.randn(model.reg.rbf_weights.shape,
                                generator=g,
                                device=model.reg.rbf_weights.device,
                                dtype=model.reg.rbf_weights.dtype) * pre_rbf_std)
            print(f"[pretrain] iter-22 bootstrap: re-seeded FoE rbf_weights "
                  f"~N(0,{pre_rbf_std}) (||rbf||={float(model.reg.rbf_weights.norm()):.4g}, "
                  f"was 0 at build) -> reg(x)!=0 at Phase-A start (was the iter-21 "
                  f"freeze cause).", flush=True)

        # Phase-A LR schedule: same per-batch warmup+cosine machinery, but its
        # OWN epoch budget and cosine period (peak lr == lr == 5e-3, eta_min ==
        # cosine_lr_min). pretrain_cosine_t_max=0 => anneal over pretrain_epochs.
        pre_epochs = max(1, int(cfg.get("pretrain_epochs", 60)))
        _pctm = cfg.get("pretrain_cosine_t_max", 0)
        pre_ctm_ep = int(_pctm) if (_pctm is not None and int(_pctm) > 0) else pre_epochs
        pre_steps_per_epoch = max(1, math.ceil(pre_n / bs))
        pre_total_steps = max(1, pre_ctm_ep * pre_steps_per_epoch)

        def _pre_lr_at(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return peak_lr * float(step + 1) / float(warmup_steps)
            if not use_cosine:
                return peak_lr
            denom = max(1, pre_total_steps - warmup_steps)
            prog = min(1.0, float(step - warmup_steps) / float(denom))
            return eta_min + 0.5 * (peak_lr - eta_min) * (1.0 + math.cos(math.pi * prog))

        print(f"[pretrain] PHASE A (projection-free FoE denoiser pretrain): "
              f"pre_n={pre_n} (requested {pre_target_n}) epochs<={pre_epochs} "
              f"budget={pretrain_s:.0f}s cosine_t_max={pre_ctm_ep} "
              f"peak_lr={peak_lr:.3g} D(x)=clamp(x-alpha*reg(x),0,{model.clip_max})",
              flush=True)
        pa_start = time.time()
        pgstep = 0
        pretrain_first_loss = None   # iter-22: epoch-1 denoise_loss, for the decrease guard
        for ep in range(pre_epochs):
            model.train()
            perm = torch.randperm(pre_n)
            running = 0.0
            n_batches = 0
            ep_lr0 = None
            for i in range(0, pre_n, bs):
                cur_lr = _pre_lr_at(pgstep)
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
                if ep_lr0 is None:
                    ep_lr0 = cur_lr
                idx = perm[i:i + bs]
                x_in = pre_u0[idx]
                truth = pre_ph[idx]
                den = model.denoise(x_in)        # NO projection in the loop
                loss = supervised_recon_loss(den, truth,
                                             lambda_neg=cfg["lambda_neg"], base="mse")
                opt.zero_grad()
                loss.backward()
                clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
                running += float(loss.detach().cpu())
                n_batches += 1
                pgstep += 1
            pretrain_last_loss = running / max(1, n_batches)
            if pretrain_first_loss is None:
                pretrain_first_loss = pretrain_last_loss
            pretrain_epochs_done = ep + 1
            print(f"[pretrain] epoch {ep+1}/{pre_epochs}  "
                  f"denoise_loss={pretrain_last_loss:.6g}  "
                  f"lr={ep_lr0:.3g}->{_pre_lr_at(pgstep - 1):.3g}  "
                  f"alpha={float(model.alpha.detach().cpu()):.4g}", flush=True)
            if time.time() - pa_start > pretrain_s:
                print(f"[pretrain] phase-A wall ({pretrain_s:.0f}s) reached at "
                      f"epoch {ep+1}", flush=True)
                break
        pretrain_time = time.time() - pa_start
        print(f"[pretrain] PHASE A done: {pretrain_epochs_done} epochs x {pre_n} "
              f"= {pretrain_epochs_done * pre_n} sample-passes in "
              f"{pretrain_time:.1f}s (vs iter-7 cold ~1.6k). PHASE B (end-to-end "
              f"K={cfg['n_iter']} DC finetune) now starts WARM.", flush=True)

        # iter-22 MANDATORY VERIFICATION (the guard iter-21 lacked): Phase-A
        # denoise_loss MUST DECREASE. A flat loss (the iter-21 freeze: stuck at
        # the 1.9e-4 identity floor across ALL epochs) means the bootstrap is
        # STILL broken (e.g. foe_rbf_init_std_pretrain back at 0, or the denoiser
        # is degenerate) -> FAIL LOUDLY here BEFORE wasting Phase B + the eval.
        # A healthy denoiser pretrain drives denoise_loss WELL BELOW the constant
        # identity floor; we require a >=1% relative drop ep1 -> last as the
        # minimal "it is actually learning" gate.
        pretrain_rel_drop = None
        if pretrain_first_loss is not None and pretrain_last_loss is not None \
                and pretrain_epochs_done >= 2:
            pretrain_rel_drop = ((pretrain_first_loss - pretrain_last_loss)
                                 / max(abs(pretrain_first_loss), 1e-12))
            print(f"[pretrain] DECREASE CHECK: epoch-1 denoise_loss="
                  f"{pretrain_first_loss:.6g}  last={pretrain_last_loss:.6g}  "
                  f"rel_drop={pretrain_rel_drop:+.3%}  (must be >= +1.00% — a flat "
                  f"loss == the iter-21 identity-floor freeze).", flush=True)
            assert pretrain_rel_drop >= 0.01, (
                f"Phase-A denoise_loss did NOT decrease (rel_drop="
                f"{pretrain_rel_drop:+.3%} < +1.00%): the denoiser pretrain is "
                f"FROZEN (ep1={pretrain_first_loss:.6g}, last="
                f"{pretrain_last_loss:.6g}). This is the iter-21 failure mode "
                f"(reg(x)=0 => D(x)=identity => constant loss). Check "
                f"foe_rbf_init_std_pretrain (>0 required). ABORTING.")
        else:
            print(f"[pretrain] DECREASE CHECK skipped: only "
                  f"{pretrain_epochs_done} Phase-A epoch(s) completed (need >=2 to "
                  f"compare). Increase pretrain_frac / pretrain_epochs.", flush=True)

    gstep = 0
    epochs_done = 0
    rng = np.random.default_rng(int(cfg["seed"]) + 23)  # iter-23: view-offset RNG
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(cfg["train_n"])
        running = 0.0
        n_batches = 0
        ep_lr0 = None
        for i in range(0, cfg["train_n"], bs):
            cur_lr = _lr_at(gstep)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            if ep_lr0 is None:
                ep_lr0 = cur_lr
            idx = perm[i:i + bs]
            u0 = train_u0[idx]
            truth = train_ph[idx]
            if use_subset:
                # iter-23 ORDERED-SUBSETS DC: pick a random angular offset, swap
                # in the reduced (ps, off) projector + its ‖R_SᵀR_S‖, and feed
                # the STRIDED sinogram g[:, :, off::r, :] (the S measured columns).
                ps_key = float(_trk[int(idx[0])])
                off = int(rng.integers(0, r_train)) if rand_offset else 0
                model.projector = subset_projs[(round(ps_key, 5), off)]
                sino = train_noisy[idx][:, :, off::r_train, :].contiguous()
                dcn = torch.tensor(subset_norms[(round(ps_key, 5), off)],
                                   device=u0.device, dtype=u0.dtype)
                pred = model(u0, sino, dc_norm=dcn)
            else:
                if per_ps:
                    model.projector = _projs[float(_trk[int(idx[0])])]
                sino = train_noisy[idx]
                pred = model(u0, sino)
            loss = supervised_recon_loss(pred, truth,
                                         lambda_neg=cfg["lambda_neg"], base="mse")
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
            running += float(loss.detach().cpu())
            n_batches += 1
            gstep += 1
        epochs_done = ep + 1
        avg_loss = running / max(1, n_batches)
        print(f"[train] epoch {ep+1}/{cfg['epochs']}  loss={avg_loss:.6g}  "
              f"lr={ep_lr0:.3g}->{_lr_at(gstep - 1):.3g}  "
              f"alpha={float(model.alpha.detach().cpu()):.4g}"
              f"{f'  [S={S_views} views]' if use_subset else ''}",
              flush=True)
        if time.time() - train_start > cfg.get("max_train_s", 1800):
            print(f"[train] wall ({cfg.get('max_train_s', 1800)}s) reached at epoch {ep+1}",
                  flush=True)
            break
    train_time = time.time() - train_start
    print(f"[train] iter-23 DONE: epochs_done={epochs_done} (vs iter-7 ~8) "
          f"n_train_views={S_views if use_subset else int(cfg['n_angles'])} "
          f"(eval=full {int(cfg['n_angles'])})  train_time={train_time:.1f}s", flush=True)

    # FULL-VIEW INFERENCE: the model's normal forward with the FULL per-sample
    # projector (_projs[vrk[i]], n_angles=2304), the FULL val sinogram, and the
    # build-time FULL dc_norm (dc_norm=None -> self.dc_norm). ONLY training
    # subsampled views; the scored recon is full-view (no train/test mismatch).
    model.eval()
    preds = []
    with torch.no_grad():
        chunk = 1 if per_ps else max(1, bs)
        for i in range(0, val_u0.shape[0], chunk):
            if per_ps:
                model.projector = _projs[float(_vrk[i])]
            preds.append(model(val_u0[i:i + chunk], val_noisy[i:i + chunk]))
    pred = torch.cat(preds, dim=0)

    # baseline = the LD-FBP starting point (the headroom anchor).
    with torch.no_grad():
        if per_ps:
            val_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    val_fbp = val_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=val_fbp,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    val_score = val_ssim

    result = {
        "val_score": val_score, "headroom": headroom,
        "val_ssim": val_ssim, "val_psnr": val_psnr, "val_rmse": val_rmse,
        "val_ssim_std": metrics["val_ssim_std"],
        "val_psnr_std": metrics["val_psnr_std"],
        "val_rmse_std": metrics["val_rmse_std"],
        "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_total / 1e6,
        "reg_type": cfg["reg_type"],
        "reg_params": reg_params,
        # iter-24 STEERABLE-FoE telemetry (rotation-equivariant analysis bank)
        "steer_n_rad": int(cfg.get("steer_n_rad", 7)),
        "steer_max_order": int(cfg.get("steer_max_order", 3)),
        "steer_coeff_init_std": float(cfg.get("steer_coeff_init_std", 0.05)),
        "steer_n_basis": (int(model.reg.n_basis)
                          if hasattr(model.reg, "n_basis") else None),
        "learned_init": bool(model.use_learned_init),
        "init_clamp": bool(model.init_clamp),
        "init_params": init_params,
        "init_channels": int(cfg.get("init_channels", 16)),
        "init_layers": int(cfg.get("init_layers", 1)),
        "n_iter": cfg["n_iter"],
        "epochs": cfg["epochs"],
        "cosine_lr": bool(cfg.get("cosine_lr", False)),
        "cosine_t_max": cosine_t_max_ep,
        "lr": float(cfg.get("lr", 0.0)),
        "warmup_frac": float(cfg.get("warmup_frac", 0.0)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "per_step_alpha": bool(model.per_step_alpha),
        "momentum": bool(model.use_momentum),
        "alpha_learned": float(model.alpha.detach().cpu()),  # mean over steps
        "alpha_per_step": ([float(model._alpha_k(k).detach().cpu())
                            for k in range(model.n_iter)]
                           if model.log_alpha is not None else None),
        "beta_learned": (float(model.beta.detach().cpu())
                         if model.beta_raw is not None else None),
        "train_n": cfg["train_n"], "val_n": cfg["val_n"],
        "train_time_s": train_time, "config": cfg,
        # iter-21 TWO-PHASE trainer telemetry
        "two_phase": two_phase,
        "pretrain_frac": float(cfg.get("pretrain_frac", 0.0)) if two_phase else 0.0,
        "pretrain_time_s": pretrain_time,
        "pretrain_epochs_done": pretrain_epochs_done,
        "pretrain_n_used": pretrain_n_used,
        "pretrain_last_loss": pretrain_last_loss,
        # iter-22 bootstrap-fix telemetry: the Phase-A decrease guard.
        "pretrain_first_loss": pretrain_first_loss,
        "pretrain_rel_drop": pretrain_rel_drop,
        "foe_rbf_init_std_pretrain": float(cfg.get("foe_rbf_init_std_pretrain", 0.0)),
        "finetune_time_s": max(0.0, train_time - pretrain_time),
        # iter-23 ORDERED-SUBSETS DC training telemetry
        "train_view_subset": r_train,
        "train_view_random_offset": bool(cfg.get("train_view_random_offset", True)),
        "ordered_subsets_active": bool(use_subset),
        "n_train_views": int(S_views if use_subset else int(cfg["n_angles"])),
        "n_eval_views": int(cfg["n_angles"]),
        "epochs_done": int(epochs_done),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    _beta_s = f"{result['beta_learned']:.3g}" if result['beta_learned'] is not None else "off"
    print(f"[solver] ParamEfficient: val_score={val_score:.4f} "
          f"headroom={headroom:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"params={params_total}  alpha_mean={result['alpha_learned']:.4g}  "
          f"beta={_beta_s}  time={train_time:.1f}s  "
          f"[iter-23 OS: r={r_train} active={use_subset} "
          f"n_train_views={result['n_train_views']} eval={int(cfg['n_angles'])} "
          f"epochs_done={epochs_done} vs iter-7~8]  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label=f"ParamEff[{cfg['reg_type']}]",
            headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
