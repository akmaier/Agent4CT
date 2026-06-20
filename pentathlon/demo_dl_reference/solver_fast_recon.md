# Fast few-step diffusion-prior reconstruction (`flow_recon` / `wdm_recon`)

A NEW solver family: a **rectified-flow / flow-matching** prior trained on
ground-truth CT images, used as a few-step generative prior for CT
reconstruction from a measured (sparse-view / low-dose) sinogram. It is the
"fast" counterpart of the existing `diffusion_recon_dcstep_*` pair: where the
DDPM prior needs 50–1000 reverse steps, a flow-matching prior reconstructs in
≈6–8 Euler steps.

Two files implement the whole family (the "4 solvers" are 4 checkpoints + 4
registry entries):

| File | Role |
|---|---|
| [`solver_fast_diffusion.py`](solver_fast_diffusion.py) | **Trainer** — flow matching on Mayo (or any staged) truth. |
| [`solver_fast_recon.py`](solver_fast_recon.py) | **Recon** — few-step DC-guided sampler. |

Both mirror the structure + contracts of `solver_ddpm.py` (trainer) and
`solver_diffusion_recon.py` (recon) so they drop straight into the calibrated
metric, the dataset dispatch, and the dashboard/registry pipeline.

---

## 1. Method

### 1.1 Flow matching (rectified flow)

We train a velocity field `v_θ(x_t, t)` to transport the standard normal at
`t=1` to the data distribution at `t=0` along the straight path

```
x_t = (1 - t)·x0 + t·x1 ,   x1 ~ N(0, I) ,   t ~ U(0,1)
```

with target velocity `v = x1 - x0` (constant along the straight line). The
loss is the simple velocity MSE

```
L = E_{x0, x1, t} || v_θ(x_t, t) - (x1 - x0) ||²
```

This is the rectified-flow / conditional-flow-matching objective (Liu et al.
2022, arXiv:2209.03003; Lipman et al. 2022, arXiv:2210.02747). Continuous time
`t ∈ [0,1]` is fed to the same sinusoidal `TimeEmb` the DDPM uses, scaled by
`×1000` so it occupies the numeric band that embedding was built for.

Because the path is straight, generation needs only a few Euler steps from
`t=1` to `t=0`:

```
v = v_θ(x, t);   x ← x − dt·v;   t ← t − dt
```

### 1.2 Pixel vs wavelet (WDM) variants

* **`flow_recon` (pixel)** — flow matching directly in image space. The
  velocity U-Net has `in_ch = out_ch = 1`.
* **`wdm_recon` (wavelet)** — flow matching in the single-level 2D **Haar**
  DWT domain (the WDM idea, Friedrich et al. 2024, arXiv:2402.19043, made
  few-step via flow matching). The truth image is transformed to its 4 Haar
  subbands `[LL, LH, HL, HH]` (each `H/2 × W/2`); the velocity U-Net has
  `in_ch = out_ch = 4`. At recon time the differentiable inverse DWT maps the
  clean estimate back to the image so the data-consistency gradient can flow
  through it.

The Haar transform is implemented inline (orthonormal, exact inverse) in
`solver_fast_diffusion.py` as `haar_dwt` / `haar_idwt`.

### 1.3 Few-step DC-guided reconstruction (`solver_fast_recon.py`)

Reconstruction integrates the flow ODE backwards from `init_t` to 0 with a few
Euler steps. At each step:

1. `v = v_θ(x_t, t)`; clean estimate (model domain) `x0_hat_dom = x_t − t·v`
   (from `x_t = (1−t)x0 + t·x1` with `v = x1 − x0` ⇒ `x0 = x_t − t·v`).
2. Map to image: pixel → identity; wavelet → `haar_idwt`. Denorm to μ:
   `x0_img = clamp(x0_hat_img, 0, 1)·out_scale`.
3. **DPS guidance** (Chung et al. 2023, arXiv:2209.14687): residual
   `r = A·x0_img − y`, `sse = Σ r²`, `grad = ∂ sse / ∂ x_t`, adaptive step
   `step = (η / ‖r‖)·grad` (Chung eq. 12 — the adaptive scale is what lets a
   single `η` work across noise levels). Differentiable through IDWT in
   wavelet mode.
4. Euler update toward `t=0`: `x_t ← x_t − dt·v − step`.
5. **Optional hard CG DC-step** (`fd_recon_dc=1`): a *real* conjugate-gradient
   solve of `min_x ‖A x − y‖²` (normal equations `AᵀA x = Aᵀ y`, `Aᵀ` =
   back-projection) warm-started at `x0_img`, blended with `relax`. The
   projected image is then **deterministically re-embedded** at the next time
   level reusing the **same noise direction** recovered from the current state
   (`noise = (x_t − (1−t)·x0_hat_dom)/t`) — *no fresh randn, no stale eps*.
   This is the corrected version of the buggy `dc_step_cg` /
   re-injection in `solver_diffusion_recon.py`.

At `t→0` the state `x_t` *is* the clean estimate; it is inverse-transformed
(wavelet), denormalised, and `clamp_min(0)`'d.

---

## 2. Data regimes (constrained / unconstrained)

Mirrors `ddpm_mode` exactly. **Both load from the `train` split only** — never
val/test — so there is no leakage into the inference-time evaluation.

* **`unconstrained`** — `fd_n_train` train slices (the "best the prior can be"
  upper bound).
* **`constrained`** — only the `fd_n_train_constrained` train slices the other
  dl_reference solvers see (apples-to-apples).

Cross with the two domains → the four advertised solvers:

| Solver | domain | mode | checkpoint (suggested) |
|---|---|---|---|
| `flow_recon` unconstrained | pixel   | unconstrained | `fast_diffusion_pixel_unconstrained.pt` |
| `flow_recon` constrained   | pixel   | constrained   | `fast_diffusion_pixel_constrained.pt`   |
| `wdm_recon` unconstrained  | wavelet | unconstrained | `fast_diffusion_wavelet_unconstrained.pt` |
| `wdm_recon` constrained    | wavelet | constrained   | `fast_diffusion_wavelet_constrained.pt`   |

---

## 3. Config knobs + defaults

### Trainer (`solver_fast_diffusion.py`, env `FAST_DIFFUSION_CONFIG_PATH`)

| Key | Default | Meaning |
|---|---|---|
| `fd_domain` | `"pixel"` | `"pixel"` or `"wavelet"`. |
| `fd_mode` | `"unconstrained"` | `"constrained"` or `"unconstrained"`. |
| `fd_n_train` | `3000` | train slices when unconstrained. |
| `fd_n_train_constrained` | `200` | train slices when constrained. |
| `fd_n_val` | `100` | held-out flow-loss val slices. |
| `fd_out_scale` | `0.05` | μ → [0,1] normaliser (auto = `display_max`). |
| `fd_out_scale_auto` | `True` | auto-set `fd_out_scale` from the dataset. |
| `fd_ch` | `32` | base channels of `FlowUNet`. |
| `fd_epochs` | `30` | training epochs. |
| `fd_batch` | `8` | batch size. |
| `fd_lr` | `2e-4` | Adam learning rate. |
| `fd_weight_decay` | `0.0` | Adam weight decay. |
| `fd_ema_decay` | `0.999` | EMA decay (EMA weights are the shipped ckpt). |
| `fd_gen_steps` | `6` | Euler steps for the training-figure samples. |
| `fd_train_wall_s` | `3600` | wall-clock cap (s). |
| `fd_ckpt` | `…/fast_diffusion_search.pt` | checkpoint output path. |

### Recon (`solver_fast_recon.py`, env `FAST_RECON_CONFIG_PATH`)

| Key | Default | Meaning |
|---|---|---|
| `val_n` | `20` | val scenes to reconstruct (Mayo: set 214 for the full L277 split). |
| `fd_recon_ckpt` | `…/fast_diffusion_unconstrained.pt` | prior ckpt (env `FAST_RECON_CKPT` overrides). |
| `fd_recon_steps` | `8` | Euler steps from `init_t` to 0. |
| `fd_recon_init` | `"fbp"` | `"fbp"` (warm-start) or `"noise"`. |
| `fd_recon_init_t` | `0.7` | for fbp init, embed FBP at this flow time. |
| `fd_recon_eta` | `1.0` | DPS guidance scale (adaptive). |
| `fd_recon_dc` | `0` | `0` off, `1` hard CG DC-step each Euler step. |
| `fd_recon_dc_n_cg` | `5` | inner CG iterations. |
| `fd_recon_dc_relax` | `0.5` | blend of CG-projected image (0..1). |

The **domain is NOT a recon config knob** — it is auto-detected from the loaded
checkpoint (`fd_domain`), so the recon always matches its prior.

---

## 4. Checkpoint contract

`solver_fast_diffusion.py` writes (and `solver_fast_recon.py` reads):

```python
{
  "model_state":      <EMA weights>,        # NOT the raw training weights
  "fd_domain":        "pixel" | "wavelet",
  "fd_ch":            int,                   # FlowUNet base channels
  "fd_out_scale":     float,                 # μ -> [0,1] normaliser
  "fd_mode":          "constrained" | "unconstrained",
  "n_train":          int,
  "train_seed":       int,                   # phantoms-path seed offset
  "final_train_loss": float,
  "final_val_loss":   float,                 # final flow-matching val MSE
  "best_val_loss":    float,
  "history":          [{epoch, train_loss, val_loss, elapsed_s}, ...],
  "config":           {...},                 # full resolved cfg
}
```

The **EMA weights** are the shipped `model_state` — flow-matching samples are
noticeably more stable from EMA weights than from the last raw iterate; the
audit flagged the absence of EMA as a gap, so it is fixed here.

---

## 5. How it plugs into the calibrated metric

Identical to `solver_diffusion_recon.py`:

* The recon `clamp_min(0)`'s every prediction, then calls
  `ddssl_ldct.metrics.evaluate_calibrated(pred, truth, baseline=val_fbp, …)`.
  That applies the two-point intensity calibration + the dataset's FOV mask
  (Mayo: 321 px detector-geometry FOV; others: 256 px Sidky inscribed circle),
  and computes PSNR / SSIM / RMSE / **headroom** = `max(0, 1 − rmse/baseline_rmse)`.
* For Mayo the val split is a single patient (L277) reconstructed at its native
  pixel-spacing via the per-sample-ps probe (copied verbatim from
  `solver_diffusion_recon.py`).
* A standard `make_4panel_comparison` figure (`solver_label="FastDiff"`) is
  written to `comparison.png`, satisfying the "every reported result needs an
  image" rule.

The **trainer**'s `result.json` reports a *training surrogate* (the flow-loss),
not a reconstruction metric: `val_score = −val_flow_loss`,
`headroom = max(0, 1 − val_flow_loss/1.0)`. The reconstruction headroom comes
from the recon solver.

---

## 6. How to train + reconstruct

Train (cluster) — one variant per `solver_fast_diffusion.py` call, selected by
env (see [`cluster/slurm/train_fast_diffusion_mayo.sbatch`](../../cluster/slurm/train_fast_diffusion_mayo.sbatch)):

```bash
FD_DOMAIN=pixel   FD_MODE=unconstrained \
FD_CKPT=$CKPT/fast_diffusion_pixel_unconstrained.pt \
  sbatch cluster/slurm/train_fast_diffusion_mayo.sbatch
# ...and the other 3 (domain × mode) combinations.
```

Reconstruct (per checkpoint):

```bash
AGENT4CT_DATASET=mayo_ldct_2d \
FAST_RECON_CKPT=$CKPT/fast_diffusion_pixel_unconstrained.pt \
FAST_RECON_CONFIG_PATH=/tmp/recon_cfg.json \
  python pentathlon/demo_dl_reference/solver_fast_recon.py <out_dir>
# recon_cfg.json e.g. {"val_n":214,"fd_recon_steps":8,"fd_recon_init":"fbp",
#                      "fd_recon_init_t":0.7,"fd_recon_eta":1.0,"fd_recon_dc":1}
```

---

## 7. Expected behaviour (honest note)

On **dense-view Mayo** (2304 views, the current real dataset) the inverse
problem is already well-posed and the FBP baseline is strong (HD-FBP headroom
0). A generative prior cannot add much structure the data does not already
constrain, so the recon headroom on dense Mayo is **regime-bounded** — expect
it to be *competitive with, not dramatically above*, the supervised denoisers,
much like the existing `diffusion_recon_dcstep_*` on Mayo. The flow-matching
prior's advantage over the DDPM prior here is **speed** (≈8 steps vs ≈50), not
a headroom jump.

The **real payoff is sparse-view / few-view** reconstruction, where the data
under-constrains the image and a learned prior fills the null space. The
`wdm_recon` (wavelet) variant is expected to help most on high-frequency
structure (the LH/HL/HH subbands), again primarily in the sparse regime. None
of this is demonstrated here — it is the design hypothesis; the numbers come
from running the recon and reading the calibrated headroom + the figure.

---

## References

* Liu X. *et al.* "Flow Straight and Fast: Learning to Generate and Transfer
  Data with Rectified Flow." 2022. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
* Lipman Y. *et al.* "Flow Matching for Generative Modeling." 2022.
  [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
* Friedrich P. *et al.* "WDM: 3D Wavelet Diffusion Models for High-Resolution
  Medical Image Synthesis." 2024. [arXiv:2402.19043](https://arxiv.org/abs/2402.19043)
* Chung H. *et al.* "Diffusion Posterior Sampling for General Noisy Inverse
  Problems." ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
