# `solver_ram.py` — RAM (Terris 2025) zero-shot frozen-prior reconstruction

Companion design doc. RAM is the **only frozen-prior solver** in the
pentathlon — it loads a pretrained Restoration of Anything Model
(Terris et al. 2025) checkpoint and runs CT reconstruction as
zero-shot inference, with no per-dataset training.

The RAM prior is a 35.6 M-param generic image-restoration foundation
model. The solver wraps it in a PyroNN CT data-consistency loop:

1. Initialize x_0 = FBP-of-noisy.
2. Run RAM on x_0 to produce a denoised image x_RAM.
3. Optionally blend with FBP (`ram_post_fbp_blend`).
4. Optionally project + back-project to enforce sino consistency.

No backward pass through RAM by default (`ram_finetune=False`).

## Cross-dataset record

| Dataset | hr | Source | Notes |
|---|---:|---|---|
| `demo_dl` | **0.4648** | TPE `demo-intensity-calibrated-tpe-ram-zeroshot-search-20260521-01` (σ=0.075, blend=0.42, factor=0.42, train_n=0) | rank 6 on demo-DL. Strong zero-shot result — RAM's natural-image prior transfers cleanly to ellipse phantoms. |
| `breast_ct` | **0.3077** | TPE `breast-ct-calibrated-tpe-ram-zeroshot-search-20260522-01` (σ=8.1e-3, factor=0.40, blend=0.50, multiscale=False, train_n=0) | rank 11 on breast-CT. RAM's prior is too generic for fibroglandular detail — beaten by tuned dual-domain solvers. |
| `mayo_ldct` | **0** | Step-2 iter-3 hr=0 (3 consecutive hr=0). SSIM crept 0.40→0.48 across iters but PSNR ceiling 12.45 < baseline 12.59 | **STOP** — `ram.pth.tar` (natural-image prior) cannot bridge to Mayo's μ-range. Even with `ram_factor` and `ram_post_fbp_blend` sweeps, RAM outputs sit below baseline FBP. |

## 2026-06-03 — Mayo verdict

The Mayo Step-2 agentic loop dispatched RAM zero-shot for 3 iters:
- iter-1: default σ=5e-3, factor=1.0, blend=0.0 → hr=0 SSIM 0.40
- iter-2: σ=1e-2, factor=0.5, blend=0.0 → hr=0 SSIM 0.45
- iter-3: σ=2e-2, factor=0.4, blend=0.3 → hr=0 SSIM 0.48

SSIM climbed across iters (RAM is doing real work) but **PSNR
plateaued at 12.45 dB** — below the FBP baseline of 12.59 dB. The
RAM prior denoises in display-intensity space; Mayo's μ-attenuation
distribution (0.0-0.0306 mm⁻¹) doesn't match what the natural-image
prior expects.

**Verdict**: STOP. RAM's frozen-natural-image prior is a wrong fit
for Mayo's HU-distributed scans without finetuning. Future work
could try `ram_finetune=True` on Mayo train_n=50, but that's a
~36 M-param finetune which exceeds the Q6000 24-GB cap.

## CONFIG defaults

```python
CONFIG = {
    "val_n":                 20,
    "ram_ckpt_path":         "/cluster/maier/Agent4CT/checkpoints/ram.pth.tar",
    "ram_sigma":             5e-3,
    "ram_input_norm":        "display_max",    # "display_max" | "fbp_max" | "none"
    "ram_clamp_output":      True,
    "ram_finetune":          False,            # zero-shot; finetune hurt by -0.17 on breast-CT
    "ram_finetune_epochs":   0,
    "ram_finetune_lr":       1e-4,
    "ram_factor":            1.0,              # 0 = skip prox_l2 realign
    "ram_post_fbp_blend":    0.0,              # 0 = pure RAM, 1 = pure FBP
    "ram_disable_multiscale": False,
    "ram_disable_cudnn":     False,
    "ram_use_deepinv_tomo":  False,
}
```

## Hints for the next autoresearch agent

- **`ram_finetune=False` is the right default** on every dataset
  tested. The 2026-05-21 TPE search ran `ram_finetune=True` on
  breast-CT — it hit hr=0.13 vs zero-shot's 0.30. The 36 M-param
  finetune overfits the small CT train set.
- **`ram_input_norm` matters**. On breast-CT, "adjoint_max" was the
  only norm that worked; "display_max" / "fbp_max" / "none" all
  yielded hr=0 in early sweeps. Lock `adjoint_max` when porting RAM
  to a new dataset.
- **`ram_post_fbp_blend` is a soft "trust" knob**: blend=0.0 = pure
  RAM (more denoising, more risk of detail loss); blend=1.0 = pure
  FBP (no denoising). Sweet spot on breast-CT is 0.42-0.50.
- **Don't run RAM on real-helical data** like Mayo without
  finetuning. The μ-distribution mismatch breaks the prior. If you
  must, plan a stage with finetune_epochs=5-10 at lr=1e-4 and
  expect ~2 GPU-hours per trial.
- **No backward-pass during inference** is the whole point. If you
  catch yourself wanting to train RAM end-to-end, you're outside
  the zero-shot use case and should switch to a learned-prior
  solver like DD-UNet sup or LPD.

## RAM checkpoint provenance

`/cluster/maier/Agent4CT/checkpoints/ram.pth.tar` is the Terris et al.
2025 release. Architecture details (multi-scale denoising network with
batch-norm, optional `multiscale=True`) are documented in the original
paper. The checkpoint is loaded once per process and held in GPU
memory for the entire TPE/agentic run.
