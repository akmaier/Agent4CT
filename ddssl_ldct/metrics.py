from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FOV mask
# ---------------------------------------------------------------------------
#
# Every released val_fbp128 in the Sidky 2021 DL-Sparse-View challenge is
# masked outside the inscribed circle of the 512x512 grid (= 21.46 % of the
# pixels exactly 0, matching `1 − π/4` to four decimals). For a fair score
# our recon must be masked identically; otherwise corner FBP-undershoot
# garbage corrupts SSIM/PSNR/RMSE while leaving Sidky's clean recon
# unaffected. This helper is used automatically by `evaluate_calibrated`.


@lru_cache(maxsize=32)
def _fov_mask_cached(size: int, radius_pix: float,
                      device_str: str, dtype_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r2 = xx * xx + yy * yy
    return (r2 <= radius_pix * radius_pix).to(dtype)


def fov_mask(size: int, *, radius_pix: float | None = None,
              device: torch.device | str | None = None,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Circular FOV mask, shape `(size, size)`, float in {0, 1}.

    Defaults: `radius_pix = size / 2` (= largest circle inscribed in the
    square image grid — the Sidky 2021 convention; see
    `challenges/dl_sparse_view/README.md`). Override `radius_pix` for
    larger or smaller circles. Cached so repeated calls during a search
    don't reallocate.
    """
    if radius_pix is None:
        radius_pix = float(size) / 2.0
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)
    dtype_str = str(dtype).rsplit(".", 1)[-1]
    return _fov_mask_cached(int(size), float(radius_pix),
                              str(device), dtype_str)


# ===========================================================================
# Intensity calibration
# ===========================================================================
#
# Reconstructions from different solvers come out at different absolute
# intensity scales — a learned U-Net might output values centred near zero,
# a diffusion sampler near display_max/2, classical FBP near the true μ
# range, etc. PSNR/SSIM/RMSE on such uncalibrated images favour solvers
# whose internal bias happens to land near the truth scale, even when
# their *structure* is worse than a competitor's.
#
# To make the dl_sparse_view leaderboard comparable, we apply a single
# linear two-point calibration to every reconstruction before scoring:
#
#     pred_cal = a · (pred - bg_pred)       with        a = fg_truth / (fg_pred - bg_pred)
#     pred_cal = clamp(pred_cal, 0, display_max)
#
# where `fg_pred` / `bg_pred` are the means of `pred` inside / outside a
# foreground mask defined from the *ground truth*, and `fg_truth` is the
# truth's mean inside the same mask. This forces:
#   - the background to land at exactly 0 (matching the air outside the body);
#   - the foreground tissue mean to match the truth's tissue mean;
#   - negative artefacts to be clipped away.
#
# It is the standard pre-scoring step in CT recon benchmarks (cf. Wagner
# et al. 2022, Hammernik et al. 2018). Without it our metrics drift run-to-
# run with the solver's internal bias and the leaderboard is meaningless.
# A simple ReLU at the solver boundary is *not* sufficient — it fixes the
# negative tail but leaves the bias and span uncalibrated.


import os as _os_metrics  # env-keyed production defaults (Mayo hard-wiring)

_MAYO_CAL_LOGGED = False


def _resolve_bg_target(bg_target):
    """Resolve an unspecified ``bg_target`` from the environment so the Mayo
    production calibration is **hard-wired**: when ``AGENT4CT_DATASET=mayo_ldct_2d``
    (set by every Mayo dispatch and propagated to each solver subprocess), an
    omitted ``bg_target`` defaults to ``"truth"`` — Mayo truth background sits at
    ~+0.0005 μ (≠ 0), and the legacy ``None``/bg→0 form costs up to +0.042 SSIM
    (findings.md 2026-06-13). ``AGENT4CT_BG_TARGET`` overrides explicitly. Other
    datasets (breast_ct / demo_dl) are untouched — they stay ``None`` (legacy)."""
    global _MAYO_CAL_LOGGED
    if bg_target is not None:
        return bg_target
    env = _os_metrics.environ.get("AGENT4CT_BG_TARGET")
    if env is None and _os_metrics.environ.get("AGENT4CT_DATASET") == "mayo_ldct_2d":
        env = "truth"
    if env is None:
        return None
    resolved = env if env == "truth" else float(env)
    if not _MAYO_CAL_LOGGED:
        print(f"[metrics] HARD-WIRED Mayo calibration: bg_target={resolved!r} "
              f"(AGENT4CT_DATASET={_os_metrics.environ.get('AGENT4CT_DATASET')!r})",
              flush=True)
        _MAYO_CAL_LOGGED = True
    return resolved


def intensity_calibrate(pred: torch.Tensor, truth: torch.Tensor, *,
                        fg_threshold: float | None = None,
                        display_max: float | None = None,
                        bg_target: "float | str | None" = None) -> torch.Tensor:
    """Linear two-point calibration of `pred` against `truth`.

    Pixels where `truth > fg_threshold` define the foreground mask; the
    complement is background. We then:
      1. compute `bg_pred = pred[bg].mean()`
      2. compute `fg_pred = pred[fg].mean()`, `fg_truth = truth[fg].mean()`
      3. solve for the affine that maps `fg_pred → fg_truth` and
         `bg_pred → bg_dst` (see `bg_target`).
      4. clip below 0; if `display_max` is given, also clip above it.

    `bg_target` — where the background level is mapped:
      * ``None`` (default): map `bg_pred → 0`, i.e. `a = fg_truth/(fg_pred-bg_pred)`,
        `pred_cal = a·(pred - bg_pred)`. This is the historical one-point-anchored
        form and **assumes truth's background is 0**. Unchanged for all existing
        callers (breast_ct / demo_dl, whose background μ is different/unknown).
      * ``"truth"``: map `bg_pred → bg_truth = truth[bg].mean()` — the proper
        two-point affine `pred_cal = bg_truth + a·(pred - bg_pred)` with
        `a = (fg_truth - bg_truth)/(fg_pred - bg_pred)`. Use for Mayo, where the
        truth background sits at ~+0.0005 μ (air/low tissue ≠ 0); the ``None``
        form leaves the recon ~0.0005 μ too dark and costs up to +0.042 SSIM
        (verified across all 10 Wagner patients, 2026-06-13). See findings.md.
      * a float: map `bg_pred → that explicit value`.

    If either mask is empty the input is returned unchanged. Operates
    pixel-wise; works on `(H,W)`, `(1,H,W)`, `(B,1,H,W)`, etc.
    """
    bg_target = _resolve_bg_target(bg_target)
    if fg_threshold is None:
        tmin = float(truth.min())
        tmax = float(truth.max())
        fg_threshold = tmin + 0.05 * (tmax - tmin)
    fg_mask = truth > fg_threshold
    bg_mask = ~fg_mask
    if not bool(fg_mask.any()) or not bool(bg_mask.any()):
        # Degenerate (uniform truth) — nothing to calibrate.
        out = pred.clamp_min(0.0)
        if display_max is not None:
            out = out.clamp(0.0, display_max)
        return out
    bg_pred  = pred[bg_mask].mean()
    fg_pred  = pred[fg_mask].mean()
    fg_truth = truth[fg_mask].mean()
    span = (fg_pred - bg_pred).clamp_min(torch.tensor(1e-9, device=pred.device,
                                                       dtype=pred.dtype))
    if bg_target is None:
        # legacy: bg_pred -> 0 (assumes truth background == 0)
        a = fg_truth / span
        pred_cal = a * (pred - bg_pred)
    else:
        if isinstance(bg_target, str):
            if bg_target != "truth":
                raise ValueError(f"bg_target str must be 'truth', got {bg_target!r}")
            bg_dst = truth[bg_mask].mean()
        else:
            bg_dst = torch.as_tensor(float(bg_target), device=pred.device,
                                      dtype=pred.dtype)
        # proper two-point affine: bg_pred -> bg_dst, fg_pred -> fg_truth
        a = (fg_truth - bg_dst) / span
        pred_cal = bg_dst + a * (pred - bg_pred)
    pred_cal = pred_cal.clamp_min(0.0)
    if display_max is not None:
        pred_cal = pred_cal.clamp(0.0, float(display_max))
    return pred_cal


def evaluate_calibrated(pred: torch.Tensor, truth: torch.Tensor,
                         baseline: torch.Tensor | None = None,
                         *, display_min: float, display_max: float,
                         fg_threshold: float | None = None,
                         fov: torch.Tensor | bool = True,
                         bg_target: "float | str | None" = None) -> dict:
    """Full standard evaluation: calibrate `pred` (and optionally `baseline`)
    against `truth`, apply a circular FOV mask, then compute
    PSNR/SSIM/RMSE/headroom over the masked region.

    FOV mask: ``fov=True`` (default) applies the inscribed-circle mask
    (radius = `truth.shape[-1]/2` px) — matching Sidky 2021's released
    val_fbp128 convention. Pass a custom mask tensor to override, or
    ``fov=False`` to disable. Masked pixels contribute 0 to RMSE/PSNR and
    are silently passed through SSIM (small structural-window effect at
    the FOV boundary, negligible in practice).

    Returns a dict including `pred_cal` (calibrated tensor for downstream
    figure-making), the `fov_mask` tensor used, and all four standard
    scalars: psnr, ssim, rmse, headroom. If `baseline` is supplied, it is
    calibrated identically and its psnr/rmse are reported as `baseline_*`;
    headroom is computed as `max(0, 1 - rmse / baseline_rmse)`.
    """
    if fg_threshold is None:
        fg_threshold = display_min + 0.05 * (display_max - display_min)
    dr = float(display_max - display_min)

    pred_cal = intensity_calibrate(pred, truth,
                                    fg_threshold=fg_threshold,
                                    display_max=display_max,
                                    bg_target=bg_target)

    # FOV mask handling. The mask is applied ONLY to LOCAL copies used for the
    # metric (SSIM/RMSE/PSNR/headroom). The returned `pred_cal`/`baseline_cal`
    # are the UNMASKED calibrated recons, so figures (make_4panel_comparison)
    # display the FULL reconstruction — recon and GT both full-frame, no circular
    # mask (the "mask=False for display" convention). The metric numbers are
    # byte-identical to the previous in-place-masked computation (mask is 0/1 and
    # nan_to_num is identity on finite pixels), so NO re-scoring is needed.
    if isinstance(fov, bool):
        mask_2d = (fov_mask(truth.shape[-1], device=pred_cal.device,
                            dtype=pred_cal.dtype) if fov else None)
    else:
        mask_2d = fov.to(device=pred_cal.device, dtype=pred_cal.dtype)

    # Sanitize non-finite recon pixels (some solvers' norm layers emit NaN/Inf on
    # near-uniform air slices; a single bad slice would poison the whole-batch
    # SSIM/PSNR aggregate). Done on the UNMASKED pred_cal so display + metric agree.
    _n_bad = int((~torch.isfinite(pred_cal)).sum())
    if _n_bad:
        print(f"[metrics] WARN: {_n_bad} non-finite calibrated-pred px -> 0 "
              f"(degenerate/air slice?)", flush=True)
    pred_cal = torch.nan_to_num(pred_cal, nan=0.0,
                                posinf=float(display_max), neginf=0.0)

    # Local masked copies for the METRIC only (display tensors stay full-frame).
    pred_m  = pred_cal * mask_2d if mask_2d is not None else pred_cal
    truth_m = truth    * mask_2d if mask_2d is not None else truth

    result = {
        "pred_cal":     pred_cal,            # UNMASKED — for display / figures
        "fov_mask":     mask_2d,
        "val_psnr":     float(psnr(pred_m, truth_m, data_range=dr).cpu()),
        "val_ssim":     float(ssim(pred_m, truth_m, data_range=dr).cpu()),
        "val_rmse":     float(((pred_m - truth_m) ** 2).mean().sqrt().cpu()),
        "fg_threshold": float(fg_threshold),
        "calibration":  "intensity_calibrate (two-point linear, bg->0, fg_mean->truth_fg_mean)"
                        + ("; metric FOV-masked, display unmasked" if mask_2d is not None else ""),
    }
    if baseline is not None:
        baseline_cal = intensity_calibrate(baseline, truth,
                                            fg_threshold=fg_threshold,
                                            display_max=display_max,
                                            bg_target=bg_target)
        baseline_cal = torch.nan_to_num(baseline_cal, nan=0.0,
                                        posinf=float(display_max), neginf=0.0)
        base_m = baseline_cal * mask_2d if mask_2d is not None else baseline_cal
        bl_rmse = float(((base_m - truth_m) ** 2).mean().sqrt().cpu())
        result["baseline_psnr"] = float(psnr(base_m, truth_m, data_range=dr).cpu())
        result["baseline_ssim"] = float(ssim(base_m, truth_m, data_range=dr).cpu())
        result["baseline_rmse"] = bl_rmse
        result["baseline_cal"]  = baseline_cal    # UNMASKED — for display / figures
        result["headroom"]      = max(0.0, 1.0 - result["val_rmse"] / max(bl_rmse, 1e-12))
    return result


# ===========================================================================
# Standard 4-panel comparison figure
# ===========================================================================
#
# All dl_sparse_view solvers should emit a comparison.png with the same
# layout so the dashboard cards are visually comparable. The layout is one
# row per scene with four columns: truth | FBP | recon (calibrated) |
# difference (recon - truth, diverging colormap).

def make_4panel_comparison(truth: torch.Tensor, fbp: torch.Tensor,
                           recon: torch.Tensor, out_path: Path | str, *,
                           display_min: float, display_max: float,
                           n_show: int = 4, solver_label: str = "recon",
                           headroom: float | None = None,
                           dpi: int = 80) -> None:
    """Write a standardised comparison.png to `out_path`.

    Layout: 4 columns × `n_show` rows.
      Col 1: ground truth   (gray, [display_min, display_max])
      Col 2: FBP baseline   (gray, same range)
      Col 3: solver recon   (gray, same range) + suptitle with solver label
      Col 4: difference recon - truth (bwr, symmetric ±(L/2))
    Each row's third column also reports per-scene PSNR / SSIM / RMSE
    computed against the ground truth (with the standard data_range), so
    the figure stays self-describing alongside the headline metrics in
    result.json. The first row additionally shows the overall headroom in
    the solver-column title.

    `truth`, `fbp`, `recon` may be (B,1,H,W) or (B,H,W); we take the first
    `n_show` scenes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Allow the test-showcase runner to widen the montage (e.g. 5 rows = one
    # per held-out test patient) without per-solver edits.
    n_show = int(os.environ.get("AGENT4CT_FIG_NSHOW", n_show))

    def _arr(t, i):
        x = t[i]
        if x.dim() == 3:
            x = x[0]
        return x.detach().cpu().numpy()

    def _per_scene_metrics(rec_t: torch.Tensor, gt_t: torch.Tensor) -> tuple[float, float, float]:
        # Convert single-scene 2D tensors to (1, 1, H, W) for psnr/ssim.
        r = rec_t.view(1, 1, *rec_t.shape[-2:]).float()
        g = gt_t.view(1, 1, *gt_t.shape[-2:]).float()
        dr = display_max - display_min
        try:
            ps = float(psnr(r, g, data_range=dr).cpu())
        except Exception:
            ps = float("nan")
        try:
            ss = float(ssim(r, g, data_range=dr).cpu())
        except Exception:
            ss = float("nan")
        rm = float(((r - g) ** 2).mean().sqrt().cpu())
        return ps, ss, rm

    n = min(n_show, truth.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    diff_lim = (display_max - display_min) / 2.0
    if headroom is not None:
        fig.suptitle(f"{solver_label}   overall headroom = {headroom:.3f}",
                     fontsize=11, y=1.0)
    for r in range(n):
        gt_t  = truth[r] if truth[r].dim() == 2 else truth[r, 0] if truth[r].dim() == 3 else truth[r].squeeze()
        fb_t  = fbp[r] if fbp[r].dim() == 2 else fbp[r, 0] if fbp[r].dim() == 3 else fbp[r].squeeze()
        rc_t  = recon[r] if recon[r].dim() == 2 else recon[r, 0] if recon[r].dim() == 3 else recon[r].squeeze()
        gt   = gt_t.detach().cpu().numpy()
        ifbp = fb_t.detach().cpu().numpy()
        irec = rc_t.detach().cpu().numpy()
        diff = irec - gt
        rec_psnr, rec_ssim, rec_rmse = _per_scene_metrics(rc_t, gt_t)
        fbp_psnr, fbp_ssim, fbp_rmse = _per_scene_metrics(fb_t, gt_t)
        for c, (img, name, vmin, vmax, cmap) in enumerate([
            (gt,   "truth",                            display_min, display_max, "gray"),
            (ifbp, f"FBP\nPSNR={fbp_psnr:.1f} SSIM={fbp_ssim:.3f} RMSE={fbp_rmse:.4f}",
                                                       display_min, display_max, "gray"),
            (irec, f"{solver_label}\nPSNR={rec_psnr:.1f} SSIM={rec_ssim:.3f} RMSE={rec_rmse:.4f}",
                                                       display_min, display_max, "gray"),
            (diff, "diff (rec - truth)",               -diff_lim,    diff_lim,    "bwr"),
        ]):
            axes[r, c].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, c].set_title(name, fontsize=9); axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Training-time non-negativity penalty (CONVENTIONS.md rule 6)
# ===========================================================================
#
# End-to-end-trained CT recon networks (ItNet, U-Swin, Hammernik VN, DD,
# etc.) need to actively suppress negative outputs during training, not
# just clip them at evaluation. A naive hard ReLU on the output kills
# gradients for any pixel with a negative pre-activation — the network
# then has no signal to push those pixels up, and the background fills
# with negative streaks that downstream intensity-calibration cannot
# undo cleanly.
#
# The fix is a smooth quadratic penalty on the negative part of `pred`,
# added to the supervised loss:
#     loss = base(pred, target) + lambda_neg · mean( max(0, -pred)^2 )
# This is differentiable everywhere (gradient is 2·max(0, -pred) which is
# nonzero exactly when pred is negative), so the network gets a constant
# push to drive negative pre-activations toward zero. Default
# `lambda_neg = 1.0` puts the penalty at roughly the same scale as the
# MSE term for our [0, 0.05] μ-range targets.


def negativity_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty on the negative part of `pred`. Returns a scalar.
    Differentiable everywhere; gradient is 2·max(0, -pred), nonzero only
    where pred < 0. Use as `lambda_neg · negativity_penalty(pred)` added
    to the base reconstruction loss during training."""
    return pred.clamp(max=0.0).pow(2).mean()


def supervised_recon_loss(pred: torch.Tensor, target: torch.Tensor,
                          *, lambda_neg: float = 1.0,
                          base: str = "mse") -> torch.Tensor:
    """Reconstruction loss with built-in non-negativity penalty.

    base = "mse" -> torch.nn.functional.mse_loss
    base = "l1"  -> torch.nn.functional.l1_loss
    """
    if base == "mse":
        recon = F.mse_loss(pred, target)
    elif base == "l1":
        recon = F.l1_loss(pred, target)
    else:
        raise ValueError(f"unknown base loss {base!r}")
    return recon + lambda_neg * negativity_penalty(pred)


def clip_and_step(optimizer, loss, grad_clip: float = 0.0) -> bool:
    """Clip the global grad-norm then ``optimizer.step()``, but SKIP the step
    (and zero the grads) if the loss OR the gradient norm is nonfinite.

    Why this exists: Mayo's 2304-view FBP adjoint amplifies training gradients
    ~18x vs demo_dl's 128 views (ramp ``|freq|`` weighting summed over many more
    views), so the backward occasionally overflows to a nonfinite gradient and
    an unguarded ``opt.step()`` poisons every weight -> the recon collapses to a
    constant (calibrated SSIM 0.3089 on the Mayo val split, identical across all
    architectures because the output is data-independent). A finite *loss* can
    still carry an Inf *gradient* whose clip yields NaN, so guarding the loss
    alone is insufficient — we check the returned grad norm too. See
    docs/findings.md 2026-06-14.

    grad_clip <= 0 -> no clipping (norm is still computed so nonfinite-gradient
    batches are skipped; demo_dl/breast keep their exact behaviour since they
    never produce one). Returns True iff the step was applied. Model-agnostic:
    pulls params straight from ``optimizer.param_groups`` so it drops into any
    solver regardless of the model variable name.
    """
    params = [p for g in optimizer.param_groups for p in g["params"]]
    max_norm = grad_clip if (grad_clip and grad_clip > 0) else float("inf")
    gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if torch.isfinite(loss) and torch.isfinite(gnorm):
        optimizer.step()
        return True
    optimizer.zero_grad(set_to_none=True)
    return False


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> torch.Tensor:
    if data_range is None:
        data_range = float(target.amax() - target.amin())
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    k = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    w = torch.exp(-0.5 * (k / sigma) ** 2)
    w = w / w.sum()
    return w.outer(w)


def ssim(pred: torch.Tensor, target: torch.Tensor,
          data_range: float | None = None,
          window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM for floating-point images.

    Uses the Wang et al. 2004 stabilisers C1 = (0.01·L)², C2 = (0.03·L)²
    where L is `data_range`. Earlier versions of this function used C1=C2=0
    on the argument that calibrated attenuation coefficients don't need
    them — that was wrong: after intensity calibration the background
    pixels of both pred and truth are exactly 0, so local 11×11 windows
    over background give mu² + sigma² ≈ 0 on both sides and the
    SSIM ratio becomes 0/0 → NaN. The Wang constants stabilise this
    edge case without measurably changing SSIM on textured regions.
    """
    if data_range is None:
        data_range = float(target.amax() - target.amin())
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype)[None, None]
    pad = window_size // 2
    mu_x = F.conv2d(pred, w, padding=pad)
    mu_y = F.conv2d(target, w, padding=pad)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    sigma_xx = F.conv2d(pred * pred, w, padding=pad) - mu_xx
    sigma_yy = F.conv2d(target * target, w, padding=pad) - mu_yy
    sigma_xy = F.conv2d(pred * target, w, padding=pad) - mu_xy
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    return (num / den).mean()
