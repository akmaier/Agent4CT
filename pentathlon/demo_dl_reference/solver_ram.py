"""Reference: RAM (Terris 2025) — Reconstruct Anything Model.

arXiv 2503.08915. Code https://github.com/matthieutrs/ram (BSD-3-Clause).
See literature/terris_2025_ram.md for the architectural notes and the port
plan.

This solver loads the pretrained RAM checkpoint from HuggingFace
(`mterris/ram/ram.pth.tar`, 143 MB) and runs it on our 128-angle / 512²
fan-beam geometry. RAM is **non-iterative** (single forward U-Net pass)
but consumes the forward operator at inference via in-block Krylov
embeddings — so we expose our PyronnFanBeamProjector as a thin
`deepinv.physics.LinearPhysics` adapter (`PyronnFanBeamPhysics` below).

The agentic search varies the inference-time knobs:
  - ram_sigma: noise level conditioning passed to physics.noise_model
  - ram_input_norm: how to scale x/y into ~[0,1] before the network
  - ram_clamp_output: clip the prediction to [display_min, display_max]
  - ram_finetune: do a few epochs of self-supervised SURE/EI finetune
  - ram_finetune_epochs / ram_finetune_lr
  - ram_factor: operator gain estimate (or 0 to skip prox_l2 realignment)
  - ram_post_fbp_blend: linear blend with the FBP-init image
  - ram_disable_multiscale: bypass MultiScaleLinearPhysics (single scale)

Cluster usage:
    DIFFUSION_RECON_CONFIG_PATH=...   # used by other solvers
    RAM_CONFIG_PATH=/path/to/cfg.json # used here
    python solver_ram.py <out_dir>
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose
from ddssl_ldct.metrics import psnr, ssim, evaluate_calibrated, make_4panel_comparison


CONFIG = {
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,
    "train_n":       400,             # unused (RAM is pretrained)
    "val_n":         20,              # smaller default to keep per-iter cost low
    "noise_i0":      1e5,
    "noise_sigma_e": 10.0,
    "seed":          42,
    "display_min":   0.0,
    "display_max":   0.05,
    # ---- RAM-specific knobs (see docstring) ---------------------------------
    "ram_ckpt_path":         "/cluster/maier/Agent4CT/checkpoints/ram.pth.tar",
    "ram_sigma":             5e-3,
    "ram_input_norm":        "display_max",   # "display_max" | "fbp_max" | "none"
    "ram_clamp_output":      True,
    "ram_finetune":          False,
    "ram_finetune_epochs":   0,
    "ram_finetune_lr":       1e-4,
    "ram_factor":            1.0,             # 0 = skip prox_l2 realign
    "ram_post_fbp_blend":    0.0,             # 0 = pure RAM, 1 = pure FBP
    "ram_disable_multiscale": False,
    "ram_disable_cudnn":     False,
    "ram_use_deepinv_tomo":  False,  # if True: skip PyroNN adapter, use deepinv's
                                     # built-in parallel-beam Tomography operator
                                     # (diagnostic: confirms RAM works at our image size)
}


# ---------------------------------------------------------------------------
# PyroNN ↔ deepinv adapter. Constructed lazily inside main() so importing this
# module is cheap (deepinv pulls in a lot of transitive deps).
def _estimate_op_scale(proj, image_size, device, n_iter=25):
    """Power iteration to estimate ||A^T A||^(1/2) = ||A||. We then divide
    both A and A^T by this value so that ||A_normalised|| ≈ 1, matching
    what RAM was trained on (deepinv operators with normalize=True).

    Without this, RAM's in-block Krylov tower [A^T y, (A^T A) A^T y, ...]
    diverges to NaN over ~50 stacked physics evaluations.
    """
    x = torch.randn(1, 1, image_size, image_size, device=device)
    x = x / x.norm()
    for _ in range(n_iter):
        y = proj.forward_project(x)
        x_new = proj.back_project(y)
        norm = float(x_new.norm())
        if norm < 1e-12:
            return 1.0
        x = x_new / norm
    # `norm` after the last step is ||A^T A x|| / ||x|| ≈ ||A^T A|| = ||A||^2
    return float(norm) ** 0.5


def _make_physics(proj: "PyronnFanBeamProjector", sigma: float, device: str,
                  *, op_scale: float = 1.0):
    import deepinv as dinv  # lazy import
    from deepinv.physics import LinearPhysics

    def _to_4d(t):
        # Robust: PYRO-NN projector accepts (B,H,W) or (B,1,H,W); deepinv +
        # RAM consume (B,1,H,W). Unify at the adapter boundary.
        return t.unsqueeze(1) if t.dim() == 3 else t

    class PyronnFanBeamPhysics(LinearPhysics):
        def __init__(self):
            super().__init__(noise_model=dinv.physics.GaussianNoise(sigma=sigma))
            self.projector = proj
            self.factor = 1.0
            # Operator-norm divisor. A_normalised = A / op_scale; A^T_normalised
            # = A^T / op_scale. With op_scale = ||A||, the composed A^T A has
            # spectral radius 1, so RAM's Krylov tower stays bounded.
            self.op_scale = op_scale

        def A(self, x, **_):
            x4 = _to_4d(x)
            out = self.projector.forward_project(x4) / self.op_scale
            return _to_4d(out)

        def A_adjoint(self, y, **_):
            y4 = _to_4d(y)
            out = self.projector.back_project(y4) / self.op_scale
            return _to_4d(out)

        def update_parameters(self, **_):
            return None

    return PyronnFanBeamPhysics()


def _build_phantoms_or_load(geom, n, seed, device):
    """Returns a (n, 1, H, W) float tensor of μ-images.

    `random_ellipses_phantom(...)[0]` is already (1, H, W) (the function adds
    a leading (None, None) so its full return is (1, 1, H, W) — indexing [0]
    drops the outermost). Stacking N of those gives (N, 1, H, W) directly;
    no extra unsqueeze is needed.
    """
    phantoms = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)                       # (N, 1, H, W)
    assert phantoms.dim() == 4 and phantoms.shape[1] == 1, phantoms.shape
    return phantoms


def _build_dataset(geom, n, seed, i0, sigma_e, device):
    proj = PyronnFanBeamProjector(geom).to(device)
    phantoms = _build_phantoms_or_load(geom, n, seed, device)
    with torch.no_grad():
        clean = proj.forward_project(phantoms)
        noisy = simulate_low_dose(clean, i0=i0, sigma_e=sigma_e, seed=seed + 10_000)
    return phantoms, clean, noisy, proj


def _load_ram(ckpt_path: Path, device: str):
    """Load RAM model + pretrained weights. Raises if either deepinv or the
    checkpoint is missing — caller should turn that into a useful error."""
    try:
        from ram import RAM
    except ImportError as e:
        raise RuntimeError(
            "RAM not installed. On the cluster: "
            "`pip install git+https://github.com/matthieutrs/ram`"
        ) from e

    # `RAM.__init__` accepts pretrained=True to auto-download from HuggingFace,
    # but we want a local cluster path for reproducibility. Construct empty,
    # then load_state_dict from local file.
    model = RAM(in_channels=[1, 2, 3], device=device, pretrained=False)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"RAM checkpoint missing at {ckpt_path}. "
            f"Pre-stage with: huggingface-cli download mterris/ram ram.pth.tar "
            f"--local-dir $(dirname {ckpt_path})"
        )
    state = torch.load(ckpt_path, map_location=device)
    # The checkpoint may be wrapped (state['model'] or similar). Try both.
    sd = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ram] state_dict load: {len(missing)} missing, "
          f"{len(unexpected)} unexpected keys "
          f"(total checkpoint keys: {len(sd)})", flush=True)
    if missing[:3]:
        print(f"[ram]   missing sample: {missing[:3]}", flush=True)
    if unexpected[:3]:
        print(f"[ram]   unexpected sample: {unexpected[:3]}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _maybe_disable_multiscale():
    """Hot-patch RAM so MultiScaleLinearPhysics is a no-op (scales=[1] only).

    Class-level monkey-patch; affects every subsequent RAM forward call.
    """
    try:
        from ram.models.ram import MultiScaleLinearPhysics
    except ImportError:
        return
    # Replace MultiScaleLinearPhysics.__init__ to skip the scale composition.
    orig_init = MultiScaleLinearPhysics.__init__

    def patched_init(self, physics, *args, **kwargs):
        # Bypass: just hold the base physics, ignore scales.
        orig_init(self, physics, *args, **kwargs)
        self.scales = [1]   # only the base scale

    MultiScaleLinearPhysics.__init__ = patched_init


def _ram_reconstruct(model, physics, y_norm, x_init_norm,
                     finetune_epochs: int, finetune_lr: float,
                     post_fbp_blend: float, device: str):
    """Run RAM on a single (B, 1, A, D) sinogram. Returns (B, 1, H, W) prediction
    in the same normalised range as y_norm (caller denormalises)."""
    if finetune_epochs > 0:
        try:
            from ram import finetune
        except ImportError:
            print("[ram] finetune unavailable — skipping", flush=True)
        else:
            # Unfreeze parameters for finetune; we froze them in _load_ram for
            # the pure zero-shot inference path.
            for p in model.parameters():
                p.requires_grad_(True)
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ram] finetune {finetune_epochs} epochs @ lr={finetune_lr}  "
                  f"trainable={n_train/1e6:.2f}M", flush=True)
            model = finetune(model, y_norm, physics,
                             lr=finetune_lr, max_iter=finetune_epochs)
            # Re-freeze for the actual inference call below.
            for p in model.parameters():
                p.requires_grad_(False)

    with torch.no_grad():
        x_hat = model(y_norm, physics=physics)

    if post_fbp_blend > 0.0:
        x_hat = (1.0 - post_fbp_blend) * x_hat + post_fbp_blend * x_init_norm

    return x_hat


# ---------------------------------------------------------------------------
def main(out_dir: Path, cfg: dict | None = None) -> dict:
    env_path = os.environ.get("RAM_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        cfg = {**CONFIG, **json.loads(Path(env_path).read_text()), **(cfg or {})}
    else:
        cfg = {**CONFIG, **(cfg or {})}

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    print(f"[solver] RAM zero-shot device={device}", flush=True)
    print(f"[solver] cfg={json.dumps({k:v for k,v in cfg.items() if k.startswith('ram_')}, default=str)}",
          flush=True)

    if cfg.get("ram_disable_cudnn", False):
        print("[solver] disabling cuDNN entirely (PyTorch native fallback)",
              flush=True)
        torch.backends.cudnn.enabled = False
    if cfg.get("ram_disable_multiscale", False):
        print("[solver] disabling multiscale physics conditioning", flush=True)
        _maybe_disable_multiscale()

    # Build val set + projector (same val seed convention as other solvers).
    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"])
    val_ph, _, val_noisy, proj = _build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)
    with torch.no_grad():
        val_fbp = torch.clamp(proj.fbp(val_noisy), min=0.0)

    # Load model.
    model = _load_ram(Path(cfg["ram_ckpt_path"]), device)
    if cfg.get("ram_use_deepinv_tomo", False):
        # Diagnostic mode: ignore our PyroNN adapter; use deepinv's own
        # parallel-beam Tomography so we can compare RAM behaviour.
        import deepinv as dinv
        physics = dinv.physics.Tomography(
            img_width=cfg["image_size"],
            angles=cfg["n_angles"],
            noise_model=dinv.physics.GaussianNoise(sigma=cfg["ram_sigma"]),
            device=device, normalize=True)
        # Override the sinogram for each scene with our actual val_noisy,
        # since deepinv's Tomography forward differs from PyroNN's; we
        # still want to evaluate on the same val phantoms. Recompute val
        # sinos via the deepinv operator on val_ph:
        with torch.no_grad():
            val_noisy = physics(val_ph)  # noisy sinograms in deepinv conventions
            # FBP-init for the new operator: deepinv has A_dagger ≈ FBP
            val_fbp = physics.A_dagger(val_noisy)
        print(f"[solver] DIAGNOSTIC mode: using deepinv Tomography "
              f"(parallel-beam, angles={cfg['n_angles']}, img={cfg['image_size']})",
              flush=True)
    else:
        # Estimate operator norm so the normalised A satisfies ||A||≈1,
        # matching RAM's training-time deepinv operator conventions.
        op_scale = _estimate_op_scale(proj, cfg["image_size"], device)
        print(f"[solver] estimated ||A|| ≈ {op_scale:.4g} -> normalised "
              f"operator A/op_scale will have spectral norm ~1", flush=True)
        physics = _make_physics(proj, cfg["ram_sigma"], device,
                                 op_scale=op_scale)
        physics.factor = float(cfg["ram_factor"])
        # IMPORTANT: scale the val sinograms by 1/op_scale too, so that
        # y_normalised = A_normalised x = (A x)/op_scale matches what the
        # physics adapter projects. Keep val_fbp in the original mu range
        # (it's only used for normalisation in the input-norm step).
        val_noisy = val_noisy / op_scale

    # Run each val scene one at a time (memory + simpler). Force 4D shapes
    # (B, 1, *) so the deepinv physics adapter + RAM heads see canonical
    # single-channel inputs. simulate_low_dose can collapse the channel dim,
    # and val_fbp comes from proj.fbp which may also drop it — re-add here.
    out_scale = float(cfg["display_max"])
    norm_mode = cfg["ram_input_norm"]
    preds = []
    t0 = time.time()

    def _b1(t):
        # Ensure leading shape is (B, 1, *).
        if t.dim() == 3:
            return t.unsqueeze(1)
        if t.dim() == 4:
            return t
        raise ValueError(f"unexpected shape {t.shape}")

    for i in range(cfg["val_n"]):
        y = _b1(val_noisy[i:i + 1])
        x_init = _b1(val_fbp[i:i + 1])

        # Compute normalisation scale and rescaled inputs. Whatever `denom`
        # we apply to y, the image-domain prediction comes out in [x / denom]
        # units — so the inverse-scale multiplier at the end must be the
        # SAME `denom`, not `out_scale`.
        if norm_mode == "display_max":
            denom = out_scale
            y_n = y / denom
            x_init_n = x_init / denom
        elif norm_mode == "fbp_max":
            denom = float(x_init.max().clamp(min=1e-6))
            y_n = y / denom
            x_init_n = x_init / denom
        elif norm_mode == "adjoint_max":
            # Scale sino so that A_norm^T(y/s) has max ≈ 1 — what RAM was
            # trained on in image-domain. Note that y is already in the
            # operator-normalised sino space (we divided val_noisy by
            # op_scale upstream), so `denom` here is on the normalised sino.
            with torch.no_grad():
                adj = physics.A_adjoint(y)  # A^T_normalised
                denom = float(adj.abs().max().clamp(min=1e-6))
            y_n = y / denom
            x_init_n = x_init / x_init.max().clamp(min=1e-6)
        else:
            denom = 1.0
            y_n = y
            x_init_n = x_init

        if i == 0:
            with torch.no_grad():
                adj_n = physics.A_adjoint(y_n)
            print(f"[norm] mode={norm_mode}  denom={denom:.4g}  "
                  f"y range=[{float(y.min()):.3g},{float(y.max()):.3g}] -> "
                  f"y_n range=[{float(y_n.min()):.3g},{float(y_n.max()):.3g}]  "
                  f"FBP_max={float(x_init.max()):.4g}  "
                  f"A_norm^T(y_n) range=[{float(adj_n.min()):.3g}, "
                  f"{float(adj_n.max()):.3g}]", flush=True)

        # RAM expects `physics` to project on the same scale as y_n. The
        # operator is linear, so re-using the unscaled projector with
        # y_n = (1/denom) * y gives self-consistent inference.

        x_hat_n = _ram_reconstruct(
            model, physics, y_n, x_init_n,
            finetune_epochs=int(cfg["ram_finetune_epochs"]) if cfg["ram_finetune"] else 0,
            finetune_lr=float(cfg["ram_finetune_lr"]),
            post_fbp_blend=float(cfg["ram_post_fbp_blend"]),
            device=device,
        )
        if i == 0:
            n_nan = int(torch.isnan(x_hat_n).sum())
            n_inf = int(torch.isinf(x_hat_n).sum())
            print(f"[ram] x_hat_n range=[{float(x_hat_n[~torch.isnan(x_hat_n)].min() if n_nan==0 else float('nan')):.3g}, "
                  f"{float(x_hat_n[~torch.isnan(x_hat_n)].max() if n_nan==0 else float('nan')):.3g}]  "
                  f"nan={n_nan}  inf={n_inf}", flush=True)
        # Always denormalise by the SAME `denom` we scaled y by — the
        # operator is linear so input scale * 1/denom -> output scale *
        # 1/denom; multiplying back by denom recovers mu-domain units.
        pred_i = x_hat_n * denom
        if cfg.get("ram_clamp_output", True):
            pred_i = pred_i.clamp(float(cfg["display_min"]), out_scale)
        preds.append(pred_i.detach())
        if (i + 1) % 5 == 0:
            print(f"[infer] {i+1}/{cfg['val_n']}  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    sample_time = time.time() - t0
    pred = torch.cat(preds, 0)               # (n, 1, H, W)
    val_ph_4d = val_ph                        # (n, 1, H, W)

    # Intensity-calibrated evaluation (CONVENTIONS.md rule 4): two-point
    # linear calibration of BOTH pred and baseline against truth before any
    # PSNR/SSIM/RMSE so the leaderboard is comparable across solvers.
    # Restore pre-calibration ReLU clamp (CONVENTIONS.md rule 2):
    # negative outliers in the raw pred would otherwise pull the bg mean
    # negative inside evaluate_calibrated and bias the linear calibration.
    pred = pred.clamp_min(0.0)
    val_fbp = val_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph_4d, baseline=val_fbp,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        "val_score": val_ssim,
        "val_psnr": val_psnr, "val_ssim": val_ssim, "val_rmse": val_rmse,
        "baseline_psnr": baseline_psnr, "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse,
        "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_M": params_M,
        "train_n": 0, "val_n": int(pred.shape[0]),
        "train_time_s": sample_time,
        "config": cfg,
        "ram_pretrained": True,
        "ram_finetune_used": bool(cfg["ram_finetune"]),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] hr={headroom:.4f}  SSIM={val_ssim:.4f}  PSNR={val_psnr:.2f}"
          f"  RMSE={val_rmse:.5f}  baseline_PSNR={baseline_psnr:.2f}  "
          f"(intensity-calibrated)", flush=True)

    # Standardised 4-panel comparison: truth | FBP_cal | RAM_cal | diff.
    try:
        make_4panel_comparison(
            truth=val_ph_4d, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="RAM", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()
    main(args.out_dir)
