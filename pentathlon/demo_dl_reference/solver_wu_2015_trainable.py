"""Wu 2015 with **end-to-end trainable scalar hyperparameters**.

The classical Wu et al. 2015 reconstruction (`solver_wu_2015.py`) has
several knobs that the paper hand-tunes:

  - `wu_soft_thresh`        — magnitude of the soft-threshold applied to
                              the residual reco between outer iterations
                              (one scalar; we make it per-outer-iter)
  - band roll-off sigmoid   — `sigmoid(-10 (f·ΔR − 1))`: slope 10 and
                              offset 1 (Nyquist boundary) — both fixed
                              in the original
  - per-band weights        — original code uses an equal sum-normalised
                              triangular ramp; we add a learnable
                              per-band log-scale that multiplies each
                              band's contribution before sum-normalising
  - residual blend          — original adds the residual reco at unit
                              weight; we let the optimiser learn a per-
                              outer-iter blend coefficient

Everything that is fundamentally discrete (`wu_n_bands`, `wu_n_outer`,
`wu_motion_range`, `wu_motion_window`) stays as a fixed config; only
the continuous knobs become `nn.Parameter`. Training is supervised L2
+ non-negativity penalty against the clean phantom, on the full 128
projections (same recipe as `solver_dual_ddomain_supervised.py`).

Gradients flow through:
  - `proj.forward_project` / `proj.back_project` (PYRO-NN PyTorch autograd)
  - the differentiable soft-threshold, sigmoid, FFT band filtering, and
    image-domain residual blending

`feature_preserving_interp` selects per-pixel shifts via `torch.where`;
the shift choice itself is non-differentiable but the selected midpoint
values (linear combinations of the input) carry gradients. In
practice this is enough for a useful gradient signal.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.metrics import (
    evaluate_calibrated, make_4panel_comparison, supervised_recon_loss, clip_and_step,
)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS

# Re-use the helper functions from the non-trainable Wu solver where
# possible (these don't allocate trainable state).
from pentathlon.demo_dl_reference.solver_wu_2015 import (
    _band_filters,
    _radial_distance_map,
    _build_dense_projector,
    feature_preserving_interp,
    build_dataset,
)


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # Wu 2015 structural hyperparameters (fixed, integer-valued).
    "wu_n_bands":       4,
    "wu_n_outer":       2,
    "wu_motion_range":  5,
    "wu_motion_window": 2,
    # Initial values for the trainable scalars.
    "wu_soft_thresh":   0.0015,   # per-outer-iter soft threshold init
    "wu_sigmoid_slope": 10.0,     # sigmoid steepness around Nyquist
    "wu_sigmoid_offset": 1.0,     # Nyquist threshold = f·ΔR == 1
    "wu_band_scale":    1.0,      # per-band scale init (one param per band)
    "wu_residual_blend": 1.0,     # per-outer-iter residual blend init
    # Training
    "epochs":         10,
    "batch_size":     1,
    "lr":             1e-2,       # scalar params — relatively big lr is fine
    "weight_decay":   0.0,        # AdamW decoupled wd, applied to all
                                  # trainable scalars; small values (e.g.
                                  # 1e-3) pull the optimum back toward
                                  # paper defaults and prevent the
                                  # high-frequency band scales from
                                  # collapsing to zero on small datasets.
    "loss_base":      "mse",      # "mse" or "l1"; l1 is less sensitive
                                  # to outliers and damps the band-scale
                                  # runaway behaviour seen with MSE.
    # Hard clamps on the trainable scalars to keep the optimiser out
    # of the runaway regimes observed in iters 1-9 (2026-05-22):
    # band_scale[0] amplified to 53x with train_n=2000, killing val
    # PSNR. soft_thresh[1] went to 0.30 (larger than the breast's
    # entire dynamic range), nullifying the second outer iter.
    "wu_band_scale_min":  0.1,
    "wu_band_scale_max":  3.0,
    "wu_blend_min":      -0.5,
    "wu_blend_max":       2.0,
    "wu_soft_thresh_min": 1.0e-5,
    "wu_soft_thresh_max": 0.05,
    "lambda_neg":     1.0,
}


class TrainableWu2015(nn.Module):
    """End-to-end-trainable Wu 2015 reconstruction.

    Trainable parameters (all very small):
      - ``log_band_scale``  (n_bands,)  — per-band multiplicative scale
      - ``sigmoid_slope``   ()          — band roll-off steepness
      - ``sigmoid_offset``  ()          — Nyquist threshold offset
      - ``log_soft_thresh`` (n_outer,)  — per-iter soft threshold
      - ``residual_blend``  (n_outer,)  — per-iter residual blend coef

    Total: n_bands + 2 + n_outer + n_outer trainable scalars.
    For the defaults (n_bands=4, n_outer=2) this is 10 params.
    """

    def __init__(self, geometry: FanBeamGeometry, cfg: dict, device: str):
        super().__init__()
        self.geometry = geometry
        self.proj = PyronnFanBeamProjector(geometry)
        self.dense_proj = _build_dense_projector(geometry, factor=2, device=device)

        n_bands = int(cfg["wu_n_bands"])
        n_outer = int(cfg["wu_n_outer"])
        self.n_bands = n_bands
        self.n_outer = n_outer
        self.motion_range = int(cfg["wu_motion_range"])
        self.motion_window = int(cfg["wu_motion_window"])

        # Pre-compute fixed pieces and register as buffers so they
        # move with .to(device) but are NOT optimised.
        bf, bc = _band_filters(geometry.n_det, geometry.det_spacing,
                                n_bands, device=device)
        self.register_buffer("band_filters", bf)              # (B, D)
        self.register_buffer("band_centers", bc)              # (B,)
        r = _radial_distance_map(geometry.image_size, geometry.pixel_spacing,
                                 device=device)
        self.register_buffer("r_map", r)                      # (H, W)
        self.delta_beta = 2.0 * math.pi / float(geometry.n_angles)

        # Trainable scalars.
        self.log_band_scale = nn.Parameter(
            torch.full((n_bands,), math.log(float(cfg["wu_band_scale"]))))
        self.sigmoid_slope = nn.Parameter(
            torch.tensor(float(cfg["wu_sigmoid_slope"])))
        self.sigmoid_offset = nn.Parameter(
            torch.tensor(float(cfg["wu_sigmoid_offset"])))
        self.log_soft_thresh = nn.Parameter(
            torch.full((n_outer,), math.log(float(cfg["wu_soft_thresh"]))))
        self.residual_blend = nn.Parameter(
            torch.full((n_outer,), float(cfg["wu_residual_blend"])))

        # Clamp bounds (registered as buffers so they move to device and
        # appear in state_dict but are not optimised).
        self.register_buffer("band_scale_min",
                             torch.tensor(float(cfg["wu_band_scale_min"])))
        self.register_buffer("band_scale_max",
                             torch.tensor(float(cfg["wu_band_scale_max"])))
        self.register_buffer("blend_min",
                             torch.tensor(float(cfg["wu_blend_min"])))
        self.register_buffer("blend_max",
                             torch.tensor(float(cfg["wu_blend_max"])))
        self.register_buffer("soft_thresh_min",
                             torch.tensor(float(cfg["wu_soft_thresh_min"])))
        self.register_buffer("soft_thresh_max",
                             torch.tensor(float(cfg["wu_soft_thresh_max"])))

    # ------------------------------------------------------------------ #
    def _aliasing_free_fbp(self, sino: torch.Tensor) -> torch.Tensor:
        """View-aliasing-free reconstruction with trainable band weighting."""
        device = sino.device
        H, W = self.r_map.shape
        delta_R = self.r_map * self.delta_beta                 # (H, W)
        fR = self.band_centers.view(-1, 1, 1) * delta_R.unsqueeze(0)  # (B, H, W)
        weights = torch.sigmoid(
            -self.sigmoid_slope * (fR - self.sigmoid_offset))  # (B, H, W)
        band_scale = torch.clamp(
            torch.exp(self.log_band_scale),
            min=float(self.band_scale_min), max=float(self.band_scale_max),
        ).view(-1, 1, 1)
        weights = weights * band_scale
        wsum = weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
        weights = weights / wsum                                # (B, H, W)

        sino4 = sino if sino.dim() == 4 else sino.unsqueeze(1)
        N = sino4.shape[0]
        spec = torch.fft.fft(sino4, dim=-1, norm="ortho")
        out = torch.zeros(N, 1, H, W, device=device, dtype=torch.float32)
        rw = self.proj._redundancy_weights.to(device)
        for i in range(self.n_bands):
            h = self.band_filters[i]
            filtered_spec = spec * h.view(1, 1, 1, -1)
            filtered_sino = torch.fft.ifft(filtered_spec, dim=-1, norm="ortho").real
            if rw.shape[0] == filtered_sino.shape[-2]:
                weighted = filtered_sino * rw.view(1, 1, *rw.shape)
            else:
                weighted = filtered_sino
            bp = self.proj.back_project(weighted)
            out = out + bp * weights[i].view(1, 1, H, W)
        return out

    # ------------------------------------------------------------------ #
    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """End-to-end Wu 2015 reconstruction with trainable knobs."""
        g = self._aliasing_free_fbp(sino)
        for it in range(self.n_outer):
            # Differentiable forward project for the residual (gradient
            # to the trainable image-domain blend coefficient flows back).
            fp = self.proj.forward_project(g)
            residual_sino = sino - fp
            # Feature-preserving interpolation (shift selection is
            # piecewise-constant w.r.t. cost; midpoint values carry
            # gradient through their linear combination of inputs).
            residual_dense = feature_preserving_interp(
                residual_sino, max_shift=self.motion_range,
                window=self.motion_window)
            residual_img = self.dense_proj.fbp(residual_dense, filter_name="hann")
            soft_thresh = torch.clamp(
                torch.exp(self.log_soft_thresh[it]),
                min=float(self.soft_thresh_min),
                max=float(self.soft_thresh_max),
            )
            residual_img = torch.sign(residual_img) * torch.clamp(
                residual_img.abs() - soft_thresh, min=0.0)
            blend = torch.clamp(
                self.residual_blend[it],
                min=float(self.blend_min), max=float(self.blend_max),
            )
            g = g + blend * residual_img
        return g.clamp_min(0.0)


def build_dataset(geom, n, seed, i0, sigma_e, device):  # noqa: F811
    """Local override of the shared loader: return per-sample ps (4-tuple)."""
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    import os
    env_path = os.environ.get("WU_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        env_cfg = json.loads(Path(env_path).read_text())
        cfg = {**CONFIG, **env_cfg, **(cfg or {})}
        print(f"[solver] Loaded config from {env_path}", flush=True)
    else:
        cfg = {**CONFIG, **(cfg or {})}
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg, default=str)}",
          flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    train_ph, train_clean, train_noisy, train_ps = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000,
        cfg["noise_i0"], cfg["noise_sigma_e"], device)

    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)
    # Wu's residual path uses a 2x-dense projector. Build one per distinct ps
    # (lazy, cached), keyed identically to _projs, so model.dense_proj can be
    # swapped per slice alongside model.proj.
    _dense = {}
    def _dense_for(k):
        if k not in _dense:
            _dense[k] = _build_dense_projector(_projs[k].geom, factor=2, device=device)
        return _dense[k]
    proj = PyronnFanBeamProjector(geom).to(device)
    with torch.no_grad():
        if per_ps:
            fbp_init = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            fbp_init = torch.clamp(proj.fbp(val_noisy), min=0.0)

    model = TrainableWu2015(geom, cfg, device=device).to(device)
    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[solver] Trainable-Wu2015 params: {params_total} "
          f"(n_bands={cfg['wu_n_bands']}, n_outer={cfg['wu_n_outer']})", flush=True)

    # AdamW for decoupled weight decay (pulls params toward 0, but our
    # init is near sane paper defaults so we keep wd small; wd=0
    # recovers plain Adam behaviour).
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg.get("weight_decay", 0.0))

    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        n_seen = 0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if per_ps:
                k = float(_trk[int(idx[0])])
                model.proj = _projs[k]
                model.dense_proj = _dense_for(k)
            sino = train_noisy[idx].to(device)
            truth = train_ph[idx].to(device)
            pred = model(sino)
            loss = supervised_recon_loss(pred, truth,
                                          lambda_neg=cfg["lambda_neg"],
                                          base=cfg.get("loss_base", "mse"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
            running += float(loss.detach().cpu()) * idx.numel()
            n_seen += idx.numel()
        mean_loss = running / max(1, n_seen)

        with torch.no_grad():
            bs = torch.exp(model.log_band_scale).tolist()
            st = torch.exp(model.log_soft_thresh).tolist()
            rb = model.residual_blend.tolist()
            sl = float(model.sigmoid_slope)
            so = float(model.sigmoid_offset)
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  "
              f"band_scale={['%.3f'%v for v in bs]} "
              f"sigmoid(slope={sl:.3f} offset={so:.3f}) "
              f"soft_thresh={['%.5f'%v for v in st]} "
              f"blend={['%.3f'%v for v in rb]}", flush=True)

    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        chunk = 1 if per_ps else cfg.get("val_chunk", 4)  # Wu memory-heavy (dense proj)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            if per_ps:
                k = float(_vrk[i])
                model.proj = _projs[k]
                model.dense_proj = _dense_for(k)
            preds.append(model(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)

    pred = pred.clamp_min(0.0)
    fbp_init = fbp_init.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=fbp_init,
        display_min=cfg["display_min"], display_max=cfg["display_max"])
    pred_cal = metrics["pred_cal"]
    baseline_cal = metrics["baseline_cal"]
    val_psnr, val_ssim, val_rmse = metrics["val_psnr"], metrics["val_ssim"], metrics["val_rmse"]
    baseline_psnr, baseline_rmse = metrics["baseline_psnr"], metrics["baseline_rmse"]
    headroom = metrics["headroom"]
    val_score = val_ssim

    result = {
        "val_score": val_score, "val_psnr": val_psnr, "val_ssim": val_ssim,
        "val_rmse": val_rmse, "baseline_psnr": baseline_psnr,
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_rmse": baseline_rmse, "headroom": headroom,
        "calibration": metrics["calibration"],
        "fg_threshold": metrics["fg_threshold"],
        "params_total": params_total, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time,
        "config": cfg,
        "training_scheme": "supervised_l2_full_views_128",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Trainable-Wu2015: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} params={params_total}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="Wu2015-L2", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
