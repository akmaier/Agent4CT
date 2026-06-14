"""Reference: Dual-Domain bilateral filters trained with **supervised L2**
on **all 128 projections** (no Noise2Inverse split).

Motivation (2026-05-22):
  The Noise2Inverse self-supervised pipeline (`solver_dual_ddomain_bilateral_n2i.py`)
  splits the 128 fan-beam views into two half-sets of 64 and uses the FBP
  of one half-set as a (noisy) target for the other. On breast-CT the
  N2I targets have an irreducible noise floor — minimising MSE against
  noisy targets *encourages over-smoothing*: the bilateral filter's
  spatial sigmas keep growing across training (img_sx 0.5 → 1.08 in 2
  epochs in the agentic iter-1, 0.3 → 0.87 in iter-2), and the
  reconstruction PSNR drops below the FBP baseline.

  This variant removes that bias entirely:
    - Full-view forward path: ``img_dn(R_full.fbp(proj_dn(sino_full)))``.
      The reconstruction sees all 128 angles, not 64.
    - Loss: ``mse(pred, clean_phantom) + lambda_neg * negativity_penalty(pred)``.
      No half-set split, no noisy target, no self-supervision.

  Trade-off: this needs the clean phantom at train time. On the breast-CT
  staged dataset we have it; this solver is therefore not "fair" against
  truly unsupervised baselines, but it isolates the question
  "is N2I the bottleneck for low-parameter BFs on dense scans?".

Everything else mirrors `solver_dual_ddomain_bilateral_n2i.py`:
  - Same `TrainableBilateralFilter2d` (6 trainable parameters total)
  - Same intensity-calibrated evaluation
  - Same comparison.png panel layout
"""
from __future__ import annotations
import argparse
import json
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
from ddssl_ldct.models import TrainableBilateralFilter2d
from ddssl_ldct.metrics import (
    evaluate_calibrated, make_4panel_comparison,
    supervised_recon_loss, clip_and_step,
)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # Bilateral filter initial parameters (init kept conservative — the
    # whole point of this variant is that the L2 supervised loss should
    # not blow up the spatial sigmas the way N2I does).
    #
    # ``proj_n_bf`` / ``img_n_bf`` chain N independent Trainable BF
    # layers in series in each domain (Wagner 2022 §3.2, also used as
    # the BF-tail trick in pentathlon/dl_sparse_view/solver.py iter-57).
    # Each BF contributes 3 trainable params (σ_x, σ_y, σ_r); 3 BFs per
    # domain → 9 params per domain → 18 params total. Still ≪ U-Net.
    "proj_n_bf":     1,
    "img_n_bf":      1,
    "proj_kernel":   3,
    "proj_sx":       0.01,    # along-detector — keep tiny to avoid radial blur
    "proj_sy":       0.3,     # along-angle — modest, helps interpolate sparse views
    "proj_sr":       0.0005,  # intensity sigma in proj domain: very small (user direction)
    "img_kernel":    5,
    "img_sx":        0.5,
    "img_sy":        0.5,
    "img_sr":        0.02,
    # Training
    "epochs":        10,
    "batch_size":    1,
    "lr":            5e-3,    # Wagner's recommended BF lr (vs 5e-5 for U-Nets)
    "optimizer":     "adam",
    "lambda_neg":    1.0,     # non-negativity penalty weight on image-domain output
}


class BilateralFilterStack(nn.Module):
    """N independent TrainableBilateralFilter2d layers in series.

    Per Wagner 2022 §3.2: cascading bilateral filters increases the
    effective receptive field while keeping the parameter count tiny
    (3 trainable params per BF). All N filters share kernel size but
    are initialised from the same σ values and learn independently.
    """

    def __init__(self, n_filters: int, kernel_size: int,
                 sigma_x: float, sigma_y: float, sigma_r: float):
        super().__init__()
        assert n_filters >= 1
        self.filters = nn.ModuleList([
            TrainableBilateralFilter2d(kernel_size=kernel_size,
                                       sigma_x=sigma_x, sigma_y=sigma_y,
                                       sigma_r=sigma_r)
            for _ in range(n_filters)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bf in self.filters:
            x = bf(x)
        return x

    @torch.no_grad()
    def sigmas(self) -> list[tuple[float, float, float]]:
        return [
            (float(torch.exp(bf.log_sx).cpu()),
             float(torch.exp(bf.log_sy).cpu()),
             float(torch.exp(bf.log_sr).cpu()))
            for bf in self.filters
        ]


def build_dataset(geom, n, seed, i0, sigma_e, device):
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)   # 4-tuple; ps=None for non-mayo


class FullViewBilateralPipeline(nn.Module):
    """Single-pass DD-BF: proj_dn -> FBP(128 views) -> img_dn.

    ``proj_dn`` and ``img_dn`` are each ``BilateralFilterStack`` instances
    (one or more chained Trainable BFs).
    """

    def __init__(self, geometry: FanBeamGeometry,
                 proj_dn: nn.Module, img_dn: nn.Module):
        super().__init__()
        self.geometry = geometry
        self.proj_dn = proj_dn
        self.img_dn = img_dn
        self.R_full = PyronnFanBeamProjector(geometry)

    def forward(self, sino_full: torch.Tensor) -> torch.Tensor:
        s = self.proj_dn(sino_full)
        r = self.R_full.fbp(s)
        return self.img_dn(r)


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    import os
    env_config_path = os.environ.get("DD_CONFIG_PATH")
    if env_config_path and Path(env_config_path).exists():
        with open(env_config_path) as f:
            env_cfg = json.load(f)
        cfg = {**CONFIG, **env_cfg}
        print(f"[solver] Loaded config from {env_config_path}")
    elif cfg is not None:
        cfg = {**CONFIG, **cfg}
    else:
        cfg = CONFIG.copy()
    from ddssl_ldct.staged_dataset import get_dataset_kind, geometry_overrides
    cfg["dataset_kind"] = get_dataset_kind(cfg)
    if cfg["dataset_kind"] != "phantoms":
        cfg.update(geometry_overrides(cfg["dataset_kind"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg, default=str)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"], sod=cfg["sod"], sdd=cfg["sdd"],
    )

    train_ph, train_clean, train_noisy, train_ps = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy, val_ps = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # PER-SAMPLE geometry (Mayo canonical): reconstruct each slice at its own
    # ps_eff via a per-ps projector cache; swap pipe.R_full per sample (bs=1).
    from ddssl_ldct.staged_dataset import (mayo_per_sample_setup,
                                           mayo_per_sample_fbp)
    per_ps, _projs, _trk, _vrk = mayo_per_sample_setup(train_ps, val_ps, cfg, device)

    with torch.no_grad():
        if per_ps:
            ld_fbp = mayo_per_sample_fbp(_projs, _vrk, val_noisy, cfg["image_size"])
        else:
            R_full = PyronnFanBeamProjector(geom).to(device)
            ld_fbp = torch.clamp(R_full.fbp(val_noisy), min=0.0)

    proj_dn = BilateralFilterStack(
        n_filters=cfg["proj_n_bf"], kernel_size=cfg["proj_kernel"],
        sigma_x=cfg["proj_sx"], sigma_y=cfg["proj_sy"], sigma_r=cfg["proj_sr"],
    )
    img_dn = BilateralFilterStack(
        n_filters=cfg["img_n_bf"], kernel_size=cfg["img_kernel"],
        sigma_x=cfg["img_sx"], sigma_y=cfg["img_sy"], sigma_r=cfg["img_sr"],
    )
    pipe = FullViewBilateralPipeline(geom, proj_dn, img_dn).to(device)

    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] Supervised-L2 BF stack: proj_n_bf={cfg['proj_n_bf']} "
          f"img_n_bf={cfg['img_n_bf']} params_total={params_total} "
          f"(proj={sum(p.numel() for p in proj_dn.parameters())}, "
          f"img={sum(p.numel() for p in img_dn.parameters())})", flush=True)

    opt = torch.optim.Adam(pipe.parameters(), lr=cfg["lr"])

    t0 = time.time()
    for ep in range(cfg["epochs"]):
        pipe.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        n_seen = 0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if per_ps:                      # swap to this slice's ps projector
                pipe.R_full = _projs[float(_trk[int(idx[0])])]
            sino = train_noisy[idx].to(device)
            truth = train_ph[idx].to(device)
            pred = pipe(sino)
            loss = supervised_recon_loss(pred, truth,
                                          lambda_neg=cfg["lambda_neg"], base="mse")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_and_step(opt, loss, cfg.get("grad_clip", 0.0))
            running += float(loss.detach().cpu()) * idx.numel()
            n_seen += idx.numel()
        mean_loss = running / max(1, n_seen)

        proj_sigmas = proj_dn.sigmas()
        img_sigmas = img_dn.sigmas()
        # Compact log: print each BF in the stack on one line.
        proj_str = "; ".join(
            f"σx={sx:.4f} σy={sy:.4f} σr={sr:.5f}" for (sx, sy, sr) in proj_sigmas)
        img_str = "; ".join(
            f"σx={sx:.3f} σy={sy:.3f} σr={sr:.4f}" for (sx, sy, sr) in img_sigmas)
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  "
              f"proj[{proj_str}]  img[{img_str}]", flush=True)

    train_time = time.time() - t0

    pipe.eval()
    with torch.no_grad():
        chunk = cfg.get("val_chunk", 10)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            if per_ps:                      # val_chunk=1 for Mayo -> one ps/slice
                pipe.R_full = _projs[float(_vrk[i])]
            preds.append(pipe(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)

    pred = pred.clamp_min(0.0)
    ld_fbp = ld_fbp.clamp_min(0.0)
    metrics = evaluate_calibrated(
        pred, val_ph, baseline=ld_fbp,
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
    print(f"[solver] Supervised-L2 DD-BF: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} params={params_total}  "
          f"(intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="DD-BF-L2", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
