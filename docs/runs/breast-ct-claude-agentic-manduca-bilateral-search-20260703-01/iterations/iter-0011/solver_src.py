"""Reference: **TRAINABLE projection-space bilateral denoising with a CT
noise model** — Manduca et al., *Projection space denoising with bilateral
filtering and CT noise modeling for dose reduction in CT*, Med. Phys. 36(11)
2009 — built on top of Wagner's trainable dual-domain bilateral filter
(`solver_dual_ddomain_bilateral_supervised.py`).

Why this solver exists (the Manduca idea)
------------------------------------------
A plain bilateral filter with a *fixed* range bandwidth is NOT noise-adaptive
in the log-sinogram (line-integral) domain: CT noise is signal-dependent —
the variance of a log line-integral ``P`` scales like ``exp(P)/N0`` (few photons
through dense paths → huge variance; many photons through air → tiny variance).
A single range σ therefore over-smooths thin/low-attenuation rays and
under-smooths thick/high-attenuation ones.

Manduca's fix (paper §II.B–E) is a **variance-stabilizing (Anscombe-like)
transform**: take the projection back to photon counts ``N = N0·exp(−P)`` and
apply the square-root transform ``Q = sqrt(N)``. Under Poisson statistics the
sqrt transform makes the noise std approximately *constant* (≈0.5) regardless
of the local count level. A bilateral filter with a single *fixed* range σ run
on ``Q`` is then **implicitly noise-adaptive** in the original domain — this is
the paper's key result. After filtering we invert: ``N̂ = Q̂²`` and
``P̂ = −ln(N̂/N0)`` to recover a denoised log-sinogram, then FBP.

Pipeline (this solver)
----------------------
1.  Input LD log-sinogram ``P`` (line integrals; the dataset's real sino).
2.  ``N = N0·exp(−P)``      — photon counts (N0 = incident blank-scan flux,
    optionally a trainable per-detector "bowtie" profile).
3.  ``Q = sqrt(N)``         — Anscombe/sqrt variance stabilization.
4.  Bilateral-filter ``Q`` in the sinogram domain (one or more chained
    ``TrainableBilateralFilter2d``; paper locks spatial d/w≈1/6, kernel w≈5,
    range σ∈[0.7,2.8] in sqrt-count units).
5.  ``N̂ = Q̂²``  ;  ``P̂ = −ln(max(N̂,ε)/N0)``  — denoised log-sinogram.
6.  FBP(``P̂``) → image (per-sample-ps projector for Mayo, like the base).
7.  Optional light trainable **image-domain** bilateral tail (like DD-BF).

Trainable end-to-end (supervised L2 vs clean truth images), EXACTLY like the
base supervised DD-BF solver: the proj-bilateral bandwidths (σ stack), the
optional image-bilateral, the scalar ``N0`` (and the per-detector bowtie if
enabled) are all learnable. Everything non-algorithmic (data loading, geometry/
per-sample projector, FBP, supervised loop, ``evaluate_calibrated`` metric, the
return dict, comparison.png) mirrors
`solver_dual_ddomain_bilateral_supervised.py` byte-for-byte; ONLY the
projection denoiser core is swapped for the Manduca noise-modeled bilateral.

Config schema (agentic-search knobs)
------------------------------------
Projection-domain Manduca bilateral:
  proj_n_bf      (int)   stacked proj-bilateral filters on Q. **key lever**. default 1
  proj_sr        (float) range bandwidth σ on the sqrt-count image Q (paper
                         0.7–2.8). **key lever** — too small = no denoise, too
                         big = streaks. default 1.5
  proj_kernel    (int)   bilateral spatial kernel size w (odd; paper ~5). default 5
                         (overridden each build from proj_w when proj_w is set)
  proj_w         (int)   alias for the spatial kernel width (paper "w"=5). When
                         set, it drives proj_kernel. default 5
  bf_d_w_ratio   (float) paper's locked spatial-σ / kernel-width ratio (d/w≈1/6):
                         proj spatial σ = bf_d_w_ratio * proj_w. **key lever**.
                         default 1/6
manduca_N0     (float) incident blank-scan flux N0 for N=N0·exp(−P) and the
                       inverse. **key lever**. default = dataset i0 (cfg["noise_i0"]).
manduca_bowtie (bool)  if True, a TRAINABLE per-detector log-N0 profile is added
                       to log(N0) (the "bowtie"/blank-scan non-uniformity), so the
                       noise model is per-ray. default False (scalar N0).
Optional image-domain bilateral tail (like DD-BF):
  img_n_bf       (int)   stacked image-bilateral filters (0 disables the tail).
                         default 1
  img_kernel     (int)   image bilateral kernel size (odd). default 5
  img_sx,img_sy  (float) image bilateral spatial σ. default 0.5, 0.5
  img_sr         (float) image bilateral range σ (image-intensity units). default 0.02
Training (mirrors the base):
  train_n        (int)   #train slices. default from DEMO_DL_DEFAULTS (Mayo: stratified)
  val_n          (int)   #val slices (Mayo default = all 214 L277). inherited
  epochs         (int)   supervised epochs. default 10 (budget-safe)
  batch_size     (int)   default 1 (required for Mayo per-sample-ps)
  lr             (float) Adam lr (Wagner BF default 5e-3). default 5e-3
  lambda_neg     (float) non-negativity penalty weight on the image output. default 1.0
  grad_clip      (float) grad-norm clip (Mayo 2304-view FBP adjoint needs >0). default 1.0
  seed           (int)   default from DEMO_DL_DEFAULTS

Key agentic levers (tune these first): proj_sr, proj_n_bf, bf_d_w_ratio,
manduca_N0, then img_n_bf / img_sr for the image-domain polish.

Self-supervised? No — supervised L2 vs clean truth (paper validates against a
high-dose reference; here the staged Mayo truth). Mirrors the base supervised
DD-BF solver; for an unsupervised N2I variant see
`solver_dual_ddomain_bilateral_n2i.py`.
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
from ddssl_ldct.models import TrainableBilateralFilter2d
from ddssl_ldct.metrics import (
    evaluate_calibrated, make_4panel_comparison,
    supervised_recon_loss, clip_and_step,
)
from challenges.demo_dl.geometry import DEFAULTS as DEMO_DL_DEFAULTS


CONFIG = {
    **DEMO_DL_DEFAULTS,
    # --- Projection-domain Manduca noise-modeled bilateral (the core) ------
    # The bilateral runs on Q = sqrt(N0·exp(-P)) (sqrt photon counts), where the
    # Anscombe transform makes the noise std ~constant so a single range σ is
    # implicitly noise-adaptive in the line-integral domain (paper §II.B-E).
    "proj_n_bf":      1,        # stacked proj-bilateral filters (key lever)
    "proj_sr":        1.5,      # range σ on Q, in sqrt-count units (paper 0.7-2.8; key lever)
    "proj_w":         5,        # spatial kernel width w (paper ~5)
    "proj_kernel":    5,        # spatial kernel size (set from proj_w each build)
    "bf_d_w_ratio":   1.0 / 6.0,  # paper d/w≈1/6 -> proj spatial σ = ratio*w (key lever)
    # --- CT noise model knobs ---------------------------------------------
    "manduca_N0":     None,     # incident flux N0; None -> default to cfg["noise_i0"]
    "manduca_bowtie": False,    # trainable per-detector log-N0 profile (bowtie)
    # --- Optional image-domain bilateral tail (like DD-BF) ----------------
    "img_n_bf":       1,        # 0 disables the image-domain bilateral tail
    "img_kernel":     5,
    "img_sx":         0.5,
    "img_sy":         0.5,
    "img_sr":         0.02,
    # --- Training (mirrors the base supervised DD-BF) ----------------------
    "epochs":         10,
    "batch_size":     1,
    "lr":             5e-3,     # Wagner's BF lr (vs 5e-5 for U-Nets)
    "optimizer":     "adam",
    "lambda_neg":     1.0,      # non-negativity penalty on image-domain output
    "grad_clip":      1.0,      # >0 for Mayo's 2304-view FBP adjoint
}


class BilateralFilterStack(nn.Module):
    """N independent TrainableBilateralFilter2d layers in series.

    Per Wagner 2022 §3.2: cascading bilateral filters increases the
    effective receptive field while keeping the parameter count tiny
    (3 trainable params per BF). All N filters share kernel size but
    are initialised from the same σ values and learn independently.

    n_filters == 0 -> identity (used to disable the optional image tail).
    """

    def __init__(self, n_filters: int, kernel_size: int,
                 sigma_x: float, sigma_y: float, sigma_r: float):
        super().__init__()
        assert n_filters >= 0
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


class ManducaProjDenoiser(nn.Module):
    """Projection-space bilateral denoising WITH the CT noise model (Manduca 2009).

    Forward (on a log-sinogram / line-integral tensor P, shape (B,1,A,D)):
        N  = N0 · exp(-P)            photon counts  (N0 = blank-scan flux)
        Q  = sqrt(N)                 Anscombe/sqrt variance stabilization
        Q̂  = bilateral_stack(Q)      single fixed range σ -> implicitly noise-adaptive
        N̂  = Q̂²
        P̂  = -ln(max(N̂, ε) / N0)     denoised log-sinogram

    Trainable parameters:
      - the bilateral stack's (σx, σy, σr) per filter (3 each), AND
      - log_N0 (scalar) so the incident flux is learnable, AND
      - (optional) a per-detector log-N0 profile (the "bowtie") when
        ``bowtie`` is True — added to log_N0 broadcast over detector columns.

    N0 is parameterised in log-space (log_N0) so it stays strictly positive
    under gradient descent and the inverse -ln(N̂/N0) is well defined.
    """

    def __init__(self, n_bf: int, kernel_size: int, sigma_x: float,
                 sigma_y: float, sigma_r: float, N0: float,
                 n_det: int, bowtie: bool = False, eps: float = 1.0):
        super().__init__()
        self.stack = BilateralFilterStack(
            n_filters=max(1, int(n_bf)), kernel_size=kernel_size,
            sigma_x=sigma_x, sigma_y=sigma_y, sigma_r=sigma_r)
        self.log_N0 = nn.Parameter(torch.tensor(math.log(max(N0, 1.0))))
        self.bowtie = bool(bowtie)
        if self.bowtie:
            # Per-detector additive log-N0 offset (blank-scan non-uniformity).
            # Init zero -> identical to the scalar-N0 model at step 0.
            self.bowtie_logN0 = nn.Parameter(torch.zeros(1, 1, 1, int(n_det)))
        self.eps = float(eps)

    def _logN0_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-N0 broadcastable to x (B,1,A,D). Scalar, or per-detector
        when the trainable bowtie is enabled."""
        lN0 = self.log_N0
        if self.bowtie:
            # (1,1,1,D) broadcasts over batch + angle. Guard a detector-count
            # mismatch (e.g. transposed sino) by falling back to the scalar.
            if x.shape[-1] == self.bowtie_logN0.shape[-1]:
                lN0 = lN0 + self.bowtie_logN0
        return lN0

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        logN0 = self._logN0_map(P)                 # scalar or (1,1,1,D)
        # N = N0 * exp(-P) = exp(logN0 - P); clamp the exponent for fp stability
        # (very negative P -> huge N; the staged log-sinos are O(1-10) so this
        # clamp never bites on real data, it only guards pathological inputs).
        N = torch.exp((logN0 - P).clamp(max=30.0))
        Q = torch.sqrt(N.clamp_min(0.0) + 1e-8)    # Anscombe/sqrt stabilization
        Qhat = self.stack(Q)
        Nhat = Qhat.clamp_min(0.0) ** 2
        Phat = logN0 - torch.log(Nhat.clamp_min(self.eps))   # -ln(N̂/N0)
        return Phat

    @torch.no_grad()
    def sigmas(self) -> list[tuple[float, float, float]]:
        return self.stack.sigmas()

    @torch.no_grad()
    def N0_value(self) -> float:
        return float(torch.exp(self.log_N0).cpu())


def build_dataset(geom, n, seed, i0, sigma_e, device):
    from ddssl_ldct.staged_dataset import load_val_split
    import os
    kind = os.environ.get("AGENT4CT_DATASET", "phantoms")
    split = "val" if (seed % 100_000) >= 1000 else "train"
    return load_val_split(kind, split, n, device=device,
                          seed=seed, noise_i0=i0, noise_sigma_e=sigma_e,
                          geom=geom, return_ps=True)   # 4-tuple; ps=None for non-mayo


class ManducaBilateralPipeline(nn.Module):
    """Single-pass Manduca DD-BF: proj_dn(noise-modeled) -> FBP -> img_dn.

    ``proj_dn`` is a ``ManducaProjDenoiser`` (CT-noise-modeled bilateral on the
    sqrt-count image). ``img_dn`` is a ``BilateralFilterStack`` (possibly empty,
    i.e. identity) for the optional image-domain polish. ``R_full`` is swapped
    per sample for the Mayo per-ps geometry (mirrors the base).
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

    # Manduca proj denoiser knobs: spatial kernel width + the paper's d/w ratio.
    proj_kernel = int(cfg.get("proj_w", cfg["proj_kernel"]))
    if proj_kernel % 2 == 0:                     # bilateral needs an odd kernel
        proj_kernel += 1
    proj_spatial_sigma = max(1e-3, float(cfg["bf_d_w_ratio"]) * float(proj_kernel))
    N0 = cfg.get("manduca_N0")
    if N0 is None:
        N0 = cfg["noise_i0"]                     # default incident flux = dataset i0
    N0 = float(N0)

    proj_dn = ManducaProjDenoiser(
        n_bf=cfg["proj_n_bf"], kernel_size=proj_kernel,
        sigma_x=proj_spatial_sigma, sigma_y=proj_spatial_sigma,
        sigma_r=cfg["proj_sr"], N0=N0, n_det=cfg["n_det"],
        bowtie=bool(cfg.get("manduca_bowtie", False)),
    )
    img_dn = BilateralFilterStack(
        n_filters=int(cfg["img_n_bf"]), kernel_size=cfg["img_kernel"],
        sigma_x=cfg["img_sx"], sigma_y=cfg["img_sy"], sigma_r=cfg["img_sr"],
    )
    pipe = ManducaBilateralPipeline(geom, proj_dn, img_dn).to(device)

    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] Manduca proj-bilateral (supervised L2): proj_n_bf={cfg['proj_n_bf']} "
          f"img_n_bf={cfg['img_n_bf']} N0_init={N0:.3g} bowtie={bool(cfg.get('manduca_bowtie', False))} "
          f"proj_kernel={proj_kernel} proj_spatial_sigma={proj_spatial_sigma:.4f} "
          f"proj_sr={cfg['proj_sr']} params_total={params_total} "
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
        proj_str = "; ".join(
            f"σx={sx:.3f} σy={sy:.3f} σr={sr:.4f}" for (sx, sy, sr) in proj_sigmas)
        img_str = ("; ".join(
            f"σx={sx:.3f} σy={sy:.3f} σr={sr:.4f}" for (sx, sy, sr) in img_sigmas)
            or "(identity)")
        print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}  "
              f"N0={proj_dn.N0_value():.4g}  proj[{proj_str}]  img[{img_str}]", flush=True)

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
        "params_M": params_total / 1e6, "params_total": params_total, "train_n": cfg["train_n"],
        "val_n": cfg["val_n"], "train_time_s": train_time,
        "N0_learned": proj_dn.N0_value(),
        "config": cfg,
        "training_scheme": "manduca_projection_bilateral_supervised_l2",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] Manduca proj-BF: val_score={val_score:.4f} headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} RMSE={val_rmse:.5f} "
          f"baseline_PSNR={baseline_psnr:.2f} params={params_total}  "
          f"N0={proj_dn.N0_value():.4g}  (intensity-calibrated)", flush=True)

    try:
        make_4panel_comparison(
            truth=val_ph, fbp=baseline_cal, recon=pred_cal,
            out_path=out_dir / "comparison.png",
            display_min=cfg["display_min"], display_max=cfg["display_max"],
            n_show=4, solver_label="Manduca-BF", headroom=headroom)
    except Exception as e:
        print(f"[solver] comparison.png failed: {e}", flush=True)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    args = p.parse_args()
    main(Path(args.out_dir))
