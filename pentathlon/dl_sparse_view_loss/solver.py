"""DL-Sparse-View CT solver — the file the autoresearch agent edits.

Per the autoresearch contract, **this is the only file the agent should
modify per iteration**. The harness, geometry, projector, simulation, and
metric definitions are *fixed* (see ``ddssl_ldct/`` and
``challenges/dl_sparse_view/README.md``).

Inputs:
    - reads the geometry from the constants below
    - synthesises random-ellipse phantoms as a stand-in for the real
      AAPM DL-Sparse-View breast phantom (real data download is gated;
      see challenges/dl_sparse_view/README.md). The forward problem is
      the same: 2D fan-beam, sparse views, perfectly-known ground truth.

Outputs (written into the directory passed as argv[1]):
    - result.json with keys: val_score, val_psnr, val_ssim, val_rmse,
      headroom, baseline_score, oracle_score, params_M, train_n,
      change_class, rationale, advice_for_others
    - comparison.png — reference / low-view FBP / dual-domain / phantom

Scoring (Choice A, headroom-recovered):
    baseline = sparse-view FBP, no learning
    oracle   = RMSE = 0 against the truth (score = 1.0 by construction)
    score    = 1 - val_rmse / baseline_rmse  ∈ [0, 1]

Anti-overfit rules (see docs/agents.md):
    - total trainable parameters under ~10 × train_n × pixels_per_sample
    - stage_gap > 0 surfaces in the journal; agent must regularise next
    - one change per iteration
    - cite the iteration that inspired the change in --rationale
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure the project package is importable when this script is run from a
# Slurm-allocated working dir.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector
from ddssl_ldct.phantoms import random_ellipses_phantom
from ddssl_ldct.simulate import simulate_low_dose, split_projections
from ddssl_ldct.models import SmallUNet, TrainableBilateralFilter2d
from ddssl_ldct.training import DualDomainPipeline, train
from ddssl_ldct.metrics import psnr, ssim


# ----------------------------------------------------------------------- #
#  Loss bank — agent edits "train_loss" in CONFIG to switch.
# ----------------------------------------------------------------------- #

def _loss_fn(name: str, cfg: dict):
    """Return a function (y_hat, y_tgt) -> scalar loss."""
    if name == "mse":
        return F.mse_loss
    if name == "l1":
        return F.l1_loss
    if name == "charbonnier":
        eps = cfg.get("charbonnier_eps", 1e-3)
        def charbonnier(y_hat, y_tgt):
            return torch.sqrt((y_hat - y_tgt) ** 2 + eps * eps).mean()
        return charbonnier
    if name == "huber":
        delta = cfg.get("huber_delta", 1e-3)
        def huber(y_hat, y_tgt):
            return F.huber_loss(y_hat, y_tgt, delta=delta)
        return huber
    raise ValueError(f"unknown train_loss={name!r}")


def _custom_training_step(pipe: DualDomainPipeline, sino_full: torch.Tensor,
                          loss_fn) -> dict:
    """Mirror DualDomainPipeline.training_step but with a configurable loss."""
    x_a, x_b = split_projections(sino_full)
    y_hat_a = pipe._half_pipeline(x_a)
    with torch.no_grad():
        y_tgt_b = pipe.R_half.fbp(x_b)
    loss_ab = loss_fn(y_hat_a, y_tgt_b)

    y_hat_b = pipe._half_pipeline(x_b)
    with torch.no_grad():
        y_tgt_a = pipe.R_half.fbp(x_a)
    loss_ba = loss_fn(y_hat_b, y_tgt_a)

    loss = 0.5 * (loss_ab + loss_ba)
    return {"loss": loss}


# ----------------------------------------------------------------------- #
#  CONFIG  —  the agent edits this block + the model-builder below.
# ----------------------------------------------------------------------- #

CONFIG = {
    # Geometry (fixed for now — Wagner / Siemens AS @ 128 sparse views).
    "image_size":    512,
    "pixel_spacing": 0.7,
    "n_angles":      128,       # sparse view!
    "n_det":         736,
    "det_spacing":   1.2858,
    "sod":           595.0,
    "sdd":           1085.6,

    # Training subset for the 5-minute iteration budget.
    "train_n":       400,
    "val_n":         100,

    # Editable: training schedule.
    # iter-15 (DISCARD, hr=0.5611): epochs 10 OVERFITS the dual-domain
    # noise target (-1.96pp). 8 is at the sweet spot. DO NOT increase.
    "epochs":        8,
    "batch_size":    1,
    # iter-13 (DISCARD, hr=0.5777): lr=2e-4 near-flat. LR is not the
    # bottleneck in [1e-4, 2e-4]; revert to 1e-4 known baseline.
    "lr":            1e-4,
    "optimizer":     "adamw",   # adam | adamw
    # iter-16: weight_decay 1e-4 -> 1e-3 (10x). iter-15 showed the
    # pipeline OVERFITS the dual-domain noise target with more epochs.
    # Stronger wd specifically combats noise-fitting: penalizes large
    # weights that encode high-frequency noise pattern. AdamW decouples
    # wd from gradient so this is true L2 regularization, not LR-scaled.
    # Cross-agent context: main agent at iter-32 testing wd=0 (opposite
    # direction); we test wd=10x to see which side of the optimum we're
    # on. Free experiment - zero compute cost.
    "weight_decay":  1e-3,

    # Editable: training loss. "mse" (default Wagner-style; pipeline.training_step),
    # "l1", "charbonnier" (smooth L1 with epsilon), or "huber".
    # iter-3 (dl-sparse-view-loss) found Charbonnier (eps=1e-3) drops headroom
    # 0.5831 -> 0.5606 vs Agent B iter-2's MSE baseline. Same L1/MSE trade-off
    # main-agent iter-6 saw: sharper edges, worse pixel-RMSE. Reverted to MSE.
    # Next loss to try: Huber with a *tiny* delta (1e-3 or smaller) so most
    # samples stay in the quadratic regime (preserving RMSE/headroom) and only
    # outliers get L1-treatment. Or: weighted MSE + small TV regulariser.
    "train_loss":    "mse",
    "charbonnier_eps": 1e-3,
    "huber_delta":   1e-3,

    # iter-9 (DISCARD, hr=0.5620 vs 0.5807): per-batch flip aug -1.87pp.
    # All three flip variants (D4 per-epoch, hflip+vflip per-epoch,
    # per-batch flip) fail. Mechanism: flips create OOD orientations the
    # baseline never sees, hurts net's prior on the random-ellipse
    # phantom distribution. Disable.
    "aug_flip":      False,
    "aug_flip_seed": 1234,

    # iter-10 (DISCARD, hr=0.5810 vs 0.5807): MIXUP near-neutral (+0.03pp).
    # Doesn't hurt (unlike flips) but doesn't help. Random-ellipse phantoms
    # already saturate sample diversity; gain elsewhere.
    "aug_mixup":     False,
    "aug_mixup_alpha": 0.4,
    "aug_mixup_seed":  5678,

    # iter-11 (DISCARD, hr=0.5741): SWA over last 2 epochs hurt RMSE
    # (-0.66pp) because the model is still rapidly improving at end-of-
    # training with constant LR; averaging with a less-trained snapshot
    # regresses pixel accuracy. SSIM rose but RMSE dropped → SWA needs
    # LR decay to work here.
    "swa":            False,
    "swa_start_epoch": 7,

    # iter-12 (DISCARD, hr=0.5700): cosine LR 1e-4 -> 1e-6 starved
    # late-training learning rate (-1.07pp). Model is LR-LIMITED at this
    # epoch budget. Revert to constant LR.
    "lr_schedule":     "constant", # "constant" (default) | "cosine"
    "lr_min_ratio":    0.01,


    # Editable: model architecture.
    "unet_c":        16,
    # "unet" | "unet_plus_bf" (Wagner 2022 BF tail) | "resnet"
    #   resnet: plain residual stack at full res (DnCNN / Sidky-2022 top-team
    #           family, cross-ported from spawn agent dl-sparse-view-res
    #           iter-2 which beat the U-Net+BF baseline on headroom).
    "img_denoiser":  "resnet",
    # Editable: residual-stack architecture (only used when img_denoiser="resnet").
    # Default to spawn agent B iter-2 winner (6 blocks, c=32, GroupNorm, ReLU).
    "res_blocks":    6,
    "res_channels":  32,
    "res_norm":      "group",   # group | batch | none
    # iter-14 (DISCARD, hr=0.5729): GELU -0.78pp. Revert to ReLU.
    "res_act":       "relu",    # relu | gelu | swish
    "res_kernel":    3,
    "res_dropout":   0.0,
    "res_residual":  True,      # global residual head (predicts noise)

    # Noise simulation — kept fixed so headroom is comparable across iter.
    "noise_i0":      5e4,
    "noise_sigma_e": 5.0,
    "seed":          42,

    # Intensity calibration. CT images live on a standard scale (HU for
    # real data; for the synthetic ellipse phantoms here the canonical
    # max is 0.05 in attenuation-coefficient units). PSNR and SSIM are
    # computed with a FIXED data_range = display_max - display_min so
    # numbers are comparable across iterations and across agents — auto
    # data-range drifts with FBP overshoot and breaks comparability.
    # Display vmin/vmax are the same fixed range for every column so a
    # comparison image's grey value means the same thing across columns.
    "display_min":   0.0,
    "display_max":   0.05,
}


# ----------------------------------------------------------------------- #
#  Model builder — agent edits HERE for architecture changes.
# ----------------------------------------------------------------------- #

class UNetPlusBF(nn.Module):
    """SmallUNet followed by a Wagner-2022 4-param trainable bilateral
    filter. The BF is interpretable (3 spatial σ + 1 range σ) and acts
    as an edge-preserving sharpening / denoising layer on top of the
    U-Net's coarse denoising."""
    def __init__(self, c: int = 16, bf_kernel: int = 7,
                 bf_sigma_x: float = 1.5, bf_sigma_y: float = 1.5,
                 bf_sigma_r: float = 0.01):
        super().__init__()
        self.unet = SmallUNet(c=c)
        self.bf = TrainableBilateralFilter2d(
            kernel_size=bf_kernel,
            sigma_x=bf_sigma_x, sigma_y=bf_sigma_y, sigma_r=bf_sigma_r)
    def forward(self, x):
        return self.bf(self.unet(x))


# ----------------------------------------------------------------------- #
#  Residual-stack denoiser  (cross-ported from spawn agent
#  dl-sparse-view-res iter-2 winner: 6 blocks @ c=32, GroupNorm + ReLU,
#  global residual head; 0.225 M params total, beats U-Net+BF on headroom).
# ----------------------------------------------------------------------- #

def _pick_groups(c: int, target: int = 8) -> int:
    for g in range(min(c, target), 0, -1):
        if c % g == 0:
            return g
    return 1


def _make_norm(name: str, c: int) -> nn.Module:
    if name == "group":
        return nn.GroupNorm(_pick_groups(c), c)
    if name == "batch":
        return nn.BatchNorm2d(c)
    return nn.Identity()


def _make_act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU(inplace=False)
    return nn.ReLU(inplace=True)


class ResBlock(nn.Module):
    """Standard residual block: conv -> norm -> act -> (drop) -> conv -> norm -> + x."""
    def __init__(self, c: int, kernel: int = 3, norm: str = "group",
                 act: str = "relu", dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n1 = _make_norm(norm, c)
        self.act1 = _make_act(act)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(c, c, kernel, padding=pad)
        self.n2 = _make_norm(norm, c)
    def forward(self, x):
        h = self.conv1(x)
        h = self.n1(h)
        h = self.act1(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.n2(h)
        return x + h


class ResidualStack(nn.Module):
    """Single-resolution residual stack with zero-init tail (identity start).

    Architecture:
        head conv (1 -> c) -> act
        N x ResBlock(c)
        tail conv (c -> 1, zero-init)
        + (optional) global residual: y = x - tail(features)   (predicts noise)
    """
    def __init__(self, n_blocks: int = 6, c: int = 32, kernel: int = 3,
                 norm: str = "group", act: str = "relu",
                 dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.residual = residual
        pad = kernel // 2
        self.head = nn.Conv2d(1, c, kernel, padding=pad)
        self.head_act = _make_act(act)
        self.blocks = nn.Sequential(*[
            ResBlock(c, kernel=kernel, norm=norm, act=act, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.tail = nn.Conv2d(c, 1, kernel, padding=pad)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
    def forward(self, x):
        h = self.head(x)
        h = self.head_act(h)
        h = self.blocks(h)
        y = self.tail(h)
        return x - y if self.residual else y


def build_denoisers(cfg: dict) -> tuple[nn.Module, nn.Module]:
    """Return (proj_denoiser, image_denoiser)."""
    family = cfg.get("img_denoiser", "unet")
    if family == "resnet":
        def make_res():
            return ResidualStack(
                n_blocks=cfg["res_blocks"],
                c=cfg["res_channels"],
                kernel=cfg["res_kernel"],
                norm=cfg["res_norm"],
                act=cfg["res_act"],
                dropout=cfg["res_dropout"],
                residual=cfg["res_residual"],
            )
        # Symmetric residual denoisers on both domains (matches spawn agent B).
        return make_res(), make_res()
    c = cfg["unet_c"]
    proj = SmallUNet(c=c)
    if family == "unet_plus_bf":
        img = UNetPlusBF(c=c)
    else:
        img = SmallUNet(c=c)
    return proj, img


# ----------------------------------------------------------------------- #
#  Data + training harness — leave alone (changes here count as solver
#  changes, but are usually unproductive — prefer model / schedule edits).
# ----------------------------------------------------------------------- #

def build_geometry(cfg: dict) -> FanBeamGeometry:
    return FanBeamGeometry(
        image_size=cfg["image_size"], pixel_spacing=cfg["pixel_spacing"],
        n_angles=cfg["n_angles"], n_det=cfg["n_det"],
        det_spacing=cfg["det_spacing"],
        sod=cfg["sod"], sdd=cfg["sdd"],
    )


def build_dataset(geom, n, seed, i0, sigma_e, device):
    proj = PyronnFanBeamProjector(geom).to(device)
    phantoms = torch.stack([
        random_ellipses_phantom(size=geom.image_size, n_ellipses=10, seed=seed + i)[0]
        for i in range(n)
    ]).to(device)
    with torch.no_grad():
        clean = proj.forward_project(phantoms)
        noisy = simulate_low_dose(clean, i0=i0, sigma_e=sigma_e, seed=seed + 10_000)
    return phantoms, clean, noisy


def make_optimizer(params, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def main(out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[solver] device={device}  config={json.dumps(cfg)}", flush=True)
    torch.manual_seed(cfg["seed"])

    geom = build_geometry(cfg)
    train_ph, train_clean, train_noisy = build_dataset(
        geom, cfg["train_n"], cfg["seed"], cfg["noise_i0"], cfg["noise_sigma_e"], device)
    val_ph, val_clean, val_noisy = build_dataset(
        geom, cfg["val_n"], cfg["seed"] + 1000, cfg["noise_i0"], cfg["noise_sigma_e"], device)

    # Reference (oracle-ish) recon used for PSNR/SSIM logging.
    with torch.no_grad():
        R_full = PyronnFanBeamProjector(geom).to(device)
        val_ref = R_full.fbp(val_clean)
        ld_fbp = R_full.fbp(val_noisy)

    # --- Build the dual-domain pipeline -------------------------------- #
    proj_dn, img_dn = build_denoisers(cfg)
    pipe = DualDomainPipeline(
        geometry=geom, proj_denoiser=proj_dn, image_denoiser=img_dn,
    ).to(device)
    # Replace train's default optimiser with our config.
    params_total = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    print(f"[solver] params = {params_total/1e6:.3f} M", flush=True)

    # Manual training loop (the upstream `train` helper hard-codes Adam;
    # we use our chosen optimiser instead).
    opt = make_optimizer(pipe.parameters(), cfg)
    # Optional cosine LR schedule (iter-12): step once per epoch from
    # lr -> lr * lr_min_ratio with a half-cosine curve. Standard SGDR /
    # cosine annealing (Loshchilov 2017). The natural complement to SWA
    # (iter-11): with cosine decay the final epochs barely move weights,
    # giving the same "flat optimum" effect for free.
    lr_schedule = cfg.get("lr_schedule", "constant")
    if lr_schedule == "cosine":
        eta_min = cfg["lr"] * float(cfg.get("lr_min_ratio", 0.01))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["epochs"], eta_min=eta_min)
        print(f"[solver] LR cosine: {cfg['lr']:.2e} -> {eta_min:.2e} over {cfg['epochs']} epochs",
              flush=True)
    else:
        scheduler = None
    loss_name = cfg.get("train_loss", "mse")
    loss_fn = _loss_fn(loss_name, cfg)
    # iter-10 (dl-sparse-view-loss): MIXUP augmentation. Per-batch,
    # sample a partner phantom index j and a mixing weight λ ~ Beta(α, α),
    # mix at the phantom level (lin combo of attenuation maps), re-project
    # through the SAME geometry, simulate noise. Stays in the random-
    # ellipse distribution (sum of ellipses ≈ another valid phantom).
    # Different mechanism than flip (which was OOD): mixup is a label-
    # smoothing-in-pixel-space regulariser. With α=0.4, λ peaks near 0/1
    # so most batches see "mostly one phantom + a faint trace of another"
    # — gentle regularisation, not heavy mixing.
    aug_mixup = bool(cfg.get("aug_mixup", False))
    aug_alpha = float(cfg.get("aug_mixup_alpha", 0.4))
    # SWA: collect snapshots starting at swa_start_epoch (1-indexed) and
    # average them at end of training. GroupNorm = no BN-stat refresh needed.
    use_swa = bool(cfg.get("swa", False))
    swa_start = int(cfg.get("swa_start_epoch", cfg["epochs"]))
    swa_state = None
    swa_count = 0
    print(f"[solver] train_loss={loss_name}  aug_mixup={aug_mixup} alpha={aug_alpha} "
          f"swa={use_swa} swa_start={swa_start}",
          flush=True)
    aug_rng = torch.Generator(device="cpu").manual_seed(int(cfg.get("aug_mixup_seed", 5678)))
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        pipe.train()
        perm = torch.randperm(train_noisy.shape[0])
        running = 0.0
        n_mix = 0
        n_idn = 0
        for i in range(0, train_noisy.shape[0], cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if aug_mixup:
                # λ ~ Beta(α, α). For α=0.4 the mode is at 0 and 1 →
                # gentle mixing (λ usually small). Pick a random partner
                # index uniformly from the training pool (≠ self).
                lam = float(torch.distributions.Beta(aug_alpha, aug_alpha).sample(
                    sample_shape=()).item())
                # Convert idx -> partner idx (random; different image)
                bs = idx.shape[0]
                j = torch.randint(0, train_ph.shape[0], (bs,), generator=aug_rng)
                # Avoid self-mix if it lands on the same idx
                same = (j == idx)
                if same.any():
                    j[same] = (j[same] + 1) % train_ph.shape[0]
                # Skip very small λ values that are essentially identity
                # (saves compute by reusing cached noisy).
                if abs(lam - 0.0) < 1e-3 or abs(lam - 1.0) < 1e-3:
                    batch = train_noisy[idx].to(device)
                    n_idn += 1
                else:
                    with torch.no_grad():
                        ph_a = train_ph[idx]
                        ph_b = train_ph[j]
                        ph_mix = lam * ph_a + (1.0 - lam) * ph_b
                        clean_b = R_full.forward_project(ph_mix)
                        batch = simulate_low_dose(
                            clean_b,
                            i0=cfg["noise_i0"], sigma_e=cfg["noise_sigma_e"],
                            seed=cfg["seed"] + 30_000 + ep * 100_000 + i,
                        )
                    n_mix += 1
            else:
                batch = train_noisy[idx].to(device)
            if loss_name == "mse":
                losses = pipe.training_step(batch)
            else:
                losses = _custom_training_step(pipe, batch, loss_fn)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            running += float(losses["loss"].detach().cpu())
        mean_loss = running / max(1, train_noisy.shape[0])
        cur_lr = opt.param_groups[0]["lr"]
        if aug_mixup:
            print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}"
                  f"  lr={cur_lr:.2e}  mixup: mix={n_mix} idn={n_idn}", flush=True)
        else:
            print(f"[solver] epoch {ep+1:3d}/{cfg['epochs']}  loss={mean_loss:.5f}"
                  f"  lr={cur_lr:.2e}", flush=True)
        if scheduler is not None:
            scheduler.step()
        # SWA: snapshot at end of each epoch in [swa_start, last_epoch]
        if use_swa and (ep + 1) >= swa_start:
            cur = {k: v.detach().clone() for k, v in pipe.state_dict().items()}
            if swa_state is None:
                swa_state = cur
                swa_count = 1
            else:
                swa_count += 1
                for k in swa_state:
                    if swa_state[k].is_floating_point():
                        swa_state[k] = swa_state[k] + (cur[k] - swa_state[k]) / swa_count
                    else:
                        # buffers like num_batches_tracked: just keep latest
                        swa_state[k] = cur[k]
            print(f"[solver] SWA snapshot {swa_count} taken at end of epoch {ep+1}",
                  flush=True)
    train_time = time.time() - t0
    print(f"[solver] training took {train_time:.1f}s", flush=True)
    if use_swa and swa_state is not None:
        pipe.load_state_dict(swa_state)
        print(f"[solver] SWA: averaged {swa_count} snapshots -> validation",
              flush=True)

    # --- Validation --------------------------------------------------- #
    pipe.eval()
    with torch.no_grad():
        # Chunked validation so a smaller GPU (or a wider BF kernel) does
        # not OOM on val_n images in one batch. Concatenate the per-chunk
        # predictions.
        chunk = cfg.get("val_chunk", 10)
        preds = []
        for i in range(0, val_noisy.shape[0], chunk):
            preds.append(pipe.predict(val_noisy[i:i + chunk]))
        pred = torch.cat(preds, dim=0)
    # Log magnitudes so we can verify projection-count scaling in the journal.
    print(f"[solver] phantom range = [{float(val_ph.min()):.4f}, "
          f"{float(val_ph.max()):.4f}]", flush=True)
    print(f"[solver] val_ref range = [{float(val_ref.min()):.4f}, "
          f"{float(val_ref.max()):.4f}]", flush=True)
    print(f"[solver] ld_fbp  range = [{float(ld_fbp.min()):.4f}, "
          f"{float(ld_fbp.max()):.4f}]", flush=True)
    print(f"[solver] pred    range = [{float(pred.min()):.4f}, "
          f"{float(pred.max()):.4f}]", flush=True)

    # Fixed data_range for PSNR/SSIM (calibration to the canonical phantom
    # max). Without this, auto-data-range = val_ref.max() - val_ref.min()
    # drifts iteration-to-iteration with FBP overshoot and silently
    # changes the PSNR denominator, breaking cross-iter comparability.
    data_range = cfg["display_max"] - cfg["display_min"]
    val_psnr = float(psnr(pred, val_ref, data_range=data_range).cpu())
    val_ssim = float(ssim(pred, val_ref, data_range=data_range).cpu())
    val_rmse = float(((pred - val_ref) ** 2).mean().sqrt().cpu())
    baseline_psnr = float(psnr(ld_fbp, val_ref, data_range=data_range).cpu())
    baseline_rmse = float(((ld_fbp - val_ref) ** 2).mean().sqrt().cpu())
    # Headroom = fraction of the gap between noisy baseline and perfect
    # reconstruction (oracle = 0 RMSE).
    headroom = max(0.0, 1.0 - val_rmse / max(baseline_rmse, 1e-12))
    val_score = val_ssim   # primary single-number score for the run

    # --- Comparison figure ------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_show = min(3, cfg["val_n"])
        fig, ax = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            ax = ax[None]
        # Common display vmin/vmax = the calibration scale (display_min,
        # display_max). All four columns then share the *same* grey
        # mapping — a grey value of, say, 50% means the same physical
        # intensity in every column. Anything outside [display_min,
        # display_max] (FBP overshoot etc.) is clipped at the display, not
        # in the data.
        vmin, vmax = cfg["display_min"], cfg["display_max"]
        for i in range(n_show):
            ax[i, 0].imshow(val_ref[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 0].set_title("reference" if i == 0 else "")
            ax[i, 1].imshow(ld_fbp[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 1].set_title(f"sparse-view FBP  (PSNR={baseline_psnr:.1f})" if i == 0 else "")
            ax[i, 2].imshow(pred[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 2].set_title(f"dual-domain  (PSNR={val_psnr:.1f} SSIM={val_ssim:.3f})" if i == 0 else "")
            ax[i, 3].imshow(val_ph[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i, 3].set_title("phantom" if i == 0 else "")
            for a in ax[i]:
                a.set_axis_off()
        plt.tight_layout()
        figpath = out_dir / "comparison.png"
        plt.savefig(figpath, dpi=120)
        print(f"[solver] saved {figpath}", flush=True)
    except Exception as e:
        print(f"[solver] figure failed: {e}", flush=True)

    result = {
        "val_score":      val_score,
        "val_psnr":       val_psnr,
        "val_ssim":       val_ssim,
        "val_rmse":       val_rmse,
        "baseline_psnr":  baseline_psnr,
        "baseline_rmse":  baseline_rmse,
        "headroom":       headroom,
        "params_M":       params_total / 1e6,
        "train_n":        cfg["train_n"],
        "val_n":          cfg["val_n"],
        "train_time_s":   train_time,
        "config":         cfg,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[solver] result: val_score={val_score:.4f}  headroom={headroom:.4f} "
          f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}",
          flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="directory to write result.json + comparison.png")
    args = p.parse_args()
    main(Path(args.out_dir))
