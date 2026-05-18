"""Unconditional DDPM sanity sampling.

Loads a checkpoint produced by solver_ddpm.py and draws N samples via
plain DDIM (no DC guidance). The output PNG decides whether the DDPM
itself is the bug for the failed diffusion_recon search:

    - samples look like ellipse phantoms → DDPM is fine, the guidance
      gradient is what breaks recon (need eta tuning / sampler fix)
    - samples look like noise / uniform fields → DDPM is under-trained
      (need longer training, larger model, or fix the noise schedule).

Usage:
    python scripts/ddpm_sanity_sample.py <ckpt_path> <out_dir> [n_steps]
"""
from __future__ import annotations
import sys, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from pentathlon.demo_dl_reference.solver_ddpm import SmallDDPM, NoiseSchedule


def x0_from_eps(xt, eps, ab):
    return (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-6)


def main():
    ckpt_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    n_samples = 6
    image_size = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device)
    print(f"[sanity] loaded {ckpt_path}: mode={state.get('ddpm_mode')} "
          f"ch={state['ddpm_ch']} T={state['ddpm_n_steps']} "
          f"final_val_loss={state.get('final_val_loss')}", flush=True)
    sched = NoiseSchedule(T=state["ddpm_n_steps"], device=device)
    model = SmallDDPM(ch=state["ddpm_ch"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(n_samples, 1, image_size, image_size, device=device)
    T = sched.T
    times = torch.linspace(T, 1, n_steps + 1).long().tolist()
    print(f"[sanity] {n_steps}-step DDIM sampling…", flush=True)
    with torch.no_grad():
        for k in range(n_steps):
            t_now = times[k]; t_next = times[k + 1] if k + 1 < len(times) else 0
            t_tensor = torch.tensor([t_now] * n_samples, device=device)
            eps = model(x, t_tensor)
            ab_now = sched.alpha_bar[t_now]
            ab_next = sched.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0,
                                                                               device=device)
            x0_hat = x0_from_eps(x, eps, ab_now).clamp(0.0, 1.0)
            x = ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps
            if k % 20 == 0:
                print(f"  step {k}/{n_steps} x.mean={x.mean():.3f} "
                      f"x.std={x.std():.3f} x.min={x.min():.3f} x.max={x.max():.3f}",
                      flush=True)
    out_scale = state.get("ddpm_out_scale", 0.05)
    samples = (x * out_scale).clamp(0.0, out_scale).cpu()

    # Stats
    flat = samples.flatten()
    print(f"[sanity] FINAL samples: mean={flat.mean():.4f} std={flat.std():.4f} "
          f"min={flat.min():.4f} max={flat.max():.4f}", flush=True)
    n_constant = sum(float(s.std()) < 1e-4 for s in samples)
    print(f"[sanity] {n_constant}/{n_samples} samples are essentially constant",
          flush=True)

    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, n_samples, figsize=(2 * n_samples, 2.5))
    for i in range(n_samples):
        ax[i].imshow(samples[i, 0], cmap="gray", vmin=0.0, vmax=out_scale)
        ax[i].set_title(f"#{i+1}", fontsize=9); ax[i].set_axis_off()
    plt.suptitle(f"{ckpt_path.name}  (n_steps={n_steps})", fontsize=10)
    plt.tight_layout()
    out_png = out_dir / "ddpm_sanity_samples.png"
    plt.savefig(out_png, dpi=120)
    print(f"[sanity] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
