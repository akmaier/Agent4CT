"""Forward / back projection consistency test for the PYRO-NN wrapper.

Designed to surface any mismatch between forward_project() and back_project()
in our wrapper — orientation flips, rotation-direction mismatches, half-pixel
shifts, scale errors, etc.

Three targeted tests:

  1. Constant-image line integral. Forward-project a (1.0) image and check
     the central ray's value against the analytic line length through the
     image. Catches scaling / unit errors.

  2. Off-centre delta source. Place a delta one pixel right of centre,
     forward-project, then back-project the sinogram (no filter). The
     running max of the unfiltered BP should sit at the same pixel —
     symmetrically smeared. Any rotation-direction / axis-flip bug
     mis-places that maximum.

  3. Shepp-Logan round-trip. FBP a Shepp-Logan, then estimate any sub-pixel
     shift between the FBP and the original phantom via 2D
     cross-correlation. > 0.5-pixel shift is a smoking gun for a
     frame-of-reference mismatch.

Run on a GPU node:
    sbatch cluster/slurm/debug_geometry.sbatch
or interactively:
    srun --gres=gpu:1 --pty bash
    PYTHONPATH=. python scripts/debug_geometry.py
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
import numpy as np

from ddssl_ldct.geometry import FanBeamGeometry
from ddssl_ldct.phantoms import shepp_logan
from ddssl_ldct.pyronn_projector import PyronnFanBeamProjector


def main():
    assert torch.cuda.is_available(), "PYRO-NN requires CUDA."
    out = Path("runs/debug_geometry")
    out.mkdir(parents=True, exist_ok=True)

    # Wagner geometry (the production setting we use)
    geom = FanBeamGeometry()
    print(f"[geom] {geom}")
    R = PyronnFanBeamProjector(geom).to("cuda")

    # ---------------------------------------------------------------- #
    # Test 1: constant-image line integral.
    # ---------------------------------------------------------------- #
    print("\n=== Test 1: constant-image line integral ===")
    img = torch.ones(1, 1, geom.image_size, geom.image_size, device="cuda")
    sino = R.forward_project(img)
    # The longest ray through a square image of side L = (n - 1) * px is at
    # most L*sqrt(2). At θ = 0 the central ray goes straight through the
    # image height = (n - 1) * px = 511 * 0.7 = 357.7 mm.
    expected = (geom.image_size - 1) * geom.pixel_spacing
    centre_det = geom.n_det // 2
    angle0_centre = sino[0, 0, 0, centre_det].item()
    print(f"  expected line length:    {expected:.2f}")
    print(f"  sino[θ=0, u=center]:     {angle0_centre:.2f}")
    print(f"  ratio:                   {angle0_centre / expected:.4f}")
    pct = 100 * (angle0_centre - expected) / expected
    print(f"  deviation:               {pct:+.2f}%  "
          f"({'OK' if abs(pct) < 2 else 'CHECK'})")

    # ---------------------------------------------------------------- #
    # Test 2: off-centre delta, forward + unfiltered BP.
    # ---------------------------------------------------------------- #
    print("\n=== Test 2: off-centre delta, forward + back ===")
    img2 = torch.zeros(1, 1, geom.image_size, geom.image_size, device="cuda")
    # Place the delta 30 pixels right of centre, on the centre row.
    drow, dcol = geom.image_size // 2, geom.image_size // 2 + 30
    img2[0, 0, drow, dcol] = 1.0
    sino2 = R.forward_project(img2)
    bp = R.back_project(sino2)  # unfiltered

    # Find the BP max — should sit at the same (drow, dcol).
    flat = bp[0, 0].cpu().numpy()
    bp_row, bp_col = np.unravel_index(np.argmax(flat), flat.shape)
    drow_diff = bp_row - drow
    dcol_diff = bp_col - dcol
    print(f"  delta at:                row={drow}, col={dcol}")
    print(f"  BP max at:               row={bp_row}, col={bp_col}")
    print(f"  shift:                   Δrow={drow_diff:+d}, Δcol={dcol_diff:+d}  "
          f"({'OK' if abs(drow_diff) + abs(dcol_diff) <= 1 else 'CHECK'})")

    # Sinogram peak-trace as a function of angle should be a sinusoid; check
    # that the peak at θ=0 sits roughly at u-index corresponding to dcol via
    # the fan-beam projection equation:
    #   u = SDD * x_world / (SOD - y_world)
    x_world = (dcol - (geom.image_size - 1) / 2) * geom.pixel_spacing
    y_world = -(drow - (geom.image_size - 1) / 2) * geom.pixel_spacing   # row 0 = +y top
    u_expected = geom.sdd * x_world / (geom.sod - y_world)
    u_idx_expected = u_expected / geom.det_spacing + (geom.n_det - 1) / 2
    # Where is the actual peak in row 0 of the sinogram?
    u_idx_actual = int(sino2[0, 0, 0].argmax().item())
    print(f"  expected u-index @ θ=0:  {u_idx_expected:.2f}")
    print(f"  actual u-index @ θ=0:    {u_idx_actual}")
    err = u_idx_actual - u_idx_expected
    print(f"  detector mis-index:      {err:+.2f}  "
          f"({'OK' if abs(err) < 1 else 'CHECK (potential axis flip)'})")

    # ---------------------------------------------------------------- #
    # Test 3: Shepp-Logan round-trip + cross-correlation shift.
    # ---------------------------------------------------------------- #
    print("\n=== Test 3: Shepp-Logan FBP round-trip ===")
    phantom = shepp_logan(size=geom.image_size).to("cuda")
    sino_p = R.forward_project(phantom)
    reco = R.fbp(sino_p, filter_name="hann")
    print(f"  phantom range: [{phantom.min().item():.4f}, {phantom.max().item():.4f}]")
    print(f"  reco range:    [{reco.min().item():.4f}, {reco.max().item():.4f}]")

    # Sub-pixel shift estimate via FFT cross-correlation.
    p_np = phantom[0, 0].cpu().numpy().astype(np.float32)
    r_np = reco[0, 0].cpu().numpy().astype(np.float32)
    P = np.fft.fft2(p_np)
    R_ = np.fft.fft2(r_np)
    cps = P * np.conj(R_)
    cps /= np.abs(cps) + 1e-12  # phase correlation
    corr = np.fft.ifft2(cps).real
    # Find peak (could be at edge if shift is small)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    # Convert wraparound to signed shift
    sy = peak[0] if peak[0] < corr.shape[0] // 2 else peak[0] - corr.shape[0]
    sx = peak[1] if peak[1] < corr.shape[1] // 2 else peak[1] - corr.shape[1]
    print(f"  phantom→reco shift:      Δrow={sy:+d}, Δcol={sx:+d}  "
          f"({'OK' if abs(sy) + abs(sx) <= 1 else 'CHECK (frame-of-reference mismatch)'})")

    # ---------------------------------------------------------------- #
    # Save diagnostic figure
    # ---------------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, figsize=(14, 7))
        ax[0, 0].imshow(img[0, 0].cpu(), cmap="gray"); ax[0, 0].set_title("T1: const image")
        ax[0, 1].imshow(sino[0, 0].cpu(), cmap="gray", aspect="auto")
        ax[0, 1].set_title(f"T1: sino  (max={sino.max():.1f})")
        ax[0, 2].imshow(img2[0, 0].cpu(), cmap="gray"); ax[0, 2].set_title(f"T2: delta @ ({drow},{dcol})")
        ax[0, 3].imshow(bp[0, 0].cpu(), cmap="gray")
        ax[0, 3].set_title(f"T2: BP max @ ({bp_row},{bp_col})")
        ax[1, 0].imshow(phantom[0, 0].cpu(), cmap="gray"); ax[1, 0].set_title("T3: phantom")
        ax[1, 1].imshow(sino_p[0, 0].cpu(), cmap="gray", aspect="auto"); ax[1, 1].set_title("T3: sino")
        ax[1, 2].imshow(reco[0, 0].cpu(), cmap="gray",
                        vmin=float(phantom.min()), vmax=float(phantom.max()))
        ax[1, 2].set_title("T3: FBP")
        # Difference plot
        diff = (reco - phantom)[0, 0].cpu()
        vmax = float(diff.abs().max())
        ax[1, 3].imshow(diff, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax[1, 3].set_title(f"T3: reco − phantom  (±{vmax:.4f})")
        for a in ax.flat:
            a.axis("off")
        plt.tight_layout()
        figpath = out / "debug_geometry.png"
        plt.savefig(figpath, dpi=130)
        print(f"\nsaved {figpath}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
