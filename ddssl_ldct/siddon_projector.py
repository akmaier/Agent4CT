"""Siddon line-intersection fan-beam projector with exact-transpose back-projector.

Implements the forward operator from the Sidky 2022 AAPM DL-Sparse-View CT
challenge (literature/sidky_2022_dl_sparse_view_2109.09640.md, §II.B):

    g = X · f          # discrete-to-discrete linear model
    X[r, p] = length of ray r through pixel p   (Siddon 1985)

and the matched (= "ray-driven", = matrix transpose) back-projector:

    f_recon = X^T · g
    (X^T · g)[p] = Σ_r X[r, p] · g[r]

Sidky's released `val_fbp128.h5` is generated as

    fbp = X^T · (ramp · g)             (paper §II.B, line 150)

so an FBP through this projector should reproduce it to high precision.

This matters because:

  - pyronn uses a *Joseph-flavour* forward projector and a *pixel-driven*
    back-projector (interpolated trapezoid rule), which is not the
    transpose of its forward operator. Per Sidky §II.B that mismatch moves
    the recovery problem from category (i) (exact knowledge of X) to (iii)
    (approximate) and produces the boundary blue ring + interior haze
    visible in the breast diff images.

  - We need a *matched* pair both to reproduce Sidky's FBP128 reference and
    to plug into iterative / unrolled solvers that require X^T-back-proj.

Implementation: pure-PyTorch batched along rays. Runs on CPU and CUDA.
For 128×1024 rays through a 512×512 image, ~0.5-1 GB GPU peak memory and
a few hundred milliseconds per call on a Quadro RTX 8000. No external
dependencies beyond torch / numpy.

Geometry convention (matches pyronn's `circular_trajectory_2d(..., True)`):
  - Image centred at (0, 0). image[i_row, i_col] = pixel at
    (x, y) = ((i_col - (N-1)/2) · Δ, ((N-1)/2 - i_row) · Δ)   (y-up).
  - View β = 0:
      source     S = (0, +sod)
      det centre C = (0, -(sdd-sod))
      det u-axis u = (+1, 0)
  - View β > 0: rotate source / detector counter-clockwise about iso.

UNIT CONSISTENCY (this *is* a thing — Sidky's breast_ct truth is in 1/cm,
but our :class:`FanBeamGeometry` is in mm):

  - Forward gives ``g[r] = Σ_p μ[p] · X[r,p]``. The pixel value μ[p] has units
    1/length₁, and Siddon's path length X[r,p] has units length₂.
    `g` is *only* a dimensionless line integral when length₁ ≡ length₂.
  - The projector internally treats every position / spacing as a single
    number with no unit. If the geometry uses mm and the image's attenuation
    is in 1/cm, the forward sino is 10× too big and the FBP recon is off
    by a fixed factor that intensity-calibration will hide. To fix this at
    the source, pass ``length_unit_scale`` so the geometry is rescaled
    internally. For breast_ct (geometry in mm, μ in 1/cm) use 0.1; the
    forward then matches Sidky's released sinograms in absolute units.

FBP normalisation: ``fbp(g) = X^T · ramp(g) · (Δβ / 2)`` for a 2π scan.
The ``Δβ`` discretises the angle integral and the ``1/2`` is the
full-scan redundancy weight (each ray sampled twice). Short scans need
Parker weights applied externally; we just use ``Δβ`` with no /2 in that
case. With ``length_unit_scale=0.1`` this reproduces Sidky's
``val_fbp128.h5`` to high precision — see `scripts/debug_breast_ct_siddon.py`.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from .geometry import FanBeamGeometry


class SiddonFanBeamProjector(nn.Module):
    """Siddon line-intersection 2D fan-beam projector + matched X^T back-projector.

    Same input/output shape conventions as :class:`PyronnFanBeamProjector`:

      forward_project: ``(H, W)`` → ``(A, N_det)``
                       ``(B, 1, H, W)`` → ``(B, 1, A, N_det)``
      back_project:    inverse shapes.
      fbp(sino):       ramp filter (unapodized Kak-Slaney ram-lak)  followed
                       by ``back_project``  — matches Sidky's FBP128 recipe.

    Args:
        geom: :class:`FanBeamGeometry`. Reads image_size / pixel_spacing /
              n_angles / n_det / det_spacing / sod / sdd / angle_start /
              angle_end. The angular sweep is uniform over
              ``[angle_start, angle_end)`` with ``n_angles`` samples (final
              endpoint is *excluded* — matches `circular_trajectory_2d`).
        ray_batch: chunk size along the (view × det) ray axis. Trade-off
                   between GPU memory peak and Python overhead. 4096 is a
                   good default for 1024-channel sinograms on 16 GB GPUs.
        length_unit_scale: multiplier applied to every distance in
                   ``geom`` (sod, sdd, pixel_spacing, det_spacing) before
                   caching. Use to convert geometry units to match the
                   image attenuation's inverse unit. For breast_ct (μ in
                   1/cm, FanBeamGeometry in mm) pass ``0.1``. Default 1.0.
    """

    def __init__(self, geom: FanBeamGeometry, *,
                 ray_batch: int = 4096,
                 length_unit_scale: float = 1.0):
        super().__init__()
        self.geom = geom
        self.ray_batch = int(ray_batch)
        self.length_unit_scale = float(length_unit_scale)

        # Pre-compute per-ray source and detector positions, in *scaled*
        # geometry units. All (A · N_det) rays are unrolled into a single
        # (R,) axis. ``length_unit_scale`` rescales every distance so the
        # projector operates in consistent units with the image μ.
        A, N_det = geom.n_angles, geom.n_det
        s = self.length_unit_scale
        sod, sdd = float(geom.sod) * s, float(geom.sdd) * s
        d_spacing = float(geom.det_spacing) * s
        # Also expose the scaled pixel spacing for use in the Siddon kernel.
        self._pixel_spacing_scaled = float(geom.pixel_spacing) * s
        self._det_spacing_scaled = d_spacing

        # Uniform sweep over [angle_start, angle_end), N samples.
        betas = torch.linspace(
            float(geom.angle_start), float(geom.angle_end), A + 1, dtype=torch.float64
        )[:-1]                                                       # (A,)
        sin_b, cos_b = torch.sin(betas), torch.cos(betas)

        # View β=0 → source on +y, detector centre on -y; rotate CCW with β.
        #   S(β) = (-sin β · sod,  +cos β · sod)
        #   C(β) = (+sin β · (sdd-sod), -cos β · (sdd-sod))
        #   u(β) = (-cos β, -sin β)        ← det channel 0 on +x side at β=0
        # This det-axis sign matches Sidky's released sinogram channel
        # convention (per user, conv-sweep variant det=0).
        sx = -sin_b * sod
        sy =  cos_b * sod
        cx =  sin_b * (sdd - sod)
        cy = -cos_b * (sdd - sod)
        ux = -cos_b
        uy = -sin_b
        d_offsets = (torch.arange(N_det, dtype=torch.float64) - (N_det - 1) / 2.0) * d_spacing  # (N_det,)
        # Per-ray detector position: dx[a, d] = cx[a] + d_offsets[d] · ux[a]
        dx_full = cx[:, None] + d_offsets[None, :] * ux[:, None]     # (A, N_det)
        dy_full = cy[:, None] + d_offsets[None, :] * uy[:, None]
        sx_full = sx[:, None].expand_as(dx_full)
        sy_full = sy[:, None].expand_as(dy_full)
        # Flatten to (R,) with R = A * N_det.
        for name, t in (("sx", sx_full), ("sy", sy_full),
                        ("dx", dx_full), ("dy", dy_full)):
            self.register_buffer(f"_{name}", t.reshape(-1).contiguous().to(torch.float32),
                                 persistent=False)

        # Pre-compute the ramp-filter spectrum (Kak-Slaney) so .fbp() doesn't
        # rebuild it each call. We rebuild lazily on the right device / dtype.
        self._ramlak_cache: dict[tuple, torch.Tensor] = {}

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_image(image: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Accept (H,W) / (1,H,W) / (B,1,H,W). Return (B, H, W), original ndim."""
        nd = image.dim()
        if nd == 2:
            return image[None].contiguous(), nd
        if nd == 3:
            assert image.shape[0] == 1, "(C,H,W) only supports C=1"
            return image.contiguous(), nd
        if nd == 4:
            assert image.shape[1] == 1, "(B,C,H,W) only supports C=1"
            return image[:, 0].contiguous(), nd
        raise ValueError(f"image must have 2, 3, or 4 dims; got {nd}")

    @staticmethod
    def _restore_image_ndim(out: torch.Tensor, original_ndim: int) -> torch.Tensor:
        # out is (B, A, D) for sinogram or (B, H, W) for image.
        if original_ndim == 4:
            return out[:, None]
        if original_ndim == 3:
            return out[0:1]
        return out[0]

    @staticmethod
    def _normalize_sino(sino: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Accept (A,D) / (1,A,D) / (B,1,A,D). Return (B, A, D), original ndim."""
        nd = sino.dim()
        if nd == 2:
            return sino[None].contiguous(), nd
        if nd == 3:
            assert sino.shape[0] == 1, "(C,A,D) only supports C=1"
            return sino.contiguous(), nd
        if nd == 4:
            assert sino.shape[1] == 1, "(B,C,A,D) only supports C=1"
            return sino[:, 0].contiguous(), nd
        raise ValueError(f"sino must have 2, 3, or 4 dims; got {nd}")

    # ── Siddon kernel ───────────────────────────────────────────────────────

    def _siddon_segments(self, ray_slice: slice
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For a chunk of rays, return:

            weights : (B, 2N+1)  path-length per pixel-segment (mm)
            i_row   : (B, 2N+1)  pixel row index (int64, clamped to [0, N-1])
            i_col   : (B, 2N+1)  pixel column index

        ``weights`` are *zero* outside valid segments, so the masked
        arithmetic in forward / back-projection collapses to "real" entries
        without an extra boolean indexing step.
        """
        geom = self.geom
        N = int(geom.image_size)
        delta = self._pixel_spacing_scaled
        device = self._sx.device
        dtype = self._sx.dtype

        sx = self._sx[ray_slice]; sy = self._sy[ray_slice]
        dx = self._dx[ray_slice]; dy = self._dy[ray_slice]
        B = sx.shape[0]

        # Direction vector (mm); guard near-axis-aligned rays with eps.
        vx = dx - sx
        vy = dy - sy
        eps = torch.full_like(vx, 1e-9)
        vx_safe = torch.where(torch.abs(vx) < 1e-12, eps, vx)
        vy_safe = torch.where(torch.abs(vy) < 1e-12, eps, vy)

        # Grid-line coordinates of the (N+1) vertical and (N+1) horizontal
        # boundaries of the pixel grid, centred on the iso.
        k = torch.arange(N + 1, device=device, dtype=dtype)
        grid = -N / 2.0 * delta + k * delta                      # (N+1,)

        # α values where the ray crosses each grid line.
        alpha_x = (grid[None, :] - sx[:, None]) / vx_safe[:, None]   # (B, N+1)
        alpha_y = (grid[None, :] - sy[:, None]) / vy_safe[:, None]   # (B, N+1)
        alphas = torch.cat([alpha_x, alpha_y], dim=-1)               # (B, 2N+2)

        # α-range that lies inside the grid extent.
        a_x_min = torch.minimum(alpha_x[:, 0], alpha_x[:, -1])
        a_x_max = torch.maximum(alpha_x[:, 0], alpha_x[:, -1])
        a_y_min = torch.minimum(alpha_y[:, 0], alpha_y[:, -1])
        a_y_max = torch.maximum(alpha_y[:, 0], alpha_y[:, -1])
        a_min = torch.maximum(a_x_min, a_y_min)                       # (B,)
        a_max = torch.minimum(a_x_max, a_y_max)

        alphas_sorted, _ = torch.sort(alphas, dim=-1)                 # (B, 2N+2)

        # Per-segment midpoint and length in α.
        alpha_mid = 0.5 * (alphas_sorted[:, 1:] + alphas_sorted[:, :-1])  # (B, 2N+1)
        d_alpha   = alphas_sorted[:, 1:] - alphas_sorted[:, :-1]          # (B, 2N+1)

        # Valid: midpoint inside the grid clip, and length > 0.
        valid = (alpha_mid >= a_min[:, None]) & (alpha_mid <= a_max[:, None]) & (d_alpha > 0)

        # Cartesian position of the midpoint → pixel index (row, col).
        px = sx[:, None] + alpha_mid * vx[:, None]
        py = sy[:, None] + alpha_mid * vy[:, None]
        #   col i_col ∈ [0, N): DECREASES with px (left-right flipped from the
        #     natural "x grows to the right" convention). col 0 sits at the
        #     +x side of the image; this matches Sidky's released
        #     val_truth / val_fbp128 storage layout (per user inspection).
        #     i_col = floor((N·Δ/2 - px) / Δ)
        #   row i_row ∈ [0, N): y_up=0 convention — image[0, :] is the BOTTOM
        #     of the image (y_bot = −N·Δ/2). Row index increases with py.
        #     i_row = floor((py + N·Δ/2) / Δ)
        i_col = torch.floor(((N * delta) / 2.0 - px) / delta).to(torch.int64)
        i_row = torch.floor((py + (N * delta) / 2.0) / delta).to(torch.int64)
        in_bounds = (i_col >= 0) & (i_col < N) & (i_row >= 0) & (i_row < N)
        valid = valid & in_bounds

        i_col = i_col.clamp(0, N - 1)
        i_row = i_row.clamp(0, N - 1)

        ray_len = torch.sqrt(vx * vx + vy * vy)                       # (B,)
        weights = d_alpha * ray_len[:, None] * valid.to(dtype)        # (B, 2N+1)
        return weights, i_row, i_col

    # ── public API ──────────────────────────────────────────────────────────

    def forward_project(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, 1, H, W) → (B, 1, A, N_det)``, Siddon line integrals."""
        imgs, nd = self._normalize_image(image)
        B_img = imgs.shape[0]
        device = imgs.device
        if imgs.dtype != torch.float32:
            imgs = imgs.float()
        N = self.geom.image_size
        A = self.geom.n_angles
        N_det = self.geom.n_det
        R = A * N_det
        sino_flat = torch.zeros(B_img, R, device=device, dtype=imgs.dtype)
        imgs_flat = imgs.reshape(B_img, N * N)                        # (B, N²)
        for start in range(0, R, self.ray_batch):
            end = min(start + self.ray_batch, R)
            sl = slice(start, end)
            weights, i_row, i_col = self._siddon_segments(sl)         # (b, M)
            pixel_idx = (i_row * N + i_col).reshape(-1)               # (b·M,)
            # Gather pixel values per ray segment: (B, b·M)
            vals = imgs_flat.index_select(1, pixel_idx)
            vals = vals.view(B_img, weights.shape[0], weights.shape[1])
            sino_flat[:, sl] = (vals * weights[None]).sum(dim=-1)
        sino = sino_flat.view(B_img, A, N_det)
        return self._restore_image_ndim(sino, nd)

    def back_project(self, sino: torch.Tensor) -> torch.Tensor:
        """``(B, 1, A, N_det) → (B, 1, H, W)`` via exact transpose of forward.

        This is X^T — i.e. for every ray r and every pixel p it crosses,
        accumulate ``X[r, p] · g[r]`` into ``image[p]``. The same path lengths
        are used as in :meth:`forward_project`.
        """
        s, nd = self._normalize_sino(sino)
        B_sin = s.shape[0]
        device = s.device
        if s.dtype != torch.float32:
            s = s.float()
        N = self.geom.image_size
        A = self.geom.n_angles
        N_det = self.geom.n_det
        R = A * N_det
        s_flat = s.reshape(B_sin, R).contiguous()
        recon_flat = torch.zeros(B_sin, N * N, device=device, dtype=s.dtype)
        for start in range(0, R, self.ray_batch):
            end = min(start + self.ray_batch, R)
            sl = slice(start, end)
            weights, i_row, i_col = self._siddon_segments(sl)         # (b, M)
            b, M = weights.shape
            pixel_idx = (i_row * N + i_col).reshape(-1)               # (b·M,)
            s_chunk = s_flat[:, sl]                                   # (B, b)
            contrib = s_chunk[:, :, None] * weights[None]             # (B, b, M)
            recon_flat.index_add_(1, pixel_idx, contrib.view(B_sin, b * M))
        recon = recon_flat.view(B_sin, N, N)
        return self._restore_image_ndim(recon, nd)

    # ── ramp filter (Kak-Slaney) + FBP via X^T ──────────────────────────────

    def _ramlak(self, M: int, device, dtype) -> torch.Tensor:
        """Kak-Slaney spatial-domain ram-lak impulse, FFT'd to frequency domain,
        for an array of length ``M``. Caller picks ``M`` — typically ``M = 2·N_det``
        to enable proper linear (zero-padded) convolution.

        The analytic ramp filter has ``H(ν=0) = |0| = 0``. The discrete DFT of
        the truncated Kak-Slaney spatial impulse leaves a residual
        ``DFT[0] ≈ 2/(π²·M·τ²)`` (= 0.08 for our M=2048, τ=0.03516 cm setup);
        zero-padding M=N→2N halves it but does not kill it. The
        "by the book" discrete ramp has ``H[0]=0``, so we clamp the DC bin
        explicitly after the DFT.
        """
        key = (M, device, dtype)
        if key in self._ramlak_cache:
            return self._ramlak_cache[key]
        d = self._det_spacing_scaled
        h = np.zeros(M, dtype=np.float64)
        h[0] = 0.25 / (d * d)                           # textbook 1/(4τ²)
        odd_v = -1.0 / (math.pi * math.pi * d * d)
        for i in range(1, M):
            if i < M / 2 and (i % 2) == 1:
                h[i] = odd_v / (i * i)
            elif i >= M / 2:
                tmp = M - i
                if (tmp % 2) == 1:
                    h[i] = odd_v / (tmp * tmp)
        f_np = np.real(np.fft.fft(h)).astype(np.float32)
        f_np[0] = f_np[0] * 0.5                         # ← halve H[0] only
        f = torch.as_tensor(f_np, dtype=dtype, device=device)
        self._ramlak_cache[key] = f
        return f

    def filter_sino(self, sino: torch.Tensor) -> torch.Tensor:
        """Kak-Slaney ram-lak filter along the detector axis.

        The sinogram is zero-padded from ``N`` to ``2N`` along the detector
        axis before the FFT and the ramp kernel is built at length ``2N``,
        so the FFT-domain multiply implements a *linear* convolution rather
        than a circular one. After the inverse FFT, only the first ``N``
        samples are kept (the padded tail is discarded).

        This is the textbook prescription (Kak & Slaney 1988, §3.3.3): with
        a length-``N`` FFT the circular convolution wraps the long 1/n²
        kernel tails around the array boundary, contaminating the filter's
        low-frequency response — the residual ``DFT[0] ≈ 2/(π²·N·Δ²)``
        that any pure length-``N`` Kak-Slaney FFT carries. Doubling the
        FFT length removes the wrap (and halves the residual DC).
        """
        s, nd = self._normalize_sino(sino)
        x = s.unsqueeze(1)                                            # (B, 1, A, N)
        N = x.shape[-1]
        M = 2 * N
        pad = torch.zeros(x.shape[:-1] + (N,), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=-1)                           # (B, 1, A, 2N)
        f = self._ramlak(M, x.device, x.dtype)                        # (2N,)
        spec = torch.fft.fft(x_pad, dim=-1, norm="ortho")
        spec = spec * f
        y_full = torch.fft.ifft(spec, dim=-1, norm="ortho").real      # (B, 1, A, 2N)
        y = y_full[:, 0, :, :N]                                       # truncate → (B, A, N)
        return self._restore_image_ndim(y, nd)

    def fbp(self, sino: torch.Tensor) -> torch.Tensor:
        """Sidky-style FBP: ``X^T · (ramp · g) · (Δβ/2)``.

        ``Δβ = (angle_end - angle_start) / n_angles`` discretises the FBP
        angle integral; the ``1/2`` is the standard 2π full-scan redundancy
        weight (each line is sampled twice). For non-2π scans the ``1/2``
        is dropped — callers wanting Parker / fan-redundancy weighting
        should pre-weight ``sino`` themselves before this call.
        """
        ang_range = float(self.geom.angle_end - self.geom.angle_start)
        d_beta = ang_range / float(self.geom.n_angles)
        if abs(ang_range - 2.0 * math.pi) < 1e-3:
            weight = d_beta / 2.0
        else:
            weight = d_beta
        return self.back_project(self.filter_sino(sino)) * weight

    # forward(image) → sinogram for use as a torch.nn.Module
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_project(image)
