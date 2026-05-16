# Wu 2015 — Novel FBP for Sparse-View CT

Classical (non-learned) FBP-based sparse-view CT reconstruction with
two key ideas: a **radial-position-dependent ramp filter** that produces
a view-aliasing-free image (cutting frequencies that exceed the local
Nyquist rate set by `s · Δβ`), followed by **feature-preserving
sinogram interpolation** that restores the high-frequency detail
through motion-compensated up-sampling of the residual sinogram. The
authors report mean-absolute-error improvements vs linear sinogram
interpolation at down-sampling factors of 4–8 on thorax (Siemens
SOMATOM) and head (Artis Zeego) XCAT phantom simulations.

## Citation

```bibtex
@inproceedings{faucris.113449864,
  author        = {Wu, Meng and Maier, Andreas and Yang, Qiao and Fahrig, Rebecca},
  booktitle     = {The 13th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine},
  faupublication = {yes},
  note          = {UnivIS-Import:2016-06-01:Pub.2015.tech.IMMD.IMMD5.anovel{\_}3},
  pages         = {202-205},
  peerreviewed  = {unknown},
  title         = {{A} {Novel} {Filtered} {Backprojection}-{Based} {Algorithm} for {Sparse} {View} {CT} {Image} {Reconstruction}},
  url           = {https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2015/Wu15-ANF.pdf},
  venue         = {New Port, Rhode Island, US},
  year          = {2015}
}
```

PDF: [papers/Wu_2015_ct_meeting.pdf](../papers/Wu_2015_ct_meeting.pdf)

## Algorithm

Schematic (Fig. 1 in the paper):

```
sparse-view sino  ──► aliasing-free FBP ──► forward project
                                                  │
                          subtract  ◄─────────────┘
                              │
                              ▼
                 feature-preserving interpolation
                              │
                              ▼
                          FBP  ──► soft-threshold ──► adaptive merge
                                                            ▲
                                                            │
                                          aliasing-free FBP ┘
                              (repeat 2–3 iterations)
```

### A. View-aliasing-free reconstruction

For a parallel-beam scan with angular sampling `Δβ`, the **local Nyquist
radial sampling** at distance `s` from the rotation centre is

  `Δ̃_R(s) = s · Δβ`         (Eq. 2)

Frequencies above `1 / (2·Δ̃_R)` cannot be sampled at the corresponding
radius without aliasing. The authors split the ramp filter into eight
triangular frequency segments `h_i` with centre frequencies `f_i`,
back-project each band separately, and combine per-pixel with a sigmoid
weight that suppresses bands the local Δ̃_R can't support:

  `c_i(x) = 1 / (1 + exp(10 · (f_i · Δ̃_R(s) − 1)))`        (Eq. 5)
  `g(x) = Σ_i c_i(x) · B(h_i ∗ w · y)(x)`                  (Eq. 4)

Pixels near the centre (small `s`) keep all bands → full resolution.
Pixels at the edge keep only low-frequency bands → blurred but
aliasing-free. The result is shown in Fig. 3 of the paper.

### B. Residual extraction in projection space

Forward-project the aliasing-free image and subtract from the measured
sparse-view sinogram. What's left is exactly the part of the data the
aliasing-free reconstruction couldn't accommodate — the high-frequency
structures that are responsible for the streak artifacts.

### C. Feature-preserving interpolation (Eqs. 6–8)

For each pair of adjacent projections `(β₁, β₂)`, interpolate the
midpoint view by symmetric motion-compensated averaging:

  `ỹ(u, (β₁+β₂)/2) = ½ · [y(u−t, β₁) + y(u+t, β₂)]`

where `t` is the integer shift that minimises the L1 distance between
the two windowed local-pixel patches:

  `t = argmin_t ‖y(N_{u−t}, β₁) − y(N_{u+t}, β₂)‖₁`

This is *symmetric* motion estimation — both projections contribute
equally and the shift is half the inter-view motion. For non-midpoint
interpolation fractions α∈(0,1) the formula generalises to Eq. 8.

### D. Adaptive merge

FBP the interpolated residual sinogram → restored high-frequency image.
Apply a **soft threshold** to suppress residual streaks (the
interpolation can't perfectly recover the smallest aliased structures),
then add the survivors to the aliasing-free image. Repeat the
residual-extract → interpolate → FBP → merge cycle 2–3 times.

## Reported results (mean absolute error, HU; lower is better)

| Phantom × downsampling | Linear interp. | Direct merge | Adaptive merge |
|---|---:|---:|---:|
| Thorax × 6 | 26 | 24 | **19** |
| Thorax × 8 | 34 | 32 | **23.5** |
| Head × 4   | 15 | 13 | **12** |
| Head × 6   | 24.5 | 21 | **18** |

The adaptive variant consistently dominates linear interpolation and
the direct (no soft-thresholding) variant. Total compute is ~2–4×
that of a normal full-view FBP.

## Why this matters for Agent4CT

The DL-Sparse-View challenge uses 128-view fan-beam at 512×512 — a
mismatch ratio firmly in the "view-aliased FBP" regime. Wu 2015 is
the canonical *non-learned* baseline for this regime: it is what a
"clever classical FBP" gives you before any neural network is
introduced. Useful for the leaderboard as:

- A **reference floor** for any learned method to beat (or fail to
  beat) on the same noise/geometry.
- A diagnostic: if a learned method beats plain FBP but doesn't beat
  Wu 2015, the gain is coming from sinogram-interpolation
  capabilities the network has implicitly learned — not from
  prior-driven detail.
- A **drop-in initialisation**: the aliasing-free image is a good
  warm-start for any iterative or unrolled network.

See [pentathlon/demo_dl_reference/solver_wu_2015.py](../pentathlon/demo_dl_reference/solver_wu_2015.py)
for the reference implementation used in our pentathlon harness.
