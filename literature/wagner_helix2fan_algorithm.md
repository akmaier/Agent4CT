# Wagner `helix2fan` algorithm — full pipeline + flying focal spot

Companion notes for the Wagner et al. 2023 ISBI paper
(`literature/2211.01111_Wagner_DualDomainDenoising_LDCT.md`) and its
open-source code release:

- **Repo**: <https://github.com/faebstn96/helix2fan>
- **Algorithm**: helical cone-beam → flat-detector → 2D circular fan-beam,
  using single-slice rebinning per Noo et al. 1999
  (<https://doi.org/10.1088/0031-9155/44/2/019>).
- **Purpose**: convert Mayo `LDCT-and-Projection-data` DICOM-CT-PD
  helical projections into 2D fan-beam sinograms usable with a
  conventional / differentiable FBP.
- **Local mirror**: `ddssl_ldct/helix2fan.py` (pure-NumPy port). The
  port has known bugs — see "Differences from Wagner" at the bottom.

## Pipeline overview

Mayo LDCT-and-Projection-Data ships **cone-beam helical sinograms on a
cylindrical (curved) detector**: each readout is a `(nv=64, nu=736)`
patch covering 64 detector rows in the cone (z) direction and 736
channels in the fan direction, with the source moving along a helical
trajectory at ≈ -0.4 mm/view (descending) and ≈ 60 full rotations
over a ~38 000-readout series. Wagner converts this to **fan-beam
sinograms on a flat detector** in two consecutive remaps, both
implemented as functions in `rebinning_functions.py`:

```
  DICOM-CT-PD .dcm files          read_data.py / read_dicom_ctpd
        │  cone-beam · curved detector · helical · (n_proj, nv, nu)
        │  ≈ 37 000 readouts; nv=64 cone rows; nu=736 fan channels
        ▼
  raw_projections (n_proj, nv, nu)
        │
        │  Step 1: per-readout curved → flat detector remap
        │  rebinning_functions.py::rebin_curved_to_flat_detector
        │  (only changes the detector shape; still cone-beam,
        │  still helical, still 64 cone rows)
        ▼
  proj_flat_detector (n_proj, nv, nu)
        │  cone-beam · *flat* detector · helical
        │
        │  Step 2: helical → circular fan-beam
        │  single-slice rebinning per Noo et al. 1999
        │  rebinning_functions.py::rebin_helical_to_fan_beam_trajectory
        │  (this is the actual cone → fan step — see below)
        ▼
  proj_fan_geometry (rotview, nu, nz_rebinned)
        │  fan-beam · flat detector · circular trajectory
        │  no v dimension; one 2D fan-beam sinogram per output z-slice
        │
        │  Step 3 (downstream): per z-slice, FBP a (rotview, nu) sino.
        ▼
  reconstruction (nz_rebinned, image_size, image_size)
```

### Wagner's repo structure: one file, two functions — no second script

The user's natural question — *"the LDCT data is cone-beam, we need
fan-beam on a flat detector, is there a second script that does this?"*
— has a subtle answer:

**There is no second `.py` file. There is a second *function*,
inside `rebinning_functions.py`, that does the actual cone → fan
conversion.** The full list of files Wagner ships (verified against
upstream commit `819fac95` via independent SHA-1 hash, see
"Provenance" below) is:

```
helix2fan/
├── main.py                       # orchestrator: runs Step 1, then Step 2
├── read_data.py                  # DICOM-CT-PD loader (geometry + pixels)
├── rebinning_functions.py        # both Step 1 and Step 2 live here
├── helper.py                     # TIFF I/O
├── reco_example_fan_beam.py      # downstream FBP on the rebinned sino
├── requirements.txt
├── README.rst
└── torch-radon_fix/
```

`faebstn96` (Wagner) has six other public repos but none of them are
related to cone-beam → fan-beam rebinning. `torch-radon` is the
forward/back-projector. The trainable-bilateral-filter repos are the
backbone of `solver_dual_ddomain_bilateral_*.py` in our project.
`geometry_gradients_CT` is gradient-based geometry calibration. There
is no separate "cone-to-fan-beam-flat-detector" repo.

The mapping function-by-function:

| Stage | Function in `rebinning_functions.py` | What changes | What's preserved |
|---|---|---|---|
| 1 | `rebin_curved_to_flat_detector` (lines 48 / 92) | detector geometry curved → **flat** | still cone-beam, still helical, still 64 cone rows |
| 2 | `rebin_helical_to_fan_beam_trajectory` (line 150) | **cone-beam → fan-beam** *and* helical → circular | flat detector, but now no v axis (one row per output z) |

So **Step 2** is what the user is asking about — and it does both the
cone-to-fan conversion (by collapsing the 64 cone rows into a single
in-plane fan-beam row per output z-slice via Noo 1999 SSR
interpolation) **and** the helical → circular conversion (by
gathering only the readouts that share each output azimuth) in one
pass. Mathematically the cone→fan collapse is the linear interp
along v at `v_precise` plus the cosine path-length weight `w`; the
helical → circular collapse is the `proj_helic[s_angle::rotview]`
gather inside the same loop. See the "Step 2" section below for the
full derivation. Our local port in `ddssl_ldct/helix2fan.py` mirrors
this two-function architecture (`rebin_curved_to_flat` and
`rebin_helical_to_fan`), so we are doing the cone → fan conversion —
the prior featureless-disc symptom was bugs inside these functions
(Bug 1, the centering of the curved → flat remap, was the dominant
one), not a missing pipeline step.

If we ever need cone→fan accuracy beyond what Noo SSR provides
(e.g. larger pitch, larger cone angle), the standard upgrade is
**ASSR** (Advanced Single-Slice Rebinning, Kachelrieß et al. 2000)
— a tilted-plane fit rather than the horizontal-plane assumption SSR
makes. For the Mayo helical pitch ≈ 252 mm/rotation and cone half-
angle arctan(32·1.095 / 1085.6) ≈ 1.85°, SSR is what Wagner uses and
the standard pragmatic choice.

The crucial structural point — and the one the local port currently
misses on the FBP side — is that **the output is a *single 360°
rotation* per z-slice**: `proj_fan_geometry[s_angle, :, i_z]` is the
fan-beam projection at azimuth `s_angle * 2π / rotview` for the
output slice at `z_out[i_z]`. Even though the helix has 60+ rotations
of data, the SSR algorithm condenses them into one virtual 2π scan
per output slice, by picking the helical readouts that cover each
(azimuth, z) combination and interpolating axially.

## DICOM-CT-PD geometry tags

Reference: *DICOM-CT-PD User Manual Version 3*
(<https://doi.org/10.7937/9NPB-2637>).

| Tag (private) | Field | Meaning |
|---|---|---|
| `(0x7029,0x1002)` | `du` | Detector channel pitch in *fan* direction (mm). |
| `(0x7029,0x1006)` | `dv` | Detector row pitch in *cone* direction (mm). |
| `(0x7031,0x1003)` | `sod` | Source-to-isocentre distance (mm). |
| `(0x7031,0x1031)` | `sdd` | Source-to-detector distance (mm). |
| `(0x7031,0x1033)` | `(u0, v0)` | Central detector element. Float, between physical cells. |
| `(0x7031,0x1002)` | z | Source-z position at this readout (mm). |
| `(0x7031,0x1001)` | φ | Gantry angle at this readout (rad). |
| `(0x7041,0x1001)` | `water_mu` | μ of water at the tube spectrum, for HU conversion. |
| `(0x7033,0x100B)` | `dangles` | **FFS** azimuthal source-shift per readout (rad). |
| `(0x7033,0x100C)` | `dz` | **FFS** axial source-shift per readout (mm). |
| `(0x7033,0x100D)` | `drho` | **FFS** radial source-shift per readout (mm). |
| `Rows` | nu | Number of detector cells in the fan direction (e.g. 736). |
| `Columns` | nv | Number of detector rows in the cone direction (e.g. 64). |
| `RescaleSlope`, `RescaleIntercept` | — | Convert stored uint16 pixel values → line-integral μ·L (1/cm units · path-length). |

Pixel-data buffer ordering in Wagner's `read_projections`:

```python
proj_array = np.frombuffer(dataset.PixelData, 'H').astype('float32')
proj_array = proj_array.reshape([cols, rows], order='F')
proj_array = proj_array * rescale_slope + rescale_intercept
raw_projections[i] = proj_array[:, ::-1]    # u-axis flip
```

Net effect on a single readout: `(nv, nu)` array, then the *u* axis
(fan direction) is reversed. The flip means the central element after
loading is at index `nu - 1 - u0` (i.e. ≈ `nu - u0`), not `u0`.
**This single line determines the sign convention for the entire rest
of the pipeline** (see "Differences from Wagner" — this is bug 1 in
the local port).

The angles come out as

```python
angles = np.array([unpack_tag(d, 0x70311001) for d in data_headers]) + (np.pi / 2)
angles = - np.unwrap(angles) - np.pi
```

i.e. monotonically increasing from a negative value (Wagner's
convention), opposite of the raw DICOM sign.

Pitch is **not** stored as a tag in Mayo data; Wagner computes it from
the z-positions and angles:

```python
pitch = (z_positions[-1] - z_positions[0]) / total_rotations
total_rotations = (angles[-1] - angles[0]) / (2 * np.pi)
```

The "views per rotation" is then

```python
rotview = round(len(angles) / total_rotations)
```

For the Mayo L014 series this evaluates to **630** views per rotation
(Siemens SOMATOM Definition AS/AS+ acquisition). This is *not* a bug
— it's how the scanner sampled — even though Wagner's paper used
2304 angles per rotation for the breast-CT sub-experiments (a
different sensor / mode).

## Step 1: curved → flat detector remap

The Siemens detector is a **cylindrical arc**: each channel sits at a
fixed arc-radius `sdd` from the source, with channel-to-channel
**arc-angle** spacing

```
dphi_curved = 2 * arctan(du / (2 * sdd))
```

(small-angle: `dphi ≈ du / sdd`). The fan algorithm wants a **flat**
detector at distance `sdd` from the source, with pixel pitch `du` in
the fan direction.

Wagner's coordinate frame (`rebinning_functions.py`,
`_rebin_curved_to_flat_detector_core`):

- Origin at the source.
- `+y` toward the detector centre.
- `+x` along the fan (u) direction.
- `+z` along the cone (v) direction.

For each *virtual flat-detector pixel* `(i_u, i_v)`:

```python
x_det = (i_u - nu/2 + 0.5) * du
z_det = (i_v - nv/2 + 0.5) * dv
p_flat = [x_det, sdd, z_det]
```

(The `+0.5` shift puts the sample at the pixel *centre*, not the
corner; identical convention as PYRO-NN.)

The ray from the source through this flat pixel intersects the curved
detector at

```python
p_curved = p_flat / |p_flat| * sdd     # vector of length sdd
phi      = arcsin(p_curved[0] / sdd)   # arc-angle (small-x: ≈ x_det / sdd)
```

The point's curved-detector channel index is

```python
i_u_curved = phi / dphi_curved + (nu - u0)
i_v_curved = p_curved[2] / dv + (nv - v0)
```

— note **both** offsets are `nu - u0` / `nv - v0`, not `u0` / `v0`,
because of the channel flip at load time. The remap is a **bilinear**
interpolation of `proj_curved[i_angle, :, :]` at that fractional
`(i_v_curved, i_u_curved)` location, with the v-coordinate also a
function of `i_u` (because the curved arc *bulges* in z by a small
cos-factor as you move toward the corners of the flat).

This is a *small* per-pixel correction relative to a pure u-only
remap, but it matters at the detector corners where the cone angle
is largest (and where lung apices / pelvis rim live).

## Step 2: helical → circular fan-beam (Noo 1999 SSR)

The single-slice rebinning algorithm condenses a helical scan
covering many turns into a 2D fan-beam sinogram at each desired
output z-slice.

For each output azimuth `s_angle ∈ [0, rotview)` and each output
z-slice `z_out`, gather the helical readouts that hit `s_angle`
modulo a full rotation:

```python
idx_helix = arange(s_angle, n_proj, rotview)   # one per turn
z_src_list = z_positions[idx_helix]
```

Each such readout is *at* azimuth `s_angle` but at a *different
source-z* (because the helix has advanced). For each `(s_angle, z_out)`
target we pick the *one* readout whose `z_src` is closest to `z_out`
(within `± pitch / 2`) and interpolate axially within that readout's
flat-detector v-extent to recover the line-integral that *would* have
been measured if the source had been at the same azimuth `s_angle` but
at axial position `z_out`. Noo et al. 1999 Eq. (1)–(2):

```python
v_precise = dZ * (u_mm**2 + sdd**2) / (sod * sdd)        # Eq. (2)
v_val     = interp(v_precise, v_grid, proj_flat[idx, :, i_u])
w         = sqrt(u_mm**2 + sdd**2) / sqrt(u_mm**2 + v_precise**2 + sdd**2)   # Eq. (1)
proj_rebinned[s_angle, i_u, i_z_out] = w * v_val
```

with

- `u_mm = (i_u - nu/2 + 0.5) * du` (flat-detector u in mm, source frame)
- `dZ = z_src - z_out` (axial offset between actual readout and target)
- `sod = source-to-object`, `sdd = source-to-detector`.

The amplitude weight `w` is the SSR cosine-correction: a ray that
intersects the imaging plane at non-zero `v_precise` traverses a
**longer** path through the patient than the corresponding fan-beam
ray, so the line-integral has to be scaled by the ratio of path
lengths.

This formula assumes **every** target `(s_angle, z_out)` has *some*
helical readout within ± pitch / 2 in z — i.e. the helix pitch is
smaller than the detector axial extent `nv * dv` (Wagner says
"`distance = 0.5 * args.pitch`. Full scan. For short scan see Noo et
al."). When pitch > axial extent, some `(s_angle, z_out)` cells go
unwritten and the FBP image will have a wedge of streak artefacts.

## Step 3 (downstream): FBP reconstruction per z-slice

`reco_example_fan_beam.py` uses **torch-radon**'s `RadonFanbeam` with:

```python
angles = metadata['angles'][:rotview] + np.pi / 2
RadonFanbeam(
    image_size=512,
    angles=angles,                           # rotview-long
    source_distance=metadata['dso'] / voxel_size,
    det_distance=metadata['ddo'] / voxel_size,
    det_count=nu,                            # 736
    det_spacing=metadata['du'] / voxel_size,
    clip_to_circle=False,
)

prj = np.flip(projections[:, :, i_z], axis=1)    # "Siemens flip"
filtered = radon.filter_sinogram(prj * voxel_scaling, filter_name=args.fbp_filter)
fbp = radon.backprojection(filtered)
```

Three subtle but essential details:

1. **Angles are the *original* helical angles for `rotview` readouts,
   plus `π/2`.** This re-aligns Wagner's monotonically-increasing,
   negative-start convention with torch-radon's expected `[0, 2π)`
   layout.
2. **`det_distance = ddo`, not `sdd`.** `ddo = sdd - sod` is the
   isocentre-to-detector distance. torch-radon parameterises the
   geometry from the *isocentre*, not the source.
3. **Per-slice flip along the u axis** (`axis=1` of the
   `(rotview, nu)` slice) — required to match torch-radon's
   right-handed coordinate system to the curved-detector flip Wagner
   applied at load time.

Voxel scaling: all distances are rescaled by `1 / voxel_size` so
that the reconstructed grid has pixel pitch `voxel_size` mm rather
than 1 (torch-radon's unit length default).

HU conversion at the end:

```python
fbp_hu = 1000 * (fbp - hu_factor) / hu_factor       # WaterAttenuationCoefficient
```

## Flying focal spot — what Wagner does NOT do, and what proper handling looks like

The Siemens SOMATOM `Definition AS/AS+` (the scanner in the Mayo
LDCT projection data) deflects the X-ray source in three independent
axes during the acquisition, on a per-readout basis:

| Axis | DICOM tag | Symbol | Magnitude (typical) |
|---|---|---|---|
| Azimuthal (in-plane rotation) | `(0x7033,0x100B)` | `dangles` | ± 0.5 × `dphi_view` rad |
| Axial (z) | `(0x7033,0x100C)` | `dz` | ± 0.5 × `dv` mm |
| Radial (toward detector) | `(0x7033,0x100D)` | `drho` | ± 1 mm |

The **physical purpose** of FFS is to double (or quadruple) the
in-plane and axial sampling density: the scanner takes two (or four)
adjacent readouts with the source at slightly shifted positions
relative to a nominal trajectory, *interleaved by the deflection
pattern*. Without correction those readouts get treated as samples
from the same source position they would have had **without**
deflection — i.e. as if every other readout were "duplicated" at the
nominal angle / z. The effect on the reconstruction is a halving of
the effective azimuthal sampling and visible aliasing streaks
(especially around high-contrast bone / metal edges).

### What Wagner's code does

`read_data.py` reads the three FFS tags into `args.dangles`,
`args.dz`, `args.drho`. **They are never used in the rebinning math.**
A commented-out line in `rebin_curved_to_flat_detector` shows what
the axial correction *would* look like:

```python
# dz_ffs = (args.ddo / args.dso) * args.dz[i_angle]
```

Wagner's `README` is candid:

> Right now the helical to fan beam geometry rebinning does not
> properly correct for the flying focal spot (FFS) acquisition. All
> required geometry parameters are correctly read out from the
> DICOM-CT-PD raw data (`--dangles`, `--dz`, `--drho`). However,
> torch-radon currently does not support shifting the source
> position relative to the detector which would be required to
> accurately correct for FFS.

So Wagner's code is *FFS-aware but not FFS-correcting*: it knows the
shifts exist but assumes them away.

### What proper FFS correction should do

For each helical readout `i` with source shifts `(dphi_i, dz_i, drho_i)`,
the **effective source position** is

```
phi_eff_i = phi_nominal_i + dphi_i
z_eff_i   = z_nominal_i   + dz_i
rho_eff_i = sod           + drho_i        # radial deflection from isocentre toward detector
```

i.e. the source sits at `(rho_eff_i * cos(phi_eff_i), rho_eff_i * sin(phi_eff_i), z_eff_i)`
in the patient frame, not at the nominal helical point. The two
consequences are:

1. **Curved → flat remap (Step 1) must use `rho_eff_i` for `sod` and
   `sod + drho_i` for `sdd`** (since the detector arc stays put but
   the source moves radially). With `drho` ≈ ±1 mm and `sod` = 595 mm
   this is a 0.17 % correction in the magnification — small but
   non-negligible for sub-pixel accuracy.

2. **Helical → fan SSR (Step 2) must use `phi_eff_i` and `z_eff_i`
   for binning into `s_angle` and the z-interpolation window**, not
   the nominal `phi_nominal_i`, `z_nominal_i`. Specifically:
   - The "which output `s_angle` does this readout hit?" binning
     decision changes — the deflection moves the readout into the
     *next* (or previous) azimuthal bin in the SSR output grid.
   - The `dZ = z_src - z_out` quantity in Noo Eq. (2) is computed
     against `z_eff_i`, not `z_nominal_i`. With `dz` ≈ ±0.5 mm and
     pitch ≈ 252 mm/rotation this re-sorts which output z-slices
     the readout contributes to.

3. **Single-source-position back-projection (Step 3)** is no longer
   geometrically correct. Properly, each readout would back-project
   from its own `(rho_eff_i, phi_eff_i, z_eff_i)`, requiring an FBP
   implementation that supports a *non-circular* source trajectory.
   torch-radon can't do this; PYRO-NN can't do this either. The
   pragmatic alternative is to:

   - **Group readouts by FFS state**: typically the scanner cycles
     through 2 or 4 fixed deflection states (`±dphi, ±dz`) per
     rotation, so the readouts fall into 2 or 4 sub-sets each of
     which *is* on a circular trajectory (just shifted) — and
     reconstruct each sub-set separately, then average. Effectively
     doubles the angular sampling density of each sub-reconstruction.
   - **Or rebin to a common nominal trajectory first**: shift each
     readout's projection columns by an amount that *undoes* the
     FFS deflection at the level of the detector signal (small
     bilinear shift in u for `dphi`, in v for `dz`, and a slight
     amplitude rescale for `drho`). This is the approach Wagner's
     commented `dz_ffs = (ddo / dso) * dz[i_angle]` line was
     starting on — `ddo / dso` is the lever-arm factor that maps a
     source-z shift into an apparent detector-v shift.

A reasonable **first cut** for our pipeline:

- Read `dangles`, `dz`, `drho` (we already do).
- In Step 2, replace `z_positions` with `z_eff = z_positions + dz`
  before the `idx_helix` and `z_src_list` lookups.
- Replace `gantry_angles` with `phi_eff = gantry_angles + dangles`
  before the `rotview` and `s_angle` binning, *and* before computing
  `total_rotations`.
- In Step 1, scale the per-readout output by `(sod_eff / sod)` so
  the line-integrals are normalised against the nominal SOD geometry
  the downstream FBP assumes (so `drho` ≠ 0 doesn't bias the recon).
- Leave Step 3 (FBP) untouched — accept that residual FFS aliasing
  remains, but the in-plane and axial sampling-density gains are
  recovered.

This won't be as accurate as a true source-shifting FBP (which would
need a custom kernel), but it should comfortably get us past the
SSIM=0.18 disaster we currently observe and into the "useful for
training" regime that Wagner's paper validates (SSIM ≳ 0.85).

## Differences between Wagner's code and our local port (`ddssl_ldct/helix2fan.py`)

Read carefully against `rebinning_functions.py` lines 92-135 and
`read_data.py` lines 77-97:

### Bug 1 — central element offset in the curved→flat remap

Wagner uses `(nu - u0)` after the `arr[:, ::-1]` channel flip. Our
port uses `u0`:

```python
# ours (ddssl_ldct/helix2fan.py line 298):
u_curved_idx = u0 + phi * (sdd / du)

# Wagner (rebinning_functions.py line 80):
i_p_on_curved_det_polar = [phi / dphi_curved + (nu - u0), p_curved[2]/dv + (nv - v0)]
```

For L014 with `nu=736`, `u0=369.625`: Wagner's centre lives at index
**366.375**, ours at **369.625**. That's a **3.25-channel** (≈4 mm)
mirroring of the entire fan around the wrong axis. After back-
projection this looks like a low-pass filter (rays from "left" and
"right" of the truth get averaged, and the asymmetric
high-frequencies cancel) — exactly the featureless-disc artefact
visible in `results/breast_debug/validate_L014_fulldose.png`.

### Bug 2 — v-axis identity in curved→flat

Our port skips the cone-direction bilinear (treats v as identity);
Wagner does a full 2D bilinear with the small `cos(phi)` v-shift.
Smaller effect than Bug 1 but visible at the cone-angle extremes.

### Bug 3 — angle convention not normalised

Our port keeps the raw DICOM gantry angles. Wagner applies
`+ π/2` and `-unwrap - π` to remap to a monotonic-increasing
negative-start convention; the downstream FBP then undoes the `+π/2`
back. If the FBP receives our raw angles, the start angle and
rotation direction are wrong.

### Bug 4 — no `Siemens flip` per slice in our validation FBP

`reco_example_fan_beam.py` does `np.flip(projections[:, :, i], axis=1)`
before filtering. Our `validate_mayo_helix2fan.py` likely just hands
the sinogram to `PyronnFanBeamProjector.fbp` directly without that
flip, so the back-projection direction is swapped.

### Bug 5 — FBP geometry uses `sdd`, not `ddo`

`reco_example_fan_beam.py` uses `det_distance = ddo = sdd - sod`,
not `sdd`. Our validation script's FBP may use `sdd` (≈1085.6) where
it should use `ddo` (≈490.6) — would compress / mis-magnify the
reconstruction.

### Bug 6 — no FFS correction at all

Both Wagner and our port skip FFS. For the *first cut* fix described
above (just shifting `phi` and `z` by the per-readout `dangles` /
`dz`) our port is ready to add this — the tags are read but not used.

### Bug 7 — possible HU normalisation

`reco_example_fan_beam.py` divides the FBP output by `hu_factor =
water_mu` (the water attenuation coefficient at the tube spectrum)
and rescales to HU. Our validation compares against truth in either
μ-units or HU; depending on which, the intensity calibration may
need to mirror Wagner's `1000 * (fbp - hu_factor) / hu_factor`.

## Why the FBP looks like a featureless disc

Combining Bugs 1, 3, 4, 5 explains the symptom precisely:

- Bug 1 mirrors every ray to the wrong fan side. Each individual
  back-projection still hits the correct radius from isocentre, so
  the *envelope* (a disc roughly the patient diameter) is preserved.
  But the **azimuthal phase** of every ray is mirrored, so the
  cross-sectional contrast cancels out: opposing rays (which should
  reinforce true structures) instead destructively interfere.
- Bug 3 + Bug 4 swap the rotation direction and shift the starting
  angle, scrambling the angular ordering — so even where Bug 1's
  cancellation isn't perfect, the residual signal lands at the wrong
  angular position.
- Bug 5 mis-magnifies what little survives.

The net result is **a featureless disc with diagonal streak
artefacts** — almost exactly the pattern Wagner's README warns about
for missing-angle slices, but here every slice is "missing" in a
geometry-mismatch sense rather than a physical-coverage sense.

## Pragmatic fix order for `ddssl_ldct/helix2fan.py` + `validate_mayo_helix2fan.py`

1. **Fix Bug 1** in `rebin_curved_to_flat`: change `u0 + phi * sdd /
   du` to `(nu - u0) + phi * sdd / du`. Add the matching `(nv - v0)`
   offset and the per-flat-pixel `p_curved[2]` bilinear in v.
2. **Fix Bug 3** in `read_dicom_ctpd`: add a `gantry_angles_unwrapped`
   array following Wagner's `+ π/2` then `-unwrap - π` recipe, and
   recompute `total_rotations` / `rotview` from it.
3. **Fix Bug 4/5** in `validate_mayo_helix2fan.py`: apply
   `np.flip(sino, axis=-1)` (u-axis flip) per slice before FBP, and
   feed `PyronnFanBeamProjector(sod=sod, sdd=sod + ddo)` — equivalently
   compute `det_distance = ddo` if the projector takes that.
4. **Re-run validate_mayo_helix2fan_L014.sbatch**. Target: SSIM ≥
   0.85 on the L014 fulldose slice at z = 253.5 mm.
5. **Add FFS correction (`dz`, `dangles`)** to Step 2 of the
   rebinning — the first-cut shift-`phi` and shift-`z` approach
   above. Re-validate; expected gain over (1-4) is in the
   "+0.02 SSIM, no more streaks at high-contrast edges" range.

After (4) passes the SSIM threshold, the remaining 9/19 patients
(`L004, L033, L064, L107, L143, L221, L260, L288, L299`) can be
re-rebinned in a single `cluster/slurm/rebin_mayo_helix2fan.sbatch`
batch using the existing 16-CPU joblib pipeline (the workplan
estimate is ~10 h CPU for all 10 patients × 2 dose series).

## Citation

If our port ever becomes the basis of a publishable result, cite both
Wagner's ISBI 2023 paper and the Noo et al. 1999 SSR paper:

```bibtex
@inproceedings{wagner2022dual,
  title={On the Benefit of Dual-domain Denoising in a Self-Supervised Low-dose CT Setting},
  author={Wagner, Fabian and Thies, Mareike and Pfaff, Laura and Aust, Oliver
          and Pechmann, Sabrina and Maul, Noah and Rohleder, Maximilian
          and Gu, Mingxuan and Utz, Jonas and Denzinger, Felix and Maier, Andreas},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5}, year={2023}, organization={IEEE},
  doi={10.1109/ISBI53787.2023.10230511},
}

@article{noo1999singleslice,
  title={Single-slice rebinning method for helical cone-beam CT},
  author={Noo, F and Defrise, M and Clackdoyle, R},
  journal={Physics in Medicine \& Biology},
  volume={44}, number={2}, pages={561}, year={1999},
  doi={10.1088/0031-9155/44/2/019},
}
```
