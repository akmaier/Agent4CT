# Transformers for CT — Side-by-Side: TransCT vs. U-Swin

Two recent transformer-based CT reconstruction papers, both addressing
image-domain restoration but for **different degradation regimes**.
Decisive verdict for the Agent4CT sparse-view pentathlon: **U-Swin is
the better-suited reference** because limited-angle streak artifacts
(its target) are structurally identical to sparse-view aliasing
artifacts; TransCT targets Poisson noise from low-dose acquisition, a
fundamentally different distortion.

## Paper A — TransCT (MICCAI 2021)

```bibtex
@inproceedings{zhang2021transct,
  title     = {{TransCT}: Dual-Path Transformer for Low Dose Computed Tomography},
  author    = {Zhang, Zhicheng and Yu, Lequan and Liang, Xiaokun and
               Zhao, Wei and Xing, Lei},
  booktitle = {MICCAI 2021},
  pages     = {55--64},
  year      = {2021},
  doi       = {10.1007/978-3-030-87231-1_6},
  note      = {arXiv:2103.00634, code: github.com/zzc623/TransCT}
}
```

PDF: [papers/Zhang_2021_MICCAI_TransCT_2103.00634.pdf](../papers/Zhang_2021_MICCAI_TransCT_2103.00634.pdf)

**Task**: Low-dose CT (LDCT) → normal-dose CT (NDCT). Pure image-domain
denoising; the FBP reconstruction is already done.

**Architecture** (paper Fig. 1):

1. **Frequency decomposition**: Gaussian filter (σ = 1.5) splits the
   LDCT image into `X_L` (low-freq smooth part) and `X_H = X − X_L`
   (high-freq part containing both detail *and* noise).
2. **Feature extraction from `X_L`** (two paths):
   - `X_Lc1`, `X_Lc2` — content features via two shallow CNNs.
   - `X_Lt` — latent texture features (H/32 × W/32 × 256) via deeper
     CNN.
3. **Feature extraction from `X_H`**: sub-pixel layer + 3 conv layers →
   `X_Hf` (H/16 × W/16 × 256).
4. **Transformer**: tokenise `X_Lt` → `S_L`, `X_Hf` → `S_H`.
   3 encoder blocks process `S_L` (self-attention). 3 decoder blocks
   process `S_H` using `S_L` as key/value (cross-attention). Each
   block: multi-head self-attention + MLP + residual.
5. **Piecewise reconstruction**: combine transformer output `Y` with
   `X_Lc1`, `X_Lc2` via two ResNet upsamplers (Conv + LReLU +
   sub-pixel) → full-res NDCT image.

**Loss**: MSE between predicted and true NDCT.

**Training**: 300 epochs, Adam, lr=1e-4 → 1e-5, batch 8.

**Dataset**: AAPM 2016 Low-Dose CT Grand Challenge; quarter-dose
inputs → standard-dose targets.

**Why low-dose CT?** Image is FBP-recoverable (no missing data), but
contaminated with **Poisson noise** that lives mostly in the
high-frequency band. The dual-path design lets the network learn how
HF noise correlates with LF context.

## Paper B — U-Swin (Phys. Med. Biol. 2024)

```bibtex
@article{xu2024hybrid,
  title     = {Hybrid {U}-{N}et and {S}win-transformer network for
               limited-angle cardiac computed tomography},
  author    = {Xu, Yongshun and Han, Shuo and Wang, Dayang and
               Wang, Ge and Maltz, Jonathan S. and Yu, Hengyong},
  journal   = {Physics in Medicine and Biology},
  volume    = {69},
  number    = {10},
  pages     = {105012},
  year      = {2024},
  doi       = {10.1088/1361-6560/ad3db9}
}
```

(IOP / Springer paper. PDF wasn't directly accessible without
authentication; key architecture details summarised from the paper's
abstract + IOP / PMC landing pages.)

**Task**: Limited-angle cardiac CT reconstruction. Input: FBP of an
angularly-truncated sinogram (≤180° + fan). Output: corrected image
with reduced streak artifacts.

**Architecture**:

1. **U-Net branch**: standard 4-level encoder/decoder with skip
   connections — restores structural information that the limited-angle
   FBP has missed (the spatial-prior side of the hybrid).
2. **Swin transformer branch**: shifted-window self-attention modules
   that gather global context across the image — the long-range
   dependency side that catches coherent streaks bridging the FOV.
3. **Hybrid fusion**: U-Net features are augmented at multiple scales
   by Swin attention blocks; the final decoder fuses both streams to
   produce the cleaned image.

**Loss**: standard image-domain reconstruction loss (MSE / L1 / SSIM —
the paper reports outperforming "state-of-the-art deep learning-based
methods" on synthetic XCAT and clinical COCA datasets).

**Why limited-angle CT?** Limited-angle FBP has *streak artifacts*
caused by the unsampled angular wedge. These are non-local, coherent
structures — exactly what Swin self-attention is designed to model.

## Side-by-side comparison

| | **TransCT** (Zhang 2021) | **U-Swin** (Xu 2024) |
|---|---|---|
| **Target degradation** | LDCT Poisson noise (dose reduction) | Limited-angle streak artifacts (angular under-sampling) |
| **Input image** | FBP of full-view but noisy projections | FBP of truncated-angle projections |
| **Architecture style** | Frequency-split dual-path: CNN encoders → transformer encoder/decoder with cross-attention | Hybrid U-Net + Swin transformer with multi-scale fusion |
| **Self-attention type** | Vanilla multi-head (global tokens) | Swin (shifted-window, local + cross-window) |
| **Domain prior baked in** | Frequency decomposition (HF noise / LF content) | Local CNN spatial prior (U-Net skips) |
| **Receptive field** | Global via transformer | Local (U-Net) + windowed-global (Swin) |
| **Loss** | MSE on NDCT target | Image-space reconstruction (paper-dependent) |
| **Datasets** | AAPM 2016 LDCT Challenge (Mayo) | XCAT (synthetic) + COCA (clinical cardiac) |
| **Hardware claim** | TITAN X 12 GB, 300 epochs | Not specified in abstract |

### Why **U-Swin wins for our sparse-view pentathlon**

| Reason | TransCT | U-Swin |
|---|---|---|
| Matches the **artifact type** in our 128-view geometry (coherent streaks across FOV) | No — designed for unstructured Poisson noise | **Yes** — designed for exactly this artifact class |
| Uses a **modern transformer backbone** that's parameter-efficient on 512² | Vanilla MHA on small token grids (32×32 tokens) | **Swin** windowed self-attention scales well to 512² |
| **CNN prior** that helps with limited data | None inside the transformer path | U-Net skip connections at every level |
| Available **open-source reference** | github.com/zzc623/TransCT (TF) | None public (architecture described in paper) |

Open-source availability is one point in TransCT's favour, but the
architectural mismatch (denoising vs. de-streaking) is decisive.

### Why **TransCT is the runner-up reference**

If we ever turn the pentathlon toward **noise-domain LDCT** (e.g. the
Mayo Wagner subset, where the challenge is Poisson statistics rather
than missing views), TransCT becomes the right reference: its
frequency-decomposition + cross-attention recipe is exactly tuned to
extract noise-free textures from a noisy LF/HF decomposition.

## Implemented in this repo

`pentathlon/demo_dl_reference/solver_uswin.py` reproduces a faithful
U-Swin variant adapted to the pentathlon's 128-view sparse-view fan-beam
geometry:

- **Input**: FBP of the noisy sparse-view sinogram (μ mm⁻¹, clipped at 0).
- **Architecture**: 4-level U-Net (Conv-GroupNorm-ReLU double blocks)
  augmented with a Swin transformer block at each encoder level. We use
  the well-tested `timm.SwinBlock`-style implementation (sliding-window
  MSA with shifted-window pattern), or a hand-rolled equivalent if
  timm isn't available on the cluster.
- **Loss**: MSE on the truth phantom (matches the rest of the
  demo_dl_reference suite for fair comparison).
- **Training**: Adam, lr ∈ [1e-4, 1e-3], 8–16 epochs, batch 4. Tunable
  via the `learned_solver_search_agent.py` 20-iter random search.

The full implementation, sbatch wrappers, and search results are
documented in `pentathlon/demo_dl_reference/README.md` Sect. 7.

## Open question worth posing

Both papers operate purely in the image domain. Neither uses a
data-consistency step against the original sinogram (cf. ItNet v3,
Hammernik-VN). The strongest *combination* for sparse-view would
plausibly be **U-Swin as the denoiser inside an unrolled
data-consistency loop** — i.e. an ItNet-V4 with U-Swin replacing the
5-level U-Net. That experiment isn't in either paper but is a one-line
swap in `solver_itnet_v3.py`.
