# WDM: 3D Wavelet Diffusion Models for High-Resolution Medical Image Synthesis

> Structured reading note (summary in my own words + short abstract quote). **Not a
> verbatim reproduction** — the chapter is copyrighted (paywalled on Springer; the
> authors' open preprint is on arXiv). Read the PDF for exact equations/derivations.

## Bibliographic
- **Authors:** Paul Friedrich, Julia Wolleb, Florentin Bieder, Alicia Durrer, Philippe C. Cattin (Department of Biomedical Engineering, University of Basel).
- **Venue:** DGM4MICCAI 2024 (4th MICCAI Workshop on Deep Generative Models), Marrakesh, 2024-10-10. Springer LNCS, proceedings DOI `10.1007/978-3-031-72744-3`, **chapter `_2`**.
- **Open preprint:** arXiv:2402.19043 (v2) — <https://arxiv.org/abs/2402.19043> (arXiv non-exclusive license; **not** CC BY).
- **PDF (local):** [`papers/friedrich_2024_wdm_3d_wavelet_diffusion.pdf`](../papers/friedrich_2024_wdm_3d_wavelet_diffusion.pdf) (gitignored).
- **Code:** <https://github.com/pfriedri/wdm-3d>.

## Abstract (verbatim)
> "Due to the three-dimensional nature of CT- or MR-scans, generative modeling of medical images is a particularly challenging task. Existing approaches mostly apply patch-wise, slice-wise, or cascaded generation techniques to fit the high-dimensional data into the limited GPU memory. However, these approaches may introduce artifacts and potentially restrict the model's applicability for certain downstream tasks. This work presents WDM, a wavelet-based medical image synthesis framework that applies a diffusion model on wavelet decomposed images. The presented approach is a simple yet effective way of scaling 3D diffusion models to high resolutions and can be trained on a single 40GB GPU. Experimental results on BraTS and LIDC-IDRI unconditional image generation at a resolution of 128×128×128 demonstrate state-of-the-art image fidelity (FID) and sample diversity (MS-SSIM) scores compared to recent GANs, Diffusion Models, and Latent Diffusion Models. Our proposed method is the only one capable of generating high-quality images at a resolution of 256×256×256, outperforming all comparing methods."

## The idea (in my words)
The memory bottleneck of 3D diffusion is that the U-Net processes the full `H×W×D` voxel grid at every denoising step. WDM sidesteps this by running the **diffusion process in the wavelet domain** instead of pixel space:

1. **Decompose** the volume with a single-level 3D **discrete wavelet transform** (Haar). This produces **8 subbands** (LLL, LLH, …, HHH), each at **half resolution per axis** (`H/2 × W/2 × D/2`), stacked channel-wise → a tensor of shape `(8, H/2, W/2, D/2)`.
2. **Diffuse** on that concatenated wavelet tensor with a 3D U-Net. The spatial footprint the network sees is **1/8 the voxels** of the original volume (the 8× factor is moved into channels, which is far cheaper than spatial extent for a conv U-Net) — this is what makes `128³` and even `256³` trainable on one 40 GB GPU.
3. **Reconstruct** by predicting the wavelet coefficients through the reverse process and applying the **inverse DWT (IDWT)** to return to image space.

It is a standard DDPM-style denoiser retargeted to wavelet coefficients (check the PDF/code for the exact parameterization and loss — the repo uses a 3D U-Net with the diffusion objective on the coefficient tensor). The contribution is the *simplicity* of the wavelet reparameterization vs. patch/slice/cascade or latent-diffusion alternatives, and that it preserves spatial structure (no patch seams, no autoencoder reconstruction error).

## Evaluation
- **Task:** unconditional 3D image **synthesis** (not reconstruction / inverse problems).
- **Datasets:** BraTS (3D brain MR), LIDC-IDRI (3D lung CT).
- **Resolutions:** `128³` (vs GANs, DMs, LDMs) and `256³` (where it is reportedly the only method producing high-quality samples).
- **Metrics:** **FID** (fidelity) and **MS-SSIM** (sample diversity — lower pairwise MS-SSIM = more diverse). Claims SOTA FID + diversity at `128³`.

## Relevance to this repo
- This is a **prior-training / generative-synthesis** technique, **not** a reconstruction method. It does not directly give a CT *solver*. But it is the standard recipe for **scaling a diffusion prior** in fixed GPU memory, which is exactly the bottleneck for our diffusion-recon work.
- Our Mayo/breast diffusion priors (`ddpm_mayo_*_v4`) are **2D, pixel-space, small** (`ch=96`, 512² slices) and **under-trained** (50–200 train slices, no EMA). If we ever want a **3D / volumetric** Mayo prior (true cross-slice consistency, which the 2D rebinned setup throws away), the wavelet-domain trick here is how to make a `512²`-or-larger 3D DDPM fit in memory.
- More immediately, the **wavelet reparameterization is orthogonal to the DPS/DC-step recon** and could pair with the planned **DDPM v5** prior retrain (full train pool + EMA) — a wavelet-domain prior would be cheaper to scale up in capacity than the current pixel-space `SmallDDPM`.
- Caveat consistent with our audit: a stronger generative prior still only helps reconstruction where the measurements are **incomplete**; on dense 2304-view Mayo (near-complete inverse) even an excellent prior is unlikely to clear hr>0. WDM's payoff for us would be on the **sparse-view** benchmarks or for volumetric synthesis, not dense Mayo.
