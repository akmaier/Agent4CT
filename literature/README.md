# Literature

Offline-friendly markdown copies of every external reference Agent4CT
depends on. Each document opens with a `_Source: <url>_` line pointing
back to the canonical version — please cite the original when publishing.

## Tutorials

| File | What it is | Source |
|---|---|---|
| [`artifact_gallery.md`](artifact_gallery.md) | CONRAD CT artefact catalogue (detector shift, flower / cupping, filter discretisation, limited-angle). Useful as a visual sanity-check when a recon pipeline misbehaves. | <https://www5.cs.fau.de/conrad/tutorials/artifact-gallery/> |
| [`conrad_api_tutorials.md`](conrad_api_tutorials.md) | The full CONRAD API tutorials tree (basic / reconstruction / advanced + sub-pages). Helpful when porting geometry / projector conventions between CONRAD and PYRO-NN. | <https://www5.cs.fau.de/conrad/tutorials/api-tutorials/> |

Companion image folders:

- [`artifact_gallery_images/`](artifact_gallery_images) — 11 figures
- [`conrad_api_tutorials_images/`](conrad_api_tutorials_images) — 39 figures

## Papers

Recon backbone & agentic framework:

| File | Citation |
|---|---|
| [`2604.13282_Agent4MR.md`](2604.13282_Agent4MR.md) | Zaiss, Aly, …, Maier — *Agent4MR: Agentic MR sequence development with LLMs* (2026). <https://arxiv.org/abs/2604.13282> |
| [`2201.10345_Wagner_TrainableBilateralFilter_MedPhys2022.md`](2201.10345_Wagner_TrainableBilateralFilter_MedPhys2022.md) | Wagner *et al.* — *Ultra-low Parameter Denoising: Trainable Bilateral Filter Layers in CT* (Med. Phys. 2022). <https://arxiv.org/abs/2201.10345> |
| [`2211.01111_Wagner_DualDomainDenoising_LDCT.md`](2211.01111_Wagner_DualDomainDenoising_LDCT.md) | Wagner *et al.* — *On the Benefit of Dual-domain Denoising in a Self-Supervised LDCT Setting* (ISBI 2023). <https://arxiv.org/abs/2211.01111> |
| [`2112.03678_Maier_ProprietaryDECT_BVM2022.md`](2112.03678_Maier_ProprietaryDECT_BVM2022.md) | Maier *et al.* — *Does Proprietary Software Still Offer Protection of Intellectual Property in the Age of Machine Learning? — A Case Study using Dual Energy CT Data* (BVM 2022). Cite when discussing SSIM-grade vendor-algorithm approximation. <https://arxiv.org/abs/2112.03678> |

Pentathlon challenges (one report per challenge):

| File | Challenge | Citation |
|---|---|---|
| [`mccollough_2017_mayo_ldct.md`](mccollough_2017_mayo_ldct.md) | Mayo LDCT 2016 | McCollough *et al.* — *Results of the 2016 Low Dose CT Grand Challenge* (Med. Phys. 2017). <https://pmc.ncbi.nlm.nih.gov/articles/PMC5656004/> |
| [`sidky_2022_dl_sparse_view_2109.09640.md`](sidky_2022_dl_sparse_view_2109.09640.md) | DL-Sparse-View CT 2021 | Sidky & Pan — *Report on the AAPM DL-Sparse-View CT Grand Challenge* (Med. Phys. 2022). <https://arxiv.org/abs/2109.09640> |
| [`abadi_2025_truect.md`](abadi_2025_truect.md) | TrueCT 2022 | Abadi *et al.* — *AAPM Truth-based CT (TrueCT) Reconstruction Grand Challenge* (Med. Phys. 2025). <https://pmc.ncbi.nlm.nih.gov/articles/PMC11973969/> |
| [`haneda_2025_ctmar.md`](haneda_2025_ctmar.md) | CT-MAR 2024 | Haneda *et al.* — *AAPM CT Metal Artifact Reduction Grand Challenge* (Med. Phys. 2025). <https://pmc.ncbi.nlm.nih.gov/articles/PMC12757780/> |
| [`sidky_2024_dl_spectral_2212.06718.md`](sidky_2024_dl_spectral_2212.06718.md) | DL-Spectral CT 2022 | Sidky & Pan — *Report on the AAPM DL-Spectral CT Grand Challenge* (Med. Phys. 2024). <https://arxiv.org/abs/2212.06718> |

## Notes on the conversion

- Web pages were crawled and rendered with `html2text` after stripping the
  TYPO3 navigation chrome (banners, breadcrumbs, mail/UnivIS footers). Inline
  figures are downloaded into the image subfolder and referenced by relative
  path so the markdown renders correctly on GitHub or in any local viewer.
- PDFs were rendered with [`pymupdf4llm`](https://pypi.org/project/pymupdf4llm/),
  which preserves section structure and figure captions; figures themselves
  are summarised by their OCR / alt-text rather than re-embedded, so the
  markdown stays small and reviewable.
- The build script is at [`scripts/build_literature.py`](../scripts/build_literature.py).
  Run `python scripts/build_literature.py` (sections: `gallery api papers`)
  to rebuild from scratch if the upstream pages or PDFs change.
- The original PDFs live under [`papers/`](../papers/) and are gitignored —
  the markdowns under `literature/` are the version-controlled record.
