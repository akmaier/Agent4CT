# Pentathlon Solvers

Per-challenge solver directories plus reference implementations used to
benchmark the agentic loops in `docs/runs/`.

| Directory | Role |
|---|---|
| [`demo_dl_reference/`](demo_dl_reference/) | Classical and reference learned baselines (FBP, TV, Wu 2015, Dual-Domain, ItNet) for the DL-Sparse-View challenge. Includes both single-run references and 20-iter hyperparameter searches. **All quantitative results — search tables, head-to-head, hyperparameters — live in [demo_dl_reference/README.md](demo_dl_reference/README.md).** |
| `dl_sparse_view/` | Main DL-Sparse-View solver (Claude-driven, 150-iter campaign, NAFNet + SWA + Bilateral Filter). |
| `dl_sparse_view_iter/` | Agent A — unrolled-iterative NAFNet+BF variant. |
| `dl_sparse_view_res/` | Agent B — ResidualStack with batch-norm. |
| `dl_sparse_view_loss/` | Agent C — ResidualStack with augmentation + AdamW. |
| `dl_sparse_view_{deepseek,gptoss,kimi,mistral}/` | Non-Claude LLM solver attempts (none cleared baseline). |

## Where to find what

- **Reference / search results for DL-Sparse-View** → [`demo_dl_reference/README.md`](demo_dl_reference/README.md)
- **150-iter campaign histories** → `docs/runs/dl-sparse-view-*-20260513-*/`
- **Stage-check verdicts** → `docs/runs/<slug>/stages.tsv`
- **Live dashboard** → <https://akmaier.github.io/Agent4CT/dashboard.html>

## SSIM convention

Every solver in this tree uses the canonical implementation in
[`ddssl_ldct/metrics.py`](../ddssl_ldct/metrics.py):
- `C1 = C2 = 0` (no stabilisation constants — values are calibrated
  μ mm⁻¹, not 8-bit display).
- Compared against the **ground-truth phantom**, never the noiseless
  FBP reference.

This makes SSIM numbers directly comparable across `demo_dl_reference/`
and the Claude-driven 150-iter campaigns.

## Literature index

Reconstructions follow the per-paper derivations summarised in
[`../literature/`](../literature/):

- [Wu 2015 — Novel FBP for sparse-view CT](../literature/wu_2015_sparse_view_fbp.md)
- [Wagner 2022 — Trainable bilateral filter](../literature/2201.10345_Wagner_TrainableBilateralFilter_MedPhys2022.md)
- [Wagner 2023 — Dual-domain Noise2Inverse](../literature/2211.01111_Wagner_DualDomainDenoising_LDCT.md)
- [Sidky 2022 — DL-Sparse-View challenge](../literature/sidky_2022_dl_sparse_view_2109.09640.md)

---

*Last updated: 2026-05-16*
