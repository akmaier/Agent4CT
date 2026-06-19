---
title: Agent4CT
description: Agentic autoresearch for CT reconstruction — five challenges, one Pentathlon.
---

**Agent4CT** is a continuously-running LLM agent that improves a CT reconstruction codebase by editing it, running short experiments on a Slurm GPU cluster, and keeping or discarding changes based on the resulting metrics. The pattern is borrowed from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and generalised to **five** CT-imaging benchmarks — the **Pentathlon**.

<div class="a4c-callout">
<strong>Live</strong> · The dashboard at <a href="dashboard.html">dashboard.html</a> shows every run, every iteration, and the agents' shared scratch pad — including the side-by-side reconstruction images for each iteration.
</div>

<div class="a4c-callout warn">
<strong>Active rebuild</strong> · The <strong>Mayo-LDCT</strong> leaderboard (real helical data, Wagner split) is being rebuilt from scratch on the LME cluster — 19 solvers driven to the iter-20 hard stop. Per-dataset progress (Mayo-LDCT, Breast-CT, Demo-DL) is on the <a href="dashboard.html">dashboard</a>.
</div>

## The Pentathlon

Five public AAPM CT reconstruction challenges, run on the same recon backbone (PYRO-NN fan-beam, Wagner / Siemens-AS geometry):

<div class="a4c-grid">
  <div class="card">
    <h3>Mayo LDCT</h3>
    <p>AAPM Low Dose CT Grand Challenge (2016) — abdomen / chest / head, 25 % low-dose.</p>
    <p class="stat">Metric: PSNR + SSIM vs high-dose recon</p>
  </div>
  <div class="card">
    <h3>DL-Sparse-View CT</h3>
    <p>2021 — 2D breast phantom, 128-view sparse fan-beam, perfectly-known truth.</p>
    <p class="stat">Metric: RMSE vs exact phantom</p>
  </div>
  <div class="card">
    <h3>TrueCT</h3>
    <p>Truth-based CT (2022) — 200 virtual patients, mono-energetic phantom truth.</p>
    <p class="stat">Metric: PSNR + RMSE vs mono-energetic truth</p>
  </div>
  <div class="card">
    <h3>CT-MAR</h3>
    <p>CT Metal Artifact Reduction (2024) — sinogram + image pairs, 8-metric IQ score.</p>
    <p class="stat">Metric: aggregate clinical IQ</p>
  </div>
  <div class="card">
    <h3>DL-Spectral CT</h3>
    <p>Spectral CT (2022) — 1000 cases, two kVp, three-tissue map decomposition.</p>
    <p class="stat">Metric: tissue-map RMSE</p>
  </div>
</div>

See [Pentathlon](pentathlon.html) for the headroom-recovered scoring and per-challenge train/val/test splits.

## How the loop runs

Each iteration is one 5-minute Slurm job. The agent edits `pentathlon/<challenge>/solver.py`, the harness runs the job, computes the validation metric, and decides keep / discard via `git`. Every 30 iterations a 1-hour **stage** job runs on a 3× larger subset to catch overfitting. Test sets are touched exactly once, at the end. See [Agents](agents.html).

## 🏆 Leaderboards

Best-of-best per solver per dataset under the canonical
calibrated-SSIM-headroom scoring convention.

The champion per dataset (single canonical ranking = **headroom**, SSIM
tiebreak) is rendered live from the registry below, so it never drifts from the
dataset boards.

<div data-champions data-link-base="leaderboards/">loading champions…</div>

<script src="assets/table.js"></script>
<script src="assets/leaderboard.js"></script>

See [Leaderboards](leaderboards/) for the full per-solver rankings (every solver,
below-baseline dimmed, with comparison images) and the cross-dataset summary.

## Quick links

- 🏆 [Leaderboards](leaderboards/) — calibrated headroom rankings per dataset
- 📊 [Live dashboard](dashboard.html) — every run, every iteration, scratch pad with images
- 🧪 [Setup](setup.html) — env, cluster, data
- 🥇 [Pentathlon](pentathlon.html) — challenges + scoring
- 🧠 [Agents](agents.html) — the autoresearch loop
- ⚡ [Performance](performance.html) — PYRO-NN + NFS tips
- 📓 [Findings](findings.html) — cross-cutting insights from the autoresearch loop
- 📦 [GitHub](https://github.com/akmaier/Agent4CT)
