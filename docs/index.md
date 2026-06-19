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

| Dataset | Top solver | hr | Leaderboard |
|---|---|---:|---|
| **Breast-CT** (Sidky synthetic + anatomy, 128 views) | Learned Primal-Dual *(TPE iter-11)* | **0.9062** | [breast_ct](leaderboards/breast_ct.html) |
| **Demo-DL** (Sidky synthetic ellipses, 128 views) | DD-UNet supervised L2 *(TPE iter-15)* | **0.4950** | [demo_dl](leaderboards/demo_dl.html) |
| **Mayo-LDCT** (real helical, Wagner split) | DD-UNet supervised L2 _(live, iter-2/20)_ | **0.3390** | [mayo_ldct](leaderboards/mayo_ldct.html) |

[![Breast-CT champion](runs/breast-ct-calibrated-tpe-lpd-search-20260524-01/iterations/iter-0011/comparison.png)](leaderboards/breast_ct.html)
[![Demo-DL champion](runs/demo-intensity-calibrated-tpe-dual-domain-supervised-search-20260601-01/iterations/iter-0015/comparison.png)](leaderboards/demo_dl.html)

*Top: Breast-CT champion (Learned Primal-Dual, TPE iter-11 — hr 0.9062).
Bottom: Demo-DL champion (DD-UNet supervised L2, TPE iter-15 — hr 0.4950).
Click any leaderboard link for the full per-solver ranking with comparison images.
**Mayo-LDCT is being rebuilt (2026-06-14)** from a clean HD/LD FBP baseline after
the `bg→0` calibration-bug fix — see its leaderboard for status.*

See [Leaderboards](leaderboards/) for the cross-dataset summary and
per-solver rankings with comparison images.

## Quick links

- 🏆 [Leaderboards](leaderboards/) — calibrated headroom rankings per dataset
- 📊 [Live dashboard](dashboard.html) — every run, every iteration, scratch pad with images
- 🧪 [Setup](setup.html) — env, cluster, data
- 🥇 [Pentathlon](pentathlon.html) — challenges + scoring
- 🧠 [Agents](agents.html) — the autoresearch loop
- ⚡ [Performance](performance.html) — PYRO-NN + NFS tips
- 📓 [Findings](findings.html) — cross-cutting insights from the autoresearch loop
- 📦 [GitHub](https://github.com/akmaier/Agent4CT)
