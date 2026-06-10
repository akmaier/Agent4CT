---
title: Leaderboards
description: Calibrated headroom rankings across the three CT reconstruction benchmarks — full 19-solver inventory exercised on each.
---

# Leaderboards

Best-of-best per solver per dataset under the canonical
**calibrated-SSIM-headroom** scoring convention
([`evaluate_calibrated`](https://github.com/akmaier/Agent4CT/blob/main/ddssl_ldct/metrics.py)).
All 3 leaderboards have **full 19-solver inventory coverage** as of
2026-06-09.

## Champion comparison images

The rank-1 solver on each dataset, click any image to open the full leaderboard:

| Dataset | Champion | hr | Visual |
|---|---|---:|---|
| **Mayo-LDCT** (real helical, Wagner split) | DD-UNet supervised L2 (Step-3 TPE iter-12) | **0.3890** | [![Mayo DD-UNet sup TPE](../runs/mayo-ldct-2d-calibrated-tpe-dual-domain-supervised-search-20260608-02/iterations/iter-0012/comparison.png)](mayo_ldct.html) |
| **Breast-CT** (Sidky synthetic + anatomy, 128 views) | Learned Primal-Dual (TPE iter-11) | **0.9062** | [![Breast-CT LPD TPE](../runs/breast-ct-calibrated-tpe-lpd-search-20260524-01/iterations/iter-0011/comparison.png)](breast_ct.html) |
| **Demo-DL** (Sidky synthetic ellipses, 128 views) | DD-UNet supervised L2 (TPE iter-15) | **0.4950** | [![Demo-DL DD-UNet sup TPE](../runs/demo-intensity-calibrated-tpe-dual-domain-supervised-search-20260601-01/iterations/iter-0015/comparison.png)](demo_dl.html) |

## Cross-dataset summary

| Dataset | Above baseline | Below baseline | Total | Coverage |
|---|---:|---:|---:|---|
| **Demo-DL** | 19 | 1 (TV-iter sup STOP) | 20 | **19/19 inventory** |
| **Breast-CT** | 16 | 13 | 29 | **19/19 inventory** |
| **Mayo-LDCT** | 12 | 10 + 1 deprioritised | 23 | **19/19 inventory** |

## Top 5 per dataset

### [Mayo-LDCT →](mayo_ldct.html)

| Rank | Solver | hr |
|---:|---|---:|
| 1 | DD-UNet supervised L2 *(Step-3 TPE)* | 0.3890 |
| 2 | Learned Primal-Dual *(Step-3 TPE)* | 0.3063 |
| 3 | USwin *(Step-3 TPE)* | 0.2492 |
| 4 | diff_recon DCstep unconstrained (DDPM v4) *(Step-3 TPE)* | 0.2377 |
| 5 | diff_recon DCstep unconstrained (DDPM v2) *(Step-3 TPE)* | 0.2352 |

### [Breast-CT →](breast_ct.html)

| Rank | Solver | hr |
|---:|---|---:|
| 1 | Learned Primal-Dual *(TPE)* | 0.9062 |
| 2 | DD-UNet supervised L2 *(TPE)* | 0.8361 |
| 3 | LPD (agentic seed) | 0.8290 |
| 4 | DD-UNet supervised L2 (agentic seed) | 0.8120 |
| 5 | ITNet v3 *(TPE)* | 0.7342 |

### [Demo-DL →](demo_dl.html)

| Rank | Solver | hr |
|---:|---|---:|
| 1 | DD-UNet supervised L2 *(TPE)* | 0.4950 |
| 2 | Learned Primal-Dual *(TPE)* | 0.4947 |
| 3 | ITNet v3 *(TPE)* | 0.4676 |
| 4 | **ItNet v1** *(Step-3 TPE — 2026-06-09 inventory-gap closure)* | 0.4665 |
| 5 | USwin *(TPE)* | 0.4655 |

## Recent additions (2026-06-08/09 session)

- **Mayo Step-3 TPE phase 1** lifted top-4 plateaued positives by
  +25% to +191%: DD-UNet sup 0.1337 → 0.3890 (+191%), LPD 0.2445 →
  0.3063 (+25%), USwin 0.1425 → 0.2492 (+75%), ItNet v3 0.1336 →
  0.2181 (+63%).
- **Mayo Step-3 TPE phase 2** overturned Hammernik VN's Step-2 STOP:
  found a working corner at hr=0.0551 (vn_T=5, n_filters=16,
  kernel=11, λ_init=2.3e-3). The first and only TPE-rescued STOP.
- **Mayo Step-3 TPE phase 3** discovered a previously-unexplored
  very-low-eta corner for diff_recon: UNCON modes converge at
  eta≈0.3, CON modes at mid-eta=1.5-7. Lifted UNCON v4 by +37%.
- **Inventory gap closure**: ItNet v1 cleared baseline on demo-DL
  (rank 4, 0.4665) AND breast-CT (rank 15, 0.1703), surprising the
  Mayo hr=0 verdict. Wu trainable on breast-CT lifted +45% via TPE
  (0.2189 → 0.3170). TV-iter supervised confirmed STOP on all 3
  datasets (FBP-init no-op verdict, only solver hr=0 everywhere).

## Methodology

See [`solver_plan.md`](https://github.com/akmaier/Agent4CT/blob/main/solver_plan.md)
for the canonical methodology — calibrated metric, per-solver
hyperparameter spaces, autoresearch+TPE protocol.

Per-solver design docs and cross-dataset transfer records:
[`pentathlon/demo_dl_reference/`](https://github.com/akmaier/Agent4CT/tree/main/pentathlon/demo_dl_reference/).
Every canonical-19 solver has a dedicated `.md` design doc with
cross-dataset hr record, CONFIG defaults, and "hints for the next
autoresearch agent".

Cross-cutting findings (DDPM training quality NOT predictive of DPS
performance, FBP-init 1st-step-no-op mechanism, etc.) live in
[`docs/findings.md`](../findings.html).
