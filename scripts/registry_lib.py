"""Shared helpers for the result register (build_registry.py + validate_registry.py).

ONE place for: challenge_from_slug, the solver display-name map, the canonical
ranking (headroom desc, val_ssim tiebreak; discard / non-finite / hr<=0 excluded
from the rank but still rendered), and the trainable-param formulas. Both the
builder and the gate import from here so they can never disagree.

Torch-free. The numbers come exclusively from the immutable per-iter
`observation.json` records already on disk — no LLM, no cluster needed.

See result_register_refactor_plan.md §3 (data model) and §7 (decisions).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO / "docs" / "runs"
SCHEMA_VERSION = 3

# ---------------------------------------------------------------------------
# challenge <- slug  (NEVER the manifest/observation `challenge` field, which is
# hardcoded "dl_sparse_view" on every run). First matching prefix wins; the
# trailing ("demo-", demo_dl) catches the demo_dl families (demo-intensity-*,
# demo-fair-*) that don't carry the "demo-dl-" prefix.
# ---------------------------------------------------------------------------
_PREFIX_TO_CHALLENGE = [
    ("mayo-ldct", "mayo_ldct"),
    ("breast-ct", "breast_ct"),
    ("dl-sparse-view", "dl_sparse_view"),
    ("dl-spectral", "dl_spectral"),
    ("ct-mar", "ct_mar"),
    ("truect", "truect"),
    ("demo-dl", "demo_dl"),
    ("demo-", "demo_dl"),
]

DATASET_LABELS = {
    "mayo_ldct": "Mayo-LDCT", "breast_ct": "Breast-CT", "demo_dl": "Demo-DL",
    "dl_sparse_view": "DL-Sparse-View", "dl_spectral": "DL-Spectral",
    "ct_mar": "CT-MAR", "truect": "TrueCT",
}


def challenge_from_slug(slug: str, fallback: str | None = None) -> str | None:
    for prefix, ch in _PREFIX_TO_CHALLENGE:
        if (slug or "").startswith(prefix):
            return ch
    return fallback


def campaign_from_slug(slug: str) -> str:
    """The `search-YYYYMMDD-NN` tag (or the trailing `YYYYMMDD-NN` for the early
    runs that omit `-search`). Empty string if neither pattern matches."""
    m = re.search(r"(search-\d{8}-\d{2})$", slug or "")
    if m:
        return m.group(1)
    m = re.search(r"(\d{8}-\d{2})$", slug or "")
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Solver identity: a stable solver_key (family) + a human display name. The key
# is recovered from the slug by stripping the run-id, the `-search` tag, the
# harness infix (claude-agentic- / calibrated-tpe- / calibrated- / intensity-
# calibrated-tpe-), and the dataset dash-prefix. Used to dedupe to one row per
# solver family on the leaderboard.
# ---------------------------------------------------------------------------
def solver_key(slug: str) -> str:
    s = re.sub(r"-\d{8}-\d{2}$", "", slug or "")
    s = re.sub(r"-search$", "", s)
    for infix in ("claude-agentic-", "intensity-calibrated-tpe-",
                  "calibrated-tpe-", "calibrated-", "fair-"):
        i = s.find(infix)
        if i >= 0:
            s = s[i + len(infix):]
            break
    for dash, _ in _PREFIX_TO_CHALLENGE:
        if s.startswith(dash):
            s = s[len(dash):]
            break
    # normalise a few historical variant suffixes to one family key
    s = re.sub(r"-(breast|mayo)(-v\d)?$", "", s)
    s = re.sub(r"-v2$", "", s) if s.endswith("-zeroshot-v2") else s
    return s or slug


# Display names keyed by solver_key (dashed). Falls back to a title-cased key.
DISPLAY_NAMES = {
    "dual-domain-supervised": "DD-UNet supervised L2",
    "dual-domain-unet-l2": "DD-UNet supervised L2",
    "dual-domain-bilateral-supervised": "DD-BF supervised L2",
    "dual-domain-bf-l2": "DD-BF supervised L2",
    "dual-domain-n2i": "DD-UNet N2I (per-image)",
    "dual-domain": "DD-UNet N2I (per-image)",
    "dual-domain-bilateral-n2i": "DD-BF N2I (per-image)",
    "dual-domain-bf": "DD-BF N2I (per-image)",
    "itnet": "ITNet v1",
    "itnet-v2": "ITNet v2",
    "itnet-v3": "ITNet v3",
    "learned-primal-dual": "Learned Primal-Dual",
    "lpd": "Learned Primal-Dual",
    "hammernik": "Hammernik VN (2017)",
    "hammernik-2017": "Hammernik VN (2017)",
    "hammernik-vn": "Hammernik VN (MRI port)",
    "uswin": "U-Swin",
    "wu": "Wu 2015 (non-trainable)",
    "wu-2015-trainable": "Wu 2015 trainable",
    "wu-2015-l2": "Wu 2015 trainable",
    "ram": "RAM (zero-shot)",
    "ram-zeroshot": "RAM (zero-shot)",
    "naf": "NAF",
    "r2gaussian": "R2-Gaussian",
    "tv": "TV-iterative",
    "tv-iterative": "TV-iterative",
    "tv-v2": "TV-iterative",
    "tv-iterative-supervised": "TV-iterative (unrolled)",
    "diff-recon-dcstep-constrained": "Diffusion (constrained DPS+DC)",
    "diff-recon-dcstep-constrained-mayo-v4": "Diffusion (constrained DPS+DC)",
    "diff-recon-dcstep-unconstrained": "Diffusion (unconstrained DPS)",
    "diff-recon-dcstep-unconstrained-mayo-v4": "Diffusion (unconstrained DPS)",
    "fastdiff-flow-pixel-constrained": "Fast-diffusion flow (pixel, constrained DC)",
    "fastdiff-flow-pixel-unconstrained": "Fast-diffusion flow (pixel, unconstrained)",
    "fastdiff-wdm-wavelet-constrained": "Fast-diffusion WDM (wavelet, constrained DC)",
    "fastdiff-wdm-wavelet-unconstrained": "Fast-diffusion WDM (wavelet, unconstrained)",
    "manduca-bilateral": "Manduca proj-bilateral (trainable)",
    "manhart-pwls-tv": "Manhart PWLS-TV (ray-weighted)",
    "param-efficient": "Param-efficient (evolved)",
}


def display_name(slug: str) -> str:
    key = solver_key(slug)
    if key in DISPLAY_NAMES:
        return DISPLAY_NAMES[key]
    return key.replace("-", " ")


# ---------------------------------------------------------------------------
# Params resolution: observation.params_M -> trainable_from_cfg(cfg_full) ->
# solver_params backstop. Records which source won.  The bilateral/Wu formulas
# (verified) are the ONLY copy now (gen_mayo_leaderboard.py + solver_params.json
# are retired). Returns an INTEGER trainable-param count (not millions).
# ---------------------------------------------------------------------------
def trainable_from_cfg(key: str, cfg: dict):
    if not cfg:
        return None
    if key in ("dual-domain-bilateral-supervised", "dual-domain-bilateral-n2i",
               "dual-domain-bf", "dual-domain-bf-l2"):
        return 3 * (int(cfg.get("proj_n_bf", 1)) + int(cfg.get("img_n_bf", 1)))
    if key in ("wu-2015-trainable", "wu-2015-l2"):
        return int(cfg.get("wu_n_bands", 0)) + 2 + 2 * int(cfg.get("wu_n_outer", 0))
    return None


def resolve_params_M(key: str, obs: dict, backstop: dict, slug: str):
    """-> (params_M float|None, source str). params_M is in MILLIONS."""
    pm = obs.get("params_M")
    if isinstance(pm, (int, float)) and math.isfinite(pm):
        return float(pm), "observation"
    cnt = trainable_from_cfg(key, obs.get("cfg_full") or {})
    if cnt is not None:
        return cnt / 1e6, "trainable_from_cfg"
    if slug in backstop:
        return backstop[slug] / 1e6, "solver_params_backstop"
    return None, "none"


# ---------------------------------------------------------------------------
# The ONE canonical ranking. rank key = (-headroom, -val_ssim). A run is
# RANKABLE iff status != discard, headroom finite and > 0. Non-rankable runs get
# an excluded_reason and sort BELOW all rankable ones (rendered dimmed) — they
# are never sliced away, so every solver always shows (never top-N).
# ---------------------------------------------------------------------------
def excluded_reason(status: str, headroom, val_ssim) -> str | None:
    st = (status or "").strip().lower()
    if st == "discard":
        return "discard"
    if headroom is None or not math.isfinite(headroom):
        return "non-finite"
    if headroom <= 0:
        return "hr<=0"
    return None


def rank_sort_key(row: dict):
    """Sort rows: rankable first (headroom desc, val_ssim desc), excluded last
    (still by headroom/ssim so the ordering is stable & meaningful)."""
    excluded = 1 if row.get("excluded_reason") else 0
    hr = row.get("headroom")
    hr = hr if (isinstance(hr, (int, float)) and math.isfinite(hr)) else -math.inf
    ss = row.get("val_ssim")
    ss = ss if (isinstance(ss, (int, float)) and math.isfinite(ss)) else -math.inf
    return (excluded, -hr, -ss)


# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_json(p: Path):
    return json.loads(p.read_text())
