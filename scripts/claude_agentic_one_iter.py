"""Claude-driven agentic search: run ONE iter of a solver with a chosen config,
then record the observation in the dashboard's standard layout.

For each hr=0 breast-ct solver, Claude proposes the next-iter config based on
the prior iter history + the user's hint, dispatches one iter via this driver,
reads the result, and iterates. Slug naming: `breast-ct-claude-agentic-<solver>-search-<DATE>-NN`.

Usage:
    AGENT4CT_DATASET=breast_ct \
    SLUG=breast-ct-claude-agentic-ram-zeroshot-search-20260521-01 \
    ITER_N=1 \
    SOLVER=ram_zeroshot \
    CFG_JSON=/tmp/iter_1_cfg.json \
    python scripts/claude_agentic_one_iter.py
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS_RUNS = REPO / "docs" / "runs"
RUNS_BASE = Path(os.environ.get("AGENT4CT_RUNS", "/cluster/maier/Agent4CT/runs"))

SOLVER_MAP = {
    "ram_zeroshot":         ("pentathlon/demo_dl_reference/solver_ram.py",          "RAM_CONFIG_PATH"),
    "tv_iterative":         ("pentathlon/demo_dl_reference/solver_tv_search.py",     "TV_CONFIG_PATH"),
    # Code-evolving parameter-efficient campaign (search-20260624-01): each iter
    # REWRITES the architecture (dispatch sets SOLVER_SRC_OVERRIDE to a per-iter
    # snapshot). solver_param_efficient.py is the canonical/seed path.
    "param_efficient":      ("pentathlon/demo_dl_reference/solver_param_efficient.py", "PARAM_EFFICIENT_CONFIG_PATH"),
    # Noise2Inverse self-supervised dual-domain variants (file names made
    # explicit 2026-05-22 — see docs/findings.md re: N2I vs supervised L2).
    "dual_domain_n2i":           ("pentathlon/demo_dl_reference/solver_dual_ddomain_n2i.py",            "DD_CONFIG_PATH"),
    "dual_domain_bilateral_n2i": ("pentathlon/demo_dl_reference/solver_dual_ddomain_bilateral_n2i.py", "DD_CONFIG_PATH"),
    # Back-compat aliases — older agentic configs use these short keys.
    "dual_domain":          ("pentathlon/demo_dl_reference/solver_dual_ddomain_n2i.py",            "DD_CONFIG_PATH"),
    "dual_domain_bilateral": ("pentathlon/demo_dl_reference/solver_dual_ddomain_bilateral_n2i.py", "DD_CONFIG_PATH"),
    # Supervised-L2 variants (2026-05-22): full 128 views, MSE against clean
    # phantom instead of Noise2Inverse self-supervision. Built to test
    # whether N2I's noise-floor was what was pulling the dual-domain
    # solvers to over-smooth on dense breast scans (yes, it was — see
    # docs/findings.md).
    "dual_domain_bilateral_supervised": (
        "pentathlon/demo_dl_reference/solver_dual_ddomain_bilateral_supervised.py", "DD_CONFIG_PATH"),
    "dual_domain_supervised": (
        "pentathlon/demo_dl_reference/solver_dual_ddomain_supervised.py", "DD_CONFIG_PATH"),
    # Manduca 2009 projection-space bilateral with CT noise model (2026-06-23):
    # variance-stabilized (N0*exp(-P) -> sqrt) sinogram bilateral so the range
    # bandwidth is implicitly noise-adaptive; trainable end-to-end on top of the
    # Wagner DD-BF supervised recipe. Paper: papers/Medical Physics - 2009 - Manduca.
    "manduca_bilateral": (
        "pentathlon/demo_dl_reference/solver_manduca_bilateral.py", "DD_CONFIG_PATH"),
    # Manhart 2013 statistical ray weighting (PWLS) on top of TV-iterative
    # (2026-06-23): inverse-variance ray weight W=exp(-gamma*p) inserted into the
    # TV data residual (gamma=0 -> plain TV). Paper: papers/Manhart13-DRW.pdf.
    "manhart_pwls_tv": (
        "pentathlon/demo_dl_reference/solver_manhart_pwls_tv.py", "TV_CONFIG_PATH"),
    # Fast few-step diffusion-prior recon (solver_fast_recon.py) — 4 variants
    # (pixel/wdm-wavelet x constrained/unconstrained) selected by the per-run
    # cfg fd_recon_ckpt + fd_recon_dc. Onboarded one-shot in search-20260620-01;
    # joined the generic agentic loop 2026-06-23 (search-20260620-01 continues).
    "fastdiff_flow_pixel_unconstrained":   ("pentathlon/demo_dl_reference/solver_fast_recon.py", "FAST_RECON_CONFIG_PATH"),
    "fastdiff_flow_pixel_constrained":     ("pentathlon/demo_dl_reference/solver_fast_recon.py", "FAST_RECON_CONFIG_PATH"),
    "fastdiff_wdm_wavelet_unconstrained":  ("pentathlon/demo_dl_reference/solver_fast_recon.py", "FAST_RECON_CONFIG_PATH"),
    "fastdiff_wdm_wavelet_constrained":    ("pentathlon/demo_dl_reference/solver_fast_recon.py", "FAST_RECON_CONFIG_PATH"),
    # End-to-end-trainable Wu 2015 (2026-05-22): per-band scales,
    # sigmoid slope/offset, per-iter soft thresholds + residual blends
    # wrapped as nn.Parameter and trained supervised-L2 vs clean phantom.
    "wu_2015_trainable": (
        "pentathlon/demo_dl_reference/solver_wu_2015_trainable.py", "WU_CONFIG_PATH"),
    # Learned Primal-Dual (Adler & Öktem 2018, IEEE TMI; literature/
    # 1707.06474). Unrolled PDHG with CNN proximals + PYRO-NN
    # forward/back-projector kept differentiable inside the network.
    "learned_primal_dual": (
        "pentathlon/demo_dl_reference/solver_learned_primal_dual.py", "LPD_CONFIG_PATH"),
    # Per-scene implicit-field / splat reconstructions (2026-05-23):
    # both are stuck at hr=0 in TPE searches because the TPE bounds
    # capped n_iter way below what the original papers use. Agentic
    # first move: bump n_iter ~5-25x with smaller val_n.
    "naf":         ("pentathlon/demo_dl_reference/solver_naf.py",        "NAF_CONFIG_PATH"),
    "r2gaussian":  ("pentathlon/demo_dl_reference/solver_r2gaussian.py", "R2G_CONFIG_PATH"),
    # Unrolled TV-GD with per-iter learnable step + lambda (2026-05-23):
    # supervised-L2 mirror of solver_tv_iterative.py (non-trainable) and
    # solver_tv_search.py (TPE-searched). 10 unrolled iters × 2 scalars
    # = 20 trainable params. Mirrors the DD-BF / DD-UNet supervised-L2
    # recipe (docs/findings.md 2026-05-22) on a TV iteration instead of
    # a U-Net or BF chain.
    "tv_iterative_supervised": (
        "pentathlon/demo_dl_reference/solver_tv_iterative_supervised.py",
        "TV_SUP_CONFIG_PATH"),
    # Image-domain transformers / unrolled networks (added 2026-06-03 for
    # the Mayo-LDCT autoresearch rollout — all three were used in TPE
    # searches but were not yet wired into the agentic dispatcher).
    "uswin":
        ("pentathlon/demo_dl_reference/solver_uswin.py", "USWIN_CONFIG_PATH"),
    # All three ItNet solvers read ITNET_CONFIG_PATH (2026-06-08 audit found
    # the v3/v2 env-var names in this map didn't match what the solvers
    # actually os.environ.get() — silently dropping the agentic cfg and
    # falling back to the hardcoded defaults that OOMed on Mayo).
    "itnet_v3":
        ("pentathlon/demo_dl_reference/solver_itnet_v3.py", "ITNET_CONFIG_PATH"),
    "itnet_v2":
        ("pentathlon/demo_dl_reference/solver_itnet_v2.py", "ITNET_CONFIG_PATH"),
    "hammernik_vn":
        ("pentathlon/demo_dl_reference/solver_hammernik_vn.py", "HAMMERNIK_VN_CONFIG_PATH"),
    "hammernik":
        ("pentathlon/demo_dl_reference/solver_hammernik_2017.py", "HAMMERNIK_CONFIG_PATH"),
    # Final two solver_plan.md inventory entries (added 2026-06-07 for Mayo
    # coverage audit): ItNet v1 (the original 5-iter version that the v2/v3
    # variants superseded) and Wu 2015 non-trainable (frozen filter-band
    # modulation FBP, no params — the baseline that the trainable variant
    # extends).
    "itnet":  # solver_itnet.py also reads ITNET_CONFIG_PATH after the 2026-06-08 patch
        ("pentathlon/demo_dl_reference/solver_itnet.py", "ITNET_CONFIG_PATH"),
    "wu_2015":
        ("pentathlon/demo_dl_reference/solver_wu_2015.py", "WU_CONFIG_PATH"),
    # Hammernik 2017 image-space variational network (Mayo rebuild 2026-06-14):
    # per-sample-ps-wired; image-domain denoiser of the FBP (no in-model proj).
    "hammernik_2017":
        ("pentathlon/demo_dl_reference/solver_hammernik_2017.py", "HAMMERNIK_CONFIG_PATH"),
    # Diffusion posterior sampling with a Mayo-trained DDPM prior
    # (2026-06-07). Two variants share the same solver and env var; the
    # ckpt path is passed inside the per-iter CFG_JSON as `recon_ckpt`.
    # Architecture: SmallDDPM ch=64 / batch=2 at 512² (job 762815 unconstrained,
    # 762819 constrained) — the breast v3 architecture (ch=128) OOMed on
    # Mayo's 4× larger images.
    "diffusion_recon_dcstep_unconstrained_mayo_v2": (
        "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "DIFFUSION_RECON_CONFIG_PATH"),
    "diffusion_recon_dcstep_constrained_mayo_v2": (
        "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "DIFFUSION_RECON_CONFIG_PATH"),
    # Diff_recon against the v3 Mayo DDPM ckpts (ch=96, batch=1, 8.594 M
    # params — 2.25× v2's 3.823 M). Trained 2026-06-07. Bet: bigger prior
    # lifts UNCON beyond v2's hr=0.2095 plateau.
    "diffusion_recon_dcstep_unconstrained_mayo_v3": (
        "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "DIFFUSION_RECON_CONFIG_PATH"),
    "diffusion_recon_dcstep_constrained_mayo_v3": (
        "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "DIFFUSION_RECON_CONFIG_PATH"),
    # diff_recon against the v4 Mayo DDPM ckpts (ch=96, batch=2, ep=120 —
    # 4× v3's effective training because v3 was under-trained at batch=1).
    # Trained 2026-06-08, unconstrained ckpt landed at 12:49.
    "diffusion_recon_dcstep_unconstrained_mayo_v4": (
        "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "DIFFUSION_RECON_CONFIG_PATH"),
    "diffusion_recon_dcstep_constrained_mayo_v4": (
        "pentathlon/demo_dl_reference/solver_diffusion_recon.py",
        "DIFFUSION_RECON_CONFIG_PATH"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Challenge derived from the SLUG prefix — NOT hardcoded "dl_sparse_view". Mirrors
# scripts/registry_lib.challenge_from_slug (kept inline so this cluster-side
# recorder has no extra import dependency). First matching prefix wins.
_PREFIX_TO_CHALLENGE = [
    ("mayo-ldct", "mayo_ldct"), ("breast-ct", "breast_ct"),
    ("dl-sparse-view", "dl_sparse_view"), ("dl-spectral", "dl_spectral"),
    ("ct-mar", "ct_mar"), ("truect", "truect"),
    ("demo-dl", "demo_dl"), ("demo-", "demo_dl"),
]


def challenge_from_slug(slug: str, fallback: str = "dl_sparse_view") -> str:
    for prefix, ch in _PREFIX_TO_CHALLENGE:
        if (slug or "").startswith(prefix):
            return ch
    return fallback


def git_head_sha() -> str:
    """Short HEAD sha for the commit column (was written empty). Empty on failure
    (e.g. detached/no-git checkout) — never raises."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> int:
    slug    = os.environ["SLUG"]
    iter_n  = int(os.environ["ITER_N"])
    solver_key = os.environ["SOLVER"]
    cfg_json = Path(os.environ["CFG_JSON"])
    if solver_key not in SOLVER_MAP:
        raise ValueError(f"unknown solver {solver_key!r}; choices: {list(SOLVER_MAP)}")
    solver_path, env_var = SOLVER_MAP[solver_key]
    # Code-evolving campaigns rewrite the solver each iteration. SOLVER_SRC_OVERRIDE
    # points the driver at a per-iter solver source snapshot (so parallel
    # architecture variants never collide and each iter is reproducible from its
    # own code); default is the canonical SOLVER_MAP path.
    _src_override = os.environ.get("SOLVER_SRC_OVERRIDE", "").strip()
    solver_src = Path(_src_override) if _src_override else (REPO / solver_path)

    # Provenance from the slug + env (was hardcoded to the breast-CT bring-up
    # values). AGENT / MODEL / TRAIN_N override the defaults so Mayo/demo runs
    # carry the right identity in observation.json + results.tsv.
    challenge = challenge_from_slug(slug)
    agent = os.environ.get("AGENT", f"{solver_key}-claude-agentic")
    model = os.environ.get("MODEL", "claude-opus-4-8-1m")
    train_n = int(os.environ.get("TRAIN_N", "400"))
    head_sha = git_head_sha()

    # Per-iter working dir (raw solver output goes here).
    iter_dir = RUNS_BASE / f"{slug}-iter-{iter_n:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Dashboard slug dir.
    dash_slug = DOCS_RUNS / slug
    dash_iter = dash_slug / "iterations" / f"iter-{iter_n:04d}"
    dash_iter.mkdir(parents=True, exist_ok=True)

    # Snapshot the exact solver source for this iter (provenance; essential for
    # code-evolving campaigns where the architecture itself changes per iter).
    try:
        shutil.copyfile(solver_src, dash_iter / "solver_src.py")
    except Exception as _e:
        print(f"[agentic] solver-src snapshot skipped: {_e}", flush=True)

    # Make sure the manifest exists; create on first iter.
    manifest_path = dash_slug / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "slug": slug, "challenge": challenge,
            "slug_prefix": "-".join(slug.split("-")[:-2]),
            "started": utc_now_iso(),
            "agent": agent,
            "model": model,
            "status": "running",
            "notes": "Claude-driven per-iter agentic search; bounds informed by user-provided hints.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # results.tsv header on first iter. New trailing columns (psnr/rmse/elapsed_s/
    # params_M) make the tsv self-sufficient for the leaderboard without the
    # observation.json round-trip; the registry builder still prefers the obs.
    tsv_path = dash_slug / "results.tsv"
    if not tsv_path.exists():
        tsv_path.write_text("iter\tcommit\tval_score\theadroom\tstatus\tchange_class\t"
                            "agent\tmodel\tval_psnr\tval_rmse\telapsed_s\tparams_M\trationale\n")

    # Run the solver with the chosen config.
    cfg = json.loads(cfg_json.read_text())
    print(f"[agentic] iter={iter_n}  solver={solver_key}  cfg={cfg}", flush=True)
    os.environ[env_var] = str(cfg_json)
    t0 = time.time()
    import subprocess
    cmd = [sys.executable, str(solver_src), str(iter_dir)]
    res = subprocess.run(cmd, env=os.environ.copy())
    if res.returncode != 0:
        print(f"[agentic] solver returned exit code {res.returncode}", flush=True)
        # Still record so we know it crashed.
    elapsed = time.time() - t0

    # Read solver's result.json + copy comparison.png.
    rj = iter_dir / "result.json"
    if rj.exists():
        result = json.loads(rj.read_text())
    else:
        result = {"val_score": None, "headroom": 0.0, "error": "no result.json"}
    cmp_src = iter_dir / "comparison.png"
    if cmp_src.exists():
        shutil.copyfile(cmp_src, dash_iter / "comparison.png")

    # Build observation.json (the per-iter dashboard record).
    rationale_keys = list(cfg.keys())
    rationale = f"{solver_key}-claude-agentic: " + ", ".join(
        f"{k}={cfg[k]}" for k in rationale_keys
    )
    obs = {
        "ts": utc_now_iso(),
        "run_id": slug,
        "iter": iter_n,
        "challenge": challenge,
        "commit": head_sha,
        "change_class": "architecture",
        "rationale": rationale,
        "val_score":   result.get("val_score"),
        "headroom":    result.get("headroom") or 0.0,
        "val_ssim":    result.get("val_ssim"),
        "val_psnr":    result.get("val_psnr"),
        "val_rmse":    result.get("val_rmse"),
        "val_ssim_std": result.get("val_ssim_std"),
        "val_psnr_std": result.get("val_psnr_std"),
        "val_rmse_std": result.get("val_rmse_std"),
        "kept":        (result.get("headroom") or 0.0) > 0,
        "status":      "keep" if (result.get("headroom") or 0.0) > 0 else "discard",
        "params_M":    result.get("params_M"),
        "train_n":     train_n,
        "val_n":       result.get("val_n", 100),
        "agent":       agent,
        "model":       model,
        "advice_for_others": rationale,
        "comparison_image": f"runs/{slug}/iterations/iter-{iter_n:04d}/comparison.png",
        "elapsed_s": elapsed,
        "cfg_full":  cfg,
    }
    (dash_iter / "observation.json").write_text(json.dumps(obs, indent=2))

    # Append results.tsv row (with the git SHA in the commit column + the
    # psnr/rmse/elapsed_s/params_M metrics in their new trailing columns).
    def _g(x, fmt="{:.6g}"):
        return fmt.format(x) if isinstance(x, (int, float)) else ""
    val_s = _g(obs["val_score"])
    hr_s  = _g(obs["headroom"])
    psnr_s = _g(obs["val_psnr"])
    rmse_s = _g(obs["val_rmse"], "{:.6e}")
    elapsed_s = _g(obs["elapsed_s"], "{:.1f}")
    params_s = _g(obs["params_M"])
    with tsv_path.open("a") as f:
        f.write(
            f"{iter_n}\t{head_sha}\t{val_s}\t{hr_s}\t{obs['status']}\t{obs['change_class']}\t"
            f"{obs['agent']}\t{obs['model']}\t{psnr_s}\t{rmse_s}\t{elapsed_s}\t{params_s}\t{rationale}\n"
        )

    print(f"[agentic] iter {iter_n} done  ssim={val_s}  hr={hr_s}  elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
