"""Smoke test for the per-image N2I rewrite (not a result claim — tiny budget).

Runs both rewritten N2I solvers' main() on Mayo with a small config that
exercises: the warm-start pre-pass, the per-image fine-tune loop, the
per-sample-ps half-angle projector, and the evaluate_calibrated scoring +
comparison.png path. Prints val_score/headroom and asserts finiteness + figure.
Writes into docs/_n2i_smoke/ (browsable repo dir, NOT /tmp).
"""
import importlib.util
import math
from pathlib import Path

REPO = Path("/cluster/maier/Agent4CT")
out = REPO / "docs" / "_n2i_smoke"
out.mkdir(parents=True, exist_ok=True)

cfg = dict(warm_start=True, pretrain_steps=20, pretrain_epochs=1,
           n_iter=40, val_n=2, grad_clip=1.0, outer_wall_s=1500, seed=42)

ok = True
for name, f in [("n2i", "pentathlon/demo_dl_reference/solver_dual_ddomain_n2i.py"),
                ("bilat", "pentathlon/demo_dl_reference/solver_dual_ddomain_bilateral_n2i.py")]:
    spec = importlib.util.spec_from_file_location("s_" + name, REPO / f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = out / name
    d.mkdir(exist_ok=True)
    res = mod.main(d, dict(cfg))
    vs = res.get("val_score") if isinstance(res, dict) else None
    hr = res.get("headroom") if isinstance(res, dict) else None
    png = (d / "comparison.png").exists()
    scheme = res.get("training_scheme") if isinstance(res, dict) else None
    print(f"[SMOKE] {name}: val_score={vs} headroom={hr} scheme={scheme} png={png} -> {d}",
          flush=True)
    if vs is None or not math.isfinite(vs) or not png:
        ok = False

print("[SMOKE] ALL_OK" if ok else "[SMOKE] FAILED", flush=True)
