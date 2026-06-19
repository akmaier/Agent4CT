#!/usr/bin/env python -u
"""v4b — identify ONE physical Mayo-LDCT geometry by projection-domain forward matching.

Why (findings.md 2026-06-12 v4 entry): the v3 production stack needs two
unphysical degrees of freedom — a different (sod, sdd) for the SSR rebin
than for the FBP, and a z-scaling s_z=1.001665 — to reach SSIM 0.957.
Tag forensics (2026-06-12 probe) show why effective-parameter soup was
inevitable:

  * dv tag 1.0947227 == 0.6 mm x (1085.6/595.0) to 7 digits — the row
    pitch is DERIVED from the nominal magnification, not measured.
  * per-readout z tags are machine-precision uniform (synthesized) and
    encode pitch 30.6567 mm/rev, while the AS+ design pitch is
    0.8 x 38.4 = 30.72 mm/rev. v3's s_z lands at 30.708.

This script fits the geometry ONCE, in projection domain:

    truth volume (154 B30f slices as mu) --differentiable ray casting-->
        simulated curved-detector helical line integrals
    vs measured DICOM-CT-PD projections (native curved, no rebin)

v4b fixes after run 1 (SLURM 763576, FAILED structurally):
  * Run 1 rotated the source around patient-frame (0,0) while the volume
    sat at its patient-coords centre — but the scanner iso is at
    DataCollectionCenterPatient (L014: (0, -136.5) mm). 136.5 mm of
    baked-in y-misalignment meant NO convention combo could win and the
    optimizer collapsed dv 1.09->0.65, dy->+94. v4b works in the
    ISO-CENTRED frame (volume placed at truth_centre - iso).
  * Staged release: stage 1 fits rigid+scale-z only (sod, sdd, s_z, z0,
    phi0, dx, dy); dv / u0 / v0 / s_xy unfreeze in stage 2. Prevents the
    shrink-the-cone collapse.
  * Contiguous channel blocks + channel-gradient loss term (geometry
    lives in edges; robust to B30f-kernel intensity transfer).
  * float32 ray grids + chunked final eval (run 1 OOM'd at eval).

Output:
    results/breast_debug/L014_forward_geom_fit_v4.json
    results/mayo_debug/L014_forward_geom_fit_v4.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ddssl_ldct.helix2fan import read_dicom_ctpd
from scripts.fit_rebin_end2end_L014 import _list_truth, _mu

DEV = "cuda"
SEED = 17

RIGID_IDX = [0, 1, 2, 3, 7, 8, 9]            # sod sdd s_z z0 phi0 dx dy
ALL_IDX = list(range(11))


def load_truth_volume(raw_dir: Path):
    """(D,H,W) mu volume on the SOURCE-frame z axis (ascending) + grid meta.

    In-plane coordinates are ISO-CENTRED: the volume centre is placed at
    (truth image centre - DataCollectionCenterPatient), so the source
    trajectory legitimately rotates around (0, 0).
    """
    truth_files = _list_truth(raw_dir)
    truth_files.sort(key=lambda t: -t[0])   # source z = -patient z, ascending
    slices, src_z = [], []
    ps = None; ipp = None; dccp = None
    for pZ, fp in truth_files:
        mu, ds = _mu(fp)
        slices.append(mu)
        src_z.append(-pZ)
        if ps is None:
            ps = float(ds.PixelSpacing[0])
            ipp = tuple(float(v) for v in ds.ImagePositionPatient)
            try:
                dccp = tuple(float(v) for v in ds[0x0018, 0x9313].value)
            except KeyError:
                dccp = (0.0, 0.0, 0.0)
            iop = [float(v) for v in getattr(ds, "ImageOrientationPatient",
                                              (1, 0, 0, 0, 1, 0))]
            if max(abs(iop[0] - 1), abs(iop[4] - 1)) > 1e-3:
                print(f"[v4] WARNING non-axial IOP: {iop}", flush=True)
    vol = np.stack(slices, axis=0)
    src_z = np.asarray(src_z, dtype=np.float64)
    dzs = np.diff(src_z)
    H, W = vol.shape[1], vol.shape[2]
    cx_pat = ipp[0] + (W - 1) / 2.0 * ps
    cy_pat = ipp[1] + (H - 1) / 2.0 * ps
    print(f"[v4] truth volume {vol.shape}  ps={ps:.6f}  "
          f"z[{src_z[0]:.1f},{src_z[-1]:.1f}] dz={dzs.mean():.4f}±{dzs.std():.5f}",
          flush=True)
    print(f"[v4] image centre (patient) = ({cx_pat:+.2f}, {cy_pat:+.2f})  "
          f"DataCollectionCenter = ({dccp[0]:+.2f}, {dccp[1]:+.2f})  "
          f"→ iso-frame volume centre = ({cx_pat - dccp[0]:+.2f}, "
          f"{cy_pat - dccp[1]:+.2f})", flush=True)
    meta = {
        "ps": ps,
        "cx": cx_pat - dccp[0],          # ISO-centred frame
        "cy": cy_pat - dccp[1],
        "z0c": float(src_z[0]),
        "dz": float(dzs.mean()),
        "D": vol.shape[0], "H": H, "W": W,
        "z_lo": float(src_z[0] - 1.5), "z_hi": float(src_z[-1] + 1.5),
        "cx_patient": cx_pat, "cy_patient": cy_pat, "dccp": list(dccp),
    }
    return torch.from_numpy(vol).float(), meta


class ForwardModel:
    def __init__(self, vol, meta, geom, conv, device=DEV):
        self.vol = vol.to(device)[None, None]
        self.meta = meta
        self.conv = conv
        g = geom
        self.ang = torch.from_numpy(np.asarray(
            g["gantry_angles_corrected"], np.float64)).to(device)
        self.z_pos = torch.from_numpy(np.asarray(
            g["z_positions"], np.float64)).to(device)
        n = self.z_pos.shape[0]
        self.ffs_dz = torch.from_numpy(np.asarray(
            g.get("ffs_dz", np.zeros(n)), np.float64)).to(device)
        self.ffs_drho = torch.from_numpy(np.asarray(
            g.get("ffs_drho", np.zeros(n)), np.float64)).to(device)
        self.u0 = float(g["u0"]); self.v0 = float(g["v0"])
        self.theta_p = float(g["du"]) / float(g["sdd"])
        self.z_c = float(self.z_pos.mean())
        self.sod0, self.sdd0, self.dv0 = float(g["sod"]), float(g["sdd"]), float(g["dv"])

    def params_init(self, device=DEV):
        return torch.zeros(11, device=device, requires_grad=True)

    def decode(self, p):
        return {
            "sod": self.sod0 + 5.0 * p[0],
            "sdd": self.sdd0 + 5.0 * p[1],
            "s_z": 1.0 + 0.003 * p[2],
            "z0": 5.0 * p[3],
            "dv": self.dv0 * (1.0 + 0.003 * p[4]),
            "u0_off": 3.0 * p[5],
            "v0_off": 2.0 * p[6],
            "phi0": 0.05 * p[7],
            "dx": 5.0 * p[8],
            "dy": 5.0 * p[9],
            "s_xy": 1.0 + 0.005 * p[10],
        }

    def simulate(self, p, k_idx, r_idx, c_idx, T):
        q = self.decode(p)
        m, cv = self.meta, self.conv
        B = k_idx.shape[0]
        ang = self.ang[k_idx]; zp = self.z_pos[k_idx]
        dzf = self.ffs_dz[k_idx]; drf = self.ffs_drho[k_idx]

        phi = cv["s_phi"] * ang + cv["phi0_g"] + q["phi0"]
        z_nom = self.z_c + q["s_z"] * (zp - self.z_c) + q["z0"]
        z_src = z_nom + dzf
        r_src = q["sod"] + drf
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        src64 = torch.stack([r_src * cphi, r_src * sphi, z_src], -1)
        nomxy = torch.stack([q["sod"] * cphi, q["sod"] * sphi], -1)

        gam = cv["s_g"] * (c_idx.double() - (self.u0 + q["u0_off"])) * self.theta_p
        a = phi[:, None] + gam[None, :]
        det_xy = nomxy[:, None, :] + q["sdd"] * torch.stack(
            [-torch.cos(a), -torch.sin(a)], -1)                     # (B,C,2)
        det_z = (z_nom[:, None]
                 + cv["s_v"] * (r_idx.double()[None, :]
                                 - (self.v0 + q["v0_off"])) * q["dv"])  # (B,R)

        R, C = r_idx.shape[0], c_idx.shape[0]
        src = src64.float()
        det = torch.empty(B, R, C, 3, dtype=torch.float32, device=src.device)
        det[..., 0] = det_xy[:, None, :, 0].float()
        det[..., 1] = det_xy[:, None, :, 1].float()
        det[..., 2] = det_z[:, :, None].float()

        s = src[:, None, None, :]
        d = det - s
        denom = d[..., 2]
        zs = s[..., 2]
        guard = torch.where(denom.abs() < 1e-6,
                             torch.full_like(denom, 1e-6), denom)
        t0 = (m["z_lo"] - zs) / guard
        t1 = (m["z_hi"] - zs) / guard
        flat = denom.abs() < 1e-6
        inside = (zs > m["z_lo"]) & (zs < m["z_hi"])
        t_lo = torch.clamp(torch.minimum(t0, t1), 0.0, 1.0)
        t_hi = torch.clamp(torch.maximum(t0, t1), 0.0, 1.0)
        t_lo = torch.where(flat, torch.where(inside, torch.zeros_like(t_lo),
                                              torch.ones_like(t_lo)), t_lo)
        t_hi = torch.where(flat, torch.ones_like(t_hi), t_hi)
        seg = (t_hi - t_lo).clamp(min=0.0)
        valid = seg > 1e-4

        tt = t_lo[..., None] + seg[..., None] * (
            (torch.arange(T, device=src.device, dtype=torch.float32) + 0.5) / T)
        pnt = s[..., None, :] + tt[..., None] * d[..., None, :]

        sxy = float(m["ps"]) * q["s_xy"]
        ix = (pnt[..., 0] - (m["cx"] + q["dx"])) / sxy + (m["W"] - 1) / 2.0
        iy = (pnt[..., 1] - (m["cy"] + q["dy"])) / sxy + (m["H"] - 1) / 2.0
        iz = (pnt[..., 2] - m["z0c"]) / m["dz"]
        gx = (2.0 * (ix + 0.5) / m["W"]) - 1.0
        gy = (2.0 * (iy + 0.5) / m["H"]) - 1.0
        gz = (2.0 * (iz + 0.5) / m["D"]) - 1.0
        grid = torch.stack([gx, gy, gz], -1).reshape(1, B * R * C, T, 1, 3)
        sampled = F.grid_sample(self.vol, grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)
        mu_mean = sampled.reshape(B, R, C, T).mean(-1)
        length = d.norm(dim=-1) * seg
        return mu_mean * length, valid


def affine_fit(sim, meas, valid):
    s = sim.detach()[valid]; t = meas[valid]
    n = s.numel()
    ss, sm = (s * s).sum(), s.sum()
    den = ss * n - sm * sm
    if den.abs() < 1e-12:
        return torch.tensor(1.0, device=s.device), torch.tensor(0.0, device=s.device)
    a = ((s * t).sum() * n - sm * t.sum()) / den
    b = (t.sum() - a * sm) / n
    return a, b


def loss_fn(sim, meas, valid):
    a, b = affine_fit(sim, meas, valid)
    cal = a * sim + b
    l_int = F.huber_loss(cal[valid], meas[valid], delta=0.3)
    # channel-gradient term — channels are CONTIGUOUS blocks, so the
    # finite difference along the last axis is a physical edge signal.
    g_cal = cal[..., 1:] - cal[..., :-1]
    g_meas = meas[..., 1:] - meas[..., :-1]
    g_valid = valid[..., 1:] & valid[..., :-1]
    l_grad = F.huber_loss(g_cal[g_valid], g_meas[g_valid], delta=0.15)
    return l_int + l_grad, a, b


def main() -> int:
    import os
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    root = Path(os.environ.get("AGENT4CT_DATA",
                                "/cluster/maier/Agent4CT/data")) / "mayo_ldct"
    raw = root / "raw" / "L014"

    import pydicom
    SOP_CT = "1.2.840.10008.5.1.4.1.1.2"
    series = None
    for sd in sorted(raw.iterdir()):
        if not sd.is_dir():
            continue
        f0 = next(sd.iterdir(), None)
        if f0 is None:
            continue
        try:
            h = pydicom.dcmread(str(f0), stop_before_pixels=True)
        except Exception:
            continue
        desc = getattr(h, "SeriesDescription", "").lower()
        if getattr(h, "SOPClassUID", "") != SOP_CT and "full" in desc:
            series = sd; break
    print(f"[v4] projection series: {series}", flush=True)
    print(f"[v4] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"[v4] reading DICOM-CT-PD (slow) ...", flush=True)
    proj, geom = read_dicom_ctpd(series)
    n_proj, n_rows, n_chan = proj.shape
    print(f"[v4] proj {proj.shape}  sod={geom['sod']:.3f} sdd={geom['sdd']:.3f} "
          f"du={geom['du']:.6f} dv={geom['dv']:.6f} u0={geom['u0']:.2f} "
          f"v0={geom['v0']:.2f} pitch={geom['pitch_mm']:.4f}", flush=True)
    print(f"[v4] proj stats: mean={proj.mean():.4f} max={proj.max():.4f}",
          flush=True)

    vol, meta = load_truth_volume(raw)

    z_pos = np.asarray(geom["z_positions"], np.float64)
    ok = (z_pos > meta["z_lo"] + 30.0) & (z_pos < meta["z_hi"] - 30.0)
    k_ok = np.where(ok)[0]
    print(f"[v4] usable readouts {k_ok.shape[0]}/{n_proj}", flush=True)

    proj_t = torch.from_numpy(proj)

    def batch(Bk, Cblk):
        k = torch.from_numpy(rng.choice(k_ok, Bk, replace=False)).long()
        r = torch.arange(n_rows)
        c0 = int(rng.integers(0, n_chan - Cblk))
        c = torch.arange(c0, c0 + Cblk)
        meas = proj_t[k][:, :, c].to(DEV).float()
        return k.to(DEV), r.to(DEV), c.to(DEV), meas

    def masked_step(p, opt, free_idx):
        mask = torch.zeros_like(p)
        mask[free_idx] = 1.0
        p.grad.mul_(mask)
        opt.step()

    # ---- pre-stage: convention grid (rigid params only) -------------------
    combos = [
        {"s_phi": sp, "s_g": sg, "s_v": sv, "phi0_g": pg}
        for sp in (+1.0, -1.0) for sg in (+1.0, -1.0)
        for sv in (+1.0, -1.0)
        for pg in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
    ]
    pre = []
    for ci, cv in enumerate(combos):
        fm = ForwardModel(vol, meta, geom, cv)
        p = fm.params_init()
        opt = torch.optim.Adam([p], lr=0.05)
        first = None; last = []
        for it in range(120):
            k, r, c, meas = batch(8, 48)
            sim, valid = fm.simulate(p, k, r, c, T=160)
            loss, a, b = loss_fn(sim, meas, valid)
            if first is None:
                first = float(loss)
            opt.zero_grad(); loss.backward()
            masked_step(p, opt, RIGID_IDX)
            if it >= 80:
                last.append(float(loss))
        pre.append((float(np.mean(last)), ci))
        print(f"[v4] combo {ci:2d} s_phi={cv['s_phi']:+.0f} s_g={cv['s_g']:+.0f} "
              f"s_v={cv['s_v']:+.0f} phi0={cv['phi0_g']:.2f}  "
              f"loss0={first:.5f} loss={pre[-1][0]:.5f}", flush=True)
    pre.sort()
    best_combo = combos[pre[0][1]]
    print(f"[v4] BEST combo: {best_combo}  (loss {pre[0][0]:.5f}; "
          f"runner-up {pre[1][0]:.5f})", flush=True)

    # ---- stage 1: rigid + scale-z -----------------------------------------
    fm = ForwardModel(vol, meta, geom, best_combo)
    p = fm.params_init()
    opt = torch.optim.Adam([p], lr=0.03)
    hist = []

    def log_it(stage, it, loss, a):
        q = {kk: float(vv) for kk, vv in fm.decode(p.detach()).items()}
        print(f"[v4] {stage} it {it:4d} loss={float(loss):.5f} "
              f"sod={q['sod']:.3f} sdd={q['sdd']:.3f} "
              f"mag={q['sdd']/q['sod']:.5f} s_z={q['s_z']:.6f} "
              f"z0={q['z0']:+.3f} dv={q['dv']:.6f} "
              f"u0o={q['u0_off']:+.3f} v0o={q['v0_off']:+.3f} "
              f"phi0={q['phi0']:+.5f} dx={q['dx']:+.3f} dy={q['dy']:+.3f} "
              f"s_xy={q['s_xy']:.6f} a={float(a):.4f}", flush=True)
        hist.append({"stage": stage, "iter": it, "loss": float(loss), **q})

    for it in range(1200):
        k, r, c, meas = batch(24, 64)
        sim, valid = fm.simulate(p, k, r, c, T=288)
        loss, a, b = loss_fn(sim, meas, valid)
        opt.zero_grad(); loss.backward()
        masked_step(p, opt, RIGID_IDX)
        if it % 100 == 0 or it == 1199:
            log_it("S1", it, loss, a)

    # ---- stage 2: release detector/grid scale params ----------------------
    opt = torch.optim.Adam([p], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500,
                                                        eta_min=0.001)
    for it in range(1500):
        k, r, c, meas = batch(24, 64)
        sim, valid = fm.simulate(p, k, r, c, T=288)
        loss, a, b = loss_fn(sim, meas, valid)
        opt.zero_grad(); loss.backward()
        masked_step(p, opt, ALL_IDX)
        sched.step()
        if it % 100 == 0 or it == 1499:
            log_it("S2", it, loss, a)

    # ---- final eval (chunked) ---------------------------------------------
    with torch.no_grad():
        k_all = torch.from_numpy(rng.choice(k_ok, 48, replace=False)).long()
        r = torch.arange(n_rows).to(DEV)
        c = torch.arange(0, n_chan, 4).to(DEV)
        sims, meass, valids = [], [], []
        for i0 in range(0, 48, 8):
            kc = k_all[i0:i0 + 8]
            meas = proj_t[kc][:, :, c.cpu()].to(DEV).float()
            sim, valid = fm.simulate(p, kc.to(DEV), r, c, T=512)
            sims.append(sim); meass.append(meas); valids.append(valid)
        sim = torch.cat(sims); meas = torch.cat(meass); valid = torch.cat(valids)
        a, b = affine_fit(sim, meas, valid)
        sim_cal = a * sim + b
        final_l2 = float(F.mse_loss(sim_cal[valid], meas[valid]))
        rel = float((sim_cal[valid] - meas[valid]).abs().mean()
                    / meas[valid].abs().mean())

    q = {kk: float(vv) for kk, vv in fm.decode(p.detach()).items()}
    mag = q["sdd"] / q["sod"]
    pitch_eff = q["s_z"] * float(geom["pitch_mm"])
    derived = {
        "mag_fit": mag,
        "dv_fit": q["dv"],
        "dv_selfconsistent_0.6xmag": 0.6 * mag,
        "dv_ratio": q["dv"] / (0.6 * mag),
        "pitch_tag": float(geom["pitch_mm"]),
        "pitch_eff_fit": pitch_eff,
        "pitch_design_0.8x38.4": 30.72,
        "s_xy_fit": q["s_xy"],
        "truth_ps_eff": meta["ps"] * q["s_xy"],
        "powell_fbp": {"sod": 595.362, "sdd": 1086.803, "ps": 0.700857},
        "v3_ssr": {"sod": 592.829, "sdd": 1087.268, "s_z": 1.001665},
        "final_eval": {"l2": final_l2, "rel_abs_err": rel,
                        "affine_a": float(a), "affine_b": float(b)},
    }
    out = {
        "convention": best_combo,
        "convention_grid": [{"loss": l, **combos[i]} for l, i in pre],
        "fitted": q,
        "derived": derived,
        "history": hist,
        "meta": meta,
        "n_usable_readouts": int(k_ok.shape[0]),
    }
    oj = REPO / "results" / "breast_debug" / "L014_forward_geom_fit_v4.json"
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(out, indent=2))
    print(f"[v4] wrote {oj}", flush=True)
    print(f"[v4] FITTED: sod={q['sod']:.3f} sdd={q['sdd']:.3f} mag={mag:.5f} "
          f"s_z={q['s_z']:.6f} dv={q['dv']:.6f} (0.6xmag={0.6*mag:.6f}) "
          f"pitch_eff={pitch_eff:.4f} s_xy={q['s_xy']:.6f} "
          f"rel|err|={rel:.4f}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    it_h = np.arange(len(hist))
    axes[0, 0].plot(it_h, [h["loss"] for h in hist]); axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("huber+grad loss (S1 then S2)")
    axes[0, 1].plot(it_h, [h["sod"] for h in hist], label="sod")
    axes[0, 1].plot(it_h, [h["sdd"] - 490 for h in hist], label="sdd-490")
    axes[0, 1].legend(); axes[0, 1].set_title("sod / sdd (mm)")
    axes[0, 2].plot(it_h, [h["s_z"] for h in hist], label="s_z")
    axes[0, 2].plot(it_h, [h["s_xy"] for h in hist], label="s_xy")
    axes[0, 2].axhline(30.72 / float(geom["pitch_mm"]), ls="--", c="k",
                        label="s_z design-pitch")
    axes[0, 2].legend(); axes[0, 2].set_title("scale factors")
    with torch.no_grad():
        ksel = torch.from_numpy(
            k_ok[np.linspace(1000, k_ok.shape[0] - 1000, 3).astype(int)]).long()
        r = torch.arange(n_rows).to(DEV)
        c = torch.arange(n_chan).to(DEV)
        for j in range(3):
            kc = ksel[j:j + 1]
            meas = proj_t[kc][:, :, c.cpu()].to(DEV).float()
            sim, valid = fm.simulate(p, kc.to(DEV), r, c, T=512)
            a2, b2 = affine_fit(sim, meas, valid)
            prof_m = meas[0, 31, :].cpu().numpy()
            prof_s = (a2 * sim + b2)[0, 31, :].cpu().numpy()
            ax = axes[1, j]
            ax.plot(prof_m, lw=0.7, label="measured (row 31)")
            ax.plot(prof_s, lw=0.7, label="simulated")
            ax.set_title(f"readout {int(kc[0])} central-row profile")
            ax.legend(fontsize=7)
    fig.suptitle(f"L014 forward geometry fit v4b — sod={q['sod']:.2f} "
                 f"sdd={q['sdd']:.2f} s_z={q['s_z']:.5f} dv={q['dv']:.5f} "
                 f"s_xy={q['s_xy']:.5f}  rel|err|={rel:.4f}")
    fig.tight_layout()
    op = REPO / "results" / "mayo_debug" / "L014_forward_geom_fit_v4.png"
    op.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(op, dpi=110, bbox_inches="tight")
    print(f"[v4] wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
