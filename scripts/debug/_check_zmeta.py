import h5py, numpy as np
R = "/cluster/maier/Agent4CT/data/mayo_ldct"
v = h5py.File(f"{R}/staged/val_sino_lowdose.h5", "r")
zm = v["z_meta"][:] if "z_meta" in v else None
print("staged keys:", list(v.keys()))
if zm is not None:
    pz, sz = zm[:, 0], zm[:, 1]
    print(f"staged val n={zm.shape[0]} patient_z[{pz.min():.1f},{pz.max():.1f}] "
          f"source_z[{sz.min():.1f},{sz.max():.1f}]")
    print("first 5 patient_z:", np.round(pz[:5], 1))
    d = np.diff(pz)
    print("patient_z monotonic asc/desc:", bool(np.all(d > 0)), bool(np.all(d < 0)),
          "| n_unique:", len(np.unique(np.round(pz, 2))))
for sub in ["staged_helix2fan_v3", "staged_helix2fan_ssr_fitted", "staged_helix2fan"]:
    p = f"{R}/{sub}/L277_sino_lowdose_z_grid.npy"
    try:
        zg = np.load(p)
        ir = int(((sz >= zg.min()) & (sz <= zg.max())).sum()) if zm is not None else -1
        print(f"{sub}: L277 z_grid[{zg.min():.1f},{zg.max():.1f}] n={zg.size} "
              f"| staged source_z in-range: {ir}/{zm.shape[0] if zm is not None else '?'}")
    except Exception as e:
        print(f"{sub}: {e}")
