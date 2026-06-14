import json
d = json.load(open("results/mayo_debug/canonical_hd_ld/canonical_hd_ld_metrics.json"))
base = {"L145": 0.9469, "L186": 0.9482, "L209": 0.9453, "L219": 0.9587, "L277": 0.9331,
        "L014": 0.9564, "L056": 0.9605, "L058": 0.9330, "L075": 0.9626, "L123": 0.9631}
print("per-patient canonical HD (vs baseline HD):")
for p, v in d["patients"].items():
    bh = base.get(p, 0.0)
    print("  {} ({:5}) n={:3}  HD {:.4f} (base {:.4f}, {:+.3f})  LD {:.4f}".format(
        p, v["split"], v["n"], v["hd_ssim"], bh, v["hd_ssim"] - bh, v["ld_ssim"]))
o = d["overall"]
print("OVERALL n={} HD {:.4f} LD {:.4f} gap {:.4f}".format(
    o["n"], o["hd_ssim"], o["ld_ssim"], o["gap"]))
