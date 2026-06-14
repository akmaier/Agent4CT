import json
d = json.load(open("v3allslices_metrics.json"))
print("PER-PATIENT  (HD=oracle, LD=baseline; calibrated SSIM)")
print("{:6}{:6}{:>7}{:>5}{:>9}{:>9}{:>7}{:>8}".format(
    "pt", "split", "n/tot", "ps", "HD_ssim", "LD_ssim", "gap", "zres"))
for pat, p in d["patients"].items():
    print("{:6}{:6}{:>3}/{:<3}{:>5.3f}{:9.4f}{:9.4f}{:7.4f}{:8.2f}".format(
        pat, p["split"], p["n_eval"], p["n_truth"], p["truth_ps"],
        p["hd_ssim"]["mean"], p["ld_ssim"]["mean"],
        p["hd_ssim"]["mean"] - p["ld_ssim"]["mean"], p["zres_mean"]))
print("\nPER-SPLIT")
print("{:6}{:>6}{:>9}{:>9}{:>9}{:>9}{:>7}".format(
    "split", "n", "HD_ssim", "LD_ssim", "HD_psnr", "LD_psnr", "gap"))
for sp, s in d["splits"].items():
    print("{:6}{:>6}{:9.4f}{:9.4f}{:9.2f}{:9.2f}{:7.4f}".format(
        sp, s["n_eval"], s["hd_ssim"]["mean"], s["ld_ssim"]["mean"],
        s["hd_psnr"]["mean"], s["ld_psnr"]["mean"],
        s["hd_ssim"]["mean"] - s["ld_ssim"]["mean"]))
o = d["overall"]
print("{:6}{:>6}{:9.4f}{:9.4f}{:9.2f}{:9.2f}{:7.4f}".format(
    "ALL", o["n_eval"], o["hd_ssim"]["mean"], o["ld_ssim"]["mean"],
    o["hd_psnr"]["mean"], o["ld_psnr"]["mean"],
    o["hd_ssim"]["mean"] - o["ld_ssim"]["mean"]))
if d.get("errors"):
    print("\nERRORS:", d["errors"])
