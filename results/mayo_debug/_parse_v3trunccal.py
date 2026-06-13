import json, statistics as st

def load(path):
    d = json.load(open(path))
    pts = d.get("patients") or d.get("results") or d
    out = {}
    for p in pts:
        out[p["patient"]] = (p["split"], p["metrics"]["hd_tc"]["ssim_cal"],
                             p["metrics"]["ld_tc"]["ssim_cal"])
    return out

before = load("wagner_gt_hd_ld_fbp_v3trunc.json")     # uncorrected (bg->0 bug)
after  = load("wagner_gt_hd_ld_fbp_v3trunccal.json")  # corrected (bg_target=truth)

print("HD+truncation-corr SSIM_cal   |   before(bg->0)  after(bg=truth)   delta")
order = ["L145","L186","L209","L219","L277","L014","L056","L058","L075","L123"]
db, da = [], []
for k in order:
    sp, hb, _ = before[k]
    _, ha, _ = after[k]
    db.append(hb); da.append(ha)
    print(f"  {k:5} {sp:5}                      {hb:7.4f}        {ha:7.4f}     {ha-hb:+.4f}")
print(f"  {'MEAN':5}                            {st.mean(db):7.4f}        {st.mean(da):7.4f}     {st.mean(da)-st.mean(db):+.4f}")
