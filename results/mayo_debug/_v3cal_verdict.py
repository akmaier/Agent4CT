v3  = dict(sod=592.829, sdd=1087.268, s_z=1.001665, dz=-0.159)
v3c = dict(sod=592.8233642578125, sdd=1087.2691650390625,
           s_z=1.0016603469848633, dz=-0.16086938977241516)
th  = dict(sod=0.5, sdd=0.5, s_z=0.0003, dz=0.1)
print("{:7}{:>13}{:>14}{:>12}{:>9}  verdict".format("param", "v3 prod", "v3cal refit", "delta", "|thr|"))
ok = True
for k in ["sod", "sdd", "s_z", "dz"]:
    d = v3c[k] - v3[k]
    within = abs(d) < th[k]
    ok = ok and within
    print("{:7}{:13.6f}{:14.6f}{:+12.6f}{:9.4f}  {}".format(k, v3[k], v3c[k], d, th[k], "OK" if within else "SHIFT"))
print()
print("post_fbp lo (learnable floor):  v3 = relu@0 (=0)   v3cal = -4.39e-05  (settled ~0)")
print("VERDICT:", "STABLE -- calibration bug did NOT bias v3 geometry; v3 production STANDS" if ok else "SHIFTED -- flag for user")
