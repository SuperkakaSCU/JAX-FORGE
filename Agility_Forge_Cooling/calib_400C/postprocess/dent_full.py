"""Each hit's own flat, located by what the hit changed (read-only).

For every 0.5 mm slab within z +- 12 mm and every theta (1 deg), the caliper
width before (xpk) and after (xpk1) the hit is computed.  The hit's own dent is
the (slab, theta) whose width dropped most.  Its flat is the smallest AFTER
width within 3 mm and 8 deg of that point, restricted to cells that dropped by
at least half the largest drop (so an older, untouched flat cannot be picked).
Half of that width, less the scanner bias, is the closure the bar kept.

usage: dent_full.py <series.h5> <calibration dir>
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

H5, OUT = sys.argv[1], Path(sys.argv[2])
Q, DX = 0.001, 0.5
TH = np.radians(np.arange(0.0, 180.0, 1.0))
NRM = np.stack([np.cos(TH), np.sin(TH)])
f = h5py.File(H5, "r")


def clean(a):
    a = np.asarray(a)
    return a[np.isfinite(a).all(1)]


def widths(P, edges):
    o = np.argsort(P[:, 0], kind="stable")
    P = P[o]
    b = np.searchsorted(P[:, 0], edges)
    W = np.full((len(edges) - 1, len(TH)), np.nan)
    for i in range(len(edges) - 1):
        Qp = P[b[i]:b[i + 1], 1:3]
        if len(Qp) >= 1000:
            pr = Qp @ NRM
            W[i] = np.quantile(pr, 1 - Q, 0) - np.quantile(pr, Q, 0)
    return W


edges = np.arange(5.0, 95.0 + DX, DX)
bias = float(np.nanmedian(np.nanmin(widths(clean(f["step_00000"]["xpk"]), edges), 1))) - 38.1
ach = {r["hit"]: r for r in json.loads((OUT / "rho_achieved.json").read_text())["hits"]}
st = {r["hit"]: r for r in json.loads((OUT / "stroke_rho.json").read_text())}
print(f"{Path(H5).stem}: scanner bias on the width {bias:+.3f} mm")
print(f"{'hit':>3} {'z':>6} {'x flat':>6} {'phi':>5} {'theta':>5} {'dphi':>5} | {'cmd':>6} "
      f"{'s2':>6} | {'SCAN':>6} | {'scan-cmd':>8} {'scan-s2':>7} | {'drop':>5}")
rows = []
for k in range(int(f.attrs["n_steps"])):
    g = f[f"step_{k:05d}"]
    a = dict(g.attrs)
    z, phi, cmd = float(a["z"]), np.degrees(a["phi"]) % 180.0, float(a["rho"])
    e = np.arange(z - 12.0, z + 12.0 + DX, DX)
    xc = 0.5 * (e[:-1] + e[1:])
    W0, W1 = widths(clean(g["xpk"]), e), widths(clean(g["xpk1"]), e)
    drop = np.nan_to_num(W0 - W1, nan=-np.inf)
    i, j = np.unravel_index(np.argmax(drop), drop.shape)
    dmax = drop[i, j]
    dth = np.abs((np.degrees(TH)[None, :] - np.degrees(TH[j]) + 90.0) % 180.0 - 90.0)
    mask = (np.abs(xc - xc[i])[:, None] <= 3.0) & (dth <= 8.0) & (drop >= 0.5 * dmax)
    cand = np.where(mask, W1, np.inf)
    ii, jj = np.unravel_index(np.argmin(cand), cand.shape)
    h = 0.5 * (W1[ii, jj] - bias)
    hit = k + 1
    s2 = st.get(hit, {}).get("rho_stroke2", np.nan)
    t = np.degrees(TH[jj])
    dphi = (t - phi + 90.0) % 180.0 - 90.0
    rows.append(dict(hit=hit, z=z, x_flat=float(xc[ii]), phi=phi, theta=float(t),
                     rho_cmd=cmd, rho_s2=s2, rho_achieved=ach[hit]["rho_achieved"],
                     rho_scan=float(h), drop=float(dmax)))
    print(f"{hit:>3} {z:>6.2f} {xc[ii]:>6.2f} {phi:>5.1f} {t:>5.0f} {dphi:>+5.0f} | {cmd:>6.2f} "
          f"{s2:>6.2f} | {h:>6.2f} | {h - cmd:>+8.2f} {h - s2:>+7.2f} | {dmax:>5.2f}")
json.dump(rows, open(Path(sys.argv[0]).with_name(f"dent_full_{Path(H5).stem}.json"), "w"), indent=1)
f.close()
