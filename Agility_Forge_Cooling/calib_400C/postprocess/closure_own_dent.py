"""Closure each hit kept after its release, measured on the hit's OWN flat (read-only).

usage: closure_own_dent.py <sim dir> <dent_full json> <phi|theta> [half window deg, default 8]

run_quality.py takes the smallest caliper width in ANY direction within +-10 mm of the hit, so a deeper neighbouring
dent wins (roll3.1 hit 11 next to hit 8, hit 5 next to hit 2).  Here the direction is restricted to the hit's own flat:
within the window around the commanded angle (phi, commanded-position runs) or the scan angle (theta, scan-pose runs).
Also prints the bar length and the mesh volume of every state present.
"""
import json
import sys
from pathlib import Path

import meshio
import numpy as np

D, JS, KEY = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
DT = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
scan = {r["hit"]: r for r in json.load(open(JS))}
F = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])   # all six faces outward (the volume needs a consistent orientation)


def load(vtu, n=10):
    m = meshio.read(vtu)
    P, C = m.points, m.cells_dict["hexahedron"]
    faces = C[:, F].reshape(-1, 4)
    _, inv, cnt = np.unique(np.sort(faces, 1), axis=0, return_inverse=True, return_counts=True)
    bnd = faces[cnt[inv.ravel()] == 1]
    s = np.linspace(0.0, 1.0, n)
    u, v = (a.ravel() for a in np.meshgrid(s, s))
    w = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], 1)
    S = np.einsum("sk,fkd->fsd", w, P[bnd]).reshape(-1, 3)
    # volume from the boundary quads (divergence theorem, each quad as two triangles about its centre is not needed:
    # sum over the 4 triangles of the quad fan); sign fixed by taking the absolute value of the total
    q = P[bnd]; c = q.mean(1)
    vol = 0.0
    for i in range(4):
        a, b = q[:, i], q[:, (i + 1) % 4]
        vol += np.einsum("ij,ij->i", c, np.cross(a, b)).sum() / 6.0
    return S, abs(vol), float(P[:, 0].min()), float(P[:, 0].max())


print(f"{D.name}: own-flat closure, direction within +-{DT:g} deg of the {'commanded' if KEY == 'phi' else 'scan'} angle")
print("hit | angle | sim half-width (at x, theta) | scan  | command | sim - scan | sim - command | bar x min..max | volume mm3")
for vtu in sorted(D.glob("sim_[0-9][0-9][0-9][0-9].vtu")):
    k = int(vtu.stem.split("_")[1])
    S, vol, x0b, x1b = load(vtu)
    if k not in scan:
        print(f"{k:>3} |   -   | {'-':>28} |   -   |    -    |     -      |       -       | {x0b:7.2f}..{x1b:7.2f} | {vol:10.1f}")
        continue
    s = scan[k]
    ang = s[KEY] % 180.0
    ths = np.radians(ang + np.arange(-DT, DT + 0.01, 1.0))
    N = np.stack([np.cos(ths), np.sin(ths)])
    best = (np.inf, None, None)
    for x0 in np.arange(s["z"] - 10.0, s["z"] + 10.0, 0.5):
        Q = S[(S[:, 0] >= x0) & (S[:, 0] < x0 + 0.5), 1:3]
        if len(Q) < 200:
            continue
        pr = Q @ N
        w = pr.max(0) - pr.min(0)
        j = int(np.argmin(w))
        if w[j] < best[0]:
            best = (float(w[j]), float(x0 + 0.25), float(np.degrees(ths[j])))
    w, x, t = best
    print(f"{k:>3} | {ang:5.1f} | {0.5 * w:7.3f} (x {x:6.2f}, {t:5.1f} deg) | {s['rho_scan']:6.3f} | {s['rho_cmd']:6.2f}  | "
          f"{0.5 * w - s['rho_scan']:+9.3f}  | {0.5 * w - s['rho_cmd']:+11.3f}   | {x0b:7.2f}..{x1b:7.2f} | {vol:10.1f}")
