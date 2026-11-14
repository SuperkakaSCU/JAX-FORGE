"""Animated 3D scan-vs-simulation geometry over the first hits (read-only).

usage: gif_hits_3d.py <series.h5> <dent_full json> <sim dir> <initial mesh> <n_hits> <out.gif> [label]

States: before hit 1, after hit 1 .. after hit n.  Scan: step_00000 xpk, then
step_{k-1} xpk1 (after hit k).  Simulation: the initial billet mesh, then
sim_000k.vtu (after the final release of hit k).  Every surface is resampled on
the same (polar angle, x) grid as build_3d_hit.py (0.5 mm slabs, 2 deg bins,
radius = median distance from the slab's least-squares centre of symmetry; the
scanner's median-radius bias 0.199 mm removed from the scans).

Top: the two 3D surfaces at true scale, coloured by the radius change from the
same data set's initial state (blue = pressed in, red = bulged out); the bar
turns once about its own axis per state.  Bottom: the unrolled difference
r_sim - r_scan of the current state, with the die centres of the hits done so
far as the simulation placed them (both dies: phi and phi + 180), numbered.
Under it a table of the closure each hit kept (half-width, mm): command /
experiment (dent_full's rho_scan, measured on the scan's own flat) / simulation
(smallest caliper half-width within +-4 deg of the simulated die angle and
+-4 mm of the simulated die centre, state right after the hit).
With the environment variable MESH=1 the simulation panel shows the finite-element mesh itself instead of the
resampled surface: the outer skin of the HEX8 mesh of that state (sim_000k.vtu = the mesh hit k was run on,
deformed) with its element edges, restricted to the cells whose centroid lies in the plotted x range; the two
section cuts show the O-grid.  Colour = radius change of the face (mean vertex radius about the slab centre
line, minus the initial radius), same scale as the scan panel.
(2026-11-10: the per-hit text boxes on the map overlapped from hit 6 on and
measured the scan along the COMMANDED angle, which is wrong where the real flat
is 12-14 deg off, roll3.1 hits 8 and 11; previous version gif_hits_3d_labels_v1.py.bak.)
"""
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.colors import TwoSlopeNorm

H5, JS, SIM, MESH0, NH, OUT = sys.argv[1:7]
NH = int(NH)
LABEL = sys.argv[7] if len(sys.argv) > 7 else ""
rows = {r["hit"]: r for r in json.load(open(JS))}
DX, DTH, BIAS_R = 0.5, 2.0, 0.199
X0, X1 = 0.0, 102.0
xs = np.arange(X0, X1 + 1e-9, DX)
edges = np.arange(0.0, 360.0 + 1e-9, DTH)
ths = edges[:-1] + DTH / 2


def sort_x(P):
    return P[np.argsort(P[:, 0], kind="stable")]


def scan(step, key):
    with h5py.File(H5, "r") as f:
        P = np.asarray(f[f"step_{step:05d}"][key])
    return sort_x(P[np.isfinite(P).all(1)])


def sim(path, n=14):
    m = meshio.read(path)
    P, C = m.points, m.cells_dict["hexahedron"]
    F = np.array([(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    faces = C[:, F].reshape(-1, 4)
    _, inv, cnt = np.unique(np.sort(faces, 1), axis=0, return_inverse=True, return_counts=True)
    bnd = faces[cnt[inv.ravel()] == 1]
    s = np.linspace(0.0, 1.0, n)
    u, v = (a.ravel() for a in np.meshgrid(s, s))
    w = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], 1)
    return sort_x(np.einsum("sk,fkd->fsd", w, P[bnd]).reshape(-1, 3))


def slab_centre(Q, q=0.002):
    th = np.radians(np.arange(0.0, 180.0, 5.0))
    N = np.stack([np.cos(th), np.sin(th)], 1)
    p = Q @ N.T
    mid = 0.5 * (np.quantile(p, 1 - q, 0) + np.quantile(p, q, 0))
    return np.linalg.lstsq(N, mid, rcond=None)[0]


def grid(P, bias):
    x = P[:, 0]
    R = np.full((len(ths), len(xs)), np.nan)
    for j, x0 in enumerate(xs):
        a, b = np.searchsorted(x, [x0 - DX / 2, x0 + DX / 2])
        Q = P[a:b, 1:3]
        if len(Q) < 50:
            continue
        d = Q - slab_centre(Q)
        r = np.hypot(d[:, 0], d[:, 1]) - bias
        k = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0 // DTH).astype(int)
        cnt = np.bincount(k, minlength=len(ths))
        order = np.lexsort((r, k))
        rs = r[order]
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        for i in np.nonzero(cnt >= 3)[0]:
            R[i, j] = rs[start[i] + cnt[i] // 2]
        col = R[:, j]
        ok = np.isfinite(col)
        if 3 <= ok.sum() < len(col):
            t = np.arange(len(col))
            R[:, j] = np.interp(t, t[ok], col[ok], period=len(col))
    return R


cache = Path(OUT).with_suffix(".npz")
if cache.exists():
    z = np.load(cache)
    SC, SM = list(z["scan"]), list(z["sim"])
else:
    SC = [grid(scan(0, "xpk"), BIAS_R)] + [grid(scan(k - 1, "xpk1"), BIAS_R) for k in range(1, NH + 1)]
    SM = [grid(sim(MESH0), 0.0)] + [grid(sim(f"{SIM}/sim_{k:04d}.vtu"), 0.0) for k in range(1, NH + 1)]
    np.savez_compressed(cache, scan=np.array(SC), sim=np.array(SM))
for nm, G in (("scan", SC), ("sim", SM)):
    for s, R in enumerate(G):
        print(f"{nm} state {s}: {np.isfinite(R).mean() * 100:.1f} % filled, r {np.nanmin(R):.2f}-{np.nanmax(R):.2f} mm")


def closure(R, k):
    h = rows[k]
    i1 = int(round((h["phi"] % 360.0) / DTH - 0.5)) % len(ths)
    i2 = (i1 + len(ths) // 2) % len(ths)
    sel = np.abs(xs - h["z"]) <= 3.0
    hw = 0.5 * (R[i1, sel] + R[i2, sel])
    return float(np.nanmin(hw))


CL = {k: (closure(SC[k], k), closure(SM[k], k)) for k in range(1, NH + 1)}

# caliper closure as in dent_full.py: half the 0.1 %-quantile width along the die
# direction (+-8 deg, 1 deg steps) in 0.5 mm slabs within +-3 mm of the commanded
# centre, minimum; scanner width bias 0.452 mm (virgin bar) removed from the scans
W_BIAS = 0.452


def caliper(P, k, q=0.001):
    h = rows[k]
    th = np.radians(h["phi"] + np.arange(-8.0, 8.5, 1.0))
    N = np.stack([np.cos(th), np.sin(th)])
    best = np.inf
    for x0 in np.arange(h["z"] - 3.0, h["z"] + 3.0, DX):
        a, b = np.searchsorted(P[:, 0], [x0, x0 + DX])
        if b - a < 200:
            continue
        pr = P[a:b, 1:3] @ N
        best = min(best, float(np.min(np.quantile(pr, 1 - q, 0) - np.quantile(pr, q, 0))))
    return 0.5 * best


CC = {k: (caliper(scan(k - 1, "xpk1"), k) - 0.5 * W_BIAS, caliper(sim(f"{SIM}/sim_{k:04d}.vtu"), k))
      for k in range(1, NH + 1)}

# die angle / centre of every hit as the simulation placed it (= the command, or the scan pose with --hit-positions)
USED = {}
for r_ in json.load(open(Path(SIM) / "sim_log_strokes.json")):
    USED.setdefault(r_["hit"], (float(r_["phi"]), float(r_["z"])))
for k in range(1, NH + 1):
    USED.setdefault(k, (rows[k]["phi"], rows[k]["z"]))


def own_flat(P, k):
    """Closure the simulated bar kept on hit k's own flat (as scripts/closure_own_dent.py)."""
    phi, zc = USED[k]
    th = np.radians(phi + np.arange(-4.0, 4.5, 1.0))
    N = np.stack([np.cos(th), np.sin(th)])
    best = np.inf
    for x0 in np.arange(zc - 4.0, zc + 4.0, DX):
        a, b = np.searchsorted(P[:, 0], [x0, x0 + DX])
        if b - a < 200:
            continue
        pr = P[a:b, 1:3] @ N
        best = min(best, float(np.min(pr.max(0) - pr.min(0))))
    return 0.5 * best


OWN = {k: own_flat(sim(f"{SIM}/sim_{k:04d}.vtu"), k) for k in range(1, NH + 1)}
for k in range(1, NH + 1):
    print(f"hit {k}: own flat (phi {USED[k][0]:.1f}, z {USED[k][1]:.2f}) sim {OWN[k]:.3f} | command {rows[k]['rho_cmd']:.3f} "
          f"| scan {rows[k]['rho_scan']:.3f} | sim - scan {OWN[k] - rows[k]['rho_scan']:+.3f}")
for k in range(1, NH + 1):
    (a, b), (c, d) = CL[k], CC[k]
    print(f"hit {k}: flat centre scan {a:.3f} sim {b:.3f} ({b - a:+.3f}) | caliper scan {c:.3f} sim {d:.3f} "
          f"({d - c:+.3f}) mm | command {rows[k]['rho_cmd']:.3f}, dent_full rho_scan {rows[k]['rho_scan']:.3f}")
json.dump({str(k): dict(centre_scan=CL[k][0], centre_sim=CL[k][1], caliper_scan=CC[k][0], caliper_sim=CC[k][1],
                        command=rows[k]["rho_cmd"], own_flat_sim=OWN[k], rho_scan=rows[k]["rho_scan"],
                        phi_used=USED[k][0], z_used=USED[k][1]) for k in range(1, NH + 1)},
          open(Path(OUT).with_suffix(".closure.json"), "w"), indent=1)

# ---- plotting
INK, INK2, GRID = "#1f2328", "#59636e", "#d0d7de"
STEP_X = 2                                 # 2 deg x 1 mm for the 3D surfaces
# stepped scales (a GIF frame holds 256 colours): 0.5 mm bins for the radius
# change (-5..+1.5, white = within +-0.5 mm), 0.2 mm bins for the difference
from matplotlib.colors import BoundaryNorm, ListedColormap
_rb = plt.get_cmap("RdBu_r")
LEV_D = np.arange(-5.0, 1.5 + 1e-9, 0.5)
CM_D = ListedColormap([_rb(0.5 * i / 9.0) for i in range(9)] + ["#f7f7f7", "#f7f7f7"] + [_rb(0.5 + 0.5 * (i + 1) / 3.0) for i in range(2)])
NORM_D = BoundaryNorm(LEV_D, CM_D.N)
_po = plt.get_cmap("PuOr_r")
LEV_E = np.arange(-1.0, 1.0 + 1e-9, 0.2)
CM_E = ListedColormap([_po(0.05 + 0.45 * i / 4.0) for i in range(4)] + ["#f7f7f7", "#f7f7f7"] + [_po(0.5 + 0.45 * (i + 1) / 4.0) for i in range(4)])
CM_E.set_under(_po(0.0)); CM_E.set_over(_po(1.0))
CM_D.set_under(_rb(0.0)); CM_D.set_over(_rb(1.0))
NORM_E = BoundaryNorm(LEV_E, CM_E.N)
NF = 18                                    # frames per state (20 deg per frame)

fig = plt.figure(figsize=(11.0, 11.9), dpi=80)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], width_ratios=[1, 0.025],
                      left=0.07, right=0.92, top=0.958, bottom=0.138, hspace=0.10, wspace=0.03)
axt = fig.add_axes([0.07, 0.014, 0.85, 0.082])
ax3 = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
axm = fig.add_subplot(gs[2, 0])
cb1 = fig.add_subplot(gs[0:2, 1])
cb2 = fig.add_subplot(gs[2, 1])
fig.colorbar(plt.cm.ScalarMappable(norm=NORM_D, cmap=CM_D), cax=cb1, extend="both").set_label("radius change from initial (mm)", fontsize=9, color=INK2)
fig.colorbar(plt.cm.ScalarMappable(norm=NORM_E, cmap=CM_E), cax=cb2, extend="both").set_label("r_sim - r_scan (mm)", fontsize=9, color=INK2)
for c in (cb1, cb2):
    c.tick_params(labelsize=8, colors=INK2)

xt = xs[::STEP_X]


# orthographic camera: c points from the scene to the camera; bar axis = x
import os
MESHVIEW = bool(os.environ.get("MESH"))        # mesh view: a little more from the end, so that the section cut shows the O-grid
CAM = np.array([-0.45 if MESHVIEW else -0.22, -1.0, 0.42]); CAM /= np.linalg.norm(CAM)
RIGHT = np.array([1.0, 0.0, 0.0]) - CAM[0] * CAM; RIGHT /= np.linalg.norm(RIGHT)   # bar axis horizontal on screen
UP = np.cross(CAM, RIGHT)
LIGHT = CAM + np.array([0.3, 0.0, 0.6]); LIGHT /= np.linalg.norm(LIGHT)
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon


def surf(ax, R, R0, omega, title):
    ax.cla()
    Rs = np.vstack([R, R[:1]])[:, ::STEP_X]
    D = np.vstack([R - R0, (R - R0)[:1]])[:, ::STEP_X]
    T = np.radians(np.r_[ths, ths[0] + 360.0])[:, None] + omega
    X = np.broadcast_to(xt[None, :], Rs.shape)
    P = np.stack([X, Rs * np.cos(T), Rs * np.sin(T)], -1)            # (nth+1, nx, 3)
    q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2)  # quads (nth, nx-1, 4, 3)
    n = np.cross(q[:, :, 2] - q[:, :, 0], q[:, :, 1] - q[:, :, 3])
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ctr = q.mean(2)
    radial = ctr.copy(); radial[..., 0] = 0.0
    n *= np.sign(np.sum(n * radial, -1, keepdims=True))                # outward
    vis = (n @ CAM) > 0.0
    d = 0.25 * (D[:-1, :-1] + D[:-1, 1:] + D[1:, 1:] + D[1:, :-1])
    rgb = CM_D(NORM_D(np.nan_to_num(d)))[..., :3]
    lam = np.clip(n @ LIGHT, 0.0, 1.0)[..., None]
    rgb = np.clip(rgb * (0.45 + 0.65 * lam), 0, 1)
    depth = ctr @ CAM
    qv, rv, dv = q[vis], rgb[vis], depth[vis]
    o = np.argsort(dv)
    uv = np.stack([qv @ RIGHT, qv @ UP], -1)[o]
    ax.add_collection(PolyCollection(uv, facecolors=rv[o], edgecolors=rv[o], linewidths=0.3))
    # end cap facing the camera (bar end at x0 or x1)
    j = 0 if CAM[0] < 0 else -1
    ring = P[:, j]
    ax.add_patch(Polygon(np.stack([ring @ RIGHT, ring @ UP], -1), closed=True, fc="#9aa3ab", ec="#7d868f", lw=0.5))
    allp = P.reshape(-1, 3)
    u_, v_ = allp @ RIGHT, allp @ UP
    ax.set_xlim(u_.min() - 1, u_.max() + 1); ax.set_ylim(v_.min() - 0.5, v_.max() + 0.5)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.text(0.0, 1.0, title, transform=ax.transAxes, ha="left", va="top", fontsize=11, color=INK)
    return u_.min() - 1, u_.max() + 1, v_.min() - 0.5, v_.max() + 0.5


# ---- finite-element mesh view of the simulation (MESH=1)
HEXF = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
R_INIT = float(np.nanmean(SM[0]))


def mesh_faces(path):
    """Skin (and the two section cuts) of the HEX8 mesh restricted to X0 <= x <= X1, centred on the slab centre line."""
    m = meshio.read(path)
    P, C = m.points.astype(float), m.cells_dict["hexahedron"]
    S = sim(path, n=6)
    xc = np.arange(np.floor(P[:, 0].min()), np.ceil(P[:, 0].max()) + 1e-9, 1.0)
    cen = np.full((len(xc), 2), np.nan)
    for i, x0 in enumerate(xc):
        a, b = np.searchsorted(S[:, 0], [x0 - 0.5, x0 + 0.5])
        if b - a >= 50:
            cen[i] = slab_centre(S[a:b, 1:3])
    ok = np.isfinite(cen[:, 0]) & (xc >= P[:, 0].min() + 1.0) & (xc <= P[:, 0].max() - 1.0)   # not the end faces
    Pc = np.stack([P[:, 0], P[:, 1] - np.interp(P[:, 0], xc[ok], cen[ok, 0]),
                   P[:, 2] - np.interp(P[:, 0], xc[ok], cen[ok, 1])], 1)
    xcell = P[C][:, :, 0].mean(1)
    Ck = C[(xcell >= X0) & (xcell <= X1)]
    faces = Ck[:, HEXF].reshape(-1, 4)
    _, inv, cnt = np.unique(np.sort(faces, 1), axis=0, return_inverse=True, return_counts=True)
    bsel = cnt[inv.ravel()] == 1
    bf, owner = faces[bsel], np.repeat(np.arange(len(Ck)), 6)[bsel]
    allf = np.sort(C[:, HEXF].reshape(-1, 4), 1)
    _, inv2, cnt2 = np.unique(allf, axis=0, return_inverse=True, return_counts=True)
    skin = set(map(tuple, allf[cnt2[inv2.ravel()] == 1]))
    cut = np.array([tuple(k) not in skin for k in np.sort(bf, 1)])
    off = np.abs(cen[ok]).max() if ok.any() else 0.0
    print(f"mesh {Path(path).name}: {len(P)} nodes / {len(C)} cells, shown {len(Ck)} cells, {int((~cut).sum())} skin + "
          f"{int(cut.sum())} section faces, centre line within {off:.2f} mm of the axis", flush=True)
    return Pc[bf], cut, Pc[Ck[owner]].mean(1), len(P), len(C)


def surf_mesh(ax, MF, omega, title):
    ax.cla()
    q, cut, cc, nn, nc = MF
    c_, s_ = np.cos(omega), np.sin(omega)
    Rm = np.array([[1.0, 0.0, 0.0], [0.0, c_, -s_], [0.0, s_, c_]])
    qr, ccr = q @ Rm.T, cc @ Rm.T
    n = np.cross(qr[:, 2] - qr[:, 0], qr[:, 1] - qr[:, 3])
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ctr = qr.mean(1)
    n *= np.sign(np.sum(n * (ctr - ccr), -1, keepdims=True))            # outward from the owning cell
    vis = (n @ CAM) > 0.0
    rgb = np.array(CM_D(NORM_D(np.hypot(q[:, :, 1], q[:, :, 2]).mean(1) - R_INIT)))[:, :3]
    rgb = np.clip(rgb * (0.45 + 0.65 * np.clip(n @ LIGHT, 0.0, 1.0)[:, None]), 0, 1)
    rgb[cut] = (0.84, 0.86, 0.89)                                       # section cuts: flat light grey, not shaded
    o = np.argsort((ctr @ CAM)[vis])
    uv = np.stack([qr[vis] @ RIGHT, qr[vis] @ UP], -1)[o]
    ax.add_collection(PolyCollection(uv, facecolors=rgb[vis][o], edgecolors=(0.12, 0.14, 0.16, 0.6), linewidths=0.35))
    allp = qr.reshape(-1, 3)
    u_, v_ = allp @ RIGHT, allp @ UP
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.text(0.0, 1.0, f"{title}: FE mesh of this state, {nn:,} nodes / {nc:,} hexahedra, cut to the scanned length",
            transform=ax.transAxes, ha="left", va="top", fontsize=11, color=INK)
    return u_.min() - 1, u_.max() + 1, v_.min() - 0.5, v_.max() + 0.5


MF = ([mesh_faces(MESH0)] + [mesh_faces(f"{SIM}/sim_{k:04d}.vtu") for k in range(1, NH + 1)]) if MESHVIEW else None


def draw_pair(s, om):
    la = surf(ax3[0], SC[s], SC[0], om, "experiment (scan)")
    if MESHVIEW:
        lb = surf_mesh(ax3[1], MF[s], om, "simulation")
        for ax in ax3:                                                  # same window = same scale and x alignment
            ax.set_xlim(min(la[0], lb[0]), max(la[1], lb[1])); ax.set_ylim(min(la[2], lb[2]), max(la[3], lb[3]) + 3.2)  # room for the title
    else:
        surf(ax3[1], SM[s], SM[0], om, "simulation")


def diffmap(s):
    axm.cla()
    E = SM[s] - SC[s]
    axm.imshow(E, origin="lower", aspect="auto", extent=(X0 - DX / 2, X1 + DX / 2, 0, 360), cmap=CM_E, norm=NORM_E,
               interpolation="nearest")
    for k in range(1, s + 1):
        phi, zc = USED[k]
        for ph in (phi % 360.0, (phi + 180.0) % 360.0):
            axm.plot([zc - 3.54, zc + 3.54], [ph, ph], color=INK, lw=1.2)
            axm.text(zc + 4.3, min(max(ph, 9.0), 349.0), f"{k}", fontsize=8, color=INK, ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=GRID, lw=0.6, alpha=0.9))
    axm.set_xlim(X0, X1); axm.set_ylim(0, 360); axm.set_yticks([0, 90, 180, 270, 360])
    axm.set_xlabel("x along the bar (mm)", fontsize=9, color=INK2)
    axm.set_ylabel("angle (deg)", fontsize=9, color=INK2)
    axm.tick_params(labelsize=8, colors=INK2)
    for sp in axm.spines.values():
        sp.set_color(GRID)
    axm.set_title("unrolled difference of the current state (black bars: die centres in the simulation, 7.08 mm flat, "
                  "numbered by hit)", fontsize=9.5, color=INK2, loc="left")
    # closure table: hits done so far, the latest one in the darker ink
    axt.cla(); axt.set_axis_off(); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.text(0.0, 0.98, "closure half-width kept after each hit (mm):   command / experiment / simulation",
             fontsize=9.5, color=INK2, ha="left", va="top")
    for k in range(1, s + 1):
        col, row = (k - 1) // 3, (k - 1) % 3
        axt.text(0.005 + 0.255 * col, 0.64 - 0.25 * row,
                 f"hit {k:>2}: {rows[k]['rho_cmd']:5.2f} / {rows[k]['rho_scan']:5.2f} / {OWN[k]:5.2f}",
                 fontsize=8, family="monospace", color=INK if k == s else INK2, ha="left", va="center")


STATES = ["before hit 1 (initial bar)"] + [f"after hit {k}" for k in range(1, NH + 1)]
import os
if os.environ.get("STILL"):
    s_ = int(os.environ.get("STILL_STATE", NH))
    diffmap(s_)
    draw_pair(s_, np.radians(float(os.environ.get("STILL_DEG", 0.0))))
    fig.suptitle(f"{Path(H5).stem}, {STATES[s_]}   {LABEL}", fontsize=12, color=INK, x=0.07, ha="left")
    fig.savefig(os.environ["STILL"], dpi=80)
    sys.exit(0)
writer = PillowWriter(fps=8)
with writer.saving(fig, OUT, dpi=80):
    for s in range(NH + 1):
        diffmap(s)
        for f in range(NF):
            om = np.radians(360.0 * f / NF)
            draw_pair(s, om)
            fig.suptitle(f"{Path(H5).stem}, {STATES[s]}   {LABEL}", fontsize=12, color=INK, x=0.07, ha="left")
            writer.grab_frame()
        print(f"state {s} done", flush=True)
    # last frame of every state as a still too
fig.savefig(Path(OUT).with_suffix(".png"), dpi=100)
print("wrote", OUT)
