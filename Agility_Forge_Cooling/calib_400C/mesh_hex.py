"""Structured HEX8 billet: an O-grid cross-section extruded along the bar axis.

Why this exists
---------------
On constant-strain TET4 the F-bar correction in lib/constitutive.py is the
identity, so the element locks under isochoric plastic flow -- that is what
spreads every imprint into a 5-8 mm ramp.  F-bar was designed for the 8-node
hexahedron, where it does act, and the package's centre point (0.5, 0.5, 0.5)
is correct for a hexahedron.  This builds the hex mesh for that test.

Cross-section: a central square of half-side s = core * R with n x n cells,
surrounded by four ring blocks that blend the square's edges into the circle
over m radial layers.  Each quarter arc carries n cells, so the circumferential
spacing on the surface is pi*R / (2n).  The section is extruded over
x in [-pad, 100 + pad] with nx uniform layers -- the same extent as mesh.py, so
the axial restraint on the far face stays clear of the contact zone.

Cells are written in meshio/VTK hexahedron order with positive volume.
run_cold.py permutes them into jax_fem_checkpoint's own order (lib/hex8.py).

Usage
-----
    python mesh_hex.py --h 2.8 --out ../data_400C/calibration/billet_hex_h28.msh
    python mesh_hex.py --h 1.4 --out ../data_400C/calibration/billet_hex_h14.msh
"""
import argparse
import sys
from pathlib import Path

import meshio
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from paths import OUT, R_STOCK_MM
from hex8 import hex_volumes, VTK_FACES

FORGE_WINDOW = (0.0, 100.0)


def ogrid_section(R, n, m, core):
    """-> (points2d (k, 2) in (y, z), quads (q, 4) counter-clockwise seen from +x)."""
    s = core * R
    pts, quads = [], []

    def add(p):
        pts.append(p)
        return len(pts) - 1

    # central square
    g = onp.linspace(-s, s, n + 1)
    sq = onp.array([[add((y, z)) for z in g] for y in g])          # sq[i, j] -> (y_i, z_j)
    for i in range(n):
        for j in range(n):
            quads.append((sq[i, j], sq[i + 1, j], sq[i + 1, j + 1], sq[i, j + 1]))

    # four ring blocks; block 0 sits on the +y side, the others are rotations
    t = onp.linspace(0.0, 1.0, n + 1)
    rho = onp.linspace(0.0, 1.0, m + 1)
    for k in range(4):
        c, sn = onp.cos(k * onp.pi / 2), onp.sin(k * onp.pi / 2)
        rot = onp.array([[c, -sn], [sn, c]])
        inner = onp.stack([onp.full_like(t, s), -s + 2 * s * t], axis=1)      # square edge
        ang = -onp.pi / 4 + t * onp.pi / 2
        outer = R * onp.stack([onp.cos(ang), onp.sin(ang)], axis=1)          # arc
        ids = onp.empty((n + 1, m + 1), dtype=int)
        for a in range(n + 1):
            for b in range(m + 1):
                p = (1 - rho[b]) * inner[a] + rho[b] * outer[a]
                ids[a, b] = add(tuple(rot @ p))
        for a in range(n):
            for b in range(m):
                quads.append((ids[a, b], ids[a, b + 1], ids[a + 1, b + 1], ids[a + 1, b]))

    P = onp.array(pts)
    Q = onp.array(quads)
    # merge coincident nodes (square boundary == ring layer 0, shared block seams)
    key = onp.round(P / 1e-9).astype(onp.int64)
    _, first, inv = onp.unique(key, axis=0, return_index=True, return_inverse=True)
    P = P[first]
    Q = inv.reshape(-1)[Q]
    # orientation: counter-clockwise in (y, z), i.e. normal along +x
    a = P[Q]
    area = 0.5 * onp.sum(a[:, :, 0] * onp.roll(a[:, :, 1], -1, axis=1)
                         - onp.roll(a[:, :, 0], -1, axis=1) * a[:, :, 1], axis=1)
    Q[area < 0] = Q[area < 0][:, ::-1]
    return P, Q


def extrude(P2, Q, xs):
    n2 = len(P2)
    pts = onp.concatenate([onp.column_stack([onp.full(n2, x), P2]) for x in xs], axis=0)
    cells = []
    for k in range(len(xs) - 1):
        lo, hi = k * n2, (k + 1) * n2
        cells.append(onp.column_stack([Q + lo, Q + hi]))           # VTK: bottom face, top face
    return pts, onp.concatenate(cells, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, required=True,
                    help="target edge length on the outer surface and along the axis, mm")
    ap.add_argument("--core", type=float, default=0.5,
                    help="half-side of the central square as a fraction of R")
    ap.add_argument("--pad", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=R_STOCK_MM)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    R = args.radius
    n = max(2, int(round(onp.pi * R / 2 / args.h)))                  # cells per quarter arc
    m = max(1, int(round(R * (1 - args.core) / args.h)))             # radial ring layers
    x_lo, x_hi = FORGE_WINDOW[0] - args.pad, FORGE_WINDOW[1] + args.pad
    nx = max(1, int(round((x_hi - x_lo) / args.h)))
    P2, Q = ogrid_section(R, n, m, args.core)
    pts, cells = extrude(P2, Q, onp.linspace(x_lo, x_hi, nx + 1))

    vol = hex_volumes(pts, cells)
    e = onp.vstack([cells[:, f][:, [0, 1]] for f in VTK_FACES] +
                   [cells[:, f][:, [1, 2]] for f in VTK_FACES])
    e.sort(axis=1)
    e = onp.unique(e, axis=0)
    L = onp.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
    r = onp.hypot(pts[:, 1], pts[:, 2])
    surf_e = (r[e[:, 0]] > R - 1e-6) & (r[e[:, 1]] > R - 1e-6)
    print(f"O-grid: {n} cells per quarter arc, {m} ring layers, core half-side {args.core*R:.2f} mm, "
          f"{nx} axial layers")
    print(f"  {len(pts):,} nodes, {len(cells):,} hexahedra, {3*len(pts):,} displacement dofs")
    print(f"  edge length: median {onp.median(L):.2f} mm (range {L.min():.2f}-{L.max():.2f}); "
          f"on the outer surface median {onp.median(L[surf_e]):.2f} mm; axial {(x_hi-x_lo)/nx:.2f} mm")
    print(f"  volume {vol.sum():.1f} mm^3 (cylinder {onp.pi*R**2*(x_hi-x_lo):.1f}); "
          f"min cell volume {vol.min():.3e}, non-positive: {(vol <= 0).sum()}")
    assert (vol > 0).all(), "non-positive hex volume"
    meshio.Mesh(pts, {"hexahedron": cells}).write(args.out, file_format="gmsh22", binary=False)
    back = meshio.read(args.out)
    assert onp.array_equal(back.cells_dict["hexahedron"], cells), "hex ordering changed on write"
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
