"""Inter-hit remeshing of the forged bar: a fresh structured HEX8 mesh fitted
to the deformed outer surface.

Why this exists
---------------
Between hits the bar is reheated, so the state is reset (run_cold
--hit-reset full) and the only thing a hit hands to the next one is its shape.
That makes the start of a hit the natural place for a true remesh: nothing has
to be mapped, the new mesh only has to reproduce the surface.  Node relocation
on the fixed connectivity (lib/mesh_smooth.py) cannot change where the layers
are, and the 1 mm die walls and 9 mm grooves of the deep hits need axial
layers of ~0.5 mm there and ~3 mm elsewhere.

Method
------
The bar stays a sweepable solid: every cross-section perpendicular to the bar
axis is a star-shaped outline (the circle flattened by the dies, bulged at the
sides).  So the mesh of mesh_hex.py -- an O-grid section extruded along x --
is simply regenerated on the deformed body:

  1. the lateral boundary of the current mesh (deformed coordinates) is cut
     by planes x = x_j at chosen axial stations; each cut is a closed
     polyline, sampled by rays from its centroid at the O-grid's angles, so
     every new surface node lies ON the old faceted surface;
  2. stations are placed by a spacing field: h_fine inside the "feature
     zones" (where the old surface is steep in x, i.e. the walls of existing
     imprints, and around the walls of the hit about to come), h_coarse
     elsewhere, graded in between;
  3. the O-grid section (central square + four ring blocks, optionally biased
     towards the surface) is built on each outline with the same 2-D
     connectivity, and consecutive stations are joined into hexahedra.

The result is a new connectivity (the number of axial layers changes), so
the caller rebuilds the problem from scratch; with --hit-reset full there is
no state to transfer.  Cells are returned in meshio/VTK order with positive
volume.  Units: mm.
"""
import numpy as onp

from hex8 import VTK_FACES, hex_volumes, outer_surface_nodes_hex
from mesh_quality import scaled_jacobian


# ----------------------------------------------------------------------------
# surface
# ----------------------------------------------------------------------------
def template_from_original(pts0, cells):
    """Read the O-grid template (n cells per quarter arc, m ring layers) and
    the lateral node set INCLUDING the rims off the virgin extruded mesh of
    mesh_hex.py, where the stations are planes and the caps are at the two
    x extremes.  -> dict(n, m, lateral_nodes)."""
    X = onp.asarray(pts0, dtype=float)
    C = onp.asarray(cells)
    F = onp.concatenate([C[:, f] for f in VTK_FACES])
    key = onp.sort(F, axis=1)
    _, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    Q = F[idx[cnt == 1]]
    xq = X[Q][:, :, 0]
    cap = (onp.abs(xq - X[:, 0].min()) < 1e-6).all(axis=1) | (onp.abs(xq - X[:, 0].max()) < 1e-6).all(axis=1)
    lateral_nodes = onp.unique(Q[~cap])
    n_sec = len(onp.unique(onp.round(X[:, 0] / 1e-6)))
    n = len(lateral_nodes) // n_sec // 4
    m = int(round((len(X) // n_sec - (n + 1) ** 2) / (4 * n)))
    assert (n + 1) ** 2 + 4 * n * m == len(X) // n_sec, "not an O-grid extrusion"
    return dict(n=int(n), m=int(m), lateral_nodes=lateral_nodes)


def boundary_split(x, cells, lateral_nodes):
    """Boundary of a VTK hex mesh at coordinates x, split topologically into the
    lateral surface (faces whose four nodes are in `lateral_nodes`, the rims
    included) and the two end caps, as triangle arrays (n, 3, 3).  The caps
    are told apart by their mean x.  The free end is not plane after a few
    hits (it bulges by millimetres), so nothing here reads the caps off
    coordinates."""
    C = onp.asarray(cells)
    F = onp.concatenate([C[:, f] for f in VTK_FACES])
    key = onp.sort(F, axis=1)
    _, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    Q = F[idx[cnt == 1]]
    on_surf = onp.isin(Q, onp.asarray(lateral_nodes, dtype=int)).all(axis=1)
    X = onp.asarray(x, dtype=float)
    tri = lambda q: onp.concatenate([X[q][:, [0, 1, 2]], X[q][:, [0, 2, 3]]], axis=0)
    caps = Q[~on_surf]
    xm = X[caps][:, :, 0].mean(axis=1)
    cut = 0.5 * (xm.min() + xm.max())
    return tri(Q[on_surf]), tri(caps[xm < cut]), tri(caps[xm >= cut])


def cap_height(cap_tris, yz, fallback):
    """x of the cap surface above the (y, z) points: barycentric interpolation
    in the triangle whose (y, z) projection contains the point; the nearest cap
    vertex where none does (rim round-off).  fallback: x used if the cap is empty."""
    if len(cap_tris) == 0:
        return onp.full(len(yz), float(fallback))
    A, B, Cc = cap_tris[:, 0, 1:], cap_tris[:, 1, 1:], cap_tris[:, 2, 1:]
    v0, v1 = B - A, Cc - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    out = onp.empty(len(yz))
    verts = cap_tris.reshape(-1, 3)
    for i, p in enumerate(yz):
        v2 = p[None, :] - A
        d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
        with onp.errstate(divide="ignore", invalid="ignore"):
            v = (d11 * d20 - d01 * d21) / den
            w = (d00 * d21 - d01 * d20) / den
        u = 1.0 - v - w
        inside = onp.where((u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6))[0]
        if len(inside):
            k = inside[0]
            out[i] = u[k] * cap_tris[k, 0, 0] + v[k] * cap_tris[k, 1, 0] + w[k] * cap_tris[k, 2, 0]
        else:
            out[i] = verts[onp.argmin(onp.hypot(*(verts[:, 1:] - p).T)), 0]
    return out


def plane_cut(T, xs):
    """Segments (k, 2, 2) in (y, z) where the plane x = xs cuts the triangles T."""
    X = T[:, :, 0]
    above = X > xs
    straddle = (above.sum(axis=1) % 3) != 0
    if not straddle.any():
        return onp.zeros((0, 2, 2))
    Tt = T[straddle]
    ab = above[straddle]
    segs = []
    for tri, a in zip(Tt, ab):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if a[i] != a[j]:
                t = (xs - tri[i, 0]) / (tri[j, 0] - tri[i, 0])
                pts.append(tri[i, 1:] + t * (tri[j, 1:] - tri[i, 1:]))
        if len(pts) == 2:
            segs.append(pts)
    return onp.asarray(segs, dtype=float).reshape(-1, 2, 2)


def outline_radii(segs, angles):
    """Centroid c of the cut and the distance r(theta) from c to the outline
    along each angle (the farthest crossing: the outline is star-shaped)."""
    mid = segs.mean(axis=1)
    L = onp.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    c = (mid * L[:, None]).sum(axis=0) / L.sum()
    d = onp.stack([onp.cos(angles), onp.sin(angles)], axis=1)        # (a, 2)
    A = segs[:, 0] - c                                                # (k, 2)
    E = segs[:, 1] - segs[:, 0]                                       # (k, 2)
    # solve  t d = A + s E  ->  [d, -E] [t, s]^T = A   for every (angle, segment)
    det = d[:, None, 0] * (-E[None, :, 1]) - d[:, None, 1] * (-E[None, :, 0])   # (a, k)
    with onp.errstate(divide="ignore", invalid="ignore"):
        t = (A[None, :, 0] * (-E[None, :, 1]) - A[None, :, 1] * (-E[None, :, 0])) / det
        s = (d[:, None, 0] * A[None, :, 1] - d[:, None, 1] * A[None, :, 0]) / det
    ok = onp.isfinite(t) & (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
    t = onp.where(ok, t, -onp.inf)
    r = t.max(axis=1)
    if not onp.isfinite(r).all():
        raise RuntimeError("outline not star-shaped about its centroid (a ray misses it)")
    # Area preservation.  The sampled polygon is inscribed in the old outline
    # at shifted angles and loses ~(2 pi / N)^2 / 6 of the area (0.3 % for
    # N = 44, i.e. 0.4 % of the volume per remesh).  The old outline's area is
    # exact from its segments (shoelace about c, orientation-free), and the
    # radii are scaled uniformly to match it: a surface shift of ~0.2 % of r
    # (0.03 mm), the volume kept to round-off.
    area_old = 0.5 * onp.sum(onp.abs(A[:, 0] * E[:, 1] - A[:, 1] * E[:, 0]))   # fan of triangles about c
    dth = onp.diff(onp.concatenate([angles, [angles[0] + 2 * onp.pi]]))
    r_next = onp.roll(r, -1)
    area_new = 0.5 * onp.sum(r * r_next * onp.sin(dth))
    r = r * onp.sqrt(area_old / area_new)
    return c, r


# ----------------------------------------------------------------------------
# stations
# ----------------------------------------------------------------------------
def feature_zones(T, slope_deg=20.0, margin=0.5):
    """x-intervals where the old lateral surface is steep in x (imprint walls).
    slope_deg <= 0 disables this: an existing wall is then spanned by coarse
    layers (one 2.8 mm layer over a 4 mm drop has SJ ~0.5), whereas several
    fine layers on an 80 deg wall are all skewed to SJ ~0.2 -- the skew is set
    by the wall slope, not by the layer thickness."""
    if slope_deg <= 0:
        return []
    nrm = onp.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= onp.linalg.norm(nrm, axis=1)[:, None] + 1e-30
    steep = onp.abs(nrm[:, 0]) > onp.sin(onp.radians(slope_deg))
    if not steep.any():
        return []
    X = T[steep][:, :, 0]
    iv = onp.stack([X.min(axis=1) - margin, X.max(axis=1) + margin], axis=1)
    iv = iv[onp.argsort(iv[:, 0])]
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(m) for m in merged]


def stations(x_lo, x_hi, zones, h_coarse, h_fine, grade=1.3, dx=0.02):
    """Axial stations from a spacing field: h_fine inside the zones, h_coarse
    elsewhere, neighbouring spacings within the ratio `grade`."""
    g = onp.arange(x_lo, x_hi + dx, dx)
    s = onp.full(len(g), float(h_coarse))
    for a, b in zones:
        s[(g >= a) & (g <= b)] = h_fine
    slope = grade - 1.0                       # ds/dx <= slope  <=> ratio of neighbours <= grade
    for i in range(1, len(s)):
        s[i] = min(s[i], s[i - 1] + slope * dx)
    for i in range(len(s) - 2, -1, -1):
        s[i] = min(s[i], s[i + 1] + slope * dx)
    cum = onp.concatenate([[0.0], onp.cumsum(2.0 * dx / (s[1:] + s[:-1]))])
    N = max(2, int(onp.ceil(cum[-1])))
    xs = onp.interp(onp.linspace(0.0, cum[-1], N + 1), cum, g)
    xs[0], xs[-1] = x_lo, x_hi
    return xs


# ----------------------------------------------------------------------------
# O-grid section on an arbitrary star-shaped outline
# ----------------------------------------------------------------------------
def ogrid_template(n, m):
    """2-D connectivity of the O-grid (shared by all stations) and, per node,
    (kind, i, j): kind 0 = central square (i, j on the (n+1)^2 grid), kind 1 =
    ring block k (encoded in i = k*(n+1)+a, j = b).  Returns (node table,
    quads, outer node ids in angle order, angles)."""
    nodes, quads, lookup = [], [], {}

    def nid(key):
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(key)
        return lookup[key]

    for i in range(n):
        for j in range(n):
            quads.append((nid(("sq", i, j)), nid(("sq", i + 1, j)), nid(("sq", i + 1, j + 1)), nid(("sq", i, j + 1))))
    # ring blocks: block k spans global angles -45 + 90k .. 45 + 90k deg;
    # its inner edge is the square's side k, its layer-0 nodes ARE square nodes
    def ring_key(k, a, b):
        if b == 0:                               # on the square
            # square side k, parameter a along it (counter-clockwise)
            if k == 0:   return ("sq", n, a)
            if k == 1:   return ("sq", n - a, n)
            if k == 2:   return ("sq", 0, n - a)
            return ("sq", a, 0)
        if a == 0 and k > 0:                     # seam with the previous block
            return ("rg", k - 1, n, b)
        if a == n and k == 3:                    # seam closing the ring
            return ("rg", 0, 0, b)
        return ("rg", k, a, b)
    for k in range(4):
        for a in range(n):
            for b in range(m):
                quads.append((nid(ring_key(k, a, b)), nid(ring_key(k, a, b + 1)),
                              nid(ring_key(k, a + 1, b + 1)), nid(ring_key(k, a + 1, b))))
    outer, angles = [], []
    for k in range(4):
        for a in range(n):
            outer.append(nid(ring_key(k, a, m)))
            angles.append(-onp.pi / 4 + k * onp.pi / 2 + a / n * onp.pi / 2)
    return nodes, onp.asarray(quads), onp.asarray(outer), onp.asarray(angles)


def section_points(nodes, n, m, c, r_outer, angles, half_side, surf_bias=1.0):
    """(y, z) of every template node for an outline given by r(angle) about c."""
    s = half_side
    gsq = onp.linspace(-s, s, n + 1)
    rho = (onp.arange(m + 1) / m) ** (1.0 / surf_bias)
    outer_xy = c + r_outer[:, None] * onp.stack([onp.cos(angles), onp.sin(angles)], axis=1)
    P = onp.zeros((len(nodes), 2))
    for idx, key in enumerate(nodes):
        if key[0] == "sq":
            P[idx] = c + onp.array([gsq[key[1]], gsq[key[2]]])
        else:
            _, k, a, b = key
            # inner point on square side k at parameter a; outer point on the outline
            t = a / n
            if k == 0:   inner = onp.array([s, -s + 2 * s * t])
            elif k == 1: inner = onp.array([s - 2 * s * t, s])
            elif k == 2: inner = onp.array([-s, s - 2 * s * t])
            else:        inner = onp.array([-s + 2 * s * t, -s])
            ia = k * n + a
            P[idx] = c + (1 - rho[b]) * inner + rho[b] * (outer_xy[ia] - c)
    return P


def orient_ccw(P, Q):
    a = P[Q]
    area = 0.5 * onp.sum(a[:, :, 0] * onp.roll(a[:, :, 1], -1, axis=1)
                         - onp.roll(a[:, :, 0], -1, axis=1) * a[:, :, 1], axis=1)
    Q = Q.copy()
    Q[area < 0] = Q[area < 0][:, ::-1]
    return Q


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def remesh_ogrid(x, cells, template, h_coarse, h_fine, core=0.5, surf_bias=1.0,
                 zones_extra=(), grade=1.3, slope_deg=20.0, zone_margin=0.5, verbose=True):
    """x: deformed node coordinates of the current VTK hex mesh `cells`;
    template: dict(n, m, lateral_nodes) of that mesh (template_from_original,
    or the `template` entry of the previous remesh report).
    h_coarse / h_fine: axial layer thickness away from / inside feature zones.
    zones_extra: additional x-intervals to refine (the walls of the next hit).
    -> (points, cells_vtk, surf_nodes, report); report["template"] describes
    the new mesh the same way, for the next remesh.
    The end caps are kept as they are (the free end bulges): the lateral
    surface is sectioned up to the rim, and the two end stations take their
    x from the old cap surface above each node."""
    x = onp.asarray(x, dtype=float)
    cells = onp.asarray(cells)
    n, m = int(template["n"]), int(template["m"])
    T, cap_lo, cap_hi = boundary_split(x, cells, template["lateral_nodes"])
    # the lateral outline is complete between the innermost rim points
    rim_lo = cap_lo[:, :, 0].max() if len(cap_lo) else x[:, 0].min()
    rim_hi = cap_hi[:, :, 0].min() if len(cap_hi) else x[:, 0].max()
    x_lo, x_hi = float(rim_lo) + 0.05, float(rim_hi) - 0.05
    zones = feature_zones(T, slope_deg, zone_margin) + [tuple(z) for z in zones_extra]
    xs = stations(x_lo, x_hi, zones, h_coarse, h_fine, grade)
    nodes, Q2, outer_ids, angles = ogrid_template(n, m)
    n2 = len(nodes)
    pts = onp.zeros((len(xs), n2, 3))
    r_min_all = onp.inf
    for j, xj in enumerate(xs):
        segs = plane_cut(T, xj)
        if len(segs) < 3:
            raise RuntimeError(f"empty cut at x = {xj:.3f}")
        c, r = outline_radii(segs, angles)
        r_min_all = min(r_min_all, float(r.min()))
        half_side = min(core * float(r.mean()), 0.6 * float(r.min()))
        P2 = section_points(nodes, n, m, c, r, angles, half_side, surf_bias)
        pts[j, :, 0] = xj
        pts[j, :, 1:] = P2
    # end stations: the old cap surface above each node (plane caps stay plane)
    pts[0, :, 0] = onp.minimum(cap_height(cap_lo, pts[0, :, 1:], x_lo), x_lo)
    pts[-1, :, 0] = onp.maximum(cap_height(cap_hi, pts[-1, :, 1:], x_hi), x_hi)
    cap_dev = (float(onp.abs(pts[0, :, 0] - x_lo).max()), float(onp.abs(pts[-1, :, 0] - x_hi).max()))
    Q2 = orient_ccw(pts[0, :, 1:], Q2)
    C = []
    for k in range(len(xs) - 1):
        lo, hi = k * n2, (k + 1) * n2
        C.append(onp.column_stack([Q2 + lo, Q2 + hi]))
    C = onp.concatenate(C, axis=0)
    # Global volume correction.  Each station's area is exact, but between
    # stations the new surface is a ruled surface where the old one was
    # bilinear on a different layering, which is worth up to +-0.07 % of the
    # volume per remesh (12 remeshes could drift 0.5 %).  Every section is
    # scaled about its own centroid by one factor so the total volume is the
    # old one: a surface shift of ~0.03 % of r, i.e. a few microns.
    vol_old = float(hex_volumes(x, cells).sum())
    cen = pts[:, :, 1:].mean(axis=1, keepdims=True)
    for _ in range(3):
        f = onp.sqrt(vol_old / float(hex_volumes(pts.reshape(-1, 3), C).sum()))
        pts[:, :, 1:] = cen + f * (pts[:, :, 1:] - cen)
    P = pts.reshape(-1, 3)
    vol = hex_volumes(P, C)
    if (vol <= 0).any():
        raise RuntimeError(f"{int((vol <= 0).sum())} non-positive hexahedra in the remesh")
    sj = scaled_jacobian(P, C)
    lateral = (outer_ids[None, :] + n2 * onp.arange(len(xs))[:, None]).reshape(-1)   # rims included
    surf = (outer_ids[None, :] + n2 * onp.arange(1, len(xs) - 1)[:, None]).reshape(-1)  # contact candidates
    rep = dict(n_points=int(len(P)), n_cells=int(len(C)), n_stations=int(len(xs)), n=int(n), m=int(m),
               zones=[(float(a), float(b)) for a, b in zones],
               dx_min=float(onp.diff(xs).min()), dx_max=float(onp.diff(xs).max()),
               volume_old=vol_old, volume_new=float(vol.sum()),
               min_sj=float(sj.min()), n_below_0p4=int((sj < 0.4).sum()),
               r_min=float(r_min_all), old_n_points=int(len(x)), old_n_cells=int(len(cells)),
               cap_height_span=cap_dev,
               old_min_sj=float(scaled_jacobian(x, cells).min()))
    rep["volume_change_rel"] = (rep["volume_new"] - rep["volume_old"]) / rep["volume_old"]
    rep["template"] = dict(n=n, m=m, lateral_nodes=lateral)
    if verbose:
        print(f"  [remesh] O-grid {n} x {m}, {len(xs)} stations (dx {rep['dx_min']:.2f}-{rep['dx_max']:.2f} mm, "
              f"{len(zones)} fine zones): {len(P)} nodes / {len(C)} cells (was {len(x)} / {len(cells)}); "
              f"min SJ {rep['old_min_sj']:.3f} -> {rep['min_sj']:.3f} ({rep['n_below_0p4']} cells < 0.4); "
              f"volume {100 * rep['volume_change_rel']:+.4f} %; end caps rise {cap_dev[0]:.2f} / {cap_dev[1]:.2f} mm "
              f"over the last station", flush=True)
    return P, C, surf, rep
