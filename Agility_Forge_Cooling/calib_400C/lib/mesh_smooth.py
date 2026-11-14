"""Node relocation of a deformed hexahedral mesh at fixed connectivity.

Why this exists
---------------
Between two hits the bar is reheated; with --hit-reset the released shape
becomes a stress-free reference configuration, so nothing has to be carried
from the old mesh to the new one except the SHAPE.  That makes the reheat the
one place where the mesh can be repaired for free: the distortion a hit leaves
in the outer element layer (min scaled Jacobian 0.19 after five hits on mesh A,
all of it in the surface layer) is otherwise inherited by every later hit.

Method
------
Target-matrix shape optimisation.  For every corner of every cell, A holds the
three edge vectors of the deformed corner and W those of the same corner in the
ORIGINAL mesh; T = A W^-1.  The mean-ratio shape metric

    mu(T) = |T|_F^2 / (3 det(T)^(2/3))        (1 for T = scale x rotation)

is minimised (as the mean of mu^p over all corners), so each cell is driven
back to the shape it was designed with, at any size and orientation.  det(T)
is regularised near zero so that a folded corner has a finite, repelling value.
Adam steps on the node positions, with the boundary nodes put back on the OLD
boundary after every step:

    interior nodes          free
    smooth boundary nodes   closest point on the old boundary triangles
    crease nodes            closest point on the old crease polyline (a crease
                            is a boundary edge whose two faces meet at more
                            than `crease_deg`: the bar rims, imprint edges)
    crease ends, junctions  fixed

so the body keeps its boundary surface up to the faceting of the old mesh.
The state of lowest energy is returned, and only if it is no worse than the
input in minimum scaled Jacobian.

Cells in meshio/VTK order.  Units: mm.
"""
import numpy as onp
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree

from lib.hex8 import VTK_FACES, hex_volumes
from lib.mesh_quality import _HEX_CORNERS, scaled_jacobian


# ----------------------------------------------------------------------------
# boundary description of the old mesh
# ----------------------------------------------------------------------------
def boundary_quads(cells):
    """Boundary faces of a VTK hex mesh, outward oriented, (n_faces, 4)."""
    F = onp.concatenate([cells[:, f] for f in VTK_FACES])
    key = onp.sort(F, axis=1)
    _, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    return F[idx[cnt == 1]]


def classify_boundary(x, quads, crease_deg):
    """-> (node class, crease segments).  Class: -1 interior, 0 smooth boundary,
    1 crease (slides along it), 2 fixed."""
    n = len(x)
    P = x[quads]
    nrm = onp.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    nrm /= onp.maximum(onp.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    ed = onp.concatenate([quads[:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))])
    fid = onp.tile(onp.arange(len(quads)), 4)
    ks = onp.sort(ed, axis=1)
    o = onp.lexsort((ks[:, 1], ks[:, 0]))
    ks, fid = ks[o], fid[o]
    if len(ks) % 2 or not onp.array_equal(ks[0::2], ks[1::2]):
        raise ValueError("the boundary is not a closed surface")
    cosang = onp.clip((nrm[fid[0::2]] * nrm[fid[1::2]]).sum(1), -1.0, 1.0)
    crease = ks[0::2][onp.degrees(onp.arccos(cosang)) > crease_deg]
    cls = onp.full(n, -1)
    cls[onp.unique(quads)] = 0
    deg = onp.bincount(crease.ravel(), minlength=n)
    cls[deg == 2] = 1
    cls[(deg > 0) & (deg != 2)] = 2
    return cls, crease


def _closest_on_segments(p, a, b):
    """Closest points to p (n, 1, 3) on segments a-b (n, k, 3) -> (n, k, 3)."""
    ab = b - a
    t = onp.clip(((p - a) * ab).sum(-1) / onp.maximum((ab * ab).sum(-1), 1e-300), 0.0, 1.0)
    return a + t[..., None] * ab


def _closest_on_triangles(p, a, b, c):
    """Closest points to p (n, 1, 3) on triangles (n, k, 3) each -> (n, k, 3)."""
    ab, ac = b - a, c - a
    nrm = onp.cross(ab, ac)
    nn = onp.maximum((nrm * nrm).sum(-1), 1e-300)
    ap = p - a
    q = p - ((ap * nrm).sum(-1) / nn)[..., None] * nrm
    v = q - a
    s = (onp.cross(v, ac) * nrm).sum(-1) / nn
    t = (onp.cross(ab, v) * nrm).sum(-1) / nn
    inside = (s >= 0.0) & (t >= 0.0) & (s + t <= 1.0)
    cand = onp.stack([_closest_on_segments(p, a, b), _closest_on_segments(p, b, c),
                      _closest_on_segments(p, c, a)], axis=0)            # (3, n, k, 3)
    d = ((cand - p[None]) ** 2).sum(-1)
    best = onp.take_along_axis(cand, onp.argmin(d, axis=0)[None, ..., None], axis=0)[0]
    return onp.where(inside[..., None], q, best)


class OldBoundary:
    """Closest-point projection onto the boundary of the mesh at positions x."""

    def __init__(self, x, quads, crease, k_tri=24, k_seg=8):
        self.tri = onp.concatenate([x[quads[:, [0, 1, 2]]], x[quads[:, [0, 2, 3]]]])   # (nt, 3, 3)
        self.tri_tree = cKDTree(self.tri.mean(axis=1))
        self.k_tri = min(k_tri, len(self.tri))
        self.seg = x[crease] if len(crease) else onp.zeros((0, 2, 3))
        self.seg_tree = cKDTree(self.seg.mean(axis=1)) if len(crease) else None
        self.k_seg = min(k_seg, len(self.seg))

    def on_surface(self, p):
        _, idx = self.tri_tree.query(p, k=self.k_tri)
        T = self.tri[idx]                                                # (n, k, 3, 3)
        c = _closest_on_triangles(p[:, None, :], T[:, :, 0], T[:, :, 1], T[:, :, 2])
        d = ((c - p[:, None, :]) ** 2).sum(-1)
        j = onp.argmin(d, axis=1)
        return c[onp.arange(len(p)), j], onp.sqrt(d[onp.arange(len(p)), j])

    def on_crease(self, p):
        _, idx = self.seg_tree.query(p, k=self.k_seg)
        idx = idx.reshape(len(p), -1)
        S = self.seg[idx]
        c = _closest_on_segments(p[:, None, :], S[:, :, 0], S[:, :, 1])
        d = ((c - p[:, None, :]) ** 2).sum(-1)
        j = onp.argmin(d, axis=1)
        return c[onp.arange(len(p)), j]


# ----------------------------------------------------------------------------
# shape energy
# ----------------------------------------------------------------------------
def _corner_frames(x, cells):
    """(n_cells, 8, 3, 3): edge vectors of every corner as matrix columns."""
    P = x[cells]
    return onp.stack([onp.stack([P[:, a] - P[:, c], P[:, b] - P[:, c], P[:, d] - P[:, c]], axis=-1)
                      for c, a, b, d in _HEX_CORNERS], axis=1)


def make_energy(cells, x_ref, power, eps):
    Winv = jnp.asarray(onp.linalg.inv(_corner_frames(x_ref, cells)))     # (nc, 8, 3, 3)
    C = jnp.asarray(cells)
    corners = _HEX_CORNERS

    def energy(x):
        P = x[C]
        tot = 0.0
        for k, (c, a, b, d) in enumerate(corners):
            A = jnp.stack([P[:, a] - P[:, c], P[:, b] - P[:, c], P[:, d] - P[:, c]], axis=-1)
            T = A @ Winv[:, k]
            det = jnp.linalg.det(T)
            det_r = 0.5 * (det + jnp.sqrt(det * det + eps * eps))
            mu = jnp.sum(T * T, axis=(1, 2)) / (3.0 * det_r ** (2.0 / 3.0))
            tot = tot + jnp.sum(mu ** power)
        return tot / (8.0 * C.shape[0])

    return jax.jit(jax.value_and_grad(energy))


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def gauss_volume(x, cells):
    """Volume of the trilinear hexahedra by 2x2x2 Gauss quadrature of det J.
    Exact for the trilinear map, and its sum over the mesh depends on the
    boundary nodes only (shared faces are the same bilinear patch from both
    sides) -- unlike a tetrahedral split, which moved by 0.03 % when only
    interior nodes were relocated."""
    g = 1.0 / onp.sqrt(3.0)
    xi = onp.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
    P = x[cells]
    vol = onp.zeros(len(cells))
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                f = 1.0 + xi * onp.array([a, b, c])
                dN = onp.stack([xi[:, 0] * f[:, 1] * f[:, 2], xi[:, 1] * f[:, 0] * f[:, 2],
                                xi[:, 2] * f[:, 0] * f[:, 1]], axis=1) / 8.0
                vol += onp.linalg.det(onp.einsum("nia,ib->nab", P, dN))
    return vol


def smooth_mesh(x, cells, x_ref, mode="interior", crease_deg=35.0, iters=400, lr=0.03,
                lr_end=0.003, power=4.0, eps=1e-3, verbose=True, slide_near=None,
                slide_radius=0.0, restarts=1, clamp_fn=None):
    """Relocate the nodes of the deformed mesh x (n, 3) -> (x_new, report).

    mode = "interior": boundary nodes fixed (the surface is reproduced exactly)
           "surface" : boundary nodes slide on the old boundary as well
    x_ref: node coordinates of the original mesh (target cell shapes).
    slide_near, slide_radius: in surface mode, let only the boundary nodes
           within slide_radius (mm) of the points slide_near (k, 3) slide; the
           rest of the boundary stays fixed.  Sliding along a faceted convex
           surface always moves nodes inside it, so letting the whole bar
           surface slide cost 0.17 % of the volume per call; the repair is
           only needed around the cells that are failing.
    clamp_fn: optional y -> y applied to the node positions after every
           projection step, for an obstacle the surface must stay out of
           (the die, inside a stroke).  Sliding along the old faceted surface
           carried nodes up to 4 mm into the die at a punch corner, where the
           chord between two surface nodes cuts through the die; the dies then
           had to back off by that much and re-press, which undid the stroke.
    restarts: Adam is restarted (moments cleared) this many times over the
           iterations.  A folded corner gives a few nodes a gradient 1e5-1e9
           times the rest; Adam's second moment then remembers it for ~1000
           iterations and those nodes stop moving once the fold is out.  With
           power 4 and one run the failing cells of hit 6 stayed at SJ 0.16;
           with power 2 they reached 0.4-0.7.
    """
    if mode not in ("interior", "surface"):
        raise ValueError(f"unknown smoothing mode {mode!r}")
    x = onp.asarray(x, dtype=float)
    cells = onp.asarray(cells)
    quads = boundary_quads(cells)
    cls, crease = classify_boundary(x, quads, crease_deg)
    old = OldBoundary(x, quads, crease)
    if mode == "interior":
        cls = onp.where(cls >= 0, 2, cls)
    elif slide_near is not None and slide_radius > 0:
        near = onp.zeros(len(x), dtype=bool)
        bnd = onp.where(cls >= 0)[0]
        d, _ = cKDTree(onp.asarray(slide_near, dtype=float).reshape(-1, 3)).query(x[bnd])
        near[bnd[d <= slide_radius]] = True
        cls = onp.where((cls >= 0) & ~near, 2, cls)
    free = cls != 2
    smooth = onp.where(cls == 0)[0]
    slide = onp.where(cls == 1)[0]

    val_grad = make_energy(cells, onp.asarray(x_ref, dtype=float), power, eps)
    q0 = scaled_jacobian(x, cells)
    v0 = float(gauss_volume(x, cells).sum())

    y = x.copy()
    m = onp.zeros_like(y)
    v = onp.zeros_like(y)
    b1, b2 = 0.9, 0.999
    best_E, best_y, E0 = onp.inf, y.copy(), None
    restarts = max(1, int(restarts))
    span = int(onp.ceil(iters / restarts))
    for it in range(1, iters + 1):
        E, g = val_grad(jnp.asarray(y))
        E = float(E)
        g = onp.array(g)
        if E0 is None:
            E0 = E
        if onp.isfinite(E) and E < best_E:
            best_E, best_y = E, y.copy()
        if not onp.all(onp.isfinite(g)):
            break
        g[~free] = 0.0
        j = (it - 1) % span + 1              # iteration within the current Adam run
        if j == 1:
            m[:] = 0.0
            v[:] = 0.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = lr_end + 0.5 * (lr - lr_end) * (1.0 + onp.cos(onp.pi * (j - 1) / span))
        y = y - step * (m / (1 - b1 ** j)) / (onp.sqrt(v / (1 - b2 ** j)) + 1e-12)
        y[~free] = x[~free]
        if len(smooth):
            y[smooth] = old.on_surface(y[smooth])[0]
        if len(slide):
            y[slide] = old.on_crease(y[slide])
        if clamp_fn is not None:
            y = onp.asarray(clamp_fn(y), dtype=float)
            y[~free] = x[~free]
    E, _ = val_grad(jnp.asarray(y))
    if onp.isfinite(float(E)) and float(E) < best_E:
        best_E, best_y = float(E), y.copy()

    q1 = scaled_jacobian(best_y, cells)
    accepted = bool(q1.min() >= q0.min() - 1e-9)
    y = best_y if accepted else x.copy()
    q1 = scaled_jacobian(y, cells)
    # how far the new boundary facets are from the old boundary
    Pq = y[quads]
    samples = onp.concatenate([Pq.mean(axis=1)] + [0.5 * (Pq[:, a] + Pq[:, b])
                                                   for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))])
    # measured against the same samples of the OLD facets, because a warped
    # quad is not its two triangles: that offset is not a change of shape
    Po = x[quads]
    samples_old = onp.concatenate([Po.mean(axis=1)] + [0.5 * (Po[:, a] + Po[:, b])
                                                       for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))])
    dev = onp.maximum(old.on_surface(samples)[1] - old.on_surface(samples_old)[1].max(), 0.0) \
        if mode == "surface" else onp.zeros(len(samples))
    v1 = float(gauss_volume(y, cells).sum())
    rep = dict(mode=mode, accepted=accepted, iters=int(iters),
               energy_before=float(E0), energy_after=float(best_E),
               min_sj_before=float(q0.min()), min_sj_after=float(q1.min()),
               n_below_0p2_before=int((q0 < 0.2).sum()), n_below_0p2_after=int((q1 < 0.2).sum()),
               n_below_0p4_before=int((q0 < 0.4).sum()), n_below_0p4_after=int((q1 < 0.4).sum()),
               n_inverted_before=int((q0 <= 0).sum()), n_inverted_after=int((q1 <= 0).sum()),
               max_move_mm=float(onp.linalg.norm(y - x, axis=1).max()),
               max_boundary_move_mm=float(onp.linalg.norm((y - x)[onp.unique(quads)], axis=1).max()),
               surface_dev_max_mm=float(dev.max()), surface_dev_rms_mm=float(onp.sqrt((dev ** 2).mean())),
               volume_before=v0, volume_after=v1, volume_change_rel=(v1 - v0) / v0,
               n_smooth=int(len(smooth)), n_crease=int(len(slide)),
               n_fixed_boundary=int(((cls == 2)).sum()), n_crease_edges=int(len(crease)))
    if verbose:
        print(f"  [smooth] {mode}: min SJ {rep['min_sj_before']:.3f} -> {rep['min_sj_after']:.3f}, "
              f"cells < 0.4: {rep['n_below_0p4_before']} -> {rep['n_below_0p4_after']}, energy "
              f"{rep['energy_before']:.4f} -> {rep['energy_after']:.4f}, max move "
              f"{rep['max_move_mm']:.2f} mm (boundary {rep['max_boundary_move_mm']:.2f}), surface "
              f"deviation max {rep['surface_dev_max_mm']:.3f} / rms {rep['surface_dev_rms_mm']:.4f} mm, "
              f"volume {100 * rep['volume_change_rel']:+.4f} %"
              f"{'' if accepted else '  -- REJECTED, mesh left as it was'}", flush=True)
    return y, rep
