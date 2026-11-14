"""Element distortion monitor: corner scaled Jacobian of the deformed mesh.

Why this exists
---------------
A Lagrangian mesh pressed over twelve hits can distort until an element
inverts.  Newton then fails with NaN and the run dies with no indication of
where or why.  This measures, after every converged step, how close each
element is to inverting, so the run can stop with a report first.

Measure
-------
At every corner of a cell take the three edge vectors to the neighbouring
corners, normalise them to unit length and take the determinant.  This is the
standard "scaled Jacobian": 1 for a right-angled corner, 0 when the three
edges become coplanar (the element is about to invert), negative when it has
inverted.  The cell value is the minimum over its corners.  Corners are used
rather than quadrature points because a trilinear hexahedron can fold at a
corner while every Gauss point still reports a positive Jacobian.

Cells must be given in meshio/VTK node order (run_cold.py's ``cells_out``).
"""
import numpy as onp

# VTK hexahedron: corner -> its three neighbours, ordered so that the
# determinant is positive for a valid (positively oriented) cell
_HEX_CORNERS = ((0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
                (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3))
# tetrahedron (corner nodes 0-3 of TET4 / TET10)
_TET_CORNERS = ((0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3), (3, 0, 2, 1))


def corner_table(cells):
    n = cells.shape[1]
    if n == 8:
        return _HEX_CORNERS
    if n in (4, 10):
        return _TET_CORNERS
    raise ValueError(f"no corner table for cells with {n} nodes")


def scaled_jacobian(x, cells):
    """-> (n_cells,) minimum corner scaled Jacobian of each cell at positions x."""
    x = onp.asarray(x, dtype=float)
    C = onp.asarray(cells)
    out = onp.full(len(C), onp.inf)
    for c, a, b, d in corner_table(C):
        p = x[C[:, c]]
        e = [x[C[:, k]] - p for k in (a, b, d)]
        e = [v / onp.maximum(onp.linalg.norm(v, axis=1, keepdims=True), 1e-300) for v in e]
        det = onp.einsum("ij,ij->i", onp.cross(e[0], e[1]), e[2])
        out = onp.minimum(out, det)
    return out


_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def min_edge_length(x, cells):
    """Shortest element edge of the mesh at positions x (corner nodes only)."""
    x = onp.asarray(x, dtype=float)
    C = onp.asarray(cells)
    edges = _HEX_EDGES if C.shape[1] == 8 else _TET_EDGES
    return float(min(onp.linalg.norm(x[C[:, a]] - x[C[:, b]], axis=1).min()
                     for a, b in edges))


class DistortionStop(Exception):
    """Raised when the deformed mesh crosses the distortion threshold.

    Deliberately NOT a RuntimeError / FloatingPointError / AssertionError:
    the time stepper catches those and retries with a smaller dt, which cannot
    undo distortion that is already in the converged state.
    """

    def __init__(self, report):
        super().__init__(report["message"])
        self.report = report


class QualityMonitor:
    """Tracks the minimum scaled Jacobian of the deformed mesh.

    ``check(sol_u, where)`` returns the per-cell values and raises
    DistortionStop when the minimum drops below ``stop_below``.
    """

    def __init__(self, pts, cells, stop_below=0.05, warn_below=0.2):
        self.pts = onp.asarray(pts)
        self.cells = onp.asarray(cells)
        self.stop_below = float(stop_below)
        self.warn_below = float(warn_below)
        self.q_ref = scaled_jacobian(self.pts, self.cells)
        self.history = []          # (where, min q, n cells below warn)
        self._warned = False
        print(f"  [quality] reference mesh: min scaled Jacobian {self.q_ref.min():.3f} "
              f"(median {onp.median(self.q_ref):.3f}); stop below {self.stop_below}, "
              f"warn below {self.warn_below}", flush=True)

    def report(self, q, x, where, i=None):
        """Distortion report for the per-cell values q at positions x."""
        if i is None:
            i = int(onp.argmin(q))
        qmin = float(q[i])
        ctr0 = self.pts[self.cells[i]].mean(axis=0)
        ctr = x[self.cells[i]].mean(axis=0)
        worst = onp.argsort(q)[:10]
        return dict(
            message=(f"mesh distortion at {where}: min scaled Jacobian {qmin:.4f} "
                     f"< {self.stop_below} in cell {i}"),
            where=where, min_scaled_jacobian=qmin, cell=i,
            cell_ref_centre_mm=[float(v) for v in ctr0],
            cell_def_centre_mm=[float(v) for v in ctr],
            cell_ref_radius_mm=float(onp.hypot(ctr0[1], ctr0[2])),
            cell_ref_angle_deg=float(onp.degrees(onp.arctan2(ctr0[2], ctr0[1]))),
            n_cells_below_stop=int((q < self.stop_below).sum()),
            n_cells_below_warn=int((q < self.warn_below).sum()),
            n_cells_inverted=int((q <= 0.0).sum()),
            worst_cells=[[int(c), float(q[c])] for c in worst])

    def evaluate(self, sol_u, where, record=True):
        """-> (q, report or None): the per-cell values at sol_u and, when the
        minimum is below stop_below, the report check() would raise with.
        Does not raise; the caller decides what a failed state means (a
        trial state is discarded, a committed one ends the run)."""
        x = self.pts + onp.asarray(sol_u)
        q = scaled_jacobian(x, self.cells)
        i = int(onp.argmin(q))
        qmin = float(q[i])
        n_warn = int((q < self.warn_below).sum())
        if record:
            self.history.append((where, qmin, n_warn))
        if qmin < self.warn_below and not self._warned:
            print(f"  [quality] WARNING {where}: min scaled Jacobian {qmin:.3f} "
                  f"({n_warn} cells below {self.warn_below})", flush=True)
            self._warned = True
        if qmin < self.stop_below:
            return q, self.report(q, x, where, i)
        return q, None

    def check(self, sol_u, where):
        q, rep = self.evaluate(sol_u, where)
        if rep is not None:
            raise DistortionStop(rep)
        return q
