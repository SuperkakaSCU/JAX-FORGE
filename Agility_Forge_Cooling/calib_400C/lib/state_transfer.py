"""Transfer of Gauss-point state between two meshes of the same connectivity.

Why this exists
---------------
A relocation inside a stroke (run_cold --stroke-relocate) makes the current
shape the reference of a fresh problem.  The material state must follow the
material, not the mesh: quadrature points move with the nodes, so every state
field is re-sampled at the new Gauss points from the old ones, measured in
the deformed configuration both meshes describe.  Every new Gauss point takes
the inverse-distance weighted mean of its k nearest OLD Gauss points; where
the mesh did not move the nearest old point coincides with the new one and
the value is exact.  Piecewise, bounded, no overshoot; the smearing is of the
order of one element in the region that was relocated, nowhere else.

What is transferred (see run_cold's relocation block)
  alpha        hardening, scalar
  be_bar       elastic left Cauchy-Green tensor (unimodular part) -- a SPATIAL
               tensor, so changing the reference does not change it; the
               component-wise IDW mean of symmetric positive definite tensors
               is symmetric positive definite
  J_ref        total volume ratio of the new reference to the original body,
               scalar (uniform per cell under F-bar)
The stress is then recovered exactly by the constitutive map at the first
evaluation on the new reference (F = I, be_old = be_bar, pressure from J_ref).
"""
import numpy as onp
from scipy.spatial import cKDTree


def gauss_points(x, cells, shape_vals):
    """Physical positions of the quadrature points, (n_cells, n_quad, 3).

    x: node coordinates of the configuration wanted (reference + displacement),
    cells: connectivity in the order shape_vals refers to (the package's),
    shape_vals: (n_quad, n_nodes) shape functions at the quadrature points."""
    return onp.einsum("qn,cnd->cqd", onp.asarray(shape_vals), onp.asarray(x)[onp.asarray(cells)])


def transfer_field(xg_old, v_old, xg_new, k=4, exact_tol=1e-9):
    """IDW re-sampling of a Gauss-point field of any trailing shape.

    xg_old, xg_new: (..., 3) Gauss-point positions in the same configuration;
    v_old: (*xg_old.shape[:-1], *tail) -> v_new: (*xg_new.shape[:-1], *tail)."""
    P_old = onp.asarray(xg_old, dtype=float).reshape(-1, 3)
    P_new = onp.asarray(xg_new, dtype=float).reshape(-1, 3)
    lead = onp.asarray(xg_old).shape[:-1]
    v = onp.asarray(v_old, dtype=float)
    tail = v.shape[len(lead):]
    v = v.reshape(len(P_old), -1)
    k = max(1, min(int(k), len(P_old)))
    d, idx = cKDTree(P_old).query(P_new, k=k)
    d = d.reshape(len(P_new), k)
    idx = idx.reshape(len(P_new), k)
    w = 1.0 / onp.maximum(d, exact_tol) ** 2
    hit = d[:, 0] <= exact_tol                 # coincident point: take it alone
    w[hit] = 0.0
    w[hit, 0] = 1.0
    w = w / w.sum(axis=1, keepdims=True)
    out = onp.einsum("nk,nkm->nm", w, v[idx])
    return out.reshape(onp.asarray(xg_new).shape[:-1] + tail)


def transfer_scalar(xg_old, v_old, xg_new, k=4, exact_tol=1e-9):
    """-> v_new with the shape of xg_new[..., 0]."""
    return transfer_field(xg_old, v_old, xg_new, k=k, exact_tol=exact_tol)


def committed_state(int_vars):
    """(be_bar, J_committed) of a committed state, from the internal variables.

    be_old is stored as be_bar * det(F_old)^(2/3) (lib/constitutive.py
    return_map); J_committed = J_ref * det(F_old) is the volume ratio of the
    committed shape to the original body.  Both are what the rebased problem
    needs: Be_old = be_bar (F_old = I) and J_ref = J_committed."""
    F_old = onp.asarray(int_vars[0], dtype=float)
    be_old = onp.asarray(int_vars[1], dtype=float)
    J_ref = onp.asarray(int_vars[6], dtype=float)
    detF = onp.linalg.det(F_old)
    be_bar = be_old * detF[..., None, None] ** (-2.0 / 3.0)
    return be_bar, J_ref * detF
