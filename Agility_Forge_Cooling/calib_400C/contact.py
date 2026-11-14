"""Two flat rigid platens closing on the bar, re-detected every load step.

The machine reports a *closure radius* rho, not a stroke, so the bite for a
hit is whatever it takes to bring the current surface down to rho:

    d = half the rotated-frame span of the contact band  -  rho

Contact is Updated-Lagrangian -- the active set and the prescribed
displacements are rebuilt from the last converged shape at every step, so the
band that was pressed in an earlier hit is seen as it now is, not as it was.

Frames: the bar axis is x; a hit at angle phi is handled by rotating the
geometry by -phi about x, which puts the press direction on +y.
"""
import numpy as onp
import jax.numpy as np


def outer_surface_nodes(points, cells):
    """Nodes on the boundary, from mesh topology (shape-independent)."""
    faces = onp.concatenate([cells[:, [0, 2, 1]], cells[:, [0, 1, 3]],
                             cells[:, [1, 2, 3]], cells[:, [0, 3, 2]]], axis=0)
    key = onp.sort(faces, axis=1)
    _, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    boundary = faces[idx[cnt == 1]]
    nodes = onp.unique(boundary)
    # drop the two end caps: they are not part of the forged surface
    x = points[nodes, 0]
    keep = (x > x.min() + 1e-6) & (x < x.max() - 1e-6)
    return nodes[keep]


def rot_x(phi_deg):
    """Global -> press frame, matching build_cylinder_press_bcs exactly.

    That BC builds its rotation with scipy's Rotation.from_euler("x", phi) and
    maps global to press frame as R @ p, so the platen normal in global
    coordinates is R.T @ e_y = (0, cos phi, -sin phi).  This function used to
    return the TRANSPOSE of that matrix, i.e. rot_x(-phi).  Everything built on
    it -- the bite, and above all the projection that turns the nodal reaction
    into a press load -- was therefore resolved along (0, cos phi, +sin phi),
    which is 89.9 degrees away from the true platen normal at phi = 45.

    The damage was invisible at phi = 0, 90, 180, 270, where the two directions
    are parallel up to sign, and total at odd multiples of 45, where they are
    orthogonal and the projection cancels: at a fixed 1.97 mm bite on a round
    bar the reported load ran 139, 35, 135, 33 kN over phi = 0, 45, 90, 135
    while the displacement field was the same in all four.  The reaction
    magnitude was correct throughout (138 kN); only its projection was wrong.
    """
    c, s = onp.cos(onp.radians(phi_deg)), onp.sin(onp.radians(phi_deg))
    return onp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def build_press_bcs(points, surf_nodes, hit, sol_u=None, tol_contact=0.05,
                    min_nodes=8):
    """Return (dirichlet_bc_info, bc_update_fn, info).

    hit: dict(phi_deg, z_mm, rho_mm, band_mm)
    sol_u: displacement at the start of this hit, or None for the first.
    """
    pts = onp.asarray(points)
    u0 = onp.zeros_like(pts) if sol_u is None else onp.asarray(sol_u)
    R = rot_x(hit["phi_deg"])

    xdef0 = pts + u0
    lo = hit["z_mm"] - 0.5 * hit["band_mm"]
    hi = hit["z_mm"] + 0.5 * hit["band_mm"]
    band = (xdef0[surf_nodes, 0] >= lo) & (xdef0[surf_nodes, 0] <= hi)
    cand = surf_nodes[band]
    if len(cand) < min_nodes:
        raise RuntimeError(f"only {len(cand)} candidate nodes in band [{lo:.1f},{hi:.1f}]")

    rot0 = (R @ xdef0[cand].T).T
    y_min0, y_max0 = float(rot0[:, 1].min()), float(rot0[:, 1].max())
    depth = 0.5 * (y_max0 - y_min0) - hit["rho_mm"]

    # Minimal rigid-body restraint on the far end face, away from every hit.
    # ux on the whole face kills x-translation and rotation about y and z; one
    # node pinned in y and z kills the remaining translations; rotation about x
    # needs a second node with a real lever arm -- taking the node nearest the
    # axis leaves it almost unconstrained (1.5 mm arm on a 19 mm bar), so the
    # outermost node is used instead.
    x_far = pts[:, 0].min()
    face = onp.where(onp.abs(pts[:, 0] - x_far) < 1e-6)[0]
    radius = pts[face, 1] ** 2 + pts[face, 2] ** 2
    ordered = face[onp.argsort(radius)]
    n_axis = ordered[0]            # closest to the axis -> pin y and z
    n_arm = ordered[-1]            # farthest -> constrain rotation about x
    rb_locs = [face, onp.array([n_axis]), onp.array([n_axis]), onp.array([n_arm])]
    rb_vecs = [0, 1, 2, 1]
    rb_vals = [onp.zeros(len(face))] + [onp.zeros(1)] * 3

    def compute_contact(u_now, scale):
        xdef = pts + onp.asarray(u_now)
        rot = (R @ xdef[cand].T).T
        y_L = y_min0 + scale * depth
        y_R = y_max0 - scale * depth
        left = onp.where(rot[:, 1] <= y_L + tol_contact)[0]
        right = onp.where(rot[:, 1] >= y_R - tol_contact)[0]

        def target(loc, y_plane):
            if len(loc) == 0:
                return onp.zeros((0, 3))
            tgt = rot[loc].copy()
            tgt[:, 1] = y_plane                       # normal onto the platen
            return (R.T @ (tgt - (R @ pts[cand[loc]].T).T).T).T

        return cand[left], cand[right], target(left, y_L), target(right, y_R)

    l0, r0, uL0, uR0 = compute_contact(u0, 0.0)
    locs = rb_locs + [l0, l0, l0, r0, r0, r0]
    vecs = rb_vecs + [0, 1, 2, 0, 1, 2]
    vals = rb_vals + [uL0[:, 0], uL0[:, 1], uL0[:, 2],
                      uR0[:, 0], uR0[:, 1], uR0[:, 2]]
    bc_info = [locs, vecs, vals]

    state = {"left": l0, "right": r0}

    def bc_update_fn(problem, u_now, scale):
        left, right, uL, uR = compute_contact(u_now, scale)
        state["left"], state["right"] = left, right
        problem.fes[0].update_Dirichlet_boundary_conditions(
            [rb_locs + [left, left, left, right, right, right],
             rb_vecs + [0, 1, 2, 0, 1, 2],
             rb_vals + [uL[:, 0], uL[:, 1], uL[:, 2], uR[:, 0], uR[:, 1], uR[:, 2]]])
        return len(left), len(right)

    info = dict(candidates=cand, depth_mm=float(depth),
                y_min0=y_min0, y_max0=y_max0, R=R, state=state)
    return bc_info, bc_update_fn, info


def press_force_kn(residual_vec, node_inds, R, n_nodes):
    """Reaction on one platen, resolved along the press normal, in kN.

    The residual at a constrained dof is the force the constraint carries, so
    summing it over the upper platen's nodes and projecting on the rotated y
    gives what a load cell under that platen would read.
    """
    res = onp.asarray(residual_vec).reshape(n_nodes, 3)
    if len(node_inds) == 0:
        return 0.0
    f_rot = (R @ res[node_inds].T).T
    return float(onp.sum(f_rot[:, 1]) / 1e3)          # N -> kN
