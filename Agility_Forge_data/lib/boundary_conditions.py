"""Boundary-condition helpers for cylinder side-compression forging.

Route B — Updated-Lagrangian dynamic contact.
The platen is a rigid plane advancing inward by ``compression_displacement``
over each hit.  The contact node set and the prescribed displacements are
re-detected at EVERY load step from the current deformed configuration
(``pts + sol_u``), so contact follows the actual bulged/flattened surface and
material that is pushed away is released automatically.

Contains:
  - build_cylinder_press_bcs:  per-hit dynamic-contact Dirichlet + thermal BC
  - refresh_problem_surface_integrals:  recompute surface data after
      location_fns change
"""

import numpy as onp
import jax
import jax.numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# Cylinder side-compression BCs for thermo-mechanical problem
# =============================================================================
def build_cylinder_press_bcs(mesh, R, H, hit, current_sol_u=None, T_linear_fn=None):
    """
    Build dynamic-contact Dirichlet and thermal BCs for a single compression hit.

    Returns:
      dirichlet_bc_info_u (initial placeholder, rebuilt every step)
      dirichlet_bc_info_T
      location_fns_convect (thermal convection surface, static for this hit)
      bc_update_fn(problem, sol_u, step_count, scale)
      surface_inds (node indices on the outer cylindrical surface — for cooling)

    Parameters:
    -----------
    hit : Hit
        Hit object containing x_min_band, x_max_band, compression_displacement,
        rotation_euler_x
    current_sol_u : array or None
        Displacement field accumulated at the START of this hit (num_nodes, 3).
        Used to define the initial deformed surface from which the rigid platen
        begins to advance. If None (first hit), the reference mesh is used.
    T_linear_fn : callable or None
        Function to compute initial temperature profile (point) -> temp_value.

    Route B (Updated-Lagrangian dynamic contact)
    --------------------------------------------
    The platen is a rigid plane that advances inward by
    ``compression_displacement`` over the hit.  At EVERY load step the contact
    set and prescribed displacements are RE-DETECTED from the current deformed
    configuration (``pts + sol_u``):

      * A candidate node (outer-surface node inside the axial x-band) is in
        contact when its current rotated normal coordinate has reached /
        penetrated the moving platen plane.
      * Each contacting node is projected back onto the platen plane while
        keeping its current tangential position (frictional-stick increment).
      * Non-contacting nodes are left free, so material pushed away from the
        platen is released automatically.

    The mechanical Dirichlet BC is therefore rebuilt every step (its node set
    may change size); the thermal-convection surface is kept STATIC for the hit
    to avoid re-jitting surface integrals during the Newton solves.
    """

    pts = onp.array(mesh.points)

    if current_sol_u is not None:
        current_sol_u_np = onp.array(current_sol_u)
        print("  [BC] Route B multi-hit: dynamic contact on deformed config (pts + sol_u)")
    else:
        current_sol_u_np = onp.zeros_like(pts)
        print("  [BC] Route B first hit: dynamic contact on reference config")

    # ===== Handle rotation =====
    rot_obj = Rotation.from_euler("x", hit.rotation_euler_x, degrees=True)
    rotation_quaternion = rot_obj.as_quat(canonical=False)  # [x, y, z, w]
    rot_matrix = onp.array(rot_obj.as_matrix())

    def transform_points(points):
        # global -> rotated frame
        return points @ rot_matrix.T

    def to_global_vectors(vectors):
        # rotated frame -> global
        return vectors @ rot_matrix

    # band in x where platens act (reference configuration)
    x_min_band = hit.x_min_band * H
    x_max_band = hit.x_max_band * H
    compression_displacement = float(hit.compression_displacement)

    print("Applied rotation quaternion: {0}".format(rotation_quaternion))
    print("Hit: x_band=[{0}, {1}], disp={2}, rot_x={3}".format(
        hit.x_min_band, hit.x_max_band, hit.compression_displacement, hit.rotation_euler_x))
    print("x_min_band:{0}, x_max_band:{1}".format(x_min_band, x_max_band))

    # tolerances
    tol_r = 0.25 * (2 * R / 5)
    tol_contact = 0.10  # mm: thickness of the contact-detection layer near platen
    R0 = float(R)

    def in_x_band_np(points):
        return onp.logical_and(points[:, 0] >= x_min_band, points[:, 0] <= x_max_band)

    def on_outer_surface_np(points):
        radii = onp.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
        return onp.abs(radii - R0) <= tol_r

    # ----- outer surface node set (reference) for cooling -----
    surface_mask = on_outer_surface_np(pts)
    surface_inds = np.array(onp.where(surface_mask)[0], dtype=int)
    print(f"[BC] outer surface node count: {len(surface_inds)}")

    # ----- candidate contact nodes: outer-surface lateral nodes within x-band -----
    # Boundary topology does not change with deformation, so the candidate set is
    # fixed (reference); the dynamic contact set is a subset re-selected per step.
    candidate_mask = surface_mask & in_x_band_np(pts)
    candidate_inds = onp.where(candidate_mask)[0].astype(onp.int32)
    n_cand = len(candidate_inds)
    print(f"[BC] candidate contact node count (x-band outer surface): {n_cand}")
    if n_cand == 0:
        print("[BC WARNING] no candidate contact nodes. Check x_band / tol_r / mesh.")

    # reference rotated coordinates of candidates
    X_cand = pts[candidate_inds]
    X_cand_rot = transform_points(X_cand)                       # (n_cand, 3)

    # initial deformed (rotated) coordinates at the START of this hit
    xdef0_cand = X_cand + current_sol_u_np[candidate_inds]
    xdef0_cand_rot = transform_points(xdef0_cand)
    if n_cand > 0:
        y_rot_min0 = float(xdef0_cand_rot[:, 1].min())
        y_rot_max0 = float(xdef0_cand_rot[:, 1].max())
    else:
        y_rot_min0 = float(transform_points(pts)[:, 1].min())
        y_rot_max0 = float(transform_points(pts)[:, 1].max())
    print("Initial deformed rotated-Y range (contact band): [{0}, {1}]".format(y_rot_min0, y_rot_max0))
    print("Platen total travel per side: {0}".format(compression_displacement))

    # ===== minimal rigid constraints (reference configuration) =====
    x_bottom = float(pts[:, 0].min())
    y_min_initial = float(pts[:, 1].min())

    def bottom_face(p):
        return np.isclose(p[0], x_bottom, atol=1e-5)

    pin_id = onp.argmin(
        (pts[:, 0] - x_bottom) ** 2 +
        (pts[:, 1] - y_min_initial) ** 2 +
        (pts[:, 2] - 0.0) ** 2
    )
    pin_point = pts[pin_id]
    print("pin_id", pin_id)

    def pin_node(p):
        return (np.abs(p[0] - pin_point[0]) < 1e-12) & \
               (np.abs(p[1] - pin_point[1]) < 1e-12) & \
               (np.abs(p[2] - pin_point[2]) < 1e-12)

    def zero_val(p):
        return 0.0

    vecs = [0, 1, 2,
            0, 1, 2,
            0, 1, 2]

    # ===== core: dynamic contact detection on the current deformed config =====
    def compute_contact(sol_u_np, scale):
        """Detect contact nodes and prescribed (total) displacements at `scale`.

        Returns (left_inds, right_inds, u_left, u_right) where u_* are total
        displacement vectors (global frame) that project the contacting nodes
        onto the moving platen plane while preserving their current tangential
        position.
        """
        if n_cand == 0:
            empty_i = onp.zeros((0,), dtype=onp.int32)
            empty_u = onp.zeros((0, 3), dtype=pts.dtype)
            return empty_i, empty_i, empty_u, empty_u

        # current deformed rotated coordinates of candidate nodes: (X + u) rotated
        u_cand = sol_u_np[candidate_inds]
        xdef_cand_rot = X_cand_rot + transform_points(u_cand)
        yc = xdef_cand_rot[:, 1]

        # moving platen planes (rotated normal coordinate)
        y_L = y_rot_min0 + scale * compression_displacement      # lower platen moves up
        y_R = y_rot_max0 - scale * compression_displacement      # upper platen moves down

        left_local = yc <= (y_L + tol_contact)
        right_local = yc >= (y_R - tol_contact)

        # resolve overlap by midline so each node is assigned to one platen only
        overlap = left_local & right_local
        if onp.any(overlap):
            midline = 0.5 * (y_L + y_R)
            to_right = overlap & (yc >= midline)
            to_left = overlap & (yc < midline)
            left_local = left_local & ~to_right
            right_local = right_local & ~to_left

        left_loc = onp.where(left_local)[0]
        right_loc = onp.where(right_local)[0]
        left_inds = candidate_inds[left_loc]
        right_inds = candidate_inds[right_loc]

        # project onto platen plane: keep current tangential, set normal to plane
        def _targets(loc, y_plane):
            if len(loc) == 0:
                return onp.zeros((0, 3), dtype=pts.dtype)
            target_rot = xdef_cand_rot[loc].copy()
            target_rot[:, 1] = y_plane
            u_rot = target_rot - X_cand_rot[loc]     # total displacement (rotated)
            return to_global_vectors(u_rot)          # -> global frame

        u_left = _targets(left_loc, y_L)
        u_right = _targets(right_loc, y_R)
        return left_inds, right_inds, u_left, u_right

    # ----- initial placeholder BC (scale = 0 on current deformed config) -----
    left0, right0, uL0, uR0 = compute_contact(current_sol_u_np, 0.0)
    location_fns_u = [bottom_face, pin_node, pin_node,
                      left0, left0, left0,
                      right0, right0, right0]
    value_fns = [zero_val, zero_val, zero_val,
                 uL0[:, 0], uL0[:, 1], uL0[:, 2],
                 uR0[:, 0], uR0[:, 1], uR0[:, 2]]
    dirichlet_bc_info_u = [location_fns_u, vecs, value_fns]
    print(f"[BC] initial contact: left={len(left0)}, right={len(right0)}, pin=1")

    def bc_update_fn(problem, sol_u, step_count, scale):
        """Re-detect contact and rebuild the mechanical Dirichlet BC this step.

        Updated-Lagrangian: contact is evaluated from the last converged
        displacement (`sol_u`), so the active set is fixed during each Newton
        solve and updated between steps.
        """
        sol_u_np = onp.asarray(sol_u)
        left_inds, right_inds, u_left, u_right = compute_contact(sol_u_np, scale)

        loc_fns = [bottom_face, pin_node, pin_node,
                   left_inds, left_inds, left_inds,
                   right_inds, right_inds, right_inds]
        val_fns = [zero_val, zero_val, zero_val,
                   u_left[:, 0], u_left[:, 1], u_left[:, 2],
                   u_right[:, 0], u_right[:, 1], u_right[:, 2]]
        problem.fes[0].update_Dirichlet_boundary_conditions([loc_fns, vecs, val_fns])

        y_L = y_rot_min0 + scale * compression_displacement
        y_R = y_rot_max0 - scale * compression_displacement
        print(f"  [BC step {step_count}] scale={scale:.4f} contact left={len(left_inds)} "
              f"right={len(right_inds)} (y_L={y_L:.4f}, y_R={y_R:.4f})")

        # =====================================================================
        # [DIAG-C] Magnitude of the per-step prescribed contact displacement.
        # If hit-N step 1 prescribes near ~tol_contact (0.10 mm) for many
        # nodes, the platen-band ingest at this step is a dt-INDEPENDENT
        # geometric kick and is the proximate cause of non-convergence.
        # =====================================================================
        try:
            uL_max = float(onp.max(onp.abs(u_left))) if u_left.size > 0 else 0.0
            uR_max = float(onp.max(onp.abs(u_right))) if u_right.size > 0 else 0.0
            uL_norm = float(onp.linalg.norm(u_left)) if u_left.size > 0 else 0.0
            uR_norm = float(onp.linalg.norm(u_right)) if u_right.size > 0 else 0.0
            print(f"  [DIAG-C step {step_count}] |u_left|_inf={uL_max:.6e} mm, "
                  f"|u_right|_inf={uR_max:.6e} mm, "
                  f"||u_left||_2={uL_norm:.4e}, ||u_right||_2={uR_norm:.4e}, "
                  f"tol_contact={tol_contact}")
        except Exception as _diag_e:
            print(f"  [DIAG-C step {step_count}] contact-magnitude check failed: {_diag_e}")

    # ===== Thermal convection surface (STATIC for this hit) =====
    # Split the candidate band into lower/upper halves by the initial deformed
    # mid-plane; used as the convective heat-loss surface near the platens.
    if n_cand > 0:
        mid0 = 0.5 * (y_rot_min0 + y_rot_max0)
        yc0 = xdef0_cand_rot[:, 1]
        conv_left_inds = np.array(candidate_inds[yc0 <= mid0], dtype=int)
        conv_right_inds = np.array(candidate_inds[yc0 > mid0], dtype=int)
    else:
        conv_left_inds = np.array([], dtype=int)
        conv_right_inds = np.array([], dtype=int)

    def left_reach_nodes(p, ind):
        return np.isin(ind, conv_left_inds)

    def right_reach_nodes(p, ind):
        return np.isin(ind, conv_right_inds)

    dirichlet_bc_info_T = None
    location_fns_convect = [left_reach_nodes, right_reach_nodes]

    return dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds


# =============================================================================
# Refresh surface integrals after location_fns change
# =============================================================================
def refresh_problem_surface_integrals(problem):
    """Refresh surface integral data on an existing Problem instance after
    ``problem.location_fns`` has been changed.

    This recomputes ``boundary_inds_list``, ``selected_face_shape_grads``,
    ``nanson_scale``, ``selected_face_shape_vals``, ``physical_surface_quad_points``,
    ``I``, ``J`` indices, and then calls ``pre_jit_fns()`` to rebuild JITted kernels.
    """
    problem.boundary_inds_list = problem.fes[0].get_boundary_conditions_inds(problem.location_fns)

    # Rebuild I and J sparse matrix indices
    def find_ind(*x):
        inds = []
        for i in range(len(x)):
            crt_ind = problem.fes[i].vec * x[i][:, None] + np.arange(problem.fes[i].vec)[None, :] + problem.offset[i]
            inds.append(crt_ind.reshape(-1))
        return np.hstack(inds)

    cells_list = [fe.cells for fe in problem.fes]
    inds = onp.array(jax.vmap(find_ind)(*cells_list))
    I = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
    J = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)

    # Add surface integral indices for each boundary
    problem.cells_list_face_list = []
    for boundary_inds in problem.boundary_inds_list:
        cells_list_face = [cells[boundary_inds[:, 0]] for cells in cells_list]
        inds_face = onp.array(jax.vmap(find_ind)(*cells_list_face))
        I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
        J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
        I = onp.hstack((I, I_face))
        J = onp.hstack((J, J_face))
        problem.cells_list_face_list.append(cells_list_face)

    problem.I = I
    problem.J = J

    # Rebuild surface integral geometry
    selected_face_shape_grads = []
    nanson_scale = []
    selected_face_shape_vals = []
    physical_surface_quad_points = []

    for boundary_inds in problem.boundary_inds_list:
        s_shape_grads = []
        n_scale = []
        s_shape_vals = []
        for fe in problem.fes:
            face_shape_grads_physical, nanson_scale_i = fe.get_face_shape_grads(boundary_inds)
            selected_face_shape_vals_i = fe.face_shape_vals[boundary_inds[:, 1]]
            s_shape_grads.append(face_shape_grads_physical)
            n_scale.append(nanson_scale_i)
            s_shape_vals.append(selected_face_shape_vals_i)

        s_shape_grads = onp.concatenate(s_shape_grads, axis=2)
        n_scale = onp.transpose(onp.stack(n_scale), axes=(1, 0, 2))
        s_shape_vals = onp.concatenate(s_shape_vals, axis=2)
        physical_surface_quad_points_i = problem.fes[0].get_physical_surface_quad_points(boundary_inds)

        selected_face_shape_grads.append(s_shape_grads)
        nanson_scale.append(n_scale)
        selected_face_shape_vals.append(s_shape_vals)
        physical_surface_quad_points.append(physical_surface_quad_points_i)

    problem.selected_face_shape_grads = selected_face_shape_grads
    problem.nanson_scale = nanson_scale
    problem.selected_face_shape_vals = selected_face_shape_vals
    problem.physical_surface_quad_points = physical_surface_quad_points

    # Reset per-surface internal vars placeholder
    problem.internal_vars_surfaces = [() for _ in range(len(problem.boundary_inds_list))]

    # Re-jit kernels
    problem.pre_jit_fns()
