"""Boundary-condition helpers for cylinder side-compression forging.

Contains:
  - build_cylinder_press_bcs:  per-hit Dirichlet + Neumann BC setup
      (Version 6 multi-hit displacement continuity fix included)
  - refresh_problem_surface_integrals:  recompute surface data after
      location_fns change
"""

import numpy as onp
import jax
import jax.numpy as np
from functools import reduce

from scipy.spatial.transform import Rotation


# =============================================================================
# Cylinder side-compression BCs for thermo-mechanical problem
# =============================================================================
def build_cylinder_press_bcs(mesh, R, H, hit, current_sol_u=None, T_linear_fn=None):
    """
    Build Dirichlet and thermal BCs for a single compression hit.

    Returns:
      dirichlet_bc_info_u (initial placeholder)
      dirichlet_bc_info_T
      location_fns_convect (with rotation consideration)
      bc_update_fn(problem, sol_u, step_count, scale)
      surface_inds (node indices on the outer cylindrical surface — for cooling)

    Parameters:
    -----------
    hit : Hit
        Hit object containing x_min_band, x_max_band, compression_displacement,
        rotation_euler_x
    current_sol_u : array or None
        Current displacement field (num_nodes, 3). If provided, contact nodes are
        selected based on deformed coordinates (mesh.points + current_sol_u).
        If None, uses initial mesh coordinates.
    T_linear_fn : callable or None
        Function to compute initial temperature profile (point) -> temp_value.
        If None, constant T0 is used.
    """

    pts = onp.array(mesh.points)

    # Use deformed coordinates if current_sol_u is provided
    if current_sol_u is not None:
        current_sol_u_np = onp.array(current_sol_u)
        pts_for_selection = pts + current_sol_u_np
        print(f"  [BC] Using deformed coordinates (displacement applied)")
    else:
        pts_for_selection = pts
        print(f"  [BC] Using initial coordinates")

    # ===== Build residual displacement lookup for multi-hit continuity =====
    # NOTE: The lookup must use pure JAX operations (no scipy/numpy calls)
    # because it is invoked inside jax.vmap where `p` is a JAX tracer.
    if current_sol_u is not None:
        _pts_jax = np.array(pts)
        _residual_u_jax = np.array(current_sol_u)

        def residual_u(p):
            """Look up Hit (N-1) residual displacement for reference point p.
            Pure-JAX implementation: finds nearest reference node via argmin."""
            dists_sq = np.sum((_pts_jax - p[None, :]) ** 2, axis=1)
            idx = np.argmin(dists_sq)
            return _residual_u_jax[idx]

        print(f"  [BC] Residual displacement lookup built "
              f"(max |u_res| = {onp.max(onp.abs(onp.array(current_sol_u))):.6e})")
    else:
        def residual_u(p):
            return np.zeros(3)

    # ===== Handle rotation =====
    rot_obj = Rotation.from_euler("x", hit.rotation_euler_x, degrees=True)
    rotation_quaternion = rot_obj.as_quat()  # [x, y, z, w]
    rot_matrix = rot_obj.as_matrix()

    def transform_point(p):
        return rot_matrix @ p

    def inverse_transform_point(p):
        return rot_matrix.T @ p

    y_min_initial = float(pts_for_selection[:, 1].min())
    y_max_initial = float(pts_for_selection[:, 1].max())

    print("y_min (in coordinates used for selection):{0}, y_max:{1}".format(y_min_initial, y_max_initial))
    print("Applied rotation quaternion: {0}".format(rotation_quaternion))
    print("Hit: x_band=[{0}, {1}], disp={2}, rot_x={3}".format(
        hit.x_min_band, hit.x_max_band, hit.compression_displacement, hit.rotation_euler_x))

    # band in x where platens act
    x_min_band = hit.x_min_band * H
    x_max_band = hit.x_max_band * H
    compression_displacement = hit.compression_displacement
    print("x_min_band:{0}, x_max_band:{1}".format(x_min_band, x_max_band))

    # tolerances
    tol_r = 0.25 * (2 * R / 5)
    tol_y = 0.10

    R0 = float(R)

    def in_x_band(p):
        return np.logical_and(p[0] >= x_min_band, p[0] <= x_max_band)

    def on_outer_surface(p):
        r = np.sqrt(p[1] ** 2 + p[2] ** 2)
        return np.abs(r - R0) <= tol_r

    # compute list of node indices on the outer surface (for cooling)
    radii = onp.sqrt(pts_for_selection[:, 1] ** 2 + pts_for_selection[:, 2] ** 2)
    mask = onp.abs(radii - R0) <= tol_r
    surface_inds = onp.where(mask)[0]
    surface_inds = np.array(surface_inds, dtype=int)
    print(f"[BC] outer surface node count: {len(surface_inds)}")

    # ===== Compute target platen lines in ROTATED coordinates =====
    mask_pts = onp.abs(onp.sqrt(pts_for_selection[:, 1] ** 2 + pts_for_selection[:, 2] ** 2) - R0) <= tol_r
    sample_points = pts_for_selection[mask_pts]
    if len(sample_points) > 0:
        sample_rotated = onp.array([transform_point(p) for p in sample_points])
        y_rot_min = float(sample_rotated[:, 1].min())
        y_rot_max = float(sample_rotated[:, 1].max())
        print("Rotated Y range: [{0}, {1}]".format(y_rot_min, y_rot_max))
    else:
        y_rot_min = y_min_initial
        y_rot_max = y_max_initial

    y_target_left_rot = y_rot_min + compression_displacement
    y_target_right_rot = y_rot_max - compression_displacement

    # ---- node-selection predicates ----
    def left_reach_nodes(p):
        p_rot = transform_point(p)
        return reduce(np.logical_and, (
            on_outer_surface(p),
            in_x_band(p),
            p_rot[1] <= y_target_left_rot + tol_y
        ))

    def right_reach_nodes(p):
        p_rot = transform_point(p)
        return reduce(np.logical_and, (
            on_outer_surface(p),
            in_x_band(p),
            p_rot[1] >= y_target_right_rot - tol_y
        ))

    # minimal rigid constraints
    def bottom_face(p):
        print("bottom surface:", float(pts[:, 0].min()))
        return np.isclose(p[0], float(pts[:, 0].min()), atol=1e-5)

    pin_id = onp.argmin(
        (pts[:, 0] - float(pts[:, 0].min())) ** 2 +
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

    # ===== Displacement computation (in rotated frame) =====
    def left_disp_vec(p):
        p_rot = transform_point(p)
        disp_rot_y = np.maximum(0.0, y_target_left_rot - p_rot[1])
        disp_vec_rot = np.array([0.0, disp_rot_y, 0.0])
        return inverse_transform_point(disp_vec_rot)

    def right_disp_vec(p):
        p_rot = transform_point(p)
        disp_rot_y = np.minimum(0.0, y_target_right_rot - p_rot[1])
        disp_vec_rot = np.array([0.0, disp_rot_y, 0.0])
        return inverse_transform_point(disp_vec_rot)

    # Initial placeholder BC info
    location_fns_u = [bottom_face, pin_node, pin_node,
                      left_reach_nodes, left_reach_nodes, left_reach_nodes,
                      right_reach_nodes, right_reach_nodes, right_reach_nodes]
    vecs = [0, 1, 2,
            0, 1, 2,
            0, 1, 2]
    value_fns = [zero_val] * len(location_fns_u)
    dirichlet_bc_info_u = [location_fns_u, vecs, value_fns]

    # Debug counts
    left_n = int(onp.sum(onp.array([bool(left_reach_nodes(p)) for p in pts_for_selection])))
    right_n = int(onp.sum(onp.array([bool(right_reach_nodes(p)) for p in pts_for_selection])))
    print(f"[BC] left_reach_nodes={left_n}, right_reach_nodes={right_n}, pin=1")
    if left_n == 0 or right_n == 0:
        print("[BC WARNING] selected 0 nodes on one side. Increase tol_r / tol_x or refine mesh.")

    def bc_update_fn(problem, sol_u, step_count, scale):
        """
        Update Dirichlet BCs for mechanical problem.

        Multi-hit continuity: for hit_idx > 0 the prescribed displacement ramps
        from the previous hit's residual displacement to the current hit's target:
            u = u_residual + scale * (u_target - u_residual)
        When current_sol_u is None (first hit), residual_u returns zeros and this
        reduces to the original  u = scale * u_target.
        """
        new_value_fns = [
            zero_val,
            zero_val,
            zero_val,
            (lambda p, idx=0: residual_u(p)[idx] + scale * (left_disp_vec(p)[idx] - residual_u(p)[idx])),
            (lambda p, idx=1: residual_u(p)[idx] + scale * (left_disp_vec(p)[idx] - residual_u(p)[idx])),
            (lambda p, idx=2: residual_u(p)[idx] + scale * (left_disp_vec(p)[idx] - residual_u(p)[idx])),
            (lambda p, idx=0: residual_u(p)[idx] + scale * (right_disp_vec(p)[idx] - residual_u(p)[idx])),
            (lambda p, idx=1: residual_u(p)[idx] + scale * (right_disp_vec(p)[idx] - residual_u(p)[idx])),
            (lambda p, idx=2: residual_u(p)[idx] + scale * (right_disp_vec(p)[idx] - residual_u(p)[idx])),
        ]
        problem.fes[0].update_Dirichlet_boundary_conditions([location_fns_u, vecs, new_value_fns])

    # Thermal BCs
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
