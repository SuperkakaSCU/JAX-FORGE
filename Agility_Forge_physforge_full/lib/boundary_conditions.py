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


def rigid_body_constraints(pts, axial_constraint="face"):
    """Rigid-body suppression as (location list, component list, value list).

    Split out of ``build_cylinder_press_bcs`` so that the unloading phase
    between hits (``build_unload_bcs``) can install exactly the same rows.
    See the ``axial_constraint`` entry in that docstring for the modes.
    """
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

    # ----- [v5] minimal 6-constraint alternative on 3 non-collinear nodes -----
    if axial_constraint == "minimal3":
        face = onp.where(onp.abs(pts[:, 0] - x_bottom) <= 1e-5)[0]
        if len(face) < 3:
            raise ValueError(f"only {len(face)} nodes on the x=0 face -- not enough to build the minimal3 constraint")
        fy, fz = pts[face, 1], pts[face, 2]
        ymin, ymax = fy.min(), fy.max()
        zmin, zmax = fz.min(), fz.max()

        def _corner(ty, tz, taken):
            d = (fy - ty) ** 2 + (fz - tz) ** 2
            order = onp.argsort(d)
            for k in order:
                if int(face[k]) not in taken:
                    return int(face[k])
            raise ValueError("could not pick three distinct corner nodes")

        iA = _corner(ymin, zmin, set())
        iB = _corner(ymax, zmax, {iA})
        iC = _corner(ymin, zmax, {iA, iB})
        rb_nodes = [iA, iB, iC]
        A, B, C = (pts[i] for i in rb_nodes)
        # Degeneracy check: B must differ from A in both y and z; C must
        # differ from A in z.
        tolg = 1e-6 * max(ymax - ymin, zmax - zmin, 1.0)
        if abs(B[2] - A[2]) <= tolg or abs(B[1] - A[1]) <= tolg or abs(C[2] - A[2]) <= tolg:
            raise ValueError(
                f"degenerate minimal3 nodes: A={A}, B={B}, C={C} (they cannot span all rotation modes)")

        nA = onp.array([iA], dtype=onp.int32)
        nB = onp.array([iB], dtype=onp.int32)
        nC = onp.array([iC], dtype=onp.int32)
        rb_locs = [nA, nA, nA, nB, nB, nC]           # A:ux,uy,uz  B:ux,uy  C:ux
        rb_vecs = [0, 1, 2, 0, 1, 0]
        rb_vals = [zero_val] * 6
        print(f"[BC][v5] minimal3 axial constraint: A(node {iA})={A.round(3)} ux,uy,uz | "
              f"B(node {iB})={B.round(3)} ux,uy | C(node {iC})={C.round(3)} ux "
              f"-> 6 scalar constraints in total (the face mode uses {len(face)+2})")
    elif axial_constraint == "face_rx":
        # [v6] The face mode plus the missing rotation-about-x constraint.
        # The face mode (291 constraints) leaves one rigid-body mode free:
        # rotation about the axis through the pin parallel to x.  That mode
        # produces no ux at all, so all 289 end-face constraints are blind to
        # it, and the pin sits on the rotation axis so its uy=uz=0 are
        # satisfied automatically.  Constraining uy on one node far from the
        # pin locks it.  Unlike minimal3, the 289 end-face ux constraints are
        # KEPT, so the face stays planar -- none of the ~10 mm tearing artifact
        # minimal3 produced -- and the added constraint is transverse (uy), so
        # it does not impede axial flow.
        face = onp.where(onp.abs(pts[:, 0] - x_bottom) <= 1e-5)[0]
        zref = float(pin_point[2])
        # Pick the end-face node farthest in z from the pin: the longest
        # lever arm needs the smallest reaction, hence the weakest artifact.
        iQ = int(face[onp.argmax(onp.abs(pts[face, 2] - zref))])
        Q = pts[iQ]
        lever = abs(float(Q[2]) - zref)
        if lever <= 1e-9:
            raise ValueError("face_rx: no end-face node with a z coordinate different from the pin, so rotation about x cannot be constrained")
        nQ = onp.array([iQ], dtype=onp.int32)
        rb_locs = [bottom_face, pin_node, pin_node, nQ]
        rb_vecs = [0, 1, 2, 1]                      # bottom:ux  pin:uy,uz  Q:uy
        rb_vals = [zero_val] * 4
        print(f"[BC][v6] face_rx constraint: {len(face)} end-face nodes ux=0, pin(node "
              f"{int(pin_id)}) uy=uz=0, Q(node {iQ})={Q.round(3)} uy=0 "
              f"(lever |dz|={lever:.2f}mm) -> {len(face)+3} scalar constraints in total; "
              f"rotation about x is locked")
    else:
        rb_locs = [bottom_face, pin_node, pin_node]  # bottom:ux  pin:uy,uz
        rb_vecs = [0, 1, 2]
        rb_vals = [zero_val] * 3

    return rb_locs, rb_vecs, rb_vals


# =============================================================================
# Cylinder side-compression BCs for thermo-mechanical problem
# =============================================================================
def build_cylinder_press_bcs(mesh, R, H, hit, current_sol_u=None, T_linear_fn=None,
                             surface_mask_fn=None, axial_constraint="face"):
    """
    Build dynamic-contact Dirichlet and thermal BCs for a single compression hit.

    Die-contact heat flux ``h (T_platen - T)`` acts only on the faces whose nodes
    are in contact at the current step, through the nodal 0/1 mask published
    every step (not on the whole candidate band around the circumference).

    Axial band: the candidates are the outer-surface material that is CURRENTLY
    between x_min_band and x_max_band in the lab frame (the test is applied to
    ``pts + current_sol_u``), which is what a machine axis commanded to a fixed
    position presses.  The candidate set is fixed for the duration of the hit;
    ``x_min_band``/``x_max_band`` are lab coordinates and may exceed the
    reference length.  (Selecting on the reference coordinates made the die
    follow the material: the hit landed at the band's accumulated axial
    displacement from the commanded position.)  Rotation is about x, so it does
    not mix into the band test.

    The die only pushes: a captured node's normal target is clamped so the
    constraint can only move it INTO the body (``max(y, y_L)`` on the lower
    platen, ``min(y, y_R)`` on the upper); a captured node that does not
    penetrate is pinned where it is.  (Projecting every node of the
    ``tol_contact`` capture layer onto the plane pulled the workpiece outward.)

    axial_constraint : {"face", "face_rx", "minimal3"}
        [v5] How the axial rigid-body motion is suppressed.

        * ``"face"`` (default, = v1..v4) — every node on the x=0 face has
          ``ux=0`` (289 nodes on the 30x16x16 mesh) plus one pinned node with
          ``uy=uz=0``.  That is 291 scalar constraints where 6 suffice, and it
          welds the end face to a rigid wall: material cannot flow axially out
          of that end.  Measured consequence (v4): the simulated axial flow is
          only ~37% of the scans.  It ALSO leaves the rotation about x
          unconstrained — that mode produces no ux at all (so the 289 ux
          constraints do nothing for it) and the pin sits on the rotation axis
          — so it is suppressed only by contact.
        * ``"face_rx"`` [v6] — ``face`` plus ONE extra ``uy`` constraint on the
          end-face node farthest in z from the pin.  That is the minimal fix for
          the missing mode: the 289 distributed ux constraints are kept (so the
          end face stays planar — no tearing), and the added constraint is
          transverse, so it does not impede axial flow.
        * ``"minimal3"`` — exactly 6 constraints on 3 non-collinear nodes of
          the x=0 face, which removes all 6 rigid-body modes and nothing else:

              A = (0, y_min, z_min):  ux, uy, uz
              B = (0, y_max, z_max):  ux, uy
              C = (0, y_min, z_max):  ux

          A kills the 3 translations; B.uy kills rotation about x (B differs
          from A in z); B.ux kills rotation about z (differs in y); C.ux kills
          rotation about y (differs in z).  Material is then free to flow
          axially at both ends, as in free forging.  MEASURED CAVEAT: with only
          3 pinned nodes carrying the whole axial reaction, the x=0 face tears —
          v5 showed ux spanning [-10.19, 0] mm on that face with an 8.9 mm jump
          between adjacent nodes, and its apparent length gain came entirely
          from that artifact (x_max was 6.2 mm SHORTER than v4).  Prefer
          ``face_rx``.

    surface_mask_fn : callable(points)->bool mask or None
        Reference-config test that selects the workpiece OUTER surface nodes
        (used for cooling and as the contact-candidate set).  If None (default),
        the cylindrical test ``|radius - R| <= tol_r`` is used.  Pass a box
        lateral-face test (e.g. from Agility_Forge_physforge_full) to run a
        rectangular billet instead of a cylinder — every other part of the
        contact/rotation/cooling logic is geometry-agnostic.

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
    print("x_min_band:{0}, x_max_band:{1} (selected in the deformed configuration)".format(
        x_min_band, x_max_band))

    # tolerances
    tol_r = 0.25 * (2 * R / 5)
    tol_contact = 0.10  # mm: thickness of the contact-detection layer near platen
    R0 = float(R)

    def in_x_band_np(points):
        return onp.logical_and(points[:, 0] >= x_min_band, points[:, 0] <= x_max_band)

    if surface_mask_fn is None:
        def on_outer_surface_np(points):
            radii = onp.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
            return onp.abs(radii - R0) <= tol_r
    else:
        # Box / arbitrary billet: caller supplies the outer-surface test.
        on_outer_surface_np = surface_mask_fn

    # ----- outer surface node set (reference) for cooling -----
    surface_mask = on_outer_surface_np(pts)
    surface_inds = np.array(onp.where(surface_mask)[0], dtype=int)
    print(f"[BC] outer surface node count: {len(surface_inds)}")

    # ----- candidate contact nodes: outer-surface lateral nodes within x-band -----
    # The candidate set is fixed for the hit; the dynamic contact set is a
    # subset re-selected per load step.  The band test is a geometric window on
    # the axial coordinate, not a topological property, so it is applied in the
    # configuration the die actually sees (see the docstring).
    band_pts = pts + current_sol_u_np
    candidate_mask = surface_mask & in_x_band_np(band_pts)
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

    rb_locs, rb_vecs, rb_vals = rigid_body_constraints(pts, axial_constraint)
    n_rb = len(rb_vecs)
    vecs = rb_vecs + [0, 1, 2, 0, 1, 2]

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

        # Keep the current tangential position; the normal target is clamped
        # so the constraint only ever moves a node INTO the body: the lower
        # platen occupies y < y_L, so admissible material has y >= y_L and the
        # target is max(y, y_L); the upper platen gives min(y, y_R).  A captured
        # node that is not penetrating is pinned where it is rather than
        # dragged out to the plane.
        def _targets(loc, y_plane, side):
            if len(loc) == 0:
                return onp.zeros((0, 3), dtype=pts.dtype)
            target_rot = xdef_cand_rot[loc].copy()
            y_now = target_rot[:, 1]
            target_rot[:, 1] = (onp.maximum(y_now, y_plane) if side == "left"
                                else onp.minimum(y_now, y_plane))
            u_rot = target_rot - X_cand_rot[loc]     # total displacement (rotated)
            return to_global_vectors(u_rot)          # -> global frame

        u_left = _targets(left_loc, y_L, "left")
        u_right = _targets(right_loc, y_R, "right")
        return left_inds, right_inds, u_left, u_right

    # ----- initial placeholder BC (scale = 0 on current deformed config) -----
    left0, right0, uL0, uR0 = compute_contact(current_sol_u_np, 0.0)
    location_fns_u = rb_locs + [left0, left0, left0,
                                right0, right0, right0]
    value_fns = rb_vals + [uL0[:, 0], uL0[:, 1], uL0[:, 2],
                           uR0[:, 0], uR0[:, 1], uR0[:, 2]]
    dirichlet_bc_info_u = [location_fns_u, vecs, value_fns]
    print(f"[BC] initial contact: left={len(left0)}, right={len(right0)}, "
          f"rigid-body constraints={n_rb}")

    def bc_update_fn(problem, sol_u, step_count, scale):
        """Re-detect contact and rebuild the mechanical Dirichlet BC this step.

        Updated-Lagrangian: contact is evaluated from the last converged
        displacement (`sol_u`), so the active set is fixed during each Newton
        solve and updated between steps.
        """
        sol_u_np = onp.asarray(sol_u)
        left_inds, right_inds, u_left, u_right = compute_contact(sol_u_np, scale)

        loc_fns = rb_locs + [left_inds, left_inds, left_inds,
                             right_inds, right_inds, right_inds]
        val_fns = rb_vals + [u_left[:, 0], u_left[:, 1], u_left[:, 2],
                             u_right[:, 0], u_right[:, 1], u_right[:, 2]]
        problem.fes[0].update_Dirichlet_boundary_conditions([loc_fns, vecs, val_fns])

        # [die-contact] Publish the ACTIVE contact set as a nodal 0/1 field.
        # The thermal Neumann surface (location_fns_convect) stays STATIC -- the
        # candidate band -- so no surface integral is rebuilt and nothing is
        # re-jitted.  Only this mask changes per step, and the surface kernel
        # multiplies the die-contact flux by it, so heat is drawn only where the
        # billet actually touches a platen.  Without it the flux was applied to
        # every candidate node around the FULL circumference, i.e. die chilling
        # was spread over several times too much area, including the faces the
        # platens never reach.
        contact_mask = onp.zeros((len(pts), 1), dtype=onp.float64)
        if len(left_inds):
            contact_mask[onp.asarray(left_inds), 0] = 1.0
        if len(right_inds):
            contact_mask[onp.asarray(right_inds), 0] = 1.0
        problem.contact_mask_nodes = np.array(contact_mask)
        # Published so the unloading phase between hits knows which nodes
        # carried the die reaction at the end of this hit.
        problem.contact_left_inds = onp.asarray(left_inds)
        problem.contact_right_inds = onp.asarray(right_inds)

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

    def air_faces(p):
        # [air] every face; refresh_problem_surface_integrals keeps the true
        # boundary and drops the clamped end face (AIR_SURFACE)
        return np.isfinite(p[0])

    dirichlet_bc_info_T = None
    location_fns_convect = [left_reach_nodes, right_reach_nodes, air_faces]

    return dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds


# =============================================================================
# Force-controlled unloading between hits
# =============================================================================
def self_equilibrate(X, R):
    """Remove the rigid-body resultant (net force and net moment) from nodal
    forces ``R`` at positions ``X`` with the least-norm correction.

    Returns ``(R_eq, delta, resultant)`` where ``R = R_eq + delta``, ``R_eq``
    has zero resultant and ``resultant = (F, M)`` is what was removed.
    """
    X = onp.asarray(X, dtype=onp.float64); R = onp.asarray(R, dtype=onp.float64)
    n = len(X)
    if n == 0:
        return R, onp.zeros_like(R), (onp.zeros(3), onp.zeros(3))
    c = X.mean(axis=0)
    A = onp.zeros((3 * n, 6))
    for i, x in enumerate(X - c):
        A[3*i:3*i+3, 0:3] = onp.eye(3)
        A[3*i:3*i+3, 3:6] = onp.array([[0.0, x[2], -x[1]], [-x[2], 0.0, x[0]], [x[1], -x[0], 0.0]])
    r = A.T @ R.reshape(-1)                      # [sum R ; sum (x-c) x R] up to sign
    q = onp.linalg.solve(A.T @ A, r)
    delta = (A @ q).reshape(-1, 3)
    F = R.sum(axis=0)
    M = onp.cross(X - c, R).sum(axis=0)
    return R - delta, delta, (F, M)


def build_unload_bcs(mesh, reaction_u, contact_inds, axial_constraint="face",
                     verbose=True, sol_u=None, balance="self", mode="peel",
                     left_inds=None, right_inds=None, rot_deg=0.0):
    """Boundary conditions for releasing the die between two hits.

    mode : {"peel", "force"}
        ``"peel"`` (default) -- displacement-controlled release.  Every node
        that is in contact at the end of the hit gets a rank in [0, 1), by
        distance from its platen patch's centroid in the die plane, outermost
        first (the order a real contact separates in).  At release fraction
        ``s`` the nodes with rank below ``s`` are free and the others are
        PINNED (all three components) where the hit left them; at ``s = 1``
        the body is free.  The band is thus held kinematically until the last
        increment, which drops only the innermost nodes' reaction.
        ``"force"`` -- the force-controlled release described below.  It is
        exact at ``lambda = 1`` but it is LOAD CONTROL on a body whose
        pinched band is held in rotation only by the bending stiffness of a
        hot bar: measured on the cylinder demo (1096 C under the die, E of
        6 GPa, 13 kN pinch) the free end swung 20 mm at ``lambda = 0.3`` and
        came back to 12 mm at ``lambda = 0`` -- an unstable equilibrium path
        with plastic rotation, not springback.  It behaves on a stocky box
        billet.  Kept as an option; prefer ``"peel"``.

    At the end of a hit the contact nodes are held by Dirichlet rows, and the
    force those rows exert is the (unconstrained) internal force at those dofs,
    i.e. the residual before boundary conditions are applied.  Dropping all the
    rows at once asks Newton to recover the whole springback in one solve,
    which fails for large bites independently of dt (the jump is in the
    constraint SET, not in the load).  Here the rows are dropped but the force
    is kept: ``lambda * R`` is applied as an external nodal force on the former
    contact nodes, and the stepper's load ramp takes ``lambda`` from 1 to 0.
    At ``lambda = 1`` the state is already in equilibrium (zero Newton
    iterations); at ``lambda = 0`` the body is free of the die.

    balance : {"self", "raw"}
        ``"self"``: only the self-equilibrated part of ``R`` is released.  The
        two platen reactions never cancel exactly (the contact sets are
        detected discretely, e.g. 29 against 27 nodes), and during the hit the
        imbalance is carried by the clamped end through the bar.  Applying the
        raw ``lambda R`` to the free bar makes that imbalance a transverse load
        on a cantilever whose die zone is at 1000 C and at yield: measured on
        the cylinder demo, the free end swung 10 mm (a 5.7 deg rotation about
        z, growing linearly with x beyond the die) -- a plastic hinge, not
        springback.  A die pair can only pinch, so the rigid-body resultant
        (net force and moment) is removed from ``R`` with the least-norm
        correction and the clamp keeps carrying it.  The residual at
        ``lambda = 1`` is then the removed imbalance, released in the first
        increment.  ``"raw"`` applies ``R`` as captured.

    The die and the workpiece do not touch during unloading, so no contact is
    detected and the die heat flux mask is zero.  Only the rigid-body rows of
    ``rigid_body_constraints`` remain -- with one addition for the ``"face"``
    mode: that mode leaves the rotation about x free (it is held only by the
    platens during a hit), so with no contact it is a zero-energy mode, the
    tangent is singular, and the release drifts along it (measured: 24 mm of
    "springback" on the cylinder demo, a rigid rotation).  The ``face_rx``
    node is therefore added with its ``uy`` HELD AT ITS CURRENT VALUE (not
    zero), which removes the mode without moving anything.  ``sol_u`` is the
    displacement at the start of the release and is required for ``"face"``.

    Parameters
    ----------
    reaction_u : (N, 3) array
        Internal force at the converged end-of-hit state, global frame.
    contact_inds : int array
        Nodes that were in contact at the last step of the hit.

    Returns
    -------
    dirichlet_bc_info_u, bc_update_fn(problem, sol_u, step_count, scale)
    """
    pts = onp.array(mesh.points)
    if mode not in ("peel", "force"):
        raise ValueError(f"mode must be 'peel' or 'force', got {mode!r}")
    rb_locs, rb_vecs, rb_vals = rigid_body_constraints(pts, axial_constraint)
    rb_locs, rb_vecs, rb_vals = list(rb_locs), list(rb_vecs), list(rb_vals)
    if axial_constraint == "face":
        if sol_u is None:
            raise ValueError("build_unload_bcs needs sol_u to hold the free rotation "
                             "mode of the 'face' constraint")
        x_bottom = float(pts[:, 0].min())
        face = onp.where(onp.abs(pts[:, 0] - x_bottom) <= 1e-5)[0]
        pin_id = int(onp.argmin((pts[:, 0] - x_bottom) ** 2
                                + (pts[:, 1] - float(pts[:, 1].min())) ** 2
                                + pts[:, 2] ** 2))
        zref = float(pts[pin_id, 2])
        iQ = int(face[onp.argmax(onp.abs(pts[face, 2] - zref))])
        uy_hold = float(onp.asarray(sol_u)[iQ, 1])

        def hold_val(p, _v=uy_hold):
            return _v

        rb_locs.append(onp.array([iQ], dtype=onp.int32))
        rb_vecs.append(1)
        rb_vals.append(hold_val)
        if verbose:
            print(f"[unload] 'face' mode: rotation about x held by pinning uy of "
                  f"node {iQ} at its current value {uy_hold:+.4f} mm")
    contact_inds = onp.asarray(contact_inds, dtype=int)

    n_nodes = len(pts)
    dirichlet_bc_info_u = [list(rb_locs), list(rb_vecs), list(rb_vals)]

    if mode == "peel":
        if sol_u is None:
            raise ValueError("build_unload_bcs(mode='peel') needs sol_u")
        u_hold = onp.asarray(sol_u, dtype=onp.float64)
        Rm = onp.asarray(Rotation.from_euler("x", float(rot_deg), degrees=True).as_matrix())
        x_def_rot = (pts + u_hold) @ Rm.T
        rank = onp.full(n_nodes, -1.0)
        sides = [onp.asarray(a, dtype=int) for a in (left_inds, right_inds) if a is not None]
        if not sides:
            sides = [contact_inds]
        for idx in sides:
            if len(idx) == 0:
                continue
            face = x_def_rot[idx][:, [0, 2]]              # along the die, across it
            d = onp.linalg.norm(face - face.mean(axis=0), axis=1)
            order = onp.argsort(-d)                       # farthest from the centre first
            r = onp.empty(len(idx))
            r[order] = onp.arange(len(idx)) / float(len(idx))
            rank[idx] = r
        held0 = onp.where(rank >= 0.0)[0]
        if verbose:
            print(f"[unload] peel: {len(held0)} contact nodes released outermost-first "
                  f"over the stroke; rigid-body constraints={len(rb_vecs)}")

        def bc_update_fn(problem, sol_u_now, step_count, scale):
            held = onp.where((rank >= 0.0) & (rank >= float(scale)))[0].astype(onp.int32)
            locs = list(rb_locs) + [held, held, held]
            vecs = list(rb_vecs) + [0, 1, 2]
            vals = list(rb_vals) + [u_hold[held, 0], u_hold[held, 1], u_hold[held, 2]]
            problem.fes[0].update_Dirichlet_boundary_conditions([locs, vecs, vals])
            problem.contact_mask_nodes = np.zeros((n_nodes, 1))
            problem.contact_left_inds = onp.zeros((0,), dtype=int)
            problem.contact_right_inds = onp.zeros((0,), dtype=int)
            problem.f_ext_nodes = None
            if verbose:
                print(f"  [unload step {step_count}] s={float(scale):.4f} "
                      f"held={len(held)} of {len(held0)}")

        return dirichlet_bc_info_u, bc_update_fn

    # ---- mode == "force" ----
    R_full = onp.zeros_like(pts, dtype=onp.float64)
    if len(contact_inds):
        R_c = onp.asarray(reaction_u)[contact_inds]
        X_c = pts[contact_inds] + (0.0 if sol_u is None else onp.asarray(sol_u)[contact_inds])
        R_eq, delta, (F, M) = self_equilibrate(X_c, R_c)
        if verbose:
            print(f"[unload] reaction resultant over the contact nodes: force "
                  f"({F[0]:+.3e}, {F[1]:+.3e}, {F[2]:+.3e}) N, moment "
                  f"({M[0]:+.3e}, {M[1]:+.3e}, {M[2]:+.3e}) N.mm; |R| = "
                  f"{onp.linalg.norm(R_c):.4e} N, imbalance |delta| = "
                  f"{onp.linalg.norm(delta):.4e} N")
        if balance == "self":
            R_full[contact_inds] = R_eq
        elif balance == "raw":
            R_full[contact_inds] = R_c
        else:
            raise ValueError(f"balance must be 'self' or 'raw', got {balance!r}")
    R_full_j = np.array(R_full)
    force_norm = float(onp.linalg.norm(R_full))
    if verbose:
        print(f"[unload] releasing {len(contact_inds)} contact nodes (force, {balance}); "
              f"|f| = {force_norm:.4e} N, rigid-body constraints={len(rb_vecs)}")

    def bc_update_fn(problem, sol_u_now, step_count, scale):
        lam = 1.0 - float(scale)
        problem.fes[0].update_Dirichlet_boundary_conditions(
            [list(rb_locs), list(rb_vecs), list(rb_vals)])
        problem.contact_mask_nodes = np.zeros((n_nodes, 1))
        problem.contact_left_inds = onp.zeros((0,), dtype=int)
        problem.contact_right_inds = onp.zeros((0,), dtype=int)
        problem.f_ext_nodes = lam * R_full_j
        if verbose:
            print(f"  [unload step {step_count}] lambda={lam:.4f} "
                  f"|f_ext| = {lam * force_norm:.4e} N")

    return dirichlet_bc_info_u, bc_update_fn


# =============================================================================
# Face area weights
# =============================================================================
def reference_nanson_scale(fe, boundary_inds):
    """Area weights of the selected faces (num_selected_faces, num_face_quads).

    Nanson's formula, da = det(J) |N . J^-1| dA, with N the REFERENCE facet
    normal.  ``FiniteElement.get_face_shape_grads`` in jax_fem_checkpoint uses
    the physical face normal for N, which is right only where the reference
    and physical axes coincide (an axis-aligned HEX8); on TET4 it over-counts
    (box y faces: 3.42 x and 1.98 x on alternate triangles, 2.70 x in total).
    The sum over a face of these weights equals its geometric area.  jax_fem_
    checkpoint is shared with other applications and is not modified here.
    """
    inds = onp.asarray(boundary_inds)
    coos = onp.take(onp.asarray(fe.points), onp.asarray(fe.cells), axis=0)[inds[:, 0]]
    grads = onp.asarray(fe.face_shape_grads_ref)[inds[:, 1]]
    J = onp.sum(coos[:, None, :, :, None] * grads[:, :, :, None, :], axis=2)
    N = onp.asarray(fe.face_normals)[inds[:, 1]]
    # The sign of the owner cell's volume weights.  On HEX8, jax_fem_checkpoint's
    # volume shape functions (basis_new.py, re_order) give det J < 0 on a
    # right-handed mesh, so every JxW is negative, which only flips the sign of whole
    # volume terms; its face shape functions give det J > 0.  A positive surface
    # weight would then enter with the opposite sign to the volume terms: the die
    # would heat the bar.  TET4: +1.
    vol_sign = onp.sign(onp.asarray(fe.JxW)[inds[:, 0], 0])[:, None]
    return (onp.linalg.norm((N[:, None, None, :] @ onp.linalg.inv(J))[:, :, 0, :], axis=-1)
            * onp.abs(onp.linalg.det(J))
            * onp.asarray(fe.face_quad_weights)[inds[:, 1]]
            * vol_sign)


# Index of the air surface in location_fns ([lower die, upper die, air]).
AIR_SURFACE = 2


# =============================================================================
# True boundary faces
# =============================================================================
def true_boundary_face_flags(fe):
    """(num_cells, num_faces_per_cell) bool: the face belongs to this cell only.

    ``FiniteElement.get_boundary_conditions_inds`` (jax_fem_checkpoint) selects
    every face of every cell whose vertices all satisfy the location function
    and never checks that the face is on the boundary.  The surface location
    functions of this package test NODE SETS (``np.isin(ind, ...)``), so an
    interior face whose vertices are all in the set was selected too -- and,
    being shared by two cells, counted twice -- which multiplied the area of
    the die heat flux (box mesh 1.77 x, rect meshes 1.25-1.49 x the true area).
    A face shared by two cells is interior; a face of one cell only is on the
    boundary.
    """
    cfi = onp.take(onp.asarray(fe.cells), onp.asarray(fe.face_inds), axis=1)  # (nc, nf, nv)
    keys = onp.sort(cfi.reshape(-1, cfi.shape[-1]), axis=1)
    _, inv, cnt = onp.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return (cnt[onp.asarray(inv).reshape(-1)] == 1).reshape(cfi.shape[:2])


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
    selected = problem.fes[0].get_boundary_conditions_inds(problem.location_fns)
    # keep only the faces on the true boundary (see true_boundary_face_flags);
    # the connectivity does not change in this package, so the flags are cached
    fe0 = problem.fes[0]
    flags = getattr(problem, "_true_boundary_faces", None)
    if flags is None or flags.shape[0] != fe0.num_cells:
        flags = true_boundary_face_flags(fe0)
        problem._true_boundary_faces = flags
    problem.boundary_inds_list = []
    for k, b in enumerate(selected):
        b = onp.asarray(b)
        keep = flags[b[:, 0], b[:, 1]] if len(b) else onp.zeros((0,), dtype=bool)
        if len(b) and not keep.all() and k != AIR_SURFACE:
            print(f"  [surface] location {k}: {int(keep.sum())} of {len(b)} selected faces "
                  f"are on the boundary; {int((~keep).sum())} interior faces dropped")
        b = b[keep]
        if k == AIR_SURFACE and len(b):
            # [air] the end face x = x_min is clamped (held by the manipulator
            # in the rig): no air there, it stays adiabatic as before
            pts0 = onp.asarray(fe0.points)
            fv = onp.take_along_axis(onp.asarray(fe0.cells)[b[:, 0]],
                                     onp.asarray(fe0.face_inds)[b[:, 1]], axis=1)   # (nf, nv)
            fx = pts0[fv][..., 0]
            on_end = onp.all(onp.abs(fx - pts0[:, 0].min()) <= 1e-5, axis=1)
            b = b[~on_end]
            if not getattr(problem, "_air_reported", False):
                print(f"  [surface] air: {len(b)} outer faces (clamped end face excluded, "
                      f"{int(on_end.sum())} faces)")
                problem._air_reported = True
        problem.boundary_inds_list.append(b)

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
            # the library's area weights are replaced by the reference-normal
            # ones (see reference_nanson_scale); its shape gradients are kept
            face_shape_grads_physical, _ = fe.get_face_shape_grads(boundary_inds)
            nanson_scale_i = reference_nanson_scale(fe, boundary_inds)
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
