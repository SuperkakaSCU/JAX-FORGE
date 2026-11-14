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
def build_cylinder_press_bcs(mesh, R, H, hit, current_sol_u=None, T_linear_fn=None,
                             surface_mask_fn=None, axial_constraint="face",
                             band_config="reference", band_sol_u=None,
                             edge_radius=0.0, axial_slip=False, die_contact=None,
                             die_start=None):
    """
    die_start : None or (y0_L, y0_R)
        Force-based contact only.  Start the die planes at these positions
        instead of the datum read off the band.  Used when a compression
        stroke is CONTINUED after a relocation of the mesh inside the stroke
        (run_cold --stroke-relocate): the caller backs the die off its last
        pose by the largest penetration of the relocated nodes and sets the
        travel so that the stroke still ends on y_c -+ rho of the original
        stroke; the band datum would re-centre the dies on the relocated
        nodes instead.

    Build dynamic-contact Dirichlet and thermal BCs for a single compression hit.

    axial_constraint : {"face", "face_rx", "face_fixed", "minimal3"}
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
          transverse, so it does not impede axial flow.  MEASURED CAVEAT
          (roll3.1, 12 hits, 2026-11-10): the rotation lock is carried by two
          single nodes (the pin and Q); they yield at ~1 kN (~20 N m) and the
          whole bar, held end included, rolls about its axis under the dies
          (-8 deg in hit 4, +17 in hit 8, -54 / -49 in hits 11 / 12), so later
          dents land at wrong angles on the bar.  Use ``face_fixed`` for a
          multi-hit series.
        * ``"face_fixed"`` [v7] -- every node of the end face has
          ``ux = uy = uz = 0``: the held end is a rigid grip.  The torque about
          the bar axis is carried by the whole face (hundreds of nodes) instead
          of two, so it stays elastic and the bar cannot roll.  The face cannot
          change shape either; in an isothermal run that costs nothing (no
          thermal strain), and the nearest die band is ~27 mm away.
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
        lateral-face test (e.g. from Agility_Forge_physforge) to run a
        rectangular billet instead of a cylinder — every other part of the
        contact/rotation/cooling logic is geometry-agnostic.

    band_config : {"reference", "deformed"}
        Configuration in which the axial x-band of contact candidates is
        selected.

        * ``"reference"`` (default, = v1..v6) — the band is the material that
          STARTED between x_min_band and x_max_band.  Cheap and fixed, but it
          makes the die follow the material instead of staying put: the stock
          elongates under the press, so the selected material has drifted to
          lab position x + u_x by the time the die arrives, and the hit lands
          there rather than at the commanded z.  The offset is exactly the
          mean axial displacement of the band, which grows with accumulated
          strain -- under 1 mm on roll1 (+2.5 mm total elongation), 2-8 mm on
          roll2 (+10 mm), and up to 16 mm on roll3 (+35 mm), i.e. larger than
          the 11 mm band itself.
        * ``"deformed"`` — the band is the material that is CURRENTLY between
          x_min_band and x_max_band in the lab frame, i.e. the test is applied
          to ``pts[:, 0] + current_sol_u[:, 0]``.  This is what a machine axis
          commanded to a fixed z actually presses.  The candidate set is still
          fixed for the duration of the hit (re-selected per hit, not per load
          step).  Measured justification: the band's mean axial motion WITHIN a
          single hit runs 0.07-0.60 mm on roll2 and up to 1.13 mm on roll3,
          against a cumulative drift of up to 16 mm -- 8% or less of what is
          being corrected on every hit past the third, so re-selecting per load
          step would buy little and would re-jit the surface integrals.

        Rotation is about x, so it does not mix into the band test either way.

    axial_slip : bool
        False (default) is full stick: a contact node has all three
        displacement components prescribed.  True leaves the AXIAL (x)
        component of contact nodes free -- frictionless sliding along the bar
        axis -- while the normal and circumferential components stay
        prescribed.  The press rotates about x, so its normal and
        circumferential directions span exactly global y and z, and the
        prescription is still a plain component-wise Dirichlet condition.
        This relieves the surface shear that full stick forces into the first
        element layer when material flows axially under the die.
    edge_radius : float
        Smooth-transition regularization of the die edge, mm.  0 (default) is
        the sharp flat band.  r > 0 adds a tangent circular fillet of radius r
        OUTSIDE each band edge: the platen surface is flat over the band and
        recedes by g(s) = r - sqrt(r^2 - s^2) at distance s in [0, r] past the
        edge, and is absent beyond.  The flat part, and so the travel bite_for
        reads off it, is unchanged; only the corner is rounded.  Candidates are
        taken over the band widened by r on each side, while the platen start
        planes are still read off the flat band alone.
    band_sol_u : array or None
        Displacement field used to place the axial band, when it must differ
        from ``current_sol_u``.  Pass the field at the START OF THE HIT to keep
        one band for all three strokes of that hit.

        Why that matters: with the band re-selected per stroke, the retract of
        hit 8 picked up material the compression stroke had never pressed --
        391 candidates became 346 -- and that unpressed material, standing
        0.587 mm proud, set ``y_rot_max0``.  The upper platen therefore started
        the retract 0.587 mm outside its own flat and held 1 node where the
        lower one held 54.  Measured on hit 8 of roll3.2: the lower platen
        finished the compression at -17.3180 + 7.892 = -9.4260 and the retract
        read -9.4259, agreeing to 1e-4, while the upper finished at
        16.9665 - 7.892 = +9.0745 and the retract read +9.6612.  One band per
        hit removes that disagreement, because every node the platens pressed
        is still in the set when the retract reads its datum.

        The cost is that the band no longer follows the material WITHIN a hit.
        That motion was measured at 0.07-1.13 mm against a cumulative drift of
        up to 16 mm -- 8% or less of what band_config="deformed" is there to
        correct -- so the between-hit correction is kept and only the
        within-hit refinement is given up.

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
    print("x_min_band:{0}, x_max_band:{1} (selected in the {2} configuration{3})".format(
        x_min_band, x_max_band, band_config,
        "" if band_sol_u is None else ", frozen at the start of the hit"))

    # tolerances
    tol_r = 0.25 * (2 * R / 5)
    tol_contact = 0.10  # mm: thickness of the contact-detection layer near platen
    R0 = float(R)

    def in_x_band_np(points):
        return onp.logical_and(points[:, 0] >= x_min_band, points[:, 0] <= x_max_band)

    r_edge = float(edge_radius)
    if r_edge < 0.0:
        raise ValueError(f"edge_radius must be >= 0, got {edge_radius!r}")
    x_mid_band = 0.5 * (x_min_band + x_max_band)
    half_band = 0.5 * (x_max_band - x_min_band)

    def in_x_reach_np(points):
        # flat band plus the fillets; identical to in_x_band_np when r_edge == 0
        return onp.logical_and(points[:, 0] >= x_min_band - r_edge,
                               points[:, 0] <= x_max_band + r_edge)

    def platen_recess(x):
        """Normal offset of the platen surface from its flat plane at axial x.

        0 on the flat band, the fillet profile within r_edge of an edge, +inf
        beyond (no die there).  With r_edge == 0 every candidate lies on the
        flat band, so this is identically 0 for them.
        """
        s_out = onp.abs(onp.asarray(x) - x_mid_band) - half_band
        g = onp.zeros_like(s_out)
        if r_edge > 0.0:
            on_fillet = (s_out > 0.0) & (s_out <= r_edge)
            g[on_fillet] = r_edge - onp.sqrt(r_edge ** 2 - s_out[on_fillet] ** 2)
            g[s_out > r_edge] = onp.inf
        return g

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
    # The candidate set is fixed for the hit; the dynamic contact set is a subset
    # re-selected per load step.  Which material is a candidate depends on
    # band_config -- see the docstring.
    if band_config == "deformed":
        # The die is a machine axis at a fixed lab coordinate, so the band has
        # to be the material that is there NOW, not the material that started
        # there.  pts + u is the current configuration; only the x component
        # matters for an axial band.  band_sol_u overrides which "now" that is
        # -- see its docstring entry.
        band_u = (current_sol_u_np if band_sol_u is None
                  else onp.asarray(band_sol_u))
        band_pts = pts + band_u
    elif band_config == "reference":
        band_pts = pts
    else:
        raise ValueError(f"band_config must be 'reference' or 'deformed', got {band_config!r}")
    candidate_mask = surface_mask & in_x_reach_np(band_pts)
    candidate_inds = onp.where(candidate_mask)[0].astype(onp.int32)
    n_cand = len(candidate_inds)
    # the subset on the flat band: the platen start planes are read off it
    # alone, which is what bite_for measures the travel on
    cand_flat = in_x_band_np(band_pts[candidate_inds])
    print(f"[BC] candidate contact node count (x-band outer surface): {n_cand}"
          + ("" if r_edge == 0.0 else
             f" ({int(cand_flat.sum())} on the flat band, edge fillet r = {r_edge} mm)"))
    if n_cand == 0:
        print("[BC WARNING] no candidate contact nodes. Check x_band / tol_r / mesh.")

    # reference rotated coordinates of candidates
    X_cand = pts[candidate_inds]
    X_cand_rot = transform_points(X_cand)                       # (n_cand, 3)

    # initial deformed (rotated) coordinates at the START of this hit
    xdef0_cand = X_cand + current_sol_u_np[candidate_inds]
    xdef0_cand_rot = transform_points(xdef0_cand)
    if n_cand > 0 and cand_flat.any():
        y_rot_min0 = float(xdef0_cand_rot[cand_flat, 1].min())
        y_rot_max0 = float(xdef0_cand_rot[cand_flat, 1].max())
    else:
        y_rot_min0 = float(transform_points(pts)[:, 1].min())
        y_rot_max0 = float(transform_points(pts)[:, 1].max())
    # ----- peel order for a retract -------------------------------------
    # Freeing a whole contact patch in one step is what breaks the release.
    # On hit 6 of roll3 the patch carries ~130 kN; dropping all of it at once
    # makes the first Newton step large enough to run away, and the solve went
    # to NaN on six consecutive retries at the normal starting residual of
    # 9.6e4 -- halving dt cannot help, because the change is in the constraint
    # SET, not in the load magnitude.
    #
    # So the patch is peeled instead.  Each node that starts the retract in
    # contact is given a rank in [0, 1); at travel fraction s every node whose
    # rank is below s has been let go, and the ones still held are PINNED where
    # they were when the retract began -- never dragged after the plane, which
    # is the pull a die cannot exert.  By s = 1 the patch is free and the state
    # is the same free-surface equilibrium as an instantaneous release, reached
    # over ~15 steps instead of one.  Ranks run outermost-first, by distance
    # from the patch centroid in the die's own plane, because that is the order
    # a real contact separates in.
    peel_rank_L = onp.full(n_cand, -1.0)
    peel_rank_R = onp.full(n_cand, -1.0)
    if n_cand > 0 and compression_displacement < 0.0:
        y0_cand = xdef0_cand_rot[:, 1]
        g0_cand = platen_recess(xdef0_cand_rot[:, 0])
        in_L0 = y0_cand <= (y_rot_min0 - g0_cand + tol_contact)
        in_R0 = y0_cand >= (y_rot_max0 + g0_cand - tol_contact)

        def _peel(mask, out):
            idx = onp.where(mask)[0]
            if len(idx) == 0:
                return
            face = xdef0_cand_rot[idx][:, [0, 2]]      # along the die, across it
            d = onp.linalg.norm(face - face.mean(axis=0), axis=1)
            order = onp.argsort(-d)                    # farthest from centre first
            rank = onp.empty(len(idx))
            rank[order] = onp.arange(len(idx)) / float(len(idx))
            out[idx] = rank

        _peel(in_L0, peel_rank_L)
        _peel(in_R0, peel_rank_R)
        print(f"[BC] retract: peeling {int((peel_rank_L >= 0).sum())} left and "
              f"{int((peel_rank_R >= 0).sum())} right contact nodes over the stroke")

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
    elif axial_constraint == "face_fixed":
        # [v7] The whole end face is held in all three directions (rigid grip).
        # face_rx locks the rotation about x with two single nodes; under the
        # torque of a die that lands on an inclined flat they yield (~1 kN) and
        # the bar rolls about its axis.  Here every end-face node carries a
        # share of that torque, so the hold stays elastic.
        face = onp.where(onp.abs(pts[:, 0] - x_bottom) <= 1e-5)[0]
        if len(face) < 3:
            raise ValueError(f"face_fixed: only {len(face)} nodes on the end face")
        rb_locs = [bottom_face, bottom_face, bottom_face]   # end face: ux, uy, uz
        rb_vecs = [0, 1, 2]
        rb_vals = [zero_val] * 3
        _rf = onp.hypot(pts[face, 1] - pts[face, 1].mean(), pts[face, 2] - pts[face, 2].mean())
        print(f"[BC][v7] face_fixed constraint: {len(face)} end-face nodes ux=uy=uz=0 "
              f"(face radius up to {_rf.max():.2f} mm) -> {3 * len(face)} scalar constraints "
              f"in total; the held end is a rigid grip (no roll about x, no in-plane motion)")
    elif axial_constraint == "face":
        rb_locs = [bottom_face, pin_node, pin_node]  # bottom:ux  pin:uy,uz
        rb_vecs = [0, 1, 2]
        rb_vals = [zero_val] * 3
    else:
        raise ValueError(f"unknown axial_constraint {axial_constraint!r}")

    n_rb = len(rb_vecs)
    contact_comps = (1, 2) if axial_slip else (0, 1, 2)
    vecs = rb_vecs + list(contact_comps) + list(contact_comps)
    if axial_slip:
        print("[BC] axial slip: contact nodes free along x (frictionless), "
              "held in y and z", flush=True)

    def _contact_bc(left_i, right_i, u_l, u_r):
        locs = [left_i] * len(contact_comps) + [right_i] * len(contact_comps)
        vals = ([u_l[:, c] for c in contact_comps] + [u_r[:, c] for c in contact_comps])
        return locs, vals

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
        # local recess of the platen surface (0 on the flat band); rotation is
        # about x, so the rotated x is the lab x the band is defined in
        gc = platen_recess(xdef_cand_rot[:, 0])

        # moving platen planes (rotated normal coordinate)
        y_L = y_rot_min0 + scale * compression_displacement      # lower platen moves up
        y_R = y_rot_max0 - scale * compression_displacement      # upper platen moves down

        # A die can push and never pull, so the tol_contact capture band is only
        # applied while the platen ADVANCES.  On a retract the band kept every
        # node that had just been pressed -- it still sits within 0.1 mm of a
        # plane that is moving away -- and the projection below then prescribed
        # it to FOLLOW the plane outward, i.e. the die dragged the workpiece.
        # That is what broke hit 8 of both roll3 runs: the retract opened with
        # 54 nodes held on the left platen against 1 on the right, so a fully
        # yielded region (alpha ~ 3.7, and Q = 0, so no hardening) was pulled
        # from one side with only the 4 rigid-body constraints opposing it, and
        # Newton could not converge at any step size -- dt was cut by four
        # decades with no effect.  With no capture band on the retract, only
        # genuinely penetrating nodes are held, so the patch is freed at once,
        # which is also what the machine does.
        if compression_displacement >= 0.0:
            left_local = yc <= (y_L - gc + tol_contact)
            right_local = yc >= (y_R + gc - tol_contact)
            if axial_slip:
                # With the axial component free, a contact node can slide past
                # the band edge.  The candidate set is fixed at the start of the
                # hit, so without this test such a node stayed held and kept
                # being pressed -- the die effectively widened with the
                # material.  Only nodes whose CURRENT x lies under the die
                # (flat band plus fillets) may be in contact.  Under full stick
                # a held node cannot move in x, so this is skipped there to
                # keep that path exactly as before.
                under_die = in_x_reach_np(xdef_cand_rot)
                n_out = int(((left_local | right_local) & ~under_die).sum())
                if n_out:
                    print(f"  [BC] axial slip: {n_out} candidate node(s) slid past "
                          f"the band edge and are released", flush=True)
                left_local = left_local & under_die
                right_local = right_local & under_die
        else:
            # retract: hold the part of the patch whose peel rank is not yet
            # reached, release the rest (see the peel order built above)
            left_local = (peel_rank_L >= 0.0) & (peel_rank_L >= scale)
            right_local = (peel_rank_R >= 0.0) & (peel_rank_R >= scale)

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

        # Project onto the platen plane: keep the current tangential position,
        # set the normal one to the plane -- but CLAMPED so the constraint can
        # only move a node INTO the body, never out of it.  The lower platen
        # occupies y < y_L, so admissible material has y >= y_L and the target
        # is max(y, y_L); the upper platen gives min(y, y_R).  Without the
        # clamp the plane drags the node in whichever direction it happens to
        # move, which is the non-physical pull described above.  A node that is
        # held but not penetrating is simply pinned where it is, so the
        # frictional-stick character of the contact is unchanged.
        def _targets(loc, y_flat, side):
            if len(loc) == 0:
                return onp.zeros((0, 3), dtype=pts.dtype)
            target_rot = xdef_cand_rot[loc].copy()
            if compression_displacement < 0.0:
                # held, not driven: keep the node exactly where the retract
                # found it, so the die neither pushes nor pulls it
                target_rot[:, 1] = xdef0_cand_rot[loc, 1]
            else:
                y_now = target_rot[:, 1]
                y_plane = (y_flat - gc[loc]) if side == "left" else (y_flat + gc[loc])
                target_rot[:, 1] = (onp.maximum(y_now, y_plane) if side == "left"
                                    else onp.minimum(y_now, y_plane))
            u_rot = target_rot - X_cand_rot[loc]     # total displacement (rotated)
            return to_global_vectors(u_rot)          # -> global frame

        u_left = _targets(left_loc, y_L, "left")
        u_right = _targets(right_loc, y_R, "right")
        return left_inds, right_inds, u_left, u_right

    # ===== force-based contact (lib/die_contact.py) =====
    # The die is not a Dirichlet condition: only the rigid-body constraints
    # are prescribed, and the contact object receives the die pose for every
    # step -- the same flat planes y_L / y_R as above, read off the flat band,
    # with the die profile (fillet, flank) hung on them -- and applies the
    # normal and friction forces itself, at every Newton iteration, on all
    # of its candidate nodes.  The travel bookkeeping of the hit is unchanged.
    if die_contact is not None:
        dirichlet_bc_info_u = [list(rb_locs), list(rb_vecs), list(rb_vals)]
        print(f"[BC] force-based die contact on {die_contact.nc} candidate nodes "
              f"({die_contact.profile.describe()}), rigid-body constraints={n_rb}")

        # A RELEASE starts from where the die actually is -- the pose of the
        # last compression step -- not from a datum re-read off the band.
        # The band holds nodes that slid past the die edge and stand proud
        # of the pressed patch (1.6-1.8 mm on hit 2 of roll3.1), and the
        # re-read datum is theirs: the die would open 1.8 mm clear of the
        # patch at scale 0 and drop the whole press load in the first step.
        # That is what the prescribed-displacement model does there
        # ("contact left=0 right=0" at step 1 of the release); it survives
        # at 85 kN and diverges at 102 kN (Q 60 run, mesh A).
        y0_L, y0_R = y_rot_min0, y_rot_max0
        if compression_displacement < 0.0 and onp.isfinite(die_contact.y_L) \
                and onp.isfinite(die_contact.y_R) \
                and onp.allclose(die_contact.R, rot_matrix):
            y0_L, y0_R = die_contact.y_L, die_contact.y_R
            print(f"[BC] release starts from the last die pose (y_L={y0_L:.4f}, "
                  f"y_R={y0_R:.4f}); band datum would be ({y_rot_min0:.4f}, "
                  f"{y_rot_max0:.4f})")
        if die_start is not None and compression_displacement >= 0.0:
            y0_L, y0_R = float(die_start[0]), float(die_start[1])
            print(f"[BC] continued stroke: dies start at (y_L={y0_L:.4f}, y_R={y0_R:.4f}) "
                  f"and travel {compression_displacement:.4f} mm per side; band datum "
                  f"would be ({y_rot_min0:.4f}, {y_rot_max0:.4f})")

        # A compression stroke that starts from the band datum starts with the
        # whole die face clear of the bar (2026-11-09).  The datum is read off
        # the FLAT band only, but the die face goes on beyond it (fillet /
        # sigmoid wall), and where the bar next to the band stands proud of it
        # -- a band laid on an earlier, thinner flat, or the bulge a release
        # leaves at the edge of the pressed patch -- the wall already sat inside
        # the bar before the die moved: 0.92 mm on 38 nodes for hit 6 of roll3.1
        # at the scanned position, 2.3 MN of penalty force in the first step,
        # the solve rolled the bar about its axis and failed at any step size
        # (0.01-0.06 mm on the second strokes, 0.10 mm on hit 6 at the commanded
        # position).  Both dies open by the deepest penetration of the face,
        # re-checked until none is left, and the travel grows by the same
        # amount, so the end pose -- the closure target -- is unchanged.  The
        # driver reads die_contact.start_backoff to keep its travel bookkeeping
        # (and the platen speed) consistent.  A continued stroke (die_start,
        # which the relocation already backs off) and a release keep their own
        # start.
        die_contact.start_backoff = 0.0
        if die_start is None and compression_displacement > 0.0:
            backoff = 0.0
            # up to 200 passes: on the inclined wall one pass removes only
            # cos(slope) of the normal penetration (8 passes left up to
            # 0.02 mm on steep contacts; 2026-11-10)
            for _ in range(200):
                die_contact.set_pose(rot_matrix, x_mid_band, y0_L, y0_R)
                d0 = die_contact._geometry(current_sol_u_np)[0]
                fin = onp.isfinite(d0)
                pen0 = float(max(0.0, -d0[fin].min())) if fin.any() else 0.0
                if pen0 <= 1e-6:
                    break
                y0_L, y0_R = y0_L - pen0, y0_R + pen0
                backoff += pen0
            if backoff > 0.0:
                die_contact.set_pose(rot_matrix, x_mid_band, y0_L, y0_R)
                d0 = die_contact._geometry(current_sol_u_np)[0]
                n_in = int((d0[onp.isfinite(d0)] < -1e-6).sum())
                compression_displacement += backoff
                die_contact.start_backoff = backoff
                print(f"[BC] die start: the die face sat {backoff:.4f} mm inside the bar at the "
                      f"band datum ({y_rot_min0:.4f}, {y_rot_max0:.4f}); dies start at "
                      f"(y_L={y0_L:.4f}, y_R={y0_R:.4f}) and travel {compression_displacement:.4f} "
                      f"mm per side (end pose unchanged; {n_in} nodes still inside after the check)")

        def bc_update_force(problem, sol_u, step_count, scale):
            sol_u_np = onp.asarray(sol_u)
            y_L = y0_L + scale * compression_displacement
            y_R = y0_R - scale * compression_displacement
            die_contact.set_pose(rot_matrix, x_mid_band, y_L, y_R)
            problem.fes[0].update_Dirichlet_boundary_conditions(
                [list(rb_locs), list(rb_vecs), list(rb_vals)])
            # the active set at the last converged state, for the thermal
            # mask and the driver's bookkeeping (the forces themselves are
            # re-evaluated inside Newton)
            left_inds, right_inds = die_contact.active_sets(sol_u_np)
            contact_mask = onp.zeros((len(pts), 1), dtype=onp.float64)
            contact_mask[left_inds, 0] = 1.0
            contact_mask[right_inds, 0] = 1.0
            problem.contact_mask_nodes = np.array(contact_mask)
            problem.contact_left_inds = onp.asarray(left_inds)
            problem.contact_right_inds = onp.asarray(right_inds)
            print(f"  [BC step {step_count}] scale={scale:.4f} contact left={len(left_inds)} "
                  f"right={len(right_inds)} (y_L={y_L:.4f}, y_R={y_R:.4f}) [force-based]")

        if n_cand > 0:
            mid0 = 0.5 * (y_rot_min0 + y_rot_max0)
            yc0 = xdef0_cand_rot[:, 1]
            conv_left_inds = np.array(candidate_inds[yc0 <= mid0], dtype=int)
            conv_right_inds = np.array(candidate_inds[yc0 > mid0], dtype=int)
        else:
            conv_left_inds = np.array([], dtype=int)
            conv_right_inds = np.array([], dtype=int)

        def left_reach_nodes_f(p, ind):
            return np.isin(ind, conv_left_inds)

        def right_reach_nodes_f(p, ind):
            return np.isin(ind, conv_right_inds)

        return (dirichlet_bc_info_u, None, [left_reach_nodes_f, right_reach_nodes_f],
                bc_update_force, surface_inds)

    # ----- initial placeholder BC (scale = 0 on current deformed config) -----
    left0, right0, uL0, uR0 = compute_contact(current_sol_u_np, 0.0)
    _l, _v = _contact_bc(left0, right0, uL0, uR0)
    location_fns_u = rb_locs + _l
    value_fns = rb_vals + _v
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

        _l, _v = _contact_bc(left_inds, right_inds, u_left, u_right)
        loc_fns = rb_locs + _l
        val_fns = rb_vals + _v
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

        # [force] Publish the active contact set so the driver can read the
        # press load off the reaction at the constrained dofs.  The hot runs
        # never needed this; the cold 6063 experiment records a 633 Hz force
        # trace, which is the strongest signal available for identifying the
        # hardening law.
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
