"""Main driver for multi-hit thermo-mechanical forging simulation.

Version 8 — Multi-hit rotational forging with robust hit-to-hit handoff.
Builds on the Version 7 modular refactor (constitutive / time-stepping / BC).

Key Version 8 changes:
  - 4-hit rotational schedule (0/90/180/270 deg, 3.0 mm press, 1.2 s each).
  - Updated-Lagrangian dynamic contact (Route B): the contact set and the
    prescribed displacements are re-detected every load step from the current
    deformed surface, replacing the static reference-config node selection.
  - Thermal-state relaxation between hits: T_old is ramped from the cooled
    end-state to the reheated profile over 10 increments, each with a mechanical
    re-equilibration, removing the dt-independent thermal-stress mismatch that
    otherwise stalls Newton on the first step of a reheated hit.
  - Robust state handoff: AutomaticTimeStepperTM.seed_state() resets the
    extrapolation history (fixing the doubled first-step initial guess), and the
    contact BCs are rebuilt on the post-relaxation surface.
  - Numerics: high-temperature modulus floor (prevents a near-singular tangent)
    and a cached JIT internal-variable update (removes per-step recompilation).

Usage:
    python -m applications.Agility_Forge.main
    # or
    python main.py          (when CWD is Agility_Forge/)
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("[Warning] open3d not available, mesh conversion functions will be disabled")

import json
import shutil
import glob
import argparse
import numpy as onp
import jax
import jax.numpy as np

from jax import config
config.update("jax_enable_x64", True)

_ = jax.devices()

from jax_fem_checkpoint.generate_mesh import get_meshio_cell_type, Mesh

from applications.Agility_Forge.mesh_container import MeshContainer

from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.lib.constitutive import ThermalMechanical
from applications.Agility_Forge.lib.time_stepper import AutomaticTimeStepperTM
from applications.Agility_Forge.lib.boundary_conditions import (
    build_cylinder_press_bcs, build_unload_bcs, refresh_problem_surface_integrals)


# =============================================================================
# Mesh utilities
# =============================================================================
def open3d_to_json(o3d_mesh, json_path):
    """Convert ForgeDataset mesh (Open3D) to the JSON format expected by jax-forge."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    data = {
        "Vertices": vertices.flatten().tolist(),
        "Triangles": faces.flatten().tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def convert_meshio_to_jaxfem(meshio_mesh, ele_type="TET4"):
    """Convert a meshio mesh to a jax-fem Mesh object."""
    print("Converting to jax-fem mesh...")
    cell_type = get_meshio_cell_type(ele_type)
    return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# =============================================================================
# Main thermo-mechanical driver for multi-hit forging simulation
# =============================================================================
# =============================================================================
# Force-controlled unloading between hits
# =============================================================================
def run_unloading(problem, mesh, sol_u, sol_dT, int_vars, reaction_u,
                  axial_constraint="face", n_steps=10, total_time=1e-3,
                  rho_ini=None, T_old_last=None, dt_last=None, balance="self", vtk_dir=None,
                  mode="peel", rot_deg=0.0):
    """Release the previous hit's die reaction gradually.

    The contact rows are replaced by the external force ``lambda * R`` on the
    former contact nodes (``build_unload_bcs``) and ``lambda`` is ramped from
    1 to 0 by the stepper's load schedule, with retries on failure.  Before
    stepping, the residual at ``lambda = 1`` is evaluated as a consistency
    check: with the last step's own ``T_old``/``dt`` the released state must
    reproduce the captured reaction, so the residual at the released nodes
    is round-off; with the end-of-hit temperature the difference is the
    thermal-stress change over one step, absorbed by the first increment.

    Returns ``(ok, sol_u, sol_dT, int_vars, bc_update_fn)``; the last item
    installs the contact-free configuration (use it with ``scale = 1`` for a
    subsequent contact-free phase such as the reheat ramp).
    """
    if rho_ini is None:
        rho_ini = np.array([1.0, 1.0, 1.0, 1.0])
    left = getattr(problem, "contact_left_inds", onp.zeros((0,), dtype=int))
    right = getattr(problem, "contact_right_inds", onp.zeros((0,), dtype=int))
    contact = onp.concatenate([onp.asarray(left, dtype=int), onp.asarray(right, dtype=int)])
    dbc_u, bc_update = build_unload_bcs(mesh, reaction_u, contact, axial_constraint,
                                        sol_u=sol_u, balance=balance, mode=mode,
                                        left_inds=left, right_inds=right, rot_deg=rot_deg)
    problem.dirichlet_bc_info = [dbc_u, None]

    dt0 = float(total_time) / int(n_steps)
    pts = onp.asarray(mesh.points)
    end_face = onp.abs(pts[:, 0] - pts[:, 0].min()) <= 1e-5
    R_rel = float(onp.linalg.norm(onp.asarray(reaction_u)[contact])) if len(contact) else 0.0
    bc_update(problem, sol_u, 0, 0.0)
    checks = [("end-of-hit T_old", sol_dT, dt0)] if mode == "force" else []
    if T_old_last is not None and mode == "force":
        checks.insert(0, ("last step's T_old/dt", T_old_last, float(dt_last or dt0)))
    for label, T_chk, dt_chk in checks:
        problem.set_timestep(dt_chk)
        problem.set_params([int_vars, 0.0, sol_u, rho_ini, T_chk])
        res_u = onp.asarray(problem.compute_residual([sol_u, sol_dT])[0])
        r_contact = float(onp.linalg.norm(res_u[contact])) if len(contact) else 0.0
        r_free = float(onp.linalg.norm(res_u[~end_face]))
        print(f"  [unload] lambda=1 check ({label}): |r| at the {len(contact)} released "
              f"nodes = {r_contact:.3e} N, over all non-end-face nodes = {r_free:.3e} N, "
              f"released |R| = {R_rel:.3e} N")

    stepper = AutomaticTimeStepperTM(
        problem, total_time=float(total_time), initial_dt=dt0,
        min_dt=dt0 / 2 ** 8, max_dt=dt0, max_retries=10,
        increase_factor=1.0, decrease_factor=0.5, line_search_after=0,
        cool_factor=0.0, surface_inds=None)
    stepper.seed_state(np.array(sol_u), np.array(sol_dT), int_vars)
    ok = stepper.run(bc_update_fn=bc_update,
                     bc_params_fn=lambda step, scale: (step, scale),
                     rho_ini=rho_ini, vtk_dir=vtk_dir, save_every=1)
    if ok:
        du = onp.asarray(stepper.sol_u) - onp.asarray(sol_u)
        print(f"  [unload] released in {stepper.step_count} steps; Newton iterations "
              f"per step {stepper.iters_hist}; springback |du|_inf = "
              f"{float(onp.abs(du).max()):.4f} mm")
    else:
        print("  [unload] FAILED")
    problem.f_ext_nodes = None
    return ok, stepper.sol_u, stepper.sol_dT, stepper.int_vars, bc_update


# =============================================================================
# Options
# =============================================================================
class ModelOptions:
    """Defaults for a programmatic call (``args=None``)."""
    unload = "peel"
    unload_steps = 10
    unload_time = 1e-3
    unload_balance = "self"
    axial_constraint = "face"
    tag = ""
    n_hits = None
    schedule = "three"
    unload_vtk = False


def add_model_options(p):
    """Between-hit release, rigid-body and run options.

    The model corrections of November 2026 (README.md, section 6) are built in;
    their previous behaviour is no longer selectable.
    """
    p.add_argument("--unload", choices=["peel", "force"], default="peel",
                   help="between hits: 'peel' frees the previous hit's contact "
                        "nodes progressively, outermost first, pinning the rest "
                        "where the hit left them (displacement control); 'force' "
                        "ramps the die reaction to zero as an external force "
                        "(load control: unstable on slender hot bars, see "
                        "build_unload_bcs)")
    p.add_argument("--unload-steps", type=int, default=10, dest="unload_steps",
                   help="number of release increments (lambda 1 -> 0)")
    p.add_argument("--unload-time", type=float, default=1e-3, dest="unload_time",
                   help="pseudo-time spanned by the release, s (thermal step)")
    p.add_argument("--unload-balance", choices=["self", "raw"], default="self",
                   dest="unload_balance",
                   help="'force' release only: release only the self-equilibrated "
                        "part of the die reaction ('self'), or the reaction as "
                        "captured ('raw', which loads the free bar with the platen "
                        "imbalance)")
    p.add_argument("--axial-constraint", choices=["face", "face_rx", "minimal3"],
                   default="face", dest="axial_constraint",
                   help="rigid-body suppression (see build_cylinder_press_bcs)")
    p.add_argument("--schedule", choices=["three", "four"], default="three",
                   help="scripted hits: 'three' = 0 / -15 / 30 deg, 2.5-3 mm; "
                        "'four' = the earlier rotational schedule, 0 / 90 / 180 / "
                        "270 deg, 3 mm")
    p.add_argument("--n-hits", type=int, default=None, dest="n_hits",
                   help="run only the first N of the scripted hits")
    p.add_argument("--unload-vtk", action="store_true", dest="unload_vtk",
                   help="write a VTK frame per release increment (diagnostics)")
    p.add_argument("--tag", default="",
                   help="suffix for the VTK directory and the per-hit plots, so "
                        "several runs can coexist under data/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Agility_Forge -- multi-hit thermo-mechanical forging demo (cylinder).")
    add_model_options(p)
    return p.parse_args()


def run_thermo_mech_cylinder_press_multi_hits(args=None):
    """
    Run a multi-hit forging simulation:
    - Hit 1 at position x_band=[0.5, 0.7]*H with rotation 0°
    - Cool to initial temperature profile
    - Hit 2 at position x_band=[0.65, 0.85]*H with rotation -15°
    - Hit 3 at position x_band=[0.55, 0.75]*H with rotation 30°

    Key: F and alpha (plasticity history) carried forward; T reset;
         displacement continuity via ramp formula.
    """
    # Directories
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, "data")
    msh_dir = os.path.join(data_dir, "msh/jax_forge")
    if args is None:
        args = ModelOptions()
    tag = args.tag or ""
    vtk_dir = os.path.join(data_dir, "vtk_tm_press_T_convect_tet_two_hits" + tag)
    print(f"Options: unload={args.unload} ({args.unload_steps} steps, "
          f"{args.unload_time} s), axial_constraint={args.axial_constraint}")
    assets_dir = os.path.join(msh_dir, "assets")
    stock_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.obj")
    json_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.json")

    os.makedirs(vtk_dir, exist_ok=True)

    # clean VTK directory
    for item in glob.glob(os.path.join(vtk_dir, "*")):
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
        except (OSError, Exception) as e:
            print(f"Warning: Could not remove {item}: {e}")

    # clean per-hit step/scale plots from previous runs (avoid stale duplicates,
    # e.g. a hit_4 plot left over when this run only does 3 hits)
    for pattern in (f"hit_*{tag}_step_data.png", f"hit_*{tag}_step_data_scales.txt"):
        for item in glob.glob(os.path.join(data_dir, pattern)):
            try:
                os.remove(item)
            except OSError as e:
                print(f"Warning: Could not remove {item}: {e}")

    # Load initial mesh
    if o3d is not None:
        mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
        open3d_to_json(mesh_0, json_mesh_path)
    elif not os.path.exists(json_mesh_path):
        raise SystemExit(f"open3d is not installed and {json_mesh_path} does not exist")
    else:
        print(f"[mesh] open3d not available; using the existing {json_mesh_path}")
    stock_mesh = MeshContainer.from_json(json_mesh_path)
    ele_type = "TET4"
    mesh = convert_meshio_to_jaxfem(stock_mesh.vtk, ele_type=ele_type)

    # Mesh dimensions
    H = np.max(mesh.points[:, 0])
    R_y = np.max(mesh.points[:, 1])
    R_z = np.max(mesh.points[:, 2])
    R = R_y

    print("H={0}, R_y={1}, R_z={2}".format(H, R_y, R_z))

    # === Initial temperature profile ===
    T_top = 1096.0  # °C
    T_bot = 676.0   # °C
    x_min = np.min(mesh.points[:, 0])
    x_top = 72.0

    def T_linear_x(point):
        """Piecewise temperature profile along x."""
        x = point[0]
        if x > x_top:
            return T_top
        xc = np.clip(x, x_min, x_top)
        return T_bot + (T_top - T_bot) * ((xc - x_min) / (x_top - x_min))

    # === Define hits ===
    # hit_1 = Hit(
    #     x_min_band=0.5,
    #     x_max_band=0.7,
    #     compression_displacement=3.0,
    #     rotation_euler_x=0.0,
    #     total_time=1.2,
    # )

    # hit_2 = Hit(
    #     x_min_band=0.51,
    #     x_max_band=0.71,
    #     compression_displacement=3.0,
    #     rotation_euler_x=90.0,
    #     total_time=1.2,
    # )

    # hit_3 = Hit(
    #     x_min_band=0.5,
    #     x_max_band=0.7,
    #     compression_displacement=3.0,
    #     rotation_euler_x=180.0,
    #     total_time=1.2,
    # )


    # hit_4 = Hit(
    #     x_min_band=0.5,
    #     x_max_band=0.7,
    #     compression_displacement=3.0,
    #     rotation_euler_x=270.0,
    #     total_time=1.2,
    # )

    
    # hits = [hit_1, hit_2, hit_3, hit_4]

    
    # Another set of hits with smaller compression and different rotations, for testing:
    hit_1 = Hit(
        x_min_band=0.5,
        x_max_band=0.7,
        compression_displacement=2.5,
        rotation_euler_x=0.0,
        total_time=0.8,
    )

    hit_2 = Hit(
        x_min_band=0.65,
        x_max_band=0.85,
        compression_displacement=2.5,
        rotation_euler_x=-15.0,
        total_time=0.8,
    )

    hit_3 = Hit(
        x_min_band=0.55,
        x_max_band=0.75,
        compression_displacement=3.0,
        rotation_euler_x=30.0,
        total_time=1.2,
    )

    hits = [hit_1, hit_2, hit_3]
    if getattr(args, "schedule", "three") == "four":
        # The earlier rotational schedule (commented out above): 3 mm, 1.2 s,
        # 0 / 90 / 180 / 270 deg.
        bands = [(0.5, 0.7), (0.51, 0.71), (0.5, 0.7), (0.5, 0.7)]
        hits = [Hit(x_min_band=b[0], x_max_band=b[1], compression_displacement=3.0,
                    rotation_euler_x=rot, total_time=1.2)
                for b, rot in zip(bands, (0.0, 90.0, 180.0, 270.0))]
    if getattr(args, "n_hits", None):
        hits = hits[:args.n_hits]


    # === Build initial temperature field ===
    pts = onp.array(mesh.points)
    sol_dT_initial = onp.zeros((len(pts), 1), dtype=onp.float64)
    for i in range(len(pts)):
        sol_dT_initial[i, 0] = float(T_linear_x(np.array(pts[i])))

    sol_dT0 = np.array(sol_dT_initial)
    sol_u0 = np.zeros((len(mesh.points), 3))

    # === Build problem once ===
    print("\n" + "=" * 80)
    print("Building coupled thermo-mechanical problem...")
    print("=" * 80)

    def _no_location(p):
        return False

    problem = ThermalMechanical(
        mesh=[mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        gauss_order=[2, 2],
        dirichlet_bc_info=[None, None],
        location_fns=[_no_location, _no_location]
    )

    # === Multi-hit loop ===
    current_sol_u = sol_u0
    current_sol_dT = sol_dT0
    current_int_vars = problem.internal_vars
    last_res_u = None            # die reaction at the end of the previous hit
    last_T_old, last_dt = None, None
    prev_rot = 0.0               # rotation of the previous hit, for the peel order

    def build_hit_bcs(hit, u_for_bc):
        out = build_cylinder_press_bcs(
            mesh, R=R, H=H, hit=hit, current_sol_u=u_for_bc, T_linear_fn=T_linear_x,
            axial_constraint=args.axial_constraint)
        problem.dirichlet_bc_info = [out[0], out[1]]
        problem.location_fns = out[2]
        refresh_problem_surface_integrals(problem)
        return out

    for hit_idx, hit in enumerate(hits):
        print("\n" + "=" * 80)
        print(f"HIT {hit_idx + 1}/{len(hits)}")
        print("=" * 80)
        print(f"Hit params: x_band=[{hit.x_min_band}, {hit.x_max_band}]*H, "
              f"disp={hit.compression_displacement}, rot_x={hit.rotation_euler_x}")

        # ------------------------------------------------------------------
        # Between-hit transition
        # ------------------------------------------------------------------
        # The previous hit's contact is released gradually (run_unloading,
        # unload="peel" or "force") with only rigid-body rows installed, the
        # reheat ramp runs contact-free, and the new hit's rows are built
        # once, on the unloaded and reheated surface.  The first hit has no
        # previous contact: its rows are built directly.
        if hit_idx == 0:
            (dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect,
             bc_update_fn, surface_inds) = build_hit_bcs(hit, None)
        else:
            if last_res_u is None:
                raise RuntimeError(f"hit {hit_idx + 1}: no die reaction recorded for "
                                   f"the previous hit, so its contact cannot be released")
            ok_unl, current_sol_u, current_sol_dT, current_int_vars, unload_bc_update = \
                run_unloading(problem, mesh, current_sol_u, current_sol_dT,
                              current_int_vars, last_res_u,
                              axial_constraint=args.axial_constraint,
                              n_steps=args.unload_steps, total_time=args.unload_time,
                              T_old_last=last_T_old, dt_last=last_dt,
                              balance=args.unload_balance, mode=args.unload,
                              rot_deg=prev_rot,
                              vtk_dir=(os.path.join(vtk_dir, f"unload_{hit_idx}")
                                       if args.unload_vtk else None))
            if not ok_unl:
                print(f"\n[ERROR] Hit {hit_idx + 1}: unloading of the previous hit failed")
                return False
            ramp_bc_update = unload_bc_update
            ramp_params = (lambda step, scale: (step, 1.0))      # lambda = 0: fully released

        # Reset temperature between hits
        if hit_idx > 0:
            print("\nResetting temperature to initial profile (cool-down between hits)...")

            # === Thermal-state relaxation by gradual temperature ramp ===
            # The carried mechanical history (be_old / F_old / alpha) was
            # equilibrated under the PREVIOUS hit's (cooled) temperature.  If we
            # reset the temperature to the reheated profile in one shot, the new
            # T_old makes the thermal-stress term and the (much lower) high-T
            # yield strength jump abruptly, leaving a dt-INDEPENDENT static
            # equilibrium mismatch that Newton cannot bridge in a single solve
            # (it stalls even at dt -> 1e-7).
            #
            # Instead, ramp T_old linearly from the cooled end-state to the
            # reheated profile over several increments, doing one mechanical
            # re-equilibration at each increment.  Each step then carries only
            # a small thermal mismatch and converges.
            T_cold = onp.array(current_sol_dT)        # cooled end-state
            T_hot = onp.array(sol_dT_initial)         # reheated target profile
            n_ramp = 10
            relax_dt = 1e-4
            print(f"\nRunning thermal-state relaxation: {n_ramp}-step temperature "
                  f"ramp (cold -> reheated), contact-free...")

            for k in range(1, n_ramp + 1):
                frac = k / n_ramp
                T_blend = np.array((1.0 - frac) * T_cold + frac * T_hot)
                print(f"  [relax {k}/{n_ramp}] T_old ramp frac={frac:.2f}, "
                      f"T min/max = {float(np.min(T_blend)):.1f}/{float(np.max(T_blend)):.1f} C")

                relax_stepper = AutomaticTimeStepperTM(
                    problem,
                    total_time=relax_dt,
                    initial_dt=relax_dt,
                    min_dt=1e-8,
                    max_dt=relax_dt,
                    max_retries=10,
                    increase_factor=1.0,
                    decrease_factor=0.5,
                    line_search_after=0,
                    cool_factor=0.0,            # no extra cooling during ramp
                    surface_inds=None,
                )
                # Seed state AND reset extrapolation history (avoids the
                # doubled-initial-guess bug on the first sub-step).
                relax_stepper.seed_state(current_sol_u, T_blend, current_int_vars)

                ok_relax = relax_stepper.run(
                    bc_update_fn=ramp_bc_update,
                    bc_params_fn=ramp_params,
                    rho_ini=np.array([1.0, 1.0, 1.0, 1.0]),
                    vtk_dir=None,
                    save_every=1,
                )
                if not ok_relax:
                    print(f"\n[ERROR] Hit {hit_idx + 1} thermal-state relaxation "
                          f"failed at ramp step {k}/{n_ramp}!")
                    print(f"Report: {relax_stepper.step_report[-5:]}")
                    return False

                # Carry the re-equilibrated mechanical state to the next increment.
                # Temperature is prescribed by the ramp, so only u / int_vars advance.
                current_sol_u = relax_stepper.sol_u
                current_int_vars = relax_stepper.int_vars

            # [DIAG-B] last relax-solved T vs the target T_old hit-N will use.
            try:
                T_relaxed = np.array(relax_stepper.sol_dT)
                T_target = np.array(sol_dT_initial)
                dT_field = T_relaxed - T_target
                l2 = float(np.linalg.norm(dT_field))
                linf = float(np.max(np.abs(dT_field)))
                t_rel_mn = float(np.min(T_relaxed)); t_rel_mx = float(np.max(T_relaxed))
                t_tgt_mn = float(np.min(T_target));  t_tgt_mx = float(np.max(T_target))
                print(f"[DIAG-B] last relax-solved T:  min/max = {t_rel_mn:.2f}/{t_rel_mx:.2f} C")
                print(f"[DIAG-B] target sol_dT_initial: min/max = {t_tgt_mn:.2f}/{t_tgt_mx:.2f} C")
                print(f"[DIAG-B] ||T_relaxed - T_target||_2 = {l2:.4e}, "
                      f"max|dT| = {linf:.4f} C")
            except Exception as _diag_e:
                print(f"[DIAG-B] temperature-mismatch check failed: {_diag_e}")

            # Finalize: mechanical state now equilibrated under the reheated profile
            current_sol_dT = sol_dT_initial
            print("Relaxation complete: mechanical state re-equilibrated under reheated T_old.")

            # Build the contact BCs on the POST-relaxation surface so the platen reference planes align with the actual
            # (thermally expanded) starting geometry, avoiding a dt-independent
            # contact jump on the first step.
            (dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect,
             bc_update_fn, surface_inds) = build_hit_bcs(hit, current_sol_u)

        problem.f_ext_nodes = None          # no external force during a hit

        # Create stepper
        stepper = AutomaticTimeStepperTM(
            problem,
            total_time=hit.total_time,
            initial_dt=1e-3,
            min_dt=1e-6,
            max_dt=0.05,
            max_retries=10,
            increase_factor=1.3,
            decrease_factor=0.5,
            line_search_after=0,
            cool_factor=0.8,
            surface_inds=surface_inds,
        )

        # Seed state AND reset extrapolation history (avoids the
        # doubled-initial-guess bug on the first step of each hit).
        stepper.seed_state(current_sol_u, current_sol_dT, current_int_vars)

        # Run
        print(f"\nStarting hit {hit_idx + 1} solver...")
        rho_ini = np.array([1.0, 1.0, 1.0, 1.0])
        ok = stepper.run(
            bc_update_fn=bc_update_fn,
            bc_params_fn=lambda step, scale: (step, scale),
            rho_ini=rho_ini,
            vtk_dir=os.path.join(vtk_dir, f"hit_{hit_idx + 1}"),
            save_every=1,
        )

        if not ok:
            print(f"\n[ERROR] Hit {hit_idx + 1} solver failed to converge!")
            print(f"Report: {stepper.step_report[-5:]}")
            return False

        # Extract state at end of hit
        current_sol_u = stepper.sol_u
        current_sol_dT = stepper.sol_dT
        current_int_vars = stepper.int_vars
        last_res_u = stepper.last_res_u      # die reaction, for the unloading
        last_T_old, last_dt = stepper.last_T_old, stepper.last_dt
        prev_rot = float(hit.rotation_euler_x)

        print(f"\nHit {hit_idx + 1} completed successfully")
        print(f"  Converged in {stepper.step_count} steps")
        print(f"  Wall time: {stepper.total_wall:.2f}s")
        print(f"  Internal vars shape: F={current_int_vars[0].shape}, alpha={current_int_vars[2].shape}")

        plot_path = os.path.join(data_dir, f"hit_{hit_idx + 1}{tag}_step_data.png")
        stepper.plot_step_data(save_path=plot_path)


    # === Summary ===
    print("\n" + "=" * 80)
    print("ALL HITS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total VTK output dir: {vtk_dir}")
    print(f"Final displacement norm: {np.linalg.norm(current_sol_u):.6e}")
    print(f"Final temp min/max: {np.min(current_sol_dT):.2f} / {np.max(current_sol_dT):.2f} °C")

    return True


if __name__ == "__main__":
    run_thermo_mech_cylinder_press_multi_hits(parse_args())
