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
from applications.Agility_Forge.lib.boundary_conditions import build_cylinder_press_bcs, refresh_problem_surface_integrals


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
def run_thermo_mech_cylinder_press_multi_hits():
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
    vtk_dir = os.path.join(data_dir, "vtk_tm_press_T_convect_tet_two_hits")
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
    for pattern in ("hit_*_step_data.png", "hit_*_step_data_scales.txt"):
        for item in glob.glob(os.path.join(data_dir, pattern)):
            try:
                os.remove(item)
            except OSError as e:
                print(f"Warning: Could not remove {item}: {e}")

    # Load initial mesh
    mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
    open3d_to_json(mesh_0, json_mesh_path)
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

    for hit_idx, hit in enumerate(hits):
        print("\n" + "=" * 80)
        print(f"HIT {hit_idx + 1}/{len(hits)}")
        print("=" * 80)
        print(f"Hit params: x_band=[{hit.x_min_band}, {hit.x_max_band}]*H, "
              f"disp={hit.compression_displacement}, rot_x={hit.rotation_euler_x}")

        current_sol_u_for_bc = current_sol_u if hit_idx > 0 else None
        dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
            build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                     current_sol_u=current_sol_u_for_bc, T_linear_fn=T_linear_x)

        problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
        problem.location_fns = location_fns_convect
        refresh_problem_surface_integrals(problem)

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
            # Instead, ramp T_old linearly from the hit-1 cold end-state to the
            # reheated profile over several increments, doing one mechanical
            # re-equilibration (scale = 0, platen fixed) at each increment.  Each
            # step then carries only a small thermal mismatch and converges.
            T_cold = onp.array(current_sol_dT)        # hit-1 cooled end-state
            T_hot = onp.array(sol_dT_initial)         # reheated target profile
            n_ramp = 10
            relax_dt = 1e-4
            print(f"\nRunning thermal-state relaxation: {n_ramp}-step temperature "
                  f"ramp (cold -> reheated), scale=0, no contact motion...")

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
                    surface_inds=surface_inds,
                )
                # Seed state AND reset extrapolation history (avoids the
                # doubled-initial-guess bug on the first sub-step).
                relax_stepper.seed_state(current_sol_u, T_blend, current_int_vars)

                ok_relax = relax_stepper.run(
                    bc_update_fn=bc_update_fn,
                    bc_params_fn=lambda step, scale: (step, 0.0),  # force scale=0
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

            # =================================================================
            # [DIAG-B] Compare the last relax step's *solved* temperature
            # against the target sol_dT_initial that hit-N is about to use
            # as T_old.  A large delta => hit-N's first step sees a non-
            # equilibrium T_old (dt-independent thermal-stress jump).
            # =================================================================
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
                      f"max|ΔT| = {linf:.4f} C")
            except Exception as _diag_e:
                print(f"[DIAG-B] temperature-mismatch check failed: {_diag_e}")

            # Finalize: mechanical state now equilibrated under the reheated profile
            current_sol_dT = sol_dT_initial
            print("Relaxation complete: mechanical state re-equilibrated under reheated T_old.")

            # Rebuild the contact BCs on the POST-relaxation surface.
            # The BCs above were built from the hit-(N-1) end displacement and the
            # platen reference planes (y_rot_min0/max0) were frozen at that cold
            # geometry.  The relaxation reheats and re-equilibrates the body
            # (thermal expansion), so by the time this hit starts the actual
            # surface differs from that frozen reference.  Rebuilding here aligns
            # the platen planes with the real starting geometry, avoiding a
            # dt-independent contact jump on the first step.
            dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
                build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                         current_sol_u=current_sol_u, T_linear_fn=T_linear_x)
            problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
            problem.location_fns = location_fns_convect
            refresh_problem_surface_integrals(problem)

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

        print(f"\nHit {hit_idx + 1} completed successfully")
        print(f"  Converged in {stepper.step_count} steps")
        print(f"  Wall time: {stepper.total_wall:.2f}s")
        print(f"  Internal vars shape: F={current_int_vars[0].shape}, alpha={current_int_vars[2].shape}")

        plot_path = os.path.join(data_dir, f"hit_{hit_idx + 1}_step_data.png")
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
    run_thermo_mech_cylinder_press_multi_hits()
