"""Main driver for multi-hit thermo-mechanical forging simulation.

Version 7 — Modular refactor of jax_forge_hits.py (Version 6).
All constitutive, time-stepping, and BC logic factored into separate modules.

Usage:
    python -m applications.Agility_Forge.main
    # or
    python main.py          (when CWD is Agility_Forge/)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    hit_1 = Hit(
        x_min_band=0.5,
        x_max_band=0.7,
        compression_displacement=1.63,
        rotation_euler_x=0.0,
        total_time=0.8,
    )

    hit_2 = Hit(
        x_min_band=0.65,
        x_max_band=0.85,
        compression_displacement=1.63,
        rotation_euler_x=-15.0,
        total_time=0.8,
    )

    hit_3 = Hit(
        x_min_band=0.55,
        x_max_band=0.75,
        compression_displacement=1.83,
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
            current_sol_dT = sol_dT_initial

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

        stepper.sol_list = [current_sol_u, current_sol_dT]
        stepper.sol_u, stepper.sol_dT = [current_sol_u, current_sol_dT]
        stepper.int_vars = current_int_vars

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
