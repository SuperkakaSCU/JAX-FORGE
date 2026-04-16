"""Main driver for multi-hit forging data generation (Version 8).

Reads forging.yml, expands band_positions × depths × rotations into
hit sequences (like cogging), runs multiple series with noise, and
stores (input, hit_params) → output training pairs in SQLite.

Usage:
    python -m applications.Agility_Forge_data.main
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("[Warning] open3d not available, mesh conversion functions will be disabled")

import json
import math
import shutil
import glob
import time
import yaml
import numpy as onp
import jax
import jax.numpy as np

from jax import config
config.update("jax_enable_x64", True)

_ = jax.devices()

from jax_fem_checkpoint.generate_mesh import get_meshio_cell_type, Mesh

from applications.Agility_Forge_data.mesh_container import MeshContainer
from applications.Agility_Forge_data.hit_config import Hit
from applications.Agility_Forge_data.lib.constitutive import ThermalMechanical
from applications.Agility_Forge_data.lib.time_stepper import AutomaticTimeStepperTM
from applications.Agility_Forge_data.lib.boundary_conditions import build_cylinder_press_bcs, refresh_problem_surface_integrals
from applications.Agility_Forge_data.data_storage import ForgingDBMS


# =============================================================================
# Mesh utilities (same as V7)
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
# Hit expansion: band_positions × depths × rotations → list[Hit]
# =============================================================================
def expand_hits(sequence_config, scan_config, H, noise_config=None, add_noise=False, rng=None):
    """Expand a sequence config into a list of Hit objects.

    Mimics cogging's three-layer loop:
        for band_center in positions:
            for d in depths:
                for r in rotations:
                    Hit(...)

    Parameters
    ----------
    sequence_config : dict
        From YAML, e.g. {'position_configs': [{'depths': [...], 'rotations': [...]}]}
    scan_config : dict
        From YAML, e.g. {'band_width': 0.2, 'hot_zone': [0.3, 0.9], 'overlap': 0.3, 'v_platen': 2.04}
    H : float
        Mesh height (max x coordinate).
    noise_config : dict or None
        From YAML, e.g. {'band_center_scale': 0.05, 'depth_scale': 0.10, 'rotation_width': 5.0}
    add_noise : bool
        Whether to apply noise.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    hits : list[Hit]
        Ordered list of Hit objects for this series.
    """
    if rng is None:
        rng = onp.random.default_rng()

    band_width = scan_config["band_width"]
    hot_zone_min, hot_zone_max = scan_config["hot_zone"]
    overlap = scan_config["overlap"]
    v_platen = scan_config["v_platen"]

    # Compute band center positions (like cogging's translations_queue)
    effective_step = band_width * (1.0 - overlap)
    forgeable_range = hot_zone_max - hot_zone_min
    n_positions = max(1, math.ceil(forgeable_range / effective_step))

    # Centers: from (hot_zone_min + band_width/2) to (hot_zone_max - band_width/2)
    center_min = hot_zone_min + band_width / 2.0
    center_max = hot_zone_max - band_width / 2.0
    if n_positions == 1:
        band_centers = [0.5 * (center_min + center_max)]
    else:
        band_centers = list(onp.linspace(center_min, center_max, n_positions))

    hits = []
    position_configs = sequence_config["position_configs"]

    for center in band_centers:
        # Apply noise to band center
        noised_center = center
        if add_noise and noise_config:
            center_noise = rng.normal(0.0, noise_config["band_center_scale"])
            noised_center = center + center_noise
            # Clamp to hot zone
            noised_center = max(hot_zone_min + band_width / 2.0,
                                min(hot_zone_max - band_width / 2.0, noised_center))

        x_min_band = noised_center - band_width / 2.0
        x_max_band = noised_center + band_width / 2.0

        for pc in position_configs:
            for d in pc["depths"]:
                # Apply noise to depth
                noised_d = d
                if add_noise and noise_config:
                    d_noise = rng.normal(0.0, noise_config["depth_scale"] * abs(d))
                    noised_d = d + d_noise
                    noised_d = max(0.1, noised_d)  # safety floor

                for r in pc["rotations"]:
                    # Apply noise to rotation
                    noised_r = r
                    if add_noise and noise_config:
                        r_noise = rng.normal(0.0, noise_config["rotation_width"])
                        noised_r = r + r_noise

                    total_time = noised_d / v_platen

                    hit = Hit(
                        x_min_band=float(x_min_band),
                        x_max_band=float(x_max_band),
                        compression_displacement=float(noised_d),
                        rotation_euler_x=float(noised_r),
                        total_time=float(total_time),
                    )
                    hits.append(hit)

    return hits


# =============================================================================
# Run one series (multi-hit forging on a single workpiece)
# =============================================================================
def run_forging_series(hits, mesh, problem, sol_u0, sol_dT_initial, H, R,
                       T_linear_fn, db, series_id, save_vtk=False, vtk_dir=None):
    """Execute a single series of hits on the same workpiece.

    State carried across hits: F, alpha (plasticity history).
    State reset between hits: temperature (cool-down assumption).
    Displacement continuity: ramp formula from V6/V7.

    Parameters
    ----------
    hits : list[Hit]
    mesh : jax_fem Mesh
    problem : ThermalMechanical
    sol_u0 : array (N, 3)
    sol_dT_initial : array (N, 1)
    H, R : float
    T_linear_fn : callable
    db : ForgingDBMS
    series_id : str
    save_vtk : bool
    vtk_dir : str or None

    Returns
    -------
    success : bool
    """
    current_sol_u = sol_u0
    current_sol_dT = np.array(sol_dT_initial)
    current_int_vars = problem.internal_vars

    for hit_idx, hit in enumerate(hits):
        step_number = hit_idx + 1
        print(f"\n{'='*80}")
        print(f"  Series {series_id[:8]}... | HIT {step_number}/{len(hits)}")
        print(f"{'='*80}")
        print(f"  Params: band=[{hit.x_min_band:.3f}, {hit.x_max_band:.3f}]*H, "
              f"disp={hit.compression_displacement:.3f}, "
              f"rot={hit.rotation_euler_x:.1f}°, "
              f"time={hit.total_time:.3f}s")

        # Record input state BEFORE the hit
        input_u = current_sol_u
        input_T = current_sol_dT

        # Build BCs for this hit
        current_sol_u_for_bc = current_sol_u if hit_idx > 0 else None
        try:
            dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, \
                bc_update_fn, surface_inds = \
                build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                         current_sol_u=current_sol_u_for_bc,
                                         T_linear_fn=T_linear_fn)
        except Exception as e:
            print(f"  [ERROR] BC construction failed: {e}")
            db.fail_series(series_id, step_number)
            return False

        problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
        problem.location_fns = location_fns_convect
        refresh_problem_surface_integrals(problem)

        # Reset temperature between hits
        if hit_idx > 0:
            print("  Resetting temperature to initial profile (cool-down)...")
            current_sol_dT = np.array(sol_dT_initial)

        # Create stepper
        # Later hits often have sharper BC/contact changes (new band + carried
        # plastic history). Use a gentler first dt to improve Step-1 robustness.
        initial_dt = 1e-3 if hit_idx == 0 else 2e-4
        max_dt = 0.05 if hit_idx == 0 else 0.02
        stepper = AutomaticTimeStepperTM(
            problem,
            total_time=hit.total_time,
            initial_dt=initial_dt,
            min_dt=1e-6,
            max_dt=max_dt,
            max_retries=10,
            increase_factor=1.3,
            decrease_factor=0.5,
            line_search_after=0,
            cool_factor=0.8,
            surface_inds=surface_inds,
        )

        # IMPORTANT: initialize stepper state/history from carried-over hit state.
        # Otherwise sol_hist remains zeros from constructor and first-step
        # extrapolation becomes ~2*current_state, which often destabilizes
        # convergence at Hit k (k>=2), Step 1.
        seed_u = np.array(current_sol_u)
        seed_T = np.array(current_sol_dT)

        stepper.sol_list = [seed_u, seed_T]
        stepper.sol_u, stepper.sol_dT = seed_u, seed_T
        stepper.sol_hist = [[seed_u, seed_T], [seed_u, seed_T]]
        stepper.dt_hist = [stepper.dt, stepper.dt]
        stepper.int_vars = current_int_vars

        # Run solver
        rho_ini = np.array([1.0, 1.0, 1.0, 1.0])
        hit_vtk_dir = None
        if save_vtk and vtk_dir:
            hit_vtk_dir = os.path.join(vtk_dir, f"hit_{step_number}")
            os.makedirs(hit_vtk_dir, exist_ok=True)

        hit_start = time.time()
        try:
            ok = stepper.run(
                bc_update_fn=bc_update_fn,
                bc_params_fn=lambda step, scale: (step, scale),
                rho_ini=rho_ini,
                vtk_dir=hit_vtk_dir,
                save_every=1,
            )
        except Exception as e:
            print(f"  [ERROR] Solver raised exception: {e}")
            ok = False

        hit_wall = time.time() - hit_start

        if not ok:
            print(f"  [FAIL] Hit {step_number} solver did not converge!")
            db.fail_series(series_id, step_number)
            return False

        # Extract result state
        result_u = stepper.sol_u
        result_T = stepper.sol_dT
        current_int_vars = stepper.int_vars

        # Record to DB
        db.record_hit(
            series_id=series_id,
            step_number=step_number,
            hit=hit,
            input_u=input_u,
            input_T=input_T,
            result_u=result_u,
            result_T=result_T,
            solver_steps=stepper.step_count,
            wall_time=hit_wall,
        )

        # Carry forward for next hit
        current_sol_u = result_u
        current_sol_dT = result_T

        print(f"  Hit {step_number} OK — {stepper.step_count} steps, {hit_wall:.1f}s")

    # All hits completed
    db.complete_series(series_id)
    return True


# =============================================================================
# Main entry point
# =============================================================================
def main():
    """Read forging.yml, expand hits, run series loop, record to DB."""

    # --- Paths ---
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, "data")
    config_path = os.path.join(crt_file_path, "configs", "forging.yml")
    db_path = os.path.join(data_dir, "forge_database.db")
    msh_dir = os.path.join(data_dir, "msh/jax_forge")
    assets_dir = os.path.join(msh_dir, "assets")
    stock_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.obj")
    json_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.json")

    os.makedirs(data_dir, exist_ok=True)

    # --- Load config ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    scan_config = cfg["scan"]
    noise_config = cfg["noise"]
    run_config = cfg["run"]

    n_series = run_config["n_series"]
    add_noise = run_config["add_noise"]
    save_vtk = run_config["save_vtk"]
    seed = run_config.get("random_seed", 42)

    # Sequence types: everything in YAML that has 'position_configs'
    sequence_names = [k for k in cfg if isinstance(cfg[k], dict)
                      and "position_configs" in cfg[k]]
    print(f"Found {len(sequence_names)} sequence types: {sequence_names}")

    # --- Load mesh ---
    print("\nLoading mesh...")
    mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
    open3d_to_json(mesh_0, json_mesh_path)
    stock_mesh = MeshContainer.from_json(json_mesh_path)
    ele_type = "TET4"
    mesh = convert_meshio_to_jaxfem(stock_mesh.vtk, ele_type=ele_type)

    H = float(np.max(mesh.points[:, 0]))
    R_y = float(np.max(mesh.points[:, 1]))
    R = R_y
    print(f"Mesh: H={H:.2f}, R={R:.2f}, nodes={len(mesh.points)}")

    # --- Initial temperature profile ---
    T_top = 1096.0
    T_bot = 676.0
    x_min = float(np.min(mesh.points[:, 0]))
    x_top = 72.0

    def T_linear_x(point):
        x = point[0]
        if x > x_top:
            return T_top
        xc = np.clip(x, x_min, x_top)
        return T_bot + (T_top - T_bot) * ((xc - x_min) / (x_top - x_min))

    pts = onp.array(mesh.points)
    sol_dT_initial = onp.zeros((len(pts), 1), dtype=onp.float64)
    for i in range(len(pts)):
        sol_dT_initial[i, 0] = float(T_linear_x(np.array(pts[i])))

    sol_u0 = np.zeros((len(mesh.points), 3))

    # --- Build problem once ---
    print("\nBuilding coupled thermo-mechanical problem...")

    def _no_location(p):
        return False

    problem = ThermalMechanical(
        mesh=[mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        gauss_order=[2, 2],
        dirichlet_bc_info=[None, None],
        location_fns=[_no_location, _no_location],
    )

    # --- Open DB ---
    db = ForgingDBMS(db_path, new=True)
    print(f"Database created: {db_path}")

    # --- Main loop: sequence_types × n_series ---
    rng = onp.random.default_rng(seed)
    total_series = 0
    total_hits_completed = 0
    total_failed = 0
    t_global = time.time()

    for seq_name in sequence_names:
        seq_cfg = cfg[seq_name]
        print(f"\n{'#'*80}")
        print(f"# Sequence type: {seq_name}")
        print(f"#   description: {seq_cfg.get('description', '')}")
        print(f"#   n_series: {n_series}")
        print(f"{'#'*80}")

        for series_idx in range(n_series):
            # Expand hits with noise
            hits = expand_hits(
                seq_cfg, scan_config, H,
                noise_config=noise_config,
                add_noise=add_noise,
                rng=rng,
            )
            n_hits = len(hits)

            print(f"\n--- {seq_name} series {series_idx+1}/{n_series} | {n_hits} hits ---")

            # Start series in DB
            series_id = db.start_series(seq_name, n_hits)

            # VTK directory
            vtk_dir = None
            if save_vtk:
                vtk_dir = os.path.join(data_dir,
                                       f"vtk_{seq_name}_s{series_idx:04d}")
                os.makedirs(vtk_dir, exist_ok=True)

            # Reset problem internal vars to initial state for each new series
            problem.custom_init()

            # Run the series
            ok = run_forging_series(
                hits=hits,
                mesh=mesh,
                problem=problem,
                sol_u0=sol_u0,
                sol_dT_initial=sol_dT_initial,
                H=H,
                R=R,
                T_linear_fn=T_linear_x,
                db=db,
                series_id=series_id,
                save_vtk=save_vtk,
                vtk_dir=vtk_dir,
            )

            total_series += 1
            if ok:
                total_hits_completed += n_hits
                print(f"  Series {series_id[:8]}... COMPLETED ({n_hits} hits)")
            else:
                total_failed += 1
                completed_hits = db.get_hit_count(series_id)
                total_hits_completed += completed_hits
                print(f"  Series {series_id[:8]}... FAILED "
                      f"({completed_hits}/{n_hits} hits saved)")

    # --- Summary ---
    elapsed = time.time() - t_global
    print(f"\n{'='*80}")
    print(f"DATA GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total series: {total_series} (failed: {total_failed})")
    print(f"  Total hits recorded: {total_hits_completed}")
    print(f"  Total wall time: {elapsed/3600:.2f} hours")
    print(f"  Database: {db_path}")
    db.summary()
    db.close()


if __name__ == "__main__":
    main()
