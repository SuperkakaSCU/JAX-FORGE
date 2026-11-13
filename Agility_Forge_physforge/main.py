"""Main driver for multi-hit forging data generation (Version 8).

Reads forging.yml, expands band_positions × depths × rotations into
hit sequences (like cogging), runs multiple series with noise, and
stores (input, hit_params) → output training pairs in SQLite.

Run from the repository root (the directory that contains both
``applications/`` and ``jax_fem_checkpoint/``), e.g.:
    cd /path/to/jax-fem-0012
    python -m applications.Agility_Forge_physforge.main           # full run, all of forging.yml

Command-line options (each overrides forging.yml; omit to keep the YAML value):
    -s, --sequence NAME [NAME ...]  sequence type(s) to run, e.g. -s one_press
                                    (default: all defined in forging.yml)
    -n, --n-series N                series per sequence (default: run.n_series)
    --seed N                        random seed         (default: run.random_seed)
    --save-vtk / --no-save-vtk      toggle VTK output   (default: run.save_vtk)
    --noise / --no-noise            toggle hit noise    (default: run.add_noise)
    -c, --config PATH               forging.yml path    (default: configs/forging.yml)
    --db PATH                       output SQLite path  (default: data/forge_database.db)

    Available sequences: one_press, press_rotate, progressive, heavy_forge.

Examples:
    # single-hit only, 3 series, VTK on
    python -m applications.Agility_Forge_physforge.main -s one_press -n 3 --save-vtk
    # single-hit, deterministic (no noise), to a scratch DB
    python -m applications.Agility_Forge_physforge.main -s one_press -n 3 --no-noise --db data/smoke.db
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from dataclasses import replace
import jax
import jax.numpy as np

from jax import config
config.update("jax_enable_x64", True)

_ = jax.devices()

from jax_fem_checkpoint.generate_mesh import get_meshio_cell_type, Mesh

from applications.Agility_Forge_physforge.mesh_container import MeshContainer
from applications.Agility_Forge_physforge.hit_config import Hit
from applications.Agility_Forge_physforge.lib.constitutive import ThermalMechanical
from applications.Agility_Forge_physforge.lib.time_stepper import AutomaticTimeStepperTM
from functools import partial
from applications.Agility_Forge_physforge.lib.boundary_conditions import (
    build_cylinder_press_bcs, build_unload_bcs, refresh_problem_surface_integrals)
from applications.Agility_Forge_physforge.data_storage import ForgingDBMS


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
# [v2] Axial band re-mapping onto the deformed configuration
# =============================================================================
def _remap_band_deformed(hit, mesh, current_sol_u, H):
    """Re-map a hit's axial contact band from the DEFORMED bar onto reference x.

    The recorded ``u2`` is the hammer position on the *physical* (elongating)
    bar, but the band test in ``build_cylinder_press_bcs`` is evaluated on the
    REFERENCE mesh.  v1 divided ``u2`` by the reference length H, so once the
    bar elongated past H the band ran off the end and was clamped (hit 13 in the
    17-hit run collapsed to 1.66 mm, i.e. below one element).

    Here the current deformed length ``L_def`` is measured from ``pts + sol_u``
    and used for a first-order (uniform-stretch) pull-back:

        s        = u2 / L_def                 normalised position on the bar
        c_ref    = s * H                      -> reference-config band centre
        half_ref = (w/2) * (H / L_def)        band that is ``w`` mm once deformed

    Scaling the half-width keeps the hammer footprint equal to ``w`` mm on the
    deformed bar instead of letting it stretch with the material.
    """
    pts = onp.array(mesh.points)
    x_def = (pts + onp.array(current_sol_u))[:, 0]
    L_def = float(x_def.max() - x_def.min())
    if not onp.isfinite(L_def) or L_def <= 1e-9:
        return hit, None

    s = min(max(hit.axial_pos_mm / L_def, 0.0), 1.0)
    c_ref = s * H
    half_ref = 0.5 * float(hit.band_width_mm) * (H / L_def)
    # [band] Warn when the die runs off the end of the bar.  Clamping is
    # physically right -- a die pressed past the end only contacts part of its
    # face -- but it silently shrinks the footprint, and the printed band width
    # then looks like a narrower die rather than a partially engaged one.
    lo_raw, hi_raw = c_ref - half_ref, c_ref + half_ref
    over_lo = max(0.0, -lo_raw)
    over_hi = max(0.0, hi_raw - float(H))
    if over_lo > 1e-9 or over_hi > 1e-9:
        requested = hi_raw - lo_raw
        effective = max(0.0, min(hi_raw, float(H)) - max(lo_raw, 0.0))
        pct = 100.0 * effective / requested if requested > 1e-12 else 0.0
        side = "low end (x=0)" if over_lo >= over_hi else "high end (x=H)"
        print(f"  [band][warn] die overhangs the {side} by "
              f"{max(over_lo, over_hi):.2f} mm (reference coords); only "
              f"{effective:.2f} of {requested:.2f} mm is on the bar "
              f"({pct:.0f}% of the die engaged).")

    xmin = max(0.0, min(1.0, (c_ref - half_ref) / H))
    xmax = max(0.0, min(1.0, (c_ref + half_ref) / H))
    info = (L_def, s, (xmax - xmin) * H)
    return replace(hit, x_min_band=xmin, x_max_band=xmax), info


# =============================================================================
# Run one series (multi-hit forging on a single workpiece)
# =============================================================================
# =============================================================================
# Force-controlled unloading between hits
# =============================================================================
def run_unloading(problem, mesh, sol_u, sol_dT, int_vars, reaction_u,
                  axial_constraint="face", n_steps=10, total_time=1e-3,
                  rho_ini=None, T_old_last=None, dt_last=None, balance="self", mode="peel",
                  rot_deg=0.0, vtk_dir=None):
    """Release the previous hit's die reaction gradually.

    The contact rows are replaced by the external force ``lambda * R`` on the
    former contact nodes (``build_unload_bcs``) and ``lambda`` is ramped from
    1 to 0 by the stepper's load schedule, with retries on failure.  Before
    stepping, the residual at ``lambda = 1`` is evaluated as a consistency
    check: the released state must still be in equilibrium, so the residual
    at the released nodes should be at round-off.

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

    # ---- lambda = 1 consistency check ----
    # Evaluated twice: with the T_old and dt of the last converged step (must
    # reproduce the captured reaction, so the residual at the released nodes
    # is round-off), and with the end-of-hit temperature the release will
    # actually use (the difference is the thermal-stress change over one
    # step, which the first release increment has to absorb).
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


def run_forging_series(hits, mesh, problem, sol_u0, sol_dT_initial, H, R,
                       T_linear_fn, db, series_id, save_vtk=False, vtk_dir=None,
                       bc_builder=None, checkpoint_dir=None, resume_from=0,
                       unload="peel", unload_steps=10, unload_time=1e-3,
                       axial_constraint="face", unload_balance="self",
                       unload_vtk=False):
    """Execute a single series of hits on the same workpiece.

    State carried across hits: F, alpha (plasticity history) and displacement.
    Between hits (Version 8): temperature is reset to the initial profile via a
    10-step thermal-state relaxation ramp (mechanical re-equilibration at
    scale=0), then the contact BC is rebuilt on the post-relaxation surface.
    Contact follows Route B (Updated-Lagrangian dynamic contact, re-detected
    every load step).

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
    # bc_builder(mesh, R, H, hit, current_sol_u, T_linear_fn) -> BC tuple.
    # Default = cylinder side-compression; physforge passes a box variant.
    if bc_builder is None:
        bc_builder = build_cylinder_press_bcs

    current_sol_u = sol_u0
    current_sol_dT = np.array(sol_dT_initial)
    current_int_vars = problem.internal_vars
    last_res_u = None            # die reaction at the end of the previous hit
    last_T_old, last_dt = None, None
    prev_rot = 0.0               # rotation of the previous hit, for the peel order

    # [v6a] Checkpointing.  The plasticity history (F, be, alpha, ...) lives
    # only in memory -- neither the DB nor the VTK stores it -- so without a
    # checkpoint an interrupted run cannot be continued: the material would
    # forget the work hardening it has already accumulated.  Writing
    # (sol_u, sol_dT, int_vars) after every hit makes a true resume possible.
    if resume_from > 0:
        if not checkpoint_dir:
            raise ValueError("resume_from > 0 also requires checkpoint_dir")
        ck = os.path.join(checkpoint_dir, f"ck_hit_{resume_from}.npz")
        if not os.path.exists(ck):
            raise FileNotFoundError(f"checkpoint not found: {ck}")
        z = onp.load(ck)
        current_sol_u = np.array(z["sol_u"])
        current_sol_dT = np.array(z["sol_dT"])
        current_int_vars = [np.array(z[f"iv_{i}"]) for i in range(int(z["n_iv"]))]
        # what the release of the checkpointed hit needs (see the save below)
        if "res_u" not in z.files:
            raise ValueError(f"{ck} has no die reaction / contact set of its last hit "
                             f"(written before the release state was checkpointed), so "
                             f"the release between hits cannot be continued from it")
        last_res_u = np.array(z["res_u"])
        last_T_old = np.array(z["T_old_last"]) if "T_old_last" in z.files else None
        last_dt = float(z["dt_last"]) if "dt_last" in z.files else None
        prev_rot = float(z["rot_last"])
        problem.contact_left_inds = onp.asarray(z["contact_left_inds"])
        problem.contact_right_inds = onp.asarray(z["contact_right_inds"])
        print(f"[CKPT] resumed from {ck}: {resume_from} hits done, "
              f"{len(current_int_vars)} int_vars components, "
              f"|u|_inf={float(onp.abs(z['sol_u']).max()):.3f}mm "
              f"-> continuing with hit {resume_from+1}")

    for hit_idx, hit in enumerate(hits):
        step_number = hit_idx + 1
        if step_number <= resume_from:
            continue                       # [v6a] already done in the checkpoint
        print(f"\n{'='*80}")
        print(f"  Series {series_id[:8]}... | HIT {step_number}/{len(hits)}")
        print(f"{'='*80}")

        # [v2] Re-map the axial band onto the current deformed length.  No-op
        # when axial_pos_mm is None (v1 behaviour).
        if getattr(hit, "axial_pos_mm", None) is not None:
            hit, remap = _remap_band_deformed(hit, mesh, current_sol_u, H)
            if remap is not None:
                L_def, s, w_ref = remap
                print(f"  [v2 band] u2={hit.axial_pos_mm:.2f}mm on deformed "
                      f"L={L_def:.2f}mm -> s={s:.3f}, band={w_ref:.2f}mm "
                      f"(ref) for {hit.band_width_mm:.2f}mm deformed footprint")

        print(f"  Params: band=[{hit.x_min_band:.3f}, {hit.x_max_band:.3f}]*H, "
              f"disp={hit.compression_displacement:.3f}, "
              f"rot={hit.rotation_euler_x:.1f}°, "
              f"time={hit.total_time:.3f}s")

        # Record input state BEFORE the hit
        input_u = current_sol_u
        input_T = current_sol_dT

        # ------------------------------------------------------------------
        # Between-hit transition
        # ------------------------------------------------------------------
        # The previous hit's contact is released gradually (run_unloading,
        # unload="peel" or "force") with only rigid-body rows installed, the
        # reheat ramp runs contact-free, and the new hit's rows are built
        # once, on the unloaded and reheated surface.  The first hit has no
        # previous contact: its rows are built directly.

        def _build_hit_bcs(u_for_bc):
            out = bc_builder(mesh, R=R, H=H, hit=hit, current_sol_u=u_for_bc,
                             T_linear_fn=T_linear_fn)
            problem.dirichlet_bc_info = [out[0], out[1]]
            problem.location_fns = out[2]
            refresh_problem_surface_integrals(problem)
            return out

        if hit_idx == 0:
            try:
                (dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect,
                 bc_update_fn, surface_inds) = _build_hit_bcs(None)
            except Exception as e:
                print(f"  [ERROR] BC construction failed: {e}")
                db.fail_series(series_id, step_number)
                return False
        else:
            ok_unl, current_sol_u, current_sol_dT, current_int_vars, unload_bc_update = \
                run_unloading(problem, mesh, current_sol_u, current_sol_dT,
                              current_int_vars, last_res_u,
                              axial_constraint=axial_constraint,
                              n_steps=unload_steps, total_time=unload_time,
                              T_old_last=last_T_old, dt_last=last_dt,
                              balance=unload_balance, mode=unload, rot_deg=prev_rot,
                              vtk_dir=(os.path.join(vtk_dir, f"unload_{hit_idx}")
                                       if (unload_vtk and vtk_dir) else None))
            if not ok_unl:
                print(f"  [ERROR] Hit {step_number}: unloading of the previous hit failed")
                db.fail_series(series_id, step_number)
                return False
            ramp_bc_update = unload_bc_update
            ramp_params = (lambda step, scale: (step, 1.0))      # lambda = 0: fully released

        # Reset temperature between hits (Version 8: thermal-state relaxation)
        if hit_idx > 0:
            print("  Resetting temperature to initial profile (cool-down between hits)...")

            # === Thermal-state relaxation by gradual temperature ramp ===
            # The carried mechanical history (be_old / F_old / alpha) was
            # equilibrated under the PREVIOUS hit's (cooled) temperature.  If we
            # reset the temperature to the reheated profile in one shot, the new
            # T_old makes the thermal-stress term and the (much lower) high-T
            # yield strength jump abruptly, leaving a dt-INDEPENDENT static
            # equilibrium mismatch that Newton cannot bridge in a single solve.
            # Instead ramp T_old linearly from the cooled end-state to the
            # reheated profile over several increments, doing one mechanical
            # re-equilibration at each increment.
            T_cold = onp.array(current_sol_dT)        # previous-hit cooled end-state
            T_hot = onp.array(sol_dT_initial)         # reheated target profile
            n_ramp = 10
            relax_dt = 1e-4
            print(f"  Running thermal-state relaxation: {n_ramp}-step temperature "
                  f"ramp (cold -> reheated), contact-free...")

            for k in range(1, n_ramp + 1):
                frac = k / n_ramp
                T_blend = np.array((1.0 - frac) * T_cold + frac * T_hot)
                print(f"    [relax {k}/{n_ramp}] T_old ramp frac={frac:.2f}, "
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
                    print(f"  [ERROR] Hit {step_number} thermal-state relaxation "
                          f"failed at ramp step {k}/{n_ramp}!")
                    db.fail_series(series_id, step_number)
                    return False

                # Carry the re-equilibrated mechanical state to the next increment.
                current_sol_u = relax_stepper.sol_u
                current_int_vars = relax_stepper.int_vars

            # [DIAG-B] last relax-solved T vs the target T_old hit-N will use.
            try:
                T_relaxed = np.array(relax_stepper.sol_dT)
                T_target = np.array(sol_dT_initial)
                dT_field = T_relaxed - T_target
                l2 = float(np.linalg.norm(dT_field))
                linf = float(np.max(np.abs(dT_field)))
                print(f"  [DIAG-B] last relax-solved T:  min/max = "
                      f"{float(np.min(T_relaxed)):.2f}/{float(np.max(T_relaxed)):.2f} C")
                print(f"  [DIAG-B] ||T_relaxed - T_target||_2 = {l2:.4e}, "
                      f"max|dT| = {linf:.4f} C")
            except Exception as _diag_e:
                print(f"  [DIAG-B] temperature-mismatch check failed: {_diag_e}")

            # Finalize: mechanical state now equilibrated under the reheated profile
            current_sol_dT = sol_dT_initial
            print("  Relaxation complete: mechanical state re-equilibrated under reheated T_old.")

            # Build the contact BCs on the POST-relaxation surface so the platen reference planes align with the actual
            # starting geometry, avoiding a dt-independent contact jump on the
            # first step.
            try:
                (dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect,
                 bc_update_fn, surface_inds) = _build_hit_bcs(current_sol_u)
            except Exception as e:
                print(f"  [ERROR] BC (re)build after relaxation failed: {e}")
                db.fail_series(series_id, step_number)
                return False

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
        stepper.seed_state(np.array(current_sol_u), np.array(current_sol_dT), current_int_vars)

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
        current_int_vars = stepper.int_vars
        last_res_u = stepper.last_res_u      # die reaction, for the unloading
        last_T_old, last_dt = stepper.last_T_old, stepper.last_dt
        prev_rot = float(hit.rotation_euler_x)

        # [v6a] Persist the full state (including plasticity history) so an
        # interrupted run can be resumed.
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            iv = [onp.asarray(a) for a in current_int_vars]
            onp.savez_compressed(
                os.path.join(checkpoint_dir, f"ck_hit_{step_number}.npz"),
                sol_u=onp.asarray(current_sol_u), sol_dT=onp.asarray(current_sol_dT),
                n_iv=onp.array(len(iv)), **{f"iv_{i}": a for i, a in enumerate(iv)},
                # the release before the next hit needs these (resume)
                **({"res_u": onp.asarray(last_res_u)} if last_res_u is not None else {}),
                rot_last=onp.array(prev_rot),
                contact_left_inds=onp.asarray(getattr(problem, "contact_left_inds", [])),
                contact_right_inds=onp.asarray(getattr(problem, "contact_right_inds", [])),
                **({"T_old_last": onp.asarray(last_T_old)} if last_T_old is not None else {}),
                **({"dt_last": onp.array(float(last_dt))} if last_dt is not None else {}))
            prev = os.path.join(checkpoint_dir, f"ck_hit_{step_number-1}.npz")
            if step_number > 1 and os.path.exists(prev):
                os.remove(prev)          # keep only the latest, to save disk

        print(f"  Hit {step_number} OK — {stepper.step_count} steps, {hit_wall:.1f}s")

    # All hits completed
    db.complete_series(series_id)
    return True


# =============================================================================
# Main entry point
# =============================================================================
def main(args=None):
    """Read forging.yml, expand hits, run series loop, record to DB.

    CLI overrides (``args``, from ``parse_args()``) take precedence over the
    YAML values; any field left as None keeps the YAML/default value.
    """

    # --- Paths ---
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, "data")
    config_path = os.path.join(crt_file_path, "configs", "forging.yml")
    if args is not None and args.config:
        config_path = args.config
    db_path = os.path.join(data_dir, "forge_database.db")
    if args is not None and getattr(args, "db", None):
        db_path = args.db
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

    # CLI overrides (None = keep the YAML value)
    if args is not None:
        if args.n_series is not None:
            n_series = args.n_series
        if args.add_noise is not None:
            add_noise = args.add_noise
        if args.save_vtk is not None:
            save_vtk = args.save_vtk
        if args.seed is not None:
            seed = args.seed

    # Sequence types: everything in YAML that has 'position_configs'
    available = [k for k in cfg if isinstance(cfg[k], dict)
                 and "position_configs" in cfg[k]]
    if args is not None and args.sequence:
        unknown = [s for s in args.sequence if s not in available]
        if unknown:
            raise SystemExit(
                f"Unknown sequence(s): {unknown}. Available: {available}")
        sequence_names = list(args.sequence)
    else:
        sequence_names = available
    print(f"Running {len(sequence_names)} sequence type(s): {sequence_names} "
          f"(available: {available})")
    print(f"Config: n_series={n_series}, add_noise={add_noise}, "
          f"save_vtk={save_vtk}, seed={seed}")

    # --- Load mesh ---
    print("\nLoading mesh...")
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

    # --- Between-hit release and rigid-body options (CLI; defaults when args is None) ---
    if args is not None:
        bc_builder = partial(build_cylinder_press_bcs,
                             axial_constraint=args.axial_constraint)
        series_kwargs = dict(bc_builder=bc_builder,
                             unload=args.unload, unload_steps=args.unload_steps,
                             unload_time=args.unload_time,
                             axial_constraint=args.axial_constraint,
                             unload_balance=args.unload_balance,
                             unload_vtk=args.unload_vtk)
        print(f"Options: unload={args.unload} ({args.unload_steps} steps, "
              f"{args.unload_time} s), axial_constraint={args.axial_constraint}")
    else:
        series_kwargs = {}

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
                **series_kwargs,
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


import argparse


def parse_args():
    """Command-line overrides for the data-generation driver.

    All options default to None and fall back to forging.yml. Examples:
        # only the single-hit sequence, 3 series, VTK on
        python -m applications.Agility_Forge_physforge.main -s one_press -n 3 --save-vtk
        # two sequences, no noise
        python -m applications.Agility_Forge_physforge.main -s one_press press_rotate --no-noise
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Agility_Forge_physforge — multi-hit forging data generation (V8).")
    p.add_argument("--sequence", "-s", nargs="+", metavar="NAME", default=None,
                   help="sequence type(s) to run, e.g. -s one_press "
                        "(default: all defined in forging.yml)")
    p.add_argument("--n-series", "-n", type=int, default=None, dest="n_series",
                   help="override run.n_series (number of series per sequence)")
    p.add_argument("--seed", type=int, default=None,
                   help="override run.random_seed")
    p.add_argument("--save-vtk", action=argparse.BooleanOptionalAction,
                   default=None, help="override run.save_vtk (--save-vtk / --no-save-vtk)")
    p.add_argument("--noise", action=argparse.BooleanOptionalAction,
                   default=None, dest="add_noise",
                   help="override run.add_noise (--noise / --no-noise)")
    p.add_argument("--config", "-c", default=None,
                   help="path to forging.yml (default: configs/forging.yml)")
    p.add_argument("--db", default=None,
                   help="output SQLite path (default: data/forge_database.db)")
    add_model_options(p)
    return p.parse_args()


def add_model_options(p):
    """Between-hit release and rigid-body options shared by the drivers.

    The model corrections of November 2026 (README, Model corrections) are
    built in; their previous behaviour is no longer selectable.
    """
    p.add_argument("--unload", choices=["peel", "force"], default="peel",
                   help="between hits: 'peel' frees the previous hit's contact "
                        "nodes progressively, outermost first, pinning the rest "
                        "where the hit left them (displacement control); 'force' "
                        "ramps the die reaction to zero as an external force "
                        "(load control: unstable on slender hot bars, see "
                        "build_unload_bcs)")
    p.add_argument("--unload-vtk", action="store_true", dest="unload_vtk",
                   help="write a VTK frame per release increment (diagnostics)")
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


if __name__ == "__main__":
    main(parse_args())
