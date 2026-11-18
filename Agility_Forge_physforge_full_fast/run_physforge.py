"""Physical-forge reproduction driver (Agility_Forge_physforge_full_fast).

Drives the V8 multi-hit thermo-mechanical forging solver with a REAL forge hit
sequence recorded in `2026-06-29.pt` (exported to .npz by the extraction step),
on a clean rectangular-box TET billet that approximates the physical
rounded-rectangle stock.

Action convention (reverse-engineered from the .pt before/after scans):
    u = (bite_depth_mm, rotation_rad [absolute, 0 or pi/2], axial_position_mm)

The only geometry-specific change vs the stock cylinder pipeline is the
outer-surface test: a box lateral-face test is injected via
`build_cylinder_press_bcs(..., surface_mask_fn=...)`.  All rotation / dynamic
contact / cooling / rigid-constraint logic is reused unchanged.

Usage (from jax-fem root):
    python -m applications.Agility_Forge_physforge_full_fast.run_physforge \
        --n-hits 3 --band-width 8 --temp 1000 --save-vtk
"""
import os
import argparse
from functools import partial

import numpy as onp
import jax
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)
_ = jax.devices()

from jax_fem_checkpoint.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh

from applications.Agility_Forge_physforge_full_fast.hit_config import Hit
from applications.Agility_Forge_physforge_full_fast.lib.constitutive import ThermalMechanical
from applications.Agility_Forge_physforge_full_fast.lib.boundary_conditions import build_cylinder_press_bcs
from applications.Agility_Forge_physforge_full_fast.data_storage import ForgingDBMS
from applications.Agility_Forge_physforge_full_fast.main import run_forging_series
from applications.Agility_Forge_physforge_full_fast.lib import profiling as _profiling
_profiling.maybe_install()          # AGF_PROFILE=1 only; numerics unchanged


# =============================================================================
# Box billet mesh (rounded-rect approximated as a rectangular prism)
# =============================================================================
def build_box_billet(Lx, Ly, Lz, Nx, Ny, Nz, data_dir, ele_type="TET4"):
    """TET4 box [0,Lx] x [-Ly/2,Ly/2] x [-Lz/2,Lz/2] (axis = x, cross-section
    centered on the x-axis so rotation about x stays about the bar centerline)."""
    m = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir, ele_type=ele_type)
    pts = onp.array(m.points, dtype=onp.float64)
    pts[:, 1] -= Ly / 2.0
    pts[:, 2] -= Lz / 2.0
    cells = m.cells_dict[get_meshio_cell_type(ele_type)]
    return Mesh(pts, cells)


# Local face tables: which corner nodes make up each face of an element.
# TET4 has 4 triangular faces, HEX8 has 6 quadrilateral faces (VTK ordering).
_FACE_TABLE = {
    "TET4": onp.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    "HEX8": onp.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]),
}


def load_billet_mesh(path, ele_type=None):
    """Load a billet mesh from file (anything meshio reads: .msh, .vtk, .inp...).

    The bar axis must be x; the cross-section is arbitrary.  Returns
    ``(Mesh, ele_type)`` -- the element type is taken from the file when not
    given, preferring TET4/HEX8 if the file holds several cell blocks.
    """
    import meshio
    m = meshio.read(path)
    prefer = ["TET4", "HEX8"]
    available = {t: get_meshio_cell_type(t) for t in prefer}
    if ele_type is None:
        for t, mio in available.items():
            if mio in m.cells_dict:
                ele_type = t
                break
        else:
            raise ValueError(
                f"{path} holds none of {prefer}; cell blocks are "
                f"{sorted(m.cells_dict)}")
    mio = get_meshio_cell_type(ele_type)
    if mio not in m.cells_dict:
        raise ValueError(f"{path} has no {ele_type} ({mio}) cells; "
                         f"available: {sorted(m.cells_dict)}")
    pts = onp.array(m.points, dtype=onp.float64)
    print(f"[mesh] loaded {path}: {len(pts)} nodes, "
          f"{len(m.cells_dict[mio])} {ele_type} cells, "
          f"x=[{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]")
    return Mesh(pts, m.cells_dict[mio]), ele_type


def make_topological_surface_test(mesh, ele_type="TET4"):
    """Outer lateral surface taken from mesh TOPOLOGY rather than geometry.

    A face that belongs to exactly one element is on the boundary.  Dropping the
    boundary faces whose nodes all lie on x=min or x=max leaves the lateral
    surface.  Nothing here assumes a particular cross-section, so the same code
    handles a box, a cylinder, a rounded rectangle or a tapered/stepped bar --
    only the "axis is x" convention is used, and that is fixed for this package.

    Verified to select the identical node set as ``make_box_surface_test`` on the
    30x16x16 box billet (1984 nodes), so switching to it does not change results
    for the existing rectangular case.

    Returns a callable with the same contract as ``make_box_surface_test``: it
    takes reference-configuration points and returns a boolean node mask.  The
    mask is precomputed, since the topology does not change with deformation.
    """
    if ele_type not in _FACE_TABLE:
        raise ValueError(
            f"topological surface test supports {sorted(_FACE_TABLE)}, got {ele_type}")
    ft = _FACE_TABLE[ele_type]
    cells = onp.asarray(mesh.cells)
    faces = cells[:, ft].reshape(-1, ft.shape[1])
    uniq, counts = onp.unique(onp.sort(faces, axis=1), axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        raise ValueError("no boundary faces found -- is the mesh connectivity valid?")

    x = onp.asarray(mesh.points)[:, 0]
    span = float(x.max() - x.min())
    tol = 1e-6 * span if span > 0 else 1e-9
    on_lo = onp.all(onp.abs(x[boundary] - x.min()) <= tol, axis=1)
    on_hi = onp.all(onp.abs(x[boundary] - x.max()) <= tol, axis=1)
    lateral = boundary[~(on_lo | on_hi)]

    mask = onp.zeros(len(x), dtype=bool)
    mask[onp.unique(lateral)] = True
    print(f"[mesh] topological surface: {len(boundary)} boundary faces, "
          f"{len(lateral)} lateral, {int(mask.sum())} nodes ({ele_type})")

    def surface_mask_fn(points):
        return mask

    return surface_mask_fn


def make_box_surface_test(mesh, rtol=1e-4):
    """Reference-config test: node is on the billet's 4 lateral faces
    (|y|=Ly/2 or |z|=Lz/2).  End caps (x=0, x=Lx) are excluded so they are not
    treated as contact/cooling surface."""
    pts = onp.array(mesh.points)
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    tol = rtol * max(ymax - ymin, zmax - zmin)

    def surface_mask_fn(points):
        p = onp.asarray(points)
        on_y = (onp.abs(p[:, 1] - ymin) <= tol) | (onp.abs(p[:, 1] - ymax) <= tol)
        on_z = (onp.abs(p[:, 2] - zmin) <= tol) | (onp.abs(p[:, 2] - zmax) <= tol)
        return on_y | on_z

    return surface_mask_fn


# =============================================================================
# Hit sequence from the exported .npz
# =============================================================================
def load_hits(npz_path, H, band_width_mm, v_platen=2.04, n_hits=None,
              closure_factor=1.0, axial_mode="reference", scan_depths=None,
              depth_scale=1.0):
    """Map the recorded action u=(u0, u1_rad, u2_mm) to Hit objects.

    Depth convention.  ``u0`` is the TOTAL closure of the hit and each of the
    two symmetric platens travels ``compression_displacement`` (per-side), so
    the per-side travel is ``u0 * closure_factor / 2``.

    ``closure_factor`` (v2) calibrates the hammer STROKE down to the PERMANENT
    reduction.  v1 used 1.0, i.e. a total closure of u0 = 8.5..17.4 mm, while
    the permanent closure measured from the scans is only ~3-7 mm -- the run
    therefore over-deformed the bar (+55% elongation over 17 hits).  With
    ``closure_factor = 0.4`` the total closure spans 3.4..7.0 mm, matching the
    measured range.  Set 1.0 to reproduce v1 exactly.

    ``axial_mode`` (v2) selects how ``u2`` is turned into a contact band:
      * ``"reference"`` -- v1: divide by the reference length H.  Once the bar
        elongates past H the band runs off the end and is clamped (hit 13
        collapsed to 1.66 mm < one element).
      * ``"deformed"``  -- carry u2 and the footprint on the Hit and let
        ``main._remap_band_deformed`` re-map them onto the current deformed
        length before each hit.

    ``scan_depths`` (v4) overrides the depth entirely.  It is the per-side
    travel measured from the before/after scans in ``2026-06-29.pt``
    (see extract_scan_depths.py), so no calibration constant is involved.
    ``u0`` is NOT the plastic reduction: the measured ratio
    ``permanent/u0`` ranges 0.26..1.19 across the 17 hits, so no single
    ``closure_factor`` can reproduce it.  When given, ``closure_factor``
    is ignored.
    """
    d = onp.load(npz_path)
    depth = d["depth_mm"]        # u0  (total closure, mm)
    rot_deg = d["rotation_deg"]  # degrees(u1), absolute orientation
    axial = d["axial_pos_mm"]    # u2  (axial position, mm)
    n = len(depth) if n_hits is None else min(n_hits, len(depth))
    hits = []
    for i in range(n):
        c = float(axial[i])
        # Initial band on the reference bar.  In "deformed" mode this is only a
        # seed: run_forging_series recomputes it per hit from the deformed length.
        # The band is selected in the deformed configuration and the bar may be
        # longer than H, so the upper end is not clamped to the reference
        # length; the BC builder warns if the band finds no material.
        xmin = max(0.0, (c - band_width_mm / 2.0) / H)
        xmax = max(0.0, (c + band_width_mm / 2.0) / H)
        if scan_depths is not None:
            per_side_disp = float(scan_depths[i])      # measured per-side travel
        else:
            per_side_disp = float(depth[i]) * float(closure_factor) / 2.0
        # [stress-test] scale every bite; 1.0 leaves the measured value alone
        per_side_disp *= float(depth_scale)
        hits.append(Hit(
            x_min_band=xmin,
            x_max_band=xmax,
            compression_displacement=per_side_disp,
            rotation_euler_x=float(rot_deg[i]),
            total_time=per_side_disp / v_platen,
            axial_pos_mm=(c if axial_mode == "deformed" else None),
            band_width_mm=(float(band_width_mm) if axial_mode == "deformed" else None),
        ))
    return hits


def main():
    ap = argparse.ArgumentParser(description="Physical-forge reproduction (box billet).")
    ap.add_argument("--hits", default=None, help="path to exported hits .npz")
    ap.add_argument("--n-hits", type=int, default=None, dest="n_hits",
                    help="only run the first N hits (default: all 17)")
    ap.add_argument("--band-width", type=float, default=8.0, dest="band_width",
                    help="axial contact-band width in mm (hammer footprint)")
    ap.add_argument("--temp", type=float, default=1000.0,
                    help="uniform billet temperature in C (physical value not in .pt)")
    ap.add_argument("--nx", type=int, default=30)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--save-vtk", action="store_true")
    ap.add_argument("--db", default=None)
    # ---- v2 options (defaults reproduce v1 byte-for-byte) ----
    ap.add_argument("--closure-factor", type=float, default=1.0, dest="closure_factor",
                    help="[v2] scale u0 (hammer stroke) to the permanent closure. "
                         "1.0 = v1; 0.4 matches the ~3-7 mm measured from the scans")
    ap.add_argument("--axial-mode", choices=["reference", "deformed"],
                    default="reference", dest="axial_mode",
                    help="[v2] map u2 on the reference bar (v1) or on the current "
                         "deformed length (fixes the hit-13 band clamp)")
    ap.add_argument("--vtk-subdir", default="vtk_physforge", dest="vtk_subdir",
                    help="[v2] VTK output sub-directory (change it to avoid "
                         "overwriting a previous run)")
    # ---- v4 option: drive the depth from the measured scans ----
    ap.add_argument("--depth-source", choices=["u0", "scan"], default="u0",
                    dest="depth_source",
                    help="[v4] 'u0' = derive depth from u0 (v1/v2/v3); 'scan' = use "
                         "the per-side reduction measured from the .pt before/after "
                         "scans, which needs no calibration constant")
    ap.add_argument("--scan-depths", default=None, dest="scan_depths_path",
                    help="[v4] path to physforge_scan_depths.npz "
                         "(default: alongside this script)")
    ap.add_argument("--depth-scale", type=float, default=1.0, dest="depth_scale",
                    help="multiply every per-hit bite by this factor. 1.0 = the "
                         "measured reduction; >1 drives a larger applied strain "
                         "per hit, for stress-testing convergence")
    ap.add_argument("--mesh", default=None,
                    help="load the billet from a mesh file (meshio-readable) "
                         "instead of generating a box; the bar axis must be x")
    ap.add_argument("--ele-type", default=None, dest="ele_type",
                    choices=["TET4", "HEX8"],
                    help="element type (default: TET4 for a generated box, or "
                         "auto-detected from --mesh)")
    ap.add_argument("--surface-mode", default="topological", dest="surface_mode",
                    choices=["topological", "box"],
                    help="how the outer lateral surface is identified. "
                         "'topological' works for any cross-section; 'box' is "
                         "the original hand-written rectangular test")
    ap.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir",
                    help="write (u, T, plasticity history) to this directory after every hit, so an interrupted run can be resumed")
    ap.add_argument("--resume-from", type=int, default=0, dest="resume_from",
                    help="resume from the checkpoint of hit N and continue with hit N+1 (requires --checkpoint-dir)")
    from applications.Agility_Forge_physforge_full_fast.main import add_model_options
    add_model_options(ap)
    # ---- fast solver options (lib/fast_newton.py, lib/linear_solvers.py) ----
    ap.add_argument("--newton", default="fast", choices=["fast", "original"],
                    help="'fast': device-resident assembly and Newton (default); "
                         "'original': jax_fem_checkpoint's solver, for comparison")
    ap.add_argument("--linear-solver", default="auto", dest="linear_solver",
                    help="auto (AMGX if pyamgx imports, else bicgstab) | bicgstab (the original method) | bicgstab_jacobi | gmres_jacobi | "
                         "cusolver | pardiso | mumps | amgxb[:agg|agg_dilu|dilu] (node blocks) | amgx[:fgmres_amg|bicgstab_amg|fgmres_ilu]")
    ap.add_argument("--second-solver", default="amgxbe:gs", dest="second_solver",
                    help="AMGX set-up tried when the main AMGX solve fails the check, before BiCGSTAB and "
                         "the direct solver: amgxbe:gs (default; Dirichlet columns eliminated, Gauss-Seidel "
                         "smoother) | amgxbe:bgs | none (straight to BiCGSTAB, as before)")
    ap.add_argument("--kernel-chunk", type=int, default=None, dest="kernel_chunk",
                    help="cells per element-kernel batch (default: all cells at once)")
    ap.add_argument("--fast-profile", action="store_true", dest="fast_profile",
                    help="synchronise the GPU around each part of the solve for timing")
    ap.add_argument("--fast-report", default=None, dest="fast_report",
                    help="write the solver timing and counters (JSON) to this file at exit")
    ap.add_argument("--fast-verify", type=int, default=0, dest="fast_verify",
                    help="compare residual and Jacobian with jax_fem_checkpoint's assembly "
                         "on the first N Newton solves (diagnostic, slow)")
    args = ap.parse_args()
    from applications.Agility_Forge_physforge_full_fast.lib import fast_newton
    fast_newton.CONFIG.update(newton=args.newton, linear_solver=args.linear_solver,
                              second_solver=args.second_solver, chunk=args.kernel_chunk, profile=args.fast_profile,
                              verify=args.fast_verify)
    import atexit
    import json

    def _fast_report():
        if args.newton != "fast":
            return
        rep = fast_newton.report()
        if args.fast_report:
            with open(args.fast_report, "w") as f:
                json.dump(rep, f, indent=1)
    atexit.register(_fast_report)
    if args.axial_mode == "deformed":
        raise SystemExit("the contact band is already selected on the deformed bar "
                         "(build_cylinder_press_bcs); --axial-mode deformed would "
                         "correct it a second time. Use --axial-mode reference.")

    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    npz = args.hits or os.path.join(data_dir, "physforge_hits_2026-06-29.npz")

    meta = onp.load(npz)
    Lx, Ly, Lz = float(meta["Lx"]), float(meta["Ly"]), float(meta["Lz"])
    print(f"Initial billet (box approx of rounded-rect): "
          f"{Lx:.2f} x {Ly:.2f} x {Lz:.2f} mm")

    if args.mesh:
        mesh, ele_type = load_billet_mesh(args.mesh, args.ele_type)
    else:
        ele_type = args.ele_type or "TET4"
        mesh = build_box_billet(Lx, Ly, Lz, args.nx, args.ny, args.nz, data_dir, ele_type)
    H = float(onp.max(onp.array(mesh.points)[:, 0]))
    print(f"Box TET mesh: nodes={len(mesh.points)}, cells={len(mesh.cells)}, "
          f"H(length)={H:.2f}")

    # Uniform hot temperature: the physical forge temperature is not stored in
    # the .pt, so we assume a uniform billet temperature (override with --temp).
    T_forge = args.temp

    def T_uniform(point):
        return T_forge

    pts = onp.array(mesh.points)
    sol_dT_initial = onp.full((len(pts), 1), T_forge, dtype=onp.float64)
    sol_u0 = np.zeros((len(pts), 3))

    # [v4] measured per-side reduction from the before/after scans
    scan_depths = None
    if args.depth_source == "scan":
        sp = args.scan_depths_path or os.path.join(data_dir, "physforge_scan_depths.npz")
        if not os.path.exists(sp):
            raise SystemExit(
                f"--depth-source scan requires {sp}\n"
                f"Generate it first: python -m applications.Agility_Forge_physforge_full_fast."
                f"extract_scan_depths")
        sd = onp.load(sp)
        scan_depths = sd["depth_per_side_mm"]
        u0_ref = sd["u0_mm"]
        print(f"[v4] driving with scan-measured bite depths ({sp})")
        print(f"[v4]   mean per side {scan_depths.mean():.2f} mm "
              f"(mean u0/2 {(u0_ref/2).mean():.2f} mm, "
              f"i.e. v1 was {(u0_ref/2).mean()/scan_depths.mean():.2f}x the measured value)")

    hits = load_hits(npz, H, args.band_width, n_hits=args.n_hits,
                     depth_scale=args.depth_scale,
                     closure_factor=args.closure_factor, axial_mode=args.axial_mode,
                     scan_depths=scan_depths)
    print(f"Loaded {len(hits)} hits (band_width={args.band_width} mm, T={T_forge} C, "
          f"closure_factor={args.closure_factor}, axial_mode={args.axial_mode}). "
          f"First hits:")
    for k, h in enumerate(hits[:4]):
        print(f"  hit{k+1}: disp={h.compression_displacement:.2f}mm "
              f"rot={h.rotation_euler_x:.0f}deg "
              f"band=[{h.x_min_band:.3f},{h.x_max_band:.3f}]*H "
              f"t={h.total_time:.3f}s")

    print("\nBuilding coupled thermo-mechanical problem...")

    def _no_location(p):
        return False

    problem = ThermalMechanical(
        mesh=[mesh, mesh], vec=[3, 1], dim=3,
        ele_type=[ele_type, ele_type], gauss_order=[2, 2],
        dirichlet_bc_info=[None, None],
        location_fns=[_no_location, _no_location, _no_location],
    )

    if args.surface_mode == "box":
        surface_test = make_box_surface_test(mesh)
    else:
        surface_test = make_topological_surface_test(mesh, ele_type)
    bc_builder = partial(build_cylinder_press_bcs, surface_mask_fn=surface_test,
                         axial_constraint=args.axial_constraint)
    print(f"Options: unload={args.unload} ({args.unload_steps} steps, "
          f"{args.unload_time} s), axial_constraint={args.axial_constraint}")
    if args.axial_constraint != "face":
        print(f"[v5] axial rigid-body constraint mode: {args.axial_constraint}")

    db_path = args.db or os.path.join(data_dir, "physforge.db")
    db = ForgingDBMS(db_path, new=True)
    series_id = db.start_series("physforge_2026-06-29", len(hits))
    problem.custom_init()
    from applications.Agility_Forge_physforge_full_fast.main import configure_air
    configure_air(problem, 2.0 * (max(Ly, Lz) / 2.0), T_forge, args)

    vtk_dir = None
    if args.save_vtk:
        vtk_dir = os.path.join(data_dir, args.vtk_subdir)
        os.makedirs(vtk_dir, exist_ok=True)

    ok = run_forging_series(
        hits=hits, mesh=mesh, problem=problem, sol_u0=sol_u0,
        sol_dT_initial=sol_dT_initial, H=H, R=max(Ly, Lz) / 2.0,
        T_linear_fn=T_uniform, db=db, series_id=series_id,
        save_vtk=args.save_vtk, vtk_dir=vtk_dir, bc_builder=bc_builder,
        checkpoint_dir=args.checkpoint_dir, resume_from=args.resume_from,
        unload=args.unload, unload_steps=args.unload_steps,
        unload_time=args.unload_time, axial_constraint=args.axial_constraint,
        unload_balance=args.unload_balance, unload_vtk=args.unload_vtk,
    )
    print(f"\n{'='*72}")
    print(f"physforge result: {'ALL HITS OK' if ok else 'FAILED (see [FAIL]/[ERROR] above)'}")
    print(f"{'='*72}")
    db.close()


if __name__ == "__main__":
    main()
