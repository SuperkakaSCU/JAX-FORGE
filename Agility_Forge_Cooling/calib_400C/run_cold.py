"""Replay the twelve room-temperature 6063 hits and record the press force.

Built on the solver stack copied into calib/lib/ from Agility_Forge_physforge
(adaptive step size with retry, F-bar J2 return map, Updated-Lagrangian
contact), with the constitutive law reduced to constant room-temperature 6063
properties.  What is new here is the press force: the reaction carried by the
upper platen's constrained dofs, projected on the press normal, so the
computed load can be put beside the 633 Hz record from the machine.
"""
import argparse
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

import numpy as onp
import meshio

import jax
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)

from jax_fem_checkpoint.generate_mesh import Mesh, get_meshio_cell_type

from paths import OUT, ACTIONS, R_STOCK_MM, T_ISO_C, SERIES
from contact import outer_surface_nodes, rot_x, press_force_kn
from lib.constitutive import ThermalMechanical
from lib.time_stepper import AutomaticTimeStepperTM
from lib.boundary_conditions import build_cylinder_press_bcs, refresh_problem_surface_integrals
from lib.hit_config import Hit
from lib.mesh_quality import QualityMonitor, DistortionStop

_LZ = re.compile(r"(:\s*)0+(\d)")
T_ROOM = T_ISO_C      # isothermal 400 C: uniform, constant, no thermal strain (T0 matches)
V_PLATEN = 2.04
SLUG = {"single": "single", "stroke 1": "s1", "release": "rel", "stroke 2": "s2", "release 2": "rel2"}          # mm/s, only sets the pseudo-time of a hit


def load_actions(n=None):
    out = []
    for line in ACTIONS.read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(_LZ.sub(r"\1\2", line))["action"]
        if not a.get("no_op"):
            out.append(a)
    return out[:n] if n else out


def bite_for(pts, surf, sol_u, phi_deg, z_mm, rho_mm, band_mm,
             band_config="reference", band_sol_u=None):
    """Platen travel per side needed to close on an absolute radius rho_mm.

    This MUST mirror what build_cylinder_press_bcs actually does, which is a
    symmetric travel read off the candidate set's rotated-y extremes:

        y_L = y_min0 + scale * d        y_R = y_max0 - scale * d

    Writing y_c = (y_min0 + y_max0) / 2, both platens land on y_c -+ rho for the
    SAME travel d = (y_max0 - y_min0) / 2 - rho, i.e. half the full span minus
    the target radius.  So the span has to be taken over the whole circumference
    of the candidate band -- the same nodes the BC selects -- and not over the
    strip the die faces.  Restricting it to the die face looks more physical but
    silently disagrees with the platen the BC actually builds: on hit 10 it gave
    a 0.145 mm bite where the consistent value is 1.185 mm, against 1.22 mm
    measured off the scan, and the hit came out at 5.5 kN against 78.9 kN.

    Band membership follows ``band_config`` and ``band_sol_u``, which MUST be
    the same values handed to build_cylinder_press_bcs: reading the travel off
    one band and building the platen on another is the exact mistake that put
    hit 1's second stroke 0.9 mm clear of the bar.  See the docstrings there.

    Note the two displacement fields do different jobs.  ``band_sol_u`` only
    decides WHICH nodes are in the band and is frozen at the start of the hit;
    ``sol_u`` is the state at the start of THIS stroke and is what the span is
    measured on, because the span has to be the one the platen will actually
    see.
    """
    lo, hi = z_mm - 0.5 * band_mm, z_mm + 0.5 * band_mm
    xs = pts[surf, 0]
    if band_config == "deformed":
        bu = band_sol_u if band_sol_u is not None else sol_u
        if bu is not None:
            xs = xs + onp.asarray(bu)[surf, 0]
    band = surf[(xs >= lo) & (xs <= hi)]
    if len(band) < 20:
        return None, 0
    xdef = pts[band] + (onp.zeros((len(band), 3)) if sol_u is None
                        else onp.asarray(sol_u)[band])
    y = (rot_x(phi_deg) @ xdef.T).T[:, 1]
    return 0.5 * (y.max() - y.min()) - rho_mm, len(band)


# TET10 face -> (corner, corner, corner, mid, mid, mid), meshio/VTK node order
_TET10_FACES = ([0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7],
                [1, 2, 3, 5, 9, 8], [0, 2, 3, 6, 9, 7])


def outer_surface_nodes_any(points, cells):
    """outer_surface_nodes for TET4 or TET10.

    contact.outer_surface_nodes works on the four corner nodes only, which on a
    TET10 mesh would leave the midside nodes of the boundary facets out of the
    span bite_for measures -- while the BC's candidate set (a radius test)
    does include them.  For TET10 the boundary faces are found from their
    corners and all six nodes of each are returned; for TET4 this is exactly
    the original function.
    """
    if cells.shape[1] == 4:
        return outer_surface_nodes(points, cells)
    corner = outer_surface_nodes(points, cells[:, :4])          # drops end caps too
    faces = onp.concatenate([cells[:, f] for f in _TET10_FACES], axis=0)
    key = onp.sort(faces[:, :3], axis=1)
    _, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    bnd = faces[idx[cnt == 1]]
    keep = onp.isin(bnd[:, :3], corner).all(axis=1)             # lateral faces only
    return onp.unique(bnd[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=OUT.parent / "billet.msh")
    ap.add_argument("--ele-type", choices=("TET4", "TET10", "HEX8"), default="TET4",
                    help="element of the mesh file.  TET10 is the locking test: the "
                         "F-bar correction is the identity on constant-strain TET4, "
                         "so only on TET10 does it actually act.  Build a TET10 mesh "
                         "on the same corners with make_tet10.py.  HEX8 is the "
                         "hexahedron F-bar was designed for; build the mesh with "
                         "mesh_hex.py (cells in meshio/VTK order -- they are permuted "
                         "into the package's own order here, see lib/hex8.py).")
    ap.add_argument("--stop-after", choices=("none", "stroke1", "release"), default="none",
                    help="end the run after this stroke of the LAST hit simulated -- for "
                         "single-hit studies that only need stroke 1, or stroke 1 and "
                         "the release, without paying for stroke 2.")
    ap.add_argument("--n-hits", type=int, default=12)
    ap.add_argument("--fast-linear", action="store_true", dest="fast_linear",
                    help="with --linear-solver mumps: build the CSR pattern once per mesh, reuse the "
                         "MUMPS analysis between solves of the same pattern, and reuse the residual the "
                         "Newton update computed in the line search (lib/fast_linear.py). Same equations, "
                         "equal results up to round-off.")
    ap.add_argument("--linear-solver", choices=("jax", "direct", "hybrid", "mumps", "gpu-hybrid"),
                    default="jax",
                    help="'jax' is the package default: an unpreconditioned "
                         "bicgstab whose convergence check the package has "
                         "commented out, so a failed solve is accepted silently. "
                         "'direct' uses a scipy sparse LU instead -- slower per "
                         "solve, but it cannot return a wrong answer, which is "
                         "what kills the roll3 releases.  'hybrid' runs bicgstab "
                         "and redoes only the solves whose residual fails the "
                         "check the package has commented out -- 28% at the worst "
                         "release, near zero elsewhere.  'mumps' factors every "
                         "solve with PETSc/MUMPS -- needed on TET10, where bicgstab "
                         "fails from the first solve and scipy's LU is far too slow.  "
                         "'gpu-hybrid' runs a block-Jacobi-preconditioned bicgstab on the "
                         "GPU and redoes only the solves that fail the residual check with "
                         "MUMPS on the CPU.")
    ap.add_argument("--release-abs-tol", type=float, default=10.0,
                    help="Newton tolerance for the release stroke, as an ABSOLUTE "
                         "residual in newtons; compression strokes keep the solver "
                         "default.  The release needs its own because it frees "
                         "every contact dof at once, so the whole platen reaction "
                         "enters the convergence test instead of being eliminated "
                         "with the constraints, and the iteration stalls around "
                         "2 N on a 1e5 N problem.  It must NOT be expressed "
                         "relatively -- see abs_tol in AutomaticTimeStepperTM for "
                         "the measured case where a relative 1e-4 declared a step "
                         "converged at 4e15 N and committed that state.")
    ap.add_argument("--release-accept-mm", type=float, default=3.0,
                    help="If a RELEASE stroke runs out of Newton iterations, its "
                         "last iterate is still accepted when it is finite and has "
                         "moved the bar by no more than this, in mm.  Measured with "
                         "unilateral contact, a legitimate release moves the bar "
                         "1.0-1.9 mm -- not local springback but the whole bar "
                         "relaxing and straightening, the free end moving most -- so "
                         "3 mm accepts a real one while a divergence, which runs to "
                         "tens of mm or to NaN, is still refused.  Compression "
                         "strokes are never accepted this way: their reaction is the "
                         "press load being measured.")
    ap.add_argument("--band-config", choices=("reference", "deformed"),
                    default="reference",
                    help="configuration in which the axial contact band is selected. "
                         "'reference' (default) presses the material that STARTED at z, "
                         "so the hit drifts downstream with the elongation; 'deformed' "
                         "presses whatever material is at lab coordinate z NOW, which is "
                         "what a machine axis commanded to a fixed z does.")
    ap.add_argument("--band", type=float, default=11.0,
                    help="axial contact width of the die, mm.  Tool2.stl has a 9.9 mm "
                         "flat that chamfers out to 13.4 mm at 3 mm height; on the roll2 "
                         "scans the half-depth imprint width of the three virgin-surface "
                         "hits is 11-12 mm at 2.6-3.4 mm bites (9.5 mm was measured on "
                         "roll1 at <= 2 mm bites).")
    ap.add_argument("--edge-radius", type=float, default=0.0,
                    help="smooth-transition regularization: radius (mm) of a tangent "
                         "circular fillet added outside each edge of the flat band. "
                         "0 (default) is the sharp-edged band.")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="state_hitNN.npz written at the end of a hit (in <run>/checkpoints/); "
                         "the run continues with hit NN+1 from that exact state.  The "
                         "mesh and element type must match; the material parameters "
                         "given on this command line are used from here on.")
    ap.add_argument("--min-scaled-jacobian", type=float, default=0.05,
                    help="stop the run, with a report, when the minimum corner scaled "
                         "Jacobian of the deformed mesh drops below this (1 = perfect "
                         "corner, 0 = about to invert, negative = inverted corner).  "
                         "0 disables the stop; a negative value lets corners invert down "
                         "to it, to see where Newton itself gives out.")
    ap.add_argument("--warn-scaled-jacobian", type=float, default=0.2)
    ap.add_argument("--accept-max-refusals", type=int, default=3,
                    help="a converged trial state whose minimum scaled Jacobian is "
                         "below --min-scaled-jacobian is discarded and the step retried "
                         "with half the dt (pre-commit check in the stepper); after this "
                         "many refusals of one step the run stops with the report of "
                         "the refused state, which is written as distortion_trial.vtu")
    ap.add_argument("--accept-max-refusals-total", type=int, default=8,
                    help="stop the run once this many trial states have been refused "
                         "within one stroke, whatever the step: a stroke that creeps on "
                         "in halved steps with the element one refusal from folding is "
                         "a contact / mesh problem, not a step-size one")
    ap.add_argument("--quality-max-drop", type=float, default=0.15,
                    help="also refuse a trial state whose minimum scaled Jacobian is "
                         "lower than that of the last committed state by more than "
                         "this, so that a fast-collapsing element forces smaller steps "
                         "before it folds.  0 disables.")
    ap.add_argument("--release-accept-res", type=float, default=1000.0,
                    help="a RELEASE step that runs out of Newton iterations is only "
                         "accepted (see --release-accept-mm) if its residual on the "
                         "free dofs is below this, in newtons; a stalled release "
                         "sits at a few N, a failed one at many thousands")
    ap.add_argument("--max-bite-step", type=float, default=0.25,
                    help="cap on the die travel per step as a fraction of the "
                         "shortest element edge of the mesh, so that one increment "
                         "never pushes a surface element by more than this fraction "
                         "of its size whatever --max-dt says; 0 disables")
    ap.add_argument("--final-release", action="store_true",
                    help="(--strokes two) retract the die after the second stroke of every "
                         "hit as well, to the same radius as the retract between the "
                         "strokes.  Without it the load of a hit is dropped in the FIRST "
                         "step of the next hit, when the die jumps to its new position -- "
                         "85 kN converges, 113 kN (Q 60, h20) does not.  Meant for the "
                         "force-based contact, whose release starts from the last die pose.")
    ap.add_argument("--contact", choices=("dirichlet", "penalty"), default="dirichlet",
                    help="'dirichlet' (default): contact nodes are prescribed onto the "
                         "flat die plane (lib/boundary_conditions.py).  'penalty': "
                         "force-based nodal contact against the die profile "
                         "(lib/die_contact.py): unilateral penalty normal force, "
                         "Coulomb friction in both tangential directions with "
                         "--friction-mu, every lateral surface node a candidate at "
                         "every Newton iteration, release when the gap opens.")
    ap.add_argument("--die-profile", choices=("flat", "fillet", "sigmoid"), default="flat",
                    help="(penalty contact) axial profile hung on the flat band of "
                         "--band: 'flat' = sharp band edge and nothing beyond; "
                         "'fillet' = tangent arc --die-fillet-r then a --die-flank-deg "
                         "flank up to --die-height (Tool2.stl: band 7.08, r 3.175, "
                         "83 deg, 25.4 mm); 'sigmoid' = logistic wall of width "
                         "--die-sigmoid-w up to --die-height")
    ap.add_argument("--die-normal", choices=("surface", "radial"), default="surface",
                    help="(penalty contact) 'surface': the die force acts along the true "
                         "normal of the profile.  'radial': the die only transmits force "
                         "along the press direction (vertical gap to the profile), so an "
                         "inclined face does not wedge the surface nodes axially outward")
    ap.add_argument("--die-soft-edge", type=float, default=0.0,
                    help="(penalty contact, flat profile) width in mm of a logistic "
                         "window on the contact stiffness, 0.5 at the band edge: the "
                         "die face stays flat and its stiffness fades out instead of "
                         "ending, so no node is wedged outward by an inclined face and "
                         "none loses a finite force by crossing an edge.  0 = sharp edge.")
    ap.add_argument("--die-fillet-r", type=float, default=3.175)
    ap.add_argument("--die-flank-deg", type=float, default=83.0)
    ap.add_argument("--die-height", type=float, default=25.4)
    ap.add_argument("--die-sigmoid-w", type=float, default=0.5)
    ap.add_argument("--contact-penalty", type=float, default=1.0,
                    help="(penalty contact) normal penalty stiffness per node as a "
                         "multiple of E x shortest element edge; 1 puts the "
                         "penetration at a few hundredths of a mm under the "
                         "measured loads")
    ap.add_argument("--dump-contact", action="store_true",
                    help="write, for every accepted step of a compression stroke, the "
                         "contact nodes with their die-frame position, reaction and "
                         "friction force to <run>/contact_hitNN_<stroke>/step_NNNN.npz")
    ap.add_argument("--skip-hits", default="",
                    help="comma-separated 1-based hit numbers to leave out of the "
                         "sequence (e.g. '2'); later hits are pressed on the bar as "
                         "it is without them")
    ap.add_argument("--friction-mu", type=float, default=None,
                    help="Coulomb friction coefficient along the bar axis (lib/friction.py). "
                         "Implies --axial-slip: contact nodes are held in y and z, and their "
                         "axial motion is resisted by at most mu times the normal force.")
    ap.add_argument("--friction-delta", type=float, default=0.01,
                    help="slip (mm) at which a sticking node reaches the Coulomb limit")
    ap.add_argument("--axial-slip", action="store_true",
                    help="contact nodes slide freely along the bar axis (x) instead of "
                         "sticking; normal and circumferential components stay held")
    ap.add_argument("--axial-constraint", choices=("face_rx", "face_fixed"), default="face_rx",
                    dest="axial_constraint",
                    help="how the held end (x = x_min face) is restrained.  face_rx: ux = 0 on "
                         "the face, one pin node uy = uz = 0, one node uy = 0 (rotation lock on "
                         "two nodes; they yield in a multi-hit series and the bar rolls about its "
                         "axis).  face_fixed: ux = uy = uz = 0 on every node of the face (rigid "
                         "grip; use it for a multi-hit series)")
    ap.add_argument("--fbar-form", choices=("legacy", "stress"), default="legacy",
                    help="pairing of the F-bar stress with the shape-function gradients; "
                         "see ThermalMechanical.fbar_form")
    ap.add_argument("--hit-reset", choices=("none", "stress", "full"), default="none",
                    help="what the reheat between two hits does to the state.  'none': the "
                         "state is carried (previous behaviour).  'full': every hit starts "
                         "stress-free and unhardened on the current shape, which becomes the "
                         "new reference configuration.  'stress': the same, but the "
                         "accumulated plastic strain (hardening) is kept.  The strokes of "
                         "one hit always share their state.")
    ap.add_argument("--hit-smooth", choices=("none", "interior", "surface"), default="none",
                    help="relocate the nodes at the reheat between two hits (needs "
                         "--hit-reset full, so that no state has to be mapped): 'interior' "
                         "keeps every boundary node, so the shape is unchanged; 'surface' "
                         "also slides boundary nodes on the old boundary.  lib/mesh_smooth.py")
    ap.add_argument("--hit-remesh", choices=("none", "ogrid"), default="none",
                    help="rebuild the mesh at the start of EVERY hit (the first included) on the "
                         "current shape: a fresh O-grid x axial-sweep HEX8 mesh fitted to the "
                         "deformed surface, with fine axial layers at the imprint walls and at "
                         "the walls of the coming hit (lib/remesh_ogrid.py).  Needs --hit-reset "
                         "full (no state to map) and HEX8; replaces --hit-smooth")
    ap.add_argument("--remesh-h-fine", type=float, default=0.7,
                    help="axial layer thickness inside the fine zones, mm")
    ap.add_argument("--remesh-h-coarse", type=float, default=0.0,
                    help="axial layer thickness elsewhere, mm (0: the original mesh's spacing)")
    ap.add_argument("--remesh-slope-deg", type=float, default=0.0,
                    help="a surface facet steeper than this in x marks a fine zone; 0 (default) "
                         "refines only the coming hit's walls: fine layers on an existing 80 deg "
                         "wall are skewed to SJ ~0.2 whatever their thickness (r2 vs r2s0)")
    ap.add_argument("--remesh-wall-zone", type=float, default=3.5,
                    help="fine zone beyond each wall of the coming hit's band, mm")
    ap.add_argument("--remesh-grade", type=float, default=1.3)
    ap.add_argument("--remesh-core", type=float, default=0.5)
    ap.add_argument("--remesh-surf-bias", type=float, default=1.0,
                    help="> 1 thins the ring layers towards the surface")
    ap.add_argument("--stroke-relocate", choices=("none", "interior", "surface"), default="none",
                    help="relocate the nodes INSIDE a compression stroke when the quality "
                         "guard stops it: the last committed shape becomes a stress-free "
                         "reference, the mesh is smoothed (lib/mesh_smooth.py, 'surface' lets "
                         "the boundary nodes near the failing cells slide on the old surface), "
                         "the hardening alpha is re-sampled onto the new Gauss points "
                         "(lib/state_transfer.py), the dies back off by the largest penetration "
                         "of the relocated nodes and the stroke continues to the same target. "
                         "Force-based contact and --hit-reset full only.")
    ap.add_argument("--relocate-max", type=int, default=6,
                    help="relocations allowed per stroke")
    ap.add_argument("--relocate-min-advance", type=float, default=0.1,
                    help="mm of die travel a continued stroke must make before the next "
                         "relocation; less than this ends the run as a distortion stop "
                         "(not applied to the first relocation of a stroke)")
    ap.add_argument("--relocate-radius", type=float, default=6.0,
                    help="surface mode: boundary nodes within this distance (mm) of the "
                         "cells below --relocate-quality may slide")
    ap.add_argument("--relocate-quality", type=float, default=0.4,
                    help="cells below this scaled Jacobian define the region to repair")
    ap.add_argument("--relocate-power", type=float, default=4.0)
    ap.add_argument("--relocate-iters", type=int, default=800)
    ap.add_argument("--relocate-restarts", type=int, default=4)
    ap.add_argument("--relocate-crease-deg", type=float, default=50.0)
    ap.add_argument("--relocate-test-at", type=float, default=None,
                    help="TEST: force one relocation in the first compression stroke of the "
                         "first hit of this run when the stroke scale reaches this value "
                         "(no quality stop needed)")
    ap.add_argument("--relocate-test-nomove", action="store_true",
                    help="TEST: skip the smoothing at a relocation and keep the committed "
                         "node positions, so the rebase + state transfer must reproduce "
                         "the uninterrupted run exactly")
    ap.add_argument("--stroke-remesh", choices=("none", "ogrid"), default="none",
                    help="remesh INSIDE a compression stroke when the quality guard stops it and "
                         "the relocation (smoothing) cannot repair the mesh (its min SJ stays below "
                         "--relocate-min-quality): a fresh O-grid mesh is fitted to the committed "
                         "shape (lib/remesh_ogrid.py, fine layers at this hit's walls), the state "
                         "(alpha, be_bar, J_ref) is re-sampled at the new Gauss points and the "
                         "friction memory is mapped to the nearest new surface node with its "
                         "total conserved.  Needs --hit-remesh ogrid and --stroke-relocate")
    ap.add_argument("--relocate-min-quality", type=float, default=0.2,
                    help="a relocation whose smoothed mesh stays below this min scaled Jacobian "
                         "is discarded in favour of an in-stroke remesh (--stroke-remesh)")
    ap.add_argument("--clamp-min-quality", type=float, default=0.1,
                    help="in-stroke remesh: after new surface nodes inside the die are put back "
                         "on the die face, the min scaled Jacobian must stay at least this (or the "
                         "remeshed value if that is lower); otherwise the clamp limit is halved "
                         "(0.3 h_wall -> /2 -> /4) and finally the clamp is skipped, the rest "
                         "being left to the die back-off.  0 = old behaviour (one limit, no check)")
    ap.add_argument("--stroke-remesh-first", action="store_true",
                    help="with --stroke-remesh: remesh at every guard stop without trying the "
                         "relocation first")
    ap.add_argument("--remesh-wall-aspect", type=float, default=0.7,
                    help="in-stroke remesh: axial layer thickness at the current imprint wall as "
                         "a fraction of the wall height (a sweep mesh sees the near-vertical wall "
                         "as a step between two stations whose cell has SJ dx/sqrt(dx^2+H^2): 0.7 "
                         "gives ~0.57; fine 0.7 mm layers gave 0.08 at H 3.4 mm).  The wall is a "
                         "slanted facet until the next hit's remesh.  Floor: --remesh-h-fine")
    ap.add_argument("--remesh-n", type=int, default=0,
                    help="O-grid cells per quarter arc for every remesh of this run (4n nodes "
                         "around; 0: keep the template's, 11 on the h28 meshes)")
    ap.add_argument("--smooth-power", type=float, default=4.0,
                    help="exponent of the shape metric in the smoothing energy; larger "
                         "values weigh the worst corners more")
    ap.add_argument("--sig0", type=float, default=75.0)
    ap.add_argument("--Q", type=float, default=0.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--max-dt", type=float, default=0.08)
    ap.add_argument("--tag", default="cold")
    ap.add_argument("--save-vtk", action="store_true")
    ap.add_argument("--rho-source", choices=("achieved", "command"),
                    default="achieved",
                    help="Closure radius to drive the platen to.  'achieved' "
                         "reads rho_achieved.json, which inverts the machine's "
                         "own position-vs-rho line to recover what each hit "
                         "actually reached; the press saturates at 110.07 kN "
                         "and hit 7 stalled 0.60 mm short of its command, so "
                         "simulating the command asks for a bite the bar never "
                         "saw.  'command' uses the action file verbatim.")
    ap.add_argument("--strokes", choices=("one", "two"), default="one",
                    help="'two' reproduces what the press actually does: each "
                         "hit closes part way, the ram backs off past the bar "
                         "surface, then it closes again onto the command.  "
                         "Only the contact model can tell the difference -- it "
                         "is fully sticking, so a node that touches the die "
                         "cannot slide again until the constraint is released, "
                         "and a single monotonic closure never releases it.  "
                         "Radii come from stroke_rho.json.")
    ap.add_argument("--hit-positions", default=None, dest="hit_positions",
                    help="JSON file {'hits': [{'hit': k, 'phi': deg, 'z': mm}, ...]} whose angle "
                         "and axial position replace the commanded phi and z of those hits IN "
                         "MEMORY (e.g. the positions measured from the scans); the action file "
                         "and every data file stay as they are.  Default: the commanded values.")
    ap.add_argument("--stroke-target", choices=("press", "command"), default="press",
                    dest="stroke_target",
                    help="closure radii of the two strokes (--strokes two).  'press' "
                         "(previous behaviour): stroke_rho.json as it is, i.e. each "
                         "stroke's deepest ram position through the series' "
                         "position-vs-rho line.  'command': the second stroke closes "
                         "on the commanded rho of the action file; the first stroke and "
                         "the retract keep their offsets from the second stroke in "
                         "stroke_rho.json.  On roll3.1 the line overshoots the deep "
                         "hits: the post-hit scans give 9.35-9.55 mm for hit 8 against "
                         "8.48 from the line and 9.30 commanded (hit 10: 9.72 / 8.77 / "
                         "9.45).  No data file is changed; the radii are computed here.")
    ap.add_argument("--single", default=None,
                    help="phi,z,rho -- run ONE synthetic hit on the virgin bar "
                         "instead of the recorded sequence.  Used to separate "
                         "the effect of axial position from that of roll angle: "
                         "hits 1 and 2 close on the same local geometry yet come "
                         "out a factor of three apart, and the recorded sequence "
                         "varies both at once so it cannot say which matters.")
    args = ap.parse_args()

    out = OUT / f"sim_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    if args.hit_smooth != "none" and args.hit_reset != "full":
        raise SystemExit("--hit-smooth moves the quadrature points, so no state can be "
                         "carried across it: it needs --hit-reset full")
    smooth_log = []
    remesh_log = []
    if args.hit_remesh != "none" and (args.hit_reset != "full" or args.ele_type != "HEX8"):
        raise SystemExit("--hit-remesh needs --hit-reset full and --ele-type HEX8")
    if args.stroke_relocate != "none" and (args.hit_reset != "full" or args.contact != "penalty"):
        raise SystemExit("--stroke-relocate needs --hit-reset full and --contact penalty")
    if args.stroke_remesh != "none" and (args.hit_remesh == "none" or args.stroke_relocate == "none"):
        raise SystemExit("--stroke-remesh needs --hit-remesh ogrid and --stroke-relocate")
    relocate_log = []
    skip_hits = {int(v) for v in args.skip_hits.split(",") if v.strip()}
    if args.friction_mu is not None and not args.axial_slip:
        args.axial_slip = True
        print("  [friction] --friction-mu given: --axial-slip switched on", flush=True)

    # Record exactly which code produced this run: a sha256 of every .py file
    # of the code tree it was launched from, plus the full command line.
    import hashlib
    _code_root = Path(__file__).resolve().parent
    _h = hashlib.sha256()
    for _f in sorted(_code_root.rglob("*.py")):
        if "__pycache__" in _f.parts:
            continue
        _h.update(str(_f.relative_to(_code_root)).encode()); _h.update(_f.read_bytes())
    code_info = dict(code_dir=str(_code_root), code_sha256=_h.hexdigest(), argv=sys.argv)
    (out / "run_info.json").write_text(json.dumps(code_info, indent=2))
    print(f"  [code] {code_info['code_dir']}  sha256 {code_info['code_sha256'][:16]}", flush=True)

    mio = meshio.read(args.mesh)
    ele_type = args.ele_type
    if ele_type == "TET10":
        # The package evaluates the F-bar centre gradients at the hexahedron's
        # reference centre, outside a tetrahedron; harmless on constant-strain
        # TET4, wrong on TET10.  Must run before the Problem builds its FEs.
        # See lib/fbar_center.py.
        from lib.fbar_center import install_centroid_center
        install_centroid_center()
        print("  [fbar] centre gradients evaluated at the reference-cell centroid", flush=True)
    cell_type = get_meshio_cell_type(ele_type)
    # cells_out: the connectivity in meshio/VTK order, used for every file
    # written here and for the surface; the package gets its own order.
    cells_out = onp.asarray(mio.cells_dict[cell_type])
    if ele_type == "HEX8":
        from lib.hex8 import to_package, outer_surface_nodes_hex
        mesh = Mesh(mio.points, to_package(cells_out))
        pts = onp.array(mesh.points)
        surf = outer_surface_nodes_hex(pts, cells_out)
        print("  [hex8] cells permuted into jax_fem_checkpoint's HEX8 node order", flush=True)
    else:
        mesh = Mesh(mio.points, cells_out)
        pts = onp.array(mesh.points)
        surf = outer_surface_nodes_any(pts, cells_out)
    pts0 = pts.copy()        # the original mesh: target cell shapes for --hit-smooth
    remesh_template = None
    if args.hit_remesh != "none":
        from lib.remesh_ogrid import template_from_original, remesh_ogrid
        remesh_template = template_from_original(pts, cells_out)
        _xs0 = onp.unique(onp.round(pts[:, 0] / 1e-6)) * 1e-6
        h_axial0 = float(onp.median(onp.diff(_xs0)))
        if args.remesh_h_coarse <= 0:
            args.remesh_h_coarse = h_axial0
        print(f"  [remesh] O-grid template n={remesh_template['n']} m={remesh_template['m']}, "
              f"{len(_xs0)} stations of {h_axial0:.2f} mm; remesh before every hit with layers "
              f"{args.remesh_h_fine:.2f} mm (fine) / {args.remesh_h_coarse:.2f} mm (coarse)", flush=True)
    H = float(pts[:, 0].max())
    R = float(onp.hypot(pts[:, 1], pts[:, 2]).max())
    print(f"mesh {len(pts)} nodes / {len(mesh.cells)} {ele_type} cells, {len(surf)} surface nodes")
    from lib.mesh_quality import min_edge_length
    h_min = min_edge_length(pts, cells_out)
    print(f"  [mesh] shortest element edge {h_min:.3f} mm", flush=True)
    print(f"6063 isothermal {T_ISO_C:.0f} C [{SERIES}]: sig0={args.sig0} Q={args.Q} b={args.b}, "
          f"die band {args.band} mm\n", flush=True)

    def T_uniform(point):
        return T_ROOM

    def no_location(p):
        return False

    ThermalMechanical.sig0_override = args.sig0
    ThermalMechanical.fbar_form = args.fbar_form
    print(f"  [fbar] internal force form: {args.fbar_form}", flush=True)
    if args.fast_linear:
        if args.linear_solver != "mumps":
            raise SystemExit("--fast-linear needs --linear-solver mumps")
        from lib.fast_linear import install as _install_fast_linear
        _install_fast_linear()
        print("  [fast] fast linear algebra on (cached CSR pattern, MUMPS analysis reuse, "
              "residual reuse in the line search)", flush=True)
    problem = ThermalMechanical(
        mesh=[mesh, mesh], vec=[3, 1], dim=3,
        ele_type=[ele_type, ele_type], gauss_order=[2, 2],
        dirichlet_bc_info=[None, None],
        location_fns=[no_location, no_location])
    problem.custom_init()
    friction = None
    die_contact = None
    if args.contact == "penalty":
        from lib.die_contact import DieProfile, NodalDieContact
        profile = DieProfile(args.die_profile, 0.5 * args.band, fillet_r=args.die_fillet_r,
                             flank_deg=args.die_flank_deg, height=args.die_height,
                             sigmoid_w=args.die_sigmoid_w, soft_w=args.die_soft_edge)
        k_n = args.contact_penalty * problem.E * h_min
        die_contact = NodalDieContact(pts, surf, profile, k_n,
                                      mu=(args.friction_mu or 0.0), delta=args.friction_delta,
                                      normal=args.die_normal)
        problem.die_contact = die_contact
        print(f"  [contact] force-based ({args.die_normal} normal): {profile.describe()}; penalty k_n = "
              f"{k_n:.3e} N/mm per node ({args.contact_penalty} x E h_min); Coulomb "
              f"mu = {args.friction_mu or 0.0} in both tangential directions, "
              f"delta = {args.friction_delta} mm; {len(surf)} candidate nodes", flush=True)
        if args.edge_radius > 0:
            raise SystemExit("--edge-radius belongs to the dirichlet contact; use --die-profile")
    elif args.friction_mu is not None:
        from lib.friction import NodalAxialFriction
        friction = NodalAxialFriction(len(pts), args.friction_mu, args.friction_delta)
        print(f"  [friction] axial Coulomb friction mu = {args.friction_mu}, "
              f"delta = {args.friction_delta} mm", flush=True)
    problem.sig0 = args.sig0
    problem.Q = args.Q
    problem.b = args.b

    # Contact candidates are the lateral boundary nodes of the mesh (`surf`,
    # from the face topology; the same set bite_for measures the travel on),
    # NOT a radial shell.  The earlier test r > 0.85 R took the outer 2.9 mm:
    # on the 2.0 / 1.4 / 1.0 mm meshes that is one to three layers of
    # interior nodes (4752 / 16920 / 51352 of them), and compute_contact
    # captured such a node as soon as the surface layer above it had thinned
    # to within tol_contact of the die plane, then pinned it onto the plane --
    # the surface element went to zero thickness (h14 stopped in hit 1,
    # h20 in hit 2, each with the sub-surface node exactly on the die plane).
    # Mesh A carries 940 shell-interior nodes 1.9 mm deep, uncaptured so far.
    surf_mask = onp.zeros(len(pts), dtype=bool)
    surf_mask[onp.asarray(surf, dtype=int)] = True

    def surface_test(p):
        if len(p) != len(pts):
            raise ValueError("the surface mask is defined on the full node array")
        return surf_mask

    bc_builder = partial(build_cylinder_press_bcs, surface_mask_fn=surface_test,
                         axial_constraint=args.axial_constraint,
                         band_config=args.band_config,
                         edge_radius=args.edge_radius,
                         axial_slip=(args.axial_slip and die_contact is None),
                         die_contact=die_contact)

    sol_u = np.zeros((len(pts), 3))
    sol_dT = np.full((len(pts), 1), T_ROOM)
    int_vars = problem.internal_vars
    log = []
    strokes_log = []
    t0 = time.time()

    # ----- distortion monitor (see lib/mesh_quality.py) -----
    quality = QualityMonitor(pts, cells_out,
                             stop_below=(args.min_scaled_jacobian
                                         if args.min_scaled_jacobian != 0 else -onp.inf),
                             warn_below=args.warn_scaled_jacobian)
    print(f"  [quality] die travel per step capped at {args.max_bite_step} x h_min "
          f"= {args.max_bite_step * h_min:.3f} mm", flush=True)
    # minimum scaled Jacobian of the last COMMITTED state, for the drop test
    q_last = [None]

    # --relocate-test-at: fires once, as a DistortionStop raised from the
    # acceptance hook (the stepper does not catch it), on the committed state
    relocate_test = [args.relocate_test_at]

    def accept_trial(u_trial, step, scale, dt, where="trial"):
        """Pre-commit acceptance of a converged trial state (stepper accept_fn)."""
        if relocate_test[0] is not None and scale >= relocate_test[0] and step > 1:
            relocate_test[0] = None
            raise DistortionStop(dict(
                message=f"--relocate-test-at {args.relocate_test_at}: forced relocation",
                cell=-1, min_scaled_jacobian=float("nan"), trial_sol_u=onp.asarray(u_trial),
                refusals=0, refusals_total=0, dt=float(dt), scale=float(scale)))
        q, rep = quality.evaluate(u_trial, f"{where} step {step} (scale {scale:.4f})",
                                  record=False)
        if rep is not None:
            return False, rep
        qmin = float(q.min())
        if args.quality_max_drop > 0 and q_last[0] is not None \
                and q_last[0] - qmin > args.quality_max_drop:
            rep = quality.report(q, pts + onp.asarray(u_trial),
                                 f"{where} step {step} (scale {scale:.4f})")
            rep["message"] = (f"min scaled Jacobian fell from {q_last[0]:.3f} to "
                              f"{qmin:.3f} in one step (cell {rep['cell']}), more than "
                              f"--quality-max-drop {args.quality_max_drop}")
            return False, rep
        return True, None

    # ----- per-hit checkpoints -----
    ckpt_dir = out / "checkpoints"
    mesh_sig = dict(n_points=int(len(pts)), n_cells=int(len(cells_out)),
                    ele_type=ele_type, cells_sum=int(onp.asarray(cells_out, dtype=onp.int64).sum()))

    def save_checkpoint(hits_done):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        f = ckpt_dir / f"state_hit{hits_done:02d}.npz"
        iv = {f"int_var_{i}": onp.asarray(v) for i, v in enumerate(int_vars)}
        meta = dict(hits_done=hits_done, series=SERIES, mesh=str(args.mesh), mesh_sig=mesh_sig,
                    sig0=args.sig0, Q=args.Q, b=args.b, band=args.band,
                    fbar_form=args.fbar_form, hit_reset=args.hit_reset,
                    hit_smooth=args.hit_smooth, smooth_power=args.smooth_power,
                    stroke_relocate=args.stroke_relocate, relocate_log=relocate_log,
                    band_config=args.band_config, strokes=args.strokes,
                    stroke_target=args.stroke_target, hit_positions=args.hit_positions,
                    rho_source=args.rho_source, edge_radius=args.edge_radius,
                    axial_slip=args.axial_slip, friction_mu=args.friction_mu,
                    friction_delta=args.friction_delta, contact=args.contact,
                    die_profile=args.die_profile, die_fillet_r=args.die_fillet_r,
                    die_flank_deg=args.die_flank_deg, die_height=args.die_height,
                    die_sigmoid_w=args.die_sigmoid_w, die_soft_edge=args.die_soft_edge,
                    die_normal=args.die_normal,
                    contact_penalty=args.contact_penalty,
                    axial_constraint=args.axial_constraint,
                    code=code_info, skip_hits=sorted(skip_hits), log=log, strokes_log=strokes_log)
        if args.hit_reset != "none":
            iv["ref_points"] = onp.asarray(pts)
        if remesh_template is not None:
            # the connectivity changes at every remesh: store it with the state
            iv["cells"] = onp.asarray(cells_out)
            iv["surf"] = onp.asarray(surf)
            iv["lateral_nodes"] = onp.asarray(remesh_template["lateral_nodes"])
            meta["remesh_nm"] = [int(remesh_template["n"]), int(remesh_template["m"])]
            meta["remesh_log"] = remesh_log
        onp.savez(f, sol_u=onp.asarray(sol_u), sol_dT=onp.asarray(sol_dT),
                  n_int_vars=len(int_vars), meta=json.dumps(meta), **iv)
        print(f"    [checkpoint] state after hit {hits_done} -> {f}", flush=True)

    # geometry of every accepted Newton iterate, for the [newton] line
    from lib.mesh_quality import scaled_jacobian as _sj_fn
    problem.newton_monitor = lambda u_: f"min SJ {float(_sj_fn(pts + u_, cells_out).min()):.4f}"

    def rebase_model(points, new_cells=None, new_surf=None):
        """Everything that is built on the reference coordinates, rebuilt on
        `points`: mesh, problem, die contact, distortion monitor and BC
        builder.  Same connectivity unless `new_cells` is given (a remesh):
        then the connectivity, the lateral surface set, the smoothing target
        and the checkpoint mesh signature are replaced as well.  h_min (and
        with it the penalty stiffness) stays that of the original mesh.  The
        lateral surface set is topological: the free end face is no longer
        plane, so it must not be re-derived from coordinates."""
        nonlocal cells_out, surf, surf_mask, pts0, mesh_sig
        if new_cells is not None:
            cells_out = onp.asarray(new_cells)
            surf = onp.asarray(new_surf, dtype=int)
            surf_mask = onp.zeros(len(points), dtype=bool)
            surf_mask[surf] = True
            pts0 = onp.array(points, dtype=float)
            mesh_sig = dict(n_points=int(len(points)), n_cells=int(len(cells_out)),
                            ele_type=ele_type, cells_sum=int(onp.asarray(cells_out, dtype=onp.int64).sum()))
        m = Mesh(points, to_package(cells_out)) if ele_type == "HEX8" else Mesh(points, cells_out)
        p = onp.array(m.points)
        prob = ThermalMechanical(
            mesh=[m, m], vec=[3, 1], dim=3,
            ele_type=[ele_type, ele_type], gauss_order=[2, 2],
            dirichlet_bc_info=[None, None],
            location_fns=[no_location, no_location])
        prob.custom_init()
        dc = None
        if args.contact == "penalty":
            dc = NodalDieContact(p, surf, profile, k_n,
                                 mu=(args.friction_mu or 0.0), delta=args.friction_delta,
                                 normal=args.die_normal)
            prob.die_contact = dc
        prob.sig0, prob.Q, prob.b = args.sig0, args.Q, args.b
        prob.newton_monitor = lambda u_: f"min SJ {float(_sj_fn(p + u_, cells_out).min()):.4f}"
        qm = QualityMonitor(p, cells_out,
                            stop_below=(args.min_scaled_jacobian
                                        if args.min_scaled_jacobian != 0 else -onp.inf),
                            warn_below=args.warn_scaled_jacobian)
        bcb = partial(build_cylinder_press_bcs, surface_mask_fn=surface_test,
                      axial_constraint=args.axial_constraint, band_config=args.band_config,
                      edge_radius=args.edge_radius,
                      axial_slip=(args.axial_slip and dc is None), die_contact=dc)
        return (m, p, prob, dc, qm, bcb, float(p[:, 0].max()),
                float(onp.hypot(p[:, 1], p[:, 2]).max()))

    k_start = 0
    if args.resume_from is not None:
        d = onp.load(args.resume_from, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if "cells" in d.files:
            if args.hit_remesh == "none":
                raise SystemExit("--resume-from: the checkpoint carries a remeshed connectivity; "
                                 "resume with --hit-remesh")
            remesh_template = dict(n=int(meta["remesh_nm"][0]), m=int(meta["remesh_nm"][1]),
                                   lateral_nodes=onp.asarray(d["lateral_nodes"]))
            remesh_log = meta.get("remesh_log", [])
            print(f"  [resume] connectivity taken from the checkpoint ({len(d['cells'])} cells, "
                  f"remeshed run)", flush=True)
        elif meta["mesh_sig"] != mesh_sig:
            raise SystemExit(f"--resume-from {args.resume_from}: checkpoint mesh "
                             f"{meta['mesh_sig']} does not match this mesh {mesh_sig}")
        # checkpoints written before the series was recorded carry none; they
        # live in the series' own output directory, so take that as the series
        ck_series = meta.get("series") or (SERIES if Path(OUT).resolve() in Path(args.resume_from).resolve().parents else None)
        if ck_series != SERIES:
            raise SystemExit(f"--resume-from: checkpoint is for {ck_series}, "
                             f"this run is {SERIES}")
        changed = {k: (meta.get(k), getattr(args, k)) for k in
                   ("sig0", "Q", "b", "band", "band_config", "strokes", "rho_source",
                    "edge_radius", "axial_slip", "friction_mu") if meta.get(k, False) != getattr(args, k)}
        if meta.get("fbar_form", "legacy") != args.fbar_form:
            changed["fbar_form"] = (meta.get("fbar_form", "legacy"), args.fbar_form)
        # checkpoints written before --stroke-target existed used 'press'
        if meta.get("stroke_target", "press") != args.stroke_target:
            changed["stroke_target"] = (meta.get("stroke_target", "press"), args.stroke_target)
        if meta.get("hit_positions") != args.hit_positions:
            changed["hit_positions"] = (meta.get("hit_positions"), args.hit_positions)
        # checkpoints written before --axial-constraint existed used face_rx
        if meta.get("axial_constraint", "face_rx") != args.axial_constraint:
            changed["axial_constraint"] = (meta.get("axial_constraint", "face_rx"),
                                           args.axial_constraint)
        if changed:
            print(f"  [resume] NOTE settings differ from the checkpointed run "
                  f"(checkpoint, now): {changed}", flush=True)
        sol_u = np.asarray(d["sol_u"])
        sol_dT = np.asarray(d["sol_dT"])
        if float(np.max(np.abs(sol_dT - T_ROOM))) > 1e-9:
            print(f"  [resume] WARNING checkpoint from a non-isothermal run (T up to "
                  f"{float(np.max(sol_dT)):.1f} C): T reset to {T_ROOM:.0f} C, but its stress "
                  f"state includes the old thermal pressure; rerun from hit 1 for final results",
                  flush=True)
            sol_dT = np.full_like(sol_dT, T_ROOM)
        int_vars = [np.asarray(d[f"int_var_{i}"]) for i in range(int(d["n_int_vars"]))]
        if len(int_vars) == 8:
            # checkpoint written before the J_ref slot existed: its reference
            # was treated as stress-free, i.e. J_ref = 1
            int_vars.insert(6, np.ones(int_vars[2].shape))
            print("  [resume] checkpoint predates the J_ref internal variable: J_ref = 1 assumed",
                  flush=True)
        log = meta["log"]
        strokes_log = meta["strokes_log"]
        relocate_log = meta.get("relocate_log", [])
        k_start = int(meta["hits_done"])
        if "ref_points" in d.files:
            # checkpoint of a --hit-reset run: its displacement is measured from
            # the reference configuration of its last hit
            if "cells" in d.files:
                (mesh, pts, problem, die_contact, quality, bc_builder, H, R) = rebase_model(
                    onp.asarray(d["ref_points"]), onp.asarray(d["cells"]), onp.asarray(d["surf"]))
            else:
                (mesh, pts, problem, die_contact, quality, bc_builder, H, R) = rebase_model(
                    onp.asarray(d["ref_points"]))
            print("  [resume] reference configuration taken from the checkpoint "
                  "(--hit-reset run)", flush=True)
        if meta.get("hit_reset", "none") != args.hit_reset:
            print(f"  [resume] NOTE hit_reset differs (checkpoint {meta.get('hit_reset', 'none')}, "
                  f"now {args.hit_reset})", flush=True)
        print(f"  [resume] state after hit {k_start} from {args.resume_from}; "
              f"continuing with hit {k_start + 1}", flush=True)
        if (args.hit_smooth != "none" or args.hit_remesh != "none") and args.hit_reset == "full":
            # the mesh is smoothed / rebuilt before the next hit starts; report only
            q_res, rep_res = quality.evaluate(sol_u, f"resumed state after hit {k_start}")
            if rep_res is not None:
                print(f"  [resume] {rep_res['message']} -- smoothing follows", flush=True)
        else:
            quality.check(sol_u, f"resumed state after hit {k_start}")

    def distortion_stop(e, k, label):
        rep_ = dict(e.report, hit=k + 1, stroke=label,
                    last_checkpoint=(f"state_hit{k:02d}.npz" if k > 0 else None))
        from lib.mesh_quality import scaled_jacobian
        u_trial = rep_.pop("trial_sol_u", None)
        if u_trial is not None:
            # the refused trial state (never committed): the last committed
            # state goes to distortion_stop.vtu as before
            x_tr = pts + onp.asarray(u_trial)
            meshio.Mesh(x_tr, {cell_type: cells_out},
                        point_data={"Displacement": onp.asarray(u_trial)},
                        cell_data={"scaled_jacobian": [scaled_jacobian(x_tr, cells_out)]}
                        ).write(out / "distortion_trial.vtu")
            rep_["trial_field"] = str(out / "distortion_trial.vtu")
        (out / "distortion_stop.json").write_text(json.dumps(rep_, indent=2))
        x_def = pts + onp.asarray(sol_u_last[0])
        meshio.Mesh(x_def, {cell_type: cells_out},
                    point_data={"Displacement": onp.asarray(sol_u_last[0])},
                    cell_data={"scaled_jacobian": [scaled_jacobian(x_def, cells_out)]}
                    ).write(out / "distortion_stop.vtu")
        print(f"    [STOP] {e}\n           cell centre (reference) x={rep_['cell_ref_centre_mm'][0]:.2f} mm, "
              f"r={rep_['cell_ref_radius_mm']:.2f} mm, angle={rep_['cell_ref_angle_deg']:.1f} deg; "
              f"{rep_['n_cells_inverted']} inverted, {rep_['n_cells_below_stop']} below the "
              f"threshold.  Report: {out / 'distortion_stop.json'}, field: "
              f"{out / 'distortion_stop.vtu'}", flush=True)
    sol_u_last = [sol_u]

    actions = load_actions(args.n_hits)
    if args.hit_positions:
        hp = json.loads(Path(args.hit_positions).read_text())
        over = {int(r["hit"]): r for r in hp["hits"]}
        print(f"  [positions] angle and axial position from {args.hit_positions} "
              f"({hp.get('source', 'no source given')}); commanded values kept as phi_command / z_command",
              flush=True)
        for i, a in enumerate(actions):
            r = over.get(i + 1)
            a["phi_command"], a["z_command"] = a["phi"], a["z"]
            if r is not None:
                a["phi"], a["z"] = float(r["phi"]), float(r["z"])
            print(f"    hit {i + 1:2d}: phi {a['phi_command']:6.1f} -> {a['phi']:6.1f} deg, "
                  f"z {a['z_command']:6.2f} -> {a['z']:6.2f} mm", flush=True)
    if args.rho_source == "achieved":
        f = OUT / "rho_achieved.json"
        if f.exists():
            eff = {r["hit"]: r for r in json.loads(f.read_text())["hits"]}
            for i, a in enumerate(actions):
                r = eff.get(i + 1)
                if r is not None:
                    a["rho_command"] = a["rho"]
                    a["rho"] = r["rho_achieved"]
                    if abs(r["shortfall_mm"]) > 0.1:
                        print(f"  [rho] hit {i+1}: press stalled at "
                              f"{r['peak_kn']:.1f} kN, {r['shortfall_mm']:+.3f} mm "
                              f"short -- driving to {a['rho']:.3f} not "
                              f"{a['rho_command']:.2f}", flush=True)
        else:
            print(f"  [rho] {f} missing, falling back to commanded rho", flush=True)
    if args.single:
        phi, z, rho = (float(v) for v in args.single.split(","))
        actions = [dict(phi=phi, z=z, rho=rho)]

    # Each action becomes a list of (rho_target, label).  A rho ABOVE the bar
    # surface is a release: no candidate node can reach the platen, the contact
    # set empties, and the body relaxes against the rigid-body constraints
    # alone -- which is the whole point, since that is the moment the sticking
    # tangential lock is let go.
    if args.strokes == "two":
        f = OUT / "stroke_rho.json"
        st = {r["hit"]: r for r in json.loads(f.read_text())}
        plans = []
        for i in range(len(actions)):
            r = st.get(i + 1)
            if r is None:
                plans.append([(actions[i]["rho"], "single")])
                continue
            r1, r_ret, r2 = r["rho_stroke1"], r["rho_retract"], r["rho_stroke2"]
            if args.stroke_target == "command":
                # second stroke on the command, the others keep their offsets
                # from it; the files are read as they are
                cmd = float(actions[i].get("rho_command", actions[i]["rho"]))
                off = cmd - r2
                r1, r_ret, r2 = r1 + off, r_ret + off, cmd
                print(f"  [strokes] hit {i+1}: closure on the command {cmd:.3f} mm "
                      f"(stroke_rho.json {r['rho_stroke1']:.3f} / {r['rho_stroke2']:.3f}, "
                      f"now {r1:.3f} / {r2:.3f}, shift {off:+.3f} mm)", flush=True)
            plans.append([(r1, "stroke 1"),
                          (r_ret, "release"),
                          (r2, "stroke 2")])
            if args.final_release:
                # the press lifts off after the second stroke as well
                plans[-1].append((r_ret, "release 2"))
        print(f"  [strokes] two-stroke kinematics from {f.name}, closure targets: "
              f"{args.stroke_target}", flush=True)
    else:
        plans = [[(a["rho"], "single")] for a in actions]

    failed = False          # read by the exit status even when no hit is left to run
    for k, a in enumerate(actions):
        if k < k_start:
            continue
        if (k + 1) in skip_hits:
            print(f"--- hit {k+1} skipped (--skip-hits)", flush=True)
            continue
        # Sign of the roll angle.  build_cylinder_press_bcs rotates with
        # scipy's from_euler("x", psi) and maps global to press frame as R @ p,
        # so its platen lands at bar angle theta = -psi.  The rig's own scans
        # put the flat at theta = +phi: over all twelve hits the angular
        # position of the radius drop in roll1.h5 matches +phi to about 3 deg
        # (hit 3: 62.5 measured against 62.8; hit 9: 177.5 against 177.7),
        # while -phi is off by up to 54 deg.  So the BC must be driven with
        # -phi.  On a round bar the mirror is invisible -- the bar is
        # axisymmetric -- but across a twelve-hit sequence it puts every hit
        # after the first on the wrong side of the flats already there.
        phi_sim = -a["phi"]
        print(f"--- hit {k+1}  phi={a['phi']:.1f} z={a['z']:.2f} "
              f"rho={a['rho']:.2f}", flush=True)
        zones_hit = None
        if args.hit_remesh != "none":
            # Remesh on the current shape (the virgin bar before hit 1 as well:
            # the walls of the first band need the fine layers too).  Reheat
            # reset: the new mesh starts stress-free and unhardened, u = 0.
            from lib.mesh_quality import scaled_jacobian as _sj
            x_cur = pts + onp.asarray(sol_u)
            wz, hb = args.remesh_wall_zone, 0.5 * args.band
            zones_next = [(a["z"] - hb - wz, a["z"] - hb + 0.5), (a["z"] + hb - 0.5, a["z"] + hb + wz)]
            zones_hit = zones_next
            if args.remesh_n > 0 and int(remesh_template["n"]) != args.remesh_n:
                print(f"  [remesh] circumferential resolution {4 * int(remesh_template['n'])} -> "
                      f"{4 * args.remesh_n} nodes (--remesh-n)", flush=True)
                remesh_template = dict(remesh_template, n=int(args.remesh_n))
            meshio.Mesh(x_cur, {cell_type: cells_out},
                        cell_data={"scaled_jacobian": [_sj(x_cur, cells_out)]}
                        ).write(out / f"mesh_hit{k+1:02d}_before_remesh.vtu")
            x_new, cells_new, surf_new, rrep = remesh_ogrid(
                x_cur, cells_out, remesh_template, args.remesh_h_coarse, args.remesh_h_fine,
                core=args.remesh_core, surf_bias=args.remesh_surf_bias, zones_extra=zones_next,
                grade=args.remesh_grade, slope_deg=args.remesh_slope_deg)
            remesh_template = rrep.pop("template")
            remesh_log.append(dict(rrep, before_hit=k + 1, zones_next=[list(map(float, z)) for z in zones_next]))
            (out / "remesh_log.json").write_text(json.dumps(remesh_log, indent=2))
            meshio.Mesh(x_new, {cell_type: cells_new},
                        cell_data={"scaled_jacobian": [_sj(x_new, cells_new)]}
                        ).write(out / f"mesh_hit{k+1:02d}_remesh.vtu")
            (mesh, pts, problem, die_contact, quality, bc_builder, H, R) = rebase_model(
                x_new, cells_new, surf_new)
            sol_u = np.zeros((len(pts), 3))
            sol_dT = np.full((len(pts), 1), T_ROOM)
            sol_u_last[0] = sol_u
            int_vars = list(problem.internal_vars)
            print(f"  [reset] hit {k+1} starts on a fresh mesh ({len(pts)} nodes, {len(cells_out)} cells), "
                  f"stress-free reference, reference min SJ {quality.q_ref.min():.3f}", flush=True)
        elif args.hit_reset != "none" and float(onp.abs(onp.asarray(sol_u)).max()) > 0.0:
            # Reheat between hits: the bar is brought back to temperature, the
            # residual stress relaxes ('stress') and the hardening recovers as
            # well ('full').  The released shape is in equilibrium with zero
            # stress, so it simply becomes the reference configuration of a
            # fresh problem: u = 0, F = be = I.  Per-step VTU files of this hit
            # therefore hold the displacement of THIS hit only; sim_*.vtu
            # points are the deformed shape as before.
            alpha_prev = onp.asarray(int_vars[2])
            u_prev = onp.asarray(sol_u)
            x_new = pts + u_prev
            if args.hit_smooth != "none":
                from lib.mesh_smooth import smooth_mesh
                from lib.mesh_quality import scaled_jacobian as _sj
                x_new, srep = smooth_mesh(x_new, cells_out, pts0, mode=args.hit_smooth,
                                          power=args.smooth_power)
                smooth_log.append(dict(srep, before_hit=k + 1))
                (out / "smooth_log.json").write_text(json.dumps(smooth_log, indent=2))
                meshio.Mesh(x_new, {cell_type: cells_out},
                            point_data={"relocation": x_new - (pts + u_prev)},
                            cell_data={"scaled_jacobian": [_sj(x_new, cells_out)],
                                       "scaled_jacobian_before": [_sj(pts + u_prev, cells_out)]}
                            ).write(out / f"mesh_hit{k+1:02d}_start.vtu")
            (mesh, pts, problem, die_contact, quality, bc_builder, H, R) = rebase_model(x_new)
            sol_u = np.zeros((len(pts), 3))
            sol_u_last[0] = sol_u
            int_vars = list(problem.internal_vars)
            if args.hit_reset == "stress":
                int_vars[2] = np.asarray(alpha_prev)
            print(f"  [reset] hit {k+1} starts from the current shape as a stress-free reference "
                  f"(--hit-reset {args.hit_reset}): previous |u|_inf {onp.abs(u_prev).max():.3f} mm, "
                  f"alpha max {alpha_prev.max():.3f} "
                  f"{'kept' if args.hit_reset == 'stress' else 'dropped'}, reference min SJ "
                  f"{quality.q_ref.min():.3f}", flush=True)
        failed = False
        # One band for all three strokes of this hit.  Re-selecting it per
        # stroke let the retract read its platen datum off material the
        # compression stroke never touched; see band_sol_u in
        # build_cylinder_press_bcs.
        # Taken unconditionally: on hit 1 sol_u is still zero, so the band is
        # the reference one, which is what virgin stock needs anyway.  Passing
        # None here instead would let hit 1's own retract and second stroke
        # fall back to per-stroke selection, the very thing being removed.
        band_u_hit = onp.asarray(sol_u)
        # A compression stroke stopped by the quality guard can be continued
        # after a relocation of the mesh (--stroke-relocate): the same
        # (rho_t, label) entry is inserted after the current one and `restart`
        # carries what the continuation differs in -- the die start pose and
        # the remaining travel -- instead of the band datum.
        plan_k = list(plans[k])
        restart = [None]
        n_reloc_stroke = [0]
        pose_committed = [None]       # (y_L, y_R) of the last COMMITTED step
        # the segments of a stroke continued after relocations: their force
        # history as die travel from the start of the stroke (sim_log_strokes
        # used to keep only the last segment)
        seg_acc = [None]
        start_bo = [0.0]              # die start back-off of the current stroke (lib/boundary_conditions)
        for si, (rho_t, label) in enumerate(plan_k):
          # Only the very first stroke of the very first hit sees virgin stock.
          # Every later stroke -- including the second stroke of hit 1 -- must
          # be built on the deformed configuration, and bite_for and the BC
          # builder have to agree on that: reading the travel off the deformed
          # bar while building the platen on the reference one put hit 1's
          # second stroke 0.9 mm clear of a bar it was supposed to be pressing,
          # and it recorded 0 kN against 0 contact nodes.
          virgin = (k == 0 and si == 0)
          die_start = None
          cont_state = None
          if restart[0] is not None:
            rs = restart[0]
            restart[0] = None
            depth, n_band = rs["depth"], 0
            die_start = rs["die_start"]
            cont_state = rs.get("contact_state")
            print(f"    {label} (continued after relocation {n_reloc_stroke[0]}): rho -> {rho_t:.3f}  "
                  f"remaining travel/side = {depth:+.3f} mm from dies at "
                  f"({die_start[0]:.4f}, {die_start[1]:.4f})", flush=True)
          else:
            n_reloc_stroke[0] = 0
            bite_full = [None]
            start_bo[0] = 0.0
            seg_acc[0] = dict(history_travel_mm=[], segments=0, steps=0, wall_s=0.0)
            depth, n_band = bite_for(pts, surf,
                                     None if virgin else onp.asarray(sol_u),
                                     phi_sim, a["z"], rho_t, args.band,
                                     band_config=args.band_config,
                                     band_sol_u=band_u_hit)
            print(f"    {label}: rho -> {rho_t:.3f}  travel/side = {depth:+.3f} mm "
                  f"({n_band} band nodes)", flush=True)
          if bite_full[0] is None:
            bite_full[0] = depth
          if depth is None or (not label.startswith("release") and depth <= 0.01):
            print("      no contact, skipped", flush=True)
            continue
          if label.startswith("release") and depth >= -0.01:
            print("      already clear of the platen, nothing to release",
                  flush=True)
            continue
          if True:
            hit = Hit(x_min_band=max(0.0, (a["z"] - 0.5 * args.band) / H),
                      x_max_band=min(1.0, (a["z"] + 0.5 * args.band) / H),
                      compression_displacement=float(depth),
                      rotation_euler_x=float(phi_sim),
                      total_time=max(0.05, float(depth) / V_PLATEN))

            dbc_u, dbc_T, loc_fns, bc_update_fn, surf_inds = bc_builder(
                mesh, R=R, H=H, hit=hit,
                current_sol_u=(None if virgin else sol_u),
                band_sol_u=band_u_hit,
                T_linear_fn=T_uniform, die_start=die_start)
            problem.dirichlet_bc_info = [dbc_u, None]
            problem.location_fns = [no_location, no_location]
            refresh_problem_surface_integrals(problem)
            # the builder opened the dies until the whole die face was clear of
            # the bar (die start back-off): the stroke travels that much
            # further, at the same platen speed, to the same end pose
            _bo = float(getattr(getattr(problem, "die_contact", None), "start_backoff", 0.0) or 0.0)
            if _bo > 0.0 and die_start is None and not label.startswith("release"):
                depth = float(depth) + _bo
                bite_full[0] = depth
                start_bo[0] = _bo
                hit = Hit(x_min_band=hit.x_min_band, x_max_band=hit.x_max_band,
                          compression_displacement=float(depth),
                          rotation_euler_x=hit.rotation_euler_x,
                          total_time=max(0.05, float(depth) / V_PLATEN))
                print(f"      die start backed off {_bo:.4f} mm (die face clear of the bar): "
                      f"travel/side = {depth:+.3f} mm", flush=True)

            problem.press_rot = rot_x(phi_sim)
            # dt cap from the die travel per step: depth mm over total_time s
            max_dt_stroke = args.max_dt
            if args.max_bite_step > 0 and not label.startswith("release") and depth > 0:
                max_dt_stroke = min(args.max_dt,
                                    hit.total_time * args.max_bite_step * h_min / depth)
            print(f"      dt <= {max_dt_stroke:.4f} s = {depth * max_dt_stroke / hit.total_time:.3f} mm "
                  f"of travel per step ({depth * max_dt_stroke / hit.total_time / h_min:.2f} h_min)",
                  flush=True)
            _where = f"hit {k+1} {label}"
            stepper = AutomaticTimeStepperTM(
                problem, total_time=hit.total_time, initial_dt=min(hit.total_time / 20.0, max_dt_stroke),
                min_dt=1e-6, max_dt=max_dt_stroke, max_retries=10,
                increase_factor=1.3, decrease_factor=0.5, line_search_after=0,
                cool_factor=0.0, surface_inds=surf_inds,
                # Only the release gets the loose tolerance; the compression
                # strokes keep 1e-8 because their reaction IS the measurement.
                abs_tol=(args.release_abs_tol if label.startswith("release") else None),
                accept_motion_mm=(args.release_accept_mm if label.startswith("release") else None),
                accept_res_n=(args.release_accept_res if label.startswith("release") else None),
                accept_fn=partial(accept_trial, where=_where),
                accept_max_refusals=args.accept_max_refusals,
                accept_max_refusals_total=args.accept_max_refusals_total,
                linear_solver=args.linear_solver)
            stepper.seed_state(sol_u, sol_dT, int_vars)
            q_last[0] = float(quality.evaluate(sol_u, f"{_where} start", record=False)[0].min())

            # Per-step reaction history.  bc_update_fn is handed the last CONVERGED
            # solution, whose internal_vars are the ones the residual is built from,
            # so the section load evaluated here is a genuine equilibrium reaction
            # for that step -- it turns each hit into a force-vs-closure curve that
            # can be compared against the 633 Hz press trace, instead of a single
            # end-of-hit number.
            hist = []
            # friction acts during compression only; each stroke starts clean
            # (the contact set is empty at the start of every stroke)
            use_fr = friction is not None and not label.startswith("release")
            if friction is not None:
                friction.reset()
            problem.friction = friction if use_fr else None
            if die_contact is not None:
                # the force-based contact acts on every stroke, the release
                # included (its planes move away and the forces fade out);
                # only the friction memory starts clean
                if onp.isfinite(die_contact.y_L):
                    _ev = die_contact._evaluate(onp.asarray(sol_u), False)
                    print(f"  [reset] friction memory dropped at the start of {_where}: "
                          f"sum|Ft| {onp.linalg.norm(_ev['Ft'], axis=1).sum() / 1e3:.3f} kN, "
                          f"|Ft|_2 {onp.linalg.norm(_ev['Ft']) / 1e3:.3f} kN on {int(_ev['active'].sum())} "
                          f"active nodes at the last pose", flush=True)
                die_contact.reset()
                if cont_state is not None:
                    # A CONTINUED stroke is the same stroke: the friction
                    # memory (committed friction force, lagged normal force,
                    # stick stiffness) belongs to the material under the die
                    # and is carried over the relocation; the slip anchors are
                    # the relocated positions (u_ref = 0 on the new reference).
                    # The pose is set now so that the commit at step 1, which
                    # runs before the first pose update, sees the dies where
                    # they are instead of at infinity and keeps N.
                    die_contact.set_pose(cont_state["R"], cont_state["x_mid"], die_start[0], die_start[1])
                    die_contact.N = onp.array(cont_state["N"])
                    die_contact.k_t = onp.array(cont_state["k_t"])
                    die_contact.Ft_old = onp.array(cont_state["Ft_old"])
                    die_contact.active_old = onp.array(cont_state["active_old"])
                    _ev = die_contact._evaluate(onp.asarray(sol_u), False)
                    print(f"  [reset] continued stroke: friction memory carried over the relocation: "
                          f"sum|Ft| {onp.linalg.norm(_ev['Ft'], axis=1).sum() / 1e3:.3f} kN on "
                          f"{int(_ev['active'].sum())} active nodes, load {die_contact.press_forces_kn(onp.asarray(sol_u))[0]:.2f} / "
                          f"{die_contact.press_forces_kn(onp.asarray(sol_u))[1]:.2f} kN at the start pose", flush=True)
                    # The carried state on the new reference must be in equilibrium
                    # with the dies at the start pose of the continuation.  This
                    # check used to run before the pose was set, with the dies at
                    # infinity, so it reported the missing die load as imbalance
                    # (|R| ~1e4 N, "load 0.00 kN").  Diagnostic only: the stepper
                    # sets its own parameters before the first solve.
                    try:
                        problem.set_params([int_vars, 0.0, sol_u, np.array([1.0, 1.0, 1.0, 1.0]), sol_dT])
                        _r = onp.array(problem.compute_residual([sol_u, sol_dT])[0])
                        # constrained rows hold the constraint reactions, not an imbalance
                        _fe = problem.fes[0]
                        for _ni, _vi in zip(_fe.node_inds_list, _fe.vec_inds_list):
                            _r[onp.asarray(_ni), onp.asarray(_vi)] = 0.0
                        _ld = die_contact.press_forces_kn(onp.asarray(sol_u))
                        print(f"  [relocate] equilibrium on the new reference at u = 0 with the dies at "
                              f"the start pose (free rows): |R|_2 {onp.linalg.norm(_r):.3e} N, max node "
                              f"{onp.linalg.norm(_r, axis=1).max():.3e} N, die load {_ld[0]:.2f} / "
                              f"{_ld[1]:.2f} kN", flush=True)
                    except Exception as _e:
                        print(f"  [relocate] equilibrium check skipped: {_e}", flush=True)
            fr_log = []
            committed_step = [None]
            pose_committed[0] = None

            def bc_update_traced(problem_, sol_u_, step_count, scale, _inner=bc_update_fn,
                                 _where=_where):
                # sol_u_ is the last COMMITTED state.  The stepper has already
                # accepted it through accept_trial before committing it, so
                # this check can only fire if the thresholds disagree; it is
                # kept as the backstop, and it records the committed value
                # the drop test of the next trial is measured against.
                sol_u_last[0] = sol_u_
                q_last[0] = float(quality.check(
                    sol_u_, f"{_where} committed before step {step_count} (scale {scale:.4f})").min())
                if use_fr:
                    # accept the friction state of the converged step
                    friction.commit(sol_u_)
                if die_contact is not None and step_count != committed_step[0]:
                    # accept the converged step at the pose it was solved at:
                    # friction memory and the lagged normal force; the load
                    # on the dies is read here, before the pose moves on.
                    # Only on the FIRST call for this step: a retry (refused
                    # trial, Newton failure) calls back with the same committed
                    # state after the pose has already advanced, and committing
                    # it again against that pose would read a fake penetration
                    # (measured: 0.17 mm, 442 kN at a refused step of the fillet
                    # run on mesh A) into the lagged normal force and friction.
                    committed_step[0] = step_count
                    # the pose the committed state was solved at (set_pose for
                    # the next step happens in _inner below); none before step 1
                    pose_committed[0] = ((die_contact.y_L, die_contact.y_R)
                                         if step_count > 1 else None)
                    _f0 = die_contact._evaluate(onp.asarray(sol_u_), False)["f"]
                    die_contact.commit(sol_u_)
                    # force change at the SAME displacement and pose: what the
                    # state update alone puts back into the residual
                    _df = die_contact._evaluate(onp.asarray(sol_u_), False)["f"] - _f0
                    print(f"  [commit] step {step_count}: contact force change at fixed u: "
                          f"|dF|_2 {onp.linalg.norm(_df):.4e} N, max node "
                          f"{onp.linalg.norm(_df, axis=1).max():.4e} N", flush=True)
                    F_lo_, F_up_ = die_contact.press_forces_kn(sol_u_)
                    hist.append((float(scale), float(F_up_)))
                    st = die_contact.stats(sol_u_)
                    fr_log.append([round(float(scale), 5), st["n_active"], st["n_slip"],
                                   round(st["sum_abs_Ft_kN"], 4), round(st["sum_N_kN"], 4)])
                    print(f"  [contact] step {step_count}: {st['n_active']} nodes "
                          f"({st['n_lower']} lower / {st['n_upper']} upper), "
                          f"{st['n_slip']} sliding, max penetration {st['max_pen_mm']:.4f} mm, "
                          f"load {st['F_lower_kN']:.2f} / {st['F_upper_kN']:.2f} kN, "
                          f"sum|Ft| {st['sum_abs_Ft_kN']:.3f} kN", flush=True)
                    if args.dump_contact and not label.startswith("release"):
                        ev = die_contact._evaluate(onp.asarray(sol_u_), False)
                        a = ev["active"]
                        dd = out / f"contact_hit{k+1:02d}_{SLUG[label]}"
                        dd.mkdir(exist_ok=True)
                        onp.savez(dd / f"step_{step_count - 1:04d}.npz",
                                  step=step_count - 1, scale=float(scale), phi=float(phi_sim),
                                  nodes=die_contact.cand[a], side=ev["side"][a],
                                  x_die=ev["p"][a], f_die=ev["f_die"][a], gap=ev["d"][a],
                                  band=onp.array([hit.x_min_band * H, hit.x_max_band * H]))
                _inner(problem_, sol_u_, step_count, scale)
                N_node = None
                if die_contact is not None:
                    return
                try:
                    r = problem_.compute_residual([np.array(sol_u_),
                                                   np.array(stepper.sol_dT)])[0]
                    xd = pts + onp.asarray(sol_u_)
                    yr = (Rm_hit @ xd.T).T[:, 1]
                    fr = (Rm_hit @ onp.asarray(r).reshape(len(pts), 3).T).T
                    N_node = onp.abs(fr[:, 1])        # press-normal reaction per node
                    hist.append((float(scale),
                                 float(onp.sum(fr[yr > 0.0, 1]) / 1e3)))
                    if args.dump_contact and not label.startswith("release"):
                        # committed state of step_count-1 and the contact set
                        # that was held during it (the sets are refreshed by
                        # _inner above for the step about to start)
                        cl = onp.asarray(problem_.contact_left_inds, dtype=int)
                        cr = onp.asarray(problem_.contact_right_inds, dtype=int)
                        nodes_ = onp.concatenate([cl, cr])
                        xr_all = (Rm_hit @ xd.T).T
                        F_fr = (friction.force_and_stiffness(onp.asarray(sol_u_))[0]
                                if use_fr else onp.zeros(len(pts)))
                        dd = out / f"contact_hit{k+1:02d}_{SLUG[label]}"
                        dd.mkdir(exist_ok=True)
                        onp.savez(dd / f"step_{step_count - 1:04d}.npz",
                                  step=step_count - 1, scale=float(scale), phi=float(phi_sim),
                                  left=cl, right=cr, nodes=nodes_,
                                  x_die=xr_all[nodes_], f_die=fr[nodes_],
                                  f_friction_x=F_fr[nodes_],
                                  band=onp.array([hit.x_min_band * H, hit.x_max_band * H]))
                except Exception as e:
                    print(f"  [force-hist] skipped at step {step_count}: {e}", flush=True)
                if use_fr:
                    nodes = onp.concatenate([onp.asarray(problem_.contact_left_inds, dtype=int),
                                             onp.asarray(problem_.contact_right_inds, dtype=int)])
                    friction.set_contact(nodes, N_node if N_node is not None
                                         else onp.zeros(len(pts)))
                    st = friction.stats(sol_u_)
                    fr_log.append([round(float(scale), 5), st["n_active"], st["n_slip"],
                                   round(st["sum_abs_F_kN"], 4), round(st["sum_N_kN"], 4)])
                    print(f"  [friction] step {step_count}: {st['n_active']} nodes, "
                          f"{st['n_slip']} sliding at the last state, "
                          f"sum|F| {st['sum_abs_F_kN']:.3f} kN of mu*N {args.friction_mu * st['sum_N_kN']:.3f} kN",
                          flush=True)

            Rm_hit = rot_x(phi_sim)
            try:
                ok = stepper.run(bc_update_fn=bc_update_traced,
                                 bc_params_fn=lambda step, scale: (step, scale),
                                 rho_ini=np.array([1.0, 1.0, 1.0, 1.0]),
                                 vtk_dir=(str(out / (f"vtk_hit{k+1:02d}_{SLUG[label]}"
                                                     + (f"_r{n_reloc_stroke[0]}" if n_reloc_stroke[0] else "")))
                                          if args.save_vtk else None),
                                 save_every=1)
                sol_u_last[0] = stepper.sol_u
                q_end = quality.check(stepper.sol_u, f"hit {k+1} {label} end")
            except DistortionStop as e:
                travelled = float(stepper.current_time / stepper.total_time) * float(depth)
                can_relocate = (args.stroke_relocate != "none" and not label.startswith("release")
                                and die_contact is not None
                                and n_reloc_stroke[0] < args.relocate_max
                                # the minimum advance guards a CONTINUED segment (no progress
                                # after a relocation); the first relocation of a stroke is free
                                and (n_reloc_stroke[0] == 0
                                     or travelled >= args.relocate_min_advance)
                                and (pose_committed[0] is not None or die_start is not None))
                if not can_relocate:
                    if args.stroke_relocate != "none" and not label.startswith("release"):
                        print(f"    [relocate] not continued: {n_reloc_stroke[0]} relocation(s) so far "
                              f"(max {args.relocate_max}), {travelled:.3f} mm travelled since the last "
                              f"one (min {args.relocate_min_advance}, applies after the first)", flush=True)
                    distortion_stop(e, k, label)
                    failed = True
                    break
                # ---- relocation inside the stroke ----
                from lib.mesh_smooth import smooth_mesh
                from lib.mesh_quality import scaled_jacobian as _sj
                from lib.state_transfer import gauss_points, transfer_field, committed_state
                u_c = onp.asarray(stepper.sol_u)                   # last committed state
                iv_c = list(stepper.int_vars)
                contact_state = dict(N=die_contact.N.copy(), k_t=die_contact.k_t.copy(),
                                     Ft_old=die_contact.Ft_old.copy(), active_old=die_contact.active_old.copy(),
                                     R=die_contact.R.copy(), x_mid=float(die_contact.x_mid))
                yL_c, yR_c = pose_committed[0] if pose_committed[0] is not None else die_start
                x_c = pts + u_c
                q_c = _sj(x_c, cells_out)
                bad_cells = onp.where(q_c < args.relocate_quality)[0]
                near = x_c[onp.unique(cells_out[bad_cells])] if len(bad_cells) else x_c[onp.unique(cells_out[[int(onp.argmin(q_c))]])]
                # the die at the committed pose is an obstacle for the sliding
                # surface nodes: a node that ends up inside it is put back on
                # the die face along the face normal (die frame p = x R^T)
                dc_clamp = NodalDieContact(onp.zeros_like(x_c), surf, profile, k_n, mu=0.0,
                                           normal=args.die_normal)
                dc_clamp.set_pose(die_contact.R, die_contact.x_mid, yL_c, yR_c)
                # the committed state already penetrates by the penalty's
                # ~0.05-0.09 mm; only penetration beyond that is clamped, so
                # the clamp leaves the committed contact nodes alone (clamping
                # them to zero gap at the first iteration raised the energy
                # above the start and the smoother returned the input mesh)
                _d0 = dc_clamp._geometry(x_c)[0]
                pen_tol = float(max(0.0, -onp.min(_d0[onp.isfinite(_d0)]))) if onp.isfinite(_d0).any() else 0.0

                def die_clamp(y, _dc=dc_clamp, _tol=pen_tol):
                    d_, n_, _, p_, _ = _dc._geometry(y)
                    pen_ = onp.where(onp.isfinite(d_), onp.maximum(0.0, -d_ - _tol), 0.0)
                    if not (pen_ > 0.0).any():
                        return y
                    y = onp.array(y, dtype=float)
                    y[_dc.cand] = (p_ + pen_[:, None] * n_) @ _dc.R
                    return y

                method = "relocate"
                srep = None
                if args.relocate_test_nomove:
                    x_new = onp.array(x_c)
                    srep = dict(accepted=True, min_sj_before=float(q_c.min()), min_sj_after=float(q_c.min()),
                                max_boundary_move_mm=0.0, surface_dev_max_mm=0.0, volume_change_rel=0.0,
                                test_nomove=True)
                elif not (args.stroke_remesh != "none" and args.stroke_remesh_first):
                    x_new, srep = smooth_mesh(x_c, cells_out, pts0, mode=args.stroke_relocate,
                                              power=args.relocate_power, iters=args.relocate_iters,
                                              restarts=args.relocate_restarts,
                                              crease_deg=args.relocate_crease_deg,
                                              slide_near=near, slide_radius=args.relocate_radius,
                                              clamp_fn=die_clamp)
                do_remesh = (args.stroke_remesh != "none" and not args.relocate_test_nomove
                             and (srep is None or not srep["accepted"]
                                  or srep["min_sj_after"] < args.relocate_min_quality))
                cells_new, surf_new, rrep_m = cells_out, surf, None
                if do_remesh:
                    # ---- in-stroke remesh: the committed shape gets a fresh O-grid
                    # mesh (same procedure as between hits); the state and the
                    # friction memory follow the material
                    from lib.remesh_ogrid import remesh_ogrid
                    from hex8 import hex_volumes
                    from scipy.spatial import cKDTree as _KDTree
                    if srep is not None:
                        print(f"    [relocate] smoothing reaches only min SJ {srep['min_sj_after']:.3f} "
                              f"(< {args.relocate_min_quality}" + ("" if srep["accepted"] else ", rejected") +
                              "): remeshing instead", flush=True)
                    tpl = dict(remesh_template, n=int(args.remesh_n)) if args.remesh_n > 0 else remesh_template
                    # Wall height of the current imprint: free-surface radius just outside
                    # the band (die frame, near the press plane) minus the die half-gap.
                    # The layer at the wall is a fraction of it (--remesh-wall-aspect):
                    # fine layers across a near-vertical step give the step cell
                    # SJ dx/sqrt(dx^2+H^2) (0.08 measured at 0.7 mm / 3.4 mm) and their
                    # facets' chords dipped 1.5 mm into the die.
                    _pf = x_c[surf] @ die_contact.R.T
                    _s = onp.abs(_pf[:, 0] - die_contact.x_mid)
                    _hb = 0.5 * args.band
                    _outside = (_s > _hb + 1.0) & (_s < _hb + 4.0) & (onp.abs(_pf[:, 2]) < 3.0)
                    _half_gap = 0.5 * (yR_c - yL_c)
                    _r_out = float(onp.median(onp.abs(_pf[_outside, 1]))) if _outside.any() else _half_gap
                    H_wall = max(0.0, _r_out - _half_gap)
                    h_wall = max(args.remesh_h_fine, args.remesh_wall_aspect * H_wall)
                    print(f"    [remesh] imprint wall height {H_wall:.2f} mm (free surface r {_r_out:.2f}, half-gap "
                          f"{_half_gap:.2f}): wall layers {h_wall:.2f} mm", flush=True)
                    try:
                        x_new, cells_new, surf_new, rrep_m = remesh_ogrid(
                            x_c, cells_out, tpl, args.remesh_h_coarse, h_wall,
                            core=args.remesh_core, surf_bias=args.remesh_surf_bias,
                            zones_extra=(zones_hit or ()), grade=args.remesh_grade,
                            slope_deg=args.remesh_slope_deg)
                    except RuntimeError as _re:
                        print(f"    [remesh] in-stroke remesh failed: {_re}", flush=True)
                        distortion_stop(e, k, label)
                        failed = True
                        break
                    method = "remesh"
                    rrep_m["wall_height_mm"], rrep_m["wall_layer_mm"] = float(H_wall), float(h_wall)
                    # A new surface node re-sampled on an old facet between two wall
                    # nodes lies INSIDE the die: the facet is a chord of the curved
                    # wall (0.7 mm layers on the w = 0.5 sigmoid dip ~0.4 mm vertically
                    # although the new surface is within 0.03 mm of the old one).
                    # Such a node is put back on the die face along the true face
                    # normal, only beyond the committed penalty penetration, so the
                    # dies need not back off (measured without this: back-off 0.25 mm,
                    # load 48 -> 13 kN at the first step after the remesh).
                    _dc_c = NodalDieContact(onp.zeros_like(x_new), surf_new, profile, k_n, mu=0.0, normal="surface")
                    _dc_c.set_pose(die_contact.R, die_contact.x_mid, yL_c, yR_c)
                    _d, _n, _, _p, _ = _dc_c._geometry(x_new)
                    # a node deeper than the committed penetration goes all the way to
                    # the face (zero gap, zero force; the next step loads it): leaving it
                    # at the committed penetration gave every such node the committed
                    # MAXIMUM nodal force at once (measured: 110 kN at the start pose
                    # against 48 kN committed)
                    _pen_all = onp.where(onp.isfinite(_d) & (-_d > pen_tol), onp.maximum(0.0, -_d), 0.0)
                    # a move of more than a fraction of the local layer inverts the cell
                    # behind the node (1.47 mm moves gave SJ -0.57): the rest stays as
                    # penetration and the dies back off by it
                    _lim0 = 0.3 * h_wall
                    # Only the surface node moves, so the thin wall cell behind it is
                    # squashed: the clamp alone took a remeshed min SJ 0.198 to 0.046
                    # (below the stop line, so the continued segment could not advance).
                    # Quality check: halve the limit until the min SJ stays at least
                    # --clamp-min-quality (or the remeshed value, if that is lower);
                    # last resort no clamp at all.  What is not put back is left to
                    # the die back-off below.
                    _sj_rem = float(rrep_m["min_sj"])
                    _q_need = min(args.clamp_min_quality, _sj_rem)
                    _levels = [_lim0] if args.clamp_min_quality <= 0 else [_lim0, _lim0 / 2, _lim0 / 4, 0.0]
                    _x_rem = onp.array(x_new, dtype=float)
                    _vol0 = float(hex_volumes(_x_rem, cells_new).sum())
                    _tries = []
                    n_clamped, clamp_max, n_limited, _lim = 0, 0.0, 0, _lim0
                    for _lim in _levels:
                        _pen = onp.minimum(_pen_all, _lim)
                        n_limited = int((_pen_all > _lim).sum())
                        n_clamped, clamp_max = int((_pen > 0.0).sum()), float(_pen.max()) if _pen.size else 0.0
                        if n_clamped == 0:
                            x_new, _sj_try = _x_rem, _sj_rem
                        else:
                            x_new = _x_rem.copy()
                            x_new[surf_new] = (_p + _pen[:, None] * _n) @ _dc_c.R
                            _sj_try = float(_sj(x_new, cells_new).min())
                        _tries.append((float(_lim), _sj_try))
                        if args.clamp_min_quality <= 0 or _sj_try >= _q_need or n_clamped == 0:
                            break
                    if len(_tries) > 1:
                        print(f"    [remesh] die clamp quality check (min SJ >= {_q_need:.3f}): "
                              + ", ".join(f"limit {a:.3f} mm -> SJ {b:.3f}" for a, b in _tries), flush=True)
                    if n_limited:
                        print(f"    [remesh] {n_limited} nodes deeper than the clamp limit {_lim:.3f} mm "
                              f"(penetration left to the die back-off)", flush=True)
                    if n_clamped:
                        print(f"    [remesh] {n_clamped} new surface nodes inside the die at the committed pose "
                              f"(max {clamp_max:.3f} mm, committed penetration {pen_tol:.3f} mm) put back "
                              f"on the die face; volume {100 * (float(hex_volumes(x_new, cells_new).sum()) / _vol0 - 1):+.4f} %, "
                              f"min SJ {_sj_rem:.3f} -> {_sj_try:.3f}", flush=True)
                        rrep_m["min_sj_after_die_clamp"] = _sj_try
                        rrep_m["volume_change_rel"] = float(hex_volumes(x_new, cells_new).sum()) / rrep_m["volume_old"] - 1.0
                    rrep_m["die_clamp_nodes"], rrep_m["die_clamp_max_mm"] = n_clamped, clamp_max
                    rrep_m["die_clamp_limit_mm"], rrep_m["die_clamp_tries"] = float(_lim), _tries
                    # distance of every new surface node to the nearest OLD surface node
                    # (a node-spacing measure, not the surface deviation, which is
                    # ~0.03 mm by construction: the new nodes lie on the old facets)
                    _dsurf, _near = _KDTree(x_c[surf]).query(x_new[surf_new])
                    srep = dict(accepted=True, min_sj_before=float(q_c.min()),
                                min_sj_after=float(rrep_m.get("min_sj_after_die_clamp", rrep_m["min_sj"])),
                                max_boundary_move_mm=clamp_max, surface_dev_max_mm=clamp_max,
                                volume_change_rel=float(rrep_m["volume_change_rel"]))
                    # friction memory: lagged normal force, stick stiffness and committed
                    # friction force are nodal quantities; each new surface node takes the
                    # nearest old node's, and the totals are conserved (a change of node
                    # density under the die must not change the friction capacity mu*sum N)
                    _N_old, _Ft_old = contact_state["N"], contact_state["Ft_old"]
                    _N_map = _N_old[_near]
                    _scale = float(_N_old.sum() / _N_map.sum()) if _N_map.sum() > 0.0 else 1.0
                    contact_state = dict(N=_N_map * _scale, k_t=contact_state["k_t"][_near] * _scale,
                                         Ft_old=_Ft_old[_near] * _scale,
                                         active_old=contact_state["active_old"][_near],
                                         R=contact_state["R"], x_mid=contact_state["x_mid"])
                    print(f"    [remesh] friction memory mapped to {len(surf_new)} surface nodes (was {len(surf)}): "
                          f"{int(contact_state['active_old'].sum())} active (was {int((_N_old > 0).sum())}), "
                          f"nodal scale {_scale:.4f}; sum N {_N_old.sum() / 1e3:.2f} kN, sum|Ft| "
                          f"{onp.linalg.norm(_Ft_old, axis=1).sum() / 1e3:.3f} -> "
                          f"{onp.linalg.norm(contact_state['Ft_old'], axis=1).sum() / 1e3:.3f} kN; nearest old "
                          f"surface node within {_dsurf.max():.3f} mm", flush=True)
                # The material state follows the material: hardening alpha, the
                # elastic tensor be_bar (spatial, so unchanged by the new
                # reference) and the total volume ratio J of the committed shape
                # are re-sampled at the Gauss points of the rebased mesh from the
                # old Gauss points, both at their deformed positions.  Where the
                # mesh did not move the transfer is exact and the stress of the
                # first evaluation on the new reference equals the committed one.
                fe_old = problem.fes[0]
                xg_old = gauss_points(x_c, fe_old.cells, fe_old.shape_vals)
                alpha_old = onp.asarray(iv_c[2])
                be_bar_old, J_old = committed_state(iv_c)
                # penetration of the relocated nodes at the committed die pose
                # beyond the committed penalty penetration (that one belongs to
                # the equilibrium being carried over): the dies back off by that
                # much and travel it again
                dc_probe = NodalDieContact(x_new, surf_new, profile, k_n, mu=0.0, normal=args.die_normal)
                dc_probe.set_pose(die_contact.R, die_contact.x_mid, yL_c, yR_c)
                # The back-off is iterated (2026-11-10).  One shot, back_off =
                # pen_max - pen_tol, assumes that opening the dies by d lowers
                # every penetration by d.  That holds on the flat; on the
                # inclined wall the normal distance grows only by d cos(slope),
                # so wall nodes stayed inside the die face (0.165 mm on 4 nodes
                # after remesh #5 of hit 8, scanned positions: 15 kN on one node
                # at the first step of the continuation, four cells crushed, run
                # stopped; 0.11-0.12 mm twice in the commanded run, survived).
                # The first pass is the old formula, so a continuation that was
                # already clear is unchanged.
                back_off, _n_bo = 0.0, 0
                for _n_bo in range(1, 101):
                    dc_probe.set_pose(die_contact.R, die_contact.x_mid, yL_c - back_off, yR_c + back_off)
                    d_probe = dc_probe._geometry(onp.zeros_like(x_new))[0]
                    pen_max = float(max(0.0, -onp.min(d_probe[onp.isfinite(d_probe)]))) if onp.isfinite(d_probe).any() else 0.0
                    _exc = pen_max - pen_tol
                    if _exc <= 1e-6:
                        break
                    back_off += _exc
                if _n_bo > 1:
                    print(f"    [relocate] die back-off iterated {_n_bo} times: {back_off:.4f} mm, penetration left "
                          f"{pen_max:.4f} mm (committed {pen_tol:.4f} mm)", flush=True)
                y0_L, y0_R = yL_c - back_off, yR_c + back_off
                depth_new = 0.5 * (y0_R - y0_L) - rho_t
                n_reloc_stroke[0] += 1
                tag_r = f"hit{k+1:02d}_{SLUG[label]}_r{n_reloc_stroke[0]}"
                if method == "remesh":
                    meshio.Mesh(x_c, {cell_type: cells_out},
                                cell_data={"scaled_jacobian": [q_c]}).write(out / f"mesh_{tag_r}_before.vtu")
                    meshio.Mesh(x_new, {cell_type: cells_new},
                                cell_data={"scaled_jacobian": [_sj(x_new, cells_new)]}).write(out / f"mesh_{tag_r}.vtu")
                    (mesh, pts, problem, die_contact, quality, bc_builder, H, R) = rebase_model(
                        x_new, cells_new, surf_new)
                    remesh_template = rrep_m.pop("template")
                    remesh_log.append(dict(rrep_m, in_stroke=True, hit=k + 1, stroke=label,
                                           relocation=n_reloc_stroke[0],
                                           zones_hit=[list(map(float, z)) for z in (zones_hit or ())]))
                    (out / "remesh_log.json").write_text(json.dumps(remesh_log, indent=2))
                else:
                    meshio.Mesh(x_new, {cell_type: cells_out},
                                point_data={"relocation": x_new - x_c},
                                cell_data={"scaled_jacobian": [_sj(x_new, cells_out)],
                                           "scaled_jacobian_before": [q_c]}
                                ).write(out / f"mesh_{tag_r}.vtu")
                    (mesh, pts, problem, die_contact, quality, bc_builder, H, R) = rebase_model(x_new)
                fe_new = problem.fes[0]
                xg_new = gauss_points(x_new, fe_new.cells, fe_new.shape_vals)
                alpha_new = transfer_field(xg_old, alpha_old, xg_new, k=4)
                be_bar_new = transfer_field(xg_old, be_bar_old, xg_new, k=4)
                J_new = transfer_field(xg_old, J_old, xg_new, k=4)
                int_vars = list(problem.internal_vars)
                int_vars[1] = np.asarray(be_bar_new)      # F_old = I, so be_old = be_bar
                int_vars[2] = np.asarray(alpha_new)
                int_vars[6] = np.asarray(J_new)
                sol_u = np.zeros((len(pts), 3))
                if method == "remesh":
                    # A remesh can change the node count (hit 8 of roll3.1: 68 ->
                    # 69 stations, 18768 -> 19044 nodes) and the temperature was
                    # carried at the old size, so the solver failed to unflatten
                    # the dofs.  Rebuild it on the new nodes as at the start of
                    # a hit; exact only because the model is isothermal.
                    if float(np.max(sol_dT) - np.min(sol_dT)) > 1e-9:
                        raise RuntimeError("in-stroke remesh with a non-uniform temperature: "
                                           "the temperature transfer is not implemented")
                    sol_dT = np.full((len(pts), 1), float(np.max(sol_dT)))
                sol_u_last[0] = sol_u
                band_u_hit = onp.asarray(sol_u)
                rrep = dict(srep, method=method, hit=k + 1, stroke=label, relocation=n_reloc_stroke[0],
                            at_scale=float(stepper.current_time / stepper.total_time),
                            travelled_mm=travelled, remaining_mm=float(depth_new),
                            back_off_mm=back_off, dies_committed=[float(yL_c), float(yR_c)],
                            n_cells_repaired=int(len(bad_cells)),
                            n_points=int(len(pts)), n_cells=int(len(cells_out)),
                            alpha_max_before=float(alpha_old.max()), alpha_max_after=float(alpha_new.max()),
                            alpha_mean_before=float(alpha_old.mean()), alpha_mean_after=float(alpha_new.mean()),
                            J_ref_min=float(J_new.min()), J_ref_max=float(J_new.max()),
                            be_bar_dev_max=float(onp.abs(be_bar_new - onp.eye(3)).max()),
                            state_transfer="alpha, be_bar, J_ref",
                            stop_message=str(e))
                relocate_log.append(rrep)
                (out / "relocate_log.json").write_text(json.dumps(relocate_log, indent=2))
                print(f"    [{method}] hit {k+1} {label} #{n_reloc_stroke[0]}: guard stop at scale "
                      f"{rrep['at_scale']:.4f} ({travelled:.3f} mm travelled); {len(bad_cells)} cells below "
                      f"{args.relocate_quality}; min SJ {srep['min_sj_before']:.3f} -> {srep['min_sj_after']:.3f}, "
                      f"boundary move {srep['max_boundary_move_mm']:.2f} mm, surface deviation "
                      f"{srep['surface_dev_max_mm']:.3f} mm, volume {100 * srep['volume_change_rel']:+.4f} %; "
                      f"alpha max {alpha_old.max():.3f} -> {alpha_new.max():.3f} (mean {alpha_old.mean():.4f} -> "
                      f"{alpha_new.mean():.4f}); be_bar and J carried (J_ref {J_new.min():.4f}..{J_new.max():.4f}); "
                      f"dies back off {back_off:.3f} mm, remaining travel {depth_new:.3f} mm"
                      + (f"; mesh {len(pts)} nodes / {len(cells_out)} cells" if method == "remesh" else ""), flush=True)
                if not srep["accepted"]:
                    print("    [relocate] smoothing was rejected (no gain): stopping", flush=True)
                    distortion_stop(e, k, label)
                    failed = True
                    break
                # keep this segment's force history before the continuation
                _off = float(bite_full[0]) - float(depth)
                seg_acc[0]["history_travel_mm"] += [[round(_off + c * float(depth), 5), round(f, 3)]
                                                    for c, f in hist]
                seg_acc[0]["segments"] += 1
                seg_acc[0]["steps"] += int(stepper.step_count)
                seg_acc[0]["wall_s"] += float(sum(d[2] for d in stepper.step_data))
                restart[0] = dict(depth=float(depth_new), die_start=(float(y0_L), float(y0_R)),
                                  contact_state=contact_state)
                plan_k.insert(si + 1, (rho_t, label))
                continue
            if getattr(stepper, "n_accepted_loose", 0):
                print(f"      note: {stepper.n_accepted_loose} step(s) of this "
                      f"{label} were accepted below tolerance", flush=True)
            if not ok:
                print(f"    [FAIL] hit {k+1} {label} did not converge; last failure: "
                      f"{getattr(stepper, 'last_failure', 'n/a')}", flush=True)
                failed = True
                break

            sol_u, sol_dT, int_vars = stepper.sol_u, stepper.sol_dT, stepper.int_vars

            # Press load = reaction carried by the upper platen.  compute_residual
            # is the UNCONSTRAINED internal force, so at a prescribed dof it is
            # exactly the force the constraint applies; projecting on the rotated
            # y (the press normal) and summing over that platen's nodes is what a
            # load cell under it would read.
            # The reaction was captured inside the stepper, at the only moment the
            # problem parameters match the converged solution; see the note there.
            res = stepper.last_res_u
            right = getattr(problem, "contact_right_inds", onp.array([], dtype=int))
            left = getattr(problem, "contact_left_inds", onp.array([], dtype=int))
            Rm = rot_x(phi_sim)
            F_r = press_force_kn(res, right, Rm, len(pts))
            F_l = press_force_kn(res, left, Rm, len(pts))
            F = 0.5 * (abs(F_r) + abs(F_l))
            if die_contact is not None:
                # with force-based contact the converged residual is zero at
                # the contact nodes; the load is the sum of the die forces
                F_l, F_r = die_contact.press_forces_kn(sol_u)
                F = 0.5 * (abs(F_r) + abs(F_l))

            # Section load.  The published contact set is detected at the START of
            # a step from the previous solution, so nodes that come into contact
            # during the final increment are absent from it and their share of the
            # load is lost -- with a 2.2 mm mesh and a ~1.4 mm bite that is a large
            # and erratic fraction.  Cutting the body at the platen mid-plane and
            # summing the reaction over the whole half-body instead counts every
            # constrained dof on that side, whenever it was detected.  Away from
            # the constrained dofs the residual is the equilibrium residual, i.e.
            # numerical noise, so the free nodes contribute nothing.
            xdef_all = pts + onp.asarray(sol_u)
            y_rot_all = (Rm @ xdef_all.T).T[:, 1]
            f_rot_all = (Rm @ onp.asarray(res).reshape(len(pts), 3).T).T
            F_up = float(onp.sum(f_rot_all[y_rot_all > 0.0, 1]) / 1e3)
            F_lo = float(onp.sum(f_rot_all[y_rot_all <= 0.0, 1]) / 1e3)
            F_sec = 0.5 * (abs(F_up) + abs(F_lo))
            if die_contact is not None:
                F_up, F_lo, F_sec = F_r, F_l, F

            # F_up and F_lo are not independent: the internal force vector sums to
            # zero for any displacement field, so F_lo = -F_up identically and
            # their agreement proves nothing.  What the half-body sum DOES pick up
            # besides the platen is the reaction of the rigid-body constraints on
            # the x = x_min end face, which fall in whichever half the roll puts
            # them.  Measure that contamination instead of assuming it away, and
            # report a variant that excludes the pad the constraints live in.
            # Where does the residual actually live?  In a converged solution it is
            # the reaction at constrained dofs and nothing else, so a half-body sum
            # is the platen load.  If instead it is spread over the free nodes, any
            # half-body sum is meaningless and rotates with the cut plane -- which
            # is exactly the symptom: at fixed bite the load varied by 4x with the
            # roll angle while the displacement field did not move at all.
            plane = float(onp.max((Rm @ (pts + onp.asarray(sol_u)).T).T[:, 1])) - 0.10
            y_all = (Rm @ (pts + onp.asarray(sol_u)).T).T[:, 1]
            act = y_all >= plane
            rbm = onp.abs(pts[:, 0] - pts[:, 0].min()) <= 1e-5
            free = ~act & ~rbm & (y_all > 0.0)
            print(f"    [res] |f| total {onp.linalg.norm(f_rot_all):.3e} N | "
                  f"active-contact {onp.linalg.norm(f_rot_all[act]):.3e} ({act.sum()} nodes) | "
                  f"end-face {onp.linalg.norm(f_rot_all[rbm]):.3e} ({rbm.sum()}) | "
                  f"free-upper {onp.linalg.norm(f_rot_all[free]):.3e} ({free.sum()})", flush=True)
            print(f"    [res] Fy: active {onp.sum(f_rot_all[act,1])/1e3:+.3f} kN | "
                  f"end-face {onp.sum(f_rot_all[rbm,1])/1e3:+.3f} kN | "
                  f"free-upper {onp.sum(f_rot_all[free,1])/1e3:+.3f} kN", flush=True)

            far = pts[:, 0] > 0.0
            F_up_x0 = float(onp.sum(f_rot_all[(y_rot_all > 0.0) & far, 1]) / 1e3)
            rb = onp.abs(pts[:, 0] - pts[:, 0].min()) <= 1e-5
            F_rb = float(onp.sum(f_rot_all[rb, 1]) / 1e3)
            # Reaction of the held end face: resultant force (lab frame) and its
            # moment about the bar axis (x through y = z = 0), i.e. the torque the
            # grip carries to keep the bar from rolling under the dies.
            _rr = onp.asarray(res).reshape(len(pts), 3)[rb]
            _xr = (pts + onp.asarray(sol_u))[rb]
            hold_f = (_rr.sum(axis=0) / 1e3).tolist()
            hold_mx = float(onp.sum(_xr[:, 1] * _rr[:, 2] - _xr[:, 2] * _rr[:, 1]) / 1e3)
            print(f"    [hold] end-face reaction ({int(rb.sum())} nodes, {args.axial_constraint}): "
                  f"F = ({hold_f[0]:+.3f}, {hold_f[1]:+.3f}, {hold_f[2]:+.3f}) kN, moment about "
                  f"the bar axis {hold_mx:+.2f} N m, largest node force "
                  f"{float(onp.linalg.norm(_rr, axis=1).max()) / 1e3:.3f} kN", flush=True)

            rec = dict(hit=k + 1, stroke=label, phi=a["phi"], z=a["z"],
                       phi_command=a.get("phi_command", a["phi"]), z_command=a.get("z_command", a["z"]),
                       rho=float(rho_t), bite_mm=float(bite_full[0]),
                       start_backoff_mm=float(start_bo[0]),
                       relocations=int(n_reloc_stroke[0]),
                       edge_radius_mm=float(args.edge_radius),
                       axial_slip=bool(args.axial_slip),
                       contact=args.contact,
                       die_profile=(args.die_profile if die_contact is not None else None),
                       friction_mu=args.friction_mu,
                       friction_log=fr_log,
                       steps=int(stepper.step_count),
                       force_kn=float(F_sec), force_section_kn=float(F_sec),
                       force_section_upper_kn=float(abs(F_up)),
                       force_section_lower_kn=float(abs(F_lo)),
                       force_section_x0_kn=float(F_up_x0),
                       force_rigidbody_kn=float(F_rb),
                       axial_constraint=args.axial_constraint,
                       hold_force_kn=[float(v) for v in hold_f],
                       hold_moment_x_nm=hold_mx,
                       force_contactset_kn=float(F),
                       force_upper_kn=float(abs(F_r)),
                       force_lower_kn=float(abs(F_l)),
                       n_contact=[int(len(left)), int(len(right))],
                       min_scaled_jacobian=float(q_end.min()),
                       history=[[round(c, 5), round(f, 3)] for c, f in hist],
                       # per accepted step: [step, scale, wall_s, dt, newton_iters, retries]
                       step_log=[list(d) for d in stepper.step_data],
                       wall_s=float(sum(d[2] for d in stepper.step_data)))
            # 'history', 'steps' and 'wall_s' above are the last segment only;
            # these cover every segment of a stroke continued after relocations
            _acc = seg_acc[0] or dict(history_travel_mm=[], segments=0, steps=0, wall_s=0.0)
            _off = float(bite_full[0]) - float(depth)
            rec["history_travel_mm"] = _acc["history_travel_mm"] + [
                [round(_off + c * float(depth), 5), round(f, 3)] for c, f in hist]
            rec["segments"] = int(_acc["segments"]) + 1
            rec["steps_all_segments"] = int(_acc["steps"]) + int(stepper.step_count)
            rec["wall_s_all_segments"] = float(_acc["wall_s"]) + rec["wall_s"]
            strokes_log.append(rec)
            # sim_log.json stays one entry per hit -- the LAST stroke, which is
            # the one that lands on the commanded closure and carries the peak
            # the calibration is fitted against.  Every stroke is kept
            # separately in sim_log_strokes.json.
            if not label.startswith("release") and len(right) == 0 and len(left) == 0:
                print(f"      [WARN] {label} made no contact -- the platen "
                      f"never reached the bar", flush=True)
            if not label.startswith("release"):
                if log and log[-1]["hit"] == k + 1:
                    log[-1] = rec
                else:
                    log.append(rec)
            print(f"      {label} OK - {stepper.step_count} steps, "
                  f"section {F_sec:.2f} kN (x>0 {abs(F_up_x0):.2f}, "
                  f"rigid {F_rb:+.2f})  contact-set {F:.2f} kN "
                  f"(n {len(left)}/{len(right)})", flush=True)
            st = getattr(__import__("jax_fem_checkpoint.solver_latest",
                                    fromlist=["_"]), "_agf_gpu_stats", None)
            if st is not None:
                n = st["gpu_ok"] + st["fallback"]
                print(f"      [linear] so far {n} solves: {st['gpu_ok']} on the GPU "
                      f"({st['t_gpu']:.0f} s), {st['fallback']} redone with MUMPS "
                      f"({st['t_fallback']:.0f} s)", flush=True)
                rec["linear_solves"] = dict(st)
            (out / "sim_log.json").write_text(json.dumps(log, indent=2))
            (out / "sim_log_strokes.json").write_text(
                json.dumps(strokes_log, indent=2))
            if args.save_vtk and getattr(stepper, "last_cell_infos", None) is not None:
                cd = {n: [v] for n, v in stepper.last_cell_infos}
                meshio.Mesh(pts + onp.asarray(sol_u), {cell_type: cells_out},
                            point_data={"Displacement": onp.asarray(sol_u),
                                        "Temperature": onp.asarray(sol_dT)},
                            cell_data=cd).write(out / f"sim_{k+1:04d}_{SLUG[label]}.vtu")
            meshio.Mesh(pts + onp.asarray(sol_u),
                        {cell_type: cells_out}).write(
                out / f"sim_{k+1:04d}.vtu")

            if (k == len(actions) - 1 and
                    ((args.stop_after == "stroke1" and label == "stroke 1") or
                     (args.stop_after == "release" and label == "release"))):
                print(f"    [stop-after] {args.stop_after} reached -- ending the run here",
                      flush=True)
                break

        if failed:
            break
        hit_done = not (k == len(actions) - 1 and args.stop_after != "none")
        if hit_done:
            save_checkpoint(k + 1)

    print(f"\n{len(log)}/{args.n_hits} hits in {(time.time()-t0)/60:.1f} min -> {out}",
          flush=True)
    # exit status: 0 only when every requested hit was completed (or the run
    # ended where --stop-after asked it to); a STOP or FAIL used to return 0 too
    if args.fast_linear:
        from lib.fast_linear import stats as _fast_stats
        print("  " + _fast_stats(), flush=True)
    complete = (not failed) and (len(log) >= args.n_hits or args.stop_after != "none")
    if not complete:
        print(f"[EXIT 1] run incomplete: {len(log)}/{args.n_hits} hits", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
