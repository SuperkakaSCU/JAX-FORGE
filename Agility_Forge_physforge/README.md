# Agility_Forge_physforge

Reproduction of a **recorded physical forging run** (17 hits, `2026-06-29`) on a
rectangular billet, using the same thermo-mechanical solver as
`Agility_Forge_data`.

Where `Agility_Forge_data` generates synthetic training data on a cylinder, this
package replays measured hit parameters and compares the simulated bar against
the before/after 3-D scans of the real workpiece.

---

## 1. Quick start

Run from the **repository root** (the directory containing `applications/` and
`jax_fem_checkpoint/`):

```bash
python -m applications.Agility_Forge_physforge.run_physforge \
    --n-hits 2 --save-vtk \
    --depth-source scan --axial-mode deformed --axial-constraint face_rx
```

Two hits, ~5 min on GPU. The recorded hit parameters ship with the package
(`data/physforge_hits_2026-06-29.npz`, `data/physforge_scan_depths.npz`), so no
external data file is needed.

Output:

```
applications/Agility_Forge_physforge/data/vtk_physforge/hit_<N>/tm_<step>.vtu
applications/Agility_Forge_physforge/data/physforge_full17.db
```

## 2. Recommended settings

The three flags above are **not** the defaults — the defaults reproduce the
original v1 run bit-for-bit, which is kept for reference but is known to be
wrong. Use these for any new work:

| flag | value | why |
|---|---|---|
| `--depth-source` | `scan` | `u0` in the record is the hammer **stroke**, not the plastic reduction. The measured ratio ranges 0.26-1.19 across the 17 hits, so no single scale factor works; `scan` takes the per-hit reduction straight from the before/after scans. |
| `--axial-mode` | `deformed` | The bar elongates 59 -> 93 mm. Mapping the hammer position onto the *undeformed* length runs the contact band off the end (v1 collapsed to a 1.66 mm band with 64 contact nodes at hit 13). |
| `--axial-constraint` | `face_rx` | The default `face` leaves one rigid-body mode unconstrained (rotation about x through the pin: it produces no `ux`, so none of the 289 face constraints see it). `face_rx` adds one transverse constraint to fix it. |

Full 17-hit run:

```bash
python -m applications.Agility_Forge_physforge.run_physforge --save-vtk \
    --depth-source scan --axial-mode deformed --axial-constraint face_rx \
    --checkpoint-dir ~/physforge_ck --vtk-subdir vtk_physforge_v7 \
    --db applications/Agility_Forge_physforge/data/physforge_v7.db
```

~60 min on GPU. `--checkpoint-dir` writes `(displacement, temperature,
plasticity history)` after every hit; if the run is interrupted, resume with
`--resume-from N` where N is the last completed hit. Without a checkpoint there
is no way to continue: the plastic history lives only in memory, and neither the
VTK nor the DB stores it.

## 3. Command-line options

| flag | default | meaning |
|---|---|---|
| `--hits PATH` | `data/physforge_hits_2026-06-29.npz` | recorded hit parameters |
| `--n-hits N` | all 17 | run only the first N hits |
| `--depth-source {u0,scan}` | `u0` | bite depth from the raw stroke, or from the scans |
| `--scan-depths PATH` | `data/physforge_scan_depths.npz` | measured per-hit reduction |
| `--closure-factor F` | 1.0 | scales `u0`; ignored when `--depth-source scan` |
| `--axial-mode {reference,deformed}` | `reference` | map the hammer position on the original or the current bar |
| `--axial-constraint {face,face_rx,minimal3}` | `face` | how axial rigid-body motion is suppressed |
| `--band-width MM` | 8.0 | hammer footprint (assumed, not measured) |
| `--mesh PATH` | none | load the billet from a mesh file instead of generating a box; the bar axis must be x |
| `--ele-type {TET4,HEX8}` | auto | element type (auto-detected from `--mesh`, TET4 for a generated box) |
| `--surface-mode {topological,box}` | `topological` | how the outer lateral surface is found |
| `--depth-scale F` | 1.0 | multiply every bite by F, for stress-testing convergence |
| `--temp C` | 1000 | billet temperature (not in the record) |
| `--nx --ny --nz` | 30 16 16 | mesh resolution |
| `--checkpoint-dir DIR` | none | write per-hit checkpoints |
| `--resume-from N` | 0 | continue from checkpoint N |
| `--vtk-subdir NAME` | `vtk_physforge` | output folder under `data/` |
| `--save-vtk` / `--db PATH` | — | output control |

## 4. Arbitrary cross-sections

The bar axis must be x and the platens close along y, but the cross-section is
free.  Pass any meshio-readable file:

```bash
python -m applications.Agility_Forge_physforge.run_physforge \
    --mesh path/to/billet.msh --n-hits 2 --save-vtk \
    --depth-source scan --axial-mode deformed --axial-constraint face_rx
```

The default `--surface-mode topological` finds the outer lateral surface from
mesh connectivity -- a face belonging to exactly one element is on the boundary,
and boundary faces whose nodes all sit at x=min or x=max are the end caps.
Nothing about the cross-section is assumed, so the same code handles a box, a
cylinder, a rounded rectangle or a tapered bar.  (`--surface-mode box` keeps the
older hand-written rectangular test; on a box mesh the two select an identical
node set.)

`make_test_meshes.py` generates the five meshes used to validate this:

```bash
python -m applications.Agility_Forge_physforge.make_test_meshes
```

Measured, 2 hits each, all with 37/38 Newton steps and zero NaN:

| mesh | element | nodes | surface nodes |
|---|---|---:|---:|
| rectangular | TET4 | 1767 | 790 |
| cylinder | TET4 | 1371 | 601 |
| rounded rectangle | TET4 | 1757 | 766 |
| tapered (0.53 taper) | TET4 | 1258 | 668 |
| rectangular | HEX8 | 1296 | 512 |


## 5. Layout

```
run_physforge.py             entry point: box mesh, hit loading, CLI
main.py                      per-hit loop, band re-mapping, checkpoints
hit_config.py                Hit dataclass
lib/constitutive.py          J2 finite-strain thermo-plasticity
lib/boundary_conditions.py   dynamic contact, rigid-body constraints, die cooling
lib/time_stepper.py          adaptive stepping with step-halving
read_pt.py                   torch-free reader for the original .pt record
extract_scan_depths.py       regenerates the bundled .npz from the .pt
make_test_meshes.py          builds the five validation meshes (needs gmsh)
data/physforge_hits_*.npz    recorded hit parameters (bundled)
data/physforge_scan_depths.npz   measured per-hit reduction (bundled)
data/                        also receives VTK output and the SQLite DB
```

`read_pt.py` and `extract_scan_depths.py` are only needed to regenerate the
bundled `.npz` files from the original 318 MB `2026-06-29.pt`; normal runs do not
touch them.

## 6. Relationship to `Agility_Forge_data`

This package started as a fork and adds three things: a box-billet surface test
injected via `surface_mask_fn`, a `bc_builder` hook in `run_forging_series`, and
`run_physforge.py` itself. The solver, material model and time stepper are
shared and unmodified.

Both packages received the same die-contact mask fix (heat flux restricted to
nodes actually touching a platen, instead of every candidate node around the
circumference).
