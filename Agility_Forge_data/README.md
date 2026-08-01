# Agility_Forge_data

Multi-hit thermo-mechanical forging simulation on a cylindrical billet, used to
generate (input state, hit parameters) -> (output state) training pairs.

Each *hit* presses two rigid platens into the bar over a narrow axial band, then
the bar is rotated about its axis before the next hit. Contact is re-detected at
every load step from the current deformed shape (updated-Lagrangian), so material
pushed away from a platen is released automatically. Between hits the temperature
is relaxed back towards the initial profile.

---

## 1. Quick start

Run from the **repository root** (the directory containing `applications/` and
`jax_fem_checkpoint/`), not from inside this folder:

```bash
python -m applications.Agility_Forge_data.main -s one_press -n 1 --no-noise --save-vtk
```

That is 5 hits, ~12 min on GPU. Results go to

```
applications/Agility_Forge_data/data/vtk_one_press_s0000/hit_<N>/tm_<step>.vtu
applications/Agility_Forge_data/data/forge_database.db
```

Each `.vtu` carries `Displacement` (mm) and `Temperature` (deg C) on 7533 nodes.

## 2. Command-line options

Every flag overrides the matching entry in `configs/forging.yml`; omit it to keep
the YAML value.

| flag | meaning |
|---|---|
| `-s, --sequence NAME [NAME ...]` | which sequence(s) to run (default: all) |
| `-n, --n-series N` | independent series per sequence (default: `run.n_series`) |
| `--seed N` | RNG seed for the hit noise |
| `--save-vtk` / `--no-save-vtk` | write per-step VTK |
| `--noise` / `--no-noise` | randomise bite depth, rotation and band centre |
| `-c, --config PATH` | alternative `forging.yml` |
| `--db PATH` | SQLite output path |

Use `--no-noise` for reproducible single runs; leave noise on when generating
training data.

## 3. Available sequences

Stock is 96.52 mm long with a 15.9 x 15.9 mm cross-section; the die footprint is
19.3 mm (`scan.band_width_mm`).

### Data-generation sequences

| sequence | hits | bite | landing points | purpose |
|---|---:|---:|---|---|
| `one_press` | 5 | 0.40 mm | 0.4 ... 0.8 | one depth, no rotation |
| `press_rotate` | 10 | 0.40 mm | 0.4 ... 0.8 | one depth, 0 deg and 90 deg |
| `progressive` | 20 | 0.40 mm | 0.4 ... 0.8 | two depths, two rotations |
| `heavy_forge` | 40 | 0.35 mm | 0.4 ... 0.8 | two depths, four rotations |

```bash
# 100 noisy series of the 20-hit sequence, no VTK (dataset generation)
python -m applications.Agility_Forge_data.main -s progressive -n 100 --noise --no-save-vtk

# every sequence, 3 series each
python -m applications.Agility_Forge_data.main -n 3
```

### Validation scenarios

These exercise the deformed-band mapping and the die-contact model at their
limits. `bite %` is the fraction of the 15.9 mm section removed per hit.

| sequence | hits | bite | landing | what it probes |
|---|---:|---|---|---|
| `s_baseline` | 10 | 0.40 mm (2.5%) | 0.4 ... 0.8 | small-strain reference |
| `s_edge_low` | 8 | 0.40 mm (2.5%) | 0.05 | die overhangs the **clamped** end |
| `s_edge_high` | 8 | 0.40 mm (2.5%) | 0.95 | die overhangs the **free** end |
| `s_norot` | 10 | 0.40 mm (2.5%) | 0.4 ... 0.8 | no rotation, section flattens one way |
| `b_baseline` | 10 | 2.50 mm (15.7%) | 0.4 ... 0.8 | large-strain reference |
| `b_concentrated` | 8 | 2.50 mm (15.7%) | 0.50 | every hit on one spot |
| `b_edge` | 8 | 2.50 mm (15.7%) | 0.90 | large strain at the free end |
| `b_extreme` | 6 | 3.50 mm (22.0%) | 0.5 ... 0.7 | convergence limit |

```bash
# small-strain regression (~25 min on GPU)
python -m applications.Agility_Forge_data.main -s s_baseline -n 1 --no-noise --save-vtk

# large-strain regression (~65 min on GPU)
python -m applications.Agility_Forge_data.main -s b_baseline -n 1 --no-noise --save-vtk

# all four small-strain scenarios in one go
python -m applications.Agility_Forge_data.main -s s_baseline s_edge_low s_edge_high s_norot -n 1 --no-noise --save-vtk
```

Expected outcome (measured, no noise, GPU): all pass except `b_concentrated`,
which stops at hit 8 during the inter-hit thermal relaxation. That is not a
solver defect: 8 x 2.50 mm = 20 mm of cumulative bite on a single spot exceeds
the 15.9 mm section, the cross-section is squeezed to 16.2 x 21.8 mm and the
smallest element is down to 31% of its original volume. `b_edge` applies the same
bite at a different landing point and completes normally.

## 4. Configuring your own sequence

Add a block to `configs/forging.yml`:

```yaml
my_sequence:
  description: 'what this does'
  hot_zone: [0.3, 0.9]        # optional; overrides scan.hot_zone
  position_configs:
    - depths: [1.0, 1.5]      # mm of platen travel per side, per hit
      rotations: [0.0, 90.0]  # degrees about the bar axis
```

Hits are the full product `band_centres x depths x rotations`. Band centres are
laid out across `hot_zone` with `scan.overlap` between neighbours, so a wider hot
zone yields more hits.

Shared settings under `scan:`

| key | meaning |
|---|---|
| `band_width_mm` | physical die width in mm, constant for the whole run |
| `hot_zone` | forgeable span, as a fraction of the **current** bar length |
| `overlap` | overlap between adjacent band positions |
| `v_platen` | platen speed in mm/s (`total_time = bite / v_platen`) |

Two warnings can appear and are worth reading:

* `[band][warn] die overhangs the ... end` — the landing point puts part of the
  die past the end of the stock. Clamping is physically correct (only part of the
  die face touches metal) but the contact area, and hence the pressure, is
  reduced by the reported fraction.
* `[hits][warn] hot_zone lower bound puts the die edge at ... mm` — forging next
  to the clamped end face is legal but numerically harder, because the plastic
  zone cannot flow that way. Measured: a band reaching x = 0 produced 10-14
  step-halving events over 8 hits; the mirrored case at the free end produced
  none. Keep the lower bound at 0.3 or above unless you specifically want this.

## 5. How the die follows the bar

The bar elongates as it is forged (up to +17% in `b_baseline`), so the die
position and width are recomputed before every hit rather than being fixed to the
undeformed bar:

* `hot_zone` is a fraction of the **current** forgeable span, so "60% along the
  bar" stays at 60% as the bar grows.
* `band_width_mm` is a **physical** width. The die is steel and does not widen
  with the workpiece, so the corresponding band on the reference mesh shrinks as
  the material stretches (19.30 mm -> 16.44 mm over 10 hits in `b_baseline`).
* The mapping back onto the reference mesh **inverts the actual displacement
  field** instead of assuming uniform stretch. Cogging is deliberately
  non-uniform — hit sections elongate, untouched sections do not — and with a
  uniform-stretch assumption the die can land several millimetres off target.

Each hit prints the result:

```
[band] rod L=104.57mm, landing 47.28mm (frac 0.500) -> ref band 18.05mm for 19.30mm die
```


