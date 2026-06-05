# Agility_Forge_B — Version 8 Update Summary

**Version lineage**: Version 6 (`jax_forge_hits.py`) → Version 7 (modular
refactor, split into constitutive / time_stepper / boundary_conditions) →
**Version 8 (this release)**.

## Version 8 Change List (by category)

| Category | Change | Location |
|---|---|---|
| New physics / process | 4-hit rotational-forging schedule (rotations 0°/90°/180°/270°, 3.0 mm press, t=1.2 s) | `main.py` §1.0 |
| New physics / process | 10-step thermal-state relaxation ramp between hits | `main.py` §1.2 |
| Contact-model rewrite | Version 6&7 / predicate static selection → Version 8 per-step dynamic contact | `boundary_conditions.py` §3 |
| Numerical robustness | `seed_state` fixes the "first-step extrapolated guess doubled" bug | `time_stepper.py` §2.1 |
| Numerical robustness | Rebuild contact BC after relaxation (align with starting geometry) | `main.py` §1.3 |
| Numerical robustness | High-temperature modulus floor | `constitutive.py` §4.2 |
| Performance | Change-3: cache the JIT'd internal-variable update, remove per-step recompilation | `constitutive.py` §4.3 |
| Diagnostics (print-only) | DIAG-A / DIAG-B / DIAG-C | all three files |
| Housekeeping | Package import redirection; disable `CUDA_VISIBLE_DEVICES`; commented-out alternative schedules | `main.py` §1.1 |

---

## 1. `main.py`

### 1.0 Multi-hit schedule (relative to the github upstream)
Changed from the upstream **3 hits** (press depths 1.63 / 1.63 / 1.83 mm,
rotations 0° / −15° / 30°, `total_time = 0.8`) to a **4-hit rotational forging
schedule (large deformation, large rotation)**: uniform press depth `3.0` mm,
rotations `0° / 90° / 180° / 270°`, `total_time = 1.2`, with minor `x_band`
tweaks; several alternative schedules are kept commented out for easy switching.

### 1.1 Environment
Commented out the hard-coded device selection
`os.environ["CUDA_VISIBLE_DEVICES"]`.

### 1.2 New "thermal-state relaxation" stage between hits (core change)
Inside the `if hit_idx > 0:` branch, before `current_sol_dT = sol_dT_initial`,
a 10-step gradual temperature re-equilibration loop is inserted:

- Linearly interpolate the previous hit's cooled end-state temperature
  `T_cold = current_sol_dT` toward the reheated target profile
  `T_hot = sol_dT_initial`, in `n_ramp = 10` increments;
- Each increment builds an independent `AutomaticTimeStepperTM` with parameters:
  `total_time = initial_dt = max_dt = relax_dt = 1e-4`, `max_retries = 10`,
  `increase_factor = 1.0`, `line_search_after = 0`, `cool_factor = 0.0`;
- Inject the state via `seed_state(current_sol_u, T_blend, current_int_vars)`;
- Force `scale = 0` (platen fixed) with
  `bc_params_fn = lambda step, scale: (step, 0.0)`, performing mechanical
  re-equilibration only;
- Pass the converged `relax_stepper.sol_u` / `int_vars` on to the next increment.

**Motivation (as explained in the code comments)**: the be_old / F_old / alpha
accumulated from the previous hit were equilibrated under the cooled temperature
field. Switching T_old to the reheated profile in one shot makes the
thermal-stress term and the (much lower) high-temperature yield strength jump
simultaneously, producing a dt-independent static mismatch that Newton cannot
bridge in a single step (it stalls even at dt → 1e-7). Ramping gradually lets
each step carry only a small thermal mismatch.

### 1.3 Rebuild the contact BC after relaxation
After the relaxation loop and before creating the new hit's main stepper, call
`build_cylinder_press_bcs(...)` again with the post-relaxation `current_sol_u`,
followed by `refresh_problem_surface_integrals(problem)`. Reason: the original
BC was built from the previous hit's cooled end-state, so the reference planes
`y_rot_min0 / y_rot_max0` were frozen at the cold geometry; the thermal
expansion during relaxation makes the real starting geometry inconsistent with
that frozen reference, introducing a dt-independent contact jump on the new
hit's first step.

### 1.4 Use `seed_state(...)` instead of manual assignment
Original code:
```python
stepper.sol_list = [current_sol_u, current_sol_dT]
stepper.sol_u, stepper.sol_dT = [current_sol_u, current_sol_dT]
stepper.int_vars = current_int_vars
```
Changed to:
```python
stepper.seed_state(current_sol_u, current_sol_dT, current_int_vars)
```
This also resets `sol_hist / dt_hist`, fixing the hidden bug where "the
first-step extrapolated guess is doubled to 2× the state" (see §2.1).

### 1.5 New [DIAG-B] temperature-mismatch diagnostic (diagnostic-only, no physics impact)
At the end of the relaxation loop, print `||T_relaxed - T_target||_2` and
`max|ΔT|`, to check whether the relaxation truly converged the temperature to
the `sol_dT_initial` that hit-N is about to use.

---

## 2. `lib/time_stepper.py`

### 2.1 New `seed_state(sol_u, sol_dT, int_vars=None)` method
Key effect: **simultaneously** reset `sol_list`, `sol_hist`, `dt_hist` (and the
optional `int_vars`).

Bug description (written in the docstring): `__init__` initializes the history
to the zero solution; if a caller overwrites `sol_list` only, without touching
the history, the first `_extrapolate_guess` computes
```
guess = sol_n + (sol_n - 0)/dt * dt = 2 * sol_n
```
i.e. it doubles the seeded displacement/temperature. For a multi-hit case
carrying plastic history, this doubled guess inverts elements
(det F ≤ 0 → NaN), and the first Newton solve cannot recover at all.
`seed_state` points the history at the current seeded solution as well, so the
first step warm-starts with zero extrapolation velocity.

### 2.2 New [DIAG-A] diagnostic prints (diagnostic-only, no physics impact)
After the `print(f"\nStep ... try ...")` in the inner retry loop and before the
`try:` block, gated to `self.step_count == 1 and retries == 0` (the first try of
the first step of each stepper), print:
- `int_vars_old[0]` (F_old): `det` min/max, `#GP(det<=0)`, whether it contains NaN;
- `int_vars_old[1]` (be_old): eigvalsh min/max after symmetrization, and NaN flag;
- `int_vars_old[2]` (alpha_old): min/max;
- the seed state's `||u_old||`, `T_old` min/max;
- the residual `||R_u||` / `||R_T||` and NaN flags from
  `self.problem.compute_residual([sol_u_old, sol_dT_old])` evaluated at the seed
  state.

---

## 3. `lib/boundary_conditions.py` (structural rewrite: Route A (version 6&7) → Route B (version 8))

### 3.1 Full switch of the contact-modeling paradigm
- **Route A** (reference version `Agility_Forge (version 6&7)`,
  Total-Lagrangian static contact)
  - The contact node set is selected once on the initial reference configuration;
  - Each step ramps the total displacement linearly via
    `u = u_residual + scale * (u_target - u_residual)`;
  - Once selected, the set is fixed, and material pushed away is still forced to
    stick to the platen.
- **Route B** (this version `Agility_Forge_B`, Updated-Lagrangian dynamic contact)
  - The candidate set (outer-surface nodes within the x-band) is still taken
    from the reference configuration, but the contact subset is re-detected each
    step from `pts + sol_u` in the rotated frame;
  - Each contacting node keeps its current tangential position and is projected
    normally onto the moving platen plane (a frictional-stick increment);
  - Non-contacting nodes are left free, so material pushed away is released
    automatically;
  - The entire BC (`location_fns_u` and values) is rebuilt every step.

At the code level:
- Removed Route A's `target_total_displacement(...)` / `ramp_values(...)`;
- Added the internal function
  `compute_contact(sol_u_np, scale) -> (left_inds, right_inds, u_left, u_right)`,
  with dynamic plane positions `y_L = y_rot_min0 + scale*disp` and
  `y_R = y_rot_max0 - scale*disp`, and a tolerance-layer thickness
  `tol_contact = 0.10` mm;
- `bc_update_fn` now calls `compute_contact` and then directly
  `problem.fes[0].update_Dirichlet_boundary_conditions(...)`, printing the
  left/right contact node counts and `y_L / y_R` each step.

### 3.2 Plane reference changed from "reference surface" to "initial deformed surface"
- Route A used `y_rot_min / y_rot_max` (from all outer-surface reference points);
- Route B uses `y_rot_min0 / y_rot_max0` (from the candidate nodes' initial
  deformed coordinates `X_cand + current_sol_u_np[candidate_inds]`), ensuring
  each hit's platen advances from the actual current surface — this pairs with
  §1.3's "rebuild BC after relaxation" to jointly eliminate the geometric jump on
  hit-N's first step.

### 3.3 Thermal convection surface uses STATIC half-subsets
By the candidate nodes' initial-deformed y-midline, the x-band is split into
lower/upper halves as `conv_left_inds / conv_right_inds`; the thermal-convection
location_fns stay fixed within a hit (avoiding re-jitting the surface integrals
every step). In Route A the convection surface and contact surface were built
together.

### 3.4 New [DIAG-C] diagnostic prints (diagnostic-only, no physics impact)
At the end of `bc_update_fn`, print this step's `|u_left|_inf` / `|u_right|_inf` /
`||u_left||_2` / `||u_right||_2` and `tol_contact`, to check whether a contact
jump on the order of `tol_contact` is present.

### 3.5 Other minor fixes
- `rot_matrix = onp.array(rot_obj.as_matrix())` is explicitly onp, avoiding jax
  tracers when later mixed with numpy;
- `inverse_transform_vectors` renamed to `to_global_vectors` (clearer semantics);
- Several inline `pts[:, 0].min()` hoisted to `x_bottom = float(...)`, reducing
  repeated computation and debug-log noise.

---

## 4. `lib/constitutive.py`

### 4.1 Module path rename
`from applications.Agility_Forge.material_data import ...` →
`from applications.Agility_Forge_B.material_data import ...`.

### 4.2 New high-temperature modulus floor (numerical safety net)
- Added member `self.modulus_reduc_floor = 5.8 / 196.` (matching the table's
  tail plateau value);
- Inside `modulus_temp(T)`:
  ```python
  reduc = np.interp(T, self.modulus_temps, self.modulus_reducs)
  reduc = np.maximum(reduc, self.modulus_reduc_floor)
  return reduc * self.E
  ```
- Motivation (code comment): prevents the elastic stiffness from collapsing to
  zero — which would make the mechanical tangent near-singular and stall Newton —
  if the table is edited or T is extrapolated beyond 1100 °C. The current table
  already plateaus at this value, so on the normal path it is an identity map;
  this is a purely defensive floor.

### 4.3 Change-3: cache the JIT'd internal-variable update (performance, numerically equivalent)
`update_int_vars_gp` originally rebuilt
`jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))` on every call (the closure
comes from a fresh `get_maps()`), causing JAX to re-trace and re-compile on
**every accepted time step**. Changed to build once and cache in
`self._jit_update_int_vars`:
```python
if not hasattr(self, "_jit_update_int_vars"):
    _, update_int_vars_map, _, _, _ = self.get_maps()
    self._jit_update_int_vars = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
vmap_update = self._jit_update_int_vars
```
The closure depends only on `self`'s (constant) material parameters, so caching
is **numerically identical**; it merely removes the per-step recompilation,
giving a significant overall wall-time speedup.
