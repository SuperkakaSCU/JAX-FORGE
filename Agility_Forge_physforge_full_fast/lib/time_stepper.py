"""Adaptive time-stepper for the coupled thermo-mechanical problem.

Contains:
  - AutomaticTimeStepperTM: adaptive dt, cooling, extrapolation, VTK output
"""

import os
import time
import numpy as onp
import scipy.sparse
import scipy.sparse.linalg
import jax.numpy as np

import jax_fem_checkpoint.solver_latest as _sl
from jax_fem_checkpoint.solver_latest import solver
from jax_fem_checkpoint.utils import save_sol

from applications.Agility_Forge_physforge_full_fast.lib import fast_newton


def newton_solve(problem, solver_options):
    """Newton solve of one load step: the fast device-resident solver
    (lib/fast_newton.py) unless ``fast_newton.CONFIG['newton'] == 'original'``,
    which runs jax_fem_checkpoint's solver (CPU assembly) for comparison."""
    if fast_newton.CONFIG.get("newton", "fast") == "original":
        return solver(problem, solver_options)
    return fast_newton.solve(problem, solver_options)


# =============================================================================
# Hybrid linear solver (runtime rebinding; jax_fem_checkpoint is not modified)
# =============================================================================
HYBRID_STATS = {"solves": 0, "fallbacks": 0}



def use_hybrid_linear_solver(rel_tol=1e-6):
    """bicgstab first; redo the solve with a sparse LU when its residual fails.

    The package's ``jax_solve`` is an unpreconditioned bicgstab whose
    convergence check is commented out, so a failed linear solve is returned
    as if it had succeeded and Newton then works from a corrupted increment.
    Newton's own residual check catches most of these as "not converged" and
    costs a retry; a few can pass the loose absolute tolerance.  This wrapper
    measures ``|A x - b|`` after every solve and falls back to
    ``scipy.sparse.linalg.spsolve`` when it exceeds ``rel_tol * |b|``.  The
    direct solve is exact, so a fallback can never return a wrong increment.
    """
    if getattr(_sl, "_agf_hybrid_installed", False):
        return
    _inner = _sl.linear_solver

    def _hybrid(A, b, x0, solver_options):
        x = _inner(A, b, x0, solver_options)
        HYBRID_STATS["solves"] += 1
        if "jax_solver" not in solver_options:
            return x
        try:
            res = float(np.linalg.norm(A @ x - b))
            nb = float(np.linalg.norm(b))
        except Exception:                                   # noqa: BLE001
            return x
        if onp.isfinite(res) and res <= rel_tol * max(nb, 1.0):
            return x
        HYBRID_STATS["fallbacks"] += 1
        idx = onp.asarray(A.indices)
        A_csc = scipy.sparse.csr_matrix(
            (onp.asarray(A.data), (idx[:, 0], idx[:, 1])), shape=A.shape).tocsc()
        x_d = scipy.sparse.linalg.spsolve(A_csc, onp.asarray(b, dtype=onp.float64))
        res_d = float(onp.linalg.norm(A_csc @ x_d - onp.asarray(b)))
        print(f"  [hybrid] bicgstab residual {res:.3e} > {rel_tol:.0e} * |b|={nb:.3e}; "
              f"direct solve residual {res_d:.3e}")
        return np.asarray(x_d)

    _sl.linear_solver = _hybrid
    _sl._agf_hybrid_installed = True


class AutomaticTimeStepperTM:
    """
    Adaptive dt stepper for coupled problem with unknowns [u, T].

    - Uses extrapolated initial guess for both fields
    - Retries with smaller dt on failure
    - Activates line search after a couple retries
    - Calls:
        problem.set_timestep(dt)
        problem.set_params([int_vars, scale, sol_u_old, rho_ini, sol_dT_old])
        solver(problem, solver_options)
    """

    def __init__(
        self,
        problem,
        total_time,
        initial_dt,
        min_dt,
        max_dt,
        max_retries=8,
        increase_factor=1.2,
        decrease_factor=0.5,
        line_search_after=2,
    ):
        """
        The plasticity history (``update_int_vars_gp``) is committed with the
        quadrature temperature and time step the step's residual was built
        from (``problem.internal_vars[5:8]``), so the committed plastic state
        is the one that satisfied equilibrium.  (The original code passed the
        ``T_old``/``dt`` slots of the carried tuple, which are never written
        after ``custom_init``: T = 0 C and dt = 0.1 s.)

        Every linear solve goes through ``use_hybrid_linear_solver``: the
        package's bicgstab is checked and a failed solve is redone with a
        sparse LU.
        """
        self.problem = problem
        use_hybrid_linear_solver()
        # Internal force at the last accepted step (die reaction at the
        # constrained dofs), and Newton iterations per accepted step.
        self.last_res_u = None
        self.last_T_old = None
        self.last_dt = None
        self.iters_hist = []
        self.total_time = float(total_time)
        self.dt = float(initial_dt)
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)
        self.max_retries = int(max_retries)
        self.increase_factor = float(increase_factor)
        self.decrease_factor = float(decrease_factor)
        self.line_search_after = int(line_search_after)

        self.current_time = 0.0
        self.step_count = 0

        # State
        self.int_vars = self.problem.internal_vars

        # Initialize solution list [u, dT] (match your usage)
        self.sol_u = np.zeros((self.problem.fes[0].num_total_nodes, self.problem.fes[0].vec))
        self.sol_dT = np.zeros((self.problem.fes[1].num_total_nodes, self.problem.fes[1].vec))
        self.sol_list = [self.sol_u, self.sol_dT]

        # History for extrapolation (two previous steps)
        self.sol_hist = [self.sol_list, self.sol_list]
        self.dt_hist = [self.dt, self.dt]

        self.step_report = []
        self.total_wall = 0.0

        # Step data for plotting: [(step_count, scale, wall_time), ...]
        self.step_data = []

    def seed_state(self, sol_u, sol_dT, int_vars=None):
        """Seed the stepper with an initial [u, T] state (and optional int_vars).

        Crucially this ALSO resets the extrapolation history (``sol_hist`` /
        ``dt_hist``) to the seeded state.  ``__init__`` initialises the history
        to the zero solution; if a caller overwrites ``sol_list`` without
        touching the history, the first ``_extrapolate_guess`` computes
            guess = sol_n + (sol_n - 0)/dt * dt = 2 * sol_n
        i.e. it DOUBLES the seeded displacement/temperature.  For a carried-over
        deformed, plastically-loaded state (multi-hit) this doubled guess inverts
        elements (det F <= 0 -> NaN) and the first Newton solve cannot recover.
        Resetting the history here makes the first step warm-start from the exact
        seeded state (zero extrapolation velocity).
        """
        self.sol_list = [sol_u, sol_dT]
        self.sol_u, self.sol_dT = sol_u, sol_dT
        self.sol_hist = [self.sol_list, self.sol_list]
        self.dt_hist = [self.dt, self.dt]
        if int_vars is not None:
            self.int_vars = int_vars

    def _extrapolate_guess(self):
        """Linear extrapolation for [u, T] using last step."""
        sol_n = self.sol_list
        sol_nm1 = self.sol_hist[-1]
        dt_nm1 = self.dt_hist[-1] + 1e-12

        guess = [
            sol_n[0] + (sol_n[0] - sol_nm1[0]) / dt_nm1 * self.dt,
            sol_n[1] + (sol_n[1] - sol_nm1[1]) / dt_nm1 * self.dt,
        ]
        return guess

    def run(self, bc_update_fn=None, bc_params_fn=None, rho_ini=None, vtk_dir=None, save_every=1):
        """
        Execute adaptive time stepping for the coupled thermo-mechanical problem.

        Notes:
        - The stepper's ``total_time`` is set at construction and represents the
          duration of the current hit.
        - During each while-loop iteration ``self.dt`` may be modified by retries
          or by the increase_factor logic.
        - Heat loss to the air is a boundary condition of the problem
          (natural convection + radiation, lib/air.py); the stepper does not
          change temperatures itself.
        """
        if rho_ini is None:
            rho_ini = np.array([1.0, 1.0, 1.0, 1.0])

        t0 = time.time()

        while self.current_time < self.total_time and not np.isclose(self.current_time, self.total_time):
            print("*****Outer loop for updating time: Current Time: {0}, Total Time: {1}, Step count: {2}".format(
                self.current_time, self.total_time, self.step_count))
            self.step_count += 1
            step_start = time.time()

            # Ensure dt doesn't overshoot final time
            if self.current_time + self.dt > self.total_time:
                self.dt = self.total_time - self.current_time

            dt_this = float(self.dt)

            retries = 0
            converged = False

            sol_u_old = self.sol_list[0]
            sol_dT_old = self.sol_list[1]
            print("*****Get int_vars_old at outer_loop...")
            int_vars_old = self.int_vars

            initial_guess = self._extrapolate_guess()

            while not converged and retries < self.max_retries:
                if retries > 0:
                    dt_this *= self.decrease_factor
                    if dt_this < self.min_dt:
                        self.step_report.append((self.step_count, self.current_time, dt_this, "FAILED(dt<min)"))
                        self.total_wall = time.time() - t0
                        return False

                next_time = self.current_time + dt_this
                scale = next_time / self.total_time  # 0..1 loading ramp

                print("***Inner loop for checking convergence: Retries: {0}, Current dt: {1}, Scale: {2}, next_time: {3}".format(
                    retries, dt_this, scale, next_time))
                print("***Update dt_this and bc with scale...")
                self.problem.set_timestep(dt_this)

                # Update BCs for this scale
                if bc_update_fn is not None:
                    bc_params = bc_params_fn(self.step_count, scale) if bc_params_fn else (scale,)
                    bc_update_fn(self.problem, self.sol_list[0], *bc_params)

                # Update parameters for coupled kernel
                print("***Update int_vars_old")
                self.problem.set_params([int_vars_old, scale, sol_u_old, rho_ini, sol_dT_old])

                # Solver options
                use_ls = (retries >= self.line_search_after)
                solver_options = {
                    "jax_solver": {},
                    "initial_guess": initial_guess,
                    "line_search_flag": use_ls,
                    'return_full_info': True
                }

                print(
                    f"\nStep {self.step_count} try {retries+1} | "
                    f"time={self.current_time:.6f} dt={dt_this:.3e} scale={scale:.6f} ls={use_ls}"
                )

                # =========================================================
                # [DIAG-A] On the FIRST try of the FIRST step of this stepper
                # only: report carried-state quality and seed-state residual.
                # This brackets whether the hot-start is already ill-posed
                # (NaN / inverted elements) before Newton even begins.
                # =========================================================
                if self.step_count == 1 and retries == 0:
                    try:
                        F_old_arr = int_vars_old[0]
                        be_old_arr = int_vars_old[1]
                        det_F = np.linalg.det(F_old_arr)
                        n_bad_det = int(np.sum(det_F <= 0.0))
                        det_min = float(np.min(det_F))
                        det_max = float(np.max(det_F))
                        det_has_nan = bool(np.any(np.isnan(det_F)))
                        # symmetrise be_old for stable eigvalsh; in batches of
                        # 1e6 matrices (cuSolver's batched eigensolver rejects
                        # the ~1e7 integration points of a 2M-dof mesh, and the
                        # failed call left its memory behind at every stepper)
                        be_flat = be_old_arr.reshape(-1, 3, 3)
                        be_eig_min, be_eig_max = np.inf, -np.inf
                        for _s in range(0, be_flat.shape[0], 1_000_000):
                            _b = be_flat[_s:_s + 1_000_000]
                            _e = np.linalg.eigvalsh(0.5 * (_b + np.swapaxes(_b, -1, -2)))
                            be_eig_min = min(be_eig_min, float(np.min(_e)))
                            be_eig_max = max(be_eig_max, float(np.max(_e)))
                        be_has_nan = bool(np.any(np.isnan(be_old_arr)))
                        alpha_old_arr = int_vars_old[2]
                        alpha_min = float(np.min(alpha_old_arr))
                        alpha_max = float(np.max(alpha_old_arr))
                        u_old_norm = float(np.linalg.norm(sol_u_old))
                        T_old_min = float(np.min(sol_dT_old))
                        T_old_max = float(np.max(sol_dT_old))
                        print(f"[DIAG-A] carried F_old: det min/max = {det_min:.6e}/{det_max:.6e}, "
                              f"#GP(det<=0) = {n_bad_det}, NaN={det_has_nan}")
                        print(f"[DIAG-A] carried be_old: eig min/max = {be_eig_min:.6e}/{be_eig_max:.6e}, NaN={be_has_nan}")
                        print(f"[DIAG-A] carried alpha_old: min/max = {alpha_min:.6e}/{alpha_max:.6e}")
                        print(f"[DIAG-A] seed state: ||u_old||={u_old_norm:.6e}, T_old min/max={T_old_min:.2f}/{T_old_max:.2f} C")
                    except Exception as _diag_e:
                        print(f"[DIAG-A] carried-state stats failed: {_diag_e}")

                    try:
                        # evaluate the residual at the seed state; set_params
                        # has already been called just above with int_vars_old
                        # / sol_u_old / sol_dT_old, so this is the true hot-
                        # start residual the Newton solver will start from.
                        res_list = self.problem.compute_residual([sol_u_old, sol_dT_old])
                        res_u_norm = float(np.linalg.norm(res_list[0]))
                        res_T_norm = float(np.linalg.norm(res_list[1]))
                        res_u_nan = bool(np.any(np.isnan(res_list[0])))
                        res_T_nan = bool(np.any(np.isnan(res_list[1])))
                        print(f"[DIAG-A] seed-state residual: ||R_u||={res_u_norm:.6e} (NaN={res_u_nan}), "
                              f"||R_T||={res_T_norm:.6e} (NaN={res_T_nan})")
                    except Exception as _diag_e:
                        print(f"[DIAG-A] seed-state residual eval failed: {_diag_e}")

                try:
                    print("** Try simulation with dt_this:{0}**".format(self.problem.dt))
                    new_sol_list, iteration_counter, has_converged = newton_solve(self.problem, solver_options)

                    if not has_converged:
                        raise RuntimeError("Solver failed to converge within max_iters")

                    self.sol_list = new_sol_list
                    self.sol_u, self.sol_dT = new_sol_list
                    self.iters_hist.append(int(iteration_counter))
                    # The fields this step's residual was built with, so a
                    # later consistency check can reproduce it exactly.
                    self.last_T_old = sol_dT_old
                    self.last_dt = dt_this

                    # Internal force at the converged state, with the step's
                    # own parameters still installed.  At a Dirichlet dof it
                    # is the reaction; the unloading phase between hits
                    # releases exactly this force.
                    try:
                        if hasattr(self.problem, "internal_force"):
                            self.last_res_u = self.problem.internal_force(new_sol_list)[0]
                        else:
                            self.last_res_u = self.problem.compute_residual(new_sol_list)[0]
                    except Exception as _e:                      # noqa: BLE001
                        print("Warning: failed to capture the internal force", _e)
                        self.last_res_u = None

                    print("** Update internal variables after convergence")

                    # --- Debug: report temperature change after solver step ---
                    try:
                        dT_diff = np.linalg.norm(self.sol_dT - sol_dT_old)
                        print(f"  Debug: sol_dT min/max = {np.min(self.sol_dT):.6e}/{np.max(self.sol_dT):.6e}, ||ΔT||={dT_diff:.6e}")
                    except Exception as _e:
                        print("  Debug: failed to compute sol_dT stats", _e)

                    # Update internal variables after convergence
                    a1, a2, a3, a4, a5, a6, a7, a8 = self.int_vars
                    # The residual of this step was built with
                    # problem.internal_vars = [a1..a5, thetas, T_old_q, dt_q]
                    # (set by set_params / set_timestep); commit the history
                    # with the same fields.
                    iv_step = (a1, a2, a3, a4, a5,
                               self.problem.internal_vars[5],
                               self.problem.internal_vars[6],
                               self.problem.internal_vars[7])
                    int_vars_u = self.problem.update_int_vars_gp(self.sol_u, iv_step)
                    a1_updated, a2_updated, a3_updated, a4_updated, a5_updated, _ = int_vars_u
                    self.int_vars = (a1_updated, a2_updated, a3_updated, a4_updated, a5_updated, a6, a7, a8)

                    # accept step
                    self.current_time = next_time
                    converged = True

                    # update history
                    self.sol_hist.pop(0)
                    self.sol_hist.append(self.sol_list)
                    self.dt_hist.pop(0)
                    self.dt_hist.append(dt_this)

                    # mild dt increase if no retries
                    if retries == 0:
                        self.dt = min(self.max_dt, self.dt * self.increase_factor)

                    self.step_report.append((self.step_count, self.current_time, dt_this, "OK"))

                    # Record step data for plotting
                    step_wall_time = time.time() - step_start
                    self.step_data.append((self.step_count, scale, step_wall_time))

                    # Optional VTK output
                    if vtk_dir is not None and (self.step_count % save_every == 0):
                        vtk_path = os.path.join(vtk_dir, f"tm_{self.step_count:04d}.vtu")
                        save_sol(
                            self.problem.fes[0],
                            self.sol_u,
                            vtk_path,
                            point_infos=[("Displacement", self.sol_u), ("Temperature", self.sol_dT)],
                        )
                        print(f"  Saved: {vtk_path}")

                except (RuntimeError, FloatingPointError, AssertionError) as e:
                    print(f"  Solver failed or NaN detected: {e}")
                    retries += 1
                    self.int_vars = int_vars_old
                    initial_guess = [sol_u_old, sol_dT_old]

            if not converged:
                self.step_report.append((self.step_count, self.current_time, dt_this, f"FAILED(retries={retries})"))
                self.total_wall = time.time() - t0
                return False

        self.total_wall = time.time() - t0
        print("\n--- Coupled simulation finished successfully ---")
        return True

    def plot_step_data(self, save_path=None):
        """
        Plot step_count vs scale with wall time annotations.

        Parameters:
        -----------
        save_path : str or None
            If provided, save the plot to this path. Otherwise, display it.
        """
        if not self.step_data:
            print("No step data to plot")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        step_counts, scales, wall_times = zip(*self.step_data)

        plt.figure(figsize=(10, 6))
        plt.scatter(step_counts, scales, s=50, alpha=0.7)

        for step_count, scale, wall_time in self.step_data:
            plt.annotate(f'{int(wall_time)}s',
                         (step_count, scale),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=6,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        plt.xlabel('Step Count')
        plt.ylabel('Scale')
        plt.title('Step Count vs Scale with Wall Time Annotations')
        plt.grid(True, alpha=0.3)

        print("Scales used:", scales)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Step data plot saved to: {save_path}")

            try:
                base, _ = os.path.splitext(save_path)
                txt_path = base + "_scales.txt"
                with open(txt_path, 'w') as f:
                    for s in scales:
                        f.write(f"{s}\n")
                print(f"Scale values saved to: {txt_path}")
            except Exception as e:
                print(f"Failed to save scales to text file: {e}")
        else:
            plt.show()
