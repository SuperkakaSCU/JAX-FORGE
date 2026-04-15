"""Adaptive time-stepper for the coupled thermo-mechanical problem.

Contains:
  - AutomaticTimeStepperTM: adaptive dt, cooling, extrapolation, VTK output
"""

import os
import time
import numpy as onp
import jax.numpy as np

from jax_fem_checkpoint.solver_latest import solver
from jax_fem_checkpoint.utils import save_sol


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
        cool_factor=1.0,
        surface_inds=None,
    ):
        self.problem = problem
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

        # cooling parameters
        self.cool_factor = float(cool_factor)
        if surface_inds is None:
            self.surface_inds = np.array([], dtype=int)
        else:
            self.surface_inds = np.array(surface_inds, dtype=int)

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
          or by the increase_factor logic.  Immediately after ``dt_this`` is
          computed the code applies a simple cooling law to the nodes listed in
          ``self.surface_inds``:
              self.sol_list[1][self.surface_inds] *=
                  0.99**(dt_this/self.cool_factor)
          ``cool_factor`` can be tuned when the stepper is constructed.
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

            # apply cooling to surface nodes according to current dt
            if self.surface_inds.size > 0 and self.cool_factor > 0:
                try:
                    print("Applying cooling", "dt_this:", dt_this, "cool_factor:", self.cool_factor)
                    coeff = 0.99 ** (dt_this / self.cool_factor)
                    arr = np.array(self.sol_list[1])
                    arr = arr.at[self.surface_inds].multiply(coeff)
                    self.sol_list[1] = arr
                    self.sol_dT = arr
                except Exception as _e:
                    print("Warning: failed to apply cooling", _e)

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

                try:
                    print("** Try simulation with dt_this:{0}**".format(self.problem.dt))
                    new_sol_list, iteration_counter, has_converged = solver(self.problem, solver_options)

                    if not has_converged:
                        raise RuntimeError("Solver failed to converge within max_iters")

                    self.sol_list = new_sol_list
                    self.sol_u, self.sol_dT = new_sol_list

                    print("** Update internal variables after convergence")

                    # --- Debug: report temperature change after solver step ---
                    try:
                        dT_diff = np.linalg.norm(self.sol_dT - sol_dT_old)
                        print(f"  Debug: sol_dT min/max = {np.min(self.sol_dT):.6e}/{np.max(self.sol_dT):.6e}, ||ΔT||={dT_diff:.6e}")
                    except Exception as _e:
                        print("  Debug: failed to compute sol_dT stats", _e)

                    # Update internal variables after convergence
                    a1, a2, a3, a4, a5, a6, a7, a8 = self.int_vars
                    int_vars_u = self.problem.update_int_vars_gp(self.sol_u, self.int_vars)
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
