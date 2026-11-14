"""ThermalMechanical Problem class — J2 finite-strain thermo-plasticity.

Contains:
  - Constitutive model (temperature-dependent J2 with F-bar)
  - Coupled weak form (universal_kernel + surface Neumann)
  - Post-processing helpers (stress, von Mises, internal-variable update)
"""

import numpy as onp
import jax
import jax.numpy as np

from jax_fem_checkpoint.problem import Problem

# Room-temperature hardening: constant tables, no temperature dependence.
Q_temps = np.array([0.0, 1000.0])
Q_reducs = np.array([1.0, 1.0])
b_temps = np.array([0.0, 1000.0])
b_reducs = np.array([1.0, 1.0])


# =============================================================================
class ThermalMechanical(Problem):

    # How the F-bar stress is paired with the shape-function gradients.  Read
    # when the kernels are traced, so set it before the first residual.
    #   "legacy" : P = tau(Fbar) Fbar^-T.  Differs from "stress" by the factor
    #              (J / J_c)^(-2/3) wherever J varies inside the element; equal
    #              under uniform deformation and on constant-strain TET4.
    #   "stress" : stress-replacement F-bar, int sigma(Fbar) : grad_x N dv, i.e.
    #              P = (J / J_c) tau(Fbar) F^-T with F the actual gradient.
    fbar_form = "legacy"

    def custom_init(self):
        self.fe_u = self.fes[0]
        self.fe_T = self.fes[1]

        self.S = 1.0  # unity/meters

        # Define parameters
        self.T0 = 400.0  # isothermal reference, C (no thermal strain at T = T0)

        # --- 6063 aluminium, isothermal 400 C ------------------------------
        # Elastic constants and the initial yield are handbook values for
        # ASTM B221 stock (data/calibration/material_6063_T52.json).  Q and b
        # are the Voce hardening parameters this experiment exists to
        # identify, so the driver overwrites them before custom_init().
        self.E = 54.0e3 / (self.S**2)    # MPa; handbook order at 400 C (~20 % below RT); not identifiable here
        self.sig0 = 75.0 / (self.S**2)   # MPa; MPM's choice, and what the roll1/roll2 force ratio implies (~65-70)
        self.nu = 0.33

        rho = 2700e-12 / (self.S**3)     # 2700 kg/m^3 -> tonne/mm^3

        # Isothermal model: no thermal strain.  This used to be 23.4e-6 with a
        # comment calling it inactive, but get_tau applies it whatever the
        # tables do: the dissipation heated the bar from 400 to 416 C, about
        # 59 MPa of isotropic stress.
        self.alpha_expansion = 0.0       # 1/K
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.kappa = self.alpha_expansion * (2 * self.mu + 3 * self.lmbda)

        self.C = 1000.e6 * rho           # specific heat per unit volume (400 C)
        self.k = 200.0 / self.S          # thermal conductivity

        self.dt = 0.1  # secs

        # No plastic dissipation either, so T stays uniform at T0 and the thermal
        # rows of the Newton residual are exactly zero.  With Chi = 0.85 they
        # were not: k lap(T) survives any dt, in mW, inside the norm the Newton
        # test reads in N (a first residual of 948 that did not change as dt
        # was halved, on hit 4 of pen_flat_h20_12).
        self.Chi = 0.0
        self.Q = 0.0 / (self.S**2)       # hot working: dynamic recovery, no net hardening (MPM: JC_B = 0)
        self.b = 1.0                     # irrelevant while Q = 0

        # Every temperature table is reduced to two identical knots, so
        # np.interp returns the constant whatever T is and the thermal field
        # cannot reach the mechanics.  Keeping the tables (rather than
        # stripping the interpolation out) leaves the return map, F-bar and
        # contact code byte-identical to the version already exercised on the
        # hot 15-5PH runs.
        self.yield_stress_temps = np.array([0.0, 1000.0])
        self.yield_stress_reducs = np.array([1.0, 1.0])
        self.modulus_temps = np.array([0.0, 1000.0])
        self.modulus_reducs = np.array([1.0, 1.0])

        # High-temperature modulus floor (safety net): clamp the modulus
        # reduction factor from below so the elastic stiffness never collapses
        # toward zero in the thermally softened region, which would make the
        # mechanical tangent near-singular and stall Newton.  The table already
        # plateaus at 5.8/196; this guarantees the floor even if the table is
        # edited or T is extrapolated beyond 1100 C.
        self.modulus_reduc_floor = 5.8 / 196.

        # Deformation gradient (identity initially)
        self.F_old = np.repeat(
            np.repeat(np.eye(self.dim)[None, None, :, :], len(self.fe_u.cells), axis=0),
            self.fe_u.num_quads, axis=1)

        # Elastic left Cauchy-Green tensor (identity initially)
        self.Be_old = np.array(self.F_old)

        # Accumulated plastic strain (scalar, zero initially)
        self.alpha_old = np.zeros((len(self.fe_u.cells), self.fe_u.num_quads))

        # Shape function gradients at element center for F-bar method
        self.shape_grads_center = self.fe_u.shape_grads_center
        self.ugrad_center = np.zeros_like(self.F_old)

        self.fe_u.flex_inds = np.arange(len(self.fe_u.cells))

        # Material parameters per cell (11 parameters, currently using only 4)
        full_params = np.ones((self.fe_u.num_cells, 11))
        self.thetas = np.repeat(full_params[:, None, :], self.fe_u.num_quads, axis=1)

        # Reference volume ratio J_ref = det(F) of the current reference
        # configuration measured from the ORIGINAL stress-free one.  It is 1
        # for a virgin mesh and after a reheat reset (stress-free by physics),
        # and it is carried per Gauss point through a relocation inside a
        # stroke (run_cold --stroke-relocate), where the committed shape
        # becomes the new reference but the state is kept: the pressure is a
        # function of the TOTAL volume ratio J_ref * det(F), not of det(F)
        # measured from the new reference.
        self.J_ref = np.ones((len(self.fe_u.cells), self.fe_u.num_quads))

        # [6]: J_ref   [-2]: T_old   [-1]: dt
        self.internal_vars = [
            self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center,
            self.ugrad_center, self.thetas, self.J_ref,
            np.zeros((len(self.fe_T.cells), self.fe_T.num_quads)),
            self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads)),
        ]

        self.nodes_to_compare = None
        self.target_displacement = None

    # -----------------------------------------------------------------
    def set_params(self, params):
        print("Update sol_u_old and sol_dT_old ...")
        int_vars, scale, sol, rho_ini, sol_dT_old = params

        scale = jax.lax.stop_gradient(scale)
        int_vars = jax.lax.stop_gradient(int_vars)
        sol = jax.lax.stop_gradient(sol)

        self.scale = scale
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = int_vars

        scales = scale * onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        full_params = np.ones((self.fe_u.num_cells, len(rho_ini)))
        full_params = full_params.at[self.fe_u.flex_inds].set(rho_ini)
        self.thetas = np.repeat(full_params[:, None, :], self.fe_u.num_quads, axis=1)

        self.internal_vars = [
            a1, a2, a3, a4, a5, self.thetas, a7,
            self.fe_T.convert_from_dof_to_quad(sol_dT_old),
            self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads)),
        ]

        sol_dT_old_left = self.fes[1].convert_from_dof_to_face_quad(sol_dT_old, self.boundary_inds_list[0])
        sol_dT_old_right = self.fes[1].convert_from_dof_to_face_quad(sol_dT_old, self.boundary_inds_list[1])

        # [die-contact] Nodal contact mask -> face quadrature points, so the
        # surface kernel can switch the die flux off where nothing touches.
        # Interpolating through the shape functions gives values in [0, 1]
        # rather than a hard 0/1: a face straddling the edge of the contact
        # patch gets partial flux, which is both physically sensible and
        # gentler on Newton than a jump.  Fallback of all-ones reproduces the
        # previous behaviour for the seed residual evaluation, before
        # bc_update_fn has run for the first time.
        contact_mask = getattr(self, "contact_mask_nodes", None)
        if contact_mask is None:
            contact_mask = np.ones((self.fes[1].num_total_nodes, 1))
        mask_left = self.fes[1].convert_from_dof_to_face_quad(contact_mask, self.boundary_inds_list[0])
        mask_right = self.fes[1].convert_from_dof_to_face_quad(contact_mask, self.boundary_inds_list[1])

        self.internal_vars_surfaces = [[sol_dT_old_left, mask_left],
                                       [sol_dT_old_right, mask_right]]

    # =================================================================
    # Axial contact friction (lib/friction.py)
    # =================================================================
    # When self.friction is set, the die's axial friction force on the
    # contact nodes is added to the residual as an external nodal force, and
    # its derivative to the tangent as a diagonal entry on each x dof.  The
    # package assembles the matrix from (self.I, self.J, self.V) and sums
    # duplicates, so the diagonal is appended as extra triplets: one per node,
    # always, with zero stiffness where friction is inactive, so the triplet
    # count does not depend on the active set.  self.I / self.J are rebuilt by
    # refresh_problem_surface_integrals at every stroke; the extension is
    # re-applied whenever their length shows it is missing.
    # Generalised to a list of nodal force terms: the axial friction
    # (self.friction, lib/friction.py) and the force-based die contact
    # (self.die_contact, lib/die_contact.py).  Each term provides
    # (I_ext, J_ext, n_ext) -- the triplet positions of its tangent, fixed for
    # the life of the term -- and nodal_terms(u, with_tangent) -> (F, V_ext):
    # the force of the die on the nodes, subtracted from the residual, and
    # dF/du on those positions, appended as -V_ext.  The base triplets
    # (cells + faces) always come first in self.I / self.J; whenever the set
    # of terms or the base length changes the extension is rebuilt from the
    # base part, so refresh_problem_surface_integrals and strokes without a
    # term both stay consistent.
    def _nodal_terms(self):
        return [t for t in (getattr(self, "friction", None),
                            getattr(self, "die_contact", None)) if t is not None]

    def _nodal_force_terms(self, res_list, sol_list, with_tangent):
        terms = self._nodal_terms()
        if with_tangent:
            n_base = len(self.V)
            layout = tuple((id(t), int(t.n_ext)) for t in terms)
            n_ext = sum(n for _, n in layout)
            if len(self.I) < n_base:
                raise RuntimeError(f"sparse triplets out of step (I {len(self.I)} < V {n_base})")
            if getattr(self, "_ext_layout", None) != layout or len(self.I) != n_base + n_ext:
                self.I = onp.hstack([self.I[:n_base]] + [t.I_ext for t in terms])
                self.J = onp.hstack([self.J[:n_base]] + [t.J_ext for t in terms])
                self._ext_layout = layout
        if not terms:
            return res_list
        u = onp.asarray(sol_list[0])
        res_u = res_list[0]
        for t in terms:
            F, V = t.nodal_terms(u, with_tangent)
            res_u = res_u - np.asarray(F)
            if with_tangent:
                self.V = onp.hstack((self.V, -onp.asarray(V)))
        return [res_u] + list(res_list[1:])

    def newton_update(self, sol_list):
        res_list = super().newton_update(sol_list)
        return self._nodal_force_terms(res_list, sol_list, True)

    def compute_residual(self, sol_list):
        res_list = super().compute_residual(sol_list)
        return self._nodal_force_terms(res_list, sol_list, False)

    def set_timestep(self, dt):
        self.dt = dt
        self.internal_vars[-1] = self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads))

    # =================================================================
    # Constitutive model & maps
    # =================================================================
    def get_maps(self):
        """Define constitutive model and post-processing functions."""
        def modulus_temp(T):
            T = np.asarray(T).reshape(())
            reduc = np.interp(T, self.modulus_temps, self.modulus_reducs)
            reduc = np.maximum(reduc, self.modulus_reduc_floor)
            return reduc * self.E

        def yield_stress_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T, self.yield_stress_temps, self.yield_stress_reducs) * self.sig0

        def get_partial_tensor_map(F_old, be_old, alpha_old, shape_grads_center,
                                   ugrad0, theta, J_ref, T_old, dt):
            """J2 plasticity with nonlinear isotropic hardening, F-bar method.

            Kinematics are measured from the CURRENT reference configuration
            (F, F_old, be_old = be_bar det(F_old)^(2/3)); J_ref is that
            reference's own volume ratio to the original stress-free body, so
            the total volume ratio is J_ref * det(F).  The Kirchhoff stress
            tau is per unit ORIGINAL volume; per unit current-reference volume
            it is tau / J_ref, which is what the weak form integrates."""
            K = modulus_temp(T_old) / (3. * (1. - 2. * self.nu))
            G = modulus_temp(T_old) / (2. * (1. + self.nu))
            sig0 = yield_stress_temp(T_old)

            def first_PK_stress(u_grad):
                F, _, _, tau, _, _ = return_map(u_grad)
                if self.fbar_form == "stress":
                    F_act = u_grad + np.eye(self.dim)
                    return ((np.linalg.det(F_act) / np.linalg.det(F))
                            * tau @ np.linalg.inv(F_act).T) / J_ref
                if self.fbar_form != "legacy":
                    raise ValueError(f"unknown fbar_form {self.fbar_form!r}")
                P = tau @ np.linalg.inv(F).T / J_ref
                return P

            def update_int_vars(u_grad):
                F, be_bar, alpha, _, ugrad0_updated, Delta_gamma_final = return_map(u_grad)
                return F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta

            def compute_cauchy_stress(u_grad):
                F, _, _, tau, _, _ = return_map(u_grad)
                J = J_ref * np.linalg.det(F)
                sigma = (1. / J) * tau
                return sigma

            def compute_lagrangian_strain(u_grad):
                F = u_grad + np.eye(self.dim)
                strain = 0.5 * (F @ F.T - np.eye(self.dim))
                return strain

            def compute_detla_gamma(u_grad):
                _, _, _, _, _, Delta_gamma_final = return_map(u_grad)
                return Delta_gamma_final

            def get_tau(F, be_bar, T_old):
                J = J_ref * np.linalg.det(F)          # total volume ratio
                p = 0.5 * K * (J**2 - 1.) / J
                s = G * deviatoric(be_bar)
                beta = modulus_temp(T_old) / (1.0 - 2.0 * self.nu) * self.alpha_expansion
                tau = J * (p - beta * (T_old - self.T0)) * np.eye(self.dim) + s
                return tau

            def deviatoric(A):
                return A - 1. / self.dim * np.trace(A) * np.eye(self.dim)

            def hard_satu_temp(T):
                T = np.asarray(T).reshape(())
                return np.interp(T, Q_temps, Q_reducs) * self.Q

            def hard_rate_temp(T):
                T = np.asarray(T).reshape(())
                return np.interp(T, b_temps, b_reducs) * self.b

            def K_fun(a, T):
                Q_ = hard_satu_temp(T)
                b_ = hard_rate_temp(T)
                return Q_ * (1. - np.exp(-b_ * a))

            def return_map(u_grad):
                be_bar_old = (np.linalg.det(F_old)**(-2. / 3.)) * be_old
                Fact = u_grad + np.eye(self.dim)
                F0 = ugrad0 + np.eye(self.dim)
                F = ((np.linalg.det(F0) / np.linalg.det(Fact))**(1. / 3.)) * Fact

                F_old_inv = np.linalg.inv(F_old)
                f = F @ F_old_inv

                f_bar = (np.linalg.det(f)**(-1. / 3.)) * f
                be_bar_trial = f_bar @ be_bar_old @ f_bar.T
                s_trial = G * deviatoric(be_bar_trial)
                Ie_bar = (1. / 3.) * np.trace(be_bar_trial)
                G_bar = Ie_bar * G

                yield_f_trial = np.linalg.norm(s_trial) - np.sqrt(2. / 3.) * (sig0 + K_fun(alpha_old, T_old))

                Delta_gamma = np.where(yield_f_trial > 0., newton_conv_mod(s_trial, be_bar_trial), 0.)
                direction = np.where(Delta_gamma > 0., s_trial / np.linalg.norm(s_trial), 0.)

                s = s_trial - 2. * G_bar * Delta_gamma * direction
                alpha = alpha_old + np.sqrt(2. / 3.) * Delta_gamma
                be_bar = (s / G) + Ie_bar * np.eye(self.dim)

                tau = get_tau(F, be_bar, T_old)
                be_updated = be_bar * (np.linalg.det(F)**(2. / 3.))

                return F, be_updated, alpha, tau, ugrad0, Delta_gamma

            def newton_conv_mod(s_trial, be_bar_trial):
                Ie_bar = (1. / 3.) * np.trace(be_bar_trial)
                G_bar = Ie_bar * G

                def implicit_residual(d_gamma):
                    alpha_eval = alpha_old + np.sqrt(2. / 3.) * d_gamma
                    stress_dgamma = np.where(alpha_eval > 0., K_fun(alpha_eval, T_old), 0.)
                    res = (np.linalg.norm(s_trial) - np.sqrt(2. / 3.) * sig0
                           - (np.sqrt(2. / 3.) * stress_dgamma + 2. * G_bar * d_gamma))
                    return res

                def body_fun(carry, _):
                    d_gamma, converged = carry
                    res = implicit_residual(d_gamma)
                    res_grad = jax.grad(implicit_residual)(d_gamma)
                    d_gamma_u = jax.lax.cond(converged, lambda d: d,
                                             lambda d: d - (res / res_grad), d_gamma)
                    res = implicit_residual(d_gamma_u)
                    converged_updated = np.linalg.norm(res) < tol
                    return (d_gamma_u, converged_updated), None

                tol = 1.e-6
                max_iters = 50
                carry_init = (0.0, False)
                (d_gamma_final, _), _ = jax.lax.scan(body_fun, carry_init, None, length=max_iters)
                return d_gamma_final

            return (first_PK_stress, update_int_vars, compute_cauchy_stress,
                    compute_lagrangian_strain, compute_detla_gamma)

        # ----- top-level map wrappers -----
        def tensor_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, J_ref, T_old, dt):
            fn, _, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, J_ref, T_old, dt)
            return fn(u_grad)

        def update_int_vars_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, J_ref, T_old, dt):
            _, fn, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, J_ref, T_old, dt)
            return fn(u_grad)

        def compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, J_ref, T_old, dt):
            _, _, fn, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, J_ref, T_old, dt)
            return fn(u_grad)

        def compute_lagrangian_strain_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, J_ref, T_old, dt):
            _, _, _, fn, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, J_ref, T_old, dt)
            return fn(u_grad)

        def compute_detla_gamma_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, J_ref, T_old, dt):
            _, _, _, _, fn = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, J_ref, T_old, dt)
            return fn(u_grad)

        return (tensor_map, update_int_vars_map, compute_cauchy_stress_map,
                compute_lagrangian_strain_map, compute_detla_gamma_map)

    # ----- convenience accessors -----
    def get_stress_return_map(self):
        return self.get_maps()[0]

    def get_update_int_vars_map(self):
        return self.get_maps()[1]

    def get_cauchy_stress_map(self):
        return self.get_maps()[2]

    def get_detla_gamma_map(self):
        return self.get_maps()[4]

    # =================================================================
    # Universal kernel (coupled weak form)
    # =================================================================
    def get_universal_kernel(self):
        stress_return_map = self.get_stress_return_map()
        int_vars_map = self.get_update_int_vars_map()
        detla_gamma_map = self.get_detla_gamma_map()

        def hard_satu_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T, Q_temps, Q_reducs) * self.Q

        def hard_rate_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T, b_temps, b_reducs) * self.b

        def K_fun(a, T):
            Q_ = hard_satu_temp(T)
            b_ = hard_rate_temp(T)
            return Q_ * (1. - np.exp(-b_ * a))
        vmap_K_fun = jax.vmap(K_fun)

        def yield_stress_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T, self.yield_stress_temps, self.yield_stress_reducs) * self.sig0

        def universal_kernel(
                cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW,
                *cell_internal_vars):
            a1, a2, a3, a4, a5, a6, a7, T_old, dt = cell_internal_vars
            shape_grads_center = a4

            # Split
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_u, cell_sol_T = cell_sol_list
            cell_shape_grads_list = [
                cell_shape_grads[:, self.num_nodes_cumsum[i]:self.num_nodes_cumsum[i+1], :]
                for i in range(self.num_vars)]
            cell_shape_grads_u, cell_shape_grads_T = cell_shape_grads_list
            cell_v_grads_JxW_list = [
                cell_v_grads_JxW[:, self.num_nodes_cumsum[i]:self.num_nodes_cumsum[i+1], :, :]
                for i in range(self.num_vars)]
            cell_v_grads_JxW_u, cell_v_grads_JxW_T = cell_v_grads_JxW_list
            _, cell_JxW_T = cell_JxW[0], cell_JxW[1]

            # ---- Mechanical part ----
            u_grads = np.sum(cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :], axis=1)
            u_grads_center = np.sum(
                cell_sol_u[None, :, :, None] * shape_grads_center[:, :, None, :], axis=1)
            u_grads_center_reshape = u_grads_center.reshape(-1, self.fe_u.vec, self.dim)

            cell_internal_vars_updated = (a1, a2, a3, a4, u_grads_center_reshape, a6, a7, T_old, dt)
            u_grads_reshape = u_grads.reshape(-1, self.fe_u.vec, self.dim)

            u_physics = jax.vmap(stress_return_map)(
                u_grads_reshape, *cell_internal_vars_updated).reshape(u_grads.shape)
            val4 = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))

            # ---- Thermal part ----
            T = np.sum(cell_sol_T[None, :, :] * self.fe_T.shape_vals[:, :, None], axis=1)

            # dt comes from the internal variable, not from self.dt: the kernel
            # is jitted, so self.dt is frozen at the first trace of the stroke
            # and every later step, however small, would conduct for that dt.
            val1 = (self.C * np.sum(
                ((T - T_old) / dt[:, None])[:, None, :] * self.fe_T.shape_vals[:, :, None]
                * cell_JxW_T[:, None, None], axis=0))

            detla_gammas = jax.vmap(detla_gamma_map)(u_grads_reshape, *cell_internal_vars_updated)

            # Hybrid explicit/implicit dissipation
            K = vmap_K_fun(a3, T_old)
            sig0 = jax.vmap(yield_stress_temp)(T_old)
            flow_stresses_old = sig0 + K

            _, _, alphas, _, _, _ = jax.vmap(int_vars_map)(
                u_grads_reshape, *cell_internal_vars_updated)
            # (flow_stresses_new computed but not used — semi-implicit for convergence)

            tmp2 = -detla_gammas * np.sqrt(2. / 3.) * flow_stresses_old / dt
            val2 = self.Chi * np.sum(
                tmp2[:, None, None] * self.fe_T.shape_vals[:, :, None]
                * cell_JxW_T[:, None, None], axis=0)

            T_grads = np.sum(
                cell_sol_T[None, :, :, None] * cell_shape_grads_T[:, :, None, :], axis=1)
            val3 = np.sum(self.k * T_grads[:, None, :, :] * cell_v_grads_JxW_T, axis=(0, -1))

            weak_form = [val4, val1 + val2 + val3]
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel

    # =================================================================
    # Surface Neumann kernels (thermal convection)
    # =================================================================
    def get_universal_kernels_surface(self):
        def thermal_neumann(T, old_T_face, contact_mask):
            # Die-contact heat extraction.  h is the interface conductance and
            # T0 the platen temperature; the mask (in [0, 1]) restricts the
            # flux to the part of the candidate band that is actually pressed
            # against a platen at this step.
            h = 10.0   # mW/mm^2-K  == 1e4 W/m^2-K, typical die interface
            T0 = 25.   # platen temperature (deg C)
            q_contact = contact_mask[0] * h * (T0 - T[0])
            return -np.array([q_contact])

        def _surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                            face_nanson_scale, old_T_face, contact_mask):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_T = cell_sol_list[1]
            face_shape_vals = face_shape_vals[:, -self.fes[1].num_nodes:]
            face_nanson_scale = face_nanson_scale[0]

            T = np.sum(cell_sol_T[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(thermal_neumann)(T, old_T_face, contact_mask)
            val_T = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None]
                           * face_nanson_scale[:, None, None], axis=0)
            val_u = np.zeros((self.fes[0].num_nodes, self.fes[0].vec))
            val = [val_u, val_T]
            return jax.flatten_util.ravel_pytree(val)[0]

        def convection_neumann_left(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                                    face_nanson_scale, old_T_left, mask_left):
            return _surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                                   face_nanson_scale, old_T_left, mask_left)

        def convection_neumann_right(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                                     face_nanson_scale, old_T_right, mask_right):
            return _surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                                   face_nanson_scale, old_T_right, mask_right)

        return [convection_neumann_left, convection_neumann_right]

    # =================================================================
    # Post-processing helpers
    # =================================================================
    def update_int_vars_gp(self, sol, int_vars):
        """Update internal variables at all Gauss points."""
        # Change-3: build the jitted vmap-vmap update ONCE and cache it.
        # Previously this re-created `jax.jit(jax.vmap(jax.vmap(...)))` from a
        # freshly-closed `get_maps()` on every call, so JAX re-traced and
        # re-compiled it for every accepted time step (called once per step at
        # time_stepper.py).  The closure only depends on `self`'s (constant)
        # material parameters, so caching is numerically identical — it merely
        # removes the per-step recompilation.
        if not hasattr(self, "_jit_update_int_vars"):
            _, update_int_vars_map, _, _, _ = self.get_maps()
            self._jit_update_int_vars = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        vmap_update = self._jit_update_int_vars

        u_grads = np.sum(
            np.take(sol, self.fe_u.cells, axis=0)[:, None, :, :, None]
            * self.fe_u.shape_grads[:, :, :, None, :], axis=2)

        shape_grads_center = self.fe_u.shape_grads_center
        u_gradsc = np.sum(
            np.take(sol, self.fe_u.cells, axis=0)[:, None, :, :, None]
            * shape_grads_center[:, :, :, None, :], axis=2)

        a1, a2, a3, a4, a5, a6, a7, a8, a9 = int_vars
        int_vars_updated = (a1, a2, a3, a4, u_gradsc, a6, a7, a8, a9)
        return vmap_update(u_grads, *int_vars_updated)

    def compute_cell_fields(self, sol, int_vars):
        """Cell-averaged Cauchy stress and von Mises at a converged state.

        Mirrors update_int_vars_gp exactly: the F-bar centre gradient is
        recomputed from `sol` and substituted for the stored one, so the stress
        written to disk is the stress the residual was built from.  Called with
        the PRE-commit internal variables (the ones Newton converged against);
        after the return map is committed the same call would describe a state
        the solver never equilibrated.  The jitted vmap is cached, as in
        update_int_vars_gp, so this costs one forward pass per step and no
        recompilation.
        """
        if not hasattr(self, "_jit_cell_stress"):
            _, _, cs_map, _, _ = self.get_maps()
            self._jit_cell_stress = jax.jit(jax.vmap(jax.vmap(cs_map)))
        cells_sol = np.take(sol, self.fe_u.cells, axis=0)
        u_grads = np.sum(cells_sol[:, None, :, :, None]
                         * self.fe_u.shape_grads[:, :, :, None, :], axis=2)
        u_gradsc = np.sum(cells_sol[:, None, :, :, None]
                          * self.fe_u.shape_grads_center[:, :, :, None, :], axis=2)
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = int_vars
        sig_q = self._jit_cell_stress(u_grads, a1, a2, a3, a4, u_gradsc, a6, a7, a8, a9)
        sig = np.mean(sig_q, axis=1)                       # (ncell, 3, 3)
        dev = sig - np.trace(sig, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
        vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
        return sig, vm

    def compute_stress(self, sol, int_vars):
        """Compute Cauchy stress at all Gauss points."""
        _, _, compute_cauchy_stress, _, _ = self.get_maps()
        vmap_cs = jax.jit(jax.vmap(jax.vmap(compute_cauchy_stress)))

        u_grads = np.sum(
            np.take(sol, self.fe_u.cells, axis=0)[:, None, :, :, None]
            * self.fe_u.shape_grads[:, :, :, None, :], axis=2)
        sigma = vmap_cs(u_grads, *int_vars)
        print("sigma", sigma.shape)
        return sigma

    def compute_von_mises(self, s):
        """Compute von Mises equivalent stress."""
        def von_mises(sigma):
            return np.sqrt(
                0.5 * ((sigma[0][0] - sigma[1][1])**2
                       + (sigma[1][1] - sigma[2][2])**2
                       + (sigma[2][2] - sigma[0][0])**2)
                + 3 * ((sigma[0][1])**2 + (sigma[1][2])**2 + (sigma[2][0])**2))
        return jax.vmap(von_mises)(s)
