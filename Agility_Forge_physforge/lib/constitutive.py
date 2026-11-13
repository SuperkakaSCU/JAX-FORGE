"""ThermalMechanical Problem class — J2 finite-strain thermo-plasticity.

Contains:
  - Constitutive model (temperature-dependent J2 with F-bar)
  - Coupled weak form (universal_kernel + surface Neumann)
  - Post-processing helpers (stress, von Mises, internal-variable update)

Thermal discretisation (corrected November 2026; the README, Model
corrections, has the measurements):

* The heat-capacity term and the die-contact heat transfer are row-sum lumped
  onto the nodes: node i stores heat as C/dt (T_i - T_old_i) int N_i and loses
  it to the die as h (T0 - T_i) int m N_i, with its own temperature only.  The
  consistent terms of the original weak form couple neighbouring nodes with
  positive weights, so the implicit update was not monotone and the billet
  left the range between the die temperature and its hottest state.
* The thermal rate terms divide by the step's own dt (the per-quadrature
  ``dt`` argument that ``set_timestep`` / ``set_params`` update).  The
  original kernel divided by ``self.dt``, a Python float that JIT compiled in
  as a constant, so every hit kept the dt of its first solve (1e-3 s).
* The die-contact surface integral uses reference-normal area weights
  (``reference_nanson_scale`` in lib/boundary_conditions.py).
"""

import numpy as onp
import jax
import jax.numpy as np

from jax_fem_checkpoint.problem import Problem

from applications.Agility_Forge_physforge.material_data import Q_temps, Q_reducs, b_temps, b_reducs


# Die interface conductance (mW/mm^2-K == 1e4 W/m^2-K) and platen temperature (C),
# used by both forms of the die-contact kernel.
DIE_H = 10.0
DIE_T = 25.0


# =============================================================================
class ThermalMechanical(Problem):

    def custom_init(self):
        self.fe_u = self.fes[0]
        self.fe_T = self.fes[1]

        self.S = 1.0  # unity/meters

        # Define parameters
        self.T0 = 20.0   # ambient temperature, C
        self.E = 196.e3 / (self.S**2)   # young's modulus, MPa
        self.sig0 = 1213. / (self.S**2)
        self.nu = 0.272

        ### Hu: Reference -- https://www.spacematdb.com/spacemat/manudatasheets/15-5_PH_Data_Sheet.pdf
        rho = 7800e-12 / (self.S**3)  # density, 7800 kg/m^3 --> tonne/mm^3

        self.alpha_expansion = 13.5e-6
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.kappa = self.alpha_expansion * (2 * self.mu + 3 * self.lmbda)

        ## Assumption of 900K
        self.C = 600.e6 * rho   # specific heat per unit volume
        self.k = 25.0 / self.S  # thermal conductivity

        self.dt = 0.1  # secs

        self.Chi = 0.85
        self.Q = 145. / (self.S**2)  # Hardening saturation MPa
        self.b = 25.                 # Hardening rate

        ## For 24~538C, Reference: H925: ARMCO Table 13 – Short-Time Tensile Properties
        ## For 682~1100C, Reference: Agility Forge Comparison
        self.yield_stress_temps = np.array([24, 204, 315, 426, 538, 
                                            682, 800, 898, 1000, 1100])
        self.yield_stress_reducs = np.array([1213, 1048, 965, 869, 634, 
                                             330, 240, 100., 70., 70]) / 1213.


        ## For 22~315C, Reference: https://www.spacematdb.com/spacemat/manudatasheets/15-5_PH_Data_Sheet.pdf
        ## For 400C, Reference: https://www.scribd.com/document/441470740/15-5-PH-technical-data
        ## For 500C, Reference: https://kaspercalc.com/Chapter2.6%28steel%29.html
        ## For 682~1100C, Reference: Agility Forge Comparison
        self.modulus_temps = np.array([22, 38, 93, 149, 204, 260, 315, 
                                      400, 500, 
                                      682, 800, 898, 1000, 1100])
        self.modulus_reducs = np.array([1., 0.996, 0.978, 0.962, 0.947, 0.930, 0.914,
                                       170/196, 160/196,
                                       90/196, 80/196, 6/196, 5.8/196, 5.8/196])

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

        # [-2]: T_old   [-1]: dt
        self.internal_vars = [
            self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center,
            self.ugrad_center, self.thetas,
            np.zeros((len(self.fe_T.cells), self.fe_T.num_quads)),
            self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads)),
        ]

        self.nodes_to_compare = None
        # External nodal force on the displacement field (N, 3) or None.  Set
        # by the unloading phase between hits; see compute_residual below.
        self.f_ext_nodes = None
        self.target_displacement = None

    # -----------------------------------------------------------------
    def set_params(self, params):
        print("Update sol_u_old and sol_dT_old ...")
        int_vars, scale, sol, rho_ini, sol_dT_old = params

        scale = jax.lax.stop_gradient(scale)
        int_vars = jax.lax.stop_gradient(int_vars)
        sol = jax.lax.stop_gradient(sol)

        self.scale = scale
        a1, a2, a3, a4, a5, a6, a7, a8 = int_vars

        scales = scale * onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        full_params = np.ones((self.fe_u.num_cells, len(rho_ini)))
        full_params = full_params.at[self.fe_u.flex_inds].set(rho_ini)
        self.thetas = np.repeat(full_params[:, None, :], self.fe_u.num_quads, axis=1)

        self.internal_vars = [
            a1, a2, a3, a4, a5, self.thetas,
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

    def set_timestep(self, dt):
        self.dt = dt
        self.internal_vars[-1] = self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads))

    # -----------------------------------------------------------------
    # External nodal force (force-controlled unloading between hits)
    # -----------------------------------------------------------------
    # The residual is r(u) = f_int(u) - f_ext.  ``f_ext_nodes`` is a plain
    # nodal array, independent of u, so the tangent is unchanged and the row
    # elimination for Dirichlet dofs applies after the subtraction exactly as
    # before.  With ``f_ext_nodes = None`` nothing changes.
    def _subtract_f_ext(self, res_list):
        f_ext = getattr(self, "f_ext_nodes", None)
        if f_ext is None:
            return res_list
        return [res_list[0] - f_ext] + list(res_list[1:])

    def compute_residual(self, sol_list):
        return self._subtract_f_ext(super().compute_residual(sol_list))

    def newton_update(self, sol_list):
        return self._subtract_f_ext(super().newton_update(sol_list))

    def internal_force(self, sol_list):
        """Unconstrained internal force f_int(u), no external nodal force.

        At a Dirichlet dof this is the force the constraint exerts on the
        body, i.e. the die reaction at a contact node.
        """
        return super().compute_residual(sol_list)

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
                                   ugrad0, theta, T_old, dt):
            """J2 plasticity with nonlinear isotropic hardening, F-bar method."""
            K = modulus_temp(T_old) / (3. * (1. - 2. * self.nu))
            G = modulus_temp(T_old) / (2. * (1. + self.nu))
            sig0 = yield_stress_temp(T_old)

            def first_PK_stress(u_grad):
                F, _, _, tau, _, _ = return_map(u_grad)
                P = tau @ np.linalg.inv(F).T
                return P

            def update_int_vars(u_grad):
                F, be_bar, alpha, _, ugrad0_updated, Delta_gamma_final = return_map(u_grad)
                return F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta

            def compute_cauchy_stress(u_grad):
                F, _, _, tau, _, _ = return_map(u_grad)
                J = np.linalg.det(F)
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
                J = np.linalg.det(F)
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
        def tensor_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            fn, _, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, T_old, dt)
            return fn(u_grad)

        def update_int_vars_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, fn, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, T_old, dt)
            return fn(u_grad)

        def compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, _, fn, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, T_old, dt)
            return fn(u_grad)

        def compute_lagrangian_strain_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, _, _, fn, _ = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, T_old, dt)
            return fn(u_grad)

        def compute_detla_gamma_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, _, _, _, fn = get_partial_tensor_map(F_old, Be_old, alpha_old,
                                                    shape_grads_center, ugrad0, thetas, T_old, dt)
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

        # The lumped capacity term needs the NODAL T_old of the cell; T_old
        # arrives at the quadrature points as the interpolant of the nodal
        # sol_dT_old (set_params), so the nodal values are recovered exactly
        # with the left inverse of the shape-value matrix (4 x 4 and well
        # conditioned for TET4 with the 4-point rule).
        S = onp.asarray(self.fes[1].shape_vals, dtype=onp.float64)   # (nq, nn)
        P = onp.linalg.pinv(S)
        if not onp.allclose(P @ S, onp.eye(S.shape[1]), atol=1e-10):
            raise ValueError(
                f"the lumped heat capacity needs the nodal T_old to be "
                f"recoverable from its {S.shape[0]} quadrature values on a "
                f"{S.shape[1]}-node element; use a higher gauss_order")
        P_T = np.asarray(P)

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
            a1, a2, a3, a4, a5, a6, T_old, dt = cell_internal_vars
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

            cell_internal_vars_updated = (a1, a2, a3, a4, u_grads_center_reshape, a6, T_old, dt)
            u_grads_reshape = u_grads.reshape(-1, self.fe_u.vec, self.dim)

            u_physics = jax.vmap(stress_return_map)(
                u_grads_reshape, *cell_internal_vars_updated).reshape(u_grads.shape)
            val4 = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))

            # ---- Thermal part ----
            T = np.sum(cell_sol_T[None, :, :] * self.fe_T.shape_vals[:, :, None], axis=1)

            # row-sum lumped heat capacity, divided by the step's own dt
            T_nodes = cell_sol_T[:, 0]
            T_old_nodes = P_T @ T_old.reshape(-1)
            val1 = self.C * np.sum(
                ((T_nodes - T_old_nodes)[None, :] / dt[:, None])
                * self.fe_T.shape_vals
                * cell_JxW_T[:, None], axis=0)[:, None]

            detla_gammas = jax.vmap(detla_gamma_map)(u_grads_reshape, *cell_internal_vars_updated)

            # Hybrid explicit/implicit dissipation
            K = vmap_K_fun(a3, T_old)
            sig0 = jax.vmap(yield_stress_temp)(T_old)
            flow_stresses_old = sig0 + K

            _, _, alphas, _, _, _ = jax.vmap(int_vars_map)(
                u_grads_reshape, *cell_internal_vars_updated)
            # (flow_stresses_new computed but not used — semi-implicit for convergence)

            tmp2 = (-detla_gammas * np.sqrt(2. / 3.) * flow_stresses_old
                    / dt)
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
    # Surface kernels: die-contact heat transfer
    # =================================================================
    def get_universal_kernels_surface(self):
        # Die-contact heat extraction q = h (T0 - T): h = DIE_H is the interface
        # conductance, T0 = DIE_T the platen temperature; the mask (in [0, 1])
        # restricts the flux to the part of the candidate band that is actually
        # pressed against a platen at this step.  (The two kernels are the two
        # platens; the old name "convection" is kept for the kernel functions.)

        def _surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                            face_nanson_scale, old_T_face, contact_mask):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_T = cell_sol_list[1]
            face_shape_vals = face_shape_vals[:, -self.fes[1].num_nodes:]
            face_nanson_scale = face_nanson_scale[0]

            # row-sum lumped: each node is pulled toward the die temperature by
            # its own temperature only; the contact mask stays at the
            # quadrature points (how much of the face is pressed).
            T_nodes = cell_sol_T[:, 0]
            q = contact_mask[:, 0][:, None] * DIE_H * (DIE_T - T_nodes[None, :])
            val_T = -np.sum(q * face_shape_vals
                            * face_nanson_scale[:, None], axis=0)[:, None]
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

        a1, a2, a3, a4, a5, a6, a7, a8 = int_vars
        int_vars_updated = (a1, a2, a3, a4, u_gradsc, a6, a7, a8)
        return vmap_update(u_grads, *int_vars_updated)

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
