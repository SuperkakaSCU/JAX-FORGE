## Version 2: Add function of rotation through quat or angle at x-axis, and update the internal variables accordingly.
## Version 2.1: Make sure the force is apllied at the right direction after rotation.
## Version 2.2: Remove the rotation during bc_update_fn

## Version 3: Consider Multi-hits
## Version 3.1: Start with two hits, and make sure the second hit is applied on the deformed mesh after the first hit.
## Version 3.2: Make sure the second hit is applied on the right location by selecting contact nodes based on the deformed configuration after the first hit.
## Version 3.3: Refresh precomputed surface integrals/kernels after updating the BCs for the second hit, so that the convection Neumann BCs take effect correctly.

## Version 4: Check cooling factor with video

## Version 5.1: Definition of IC of temperature field
## Version 5.2: Add refine mesh function and test with a refined mesh. (work for jax_forge_test) (num_cuts matters)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("[Warning] open3d not available, mesh conversion functions will be disabled")
import json
import shutil


import time
import glob
import numpy as onp
import jax
import jax.numpy as np
from functools import reduce
try:
    import meshio
except ImportError:
    meshio = None
    print("[Warning] meshio not available, mesh conversion utilities disabled")

import gmsh
from pathlib import Path

from jax import config
config.update("jax_enable_x64", True)

_ = jax.devices()

from jax_fem_checkpoint.problem import Problem       ## num_cuts = 50
from jax_fem_checkpoint.solver_latest import solver  # keep your solver
from jax_fem_checkpoint.utils import save_sol
from jax_fem_checkpoint.generate_mesh import get_meshio_cell_type, Mesh, cylinder_mesh_gmsh

from applications.Example.Fine_mesh.mesh_container import MeshContainer

from dataclasses import dataclass
from scipy.spatial.transform import Rotation
# from forge.jax_forge_ind.utils.utils import get_jaxforge_root


# =============================================================================
# Hit configuration dataclass for multi-hit simulations
# =============================================================================
@dataclass
class Hit:
    """
    Parameters for a single compression hit.
    
    Parameters:
    -----------
    x_min_band : float
        Lower bound of contact band along tool axis (relative to H)
    x_max_band : float
        Upper bound of contact band along tool axis (relative to H)
    compression_displacement : float
        Total platen displacement in this hit (mm)
    rotation_euler_x : float
        Rotation angle around X-axis in degrees
    total_time : float
        Duration (seconds) allocated for this hit's time-stepping.
        Each hit may have a different overall time; passed directly to the
        AutomaticTimeStepperTM constructor.
    """
    x_min_band: float
    x_max_band: float
    compression_displacement: float
    rotation_euler_x: float = 0.0
    total_time: float = 1.0



## HU: 15-5PH -- 20~600 are based on ChatGPT, waiting for calibration
Q_temps = np.array([20,    200,  300,   426,   538,  682,  800,  898,  1000,  1100])
Q_reducs = np.array([145., 110.,  131., 219.,  276,  250., 300., 220.,  80.,   80.])/145.0

## HU: 15-5PH -- 20~600 are based on ChatGPT, waiting for calibration
b_temps = np.array([20,   100, 200, 300, 400, 500, 682, 800., 898, 1000, 1100])
b_reducs = np.array([25., 20., 12., 8.,  4.,  4.,  4.8, 4.7,  4.5,  4.3,   4.3])/25.

# -----------------------------------------------------------------------------
class ThermalMechanical(Problem):

    def custom_init(self):
        self.fe_u = self.fes[0]
        self.fe_T = self.fes[1]

        self.S = 1.0 # unity/meters

        # Define parameters
        self.T0 = 20.0  # ambient temperature, C
        self.E = 196.e3 / (self.S**2) # young's modulus, MPa
        self.sig0 = 1172. / (self.S**2)
        self.nu = 0.272
        

        ### Hu: Refernce -- https://www.spacematdb.com/spacemat/manudatasheets/15-5_PH_Data_Sheet.pdf
        rho = 7800e-12 / (self.S**3) # density, 7800 kg/m^3 --> tone/mm^3
        
        self.alpha_expansion = 13.5e-6
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.kappa = self.alpha_expansion * (2 * self.mu + 3 * self.lmbda)


        ### Hu: Reference -- https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=MQM15AF
        # self.C = 420.e6 * rho  # specific heat per unit volume at constant strain, 420J/kg-K --> mJ/tone-K
        # self.k = 17.8 / self.S # thermal conductivity, 17.8 W/m-K -->  mW/mm-K
        # self.alpha_expansion = 11.3e-6  # thermal expansion coefficient, 11.3e-6(m/m)/K --> (mm/mm)/K

        ## Assumption of 900K
        
        self.C = 600.e6 * rho  # specific heat per unit volume at constant strain, 420J/kg-K --> mJ/tone-K
        self.k = 25.0 / self.S # thermal conductivity, 17.8 W/m-K -->  mW/mm-K

        self.dt = 0.1 # secs


        self.Chi = 0.85
        self.Q = 145./ (self.S**2)     # Hardening saturation MPa
        self.b = 25.      # Hardening rate

        self.yield_stress_temps = np.array([20,    200,  300, 400, 500, 682, 800, 898, 1000, 1100])
        self.yield_stress_reducs = np.array([1172, 1048, 965, 869, 634, 330, 240, 100., 70.,   70])/1172.

        self.modulus_temps = np.array([ 20, 100,   200,   300,   400,   500,   682,     800,    898,    1000,    1100])
        self.modulus_reducs = np.array([1., 0.996, 0.978, 0.963, 0.947, 0.930, 90/196,  80/196, 6/196,  5.8/196,  5.8/196])


        # Deformation gradient (identity initially)
        self.F_old = np.repeat(np.repeat(np.eye(self.dim)[None, None, :, :], len(self.fe_u.cells), axis=0),
                               self.fe_u.num_quads, axis=1)

        # Elastic left Cauchy-Green tensor (identity initially)
        self.Be_old = np.array(self.F_old)
        
        # Accumulated plastic strain (scalar, zero initially)
        self.alpha_old = np.zeros((len(self.fe_u.cells), self.fe_u.num_quads))
        
        # Shape function gradients at element center for F-bar method
        # (num_cells, num_quads, num_nodes, dim)
        self.shape_grads_center = self.fe_u.shape_grads_center
        self.ugrad_center = np.zeros_like(self.F_old)
        
        self.fe_u.flex_inds = np.arange(len(self.fe_u.cells))
        
        # Material parameters per cell (11 parameters, currently using only 4)
        full_params = np.ones((self.fe_u.num_cells, 11))
        self.thetas = np.repeat(full_params[:, None, :], self.fe_u.num_quads, axis=1)

        # [-2]: dT_old
        # [-1]: dt

        self.internal_vars = [self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center, 
                             self.ugrad_center, self.thetas, np.zeros((len(self.fe_T.cells), self.fe_T.num_quads)),
                             self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads))]

        self.nodes_to_compare = None
        self.target_displacement = None


    ## [a1, a2, a3, a4, a5, thetas, T_old, dt]
    def set_params(self, params):
        print("Update sol_u_old and sol_dT_old ...")
        int_vars, scale, sol, rho_ini, sol_dT_old = params

        scale = jax.lax.stop_gradient(scale)
        int_vars = jax.lax.stop_gradient(int_vars)
        sol = jax.lax.stop_gradient(sol)

        self.scale = scale
        a1, a2, a3, a4, a5, a6, a7, a8 = int_vars
        # p_mag = self.pressure_mag * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))

        ### Hu: TODO: This unnecessary right now
        # scales = scale * np.ones((len(self.boundary_inds_list[0]), self.fes[0].num_face_quads))
        scales = scale * onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        # self.update_nanson_scale(sol)
        # sol_quad_surface = self.fes[0].convert_from_dof_to_face_quad(sol, self.boundary_inds_list[0])
        # norm_quad_surface = self.fes[0].get_physical_surface_norm(sol_quad_surface, self.boundary_inds_list[0])
        # self.internal_vars_surfaces = [[scales, norm_quad_surface]]

        full_params = np.ones((self.fe_u.num_cells, len(rho_ini)))
        full_params = full_params.at[self.fe_u.flex_inds].set(rho_ini)
        self.thetas = np.repeat(full_params[:, None, :], self.fe_u.num_quads, axis=1)


        self.internal_vars = [a1, a2, a3, a4, a5, self.thetas,
                             self.fe_T.convert_from_dof_to_quad(sol_dT_old),
                             self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads))]

        sol_dT_old_left = self.fes[1].convert_from_dof_to_face_quad(sol_dT_old, self.boundary_inds_list[0])
        sol_dT_old_right = self.fes[1].convert_from_dof_to_face_quad(sol_dT_old, self.boundary_inds_list[1])
        self.internal_vars_surfaces = [[sol_dT_old_left], [sol_dT_old_right]]


    def set_timestep(self, dt):
        self.dt = dt
        self.internal_vars[-1] = self.dt * np.ones((self.fe_T.num_cells, self.fe_T.num_quads))

    


    def get_maps(self):
        """Define constitutive model and post-processing functions"""
        def modulus_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T,self.modulus_temps,self.modulus_reducs)*self.E
        
        def yield_stress_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T,self.yield_stress_temps,self.yield_stress_reducs)*self.sig0

        # def get_partial_tensor_map(F_old, be_old, alpha_old, shape_grads_center, ugrad0, theta):
        def get_partial_tensor_map(F_old, be_old, alpha_old, shape_grads_center, ugrad0, theta, T_old, dt):
            """
            J2 plasticity with nonlinear isotropic hardening
            F-bar method for volumetric locking prevention
            """
            ### Hu: Change it to temperature-dependent
            
            # Material parameters
            # E = 100.e3   # Young's modulus
            # sig0 = 100.0 # Initial yield stress
            # K = E / (3. * (1. - 2. * self.nu))  # Bulk modulus
            # G = E / (2. * (1. + self.nu))      # Shear modulus
            
            # Temperature-dependent Material parameters
            K = modulus_temp(T_old) / (3. * (1. - 2. * self.nu))
            G = modulus_temp(T_old) / (2. * (1. + self.nu))      # Shear modulus
            sig0 = yield_stress_temp(T_old)

            # print("F_old", F_old.shape)  # (3, 3)
            # print("K", K.shape) # ()
            

            def first_PK_stress(u_grad):
                """Compute first Piola-Kirchhoff stress P = τ F^{-T}"""
                F, _, _, tau, _, _ = return_map(u_grad)
                
                ### P -- first PK stress; tau -- Kirchhoff stress
                P = tau @ np.linalg.inv(F).T
                return P

            def update_int_vars(u_grad):
                """Update internal variables after converged increment"""
                F, be_bar, alpha, _, ugrad0_updated, Delta_gamma_final = return_map(u_grad)
                return F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta

            def compute_cauchy_stress(u_grad):
                """Compute Cauchy stress σ = (1/J) τ"""
                F, _, _, tau, _, _ = return_map(u_grad)
                J = np.linalg.det(F)
                sigma = (1. / J) * tau
                return sigma

            def compute_lagrangian_strain(u_grad):
                """Compute Green-Lagrange strain E = 0.5(F^T F - I)"""
                F = u_grad + np.eye(self.dim)
                strain = 0.5 * (F @ F.T - np.eye(self.dim))
                return strain

            def compute_detla_gamma(u_grad):
                _, _, _, _, _, Delta_gamma_final = return_map(u_grad)
                return Delta_gamma_final


            # def get_tau(F, be_bar):
            def get_tau(F, be_bar, T_old):
                """Compute Kirchhoff stress from elastic left Cauchy-Green"""
                J = np.linalg.det(F)
                ## Hu: Eq. (70) & Eq. (71)
                # tau = 0.5 * K * (J**2 - 1.) * np.eye(self.dim) + G * deviatoric(be_bar)

                p = 0.5 * K * (J**2 - 1.)/J
                s = G * deviatoric(be_bar)

                beta = modulus_temp(T_old)/(1.0 - 2.0*self.nu)*self.alpha_expansion

                tau = J*(p - beta*(T_old-self.T0))*np.eye(self.dim) + s


                return tau

            def deviatoric(A):
                """Deviatoric part of tensor A"""
                return A - 1. / self.dim * np.trace(A) * np.eye(self.dim)

            def hard_satu_temp(T):
                T = np.asarray(T).reshape(())
                return np.interp(T, Q_temps, Q_reducs)*self.Q

            def hard_rate_temp(T):
                T = np.asarray(T).reshape(())
                return np.interp(T, b_temps, b_reducs)*self.b

            def K_fun(a, T):
                """Nonlinear isotropic hardening: K(α) = Q(1 - exp(-bα))"""
                Q = hard_satu_temp(T)
                b = hard_rate_temp(T)
                nonlinear = Q * (1. - np.exp(-b * a))
                return nonlinear

            def return_map(u_grad):
                """
                Radial return mapping algorithm for J2 plasticity
                F-bar method for near-incompressibility
                """
                # Modified deformation gradient (F-bar method)
                ### Hu: Eq. (9) -- be: elastic part of left Cauchy-Green tensors
                be_bar_old = (np.linalg.det(F_old)**(-2. / 3.)) * be_old


                Fact = u_grad + np.eye(self.dim)
                F0 = ugrad0 + np.eye(self.dim)

                
                F = ((np.linalg.det(F0) / np.linalg.det(Fact))**(1. / 3.)) * Fact


                ### Hu: Eq. (9.3.5) -- f: relative deformation gradient
                ### Hu: Eq. (9.3.5) -- f_bar: volume-preserving deformation gradient
                F_old_inv = np.linalg.inv(F_old)
                f = F @ F_old_inv   # Incremental deformation gradient
                
                
                # Elastic predictor (9.3.14)
                f_bar = (np.linalg.det(f)**(-1. / 3.)) * f    # Eq. (68)
                be_bar_trial = f_bar @ be_bar_old @ f_bar.T   # Eq. (68)
                s_trial = G * deviatoric(be_bar_trial)        # Eq. (70)/(24)
                # Coefficient for Return-mapping algorithm
                Ie_bar = (1. / 3.) * np.trace(be_bar_trial)
                G_bar = Ie_bar * G
                

                # Check yield condition
                # Eq. (73)
                yield_f_trial = np.linalg.norm(s_trial) - np.sqrt(2. / 3.) * (sig0 + K_fun(alpha_old, T_old))
                
                # Plastic corrector (if yielding)
                Delta_gamma = np.where(yield_f_trial > 0., newton_conv_mod(s_trial, be_bar_trial), 0.)
                direction = np.where(Delta_gamma > 0., s_trial / np.linalg.norm(s_trial), 0.)
                
                # Updated stress and internal variables
                s = s_trial - 2. * G_bar * Delta_gamma * direction
                alpha = alpha_old + np.sqrt(2. / 3.) * Delta_gamma
                be_bar = (s / G) + Ie_bar * np.eye(self.dim)   # Eq. (9.3.33)
                
                # tau = get_tau(F, be_bar)
                tau = get_tau(F, be_bar, T_old)

                be_updated = be_bar * (np.linalg.det(F)**(2. / 3.))
                
                return F, be_updated, alpha, tau, ugrad0, Delta_gamma

            def newton_conv_mod(s_trial, be_bar_trial):
                """
                Newton-Raphson iteration for plastic multiplier Δγ
                Solves: ||s_trial|| - √(2/3)[σ_0 + K(α)] - 2G̅Δγ = 0
                """
                Ie_bar = (1. / 3.) * np.trace(be_bar_trial)
                G_bar = Ie_bar * G

                def implicit_residual(d_gamma):
                    """Consistency condition residual"""
                    alpha_eval = alpha_old + np.sqrt(2. / 3.) * d_gamma
                    stress_dgamma = np.where(alpha_eval > 0., K_fun(alpha_eval, T_old), 0.)
                    res = (np.linalg.norm(s_trial) - np.sqrt(2. / 3.) * sig0
                           - (np.sqrt(2. / 3.) * stress_dgamma + 2. * G_bar * d_gamma))
                    return res

                def body_fun(carry, _):
                    """Newton iteration body"""
                    d_gamma, converged = carry
                    res = implicit_residual(d_gamma)
                    res_grad = jax.grad(implicit_residual)(d_gamma)
                    
                    # Update only if not converged
                    d_gamma_u = jax.lax.cond(converged, lambda d: d, lambda d: d - (res / res_grad), d_gamma)

                    # Check convergence
                    res = implicit_residual(d_gamma_u)
                    converged_updated = np.linalg.norm(res) < tol
                    return (d_gamma_u, converged_updated), None

                tol = 1.e-6
                max_iters = 50
                init_d_gamma = 0.0
                carry_init = (init_d_gamma, False)

                ## Hu: lax.scan(f, init, xs, length=None)
                ## Hu: f(carry, x) → (carry_new, y)
                # Fixed-point iteration using scan
                (d_gamma_final, _), _ = jax.lax.scan(body_fun, carry_init, None, length=max_iters)

                return d_gamma_final

            return (first_PK_stress, update_int_vars, compute_cauchy_stress, compute_lagrangian_strain,
                    compute_detla_gamma)

        
        def tensor_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            first_PK_stress, _, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                 ugrad0, thetas, T_old, dt)
            return first_PK_stress(u_grad)

        def update_int_vars_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, update_int_vars, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                 ugrad0, thetas, T_old, dt)
            return update_int_vars(u_grad)

        def compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, _, compute_cauchy_stress, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                       ugrad0, thetas, T_old, dt)
            return compute_cauchy_stress(u_grad)

        def compute_lagrangian_strain_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, _, _, compute_lagrangian_strain, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                           ugrad0, thetas, T_old, dt)
            return compute_lagrangian_strain(u_grad)

        def compute_detla_gamma_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt):
            _, _, _, _, compute_detla_gamma = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                            ugrad0, thetas, T_old, dt)
            return compute_detla_gamma(u_grad)
            
        return (tensor_map, update_int_vars_map, compute_cauchy_stress_map, compute_lagrangian_strain_map,
                compute_detla_gamma_map)



    def get_stress_return_map(self):
        """Get first Piola-Kirchhoff stress map"""
        stress_return_map, _, _, _, _ = self.get_maps()
        return stress_return_map

    def get_update_int_vars_map(self):
        '''Get updated values for flow stress'''
        _, int_vars_map, _, _, _ = self.get_maps()
        return int_vars_map


    def get_cauchy_stress_map(self):
        """Get Cauchy stress map for future tau"""
        _, _, cauchy_return_map, _, _ = self.get_maps()
        return cauchy_return_map


    def get_detla_gamma_map(self):
        '''Get detal_gammp map for D_mech'''
        _, _, _, _, detla_gamma_map = self.get_maps()
        return detla_gamma_map


    def get_universal_kernel(self):
        stress_return_map = self.get_stress_return_map()
        int_vars_map = self.get_update_int_vars_map()
        cauchy_return_map = self.get_cauchy_stress_map()
        detla_gamma_map = self.get_detla_gamma_map()

        def get_J(u_grad):
            F = u_grad + np.eye(self.dim)
            J = np.linalg.det(F)
            return J

        def get_tau(cauchy_stress, J):
            return cauchy_stress*J
        vmap_get_tau = jax.vmap(get_tau, in_axes=(0, 0))

        def hard_satu_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T, Q_temps, Q_reducs)*self.Q

        def hard_rate_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T, b_temps, b_reducs)*self.b

        def K_fun(a, T):
            """Nonlinear isotropic hardening: K(α) = Q(1 - exp(-bα))"""
            Q = hard_satu_temp(T)
            b = hard_rate_temp(T)
            nonlinear = Q * (1. - np.exp(-b * a))
            return nonlinear
        vmap_K_fun = jax.vmap(K_fun)

        def deviatoric(A):
            """Deviatoric part of tensor A"""
            return A - 1. / self.dim * np.trace(A) * np.eye(self.dim)
        vmap_deviatoric =  jax.vmap(deviatoric)


        def safe_frob_norm(A):
            eps=1e-8
            return np.sqrt(np.sum(A*A) + eps)
        vmap_safe_frob_norm = jax.vmap(safe_frob_norm)


        def compute_lagrangian_strain(u_grad):
            """Compute Green-Lagrange strain E = 0.5(F^T F - I)"""
            F = u_grad + np.eye(self.dim)
            strain = 0.5 * (F @ F.T - np.eye(self.dim))
            return strain
        vmap_compute_lagrangian_strain = jax.vmap(compute_lagrangian_strain)

        def F_to_lagrangian_strain(F):
            return 0.5 * (F @ F.T - np.eye(self.dim))
        vmap_F_to_lagrangian_strain = jax.vmap(F_to_lagrangian_strain)

        def yield_stress_temp(T):
            T = np.asarray(T).reshape(())
            return np.interp(T,self.yield_stress_temps,self.yield_stress_reducs)*self.sig0

        ## [u, T]
        ## [a1, a2, a3, a4, a5, thetas, T_old, dt]
        def universal_kernel(
            cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            # F_old, be_old, alpha_old, shape_grads_center, ugrad0, theta, T_old, dt = cell_internal_vars
            a1, a2, a3, a4, a5, a6, T_old, dt = cell_internal_vars
            shape_grads_center = a4

            # print("a1: {0}, a2: {1}, a3:{2}, a4:{3}, a5:{4}, a6:{5}, T_old:{6}, dt:{7}".format(a1.shape, a2.shape, a3.shape, a4.shape, a5.shape, a6.shape, T_old.shape, dt.shape))

            # Split
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_u, cell_sol_T = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i + 1], :] for i in range(self.num_vars)]
            cell_shape_grads_u, cell_shape_grads_T = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i + 1], :, :] for i in range(self.num_vars)]
            cell_v_grads_JxW_u, cell_v_grads_JxW_T = cell_v_grads_JxW_list
            _, cell_JxW_T = cell_JxW[0], cell_JxW[1]

            ################################################################################
            ##### Mechanical Part
            ################################################################################
            u_grads = np.sum(cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :], axis=1)

            ## Hu: (None, num_nodes, vec, None) * (num_quads, num_nodes, None, dim)
            u_grads_center = cell_sol_u[None, :, :, None] * shape_grads_center[:, :, None, :]
            u_grads_center = np.sum(u_grads_center, axis=1)  ## Hu: (num_quads, 1, vec, dim)
            u_grads_center_reshape = u_grads_center.reshape(-1, self.fe_u.vec, self.dim)  # Hu: (num_quads, vec, dim)

            cell_internal_vars_updated = (a1, a2, a3, a4, u_grads_center_reshape, a6, T_old, dt)


            # Handles the term 'inner(first_PK_stress,grad(v)) * dx'
            # u_physics = jax.vmap(stress_map)(u_grads, sigmas_old, epsilons_old, T, T-T_old)
            u_grads_reshape = u_grads.reshape(-1, self.fe_u.vec, self.dim)


            # tensor_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt)
            u_physics = jax.vmap(stress_return_map)(u_grads_reshape, *cell_internal_vars_updated).reshape(u_grads.shape)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec)
            val4 = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))



            ################################################################################
            ##### Thermal Part
            ################################################################################

            # Handles the term 'C * (T_crt-T_old)/dt * Q * dx'
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec)
            T = np.sum(cell_sol_T[None, :, :] * self.fe_T.shape_vals[:, :, None], axis=1)

            # print("T:{0}, T_old:{1}".format(T.shape, T_old.shape))
            # (num_quads, 1, num_vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) 
            # -> (num_nodes, vec)
            val1 = (self.C/self.dt* np.sum((T - T_old)[:, None, :]
                    * self.fe_T.shape_vals[:, :, None]
                    * cell_JxW_T[:, None, None],
                    axis=0,
                )
            )


            detla_gammas = jax.vmap(detla_gamma_map)(u_grads_reshape, *cell_internal_vars_updated)  # (num_quads, )

            ## Hu: Hybrid explicit/implicit
            K = vmap_K_fun(a3, T_old) # (num_quads)
            sig0 = jax.vmap(yield_stress_temp)(T_old) # (num_quads)
            flow_stresses_old = sig0 + K  # (num_quads)

            ## Hu: Totally implicit
            # F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta
            _, _, alphas, _, _, _ = jax.vmap(int_vars_map)(u_grads_reshape, *cell_internal_vars_updated)  # (num_quads, )
            
            K_new = vmap_K_fun(alphas, T) # (num_quads)
            sig0_new = jax.vmap(yield_stress_temp)(T)
            flow_stresses_new = sig0_new + K_new  # (num_quads)

            tmp2 = -detla_gammas*np.sqrt(2. / 3.)*flow_stresses_old/self.dt # (num_quads,)
            # tmp2 = -detla_gammas*np.sqrt(2. / 3.)*flow_stresses_new/self.dt # (num_quads,)
            val2 = self.Chi * np.sum(tmp2[:, None, None] * self.fe_T.shape_vals[:, :, None] * cell_JxW_T[:, None, None], axis=0)


            # Handles the term 'kappa * T0 * tr((epsilon_crt-epsilon_old)/dt) * Q * dx'
            # Handles the term 'k * inner(grad(T_crt),grad(Q)) * dx'
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) 
            # -> (num_quads, num_nodes, vec, dim)
            # -> (num_quads, vec, dim)
            T_grads = np.sum(
                cell_sol_T[None, :, :, None] * cell_shape_grads_T[:, :, None, :], axis=1
            )
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec)
            val3 = np.sum(self.k * T_grads[:, None, :, :] * cell_v_grads_JxW_T, axis=(0, -1))


            weak_form = [val4, val1 + val2 + val3]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel


    def get_universal_kernels_surface(self):
        # Neumann BC values for thermal problem
        # Heat transfer between press and rod
        ## Hu: location_fns = [left, right]  
        ## Hu: surface_map(u, x, *cell_internal_vars_surface)

        def thermal_neumann(T, old_T_face):
            h = 10.0 # heat convection coefficien 1.0e4 W/m^2-K --> 10.0 mW/mm^2-K
            # h = 0.0
            T0 = 25. # room Temp.
            # q_contact = h*(T0 - old_T_face[0])
            q_contact = h*(T0 - T[0])
            q = q_contact
            return -np.array([q])
            # return np.array([q])


        def convection_neumann_left(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, old_T_left):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_face_quads, dim)
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # face_nanson_scale: (num_vars, num_face_quads)
            ## Hu: old_T_top -- (num_face_quads, vec)
            
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_T = cell_sol_list[1]
            face_shape_vals = face_shape_vals[:, -self.fes[1].num_nodes:]
            face_nanson_scale = face_nanson_scale[0]


            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            T = np.sum(cell_sol_T[None, :, :] * face_shape_vals[:, :, None], axis=1)


            u_physics = jax.vmap(thermal_neumann)(T, old_T_left)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val_T = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)
            val_u = np.zeros((self.fes[0].num_nodes, self.fes[0].vec))

            ## Hu: [u, T]
            ## only for Temp field
            val = [val_u, val_T]

            return jax.flatten_util.ravel_pytree(val)[0]


        def convection_neumann_right(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, old_T_right):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_face_quads, dim)
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # face_nanson_scale: (num_vars, num_face_quads)
            
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_T = cell_sol_list[1]
            face_shape_vals = face_shape_vals[:, -self.fes[1].num_nodes:]
            face_nanson_scale = face_nanson_scale[0]


            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            T = np.sum(cell_sol_T[None, :, :] * face_shape_vals[:, :, None], axis=1)

            u_physics = jax.vmap(thermal_neumann)(T, old_T_right)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val_T = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)
            val_u = np.zeros((self.fes[0].num_nodes, self.fes[0].vec))

            ## Hu: [u, T]
            ## only for Temp field
            val = [val_u, val_T]

            return jax.flatten_util.ravel_pytree(val)[0]

        ## Hu: location_fns = [left, right]
        return [convection_neumann_left, convection_neumann_right]


    def update_int_vars_gp(self, sol, int_vars):
        """Update internal variables at all Gauss points"""
        _, update_int_vars_map, _, _, _ = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        
        # Compute displacement gradient at quadrature points
        u_grads1 = (np.take(sol, self.fe_u.cells, axis=0)[:, None, :, :, None] *
                   self.fe_u.shape_grads[:, :, :, None, :])
        u_grads = np.sum(u_grads1, axis=2)
        
        # Compute displacement gradient at element centers
        shape_grads_center = self.fe_u.shape_grads_center
        u_grads1c = (np.take(sol, self.fe_u.cells, axis=0)[:, None, :, :, None] *
                    shape_grads_center[:, :, :, None, :])
        u_gradsc = np.sum(u_grads1c, axis=2)
        
        a1, a2, a3, a4, a5, a6, a7, a8 = int_vars
        int_vars_updated = (a1, a2, a3, a4, u_gradsc, a6, a7, a8)

        updated_int_vars = vmap_update_int_vars_map(u_grads, *int_vars_updated)
        return updated_int_vars


    def compute_stress(self, sol, int_vars):
        """Compute Cauchy stress at all Gauss points"""
        _, _, compute_cauchy_stress, _, _ = self.get_maps()
        vmap_compute_cauchy_stress = jax.jit(jax.vmap(jax.vmap(compute_cauchy_stress)))
        
        u_grads = (np.take(sol, self.fe_u.cells, axis=0)[:, None, :, :, None] *
                   self.fe_u.shape_grads[:, :, :, None, :])
        u_grads = np.sum(u_grads, axis=2)
        
        sigma = vmap_compute_cauchy_stress(u_grads, *int_vars)
        print("sigma", sigma.shape)

        return sigma

    def compute_von_mises(self, s):
        """Compute von Mises equivalent stress"""
        def von_mises(sigma):
            return np.sqrt(0.5 * ((sigma[0][0] - sigma[1][1])**2 + (sigma[1][1] - sigma[2][2])**2 + 
                                  (sigma[2][2] - sigma[0][0])**2) + 
                          3 * ((sigma[0][1])**2 + (sigma[1][2])**2 + (sigma[2][0])**2))

        von_mises_fn = jax.vmap(von_mises)
        return von_mises_fn(s)



# =============================================================================
# Automatic time stepper for thermo-mechanical coupled solve [u, T]
# =============================================================================
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
        # Use last two solutions and last dt
        sol_n = self.sol_list
        sol_nm1 = self.sol_hist[-1]
        dt_nm1 = self.dt_hist[-1] + 1e-12

        # v = (sol_n - sol_nm1) / dt_nm1
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
          duration of the current hit. For multi‑hit simulations, create a new
          stepper (or update ``total_time``) for each hit.
        - During each while-loop iteration ``self.dt`` may be modified by retries
          or by the increase_factor logic. Immediately after ``dt_this`` is
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
            print("*****Outer loop for updating time: Current Time: {0}, Total Time: {1}, Step count: {2}". format(self.current_time, self.total_time, self.step_count))
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
                    # ensure jax array
                    arr = np.array(self.sol_list[1])
                    # perform immutable update on indices
                    arr = arr.at[self.surface_inds].multiply(coeff)
                    # store back into sol_list and sol_dT
                    self.sol_list[1] = arr
                    self.sol_dT = arr
                except Exception as _e:
                    print("Warning: failed to apply cooling", _e)

            retries = 0
            converged = False

            # Old solution for params
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


                print("***Inner loop for checking convergence: Retries: {0}, Current dt: {1}, Scale: {2}, next_time: {3}".format(retries, dt_this, scale, next_time))
                # Update timestep in problem
                print("***Update dt_this and bc with scale...")
                self.problem.set_timestep(dt_this)

                # Update BCs for this scale
                if bc_update_fn is not None:
                    bc_params = bc_params_fn(self.step_count, scale) if bc_params_fn else (scale,)
                    bc_update_fn(self.problem, self.sol_list[0], *bc_params)  # pass current u (not required though)

                # Update parameters for coupled kernel
                # params: [int_vars, scale, sol_u_old, rho_ini, sol_dT_old]
                print("***Update int_vars_old")
                self.problem.set_params([int_vars_old, scale, sol_u_old, rho_ini, sol_dT_old])

                # Solver options
                use_ls = (retries >= self.line_search_after)
                # solver_options = {
                #     "jax_solver": {},
                #     "initial_guess": initial_guess,
                #     "line_search_flag": use_ls,
                #     'return_full_info': True
                # }

                solver_options = {
                    "AMGX_solver": {},
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
                    # new_sol_list = solver(self.problem, solver_options)  # returns [u, dT] in your code
                    new_sol_list, iteration_counter, has_converged = solver(self.problem, solver_options)  # returns [u, dT] in your code



                    #### Hu: Go into 'expect' if nan happens
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

                    # Update internal variables after convergence (mechanical part)
                    # your code updates a5 (ugrad0) and builds int_vars_copy
                    # F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas, T_old, dt (a1~a8)
                    a1, a2, a3, a4, a5, a6, a7, a8 = self.int_vars

                    # F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta (a1~a6)
                    int_vars_u = self.problem.update_int_vars_gp(self.sol_u, self.int_vars)

                    # F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta
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
                        # Save temperature (T = dT + 20 if you want) and displacement as point info
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
                    initial_guess = [sol_u_old, sol_dT_old]  # safer fallback

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

        Requires matplotlib to be installed for plotting.

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
        
        # Add wall time annotations
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
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Step data plot saved to: {save_path}")
            
            # also write scale values to a text file alongside the plot
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


# =============================================================================
# Cylinder side-compression BCs for thermo-mechanical problem
# =============================================================================
def build_cylinder_press_bcs(mesh, R, H, hit, current_sol_u=None, T_linear_fn=None):
    """
    Build Dirichlet and thermal BCs for a single compression hit.
    
    Returns:
      dirichlet_bc_info_u (initial placeholder)
      dirichlet_bc_info_T
      location_fns_convect (with rotation consideration)
      bc_update_fn(problem, sol_u, step_count, scale)
    
    Parameters:
    -----------
    hit : Hit
        Hit object containing x_min_band, x_max_band, compression_displacement, rotation_euler_x
    current_sol_u : array or None
        Current displacement field (num_nodes, 3). If provided, contact nodes are selected
        based on deformed coordinates (mesh.points + current_sol_u). If None, uses initial
        mesh coordinates.
    T_linear_fn : callable or None
        Function to compute initial temperature profile (point) -> temp_value.
        If None, constant T0 is used.
    """

    pts = onp.array(mesh.points)
    
    # Use deformed coordinates if current_sol_u is provided, otherwise use initial coordinates
    if current_sol_u is not None:
        current_sol_u_np = onp.array(current_sol_u)
        pts_for_selection = pts + current_sol_u_np  # Deformed coordinates for node selection
        print(f"  [BC] Using deformed coordinates (displacement applied)")
    else:
        pts_for_selection = pts  # Initial coordinates
        print(f"  [BC] Using initial coordinates")
    
    # ===== Handle rotation =====
    # Convert Euler angles to quaternion
    rot_obj = Rotation.from_euler("x", hit.rotation_euler_x, degrees=True)
    rotation_quaternion = rot_obj.as_quat()  # [x, y, z, w]
    
    # Create rotation matrix for coordinate transformation
    rot_matrix = rot_obj.as_matrix()  # (3, 3)
    
    def transform_point(p):
        """Apply rotation to a point"""
        p_rotated = rot_matrix @ p
        return p_rotated
    
    def inverse_transform_point(p):
        """Apply inverse rotation to a point"""
        p_rotated = rot_matrix.T @ p
        return p_rotated
    
    y_min_initial = float(pts_for_selection[:, 1].min())
    y_max_initial = float(pts_for_selection[:, 1].max())

    print("y_min (in coordinates used for selection):{0}, y_max:{1}".format(y_min_initial, y_max_initial))
    print("Applied rotation quaternion: {0}".format(rotation_quaternion))
    print("Hit: x_band=[{0}, {1}], disp={2}, rot_x={3}".format(
        hit.x_min_band, hit.x_max_band, hit.compression_displacement, hit.rotation_euler_x))

    # band in x where platens act (from hit object)
    x_min_band = hit.x_min_band * H
    x_max_band = hit.x_max_band * H
    compression_displacement = hit.compression_displacement
    print("x_min_band:{0}, x_max_band:{1}".format(x_min_band, x_max_band))

    # tolerances (tune if selection too sparse)
    tol_r = 0.25 * (2 * R / 5)    # heuristic; scale with mesh
    tol_y = 0.10                  # mm-ish; widen if needed

    R0 = float(R)

    def in_x_band(p):
        """Check if point is in axial band (in original coordinates)"""
        return np.logical_and(p[0] >= x_min_band, p[0] <= x_max_band)

    def on_outer_surface(p):
        """Check if point is on outer cylindrical surface"""
        r = np.sqrt(p[1]**2 + p[2]**2)
        return np.abs(r - R0) <= tol_r
    
    # compute list of node indices lying on the outer surface (for cooling)
    # use plain NumPy to avoid JAX array conversion issues
    radii = onp.sqrt(pts_for_selection[:, 1] ** 2 + pts_for_selection[:, 2] ** 2)
    mask = onp.abs(radii - R0) <= tol_r
    surface_inds = onp.where(mask)[0]
    surface_inds = np.array(surface_inds, dtype=int)
    print(f"[BC] outer surface node count: {len(surface_inds)}")
    
    # ===== Compute target platen lines in ROTATED coordinates =====
    # This is key: we need to find the platen positions in the rotated frame
    # The platens move along the Y-axis in their local frame
    
    # Transform a few reference points to rotated frame to find effective Y bounds
    # use numpy-based mask to avoid JAX array conversion problems
    mask_pts = onp.abs(onp.sqrt(pts_for_selection[:, 1]**2 + pts_for_selection[:, 2]**2) - R0) <= tol_r
    sample_points = pts_for_selection[mask_pts]
    if len(sample_points) > 0:
        # Transform sample points to rotated coordinates
        sample_rotated = onp.array([transform_point(p) for p in sample_points])
        y_rot_min = float(sample_rotated[:, 1].min())
        y_rot_max = float(sample_rotated[:, 1].max())
        print("Rotated Y range: [{0}, {1}]".format(y_rot_min, y_rot_max))
    else:
        y_rot_min = y_min_initial
        y_rot_max = y_max_initial
    
    # final target platen lines (in rotated coordinates)
    y_target_left_rot = y_rot_min + compression_displacement
    y_target_right_rot = y_rot_max - compression_displacement

    # constant node sets: all circumferential nodes within reach
    def left_reach_nodes(p):
        """Nodes reachable by left platen (after rotation)"""
        # Transform point to rotated frame
        p_rot = transform_point(p)
        return reduce(np.logical_and, (
            on_outer_surface(p),
            in_x_band(p),
            p_rot[1] <= y_target_left_rot + tol_y  # Check Y in rotated frame
        ))

    def right_reach_nodes(p):
        """Nodes reachable by right platen (after rotation)"""
        # Transform point to rotated frame
        p_rot = transform_point(p)
        return reduce(np.logical_and, (
            on_outer_surface(p),
            in_x_band(p),
            p_rot[1] >= y_target_right_rot - tol_y  # Check Y in rotated frame
        ))

        
    # minimal rigid constraints
    def bottom_face(p):
        print("bottom surface:", float(pts[:, 0].min()))
        return np.isclose(p[0], float(pts[:, 0].min()), atol=1e-5)

    
    # pin one node in x,y to avoid drift
    pin_id = onp.argmin(
        (pts[:, 0] - float(pts[:, 0].min())) ** 2 +
        (pts[:, 1] - y_min_initial) ** 2 +
        (pts[:, 2] - 0.0) ** 2
    )
    pin_point = pts[pin_id]

    print("pin_id", pin_id)

    def pin_node(p):
        return (np.abs(p[0] - pin_point[0]) < 1e-12) & \
               (np.abs(p[1] - pin_point[1]) < 1e-12) & \
               (np.abs(p[2] - pin_point[2]) < 1e-12)

    def zero_val(p):
        return 0.0

    # ===== Displacement computation (in rotated frame) =====
    def left_disp_vec(p):
        """Displacement vector (orig frame) to reach left platen (rotated frame)"""
        p_rot = transform_point(p)
        disp_rot_y = np.maximum(0.0, y_target_left_rot - p_rot[1])
        disp_vec_rot = np.array([0.0, disp_rot_y, 0.0])
        disp_vec_orig = inverse_transform_point(disp_vec_rot)
        return disp_vec_orig  # array([dx, dy, dz])

    def right_disp_vec(p):
        """Displacement vector (orig frame) to reach right platen (rotated frame)"""
        p_rot = transform_point(p)
        disp_rot_y = np.minimum(0.0, y_target_right_rot - p_rot[1])
        disp_vec_rot = np.array([0.0, disp_rot_y, 0.0])
        disp_vec_orig = inverse_transform_point(disp_vec_rot)
        return disp_vec_orig

    # Initial placeholder BC info (values overwritten each step)
    # We constrain each component (0,1,2) for left and right contact regions,
    # so repeat the location function for each component and set vecs accordingly.
    location_fns_u = [bottom_face, pin_node, pin_node,
                      left_reach_nodes, left_reach_nodes, left_reach_nodes,
                      right_reach_nodes, right_reach_nodes, right_reach_nodes]
    vecs =             [0,          1,        2,
                        0,               1,               2,
                        0,               1,               2]
    # initial values zero; bc_update_fn will replace these with proper scaled components
    value_fns = [zero_val] * len(location_fns_u)
    dirichlet_bc_info_u = [location_fns_u, vecs, value_fns]

    # Debug counts (optional)
    left_n = int(onp.sum(onp.array([bool(left_reach_nodes(p)) for p in pts_for_selection])))
    right_n = int(onp.sum(onp.array([bool(right_reach_nodes(p)) for p in pts_for_selection])))
    print(f"[BC] left_reach_nodes={left_n}, right_reach_nodes={right_n}, pin=1")
    if left_n == 0 or right_n == 0:
        print("[BC WARNING] selected 0 nodes on one side. Increase tol_r / tol_x or refine mesh.")

    def bc_update_fn(problem, sol_u, step_count, scale):
        """
        Update Dirichlet BCs for mechanical problem.

        Single-hit assumption: rotation is static (computed once at BC creation),
        so we use the precomputed `left_disp_vec` and `right_disp_vec`.
        """
        # Build per-component value functions (matches expanded vecs/location_fns)
        new_value_fns = [
            zero_val,
            zero_val,
            zero_val,
            (lambda p, idx=0: scale * left_disp_vec(p)[idx]),
            (lambda p, idx=1: scale * left_disp_vec(p)[idx]),
            (lambda p, idx=2: scale * left_disp_vec(p)[idx]),
            (lambda p, idx=0: scale * right_disp_vec(p)[idx]),
            (lambda p, idx=1: scale * right_disp_vec(p)[idx]),
            (lambda p, idx=2: scale * right_disp_vec(p)[idx]),
        ]
        problem.fes[0].update_Dirichlet_boundary_conditions([location_fns_u, vecs, new_value_fns])

    # Thermal BCs (no Dirichlet on temperature; convection via Neumann)
    dirichlet_bc_info_T = None

    # Neumann B.C. [left, right]
    location_fns_convect = [left_reach_nodes, right_reach_nodes]

    return dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds


def refresh_problem_surface_integrals(problem):
    """Refresh surface integral data on an existing Problem instance after
    `problem.location_fns` has been changed.

    This recomputes `boundary_inds_list`, `selected_face_shape_grads`,
    `nanson_scale`, `selected_face_shape_vals`, `physical_surface_quad_points`,
    `I`, `J` indices, and then calls `pre_jit_fns()` to rebuild the JITted kernels.
    """
    # Recompute boundary indices from current location_fns
    problem.boundary_inds_list = problem.fes[0].get_boundary_conditions_inds(problem.location_fns)

    # Rebuild I and J sparse matrix indices (matching __post_init__ logic)
    # Start with volume integral indices (from all cells)
    def find_ind(*x):
        inds = []
        for i in range(len(x)):
            crt_ind = problem.fes[i].vec * x[i][:, None] + np.arange(problem.fes[i].vec)[None, :] + problem.offset[i]
            inds.append(crt_ind.reshape(-1))
        return np.hstack(inds)

    cells_list = [fe.cells for fe in problem.fes]
    inds = onp.array(jax.vmap(find_ind)(*cells_list))
    I = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
    J = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)

    # Add surface integral indices for each boundary
    problem.cells_list_face_list = []
    for boundary_inds in problem.boundary_inds_list:
        cells_list_face = [cells[boundary_inds[:, 0]] for cells in cells_list]
        inds_face = onp.array(jax.vmap(find_ind)(*cells_list_face))
        I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
        J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
        I = onp.hstack((I, I_face))
        J = onp.hstack((J, J_face))
        problem.cells_list_face_list.append(cells_list_face)

    problem.I = I
    problem.J = J

    # Rebuild surface integral geometry
    selected_face_shape_grads = []
    nanson_scale = []
    selected_face_shape_vals = []
    physical_surface_quad_points = []

    for boundary_inds in problem.boundary_inds_list:
        s_shape_grads = []
        n_scale = []
        s_shape_vals = []
        for fe in problem.fes:
            face_shape_grads_physical, nanson_scale_i = fe.get_face_shape_grads(boundary_inds)
            selected_face_shape_vals_i = fe.face_shape_vals[boundary_inds[:, 1]]
            s_shape_grads.append(face_shape_grads_physical)
            n_scale.append(nanson_scale_i)
            s_shape_vals.append(selected_face_shape_vals_i)

        # Concatenate per-variable contributions (match __post_init__ layout)
        s_shape_grads = onp.concatenate(s_shape_grads, axis=2)
        n_scale = onp.transpose(onp.stack(n_scale), axes=(1, 0, 2))
        s_shape_vals = onp.concatenate(s_shape_vals, axis=2)
        physical_surface_quad_points_i = problem.fes[0].get_physical_surface_quad_points(boundary_inds)

        selected_face_shape_grads.append(s_shape_grads)
        nanson_scale.append(n_scale)
        selected_face_shape_vals.append(s_shape_vals)
        physical_surface_quad_points.append(physical_surface_quad_points_i)

    problem.selected_face_shape_grads = selected_face_shape_grads
    problem.nanson_scale = nanson_scale
    problem.selected_face_shape_vals = selected_face_shape_vals
    problem.physical_surface_quad_points = physical_surface_quad_points

    # Reset per-surface internal vars placeholder (will be set in set_params)
    problem.internal_vars_surfaces = [() for _ in range(len(problem.boundary_inds_list))]

    # Re-jit kernels so surface integrals use updated geometry
    problem.pre_jit_fns()


# =============================================================================
# Main thermo-mechanical driver for two-hit forging simulation
# =============================================================================
def run_thermo_mech_cylinder_press_two_hits():
    """
    Run a two-hit forging simulation:
    - Hit 1 at position x_band=[0.2, 0.4]*H with rotation
    - Cool to initial temperature profile
    - Hit 2 at position x_band=[0.5, 0.7]*H with different rotation
    
    Key: F and alpha (plasticity history) carried forward; T reset; displacement resets each hit.
    """
    # Directories (match your pattern)
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, "data")
    msh_dir = os.path.join(data_dir, "msh/jax_forge")
    vtk_dir = os.path.join(data_dir, "vtk_tm_press_T_convect_tet_two_hits")
    assets_dir = os.path.join(msh_dir, "assets")
    stock_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.obj")
    json_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.json")

    os.makedirs(vtk_dir, exist_ok=True)

    # clean VTK directory (recursively remove all subdirectories)
    for item in glob.glob(os.path.join(vtk_dir, "*")):
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)  # Remove directories recursively
            else:
                os.remove(item)      # Remove files
        except (OSError, Exception) as e:
            print(f"Warning: Could not remove {item}: {e}")

    def open3d_to_json(o3d_mesh, json_path):
        """
        Jax-Forge expects meshes in a particular json format. Coverts ForgeDataset mesh for jax.
        """
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        data = {
            "Vertices": vertices.flatten().tolist(),
            "Triangles": faces.flatten().tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    
    def convert_meshio_to_jaxfem(meshio_mesh, ele_type="TET4"):
        print("Converting to jax-fem mesh...")
        cell_type = get_meshio_cell_type(ele_type)
        return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Load initial mesh
    mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
    open3d_to_json(mesh_0, json_mesh_path)
    stock_mesh = MeshContainer.from_json(json_mesh_path)

    #### Hu: Refine mesh before applying actions
    print("Refining mesh with gmsh...")
    surf_json = stock_mesh.serialize_obj()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    stock_mesh_refine = MeshContainer(
        surf_json,
        is_json=True,
        is_surface=True,
        standalone=True,
        tmp_filepath=msh_dir,
    )

    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 4)

    # stats = stock_mesh_refine.get_volumetric(
    #     mesh_density=2.0,
    #     refinement_factor=1.2,
    #     force_delaunay=False,
    # )

    stats = stock_mesh_refine.get_volumetric(
        mesh_density=2.5,
        refinement_factor=10.0,
        force_delaunay=False,
    )

    gmsh.finalize()

    print("refined stats:", stats)

    ele_type = "TET4"
    mesh = convert_meshio_to_jaxfem(stock_mesh_refine.vtk, ele_type=ele_type)

    ## Mesh dimensions
    ## (-5.0~96)
    H = np.max(mesh.points[:, 0])
    R_y = np.max(mesh.points[:, 1])
    R_z = np.max(mesh.points[:, 2])
    R = R_y

    print("H={0}, R_y={1}, R_z={2}".format(H, R_y, R_z))

    # === Initial temperature profile (this is reused after each hit to "cool" the material) ===
    # T_top = 1127.1  # °C
    # T_bot = 666.6   # °C

    # Initial temperature profile:
    # linear along x from bottom (T_bot) to x=72 (T_top),
    # and constant T_top for x > 72.
    T_top = 1096.0  # °C
    T_bot = 676.0   # °C
    x_min = np.min(mesh.points[:, 0])
    x_top = 72.0

    def T_linear_x(point):
        """Piecewise temperature profile along x."""
        x = point[0]
        if x > x_top:
            return T_top
        xc = np.clip(x, x_min, x_top)
        return T_bot + (T_top - T_bot) * ((xc - x_min) / (x_top - x_min))

    # === Define the two hits ===
    # 7.4 sec ~ 8.2 sec from experiment
    hit_1 = Hit(
        x_min_band=0.5,  # 20% of H
        x_max_band=0.7,  # 40% of H (band width = 0.2*H = 10mm for typical H=100mm forging)
        compression_displacement=1.63,
        rotation_euler_x=0.0,
        total_time=0.8,   # duration for hit 1 (seconds)
    )
    
    hit_2 = Hit(
        x_min_band=0.65,  # 50% of H
        x_max_band=0.85,  # 70% of H (overlaps with first hit region)
        compression_displacement=1.63,
        rotation_euler_x=-15.0,  # Opposite rotation
        total_time=0.8,   # shorter hit
    )

    hit_3 = Hit(
        x_min_band=0.55,  # 50% of H
        x_max_band=0.75,  # 70% of H (overlaps with first hit region)
        compression_displacement=1.83,
        rotation_euler_x=30.0,  # Opposite rotation
        total_time=1.2,   # longer final hit
    )

    # hits = [hit_1, hit_2, hit_3]
    hits = [hit_1]

    # === Build initial problem (single mesh, shared across all hits) ===
    pts = onp.array(mesh.points)
    sol_dT_initial = onp.zeros((len(pts), 1), dtype=onp.float64)
    for i in range(len(pts)):
        sol_dT_initial[i, 0] = float(T_linear_x(np.array(pts[i])))

    # Initial conditions
    sol_dT0 = np.array(sol_dT_initial)
    sol_u0 = np.zeros((len(mesh.points), 3))  # 3D displacement
    
    # Build problem once, reuse for all hits
    print("\n" + "="*80)
    print("Building coupled thermo-mechanical problem...")
    print("="*80)
    
    # Placeholder location functions so Problem.__post_init__ can query boundaries.
    # These will be replaced per-hit before running a stepper.
    def _no_location(p):
        return False

    problem = ThermalMechanical(
        mesh=[mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        gauss_order=[2, 2],
        dirichlet_bc_info=[None, None],  # Will be updated per hit
        location_fns=[_no_location, _no_location]  # placeholders, updated per hit
    )

    # === Multi-hit loop ===
    current_sol_u = sol_u0
    current_sol_dT = sol_dT0
    current_int_vars = problem.internal_vars  # Will be updated after each hit

    for hit_idx, hit in enumerate(hits):
        print("\n" + "="*80)
        print(f"HIT {hit_idx + 1}/{len(hits)}")
        print("="*80)
        print(f"Hit params: x_band=[{hit.x_min_band}, {hit.x_max_band}]*H, "
              f"disp={hit.compression_displacement}, rot_x={hit.rotation_euler_x}")

        # Build BCs for this specific hit
        # For hit_idx > 0, pass current_sol_u so contact nodes are selected from deformed mesh
        current_sol_u_for_bc = current_sol_u if hit_idx > 0 else None
        dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
            build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit, current_sol_u=current_sol_u_for_bc, T_linear_fn=T_linear_x)

        # Update problem with new BCs
        problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
        problem.location_fns = location_fns_convect

        # Refresh precomputed surface integrals/kernels so convection Neumann BCs take effect
        refresh_problem_surface_integrals(problem)

        # === Reset temperature to initial profile (simulating cooling between hits) ===
        if hit_idx > 0:
            print("\nResetting temperature to initial profile (cool-down between hits)...")
            current_sol_dT = sol_dT_initial  # Reset to initial gradient
            # Keep F, alpha, and other plasticity variables from previous hit!
            # current_int_vars remains unchanged

        # === Create stepper for this hit ===
        # cooling behaviour: specify cool_factor (adjust as needed)
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

        # Initialize stepper with current state
        stepper.sol_list = [current_sol_u, current_sol_dT]
        stepper.sol_u, stepper.sol_dT = [current_sol_u, current_sol_dT]
        stepper.int_vars = current_int_vars
        
        # === Run this hit ===
        print(f"\nStarting hit {hit_idx + 1} solver...")
        rho_ini = np.array([1.0, 1.0, 1.0, 1.0])
        ok = stepper.run(
            bc_update_fn=bc_update_fn,
            bc_params_fn=lambda step, scale: (step, scale),
            rho_ini=rho_ini,
            vtk_dir=os.path.join(vtk_dir, f"hit_{hit_idx+1}"),
            save_every=1,
        )

        if not ok:
            print(f"\n[ERROR] Hit {hit_idx + 1} solver failed to converge!")
            print(f"Report: {stepper.step_report[-5:]}")
            return False

        # === Extract state at end of this hit ===
        current_sol_u = stepper.sol_u
        current_sol_dT = stepper.sol_dT
        current_int_vars = stepper.int_vars  # Carry plasticity forward!

        print(f"\nHit {hit_idx + 1} completed successfully")
        print(f"  Converged in {stepper.step_count} steps")
        print(f"  Wall time: {stepper.total_wall:.2f}s")
        print(f"  Internal vars shape: F={current_int_vars[0].shape}, alpha={current_int_vars[2].shape}")

        # Generate step data plot for this hit
        plot_path = os.path.join(data_dir, f"hit_{hit_idx+1}_step_data.png")
        stepper.plot_step_data(save_path=plot_path)

    # === Summary ===
    print("\n" + "="*80)
    print("ALL HITS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total VTK output dir: {vtk_dir}")
    print(f"Final displacement norm: {np.linalg.norm(current_sol_u):.6e}")
    print(f"Final temp min/max: {np.min(current_sol_dT):.2f} / {np.max(current_sol_dT):.2f} °C")

    return True



if __name__ == "__main__":
    # Two-hit simulation (recommended):
    run_thermo_mech_cylinder_press_two_hits()
