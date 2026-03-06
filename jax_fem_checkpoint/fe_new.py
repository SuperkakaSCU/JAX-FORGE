import numpy as onp
import jax
import jax.numpy as np
import sys
import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union
from jax_fem_checkpoint.generate_mesh import Mesh  #basis_new
from jax_fem_checkpoint.basis_new import get_face_shape_vals_and_grads, get_shape_vals_and_grads, get_shape_vals_and_grads_center
from jax_fem_checkpoint import logger


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


@dataclass
class FiniteElement:
    """
    Defines finite element related to one variable (can be vector valued)

    Attributes
    ----------
    mesh : Mesh object
        The mesh object stores points (coordinates) and cells (connectivity).
    vec : int
        The number of vector variable components of the solution.
        E.g., a 3D displacement field has u_x, u_y and u_z components, so vec=3
    dim : int
        The dimension of the problem.
    ele_type : str
        Element type
    dirichlet_bc_info : [location_fns, vecs, value_fns]
        location_fns : List[Callable]
            Callable : a function that inputs a point and returns if the point satisfies the location condition
        vecs: List[int]
            integer value must be in the range of 0 to vec - 1,
            specifying which component of the (vector) variable to apply Dirichlet condition to
        value_fns : List[Callable]
            Callable : a function that inputs a point and returns the Dirichlet value
    periodic_bc_info : [location_fns_A, location_fns_B, mappings, vecs]
        location_fns_A : List[Callable]
            Callable : location function for boundary A
        location_fns_B : List[Callable]
            Callable : location function for boundary B
        mappings : List[Callable]
            Callable: function mapping a point from boundary A to boundary B
        vecs: List[int]
            which component of the (vector) variable to apply periodic condition to
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str
    gauss_order: int
    dirichlet_bc_info: Optional[List[Union[List[Callable], List[int], List[Callable]]]]
    periodic_bc_info: Optional[List[Union[List[Callable], List[Callable], List[Callable], List[int]]]] = None

    def __post_init__(self):
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes * self.vec
        self.init_norm = None

        start = time.time()
        logger.debug(f"Computing shape function values, gradients, etc.")

        self.shape_vals, self.shape_grads_ref, self.quad_weights, self.quad_points_ref \
        = get_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
        = get_face_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.shape_values_c,self.shape_grads_ref_c = get_shape_vals_and_grads_center(self.ele_type, self.gauss_order)
        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.shape_grads, self.JxW = self.get_shape_grads()
        self.shape_grads_center,self.physical_center = self.get_shape_grads_center()
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(self.dirichlet_bc_info)
        self.p_node_inds_list_A, self.p_node_inds_list_B, self.p_vec_inds_list = self.periodic_boundary_conditions()
        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]
        self.num_face_quads = self.face_quad_weights.shape[1]

        end = time.time()
        compute_time = end - start

        logger.debug(f"Done pre-computations, took {compute_time} [s]")
        logger.info(f"Solving a problem with {len(self.cells)} cells, {self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs.")

    def get_shape_grads_center(self):
        """Compute shape function gradient value
        The gradient is w.r.t physical coordinates.
        Returns
        -------
        shape_grads_physical : onp.ndarray
            (num_cells, num_quads, num_nodes, dim)
        JxW : onp.ndarray
            (num_cells, num_quads)
        """
        # assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
        physical_coos = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim)
        physical_center_points = onp.sum(self.shape_values_c[None, :, :, None] * physical_coos[:, None, :, :], axis=2)

        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] *
                                   self.shape_grads_ref_c[None, :, :, None, :], axis=2, keepdims=True)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (self.shape_grads_ref_c[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        return shape_grads_physical,physical_center_points

    def get_shape_grads(self):
        """Compute shape function gradient value
        The gradient is w.r.t physical coordinates.
        See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
        Page 147, Eq. (3.9.3)

        Returns
        -------
        shape_grads_physical : onp.ndarray
            (num_cells, num_quads, num_nodes, dim)
        JxW : onp.ndarray
            (num_cells, num_quads)
        """
        physical_dim = self.points.shape[1]
        if self.dim == physical_dim:
            assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
            if len(self.cells)<30:  # 3M -->if less then 3M million then a single batch is sufficient 3000000
                physical_coos = np.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
                # print(f'Physical coords: \n', physical_coos)
                # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
                jacobian_dx_deta = np.array(np.sum(physical_coos[:, None, :, :, None] *
                                                self.shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True))

                jacobian_det = np.linalg.det(jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
                jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
                # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
                # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
                shape_grads_physical = onp.array((self.shape_grads_ref[None, :, :, None, :]
                                        @ jacobian_deta_dx)[:, :, :, 0, :])
                JxW = onp.array(jacobian_det * self.quad_weights[None, :])
            else:
                num_cuts = 1
                batch_size = len(self.cells) // num_cuts
                shape_grads_physical_all = []
                JxW_all = []
                for i in range(num_cuts):
                    if i < num_cuts - 1:
                        batch = np.arange(i * batch_size, (i + 1) * batch_size)
                    else:
                        batch = np.arange(i * batch_size, len(self.cells))
                    physical_coos = np.take(self.points, self.cells[batch,:], axis=0)  # (num_cells, num_nodes, dim)
                    jacobian_dx_deta = np.array(np.sum(physical_coos[:, None, :, :, None] *
                                                    self.shape_grads_ref[None, :, :, None, :], axis=2,
                                                    keepdims=True))

                    jacobian_det = np.linalg.det(jacobian_dx_deta)[:, :, 0]
                    jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
                    shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :]
                                            @ jacobian_deta_dx)[:, :, :, 0, :]
                    JxW = jacobian_det * self.quad_weights[None, :]
                    shape_grads_physical_all.append(onp.array(shape_grads_physical))
                    JxW_all.append(onp.array(JxW))
                    del batch, physical_coos, jacobian_dx_deta, jacobian_deta_dx, shape_grads_physical, JxW

                    shape_grads_physical = onp.vstack((shape_grads_physical_all))
                    JxW = onp.vstack((JxW_all))
            return shape_grads_physical, JxW
        elif self.dim < physical_dim:
            print("Applying surface formulation for shell elements...")
            physical_coos = np.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, 3)

            # 1. Compute the rectangular Jacobian matrix (3x2 for our case)
            # self.shape_grads_ref is (num_quads, num_nodes, 2)
            # The result is (num_cells, num_quads, 3, 2)
            print(f'physical_coos shape: {physical_coos.shape}, self.shape_grads_ref shape: {self.shape_grads_ref.shape}')
            jacobian_dx_deta = np.einsum('cnp,qnd->cqpd', physical_coos, self.shape_grads_ref)

            # 2. Compute the metric tensor G = J^T @ J
            # (c, q, 2, 3) @ (c, q, 3, 2) -> (c, q, 2, 2)
            # 1. Compute the rectangular Jacobian matrix J (shape: num_cells, num_quads, 3, 2)
            jacobian_dx_deta = np.einsum('cnp,qnd->cqpd', physical_coos, self.shape_grads_ref)

            # 2. Compute the metric tensor G = J^T @ J.
            # This is the corrected section. We use the @ operator for clarity and correctness.
            # First, explicitly transpose J to get J^T (shape: c, q, 2, 3)
            jacobian_T = jacobian_dx_deta.transpose(0, 1, 3, 2)
            # Then, perform the matrix multiplication.
            # (c, q, 2, 3) @ (c, q, 3, 2) -> (c, q, 2, 2)
            metric_tensor_G = jacobian_T @ jacobian_dx_deta
            # 3. Invert the 2x2 metric tensor
            metric_tensor_G_inv = np.linalg.inv(metric_tensor_G)
            # 4. Compute the physical gradients using the correct surface formula
            # grad_phys = J @ G_inv @ grad_ref
            # First, temp = G_inv @ grad_ref
            # (c, q, 2, 2) @ (q, n, 2) -> (c, q, n, 2)
            temp = np.einsum('cqij,qnj->cqni', metric_tensor_G_inv, self.shape_grads_ref)

            # Then, shape_grads_physical = J @ temp
            # (c, q, 3, 2) @ (c, q, n, 2) -> (c, q, n, 3)
            shape_grads_physical = np.einsum('cqpd,cqnd->cqnp', jacobian_dx_deta, temp)

            # 5. Compute the differential area element for the integral
            # JxW = sqrt(det(G)) * quad_weights
            g_det = np.linalg.det(metric_tensor_G)
            JxW = np.sqrt(g_det) * self.quad_weights[None, :]

            return onp.array(shape_grads_physical), onp.array(JxW)
        else:
            raise ValueError(f"Element dimension ({self.dim}) cannot be greater than "
                            f"physical space dimension ({physical_dim}).")

    def update_mesh(self,du):
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
            = get_face_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.shape_grads, self.JxW = self.get_shape_grads()
        # jax.debug.print('shape_grads')
        shape_grads_old, JxW = self.get_shape_grads()
        F_old = internal_vars[0]
        # self.v_grads_JxW = shape_grads_old[:, :, :, None, :] * self.JxW[:, :, None, None, None]*np.linalg.det(F_old)[:, 0]
        self.sha_grads = (shape_grads_old[None, :, :, None, :] @ np.linalg.inv(F_old))[:, :, :, 0, :][0]
        # self.JxW = JxW[:, :, None, None, None]*np.linalg.det(F_old)[:, 0]
        #
        self.v_gradspe_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]

    def get_face_shape_grads(self, boundary_inds, sol=None):
        """Face shape function gradients and JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            (num_selected_faces, 2)

        Returns
        -------
        face_shape_grads_physical : onp.ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale : onp.ndarray
            (num_selected_faces, num_face_quads)

        """
        
        if sol is None:
            sol = onp.zeros(((self.num_total_nodes, self.dim)))
        else:
            sol = onp.array(sol)
        physical_coos = onp.take(self.points + sol, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)

        sol_quad_surface = self.convert_from_dof_to_face_quad(sol, boundary_inds)
        selected_f_normals = self.get_physical_surface_norm(sol_quad_surface, boundary_inds)[:, 0]
        # change for 2D elements
        if self.dim==2:
            selected_f_normals = self.get_physical_surface_norm(sol_quad_surface, boundary_inds)
        if self.init_norm is None:
                self.init_norm = selected_f_normals

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]
        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = onp.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        return face_shape_grads_physical, nanson_scale

    def nodes_order(self):
        ni = onp.take(self.points, self.cells, axis=0)
        lxs = np.reshape(np.lexsort((ni[:, :, 2], ni[:, :, 1], ni[:, :, 0])),(ni.shape[0], 8, 1))
        s1 = np.take_along_axis(ni, lxs, axis=1)
        return lxs

    def get_normal_vectors(self, boundary_inds, sol):
        surface = np.array([[1, 0, 3, 2], [0, 1, 5, 4], [0, 4, 7, 3], [6, 5, 1, 2], [6, 2, 3, 7],
                            [4, 5, 6, 7]], dtype=int)
        # ni = np.take((self.points + sol), self.cells, axis=0)
        surf_inds = surface[boundary_inds[:, 1]]
        surf_node_inds = np.take_along_axis(self.cells[boundary_inds[:, 0]], surf_inds, axis=1)
        surf_nodes = np.take(self.points + sol, surf_node_inds, axis=0)

        def get_norm(P):
            dir = np.cross(P[1] - P[0], P[2] - P[0], axis=0)
            # area = np.linalg.norm(dir)
            norm = dir / np.linalg.norm(dir)
            return norm

        def get_area(P):
            def dist(p1, p2):
                sub = p1 - p2
                return np.sqrt(np.sum(sub ** 2))

            a = dist(P[0], P[1])
            b = dist(P[1], P[2])
            c = dist(P[2], P[3])
            d = dist(P[3], P[0])
            diag = dist(P[0], P[2]) ** 2
            s = (a + b + c + d) / 2
            deg = np.arccos((a ** 2 + b ** 2 - diag) / (2 * a * b)) + np.arccos((c ** 2 + d ** 2 - diag) / (2 * c * d))
            area = np.sqrt((s - a) * (s - b) * (s - c) * (s - d) - a * b * c * d * (np.cos(deg / 2) ** 2))
            return area

        norm_fn = jax.vmap(get_norm)
        norm = norm_fn(surf_nodes)
        area_fn = jax.vmap(get_area)
        self.surf_area = area_fn(surf_nodes)
        if self.init_norm is None:
            self.init_norm = norm
            self.init_area = self.surf_area
        return norm


    def get_physical_quad_points(self, sol=None):
        """Compute physical quadrature points

        Returns
        -------
        physical_quad_points : onp.ndarray
            (num_cells, num_quads, dim)
        """
        if sol is None:
            sol = onp.zeros(((self.num_total_nodes, self.dim)))
        else:
            sol = onp.array(sol)
        physical_coos = onp.take(self.points + sol, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim)
        physical_quad_points = onp.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        return physical_quad_points

    def get_physical_surface_quad_points(self, boundary_inds, sol=None):
        """Compute physical quadrature points on the surface

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            ndarray shape: (num_selected_faces, 2)

        Returns
        -------
        physical_surface_quad_points : ndarray
            (num_selected_faces, num_face_quads, dim)
        """
        if sol is None:
            sol = np.zeros(((self.num_total_nodes, self.dim)))
        else:
            sol = np.array(sol)
        physical_coos = np.take(self.points + sol, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim)
        physical_surface_quad_points = np.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return physical_surface_quad_points
    
    def get_physical_surface_norm(self, quad_sol, boundary_inds):
        '''
        Parameters: boundary_inds which defines the surface
        Return: Normal vector of surface
        '''
        corr = np.array([1, -1, 1, -1, 1, -1])
        physical_surface_quad_points = self.get_physical_surface_quad_points(boundary_inds) + quad_sol
        def get_norm(P, c):
            dir = np.cross(P[1] - P[0], P[2] - P[0], axis=0)
            # area = np.linalg.norm(dir)
            norm = dir / np.linalg.norm(dir)
            norm_array = c * norm
            return np.array([norm_array] * 4)

        norm_fn = jax.vmap(get_norm)
        norm = norm_fn(physical_surface_quad_points, np.take(corr, boundary_inds[:, 1]))
        if self.dim==2:
            phyPts = physical_surface_quad_points
            # Tangent vector between two quadrature points
            dx_dy = phyPts[:, 0, :] - phyPts[:, 1, :] 
            dMag = np.sqrt(dx_dy[:, 0] ** 2 + dx_dy[:, 1] ** 2)
            dyBydx = dx_dy / dMag[:, None] # Unit tangent vectors
             # Rotate 90 degrees to get outward normal (assuming counter-clockwise ordering)
            dyBydxMag = np.column_stack((-dyBydx[:, 1], dyBydx[:, 0]))
            norm = quad_sol
            norm = np.ones_like(quad_sol)*dyBydxMag[:,None,:]
        return norm

    def Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Indices and values for Dirichlet B.C.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to to vec - 1
        vals_List : List[ndarray]
            Dirichlet values to be assigned
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        if dirichlet_bc_info is not None:
            location_fns, vecs, value_fns = dirichlet_bc_info
            assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
            for i in range(len(location_fns)):
                if callable(location_fns[i]):
                    num_args = location_fns[i].__code__.co_argcount
                    if num_args == 1:
                        location_fn = lambda point, ind: location_fns[i](point)
                    elif num_args == 2:
                        location_fn = location_fns[i]
                    else:
                        raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}")
                    node_inds = onp.argwhere(
                        jax.vmap(location_fn)(self.mesh.points, np.arange(self.num_total_nodes))).reshape(-1)
                else:
                    node_inds = location_fns[i]
                
                vec_inds = onp.ones_like(node_inds, dtype=onp.int32) * vecs[i]

                if callable(value_fns[i]) is True:
                    values = jax.vmap(value_fns[i])(self.mesh.points[node_inds].reshape(-1, self.dim)).reshape(-1)
                else:
                    values = value_fns[i]
                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]
        """
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)

    def periodic_boundary_conditions(self):
        """Not working
        """
        p_node_inds_list_A = []
        p_node_inds_list_B = []
        p_vec_inds_list = []
        if self.periodic_bc_info is not None:
            location_fns_A, location_fns_B, mappings, vecs = self.periodic_bc_info
            for i in range(len(location_fns_A)):
                node_inds_A = onp.argwhere(jax.vmap(location_fns_A[i])(self.mesh.points)).reshape(-1)
                node_inds_B = onp.argwhere(jax.vmap(location_fns_B[i])(self.mesh.points)).reshape(-1)
                points_set_A = self.mesh.points[node_inds_A]
                points_set_B = self.mesh.points[node_inds_B]

                EPS = 1e-5
                node_inds_B_ordered = []
                for node_ind in node_inds_A:
                    point_A = self.mesh.points[node_ind]
                    dist = onp.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
                    node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
                    node_inds_B_ordered.append(node_ind_B_ordered)

                node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
                vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * vecs[i]

                p_node_inds_list_A.append(node_inds_A)
                p_node_inds_list_B.append(node_inds_B_ordered)
                p_vec_inds_list.append(vec_inds)
                assert len(node_inds_A) == len(node_inds_B_ordered)

        return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list

    def get_boundary_conditions_inds(self, location_fns):
        """Given location functions, compute which faces satisfy the condition.

        Parameters
        ----------
        location_fns : List[Callable]
            Callable: a location function that inputs a point (ndarray) and returns if the point satisfies the location condition
                      e.g., lambda x: np.isclose(x[0], 0.)
                      If this location function takes 2 arguments, then the first is point and the second is index.
                      e.g., lambda x, ind: np.isclose(x[0], 0.) & np.isin(ind, np.array([1, 3, 10]))

        Returns
        -------
        boundary_inds_list : List[onp.ndarray]
            (num_selected_faces, 2)
            boundary_inds_list[k][i, 0] returns the global cell index of the ith selected face of boundary subset k
            boundary_inds_list[k][i, 1] returns the local face index of the ith selected face of boundary subset k
        """
        # TODO: assume this works for all variables, and return the same result
        cell_points = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1)  # (num_cells, num_faces, num_face_vertices, dim)
        cell_face_inds = onp.take(self.cells, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices)
        boundary_inds_list = []
        if location_fns is not None:
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:
                    location_fn = lambda point, ind: location_fns[i](point)
                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}")

                vmap_location_fn = jax.vmap(location_fn)
                def on_boundary(cell_points, cell_inds):
                    boundary_flag = vmap_location_fn(cell_points, cell_inds)
                    return onp.all(boundary_flag)

                vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
                boundary_flags = vvmap_on_boundary(cell_face_points, cell_face_inds)
                boundary_inds = onp.argwhere(boundary_flags)  # (num_selected_faces, 2)
                boundary_inds_list.append(boundary_inds)

        return boundary_inds_list

    def convert_from_dof_to_quad(self, sol):
        """Obtain quad values from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)

        Returns
        -------
        u : np.DeviceArray
            (num_cells, num_quads, vec)
        """
        # (num_total_nodes, vec) -> (num_cells, num_nodes, vec)
        cells_sol = sol[self.cells]
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        return u

    def convert_from_dof_to_face_quad(self, sol, boundary_inds):
        """Obtain surface solution from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        boundary_inds : int

        Returns
        -------
        u : np.DeviceArray
            (num_selected_faces, num_face_quads, vec)
        """
        cells_old_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        selected_cell_sols = cells_old_sol[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) 
        # -> (num_selected_faces, num_face_quads, vec)
        u = np.sum(selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        return u

    def sol_to_grad(self, sol):
        """Obtain solution gradient from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)

        Returns
        -------
        u_grads : np.DeviceArray
            (num_cells, num_quads, vec, dim)
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)
        return u_grads

    def print_BC_info(self):
        """Print boundary condition information for debugging purposes.

        TODO: Not working
        """
        if hasattr(self, 'neumann_boundary_inds_list'):
            print(f"\n\n### Neumann B.C. is specified")
            for i in range(len(self.neumann_boundary_inds_list)):
                print(f"\nNeumann Boundary part {i + 1} information:")
                print(self.neumann_boundary_inds_list[i])
                print(
                    f"Array.shape = (num_selected_faces, 2) = {self.neumann_boundary_inds_list[i].shape}"
                )
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the global cell index of the ith selected face"
                )
                print(
                    f"    Array[i, 1] returns the local face index of the ith selected face"
                )
        else:
            print(f"\n\n### No Neumann B.C. found.")

        if len(self.node_inds_list) != 0:
            print(f"\n\n### Dirichlet B.C. is specified")
            for i in range(len(self.node_inds_list)):
                print(f"\nDirichlet Boundary part {i + 1} information:")
                bc_array = onp.stack([
                    self.node_inds_list[i], self.vec_inds_list[i],
                    self.vals_list[i]
                ]).T
                print(bc_array)
                print(
                    f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}")
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the node index of the ith selected dof"
                )
                print(
                    f"    Array[i, 1] returns the vec index of the ith selected dof"
                )
                print(
                    f"    Array[i, 2] returns the value assigned to ith selected dof"
                )
        else:
            print(f"\n\n### No Dirichlet B.C. found.")
