"""Device-resident residual and Jacobian with a fixed sparsity pattern.

The coupled [u, T] problem of this package is assembled on the GPU:

* The CSR pattern of the Jacobian is built once per mesh from the node graph
  (every pair of nodes sharing a cell, times the 4 x 4 dof block of u_x, u_y,
  u_z, T).  The mesh does not change during a run, and every surface (die and
  air) term couples dofs of one cell only, so the pattern never changes.
* ``cell_slots[c, i, j]`` is the position in the CSR value array of the cell
  matrix entry (i, j) of cell c.  A Jacobian is then one scatter-add of the
  cell (and face) matrices into the value array, with no sort, no host copy
  and no format conversion.
* Dirichlet rows are applied as in ``jax_fem_checkpoint.solver_latest``
  (PETSc ``zeroRows``: the row is zeroed, its diagonal set to 1) and the
  residual as in ``apply_bc_vec`` (``r_i = u_i - u_bc_i``), through a per-dof
  mask, so the active contact set can change from step to step without new
  array shapes and without recompilation.

The element kernels are the problem's own (``problem.kernel``,
``problem.kernel_jac``, ``problem.kernel_face[i]``, ``problem.kernel_jac_face[i]``),
evaluated on all cells in one call (or in batches of ``chunk`` cells, each
batch added into the residual / value array as soon as it is computed), so the
weak form is exactly the one of the original solver.
"""
import time

import numpy as onp
import jax
import jax.numpy as np


def _np(x):
    return onp.asarray(x)


class FastSystem:
    """Pattern, scatter maps and jitted residual / Jacobian for one problem."""

    def __init__(self, problem, chunk=None, verbose=True):
        t0 = time.time()
        self.problem = problem
        self.chunk = chunk
        assert not hasattr(problem, "P_mat"), "linear multipoint constraints are not supported"
        fe_u, fe_T = problem.fes
        assert fe_u.vec == 3 and fe_T.vec == 1
        cells = _np(fe_u.cells).astype(onp.int64)
        assert onp.array_equal(cells, _np(fe_T.cells)), "u and T must share the mesh"
        N = int(fe_u.num_total_nodes)
        nc, nn = cells.shape
        self.N, self.nc, self.nn = N, nc, nn
        self.ndofs = int(problem.num_total_dofs_all_vars)
        assert self.ndofs == 4 * N
        off_T = int(problem.offset[1])
        assert off_T == 3 * N

        # local dof l of a cell: l < 3*nn -> node l//3, comp l%3 (u); else node l-3*nn (T)
        ldof = 4 * nn
        l_node = onp.concatenate([onp.repeat(onp.arange(nn), 3), onp.arange(nn)])
        l_comp = onp.concatenate([onp.tile(onp.arange(3), nn), onp.full(nn, 3)])   # 3 = T
        self.ldof = ldof

        # global dof of each local dof (same order as jax_fem's find_ind / ravel_pytree)
        gnode = cells[:, l_node]                                     # (nc, ldof)
        cell_dofs = onp.where(l_comp[None, :] < 3, 3 * gnode + l_comp[None, :], off_T + gnode)

        # node graph: unique (a, b) pairs of nodes sharing a cell, sorted by a then b
        a = onp.repeat(cells, nn, axis=1).reshape(-1)
        b = onp.tile(cells, (1, nn)).reshape(-1)
        keys = onp.unique(a * N + b)
        na, nb = keys // N, keys % N
        deg = onp.bincount(na, minlength=N)
        node_ptr = onp.concatenate([[0], onp.cumsum(deg)])

        # rows: u rows 3a+c (c = 0..2) then T rows 3N+a; every row of node a has
        # 4*deg(a) entries: u columns 3b+c for the neighbours b, then T columns 3N+b
        row_node = onp.concatenate([onp.repeat(onp.arange(N), 3), onp.arange(N)])
        row_len = 4 * deg[row_node]
        indptr = onp.concatenate([[0], onp.cumsum(row_len)]).astype(onp.int64)
        nnz = int(indptr[-1])
        self.nnz = nnz

        # column indices of every row, built per neighbour pair
        # entry (row of node a, comp ci) x (neighbour k of a, comp cj)
        k_in_a = onp.arange(len(keys)) - node_ptr[na]                # position of b among a's neighbours
        indices = onp.empty(nnz, dtype=onp.int64)
        row_of = onp.empty(nnz, dtype=onp.int64)
        for ci in range(4):
            row = (3 * na + ci) if ci < 3 else (off_T + na)
            base = indptr[row]
            d = deg[na]
            for cj in range(4):
                pos = (3 * k_in_a + cj) if cj < 3 else (3 * d + k_in_a)
                col = (3 * nb + cj) if cj < 3 else (off_T + nb)
                indices[base + pos] = col
                row_of[base + pos] = row
        assert onp.all(onp.diff(indices.reshape(-1)) [onp.diff(row_of) == 0] > 0), "columns not sorted"

        # slot of every cell matrix entry (in blocks of cells to bound the memory)
        idx_t = onp.int32 if nnz < 2**31 - 1 else onp.int64
        slots = onp.empty((nc, ldof, ldof), dtype=idx_t)
        ci = l_comp[None, :, None]
        cj = l_comp[None, None, :]
        blk = 200_000
        for s0 in range(0, nc, blk):
            g = gnode[s0:s0 + blk]
            ga = g[:, :, None]                                       # row node
            gb = g[:, None, :]                                       # column node
            kk = onp.searchsorted(keys, (ga * N + gb).reshape(-1)).reshape(len(g), ldof, ldof) - node_ptr[ga]
            row = onp.where(ci < 3, 3 * ga + ci, off_T + ga)
            pos = onp.where(cj < 3, 3 * kk + cj, 3 * deg[ga] + kk)
            sl = indptr[row] + pos                                   # (blk, ldof, ldof)
            cols = onp.where(cj < 3, 3 * gb + cj, off_T + gb)
            assert onp.array_equal(indices[sl], onp.broadcast_to(cols, sl.shape)), "slot map broken"
            assert onp.array_equal(row_of[sl], onp.broadcast_to(row, sl.shape)), "slot map broken"
            slots[s0:s0 + blk] = sl

        self.indptr_host = indptr
        self.indices_host = indices
        self.row_of_host = row_of
        self.cell_dofs_host = cell_dofs
        self.indptr = np.asarray(indptr.astype(idx_t))
        self.indices = np.asarray(indices.astype(idx_t))
        self.row_of = np.asarray(row_of.astype(idx_t))
        self.is_diag = np.asarray(indices == row_of)
        self.diag_pos = np.asarray(onp.nonzero(indices == row_of)[0].astype(idx_t))
        assert self.diag_pos.shape[0] == self.ndofs
        self.cell_dofs = np.asarray(cell_dofs.astype(onp.int32))
        self.cell_slots = np.asarray(slots)
        del slots
        self._fn_key = None
        self.setup_time = time.time() - t0
        if verbose:
            print(f"[fast] pattern: {self.ndofs} dofs, {nc} cells, nnz {nnz} "
                  f"({nnz / self.ndofs:.1f} per row), built in {self.setup_time:.2f} s", flush=True)

    # ------------------------------------------------------------------
    def _surface_data(self):
        p = self.problem
        faces = []
        for i, binds in enumerate(p.boundary_inds_list):
            binds = _np(binds)
            fc = np.asarray(binds[:, 0].astype(onp.int32)) if len(binds) else np.zeros((0,), np.int32)
            faces.append((fc, (p.physical_surface_quad_points[i], p.selected_face_shape_vals[i],
                               p.selected_face_shape_grads[i], p.nanson_scale[i])))
        return faces

    def _build_fns(self):
        """(Re)build the jitted functions when the problem's kernels changed
        (``pre_jit_fns`` runs again whenever the surface sets are refreshed)."""
        p = self.problem
        key = (id(p.kernel), id(p.kernel_jac), tuple(id(k) for k in p.kernel_face),
               tuple(id(k) for k in p.kernel_jac_face), tuple(len(_np(b)) for b in p.boundary_inds_list))
        if key == self._fn_key:
            return
        self._fn_key = key
        self._n_builds = getattr(self, "_n_builds", 0) + 1
        kernel, kernel_jac = p.kernel, p.kernel_jac
        kfaces, kjfaces = list(p.kernel_face), list(p.kernel_jac_face)
        nsurf = len(kfaces)
        ndofs, nnz = self.ndofs, self.nnz
        cell_dofs, cell_slots = self.cell_dofs, self.cell_slots
        row_of, is_diag = self.row_of, self.is_diag
        chunk, nc = self.chunk, self.nc

        def _keep(mask, x):
            return np.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0.0)

        def cells_add(dofs, cell_args, iv, with_jac):
            """cell kernels scattered into (residual, values); with ``chunk``
            in batches of ``chunk`` cells inside a loop, each batch added as
            soon as it is computed, so no full-mesh array of cell vectors or
            matrices is formed (the last batch overlaps the previous one and
            adds only its new cells)"""
            res = np.zeros(ndofs, dtype=dofs.dtype)
            data = np.zeros(nnz, dtype=dofs.dtype) if with_jac else None
            if chunk is None or chunk >= nc:
                cs = dofs[cell_dofs]
                if with_jac:
                    val, jac = kernel_jac(cs, *cell_args, *iv)
                    data = data.at[cell_slots.reshape(-1)].add(jac.reshape(-1))
                else:
                    val = kernel(cs, *cell_args, *iv)
                return res.at[cell_dofs.reshape(-1)].add(val.reshape(-1)), data
            xs = (cell_dofs, *cell_args, *iv)

            def body(i, carry):
                res, data = carry
                s = np.minimum(i * chunk, nc - chunk)
                cur = [jax.lax.dynamic_slice_in_dim(x, s, chunk, axis=0) for x in xs]
                cd, args = cur[0], cur[1:]
                new = (s + np.arange(chunk)) >= i * chunk
                cs = dofs[cd]
                if with_jac:
                    val, jac = kernel_jac(cs, *args)
                    sl = jax.lax.dynamic_slice_in_dim(cell_slots, s, chunk, axis=0)
                    data = data.at[sl.reshape(-1)].add(_keep(new, jac).reshape(-1))
                else:
                    val = kernel(cs, *args)
                res = res.at[cd.reshape(-1)].add(_keep(new, val).reshape(-1))
                return res, data
            return jax.lax.fori_loop(0, -(-nc // chunk), body, (res, data))

        def assemble_res(dofs, cell_args, iv, faces, ivs, f_ext, with_jac):
            res, data = cells_add(dofs, cell_args, iv, with_jac)
            for i in range(nsurf):
                fc, fargs = faces[i]
                if fc.shape[0] == 0:
                    continue
                fcs = dofs[cell_dofs[fc]]
                if with_jac:
                    fv, fj = kjfaces[i](fcs, *fargs, *ivs[i])
                    data = data.at[cell_slots[fc].reshape(-1)].add(fj.reshape(-1))
                else:
                    fv = kfaces[i](fcs, *fargs, *ivs[i])
                res = res.at[cell_dofs[fc].reshape(-1)].add(fv.reshape(-1))
            res = res - f_ext
            return (res, data) if with_jac else res

        @jax.jit
        def res_bc(dofs, cell_args, iv, faces, ivs, f_ext, bc_mask, bc_target):
            res = assemble_res(dofs, cell_args, iv, faces, ivs, f_ext, False)
            return np.where(bc_mask, dofs - bc_target, res)

        @jax.jit
        def res_nobc(dofs, cell_args, iv, faces, ivs, f_ext):
            return assemble_res(dofs, cell_args, iv, faces, ivs, f_ext, False)

        @jax.jit
        def res_jac_bc(dofs, cell_args, iv, faces, ivs, f_ext, bc_mask, bc_target):
            res, data = assemble_res(dofs, cell_args, iv, faces, ivs, f_ext, True)
            res = np.where(bc_mask, dofs - bc_target, res)
            data = np.where(bc_mask[row_of], is_diag.astype(data.dtype), data)
            return res, data

        self._res_bc, self._res_nobc, self._res_jac_bc = res_bc, res_nobc, res_jac_bc

    # ------------------------------------------------------------------
    def _args(self):
        p = self.problem
        cell_args = (p.physical_quad_points, p.shape_grads, p.JxW, p.v_grads_JxW)
        iv = tuple(p.internal_vars)
        faces = self._surface_data()
        ivs = tuple(tuple(v) for v in p.internal_vars_surfaces)
        f = getattr(p, "f_ext_nodes", None)
        f_ext = np.zeros(self.ndofs) if f is None else \
            np.concatenate([np.asarray(f).reshape(-1), np.zeros(self.N)])
        return cell_args, iv, faces, ivs, f_ext

    def bc_arrays(self):
        """Per-dof Dirichlet mask and target values, last entry wins (as the
        sequential ``.at[].set`` of ``apply_bc_vec`` / ``assign_bc``)."""
        p = self.problem
        mask = onp.zeros(self.ndofs, dtype=bool)
        target = onp.zeros(self.ndofs, dtype=onp.float64)
        for k, fe in enumerate(p.fes):
            for i in range(len(fe.node_inds_list)):
                nodes = _np(fe.node_inds_list[i]).astype(onp.int64)
                dofs = nodes * fe.vec + int(_np(fe.vec_inds_list[i])) + int(p.offset[k]) \
                    if onp.ndim(fe.vec_inds_list[i]) == 0 else \
                    nodes * fe.vec + _np(fe.vec_inds_list[i]).astype(onp.int64) + int(p.offset[k])
                vals = onp.broadcast_to(_np(fe.vals_list[i]).astype(onp.float64), dofs.shape)
                mask[dofs] = True
                target[dofs] = vals
        return np.asarray(mask), np.asarray(target)

    def prepare(self):
        """Call once per Newton solve (after the BCs and parameters are set)."""
        self._build_fns()
        self._cur = self._args()
        self._bc = self.bc_arrays()
        return self._bc

    def residual(self, dofs):
        return self._res_bc(dofs, *self._cur, *self._bc)

    def residual_jacobian(self, dofs):
        return self._res_jac_bc(dofs, *self._cur, *self._bc)

    def residual_nobc(self, dofs, with_f_ext=True):
        """f_int - f_ext without Dirichlet rows (the problem's compute_residual);
        ``with_f_ext=False`` gives the internal force f_int alone."""
        self._build_fns()
        cell_args, iv, faces, ivs, f_ext = self._args()
        if not with_f_ext:
            f_ext = np.zeros_like(f_ext)
        return self._res_nobc(dofs, cell_args, iv, faces, ivs, f_ext)

