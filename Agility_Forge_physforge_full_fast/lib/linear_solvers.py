"""Linear solvers for the fixed-pattern CSR system of ``FastSystem``.

Every backend solves ``A x = b`` for the Newton increment, where ``A`` is the
CSR matrix (value array on the GPU, pattern from ``FastSystem``) with the
Dirichlet rows already replaced by identity rows, and ``x0`` carries the
Dirichlet increments.

Backends (``make_solver(name, system)``):

* ``bicgstab``        the original method: unpreconditioned BiCGSTAB on the GPU,
                      rtol = atol = 1e-10, at most 10000 iterations (the
                      algorithm of jax.scipy.sparse.linalg.bicgstab, here with
                      an iteration count).
* ``bicgstab_jacobi`` the same with a diagonal (Jacobi) preconditioner.
* ``gmres_jacobi``    restarted GMRES(50) with a Jacobi preconditioner (GPU).
* ``cusolver``        sparse QR direct solve on the GPU
                      (jax.experimental.sparse.linalg.spsolve).
* ``pardiso``         Intel MKL PARDISO on the CPU (pypardiso); the symbolic
                      analysis is done once per pattern, every solve is a
                      numerical factorisation + solve.
* ``mumps``           MUMPS LU on the CPU through petsc4py; the symbolic
                      analysis is reused while the pattern is unchanged.
* ``amgx``            NVIDIA AMGX on the GPU (pyamgx): FGMRES or BiCGSTAB with an
                      algebraic multigrid preconditioner, rtol 1e-10.

Each call returns ``(x, info)`` with the solver's iteration count where it
has one.
"""
import os
import time
from functools import partial

import numpy as onp
import jax
import jax.numpy as np
from jax import lax


# ---------------------------------------------------------------------------
def csr_matvec(data, indices, row_of, x, n):
    return jax.ops.segment_sum(data * x[indices], row_of, num_segments=n, indices_are_sorted=True)


def _bicgstab(A, b, x0, M, tol, atol, maxiter):
    """BiCGSTAB as in jax.scipy.sparse.linalg (non-legacy tolerance), with the
    iteration count returned (negative = breakdown code)."""
    bs = np.vdot(b, b).real
    atol2 = np.maximum(np.square(tol) * bs, np.square(atol))

    def cond_fun(v):
        x, r, *_, k = v
        rs = np.vdot(r, r).real
        return (rs > atol2) & (k < maxiter) & (k >= 0)

    def body_fun(v):
        x, r, rhat, alpha, omega, rho, p, q, k = v
        rho_ = np.vdot(rhat, r)
        beta = rho_ / rho * alpha / omega
        p_ = r + beta * (p - omega * q)
        phat = M(p_)
        q_ = A(phat)
        alpha_ = rho_ / np.vdot(rhat, q_)
        s = r - alpha_ * q_
        exit_early = np.vdot(s, s).real < atol2
        shat = M(s)
        t = A(shat)
        omega_ = np.vdot(t, s) / np.vdot(t, t)
        x_ = np.where(exit_early, x + alpha_ * phat, x + (alpha_ * phat + omega_ * shat))
        r_ = np.where(exit_early, s, s - omega_ * t)
        k_ = np.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
        k_ = np.where(rho_ == 0, -10, k_)
        return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

    r0 = b - A(x0)
    one = np.asarray(1.0, dtype=b.dtype)
    x, *_, k = lax.while_loop(cond_fun, body_fun, (x0, r0, r0, one, one, one, r0, r0, 0))
    return x, k


class JaxKrylov:
    def __init__(self, system, method="bicgstab", precond=None, tol=1e-10, atol=1e-10,
                 maxiter=10000, restart=50):
        self.name = method + ("_" + precond if precond else "")
        n = system.ndofs
        indices, row_of, diag_pos = system.indices, system.row_of, system.diag_pos

        @jax.jit
        def solve(data, b, x0):
            A = lambda x: csr_matvec(data, indices, row_of, x, n)
            if precond == "jacobi":
                dinv = 1.0 / data[diag_pos]
                M = lambda x: dinv * x
            else:
                M = lambda x: x
            if method == "bicgstab":
                return _bicgstab(A, b, x0, M, tol, atol, maxiter)
            x, info = jax.scipy.sparse.linalg.gmres(A, b, x0=x0, tol=tol, atol=atol, restart=restart,
                                                    maxiter=maxiter // restart, M=M,
                                                    solve_method="batched")
            return x, np.asarray(-1)
        self._solve = solve

    def __call__(self, data, b, x0):
        x, k = self._solve(data, b, x0)
        return x, {"iters": int(k)}


class CuSolverQR:
    name = "cusolver"

    def __init__(self, system, tol=1e-14):
        from jax.experimental.sparse.linalg import spsolve
        indices = system.indices.astype(np.int32)
        indptr = system.indptr.astype(np.int32)
        self._solve = jax.jit(lambda data, b: spsolve(data, indices, indptr, b, tol=tol, reorder=1))

    def __call__(self, data, b, x0):
        return self._solve(data, b), {}


class Pardiso:
    """MKL PARDISO (real nonsymmetric, mtype 11); analysis once per pattern."""
    name = "pardiso"

    def __init__(self, system):
        import scipy.sparse as sp
        import pypardiso
        self.sp = sp
        self.n = system.ndofs
        self.indptr = system.indptr_host.astype(onp.int32)
        self.indices = system.indices_host.astype(onp.int32)
        self.ps = pypardiso.PyPardisoSolver(mtype=11)
        self._analysed = False

    def _A(self, data_h):
        return self.sp.csr_matrix((data_h, self.indices, self.indptr), shape=(self.n, self.n))

    def __call__(self, data, b, x0):
        A = self._A(onp.asarray(data))
        bh = onp.asarray(b)
        if not self._analysed:
            self.ps._check_A(A)
            self.ps.set_phase(11)
            self.ps._call_pardiso(A, bh)
            self._analysed = True
        self.ps.set_phase(23)
        x = self.ps._call_pardiso(A, bh)
        return np.asarray(x), {"refine": int(self.ps.get_iparm(7))}


class Mumps:
    """MUMPS LU through PETSc; the symbolic factorisation is reused while the
    nonzero pattern is unchanged."""
    name = "mumps"

    def __init__(self, system):
        from petsc4py import PETSc
        self.PETSc = PETSc
        n = system.ndofs
        self.indptr = system.indptr_host.astype(PETSc.IntType)
        self.indices = system.indices_host.astype(PETSc.IntType)
        self.A = PETSc.Mat().createAIJ(size=(n, n), csr=(self.indptr, self.indices,
                                                         onp.zeros(len(self.indices))))
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.x = self.A.createVecRight()
        self.b = self.A.createVecLeft()

    def __call__(self, data, b, x0):
        self.A.setValuesCSR(self.indptr, self.indices, onp.asarray(data))
        self.A.assemble()
        self.b.setArray(onp.asarray(b))
        self.ksp.solve(self.b, self.x)
        return np.asarray(self.x.getArray().copy()), {}


class Amgx:
    """NVIDIA AMGX through pyamgx (host upload of the CSR values)."""
    name = "amgx"

    CONFIGS = {
        "fgmres_amg": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "FGMRES", "max_iters": 1000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "gmres_n_restart": 50,
                       "monitor_residual": 1, "store_res_history": 0,
                       "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION",
                                          "selector": "SIZE_4", "smoother": {"solver": "MULTICOLOR_DILU"},
                                          "presweeps": 1, "postsweeps": 1, "max_iters": 1,
                                          "cycle": "V", "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
                                          "dense_lu_num_rows": 256}}},
        "bicgstab_amg": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "PBICGSTAB", "max_iters": 2000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "monitor_residual": 1,
                       "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION",
                                          "selector": "SIZE_4", "smoother": {"solver": "MULTICOLOR_DILU"},
                                          "presweeps": 1, "postsweeps": 1, "max_iters": 1,
                                          "cycle": "V", "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
                                          "dense_lu_num_rows": 256}}},
        "fgmres_ilu": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "FGMRES", "max_iters": 2000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "gmres_n_restart": 100,
                       "monitor_residual": 1,
                       "preconditioner": {"solver": "MULTICOLOR_ILU", "ilu_sparsity_level": 1,
                                          "max_iters": 1}}},
    }

    def __init__(self, system, config="fgmres_amg"):
        import pyamgx
        import scipy.sparse as sp
        self.pyamgx, self.sp = pyamgx, sp
        if not getattr(Amgx, "_initialized", False):
            pyamgx.initialize()
            Amgx._initialized = True
        self.name = "amgx_" + config
        self.cfg = pyamgx.Config().create_from_dict(self.CONFIGS[config])
        self.rsc = pyamgx.Resources().create_simple(self.cfg)
        self.n = system.ndofs
        self.indptr = system.indptr_host.astype(onp.int32)
        self.indices = system.indices_host.astype(onp.int32)
        self.M = pyamgx.Matrix().create(self.rsc)
        self.vb = pyamgx.Vector().create(self.rsc)
        self.vx = pyamgx.Vector().create(self.rsc)
        self.solver = pyamgx.Solver().create(self.rsc, self.cfg)
        self._uploaded = False

    def __call__(self, data, b, x0):
        d = onp.asarray(data)
        if not self._uploaded:
            self.M.upload_CSR(self.sp.csr_matrix((d, self.indices, self.indptr), shape=(self.n, self.n)))
            self._uploaded = True
        else:
            self.M.replace_coefficients(d)
        self.vb.upload(onp.asarray(b))
        self.vx.upload(onp.asarray(x0))
        self.solver.setup(self.M)
        self.solver.solve(self.vb, self.vx)
        x = self.vx.download()
        return np.asarray(x), {"iters": int(self.solver.iterations_number), "status": self.solver.status}


class AmgxBlock(Amgx):
    """AMGX on the node-interleaved system: the dofs are reordered node by
    node (u_x, u_y, u_z, T) and the matrix is uploaded as 4 x 4 blocks, so the
    multigrid coarsens nodes rather than single dofs."""

    BLOCK_CONFIGS = {
        "agg": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "FGMRES", "max_iters": 2000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "gmres_n_restart": 60,
                       "monitor_residual": 1,
                       "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION", "selector": "SIZE_8",
                                          "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.75},
                                          "presweeps": 2, "postsweeps": 2, "max_iters": 1, "cycle": "V",
                                          "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
                                          "dense_lu_num_rows": 256}}},
        "agg_dilu": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "FGMRES", "max_iters": 500, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "gmres_n_restart": 60,
                       "monitor_residual": 1,
                       "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION", "selector": "SIZE_4",
                                          "smoother": {"solver": "MULTICOLOR_DILU"},
                                          "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
                                          "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
                                          "dense_lu_num_rows": 256}}},
        "dilu": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "FGMRES", "max_iters": 2000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "gmres_n_restart": 100,
                       "monitor_residual": 1,
                       "preconditioner": {"solver": "MULTICOLOR_DILU", "max_iters": 1}}},
        # 'agg' with one symmetric multicolour block Gauss-Seidel sweep before
        # and after each coarsening instead of two damped block Jacobi sweeps:
        # about 3x the cost per solve, but it solves the large hard-press
        # systems on which 'agg' stalls (all 42 saved systems, 115k-2M dofs,
        # to 1e-10 in <= 340 iterations); used as the second try ('bgs': the
        # same with two sweeps and BiCGSTAB outside)
        "gs": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "FGMRES", "max_iters": 2000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "gmres_n_restart": 60,
                       "monitor_residual": 1,
                       "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION", "selector": "SIZE_8",
                                          "smoother": {"solver": "MULTICOLOR_GS", "symmetric_GS": 1},
                                          "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
                                          "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
                                          "dense_lu_num_rows": 256}}},
        "bgs": {
            "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
            "solver": {"solver": "PBICGSTAB", "max_iters": 2000, "tolerance": 1e-10,
                       "convergence": "RELATIVE_INI_CORE", "norm": "L2", "monitor_residual": 1,
                       "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION", "selector": "SIZE_8",
                                          "smoother": {"solver": "MULTICOLOR_GS", "symmetric_GS": 1},
                                          "presweeps": 2, "postsweeps": 2, "max_iters": 1, "cycle": "V",
                                          "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
                                          "dense_lu_num_rows": 256}}},
    }

    def __init__(self, system, config="agg"):
        import pyamgx
        import scipy.sparse as sp
        self.pyamgx, self.sp = pyamgx, sp
        if not getattr(Amgx, "_initialized", False):
            pyamgx.initialize()
            Amgx._initialized = True
        self.name = "amgxb_" + config
        n = system.ndofs
        N = n // 4
        d = onp.arange(n)
        p = onp.where(d < 3 * N, 4 * (d // 3) + d % 3, 4 * (d - 3 * N) + 3)       # new index of dof d
        self.p = p
        # position of every scalar nonzero in the 4 x 4 block storage
        indptr, indices = system.indptr_host, system.indices_host
        rows = onp.repeat(onp.arange(n), onp.diff(indptr))
        tag = sp.csr_matrix((onp.arange(1, len(indices) + 1, dtype=onp.float64),
                             (p[rows], p[indices])), shape=(n, n))
        bsr = sp.bsr_matrix(tag, blocksize=(4, 4))
        bsr.sort_indices()
        blocks = bsr.data                                            # (nb, 4, 4) of tags
        assert onp.count_nonzero(blocks) == len(indices), "pattern is not block-complete"
        flat = blocks.reshape(-1)
        self.gather = onp.zeros(flat.shape[0], dtype=onp.int64)     # block slot -> scalar nnz (or -1)
        nz = flat > 0
        self.gather[nz] = flat[nz].astype(onp.int64) - 1
        self.zero_slots = ~nz
        self.bindptr = bsr.indptr.astype(onp.int32)
        self.bindices = bsr.indices.astype(onp.int32)
        self.nb = blocks.shape[0]
        # block values are formed on the GPU and handed to AMGX as a device
        # pointer (__cuda_array_interface__): the matrix never visits the host
        g = np.asarray(self.gather.astype(onp.int32 if len(indices) < 2**31 - 1 else onp.int64))
        zs = np.asarray(self.zero_slots)
        self._blockify = jax.jit(lambda data: np.where(zs, 0.0, data[g]))
        self.device_values = os.environ.get("AGF_AMGX_HOST_VALUES", "0") != "1"
        self.cfg = pyamgx.Config().create_from_dict(self.BLOCK_CONFIGS[config])
        self.rsc = pyamgx.Resources().create_simple(self.cfg)
        self.M = pyamgx.Matrix().create(self.rsc)
        self.vb = pyamgx.Vector().create(self.rsc)
        self.vx = pyamgx.Vector().create(self.rsc)
        self.solver = pyamgx.Solver().create(self.rsc, self.cfg)
        self._uploaded = False

    def __call__(self, data, b, x0):
        bd_dev = self._blockify(np.asarray(data))
        bd_dev.block_until_ready()
        if not self._uploaded:
            bd = onp.ascontiguousarray(onp.asarray(bd_dev).reshape(self.nb, 4, 4))
            self.M.upload(self.bindptr, self.bindices, bd, block_dims=[4, 4])
            self._uploaded = True
        elif self.device_values:
            self.M.replace_coefficients(bd_dev)          # device pointer
        else:
            self.M.replace_coefficients(onp.asarray(bd_dev))
        bp = onp.empty_like(onp.asarray(b)); bp[self.p] = onp.asarray(b)
        xp = onp.empty_like(onp.asarray(x0)); xp[self.p] = onp.asarray(x0)
        self.vb.upload(bp, block_dim=4)
        self.vx.upload(xp, block_dim=4)
        self.solver.setup(self.M)
        self.solver.solve(self.vb, self.vx)
        # pyamgx's download() allocates only (block count) values for a block
        # vector, while AMGX writes block count x 4: pass a full-size buffer
        buf = onp.zeros(len(self.p), dtype=onp.float64)
        self.vx.download(buf)
        x = buf[self.p]
        return np.asarray(x), {"iters": int(self.solver.iterations_number), "status": self.solver.status}


class AmgxBlockElim(AmgxBlock):
    """AmgxBlock on an equivalent system in which the Dirichlet columns are
    eliminated as well: for a free row f, b_f <- b_f - sum_d A_fd b_d over the
    Dirichlet dofs d (whose rows say x_d = b_d), then A_fd = 0; the Dirichlet
    rows keep x_d = b_d, scaled by the mean |diagonal| of the free dofs of the
    same field (u or T) so that they do not stand out in the multigrid.  The
    solution is the same; the transformation runs on the GPU at every solve
    (the contact set changes from step to step)."""

    def __init__(self, system, config="gs"):
        super().__init__(system, config)
        self.name = "amgxbe_" + config
        self._system = system
        n, N = system.ndofs, system.ndofs // 4
        indices, row_of, diag_pos = system.indices, system.row_of, system.diag_pos
        isu = np.arange(n) < 3 * N

        @jax.jit
        def eliminate(data, b, bc):
            m = bc[indices] & ~bc[row_of]                    # free row, Dirichlet column
            xb = np.where(bc, b, 0.0)
            b2 = b - jax.ops.segment_sum(np.where(m, data * xb[indices], 0.0), row_of,
                                         num_segments=n, indices_are_sorted=True)
            d2 = np.where(m, 0.0, data)
            dg = np.abs(data[diag_pos])
            fu, ft = isu & ~bc, ~isu & ~bc
            a_u = np.sum(np.where(fu, dg, 0.0)) / np.maximum(np.sum(fu), 1)
            a_t = np.sum(np.where(ft, dg, 0.0)) / np.maximum(np.sum(ft), 1)
            alpha = np.where(isu, a_u, a_t)
            d2 = d2.at[diag_pos].set(np.where(bc, alpha, d2[diag_pos]))
            return d2, np.where(bc, alpha * b, b2)
        self._eliminate = eliminate

    def __call__(self, data, b, x0):
        d2, b2 = self._eliminate(np.asarray(data), np.asarray(b), self._system._bc[0])
        return super().__call__(d2, b2, x0)


def make_solver(name, system):
    if name == "auto":
        # measured on one H100 (README): PARDISO is fastest up to about 5e4
        # dofs (0.145 s at 36k, but 1.4 s at 115k and 91 s at 874k); above
        # that block AMGX (0.22 s at 115k, 0.9-2 s at 874k); BiCGSTAB if
        # neither library imports
        tried = []
        order = (["pardiso", "amgxb:agg"] if system.ndofs <= 50_000 else ["amgxb:agg", "pardiso"])
        for cand in order:
            try:
                s = make_solver(cand, system)
                print(f"[fast] linear solver: {s.name} ({system.ndofs} dofs)", flush=True)
                return s
            except Exception as e:                          # noqa: BLE001
                tried.append(f"{cand}: {type(e).__name__}: {e}")
        print("[fast] " + "; ".join(tried) + " -> using BiCGSTAB", flush=True)
        return JaxKrylov(system, "bicgstab", None)
    if name == "bicgstab":
        return JaxKrylov(system, "bicgstab", None)
    if name == "bicgstab_jacobi":
        return JaxKrylov(system, "bicgstab", "jacobi")
    if name == "gmres_jacobi":
        return JaxKrylov(system, "gmres", "jacobi")
    if name == "cusolver":
        return CuSolverQR(system)
    if name == "pardiso":
        return Pardiso(system)
    if name == "mumps":
        return Mumps(system)
    if name.startswith("amgxbe"):
        cfg = name.split(":", 1)[1] if ":" in name else "gs"
        return AmgxBlockElim(system, cfg)
    if name.startswith("amgxb"):
        cfg = name.split(":", 1)[1] if ":" in name else "agg"
        return AmgxBlock(system, cfg)
    if name.startswith("amgx"):
        cfg = name.split(":", 1)[1] if ":" in name else "fgmres_amg"
        return Amgx(system, cfg)
    raise ValueError(f"unknown linear solver {name!r}")


def available_direct(system):
    """A direct solver for the fallback: PARDISO if importable, else MUMPS,
    else scipy's spsolve on the host (the original fallback)."""
    for name in ("pardiso", "mumps"):
        try:
            return make_solver(name, system)
        except Exception:                                   # noqa: BLE001
            pass

    class _Scipy:
        name = "scipy_spsolve"

        def __init__(self, s):
            import scipy.sparse as sp
            import scipy.sparse.linalg as spl
            self.sp, self.spl, self.s = sp, spl, s

        def __call__(self, data, b, x0):
            A = self.sp.csr_matrix((onp.asarray(data), self.s.indices_host, self.s.indptr_host),
                                   shape=(self.s.ndofs, self.s.ndofs)).tocsc()
            return np.asarray(self.spl.spsolve(A, onp.asarray(b))), {}
    return _Scipy(system)
