"""Faster linear algebra for the row-elimination Newton solver (opt-in: --fast-linear).

Everything here rebinds names inside jax_fem_checkpoint.solver_latest at run
time, like the other use_* functions in lib/time_stepper.py; the package is not
edited.  The equations solved are the same; only the bookkeeping changes.

What the package does every Newton iteration (get_A + the MUMPS hook):
  scipy COO -> CSR of all triplets with duplicate summation, a PETSc AIJ matrix,
  zeroRows for the Dirichlet rows, getValuesCSR, a JAX BCOO on the device; then
  mumps_solve_host copies the BCOO back to the host, builds a scipy CSR and a
  new PETSc matrix again, and MUMPS runs analysis (ordering + symbolic
  factorisation) and numerical factorisation from scratch.

What this module does instead:
  get_A      the CSR pattern of the cell triplets is built once per mesh
             (np.unique of the row/column keys); each iteration sums the values
             into it with np.bincount.  The contact and friction triplets that
             the problem appends after the cell triplets are nodal blocks that
             lie inside the cell pattern; their positions are looked up once per
             contact layout (searchsorted) and added with np.add.at.  Dirichlet
             rows are cut down to a 1 on the diagonal, exactly the pattern and
             values PETSc zeroRows(rows, diag=1) leaves (it drops the rest of the
             row).  The result stays a host CSR.
  solve      one PETSc AIJ matrix and one KSP (preonly, LU, MUMPS, METIS
             ordering ICNTL(7)=5 as in mumps_solve_host) per pattern; new values
             are copied into the same matrix, so PETSc keeps the MUMPS analysis
             and only refactors numerically.  |Ax - b| is checked as before.
  line search  the residual at the current iterate, which the descent line
             search used to recompute, is taken from the Newton update that
             just computed it at the same dofs (exact array comparison).

Results equal the package path up to round-off (summation order of duplicate
triplets, and MUMPS pivoting with a reused analysis); verified by A/B runs.
"""
import time

import numpy as onp

_STATE = dict(pattern=None, ext=None, bc=None, ksp=None, last_res=None,
              n_build=0, n_reuse=0, t_asm=0.0, t_solve=0.0, n_solve=0)


def _fp(a):
    """Cheap fingerprint of an integer index array."""
    a = onp.asarray(a)
    if a.size == 0:
        return (0,)
    s = a[:: max(1, a.size // 4096)].astype(onp.int64)
    return (a.size, int(a.sum(dtype=onp.int64)), int(s.dot(onp.arange(s.size, dtype=onp.int64) % 1009)))


class _Pattern:
    def __init__(self, I, J, N):
        t0 = time.time()
        k = onp.asarray(I, dtype=onp.int64) * N + onp.asarray(J, dtype=onp.int64)
        self.keys, inv = onp.unique(k, return_inverse=True)
        self.inv = inv.reshape(-1)
        rows = self.keys // N
        self.N, self.nnz = N, self.keys.size
        self.indptr = onp.zeros(N + 1, dtype=onp.int64)
        onp.cumsum(onp.bincount(rows, minlength=N), out=self.indptr[1:])
        self.indices = (self.keys % N).astype(onp.int64)
        self.rows = rows
        d = onp.arange(N, dtype=onp.int64)
        self.diag = onp.searchsorted(self.keys, d * N + d)
        if not onp.all(self.keys[onp.minimum(self.diag, self.nnz - 1)] == d * N + d):
            raise RuntimeError("fast_linear: a diagonal entry is missing from the assembled pattern")
        self.build_s = time.time() - t0

    def positions(self, I, J):
        k = onp.asarray(I, dtype=onp.int64) * self.N + onp.asarray(J, dtype=onp.int64)
        pos = onp.searchsorted(self.keys, k)
        pos = onp.minimum(pos, self.nnz - 1)
        return pos if onp.all(self.keys[pos] == k) else None

    def reduced(self, bc_rows):
        """Pattern with the Dirichlet rows cut down to their diagonal, as PETSc
        zeroRows leaves them (keepnonzeropattern is off by default)."""
        is_bc = onp.zeros(self.N, dtype=bool)
        is_bc[onp.asarray(bc_rows, dtype=onp.int64)] = True
        keep = ~is_bc[self.rows] | (self.indices == self.rows)
        kidx = onp.nonzero(keep)[0]
        indptr = onp.zeros(self.N + 1, dtype=onp.int64)
        onp.cumsum(onp.bincount(self.rows[kidx], minlength=self.N), out=indptr[1:])
        red = _Reduced(kidx, indptr, self.indices[kidx])
        bcr = onp.nonzero(is_bc)[0]
        red.bc_diag = indptr[bcr]            # a reduced Dirichlet row holds only its diagonal
        return red


class _Reduced:
    def __init__(self, kidx, indptr, indices):
        self.kidx, self.indptr, self.indices = kidx, indptr, indices


class CSRSystem:
    """What get_A hands to the linear solver: the assembled, row-eliminated CSR."""

    def __init__(self, csr, pattern):
        self.csr, self.pattern, self.shape = csr, pattern, csr.shape


def install(logger=None):
    """Rebind get_A, the MUMPS solve and the line search's current residual."""
    import scipy.sparse as sp
    from jax_fem_checkpoint import solver_latest as _sl
    from lib import time_stepper as _ts

    if getattr(_sl, "_agf_fast_linear", False):
        return
    _ts.use_mumps_linear_solver()          # sets the flag so it cannot overwrite us later
    log = logger or _sl.logger
    plain_solve = _sl.umfpack_solve
    plain_apply_bc_vec = _sl.apply_bc_vec

    def bc_rows(problem):
        out = []
        for ind, fe in enumerate(problem.fes):
            for ni, vi in zip(fe.node_inds_list, fe.vec_inds_list):
                out.append(onp.asarray(ni, dtype=onp.int64) * fe.vec + onp.asarray(vi, dtype=onp.int64)
                           + int(problem.offset[ind]))
        return onp.concatenate(out) if out else onp.zeros(0, dtype=onp.int64)

    def get_A(problem, solver_options):
        if hasattr(problem, "P_mat"):
            raise NotImplementedError("fast_linear: linear multipoint constraints are not supported")
        t0 = time.time()
        st = _STATE
        N = int(problem.num_total_dofs_all_vars)
        I, J, V = onp.asarray(problem.I), onp.asarray(problem.J), onp.asarray(problem.V)
        n_ext = sum(n for _, n in getattr(problem, "_ext_layout", ()))
        n_base = I.size - n_ext
        key = (id(problem), N, n_base, _fp(I[:n_base]), _fp(J[:n_base]))
        pat = st["pattern"]
        if pat is None or st["pattern_key"] != key:
            pat = _Pattern(I[:n_base], J[:n_base], N)
            st.update(pattern=pat, pattern_key=key, ext=None, bc=None, n_build=st["n_build"] + 1)
            log.info(f"[fast] CSR pattern built: {N} dofs, {pat.nnz} nonzeros from {n_base} triplets "
                     f"in {pat.build_s:.1f} s")
        data = onp.bincount(pat.inv, weights=V[:n_base], minlength=pat.nnz)
        if n_ext:
            ek = (_fp(I[n_base:]), _fp(J[n_base:]))
            if st["ext"] is None or st["ext"][0] != ek:
                pos = pat.positions(I[n_base:], J[n_base:])
                if pos is None:
                    raise RuntimeError("fast_linear: contact/friction triplets fall outside the cell pattern")
                st["ext"] = (ek, pos)
            onp.add.at(data, st["ext"][1], V[n_base:])
        rows = bc_rows(problem)
        bk = _fp(rows)
        if st["bc"] is None or st["bc"][0] != bk:
            st["bc"] = (bk, pat.reduced(rows))
        red = st["bc"][1]
        data = data[red.kidx]
        data[red.bc_diag] = 1.0
        st["t_asm"] += time.time() - t0
        return CSRSystem(sp.csr_matrix((data, red.indices, red.indptr), shape=(N, N)), red)

    def solve(A, b):
        if not isinstance(A, CSRSystem):
            return plain_solve(A, b)
        from petsc4py import PETSc
        st = _STATE
        bb = onp.asarray(b, dtype=float)
        t0 = time.time()
        csr = A.csr
        k = st["ksp"]
        reuse = k is not None and k["pattern"] is A.pattern
        if not reuse:
            if k is not None:
                for o in (k["ksp"], k["M"], k["x"], k["rhs"]):
                    o.destroy()
            M = PETSc.Mat().createAIJ(size=csr.shape, csr=(csr.indptr.astype(PETSc.IntType),
                                                           csr.indices.astype(PETSc.IntType), csr.data))
            M.assemble()
            ksp = PETSc.KSP().create()
            ksp.setOperators(M)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            PETSc.Options()["mat_mumps_icntl_7"] = 5          # METIS ordering, as mumps_solve_host
            ksp.setFromOptions()
            k = st["ksp"] = dict(pattern=A.pattern, M=M, ksp=ksp, x=M.createVecRight(), rhs=M.createVecLeft(),
                                 ip=csr.indptr.astype(PETSc.IntType), ix=csr.indices.astype(PETSc.IntType))
        else:
            k["M"].setValuesCSR(k["ip"], k["ix"], csr.data)
            k["M"].assemble()
            k["ksp"].setOperators(k["M"])
            st["n_reuse"] += 1
        k["rhs"].setArray(bb)
        k["ksp"].solve(k["rhs"], k["x"])
        out = k["x"].getArray().copy()
        res = float(onp.linalg.norm(csr @ out - bb))
        dt = time.time() - t0
        st["t_solve"] += dt
        st["n_solve"] += 1
        log.info(f"MUMPS solve {dt:.1f} s ({'analysis reused' if reuse else 'new analysis'}), "
                 f"|Ax-b| = {res:.3e} (|b| = {onp.linalg.norm(bb):.3e})")
        return out.astype(onp.asarray(b).dtype)

    def apply_bc_vec(res_vec, dofs, problem, scale=1.):
        out = plain_apply_bc_vec(res_vec, dofs, problem, scale)
        if scale == 1.:
            _STATE["last_res"] = (id(problem), onp.array(dofs, copy=True), out)
        return out

    _sl.get_A = get_A
    _sl.umfpack_solve = solve
    _sl.apply_bc_vec = apply_bc_vec
    _sl._agf_fast_linear = True
    log.info("[fast] fast_linear installed: cached CSR pattern, MUMPS analysis reuse, residual reuse "
             "in the line search")


def cached_residual(problem, dofs):
    """The row-eliminated residual at exactly these dofs, if the last Newton update computed it."""
    lr = _STATE.get("last_res")
    if lr is None or lr[0] != id(problem):
        return None
    d = onp.asarray(dofs)
    return lr[2] if (lr[1].shape == d.shape and onp.array_equal(lr[1], d)) else None


def stats():
    st = _STATE
    return (f"[fast] patterns built {st['n_build']}, solves {st['n_solve']} "
            f"({st['n_reuse']} with the analysis reused), assembly {st['t_asm']:.0f} s, "
            f"solve {st['t_solve']:.0f} s")
