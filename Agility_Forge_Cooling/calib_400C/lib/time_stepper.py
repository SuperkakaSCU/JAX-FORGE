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

# the same module object run_cold.py catches the exception from (both the
# code directory and lib/ are on sys.path; importing it as `mesh_quality`
# would make a second, different DistortionStop class)
from lib.mesh_quality import DistortionStop


def use_hybrid_linear_solver(rel_tol=1e-6):
    """bicgstab first; redo with a direct factorisation when it fails.

    The package's bicgstab is unpreconditioned (`pc = None` is hardcoded, the
    Jacobi branch beside it is commented out) and its convergence check is
    commented out too, so a failed solve is returned as if it had succeeded.
    Measured on hit 8 of roll3.2's release, where the runs die: 129 of 455
    linear solves came back with |Ax-b| above 1e-3|b|, and the last one at
    |res| = 3.58e5 against |b| = 3.77e5 -- a "solution" that barely moved the
    residual at all.  Newton then steps on it, and the step-16 iterate had
    travelled 32.5 mm before the acceptance check refused it.

    Running everything through the direct solver instead costs 16 s a solve
    against 1 s, measured, which is 16x on the dominant term.  Redoing only the
    failures costs that on the fraction that actually fail -- 28% at the worst
    release, far less elsewhere.  The check restored here is the one the
    package has commented out, with a relative bound rather than the absolute
    0.1 it used.
    """
    from jax_fem_checkpoint import solver_latest as _sl
    import numpy as _onp
    if getattr(_sl, "_agf_hybrid_installed", False):
        return
    _inner = _sl.linear_solver

    def _hybrid(A, b, x0, solver_options):
        x = _inner(A, b, x0, solver_options)
        try:
            res = float(_onp.linalg.norm(_onp.asarray(A @ x) - _onp.asarray(b)))
            rhs = float(_onp.linalg.norm(_onp.asarray(b)))
        except Exception:
            return x
        if _onp.isfinite(res) and res <= rel_tol * max(rhs, 1.0):
            return x
        _sl.logger.warning(
            f"linear solve rejected: |Ax-b|={res:.3e} vs |b|={rhs:.3e} "
            f"-- redoing with the direct factorisation")
        return _sl.umfpack_solve_host(A, b)

    _sl.linear_solver = _hybrid
    _sl._agf_hybrid_installed = True


def use_direct_linear_solver():
    """Make the package's 'umfpack_solver' option reach a real direct solve.

    jax_fem_checkpoint offers a solver option called 'umfpack_solver', but the
    function it dispatches to is an AMGX wrapper, and pyamgx is not installed
    here -- the package says so on import.  The genuine direct solve sits next
    to it, unused: umfpack_solve_host builds a scipy CSR from the assembled
    BCOO and calls scipy.sparse.linalg.spsolve.  Nothing dispatches to it.
    Rebinding the name from here reaches it without editing the package, which
    stays an import-only dependency.

    Why bother: the default linear solver is an UNPRECONDITIONED bicgstab whose
    convergence check is commented out in the package, so a failed solve is
    returned silently.  Measured on hit 6 of roll3.2's release, where the runs
    die: the linear residual came back at 4.36e7 against a right-hand side of
    2.06e4 -- the "solution" was two thousand times worse than doing nothing --
    and Newton then stepped on it and went to NaN.  A direct factorisation
    cannot fail that way.
    """
    from jax_fem_checkpoint import solver_latest as _sl
    if _sl.umfpack_solve is not _sl.umfpack_solve_host:
        _sl.umfpack_solve = _sl.umfpack_solve_host


def _fe_for_writing(fe):
    """The FE as save_sol should see it: HEX8 cells back in meshio/VTK order.

    jax_fem_checkpoint holds HEX8 cells in its own node order (lib/hex8.py);
    written as they are, every hexahedron in the VTU would be twisted.  The
    package's save_sol only reads ele_type, points, cells and num_cells, so a
    light proxy with the cells permuted back is enough -- the package
    function itself is used unchanged.
    """
    if fe.ele_type != "HEX8":
        return fe
    from types import SimpleNamespace
    from hex8 import to_vtk
    return SimpleNamespace(ele_type=fe.ele_type, points=fe.points,
                           cells=to_vtk(onp.asarray(fe.cells)), num_cells=fe.num_cells)


def mumps_solve_host(A, b, logger=None):
    """One direct solve of A x = b with PETSc/MUMPS (METIS ordering), on the CPU.

    The assembled matrix arrives as a BCOO that may hold duplicate entries, so
    it is summed into CSR before it is handed to PETSc.
    """
    import numpy as _onp
    import scipy.sparse as _sp
    from petsc4py import PETSc
    ij = _onp.asarray(A.indices)
    Asp = _sp.csr_matrix((_onp.asarray(A.data), (ij[:, 0], ij[:, 1])), shape=A.shape)
    Asp.sum_duplicates()
    Asp.sort_indices()
    bb = _onp.asarray(b, dtype=float)
    M = PETSc.Mat().createAIJ(size=Asp.shape,
                              csr=(Asp.indptr.astype(PETSc.IntType),
                                   Asp.indices.astype(PETSc.IntType), Asp.data))
    M.assemble()
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    PETSc.Options()["mat_mumps_icntl_7"] = 5          # METIS ordering
    ksp.setFromOptions()
    x = M.createVecRight()
    rhs = M.createVecLeft()
    rhs.setArray(bb)
    t0 = time.time()
    ksp.solve(rhs, x)
    out = x.getArray().copy()
    res = float(_onp.linalg.norm(Asp @ out - bb))
    if logger is not None:
        logger.info(f"MUMPS solve {time.time() - t0:.1f} s, |Ax-b| = {res:.3e} "
                    f"(|b| = {_onp.linalg.norm(bb):.3e})")
    for o in (ksp, M, x, rhs):
        o.destroy()
    return out.astype(_onp.asarray(b).dtype)


def use_mumps_linear_solver():
    """Route the package's 'umfpack_solver' option to a MUMPS factorisation.

    Why this exists: on TET10 meshes the unpreconditioned bicgstab fails on
    the very first solve (|Ax-b| 70-700x |b|), so 'hybrid' falls through to
    scipy's spsolve on every solve -- SuperLU with its default column
    ordering, which on a 3D quadratic mesh did not finish ONE factorisation of
    the 292k-dof system in 15 minutes.  MUMPS with METIS ordering, through the
    PETSc that is already installed in this environment, factors the same
    pattern in 24 s (9 s at 177k dofs) to a relative residual of 1e-12.

    Like use_direct_linear_solver this only rebinds a name inside
    jax_fem_checkpoint.solver_latest at run time; the package is not edited.
    """
    from jax_fem_checkpoint import solver_latest as _sl
    if getattr(_sl, "_agf_mumps_installed", False):
        return

    def _mumps_solve(A, b):
        return mumps_solve_host(A, b, _sl.logger)

    _sl.umfpack_solve = _mumps_solve
    _sl._agf_mumps_installed = True


def use_descent_line_search(armijo=1e-4, max_halvings=8, roundoff_rel=1e-9):
    """Replace the package's line search by backtracking with a descent test.

    jax_fem_checkpoint.solver_latest.line_search compares the candidates
    alpha = 1, 1/2, 1/4, 1/8 with each other and never with the residual of the
    current iterate, so it can return a step that raises the residual.
    Measured on hit 4 of pen_flat_h20_12: from 0.69 to 1310, in a limit cycle
    that ran to the iteration limit, and 68 of the last 400 iterations raised
    the residual by more than 1.5x.

    Here alpha is halved until |R(x + alpha dx)| <= (1 - armijo alpha) |R(x)|
    over the free dofs, with a finite residual.  When no alpha qualifies, RuntimeError goes to the
    stepper, which retries with a smaller dt.  The convergence test itself is
    untouched.  Rebinds a name at run time; the package is not edited.

    Every iteration prints one [newton] line: mechanical and thermal residuals
    apart, the step taken and, through problem.newton_monitor(u) if set, a
    geometry measure of the accepted iterate.
    """
    import numpy as _onp
    from jax_fem_checkpoint import solver_latest as _sl
    if getattr(_sl, "_agf_descent_ls", False):
        return

    def line_search(problem, dofs, inc):
        res_fn = _sl.apply_bc(_sl.get_flatten_fn(problem.compute_residual, problem), problem)
        fe = problem.fes[0]
        n_u = fe.num_total_nodes * fe.vec
        # Rows of prescribed dofs hold u - u_bar, in mm, not a force.  At the
        # start of a step with a moving Dirichlet contact they are the whole
        # residual (0.47 on step 1 of the prescribed-displacement line, free
        # rows 0), and applying them is what creates the out-of-balance force,
        # so a descent test on the mixed norm stalls at alpha ~ 0.004.  The
        # descent test therefore reads the free rows, and a step that still has
        # prescribed motion to apply takes it whole.
        free = _onp.ones(problem.num_total_dofs_all_vars, dtype=bool)
        for ni, vi in zip(fe.node_inds_list, fe.vec_inds_list):
            free[_onp.asarray(ni) * fe.vec + _onp.asarray(vi)] = False
        now = None
        if getattr(_sl, "_agf_fast_linear", False):
            # --fast-linear: the Newton update has just computed this residual at
            # exactly these dofs (lib/fast_linear.py checks the arrays are equal)
            from lib.fast_linear import cached_residual
            now = cached_residual(problem, dofs)
        now = _onp.asarray(res_fn(dofs) if now is None else now)
        r_now = float(_onp.linalg.norm(now[free]))
        bc_gap = float(_onp.linalg.norm(now[~free]))
        applying_bc = bc_gap > 1e-9
        alpha = 1.0
        r = float("nan")
        for halvings in range(max_halvings + 1):
            trial = dofs + alpha * inc
            vec = _onp.asarray(res_fn(trial))
            r = float(_onp.linalg.norm(vec[free]))
            ok = _onp.isfinite(r) and _onp.isfinite(_onp.linalg.norm(vec))
            if ok and (applying_bc or r <= (1.0 - armijo * alpha) * r_now):
                du = float(_onp.max(_onp.abs(_onp.asarray(inc)[:n_u])))
                mon = getattr(problem, "newton_monitor", None)
                extra = "" if mon is None else " " + mon(_onp.asarray(trial)[:n_u].reshape(-1, fe.vec))
                print(f"  [newton] free R {r_now:.4e} -> {r:.4e} (mech {_onp.linalg.norm(vec[:n_u][free[:n_u]]):.4e}, "
                      f"thermal {_onp.linalg.norm(vec[n_u:]):.4e}), bc gap {bc_gap:.3e} mm, alpha {alpha:.4g}, "
                      f"max|du| {du:.4e} mm{extra}", flush=True)
                return trial
            alpha *= 0.5
        # Which kind of failure?  A residual that is already at the round-off
        # floor of the assembled forces cannot be reduced by any step: that is
        # not a missing descent direction (a contact jump, a fold), it is the
        # convergence test asking for more digits than the arithmetic has.
        # Measured: 2e-6 N on a 46 kN load (4e-11) in the sticking upset test.
        # The load scale is the larger of the reactions on the prescribed dofs
        # and the die contact forces.  Both cases still fail the step; only the
        # message differs, so that the two are never confused in a report.
        scale = 0.0
        try:
            sol_list = problem.unflatten_fn_sol_list(dofs)
            raw = _onp.asarray(problem.compute_residual(sol_list)[0]).reshape(-1)
            scale = float(_onp.linalg.norm(raw[~free[:n_u]]))
            dc = getattr(problem, "die_contact", None)
            if dc is not None:
                scale = max(scale, float(_onp.linalg.norm(
                    dc._evaluate(_onp.asarray(sol_list[0]), False)["f"])))
        except Exception:
            pass
        if scale > 0.0 and r_now <= roundoff_rel * scale:
            raise RuntimeError(f"round-off stall: free |R| = {r_now:.4e} N is {r_now / scale:.1e} "
                               f"of the load scale {scale:.4e} N and cannot be reduced further "
                               f"(last trial |R| = {r:.4e}); the convergence test is below the "
                               f"round-off floor, this is not a missing descent direction")
        raise RuntimeError(f"line search found no descent from free |R| = {r_now:.4e} "
                           f"after {max_halvings} halvings (last |R| = {r:.4e}; load scale "
                           f"{scale:.4e} N)")

    _sl.line_search = line_search
    _sl._agf_descent_ls = True


def use_gpu_hybrid_linear_solver(rel_tol=1e-6, krylov_tol=1e-8, maxiter=5000):
    """GPU first: block-Jacobi-preconditioned bicgstab; MUMPS only for failures.

    The package's own GPU path is an UNPRECONDITIONED bicgstab whose result is
    never checked (see use_hybrid_linear_solver).  This keeps the solve on the
    GPU but (1) preconditions it with the inverse of each node's 3x3 diagonal
    block, built on the GPU from the assembled matrix itself, and (2) checks
    the true residual |Ax - b| <= rel_tol * max(|b|, 1) -- the same acceptance
    test as 'hybrid'.  A solve that fails it is redone with MUMPS on the CPU
    (mumps_solve_host), not with scipy's spsolve, which is unusably slow on
    3D meshes of this size.

    The Krylov solve is jitted on (data, indices, b, x0) so that consecutive
    Newton iterations with the same sparsity pattern reuse one compilation.
    Counters on the solver module record how many solves stayed on the GPU,
    so the fallback rate can be read off the log afterwards.

    Like the other use_* functions this only rebinds a name inside
    jax_fem_checkpoint.solver_latest at run time; the package is not edited.
    """
    from functools import partial
    import numpy as _onp
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
    from jax_fem_checkpoint import solver_latest as _sl
    if getattr(_sl, "_agf_gpu_hybrid_installed", False):
        return

    @partial(jax.jit, static_argnames=("shape",))
    def _krylov(data, indices, b, x0, shape):
        A = BCOO((data, indices), shape=shape)
        r, c = indices[:, 0], indices[:, 1]
        same = (r // 3) == (c // 3)
        B = jnp.zeros((shape[0] // 3, 3, 3), dtype=data.dtype)
        B = B.at[r // 3, r % 3, c % 3].add(jnp.where(same, data, 0.0))
        det = jnp.linalg.det(B)
        B = jnp.where((jnp.abs(det) > 0.0)[:, None, None], B, jnp.eye(3, dtype=data.dtype))
        Binv = jnp.linalg.inv(B)
        pc = lambda v: jnp.einsum("nij,nj->ni", Binv, v.reshape(-1, 3)).reshape(-1)
        x, _ = jax.scipy.sparse.linalg.bicgstab(lambda v: A @ v, b, x0=x0, M=pc,
                                                tol=krylov_tol, atol=0.0, maxiter=maxiter)
        res = jnp.linalg.norm(A @ x - b)
        return x, res, jnp.linalg.norm(b)

    _sl._agf_gpu_stats = dict(gpu_ok=0, fallback=0, t_gpu=0.0, t_fallback=0.0)

    def _gpu_hybrid(A, b, x0, solver_options):
        st = _sl._agf_gpu_stats
        t0 = time.time()
        bj = jnp.asarray(b)
        x0j = jnp.zeros_like(bj) if x0 is None else jnp.asarray(x0)
        x, res, rhs = _krylov(A.data, A.indices, bj, x0j, shape=tuple(A.shape))
        x.block_until_ready()
        res, rhs = float(res), float(rhs)
        dt = time.time() - t0
        if _onp.isfinite(res) and res <= rel_tol * max(rhs, 1.0):
            st["gpu_ok"] += 1
            st["t_gpu"] += dt
            _sl.logger.info(f"GPU bicgstab+bJacobi {dt:.2f} s, |Ax-b| = {res:.3e} "
                            f"(|b| = {rhs:.3e})  [gpu {st['gpu_ok']} / fallback {st['fallback']}]")
            return x
        _sl.logger.warning(f"GPU solve rejected after {dt:.2f} s: |Ax-b|={res:.3e} vs "
                           f"|b|={rhs:.3e} -- redoing with MUMPS on the CPU")
        t1 = time.time()
        out = mumps_solve_host(A, b, _sl.logger)
        st["fallback"] += 1
        st["t_gpu"] += dt
        st["t_fallback"] += time.time() - t1
        return jnp.asarray(out)

    _sl.linear_solver = _gpu_hybrid
    _sl._agf_gpu_hybrid_installed = True


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
        abs_tol=None,
        accept_motion_mm=None,
        accept_res_n=None,
        accept_fn=None,
        accept_max_refusals=3,
        accept_max_refusals_total=8,
        linear_solver="jax",
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
        # Newton convergence tolerance as an ABSOLUTE residual in newtons.
        # None leaves the solver's own default (1e-6 N, effectively unreachable
        # for a 1e5 N problem, so convergence then rests on the relative test).
        #
        # It MUST be absolute, not relative.  The solver's relative test is
        # taken against the residual each step happens to start from:
        #
        #     while (rel_res_val > rel_tol and res_val > tol) and ...
        #     rel_res_val = res_val / res_val_initial
        #
        # so a loose rel_tol rubber-stamps a step that starts from a corrupted
        # state.  Measured, on hit 6 of roll3.2 with rel_tol = 1e-4: step 3 of
        # the release began at a residual of 3.98e19 N and the solver reported
        # "Converged in 14 iterations" -- having reached 3.98e15 N, which is
        # 1e-4 of nonsense.  That state was committed and poisoned step 4, which
        # then went to NaN on every retry.  A loose relative tolerance is worse
        # than no tolerance change at all: it turns an honest failure into the
        # silent acceptance of garbage.  An absolute bound in newtons cannot be
        # fooled that way -- 4e15 N never passes it.
        #
        # The release stroke needs a looser one.  In a pressing stroke the big
        # residual sits on the CONSTRAINED dofs -- it is the platen reaction,
        # the quantity the press load is read from -- and row elimination
        # removes it from the convergence test, so what is left converges to
        # 1e-8 easily.  On a release every one of those dofs is freed at once,
        # so the body has to redistribute the whole ~100 kN internally and all
        # of that residual enters the test.  Measured on hit 4 of roll3.2: the
        # iteration drives 7.56e4 N down to 2.17 N -- a relative 2.9e-5, i.e.
        # 0.003% out of balance -- and then stalls there, oscillating between
        # 1.6 and 2.5 rather than diverging.  Demanding 1e-8 of that solve is
        # what ends the run.  Nothing is measured on a release (its reaction
        # is 0.00 kN by definition); its only product is the state handed to
        # the next stroke, and 2 N of imbalance does not change it.
        self.abs_tol = abs_tol
        # Last-resort acceptance for a step that runs out of Newton iterations.
        #
        # The solver returns its last iterate whichever way it ends, and the
        # stepper used to discard it and kill the run.  On a release that is
        # the wrong trade: the iteration does not diverge, it STALLS -- hit 4
        # of roll3.2 drove 7.56e4 N down to 2.17 N, a relative 2.9e-5, then
        # oscillated between 1.6 and 2.5 N.  Throwing that away loses a
        # perfectly usable state over 0.003% of imbalance.
        #
        # So when this is set and the step did not formally converge, the last
        # iterate is accepted provided it is (a) finite everywhere and (b) has
        # moved the bar by no more than accept_motion_mm from the state the
        # step started in.  (a) rejects the NaN exit, which the solver reports
        # the same way as running out of iterations; (b) rejects a genuine
        # divergence, since a stalled release moves the bar by the springback,
        # measured at 0.02-0.12 mm, while a diverging one runs away.
        #
        # Set only for release strokes.  A compression stroke keeps raising,
        # because its reaction IS the press load being measured and a
        # half-solved one would be reported as data.
        self.accept_motion_mm = accept_motion_mm
        # The loose acceptance above also needs a bound on the residual the
        # stalled iterate was left with, in newtons over the FREE dofs: a
        # stalled release sits at a few newtons, a genuine failure at many
        # thousands, and "finite and did not move much" cannot tell a half-
        # solved state from a solved one on its own.  None keeps the old test.
        self.accept_res_n = accept_res_n
        self.n_accepted_loose = 0
        # Pre-commit acceptance of the converged trial state.
        #
        # Newton can converge, to 1e-8, on a displacement field in which an
        # element has folded: the quadrature points of a trilinear hexahedron
        # keep a positive Jacobian while a corner inverts, so nothing in the
        # residual objects.  Measured on three meshes of roll3.1: the corner
        # scaled Jacobian of the element at the band edge fell from 0.67 to
        # -0.07 over ten converged steps of 0.08 s, each committed, and the
        # distortion monitor -- which ran on the last converged state at the
        # START of the next step -- reported it one step after the state and
        # its internal variables had been committed.
        #
        # accept_fn(sol_u_trial, step, scale, dt) -> (ok, report) is called
        # on the converged trial state BEFORE anything is committed.  A
        # refused state is discarded and the step retried with a smaller dt,
        # like a Newton failure; after accept_max_refusals refusals of the
        # same step the run stops with the report of the last refusal
        # (DistortionStop, which the retry loop does not catch), so that a
        # geometry that collapses at every step size is reported as such
        # instead of being ground down to min_dt.
        self.accept_fn = accept_fn
        self.accept_max_refusals = int(accept_max_refusals)
        # ... and a budget over the whole stroke: measured on mesh A hit 6,
        # a refused step, halved, is accepted, the next step grows by 1.3
        # and is refused again, so the stroke creeps on in half steps with
        # the element one refusal from folding.  Past this many refusals in
        # one stroke the geometry is collapsing at every step size, which
        # is a contact or mesh problem, not a step-size one.
        self.accept_max_refusals_total = int(accept_max_refusals_total)
        self.n_refused = 0
        # "jax"    -- the package default, unpreconditioned bicgstab
        # "direct" -- scipy sparse LU, via use_direct_linear_solver above
        # "mumps"  -- PETSc/MUMPS LU on every solve, via use_mumps_linear_solver
        # "gpu-hybrid" -- preconditioned bicgstab on the GPU, MUMPS for failures
        self.linear_solver = linear_solver
        use_descent_line_search()
        if linear_solver == "direct":
            use_direct_linear_solver()
        elif linear_solver == "hybrid":
            use_hybrid_linear_solver()
        elif linear_solver == "mumps":
            use_mumps_linear_solver()
        elif linear_solver == "gpu-hybrid":
            use_gpu_hybrid_linear_solver()
        elif linear_solver != "jax":
            raise ValueError("linear_solver must be 'jax', 'direct', 'hybrid', 'mumps' or "
                             f"'gpu-hybrid', got {linear_solver!r}")

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

    def _free_residual_norm(self, sol_list):
        """|R| over the dofs that are not prescribed, at sol_list, in newtons.

        The problem still holds the parameters of the step being solved, so
        this is the out-of-balance force Newton was driving to zero; the
        Dirichlet rows carry the reactions and are left out.
        """
        r = onp.asarray(self.problem.compute_residual(list(sol_list))[0])
        fe = self.problem.fes[0]
        m = onp.zeros((fe.num_total_nodes, fe.vec), dtype=bool)
        for ni, vi in zip(fe.node_inds_list, fe.vec_inds_list):
            m[onp.asarray(ni), onp.asarray(vi)] = True
        return float(onp.linalg.norm(r[~m]))

    def _extrapolate_guess(self):
        """Linear extrapolation for [u, T] using the last step.

        The ratio is capped at 1.  Extrapolating by dt/dt_prev is only sound
        while the two steps are comparable; when dt grows the guess grows with
        it, and a guess that runs past the previous increment is not a warm
        start but a leap into territory the solver has no information about.
        Capping costs nothing when dt is steady -- the ratio is 1 anyway -- and
        it bounds the damage when dt jumps.  Together with adopting the
        retried step size in run(), this is what stops the 64x overshoot that
        killed hit 6 of roll3.
        """
        sol_n = self.sol_list
        sol_nm1 = self.sol_hist[-1]
        dt_nm1 = self.dt_hist[-1] + 1e-12
        ratio = min(1.0, self.dt / dt_nm1)

        guess = [
            sol_n[0] + (sol_n[0] - sol_nm1[0]) * ratio,
            sol_n[1] + (sol_n[1] - sol_nm1[1]) * ratio,
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
            refusals = 0

            sol_u_old = self.sol_list[0]
            sol_dT_old = self.sol_list[1]
            print("*****Get int_vars_old at outer_loop...")
            # The T_old slot of the stepper's own internal variables must hold
            # what set_params gives Newton.  It used to stay at its initial 0,
            # so update_int_vars_gp committed the return map at T = 0 (harmless
            # only while the tables are flat) and compute_cell_fields wrote
            # sig_xx/yy/zz offset by beta * T_old, about +1500 MPa.
            _iv = list(self.int_vars)
            _iv[-2] = self.problem.fe_T.convert_from_dof_to_quad(sol_dT_old)[..., 0]
            self.int_vars = tuple(_iv)
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

                # Keep the exact parameter set Newton is about to drive to zero.
                # After run() returns, problem.internal_vars has moved on (the
                # return map has been committed), so evaluating the residual at
                # the final solution then measures a state the solver never
                # equilibrated: it comes out large over the whole body instead
                # of living only on the constrained dofs, and any half-body sum
                # of it is meaningless.  Restoring these makes the post-run
                # residual the one Newton actually zeroed, so the reaction at
                # the platen can be read off it.
                self.last_params = [int_vars_old, scale, sol_u_old, rho_ini, sol_dT_old]

                # Solver options
                use_ls = (retries >= self.line_search_after)
                solver_options = {
                    "initial_guess": initial_guess,
                    "line_search_flag": use_ls,
                    'return_full_info': True
                }
                solver_options["umfpack_solver" if self.linear_solver in ("direct", "mumps")
                               else "jax_solver"] = {}
                if self.abs_tol is not None:
                    solver_options["tol"] = self.abs_tol

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
                        # symmetrise be_old for stable eigvalsh
                        be_sym = 0.5 * (be_old_arr + np.swapaxes(be_old_arr, -1, -2))
                        be_eigs = np.linalg.eigvalsh(be_sym)
                        be_eig_min = float(np.min(be_eigs))
                        be_eig_max = float(np.max(be_eigs))
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
                    new_sol_list, iteration_counter, has_converged = solver(self.problem, solver_options)

                    if not has_converged and self.accept_motion_mm is not None:
                        u_new = np.asarray(new_sol_list[0])
                        finite = bool(np.all(np.isfinite(u_new))) and \
                            bool(np.all(np.isfinite(np.asarray(new_sol_list[1]))))
                        moved = (float(np.max(np.abs(u_new - np.asarray(sol_u_old))))
                                 if finite else float("inf"))
                        res_free = float("inf")
                        if finite and self.accept_res_n is not None:
                            res_free = self._free_residual_norm(new_sol_list)
                        res_ok = (self.accept_res_n is None) or (res_free <= self.accept_res_n)
                        if finite and moved <= self.accept_motion_mm and res_ok:
                            print(f"  [accept] step {self.step_count} did not reach the "
                                  f"tolerance in {iteration_counter} iterations, but its "
                                  f"last iterate is finite, moved the bar {moved:.4f} mm "
                                  f"(limit {self.accept_motion_mm} mm) and is "
                                  f"{res_free:.3e} N out of balance on the free dofs "
                                  f"(limit {self.accept_res_n}) -- accepting it",
                                  flush=True)
                            self.n_accepted_loose += 1
                            has_converged = True
                        else:
                            print(f"  [accept] refused: finite={finite}, "
                                  f"moved={moved:.4f} mm vs limit {self.accept_motion_mm}, "
                                  f"free residual {res_free:.3e} N vs limit {self.accept_res_n}",
                                  flush=True)

                    if not has_converged:
                        raise RuntimeError("Solver failed to converge within max_iters")

                    # Pre-commit acceptance of the trial state (see __init__).
                    if self.accept_fn is not None:
                        ok, rep = self.accept_fn(new_sol_list[0], self.step_count,
                                                 scale, dt_this)
                        if not ok:
                            refusals += 1
                            self.n_refused += 1
                            print(f"  [accept] trial state of step {self.step_count} "
                                  f"(scale {scale:.4f}, dt {dt_this:.3e}) refused "
                                  f"({refusals}/{self.accept_max_refusals}): "
                                  f"{rep['message']} -- discarded", flush=True)
                            self.step_report.append((self.step_count, next_time, dt_this,
                                                     "REFUSED(quality)"))
                            if (refusals >= self.accept_max_refusals
                                    or self.n_refused >= self.accept_max_refusals_total):
                                rep = dict(rep, trial_sol_u=onp.asarray(new_sol_list[0]),
                                           refusals=refusals, refusals_total=self.n_refused,
                                           dt=float(dt_this), scale=float(scale))
                                rep["message"] += (f" (trial state refused {refusals} "
                                                   f"times at this step down to dt {dt_this:.3e}, "
                                                   f"{self.n_refused} times in this stroke)")
                                self.total_wall = time.time() - t0
                                raise DistortionStop(rep)
                            raise RuntimeError("trial state refused by the quality check")

                    self.sol_list = new_sol_list
                    self.sol_u, self.sol_dT = new_sol_list

                    # Reaction, taken here and nowhere else.  At this point the
                    # problem still holds exactly the parameters this Newton
                    # solve converged against, so the raw weak-form residual is
                    # zero at the free dofs and equals the constraint force at
                    # the prescribed ones -- which is what a load cell reads.
                    # After update_int_vars_gp below commits the return map the
                    # same call returns a residual spread over the whole body
                    # (5.9e4 N against a 1.4e5 N load), and any half-body or
                    # contact-set sum of that is meaningless: it varied by a
                    # factor of four with the roll angle while the displacement
                    # field did not move at all.
                    self.last_res_u = self.problem.compute_residual(
                        [self.sol_u, self.sol_dT])[0]
                    try:
                        fe = self.problem.fes[0]
                        r = onp.asarray(self.last_res_u)
                        m = onp.zeros((fe.num_total_nodes, fe.vec), dtype=bool)
                        for ni, vi in zip(fe.node_inds_list, fe.vec_inds_list):
                            m[onp.asarray(ni), onp.asarray(vi)] = True
                        print(f"  [RES] ||R||={onp.linalg.norm(r):.4e} | "
                              f"Dirichlet {int(m.sum())} dofs: {onp.linalg.norm(r[m]):.4e} | "
                              f"free {int((~m).sum())}: {onp.linalg.norm(r[~m]):.4e}", flush=True)
                        # Per-constraint-group reaction, resolved in the press
                        # frame.  The groups are, in order: the rigid-body set
                        # built by build_cylinder_press_bcs, then the lower and
                        # upper platen, each as three consecutive single-vec
                        # groups.  Splitting them is the only way to see which
                        # constraint is actually carrying the load.
                        Rm = getattr(self.problem, "press_rot", None)
                        if Rm is not None:
                            tot = 0.0
                            for g, (ni, vi) in enumerate(zip(fe.node_inds_list,
                                                             fe.vec_inds_list)):
                                ni = onp.asarray(ni); vi = onp.asarray(vi)
                                if len(ni) == 0:
                                    continue
                                f = onp.zeros((len(ni), 3))
                                f[onp.arange(len(ni)), vi] = r[ni, vi]
                                fy = float(onp.sum((Rm @ f.T).T[:, 1]) / 1e3)
                                tot += fy
                                print(f"        group {g:2d}: {len(ni):4d} nodes vec={vi[0]}"
                                      f"  Fy={fy:+9.3f} kN", flush=True)
                            print(f"        sum over all groups Fy={tot:+9.3f} kN", flush=True)
                    except Exception as _e:
                        print(f"  [RES] classification failed: {_e}", flush=True)

                    print("** Update internal variables after convergence")

                    # --- Debug: report temperature change after solver step ---
                    try:
                        dT_diff = np.linalg.norm(self.sol_dT - sol_dT_old)
                        print(f"  Debug: sol_dT min/max = {np.min(self.sol_dT):.6e}/{np.max(self.sol_dT):.6e}, ||ΔT||={dT_diff:.6e}")
                    except Exception as _e:
                        print("  Debug: failed to compute sol_dT stats", _e)

                    # Update internal variables after convergence
                    a1, a2, a3, a4, a5, a6, a7, a8, a9 = self.int_vars
                    int_vars_u = self.problem.update_int_vars_gp(self.sol_u, self.int_vars)
                    a1_updated, a2_updated, a3_updated, a4_updated, a5_updated, _ = int_vars_u
                    self.int_vars = (a1_updated, a2_updated, a3_updated, a4_updated, a5_updated,
                                     a6, a7, a8, a9)

                    # accept step
                    self.current_time = next_time
                    converged = True

                    # update history
                    self.sol_hist.pop(0)
                    self.sol_hist.append(self.sol_list)
                    self.dt_hist.pop(0)
                    self.dt_hist.append(dt_this)

                    # Carry forward the step size that actually worked.
                    #
                    # Retries used to shrink only the local dt_this and leave
                    # self.dt untouched, so the next step went straight back to
                    # a size that had just been shown to fail.  On hit 6 of
                    # roll3.2's release: step 1 needed six halvings and
                    # succeeded at dt = 3.906e-05, self.dt stayed at 2.500e-03,
                    # and step 2 re-attempted 64x that.  The extrapolated guess
                    # below divides by the previous step's REAL dt and
                    # multiplies by the new one, so the guess overshot by the
                    # same factor of 64 -- enough to invert elements.  Step 3
                    # then opened at a residual of 3.98e19 N and every retry
                    # went to NaN.  seed_state already carries a note about the
                    # same mechanism at a factor of 2; this is that bug again,
                    # with retries supplying the factor.
                    if retries > 0:
                        self.dt = dt_this
                    else:
                        self.dt = min(self.max_dt, self.dt * self.increase_factor)

                    self.step_report.append((self.step_count, self.current_time, dt_this, "OK"))

                    # Record step data for plotting
                    step_wall_time = time.time() - step_start
                    self.step_data.append((self.step_count, float(scale), float(step_wall_time),
                                           float(dt_this), int(iteration_counter), int(retries)))

                    # Optional VTK output -- deferred to AFTER the solver try/except.
                    # A failed write must never look like a failed solve: the state
                    # has already been advanced by this point, and a retry would
                    # re-solve the step on top of it.
                    self._pending_vtk = (vtk_dir is not None and (self.step_count % save_every == 0))
                    self._pending_int_vars_old = int_vars_old
                    self._pending_int_vars_u = int_vars_u

                except (RuntimeError, FloatingPointError, AssertionError) as e:
                    print(f"  Solver failed or NaN detected: {e}")
                    self.last_failure = str(e)
                    retries += 1
                    self.int_vars = int_vars_old
                    initial_guess = [sol_u_old, sol_dT_old]

            if converged and getattr(self, "_pending_vtk", False):
                try:
                    vtk_path = os.path.join(vtk_dir, f"tm_{self.step_count:04d}.vtu")
                    # stress from the PRE-commit internal variables Newton converged
                    # against; alpha from the freshly updated set
                    sig, vm = self.problem.compute_cell_fields(self.sol_u, self._pending_int_vars_old)
                    sig = onp.asarray(sig)
                    alpha_c = onp.asarray(np.mean(self._pending_int_vars_u[2], axis=1))
                    cell_infos = [("sig_xx", sig[:, 0, 0]), ("sig_yy", sig[:, 1, 1]),
                                  ("sig_zz", sig[:, 2, 2]), ("sig_xy", sig[:, 0, 1]),
                                  ("sig_yz", sig[:, 1, 2]), ("sig_xz", sig[:, 0, 2]),
                                  ("von_Mises", onp.asarray(vm)),
                                  ("eq_plastic_strain", alpha_c)]
                    save_sol(_fe_for_writing(self.problem.fes[0]), self.sol_u, vtk_path,
                             point_infos=[("Displacement", self.sol_u), ("Temperature", self.sol_dT)],
                             cell_infos=cell_infos)
                    self.last_cell_infos = cell_infos
                    print(f"  Saved: {vtk_path}")
                except Exception as e:
                    print(f"  [vtk] write skipped at step {self.step_count}: {e}")
                self._pending_vtk = False

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

        step_counts, scales, wall_times = zip(*[d[:3] for d in self.step_data])

        plt.figure(figsize=(10, 6))
        plt.scatter(step_counts, scales, s=50, alpha=0.7)

        for step_count, scale, wall_time, *_ in self.step_data:
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
