"""Newton solver on the device-resident system (drop-in for solver_latest.solver).

Same algorithm as ``jax_fem_checkpoint.solver_latest.solver`` with the row
elimination of Dirichlet dofs:

* stop when ``|r| / |r_0| <= rel_tol`` or ``|r| <= tol`` (defaults 1e-8 / 1e-6),
  at most ``max_iters`` (50) iterations, abort on NaN / Inf;
* the increment solves ``A dx = -r`` with ``x0`` = the Dirichlet increment;
* every linear solve is checked: if ``|A dx + r| > 1e-6 max(|r|, 1)`` it is
  redone with a direct solver (as the original package's hybrid wrapper);
* the line search is the original one (alpha = 1, then halved up to three
  times while the residual norm decreases).

Differences that do not change the iterates: the residual and the Jacobian
come from ``FastSystem`` (GPU assembly, fixed pattern); the residual at the
accepted line-search point is reused for the convergence test instead of
being recomputed; the Jacobian of the last (converged) iterate is not formed.
"""
import time
from collections import defaultdict

import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util

from applications.Agility_Forge_physforge_full_fast.lib.fast_system import FastSystem
from applications.Agility_Forge_physforge_full_fast.lib import linear_solvers as ls

CONFIG = {
    "newton": "fast",              # 'fast' | 'original' (jax_fem_checkpoint's solver)
    "linear_solver": "auto",       # see linear_solvers.make_solver
    "second_solver": "amgxbe:gs",  # AMGX set-up tried when the main AMGX solve fails the check ('none': off)
    "chunk": None,                 # cells per kernel batch (None: all at once)
    "check_rel_tol": 1e-6,         # hybrid check of every linear solve
    "profile": False,              # block_until_ready around each part (timing only)
    "verbose": False,
}
STATS = defaultdict(lambda: [0, 0.0])
COUNTS = defaultdict(int)
MAIN_ITERS = []                    # iterations of the main linear solver, per solve
T_START = time.time()


def _t(name, t0):
    s = STATS[name]
    s[0] += 1
    s[1] += time.perf_counter() - t0


def _sync(x):
    if CONFIG["profile"]:
        jax.block_until_ready(x)
    return x


def get_system(problem):
    sysm = getattr(problem, "_fast_system", None)
    if sysm is None:
        sysm = FastSystem(problem, chunk=CONFIG["chunk"])
        problem._fast_system = sysm
    return sysm


def _get_linsolvers(sysm):
    key = CONFIG["linear_solver"]
    cache = getattr(sysm, "_linsolvers", None)
    if cache is None or cache[0] != key:
        sysm._linsolvers = (key, ls.make_solver(key, sysm), None)
    k, main, direct = sysm._linsolvers
    return main, sysm


def _second_amgx(sysm, main):
    """CONFIG['second_solver'] after a failed AMGX solve ('none': go on to
    BiCGSTAB and the direct solver as before)."""
    name = CONFIG.get("second_solver", "none")
    if name in (None, "none") or not main.name.startswith("amgx") or main.name == name.replace(":", "_"):
        return None
    s2 = getattr(sysm, "_second_amgx", None)
    if s2 is None or s2[0] != name:
        try:
            sysm._second_amgx = (name, ls.make_solver(name, sysm))
        except Exception as e:                              # noqa: BLE001
            print(f"  [fast] second solver {name} unavailable ({type(e).__name__}: {e})", flush=True)
            sysm._second_amgx = (name, None)
        s2 = sysm._second_amgx
    return s2[1]


def _direct(sysm):
    k, main, direct = sysm._linsolvers
    if direct is None:
        direct = ls.available_direct(sysm)
        sysm._linsolvers = (k, main, direct)
    return direct


def _save_pattern(sysm, d):
    import os
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "pattern.npz")
    if not os.path.exists(f):
        onp.savez(f, indices=sysm.indices_host, indptr=sysm.indptr_host, N=sysm.N)


def _maybe_dump(sysm, data, b, x0):
    """AGF_DUMP_DIR: save every AGF_DUMP_EVERY-th linear system for offline
    solver benchmarks, at most AGF_DUMP_MAX of them (values, right-hand side,
    start vector and Dirichlet mask; the CSR pattern once in pattern.npz)."""
    import os
    d = os.environ.get("AGF_DUMP_DIR")
    if not d:
        return
    k = COUNTS["linear_solves"]
    if k % int(os.environ.get("AGF_DUMP_EVERY", "50")) or COUNTS["dumped"] >= int(os.environ.get("AGF_DUMP_MAX", "10")):
        return
    _save_pattern(sysm, d)
    onp.savez(os.path.join(d, f"sys_{k:05d}.npz"), data=onp.asarray(data), b=onp.asarray(b),
              x0=onp.asarray(x0), bc_mask=onp.asarray(sysm._bc[0]), newton_solve=COUNTS["newton_solves"])
    COUNTS["dumped"] += 1


def _dump_failed(sysm, data, b, x0, x_main, info, x_ok, how):
    """AGF_DUMP_FAIL_DIR: save the systems on which the main solver failed the
    check (at most AGF_DUMP_FAIL_MAX), with the accepted solution; stop the
    run after AGF_STOP_AFTER_FAIL_DUMPS of them (benchmark runs)."""
    import os
    d = os.environ.get("AGF_DUMP_FAIL_DIR")
    if not d or COUNTS["dumped_failed"] >= int(os.environ.get("AGF_DUMP_FAIL_MAX", "10")):
        return
    _save_pattern(sysm, d)
    k = COUNTS["linear_solves"]
    onp.savez(os.path.join(d, f"fail_{k:05d}.npz"), data=onp.asarray(data), b=onp.asarray(b),
              x0=onp.asarray(x0), bc_mask=onp.asarray(sysm._bc[0]), x_main=onp.asarray(x_main),
              x_ok=onp.asarray(x_ok), how=how, main_iters=int(info.get("iters", -1)),
              newton_solve=COUNTS["newton_solves"])
    COUNTS["dumped_failed"] += 1
    stop = int(os.environ.get("AGF_STOP_AFTER_FAIL_DUMPS", "0"))
    if stop and COUNTS["dumped_failed"] >= stop:
        print(f"  [fast] AGF_STOP_AFTER_FAIL_DUMPS={stop} reached, stopping", flush=True)
        raise SystemExit(0)


def linear_solve(sysm, data, b, x0):
    main, _ = _get_linsolvers(sysm)
    _maybe_dump(sysm, data, b, x0)
    t0 = time.perf_counter()
    x, info = main(data, b, x0)
    _sync(x)
    _t("linear solve (" + main.name + ")", t0)
    if "iters" in info:
        COUNTS["krylov_iters"] += max(int(info["iters"]), 0)
        MAIN_ITERS.append(int(info["iters"]))
    COUNTS["linear_solves"] += 1
    t0 = time.perf_counter()
    r = ls.csr_matvec(data, sysm.indices, sysm.row_of, x, sysm.ndofs) - b
    res, nb = float(np.linalg.norm(r)), float(np.linalg.norm(b))
    _t("solve check |Ax-b|", t0)
    if onp.isfinite(res) and res <= CONFIG["check_rel_tol"] * max(nb, 1.0):
        return x
    # second try with the stronger AMGX set-up (main solver AMGX only)
    s2 = _second_amgx(sysm, main)
    if s2 is not None:
        t0 = time.perf_counter()
        xs, infos = s2(data, b, x0)
        _sync(xs)
        _t("fallback " + s2.name, t0)
        rs = float(np.linalg.norm(ls.csr_matvec(data, sysm.indices, sysm.row_of, xs, sysm.ndofs) - b))
        COUNTS["fallbacks_amgx2"] += 1
        if onp.isfinite(rs) and rs <= CONFIG["check_rel_tol"] * max(nb, 1.0):
            print(f"  [fast] {main.name} residual {res:.3e} > {CONFIG['check_rel_tol']:.0e} * |b|={nb:.3e} "
                  f"(info {info}); {s2.name} residual {rs:.3e} ({infos})", flush=True)
            _dump_failed(sysm, data, b, x0, x, info, xs, s2.name)
            return xs
    # then the original BiCGSTAB on the GPU (cheap at every size), then
    # a direct solver (exact, but its cost grows fastest with the mesh)
    if main.name != "bicgstab":
        k2 = getattr(sysm, "_second_solver", None)
        if k2 is None:
            k2 = sysm._second_solver = ls.make_solver("bicgstab", sysm)
        t0 = time.perf_counter()
        x2, info2 = k2(data, b, x0)
        _sync(x2)
        _t("fallback bicgstab", t0)
        r2 = float(np.linalg.norm(ls.csr_matvec(data, sysm.indices, sysm.row_of, x2, sysm.ndofs) - b))
        COUNTS["fallbacks_bicgstab"] += 1
        if onp.isfinite(r2) and r2 <= CONFIG["check_rel_tol"] * max(nb, 1.0):
            print(f"  [fast] {main.name} residual {res:.3e} > {CONFIG['check_rel_tol']:.0e} * |b|={nb:.3e} "
                  f"(info {info}); bicgstab residual {r2:.3e} ({info2})", flush=True)
            _dump_failed(sysm, data, b, x0, x, info, x2, "bicgstab")
            return x2
    COUNTS["fallbacks"] += 1
    t0 = time.perf_counter()
    d = _direct(sysm)
    xd, _ = d(data, b, x0)
    _sync(xd)
    _t("fallback direct (" + d.name + ")", t0)
    rd = float(np.linalg.norm(ls.csr_matvec(data, sysm.indices, sysm.row_of, xd, sysm.ndofs) - b))
    print(f"  [fast] {main.name} residual {res:.3e} > {CONFIG['check_rel_tol']:.0e} * |b|={nb:.3e} "
          f"(info {info}); direct ({d.name}) residual {rd:.3e}", flush=True)
    _dump_failed(sysm, data, b, x0, x, info, xd, d.name)
    return xd


def solve(problem, solver_options={}):
    """Newton solve; returns (sol_list, iterations, converged) like
    ``solver_latest.solver(..., return_full_info=True)``."""
    t_all = time.perf_counter()
    sysm = get_system(problem)
    t0 = time.perf_counter()
    bc_mask, bc_target = sysm.prepare()
    _t("prepare (BC masks, args)", t0)

    if "initial_guess" in solver_options:
        guess = jax.lax.stop_gradient(solver_options["initial_guess"])
        dofs = jax.flatten_util.ravel_pytree(guess)[0]
    else:
        dofs = np.zeros(sysm.ndofs)
    rel_tol = solver_options.get("rel_tol", 1e-8)
    tol = solver_options.get("tol", 1e-6)
    max_iters = solver_options.get("max_iters", 50)
    use_ls = solver_options.get("line_search_flag", False)

    t0 = time.perf_counter()
    res, data = sysm.residual_jacobian(dofs)
    _sync(data)
    _t("residual + Jacobian", t0)
    if COUNTS["verified"] < CONFIG.get("verify", 0):
        verify_against_original(problem, sysm, dofs, res, data)
    res_val = float(np.linalg.norm(res))
    res_val_initial = max(res_val, 1.e-8)
    rel_res_val = res_val / res_val_initial

    it = 0
    nan_detected = False
    while (rel_res_val > rel_tol and res_val > tol) and it < max_iters:
        b = -res
        x0 = np.where(bc_mask, bc_target - dofs, 0.0)
        inc = linear_solve(sysm, data, b, x0)
        t0 = time.perf_counter()
        if use_ls:
            dofs, res = _line_search(sysm, dofs, inc)
        else:
            dofs = dofs + inc
            res = sysm.residual(dofs)
        _t("line search / update", t0)
        res_val = float(np.linalg.norm(res))
        rel_res_val = res_val / res_val_initial
        if rel_res_val > rel_tol and res_val > tol:
            t0 = time.perf_counter()
            res, data = sysm.residual_jacobian(dofs)
            _sync(data)
            _t("residual + Jacobian", t0)
        it += 1
        if CONFIG["verbose"]:
            print(f"  [fast] iteration {it}: |r| = {res_val:.4e}, rel {rel_res_val:.4e}", flush=True)
        if not (onp.isfinite(res_val) and bool(np.all(np.isfinite(dofs)))):
            nan_detected = True
            print(f"  [fast] NaN or Inf at iteration {it}; aborting this attempt", flush=True)
            break

    converged = (not nan_detected) and (rel_res_val <= rel_tol or res_val <= tol)
    COUNTS["newton_solves"] += 1
    COUNTS["newton_iters"] += it
    if not converged:
        COUNTS["newton_failed"] += 1
    sol_list = problem.unflatten_fn_sol_list(dofs)
    _t("Newton solve (all)", t_all)
    print(f"  [fast] Newton {'converged' if converged else 'NOT converged'} in {it} iterations, "
          f"|r| {res_val:.3e} (rel {rel_res_val:.3e}), {time.perf_counter() - t_all:.2f} s", flush=True)
    import os
    if os.environ.get("AGF_MEMSTATS"):                      # diagnostics: GPU memory after every solve
        ms = jax.devices()[0].memory_stats() or {}
        live = jax.live_arrays()
        print(f"  [fast mem] in use {ms.get('bytes_in_use', 0) / 2**30:.2f} GiB, peak "
              f"{ms.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB, live arrays {len(live)} "
              f"{sum(x.nbytes for x in live) / 2**30:.2f} GiB, fn builds {getattr(sysm, '_n_builds', 0)}", flush=True)
    stop = int(os.environ.get("AGF_MAX_SOLVES", "0"))
    if stop and COUNTS["newton_solves"] >= stop:          # benchmark runs: stop after N solves
        print(f"  [fast] AGF_MAX_SOLVES={stop} reached, stopping", flush=True)
        raise SystemExit(0)
    wall = float(os.environ.get("AGF_STOP_AFTER_S", "0"))
    if wall and time.time() - T_START >= wall:             # benchmark runs: stop after a wall time
        print(f"  [fast] AGF_STOP_AFTER_S={wall:.0f} reached, stopping", flush=True)
        raise SystemExit(0)
    return sol_list, it, converged


def _line_search(sysm, dofs, inc):
    """The original line search; also returns the residual at the accepted point."""
    def r_at(alpha):
        r = sysm.residual(dofs + alpha * inc)
        return r, float(np.linalg.norm(r))

    alpha = 1.
    r_acc, n_acc = r_at(alpha)
    for _ in range(3):
        alpha *= 0.5
        r_half, n_half = r_at(alpha)
        if n_half > n_acc:
            alpha *= 2.
            break
        r_acc, n_acc = r_half, n_half
    return dofs + alpha * inc, r_acc


def verify_against_original(problem, sysm, dofs, res, data):
    """Compare the fast residual and Jacobian with jax_fem_checkpoint's path
    (problem.newton_update + apply_bc_vec + get_A, i.e. scipy/PETSc assembly
    and zeroRows) at the same state; prints the largest differences."""
    import scipy.sparse as sp
    import jax_fem_checkpoint.solver_latest as sl
    COUNTS["verified"] += 1
    t0 = time.perf_counter()
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_o = jax.flatten_util.ravel_pytree(problem.newton_update(sol_list))[0]
    res_o = onp.asarray(sl.apply_bc_vec(res_o, dofs, problem))
    A_o = sl.get_A(problem, {})
    idx = onp.asarray(A_o.indices)
    A_o = sp.csr_matrix((onp.asarray(A_o.data), (idx[:, 0], idx[:, 1])), shape=A_o.shape)
    A_f = sp.csr_matrix((onp.asarray(data), sysm.indices_host, sysm.indptr_host), shape=A_o.shape)
    dA = (A_o - A_f).tocoo()
    r_f = onp.asarray(res)
    amax = abs(A_o).max()
    P_f = sp.csr_matrix((onp.ones(len(sysm.indices_host)), sysm.indices_host, sysm.indptr_host),
                        shape=A_o.shape)
    out_of_pattern = int(((A_o != 0).astype(float) - P_f).maximum(0).count_nonzero())
    print(f"  [fast verify] |r_fast - r_orig|_max = {onp.max(onp.abs(r_f - res_o)):.3e} "
          f"(|r|_max {onp.max(onp.abs(res_o)):.3e}); |A_fast - A_orig|_max = "
          f"{(abs(dA.data).max() if dA.nnz else 0.0):.3e} (|A|_max {amax:.3e}); "
          f"orig nonzeros outside the fast pattern: {out_of_pattern}; "
          f"nnz orig {A_o.nnz} fast {A_f.nnz}; {time.perf_counter() - t0:.1f} s", flush=True)


def report(title="fast solver"):
    lines = [f"[fast] ---- {title} ----"]
    for k, (n, s) in sorted(STATS.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"[fast] {k:34s} {n:8d} calls {s:10.2f} s {1e3 * s / max(n, 1):10.2f} ms/call")
    lines.append("[fast] counts: " + ", ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
    print("\n".join(lines), flush=True)
    return {"stats": {k: {"calls": n, "seconds": s} for k, (n, s) in STATS.items()},
            "counts": dict(COUNTS), "main_iters": list(MAIN_ITERS)}
