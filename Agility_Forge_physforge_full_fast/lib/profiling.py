"""Wall-clock profiling of the solve path (opt-in, numerics unchanged).

Enabled with the environment variable ``AGF_PROFILE=1``.  Every timed call is
followed by ``jax.block_until_ready`` on its result, so the time of an
asynchronous GPU computation is charged to the call that launched it.  The
table is printed at exit and written to ``AGF_PROFILE_OUT`` (JSON) if set.

``AGF_DUMP_DIR=<dir>`` additionally saves every ``AGF_DUMP_EVERY``-th (default
50) linear system of the Newton iterations (matrix, right-hand side, initial
guess) for offline linear-solver benchmarks, at most ``AGF_DUMP_MAX`` (20).
"""
import atexit
import json
import os
import time
from collections import defaultdict

import numpy as onp
import jax

STATS = defaultdict(lambda: [0, 0.0])      # name -> [calls, seconds]
_T0 = time.time()
_INSTALLED = False
_DUMP = {"n_solves": 0, "n_dumped": 0}


def _block(out):
    try:
        leaves = jax.tree_util.tree_leaves(out)
    except Exception:                                   # noqa: BLE001
        return
    for x in leaves:
        if hasattr(x, "block_until_ready"):
            try:
                x.block_until_ready()
            except Exception:                           # noqa: BLE001
                pass
        elif hasattr(x, "data") and hasattr(x.data, "block_until_ready"):
            x.data.block_until_ready()


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        out = fn(*a, **k)
        _block(out)
        s = STATS[name]
        s[0] += 1
        s[1] += time.perf_counter() - t
        return out
    wrapper.__wrapped__ = fn
    return wrapper


def report():
    wall = time.time() - _T0
    rows = sorted(STATS.items(), key=lambda kv: -kv[1][1])
    lines = [f"[profile] wall since import {wall:.1f} s",
             f"[profile] {'name':38s} {'calls':>8s} {'total s':>10s} {'mean ms':>10s}"]
    for k, (n, s) in rows:
        lines.append(f"[profile] {k:38s} {n:8d} {s:10.2f} {1e3 * s / max(n, 1):10.2f}")
    print("\n".join(lines), flush=True)
    out = os.environ.get("AGF_PROFILE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump({"wall_s": wall, "stats": {k: {"calls": n, "seconds": s} for k, (n, s) in rows}},
                      f, indent=1)


def _dump_system(A, b, x0):
    d = os.environ.get("AGF_DUMP_DIR")
    if not d:
        return
    every = int(os.environ.get("AGF_DUMP_EVERY", "50"))
    nmax = int(os.environ.get("AGF_DUMP_MAX", "20"))
    k = _DUMP["n_solves"]
    _DUMP["n_solves"] += 1
    if k % every or _DUMP["n_dumped"] >= nmax:
        return
    os.makedirs(d, exist_ok=True)
    onp.savez(os.path.join(d, f"sys_{k:05d}.npz"),
              data=onp.asarray(A.data), indices=onp.asarray(A.indices),
              shape=onp.asarray(A.shape), b=onp.asarray(b), x0=onp.asarray(x0))
    _DUMP["n_dumped"] += 1


def maybe_install():
    """Install the timers when AGF_PROFILE=1 (idempotent)."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("AGF_PROFILE", "0") != "1":
        return
    _INSTALLED = True
    import jax_fem_checkpoint.solver_latest as sl
    from jax_fem_checkpoint.problem import Problem
    from applications.Agility_Forge_physforge_full_fast.lib import time_stepper as ts
    from applications.Agility_Forge_physforge_full_fast.lib.constitutive import ThermalMechanical

    # element kernels: residual-only vs residual + Jacobian
    _split = Problem.split_and_compute_cell

    def split(self, cells_sol_flat, np_version, jac_flag, internal_vars):
        name = "cell kernel (res+jac)" if jac_flag else "cell kernel (res)"
        return timed(name, _split)(self, cells_sol_flat, np_version, jac_flag, internal_vars)
    Problem.split_and_compute_cell = split

    _face = Problem.compute_face

    def face(self, cells_sol_flat, np_version, jac_flag, ivs):
        name = "face kernels (res+jac)" if jac_flag else "face kernels (res)"
        return timed(name, _face)(self, cells_sol_flat, np_version, jac_flag, ivs)
    Problem.compute_face = face

    Problem.compute_residual_vars_helper = timed("residual scatter", Problem.compute_residual_vars_helper)
    Problem.compute_residual_vars = timed("compute_residual_vars (all)", Problem.compute_residual_vars)
    Problem.compute_newton_vars = timed("compute_newton_vars (all)", Problem.compute_newton_vars)

    sl.get_A = timed("get_A (CPU assembly + zeroRows)", sl.get_A)
    sl.apply_bc_vec = timed("apply_bc_vec", sl.apply_bc_vec)
    sl.line_search = timed("line_search (all)", sl.line_search)

    _jax_solve = sl.jax_solve

    def jax_solve(A, b, x0, precond):
        _dump_system(A, b, x0)
        return timed("bicgstab (jax_solve)", _jax_solve)(A, b, x0, precond)
    sl.jax_solve = jax_solve

    # the hybrid check wraps sl.linear_solver when the stepper is built; time
    # the outer function then (installed below via the stepper's __init__)
    _init = ts.AutomaticTimeStepperTM.__init__

    def init(self, *a, **k):
        _init(self, *a, **k)
        if not getattr(sl.linear_solver, "_agf_timed", False):
            f = timed("linear_solver (bicgstab + check)", sl.linear_solver)
            f._agf_timed = True
            sl.linear_solver = f
    ts.AutomaticTimeStepperTM.__init__ = init

    ts.solver = timed("Newton solve (solver_latest.solver)", ts.solver)
    ts.save_sol = timed("save_sol (VTK)", ts.save_sol)

    _run = ts.AutomaticTimeStepperTM.run

    def run(self, bc_update_fn=None, bc_params_fn=None, **k):
        if bc_update_fn is not None:
            bc_update_fn = timed("bc_update_fn (contact)", bc_update_fn)
        return timed("stepper.run (all)", _run)(self, bc_update_fn=bc_update_fn,
                                                 bc_params_fn=bc_params_fn, **k)
    ts.AutomaticTimeStepperTM.run = run

    ThermalMechanical.set_params = timed("set_params", ThermalMechanical.set_params)
    ThermalMechanical.update_int_vars_gp = timed("update_int_vars_gp", ThermalMechanical.update_int_vars_gp)
    ThermalMechanical.internal_force = timed("internal_force", ThermalMechanical.internal_force)

    atexit.register(report)
    print("[profile] timers installed", flush=True)
