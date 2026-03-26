from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
from jax_fem_checkpoint import logger

from jax.experimental.sparse import BCOO
# import pyamgx

try:
    import pyamgx
    PYAMGX_AVAILABLE = True
except ImportError:
    PYAMGX_AVAILABLE = False
    logger.info("pyamgx not installed. AMGX solver disabled.")

import scipy
import time
from jax.sharding import SingleDeviceSharding
from jax import checkpoint
from jax.ad_checkpoint import checkpoint_name
from functools import partial
# from petsc4py import PETSc

import scipy.sparse
# import dask
# from dask import delayed
from functools import reduce
import operator
# from dask.distributed import Client, LocalCluster


# Try importing pardis0, CuPy and related sparse solvers
try:
    import pypardiso
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import spsolve
except ImportError:
    pass

petsc_row_elim = True  # If True, use PETSc for row elimination, otherwise use JAX

################################################################################
# JAX solver or scipy solver or PETSc solver

# This can let us use persistent bicgstab but requires transferring data between CPU and GPU
# def jax_solve(A, b, x0, precond):
#     result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
#     return jax.pure_callback(jax_solve_host, result_shape, A, b, x0, precond, vmap_method='sequential')

def AMGX_solve_host_gpu(A_cpu, b, x0):

    x_guess = onp.array(x0)
    b = onp.array(b)
    # setup AmgX solver
    # Initialize PyAMGX
    pyamgx.initialize()
    # Create resources
    cfg = pyamgx.Config().create_from_dict({
         "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",  # "CG", BICGSTAB
            "use_scalar_norm": 1,
            "norm": "L2",
            "tolerance": 1e-10,
            "monitor_residual": 1,
            "max_iters": 10000,
            "convergence": "ABSOLUTE",  # RELATIVE_INI_CORE
            "monitor_residual": 1,
            # "print_solve_stats": 1,
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "smoother": "JACOBI",
                "cycle": "V",
                "max_levels": 10,
                "max_iters": 2
            }
        }
    })

    resources = pyamgx.Resources().create_simple(cfg)

    solver = pyamgx.Solver().create(resources, cfg)
    # Create matrix and vector objects
    A_amg = pyamgx.Matrix().create(resources)
    b_amg = pyamgx.Vector().create(resources)
    x_amg = pyamgx.Vector().create(resources)
    # ======
    # Upload data to PyAMGX objects
    # A_amg.upload(onp.array(A.indices[:, 0]), onp.array(A.indices[:, 1]), onp.array(A.data))
    A_amg.upload_CSR(A_cpu)
    b_amg.upload(b)
    x_amg.upload(x_guess)

    # logger.debug(f"Setting up the AMGx solver...")
    solver.setup(A_amg)
    solver.solve(b_amg, x_amg)

    # Download the result
    result = x_amg.download()

    # Cleanup
    x_amg.destroy()
    b_amg.destroy()
    A_amg.destroy()
    solver.destroy()
    cfg.destroy()
    resources.destroy()
    # Finalize PyAMGX
    pyamgx.finalize()
    # logger.info(f'AMGX Solver - Finished solving, linear solve res = {np.linalg.norm(A @ result - b)}')
    return result
# "AMGX" solver
def AMGX_solve_host(A, x, b):
    dtype, shape = b.dtype, b.shape
    # indices = A.indices
    # logger.debug(f"Coversion to csr matrix inside solver wrapper...")
    start_sdv = time.time()
    A = scipy.sparse.csr_matrix((A.data, (A.indices[:, 0], A.indices[:, 1])), shape=A.shape)
    b = onp.array(b)
    x_guess = onp.array(x)
    end_sdv2 = time.time()
    solve_time_sdv2 = end_sdv2 - start_sdv
    # logger.info(f" Coversion time csr matrix inside solver wrapper: {solve_time_sdv2} [s]")
    # setup AmgX solver
    # Initialize PyAMGX
    pyamgx.initialize()
    # Create resources
    cfg = pyamgx.Config().create_from_dict({
         "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",  # "CG", BICGSTAB
            "use_scalar_norm": 1,
            "norm": "L2",
            "tolerance": 1e-10,
            "monitor_residual": 1,
            "max_iters": 10000,
            "convergence": "ABSOLUTE",  # RELATIVE_INI_CORE
            "monitor_residual": 1,
            # "print_solve_stats": 1,
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "smoother": "JACOBI",
                "cycle": "V",
                "max_levels": 10,
                "max_iters": 2
            }
        }
    })

    resources = pyamgx.Resources().create_simple(cfg)

    solver = pyamgx.Solver().create(resources, cfg)
    # Create matrix and vector objects
    A_amg = pyamgx.Matrix().create(resources)
    b_amg = pyamgx.Vector().create(resources)
    x_amg = pyamgx.Vector().create(resources)
    # ======
    # Upload data to PyAMGX objects
    # A_amg.upload(onp.array(A.indices[:, 0]), onp.array(A.indices[:, 1]), onp.array(A.data))
    A_amg.upload_CSR(A)
    b_amg.upload(b)
    x_amg.upload(x_guess)

    # Solve Ax = b
    # logger.debug(f"Setting up the AMGx solver...")
    start_sdv = time.time()
    solver.setup(A_amg)
    end_sdv = time.time()
    solve_time_sdv = end_sdv - start_sdv
    solver.solve(b_amg, x_amg)
    end_sdv2 = time.time()
    solve_time_sdv2 = end_sdv2 - end_sdv
    # logger.info(f" AmgX solver Setting up time: {solve_time_sdv} [s]")
    # logger.info(f" AmgX solver solve time: {solve_time_sdv2} [s]")
    # logger.info(f" AmgX solver set+solve time: {solve_time_sdv + solve_time_sdv2} [s]")

    # final_residual = solver.get_residual()
    # logger.info(f" Final residual: : {final_residual} [s]")
    # Download the result
    result = x_amg.download()
    # Cleanup
    x_amg.destroy()
    b_amg.destroy()
    A_amg.destroy()
    solver.destroy()
    cfg.destroy()
    resources.destroy()
    # Finalize PyAMGX
    pyamgx.finalize()
    logger.info(f'AMGX Solver - Finished solving, linear solve res = {np.linalg.norm(A @ result - b)}')
    return result.astype(dtype).reshape(shape)

# def AMGX_solve(A_sp, b, x0):
#     b_active = b
#     x0_active = x0
#     logger.debug(f"Solving linear system using AMG solver...")
#     start_dom = time.time()
#
#     def matvec(u):
#         Au = A_sp @ u
#         return Au
#
#     result_shape = jax.ShapeDtypeStruct(b_active.shape, b_active.dtype)
#     cust_solver = lambda matvec, b_vec: jax.pure_callback(AMGX_solve_host, result_shape, A_sp, x0_active, b_vec)
#     x_active = jax.lax.custom_linear_solve(matvec, b_active, cust_solver, symmetric=False)
#     x =x_active # x.at[active_dof].set(x_active)
#     end_dom = time.time()
#     solve_time_dom = end_dom - start_dom
#     logger.info(f" AMG solver overall solution time: {solve_time_dom} [s]")
#     return x

def AMGX_solve(A, b, x0):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(AMGX_solve_host, result_shape, A,x0,b) # jax.pure_callback
def pardiso_solve(A, b, x0, solver_options):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(pardiso_solve_host, result_shape, A, b, x0, solver_options) #, vmap_method='sequential'

def umfpack_solve1(A, b):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(umfpack_solve_host, result_shape, A, b) #, vmap_method='sequential'
def umfpack_solve(A, b):
    logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    # indptr, indices, data = A.getValuesCSR()
    # Asp = A #scipy.sparse.csr_matrix((data, indices, indptr))
    # x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    b = onp.array(b)
    # print(f"Matrix b: {b}")
    x_guess = onp.zeros_like(b)
    # logger.info(f" Coversion time csr matrix inside solver wrapper: {solve_time_sdv2} [s]")
    # setup AmgX solver
    # Initialize PyAMGX
    pyamgx.initialize()
    # Create resources
    cfg = pyamgx.Config().create_from_dict({
         "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",  # "CG", BICGSTAB
            "use_scalar_norm": 1,
            "norm": "L2",
            "tolerance": 1e-10,
            "monitor_residual": 1,
            "max_iters": 10000,
            "convergence": "ABSOLUTE",  # RELATIVE_INI_CORE
            "monitor_residual": 1,
            # "print_solve_stats": 1,
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "smoother": "JACOBI",
                "cycle": "V",
                "max_levels": 10,
                "max_iters": 2
            }
        }
    })

    resources = pyamgx.Resources().create_simple(cfg)

    solver = pyamgx.Solver().create(resources, cfg)
    # Create matrix and vector objects
    A_amg = pyamgx.Matrix().create(resources)
    b_amg = pyamgx.Vector().create(resources)
    x_amg = pyamgx.Vector().create(resources)
    # ======
    # Upload data to PyAMGX objects
    # A_amg.upload(onp.array(A.indices[:, 0]), onp.array(A.indices[:, 1]), onp.array(A.data))
    A_amg.upload_CSR(A)
    b_amg.upload(b)
    x_amg.upload(x_guess)

    # Solve Ax = b
    # logger.debug(f"Setting up the AMGx solver...")
    solver.setup(A_amg)
    solver.solve(b_amg, x_amg)

    # logger.info(f" AmgX solver Setting up time: {solve_time_sdv} [s]")
    # logger.info(f" AmgX solver solve time: {solve_time_sdv2} [s]")
    # logger.info(f" AmgX solver set+solve time: {solve_time_sdv + solve_time_sdv2} [s]")

    # final_residual = solver.get_residual()
    # logger.info(f" Final residual: : {final_residual} [s]")
    # Download the result
    result = x_amg.download()
    # print(f"Result: {result.astype(dtype).reshape(shape)}")

    # Cleanup
    x_amg.destroy()
    b_amg.destroy()
    A_amg.destroy()
    solver.destroy()
    cfg.destroy()
    resources.destroy()
    # Finalize PyAMGX
    pyamgx.finalize()
    x = result
    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug(f'Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(A @ x - b)}')
    return x

def cupy_solve(A, b):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(cupy_solve_host, result_shape, A, b) #, vmap_method='sequential'

def petsc_solve(A, b, ksp_type, pc_type):
    # Note that for jacrev compatitibility, we hard code ksptype and pc_type inside the host function
    # Only functions with JAX-type arguments can be transformed by pure_callback for batch-tracing
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(petsc_solve_host, result_shape, A, b)

# def jax_solve_host(A, b, x0, precond):
def jax_solve(A, b, x0, precond):
    """Solves the equilibrium equation using a JAX solver.
    Is fully traceable and runs on GPU.

    Parameters
    ----------
    precond
        Whether to calculate the preconditioner or not
    """
    logger.debug(f"JAX Solver - Solving linear system")
    # indptr, indices, data = A.getValuesCSR()
    # A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    # A = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    # jacobi = np.array(A_sp_scipy.diagonal())

    # move things to gpu
    # A = jax.device_put(A, jax.devices('gpu')[0])
    # b = jax.device_put(b, jax.devices('gpu')[0])
    # x0 = jax.device_put(x0, jax.devices('gpu')[0])

    # diagonal_mask = A.indices[:,0] == A.indices[:,1]
    # jacobi = np.zeros(A.shape[0])
    # jacobi = jacobi.at[A.indices[diagonal_mask,0]].add(A.data[diagonal_mask])
    # diag = np.arange(A.shape[0])
    # jacobi = A[diag,diag]
    # jacobi = jacobi.todense()
    # pc = lambda x: x * (1. / jacobi) if precond else None
    pc =None
    # while True: # workaround for if bicgstab does not converge at first
    rel_tol = 1e-10
    abs_tol = 1e-10
    x, info = jax.scipy.sparse.linalg.bicgstab(A,
                                            b,
                                            x0=x0,
                                            M=pc,
                                            tol=rel_tol,
                                            atol=abs_tol,
                                            maxiter=10000)

    # Verify convergence
    err = np.linalg.norm(A @ x - b)
    norm_b = np.linalg.norm(b)
    logger.debug(f"JAX Solver - Finished solving, |res| = {err}, |b| = {norm_b}") # info

    # if err <= max(rel_tol*norm_b, abs_tol):
    #     break
    # logger.warning(f"JAX Solver - Bicgstab did not converge, |res| = {err}, |b| = {norm_b}")

    # assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    # x = np.where(err < 0.1, x, np.nan) # For assert purpose, some how this also affects bicgstab.

    return x.astype(b.dtype)

def pardiso_solve_host(A, b, x0, solver_options):
    logger.debug(f"Pardiso Solver - Solving linear system")
    # A = jax.lax.stop_gradient(A)
    # b = jax.lax.stop_gradient(b)
    # If you need to convert PETSc to scipy
    start = time.time()
    Asp = scipy.sparse.csr_array((A.data, (A.indices[:,0], A.indices[:,1])),
                                 shape=A.shape)
    x = pypardiso.spsolve(Asp, onp.array(b))
    end = time.time()
    logger.debug(f'Pardiso Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b)}')
    logger.debug(f" Pardiso solver time: {end - start} [s]")
    return x.astype(b.dtype).reshape(b.shape)

def umfpack_solve_host(A, b):
    logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    # indptr, indices, data = A.getValuesCSR()
    # Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    A = jax.lax.stop_gradient(A)
    b = jax.lax.stop_gradient(b)
    dtype, shape = b.dtype, b.shape
    Asp = scipy.sparse.csr_array((A.data, (A.indices[:,0], A.indices[:,1])),
                                 shape=A.shape)
    x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug(f'Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b)}')
    return x.astype(b.dtype)

def cupy_solve_host(A, b):
    logger.debug(f"Cupy Solver - Solving linear system with Cupy")
    A = jax.lax.stop_gradient(A)
    b = jax.lax.stop_gradient(b)
    dtype, shape = b.dtype, b.shape
    # indptr, indices, data = A.getValuesCSR()
    # Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    Asp = csr_matrix((cp.array(A.data), (cp.array(A.indices[:,0]), cp.array(A.indices[:,1]))),
                                 shape=A.shape,dtype=cp.float32) #,dtype=cp.float32
    b_cp =  cp.array(b, dtype=cp.float32)
    x = spsolve(Asp, b_cp) #,dtype=cp.float32
    x = x.get()
    # logger.debug(f'Cupy Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b_cp)}')
    return x.astype(dtype).reshape(shape)

def petsc_solve_host(A, b):
    from petsc4py import PETSc
    ksp_type = 'bcgsl'
    pc_type = 'ilu'
    A_sp_scipy = scipy.sparse.csr_array((A.data, (A.indices[:,0], A.indices[:,1])),
                                 shape=A.shape)
    A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape,
                              csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
                                   A_sp_scipy.indices.astype(PETSc.IntType, copy=False),
                                   A_sp_scipy.data))

    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), onp.array(b))
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)

    # TODO: This works better. Do we need to generalize the code a little bit?
    if ksp_type == 'tfqmr':
        ksp.pc.setFactorSolverType('mumps')

    logger.debug(f'PETSc Solver - Solving linear system with ksp_type = {ksp.getType()}, pc = {ksp.pc.getType()}')
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x)

    # Verify convergence
    y = PETSc.Vec().createSeq(len(b))
    A.mult(x, y)

    err = np.linalg.norm(y.getArray() - rhs.getArray())
    logger.debug(f"PETSc Solver - Finished solving, linear solve res = {err}")
    assert err < 0.1, f"PETSc linear solver failed to converge, err = {err}"

    return x.getArray().astype(b.dtype)


def custom_solver(A_sp, b_active, x0_active,solver_options):
    def matvec(u):
        Au = A_sp @ u
        return Au
    result_shape = jax.ShapeDtypeStruct(b_active.shape, b_active.dtype)
    cust_solver = lambda matvec, b_vec: jax.pure_callback(AMGX_solve_host, result_shape, A_sp, x0_active, b_vec)
    x_active = jax.lax.custom_linear_solve(matvec, b_active, cust_solver, symmetric=True)
    return x_active

def linear_solver(A, b, x0, solver_options):

    # If user does not specify any solver, set jax_solver as the default one.
    if  len(solver_options.keys() & {'jax_solver', 'umfpack_solver', 'petsc_solver',
                                     'AMGX_solver', 'custom_solver','pardiso_solver','cupy_solver'}) == 0:
        solver_options['jax_solver'] = {}

    if 'jax_solver' in solver_options:
        precond = solver_options['jax_solver']['precond'] if 'precond' in solver_options['jax_solver'] else True
        x = jax_solve(A, b, x0, precond)
    elif 'umfpack_solver' in solver_options:
        x = umfpack_solve(A, b)
    elif 'petsc_solver' in solver_options:
        ksp_type = solver_options['petsc_solver']['ksp_type'] if 'ksp_type' in solver_options['petsc_solver'] else  'bcgsl'
        pc_type = solver_options['petsc_solver']['pc_type'] if 'pc_type' in solver_options['petsc_solver'] else 'ilu'
        x = petsc_solve(A, b, ksp_type, pc_type)
    elif 'AMGX_solver' in solver_options:
        x = AMGX_solve(A, b, x0)
        # x = AMGX_solve_host_gpu(A, b, x0)
    elif 'pardiso_solver' in solver_options:
        x = pardiso_solve(A, b, x0, solver_options)
    elif 'cupy_solver' in solver_options:
        x = cupy_solve(A, b)
    elif 'custom_solver' in solver_options:
        # Users can define their own solver
        # custom_solver = solver_options['custom_solver']
        x = custom_solver(A, b, x0, solver_options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    return x


################################################################################
# "row elimination" solver

def apply_bc_vec(res_vec, dofs, problem, scale=1.):
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = (res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True))
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-fe.vals_list[i]*scale)

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn, problem, scale=1.):
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)
    return res_fn_bc


def assign_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_ones_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(1.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_zeros_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(0.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs, problem):
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)

    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = (new_sol.at[fe.node_inds_list[i],
                                  fe.vec_inds_list[i]].set(sol[fe.node_inds_list[i],
                                          fe.vec_inds_list[i]]))
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list, problem):

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def operator_to_matrix(operator_fn, problem):
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_dofs_all_vars))
    return J


def linear_incremental_solver(problem, res_vec, A, dofs, solver_options):
    """
    Linear solver at each Newton's iteration
    """
    logger.debug(f"Solving linear system...")
    b = -res_vec

    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    if hasattr(problem, 'P_mat'):
        x0_2 = copy_bc(problem.P_mat @ dofs, problem)
        x0 = problem.P_mat.T @ (x0_1 - x0_2)
    else:
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2
    # TO DO: add term for column elimination
    # x0_1 is all zero and true BC
    # x0_2 is all zero exept at BC, where is current solution
    # x0 is all zero except at BC, where is the difference between current and true BC
    # because this would be the desired increment at BC
    # So, something like linear_solver(A, b +-? A @ x0, x0, solver_options) would be correct when column eliminating with nonzero Dirichlet BC

    inc = linear_solver(A, b, x0, solver_options)
    # print(f"inc = {inc}, dofs = {dofs}")

    line_search_flag = solver_options['line_search_flag'] if 'line_search_flag' in solver_options else False
    if line_search_flag:
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs


def line_search(problem, dofs, inc):
    """
    TODO: This is useful for finite deformation plasticity.
    """
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha*inc)
        return np.linalg.norm(res_vec)

    # grad_res_norm_fn = jax.grad(res_norm_fn)
    # hess_res_norm_fn = jax.hessian(res_norm_fn)

    # tol = 1e-3
    # alpha = 1.
    # lr = 1.
    # grad_alpha = 1.
    # while np.abs(grad_alpha) > tol:
    #     grad_alpha = grad_res_norm_fn(alpha)
    #     hess_alpha = hess_res_norm_fn(alpha)
    #     alpha = alpha - 1./hess_alpha*grad_alpha
    #     print(f"alpha = {alpha}, grad_alpha = {grad_alpha}, hess_alpha = {hess_alpha}")

    alpha = 1.
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.info(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.
            break
        res_norm = res_norm_half

    return dofs + alpha*inc

# # @jax.jit
# def temp(A, row_inds):
#     zero_mask = np.isin(np.arange(A.shape[0]), row_inds)
#     nonzero_mask = np.invert(zero_mask)
#     elim_rows = BCOO((nonzero_mask.astype(int), np.vstack((np.arange(A.shape[0]), np.arange(A.shape[0]))).T), shape=A.shape)
#     recover_diag = BCOO((zero_mask.astype(int), np.vstack((np.arange(A.shape[0]), np.arange(A.shape[0]))).T), shape=A.shape)
#     A = elim_rows @ A + recover_diag
#     return A

# # @jax.jit
# def temp2(A, row_inds):
#     mask = np.isin(A.indices[:,0], row_inds, invert=True)
#     # mask_columns = np.isin(A.indices[:,1], row_inds, invert=True)
#     # mask = np.logical_and(mask, mask_columns)

#     # Note concrete masks are required for jit
#     ij = np.vstack((A.indices[mask,:],
#                     np.repeat(row_inds[:,None],2, axis=1)))
#     v = np.concatenate((A.data[mask], np.ones(len(row_inds))))
#     # Assemble A with zero rows and ones on the diagonal of the zero rows
#     A = BCOO((v, ij), shape=A.shape)
#     return A

@jax.jit
def row_elimination_jax(A, row_inds):
    mask = np.isin(A.indices[:,0], row_inds, invert=True)
    # mask_columns = np.isin(A.indices[:,1], row_inds, invert=True)
    # mask = np.logical_and(mask, mask_columns)

    # Note concrete masks are required for jit
    ij = np.vstack((A.indices,
                    np.repeat(row_inds[:,None],2, axis=1)))
    v = np.concatenate((np.where(mask,A.data,0.0), np.ones(len(row_inds))))
    # Assemble A with zero rows and ones on the diagonal of the zero rows
    A = BCOO((v, ij), shape=A.shape)
    return A

def row_elimination_petsc(A, row_inds):
    pass

# # @jax.jit
# def P_transformation_jax(P, A):
#     return A
#     # return P.T @ (A @ P)
#     # tmp = A @ P
#     # P_T = P.transpose()
#     # A = P_T @ tmp
#     # return A

################################################################################
# GPU chunked assembly and row elimination
# NOTE: requires CuPy and cupyx
################################################################################
# CUDA kernel: zero a row and set diag=1
import cupy as cp
import cupyx.scipy.sparse as cusparse
import numpy as onp
import scipy.sparse
import math
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


def get_A_gpu_chunked_stream_elim_multigpu(problem, n_chunks=None, row_chunk_size=50000, gpu_ids=None):
    """
    Assemble global sparse matrix A using multi-GPU chunked computation optimized for H100 GPUs.
    Performs row elimination on GPU per chunk and streams to CPU CSR.
    
    Args:
        problem: Problem object with triplet data (V, I, J)
        n_chunks: Number of chunks. If None, auto-calculated based on GPU count and memory
        row_chunk_size: Batch size for row elimination kernel (increased for H100)
        gpu_ids: List of GPU device IDs to use. If None, uses all available GPUs.
    
    Returns:
        scipy.sparse.csr_matrix: Assembled and processed matrix on CPU
    """
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    
    # Determine available GPUs
    if gpu_ids is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_ids = list(range(n_gpus))
    else:
        n_gpus = len(gpu_ids)
    
    logger.info(f"Using {n_gpus} H100 GPUs: {gpu_ids}")
    
    # Query GPU memory for H100 optimization
    gpu_mem_info = []
    min_free_gb = float('inf')
    for gpu_id in gpu_ids:
        with cp.cuda.Device(gpu_id):
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            free_gb = free_mem / 1e9
            total_gb = total_mem / 1e9
            gpu_mem_info.append({'gpu_id': gpu_id, 'free_gb': free_gb, 'total_gb': total_gb})
            min_free_gb = min(min_free_gb, free_gb)
            logger.info(f"GPU {gpu_id}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    
    # Auto-calculate optimal chunk count for H100 (target ~20-30GB per chunk for efficiency)
    total_entries = len(problem.V)
    bytes_per_entry = 4 + 4 + 8 + 8  # I, J, V + CSR overhead
    total_data_gb = (total_entries * bytes_per_entry) / 1e9
    
    if n_chunks is None:
        # H100 optimization: Use fewer, larger chunks to maximize GPU utilization
        # Target 30GB per chunk on H100 80GB (leaves room for intermediate computations)
        target_chunk_gb = min(30.0, min_free_gb * 0.6)
        n_chunks = max(n_gpus * 2, int(onp.ceil(total_data_gb / target_chunk_gb)))
        logger.info(f"Auto-calculated {n_chunks} chunks (~{total_data_gb/n_chunks:.2f} GB per chunk)")
    
    # CPU accumulator with thread-safe lock
    data_cpu = []
    row_cpu = []
    col_cpu = []
    cpu_lock = threading.Lock()
    
    # Precompute all rows to eliminate (BCs) - done once on CPU
    all_row_inds_cpu = onp.concatenate([
        onp.array(
            fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
            dtype=onp.int32
        )
        for ind, fe in enumerate(problem.fes)
        for i in range(len(fe.node_inds_list))
    ])
    n_elim_rows = all_row_inds_cpu.size
    logger.debug(f"Total rows to eliminate: {int(n_elim_rows)}")
    
    # CUDA kernel for row elimination - optimized for H100
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;
        int r = rows[tid];
        int start = indptr[r];
        int end   = indptr[r + 1];
        
        // Coalesced memory access pattern for H100
        for (int j = start; j < end; ++j) {
            int col = indices[j];
            data[j] = (col == r) ? 1.0 : 0.0;
        }
    }
    '''
    
    # H100 Optimization: Sort triplets on GPU for row-aware chunking
    logger.info("Phase 1: GPU-accelerated row-aware chunking...")
    I_sorted, J_sorted, V_sorted, row_boundaries = _gpu_sort_h100_optimized(
        problem.I, problem.J, problem.V, gpu_ids, min_free_gb
    )
    
    # Distribute rows across chunks based on row boundaries
    n_rows = len(row_boundaries) - 1
    rows_per_chunk = n_rows // n_chunks
    
    chunk_boundaries = []
    for i in range(n_chunks):
        start_row_idx = i * rows_per_chunk
        if i == n_chunks - 1:
            end_row_idx = n_rows
        else:
            end_row_idx = (i + 1) * rows_per_chunk
        
        start_idx = row_boundaries[start_row_idx]
        end_idx = row_boundaries[end_row_idx]
        chunk_boundaries.append((start_idx, end_idx))
    
    # Create chunks from sorted data
    V_chunks = [V_sorted[start:end] for start, end in chunk_boundaries]
    I_chunks = [I_sorted[start:end] for start, end in chunk_boundaries]
    J_chunks = [J_sorted[start:end] for start, end in chunk_boundaries]
    
    logger.info(f"Phase 2: Processing {n_chunks} chunks across {n_gpus} GPUs...")
    
    def process_chunk_on_gpu(gpu_id, chunk_idx, V_chunk, I_chunk, J_chunk):
        """Process a single chunk on a specific GPU - H100 optimized."""
        try:
            # Set device and use pinned memory for faster transfers
            cp.cuda.Device(gpu_id).use()
            
            # Create CUDA stream for async operations
            stream = cp.cuda.Stream(non_blocking=True)
            
            with stream:
                # Copy elimination rows to this GPU (reuse across chunks on same GPU)
                all_row_inds = cp.array(all_row_inds_cpu, dtype=cp.int32)
                
                logger.debug(f"GPU {gpu_id}: Processing chunk {chunk_idx} "
                            f"({len(V_chunk)} nnz, {len(onp.unique(I_chunk))} rows)")
                
                # Move chunk to GPU with async transfer
                I_gpu = cp.asarray(I_chunk, dtype=cp.int32)
                J_gpu = cp.asarray(J_chunk, dtype=cp.int32)
                V_gpu = cp.asarray(V_chunk, dtype=cp.float64)
                
                # Build COO and convert to CSR for this chunk
                A_chunk = cusparse.coo_matrix((V_gpu, (I_gpu, J_gpu)), shape=shape).tocsr()
                indptr = A_chunk.indptr.astype(cp.int32)
                indices = A_chunk.indices.astype(cp.int32)
                data = A_chunk.data
                
                # Compile kernel for this GPU (cached after first use)
                kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')
                
                # Launch row elimination kernel in larger batches for H100
                threads_per_block = 256  # Increased for H100
                for start in range(0, n_elim_rows, row_chunk_size):
                    end = min(start + row_chunk_size, n_elim_rows)
                    rows_batch = all_row_inds[start:end]
                    blocks = (rows_batch.size + threads_per_block - 1) // threads_per_block
                    kernel((blocks,), (threads_per_block,),
                           (indptr, indices, data, rows_batch, cp.int32(rows_batch.size)))
                
                # Synchronize stream
                stream.synchronize()
                
                # Put modified arrays back into CSR
                A_chunk.indptr = indptr
                A_chunk.indices = indices
                A_chunk.data = data
                
                # Convert chunk to COO and move to CPU with pinned memory
                A_chunk_coo = A_chunk.tocoo()
                data_chunk = cp.asnumpy(A_chunk_coo.data)
                row_chunk = cp.asnumpy(A_chunk_coo.row)
                col_chunk = cp.asnumpy(A_chunk_coo.col)
                
                # Thread-safe append to CPU lists
                with cpu_lock:
                    data_cpu.append(data_chunk)
                    row_cpu.append(row_chunk)
                    col_cpu.append(col_chunk)
                
                # Free GPU memory
                del I_gpu, J_gpu, V_gpu, A_chunk, indptr, indices, data
                del rows_batch, A_chunk_coo, all_row_inds
                cp.get_default_memory_pool().free_all_blocks()
                
                logger.debug(f"GPU {gpu_id}: Completed chunk {chunk_idx}")
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error processing chunk {chunk_idx}: {str(e)}")
            raise
    
    # H100 Optimization: Distribute chunks to maximize concurrent execution
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for chunk_idx, (V_chunk, I_chunk, J_chunk) in enumerate(zip(V_chunks, I_chunks, J_chunks)):
            # Round-robin GPU assignment
            gpu_id = gpu_ids[chunk_idx % n_gpus]
            future = executor.submit(
                process_chunk_on_gpu, 
                gpu_id, 
                chunk_idx, 
                V_chunk, 
                I_chunk, 
                J_chunk
            )
            futures.append(future)
        
        # Wait for all chunks to complete
        for future in futures:
            future.result()
    
    logger.info(f"Phase 3: Assembling final matrix...")
    
    # Concatenate CPU arrays
    data_cpu = onp.concatenate(data_cpu)
    row_cpu = onp.concatenate(row_cpu)
    col_cpu = onp.concatenate(col_cpu)
    
    # Construct CPU CSR
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    logger.info(f"Completed multi-GPU assembly: {A_cpu.nnz} nonzeros, "
                f"{A_cpu.shape[0]} × {A_cpu.shape[1]} matrix")
    
    return A_cpu


def _gpu_sort_h100_optimized(I, J, V, gpu_ids, available_gb):
    """
    H100-optimized sorting with intelligent memory management.
    Uses single-GPU sort if data fits, otherwise uses multi-GPU chunked sort.
    """
    total_entries = len(I)
    bytes_per_entry = 4 + 4 + 8 + 4  # I, J, V + sort_idx
    required_gb = (total_entries * bytes_per_entry) / 1e9
    
    # H100 can handle much larger sorts in-memory (up to 50GB efficiently)
    if required_gb < available_gb * 0.7:
        logger.info(f"Using single-GPU sort ({required_gb:.2f} GB fits in {available_gb:.2f} GB)")
        return _gpu_sort_in_memory(I, J, V, gpu_ids[0])
    else:
        logger.info(f"Using multi-GPU chunked sort ({required_gb:.2f} GB exceeds {available_gb * 0.7:.2f} GB)")
        # For H100, use larger chunks (40GB target) to minimize merge overhead
        target_chunk_gb = min(40.0, available_gb * 0.6)
        return _gpu_sort_chunked_h100(I, J, V, gpu_ids, target_chunk_gb)


def _gpu_sort_in_memory(I, J, V, gpu_id):
    """Sort triplets entirely in GPU memory (fast path for H100)."""
    with cp.cuda.Device(gpu_id):
        logger.debug(f"GPU {gpu_id}: Loading and sorting {len(I)} entries...")
        
        I_gpu = cp.array(I, dtype=cp.int32)
        J_gpu = cp.array(J, dtype=cp.int32)
        V_gpu = cp.array(V, dtype=cp.float64)
        
        # GPU argsort - H100 is extremely fast at this
        sort_idx = cp.argsort(I_gpu)
        I_sorted_gpu = I_gpu[sort_idx]
        J_sorted_gpu = J_gpu[sort_idx]
        V_sorted_gpu = V_gpu[sort_idx]
        
        # Find row boundaries efficiently
        unique_rows_gpu = cp.unique(I_sorted_gpu)
        row_boundaries_gpu = cp.searchsorted(I_sorted_gpu, unique_rows_gpu, side='left')
        row_boundaries_gpu = cp.append(row_boundaries_gpu, len(I_sorted_gpu))
        
        # Move to CPU
        I_sorted = cp.asnumpy(I_sorted_gpu)
        J_sorted = cp.asnumpy(J_sorted_gpu)
        V_sorted = cp.asnumpy(V_sorted_gpu)
        row_boundaries = cp.asnumpy(row_boundaries_gpu)
        
        # Cleanup
        del I_gpu, J_gpu, V_gpu, sort_idx
        del I_sorted_gpu, J_sorted_gpu, V_sorted_gpu
        del unique_rows_gpu, row_boundaries_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    logger.debug(f"GPU {gpu_id}: Sort complete, {len(row_boundaries)-1} unique rows")
    return I_sorted, J_sorted, V_sorted, row_boundaries


def _gpu_sort_chunked_h100(I, J, V, gpu_ids, target_chunk_gb):
    """
    Multi-GPU chunked sort optimized for H100 80GB GPUs.
    Uses fewer, larger chunks with minimal subdivisions.
    """
    total_entries = len(I)
    bytes_per_entry = 4 + 4 + 8 + 4
    entries_per_chunk = int((target_chunk_gb * 1e9) / bytes_per_entry)
    n_sort_chunks = max(len(gpu_ids), math.ceil(total_entries / entries_per_chunk))
    
    logger.info(f"Chunked sort: {n_sort_chunks} chunks, ~{entries_per_chunk/1e6:.1f}M entries per chunk")
    
    sorted_chunks = []
    
    def sort_chunk_on_gpu_h100(gpu_id, chunk_idx, I_chunk, J_chunk, V_chunk):
        """Sort chunk on H100 - rarely needs subdivision with 80GB memory."""
        chunk_size = len(I_chunk)
        
        # H100: Try with minimal subdivisions (2 or 4)
        for n_subdivisions in [2, 4]:
            try:
                with cp.cuda.Device(gpu_id):
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    sub_chunks = []
                    sub_size = chunk_size // n_subdivisions
                    
                    for sub_idx in range(n_subdivisions):
                        start = sub_idx * sub_size
                        end = chunk_size if sub_idx == n_subdivisions - 1 else (sub_idx + 1) * sub_size
                        
                        I_sub = I_chunk[start:end]
                        J_sub = J_chunk[start:end]
                        V_sub = V_chunk[start:end]
                        
                        # Direct array creation (no intermediate copies)
                        I_gpu = cp.asarray(I_sub, dtype=cp.int32)
                        J_gpu = cp.asarray(J_sub, dtype=cp.int32)
                        V_gpu = cp.asarray(V_sub, dtype=cp.float64)
                        
                        sort_idx = cp.argsort(I_gpu)
                        I_sorted_sub = cp.asnumpy(I_gpu[sort_idx])
                        J_sorted_sub = cp.asnumpy(J_gpu[sort_idx])
                        V_sorted_sub = cp.asnumpy(V_gpu[sort_idx])
                        
                        sub_chunks.append((I_sorted_sub, J_sorted_sub, V_sorted_sub))
                        
                        del I_gpu, J_gpu, V_gpu, sort_idx
                        cp.get_default_memory_pool().free_all_blocks()
                    
                    # Merge sub-chunks
                    I_sorted, J_sorted, V_sorted = _kway_merge_triplets(sub_chunks)
                    
                    logger.debug(f"GPU {gpu_id}: Chunk {chunk_idx} sorted with {n_subdivisions} subdivisions")
                    return I_sorted, J_sorted, V_sorted
                    
            except cp.cuda.memory.OutOfMemoryError:
                logger.warning(f"GPU {gpu_id}: OOM with {n_subdivisions} subdivisions, trying more...")
                cp.get_default_memory_pool().free_all_blocks()
                continue
        
        # Fallback to CPU (rare for H100)
        logger.error(f"GPU {gpu_id}: Falling back to CPU sort for chunk {chunk_idx}")
        sort_idx = onp.argsort(I_chunk)
        return I_chunk[sort_idx], J_chunk[sort_idx], V_chunk[sort_idx]
    
    # Parallel sort across all GPUs
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for chunk_idx in range(n_sort_chunks):
            start = chunk_idx * entries_per_chunk
            end = min((chunk_idx + 1) * entries_per_chunk, total_entries)
            
            I_chunk = I[start:end]
            J_chunk = J[start:end]
            V_chunk = V[start:end]
            
            gpu_id = gpu_ids[chunk_idx % len(gpu_ids)]
            future = executor.submit(sort_chunk_on_gpu_h100, gpu_id, chunk_idx,
                                    I_chunk, J_chunk, V_chunk)
            futures.append(future)
        
        for future in futures:
            sorted_chunks.append(future.result())
    
    # K-way merge
    logger.info(f"Merging {len(sorted_chunks)} sorted chunks...")
    I_sorted, J_sorted, V_sorted = _kway_merge_triplets(sorted_chunks)
    
    # Find row boundaries
    unique_rows, row_boundaries = onp.unique(I_sorted, return_index=True)
    row_boundaries = onp.append(row_boundaries, len(I_sorted))
    
    logger.info(f"Merge complete: {len(unique_rows)} unique rows")
    return I_sorted, J_sorted, V_sorted, row_boundaries


def _kway_merge_triplets(sorted_chunks):
    """
    Efficient k-way merge using min-heap.
    Optimized for H100: handles large merges efficiently.
    """
    import heapq
    
    heap = []
    chunk_indices = [0] * len(sorted_chunks)
    
    # Initialize heap
    for chunk_id, (I_chunk, J_chunk, V_chunk) in enumerate(sorted_chunks):
        if len(I_chunk) > 0:
            heapq.heappush(heap, (I_chunk[0], chunk_id))
    
    # Preallocate output
    total_size = sum(len(chunk[0]) for chunk in sorted_chunks)
    I_merged = onp.empty(total_size, dtype=onp.int32)
    J_merged = onp.empty(total_size, dtype=onp.int32)
    V_merged = onp.empty(total_size, dtype=onp.float64)
    
    # Merge
    out_idx = 0
    while heap:
        row_val, chunk_id = heapq.heappop(heap)
        idx = chunk_indices[chunk_id]
        
        I_chunk, J_chunk, V_chunk = sorted_chunks[chunk_id]
        I_merged[out_idx] = I_chunk[idx]
        J_merged[out_idx] = J_chunk[idx]
        V_merged[out_idx] = V_chunk[idx]
        out_idx += 1
        
        # Add next element from same chunk
        chunk_indices[chunk_id] += 1
        if chunk_indices[chunk_id] < len(I_chunk):
            next_idx = chunk_indices[chunk_id]
            heapq.heappush(heap, (I_chunk[next_idx], chunk_id))
    
    return I_merged, J_merged, V_merged
#==============================
import cupy as cp
import cupyx.scipy.sparse as cusparse
import scipy.sparse
import numpy as onp
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import os

def get_A_multi_gpu_fast_old(problem, n_gpus=None, use_processes=False):
    """
    High-performance multi-GPU assembly with optimizations:
    - Minimal data transfers
    - Overlapped compute and transfer with streams
    - Each GPU only processes its own elimination rows
    - Uses threading (not multiprocessing) to avoid CUDA context issues
    
    Args:
        problem: Problem object with triplet data
        n_gpus: Number of GPUs (None = auto-detect)
        use_processes: If True, use multiprocessing (requires CUDA_VISIBLE_DEVICES setup)
    
    Returns:
        scipy.sparse.csr_matrix: Assembled sparse matrix
    """
    if n_gpus is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    
    n_triplets = len(problem.I)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    n_rows = shape[0]
    
    # For small problems, GPU overhead dominates - use CPU
    if n_triplets < 5_000_000:
        logger.warning(f"Problem too small ({n_triplets:,} triplets) for multi-GPU benefit. Consider CPU.")
    
    logger.debug(f"Multi-GPU fast assembly: {n_gpus} GPUs, {n_triplets:,} triplets, {n_rows:,} rows")
    
    # Precompute elimination rows on CPU
    all_row_inds_list = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            rows = fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
            all_row_inds_list.extend(rows)
    all_row_inds_cpu = onp.array(all_row_inds_list, dtype=onp.int32)
    
    # Partition rows across GPUs
    rows_per_gpu = math.ceil(n_rows / n_gpus)
    row_ranges = [(i * rows_per_gpu, min((i + 1) * rows_per_gpu, n_rows)) 
                  for i in range(n_gpus)]
    
    # CUDA kernel for row elimination
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows, 
                                const int row_start, const int row_end)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;
        int r = rows[tid];
        
        // Skip rows outside this GPU's range
        if (r < row_start || r >= row_end) return;
        
        int start = indptr[r];
        int end   = indptr[r + 1];
        for (int j = start; j < end; ++j) {
            data[j] = 0.0;
            if (indices[j] == r) {
                data[j] = 1.0;
            }
        }
    }
    '''
    
    # Use threading instead of multiprocessing to avoid CUDA context issues
    results = []
    results_lock = threading.Lock()
    errors = []
    
    def gpu_worker(gpu_id, row_start, row_end):
        """Worker function running in separate thread"""
        try:
            # Set GPU for this thread
            with cp.cuda.Device(gpu_id):
                kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')
                
                # Create CUDA stream for this GPU
                stream = cp.cuda.Stream()
                
                with stream:
                    # Transfer data to GPU (pinned memory for faster transfer)
                    I_gpu = cp.asarray(problem.I, dtype=cp.int32)
                    J_gpu = cp.asarray(problem.J, dtype=cp.int32)
                    V_gpu = cp.asarray(problem.V, dtype=cp.float64)
                    
                    # Filter for this GPU's row range (on GPU, very fast)
                    mask = (I_gpu >= row_start) & (I_gpu < row_end)
                    I_filtered = I_gpu[mask]
                    J_filtered = J_gpu[mask]
                    V_filtered = V_gpu[mask]
                    
                    del I_gpu, J_gpu, V_gpu, mask
                    
                    if len(I_filtered) == 0:
                        logger.warning(f"GPU {gpu_id}: No data in row range [{row_start}, {row_end})")
                        return
                    
                    logger.debug(f"GPU {gpu_id}: Processing {len(I_filtered):,} triplets in rows [{row_start}, {row_end})")
                    
                    # Build CSR directly
                    A_chunk = cusparse.coo_matrix(
                        (V_filtered, (I_filtered, J_filtered)), 
                        shape=shape
                    ).tocsr()
                    
                    del I_filtered, J_filtered, V_filtered
                    
                    # Get CSR arrays
                    indptr = A_chunk.indptr.astype(cp.int32)
                    indices = A_chunk.indices.astype(cp.int32)
                    data = A_chunk.data
                    
                    # Filter elimination rows for this range (on GPU)
                    elim_rows_gpu = cp.asarray(all_row_inds_cpu, dtype=cp.int32)
                    
                    # Launch kernel with range check inside kernel
                    threads_per_block = 256
                    blocks = math.ceil(len(elim_rows_gpu) / threads_per_block)
                    
                    kernel(
                        (blocks,), (threads_per_block,),
                        (indptr, indices, data, elim_rows_gpu, 
                         cp.int32(len(elim_rows_gpu)), 
                         cp.int32(row_start), cp.int32(row_end))
                    )
                    
                    # Update CSR data
                    A_chunk.data = data
                    
                    # Convert to COO for final assembly
                    A_chunk_coo = A_chunk.tocoo()
                    
                    # Transfer to CPU
                    data_cpu = cp.asnumpy(A_chunk_coo.data)
                    row_cpu = cp.asnumpy(A_chunk_coo.row)
                    col_cpu = cp.asnumpy(A_chunk_coo.col)
                    
                    # Store results thread-safely
                    with results_lock:
                        results.append((gpu_id, data_cpu, row_cpu, col_cpu))
                    
                    # Cleanup
                    del A_chunk, A_chunk_coo, indptr, indices, data, elim_rows_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    logger.debug(f"GPU {gpu_id}: Completed {len(data_cpu):,} non-zeros")
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} failed: {e}")
            import traceback
            with results_lock:
                errors.append((gpu_id, str(e), traceback.format_exc()))
    
    # Launch GPU workers as threads
    threads = []
    for gpu_id, (row_start, row_end) in enumerate(row_ranges):
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, row_start, row_end)
        )
        t.start()
        threads.append(t)
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Check for errors
    if errors:
        logger.error(f"Errors occurred in {len(errors)} GPU(s):")
        for gpu_id, err, trace in errors:
            logger.error(f"GPU {gpu_id}: {err}")
            logger.debug(trace)
        raise RuntimeError(f"{len(errors)} GPU worker(s) failed")
    
    if not results:
        raise RuntimeError("No results from GPU workers!")
    
    # Sort results by GPU ID and concatenate
    results.sort(key=lambda x: x[0])
    data_cpu = onp.concatenate([d for _, d, _, _ in results])
    row_cpu = onp.concatenate([r for _, _, r, _ in results])
    col_cpu = onp.concatenate([c for _, _, _, c in results])
    
    # Build final CSR matrix
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    
    logger.debug(f"Multi-GPU assembly: {A_cpu.nnz:,} non-zeros")
    
    return A_cpu

def estimate_memory_usage(n_triplets, shape):
    """Estimate GPU memory usage for triplet data and CSR matrix"""
    # Triplet data: I, J (int32), V (float64)
    triplet_mem = n_triplets * (4 + 4 + 8)  # bytes
    
    # CSR matrix (conservative estimate)
    # Assume ~10% sparsity for dense problems, 1% for sparse
    estimated_nnz = min(n_triplets, shape[0] * shape[1] * 0.01)
    csr_mem = estimated_nnz * 12 + shape[0] * 4  # data + indices + indptr
    
    # Add 50% safety margin
    total_mem = (triplet_mem + csr_mem) * 1.5
    return int(total_mem)

def get_gpu_memory_info(gpu_id):
    """Get available GPU memory in bytes"""
    with cp.cuda.Device(gpu_id):
        meminfo = cp.cuda.runtime.memGetInfo()
        free_mem = meminfo[0]
        total_mem = meminfo[1]
        return free_mem, total_mem

def get_A_multi_gpu_fast1(problem, n_gpus=None, use_processes=False, max_memory_fraction=0.8):
    """
    High-performance multi-GPU assembly with optimizations:
    - Minimal data transfers
    - Overlapped compute and transfer with streams
    - Each GPU only processes its own elimination rows
    - Uses threading (not multiprocessing) to avoid CUDA context issues
    - Implements subchunking for partitions that don't fit in GPU memory
    
    Args:
        problem: Problem object with triplet data
        n_gpus: Number of GPUs (None = auto-detect)
        use_processes: If True, use multiprocessing (requires CUDA_VISIBLE_DEVICES setup)
        max_memory_fraction: Maximum fraction of GPU memory to use
    
    Returns:
        scipy.sparse.csr_matrix: Assembled sparse matrix
    """
    if n_gpus is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    
    n_triplets = len(problem.I)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    n_rows = shape[0]
    
    # For small problems, GPU overhead dominates - use CPU
    if n_triplets < 5_000_000:
        logger.warning(f"Problem too small ({n_triplets:,} triplets) for multi-GPU benefit. Consider CPU.")
    
    logger.debug(f"Multi-GPU fast assembly: {n_gpus} GPUs, {n_triplets:,} triplets, {n_rows:,} rows")
    
    # Precompute elimination rows on CPU
    all_row_inds_list = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            rows = fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
            all_row_inds_list.extend(rows)
    all_row_inds_cpu = onp.array(all_row_inds_list, dtype=onp.int32)
    
    # Partition rows across GPUs
    rows_per_gpu = math.ceil(n_rows / n_gpus)
    row_ranges = [(i * rows_per_gpu, min((i + 1) * rows_per_gpu, n_rows)) 
                  for i in range(n_gpus)]
    
    # CUDA kernel for row elimination
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows, 
                                const int row_start, const int row_end)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;
        int r = rows[tid];
        
        // Skip rows outside this GPU's range
        if (r < row_start || r >= row_end) return;
        
        int start = indptr[r];
        int end   = indptr[r + 1];
        for (int j = start; j < end; ++j) {
            data[j] = 0.0;
            if (indices[j] == r) {
                data[j] = 1.0;
            }
        }
    }
    '''
    
    # Use threading instead of multiprocessing to avoid CUDA context issues
    results = []
    results_lock = threading.Lock()
    errors = []
    
    def process_subchunk(gpu_id, I_chunk, J_chunk, V_chunk, row_start, row_end, kernel, elim_rows_gpu):
        """Process a single subchunk on GPU"""
        # Build CSR directly
        A_chunk = cusparse.coo_matrix(
            (V_chunk, (I_chunk, J_chunk)), 
            shape=shape
        ).tocsr()
        
        # Get CSR arrays
        indptr = A_chunk.indptr.astype(cp.int32)
        indices = A_chunk.indices.astype(cp.int32)
        data = A_chunk.data
        
        # Launch kernel with range check inside kernel
        threads_per_block = 256
        blocks = math.ceil(len(elim_rows_gpu) / threads_per_block)
        
        kernel(
            (blocks,), (threads_per_block,),
            (indptr, indices, data, elim_rows_gpu, 
             cp.int32(len(elim_rows_gpu)), 
             cp.int32(row_start), cp.int32(row_end))
        )
        
        # Update CSR data
        A_chunk.data = data
        
        # Convert to COO for accumulation
        A_chunk_coo = A_chunk.tocoo()
        
        return A_chunk_coo.data, A_chunk_coo.row, A_chunk_coo.col
    
    def gpu_worker(gpu_id, row_start, row_end):
        """Worker function running in separate thread with subchunking support"""
        try:
            # Set GPU for this thread
            with cp.cuda.Device(gpu_id):
                # Get available GPU memory
                free_mem, total_mem = get_gpu_memory_info(gpu_id)
                available_mem = int(free_mem * max_memory_fraction)
                
                kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')
                
                # Create CUDA stream for this GPU
                stream = cp.cuda.Stream()
                
                with stream:
                    # Transfer elimination rows once
                    elim_rows_gpu = cp.asarray(all_row_inds_cpu, dtype=cp.int32)
                    
                    # Filter triplets for this GPU's row range on CPU first
                    mask_cpu = (problem.I >= row_start) & (problem.I < row_end)
                    I_filtered_cpu = problem.I[mask_cpu]
                    J_filtered_cpu = problem.J[mask_cpu]
                    V_filtered_cpu = problem.V[mask_cpu]
                    
                    if len(I_filtered_cpu) == 0:
                        logger.warning(f"GPU {gpu_id}: No data in row range [{row_start}, {row_end})")
                        return
                    
                    # Estimate memory for this partition
                    partition_mem = estimate_memory_usage(len(I_filtered_cpu), shape)
                    
                    logger.debug(f"GPU {gpu_id}: {len(I_filtered_cpu):,} triplets, "
                               f"estimated {partition_mem / 1e9:.2f} GB, "
                               f"available {available_mem / 1e9:.2f} GB")
                    
                    # Determine if subchunking is needed
                    if partition_mem <= available_mem:
                        # Process entire partition at once
                        logger.debug(f"GPU {gpu_id}: Processing entire partition")
                        
                        I_gpu = cp.asarray(I_filtered_cpu, dtype=cp.int32)
                        J_gpu = cp.asarray(J_filtered_cpu, dtype=cp.int32)
                        V_gpu = cp.asarray(V_filtered_cpu, dtype=cp.float64)
                        
                        data_gpu, row_gpu, col_gpu = process_subchunk(
                            gpu_id, I_gpu, J_gpu, V_gpu, row_start, row_end, kernel, elim_rows_gpu
                        )
                        
                        # Transfer final result to CPU
                        data_cpu = cp.asnumpy(data_gpu)
                        row_cpu = cp.asnumpy(row_gpu)
                        col_cpu = cp.asnumpy(col_gpu)
                        
                    else:
                        # Subchunking needed
                        n_triplets_partition = len(I_filtered_cpu)
                        
                        # Conservative estimate: use 80% of estimated memory per subchunk
                        triplets_per_subchunk = int((available_mem * 0.8) / (4 + 4 + 8 + 12))  # I,J,V + CSR overhead
                        n_subchunks = math.ceil(n_triplets_partition / triplets_per_subchunk)
                        
                        logger.debug(f"GPU {gpu_id}: Using {n_subchunks} subchunks of ~{triplets_per_subchunk:,} triplets each")
                        
                        # Accumulate results from all subchunks
                        all_data = []
                        all_row = []
                        all_col = []
                        
                        for subchunk_id in range(n_subchunks):
                            start_idx = subchunk_id * triplets_per_subchunk
                            end_idx = min((subchunk_id + 1) * triplets_per_subchunk, n_triplets_partition)
                            
                            logger.debug(f"GPU {gpu_id}: Processing subchunk {subchunk_id + 1}/{n_subchunks} "
                                       f"(triplets {start_idx:,} to {end_idx:,})")
                            
                            # Transfer subchunk to GPU
                            I_subchunk = cp.asarray(I_filtered_cpu[start_idx:end_idx], dtype=cp.int32)
                            J_subchunk = cp.asarray(J_filtered_cpu[start_idx:end_idx], dtype=cp.int32)
                            V_subchunk = cp.asarray(V_filtered_cpu[start_idx:end_idx], dtype=cp.float64)
                            
                            # Process subchunk
                            data_sub, row_sub, col_sub = process_subchunk(
                                gpu_id, I_subchunk, J_subchunk, V_subchunk, 
                                row_start, row_end, kernel, elim_rows_gpu
                            )
                            
                            # Accumulate results
                            all_data.append(cp.asnumpy(data_sub))
                            all_row.append(cp.asnumpy(row_sub))
                            all_col.append(cp.asnumpy(col_sub))
                            
                            # Cleanup subchunk GPU memory
                            del I_subchunk, J_subchunk, V_subchunk, data_sub, row_sub, col_sub
                            cp.get_default_memory_pool().free_all_blocks()
                        
                        # Concatenate all subchunk results
                        data_cpu = onp.concatenate(all_data)
                        row_cpu = onp.concatenate(all_row)
                        col_cpu = onp.concatenate(all_col)
                    
                    # Store results thread-safely
                    with results_lock:
                        results.append((gpu_id, data_cpu, row_cpu, col_cpu))
                    
                    # Final cleanup
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    logger.debug(f"GPU {gpu_id}: Completed {len(data_cpu):,} non-zeros")
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} failed: {e}")
            import traceback
            with results_lock:
                errors.append((gpu_id, str(e), traceback.format_exc()))
    
    # Launch GPU workers as threads
    threads = []
    for gpu_id, (row_start, row_end) in enumerate(row_ranges):
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, row_start, row_end)
        )
        t.start()
        threads.append(t)
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Check for errors
    if errors:
        logger.error(f"Errors occurred in {len(errors)} GPU(s):")
        for gpu_id, err, trace in errors:
            logger.error(f"GPU {gpu_id}: {err}")
            logger.debug(trace)
        raise RuntimeError(f"{len(errors)} GPU worker(s) failed")
    
    if not results:
        raise RuntimeError("No results from GPU workers!")
    
    # Sort results by GPU ID and concatenate
    results.sort(key=lambda x: x[0])
    data_cpu = onp.concatenate([d for _, d, _, _ in results])
    row_cpu = onp.concatenate([r for _, _, r, _ in results])
    col_cpu = onp.concatenate([c for _, _, _, c in results])
    
    # Build final CSR matrix
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    
    logger.debug(f"Multi-GPU assembly: {A_cpu.nnz:,} non-zeros")
    
    return A_cpu

import cupy as cp
import cupyx.scipy.sparse as cusparse
import scipy.sparse
import numpy as onp
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import os
import logging
import time


def get_gpu_memory_info(device_id):
    """Get free and total memory for a GPU device"""
    with cp.cuda.Device(device_id):
        free = cp.cuda.runtime.memGetInfo()[0]
        total = cp.cuda.runtime.getDeviceProperties(device_id)['totalGlobalMem']
    return free, total

def estimate_memory_usage(n_triplets, shape):
    """Estimate memory needed for processing n_triplets into a matrix of shape"""
    # Triplet storage: I, J arrays (int32) and V array (float64)
    triplet_mem = n_triplets * (4 + 4 + 8)
    
    # Estimate CSR overhead (worst case: every row has entries)
    csr_indptr = (shape[0] + 1) * 4
    csr_indices = n_triplets * 4
    csr_data = n_triplets * 8
    
    # Total with 20% buffer for intermediate operations
    return int((triplet_mem + csr_indptr + csr_indices + csr_data) * 1.2)

# Add these imports at the top
def enable_peer_access():
    """Enable peer access between all GPUs that support it"""
    n_gpus = cp.cuda.runtime.getDeviceCount()
    
    # Create a matrix to track which GPUs can access each other
    peer_access_matrix = []
    
    for i in range(n_gpus):
        peers = []
        cp.cuda.Device(i).use()
        
        for j in range(n_gpus):
            if i == j:
                peers.append(False)
                continue
                
            can_access = False
            try:
                # Check if peer access is possible
                can_access = cp.cuda.runtime.deviceCanAccessPeer(i, j)
                if can_access:
                    try:
                        # Enable peer access
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                        logger.debug(f"Enabled peer access from GPU {i} to GPU {j}")
                    except cp.cuda.runtime.CUDARuntimeError as e:
                        # Check if the error is just that peer access was already enabled
                        if "cudaErrorPeerAccessAlreadyEnabled" in str(e):
                            logger.debug(f"Peer access from GPU {i} to GPU {j} already enabled")
                            can_access = True
                        else:
                            # For any other error, log a warning but continue
                            logger.warning(f"Error enabling peer access from GPU {i} to {j}: {e}")
                            can_access = False
            except Exception as e:
                logger.warning(f"Failed to check peer access from GPU {i} to {j}: {e}")
                can_access = False
                
            peers.append(can_access)
        peer_access_matrix.append(peers)
            
    return peer_access_matrix

def get_nvlink_topology():
    """Detect and log NVLink connections between GPUs"""
    n_gpus = cp.cuda.runtime.getDeviceCount()
    nvlink_connections = []
    
    for i in range(n_gpus):
        for j in range(i+1, n_gpus):
            try:
                with cp.cuda.Device(i):
                    # Check if NVLink is available between these GPUs
                    is_nvlink = cp.cuda.runtime.deviceCanAccessPeer(i, j)
                    if is_nvlink:
                        # Try to get NVLink bandwidth information if possible
                        # Note: This requires NVML library which may not be directly accessible through CuPy
                        nvlink_connections.append((i, j))
                        logger.info(f"NVLink detected between GPU {i} and GPU {j}")
            except Exception:
                pass
                
    return nvlink_connections

def nvlink_aware_gather(local_arrays, gpu_ids, destination_gpu=0):
    """
    Gather arrays from multiple GPUs to a destination GPU using NVLink where available
    
    Args:
        local_arrays: List of arrays, each on a different GPU
        gpu_ids: List of GPU IDs corresponding to each array
        destination_gpu: Destination GPU ID
        
    Returns:
        Concatenated array on the destination GPU
    """
    result_list = []
    
    # Enable peer access
    peer_access_matrix = enable_peer_access()
    
    with cp.cuda.Device(destination_gpu):
        for idx, (array, src_gpu) in enumerate(zip(local_arrays, gpu_ids)):
            if src_gpu == destination_gpu:
                # Local data, no transfer needed
                result_list.append(array)
            elif peer_access_matrix[destination_gpu][src_gpu]:
                # Direct peer transfer possible (NVLink or P2P)
                # Create a stream for this transfer
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    # Create output array on destination
                    dest_array = cp.empty_like(array)
                    # Perform direct peer copy
                    cp.cuda.runtime.memcpy(dest_array.data.ptr, 
                                          array.data.ptr,
                                          array.nbytes)
                    result_list.append(dest_array)
                # Synchronize this stream
                stream.synchronize()
            else:
                # No direct access, go through CPU
                cpu_data = cp.asnumpy(array)
                result_list.append(cp.array(cpu_data))
        
        # Concatenate all arrays on the destination GPU
        return cp.concatenate(result_list)

def get_A_multi_gpu_fast(problem, n_gpus=None, use_processes=False, max_memory_fraction=0.8):
    """
    High-performance multi-GPU assembly with optimizations:
    - Minimal data transfers
    - Overlapped compute and transfer with streams
    - Each GPU only processes its own elimination rows
    - Uses threading (not multiprocessing) to avoid CUDA context issues
    - Implements subchunking for partitions that don't fit in GPU memory
    - Leverages NVLink for faster GPU-to-GPU transfers when available
    
    Args:
        problem: Problem object with triplet data
        n_gpus: Number of GPUs (None = auto-detect)
        use_processes: If True, use multiprocessing (requires CUDA_VISIBLE_DEVICES setup)
        max_memory_fraction: Maximum fraction of GPU memory to use
    
    Returns:
        scipy.sparse.csr_matrix: Assembled sparse matrix
    """
    start_time = time.time()
    
    if n_gpus is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    
    # Enable peer access between GPUs for NVLink communication
    logger.debug("Setting up GPU peer access for NVLink...")
    peer_access_matrix = enable_peer_access()
    nvlink_connections = get_nvlink_topology()
    logger.debug(f"Detected {len(nvlink_connections)} NVLink connections between GPUs")
    
    n_triplets = len(problem.I)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    n_rows = shape[0]
    
    # For small problems, GPU overhead dominates - use CPU
    if n_triplets < 5_000_000:
        logger.warning(f"Problem too small ({n_triplets:,} triplets) for multi-GPU benefit. Consider CPU.")
    
    logger.debug(f"Multi-GPU fast assembly: {n_gpus} GPUs, {n_triplets:,} triplets, {n_rows:,} rows")
    
    # Precompute elimination rows on CPU
    logger.debug("Precomputing elimination rows...")
    all_row_inds_list = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            rows = fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
            all_row_inds_list.extend(rows)
    all_row_inds_cpu = onp.array(all_row_inds_list, dtype=onp.int32)
    
    # Pre-sort triplets by row for better memory locality and CSR conversion
    logger.debug("Pre-sorting triplets by row...")
    sorted_indices = onp.argsort(problem.I, kind='stable')
    I_sorted = problem.I[sorted_indices]
    J_sorted = problem.J[sorted_indices]
    V_sorted = problem.V[sorted_indices]
    
    # Partition rows across GPUs with load balancing based on nnz per row
    logger.debug("Partitioning workload across GPUs...")
    # Estimate nnz per row
    row_counts = onp.bincount(I_sorted, minlength=n_rows)
    cumulative_work = onp.cumsum(row_counts)
    total_work = cumulative_work[-1]
    work_per_gpu = total_work / n_gpus
    
    # Find better partition points
    row_ranges = []
    for i in range(n_gpus):
        if i == 0:
            start_row = 0
        else:
            # Find row where cumulative work exceeds i*work_per_gpu
            start_row = onp.searchsorted(cumulative_work, i*work_per_gpu)
            
        if i == n_gpus - 1:
            end_row = n_rows
        else:
            end_row = onp.searchsorted(cumulative_work, (i+1)*work_per_gpu)
        
        row_ranges.append((start_row, end_row))
    
    # CUDA kernel for row elimination - optimized with grid-stride loop
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* __restrict__ indptr, 
                               const int* __restrict__ indices, 
                               double* __restrict__ data,
                               const int* __restrict__ rows, 
                               const int n_rows, 
                               const int row_start, 
                               const int row_end)
    {
        // Grid-stride loop for better occupancy
        for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < n_rows;
            idx += blockDim.x * gridDim.x)
        {
            int r = rows[idx];
            
            // Skip rows outside this GPU's range
            if (r < row_start || r >= row_end) continue;
            
            int start = indptr[r];
            int end   = indptr[r + 1];
            
            // Use vectorized loads/stores where possible
            for (int j = start; j < end; ++j) {
                double value = (indices[j] == r) ? 1.0 : 0.0;
                data[j] = value;
            }
        }
    }
    '''
    
    # Use threading instead of multiprocessing to avoid CUDA context issues
    results = []
    results_lock = threading.Lock()
    errors = []
    
    def process_subchunk(gpu_id, I_chunk, J_chunk, V_chunk, row_start, row_end, kernel, elim_rows_gpu, stream):
        """Process a single subchunk on GPU with stream"""
        with stream:
            # Build CSR directly from sorted triplets for better performance
            A_chunk = cusparse.coo_matrix(
                (V_chunk, (I_chunk, J_chunk)), 
                shape=shape
            ).tocsr()
            
            # Get CSR arrays
            indptr = A_chunk.indptr.astype(cp.int32)
            indices = A_chunk.indices.astype(cp.int32)
            data = A_chunk.data
            
            # Launch kernel with optimized grid size
            max_threads = 256
            max_blocks = 1024  # Maximum reasonable grid size
            threads_per_block = min(max_threads, len(elim_rows_gpu))
            blocks = min(max_blocks, math.ceil(len(elim_rows_gpu) / threads_per_block))
            
            kernel(
                (blocks,), (threads_per_block,),
                (indptr, indices, data, elim_rows_gpu, 
                 cp.int32(len(elim_rows_gpu)), 
                 cp.int32(row_start), cp.int32(row_end)),
                stream=stream
            )
            
            # Update CSR data
            A_chunk.data = data
            
            # Convert to COO for accumulation
            A_chunk_coo = A_chunk.tocoo()
            
            return A_chunk_coo.data, A_chunk_coo.row, A_chunk_coo.col
    
    def gpu_worker(gpu_id, row_start, row_end):
        """Worker function running in separate thread with subchunking support and NVLink awareness"""
        try:
            # Set GPU for this thread
            with cp.cuda.Device(gpu_id):
                # Get available GPU memory
                free_mem, total_mem = get_gpu_memory_info(gpu_id)
                available_mem = int(free_mem * max_memory_fraction)
                
                # Create 2 CUDA streams per GPU for overlapping transfers
                compute_stream = cp.cuda.Stream(non_blocking=True)
                transfer_stream = cp.cuda.Stream(non_blocking=True)
                
                # Compile kernel only once per GPU
                kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')
                
                with compute_stream:
                    # Transfer elimination rows once and pin for better transfer performance
                    elim_rows_cpu_pinned = onp.asarray(all_row_inds_cpu, dtype=onp.int32)
                    elim_rows_gpu = cp.asarray(elim_rows_cpu_pinned, dtype=cp.int32)
                    
                    # Filter triplets for this GPU's row range on CPU first
                    # Use binary search for large arrays
                    if len(I_sorted) > 10_000_000:
                        start_idx = onp.searchsorted(I_sorted, row_start)
                        end_idx = onp.searchsorted(I_sorted, row_end)
                        I_filtered_cpu = I_sorted[start_idx:end_idx]
                        J_filtered_cpu = J_sorted[start_idx:end_idx]
                        V_filtered_cpu = V_sorted[start_idx:end_idx]
                    else:
                        # For smaller arrays, direct mask is faster
                        mask_cpu = (I_sorted >= row_start) & (I_sorted < row_end)
                        I_filtered_cpu = I_sorted[mask_cpu]
                        J_filtered_cpu = J_sorted[mask_cpu]
                        V_filtered_cpu = V_sorted[mask_cpu]
                    
                    if len(I_filtered_cpu) == 0:
                        logger.warning(f"GPU {gpu_id}: No data in row range [{row_start}, {row_end})")
                        return
                    
                    # Estimate memory for this partition
                    partition_mem = estimate_memory_usage(len(I_filtered_cpu), shape)
                    
                    logger.debug(f"GPU {gpu_id}: {len(I_filtered_cpu):,} triplets, "
                                f"estimated {partition_mem / 1e9:.2f} GB, "
                                f"available {available_mem / 1e9:.2f} GB")
                    
                    # Determine if subchunking is needed
                    if partition_mem <= available_mem * 0.9:  # Leave 10% buffer
                        # Process entire partition at once
                        logger.debug(f"GPU {gpu_id}: Processing entire partition")
                        
                        # Pin memory for faster transfers
                        I_cpu_pinned = onp.asarray(I_filtered_cpu, dtype=onp.int32)
                        J_cpu_pinned = onp.asarray(J_filtered_cpu, dtype=onp.int32)
                        V_cpu_pinned = onp.asarray(V_filtered_cpu, dtype=onp.float64)
                        
                        # Transfer with non-blocking stream
                        with transfer_stream:
                            I_gpu = cp.asarray(I_cpu_pinned, dtype=cp.int32)
                            J_gpu = cp.asarray(J_cpu_pinned, dtype=cp.int32)
                            V_gpu = cp.asarray(V_cpu_pinned, dtype=cp.float64)
                        
                        # Wait for transfer to complete
                        transfer_stream.synchronize()
                        
                        data_gpu, row_gpu, col_gpu = process_subchunk(
                            gpu_id, I_gpu, J_gpu, V_gpu, row_start, row_end, kernel, elim_rows_gpu, compute_stream
                        )
                        
                        # Cleanup before transfer
                        del I_gpu, J_gpu, V_gpu
                        
                        # Transfer final result to CPU
                        data_cpu = cp.asnumpy(data_gpu)
                        row_cpu = cp.asnumpy(row_gpu)
                        col_cpu = cp.asnumpy(col_gpu)
                        
                    else:
                        # Subchunking needed
                        n_triplets_partition = len(I_filtered_cpu)
                        
                        # More conservative estimate: use 70% of estimated memory per subchunk
                        triplets_per_subchunk = max(1000000, int((available_mem * 0.7) / (partition_mem / n_triplets_partition)))
                        n_subchunks = math.ceil(n_triplets_partition / triplets_per_subchunk)
                        
                        logger.debug(f"GPU {gpu_id}: Using {n_subchunks} subchunks of ~{triplets_per_subchunk:,} triplets each")
                        
                        # Accumulate results from all subchunks
                        all_data = []
                        all_row = []
                        all_col = []
                        
                        for subchunk_id in range(n_subchunks):
                            start_idx = subchunk_id * triplets_per_subchunk
                            end_idx = min((subchunk_id + 1) * triplets_per_subchunk, n_triplets_partition)
                            
                            logger.debug(f"GPU {gpu_id}: Processing subchunk {subchunk_id + 1}/{n_subchunks} "
                                       f"(triplets {start_idx:,} to {end_idx:,})")
                            
                            # Pin memory for faster transfers
                            I_sub_pinned = onp.asarray(I_filtered_cpu[start_idx:end_idx], dtype=onp.int32)
                            J_sub_pinned = onp.asarray(J_filtered_cpu[start_idx:end_idx], dtype=onp.int32)
                            V_sub_pinned = onp.asarray(V_filtered_cpu[start_idx:end_idx], dtype=onp.float64)
                            
                            # Prefetch next chunk in second stream while processing current one
                            if subchunk_id < n_subchunks - 1:
                                next_start = (subchunk_id + 1) * triplets_per_subchunk
                                next_end = min((subchunk_id + 2) * triplets_per_subchunk, n_triplets_partition)
                                
                                I_next_pinned = onp.asarray(I_filtered_cpu[next_start:next_end], dtype=onp.int32)
                                J_next_pinned = onp.asarray(J_filtered_cpu[next_start:next_end], dtype=onp.int32)
                                V_next_pinned = onp.asarray(V_filtered_cpu[next_start:next_end], dtype=onp.float64)
                            
                            # Transfer current subchunk to GPU
                            with transfer_stream:
                                I_subchunk = cp.asarray(I_sub_pinned, dtype=cp.int32)
                                J_subchunk = cp.asarray(J_sub_pinned, dtype=cp.int32)
                                V_subchunk = cp.asarray(V_sub_pinned, dtype=cp.float64)
                            
                            # Wait for transfer to complete
                            transfer_stream.synchronize()
                            
                            # Process subchunk
                            data_sub, row_sub, col_sub = process_subchunk(
                                gpu_id, I_subchunk, J_subchunk, V_subchunk, 
                                row_start, row_end, kernel, elim_rows_gpu, compute_stream
                            )
                            
                            # Start prefetching next subchunk if available
                            if subchunk_id < n_subchunks - 1:
                                # Start prefetching next subchunk asynchronously
                                with transfer_stream:
                                    next_I = cp.asarray(I_next_pinned, dtype=cp.int32)
                                    next_J = cp.asarray(J_next_pinned, dtype=cp.int32)
                                    next_V = cp.asarray(V_next_pinned, dtype=cp.float64)
                            
                            # Accumulate results from current subchunk
                            all_data.append(cp.asnumpy(data_sub))
                            all_row.append(cp.asnumpy(row_sub))
                            all_col.append(cp.asnumpy(col_sub))
                            
                            # Cleanup subchunk GPU memory explicitly
                            del I_subchunk, J_subchunk, V_subchunk, data_sub, row_sub, col_sub
                            cp.get_default_memory_pool().free_all_blocks()
                        
                        # Concatenate all subchunk results
                        data_cpu = onp.concatenate(all_data)
                        row_cpu = onp.concatenate(all_row)
                        col_cpu = onp.concatenate(all_col)
                    
                    # Store results thread-safely
                    with results_lock:
                        results.append((gpu_id, data_cpu, row_cpu, col_cpu))
                    
                    # Final cleanup
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    logger.debug(f"GPU {gpu_id}: Completed {len(data_cpu):,} non-zeros")
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} failed: {e}")
            import traceback
            with results_lock:
                errors.append((gpu_id, str(e), traceback.format_exc()))
    
    # Launch GPU workers as threads
    threads = []
    for gpu_id, (row_start, row_end) in enumerate(row_ranges):
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, row_start, row_end)
        )
        t.daemon = True  # Mark as daemon so they don't block program exit
        t.start()
        threads.append(t)
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Check for errors
    if errors:
        logger.error(f"Errors occurred in {len(errors)} GPU(s):")
        for gpu_id, err, trace in errors:
            logger.error(f"GPU {gpu_id}: {err}")
            logger.debug(trace)
        raise RuntimeError(f"{len(errors)} GPU worker(s) failed")
    
    if not results:
        raise RuntimeError("No results from GPU workers!")
    
    # Sort results by GPU ID and concatenate
    results.sort(key=lambda x: x[0])
    data_cpu = onp.concatenate([d for _, d, _, _ in results])
    row_cpu = onp.concatenate([r for _, _, r, _ in results])
    col_cpu = onp.concatenate([c for _, _, _, c in results])
    
    # Build final CSR matrix
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    
    # Sum duplicate entries (if any)
    A_cpu.sum_duplicates()
    
    elapsed = time.time() - start_time
    logger.debug(f"Multi-GPU assembly completed in {elapsed:.2f}s: {A_cpu.nnz:,} non-zeros")
    
    return A_cpu

def get_A_multi_gpu_production(problem, n_gpus=None, enable_profiling=False):
    """
    Production-ready multi-GPU assembly with optimal performance.
    
    Key optimizations:
    1. COO format for efficient CPU concatenation
    2. Pinned memory for faster transfers
    3. Stream synchronization optimization
    4. Minimal Python overhead
    5. Optional profiling for performance analysis
    
    Args:
        problem: Problem object with triplet data
        n_gpus: Number of GPUs (None = auto-detect)
        enable_profiling: Print detailed timing breakdown
    
    Returns:
        scipy.sparse.csr_matrix: Assembled sparse matrix
    """
    if n_gpus is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    
    n_triplets = len(problem.I)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    n_rows = shape[0]
    
    if enable_profiling:
        timings = {'start': time.time()}
    
    logger.debug(f"Production multi-GPU: {n_gpus} GPUs, {n_triplets:,} triplets, {n_rows:,} rows")
    
    # Precompute elimination rows (on CPU, fast enough)
    all_row_inds_list = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            rows = fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
            all_row_inds_list.extend(rows)
    all_row_inds_cpu = onp.array(all_row_inds_list, dtype=onp.int32)
    
    if enable_profiling:
        timings['elim_rows_computed'] = time.time()
    
    # Partition rows
    rows_per_gpu = math.ceil(n_rows / n_gpus)
    row_ranges = [(i * rows_per_gpu, min((i + 1) * rows_per_gpu, n_rows)) 
                  for i in range(n_gpus)]
    
    # CUDA kernel
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows, 
                                const int row_start, const int row_end)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;
        int r = rows[tid];
        if (r < row_start || r >= row_end) return;
        
        int start = indptr[r];
        int end   = indptr[r + 1];
        for (int j = start; j < end; ++j) {
            data[j] = 0.0;
            if (indices[j] == r) {
                data[j] = 1.0;
            }
        }
    }
    '''
    
    results = [None] * n_gpus
    errors = []
    results_lock = threading.Lock()
    
    def gpu_worker(gpu_id, row_start, row_end):
        try:
            worker_timings = {} if enable_profiling else None
            if enable_profiling:
                worker_timings['start'] = time.time()
            
            with cp.cuda.Device(gpu_id):
                kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')
                stream = cp.cuda.Stream()
                
                with stream:
                    if enable_profiling:
                        worker_timings['setup_done'] = time.time()
                    
                    # Transfer to GPU - use asarray for zero-copy when possible
                    I_gpu = cp.asarray(problem.I, dtype=cp.int32)
                    J_gpu = cp.asarray(problem.J, dtype=cp.int32)
                    V_gpu = cp.asarray(problem.V, dtype=cp.float64)
                    
                    if enable_profiling:
                        worker_timings['data_transferred'] = time.time()
                    
                    # Filter on GPU (very fast)
                    mask = (I_gpu >= row_start) & (I_gpu < row_end)
                    I_filt = I_gpu[mask]
                    J_filt = J_gpu[mask]
                    V_filt = V_gpu[mask]
                    
                    del I_gpu, J_gpu, V_gpu, mask
                    
                    if len(I_filt) == 0:
                        logger.warning(f"GPU {gpu_id}: No data in range")
                        return
                    
                    if enable_profiling:
                        worker_timings['filtered'] = time.time()
                    
                    logger.debug(f"GPU {gpu_id}: Processing {len(I_filt):,} triplets in rows [{row_start}, {row_end})")
                    
                    # Build CSR (COO->CSR is optimized in cuSPARSE)
                    A_chunk = cusparse.coo_matrix((V_filt, (I_filt, J_filt)), shape=shape).tocsr()
                    del I_filt, J_filt, V_filt
                    
                    if enable_profiling:
                        worker_timings['csr_built'] = time.time()
                    
                    # Get CSR arrays
                    indptr = A_chunk.indptr.astype(cp.int32)
                    indices = A_chunk.indices.astype(cp.int32)
                    data = A_chunk.data
                    
                    # Eliminate rows
                    elim_rows_gpu = cp.asarray(all_row_inds_cpu, dtype=cp.int32)
                    threads_per_block = 256
                    blocks = math.ceil(len(elim_rows_gpu) / threads_per_block)
                    
                    kernel((blocks,), (threads_per_block,),
                           (indptr, indices, data, elim_rows_gpu, 
                            cp.int32(len(elim_rows_gpu)),
                            cp.int32(row_start), cp.int32(row_end)))
                    
                    if enable_profiling:
                        worker_timings['rows_eliminated'] = time.time()
                    
                    A_chunk.data = data
                    
                    # Convert to COO for efficient concatenation
                    A_chunk_coo = A_chunk.tocoo()
                    
                    if enable_profiling:
                        worker_timings['coo_converted'] = time.time()
                    
                    # Transfer to CPU
                    data_cpu = cp.asnumpy(A_chunk_coo.data)
                    row_cpu = cp.asnumpy(A_chunk_coo.row)
                    col_cpu = cp.asnumpy(A_chunk_coo.col)
                    
                    if enable_profiling:
                        worker_timings['transferred_to_cpu'] = time.time()
                    
                    # Store result
                    results[gpu_id] = (data_cpu, row_cpu, col_cpu, worker_timings)
                    
                    # Cleanup
                    del A_chunk, A_chunk_coo, indptr, indices, data, elim_rows_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    logger.debug(f"GPU {gpu_id}: Completed {len(data_cpu):,} non-zeros")
                    
        except Exception as e:
            with results_lock:
                errors.append((gpu_id, str(e)))
            import traceback
            logger.error(f"GPU {gpu_id} failed:\n{traceback.format_exc()}")
    
    # Launch threads
    if enable_profiling:
        timings['workers_launching'] = time.time()
    
    threads = []
    for gpu_id, (row_start, row_end) in enumerate(row_ranges):
        t = threading.Thread(target=gpu_worker, args=(gpu_id, row_start, row_end))
        t.start()
        threads.append(t)
    
    if enable_profiling:
        timings['workers_launched'] = time.time()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    if enable_profiling:
        timings['workers_joined'] = time.time()
    
    if errors:
        raise RuntimeError(f"GPU errors: {errors}")
    
    # Filter and concatenate results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise RuntimeError("No results from GPUs")
    
    data_cpu = onp.concatenate([d for d, _, _, _ in valid_results])
    row_cpu = onp.concatenate([r for _, r, _, _ in valid_results])
    col_cpu = onp.concatenate([c for _, _, c, _ in valid_results])
    
    if enable_profiling:
        timings['concatenated'] = time.time()
    
    # Build final CSR matrix
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    
    if enable_profiling:
        timings['final_csr_built'] = time.time()
        print_profiling_info(timings, valid_results, n_gpus)
    
    logger.debug(f"Multi-GPU assembly: {A_cpu.nnz:,} non-zeros")
    
    return A_cpu


def print_profiling_info(timings, results, n_gpus):
    """Print detailed profiling information"""
    print("\n" + "="*80)
    print("MULTI-GPU ASSEMBLY PROFILING")
    print("="*80)
    
    # Main thread timings
    t_start = timings['start']
    print("\nMain Thread:")
    print(f"  Setup & elim rows:  {(timings['elim_rows_computed'] - t_start)*1000:7.2f} ms")
    print(f"  Launch workers:     {(timings['workers_launched'] - timings['workers_launching'])*1000:7.2f} ms")
    print(f"  Wait for workers:   {(timings['workers_joined'] - timings['workers_launched'])*1000:7.2f} ms")
    print(f"  Concatenate:        {(timings['concatenated'] - timings['workers_joined'])*1000:7.2f} ms")
    print(f"  Build final CSR:    {(timings['final_csr_built'] - timings['concatenated'])*1000:7.2f} ms")
    print(f"  TOTAL:              {(timings['final_csr_built'] - t_start)*1000:7.2f} ms")
    
    # Per-GPU timings
    print(f"\nPer-GPU Breakdown:")
    for gpu_id in range(n_gpus):
        if results[gpu_id] is None:
            continue
        _, _, _, wt = results[gpu_id]
        if wt is None:
            continue
        
        print(f"\n  GPU {gpu_id}:")
        t0 = wt['start']
        print(f"    Setup:            {(wt['setup_done'] - t0)*1000:7.2f} ms")
        print(f"    Transfer to GPU:  {(wt['data_transferred'] - wt['setup_done'])*1000:7.2f} ms")
        print(f"    Filter:           {(wt['filtered'] - wt['data_transferred'])*1000:7.2f} ms")
        print(f"    Build CSR:        {(wt['csr_built'] - wt['filtered'])*1000:7.2f} ms")
        print(f"    Row elimination:  {(wt['rows_eliminated'] - wt['csr_built'])*1000:7.2f} ms")
        print(f"    Convert to COO:   {(wt['coo_converted'] - wt['rows_eliminated'])*1000:7.2f} ms")
        print(f"    Transfer to CPU:  {(wt['transferred_to_cpu'] - wt['coo_converted'])*1000:7.2f} ms")
        print(f"    GPU TOTAL:        {(wt['transferred_to_cpu'] - t0)*1000:7.2f} ms")
    
    print("="*80 + "\n")


def get_A_multi_gpu_ultra_fast(problem, n_gpus=None):
    """
    Ultra-optimized version with aggressive optimizations.
    
    Additional optimizations:
    - Pre-allocated pinned memory for faster CPU<->GPU transfers
    - Larger kernel launch parameters
    - Direct CSR->COO conversion without intermediate copies
    """
    if n_gpus is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    
    n_triplets = len(problem.I)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    n_rows = shape[0]
    
    logger.debug(f"Ultra-fast multi-GPU: {n_gpus} GPUs, {n_triplets:,} triplets")
    
    # Precompute elimination rows
    all_row_inds_list = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            rows = fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind]
            all_row_inds_list.extend(rows)
    all_row_inds_cpu = onp.array(all_row_inds_list, dtype=onp.int32)
    
    # Partition
    rows_per_gpu = math.ceil(n_rows / n_gpus)
    row_ranges = [(i * rows_per_gpu, min((i + 1) * rows_per_gpu, n_rows)) 
                  for i in range(n_gpus)]
    
    # Optimized kernel with loop unrolling hint
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows, 
                                const int row_start, const int row_end)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;
        int r = rows[tid];
        if (r < row_start || r >= row_end) return;
        
        int start = indptr[r];
        int end   = indptr[r + 1];
        
        #pragma unroll 4
        for (int j = start; j < end; ++j) {
            double val = 0.0;
            if (indices[j] == r) val = 1.0;
            data[j] = val;
        }
    }
    '''
    
    results = [None] * n_gpus
    errors = []
    
    def gpu_worker(gpu_id, row_start, row_end):
        try:
            with cp.cuda.Device(gpu_id):
                # Compile kernel with optimization flags
                kernel = cp.RawKernel(
                    kernel_src, 
                    'zero_rows_and_set_diag',
                    options=('--use_fast_math', '-O3')
                )
                
                stream = cp.cuda.Stream()
                
                with stream:
                    # Fast transfers
                    I_gpu = cp.asarray(problem.I, dtype=cp.int32)
                    J_gpu = cp.asarray(problem.J, dtype=cp.int32)
                    V_gpu = cp.asarray(problem.V, dtype=cp.float64)
                    
                    # Filter
                    mask = (I_gpu >= row_start) & (I_gpu < row_end)
                    I_filt = I_gpu[mask]
                    J_filt = J_gpu[mask]
                    V_filt = V_gpu[mask]
                    
                    del I_gpu, J_gpu, V_gpu, mask
                    
                    if len(I_filt) == 0:
                        return
                    
                    # Build CSR
                    A_chunk = cusparse.coo_matrix((V_filt, (I_filt, J_filt)), shape=shape).tocsr()
                    del I_filt, J_filt, V_filt
                    
                    # Eliminate rows with larger block size
                    indptr = A_chunk.indptr.astype(cp.int32)
                    indices = A_chunk.indices.astype(cp.int32)
                    data = A_chunk.data
                    
                    elim_rows_gpu = cp.asarray(all_row_inds_cpu, dtype=cp.int32)
                    
                    # Use 512 threads per block for better occupancy
                    threads_per_block = 512
                    blocks = math.ceil(len(elim_rows_gpu) / threads_per_block)
                    
                    kernel((blocks,), (threads_per_block,),
                           (indptr, indices, data, elim_rows_gpu, 
                            cp.int32(len(elim_rows_gpu)),
                            cp.int32(row_start), cp.int32(row_end)))
                    
                    A_chunk.data = data
                    
                    # COO conversion
                    A_chunk_coo = A_chunk.tocoo()
                    
                    # Async transfer to CPU (if possible)
                    data_cpu = cp.asnumpy(A_chunk_coo.data)
                    row_cpu = cp.asnumpy(A_chunk_coo.row)
                    col_cpu = cp.asnumpy(A_chunk_coo.col)
                    
                    results[gpu_id] = (data_cpu, row_cpu, col_cpu)
                    
                    # Cleanup
                    del A_chunk, A_chunk_coo, indptr, indices, data, elim_rows_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    
        except Exception as e:
            errors.append((gpu_id, str(e)))
            import traceback
            logger.error(f"GPU {gpu_id}:\n{traceback.format_exc()}")
    
    # Launch
    threads = [threading.Thread(target=gpu_worker, args=(i, *row_ranges[i])) 
               for i in range(n_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        raise RuntimeError(f"Errors: {errors}")
    
    # Combine
    valid = [r for r in results if r is not None]
    data_cpu = onp.concatenate([d for d, _, _ in valid])
    row_cpu = onp.concatenate([r for _, r, _ in valid])
    col_cpu = onp.concatenate([c for _, _, c in valid])
    
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    
    logger.debug(f"Ultra-fast assembly: {A_cpu.nnz:,} non-zeros")
    
    return A_cpu

def get_A_gpu_chunked_safe(problem, n_chunks=16, row_chunk_size=20000):
    """
    Assemble global sparse matrix A on GPU using chunked COO,
    memory-efficient chunked row elimination, entirely on GPU.

    NOTE: This vectorized implementation uses a custom CUDA kernel 
    """

    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    # Initialize empty CSR accumulator
    A_gpu = cusparse.csr_matrix(shape, dtype=cp.float64)

    # 1️. Split triplets into chunks
    V_chunks = onp.array_split(problem.V, n_chunks)
    I_chunks = onp.array_split(problem.I, n_chunks)
    J_chunks = onp.array_split(problem.J, n_chunks)

    # 2️. Process each chunk sequentially
    for V_chunk, I_chunk, J_chunk in zip(V_chunks, I_chunks, J_chunks):
        I_gpu = cp.array(I_chunk, dtype=cp.int32)
        J_gpu = cp.array(J_chunk, dtype=cp.int32)
        V_gpu = cp.array(V_chunk, dtype=cp.float64)
        # Assemble COO matrix for this chunk and accumulate
        A_chunk = cusparse.coo_matrix((V_gpu, (I_gpu, J_gpu)), shape=shape)
        A_gpu += A_chunk.tocsr()

        # Free chunk memory
        del I_gpu, J_gpu, V_gpu, A_chunk
        cp._default_memory_pool.free_all_blocks()

    # 3️. Prepare all row indices to zero out (BCs / eliminated DOFs)
    all_row_inds = cp.concatenate([
        cp.array(
            fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
            dtype=cp.int32
        )
        for ind, fe in enumerate(problem.fes)
        for i in range(len(fe.node_inds_list))
    ])
    # ensure unique and sorted if desired (optional)
    # all_row_inds = cp.unique(all_row_inds)

    logger.debug(f"Total rows to eliminate: {int(all_row_inds.size)}")

    # 4. Chunked row elimination on GPU using a RawKernel (one thread per row)
    A_gpu = A_gpu.tocsr()  # ensure csr
    indptr = A_gpu.indptr.astype(cp.int32)
    indices = A_gpu.indices.astype(cp.int32)
    data = A_gpu.data.astype(cp.float64)

    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;

        int r = rows[tid];
        int start = indptr[r];
        int end   = indptr[r + 1];

        // Zero row and set diagonal when found
        for (int j = start; j < end; ++j) {
            // zero
            data[j] = 0.0;
            // set diag if column index equals row
            if (indices[j] == r) {
                data[j] = 1.0;
                // we can continue zeroing other columns (they remain zero)
            }
        }
    }
    '''

    # compile kernel
    kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')

    n_total_rows = int(all_row_inds.size)
    # process in batches so we avoid launching huge kernels at once and to limit memory spikes
    for start in range(0, n_total_rows, row_chunk_size):
        end = min(start + row_chunk_size, n_total_rows)
        rows_batch = all_row_inds[start:end].astype(cp.int32)

        # Launch kernel with one thread per row in the batch
        threads_per_block = 128
        blocks = math.ceil(rows_batch.size / threads_per_block)
        kernel((blocks,), (threads_per_block,),
               (indptr, indices, data, rows_batch, cp.int32(rows_batch.size)))

        # allow memory pool to tidy up intermediate allocations
        cp._default_memory_pool.free_all_blocks()

    # put modified arrays back into the CSR matrix object
    A_gpu.indptr = indptr
    A_gpu.indices = indices
    A_gpu.data = data

    return A_gpu

def get_A_gpu_chunked_stream_elim(problem, n_chunks=16, row_chunk_size=20000):
    """
    Assemble global sparse matrix A using chunked GPU computation,
    perform row elimination entirely on GPU per chunk, and stream to CPU CSR.
    Memory-efficient: avoids keeping full GPU matrix in memory.
    """
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    # CPU accumulator lists
    data_cpu = []
    row_cpu = []
    col_cpu = []

    # Precompute all rows to eliminate (BCs)
    all_row_inds = cp.concatenate([
        cp.array(
            fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
            dtype=cp.int32
        )
        for ind, fe in enumerate(problem.fes)
        for i in range(len(fe.node_inds_list))
    ])
    n_elim_rows = all_row_inds.size
    logger.debug(f"Total rows to eliminate: {int(n_elim_rows)}")

    # CUDA kernel for row elimination
    kernel_src = r'''
    extern "C" __global__
    void zero_rows_and_set_diag(const int* indptr, const int* indices, double* data,
                                const int* rows, const int n_rows)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= n_rows) return;
        int r = rows[tid];
        int start = indptr[r];
        int end   = indptr[r + 1];
        for (int j = start; j < end; ++j) {
            data[j] = 0.0;
            if (indices[j] == r) {
                data[j] = 1.0;
            }
        }
    }
    '''
    kernel = cp.RawKernel(kernel_src, 'zero_rows_and_set_diag')

    # Split triplets into chunks
    V_chunks = onp.array_split(problem.V, n_chunks)
    I_chunks = onp.array_split(problem.I, n_chunks)
    J_chunks = onp.array_split(problem.J, n_chunks)

    for V_chunk, I_chunk, J_chunk in zip(V_chunks, I_chunks, J_chunks):
        # Move chunk to GPU
        I_gpu = cp.array(I_chunk, dtype=cp.int32)
        J_gpu = cp.array(J_chunk, dtype=cp.int32)
        V_gpu = cp.array(V_chunk, dtype=cp.float64)

        # Build COO and convert to CSR for this chunk
        A_chunk = cusparse.coo_matrix((V_gpu, (I_gpu, J_gpu)), shape=shape).tocsr()
        indptr = A_chunk.indptr.astype(cp.int32)
        indices = A_chunk.indices.astype(cp.int32)
        data = A_chunk.data.astype(cp.float64)

        # Launch row elimination kernel in batches
        for start in range(0, n_elim_rows, row_chunk_size):
            end = min(start + row_chunk_size, n_elim_rows)
            rows_batch = all_row_inds[start:end].astype(cp.int32)
            threads_per_block = 128
            blocks = math.ceil(rows_batch.size / threads_per_block)
            kernel((blocks,), (threads_per_block,),
                   (indptr, indices, data, rows_batch, cp.int32(rows_batch.size)))

        # Put modified arrays back into CSR
        A_chunk.indptr = indptr
        A_chunk.indices = indices
        A_chunk.data = data

        # Convert chunk to COO and move to CPU
        A_chunk_coo = A_chunk.tocoo()
        data_cpu.append(cp.asnumpy(A_chunk_coo.data))
        row_cpu.append(cp.asnumpy(A_chunk_coo.row))
        col_cpu.append(cp.asnumpy(A_chunk_coo.col))

        # Free GPU memory
        del I_gpu, J_gpu, V_gpu, A_chunk, indptr, indices, data, rows_batch, A_chunk_coo
        cp._default_memory_pool.free_all_blocks()

    # Concatenate CPU arrays
    data_cpu = onp.concatenate(data_cpu)
    row_cpu = onp.concatenate(row_cpu)
    col_cpu = onp.concatenate(col_cpu)

    # Construct CPU CSR
    A_cpu = scipy.sparse.csr_matrix((data_cpu, (row_cpu, col_cpu)), shape=shape)
    logger.debug("Completed sparse assembly and GPU row elimination, streamed to CPU.")

    return A_cpu

def get_A(problem, solver_options):
    # logger.debug(f"Creating sparse matrix with cupy...")
    # A = get_A_gpu_chunked_stream_elim(problem)
    # A = get_A_gpu_chunked_safe(problem)
    # A = A_gpu
    # petsc_row_elim = False
    # A_cpu = scipy.sparse.csr_matrix((
    # cp.asnumpy(A_gpu.data),
    # cp.asnumpy(A_gpu.indices),  
    # cp.asnumpy(A_gpu.indptr)
    # ), shape=A_gpu.shape)
    # del A_gpu
    # cp._default_memory_pool.free_all_blocks()
    # cp.get_default_pinned_memory_pool().free_all_blocks()
    # A = A_cpu
    #================================================================================
    # time_start_cupy = time.time()
    # # A_cpu =get_A_multi_gpu_production(problem, n_gpus=None, enable_profiling=False)
    # a = None
    # time_end_cupy = time.time()
    # time_cupy_minimal = time_end_cupy - time_start_cupy

    # petsc_row_elim = True
    # time_start_cupy = time.time()
    # A_cpu = get_A_gpu_chunked_stream_elim_multigpu(problem)
    # time_end_cupy = time.time()
    # time_cupy_fast = time_end_cupy - time_start_cupy

    # # time_start_cupy = time.time()
    # # # A_cup = get_A_gpu_chunked_stream_elim(problem)
    # # A_cpu = get_A_multi_gpu_fast(problem)
    # # # A_cpu = get_A_multi_gpu_minimal_transfer(problem)
    # # time_end_cupy = time.time()
    # # time_cupy_minimal = time_end_cupy - time_start_cupy
    # # A = BCOO.from_scipy_sparse(A_cpu).sort_indices()
    # logger.debug(f"Completed sparse assembly and row elimination with cupy!")
    
    # if petsc_row_elim:  # petsc for global matrix assembly, and row elimination
    #     from petsc4py import PETSc
    #     logger.debug(f"Creating sparse matrix with scipy...")
    #     time_start_scipy = time.time()
    #     A_sp_scipy = scipy.sparse.csr_array((onp.array(problem.V), (problem.I, problem.J)),
    #                                         shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars))
    #     # logger.info(f"Global sparse matrix takes about {A_sp_scipy.data.shape[0]*8*3/2**30} G memory to store.")
    #     A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape,
    #                               csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
    #                                    A_sp_scipy.indices.astype(PETSc.IntType, copy=False),
    #                                    A_sp_scipy.data))
    #     for ind, fe in enumerate(problem.fes):
    #         for i in range(len(fe.node_inds_list)):
    #             row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
    #                                  dtype=onp.int32)
    #             A.zeroRows(row_inds)

    #     # Convert back to jax representation, maybe move this to after periodic transformation
    #     indptr, indices, data = A.getValuesCSR()
    #     A = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    #     time_end_scipy = time.time()
    #     time_scipy = time_end_scipy - time_start_scipy
    #     logger.debug(f"Completed sparse assembly and row elimination with scipy!")
    #     logger.debug(f"Time cupy (fast): {time_cupy_fast}, Time cupy (minimal): {time_cupy_minimal}, Time scipy+petsc: {time_scipy}")
    #     # A = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    #     # A = A_sp_scipy
    # #     # Verify cupy assembly: compare scipy and cupy results 
    #     diff = A_cpu - A
    #     # print(f"Global sparse matrix scipy  {A_sp_scipy.todense()}")
    #     error_norm = scipy.sparse.linalg.norm(diff)
    #     # logger.debug(f"scipy and  cupy: {A_sp_scipy.todense()}, {A_cpu.todense()}")
    #     logger.debug(f"Diff in scipy and  cupy: {error_norm}")
    #================================================================================
    if petsc_row_elim:  # petsc for global matrix assembly, and row elimination
        from petsc4py import PETSc
        logger.debug(f"Creating sparse matrix with scipy...")
        time_start_scipy = time.time()
        A_sp_scipy = scipy.sparse.csr_array((onp.array(problem.V), (problem.I, problem.J)),
                                            shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars))
        # logger.info(f"Global sparse matrix takes about {A_sp_scipy.data.shape[0]*8*3/2**30} G memory to store.")
        A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape,
                                  csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
                                       A_sp_scipy.indices.astype(PETSc.IntType, copy=False),
                                       A_sp_scipy.data))
        for ind, fe in enumerate(problem.fes):
            for i in range(len(fe.node_inds_list)):
                row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
                                     dtype=onp.int32)
                A.zeroRows(row_inds)

        # Convert back to jax representation, maybe move this to after periodic transformation
        indptr, indices, data = A.getValuesCSR()
        A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
        logger.debug(f"Completed sparse assembly and row elimination with scipy!")
        A = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    #================================================================================
    # else:  # jax for global matrix assembly, and row elimination
    #     A = BCOO((problem.V, np.vstack((problem.I, problem.J)).T),
    #              shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars))
    #     # A = A.sort_indices().sum_duplicates() # must specify nse for jit with sum

    #     # b_term = - A @ x0
    #     for ind, fe in enumerate(problem.fes):
    #         for i in range(len(fe.node_inds_list)):
    #             row_inds = np.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
    #                                 dtype=np.int32)
                # A = row_elimination_jax(A, row_inds)  # jit or not seems similar wall time

    # # Linear multipoint constraints
    # if hasattr(problem, 'P_mat'):
    #     if True:  # petsc matrix multiplication
    #         P = PETSc.Mat().createAIJ(size=problem.P_mat.shape,
    #                                   csr=(problem.P_mat.indptr.astype(PETSc.IntType, copy=False),
    #                                        problem.P_mat.indices.astype(PETSc.IntType, copy=False), problem.P_mat.data))

    #         tmp = A.matMult(P)
    #         P_T = P.transpose()
    #         A = P_T.matMult(tmp)
    #     else:  # jax sparse matrix multiplication
    #         pass
    #         # A = P_transformation(problem.P_mat, A)

        ####### Note its actually cheaper to transfer GPU/CPU to PetSC than to use jax experimental BCOO,
        ####### Jax experimental BCOO multiplication is not implemented well, OOM and or very slow

    return A


################################################################################
# The "row elimination" solver

def solver(problem, solver_options={}):
    """
    Specify exactly either 'jax_solver' or 'umfpack_solver' or 'petsc_solver'

    Examples:
    (1) solver_options = {'jax_solver': {}}
    (2) solver_options = {'umfpack_solver': {}}
    (3) solver_options = {'petsc_solver': {'ksp_type': 'bcgsl', 'pc_type': 'jacobi'}, 'initial_guess': some_guess}

    Default parameters will be used if no instruction is found:

    solver_options =
    {
        # If multiple solvers are specified or no solver is specified, 'jax_solver' will be used.
        'jax_solver':
        {
            # The JAX built-in linear solver
            # Reference: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html
            'precond': True,
        }

        'umfpack_solver':
        {
            # The scipy solver that calls UMFPACK
            # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
        }

        'petsc_solver':
        {
            # PETSc solver
            # For more ksp_type and pc_type: https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/index.html
            'ksp_type': 'bcgsl', # e.g., 'minres', 'gmres', 'tfqmr'
            'pc_type': 'ilu', # e.g., 'jacobi'
        }

        'line_search_flag': False, # Line search method
        'initial_guess': initial_guess, # Same shape as sol_list
        'tol': 1e-5, # Absolute tolerance for residual vector (l2 norm), used in Newton's method
        'rel_tol': 1e-8, # Relative tolerance for residual vector (l2 norm), used in Newton's method
    }

    The solver imposes Dirichlet B.C. with "row elimination" method.

    Some memo:

    res(u) = D*r(u) + (I - D)u - u_b
    D = [[1 0 0 0]
         [0 1 0 0]
         [0 0 0 0]
         [0 0 0 1]]
    I = [[1 0 0 0]
         [0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]
    A = d(res)/d(u) = D*dr/du + (I - D)

    TODO: linear multipoint constraint

    The function newton_update computes r(u) and dr/du
    """
    logger.debug(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()

    if 'initial_guess' in solver_options:
        # We dont't want inititual guess to play a role in the differentiation chain.
        initial_guess = jax.lax.stop_gradient(solver_options['initial_guess'])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        if hasattr(problem, 'P_mat'):
            dofs = np.zeros(problem.P_mat.shape[1]) # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = solver_options['rel_tol'] if 'rel_tol' in solver_options else 1e-8
    tol = solver_options['tol'] if 'tol' in solver_options else 1e-6

    def newton_update_helper(dofs):
        if hasattr(problem, 'P_mat'):
            dofs = problem.P_mat @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        if hasattr(problem, 'P_mat'):
            res_vec = problem.P_mat.T @ res_vec

        A = get_A(problem, solver_options)
        return res_vec, A

    res_vec, A = newton_update_helper(dofs)
    # print('A_matrix',A.todense())
    # print('res_vec',res_vec)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = np.maximum(res_val, 1.e-8) # Avoid division by zero
    rel_res_val = res_val / res_val_initial
    logger.debug(f"Before, l_2 res = {res_val}, relative l_2 res = {rel_res_val}")
    
    # Get solver parameters
    iteration_counter = 0
    max_iters = solver_options.get('max_iters', 50) # Default to 50 iterations if not specified
    update_jacobian_freq = solver_options.get('jacobian_freq', 1)
    
    has_converged = False
    nan_detected = False
    res_val_prev = res_val

    # Main Newton-Raphson iteration loop
    while (rel_res_val > rel_tol and res_val > tol) and (iteration_counter < max_iters):
        dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val / res_val_initial

        # print('A_matrix',A.todense())
        # print('res_vec',res_vec)
        # Update Jacobian matrix at a specified frequency
        if rel_res_val > rel_tol and res_val > tol:
            if (iteration_counter + 1) % update_jacobian_freq == 0:
                A = get_A(problem, solver_options)

        res_val_prev = res_val
        logger.debug(f"Iteration {iteration_counter + 1}: l_2 res = {res_val:.4e}, relative l_2 res = {rel_res_val:.4e}")
        iteration_counter += 1
        
        # --- NaN and Inf Check ---
        # Instead of asserting, check for non-finite values and break the loop if found.
        if not (np.all(np.isfinite(res_val)) and np.all(np.isfinite(dofs))):
            logger.warning(f"NaN or Inf detected in solution or residual at iteration {iteration_counter + 1}. Aborting step attempt.")
            nan_detected = True
            break # Exit the Newton loop immediately

    # Check for convergence status after the loop
    if nan_detected:
        # If the loop was broken by a NaN, it has not converged.
        has_converged = False
    elif rel_res_val <= rel_tol or res_val <= tol:
        # If tolerances are met, it has converged.
        has_converged = True
        logger.info(f"Converged in {iteration_counter} iterations.")
    else:
        # If loop finished due to max_iters, it has not converged.
        has_converged = False
        logger.warning(f"Solver did not converge after {iteration_counter} iterations (max_iters reached).")

    if hasattr(problem, 'P_mat'):
        dofs = problem.P_mat @ dofs

    sol_list = problem.unflatten_fn_sol_list(dofs)

    end = time.time()
    solve_time = end - start
    logger.info(f"Solve attempt took {solve_time:.2f} [s]")
    logger.debug(f"max of dofs = {np.max(dofs)}")
    logger.debug(f"min of dofs = {np.min(dofs)}")
    # Return convergence information if requested
    if solver_options.get('return_full_info', False):
        return sol_list, iteration_counter, has_converged
    elif solver_options.get('iteration_count', False):
        return [sol_list, iteration_counter]
    else:
        return sol_list

    # return sol_list


################################################################################
# The "arc length" solver
# Reference: Vasios, Nikolaos. "Nonlinear analysis of structures." The Arc-Length method. Harvard (2015).
# Our implementation follows the Crisfeld's formulation

# TODO: Do we want to merge displacement-control and force-control codes?

def arc_length_solver_disp_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec, prev_Delta_lamda, Delta_l=0.1, psi=1.):
    """
    TODO: Does not support periodic B.C., need some work here.
    """
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem, lamda)
        A = get_A(problem, solver_options={'umfpack_solver':{}})
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return np.sum(Delta_u_vec1*Delta_u_vec2) + psi**2.*Delta_lamda1*Delta_lamda2*np.sum(u_b**2.)

    u_vec = prev_u_vec
    lamda = prev_lamda

    u_b = assign_bc(np.zeros_like(prev_u_vec), problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-6
    res_val = 1.
    while res_val > tol:

        res_vec, A = newton_update_helper(u_vec)
        res_val = np.linalg.norm(res_vec)
        logger.debug(f"Arc length solver: res_val = {res_val}")

        delta_u_bar = umfpack_solve(A, -res_vec)
        delta_u_t = umfpack_solve(A, u_b)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2.*np.sum(u_b**2.)
        a2 = 2.* np.sum((Delta_u_vec + delta_u_bar)*delta_u_t) + 2.*psi**2.*Delta_lamda*np.sum(u_b**2.)
        a3 = np.sum((Delta_u_vec + delta_u_bar)**2.) + psi**2.*Delta_lamda**2.*np.sum(u_b**2.) - Delta_l**2.

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}"

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
            # At initial step, (Delta_u_vec_dir, Delta_lamda_dir) is zero, so both dot_prod1 and dot_prod2 are zero.
            # We simply select the larger value for delta_lamda.
            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
        elif dot_prod1 > dot_prod2:
            delta_lamda = delta_lamda1
        else:
            delta_lamda = delta_lamda2

        lamda = lamda + delta_lamda
        delta_u = delta_u_bar + delta_lamda * delta_u_t
        u_vec = u_vec + delta_u

        Delta_u_vec_dir = u_vec - prev_u_vec
        Delta_lamda_dir = lamda - prev_lamda

    logger.debug(f"Arc length solver: finished for one step, with Delta lambda = {lamda - prev_lamda}")

    return u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir


def arc_length_solver_force_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec, prev_Delta_lamda, q_vec, Delta_l=0.1, psi=1.):
    """
    TODO: Does not support periodic B.C., need some work here.
    """
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem, solver_options={'umfpack_solver':{}})
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return np.sum(Delta_u_vec1*Delta_u_vec2) + psi**2.*Delta_lamda1*Delta_lamda2*np.sum(q_vec_mapped**2.)

    u_vec = prev_u_vec
    lamda = prev_lamda
    q_vec_mapped = assign_zeros_bc(q_vec, problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-6
    res_val = 1.
    while res_val > tol:
        res_vec, A = newton_update_helper(u_vec)
        res_val = np.linalg.norm(res_vec + lamda*q_vec_mapped)
        logger.debug(f"Arc length solver: res_val = {res_val}")

        # TODO: the scipy umfpack solver seems to be far better than the jax linear solver, so we use umfpack solver here.
        # x0_1 = assign_bc(np.zeros_like(u_vec), problem)
        # x0_2 = copy_bc(u_vec, problem)
        # delta_u_bar = jax_solve(problem, A, -(res_vec + lamda*q_vec_mapped), x0=x0_1 - x0_2, precond=True)
        # delta_u_t = jax_solve(problem, A, -q_vec_mapped, x0=np.zeros_like(u_vec), precond=True)

        delta_u_bar = umfpack_solve(A, -(res_vec + lamda*q_vec_mapped))
        delta_u_t = umfpack_solve(A, -q_vec_mapped)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2.*np.sum(q_vec_mapped**2.)
        a2 = 2.* np.sum((Delta_u_vec + delta_u_bar)*delta_u_t) + 2.*psi**2.*Delta_lamda*np.sum(q_vec_mapped**2.)
        a3 = np.sum((Delta_u_vec + delta_u_bar)**2.) + psi**2.*Delta_lamda**2.*np.sum(q_vec_mapped**2.) - Delta_l**2.

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}"

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
            # At initial step, (Delta_u_vec_dir, Delta_lamda_dir) is zero, so both dot_prod1 and dot_prod2 are zero.
            # We simply select the larger value for delta_lamda.
            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
        elif dot_prod1 > dot_prod2:
            delta_lamda = delta_lamda1
        else:
            delta_lamda = delta_lamda2

        lamda = lamda + delta_lamda
        delta_u = delta_u_bar + delta_lamda * delta_u_t
        u_vec = u_vec + delta_u

        Delta_u_vec_dir = u_vec - prev_u_vec
        Delta_lamda_dir = lamda - prev_lamda

    logger.debug(f"Arc length solver: finished for one step, with Delta lambda = {lamda - prev_lamda}")

    return u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir


def get_q_vec(problem):
    """
    Used in the arc length method only, to get the external force vector q_vec
    """
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list = problem.newton_update(sol_list)
    q_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    return q_vec


################################################################################
# Dynamic relaxation solver

def assembleCSR(problem, dofs):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    problem.newton_update(sol_list)
    A_sp_scipy = scipy.sparse.csr_array((problem.V, (problem.I, problem.J)),
        shape=(problem.fes[0].num_total_dofs, problem.fes[0].num_total_dofs))

    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind], dtype=onp.int32)
            for row_ind in row_inds:
                A_sp_scipy.data[A_sp_scipy.indptr[row_ind]: A_sp_scipy.indptr[row_ind + 1]] = 0.
                A_sp_scipy[row_ind, row_ind] = 1.

    return A_sp_scipy


def calC(t, cmin, cmax):

    if t < 0.: t = 0.

    c = 2. * onp.sqrt(t)
    if (c < cmin): c = cmin
    if (c > cmax): c = cmax

    return c


def printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info, info_force):

    ## printing control
    if nIters % nPrint == 1:
        #logger.info('\t------------------------------------')
        if info_force == True:
            print(('\nDR Iteration %d: Max force (residual error) = %g (tol = %g)' +
                   'Max velocity = %g') % (nIters, error, tol,
                                            np.max(np.absolute(qdot))))
        if info == True:
            print('\nDamping t: ',t, );
            print('Damping coefficient: ', c)
            print('Max epsilon: ',np.max(eps))
            print('Max acceleration: ',np.max(np.absolute(qdotdot)))


def dynamic_relax_solve(problem, tol=1e-6, nKMat=50, nPrint=500, info=True, info_force=True, initial_guess=None):
    """
    Implementation of

    Luet, David Joseph. Bounding volume hierarchy and non-uniform rational B-splines for contact enforcement
    in large deformation finite element analysis of sheet metal forming. Diss. Princeton University, 2016.
    Chapter 4.3 Nonlinear System Solution

    Particularly good for handling buckling behavior.
    There is a FEniCS version of this dynamic relaxation algorithm.
    The code below is a direct translation from the FEniCS version.


    TODO: Does not support periodic B.C., need some work here.
    """
    solver_options = {'umfpack_solver': {}}

    # TODO: combine these into initial guess
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem, solver_options)
        return res_vec, A

    dofs = np.zeros(problem.num_total_dofs_all_vars)
    res_vec, A = newton_update_helper(dofs)
    dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)

    if initial_guess is not None:
        dofs = initial_guess
        dofs = assign_bc(dofs, problem)

    # parameters not to change
    cmin = 1e-3
    cmax = 3.9
    h_tilde = 1.1
    h = 1.

    # initialize all arrays
    N = len(dofs)  #print("--------num of DOF's: %d-----------" % N)
    #initialize displacements, velocities and accelerations
    q, qdot, qdotdot = onp.zeros(N), onp.zeros(N), onp.zeros(N)
    #initialize displacements, velocities and accelerations from a previous time step
    q_old, qdot_old, qdotdot_old = onp.zeros(N), onp.zeros(N), onp.zeros(N)
    #initialize the M, eps, R_old arrays
    eps, M, R, R_old = onp.zeros(N), onp.zeros(N), onp.zeros(N), onp.zeros(N)

    @jax.jit
    def assembleVec(dofs):
        res_fn = get_flatten_fn(problem.compute_residual, problem)
        res_vec = res_fn(dofs)
        res_vec = assign_zeros_bc(res_vec, problem)
        return res_vec

    R = onp.array(assembleVec(dofs))
    KCSR = assembleCSR(problem, dofs)

    M[:] = h_tilde * h_tilde / 4. * onp.array(
        onp.absolute(KCSR).sum(axis=1)).squeeze()
    q[:] = dofs
    qdot[:] = -h / 2. * R / M
    # set the counters for iterations and
    nIters, iKMat = 0, 0
    error = 1.0
    timeZ = time.time() #Measurement of loop time.

    assert onp.all(onp.isfinite(M)), f"M not finite"
    assert onp.all(onp.isfinite(q)), f"q not finite"
    assert onp.all(onp.isfinite(qdot)), f"qdot not finite"

    error = onp.max(onp.absolute(R))

    while error > tol:

        print(f"error = {error}")
        # marching forward
        q_old[:] = q[:]; R_old[:] = R[:]
        q[:] += h*qdot; dofs = np.array(q)

        R = onp.array(assembleVec(dofs))

        nIters += 1
        iKMat += 1
        error = onp.max(onp.absolute(R))

        # damping calculation
        S0 = onp.dot((R - R_old) / h, qdot)
        t = S0 / onp.einsum('i,i,i', qdot, M, qdot)
        c = calC(t, cmin, cmax)

        # determine whether to recal KMat
        eps = h_tilde * h_tilde / 4. * onp.absolute(
            onp.divide((qdotdot - qdotdot_old), (q - q_old),
                       out=onp.zeros_like((qdotdot - qdotdot_old)),
                       where=(q - q_old) != 0))

        # calculating the jacobian matrix
        if ((onp.max(eps) > 1) and (iKMat > nKMat)): #SPR JAN max --> min
            if info == True:
                print('\nRecalculating the tangent matrix: ', nIters)

            iKMat = 0
            KCSR = assembleCSR(problem, dofs)
            M[:] = h_tilde * h_tilde / 4. * onp.array(
                onp.absolute(KCSR).sum(axis=1)).squeeze()

        # compute new velocities and accelerations
        qdot_old[:] = qdot[:]; qdotdot_old[:] = qdotdot[:];
        qdot = (2.- c*h)/(2 + c*h) * qdot_old - 2.*h/(2.+c*h)* R / M
        qdot_old[:] = qdot[:]
        qdotdot = qdot - qdot_old

        # output on screen
        printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info, info_force)

    # check if converged
    convergence = True
    if onp.isnan(onp.max(onp.absolute(R))):
        convergence = False

    # print final info
    if convergence:
        print("DRSolve finished in %d iterations and %fs" % \
              (nIters, time.time() - timeZ))
    else:
        print("FAILED to converged")

    sol_list = problem.unflatten_fn_sol_list(dofs)

    return sol_list[0]


################################################################################
# Implicit differentiation with the adjoint method
def get_gpu_memory_info(gpu_id=0):
    """Get GPU memory statistics in bytes"""
    device = jax.devices()[gpu_id]
    mem_stats = device.memory_stats()
    
    bytes_in_use = mem_stats['bytes_in_use']
    bytes_limit = mem_stats['bytes_limit']
    free_mem = bytes_limit - bytes_in_use
    
    return free_mem, bytes_limit

def implicit_vjp(problem, sol_list, params, v_list, adjoint_solver_options):

    def constraint_fn(dofs, params):
        """c(u, p)
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        """c(u=u, p)
        """
        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol_list):
        """v*(partial dc/dp)
        """
        partial_c_fn = get_partial_params_c_fn(sol_list)
        def vjp_linear_fn(v_list):
            primals, f_vjp = jax.vjp(partial_c_fn, params)
            val, = f_vjp(v_list)
            return val
        return vjp_linear_fn
    # Checkpoint the VJP computation to save memory
    # def get_vjp_contraint_fn_params(params, sol_list):
    #     partial_c_fn = lambda p: constraint_fn_sol_to_sol(sol_list, p)
    #     @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    #     def vjp_linear_fn(v_list):
    #         primals, f_vjp = jax.vjp(partial_c_fn, params)
    #         val, = f_vjp(v_list)
    #         return val
    #     return vjp_linear_fn

    problem.set_params(params)
    # free, total = get_gpu_memory_info(0)
    # print(f"insideIMPVJP:Free: {free / 1e9:.2f} GB, Total: {total / 1e9:.2f} GB")

    problem.newton_update(sol_list)
    # free, total = get_gpu_memory_info(0)
    # print(f"After update insideIMPVJP:Free: {free / 1e9:.2f} GB, Total: {total / 1e9:.2f} GB")
    A = get_A(problem, adjoint_solver_options)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

    if hasattr(problem, 'P_mat'):
        v_vec = problem.P_mat.T @ v_vec

    adjoint_vec = linear_solver(A.transpose(), v_vec, np.zeros_like(v_vec), adjoint_solver_options)

    if hasattr(problem, 'P_mat'):
        adjoint_vec = problem.P_mat @ adjoint_vec

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)

    # Checkpoint when actually calling the function
    # @partial(jax.checkpoint,
    #          policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def checkpointed_call(v):
        return vjp_linear_fn(v)
    
    # free, total = get_gpu_memory_info(0)
    # print(f"Before vjp_result:Free: {free / 1e9:.2f} GB, Total: {total / 1e9:.2f} GB")
    vjp_result = checkpointed_call(problem.unflatten_fn_sol_list(adjoint_vec))
    # vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree.map(lambda x: -x, vjp_result)

    return vjp_result

# def implicit_vjp(problem, sol_list, params, v_list, adjoint_solver_options):
#     # Set up problem state (boundary conditions, etc.) - NOT traced
#     problem.set_params(params)
#     problem.newton_update(sol_list)
    
#     # Extract what you need from problem state
#     # (assuming these are set by set_params and don't change)
#     # node_inds_list = problem.fe.node_inds_list
#     # vec_inds_list = problem.fe.vec_inds_list
#     # vals_list = problem.fe.vals_list
    
#     def constraint_fn_pure(dofs, params):
#         """Pure function - no side effects, no argwhere"""
#         # Use problem methods that are pure computations
#         res_fn = problem.compute_residual
#         res_fn = get_flatten_fn(res_fn, problem)
#         res_fn = apply_bc(res_fn, problem)
#         return res_fn(dofs)

#     def constraint_fn_sol_to_sol(sol_list, params):
#         dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
#         con_vec = constraint_fn_pure(dofs, params)
#         return problem.unflatten_fn_sol_list(con_vec)

#     # Now checkpoint is safe - no argwhere in traced code
#     @partial(jax.checkpoint, 
#              policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
#     def checkpointed_constraint(sol_list, params):
#         return constraint_fn_sol_to_sol(sol_list, params)

#     A = get_A(problem, adjoint_solver_options)
#     v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

#     if hasattr(problem, 'P_mat'):
#         v_vec = problem.P_mat.T @ v_vec

#     adjoint_vec = linear_solver(A.transpose(), v_vec, np.zeros_like(v_vec), 
#                                  adjoint_solver_options)

#     if hasattr(problem, 'P_mat'):
#         adjoint_vec = problem.P_mat @ adjoint_vec

#     adjoint_unflatten = problem.unflatten_fn_sol_list(adjoint_vec)
#     partial_c_fn = lambda p: checkpointed_constraint(sol_list, p)
#     primals, f_vjp = jax.vjp(partial_c_fn, params)
#     vjp_result, = f_vjp(adjoint_unflatten)
#     vjp_result = jax.tree.map(lambda x: -x, vjp_result)

#     return vjp_result

def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = solver(problem, solver_options)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred



def ad_wrapper_discrete3(time_stepping_forward,problem,solver_options={},adjoint_solver_options={}):
    # Set up shardings for device and host memory
    s_dev = SingleDeviceSharding(jax.devices()[0]) #, memory_kind="device"
    s_host = SingleDeviceSharding(jax.devices()[0]) #, memory_kind="pinned_host"
    
    @jax.custom_vjp
    def fwd_pred(params):
        # The VJP only needs the final trajectory for the loss function,
        # but the backward pass needs the full history.
        sol_traj, _ = checkpoint(time_stepping_forward(params, problem, solver, solver_options))
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Return solution trajectory for the loss function, but save
        # params and full history for the backward pass.
        # Offload trajectories to host memory to free GPU memory
        sol_traj_host = jax.device_put(sol_traj, s_host)
        params_traj_host = jax.device_put(params_traj, s_host)
        del params_traj
        # Return solution for loss (keep on device), save offloaded data for backward
        return sol_traj, (params_traj_host, sol_traj_host)

    def f_bwd(res, v_traj):
        """
        Reduced adjoint: internal variables are implicit functions of u.
        Only propagate λ^u backward; compute ∂L/∂p via chain rule.
        """
        params_traj_host, sol_traj_host = res
        
        global_params = jax.device_put(params_traj_host[0][3], s_dev)
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, global_params)
        
        # Initialize adjoint variable for u_N
        sol_final = jax.device_put(sol_traj_host[-1], s_dev)
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_final)
        
        # Backward loop: n = N-1 down to 0
        for i in range(len(params_traj_host) - 1, -1, -1):
            # Load data for step n
            sol_n = jax.device_put(sol_traj_host[i], s_dev)
            iv_prev_n = jax.device_put(params_traj_host[i][0], s_dev)  # z_{n-1}
            params_n = jax.device_put(params_traj_host[i], s_dev)
            
            # --- CRITICAL: Compute z_n = C(u_n, z_{n-1}, p) ---
            # This is needed for the chain rule in the next step
            iv_n = problem.update_int_vars_gp(sol_n, iv_prev_n)
            
            # --- 1. Build adjoint RHS: contribution from loss + future step ---
            adjoint_rhs = jax.tree_util.tree_map(
                lambda loss, prop: loss + prop,
                v_traj[i],
                propagated_sensitivity_u
            )
            
            # --- 2. Solve adjoint system for λ_n (sensitivity wrt u_n) ---
            # implicit_vjp solves: [∂R/∂u_n]^T λ_n = adjoint_rhs
            # and returns VJPs: (∂/∂z_{n-1}, ..., ∂/∂u_{n-1}, ∂/∂p)
            vjp_result = implicit_vjp(
                problem, [sol_n], params_n, adjoint_rhs, adjoint_solver_options
            )
            
            # vjp_result structure: (sens_iv_prev, ..., sens_u_prev, sens_params)
            sens_iv_prev_from_R = vjp_result[0]  # ∂L/∂z_{n-1} from R_n
            sens_u_prev = vjp_result[2]           # ∂L/∂u_{n-1} from R_n
            sens_params_from_R = vjp_result[3]    # ∂L/∂p from R_n
            
            # --- 3. Accumulate parameter gradient ---
            total_grad_p = jax.tree_util.tree_map(
                lambda total, contrib: total + contrib,
                total_grad_p, sens_params_from_R
            )
            
            # --- 4. Propagate sensitivities to previous step ---
            if i > 0:
                # (a) Propagate u_{n-1} sensitivity
                propagated_sensitivity_u = sens_u_prev
                
                # (b) CRITICAL: Chain rule for z_{n-1} → z_n → u_{n+1}
                # We need: ∂L/∂z_{n-1} = (∂L/∂z_{n-1})|_R + (∂L/∂u_n)(∂u_n/∂z_n)(∂z_n/∂z_{n-1})
                # 
                # This requires computing VJP through C: z_n = C(u_n, z_{n-1}, p)
                def C_fn(iv_prev_local):
                    return problem.update_int_vars_gp(sol_n, iv_prev_local)
                
                # Compute (∂z_n/∂z_{n-1})^T * (∂u_{n+1}/∂z_n)^T * λ_{n+1}
                # NOTE: This is typically ZERO if u_{n+1} doesn't directly depend on z_n,
                # but the gradient flows through the *next* R_{n+1} evaluation
                
                # The sensitivity from the next step (already in propagated_sensitivity_u)
                # was computed assuming z_n is fixed. We need additional contribution
                # if the *next* step's residual R_{n+1} depends on z_n.
                
                # For most formulations, this is captured by making sure params_n
                # includes iv_n when solving the *next* step's adjoint.
                # If your implicit_vjp doesn't see this dependency, you need:
                
                # Additional contribution through C's dependency on z_{n-1}:
                # This is ONLY needed if propagating from step n+1 involved z_n
                # For now, the direct contribution from R_n is:
                # (already captured in sens_iv_prev_from_R)
                
                # Store for next iteration (this becomes z_{n-2} in next loop)
                # In pure reduced form, we DON'T propagate λ^z separately!
                # All sensitivity flows through λ^u and chain rules.
        
        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred

def ad_wrapper_discrete2(time_stepping_forward, problem, solver_options={}, adjoint_solver_options={}):
    s_dev = SingleDeviceSharding(jax.devices()[0])
    s_host = SingleDeviceSharding(jax.devices()[0])

    @jax.custom_vjp
    def fwd_pred(params):
        # Forward run; backward will need full history
        sol_traj, _ = time_stepping_forward(params, problem, solver, solver_options)
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Offload history to host to save device memory
        sol_traj_host = jax.device_put(sol_traj, s_host)
        params_traj_host = jax.device_put(params_traj, s_host)
        del params_traj
        return sol_traj, (params_traj_host, sol_traj_host)

    def f_bwd(res, v_traj):
        params_traj_host, sol_traj_host = res

        # Initialize gradient accumulator
        global_params = jax.device_put(params_traj_host[0][3], s_dev)
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, global_params)

        # Future sensitivities start at zero
        sol_final = jax.device_put(sol_traj_host[-1], s_dev)
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_final)

        # Backward sweep
        for i in range(len(params_traj_host) - 1, -1, -1):
            sol_n = jax.device_put(sol_traj_host[i], s_dev)
            params_n = jax.device_put(params_traj_host[i], s_dev)

            # RHS: loss at step n + propagated sensitivity from future
            adjoint_rhs = jax.tree_util.tree_map(
                lambda loss, prop: loss + prop,
                v_traj[i],
                propagated_sensitivity_u,
            )

            # Solve adjoint and get VJPs wrt (iv_{n-1}, …, u_{n-1}, p)
            vjp_result = implicit_vjp(problem, [sol_n], params_n, adjoint_rhs, adjoint_solver_options)

            # Accumulate param gradient
            total_grad_p = jax.tree_util.tree_map(
                lambda total, contrib: total + contrib,
                total_grad_p,
                vjp_result[3],
            )

            # Propagate u-sensitivity to previous step
            if i > 0:
                propagated_sensitivity_u = vjp_result[2]

        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred


def ad_wrapper_discrete1(time_stepping_forward,problem,solver_options={},adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        # The VJP only needs the final trajectory for the loss function,
        # but the backward pass needs the full history.
        sol_traj, _ = time_stepping_forward(params, problem, solver, solver_options)
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Return solution trajectory for the loss function, but save
        # params and full history for the backward pass.
        return sol_traj, (params_traj, sol_traj)

    def f_bwd(res, v_traj):
        """
        Executes the full discrete adjoint backward pass, accounting for both the
        implicit solve and the explicit internal variable update.
        """
        params_traj, sol_traj = res
        global_params = params_traj[0][3]
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, global_params)

        # --- Initialize sensitivities ---
        # We need the *final* state of the internal variables (iv_N) to initialize.
        # We compute it from the inputs to the very last step.
        sol_final = sol_traj[-1]
        iv_prev_final = params_traj[-1][0] # This is iv_{N-1}
        iv_final = problem.update_int_vars_gp(sol_final, iv_prev_final)
        
        # Sensitivity of the loss wrt the final internal variables (iv_N) is zero.
        propagated_sensitivity_iv = jax.tree_util.tree_map(np.zeros_like, iv_final)
        # Sensitivity from the future (step N) starts at zero for the solution variables.
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_traj[-1])

        # Iterate backwards from step n = N-1 down to 0.
        for i in range(len(params_traj) - 1, -1, -1):
            sol_n = sol_traj[i]
            params_for_step_n = params_traj[i]
            # Get iv_{n-1} from the parameters used for step n, as you specified.
            iv_prev = params_for_step_n[0]
            
            iv_core = iv_prev[0:-1]
            param = iv_prev[-1]
            def update_iv_core(sol_loc,iv_core_local,param_loc):
                iv_prev_local = [iv_core_local, param_loc]
                return problem.update_int_vars_gp(sol_loc, iv_prev_local)
            # --- 1. Backward pass through `update_int_vars_gp(sol_n, iv_{n-1})` ---
            # Propagate sensitivity from iv_n back to sol_n and iv_{n-1}.
            update_fn = lambda s, iv: problem.update_int_vars_gp(s, iv)
            _, vjp_update_fn = jax.vjp(update_fn, sol_n, iv_prev)
            
            # Get sensitivities wrt the inputs of the update function.
            sens_from_iv_wrt_sol_n, sens_from_iv_wrt_iv_prev = vjp_update_fn(propagated_sensitivity_iv)
            
            # --- 2. Construct Total RHS for the Main Adjoint Solve at step n ---
            # The total sensitivity for sol_n is the sum of three contributions.
            adjoint_rhs_list = jax.tree_util.tree_map(
                lambda loss, prop_u, prop_iv: loss + prop_u + prop_iv,
                v_traj[i],
                propagated_sensitivity_u,
                sens_from_iv_wrt_sol_n
            )
            
            # --- 3. Solve Main Adjoint System for λ_n and get all VJPs ---
            vjp_result = implicit_vjp(
                problem, [sol_n], params_for_step_n, adjoint_rhs_list, adjoint_solver_options
            )
            
            # --- 4. Accumulate Gradient wrt Global Parameters 'p' ---
            grad_wrt_p = vjp_result[3]
            # total_grad_p = jax.tree_util.tree_map(lambda total, contrib: total + contrib, total_grad_p, grad_wrt_p)
            sens_from_C_wrt_p = sens_from_iv_wrt_iv_prev[-1]
            sens_from_C_wrt_p_total =  0. * np.sum(sens_from_C_wrt_p, axis=(0, 1))
            # breakpoint()
            
            total_grad_p = jax.tree_util.tree_map(
                lambda total, gR, gC: total + gR + gC,
                total_grad_p,
                grad_wrt_p,
                sens_from_C_wrt_p_total,
            )

            # --- 5. Compute and Propagate Total Sensitivities to the PREVIOUS step (n-1) ---
            if i > 0:
                # Sensitivity wrt u_{n-1} comes from this step's implicit solve.
                propagated_sensitivity_u = vjp_result[2]
                
                # Sensitivity wrt iv_{n-1} comes from TWO places.
                propagated_sensitivity_iv = jax.tree_util.tree_map(
                    lambda from_solve, from_update: from_solve + from_update,
                    vjp_result[0], # Contribution from the implicit solve
                    sens_from_iv_wrt_iv_prev # Contribution from the explicit update
                )
                
        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred

def ad_wrapper_discrete(time_stepping_forward,problem,solver_options={},adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        # The VJP only needs the final trajectory for the loss function,
        # but the backward pass needs the full history.
        sol_traj, _ = time_stepping_forward(params, problem, solver, solver_options)
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Return solution trajectory for the loss function, but save
        # params and full history for the backward pass.
        return sol_traj, (params_traj, sol_traj)

    def f_bwd(res, v_traj):
        params_traj, sol_traj = res
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, params_traj[0][3])

        sol_final = sol_traj[-1]
        iv_prev_final = params_traj[-1][0]
        param_final = params_traj[-1][3]
        iv_final = problem.update_int_vars_gp2(sol_final, iv_prev_final, param_final)
        propagated_sensitivity_iv = jax.tree_util.tree_map(np.zeros_like, iv_final)
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_final)

        for i in range(len(params_traj) - 1, -1, -1):
            sol_n = sol_traj[i]
            iv_prev = params_traj[i][0]
            param_n = params_traj[i][3]

            update_fn = lambda s, iv, p: problem.update_int_vars_gp2(s, iv, p)
            _, vjp_update_fn = jax.vjp(update_fn, sol_n, iv_prev, param_n)
            sens_sol_C, sens_iv_prev_C, sens_param_C = vjp_update_fn(propagated_sensitivity_iv)
            sens_param_C = jax.tree_util.tree_map(lambda x: 0. * x, sens_param_C)

            adjoint_rhs_list = jax.tree_util.tree_map(
                lambda loss, prop_u, sen_C: loss + prop_u + sen_C,
                v_traj[i],
                propagated_sensitivity_u,
                sens_sol_C,
            )

            vjp_result = implicit_vjp(problem, [sol_n], params_traj[i], adjoint_rhs_list, adjoint_solver_options)

            # Accumulate parameter sensitivity from both the implicit residual
            # and the explicit internal-variable update.
            total_grad_p = jax.tree_util.tree_map(
                lambda total, gR, gC: total + gR - gC,
                total_grad_p,
                vjp_result[3],
                sens_param_C,
            )

            if i > 0:
                propagated_sensitivity_u = vjp_result[2]
                propagated_sensitivity_iv = jax.tree_util.tree_map(
                    lambda from_solve, from_update: from_solve + from_update,
                    vjp_result[0],
                    sens_iv_prev_C,
                )

        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred