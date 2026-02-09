# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator

from tqdm import tqdm
import datetime
import warnings
import shutil
import h5py
import sys
import os

import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple

from firedrake import interpolate as fd_interpolate
from firedrake import assemble, ln

from iceinfer import Timer
import rasterio

from . import models
from .problems import make_mesh, IceShelfProblem
from .objectives import PosteriorObjective, DataObjective
from .fields import create_rheology_field
from .utilities import NumpyConverter


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def initialize_objects(
    cfg,
    data_file=None,
    obj_scale=None,
    comm=None,
    data_obj=False,
    time_ind=None,
    mesh_degree=2,
):

    outdir = cfg.output_dir

    # Create rheology field
    field = create_rheology_field(cfg, hyper_params=cfg.inverse.hyper_params)

    # Initialize a firedrake -> numpy function converter
    converter = NumpyConverter(
        coord_file=data_file,
        shelf_file=cfg.shelf_coords_file,
    )

    if data_file is None:
        return field, converter

    # Make mesh
    mesh = make_mesh(meshfile=cfg.inverse.meshfile, comm=comm)

    # Now, create a problem with a more reasonable area scale
    kwargs = dict(
        nll_scale=cfg.inverse.nll_scale,
        degree=mesh_degree,
        B_norm=cfg.inverse.B_norm_type,
        n_norm=cfg.inverse.n_norm_type,
        default_vel_error=cfg.data.u_sigma,
        dirichlet_ids=cfg.inverse.dirichlet_ids,
    )
    if cfg.inverse.joint:
        problem = IceShelfProblem(cfg, mesh, **kwargs)
        problem.load_observations(data_file, time_ind=time_ind)
    else:
        problem = IceShelfProblem(cfg, mesh, fixed_exponent=True, **kwargs)
        problem.load_observations(data_file, time_ind=time_ind)
        problem.load_rheology(outdir, n_exp=cfg.inverse.fixed_n)

    # Optional boundary smoothing
    if cfg.inverse.smooth_boundary:
        problem.smooth_boundaries()
    problem.init_solvers()

    # Create the optimization objective object
    obj_scale = obj_scale or cfg.inverse.objective_scale
    if data_obj:
        objective = DataObjective(field, problem, scale=obj_scale)
    else:
        objective = PosteriorObjective(field, problem, scale=obj_scale)

    return field, problem, objective, converter


def threshold_spectral_matrix(
    L_min, target_std=0.2, apply_sensitivity=False, outdir=None
):

    # Read reference matrix data
    matrix_data = {}
    with h5py.File("aux/matrix_data_spectral.h5", "r") as fid:
        x = fid["xp"][()]
        y = fid["yp"][()]
        evals = fid["evals"][()]
        for name in ("B", "n"):
            sub_data = {}
            group = fid[name]
            matrix_data[name] = {"G": group["G"][()], "G_grid": group["G_grid"][()]}

    # Compute threshold
    Θ = 1.0
    L_min = 1e3 * L_min
    σ_c = Θ / L_min
    eval_thresh = (2 * np.pi / L_min) ** 2
    mask = evals <= eval_thresh

    # Read sensitivity if requested
    if apply_sensitivity:
        assert outdir is not None, "Must supply directory for reading sensitivity."
        path = os.path.join(outdir, "mesh_data_sensitivity.h5")
        with h5py.File(path, "r") as fid:
            S = fid["s_mesh"][()]
        SG = np.abs(S @ matrix_data["B"]["G"]) / np.sqrt(evals)
        mask *= SG >= 1e8

    # Mask the matrices and scale the columns
    evals = evals[mask]
    Sr = np.sum(1 / evals)
    for name in ("B", "n"):

        # Mask first
        G = matrix_data[name]["G"]
        G_grid = matrix_data[name]["G_grid"]
        G = G[:, mask]
        G_grid = G_grid[:, mask]

        # Eigenvalue and length scale scaling
        α = target_std / (σ_c * np.sqrt(Sr))
        Np = G.shape[1]
        for j in range(Np):
            scale = α * σ_c / np.sqrt(evals[j])
            G[:, j] *= scale
            G_grid[:, j] *= scale

        # Update dictionary
        matrix_data[name]["G"] = G
        matrix_data[name]["Gx"] = np.zeros_like(G)
        matrix_data[name]["Gy"] = np.zeros_like(G)
        matrix_data[name]["G_grid"] = G_grid
        matrix_data[name]["S"] = np.eye(Np) * α**2

    return matrix_data


# ---------------------------------------------------------------------------
# Inversion tools
# ---------------------------------------------------------------------------


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Value: {v}, Gradient norm: {e}",
            i=state.iter_num,
            v=value,
            e=optax.tree.norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# Define a class to hold state (iteration count + history)
class CallbackTracker:
    def __init__(self, verbose=True):
        self.nit = 0
        self.losses = []
        self.verbose = verbose

    def __call__(self, intermediate_result):
        # 1. Increment iteration counter
        self.nit += 1

        # 2. Save the iteration number and loss value
        loss = intermediate_result.fun
        self.losses.append(loss)

        # Optional: Print live progress
        if self.verbose and self.nit % 50 == 0:  # Print every 10 iterations
            print(f"Iter: {self.nit}, Loss: {loss:.5f}")


def optimize_bfgs(
    objective,
    x0,
    maxiter=1000,
    factr=100.0,
    gtol=1e-6,
    maxls=40,
    maxcor=10,
    method="BFGS",
    hess_inv0=None,
    verbose=True,
):

    if method == "BFGS":
        options = {
            "disp": True,
            "maxiter": maxiter,
            "hess_inv0": hess_inv0,
            "gtol": gtol,
            "c2": 0.99,
        }
    elif method == "L-BFGS-B":
        options = {
            "disp": 25,
            "maxiter": maxiter,
            "maxfun": 3 * maxiter,
            "maxls": maxls,
            "maxcor": maxcor,
            "ftol": factr * np.finfo(np.float64).eps,
            "gtol": gtol,
        }

    tracker = CallbackTracker(verbose=verbose)

    # Run optimization
    with Timer(f"Performing optimization with {method}"):
        result = minimize(
            fun=objective,
            x0=x0,
            jac=True,
            method=method,
            options=options,
            callback=tracker,
        )
        if verbose:
            print(result.message)
            print("Final function value:", result.fun)
            print("Number of iterations:", result.nit)

    return (
        result.x,
        {},
        tracker.losses,
    )


def optimize_sgd(
    objective,
    x0,
    optimizer,
    opt_state,
    n_iter,
    loss_fn=None,
    thresh=0.0,
    callback=None,
):

    @jax.jit
    def update_step(grad, state, params, loss_val, grad_val):
        updates, new_state = optimizer.update(
            grad,
            state,
            params,
            value=loss_val,
            grad=grad_val,
            value_fn=loss_fn,
        )
        params = optax.apply_updates(params, updates)
        return params, new_state

    losses = np.zeros(n_iter)

    count = 0
    for k in tqdm(range(n_iter)):
        loss_val, grad_val = objective(x0, state=opt_state)  # state kwarg for lbfgs
        x0, opt_state = update_step(
            grad_val,
            opt_state,
            x0,
            loss_val,
            grad_val,
        )
        losses[k] = loss_val
        if loss_val < thresh:
            losses = losses[:k]
            break

        if callback is not None:
            callback(k, x0, opt_state, losses[: k + 1], grad_val)

    return x0, opt_state, losses


def optimize_pre_sgld(objective, x0, optimizer, opt_state, n_iter, interval=50):

    @jax.jit
    def update_step(grad, state, params):
        updates, new_state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        return params, new_state

    losses = np.zeros(n_iter)
    samples = np.zeros((n_iter, x0.size), dtype=x0.dtype)
    vel_samples = np.zeros((n_iter, x0.size), dtype=x0.dtype)
    grad_norms = np.zeros(n_iter)

    for k in tqdm(range(n_iter)):
        loss_val, grad_val = objective(x0)
        x0, opt_state = update_step(grad_val, opt_state, x0)
        if k % interval == 0:
            print(f"Iter {k:5d}: {loss_val}")
        losses[k] = loss_val

        samples[k] = x0
        vel_samples[k] = np.array(opt_state.velocity)
        grad_norms[k] = jnp.linalg.norm(grad_val)

    return x0, opt_state, losses, samples, vel_samples, grad_norms


def init_optimizer(opt_type, params, learning_rate):

    opt_cls = getattr(optax, opt_type)
    if opt_type == "sgd":
        # hard-code the learning rate for now
        # sgd always requires some type of gradient clipping to avoid NaNs
        optimizer = optax.chain(
            optax.clip_by_global_norm(5.0),
            opt_cls(learning_rate=0.0005, momentum=0.9),
        )
    else:
        optimizer = optax.chain(
            opt_cls(learning_rate=learning_rate),
        )

    opt_state = optimizer.init(params)

    return optimizer, opt_state


def create_lr_schedule(schedule_list, ref_lr):
    """
    Create an Adam optimizer with a piecewise constant learning rate schedule.

    Args:
        schedule_list: List of [step, learning_rate] pairs, e.g.,
                      [[0, 0.05], [1000, 0.025], [2000, 0.01]]
                      First step should be 0.

    Returns:
        optax.GradientTransformation: Adam optimizer with schedule
    """
    if len(schedule_list) == 0:
        schedule_list = [[0, ref_lr]]

    if schedule_list[0][0] != 0:
        raise ValueError("Schedule must start with step 0")

    # Extract boundaries and values
    steps, learning_rates = zip(*schedule_list)

    # Boundaries are all steps except the first (step 0)
    boundaries = list(steps[1:])

    # Calculate scaling factors between consecutive learning rates
    scales = [lr2 / lr1 for lr1, lr2 in zip(learning_rates[:-1], learning_rates[1:])]

    # Create dictionary of boundaries and scales
    boundaries_and_scales = dict(zip(boundaries, scales))

    # Create the schedule
    schedule_fn = optax.piecewise_constant_schedule(
        init_value=learning_rates[0], boundaries_and_scales=boundaries_and_scales
    )

    return schedule_fn


def create_logprob_with_vjp(py_logprob):
    """
    Create a JAX-compatible log probability function with custom VJP.

    Args:
        py_logprob: Black-box function that takes x and returns (log_prob, grad)
    """

    @jax.custom_vjp
    def logprob_fn(x):
        def f_callback(x_np):
            value, _ = py_logprob(x_np)
            return value

        result_shape_and_dtype = jax.ShapeDtypeStruct((), x.dtype)
        primal_value = jax.pure_callback(f_callback, result_shape_and_dtype, x)
        return primal_value

    def logprob_fwd(x):
        def f_and_grad_callback(x_np):
            value, grad = py_logprob(x_np)
            return np.array(value), np.array(grad)

        result_shape_and_dtype = (
            jax.ShapeDtypeStruct((), x.dtype),  # for the scalar value
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # for the gradient vector
        )

        primal_value, grad_for_bwd = jax.pure_callback(
            f_and_grad_callback, result_shape_and_dtype, x
        )

        return primal_value, grad_for_bwd

    def logprob_bwd(residuals, g):
        """
        The backward pass. `residuals` is what we saved in the fwd pass.
        """
        # Unpack the pre-computed gradient we saved.
        precomputed_grad = residuals
        # The VJP is g * precomputed_grad. For a scalar function, g is 1.0.
        return (g * precomputed_grad,)

    # Register the forward and backward passes.
    logprob_fn.defvjp(logprob_fwd, logprob_bwd)

    return logprob_fn


def unpack_hess_inv(result):

    if isinstance(result.hess_inv, LinearOperator):
        hess_inv = np.zeros((result.x.size, result.x.size))
        for i in tqdm(range(result.x.size)):
            # Create a standard basis vector (e_i)
            basis_vector = np.zeros_like(result.x)
            basis_vector[i] = 1.0

            # Apply the LinearOperator to the basis vector to get the i-th column
            column_i = result.hess_inv.dot(
                basis_vector
            )  # or hess_inv_op @ basis_vector in newer NumPy versions
            hess_inv[:, i] = column_i
    else:
        hess_inv = result.hess_inv

    return hess_inv


def generate_predictions(z, field, problem, objective, converter, θ_np=None):

    problem.reset_solver()

    # Get loss function
    loss = objective.loss(z)

    # Unpack to controls
    if z is not None and θ_np is None:
        θ_np = field(z)
    else:
        assert z is None and θ_np is not None
        z = θ_np

    objective.controls = θ_np
    if problem.fixed_exponent:
        θ_B = objective.controls[0]
        θ_n = problem.θ_n
    else:
        θ_B, θ_n = objective.controls
    preds = problem.forward(objective.controls)

    # Output coordinates
    Ny_out, Nx_out = converter.out_shape
    x_out = np.linspace(converter.xarr[0], converter.xarr[-1], Nx_out)
    y_out = np.linspace(converter.yarr[0], converter.yarr[-1], Ny_out)

    # Evaluate rheology fields
    out_preds = {"z": z, "x": x_out, "y": y_out, "final_loss": loss}
    n = problem._inv_norm_n(θ_n)
    B = problem._inv_norm_B(θ_B)

    # Also evaluate reference viscosity field
    log_nu = ln(0.5 * B) + (1 - n) / n * problem.log_e_ref

    out_preds["B"] = converter(assemble(fd_interpolate(B, problem.Q)))
    out_preds["n"] = converter(assemble(fd_interpolate(n, problem.Q)))
    out_preds["log_nu"] = converter(assemble(fd_interpolate(log_nu, problem.Q)))

    for i, p in enumerate(preds):
        out_preds[f"u{i}"] = converter(p)

    return out_preds


# end of file
