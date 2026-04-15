# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator

from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple

from firedrake import interpolate as fd_interpolate
from firedrake import assemble, ln

from iceinfer import Timer

from .config_utilities import validate_cfg
from . import models
from .problems import make_mesh, IceShelfProblem
from .objectives import (
    PosteriorObjective,
    DataObjective,
    _controls_to_array,
    _set_controls_from_array,
)
from .fields import boundary_inference_enabled, create_inference_field
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
    cfg = validate_cfg(cfg)

    outdir = cfg.output_dir

    # Initialize a firedrake -> numpy function converter
    converter = NumpyConverter(
        coord_file=data_file,
        shelf_file=cfg.shelf_coords_file,
    )

    if data_file is None:
        if boundary_inference_enabled(cfg):
            raise ValueError(
                "Boundary-velocity inference requires `data_file` so boundary metadata can be built."
            )
        field = create_inference_field(cfg, hyper_params=cfg.inverse.hyper_params)
        return field, converter

    # Make mesh
    print('Mesh file:', cfg.inverse.meshfile)
    mesh = make_mesh(meshfile=cfg.inverse.meshfile, comm=comm)

    # Now, create a problem with a more reasonable area scale
    kwargs = dict(
        nll_scale=cfg.inverse.nll_scale,
        degree=mesh_degree,
        B_norm=cfg.inverse.B_norm_type,
        n_norm=cfg.inverse.n_norm_type,
        default_vel_error=cfg.data.u_sigma,
        dirichlet_ids=cfg.inverse.dirichlet_ids,
        infer_boundary_velocity=boundary_inference_enabled(cfg),
    )
    if cfg.inverse.joint:
        problem = IceShelfProblem(cfg, mesh, **kwargs)
        problem.load_observations(data_file, time_ind=time_ind)
    else:
        problem = IceShelfProblem(cfg, mesh, fixed_exponent=True, **kwargs)
        problem.load_observations(data_file, time_ind=time_ind)
        problem.load_rheology(outdir, n_exp=cfg.inverse.fixed_n)

    # Boundary reference smoothing is always applied; the config chooses the method.
    problem.smooth_boundaries()
    problem.init_solvers()

    # Create latent-to-control field after the problem is initialized
    field = create_inference_field(
        cfg,
        problem=problem,
        hyper_params=cfg.inverse.hyper_params,
    )

    # Create the optimization objective object
    obj_scale = obj_scale or cfg.inverse.objective_scale
    if data_obj:
        objective = DataObjective(field, problem, scale=obj_scale)
    else:
        objective = PosteriorObjective(field, problem, scale=obj_scale)

    return field, problem, objective, converter


def inference_metadata(field):
    n_total = int(field.N)
    n_rheo = int(getattr(field, "N_rheo", n_total))
    n_B = int(getattr(field, "N_B", n_rheo))
    n_n = int(getattr(field, "N_n", max(0, n_rheo - n_B)))
    metadata = {
        "infer_boundary_velocity": np.int8(hasattr(field, "epoch_alpha_slices")),
        "n_params_total": np.int64(n_total),
        "n_params_rheology": np.int64(n_rheo),
        "n_params_B": np.int64(n_B),
        "n_params_n": np.int64(n_n),
        "z_B_start": np.int64(0),
        "z_B_stop": np.int64(n_B),
        "z_n_start": np.int64(n_B),
        "z_n_stop": np.int64(n_B + n_n),
    }

    if hasattr(field, "epoch_alpha_slices"):
        alpha_x_start = np.zeros((field.Nt,), dtype=np.int64)
        alpha_x_stop = np.zeros((field.Nt,), dtype=np.int64)
        alpha_y_start = np.zeros((field.Nt,), dtype=np.int64)
        alpha_y_stop = np.zeros((field.Nt,), dtype=np.int64)
        for k in range(field.Nt):
            sx, sy = field.epoch_alpha_slices(k)
            alpha_x_start[k] = sx.start
            alpha_x_stop[k] = sx.stop
            alpha_y_start[k] = sy.start
            alpha_y_stop[k] = sy.stop

        metadata.update(
            {
                "n_boundary_epochs": np.int64(field.Nt),
                "boundary_n_basis": np.int64(field.n_basis),
                "boundary_basis_spacing": np.float64(field.basis_spacing),
                "boundary_spline_degree": np.int64(field.degree),
                "boundary_length": np.float64(field.boundary_length),
                "boundary_coeff_positions": np.asarray(field.coeff_positions),
                "z_alpha_x_start": alpha_x_start,
                "z_alpha_x_stop": alpha_x_stop,
                "z_alpha_y_start": alpha_y_start,
                "z_alpha_y_stop": alpha_y_stop,
            }
        )
        if getattr(field, "basis_spacing_target", None) is not None:
            metadata["boundary_basis_spacing_target"] = np.float64(
                field.basis_spacing_target
            )

    return metadata


def augment_boundary_metadata(metadata, field, problem):
    metadata = dict(metadata)
    if hasattr(field, "epoch_alpha_slices"):
        metadata.update(
            {
                "boundary_s": np.asarray(problem.boundary_s),
                "boundary_node_indices": np.asarray(problem.boundary_node_indices),
                "boundary_coords": np.asarray(problem.boundary_coords),
                "boundary_basis_matrix": np.asarray(field.A),
                "boundary_smoothing_method": np.bytes_(
                    getattr(problem, "boundary_smoothing_method", "spline")
                ),
            }
        )
        if getattr(problem, "boundary_ref_corr_length", None) is not None:
            metadata["boundary_ref_corr_length"] = np.float64(
                problem.boundary_ref_corr_length
            )
    return metadata


def print_parameter_summary(field, printer=print):
    metadata = inference_metadata(field)
    printer("Number of parameters:", int(metadata["n_params_total"]))
    printer(
        "Rheology blocks:",
        f"B={int(metadata['n_params_B'])}",
        f"n={int(metadata['n_params_n'])}",
    )
    if int(metadata["infer_boundary_velocity"]):
        printer(
            "Boundary blocks:",
            f"epochs={int(metadata['n_boundary_epochs'])}",
            f"basis={int(metadata['boundary_n_basis'])}",
            f"spacing={float(metadata['boundary_basis_spacing']):.1f}",
            f"per-epoch params={2 * int(metadata['boundary_n_basis'])}",
        )


def initialize_latent_vector(field, seed, amp):
    rng = np.random.default_rng(seed)
    z = amp * rng.standard_normal((field.N,))

    if hasattr(field, "epoch_alpha_slices"):
        z = np.asarray(z, dtype=float)
        for k in range(field.Nt):
            sx, sy = field.epoch_alpha_slices(k)
            z[sx] = 0.0
            z[sy] = 0.0

    return z


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


def _normalize_physical_controls(fields_np, problem, objective):
    """
    Convert a direct physical control array into the normalized control space.

    The input layout matches the flattened Firedrake controls used by the
    objective. Rheology controls are interpreted as physical B/n fields on the
    problem mesh; any trailing boundary-velocity controls are already in their
    native space and are passed through unchanged.
    """
    controls = objective.copy_controls()
    _set_controls_from_array(controls, np.asarray(fields_np, dtype=float))

    θ_B_raw, θ_n_control, bc_controls = problem.split_controls(controls)

    if problem.fixed_exponent:
        n_raw = assemble(fd_interpolate(problem._inv_norm_n(θ_n_control), problem.Q))
    else:
        θ_n_norm = assemble(fd_interpolate(problem._fwd_norm_n(θ_n_control), problem.Q))
        n_raw = assemble(fd_interpolate(θ_n_control, problem.Q))

    if hasattr(problem, "_fwd_norm_B"):
        θ_B_norm = assemble(fd_interpolate(problem._fwd_norm_B(θ_B_raw), problem.Q))
    else:
        θ_B_norm = assemble(
            fd_interpolate(
                ln(0.5 * θ_B_raw) + (1 - n_raw) / n_raw * problem.log_e_ref,
                problem.Q,
            )
        )

    normalized_controls = [θ_B_norm]
    if not problem.fixed_exponent:
        normalized_controls.append(θ_n_norm)
    if problem.infer_boundary_velocity:
        normalized_controls.extend(bc_controls)

    return _controls_to_array(normalized_controls)


def generate_predictions(
    z,
    field,
    problem,
    objective,
    converter=None,
    θ_np=None,
    fields_np=None,
    *,
    include_grid=True,
    include_mesh=False,
):
    """
    Run the forward model and export predictions on requested output layouts.

    By default the controls are built from the latent vector `z`. Passing
    `θ_np` overrides that path with a normalized control array directly.
    Passing `fields_np` provides a physical-space control array whose rheology
    fields are normalized here before evaluation. If `z` is omitted, the loss
    falls back to the data-objective portion when available.

    Parameters
    ----------
    include_grid
        If True, export gridded fields through `converter`. This preserves the
        historical behavior of this helper.
    include_mesh
        If True, also export velocity and boundary-control arrays directly on
        the inversion mesh.
    """
    if θ_np is not None and fields_np is not None:
        raise ValueError(
            "generate_predictions() accepts only one of `θ_np` or `fields_np`."
        )
    if z is None and θ_np is None and fields_np is None:
        raise ValueError(
            "generate_predictions() requires `z`, `θ_np`, or `fields_np`."
        )
    if include_grid and converter is None:
        raise ValueError(
            "generate_predictions() requires `converter` when `include_grid=True`."
        )

    problem.reset_solver()

    # Unpack to controls
    if θ_np is None:
        if fields_np is not None:
            θ_np = _normalize_physical_controls(fields_np, problem, objective)
        else:
            θ_np = field(z)

    eval_objective = objective
    if z is None and hasattr(objective, "data_objective"):
        eval_objective = objective.data_objective

    if hasattr(eval_objective, "evaluate_with_state"):
        loss, controls, preds = eval_objective.evaluate_with_state(z, x=θ_np)
    else:
        if z is None:
            raise ValueError(
                "generate_predictions() needs `z` when the objective cannot "
                "evaluate a direct control array."
            )
        loss = objective.loss(z)
        objective.set_control_array(θ_np)
        controls = objective.copy_controls()
        preds = problem.forward(controls)

    θ_B, θ_n, bc_controls = problem.split_controls(controls)

    # Evaluate rheology fields
    out_preds = {"z": z, "final_loss": loss}
    if include_grid:
        Ny_out, Nx_out = converter.out_shape
        x_out = np.linspace(converter.xarr[0], converter.xarr[-1], Nx_out)
        y_out = np.linspace(converter.yarr[0], converter.yarr[-1], Ny_out)
        out_preds["x"] = x_out
        out_preds["y"] = y_out
    if include_mesh:
        out_preds["mesh_coords"] = np.asarray(problem.coords)

    n = problem._inv_norm_n(θ_n)
    B = problem._inv_norm_B(θ_B)

    # Also evaluate reference viscosity field
    log_nu = ln(0.5 * B) + (1 - n) / n * problem.log_e_ref

    if include_grid:
        out_preds["B"] = converter(assemble(fd_interpolate(B, problem.Q)))
        out_preds["n"] = converter(assemble(fd_interpolate(n, problem.Q)))
        out_preds["log_nu"] = converter(assemble(fd_interpolate(log_nu, problem.Q)))

    for i, p in enumerate(preds):
        if include_grid:
            out_preds[f"u{i}"] = converter(p)
        if include_mesh:
            out_preds[f"u_mesh{i}"] = np.asarray(p.dat.data_ro.copy())

    if hasattr(field, "epoch_alpha_slices"):
        boundary_nodes = np.asarray(problem.boundary_node_indices)
        out_preds["boundary_s"] = np.asarray(problem.boundary_s)
        out_preds["boundary_L"] = np.asarray(field.boundary_length)
        out_preds["boundary_coords"] = np.asarray(problem.boundary_coords)
        out_preds["boundary_coeff_positions"] = np.asarray(field.coeff_positions)
        out_preds["boundary_basis_matrix"] = np.asarray(field.A)
        for i in range(field.Nt):
            sx, sy = field.epoch_alpha_slices(i)
            out_preds[f"u_bc{i}"] = np.asarray(
                bc_controls[i].dat.data_ro[boundary_nodes, :].copy()
            )
            out_preds[f"u_bc_ref{i}"] = np.asarray(
                problem.u_bc_ref[i].dat.data_ro[boundary_nodes, :].copy()
            )
            out_preds[f"u_obs_boundary{i}"] = np.asarray(
                problem.u_obs[i].dat.data_ro[boundary_nodes, :].copy()
            )
            if z is not None:
                out_preds[f"alpha_x{i}"] = np.asarray(z[sx])
                out_preds[f"alpha_y{i}"] = np.asarray(z[sy])
    else:
        for i, bc in enumerate(bc_controls):
            if include_grid:
                out_preds[f"u_bc{i}"] = converter(bc)
            if include_mesh:
                out_preds[f"u_bc_mesh{i}"] = np.asarray(bc.dat.data_ro.copy())

    return out_preds


def velocity_components_from_predictions(preds, n_steps):
    u0, v0 = preds["u0"]
    shape = (n_steps, *u0.shape)
    u_pred = np.zeros(shape, dtype=np.asarray(u0).dtype)
    v_pred = np.zeros(shape, dtype=np.asarray(v0).dtype)
    for i in range(n_steps):
        u_pred[i], v_pred[i] = preds[f"u{i}"]
    return u_pred, v_pred


def mesh_velocity_components_from_predictions(preds, n_steps):
    u0 = np.asarray(preds["u_mesh0"])
    if u0.ndim != 2 or u0.shape[1] != 2:
        raise ValueError(
            "Mesh velocity predictions must have shape (n_nodes, 2) for 2D "
            f"velocity fields, got {u0.shape}."
        )

    shape = (n_steps, u0.shape[0])
    u_pred = np.zeros(shape, dtype=u0.dtype)
    v_pred = np.zeros(shape, dtype=u0.dtype)
    for i in range(n_steps):
        u_mesh = np.asarray(preds[f"u_mesh{i}"])
        u_pred[i] = u_mesh[:, 0]
        v_pred[i] = u_mesh[:, 1]
    return u_pred, v_pred


# end of file
