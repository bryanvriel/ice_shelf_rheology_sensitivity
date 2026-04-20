#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import firedrake
import h5py
import jax
import numpy as np
import optax

import schurflow.config_utilities as cu
import schurflow.file_utilities as fu
import schurflow.train_utilities as tu
import schurflow.workflow_utils as wu
from schurflow.vi import (
    QuadraticControlVariate,
    SVIFullRankGaussian,
    load_params,
    natural_gradient_transform,
    save_params,
)


def parse():
    parser = argparse.ArgumentParser(
        description="Run Gaussian variational inference.",
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_iter",
        action="store",
        type=int,
        default=10000,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--opt",
        action="store",
        type=str,
        default="natgrad",
        help="Optimizer type. Use `natgrad` or any optax optimizer name.",
    )
    parser.add_argument(
        "-mrs",
        action="store",
        type=int,
        default=None,
        dest="map_restore_stage",
        help="Integer MAP stage to restore from. Default: latest stage.",
    )
    parser.add_argument(
        "--mapfile",
        action="store",
        type=str,
        default=None,
        help=(
            "Override the MAP results file used for initialization. "
            "If omitted, an .h5 value passed to --output is also treated as a "
            "MAP-file override."
        ),
    )
    parser.add_argument(
        "--control_variate",
        action="store",
        type=str,
        choices=("none", "quadratic"),
        default=None,
        help="Variance-reduction control variate for the data likelihood estimator.",
    )
    return parser.parse_known_args()


def _num_mc_samples(cfg, n_proc):
    num_mc = getattr(cfg.vi, "num_mc", 0)
    return n_proc if num_mc == 0 else num_mc


def resolve_quadratic_control_variate_curvature(
    *,
    data_hessian,
    posterior_hessian,
    prior_precision,
    objective_scale,
):
    if data_hessian is not None:
        neg_loglik_hessian = np.asarray(data_hessian, dtype=np.float64)
        source = "data_hessian"
    else:
        neg_loglik_hessian = (
            np.asarray(posterior_hessian, dtype=np.float64)
            - float(objective_scale) * np.asarray(prior_precision, dtype=np.float64)
        )
        source = "posterior_minus_prior"
    neg_loglik_hessian = 0.5 * (neg_loglik_hessian + neg_loglik_hessian.T)
    return neg_loglik_hessian, source


def _build_optimizer(args, cfg, svi_model):
    lr = getattr(cfg.vi, "lr", cfg.inverse.lr)
    if args.opt == "natgrad":
        return optax.chain(
            natural_gradient_transform(svi_model.bij, damping=1e-4),
            optax.clip_by_global_norm(5.0),
            optax.scale_by_learning_rate(lr),
        )

    return optax.chain(
        optax.clip_by_global_norm(4.0),
        getattr(optax, args.opt)(learning_rate=lr),
    )


def main(cfg, args):
    comm = firedrake.COMM_WORLD
    ensemble = firedrake.Ensemble(comm, 1)
    n_proc = ensemble.ensemble_comm.size
    rank = ensemble.ensemble_comm.rank

    def mpi_print(*msg):
        if rank == 0:
            print(*msg)
        sys.stdout.flush()
        comm.Barrier()

    num_mc_samples = _num_mc_samples(cfg, n_proc)
    checkpoint_ext = f"{num_mc_samples:02d}_{args.opt[:3]}.pkl"
    ctx = wu.initialize_workflow(
        cfg,
        args,
        comm=ensemble.comm,
        data_obj=True,
        require_vi=True,
        outfile_prefix="v",
        outfile_ext=checkpoint_ext,
        match_output_ext=True,
    )
    mapfile = wu.resolve_map_file(ctx.cfg, args, data_file=ctx.data_file)
    _, map_restore_group = wu.resolve_restore_stage(
        mapfile,
        args.map_restore_stage,
        default_to_latest=True,
    )

    mpi_print("Data file:", ctx.data_file)
    mpi_print("MAP file:", mapfile)
    mpi_print("Checkpoint file:", ctx.outfile)
    tu.print_parameter_summary(ctx.field, printer=mpi_print)
    metadata = tu.augment_boundary_metadata(
        tu.inference_metadata(ctx.field),
        ctx.field,
        ctx.problem,
    )

    def compute_log_likelihood(x):
        try:
            nll, grad_nll = ctx.objective(x)
        except firedrake.exceptions.ConvergenceError:
            nll = np.nan
            grad_nll = np.full((ctx.field.N,), np.nan, dtype=np.float64)
        return -nll, -grad_nll

    prior_mean = np.zeros(ctx.field.N)
    prior_precision = ctx.field.prior_precision()
    control_variate_mode = args.control_variate or getattr(
        ctx.cfg.vi,
        "control_variate",
        "none",
    )
    if control_variate_mode not in {"none", "quadratic"}:
        raise ValueError(
            f"Unsupported VI control variate mode: {control_variate_mode!r}"
        )
    control_variate = None
    control_variate_source = "none"
    control_variate_enabled = False
    z_map = None
    H_post = None

    if not args.restore or control_variate_mode == "quadratic":
        with h5py.File(mapfile, "r") as fid:
            mpi_print(f"Restoring {map_restore_group} from {mapfile}")
            group = fid[map_restore_group]
            z_map = group["z"][()]
            H_post = group["hessian"][()]
            data_hessian = group["data_hessian"][()] if "data_hessian" in group else None

        if control_variate_mode == "quadratic":
            neg_loglik_hessian, control_variate_source = (
                resolve_quadratic_control_variate_curvature(
                    data_hessian=data_hessian,
                    posterior_hessian=H_post,
                    prior_precision=prior_precision,
                    objective_scale=ctx.cfg.inverse.objective_scale,
                )
            )
            ref_loglik, ref_grad_loglik = compute_log_likelihood(z_map)
            if np.isfinite(ref_loglik) and np.all(np.isfinite(ref_grad_loglik)):
                control_variate = QuadraticControlVariate(
                    ref_theta=z_map,
                    ref_loglik=ref_loglik,
                    ref_grad_loglik=ref_grad_loglik,
                    neg_loglik_hessian=neg_loglik_hessian,
                    source=control_variate_source,
                )
                control_variate_enabled = True
            else:
                mpi_print(
                    "Warning: MAP log-likelihood evaluation was non-finite; "
                    "disabling quadratic control variate."
                )
                control_variate_source = "disabled_nonfinite_map"

    mpi_print("Control variate mode:", control_variate_mode)
    mpi_print("Control variate enabled:", control_variate_enabled)
    mpi_print("Control variate source:", control_variate_source)
    svi_model = SVIFullRankGaussian(
        compute_log_likelihood,
        prior_mean,
        prior_precision,
        num_mc=num_mc_samples,
        antithetic=True,
        comm=comm,
        control_variate=control_variate,
    )

    opt_state = None
    if args.restore:
        params, extra = load_params(ctx.outfile)
        opt_state = extra["opt_state"]
        loop_key = extra["key"]
    else:
        H_reg = H_post + ctx.cfg.vi.hess_reg * np.eye(z_map.size)
        params = svi_model.init_from_map_hessian(z_map, H_reg)
        seed = getattr(ctx.cfg.inverse, "opt_init_seed", 1056)
        loop_key = jax.random.PRNGKey(seed)

    optimizer = _build_optimizer(args, ctx.cfg, svi_model)
    if opt_state is None:
        opt_state = optimizer.init(params)

    def callback(t, logP, kl, cv_active, params, opt_state, key):
        if t % 10 == 0:
            elbo = logP[-1] - kl[-1]
            print(f"{t:5d} | {elbo:13.6f} | {logP[-1]:13.6f} | {kl[-1]:13.6f}")
        extra = {
            "logP": logP,
            "kl": kl,
            "cv_active": cv_active,
            "opt_state": opt_state,
            "key": key,
            "metadata": metadata,
            "mapfile": mapfile,
            "map_restore_group": map_restore_group,
            "control_variate_mode": control_variate_mode,
            "control_variate_source": control_variate_source,
            "control_variate_enabled": np.int8(control_variate_enabled),
        }
        save_params(ctx.outfile, params, extra=extra)

    params, opt_state, loop_key, logP, kl, cv_active = svi_model.fit(
        params,
        optimizer,
        opt_state,
        loop_key,
        num_steps=args.n_iter,
        jit=False,
        callback=callback,
    )
    if rank == 0:
        # Only rank 0 owns the populated histories and updated optimizer state.
        save_params(
            ctx.outfile,
            params,
            extra={
                "logP": logP,
                "kl": kl,
                "cv_active": cv_active,
                "opt_state": opt_state,
                "key": loop_key,
                "metadata": metadata,
                "mapfile": mapfile,
                "map_restore_group": map_restore_group,
                "control_variate_mode": control_variate_mode,
                "control_variate_source": control_variate_source,
                "control_variate_enabled": np.int8(control_variate_enabled),
            },
        )


if __name__ == "__main__":
    args, extra_args = parse()
    cfg = cu.load_config(
        args.cfg,
        extra_args=extra_args,
        cal=getattr(args, "cal", False),
        disc=getattr(args, "disc", False),
    )
    main(cfg, args)


# end of file
