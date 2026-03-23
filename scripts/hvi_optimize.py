#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import firedrake
import h5py
import jax
from jax import numpy as jnp
import numpy as np
import optax

import schurflow.file_utilities as fu
import schurflow.workflow_utils as wu
from schurflow.hvi import (
    SVIFullRankGaussianCorrelated,
    load_params,
    natural_gradient_transform,
    save_params,
)


def parse():
    parser = argparse.ArgumentParser(
        description="Run hierarchical Gaussian variational inference.",
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n_iter",
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
    return parser.parse_known_args()


def _num_mc_samples(cfg, n_proc):
    num_mc = getattr(cfg.vi, "num_mc", 0)
    return n_proc if num_mc == 0 else num_mc


def _hyperprior_config(cfg):
    field_map = {
        "sigma_B_min": "σ_B_min",
        "sigma_n_min": "σ_n_min",
        "sigma_B_default": "σ_B_default",
        "sigma_n_default": "σ_n_default",
        "rho_default": "ρ_default",
        "lambda_sigma": "λ_σ",
        "lambda_rho": "λ_ρ",
    }
    config = {}
    for cfg_key, model_key in field_map.items():
        value = getattr(cfg.vi, cfg_key, None)
        if value is not None:
            config[model_key] = value
    return config


def _initial_hyperparams(cfg):
    hyper_params = tuple(getattr(cfg.inverse, "hyper_params", ()))
    if len(hyper_params) == 3:
        return hyper_params
    if len(hyper_params) == 2:
        return hyper_params[0], hyper_params[1], 0.0
    return None, None, 0.0


def _build_optimizer(args, cfg, svi_model):
    lr = getattr(cfg.vi, "lr", cfg.inverse.lr)
    lr_hyper = getattr(cfg.vi, "lr_hyper", lr)
    param_labels = {
        "mean": "variational",
        "raw_tril": "variational",
        "log_σ_B_shifted": "hyper",
        "log_σ_n_shifted": "hyper",
        "raw_ρ": "hyper_corr",
    }

    if args.opt == "natgrad":
        return optax.multi_transform(
            {
                "variational": optax.chain(
                    natural_gradient_transform(svi_model.bij, damping=1e-4),
                    optax.clip_by_global_norm(5.0),
                    optax.scale_by_learning_rate(lr),
                ),
                "hyper": optax.adam(learning_rate=lr_hyper),
                "hyper_corr": optax.adam(learning_rate=0.5 * lr_hyper),
            },
            param_labels=param_labels,
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
    )
    mapfile = fu.make_results_filename(ctx.cfg, data_file=ctx.data_file)
    _, map_restore_group = wu.resolve_restore_stage(
        mapfile,
        args.map_restore_stage,
        default_to_latest=True,
    )

    mpi_print("Data file:", ctx.data_file)
    mpi_print("MAP file:", mapfile)
    mpi_print("Checkpoint file:", ctx.outfile)

    def compute_log_likelihood(x):
        try:
            nll, grad_nll = ctx.objective(x)
        except firedrake.exceptions.ConvergenceError:
            nll = np.nan
            grad_nll = np.full((ctx.field.N,), np.nan, dtype=np.float64)
        return -nll, -grad_nll

    svi_model = SVIFullRankGaussianCorrelated(
        loglik_fn=compute_log_likelihood,
        num_params_B=ctx.field.N_B,
        hyperprior_config=_hyperprior_config(ctx.cfg),
        num_mc=num_mc_samples,
        antithetic=True,
        dtype=jnp.float64,
        comm=comm,
    )

    with h5py.File(mapfile, "r") as fid:
        mpi_print(f"Restoring {map_restore_group} from {mapfile}")
        group = fid[map_restore_group]
        z_map = group["z"][()]
        H = group["hessian"][()]

    H_reg = H + ctx.cfg.vi.hess_reg * np.eye(z_map.size)
    init_σ_B, init_σ_n, init_ρ = _initial_hyperparams(ctx.cfg)
    params = svi_model.init_from_map_hessian(
        z_map,
        H_reg,
        init_σ_B=init_σ_B,
        init_σ_n=init_σ_n,
        init_ρ=init_ρ,
    )

    opt_state = None
    if args.restore:
        params, extra = load_params(ctx.outfile)
        opt_state = extra["opt_state"]
        loop_key = extra["key"]
    else:
        seed = getattr(ctx.cfg.inverse, "opt_init_seed", 803)
        loop_key = jax.random.PRNGKey(seed)

    optimizer = _build_optimizer(args, ctx.cfg, svi_model)
    if opt_state is None:
        opt_state = optimizer.init(params)

    def callback(t, hist, params, opt_state, key):
        if t % 10 == 0:
            logP, kl, σ_B, σ_n, ρ = [hist[:, j] for j in range(5)]
            elbo = logP[-1] - kl[-1]
            print(
                f"{t:5d} | "
                f"{elbo:13.6f} | "
                f"{logP[-1]:13.6f} | "
                f"{kl[-1]:13.6f} | "
                f"{σ_B[-1]:5.2f} | "
                f"{σ_n[-1]:5.2f} | "
                f"{ρ[-1]:4.2f}"
            )
        extra = {"hist": hist, "opt_state": opt_state, "key": key}
        save_params(ctx.outfile, params, extra=extra)

    params, opt_state, loop_key, hist = svi_model.fit(
        params,
        optimizer,
        opt_state,
        loop_key,
        num_steps=args.n_iter,
        jit=False,
        callback=callback,
    )
    save_params(
        ctx.outfile,
        params,
        extra={"hist": hist, "opt_state": opt_state, "key": loop_key},
    )


if __name__ == "__main__":
    args, extra_args = parse()
    cfg = wu.load_config(args, extra_args)
    main(cfg, args)


# end of file
