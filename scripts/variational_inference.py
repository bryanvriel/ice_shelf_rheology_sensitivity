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
import schurflow.workflow_utils as wu
from schurflow.vi import (
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
    return parser.parse_known_args()


def _num_mc_samples(cfg, n_proc):
    num_mc = getattr(cfg.vi, "num_mc", 0)
    return n_proc if num_mc == 0 else num_mc


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

    prior_mean = np.zeros(ctx.field.N)
    prior_precision = ctx.field.prior_precision()
    svi_model = SVIFullRankGaussian(
        compute_log_likelihood,
        prior_mean,
        prior_precision,
        num_mc=num_mc_samples,
        antithetic=True,
        comm=comm,
    )

    opt_state = None
    if args.restore:
        params, extra = load_params(ctx.outfile)
        opt_state = extra["opt_state"]
        loop_key = extra["key"]
    else:
        with h5py.File(mapfile, "r") as fid:
            mpi_print(f"Restoring {map_restore_group} from {mapfile}")
            group = fid[map_restore_group]
            z_map = group["z"][()]
            H = group["hessian"][()]

        H_reg = H + ctx.cfg.vi.hess_reg * np.eye(z_map.size)
        params = svi_model.init_from_map_hessian(z_map, H_reg)
        seed = getattr(ctx.cfg.inverse, "opt_init_seed", 1056)
        loop_key = jax.random.PRNGKey(seed)

    optimizer = _build_optimizer(args, ctx.cfg, svi_model)
    if opt_state is None:
        opt_state = optimizer.init(params)

    def callback(t, logP, kl, params, opt_state, key):
        if t % 10 == 0:
            elbo = logP[-1] - kl[-1]
            print(f"{t:5d} | {elbo:13.6f} | {logP[-1]:13.6f} | {kl[-1]:13.6f}")
        extra = {"logP": logP, "kl": kl, "opt_state": opt_state, "key": key}
        save_params(ctx.outfile, params, extra=extra)

    params, opt_state, loop_key, logP, kl = svi_model.fit(
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
        extra={"logP": logP, "kl": kl, "opt_state": opt_state, "key": loop_key},
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
