#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import argparse
import pickle
import h5py
import sys
import os

import firedrake
from mpi4py import MPI

import schurflow.train_utilities as tu
import schurflow.file_utilities as fu
from schurflow.hvi import SVIFullRankGaussian, natural_gradient_transform

import cryogen
import jbao

import jax
import jax.numpy as jnp
import optax
from optax.transforms import freeze


def parse():
    parser = argparse.ArgumentParser(
        description="Run SVI.",
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--opt',
        action="store",
        type=str,
        default="natgrad",
        help="Optimizer type. 'natgrad' or optax optimizer. Default: 'natgrad'.",
    )
    return parser.parse_known_args()


def main(args, extra_args):

    # Number of ensemble members is set to the number of MPI processes
    # We force each ensemble member to only use 1 MPI process
    comm = firedrake.COMM_WORLD
    ensemble = firedrake.Ensemble(comm, 1)
    n_proc = ensemble.ensemble_comm.size
    rank = ensemble.ensemble_comm.rank

    lr = 5e-4
    num_iter = 10000

    #num_mc_samples = n_proc
    num_mc_samples = 18

    beta_kl = 1.0

    def mpi_print(*msg):
        if rank == 0:
            print(*msg)
        sys.stdout.flush()
        comm.Barrier()

    # Load configuration
    extra_args.extend(["--data.amp", args.amp])
    extra_args.extend(["--data.loc", args.loc])
    cfg = jbao.utils.parse_toml(args.cfg, default="default_config.toml")
    cfg = jbao.utils.update_cfg_with_extra(cfg, extra_args)
    outdir = cfg.output_dir
    data_file = fu.data_filename(cfg, args)

    # Create objects
    # We set data_obj=True to create a data-only objective
    field, problem, objective, converter = tu.initialize_objects(
        cfg, data_file, data_obj=True, comm=ensemble.comm,
    )

    def compute_log_likelihood(x):
        try:
            nll, grad_nll = objective(x)
        except firedrake.exceptions.ConvergenceError:
            nll = np.nan
            grad_nll = np.full((field.N,), np.nan, dtype=np.float64)
        return -nll, -grad_nll

    # Create SVI model
    hyper = {'B': (1.0, 0.5), 'n': (2.5, 0.8)} # see test_hvi.ipynb for visualization
    svi_model = SVIFullRankGaussian(
        compute_log_likelihood,
        num_params_B=field.N_B,
        prior_var_hyperpriors=hyper,
        num_mc=num_mc_samples,
        antithetic=True,
        beta_kl=beta_kl,
        comm=comm,
    )

    ext = f"{num_mc_samples:02d}_{args.opt[:3]}.pkl"
    paramfile = fu.make_results_filename(cfg, prefix="v", ext=ext)

    # Load MAP results
    mapfile = fu.make_results_filename(cfg)
    restore_stage = fu.get_latest_stage(mapfile)
    with h5py.File(mapfile, "r") as fid:
        print(f"Restoring {restore_stage} from {mapfile}")
        group = fid[f"{restore_stage}"]
        z_map = group["z"][()]
        H = group["hessian"][()]

    # NOTE:REGULARIZATION can be important
    H_reg = H + 0.5 * np.eye(z_map.size)

    # Initial variances from penalty parameters
    init_log_var_B = np.log(1.0 / cfg.inverse.penalty_B)
    init_log_var_n = np.log(1.0 / cfg.inverse.penalty_n)

    # Initialize
    params = svi_model.init_from_map_hessian(
        z_map, H_reg, (init_log_var_B, init_log_var_n),
    )

    # If restoring, overwrite parameters
    opt_state = None
    if args.restore:
        print('restoring')
        params, extra = load_params(paramfile)
        opt_state = extra['opt_state']
        loop_key = extra['key']
    else:
        seed = 803
        key = jax.random.PRNGKey(seed)
        init_key, loop_key = jax.random.split(key)

    param_labels = {
        "mean": "variational",
        "raw_tril": "variational",
        "log_prior_var_B": "hyper_B",
        "log_prior_var_n": "hyper_n",
    }

    # Create the optimizer
    if args.opt == "natgrad":
        optimizer = optax.multi_transform(
            {
                "variational": optax.chain(
                    natural_gradient_transform(svi_model.bij, damping=1e-4),
                    optax.clip_by_global_norm(5.0),
                    optax.scale_by_learning_rate(8e-4),
                ),
                "hyper_B": optax.adam(learning_rate=2e-4),
                "hyper_n": optax.adam(learning_rate=2e-4),
            },
            param_labels=param_labels,
        )
        
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(4.0),
            getattr(optax, args.opt)(learning_rate=lr),
        )

    # Initialize optimizer state if not restored
    if opt_state is None:
        opt_state = optimizer.init(params)

    # Run the optimization loop 
    def callback(t, hist, params, opt_state, key):
        logP, kl, var_B, var_n = [hist[:, j] for j in range(4)]
        elbo = logP[-1] - kl[-1]
        if t % 10 == 0:
            print(
                f'{t:5d} | '
                f'{elbo:13.6f} | '
                f'{logP[-1]:13.6f} | '
                f'{kl[-1]:13.6f} | '
                f'{var_B[-1]:6.3f} | '
                f'{var_n[-1]:6.3f}'
            )
        extra = {"hist": hist, "opt_state": opt_state, "key": key}
        save_params(paramfile, params, extra=extra)

    svi_model.fit(
        params,
        optimizer,
        opt_state,
        loop_key,
        num_steps=num_iter,
        jit=False,
        callback=callback,
    )


def save_params(outfile, params, extra={}):
    with open(outfile, "wb") as fid:
        pickle.dump(params, fid)
        pickle.dump(extra, fid)


def load_params(outfile):
    with open(outfile, "rb") as fid:
        params = pickle.load(fid)
        extra = pickle.load(fid)
    return params, extra


if __name__ == "__main__":
    args, extra_args = parse()
    main(args, extra_args)

# end of file
