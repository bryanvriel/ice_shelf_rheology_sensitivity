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
from schurflow.hvi import SVIFullRankGaussianCorrelated, natural_gradient_transform

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
        "-n_iter",
        action="store",
        type=int,
        default=10000,
        help="Number of iterations. Default: 10000.",
    )
    parser.add_argument(
        '--opt',
        action="store",
        type=str,
        default="natgrad",
        help="Optimizer type. 'natgrad' or optax optimizer. Default: 'natgrad'.",
    )
    return parser.parse_known_args()


def main(cfg, args):

    # Number of ensemble members is set to the number of MPI processes
    # We force each ensemble member to only use 1 MPI process
    comm = firedrake.COMM_WORLD
    ensemble = firedrake.Ensemble(comm, 1)
    n_proc = ensemble.ensemble_comm.size
    rank = ensemble.ensemble_comm.rank

    def mpi_print(*msg):
        if rank == 0:
            print(*msg)
        sys.stdout.flush()
        comm.Barrier()

    outdir = cfg.output_dir
    data_file = fu.data_filename(cfg, args)

    if cfg.vi.num_mc == 0:
        num_mc_samples = n_proc
    else:
        num_mc_samples = cfg.vi.num_mc

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

    # Hyperprior configuration
    hyperprior_config = {
        'σ_B_min': cfg.vi.sigma_B_min,
        'σ_n_min': cfg.vi.sigma_n_min,
        'σ_B_default': cfg.vi.sigma_B_default,
        'σ_n_default': cfg.vi.sigma_n_default,
        'ρ_default': cfg.vi.rho_default,
        'λ_σ': cfg.vi.lambda_sigma,
        'λ_ρ': cfg.vi.lambda_rho,
    }

    # Create HVI model
    svi_model = SVIFullRankGaussianCorrelated(
        loglik_fn=compute_log_likelihood,
        num_params_B=field.N_B,
        hyperprior_config=hyperprior_config,
        num_mc=num_mc_samples,
        antithetic=True,
        dtype=jnp.float64,
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
    H_reg = H + cfg.vi.hess_reg * np.eye(z_map.size)

    # Initial hyper parameters from config
    init_σ_B, init_σ_n, init_ρ = cfg.inverse.hyper_params

    # Initialize
    params = svi_model.init_from_map_hessian(
        z_map, H_reg, init_σ_B=init_σ_B, init_σ_n=init_σ_n, init_ρ=init_ρ,
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
        "log_σ_B_shifted": "hyper",
        "log_σ_n_shifted": "hyper",
        "raw_ρ": "hyper_corr",
    }

    # Create the optimizer
    if args.opt == "natgrad":
        optimizer = optax.multi_transform(
            {
                "variational": optax.chain(
                    natural_gradient_transform(svi_model.bij, damping=1e-4),
                    optax.clip_by_global_norm(5.0),
                    optax.scale_by_learning_rate(cfg.vi.lr),
                ),
                "hyper": optax.adam(learning_rate=cfg.vi.lr_hyper),
                "hyper_corr": optax.adam(learning_rate=0.5*cfg.vi.lr_hyper),
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
        logP, kl, σ_B, σ_n, ρ = [hist[:, j] for j in range(5)]
        elbo = logP[-1] - kl[-1]
        if t % 10 == 0:
            print(
                f'{t:5d} | '
                f'{elbo:13.6f} | '
                f'{logP[-1]:13.6f} | '
                f'{kl[-1]:13.6f} | '
                f'{σ_B[-1]:5.2f} | '
                f'{σ_n[-1]:5.2f} | '
                f'{ρ[-1]:4.2f}'
            )
        extra = {"hist": hist, "opt_state": opt_state, "key": key}
        save_params(paramfile, params, extra=extra)

    svi_model.fit(
        params,
        optimizer,
        opt_state,
        loop_key,
        num_steps=args.n_iter,
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
