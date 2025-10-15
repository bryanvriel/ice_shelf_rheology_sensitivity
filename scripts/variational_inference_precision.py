#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import h5py
import sys
import os

import jbao

import schurflow.train_utilities as tu
import schurflow.file_utilities as fu
import schurflow.vi as vi
from schurflow.utilities import *
from mpi4py import MPI

import jax.random as jr
import optax


def parse():
    parser = argparse.ArgumentParser(description="Run inverson for rigidity.")

    parser.add_argument("cfg", type=str, help="Input configuration file.")

    parser.add_argument(
        "--amp",
        action="store",
        type=str,
        default="mid",
        help="Amplitude from (low, mid, high). Default: mid.",
    )

    parser.add_argument(
        "--loc",
        action="store",
        type=str,
        default="terminus",
        help="Perturbation location from (terminus, GL). Default: terminus.",
    )

    parser.add_argument(
        "-r",
        action="store_true",
        dest="restore",
        help="Restore from previous solution.",
    )
    
    parser.add_argument(
        "-mrs",
        action="store",
        type=int,
        default=None,
        dest="map_restore_stage",
        help="Integer stage to restore MAP from. Default: None.",
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

    # Create objects
    field, problem, objective, converter = tu.initialize_objects(
        cfg,
        data_file,
        data_obj=True,
        comm=ensemble.comm,
    )

    # Filename for MAP solution
    mapfile = fu.generate_filename(cfg, args.loc, args.amp)
    mpi_print("Reading from MAP file", mapfile)

    # Get restore stage for MAP
    if args.map_restore_stage is not None:
        map_restore_stage = f"stage{args.map_restore_stage}"
    else:
        map_restore_stage = fu.get_latest_stage(mapfile)
        mpi_print(" - from group", map_restore_stage)

    # Restore MAP solution and Hessian
    with h5py.File(mapfile, "r") as fid:
        group = fid[map_restore_stage]
        z = group["z"][()]
        H = group["hessian"][()]

    # Filename for output SVI
    outfile = os.path.join(outdir, f"vi_params_{args.loc}_{args.amp}.pkl")
    print("Saving parameters to", outfile)

    # Create prior
    prior = vi.PriorGaussian.from_field(np.zeros_like(z), field)

    # Build VI
    config = vi.VIConfig(
        n_samples=18,
        lr=1e-3,
        num_steps=500,
        clip_grad_norm=5.0,
    )
    vi_obj = vi.PrecisionGaussian(
        loglik_fn=objective,
        prior=prior,
        config=config,
        comm=comm,
    )

    # Create natural gradient descent optimizer
    #tx = optax.chain(
    #    vi.natgrad_gaussian_precision(diag_shift=1e-8),
    #    optax.clip_by_global_norm(3.0),
    #    optax.trace(decay=0.8, nesterov=True),
    #    optax.scale_by_learning_rate(1e-4),
    #)
    tx = optax.adamw(1e-4)

    # Initialize or restore parameters
    if args.restore:
        init_params, extra = vi.load_params(outfile)
        opt_state = extra["opt_state"]
        key = extra["rng_key"]
    else:
        init_params = vi_obj.init_from_map_hessian(z, H)
        opt_state = tx.init(init_params)
        key = jr.PRNGKey(40)
    
    # Fit
    vi_obj.fit(init_params, key, tx, opt_state, checkfile=outfile)


if __name__ == "__main__":
    args, extra_args = parse()
    cfg = jbao.utils.parse_toml(args.cfg, default="default_config.toml")
    cfg = jbao.utils.update_cfg_with_extra(cfg, extra_args)
    main(cfg, args)

# end of file
