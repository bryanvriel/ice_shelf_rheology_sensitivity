#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import warnings
import argparse
import h5py
import sys
import os

import jbao

import schurflow.train_utilities as tu
import schurflow.file_utilities as fu
import schurflow.models as models
from schurflow.utilities import *
from mpi4py import MPI
import cryogen


def parse():
    parser = argparse.ArgumentParser(
        description="Run joint inversion.",
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-rh",
        action="store_true",
        dest="read_hessian",
        help="Read pre-computed Hessian.",
    )

    parser.add_argument(
        "-p",
        action="store_true",
        dest="plot",
        help="Plot eigenvalues and exit.",
    )

    parser.add_argument(
        "-c",
        action="store_true",
        dest="compute_only",
        help="Only compute Hessian without sampling.",
    )

    parser.add_argument(
        "-d",
        action="store_true",
        dest="data_obj",
        help="Use data objective only (not posterior).",
    )

    parser.add_argument(
        "-n",
        action="store",
        type=int,
        default=200,
        dest="n_samples",
        help="Number of samples to generate. Default: 200.",
    )

    return parser.parse_known_args()


def main(args, extra_args):

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
    
    # Load configuration
    cfg = jbao.utils.parse_toml(args.cfg, default="default_config.toml")
    extra_args.extend(["--data.amp", args.amp])
    extra_args.extend(["--data.loc", args.loc])
    cfg = jbao.utils.update_cfg_with_extra(cfg, extra_args)
    outdir = cfg.output_dir
    data_file = fu.data_filename(cfg, args)

    # Create objects
    field, problem, objective, converter = tu.initialize_objects(
        cfg,
        data_file,
        comm=ensemble.comm,
        data_obj=args.data_obj,
    )
    Nt = problem.Nt
        
    # Inverse filename
    outfile = fu.make_results_filename(cfg)

    # Get restore stage
    if args.restore_stage is not None:
        restore_stage = f"stage{args.restore_stage}"
    else:
        restore_stage = fu.get_latest_stage(outfile)

    # Load MAP
    mpi_print(f'Restoring {restore_stage} from {outfile}')
    with h5py.File(outfile, "r") as fid:
        θ_map = fid[f"{restore_stage}/z"][()]
        Np = θ_map.size

    # Gradient test
    _, grad = objective(θ_map)
    grad_norm = np.linalg.norm(grad)
    mpi_print('Gradient norm at MAP:', grad_norm)
    
    # Read (un-regularized) Hessian or compute it
    if args.read_hessian:
        with h5py.File(outfile, "r") as fid:
            group = fid[f"{restore_stage}"]
            H = group["hessian"][()]
    else:
        H = compute_hessian(objective, θ_map, comm, rank)
        if rank == 0:
            out_group = "data_hessian" if args.data_obj else "hessian"
            with h5py.File(outfile, "a") as fid:
                group = fid[f"{restore_stage}"]
                if out_group in group.keys():
                    del group[out_group]
                group[out_group] = H

    if args.compute_only:
        comm.Barrier()
        sys.exit()

    # Regularization
    H = H + cfg.vi.hess_reg * np.eye(H.shape[0])
    cov = np.linalg.inv(H)

    # Compute eigenvalues 
    evals, evecs = np.linalg.eigh(H)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    if args.plot:
        if rank == 0:
            plt.plot(evals)
            plt.xlabel('Eigenvalue index')
            plt.ylabel('Eigenvalue amplitude')
            plt.axhline(0.0, color='k')
            plt.show()

            print(evecs.shape)
            #plt.plot(evecs
        comm.Barrier()
        sys.exit()

    n_samples = args.n_samples
    procN = n_samples // n_proc
    k0 = rank * procN
    if rank == (n_proc - 1):
        k1 = n_samples
    else:
        k1 = k0 + procN
    procN = k1 - k0

    mpi_print(f'Updating {outfile}')
        
    with h5py.File(outfile, "a", driver="mpio", comm=comm) as fid:

        root_group = fid[f"{restore_stage}"]

        hgroup = f"hessian_samples"
        if hgroup in root_group.keys():
            del root_group[hgroup]
        group = root_group.create_group(hgroup)
        group['reg'] = cfg.vi.hess_reg

        def _make_ds(name, shape=None):
            if shape is None:
                if 'u' in name or 'v' in name:
                    shape = (n_samples, Nt, *converter.out_shape)
                else:
                    shape = (n_samples, *converter.out_shape)
            ds = group.create_dataset(name, shape, 'f')
            return ds

        B_out = _make_ds('B_samples')
        n_out = _make_ds('n_samples')
        u_out = _make_ds('u_samples')
        v_out = _make_ds('v_samples')
        z_out = _make_ds('z_samples', shape=(n_samples, field.N))
        logp_out = group.create_dataset('logp', (n_samples,), 'd')
        group['x'] = converter.x_out
        group['y'] = converter.y_out

        for i, k in enumerate(range(k0, k1)):

            # Generate a sample and update controls with sample
            sample = laplace_low_rank_sampler(θ_map, evals, evecs, seed=k)
            θ_np = field(sample.squeeze())
            objective.controls = θ_np

            # Evaluate log probability
            nll = objective.loss(sample.squeeze())
            logp_out[k] = -1 * nll

            # Run forward model
            problem.reset_solver()
            preds = problem.forward(objective.controls)
            u_pred = np.zeros((Nt, *converter.out_shape), dtype='f')
            v_pred = np.zeros((Nt, *converter.out_shape), dtype='f')
            for j in range(Nt):
                up = converter(preds[j])
                u_pred[j] = up[0]
                v_pred[j] = up[1]

            # Unpack control fields
            if problem.fixed_exponent:
                θ_B = objective.controls
                θ_n = problem.θ_n
            else: 
                θ_B, θ_n = objective.controls

            # Un-normalize and convert to array
            B = problem._inv_norm_B(θ_B)
            n = problem._inv_norm_n(θ_n)
            B = converter(firedrake.assemble(firedrake.interpolate(B, problem.Q)))
            n = converter(firedrake.assemble(firedrake.interpolate(n, problem.Q)))

            # Save to file
            B_out[k, ...] = B
            n_out[k, ...] = n
            u_out[k, ...] = u_pred
            v_out[k, ...] = v_pred
            z_out[k, ...] = sample

            if rank == 0 and i % 5 == 0:
                print(f"Sample {i:3d} / {procN}")
                sys.stdout.flush()


def laplace_low_rank_sampler(x_map, evals, evecs, seed=None):
    # draw in the reduced m-space
    m = evals.shape[0]
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((m,))
    # scale by 1/sqrt(evals)
    scaled = evecs / np.sqrt(evals)[None,:]
    return x_map + scaled.dot(Z)


def compute_hessian(objective, θ_map, comm, rank, method="ad"):

    # 1. Get total number of workers
    size = comm.Get_size()
    Np = θ_map.size

    if method == "ad":
        hvp_fn = lambda v: objective.hvp(θ_map, v)
    elif method == "fd":
        hvp_fn = lambda v: objective.hvp_finite_diff(θ_map, v, eps=1e-6)
    else:
        raise ValueError

    def _print(msg):
        print(msg)
        sys.stdout.flush()

    if rank == 0:
        _print(f'Computing Hessian on {size} MPI ranks')

    # 2. Initialize a local container with Zeros
    # Each rank will only fill in specific rows here
    H_local = np.zeros((Np, Np))

    # 3. Distribute the loop (Cyclic Distribution)
    # Rank 0 does 0, 4, 8... Rank 1 does 1, 5, 9... (if size=4)
    for i in range(rank, Np, size):
        e_i = np.zeros(Np)
        e_i[i] = 1.0
        
        # Compute only the row assigned to this rank
        H_local[i, :] = hvp_fn(e_i)
        
        # Optional: Print progress (only Rank 0 reports its own progress to avoid spam)
        if rank == 0 and i % 5 == 0:
             _print(f"{i:3d}/{Np} (Rank 0 progress)")

    # 4. Aggregate results using Allreduce
    # Since H_local is 0.0 everywhere except the rows computed by the specific rank,
    # summing them up reconstructs the full matrix on every rank.
    H_global = np.zeros((Np, Np))
    comm.Allreduce(H_local, H_global, op=MPI.SUM)

    # 5. Symmetrize (Computation is cheap, can be done on all ranks)
    H_global = 0.5 * (H_global + H_global.T)

    return H_global

 
if __name__ == "__main__":
    args, extra_args = parse()
    main(args, extra_args)

# end of file
