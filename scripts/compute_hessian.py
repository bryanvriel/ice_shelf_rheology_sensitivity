#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import sys

import schurflow.config_utilities as cu
import schurflow.file_utilities as fu
import schurflow.train_utilities as tu
import schurflow.workflow_utils as wu
import firedrake
from mpi4py import MPI


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
    
    ctx = wu.initialize_workflow(
        cfg,
        args,
        comm=ensemble.comm,
        data_obj=args.data_obj,
        require_vi=True,
    )
    Nt = ctx.problem.Nt

    mpi_print("Data file:", ctx.data_file)
    mpi_print("Output file:", ctx.outfile)

    # Get restore stage
    _, restore_stage = wu.resolve_restore_stage(
        ctx.outfile,
        args.restore_stage,
        default_to_latest=True,
    )

    # Load MAP
    mpi_print(f"Restoring {restore_stage} from {ctx.outfile}")
    with h5py.File(ctx.outfile, "r") as fid:
        θ_map = fid[f"{restore_stage}/z"][()]

    # Gradient test
    _, grad = ctx.objective(θ_map)
    grad_norm = np.linalg.norm(grad)
    mpi_print("Gradient norm at MAP:", grad_norm)

    hessian_group = "data_hessian" if args.data_obj else "hessian"

    # Read (un-regularized) Hessian or compute it
    if args.read_hessian:
        with h5py.File(ctx.outfile, "r") as fid:
            group = fid[f"{restore_stage}"]
            H = group[hessian_group][()]
    else:
        H = compute_hessian(ctx.objective, θ_map, comm, rank)
        if rank == 0:
            with h5py.File(ctx.outfile, "a") as fid:
                group = fid[f"{restore_stage}"]
                if hessian_group in group.keys():
                    del group[hessian_group]
                group[hessian_group] = H

    if args.compute_only:
        comm.Barrier()
        sys.exit()

    # Regularization
    mpi_print(f"Regularization factor: {ctx.cfg.vi.hess_reg}")
    H = H + ctx.cfg.vi.hess_reg * np.eye(H.shape[0])

    # Compute eigenvalues 
    evals, evecs = np.linalg.eigh(H)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    if args.plot:
        if rank == 0:
            plt.plot(evals)
            plt.xlabel("Eigenvalue index")
            plt.ylabel("Eigenvalue amplitude")
            plt.axhline(0.0, color="k")
            plt.show()

            print(evecs.shape)
            #plt.plot(evecs
        comm.Barrier()
        sys.exit()

    sampler = build_laplace_sampler(evals, evecs)

    n_samples = args.n_samples
    procN = n_samples // n_proc
    k0 = rank * procN
    if rank == (n_proc - 1):
        k1 = n_samples
    else:
        k1 = k0 + procN
    procN = k1 - k0

    mpi_print(f"Updating {ctx.outfile}")
        
    with h5py.File(ctx.outfile, "a", driver="mpio", comm=comm) as fid:

        root_group = fid[f"{restore_stage}"]

        hgroup = "hessian_samples"
        if hgroup in root_group.keys():
            del root_group[hgroup]
        group = root_group.create_group(hgroup)
        group["reg"] = ctx.cfg.vi.hess_reg

        def _make_ds(name, shape=None):
            if shape is None:
                if "u" in name or "v" in name:
                    shape = (n_samples, Nt, *ctx.converter.out_shape)
                else:
                    shape = (n_samples, *ctx.converter.out_shape)
            ds = group.create_dataset(name, shape, "f")
            return ds

        B_out = _make_ds("B_samples")
        n_out = _make_ds("n_samples")
        u_out = _make_ds("u_samples")
        v_out = _make_ds("v_samples")
        z_out = _make_ds("z_samples", shape=(n_samples, ctx.field.N))
        logp_out = group.create_dataset("logp", (n_samples,), "d")
        group["x"] = ctx.converter.x_out
        group["y"] = ctx.converter.y_out

        for i, k in enumerate(range(k0, k1)):

            sample = draw_laplace_sample(θ_map, sampler, seed=k)
            preds = tu.generate_predictions(
                sample.squeeze(),
                ctx.field,
                ctx.problem,
                ctx.objective,
                ctx.converter,
            )
            u_pred, v_pred = tu.velocity_components_from_predictions(preds, Nt)

            # Save to file
            B_out[k, ...] = preds["B"]
            n_out[k, ...] = preds["n"]
            u_out[k, ...] = u_pred
            v_out[k, ...] = v_pred
            z_out[k, ...] = sample
            logp_out[k] = -1 * preds["final_loss"]

            if rank == 0 and i % 5 == 0:
                print(f"Sample {i:3d} / {procN}")
                sys.stdout.flush()


def build_laplace_sampler(evals, evecs):
    if np.any(evals <= 0.0):
        raise ValueError(
            "Regularized Hessian must be positive definite before Laplace sampling."
        )
    return evecs / np.sqrt(evals)[None, :]


def draw_laplace_sample(x_map, sampler, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((sampler.shape[1],))
    return x_map + sampler.dot(Z)


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
    cfg = cu.load_config(
        args.cfg,
        extra_args=extra_args,
        cal=getattr(args, "cal", False),
        disc=getattr(args, "disc", False),
    )
    main(cfg, args)

# end of file
