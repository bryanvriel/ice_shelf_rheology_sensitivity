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
    argv = sys.argv[1:]
    cfg_arg = None
    if argv and not argv[0].startswith("-"):
        cfg_arg = argv[0]
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Run joint inversion.",
        parents=[fu.common_parse(include_cfg=False)],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cfg",
        action="store",
        type=str,
        default=None,
        help=(
            "Input configuration file. If omitted, load the saved config "
            "from the MAP results file passed with --output."
        ),
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
        "-p_path",
        action="store",
        type=str,
        default=None,
        help="Path to save figure instead of showing it.",
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

    args, extra_args = parser.parse_known_args(argv)
    if args.cfg is not None and cfg_arg is not None:
        parser.error("Pass the config at most once, either positionally or via --cfg.")
    if args.cfg is None:
        args.cfg = cfg_arg
    return args, extra_args


def load_runtime_config(args, extra_args):
    cfg_arg = getattr(args, "cfg", None)
    if cfg_arg is False:
        cfg_arg = None
    elif isinstance(cfg_arg, str):
        cfg_arg = cfg_arg.strip()
        if cfg_arg == "":
            cfg_arg = None

    if cfg_arg is not None:
        return cu.load_config(
            cfg_arg,
            extra_args=extra_args,
            cal=getattr(args, "cal", False),
            disc=getattr(args, "disc", False),
        )

    results_file = getattr(args, "output", None)
    if results_file is None:
        raise ValueError(
            "No config file was provided. Pass either a config path or "
            "`--output <map-results.h5>` so the config can be loaded from the "
            "results file."
        )

    cfg = fu.load_cfg_from_results_file(
        results_file,
        stage=args.restore_stage,
        validate=False,
        require_vi=False,
    )
    if extra_args:
        cfg = cu.update_cfg_with_extra(cfg, extra_args)
    cfg = cu.apply_calibration_overrides(
        cfg,
        cal=getattr(args, "cal", False),
        disc=getattr(args, "disc", False),
    )
    return cu.validate_cfg(cfg, require_vi=True)


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

    def root_h5_write(outfile, update):
        # Serialize rank-0 writes so they never overlap with other ranks'
        # independent HDF5 reads on the same file.
        comm.Barrier()
        if rank == 0:
            with h5py.File(outfile, "a") as fid:
                update(fid)
                fid.flush()
        comm.Barrier()

    ctx = wu.initialize_workflow(
        cfg,
        args,
        comm=ensemble.comm,
        data_obj=args.data_obj,
        require_vi=True,
    )
    Nt = ctx.problem.Nt
    metadata = tu.augment_boundary_metadata(
        tu.inference_metadata(ctx.field),
        ctx.field,
        ctx.problem,
    )

    mpi_print("Data file:", ctx.data_file)
    mpi_print("Output file:", ctx.outfile)
    tu.print_parameter_summary(ctx.field, printer=mpi_print)

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

    def _write_metadata(group, values):
        for key, value in values.items():
            if key in group:
                del group[key]
            group[key] = value

    root_h5_write(ctx.outfile, lambda fid: _write_metadata(fid[f"{restore_stage}"], metadata))

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

        def _write_hessian(fid):
            group = fid[f"{restore_stage}"]
            if hessian_group in group.keys():
                del group[hessian_group]
            group[hessian_group] = H

        root_h5_write(ctx.outfile, _write_hessian)

    if args.compute_only:
        comm.Barrier()
        sys.exit()

    # Compute un-regularized eigenvalues
    evals, evecs = np.linalg.eigh(H)
    mpi_print(f"Min un-regularized eigenvalue:", np.min(evals))

    # Regularization
    mpi_print(f"Regularization factor: {ctx.cfg.vi.hess_reg}")
    H = H + ctx.cfg.vi.hess_reg * np.eye(H.shape[0])

    # Compute regularized eigenvalues
    evals, evecs = np.linalg.eigh(H)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    if args.plot:
        if rank == 0:

            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5.5), layout='constrained')
            ax1.plot(evals)
            ax2.semilogy(evals)
            ax2.set_xlabel("Eigenvalue index")
            ax1.set_ylim(-5, 20)
            ax1.axhline(0.0, color="k")

            if args.p_path is not None:
                fig.savefig(args.p_path, dpi=200, bbox_inches="tight")
            else:
                plt.show()

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

        save_boundary = int(metadata["infer_boundary_velocity"])
        if save_boundary:
            n_basis = int(metadata["boundary_n_basis"])
            n_boundary = int(np.asarray(metadata["boundary_s"]).size)
            alpha_x_out = group.create_dataset(
                "alpha_x_samples",
                (n_samples, Nt, n_basis),
                "d",
            )
            alpha_y_out = group.create_dataset(
                "alpha_y_samples",
                (n_samples, Nt, n_basis),
                "d",
            )
            u_bc_out = group.create_dataset(
                "u_bc_boundary_samples",
                (n_samples, Nt, n_boundary, 2),
                "f",
            )

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

            if save_boundary:
                for j in range(Nt):
                    alpha_x_out[k, j, :] = preds[f"alpha_x{j}"]
                    alpha_y_out[k, j, :] = preds[f"alpha_y{j}"]
                    u_bc_out[k, j, ...] = preds[f"u_bc{j}"]

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
    cfg = load_runtime_config(args, extra_args)
    main(cfg, args)

# end of file
