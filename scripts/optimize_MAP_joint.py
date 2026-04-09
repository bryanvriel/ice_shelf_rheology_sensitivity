#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import h5py

import schurflow.config_utilities as cu
import schurflow.train_utilities as tu
import schurflow.file_utilities as fu
import schurflow.workflow_utils as wu


def parse():
    parser = argparse.ArgumentParser(
        description="Run joint inversion.",
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_known_args()


def main(cfg, args):
    ctx = wu.initialize_workflow(cfg, args)
    print("Data file:", ctx.data_file)
    print("Output file:", ctx.outfile)
    metadata = tu.augment_boundary_metadata(
        tu.inference_metadata(ctx.field),
        ctx.field,
        ctx.problem,
    )

    # Restore from previous
    if args.restore:
        restore_stage, restore_group = wu.resolve_restore_stage(
            ctx.outfile,
            args.restore_stage,
            default_to_latest=True,
        )
        with h5py.File(ctx.outfile, "r") as fid:
            z = fid[f"{restore_group}/z"][()]
        stage = restore_stage + 1
        file_mode = "a"
    else:
        z = tu.initialize_latent_vector(
            ctx.field,
            seed=ctx.cfg.inverse.opt_init_seed,
            amp=ctx.cfg.inverse.opt_init_amp,
        )
        stage = 1
        file_mode = "w"
    tu.print_parameter_summary(ctx.field)

    # Create optimizer
    if ctx.cfg.inverse.opt_type != "bfgs":
        lr = tu.create_lr_schedule(ctx.cfg.inverse.lr_schedule, ctx.cfg.inverse.lr)
        opt, opt_state = tu.init_optimizer(ctx.cfg.inverse.opt_type, z, lr)
        print("Optimizer:", ctx.cfg.inverse.opt_type)

    # Attempt to restore optimizer state from previous stage
    if args.restore and ctx.cfg.inverse.opt_type != "bfgs":
        try:
            test_opt_state, _ = fu.load_state_from_group(
                opt_state,
                ctx.outfile,
                f"{restore_group}/optimizer",
            )
            opt_state = test_opt_state
        except ValueError:
            print("Previous optimizer state is inconsistent. Not restoring state.")
    print("Optimizing for stage", stage)

    def callback(i, z, opt_state, losses, grad_val):

        # Periodic diagnostic
        if i % 50 == 0 and grad_val is not None:
            gnorm = np.linalg.norm(grad_val)
            print(f"Iter {i:5d}: f = {losses[-1]}, g = {gnorm}")

        # Periodic prediction generation and savek
        if i % 100 == 0:

            # Generate predictions
            preds = tu.generate_predictions(
                z,
                ctx.field,
                ctx.problem,
                ctx.objective,
                ctx.converter,
            )

            # Write parameters and losses to file
            fu.write_results_to_file(
                ctx.outfile,
                preds,
                group=f"stage{stage}",
                mode=file_mode,
                extra_data={"losses": losses, **metadata},
            )

            fu.save_cfg_to_group(
                ctx.cfg,
                ctx.outfile,
                f"stage{stage}/cfg",
                verbose=False,
            )

            # Also write optimizer state
            fu.save_state_to_group(
                opt_state,
                ctx.cfg.inverse.opt_type,
                ctx.outfile,
                f"stage{stage}/optimizer",
                verbose=False,
            )

    if ctx.cfg.inverse.opt_type == "bfgs":
        z, opt_state, losses = tu.optimize_bfgs(
            ctx.objective,
            z,
            factr=ctx.cfg.inverse.bfgs_factr,
            maxcor=ctx.cfg.inverse.bfgs_maxcor,
            method="L-BFGS-B",
            maxiter=1000,
        )

    else:
        z, opt_state, losses = tu.optimize_sgd(
            ctx.objective,
            z,
            opt,
            opt_state,
            ctx.cfg.inverse.n_iter,
            callback=callback,
        )

    # Final saving of parameters and states
    callback(0, z, opt_state, losses, None)


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
