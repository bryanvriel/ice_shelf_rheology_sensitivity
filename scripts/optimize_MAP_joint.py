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
from schurflow.utilities import *

import optax

def parse():
    parser = argparse.ArgumentParser(
        description="Run joint inversion.",
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    return parser.parse_known_args()


def main(cfg, args):

    outdir = cfg.output_dir
    data_file = fu.data_filename(cfg, args)
    print('Data file:', data_file)

    # Create objects
    field, problem, objective, converter = tu.initialize_objects(
        cfg, data_file,
    )

    # Output filename
    outfile = fu.make_results_filename(cfg)
    print('Output file:', outfile)

    # Restore from previous
    if args.restore:
        with h5py.File(outfile, "r") as fid:
            z = fid[f"stage{args.restore_stage}/z"][()]
    else:
        rng = np.random.default_rng(4444)
        z = 0.05 * rng.standard_normal((field.N,))
        #z = np.zeros(field.N)
    print('Number of parameters:', z.size)

    # Create optimizer
    lr = tu.create_lr_schedule(cfg.inverse.lr_schedule, cfg.inverse.lr)
    opt, opt_state = tu.init_optimizer(cfg.inverse.opt_type, z, lr)
    print('Optimizer:', cfg.inverse.opt_type)
   
    # Attempt to restore optimizer state from previous stage 
    if args.restore:
        try:
            test_opt_state, opt_type = fu.load_state_from_group(
                opt_state,
                outfile,
                f"stage{args.restore_stage}/optimizer",
            )
            opt_state = test_opt_state
        except ValueError:
            print('Previous optimizer state is inconsistent. Not restoring state.')
    
    if args.restore:
        stage = args.restore_stage + 1
        file_mode = "a"
    else:
        stage = 1
        file_mode = "w"
    print("Optimizing for stage", stage)

    #z, opt_state, losses = tu.optimize_bfgs(
    #    objective,
    #    z,
    #    factr=0.001,
    #    maxcor=10,
    #    method="L-BFGS-B",
    #)
    #opt_state = opt.init(z)

    def callback(i, z, losses, grad_val):

        # Periodic diagnostic
        if i % 50 == 0 and grad_val is not None:
            gnorm = np.linalg.norm(grad_val)
            print(f"Iter {i:5d}: f = {losses[-1]}, g = {gnorm}")

        # Periodic prediction generation and savek
        if i % 100 == 0:

            # Generate predictions
            preds = tu.generate_predictions(
                z,
                field,
                problem,
                objective,
                converter,
            )

            # Write results to file
            fu.write_results_to_file(
                outfile,
                preds,
                group=f"stage{stage}",
                mode=file_mode,
                extra_data={"losses": losses},
            )

    # Run it
    z, opt_state, losses = tu.optimize_sgd(
        objective,
        z,
        opt,
        opt_state,
        cfg.inverse.n_iter,
        callback=callback,
    )
   
    callback(0, z, losses, None) 
    fu.save_state_to_group(
        opt_state, cfg.inverse.opt_type, outfile, f"stage{stage}/optimizer",
    )

    
if __name__ == "__main__":
    args, extra_args = parse()
    cfg = jbao.utils.parse_toml(args.cfg, default="default_config.toml")
    cfg = jbao.utils.update_cfg_with_extra(cfg, extra_args)
    main(cfg, args)

# end of file
