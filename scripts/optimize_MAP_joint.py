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
        "-rs",
        action="store",
        type=int,
        default=1,
        dest="restore_stage",
        help="Integer stage to restore from. Default: 1.",
    )

    parser.add_argument(
        "-sgd",
        action="store_true",
        help="Force optimizer to be SGD.",
    )
    
    return parser.parse_known_args()


def main(cfg, args):

    outdir = cfg.output_dir
    data_file = fu.data_filename(cfg, args)

    # Create objects
    field, problem, objective, converter = tu.initialize_objects(cfg, data_file)

    # Output filename
    outfile = fu.generate_filename(cfg, args.loc, args.amp)
    print('Output file:', outfile)

    # Restore from previous
    if args.restore:
        with h5py.File(outfile, "r") as fid:
            z = fid[f"stage{args.restore_stage}/z"][()]
    else:
        rng = np.random.default_rng(2425)
        z = 0.05 * rng.standard_normal((field.N,))
    print('Number of parameters:', z.size)

    # Create optimizer
    lr_schedule = tu.create_lr_schedule(cfg.inverse.lr_schedule)
    opt, opt_state = tu.init_optimizer(cfg.inverse.opt_type, z, lr_schedule)
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

    # Run it
    z, opt_state, losses = tu.optimize_sgd(
        objective,
        z,
        opt,
        opt_state,
        cfg.inverse.n_iter,
        verbose=50,
        thresh=0,
    )

    preds = tu.generate_predictions(
        z,
        field,
        problem,
        objective,
        converter,
    )

    fu.write_results_to_file(
        outfile,
        preds,
        group=f"stage{stage}",
        mode=file_mode,
        extra_data={"losses": losses},
    )
    fu.save_state_to_group(
        opt_state, cfg.inverse.opt_type, outfile, f"stage{stage}/optimizer",
    )

    
if __name__ == "__main__":
    args, extra_args = parse()
    cfg = jbao.utils.parse_toml(args.cfg, default="default_config.toml")
    cfg = jbao.utils.update_cfg_with_extra(cfg, extra_args)
    main(cfg, args)

# end of file
