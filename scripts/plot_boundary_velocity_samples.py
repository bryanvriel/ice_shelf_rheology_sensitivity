#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import schurflow.config_utilities as cu
import schurflow.file_utilities as fu
from schurflow.fields import (
    resolve_boundary_basis_config,
    closed_boundary_length,
    periodic_bspline_coefficient_prior_covariance_corrlen,
    periodic_bspline_design_matrix,
)
from schurflow.problems import IceShelfProblem, make_mesh


def parse():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize observed boundary velocities, the fitted boundary "
            "reference, and random periodic spline deviations about that "
            "reference for one epoch."
        ),
        parents=[fu.common_parse()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Zero-based epoch index from the data file to visualize.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of random spline realizations to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the spline coefficient draws.",
    )
    parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help="Multiply the sampled spline coefficients by this factor.",
    )
    parser.add_argument(
        "--mesh-degree",
        type=int,
        default=2,
        help="Mesh function-space degree used when loading the problem.",
    )
    parser.add_argument(
        "--quiver-samples",
        type=int,
        default=3,
        help="How many random realizations to show in the plan-view quiver panel.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution when saving to disk.",
    )
    parser.add_argument(
        "--fit-boundary-reference",
        action="store_true",
        dest="fit_boundary_reference",
        help="Force the boundary reference fit on for this plot.",
    )
    parser.add_argument(
        "--no-fit-boundary-reference",
        action="store_true",
        dest="no_fit_boundary_reference",
        help="Force the boundary reference fit off for this plot.",
    )
    parser.add_argument(
        "--smooth-boundary",
        action="store_true",
        dest="fit_boundary_reference",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-smooth-boundary",
        action="store_true",
        dest="no_fit_boundary_reference",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure even if --output is provided.",
    )
    return parser.parse_known_args()


def load_num_epochs(data_file):
    with h5py.File(data_file, "r") as fid:
        return int(fid["u"].shape[0])


def resolve_reference_fit(cfg, args):
    if args.fit_boundary_reference and args.no_fit_boundary_reference:
        raise ValueError(
            "Cannot pass both --fit-boundary-reference and "
            "--no-fit-boundary-reference."
        )

    if args.fit_boundary_reference:
        return True
    if args.no_fit_boundary_reference:
        return False
    return bool(cfg.inverse.smooth_boundary)


def initialize_problem(cfg, data_file, args):
    n_epochs_total = load_num_epochs(data_file)
    if args.epoch < 0 or args.epoch >= n_epochs_total:
        raise ValueError(
            f"Epoch {args.epoch} is out of range for {n_epochs_total} epochs."
        )

    mesh = make_mesh(meshfile=cfg.inverse.meshfile)
    kwargs = dict(
        nll_scale=cfg.inverse.nll_scale,
        degree=args.mesh_degree,
        B_norm=cfg.inverse.B_norm_type,
        n_norm=cfg.inverse.n_norm_type,
        default_vel_error=cfg.data.u_sigma,
        dirichlet_ids=cfg.inverse.dirichlet_ids,
        infer_boundary_velocity=False,
    )
    if cfg.inverse.joint:
        problem = IceShelfProblem(cfg, mesh, **kwargs)
        problem.load_observations(data_file, time_ind=[args.epoch])
    else:
        problem = IceShelfProblem(cfg, mesh, fixed_exponent=True, **kwargs)
        problem.load_observations(data_file, time_ind=[args.epoch])

    if resolve_reference_fit(cfg, args):
        problem.smooth_boundaries()
    else:
        problem._init_boundary_geometry()

    return problem


def sample_boundary_realizations(problem, cfg, n_samples, seed, sample_scale):
    boundary_coords = np.asarray(problem.boundary_coords, dtype=float)
    boundary_s = np.asarray(problem.boundary_s, dtype=float)
    boundary_nodes = np.asarray(problem.boundary_node_indices, dtype=int)
    boundary_length = closed_boundary_length(boundary_coords)

    degree = int(cfg.inverse.boundary_spline_degree)
    sigma_alpha = float(cfg.inverse.boundary_sigma_alpha)
    corr_length = float(cfg.inverse.boundary_corr_length)
    n_basis, basis_spacing, basis_spacing_target = resolve_boundary_basis_config(
        cfg,
        boundary_length,
    )

    A = periodic_bspline_design_matrix(
        boundary_s,
        boundary_length,
        n_basis=n_basis,
        degree=degree,
    )
    Sigma_alpha, coeff_positions = periodic_bspline_coefficient_prior_covariance_corrlen(
        n=n_basis,
        L=boundary_length,
        sigma_alpha=sigma_alpha,
        corr_length=corr_length,
        design_matrix=A,
        return_positions=True,
    )

    rng = np.random.default_rng(seed)
    L_alpha = np.linalg.cholesky(Sigma_alpha)
    alpha_x = sample_scale * (rng.standard_normal((n_samples, n_basis)) @ L_alpha.T)
    alpha_y = sample_scale * (rng.standard_normal((n_samples, n_basis)) @ L_alpha.T)

    u_obs = problem.u_obs[0].dat.data_ro[boundary_nodes, :].copy()
    u_ref = problem.u_bc_ref[0].dat.data_ro[boundary_nodes, :].copy()

    samples = np.repeat(u_ref[None, :, :], n_samples, axis=0)
    for i in range(n_samples):
        samples[i, :, 0] += A @ alpha_x[i]
        samples[i, :, 1] += A @ alpha_y[i]

    return {
        "boundary_coords": boundary_coords,
        "boundary_s": boundary_s,
        "boundary_length": boundary_length,
        "n_basis": n_basis,
        "basis_spacing": basis_spacing,
        "basis_spacing_target": basis_spacing_target,
        "reference_corr_length": getattr(problem, "boundary_ref_corr_length", None),
        "coeff_positions": coeff_positions,
        "u_obs": u_obs,
        "u_ref": u_ref,
        "samples": samples,
        "alpha_x": alpha_x,
        "alpha_y": alpha_y,
        "A": A,
    }


def append_periodic_endpoint(coords, values):
    coords_closed = np.vstack((coords, coords[:1]))
    values_closed = np.vstack((values, values[:1]))
    return coords_closed, values_closed


def draw_profile_panel(ax, s, obs, ref, samples, label, ref_label):
    for i, sample in enumerate(samples):
        ax.plot(
            s,
            sample,
            color="C0",
            alpha=0.22,
            linewidth=1.0,
            label="Random BC realizations" if i == 0 else None,
        )

    ax.plot(s, obs, color="0.65", linewidth=1.6, label="Observed boundary")
    ax.plot(s, ref, color="k", linewidth=2.0, label=ref_label)
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)


def draw_quiver_panel(ax, coords, u_obs, u_ref, samples, quiver_samples):
    coords_closed, _ = append_periodic_endpoint(coords, u_ref)
    ax.plot(coords_closed[:, 0], coords_closed[:, 1], color="0.8", linewidth=1.0)

    stride = max(1, len(coords) // 40)
    arrow_coords = coords[::stride]
    ref_speed = np.linalg.norm(u_ref, axis=1)
    coord_span = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        1.0,
    )
    vel_scale = max(float(ref_speed.max()), 1.0)
    arrow_scale = 0.05 * coord_span / vel_scale

    ax.quiver(
        arrow_coords[:, 0],
        arrow_coords[:, 1],
        u_obs[::stride, 0] * arrow_scale,
        u_obs[::stride, 1] * arrow_scale,
        color="0.7",
        alpha=0.8,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        label="Observed",
    )
    ax.quiver(
        arrow_coords[:, 0],
        arrow_coords[:, 1],
        u_ref[::stride, 0] * arrow_scale,
        u_ref[::stride, 1] * arrow_scale,
        color="k",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0030,
        label="Reference",
    )

    n_show = min(quiver_samples, len(samples))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_show, 1)))
    for i in range(n_show):
        ax.quiver(
            arrow_coords[:, 0],
            arrow_coords[:, 1],
            samples[i, ::stride, 0] * arrow_scale,
            samples[i, ::stride, 1] * arrow_scale,
            color=colors[i],
            alpha=0.45,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0020,
            label="Sample BC realizations" if i == 0 else None,
        )

    ax.set_title("Plan View")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def make_figure(plot_data, args, ref_label, epoch_label):
    s = np.asarray(plot_data["boundary_s"])
    u_obs = np.asarray(plot_data["u_obs"])
    u_ref = np.asarray(plot_data["u_ref"])
    samples = np.asarray(plot_data["samples"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")

    draw_profile_panel(
        axes[0, 0],
        s,
        u_obs[:, 0],
        u_ref[:, 0],
        samples[:, :, 0],
        "u_x",
        ref_label,
    )
    axes[0, 0].set_title("Boundary x-velocity")

    draw_profile_panel(
        axes[0, 1],
        s,
        u_obs[:, 1],
        u_ref[:, 1],
        samples[:, :, 1],
        "u_y",
        ref_label,
    )
    axes[0, 1].set_title("Boundary y-velocity")

    draw_profile_panel(
        axes[1, 0],
        s,
        np.linalg.norm(u_obs, axis=1),
        np.linalg.norm(u_ref, axis=1),
        np.linalg.norm(samples, axis=2),
        "|u|",
        ref_label,
    )
    axes[1, 0].set_title("Boundary speed")
    axes[1, 0].set_xlabel("Arc length")

    draw_quiver_panel(
        axes[1, 1],
        np.asarray(plot_data["boundary_coords"]),
        u_obs,
        u_ref,
        samples,
        quiver_samples=args.quiver_samples,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="best", frameon=False)

    fig.suptitle(
        (
            f"Boundary reference fit and spline deviations "
            f"(epoch {epoch_label}, n_samples={args.n_samples}, "
            f"sample_scale={args.sample_scale:g}, "
            f"n_basis={int(plot_data['n_basis'])}, "
            f"ref_corr_length={plot_data['reference_corr_length']:g})"
            if plot_data["reference_corr_length"] is not None
            else
            f"Boundary reference and spline deviations "
            f"(epoch {epoch_label}, n_samples={args.n_samples}, "
            f"sample_scale={args.sample_scale:g}, "
            f"n_basis={int(plot_data['n_basis'])})"
        ),
        fontsize=13,
    )
    return fig


def main(cfg, args):
    data_file = fu.data_filename(cfg, args)
    print("Data file:", data_file)
    print("Mesh file:", cfg.inverse.meshfile)
    print("Epoch:", args.epoch)

    problem = initialize_problem(cfg, data_file, args)
    plot_data = sample_boundary_realizations(
        problem,
        cfg,
        n_samples=args.n_samples,
        seed=args.seed,
        sample_scale=args.sample_scale,
    )
    print(
        "Boundary basis:",
        f"n_basis={int(plot_data['n_basis'])},",
        f"spacing={float(plot_data['basis_spacing']):.1f}",
    )
    if plot_data["reference_corr_length"] is not None:
        print(
            "Boundary reference:",
            "periodic B-spline fit,",
            f"ref_corr_length={float(plot_data['reference_corr_length']):.1f}",
        )
    else:
        print("Boundary reference: raw observed boundary values")

    ref_label = (
        "Fitted BC reference"
        if resolve_reference_fit(cfg, args)
        else "BC reference"
    )
    fig = make_figure(plot_data, args, ref_label, epoch_label=args.epoch)

    if args.output:
        outdir = os.path.dirname(os.path.abspath(args.output))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print("Saved figure:", args.output)

    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args, extra_args = parse()
    cfg = cu.load_config(
        args.cfg,
        extra_args=extra_args,
        cal=getattr(args, "cal", False),
        disc=getattr(args, "disc", False),
    )
    main(cfg, args)
