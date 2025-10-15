# -*- coding: utf-8 -*-

import numpy as np
import sys

import firedrake
from firedrake import (
    Constant,
    inner,
    sqrt,
    grad,
    dx,
    ds,
    sym,
    Identity,
    tr as trace,
    tanh,
)
from icepack.calculus import sym_grad, trace

import icepack
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    strain_rate_min,
)


def terminus_perturbation(**kwargs):
    r"""Return the terminus stress part of the ice shelf action functional

    The power exerted due to stress at the calving terminus :math:`\Gamma` is

    .. math::
       E(u) = \int_\Gamma\varrho gh^2u\cdot\nu\; ds

    We assume that sea level is at :math:`z = 0` for calculating the water
    depth.
    """
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    d_pert = kwargs.get("terminus_pert", 0.0)

    # Compute elevation of shelf base using hydrostatic approximation
    b = -1.0 * h * ρ_I / ρ_W

    # Compute time-dependent water depth
    d = b + d_pert

    # Compute hydrostatic stresses
    τ_I = ρ_I * g * h**2 / 2
    τ_W = ρ_W * g * d**2 / 2

    # Differential stress normal to the element faces
    ν = firedrake.FacetNormal(u.ufl_domain())
    return (τ_I - τ_W) * inner(u, ν)


def inv_norm_tanh(θ, low=0.1, high=5.0):
    A = 0.5 * (high - low)
    C0 = 0.5 * (high + low)
    return A * firedrake.tanh(θ) + C0


def unnormalize_rheology(**kwargs):
    θ_B = kwargs["norm_rigidity"]
    θ_n = kwargs["norm_exponent"]
    B = inv_norm_tanh(θ_B, low=kwargs["B_min"], high=kwargs["B_max"])
    n = inv_norm_tanh(θ_n, low=kwargs["n_min"], high=kwargs["n_max"])
    return B, n


def unnormalize_rheology_log(**kwargs):
    θ_B = kwargs["norm_rigidity"]
    θ_n = kwargs["norm_exponent"]
    B = firedrake.exp(θ_B)
    n = firedrake.exp(θ_n)
    return B, n


def normalized_rheology(**kwargs):
    B, n = unnormalize_rheology(**kwargs)
    return rigidity_with_exponent(rigidity=B, exponent=n, **kwargs)


def normalized_rheology_log(**kwargs):
    B, n = unnormalize_rheology_log(**kwargs)
    return rigidity_with_exponent(rigidity=B, exponent=n, **kwargs)


def rigidity_with_exponent(**kwargs):

    u = kwargs["velocity"]
    h = kwargs["thickness"]
    B = kwargs["rigidity"]
    n = kwargs["exponent"]
    ε_min = firedrake.Constant(strain_rate_min)

    ε = sym_grad(u)
    ε_e = effective_strain_rate(ε, ε_min)
    return 2 * n / (n + 1) * h * B * ε_e ** (1 / n + 1)


def effective_strain_rate(ε, ε_min):
    return sqrt((inner(ε, ε) + trace(ε) ** 2 + ε_min**2) / 2)


def velocity_to_effective(u):
    ε = sym_grad(u)
    ε_e = effective_strain_rate(ε, strain_rate_min)
    return ε_e


def viscosity_damaged(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    A = kwargs["fluidity"]
    D = kwargs["damage"]
    n = kwargs["exponent"]
    return custom_viscosity(
        velocity=u, thickness=h, fluidity=(1 - D) ** (-n) * A, exponent=n
    )


class CustomIceShelf(icepack.models.IceShelf):
    def __init__(
        self, viscosity=rigidity_with_exponent, terminus=terminus_perturbation
    ):
        super().__init__(viscosity=viscosity, terminus=terminus)


# Solver option constructors -------------------------------------------------


def _icepack_opts(dirichlet_ids, side_wall_ids, tol=1e-8, maxiter=100):
    return {
        "dirichlet_ids": dirichlet_ids,
        "side_wall_ids": side_wall_ids,
        "diagnostic_solver_type": "icepack",
        "tolerance": tol,
        "diagnostic_solver_parameters": {
            "tolerance": tol,
            "max_iterations": maxiter,
        },
    }


def _petsc_opts(dirichlet_ids, side_wall_ids, **_):
    return {
        "dirichlet_ids": dirichlet_ids,
        "side_wall_ids": side_wall_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_linesearch_type": "bt",
            "snes_max_it": 200,
        },
    }


def _trust_opts(dirichlet_ids, side_wall_ids, **_):
    return {
        "dirichlet_ids": dirichlet_ids,
        "side_wall_ids": side_wall_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtontr",
            "snes_max_it": 200,
        },
    }


def _cp_opts(dirichlet_ids, side_wall_ids, **_):
    return {
        "dirichlet_ids": dirichlet_ids,
        "side_wall_ids": side_wall_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_line_search_type": "cp",
        },
    }


def _multigrid_opts(dirichlet_ids, side_wall_ids, **_):
    return {
        "dirichlet_ids": dirichlet_ids,
        "side_wall_ids": side_wall_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "ksp_type": "cg",
            "pc_type": "mg",
            "pc_mg_cycle_type": "w",
            "snes_line_search_type": "cp",
        },
    }


SOLVER_OPTS = {
    "icepack": _icepack_opts,
    "petsc": _petsc_opts,
    "trust": _trust_opts,
    "cp": _cp_opts,
    "multigrid": _multigrid_opts,
}


def create_solver(
    model, solver_type, dirichlet_ids=[], side_wall_ids=[], tol=1e-8, maxiter=100
):

    try:
        opts = SOLVER_OPTS[solver_type](
            dirichlet_ids=dirichlet_ids,
            side_wall_ids=side_wall_ids,
            tol=tol,
            maxiter=maxiter,
        )
    except KeyError:
        raise KeyError(f"Unsupported solver type: {solver_type}")

    # Create the solver
    return icepack.solvers.FlowSolver(model, **opts), opts


# end of file
