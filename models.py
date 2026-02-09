# -*- coding: utf-8 -*-

from operator import itemgetter
import numpy as np
from scipy.special import erfinv
import sys

import firedrake
from firedrake import (
    Constant,
    inner,
    sqrt,
    grad,
    dot,
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


def terminus_force(**kwargs):
    Fu = kwargs["force_action"]
    ν = firedrake.FacetNormal(Fu.ufl_domain())
    return inner(Fu, ν)


def general_neumann(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    traction = kwargs['traction']
    return h * inner(traction, u)


def inv_norm_tanh(θ, low=0.1, high=5.0, tau=1.0, **kwargs):
    A = 0.5 * (high - low)
    C0 = 0.5 * (high + low)
    return A * firedrake.tanh(θ / tau) + C0


def _arctanh(x):
    eps = firedrake.Constant(1e-8)
    x_clip = firedrake.min_value(firedrake.max_value(x, -1.0 + eps), 1.0 - eps)
    return 0.5 * firedrake.ln((1.0 + x_clip) / (1.0 - x_clip))


def fwd_norm_tanh(θ, low=0.1, high=5.0, tau=1.0, **kwargs):
    A = 0.5 * (high - low)
    C0 = 0.5 * (high + low)
    return tau * _arctanh((θ - C0) / A)


def inv_norm_arctan(θ, low=0.1, high=5.0, tau=1.0, **kwargs):
    m = 0.5 * (low + high)
    r = 0.5 * (high - low)
    s = 2 / np.pi * r
    return m + s * firedrake.atan(θ / tau)


def inv_norm_cdf(θ, low=0.1, high=5.0, tau=1.0, **kwargs):
    u = 0.5 * (1.0 + firedrake.erf(θ / (tau * firedrake.sqrt(2.0))))
    return low + (high - low) * u


def fwd_norm_cdf(x, low=0.1, high=5.0, tau=1.0, **kwargs):
    out = firedrake.Function(x.function_space())
    x_arr = x.dat.data_ro
    u = (x.dat.data_ro - low) / (high - low)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    out.dat.data[:] = tau * np.sqrt(2) * erfinv(2 * u - 1)
    return out


def inv_norm_log(θ, C0=1.0, **kwargs):
    return C0 * firedrake.exp(θ)


def fwd_norm_log(x, C0=1.0, **kwargs):
    return firedrake.ln(x) - firedrake.ln(C0)


def inv_norm_softplus(θ, low=0.1, high=5.0, tau=2, **kwargs):
    loc = 0.5 * (high + low)
    scale = (high - low) / (2 * tau)

    delta = firedrake.ln(firedrake.exp(loc) - 1.0)
    slope = 1 - firedrake.exp(-loc)
    latent_scale = scale / slope

    return firedrake.ln(1 + firedrake.exp(latent_scale * θ + delta))


def fwd_norm_softplus(x, low=0.1, high=5.0, tau=2, **kwargs):
    loc = 0.5 * (high + low)
    scale = (high - low) / (2 * tau)

    delta = firedrake.ln(firedrake.exp(loc) - 1.0)
    slope = 1 - firedrake.exp(-loc)
    latent_scale = scale / slope

    out = firedrake.Function(x.function_space())
    x_arr = np.maximum(x.dat.data_ro, 1e-10)
    theta = np.log(np.expm1(x_arr)) - delta
    out.dat.data[:] = theta / latent_scale
    return out


def fwd_norm_arctan(θ, low=0.1, high=5.0, tau=1.0, **kwargs):
    L = low
    U = high
    m = 0.5 * (L + U)
    r = 0.5 * (U - L)
    eps = firedrake.Constant(1e-6)

    z = (θ - m) / r
    zc = firedrake.min_value(firedrake.max_value(z, -1 + eps), 1 - eps)

    x = 0.5 * np.pi * zc
    return tau * firedrake.tan(x)


def unnormalize_rheology(**kwargs):
    θ_B = kwargs["norm_rigidity"]
    θ_n = kwargs["norm_exponent"]
    B = inv_norm_tanh(θ_B, low=kwargs["B_min"], high=kwargs["B_max"])
    n = inv_norm_tanh(θ_n, low=kwargs["n_min"], high=kwargs["n_max"])
    return B, n


def unnormalize_rheology_arctan(**kwargs):
    θ_B = kwargs["norm_rigidity"]
    θ_n = kwargs["norm_exponent"]
    #B = inv_norm_arctan(θ_B, low=kwargs["B_min"], high=kwargs["B_max"])
    B = firedrake.exp(θ_B)
    n = inv_norm_arctan(θ_n, low=kwargs["n_min"], high=kwargs["n_max"])
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


def normalized_rheology_arctan(**kwargs):
    B, n = unnormalize_rheology_arctan(**kwargs)
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


def log_viscosity(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    log_eta = kwargs["log_viscosity"]
    eta = firedrake.exp(log_eta)
    ε_min = firedrake.Constant(strain_rate_min)

    ε = sym_grad(u)
    ε_e_sq = inner(ε, ε) + trace(ε) ** 2 + ε_min**2

    return h * eta * ε_e_sq


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


def membrane_stress(**kwargs):
    """
    Computes stress tensor.
    """
    ε, B, n = itemgetter("strain_rate", "rigidity", "exponent")(kwargs)
    ε_e = effective_strain_rate(ε, strain_rate_min)
    μ = 0.5 * B * ε_e ** (1 / n - 1)
    I = Identity(ε.ufl_domain().geometric_dimension())
    return 2 * μ * (ε + trace(ε) * I)


def gravity(**kwargs):
    r"""Return the gravitational part of the ice shelf action functional

    The gravitational part of the ice shelf action functional is

    .. math::
        E(u) = -\frac{1}{2}\int_\Omega\varrho g\nabla h^2\cdot u\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness

    Returns
    -------
    firedrake.Form
    """
    u, h = itemgetter("velocity", "thickness")(kwargs)
    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return -0.5 * ρ * g * inner(grad(h**2), u)


def gravity_with_sea_surface(**kwargs):
    u, h = itemgetter("velocity", "thickness")(kwargs)

    # sea_surface is η(x,y,t); default 0
    η = kwargs.get("sea_surface", firedrake.Constant(0.0))

    ρ_red = ρ_I * (1 - ρ_I / ρ_W)

    # existing flotation driving term
    term_flotation = -0.5 * ρ_red * g * inner(grad(h**2), u)

    # NEW: barotropic tide / ocean pressure-gradient driving
    term_eta = -ρ_I * g * inner(h * grad(η), u)

    return term_flotation + term_eta


class CustomIceShelf(icepack.models.IceShelf):
    def __init__(
        self,
        viscosity=rigidity_with_exponent,
        terminus=terminus_perturbation,
        gravity=gravity,
    ):
        super().__init__(viscosity=viscosity, terminus=terminus, gravity=gravity)


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
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
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
