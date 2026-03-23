# -*- coding: utf-8 -*-

import pickle
from typing import Iterable

import jax
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
import optax

from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


def save_params(outfile, params, extra=None):
    if extra is None:
        extra = {}
    with open(outfile, "wb") as fid:
        pickle.dump(params, fid)
        pickle.dump(extra, fid)


def load_params(outfile):
    with open(outfile, "rb") as fid:
        params = pickle.load(fid)
        extra = pickle.load(fid)
    return params, extra


def natural_gradient_transform(
    bijector: tfb.Bijector,
    damping: float = 1e-5,
    passthrough_keys: Iterable[str] = (),
) -> optax.GradientTransformation:
    passthrough_keys = tuple(passthrough_keys)

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(grads, state, params):
        dtype = params["mean"].dtype
        raw = params["raw_tril"]
        L = bijector.forward(raw)
        n = L.shape[0]

        g_m = grads["mean"]
        Lt_gm = L.T @ g_m
        nat_g_m = L @ Lt_gm + damping * g_m

        inv = lambda L_: bijector.inverse(L_)
        _, pullback_inv = jax.vjp(inv, L)
        g_raw = grads["raw_tril"]
        (g_L,) = pullback_inv(g_raw)
        g_L = jnp.tril(g_L)

        L_damped = L + damping * jnp.eye(n, dtype=dtype)
        Yt = solve_triangular(L_damped.T, g_L.T, lower=False)
        Y = Yt.T

        G_Sigma = 0.5 * (Y + Y.T)
        Z = L.T @ (G_Sigma @ L)
        nat_g_L = jnp.tril(L @ Z) + damping * g_L

        _, nat_g_raw = jax.jvp(inv, (L,), (nat_g_L,))

        updates = {
            "mean": nat_g_m,
            "raw_tril": nat_g_raw,
        }
        zero = jnp.array(0.0, dtype=dtype)
        for key in passthrough_keys:
            updates[key] = grads[key] if key in grads else zero

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# end of file
