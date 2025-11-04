# -*- coding: utf-8 -*-

from typing import Callable, Dict, Optional, Tuple
import numpy as np

import jax
from jax import numpy as jnp
from jax import random, value_and_grad
from jax.scipy.linalg import solve_triangular, cho_solve
import optax

from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print('No mpi4py installed. No MPI functionality can be used.')

Array = jnp.ndarray
Params = Dict[str, Array]  # {"mean": (n,), "raw_tril": (n*(n+1)//2,)}
Grads = Dict[str, Array]

class SVIFullRankGaussian:
    """
    Stateless SVI helper for a full-rank Gaussian variational family.

    - Keeps config only (loglik_fn, prior, dtype, MC settings, bijector).
    - Does NOT store params; all methods accept a params dict explicitly:
          params = {"mean": (n,), "raw_tril": (n*(n+1)//2,)}

    API:
      params0 = SVIFullRankGaussian.init_from_map_hessian(...)
      neg_elbo, metrics = svi.loss_with_metrics(params0, rng_key)
      posterior = svi.posterior(params0)
      samples = svi.sample(rng_key, 256, params0)
      params, opt_state, hist = svi.fit(params0, tx, opt_state, num_steps=2000, seed=0)
    """

    def __init__(
        self,
        loglik_fn: Callable[[Array], Array],
        prior_mean: Array,
        prior_precision: Array,
        *,
        num_mc: int = 32,
        antithetic: bool = True,
        dtype=jnp.float64,
        diag_shift: float = 1e-5,
        beta_kl: float = 1.0,
        comm = None,
    ):
        self.loglik_fn = loglik_fn
        self.prior_mean = jnp.asarray(prior_mean, dtype=dtype)
        self.prior_precision = jnp.asarray(prior_precision, dtype=dtype)
        self.num_mc = int(num_mc)
        self.antithetic = bool(antithetic)
        self.dtype = dtype
        self.n_dim = prior_mean.size
        self.β_kl = beta_kl

        # Bijector mapping unconstrained vector -> lower-triangular scale
        # with positive diag
        self._bij = tfb.FillScaleTriL(
            diag_bijector=tfb.Softplus(),
            diag_shift=jnp.asarray(diag_shift, dtype=dtype),
        )

        # Precompute log|Q| once
        chol_Q = jnp.linalg.cholesky(self.prior_precision)
        self._logdet_Q = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_Q)))

        # MPI logic
        if comm is not None:
            self.comm = comm
            self.mpi_rank = comm.rank
            self.num_mc_local = num_mc // comm.size
            if self.mpi_rank == (comm.size - 1):
                self.num_mc_local = num_mc - self.mpi_rank * self.num_mc_local

            # Allocate arrays for scattering and gathering
            self.scalar_counts = comm.allgather(self.num_mc_local)
            self.vector_counts = comm.allgather(self.n_dim * self.num_mc_local)
            self.eps = np.zeros((self.num_mc_local, self.n_dim))
            if self.mpi_rank == 0:
                self.logP = np.zeros(num_mc)
                self.grad_logP = np.zeros(self.n_dim * num_mc)
            else:
                self.logP = self.grad_logP = None

        else:
            self.mpi_rank = 0

    # ----- Private MPI helpers -----

    def _scatter_eps(self, array):
        self.comm.Scatterv(
            [array, self.vector_counts, MPI.DOUBLE], self.eps, root=0,
        )

    def _gather_logP(self, array):
        self.comm.Gatherv(
            array,
            [self.logP, self.scalar_counts, MPI.DOUBLE],
            root=0,
        )

    def _gather_grad_logP(self, array):
        self.comm.Gatherv(
            array,
            [self.grad_logP, self.vector_counts, MPI.DOUBLE],
            root=0,
        )

    # ----- Public properties and helpers -----

    @property
    def bij(self):
        return self._bij

    @bij.setter
    def bij(self, value):
        raise ValueError("Cannot set bij")

    # Initializer
    def init_from_map_hessian(
        self,
        theta_map: Array,
        hessian_at_map: Array,
    ) -> Params:
        """
        Build params dict from MAP and Hessian: Σ0 ≈ (H + jitter I)^{-1}, L0 = chol(Σ0).
        Returns params = {"mean": θ_MAP, "raw_tril": bij.inverse(L0)}.
        """
        dtype = self.dtype
        theta_map = jnp.asarray(theta_map, dtype=dtype)
        H = jnp.asarray(make_spd(hessian_at_map), dtype=dtype)
        n = theta_map.shape[0]

        C = jnp.linalg.cholesky(H)                             # H = C C^T
        Sigma0 = cho_solve((C, True), jnp.eye(n, dtype=dtype)) # Solve H * Σ_0 = I
        Sigma0 = 0.5 * (Sigma0 + Sigma0.T)
        L0 = jnp.linalg.cholesky(Sigma0)

        raw_tril0 = self._bij.inverse(L0).astype(dtype)
        return {"mean": theta_map, "raw_tril": raw_tril0}

    # ---------- Public evaluation helpers ----------

    def scale_tril(self, params: Params) -> Array:
        """Return L = chol(Σ) from params."""
        return self._bij.forward(params["raw_tril"]).astype(self.dtype)

    def sample_reparam(self, params: Params, eps: Array) -> Array:
        """
        Deterministic, differentiable sampler suitable for VJP/grad:
            eps ~ N(0, I) with shape (M, d)
            theta = m + eps @ L^T
        First argument is `params` (PyTree), so we can do vjp wrt params.
        """
        m = params["mean"].astype(self.dtype) # (d,)
        L = self.scale_tril(params)           # (d, d), lower-tri
        return m[None, :] + eps @ L.T         # (M, d)

    def sample_eps(self, key: Array, n: int, antithetic: Optional[bool] = None) -> Array:
        """
        Samples only the normal variable eps to be used in reparameterization.
        """
        use_antithetic = self.antithetic if antithetic is None else bool(antithetic)
        if use_antithetic:
            n_even = (n // 2) * 2
            key, sub = random.split(key)
            eps = random.normal(sub, shape=(n_even // 2, self.n_dim), dtype=self.dtype)
            eps = jnp.concatenate([eps, -eps], axis=0)
        else:
            key, sub = random.split(key)
            eps = random.normal(sub, shape=(n, self.n_dim), dtype=self.dtype)
        return eps

    def sample(
        self,
        params: Params,
        key: Array,
        n: int,
        *args,
        antithetic: Optional[bool] = None,
    ) -> Array:
        """
        Thin sampling wrapper: calls sample_eps followed by sample_reparam.
        """
        eps = self.sample_eps(key, n, antithetic)
        return self.sample_reparam(params, eps)

    def kl_fullrank_prior(
        self,
        params: Params,
    ) -> Array:
        """
        KL( N(m, Σ=L L^T) || N(μ0, Q^{-1}) )
          = 0.5 * [ tr(QΣ) + (m-μ0)^T Q (m-μ0) - n - log|Q| - log|Σ| ],
        with log|Σ| = 2 * sum(log diag(L)).
        """
        # Pack the variational params
        m = params["mean"]
        L = self.scale_tril(params)
        n = m.shape[0]

        # Cache the prior params
        mu0 = self.prior_mean
        Q = self.prior_precision
        logdet_Q = self._logdet_Q

        # Compute KL
        QL = Q @ L
        tr_QSigma = jnp.sum(QL * L)
        delta = (m - mu0)
        quad = delta @ (Q @ delta)
        logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return 0.5 * self.β_kl * (tr_QSigma + quad - n - logdet_Q - logdet_Sigma)

    def elbo_loss_vmap(
        self, params: Params, rng_key: Array
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Compute negative ELBO and components for given params & RNG.

        ELBO = E_q[log p(d|θ)] - KL(q || N(μ0, Q^{-1}))
        Returns (neg_elbo, metrics_dict)
        """
        # Cache some parameters
        dtype = self.dtype
        M = int(self.num_mc)

        # Sample from variational distribution
        thetas = self.sample(params, rng_key, M)

        # Compute log likelihood values over Monte Carlo samples
        ll_vals = jax.vmap(self.loglik_fn)(thetas)
        logP = jnp.mean(ll_vals)

        # Compute KL-divergence (deterministic)
        kl = self.kl_fullrank_prior(params)

        # Compute ELBO
        elbo = logP - kl
        neg_elbo = -elbo
        metrics = dict(elbo=elbo, logP=logP, kl=kl)
        return neg_elbo, metrics

    def elbo_loss_mpi(
        self, params: Params, rng_key: Array
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Compute negative ELBO and components for given params & RNG.

        ELBO = E_q[log p(d|θ)] - KL(q || N(μ0, Q^{-1}))
        Returns (neg_elbo, metrics_dict)
        """
        # Root does sampling from normal distribution; scatter to workers
        if self.mpi_rank == 0:
            eps = self.sample_eps(rng_key, self.num_mc)
        else:
            eps = None
        self._scatter_eps(eps)

        # Compute samples from variational distribution
        thetas = np.asarray(self.sample_reparam(params, self.eps))

        # Log likelihood
        logP_local = np.zeros(self.num_mc_local)
        grad_logP_local = np.zeros((self.num_mc_local, self.n_dim))
        for i in range(self.num_mc_local):
            logP_local[i], grad_logP_local[i, :] = self.loglik_fn(thetas[i])

        # Gather the results
        self._gather_logP(logP_local)
        self._gather_grad_logP(grad_logP_local)

        # Only root proceeds with total loss and gradient computation
        if self.mpi_rank != 0:
            return None, None, None

        # Mask for only the samples that converged (non-NaN)
        valid_mask = np.isfinite(self.logP)
        num_valid_mc = np.sum(valid_mask)

        # Compute log likelihood values over Monte Carlo samples
        logP = jnp.mean(self.logP[valid_mask])
        grad_logP = self.grad_logP.reshape(-1, self.n_dim)[valid_mask]
        eps = eps[valid_mask]

        # vjp to get gradient of logP w.r.t. variational parameters
        _, vjpfun = jax.vjp(
            lambda p: self.sample_reparam(p, eps),
            params,
        )
        grad_logP = vjpfun(grad_logP)[0]

        # Compute KL-divergence (deterministic)
        kl, grad_kl = jax.value_and_grad(
            lambda p: self.kl_fullrank_prior(p)
        )(params)

        # Negative ELBO and grad
        loss = -1 * (logP - kl)
        grad_loss = jax.tree_util.tree_map(
            lambda x, y: -1 * (x / num_valid_mc - y),
            grad_logP,
            grad_kl,
        )
        metrics = dict(
            elbo=-loss,
            logP=logP,
            kl=kl,
        )
        return loss, grad_loss, metrics

    # ---------- Training ----------
    def fit(
        self,
        params: Params,
        tx: optax.GradientTransformation,
        opt_state: optax.OptState,
        rng: Array,
        num_steps: int = 1000,
        jit: bool = False,
        callback: Optional[Callable] = None,
    ) -> Tuple[Params, optax.OptState, Dict[str, Array]]:
        """
        Optimize `params` using an externally provided optimizer & state.

        Returns (updated_params, updated_opt_state, history_dict).
        """

        if jit:
            @jax.jit
            def step(pr: Params, opt_state, rng):
                rng, sub = random.split(rng)
                (loss, metrics), grads = value_and_grad(
                    self.elbo_loss_vmap, has_aux=True
                )(pr, sub)
                updates, opt_state = tx.update(grads, opt_state, pr)
                pr = optax.apply_updates(pr, updates)
                return pr, opt_state, rng, loss, metrics
        else:
            def step(pr: Params, opt_state, rng):
                rng, sub = random.split(rng)
                loss, grads, metrics = self.elbo_loss_mpi(pr, sub)
                if self.mpi_rank == 0:
                    updates, opt_state = tx.update(grads, opt_state, pr)
                    pr = optax.apply_updates(pr, updates)
                pr = self.comm.bcast(pr, root=0)
                return pr, opt_state, rng, loss, metrics

        logP = np.zeros(num_steps)
        kl = np.zeros(num_steps)

        for t in range(num_steps):
            params, opt_state, rng, loss, metrics = step(params, opt_state, rng)

            if self.mpi_rank == 0 and callback is not None:
                logP[t] = metrics["logP"]
                kl[t] = metrics["kl"]
                callback(t, logP[:t+1], kl[:t+1], params, opt_state, rng)

        return params, opt_state, rng, logP, kl

# ---------- Natural Gradient Descent transformation --------

def make_natgrad_transform_for_svi(
    *,
    bijector: tfb.Bijector,
    dtype=jnp.float64,
    cov_damping: float = 0.0,   # optional: Σ ← Σ + λI inside preconditioner
    project_lower: bool = True, # keep lower-tri tangents for stability
) -> optax.GradientTransformation:
    """
    Natural-gradient preconditioner for q(θ)=N(m, Σ), params:
      params["mean"]     : (d,)
      params["raw_tril"] : (d*(d+1)//2,)
    with L = bijector.forward(raw_tril), Σ = L L^T.

    Returns an Optax GradientTransformation that maps Euclidean grads
    to natural-gradient directions in the SAME PyTree (mean, raw_tril).
    Place FIRST in optax.chain(...).
    """

    def init_fn(params: Params):
        return optax.EmptyState()

    def update_fn(grads: Grads, state, params: Params):
        # Unpack & reconstruct
        m   = params["mean"].astype(dtype)
        raw = params["raw_tril"].astype(dtype)
        L   = bijector.forward(raw).astype(dtype)         # (d,d), lower-tri
        Sigma = L @ L.T
        if cov_damping > 0.0:
            Sigma = Sigma + cov_damping * jnp.eye(Sigma.shape[0], dtype=dtype)

        # --- Mean block: nat-grad(m) = Σ * grad(m) ---
        g_m     = grads["mean"].astype(dtype)
        nat_g_m = Sigma @ g_m

        # --- Covariance block ---
        # Map grad wrt raw_tril  -> grad wrt L  using VJP of inverse (raw = inv(L))
        inv = lambda L_: bijector.inverse(L_)
        _, pullback_inv = jax.vjp(inv, L)
        g_raw = grads["raw_tril"].astype(dtype)
        (g_L,) = pullback_inv(g_raw)                      # (d,d)
        if project_lower:
            g_L = jnp.tril(g_L)

        # Convert g_L -> G_Sigma = 0.5 * g_L * L^{-1}   (optionally symmetrize)
        # Compute Y = g_L * L^{-1} via RIGHT triangular solve:  Y^T solves L^T Y^T = g_L^T
        Yt = solve_triangular(L.T, g_L.T, lower=False)    # Yt = L^{-T} g_L^T
        Y  = Yt.T                                         # Y = g_L L^{-1}
        G_Sigma = 0.5 * Y
        # Be safe numerically:
        G_Sigma = 0.5 * (G_Sigma + G_Sigma.T)

        # Natural direction in Σ: ΔΣ = 2 Σ GΣ Σ
        Delta_Sigma = 2.0 * (Sigma @ (G_Sigma @ Sigma))

        # Map ΔΣ -> δL with δL = 0.5 ΔΣ L^{-T}  (exactly yields dΣ = ΔΣ)
        tmp = solve_triangular(L, Delta_Sigma.T, lower=True)  # tmp = L^{-1} ΔΣ^T
        dL  = 0.5 * tmp.T                                     # δL = 0.5 ΔΣ L^{-T}
        if project_lower:
            dL = jnp.tril(dL)

        # Push δL -> δraw  using JVP of inverse bijector
        _, d_raw = jax.jvp(inv, (L,), (dL,))
        nat_g_raw = d_raw

        updates = {"mean": nat_g_m, "raw_tril": nat_g_raw}
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------- Pure helper functions (no class state) ----------

def make_spd(A, jitter_init: float = 1e-8, max_tries: int = 8) -> np.ndarray:
    """
    Ensure a symmetric positive-definite matrix close to A by adding
    diagonal jitter.
    """
    M = np.asarray(A, dtype=float)
    M = 0.5 * (M + M.T)  # symmetrize

    jitter = jitter_init
    for _ in range(max_tries + 1):
        try:
            np.linalg.cholesky(M)
            return M
        except np.linalg.LinAlgError:
            M = M + jitter * np.eye(M.shape[0])
            jitter *= 10.0

    # Last resort: shift by (-eigmin + jitter_init)
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    shift = max(0.0, -w[0] + jitter_init)
    M = M + shift * np.eye(M.shape[0])
    np.linalg.cholesky(M)  # let this raise if still not SPD
    return M


# end of file
