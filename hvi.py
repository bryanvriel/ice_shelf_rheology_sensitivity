from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np

import jax
from jax import numpy as jnp
from jax import random, value_and_grad
from jax.scipy.linalg import solve_triangular, cho_solve
from jax.scipy.special import expit
from jax.scipy.stats import norm as norm_dist
import optax

from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from .fields import make_spd

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("No mpi4py installed. No MPI functionality can be used.")

Array = jnp.ndarray
Params = Dict[str, Array]
# Structure:
# {
#   "mean": (N,),
#   "raw_tril": (N*(N+1)//2,),
#   "log_σ_B_shifted": scalar,  # log(σ_B - σ_B_min)
#   "log_σ_n_shifted": scalar,  # log(σ_n - σ_n_min)
#   "raw_ρ": scalar,              # arctanh(ρ)
# }


class SVIFullRankGaussianCorrelated:
    """
    Stateless SVI helper for a full-rank Gaussian variational family with
    hierarchical inference of prior variances and correlation for two distinct
    fields (B and n).

    This class infers the posterior of the latent variables q(theta) ~ N(m, L L^T)
    while simultaneously optimizing:
      - σ_B: prior std dev for B coefficients (with floor σ_B_min)
      - σ_n: prior std dev for n coefficients (with floor σ_n_min)
      - ρ: correlation between corresponding B and n coefficients

    The prior on (theta_B[k], theta_n[k]) for each wavelet coefficient k is:
        [theta_B[k], theta_n[k]]^T ~ N(0, Σ_block)
    where Σ_block = [[σ_B^2, ρ*σ_B*σ_n],
                         [ρ*σ_B*σ_n, σ_n^2]]

    Hyperpriors are designed to resist collapse (shrinking variances to zero).
    """

    def __init__(
        self,
        loglik_fn: Callable[[Array], Array],
        num_params_B: int,
        prior_mean: Optional[Array] = None,
        hyperprior_config: Optional[Dict] = None,
        *,
        num_mc: int = 32,
        antithetic: bool = True,
        dtype=jnp.float64,
        diag_shift: float = 1e-5,
        beta_kl: float = 1.0,
        comm=None,
    ):
        """
        Args:
            loglik_fn: Function mapping theta -> log likelihood (or (loglik, grad) tuple for MPI).
            num_params_B: Number of coefficients corresponding to field B.
                          Assumes theta = [theta_B, theta_n] with equal lengths.
            prior_mean: (Optional) Fixed prior mean. Defaults to zero vector.
            hyperprior_config: Dict containing hyperprior settings:
                {
                    'σ_B_min': float,     # Floor for σ_B (default 0.3)
                    'σ_n_min': float,     # Floor for σ_n (default 0.2)
                    'σ_B_default': float, # Default/expected σ_B (default 1.5)
                    'σ_n_default': float, # Default/expected σ_n (default 1.0)
                    'ρ_default': float,   # Default/expected correlation (default -0.3)
                    'λ_σ': float,         # Penalty strength for σ collapse (default 1.0)
                    'λ_ρ': float,         # Penalty strength for ρ deviation (default 0.5)
                }
        """
        self.loglik_fn = loglik_fn
        self.num_params_B = num_params_B

        # Parse hyperprior configuration
        cfg = hyperprior_config or {}
        self.σ_B_min = cfg.get("σ_B_min", 0.3)
        self.σ_n_min = cfg.get("σ_n_min", 0.2)
        self.σ_B_default = cfg.get("σ_B_default", 1.5)
        self.σ_n_default = cfg.get("σ_n_default", 1.0)
        self.ρ_default = cfg.get("ρ_default", -0.3)
        self.λ_σ = cfg.get("λ_σ", 1.0)
        self.λ_ρ = cfg.get("λ_ρ", 0.5)

        self.num_mc = int(num_mc)
        self.antithetic = bool(antithetic)
        self.dtype = dtype
        self.β_kl = beta_kl

        self._fixed_prior_mean = prior_mean

        self._bij = tfb.FillScaleTriL(
            diag_bijector=tfb.Softplus(),
            diag_shift=jnp.asarray(diag_shift, dtype=dtype),
        )

        # MPI logic
        if comm is not None:
            self.comm = comm
            self.mpi_rank = comm.rank
            self.num_mc_local = num_mc // comm.size
            if self.mpi_rank == (comm.size - 1):
                self.num_mc_local = num_mc - self.mpi_rank * self.num_mc_local
            self.scalar_counts = comm.allgather(self.num_mc_local)
        else:
            self.mpi_rank = 0
            self.comm = None

    def _init_mpi_buffers(self, n_dim):
        """Lazy initialization of MPI buffers once dimension is known."""
        if self.comm is None:
            return

        self.n_dim = n_dim
        self.vector_counts = self.comm.allgather(self.n_dim * self.num_mc_local)
        self.eps = np.zeros((self.num_mc_local, self.n_dim))

        if self.mpi_rank == 0:
            self.logP = np.zeros(self.num_mc)
            self.grad_logP = np.zeros(self.n_dim * self.num_mc)
        else:
            self.logP = self.grad_logP = None

    # ----- Private MPI helpers -----

    def _scatter_eps(self, array):
        self.comm.Scatterv(
            [array, self.vector_counts, MPI.DOUBLE],
            self.eps,
            root=0,
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

    # ----- Hyperparameter transformations -----

    def get_σ_B(self, params: Params) -> Array:
        """Transform unconstrained parameter to σ_B with floor."""
        return self.σ_B_min + jnp.exp(params["log_σ_B_shifted"])

    def get_σ_n(self, params: Params) -> Array:
        """Transform unconstrained parameter to σ_n with floor."""
        return self.σ_n_min + jnp.exp(params["log_σ_n_shifted"])

    def get_ρ(self, params: Params) -> Array:
        """Transform unconstrained parameter to correlation in (-1, 1)."""
        return jnp.tanh(params["raw_ρ"])

    def get_hyperparams(self, params: Params) -> Tuple[Array, Array, Array]:
        """Extract all three hyperparameters in constrained space."""
        return self.get_σ_B(params), self.get_σ_n(params), self.get_ρ(params)

    # ----- Public properties and helpers -----

    @property
    def bij(self):
        return self._bij

    def init_from_map_hessian(
        self,
        theta_map: Array,
        hessian_at_map: Array,
        init_σ_B: Optional[float] = None,
        init_σ_n: Optional[float] = None,
        init_ρ: float = 0.0,
    ) -> Params:
        """
        Build params dict from MAP and Hessian.
        Also initializes the learnable prior hyperparameters.

        Args:
            theta_map: MAP estimate of parameters.
            hessian_at_map: Hessian of negative log posterior at MAP.
            init_σ_B: Initial σ_B (defaults to σ_B_default).
            init_σ_n: Initial σ_n (defaults to σ_n_default).
            init_ρ: Initial correlation (default 0.0).
        """
        dtype = self.dtype
        theta_map = jnp.asarray(theta_map, dtype=dtype)
        H = jnp.asarray(make_spd(hessian_at_map), dtype=dtype)
        n = theta_map.shape[0]

        # Initialize MPI buffers
        if self.comm is not None:
            self._init_mpi_buffers(n)
        else:
            self.n_dim = n

        # Initialize variational covariance from Laplace approximation
        C = jnp.linalg.cholesky(H)
        Σ0 = cho_solve((C, True), jnp.eye(n, dtype=dtype))
        Σ0 = 0.5 * (Σ0 + Σ0.T)
        L0 = jnp.linalg.cholesky(Σ0)

        raw_tril0 = self._bij.inverse(L0).astype(dtype)

        # Initialize hyperparameters
        if init_σ_B is None:
            init_σ_B = self.σ_B_default
        if init_σ_n is None:
            init_σ_n = self.σ_n_default

        # Ensure initial values respect floors
        init_σ_B = max(init_σ_B, self.σ_B_min + 1e-6)
        init_σ_n = max(init_σ_n, self.σ_n_min + 1e-6)
        init_ρ = jnp.clip(init_ρ, -0.999, 0.999)

        # Transform to unconstrained space
        log_σ_B_shifted = jnp.log(init_σ_B - self.σ_B_min)
        log_σ_n_shifted = jnp.log(init_σ_n - self.σ_n_min)
        raw_ρ = jnp.arctanh(init_ρ)

        return {
            "mean": theta_map,
            "raw_tril": raw_tril0,
            "log_σ_B_shifted": jnp.asarray(log_σ_B_shifted, dtype=dtype),
            "log_σ_n_shifted": jnp.asarray(log_σ_n_shifted, dtype=dtype),
            "raw_ρ": jnp.asarray(raw_ρ, dtype=dtype),
        }

    # ---------- Public evaluation helpers ----------

    def scale_tril(self, params: Params) -> Array:
        return self._bij.forward(params["raw_tril"]).astype(self.dtype)

    def sample_reparam(self, params: Params, eps: Array) -> Array:
        m = params["mean"].astype(self.dtype)
        L = self.scale_tril(params)
        return m[None, :] + eps @ L.T

    def sample_eps(
        self, key: Array, n: int, antithetic: Optional[bool] = None
    ) -> Array:
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
        self, params: Params, key: Array, n: int, *args, antithetic=None
    ) -> Array:
        eps = self.sample_eps(key, n, antithetic)
        return self.sample_reparam(params, eps)

    def _hyperprior_log_prob(self, σ_B: Array, σ_n: Array, ρ: Array) -> Array:
        """
        Log probability of hyperparameters under collapse-resistant hyperpriors.

        The hyperprior penalizes σs being smaller than their defaults,
        but allows them to be larger without penalty. This implements a
        "soft floor" that resists the collapse behavior seen with marginal
        likelihood maximization.

        For ρ, we use a symmetric penalty centered at ρ_default.
        """
        # Asymmetric penalty on σ: penalize being smaller than default
        # log p(σ) ∝ -λ * max(0, log(σ_default) - log(σ))
        penalty_σ_B = -self.λ_σ * jnp.maximum(
            0.0, jnp.log(self.σ_B_default) - jnp.log(σ_B)
        )
        penalty_σ_n = -self.λ_σ * jnp.maximum(
            0.0, jnp.log(self.σ_n_default) - jnp.log(σ_n)
        )

        # Symmetric penalty on ρ around default
        # Using a normal prior in arctanh space
        arctanh_ρ = jnp.arctanh(ρ)
        arctanh_ρ_default = jnp.arctanh(self.ρ_default)
        penalty_ρ = -self.λ_ρ * (arctanh_ρ - arctanh_ρ_default) ** 2

        return penalty_σ_B + penalty_σ_n + penalty_ρ

    def _log_prior_wavelet_coeffs(
        self, theta: Array, σ_B: Array, σ_n: Array, ρ: Array
    ) -> Array:
        """
        Compute log p(theta | σ_B, σ_n, ρ) for the correlated prior.

        The prior has block structure: for each wavelet index k,
        [theta_B[k], theta_n[k]]^T ~ N(0, Σ_block)

        where Σ_block = [[σ_B^2,     ρ*σ_B*σ_n],
                         [ρ*σ_B*σ_n, σ_n^2]]

        This exploits the Kronecker structure for efficiency.
        """
        K = self.num_params_B
        n_dim = theta.shape[0]

        # Handle prior mean
        if self._fixed_prior_mean is not None:
            delta = theta - self._fixed_prior_mean
        else:
            delta = theta

        # Split into B and n components
        delta_B = delta[:K]
        delta_n = delta[K:]

        # Compute precision matrix elements for 2x2 block
        # det(Σ_block) = σ_B^2 * σ_n^2 * (1 - ρ^2)
        one_minus_ρ_sq = 1.0 - ρ**2
        det_block = σ_B**2 * σ_n**2 * one_minus_ρ_sq

        # Precision = Σ^{-1}
        # P = (1/det) * [[σ_n^2, -ρ*σ_B*σ_n],
        #                [-ρ*σ_B*σ_n, σ_B^2]]
        p_BB = σ_n**2 / det_block
        p_nn = σ_B**2 / det_block
        p_Bn = -ρ * σ_B * σ_n / det_block

        # Quadratic form: sum over K blocks
        # (delta_B[k], delta_n[k])^T @ P @ (delta_B[k], delta_n[k])
        quad_form = (
            p_BB * jnp.sum(delta_B**2)
            + p_nn * jnp.sum(delta_n**2)
            + 2.0 * p_Bn * jnp.sum(delta_B * delta_n)
        )

        # Log normalizing constant: K copies of 2D Gaussian
        log_norm = -K * jnp.log(2 * jnp.pi) - 0.5 * K * jnp.log(det_block)

        log_prior = log_norm - 0.5 * quad_form
        return log_prior

    def kl_fullrank_prior(self, params: Params) -> Array:
        """
        Computes KL(q(theta) || p(theta | hyperparams)) - log p(hyperparams).

        The prior has correlated block structure between B and n coefficients.
        """
        # 1. Extract variational parameters
        m = params["mean"]
        L = self.scale_tril(params)
        n_dim = m.shape[0]
        K = self.num_params_B

        # 2. Extract hyperparameters
        σ_B, σ_n, ρ = self.get_hyperparams(params)

        # 3. Compute KL divergence components
        # For KL(N(m, Σ) || N(0, Σ_prior)):
        # KL = 0.5 * [
        #     tr(Σ_prior^{-1} Σ) +
        #     m^T Σ_prior^{-1} m - n +
        #     log|Σ_prior| - log|Σ|
        # ]

        # Handle prior mean
        if self._fixed_prior_mean is not None:
            delta = m - self._fixed_prior_mean
        else:
            delta = m

        delta_B = delta[:K]
        delta_n = delta[K:]

        # Prior precision elements (same as in _log_prior_wavelet_coeffs)
        one_minus_ρ_sq = 1.0 - ρ**2
        det_block = σ_B**2 * σ_n**2 * one_minus_ρ_sq

        p_BB = σ_n**2 / det_block
        p_nn = σ_B**2 / det_block
        p_Bn = -ρ * σ_B * σ_n / det_block

        # --- Term 1: tr(Σ_prior^{-1} @ Σ_q) ---
        # Σ_q = L @ L^T
        # We need tr(P @ Σ_q) where P is block-diagonal with K copies of 2x2 precision
        #
        # Split L into blocks: L = [[L_BB, 0], [L_nB, L_nn]] (roughly)
        # Actually, L is full, so we compute:
        # Σ_q[i,j] = sum_k L[i,k] L[j,k]
        #
        # tr(P @ Σ_q) = sum over all (i,j) of P[i,j] * Σ_q[i,j]
        #
        # For our block structure:
        # tr = p_BB * sum_i Σ_q[i,i] for i in B
        #    + p_nn * sum_i Σ_q[i,i] for i in n
        #    + 2*p_Bn * sum_k Σ_q[B_k, n_k] for corresponding pairs

        # Diagonal of Σ_q
        diag_Σ = jnp.sum(L**2, axis=1)
        diag_Σ_B = diag_Σ[:K]
        diag_Σ_n = diag_Σ[K:]

        # Off-diagonal: Σ_q[k, K+k] = sum_j L[k,j] * L[K+k, j]
        # These are the covariances between corresponding B and n coefficients
        L_B = L[:K, :]  # (K, n_dim)
        L_n = L[K:, :]  # (K, n_dim)
        cov_Bn = jnp.sum(L_B * L_n, axis=1)  # (K,) - elementwise row dot products

        tr_term = (
            p_BB * jnp.sum(diag_Σ_B)
            + p_nn * jnp.sum(diag_Σ_n)
            + 2.0 * p_Bn * jnp.sum(cov_Bn)
        )

        # --- Term 2: m^T @ Σ_prior^{-1} @ m ---
        quad_term = (
            p_BB * jnp.sum(delta_B**2)
            + p_nn * jnp.sum(delta_n**2)
            + 2.0 * p_Bn * jnp.sum(delta_B * delta_n)
        )

        # --- Term 3: log|Σ_prior| ---
        # log|Σ_prior| = K * log(det_block)
        log_det_prior = K * jnp.log(det_block)

        # --- Term 4: log|Σ_q| ---
        log_det_q = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        # KL divergence
        kl_theta = 0.5 * (tr_term + quad_term - n_dim + log_det_prior - log_det_q)

        # 4. Hyperprior contribution (negative log prior on hyperparameters)
        log_hyperprior = self._hyperprior_log_prob(σ_B, σ_n, ρ)
        hyperprior_loss = -log_hyperprior

        return self.β_kl * kl_theta + hyperprior_loss

    def elbo_loss_vmap(
        self, params: Params, rng_key: Array
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Compute negative ELBO (Loss) including hyperprior regularization.
        """
        M = int(self.num_mc)
        thetas = self.sample(params, rng_key, M)

        # Log Likelihood: E_q [ log p(d|theta) ]
        ll_vals = jax.vmap(self.loglik_fn)(thetas)
        logP = jnp.mean(ll_vals)

        # KL Divergence + Hyperprior Loss
        kl_total = self.kl_fullrank_prior(params)

        elbo = logP - kl_total
        neg_elbo = -elbo

        σ_B, σ_n, ρ = self.get_hyperparams(params)
        metrics = dict(
            elbo=elbo,
            logP=logP,
            kl=kl_total,
            σ_B=σ_B,
            σ_n=σ_n,
            ρ=ρ,
        )
        return neg_elbo, metrics

    def elbo_loss_mpi(
        self, params: Params, rng_key: Array
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Compute negative ELBO with MPI parallelism for Firedrake.
        """
        # Unpack hyperparameters from current parameters
        σ_B, σ_n, ρ = self.get_hyperparams(params)

        # 1. Sample eps on Root and scatter
        if self.mpi_rank == 0:
            eps = self.sample_eps(rng_key, self.num_mc)
        else:
            eps = None
        self._scatter_eps(eps)

        # 2. Transform to theta (local)
        thetas = np.asarray(self.sample_reparam(params, self.eps))

        # 3. Compute Forward Model / Likelihood (local)
        logP_local = np.zeros(self.num_mc_local)
        grad_logP_local = np.zeros((self.num_mc_local, self.n_dim))

        for i in range(self.num_mc_local):
            logP_local[i], grad_logP_local[i, :] = self.loglik_fn(thetas[i])

        # 4. Gather Likelihoods and Gradients to Root
        self._gather_logP(logP_local)
        self._gather_grad_logP(grad_logP_local)

        if self.mpi_rank != 0:
            return None, None, None

        # --- ROOT ONLY COMPUTATION BELOW ---

        valid_mask = np.isfinite(self.logP)
        num_valid_mc = np.sum(valid_mask)
        if num_valid_mc == 0:
            metrics = dict(
                elbo=np.nan, logP=np.nan, kl=np.nan, σ_B=σ_B, σ_n=σ_n, ρ=ρ,
            )
            return 1e12, jax.tree_util.tree_map(jnp.zeros_like, params), metrics

        logP = jnp.mean(self.logP[valid_mask])
        grad_logP_vals = self.grad_logP.reshape(-1, self.n_dim)[valid_mask]
        eps_valid = eps[valid_mask]

        # 5. vjp to get gradient of logP w.r.t. variational parameters
        _, vjpfun = jax.vjp(
            lambda p: self.sample_reparam(p, eps_valid),
            params,
        )
        grad_logP_params = vjpfun(grad_logP_vals)[0]

        # Scale by 1/M (mean)
        grad_logP_params = jax.tree_util.tree_map(
            lambda x: x / num_valid_mc,
            grad_logP_params,
        )

        # 6. KL Gradient (Analytic)
        kl_val, grad_kl_params = jax.value_and_grad(self.kl_fullrank_prior)(params)

        # 7. Combine
        loss = -1 * (logP - kl_val)
        grad_loss = jax.tree_util.tree_map(
            lambda g_kl, g_lik: g_kl - g_lik, grad_kl_params, grad_logP_params
        )
        metrics = dict(
            elbo=-loss, logP=logP, kl=kl_val, σ_B=σ_B, σ_n=σ_n, ρ=ρ,
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

        if jit:
            v_g = value_and_grad(self.elbo_loss_vmap, has_aux=True)

            @jax.jit
            def step(pr, ost, r):
                r, sub = random.split(r)
                (loss, metrics), grads = v_g(pr, sub)
                updates, ost = tx.update(grads, ost, pr)
                pr = optax.apply_updates(pr, updates)
                return pr, ost, r, loss, metrics

        else:
            def step(pr, ost, r):
                rng, sub = random.split(r)
                loss, grads, metrics = self.elbo_loss_mpi(pr, sub)
                if self.mpi_rank == 0:
                    updates, ost = tx.update(grads, ost, pr)
                    pr = optax.apply_updates(pr, updates)
                ost = self.comm.bcast(ost, root=0) # not necessary, but do it anyway
                pr = self.comm.bcast(pr, root=0)
                return pr, ost, rng, loss, metrics

        # History now includes 5 columns: logP, kl, σ_B, σ_n, ρ
        hist = np.zeros((num_steps, 5))

        for t in range(num_steps):
            params, opt_state, rng, loss, metrics = step(params, opt_state, rng)

            if self.mpi_rank == 0:
                if callback or (t == num_steps - 1):
                    σ_B, σ_n, ρ = self.get_hyperparams(params)
                    hist[t] = [
                        metrics["logP"],
                        metrics["kl"],
                        float(σ_B),
                        float(σ_n),
                        float(ρ),
                    ]

                if callback is not None:
                    callback(t, hist[: t + 1], params, opt_state, rng)

        return params, opt_state, rng, hist

    # ---------- Posterior Analysis Helpers ----------
    def posterior_covariance(self, params: Params) -> Array:
        """Return the full posterior covariance matrix."""
        L = self.scale_tril(params)
        return L @ L.T

    def posterior_correlation_matrix(self, params: Params) -> Array:
        """Return the posterior correlation matrix."""
        Σ = self.posterior_covariance(params)
        std = jnp.sqrt(jnp.diag(Σ))
        return Σ / jnp.outer(std, std)

    def marginal_stds(self, params: Params) -> Tuple[Array, Array]:
        """Return marginal posterior standard deviations for B and n."""
        Σ = self.posterior_covariance(params)
        stds = jnp.sqrt(jnp.diag(Σ))
        return stds[: self.num_params_B], stds[self.num_params_B :]

    def summary(self, params: Params) -> Dict:
        """Return a summary of the current variational distribution."""
        σ_B, σ_n, ρ = self.get_hyperparams(params)
        std_B, std_n = self.marginal_stds(params)

        return {
            "prior_σ_B": float(σ_B),
            "prior_σ_n": float(σ_n),
            "prior_ρ": float(ρ),
            "posterior_mean_B": np.array(params["mean"][: self.num_params_B]),
            "posterior_mean_n": np.array(params["mean"][self.num_params_B :]),
            "posterior_std_B": np.array(std_B),
            "posterior_std_n": np.array(std_n),
        }


def natural_gradient_transform(
    bijector: tfb.Bijector,
    damping: float = 1e-5,
) -> optax.GradientTransformation:
    """
    AD-based Natural Gradient for Block-Diagonal Covariance + Scalar Hyperpriors.

    Applies Fisher Information Matrix scaling to the Gaussian parameters
    ('mean', 'raw_tril') and Identity scaling to the scalar hyperparameters.

    Updated for the correlated prior with three hyperparameters:
      - log_σ_B_shifted: log(σ_B - σ_B_min)
      - log_σ_n_shifted: log(σ_n - σ_n_min)
      - raw_ρ: arctanh(ρ)
    """

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(grads, state, params):
        # --- 1. Natural Gradient for Gaussian Variational Parameters ---

        # Setup
        m = params["mean"]
        raw = params["raw_tril"]

        # Forward pass through bijector
        L = bijector.forward(raw)

        # A. Mean Update: nat_grad_mu = (Σ + damping*I) @ grad_mu
        # Efficient computation: L @ (L.T @ g_m)
        g_m = grads["mean"]
        Lt_gm = L.T @ g_m
        nat_g_m = L @ Lt_gm + damping * g_m

        # B. Covariance Update
        # Pullback: Map gradient from raw-space to L-space
        inv = lambda L_: bijector.inverse(L_)
        _, pullback_inv = jax.vjp(inv, L)
        g_raw = grads["raw_tril"]
        (g_L,) = pullback_inv(g_raw)

        # Project gradients to lower triangle
        g_L = jnp.tril(g_L)

        # Convert g_L -> Natural Gradient in Σ space
        # Y^T = inv(L)^T @ g_L^T = solve(L.T, g_L.T)
        Yt = solve_triangular(L.T, g_L.T, lower=False)
        Y = Yt.T

        # Symmetrize the gradient in Σ space
        G_Σ = 0.5 * (Y + Y.T)

        # Natural Gradient Step in Σ space:
        # Delta_Σ = 2 * Σ @ G_Σ @ Σ
        # Delta_L = L @ Z where Z = L.T @ G_Σ @ L
        Z = L.T @ (G_Σ @ L)
        nat_g_L = L @ Z

        # Enforce lower triangularity
        nat_g_L = jnp.tril(nat_g_L)

        # Pushforward: Map nat_g_L -> nat_g_raw
        _, d_raw = jax.jvp(inv, (L,), (nat_g_L,))
        nat_g_raw = d_raw

        # --- 2. Identity Update for Hyperparameters ---

        # Pass gradients through for all three hyperparameters.
        # The downstream optimizer (e.g., Adam) will handle step sizes.
        # Using .get() allows flexibility if parameters are frozen.

        nat_g_log_σ_B = grads.get("log_σ_B_shifted", jnp.array(0.0))
        nat_g_log_σ_n = grads.get("log_σ_n_shifted", jnp.array(0.0))
        nat_g_raw_ρ = grads.get("raw_ρ", jnp.array(0.0))

        # --- 3. Construct Updates Dictionary ---
        updates = {
            "mean": nat_g_m,
            "raw_tril": nat_g_raw,
            "log_σ_B_shifted": nat_g_log_σ_B,
            "log_σ_n_shifted": nat_g_log_σ_n,
            "raw_ρ": nat_g_raw_ρ,
        }

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def estimate_prior_hyperparams_regularized(
    theta_map: np.ndarray,
    hessian_total: np.ndarray,
    num_params_B: int,
    σ_B_min: float = 0.3,
    σ_n_min: float = 0.2,
    σ_B_default: float = 1.5,
    σ_n_default: float = 1.0,
    ρ_default: float = -0.3,
    ρ_bounds: Tuple[float, float] = (-0.95, 0.95),
    regularization_strength: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Performs one MacKay-like fixed-point update step to estimate optimal prior
    hyperparameters (σ_B, σ_n, ρ) for a correlated block prior.

    This function implements evidence maximization with regularization toward
    default values to prevent collapse (shrinking variances when data are
    uninformative).

    The update formula is:
        Σ_0_new = (1 / K_eff) * [sum_k (theta_k @ theta_k^T + Σ_post_k) + λ * Σ_default]

    where:
        - theta_k = [theta_B[k], theta_n[k]]^T
        - Σ_post_k is the 2x2 posterior covariance block for pair k
        - λ is the regularization strength (pseudo-count)
        - K_eff = K + λ

    Args:
        theta_map: The MAP solution vector, shape (2*K,), concatenated as [theta_B, theta_n].
        hessian_total: The FULL Hessian of the negative log posterior at MAP,
                       shape (2*K, 2*K). This includes both likelihood and prior contributions.
        num_params_B: Number of coefficients belonging to B (= K).
                      Assumes theta_n also has K coefficients.
        σ_B_min: Hard floor for σ_B to prevent collapse.
        σ_n_min: Hard floor for σ_n to prevent collapse.
        σ_B_default: Default/prior mean for σ_B used in regularization.
        σ_n_default: Default/prior mean for σ_n used in regularization.
        ρ_default: Default/prior mean for ρ used in regularization.
        ρ_bounds: (min, max) bounds for correlation coefficient.
        regularization_strength: Effective number of pseudo-observations
                                 pulling toward defaults. Higher values provide
                                 more resistance to collapse.

    Returns:
        Tuple of (new_σ_B, new_σ_n, new_ρ).
    """
    K = num_params_B
    N = theta_map.shape[0]

    if N != 2 * K:
        raise ValueError(f"Expected theta_map of length 2*K={2*K}, got {N}")

    if hessian_total.shape != (N, N):
        raise ValueError(
            f"Expected hessian_total of shape ({N}, {N}), got {hessian_total.shape}"
        )

    # 1. Compute posterior covariance: Σ_post = H_total^{-1}
    H_spd = make_spd(hessian_total)
    L = np.linalg.cholesky(H_spd)
    Σ_post = cho_solve((L, True), np.eye(N))

    # 2. Extract block statistics from posterior covariance
    # For the correlated prior, we need:
    #   - Posterior variances for B coefficients: diag(Σ_post)[:K]
    #   - Posterior variances for n coefficients: diag(Σ_post)[K:]
    #   - Posterior covariances between corresponding B[k] and n[k]: Σ_post[k, K+k]

    diag_Σ = np.diag(Σ_post)

    post_var_B = diag_Σ[:K]  # shape (K,)
    post_var_n = diag_Σ[K:]  # shape (K,)

    # Extract covariances between theta_B[k] and theta_n[k]
    # These are the (k, K+k) elements for k = 0, ..., K-1
    post_cov_Bn = np.array([Σ_post[k, K + k] for k in range(K)])  # shape (K,)

    # Sum over all K coefficient pairs
    tr_post_BB = np.sum(post_var_B)
    tr_post_nn = np.sum(post_var_n)
    tr_post_Bn = np.sum(post_cov_Bn)

    # 3. Compute empirical second moments from MAP estimate
    theta_B = theta_map[:K]
    theta_n = theta_map[K:]

    S_BB = np.sum(theta_B**2)
    S_nn = np.sum(theta_n**2)
    S_Bn = np.sum(theta_B * theta_n)

    # 4. Total sufficient statistics: empirical + posterior uncertainty
    # This is sum_k [theta_k @ theta_k^T + Σ_post_k]
    total_BB = S_BB + tr_post_BB
    total_nn = S_nn + tr_post_nn
    total_Bn = S_Bn + tr_post_Bn

    # 5. Add regularization pseudo-observations
    # These pull the estimate toward the default prior covariance:
    #   Σ_default = [
    #       [σ_B_default^2, ρ_default*σ_B_default*σ_n_default],
    #       [ρ_default*σ_B_default*σ_n_default, σ_n_default^2],
    #   ]

    λ_reg = regularization_strength

    reg_BB = λ_reg * σ_B_default**2
    reg_nn = λ_reg * σ_n_default**2
    reg_Bn = λ_reg * ρ_default * σ_B_default * σ_n_default

    # Combined statistics with regularization
    total_BB_reg = total_BB + reg_BB
    total_nn_reg = total_nn + reg_nn
    total_Bn_reg = total_Bn + reg_Bn

    # Effective sample size (data blocks + pseudo-observations)
    K_eff = K + λ_reg

    # 6. Compute new prior covariance estimates
    new_var_B = total_BB_reg / K_eff
    new_var_n = total_nn_reg / K_eff
    new_cov_Bn = total_Bn_reg / K_eff

    # 7. Convert to (σ_B, σ_n, ρ) with constraints

    # Ensure variances are above floors
    new_σ_B = np.sqrt(np.maximum(new_var_B, σ_B_min**2))
    new_σ_n = np.sqrt(np.maximum(new_var_n, σ_n_min**2))

    # Compute correlation: ρ = cov / (σ_B * σ_n)
    new_ρ = new_cov_Bn / (new_σ_B * new_σ_n)

    # Apply all constraints
    new_σ_B = float(max(new_σ_B, σ_B_min))
    new_σ_n = float(max(new_σ_n, σ_n_min))
    new_ρ = float(np.clip(new_ρ, ρ_bounds[0], ρ_bounds[1]))

    return new_σ_B, new_σ_n, new_ρ


# end of file
