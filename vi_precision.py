# svi_precision_lower.py
# Full-rank Gaussian SVI (JAX + Optax) with precision parameterization using LOWER Cholesky.
# q(theta) = N(mu, (L L^T)^{-1}), with L lower-triangular.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import cloudpickle as pickle
import functools
import sys

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl
from jax import jit, value_and_grad
import optax

# TFP (JAX substrate)
import tensorflow_probability.substrates.jax as tfp
tfb = tfp.bijectors
tfm = tfp.math

# Global level bijector for constructing Cholesky matrix
bij_S = tfb.FillScaleTriL(
    diag_bijector=tfb.Softplus(),
    diag_shift=jnp.array(1e-8, dtype=jnp.float64),
)

from mpi4py import MPI


# ------------------------------- Utilities -------------------------------- #


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


def chol_precision_from_hessian_lower(H_map: jnp.ndarray) -> jnp.ndarray:
    """
    Robust Cholesky of the (slightly) indefinite Hessian using NumPy; returns LOWER L.
    Treats H_map as precision Λ; returns L (lower) s.t. Λ = L L^T.
    """
    H_np = make_spd(H_map)
    L_np = np.linalg.cholesky(H_np)  # lower
    return jnp.asarray(L_np)


# ------------------------------- Prior ------------------------------------ #


@jax.tree_util.register_dataclass
@dataclass
class PriorGaussian:
    m0: jax.Array  # (d,)
    Q0: jax.Array  # (d,d) precision
    L0: jax.Array  # (d,d) lower-tri chol of Q0
    logdet_Q0: float  # scalar

    @classmethod
    def from_precision(cls, m0: jnp.ndarray, Q0: jnp.ndarray) -> "PriorGaussian":
        Q0 = jnp.asarray(make_spd(Q0))
        L0 = jnp.linalg.cholesky(Q0)  # lower
        logdet_Q0 = 2.0 * float(np.sum(np.log(np.diag(L0))))
        # convert to JAX arrays
        return cls(
            m0=jnp.asarray(m0),
            Q0=jnp.asarray(Q0),
            L0=jnp.asarray(L0),
            logdet_Q0=logdet_Q0,
        )

    @classmethod
    def from_field(cls, m0: jnp.ndarray, field) -> "PriorGaussian":
        Q0 = field.prior_precision()
        L0 = np.linalg.cholesky(Q0)  # lower
        logdet_Q0 = 2.0 * float(np.sum(np.log(np.diag(L0))))
        # convert to JAX arrays
        return cls(
            m0=jnp.asarray(m0),
            Q0=jnp.asarray(Q0),
            L0=jnp.asarray(L0),
            logdet_Q0=logdet_Q0,
        )


# ------------------ Variational params via TFP bijector -------------------- #
# Params are:
#   params = {"mu": (d,), "z_tril": (d*(d+1)//2,)}
# L/S are built via FillScaleTriL.forward(z_tril)


def build_L_from_params(
    params: Dict[str, jnp.ndarray], diag_shift: float = 1e-8
) -> jnp.ndarray:
    """Lower-tri L from unconstrained vector using TFP bijector."""
    z = params["z_tril"]
    return tril_bij.forward(z)


def build_S_from_params(params):
    return tril_bij.forward(params["z_tril"])  # S lower


def pack_from_mu_L(
    mu: jnp.ndarray, L: jnp.ndarray, diag_shift: float = 1e-8
) -> Dict[str, jnp.ndarray]:
    """Inverse map: lower-tri L -> unconstrained vector z_tril."""
    z = tril_bij.inverse(L)
    return {"mu": mu, "z_tril": z}


def init_params_from_map(
    theta_map: jnp.ndarray, H_map: jnp.ndarray, diag_shift: float = 1e-8
) -> Dict[str, jnp.ndarray]:
    L0 = chol_precision_from_hessian_lower(H_map)  # lower
    return pack_from_mu_L(theta_map, L0, diag_shift=diag_shift)


# ----------------------- KL terms and sampling (lower) --------------------- #


def logdet_from_lower_chol(L: jnp.ndarray) -> float:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def trace_Q0_times_Lambda_inv_from_L(Q0: jnp.ndarray, L: jnp.ndarray) -> float:
    """
    Compute tr(Q0 * Λ^{-1}) with Λ = L L^T using two triangular solves:
      Solve L   Z = Q0  (lower)
      Solve L^T X = Z   (upper)
      Then tr(Λ^{-1} Q0) = tr(X) = tr(Q0 Λ^{-1})
    """
    Z = jsl.solve_triangular(L, Q0, lower=True)
    X = jsl.solve_triangular(L.T, Z, lower=False)
    return jnp.trace(X)


def kl_q_prior_lower(params: Dict[str, jnp.ndarray], prior: PriorGaussian) -> float:
    """
    KL[q||p] where q=N(mu, Λ^{-1}), Λ = L L^T (L lower), p=N(m0, Q0^{-1})
      KL = 0.5 * [ tr(Q0 Λ^{-1}) + (mu-m0)^T Q0 (mu-m0) - d + log|Q0| - log|Λ| ].
    """
    mu = params["mu"]
    L = build_L_from_params(params)
    d = mu.shape[0]
    delta = mu - prior.m0
    quad = delta @ (prior.Q0 @ delta)
    tr_term = trace_Q0_times_Lambda_inv_from_L(prior.Q0, L)
    logdet_Lambda = logdet_from_lower_chol(L)
    return 0.5 * (tr_term + quad - d - prior.logdet_Q0 + logdet_Lambda)


kl_q_prior_v_g = jax.jit(jax.value_and_grad(kl_q_prior_lower))


def sample_theta_from_q(
    params: Dict[str, jnp.ndarray],
    key: jax.Array,
    n_samples: int,
    antithetic: bool = True,
) -> jnp.ndarray:
    """
    Sample theta ~ q via reparameterization with LOWER L:
      theta = mu + (L^T)^{-1} eps,   eps ~ N(0, I).
    Returns (n_samples, d).
    """
    mu = params["mu"]
    L = build_L_from_params(params)
    d = mu.shape[0]

    if antithetic:
        half = (n_samples + 1) // 2
        eps = jr.normal(key, shape=(half, d))
        eps = jnp.concatenate([eps, -eps], axis=0)[:n_samples]
    else:
        eps = jr.normal(key, shape=(n_samples, d))

    # Solve L^T x = eps row-wise => x = (L^T)^{-1} eps
    def solve_one(e):
        return jsl.solve_triangular(L.T, e, lower=False)

    x = jax.vmap(solve_one)(eps)  # (n_samples, d)
    return mu + x


# ----------------------- ELBO and Optimization Step ------------------------ #


@dataclass
class VIConfig:
    n_samples: int = 16
    lr: float = 1e-3
    num_steps: int = 2000
    antithetic: bool = True
    clip_grad_norm: float | None = None  # set to e.g. 5.0 to enable


class PrecisionGaussian:
    """
    Stochastic VI for a precision-parameterized Gaussian q using LOWER Cholesky.

    Args:
      loglik_fn: function(theta) -> scalar log p(y | theta). Must be JAX-differentiable.
      prior: PriorGaussian (m0, Q0).
      config: VIConfig.
    """

    def __init__(
        self,
        loglik_fn: Callable[[jnp.ndarray], float],
        prior: PriorGaussian,
        config: VIConfig = VIConfig(),
        comm=None,
    ):
        self.loglik_fn = loglik_fn
        self.prior = prior
        self.config = config
        self.n_dim = self.prior.m0.size

        # MPI logic
        if comm is not None:
            n_mc_samples = config.n_samples
            self.comm = comm
            self.mpi_rank = comm.rank
            self._n_mc_samples = n_mc_samples // comm.size
            self.num_mc_total = n_mc_samples
            if self.mpi_rank == (comm.size - 1):
                self._n_mc_samples = n_mc_samples - self.mpi_rank * self._n_mc_samples

            # Allocate arrays for scattering and gathering
            self.scalar_counts = comm.allgather(self._n_mc_samples)
            self.vector_counts = comm.allgather(self.n_dim * self._n_mc_samples)
            self.thetas = np.zeros((self._n_mc_samples, self.n_dim))
            if self.mpi_rank == 0:
                self.logP = np.zeros(n_mc_samples)
                self.grad_logP = np.zeros(self.n_dim * n_mc_samples)
            else:
                self.logP = self.grad_logP = None
    
        else:
            self.mpi_rank = 0

    def _scatter_theta(self, array):
        self.comm.Scatterv(
            [array, self.vector_counts, MPI.DOUBLE],
            self.thetas,
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

    def elbo_loss_vmap(
        self,
        params: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
        beta: float = 1.0,
    ):
        # Cache some parameters
        n_samples = self.config.n_samples
        n_dim = self.n_dim
        antithetic = self.config.antithetic

        # Sample latent vector from variational distribution
        key, sub_key = jr.split(rng_key)
        thetas = sample_theta_from_q(params, sub_key, n_samples, antithetic)

        # Log likelihood, averaged over samples
        # Apply annealing scale factor beta
        logP = jax.vmap(self.loglik_fn)(thetas)
        logP = beta * jnp.mean(logP)

        # KL-divergence for prior (deterministic)
        kl = kl_q_prior_lower(params, self.prior)

        # Negative ELBO
        loss = -1 * (logP - kl)
        return loss, (logP, kl)

    def elbo_loss_and_terms(
        self,
        params: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
        beta: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Monte-Carlo ELBO = E_q[loglik] - KL(q||p)."""
        # Cache some parameters
        n_samples = self.config.n_samples
        n_dim = self.n_dim
        antithetic = self.config.antithetic

        # Split the RNG key to all the workers
        keys = jr.split(rng_key, self.comm.size)

        # Sample latent vector from variational distribution
        key, sub_key = jr.split(keys[self.mpi_rank])
        thetas = sample_theta_from_q(params, sub_key, n_samples, antithetic)

        # Log likelihood
        logP_local = np.zeros(self._n_mc_samples)
        grad_logP_local = np.zeros((self._n_mc_samples, self.n_dim))
        for i in range(self._n_mc_samples):
            logP_local[i], grad_logP_local[i, :] = self.loglik_fn(thetas[i])

        # Gather the results
        self._gather_logP(logP_local)
        self._gather_grad_logP(grad_logP_local)

        # Only root proceeds with total loss and gradient computation
        if self.mpi_rank != 0:
            return None, None, None

        # Average the log probability over the samples
        # Apply annealing scale factor beta
        logP = beta * np.mean(self.logP)

        # vjp to get gradient of logP w.r.t. variational parameters
        _, vjpfun = jax.vjp(
            sample_theta_from_q,
            params,
            sub_key,
            n_samples,
            antithetic,
        )
        grad_logP = vjpfun(self.grad_logP.reshape(-1, n_dim))[0]

        # KL-divergence for prior (deterministic)
        kl, grad_kl = kl_q_prior_v_g(params, self.prior)

        # Negative ELBO and grad
        loss = -1 * (logP - kl)
        grad_loss = jax.tree_util.tree_map(
            lambda x, y: -1 * (beta * x / self.num_mc_total - y),
            grad_logP,
            grad_kl,
        )

        return loss, grad_loss, (logP, kl)

    def init_from_map_hessian(
        self, theta_map: jnp.ndarray, H_map: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Initialization from Laplace:
          mu_0 = theta_map
          Λ_0  = H_map (joint Hessian of negative log posterior at MAP)
          L_0  = chol_lower(Λ_0)   (NumPy)
        """
        L0 = chol_precision_from_hessian_lower(H_map)  # lower
        return pack_from_mu_L(theta_map, L0)

    def fit(
        self,
        init_params: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
        tx: optax.GradientTransformation,
        opt_state: optax.OptState,
        *,
        jit: bool = False,
        num_steps: int | None = None,
        start_step: int = 0,
        checkfile: str | None = None,
    ) -> None:
        """
        Run SVI using an externally-supplied Optax optimizer and state.

        Args:
          init_params: current variational parameters.
          rng_key: PRNGKey for sampling.
          tx: optax.GradientTransformation.
          opt_state: state for `tx`.
          num_steps: overrides self.config.num_steps if provided.
          print_every: overrides self.config.print_every if provided.
          start_step: logical starting step index (useful when resuming from a checkpoint).
          on_step: optional callback(step, params, opt_state).

        Returns:
          params: updated variational params
          opt_state: updated optimizer state
          summary: dict with final ELBO/LL/KL at end
          rng_key: updated PRNGKey (for clean continuation)
        """
        steps = self.config.num_steps if num_steps is None else int(num_steps)

        if jit:
            @jax.jit
            def step(params, rng_key, opt_state):
                v_g_fn = jax.value_and_grad(self.elbo_loss_vmap, has_aux=True)
                (loss, aux), grad_loss = v_g_fn(params, rng_key)
                rng_key, _ = jax.random.split(rng_key)
                updates, opt_state = tx.update(grad_loss, opt_state, params)
                params = optax.apply_updates(params, updates)
            
                return params, rng_key, opt_state, loss, aux

        else:
            def step(params, rng_key, opt_state):
                # Every worker participates in Monte Carlo estimates for logP
                # Only root returns non-None values
                loss, grad_loss, aux = self.elbo_loss_and_terms(params, rng_key)
                rng_key, _ = jax.random.split(rng_key)

                # Root will perform updates
                if self.mpi_rank == 0:
                    updates, opt_state = tx.update(grad_loss, opt_state, params)
                    params = optax.apply_updates(params, updates)
                # Broadcast updated parameters
                params = self.comm.bcast(params, root=0)

                return params, rng_key, opt_state, loss, aux

        params = init_params
        losses = np.zeros(steps)
        logP = np.zeros(steps)
        kl = np.zeros(steps)
        for t in range(start_step, start_step + steps):

            # Perform update
            params, rng_key, opt_state, loss, aux = step(params, rng_key, opt_state)

            # Store losses
            losses[t] = loss
            logP[t] = aux[0]
            kl[t] = aux[1]

            # Checkpointing
            if checkfile is not None and self.mpi_rank == 0:
                print(f"Step {t:5d}/{steps} | ELBO: {-loss.item():.2f}")
                sys.stdout.flush()

                save_params(
                    checkfile,
                    params,
                    extra={
                        "losses": losses[: t + 1],
                        "logP": logP[: t + 1],
                        "kl": kl[: t + 1],
                        "opt_state": opt_state,
                        "rng_key": rng_key,
                    },
                )

        return params, rng_key, opt_state, (losses, logP, kl)


def save_params(outfile, params, extra={}):
    with open(outfile, "wb") as fid:
        pickle.dump(params, fid)
        pickle.dump(extra, fid)


def load_params(outfile):
    with open(outfile, "rb") as fid:
        params = pickle.load(fid)
        extra = pickle.load(fid)
    return params, extra


# -----------------
# Natural Gradient Descent Optimizer


def _sigmoid(x):
    # stable sigmoid; fine for float64
    return 1.0 / (1.0 + jnp.exp(-x))


def natgrad_gaussian_precision(
    diag_shift: float = 1e-8,
) -> optax.GradientTransformation:
    """
    Optax-compatible natural-gradient preconditioner for q(theta) = N(mu, (L L^T)^{-1}),
    with params = {"mu": (d,), "z_tril": (d(d+1)//2,)},
    where L = FillScaleTriL.forward(z_tril).

    Returns a GradientTransformation you can chain before an optimizer, e.g.:
        tx = optax.chain(
            natgrad_gaussian_precision_lower_tfp(diag_shift=1e-8),
            optax.scale(-lr),  # pure natural gradient descent
            # or optax.adam(lr) if you want Adam on nat-grad directions
        )
    """
    bij = tfb.FillScaleTriL(
        diag_bijector=tfb.Softplus(),
        diag_shift=jnp.array(diag_shift, dtype=jnp.float64),
    )

    def init_fn(params):
        # stateless transform
        return ()

    def update_fn(grads, state, params):
        # --- unpack ---
        g_mu = grads["mu"]  # (d,)
        z = params["z_tril"]  # (n = d(d+1)/2,)
        g_z = grads["z_tril"]  # same shape as z

        # Build L and precision Λ = L L^T
        L = bij.forward(z)  # (d,d) lower-tri with positive diag
        d = L.shape[0]
        Lam = L @ L.T

        # ---- Natural grad for mu: Λ^{-1} g_mu via two triangular solves ----
        y = jsl.solve_triangular(L, g_mu, lower=True)  # solve L y = g_mu
        g_mu_nat = jsl.solve_triangular(L.T, y, lower=False)  # solve L^T x = y

        # ---- Recover gradient wrt L from gradient wrt z (elementwise chain rule) ----
        # Map g_z (vector) -> lower-tri matrix in row-major lower order
        gL_base = tfm.fill_triangular(g_z)  # lower-tri; off-diag already correct
        z_mat = tfm.fill_triangular(z)  # pre-activations on diag
        sprime = _sigmoid(jnp.diag(z_mat))  # d softplus / dz on diag

        # Adjust diagonals: g_u = g_L_ii * s'(u)  =>  g_L_ii = g_u / s'(u)
        gL = jnp.tril(gL_base)
        gL = gL.at[jnp.arange(d), jnp.arange(d)].set(jnp.diag(gL) / sprime)

        # Convert g_L -> g_Λ using: g_L = 2 Sym(g_Λ) L  => g_Λ = 0.5 Sym(g_L L^{-1})
        # Compute right-solve: g_L L^{-1} without forming inverse
        GL_Linv = jsl.solve_triangular(L, gL.T, lower=True).T
        g_Lam = 0.5 * (GL_Linv + GL_Linv.T)  # Sym(...)

        # ---- Natural grad in Λ-space: g^nat_Λ = 2 Λ g_Λ Λ (symmetric) ----
        g_Lam_nat = Lam @ (2.0 * g_Lam) @ Lam
        g_Lam_nat = 0.5 * (g_Lam_nat + g_Lam_nat.T)  # explicit symmetrization

        # ---- Convert Λ-space natgrad back to L-space: g^nat_L = 2 g^nat_Λ L ----
        gL_nat = 2.0 * g_Lam_nat @ L
        gL_nat = jnp.tril(gL_nat)  # keep lower-tri part

        # ---- Map L-space natgrad back to z-space (chain rule on diag) ----
        gL_nat_diag = jnp.diag(gL_nat) * sprime
        # place into a lower-tri matrix with adjusted diagonals
        GLnat_mat = gL_nat.at[jnp.arange(d), jnp.arange(d)].set(gL_nat_diag)
        # and convert to vector in the same row-major lower order:
        g_z_nat = tfm.fill_triangular_inverse(GLnat_mat)

        # ---- Pack updates with the same tree structure as grads ----
        updates = {
            "mu": g_mu_nat,
            "z_tril": g_z_nat,
        }
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# end of file
