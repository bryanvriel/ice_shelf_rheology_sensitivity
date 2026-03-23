#-*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import solve_triangular
import h5py

import jax
import jax.numpy as jnp
import equinox as eqx
import cryogen

from .config_utilities import primary_penalty, validate_cfg
from .file_utilities import make_matrix_filename

# Create a mapping of types to their respective compute/inverse logic
NORM_MAPPING = {
    "tanh": (cryogen.compute_tanh_norm_params, cryogen.tanh_inverse),
    "cdf":  (cryogen.compute_cdf_norm_params, cryogen.cdf_inverse),
    "sp":   (cryogen.compute_softplus_norm_params, cryogen.softplus_inverse),
}

FWD_NORM_MAPPING = {
    "tanh": (cryogen.compute_tanh_norm_params, cryogen.tanh_forward),
    "cdf":  (cryogen.compute_cdf_norm_params, cryogen.cdf_forward),
    "sp":   (cryogen.compute_softplus_norm_params, cryogen.softplus_forward),
}

def inv_norm(θ, opts, norm_type):
    if norm_type not in NORM_MAPPING:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    param_fn, inv_fn = NORM_MAPPING[norm_type]
    p = param_fn(**opts)
    return inv_fn(θ, p)

def fwd_norm(θ, opts, norm_type):
    if norm_type not in FWD_NORM_MAPPING:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    param_fn, fwd_fn = FWD_NORM_MAPPING[norm_type]
    p = param_fn(**opts)
    return np.array(fwd_fn(θ, p))

def make_spd(
    A: np.ndarray,
    jitter_init: float = 1e-8,
    max_tries: int = 8,
) -> np.ndarray:
    """
    Ensure matrix is symmetric positive definite via jittering.

    Args:
        A: Input matrix.
        jitter_init: Initial jitter to add to diagonal.
        max_tries: Maximum number of jitter increases to attempt.

    Returns:
        SPD matrix.
    """
    M = np.asarray(A, dtype=float)
    M = 0.5 * (M + M.T)
    jitter = jitter_init
    for _ in range(max_tries + 1):
        try:
            np.linalg.cholesky(M)
            return M
        except np.linalg.LinAlgError:
            M = M + jitter * np.eye(M.shape[0])
            jitter *= 10.0
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    shift = max(0.0, -w[0] + jitter_init)
    return M + shift * np.eye(M.shape[0])


def mvn_nll(z: jnp.ndarray, S: jnp.ndarray) -> float:
    """
    Computes the un-normalized negative log-likelihood for a Gaussian prior.
    This corresponds to 0.5 * z.T @ S @ z, where S is the precision matrix.
    """
    return 0.5 * jnp.dot(z, S @ z)

# We could compute this explicitly, but let's use jax
# for now to keep the workflow general for other priors
mvn_nll_and_grad = jax.value_and_grad(mvn_nll)


def tikhonov(z: jnp.ndarray, penalty: float) -> float:
    return penalty * jnp.dot(z, z)

tikhonov_and_grad = jax.value_and_grad(tikhonov, argnums=0)


def create_rheology_field(cfg, hyper_params=None):
    cfg = validate_cfg(cfg)
    if hyper_params is None:
        hyper_params = cfg.inverse.hyper_params

    if cfg.matrix.spectral:
        raise NotImplementedError(
            "matrix.spectral is not part of the current golden path. "
            "Generate the standard matrix file instead."
        )
    else:
        matfile = make_matrix_filename(cfg)

    # Unpack hyper or compute hyper parameters
    if not cfg.inverse.joint:
        field = SingleFieldMatrix(
            matfile,
            "B",
            cfg.inverse.penalties_B,
            precondition=cfg.inverse.precondition,
        )
        return field

    # Unpack hyper parameters
    if len(hyper_params) == 0:
        σ_B = np.sqrt(1 / primary_penalty(cfg.inverse.penalties_B))
        σ_n = np.sqrt(1 / primary_penalty(cfg.inverse.penalties_n))
        ρ = 0.0
    elif len(hyper_params) == 2:
        σ_B, σ_n = hyper_params
        ρ = 0.0
    elif len(hyper_params) == 3:
        σ_B, σ_n, ρ = hyper_params
    else:
        raise ValueError('hyper_params can only have 0, 2, or 3 elements.')

    print('Field hyperparameters:', σ_B, σ_n, ρ)

    # Construct prior block covariance matrix
    Σ = np.array([
        [σ_B**2, ρ * σ_B * σ_n],
        [ρ * σ_B * σ_n, σ_n**2],
    ])

    # Compute precision (inverse of covariance)
    try:
        Λ = np.linalg.inv(Σ)
    except np.linalg.LinAlgError:
        Λ = np.linalg.inv(Σ + 1e-8 * np.eye(2))

    # Create and return joint field
    return RheologyFieldLambdaMatrix(matfile, Λ, mask=cfg.matrix.mask)


def precondition_matrix(G, S):
    G_np = np.asarray(G, dtype=float)
    S_np = make_spd(S)

    # Compute the Cholesky factor of the precision matrix S. S = L_S * L_S^T
    # This is numerically stable.
    L_S = np.linalg.cholesky(S_np)

    # We want to find G' such that G' = G @ (L_S^T)^-1.
    # This is equivalent to solving the linear system G' @ L_S^T = G.
    # We solve for the transpose G'^T in (G'^T @ L_S)^T = G^T => L_S^T @ G'^T = G^T
    # so we solve A x = b where A=L_S, x=G'^T, b=G^T.
    G_prime_T = solve_triangular(L_S, G_np.T, lower=True, trans="N", check_finite=False)
    G_np = G_prime_T.T

    # 3. The new precision matrix is the identity.
    S_np = np.eye(G_np.shape[1])

    return G_np, S_np


class SingleFieldMatrix(eqx.Module):
    """
    Handles the matrix operations for a single rheological field.
    This module is intended to be composed within a larger structure.
    """

    G: jnp.ndarray  # The core forward matrix
    S: jnp.ndarray  # The prior precision (regularization) matrix

    M: int = eqx.field(static=True)     # Number of points in the field
    N: int = eqx.field(static=True)     # Number of parameters for this field

    matrix_data: dict

    def __init__(
        self, matfile: str | dict, field_name: str, penalty: float, precondition: bool
    ):
        """
        Initializes the matrix and associated data for a single field.

        Args:
            matfile (str): The name of the file for the matrix data.
            field_name (str): The name of the field (e.g., 'B' or 'n').
            penalty (float): Penalty parameter for regularization.
            precondition (bool): Whether to apply preconditioning.
        """
        # Load data from HDF5 into NumPy arrays or copy dict
        if isinstance(matfile, dict):
            matrix_data = matfile[field_name]
        else:
            matrix_data = {}
            with h5py.File(matfile, "r") as fid:
                group = fid[field_name]
                for key in ("G", "Gx", "Gy", "G_grid"):
                    try:
                        matrix_data[key] = group[key][()]
                    except KeyError:
                        pass

        self.matrix_data = matrix_data
        G_np = matrix_data["G"]
        self.M, self.N = G_np.shape

        # Build regularization matrix (prior precision)
        S_np = cryogen.build_regularization_matrix_dict(
            penalty, matrix_data,
        )

        # Apply preconditioning if requested
        if precondition:
            G_np, S_np = precondition_matrix(G_np, S_np)

        # Convert final arrays to JAX arrays and store
        self.G = jnp.asarray(G_np)
        self.S = jnp.asarray(S_np)

    @eqx.filter_jit
    def __call__(self, z_field: jnp.ndarray) -> jnp.ndarray:
        """Performs the forward matrix-vector product for this field."""
        return self.G @ z_field

    @eqx.filter_jit
    def jvp(self, z_field: jnp.ndarray, v_field: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian-vector product (J @ v) for this field.
        """
        # eqx.jvp returns the primal output and the tangent. We only need the tangent.
        _, tangent_out = eqx.filter_jvp(self, (z_field,), (v_field,))
        return tangent_out

    @eqx.filter_jit
    def vjp(self, z_field: jnp.ndarray, v_field: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector-Jacobian product (v^T @ J) for this field.
        """
        # eqx.vjp returns the primal output and the vjp function.
        _, vjp_fun = eqx.filter_vjp(self, z_field)
        # The vjp_fun takes a cotangent vector shaped like the output of f.
        grad_z, = vjp_fun(v_field)
        return grad_z

    @eqx.filter_jit
    def prior_nll(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray]:
        """Computes the prior NLL for this field."""
        nll = mvn_nll(z_field, self.S)
        return nll

    @eqx.filter_jit
    def prior_nll_and_grad(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the prior NLL and its gradient for this field."""
        nll, grad = mvn_nll_and_grad(z_field, self.S)
        return nll, grad

    def prior_precision(self):
        return self.S

    def update_G(self, G):
        self.G[:] = G[:]


class RheologyFieldLambdaMatrix(eqx.Module):
    """
    A class to handle rheology matrices by composing two SingleFieldMatrix modules.
    This version delegates field-specific logic to its children, avoiding
    code repetition.

    Attributes:
        field_B (SingleFieldMatrix): The module for the 'B' field.
        field_n (SingleFieldMatrix): The module for the 'n' field.
        N_B (int): Number of columns for 'B' field.
        N_n (int): Number of columns for 'n' field.
        N (int): Total number of parameters.
        M (int): Total number of points in the field.
    """

    G_B: jnp.ndarray
    G_n: jnp.ndarray

    N_B: int = eqx.field(static=True)
    N_n: int = eqx.field(static=True)
    M: int = eqx.field(static=True)
    N: int = eqx.field(static=True)

    matrix_data: dict
    S: jnp.ndarray

    def __init__(self, matfile: str | dict, Lambda: jnp.ndarray, mask: str = None):
        if isinstance(matfile, dict):
            G_B = matfile["B"]["G"]
            G_n = matfile["n"]["G"]
        else:
            with h5py.File(matfile, "r") as fid:
                G_B = fid['B/G'][()]
                G_n = fid['n/G'][()]

        # Optional coefficient mask (e.g., for removing ill-constrained wavelets)
        if mask not in (None, "none"):
            with h5py.File(mask, "r") as fid:
                mask_B = fid["B"][()].astype(bool)
                mask_n = fid["n"][()].astype(bool)
                G_B = G_B[:, mask_B]
                G_n = G_n[:, mask_n]

        self.G_B = jnp.asarray(G_B)
        self.G_n = jnp.asarray(G_n)
        self.N_B = self.G_B.shape[1]
        self.N_n = self.G_n.shape[1]
        self.M = self.G_B.shape[0]
        self.N = self.N_B + self.N_n

        # Combined matrix_data
        self.matrix_data = {
            "B": {"G": self.G_B},
            "n": {"G": self.G_n},
        }

        # Build precision matrix
        self.S = cryogen.build_lambda_precision(Lambda, self.N_B, self.N_n)


    @eqx.filter_jit
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the combined forward matrix-vector product.
        z -> [θ_B, θ_n]
        """
        # Split the input vector
        z_B = z[: self.N_B]
        z_n = z[self.N_B :]
        return jnp.hstack((self.G_B @ z_B, self.G_n @ z_n))

    @eqx.filter_jit
    def jvp(self, z_field: jnp.ndarray, v_field: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian-vector product (J @ v) for this field.
        """
        # eqx.jvp returns the primal output and the tangent. We only need the tangent.
        _, tangent_out = eqx.filter_jvp(self, (z_field,), (v_field,))
        return tangent_out

    @eqx.filter_jit
    def vjp(self, z_field: jnp.ndarray, v_field: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the vector-Jacobian product (v^T @ J) for this field.
        """
        # eqx.vjp returns the primal output and the vjp function.
        _, vjp_fun = eqx.filter_vjp(self, z_field)
        # The vjp_fun takes a cotangent vector shaped like the output of f.
        grad_z, = vjp_fun(v_field)
        return grad_z

    @eqx.filter_jit
    def prior_nll(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray]:
        """Computes the prior NLL for this field."""
        nll = mvn_nll(z_field, self.S)
        return nll

    @eqx.filter_jit
    def prior_nll_and_grad(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the prior NLL and its gradient for this field."""
        nll, grad = mvn_nll_and_grad(z_field, self.S)
        return nll, grad

    def prior_precision(self):
        return self.S

# end of file
