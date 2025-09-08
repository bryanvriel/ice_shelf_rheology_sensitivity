#-*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import solve_triangular, block_diag
import h5py

import jax
import jax.numpy as jnp
import equinox as eqx

import cryogen

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


def precondition_matrix(G, S):

    # Compute the Cholesky factor of the precision matrix S. S = L_S * L_S^T
    # This is numerically stable.
    try:
        L_S = np.linalg.cholesky(S_np)
    except np.linalg.LinAlgError:
        # Add a small amount to the diagonal (Tikhonov regularization)
        # to ensure S is positive definite, then try again.
        # This is a common fix for matrices that are only positive semi-definite.
        epsilon = 1e-10 * np.mean(np.diag(S_np))
        L_S = np.linalg.cholesky(S_np + epsilon * np.eye(self.N))

    # We want to find G' such that G' = G @ (L_S^T)^-1.
    # This is equivalent to solving the linear system G' @ L_S^T = G.
    # We solve for the transpose G'^T in (G'^T @ L_S)^T = G^T => L_S^T @ G'^T = G^T
    # so we solve A x = b where A=L_S, x=G'^T, b=G^T.
    G_prime_T = solve_triangular(L_S, G_np.T, lower=True, trans="N", check_finite=False)
    G_np = G_prime_T.T

    # 3. The new precision matrix is the identity.
    S_np = np.eye(self.N)

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

    def __init__(
        self, matfile: str, field_name: str, penalties: list, precondition: bool
    ):
        """
        Initializes the matrix and associated data for a single field.

        Args:
            matfile (str): The name of the file for the matrix data.
            field_name (str): The name of the field (e.g., 'B' or 'n').
            penalties (list): List of penalty parameters for regularization.
            precondition (bool): Whether to apply preconditioning.
        """
        # Load data from HDF5 into NumPy arrays
        matrix_data = {}
        with h5py.File(matfile, "r") as fid:
            group = fid[field_name]
            for key in ("G", "Gx", "Gy"):
                matrix_data[key] = group[key][()]

        G_np = matrix_data["G"]
        self.M, self.N = G_np.shape

        # Build regularization matrix (prior precision)
        S_np = cryogen.build_regularization_matrix_dict(
            penalties, matrix_data,
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
    def prior_nll_and_grad(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the prior NLL and its gradient for this field."""
        nll, grad = mvn_nll_and_grad(z_field, self.S)
        return nll, grad

    def prior_precision(self):
        return self.S


class RheologyFieldMatrix(eqx.Module):
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

    field_B: SingleFieldMatrix
    field_n: SingleFieldMatrix

    N_B: int = eqx.field(static=True)
    N_n: int = eqx.field(static=True)
    M: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    
    def __init__(self, matfile: str, penalties: list, precondition: bool = False):

        # Create the specialized modules for each field
        self.field_B = SingleFieldMatrix(matfile, "B", penalties[0], precondition)
        self.field_n = SingleFieldMatrix(matfile, "n", penalties[1], precondition)

        # Store combined shapes for convenience
        self.N_B = self.field_B.N
        self.N_n = self.field_n.N
        self.N = self.N_B + self.N_n
        self.M = self.field_B.M

    @eqx.filter_jit
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the combined forward matrix-vector product.
        z -> [θ_B, θ_n]
        """
        # Split the input vector
        z_B = z[: self.N_B]
        z_n = z[self.N_B :]

        # Delegate to the child modules
        theta_B = self.field_B(z_B)
        theta_n = self.field_n(z_n)

        return jnp.hstack((theta_B, theta_n))

    @eqx.filter_jit
    def jvp(self, z: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the combined JVP by delegating to child modules.
        v is a tangent vector shaped like the input z.
        """
        # Split primals (z) and tangents (v) according to input shapes
        z_B, z_n = z[:self.N_B], z[self.N_B:]
        # v_B, v_n = v[0], v[1] # old way
        v_B, v_n = v[:self.N_B], v[self.N_B:]

        # Compute JVP for each field independently
        jvp_B = self.field_B.jvp(z_B, v_B)
        jvp_n = self.field_n.jvp(z_n, v_n)

        # Combine the results
        return jnp.hstack((jvp_B, jvp_n))

    @eqx.filter_jit
    def vjp(self, z: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the combined VJP by delegating to child modules.
        v is a cotangent vector shaped like the output of __call__.
        """
        # Split primals (z) according to input shapes
        z_B, z_n = z[:self.N_B], z[self.N_B:]
        v_B, v_n = v[:self.M], v[self.M:]

        # Compute VJP for each field independently
        #vjp_B = self.field_B.vjp(z_B, v[0])
        #vjp_n = self.field_n.vjp(z_n, v[1])
        vjp_B = self.field_B.vjp(z_B, v_B)
        vjp_n = self.field_n.vjp(z_n, v_n)

        # Combine the results (which are shaped like the inputs)
        return jnp.hstack((vjp_B, vjp_n))
    
    @eqx.filter_jit
    def prior_nll_and_grad(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes both the value and the gradient of the combined prior's NLL.
        """
        # Split the input vector
        z_B = z[:self.N_B]
        z_n = z[self.N_B:]

        # Delegate to the child modules
        p_B, grad_B = self.field_B.prior_nll_and_grad(z_B)
        p_n, grad_n = self.field_n.prior_nll_and_grad(z_n)

        # Combine the results
        total_nll = p_B + p_n
        total_grad = jnp.hstack((grad_B, grad_n))

        return total_nll, total_grad

    def prior_precision(self):
        return block_diag(
            self.field_B.prior_precision(), self.field_n.prior_precision()
        )


# end of file
