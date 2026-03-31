#-*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag, solve_triangular
import h5py

import jax
import jax.numpy as jnp
import equinox as eqx
import cryogen

from .config_utilities import primary_penalty, validate_cfg
from .file_utilities import make_matrix_filename

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

mvn_nll_and_grad = jax.value_and_grad(mvn_nll)


def periodic_bspline_design_matrix(s, L, n_basis, degree=3):
    """
    Build a periodic B-spline design matrix A such that
        profile(s) = A @ alpha
    where alpha has length n_basis.
    """
    s = np.asarray(s, dtype=float)
    p = degree
    n = n_basis
    h = L / n

    # Wrap all coordinates into [0, L)
    s = np.mod(s, L)

    # Uniform extended knot vector
    knots = np.arange(-p, n + p + 1, dtype=float) * h

    # Ordinary spline design matrix with n+p coefficients
    B_open = BSpline.design_matrix(s, knots, p, extrapolate=False).toarray()

    # Wrap matrix: c_ext = P @ alpha, with c_{n+r} = c_r
    P = np.zeros((n + p, n))
    P[:n, :] = np.eye(n)
    P[n:, :p] = np.eye(p)

    # Periodic design matrix
    A = B_open @ P
    return A


def periodic_bspline_coefficient_prior_covariance_corrlen(
    n,
    L,
    sigma_alpha,
    corr_length,
    design_matrix=None,
    jitter=1e-10,
    return_positions=False,
):
    """
    Construct a periodic Gaussian prior covariance for spline coefficients
    using a periodic squared-exponential kernel on the closed boundary.

    The correlation length is interpreted in the same physical arc-length
    units as ``L``. The kernel is constructed from the periodic chord
    distance

        d_p(ds) = (L / pi) * sin(pi * ds / L),

    so that for small separations it matches the standard squared-exponential
    form exp(-0.5 * ds^2 / corr_length^2).

    If ``design_matrix`` is provided, ``sigma_alpha`` is interpreted as the
    target RMS standard deviation of the induced boundary-velocity deviations
    ``delta_u = A @ alpha`` over the evaluation points. A single global scale
    factor is applied so that

        sqrt(mean(diag(A Sigma A^T))) ~= sigma_alpha.

    If ``design_matrix`` is omitted, ``sigma_alpha`` falls back to the
    coefficient standard deviation scale.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if L <= 0:
        raise ValueError("L must be positive")
    if sigma_alpha <= 0:
        raise ValueError("sigma_alpha must be positive")
    if corr_length <= 0:
        raise ValueError("corr_length must be positive")

    # Coefficient centers, uniformly spaced around the closed boundary
    s = np.linspace(0.0, L, n, endpoint=False)

    # Pairwise differences
    ds = s[:, None] - s[None, :]

    # Periodic squared-exponential kernel with physical arc-length scaling.
    periodic_distance = (L / np.pi) * np.sin(np.pi * ds / L)
    Sigma0 = np.exp(
        -0.5 * (periodic_distance / corr_length)**2
    )

    if design_matrix is not None:
        A = np.asarray(design_matrix, dtype=float)
        if A.ndim != 2 or A.shape[1] != n:
            raise ValueError(
                "design_matrix must have shape (n_points, n) matching the "
                "number of spline coefficients."
            )
        induced_diag = np.einsum("ij,jk,ik->i", A, Sigma0, A)
        induced_rms = float(np.sqrt(np.mean(induced_diag)))
        if not np.isfinite(induced_rms) or induced_rms <= 0.0:
            raise ValueError(
                "Could not determine a positive induced RMS scale from the "
                "design matrix and coefficient covariance."
            )
        scale = sigma_alpha / induced_rms
    else:
        scale = sigma_alpha

    Sigma = scale**2 * Sigma0

    if jitter > 0.0:
        Sigma = Sigma + jitter * np.eye(n)

    if return_positions:
        return Sigma, s
    return Sigma


def closed_boundary_length(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        raise ValueError("coords must have shape (n_points, 2) with n_points >= 2")

    diffs = np.diff(np.vstack((coords, coords[:1])), axis=0)
    return float(np.sqrt((diffs**2).sum(axis=1)).sum())


def boundary_basis_count_from_spacing(
    boundary_length: float,
    basis_spacing: float,
    degree: int,
) -> tuple[int, float]:
    if boundary_length <= 0:
        raise ValueError("boundary_length must be positive")
    if basis_spacing <= 0:
        raise ValueError("basis_spacing must be positive")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    n_basis = int(np.ceil(boundary_length / basis_spacing))
    n_basis = max(n_basis, degree + 1)
    actual_spacing = boundary_length / n_basis
    return n_basis, actual_spacing


def resolve_boundary_basis_config(cfg, boundary_length: float) -> tuple[int, float, float | None]:
    degree = int(cfg.inverse.boundary_spline_degree)
    basis_spacing = getattr(cfg.inverse, "boundary_basis_spacing", None)
    n_basis = getattr(cfg.inverse, "boundary_n_basis", None)

    if basis_spacing is not None and n_basis is not None:
        raise ValueError(
            "Specify only one of `inverse.boundary_basis_spacing` and "
            "`inverse.boundary_n_basis`."
        )

    if basis_spacing is not None:
        n_basis_resolved, actual_spacing = boundary_basis_count_from_spacing(
            boundary_length=boundary_length,
            basis_spacing=float(basis_spacing),
            degree=degree,
        )
        return n_basis_resolved, actual_spacing, float(basis_spacing)

    if n_basis is not None:
        n_basis_resolved = int(n_basis)
        if n_basis_resolved < degree + 1:
            raise ValueError(
                "`inverse.boundary_n_basis` must be at least "
                "`inverse.boundary_spline_degree + 1`."
            )
        return n_basis_resolved, boundary_length / n_basis_resolved, None

    raise ValueError(
        "Boundary inference requires either `inverse.boundary_basis_spacing` "
        "or `inverse.boundary_n_basis`."
    )


def boundary_inference_enabled(cfg) -> bool:
    return bool(getattr(cfg.inverse, "infer_boundary_velocity", False))


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


def create_inference_field(cfg, problem=None, hyper_params=None):
    rheology_field = create_rheology_field(cfg, hyper_params=hyper_params)

    if not boundary_inference_enabled(cfg):
        return rheology_field

    if problem is None:
        raise ValueError(
            "Boundary-velocity inference requires a problem instance with loaded data."
        )

    return JointBoundaryRheologyField(cfg, rheology_field, problem)


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


class JointBoundaryRheologyField(eqx.Module):
    rheology_field: eqx.Module
    A: jnp.ndarray
    S: jnp.ndarray
    bc_reference: jnp.ndarray
    boundary_nodes: jnp.ndarray
    coeff_positions: jnp.ndarray
    matrix_data: dict

    output_size_rheology: int = eqx.field(static=True)
    bc_size: int = eqx.field(static=True)
    bc_shape: tuple = eqx.field(static=True)
    N_rheo: int = eqx.field(static=True)
    N_B: int = eqx.field(static=True)
    N_n: int = eqx.field(static=True)
    Nt: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    M: int = eqx.field(static=True)
    boundary_length: float = eqx.field(static=True)
    basis_spacing: float = eqx.field(static=True)
    basis_spacing_target: float | None = eqx.field(static=True)

    def __init__(self, cfg, rheology_field, problem):
        if problem.boundary_node_indices is None or problem.boundary_coords is None:
            problem._init_boundary_geometry()

        degree = int(cfg.inverse.boundary_spline_degree)
        sigma_alpha = float(cfg.inverse.boundary_sigma_alpha)
        corr_length = float(cfg.inverse.boundary_corr_length)

        if len(problem.u_bc_ref) == 0:
            raise ValueError("Problem does not have any boundary-condition reference fields.")

        bc_shape = tuple(problem.u_bc_ref[0].dat.data_ro.shape)
        if len(bc_shape) != 2 or bc_shape[1] != 2:
            raise NotImplementedError(
                "Boundary-velocity inference currently requires 2D vector velocity fields."
            )

        boundary_coords = np.asarray(problem.boundary_coords, dtype=float)
        boundary_s = np.asarray(problem.boundary_s, dtype=float)
        boundary_L = closed_boundary_length(boundary_coords)
        n_basis, basis_spacing, basis_spacing_target = resolve_boundary_basis_config(
            cfg,
            boundary_L,
        )

        A_np = periodic_bspline_design_matrix(
            boundary_s,
            boundary_L,
            n_basis=n_basis,
            degree=degree,
        )
        Sigma_alpha, coeff_positions = periodic_bspline_coefficient_prior_covariance_corrlen(
            n=n_basis,
            L=boundary_L,
            sigma_alpha=sigma_alpha,
            corr_length=corr_length,
            design_matrix=A_np,
            return_positions=True,
        )
        Q_alpha = np.linalg.inv(make_spd(Sigma_alpha))

        bc_reference = np.stack(
            [u.dat.data_ro.ravel().copy() for u in problem.u_bc_ref],
            axis=0,
        )

        rheology_zero = np.zeros((rheology_field.N,), dtype=float)
        output_size_rheology = np.asarray(rheology_field(rheology_zero)).size

        blocks = [np.asarray(rheology_field.S)]
        for _ in range(problem.Nt):
            blocks.extend((Q_alpha, Q_alpha))

        self.rheology_field = rheology_field
        self.A = jnp.asarray(A_np)
        self.S = jnp.asarray(make_spd(block_diag(*blocks)))
        self.bc_reference = jnp.asarray(bc_reference)
        self.boundary_nodes = jnp.asarray(problem.boundary_node_indices, dtype=jnp.int32)
        self.coeff_positions = jnp.asarray(coeff_positions)
        self.matrix_data = getattr(rheology_field, "matrix_data", {})

        self.output_size_rheology = int(output_size_rheology)
        self.bc_size = int(np.prod(bc_shape))
        self.bc_shape = bc_shape
        self.N_rheo = rheology_field.N
        self.N_B = int(getattr(rheology_field, "N_B", rheology_field.N))
        self.N_n = int(getattr(rheology_field, "N_n", 0))
        self.Nt = problem.Nt
        self.n_basis = n_basis
        self.degree = degree
        self.N = self.N_rheo + 2 * self.Nt * self.n_basis
        self.M = rheology_field.M
        self.boundary_length = boundary_L
        self.basis_spacing = float(basis_spacing)
        self.basis_spacing_target = basis_spacing_target

    def epoch_alpha_slices(self, k: int):
        offset = self.N_rheo + 2 * k * self.n_basis
        return (
            slice(offset, offset + self.n_basis),
            slice(offset + self.n_basis, offset + 2 * self.n_basis),
        )

    def _boundary_control_from_alpha(
        self,
        ref_flat: jnp.ndarray,
        alpha_x: jnp.ndarray,
        alpha_y: jnp.ndarray,
    ) -> jnp.ndarray:
        bc = ref_flat.reshape(self.bc_shape)
        delta_x = self.A @ alpha_x
        delta_y = self.A @ alpha_y
        bc = bc.at[self.boundary_nodes, 0].add(delta_x)
        bc = bc.at[self.boundary_nodes, 1].add(delta_y)
        return bc.reshape(-1)

    def _boundary_tangent_from_alpha(
        self,
        alpha_x: jnp.ndarray,
        alpha_y: jnp.ndarray,
    ) -> jnp.ndarray:
        tangent = jnp.zeros(self.bc_shape, dtype=alpha_x.dtype)
        tangent = tangent.at[self.boundary_nodes, 0].set(self.A @ alpha_x)
        tangent = tangent.at[self.boundary_nodes, 1].set(self.A @ alpha_y)
        return tangent.reshape(-1)

    @eqx.filter_jit
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z_rheo = z[: self.N_rheo]
        outputs = [self.rheology_field(z_rheo)]

        for k in range(self.Nt):
            sx, sy = self.epoch_alpha_slices(k)
            outputs.append(
                self._boundary_control_from_alpha(
                    self.bc_reference[k],
                    z[sx],
                    z[sy],
                )
            )

        return jnp.concatenate(outputs)

    @eqx.filter_jit
    def jvp(self, z_field: jnp.ndarray, v_field: jnp.ndarray) -> jnp.ndarray:
        tangent_outputs = [self.rheology_field.jvp(z_field[: self.N_rheo], v_field[: self.N_rheo])]

        for k in range(self.Nt):
            sx, sy = self.epoch_alpha_slices(k)
            tangent_outputs.append(
                self._boundary_tangent_from_alpha(
                    v_field[sx],
                    v_field[sy],
                )
            )

        return jnp.concatenate(tangent_outputs)

    @eqx.filter_jit
    def vjp(self, z_field: jnp.ndarray, v_field: jnp.ndarray) -> jnp.ndarray:
        grad_parts = [
            self.rheology_field.vjp(
                z_field[: self.N_rheo],
                v_field[: self.output_size_rheology],
            )
        ]

        offset = self.output_size_rheology
        for _ in range(self.Nt):
            bc_grad = v_field[offset : offset + self.bc_size].reshape(self.bc_shape)
            grad_parts.append(self.A.T @ bc_grad[self.boundary_nodes, 0])
            grad_parts.append(self.A.T @ bc_grad[self.boundary_nodes, 1])
            offset += self.bc_size

        return jnp.concatenate(grad_parts)

    @eqx.filter_jit
    def prior_nll(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray]:
        nll = mvn_nll(z_field, self.S)
        return nll

    @eqx.filter_jit
    def prior_nll_and_grad(
        self, z_field: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        nll, grad = mvn_nll_and_grad(z_field, self.S)
        return nll, grad

    def prior_precision(self):
        return self.S

# end of file
