#-*- coding: utf-8 -*-

from functools import partial
import numpy as np
from scipy.spatial import cKDTree
import h5py
import os
from typing import Optional, Tuple

import firedrake
from firedrake import inner, assemble
import icepack
from icepack.calculus import grad
import rasterio


import cryogen
from cryogen import interpolate

from . import models
from .utilities import coordinate_values, const_to_R_fn


def make_mesh(meshfile=None, comm=None, mesh_span=None, mesh_spacing=None):
    if comm is not None:
        if meshfile:
            mesh = firedrake.Mesh(meshfile, comm=comm)
        else:
            N = int(cfg.model.mesh_span / cfg.model.mesh_spacing)
            mesh = firedrake.IntervalMesh(N, cfg.model.mesh_span, comm=comm)
    else:
        if meshfile:
            mesh = firedrake.Mesh(meshfile)
        else:
            N = int(cfg.model.mesh_span / cfg.model.mesh_spacing)
            mesh = firedrake.IntervalMesh(N, cfg.model.mesh_span)
    return mesh


class IceShelfProblem:

    def __init__(
        self,
        cfg,
        mesh,
        dirichlet_ids=[
            1,
        ],
        side_wall_ids=[],
        degree=2,
        side_friction=0.0,
        nll_scale=1.0,
        default_vel_error=10.0,
        solver_type="petsc",
        n_norm="tanh",
        B_norm="log",
        use_strain=False,
        fixed_exponent=False,
        temporal_cov=True,
    ):

        # Initialize functional spaces
        self.mesh = mesh
        self.Q = firedrake.FunctionSpace(self.mesh, family="CG", degree=degree)
        self.R = firedrake.FunctionSpace(self.mesh, family="R", degree=0)
        self.coords = coordinate_values(self.Q)
        self.N_pts = self.coords.shape[0]
        if self.coords.ndim == 2:
            self.V = firedrake.VectorFunctionSpace(
                self.mesh, family="CG", degree=degree
            )
            self.udim = 2
        else:
            self.V = firedrake.FunctionSpace(self.mesh, family="CG", degree=degree)
            self.udim = 1

        # Scaling factor for data NLL
        # Dividing by area after integration essentially gives us MSE
        # I originally wanted to multiply by N_pts, but it highly inflates
        # the scale of the likelihood because we assume independent observations
        # By using this "density" based scaling, the data likelihood becomes
        # invariant to the size of the modeling domain (e.g., Brinkerhoff, 2021)
        area = assemble(firedrake.Constant(1.0) * firedrake.dx(self.mesh))
        self.area = area
        self.nll_scale = firedrake.Constant(nll_scale / area)

        # Mesh metadata for assembling
        quad_degree = 3 * (degree - 1) + 2 * degree
        self.mesh_metadata = {"quadrature_degree": quad_degree}
        self._dirichlet_ids = dirichlet_ids
        self._side_wall_ids = side_wall_ids

        # Initialize flow model and normalization functions
        self.model = models.CustomIceShelf()
        self.init_normalizers(B_norm, n_norm, cfg)

        # Initialize constants
        self.Cs = firedrake.Constant(side_friction)
        self.d_pert = firedrake.Constant(0.0)
        self.dirichlet_ids = dirichlet_ids
        self.side_wall_ids = side_wall_ids
        self.solver_type = solver_type

        # Initialize function spaces for observations (potentially time-dependent)
        # These need to be loaded prior to any forward solves
        self.h = firedrake.Function(self.Q)
        self.u = []
        self.σ_u = []
        self.e_xx = []
        self.e_xy = []
        self.e_yy = []
        self.log_e_ref = firedrake.Function(self.Q)
        self.σ_eps = firedrake.Constant(0.001)
        self.σ_u_default = default_vel_error
        self.use_strain = use_strain

        # Placeholders for rheology fields (only used when fixed_exponent = True)
        # Also used for self.reset_solver()
        self.θ_B = firedrake.Function(self.Q)
        self.θ_n = firedrake.Function(self.Q)

        # Set the NLL function, depending on whether we want temporal covariance
        self.temporal_cov = temporal_cov
        if temporal_cov:
            self.data_nll = self._data_nll_covariance
        else:
            self.data_nll = self._data_nll

        # List of control variables
        # Not used here, but as an interface to FiredrakeObjective
        self.fixed_exponent = fixed_exponent
        if fixed_exponent:
            self.forward = self._forward_rigidity
        else:
            self.forward = self._forward

        self._obs_init = False
        self._solver_init = False

    @property
    def controls(self):
        if self.fixed_exponent:
            return [self.θ_B,]
        else:
            return [self.θ_B, self.θ_n]

    @controls.setter
    def controls(self, value):
        raise ValueError("Cannot set controls.")

    def grid_to_mesh_array(self, grid, grid_coords):

        mesh_coords = (self.coords[:, 0], self.coords[:, 1])

        def _interp_to_mesh(
            array: np.ndarray, grid_coords: Tuple[np.ndarray, np.ndarray],
        ) -> np.ndarray:
            """Simple interpolation of ``array`` onto the mesh coordinates."""
            return interpolate.interp(array, grid_coords, mesh_coords)

        if grid.ndim == 3:
            interpolated = np.column_stack(
                [
                    _interp_to_mesh(grid[i], grid_coords)
                    for i in range(grid.shape[0])
                ]
            )
        elif grid.ndim == 2:
            interpolated = _interp_to_mesh(grid, grid_coords)
        else:
            raise ValueError('Grid has unsupported shape.')

        return interpolated

    def load_observations(self, data_file, time_ind: Optional[list] = None):
        """Load observation fields defined on a structured grid.

        Parameters
        ----------
        data_file:
            Path to an HDF5 file containing the observation fields.
        time_ind:
            List of time indices to load. If ``None``, all epochs are used.
        """
        with h5py.File(data_file, "r") as fid:
            h, u, e_ref = [fid[key][()] for key in ('h', 'u', 'eps_ref')]
            grid_coords = (fid["x"][()], fid["y"][()])

            if time_ind is None:
                self.Nt = u.shape[0]
                time_ind = range(self.Nt)
            else:
                self.Nt = len(time_ind)

            if self.use_strain:
                e_xx, e_xy, e_yy = [
                    fid[key][()] for key in
                    ("e_xx", "e_xy", "e_yy")
                ]

            if self.temporal_cov:
                self.C_temporal = fid["C_temporal"][()]

        # Time-invariant fields
        self.h.dat.data[:] = self.grid_to_mesh_array(h, grid_coords)
        self.log_e_ref.dat.data[:] = self.grid_to_mesh_array(np.log(e_ref), grid_coords)
        self.e_ref_grid = e_ref

        # Interpolate the time epoch data
        for k in time_ind:

            # Local function spaces
            u_k = firedrake.Function(self.V)
            σ_k = firedrake.Function(self.V)
            #e_xx_k = firedrake.Function(self.Q)
            #e_xy_k = firedrake.Function(self.Q)
            #e_yy_k = firedrake.Function(self.Q)

            # Interpolate velocity components and errors
            u_k.dat.data[:] = self.grid_to_mesh_array(u[k], grid_coords)
            σ_k.dat.data[:] = self.σ_u_default
            #_interp_field(u_sigma[k], grid_coords, σ_k)

            # Interpolate strain components
            #_interp_field(e_xx[k], grid_coords, e_xx_k)
            #_interp_field(e_xy[k], grid_coords, e_xy_k)
            #_interp_field(e_yy[k], grid_coords, e_yy_k)

            # Append to lists
            self.u.append(u_k)
            self.σ_u.append(σ_k)
            #self.e_xx.append(e_xx_k)
            #self.e_xy.append(e_xy_k)
            #self.e_yy.append(e_yy_k)

        self._obs_init = True

    def smooth_boundaries(self, alpha_mag=3.0e3, alpha_theta=4.0e3):

        # Get boundary coordinates
        bc = firedrake.DirichletBC(self.V, 0, "on_boundary")
        boundary_node_indices = bc.nodes
        edge_coords = self.coords[boundary_node_indices]

        # Compute centroid
        centroid = edge_coords.mean(axis=0)

        # Compute angles from centroid
        angles = np.arctan2(
            edge_coords[:, 1] - centroid[1],
            edge_coords[:, 0] - centroid[0]
        )

        # Sort counter-clockwise
        ccw_order = np.argsort(angles)
        boundary_node_indices = boundary_node_indices[ccw_order]
        edge_coords = self.coords[boundary_node_indices]
        s_edge = compute_path_length(edge_coords[:, 0], edge_coords[:, 1])
        self.boundary_node_indices = boundary_node_indices

        def _smooth(ref_array, alpha):
            bvals = ref_array
            smooth_vals = helmholtz_1d_nonuniform(s_edge, bvals, alpha)
            return bvals, smooth_vals

        # Smooth the boundary data
        for k in range(self.Nt):
            u = self.u[k].dat.data_ro[boundary_node_indices, :]
            mag = np.linalg.norm(u, axis=1)
            theta = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
            mag, mag_smooth = _smooth(mag, alpha_mag)
            theta, theta_smooth = _smooth(theta, alpha_theta)
            u = mag_smooth * np.cos(np.radians(theta_smooth))
            v = mag_smooth * np.sin(np.radians(theta_smooth))
            self.u[k].dat.data[boundary_node_indices] = np.column_stack((u, v))

    def load_rheology(self, dirname, n_exp=None):
        print('Loading true rheology')
        try:
            with rasterio.open(os.path.join(dirname, "n_exp.tif")) as r_n:
                n_fn = icepack.interpolate(r_n, self.Q)
        except rasterio.errors.RasterioIOError:
            n_fn = firedrake.Function(self.Q)
            n_fn.dat.data[:] = 3.0

        # Overwrite with any exponent provided as kwarg
        if n_exp is not None:
            print('Setting exponent to', n_exp)
            n_fn.dat.data[:] = n_exp

        # Normalize and assemble
        θ_n = self._fwd_norm_n(n_fn)
        self.θ_n = assemble(firedrake.interpolate(θ_n, self.Q))

    def init_solvers(self):
        self.solvers = []
        for _ in range(self.Nt):
            solver, _ = models.create_solver(
                self.model,
                self.solver_type,
                dirichlet_ids=self.dirichlet_ids,
                side_wall_ids=self.side_wall_ids,
            )
            self.solvers.append(solver)
        self._solver_init = True

    def init_normalizers(self, B_norm, n_norm, cfg):
        params = {
            "B": {
                "low": cfg.model.B_min,
                "high": cfg.model.B_max,
                "tau": cfg.inverse.norm_scale,
                "C0": 1.0
            },
            "n": {
                "low": cfg.model.n_min,
                "high": cfg.model.n_max,
                "tau": cfg.inverse.norm_scale,
                "C0": 3.0
            },
        }

        # Create inverse normalizers
        def create_fn(norm_type, key):
            p = params[key]
            if norm_type == "log":
                inv = partial(models.inv_norm_log, **p)
                fwd = partial(models.fwd_norm_log, **p)
            elif norm_type == "tanh":
                inv = partial(models.inv_norm_tanh, **p)
                fwd = partial(models.fwd_norm_tanh, **p)
            elif norm_type == "arctan":
                inv = partial(models.inv_norm_arctan, **p)
                fwd = partial(models.fwd_norm_arctan, **p)
            elif norm_type == "cdf":
                inv = partial(models.inv_norm_cdf, **p)
                fwd = partial(models.fwd_norm_cdf, **p)
            elif norm_type == "sp":
                inv = partial(models.inv_norm_softplus, **p)
                fwd = partial(models.fwd_norm_softplus, **p)
            else:
                raise ValueError("Unsupported norm type.")

            return inv, fwd

        if B_norm == "viscosity":
            _inv_norm, self._fwd_norm_n = create_fn(n_norm, "n")
            def _inv_norm_n(theta):
                n = _inv_norm(theta)
                self.θ_n = assemble(firedrake.interpolate(theta, self.Q))
                return n
            self._inv_norm_n = _inv_norm_n

            def _inv_norm_B(theta_nu):
                n_exp = self._inv_norm_n(self.θ_n)
                return viscosity_to_rigidity(theta_nu, self.log_e_ref, n_exp)
            self._inv_norm_B = _inv_norm_B

        else:
            self._inv_norm_B, self._fwd_norm_B = create_fn(B_norm, "B")
            self._inv_norm_n, self._fwd_norm_n = create_fn(n_norm, "n")


    def _forward(self, params):

        # Unpack and un-normalize parameters
        θ_B, θ_n = params
        n_exp = assemble(firedrake.interpolate(self._inv_norm_n(θ_n), self.Q))
        B = assemble(firedrake.interpolate(self._inv_norm_B(θ_B), self.Q))

        # Loop over epochs to compute velocities
        u_pred = []
        for k in range(self.Nt):
            u_pred.append(
                self.solvers[k].diagnostic_solve(
                    velocity=self.u[k],
                    thickness=self.h,
                    rigidity=B,
                    exponent=n_exp,
                )
            )

        return u_pred

    def _forward_rigidity(self, params):
        θ_B = params[0]
        B = assemble(firedrake.interpolate(self._inv_norm_B(θ_B), self.Q))
        n_exp = assemble(firedrake.interpolate(self._inv_norm_n(self.θ_n), self.Q))

        # Loop over epochs to compute velocities
        u_pred = []
        for k in range(self.Nt):
            u_pred.append(
                self.solvers[k].diagnostic_solve(
                    velocity=self.u[k],
                    thickness=self.h,
                    rigidity=B,
                    exponent=n_exp,
                )
            )

        return u_pred

    def _forward_strain(self, params):

        θ_B, θ_n = params

        # Solve for velocities
        u0 = self.solver.diagnostic_solve(
            velocity=self.u0,
            thickness=self.h,
            norm_rigidity=θ_B,
            norm_exponent=θ_n,
        )
        u1 = self.solver.diagnostic_solve(
            velocity=self.u1,
            thickness=self.h,
            norm_rigidity=θ_B,
            norm_exponent=θ_n,
        )

        # Compute strain rate tensors
        ε_0 = sym_grad(u0)
        ε_1 = sym_grad(u1)

        # Extract strain rate components
        e0_xx = firedrake.project(ε_0[0, 0], Q)
        e0_xy = firedrake.project(ε_0[0, 1], Q)
        e0_yy = firedrake.project(ε_0[1, 1], Q)
        e1_xx = firedrake.project(ε_1[0, 0], Q)
        e1_xy = firedrake.project(ε_1[0, 1], Q)
        e1_yy = firedrake.project(ε_1[1, 1], Q)

        return [u0, u1, e0_xx, e0_xy, e0_yy, e1_xx, e1_xy, e1_yy]

    def _data_nll(self, u_pred):
        misfits = []
        for k in range(self.Nt):
            du = u_pred[k] - self.u[k]
            misfits.append(
                (du[0] / self.σ_u[k][0])**2 + (du[1] / self.σ_u[k][1])**2
            )
        sum_misfit = sum(misfits)
        return 0.5 * self.nll_scale * sum_misfit * firedrake.dx

    def _data_nll_covariance(self, u_pred):
        """
        Correlated-Gaussian data NLL across epochs.

        Uses:
            self.C_temporal : (Nt, Nt) inverse temporal correlation/covariance
        """
        Nt = self.Nt
        Ct = self.C_temporal

        # Build whitened residual fields for each epoch and component
        rx = []
        ry = []
        for k in range(Nt):
            du = u_pred[k] - self.u[k]
            rx.append(du[0] / self.σ_u[k][0])  # U component, whitened
            ry.append(du[1] / self.σ_u[k][1])  # V component, whitened

        # Quadratic form r^T Ct r = sum_{i,j} Ct[i,j] r_i r_j
        integrand = 0
        for i in range(Nt):
            for j in range(Nt):
                cij = float(Ct[i, j])  # scalar coefficient
                integrand += cij * (rx[i] * rx[j] + ry[i] * ry[j])

        return 0.5 * self.nll_scale * integrand * firedrake.dx

    def data_strain_nll(self, predictions):

        # u0, u1 = pred_list
        u0, u1, e0_xx, e0_xy, e0_yy, e1_xx, e1_xy, e1_yy = predictions

        δu0 = u0 - self.u0
        δu1 = u1 - self.u1

        δe0_xx = e0_xx - self.e0_xx
        δe0_yy = e0_yy - self.e0_yy
        δe0_xy = e0_xy - self.e0_xy

        δe1_xx = e1_xx - self.e1_xx
        δe1_yy = e1_yy - self.e1_yy
        δe1_xy = e1_xy - self.e1_xy

        return (
            0.5
            / self.area
            * (
                (δu0[0] / self.σ_vel) ** 2
                + (δu0[1] / self.σ_vel) ** 2
                + (δu1[0] / self.σ_vel) ** 2
                + (δu1[1] / self.σ_vel) ** 2
                + (δe0_xx / self.σ_eps) ** 2
                + (δe0_yy / self.σ_eps) ** 2
                + (δe0_xy / self.σ_eps) ** 2
                + (δe1_xx / self.σ_eps) ** 2
                + (δe1_yy / self.σ_eps) ** 2
                + (δe1_xy / self.σ_eps) ** 2
            )
            * firedrake.dx
        )

    def reset_solver(self):
        for k in range(self.Nt):
            self.solvers[k]._diagnostic_solver.setup(
                velocity=self.u[k],
                thickness=self.h,
                rigidity=self.θ_B,
                exponent=self.θ_n,
            )


def viscosity_to_rigidity(log_nu, log_eps_eff, n_exp):
    a = firedrake.Constant(np.log(2.0))
    log_B = a + log_nu + (n_exp - 1) / n_exp * log_eps_eff
    return firedrake.exp(log_B)


def helmholtz_1d_nonuniform(s, y, alpha, bc="dirichlet"):
    """
    Solve (I - alpha^2 d2/ds2) u = y on a 1D grid s (possibly nonuniform).

    bc:
      - "neumann": u'(0)=u'(L)=0 (open boundary)
      - "dirichlet": u(0)=y(0), u(L)=y(L)
    """
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    assert s.ndim == 1 and y.ndim == 1 and s.size == y.size
    n = s.size
    if n < 3:
        return y.copy()

    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError("s must be strictly increasing.")

    a = np.zeros(n)  # sub-diagonal (a[i] couples u[i-1])
    b = np.ones(n)   # main diagonal
    c = np.zeros(n)  # super-diagonal (c[i] couples u[i+1])
    rhs = y.copy()

    # Interior: u'' approx on nonuniform grid
    # u''_i ≈ 2[(u_{i+1}-u_i)/ds_i - (u_i-u_{i-1})/ds_{i-1}] / (ds_{i-1}+ds_i)
    for i in range(1, n-1):
        dsm = ds[i-1]
        dsp = ds[i]
        denom = dsm + dsp
        A =  2.0 / (dsm * denom)          # coeff for u_{i-1}
        C =  2.0 / (dsp * denom)          # coeff for u_{i+1}
        B = -(A + C)                      # coeff for u_i
        # u_i - alpha^2 * (A u_{i-1} + B u_i + C u_{i+1}) = y_i
        a[i] = -alpha**2 * A
        b[i] = 1.0 - alpha**2 * B
        c[i] = -alpha**2 * C

    # Boundary rows
    if bc == "neumann":
        # Use ghost-point reflection: u_-1 = u_1, u_n = u_{n-2}
        # For left: u''_0 ≈ 2*(u_1 - u_0)/ds0^2
        ds0 = ds[0]
        c[0] = -alpha**2 * (2.0 / ds0**2)
        b[0] = 1.0 + alpha**2 * (2.0 / ds0**2)

        # For right: u''_{n-1} ≈ 2*(u_{n-2} - u_{n-1})/ds_{n-2}^2
        dsn = ds[-1]
        a[n-1] = -alpha**2 * (2.0 / dsn**2)
        b[n-1] = 1.0 + alpha**2 * (2.0 / dsn**2)

    elif bc == "dirichlet":
        # u(0)=y(0), u(L)=y(L)
        b[0] = 1.0; c[0] = 0.0; rhs[0] = y[0]
        b[n-1] = 1.0; a[n-1] = 0.0; rhs[n-1] = y[-1]
    else:
        raise ValueError("bc must be 'neumann' or 'dirichlet'.")

    # Thomas algorithm for tridiagonal solve
    # Forward elimination
    cp = c.copy()
    bp = b.copy()
    rp = rhs.copy()

    for i in range(1, n):
        m = a[i] / bp[i-1]
        bp[i] -= m * cp[i-1]
        rp[i] -= m * rp[i-1]

    # Back substitution
    u = np.zeros(n)
    u[-1] = rp[-1] / bp[-1]
    for i in range(n-2, -1, -1):
        u[i] = (rp[i] - cp[i] * u[i+1]) / bp[i]

    return u


def compute_path_length(x, y):
    """
    Computes path length along a path specified by X and Y points. Simply accumulates
    Euclidean distance between points.
    """
    # Compute distances
    dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    # Cumulative sum
    s = np.zeros_like(x)
    s[1:] = np.cumsum(dist)

    return s


# end of file
