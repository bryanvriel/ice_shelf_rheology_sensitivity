#-*- coding: utf-8 -*-

from functools import partial
import numpy as np
import h5py
import os
from typing import Optional, Tuple

import firedrake
from firedrake import assemble
import icepack
import rasterio

from cryogen import interpolate

from . import models
from .utilities import coordinate_values


def make_mesh(meshfile=None, comm=None, mesh_span=None, mesh_spacing=None):
    if meshfile:
        if comm is not None:
            return firedrake.Mesh(meshfile, comm=comm)
        return firedrake.Mesh(meshfile)

    if mesh_span is None or mesh_spacing is None:
        raise ValueError(
            "make_mesh() requires either `meshfile` or both `mesh_span` and "
            "`mesh_spacing`."
        )

    N = int(mesh_span / mesh_spacing)
    if comm is not None:
        return firedrake.IntervalMesh(N, mesh_span, comm=comm)
    return firedrake.IntervalMesh(N, mesh_span)


class IceShelfProblem:

    def __init__(
        self,
        cfg,
        mesh,
        dirichlet_ids=None,
        side_wall_ids=None,
        degree=2,
        side_friction=0.0,
        nll_scale=1.0,
        default_vel_error=10.0,
        solver_type="petsc",
        n_norm="tanh",
        B_norm="log",
        use_strain=False,
        fixed_exponent=False,
        temporal_cov=False,
        infer_boundary_velocity=False,
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
        dirichlet_ids = [1] if dirichlet_ids is None else list(dirichlet_ids)
        side_wall_ids = [] if side_wall_ids is None else list(side_wall_ids)
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
        self.u_obs = []
        self.u_bc_ref = []
        self.u_bc = []
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

        self.temporal_cov = temporal_cov

        self.fixed_exponent = fixed_exponent
        self.infer_boundary_velocity = infer_boundary_velocity

        self.boundary_node_indices = None
        self.boundary_coords = None
        self.boundary_s = None
        self.boundary_length = None

        self._obs_init = False
        self._solver_init = False

    @property
    def u(self):
        """Backward-compatible alias for observed velocity fields."""
        return self.u_obs

    @u.setter
    def u(self, value):
        self.u_obs = value

    @property
    def controls(self):
        controls = [self.θ_B]
        if not self.fixed_exponent:
            controls.append(self.θ_n)
        if self.infer_boundary_velocity:
            controls.extend(self.u_bc)
        return tuple(controls)

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
        self.u_obs = []
        self.u_bc_ref = []
        self.u_bc = []
        self.σ_u = []
        self.e_xx = []
        self.e_xy = []
        self.e_yy = []

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
            self.u_obs.append(u_k)
            self.u_bc_ref.append(u_k.copy(deepcopy=True))
            self.u_bc.append(u_k.copy(deepcopy=True))
            self.σ_u.append(σ_k)
            #self.e_xx.append(e_xx_k)
            #self.e_xy.append(e_xy_k)
            #self.e_yy.append(e_yy_k)

        self._obs_init = True

    def _init_boundary_geometry(self):
        if self.boundary_node_indices is not None:
            return

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

        self.boundary_node_indices = boundary_node_indices
        self.boundary_coords = edge_coords
        self.boundary_s = compute_path_length(edge_coords[:, 0], edge_coords[:, 1])
        self.boundary_length = (
            self.boundary_s[-1]
            + np.sqrt(((edge_coords[0] - edge_coords[-1])**2).sum())
        )

    def smooth_boundaries(self, alpha_mag=3.0e3, alpha_theta=4.0e3):
        self._init_boundary_geometry()
        boundary_node_indices = self.boundary_node_indices
        s_edge = self.boundary_s
        boundary_length = self.boundary_length

        def _smooth(ref_array, alpha):
            bvals = ref_array
            smooth_vals = helmholtz_1d_nonuniform(
                s_edge,
                bvals,
                alpha,
                bc="periodic",
                period=boundary_length,
            )
            return bvals, smooth_vals

        # Smooth the boundary-condition reference, leaving observations untouched.
        for k in range(self.Nt):
            self.u_bc_ref[k].assign(self.u_obs[k])
            self.u_bc[k].assign(self.u_obs[k])

            u = self.u_obs[k].dat.data_ro[boundary_node_indices, :]
            mag = np.linalg.norm(u, axis=1)
            theta = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
            mag, mag_smooth = _smooth(mag, alpha_mag)
            theta, theta_smooth = _smooth(theta, alpha_theta)
            u = mag_smooth * np.cos(np.radians(theta_smooth))
            v = mag_smooth * np.sin(np.radians(theta_smooth))
            u_smooth = np.column_stack((u, v))
            self.u_bc_ref[k].dat.data[boundary_node_indices] = u_smooth
            self.u_bc[k].dat.data[boundary_node_indices] = u_smooth

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

    def _check_forward_ready(self):
        if not self._obs_init:
            raise RuntimeError("Observations must be loaded before running the forward model.")
        if not self._solver_init:
            raise RuntimeError("Solvers must be initialized before running the forward model.")

    def split_controls(self, params):
        params = tuple(params)

        if self.fixed_exponent:
            if len(params) < 1:
                raise ValueError("Expected at least one control for fixed-exponent mode.")
            θ_B = params[0]
            θ_n = self.θ_n
            offset = 1
        else:
            if len(params) < 2:
                raise ValueError("Expected at least two controls for joint rheology mode.")
            θ_B, θ_n = params[:2]
            offset = 2

        if self.infer_boundary_velocity:
            bc_controls = list(params[offset:])
            if len(bc_controls) != self.Nt:
                raise ValueError(
                    f"Expected {self.Nt} boundary-velocity controls, got {len(bc_controls)}."
                )
        else:
            bc_controls = self.u_bc

        return θ_B, θ_n, bc_controls

    def forward(self, params):
        self._check_forward_ready()
        if self.fixed_exponent:
            return self._forward_fixed_exponent(params)
        return self._forward_joint(params)

    def data_nll(self, u_pred):
        if self.temporal_cov:
            return self._data_nll_covariance(u_pred)
        return self._data_nll_independent(u_pred)

    def _forward_joint(self, params):

        # Unpack and un-normalize parameters
        θ_B, θ_n, bc_controls = self.split_controls(params)
        n_exp = assemble(firedrake.interpolate(self._inv_norm_n(θ_n), self.Q))
        B = assemble(firedrake.interpolate(self._inv_norm_B(θ_B), self.Q))

        # Loop over epochs to compute velocities
        u_pred = []
        for k in range(self.Nt):
            u_pred.append(
                self.solvers[k].diagnostic_solve(
                    velocity=bc_controls[k],
                    thickness=self.h,
                    rigidity=B,
                    flow_law_exponent=n_exp,
                )
            )

        return u_pred

    def _forward_fixed_exponent(self, params):
        θ_B, _, bc_controls = self.split_controls(params)
        B = assemble(firedrake.interpolate(self._inv_norm_B(θ_B), self.Q))
        n_exp = assemble(firedrake.interpolate(self._inv_norm_n(self.θ_n), self.Q))

        # Loop over epochs to compute velocities
        u_pred = []
        for k in range(self.Nt):
            u_pred.append(
                self.solvers[k].diagnostic_solve(
                    velocity=bc_controls[k],
                    thickness=self.h,
                    rigidity=B,
                    flow_law_exponent=n_exp,
                )
            )

        return u_pred

    def _data_nll_independent(self, u_pred):
        misfits = []
        for k in range(self.Nt):
            du = u_pred[k] - self.u_obs[k]
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
            du = u_pred[k] - self.u_obs[k]
            rx.append(du[0] / self.σ_u[k][0])  # U component, whitened
            ry.append(du[1] / self.σ_u[k][1])  # V component, whitened

        # Quadratic form r^T Ct r = sum_{i,j} Ct[i,j] r_i r_j
        integrand = 0
        for i in range(Nt):
            for j in range(Nt):
                cij = float(Ct[i, j])  # scalar coefficient
                integrand += cij * (rx[i] * rx[j] + ry[i] * ry[j])

        return 0.5 * self.nll_scale * integrand * firedrake.dx

    def reset_solver(self):
        for k in range(self.Nt):
            self.solvers[k]._diagnostic_solver.setup(
                velocity=self.u_bc[k],
                thickness=self.h,
                rigidity=self.θ_B,
                flow_law_exponent=self.θ_n,
            )


def viscosity_to_rigidity(log_nu, log_eps_eff, n_exp):
    a = firedrake.Constant(np.log(2.0))
    log_B = a + log_nu + (n_exp - 1) / n_exp * log_eps_eff
    return firedrake.exp(log_B)


def helmholtz_1d_nonuniform(s, y, alpha, bc="dirichlet", period=None):
    """
    Solve (I - alpha^2 d2/ds2) u = y on a 1D grid s (possibly nonuniform).

    bc:
      - "periodic": u, u' periodic on a closed boundary with total length `period`
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

    if bc == "periodic":
        if period is None:
            raise ValueError("`period` must be provided when bc='periodic'.")
        period = float(period)
        closing_gap = period - (s[-1] - s[0])
        if closing_gap <= 0:
            raise ValueError(
                "`period` must exceed the open path length when bc='periodic'."
            )

        ds_next = np.empty(n, dtype=float)
        ds_next[:-1] = ds
        ds_next[-1] = closing_gap
        ds_prev = np.roll(ds_next, 1)

        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            dsm = ds_prev[i]
            dsp = ds_next[i]
            denom = dsm + dsp
            A_prev = 2.0 / (dsm * denom)
            A_next = 2.0 / (dsp * denom)
            A_self = -(A_prev + A_next)

            A[i, i] = 1.0 - alpha**2 * A_self
            A[i, (i - 1) % n] = -alpha**2 * A_prev
            A[i, (i + 1) % n] = -alpha**2 * A_next

        return np.linalg.solve(A, y)

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
