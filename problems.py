#-*- coding: utf-8 -*-

import numpy as np
import h5py
import os
from typing import Optional, Tuple

import firedrake
from firedrake import inner
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
        norm="log",
        fixed_exponent=False,
    ):

        # Initialize functional spaces
        self.mesh = mesh
        self.Q = firedrake.FunctionSpace(self.mesh, family="CG", degree=degree)
        self.R = firedrake.FunctionSpace(self.mesh, family="R", degree=0)
        self.coords = coordinate_values(self.Q)
        print(self.coords.shape)
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
        area = firedrake.assemble(firedrake.Constant(1.0) * firedrake.dx(self.mesh))
        self.area = area
        self.nll_scale = firedrake.Constant(nll_scale / area)

        # Mesh metadata for assembling
        quad_degree = 3 * (degree - 1) + 2 * degree
        self.mesh_metadata = {"quadrature_degree": quad_degree}
        self._dirichlet_ids = dirichlet_ids
        self._side_wall_ids = side_wall_ids

        # Initialize flow model
        if norm == "log":
            self.model = models.CustomIceShelf(viscosity=models.normalized_rheology_log)
            self.unnormalize_rheology = models.unnormalize_rheology_log
            self.norm_type = "log"
        elif norm == "tanh":
            self.model = models.CustomIceShelf(viscosity=models.normalized_rheology)
            self.unnormalize_rheology = models.unnormalize_rheology
            self.norm_type = "tanh"
        else:
            raise ValueError

        # Initialize flow solvers
        self.solver, self.solver_opts = models.create_solver(
            self.model,
            solver_type,
            dirichlet_ids=dirichlet_ids,
            side_wall_ids=side_wall_ids,
        )

        # Static flow solver arguments (these flow through pyadoint)
        # These have to be R-space functions rather than Constants in order
        # to avoid 'Constant requires a domain to work with pyadjoint' errors
        self.solver_kwargs = {
            "B_min": const_to_R_fn(cfg.model.B_min, self.R),
            "B_max": const_to_R_fn(cfg.model.B_max, self.R),
            "n_min": const_to_R_fn(cfg.model.n_min, self.R),
            "n_max": const_to_R_fn(cfg.model.n_max, self.R),
        }

        # Initialize constants
        self.Cs = firedrake.Constant(side_friction)
        self.d_pert = firedrake.Constant(0.0)

        # Initialize function spaces for observations (potentially time-dependent)
        # These need to be loaded prior to any forward solves
        self.h = firedrake.Function(self.Q)
        self.u = []
        self.σ_u = []
        self.e_xx = []
        self.e_xy = []
        self.e_yy = []
        self.σ_eps = firedrake.Constant(0.001)
        self.σ_u_default = default_vel_error

        # Placeholders for rheology fields (only used when fixed_exponent = True)
        # Also used for self.reset_solver()
        self.θ_B = firedrake.Function(self.Q)
        self.θ_n = firedrake.Function(self.Q)

        # List of control variables
        # Not used here, but as an interface to FiredrakeObjective
        self.fixed_exponent = fixed_exponent
        if fixed_exponent:
            self.forward = self._forward_rigidity
            self.regularization = self._regularization_rigidity
        else:
            self.forward = self._forward
            self.regularization = self._regularization

        self._obs_init = False

    @property
    def controls(self):
        if self.fixed_exponent:
            return self.θ_B
        else:
            return [self.θ_B, self.θ_n]

    @controls.setter
    def controls(self, value):
        raise ValueError("Cannot set controls.")

    def load_observations(self, data_file, time_ind: Optional[list] = None):
        """Load observation fields defined on a structured grid.

        Parameters
        ----------
        data_file:
            Path to an HDF5 file containing the observation fields.
        time_ind:
            List of time indices to load. If ``None``, all epochs are used.
        """

        mesh_coords = (self.coords[:, 0], self.coords[:, 1])

        def _interp_to_mesh(array: np.ndarray,
                            grid_coords: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            """Simple interpolation of ``array`` onto the mesh coordinates."""
            return interpolate.interp(array, grid_coords, mesh_coords)

        def _interp_field(field_array: np.ndarray,
                          grid_coords: Tuple[np.ndarray, np.ndarray],
                          target_fn: firedrake.Function) -> None:
            """Interpolate ``field_array`` to ``target_fn``'s mesh.

            Parameters
            ----------
            field_array:
                ``(Ny, Nx)`` array for scalar targets or ``(2, Ny, Nx)`` for
                two-component vector targets.
            grid_coords:
                Tuple of ``(x, y)`` coordinate arrays defining ``field_array``.
            target_fn:
                Firedrake function whose ``.dat.data`` will be populated with
                the interpolated values. Modified in place.
            """
            if field_array.ndim == 3:
                interpolated = np.column_stack(
                    [_interp_to_mesh(field_array[i], grid_coords)
                     for i in range(field_array.shape[0])]
                )
            elif field_array.ndim == 2:
                interpolated = _interp_to_mesh(field_array, grid_coords)
            else:
                raise ValueError('Field array has unsupported shape.')

            target_fn.dat.data[:] = interpolated

        with h5py.File(data_file, "r") as fid:
            h, u, e_xx, e_xy, e_yy = [
                fid[key][()] for key in
                ("h", "u", "e_xx", "e_xy", "e_yy")
            ]
            grid_coords = (fid["x"][()], fid["y"][()])
        
            if time_ind is None:
                self.Nt = u.shape[0]
                time_ind = range(self.Nt)
            else:
                self.Nt = len(time_ind)

        # Thickness is time-averaged
        self.h.dat.data[:] = _interp_to_mesh(h, grid_coords)

        # Interpolate the time epoch data
        for k in time_ind:

            # Local function spaces
            u_k = firedrake.Function(self.V)
            σ_k = firedrake.Function(self.V)
            e_xx_k = firedrake.Function(self.Q)
            e_xy_k = firedrake.Function(self.Q)
            e_yy_k = firedrake.Function(self.Q)

            # Interpolate velocity components and errors
            _interp_field(u[k], grid_coords, u_k)
            #_interp_field(u_sigma[k], grid_coords, σ_k)
            σ_k.dat.data[:] = self.σ_u_default

            # Interpolate strain components
            _interp_field(e_xx[k], grid_coords, e_xx_k)
            _interp_field(e_xy[k], grid_coords, e_xy_k)
            _interp_field(e_yy[k], grid_coords, e_yy_k)

            # Append to lists
            self.u.append(u_k)
            self.σ_u.append(σ_k)
            self.e_xx.append(e_xx_k)
            self.e_xy.append(e_xy_k)
            self.e_yy.append(e_yy_k)

        self._obs_init = True

    def load_rheology(self, dirname):
        print('Loading true rheology')
        r_n = rasterio.open(os.path.join(dirname, "n_exp.tif"))
        n_fn = icepack.interpolate(r_n, self.Q)
        tanh_params = cryogen.compute_tanh_norm_params(
            self.solver_kwargs["n_min"].dat.data_ro,
            self.solver_kwargs["n_max"].dat.data_ro,
        )
        n_np = n_fn.dat.data_ro.copy()
        if self.norm_type == "tanh":
            theta_n_vals = cryogen.tanh_forward(n_np, tanh_params)
        elif self.norm_type == "log":
            theta_n_vals = np.log(n_np)
        self.θ_n.dat.data[:] = theta_n_vals

    def residual(self, u, params):
        """
        Reconstructs and returns the PDE residual F(u, m) = 0 for each epoch.

        Args:
            u (firedrake.Function): The state variable (velocity).
            params (list): The control variable(s).
        """
        # Unpack parameters
        θ_B, θ_n = params

        kwargs = dict(
            thickness=self.h,
            norm_rigidity=θ_B,
            norm_exponent=θ_n,
            **self.solver_kwargs,
        )

        # Retrieve the mesh from velocity from first epoch
        mesh = u[0].function_space().mesh()

        # --- Replicate the logic from PETScSolver.setup ---

        # Determine boundary IDs
        boundary_ids = mesh.exterior_facets.unique_markers
        ice_front_ids_comp = set(self._dirichlet_ids + self._side_wall_ids)
        ice_front_ids = list(set(boundary_ids) - ice_front_ids_comp)

        id_kwargs = {
            "side_wall_ids": self._side_wall_ids,
            "ice_front_ids": ice_front_ids,
        }
       
        res = [] 
        for k in range(self.Nt):
        
            # Create the 'action' functional
            action = self.model.action(velocity=u[k], **kwargs, **id_kwargs)

            # Compute the derivative to get the residual
            F = firedrake.derivative(action, u[k])
            res.append(F)
        
        return res

    def _forward(self, params):
        θ_B, θ_n = params
        u_pred = []
        for k in range(self.Nt):
            u_pred.append(
                self.solver.diagnostic_solve(
                    velocity=self.u[k],
                    thickness=self.h,
                    norm_rigidity=θ_B,
                    norm_exponent=θ_n,
                    **self.solver_kwargs,
                )
            )

        return u_pred

    def _forward_rigidity(self, θ_B):
        u_pred = self.solver.diagnostic_solve(
            velocity=self.u[0],
            thickness=self.h,
            norm_rigidity=θ_B,
            norm_exponent=self.θ_n,
            **self.solver_kwargs,
        )
        return [
            u_pred,
        ]

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
    
    def data_nll(self, u_pred):
        misfits = []
        for k in range(self.Nt):
            du = u_pred[k] - self.u[k]
            misfits.append(
                (du[0] / self.σ_u[k][0])**2 + (du[1] / self.σ_u[k][1])**2
            )
        sum_misfit = sum(misfits)
        return 0.5 * self.nll_scale * sum_misfit * firedrake.dx

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

    def _regularization(self, params):
        θ_B, θ_n = params

        # L_B = firedrake.Constant(5.0e3); L_n = 2 * L_B
        L_B = firedrake.Constant(2.5e3)
        L_n = 4 * L_B
        # L_B = firedrake.Constant(2.0e3); L_n = 3 * L_B
        # L_B = firedrake.Constant(5.0e3); L_n = 2 * L_B

        R_B = 0.5 / self.area * L_B**2 * inner(grad(θ_B), grad(θ_B)) * firedrake.dx
        R_n = 0.5 / self.area * L_n**2 * inner(grad(θ_n), grad(θ_n)) * firedrake.dx
        return R_B + R_n

    def _regularization_rigidity(self, θ_B):
        L_B = firedrake.Constant(30e3)
        return 0.5 / self.area * L_B**2 * inner(grad(θ_B), grad(θ_B)) * firedrake.dx

    def reset_solver(self):
        self.solver._diagnostic_solver.setup(
            velocity=self.u[0],
            thickness=self.h,
            norm_rigidity=self.θ_B,
            norm_exponent=self.θ_n,
            **self.solver_kwargs,
        )


# end of file
