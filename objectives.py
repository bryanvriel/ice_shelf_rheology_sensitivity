#-*- coding: utf-8 -*-

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Callable, Tuple, List, Union

import numpy as np
from numpy.typing import NDArray

import firedrake
from firedrake import Function, Constant
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    get_working_tape,
    set_working_tape,
    Tape,
)
import sys

Vec = NDArray[np.float64]
Controls = Union[Function, Constant, List[Function]]


@contextmanager
def annotation_context():
    """Context manager for AD annotation."""
    continue_annotation()
    try:
        yield
    finally:
        pause_annotation()


@contextmanager
def fresh_tape():
    """Context manager that provides a fresh AD tape."""
    old_tape = get_working_tape()
    set_working_tape(Tape())
    try:
        yield get_working_tape()
    finally:
        set_working_tape(old_tape)


class PosteriorObjective:
    """
    Objective class where optimization is performed in latent space. Creates
    instance of DataObjective and combines with prior information in order to
    compute the log probability of the posterior.
    """

    def __init__(self, field, problem, scale=1.0):
        continue_annotation()

        self.field = field
        self.data_objective = DataObjective(field, problem, scale=scale)
        self.scale = scale

    def __call__(self, z, **kwargs):
        """Evaluate objective function"""

        # Evaluate data log likelihood
        data_nll, grad_data = self.data_objective(z, **kwargs)

        # Evaluate prior NLL and its gradient
        prior_nll, grad_prior = self.field.prior_nll_and_grad(z)

        # Total objective and gradient (data nll is already scaled)
        obj = data_nll + self.scale * prior_nll
        obj_grad = grad_data + self.scale * grad_prior

        return (obj, obj_grad)

    def hvp_finite_diff(self, z, v, eps=1e-6):
        """
        Compute Hessian-vector product H @ v directly in latent parameter space
        using finite differences.
        """
        # Get gradient at z + eps * v
        _, grad_plus = self(z + eps * v)

        # Get gradient at z - eps * v
        _, grad_minus = self(z - eps * v)

        # Finite difference approximation
        hvp = (grad_plus - grad_minus) / (2 * eps)

        return hvp

    def hvp(self, z, v):
        """
        Compute Hessian-vector product with adjoints.
        """
        hvp_data = self.data_objective.hvp(z, v)
        hvp_prior = self.scale * (self.field.S @ v)
        return hvp_data + hvp_prior

    def loss(self, z):
        data_nll = self.data_objective.loss(z)
        prior_nll = self.field.prior_nll(z)
        return data_nll + self.scale * prior_nll

    def update_controls(self, x):
        self.data_objective.update_controls(x)

    @property
    def controls(self):
        return self.data_objective.controls

    @controls.setter
    def controls(self, value):
        self.data_objective.controls = value


class DataObjective:
    """
    Objective class where optimization is performed in latent space, but
    only returns data likelihood portion of cost function.
    """

    def __init__(self, field, problem, scale=1.0):
        self.field = field
        self.problem = problem
        self.scale = scale

        # Store original controls as template
        controls = problem.controls
        if isinstance(controls, (Function, Constant)):
            self._controls = controls.copy(deepcopy=True)
        else:
            self._controls = [ctrl.copy(deepcopy=True) for ctrl in controls]

        # Build the reduced functional (recorded on init)
        self._build_reduced_functional()

    def _build_reduced_functional(self) -> None:
        """Record forward problem and create reduced functional."""
        with fresh_tape():
            with annotation_context():
                # Run forward problem
                self.state = self.problem.forward(self._controls)

                # Assemble objective
                self.J = sum(
                    firedrake.assemble(E(self.state))
                    for E in [self.problem.data_nll]
                )

                # Create controls and reduced functional
                if isinstance(self._controls, (Function, Constant)):
                    self._control_objects = Control(self._controls)
                else:
                    self._control_objects = [Control(ctrl) for ctrl in self._controls]

                self.reduced_nll = ReducedFunctional(self.J, self._control_objects)
                self._tape = get_working_tape()

    def _controls_to_array(self) -> Vec:
        """Extract control values as a flat array."""
        if isinstance(self._controls, (Function, Constant)):
            return self._controls.dat.data_ro.ravel().copy()
        else:
            return np.concatenate([ctrl.dat.data_ro.ravel() for ctrl in self._controls])

    def _array_to_controls(self, x: Vec) -> None:
        """Update control values from a flat array."""
        if isinstance(self._controls, (Function, Constant)):
            self._controls.dat.data[:] = x.reshape(self._controls.dat.data.shape)
        else:
            offset = 0
            for ctrl in self._controls:
                size = ctrl.dat.data.size
                ctrl.dat.data[:] = x[offset:offset + size].reshape(ctrl.dat.data.shape)
                offset += size

    def _gradient_to_array(self, grad: Controls) -> Vec:
        """Convert gradient (Function or list) to flat array."""
        if isinstance(grad, (Function, Constant)):
            return grad.dat.data_ro.ravel().copy()
        else:
            return np.concatenate([g.dat.data_ro.ravel() for g in grad])

    def __call__(self, z: Vec, **kwargs) -> Tuple[float, Vec]:
        """Evaluate objective and gradient at latent parameters z.

        Parameters
        ----------
        z : Vec
            Latent parameters.

        Returns
        -------
        Tuple[float, Vec]
            (scaled loss, scaled gradient w.r.t. z)
        """
        # Map latent to physical
        x = self.field(z)
        self._array_to_controls(x)

        # Evaluate loss and gradient w.r.t. physical controls
        loss = float(self.reduced_nll(self._controls))
        grad_x = self._gradient_to_array(self.reduced_nll.derivative())

        # Chain rule: gradient w.r.t. latent parameters
        grad_z = self.field.vjp(z, grad_x)

        return (self.scale * loss, self.scale * grad_z)

    def loss(self, z: Vec, **kwargs) -> float:
        """Evaluate objective at latent parameters z (no gradient)."""
        x = self.field(z)
        self._array_to_controls(x)
        return self.scale * float(self.reduced_nll(self._controls))

    def gradient(self, z: Vec, **kwargs) -> Vec:
        """Evaluate gradient at latent parameters z."""
        _, grad = self(z, **kwargs)
        return grad

    def hvp(self, z: Vec, v: Vec) -> Vec:
        """Hessian-vector product via pyadjoint's built-in Hessian."""
        # Map to physical space
        x = self.field(z)
        self._array_to_controls(x)

        # Create direction in control space
        dx = self.field.jvp(z, v)

        # Build direction as Function(s)
        if isinstance(self._controls, (Function, Constant)):
            h = self._controls.copy(deepcopy=True)
            h.dat.data[:] = dx.reshape(h.dat.data.shape)
            directions = h
        else:
            directions = []
            offset = 0
            for ctrl in self._controls:
                h_i = ctrl.copy(deepcopy=True)
                size = ctrl.dat.data.size
                h_i.dat.data[:] = dx[offset:offset + size].reshape(h_i.dat.data.shape)
                directions.append(h_i)
                offset += size

        # Evaluate at current controls (needed to set tape state)
        _ = self.reduced_nll(self._controls)

        # Use pyadjoint's Hessian action
        Hx_dx = self.reduced_nll.hessian(directions)

        # Convert to array
        if isinstance(Hx_dx, (Function, Constant)):
            hvp_x_arr = Hx_dx.dat.data_ro.ravel()
        else:
            hvp_x_arr = np.concatenate([h.dat.data_ro.ravel() for h in Hx_dx])

        # Chain rule back to latent space
        hvp_z = self.field.vjp(z, hvp_x_arr)

        # Curvature term for nonlinear field mappings
        if hasattr(self.field, 'hvp'):
            grad_x = self._gradient_to_array(self.reduced_nll.derivative())
            hvp_z += self.field.hvp(z, v, grad_x)

        return self.scale * hvp_z

    def hvp_grad_diff(self, z: Vec, v: Vec) -> Vec:
        return self.hvp(z, v)

    @property
    def controls(self):
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            return self._controls.copy(deepcopy=True)
        else:
            controls = []
            for control in self._controls:
                controls.append(control.copy(deepcopy=True))
            return controls

    @controls.setter
    def controls(self, value: Vec) -> None:
        self._array_to_controls(value)


class PosteriorObjectiveMesh:
    """
    Objective class where optimization is performed in mesh-space (at the nodes).
    """

    def __init__(self, problem):
        continue_annotation()

        controls = problem.controls

        # Initialize the objective components
        if isinstance(controls, (firedrake.Function, firedrake.Constant)):
            self._controls = controls.copy(deepcopy=True)
            self.is_multi_control = False
        else:
            self._controls = [field.copy(deepcopy=True) for field in controls]
            self.is_multi_control = True
        self._state = problem.forward(self._controls)

        # Form data objective
        E = firedrake.assemble(problem.data_nll(self._state))
        R = firedrake.assemble(problem.regularization(self._controls))
        self.J = E + R

        # Create the reduced functional and store control objects
        if not self.is_multi_control:
            self._control_objects = Control(self._controls)
        else:
            self._control_objects = [Control(field) for field in self._controls]
        self.reduced_nll = ReducedFunctional(self.J, self._control_objects)

        pause_annotation()

    def __call__(self, x):
        """Evaluate objective function"""
        continue_annotation()

        # Reconstruct field controls
        self.update_controls(x)

        # Evaluate posterior NLL
        nll = float(self.reduced_nll(self._controls))

        # Evaluate gradient of NLL w.r.t. field controls
        grad = self.reduced_nll.derivative()
        if isinstance(grad, (firedrake.Function, firedrake.Constant)):
            grad_nll = np.array(grad.vector()[:])
        else:
            grad_nll = np.concatenate([g.vector()[:] for g in grad])
        pause_annotation()

        return (nll, grad_nll)

    def hessian_vector_product(self, x, v, eps=1e-6):
        """
        Compute Hessian-vector product H @ v directly in latent parameter space
        using finite differences.
        """
        # Get gradient at x + eps * v
        _, grad_plus = self(x + eps * v)

        # Get gradient at x - eps * v
        _, grad_minus = self(x - eps * v)

        # Finite difference approximation
        hvp = (grad_plus - grad_minus) / (2 * eps)

        return hvp

    def hessian_vector_product_5pt(self, z, v, eps=1e-6):
        """
        Compute Hessian-vector product H @ v directly in latent parameter space
        using finite differences, with a five-point stencil for O(ε**4) accuracy.
        """
        # Evaluate gradients at the five-stencil points
        εv = eps * v
        _, g_p2 = self(z + 2 * εv)
        _, g_p1 = self(z + εv)
        _, g_m1 = self(z - εv)
        _, g_m2 = self(z - 2 * εv)

        # Five-point central difference formula
        hvp = (-g_p2 + 8.0 * g_p1 - 8.0 * g_m1 + g_m2) / (12.0 * eps)

        return hvp

    def loss(self, x):
        continue_annotation()

        # Reconstruct field controls
        self.update_controls(x)

        # Evaluate posterior NLL
        return float(self.reduced_nll(self._controls))

    def update_controls(self, x):
        """
        Assign numpy array values to controls.
        """
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            self._controls.vector()[:] = x
        else:
            offset = 0
            for control in self._controls:
                size = len(control.vector()[:])
                control.vector()[:] = x[offset : offset + size]
                offset += size

    def array_to_controls(self, x):
        """Assigns values in numpy array to new control objects."""
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            controls = self._controls.copy(deepcopy=True)
            controls.vector()[:] = x
            return controls
        else:
            offset = 0
            controls = []
            for _control in self._controls:
                control = _control.copy(deepcopy=True)
                size = len(control.vector()[:])
                control.vector()[:] = x[offset : offset + size]
                offset += size
                controls.append(control)
            return controls

    def controls_to_array(self):
        """Converts controls to numpy arrays."""
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            return np.array(self._controls.vector()[:])
        else:
            return np.concatenate([ctrl.vector()[:] for ctrl in self._controls])

    @property
    def controls(self):
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            return self._controls.copy(deepcopy=True)
        else:
            controls = []
            for control in self._controls:
                controls.append(control.copy(deepcopy=True))
            return controls

    @controls.setter
    def controls(self, value):
        raise ValueError("Cannot set controls.")


# end of file
