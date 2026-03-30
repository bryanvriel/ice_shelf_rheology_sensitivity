#-*- coding: utf-8 -*-

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Tuple, List, Union

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

Vec = NDArray[np.float64]
Controls = Union[Function, Constant, List[Function]]


def _copy_control(control):
    try:
        return control.copy(deepcopy=True)
    except TypeError:
        return copy.deepcopy(control)


def _copy_controls(controls: Controls) -> Controls:
    if isinstance(controls, (Function, Constant)):
        return _copy_control(controls)
    return [_copy_control(ctrl) for ctrl in controls]


def _set_single_control(control, values: NDArray[np.float64]) -> None:
    if hasattr(control, "dat"):
        control.dat.data[:] = values.reshape(control.dat.data.shape)
    else:
        control.assign(np.asarray(values).reshape(-1)[0])


def _control_to_array(control) -> Vec:
    if hasattr(control, "dat"):
        return control.dat.data_ro.ravel().copy()
    return np.atleast_1d(float(control))


def _controls_to_array(controls: Controls) -> Vec:
    if isinstance(controls, (Function, Constant)):
        return _control_to_array(controls)
    return np.concatenate([_control_to_array(ctrl) for ctrl in controls])


def _set_controls_from_array(controls: Controls, x: Vec) -> None:
    x = np.asarray(x)
    if isinstance(controls, (Function, Constant)):
        _set_single_control(controls, x)
        return

    offset = 0
    for ctrl in controls:
        size = _control_to_array(ctrl).size
        _set_single_control(ctrl, x[offset : offset + size])
        offset += size


def _gradient_to_array(grad: Controls) -> Vec:
    if isinstance(grad, (Function, Constant)):
        return _control_to_array(grad)
    return np.concatenate([_control_to_array(g) for g in grad])


def _controls_to_control_objects(controls: Controls):
    if isinstance(controls, (Function, Constant)):
        return Control(controls)
    return [Control(ctrl) for ctrl in controls]


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

    def evaluate_with_state(self, z, x=None):
        data_nll, controls, state = self.data_objective.evaluate_with_state(z, x=x)
        prior_nll = float(np.asarray(self.field.prior_nll(z)))
        return data_nll + self.scale * prior_nll, controls, state

    def set_control_array(self, x: Vec) -> None:
        self.data_objective.set_control_array(x)

    def get_control_array(self) -> Vec:
        return self.data_objective.get_control_array()

    def copy_controls(self) -> Controls:
        return self.data_objective.copy_controls()


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
        self._controls = _copy_controls(problem.controls)

        # Build the reduced functional (recorded on init)
        self._build_reduced_functional()

    def _build_reduced_functional(self) -> None:
        """Record forward problem and create reduced functional."""
        with fresh_tape():
            with annotation_context():
                # Run forward problem
                self.state = self.problem.forward(self._controls)

                # Assemble objective
                self.J = firedrake.assemble(self.problem.data_nll(self.state))

                # Create controls and reduced functional
                self._control_objects = _controls_to_control_objects(self._controls)
                self.reduced_nll = ReducedFunctional(self.J, self._control_objects)
                self._tape = get_working_tape()

    def set_control_array(self, x: Vec) -> None:
        """Assign a flattened array to the current Firedrake controls."""
        _set_controls_from_array(self._controls, x)

    def get_control_array(self) -> Vec:
        """Return the current Firedrake controls as a flat array."""
        return _controls_to_array(self._controls)

    def copy_controls(self) -> Controls:
        """Return deep copies of the current Firedrake controls."""
        return _copy_controls(self._controls)

    def copy_state(self) -> Controls:
        """Return deep copies of the current forward state."""
        checkpoints = []
        for state_i in self.state:
            checkpoint = getattr(getattr(state_i, "block_variable", None), "checkpoint", None)
            checkpoints.append(checkpoint if checkpoint is not None else state_i)
        return _copy_controls(checkpoints)

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
        self.set_control_array(x)

        # Evaluate loss and gradient w.r.t. physical controls
        loss = float(self.reduced_nll(self._controls))
        grad_x = _gradient_to_array(self.reduced_nll.derivative())

        # Chain rule: gradient w.r.t. latent parameters
        grad_z = self.field.vjp(z, grad_x)

        return (self.scale * loss, self.scale * grad_z)

    def loss(self, z: Vec, **kwargs) -> float:
        """Evaluate objective at latent parameters z (no gradient)."""
        x = self.field(z)
        self.set_control_array(x)
        return self.scale * float(self.reduced_nll(self._controls))

    def evaluate_with_state(
        self,
        z: Vec,
        x: Vec | None = None,
    ) -> Tuple[float, Controls, Controls]:
        """Evaluate objective and return the current controls and forward state.

        This is intended for prediction/export paths that need both the loss and
        the forward solution without paying for a second solve.
        """
        if x is None:
            x = self.field(z)
        self.set_control_array(x)
        loss = self.scale * float(self.reduced_nll(self._controls))
        return loss, self.copy_controls(), self.copy_state()

    def gradient(self, z: Vec, **kwargs) -> Vec:
        """Evaluate gradient at latent parameters z."""
        _, grad = self(z, **kwargs)
        return grad

    def hvp(self, z: Vec, v: Vec) -> Vec:
        """Hessian-vector product via pyadjoint's built-in Hessian."""
        # Map to physical space
        x = self.field(z)
        self.set_control_array(x)

        # Create direction in control space
        dx = self.field.jvp(z, v)

        # Build direction as Function(s)
        if isinstance(self._controls, (Function, Constant)):
            h = _copy_control(self._controls)
            _set_single_control(h, dx)
            directions = h
        else:
            directions = []
            offset = 0
            for ctrl in self._controls:
                h_i = _copy_control(ctrl)
                size = _control_to_array(ctrl).size
                _set_single_control(h_i, dx[offset : offset + size])
                directions.append(h_i)
                offset += size

        # Evaluate at current controls (needed to set tape state)
        _ = self.reduced_nll(self._controls)

        # Use pyadjoint's Hessian action
        Hx_dx = self.reduced_nll.hessian(directions)

        # Convert to array
        hvp_x_arr = _gradient_to_array(Hx_dx)

        # Chain rule back to latent space
        hvp_z = self.field.vjp(z, hvp_x_arr)

        # Curvature term for nonlinear field mappings
        if hasattr(self.field, 'hvp'):
            grad_x = _gradient_to_array(self.reduced_nll.derivative())
            hvp_z += self.field.hvp(z, v, grad_x)

        return self.scale * hvp_z

    def hvp_grad_diff(self, z: Vec, v: Vec) -> Vec:
        return self.hvp(z, v)
# end of file
