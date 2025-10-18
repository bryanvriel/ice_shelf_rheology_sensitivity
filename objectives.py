#-*- coding: utf-8 -*-

import copy
import numpy as np
import firedrake
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
)
import sys


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
    
    def hessian_vector_product(self, z, v, eps=1e-6):
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

    def loss(self, z):
        data_nll = self.data_objective.loss(z)
        prior_nll = self.field.prior_nll(z)
        return data_nll + self.scale * prior_nll

    def update_controls(self, x):
        self.data_objective.update_controls(x)

    @property
    def controls(self):
        return self.data_objective.controls


class DataObjective:
    """
    Objective class where optimization is performed in latent space, but
    only returns data likelihood portion of cost function.
    """

    def __init__(self, field, problem, scale=1.0):
        continue_annotation()

        self.field = field
        self.problem = problem
        controls = problem.controls
        self.scale = scale

        # Initialize the objective components
        if isinstance(controls, (firedrake.Function, firedrake.Constant)):
            self._controls = copy.deepcopy(controls)
        else:
            self._controls = [copy.deepcopy(field) for field in controls]
        self.state = problem.forward(self._controls)

        # Form data objective
        self.J = sum(
            firedrake.assemble(E(self.state))
            for E in [
                problem.data_nll,
            ]
        )

        # Create the reduced functional and store control objects
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            self._control_objects = Control(self._controls)
        else:
            self._control_objects = [Control(field) for field in self._controls]

        self.reduced_nll = ReducedFunctional(self.J, self._control_objects)
        pause_annotation()

    def __call__(self, z, **kwargs):
        """Evaluate objective function"""
        continue_annotation()

        # Reconstruct field controls
        x = self.field(z)
        self.update_controls(x)

        # Evaluate data NLL
        data_nll = float(self.reduced_nll(self._controls))

        # Evaluate gradient of data NLL w.r.t. field controls
        grad = self.reduced_nll.derivative()
        if isinstance(grad, (firedrake.Function, firedrake.Constant)):
            grad_data_controls = grad.dat.data_ro
        else:
            grad_data_controls = np.stack([g.dat.data_ro for g in grad])
        pause_annotation()

        # Use vjp to get gradient w.r.t. latent parameters
        grad_data = self.field.vjp(z, grad_data_controls.ravel())

        # NOTE: we scale for numerical purposes (primarily for the fd HVP)
        # and to have larger range on the loss values
        return (self.scale * data_nll, self.scale * grad_data)

    def loss(self, z, **kwargs):
        continue_annotation()
        
        # Reconstruct field controls
        x = self.field(z)
        self.update_controls(x)

        # Evaluate data NLL
        data_nll = float(self.reduced_nll(self._controls))

        pause_annotation()
        return data_nll
    
    def update_controls(self, x):
        """
        Assign numpy array values to controls.
        """
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            self._controls.vector()[:] = x
        else:
            offset = 0
            for control in self._controls:
                size = len(control.dat.data_ro)
                control.dat.data[:] = x[offset : offset + size]
                offset += size

    def controls_array(self):
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
