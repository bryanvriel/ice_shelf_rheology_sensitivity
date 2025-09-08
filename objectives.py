#-*- coding: utf-8 -*-

import numpy as np
import firedrake
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
)
import sys


class FiredrakeObjective:

    def __init__(self, field, problem, scale=1.0):
        continue_annotation()

        self.field = field
        self.problem = problem
        controls = problem.controls
        self.scale = scale

        # Initialize the objective components
        if isinstance(controls, (firedrake.Function, firedrake.Constant)):
            self._controls = controls.copy(deepcopy=True)
        else:
            self._controls = [field.copy(deepcopy=True) for field in controls]
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
            grad_data_controls = np.array(grad.vector()[:])
        else:
            grad_data_controls = np.stack([g.vector()[:] for g in grad])
        pause_annotation()

        # Use vjp to get gradient w.r.t. latent parameters
        grad_data = self.field.vjp(z, grad_data_controls.ravel())

        # Evaluate prior NLL and its gradient
        prior_nll, grad_prior = self.field.prior_nll_and_grad(z)

        # Total objective and gradient
        # NOTE: we scale for numerical purposes (primarily for the fd HVP)
        # and to have larger range on the loss values
        obj = self.scale * (data_nll + prior_nll)
        obj_grad = self.scale * (grad_data + grad_prior)

        return (obj, obj_grad)

    def forward_jvp(self, z, v):
        """
        Computes the Jacobian-vector product J @ v, where J = du/dz.
        
        Args:
            z (np.ndarray): The point in the latent space at which
                            to compute the JVP.
            v (np.ndarray): The vector to multiply the Jacobian by.

        Returns:
            np.ndarray: The result of (du/dz @ v), as a NumPy array.
        """
        continue_annotation()
        
        # 1. INNER JVP: Compute perturbation in control space
        # m_pert = dm/dz @ v
        # jvp returns (field(z), field'(z) @ v)
        θ, θ_pert = self.field.jvp(z, v)
        
        # 2. OUTER JVP: Solve the Tangent Linear Model for u_pert
        # We need the unperturbed state u at m
        u = self.problem.forward(θ)
        
        # Define the unknown for the tangent linear system: state perturbation
        u_pert = firedrake.Function(self.problem.Q)
        
        # Get the residual of the forward problem, R(u, m)
        F = self.problem.residual_velocity(u, m)
        
        # Firedrake's 'derivative' computes the Gâteaux derivative, which is
        # what we need for the tangent linear model.
        # (∂R/∂u) @ u_pert + (∂R/∂m) @ m_pert = 0
        F_tlm = (
            firedrake.derivative(F, u, u_pert) +
            firedrake.derivative(F, m, m_pert)
        )
        
        # The term with u_pert is the LHS, the term with m_pert is the RHS.
        # Firedrake separates these automatically when creating the problem.
        # We are solving for u_pert.
        solve_problem = firedrake.LinearVariationalProblem(
            firedrake.lhs(F_tlm), 
            -firedrake.rhs(F_tlm), # Move the m_pert term to the RHS
            u_pert
        )
        solver = firedrake.LinearVariationalSolver(solve_problem)
        solver.solve()
        
        pause_annotation()
        
        # Return the result as a NumPy array
        return u_pert.vector().get_local()

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
        continue_annotation()

        # Reconstruct field controls
        x = self.field(z)
        self.update_controls(x)

        # Evaluate data NLL
        data_nll = float(self.reduced_nll(self._controls))

        # Evaluate prior NLL and its gradient
        prior_nll, grad_prior = self.field.prior_nll_and_grad(z)

        pause_annotation()

        return self.scale * (data_nll + prior_nll)

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


class DataObjective(FiredrakeObjective):

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
            grad_data_controls = np.array(grad.vector()[:])
        else:
            grad_data_controls = np.stack([g.vector()[:] for g in grad])
        pause_annotation()

        # Use vjp to get gradient w.r.t. latent parameters
        grad_nll = self.field.vjp(z, grad_data_controls)

        return (data_nll, grad_nll)


class FiredrakeObjectiveSmoothing:

    def __init__(self, field, problem):
        continue_annotation()

        self.field = field
        controls = problem.controls

        # Initialize the objective components
        if isinstance(controls, (firedrake.Function, firedrake.Constant)):
            self._controls = controls.copy(deepcopy=True)
            self._v_controls = controls.copy(deepcopy=True)
            self.is_multi_control = False
        else:
            self._controls = [field.copy(deepcopy=True) for field in controls]
            self._v_controls = [field.copy(deepcopy=True) for field in controls]
            self.is_multi_control = True
        self._state = problem.forward(self._controls)

        # Form data objective
        E_form = sum(
            firedrake.assemble(E(self._state))
            for E in [
                problem.data_nll,
            ]
        )
        R_form = sum(
            firedrake.assemble(R(self._controls))
            for R in [
                problem.regularization,
            ]
        )
        self.J = E_form + R_form

        # Call assemble here to preserve UFL forms to allow for Hessian computation
        # self.J = firedrake.assemble(J_form)

        # Create the reduced functional and store control objects
        if not self.is_multi_control:
            self._control_objects = Control(self._controls)
        else:
            self._control_objects = [Control(field) for field in self._controls]
        self.reduced_nll = ReducedFunctional(self.J, self._control_objects)

        ## Symbolic setup for Gauss-Newton Hessian-vector product ---

        ## We need the symbolic forms again to differentiate
        # R_form_hvp = sum(R(self._controls) for R in [problem.regularization,])
        # E_form_hvp = sum(E(self._state) for E in [problem.data_nll,])

        # if not self.is_multi_control:
        #    dR_v = firedrake.derivative(R_form_hvp, self._controls, self._v_controls)
        #    jvp_E = firedrake.derivative(E_form_hvp, self._controls, self._v_controls)
        # else:
        #    dR_v_components = []
        #    jvp_E_components = []
        #    for u, v in zip(self._controls, self._v_controls):
        #        dR_v_components.append(firedrake.derivative(R_form_hvp, u, v))
        #        jvp_E_components.append(firedrake.derivative(E_form_hvp, u, v))
        #
        #    # Sum the symbolic components to get the total Gateaux derivative
        #    dR_v = firedrake.assemble(sum(dR_v_components))
        #    jvp_E = firedrake.assemble(sum(jvp_E_components))
        #
        ## The results dR_v and jvp_E are now single, valid UFL expressions
        # self.hvp_R_functional = ReducedFunctional(dR_v, self._control_objects)
        # self.hvp_E_functional = ReducedFunctional(jvp_E, self._control_objects)

        pause_annotation()

    def __call__(self, z):
        """Evaluate objective function"""
        continue_annotation()

        # Reconstruct field controls
        x = self.field(z)
        self.update_controls(x)

        # Evaluate posterior NLL
        nll = float(self.reduced_nll(self._controls))

        # Evaluate gradient of NLL w.r.t. field controls
        grad = self.reduced_nll.derivative()
        if isinstance(grad, (firedrake.Function, firedrake.Constant)):
            grad_controls = np.array(grad.vector()[:])
        else:
            grad_controls = np.concatenate([g.vector()[:] for g in grad])
        pause_annotation()

        # Use vjp to get gradient w.r.t. latent parameters
        grad_nll = self.field.vjp(z, grad_controls)

        # Add contributions from prior on latent variables
        latent_nll, grad_latent = tikhonov_and_grad(z, 0.0002)
        nll += latent_nll
        grad_nll += grad_latent

        return (nll, grad_nll)

    def gauss_newton_hvp(self, z, v):
        """
        Matrix-free Gauss–Newton Hessian–vector product in latent space.

        Cost: 1 tangent + 1 adjoint PDE solve (no second-order tape).
        """
        # Map latent ─► control space direction
        dv_ctrl = np.asarray(self.field.jvp(z, v))  # J_field · v
        v_ctrl = self.array_to_controls(dv_ctrl)  # list/Function/Constant

        # Tangent linear solve  𝛅u = J_state · v_ctrl
        # delta_state = self.reduced_nll.tlm(v_ctrl)

        # Adjoint solve with RHS = delta_state   (  w  = Jᵀ · δu  )
        # w_ctrl = self.reduced_nll.adjoint(delta_state)
        w_ctrl = self.reduced_nll.adjoint(z)

        # w_ctrl lives in the same control space, flatten it:
        if isinstance(w_ctrl, (firedrake.Function, firedrake.Constant)):
            hvp_ctrl_flat = w_ctrl.vector()[:]
        else:
            hvp_ctrl_flat = np.concatenate([c.vector()[:] for c in w_ctrl])

        # Prior in latent coords: ½ zᵀ S z  ⇒  H_prior = S
        hvp_latent = self.field.vjp(z, hvp_ctrl_flat)  # J_fieldᵀ · w

        return hvp_latent

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

    def loss(self, z):
        continue_annotation()

        # Reconstruct field controls
        x = self.field(z)
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

    def update_v_controls(self, x):
        """
        Assign numpy array values to controls.
        """
        if isinstance(self._controls, (firedrake.Function, firedrake.Constant)):
            self._v_controls.vector()[:] = x
        else:
            offset = 0
            for control in self._v_controls:
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


class FiredrakeObjectiveMesh:

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
