from typing import Callable, Tuple
import numpy as np
import scipy.sparse.linalg as spla

from firedrake import (
    assemble,
    inner,
    dx,
    TestFunction,
    TrialFunction,
    Function,
    FunctionSpace,
    Constant,
    derivative,
    action,
)
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    get_working_tape,
)
from icepack.calculus import grad
from petsc4py import PETSc

from .schur import EpochCombinedTheta, EpochParamOps, pack_Bn, split_Bn

Vec = np.ndarray


# ------------------------
# Main JVP/VJP wrapper builders
# ------------------------


def build_jvp_vjp_wrappers(
    simulation,
    Q,
    V,
    theta,
):
    """
    Returns two callables operating on NumPy arrays:
      jvp(p_arr) -> du_arr                (J p)
      vjp(r_arr, weight=True) -> g_arr    (J^T Σ^{-1} r if weight=True)
    """
    continue_annotation()

    theta_B, theta_n = theta

    # Record a clean forward onto the working tape
    u = simulation(theta)

    pause_annotation()

    # Tell the tape which output(s) we care about for TLM:
    tape = get_working_tape()

    def jvp(delta_theta_vec):

        # Clear old TLM values
        tape.reset_tlm_values()
        tape.reset_variables()

        N = delta_theta_vec.size // 2
        dB = Function(Q)
        dn = Function(Q)
        dB.dat.data[:] = delta_theta_vec[:N]
        dn.dat.data[:] = delta_theta_vec[N:]

        continue_annotation()

        # Seed tangents on controls
        theta_B.block_variable.tlm_value = dB
        theta_n.block_variable.tlm_value = dn

        # Run TLM sweep
        tape.evaluate_tlm()

        # Extract du
        du = u.block_variable.tlm_value

        pause_annotation()
        return du.dat.data_ro.ravel()

    def vjp(r_vec):

        # Clear old TLM values
        tape.reset_variables()
        tape.reset_tlm_values()

        # pack r into a VectorFunction
        r_fn = Function(V)
        N = r_vec.size // 2
        r_fn.dat.data[:] = r_vec.reshape(N, 2)

        continue_annotation()

        # Seed adjoint on output
        u.block_variable.adj_value = r_fn

        # Reverse sweep
        tape.evaluate_adj()

        gB = theta_B.block_variable.adj_value
        gn = theta_n.block_variable.adj_value

        pause_annotation()

        return np.hstack((gB.dat.data_ro, gn.dat.data_ro))

    return jvp, vjp


# ------------------------
# Converting regularization functions to linear operators
# ------------------------


def make_regularization_op(V, L_reg, Theta, tau=1e-6):
    """
    Constructs the combined smoothness + small Tikhonov regularization.
    The Tikhonov term is added to avoid non-uniqueness.

    Gamma(theta) = c * K * theta + tau * M * theta
      c   = (L_reg/Theta)^2 / area
      K   = ∫ grad·grad   (H1 seminorm)
      M   = ∫ 1·          (L2 mass)
    tau > 0 removes the constant-field nullspace (SPD).

    Returns:
      apply_Gamma(x_np) -> y_np   (NumPy arrays of length V.dim())
      area (float), c (float)
    """
    mesh = V.mesh()
    area = float(assemble(Constant(1.0) * dx(domain=mesh)))
    c = (L_reg / Theta)**2 / area

    v = TestFunction(V)
    w = TrialFunction(V)

    aK = inner(grad(v), grad(w)) * dx   # stiffness
    aM = inner(v, w) * dx               # mass

    uF  = Function(V, name="h1l2_apply_in")

    def apply_Gamma(x_np: np.ndarray) -> np.ndarray:
        uF.dat.data[:] = x_np
        # yK = K x
        yK = assemble(action(aK, uF))
        # yM = M x
        yM = assemble(action(aM, uF))
        return (c * yK.dat.data_ro + tau * yM.dat.data_ro).copy()

    return apply_Gamma


def make_regularization_mat(V, L_reg, Theta, tau=1e-6):
    """
    Assemble sparse PETSc Mat for Γ_B = c*K + τ*M in theta space.
    """
    mesh = V.mesh()
    area = float(assemble(Constant(1.0) * dx(domain=mesh)))
    c = (L_reg / Theta)**2 / area

    v = TestFunction(V); w = TrialFunction(V)
    aK = inner(grad(v), grad(w)) * dx
    aM = inner(v, w) * dx

    # Assemble the combined form directly
    aG = c * aK + tau * aM
    G_fd = assemble(aG, mat_type="aij")   # Firedrake Mat
    G = G_fd.M.handle                        # PETSc.Mat
    return G


def make_MB_from_gammaB_mat(GammaB_mat: PETSc.Mat, ksp_type='cg', pc_type='gamg'):
    """
    Build M_B as a SciPy LinearOperator that applies an *approximate* solve with Γ_B.
    """
    ksp = PETSc.KSP().create()
    ksp.setOperators(GammaB_mat)
    ksp.setType(ksp_type)
    pc = ksp.getPC(); pc.setType(pc_type)
    ksp.setFromOptions()
    x = GammaB_mat.createVecRight(); y = GammaB_mat.createVecLeft()

    def mv(r_np: np.ndarray) -> np.ndarray:
        x.array = r_np
        ksp.solve(x, y)                # y ≈ Γ_B^{-1} r
        return y.array.copy()

    n = GammaB_mat.getSize()[0]
    return spla.LinearOperator((n, n), matvec=mv, dtype=np.float64)


# ------------------------
# 1) DOT (ADJOINT) TESTS
# ------------------------
def dot_test_combined(
    ep: EpochCombinedTheta, dim2N: int, ntrials: int = 5, seed: int = 0
) -> float:
    """
    Test <W J v, y> == <v, J^T W^T y> in R^{2N}.
    Returns max relative mismatch over trials.
    """
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(ntrials):
        v = rng.standard_normal(dim2N)
        y = rng.standard_normal(dim2N)
        Jv = ep.jvp_theta(v)
        WJv = ep.whitener.apply(Jv)
        WT_y = ep.whitener.applyT(y)
        JTWTy = ep.vjp_theta(WT_y)
        lhs = float(WJv @ y)
        rhs = float(v @ JTWTy)
        denom = max(1.0, abs(lhs), abs(rhs))
        errs.append(abs(lhs - rhs) / denom)
    return max(errs)


def dot_test_split(
    ep_split: EpochParamOps, N: int, ntrials: int = 5, seed: int = 1
) -> Tuple[float, float]:
    """
    Dot tests for split blocks:
      <W J_B vB, y> == <vB, J_B^T W^T y>
      <W J_n vN, y> == <vN, J_n^T W^T y>
    Returns (max_err_B, max_err_n).
    """
    rng = np.random.default_rng(seed)
    m = 2 * N
    errB = []
    errN = []
    for _ in range(ntrials):
        vB = rng.standard_normal(N)
        vN = rng.standard_normal(N)
        y = rng.standard_normal(m)

        # B block
        JvB = ep_split.jvp_B(vB)
        WJvB = ep_split.whiten(JvB)
        JTWTy_B = ep_split.vjp_B(ep_split.whitenT(y))
        lhsB = float(WJvB @ y)
        rhsB = float(vB @ JTWTy_B)
        denomB = max(1.0, abs(lhsB), abs(rhsB))
        errB.append(abs(lhsB - rhsB) / denomB)

        # n block
        JvN = ep_split.jvp_n(vN)
        WJvN = ep_split.whiten(JvN)
        JTWTy_N = ep_split.vjp_n(ep_split.whitenT(y))
        lhsN = float(WJvN @ y)
        rhsN = float(vN @ JTWTy_N)
        denomN = max(1.0, abs(lhsN), abs(rhsN))
        errN.append(abs(lhsN - rhsN) / denomN)

    return max(errB), max(errN)


# ----------------------------------
# 2) FINITE-DIFFERENCE (FD) JVP TESTS
# ----------------------------------
def fd_jvp_test_combined(
    forward_epoch_theta: Callable[[Vec], Vec],
    theta0: Vec,
    jvp_theta: Callable[[Vec], Vec],
    direction: Vec,
    whiten: Callable[[Vec], Vec] = lambda v: v,
    eps: float = 1e-6,
) -> float:
    """
    Compare jvp_theta(direction) with FD: (F(θ + eps*v) - F(θ)) / eps, both whitened.
    Returns relative error.
    """
    u0 = forward_epoch_theta(theta0)
    u1 = forward_epoch_theta(theta0 + eps * direction)
    fd = (u1 - u0) / eps
    jv = jvp_theta(direction)
    fd_w = whiten(fd)
    jv_w = whiten(jv)
    num = np.linalg.norm(jv_w - fd_w)
    den = max(1.0, np.linalg.norm(jv_w), np.linalg.norm(fd_w))
    return float(num / den)


def fd_jvp_test_split(
    forward_epoch_theta: Callable[[Vec], Vec],
    theta0: Vec,
    N: int,
    jvp_block: Callable[[Vec], Vec],  # jvp_B or jvp_n
    direction_block: Vec,  # length N
    which: str,  # 'B' or 'n'
    whiten: Callable[[Vec], Vec] = lambda v: v,
    eps: float = 1e-6,
) -> float:
    """
    FD test for split blocks using packed direction [vB; 0] or [0; vN].
    """
    if which.lower() == "b":
        v_full = pack_Bn(direction_block, np.zeros(N))
    elif which.lower() == "n":
        v_full = pack_Bn(np.zeros(N), direction_block)
    else:
        raise ValueError("which must be 'B' or 'n'")

    u0 = forward_epoch_theta(theta0)
    u1 = forward_epoch_theta(theta0 + eps * v_full)
    fd = (u1 - u0) / eps
    jv = jvp_block(direction_block)
    fd_w = whiten(fd)
    jv_w = whiten(jv)
    num = np.linalg.norm(jv_w - fd_w)
    den = max(1.0, np.linalg.norm(jv_w), np.linalg.norm(fd_w))
    return float(num / den)


# end of file
