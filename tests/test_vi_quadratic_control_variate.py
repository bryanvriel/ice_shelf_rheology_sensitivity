import numpy as np

import jax
import jax.numpy as jnp

from schurflow.scripts.variational_inference import (
    resolve_quadratic_control_variate_curvature,
)
from schurflow.vi import QuadraticControlVariate, SVIFullRankGaussian


class SerialComm:
    rank = 0
    size = 1

    def allgather(self, value):
        return [value]

    def Scatterv(self, sendbuf, recvbuf, root=0):
        del root
        send_array = sendbuf[0] if isinstance(sendbuf, (list, tuple)) else sendbuf
        recvbuf[:] = np.asarray(send_array, dtype=np.float64).reshape(recvbuf.shape)

    def Gatherv(self, sendbuf, recvbuf, root=0):
        del root
        recv_array = recvbuf[0] if isinstance(recvbuf, (list, tuple)) else recvbuf
        recv_array[:] = np.asarray(sendbuf, dtype=np.float64).reshape(recv_array.shape)

    def bcast(self, value, root=0):
        del root
        return value


def _quadratic_loglik(theta, ref_theta, ref_loglik, ref_grad, neg_loglik_hessian):
    theta = np.asarray(theta, dtype=np.float64)
    delta = theta - ref_theta
    loglik = ref_loglik + ref_grad @ delta - 0.5 * delta @ (
        neg_loglik_hessian @ delta
    )
    grad = ref_grad - neg_loglik_hessian @ delta
    return float(loglik), np.asarray(grad, dtype=np.float64)


def _make_svi(loglik_fn, control_variate=None, *, num_mc=8):
    prior_mean = np.zeros(2, dtype=np.float64)
    prior_precision = np.eye(2, dtype=np.float64)
    return SVIFullRankGaussian(
        loglik_fn,
        prior_mean,
        prior_precision,
        num_mc=num_mc,
        antithetic=True,
        comm=SerialComm(),
        control_variate=control_variate,
    )


def _make_params(svi, mean, scale_tril):
    scale_tril = jnp.asarray(scale_tril, dtype=svi.dtype)
    return {
        "mean": jnp.asarray(mean, dtype=svi.dtype),
        "raw_tril": svi.bij.inverse(scale_tril).astype(svi.dtype),
    }


def _assert_tree_allclose(tree_a, tree_b, atol=1e-8, rtol=1e-8):
    leaves_a, treedef_a = jax.tree_util.tree_flatten(tree_a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(tree_b)
    assert treedef_a == treedef_b
    for leaf_a, leaf_b in zip(leaves_a, leaves_b):
        np.testing.assert_allclose(
            np.asarray(leaf_a),
            np.asarray(leaf_b),
            atol=atol,
            rtol=rtol,
        )


def _finite_difference_grad(svi, params, step=1e-6):
    mean = np.asarray(params["mean"], dtype=np.float64)
    raw_tril = np.asarray(params["raw_tril"], dtype=np.float64)

    def expectation(mean_arr, raw_arr):
        probe = {
            "mean": jnp.asarray(mean_arr, dtype=svi.dtype),
            "raw_tril": jnp.asarray(raw_arr, dtype=svi.dtype),
        }
        return float(np.asarray(svi._quadratic_cv_expectation(probe)))

    grad_mean = np.zeros_like(mean)
    for i in range(mean.size):
        plus = mean.copy()
        minus = mean.copy()
        plus[i] += step
        minus[i] -= step
        grad_mean[i] = (expectation(plus, raw_tril) - expectation(minus, raw_tril)) / (
            2.0 * step
        )

    grad_raw = np.zeros_like(raw_tril)
    for i in range(raw_tril.size):
        plus = raw_tril.copy()
        minus = raw_tril.copy()
        plus[i] += step
        minus[i] -= step
        grad_raw[i] = (expectation(mean, plus) - expectation(mean, minus)) / (
            2.0 * step
        )

    return grad_mean, grad_raw


def _manual_expectation(mean, scale_tril, ref_theta, ref_loglik, ref_grad, hessian):
    Sigma = scale_tril @ scale_tril.T
    delta = mean - ref_theta
    return ref_loglik + ref_grad @ delta - 0.5 * (
        delta @ (hessian @ delta) + np.trace(hessian @ Sigma)
    )


def test_quadratic_control_variate_matches_exact_quadratic_loglikelihood():
    ref_theta = np.array([0.1, -0.2], dtype=np.float64)
    ref_loglik = 1.4
    ref_grad = np.array([0.3, -0.5], dtype=np.float64)
    neg_loglik_hessian = np.array([[1.8, 0.25], [0.25, 1.2]], dtype=np.float64)
    control_variate = QuadraticControlVariate(
        ref_theta=ref_theta,
        ref_loglik=ref_loglik,
        ref_grad_loglik=ref_grad,
        neg_loglik_hessian=neg_loglik_hessian,
        source="synthetic",
    )
    svi = _make_svi(
        lambda theta: _quadratic_loglik(
            theta,
            ref_theta,
            ref_loglik,
            ref_grad,
            neg_loglik_hessian,
        ),
        control_variate=control_variate,
        num_mc=8,
    )
    params = _make_params(
        svi,
        mean=np.array([0.35, -0.15], dtype=np.float64),
        scale_tril=np.array([[0.6, 0.0], [0.15, 0.5]], dtype=np.float64),
    )

    eps = svi.sample_eps(jax.random.PRNGKey(0), svi.num_mc)
    thetas = svi.sample_reparam(params, eps)
    cv_vals = np.asarray(jax.vmap(svi._quadratic_cv_value)(thetas))
    loglik_vals = np.asarray(
        [
            _quadratic_loglik(
                theta,
                ref_theta,
                ref_loglik,
                ref_grad,
                neg_loglik_hessian,
            )[0]
            for theta in np.asarray(thetas)
        ]
    )
    np.testing.assert_allclose(loglik_vals - cv_vals, 0.0, atol=1e-10, rtol=1e-10)

    loss_a, grads_a, metrics_a = svi.elbo_loss_mpi(params, jax.random.PRNGKey(3))
    loss_b, grads_b, metrics_b = svi.elbo_loss_mpi(params, jax.random.PRNGKey(11))

    expected_logP = np.asarray(svi._quadratic_cv_expectation(params))
    assert int(np.asarray(metrics_a["cv_active"])) == 1
    assert int(np.asarray(metrics_b["cv_active"])) == 1
    np.testing.assert_allclose(
        np.asarray(metrics_a["logP"]),
        expected_logP,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(metrics_b["logP"]),
        expected_logP,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(loss_a),
        np.asarray(loss_b),
        atol=1e-10,
        rtol=1e-10,
    )
    _assert_tree_allclose(grads_a, grads_b, atol=1e-10, rtol=1e-10)


def test_quadratic_control_variate_expectation_matches_closed_form():
    ref_theta = np.array([-0.2, 0.1], dtype=np.float64)
    ref_loglik = 0.75
    ref_grad = np.array([0.5, -0.15], dtype=np.float64)
    neg_loglik_hessian = np.array([[1.4, -0.3], [-0.3, 0.9]], dtype=np.float64)
    svi = _make_svi(
        lambda theta: _quadratic_loglik(
            theta,
            ref_theta,
            ref_loglik,
            ref_grad,
            neg_loglik_hessian,
        ),
        control_variate=QuadraticControlVariate(
            ref_theta=ref_theta,
            ref_loglik=ref_loglik,
            ref_grad_loglik=ref_grad,
            neg_loglik_hessian=neg_loglik_hessian,
            source="synthetic",
        ),
    )
    scale_tril = np.array([[0.7, 0.0], [0.2, 0.4]], dtype=np.float64)
    mean = np.array([0.25, -0.35], dtype=np.float64)
    params = _make_params(svi, mean=mean, scale_tril=scale_tril)

    actual = np.asarray(svi._quadratic_cv_expectation(params))
    expected = _manual_expectation(
        mean,
        scale_tril,
        ref_theta,
        ref_loglik,
        ref_grad,
        neg_loglik_hessian,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_quadratic_control_variate_expectation_gradient_matches_finite_difference():
    ref_theta = np.array([0.0, 0.2], dtype=np.float64)
    ref_loglik = 1.1
    ref_grad = np.array([-0.35, 0.45], dtype=np.float64)
    neg_loglik_hessian = np.array([[2.0, 0.1], [0.1, 1.1]], dtype=np.float64)
    svi = _make_svi(
        lambda theta: _quadratic_loglik(
            theta,
            ref_theta,
            ref_loglik,
            ref_grad,
            neg_loglik_hessian,
        ),
        control_variate=QuadraticControlVariate(
            ref_theta=ref_theta,
            ref_loglik=ref_loglik,
            ref_grad_loglik=ref_grad,
            neg_loglik_hessian=neg_loglik_hessian,
            source="synthetic",
        ),
    )
    params = _make_params(
        svi,
        mean=np.array([0.15, -0.05], dtype=np.float64),
        scale_tril=np.array([[0.55, 0.0], [0.12, 0.45]], dtype=np.float64),
    )

    _, grads = svi._quadratic_cv_expectation_value_and_grad(params)
    grad_mean_fd, grad_raw_fd = _finite_difference_grad(svi, params)

    np.testing.assert_allclose(
        np.asarray(grads["mean"]),
        grad_mean_fd,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(grads["raw_tril"]),
        grad_raw_fd,
        atol=1e-6,
        rtol=1e-6,
    )


def test_quadratic_control_variate_falls_back_to_baseline_when_any_sample_is_invalid():
    ref_theta = np.zeros(2, dtype=np.float64)
    ref_loglik = 0.9
    ref_grad = np.array([0.1, -0.2], dtype=np.float64)
    neg_loglik_hessian = np.array([[1.6, 0.2], [0.2, 0.8]], dtype=np.float64)

    def nan_loglik(theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta[0] > 0.0:
            return np.nan, np.full(theta.shape, np.nan, dtype=np.float64)
        return _quadratic_loglik(
            theta,
            ref_theta,
            ref_loglik,
            ref_grad,
            neg_loglik_hessian,
        )

    control_variate = QuadraticControlVariate(
        ref_theta=ref_theta,
        ref_loglik=ref_loglik,
        ref_grad_loglik=ref_grad,
        neg_loglik_hessian=neg_loglik_hessian,
        source="synthetic",
    )
    svi_cv = _make_svi(nan_loglik, control_variate=control_variate, num_mc=8)
    svi_base = _make_svi(nan_loglik, control_variate=None, num_mc=8)
    params = _make_params(
        svi_cv,
        mean=np.zeros(2, dtype=np.float64),
        scale_tril=np.array([[0.4, 0.0], [0.05, 0.35]], dtype=np.float64),
    )
    key = jax.random.PRNGKey(7)

    loss_cv, grads_cv, metrics_cv = svi_cv.elbo_loss_mpi(params, key)
    loss_base, grads_base, metrics_base = svi_base.elbo_loss_mpi(params, key)

    assert int(np.asarray(metrics_cv["cv_active"])) == 0
    np.testing.assert_allclose(
        np.asarray(metrics_cv["logP"]),
        np.asarray(metrics_base["logP"]),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(loss_cv),
        np.asarray(loss_base),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(metrics_cv["kl"]),
        np.asarray(metrics_base["kl"]),
        atol=1e-10,
        rtol=1e-10,
    )
    _assert_tree_allclose(grads_cv, grads_base, atol=1e-10, rtol=1e-10)


def test_resolve_quadratic_control_variate_curvature_prefers_data_hessian_then_falls_back():
    data_hessian = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64)
    posterior_hessian = np.array([[5.0, -0.5], [1.5, 4.0]], dtype=np.float64)
    prior_precision = np.array([[1.5, 0.2], [0.2, 0.8]], dtype=np.float64)

    actual_data, source_data = resolve_quadratic_control_variate_curvature(
        data_hessian=data_hessian,
        posterior_hessian=posterior_hessian,
        prior_precision=prior_precision,
        objective_scale=1.3,
    )
    np.testing.assert_allclose(
        actual_data,
        0.5 * (data_hessian + data_hessian.T),
        atol=1e-12,
        rtol=1e-12,
    )
    assert source_data == "data_hessian"

    actual_fallback, source_fallback = resolve_quadratic_control_variate_curvature(
        data_hessian=None,
        posterior_hessian=posterior_hessian,
        prior_precision=prior_precision,
        objective_scale=1.3,
    )
    expected_fallback = posterior_hessian - 1.3 * prior_precision
    expected_fallback = 0.5 * (expected_fallback + expected_fallback.T)
    np.testing.assert_allclose(
        actual_fallback,
        expected_fallback,
        atol=1e-12,
        rtol=1e-12,
    )
    assert source_fallback == "posterior_minus_prior"
