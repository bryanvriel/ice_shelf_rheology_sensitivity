#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def compute_binned_acf(x, y, residuals, max_lag, bin_size, n_subsample=5000, rng=None):
    """
    Computes 1D ACF from irregular data by binning pairwise distances.

    Parameters:
        x, y: 1D arrays of coordinates.
        residuals: 1D array of noise values.
        max_lag: Maximum distance to calculate correlation for (e.g., 5000m).
        bin_size: Width of distance bins (e.g., 200m).
        n_subsample: Max number of points to use (prevents memory crash on huge grids).

    Returns:
        lags_centers: Distance vector.
        acf_values: Correlation values.
    """
    # 1. Subsample if data is too large (N^2 problem)
    n_points = len(residuals)
    if n_points > n_subsample:
        # Randomly choose indices
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(n_points, n_subsample, replace=False)
        x, y, r = x[indices], y[indices], residuals[indices]
    else:
        r = residuals

    # 2. Subtract mean (essential for ACF)
    r = r - np.mean(r)
    var = np.var(r)

    # 3. Build KDTree for efficient neighbor search
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)

    # 4. Find all pairs within max_lag
    # valid_pairs is a set of (i, j) indices
    valid_pairs = tree.query_pairs(max_lag)
    valid_pairs = np.array(list(valid_pairs))

    if len(valid_pairs) == 0:
        raise ValueError("No pairs found within max_lag. Increase max_lag.")

    idx_i = valid_pairs[:, 0]
    idx_j = valid_pairs[:, 1]

    # 5. Calculate distances and products for these pairs
    d_x = x[idx_i] - x[idx_j]
    d_y = y[idx_i] - y[idx_j]
    dists = np.sqrt(d_x**2 + d_y**2)

    # Covariance contribution of each pair
    products = r[idx_i] * r[idx_j]

    # 6. Binning
    bins = np.arange(0, max_lag + bin_size, bin_size)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # digitize returns which bin each distance belongs to
    bin_indices = np.digitize(dists, bins)

    acf_values = []
    lags_out = []

    for k in range(1, len(bins)):
        # Find all pairs in this distance bin
        in_bin = bin_indices == k
        if np.any(in_bin):
            # Average covariance in this bin divided by variance = Correlation
            mean_cov = np.mean(products[in_bin])
            acf_values.append(mean_cov / var)
            lags_out.append(bin_centers[k - 1])

    # Add lag 0 (always 1.0)
    lags_out = np.insert(lags_out, 0, 0.0)
    acf_values = np.insert(acf_values, 0, 1.0)

    return np.array(lags_out), np.array(acf_values)


def fit_model_to_irregular(lags, acf_values, model="gaussian"):
    
    def gaussian_func(h, L):
        return np.exp(-(h**2) / (L**2))

    def exponential_func(h, L):
        return np.exp(-np.abs(h) / L)

    func_map = {"gaussian": gaussian_func, "exponential": exponential_func}

    # Drop NaNs if binning resulted in empty bins
    valid = ~np.isnan(acf_values)

    try:
        popt, _ = optimize.curve_fit(
            func_map[model], lags[valid], acf_values[valid], p0=[lags[1] * 5]
        )
        return popt[0]
    except:
        return np.nan


def build_dense_covariance(coords, sigma, L, model="gaussian"):
    """
    Constructs the dense Covariance Matrix Cd.
    NOTE: Only use this for small datasets (N < 5000 points).

    Parameters:
        coords (np.ndarray): (N, 2) array of [x, y] coordinates.
        sigma (float): Standard deviation of noise.
        L (float): Correlation length.

    Returns:
        np.ndarray: (N, N) Covariance matrix.
    """
    # Pairwise Euclidean distances
    dists = cdist(coords, coords, metric="euclidean")

    if model == "gaussian":
        correlation = np.exp(-(dists**2) / (L**2))
    elif model == "exponential":
        correlation = np.exp(-dists / L)

    Cd = (sigma**2) * correlation
    return Cd


def plot_irregular_diagnostics(x, y, resid, lags, acf, L_est, model):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # Map view of the masked residuals
    sc = axes[0].scatter(
        x,
        y,
        c=resid,
        s=5,
        cmap="RdBu_r",
        vmin=-np.std(resid) * 2,
        vmax=np.std(resid) * 2,
    )
    axes[0].set_title("Masked Residuals (Spatial)")
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0])

    # Histogram
    axes[1].hist(resid, bins=50, density=True, color="gray", alpha=0.7)
    axes[1].set_title(f"Histogram (Std={np.std(resid):.2f})")

    # Variogram/ACF Fit
    axes[2].plot(lags, acf, "ko", alpha=0.5, label="Binned Data")

    if model == "gaussian":
        y_fit = np.exp(-(lags**2) / (L_est**2))
    else:
        y_fit = np.exp(-lags / L_est)

    axes[2].plot(lags, y_fit, "r-", lw=2, label=f"Fit L={L_est:.0f}m")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Correlation")
    axes[2].set_title("Experimental Variogram/ACF")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    return fig, axes


def dispersion(ref, samples):
    mean = np.nanmean(samples, axis=0)
    std = np.nanstd(samples, axis=0)
    z = (mean - ref) / std
    return np.sqrt(np.nanmean(z**2))


# end of file
