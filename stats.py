# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Union


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


def _flatten_field_and_samples(
    f: np.ndarray,
    samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten field and samples to 1D/2D for uniform processing.

    Args:
        f: Field array, shape (M,) or (Ny, Nx).
        samples: Sample array, shape (N_samples, M) or (N_samples, Ny, Nx).

    Returns:
        f_flat: Flattened field, shape (M_total,).
        samples_flat: Flattened samples, shape (N_samples, M_total).
    """
    if f.ndim == 1:
        return f, samples
    elif f.ndim == 2:
        # 2D field (Ny, Nx) -> flatten to (Ny*Nx,)
        f_flat = f.ravel()
        # 3D samples (N, Ny, Nx) -> reshape to (N, Ny*Nx)
        N = samples.shape[0]
        samples_flat = samples.reshape(N, -1)
        return f_flat, samples_flat
    else:
        raise ValueError(f"Field must be 1D or 2D, got shape {f.shape}")


def _flatten_mask(
    mask: Optional[np.ndarray],
    reference_shape: Tuple,
) -> Optional[np.ndarray]:
    """
    Flatten mask to match flattened field.

    Args:
        mask: Mask array, shape (M,) or (Ny, Nx), or None.
        reference_shape: Shape of the field (for validation).

    Returns:
        Flattened mask, shape (M_total,), or None.
    """
    if mask is None:
        return None

    if mask.shape != reference_shape:
        raise ValueError(
            f"Mask shape {mask.shape} doesn't match field shape {reference_shape}"
        )

    return mask.ravel()


def _get_valid_mask(
    *arrays: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute combined valid mask from multiple arrays and optional explicit mask.

    A point is valid if:
    - It's True in the explicit mask (if provided)
    - It's not NaN in any of the input arrays

    Args:
        *arrays: Flattened arrays to check for NaN values.
                 Can be 1D (M,) or 2D (N_samples, M).
        mask: Optional explicit mask, shape (M,). True = valid.

    Returns:
        Boolean mask, shape (M,). True = valid point.
    """
    # Start with all True if no explicit mask
    if mask is None:
        first = arrays[0]
        M = first.shape[-1]  # Last dimension is spatial
        valid = np.ones(M, dtype=bool)
    else:
        valid = mask.copy()

    # Exclude NaN values from all arrays
    for arr in arrays:
        if arr.ndim == 1:
            valid &= np.isfinite(arr)
        elif arr.ndim == 2:
            # For 2D arrays (samples), a point is invalid if ANY sample has NaN
            valid &= np.all(np.isfinite(arr), axis=0)

    return valid


def rms_z_score_field(
    f_true: np.ndarray,
    samples: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute RMS z-score in field space (ignores correlations).

    This is the simplest approach: compute z-scores pointwise and average.

    Args:
        f_true: Ground truth field, shape (M,) or (Ny, Nx).
        samples: Posterior samples, shape (N_samples, M) or (N_samples, Ny, Nx).
        mask: Optional boolean mask, same shape as f_true. True = valid point.
              Points with NaN in f_true or any sample are automatically excluded.

    Returns:
        RMS z-score. Should be ≈ 1.0 for well-calibrated marginals.
    """
    # Flatten to 1D
    f_flat, samples_flat = _flatten_field_and_samples(f_true, samples)
    mask_flat = _flatten_mask(mask, f_true.shape)

    # Get valid points
    valid = _get_valid_mask(f_flat, samples_flat, mask=mask_flat)

    if not np.any(valid):
        return np.nan

    # Extract valid points
    f_valid = f_flat[valid]
    samples_valid = samples_flat[:, valid]

    sample_mean = np.mean(samples_valid, axis=0)
    sample_std = np.std(samples_valid, axis=0, ddof=1)

    # Avoid division by zero in regions with no uncertainty
    nonzero_std = sample_std > 1e-10

    if not np.any(nonzero_std):
        return np.nan

    z_scores = (f_valid[nonzero_std] - sample_mean[nonzero_std]) / sample_std[
        nonzero_std
    ]
    rms_z = np.sqrt(np.mean(z_scores**2))

    return float(rms_z)


def mahalanobis_low_rank(
    f_true: np.ndarray,
    samples: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
    explained_variance_threshold: float = 0.95,
    min_singular_value_fraction: float = 0.01,
    regularization: float = 0.0,
) -> Tuple[float, dict]:
    """
    Compute Mahalanobis distance using low-rank approximation of the covariance.

    This approach:
    1. Computes PCA of the posterior samples
    2. Projects truth and samples onto the principal components
    3. Computes Mahalanobis distance in this reduced space

    Args:
        f_true: Ground truth field, shape (M,) or (Ny, Nx).
        samples: Posterior samples, shape (N_samples, M) or (N_samples, Ny, Nx).
        mask: Optional boolean mask, same shape as f_true. True = valid point.
        n_components: Number of PCA components. If None, uses explained_variance_threshold.
        explained_variance_threshold: Fraction of variance to capture (default 95%).
        min_singular_value_fraction: Exclude components with singular value < this fraction
                                     of the largest singular value (default 0.01 = 1%).
                                     This prevents noise-dominated components from
                                     dominating the Mahalanobis distance.
        regularization: Add this value to all singular values before computing z-scores.
                       Helps stabilize when some components have very small variance.
                       A typical value might be 0.01 * median(singular_values).

    Returns:
        mahal_sq_normalized: Mahalanobis² / n_components (should be ≈ 1.0).
        info: Dict with additional information.
    """
    # Flatten to 1D
    f_flat, samples_flat = _flatten_field_and_samples(f_true, samples)
    mask_flat = _flatten_mask(mask, f_true.shape)

    # Get valid points
    valid = _get_valid_mask(f_flat, samples_flat, mask=mask_flat)

    if not np.any(valid):
        return np.nan, {"error": "No valid points"}

    # Extract valid points
    f_valid = f_flat[valid]
    samples_valid = samples_flat[:, valid]

    N, M_valid = samples_valid.shape

    if N < 2:
        return np.nan, {"error": "Need at least 2 samples"}

    # Center the samples
    sample_mean = np.mean(samples_valid, axis=0)
    samples_centered = samples_valid - sample_mean

    # SVD of centered samples
    U, S, Vt = np.linalg.svd(samples_centered, full_matrices=False)

    # Explained variance
    explained_variance = S**2 / (N - 1)
    total_variance = np.sum(explained_variance)

    if total_variance < 1e-10:
        return np.nan, {"error": "Zero variance in samples"}

    explained_variance_ratio = np.cumsum(explained_variance) / total_variance

    # Determine number of components based on multiple criteria
    max_components = min(len(S), N - 1)  # Can't have more than N-1 reliable components

    if n_components is None:
        # Use variance threshold
        n_components = (
            np.searchsorted(explained_variance_ratio, explained_variance_threshold) + 1
        )
        n_components = min(n_components, max_components)
    else:
        n_components = min(n_components, max_components)

    # Additional filter: exclude components with very small singular values
    # These are often noise-dominated and destabilize the Mahalanobis distance
    if min_singular_value_fraction > 0:
        sv_threshold = min_singular_value_fraction * S[0]
        valid_sv = S[:n_components] > sv_threshold
        n_components_filtered = np.sum(valid_sv)

        if n_components_filtered < n_components:
            # Report how many were filtered
            n_filtered_out = n_components - n_components_filtered
            n_components = max(n_components_filtered, 1)  # Keep at least 1
    else:
        n_filtered_out = 0

    # Project truth onto principal components
    f_centered = f_valid - sample_mean
    f_proj = Vt[:n_components, :] @ f_centered

    # Standard deviation along each PC (with optional regularization)
    pc_std = S[:n_components] / np.sqrt(N - 1)

    if regularization > 0:
        pc_std = np.sqrt(pc_std**2 + regularization**2)

    # Avoid division by zero
    nonzero_std = pc_std > 1e-10
    if not np.any(nonzero_std):
        return np.nan, {"error": "Zero variance in principal components"}

    # Mahalanobis distance in PC space
    z_pc = np.zeros(n_components)
    z_pc[nonzero_std] = f_proj[nonzero_std] / pc_std[nonzero_std]
    mahal_sq = np.sum(z_pc**2)
    n_effective = np.sum(nonzero_std)
    mahal_sq_normalized = mahal_sq / n_effective

    info = {
        "n_components": n_components,
        "n_components_effective": int(n_effective),
        "n_valid_points": int(np.sum(valid)),
        "n_samples": N,
        "explained_variance_ratio": float(
            explained_variance_ratio[
                min(n_components - 1, len(explained_variance_ratio) - 1)
            ]
        ),
        "mahal_sq": float(mahal_sq),
        "pc_z_scores": z_pc,
        "singular_values": S,
        "min_sv_used": float(S[n_components - 1]) if n_components > 0 else np.nan,
        "max_sv": float(S[0]),
    }

    return float(mahal_sq_normalized), info


def hpd_coverage_field(
    f_true: np.ndarray,
    samples: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
    min_singular_value_fraction: float = 0.01,
) -> Tuple[float, dict]:
    """
    Compute HPD coverage in field space using low-rank Mahalanobis.

    Computes what fraction of posterior samples are "further" from the mean
    than the truth (in the Mahalanobis sense).

    Args:
        f_true: Ground truth field, shape (M,) or (Ny, Nx).
        samples: Posterior samples, shape (N_samples, M) or (N_samples, Ny, Nx).
        mask: Optional boolean mask, same shape as f_true. True = valid point.
        n_components: Number of PCA components (None = auto).
        min_singular_value_fraction: Exclude components with singular value < this fraction
                                     of the largest (default 0.01).

    Returns:
        hpd_coverage: Fraction of samples further from mean than truth.
                      Should be ≈ 0.5 for well-calibrated posterior.
        info: Dict with additional information.
    """
    # Flatten to 1D
    f_flat, samples_flat = _flatten_field_and_samples(f_true, samples)
    mask_flat = _flatten_mask(mask, f_true.shape)

    # Get valid points
    valid = _get_valid_mask(f_flat, samples_flat, mask=mask_flat)

    if not np.any(valid):
        return np.nan, {"error": "No valid points"}

    # Extract valid points
    f_valid = f_flat[valid]
    samples_valid = samples_flat[:, valid]

    N, M_valid = samples_valid.shape

    if N < 2:
        return np.nan, {"error": "Need at least 2 samples"}

    # Center the samples
    sample_mean = np.mean(samples_valid, axis=0)
    samples_centered = samples_valid - sample_mean

    # SVD
    U, S, Vt = np.linalg.svd(samples_centered, full_matrices=False)

    # Determine number of components
    max_components = min(len(S), N - 1)

    if n_components is None:
        # Auto-select: 95% variance
        explained_variance = S**2 / (N - 1)
        total_var = np.sum(explained_variance)
        if total_var < 1e-10:
            return np.nan, {"error": "Zero variance"}
        cumvar = np.cumsum(explained_variance) / total_var
        n_components = np.searchsorted(cumvar, 0.95) + 1
        n_components = min(n_components, max_components)
    else:
        n_components = min(n_components, max_components)

    # Filter out small singular values
    if min_singular_value_fraction > 0:
        sv_threshold = min_singular_value_fraction * S[0]
        valid_sv = S[:n_components] > sv_threshold
        n_components = max(np.sum(valid_sv), 1)

    # Standard deviation along each PC
    pc_std = S[:n_components] / np.sqrt(N - 1)

    # Avoid division by zero
    nonzero_std = pc_std > 1e-10
    if not np.any(nonzero_std):
        return np.nan, {"error": "Zero variance in PCs"}

    # Project truth
    f_centered = f_valid - sample_mean
    f_proj = Vt[:n_components, :] @ f_centered

    z_truth = np.zeros(n_components)
    z_truth[nonzero_std] = f_proj[nonzero_std] / pc_std[nonzero_std]
    mahal_sq_truth = np.sum(z_truth**2)

    # Project each sample and compute its Mahalanobis distance
    mahal_sq_samples = np.zeros(N)
    for i in range(N):
        s_proj = Vt[:n_components, :] @ samples_centered[i]
        z_s = np.zeros(n_components)
        z_s[nonzero_std] = s_proj[nonzero_std] / pc_std[nonzero_std]
        mahal_sq_samples[i] = np.sum(z_s**2)

    # Fraction of samples further from mean than truth
    hpd_coverage = np.mean(mahal_sq_samples > mahal_sq_truth)

    info = {
        "n_components": n_components,
        "n_valid_points": int(np.sum(valid)),
        "mahal_sq_truth": float(mahal_sq_truth),
        "mahal_sq_samples_mean": float(np.mean(mahal_sq_samples)),
        "mahal_sq_samples_std": float(np.std(mahal_sq_samples)),
    }

    return float(hpd_coverage), info


def calibration_summary_field(
    f_true_B: np.ndarray,
    f_true_n: np.ndarray,
    samples_B: np.ndarray,
    samples_n: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
    min_singular_value_fraction: float = 0.01,
) -> dict:
    """
    Compute comprehensive calibration summary for B and n fields.

    Args:
        f_true_B: Ground truth B field, shape (M,) or (Ny, Nx).
        f_true_n: Ground truth n field, shape (M,) or (Ny, Nx).
        samples_B: Posterior samples of B, shape (N_samples, M) or (N_samples, Ny, Nx).
        samples_n: Posterior samples of n, shape (N_samples, M) or (N_samples, Ny, Nx).
        mask: Optional boolean mask, same shape as f_true_B. True = valid point.
        n_components: Number of PCA components for Mahalanobis (None = auto at 95% variance).
        min_singular_value_fraction: Exclude components with singular value < this fraction
                                     of the largest (default 0.01). Stabilizes metrics.

    Returns:
        Dictionary with calibration metrics.
    """
    # Flatten everything for joint analysis
    f_B_flat, samples_B_flat = _flatten_field_and_samples(f_true_B, samples_B)
    f_n_flat, samples_n_flat = _flatten_field_and_samples(f_true_n, samples_n)
    mask_flat = _flatten_mask(mask, f_true_B.shape)

    # Get combined valid mask
    valid = _get_valid_mask(
        f_B_flat, f_n_flat, samples_B_flat, samples_n_flat, mask=mask_flat
    )
    n_valid = int(np.sum(valid))

    # Concatenate B and n for joint analysis
    f_true_joint = np.concatenate([f_B_flat, f_n_flat])
    samples_joint = np.hstack([samples_B_flat, samples_n_flat])

    # Create extended mask for joint arrays
    mask_joint = np.concatenate([valid, valid])

    # RMS z-scores (marginal calibration)
    print("RMS Z-score")
    rms_z_B = rms_z_score_field(f_B_flat, samples_B_flat, mask=valid)
    rms_z_n = rms_z_score_field(f_n_flat, samples_n_flat, mask=valid)
    rms_z_joint = rms_z_score_field(f_true_joint, samples_joint, mask=mask_joint)

    # Low-rank Mahalanobis (joint calibration accounting for correlations)
    print("Mahalanobis")
    mahal_B, info_B = mahalanobis_low_rank(
        f_B_flat,
        samples_B_flat,
        mask=valid,
        n_components=n_components,
        min_singular_value_fraction=min_singular_value_fraction,
    )
    mahal_n, info_n = mahalanobis_low_rank(
        f_n_flat,
        samples_n_flat,
        mask=valid,
        n_components=n_components,
        min_singular_value_fraction=min_singular_value_fraction,
    )
    mahal_joint, info_joint = mahalanobis_low_rank(
        f_true_joint,
        samples_joint,
        mask=mask_joint,
        n_components=n_components,
        min_singular_value_fraction=min_singular_value_fraction,
    )

    # HPD coverage
    print("HPD")
    hpd_B, _ = hpd_coverage_field(
        f_B_flat,
        samples_B_flat,
        mask=valid,
        n_components=n_components,
        min_singular_value_fraction=min_singular_value_fraction,
    )
    hpd_n, _ = hpd_coverage_field(
        f_n_flat,
        samples_n_flat,
        mask=valid,
        n_components=n_components,
        min_singular_value_fraction=min_singular_value_fraction,
    )
    hpd_joint, _ = hpd_coverage_field(
        f_true_joint,
        samples_joint,
        mask=mask_joint,
        n_components=n_components,
        min_singular_value_fraction=min_singular_value_fraction,
    )

    return {
        # Marginal calibration (should be ≈ 1.0)
        "rms_z_B": rms_z_B,
        "rms_z_n": rms_z_n,
        "rms_z_joint": rms_z_joint,
        # Joint calibration - Mahalanobis (should be ≈ 1.0)
        "mahal_normalized_B": mahal_B,
        "mahal_normalized_n": mahal_n,
        "mahal_normalized_joint": mahal_joint,
        # HPD coverage (should be ≈ 0.5)
        "hpd_coverage_B": hpd_B,
        "hpd_coverage_n": hpd_n,
        "hpd_coverage_joint": hpd_joint,
        # Info
        "n_valid_points": n_valid,
        "n_samples": info_B.get("n_samples", np.nan),
        "n_components_B": info_B.get("n_components", np.nan),
        "n_components_n": info_n.get("n_components", np.nan),
        "n_components_joint": info_joint.get("n_components", np.nan),
        "explained_var_B": info_B.get("explained_variance_ratio", np.nan),
        "explained_var_n": info_n.get("explained_variance_ratio", np.nan),
        "explained_var_joint": info_joint.get("explained_variance_ratio", np.nan),
        "min_sv_fraction_used": min_singular_value_fraction,
    }


def print_calibration_summary(summary: dict):
    """Pretty-print calibration summary."""
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY (Field Space)")
    print("=" * 60)
    print(f"Valid points: {summary['n_valid_points']}")

    print(f"\n{'Metric':<30} {'B':>10} {'n':>10} {'Joint':>10} {'Target':>10}")
    print("-" * 70)

    def fmt(val):
        return f"{val:>10.3f}" if np.isfinite(val) else f"{'N/A':>10}"

    def fmt_pct(val):
        return f"{val:>10.1%}" if np.isfinite(val) else f"{'N/A':>10}"

    print(
        f"{'RMS Z-score':<30} {fmt(summary['rms_z_B'])} {fmt(summary['rms_z_n'])} "
        f"{fmt(summary['rms_z_joint'])} {'1.0':>10}"
    )
    print(
        f"{'Mahalanobis²/k (low-rank)':<30} {fmt(summary['mahal_normalized_B'])} "
        f"{fmt(summary['mahal_normalized_n'])} {fmt(summary['mahal_normalized_joint'])} {'1.0':>10}"
    )
    print(
        f"{'HPD Coverage':<30} {fmt_pct(summary['hpd_coverage_B'])} "
        f"{fmt_pct(summary['hpd_coverage_n'])} {fmt_pct(summary['hpd_coverage_joint'])} {'50%':>10}"
    )

    print(
        f"\n{'PCA components used':<30} {summary['n_components_B']:>10} "
        f"{summary['n_components_n']:>10} {summary['n_components_joint']:>10}"
    )
    print(
        f"{'Variance explained':<30} {fmt_pct(summary['explained_var_B'])} "
        f"{fmt_pct(summary['explained_var_n'])} {fmt_pct(summary['explained_var_joint'])}"
    )


# end of file
