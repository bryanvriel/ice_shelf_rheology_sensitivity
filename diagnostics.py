import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

import jax
import jax.numpy as jnp


class IceRheologyStatistics:
    def __init__(self, J_B=None, J_n=None, W=None, Cd_std=None, Cp_std=None):
        """
        Args:
            J_B, J_n (np.array, optional): Jacobians with shape
                                           (n_epoch, n_obs, n_coeffs).
            W (np.array): Wavelet design matrix (N_nodes x N_coeffs).
            Cd_std (float or np.array, optional): Data noise std dev.
                                        - Scalar: applied to all obs/epochs.
                                        - (n_obs,): applied per epoch (tiled).
                                        - (n_epoch, n_obs): explicit map.
            Cp_std (float or np.array, optional): Prior std dev.
        """
        self.J_B_flat = None
        self.J_n_flat = None
        self.W = W
        self.Cd_inv_weight = None
        self.Cp_precision_diag = None

        if J_B is None and J_n is None:
            if Cd_std is not None or Cp_std is not None:
                raise ValueError("Cd_std and Cp_std require J_B and J_n.")
            return
        if J_B is None or J_n is None:
            raise ValueError("J_B and J_n must be provided together.")

        n_epochs, n_obs, n_coeffs_B = J_B.shape
        n_epochs_n, n_obs_n, n_coeffs_n = J_n.shape
        if (n_epochs_n, n_obs_n) != (n_epochs, n_obs):
            raise ValueError(
                "J_B and J_n must have the same epoch and observation dimensions."
            )
        
        # Flatten Jacobians by merging epochs and observations into matrix rows.
        self.J_B_flat = J_B.reshape(-1, n_coeffs_B)
        self.J_n_flat = J_n.reshape(-1, n_coeffs_n)
        
        if Cd_std is not None:
            if np.isscalar(Cd_std):
                self.Cd_inv_weight = np.full(self.J_B_flat.shape[0], 1.0/Cd_std)
            
            else:
                Cd_arr = np.array(Cd_std)
                if Cd_arr.ndim == 1 and Cd_arr.shape[0] == n_obs:
                    Cd_expanded = np.tile(Cd_arr, n_epochs)
                    self.Cd_inv_weight = 1.0 / Cd_expanded
                elif Cd_arr.shape == (n_epochs, n_obs):
                    self.Cd_inv_weight = 1.0 / Cd_arr.flatten()
                else:
                    raise ValueError(
                        f"Cd_std shape {Cd_arr.shape} inconsistent with Obs "
                        f"{n_obs} or Epochs {n_epochs}"
                    )

        if Cp_std is not None:
            n_total_coeffs = n_coeffs_B + n_coeffs_n
            if np.isscalar(Cp_std):
                self.Cp_precision_diag = np.full(n_total_coeffs, 1.0/Cp_std**2)
            else:
                self.Cp_precision_diag = 1.0 / np.array(Cp_std)**2
                if self.Cp_precision_diag.shape != (n_total_coeffs,):
                    raise ValueError(
                        f"Cp_std must be scalar or have shape ({n_total_coeffs},), "
                        f"got {np.array(Cp_std).shape}"
                    )

    def compute_crlb(
        self, m_coeffs, transform_func, transform_args=(), H=None, block="n", W=None
    ):
        """
        Computes CRLB for a parameter block using JAX to autodiff the transform.
        
        Args:
            m_coeffs (np.array): The MAP wavelet coefficients for the selected block.
            transform_func (callable): A function `f(latent_field, *args)` written 
                                       using jax.numpy operations.
            transform_args (tuple): Extra arguments for the transform (bounds, shifts, etc).
            H (np.array): Optional full posterior Hessian for the combined [B, n] system.
            block (str): Parameter block to project, either "B" or "n".
            W (np.array): Optional wavelet design matrix overriding the stored matrix.
            
        Returns:
            sigma_field (np.array): Physical uncertainty field for the selected block.
        """
        if block not in ("B", "n"):
            raise ValueError(f"block must be 'B' or 'n', got {block!r}")

        m_coeffs = np.asarray(m_coeffs)
        if m_coeffs.ndim != 1:
            raise ValueError(f"m_coeffs must be one-dimensional, got {m_coeffs.shape}")

        W_matrix = self.W if W is None else W
        if W_matrix is None:
            raise ValueError("compute_crlb requires W either in __init__ or as an argument.")
        W_matrix = np.asarray(W_matrix)
        if W_matrix.ndim != 2:
            raise ValueError(f"W must be two-dimensional, got {W_matrix.shape}")
        if W_matrix.shape[1] != m_coeffs.size:
            raise ValueError(
                f"W has {W_matrix.shape[1]} columns, but m_coeffs has "
                f"{m_coeffs.size} entries"
            )

        # 1. Hessian Inversion (Standard Step)
        # ---------------------------------------------------------
        if H is None:
            missing = []
            if self.J_B_flat is None or self.J_n_flat is None:
                missing.append("J_B/J_n")
            if self.Cd_inv_weight is None:
                missing.append("Cd_std")
            if self.Cp_precision_diag is None:
                missing.append("Cp_std")
            if missing:
                raise ValueError(
                    "H=None requires these values in __init__: "
                    + ", ".join(missing)
                )
            J_total = np.hstack([self.J_B_flat, self.J_n_flat])
            J_weighted = J_total * self.Cd_inv_weight[:, np.newaxis]
            H = J_weighted.T @ J_weighted
            np.fill_diagonal(H, np.diag(H) + self.Cp_precision_diag)
        else:
            H = np.asarray(H)
            if H.ndim != 2 or H.shape[0] != H.shape[1]:
                raise ValueError(f"H must be square, got {H.shape}")

        if self.J_B_flat is not None and self.J_n_flat is not None:
            block_starts = {"B": 0, "n": self.J_B_flat.shape[1]}
            block_sizes = {"B": self.J_B_flat.shape[1], "n": self.J_n_flat.shape[1]}
            if m_coeffs.size != block_sizes[block]:
                raise ValueError(
                    f"m_coeffs for block {block!r} must have "
                    f"{block_sizes[block]} entries, got {m_coeffs.size}"
                )
            block_start = block_starts[block]
            block_stop = block_start + block_sizes[block]
        elif block == "B":
            block_start = 0
            block_stop = m_coeffs.size
        else:
            block_start = H.shape[0] - m_coeffs.size
            block_stop = H.shape[0]

        if not 0 <= block_start < block_stop <= H.shape[0]:
            raise ValueError(
                f"Could not infer block {block!r} of size {m_coeffs.size} "
                f"from H shape {H.shape}"
            )
        
        c, lower = cho_factor(H, lower=True)
        Sigma_m = cho_solve((c, lower), np.eye(H.shape[0]))
        
        # Extract the selected parameter block.
        Sigma_coeffs = Sigma_m[block_start:block_stop, block_start:block_stop]
        
        # 2. JAX Autodiff Step
        # ---------------------------------------------------------
        # Reconstruct latent field z = W @ m
        latent_z = W_matrix @ m_coeffs
        
        # Define the scalar derivative function
        # We assume the transform is element-wise: physical_i = f(z_i)
        grad_func = jax.grad(transform_func, argnums=0)
        
        # Vectorize over the latent field while treating transform args as constants.
        vmap_grad_func = jax.vmap(
            grad_func, in_axes=(0,) + (None,) * len(transform_args)
        )
        
        # Compute the local derivatives (d_physical / d_latent)
        # We convert to JAX array for computation, then back to NumPy
        d_transform = np.array(vmap_grad_func(jnp.array(latent_z), *transform_args))
        
        # 3. Chain Rule Projection
        # ---------------------------------------------------------
        # Effective Jacobian: Scale W rows by the local derivative
        W_effective = W_matrix * d_transform[:, np.newaxis]
        
        # Project Variance: diag(W_eff @ Sigma @ W_eff.T)
        Sigma_W_T = Sigma_coeffs @ W_effective.T
        variance_field = np.sum(W_effective * Sigma_W_T.T, axis=1)
        
        return np.sqrt(variance_field)

    def compute_physical_metric(self, stress_fields, strain_rates, coords, smooth_radius):
        """
        Computes M(x) split into Spatial and Temporal components.
        
        Returns:
            M_total, M_spatial, M_temporal
        """
        print("Calculating Decomposed Physical Metric...")
        
        # 1. Prepare Data
        # log_tau shape: (N_nodes, N_epochs)
        log_tau = np.log(np.vstack(stress_fields).T + 1e-10)
        
        # Leverage (Strain Rate squared), averaged over time or sum
        # Using sum to be consistent with N_epochs accumulation
        eps_sq = np.vstack(strain_rates).T ** 2
        leverage = np.sum(eps_sq, axis=1) 
        
        # 2. Build KDTree for Spatial Neighborhoods
        tree = cKDTree(coords)
        indexes = tree.query_ball_point(coords, r=smooth_radius)
        
        var_spatial_arr = np.zeros(log_tau.shape[0])
        var_temporal_arr = np.zeros(log_tau.shape[0])
        
        # 3. Iterate over nodes
        for i, neighbor_idxs in enumerate(indexes):
            if len(neighbor_idxs) < 2:
                continue
                
            # Extract the patch: (N_neighbors, N_epochs)
            patch = log_tau[neighbor_idxs, :]
            
            # --- TERM 1: Spatial Variance (averaged over time) ---
            # Calculate variance across space for each epoch, then average
            # "How much contrast is there in a single snapshot?"
            spatial_vars = np.var(patch, axis=0) # Shape: (N_epochs,)
            term1_spatial = np.mean(spatial_vars)
            
            # --- TERM 2: Temporal Variance (of the spatial mean) ---
            # Calculate the spatial mean for each epoch
            # "How much did the average stress of this patch change over time?"
            spatial_means = np.mean(patch, axis=0) # Shape: (N_epochs,)
            term2_temporal = np.var(spatial_means)
            
            var_spatial_arr[i] = term1_spatial
            var_temporal_arr[i] = term2_temporal
            
        # 4. Scale by Leverage (Strain Rate)
        M_spatial = leverage * var_spatial_arr
        M_temporal = leverage * var_temporal_arr
        M_total = M_spatial + M_temporal
        
        return M_total, M_spatial, M_temporal, leverage


# end of file
