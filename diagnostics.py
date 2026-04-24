import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

import jax
import jax.numpy as jnp


class IceRheologyStatistics:
    def __init__(self, J_B, J_n, W, Cd_std, Cp_std):
        """
        Args:
            J_B, J_n (np.array): Jacobians with shape (n_epoch, n_obs, n_coeffs).
            W (np.array): Wavelet design matrix (N_nodes x N_coeffs).
            Cd_std (float or np.array): Data noise std dev. 
                                        - Scalar: applied to all obs/epochs.
                                        - (n_obs,): applied per epoch (tiled).
                                        - (n_epoch, n_obs): explicit map.
            Cp_std (float or np.array): Prior std dev.
        """
        # 1. Flatten Jacobians: (n_epoch, n_obs, n_coeffs) -> (n_total_obs, n_coeffs)
        # We merge epochs and observations into the 'rows' of the matrix
        n_epochs, n_obs, n_coeffs_B = J_B.shape
        _, _, n_coeffs_n = J_n.shape
        
        self.J_B_flat = J_B.reshape(-1, n_coeffs_B)
        self.J_n_flat = J_n.reshape(-1, n_coeffs_n)
        self.W = W
        
        # 2. Handle Cd (Noise) - Match the flattened shape (n_epochs * n_obs)
        if np.isscalar(Cd_std):
            # Scalar broadcasts implicitly later
            self.Cd_inv_weight = np.full(self.J_B_flat.shape[0], 1.0/Cd_std)
            
        else:
            Cd_arr = np.array(Cd_std)
            if Cd_arr.ndim == 1 and Cd_arr.shape[0] == n_obs:
                # User provided 1D array for one epoch; tile it for all epochs
                Cd_expanded = np.tile(Cd_arr, n_epochs)
                self.Cd_inv_weight = 1.0 / Cd_expanded
            elif Cd_arr.shape == (n_epochs, n_obs):
                # User provided explicit noise for every epoch
                self.Cd_inv_weight = 1.0 / Cd_arr.flatten()
            else:
                raise ValueError(f"Cd_std shape {Cd_arr.shape} inconsistent with Obs {n_obs} or Epochs {n_epochs}")

        # 3. Handle Cp (Prior precision diagonal)
        n_total_coeffs = n_coeffs_B + n_coeffs_n
        if np.isscalar(Cp_std):
            self.Cp_precision_diag = np.full(n_total_coeffs, 1.0/Cp_std**2)
        else:
            self.Cp_precision_diag = 1.0 / np.array(Cp_std)**2

    def compute_crlb(self, m_coeffs, transform_func, transform_args=(), H=None, block="n"):
        """
        Computes CRLB for a parameter block using JAX to autodiff the transform.
        
        Args:
            m_coeffs (np.array): The MAP wavelet coefficients for the selected block.
            transform_func (callable): A function `f(latent_field, *args)` written 
                                       using jax.numpy operations.
            transform_args (tuple): Extra arguments for the transform (bounds, shifts, etc).
            H (np.array): Optional full posterior Hessian for the combined [B, n] system.
            block (str): Parameter block to project, either "B" or "n".
            
        Returns:
            sigma_field (np.array): Physical uncertainty field for the selected block.
        """
        block_starts = {
            "B": 0,
            "n": self.J_B_flat.shape[1],
        }
        block_sizes = {
            "B": self.J_B_flat.shape[1],
            "n": self.J_n_flat.shape[1],
        }
        if block not in block_starts:
            raise ValueError(f"block must be 'B' or 'n', got {block!r}")

        block_start = block_starts[block]
        block_stop = block_start + block_sizes[block]
        m_coeffs = np.asarray(m_coeffs)
        if m_coeffs.shape != (block_sizes[block],):
            raise ValueError(
                f"m_coeffs for block {block!r} must have shape "
                f"({block_sizes[block]},), got {m_coeffs.shape}"
            )
        if self.W.shape[1] != block_sizes[block]:
            raise ValueError(
                f"W has {self.W.shape[1]} columns, but block {block!r} has "
                f"{block_sizes[block]} coefficients"
            )

        # 1. Hessian Inversion (Standard Step)
        # ---------------------------------------------------------
        if H is None:
            J_total = np.hstack([self.J_B_flat, self.J_n_flat])
            J_weighted = J_total * self.Cd_inv_weight[:, np.newaxis]
            H = J_weighted.T @ J_weighted
            np.fill_diagonal(H, np.diag(H) + self.Cp_precision_diag)
        
        c, lower = cho_factor(H, lower=True)
        Sigma_m = cho_solve((c, lower), np.eye(H.shape[0]))
        
        # Extract the selected parameter block.
        Sigma_coeffs = Sigma_m[block_start:block_stop, block_start:block_stop]
        
        # 2. JAX Autodiff Step
        # ---------------------------------------------------------
        # Reconstruct latent field z = W @ m
        latent_z = self.W @ m_coeffs
        
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
        W_effective = self.W * d_transform[:, np.newaxis]
        
        # Project Variance: diag(W_eff @ Sigma @ W_eff.T)
        Sigma_W_T = Sigma_coeffs @ W_effective.T
        variance_n_field = np.sum(W_effective * Sigma_W_T.T, axis=1)
        
        return np.sqrt(variance_n_field)

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
