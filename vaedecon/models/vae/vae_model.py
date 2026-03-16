"""
Variational Autoencoder (VAE) implementation for cellular component deconvolution.
OnlyBelter (https://github.com/OnlyBelter, onlybelter@gmail.com)
"""

import os
import pandas as pd
import logging
from typing import Optional, List, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Gamma, kl_divergence

from ...configs import DataConfig, ModelConfig
from ...data.datasets import DatasetOutput

from ...models.base import BaseAE, reparameterize_dirichlet, reparameterize_gaussian, ModelOutput, BaseDecoder, BaseEncoder
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor, non_log2cpm_tensor

logger = logging.getLogger(__name__)


class VAE(BaseAE):
    """Variational Autoencoder model for cellular component deconvolution."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: Optional[DataConfig] = None,
        encoders: list[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        super().__init__(model_config=model_config,
                         data_config=data_config,
                         encoders=encoders,
                         decoder=decoder)

        self.n_encoders = len(self.encoders)
        assert self.n_encoders in (1, 2), 'Only 1 or 2 encoders are supported.'

        self.model_name = "VAE"
        latent_dim = model_config.latent_dim
        n_cell_types = model_config.n_cell_types

        # Constant for scaling
        self.scaling_factor = model_config.SCALING_FACTOR

        # --- Anchor & Prior Setup ---
        # Initialize orthonormal anchors for structured latent space
        self.anchor_vectors = self.make_orthonormal_anchors(latent_dim=latent_dim)
        self.register_buffer("anchors", self.anchor_vectors)

        # Logits to weight anchors (Learnable parameters to associate cell types with anchors)
        self.logits = nn.Parameter(torch.zeros(n_cell_types, latent_dim))

        # TODO: only read when it is necessary
        # --- Gene Statistics Setup ---
        if not os.path.exists(model_config.gene_mean_std_fp):
            raise FileNotFoundError(f"Gene features file not found: {model_config.gene_mean_std_fp}")

        # Optimization: Use usecols to reduce memory usage if possible
        gf_df = pd.read_csv(model_config.gene_mean_std_fp, index_col=0)
        # log2(CPM + 1) / scaling_factor space, so we can directly compare with reconstructions without extra transformations
        # Cell types have been ordered as the same order in the file "cell_type_list.txt", n_genes by n_cell_type
        g_mean = gf_df.loc[:, [c for c in gf_df.columns if c.endswith("avg")]].values
        g_std = gf_df.loc[:, [c for c in gf_df.columns if c.endswith("std")]].values

        g_mean = torch.tensor(g_mean, dtype=torch.float32)
        g_std = torch.tensor(g_std, dtype=torch.float32)
        g_mean_non_log = to_non_log_space(g_mean, self.scaling_factor)  # Convert mean GEP to non-log space
        g_std_non_log = to_non_log_space(g_std, self.scaling_factor)  # Convert gene std to non-log space

        # Register as buffers so they move to device automatically with the model
        self.register_buffer('g_mean', g_mean)
        self.register_buffer('g_std', g_std)
        self.register_buffer('g_mean_non_log', g_mean_non_log)
        self.register_buffer('g_std_non_log', g_std_non_log)

        # --- Gene Weights Calculation ---
        # Calculated once and fixed. If dynamic adjustment is needed, move to forward.
        w = torch.ones_like(self.g_mean)
        if self.model_config.loss_coefficient.get('weighting_gene_by_exp', False):
            w = self.compute_gene_weights(
                clamp_range=self.model_config.loss_coefficient.get('weight_clamp_range', (0.5, 2.0))
            )
        self.register_buffer('w', w)

        # Add mask_ratio from config or default to 0.0
        self.mask_ratio = getattr(model_config, 'mask_ratio', 0.0)

    def forward(self, inputs: DatasetOutput, **kwargs) -> ModelOutput:
        # 1. Get input and handle device automatically
        # Do not use self.to(device) manually inside forward; rely on input tensor device.
        x = inputs["data"]
        device = x.device
        batch_size = x.shape[0]

        y = inputs.get("labels")
        if y is not None:
            y = y.to(device)

        # ================== Random Gene Masking ==================
        # Only apply masking during training, not validation/testing
        if self.training and self.mask_ratio > 0:
            # Create a random mask: 1 = keep, 0 = mask
            # Shape: (Batch, Genes)
            mask = torch.rand_like(x) > self.mask_ratio

            # Apply mask: Zero out masked genes
            x_masked = x * mask.float()

            # (Optional) Scale remaining values to preserve magnitude
            # x_masked = x_masked / (1 - self.mask_ratio)

            # Use x_masked for the encoders
            x_input = x_masked
        else:
            x_input = x

        # 2. Encoders Forward Pass
        mu_list, logvar_list, prop_list, dd_alpha_list = [], [], [], []
        mu_mean_list, logvar_mean_list = [], []

        for encoder in self.encoders:
            out = encoder(x=x_input, y=y)
            mu_list.append(out.mu_all_types)  # (B, Latent, n_cell_types)
            logvar_list.append(out.logvar_all_types)
            mu_mean_list.append(out.mu_mean)  # (B, Latent)
            logvar_mean_list.append(out.logvar_mean)
            prop_list.append(out.cell_prop)  # (B, n_cell_types)

            if self.model_config.predict_cell_prop:
                # Dirichlet distribution parameters for cell type proportions
                dd_alpha_list.append(out.dd_alpha)

        # 3. Fusion (Single or Product of Experts)
        if self.n_encoders == 1:
            mu_types = mu_list[0]
            log_var_types = logvar_list[0]
            cell_prop = prop_list[0]
            mu_mean = mu_mean_list[0]
            logvar_mean = logvar_mean_list[0]
            dd_alpha = dd_alpha_list[0] if dd_alpha_list else None
        else:  # TODO: pleas check which fusion strategy is better, connect before calculating mu/logvar or after?
            # PoE Fusion for latent variables
            mu_types, log_var_types, mu_mean, logvar_mean = self._poe_fuse_per_celltype(
                mu_lists_celltype=mu_list,
                logvar_lists_celltype=logvar_list,
                mu_list_overall=mu_mean_list,
                logvar_list_overall=logvar_mean_list,
            )
            # Average Proportions (Linear Opinion Pool)
            cell_prop = torch.mean(torch.stack(prop_list, dim=0), dim=0)
            dd_alpha = torch.mean(torch.stack(dd_alpha_list, dim=0), dim=0) if dd_alpha_list else None

        n_cell_types = mu_types.shape[2]

        # 4. Reconstruction (Batch Decoding Optimization)
        # -------------------------------------------------------
        # Optimization: Merge Batch and CellType dimensions for one-time calculation
        # instead of looping through cell types.

        # Sample latent vectors: (B, Latent, C)
        # For each cell type k, sample z from a d_latent-dimensional diagonal Gaussian N(mu_k, diag(sigma_k^2))
        # via the reparameterization trick. Since the covariance is diagonal, this is equivalent to sampling
        # each of the d_latent dimensions independently from a 1D Gaussian, yielding n_cell_type independent
        # d_latent-dimensional latent representations with z_types of shape (batch_size, d_latent, n_cell_type).

        z_types = reparameterize_gaussian(mu_types, log_var_types)

        # Flatten for decoder: (B, Latent, C) -> (B, C, Latent) -> (B*C, Latent)
        # Ensure the shape is compatible with the decoder's input expectation
        z_types_flat = z_types.permute(0, 2, 1).reshape(-1, self.model_config.latent_dim)

        # One-time decoding: (B*C, Latent) -> (B*C, Genes)
        recon_flat = self.decoder(z_types_flat)["reconstruction"]

        # Restore shape: (B*C, Genes) -> (B, C, Genes) -> (B, Genes, C)
        # We need (B, Genes, C) for the subsequent matrix multiplication, ranging in (0, 1) after sigmoid activation.
        recon_x_all_types = recon_flat.view(batch_size, n_cell_types, -1).permute(0, 2, 1)
        # -------------------------------------------------------

        # 5. Scaling & Mixing

        # Log -> CPM (Batch, Genes, C)
        if not self.model_config.learn_gep_residual:
            if self.model_config.scaling_by_constant:  # Scale back up if we scaled down the input to get full GEP in log space
                recon_x_all_types = recon_x_all_types * self.scaling_factor  # From (0, 1) to (0, scaling_factor) range in log space
            # If not learning residual, decoder outputs full GEP in log space, so convert to CPM for mixing.
            recon_x_all_types_cpm = log_exp2cpm_tensor(recon_x_all_types, transpose=True)
        else:
            # If learning residual, decoder outputs residual z-score in range (-3, 3), so we need to multipl each value by the std and add back the mean GEP
            # before converting to CPM. We first convert both residual and mean GEP back to non-log space,
            # add them together to recover the full GEP, then normalize to CPM.

            # Redefine the min of z-score to guarantee all values >= 0 after adding mean GEP.
            eps = 1e-6
            mu = self.g_mean_non_log.unsqueeze(0)
            std = torch.clamp(self.g_std_non_log, min=eps).unsqueeze(0)
            z_min = torch.maximum(torch.full_like(std, -3.0), -mu / std)
            # Now recon_x_all_types is in range (z_min, 3)
            recon_x_all_types = z_min + (3.0 - z_min) * recon_x_all_types

            # Scale to (z_min*sigma, 3*sigma)
            recon_residual = recon_x_all_types * std  # Convert residual z-score to residual in non-log space by multiplying with std, shape (B, Genes, C)
            # Add mean GEP to residual in non-log space, and convert to (B, C, Genes) for the next step
            recon_x_add_g_mean = torch.transpose(recon_residual + self.g_mean_non_log, 1,2)
            # Normalize to CPM space for mixing, then transpose back to (B, Genes, C)
            recon_x_all_types_cpm = non_log2cpm_tensor(recon_x_add_g_mean).transpose(1, 2)

        # Prepare Proportions for Mixing
        if cell_prop is not None:
            # (B, C) -> (B, C, 1)
            prop_matrix = cell_prop.unsqueeze(-1)
        else:
            prop_matrix = torch.zeros((batch_size, n_cell_types, 1), device=device)

        # Mixing: (B, G, C) @ (B, C, 1) -> (B, G, 1)
        recon_x_conv = torch.bmm(recon_x_all_types_cpm, prop_matrix).squeeze(-1)

        # Gene Stats for Loss calculation
        recon_gene_mean = recon_x_all_types_cpm.mean(dim=0)  # (G, C)
        recon_gene_std = recon_x_all_types_cpm.std(dim=0)  # (G, C)

        # Convert back to log space for consistency with input features
        recon_gene_mean = torch.log2(recon_gene_mean + 1) / self.scaling_factor
        recon_gene_std = torch.log2(recon_gene_std + 1) / self.scaling_factor

        # Final Output Conversion
        recon_x_conv_log = non_log2log_cpm_tensor(recon_x_conv, transpose=False)
        if self.model_config.scaling_by_constant:
            recon_x_conv_log = recon_x_conv_log / self.scaling_factor

        # Ensure shape matches input x
        recon_x_conv_log = recon_x_conv_log.reshape(x.shape)

        # 6. Prior Calculation
        # Calculate the prior mean based on learned anchors
        anchor_weights = F.softmax(self.logits, dim=-1)  # (C, L)
        mu_prior = anchor_weights @ self.anchors  # (C, L)

        # 7. Loss Calculation
        loss_outputs = self.loss_function(
            x=x, y=y,
            logvar_types=log_var_types,
            mu_types=mu_types,
            dd_alpha=dd_alpha,
            recon_x_conv=recon_x_conv_log,
            beta=self.model_config.loss_coefficient['beta'],
            mu_prior=mu_prior,
            gamma=self.model_config.loss_coefficient['gamma'],
            recon_gene_mean=recon_gene_mean,
            recon_gene_std=recon_gene_std,
            logvar_mean=logvar_mean,
            mu_mean=mu_mean,
            device=device,
        )

        return ModelOutput(
            loss=loss_outputs[0],
            kld=loss_outputs[1],
            kld_p=loss_outputs[2],
            recon_loss_conv=loss_outputs[3],
            gene_mean_loss=loss_outputs[4],
            gene_std_loss=loss_outputs[5],
            repulsion_loss=loss_outputs[6],
            cell_prop_loss=loss_outputs[7],
            mu=mu_mean,
            mu_deconv=mu_types,
            log_var=log_var_types,
            pred_cell_prop=cell_prop,
            recon_x_conv=recon_x_conv_log,
            recon_x_all_types=recon_x_all_types_cpm,  # Usually return CPM format for analysis
        )

    def loss_function(
        self,
        x,
        recon_x_conv,
        mu_types,
        logvar_types,
        y,
        dd_alpha,
        mu_prior,
        beta,
        gamma,
        recon_gene_mean,
        recon_gene_std,
        logvar_mean,
        mu_mean,
        device,
        eps=1e-6,
    ):

        batch_size, latent_dim, n_cell_types = mu_types.shape
        lo = self.model_config.loss_coefficient

        # --- 1. Reconstruction Loss ---
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(recon_x_conv, x, reduction="none").sum(dim=-1)
        # elif self.model_config.reconstruction_loss == "bce":
            # recon_loss = F.binary_cross_entropy(recon_x_conv, x, reduction="none").sum(dim=-1)
        else:
            raise ValueError(f"Unknown resconstruction loss: {self.model_config.reconstruction_loss}, only MSE is supported")

        # --- 2. Gene Statistics Loss ---
        if lo.get('gene_mean_std_weight', 0) != 0:
            # Note: self.g_mean and self.w are buffers, so they are on the correct device
            gm_loss = F.mse_loss(recon_gene_mean, self.g_mean, reduction="none")
            gm_loss = (gm_loss * self.w).sum(dim=0).mean()  # Weighted sum over genes, mean over types
            gs_loss = F.mse_loss(recon_gene_std, self.g_std, reduction="none").sum(dim=0).mean()
        else:
            gm_loss = torch.tensor(0.0, device=device)
            gs_loss = torch.tensor(0.0, device=device)

        # --- 3. KL Divergence (Cell Proportions - Dirichlet) ---
        kld_p = torch.zeros(batch_size, device=device)
        cell_prop_loss = torch.zeros(batch_size, device=device)

        if y is not None and self.model_config.predict_cell_prop:
            # KL(Posterior || Prior)
            # Prior is Uniform Dirichlet (alpha=1)
            prior_alpha = torch.ones_like(dd_alpha)
            prior_dist = Dirichlet(prior_alpha)
            posterior_dist = Dirichlet(dd_alpha)
            kld_p = kl_divergence(posterior_dist, prior_dist)

            # Supervised Loss for Proportions
            normalized_dd_alpha = dd_alpha / torch.sum(dd_alpha, dim=-1, keepdim=True)
            cell_prop_loss = F.mse_loss(normalized_dd_alpha, y, reduction="none").sum(dim=-1)

        # --- 4. KL Divergence (Latent - Gaussian) ---
        kld_type = lo.get('kld_type', 'ave')

        if kld_type == 'sep':
            # Calculate KL for each Cell Type individually
            # mu_types: (B, L, C), logvar_types: (B, L, C)

            if mu_prior is not None:
                # Expand prior to match batch: (C, L) -> (1, L, C)
                mu_prior_expand = mu_prior.t().unsqueeze(0)

                # General KL for Gaussian:
                # KL = 0.5 * (sum(exp(logvar)) + (mu - mu_prior)^2 - L - sum(logvar))
                # Assuming Prior logvar is 0 (sigma=1)

                term1 = logvar_types.exp()  # sigma_1^2
                term2 = (mu_types - mu_prior_expand).pow(2)  # (mu_1 - mu_2)^2
                kld_z_types = 0.5 * torch.sum(term1 + term2 - 1 - logvar_types, dim=1).sum(dim=-1)
            else:
                # Standard N(0,1) prior
                kld_z_types = -0.5 * torch.sum(
                    1 + logvar_types - mu_types.pow(2) - logvar_types.exp(),
                    dim=1
                ).sum(dim=-1)

        elif kld_type == 'ave':
            # Use fused/averaged mu_mean, logvar_mean against Standard Normal
            log_var = logvar_mean
            mu = mu_mean
            kld_z_types = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        else:
            kld_z_types = torch.zeros(batch_size, device=device)

        # --- 5. Repulsion Loss ---
        repulsion_loss = torch.zeros(batch_size, device=device)
        if gamma > 0:
            # Calculate Euclidean distance between Cell Type Centroids
            mu_types_perm = mu_types.permute(0, 2, 1)  # (B, C, L)

            # Efficient distance calculation using torch.cdist
            dist_matrix = torch.cdist(mu_types_perm, mu_types_perm, p=2)  # (B, C, C)

            # Add Identity matrix * large number to avoid division by zero on the diagonal
            mask = torch.eye(n_cell_types, device=device).unsqueeze(0)
            dist_matrix = dist_matrix + mask * 1e9

            inv_dist = 1.0 / (dist_matrix + eps)
            # Zero out the diagonal contribution
            inv_dist = inv_dist * (1 - mask)

            repulsion_loss = inv_dist.sum(dim=(1, 2))

        # --- Total Loss ---
        total_loss = (
                recon_loss
                + beta * (kld_z_types + kld_p)
                + lo.get('cell_prop', 0) * cell_prop_loss
                + gamma * repulsion_loss
                + lo.get('gene_mean_std_weight', 0) * (gm_loss + gs_loss)
        ).mean()

        return (total_loss, kld_z_types.mean(), kld_p.mean(), recon_loss.mean(),
                gm_loss, gs_loss, repulsion_loss.mean(), cell_prop_loss.mean())

    def compute_gene_weights(
            self,
            low_weight_coef: float = 1.0,
            eps: float = 1e-6,
            clamp_range: Tuple[float, float] = (0.5, 2.0),
    ) -> torch.Tensor:
        """
        Compute per-gene weights for a batch of expression profiles, higher weights for low variance genes.
        """
        # 1) Choose raw weight w_g based on the variance of gene expression in each cell type
        w_g = 1.0 / (self.g_std + eps)  # (n_genes, n_cell_types)

        # 2) Normalize so that average weight = 1
        w_g = w_g / w_g.mean(dim=0, keepdim=True)

        # 3) Clamp to avoid extremes
        w_g = torch.clamp(w_g, min=clamp_range[0], max=clamp_range[1])

        # 4) Optionally up-weight low-expr genes more strongly
        if low_weight_coef != 1.0:
            w_g = w_g.pow(low_weight_coef)

        return w_g

    @staticmethod
    def make_orthonormal_anchors(latent_dim, radius=1.0):
        """
        Generate orthonormal anchors for the latent space.
        """
        random_weights = torch.randn(latent_dim, latent_dim)
        q, _ = torch.linalg.qr(random_weights)  # QR decomposition to get orthonormal vectors
        return radius * q.t()

    @staticmethod
    def _poe_fuse_core(
            mu_list: List[torch.Tensor],
            logvar_list: List[torch.Tensor],
            eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core Product of Experts (PoE) fusion logic.
        """
        if not mu_list or not logvar_list:
            raise ValueError("Input lists for PoE fusion cannot be empty.")
        if len(mu_list) != len(logvar_list):
            raise ValueError("Mismatch in the number of experts for mus and logvars.")
        if len(mu_list) == 1:
            return mu_list[0], logvar_list[0]

        # Stack expert parameters
        mus_stacked = torch.stack(mu_list, dim=0)
        logvars_stacked = torch.stack(logvar_list, dim=0)

        # Calculate precisions: P_i = 1 / sigma_i^2 = exp(-logvar_i)
        precisions_stacked = torch.exp(-logvars_stacked)

        # Sum of precisions: P_poe = sum(P_i)
        sum_of_precisions = torch.sum(precisions_stacked, dim=0)

        # Fused log variance: logvar_poe = -log(P_poe + eps)
        fused_logvar = -torch.log(sum_of_precisions + eps)

        # Weighted sum of means: sum(mu_i * P_i)
        sum_of_weighted_mus = torch.sum(mus_stacked * precisions_stacked, dim=0)

        # Fused mean: mu_poe = sum(mu_i * P_i) / (P_poe + eps)
        fused_mu = sum_of_weighted_mus / (sum_of_precisions + eps)

        return fused_mu, fused_logvar

    def _poe_fuse_per_celltype(
            self,
            mu_lists_celltype: List[torch.Tensor],
            logvar_lists_celltype: List[torch.Tensor],
            mu_list_overall: List[torch.Tensor],
            logvar_list_overall: List[torch.Tensor],
            eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform PoE fusion for per-celltype latent parameters and averaging for overall parameters.
        """
        # --- PoE fusion for per-celltype parameters ---
        fused_mu_celltype, fused_logvar_celltype = self._poe_fuse_core(
            mu_list=mu_lists_celltype,
            logvar_list=logvar_lists_celltype,
            eps=eps
        )

        # --- Averaging for overall parameters ---
        if not mu_list_overall or not logvar_list_overall:
            raise ValueError("Overall mean/logvar lists cannot be empty for averaging.")

        if len(mu_list_overall) == 1:
            avg_mu_overall = mu_list_overall[0]
            avg_logvar_overall = logvar_list_overall[0]
        else:
            # Stack for averaging
            mu_overall_stacked = torch.stack(mu_list_overall, dim=0)
            avg_mu_overall = torch.mean(mu_overall_stacked, dim=0)

            logvar_overall_stacked = torch.stack(logvar_list_overall, dim=0)
            avg_logvar_overall = torch.mean(logvar_overall_stacked, dim=0)

        return fused_mu_celltype, fused_logvar_celltype, avg_mu_overall, avg_logvar_overall


def to_non_log_space(
    x_log_scaled: torch.Tensor,
    scaling_factor: float,
    *,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert scaled-log2 features to non-log space.

    Expected transform pair:
        x_log_scaled = log2(x_non_log + 1) / scaling_factor
        x_non_log    = 2^(x_log_scaled * scaling_factor) - 1

    Args:
        x_log_scaled:
            Tensor in scaled log2 space.
        scaling_factor:
            Positive scaling factor used in preprocessing.
        clamp_min:
            Optional lower bound after inverse transform.
            Keep 0.0 for expression-like non-negative values.

    Returns:
        Tensor in non-log space.

    Notes:
        - Uses torch.pow for consistency with your current code.
        - Clamp avoids tiny negative values from numerical noise.
    """
    if scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be > 0, got {scaling_factor}")

    x_non_log = torch.pow(2.0, x_log_scaled * scaling_factor) - 1.0
    if clamp_min is not None:
        x_non_log = torch.clamp(x_non_log, min=clamp_min)
    return x_non_log


def to_log_space(
    x_non_log: torch.Tensor,
    scaling_factor: float,
    *,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert non-log features back to scaled-log2 space.

    Expected transform pair:
        x_log_scaled = log2(x_non_log + 1) / scaling_factor
        x_non_log    = 2^(x_log_scaled * scaling_factor) - 1

    Args:
        x_non_log:
            Tensor in non-log space (typically >= 0).
        scaling_factor:
            Positive scaling factor used in preprocessing.
        clamp_min:
            Clamp input before log to avoid invalid values (default 0).

    Returns:
        Tensor in scaled log2 space.
    """
    if scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be > 0, got {scaling_factor}")

    x_non_log = torch.clamp(x_non_log, min=clamp_min)
    x_log_scaled = torch.log2(x_non_log + 1.0) / scaling_factor
    return x_log_scaled
