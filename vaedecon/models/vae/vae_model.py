"""
Variational Autoencoder (VAE) implementation for cellular component deconvolution.
OnlyBelter (https://github.com/OnlyBelter, onlybelter@gmail.com)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd
import logging
from typing import Optional, List, Tuple, Union, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from ...configs import DataConfig, ModelConfig
from ...data.datasets import DatasetOutput
from ...models.base import BaseAE, reparameterize_gaussian, ModelOutput, BaseDecoder, BaseEncoder
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor, non_log2cpm_tensor

logger = logging.getLogger(__name__)


@dataclass
class LossTerms:
    """
    Container for decomposed loss terms.

    Notes:
    - `total` is a scalar (already reduced by batch mean).
    - Most other fields are scalar means for logging.
    """
    total: torch.Tensor
    kld_z: torch.Tensor
    kld_p: torch.Tensor
    recon: torch.Tensor
    gm: torch.Tensor
    gs: torch.Tensor
    repulsion: torch.Tensor
    cell_prop: torch.Tensor
    z_score_reciprocal: torch.Tensor = torch.tensor(0.0)  # Optional term for std regularization


class VAE(BaseAE):
    """
    Variational Autoencoder model for cellular component deconvolution.

    Design highlights:
    1) Supports 1 or 2 encoders.
    2) Uses posterior-level fusion (default PoE) when multiple encoders are provided.
    3) Maintains gene-statistic targets (mean/std) as registered buffers.
    4) Keeps reconstruction in biologically meaningful spaces (log/non-log/CPM) with explicit transforms.

    Why posterior-level fusion by default?
    - Fusing `mu/logvar` keeps uncertainty information from each encoder.
    - Product-of-Experts (PoE) has a clear probabilistic interpretation:
      precision-weighted combination.
    - In practice this is usually more stable/interpretable than feature fusion before latent heads.

    Config note:
    - `model_config.fusion_strategy` can be:
      - "poe_posterior" (recommended default)
      - "avg_posterior" (simple baseline)
      - "pre_latent" (placeholder: requires architecture-level support in encoders)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: Optional[DataConfig] = None,
        encoders: Optional[List[BaseEncoder]] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        super().__init__(
            model_config=model_config,
            data_config=data_config,
            encoders=encoders,
            decoder=decoder,
        )

        # ---------------------------------------------------------------------
        # Basic sanity checks
        # ---------------------------------------------------------------------
        self.z_scores = None
        self.n_encoders = len(self.encoders)
        if self.n_encoders not in (1, 2):
            raise ValueError(f"Only 1 or 2 encoders are supported, got {self.n_encoders}.")

        self.model_name = "VAE"
        self.scaling_factor = float(data_config.scaling_factor)
        if not self.data_config.scaling_by_constant:
            self.scaling_factor = 1.0  # No scaling if not specified in data config
        self.mask_ratio = float(getattr(model_config, "mask_ratio", 0.0))

        # Fusion strategy for multiple encoders
        # Recommended: poe_posterior
        self.fusion_strategy: Literal["poe_posterior", "avg_posterior", "pre_latent"] = getattr(
            model_config, "fusion_strategy", "poe_posterior"
        )
        if self.fusion_strategy not in ("poe_posterior", "avg_posterior", "pre_latent"):
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        latent_dim = int(model_config.latent_dim)
        n_cell_types = int(model_config.n_cell_types)

        # ---------------------------------------------------------------------
        # --- Anchor & Prior Setup ---
        # Initialize orthonormal anchors for structured latent space.
        # ---------------------------------------------------------------------
        anchor_vectors = self.make_orthonormal_anchors(latent_dim=latent_dim)
        self.register_buffer("anchors", anchor_vectors)

        # Logits to weight anchors (Learnable parameters to associate cell types with anchors)
        self.logits = nn.Parameter(torch.zeros(n_cell_types, latent_dim))

        # ---------------------------------------------------------------------
        # --- Gene Statistics Setup ---
        # TODO: only read when it is necessary (lazy init possible in future).
        # ---------------------------------------------------------------------
        gene_fp = model_config.gene_mean_std_fp
        if not os.path.exists(gene_fp):
            raise FileNotFoundError(f"Gene features file not found: {gene_fp}")

        # Optimization: Use usecols if file schema is fixed enough.
        gf_df = pd.read_csv(gene_fp, index_col=0)

        # log2(CPM + 1) / scaling_factor space, so we can directly compare with reconstructions without extra transformations.
        # Cell types are expected in the same order as "cell_type_list.txt".
        g_mean_np = gf_df.loc[:, [c for c in gf_df.columns if c.endswith("avg")]].values
        g_std_np = gf_df.loc[:, [c for c in gf_df.columns if c.endswith("std")]].values

        g_mean = torch.tensor(g_mean_np, dtype=torch.float32)
        g_std = torch.tensor(g_std_np, dtype=torch.float32)

        # Convert mean/std GEP to non-log space for residual-mode reconstruction.
        g_mean_non_log = to_non_log_space(g_mean, self.scaling_factor)
        g_std_non_log = to_non_log_space(g_std, self.scaling_factor)

        # Register as buffers so they move to device automatically with the model.
        self.register_buffer("g_mean", g_mean)
        self.register_buffer("g_std", g_std)
        self.register_buffer("g_mean_non_log", g_mean_non_log)
        self.register_buffer("g_std_non_log", g_std_non_log)

        # ---------------------------------------------------------------------
        # --- Gene Weights Calculation ---
        # Calculated once and fixed. If dynamic adjustment is needed, move to forward().
        # ---------------------------------------------------------------------
        w = torch.ones_like(self.g_mean)
        if self.model_config.loss_coefficient.weighting_gene_by_exp:
            w = self.compute_gene_weights(
                clamp_range=self.model_config.loss_coefficient.weight_clamp_range
            )
        self.register_buffer("w", w)

    # =========================================================================
    # Forward
    # =========================================================================
    def forward(self, inputs: "DatasetOutput", **kwargs) -> "ModelOutput":
        """
        Forward pass with optional random gene masking, encoder fusion, decoding, and loss computation.

        Input:
            inputs["data"]: (B, G)
            inputs.get("labels"): (B, C) or None
        """
        # 1. Get input and handle device automatically.
        # Do not use self.to(device) manually inside forward; rely on input tensor device.
        x = inputs["data"]
        device = x.device
        batch_size = x.shape[0]

        y = inputs.get("labels")
        if y is not None:
            y = y.to(device)

        # ================== Random Gene Masking ==================
        # Only apply masking during training, not validation/testing.
        if self.training and self.mask_ratio > 0:
            # Create a random mask: 1 = keep, 0 = mask
            # Shape: (Batch, Genes)
            mask = torch.rand_like(x) > self.mask_ratio

            # Apply mask: Zero out masked genes
            x_masked = x * mask.float()

            # (Optional) Scale remaining values to preserve magnitude.
            # x_masked = x_masked / (1 - self.mask_ratio)

            # Use x_masked for the encoders.
            x_input = x_masked
        else:
            x_input = x

        # 2. Encoders Forward Pass
        mu_list, logvar_list = [], []
        mu_mean_list, logvar_mean_list = [], []
        prop_list, dd_alpha_list = [], []

        for encoder in self.encoders:
            out = encoder(x=x_input, y=y)
            mu_list.append(out.mu_all_types)                # (B, L, C)
            logvar_list.append(out.logvar_all_types)        # (B, L, C)
            mu_mean_list.append(out.mu_mean)                # (B, L)
            logvar_mean_list.append(out.logvar_mean)        # (B, L)
            prop_list.append(out.cell_prop)                 # (B, C)

            if self.model_config.predict_cell_prop:
                # Dirichlet distribution parameters for cell type proportions.
                dd_alpha_list.append(out.dd_alpha)          # (B, C)

        # 3. Fusion (Single or Multi-Encoder)
        if self.n_encoders == 1:
            mu_types = mu_list[0]
            log_var_types = logvar_list[0]
            mu_mean = mu_mean_list[0]
            logvar_mean = logvar_mean_list[0]
            cell_prop = prop_list[0]
            dd_alpha = dd_alpha_list[0] if dd_alpha_list else None
        else:
            # If multiple encoders exist, fuse at posterior level by default.
            mu_types, log_var_types, mu_mean, logvar_mean = self._fuse_encoder_posteriors(
                mu_lists_celltype=mu_list,
                logvar_lists_celltype=logvar_list,
                mu_list_overall=mu_mean_list,
                logvar_list_overall=logvar_mean_list,
                strategy=self.fusion_strategy,
            )

            # Proportions: simple linear opinion pool (average) as baseline.
            cell_prop = torch.mean(torch.stack(prop_list, dim=0), dim=0)
            dd_alpha = torch.mean(torch.stack(dd_alpha_list, dim=0), dim=0) if dd_alpha_list else None

        n_cell_types = mu_types.shape[2]

        # 4. Reconstruction (Batch Decoding Optimization)
        # -------------------------------------------------------
        # Optimization: Merge Batch and CellType dimensions for one-time decoding
        # instead of looping through cell types.

        # Sample latent vectors: (B, L, C)
        # For each cell type k, sample z from diagonal Gaussian N(mu_k, diag(sigma_k^2))
        # via reparameterization trick.
        z_types = reparameterize_gaussian(mu_types, log_var_types)

        # Flatten for decoder: (B, L, C) -> (B, C, L) -> (B*C, L)
        z_types_flat = z_types.permute(0, 2, 1).reshape(-1, self.model_config.latent_dim)

        # One-time decoding: (B*C, L) -> (B*C, G)
        recon_flat = self.decoder(z_types_flat)["reconstruction"]

        # Restore shape: (B*C, G) -> (B, C, G) -> (B, G, C)
        # We need (B, G, C) for subsequent matrix multiplication.
        recon_x_all_types = recon_flat.view(batch_size, n_cell_types, -1).permute(0, 2, 1)
        # -------------------------------------------------------

        # 5. Scaling & Mixing

        # Log -> CPM (Batch, Genes, C)
        if not self.model_config.learn_gep_residual:
            if self.data_config.scaling_by_constant:
                # Scale back up if input was scaled down.
                recon_x_all_types = recon_x_all_types * self.scaling_factor
            # Decoder outputs full GEP in log space -> convert to CPM for mixing.
            recon_x_all_types_cpm = log_exp2cpm_tensor(recon_x_all_types, transpose=True)
        else:
            # If learning residual, decoder outputs residual z-score in range (-3, 3).
            # We multiply by std and add mean GEP in non-log space, then normalize to CPM.

            # Redefine minimum z-score to guarantee all values >= 0 after adding mean GEP.
            eps = 1e-6
            mu = self.g_mean_non_log.unsqueeze(0)                           # (1, G, C)
            std = torch.clamp(self.g_std_non_log, min=eps).unsqueeze(0)     # (1, G, C)
            z_min = torch.maximum(torch.full_like(std, -3.0), -mu / std)
            # Now recon_x_all_types is in range (z_min, 3).
            recon_x_all_types = z_min + (3.0 - z_min) * recon_x_all_types
            self.z_scores = recon_x_all_types  # For loss calculation: store the actual z-scores.

            # Convert residual z-score to residual in non-log space.
            recon_residual = recon_x_all_types * std                         # (B, G, C)

            # Add mean GEP in non-log space; transpose to (B, C, G) for non_log2cpm_tensor.
            recon_x_add_g_mean = torch.transpose(recon_residual + self.g_mean_non_log, 1, 2)

            # Normalize to CPM, then transpose back to (B, G, C).
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
        recon_gene_mean = recon_x_all_types_cpm.mean(dim=0)                 # (G, C)
        recon_gene_std = recon_x_all_types_cpm.std(dim=0)                   # (G, C)

        # Final Output Conversion
        recon_x_conv_log = non_log2log_cpm_tensor(recon_x_conv, transpose=False)
        if self.data_config.scaling_by_constant:
            recon_x_conv_log = recon_x_conv_log / self.scaling_factor

        # Ensure shape matches input x.
        recon_x_conv_log = recon_x_conv_log.reshape(x.shape)

        # 6. Prior Calculation
        # Calculate prior mean based on learned anchors.
        anchor_weights = F.softmax(self.logits, dim=-1)                     # (C, L)
        mu_prior = anchor_weights @ self.anchors                             # (C, L)

        # 7. Loss Calculation
        loss_terms = self.loss_function(
            x=x,
            y=y,
            recon_x_conv=recon_x_conv_log,
            mu_types=mu_types,
            logvar_types=log_var_types,
            dd_alpha=dd_alpha,
            mu_prior=mu_prior,
            recon_gene_mean=recon_gene_mean,
            recon_gene_std=recon_gene_std,
            logvar_mean=logvar_mean,
            mu_mean=mu_mean,
            device=device,
        )

        return ModelOutput(
            loss=loss_terms.total,
            kld=loss_terms.kld_z,
            kld_p=loss_terms.kld_p,
            recon_loss_conv=loss_terms.recon,
            gene_mean_loss=loss_terms.gm,
            gene_std_loss=loss_terms.gs,
            repulsion_loss=loss_terms.repulsion,
            cell_prop_loss=loss_terms.cell_prop,
            mu=mu_mean,
            mu_deconv=mu_types,
            log_var=log_var_types,
            pred_cell_prop=cell_prop,
            recon_x_conv=recon_x_conv_log,
            recon_x_all_types=recon_x_all_types_cpm,  # Usually return CPM format for analysis
            z_score_reciprocal=loss_terms.z_score_reciprocal,  # For monitoring potential z-score collapse when learning residuals
        )

    # =========================================================================
    # Loss Function
    # =========================================================================
    def loss_function(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        recon_x_conv: torch.Tensor,
        mu_types: torch.Tensor,
        logvar_types: torch.Tensor,
        dd_alpha: Optional[torch.Tensor],
        mu_prior: Optional[torch.Tensor],
        recon_gene_mean: torch.Tensor,
        recon_gene_std: torch.Tensor,
        logvar_mean: torch.Tensor,
        mu_mean: torch.Tensor,
        device: torch.device,
        eps: float = 1e-6,
    ) -> LossTerms:
        """
        Compute all objective terms and aggregate total loss.

        Shapes:
            x, recon_x_conv:      (B, G)
            mu_types/logvar_types:(B, L, C)
            mu_mean/logvar_mean:  (B, L)
            recon_gene_mean/std:  (G, C)
        """
        batch_size = mu_types.shape[0]
        lo = self.model_config.loss_coefficient
        beta = lo.beta
        gamma = lo.gamma
        z_score_reg_weight = lo.z_score_reg_weight

        # Optional regularization to prevent std from collapsing to zero.
        # Add 1 / mean_z_scores to the loss to encourage the model to keep larger z-scores (and thus std) from collapsing.
        if z_score_reg_weight > 0 and self.model_config.learn_gep_residual:
            mean_z_scores = self.z_scores.abs().mean(dim=(1, 2))  # (B,)
        else:
            mean_z_scores = torch.ones((batch_size,), device=device)
            z_score_reg_weight = 0.0  # No regularization if not learning residuals.

        # Convert back to log space for consistency with input features.
        recon_gene_mean = to_log_space(recon_gene_mean, self.scaling_factor)
        recon_gene_std = to_log_space(recon_gene_std, self.scaling_factor)

        # --- 1. Reconstruction Loss ---
        recon_loss = self._reconstruction_loss(x=x, recon_x_conv=recon_x_conv)              # (B,)

        # --- 2. Gene Statistics Loss ---
        gm_loss, gs_loss = self._gene_statistics_loss(
            recon_gene_mean=recon_gene_mean,
            recon_gene_std=recon_gene_std,
            device=device,
        )                                                                                   # scalar, scalar

        # --- 3. KL Divergence (Cell Proportions - Dirichlet) ---
        kld_p, cell_prop_loss = self._cell_prop_dirichlet_loss(
            y=y,
            dd_alpha=dd_alpha,
            batch_size=batch_size,
            device=device,
        )                                                                                   # (B,), (B,)

        # --- 4. KL Divergence (Latent - Gaussian) ---
        kld_z_types = self._latent_kld_loss(
            mu_types=mu_types,
            logvar_types=logvar_types,
            mu_prior=mu_prior,
            logvar_mean=logvar_mean,
            mu_mean=mu_mean,
            device=device,
        )                                                                                   # (B,)

        # --- 5. Repulsion Loss ---
        repulsion_loss = self._repulsion_loss(
            mu_types=mu_types,
            gamma=gamma,
            eps=eps,
        )                                                                                   # (B,)

        # --- Total Loss ---
        total_loss = (
            recon_loss
            + beta * (kld_z_types + kld_p)
            + lo.cell_prop * cell_prop_loss
            + gamma * repulsion_loss
            + lo.gene_mean_std_weight * (gm_loss + gs_loss)
            + z_score_reg_weight * (1 / mean_z_scores)
        ).mean()

        return LossTerms(
            total=total_loss,
            kld_z=kld_z_types.mean(),
            kld_p=kld_p.mean(),
            recon=recon_loss.mean(),
            gm=gm_loss,
            gs=gs_loss,
            repulsion=repulsion_loss.mean(),
            cell_prop=cell_prop_loss.mean(),
            z_score_reciprocal=(1 / mean_z_scores).mean(),
        )

    # -------------------------------------------------------------------------
    # Loss Components
    # -------------------------------------------------------------------------
    def _reconstruction_loss(self, x: torch.Tensor, recon_x_conv: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction loss, shape (B,)."""
        if self.model_config.reconstruction_loss == "mse":
            return F.mse_loss(recon_x_conv, x, reduction="none").sum(dim=-1)
        # elif self.model_config.reconstruction_loss == "bce":
        #     return F.binary_cross_entropy(recon_x_conv, x, reduction="none").sum(dim=-1)
        raise ValueError(
            f"Unknown reconstruction loss: {self.model_config.reconstruction_loss}, only MSE is supported"
        )

    def _gene_statistics_loss(
        self,
        recon_gene_mean: torch.Tensor,
        recon_gene_std: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gene statistic losses (scalars):
        - GM: weighted MSE for mean
        - GS: MSE for std
        """
        lo = self.model_config.loss_coefficient
        if lo.gene_mean_std_weight == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Note: self.g_mean and self.w are buffers, so they are on the correct device.
        gm_loss = F.mse_loss(recon_gene_mean, self.g_mean, reduction="none")
        gm_loss = (gm_loss * self.w).sum(dim=0).mean()  # Weighted sum over genes, mean over types
        gs_loss = F.mse_loss(recon_gene_std, self.g_std, reduction="none").sum(dim=0).mean()
        return gm_loss, gs_loss

    def _cell_prop_dirichlet_loss(
        self,
        y: Optional[torch.Tensor],
        dd_alpha: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dirichlet KL + supervised cell proportion loss.
        Returns per-sample vectors: (kld_p, cell_prop_loss), both shape (B,).
        """
        kld_p = torch.zeros(batch_size, device=device)
        cell_prop_loss = torch.zeros(batch_size, device=device)

        if y is not None and self.model_config.predict_cell_prop and dd_alpha is not None:
            # KL(Posterior || Prior), prior is Uniform Dirichlet(alpha=1)
            prior_alpha = torch.ones_like(dd_alpha)
            prior_dist = Dirichlet(prior_alpha)
            posterior_dist = Dirichlet(dd_alpha)
            kld_p = kl_divergence(posterior_dist, prior_dist)

            # Supervised loss for proportions
            normalized_dd_alpha = dd_alpha / torch.sum(dd_alpha, dim=-1, keepdim=True).clamp_min(1e-8)
            cell_prop_loss = F.mse_loss(normalized_dd_alpha, y, reduction="none").sum(dim=-1)

        return kld_p, cell_prop_loss

    def _latent_kld_loss(
        self,
        mu_types: torch.Tensor,
        logvar_types: torch.Tensor,
        mu_prior: Optional[torch.Tensor],
        logvar_mean: torch.Tensor,
        mu_mean: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Latent KL term.
        - kld_type == 'sep': KL computed per-celltype posterior (B, L, C)
        - kld_type == 'ave': KL computed from fused/averaged posterior (B, L)
        Returns per-sample vector, shape (B,).
        """
        lo = self.model_config.loss_coefficient
        kld_type = lo.kld_type
        batch_size = mu_types.shape[0]

        if kld_type == "sep":
            # Calculate KL for each Cell Type individually.
            # mu_types/logvar_types: (B, L, C)
            if mu_prior is not None:
                # Expand prior to match batch: (C, L) -> (1, L, C)
                mu_prior_expand = mu_prior.t().unsqueeze(0)

                # General KL for Gaussian:
                # KL = 0.5 * sum(exp(logvar) + (mu - mu_prior)^2 - 1 - logvar)
                # Assuming prior logvar is 0 (sigma=1).
                term1 = logvar_types.exp()                           # sigma_1^2
                term2 = (mu_types - mu_prior_expand).pow(2)         # (mu_1 - mu_2)^2
                return 0.5 * torch.sum(term1 + term2 - 1 - logvar_types, dim=1).sum(dim=-1)

            # Standard N(0,1) prior
            return -0.5 * torch.sum(
                1 + logvar_types - mu_types.pow(2) - logvar_types.exp(),
                dim=1
            ).sum(dim=-1)

        if kld_type == "ave":
            # Use fused/averaged mu_mean, logvar_mean against Standard Normal.
            return -0.5 * torch.sum(1 + logvar_mean - mu_mean.pow(2) - logvar_mean.exp(), dim=-1)

        return torch.zeros(batch_size, device=device)

    def _repulsion_loss(
        self,
        mu_types: torch.Tensor,
        gamma: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Repulsion between cell-type centroids in latent space.
        Returns per-sample vector, shape (B,).
        """
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device
        repulsion_loss = torch.zeros(batch_size, device=device)

        if gamma > 0:
            # Calculate Euclidean distance between Cell Type Centroids.
            mu_types_perm = mu_types.permute(0, 2, 1)  # (B, C, L)

            # Efficient distance calculation using torch.cdist.
            dist_matrix = torch.cdist(mu_types_perm, mu_types_perm, p=2)  # (B, C, C)

            # Add identity * large number to avoid division by zero on diagonal.
            mask = torch.eye(n_cell_types, device=device).unsqueeze(0)
            dist_matrix = dist_matrix + mask * 1e9

            inv_dist = 1.0 / (dist_matrix + eps)
            # Zero out diagonal contribution.
            inv_dist = inv_dist * (1 - mask)

            repulsion_loss = inv_dist.sum(dim=(1, 2))

        return repulsion_loss

    # =========================================================================
    # Gene Weighting
    # =========================================================================
    def compute_gene_weights(
        self,
        low_weight_coef: float = 1.0,
        eps: float = 1e-6,
        clamp_range: Tuple[float, float] = (0.5, 2.0),
    ) -> torch.Tensor:
        """
        Compute per-gene weights, higher weights for lower-variance genes.

        Steps:
        1) Raw weight from inverse std.
        2) Normalize so average weight per cell type = 1.
        3) Clamp to avoid extremes.
        4) Optional exponent scaling.
        """
        # 1) Choose raw weight based on variance of gene expression in each cell type.
        w_g = 1.0 / (self.g_std + eps)  # (G, C)

        # 2) Normalize so that average weight = 1.
        w_g = w_g / w_g.mean(dim=0, keepdim=True)

        # 3) Clamp to avoid extremes.
        w_g = torch.clamp(w_g, min=clamp_range[0], max=clamp_range[1])

        # 4) Optionally re-shape low-expression emphasis.
        if low_weight_coef != 1.0:
            w_g = w_g.pow(low_weight_coef)

        return w_g

    # =========================================================================
    # Anchors
    # =========================================================================
    @staticmethod
    def make_orthonormal_anchors(latent_dim: int, radius: float = 1.0) -> torch.Tensor:
        """
        Generate orthonormal anchors for latent space via QR decomposition.
        """
        random_weights = torch.randn(latent_dim, latent_dim)
        q, _ = torch.linalg.qr(random_weights)
        return radius * q.t()

    # =========================================================================
    # Encoder fusion
    # =========================================================================
    def _fuse_encoder_posteriors(
        self,
        mu_lists_celltype: List[torch.Tensor],
        logvar_lists_celltype: List[torch.Tensor],
        mu_list_overall: List[torch.Tensor],
        logvar_list_overall: List[torch.Tensor],
        strategy: Literal["poe_posterior", "avg_posterior", "pre_latent"],
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse multiple encoder outputs.

        Returns:
            fused_mu_celltype:      (B, L, C)
            fused_logvar_celltype:  (B, L, C)
            fused_mu_overall:       (B, L)
            fused_logvar_overall:   (B, L)
        """
        if strategy == "pre_latent":
            # Placeholder:
            # Pre-latent fusion requires explicit architecture support in encoders
            # (e.g., shared fusion block before mu/logvar heads).
            # We fallback to posterior PoE for safety.
            strategy = "poe_posterior"

        if strategy == "poe_posterior":
            fused_mu_celltype, fused_logvar_celltype = self._poe_fuse_core(
                mu_list=mu_lists_celltype,
                logvar_list=logvar_lists_celltype,
                eps=eps,
            )
            fused_mu_overall, fused_logvar_overall = self._poe_fuse_core(
                mu_list=mu_list_overall,
                logvar_list=logvar_list_overall,
                eps=eps,
            )
            return fused_mu_celltype, fused_logvar_celltype, fused_mu_overall, fused_logvar_overall

        if strategy == "avg_posterior":
            # Simple arithmetic averaging baseline.
            fused_mu_celltype = torch.mean(torch.stack(mu_lists_celltype, dim=0), dim=0)
            fused_logvar_celltype = torch.mean(torch.stack(logvar_lists_celltype, dim=0), dim=0)
            fused_mu_overall = torch.mean(torch.stack(mu_list_overall, dim=0), dim=0)
            fused_logvar_overall = torch.mean(torch.stack(logvar_list_overall, dim=0), dim=0)
            return fused_mu_celltype, fused_logvar_celltype, fused_mu_overall, fused_logvar_overall

        raise ValueError(f"Unknown fusion strategy: {strategy}")

    @staticmethod
    def _poe_fuse_core(
        mu_list: List[torch.Tensor],
        logvar_list: List[torch.Tensor],
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core Product-of-Experts (PoE) fusion logic.

        For diagonal Gaussians:
            precision_i = exp(-logvar_i)
            precision_sum = sum_i precision_i
            mu_fused = sum_i(mu_i * precision_i) / precision_sum
            logvar_fused = -log(precision_sum)
        """
        if not mu_list or not logvar_list:
            raise ValueError("Input lists for PoE fusion cannot be empty.")
        if len(mu_list) != len(logvar_list):
            raise ValueError("Mismatch in number of experts for mus and logvars.")
        if len(mu_list) == 1:
            return mu_list[0], logvar_list[0]

        # Stack expert parameters.
        mus_stacked = torch.stack(mu_list, dim=0)
        logvars_stacked = torch.stack(logvar_list, dim=0)

        # Calculate precisions: P_i = 1 / sigma_i^2 = exp(-logvar_i).
        precisions_stacked = torch.exp(-logvars_stacked)

        # Sum of precisions: P_poe = sum(P_i).
        sum_of_precisions = torch.sum(precisions_stacked, dim=0)

        # Fused log variance: logvar_poe = -log(P_poe + eps).
        fused_logvar = -torch.log(sum_of_precisions + eps)

        # Weighted sum of means: sum(mu_i * P_i).
        sum_of_weighted_mus = torch.sum(mus_stacked * precisions_stacked, dim=0)

        # Fused mean: mu_poe = sum(mu_i * P_i) / (P_poe + eps).
        fused_mu = sum_of_weighted_mus / (sum_of_precisions + eps)

        return fused_mu, fused_logvar


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
