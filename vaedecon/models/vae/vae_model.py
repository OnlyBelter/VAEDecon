"""
Variational Autoencoder (VAE) implementation for cellular component deconvolution.
OnlyBelter (https://github.com/OnlyBelter, onlybelter@gmail.com)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd
import logging
from typing import Optional, List, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from ...configs import DataConfig, ModelConfig
from ...data.datasets import DatasetOutput
from ...models.base import (
    BaseAE,
    reparameterize_gaussian,
    dirichlet_mean,
    ModelOutput,
    BaseDecoder,
    BaseEncoder,
    EPS,
    has_usable_labels,
)
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor, non_log2cpm_tensor
from ...utility.hierarchical_encoding import HIERARCHICAL_ENCODING

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
    attractor: torch.Tensor = torch.tensor(0.0)
    z_score_reciprocal: torch.Tensor = torch.tensor(0.0)  # Optional term for std regularization
    z_score_kl_loss: torch.Tensor = torch.tensor(0.0)  # New KL term for empirical z-score to N(0,1)
    low_mean_std_gene_loss: torch.Tensor = torch.tensor(0.0)  # Optional term to prevent collapse of low-mean/std genes
    hierarchical_code: torch.Tensor = torch.tensor(0.0)


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
        if self.n_encoders not in (1, 2, 3):
            raise ValueError(f"Only 1, 2, or 3 encoders are supported, got {self.n_encoders}.")

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
        self.hierarchical_code_head = nn.Linear(latent_dim, 8)

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
        avg_cols = [c for c in gf_df.columns if c.endswith("avg")]
        std_cols = [c for c in gf_df.columns if c.endswith("std")]
        self.cell_types = [
            c[:-4] if c.endswith("_avg") else (c[:-4] if c.endswith(" avg") else c[:-3])
            for c in avg_cols
        ]
        g_mean_np = gf_df.loc[:, avg_cols].values
        g_std_np = gf_df.loc[:, std_cols].values

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

        hierarchical_targets = []
        missing = []
        for ct in self.cell_types:
            code = HIERARCHICAL_ENCODING.get(ct)
            if code is None:
                missing.append(ct)
                code = [0] * 8
            hierarchical_targets.append(code)
        if missing:
            raise ValueError(f"Missing hierarchical_encoding for cell types: {missing}")
        self.register_buffer(
            "hierarchical_code_targets",
            torch.tensor(hierarchical_targets, dtype=torch.float32),
        )

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

    def save(
        self,
        model_dir: str,
        training_config: Optional["TrainingConfig"] = None,
        data_config: Optional[DataConfig] = None,
    ):
        super().save(model_dir=model_dir, training_config=training_config, data_config=data_config)

        try:
            lo = self.model_config.loss_coefficient
            low_mean_threshold = getattr(lo, "low_mean_threshold", 2.0)
            low_std_threshold = getattr(lo, "low_std_threshold", 1.0)

            low_mean_mask = self.g_mean_non_log < low_mean_threshold
            low_std_mask = self.g_std_non_log < low_std_threshold
            low_mean_or_std_mask = low_mean_mask | low_std_mask

            low_mean_counts = low_mean_mask.sum(dim=0).to(torch.int64).cpu().tolist()
            low_std_counts = low_std_mask.sum(dim=0).to(torch.int64).cpu().tolist()
            low_mean_or_std_counts = low_mean_or_std_mask.sum(dim=0).to(torch.int64).cpu().tolist()

            n_genes = int(self.g_mean_non_log.shape[0])
            n_cell_types = int(self.g_mean_non_log.shape[1])
            cell_type_labels = (
                self.cell_types
                if isinstance(getattr(self, "cell_types", None), list) and len(self.cell_types) == n_cell_types
                else [f"cell_type_{i}" for i in range(n_cell_types)]
            )

            df = pd.DataFrame(
                {
                    "cell_type": cell_type_labels,
                    "low_mean_count": low_mean_counts,
                    "low_std_count": low_std_counts,
                    "low_mean_or_std_count": low_mean_or_std_counts,
                    "n_genes": n_genes,
                    "low_mean_threshold": low_mean_threshold,
                    "low_std_threshold": low_std_threshold,
                }
            )
            out_fp = os.path.join(model_dir, "low_mean_std_gene_counts.csv")
            df.to_csv(out_fp, index=False)
            logger.info("Saved low-mean/low-std gene counts to %s", out_fp)
        except Exception as e:
            logger.warning("Failed to save low-mean/low-std gene counts: %s", e)

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
        if has_usable_labels(y):
            y = y.to(device)
        else:
            y = None

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
            mu = self.g_mean_non_log.unsqueeze(0)                           # (1, G, C)
            std = torch.clamp(self.g_std_non_log, min=EPS).unsqueeze(0)     # (1, G, C)
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
            recon_x_all_types_cpm=recon_x_all_types_cpm,
        )

        return ModelOutput(
            loss=loss_terms.total,
            kld=loss_terms.kld_z,
            kld_p=loss_terms.kld_p,
            recon_loss_conv=loss_terms.recon,
            gene_mean_loss=loss_terms.gm,
            gene_std_loss=loss_terms.gs,
            repulsion_loss=loss_terms.repulsion,
            attractor_loss=loss_terms.attractor,
            hierarchical_code_loss=loss_terms.hierarchical_code,
            cell_prop_loss=loss_terms.cell_prop,
            z_score_kl_loss=loss_terms.z_score_kl_loss,
            low_mean_std_gene_loss=loss_terms.low_mean_std_gene_loss,
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
        recon_x_all_types_cpm: Optional[torch.Tensor] = None,
    ) -> LossTerms:
        """
        Compute all objective terms and aggregate total loss.
        Parameters:
            x: Input features (B, G, C), in log space after scaling by constant
            y: Optional target labels (B, C)
            recon_x_conv: Reconstructed bulk GEPs (B, G), in log space after scaling by constant
            mu_types: Latent space mean (B, L, C)
            logvar_types: Latent space log-variance (B, L, C)
            mu_mean: Prior mean (B, L)
            logvar_mean: Prior log-variance (B, L)
            recon_gene_mean: Reconstructed gene mean per cell type across samples in a batch (G, C), in tpm space
            recon_gene_std: Reconstructed gene std per cell type across samples in a batch (G, C), in tpm space
            recon_x_all_types_cpm: GEPs of all cell types after deconvolution (B, G, C), in tpm space
            dd_alpha: Dirichlet distribution parameter (C,) for cell type proportions, if applicable.
            mu_prior: Prior mean (C, L) for latent space, if applicable.
            device: Device for tensor operations

        Shapes:
            x, recon_x_conv:      (B, G)
            mu_types/logvar_types:(B, L, C)
            mu_mean/logvar_mean:  (B, L)
            recon_gene_mean/std:  (G, C)
        """
        batch_size, n_gene = x.shape
        lo = self.model_config.loss_coefficient
        beta = lo.beta
        gamma = lo.gamma
        attractor_weight = lo.attractor_weight
        z_score_reg_weight = lo.z_score_reg_weight
        labels_available = has_usable_labels(y)

        if (
            self.training
            and self.model_config.predict_cell_prop
            and lo.cell_prop > 0
            and not labels_available
        ):
            raise ValueError(
                "predict_cell_prop=True with loss_coefficient.cell_prop > 0 "
                "requires cell-fraction labels during training."
            )

        # Optional regularization to prevent std from collapsing to zero.
        # Add 1 / mean_z_scores to the loss to encourage the model to keep larger z-scores (and thus std) from collapsing.
        if z_score_reg_weight > 0 and self.model_config.learn_gep_residual:
            mean_z_scores = self.z_scores.abs().mean(dim=(1, 2))  # (B,)
        else:
            mean_z_scores = torch.ones((batch_size,), device=device)
            # z_score_reg_weight = 0.0  # No regularization if not learning residuals.

        # --- 1. Reconstruction Loss ---
        recon_loss = self._reconstruction_loss(x=x, recon_x_conv=recon_x_conv)              # (B,)

        # --- 2. Gene Statistics Loss ---
        # If z-score KL regularization is enabled, ignore the original gene mean/std losses.
        if lo.z_score_kl_weight > 0:
            gm_loss = torch.tensor(0.0, device=device)
            gs_loss = torch.tensor(0.0, device=device)
        else:
            # Convert back to log space for consistency with input features.
            recon_gene_mean = to_log_space(recon_gene_mean, self.scaling_factor)
            recon_gene_std = to_log_space(recon_gene_std, self.scaling_factor)
            gm_loss, gs_loss = self._gene_statistics_loss(
                recon_gene_mean=recon_gene_mean,
                recon_gene_std=recon_gene_std,
                device=device,
            )                                                                               # scalar, scalar

        # --- 2.5 Z-score KL Loss ---
        if lo.z_score_kl_weight > 0:
            if recon_x_all_types_cpm is None:
                raise ValueError("recon_x_all_types_cpm is required when z_score_kl_weight > 0")
            z_score_kl_loss = self._z_score_kl_loss(recon_x_all_types_cpm)
        else:
            z_score_kl_loss = torch.tensor(0.0, device=device)

        # Identify genes with low mean or low std,
        # whose z-scores are unreliable for KL regularization.
        # Replace their predicted values with the reference mean plus small Gaussian noise.
        low_mean_threshold = getattr(lo, "low_mean_threshold", 2.0)
        low_std_threshold = getattr(lo, "low_std_threshold", 1.0)
        mask = (self.g_mean_non_log < low_mean_threshold) | (self.g_std_non_log < low_std_threshold)  # (G, C)
        mask = mask.unsqueeze(0).expand(batch_size, n_gene, -1)  # (B, G, C)

        g_mean_expanded = self.g_mean_non_log.unsqueeze(0).expand(batch_size, n_gene, -1)  # (B, G, C)
        # g_std_expanded = self.g_std_non_log.unsqueeze(0).expand(batch_size, n_gene, -1)  # (B, G, C)
        g_mean_expanded_log = to_log_space(g_mean_expanded, self.scaling_factor)  # (B, G, C)
        recon_x_all_types_log = to_log_space(recon_x_all_types_cpm, self.scaling_factor)  # (B, G, C)
        mask_f = mask.to(dtype=recon_x_all_types_log.dtype)
        diff2 = (recon_x_all_types_log - g_mean_expanded_log).pow(2) * mask_f
        denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)
        low_mean_std_gene_loss_per_sample = diff2.sum(dim=(1, 2)) / denom  # (B,)

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
        )                                                                                   # (B,)

        attractor_loss = self._attractor_loss(
            mu_types=mu_types,
            attractor_weight=attractor_weight,
        )                                                                                   # (B,)

        hierarchical_code_weight = lo.hierarchical_code_weight
        hierarchical_code_loss = self._hierarchical_code_loss(
            mu_types=mu_types,
            hierarchical_code_weight=hierarchical_code_weight,
        )                                                                                   # (B,)

        # --- Total Loss ---
        total_loss = (
            recon_loss
            + lo.low_mean_std_weight * low_mean_std_gene_loss_per_sample
            + beta * kld_z_types
            + lo.kld_p * kld_p
            + lo.cell_prop * cell_prop_loss
            + gamma * repulsion_loss
            + attractor_weight * attractor_loss
            + hierarchical_code_weight * hierarchical_code_loss
            # + lo.gene_mean_weight * gm_loss
            # + lo.gene_std_weight * gs_loss
            + lo.z_score_kl_weight * z_score_kl_loss
            # + z_score_reg_weight * (1 / mean_z_scores)
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
            attractor=attractor_loss.mean(),
            hierarchical_code=hierarchical_code_loss.mean(),
            z_score_reciprocal=(1 / mean_z_scores).mean(),
            z_score_kl_loss=z_score_kl_loss.mean(),
            low_mean_std_gene_loss=low_mean_std_gene_loss_per_sample.mean(),
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
        if lo.gene_mean_weight == 0 and lo.gene_std_weight == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Note: self.g_mean and self.w are buffers, so they are on the correct device.
        gm_loss = F.mse_loss(recon_gene_mean, self.g_mean, reduction="none")
        gm_loss = (gm_loss * self.w).sum(dim=0).mean()  # Weighted sum over genes, mean over types
        gs_loss = F.mse_loss(recon_gene_std, self.g_std, reduction="none").sum(dim=0).mean()
        return gm_loss, gs_loss

    def _z_score_kl_loss(self, recon_x_all_types_cpm: torch.Tensor) -> torch.Tensor:
        """
        Regularize predicted per-cell-type GEPs by matching their empirical z-score distribution
        (computed across samples in the current batch) to a standard normal N(0, 1).

        recon_x_all_types_cpm: (B, G, C) in non-log CPM/TPM-like space.
        Z[b, g, c] = (X[b, g, c] - mean[g, c]) / std[g, c]

        Per-celltype overall normality:
        For each cell type c, we aggregate z-scores over all samples and genes in the batch
        and approximate the empirical distribution as N(mu_c, var_c). We then compute
        KL(N(mu_c, var_c) || N(0, 1)) and average over cell types.
        """
        if recon_x_all_types_cpm.ndim != 3:
            raise ValueError(
                f"Expected recon_x_all_types_cpm with shape (B, G, C), got {recon_x_all_types_cpm.shape}"
            )

        B, G, C = recon_x_all_types_cpm.shape
        assert self.g_mean_non_log.shape == (G, C), (
            f"g_mean_non_log shape mismatch: expected ({G}, {C}), "
            f"got {self.g_mean_non_log.shape}"
        )

        denom = torch.clamp(self.g_std_non_log, min=0.1).unsqueeze(0)  # (1, G, C)
        mu = self.g_mean_non_log.unsqueeze(0)                          # (1, G, C)
        z = (recon_x_all_types_cpm - mu) / denom                       # (B, G, C)
        z = torch.clamp(z, max=10, min=-10)  # Prevent extreme z-scores from destabilizing KL calculation.

        # Per-celltype: compute mu_z[C] and var_z[C] across (B, G)
        # (B, G, C) -> (C) after mean over B and G
        mu_z = z.mean(dim=(0, 1))                                       # (C,)
        var_z = z.var(dim=(0, 1), unbiased=False).clamp_min(EPS)        # (C,)

        # KL(N(mu, var) || N(0, 1)) = 0.5 * (var + mu^2 - 1 - log(var))
        kl_per_celltype = 0.5 * (var_z + mu_z.pow(2) - 1.0 - torch.log(var_z))  # (C,)
        kl_per_celltype = kl_per_celltype.clamp_min(0.0)
        kl = kl_per_celltype.mean()                                    # scalar

        # Expand back to batch dimension (B,) for consistency with existing code
        return kl.expand(B)

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

        if self.model_config.predict_cell_prop and dd_alpha is not None:
            # KL(Posterior || Prior), prior is Uniform Dirichlet(alpha=1)
            prior_alpha = torch.ones_like(dd_alpha)
            prior_dist = Dirichlet(prior_alpha)
            posterior_dist = Dirichlet(dd_alpha)
            kld_p = kl_divergence(posterior_dist, prior_dist)

        if has_usable_labels(y) and self.model_config.predict_cell_prop and dd_alpha is not None:
            # Supervised loss for proportions
            normalized_dd_alpha = dirichlet_mean(dd_alpha)
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

    @staticmethod
    def _repulsion_loss(
            mu_types: torch.Tensor,
            gamma: float,
            margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Repulsion between cell-type centroids in latent space.
        Uses a cosine-similarity hinge penalty to encourage orthogonality between
        different cell-type embeddings.

        Loss per pair:  max(0, cos(i, j) - margin)
        → zero gradient when cosine similarity is <= margin (default 0).
        → linear penalty when embeddings become more similar (cosine increases).

        Args:
            mu_types : (B, L, C)  cell-type centroid means in latent space.
            gamma    : float       loss weight; if 0, returns zeros immediately.
            margin   : float       maximum allowed cosine similarity between any
                                  two centroids (default 0.0 for orthogonality).

        Returns:
            repulsion_loss : (B,)  per-sample repulsion scalar.
        """
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device

        if gamma == 0 or n_cell_types < 2:
            return torch.zeros(batch_size, device=device)

        x = mu_types.permute(0, 2, 1)  # (B, C, L)
        x = F.normalize(x, p=2, dim=2, eps=EPS)
        cos_matrix = x @ x.transpose(1, 2)  # (B, C, C)

        # Upper triangle mask — count each pair (i, j) only once
        triu_mask = torch.triu(
            torch.ones(n_cell_types, n_cell_types, device=device), diagonal=1
        ).unsqueeze(0)  # (1, C, C)

        # Hinge penalty: penalize only pairs with cosine similarity above margin
        hinge = torch.clamp(cos_matrix - margin, min=0.0)  # (B, C, C)
        hinge = hinge * triu_mask  # upper triangle only

        # Normalize by number of pairs so loss scale is independent of C
        n_pairs = n_cell_types * (n_cell_types - 1) / 2
        repulsion_loss = hinge.sum(dim=(1, 2)) / n_pairs  # (B,)

        return repulsion_loss

    def _attractor_loss(
        self,
        mu_types: torch.Tensor,
        attractor_weight: float,
    ) -> torch.Tensor:
        """
        Attractor loss: L2-ball constraint around the origin for cell-type embeddings.
        → zero gradient when embeddings are inside the ball.
        → linear penalty when embeddings are outside the ball.
        Returns per-sample vector, shape (B,).
        
        Args:
            mu_types : (B, L, C)  cell-type centroid means in latent space.
            attractor_weight : float       loss weight; if 0, returns zeros immediately.
        """
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device

        if attractor_weight == 0 or n_cell_types < 2:
            return torch.zeros((batch_size,), device=device)

        x = mu_types.permute(0, 2, 1)  # (B, C, L)
        l2_norm = torch.linalg.vector_norm(x, ord=2, dim=2)  # (B, C)
        radius = 1.0
        violation = torch.clamp(l2_norm - radius, min=0.0)  # (B, C)
        loss = violation.mean(dim=1)  # (B,)
        return loss

    def _hierarchical_code_loss(
        self,
        mu_types: torch.Tensor,
        hierarchical_code_weight: float,
    ) -> torch.Tensor:
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device

        if hierarchical_code_weight == 0 or n_cell_types < 1:
            return torch.zeros((batch_size,), device=device)

        targets = getattr(self, "hierarchical_code_targets", None)
        if targets is None or targets.shape[0] != n_cell_types:
            raise ValueError("hierarchical_code_targets is missing or has wrong shape")

        x = mu_types.permute(0, 2, 1)  # (B, C, L)
        logits = self.hierarchical_code_head(x)  # (B, C, 8)
        y = targets.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (B, C, 8)
        loss_mat = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # (B, C, 8)
        return loss_mat.mean(dim=(1, 2))

    # =========================================================================
    # Gene Weighting
    # =========================================================================
    def compute_gene_weights(
        self,
        low_weight_coef: float = 1.0,
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
        w_g = 1.0 / (self.g_std + EPS)  # (G, C)

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
            )
            fused_mu_overall, fused_logvar_overall = self._poe_fuse_core(
                mu_list=mu_list_overall,
                logvar_list=logvar_list_overall,
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

        # Stack expert parameters. [n_encoder, batch_size, latent_dim, n_cell_type]
        mus_stacked = torch.stack(mu_list, dim=0)
        logvars_stacked = torch.stack(logvar_list, dim=0)

        # Calculate precisions: P_i = 1 / sigma_i^2 = exp(-logvar_i).
        precisions_stacked = torch.exp(-logvars_stacked)

        # Sum of precisions: P_poe = sum(P_i). [batch_size, latent_dim, n_cell_type]
        sum_of_precisions = torch.sum(precisions_stacked, dim=0)

        # Fused log variance: logvar_poe = -log(P_poe + eps).
        fused_logvar = -torch.log(sum_of_precisions + EPS)

        # Weighted sum of means: sum(mu_i * P_i).
        sum_of_weighted_mus = torch.sum(mus_stacked * precisions_stacked, dim=0)

        # Fused mean: mu_poe = sum(mu_i * P_i) / (P_poe + eps) = sum_of_weighted_mus * e^{fused_logvar}.
        fused_mu = sum_of_weighted_mus * torch.exp(fused_logvar)

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
