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
from torch.cpu import current_device
# from sympy.core.facts import deduce_alpha_implications
from torch.distributions import Normal, Dirichlet, Gamma, kl_divergence

from ...data.datasets import DatasetOutput
# from ...models.base.base_utils import ModelOutput

from ...models.base import BaseAE, reparameterize_dirichlet, reparameterize_gaussian, ModelOutput
from ...models.nn import BaseDecoder, BaseEncoder
# from ...models.gnn import EncoderGNN, EncoderSGNN
# from ...models.base.base_config import BaseModelConfig
from .vae_config import VAEConfig
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class VAE(BaseAE):
    """Variational Autoencoder model for cellular component deconvolution."""

    def __init__(
        self,
        model_config: VAEConfig,
        encoders: list[BaseDecoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        """Initializes the VAE model.

        Args:
            model_config: The VAE configuration.
            encoders: A list with a single encoder (MLP or GNN) or two encoders (MLP and GNN).
            decoder: An optional decoder.
        """
        super().__init__(model_config=model_config, encoders=encoders, decoder=decoder)
        # set the encoder list and count the number of encoders
        # self.encoders = encoders
        self.n_encoders = len(self.encoders)
        assert self.n_encoders in (1, 2), 'Only 1 or 2 encoders are supported.'
        self.device_param = nn.Parameter(torch.empty(0))  # To easily get the device of the model

        self.model_name = "VAE"
        latent_dim = model_config.latent_dim
        n_cell_types = model_config.n_cell_types
        # (latent_dim, latent_dim), orthonormal row vectors in latent space
        self.anchor_vectors = self.make_orthonormal_anchors(latent_dim=latent_dim)
        self.register_buffer("anchors", self.anchor_vectors)
        # Learn a logit for each cell type to weight orthonormal anchors in the latent space
        logits = nn.Parameter(torch.zeros(n_cell_types, latent_dim))
        # Gene features (mean and std) for each gene across cell types buffer
        if not os.path.exists(model_config.gene_mean_std_fp):
            raise FileNotFoundError(f"Gene features file not found: {model_config.gene_mean_std_fp}")
        gf_df = pd.read_csv(model_config.gene_mean_std_fp, index_col=0)
        g_mean = gf_df.loc[:, [col for col in gf_df.columns if col.endswith("avg")]].values  # shape = (n_genes, n_cell_types)
        g_std = gf_df.loc[:, [col for col in gf_df.columns if col.endswith("std")]].values  # shape = (n_genes, n_cell_types)
        # gf_mat = gf_df.loc[self.keep_genes].values  # shape = (n_genes, n_feats)
        self.register_buffer('g_mean', torch.tensor(g_mean, dtype=torch.float32))
        self.register_buffer('g_std', torch.tensor(g_std, dtype=torch.float32))
        self.register_buffer('logits', logits)  # (n_cell_types, latent_dim)

        # Calculate gene weights based on the mean expression values across cell types
        w = torch.ones_like(self.g_mean, device=self.device)
        if self.model_config.loss_coefficient['weighting_gene_by_exp']:
            weight_clamp_range = self.model_config.loss_coefficient['weight_clamp_range']
            # construct weights for each gene across cell types, (batch_size, n_genes, n_cell_types)
            w = self.compute_gene_weights(
                low_weight_coef=1.0,
                eps=1e-6,
                clamp_range=weight_clamp_range,
            )
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))  # (n_genes, n_cell_types)


    def forward(self, inputs: DatasetOutput, **kwargs) -> ModelOutput:
        """Forward pass of the VAE model.

        Args:
            inputs: The input data to the model.

        Returns:
            An ModelOutput instance containing the model's output.
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_device = self.device_param.device
        self.to(current_device)
        x = inputs["data"].to(current_device)
        batch_size = x.shape[0]
        y = inputs.get("labels")  # cell proportions of 16 cell types
        if y is not None:
            y = y.to(current_device)

        # Call all encoders, and collect the outputs
        # prop_list is the predicted cell proportions if self.model_config.predict_cell_prop is True, otherwise it is y
        mu_list, logvar_list, prop_list, dd_alpha_list, mu_mean_list, logvar_mean_list = [], [], [], [], [], []
        dd_alpha = torch.zeros(batch_size, self.model_config.n_cell_types, device=current_device)
        for encoder in self.encoders:
            out = encoder(x=x, y=y)
            mu_list.append(out.mu_all_types)
            # mu_lists.append(out.mu_all_types)
            logvar_list.append(out.logvar_all_types)
            # logvar_lists.append(out.log_var)
            mu_mean_list.append(out.mu_mean)
            logvar_mean_list.append(out.logvar_mean)
            prop_list.append(out.cell_prop)
            if self.model_config.predict_cell_prop:
                dd_alpha_list.append(out.dd_alpha)
        if self.n_encoders == 1:
            mu_types, log_var_types, cell_prop, mu_mean, logvar_mean = (
                mu_list[0],
                logvar_list[0],
                prop_list[0],
                mu_mean_list[0],
                logvar_mean_list[0],
            )
            if self.model_config.predict_cell_prop:
                dd_alpha = dd_alpha_list[0]  # (batch_size, n_cell_types)
        else:  # two encoders
            mu_types, log_var_types, mu_mean, logvar_mean = self._poe_fuse_per_celltype(
                mu_lists_celltype=mu_list,
                logvar_lists_celltype=logvar_list,
                mu_list_overall=mu_mean_list,
                logvar_list_overall=logvar_mean_list,
            )
            cell_prop = torch.mean(torch.stack(prop_list, dim=0), dim=0)  # (batch_size, n_cell_types)
            if self.model_config.predict_cell_prop:
                dd_alpha = torch.mean(torch.stack(dd_alpha_list, dim=0), dim=0)

        # # mu_types = torch.stack(mu_list, dim=2)  # (batch_size, latent_dim, n_cell_types)
        # # log_var_types = torch.stack(logvar_list, dim=2)  # (batch_size, latent_dim, n_cell_types)
        # # mu_types = mu_list  # (batch_size, latent_dim, n_cell_types)
        # # log_var = logvar_list  # (batch_size, latent_dim), all cell types shared
        # # pred_cell_prop = None
        # dd_alpha = torch.zeros(self.model_config.n_cell_types)
        # if self.model_config.predict_cell_prop:
        #     if self.n_encoders == 1:
        #         dd_alpha = dd_alpha_list[0]
        #     else:  # two encoders
        #         dd_alpha = torch.mean(torch.stack(dd_alpha_list, dim=0), dim=0) # (batch_size, n_cell_types)
        n_cell_types = mu_types.shape[2]


        # reconstructing GEPs for all cell types
        # Sample latent type representations for each cell type
        z_type_list = [
            # reparameterize_gaussian(mu_types[:, :, i], log_var)
            reparameterize_gaussian(mu_types[:, :, i], log_var_types[:, :, i])
            for i in range(n_cell_types)
        ]  # list of tensors (batch_size, latent_dim)
        # Decode each latent type representation to its corresponding GEP
        recon_type_gep_list = [
            self.decoder(z_type)["reconstruction"].squeeze() for z_type in z_type_list
        ]  # list of tensors (batch_size, n_genes)
        # Stack the reconstructed GEPs for all cell types, shape: (batch_size, n_genes, n_cell_types)
        recon_x_all_types = torch.stack(recon_type_gep_list, dim=2)

        # recon_x_all_types should be recovered to CPM format before doing the matrix multiplication
        if self.model_config.scaling_by_constant:
            recon_x_all_types = recon_x_all_types * 20.0
        recon_x_all_types = log_exp2cpm_tensor(recon_x_all_types, transpose=True)
        cell_prop = cell_prop.reshape(-1, n_cell_types, 1).to(current_device)  # (batch_size, n_cell_types, 1)
        # y = y.reshape(-1, n_cell_types, 1).to(device)  # (batch_size, n_cell_types, 1)
        recon_x_conv = torch.bmm(recon_x_all_types, cell_prop)  # (batch_size, n_genes, 1)

        # Compare the means and stds of each gene among reconstructed GEPs across cell types
        recon_gene_mean = recon_x_all_types.mean(dim=0)  # (n_genes, n_cell_types)
        recon_gene_std = recon_x_all_types.std(dim=0)  # (n_genes, n_cell_types)
        # Convert to the same format as the gene features
        recon_gene_mean = torch.log2(recon_gene_mean + 1) / 20.0
        recon_gene_std = torch.log2(recon_gene_std + 1) / 20.0

        # if y is not None:
        #     recon_x_conv = torch.matmul(recon_x_all_types, y.reshape(-1, n_cell_types, 1))
        # elif pred_cell_prop is not None:
        #     recon_x_conv = torch.matmul(recon_x_all_types, pred_cell_prop.reshape(-1, n_cell_types, 1))
        # else:
        #     raise ValueError("y or pred_cell_prop must be provided.")
        # convert recon_x_conv to log2(TPM + 1) format
        recon_x_conv = recon_x_conv.squeeze()  # (batch_size, n_genes))
        recon_x_conv = non_log2log_cpm_tensor(recon_x_conv, transpose=False)  # (batch_size, n_genes)
        if self.model_config.scaling_by_constant:
            recon_x_conv = recon_x_conv / 20.0
        recon_x_conv = recon_x_conv.reshape(x.shape)

        # get learned logit for each cell type
        # anchor_weights = F.softmax(self.logits, dim=-1)  # (n_cell_types, latent_dim)
        # mu_prior = anchor_weights @ self.anchors  # (n_cell_types, latent_dim)
        mu_prior = None  # (n_cell_types, latent_dim)

        (loss,
         kld_z,
         kld_p,
         recon_loss_conv,
         gene_mean_loss,
         gene_std_loss,
         repulsion_loss,
         cell_prop_loss
         ) = self.loss_function(
            # recon_x=recon_x, x=x, mu=mu, log_var=log_var, y=y,
            x=x, y=y,
            logvar_types=log_var_types, mu_types=mu_types,
            dd_alpha=dd_alpha,
            recon_x_conv=recon_x_conv,
            beta=self.model_config.loss_coefficient['beta'],
            mu_prior=mu_prior,
            gamma=self.model_config.loss_coefficient['gamma'],
            recon_gene_mean= recon_gene_mean,
            recon_gene_std= recon_gene_std,
            logvar_mean=logvar_mean,
            mu_mean=mu_mean,
            device=current_device,
        )

        output = ModelOutput(
            # recon_loss=recon_loss,
            # reg_loss=kld_z,
            loss=loss,
            # recon_x=recon_x,
            # z=z,
            mu=mu_mean,
            mu_deconv=mu_types,
            # log_var=log_var,
            log_var=log_var_types,
            cell_prop_loss=cell_prop_loss,
            kld=kld_z,
            pred_cell_prop=cell_prop,
            recon_x_conv=recon_x_conv,
            recon_loss_conv=recon_loss_conv,
            recon_x_all_types=recon_x_all_types,
            gene_mean_loss=gene_mean_loss,
            gene_std_loss=gene_std_loss,
            repulsion_loss=repulsion_loss,
            kld_p=kld_p,  # kld for cellular proportions
        )
        return output

    def loss_function(self, x: torch.Tensor,
                      recon_x_conv: Optional[torch.Tensor] = None,
                      mu_types: torch.Tensor = None,
                      logvar_types: torch.Tensor = None,
                      y: Optional[torch.Tensor] = None,
                      dd_alpha: Optional[torch.Tensor] = None,
                      mu_prior: Optional[torch.Tensor] = None,
                      beta=1.0,
                      gamma=0.1,
                      eps=1e-6,
                      logvar_mean: Optional[torch.Tensor] = None,
                      mu_mean: Optional[torch.Tensor] = None,
                      recon_gene_mean: Optional[torch.Tensor] = None,
                      recon_gene_std: Optional[torch.Tensor] = None,
                      device: Optional[torch.device] = None,
                      ):
        """Calculates the loss for the VAE.

        Args:
            x: Input data.
            recon_x_conv: Reconstructed data from cell-type-specific GEPs x cellular proportions.
            mu_types: List of cell type means.
            logvar_types: List of cell type log variances.
            y: Cell proportions of the input data.
            dd_alpha: Dirichlet distribution parameters.
            mu_prior: Learnable prior means for the latent space.
            beta: Weight for the KL divergence term.
            gamma: Weight for the repulsion loss.
            eps: Small value to avoid division by zero.
            recon_gene_mean: Reconstructed gene means for each cell type across the whole batch.
            recon_gene_std: Reconstructed gene standard deviations for each cell type across the whole batch.
            device: Device to perform the calculations on.
        Returns:
            A tuple containing the total loss, KL divergence loss, cell proportion loss, and reconstruction loss.
        """
        # batch_size, n_genes = x.shape
        batch_size, latent_dim, n_cell_types = mu_types.shape

        # --- Reconstruction loss ---
        # flat_x = x.reshape(batch_size, -1)  # (batch_size, n_genes)
        # flat_recon_x_conv = recon_x_conv.reshape(batch_size, -1)  # (batch_size, n_genes)
        if self.model_config.reconstruction_loss == "mse":
            recon_loss_by_conv = F.mse_loss(recon_x_conv, x, reduction="none")  # (batch_size, n_genes)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss_by_conv = F.binary_cross_entropy(recon_x_conv, x, reduction="none")
        else:
            raise ValueError(
                f"Reconstruction loss {self.model_config.reconstruction_loss} is not implemented"
            )
        # apply weights to the reconstruction loss
        # recon_loss_by_conv = (recon_loss_by_conv * w).sum(dim=-1)  # (batch_size,)
        recon_loss_by_conv = recon_loss_by_conv.sum(dim=-1)  # (batch_size,)  # sum over genes without weights

        # --- Representation loss for gene means and stds ---
        # Sum over all genes first, then mean over cell types
        gene_mean_loss = F.mse_loss(recon_gene_mean, self.g_mean, reduction="none")
        gene_mean_loss = (gene_mean_loss * self.w).sum(dim=0).mean(dim=0)  # a scalar
        gene_std_loss = F.mse_loss(recon_gene_std, self.g_std, reduction="none").sum(dim=0).mean(dim=0)  # a scalar

        # --- KL divergence loss for cellular proportions ---
        # Prior: Uniform Dirichlet distribution (all alpha = 1)
        kld_p = torch.zeros(batch_size, device=device)
        if y is not None and self.model_config.predict_cell_prop:
            prior_alpha = torch.ones_like(dd_alpha)
            prior_dist_p = Dirichlet(prior_alpha)
            posterior_dist_p = Dirichlet(dd_alpha)
            # kld_p = kl_divergence(prior_dist_p, posterior_dist_p).sum(dim=-1)
            # https://stats.stackexchange.com/a/370048
            kld_p = kl_divergence(posterior_dist_p, prior_dist_p)

        # Cell proportions loss
        cell_prop_loss = torch.zeros(batch_size, device=device)
        if y is not None and self.model_config.predict_cell_prop:
            normalized_dd_alpha = dd_alpha / torch.sum(dd_alpha, dim=-1, keepdim=True)  # (batch_size, n_cell_types)
            cell_prop_loss = F.mse_loss(normalized_dd_alpha, y, reduction="none").sum(dim=-1)  # (batch_size,)
            # Using KL divergence loss
            # cell_prop_loss = F.kl_div(normalized_dd_alpha.log(), y, reduction='batchmean')  # (batch_size,)


        # --- Gaussian KL divergence loss for GEPs ---
        # Prior: Gaussian distribution (mean=0, std=1)
        # mu = torch.stack(mu_list, dim=1).to(self.device)  # (batch_size, n_cell_types, latent_dim)
        # logvar = torch.stack(logvar_list, dim=1).to(self.device)  # (batch_size, n_cell_types, latent_dim)
        # mu_mean = mu_list.mean(dim=-1) # (batch_size, latent_dim)
        # logvar_mean = logvar_list.mean(dim=-1)   # (batch_size, latent_dim)
        # var_mean = logvar_mean.exp()  # (batch_size, latent_dim)
        # logvar_mean = torch.log(var_mean).to(self.device)
        # logvar_mean = logvar_list.mean(dim=-1).to(self.device)  # (batch_size, latent_dim)
        # var = logvar_mean.exp()  # (batch_size, n_cell_types, latent_dim)
        # if mu_prior is not None:
        #     # (n_cell_types, latent_dim) -> (1, n_cell_types, latent_dim), then later it can broadcast to (batch_size, n_cell_types, latent_dim)
        #     mu_prior = mu_prior.unsqueeze(0).to(self.device)
        #     diff = mu - mu_prior  # (batch_size, n_cell_types, latent_dim)
        #     kld_bt = -0.5 * torch.sum(1 + logvar - diff.pow(2) - var, dim=-1)  # sum over latent_dim
        # else:
        ## kld = - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        # kld_z_per_type = -0.5 * torch.sum(1 + logvar_types - mu_types.pow(2) - logvar_types.exp(),
        #                                   dim=2)  # [batch_size, latent_dim]
        # kld_z_types = kld_z_per_type.sum(dim=1)  # Sum over latent dim, [batch_size]
        # kld_z_types = -0.5 * torch.sum(1 + logvar_mean - mu_mean.pow(2) - logvar_mean.exp(), dim=-1)  # sum over latent_dim

        log_var = logvar_mean.reshape(-1, self.model_config.latent_dim)  # batch_size x latent_dim
        # Since we decomposed bulk GEP into cell type-specific GEPs,
        # we need to sum over the embeddings of all cell types
        mu = mu_mean.reshape(-1, self.model_config.latent_dim)
        # https://stats.stackexchange.com/a/370048
        kld_z_types = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        # --- Repulsion loss ---
        if gamma > 0:
            # n_cell_types, latent_dim = mu.shape[1], mu.shape[2]
            # using broadcasting to calculate the pairwise distance
            mu_types_permuted = mu_types.permute(0, 2, 1)   # (batch_size, n_cell_types, latent_dim)
            # (B, N, 1, L) - (B, 1, N, L) = (B, N, N, L)
            diff = mu_types_permuted.unsqueeze(2) - mu_types_permuted.unsqueeze(1)  # (batch_size, n_cell_types, n_cell_types, latent_dim)
            dist2 = torch.sum(diff ** 2, dim=-1)  # (batch_size, n_cell_types, n_cell_types)
            # remove the diagonal elements (self-repulsion)
            mask = ~torch.eye(n_cell_types, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1, n_cell_types, n_cell_types)
            inv_dist = torch.where(mask,
                                   1.0 / (dist2 + eps),  # avoid division by zero)
                                   torch.zeros_like(dist2))
            # repulsion loss
            repulsion_loss = inv_dist.sum(dim=(1, 2))  # sum over each (i, j) pair, (batch_size,)
        else:
            repulsion_loss = torch.zeros(batch_size, device=x.device)

        # print('recon_loss_by_decoder.shape', recon_loss_by_decoder.shape, 'kld.shape', kld.shape,
        #       'cell_prop_loss.shape', cell_prop_loss.shape)
        lo = self.model_config.loss_coefficient
        total_loss = (
                recon_loss_by_conv
                + beta * (kld_z_types + kld_p)
                # + lo['kld'] * kld_z_types
                + lo['cell_prop'] * cell_prop_loss
                + gamma * repulsion_loss
                + gene_mean_loss
                + gene_std_loss
        ).mean(dim=0)  # average over batch size, scalar

        return (total_loss,
                kld_z_types.mean(dim=0),
                kld_p.mean(dim=0),
                recon_loss_by_conv.mean(dim=0),
                gene_mean_loss.mean(dim=0),
                gene_std_loss.mean(dim=0),
                repulsion_loss.mean(dim=0),
                cell_prop_loss.mean(dim=0),
                )

    def compute_gene_weights(
            self,
            low_weight_coef: float = 1.0,
            eps: float = 1e-6,
            clamp_range: Tuple[float, float] = (0.5, 2.0),
    ) -> torch.Tensor:
        """
        Compute per-gene weights for a batch of expression profiles, higher weights for low variance genes.

        Args:
            low_weight_coef:
                          Exponent to give more restrict constraint for low variance genes.
            eps:          Small constant to avoid div-by-zero.
            clamp_range:  (min, max) after normalization.

        Returns:
            w: (batch_size, n_genes) normalized, clamped, exponentiated weights.
        """

        # 1) choose raw weight w_g based on the variance of gene expression in each cell type
        w_g = 1.0 / (self.g_std + eps)  # (n_genes, n_cell_types)

        # 2) normalize so that average weight = 1
        w_g = w_g / w_g.mean(dim=0, keepdim=True)

        # 3) clamp to avoid extremes
        w_g = torch.clamp(w_g, min=clamp_range[0], max=clamp_range[1])

        # 4) optionally up-weight low-expr genes more strongly
        if low_weight_coef != 1.0:
            w_g = w_g.pow(low_weight_coef)

        return w_g

    @staticmethod
    def make_orthonormal_anchors(latent_dim, radius=1.0):
        """
        Generate orthonormal anchors for the latent space.
        Args:
            radius (float): The radius of the sphere on which the anchors are located.
        :return: Orthonormal vectors in the latent space. Rows are orthonormal vectors (after transpose).
        """
        random_weights = torch.randn(latent_dim, latent_dim)
        q, _ = torch.qr(random_weights)  # QR decomposition to get orthonormal vectors
        return radius * q.t()

    @staticmethod
    def _poe_fuse_core(
            mu_list: List[torch.Tensor],
            logvar_list: List[torch.Tensor],
            eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core Product of Experts (PoE) fusion for a list of mean and log variance tensors.
        Assumes all input tensors in the lists have the same shape.

        Args:
            mu_list: List of mean tensors from different experts.
                     Each tensor shape, e.g., (batch_size, latent_dim, n_cell_types)
                     or (batch_size, latent_dim).
            logvar_list: List of log variance tensors from different experts.
                         Matching shapes to elements in mu_list.
            eps: Small value for numerical stability.

        Returns:
            A tuple containing:
            - fused_mu: PoE fused mean tensor, same shape as input tensors.
            - fused_logvar: PoE fused log variance tensor, same shape as input tensors.
        """
        if not mu_list or not logvar_list:
            raise ValueError("Input lists for PoE fusion cannot be empty.")
        if len(mu_list) != len(logvar_list):
            raise ValueError(
                "Mismatch in the number of experts for mus and logvars."
            )
        if len(mu_list) == 1:  # Only one expert, no fusion needed
            return mu_list[0], logvar_list[0]

        # Stack expert parameters along a new dimension (dim=0)
        # If inputs are (B, L, C), mus_stacked becomes (num_experts, B, L, C)
        mus_stacked = torch.stack(mu_list, dim=0)
        logvars_stacked = torch.stack(logvar_list, dim=0)

        # Calculate precisions: P_i = 1 / sigma_i^2 = exp(-logvar_i)
        precisions_stacked = torch.exp(-logvars_stacked)

        # Sum of precisions from all experts: P_poe = sum(P_i)
        # Shape: (B, L, C) or (B, L) depending on input shapes
        sum_of_precisions = torch.sum(precisions_stacked, dim=0)

        # Fused log variance: logvar_poe = -log(P_poe + eps)
        fused_logvar = -torch.log(sum_of_precisions + eps)

        # Weighted sum of means by their precisions: sum(mu_i * P_i)
        sum_of_weighted_mus = torch.sum(mus_stacked * precisions_stacked, dim=0)

        # Fused mean: mu_poe = sum(mu_i * P_i) / (P_poe + eps)
        fused_mu = sum_of_weighted_mus / (sum_of_precisions + eps)

        return fused_mu, fused_logvar

    # @staticmethod
    # def _poe_fuse_per_celltype(
    #     mu_lists: List[torch.Tensor],
    #     logvar_lists: List[torch.Tensor],
    #     mu_mean_list: List[torch.Tensor],
    #     logvar_mean_list: List[torch.Tensor],
    #     eps: float = 1e-6,
    #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Perform product of experts (PoE) fusion for the mean and log variance of the latent space.
    #
    #     Args:
    #         mu_lists: List of means from different encoders. Each element: (batch_size, latent_dim, n_cell_types).
    #         logvar_lists: List of log variances from different encoders. Each element: (batch_size, latent_dim, n_cell_types).
    #         eps: Small value to avoid division by zero.
    #
    #     Returns:
    #         Fused mean and log variance.
    #     """
    #     # PoE fusion
    #     fused_mu, fused_logvar = [], []
    #     for mu1, lv1, mu2, lv2 in zip(mu_lists[0], logvar_lists[0], mu_lists[1], logvar_lists[1]):
    #         # PoE fusion for each cell type
    #         prec1 = torch.exp(-lv1)  # precision, (batch_size, latent_dim)
    #         prec2 = torch.exp(-lv2)
    #
    #         # fused precision
    #         mu_poe = (mu1 * prec1 + mu2 * prec2) / (prec1 + prec2 + eps)
    #
    #         # fused log variance
    #         lv_poe = -torch.log(prec1 + prec2 + eps)
    #
    #         fused_mu.append(mu_poe)
    #         fused_logvar.append(lv_poe)
    #     # Each mu_mean_list: (batch_size, latent_dim)
    #     mu_mean = torch.stack(mu_mean_list, dim=2).mean(dim=2)  # (batch_size, latent_dim)
    #     logvar_mean = torch.stack(logvar_mean_list, dim=2).mean(dim=2)
    #     fused_mu = torch.stack(fused_mu, dim=0)  # (batch_size, latent_dim, n_cell_types)
    #     fused_logvar = torch.stack(fused_logvar, dim=0)  # (batch_size, latent_dim, n_cell_types)
    #     return fused_mu, fused_logvar, mu_mean, logvar_mean

    def _poe_fuse_per_celltype(
            self,
            mu_lists_celltype: List[torch.Tensor],
            logvar_lists_celltype: List[torch.Tensor],
            mu_list_overall: List[torch.Tensor],
            logvar_list_overall: List[torch.Tensor],
            eps: float = 1e-8,  # Using a slightly smaller eps is common
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Product of Experts (PoE) fusion for per-celltype latent space parameters
        from multiple experts and computes a simple average for overall (e.g., mean over celltypes)
        latent space parameters.

        Args:
            mu_lists_celltype: List of mean tensors from different experts, for per-celltype data.
                               Each tensor shape: (batch_size, latent_dim, n_cell_types).
            logvar_lists_celltype: List of log variance tensors from different experts, for per-celltype data.
                                   Each tensor shape: (batch_size, latent_dim, n_cell_types).
            mu_list_overall: List of mean tensors (e.g., already averaged over celltypes before this call)
                             from different experts. Each tensor shape: (batch_size, latent_dim).
            logvar_list_overall: List of log variance tensors (overall) from different experts.
                                 Each tensor shape: (batch_size, latent_dim).
            eps: Small value for numerical stability in PoE.

        Returns:
            A tuple containing:
            - fused_mu_celltype: PoE fused mean for per-celltype data
                                 Shape: (batch_size, latent_dim, n_cell_types).
            - fused_logvar_celltype: PoE fused log variance for per-celltype data
                                     Shape: (batch_size, latent_dim, n_cell_types).
            - avg_mu_overall: Averaged mean of overall means
                              Shape: (batch_size, latent_dim).
            - avg_logvar_overall: Averaged log variance of overall logvars
                                  Shape: (batch_size, latent_dim).
        """
        # --- PoE fusion for per-celltype parameters ---
        # These lists must not be empty for the helper, handled by _poe_fuse_core
        fused_mu_celltype, fused_logvar_celltype = self._poe_fuse_core(
            mu_list=mu_lists_celltype,
            logvar_list=logvar_lists_celltype,
            eps=eps
        )

        # --- Averaging for overall parameters ---
        if not mu_list_overall or not logvar_list_overall:
            raise ValueError("Overall mean/logvar lists cannot be empty for averaging.")

        if len(mu_list_overall) == 1:  # Only one expert for overall params
            avg_mu_overall = mu_list_overall[0]
            avg_logvar_overall = logvar_list_overall[0]
        else:
            # Stack for averaging: (num_experts, batch_size, latent_dim)
            mu_overall_stacked = torch.stack(mu_list_overall, dim=0)
            avg_mu_overall = torch.mean(mu_overall_stacked, dim=0)

            logvar_overall_stacked = torch.stack(logvar_list_overall, dim=0)
            avg_logvar_overall = torch.mean(logvar_overall_stacked, dim=0)

        return fused_mu_celltype, fused_logvar_celltype, avg_mu_overall, avg_logvar_overall
