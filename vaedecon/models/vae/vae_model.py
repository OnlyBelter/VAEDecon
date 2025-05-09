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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        x = inputs["data"].to(device)
        y = inputs.get("labels")  # cell proportions of 16 cell types
        if y is not None:
            y = y.to(device)

        # Call all encoders, and collect the outputs
        mu_lists, logvar_lists, prop_list, dd_alpha_list = [], [], [], []
        for encoder in self.encoders:
            out = encoder(x=x, y=y)
            mu_lists.append(out.mu_list)
            # mu_lists.append(out.mu_all_types)
            logvar_lists.append(out.logvar_list)
            # logvar_lists.append(out.log_var)
            prop_list.append(out.cell_prop)
            if self.model_config.predict_cell_prop:
                dd_alpha_list.append(out.dd_alpha)
        if self.n_encoders == 1:
            mu_list, logvar_list, cell_prop = (
                mu_lists[0],
                logvar_lists[0],
                prop_list[0],
            )
        else:  # two encoders
            mu_list, logvar_list = self._poe_fuse_per_celltype(
                mu_lists=mu_lists,
                logvar_lists=logvar_lists,
            )
            cell_prop = torch.mean(torch.stack(prop_list, dim=0), dim=0)  # (batch_size, n_cell_types)

        mu_types = torch.stack(mu_list, dim=2)  # (batch_size, latent_dim, n_cell_types)
        log_var_types = torch.stack(logvar_list, dim=2)  # (batch_size, latent_dim, n_cell_types)
        # mu_types = mu_list  # (batch_size, latent_dim, n_cell_types)
        # log_var = logvar_list  # (batch_size, latent_dim), all cell types shared
        # pred_cell_prop = None
        dd_alpha = torch.zeros(self.model_config.n_cell_types)
        if self.model_config.predict_cell_prop:
            if self.n_encoders == 1:
                dd_alpha = dd_alpha_list[0]
            else:  # two encoders
                dd_alpha = torch.mean(torch.stack(dd_alpha_list, dim=0), dim=0) # (batch_size, n_cell_types)
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
        cell_prop = cell_prop.reshape(-1, n_cell_types, 1).to(device)  # (batch_size, n_cell_types, 1)
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
        anchor_weights = F.softmax(self.logits, dim=-1)  # (n_cell_types, latent_dim)
        # mu_prior = anchor_weights @ self.anchors  # (n_cell_types, latent_dim)
        mu_prior = None  # (n_cell_types, latent_dim)

        (loss, kld_z, kld_p, recon_loss_conv,
         gene_mean_loss, gene_std_loss) = self.loss_function(
            # recon_x=recon_x, x=x, mu=mu, log_var=log_var, y=y,
            x=x, logvar_list=log_var_types, y=y, mu_list=mu_types,
            dd_alpha=dd_alpha, recon_x_conv=recon_x_conv,
            beta=self.model_config.loss_coefficient['beta'],
            mu_prior=mu_prior,
            gamma=self.model_config.loss_coefficient['gamma'],
            recon_gene_mean= recon_gene_mean,
            recon_gene_std= recon_gene_std,
        )

        output = ModelOutput(
            # recon_loss=recon_loss,
            # reg_loss=kld_z,
            loss=loss,
            # recon_x=recon_x,
            # z=z,
            # mu=mu,
            mu_deconv=mu_types,
            # log_var=log_var,
            log_var=log_var_types,
            cell_prop_loss=kld_p,
            kld=kld_z,
            pred_cell_prop=cell_prop,
            recon_x_conv=recon_x_conv,
            recon_loss_conv=recon_loss_conv,
            recon_x_all_types=recon_x_all_types,
            gene_mean_loss=gene_mean_loss,
            gene_std_loss=gene_std_loss,
        )
        return output

    def loss_function(self, x: torch.Tensor,
                      recon_x_conv: Optional[torch.Tensor] = None,
                      mu_list: torch.Tensor = None,
                      logvar_list: torch.Tensor = None,
                      y: Optional[torch.Tensor] = None,
                      dd_alpha: Optional[torch.Tensor] = None,
                      mu_prior: Optional[torch.Tensor] = None,
                      beta=1.0,
                      gamma=0.1,
                      eps=1e-6,
                      recon_gene_mean: Optional[torch.Tensor] = None,
                      recon_gene_std: Optional[torch.Tensor] = None,
    ):
        """Calculates the loss for the VAE.

        Args:
            x: Input data.
            recon_x_conv: Reconstructed data from cell-type-specific GEPs x cellular proportions.
            mu_list: List of cell type means.
            logvar_list: List of cell type log variances.
            y: Cell proportions of the input data.
            dd_alpha: Dirichlet distribution parameters.
            mu_prior: Learnable prior means for the latent space.
            beta: Weight for the KL divergence term.
            gamma: Weight for the repulsion loss.
            eps: Small value to avoid division by zero.
            recon_gene_mean: Reconstructed gene means for each cell type across the whole batch.
            recon_gene_std: Reconstructed gene standard deviations for each cell type across the whole batch.
        Returns:
            A tuple containing the total loss, KL divergence loss, cell proportion loss, and reconstruction loss.
        """
        batch_size, n_genes = x.shape
        n_cell_types = mu_list.shape[2]

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
        kld_p = torch.zeros(batch_size, device=x.device)
        if y is not None and self.model_config.predict_cell_prop:
            prior_alpha = torch.ones_like(dd_alpha)
            prior_dist_p = Dirichlet(prior_alpha)
            posterior_dist_p = Dirichlet(dd_alpha)
            kld_p = kl_divergence(prior_dist_p, posterior_dist_p).sum(dim=-1)

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
        kld_z_per_type = -0.5 * torch.sum(1 + logvar_list - mu_list.pow(2) - logvar_list.exp(),
                                          dim=2)  # [batch_size, latent_dim]
        kld_z_types = kld_z_per_type.sum(dim=1)  # Sum over latent dim, [batch_size]
        # kld_z_types = -0.5 * torch.sum(1 + logvar_mean - mu_mean.pow(2) - var_mean, dim=-1)  # sum over latent_dim

        # kld_z_types = kld_bt.sum(dim=-1)  # sum over n_cell_types
        # kld_z_types = kld_bt

        # # --- Repulsion loss ---
        # # n_cell_types, latent_dim = mu.shape[1], mu.shape[2]
        # # using broadcasting to calculate the pairwise distance
        # diff = mu_list.unsqueeze(2) - mu_list.unsqueeze(1)  # (batch_size, n_cell_types, n_cell_types, latent_dim)
        # dist2 = torch.sum(diff ** 2, dim=-1)  # (batch_size, n_cell_types, n_cell_types)
        # # remove the diagonal elements (self-repulsion)
        # mask = ~torch.eye(n_cell_types, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1, n_cell_types, n_cell_types)
        # inv_dist = torch.where(mask,
        #                        1.0 / (dist2 + eps),  # avoid division by zero)
        #                        torch.zeros_like(dist2))
        # # repulsion loss
        # repulsion_loss = inv_dist.sum(dim=(1, 2))  # sum over each (i, j) pair, (batch_size,)

        # print('recon_loss_by_decoder.shape', recon_loss_by_decoder.shape, 'kld.shape', kld.shape,
        #       'cell_prop_loss.shape', cell_prop_loss.shape)
        # lo = self.model_config.loss_coefficient
        total_loss = (recon_loss_by_conv
                + beta * (kld_z_types + kld_p)
                # + gamma * repulsion_loss
                + gene_mean_loss
                + gene_std_loss
        ).mean(dim=0)  # average over batch size, scalar

        return (total_loss, kld_z_types.mean(dim=0), kld_p.mean(dim=0),
                recon_loss_by_conv.mean(dim=0), gene_mean_loss, gene_std_loss)

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

    def loss_function_old(self, x: torch.Tensor, mu: torch.Tensor,
                      log_var: torch.Tensor, y: Optional[torch.Tensor] = None,
                      pred_cell_prop: Optional[torch.Tensor] = None,
                      recon_x_conv: Optional[torch.Tensor] = None):
        """Calculates the loss for the VAE.

        Args:
            x: Input data.
            mu: Mean of the latent space.
            log_var: Log variance of the latent space.
            y: Cell proportions of the input data.
            pred_cell_prop: Predicted cell proportions.
            recon_x_conv: Reconstructed data from cell-type specific GEPs.

        Returns:
            A tuple containing the total loss, KL divergence loss, cell proportion loss, and reconstruction loss.
        """
        # print('recon_x.shape', recon_x.shape, 'x.shape', x.shape, 'mu.shape', mu.shape,
        #       'log_var.shape', log_var.shape, 'y.shape', y.shape, 'pred_cell_prop.shape', pred_cell_prop.shape)
        # recon_x_by_decoder = recon_x
        # recon_x_by_conv = recon_x_conv
        if self.model_config.reconstruction_loss == "mse":
            recon_loss_by_conv = F.mse_loss(
                recon_x_conv.reshape(x.shape[0], -1),  # batch_size x features (gene expression values)
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss_by_conv = F.binary_cross_entropy(
                recon_x_conv.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        else:
            raise ValueError(
                f"Reconstruction loss {self.model_config.reconstruction_loss} not implemented"
            )

        log_var = log_var.reshape(-1, self.model_config.latent_dim)  # batch_size x latent_dim
        # Since we decomposed bulk GEP into cell type-specific GEPs,
        # we need to sum over the embeddings of all cell types
        mu = mu.reshape(-1, self.model_config.latent_dim)
        kld = - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        # cell proportion loss
        cell_prop_loss = torch.zeros_like(kld)
        if y is not None and pred_cell_prop is not None:
            cell_prop_loss = F.mse_loss(
                pred_cell_prop.reshape(y.shape[0], -1),  # batch_size x features (cell proportions)
                y.reshape(y.shape[0], -1),
                reduction="none"
            ).sum(dim=-1)

        # print('recon_loss_by_decoder.shape', recon_loss_by_decoder.shape, 'kld.shape', kld.shape,
        #       'cell_prop_loss.shape', cell_prop_loss.shape)
        lo = self.model_config.loss_coefficient
        total_loss = (
                lo['recon_convolution'] * recon_loss_by_conv
                + lo['kld'] * kld
                + lo['cell_prop'] * cell_prop_loss
        ).mean(dim=0)

        return (total_loss, kld.mean(dim=0), cell_prop_loss.mean(dim=0),
                recon_loss_by_conv.mean(dim=0))

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
    def _poe_fuse_per_celltype(
        mu_lists: List[torch.Tensor],
        logvar_lists: List[torch.Tensor],
        eps: float = 1e-6,
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform product of experts (PoE) fusion for the mean and log variance of the latent space.

        Args:
            mu_lists: List of means from different encoders.
            logvar_lists: List of log variances from different encoders.
            eps: Small value to avoid division by zero.

        Returns:
            Fused mean and log variance.
        """
        # PoE fusion
        fused_mu, fused_logvar = [], []
        for mu1, lv1, mu2, lv2 in zip(mu_lists[0], logvar_lists[0], mu_lists[1], logvar_lists[1]):
            # PoE fusion for each cell type
            prec1 = torch.exp(-lv1)  # precision, (batch_size, latent_dim)
            prec2 = torch.exp(-lv2)

            # fused precision
            mu_poe = (mu1 * prec1 + mu2 * prec2) / (prec1 + prec2 + eps)

            # fused log variance
            lv_poe = -torch.log(prec1 + prec2 + eps)

            fused_mu.append(mu_poe)
            fused_logvar.append(lv_poe)
        return fused_mu, fused_logvar
