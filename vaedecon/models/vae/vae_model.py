"""
Variational Autoencoder (VAE) implementation for cellular component deconvolution.
OnlyBelter (https://github.com/OnlyBelter, onlybelter@gmail.com)
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import DatasetOutput
from ...models.base.base_utils import ModelOutput

from ...models.base import BaseAE
from ...models.nn import BaseDecoder, BaseEncoder
from ...models.base.base_config import BaseModelConfig
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class VAE(BaseAE):
    """Variational Autoencoder model for cellular component deconvolution."""

    def __init__(
        self,
        model_config: BaseModelConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        """Initializes the VAE model.

        Args:
            model_config: The VAE configuration.
            encoder: An optional encoder.
            decoder: An optional decoder.
        """
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)
        self.model_name = "VAE"

    def forward(self, inputs: DatasetOutput, **kwargs) -> ModelOutput:
        """Forward pass of the VAE model.

        Args:
            inputs: The input data to the model.

        Returns:
            An ModelOutput instance containing the model's output.
        """

        x = inputs["data"]
        y = inputs.get("labels")  # cell proportions of 16 cell types

        encoder_output = self.encoder(x=x, y=y)
        mu, log_var = (
            encoder_output.embedding,
            encoder_output.log_var,
        )
        if self.model_config.predict_cell_prop:
            pred_cell_prop = encoder_output.cell_prop,
        else:
            pred_cell_prop = None
        # log_var, pred_cell_prop = encoder_output.log_var, encoder_output.cell_prop
        mu_deconv = encoder_output.embedding_all_types  # (batch_size, latent_dim, n_cell_types)
        # cell_type_existed = encoder_output.cell_type_existed  # (batch_size, n_cell_types, 1)
        n_cell_types = mu_deconv.shape[2]
        # get all reconstructed GEPs first, then select the one based on cell_type_existed

        std = torch.exp(0.5 * log_var)
        # print('std.shape', std.shape, 'mu.shape', mu.shape)
        # z, eps = self._sample_gauss(mu, std)
        # reconstructing GEPs for the bulk mode by decoder directly
        # recon_x = self.decoder(z)["reconstruction"]  # bulk mode
        # recon_x = recon_x.reshape(x.shape)  # (batch_size, n_genes)
        # reconstructing GEPs for all cell types
        recon_x_all_types = torch.zeros(
            [x.shape[0], x.shape[1], n_cell_types], dtype=torch.float32, device=x.device
        )
        for i in range(n_cell_types):
            mu_specific_type = mu_deconv[:, :, i].reshape(mu.shape)
            z_specific_type = self._sample_gauss(mu_specific_type, std)  # same std for all cell types
            recon_x_all_types[:, :, i] = self.decoder(z_specific_type)["reconstruction"].reshape(x.shape)
        # recon_x_all_types should be recovered to CPM format before doing the matrix multiplication
        if self.model_config.scaling_by_constant:
            recon_x_all_types = recon_x_all_types * 20.0
        recon_x_all_types = log_exp2cpm_tensor(recon_x_all_types, transpose=True)
        if y is not None:
            recon_x_conv = torch.matmul(recon_x_all_types, y.reshape(-1, n_cell_types, 1))
        elif pred_cell_prop is not None:
            recon_x_conv = torch.matmul(recon_x_all_types, pred_cell_prop.reshape(-1, n_cell_types, 1))
        else:
            raise ValueError("y or pred_cell_prop must be provided.")
        # convert recon_x_conv to log2(TPM + 1) format
        recon_x_conv = non_log2log_cpm_tensor(recon_x_conv, transpose=True)
        if self.model_config.scaling_by_constant:
            recon_x_conv = recon_x_conv / 20.0
        recon_x_conv = recon_x_conv.reshape(x.shape)

        loss, kld, cell_prop_loss, recon_loss_conv = self.loss_function(
            # recon_x=recon_x, x=x, mu=mu, log_var=log_var, y=y,
            x=x, log_var=log_var, y=y, mu=mu,
            pred_cell_prop=pred_cell_prop, recon_x_conv=recon_x_conv
        )

        output = ModelOutput(
            # recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            # recon_x=recon_x,
            # z=z,
            mu=mu,
            mu_deconv=mu_deconv,
            log_var=log_var,
            cell_prop_loss=cell_prop_loss,
            kld=kld,
            pred_cell_prop=pred_cell_prop,
            recon_x_conv=recon_x_conv,
            recon_loss_conv=recon_loss_conv,
            recon_x_all_types=recon_x_all_types,
        )
        return output

    def loss_function(self, x: torch.Tensor, mu: torch.Tensor,
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

    def _sample_gauss(self, mu, std):
        """Samples from a Gaussian distribution (N(0, I)) using the reparameterization trick."""
        eps = torch.randn_like(std)
        return mu + eps * std
