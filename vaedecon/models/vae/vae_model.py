"""
Variational Autoencoder (VAE) implementation for cellular component deconvolution.
OnlyBelter (https://github.com/OnlyBelter, onlybelter@gmail.com)
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vaedecon.data.datasets import BaseDataset, GEPDataset, DatasetOutput
from ...models.base.base_utils import ModelOutput

from ...models.base import BaseAE
from ...models.nn import BaseDecoder, BaseEncoder
from ...models.nn import Encoder_MLP
from ...models.base.base_config import BaseModelConfig
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor
# from .vae_config import VAEConfig
# from pythae.models.vae.vae_model import VAE


class VAE(BaseAE):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: BaseModelConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "VAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseModelConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: DatasetOutput, **kwargs):
        """
        The VAE model

        Args:
            inputs (DatasetOutput): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        y = inputs["labels"]  # cell proportions of 16 cell types

        encoder_output = self.encoder(x=x, y=y)

        mu, log_var, pred_cell_prop = encoder_output.embedding, encoder_output.log_var, encoder_output.cell_prop
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
        recon_x_all_types = torch.zeros([x.shape[0], x.shape[1], n_cell_types], dtype=torch.float32, device=x.device)
        for i in range(n_cell_types):
            mu_specific_type = mu_deconv[:, :, i]
            mu_specific_type = mu_specific_type.reshape(mu.shape)  # only for reshaping
            z_specific_type, _ = self._sample_gauss(mu_specific_type, std)  # same std for all cell types
            recon_x_all_types[:, :, i] = self.decoder(z_specific_type)["reconstruction"].reshape(x.shape)
        # recon_x_all_types should be recovered to CPM format before doing the matrix multiplication
        if self.model_config.scaling_by_constant:
            recon_x_all_types = recon_x_all_types * 20.0
        recon_x_all_types = log_exp2cpm_tensor(recon_x_all_types, transpose=True)
        if y is not None:
            recon_x_conv = torch.matmul(recon_x_all_types, y.reshape(-1, n_cell_types, 1))
        else:
            recon_x_conv = torch.matmul(recon_x_all_types, pred_cell_prop.reshape(-1, n_cell_types, 1))
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

    def loss_function(self, x, mu, log_var, y: torch.Tensor | None = None,
                      pred_cell_prop: torch.Tensor = None, recon_x_conv: torch.Tensor = None):
        """
        The loss function of the VAE model

        - params:
        x: input data
        mu: mean of the latent space
        log_var: log variance of the latent space
        y: cell proportions of the input data
        pred_cell_prop: predicted cell proportions
        recon_x_conv: reconstructed data from the GEPs of all cell types (multiply the output of the decoder)

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
        KLD = - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        # cell proportion loss
        if y is not None:
            cell_prop_loss = F.mse_loss(
                pred_cell_prop.reshape(y.shape[0], -1),  # batch_size x features (cell proportions)
                y.reshape(y.shape[0], -1),
                reduction="none"
            ).sum(dim=-1)
        else:
            cell_prop_loss = torch.zeros_like(KLD)
        # print('recon_loss_by_decoder.shape', recon_loss_by_decoder.shape, 'KLD.shape', KLD.shape,
        #       'cell_prop_loss.shape', cell_prop_loss.shape)
        lo = self.model_config.loss_coefficient

        return ((lo['recon_convolution']*recon_loss_by_conv +
                 lo['kld']*KLD + lo['cell_prop']*cell_prop_loss).mean(dim=0),
                KLD.mean(dim=0), cell_prop_loss.mean(dim=0),
                recon_loss_by_conv.mean(dim=0))

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_var

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":
                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)
