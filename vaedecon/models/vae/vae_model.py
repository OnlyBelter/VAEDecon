import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vaedecon.data.datasets import BaseDataset, GEPDataset
from ...models.base.base_utils import ModelOutput

from ...models.base import BaseAE
from ...models.nn import BaseDecoder, BaseEncoder
from ...models.nn import Encoder_MLP
from ...models.base.base_config import BaseModelConfig
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

    def forward(self, inputs: GEPDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (GEPDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        y = inputs["labels"]  # cell proportions of 16 cell types

        encoder_output = self.encoder(x=x, y=y)

        mu, log_var, pred_cell_prop = encoder_output.embedding, encoder_output.log_var, encoder_output.cell_prop

        std = torch.exp(0.5 * log_var)
        # print('std.shape', std.shape, 'mu.shape', mu.shape)
        z, eps = self._sample_gauss(mu, std)
        # print('z.shape', z.shape)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld, cell_prop_loss = self.loss_function(recon_x, x, mu, log_var, y, pred_cell_prop)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
            mu=mu,
            log_var=log_var,
            cell_prop_loss=cell_prop_loss,
            kld=kld,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, y, pred_cell_prop):
        """
        The loss function of the VAE model

        """
        # print('recon_x.shape', recon_x.shape, 'x.shape', x.shape, 'mu.shape', mu.shape,
        #       'log_var.shape', log_var.shape, 'y.shape', y.shape, 'pred_cell_prop.shape', pred_cell_prop.shape)
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),  # batch_size x features (gene expression values)
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        else:
            raise ValueError(
                f"Reconstruction loss {self.model_config.reconstruction_loss} not implemented"
            )
        log_var = log_var.reshape(-1, self.model_config.latent_dim)  # batch_size x latent_dim
        mu = mu.reshape(-1, self.model_config.latent_dim)
        KLD = - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        # cell proportion loss
        cell_prop_loss = F.mse_loss(
            pred_cell_prop.reshape(y.shape[0], -1),  # batch_size x features (cell proportions)
            y.reshape(y.shape[0], -1),
            reduction="none"
        ).sum(dim=-1)
        # print('recon_loss.shape', recon_loss.shape, 'KLD.shape', KLD.shape,
        #       'cell_prop_loss.shape', cell_prop_loss.shape)

        return ((0.5*recon_loss + 0.2*KLD + 0.3*cell_prop_loss).mean(dim=0),
                recon_loss.mean(dim=0), KLD.mean(dim=0), cell_prop_loss.mean(dim=0))

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
