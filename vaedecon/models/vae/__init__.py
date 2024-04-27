"""This module is the implementation of a Vanilla Variational Autoencoder
(https://arxiv.org/abs/1312.6114)."""


from .vae_config import VAEConfig
from .vae_model import VAE

__all__ = ["VAE", "VAEConfig"]
