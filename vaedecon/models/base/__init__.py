"""
**Abstract class**

This is the base AuteEncoder architecture module from which all future autoencoder based
models should inherit.

It contains:

- | a :class:`~pythae.models.base.base_config.BaseModelConfig` instance containing the main model's
   parameters (*e.g.* latent dimension ...)
- | a :class:`~pythae.models.BaseAE` instance which creates a BaseAE model having a basic
   autoencoding architecture
- | The :class:`~pythae.models.base.base_utils.ModelOutput` instance used for neural nets outputs and
   model outputs of the :class:`forward` method).
"""

from .base_config import BaseModelConfig
from .base_model import BaseAE
from .base_utils import reparameterize_gaussian, reparameterize_dirichlet, ModelOutput, set_seed

__all__ = [
    "BaseModelConfig",
    "BaseAE",
    "reparameterize_gaussian",
    "reparameterize_dirichlet",
    "ModelOutput",
    "set_seed",]
