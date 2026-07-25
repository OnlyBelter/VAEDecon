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

# from vaedecon.configs.base_config import BaseModelConfig, BaseTrainerConfig, BaseConfig
from .base_model import BaseAE, BaseEncoder, BaseDecoder
from .base_utils import reparameterize_gaussian, reparameterize_dirichlet, ModelOutput, set_seed
from .base_utils import has_usable_labels
from .base_utils import LOGVAR_CLAMP_MAX, LOGVAR_CLAMP_MIN, NETWORK_CUTOFF, EPS
from .positional_encoding import PositionalEncoding


__all__ = [
    # "BaseModelConfig",
    # "BaseTrainerConfig",
    # "BaseConfig",
    "BaseAE",
    "reparameterize_gaussian",
    "reparameterize_dirichlet",
    "ModelOutput",
    "set_seed",
    "has_usable_labels",
    "BaseEncoder",
    "BaseDecoder",
    "LOGVAR_CLAMP_MAX",
    "LOGVAR_CLAMP_MIN",
    "NETWORK_CUTOFF",
    "EPS",
    "PositionalEncoding"]
