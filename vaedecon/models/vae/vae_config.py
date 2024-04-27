from pydantic import BaseModel
from typing_extensions import Literal

from vaedecon.models.base.base_config import BaseModelConfig


class VAEConfig(BaseModelConfig):
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """

    reconstruction_loss: Literal["bce", "mse"] = "mse"
