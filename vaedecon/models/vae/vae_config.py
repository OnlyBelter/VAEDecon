from pydantic import BaseModel
from typing_extensions import Literal

from vaedecon.models.base.base_config import BaseModelConfig


class VAEConfig(BaseModelConfig):
    """VAE config class.

    Args:
        input_dim (tuple): The input_data dimension.

        latent_dim (int): The latent space dimension. Default: None.

        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'

        using_positional_encoding (bool): Whether to use positional encoding in the latent space to distinguish cell types
    """

    # reconstruction_loss: Literal["bce", "mse"] = "mse"
    using_positional_encoding: bool = True
