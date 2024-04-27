from typing import Tuple, Union, Literal

import torch

from vaedecon.config import BaseConfig
# from ..nn.positional_encoding import PositionalEncoding


class BaseModelConfig(BaseConfig):
    """This is the base configuration instance of the models deriving from
    :class:`~pythae.config.BaseConfig`.

    Parameters:
        input_dim (tuple): The input_data dimension (channels X x_dim X y_dim)
        latent_dim (int): The latent space dimension. Default: None.
    """

    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = 10  # latent space dimension for each cell type
    n_cell_types: int = 10  # number of cell types
    uses_default_encoder: bool = True
    uses_default_decoder: bool = True
    reconstruction_loss: Literal["bce", "mse"] = "mse"
    # position_encoding: torch.Tensor = None


class EnvironmentConfig(BaseConfig):
    python_version: str = "3.8"
    name: str = "EnvironmentConfig"
