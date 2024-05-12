from pydantic import BaseModel
from typing_extensions import Literal
from pathlib import Path

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
    input_gene_list: str = None  # file name
    cell_type_list: str = None  # file name
    scaling_by_constant: bool = True  # whether to scale the input GEP data by a constant factor (20 by default)
    loss_coefficient: dict = {
        "recon_decoder": 0.2,  # coefficient for the reconstruction loss of the decoder
        "recon_convolution": 0.3,  # coefficient for the reconstruction loss by convolution after decoding
        "kld": 0.2,  # coefficient for the KL divergence loss
        "cell_prop": 0.3,  # coefficient for the prediction loss of cell type proportions
    }  # coefficient for each term in the total loss function
