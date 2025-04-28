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
    # parameters for GNN encoder
    gnn_col_dim: int = 5000  # dimension of the column (the number of genes)
    gnn_row_dim: int = 1000  # dimension of the row (the number of cells in each batch to construct the KNN graph)
    gnn_inter_row_dim: int = 500  # dimension of the intermediate layer of the row (gene) embeddings
    gnn_inter_col_dim: int = 500  # dimension of the intermediate layer of the column (cell) embeddings
    gnn_embd_row_dim: int = 50  # dimension of the final row (gene) embeddings
    gnn_embd_col_dim: int = 50  # dimension of the final column (cell) embeddings
    gnn_lambda_rows: float = 1.0  # weight for the row (gene) loss
    gnn_lambda_cols: float = 1.0  # weight for the column (cell) loss
    gnn_num_layers: int = 3  # number of layers in the GNN
    gnn_drop_p: float = 0.1  # dropout probability
    ppi_file_path: str = None  # file path to the PPI
    gene_hidden_dim: int = 10  # each gene expression value will be increased to this dimension (1d -> higher dimension)


class EnvironmentConfig(BaseConfig):
    python_version: str = "3.8"
    name: str = "EnvironmentConfig"
