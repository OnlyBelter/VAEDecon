"""Proposed multilayer perceptron architectures as a baseline"""

from typing import List, Optional

import torch
import numpy as np
import torch.nn as nn

from ...base import BaseModelConfig
from ..positional_encoding import PositionalEncoding
from ....models.base.base_utils import ModelOutput
from ..base_architectures import BaseDecoder, BaseEncoder
# from ..utils import ResBlock


class EncoderMLP(BaseEncoder):
    """
    A Normal MLP encoder.
    """

    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.using_positional_encoding = args.using_positional_encoding
        self.hidden_dims = args.encoder_hidden_dims if hasattr(args, 'encoder_hidden_dims') else [1024, 512, 512]
        self.dropout_rate = args.encoder_dropout_rate
        self.layers = nn.ModuleList()
        if self.using_positional_encoding:
            self.position_encoding = position_encoding()
        else:
            self.position_encoding = None
        # self.n_channels = 1

        input_size = np.prod(self.input_dim)
        for i, hidden_dim_size in enumerate(self.hidden_dims):
            layer = nn.ModuleDict({
                'linear': nn.Linear(input_size, hidden_dim_size),
                'norm': nn.LayerNorm(hidden_dim_size, eps=1e-6),
                'activation': nn.ReLU(),
                'dropout': nn.Dropout(p=self.dropout_rate[i]) if (
                        self.dropout_rate[i] > 0) else nn.Identity(),
            })
            self.layers.append(layer)
            input_size = hidden_dim_size

        # self.layers = layers
        self.depth = len(self.layers)

        self.embedding = nn.Linear(in_features=self.hidden_dims[-1],
                                   out_features=args.latent_dim * args.n_cell_types)
        self.log_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.cell_prop = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.n_cell_types),
            nn.Softmax(dim=1)
        )
        # self.position_encoding = self.position_encoding.to(self.embedding.weight.device)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                output_layer_levels: Optional[List[int]] = None) -> ModelOutput:
        """Forward method

        Args:
            x (torch.Tensor): The input data
            y (torch.Tensor, optional): The cell proportions of the input data. Defaults to None.
            output_layer_levels (List[int], optional): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth
        if self.position_encoding is not None:
            self.position_encoding = self.position_encoding.to(x.device)

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.view(x.size(0), -1)  # flatten the input
        first_layer_output = None
        for i, layer in enumerate(self.layers):
            out = layer['linear'](out)
            out = layer['norm'](out)
            out = layer['activation'](out)
            if i == 0:
                first_layer_output = out
            elif i < max_depth - 1:
                if out.size(-1) != first_layer_output.size(-1):
                    # project the output of the first layer to the output of the current layer
                    first_layer_output = nn.Linear(first_layer_output.size(-1),
                                                   out.size(-1)).to(out.device)(first_layer_output)
                # residual connection
                out = out + first_layer_output
            out = layer['dropout'](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
    # if i + 1 == self.depth:
        # output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        # using the proposed structure of latent space
        embedding_all_types = self.embedding(out)  # (batch_size, latent_dim, n_cell_types)
        embedding_all_types = embedding_all_types.view((-1, self.latent_dim, self.n_cell_types))
        # TODO: getting cell proportions from DeSide
        output["cell_prop"] = self.cell_prop(out).view((-1, self.n_cell_types, 1))
        # print(embedding_all_types.shape, output["cell_prop"].shape)
        if y is not None:
            y = y.view((-1, self.n_cell_types, 1))
            # embedding = torch.matmul(embedding_all_types, y)  # bulk mode embedding
            cell_type_existed = (y > 0.01).type(torch.int8).type(torch.float32)
        else:
            # embedding = torch.matmul(embedding_all_types, output["cell_prop"])
            cell_type_existed = (output["cell_prop"] > 0.01).type(torch.int8).type(torch.float32)

        # assume y is unknown, using the average embedding of all cell types as the output miu of encoder
        # and calculate the KL divergence loss based on this miu
        embedding = torch.mean(embedding_all_types, dim=2, keepdim=True)  # (batch_size, latent_dim, 1)

        # (latent_dim, n_cell_types) x (batch_size, n_cell_types, 1) -> (batch_size, latent_dim, 1)
        if self.position_encoding is not None:
            position_encoding_cell_type = torch.matmul(self.position_encoding, cell_type_existed)
            output["embedding"] = embedding + position_encoding_cell_type
        else:
            output["embedding"] = embedding
        output["log_var"] = self.log_var(out).reshape((-1, self.latent_dim, 1))
        output['embedding_all_types'] = embedding_all_types
        output['cell_type_existed'] = cell_type_existed

        return output


class DecoderMLP(BaseDecoder):
    """
    A normal MLP decoder.
    """

    def __init__(self, args: BaseModelConfig):
        super().__init__()

        self.input_dim = args.input_dim  # input dimension of the Encoder
        self.latent_dim = args.latent_dim
        self.hidden_dims = args.decoder_hidden_dims if hasattr(args, 'decoder_hidden_dims') else [512, 512, 1024]
        self.dropout_rate = args.decoder_dropout_rate
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        # layers = nn.ModuleList()
        input_size = self.latent_dim
        for i, hidden_dim_size in enumerate(self.hidden_dims):
            layer = nn.ModuleDict({
                'linear': nn.Linear(input_size, hidden_dim_size),
                'norm': nn.LayerNorm(hidden_dim_size, eps=1e-6),
                'activation': nn.ReLU(),
                'dropout': nn.Dropout(p=self.dropout_rate[i]) if (
                        self.dropout_rate[i] > 0) else nn.Identity(),
            })
            self.layers.append(layer)
            input_size = hidden_dim_size

        # the last layer
        self.layers.append(
            nn.ModuleDict({
                'linear': nn.Linear(self.hidden_dims[-1], np.prod(self.input_dim)),
                'norm': nn.Identity(),
                'activation': nn.Softplus(beta=100, threshold=1),  # make sure the output is positive
                'dropout': nn.Dropout(p=self.dropout_rate[i]) if (
                        self.dropout_rate[-1] > 0) else nn.Identity(),
            })
        )

        # self.layers = layers
        self.depth = len(self.layers)

    def forward(self, z: torch.Tensor, output_layer_levels: Optional[List[int]] = None) -> ModelOutput:
        """Forward method

        Args:
            z (torch.Tensor): The latent code
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z.reshape(-1, 1, self.latent_dim)
        # print('z.shape', z.shape)

        latent_output = None
        for i, layer in enumerate(self.layers):
            out = layer['linear'](out)
            out = layer['norm'](out)
            out = layer['activation'](out)
            if i == 0:
                latent_output = out
            elif i < max_depth - 1:
                if out.size(-1) != latent_output.size(-1):
                    # project the output of the first layer to the output of the current layer
                    latent_output = nn.Linear(latent_output.size(-1), out.size(-1))(latent_output)
                # residual connection
                out = out + latent_output
            out = layer['dropout'](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

        # out = torch.clamp(self.relu(out), max=1.0)
        output["reconstruction"] = out
        return output


class ClampLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(ClampLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)
