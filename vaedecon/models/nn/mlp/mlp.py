"""Proposed multilayer perceptron architectures as a baseline"""

from typing import List

import torch
import torch.nn as nn
import numpy as np

from ...base import BaseModelConfig
from ..positional_encoding import PositionalEncoding
from ....models.base.base_utils import ModelOutput
from ..base_architectures import BaseDecoder, BaseEncoder
# from ..utils import ResBlock


class Encoder_MLP(BaseEncoder):
    """
    A Normal MLP encoder.

    """

    def __init__(self, args: BaseModelConfig, position_encoding: PositionalEncoding):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.using_positional_encoding = args.using_positional_encoding
        if self.using_positional_encoding:
            self.position_encoding = position_encoding()
        else:
            self.position_encoding = None
        # self.n_channels = 1

        layers = nn.ModuleList()
        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.input_dim), 512),
                nn.ReLU())
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, args.latent_dim * args.n_cell_types)
        self.log_var = nn.Linear(512, self.latent_dim)
        self.cell_prop = nn.Sequential(
            nn.Linear(512, self.n_cell_types),
            nn.Softmax(dim=1)
        )
        # self.position_encoding = self.position_encoding.to(self.embedding.weight.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None,
                output_layer_levels: List[int] = None) -> ModelOutput:
        """Forward method

        Args:
            x (torch.Tensor): The input data
            y (torch.Tensor): The cell proportions of the input data
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
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

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                # output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                # using the proposed structure of latent space
                embedding_all_types = self.embedding(out)  # (batch_size, latent_dim, n_cell_types)
                embedding_all_types = embedding_all_types.reshape((-1, self.latent_dim, self.n_cell_types))
                output["cell_prop"] = self.cell_prop(out).reshape((-1, self.n_cell_types, 1))
                # print(embedding_all_types.shape, output["cell_prop"].shape)
                if y is not None:
                    y = y.reshape((-1, self.n_cell_types, 1))
                    embedding = torch.matmul(embedding_all_types, y)  # bulk mode embedding
                    cell_type_existed = (y > 0.01).type(torch.int8).type(torch.float32)
                else:
                    embedding = torch.matmul(embedding_all_types, output["cell_prop"])
                    cell_type_existed = (output["cell_prop"] > 0.01).type(torch.int8).type(torch.float32)

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


class Decoder_MLP(BaseDecoder):
    """
    A normal MLP decoder.
    """

    def __init__(self, args: BaseModelConfig):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim  # input dimension of the Encoder
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(self.latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, np.prod(self.input_dim)),
                nn.Sigmoid(),  # to ensure the output is in the range [0, 1]
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
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

        for i in range(max_depth):
            # print('i', i, self.layers[i])
            out = self.layers[i](out)

            # if i == 0:
            #     out = out.reshape(z.shape[0], 128, 4, 4)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output
