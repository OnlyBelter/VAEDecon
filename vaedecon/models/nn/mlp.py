"""Proposed multilayer perceptron architectures as a baseline"""

from typing import List, Optional, Union

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ...models.base import (BaseModelConfig, ModelOutput, reparameterize_dirichlet,
                                  LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, EPS,
                                  BaseEncoder, BaseDecoder)
from ...models.nn.positional_encoding import PositionalEncoding
# from ....models.base.base_utils import
# from ..base_architectures import BaseDecoder, BaseEncoder
# from ..utils import ResBlock


class EncoderMLP(BaseEncoder):
    """
    A Normal MLP encoder.
    """

    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.using_positional_encoding = args.using_positional_encoding

        # Hidden dimension for the MLP body
        self.hidden_dims: List[int] = getattr(args, 'encoder_hidden_dims', [1024, 512, 512])

        self.dropout_rate = args.encoder_dropout_rate
        self.layers = nn.ModuleList()
        self.predict_cell_prop = args.predict_cell_prop
        # self.n_channels = 1
        self.device_param = nn.Parameter(torch.empty(0))  # To easily get the device of the model

        # Positional Encoding
        if self.using_positional_encoding:
            if position_encoding is None:
                raise ValueError("If using positional encoding, the position_encoding parameter must be provided.")
            self.position_encoding = position_encoding

        # Build MLP layers
        current_input_size = np.prod(self.input_dim)
        for i, hidden_dim_size in enumerate(self.hidden_dims):
            layer_block = nn.Sequential(
                nn.Linear(current_input_size, hidden_dim_size),
                nn.LayerNorm(hidden_dim_size, eps=EPS),
                nn.GELU(),  # GELU or LeakyReLU often perform better than ReLU
                nn.Dropout(p=self.dropout_rate[i]) if (self.dropout_rate[i] > 0) else nn.Identity(),
            )
            self.layers.append(layer_block)
            current_input_size = hidden_dim_size

        # self.layers = layers
        self.depth = len(self.layers)

        # self.fc_mu = nn.Linear(in_features=self.hidden_dims[-1],
        #                            out_features=args.latent_dim * args.n_cell_types)
        # Output heads for mu and logvar (vectorized for speeding up)
        self.fc_mu_logvar = nn.Linear(self.hidden_dims[-1], self.n_cell_types * self.latent_dim * 2)
        # self.fc_mu_list = nn.ModuleList(
        #     [nn.Linear(self.hidden_dims[-1], self.latent_dim) for _ in range(self.n_cell_types)]
        # )
        # self.fc_logvar_list = nn.ModuleList(
        #     [nn.Linear(self.hidden_dims[-1], self.latent_dim) for _ in range(self.n_cell_types)]
        # )
        # self.log_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        if self.predict_cell_prop:
            # self.cell_prop = nn.Sequential(
            #     nn.Linear(self.hidden_dims[-1], self.n_cell_types),
            #     nn.Softmax(dim=1)
            # )
            # Proportion head (outputs Dirichlet distribution parameters)
            self.fc_dd_alpha = nn.Linear(self.hidden_dims[-1], self.n_cell_types)
        # self.position_encoding = self.position_encoding.to(self.embedding.weight.device)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                output_layer_levels: Optional[List[int]] = None, eps: float = EPS) -> ModelOutput:
        """Forward method

        Args:
            x (torch.Tensor): The input data (cells x genes)
            y (torch.Tensor, optional): The cell proportions of the input data. Defaults to None.
            output_layer_levels (List[int], optional): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.
            eps (float, optional): A small value to avoid division by zero. Default: 1e-6.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level.
        """
        current_device = self.device_param.device  # Infer device from dummy parameter
        x = x.to(current_device)  # B, n_genes

        # max_depth = self.depth
        if self.using_positional_encoding and self.position_encoding is not None:
            self.position_encoding = self.position_encoding.to(current_device)

        if output_layer_levels is not None:
            # Check if the output layer levels are valid
            max_requested_depth = 0
            for level in output_layer_levels:
                if level == -1:
                    max_requested_depth = max(max_requested_depth, self.depth)
                elif not (0 < level <= self.depth):
                    raise ValueError(
                        f"Invalid output layer level: {level}. "
                        f"Must be between 1 and {self.depth} or -1 for the last layer."
                    )
                else:
                    max_requested_depth = max(max_requested_depth, level)

        output = ModelOutput()
        out = x.view(x.size(0), -1)  # flatten the input
        for i, layer_block in enumerate(self.layers):
            out = layer_block(out)
            if output_layer_levels is not None:
                if (i + 1) in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
        # Store final common embedding if requested as -1
        if output_layer_levels is not None and -1 in output_layer_levels:
            output[f"embedding_layer_{-1}"] = out

        # # using the proposed structure of latent space
        # # mu_all_types = self.fc_mu(out)  # (batch_size, latent_dim * n_cell_types)
        # mu_list = [mu(out) for mu in self.fc_mu_list]
        # logvar_list = [logvar(out) for logvar in self.fc_logvar_list]
        # # combine the mu_list into a tensor
        # mu_all_types = torch.stack(mu_list, dim=2)  # (batch_size, latent_dim, n_cell_types)
        # mu_mean = torch.mean(mu_all_types, dim=2)  # (batch_size, latent_dim)
        # # mu_all_types = mu_all_types.view((-1, self.latent_dim, self.n_cell_types))
        # logvar_all_types = torch.stack(logvar_list, dim=2)   # (batch_size, latent_dim, n_cell_types)

        # Predict mu and logvar for each cell type (vectorized)
        mu_logvar_flat = self.fc_mu_logvar(out)  # (batch_size, n_cell_types * latent_dim * 2)
        mu_logvar_structured = mu_logvar_flat.view(-1, self.n_cell_types, 2 * self.latent_dim)
        # Split into mu and logvar
        # mu_all_types_raw, logvar_all_types_raw shape: (batch_size, n_cell_types, latent_dim)
        mu_all_types_raw, logvar_all_types_raw = torch.chunk(mu_logvar_structured, chunks=2, dim=-1)
        # Permute to (batch_size, latent_dim, n_cell_types)
        mu_all_types = mu_all_types_raw.permute(0, 2, 1)  # (batch_size, latent_dim, n_cell_types)
        logvar_all_types = logvar_all_types_raw.permute(0, 2, 1)  # (batch_size, latent_dim, n_cell_types)
        # Clamp logvar to avoid numerical issues
        logvar_all_types = torch.clamp(logvar_all_types, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        mu_mean = mu_all_types.mean(dim=-1)  # (batch_size, latent_dim)
        logvar_mean = logvar_all_types.mean(dim=-1)

        # log_var = self.log_var(out)
        # TODO: getting cell proportions from DeSide
        if self.predict_cell_prop:
            # output["cell_prop"] = self.cell_prop(cell_embedding_before_mu).view((-1, self.n_cell_types, 1))
            dd_alpha = F.softplus(self.fc_dd_alpha(out)) + eps
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha, device=current_device)
        elif y is not None:
            cell_prop = y.to(current_device)
        else:
            raise NotImplementedError('If self.predict_cell_prop is False, '
                                      'y (cell proportions of cell types) must be provided. '
                                      'It can be predicted by DeSide.')
        # Using position encoding to shift mu for each cell type, adding the positional encoding
        if self.using_positional_encoding and self.position_encoding is not None and cell_prop is not None:
            # Ensure cell_prop is (B, n_cell_types)
            if cell_prop.ndim == 3 and cell_prop.shape[-1] == 1:  # if it's (B, N, 1)
                current_cell_prop = cell_prop.squeeze(-1)
            else:
                current_cell_prop = cell_prop
            exists = (current_cell_prop >= 0.01).float()  # (B, n_cell_types)

            # Assuming self.position_encoding() returns a tensor of shape (n_cell_types, latent_dim)
            pe_matrix = self.position_encoding  # (n_cell_types, latent_dim)
            if pe_matrix.device != current_device: pe_matrix = pe_matrix.to(current_device)

            # mu_all_types is (B, latent_dim, n_cell_types)
            # pe_matrix is (n_cell_types, latent_dim)
            # Reshape pe_matrix to (1, latent_dim, n_cell_types) for broadcasting
            pe_to_add = pe_matrix.t().unsqueeze(0)  # (1, latent_dim, n_cell_types)
            exists_mask = exists.unsqueeze(1)  # (B, 1, n_cell_types)

            mu_all_types = mu_all_types + (exists_mask * pe_to_add)  # (B, latent_dim, n_cell_types)
            # Recalculate mu_mean after adding positional encoding
            mu_mean = mu_all_types.mean(dim=-1)  # (B, latent_dim)

        # output['mu_list'] = mu_list
        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['logvar_all_types'] = logvar_all_types
        output['mu_all_types'] = mu_all_types
        # output['log_var'] = log_var
        # output['logvar_list'] = logvar_list
        output['cell_type_existed'] = (cell_prop >= 0.01).float()  # (B, n_cell_types)
        output['cell_prop'] = cell_prop

        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the MLP layers to extract features before the final VAE heads.
        """
        current_device = self.device_param.device  # Ensure device_param is defined
        x = x.to(current_device)
        out = x.view(x.size(0), -1)  # flatten the input
        for i, layer_block in enumerate(self.layers):
            out = layer_block(out)
        return out  # This is the tensor before fc_mu_logvar and fc_dd_alpha

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                }


class DecoderMLP(BaseDecoder):
    """
    A normal MLP decoder.
    """

    def __init__(self, args: BaseModelConfig):
        super().__init__()
        self.args = args
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
                'activation': nn.Softplus(threshold=1),  # make sure the output is > 0
                'dropout': nn.Dropout(p=self.dropout_rate[-1]) if (
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
        z = z.to(self.device)  # B, latent_dim

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            # if -1 in output_layer_levels:
            #     max_depth = self.depth
            # else:
            #     max_depth = max(output_layer_levels)

        out = z.reshape(-1, 1, self.latent_dim)
        # print('z.shape', z.shape)

        output = ModelOutput()
        for i, layer in enumerate(self.layers):
            out = layer['linear'](out)
            out = layer['norm'](out)
            out = layer['activation'](out)
            out = layer['dropout'](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

        # out = torch.clamp(self.relu(out), max=1.0)
        # out = torch.where(out >= 1, 1 - 1e-2, out)  # clamp the output to [0, 0.99], since it is scaled by a constant (default is 20)
        output["reconstruction"] = out
        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                }


class ClampLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(ClampLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)
