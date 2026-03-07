"""Proposed multilayer perceptron architectures as a baseline"""

from typing import List, Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import warnings

from ...models.base import (BaseModelConfig, ModelOutput, reparameterize_dirichlet,
                                  LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, EPS,
                                  BaseEncoder, BaseDecoder)
from vaedecon.models.gnn.positional_encoding import PositionalEncoding
# from ....models.base.base_utils import
# from ..base_architectures import BaseDecoder, BaseEncoder
# from ..utils import ResBlock


class EncoderMLP(BaseEncoder):
    """
    A Vectorized MLP encoder that predicts parameters for all cell types simultaneously.
    """

    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.using_positional_encoding = args.using_positional_encoding
        self.predict_cell_prop = args.predict_cell_prop

        # Configuration
        self.hidden_dims: List[int] = getattr(args, 'encoder_hidden_dims', [1024, 512, 512])
        self.dropout_rate = args.encoder_dropout_rate

        # --- Body Layers ---
        self.layers = nn.ModuleList()
        current_input_size = np.prod(self.input_dim)

        for i, hidden_dim_size in enumerate(self.hidden_dims):
            block = nn.Sequential(
                nn.Linear(current_input_size, hidden_dim_size),
                nn.LayerNorm(hidden_dim_size, eps=EPS),
                nn.GELU(),
                nn.Dropout(p=self.dropout_rate[i]) if self.dropout_rate[i] > 0 else nn.Identity(),
            )
            self.layers.append(block)
            current_input_size = hidden_dim_size

        self.depth = len(self.layers)

        # --- Heads ---
        # 1. Mu and LogVar Head (Vectorized for all cell types)
        # Output shape: [Batch, n_cell_types * latent_dim * 2]
        self.fc_mu_logvar = nn.Linear(self.hidden_dims[-1], self.n_cell_types * self.latent_dim * 2)

        # 2. Cell Proportion Head (Dirichlet parameters)
        if self.predict_cell_prop:
            self.fc_dd_alpha = nn.Linear(self.hidden_dims[-1], self.n_cell_types)

        # Positional Encoding
        if self.using_positional_encoding:
            if position_encoding is None:
                raise ValueError("position_encoding parameter must be provided if using_positional_encoding is True.")
            self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                output_layer_levels: Optional[List[int]] = None, eps: float = EPS) -> ModelOutput:

        # x is already on the correct device. No need to move it.
        # Flatten input: (B, Genes)
        out = x.view(x.size(0), -1)

        # --- Feature Extraction (MLP Body) ---
        output = ModelOutput()
        for i, layer_block in enumerate(self.layers):
            out = layer_block(out)
            if output_layer_levels and (i + 1) in output_layer_levels:
                output[f"embedding_layer_{i + 1}"] = out

        if output_layer_levels and -1 in output_layer_levels:
            output[f"embedding_layer_{-1}"] = out

        # --- Latent Space Projection (Heads) ---

        # 1. Predict Mu and LogVar
        # Flat shape: (B, C * L * 2)
        mu_logvar_flat = self.fc_mu_logvar(out)

        # Reshape to (B, C, 2*L)
        mu_logvar_structured = mu_logvar_flat.view(-1, self.n_cell_types, 2 * self.latent_dim)

        # Split and Permute
        # raw shapes: (B, C, L)
        mu_raw, logvar_raw = torch.chunk(mu_logvar_structured, chunks=2, dim=-1)

        # Permute to standard format: (B, Latent, CellTypes)
        mu_all_types = mu_raw.permute(0, 2, 1)
        logvar_all_types = logvar_raw.permute(0, 2, 1)

        # Numerical stability
        logvar_all_types = torch.clamp(logvar_all_types, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        # 2. Predict Cell Proportions
        if self.predict_cell_prop:
            # Softplus ensures alpha > 0
            dd_alpha = F.softplus(self.fc_dd_alpha(out)) + eps
            output['dd_alpha'] = dd_alpha
            # Pass device explicitly to random generator if needed, or rely on tensor ops
            cell_prop = reparameterize_dirichlet(dd_alpha, device=x.device)
        elif y is not None:
            cell_prop = y
        else:
            cell_prop = None
            # Only warn once per runtime ideally, but keeping logic simple here
            # warnings.warn(...)

        # --- Positional Encoding Logic ---
        if self.using_positional_encoding and self.position_encoding is not None and cell_prop is not None:
            # Ensure cell_prop is (B, C)
            if cell_prop.ndim == 3: cell_prop = cell_prop.squeeze(-1)

            # Mask for existing cell types
            exists = (cell_prop >= 0.01).float()  # (B, C)
            exists_mask = exists.unsqueeze(1)  # (B, 1, C)

            # PE Matrix: (C, L) -> (1, L, C) for broadcasting
            # We use the registered buffer or move it to x.device
            pe_matrix = self.position_encoding.to(x.device)
            pe_to_add = pe_matrix.t().unsqueeze(0)

            # Add PE only to existing cell types
            mu_all_types = mu_all_types + (exists_mask * pe_to_add)

        # --- Final Aggregation ---
        mu_mean = mu_all_types.mean(dim=-1)  # (B, L)
        logvar_mean = logvar_all_types.mean(dim=-1)  # (B, L)

        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['logvar_all_types'] = logvar_all_types
        output['mu_all_types'] = mu_all_types
        output['cell_prop'] = cell_prop

        if cell_prop is not None:
            output['cell_type_existed'] = (cell_prop >= 0.01).float()

        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(x.size(0), -1)
        for layer_block in self.layers:
            out = layer_block(out)
        return out

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__}


class DecoderMLP(BaseDecoder):
    """
    A Shared MLP decoder.
    It can decode a single latent vector (B, L) OR a batch of cell types (B, L, C).
    """

    def __init__(self, args: BaseModelConfig):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim  # The output dimension, same as the input dimension of encoder
        self.latent_dim = args.latent_dim
        self.hidden_dims = getattr(args, 'decoder_hidden_dims', [512, 512, 1024])
        self.dropout_rate = args.decoder_dropout_rate

        self.layers = nn.ModuleList()
        input_size = self.latent_dim
        output_dim = int(np.prod(self.input_dim))

        # Build layers
        for i, hidden_dim_size in enumerate(self.hidden_dims):
            block = nn.Sequential(
                nn.Linear(input_size, hidden_dim_size),
                nn.LayerNorm(hidden_dim_size, eps=1e-6),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate[i]) if self.dropout_rate[i] > 0 else nn.Identity(),
            )
            self.layers.append(block)
            input_size = hidden_dim_size

        # Output layer (Gene expression reconstruction)
        # Output constrained to (0, 1)
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], output_dim),
            nn.Sigmoid(),
            # nn.Dropout(p=self.dropout_rate[-1]) if self.dropout_rate[-1] > 0 else nn.Identity(),
        )

        self.depth = len(self.layers) + 1

    def forward(self, z: torch.Tensor, output_layer_levels: Optional[List[int]] = None) -> ModelOutput:
        """
        Args:
            z: (Batch, Latent) OR (Batch, Latent, n_cell_types)
            output_layer_levels: Optional list of layer indices to output intermediate results from.
        Returns:
            ModelOutput containing 'reconstruction' and optionally intermediate layers.
        """
        # Handle Shape: We want to process everything as a flat batch of vectors
        original_shape = z.shape

        if z.dim() == 3:
            # Case: (Batch, Latent, n_cell_types) -> We want to decode each type individually
            # Permute to (Batch, n_cell_types, Latent)
            z = z.permute(0, 2, 1)
            # Flatten to (Batch * n_cell_types, Latent)
            out = z.reshape(-1, self.latent_dim)
        else:
            # Case: (Batch, Latent)
            out = z

        output = ModelOutput()

        # Run through hidden layers
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if output_layer_levels and (i + 1) in output_layer_levels:
                output[f"reconstruction_layer_{i + 1}"] = out

        # Final layer
        out = self.final_layer(out)

        # Reshape back to original structure if necessary
        if len(original_shape) == 3:
            # out is currently (Batch * n_cell_types, Genes)
            # Reshape to (Batch, n_cell_types, Genes)
            # Then permute to (Batch, Genes, n_cell_types) if that is your preferred format
            # Usually for reconstruction we want (Batch, Genes, n_cell_types) to sum over types later
            out = out.view(original_shape[0], original_shape[2], -1)  # (B, C, Genes)
            out = out.permute(0, 2, 1)  # (B, Genes, C)

        output["reconstruction"] = out
        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__}


class ClampLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(ClampLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)
