import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

# ... (Keep your existing imports) ...
from ...models.base import (BaseModelConfig, ModelOutput, reparameterize_dirichlet,
                            LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, EPS,
                            BaseEncoder, BaseDecoder)
from vaedecon.models.base.positional_encoding import PositionalEncoding


class ResidualBlock(nn.Module):
    """
    A simple residual block for MLPs.
    Structure: Input -> [Linear->LN->GELU->Dropout] x 2 -> Add Input
    """

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=EPS),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=EPS),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderResMLP(BaseEncoder):
    """
    Residual MLP Encoder.
    """

    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.using_positional_encoding = args.using_positional_encoding
        self.predict_cell_prop = args.predict_cell_prop

        # Config
        # Suggestion: Make hidden dims deeper, e.g., [1024, 1024, 512, 512]
        self.hidden_dims: List[int] = getattr(args, 'encoder_hidden_dims', [1024, 512, 512])
        self.dropout_rate = args.encoder_dropout_rate

        self.layers = nn.ModuleList()
        current_input_size = np.prod(self.input_dim)

        # Build Residual Layers
        for i, hidden_dim_size in enumerate(self.hidden_dims):
            # 1. Dimensionality Reduction / Projection Step
            projector = nn.Sequential(
                nn.Linear(current_input_size, hidden_dim_size),
                nn.LayerNorm(hidden_dim_size, eps=EPS),
                nn.GELU(),
                nn.Dropout(p=self.dropout_rate[i]) if self.dropout_rate[i] > 0 else nn.Identity(),
            )

            # 2. Residual Step (Deepening the representation)
            # We add a residual block at this dimension level to increase capacity
            res_block = ResidualBlock(hidden_dim_size, dropout_rate=self.dropout_rate[i])

            self.layers.append(nn.Sequential(projector, res_block))
            current_input_size = hidden_dim_size

        self.depth = len(self.layers)

        # --- Heads (Same as before) ---
        self.fc_mu_logvar = nn.Linear(self.hidden_dims[-1], self.n_cell_types * self.latent_dim * 2)
        if self.predict_cell_prop:
            self.fc_dd_alpha = nn.Linear(self.hidden_dims[-1], self.n_cell_types)

        # Positional Encoding setup (Same as before)
        if self.using_positional_encoding:
            if position_encoding is None:
                raise ValueError("position_encoding parameter must be provided.")
            self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                output_layer_levels: Optional[List[int]] = None, eps: float = EPS) -> ModelOutput:

        out = x.view(x.size(0), -1)

        output = ModelOutput()
        for i, layer_block in enumerate(self.layers):
            out = layer_block(out)
            if output_layer_levels and (i + 1) in output_layer_levels:
                output[f"embedding_layer_{i + 1}"] = out

        if output_layer_levels and -1 in output_layer_levels:
            output[f"embedding_layer_{-1}"] = out

        # --- Latent Space Projection (Same logic as previous) ---
        mu_logvar_flat = self.fc_mu_logvar(out)
        mu_logvar_structured = mu_logvar_flat.view(-1, self.n_cell_types, 2 * self.latent_dim)
        mu_raw, logvar_raw = torch.chunk(mu_logvar_structured, chunks=2, dim=-1)

        mu_all_types = mu_raw.permute(0, 2, 1)
        logvar_all_types = logvar_raw.permute(0, 2, 1)
        logvar_all_types = torch.clamp(logvar_all_types, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        if self.predict_cell_prop:
            dd_alpha = F.softplus(self.fc_dd_alpha(out)) + eps
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha, device=x.device)
        elif y is not None:
            cell_prop = y
        else:
            cell_prop = None

        # Positional Encoding Logic (Same as before)
        if self.using_positional_encoding and self.position_encoding is not None and cell_prop is not None:
            if cell_prop.ndim == 3: cell_prop = cell_prop.squeeze(-1)
            exists = (cell_prop >= 0.01).float()
            exists_mask = exists.unsqueeze(1)
            pe_matrix = self.position_encoding.to(x.device)
            pe_to_add = pe_matrix.t().unsqueeze(0)
            mu_all_types = mu_all_types + (exists_mask * pe_to_add)

        mu_mean = mu_all_types.mean(dim=-1)
        logvar_mean = logvar_all_types.mean(dim=-1)

        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['logvar_all_types'] = logvar_all_types
        output['mu_all_types'] = mu_all_types
        output['cell_prop'] = cell_prop
        if cell_prop is not None:
            output['cell_type_existed'] = (cell_prop >= 0.01).float()

        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()}, "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__}


class DecoderResMLP(BaseDecoder):
    """
    Residual MLP Decoder.
    """

    def __init__(self, args: BaseModelConfig):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dims = getattr(args, 'decoder_hidden_dims', [512, 512, 1024])
        self.dropout_rate = args.decoder_dropout_rate

        self.layers = nn.ModuleList()
        input_size = self.latent_dim

        for i, hidden_dim_size in enumerate(self.hidden_dims):
            # 1. Projection
            projector = nn.Sequential(
                nn.Linear(input_size, hidden_dim_size),
                nn.LayerNorm(hidden_dim_size, eps=1e-6),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate[i]) if self.dropout_rate[i] > 0 else nn.Identity(),
            )
            # 2. Residual Block
            res_block = ResidualBlock(hidden_dim_size, dropout_rate=self.dropout_rate[i])

            self.layers.append(nn.Sequential(projector, res_block))
            input_size = hidden_dim_size

        # Output layer
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], np.prod(self.input_dim)),
            nn.Softplus(threshold=1),
            nn.Dropout(p=self.dropout_rate[-1]) if self.dropout_rate[-1] > 0 else nn.Identity(),
        )

        self.depth = len(self.layers) + 1

    def forward(self, z: torch.Tensor, output_layer_levels: Optional[List[int]] = None) -> ModelOutput:
        original_shape = z.shape

        if z.dim() == 3:
            z = z.permute(0, 2, 1)
            out = z.reshape(-1, self.latent_dim)
        else:
            out = z

        output = ModelOutput()

        for i, layer in enumerate(self.layers):
            out = layer(out)
            if output_layer_levels and (i + 1) in output_layer_levels:
                output[f"reconstruction_layer_{i + 1}"] = out

        out = self.final_layer(out)

        if len(original_shape) == 3:
            out = out.view(original_shape[0], original_shape[2], -1)
            out = out.permute(0, 2, 1)

        output["reconstruction"] = out
        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()}, "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__}
