import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from ...models.base import (BaseModelConfig, ModelOutput, reparameterize_dirichlet,
                            LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, EPS,
                            BaseEncoder)
from vaedecon.models.gnn.positional_encoding import PositionalEncoding


class GeneTransformerEncoder(BaseEncoder):
    """
    Transformer-based Encoder for Gene Expression.
    Treats genes as a set of tokens and uses learnable 'Cell Type Tokens'
    to query the gene set for deconvolution.
    """

    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim  # Number of genes (G)
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.predict_cell_prop = args.predict_cell_prop
        self.using_positional_encoding = args.using_positional_encoding

        # Transformer Config
        # You might want to add these to your args
        self.d_model = getattr(args, 'transformer_d_model', 256)
        self.nhead = getattr(args, 'transformer_nhead', 8)
        self.num_layers = getattr(args, 'transformer_num_layers', 4)
        self.dim_feedforward = getattr(args, 'transformer_dim_feedforward', 1024)
        self.dropout = getattr(args, 'transformer_dropout', 0.1)

        # 1. Gene Identity Embedding (The "Position" of the gene in the set)
        # Shape: (G, d_model)
        self.gene_id_embedding = nn.Parameter(torch.randn(np.prod(self.input_dim), self.d_model))

        # 2. Value Embedding (Project scalar expression value to vector)
        # We project the scalar value x_i to d_model and add it to gene_id_embedding
        self.value_projector = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU()
        )

        # 3. Cell Type Query Tokens (Learnable tokens that will become our latent variables)
        # Shape: (n_cell_types, d_model)
        self.cell_type_queries = nn.Parameter(torch.randn(self.n_cell_types, self.d_model))

        # 4. Global Context Token (For predicting cell proportions)
        if self.predict_cell_prop:
            self.global_query = nn.Parameter(torch.randn(1, self.d_model))

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,  # Important: inputs are (Batch, Seq, Dim)
            norm_first=True  # Pre-LN is generally more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 6. Heads
        # Map from d_model to latent_dim * 2 (mu, logvar)
        self.fc_mu_logvar = nn.Linear(self.d_model, self.latent_dim * 2)

        if self.predict_cell_prop:
            self.fc_dd_alpha = nn.Linear(self.d_model, self.n_cell_types)

        # External Positional Encoding (Optional, for spatial/temporal data)
        if self.using_positional_encoding:
            if position_encoding is None:
                raise ValueError("position_encoding parameter must be provided.")
            self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                output_layer_levels: Optional[List[int]] = None, eps: float = EPS) -> ModelOutput:

        # x shape: (Batch, Genes)
        B, G = x.shape
        device = x.device

        # --- 1. Prepare Input Sequence ---

        # A. Embed Genes
        # Reshape x for projection: (B, G, 1)
        x_reshaped = x.unsqueeze(-1)
        # Project values: (B, G, d_model)
        val_emb = self.value_projector(x_reshaped)
        # Add Gene Identity (Broadcasting): (1, G, d_model) + (B, G, d_model)
        gene_tokens = self.gene_id_embedding.unsqueeze(0) + val_emb

        # B. Prepare Queries (Cell Types)
        # Expand queries for batch: (B, n_cell_types, d_model)
        type_tokens = self.cell_type_queries.unsqueeze(0).expand(B, -1, -1)

        # C. Prepare Global Query (if needed)
        if self.predict_cell_prop:
            global_token = self.global_query.unsqueeze(0).expand(B, -1, -1)
            # Concatenate: [Global, Type1...TypeK, Gene1...GeneG]
            # Sequence Length = 1 + K + G
            full_seq = torch.cat([global_token, type_tokens, gene_tokens], dim=1)
        else:
            # Concatenate: [Type1...TypeK, Gene1...GeneG]
            full_seq = torch.cat([type_tokens, gene_tokens], dim=1)

        # --- 2. Transformer Pass ---
        # Self-attention allows Type tokens to attend to Gene tokens
        out_seq = self.transformer(full_seq)

        # --- 3. Extract Outputs ---

        # Identify indices
        start_idx = 0
        if self.predict_cell_prop:
            # Extract Global Token (Index 0)
            global_out = out_seq[:, 0, :]  # (B, d_model)
            start_idx = 1

        # Extract Cell Type Tokens
        # (B, n_cell_types, d_model)
        type_out = out_seq[:, start_idx: start_idx + self.n_cell_types, :]

        # --- 4. Latent Projection ---

        # Predict Mu/LogVar for each cell type
        # Input: (B, C, d_model) -> Output: (B, C, L*2)
        mu_logvar = self.fc_mu_logvar(type_out)
        mu_raw, logvar_raw = torch.chunk(mu_logvar, chunks=2, dim=-1)

        # Permute to standard format: (B, Latent, CellTypes)
        mu_all_types = mu_raw.permute(0, 2, 1)
        logvar_all_types = logvar_raw.permute(0, 2, 1)
        logvar_all_types = torch.clamp(logvar_all_types, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        # --- 5. Cell Proportions ---
        output = ModelOutput()

        if self.predict_cell_prop:
            # Use the Global Token to predict proportions
            dd_alpha = F.softplus(self.fc_dd_alpha(global_out)) + eps
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha, device=device)
        elif y is not None:
            cell_prop = y
        else:
            cell_prop = None

        # --- 6. External Positional Encoding Logic (Legacy support) ---
        # (Same logic as your MLP encoder)
        if self.using_positional_encoding and self.position_encoding is not None and cell_prop is not None:
            if cell_prop.ndim == 3: cell_prop = cell_prop.squeeze(-1)
            exists = (cell_prop >= 0.01).float()
            exists_mask = exists.unsqueeze(1)
            pe_matrix = self.position_encoding.to(device)
            pe_to_add = pe_matrix.t().unsqueeze(0)
            mu_all_types = mu_all_types + (exists_mask * pe_to_add)

        # --- 7. Final Aggregation ---
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
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__}
