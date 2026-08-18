import torch
import torch.nn as nn
from typing import List, Optional

from ...configs import ModelConfig
from ...models.base import (ModelOutput, build_cell_prop_from_head_output,
                            get_cell_prop_head_output_dim,
                            LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, EPS,
                            BaseEncoder, has_usable_labels,
                            resolve_cancer_cell_type_index_from_config)
from vaedecon.models.base.positional_encoding import PositionalEncoding


class _TokenRefinementBranch(nn.Module):
    """Lightweight residual refinement for one token family."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.norm(x)
        refined = self.linear(refined)
        refined = self.activation(refined)
        refined = self.dropout(refined)
        return x + refined


class GeneTransformerEncoder(BaseEncoder):
    """
    Optimized Transformer Encoder using Cross-Attention (Perceiver-like).
    Scales linearly with the number of genes O(N_genes), not quadratically.
    """

    def __init__(self, args: ModelConfig, data_config=None, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()
        self.args = args
        self.data_config = data_config
        if isinstance(args.input_dim, (tuple, list)):
            self.input_dim = args.input_dim[1]  # Assume (1, G) format
        else:
            self.input_dim = args.input_dim  # Number of genes (G)
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.predict_cell_prop = args.predict_cell_prop
        self.using_positional_encoding = args.using_positional_encoding
        self.cell_prop_activation_function = args.cell_prop_activation_function
        self.cancer_cell_type_index = None

        # Config
        self.d_model = getattr(args, 'transformer_d_model', 256)
        self.nhead = getattr(args, 'transformer_nhead', 8)
        self.num_layers = getattr(args, 'transformer_num_layers', 4)  # Can be deeper now
        self.dim_feedforward = getattr(args, 'transformer_dim_feedforward', 1024)
        self.dropout = getattr(args, 'transformer_dropout', 0.1)

        # 1. Gene Embedding
        # We use a smaller linear layer or shared embedding to save parameters if G is huge
        self.gene_id_embedding = nn.Parameter(torch.randn(self.input_dim, self.d_model))
        self.value_projector = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU()
        )

        # 2. Explicit queries (global + per-cell-type)
        if self.predict_cell_prop:
            self.global_query = nn.Parameter(torch.randn(1, self.d_model))
        else:
            self.global_query = None
        self.cell_type_queries = nn.Parameter(torch.randn(self.n_cell_types, self.d_model))

        # 3. Cross-Attention Layer (The "Compression" Step)
        # Queries: Latents (Small), Keys/Values: Genes (Large)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(self.d_model)
        self.norm_latents = nn.LayerNorm(self.d_model)

        # 4. Self-Attention Stack (Deep processing on the small latent space)
        # Once data is compressed to K tokens, we can do deep processing cheaply
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

        # 5. Task-specific refinement branches and heads
        self.global_branch = (
            _TokenRefinementBranch(self.d_model, self.dropout)
            if self.predict_cell_prop else None
        )
        self.cell_type_branch = _TokenRefinementBranch(self.d_model, self.dropout)
        self.fc_mu_logvar = nn.Linear(self.d_model, self.latent_dim * 2)
        if self.predict_cell_prop:
            if self.cell_prop_activation_function == "sigmoid":
                self.cancer_cell_type_index = resolve_cancer_cell_type_index_from_config(self.args)
            self.fc_dd_alpha = nn.Linear(
                self.d_model,
                get_cell_prop_head_output_dim(
                    n_cell_types=self.n_cell_types,
                    activation_function=self.cell_prop_activation_function,
                ),
            )

        # Positional Encoding
        if self.using_positional_encoding:
            if position_encoding is None:
                raise ValueError("position_encoding parameter must be provided.")
            self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                output_layer_levels: Optional[List[int]] = None, eps: float = EPS) -> ModelOutput:

        B, _ = x.shape
        device = x.device

        # --- 1. Prepare Gene Inputs (Keys/Values) ---
        # (B, G, 1) -> (B, G, d_model)
        val_emb = self.value_projector(x.unsqueeze(-1))
        # Add Identity: (1, G, d_model) + (B, G, d_model)
        gene_kv = self.gene_id_embedding.unsqueeze(0) + val_emb

        # --- 2. Prepare explicit latent queries ---
        cell_type_queries = self.cell_type_queries.unsqueeze(0).expand(B, -1, -1)
        if self.predict_cell_prop:
            global_query = self.global_query.unsqueeze(0).expand(B, -1, -1)
            latents = torch.cat((global_query, cell_type_queries), dim=1)
        else:
            latents = cell_type_queries

        # --- 3. Cross-Attention (The Magic Fix) ---
        # Q: Latents, K: Genes, V: Genes
        # This extracts info from 10,000 genes into ~20 vectors
        # Use PyTorch 2.0 Scaled Dot Product Attention if available for speed
        attn_out, _ = self.cross_attn(
            query=self.norm_latents(latents),
            key=gene_kv,
            value=gene_kv
        )
        # Residual connection
        latents = latents + attn_out

        # --- 4. Deep Processing (Self-Attention) ---
        # Now we only process the small latent sequence
        latents = self.transformer(latents)

        # --- 5. Extract Outputs ---
        start_idx = 0
        if self.predict_cell_prop:
            global_out = latents[:, 0, :]
            start_idx = 1

        type_out = latents[:, start_idx: start_idx + self.n_cell_types, :]
        if self.predict_cell_prop and self.global_branch is not None:
            global_out = self.global_branch(global_out)
        type_out = self.cell_type_branch(type_out)

        # --- 6. Latent Projection (Same as before) ---
        mu_logvar = self.fc_mu_logvar(type_out)
        mu_raw, logvar_raw = torch.chunk(mu_logvar, chunks=2, dim=-1)

        mu_all_types = mu_raw.permute(0, 2, 1)
        logvar_all_types = logvar_raw.permute(0, 2, 1)
        logvar_all_types = torch.clamp(logvar_all_types, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX)

        # --- 7. Cell Proportions ---
        output = ModelOutput()
        if self.predict_cell_prop:
            cell_prop, dd_alpha = build_cell_prop_from_head_output(
                head_output=self.fc_dd_alpha(global_out),
                activation_function=self.cell_prop_activation_function,
                n_cell_types=self.n_cell_types,
                eps=eps,
                cancer_cell_type_index=self.cancer_cell_type_index,
            )
            output['dd_alpha'] = dd_alpha
        elif has_usable_labels(y):
            cell_prop = y
        else:
            cell_prop = None
            output['dd_alpha'] = None

        # --- 8. Positional Encoding Logic (Legacy) ---
        if self.using_positional_encoding and self.position_encoding is not None and cell_prop is not None:
            if cell_prop.ndim == 3: cell_prop = cell_prop.squeeze(-1)
            exists = (cell_prop >= 0.01).float()
            exists_mask = exists.unsqueeze(1)
            pe_matrix = self.position_encoding().to(device)
            pe_to_add = pe_matrix.unsqueeze(0)
            mu_all_types = mu_all_types + (exists_mask * pe_to_add)

        # --- 9. Final Aggregation ---
        mu_mean = mu_all_types.mean(dim=-1)
        logvar_mean = logvar_all_types.mean(dim=-1)

        bulk_context_feature = global_out if self.predict_cell_prop else type_out.mean(dim=1)

        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['logvar_all_types'] = logvar_all_types
        output['mu_all_types'] = mu_all_types
        output['bulk_context_feature'] = bulk_context_feature
        output['cell_type_context_features'] = type_out
        output['cell_prop_feature'] = bulk_context_feature
        output['cell_prop'] = cell_prop
        if cell_prop is not None:
            output['cell_type_existed'] = (cell_prop >= 0.01).float()

        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__}
