import os
import logging
from typing import Optional, List, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx  # Kept for fallback, though tensor ops are preferred
from torch_geometric.utils import from_networkx, add_self_loops
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint

# Assuming these imports exist in your project structure
from ..vae.vae_config import VAEConfig
from ...models.base import (ModelOutput, reparameterize_dirichlet, LOGVAR_CLAMP_MIN,
                            LOGVAR_CLAMP_MAX, EPS, NETWORK_CUTOFF, BaseEncoder)
from vaedecon.models.gnn.positional_encoding import PositionalEncoding

logger = logging.getLogger(__name__)
# Only add handler if not already added to avoid duplicate logs
if not logger.handlers:
    console = logging.StreamHandler()
    logger.addHandler(console)
    logger.setLevel(logging.INFO)


class EncoderSGNN(BaseEncoder):
    """
    GNN-based encoder using a PPI network + optional gene-level features.
    Refactored for correct PyG batching and performance.
    """

    def __init__(self, args: VAEConfig, position_encoding: Optional[PositionalEncoding] = None):
        super().__init__()
        self.args = args
        # Dummy parameter to track device
        self.device_param = nn.Parameter(torch.empty(0))

        self.use_gradient_checkpointing = getattr(args, 'use_gradient_checkpointing', False)
        self.max_batch_size_for_gnn = getattr(args, 'max_batch_size_for_gnn', 32)

        # --- 1. Load Gene List ---
        if args.input_gene_list_fp is None or not os.path.exists(args.input_gene_list_fp):
            raise FileNotFoundError(f"Gene list file not found: {args.input_gene_list_fp}")
        # Efficiently read only first column
        self.gene_list: List[str] = pd.read_csv(args.input_gene_list_fp, header=None, usecols=[0]).iloc[:, 0].tolist()

        # --- 2. Load PPI Network ---
        if not os.path.exists(args.ppi_file_path):
            raise FileNotFoundError(f"PPI file not found: {args.ppi_file_path}")
        net_df = pd.read_csv(args.ppi_file_path)

        # Filter and Standardize Columns
        if args.biogrid_flag:
            required_cols = ["g1_symbol", "g2_symbol"]
            if not all(col in net_df.columns for col in required_cols):
                raise ValueError(f"Biogrid PPI must have {required_cols}")
            net = net_df[required_cols].copy()
            net.columns = ["source", "target"]
        else:
            required_cols = ["g1_symbol", "g2_symbol", "conn"]
            if not all(col in net_df.columns for col in required_cols):
                raise ValueError(f"PPI file must have {required_cols}")
            net = net_df[required_cols].copy()
            net = net[net.conn >= NETWORK_CUTOFF]
            net.columns = ["source", "target", "conn"]

        # Optimized Filtering
        # Create a set for O(1) lookup
        gene_set = set(self.gene_list)

        # Filter edges where both nodes are in our gene list
        mask = net['source'].isin(gene_set) & net['target'].isin(gene_set)
        net = net[mask]
        net = net[net['source'] != net['target']]  # Remove self-loops (PyG adds them usually or we handle later)
        net.drop_duplicates(subset=["source", "target"], inplace=True)

        # Identify genes actually present in the graph
        unique_genes_in_ppi = pd.unique(net[["source", "target"]].values.ravel("K"))
        # Preserve order from original gene_list for consistency
        self.graph_gene_list = [gene for gene in self.gene_list if gene in set(unique_genes_in_ppi)]

        if not self.graph_gene_list:
            raise ValueError("No common genes found between PPI and input gene list.")

        # Map genes to 0..N indices
        gene_to_idx_map = {gene: idx for idx, gene in enumerate(self.graph_gene_list)}
        net['source_idx'] = net['source'].map(gene_to_idx_map)
        net['target_idx'] = net['target'].map(gene_to_idx_map)

        # Build Edge Index directly (Faster than NetworkX for large graphs)
        # Shape: [2, Num_Edges]
        source_indices = torch.tensor(net['source_idx'].values, dtype=torch.long)
        target_indices = torch.tensor(net['target_idx'].values, dtype=torch.long)
        edge_index = torch.stack([source_indices, target_indices], dim=0)

        # Make undirected if needed (usually PPIs are undirected)
        # PyG SAGEConv assumes directed edges for message passing, so for undirected graphs,
        # we usually need both (i,j) and (j,i).
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        # Remove duplicates after making undirected
        edge_index = torch.unique(edge_index, dim=1)

        self.register_buffer('edge_index', edge_index)

        # Indices to slice the input tensor x
        self.input_gene_filter_indices = [self.gene_list.index(g) for g in self.graph_gene_list]
        # Register as buffer to move to device automatically, but as long tensor
        self.register_buffer('filter_indices_tensor', torch.tensor(self.input_gene_filter_indices, dtype=torch.long))

        # --- 3. Gene Features ---
        if not os.path.exists(args.gene_mean_std_fp):
            raise FileNotFoundError(f"Gene features file not found: {args.gene_mean_std_fp}")

        gf_df = pd.read_csv(args.gene_mean_std_fp, index_col=0)
        # Reindex handles missing genes by putting NaN, then we fill 0
        gf_df_reindexed = gf_df.reindex(self.graph_gene_list).fillna(0.0)

        gf_mat = gf_df_reindexed.values
        self.register_buffer('gene_features', torch.tensor(gf_mat, dtype=torch.float32))

        # --- 4. Model Dimensions & Layers ---
        self.gnn_n_genes = len(self.graph_gene_list)
        self.gene_hidden_dim = args.gene_hidden_dim
        self.cell_latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.embd_col_dim = args.gnn_embd_col_dim
        self.drop_p = args.gnn_drop_p
        self.num_layers = args.gnn_num_layers
        self.predict_cell_prop = args.predict_cell_prop

        self.position_encoding = position_encoding if self.args.using_positional_encoding else None

        # GNN Encoder
        self.encoder = PPIEncoder(
            in_feats=1 + self.gene_features.shape[1],
            gene_hidden_dim=self.gene_hidden_dim,
            latent_dim=self.cell_latent_dim,
            num_layers=self.num_layers,
            drop_p=self.drop_p,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            num_nodes=self.gnn_n_genes  # Pass number of nodes for batching logic
        )

        # Post-GNN MLPs
        self.cell_mlp = nn.Sequential(
            nn.Linear(self.gene_hidden_dim * self.cell_latent_dim, self.embd_col_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.fc_mu_logvar = nn.Linear(self.embd_col_dim, self.n_cell_types * self.cell_latent_dim * 2)

        if self.predict_cell_prop:
            self.gnn_dd_alpha = nn.Linear(self.embd_col_dim, self.n_cell_types)

    def forward(self, x: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                sample_ids: Optional[List[str]] = None) -> ModelOutput:

        current_device = self.device_param.device
        x = x.to(current_device)
        batch_size = x.size(0)

        # Inference Split
        if batch_size > self.max_batch_size_for_gnn and not self.training:
            return self._forward_split_batch(x, y, sample_ids)

        # 1. Prepare Node Features
        # x: [B, All_Genes] -> x_sub: [B, Graph_Genes]
        x_sub = torch.index_select(x, 1, self.filter_indices_tensor).unsqueeze(-1)  # [B, N, 1]

        # Expand gene features: [B, N, F_gene]
        gf = self.gene_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate: [B, N, 1 + F_gene]
        node_features = torch.cat([x_sub, gf], dim=-1)

        # 2. GNN Forward
        # Note: PPIEncoder handles the flattening internally
        gene_embeddings = self.encoder(node_features, self.edge_index)

        # 3. MLP & Output Heads
        cell_embedding = self.cell_mlp(gene_embeddings)

        mu_logvar_flat = self.fc_mu_logvar(cell_embedding)
        mu_logvar_structured = mu_logvar_flat.view(-1, self.n_cell_types, self.cell_latent_dim * 2)

        mu_raw, logvar_raw = torch.chunk(mu_logvar_structured, 2, dim=-1)

        # Permute to [B, Latent, CellTypes] if that's what downstream expects
        # Based on your code: permute(0, 2, 1) -> [B, Latent_Dim, N_Cell_Types]
        mu_all_types = mu_raw.permute(0, 2, 1).contiguous()
        logvar_all_types = logvar_raw.permute(0, 2, 1).contiguous()

        logvar_all_types = torch.clamp(logvar_all_types, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

        mu_mean = torch.mean(mu_all_types, dim=2)
        logvar_mean = torch.mean(logvar_all_types, dim=2)

        output = ModelOutput()

        # Cell Proportions
        if self.predict_cell_prop:
            dd_alpha = F.softplus(self.gnn_dd_alpha(cell_embedding)) + EPS
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha, device=current_device)
        elif y is not None:
            cell_prop = y.to(current_device)
        else:
            # Fallback or Error
            cell_prop = torch.zeros(batch_size, self.n_cell_types, device=current_device)
            if self.training:
                raise ValueError("y must be provided if predict_cell_prop is False")

        output['cell_prop'] = cell_prop

        # Positional Encoding
        if self.position_encoding is not None:
            pe_matrix = self.position_encoding().to(current_device)
            exists = (cell_prop >= 0.01).float()
            # Broadcasting PE
            pe_to_add = pe_matrix.t().unsqueeze(0)  # [1, Latent, Types]
            exists_mask = exists.unsqueeze(1)  # [B, 1, Types]
            mu_all_types = mu_all_types + (pe_to_add * exists_mask)

        output['mu_all_types'] = mu_all_types
        output['logvar_all_types'] = logvar_all_types
        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['cell_type_existed'] = (cell_prop >= 0.01).float()

        return output

    def _forward_split_batch(self, x, y, sample_ids):
        """Split batch inference without clearing CUDA cache aggressively."""
        batch_size = x.size(0)
        chunk_size = self.max_batch_size_for_gnn
        outputs = []

        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            x_chunk = x[i:end_idx]
            y_chunk = y[i:end_idx] if y is not None else None

            # No gradients needed for inference usually
            with torch.no_grad():
                chunk_output = self.forward(x_chunk, y_chunk, None)
            outputs.append(chunk_output)

            # REMOVED torch.cuda.empty_cache() - this hurts performance more than it helps

        merged = ModelOutput()
        # Assuming ModelOutput is a dict-like or has keys
        keys = outputs[0].keys() if isinstance(outputs[0], dict) else outputs[0].__dict__.keys()

        for key in keys:
            vals = [out[key] for out in outputs if out[key] is not None]
            if vals:
                merged[key] = torch.cat(vals, dim=0)
            else:
                merged[key] = None
        return merged

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the cell embedding (output of cell_mlp) for use in fused models.
        Restored and optimized for the new GNN implementation.
        """
        current_device = self.device_param.device
        x = x.to(current_device)
        batch_size = x.size(0)

        # Handle batch splitting for memory safety (same logic as before)
        if batch_size > self.max_batch_size_for_gnn:
            features_list = []
            for i in range(0, batch_size, self.max_batch_size_for_gnn):
                end_idx = min(i + self.max_batch_size_for_gnn, batch_size)
                x_chunk = x[i:end_idx]

                # Process chunk
                chunk_features = self._extract_features_single(x_chunk)
                features_list.append(chunk_features)

            return torch.cat(features_list, dim=0)
        else:
            return self._extract_features_single(x)

    def _extract_features_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal helper to run the GNN and Cell MLP without the final projection heads.
        """
        batch_size = x.size(0)

        # 1. Prepare Node Features (Using the optimized buffers from __init__)
        # Select only graph genes: [B, Graph_Genes, 1]
        x_sub = torch.index_select(x, 1, self.filter_indices_tensor).unsqueeze(-1)

        # Expand gene features: [B, Graph_Genes, F_gene]
        gf = self.gene_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate: [B, Graph_Genes, 1 + F_gene]
        node_features = torch.cat([x_sub, gf], dim=-1)

        # 2. GNN Forward (Returns [B, N, Hidden])
        gene_embeddings = self.encoder(node_features, self.edge_index)

        # 3. MLP (Get the cell embedding: [B, embd_col_dim])
        cell_embedding = self.cell_mlp(gene_embeddings)

        return cell_embedding

    def get_config(self):
        return {
            "params": {"args": self.args.to_dict()},
            "module_name": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }


class PPIEncoder(nn.Module):  # Changed from L.LightningModule to nn.Module
    def __init__(self,
                 in_feats: int,
                 gene_hidden_dim: int,
                 latent_dim: int,
                 num_nodes: int,
                 num_layers=3,
                 drop_p=0.1,
                 use_gradient_checkpointing=False):
        super().__init__()
        self.gene_hidden_dim = gene_hidden_dim
        self.num_nodes = num_nodes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # GNN Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()  # Optional: Skip connections help deep GNNs

        current_dim = in_feats
        for i in range(num_layers):
            self.convs.append(SAGEConv(current_dim, gene_hidden_dim))
            # SAGEConv doesn't include activation/norm by default
            self.norms.append(nn.LayerNorm(gene_hidden_dim))

            # Linear projection for skip connection if dims change
            if current_dim != gene_hidden_dim:
                self.skips.append(nn.Linear(current_dim, gene_hidden_dim))
            else:
                self.skips.append(nn.Identity())

            current_dim = gene_hidden_dim

        self.dropout = nn.Dropout(drop_p)
        self.activation = nn.GELU()

        # Attention Pooling Heads
        self.feature_projector = nn.Linear(gene_hidden_dim, gene_hidden_dim)
        self.attention_queries = nn.Parameter(torch.empty(1, latent_dim, gene_hidden_dim))
        nn.init.xavier_uniform_(self.attention_queries)
        self.attention_scale_factor = 1.0 / (gene_hidden_dim ** 0.5)

        self.output_norm = nn.LayerNorm(gene_hidden_dim * latent_dim)
        self.final_activation = nn.GELU()

        # Cache for batched edge index
        self._cached_batch_size = -1
        self._cached_edge_index = None

    def _get_batched_edge_index(self, edge_index, batch_size, device):
        """
        Creates a block-diagonal edge index for the batch.
        Caches result to avoid re-computation.
        """
        if self._cached_batch_size == batch_size and self._cached_edge_index is not None:
            return self._cached_edge_index

        # edge_index: [2, E]
        # We want to repeat this for every sample in batch, adding offset
        num_edges = edge_index.size(1)

        # Create offsets: [0, N, 2N, ...]
        offsets = torch.arange(batch_size, device=device) * self.num_nodes
        offsets = offsets.view(-1, 1)  # [B, 1]

        # Broadcast addition
        # We need to repeat edge_index B times
        # src: [B, E], dst: [B, E]
        src = edge_index[0].unsqueeze(0) + offsets
        dst = edge_index[1].unsqueeze(0) + offsets

        batched_src = src.view(-1)  # [B*E]
        batched_dst = dst.view(-1)  # [B*E]

        batched_edge_index = torch.stack([batched_src, batched_dst], dim=0)

        self._cached_batch_size = batch_size
        self._cached_edge_index = batched_edge_index
        return batched_edge_index

    def forward(self, x: torch.Tensor, ppi_edge_index: torch.Tensor):
        """
        x: [Batch, Num_Genes, Features]
        ppi_edge_index: [2, Num_Edges] (Single graph structure)
        """
        batch_size, num_genes, num_feats = x.shape

        # 1. Flatten for PyG: [B*N, F]
        x_flat = x.view(batch_size * num_genes, num_feats)

        # 2. Get Batched Edge Index: [2, B*E]
        batched_edge_index = self._get_batched_edge_index(ppi_edge_index, batch_size, x.device)

        embedded = x_flat

        # 3. GNN Layers
        for i, conv in enumerate(self.convs):

            identity = embedded

            if self.use_gradient_checkpointing and self.training:
                out = checkpoint(conv, embedded, batched_edge_index, use_reentrant=False)
            else:
                out = conv(embedded, batched_edge_index)

            out = self.norms[i](out)
            out = self.activation(out)
            out = self.dropout(out)

            # Residual Connection
            if hasattr(self.skips, '__getitem__'):  # Safety check
                skip_val = self.skips[i](identity)
                out = out + skip_val

            embedded = out

        # 4. Reshape back: [B, N, Hidden]
        embedded = embedded.view(batch_size, num_genes, -1)

        # 5. Attention Pooling
        # Project features
        projected = self.feature_projector(embedded)  # [B, N, H]
        projected = F.gelu(projected)

        # Calculate Attention Scores: Q * K^T
        # Q: [1, Latent, H] -> [B, Latent, H]
        # K: projected [B, N, H] -> transpose -> [B, H, N]
        # Result: [B, Latent, N]
        attn_scores = torch.bmm(
            self.attention_queries.expand(batch_size, -1, -1),
            projected.transpose(1, 2)
        ) * self.attention_scale_factor

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Latent, N]

        # Weighted Sum: V (projected)
        # [B, Latent, N] x [B, N, H] -> [B, Latent, H]
        m_k_features = torch.bmm(attn_weights, projected)

        # Flatten: [B, Latent * H]
        output_flat = m_k_features.reshape(batch_size, -1)

        output_processed = self.output_norm(output_flat)
        output_processed = self.final_activation(output_processed)

        return output_processed
