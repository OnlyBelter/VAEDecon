import os
import logging
from typing import Optional, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configs import ModelConfig, DataConfig
from ...models.base import (ModelOutput, build_cell_prop_from_head_output,
                            get_cell_prop_head_output_dim, LOGVAR_CLAMP_MIN,
                            LOGVAR_CLAMP_MAX, EPS, NETWORK_CUTOFF, BaseEncoder,
                            resolve_cancer_cell_type_index_from_config)
from vaedecon.models.base.positional_encoding import PositionalEncoding

logger = logging.getLogger(__name__)
# Only add handler if not already added to avoid duplicate logs
if not logger.handlers:
    console = logging.StreamHandler()
    logger.addHandler(console)
    logger.setLevel(logging.INFO)


class EncoderSGNN(BaseEncoder):
    """
    GNN-based encoder using a PPI network + optional gene-level features.
    Pipeline:
      1. Load gene list and PPI network from files, build graph structure.
      2. Prepare node features for each gene (expression value + prior gene features).
      3. Extract graph-level representation via PPIEncoder (GNN + Attention Pooling).
      4. Map to VAE parameters (mu / logvar) via MLP heads.
    """
    # TODO: consider using specific configure class for each encoder type, instead of a monolithic ModelConfig with many optional fields.
    def __init__(
        self,
        args: ModelConfig,
        data_config: DataConfig = None,
        position_encoding: Optional[PositionalEncoding] = None
    ):
        super().__init__()
        self.args = args
        self.data_config = data_config
        # Dummy parameter used solely to track the current device.
        # nn.Parameter automatically moves with .to(device).
        self.device_param = nn.Parameter(torch.empty(0))

        self.use_gradient_checkpointing = getattr(args, 'use_gradient_checkpointing', False)

        # During inference, batches larger than this are split to avoid OOM.
        self.max_batch_size_for_gnn = getattr(args, 'max_batch_size_for_gnn', 32)

        # --- 1. Load Gene List ---
        if args.input_gene_list_fp is None or not os.path.exists(args.input_gene_list_fp):
            raise FileNotFoundError(f"Gene list file not found: {args.input_gene_list_fp}")
        # Efficiently read only first column
        self.gene_list: List[str] = pd.read_csv(args.input_gene_list_fp, header=None, usecols=[0]).iloc[:, 0].tolist()

        # --- 2. Load and Process PPI Network ---
        if not data_config.ppi_file_path or not os.path.exists(data_config.ppi_file_path):
            raise FileNotFoundError(f"PPI file not found: {data_config.ppi_file_path}")
        net_df = pd.read_csv(data_config.ppi_file_path)

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
            net = net[net.conn >= NETWORK_CUTOFF]  # Only keep edges above cutoff
            net.columns = ["source", "target", "conn"]

        # Use a set for O(1) membership lookup when filtering edges.
        gene_set = set(self.gene_list)
        mask = net['source'].isin(gene_set) & net['target'].isin(gene_set)
        net = net[mask]
        net = net[net['source'] != net['target']]  # Remove self-loops
        net.drop_duplicates(subset=["source", "target"], inplace=True)

        # Retain only genes that actually appear in the filtered PPI,
        # preserving the original order from gene_list for reproducibility.
        unique_genes_in_ppi = pd.unique(net[["source", "target"]].values.ravel("K"))
        self.graph_gene_list = [g for g in self.gene_list if g in set(unique_genes_in_ppi)]

        if not self.graph_gene_list:
            raise ValueError("No common genes found between PPI and input gene list.")

        # Map gene names to contiguous integer indices {0, ..., N-1}.
        gene_to_idx_map = {gene: idx for idx, gene in enumerate(self.graph_gene_list)}
        net['source_idx'] = net['source'].map(gene_to_idx_map)
        net['target_idx'] = net['target'].map(gene_to_idx_map)

        # Build edge_index in PyG format: shape [2, num_edges].
        # Row 0 = source node indices, Row 1 = target node indices.
        source_indices = torch.tensor(net['source_idx'].values, dtype=torch.long)
        target_indices = torch.tensor(net['target_idx'].values, dtype=torch.long)
        edge_index = torch.stack([source_indices, target_indices], dim=0)

        # PPI graphs are undirected; SAGEConv performs directed message passing,
        # so we explicitly add both (i->j) and (j->i) directions.
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        # Remove duplicates after making undirected
        edge_index = torch.unique(edge_index, dim=1)

        # register_buffer: not a learnable parameter, but moves with .to(device)
        # and is included in state_dict for checkpointing.
        self.register_buffer('edge_index', edge_index)

        # Indices to slice the full input gene vector down to graph genes only.
        self.input_gene_filter_indices = [self.gene_list.index(g) for g in self.graph_gene_list]
        self.register_buffer('filter_indices_tensor', torch.tensor(self.input_gene_filter_indices, dtype=torch.long))

        # --- 3. Load Prior Gene Features ---
        # These are per-gene statistics (e.g., mean, std across cell types) used as auxiliary node features.
        if not os.path.exists(args.gene_mean_std_fp):
            raise FileNotFoundError(f"Gene features file not found: {args.gene_mean_std_fp}")

        gf_df = pd.read_csv(args.gene_mean_std_fp, index_col=0)
        # reindex ensures the row order matches graph_gene_list; missing genes are filled with 0.
        gf_df_reindexed = gf_df.reindex(self.graph_gene_list).fillna(0.0)
        gf_mat = gf_df_reindexed.values  # Shape: [N_genes_in_graph, F_gene], F means node features per gene
        self.register_buffer('gene_features', torch.tensor(gf_mat, dtype=torch.float32))

        # --- 4. Model Dimensions and Sub-modules ---
        self.gnn_n_genes = len(self.graph_gene_list)  # Number of graph nodes N
        self.gene_hidden_dim = args.gene_hidden_dim  # Hidden dimension H per node after GNN
        self.cell_latent_dim = args.latent_dim  # Number of attention queries (≈ latent slots)
        self.n_cell_types = args.n_cell_types
        self.embd_col_dim = args.gnn_embd_col_dim
        self.drop_p = args.gnn_drop_p
        self.num_layers = args.gnn_num_layers
        self.predict_cell_prop = args.predict_cell_prop
        self.cell_prop_activation_function = args.cell_prop_activation_function
        self.cancer_cell_type_index = None

        self.position_encoding = position_encoding if self.args.using_positional_encoding else None
        
        # Determine top-k for attention
        self.topk = min(getattr(args, 'gnn_topk_attention', 1024), self.gnn_n_genes)

        # GNN Encoder: input feature dim = 1 (expression) + F_gene (prior features)
        self.encoder = PPIEncoder(
            in_feats=1 + self.gene_features.shape[1],
            gene_hidden_dim=self.gene_hidden_dim,
            latent_dim=self.cell_latent_dim,
            num_layers=self.num_layers,
            drop_p=self.drop_p,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            num_nodes=self.gnn_n_genes,  # Pass number of nodes for batching logic
            return_attention=getattr(args, 'return_gnn_attention', False),
            topk=self.topk,
            ppi_edge_index=edge_index,
        )

        # Post-GNN MLP: compress the flattened attention-pooled representation
        # Input: [B, gene_hidden_dim * cell_latent_dim]
        # Output: [B, embd_col_dim]
        self.cell_mlp = nn.Sequential(
            nn.Linear(self.gene_hidden_dim * self.cell_latent_dim, self.embd_col_dim),
            nn.LeakyReLU(inplace=True)
        )

        # VAE parameter heads: predict mu and logvar for every cell type simultaneously.
        # Output: [B, n_cell_types * cell_latent_dim * 2]
        self.fc_mu_logvar = nn.Linear(self.embd_col_dim, self.n_cell_types * self.cell_latent_dim * 2)

        if self.predict_cell_prop:
            if self.cell_prop_activation_function == "sigmoid":
                self.cancer_cell_type_index = resolve_cancer_cell_type_index_from_config(self.args)
            self.gnn_dd_alpha = nn.Linear(
                self.embd_col_dim,
                get_cell_prop_head_output_dim(
                    n_cell_types=self.n_cell_types,
                    activation_function=self.cell_prop_activation_function,
                ),
            )

    def forward(
        self, x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        sample_ids: Optional[List[str]] = None
    ) -> ModelOutput:
        """
        Args:
            x:          Gene expression matrix [B, All_Genes].
            y:          Cell proportion labels [B, n_cell_types];
                        required when predict_cell_prop is False.
            sample_ids: Optional sample identifiers for tracking.

        Returns:
            ModelOutput with fields:
              cell_prop          [B, n_cell_types]
              mu_all_types       [B, latent_dim, n_cell_types]
              logvar_all_types   [B, latent_dim, n_cell_types]
              mu_mean            [B, latent_dim]
              logvar_mean        [B, latent_dim]
              cell_type_existed  [B, n_cell_types]  binary presence mask
        """

        current_device = self.device_param.device
        x = x.to(current_device)
        batch_size = x.size(0)

        # Split oversized inference batches to avoid GPU OOM.
        if batch_size > self.max_batch_size_for_gnn and not self.training:
            return self._forward_split_batch(x, y, sample_ids)

        # --- Step 1. Prepare Node Features ---
        # x: [B, All_Genes] -> x_sub: [B, N, 1]
        x_sub = torch.index_select(x, 1, self.filter_indices_tensor).unsqueeze(-1)

        # Broadcast prior gene features across the batch: [N, F_gene] -> [B, N, F_gene]
        gf = self.gene_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate expression + prior features: [B, N, 1 + F_gene]
        node_features = torch.cat([x_sub, gf], dim=-1)

        # --- Step 2. GNN Forward ---
        # PPIEncoder handles flattening, message passing, and attention pooling.
        # Output: [B, latent_dim * gene_hidden_dim]
        # Unpack tuple when return_attention is enabled
        encoder_out = self.encoder(node_features)
        if isinstance(encoder_out, tuple):
            gene_embeddings, attn_weights = encoder_out
        else:
            gene_embeddings = encoder_out

        # --- Step 3: MLP Compression ---
        # [B, latent_dim * gene_hidden_dim] -> MLP -> [B, embd_col_dim]
        cell_embedding = self.cell_mlp(gene_embeddings)

        # --- Step 4: Predict mu and logvar ---
        # [B, embd_col_dim] -> [B, n_cell_types * latent_dim * 2]
        mu_logvar_flat = self.fc_mu_logvar(cell_embedding)

        # Reshape: [B, n_cell_types, latent_dim * 2]
        mu_logvar_structured = mu_logvar_flat.reshape(-1, self.n_cell_types, self.cell_latent_dim * 2)

        # Split along last dim: each → [B, n_cell_types, latent_dim]
        mu_raw, logvar_raw = torch.chunk(mu_logvar_structured, 2, dim=-1)

        # Permute to [B, latent_dim, n_cell_types] to match downstream convention
        mu_all_types = mu_raw.permute(0, 2, 1).contiguous()
        logvar_all_types = logvar_raw.permute(0, 2, 1).contiguous()

        # Clamp logvar to prevent KL divergence from exploding
        logvar_all_types = torch.clamp(logvar_all_types, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

        # Aggregate across cell types → overall latent representation [B, latent_dim]
        mu_mean = torch.mean(mu_all_types, dim=2)
        logvar_mean = torch.mean(logvar_all_types, dim=2)

        output = ModelOutput()

        # --- Step 5: Cell Proportions ---
        if self.predict_cell_prop:
            cell_prop, dd_alpha = build_cell_prop_from_head_output(
                head_output=self.gnn_dd_alpha(cell_embedding),
                activation_function=self.cell_prop_activation_function,
                n_cell_types=self.n_cell_types,
                eps=EPS,
                cancer_cell_type_index=self.cancer_cell_type_index,
            )
            output['dd_alpha'] = dd_alpha
        elif y is not None:
            cell_prop = y.to(current_device)
            output['dd_alpha'] = None
        else:
            # Fallback or Error
            cell_prop = torch.zeros(batch_size, self.n_cell_types, device=current_device)
            output['dd_alpha'] = None
            if self.training:
                raise ValueError("y must be provided if predict_cell_prop is False")

        output['cell_prop'] = cell_prop

        # Step 6: Positional Encoding (optional)
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

        merged = ModelOutput()
        # Assuming ModelOutput is a dict-like or has keys
        keys = outputs[0].keys() if isinstance(outputs[0], dict) else outputs[0].__dict__.keys()

        for key in keys:
            vals = [out[key] for out in outputs if out[key] is not None]
            merged[key] = torch.cat(vals, dim=0) if vals else None
        return merged

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run GNN + cell_mlp only, without the final VAE parameter heads.
        Useful for feature extraction in fused or downstream models.
        Returns: [B, embd_col_dim]
        """
        current_device = self.device_param.device
        x = x.to(current_device)
        batch_size = x.size(0)

        # Handle batch splitting for memory safety (same logic as before)
        if batch_size > self.max_batch_size_for_gnn:
            features_list = []
            for i in range(0, batch_size, self.max_batch_size_for_gnn):
                end_idx = min(i + self.max_batch_size_for_gnn, batch_size)
                features_list.append(self._extract_features_single(x[i:end_idx]))
            return torch.cat(features_list, dim=0)
        else:
            return self._extract_features_single(x)

    def _extract_features_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal helper: GNN + MLP forward for a single chunk.
        """
        batch_size = x.size(0)

        # 1. Prepare Node Features
        x_sub = torch.index_select(x, 1, self.filter_indices_tensor).unsqueeze(-1)  # [B, N, 1]

        # Expand gene features: [B, N, F_gene]
        gf = self.gene_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate: [B, N, 1 + F_gene]
        node_features = torch.cat([x_sub, gf], dim=-1)

        # 2. GNN Forward (Returns [B, latent_dim * Hidden]), flattened gene embeddings for each sample
        encoder_out = self.encoder(node_features)
        flattened_gene_embeddings = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out

        # 3. MLP (Get the cell embedding: [B, embd_col_dim])
        cell_embedding = self.cell_mlp(flattened_gene_embeddings)

        return cell_embedding

    def get_config(self):
        return {
            "params": {"args": self.args.to_dict()},
            "module_name": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }


class PPIEncoder(nn.Module):
    """
    GNN Encoder operating on a PPI graph.

    Responsibilities:
      1. Build a sparse adjacency matrix from the PPI edge index (done once in __init__).
      2. Multi-layer GraphSAGE message passing with residual connections.
      3. Jumping Knowledge (JK) fusion: aggregate all layer outputs for
         richer, multi-scale gene representations.
      4. Learned-query cross-attention pooling with top-k sparsity: compress
         N gene node representations into a fixed-length vector.

    Input:  x [B, N, F]
    Output: [B, latent_dim * gene_hidden_dim]
            (optionally also attn_weights [B, latent_dim, N])

    Notation used throughout:
        B          : batch size
        N          : number of gene nodes
        F          : input feature dimension per gene
        H          : gene_hidden_dim (hidden feature dimension after each GNN layer)
        L          : num_layers
        E          : number of edges in the PPI graph
        latent_dim : number of learnable query slots for attention pooling
        k          : topk (number of genes selected per query in sparse attention)
    """

    def __init__(self,
                 in_feats: int,
                 gene_hidden_dim: int,
                 latent_dim: int,
                 num_nodes: int,
                 ppi_edge_index: torch.Tensor,
                 num_layers: int = 3,
                 drop_p: float = 0.1,
                 use_gradient_checkpointing: bool = False,
                 return_attention: bool = False,
                 topk: int = 1024):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.gene_hidden_dim = gene_hidden_dim  # H: hidden dim per gene node
        self.num_nodes       = num_nodes        # N: total number of gene nodes
        self.latent_dim      = latent_dim       # number of attention query slots
        self.num_layers      = num_layers       # L: number of GraphSAGE layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.return_attention = return_attention

        # Top-k genes to attend to per query slot; capped at N
        self.topk = min(topk, self.num_nodes)

        # ── Step 1: Pre-Build Sparse Adjacency Matrix ─────────────────────────
        # Convert the edge index [2, E] into a sparse COO adjacency matrix A
        # of shape [N, N], using pure PyTorch (no torch_sparse dependency).
        #
        # Note on row/col convention:
        #   ppi_edge_index[0] = source nodes  (message senders)
        #   ppi_edge_index[1] = target nodes  (message receivers)
        # We construct A such that A[target, source] = 1, so that the
        # matrix-vector product A @ h aggregates source features into targets.
        #
        #   row indices : ppi_edge_index[1]  (targets)  shape [E]
        #   col indices : ppi_edge_index[0]  (sources)  shape [E]
        #   values      : all-ones           shape [E]
        #   A           : sparse [N, N],  A[i,j] = 1 if gene j -> gene i
        row = ppi_edge_index[1]  # target node indices, shape [E]
        col = ppi_edge_index[0]  # source node indices, shape [E]
        val = torch.ones(row.size(0), dtype=torch.float32, device=row.device)  # shape [E]
        adj = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),  # shape [2, E]
            values=val,                       # shape [E]
            size=(num_nodes, num_nodes),      # [N, N]
        ).coalesce()  # coalesce() merges any duplicate (row, col) entries

        # Degree vector: deg[i] = number of in-neighbors of node i, shape [N]
        # Clamped to >= 1.0 to avoid division by zero for isolated nodes.
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)  # [N]

        # register_buffer does NOT support sparse tensors directly.
        # Instead, store the sparse components (indices + values) as dense
        # buffers and reconstruct the sparse tensor in forward().
        # Buffers are:
        #   - automatically moved with .to(device) / .cuda()
        #   - saved and restored in state_dict checkpoints
        #   - NOT treated as trainable parameters
        self.register_buffer("_adj_indices", adj.indices())  # [2, E]
        self.register_buffer("_adj_values",  adj.values())   # [E]
        self.register_buffer("_deg",         deg)            # [N]

        # ── Step 2: GraphSAGE Layer Parameters ────────────────────────────────
        # Each layer i has four components:
        #   self_linear  : W_self^(i)  [current_dim -> H]  transforms the node\'s own features
        #   neigh_linear : W_neigh^(i) [current_dim -> H]  transforms aggregated neighbor features
        #   skip_linear  : W_skip^(i)  [current_dim -> H]  residual projection (Identity if dims match)
        #   norm         : LayerNorm(H) applied after the SAGE update, before activation
        self.norms         = nn.ModuleList()
        self.self_linears  = nn.ModuleList()
        self.neigh_linears = nn.ModuleList()
        self.skip_linears  = nn.ModuleList()

        current_dim = in_feats  # tracks the feature dimension across layers (F -> H -> H -> ...)
        for i in range(num_layers):
            self.norms.append(nn.LayerNorm(gene_hidden_dim))
            self.self_linears.append(nn.Linear(current_dim, gene_hidden_dim))
            self.neigh_linears.append(nn.Linear(current_dim, gene_hidden_dim))
            self.skip_linears.append(
                # Use a learned projection when dimensions differ (first layer: F != H),
                # otherwise use Identity to avoid unnecessary parameters.
                nn.Linear(current_dim, gene_hidden_dim) if current_dim != gene_hidden_dim else nn.Identity()
            )
            current_dim = gene_hidden_dim  # all subsequent layers operate at dim H

        self.dropout    = nn.Dropout(drop_p)
        self.activation = nn.GELU()

        # ── Step 3: Jumping Knowledge (JK) Fusion ─────────────────────────────
        # Problem: using only the last GNN layer discards information from
        # earlier layers. Shallow layers capture local PPI neighborhoods;
        # deeper layers capture global pathway-level context. JK fusion
        # combines all layers so the final representation is multi-scale.
        #
        # Implementation: concatenate all L layer outputs along the feature dim,
        # then project back to H with a learned linear layer.
        #
        #   Input  shape: [B, N, L * H]
        #   Output shape: [B, N, H]
        self.jk_fusion = nn.Linear(num_layers * gene_hidden_dim, gene_hidden_dim)

        # ── Step 4: Learned-Query Cross-Attention Pooling ─────────────────────
        # Goal: compress N gene representations [B, N, H] into a fixed-length
        # vector [B, latent_dim * H] using learned cross-attention.
        #
        # latent_dim learnable Query vectors each attend over all N gene nodes,
        # producing latent_dim pooled slot representations. This is more
        # expressive than simple mean/max pooling because each query can
        # specialize to a different functional gene group (e.g., immune, metabolic).

        # Separate Key and Value projectors:
        #   key_projector   : determines which genes each query attends to ("where to look")
        #   value_projector : determines what information is aggregated   ("what to read")
        # Sharing K = V is a valid simplification but limits expressiveness by
        # coupling similarity matching with information aggregation.
        self.key_projector   = nn.Linear(gene_hidden_dim, gene_hidden_dim)
        self.value_projector = nn.Linear(gene_hidden_dim, gene_hidden_dim)

        # Learnable Query matrix: shape [1, latent_dim, H]
        # Xavier uniform init keeps initial attention scores in a stable range.
        self.attention_queries = nn.Parameter(torch.empty(1, latent_dim, gene_hidden_dim))
        nn.init.xavier_uniform_(self.attention_queries)

        # Scale factor 1/sqrt(H): prevents dot-product scores from growing too
        # large, which would push softmax into saturation (near-zero gradients).
        self.attention_scale_factor = 1.0 / (gene_hidden_dim ** 0.5)

        # Output normalization applied after flattening the pooled slots
        self.output_norm      = nn.LayerNorm(gene_hidden_dim * latent_dim)
        self.final_activation = nn.GELU()



    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Node features, shape [B, N, F].
                B = batch size, N = number of genes, F = features per gene.

        Returns:
            output_processed : [B, latent_dim * H]
            attn_weights     : [B, latent_dim, N]  (only when return_attention=True)
        """
        batch_size, num_genes, num_feats = x.shape  # B, N, F

        # Validate that the runtime graph size matches the encoder\'s configuration.
        if num_genes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} genes (nodes), but got {num_genes}. "
                f"Ensure input gene filtering is consistent with PPIEncoder initialization."
            )

        # Reconstruct the sparse adjacency matrix from buffered indices and values.
        # This is a lightweight operation (no new memory allocation for the data).
        # The buffers are already on the correct device thanks to register_buffer.
        if not hasattr(self, "_adj_cache") or self._adj_cache_device != x.device:
            self._adj_cache = torch.sparse_coo_tensor(
                self._adj_indices,
                self._adj_values,
                size=(self.num_nodes, self.num_nodes),
            ).coalesce()
            self._adj_cache_device = x.device
        adj = self._adj_cache  # [N, N] sparse COO

        # ── Step 2: Multi-layer GraphSAGE Message Passing ─────────────────────
        # For each layer i, the update rule is:
        ##
        #   h^(i) = LayerNorm( W_self^(i) h^(i-1) + W_neigh^(i) Ã h^(i-1) ) + W_skip^(i) h^(i-1)
        #
        #   where  Ã = D⁻¹A  (degree-normalized adjacency matrix)
        #          Ã h^(i-1) = A @ h^(i-1) / deg   (mean neighbor aggregation)
        #
        ## In each step code:
        #   h_neigh^(i) = (1 / deg) * A @ h^(i-1)                 mean neighbor aggregation
        #   h_raw^(i)   = W_self^(i) h^(i-1) + W_neigh^(i) h_neigh^(i)
        #   h_norm^(i)  = LayerNorm( h_raw^(i) )
        #   h_act^(i)   = Dropout( GELU( h_norm^(i) ) )
        #   h^(i)       = h_act^(i) + W_skip^(i) h^(i-1)          residual connection
        #
        # Shape progression per layer:
        #   embedded        : [B, N, current_dim]  (current_dim = F for i=0, H for i>0)
        #   embedded_flat   : [N, B * current_dim] (reshape for batched sparse matmul)
        #   neigh_flat      : [N, B * current_dim] (result of A @ embedded_flat)
        #   neigh           : [B, N, current_dim]  (reshape back, then degree-normalize)
        #   out             : [B, N, H]             (after self + neigh linear transforms)
        #   identity        : [B, N, H]             (skip connection)
        #   embedded (new)  : [B, N, H]             (out + identity)
        #
        # Key trick: instead of looping over the batch dimension, we reshape
        # [B, N, C] -> [N, B*C] so that a single sparse matmul A @ [N, B*C]
        # simultaneously aggregates neighbors for all B samples.

        embedded      = x            # [B, N, F]  initial node features
        layer_outputs = []           # will collect h^(1), h^(2), ..., h^(L) for JK fusion

        for i in range(self.num_layers):
            cin = embedded.shape[-1]  # current feature dim: F (i=0) or H (i>0)

            # ── 2a. Batched Neighbor Aggregation via Sparse MatMul ─────────────
            # Reshape [B, N, C] -> [N, B*C] to enable a single sparse matmul.
            #   embedded_flat : [N, B * cin]
            #   neigh_flat    : [N, B * cin]   A @ embedded_flat aggregates neighbor features
            #   neigh         : [B, N, cin]    reshape back to batch format
            #   neigh (norm)  : [B, N, cin]    divide by degree for mean aggregation
            embedded_flat = embedded.permute(1, 0, 2).reshape(num_genes, batch_size * cin)  # [N, B*C]
            neigh_flat    = torch.sparse.mm(adj, embedded_flat)                              # [N, B*C]
            neigh         = neigh_flat.reshape(num_genes, batch_size, cin).permute(1, 0, 2) # [B, N, C]
            neigh         = neigh / self._deg.view(1, num_genes, 1)                          # [B, N, C]

            # ── 2b. GraphSAGE Linear Transform + Norm + Activation ────────────
            #   out : [B, N, H]
            out = self.self_linears[i](embedded) + self.neigh_linears[i](neigh)  # [B, N, H]
            out = self.norms[i](out)       # LayerNorm over H
            out = self.activation(out)     # GELU
            out = self.dropout(out)        # Dropout

            # ── 2c. Residual (Skip) Connection ────────────────────────────────
            #   identity : [B, N, H]  (Linear projection or Identity)
            #   embedded : [B, N, H]  (updated node features for next layer)
            identity = self.skip_linears[i](embedded)  # [B, N, H]
            embedded = out + identity                  # [B, N, H]

            layer_outputs.append(embedded)  # save h^(i) for JK fusion

        # ── Step 3: Jumping Knowledge (JK) Fusion ─────────────────────────────
        # Concatenate all L layer outputs along the feature dimension, then
        # project back to H with a learned linear layer.
        #
        #   jk_input : [B, N, L * H]   (concat of h^(1), ..., h^(L))
        #   embedded : [B, N, H]       (after jk_fusion linear + GELU)
        jk_input = torch.cat(layer_outputs, dim=-1)  # [B, N, L * H]
        embedded  = self.jk_fusion(jk_input)          # [B, N, H]
        embedded  = self.activation(embedded)          # GELU

        # ── Step 4: Learned-Query Cross-Attention Pooling ─────────────────────
        # Compress N gene representations [B, N, H] into latent_dim pooled
        # slot vectors [B, latent_dim, H], then flatten to [B, latent_dim * H].

        # ── 4a. Key and Value Projections ─────────────────────────────────────
        #   K = GELU( W_key   @ embedded )   [B, N, H]  "where to look"
        #   V = GELU( W_value @ embedded )   [B, N, H]  "what to read"
        keys   = F.gelu(self.key_projector(embedded))    # [B, N, H]
        values = F.gelu(self.value_projector(embedded))  # [B, N, H]

        # ── 4b. Attention Score Computation ───────────────────────────────────
        # Scaled dot-product attention between learnable queries Q and keys K:
        #
        #   scores = Q @ K^T / sqrt(H)
        #
        #   Q      : attention_queries expanded  [B, latent_dim, H]
        #   K^T    : keys transposed             [B, H, N]
        #   scores : [B, latent_dim, N]
        #            scores[b, q, n] = similarity of query q to gene n in sample b
        attn_scores = torch.bmm(
            self.attention_queries.expand(batch_size, -1, -1),  # [B, latent_dim, H]
            keys.transpose(1, 2)                                 # [B, H, N]
        ) * self.attention_scale_factor                          # [B, latent_dim, N]

        # ── 4c. Top-k Sparse Attention ────────────────────────────────────────
        # Retain only the top-k highest-scoring genes per query slot.
        # This reduces noise from irrelevant genes and lowers memory usage.
        #
        #   topk_scores   : [B, latent_dim, k]  raw scores of selected genes
        #   topk_indices  : [B, latent_dim, k]  gene indices of selected genes
        #   topk_weights  : [B, latent_dim, k]  softmax-normalized attention weights
        topk_scores, topk_indices = torch.topk(attn_scores, k=self.topk, dim=-1)  # [B, latent_dim, k]
        topk_weights = F.softmax(topk_scores, dim=-1)                              # [B, latent_dim, k]

        # ── 4d. Reconstruct Full Attention Map (optional) ─────────────────────
        # For interpretability: scatter top-k weights back into a dense [B, latent_dim, N]
        # tensor so downstream code can identify which genes each query focuses on.
        if self.return_attention:
            attn_weights = torch.zeros_like(attn_scores)           # [B, latent_dim, N]
            attn_weights.scatter_(-1, topk_indices, topk_weights)  # fill top-k positions

        # ── 4e. Gather Top-k Value Vectors ────────────────────────────────────
        # Retrieve the value vectors V[n] for the top-k selected gene indices.
        #
        #   values              : [B, N, H]
        #   values_expanded     : [B, latent_dim, N, H]  broadcast over query slots
        #                         .expand() creates a memory-efficient view (no copy);
        #                         safe here since torch.gather only reads from it.
        #   topk_indices_exp    : [B, latent_dim, k, H]  index tensor expanded over H
        #   gathered_values     : [B, latent_dim, k, H]  value vectors for top-k genes
        values_expanded       = values.unsqueeze(1).expand(-1, self.latent_dim, -1, -1)          # [B, latent_dim, N, H]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.gene_hidden_dim)  # [B, latent_dim, k, H]
        gathered_values       = torch.gather(values_expanded, dim=2, index=topk_indices_expanded)    # [B, latent_dim, k, H]

        # ── 4f. Weighted Aggregation over Top-k Genes ─────────────────────────
        # Compute a weighted sum of the top-k value vectors for each query slot:
        #
        #   pooled[b, q, :] = sum_{j=1}^{k} topk_weights[b, q, j] * gathered_values[b, q, j, :]
        #
        #   topk_weights unsqueezed : [B, latent_dim, k, 1]   (broadcast over H)
        #   gathered_values         : [B, latent_dim, k, H]
        #   pooled                  : [B, latent_dim, H]
        pooled = (topk_weights.unsqueeze(-1) * gathered_values).sum(dim=2)  # [B, latent_dim, H]

        # ── 4g. Flatten Pooled Slots ───────────────────────────────────────────
        # Concatenate all latent_dim slot vectors into a single representation.
        #   output_flat : [B, latent_dim * H]
        output_flat = pooled.reshape(batch_size, -1)  # [B, latent_dim * H]

        # ── Step 5: Output Normalization ──────────────────────────────────────
        # Apply LayerNorm over the full flattened vector, then GELU activation.
        #   output_processed : [B, latent_dim * H]
        output_processed = self.output_norm(output_flat)         # LayerNorm
        output_processed = self.final_activation(output_processed)  # GELU

        # Optionally return the full attention weight map for downstream
        # interpretability (e.g., identifying which genes each latent query
        # focuses on across cell types or conditions).
        if self.return_attention:
            return output_processed, attn_weights  # [B, latent_dim * H], [B, latent_dim, N]

        return output_processed  # [B, latent_dim * H]
