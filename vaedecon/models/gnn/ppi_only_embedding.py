import os
import logging
from typing import Optional, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint

from ...configs import ModelConfig, DataConfig
from ...models.base import (ModelOutput, reparameterize_dirichlet, LOGVAR_CLAMP_MIN,
                            LOGVAR_CLAMP_MAX, EPS, NETWORK_CUTOFF, BaseEncoder)
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

        self.position_encoding = position_encoding if self.args.using_positional_encoding else None

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
            self.gnn_dd_alpha = nn.Linear(self.embd_col_dim, self.n_cell_types)

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
        encoder_out = self.encoder(node_features, self.edge_index)
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
            # softplus ensures alpha > 0; EPS prevents numerical instability
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
        encoder_out = self.encoder(node_features, self.edge_index)
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
      1. Multi-layer GraphSAGE message passing with residual connections.
      2. Jumping Knowledge (JK) fusion: aggregate all layer outputs for
         richer, multi-scale gene representations.
      3. Learned-query cross-attention pooling: compress N gene node
         representations into a fixed-length vector.

    Input:  x [B, N, F],  ppi_edge_index [2, E]
    Output: [B, latent_dim * gene_hidden_dim]
            (optionally also attn_weights [B, latent_dim, N])
    """
    def __init__(self,
                 in_feats: int,
                 gene_hidden_dim: int,
                 latent_dim: int,
                 num_nodes: int,
                 num_layers=3,
                 drop_p=0.1,
                 use_gradient_checkpointing=False,
                 return_attention: bool=False):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.gene_hidden_dim = gene_hidden_dim  # Dimension of each gene's embedding after GNN layers
        self.num_nodes = num_nodes  # Number of nodes in the graph, i.e., number of genes considered
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.return_attention = return_attention

        # GNN Layers
        # Each layer: SAGEConv → LayerNorm → GELU → Dropout → residual add
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # Skip connections: Linear projection when dims differ, Identity otherwise
        self.skips = nn.ModuleList()

        current_dim = in_feats
        for i in range(num_layers):
            # SAGEConv layer: input_dim -> gene_hidden_dim, h_v = W1 * h_v + W2 * mean(h_neighbors)
            self.convs.append(SAGEConv(current_dim, gene_hidden_dim))
            self.norms.append(nn.LayerNorm(gene_hidden_dim))
            self.skips.append(
                nn.Linear(current_dim, gene_hidden_dim)
                if current_dim != gene_hidden_dim else nn.Identity()
            )
            current_dim = gene_hidden_dim

        self.dropout = nn.Dropout(drop_p)
        self.activation = nn.GELU()

        # ── Jumping Knowledge (JK) Fusion ──────────────────────────
        # Problem: using only the last GNN layer discards information from
        # earlier layers. Shallow layers capture local PPI neighborhoods;
        # deeper layers capture global pathway-level context. JK fusion
        # combines all layers so the final representation is multi-scale.
        #
        # Implementation: concatenate all layer outputs along the feature dim,
        # then project back to gene_hidden_dim with a learned linear layer.
        #
        # Input to jk_fusion: [B*N, num_layers * gene_hidden_dim]
        # Output:              [B*N, gene_hidden_dim]
        self.jk_fusion = nn.Linear(num_layers * gene_hidden_dim, gene_hidden_dim)

        # ── Attention Pooling ───────────────────────────────────────
        # Goal: compress N gene node representations [B, N, H] into a fixed
        # vector [B, latent_dim * H] using learned cross-attention.
        #
        # latent_dim learnable Query vectors each attend over all N nodes,
        # producing latent_dim pooled slot representations. This is more
        # expressive than simple mean/max pooling because each query can
        # specialize to a different functional gene group.

        # Separate Key and Value projectors for richer attention
        # Previously K and V shared the same projection (feature_projector),
        # which limits expressiveness. Separate projectors allow the model to
        # learn different transformations for computing similarity (K) vs.
        # aggregating information (V).
        self.key_projector = nn.Linear(gene_hidden_dim, gene_hidden_dim)
        self.value_projector = nn.Linear(gene_hidden_dim, gene_hidden_dim)

        # Learnable Query matrix: [1, latent_dim, H]
        # Xavier init keeps attention scores in a stable range at the start.
        self.attention_queries = nn.Parameter(torch.empty(1, latent_dim, gene_hidden_dim))
        nn.init.xavier_uniform_(self.attention_queries)

        # Scale factor 1/sqrt(H) prevents dot-product scores from growing too
        # large, which would push softmax into saturation (near-zero gradients).
        self.attention_scale_factor = 1.0 / (gene_hidden_dim ** 0.5)

        # Output normalization after flattening the pooled slots
        self.output_norm = nn.LayerNorm(gene_hidden_dim * latent_dim)
        self.final_activation = nn.GELU()

        # ── Edge Index Cache ────────────────────────────────────
        # Rebuilding the block-diagonal edge index every forward pass is wasteful.
        # Cache it and reuse as long as batch_size, device, and graph structure
        # remain unchanged.
        self._cached_edge_index_key = None
        self._cached_edge_index = None

    # Extract gradient checkpointing logic into a dedicated method.
    # Previously the closure `def conv_forward(inp, ei=...)` was defined inside
    # a for-loop, which can cause subtle variable capture bugs in Python.
    # A dedicated method avoids this and keeps the forward loop clean.
    def _run_conv(self, conv: SAGEConv, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run a single SAGEConv layer, with optional gradient checkpointing."""
        if self.use_gradient_checkpointing and self.training:
            # Gradient checkpointing trades compute for memory:
            # intermediate activations are not stored during the forward pass
            # and are recomputed during backprop. Useful for large graphs.
            def conv_fn(inp):
                return conv(inp, edge_index)
            return checkpoint(conv_fn, x, use_reentrant=False)
        return conv(x, edge_index)

    def _get_batched_edge_index(self, edge_index: torch.Tensor,
                                batch_size: int,
                                device: torch.device) -> torch.Tensor:
        """
        Replicate the single-graph edge_index B times with node index offsets,
        forming a block-diagonal graph where each sample's subgraph is isolated.

        Single graph edge (u, v) for sample i becomes (u + i*N, v + i*N).

        Cache key includes batch_size, device, memory address, and edge count
        to safely detect any change in the graph structure or execution context.
        """
        cache_key = (batch_size, str(device), edge_index.data_ptr(), edge_index.shape[1])
        if self._cached_edge_index_key == cache_key and self._cached_edge_index is not None:
            return self._cached_edge_index

        # edge_index: [2, E]
        # offsets: [B, 1], values [0, N, 2N, ..., (B-1)*N]
        offsets = torch.arange(batch_size, device=device).view(-1, 1) * self.num_nodes

        # Broadcast: edge_index[0] [E] → [1, E] + [B, 1] → [B, E]
        src = edge_index[0].unsqueeze(0) + offsets
        dst = edge_index[1].unsqueeze(0) + offsets

        # ：[B*E] → stack → [2, B*E]
        # Flatten src and dst to 1D tensors and combine them into edge_index format
        batched_edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)

        self._cached_edge_index_key = cache_key
        self._cached_edge_index = batched_edge_index
        return batched_edge_index

    def forward(self, x: torch.Tensor, ppi_edge_index: torch.Tensor):
        """
        Args:
            x:              Node features [B, N, F].
                            B = batch size, N = number of genes, F = features per gene.
            ppi_edge_index: Single-graph edge index [2, E].
                            Row 0 = source nodes, Row 1 = target nodes.

        Returns:
            output_processed: [B, latent_dim * gene_hidden_dim]
            attn_weights (optional): [B, latent_dim, N]  returned when return_attention=True
        """
        batch_size, num_genes, num_feats = x.shape

        # Validate that the input graph size matches the encoder's configuration.
        if num_genes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} genes (nodes), but got {num_genes}. "
                f"Ensure input gene filtering is consistent with PPIEncoder initialization."
            )

        # ── Step 1: Flatten to PyG format ────────────────────────────────────
        # PyG message passing requires shape [total_nodes, F].
        # Concatenate B graphs into one large graph: [B*N, F]
        x_flat = x.reshape(batch_size * num_genes, num_feats)

        # ── Step 2: Build Block-Diagonal Edge Index ───────────────────────────
        # Replicate single-graph edges for all B samples: [2, E] → [2, B*E]
        batched_edge_index = self._get_batched_edge_index(ppi_edge_index, batch_size, x.device)

        # ── Step 3: Multi-Layer GNN Message Passing ───────────────────────────
        # Collect outputs from every layer for JK fusion.
        # Previously only the last layer's output was used, discarding the
        # multi-scale structural information captured by earlier layers.
        embedded = x_flat  # Initial node features, [B*N, F]
        layer_outputs = []  # Will hold [B*N, H] tensors for each layer
        for i, conv in enumerate(self.convs):
            # Compute skip connection on the input before transformation
            identity = self.skips[i](embedded)  # Apply skip first, [B*N, H]

            # GraphSAGE message passing:
            # h_v^(l+1) = W1·h_v^(l) + W2·mean_{u∈N(v)} h_u^(l)
            out = self._run_conv(conv, embedded, batched_edge_index)  # [B*N, H]

            out = self.norms[i](out)      # LayerNorm: stabilize feature distributions
            out = self.activation(out)    # GELU non-linearity: smooth and effective for GNNs
            out = self.dropout(out)      # Dropout: regularization to prevent overfitting

            # Residual Connection
            embedded = out + identity  # [B*N, H]

            # Save this layer's output for JK fusion
            layer_outputs.append(embedded)

        # ── Step 4: Jumping Knowledge (JK) Fusion ────────────────────────────
        # Concatenate all layer outputs and project to gene_hidden_dim.
        # This gives each node a representation that integrates information from
        # 1-hop up to num_layers-hop neighborhoods simultaneously.
        #
        # layer_outputs: list of num_layers tensors, each [B*N, H]
        # After cat: [B*N, num_layers * H]
        # After jk_fusion: [B*N, H]
        jk_input = torch.cat(layer_outputs, dim=-1)   # [B*N, num_layers * H]
        embedded = self.jk_fusion(jk_input)            # [B*N, H]
        embedded = self.activation(embedded)           # Non-linearity after fusion

        # ── Step 5: Restore Batch Dimension ───────────────────────
        # [B*N, H] → [B, N, H]
        embedded = embedded.reshape(batch_size, num_genes, self.gene_hidden_dim)

        # ── Step 6: Learned-Query Cross-Attention Pooling ─────────────────────
        # Compress N gene representations into latent_dim pooled slot vectors.
        #
        # Use separate Key and Value projectors.
        # K determines which genes each query attends to (similarity).
        # V determines what information is aggregated (content).
        # Sharing K=V (original code) is a valid simplification but limits
        # the model's ability to decouple "where to look" from "what to read".

        keys = F.gelu(self.key_projector(embedded))      # [B, N, H]
        values = F.gelu(self.value_projector(embedded))  # [B, N, H]

        # Calculate Attention Scores: Q * K^T / sqrt(H)
        # Q: attention_queries, [1, latent_dim, H] -> expand -> [B, latent_dim, H]
        # K^T: projected [B, N, H] -> transpose -> [B, H, N]
        # scores: [B, latent_dim, N], i.e., bmm: [B, latent_dim, H] x [B, H, N] -> [B, latent_dim, N]
        attn_scores = torch.bmm(  # Batched Matrix Multiplication
            self.attention_queries.expand(batch_size, -1, -1),  # [B, latent_dim, H]
            keys.transpose(1, 2)  # [B, H, N]
        ) * self.attention_scale_factor

        # Softmax over the N gene dimension → attention weights sum to 1 per query
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Latent, N]

        # Weighted aggregation of Value vectors
        # [B, latent_dim, N] × [B, N, H] → [B, latent_dim, H]
        pooled = torch.bmm(attn_weights, values)

        # Flatten pooled slots: [B, latent_dim, H] -> [B, latent_dim * H]
        output_flat = pooled.reshape(batch_size, -1)

        # ── Step 7: Output Normalization ───────────────────────────
        output_processed = self.output_norm(output_flat)
        output_processed = self.final_activation(output_processed)

        # Optionally return attention weights for downstream
        # interpretability analysis (e.g., identifying which genes each
        # latent query focuses on across cell types).
        if self.return_attention:
            return output_processed, attn_weights

        return output_processed
