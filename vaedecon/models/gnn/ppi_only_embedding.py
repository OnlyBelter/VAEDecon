import os
import logging
from typing import Optional, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Sequential, SAGEConv
from torch.utils.checkpoint import checkpoint
import lightning as L

from ..vae.vae_config import VAEConfig
from ...models.base import (ModelOutput, reparameterize_dirichlet, LOGVAR_CLAMP_MIN,
                            LOGVAR_CLAMP_MAX, EPS, NETWORK_CUTOFF, BaseEncoder)
from vaedecon.models.gnn.positional_encoding import PositionalEncoding

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class EncoderSGNN(BaseEncoder):
    """
    GNN-based encoder using a PPI network + optional gene-level features to produce
    per-cell latent embeddings (mu, logvar) and optional cell-type proportions.
    - Treat each sample independently.

    Memory optimizations:
    - Uses gradient checkpointing during training
    - Implements batch splitting for large inputs
    """

    def __init__(self, args: VAEConfig, position_encoding: Optional[PositionalEncoding] = None):
        """
        :param args: parameters for the GNN encoder
        """
        super().__init__()
        self.args = args
        self.device_param = nn.Parameter(torch.empty(0))

        # ===== Memory optimization flags =====
        self.use_gradient_checkpointing = getattr(args, 'use_gradient_checkpointing', False)
        self.max_batch_size_for_gnn = getattr(args, 'max_batch_size_for_gnn', 32)  # Split if larger

        # --- load and prepare gene list & PPI network ---
        if args.input_gene_list_fp is None or not os.path.exists(args.input_gene_list_fp):
            raise FileNotFoundError(f"Gene list file not found: {args.input_gene_list_fp}")
        self.gene_list: List[str] = pd.read_csv(args.input_gene_list_fp, header=None).iloc[:, 0].tolist()

        if not os.path.exists(args.ppi_file_path):
            raise FileNotFoundError(f"PPI file not found: {args.ppi_file_path}")
        net_df = pd.read_csv(args.ppi_file_path)

        # Select relevant columns
        if args.biogrid_flag:
            if not all(col in net_df.columns for col in ["g1_symbol", "g2_symbol"]):
                raise ValueError("For biogrid_flag=True, PPI file must contain 'g1_symbol' and 'g2_symbol' columns.")
            net = net_df[["g1_symbol", "g2_symbol"]].copy()
            net.columns = ["source", "target"]
        else:
            if not all(col in net_df.columns for col in ["g1_symbol", "g2_symbol", "conn"]):
                raise ValueError(
                    "For biogrid_flag=False, PPI file must contain 'g1_symbol', 'g2_symbol', and 'conn' columns.")
            net = net_df[["g1_symbol", "g2_symbol", "conn"]].copy()
            net = net[net.conn >= NETWORK_CUTOFF]
            net.columns = ["source", "target", "conn"]

        net.drop_duplicates(subset=["source", "target"], inplace=True)
        mask = net.source.isin(self.gene_list) & net.target.isin(self.gene_list)
        net = net[mask]
        net = net[net.source != net.target]

        # Create gene mapping
        unique_genes_in_ppi = pd.unique(net[["source", "target"]].values.ravel("K"))
        self.graph_gene_list = [gene for gene in self.gene_list if gene in unique_genes_in_ppi]

        if not self.graph_gene_list:
            raise ValueError("No common genes found between the PPI network and the input gene list.")

        gene_to_idx_map = {gene: idx for idx, gene in enumerate(self.graph_gene_list)}
        net['source_idx'] = net['source'].map(gene_to_idx_map)
        net['target_idx'] = net['target'].map(gene_to_idx_map)
        net.dropna(subset=['source_idx', 'target_idx'], inplace=True)

        # Build graph
        g = nx.Graph()
        g.add_nodes_from(range(len(self.graph_gene_list)))
        g.add_edges_from(net[['source_idx', 'target_idx']].values.tolist())

        pyg_data = from_networkx(g)
        edge_index = pyg_data.edge_index
        self.register_buffer('edge_index', edge_index)

        self.input_gene_filter_indices: List[int] = [
            self.gene_list.index(gene) for gene in self.graph_gene_list
        ]

        # Gene features
        if not os.path.exists(args.gene_mean_std_fp):
            raise FileNotFoundError(f"Gene features file not found: {args.gene_mean_std_fp}")
        gf_df = pd.read_csv(args.gene_mean_std_fp, index_col=0)
        gf_df_reindexed = gf_df.reindex(self.graph_gene_list)
        if gf_df_reindexed.isnull().values.any():
            missing_genes = gf_df_reindexed[gf_df_reindexed.isnull().any(axis=1)].index.tolist()
            logger.warning(f"Missing genes in features file (zero-filled): {missing_genes}")
            gf_df_reindexed.fillna(0, inplace=True)

        gf_mat = gf_df_reindexed.values
        self.register_buffer('gene_features', torch.tensor(gf_mat, dtype=torch.float32))

        # Model dimensions
        self.gnn_n_genes = len(self.graph_gene_list)
        self.gene_hidden_dim = args.gene_hidden_dim
        self.cell_latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.embd_col_dim = args.gnn_embd_col_dim
        self.drop_p = args.gnn_drop_p
        self.num_layers = args.gnn_num_layers
        self.predict_cell_prop = args.predict_cell_prop

        self.position_encoding = position_encoding if self.args.using_positional_encoding else None

        # --- Define encoder ---
        self.encoder = PPIEncoder(
            in_feats=1 + self.gene_features.shape[1],
            gene_hidden_dim=self.gene_hidden_dim,
            num_layers=self.num_layers,
            drop_p=self.drop_p,
            latent_dim=self.cell_latent_dim,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        # Cell-level MLP
        self.cell_mlp = nn.Sequential(
            nn.Linear(self.gene_hidden_dim * self.cell_latent_dim, self.embd_col_dim),
            nn.LeakyReLU(inplace=True)
        )

        # Single layer for mu and logvar
        self.fc_mu_logvar = nn.Linear(self.embd_col_dim, self.n_cell_types * self.cell_latent_dim * 2)

        if self.predict_cell_prop:
            self.gnn_dd_alpha = nn.Linear(self.embd_col_dim, self.n_cell_types)

    def forward(self, x: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                sample_ids: Optional[List[str]] = None
                ) -> ModelOutput:
        """
        Forward pass with memory optimization
        """
        current_device = self.device_param.device
        x = x.to(current_device)
        batch_size = x.size(0)

        # ===== Memory Optimization: Split large batches =====
        if batch_size > self.max_batch_size_for_gnn and not self.training:
            # Split batch during inference
            return self._forward_split_batch(x, y, sample_ids)

        # Filter genes
        x_sub = x[:, self.input_gene_filter_indices].unsqueeze(-1)

        # Combine with gene features (avoid unnecessary copies)
        gf = self.gene_features.unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([x_sub, gf], dim=-1)

        # Clear intermediate variables
        del x_sub, gf

        # GNN encoder with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            gene_embeddings_from_gnn = checkpoint(
                self.encoder, node_features, self.edge_index,
                use_reentrant=False
            )
        else:
            gene_embeddings_from_gnn = self.encoder(node_features, self.edge_index)

        del node_features  # Free memory

        # MLP
        cell_embedding_before_heads = self.cell_mlp(gene_embeddings_from_gnn)
        del gene_embeddings_from_gnn

        # Get mu and logvar
        mu_logvar_flat = self.fc_mu_logvar(cell_embedding_before_heads)
        mu_logvar_structured = mu_logvar_flat.view(-1, self.n_cell_types, self.cell_latent_dim * 2)
        del mu_logvar_flat

        mu_all_types_raw, logvar_all_types_raw = torch.chunk(mu_logvar_structured, 2, dim=-1)
        del mu_logvar_structured

        mu_all_types = mu_all_types_raw.permute(0, 2, 1).contiguous()
        logvar_all_types = logvar_all_types_raw.permute(0, 2, 1).contiguous()
        del mu_all_types_raw, logvar_all_types_raw

        # Clamp logvar
        logvar_all_types = torch.clamp(logvar_all_types, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

        mu_mean = torch.mean(mu_all_types, dim=2)
        logvar_mean = torch.mean(logvar_all_types, dim=2)

        # Cell proportions
        output = ModelOutput()
        if self.predict_cell_prop:
            dd_alpha = F.softplus(self.gnn_dd_alpha(cell_embedding_before_heads)) + EPS
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha, device=current_device)
        elif y is not None:
            cell_prop = y.to(current_device)
        else:
            raise NotImplementedError('If self.predict_cell_prop is False, y must be provided.')

        output['cell_prop'] = cell_prop

        # Positional Encoding
        if self.position_encoding is not None:
            pe_matrix = self.position_encoding().to(current_device)
            exists = (cell_prop >= 0.01).float()
            pe_to_add = pe_matrix.t().unsqueeze(0)
            exists_mask = exists.unsqueeze(1)
            mu_all_types = mu_all_types + (pe_to_add * exists_mask)

        output['mu_all_types'] = mu_all_types
        output['logvar_all_types'] = logvar_all_types
        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['cell_type_existed'] = (cell_prop >= 0.01).float()  # (B, n_cell_types)

        return output

    def _forward_split_batch(self, x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             sample_ids: Optional[List[str]] = None) -> ModelOutput:
        """
        Split large batch into smaller chunks for inference
        """
        batch_size = x.size(0)
        chunk_size = self.max_batch_size_for_gnn

        outputs = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            x_chunk = x[i:end_idx]
            y_chunk = y[i:end_idx] if y is not None else None

            # Process chunk
            # with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
            #     chunk_output = self.forward(x_chunk, y_chunk, None)

            chunk_output = self.forward(x_chunk, y_chunk, None)
            outputs.append(chunk_output)

            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Merge outputs
        merged_output = ModelOutput()
        for key in outputs[0].keys():
            merged_output[key] = torch.cat([out[key] for out in outputs], dim=0)

        return merged_output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features with memory optimization
        """
        current_device = self.device_param.device
        x = x.to(current_device)
        batch_size = x.size(0)

        # Split if batch too large
        if batch_size > self.max_batch_size_for_gnn:
            features = []
            for i in range(0, batch_size, self.max_batch_size_for_gnn):
                end_idx = min(i + self.max_batch_size_for_gnn, batch_size)
                x_chunk = x[i:end_idx]

                with torch.cuda.amp.autocast(enabled=True):
                    chunk_features = self._extract_features_single(x_chunk)

                features.append(chunk_features)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return torch.cat(features, dim=0)
        else:
            return self._extract_features_single(x)

    def _extract_features_single(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function for feature extraction"""
        current_device = self.device_param.device
        batch_size = x.size(0)

        x_sub = x[:, self.input_gene_filter_indices].unsqueeze(-1)
        gf = self.gene_features.unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([x_sub, gf], dim=-1)

        del x_sub, gf

        gene_embeddings_from_gnn = self.encoder(node_features, self.edge_index)
        del node_features

        cell_embedding_before_heads = self.cell_mlp(gene_embeddings_from_gnn)
        del gene_embeddings_from_gnn

        return cell_embedding_before_heads

    def get_config(self):
        return {
            "params": {"args": self.args.to_dict()},
            "module_name": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }


class PPIEncoder(L.LightningModule):
    def __init__(self,
                 in_feats: int,
                 gene_hidden_dim: int,
                 latent_dim: int,
                 num_layers=3,
                 drop_p=0.1,
                 use_gradient_checkpointing=False):
        """
        GNN encoder with memory optimization
        """
        super().__init__()
        self.device_param = nn.Parameter(torch.empty(0))
        self.use_gradient_checkpointing = use_gradient_checkpointing

        layers_list = []
        current_sage_in_dim = in_feats

        for i in range(num_layers):
            sage_conv_module = SAGEConv(current_sage_in_dim, gene_hidden_dim)

            block_definition = [
                (sage_conv_module, 'x, edge_index -> x1'),
                (nn.Dropout(drop_p, inplace=True), 'x1 -> x2'),  # inplace=True
            ]

            if i < num_layers - 1:
                block_definition.append((nn.GELU(), 'x2 -> x_out'))
            else:
                block_definition.append((nn.Identity(), 'x2 -> x_out'))

            layers_list.append(Sequential('x, edge_index', block_definition))
            current_sage_in_dim = gene_hidden_dim

        self.gnn_layers = nn.ModuleList(layers_list)

        # Transformation heads
        self.feature_projector = nn.Linear(gene_hidden_dim, gene_hidden_dim)

        # Attention pooling
        self.attention_queries = nn.Parameter(torch.empty(1, latent_dim, gene_hidden_dim))
        nn.init.xavier_uniform_(self.attention_queries)
        self.attention_scale_factor = 1.0 / (gene_hidden_dim ** 0.5)

        self.output_norm = nn.LayerNorm(gene_hidden_dim * latent_dim)
        self.final_activation = nn.GELU()

    def forward(self, x: torch.Tensor, ppi_edge_index):
        """
        Forward with memory optimization
        """
        current_device = self.device_param.device
        ppi_edge_index = ppi_edge_index.to(current_device)

        batch_size, num_genes, num_node_features = x.shape
        embedded = x

        # Apply GNN layers with optional checkpointing
        for i, gnn_block in enumerate(self.gnn_layers):
            if self.use_gradient_checkpointing and self.training:
                # Use checkpointing for GNN layers
                embedded = checkpoint(
                    self._gnn_block_forward, gnn_block, embedded, ppi_edge_index,
                    use_reentrant=False
                )
            else:
                embedded = gnn_block(x=embedded, edge_index=ppi_edge_index)

        # Reshape back
        embedded = embedded.view(batch_size, num_genes, -1)

        # Project features
        projected_node_features = self.feature_projector(embedded)
        projected_node_features = F.gelu(projected_node_features)
        del embedded

        # Attention pooling (more memory efficient)
        attention_scores = torch.bmm(
            self.attention_queries.expand(batch_size, -1, -1),
            projected_node_features.transpose(-2, -1)
        ) * self.attention_scale_factor

        attention_weights = F.softmax(attention_scores, dim=-1)
        del attention_scores

        m_k_features = torch.bmm(attention_weights, projected_node_features)
        del attention_weights, projected_node_features

        # Flatten and normalize
        output_flat = m_k_features.reshape(batch_size, -1)
        del m_k_features

        output_processed = self.output_norm(output_flat)
        output_processed = self.final_activation(output_processed)

        return output_processed

    @staticmethod
    def _gnn_block_forward(gnn_block, x, edge_index):
        """Helper for gradient checkpointing"""
        return gnn_block(x=x, edge_index=edge_index)
