import os
from typing import Optional, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# import scanpy as sc
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Sequential, SAGEConv
import lightning as L

from ..nn.base_architectures import BaseEncoder
# from ..base import BaseModelConfig
from ..vae.vae_config import VAEConfig
from ...models.base import ModelOutput, reparameterize_dirichlet
from ...models.nn.positional_encoding import PositionalEncoding
# from .common_functions_for_gnn import build_network, nx_to_pyg_edge_index

EPS = 1e-6
# MAX_LOGSTD = 10
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10
NETWORK_CUTOFF = 0.5
# EXPRESSION_CUTOFF = 0.0


class EncoderSGNN(BaseEncoder):
    """
    GNN-based encoder using a PPI network + optional gene-level features to produce
    per-cell latent embeddings (mu, logvar) and optional cell-type proportions.
    - Treat each sample independently.
    """
    def __init__(self, args: VAEConfig, position_encoding: Optional[PositionalEncoding] = None):
        """
        :param args: parameters for the GNN encoder
        """
        super().__init__()
        self.args = args
        self.device_param = nn.Parameter(torch.empty(0))  # To easily get the device of the model

        # --- load and prepare gene list & PPI network ---
        if args.input_gene_list_fp is None or not os.path.exists(args.input_gene_list_fp):
            raise FileNotFoundError(f"Gene list file not found: {args.input_gene_list_fp}")
        self.gene_list: List[str] = pd.read_csv(args.input_gene_list_fp, header=None).iloc[:, 0].tolist()

        if not os.path.exists(args.ppi_file_path):
            raise FileNotFoundError(f"PPI file not found: {args.ppi_file_path}")
        net_df = pd.read_csv(args.ppi_file_path)
        # Select relevant columns, handle potential missing "conn" for biogrid
        if args.biogrid_flag:
            if not all(col in net_df.columns for col in ["g1_symbol", "g2_symbol"]):
                raise ValueError("For biogrid_flag=True, PPI file must contain 'g1_symbol' and 'g2_symbol' columns.")
            # net = net_df[net.conn >= NETWORK_CUTOFF]
            net = net_df[["g1_symbol", "g2_symbol"]].copy()
            net.columns = ["source", "target"]
        else:
            if not all(col in net_df.columns for col in ["g1_symbol", "g2_symbol", "conn"]):
                raise ValueError("For biogrid_flag=False, PPI file must contain 'g1_symbol', 'g2_symbol', and 'conn' columns.")
            net = net_df[["g1_symbol", "g2_symbol", "conn"]].copy()
            net = net[net.conn >= NETWORK_CUTOFF]
            net.columns = ["source", "target", "conn"]
        net.drop_duplicates(subset=["source", "target"], inplace=True)   # remove duplicates
        # Filter edges where both genes are in the gene list
        mask = net.source.isin(self.gene_list) & net.target.isin(self.gene_list)
        net = net[mask]
        # Filter out self-loops
        net = net[net.source != net.target]

        # # intersection genes in the PPI network and the input data, and the edge index in the PPI network
        # keep_genes = pd.unique(net[["source", "target"]].values.ravel())
        # G = nx.from_pandas_edgelist(net, "source", "target")
        # data = from_networkx(G)
        # edge_index = torch.tensor(data.edge_index, dtype=torch.long)
        # self.keep_genes: List[str] = list(keep_genes)
        # self.keep_idx: List[int] = [self.gene_list.index(g) for g in self.gene_list if g in self.keep_genes]
        # self.register_buffer('edge_index', edge_index)

        # Create a mapping from gene symbol to a contiguous ID for graph construction
        # This ensures that node indices in the graph correspond to rows in our filtered expression matrix
        unique_genes_in_ppi = pd.unique(net[["source", "target"]].values.ravel("K"))
        self.graph_gene_list = [gene for gene in self.gene_list if gene in unique_genes_in_ppi]

        if not self.graph_gene_list:
            raise ValueError("No common genes found between the PPI network and the input gene list.")

        gene_to_idx_map = {gene: idx for idx, gene in enumerate(self.graph_gene_list)}

        # Map gene names in edges to their new indices
        net['source_idx'] = net['source'].map(gene_to_idx_map)
        net['target_idx'] = net['target'].map(gene_to_idx_map)

        # Drop rows where mapping failed (genes in PPI but not in filtered graph_gene_list - should not happen if logic is correct)
        net.dropna(subset=['source_idx', 'target_idx'], inplace=True)

        # Build graph with contiguous indices
        g = nx.Graph()
        g.add_nodes_from(range(len(self.graph_gene_list)))  # Nodes are 0 to N-1
        g.add_edges_from(net[['source_idx', 'target_idx']].values.tolist())

        pyg_data = from_networkx(g)
        edge_index = pyg_data.edge_index  # Already a tensor
        self.register_buffer('edge_index', edge_index)

        # Indices to select relevant genes from input 'x'
        self.input_gene_filter_indices: List[int] = [
            self.gene_list.index(gene) for gene in self.graph_gene_list
        ]  # self.graph_gene_list is a subset of self.gene_list with the same order

        # Gene features (mean and std) for each gene across cell types buffer
        if not os.path.exists(args.gene_mean_std_fp):
            raise FileNotFoundError(f"Gene features file not found: {args.gene_mean_std_fp}")
        gf_df = pd.read_csv(args.gene_mean_std_fp, index_col=0)
        # Reindex gf_df to match self.graph_gene_list and handle missing genes
        gf_df_reindexed = gf_df.reindex(self.graph_gene_list)
        # Check for NaNs after reindexing (genes in graph_gene_list but not in gf_df)
        if gf_df_reindexed.isnull().values.any():
            missing_genes = gf_df_reindexed[gf_df_reindexed.isnull()].any(axis=1).index.tolist()
            print(f"Warning: The following genes in the graph are missing from the gene features file and "
                  f"will be zero-filled: {missing_genes}")
            gf_df_reindexed.fillna(0, inplace=True)

        gf_mat = gf_df_reindexed.values  # shape = (n_genes_in_graph [intersection genes with gene_list], n_feats)
        self.register_buffer('gene_features', torch.tensor(gf_mat, dtype=torch.float32))

        self.gnn_n_genes = len(self.graph_gene_list)  # the number of genes in the graph
        self.gene_hidden_dim = args.gene_hidden_dim
        self.cell_latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.embd_col_dim = args.gnn_embd_col_dim  # cell embeddings by the GNN, the layer before mu
        self.drop_p = args.gnn_drop_p
        self.num_layers = args.gnn_num_layers
        self.predict_cell_prop = args.predict_cell_prop

        self.position_encoding = position_encoding if self.args.using_positional_encoding else None

        # --- define the encoder ---
        # a graph encoder with the same shape as the input, encoding the gene features and PPI information by GNN
        self.encoder = PPIEncoder(
            in_feats = 1 + self.gene_features.shape[1],  # registered buffer, exp value + gene features
            gene_hidden_dim=self.gene_hidden_dim,
            num_layers=self.num_layers,
            drop_p=self.drop_p,
            num_genes=self.gnn_n_genes)
        # cell-level MLP -> mu/logvar/proportion heads
        self.cell_mlp = nn.Sequential(
            nn.Linear(self.gnn_n_genes, self.embd_col_dim),  # Input is output of PPIEncoder
            nn.LeakyReLU(inplace=True)
        )
        # self.gcc_mu_list = nn.ModuleList(
        #     [nn.Linear(self.embd_col_dim, self.cell_latent_dim) for _ in range(self.n_cell_types)]
        # )
        # # the log variance of the cell embeddings for all cell types in the VAE model
        # self.gnn_logvar_list = nn.ModuleList(
        #     [nn.Linear(self.embd_col_dim, self.cell_latent_dim) for _ in range(self.n_cell_types)]
        # )
        # Using a single layer to get mu and logvar for all cell types
        self.fc_mu_logvar = nn.Linear(self.embd_col_dim, self.n_cell_types * self.cell_latent_dim * 2)

        if self.predict_cell_prop:
            self.gnn_dd_alpha = nn.Linear(self.embd_col_dim, self.n_cell_types)

    def forward(self, x: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                sample_ids: Optional[List[str]] = None
                ) -> ModelOutput:
        """
        :param x: input node features, gene expression matrix (n_cells, n_genes)
        :param y: The cell proportions of the input data. Defaults to None.
        :param sample_ids: The sample IDs of the input data. Defaults to None.
        :return: gene bulk_embedding, cell bulk_embedding, reconstructed gene expression
        """
        current_device = self.device_param.device  # Get model's current device
        x = x.to(current_device)  # B, n_genes

        # Filter x to only include genes in the graph, in the correct order
        x_sub = x[:, self.input_gene_filter_indices]
        x_sub = x_sub.unsqueeze(-1)  # (batch size, self.gnn_n_genes, 1) for expression values

        # Combine with gene features
        # self.gene_features: (self.gnn_n_genes, n_static_features)
        gf = self.gene_features.unsqueeze(0).expand(x_sub.shape[0], -1, -1)  # (batch_size, n_genes, n_features)
        # node_features: (batch_size, gnn_n_genes, 1 + n_static_features)
        node_features = torch.cat([x_sub, gf], dim=-1)
        # x_sub = torch.cat((x_sub, gf), dim=-1)  # (batch_size, n_genes, 1 + n_features)

        # GNN encoder
        gene_embeddings_from_gnn = self.encoder(node_features, self.edge_index)  # gene bulk_embedding with shape (B, n_genes)

        # MLP to get sample-level embeddings
        cell_embedding_before_heads = self.cell_mlp(gene_embeddings_from_gnn)  # (B, self.embd_col_dim)

        # # embedding_all_types = self.deconvolution_layer(cell_embedding_before_mu)
        # mu_list = [mu(cell_embedding_before_heads) for mu in self.gcc_mu_list]
        # logvar_list = [logvar(cell_embedding_before_heads) for logvar in self.gnn_logvar_list]
        # mu_all_types = torch.stack(mu_list, dim=2)  # (B, latent_dim, n_cell_types)
        # logvar_all_types = torch.stack(logvar_list, dim=2)  # (B, latent_dim, n_cell_types)

        # Get all mu and logvar values from a single linear layer
        # mu_logvar_flat: (batch_size, self.n_cell_types * self.cell_latent_dim * 2)
        mu_logvar_flat = self.fc_mu_logvar(cell_embedding_before_heads)

        # Reshape to separate mu and logvar for each cell type
        # mu_logvar_structured: (batch_size, self.n_cell_types, self.cell_latent_dim * 2)
        mu_logvar_structured = mu_logvar_flat.view(
            -1, self.n_cell_types, self.cell_latent_dim * 2
        )

        # Split into mu and logvar
        # mu_all_types_raw, logvar_all_types_raw: (batch_size, self.n_cell_types, self.cell_latent_dim)
        mu_all_types_raw, logvar_all_types_raw = torch.chunk(mu_logvar_structured, 2, dim=-1)

        # Permute to (batch_size, self.cell_latent_dim, self.n_cell_types) to match original output
        mu_all_types = mu_all_types_raw.permute(0, 2, 1).contiguous()
        logvar_all_types = logvar_all_types_raw.permute(0, 2, 1).contiguous()

        # IMPORTANT: Clamp logvar for numerical stability BEFORE using it in KL divergence or reparameterization
        logvar_all_types = torch.clamp(logvar_all_types, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

        mu_mean = torch.mean(mu_all_types, dim=2)  # (B, latent_dim)
        logvar_mean = torch.mean(logvar_all_types, dim=2)  # (B, latent_dim)

        # TODO: getting cell proportions from DeSide
        output = ModelOutput()
        if self.predict_cell_prop:
            # output["cell_prop"] = self.cell_prop(cell_embedding_before_mu).view((-1, self.n_cell_types, 1))
            dd_alpha = F.softplus(self.gnn_dd_alpha(cell_embedding_before_heads)) + EPS
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha, device=current_device)
        elif y is not None:
            cell_prop = y.to(self.device)
        else:
            raise NotImplementedError('If self.predict_cell_prop is False, '
                                      'y (cell proportions of cell types) must be provided. '
                                      'It can be predicted by DeSide.')
        output['cell_prop'] = cell_prop  # (B, n_cell_types)

        # Positional Encoding (Vectorized)
        # mu_all_types is (B, latent_dim, n_cell_types)
        # We want to add PE per cell type. PE should be (n_cell_types, latent_dim)
        if self.position_encoding is not None:
            # Assuming self.position_encoding.weight is (n_cell_types, latent_dim)
            # or self.position_encoding() returns that.
            pe_matrix = self.position_encoding()  # (n_cell_types, latent_dim)
            pe_matrix = pe_matrix.to(current_device)

            # 'exists' indicates which cell types are present enough to get PE
            # cell_prop: (B, n_cell_types)
            exists = (cell_prop >= 0.01).float()  # (B, n_cell_types)

            # Reshape for broadcasting:
            # mu_all_types: (B, latent_dim, n_cell_types)
            # pe_matrix: (n_cell_types, latent_dim) -> (1, latent_dim, n_cell_types) for broadcasting
            # exists: (B, n_cell_types) -> (B, 1, n_cell_types) for broadcasting
            pe_to_add = pe_matrix.t().unsqueeze(0)  # (1, latent_dim, n_cell_types)
            exists_mask = exists.unsqueeze(1)  # (B, 1, n_cell_types)

            mu_all_types = mu_all_types + (pe_to_add * exists_mask)

        # output['mu_list'] = mu_list
        # output['logvar_list'] = logvar_list
        output['mu_all_types'] = mu_all_types
        output['logvar_all_types'] = logvar_all_types
        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        # output['cell_prop'] = cell_prop
        output['cell_type_existed'] = (cell_prop >= 0.01).float()  # (B, n_cell_types)

        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                }


class PPIEncoder(L.LightningModule):
    def __init__(self,
                 in_feats: int,
                 gene_hidden_dim: int,
                 num_genes: int,
                 num_layers=3,
                 drop_p=0.25):
        """
        Construct a GNN encoder only using the PPI network.
        :param in_feats: The number of input features for each gene (1 for expression values, more for other features)
        :param gene_hidden_dim:
        :param num_layers:
        :param drop_p:
        """
        super().__init__()
        self.device_param = nn.Parameter(torch.empty(0))  # To easily get the device of the model
        self.output_norm = nn.LayerNorm(num_genes)

        layers_list = []
        current_dim = in_feats

        for i in range(num_layers):
            # Last layer might have different activation or no activation if followed by another transform
            is_last_layer_in_gnn_stack = (i == num_layers - 1)
            sage_layer = SAGEConv(current_dim, gene_hidden_dim)
            # For SAGEConv, output is (num_nodes, gene_hidden_dim).
            # If input is (batch_size, num_nodes, current_dim), PyG SAGEConv might expect
            # (batch_size * num_nodes, current_dim) and then reshape.
            # The Sequential syntax handles this if x is (N, F_in).
            # If x is (B, N, F_in), need to be careful.
            # Assuming input to SAGEConv will be (B * N, F_in)

            block_layers = [
                (sage_layer, 'x, edge_index -> x'),  # SAGEConv updates node features
                (nn.Dropout(drop_p), 'x -> x'),  # Dropout layer
            ]
            if not is_last_layer_in_gnn_stack:
                block_layers.append(nn.LeakyReLU(inplace=True))
            # else:
                # No activation or Softplus like before if needed, but often not for intermediate GNN layer.
                # block_layers.append(nn.Softplus()) # As in original code for the last internal GNN layer
                                                    # but consider if this is the true "output" layer.

            layers_list.append(Sequential('x, edge_index', block_layers))
            current_dim = gene_hidden_dim # Update current_dim for the next layer

        # layers = [
        #     Sequential('x, edge_index', [
        #     # First GraphSAGE layer with input dimension 1
        #     # TODO: monitor: here will become to a 3 dimensional tensor
        #     (SAGEConv(in_feats, gene_hidden_dim), 'x, edge_index -> x1'),
        #     (nn.Dropout(drop_p), 'x1 -> x2'),
        #     nn.LeakyReLU(inplace=True),
        # ])]
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.col_dim = col_dim
        # self.gene_hidden_dim = gene_hidden_dim
        # self.num_layers = num_layers
        # self.gene_features = gene_features

        # First, define the initial layer separately (input dimension is 1 for gene expression values, the features for each gene)

        # # Then define the remaining layers
        # for _ in range(num_layers - 1):
        #     layers.append(Sequential('x, edge_index', [
        #         # Subsequent GraphSAGE layers with input dimension self.col_dim
        #         (SAGEConv(gene_hidden_dim, gene_hidden_dim), 'x, edge_index -> x1'),
        #         (nn.Dropout(drop_p), 'x1 -> x2'),
        #         nn.LeakyReLU(inplace=True) if _ != num_layers - 2 else nn.Softplus(),
        #     ]))

        # Combine the initial layer with subsequent layers
        self.gnn_layers = nn.ModuleList(layers_list)

        # After GNN layers, features are (batch_size, n_genes, gene_hidden_dim)
        # We need to get to (batch_size, n_genes) for the cell_mlp

        # # Final layer to reduce from (batch_size, num_genes, hidden_dim) to (batch_size, final_dim)
        # # Attention-based pooling
        # self.attention = nn.Sequential(
        #     nn.Linear(gene_hidden_dim, 1, bias=False),
        #     nn.Softmax(dim=2)
        # )

    def forward(self, x: torch.Tensor, ppi_edge_index):
        """
        :param x: Node features (batch_size, n_genes, 1 + n_features)
          - each row is one node in the graph
        :param ppi_edge_index: PPI graph edge index (2, num_edges)
        :return: embedded x with the same shape as x without the gene features (batch_size, n_genes)
        """
        current_device = self.device_param.device
        ppi_edge_index = ppi_edge_index.to(current_device)

        batch_size, num_genes, num_node_features = x.shape

        # Reshape for PyG layers: (batch_size * n_genes, num_node_features)
        # embedded = x.reshape(batch_size * num_genes, num_node_features)

        # knn_edge_index = knn_edge_index.to(self.device)
        embedded = x

        for layer in self.gnn_layers:
            embedded = layer(embedded, ppi_edge_index)  # Now embedded is (B*N, gene_hidden_dim)

        # Reshape back to (batch_size, n_genes, gene_hidden_dim)
        embedded = embedded.view(batch_size, num_genes, -1)  # -1 infers gene_hidden_dim

        embedded = embedded.sum(-1)  # sum over the gene features
        # embedded = F.softmax(embedded, dim=-1)  # (batch_size, n_genes)， normalize to sum to 1 across genes
        embedded = self.output_norm(embedded)

        return embedded
