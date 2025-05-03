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

        # --- load and prepare gene list & PPI network ---
        if args.input_gene_list_fp is None or not os.path.exists(args.input_gene_list_fp):
            raise FileNotFoundError(f"Gene list file not found: {args.input_gene_list_fp}")
        self.gene_list: List[str] = (
            pd.read_csv(args.input_gene_list_fp, header=None, index_col=0)
            .index.tolist()
        )

        if not os.path.exists(args.ppi_file_path):
            raise FileNotFoundError(f"PPI file not found: {args.ppi_file_path}")
        net = pd.read_csv(args.ppi_file_path)[["g1_symbol", "g2_symbol", "conn"]].drop_duplicates()
        if not args.biogrid_flag:
            net = net[net.conn >= NETWORK_CUTOFF]
        net.columns = ["source", "target"] if args.biogrid_flag else ["source", "target", "conn"]
        mask = (
                (net.source != net.target)
                & net.source.isin(self.gene_list)
                & net.target.isin(self.gene_list)
        )
        net = net[mask]

        # intersection genes in the PPI network and the input data, and the edge index in the PPI network
        keep_genes = pd.unique(net[["source", "target"]].values.ravel())
        G = nx.from_pandas_edgelist(net, "source", "target")
        data = from_networkx(G)
        edge_index = torch.tensor(data.edge_index, dtype=torch.long)
        self.keep_genes: List[str] = list(keep_genes)
        self.keep_idx: List[int] = [self.gene_list.index(g) for g in self.gene_list if g in self.keep_genes]
        self.register_buffer('edge_index', edge_index)

        # Gene features (mean and std) for each gene across cell types buffer
        if not os.path.exists(args.gene_mean_std_file_path):
            raise FileNotFoundError(f"Gene features file not found: {args.gene_mean_std_file_path}")
        gf_df = pd.read_csv(args.gene_mean_std_file_path, index_col=0)
        gf_mat = gf_df.iloc[self.keep_idx, :].values  # shape = (n_genes, n_feats)
        self.register_buffer('gene_features', torch.tensor(gf_mat, dtype=torch.float32))

        self.gnn_n_genes = len(self.keep_genes)  # the number of genes in the graph
        self.gene_hidden_dim = args.gene_hidden_dim
        # self.inter_col_dim = args.gnn_inter_col_dim
        self.cell_latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.embd_col_dim = args.gnn_embd_col_dim  # cell embeddings by the GNN, the layer before mu
        self.drop_p = args.gnn_drop_p
        self.num_layers = args.gnn_num_layers
        self.predict_cell_prop = args.predict_cell_prop
        # self.gene_list = None
        # self.gene_features = None
        self.biogrid_flag = args.biogrid_flag

        self.position_encoding = position_encoding if self.args.using_positional_encoding else None

        # --- define the encoder ---
        # a graph encoder with the same shape as the input, encoding the gene features and PPI information by GNN
        self.encoder = PPIEncoder(
            in_feats = 1 + self.gene_features.shape[1],  # registered buffer
            gene_hidden_dim=self.gene_hidden_dim,
            num_layers=self.num_layers,
            drop_p=self.drop_p)
        # cell-level MLP -> mu/logvar/proportion heads
        self.cell_mlp = nn.Sequential(
            nn.Linear(self.gnn_n_genes, self.embd_col_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.gcc_mu_list = nn.ModuleList(
            [nn.Linear(self.embd_col_dim, self.cell_latent_dim) for _ in range(self.n_cell_types)]
        )
        # the log variance of the cell embeddings for all cell types in the VAE model
        self.gnn_logvar_list = nn.ModuleList(
            [nn.Linear(self.embd_col_dim, self.cell_latent_dim) for _ in range(self.n_cell_types)]
        )
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
        x = x.to(self.device)  # B, n_genes
        x_sub = x[:, self.keep_idx].reshape(-1, self.gnn_n_genes, 1)  # (batch size, n_genes, 1)

        # Combine with gene features
        gf = self.gene_features.unsqueeze(0).expand(x_sub.shape[0], -1, -1)  # (batch_size, n_genes, n_features)
        x_sub = torch.cat((x_sub, gf), dim=-1)  # (batch_size, n_genes, 1 + n_features)
        # GNN encoder
        ppi_attention = self.encoder(x_sub, self.edge_index)  # gene bulk_embedding with shape (B, n_genes)

        cell_embedding_before_mu = self.cell_mlp(ppi_attention)  # (B, self.embd_col_dim)
        # embedding_all_types = self.deconvolution_layer(cell_embedding_before_mu)
        mu_list = [mu(cell_embedding_before_mu) for mu in self.gcc_mu_list]
        logvar_list = [logvar(cell_embedding_before_mu) for logvar in self.gnn_logvar_list]

        # TODO: getting cell proportions from DeSide
        output = ModelOutput()
        if self.predict_cell_prop:
            # output["cell_prop"] = self.cell_prop(cell_embedding_before_mu).view((-1, self.n_cell_types, 1))
            dd_alpha = F.softplus(self.gnn_dd_alpha(cell_embedding_before_mu)) + EPS
            output['dd_alpha'] = dd_alpha
            cell_prop = reparameterize_dirichlet(dd_alpha)
        elif y is not None:
            cell_prop = y.to(self.device)
        else:
            raise NotImplementedError('If self.predict_cell_prop is False, '
                                      'y (cell proportions of cell types) must be provided. '
                                      'It can be predicted by DeSide.')
        # Using position encoding to shift mu for each cell type, adding the positional encoding
        if self.position_encoding is not None:
            exists = (cell_prop >= 0.01).float()  # (B, n_cell_types)
            position_encoding_cell_type = torch.matmul(self.position_encoding(), exists)
            mu_list = [mu_list[i] + position_encoding_cell_type[i, :].unsqueeze(1) for i in range(self.n_cell_types)]

        output['mu_list'] = mu_list
        output['logvar_list'] = logvar_list
        output['cell_prop'] = cell_prop
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
                 num_layers=3,
                 drop_p=0.25):
        """
        Construct a GNN encoder only using the PPI network.
        :param gene_hidden_dim:
        :param num_layers:
        :param drop_p:
        :param gene_features: the means and stds of the gene expression values across cell types, used as gene features

        """
        super().__init__()
        layers = [
            Sequential('x, edge_index', [
            # First GraphSAGE layer with input dimension 1
            # TODO: monitor: here will become to a 3 dimensional tensor
            (SAGEConv(in_feats, gene_hidden_dim), 'x, edge_index -> x1'),
            (nn.Dropout(drop_p), 'x1 -> x2'),
            nn.LeakyReLU(inplace=True),
        ])]
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.col_dim = col_dim
        # self.gene_hidden_dim = gene_hidden_dim
        # self.num_layers = num_layers
        # self.gene_features = gene_features

        # First, define the initial layer separately (input dimension is 1 for gene expression values, the features for each gene)

        # Then define the remaining layers
        for _ in range(num_layers - 1):
            layers.append(Sequential('x, edge_index', [
                # Subsequent GraphSAGE layers with input dimension self.col_dim
                (SAGEConv(gene_hidden_dim, gene_hidden_dim), 'x, edge_index -> x1'),
                (nn.Dropout(drop_p), 'x1 -> x2'),
                nn.LeakyReLU(inplace=True) if _ != num_layers - 2 else nn.Softplus(),
            ]))

        # Combine the initial layer with subsequent layers
        self.gnn_layers = nn.ModuleList(layers)

        # # Final layer to reduce from (batch_size, num_genes, hidden_dim) to (batch_size, final_dim)
        # # Attention-based pooling
        # self.attention = nn.Sequential(
        #     nn.Linear(gene_hidden_dim, 1, bias=False),
        #     nn.Softmax(dim=2)
        # )

    def forward(self, x: torch.Tensor, ppi_edge_index):
        """
        :param x: input node features, gene expression matrix (batch_size, n_genes, 1 + n_features)
          - each row is one node in the graph
        :param ppi_edge_index: PPI graph edge index
        :return: embedded x with the same shape as x without the gene features
        """

        # knn_edge_index = knn_edge_index.to(self.device)
        ppi_edge_index = ppi_edge_index.to(self.device)
        embedded = x

        for layer in self.gnn_layers:
            embedded = layer(embedded, ppi_edge_index)
        embedded = embedded.sum(-1)  # sum over the gene features
        embedded = F.softmax(embedded, dim=-1)  # (batch_size, n_genes)， normalize to sum to 1 across genes

        return embedded
