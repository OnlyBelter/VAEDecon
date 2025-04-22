import os
import torch
import torch.nn as nn
import scanpy as sc
from ..nn.base_architectures import BaseEncoder
from ..base import BaseModelConfig
from ...models.base.base_utils import ModelOutput
from ...models.nn.positional_encoding import PositionalEncoding
from typing import Optional
from torch_geometric.nn import Sequential, SAGEConv
import lightning as L
import pandas as pd
from .common_functions_for_gnn import build_network, nx_to_pyg_edge_index

EPS = 1e-15
MAX_LOGSTD = 10
NETWORK_CUTOFF = 0.5
EXPRESSION_CUTOFF = 0.0


class EncoderSGNN(BaseEncoder):
    """
    Single GNN which only including the PPI (gene embeddings) to represent each cell (cell embeddings).
    - Treat each sample independently.
    """
    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        """
        :param args: parameters for the GNN encoder
        """
        super(EncoderSGNN, self).__init__()
        self.args = args
        self.col_dim = args.gnn_col_dim  # dimension of the column (the number of genes)
        # self.row_dim = args.gnn_row_dim
        self.gene_hidden_dim = args.gene_hidden_dim
        self.inter_row_dim = args.gnn_inter_row_dim
        self.embd_row_dim = args.gnn_embd_row_dim
        self.inter_col_dim = args.gnn_inter_col_dim
        self.cell_latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.embd_col_dim = args.gnn_embd_col_dim  # cell embeddings by the GNN
        self.drop_p = args.gnn_drop_p
        self.num_layers = args.gnn_num_layers
        self.predict_cell_prop = args.predict_cell_prop
        self.gene_list = args.gene_list
        if os.path.exists(args.ppi_file_path):
            self.ppi_file_path = args.ppi_file_path
        else:
            raise FileNotFoundError(f"File {args.ppi_file_path} not found.")
        # self.lambda_rows = lambda_rows
        # self.lambda_cols = lambda_cols

        # graph encoder with the same shape as the input
        self.encoder = PPIEncoder(gene_hidden_dim=self.gene_hidden_dim, num_layers=self.num_layers,
                                  drop_p=self.drop_p)
        # gene embeddings
        # self.rows_encoder = DimEncoder(self.row_dim, self.inter_row_dim, self.embd_row_dim,
        #                                drop_p=self.drop_p, scale_param=None, reducer=False)
        # cell embeddings / GEP embeddings for all cell types
        # self.cols_encoder = DimEncoder(self.col_dim, self.inter_col_dim, self.embd_col_dim,
        #                                drop_p=self.drop_p, reducer=False)
        self.cell_embedding_layer = nn.Sequential(
            nn.Linear(self.col_dim, self.embd_col_dim),
            nn.LeakyReLU(inplace=True)
        )
        # linear mapping to get the embeddings of all cell types
        # (n_samples, n_genes) -> (n_samples, cell_latent_dim x n_cell_types)
        self.deconvolution_layer = nn.Linear(self.embd_col_dim, self.cell_latent_dim * self.n_cell_types)
        # the log variance of the cell embeddings in the VAE model
        self.log_var = nn.Linear(self.embd_col_dim, self.cell_latent_dim)
        if self.predict_cell_prop:
            self.cell_prop = nn.Sequential(
                nn.Linear(self.embd_col_dim, self.n_cell_types),
                nn.Softmax(dim=1)
            )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, sample_ids: list = None) -> ModelOutput:
        """
        :param x: input node features, gene expression matrix (n_cells, n_genes)
        :param y: The cell proportions of the input data. Defaults to None.
        :param sample_ids: The sample IDs of the input data. Defaults to None.
        :return: gene embedding, cell embedding, reconstructed gene expression
        """
        obs = pd.DataFrame(data=None, index=sample_ids)
        var = pd.DataFrame(data=None, index=self.gene_list)
        obj = sc.AnnData(X=x.detach().cpu().numpy(), var=var, obs=obs)
        x_t = x.T
        if obj.raw is None:
            obj.raw = obj.copy()
        ppi = None
        try:
            # print(f'Loading human PPI from: {self.ppi_file_path}...')
            net = pd.read_csv(self.ppi_file_path)[
                ["g1_symbol", "g2_symbol", "conn"]].drop_duplicates()
            net, ppi, node_feature = build_network(obj, net, human_flag=True)
            obj = obj[:, node_feature.index]
            x_t = torch.from_numpy(node_feature.values)
            # print(f"N genes: {node_feature.shape}")
        except Exception as e:
            print(f"Error during network construction: {e}")
        ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)  # PPI graph edge index
        # ppi_edge_index = ppi_edge_index

        output = ModelOutput()
        x = x_t.T.view(-1, self.col_dim, 1)  # (n_cells, n_genes, 1)
        ppi_attention = self.encoder(x, ppi_edge_index).squeeze()  # gene embedding with shape (n_cells, n_genes)
        # gene embedding (n_genes, self.embd_row_dim)
        # embedded_rows = self.rows_encoder(embedded, ppi_edge_index)
        # cell embedding (n_cells, self.embd_col_dim)
        # cell_embedding_by_gnn = self.cols_encoder(embedded.T, knn_edge_index, inference=True)
        cell_embedding = self.cell_embedding_layer(ppi_attention)  # (n_cells, self.embd_col_dim)
        embedding_all_types = self.deconvolution_layer(cell_embedding)
        embedding_all_types = embedding_all_types.view(-1, self.cell_latent_dim, self.n_cell_types)
        embedding = torch.mean(embedding_all_types, dim=2, keepdim=True)  # bulk mode embedding
        log_var = self.log_var(cell_embedding).reshape(-1, self.cell_latent_dim, 1)  # todo: check here
        # out_features = self.feature_decoder(embedded_cols)  # can add linear layers directly for GEP reconstruction

        # TODO: getting cell proportions from DeSide
        if self.predict_cell_prop:
            output["cell_prop"] = self.cell_prop(cell_embedding).view((-1, self.n_cell_types, 1))
        # print(embedding_all_types.shape, output["cell_prop"].shape)
        if y is not None:
            y = y.view((-1, self.n_cell_types, 1))
            # embedding = torch.matmul(embedding_all_types, y)  # bulk mode embedding
            cell_type_existed = (y >= 0.01).type(torch.int8).type(torch.float32)
        elif self.predict_cell_prop:
            # embedding = torch.matmul(embedding_all_types, output["cell_prop"])
            cell_type_existed = (output["cell_prop"] > 0.01).type(torch.int8).type(torch.float32)
        else:
            raise NotImplementedError('If self.predict_cell_prop is False, '
                                      'y (cell proportions of cell types) must be provided. '
                                      'It can be predicted by DeSide.')

        output['embedding'] = embedding  # the mean of all cell types
        output['log_var'] = log_var
        output['embedding_all_types'] = embedding_all_types
        output['cell_type_existed'] = cell_type_existed
        # output['gene_embedding'] = embedded_rows

        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                }


class PPIEncoder(L.LightningModule):
    def __init__(self, gene_hidden_dim, num_layers=3, drop_p=0.25):
        super(PPIEncoder, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        # self.col_dim = col_dim
        self.gene_hidden_dim = gene_hidden_dim
        self.num_layers = num_layers

        # First, define the initial layer separately (input dimension is 1 for gene expression values)
        initial_layer = Sequential('x, edge_index', [
            # First GraphSAGE layer with input dimension 1
            # TODO: monitor: here will become to a 3 dimensional tensor
            (SAGEConv(1, self.gene_hidden_dim), 'x, edge_index -> x1'),
            (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
            nn.LeakyReLU(inplace=True),
        ])

        # Then define the remaining layers
        subsequent_layers = [
            Sequential('x, edge_index', [
                # Subsequent GraphSAGE layers with input dimension self.col_dim
                (SAGEConv(self.gene_hidden_dim, self.gene_hidden_dim), 'x, edge_index -> x1'),
                (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
                nn.LeakyReLU(inplace=True),
            ]) for _ in range(num_layers - 1)
        ]

        # Combine the initial layer with subsequent layers
        self.gnn_layers = nn.ModuleList([initial_layer] + subsequent_layers)

        # Final layer to reduce from (batch_size, num_genes, hidden_dim) to (batch_size, final_dim)
        # Option 1: Attention-based pooling
        self.attention = nn.Sequential(
            nn.Linear(self.gene_hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # self.rows_layers = nn.ModuleList(gnn_layers + [attention])

        # self.cols_layers = nn.ModuleList([
        #     Sequential('x, edge_index', [
        #         (SAGEConv(self.col_dim, self.col_dim), 'x, edge_index -> x1'),
        #         nn.LeakyReLU(inplace=True),
        #         (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
        #     ]) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, ppi_edge_index):
        """
        :param x: input node features, gene expression matrix (batch_size, n_genes, 1)
          - each row is one node in the graph
        :param ppi_edge_index: PPI graph edge index
        :return: embedded x with the same shape as x
        """
        x = x.to(self.device)
        # knn_edge_index = knn_edge_index.to(self.device)
        ppi_edge_index = ppi_edge_index.to(self.device)
        embedded = x.clone()

        for i in range(self.num_layers):
            # embedded = self.cols_layers[i](embedded.T, knn_edge_index).T
            embedded = self.gnn_layers[i](embedded, ppi_edge_index)
        embedded = self.attention(embedded)

        return embedded


