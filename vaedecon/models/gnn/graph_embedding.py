from abc import ABC
import os
import torch
import torch.nn as nn
import scanpy as sc
import torch.nn.functional as F
from ..nn.base_architectures import BaseEncoder
from ..base import BaseModelConfig
from ...models.base.base_utils import ModelOutput
from ...models.nn.positional_encoding import PositionalEncoding
from typing import Optional
from torch_geometric.nn import Sequential, GCNConv, TransformerConv, SAGEConv
import lightning as L
from torch_geometric.utils import softmax, convert
import math
import numpy as np
import pandas as pd
from .common_functions_for_gnn import build_network, nx_to_pyg_edge_index, build_knn_graph

EPS = 1e-15
MAX_LOGSTD = 10
NETWORK_CUTOFF = 0.5
EXPRESSION_CUTOFF = 0.0


class EncoderGNN(BaseEncoder):
    """
    GNN encoder to integrate PPI (gene embeddings) and cell-cell co-expression network (cell embeddings).
    """
    def __init__(self, args: BaseModelConfig, position_encoding: Optional[PositionalEncoding] = None):
        """
        :param args: parameters for the GNN encoder
        """
        super(EncoderGNN, self).__init__()
        self.args = args
        self.col_dim = args.gnn_n_genes
        self.row_dim = args.gnn_row_dim
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
        self.encoder = MutualEncoder(self.col_dim, self.row_dim, self.num_layers, self.drop_p)
        # gene embeddings
        self.rows_encoder = DimEncoder(self.row_dim, self.inter_row_dim, self.embd_row_dim,
                                       drop_p=self.drop_p, scale_param=None, reducer=False)
        # cell embeddings / GEP embeddings for all cell types
        self.cols_encoder = DimEncoder(self.col_dim, self.inter_col_dim, self.embd_col_dim,
                                       drop_p=self.drop_p, reducer=False)
        # linear mapping to get the embeddings of all cell types
        self.deconvolution_layer = nn.Linear(self.embd_col_dim, self.cell_latent_dim * self.n_cell_types)
        # the log variance of the cell embeddings in the VAE model
        self.log_var = nn.Linear(self.embd_col_dim, self.cell_latent_dim)
        if self.predict_cell_prop:
            self.cell_prop = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.n_cell_types),
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
        knn_edge_index = build_knn_graph(obj)  # KNN graph edge index, cell-cell interaction network

        output = ModelOutput()
        embedded = self.encoder(x_t, knn_edge_index, ppi_edge_index)  # gene-cell embedding with the same shape as x.T
        # gene embedding (n_genes, self.embd_row_dim)
        embedded_rows = self.rows_encoder(embedded, ppi_edge_index)
        # cell embedding (n_cells, self.embd_col_dim)
        cell_embedding_by_gnn = self.cols_encoder(embedded.T, knn_edge_index, inference=True)
        embedding_all_types = self.deconvolution_layer(cell_embedding_by_gnn)
        embedding_all_types = embedding_all_types.view(-1, self.cell_latent_dim, self.n_cell_types)
        embedding = torch.mean(embedding_all_types, dim=2, keepdim=True)
        log_var = self.log_var(cell_embedding_by_gnn).reshape(-1, self.cell_latent_dim, 1)
        # out_features = self.feature_decoder(embedded_cols)  # can add linear layers directly for GEP reconstruction

        # TODO: getting cell proportions from DeSide
        if self.predict_cell_prop:
            output["cell_prop"] = self.cell_prop(cell_embedding_by_gnn).view((-1, self.n_cell_types, 1))
        # print(embedding_all_types.shape, output["cell_prop"].shape)
        if y is not None:
            y = y.view((-1, self.n_cell_types, 1))
            # embedding = torch.matmul(embedding_all_types, y)  # bulk mode embedding
            cell_type_existed = (y > 0.01).type(torch.int8).type(torch.float32)
        elif self.predict_cell_prop:
            # embedding = torch.matmul(embedding_all_types, output["cell_prop"])
            cell_type_existed = (output["cell_prop"] > 0.01).type(torch.int8).type(torch.float32)
        else:
            raise NotImplementedError('If self.predict_cell_prop is False, '
                                      'y (cell proportions of cell types) must be provided. '
                                      'It can be predicted by DeSide.')

        output['embedding'] = embedding
        output['log_var'] = log_var
        output['embedding_all_types'] = embedding_all_types
        output['cell_type_existed'] = cell_type_existed
        output['gene_embedding'] = embedded_rows

        return output

    def get_config(self):
        return {"params": {"args": self.args.to_dict()},
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                }


class MutualEncoder(L.LightningModule):
    def __init__(self, col_dim, row_dim, num_layers=4, drop_p=0.25):
        super(MutualEncoder, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.col_dim = col_dim
        self.row_dim = row_dim
        self.num_layers = num_layers

        self.rows_layers = nn.ModuleList([
            # x: The node feature matrix (typically shape [num_nodes, num_features])
            # edge_index: The edge index matrix (typically shape [2, num_edges])
            Sequential('x, edge_index', [
                # GraphSAGE Convolutional layer from PyTorch Geometric
                (SAGEConv(self.row_dim, self.row_dim), 'x, edge_index -> x1'),
                (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
                nn.LeakyReLU(inplace=True),
            ]) for _ in range(num_layers)])

        self.cols_layers = nn.ModuleList([
            Sequential('x, edge_index', [
                (SAGEConv(self.col_dim, self.col_dim), 'x, edge_index -> x1'),
                nn.LeakyReLU(inplace=True),
                (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
            ]) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, knn_edge_index, ppi_edge_index):
        """
        :param x: input node features, gene expression matrix (n_genes, n_cells)
        :param knn_edge_index: KNN graph edge index, cell-cell interaction network
        :param ppi_edge_index: PPI graph edge index
        :return: embedded x with the same shape as x
        """
        x = x.to(self.device)
        knn_edge_index = knn_edge_index.to(self.device)
        ppi_edge_index = ppi_edge_index.to(self.device)
        embedded = x.clone()

        for i in range(self.num_layers):
            embedded = self.cols_layers[i](embedded.T, knn_edge_index).T
            embedded = self.rows_layers[i](embedded, ppi_edge_index)

        return embedded


class TransformerConvReducerLayer(TransformerConv, ABC):
    """
    A modified TransformerConv layer that uses standardization and sigmoid activation
    for attention weights instead of softmax when scale_param is provided.

    Args:
        in_channels (int): Size of input node features
        out_channels (int): Size of output node features
        heads (int, optional): Number of attention heads. Default: 1
        dropout (float, optional): Dropout probability of attention weights. Default: 0
        add_self_loops (bool, optional): If True, adds self-loops to the graph. Default: True
        scale_param (float or None, optional): Parameter controlling the scaling of
            standardized attention scores. If None, uses softmax instead. Default: 2
        **kwargs: Additional arguments passed to TransformerConv
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0, add_self_loops=True,
                 scale_param: float | None = 2.0, **kwargs):
        super().__init__(in_channels, out_channels, heads, dropout, add_self_loops, **kwargs)
        self.threshold_alpha = None
        self.scale_param = scale_param

    def message(self, query_i, key_j, value_j,
                edge_attr, index, ptr,
                size_i):

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if not self.scale_param is None:
            alpha = alpha - alpha.mean()
            alpha = alpha / ((1 / self.scale_param) * alpha.std())
            alpha = F.sigmoid(alpha)
        else:
            alpha = softmax(alpha, index, ptr, size_i)
        self.threshold_alpha = alpha

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out


class DimEncoder(L.LightningModule):
    """
    Encoder for dimension reduction using Graph Neural Networks (GNNs) and attention mechanisms.

    This class implements a neural network architecture that reduces high-dimensional
    data (like gene expression data) to a lower-dimensional embedding space while
    preserving structural relationships through graph-based learning.

    The encoder consists of:
    1. A GCN layer followed by LeakyReLU and Dropout
    2. An attention-based layer (either TransformerConv or TransformerConvReducerLayer)

    The attention mechanism allows the model to learn which connections in the graph
    are most important for the embedding task.
    """
    def __init__(self, feature_dim, inter_dim, embd_dim, reducer=False, drop_p=0.2, scale_param: float | None = 3.0):
        """
        :param feature_dim: dimension of the column (the number of genes)
        :param inter_dim: dimension of the intermediate layer
        :param embd_dim: dimension of the embedding
        :param reducer: if True, TransformerConvReducerLayer will be used, otherwise TransformerConv will be used
        :param drop_p: dropout probability
        :param scale_param: scale parameter for the attention layer
        """
        super(DimEncoder, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.reducer = reducer

        self.encoder = Sequential('x, edge_index', [
            (GCNConv(self.feature_dim, self.inter_dim), 'x, edge_index -> x1'),
            nn.LeakyReLU(inplace=True),
            (nn.Dropout(drop_p, inplace=False), 'x1-> x2')
        ])
        if self.reducer:
            self.atten_layer = TransformerConvReducerLayer(self.inter_dim, self.embd_dim, dropout=drop_p,
                                                           add_self_loops=False, heads=1, scale_param=scale_param)
        else:
            self.atten_layer = TransformerConv(self.inter_dim, self.embd_dim, dropout=drop_p)

        self.atten_map = None
        self.atten_weights = None
        self.plot_count = 0

    def reduce_network(self, threshold=0.2, min_connect=10):
        self.plot_count += 1
        graph = self.atten_weights.cpu().detach().numpy()
        threshold_bound = np.percentile(graph, 10)
        threshold = min(threshold, threshold_bound)
        df = pd.DataFrame(
            {"v1": self.atten_map[0].cpu().detach().numpy(), "v2": self.atten_map[1].cpu().detach().numpy(),
             "atten": graph.squeeze()})
        saved_edges = df.groupby('v1')['atten'].nlargest(min_connect).index.values
        saved_edges = [v2 for _, v2 in saved_edges]
        df.iloc[saved_edges, 2] = threshold + EPS
        indexes = list(df.loc[df.atten >= threshold].index)
        atten_map = self.atten_map[:, indexes]
        self.atten_map = None
        self.atten_weights = None
        return atten_map, df

    def forward(self, x, edge_index, inference=False):
        """
        Forward pass of the encoder.

        Processes the input features through the GNN layers and returns
        the embedded representation.

        Parameters:
        -----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, feature_dim]
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape [2, num_edges]
        inference : bool, default=False
            If True, attention maps will not be stored (used during inference)
            If False, attention maps will be stored for later analysis (during training)

        Returns:
        --------
        torch.Tensor
            Embedded node features of shape [num_nodes, embd_dim]
        """

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        embedded = x.clone()
        embedded = self.encoder(embedded, edge_index)
        embedded, atten_map = self.atten_layer(embedded, edge_index, return_attention_weights=True)
        if self.reducer and not inference:
            if self.atten_map is None:
                self.atten_map = atten_map[0].detach()
                self.atten_weights = atten_map[1].detach()
            else:
                self.atten_map = torch.concat([self.atten_map.T, atten_map[0].detach().T]).T
                self.atten_weights = torch.concat([self.atten_weights, atten_map[1].detach()])

        return embedded
