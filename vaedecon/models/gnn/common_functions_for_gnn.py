import torch
import networkx as nx
import scanpy as sc
from torch_geometric.utils import softmax, convert
import numpy as np
import pandas as pd

EPS = 1e-15
MAX_LOGSTD = 10
NETWORK_CUTOFF = 0.5
EXPRESSION_CUTOFF = 0.0


def build_network(obj, net, biogrid_flag=False, human_flag=False):
    """
    Build a gene-gene network from the provided interaction information.
    Args:
      obj (anndata.AnnData): Single-cell data object (AnnData) containing gene expression data.
      net (pandas.DataFrame): DataFrame containing gene interactions (Source, Target, and Conn columns).
      biogrid_flag (bool, optional): If True, columns for net are set to ["Source", "Target"] only.
      human_flag (bool, optional): If True, keeps gene names unchanged; otherwise adjusts gene name casing.
    Returns:
      tuple:
        pandas.DataFrame: Filtered interaction DataFrame for valid genes.
        networkx.Graph: Graph representation of the gene network.
        pandas.DataFrame: Node-level gene expression features.
    """
    if not biogrid_flag:
        net.columns = ["Source", "Target", "Conn"]
        net = net.loc[net.Conn >= NETWORK_CUTOFF]

    else:
        net.columns = ["Source", "Target"]

    if not human_flag:
        net["Source"] = net["Source"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        net["Target"] = net["Target"].apply(lambda x: x[0] + x[1:].lower()).astype(str)

    genes = list(pd.concat([net.Source, net.Target]).drop_duplicates())
    genes = obj.var[obj.var.index.isin(genes)].index
    node_feature = sc.get.obs_df(obj.raw.to_adata(), list(genes)).T  # genes x cells
    node_feature["non_zero"] = node_feature.apply(lambda x: x.astype(bool).sum(), axis=1)
    node_feature = node_feature.loc[node_feature.non_zero > node_feature.shape[1] * EXPRESSION_CUTOFF]
    node_feature.drop("non_zero", axis=1, inplace=True)

    net = net.loc[net.Source != net.Target]
    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]

    gp = nx.from_pandas_edgelist(net, "Source", "Target")

    node_feature = node_feature.loc[list(gp.nodes)]

    return net, gp, node_feature


def nx_to_pyg_edge_index(G, mapping=None, device='cpu'):
    G = G.to_directed() if not nx.is_directed(G) else G
    if mapping is None:
        mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long).to(device)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index, mapping


def build_knn_graph(obj):
    if "distances" not in obj.obsp:
        sc.pp.neighbors(obj, n_neighbors=25, n_pcs=15, use_rep="X", metric="euclidean")
    graph = obj.obsp["distances"].toarray()
    graph = (graph > 0).astype(int)
    graph = nx.from_numpy_array(np.matrix(graph))
    ppi_geo = convert.from_networkx(graph)
    edge_index = ppi_geo.edge_index
    # sc.pp.highly_variable_genes(obj)
    return edge_index
