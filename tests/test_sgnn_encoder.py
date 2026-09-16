from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from vaedecon.configs import DataConfig, ModelConfig, VAEDeconConfig
from vaedecon.models.gnn import EncoderSGNN


class DummyPositionalEncoding(nn.Module):
    def __init__(self, n_cell_types: int, latent_dim: int, fill_value: float = 1.0):
        super().__init__()
        self.register_buffer(
            "encoding",
            torch.full((n_cell_types, latent_dim), fill_value, dtype=torch.float32),
        )

    def forward(self) -> torch.Tensor:
        return self.encoding


def _write_sgnn_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    gene_list_fp = tmp_path / "genes.txt"
    gene_list_fp.write_text("A\nB\nC\n", encoding="utf-8")

    gene_stats_fp = tmp_path / "gene_stats.csv"
    pd.DataFrame(
        {
            "Type1_avg": [1.0, 2.0, 3.0],
            "Type1_std": [0.1, 0.2, 0.3],
            "Type2_avg": [1.5, 2.5, 3.5],
            "Type2_std": [0.2, 0.3, 0.4],
        },
        index=["A", "B", "C"],
    ).to_csv(gene_stats_fp)

    ppi_fp = tmp_path / "ppi.csv"
    pd.DataFrame(
        {
            "g1_symbol": ["A"],
            "g2_symbol": ["B"],
            "conn": [0.9],
        }
    ).to_csv(ppi_fp, index=False)

    return gene_list_fp, gene_stats_fp, ppi_fp


def _build_encoder(
    tmp_path: Path,
    **model_overrides,
) -> EncoderSGNN:
    gene_list_fp, gene_stats_fp, ppi_fp = _write_sgnn_fixture_files(tmp_path)
    model_kwargs = dict(
        input_gene_list_fp=gene_list_fp,
        gene_mean_std_fp=gene_stats_fp,
        latent_dim=2,
        n_cell_types=2,
        input_dim=(1, 3),
        gnn_n_genes=3,
        gene_hidden_dim=4,
        gnn_embd_col_dim=6,
        gnn_num_layers=2,
        gnn_drop_p=0.0,
        gnn_topk_attention=2,
        predict_cell_prop=False,
        using_positional_encoding=False,
    )
    model_kwargs.update(model_overrides)
    model = ModelConfig(**model_kwargs)
    data = DataConfig(ppi_file_path=ppi_fp)
    position_encoding = None
    if model.using_positional_encoding:
        position_encoding = DummyPositionalEncoding(
            n_cell_types=model.n_cell_types,
            latent_dim=model.latent_dim,
        )
    return EncoderSGNN(
        args=model,
        data_config=data,
        position_encoding=position_encoding,
    )


def test_config_accepts_new_sgnn_fields():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "gnn_gene_retention_mode": "keep_isolated",
                "gnn_network_cutoff": 0.4,
                "gnn_edge_weight_mode": "weighted_mean",
                "gnn_query_mode": "sample_conditioned",
                "gnn_dropedge_rate": 0.1,
                "gnn_attention_dropout_rate": 0.2,
            }
        }
    )

    assert loaded.model.gnn_gene_retention_mode == "keep_isolated"
    assert loaded.model.gnn_network_cutoff == 0.4
    assert loaded.model.gnn_edge_weight_mode == "weighted_mean"
    assert loaded.model.gnn_query_mode == "sample_conditioned"
    assert loaded.model.gnn_dropedge_rate == 0.1
    assert loaded.model.gnn_attention_dropout_rate == 0.2


def test_encoder_sgnn_keep_isolated_retains_non_edge_genes(tmp_path: Path):
    encoder = _build_encoder(
        tmp_path,
        gnn_gene_retention_mode="keep_isolated",
    )

    assert encoder.graph_gene_list == ["A", "B", "C"]
    assert encoder.connected_gene_count == 2
    assert encoder.isolated_gene_count == 1
    assert encoder.retained_edge_count == 1
    assert encoder.graph_diagnostics["isolated_genes"] == 1


def test_encoder_sgnn_drop_isolated_matches_legacy_behavior(tmp_path: Path):
    encoder = _build_encoder(
        tmp_path,
        gnn_gene_retention_mode="drop_isolated",
    )

    assert encoder.graph_gene_list == ["A", "B"]
    assert encoder.connected_gene_count == 2
    assert encoder.isolated_gene_count == 0
    assert encoder.filter_indices_tensor.tolist() == [0, 1]


def test_encoder_sgnn_weighted_edge_mode_preserves_conn_values(tmp_path: Path):
    weighted_encoder = _build_encoder(
        tmp_path / "weighted",
        gnn_edge_weight_mode="weighted_mean",
    )
    binary_encoder = _build_encoder(
        tmp_path / "binary",
        gnn_edge_weight_mode="binary_mean",
    )

    torch.testing.assert_close(
        weighted_encoder.encoder._base_adj_values,
        torch.tensor([0.9, 0.9], dtype=torch.float32),
    )
    torch.testing.assert_close(
        binary_encoder.encoder._base_adj_values,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )


def test_encoder_sgnn_sample_conditioned_queries_keep_mu_mean_consistent(tmp_path: Path):
    encoder = _build_encoder(
        tmp_path,
        gnn_query_mode="sample_conditioned",
        using_positional_encoding=True,
    )
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [
            [0.6, 0.4],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    output = encoder(x, y)

    assert output["cell_prop_feature"].shape == (2, encoder.embd_col_dim)
    assert output["mu_all_types"].shape == (2, encoder.cell_latent_dim, encoder.n_cell_types)
    torch.testing.assert_close(output["mu_mean"], output["mu_all_types"].mean(dim=2))
