from pathlib import Path
from types import SimpleNamespace

import torch

from vaedecon.models.nn import DeSideCellPropPredictor


def _write_gene_list(path: Path, genes: list[str]) -> None:
    path.write_text("\n".join(genes) + "\n", encoding="utf-8")


def _write_gmt(path: Path) -> None:
    path.write_text(
        "pathway_a\tdesc\tgene_a\tgene_b\n"
        "pathway_b\tdesc\tgene_b\tgene_c\n",
        encoding="utf-8",
    )


def test_deside_predictor_returns_cell_prop_and_bulk_context_only(tmp_path: Path):
    gene_list_fp = tmp_path / "genes.txt"
    gmt_fp = tmp_path / "pathways.gmt"
    _write_gene_list(gene_list_fp, ["gene_a", "gene_b", "gene_c"])
    _write_gmt(gmt_fp)

    args = SimpleNamespace(
        input_dim=(1, 3),
        input_gene_list_fp=gene_list_fp,
        n_cell_types=2,
        predict_cell_prop=True,
        cell_prop_activation_function="softmax",
        deside_normalization="layer_normalization",
        deside_normalization_layer=[0, 0, 1],
        deside_pathway_network=True,
        deside_hidden_dims=[4, 3],
        deside_dropout_rate=[0.0, 0.0],
        deside_pathway_hidden_dims=[3, 3],
        deside_pathway_dropout_rate=[0.0, 0.0],
    )
    data_config = SimpleNamespace(
        pathway_file_path=[gmt_fp],
        scaling_by_constant=True,
        scaling_factor=20.0,
    )

    predictor = DeSideCellPropPredictor(args=args, data_config=data_config)
    output = predictor(torch.rand(2, 3))

    assert output["cell_prop"].shape == (2, 2)
    assert output["bulk_context_feature"].shape == (2, 3)
    assert "mu_all_types" not in output
    assert "logvar_all_types" not in output
    assert "cell_prop_feature" not in output
