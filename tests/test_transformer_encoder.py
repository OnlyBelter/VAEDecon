from types import SimpleNamespace

import torch

from vaedecon.models.nn.transformer import GeneTransformerEncoder
from vaedecon.models.base.positional_encoding import PositionalEncoding


def _build_transformer_config(**overrides):
    base = {
        "input_dim": (1, 8),
        "latent_dim": 4,
        "n_cell_types": 3,
        "predict_cell_prop": True,
        "cell_prop_activation_function": "softmax",
        "using_positional_encoding": False,
        "transformer_d_model": 16,
        "transformer_nhead": 4,
        "transformer_num_layers": 1,
        "transformer_dim_feedforward": 32,
        "transformer_dropout": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gene_transformer_encoder_returns_expected_shapes():
    encoder = GeneTransformerEncoder(args=_build_transformer_config())
    x = torch.randn(2, 8)

    output = encoder(x)

    assert output["cell_prop"].shape == (2, 3)
    assert output["mu_all_types"].shape == (2, 4, 3)
    assert output["logvar_all_types"].shape == (2, 4, 3)
    assert output["bulk_context_feature"].shape == (2, 16)
    assert output["cell_type_context_features"].shape == (2, 3, 16)
    assert torch.equal(output["cell_prop_feature"], output["bulk_context_feature"])


def test_gene_transformer_encoder_uses_global_branch_for_bulk_context():
    encoder = GeneTransformerEncoder(args=_build_transformer_config())
    x = torch.randn(2, 8)

    output = encoder(x)

    with torch.no_grad():
        val_emb = encoder.value_projector(x.unsqueeze(-1))
        gene_kv = encoder.gene_id_embedding.unsqueeze(0) + val_emb
        global_query = encoder.global_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        cell_type_queries = encoder.cell_type_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        latents = torch.cat((global_query, cell_type_queries), dim=1)
        attn_out, _ = encoder.cross_attn(
            query=encoder.norm_latents(latents),
            key=gene_kv,
            value=gene_kv,
        )
        latents = latents + attn_out
        latents = encoder.transformer(latents)
        raw_global = latents[:, 0, :]
        expected_bulk = encoder.global_branch(raw_global)

    assert torch.allclose(output["bulk_context_feature"], expected_bulk)
    assert not torch.allclose(output["bulk_context_feature"], raw_global)


def test_gene_transformer_encoder_uses_cell_type_branch_for_context_tokens():
    encoder = GeneTransformerEncoder(args=_build_transformer_config())
    x = torch.randn(2, 8)

    output = encoder(x)

    with torch.no_grad():
        val_emb = encoder.value_projector(x.unsqueeze(-1))
        gene_kv = encoder.gene_id_embedding.unsqueeze(0) + val_emb
        global_query = encoder.global_query.unsqueeze(0).expand(x.shape[0], -1, -1)
        cell_type_queries = encoder.cell_type_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        latents = torch.cat((global_query, cell_type_queries), dim=1)
        attn_out, _ = encoder.cross_attn(
            query=encoder.norm_latents(latents),
            key=gene_kv,
            value=gene_kv,
        )
        latents = latents + attn_out
        latents = encoder.transformer(latents)
        raw_type_out = latents[:, 1:, :]
        expected_type_out = encoder.cell_type_branch(raw_type_out)

    assert torch.allclose(output["cell_type_context_features"], expected_type_out)
    assert not torch.allclose(output["cell_type_context_features"], raw_type_out)


def test_gene_transformer_encoder_without_cell_prop_keeps_shapes():
    encoder = GeneTransformerEncoder(
        args=_build_transformer_config(
            predict_cell_prop=False,
            using_positional_encoding=False,
        )
    )
    x = torch.randn(2, 8)
    y = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]], dtype=torch.float32)

    output = encoder(x, y=y)

    assert torch.equal(output["cell_prop"], y)
    assert "dd_alpha" not in output
    assert output["mu_all_types"].shape == (2, 4, 3)
    assert output["logvar_all_types"].shape == (2, 4, 3)
    assert output["bulk_context_feature"].shape == (2, 16)
    assert output["cell_type_context_features"].shape == (2, 3, 16)


def test_gene_transformer_encoder_with_positional_encoding_runs():
    config = _build_transformer_config(
        using_positional_encoding=True,
        latent_dim=3,
    )
    position_encoding = PositionalEncoding(
        d_model=config.latent_dim,
        max_len=config.n_cell_types,
    )
    encoder = GeneTransformerEncoder(args=config, position_encoding=position_encoding)
    x = torch.randn(2, 8)

    output = encoder(x)

    assert output["mu_all_types"].shape == (2, 3, 3)
    assert output["logvar_all_types"].shape == (2, 3, 3)
