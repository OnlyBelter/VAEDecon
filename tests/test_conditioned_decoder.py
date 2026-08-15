import pytest
import torch

from vaedecon.configs import ModelConfig
from vaedecon.models.nn import DecoderConditionalMLP, DecoderMLP


def _build_model_config(**overrides) -> ModelConfig:
    base = {
        "input_dim": (1, 8),
        "latent_dim": 4,
        "n_cell_types": 3,
        "decoder_hidden_dims": [6, 5],
        "decoder_dropout_rate": [0.0, 0.0],
        "conditional_decoder_cell_type_emb_dim": 3,
        "conditional_decoder_context_dim": 7,
        "conditional_decoder_dropout_rate": 0.0,
    }
    base.update(overrides)
    return ModelConfig(**base)


def test_decoder_conditional_mlp_returns_expected_shape():
    decoder = DecoderConditionalMLP(args=_build_model_config())
    z = torch.randn(6, 4)
    cell_type_indices = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    bulk_context = torch.randn(6, 7)

    output = decoder(
        z,
        cell_type_indices=cell_type_indices,
        bulk_context=bulk_context,
    )

    assert output["reconstruction"].shape == (6, 8)


def test_decoder_conditional_mlp_requires_conditioning_inputs():
    decoder = DecoderConditionalMLP(args=_build_model_config())
    z = torch.randn(6, 4)

    with pytest.raises(ValueError, match="requires both cell_type_indices and bulk_context"):
        decoder(z, cell_type_indices=None, bulk_context=None)


def test_decoder_mlp_shape_is_unchanged():
    decoder = DecoderMLP(args=_build_model_config())
    z = torch.randn(2, 4, 3)

    output = decoder(z)

    assert output["reconstruction"].shape == (2, 8, 3)
