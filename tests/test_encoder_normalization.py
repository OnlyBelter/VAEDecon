import importlib.resources as resources
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from vaedecon.configs import ModelConfig
from vaedecon.models.nn.mlp import EncoderMLP
from vaedecon.models.nn.res_mlp import EncoderResMLP


ENCODER_CLASSES = [EncoderMLP, EncoderResMLP]


def _model_config(hidden_dims=(5, 3), **overrides):
    config = {
        "input_dim": (1, 4),
        "latent_dim": 2,
        "n_cell_types": 3,
        "gnn_n_genes": 4,
        "encoder_hidden_dims": list(hidden_dims),
        "encoder_dropout_rate": [0.0] * len(hidden_dims),
    }
    config.update(overrides)
    return ModelConfig(**config)


def _stage_normalizations(encoder, encoder_cls, stage):
    if encoder_cls is EncoderMLP:
        return [encoder.layers[stage][1]]
    residual = encoder.layers[stage][1].block
    return [encoder.layers[stage][0][1], residual[1], residual[5]]


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
def test_omitted_normalization_mask_preserves_hidden_normalization(encoder_cls):
    config = _model_config()
    encoder = encoder_cls(config)

    assert config.encoder_normalization_layer == [0, 1, 1]
    assert isinstance(encoder.input_normalization, nn.Identity)
    for stage in range(len(config.encoder_hidden_dims)):
        assert all(isinstance(module, nn.LayerNorm) for module in _stage_normalizations(
            encoder, encoder_cls, stage
        ))


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
def test_input_normalization_runs_before_first_projection(encoder_cls):
    encoder = encoder_cls(
        _model_config(encoder_normalization_layer=[1, 0, 0])
    )
    observed_inputs = []
    first_linear = encoder.layers[0][0] if encoder_cls is EncoderMLP else encoder.layers[0][0][0]
    hook = first_linear.register_forward_pre_hook(
        lambda _module, inputs: observed_inputs.append(inputs[0].detach())
    )
    inputs = torch.tensor([[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 9.0]])

    encoder(inputs)
    hook.remove()

    expected = torch.nn.functional.layer_norm(
        inputs, normalized_shape=(inputs.shape[-1],), eps=encoder.input_normalization.eps
    )
    torch.testing.assert_close(observed_inputs[0], expected)


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
def test_each_hidden_mask_bit_controls_its_normalization_modules(encoder_cls):
    n_stages = 3
    for enabled_stage in range(n_stages):
        mask = [0] + [0] * n_stages
        mask[enabled_stage + 1] = 1
        encoder = encoder_cls(
            _model_config(
                hidden_dims=(5, 4, 3),
                encoder_normalization_layer=mask,
            )
        )

        for stage in range(n_stages):
            modules = _stage_normalizations(encoder, encoder_cls, stage)
            expected_type = nn.LayerNorm if stage == enabled_stage else nn.Identity
            assert all(isinstance(module, expected_type) for module in modules)
            if encoder_cls is EncoderResMLP:
                assert len(modules) == 3


@pytest.mark.parametrize(
    "mask",
    [
        [0, 2, 1],
        [0, -1, 1],
        [0, 1.0, 1],
        [0, True, 1],
        "011",
    ],
)
def test_invalid_normalization_mask_values_fail(mask):
    with pytest.raises(ValueError, match="encoder_normalization_layer"):
        _model_config(encoder_normalization_layer=mask)


@pytest.mark.parametrize("mask", [[0], [0, 1], [0, 1, 1, 0]])
def test_invalid_normalization_mask_length_fails(mask):
    with pytest.raises(
        ValueError,
        match=r"encoder_normalization_layer length must equal len\(encoder_hidden_dims\) \+ 1",
    ):
        _model_config(encoder_normalization_layer=mask)


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
def test_default_mask_preserves_legacy_state_dict_and_forward(encoder_cls):
    legacy_style = encoder_cls(_model_config())
    explicit_default = encoder_cls(
        _model_config(encoder_normalization_layer=[0, 1, 1])
    )
    explicit_default.load_state_dict(legacy_style.state_dict(), strict=True)

    keys = set(legacy_style.state_dict())
    if encoder_cls is EncoderMLP:
        expected_keys = {
            f"layers.{stage}.{layer}.{parameter}"
            for stage in range(2)
            for layer in (0, 1)
            for parameter in ("weight", "bias")
        } | {"fc_mu_logvar.weight", "fc_mu_logvar.bias"}
    else:
        expected_keys = {"fc_mu_logvar.weight", "fc_mu_logvar.bias"}
        for stage in range(2):
            expected_keys.update(
                f"layers.{stage}.0.{layer}.{parameter}"
                for layer in (0, 1)
                for parameter in ("weight", "bias")
            )
            expected_keys.update(
                f"layers.{stage}.1.block.{layer}.{parameter}"
                for layer in (0, 1, 4, 5)
                for parameter in ("weight", "bias")
            )
    assert keys == expected_keys

    inputs = torch.randn(3, 4)
    torch.testing.assert_close(
        legacy_style(inputs)["cell_prop_feature"],
        explicit_default(inputs)["cell_prop_feature"],
    )


def test_packaged_example_config_sets_preserving_default_mask():
    config_text = resources.files("vaedecon.configs").joinpath("example_config.yaml").read_text()
    parsed = yaml.safe_load(config_text)
    model_config = ModelConfig(**parsed["model"])

    assert model_config.encoder_normalization_layer == [0, 1, 1, 1, 1]


def test_deside_reproduction_six_stage_config_parses_preserving_default_mask():
    reproduction_config = (
        Path(__file__).resolve().parents[2]
        / "VAEDecon_example/config_2ds_n-neighbor_200K_reproduce_DeSide.yaml"
    )
    if reproduction_config.is_file():
        parsed = yaml.safe_load(reproduction_config.read_text(encoding="utf-8"))
        model_settings = parsed["model"]
    else:
        model_settings = {
            "input_dim": (1, 8),
            "gnn_n_genes": 8,
            "encoder_hidden_dims": [512, 2048, 2048, 2048, 2048, 512],
            "encoder_dropout_rate": [0, 0.1, 0.1, 0.1, 0.1, 0],
            "encoder_normalization_layer": [0, 1, 1, 1, 1, 1, 1],
        }
    model_config = ModelConfig(**model_settings)

    assert model_config.encoder_normalization_layer == [0, 1, 1, 1, 1, 1, 1]
