from pathlib import Path

import pytest
import torch

from vaedecon.configs import VAEDeconConfig
from vaedecon.workflow.train import VAEDeconTrainer


def _base_staged_training_dict() -> dict:
    return {
        "enabled": True,
        "stages": [
            {
                "name": "cell_prop_predictor_pretrain",
                "max_epochs": 5,
                "train_modules": ["cell_prop_predictor"],
                "freeze_modules": ["encoders", "decoder"],
                "learning_rate_scale": 1.0,
                "loss_overrides": {
                    "cell_prop": 100.0,
                    "cell_type_sct_gep_weight": 0.0,
                },
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 2,
                    "min_delta": 0.0,
                },
            },
            {
                "name": "reconstruction_training",
                "max_epochs": 5,
                "train_modules": ["encoders", "decoder"],
                "freeze_modules": ["cell_prop_predictor"],
                "learning_rate_scale": 1.0,
                "loss_overrides": {
                    "cell_prop": 0.0,
                },
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 2,
                    "min_delta": 0.0,
                },
            },
            {
                "name": "joint_finetune",
                "max_epochs": 5,
                "train_modules": ["cell_prop_predictor", "encoders", "decoder"],
                "freeze_modules": [],
                "learning_rate_scale": 0.1,
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 2,
                    "min_delta": 0.0,
                },
            },
        ],
    }


def _base_staged_config_dict(tmp_path: Path) -> dict:
    predictor_alias = "cell_prop_predictor"
    return {
        "training": {
            "output_dir": str(tmp_path),
            "naming_postfix": "staged-training-test",
            "device": "cpu",
            "batch_size": 8,
            "staged_training": _base_staged_training_dict(),
        },
        "model": {
            "predict_cell_prop": True,
            "encoders": ["EncoderMLP"],
            "encoder_aliases": ["mlp_main"],
            "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "cell_prop_predictor_alias": predictor_alias,
            "encoder_output_routing": {
                "cell_prop_source": predictor_alias,
                "latent_posterior_source": "mlp_main",
                "decoder_context_source": predictor_alias,
            },
        },
    }


class DummyStageModel(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoders = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        self.decoder = torch.nn.Linear(4, 4)
        self.cell_prop_predictor = torch.nn.Linear(4, 2)
        self.model_config = model_config


def test_staged_training_config_accepts_selected_stage_with_init_checkpoint(tmp_path: Path):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["training"]["staged_training"]["run_stages"] = ["reconstruction_training"]
    config_dict["training"]["staged_training"]["stage_init_checkpoints"] = {
        "reconstruction_training": str(tmp_path / "predictor.ckpt"),
    }

    loaded = VAEDeconConfig.from_dict(config_dict)

    assert loaded.training.staged_training is not None
    assert loaded.training.staged_training.run_stages == ["reconstruction_training"]


def test_staged_training_rejects_missing_init_checkpoint_for_skipped_predecessor(tmp_path: Path):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["training"]["staged_training"]["run_stages"] = ["reconstruction_training"]

    with pytest.raises(ValueError, match="requires an init checkpoint"):
        VAEDeconConfig.from_dict(config_dict)


def test_staged_training_rejects_non_contiguous_stage_order(tmp_path: Path):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["training"]["staged_training"]["run_stages"] = [
        "cell_prop_predictor_pretrain",
        "joint_finetune",
    ]
    config_dict["training"]["staged_training"]["stage_init_checkpoints"] = {
        "joint_finetune": str(tmp_path / "stage2.ckpt"),
    }

    with pytest.raises(ValueError, match="canonical contiguous stage order"):
        VAEDeconConfig.from_dict(config_dict)


def test_staged_training_requires_predictor_branch(tmp_path: Path):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["model"].pop("cell_prop_predictor_cls")
    config_dict["model"]["encoder_output_routing"]["cell_prop_source"] = "fused"
    config_dict["model"]["encoder_output_routing"]["decoder_context_source"] = "fused"

    with pytest.raises(ValueError, match="requires model.cell_prop_predictor_cls"):
        VAEDeconConfig.from_dict(config_dict)


def test_staged_training_reconstruction_requires_predictor_routing(tmp_path: Path):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["training"]["staged_training"]["run_stages"] = ["reconstruction_training"]
    config_dict["training"]["staged_training"]["stage_init_checkpoints"] = {
        "reconstruction_training": str(tmp_path / "predictor.ckpt"),
    }
    config_dict["model"]["encoder_output_routing"]["decoder_context_source"] = "fused"

    with pytest.raises(ValueError, match="decoder_context_source"):
        VAEDeconConfig.from_dict(config_dict)


def test_apply_stage_module_trainability_freezes_expected_modules(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    dummy_model = DummyStageModel(model_config=config.model)

    mode_overrides = trainer._apply_stage_module_trainability(
        dummy_model,
        train_modules=["cell_prop_predictor"],
    )

    assert mode_overrides == {
        "encoders": "eval",
        "decoder": "eval",
        "cell_prop_predictor": "train",
    }
    assert all(not parameter.requires_grad for parameter in dummy_model.encoders.parameters())
    assert all(not parameter.requires_grad for parameter in dummy_model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in dummy_model.cell_prop_predictor.parameters())


def test_stage_loss_overrides_reset_and_apply(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    dummy_model = DummyStageModel(model_config=config.model.model_copy(deep=True))
    base_model_config = config.model.model_copy(deep=True)

    trainer._apply_stage_loss_overrides(
        dummy_model,
        {
            "cell_prop": 0.0,
            "hierarchical_code_weight": 3.0,
            "cell_type_existence_shift_scale": 0.25,
        },
    )
    assert dummy_model.model_config.loss_coefficient.cell_prop == pytest.approx(0.0)
    assert dummy_model.model_config.loss_coefficient.hierarchical_code_weight == pytest.approx(3.0)
    assert dummy_model.model_config.cell_type_existence_shift_scale == pytest.approx(0.25)

    trainer._reset_stage_loss_overrides(dummy_model, base_model_config)
    assert (
        dummy_model.model_config.loss_coefficient.cell_prop
        == pytest.approx(base_model_config.loss_coefficient.cell_prop)
    )
    assert (
        dummy_model.model_config.loss_coefficient.hierarchical_code_weight
        == pytest.approx(base_model_config.loss_coefficient.hierarchical_code_weight)
    )
    assert (
        dummy_model.model_config.cell_type_existence_shift_scale
        == pytest.approx(base_model_config.cell_type_existence_shift_scale)
    )


def test_stage1_monitor_is_forced_to_val_cell_prop_loss(tmp_path: Path):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["training"]["staged_training"]["stages"][0]["early_stopping"]["monitor"] = "val_loss"

    loaded = VAEDeconConfig.from_dict(config_dict)

    assert loaded.training.staged_training is not None
    assert loaded.training.staged_training.stages[0].early_stopping.monitor == "val_cell_prop_loss"


def test_load_cell_prop_predictor_checkpoint_updates_only_predictor_weights(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    dummy_model = DummyStageModel(model_config=config.model)

    original_encoder_weight = dummy_model.encoders[0].weight.detach().clone()
    predictor_state = {
        "state_dict": {
            "model.cell_prop_predictor.weight": torch.full_like(dummy_model.cell_prop_predictor.weight, 2.0),
            "model.cell_prop_predictor.bias": torch.full_like(dummy_model.cell_prop_predictor.bias, -1.0),
        }
    }
    checkpoint_path = tmp_path / "predictor_only.ckpt"
    torch.save(predictor_state, checkpoint_path)

    trainer._load_cell_prop_predictor_checkpoint(dummy_model, checkpoint_path)

    assert torch.allclose(dummy_model.cell_prop_predictor.weight, torch.full_like(dummy_model.cell_prop_predictor.weight, 2.0))
    assert torch.allclose(dummy_model.cell_prop_predictor.bias, torch.full_like(dummy_model.cell_prop_predictor.bias, -1.0))
    assert torch.allclose(dummy_model.encoders[0].weight, original_encoder_weight)
