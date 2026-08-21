from pathlib import Path

import pandas as pd
import pytest
import torch

from vaedecon.configs import VAEDeconConfig
import vaedecon.workflow.inference as inference_workflow
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
        "data": {
            "sct_file_path": [str(tmp_path / "train_sct.h5ad")],
        },
        "training": {
            "output_dir": str(tmp_path),
            "naming_postfix": "staged-training-test",
            "device": "cpu",
            "batch_size": 8,
            "staged_training": _base_staged_training_dict(),
        },
        "model": {
            "predict_cell_prop": True,
            "cell_prop_activation_function": "sigmoid",
            "cancer_cell_type_name": "Cancer Cells",
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


class DummySaveModel(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

    def save(self, model_dir, training_config=None, data_config=None):
        Path(model_dir, "saved_marker.txt").write_text("saved", encoding="utf-8")


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


def test_stage1_training_config_always_logs_cell_prop_loss(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    config.training.prog_bar_metrics = ["loss", "kld"]
    trainer = VAEDeconTrainer(config=config)

    stage_training_config = trainer._build_stage_training_config(
        config.training,
        stage_cfg=config.training.staged_training.stages[0],
    )

    assert "cell_prop_loss" in stage_training_config.prog_bar_metrics


def test_stage1_loss_plot_is_saved_from_losses_csv(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    stage_dir = tmp_path / "stage_cell_prop_predictor_pretrain"
    stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_cell_prop_loss_epoch": [5.0, 2.5, 1.5],
            "val_cell_prop_loss": [6.0, 3.0, 2.0],
        }
    ).to_csv(stage_dir / "losses.csv", index=False)

    trainer._plot_stage_training_history(
        stage_name="cell_prop_predictor_pretrain",
        stage_dir=stage_dir,
    )

    assert (stage_dir / "loss.png").exists()


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


def test_post_stage_prediction_uses_cell_prop_focused_inference_for_stage1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["data"]["test_sets"] = {
        "Test_set1": {
            "test_set_file_path": str(tmp_path / "test_set.csv"),
        }
    }
    config = VAEDeconConfig.from_dict(config_dict)
    trainer = VAEDeconTrainer(config=config)
    stage_dir = tmp_path / "stage_cell_prop_predictor_pretrain"
    stage_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    class DummyPredictor:
        def __init__(self, model_dir, config, device):
            captured["model_dir"] = model_dir
            captured["config"] = config
            captured["device"] = device

        def predict_configured_test_sets(self, output_dir=None, dataset_type="test", visualize=True):
            captured["output_dir"] = output_dir
            captured["dataset_type"] = dataset_type
            captured["visualize"] = visualize
            return {}

    monkeypatch.setattr(inference_workflow, "VAEDeconPredictor", DummyPredictor)

    result = trainer._run_post_stage_test_set_prediction(
        stage_name="cell_prop_predictor_pretrain",
        stage_dir=stage_dir,
    )

    assert result == stage_dir / "test_results"
    assert captured["model_dir"] == str(stage_dir)
    assert captured["device"] == trainer.device
    assert captured["output_dir"] == str(stage_dir / "test_results")
    assert captured["dataset_type"] == "test"
    assert captured["visualize"] is True
    stage_config = captured["config"]
    assert stage_config.model.model_dir == stage_dir
    assert stage_config.evaluation.save_reconstructed_gep is False
    assert stage_config.evaluation.plot_single_cell_gep is False
    assert stage_config.evaluation.plot_bulk_gep is False
    assert stage_config.evaluation.plot_latent_space is False


@pytest.mark.parametrize("stage_name", ["reconstruction_training", "joint_finetune"])
def test_post_stage_prediction_uses_full_inference_for_later_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
):
    config_dict = _base_staged_config_dict(tmp_path)
    config_dict["data"]["test_sets"] = {
        "Test_set1": {
            "test_set_file_path": str(tmp_path / "test_set.csv"),
        }
    }
    config = VAEDeconConfig.from_dict(config_dict)
    trainer = VAEDeconTrainer(config=config)
    stage_dir = tmp_path / f"stage_{stage_name}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    class DummyPredictor:
        def __init__(self, model_dir, config, device):
            captured["model_dir"] = model_dir
            captured["config"] = config
            captured["device"] = device

        def predict_configured_test_sets(self, output_dir=None, dataset_type="test", visualize=True):
            captured["output_dir"] = output_dir
            captured["dataset_type"] = dataset_type
            captured["visualize"] = visualize
            return {}

    monkeypatch.setattr(inference_workflow, "VAEDeconPredictor", DummyPredictor)

    result = trainer._run_post_stage_test_set_prediction(
        stage_name=stage_name,
        stage_dir=stage_dir,
    )

    assert result == stage_dir / "test_results"
    assert captured["model_dir"] == str(stage_dir)
    assert captured["device"] == trainer.device
    assert captured["output_dir"] == str(stage_dir / "test_results")
    assert captured["dataset_type"] == "test"
    assert captured["visualize"] is True
    stage_config = captured["config"]
    assert stage_config.model.model_dir == stage_dir
    assert stage_config.evaluation.save_reconstructed_gep is True
    assert stage_config.evaluation.plot_single_cell_gep is True
    assert stage_config.evaluation.plot_bulk_gep is True
    assert stage_config.evaluation.plot_latent_space is True


def test_post_stage_prediction_skips_when_no_test_set_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    stage_dir = tmp_path / "stage_cell_prop_predictor_pretrain"
    stage_dir.mkdir(parents=True, exist_ok=True)

    def _unexpected_predictor(*args, **kwargs):
        raise AssertionError("VAEDeconPredictor should not be constructed without configured test sets")

    monkeypatch.setattr(inference_workflow, "VAEDeconPredictor", _unexpected_predictor)

    result = trainer._run_post_stage_test_set_prediction(
        stage_name="cell_prop_predictor_pretrain",
        stage_dir=stage_dir,
    )

    assert result is None


def test_promote_stage_outputs_skips_test_results_in_final_model(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    stage_dir = tmp_path / "stage_joint_finetune"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "losses.csv").write_text("epoch,val_loss\n0,1.0\n", encoding="utf-8")
    (stage_dir / "test_results").mkdir()
    (stage_dir / "test_results" / "stage_result.txt").write_text("stage3", encoding="utf-8")

    trainer.model_dir.mkdir(parents=True, exist_ok=True)
    (trainer.model_dir / "test_results").mkdir(exist_ok=True)
    (trainer.model_dir / "test_results" / "stale.txt").write_text("old", encoding="utf-8")

    model = DummySaveModel(model_config=config.model.model_copy(deep=True))
    trainer._promote_stage_outputs_to_final_model(
        final_stage_dir=stage_dir,
        model=model,
        base_training_config=config.training.model_copy(deep=True),
        stage_summary_rows=[
            {
                "stage_name": "joint_finetune",
                "stage_dir": str(stage_dir),
                "test_results_dir": str(stage_dir / "test_results"),
            }
        ],
    )

    assert (trainer.model_dir / "losses.csv").exists()
    assert not (trainer.model_dir / "test_results").exists()
    assert (trainer.model_dir / "saved_marker.txt").exists()
