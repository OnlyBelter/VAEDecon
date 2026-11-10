from pathlib import Path

import pandas as pd
import pytest
import torch

from vaedecon.configs import VAEDeconConfig
from vaedecon.configs.default_config import GEPDatasetConfig
from vaedecon.data import GEPDataset
import vaedecon.workflow.inference as inference_workflow
from vaedecon.workflow.train import (
    VAEDeconTrainer, _preflight_training_outputs, _preserve_existing_outputs,
    _should_reuse_staged_test_results, _split_staged_dataset, train_vaedecon,
)


def test_staged_split_is_reproducible_independently_of_global_rng():
    dataset = list(range(100))
    train, validation = _split_staged_dataset(dataset, 0.8, 0.2, 42)
    torch.rand(1000)
    train_again, validation_again = _split_staged_dataset(dataset, 0.8, 0.2, 42)
    other_train, _ = _split_staged_dataset(dataset, 0.8, 0.2, 43)

    assert train.indices == train_again.indices
    assert validation.indices == validation_again.indices
    assert train.indices != other_train.indices
    assert len(train) == 80
    assert len(validation) == 20


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


def test_staged_cli_reuses_existing_prediction_unless_training_is_skipped(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    test_results_dir = tmp_path / "final_model" / "test_results"

    assert not _should_reuse_staged_test_results(
        config,
        test_results_dir,
        skip_training=False,
    )

    test_results_dir.mkdir(parents=True)
    assert _should_reuse_staged_test_results(
        config,
        test_results_dir,
        skip_training=False,
    )
    assert not _should_reuse_staged_test_results(
        config,
        test_results_dir,
        skip_training=True,
    )

    config.training.staged_training.enabled = False
    assert not _should_reuse_staged_test_results(
        config,
        test_results_dir,
        skip_training=False,
    )


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


@pytest.mark.parametrize("selection", [None, [], ["cell_prop_predictor_pretrain", "reconstruction_training"]])
def test_staged_training_rejects_unused_checkpoint(tmp_path, selection):
    config = _base_staged_config_dict(tmp_path)
    stage = config["training"]["staged_training"]
    if selection is not None:
        stage["run_stages"] = selection
    stage["stage_init_checkpoints"] = {"reconstruction_training": str(tmp_path / "predictor.pt")}
    with pytest.raises(ValueError, match="Unused stage_init_checkpoints.*target stage"):
        VAEDeconConfig.from_dict(config)


def test_staged_training_rejects_checkpoint_for_unselected_target(tmp_path):
    config = _base_staged_config_dict(tmp_path)
    stage = config["training"]["staged_training"]
    stage["run_stages"] = ["cell_prop_predictor_pretrain"]
    stage["stage_init_checkpoints"] = {"reconstruction_training": str(tmp_path / "predictor.pt")}
    with pytest.raises(ValueError, match="Unused stage_init_checkpoints"):
        VAEDeconConfig.from_dict(config)


def _selected_reconstruction_config(tmp_path, checkpoint):
    config = _base_staged_config_dict(tmp_path)
    stage = config["training"]["staged_training"]
    stage["run_stages"] = ["reconstruction_training"]
    stage["stage_init_checkpoints"] = {"reconstruction_training": str(checkpoint)}
    return VAEDeconConfig.from_dict(config)


def test_missing_checkpoint_fails_before_reuse_or_output_creation(tmp_path):
    model_dir = tmp_path / "staged-training-test" / "final_model"
    model_dir.mkdir(parents=True)
    (model_dir / "best.ckpt").write_text("old")
    config = _selected_reconstruction_config(tmp_path, tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="reconstruction_training.*missing or unreadable"):
        train_vaedecon(config=config)
    assert (model_dir / "best.ckpt").read_text() == "old"
    assert list(model_dir.iterdir()) == [model_dir / "best.ckpt"]


def test_noninteractive_preflight_preserves_existing_outputs(tmp_path, monkeypatch):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    root = tmp_path / "staged-training-test"
    stage_dir = root / "stage_cell_prop_predictor_pretrain"
    stage_dir.mkdir(parents=True)
    (stage_dir / "model.pt").write_text("old")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(FileExistsError, match="Noninteractive"):
        train_vaedecon(config=config)
    assert (stage_dir / "model.pt").read_text() == "old"
    assert not (root / "final_model").exists()


def test_approved_outputs_are_preserved_in_backup(tmp_path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    root = tmp_path / "staged-training-test"
    stage = root / "stage_cell_prop_predictor_pretrain"
    stage.mkdir(parents=True)
    (stage / "model.pt").write_text("model")
    (root / "training_error.txt").write_text("error")
    _preflight_training_outputs(config, root / "final_model", overwrite=True)
    backups = list(root.glob("previous_outputs_*"))
    assert len(backups) == 1
    assert (backups[0] / stage.name / "model.pt").read_text() == "model"
    assert (backups[0] / "training_error.txt").read_text() == "error"
    assert not stage.exists()


def test_unselected_stage_checkpoint_is_preserved(tmp_path):
    root = tmp_path / "staged-training-test"
    checkpoint = root / "stage_cell_prop_predictor_pretrain" / "predictor.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    config = _selected_reconstruction_config(tmp_path, checkpoint)
    _preflight_training_outputs(config, root / "final_model")
    assert checkpoint.read_text() == "checkpoint"


def test_checkpoint_in_selected_output_rejected_even_when_authorized(tmp_path):
    root = tmp_path / "staged-training-test"
    checkpoint = root / "stage_reconstruction_training" / "predictor.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    config = _selected_reconstruction_config(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="overlaps a training output"):
        _preflight_training_outputs(config, root / "final_model", overwrite=True)
    assert checkpoint.read_text() == "checkpoint"


@pytest.mark.parametrize("answer", ["n", ""])
def test_refused_output_confirmation_preserves_files(tmp_path, monkeypatch, answer):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    root = tmp_path / "staged-training-test"
    final = root / "final_model"
    final.mkdir(parents=True)
    (final / "config.yaml").write_text("original")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    with pytest.raises(FileExistsError, match="cancelled"):
        _preflight_training_outputs(config, final)
    assert (final / "config.yaml").read_text() == "original"


def test_confirmation_rechecks_files_before_backup(tmp_path, monkeypatch):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    root = tmp_path / "staged-training-test"
    final = root / "final_model"
    final.mkdir(parents=True)
    (final / "old.txt").write_text("old")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def approve_after_change(prompt):
        (final / "new.txt").write_text("new")
        return "yes"

    monkeypatch.setattr("builtins.input", approve_after_change)
    with pytest.raises(FileExistsError, match="changed after confirmation"):
        _preflight_training_outputs(config, final)
    assert (final / "old.txt").read_text() == "old"
    assert (final / "new.txt").read_text() == "new"


def test_interactive_approval_preserves_outputs(tmp_path, monkeypatch):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    root = tmp_path / "staged-training-test"
    final = root / "final_model"
    final.mkdir(parents=True)
    (final / "model.pt").write_text("old")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    _preflight_training_outputs(config, final)
    backup = next(root.glob("previous_outputs_*"))
    assert (backup / "final_model" / "model.pt").read_text() == "old"


def test_confirmation_eof_preserves_outputs(tmp_path, monkeypatch):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    root = tmp_path / "staged-training-test"
    final = root / "final_model"
    final.mkdir(parents=True)
    (final / "model.pt").write_text("old")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def close_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", close_input)
    with pytest.raises(FileExistsError, match="cancelled"):
        _preflight_training_outputs(config, final)
    assert (final / "model.pt").read_text() == "old"


def test_symlinked_checkpoint_inside_selected_stage_is_protected(tmp_path):
    root = tmp_path / "staged-training-test"
    checkpoint = root / "stage_reconstruction_training" / "predictor.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    link = tmp_path / "checkpoint-link.pt"
    link.symlink_to(checkpoint)
    config = _selected_reconstruction_config(tmp_path, link)
    with pytest.raises(ValueError, match="overlaps a training output"):
        _preflight_training_outputs(config, root / "final_model", overwrite=True)
    assert checkpoint.read_text() == "checkpoint"


def test_direct_trainer_fails_before_preprocessing(tmp_path, monkeypatch):
    config = _selected_reconstruction_config(tmp_path, tmp_path / "missing.pt")
    trainer = VAEDeconTrainer(config)
    monkeypatch.setattr(trainer, "_prepare_data", lambda: pytest.fail("preprocessing ran"))
    with pytest.raises(ValueError, match="missing or unreadable"):
        trainer.train()
    assert not Path(trainer.model_dir).exists()


def test_final_promotion_rejects_new_files(tmp_path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config)
    trainer.model_dir.mkdir(parents=True)
    from vaedecon.workflow.train import _snapshot
    trainer._final_output_snapshot = _snapshot(trainer.model_dir)
    (trainer.model_dir / "unexpected.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="changed during training"):
        trainer._promote_stage_outputs_to_final_model(
            final_stage_dir=tmp_path,
            model=DummySaveModel(config.model),
            base_training_config=config.training,
            stage_summary_rows=[],
        )
    assert (trainer.model_dir / "unexpected.txt").read_text() == "keep"


@pytest.mark.parametrize("force", [False, True])
def test_incomplete_or_forced_cache_is_backed_up(tmp_path, force, monkeypatch):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    config.data.data_dir = tmp_path
    config.data.force_reprocess = force
    trainer = VAEDeconTrainer(config)
    file_paths, targets = trainer._resolve_training_dataset_inputs()
    cache = trainer._resolve_processed_training_set_dir(
        training_file_paths=file_paths, training_target_sets=targets
    )
    cache.mkdir(parents=True)
    (cache / "sentinel.txt").write_text("preserve")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(FileExistsError, match="Noninteractive"):
        _preflight_training_outputs(config, Path(trainer.model_dir))
    assert (cache / "sentinel.txt").read_text() == "preserve"
    _preflight_training_outputs(config, Path(trainer.model_dir), overwrite=True)
    backup = next(cache.parent.glob("previous_outputs_*"))
    assert (backup / cache.name / "sentinel.txt").read_text() == "preserve"


def test_cache_rebuild_guard_preserves_late_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "late.txt").write_text("preserve")
    config = GEPDatasetConfig(
        file_paths=[], processed_data_dir=cache, force_reprocess=True,
        protect_existing_cache=True,
    )
    with pytest.raises(FileExistsError, match="existing files"):
        GEPDataset(config)
    assert (cache / "late.txt").read_text() == "preserve"


@pytest.mark.parametrize("metadata,should_replace", [("{}", False), ("[]", True)])
def test_cache_preflight_distinguishes_reusable_metadata(tmp_path, metadata, should_replace):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    config.data.data_dir = tmp_path
    trainer = VAEDeconTrainer(config)
    file_paths, targets = trainer._resolve_training_dataset_inputs()
    cache = trainer._resolve_processed_training_set_dir(
        training_file_paths=file_paths, training_target_sets=targets
    )
    cache.mkdir(parents=True)
    for name in ("data.npy", "gene_list.txt", "sample_ids.txt"):
        (cache / name).write_text("data")
    (cache / "metadata.json").write_text(metadata)
    from vaedecon.workflow.train import _training_cache_to_replace
    assert (_training_cache_to_replace(config) == cache) is should_replace


def test_input_configuration_inside_output_is_protected(tmp_path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    final = tmp_path / "staged-training-test" / "final_model"
    final.mkdir(parents=True)
    source = final / "input.yaml"
    source.write_text("original")
    with pytest.raises(ValueError, match="overlaps a training output"):
        _preflight_training_outputs(config, final, overwrite=True, config_file=str(source))
    assert source.read_text() == "original"


def test_config_snapshot_does_not_overwrite_late_file(tmp_path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("preserve")
    from vaedecon.workflow.train import _save_config
    with pytest.raises(FileExistsError):
        _save_config(config, tmp_path, None)
    assert path.read_text() == "preserve"


def test_prediction_outputs_are_backed_up_or_preserved(tmp_path, monkeypatch):
    result = tmp_path / "test_results"
    result.mkdir()
    (result / "prediction.csv").write_text("old")
    used_config = tmp_path / "used_config.yaml"
    used_config.write_text("old config")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(FileExistsError, match="Noninteractive"):
        _preserve_existing_outputs([result, used_config])
    assert (result / "prediction.csv").read_text() == "old"
    _preserve_existing_outputs([result, used_config], overwrite=True)
    backup = next(tmp_path.glob("previous_outputs_*"))
    assert (backup / "test_results" / "prediction.csv").read_text() == "old"
    assert (backup / "used_config.yaml").read_text() == "old config"


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


def test_stage2_loss_plot_is_saved_from_losses_csv(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    stage_dir = tmp_path / "stage_reconstruction_training"
    stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_loss_epoch": [30.0, 20.0, 10.0],
            "val_loss": [32.0, 21.0, 11.0],
            "train_cell_type_sct_gep_loss_epoch": [0.02, 0.01, 0.005],
            "val_cell_type_sct_gep_loss": [0.03, 0.015, 0.006],
        }
    ).to_csv(stage_dir / "losses.csv", index=False)

    trainer._plot_stage_training_history(
        stage_name="reconstruction_training",
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


def test_stage_local_test_prediction_is_skipped_for_joint_finetune(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)

    assert trainer._should_run_stage_local_test_set_prediction("cell_prop_predictor_pretrain") is True
    assert trainer._should_run_stage_local_test_set_prediction("reconstruction_training") is True
    assert trainer._should_run_stage_local_test_set_prediction("joint_finetune") is False


def test_final_model_test_prediction_targets_final_model_test_results(
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
    final_stage_dir = tmp_path / "stage_joint_finetune"
    final_stage_dir.mkdir(parents=True, exist_ok=True)
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

    result = trainer._run_final_model_test_set_prediction(
        final_stage_name="joint_finetune",
        final_stage_dir=final_stage_dir,
    )

    assert result == trainer.model_dir / "test_results"
    assert captured["model_dir"] == str(trainer.model_dir)
    assert captured["device"] == trainer.device
    assert captured["output_dir"] == str(trainer.model_dir / "test_results")
    assert captured["dataset_type"] == "test"
    assert captured["visualize"] is True
    final_config = captured["config"]
    assert final_config.model.model_dir == trainer.model_dir


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


def test_promotion_preserves_existing_test_results(tmp_path: Path):
    config = VAEDeconConfig.from_dict(_base_staged_config_dict(tmp_path))
    trainer = VAEDeconTrainer(config=config)
    results = trainer.model_dir / "test_results"
    results.mkdir(parents=True)
    (results / "stale.txt").write_text("old")
    with pytest.raises(FileExistsError, match="prediction output"):
        trainer._promote_stage_outputs_to_final_model(
            final_stage_dir=tmp_path,
            model=DummySaveModel(config.model),
            base_training_config=config.training,
            stage_summary_rows=[],
        )
    assert (results / "stale.txt").read_text() == "old"
