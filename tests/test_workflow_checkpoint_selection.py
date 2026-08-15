import json
from pathlib import Path

import pytest

from vaedecon.configs import TrainingConfig
from vaedecon.workflow.workflow import _apply_training_config_override
from vaedecon.workflow.workflow import _resolve_trained_checkpoint_path


def test_resolve_trained_checkpoint_path_returns_single_checkpoint(tmp_path: Path):
    only_ckpt = tmp_path / "model.ckpt"
    only_ckpt.write_bytes(b"checkpoint")

    resolved = _resolve_trained_checkpoint_path(
        model_dir=tmp_path,
        training_config=TrainingConfig(),
    )

    assert resolved == only_ckpt


def test_apply_training_config_override_updates_saved_model_selection():
    saved = TrainingConfig(saved_model_selection="best")
    override = TrainingConfig(saved_model_selection="last")

    resolved = _apply_training_config_override(
        saved_training_config=saved,
        training_config_override=override,
    )

    assert resolved.saved_model_selection == "last"


def test_resolve_trained_checkpoint_path_uses_last_checkpoint_selection(tmp_path: Path):
    (tmp_path / "best_model_epoch=3.ckpt").write_bytes(b"best")
    last_ckpt = tmp_path / "last_model.ckpt"
    last_ckpt.write_bytes(b"last")

    resolved = _resolve_trained_checkpoint_path(
        model_dir=tmp_path,
        training_config=TrainingConfig(saved_model_selection="last"),
    )

    assert resolved == last_ckpt


def test_resolve_trained_checkpoint_path_rejects_best_only_checkpoint_for_last_selection(tmp_path: Path):
    best_ckpt = tmp_path / "best_model_epoch=33.ckpt"
    best_ckpt.write_bytes(b"best")

    with pytest.raises(FileNotFoundError, match="saved_model_selection='last'"):
        _resolve_trained_checkpoint_path(
            model_dir=tmp_path,
            training_config=TrainingConfig(saved_model_selection="last"),
        )


def test_resolve_trained_checkpoint_path_uses_metadata_for_best_checkpoint(tmp_path: Path):
    best_ckpt = tmp_path / "best_model_epoch=7.ckpt"
    best_ckpt.write_bytes(b"best")
    (tmp_path / "last_model.ckpt").write_bytes(b"last")
    (tmp_path / "checkpoint_paths.json").write_text(
        json.dumps(
            {
                "best_model_path": str(best_ckpt),
                "last_model_path": str(tmp_path / "last_model.ckpt"),
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_trained_checkpoint_path(
        model_dir=tmp_path,
        training_config=TrainingConfig(saved_model_selection="best"),
    )

    assert resolved == best_ckpt


def test_resolve_trained_checkpoint_path_raises_for_ambiguous_best_checkpoint(tmp_path: Path):
    (tmp_path / "epoch=1.ckpt").write_bytes(b"a")
    (tmp_path / "epoch=2.ckpt").write_bytes(b"b")

    with pytest.raises(FileNotFoundError, match="Could not resolve the requested best checkpoint"):
        _resolve_trained_checkpoint_path(
            model_dir=tmp_path,
            training_config=TrainingConfig(saved_model_selection="best"),
        )
