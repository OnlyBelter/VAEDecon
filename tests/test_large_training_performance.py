from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import lightning as L
from torch.utils.data import DataLoader, Subset

from vaedecon.configs import DataConfig, ModelConfig, TrainingConfig
from vaedecon.data.datasets import GEPDataset, collate_dataset_output
from vaedecon.models.vae.vae_model import VAE
from vaedecon.trainers.base_trainer import (
    BaseTrainerL,
    PLTrainer,
    _stage_dataset_view,
    _stage_sct_target_requirements,
    _TrainingProgressLogger,
)
from vaedecon.workflow.workflow import create_model


class _NoTargetReadsDataset(GEPDataset):
    """Fail if an optimized stage tries to read either matched target."""

    def __init__(self):
        self._n_samples = 4
        self._data = np.arange(12, dtype=np.float32).reshape(4, 3)
        self._labels = np.full((4, 2), 0.5, dtype=np.float32)
        self.sample_ids = [f"sample_{i}" for i in range(4)]

    @property
    def true_sct_gep(self):
        raise AssertionError("Matched expression targets must not be read")

    @property
    def true_sct_gep_present_mask(self):
        raise AssertionError("Matched target masks must not be read")


class _ProbeVAE(VAE):
    """Small tensor-only model exercising the staged trainer boundary."""

    def __init__(self, residual_weight=0.0):
        torch.nn.Module.__init__(self)
        self.model_name = "VAE"
        self.model_config = SimpleNamespace(
            loss_coefficient=SimpleNamespace(
                cell_type_sct_gep_weight=0.0,
                inter_sample_similarity_weight=0.0,
                per_sample_residual_var_weight=residual_weight,
            )
        )
        self.cell_types = ["CT1", "CT2"]
        self.training_sct_per_sample_residual_var_by_sample = {
            "set::one": np.array([1.0, 2.0], dtype=np.float32),
            "set::two": np.array([3.0, 4.0], dtype=np.float32),
            "set::three": np.array([5.0, 6.0], dtype=np.float32),
        }

    def forward(self, inputs, **kwargs):
        assert "sample_id" not in inputs
        result = inputs["data"].sum()
        target = inputs.get("per_sample_residual_var_targets")
        if target is not None:
            result = result + target.sum()
        return {"loss": result}


def _batch(sample_ids):
    n = len(sample_ids)
    return {
        "data": torch.ones((n, 3)),
        "labels": torch.full((n, 2), 0.5),
        "true_sct_gep_present_mask": torch.ones((n, 2), dtype=torch.bool),
        "sample_id": sample_ids,
    }


def _real_vae(tmp_path, residual_weight, *, dedicated_predictor=False):
    stats_path = tmp_path / "gene_stats.csv"
    pd.DataFrame(
        {
            "CD4 T_avg": [1.0, 1.0, 1.0],
            "Cancer Cells_avg": [1.0, 1.0, 1.0],
            "CD4 T_std": [0.2, 0.2, 0.2],
            "Cancer Cells_std": [0.2, 0.2, 0.2],
        },
        index=["g1", "g2", "g3"],
    ).to_csv(stats_path)
    target_path = tmp_path / "variance.csv"
    pd.DataFrame(
        {"CD4 T": [1.0, 2.0], "Cancer Cells": [2.0, 3.0]},
        index=["set::one", "set::two"],
    ).to_csv(target_path)
    predictor_settings = {}
    if dedicated_predictor:
        predictor_settings = {
            "encoder_aliases": ["mlp_main"],
            "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "deside_pathway_network": False,
            "deside_hidden_dims": [4],
            "deside_dropout_rate": [0.0],
            "deside_normalization_layer": [0, 0],
            "conditional_decoder_cell_type_emb_dim": 2,
            "conditional_decoder_context_dim": 4,
            "conditional_decoder_dropout_rate": 0.0,
            "encoder_output_routing": {
                "cell_prop_source": "cell_prop_predictor",
                "latent_posterior_source": "mlp_main",
                "decoder_context_source": "cell_prop_predictor",
            },
        }
    model_config = ModelConfig(
        input_dim=(1, 3),
        latent_dim=2,
        n_cell_types=2,
        encoders=["EncoderMLP"],
        encoder_hidden_dims=[4],
        encoder_dropout_rate=[0.0],
        decoder_hidden_dims=[4],
        decoder_dropout_rate=[0.0],
        predict_cell_prop=True,
        cell_prop_activation_function="sigmoid" if dedicated_predictor else "softmax",
        cancer_cell_type_name="Cancer Cells",
        gene_mean_std_fp=stats_path,
        training_sct_per_sample_residual_var_fp=target_path,
        learn_gep_residual=True,
        learn_gep_residual_mode="mean_centered",
        loss_coefficient={
            "beta": 0.0,
            "gamma": 0.0,
            "cell_prop": 10.0,
            "weighting_gene_by_exp": False,
            "gene_mean_weight": 0.0,
            "low_mean_std_weight": 0.0,
            "per_sample_residual_var_weight": residual_weight,
        },
        **predictor_settings,
    )
    return create_model(
        model_config, DataConfig(scaling_by_constant=False),
        ["EncoderMLP"],
        ["DecoderConditionalMLP" if dedicated_predictor else "DecoderMLP"],
        device="cpu",
    )


@pytest.mark.parametrize(
    ("residual_weight", "dedicated_predictor"),
    [(0.0, False), (1.0, False), (0.0, True), (1.0, True)],
)
def test_real_vae_stage_loss_and_gradients_match(tmp_path, residual_weight, dedicated_predictor):
    model = _real_vae(tmp_path, residual_weight, dedicated_predictor=dedicated_predictor)
    model.train()
    if dedicated_predictor:
        model.encoders.requires_grad_(False)
        model.decoder.requires_grad_(False)
    trainer = PLTrainer(model, TrainingConfig(), optimize_stage_inputs=True)
    batch = _batch(["set::one", "set::two"])
    batch["true_sct_gep"] = torch.ones((2, 3, 2))

    if dedicated_predictor:
        with torch.no_grad():
            model(batch)
    torch.manual_seed(17)
    baseline = model(batch)
    baseline.loss.backward()
    expected_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    optimized_batch = dict(batch)
    optimized_batch["true_sct_gep"] = torch.empty((2, 0))
    if residual_weight == 0:
        optimized_batch["true_sct_gep_present_mask"] = torch.empty((2, 0))
    torch.manual_seed(17)
    optimized = trainer(optimized_batch)
    optimized.loss.backward()

    torch.testing.assert_close(optimized.loss, baseline.loss)
    torch.testing.assert_close(optimized.cell_prop_loss, baseline.cell_prop_loss)
    torch.testing.assert_close(optimized.recon_loss_conv, baseline.recon_loss_conv)
    assert expected_grads
    if dedicated_predictor:
        assert any(name.startswith("cell_prop_predictor.") for name in expected_grads)
    for name, parameter in model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(parameter.grad, expected_grads[name])
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("workers", [0, 2])
def test_stage_view_skips_targets_for_nested_subsets_and_workers(workers):
    dataset = _NoTargetReadsDataset()
    subset = Subset(Subset(dataset, [3, 1, 2]), [1, 0])
    view = _stage_dataset_view(subset, include_gep=False, include_mask=False)
    loader = DataLoader(
        view, batch_size=2, num_workers=workers, collate_fn=collate_dataset_output
    )

    batch = next(iter(loader))
    assert batch["sample_id"] == ["sample_1", "sample_3"]
    assert batch["true_sct_gep"].numel() == 0
    assert batch["true_sct_gep_present_mask"].numel() == 0
    torch.testing.assert_close(batch["data"], torch.tensor([[3., 4., 5.], [9., 10., 11.]]))
    with pytest.raises(AssertionError, match="Matched expression"):
        dataset[0]


def test_stage_views_retain_required_expression_and_mask():
    dataset = GEPDataset.__new__(GEPDataset)
    dataset._n_samples = 2
    dataset._data = np.ones((2, 3), dtype=np.float32)
    dataset._labels = np.ones((2, 2), dtype=np.float32)
    dataset._true_sct_gep = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    dataset._true_sct_gep_present_mask = np.array([[True, False], [True, True]])
    dataset.sample_ids = ["set::one", "set::two"]
    subset = Subset(dataset, [1])

    mask_only = _stage_dataset_view(subset, include_gep=False, include_mask=True)[0]
    assert mask_only.true_sct_gep.numel() == 0
    assert mask_only.true_sct_gep_present_mask.tolist() == [True, True]
    full = _stage_dataset_view(subset, include_gep=True, include_mask=True)[0]
    torch.testing.assert_close(full.true_sct_gep, torch.tensor(dataset._true_sct_gep[1]))
    assert full.sample_id == "set::two"

    class _ShapeOnlyTarget:
        size = 12

        def __getitem__(self, index):
            raise AssertionError("Expression target must not be indexed")

    dataset._true_sct_gep = _ShapeOnlyTarget()
    mask_only = _stage_dataset_view(subset, include_gep=False, include_mask=True)[0]
    assert mask_only.true_sct_gep.numel() == 0
    assert mask_only.true_sct_gep_present_mask.tolist() == [True, True]


def test_stage_view_preserves_custom_dataset_and_subset_behavior():
    class CustomDataset(_NoTargetReadsDataset):
        def __getitem__(self, index):
            return {"sample_id": f"custom_{index}"}

    class CustomSubset(Subset):
        def __getitem__(self, index):
            return {"sample_id": "custom_subset"}

        def __getitems__(self, indices):
            return [self[index] for index in indices]

    custom_dataset = CustomDataset()
    custom_subset = CustomSubset(_NoTargetReadsDataset(), [0])
    for dataset in (custom_dataset, Subset(custom_dataset, [0]), custom_subset):
        assert _stage_dataset_view(dataset, False, False) is dataset
    assert _stage_dataset_view(custom_dataset, False, False)[0]["sample_id"] == "custom_0"
    assert _stage_dataset_view(custom_subset, False, False)[0]["sample_id"] == "custom_subset"


def test_stage_trainer_rebuilds_views_for_changing_requirements(tmp_path):
    model = _real_vae(tmp_path, residual_weight=0.0)
    dataset = _NoTargetReadsDataset()
    subset = Subset(dataset, [0, 1])
    config = TrainingConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        train_dataloader_num_workers=0,
        eval_dataloader_num_workers=0,
    )
    first = BaseTrainerL(
        model, str(tmp_path / "first"), subset, subset, config,
        stage_name="cell_prop_predictor_pretrain",
    )
    assert first.train_loader.dataset is not subset
    assert next(iter(first.train_loader))["true_sct_gep"].numel() == 0
    assert next(iter(first.eval_loader))["true_sct_gep"].numel() == 0

    model.model_config.loss_coefficient.cell_type_sct_gep_weight = 1.0
    second = BaseTrainerL(
        model, str(tmp_path / "second"), subset, subset, config,
        stage_name="reconstruction_training",
    )
    assert second.train_loader.dataset is subset
    assert second.eval_loader.dataset is subset
    assert first.train_loader.dataset is not second.train_loader.dataset


def test_stage_requirements_cover_active_and_scheduled_losses():
    model = _ProbeVAE()
    config = SimpleNamespace(aux_loss_schedules={}, adaptive_aux_loss_schedule=None)
    assert _stage_sct_target_requirements(model, config) == (False, False)
    model.model_config.loss_coefficient.per_sample_residual_var_weight = 1.0
    assert _stage_sct_target_requirements(model, config) == (False, True)
    model.model_config.loss_coefficient.inter_sample_similarity_weight = 1.0
    assert _stage_sct_target_requirements(model, config) == (True, True)
    model.model_config.loss_coefficient.inter_sample_similarity_weight = 0.0
    model.model_config.loss_coefficient.per_sample_residual_var_weight = 0.0
    config.aux_loss_schedules["cell_type_sct_gep_weight"] = object()
    assert _stage_sct_target_requirements(model, config) == (True, True)
    config.aux_loss_schedules = {}
    config.adaptive_aux_loss_schedule = SimpleNamespace(
        enabled=True, targets={"cell_type_sct_gep_weight": object()}
    )
    assert _stage_sct_target_requirements(model, config) == (True, True)
    assert _stage_sct_target_requirements(torch.nn.Linear(1, 1), config) == (True, True)


def test_residual_tensor_targets_match_ids_and_gradients():
    vae = _ProbeVAE(residual_weight=1.0)
    values = torch.randn((2, 4, 2), generator=torch.Generator().manual_seed(10))
    first = values.clone().requires_grad_()
    second = values.clone().requires_grad_()
    mask = torch.tensor([[True, True], [True, False]])
    labels = torch.ones((2, 2))
    ids = ["set::one", "set::two"]
    targets = vae._lookup_per_sample_residual_var_targets(
        ids, device=torch.device("cpu"), dtype=torch.float32
    )
    reference = vae._per_sample_residual_variance_loss(first, ids, mask, labels, 0.1)
    optimized = vae._per_sample_residual_variance_loss(
        second, None, mask, labels, 0.1, target_var=targets
    )
    reference.sum().backward()
    optimized.sum().backward()
    torch.testing.assert_close(optimized, reference)
    torch.testing.assert_close(second.grad, first.grad)
    with pytest.raises(ValueError, match="target shape"):
        vae._per_sample_residual_variance_loss(
            second, None, mask, labels, 0.1, target_var=targets[:1]
        )


def test_staged_inputs_preserve_ids_for_diagnostics_and_missing_id_behavior():
    model = _ProbeVAE(residual_weight=1.0)
    trainer = PLTrainer(model, TrainingConfig(), optimize_stage_inputs=True)
    batch = _batch(["set::two", "set::one"])
    assert trainer(batch)["loss"].item() == 16.0
    assert batch["sample_id"] == ["set::two", "set::one"]
    with pytest.raises(ValueError, match="Missing per-sample residual variance targets"):
        trainer(_batch(["set::missing"]))
    model.eval()
    assert torch.isnan(trainer(_batch(["set::missing"]))["loss"])


def test_real_vae_missing_ids_keep_training_and_validation_semantics(tmp_path):
    model = _real_vae(tmp_path, residual_weight=1.0)
    trainer = PLTrainer(model, TrainingConfig(), optimize_stage_inputs=True)
    batch = _batch(["set::missing", "set::one"])
    model.train()
    with pytest.raises(ValueError, match="Missing per-sample residual variance targets"):
        trainer(batch)
    model.eval()
    torch.manual_seed(71)
    expected = model(batch)
    torch.manual_seed(71)
    actual = trainer(batch)
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(
        actual.per_sample_residual_var_loss,
        expected.per_sample_residual_var_loss,
    )


def test_changing_sample_ids_do_not_recompile_tensor_only_stage():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(_ProbeVAE(residual_weight=1.0), backend=backend)
    trainer = PLTrainer(compiled, TrainingConfig(), optimize_stage_inputs=True)
    for sample_id in ["set::one", "set::two", "set::three"]:
        assert torch.isfinite(trainer(_batch([sample_id]))["loss"])
    assert len(graphs) == 1


def test_real_compiled_vae_matches_direct_inputs_for_changing_ids(tmp_path):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    model = _real_vae(tmp_path, residual_weight=1.0)
    model.train()
    trainer = PLTrainer(
        torch.compile(model, backend="eager"),
        TrainingConfig(),
        optimize_stage_inputs=True,
    )
    for sample_id in ["set::one", "set::two"]:
        batch = _batch([sample_id, sample_id])
        batch["true_sct_gep"] = torch.ones((2, 3, 2))
        torch.manual_seed(23)
        expected = model(batch)
        torch.manual_seed(23)
        actual = trainer(batch)
        torch.testing.assert_close(actual.loss, expected.loss)
        torch.testing.assert_close(
            actual.per_sample_residual_var_loss,
            expected.per_sample_residual_var_loss,
        )


def test_lightning_training_reports_progress_without_reading_targets(tmp_path, caplog, capsys):
    model = _real_vae(tmp_path, residual_weight=0.0)
    dataset = _NoTargetReadsDataset()
    train_view = _stage_dataset_view(Subset(dataset, [0, 1]), False, False)
    val_view = _stage_dataset_view(Subset(dataset, [2, 3]), False, False)
    loader_args = {"batch_size": 2, "collate_fn": collate_dataset_output}
    pl_model = PLTrainer(
        model,
        TrainingConfig(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
        ),
        optimize_stage_inputs=True,
    )
    lightning = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        callbacks=[_TrainingProgressLogger("cell_prop_predictor_pretrain", 0)],
    )
    with caplog.at_level("INFO", logger="vaedecon.trainers.base_trainer"):
        lightning.fit(
            pl_model,
            train_dataloaders=DataLoader(train_view, **loader_args),
            val_dataloaders=DataLoader(val_view, **loader_args),
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("train: 1/1 batches" in message for message in messages)
    assert any("validation: 1/1 batches" in message for message in messages)
    captured = capsys.readouterr()
    assert "Epoch 0" in captured.out + captured.err


def test_progress_reports_training_and_validation_and_is_rank_zero_only(caplog):
    progress = _TrainingProgressLogger("cell_prop_predictor_pretrain", interval_seconds=0)
    trainer = SimpleNamespace(
        is_global_zero=True, current_epoch=3, num_training_batches=4,
        num_val_batches=[2], sanity_checking=False,
    )
    with caplog.at_level("INFO", logger="vaedecon.trainers.base_trainer"):
        progress.on_train_epoch_start(trainer, None)
        progress.on_train_batch_end(trainer, None, None, None, 0)
        progress.on_validation_epoch_start(trainer, None)
        progress.on_validation_batch_end(trainer, None, None, None, 0)
        progress.on_validation_epoch_end(trainer, None)
        progress.on_train_epoch_end(trainer, None)
    messages = [record.getMessage() for record in caplog.records]
    assert any("epoch 3 train: 1/4 batches" in message for message in messages)
    assert any("epoch 3 validation: 1/2 batches" in message for message in messages)
    assert any("validation" in message and "(finished)" in message for message in messages)
    count = len(messages)
    trainer.is_global_zero = False
    progress.on_train_batch_end(trainer, None, None, None, 1)
    assert len(caplog.records) == count
