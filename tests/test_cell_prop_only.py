import pytest
import torch
import anndata
import copy
import importlib
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

from vaedecon.configs import DataConfig, GEPDatasetConfig, ModelConfig, TrainingConfig, VAEDeconConfig
from vaedecon.models.cell_prop_only import CellPropOnlyModel
from vaedecon.models.vae.vae_model import VAE
from vaedecon.workflow.inference import VAEDeconPredictor, _StagedInferenceDataset
from vaedecon.workflow.bulk_prediction import (
    _predict_expression_batch, _save_proportion_metrics, postprocess_predictions,
)
from vaedecon.workflow.train import VAEDeconTrainer
from vaedecon.workflow.workflow import _maybe_compile_model
from vaedecon.trainers.base_trainer import BaseTrainerL, PLTrainer
from vaedecon.cli.easy_train import main as easy_train_main


@pytest.mark.parametrize("loss_type", ["mse", "l1_kl", "l1_rmse", "deside_mae_rmse"])
def test_predictor_only_uses_supervised_loss_without_vae(tmp_path, loss_type):
    cell_types = tmp_path / "cell_types.txt"
    cell_types.write_text("Immune\nCancer Cells\nStromal\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 3),
        n_cell_types=3,
        cell_type_fp=cell_types,
        predict_cell_prop=True,
        cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid",
        cancer_cell_type_name="Cancer Cells",
        cell_prop_loss_type=loss_type,
        deside_pathway_network=False,
        deside_hidden_dims=[4],
        deside_dropout_rate=[0.0],
        deside_normalization_layer=[0, 0],
        loss_coefficient={"cell_prop": 2.0},
    )
    model = CellPropOnlyModel(config, DataConfig())
    x = torch.tensor([[0.1, 0.3, 0.4], [0.6, 0.2, 0.1]])
    y = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.1, 0.5]])
    output = model({"data": x, "labels": y})
    prediction = model.cell_prop_predictor(x)
    raw = prediction["raw_non_cancer_cell_prop"]
    target = y[:, [0, 2]]
    expected = VAE._cell_prop_supervision_loss(model, raw, target).mean()
    torch.testing.assert_close(output.cell_prop_loss, expected)
    torch.testing.assert_close(output.loss, 2 * expected)
    output.loss.backward()
    assert any(p.grad is not None for p in model.cell_prop_predictor.parameters())
    assert not hasattr(model, "decoder")
    assert not hasattr(model, "encoders")


def test_predictor_postprocessing_thresholds_before_normalization(tmp_path):
    cell_types = tmp_path / "cell_types.txt"
    cell_types.write_text("Immune\nCancer Cells\nStromal\nOther\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 3), n_cell_types=4, cell_type_fp=cell_types,
        predict_cell_prop=True, cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        deside_pathway_network=False, deside_hidden_dims=[4],
        deside_dropout_rate=[0.0], deside_normalization_layer=[0, 0],
        deside_prediction_threshold=0.05,
    )
    model = CellPropOnlyModel(config, DataConfig())
    result = postprocess_predictions(np.array([[0.60, 0.42, 0.04]]), model)
    np.testing.assert_allclose(result, [[0.60 / 1.02, 0.0, 0.42 / 1.02, 0.0]])


def test_predictor_training_forward_excludes_sample_ids(tmp_path, monkeypatch):
    cells = tmp_path / "cells.txt"
    cells.write_text("CD4 T\nCancer Cells\nCAFs\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 2), n_cell_types=3, cell_type_fp=cells,
        predict_cell_prop=True, cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        deside_pathway_network=False, deside_hidden_dims=[4],
        deside_dropout_rate=[0.0], deside_normalization_layer=[0, 0],
    )
    model = CellPropOnlyModel(config, DataConfig())
    original = model.forward
    captured = {}

    def capture(inputs, **kwargs):
        captured.update(inputs)
        return original(inputs, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    trainer = PLTrainer(model, TrainingConfig(), optimize_stage_inputs=True)
    trainer({"data": torch.ones(1, 2), "sample_id": ["sample-1"]})
    assert set(captured) == {"data"}


def test_compile_policy_respects_cuda_and_configuration(monkeypatch):
    calls = []
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(torch, "compile", lambda value: calls.append(value) or value)
    config = ModelConfig(torch_compile=True)
    assert _maybe_compile_model(model, config, "cuda") is model
    assert calls == [model]
    calls.clear()
    assert _maybe_compile_model(model, config, "cpu") is model
    assert calls == []
    config.torch_compile = False
    assert _maybe_compile_model(model, config, "cuda") is model
    assert calls == []


def test_pathway_inference_normalizes_even_when_gene_sets_match(tmp_path, monkeypatch):
    cells = tmp_path / "cells.txt"
    cells.write_text("CD4 T\nCancer Cells\nCAFs\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 2), n_cell_types=3, cell_type_fp=cells,
        predict_cell_prop=True, cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        deside_pathway_network=False, deside_hidden_dims=[4],
        deside_dropout_rate=[0.0], deside_normalization_layer=[0, 0],
    )
    model = CellPropOnlyModel(config, DataConfig(scaling_by_constant=False))
    model.cell_prop_predictor.pathway_enabled = True
    captured = {}

    def forward(inputs):
        captured["data"] = inputs["data"]
        return {"raw_non_cancer_cell_prop": torch.tensor([[0.2, 0.3]])}

    monkeypatch.setattr(model, "forward", forward)
    _predict_expression_batch(
        model, np.log2([[3.0, 4.0]]), ["g1", "g2"], ["g1", "g2"], "cpu"
    )
    np.testing.assert_allclose(captured["data"].numpy()[0], np.log2([400001, 600001]), rtol=1e-6)


def test_proportion_metrics_exclude_nonfinite_and_preserve_truth_precision(tmp_path):
    truth = pd.DataFrame({"rare": [0.0004123456789, np.nan, 0.2]}, index=["a", "b", "c"])
    prediction = pd.DataFrame({"rare": [0.0004, 0.1, np.inf]}, index=truth.index)
    _save_proportion_metrics(prediction, truth, tmp_path)
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert metrics.loc[0, "n"] == 1
    assert metrics.loc[0, "rmse"] == pytest.approx(0.0000123456789)
    saved_truth = pd.read_csv(tmp_path / "true_cell_prop.csv", index_col=0)
    assert saved_truth.loc["a", "rare"] == pytest.approx(truth.loc["a", "rare"])


def test_complement_targets_are_supervised_and_returned_as_fractions(tmp_path):
    cell_types = tmp_path / "cell_types.txt"
    cell_types.write_text("CD4 T\nCancer Cells\nCAFs\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 3), n_cell_types=3, cell_type_fp=cell_types,
        predict_cell_prop=True, cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        deside_one_minus_alpha=True, deside_pathway_network=False,
        deside_hidden_dims=[4], deside_dropout_rate=[0.0], deside_normalization_layer=[0, 0],
        loss_coefficient={"cell_prop": 1.0},
    )
    model = CellPropOnlyModel(config, DataConfig())
    x = torch.tensor([[0.2, 0.5, 0.8]])
    y = torch.tensor([[0.2, 0.5, 0.3]])
    result = model({"data": x, "labels": y})
    raw = result.raw_non_cancer_cell_prop
    expected_loss = VAE._cell_prop_supervision_loss(model, raw, 1 - y[:, [0, 2]]).mean()
    torch.testing.assert_close(result.loss, expected_loss)
    expected = 1 - raw
    expected = expected / expected.sum(dim=1, keepdim=True).clamp_min(1)
    torch.testing.assert_close(result.pred_cell_prop[:, [0, 2]], expected)


def test_standalone_predictor_matches_sibling_deside(tmp_path):
    source = Path(__file__).resolve().parents[2] / "DeSide/deside/models/deside_mlp.py"
    if not source.is_file():
        pytest.skip("Sibling DeSide repository is unavailable")
    spec = importlib.util.spec_from_file_location("deside_test_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cell_types = tmp_path / "cell_types.txt"
    cell_types.write_text("CD4 T\nCancer Cells\nCAFs\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 3), n_cell_types=3, cell_type_fp=cell_types,
        predict_cell_prop=True, cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        cell_prop_loss_type="deside_mae_rmse", cell_prop_loss_alpha_weight=0.4,
        deside_pathway_network=False, deside_hidden_dims=[4],
        deside_dropout_rate=[0.0], deside_normalization_layer=[0, 0],
        loss_coefficient={"cell_prop": 1},
    )
    model = CellPropOnlyModel(config, DataConfig())
    reference = module.DeSideRegressor({
        "input_dim": 3, "output_dim": 2, "hidden_units": [4], "dropout_rates": [0.0],
        "normalization": "layer_normalization", "normalization_layer": [0, 0],
        "pathway_network": False, "last_layer_activation": "sigmoid",
    })
    reference.load_state_dict(model.cell_prop_predictor.state_dict(), strict=True)
    x = torch.tensor([[0.2, 0.5, 0.8], [0.1, 0.4, 0.6]])
    y = torch.tensor([[0.2, 0.4, 0.4], [0.3, 0.5, 0.2]])
    actual = model({"data": x, "labels": y})
    predicted = reference(x)
    target = y[:, [0, 2]]
    diff = predicted - target
    reference_loss = 0.4 * diff.abs().mean() + 0.6 * diff.square().mean().sqrt()
    torch.testing.assert_close(actual.raw_non_cancer_cell_prop, predicted, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.loss, reference_loss, rtol=1e-5, atol=1e-6)
    actual.loss.backward()
    reference_loss.backward()
    for actual_param, expected_param in zip(model.cell_prop_predictor.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-5, atol=1e-6)


def test_pathway_profile_matches_reference_and_optimizer_step(tmp_path):
    genes = tmp_path / "genes.txt"
    cells = tmp_path / "cells.txt"
    pathway = tmp_path / "pathways.gmt"
    genes.write_text("g1\ng2\ng3\n", encoding="utf-8")
    cells.write_text("CD4 T\nCancer Cells\nCAFs\n", encoding="utf-8")
    pathway.write_text("p1\tdesc\tg1\tg2\np2\tdesc\tg2\tg3\n", encoding="utf-8")
    config = ModelConfig(
        input_dim=(1, 3), n_cell_types=3, input_gene_list_fp=genes, cell_type_fp=cells,
        predict_cell_prop=True, cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        deside_pathway_network=True, deside_hidden_dims=[4], deside_pathway_hidden_dims=[3],
        deside_dropout_rate=[0.0], deside_pathway_dropout_rate=[0.0],
        deside_normalization_layer=[0, 0], cell_prop_loss_type="deside_mae_rmse",
        loss_coefficient={"cell_prop": 1},
    )
    model = CellPropOnlyModel(config, DataConfig(pathway_file_path=[pathway]))
    linear = torch.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]])
    inputs = torch.log2(linear + 1) / model.data_config.scaling_factor
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    expected_profile = torch.log2((linear / linear.sum(dim=1, keepdim=True) * 1e6) @ mask + 1)
    expected_profile /= model.data_config.scaling_factor
    torch.testing.assert_close(model.cell_prop_predictor.pathway_mask, mask)
    torch.testing.assert_close(
        model.cell_prop_predictor._compute_pathway_profile(inputs),
        expected_profile, rtol=1e-5, atol=1e-6,
    )

    source = Path(__file__).resolve().parents[2] / "DeSide/deside/models/deside_mlp.py"
    if not source.is_file():
        return
    spec = importlib.util.spec_from_file_location("deside_pathway_test_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.DeSideRegressor({
        "input_dim": 3, "output_dim": 2, "hidden_units": [4], "dropout_rates": [0.0],
        "normalization": "layer_normalization", "normalization_layer": [0, 0],
        "pathway_network": True, "n_pathway": 2, "pathway_hidden_units": [3],
        "pathway_dropout_rates": [0.0], "last_layer_activation": "sigmoid",
    })
    state = model.cell_prop_predictor.state_dict()
    reference.load_state_dict({key: value for key, value in state.items() if key != "pathway_mask"})
    labels = torch.tensor([[0.2, 0.5, 0.3], [0.3, 0.5, 0.2]])
    actual = model({"data": inputs, "labels": labels})
    expected = reference(inputs, expected_profile)
    error = expected - labels[:, [0, 2]]
    alpha = config.cell_prop_loss_alpha_weight
    loss = alpha * error.abs().mean() + (1 - alpha) * error.square().mean().sqrt()
    torch.testing.assert_close(actual.raw_non_cancer_cell_prop, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.loss, loss, rtol=1e-5, atol=1e-6)
    actual.loss.backward()
    loss.backward()
    optim_actual = torch.optim.Adam(model.cell_prop_predictor.parameters(), lr=0.001)
    optim_expected = torch.optim.Adam(reference.parameters(), lr=0.001)
    optim_actual.step()
    optim_expected.step()
    for actual_param, expected_param in zip(model.cell_prop_predictor.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_param, expected_param, rtol=1e-5, atol=1e-6)


def test_stage_one_trains_without_sct_inputs(tmp_path, monkeypatch):
    path = tmp_path / "bulk.h5ad"
    labels = pd.DataFrame(
        {"Immune": [0.3] * 10, "Cancer Cells": [0.4] * 10, "Stromal": [0.3] * 10},
        index=[f"sample_{i}" for i in range(10)],
    )
    anndata.AnnData(
        np.ones((10, 3), dtype=np.float32),
        obs=labels,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    ).write_h5ad(path)
    config = VAEDeconConfig.from_dict({
        "data": {
            "data_dir": str(tmp_path),
            "remove_low_var_genes": False,
            "training_target_sets": {"bulk": {"training_set_file_path": str(path)}},
        },
        "training": {
            "output_dir": str(tmp_path),
            "naming_postfix": "run",
            "device": "cpu",
            "batch_size": 4,
            "train_dataloader_num_workers": 0,
            "eval_dataloader_num_workers": 0,
            "staged_training": {
                "enabled": True,
                "run_stages": ["cell_prop_predictor_pretrain"],
                "stages": [{
                    "name": "cell_prop_predictor_pretrain",
                    "max_epochs": 1,
                    "train_modules": ["cell_prop_predictor"],
                    "loss_overrides": {"cell_prop": 1.0},
                }],
            },
        },
        "model": {
            "predict_cell_prop": True,
            "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "cell_prop_activation_function": "sigmoid",
            "cancer_cell_type_name": "Cancer Cells",
            "encoder_output_routing": {"cell_prop_source": "cell_prop_predictor"},
            "deside_pathway_network": False,
            "deside_hidden_dims": [4],
            "deside_dropout_rate": [0.0],
            "deside_normalization_layer": [0, 0],
        },
    })
    with_unused_sct = copy.deepcopy(config)
    with_unused_sct.training.naming_postfix = "unused_sct"
    with_unused_sct.data.training_target_sets["bulk"].training_sct_gep_file_path = str(tmp_path / "missing_sct.h5ad")
    with_unused_sct.data.training_target_sets["bulk"].training_set_sample2cell_id_file_path = str(tmp_path / "missing.csv")
    trainer = VAEDeconTrainer(config)
    monkeypatch.setattr(trainer, "_compute_gene_statistics", lambda *args: pytest.fail("SCT statistics accessed"))
    monkeypatch.setattr(trainer, "_create_model", lambda *args, **kwargs: pytest.fail("VAE constructed"))
    trainer.train()
    final = tmp_path / "run" / "final_model"
    assert (final / "model_kind.json").is_file()
    assert (final / "cell_prop_predictor_state_dict.pt").is_file()
    manifest = pd.read_csv(final / "split_manifest.csv")
    assert manifest.groupby("split").size().to_dict() == {"train": 8, "validation": 2}
    config_file = tmp_path / "bulk_config.yaml"
    with_unused_sct.to_yaml(config_file)
    assert easy_train_main(["-c", str(config_file)]) == 0
    other = tmp_path / "unused_sct" / "final_model"
    assert (other / "split_manifest.csv").read_text() == (final / "split_manifest.csv").read_text()
    best = torch.load(final / "cell_prop_predictor_state_dict.pt", weights_only=False)["state_dict"]
    other_best = torch.load(other / "cell_prop_predictor_state_dict.pt", weights_only=False)["state_dict"]
    for key in best:
        torch.testing.assert_close(best[key], other_best[key])
    predictor = VAEDeconPredictor(str(final), config=config, device="cpu")
    results = predictor.predict(str(path), str(tmp_path / "prediction"))
    predicted = pd.read_csv(results["pred_cell_prop_file_path"], index_col=0)
    assert predicted.columns.tolist() == ["Immune", "Cancer Cells", "Stromal"]
    assert predicted.shape == (10, 3)
    metrics = pd.read_csv(Path(results["pred_cell_prop_file_path"]).parent / "metrics.csv")
    assert metrics.cell_type.tolist() == predicted.columns.tolist()
    assert metrics["n"].tolist() == [10, 10, 10]
    csv_path = tmp_path / "bulk.csv"
    pd.DataFrame(np.ones((10, 3)), index=labels.index, columns=["g1", "g2", "g3"]).to_csv(csv_path)
    csv_results = predictor.predict(str(csv_path), str(tmp_path / "csv_prediction"))
    csv_predicted = pd.read_csv(csv_results["pred_cell_prop_file_path"], index_col=0)
    np.testing.assert_allclose(csv_predicted.to_numpy(), predicted.to_numpy(), rtol=1e-5)
    trainer._load_cell_prop_predictor_checkpoint(
        predictor.model, final / "cell_prop_predictor_state_dict.pt"
    )
    wrong_genes = tmp_path / "wrong_genes.txt"
    wrong_genes.write_text("g3\ng2\ng1\n", encoding="utf-8")
    predictor.model.model_config.input_gene_list_fp = wrong_genes
    with pytest.raises(ValueError, match="gene order"):
        trainer._load_cell_prop_predictor_checkpoint(
            predictor.model, final / "cell_prop_predictor_state_dict.pt"
        )


def test_stage_one_handoff_to_reconstruction(tmp_path, monkeypatch):
    genes = ["g1", "g2", "g3"]
    cells = ["CD4 T", "Cancer Cells", "CAFs"]
    bulk_path, sct_path = tmp_path / "bulk.h5ad", tmp_path / "sct.h5ad"
    bulk_labels = pd.DataFrame(
        np.tile([0.3, 0.4, 0.3], (10, 1)),
        index=[f"b{i}" for i in range(10)], columns=cells,
    )
    sct_labels = pd.DataFrame(
        np.repeat(np.eye(3), 2, axis=0),
        index=[f"s{i}" for i in range(6)], columns=cells,
    )
    anndata.AnnData(
        np.tile([2.0, 3.0, 4.0], (10, 1)).astype(np.float32),
        obs=bulk_labels, var=pd.DataFrame(index=genes),
    ).write_h5ad(bulk_path)
    anndata.AnnData(
        np.arange(18, dtype=np.float32).reshape(6, 3) / 10 + 1,
        obs=sct_labels, var=pd.DataFrame(index=genes),
    ).write_h5ad(sct_path)
    config = VAEDeconConfig.from_dict({
        "data": {
            "data_dir": str(tmp_path),
            "training_target_sets": {"bulk": {"training_set_file_path": str(bulk_path)}},
            "sct_file_path": [str(sct_path)],
            "remove_low_var_genes": False,
        },
        "training": {
            "output_dir": str(tmp_path), "naming_postfix": "handoff", "device": "cpu",
            "batch_size": 4, "train_dataloader_num_workers": 0, "eval_dataloader_num_workers": 0,
            "staged_training": {
                "enabled": True,
                "stages": [
                    {"name": "cell_prop_predictor_pretrain", "max_epochs": 1,
                     "train_modules": ["cell_prop_predictor"], "loss_overrides": {"cell_prop": 1}},
                    {"name": "reconstruction_training", "max_epochs": 1,
                     "train_modules": ["encoders", "decoder"], "freeze_modules": ["cell_prop_predictor"]},
                ],
            },
        },
        "model": {
            "predict_cell_prop": True, "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "cell_prop_activation_function": "sigmoid", "cancer_cell_type_name": "Cancer Cells",
            "encoders": ["EncoderMLP"], "encoder_aliases": ["main"],
            "encoder_output_routing": {
                "cell_prop_source": "cell_prop_predictor",
                "decoder_context_source": "cell_prop_predictor",
                "latent_posterior_source": "main",
            },
            "decoders": ["DecoderConditionalMLP"], "latent_dim": 2,
            "encoder_hidden_dims": [4], "encoder_dropout_rate": [0.0],
            "decoder_hidden_dims": [4], "decoder_dropout_rate": [0.0],
            "deside_pathway_network": False, "deside_hidden_dims": [4],
            "deside_dropout_rate": [0.0], "deside_normalization_layer": [0, 0],
            "conditional_decoder_cell_type_emb_dim": 2, "conditional_decoder_context_dim": 4,
            "conditional_decoder_dropout_rate": 0.0,
            "loss_coefficient": {"beta": 0, "gamma": 0, "cell_prop": 0,
                                 "gene_mean_weight": 0, "low_mean_std_weight": 0},
        },
    })
    resume_config = copy.deepcopy(config)
    failure_config = copy.deepcopy(config)
    observed_splits = {}
    original_init = BaseTrainerL.__init__
    from vaedecon.data.datasets import GEPPreprocessor

    def reject_bulk_reprocessing(*args, **kwargs):
        raise AssertionError("Reconstruction must reuse the staged bulk cache")

    monkeypatch.setattr(GEPPreprocessor, "run", reject_bulk_reprocessing)

    def capture_init(self, *args, **kwargs):
        if kwargs.get("stage_name") == "reconstruction_training":
            for label, subset in (("train", kwargs["train_dataset"]), ("validation", kwargs["eval_dataset"])):
                observed_splits[label] = {
                    subset.dataset.sample_ids[i] for i in subset.indices
                    if subset.dataset.sample_ids[i].startswith("bulk::")
                }
                for i in subset.indices:
                    if subset.dataset.sample_ids[i].startswith("bulk::"):
                        torch.testing.assert_close(
                            subset.dataset[i].data,
                            torch.tensor(np.log2(np.array([3, 7, 15]) / 25 * 1e6 + 1) / 20,
                                         dtype=torch.float32),
                        )
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(BaseTrainerL, "__init__", capture_init)
    VAEDeconTrainer(config).train()
    root = tmp_path / "handoff"
    summary = pd.read_csv(root / "staged_training_summary.csv")
    assert summary.stage_name.tolist() == ["cell_prop_predictor_pretrain", "reconstruction_training"]
    assert summary.init_source.iloc[1] == "previous_stage:cell_prop_predictor_pretrain"
    assert (root / "final_model" / "checkpoint_paths.json").is_file()
    assert (root / "final_model" / "input_contract.json").is_file()
    manifest = pd.read_csv(root / "stage_cell_prop_predictor_pretrain" / "split_manifest.csv")
    for split, sample_ids in observed_splits.items():
        expected = {f"{row.source}::{row.sample_id}" for row in manifest.itertuples() if row.split == split}
        assert sample_ids == expected
    assert set(observed_splits) == {"train", "validation"}
    checkpoint = root / "stage_cell_prop_predictor_pretrain" / "cell_prop_predictor_state_dict.pt"
    resume_config.training.naming_postfix = "resume"
    resume_config.training.staged_training = resume_config.training.staged_training.model_copy(update={
        "run_stages": ["reconstruction_training"],
        "stage_init_checkpoints": {"reconstruction_training": str(checkpoint)},
    })
    observed_splits.clear()
    VAEDeconTrainer(resume_config).train()
    for split, sample_ids in observed_splits.items():
        expected = {f"{row.source}::{row.sample_id}" for row in manifest.itertuples() if row.split == split}
        assert sample_ids == expected
    assert (tmp_path / "resume" / "final_model" / "checkpoint_paths.json").is_file()
    failure_config.training.naming_postfix = "later_failure"
    failing_trainer = VAEDeconTrainer(failure_config)

    def fail_reconstruction(**kwargs):
        raise FileNotFoundError("SCT reference missing")

    monkeypatch.setattr(failing_trainer, "_prepare_data", fail_reconstruction)
    with pytest.raises(RuntimeError, match="Stage-one predictor is saved"):
        failing_trainer.train()
    stage_dir = tmp_path / "later_failure" / "stage_cell_prop_predictor_pretrain"
    assert (stage_dir / "cell_prop_predictor_state_dict.pt").is_file()
    assert (stage_dir / "checkpoint_paths.json").is_file()


@pytest.mark.parametrize("scaling", [False, True])
def test_staged_full_inference_uses_saved_gene_order_and_excludes_zeros(tmp_path, scaling):
    genes = tmp_path / "genes.txt"
    genes.write_text("g2\ng_missing\ng1\n", encoding="utf-8")
    path = tmp_path / "test.h5ad"
    values = np.array([[1, 3, 100], [1, 0, 200], [0, 0, 300]], dtype=np.float32)
    anndata.AnnData(
        X=np.log2(values + 1),
        obs=pd.DataFrame({"cell": [0.1, 0.2, 0.3]}, index=["a", "b", "zero"]),
        var=pd.DataFrame(index=["g1", "g2", "extra"]),
    ).write_h5ad(path)
    config = GEPDatasetConfig(
        file_paths=[path], gene_list_file=genes, persist_processed_data=False, processed_data_dir=None,
        remove_low_var_genes=True, scaling_by_constant=scaling, scaling_factor=20,
        chunk_size=1,
    )
    dataset = _StagedInferenceDataset(config, tmp_path / "results")
    assert dataset.gene_list == ["g2", "g_missing", "g1"]
    assert dataset.sample_ids == ["a", "b"]
    expected = np.log2(np.array([[750000, 0, 250000], [0, 0, 1000000]]) + 1)
    if scaling:
        expected /= 20
    np.testing.assert_allclose(dataset.data, expected, rtol=1e-6)
    assert dataset.get_cell_prop().index.tolist() == ["a", "b"]
    excluded = pd.read_csv(tmp_path / "results" / "excluded_samples.csv")
    assert excluded["sample_id"].tolist() == ["zero"]
    assert excluded["reason"].tolist() == ["zero_selected_expression"]


def test_reconstruction_preflight_checks_stage_metadata_and_cache_identity(tmp_path):
    bulk = tmp_path / "bulk.h5ad"
    anndata.AnnData(
        np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame({"CD4 T": [0.2, 0.3], "Cancer Cells": [0.8, 0.7]},
                         index=["s1", "s2"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    ).write_h5ad(bulk)
    genes = tmp_path / "genes.txt"
    cells = tmp_path / "cells.txt"
    genes.write_text("g1\ng2\n", encoding="utf-8")
    cells.write_text("CD4 T\nCancer Cells\n", encoding="utf-8")
    config = VAEDeconConfig.from_dict({
        "data": {
            "data_dir": str(tmp_path), "remove_low_var_genes": True,
            "training_target_sets": {"bulk": {"training_set_file_path": str(bulk)}},
        },
        "training": {"output_dir": str(tmp_path), "device": "cpu",
                     "staged_training": {"enabled": True, "stages": [
                         {"name": "cell_prop_predictor_pretrain", "max_epochs": 1,
                          "train_modules": ["cell_prop_predictor"]},
                     ]}},
        "model": {
            "predict_cell_prop": True, "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "cell_prop_activation_function": "sigmoid", "cancer_cell_type_name": "Cancer Cells",
        },
    })
    trainer = VAEDeconTrainer(config)

    def prepare():
        return trainer._build_gepdataset_config(
            bulk_stage_transition=True, require_matched_targets=False,
            stage_gene_list_fp=genes, stage_cell_type_fp=cells, bulk_fingerprint="source-id",
        )

    first = prepare()
    assert first.gene_list_file == genes
    assert first.remove_low_var_genes is False
    assert config.data.remove_low_var_genes is True
    genes.write_text("g2\ng1\n", encoding="utf-8")
    assert prepare().processed_data_dir != first.processed_data_dir
    genes.write_text("g1\nmissing_gene\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing 1 stage-one genes: missing_gene"):
        prepare()
    genes.write_text("g1\ng2\n", encoding="utf-8")
    cells.write_text("CD4 T\nmissing_type\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing stage-one cell types: missing_type"):
        prepare()


@pytest.mark.parametrize("compiled", [False, True])
def test_three_stages_share_filtered_genes_labels_and_matched_targets(tmp_path, monkeypatch, compiled):
    genes = ["g1", "g2", "g3"]
    cells = ["CD4 T", "Cancer Cells", "CAFs"]
    bulk_path, sct_path, mapping_path = tmp_path / "bulk.h5ad", tmp_path / "sct.h5ad", tmp_path / "mapping.csv"
    sample_names = [f"b{i}" for i in range(12)] + ["bulk_zero"]
    labels = pd.DataFrame(
        np.tile([0.0004, 0.4996, 0.5], (13, 1)), index=sample_names, columns=cells
    )
    tpm = np.array([[100000 + 1000 * i, 899990 - 1000 * i, 10] for i in range(12)] + [[0, 0, 0]])
    anndata.AnnData(
        np.log2(tpm + 1).astype(np.float32), obs=labels, var=pd.DataFrame(index=genes)
    ).write_h5ad(bulk_path)
    sct_names = [f"s{i}" for i in range(6)]
    sct_labels = pd.DataFrame(
        np.repeat(np.eye(3), 2, axis=0), index=sct_names, columns=cells
    )
    sct_labels.loc["s4"] = [0.0004, 0.0, 0.9996]
    anndata.AnnData(
        np.arange(18, dtype=np.float32).reshape(6, 3) / 10 + 1,
        obs=sct_labels, var=pd.DataFrame(index=genes),
    ).write_h5ad(sct_path)
    pd.DataFrame(
        [(sample, cell_type, sct_names[index * 2])
         for sample in sample_names for index, cell_type in enumerate(cells)],
        columns=["sample", "cell_type", "selected_cell_id"],
    ).set_index("sample").to_csv(mapping_path)
    config = VAEDeconConfig.from_dict({
        "data": {
            "data_dir": str(tmp_path), "remove_low_var_genes": True, "min_var": 1,
            "sct_file_path": [str(sct_path)],
            "training_target_sets": {"bulk": {
                "training_set_file_path": str(bulk_path),
                "training_set_sample2cell_id_file_path": str(mapping_path),
                "training_sct_gep_file_path": str(sct_path),
            }},
            "test_sets": {"test": {"test_set_file_path": str(bulk_path)}},
        },
        "training": {
            "output_dir": str(tmp_path), "naming_postfix": "variance_handoff", "device": "cpu",
            "saved_model_selection": "last",
            "batch_size": 4, "train_dataloader_num_workers": 0, "eval_dataloader_num_workers": 0,
            "staged_training": {"enabled": True, "stages": [
                {"name": "cell_prop_predictor_pretrain", "max_epochs": 1,
                 "train_modules": ["cell_prop_predictor"], "loss_overrides": {"cell_prop": 1}},
                {"name": "reconstruction_training", "max_epochs": 1,
                 "train_modules": ["encoders", "decoder"], "freeze_modules": ["cell_prop_predictor"]},
                {"name": "joint_finetune", "max_epochs": 1,
                 "train_modules": ["cell_prop_predictor", "encoders", "decoder"]},
            ]},
        },
        "model": {
            "predict_cell_prop": True, "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "cell_prop_activation_function": "sigmoid", "cancer_cell_type_name": "Cancer Cells",
            "encoders": ["EncoderMLP"], "encoder_aliases": ["main"],
            "encoder_output_routing": {
                "cell_prop_source": "cell_prop_predictor", "decoder_context_source": "cell_prop_predictor",
                "latent_posterior_source": "main",
            },
            "decoders": ["DecoderConditionalMLP"], "latent_dim": 2,
            "encoder_hidden_dims": [4], "encoder_dropout_rate": [0.0],
            "decoder_hidden_dims": [4], "decoder_dropout_rate": [0.0],
            "deside_pathway_network": False, "deside_hidden_dims": [4],
            "deside_dropout_rate": [0.0], "deside_normalization_layer": [0, 0],
            "conditional_decoder_cell_type_emb_dim": 2, "conditional_decoder_context_dim": 4,
            "conditional_decoder_dropout_rate": 0.0,
            "loss_coefficient": {"beta": 0, "gamma": 0, "cell_prop": 0,
                                 "gene_mean_weight": 0, "low_mean_std_weight": 0,
                                 "cell_type_sct_gep_weight": 0.01},
        },
        "evaluation": {
            "plot_cell_proportions": False, "plot_single_cell_gep": False,
            "plot_bulk_gep": False, "plot_latent_space": False,
        },
    })
    if compiled:
        config.model.torch_compile = True
        train_module = importlib.import_module("vaedecon.workflow.train")
        monkeypatch.setattr(
            train_module, "_maybe_compile_model",
            lambda model, *_: torch.compile(model, backend="eager"),
        )
    resume_config = copy.deepcopy(config)
    seen = {}
    checkpoint_selections = []
    inference_module = importlib.import_module("vaedecon.workflow.inference")
    original_resolve = inference_module._resolve_trained_checkpoint_path

    def capture_resolve(*args, **kwargs):
        if Path(kwargs["model_dir"]).name == "stage_cell_prop_predictor_pretrain":
            checkpoint_selections.append(kwargs["training_config"].saved_model_selection)
        return original_resolve(*args, **kwargs)

    original_init = BaseTrainerL.__init__
    original_prepare = VAEDeconTrainer._prepare_data

    def capture_prepare(self, *args, **kwargs):
        stage_gene_list = kwargs.get("stage_gene_list_fp")
        if stage_gene_list is not None:
            stage_results = Path(stage_gene_list).parent / "test_results" / "bulk" / "predicted_cell_prop.csv"
            assert stage_results.is_file()
        return original_prepare(self, *args, **kwargs)

    def capture_init(self, *args, **kwargs):
        stage = kwargs.get("stage_name")
        subset = kwargs["train_dataset"]
        if stage == "cell_prop_predictor_pretrain":
            dataset = subset.dataset
            assert dataset.gene_list == ["g1", "g2"]
            assert dataset[subset.indices[0]].labels[0].item() == pytest.approx(0.0004)
        elif stage in {"reconstruction_training", "joint_finetune"}:
            dataset = subset.dataset
            assert dataset.full_dataset.gene_list == ["g1", "g2"]
            assert dataset.full_dataset.true_sct_gep.shape[1:] == (2, 3)
            sct_index = dataset.sample_ids.index("sct::s4")
            assert dataset[sct_index].labels[0].item() == pytest.approx(0.0004)
            if stage == "reconstruction_training":
                source = tmp_path / "variance_handoff" / "stage_cell_prop_predictor_pretrain" / "cell_prop_predictor_state_dict.pt"
                expected = torch.load(source, weights_only=False)["state_dict"]
                for key, value in kwargs["model"].cell_prop_predictor.state_dict().items():
                    torch.testing.assert_close(value, expected[key])
            for i in subset.indices:
                sample_id = dataset.sample_ids[i]
                if sample_id.startswith("bulk::"):
                    assert dataset[i].labels[0].item() == pytest.approx(0.0004)
                    bulk_index = dataset.bulk_indices[sample_id]
                    torch.testing.assert_close(dataset[i].data, dataset.bulk_dataset[bulk_index].data)
            seen[stage] = {
                label: {data.dataset.sample_ids[i] for i in data.indices if data.dataset.sample_ids[i].startswith("bulk::")}
                for label, data in (("train", kwargs["train_dataset"]), ("validation", kwargs["eval_dataset"]))
            }
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(BaseTrainerL, "__init__", capture_init)
    monkeypatch.setattr(VAEDeconTrainer, "_prepare_data", capture_prepare)
    monkeypatch.setattr(inference_module, "_resolve_trained_checkpoint_path", capture_resolve)
    monkeypatch.setattr(VAEDeconTrainer, "_should_run_stage_local_test_set_prediction", lambda *_: False)
    monkeypatch.setattr(VAEDeconTrainer, "_run_final_model_test_set_prediction", lambda *_, **__: None)
    VAEDeconTrainer(config).train()
    root = tmp_path / "variance_handoff"
    stage_one = root / "stage_cell_prop_predictor_pretrain"
    excluded = pd.read_csv(stage_one / "excluded_samples.csv")
    assert excluded["sample_id"].tolist() == ["bulk::bulk_zero"]
    assert excluded["reason"].tolist() == ["zero_common_expression"]
    assert (stage_one / "test_results" / "bulk" / "predicted_cell_prop.csv").is_file()
    inference_config = copy.deepcopy(config)
    inference_config.training.saved_model_selection = "best"
    predictor = VAEDeconPredictor(str(stage_one), config=inference_config, device="cpu")
    predicted = predictor.predict(str(bulk_path), str(tmp_path / "best_check"))
    stage_predictions = pd.read_csv(
        stage_one / "test_results" / "bulk" / "predicted_cell_prop.csv", index_col=0
    )
    best_predictions = pd.read_csv(predicted["pred_cell_prop_file_path"], index_col=0)
    pd.testing.assert_frame_equal(stage_predictions, best_predictions)
    inference_input = tmp_path / "constant_test.h5ad"
    anndata.AnnData(
        np.log2(np.array([[5, 5, 100], [5, 5, 200], [0, 0, 300]], dtype=np.float32) + 1),
        obs=pd.DataFrame(np.tile([0.2, 0.3, 0.5], (3, 1)),
                         index=["test_a", "test_b", "zero"], columns=cells),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(inference_input)
    inference_config.evaluation.remove_low_var_genes = True
    for model_dir, output_name in (
        (stage_one, "predictor"),
        (root / "stage_reconstruction_training", "reconstruction"),
        (root / "final_model", "full"),
    ):
        predictor = VAEDeconPredictor(str(model_dir), config=copy.deepcopy(inference_config), device="cpu")
        results = predictor.predict(str(inference_input), str(tmp_path / output_name))
        output = pd.read_csv(results["pred_cell_prop_file_path"], index_col=0)
        assert output.index.tolist() == ["test_a", "test_b"]
        exclusions = pd.read_csv(tmp_path / output_name / "constant_test" / "excluded_samples.csv")
        assert exclusions["sample_id"].tolist() == ["zero"]
    assert checkpoint_selections and set(checkpoint_selections) == {"best"}
    assert seen["reconstruction_training"] == seen["joint_finetune"]
    assert (root / "final_model" / "input_gene_list.txt").read_text() == (stage_one / "input_gene_list.txt").read_text()
    original_splits = copy.deepcopy(seen)
    resume_config.training.naming_postfix = "variance_resume"
    resume_config.training.staged_training = resume_config.training.staged_training.model_copy(update={
        "run_stages": ["reconstruction_training", "joint_finetune"],
        "stage_init_checkpoints": {
            "reconstruction_training": str(stage_one / "cell_prop_predictor_state_dict.pt")
        },
    })
    changed_source_config = copy.deepcopy(resume_config)
    changed_model_config = copy.deepcopy(resume_config)
    seen.clear()
    VAEDeconTrainer(resume_config).train()
    assert seen == original_splits
    changed_source_config.training.naming_postfix = "variance_changed_source"
    changed_model_config.training.naming_postfix = "variance_changed_model"
    changed_model_config.model.deside_hidden_dims = [5]
    monkeypatch.setattr(
        VAEDeconTrainer, "_prepare_data",
        lambda *_, **__: pytest.fail("changed source reached reconstruction preparation"),
    )
    with pytest.raises(ValueError, match="deside_hidden_dims"):
        VAEDeconTrainer(changed_model_config).train()
    bulk_path.touch()
    with pytest.raises(ValueError, match="preprocessing or source files have changed"):
        VAEDeconTrainer(changed_source_config).train()
