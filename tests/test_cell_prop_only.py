import pytest
import torch
import anndata
import copy
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

from vaedecon.configs import DataConfig, ModelConfig, VAEDeconConfig
from vaedecon.models.cell_prop_only import CellPropOnlyModel
from vaedecon.models.vae.vae_model import VAE
from vaedecon.workflow.inference import VAEDeconPredictor
from vaedecon.workflow.bulk_prediction import (
    _predict_expression_batch, _save_proportion_metrics, postprocess_predictions,
)
from vaedecon.workflow.train import VAEDeconTrainer
from vaedecon.trainers.base_trainer import BaseTrainerL
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
                            subset.dataset[i].data, torch.tensor([0.1, 0.15, 0.2])
                        )
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(BaseTrainerL, "__init__", capture_init)
    VAEDeconTrainer(config).train()
    root = tmp_path / "handoff"
    summary = pd.read_csv(root / "staged_training_summary.csv")
    assert summary.stage_name.tolist() == ["cell_prop_predictor_pretrain", "reconstruction_training"]
    assert summary.init_source.iloc[1] == "previous_stage:cell_prop_predictor_pretrain"
    assert (root / "final_model" / "checkpoint_paths.json").is_file()
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
