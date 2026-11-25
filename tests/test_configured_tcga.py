import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from vaedecon.configs import DataConfig, GEPDatasetConfig, ModelConfig, VAEDeconConfig
from vaedecon.configs.default_config import TCGAComparisonConfig, TestSetConfig as ConfiguredTestSet
from vaedecon.models.cell_prop_only import CellPropOnlyModel
from vaedecon.workflow import tcga
from vaedecon.workflow.bulk_prediction import prepare_linear_tpm
from vaedecon.workflow.inference import VAEDeconPredictor, _StagedInferenceDataset


def _dataset_config(tmp_path, source):
    genes = tmp_path / "genes.txt"
    genes.write_text("g2\nmissing\ng1\n", encoding="utf-8")
    return GEPDatasetConfig(
        file_paths=[source], gene_list_file=genes, persist_processed_data=False,
        processed_data_dir=None, scaling_by_constant=True, scaling_factor=20,
        chunk_size=1,
    )


def _predictor(tmp_path, monkeypatch, *, full=False, staged=True):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    genes = model_dir / "input_gene_list.txt"
    genes.write_text("g2\nmissing\ng1\n", encoding="utf-8")
    cells = model_dir / "cell_type_list.txt"
    cells.write_text("Immune\nCancer Cells\n", encoding="utf-8")
    if staged:
        (model_dir / "input_contract.json").write_text(
            '{"input_contract": "staged_tpm_v1"}', encoding="utf-8"
        )
    model_config = ModelConfig(
        input_dim=(1, 3), n_cell_types=2, cell_type_fp=cells,
        input_gene_list_fp=genes, predict_cell_prop=True,
        cell_prop_predictor_cls="DeSideCellPropPredictor",
        cell_prop_activation_function="sigmoid", cancer_cell_type_name="Cancer Cells",
        deside_pathway_network=False, deside_hidden_dims=[4],
        deside_dropout_rate=[0.0], deside_normalization_layer=[0, 0],
    )
    data_config = DataConfig.model_validate(
        {"scaling_by_constant": True, "scaling_factor": 20},
        context={"defer_sct_requirements": True},
    )
    predictor = VAEDeconPredictor.__new__(VAEDeconPredictor)
    predictor.model_dir = str(model_dir)
    predictor.config = VAEDeconConfig(model=model_config, data=data_config)
    predictor.config.evaluation.val_batch_size = 1
    predictor.device = "cpu"
    predictor.model = (
        SimpleNamespace(data_config=data_config) if full
        else CellPropOnlyModel(model_config, data_config).eval()
    )
    monkeypatch.setattr(predictor, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(tcga, "_resolve_trained_checkpoint_path", lambda *_: model_dir / "best.ckpt")
    return predictor


def test_tcga_config_is_explicit_and_rejects_simulated_references():
    assert ConfiguredTestSet(test_set_file_path="test.csv").dataset_type == "test"
    with pytest.raises(ValueError, match="linear TPM CSV"):
        ConfiguredTestSet(dataset_type="tcga", test_set_file_path="test.h5ad")
    with pytest.raises(ValueError, match="cannot specify SCT"):
        ConfiguredTestSet(dataset_type="tcga", test_set_file_path="test.csv", sct_gep_file_path="ref.h5ad")


def test_tcga_configuration_round_trips_through_yaml(tmp_path):
    config = VAEDeconConfig.from_dict({
        "data": {
            "sct_file_path": ["sct.h5ad"],
            "test_sets": {
                "TCGA": {
                    "dataset_type": "tcga",
                    "test_set_file_path": "tcga.csv",
                    "tcga_comparisons": {"cancer_purity_file_path": "cpe.csv"},
                }
            },
        }
    })
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    restored = VAEDeconConfig.from_yaml(path).data.test_sets["TCGA"]
    assert restored.dataset_type == "tcga"
    assert restored.tcga_comparisons.cancer_purity_file_path == "cpe.csv"


def test_linear_tpm_normalization_matches_standalone_contract():
    values = np.array([[1, 3, 200], [0, 0, 1]], dtype=np.float32)
    aligned, valid = prepare_linear_tpm(values, ["g1", "g2", "extra"], ["g2", "missing", "g1"], 20)
    np.testing.assert_allclose(aligned[0], np.log2([750001, 1, 250001]) / 20, rtol=1e-6)
    assert valid.tolist() == [True, False]
    rounded, _ = prepare_linear_tpm(values, ["g1", "g2", "extra"], ["g2", "missing", "g1"], 20, True)
    np.testing.assert_allclose(rounded[0], np.round(np.log2([750001, 1, 250001]), 3) / 20, rtol=1e-6)
    tiny, keep = prepare_linear_tpm([[1e-15]], ["g1"], ["g1"])
    assert keep.tolist() == [True]
    assert tiny[0, 0] == pytest.approx(np.log2(1000001), rel=1e-6)


def test_tcga_dataset_preserves_ids_and_records_zero_samples(tmp_path):
    source = tmp_path / "tcga.csv"
    source.write_text("id,g1,g2,extra\n001,1,3,2\nzero,0,0,1\n", encoding="utf-8")
    dataset = _StagedInferenceDataset(_dataset_config(tmp_path, source), tmp_path / "out", linear_tpm=True)
    assert dataset.sample_ids == ["001"]
    assert dataset.gene_list == ["g2", "missing", "g1"]
    np.testing.assert_allclose(dataset.data[0], np.log2([750001, 1, 250001]) / 20, rtol=1e-6)
    assert pd.read_csv(tmp_path / "out/excluded_samples.csv").sample_id.tolist() == ["zero"]


def test_tcga_dataset_preserves_numeric_only_sample_ids(tmp_path):
    source = tmp_path / "tcga.csv"
    source.write_text("id,g1,g2\n001,1,3\n002,2,2\n", encoding="utf-8")
    dataset = _StagedInferenceDataset(_dataset_config(tmp_path, source), tmp_path / "out", linear_tpm=True)
    assert dataset.sample_ids == ["001", "002"]


def test_legacy_full_model_retains_zero_expression_samples(tmp_path):
    source = tmp_path / "tcga.csv"
    source.write_text("id,g1,g2\ns1,1,3\nzero,0,0\n", encoding="utf-8")
    dataset = _StagedInferenceDataset(
        _dataset_config(tmp_path, source), tmp_path / "out", linear_tpm=True, legacy_round=True
    )
    assert dataset.sample_ids == ["s1", "zero"]
    np.testing.assert_array_equal(dataset.data[1], [0, 0, 0])
    assert pd.read_csv(tmp_path / "out/excluded_samples.csv").empty


@pytest.mark.parametrize(
    "contents, message",
    [
        ("id,g1,g1\na,1,2\n", "duplicate genes"),
        ("id,g1,g2\na,1,2\na,3,4\n", "duplicate sample"),
        ("id,g1,g2\n,1,2\n", "sample IDs"),
        ("id,g1,g2\na,-1,2\n", "finite nonnegative"),
        ("id,g1,g2\na,nan,2\n", "finite nonnegative"),
        ("id,other\na,1\n", "no genes in common"),
        ("id,g1,g2\na,0,0\n", "nonzero expression"),
    ],
)
def test_tcga_dataset_rejects_invalid_inputs(tmp_path, contents, message):
    source = tmp_path / "tcga.csv"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _StagedInferenceDataset(_dataset_config(tmp_path, source), tmp_path / "out", linear_tpm=True)


def test_predictor_only_tcga_runs_and_protects_existing_results(tmp_path, monkeypatch):
    predictor = _predictor(tmp_path, monkeypatch)
    wrong_genes = tmp_path / "other_stage_genes.txt"
    wrong_genes.write_text("other\n", encoding="utf-8")
    predictor.config.model.input_gene_list_fp = wrong_genes
    source = tmp_path / "merged_tpm.csv"
    source.write_text("id,g1,g2\ns1,1,3\nzero,0,0\n", encoding="utf-8")
    test_set = ConfiguredTestSet(dataset_type="tcga", test_set_file_path=source)
    output = tmp_path / "results"
    result = tcga.predict_tcga_test_set(predictor, test_set, output)
    directory = output / "merged_tpm/tcga"
    prediction = pd.read_csv(result["pred_cell_prop_file_path"], index_col=0)
    assert prediction.shape == (1, 2)
    np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-6)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["missing_model_genes_zero_filled"] == ["missing"]
    assert manifest["excluded_sample_count"] == 1
    assert manifest["comparisons"]["deside"]["status"] == "skipped"
    assert manifest["comparisons"]["cpe"]["status"] == "skipped"
    assert not (directory / "gep").exists()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        tcga.predict_tcga_test_set(predictor, test_set, output)


@pytest.mark.parametrize("staged", [True, False])
def test_full_model_tcga_requests_geps_and_uses_correct_preprocessing(tmp_path, monkeypatch, staged):
    predictor = _predictor(tmp_path, monkeypatch, full=True, staged=staged)
    source = tmp_path / "merged_tpm.csv"
    source.write_text("id,g1,g2\ns1,1,3\n", encoding="utf-8")
    test_set = ConfiguredTestSet(dataset_type="tcga", test_set_file_path=source)
    seen = {}

    def fake_evaluate(**kwargs):
        seen.update(kwargs)
        folder = Path(kwargs["result_dir"]) / kwargs["result_set_name"]
        pd.DataFrame({"Immune": [0.2], "Cancer Cells": [0.8]}, index=["s1"]).to_csv(
            folder / "predicted_cell_prop.csv"
        )
        (folder / "gep").mkdir()
        for cell_type in ("Immune", "Cancer Cells"):
            pd.DataFrame([[0, 0, 0]], index=["s1"], columns=["g2", "missing", "g1"]).to_csv(
                folder / "gep" / f"reconstructed_gep_{cell_type}_log2p1.csv"
            )
        return {"pred_cell_prop_file_path": str(folder / "predicted_cell_prop.csv"),
                "cell_types": ["Immune", "Cancer Cells"]}

    monkeypatch.setattr(tcga, "evaluate_model", fake_evaluate)
    tcga.predict_tcga_test_set(predictor, test_set, tmp_path / "results")
    expected = np.log2([750001, 1, 250001])
    if not staged:
        expected = np.round(expected, 3)
    np.testing.assert_allclose(seen["test_set"].data[0], expected / 20, rtol=1e-6)
    assert seen["save_reconstructed_geps"] is True
    assert seen["dataset_type"] == "tcga"


def test_full_model_writes_sample_specific_geps(tmp_path, monkeypatch):
    predictor = _predictor(tmp_path, monkeypatch, full=True)

    class TinyFullModel(torch.nn.Module):
        def __init__(self, data_config):
            super().__init__()
            self.data_config = data_config

        def forward(self, batch):
            n, genes = batch["data"].shape
            return {
                "pred_cell_prop": torch.tensor([[0.2, 0.8]]).repeat(n, 1),
                "recon_x_all_types": torch.ones(n, genes, 2),
            }

    predictor.model = TinyFullModel(predictor.config.data).eval()
    source = tmp_path / "merged_tpm.csv"
    source.write_text("id,g1,g2\ns1,1,3\ns2,2,2\n", encoding="utf-8")
    test_set = ConfiguredTestSet(dataset_type="tcga", test_set_file_path=source)
    result = tcga.predict_tcga_test_set(predictor, test_set, tmp_path / "results")
    folder = Path(result["test_set_result_dir"])
    gep = pd.read_csv(folder / "gep/reconstructed_gep_Cancer Cells_log2p1.csv", index_col=0)
    assert gep.index.tolist() == ["s1", "s2"]
    assert gep.columns.tolist() == ["g2", "missing", "g1"]
    assert (folder / "run_manifest.json").is_file()


def test_mixed_test_sets_dispatch_tcga_without_simulated_references(tmp_path, monkeypatch):
    predictor = _predictor(tmp_path, monkeypatch)
    predictor.config.data.test_sets = {
        "simulated": ConfiguredTestSet(test_set_file_path="sim.h5ad", sct_gep_file_path="ref.h5ad"),
        "tcga": ConfiguredTestSet(test_set_file_path="tcga.csv", dataset_type="tcga"),
    }
    calls = []
    monkeypatch.setattr(predictor, "predict_and_visualize", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(tcga, "predict_tcga_test_set", lambda *args: calls.append(args) or {})
    result = predictor.predict_configured_test_sets(output_dir=str(tmp_path))
    assert set(result) == {"simulated", "tcga"}
    assert calls[0]["sct_gep_file_path"] == "ref.h5ad"
    assert calls[1][1].sct_gep_file_path == ""


def test_tcga_runs_from_post_stage_prediction(tmp_path, monkeypatch):
    from vaedecon.workflow import inference
    from vaedecon.workflow.train import VAEDeconTrainer

    predictor = _predictor(tmp_path, monkeypatch)
    source = tmp_path / "merged_tpm.csv"
    source.write_text("id,g1,g2\ns1,1,3\n", encoding="utf-8")
    predictor.config.data.test_sets = {
        "TCGA": ConfiguredTestSet(dataset_type="tcga", test_set_file_path=source)
    }
    trainer = VAEDeconTrainer.__new__(VAEDeconTrainer)
    trainer.config = predictor.config
    trainer.device = "cpu"
    monkeypatch.setattr(inference, "VAEDeconPredictor", lambda **kwargs: predictor)
    stage = tmp_path / "stage_cell_prop_predictor_pretrain"
    output = trainer._run_post_stage_test_set_prediction(
        stage_name="cell_prop_predictor_pretrain", stage_dir=stage
    )
    assert (output / "merged_tpm/tcga/predicted_cell_prop.csv").is_file()
    assert (output / "merged_tpm/tcga/run_manifest.json").is_file()


def test_comparisons_run_independently_and_report_failures(tmp_path, monkeypatch):
    deside = tmp_path / "deside.csv"
    deside.write_text("placeholder", encoding="utf-8")
    calls = []
    monkeypatch.setattr(tcga, "compare_cancer_cell_ccc", lambda *args: calls.append(args) or {"count": 2})
    references = TCGAComparisonConfig(deside_proportion_file_path=deside)
    status = tcga._comparison_status(tmp_path, references)
    assert status["deside"]["status"] == "complete"
    assert status["cpe"]["status"] == "skipped"
    assert len(calls) == 1

    def fail(*args):
        raise ValueError("bad reference")

    monkeypatch.setattr(tcga, "compare_cancer_cell_ccc", fail)
    status = tcga._comparison_status(tmp_path, references)
    assert status["deside"]["status"] == "failed"
    assert "bad reference" in status["deside"]["error"]
    assert status["cpe"]["status"] == "skipped"


def test_package_comparisons_produce_reference_metrics(tmp_path):
    result = tmp_path / "tcga"
    result.mkdir()
    pd.DataFrame(
        {"Cancer Cells": [0.2, 0.8, 0.3, 0.7], "Immune": [0.8, 0.2, 0.7, 0.3]},
        index=["s1", "s2", "s3", "s4"],
    ).to_csv(result / "predicted_cell_prop.csv")
    deside = tmp_path / "deside.csv"
    pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "cancer_type": ["LIHC", "LIHC", "BLCA", "BLCA"],
        "Cancer Cells": [0.2, 0.8, 0.3, 0.7],
        "Immune": [0.8, 0.2, 0.7, 0.3],
    }).to_csv(deside, index=False)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"cancer_type": ["LIHC", "LIHC", "BLCA", "BLCA"]},
                 index=["s1", "s2", "s3", "s4"]).to_csv(mapping)
    cpe = tmp_path / "cpe.csv"
    pd.DataFrame({"Sample ID": ["s1", "s2", "s3", "s4"],
                  "Cancer type": ["LIHC", "LIHC", "BLCA", "BLCA"],
                  "CPE": [0.2, 0.8, 0.3, 0.7]}).to_csv(cpe, index=False)
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame([[0.1] * 5, [0.2] * 5], index=["LIHC", "BLCA"],
                 columns=["DeSide-D1"] + [f"DeSide-D1_{i}" for i in range(1, 5)]).to_csv(baseline)
    references = TCGAComparisonConfig(
        deside_proportion_file_path=deside, cancer_purity_file_path=cpe,
        cancer_type_mapping_file_path=mapping, deside_cpe_ccc_file_path=baseline,
    )
    status = tcga._comparison_status(result, references)
    assert status["deside"]["status"] == "complete"
    assert status["cpe"]["status"] == "complete"
    metrics = pd.read_csv(status["cpe"]["metrics_file"], index_col=0)
    assert metrics.loc["LIHC", "vaedecon_ccc"] == pytest.approx(1)
    assert metrics.loc["LIHC", "deside_ccc_mean"] == pytest.approx(0.1)
    assert Path(status["deside"]["cell_type_figure_file"]).is_file()
    assert Path(status["cpe"]["scatter_figure_file"]).is_file()


def test_cpe_matching_preserves_leading_zeros_in_sample_ids(tmp_path):
    from vaedecon.workflow.tcga_comparison import _load_cpe_matched_predictions

    pd.DataFrame({"Cancer Cells": [0.2]}, index=["001"]).to_csv(tmp_path / "predicted_cell_prop.csv")
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"cancer_type": ["LIHC"]}, index=["001"]).to_csv(mapping)
    purity = tmp_path / "purity.csv"
    pd.DataFrame({"Sample ID": ["001"], "Cancer type": ["LIHC"], "CPE": [0.3]}).to_csv(
        purity, index=False
    )
    matched = _load_cpe_matched_predictions(tmp_path, purity, mapping)
    assert matched.index.tolist() == ["001"]
    assert matched["has_cpe_sample"].tolist() == [True]


def test_checkpoint_marked_incompatible_is_rejected(tmp_path, monkeypatch):
    predictor = _predictor(tmp_path, monkeypatch)
    (Path(predictor.model_dir) / "model_kind.json").write_text(
        '{"normalize_training_intersection": false}', encoding="utf-8"
    )
    source = tmp_path / "tcga.csv"
    source.write_text("id,g1,g2\ns1,1,3\n", encoding="utf-8")
    test_set = ConfiguredTestSet(dataset_type="tcga", test_set_file_path=source)
    with pytest.raises(ValueError, match="incompatible"):
        tcga.predict_tcga_test_set(predictor, test_set, tmp_path / "results")


def test_training_output_preflight_protects_tcga_comparison_inputs(tmp_path):
    from vaedecon.workflow.train import _preflight_training_outputs

    root = tmp_path / "run"
    reference = root / "test_results" / "deside.csv"
    reference.parent.mkdir(parents=True)
    reference.write_text("reference", encoding="utf-8")
    config = VAEDeconConfig()
    config.data.test_sets["TCGA"] = ConfiguredTestSet(
        dataset_type="tcga", test_set_file_path=tmp_path / "tcga.csv",
        tcga_comparisons={"deside_proportion_file_path": reference},
    )
    with pytest.raises(ValueError, match="overlaps"):
        _preflight_training_outputs(config, root / "final_model", overwrite=True)
    assert reference.read_text(encoding="utf-8") == "reference"
