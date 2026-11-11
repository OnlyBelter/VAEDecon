import anndata
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from vaedecon.configs import DataConfig, VAEDeconConfig
from vaedecon.data.bulk_dataset import BulkOnlyDataset
from vaedecon.data.bulk_sources import group_cell_fractions, inspect_bulk_sources, resolve_bulk_source_paths


def _write_bulk(path, labels, genes):
    anndata.AnnData(
        X=np.ones((len(labels), len(genes)), dtype=np.float32),
        obs=labels,
        var=pd.DataFrame(index=genes),
    ).write_h5ad(path)


def _config(paths):
    return DataConfig.model_validate(
        {"training_target_sets": {
            name: {"training_set_file_path": str(path)}
            for name, path in paths.items()
        }},
        context={"defer_sct_requirements": True},
    )


def test_bulk_sources_align_labels_and_exclude_zero_rows(tmp_path):
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    _write_bulk(
        a,
        pd.DataFrame({"rare": [0.0004, 0.0], "other": [0.9996, 0.0]},
                     index=["sample", "zero"]),
        ["g2", "g1", "g3"],
    )
    _write_bulk(
        b,
        pd.DataFrame({"other": [0.8], "rare": [0.2]}, index=["sample"]),
        ["g1", "g2", "g4"],
    )
    inspected = inspect_bulk_sources(_config({"first": a, "second": b}))
    assert inspected.gene_names == ("g2", "g1")
    assert inspected.cell_types == ("rare", "other")
    assert inspected.excluded_sample_ids == ("first::zero",)
    assert inspected.sources[0].sample_ids == ("sample", "zero")
    np.testing.assert_allclose(inspected.sources[0].labels[0], [0.0004, 0.9996])
    np.testing.assert_allclose(inspected.sources[1].labels[0], [0.2, 0.8])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_bulk_sources_reject_invalid_labels(tmp_path, bad):
    path = tmp_path / "a.h5ad"
    _write_bulk(path, pd.DataFrame({"cell": [bad]}, index=["sample"]), ["g1"])
    with pytest.raises(ValueError, match="fractions"):
        inspect_bulk_sources(_config({"first": path}))


def test_bulk_sources_reject_incompatible_cell_types(tmp_path):
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    _write_bulk(a, pd.DataFrame({"first": [1.0]}, index=["sample"]), ["g1"])
    _write_bulk(b, pd.DataFrame({"second": [1.0]}, index=["sample"]), ["g1"])
    with pytest.raises(ValueError, match="incompatible cell types"):
        inspect_bulk_sources(_config({"a": a, "b": b}))


def test_cell_type_grouping_uses_deside_column_order_and_selection():
    labels = pd.DataFrame({"T1": [0.1], "Cancer": [0.5], "T2": [0.2], "Other": [0.2]})
    grouped = group_cell_fractions(labels, {"T": ["T1", "T2"]})
    assert grouped.columns.tolist() == ["Cancer", "Other", "T"]
    np.testing.assert_allclose(grouped.iloc[0].to_numpy(), [0.5, 0.2, 0.3])
    selected = group_cell_fractions(labels, {"T": ["T1", "T2"]}, ["T", "Cancer"])
    assert selected.columns.tolist() == ["T", "Cancer"]


def test_bulk_source_paths_reject_conflicts_and_duplicates(tmp_path):
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    config = _config({"first": a})
    config.simu_bulk_file_path = [b]
    with pytest.raises(ValueError, match="same bulk files"):
        resolve_bulk_source_paths(config)
    config = _config({"first": a, "second": a})
    with pytest.raises(ValueError, match="duplicate bulk paths"):
        resolve_bulk_source_paths(config)


def test_stage_one_config_defers_sct_requirements():
    config = VAEDeconConfig.from_dict({
        "data": {"training_target_sets": {
            "bulk": {"training_set_file_path": "/data/bulk.h5ad"}
        }},
        "training": {"staged_training": {
            "enabled": True,
            "run_stages": ["cell_prop_predictor_pretrain"],
            "stages": [{
                "name": "cell_prop_predictor_pretrain",
                "max_epochs": 1,
                "train_modules": ["cell_prop_predictor"],
            }],
        }},
        "model": {
            "predict_cell_prop": True,
            "cell_prop_predictor_cls": "DeSideCellPropPredictor",
            "cell_prop_activation_function": "sigmoid",
            "cancer_cell_type_name": "Cancer Cells",
        },
    })
    config.data.force_reprocess = True
    assert config.data.training_target_sets["bulk"].training_sct_gep_file_path == ""


def test_bulk_dataset_is_chunked_namespaced_and_cacheable(tmp_path):
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    _write_bulk(a, pd.DataFrame({"rare": [0.0004], "other": [0.9996]}, index=["s"]), ["g2", "g1"])
    _write_bulk(b, pd.DataFrame({"other": [0.8], "rare": [0.2]}, index=["s"]), ["g1", "g2"])
    config = _config({"first": a, "second": b})
    config.data_dir = tmp_path
    config.remove_low_var_genes = False
    config.scaling_by_constant = False
    config.chunk_size = 1
    dataset = BulkOnlyDataset(config)
    assert dataset.get_sample_ids() == ["first::s", "second::s"]
    assert dataset.get_gene_list() == ["g2", "g1"]
    np.testing.assert_allclose(dataset[0].labels.numpy(), [0.0004, 0.9996])
    np.testing.assert_allclose(dataset.data[0], [np.log2(500001)] * 2, rtol=1e-6)
    cached = BulkOnlyDataset(config)
    np.testing.assert_allclose(cached.data, dataset.data)


def test_bulk_cache_tracks_gene_filter_reference_changes(tmp_path):
    source = tmp_path / "bulk.h5ad"
    reference = tmp_path / "reference.csv"
    labels = pd.DataFrame({"cell": [1.0, 1.0]}, index=["s1", "s2"])
    anndata.AnnData(
        X=np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32),
        obs=labels, var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(source)
    pd.DataFrame({"cell": [1.0]}, index=["g1"]).to_csv(reference)
    config = _config({"bulk": source})
    config.data_dir = tmp_path
    config.remove_low_var_genes = True
    config.min_var = 0.0
    config.cell_cell2ave_exp_file_path = reference
    first = BulkOnlyDataset(config)
    assert first.gene_list == ["g1"]
    pd.DataFrame({"cell": [1.0, 2.0]}, index=["g1", "g2"]).to_csv(reference)
    second = BulkOnlyDataset(config)
    assert second.gene_list == ["g1", "g2"]
    assert first.processed_data_dir != second.processed_data_dir


def test_single_source_pathway_training_normalizes_gene_expression(tmp_path):
    source = tmp_path / "bulk.h5ad"
    anndata.AnnData(
        X=np.log2(np.array([[2.0, 3.0]]) + 1).astype(np.float32),
        obs=pd.DataFrame({"cell": [1.0]}, index=["s"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(source)
    config = _config({"bulk": source})
    config.data_dir = tmp_path
    config.remove_low_var_genes = False
    config.scaling_by_constant = False
    model_config = SimpleNamespace(
        deside_pathway_network=True, deside_group_cell_types={},
        deside_cell_types=None, cell_prop_activation_function="softmax",
    )
    pathway_data = BulkOnlyDataset(config, model_config)
    plain_data = BulkOnlyDataset(config)
    np.testing.assert_allclose(pathway_data.data[0], np.log2([400001, 600001]), rtol=1e-6)
    np.testing.assert_allclose(plain_data.data[0], np.log2([400001, 600001]), rtol=1e-6)
    assert pathway_data.processed_data_dir == plain_data.processed_data_dir


def test_bulk_excludes_zero_expression_before_and_after_filtering(tmp_path):
    path = tmp_path / "bulk.h5ad"
    values = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 3, 0]], dtype=np.float32)
    anndata.AnnData(
        X=np.log2(values + 1),
        obs=pd.DataFrame({"cell": [1.0] * 4}, index=["common_zero", "selected_zero", "a", "b"]),
        var=pd.DataFrame(index=["g1", "g2", "unused"]),
    ).write_h5ad(path)
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"cell": [1.0]}, index=["g2"]).to_csv(reference)
    config = _config({"bulk": path})
    config.data_dir = tmp_path
    config.chunk_size = 2
    config.remove_low_var_genes = True
    config.min_var = 0.0
    config.cell_cell2ave_exp_file_path = reference
    config.scaling_by_constant = False
    dataset = BulkOnlyDataset(config)
    assert dataset.sample_ids == ["bulk::a", "bulk::b"]
    assert dataset.gene_list == ["g2"]
    assert {row["sample_id"]: row["reason"] for row in dataset.exclusions} == {
        "bulk::common_zero": "zero_common_expression",
        "bulk::selected_zero": "zero_selected_expression",
    }
    np.testing.assert_allclose(dataset.data[:, 0], np.log2(1_000_001), rtol=1e-6)
    cached = BulkOnlyDataset(config)
    assert cached.sample_ids == dataset.sample_ids
    assert cached.exclusions == dataset.exclusions
    np.testing.assert_array_equal(cached.data, dataset.data)


@pytest.mark.parametrize("filter_genes", [False, True])
def test_all_zero_bulk_records_exclusions_and_fails(tmp_path, filter_genes):
    path = tmp_path / "bulk.h5ad"
    anndata.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"cell": [1.0, 1.0]}, index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(path)
    config = _config({"bulk": path})
    config.data_dir = tmp_path
    config.remove_low_var_genes = filter_genes
    with pytest.raises(ValueError, match="No bulk samples have nonzero expression"):
        BulkOnlyDataset(config)
    manifests = list((tmp_path / "processed_training_sets").glob("bulk_only_*/excluded_samples.csv"))
    assert len(manifests) == 1
    excluded = pd.read_csv(manifests[0])
    assert excluded["sample_id"].tolist() == ["bulk::a", "bulk::b"]
    assert excluded["reason"].tolist() == ["zero_common_expression"] * 2


def test_pathway_intersection_filters_before_normalization(tmp_path):
    source = tmp_path / "bulk.h5ad"
    pathway = tmp_path / "pathways.gmt"
    pathway.write_text("p1\tdesc\tg1\tg3\n", encoding="utf-8")
    anndata.AnnData(
        X=np.log2(np.array([[1.0, 50.0, 3.0]]) + 1).astype(np.float32),
        obs=pd.DataFrame({"cell": [1.0]}, index=["s"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    ).write_h5ad(source)
    config = _config({"bulk": source})
    config.data_dir = tmp_path
    config.remove_low_var_genes = False
    config.scaling_by_constant = False
    config.pathway_file_path = [pathway]
    model_config = SimpleNamespace(
        deside_pathway_network=True, deside_input_gene_list="intersection_with_pathway_genes",
        deside_group_cell_types={}, deside_cell_types=None, cell_prop_activation_function="softmax",
    )
    dataset = BulkOnlyDataset(config, model_config)
    assert dataset.gene_list == ["g1", "g3"]
    np.testing.assert_allclose(dataset.data[0], np.log2([250001, 750001]), rtol=1e-6)


def test_linear_tpm_variance_uses_all_valid_bulk_rows_and_common_genes(tmp_path, caplog):
    caplog.set_level("INFO", logger="vaedecon.data.bulk_dataset")
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    a_values = np.array([[0, 4, 6, 50], [2, 2, 6, 50], [9, 1, 0, 50]], dtype=np.float32)
    b_values = np.array([[6, 0, 4, 30], [6, 2, 2, 30]], dtype=np.float32)
    labels_a = pd.DataFrame({"rare": [0.0004, 0.0004, 0], "other": [0.9996, 0.9996, 0]},
                            index=["s1", "s2", "excluded"])
    labels_b = pd.DataFrame({"other": [0.9996, 0.9996], "rare": [0.0004, 0.0004]},
                            index=["s1", "s2"])
    for path, values, labels, genes in (
        (a, a_values, labels_a, ["g1", "g2", "g3", "only_a"]),
        (b, b_values, labels_b, ["g3", "g2", "g1", "only_b"]),
    ):
        anndata.AnnData(
            X=np.log2(values + 1).astype(np.float32),
            obs=labels, var=pd.DataFrame(index=genes),
        ).write_h5ad(path)
    config = _config({"first": a, "second": b})
    config.data_dir = tmp_path
    config.chunk_size = 1
    config.remove_low_var_genes = True
    config.min_var = 1.0
    config.scaling_by_constant = False
    dataset = BulkOnlyDataset(config)
    common = np.array([[0, 4, 6], [2, 2, 6], [4, 0, 6], [2, 2, 6]], dtype=float)
    reference = common / common.sum(axis=1, keepdims=True) * 1e6
    expected_genes = [g for g, v in zip(["g1", "g2", "g3"], reference.var(axis=0, ddof=1)) if v > 1]
    assert dataset.gene_list == expected_genes == ["g1", "g2"]
    assert dataset.sample_ids == ["first::s1", "first::s2", "second::s1", "second::s2"]
    np.testing.assert_allclose(dataset[0].labels.numpy(), [0.0004, 0.9996])
    np.testing.assert_allclose(dataset.data[0], np.log2([1, 1_000_001]), rtol=1e-6)
    assert "linear TPM variance" in caplog.text


def test_bulk_checkpoint_fingerprint_rejects_changed_source(tmp_path):
    path = tmp_path / "bulk.h5ad"
    labels = pd.DataFrame({"cell": [1.0, 1.0]}, index=["s1", "s2"])
    _write_bulk(path, labels, ["g1", "g2"])
    config = _config({"bulk": path})
    config.data_dir = tmp_path
    config.remove_low_var_genes = False
    dataset = BulkOnlyDataset(config)
    _write_bulk(path, labels, ["g1", "g2", "g3"])
    with pytest.raises(ValueError, match="source files changed"):
        dataset.validate_source_identity()
    with pytest.raises(ValueError, match="preprocessing or source files have changed"):
        BulkOnlyDataset(config, expected_fingerprint=dataset.fingerprint)


def test_linear_tpm_variance_threshold_uses_sample_variance(tmp_path):
    path = tmp_path / "bulk.h5ad"
    tpm = np.array([[1, 3, 999996], [3, 2, 999995],
                    [1, 3, 999996], [3, 2, 999995]], dtype=float)
    log_values = np.log2(tpm + 1).astype(np.float32)
    anndata.AnnData(
        X=log_values, obs=pd.DataFrame({"cell": [1.0] * 4}, index=["s1", "s2", "s3", "s4"]),
        var=pd.DataFrame(index=["above", "below", "background"]),
    ).write_h5ad(path)
    config = _config({"bulk": path})
    config.data_dir = tmp_path
    config.remove_low_var_genes = True
    config.scaling_by_constant = False
    config.min_var = 1.0
    dataset = BulkOnlyDataset(config)
    linear = np.exp2(log_values.astype(np.float64)) - 1
    normalized = linear / linear.sum(axis=1, keepdims=True) * 1e6
    variances = normalized.var(axis=0, ddof=1)
    assert variances[0] > 1
    assert variances[1] < 1
    assert "above" in dataset.gene_list
    assert "below" not in dataset.gene_list


def test_linear_filter_does_not_reuse_log_space_cache(tmp_path):
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    labels = pd.DataFrame({"cell": [1.0, 1.0]}, index=["s1", "s2"])
    for path in (a, b):
        anndata.AnnData(
            np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
            obs=labels, var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(path)
    config = _config({"first": a, "second": b})
    config.data_dir = tmp_path
    config.remove_low_var_genes = True
    config.scaling_by_constant = False
    config.min_var = 1.0
    old_payload = {
        "version": 2,
        "sources": [(name, str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                    for name, path in (("first", a), ("second", b))],
        "genes": ["g1", "g2"], "group_cell_types": {}, "cell_types": None,
        "scale": None, "normalize_intersection": True, "remove_low_var_genes": True,
        "min_var": 1.0, "reference": None,
    }
    old_fingerprint = hashlib.sha256(
        json.dumps(old_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    old_dir = tmp_path / "processed_training_sets" / f"bulk_only_{old_fingerprint}"
    old_dir.mkdir(parents=True)
    (old_dir / "metadata.json").write_text('{"gene_list": ["g1"]}', encoding="utf-8")
    np.save(old_dir / "data.npy", np.zeros((4, 1), dtype=np.float32))
    dataset = BulkOnlyDataset(config)
    assert dataset.processed_data_dir != old_dir
    assert dataset.gene_list == ["g1", "g2"]
    assert (old_dir / "metadata.json").read_text(encoding="utf-8") == '{"gene_list": ["g1"]}'
