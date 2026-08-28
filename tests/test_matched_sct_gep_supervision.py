from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch

import vaedecon.data.datasets as datasets_module
from vaedecon.configs.default_config import GEPDatasetConfig
from vaedecon.data.datasets import GEPDataset
from vaedecon.data.datasets import build_matched_sct_gep_training_targets
from vaedecon.models.vae.vae_model import VAE, to_log_space
from vaedecon.utility import non_log2log_cpm
from vaedecon.utility.read_file import ReadExp


def _write_h5ad(path: Path, x: np.ndarray, obs_names: list[str], var_names: list[str], obs: pd.DataFrame) -> None:
    obs = obs.copy()
    obs.index = pd.Index(list(obs_names), dtype=object)
    adata = ad.AnnData(
        X=np.asarray(x, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(list(var_names), dtype=object)),
    )
    adata.obs_names = pd.Index(list(obs_names), dtype=object)
    adata.var_names = pd.Index(list(var_names), dtype=object)
    adata.write_h5ad(path)


def test_build_matched_sct_gep_training_targets_returns_dense_tensor(tmp_path: Path):
    genes = ["gene_a", "gene_b"]
    sct_path = tmp_path / "sct.h5ad"
    _write_h5ad(
        sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
    )
    mapping_path = tmp_path / "sample2cell.csv"
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b", "cell_c"],
        },
        index=["sample_1", "sample_1", "sample_2"],
    ).to_csv(mapping_path)

    result = build_matched_sct_gep_training_targets(
        sct_gep_dataset_file_path=sct_path,
        sample2cell_id_file_path=mapping_path,
        bulk_sample_ids=["sample_1", "sample_2"],
        target_gene_list=genes,
        cell_types=["CT1", "CT2"],
        cache_sct_query_results=False,
    )

    assert result["true_sct_gep"].shape == (2, 2, 2)
    assert result["true_sct_gep_present_mask"].tolist() == [[True, True], [True, False]]
    aligned = result["aligned_sct_geps_df"]
    np.testing.assert_allclose(result["true_sct_gep"][0, :, 0], aligned.loc["cell_a", :].to_numpy(dtype=np.float32))
    np.testing.assert_allclose(result["true_sct_gep"][0, :, 1], aligned.loc["cell_b", :].to_numpy(dtype=np.float32))
    np.testing.assert_allclose(result["true_sct_gep"][1, :, 0], aligned.loc["cell_c", :].to_numpy(dtype=np.float32))


def test_gepdataset_caches_true_sct_gep_and_masks_nonbulk_rows(tmp_path: Path):
    genes = ["gene_a", "gene_b"]
    bulk_path = tmp_path / "bulk.h5ad"
    aux_sct_path = tmp_path / "aux_sct.h5ad"
    ref_sct_path = tmp_path / "ref_sct.h5ad"
    mapping_path = tmp_path / "sample2cell.csv"

    _write_h5ad(
        bulk_path,
        x=np.array([[2.0, 4.0], [3.0, 9.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame(
            {
                "CT1": [0.7, 0.8],
                "CT2": [0.3, 0.2],
            },
            index=["sample_1", "sample_2"],
        ),
    )
    _write_h5ad(
        aux_sct_path,
        x=np.array([[1.5, 1.0]], dtype=np.float32),
        obs_names=["standalone_sct_row"],
        var_names=genes,
        obs=pd.DataFrame(
            {
                "CT1": [1.0],
                "CT2": [0.0],
            },
            index=["standalone_sct_row"],
        ),
    )
    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
    )
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b", "cell_c"],
        },
        index=["sample_1", "sample_1", "sample_2"],
    ).to_csv(mapping_path)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_path, aux_sct_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_path,
                    "training_set_sample2cell_id_file_path": mapping_path,
                    "training_sct_gep_file_path": ref_sct_path,
                }
            },
        )
    )

    assert dataset.true_sct_gep.shape == (3, 2, 2)
    assert dataset.true_sct_gep_present_mask.tolist() == [
        [True, True],
        [True, False],
        [False, False],
    ]
    sample_1 = dataset[0]
    aligned = build_matched_sct_gep_training_targets(
        sct_gep_dataset_file_path=ref_sct_path,
        sample2cell_id_file_path=mapping_path,
        bulk_sample_ids=["sample_1", "sample_2"],
        target_gene_list=genes,
        cell_types=["CT1", "CT2"],
        cache_sct_query_results=False,
    )["aligned_sct_geps_df"]
    np.testing.assert_allclose(
        sample_1.true_sct_gep[:, 0].numpy(),
        aligned.loc["cell_a", :].to_numpy(dtype=np.float32),
    )
    assert dataset[2].true_sct_gep_present_mask.tolist() == [False, False]


def test_gepdataset_namespaces_duplicate_bulk_sample_ids_across_training_sets(tmp_path: Path):
    genes = ["gene_a", "gene_b"]
    bulk_a_path = tmp_path / "bulk_a.h5ad"
    bulk_b_path = tmp_path / "bulk_b.h5ad"
    ref_sct_path = tmp_path / "ref_sct.h5ad"
    mapping_a_path = tmp_path / "sample2cell_a.csv"
    mapping_b_path = tmp_path / "sample2cell_b.csv"

    _write_h5ad(
        bulk_a_path,
        x=np.array([[2.0, 4.0], [3.0, 9.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0, 1.0]}, index=["sample_1", "sample_2"]),
    )
    _write_h5ad(
        bulk_b_path,
        x=np.array([[5.0, 7.0], [11.0, 13.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0, 1.0]}, index=["sample_1", "sample_2"]),
    )
    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c", "cell_d"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c", "cell_d"]),
    )
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b"],
        },
        index=["sample_1", "sample_2"],
    ).to_csv(mapping_a_path)
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT1"],
            "selected_cell_id": ["cell_c", "cell_d"],
        },
        index=["sample_1", "sample_2"],
    ).to_csv(mapping_b_path)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_a_path, bulk_b_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_a_path,
                    "training_set_sample2cell_id_file_path": mapping_a_path,
                    "training_sct_gep_file_path": ref_sct_path,
                },
                "Train_set2": {
                    "training_set_file_path": bulk_b_path,
                    "training_set_sample2cell_id_file_path": mapping_b_path,
                    "training_sct_gep_file_path": ref_sct_path,
                },
            },
        )
    )

    assert dataset.sample_ids == [
        "Train_set1::sample_1",
        "Train_set1::sample_2",
        "Train_set2::sample_1",
        "Train_set2::sample_2",
    ]
    assert dataset.true_sct_gep_present_mask.tolist() == [[True], [True], [True], [True]]

    aligned_a = build_matched_sct_gep_training_targets(
        sct_gep_dataset_file_path=ref_sct_path,
        sample2cell_id_file_path=mapping_a_path,
        bulk_sample_ids=["sample_1", "sample_2"],
        target_gene_list=genes,
        cell_types=["CT1"],
        cache_sct_query_results=False,
    )["aligned_sct_geps_df"]
    aligned_b = build_matched_sct_gep_training_targets(
        sct_gep_dataset_file_path=ref_sct_path,
        sample2cell_id_file_path=mapping_b_path,
        bulk_sample_ids=["sample_1", "sample_2"],
        target_gene_list=genes,
        cell_types=["CT1"],
        cache_sct_query_results=False,
    )["aligned_sct_geps_df"]
    np.testing.assert_allclose(
        dataset[0].true_sct_gep[:, 0].numpy(),
        aligned_a.loc["cell_a", :].to_numpy(dtype=np.float32),
    )
    np.testing.assert_allclose(
        dataset[2].true_sct_gep[:, 0].numpy(),
        aligned_b.loc["cell_c", :].to_numpy(dtype=np.float32),
    )


def test_gepdataset_loads_shared_sct_reference_once_per_run(tmp_path: Path, monkeypatch):
    genes = ["gene_a", "gene_b"]
    bulk_a_path = tmp_path / "bulk_a.h5ad"
    bulk_b_path = tmp_path / "bulk_b.h5ad"
    ref_sct_path = tmp_path / "shared_ref_sct.h5ad"
    mapping_a_path = tmp_path / "sample2cell_a.csv"
    mapping_b_path = tmp_path / "sample2cell_b.csv"

    _write_h5ad(
        bulk_a_path,
        x=np.array([[2.0, 4.0], [3.0, 9.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0, 1.0]}, index=["sample_1", "sample_2"]),
    )
    _write_h5ad(
        bulk_b_path,
        x=np.array([[5.0, 7.0], [11.0, 13.0]], dtype=np.float32),
        obs_names=["sample_3", "sample_4"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0, 1.0]}, index=["sample_3", "sample_4"]),
    )
    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c", "cell_d"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c", "cell_d"]),
    )
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b"],
        },
        index=["sample_1", "sample_2"],
    ).to_csv(mapping_a_path)
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT1"],
            "selected_cell_id": ["cell_c", "cell_d"],
        },
        index=["sample_3", "sample_4"],
    ).to_csv(mapping_b_path)

    original_read_h5ad = datasets_module.ReadH5AD
    shared_load_counter = {"count": 0}

    class CountingReadH5AD:
        def __init__(self, file_path, *args, **kwargs):
            if Path(file_path).expanduser().resolve() == ref_sct_path.resolve():
                shared_load_counter["count"] += 1
            self._inner = original_read_h5ad(file_path, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(datasets_module, "ReadH5AD", CountingReadH5AD)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_a_path, bulk_b_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_a_path,
                    "training_set_sample2cell_id_file_path": mapping_a_path,
                    "training_sct_gep_file_path": ref_sct_path,
                },
                "Train_set2": {
                    "training_set_file_path": bulk_b_path,
                    "training_set_sample2cell_id_file_path": mapping_b_path,
                    "training_sct_gep_file_path": ref_sct_path,
                },
            },
        )
    )

    assert shared_load_counter["count"] == 1
    assert dataset.true_sct_gep_present_mask.tolist() == [[True], [True], [True], [True]]


def test_gepdataset_grouped_bulk_loading_preserves_input_order(tmp_path: Path):
    genes = ["gene_a", "gene_b", "gene_c"]
    bulk_a_path = tmp_path / "bulk_a.h5ad"
    bulk_b_path = tmp_path / "bulk_b.h5ad"
    bulk_c_path = tmp_path / "bulk_c.h5ad"
    ref_shared_path = tmp_path / "shared_ref_sct.h5ad"
    ref_other_path = tmp_path / "other_ref_sct.h5ad"
    mapping_a_path = tmp_path / "sample2cell_a.csv"
    mapping_b_path = tmp_path / "sample2cell_b.csv"
    mapping_c_path = tmp_path / "sample2cell_c.csv"

    _write_h5ad(
        bulk_a_path,
        x=np.array([[2.0, 4.0, 8.0]], dtype=np.float32),
        obs_names=["sample_a"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_a"]),
    )
    _write_h5ad(
        bulk_b_path,
        x=np.array([[3.0, 9.0, 27.0]], dtype=np.float32),
        obs_names=["sample_b"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_b"]),
    )
    _write_h5ad(
        bulk_c_path,
        x=np.array([[5.0, 25.0, 125.0]], dtype=np.float32),
        obs_names=["sample_c"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_c"]),
    )
    _write_h5ad(
        ref_shared_path,
        x=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
    )
    _write_h5ad(
        ref_other_path,
        x=np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
        obs_names=["cell_c"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_c"]),
    )
    pd.DataFrame({"cell_type": ["CT1"], "selected_cell_id": ["cell_a"]}, index=["sample_a"]).to_csv(mapping_a_path)
    pd.DataFrame({"cell_type": ["CT1"], "selected_cell_id": ["cell_b"]}, index=["sample_b"]).to_csv(mapping_b_path)
    pd.DataFrame({"cell_type": ["CT1"], "selected_cell_id": ["cell_c"]}, index=["sample_c"]).to_csv(mapping_c_path)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_a_path, bulk_b_path, bulk_c_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_a_path,
                    "training_set_sample2cell_id_file_path": mapping_a_path,
                    "training_sct_gep_file_path": ref_shared_path,
                },
                "Train_set2": {
                    "training_set_file_path": bulk_b_path,
                    "training_set_sample2cell_id_file_path": mapping_b_path,
                    "training_sct_gep_file_path": ref_shared_path,
                },
                "Train_set3": {
                    "training_set_file_path": bulk_c_path,
                    "training_set_sample2cell_id_file_path": mapping_c_path,
                    "training_sct_gep_file_path": ref_other_path,
                },
            },
        )
    )

    assert dataset.sample_ids == [
        "Train_set1::sample_a",
        "Train_set2::sample_b",
        "Train_set3::sample_c",
    ]
    assert dataset.true_sct_gep_present_mask.tolist() == [[True], [True], [True]]


def test_gepdataset_discovers_common_genes_and_preserves_gene_list_order(tmp_path: Path):
    h5ad_path = tmp_path / "bulk_a.h5ad"
    csv_path = tmp_path / "bulk_b.csv"
    gene_list_path = tmp_path / "gene_list.txt"

    _write_h5ad(
        h5ad_path,
        x=np.array([[2.0, 4.0, 8.0]], dtype=np.float32),
        obs_names=["sample_a"],
        var_names=["gene_a", "gene_b", "gene_c"],
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_a"]),
    )
    pd.DataFrame(
        [[3.0, 9.0, 27.0]],
        index=["sample_b"],
        columns=["gene_b", "gene_c", "gene_d"],
    ).to_csv(csv_path)
    gene_list_path.write_text("gene_c\ngene_b\ngene_x\n", encoding="utf-8")

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[h5ad_path, csv_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            remove_low_var_genes=False,
            gene_list_file=gene_list_path,
        )
    )

    assert dataset.gene_list == ["gene_c", "gene_b"]
    assert dataset.sample_ids == ["sample_a", "sample_b"]
    assert dataset.data.shape == (2, 2)
    assert (tmp_path / "processed" / "common_gene_list.txt").read_text(encoding="utf-8").splitlines() == ["gene_c", "gene_b"]


def test_gepdataset_reuses_saved_common_gene_list_during_force_reprocess(tmp_path: Path, monkeypatch):
    bulk_a_path = tmp_path / "bulk_a.h5ad"
    bulk_b_path = tmp_path / "bulk_b.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "common_gene_list.txt").write_text("gene_c\ngene_b\n", encoding="utf-8")

    _write_h5ad(
        bulk_a_path,
        x=np.array([[2.0, 4.0, 8.0]], dtype=np.float32),
        obs_names=["sample_a"],
        var_names=["gene_a", "gene_b", "gene_c"],
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_a"]),
    )
    pd.DataFrame(
        [[3.0, 9.0, 27.0]],
        index=["sample_b"],
        columns=["gene_b", "gene_c", "gene_d"],
    ).to_csv(bulk_b_path)

    def _unexpected_discovery(*_args, **_kwargs):
        raise AssertionError("Common gene discovery should not run when the cache-local gene list already exists.")

    monkeypatch.setattr(datasets_module, "_discover_final_target_gene_list", _unexpected_discovery)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_a_path, bulk_b_path],
            processed_data_dir=processed_dir,
            force_reprocess=True,
            scaling_by_constant=False,
            remove_low_var_genes=False,
        )
    )

    assert dataset.gene_list == ["gene_c", "gene_b"]
    assert dataset.sample_ids == ["sample_a", "sample_b"]
    assert dataset.data.shape == (2, 2)


def test_gepdataset_renormalizes_h5ad_after_gene_removal_before_recovering_log_space(tmp_path: Path):
    h5ad_path = tmp_path / "bulk_a.h5ad"
    gene_list_path = tmp_path / "gene_list.txt"

    full_tpm = np.array([[250000.0, 250000.0, 500000.0]], dtype=np.float32)
    full_log = np.log2(full_tpm + 1.0).astype(np.float32)
    _write_h5ad(
        h5ad_path,
        x=full_log,
        obs_names=["sample_a"],
        var_names=["gene_a", "gene_b", "gene_c"],
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_a"]),
    )
    gene_list_path.write_text("gene_b\ngene_c\n", encoding="utf-8")

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[h5ad_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            remove_low_var_genes=False,
            gene_list_file=gene_list_path,
        )
    )

    expected = non_log2log_cpm(
        pd.DataFrame([[250000.0, 500000.0]], index=["sample_a"], columns=["gene_b", "gene_c"]),
        transpose=False,
    )

    assert dataset.gene_list == ["gene_b", "gene_c"]
    np.testing.assert_allclose(dataset.data[0], expected.iloc[0].to_numpy(dtype=np.float32), rtol=1e-4, atol=1e-3)


def test_gepdataset_honors_configured_parallel_load_worker_limit(tmp_path: Path, monkeypatch):
    genes = ["gene_a", "gene_b"]
    bulk_paths = [tmp_path / f"bulk_{i}.h5ad" for i in range(3)]
    mapping_paths = [tmp_path / f"sample2cell_{i}.csv" for i in range(3)]
    ref_sct_path = tmp_path / "shared_ref_sct.h5ad"

    for idx, (bulk_path, mapping_path) in enumerate(zip(bulk_paths, mapping_paths, strict=True), start=1):
        _write_h5ad(
            bulk_path,
            x=np.array([[float(idx), float(idx + 1)]], dtype=np.float32),
            obs_names=[f"sample_{idx}"],
            var_names=genes,
            obs=pd.DataFrame({"CT1": [1.0]}, index=[f"sample_{idx}"]),
        )
        pd.DataFrame(
            {"cell_type": ["CT1"], "selected_cell_id": [f"cell_{idx}"]},
            index=[f"sample_{idx}"],
        ).to_csv(mapping_path)

    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_1", "cell_2", "cell_3"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_1", "cell_2", "cell_3"]),
    )

    recorded_workers: list[int] = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            recorded_workers.append(int(max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            return [func(item) for item in iterable]

    monkeypatch.setattr(datasets_module, "ThreadPoolExecutor", RecordingExecutor)

    GEPDataset(
        GEPDatasetConfig(
            file_paths=bulk_paths,
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            remove_low_var_genes=False,
            max_parallel_source_file_loads=2,
            training_target_sets={
                f"Train_set{i}": {
                    "training_set_file_path": bulk_path,
                    "training_set_sample2cell_id_file_path": mapping_path,
                    "training_sct_gep_file_path": ref_sct_path,
                }
                for i, (bulk_path, mapping_path) in enumerate(zip(bulk_paths, mapping_paths, strict=True), start=1)
            },
        )
    )

    assert recorded_workers == [2]


def test_gepdataset_cleans_up_temporary_group_intermediates(tmp_path: Path):
    genes = ["gene_a", "gene_b"]
    bulk_path = tmp_path / "bulk.h5ad"

    _write_h5ad(
        bulk_path,
        x=np.array([[2.0, 4.0]], dtype=np.float32),
        obs_names=["sample_1"],
        var_names=genes,
        obs=pd.DataFrame({"CT1": [1.0]}, index=["sample_1"]),
    )

    processed_dir = tmp_path / "processed"
    GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_path],
            processed_data_dir=processed_dir,
            force_reprocess=True,
            scaling_by_constant=False,
            remove_low_var_genes=False,
        )
    )

    assert not (processed_dir / "_tmp_preprocess").exists()


def test_gepdataset_writes_true_sct_gep_directly_to_npy_cache(tmp_path: Path):
    genes = ["gene_a", "gene_b"]
    bulk_path = tmp_path / "bulk.h5ad"
    ref_sct_path = tmp_path / "ref_sct.h5ad"
    mapping_path = tmp_path / "sample2cell.csv"
    processed_dir = tmp_path / "processed"

    _write_h5ad(
        bulk_path,
        x=np.array([[2.0, 4.0], [3.0, 9.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame(
            {
                "CT1": [0.7, 0.8],
                "CT2": [0.3, 0.2],
            },
            index=["sample_1", "sample_2"],
        ),
    )
    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
    )
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b", "cell_c"],
        },
        index=["sample_1", "sample_1", "sample_2"],
    ).to_csv(mapping_path)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_path],
            processed_data_dir=processed_dir,
            force_reprocess=True,
            scaling_by_constant=False,
            cache_compress=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_path,
                    "training_set_sample2cell_id_file_path": mapping_path,
                    "training_sct_gep_file_path": ref_sct_path,
                }
            },
        )
    )

    assert (processed_dir / "true_sct_gep.npy").exists()
    assert (processed_dir / "true_sct_gep_present_mask.npy").exists()
    assert dataset.true_sct_gep.shape == (2, 2, 2)
    assert dataset.true_sct_gep_present_mask.tolist() == [[True, True], [True, False]]


def test_gepdataset_builds_matched_targets_without_align_with_gene_list(tmp_path: Path, monkeypatch):
    genes = ["gene_a", "gene_b"]
    bulk_path = tmp_path / "bulk.h5ad"
    ref_sct_path = tmp_path / "ref_sct.h5ad"
    mapping_path = tmp_path / "sample2cell.csv"

    _write_h5ad(
        bulk_path,
        x=np.array([[2.0, 4.0], [3.0, 9.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame(
            {
                "CT1": [0.7, 0.8],
                "CT2": [0.3, 0.2],
            },
            index=["sample_1", "sample_2"],
        ),
    )
    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
    )
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b", "cell_c"],
        },
        index=["sample_1", "sample_1", "sample_2"],
    ).to_csv(mapping_path)

    def _unexpected_align(*_args, **_kwargs):
        raise AssertionError("align_with_gene_list should not be used in the optimized matched-target path.")

    monkeypatch.setattr(ReadExp, "align_with_gene_list", _unexpected_align)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_path],
            processed_data_dir=tmp_path / "processed",
            force_reprocess=True,
            scaling_by_constant=False,
            remove_low_var_genes=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_path,
                    "training_set_sample2cell_id_file_path": mapping_path,
                    "training_sct_gep_file_path": ref_sct_path,
                }
            },
        )
    )

    assert dataset.true_sct_gep.shape == (2, 2, 2)
    assert dataset.true_sct_gep_present_mask.tolist() == [[True, True], [True, False]]


def test_gepdataset_scales_true_sct_gep_in_place_for_npy_cache(tmp_path: Path):
    genes = ["gene_a", "gene_b"]
    bulk_path = tmp_path / "bulk.h5ad"
    ref_sct_path = tmp_path / "ref_sct.h5ad"
    mapping_path = tmp_path / "sample2cell.csv"
    processed_dir = tmp_path / "processed"

    _write_h5ad(
        bulk_path,
        x=np.array([[2.0, 4.0], [3.0, 9.0]], dtype=np.float32),
        obs_names=["sample_1", "sample_2"],
        var_names=genes,
        obs=pd.DataFrame(
            {
                "CT1": [0.7, 0.8],
                "CT2": [0.3, 0.2],
            },
            index=["sample_1", "sample_2"],
        ),
    )
    _write_h5ad(
        ref_sct_path,
        x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        obs_names=["cell_a", "cell_b", "cell_c"],
        var_names=genes,
        obs=pd.DataFrame(index=["cell_a", "cell_b", "cell_c"]),
    )
    pd.DataFrame(
        {
            "cell_type": ["CT1", "CT2", "CT1"],
            "selected_cell_id": ["cell_a", "cell_b", "cell_c"],
        },
        index=["sample_1", "sample_1", "sample_2"],
    ).to_csv(mapping_path)

    dataset = GEPDataset(
        GEPDatasetConfig(
            file_paths=[bulk_path],
            processed_data_dir=processed_dir,
            force_reprocess=True,
            scaling_by_constant=True,
            scaling_factor=2.0,
            cache_compress=False,
            training_target_sets={
                "Train_set1": {
                    "training_set_file_path": bulk_path,
                    "training_set_sample2cell_id_file_path": mapping_path,
                    "training_sct_gep_file_path": ref_sct_path,
                }
            },
        )
    )

    aligned = build_matched_sct_gep_training_targets(
        sct_gep_dataset_file_path=ref_sct_path,
        sample2cell_id_file_path=mapping_path,
        bulk_sample_ids=["sample_1", "sample_2"],
        target_gene_list=genes,
        cell_types=["CT1", "CT2"],
        cache_sct_query_results=False,
    )["true_sct_gep"]

    np.testing.assert_allclose(dataset.true_sct_gep, aligned / 2.0, rtol=1e-6, atol=1e-6)


def test_matched_sct_gep_loss_masks_low_prop_cell_types():
    vae = VAE.__new__(VAE)
    vae.scaling_factor = 1.0

    recon_cpm = torch.tensor(
        [[[3.0, 100.0], [7.0, 100.0]]],
        dtype=torch.float32,
    )
    true_sct_gep = torch.tensor(
        [[[2.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float32,
    )
    true_sct_gep[:, :, 0] = to_log_space(recon_cpm[:, :, 0], scaling_factor=1.0)
    true_sct_gep[:, :, 1] = 0.0
    true_sct_gep_present_mask = torch.tensor([[True, True]])
    true_cell_prop = torch.tensor([[0.8, 0.01]], dtype=torch.float32)

    loss = vae._matched_sct_gep_supervision_loss(
        recon_x_all_types_cpm=recon_cpm,
        true_sct_gep=true_sct_gep,
        true_sct_gep_present_mask=true_sct_gep_present_mask,
        true_cell_prop=true_cell_prop,
        cell_prop_threshold=0.1,
    )

    assert torch.allclose(loss, torch.zeros_like(loss))

    true_sct_gep[:, :, 0] = true_sct_gep[:, :, 0] + 1.0
    loss_with_active_error = vae._matched_sct_gep_supervision_loss(
        recon_x_all_types_cpm=recon_cpm,
        true_sct_gep=true_sct_gep,
        true_sct_gep_present_mask=true_sct_gep_present_mask,
        true_cell_prop=true_cell_prop,
        cell_prop_threshold=0.1,
    )

    assert torch.allclose(loss_with_active_error, torch.tensor([1.0], dtype=torch.float32))


def test_loss_function_skips_matched_sct_gep_supervision_during_inference():
    vae = VAE.__new__(VAE)
    vae.training = False
    vae.scaling_factor = 1.0
    vae.data_config = SimpleNamespace(training_sct_gep_cell_prop_threshold=0.1)
    vae.model_config = SimpleNamespace(
        predict_cell_prop=False,
        learn_gep_residual=False,
        loss_coefficient=SimpleNamespace(
            beta=1.0,
            gamma=0.0,
            attractor_weight=0.0,
            z_score_reg_weight=0.0,
            z_score_kl_weight=0.0,
            low_mean_std_weight=0.0,
            low_mean_threshold=2.0,
            low_std_threshold=1.0,
            cross_sample_gene_var_weight=0.0,
            cell_type_sct_gep_weight=1.0,
            kld_p=0.0,
            cell_prop=0.0,
            hierarchical_code_weight=0.0,
        ),
    )
    vae.g_mean_non_log = torch.ones((2, 1), dtype=torch.float32)
    vae.g_std_non_log = torch.ones((2, 1), dtype=torch.float32)
    vae._reconstruction_loss = lambda x, recon_x_conv: torch.zeros((x.shape[0],), device=x.device)
    vae._gene_statistics_loss = lambda recon_gene_mean, recon_gene_std, device: (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    vae._cell_prop_dirichlet_loss = lambda y, dd_alpha, pred_cell_prop, batch_size, device: (
        torch.zeros((batch_size,), device=device),
        torch.zeros((batch_size,), device=device),
    )
    vae._latent_kld_loss = lambda mu_types, logvar_types, mu_prior, logvar_mean, mu_mean, device: (
        torch.zeros((mu_types.shape[0],), device=device)
    )
    vae._repulsion_loss = lambda mu_types, gamma: torch.zeros((mu_types.shape[0],), device=mu_types.device)
    vae._attractor_loss = lambda mu_types, attractor_weight: torch.zeros((mu_types.shape[0],), device=mu_types.device)
    vae._hierarchical_code_loss = lambda mu_types, hierarchical_code_weight: (
        torch.zeros((mu_types.shape[0],), device=mu_types.device)
    )
    vae._matched_sct_gep_supervision_loss = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("inference should not call matched sctGEP supervision")
    )

    x = torch.zeros((2, 2), dtype=torch.float32)
    mu_types = torch.zeros((2, 3, 1), dtype=torch.float32)
    logvar_types = torch.zeros((2, 3, 1), dtype=torch.float32)
    recon_gene_mean = torch.ones((2, 1), dtype=torch.float32)
    recon_gene_std = torch.ones((2, 1), dtype=torch.float32)
    recon_x_all_types_cpm = torch.ones((2, 2, 1), dtype=torch.float32)

    loss_terms = vae.loss_function(
        x=x,
        y=None,
        recon_x_conv=x,
        mu_types=mu_types,
        logvar_types=logvar_types,
        pred_cell_prop=None,
        existence_logits=None,
        dd_alpha=None,
        mu_prior=torch.zeros((1, 3), dtype=torch.float32),
        recon_gene_mean=recon_gene_mean,
        recon_gene_std=recon_gene_std,
        logvar_mean=torch.zeros((2, 3), dtype=torch.float32),
        mu_mean=torch.zeros((2, 3), dtype=torch.float32),
        device=x.device,
        recon_x_all_types_cpm=recon_x_all_types_cpm,
        true_sct_gep=None,
        true_sct_gep_present_mask=None,
    )

    assert torch.allclose(
        loss_terms.cell_type_sct_gep,
        torch.tensor(0.0, dtype=torch.float32),
    )


def test_per_sample_residual_variance_loss_matches_log_variance_targets():
    vae = VAE.__new__(VAE)
    vae.training = True
    vae.cell_types = ["CT1", "CT2"]
    vae.training_sct_per_sample_residual_var_by_sample = {
        "sample_1": np.array([1.0, 4.0], dtype=np.float32),
        "sample_2": np.array([9.0, np.nan], dtype=np.float32),
    }

    pred_residual_log = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 4.0]],
            [[1.0, 2.0], [5.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    true_sct_gep_present_mask = torch.tensor(
        [
            [True, True],
            [True, False],
        ],
        dtype=torch.bool,
    )
    true_cell_prop = torch.ones((2, 2), dtype=torch.float32)

    loss = vae._per_sample_residual_variance_loss(
        pred_residual_log=pred_residual_log,
        sample_ids=["sample_1", "sample_2"],
        true_sct_gep_present_mask=true_sct_gep_present_mask,
        true_cell_prop=true_cell_prop,
        cell_prop_threshold=0.1,
    )

    assert torch.allclose(
        loss,
        torch.tensor([0.0, np.log(9.0 / 4.0)], dtype=torch.float32),
        atol=1e-6,
    )


def test_inter_sample_similarity_loss_zero_when_prediction_matches_truth():
    vae = VAE.__new__(VAE)
    vae.g_mean = torch.zeros((2, 2), dtype=torch.float32)

    pred_residual_log = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[1.0, 1.0], [2.0, 3.0]],
            [[2.0, 5.0], [4.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    true_sct_gep = pred_residual_log.clone()
    true_sct_gep_present_mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
        ],
        dtype=torch.bool,
    )
    true_cell_prop = torch.tensor(
        [
            [0.2, 0.2],
            [0.2, 0.001],
            [0.2, 0.2],
        ],
        dtype=torch.float32,
    )

    loss = vae._inter_sample_similarity_loss(
        pred_residual_log=pred_residual_log,
        true_sct_gep=true_sct_gep,
        true_sct_gep_present_mask=true_sct_gep_present_mask,
        true_cell_prop=true_cell_prop,
        cell_prop_threshold=0.01,
    )

    assert torch.allclose(loss, torch.zeros(3, dtype=torch.float32), atol=1e-6)


def test_pairwise_cosine_similarity_matrix_matches_expected_geometry():
    vae = VAE.__new__(VAE)
    residuals = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    matrix = vae._pairwise_cosine_similarity_matrix(residuals)

    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0 / np.sqrt(2.0)],
            [0.0, 1.0, 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(matrix, expected, atol=1e-6)


def test_inter_sample_similarity_loss_masks_low_prop_and_skips_singleton_cell_types():
    vae = VAE.__new__(VAE)
    vae.g_mean = torch.zeros((2, 2), dtype=torch.float32)

    true_sct_gep = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[2.0, 2.0], [0.0, 1.0]],
            [[0.0, 3.0], [1.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    pred_residual_log = true_sct_gep.clone()
    pred_residual_log[:, :, 0] = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    true_sct_gep_present_mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
        ],
        dtype=torch.bool,
    )
    true_cell_prop = torch.tensor(
        [
            [0.2, 0.2],
            [0.2, 0.001],
            [0.2, 0.2],
        ],
        dtype=torch.float32,
    )

    loss = vae._inter_sample_similarity_loss(
        pred_residual_log=pred_residual_log,
        true_sct_gep=true_sct_gep,
        true_sct_gep_present_mask=true_sct_gep_present_mask,
        true_cell_prop=true_cell_prop,
        cell_prop_threshold=0.01,
    )

    assert torch.all(loss > 0)
    assert loss.shape == (3,)
