from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from vaedecon.configs.default_config import GEPDatasetConfig
from vaedecon.data.datasets import GEPDataset
from vaedecon.data.datasets import build_matched_sct_gep_training_targets
from vaedecon.models.vae.vae_model import VAE, to_log_space


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

    assert loss_with_active_error.item() > 0
