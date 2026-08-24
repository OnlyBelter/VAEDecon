from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from vaedecon.configs import VAEDeconConfig, LossCoefficient
from vaedecon.workflow.inference import VAEDeconPredictor
from vaedecon.workflow.train import VAEDeconTrainer, train_vaedecon


class _DummyDebugDataset:
    def __init__(self, n_samples: int):
        self._n_samples = n_samples
        self._sample_ids = [f"sample_{i}" for i in range(n_samples)]

    def __len__(self):
        return self._n_samples

    def get_sample_ids(self):
        return self._sample_ids


class _DummyMatchedSCTDataset:
    def __init__(self, *, sample_ids, gene_list, cell_types, true_sct_gep, true_sct_gep_present_mask):
        self._sample_ids = [str(sample_id) for sample_id in sample_ids]
        self.gene_list = list(gene_list)
        self.cell_types = list(cell_types)
        self.true_sct_gep = np.asarray(true_sct_gep, dtype=np.float32)
        self.true_sct_gep_present_mask = np.asarray(true_sct_gep_present_mask, dtype=bool)

    def get_sample_ids(self):
        return self._sample_ids


def test_trainer_prefers_dedicated_sct_reference_for_gene_mean_std(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/test_set_sct_gep.h5ad",
                "gene_mean_std_sct_gep_file_path": "./datasets/dedicated_gene_mean_std_ref.h5ad",
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)

    assert trainer._resolve_gene_mean_std_sct_gep_paths() == [Path(
        "./datasets/dedicated_gene_mean_std_ref.h5ad"
    )]


def test_trainer_falls_back_to_all_training_sct_paths_for_gene_mean_std(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_file_path": [
                    "./datasets/train_sct_a.h5ad",
                    "./datasets/train_sct_b.h5ad",
                ],
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)

    assert trainer._resolve_gene_mean_std_sct_gep_paths() == [
        Path("./datasets/train_sct_a.h5ad"),
        Path("./datasets/train_sct_b.h5ad"),
    ]


def test_trainer_prefers_training_sct_paths_over_test_set_sct_reference(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/test_set_sct_gep.h5ad",
                "sct_file_path": [
                    "./datasets/train_sct_a.h5ad",
                    "./datasets/train_sct_b.h5ad",
                ],
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)

    assert trainer._resolve_gene_mean_std_sct_gep_paths() == [
        Path("./datasets/train_sct_a.h5ad"),
        Path("./datasets/train_sct_b.h5ad"),
    ]


def test_trainer_keeps_gene_mean_std_output_under_model_dir(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/test_set_sct_gep.h5ad",
                "scaling_by_constant": True,
                "scaling_factor": 20.0,
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)

    assert trainer._build_gene_mean_std_output_path() == (
        tmp_path / "final_model" / "gene_mean_std_log2p1_scaled_by_20.0.csv"
    )


def test_loss_coefficient_defaults_and_validation_cross_sample_gene_var_weight():
    lo = LossCoefficient()
    assert lo.cross_sample_gene_var_weight == 0.0

    lo = LossCoefficient(cross_sample_gene_var_weight=2.5)
    assert lo.cross_sample_gene_var_weight == 2.5

    with pytest.raises(Exception):
        LossCoefficient(cross_sample_gene_var_weight=-1.0)


def test_loss_coefficient_defaults_and_validation_per_sample_residual_var_weight():
    lo = LossCoefficient()
    assert lo.per_sample_residual_var_weight == 0.0

    lo = LossCoefficient(per_sample_residual_var_weight=2.5)
    assert lo.per_sample_residual_var_weight == 2.5

    with pytest.raises(Exception):
        LossCoefficient(per_sample_residual_var_weight=-1.0)


def test_loss_coefficient_defaults_and_validation_inter_sample_similarity_weight():
    lo = LossCoefficient()
    assert lo.inter_sample_similarity_weight == 0.0

    lo = LossCoefficient(inter_sample_similarity_weight=2.5)
    assert lo.inter_sample_similarity_weight == 2.5

    with pytest.raises(Exception):
        LossCoefficient(inter_sample_similarity_weight=-1.0)


def test_trainer_cross_sample_gene_var_output_path_naming(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/test_set_sct_gep.h5ad",
                "scaling_by_constant": True,
                "scaling_factor": 20.0,
            },
            "model": {
                "model_dir": tmp_path / "final_model",
                "loss_coefficient": {"cross_sample_gene_var_weight": 0.0},
            },
        }
    )
    trainer = VAEDeconTrainer(config=config)
    assert trainer._build_training_sct_cross_sample_gene_var_output_path() == (
        tmp_path / "final_model" / "training_sct_cross_sample_gene_variances_log2p1_scaled_by_20.0.csv"
    )


def test_trainer_per_sample_residual_var_output_path_naming(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/test_set_sct_gep.h5ad",
                "scaling_by_constant": True,
                "scaling_factor": 20.0,
            },
            "model": {
                "model_dir": tmp_path / "final_model",
                "learn_gep_residual": True,
                "learn_gep_residual_mode": "mean_centered",
                "loss_coefficient": {"per_sample_residual_var_weight": 0.0},
            },
        }
    )
    trainer = VAEDeconTrainer(config=config)
    assert trainer._build_training_sct_per_sample_residual_var_output_path() == (
        tmp_path / "final_model" / "training_sct_per_sample_residual_variance_log2p1_scaled_by_20.0.csv"
    )


def test_trainer_saves_per_sample_residual_variance_targets(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    gene_mean_std_fp = model_dir / "gene_mean_std_log2p1_scaled_by_20.0.csv"
    gene_mean_std_fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "CT1_avg": [1.0, 2.0],
            "CT1_std": [0.1, 0.2],
            "CT2_avg": [0.0, 0.0],
            "CT2_std": [0.1, 0.1],
        },
        index=["g1", "g2"],
    ).to_csv(gene_mean_std_fp, float_format="%g")

    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/test_set_sct_gep.h5ad",
                "scaling_by_constant": True,
                "scaling_factor": 20.0,
            },
            "model": {
                "model_dir": model_dir,
                "learn_gep_residual": True,
                "learn_gep_residual_mode": "mean_centered",
                "loss_coefficient": {"per_sample_residual_var_weight": 1.0},
            },
        }
    )
    trainer = VAEDeconTrainer(config=config)
    trainer.config.model.gene_mean_std_fp = gene_mean_std_fp

    dataset = _DummyMatchedSCTDataset(
        sample_ids=["sample_1", "sample_2"],
        gene_list=["g1", "g2"],
        cell_types=["CT1", "CT2"],
        true_sct_gep=np.array(
            [
                [[2.0, 0.0], [4.0, 0.0]],
                [[1.0, 1.0], [2.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        true_sct_gep_present_mask=np.array(
            [
                [True, False],
                [True, True],
            ],
            dtype=bool,
        ),
    )

    out_fp = trainer._prepare_and_save_training_sct_per_sample_residual_var(dataset)
    out_df = pd.read_csv(out_fp, index_col=0)

    assert list(out_df.index) == ["sample_1", "sample_2"]
    assert list(out_df.columns) == ["CT1", "CT2"]
    assert out_df.loc["sample_1", "CT1"] == pytest.approx(0.25)
    assert pd.isna(out_df.loc["sample_1", "CT2"])
    assert out_df.loc["sample_2", "CT1"] == pytest.approx(0.0)
    assert out_df.loc["sample_2", "CT2"] == pytest.approx(1.0)


def test_trainer_rejects_unused_training_target_sets_early(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                "simu_bulk_file_path": [
                    "./datasets/segment_bulk.h5ad",
                ],
                "training_target_sets": {
                    "Train_set1": {
                        "training_set_file_path": "./datasets/random_bulk.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/random_sample2cell.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                    },
                    "Train_set4": {
                        "training_set_file_path": "./datasets/segment_bulk.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/segment_sample2cell.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                    },
                },
            },
            "model": {
                "model_dir": tmp_path / "final_model",
                "loss_coefficient": {"cell_type_sct_gep_weight": 1.0},
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)

    with pytest.raises(
        ValueError,
        match="data.training_target_sets contains bulk files that are not present in data.simu_bulk_file_path",
    ):
        trainer._build_gepdataset_config()


def test_trainer_uses_training_target_set_bulk_paths_when_simu_paths_missing(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                "training_target_sets": {
                    "Train_set1": {
                        "training_set_file_path": "./datasets/random_bulk.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/random_sample2cell.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                    },
                    "Train_set4": {
                        "training_set_file_path": "./datasets/segment_bulk.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/segment_sample2cell.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                    },
                },
            },
            "model": {
                "model_dir": tmp_path / "final_model",
                "loss_coefficient": {"cell_type_sct_gep_weight": 1.0},
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)
    dataset_config = trainer._build_gepdataset_config()

    assert [str(path) for path in dataset_config.file_paths] == [
        "./datasets/random_bulk.h5ad",
        "./datasets/segment_bulk.h5ad",
    ]


def test_trainer_rejects_duplicate_training_target_set_bulk_paths(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                "training_target_sets": {
                    "Train_set1": {
                        "training_set_file_path": "./datasets/shared_bulk.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/shared_sample2cell_a.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                    },
                    "Train_set2": {
                        "training_set_file_path": "./datasets/shared_bulk.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/shared_sample2cell_b.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                    },
                },
            },
            "model": {
                "model_dir": tmp_path / "final_model",
                "loss_coefficient": {"cell_type_sct_gep_weight": 1.0},
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)

    with pytest.raises(
        ValueError,
        match="data.training_target_sets must not reuse the same training_set_file_path",
    ):
        trainer._build_gepdataset_config()


def test_inference_builds_dataset_config_with_sct_gene_mean_std_refs(tmp_path: Path):
    test_set_fp = tmp_path / "test_set.h5ad"
    test_set_fp.write_text("not-a-real-h5ad")

    model_dir = tmp_path / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/legacy_sct_gep.h5ad",
                "gene_mean_std_sct_gep_file_path": "./datasets/dedicated_gene_mean_std_ref.h5ad",
                "pooled_sc_h5ad_path": "./datasets/merged_sc.h5ad",
                "pooled_sc_cell_type_col": "cell_type",
                "pooled_sc_cell_subtype_col": "cell_subtype",
                "pooled_sc_sample_size": 1,
                "pooled_sc_seed": 42,
            },
            "model": {
                "model_dir": model_dir,
            },
        }
    )

    predictor = VAEDeconPredictor(
        model_dir=str(model_dir),
        config=config,
        device="cpu",
    )

    dataset_cfg = predictor._build_gepdataset_config(
        data_file_path=str(test_set_fp),
        dataset_type="test",
    )

    assert dataset_cfg.gene_mean_std_source == "sct_gep"
    assert dataset_cfg.gene_mean_std_sct_gep_file_path == Path(
        "./datasets/dedicated_gene_mean_std_ref.h5ad"
    )
    assert dataset_cfg.sct_gep_file_path == Path("./datasets/legacy_sct_gep.h5ad")
    assert dataset_cfg.pooled_sc_h5ad_path == Path("./datasets/merged_sc.h5ad")
    assert dataset_cfg.pooled_sc_cell_type_col == "cell_type"
    assert dataset_cfg.pooled_sc_cell_subtype_col == "cell_subtype"
    assert dataset_cfg.pooled_sc_sample_size == 1
    assert dataset_cfg.pooled_sc_seed == 42


def test_trainer_builds_reproducible_debug_overfit_subset_and_manifest(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
            },
            "training": {
                "debug_overfit": {
                    "enabled": True,
                    "subset_size": 5,
                    "subset_seed": 7,
                    "use_training_subset_as_eval": True,
                }
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)
    dataset = _DummyDebugDataset(n_samples=20)

    train_subset, eval_subset = trainer._build_debug_overfit_subsets(dataset)

    expected_indices = torch.randperm(20, generator=torch.Generator().manual_seed(7))[:5].tolist()
    expected_indices = sorted(int(idx) for idx in expected_indices)

    assert train_subset.indices == expected_indices
    assert eval_subset is train_subset

    manifest_path = tmp_path / "final_model" / "debug_overfit_subset.csv"
    manifest_df = pd.read_csv(manifest_path)
    assert manifest_df["debug_subset_row_index"].tolist() == expected_indices
    assert manifest_df["sample_id"].tolist() == [f"sample_{i}" for i in expected_indices]


def test_trainer_rejects_debug_overfit_subset_larger_than_dataset(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
            },
            "training": {
                "debug_overfit": {
                    "enabled": True,
                    "subset_size": 100,
                }
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )

    trainer = VAEDeconTrainer(config=config)
    dataset = _DummyDebugDataset(n_samples=10)

    with pytest.raises(ValueError, match="debug_overfit.subset_size"):
        trainer._build_debug_overfit_subsets(dataset)


def test_debug_overfit_mode_can_seed_generated_test_set_config(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/train_sct_ref.h5ad",
                "training_target_sets": {
                    "Train_set1": {
                        "training_set_file_path": str(tmp_path / "train1.h5ad"),
                        "training_set_sample2cell_id_file_path": str(tmp_path / "train1_sample2cell.csv"),
                        "training_sct_gep_file_path": str(tmp_path / "train1_sct.h5ad"),
                    }
                },
            },
            "training": {
                "debug_overfit": {
                    "enabled": True,
                    "subset_size": 2,
                    "subset_seed": 1,
                    "use_training_subset_as_eval": True,
                }
            },
            "model": {
                "model_dir": tmp_path / "final_model",
            },
        }
    )
    trainer = VAEDeconTrainer(config=config)

    class _NamespacedDataset(_DummyDebugDataset):
        def __init__(self):
            super().__init__(n_samples=3)
            self._sample_ids = [
                "Train_set1::sample_a",
                "Train_set1::sample_b",
                "Train_set1::sample_c",
            ]
            self.cell_prop = pd.DataFrame(
                [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]],
                index=self._sample_ids,
                columns=["A", "B"],
            )

        def get_cell_prop(self):
            return self.cell_prop

    dataset = _NamespacedDataset()
    bulk_df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=["sample_a", "sample_b", "sample_c"],
        columns=["g1", "g2"],
    )
    from anndata import AnnData
    AnnData(
        X=bulk_df.values.astype("float32"),
        obs=pd.DataFrame(dataset.cell_prop.values, index=bulk_df.index, columns=dataset.cell_prop.columns),
        var=pd.DataFrame(index=bulk_df.columns),
    ).write_h5ad(tmp_path / "train1.h5ad")
    pd.DataFrame(
        {"cell_type": ["A", "B"], "selected_cell_id": ["c1", "c2"]},
        index=["sample_a", "sample_b"],
    ).to_csv(tmp_path / "train1_sample2cell.csv")
    AnnData(
        X=bulk_df.values[:2].astype("float32"),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=bulk_df.columns),
    ).write_h5ad(tmp_path / "train1_sct.h5ad")

    train_subset = torch.utils.data.Subset(dataset, [0, 1])
    trainer._prepare_debug_overfit_test_sets(dataset=dataset, train_set=train_subset)

    assert "Debug_overfit_Train_set1" in trainer.config.data.test_sets
    debug_test = trainer.config.data.test_sets["Debug_overfit_Train_set1"]
    assert Path(debug_test.test_set_file_path).exists()
    assert Path(debug_test.test_set_sample2cell_id_file_path).exists()
    assert str(debug_test.sct_gep_file_path).endswith("train1_sct.h5ad")


def test_train_vaedecon_reuses_saved_config_with_test_sets_when_checkpoint_exists(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best_model.ckpt").write_text("placeholder", encoding="utf-8")

    saved_cfg = VAEDeconConfig.from_dict(
        {
            "data": {
                "test_sets": {
                    "Debug_overfit_Train_set2": {
                        "test_set_file_path": str(tmp_path / "debug_subset.h5ad"),
                        "test_set_sample2cell_id_file_path": str(tmp_path / "debug_subset_sample2cell.csv"),
                        "sct_gep_file_path": str(tmp_path / "debug_subset_sct.h5ad"),
                    }
                }
            },
            "model": {
                "model_dir": model_dir,
            },
        }
    )
    saved_cfg.to_yaml(model_dir / "config.yaml")

    input_cfg = VAEDeconConfig.from_dict(
        {
            "model": {
                "model_dir": model_dir,
            }
        }
    )

    returned_cfg = train_vaedecon(config=input_cfg)

    assert "Debug_overfit_Train_set2" in returned_cfg.data.test_sets
    assert returned_cfg.data.test_set_file_path == Path(tmp_path / "debug_subset.h5ad")


def test_compute_training_sct_cross_sample_gene_var_roundtrip(tmp_path: Path, monkeypatch):
    """Smoke test for the variance CSV export: verify shape, columns, gene index order."""
    pytest.importorskip("anndata")
    import anndata as an

    from vaedecon.utility import compute_training_sct_cross_sample_gene_var

    genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
    cell_types = ["CT1", "CT2"]
    n_sct_per_ct = 20

    obs_rows = []
    x_rows = []
    rng = np.random.default_rng(0)
    for i, ct in enumerate(cell_types):
        for k in range(n_sct_per_ct):
            # Vary baseline per sample so cross-sample var is not zero
            baseline = 0.5 + 0.1 * k
            vec = np.clip(
                rng.normal(loc=1.0 + i + baseline, scale=0.25, size=len(genes)),
                1e-2,
                None,
            )
            x_rows.append(2 ** vec - 1)  # non-log space for log-space var
            row = {c: 1 if c == ct else 0 for c in cell_types}
            obs_rows.append(row)

    x = np.asarray(x_rows, dtype=np.float32)
    obs = pd.DataFrame(
        obs_rows,
        index=pd.Index([f"r{i}" for i in range(len(x_rows))], dtype=object),
    )
    adata = an.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=object)))
    sct_fp = tmp_path / "sct.h5ad"
    adata.write_h5ad(sct_fp)

    gene_list_fp = tmp_path / "genes.txt"
    gene_list_fp.write_text("".join(f"{g}\n" for g in genes))
    cell_type_fp = tmp_path / "cts.txt"
    cell_type_fp.write_text("".join(f"{c}\n" for c in cell_types))

    out_fp = tmp_path / "out_var.csv"
    compute_training_sct_cross_sample_gene_var(
        sct_dataset_fp=sct_fp,
        result_fp=out_fp,
        gene_list_fp=gene_list_fp,
        cell_type_fp=cell_type_fp,
        scaling_by_constant=False,
        log2p1=True,
        scaling_factor=20.0,
    )

    df = pd.read_csv(out_fp, index_col=0)
    assert list(df.index) == genes
    assert list(df.columns) == [f"{c}_var" for c in cell_types]
    assert df.shape == (len(genes), len(cell_types))
    # Variance must be non-negative and finite
    assert np.all(np.isfinite(df.values))
    assert np.all(df.values >= 0.0)


def test_compute_training_sct_cross_sample_gene_var_pools_multiple_sct_files(tmp_path: Path):
    pytest.importorskip("anndata")
    import anndata as an

    from vaedecon.utility import compute_training_sct_cross_sample_gene_var
    from vaedecon.utility.read_file import ReadExp

    genes = ["GeneA", "GeneB"]
    cell_types = ["CT1", "CT2"]

    def _write_sct(fp: Path, ct1_rows: list[list[float]], ct2_rows: list[list[float]]):
        obs_rows = []
        x_rows = []
        for rows, ct in [(ct1_rows, "CT1"), (ct2_rows, "CT2")]:
            for vec in rows:
                # The helper expects SCT matrices in log-space and internally
                # converts them back to CPM via ReadExp(..., exp_type="log_space").to_tpm().
                x_rows.append(np.log2(np.asarray(vec, dtype=np.float32) + 1.0))
                obs_rows.append({c: 1 if c == ct else 0 for c in cell_types})
            adata = an.AnnData(
                X=np.asarray(x_rows, dtype=np.float32),
                obs=pd.DataFrame(
                    obs_rows,
                    index=pd.Index([f"{fp.stem}_{i}" for i in range(len(x_rows))], dtype=object),
                ),
                var=pd.DataFrame(index=pd.Index(genes, dtype=object)),
            )
        adata.write_h5ad(fp)

    sct_fp_a = tmp_path / "sct_a.h5ad"
    sct_fp_b = tmp_path / "sct_b.h5ad"
    _write_sct(sct_fp_a, ct1_rows=[[1.0, 3.0], [2.0, 4.0]], ct2_rows=[[5.0, 7.0], [6.0, 8.0]])
    _write_sct(sct_fp_b, ct1_rows=[[3.0, 5.0], [4.0, 6.0]], ct2_rows=[[7.0, 9.0], [8.0, 10.0]])

    gene_list_fp = tmp_path / "genes.txt"
    gene_list_fp.write_text("".join(f"{g}\n" for g in genes))
    cell_type_fp = tmp_path / "cts.txt"
    cell_type_fp.write_text("".join(f"{c}\n" for c in cell_types))

    out_fp = tmp_path / "out_var_multi.csv"
    compute_training_sct_cross_sample_gene_var(
        sct_dataset_fp=[sct_fp_a, sct_fp_b],
        result_fp=out_fp,
        gene_list_fp=gene_list_fp,
        cell_type_fp=cell_type_fp,
        scaling_by_constant=False,
        log2p1=False,
        scaling_factor=20.0,
    )

    df = pd.read_csv(out_fp, index_col=0)
    def _expected_var(rows: list[list[float]]) -> np.ndarray:
        x_df = pd.DataFrame(
            np.log2(np.asarray(rows, dtype=np.float32) + 1.0),
            columns=genes,
        )
        exp_obj = ReadExp(x_df, exp_type="log_space")
        exp_obj.align_with_gene_list(gene_list=genes, fill_not_exist=True, pathway_list=True)
        exp_obj.to_tpm()
        exp = exp_obj.get_exp().values.astype(np.float64)
        return np.var(exp, axis=0, ddof=1)

    expected_ct1 = _expected_var([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]])
    expected_ct2 = _expected_var([[5.0, 7.0], [6.0, 8.0], [7.0, 9.0], [8.0, 10.0]])
    assert df["CT1_var"].values == pytest.approx(expected_ct1)
    assert df["CT2_var"].values == pytest.approx(expected_ct2)


def test_vae_cross_sample_gene_variance_loss_math():
    """Ensure the loss math matches: mean |recon_var - target_var| scaled per-sample repeat."""
    import torch
    from unittest.mock import MagicMock

    from vaedecon.models.vae.vae_model import VAE

    B, G, C = 5, 4, 2
    # Build a mock VAE: we only need attributes g_cross_sample_gene_var and _cross_sample_gene_variance_loss
    m = MagicMock(spec=VAE)
    target = torch.tensor([
        [0.1, 0.4],
        [0.0, 0.2],
        [0.5, 0.05],
        [0.03, 0.1],
    ], dtype=torch.float32)
    m.g_cross_sample_gene_var = target
    m._cross_sample_gene_variance_loss = VAE._cross_sample_gene_variance_loss.__get__(m, VAE)

    # Construct x such that cross-sample variance per (gene,ct) is explicit.
    # Set per-(g,c) column means = 0 and each column has variance tg,c.
    # Choose pattern: repeat (a, -a, a, -a, 0) -> var = (sum sq_dev)/(B-1)
    # For each (g,c) we want var = t; sum sq_dev = (B-1)*t
    # Use pattern: sqrt(t*(B-1)/s) * signs; s=number of non-zero entries.
    def build_col(t):
        pattern = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.0])  # B=5, 4 non-zero entries
        s = (pattern != 0).sum().float()
        # sum sq_dev = (B-1)*t -> (non-zero magnitude)^2 * s = (B-1)*t
        mag = torch.sqrt((t * (B - 1)) / s) if t > 0 else 0.0
        return pattern * mag

    x = torch.zeros((B, G, C), dtype=torch.float32)
    for g in range(G):
        for c in range(C):
            x[:, g, c] = build_col(float(target[g, c]))

    per_sample = m._cross_sample_gene_variance_loss(
        recon_x_all_types_log=x,
        batch_size=B,
    )
    assert per_sample.shape == (B,)
    # All elements equal (expand)
    diffs = (per_sample - per_sample[0]).abs().max().item()
    assert diffs < 1e-6
    # Mean absolute difference should be ~0 (accounting for float)
    assert per_sample[0].item() < 1e-5

    # Now shift every value by +constant. Variance unchanged -> still zero.
    per_sample2 = m._cross_sample_gene_variance_loss(
        recon_x_all_types_log=x + 1.23,
        batch_size=B,
    )
    assert per_sample2[0].item() < 1e-5
