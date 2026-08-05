from pathlib import Path

from vaedecon.configs import VAEDeconConfig
from vaedecon.workflow.inference import VAEDeconPredictor
from vaedecon.workflow.train import VAEDeconTrainer


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

    assert trainer._resolve_gene_mean_std_sct_gep_path() == Path(
        "./datasets/dedicated_gene_mean_std_ref.h5ad"
    )


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

