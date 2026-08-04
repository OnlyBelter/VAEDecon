from pathlib import Path

from vaedecon.configs import VAEDeconConfig
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
