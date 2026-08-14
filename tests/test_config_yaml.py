import importlib.resources as resources
from pathlib import Path

import pytest
import yaml

from vaedecon.configs import VAEDeconConfig


def test_example_config_resource_loads(tmp_path: Path):
    text = resources.files("vaedecon.configs").joinpath("example_config.yaml").read_text()
    cfg = yaml.safe_load(text)
    assert isinstance(cfg, dict)

    p = tmp_path / "example_config.yaml"
    p.write_text(text)
    loaded = VAEDeconConfig.from_yaml(p)
    assert loaded is not None
    assert "Test_set1" in loaded.data.test_sets


def test_legacy_single_test_set_populates_test_sets():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "test_set_file_path": "./datasets/test_set1.h5ad",
                "sct_gep_file_path": "./datasets/sct_gep.h5ad",
                "test_set_sample2cell_id_file_path": "./datasets/test_set1_sample2cell.csv",
            }
        }
    )
    assert len(loaded.data.test_sets) == 1
    only = next(iter(loaded.data.test_sets.values()))
    assert str(only.test_set_file_path) == "./datasets/test_set1.h5ad"
    assert str(only.sct_gep_file_path) == "./datasets/sct_gep.h5ad"
    assert (
        str(only.test_set_sample2cell_id_file_path)
        == "./datasets/test_set1_sample2cell.csv"
    )


def test_named_test_sets_backfill_legacy_fields():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "test_sets": {
                    "Test_set1": {
                        "test_set_file_path": "./datasets/test_set1.h5ad",
                        "sct_gep_file_path": "./datasets/sct_gep_1.h5ad",
                        "test_set_sample2cell_id_file_path": "./datasets/test_set1_sample2cell.csv",
                    },
                    "Test_set2": {
                        "test_set_file_path": "./datasets/test_set2.h5ad",
                        "sct_gep_file_path": "./datasets/sct_gep_2.h5ad",
                        "test_set_sample2cell_id_file_path": "./datasets/test_set2_sample2cell.csv",
                    },
                }
            }
        }
    )
    assert list(loaded.data.test_sets.keys()) == ["Test_set1", "Test_set2"]
    assert str(loaded.data.test_set_file_path) == "./datasets/test_set1.h5ad"
    assert str(loaded.data.sct_gep_file_path) == "./datasets/sct_gep_1.h5ad"
    assert (
        str(loaded.data.test_set_sample2cell_id_file_path)
        == "./datasets/test_set1_sample2cell.csv"
    )


def test_sct_gep_gene_mean_std_reference_falls_back_to_existing_path():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_gep_file_path": "./datasets/sct_gep.h5ad",
            }
        }
    )
    assert str(loaded.data.sct_gep_file_path) == "./datasets/sct_gep.h5ad"
    assert str(loaded.data.gene_mean_std_sct_gep_file_path) == ""


def test_sct_gep_gene_mean_std_reference_accepts_dedicated_path():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "gene_mean_std_sct_gep_file_path": "./datasets/gene_mean_std_ref.h5ad",
            }
        }
    )
    assert str(loaded.data.gene_mean_std_sct_gep_file_path) == "./datasets/gene_mean_std_ref.h5ad"


def test_sct_gep_gene_mean_std_reference_accepts_training_sct_list():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "sct_file_path": [
                    "./datasets/train_sct_a.h5ad",
                    "./datasets/train_sct_b.h5ad",
                ],
            }
        }
    )
    assert [str(p) for p in loaded.data.sct_file_path] == [
        "./datasets/train_sct_a.h5ad",
        "./datasets/train_sct_b.h5ad",
    ]


def test_sct_gep_gene_mean_std_reference_accepts_training_target_sets():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "gene_mean_std_source": "sct_gep",
                "training_target_sets": {
                    "Train_set1": {
                        "training_set_file_path": "./datasets/train_bulk_a.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/train_bulk_a_sample2cell.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_a.h5ad",
                    }
                },
            }
        }
    )
    only = loaded.data.training_target_sets["Train_set1"]
    assert str(only.training_sct_gep_file_path) == "./datasets/train_sct_a.h5ad"


def test_training_target_sets_require_complete_bundle():
    with pytest.raises(ValueError, match="training_target_sets entries must define"):
        VAEDeconConfig.from_dict(
            {
                "data": {
                    "gene_mean_std_source": "sct_gep",
                    "training_target_sets": {
                        "Train_set1": {
                            "training_set_file_path": "./datasets/train_bulk_a.h5ad",
                            "training_sct_gep_file_path": "./datasets/train_sct_a.h5ad",
                        }
                    },
                }
            }
        )


def test_sct_gep_gene_mean_std_reference_requires_any_reference_path():
    with pytest.raises(
        ValueError,
        match="gene_mean_std_sct_gep_file_path, sct_gep_file_path, or sct_file_path must be set",
    ):
        VAEDeconConfig.from_dict(
            {
                "data": {
                    "gene_mean_std_source": "sct_gep",
                    "sct_gep_file_path": "",
                    "gene_mean_std_sct_gep_file_path": "",
                }
            }
        )


def test_duplicate_yaml_keys_raise(tmp_path: Path):
    p = tmp_path / "dup.yaml"
    p.write_text("evaluation:\n  val_batch_size: 128\n  val_batch_size: 64\n")
    with pytest.raises(ValueError):
        _ = VAEDeconConfig.from_yaml(p)


def test_sigmoid_cell_prop_requires_cancer_cell_type_name():
    with pytest.raises(ValueError, match="cancer_cell_type_name must be set"):
        VAEDeconConfig.from_dict(
            {
                "model": {
                    "predict_cell_prop": True,
                    "cell_prop_activation_function": "sigmoid",
                }
            }
        )


def test_sigmoid_cell_prop_rejects_dirichlet_kld():
    with pytest.raises(ValueError, match="loss_coefficient\\['kld_p'\\] must be 0"):
        VAEDeconConfig.from_dict(
            {
                "model": {
                    "predict_cell_prop": True,
                    "cell_prop_activation_function": "sigmoid",
                    "cancer_cell_type_name": "Cancer Cells",
                    "loss_coefficient": {
                        "kld_p": 0.1,
                    },
                }
            }
        )


def test_softmax_cell_prop_rejects_dirichlet_kld():
    with pytest.raises(ValueError, match="loss_coefficient\\['kld_p'\\] must be 0"):
        VAEDeconConfig.from_dict(
            {
                "model": {
                    "predict_cell_prop": True,
                    "cell_prop_activation_function": "softmax",
                    "loss_coefficient": {
                        "kld_p": 0.1,
                    },
                }
            }
        )


def test_softmax_cell_prop_does_not_require_cancer_cell_type_name():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "predict_cell_prop": True,
                "cell_prop_activation_function": "softmax",
                "loss_coefficient": {
                    "kld_p": 0.0,
                },
            }
        }
    )
    assert loaded.model.cell_prop_activation_function == "softmax"


def test_sigmoid_all_norm_cell_prop_rejects_dirichlet_kld():
    with pytest.raises(ValueError, match="loss_coefficient\\['kld_p'\\] must be 0"):
        VAEDeconConfig.from_dict(
            {
                "model": {
                    "predict_cell_prop": True,
                    "cell_prop_activation_function": "sigmoid_all_norm",
                    "loss_coefficient": {
                        "kld_p": 0.1,
                    },
                }
            }
        )


def test_sigmoid_all_norm_cell_prop_config_loads_without_cancer_cell_type_name():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "predict_cell_prop": True,
                "cell_prop_activation_function": "sigmoid_all_norm",
                "cell_prop_loss_type": "l1_kl",
                "cell_prop_loss_kl_weight": 0.5,
                "cell_prop_loss_weighting": "low_prop_inverse",
                "cell_prop_loss_low_prop_epsilon": 0.02,
                "cell_prop_loss_weight_clamp": [1.0, 4.0],
                "loss_coefficient": {
                    "kld_p": 0.0,
                },
            }
        }
    )
    assert loaded.model.cell_prop_activation_function == "sigmoid_all_norm"
    assert loaded.model.cell_prop_loss_weighting == "low_prop_inverse"
    assert loaded.model.cell_prop_loss_low_prop_epsilon == 0.02
    assert loaded.model.cell_prop_loss_weight_clamp == (1.0, 4.0)


def test_cell_prop_loss_weight_clamp_requires_positive_ordered_bounds():
    with pytest.raises(ValueError, match="cell_prop_loss_weight_clamp must satisfy 0 < min <= max"):
        VAEDeconConfig.from_dict(
            {
                "model": {
                    "cell_prop_loss_weight_clamp": [0.0, 4.0],
                }
            }
        )


def test_cell_type_existence_requires_predict_cell_prop():
    with pytest.raises(ValueError, match="cell_type_existence_weight"):
        VAEDeconConfig.from_dict(
            {
                "model": {
                    "predict_cell_prop": False,
                    "loss_coefficient": {
                        "cell_type_existence_weight": 0.1,
                    },
                }
            }
        )


def test_cell_prop_fusion_and_loss_config_loads():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "predict_cell_prop": True,
                "cell_prop_activation_function": "softmax",
                "cell_prop_fusion_strategy": "shared_feature_gated",
                "cell_prop_fusion_dim": 128,
                "cell_prop_head_hidden_dims": [256, 128],
                "cell_prop_head_dropout_rate": 0.2,
                "cell_prop_loss_type": "l1_kl",
                "cell_prop_loss_kl_weight": 0.7,
            }
        }
    )
    assert loaded.model.cell_prop_fusion_strategy == "shared_feature_gated"
    assert loaded.model.cell_prop_fusion_dim == 128
    assert loaded.model.cell_prop_head_hidden_dims == [256, 128]
    assert loaded.model.cell_prop_loss_type == "l1_kl"
    assert loaded.model.cell_prop_loss_kl_weight == 0.7


def test_aux_loss_schedules_reject_unknown_target():
    with pytest.raises(ValueError, match="aux_loss_schedules contains unsupported targets"):
        VAEDeconConfig.from_dict(
            {
                "training": {
                    "aux_loss_schedules": {
                        "unknown_weight": {
                            "type": "linear",
                            "start_epoch": 0,
                            "end_epoch": 10,
                            "start_value": 0.0,
                            "end_value": 1.0,
                        }
                    }
                }
            }
        )
