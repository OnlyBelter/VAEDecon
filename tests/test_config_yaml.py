import importlib.resources as resources
from pathlib import Path

import pytest
import yaml

from vaedecon.configs import VAEDeconConfig
from vaedecon.workflow.train import VAEDeconTrainer


def test_example_config_resource_loads(tmp_path: Path):
    text = resources.files("vaedecon.configs").joinpath("example_config.yaml").read_text()
    cfg = yaml.safe_load(text)
    assert isinstance(cfg, dict)

    p = tmp_path / "example_config.yaml"
    p.write_text(text)
    loaded = VAEDeconConfig.from_yaml(p)
    assert loaded is not None
    assert "Test_set1" in loaded.data.test_sets


def test_build_trainer_config_preserves_saved_model_selection(tmp_path: Path):
    config = VAEDeconConfig.from_dict(
        {
            "training": {
                "output_dir": str(tmp_path),
                "naming_postfix": "preserve-training-config-fields",
                "batch_size": 60,
                "seed": 10,
                "device": "cpu",
                "train_split": 0.9,
                "val_split": 0.1,
                "saved_model_selection": "last",
            }
        }
    )

    trainer = VAEDeconTrainer(config=config)
    trainer_config = trainer._build_trainer_config()

    assert trainer_config.saved_model_selection == "last"
    assert trainer_config.seed == 10
    assert trainer_config.device == "cpu"
    assert trainer_config.naming_postfix == "preserve-training-config-fields"
    assert trainer_config.train_split == pytest.approx(0.9)
    assert trainer_config.val_split == pytest.approx(0.1)
    assert trainer_config.per_device_train_batch_size == 60
    assert trainer_config.per_device_eval_batch_size == 60


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


def test_debug_overfit_training_config_loads():
    loaded = VAEDeconConfig.from_dict(
        {
            "training": {
                "debug_overfit": {
                    "enabled": True,
                    "subset_size": 100,
                    "subset_seed": 10,
                    "use_training_subset_as_eval": True,
                    "num_epochs_override": 10000,
                    "n_early_stopping_patience_override": 10000,
                    "disable_early_stopping": True,
                    "save_post_train_predictions": True,
                }
            }
        }
    )
    assert loaded.training.debug_overfit.enabled is True
    assert loaded.training.debug_overfit.subset_size == 100
    assert loaded.training.debug_overfit.use_training_subset_as_eval is True
    assert loaded.training.debug_overfit.num_epochs_override == 10000


def test_training_saved_model_selection_config_loads():
    loaded = VAEDeconConfig.from_dict(
        {
            "training": {
                "saved_model_selection": "last",
            }
        }
    )
    assert loaded.training.saved_model_selection == "last"


def test_training_saved_model_selection_defaults_to_best():
    loaded = VAEDeconConfig.from_dict({})
    assert loaded.training.saved_model_selection == "best"


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


def test_predict_cell_prop_false_allows_activation_for_oracle_mixing():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "predict_cell_prop": False,
                "cell_prop_activation_function": "sigmoid_all_norm",
                "cell_type_existence_shift_scale": 0.0,
                "loss_coefficient": {
                    "kld_p": 0.0,
                    "cell_prop": 0.0,
                    "cell_type_existence_weight": 0.0,
                },
            }
        }
    )
    assert loaded.model.predict_cell_prop is False
    assert loaded.model.cell_prop_activation_function == "sigmoid_all_norm"


def test_predict_cell_prop_false_allows_existence_shift_for_oracle_mixing():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "predict_cell_prop": False,
                "cell_prop_activation_function": "sigmoid_all_norm",
                "cell_type_existence_shift_scale": 0.25,
                "loss_coefficient": {
                    "kld_p": 0.0,
                    "cell_prop": 0.0,
                    "cell_type_existence_weight": 0.0,
                },
            }
        }
    )
    assert loaded.model.predict_cell_prop is False
    assert loaded.model.cell_type_existence_shift_scale == 0.25


def test_conditioned_decoder_config_loads():
    loaded = VAEDeconConfig.from_dict(
        {
            "model": {
                "decoders": ["DecoderConditionalMLP"],
                "conditional_decoder_cell_type_emb_dim": 48,
                "conditional_decoder_context_dim": 192,
                "conditional_decoder_dropout_rate": 0.15,
            }
        }
    )
    assert loaded.model.decoders == ["DecoderConditionalMLP"]
    assert loaded.model.conditional_decoder_cell_type_emb_dim == 48
    assert loaded.model.conditional_decoder_context_dim == 192
    assert loaded.model.conditional_decoder_dropout_rate == 0.15


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


def test_adaptive_aux_loss_schedule_config_loads():
    loaded = VAEDeconConfig.from_dict(
        {
            "data": {
                "training_target_sets": {
                    "Train_set1": {
                        "training_set_file_path": "./datasets/train_bulk_a.h5ad",
                        "training_set_sample2cell_id_file_path": "./datasets/train_bulk_a_sample2cell.csv",
                        "training_sct_gep_file_path": "./datasets/train_sct_a.h5ad",
                    }
                },
            },
            "training": {
                "adaptive_aux_loss_schedule": {
                    "enabled": True,
                    "monitor": "val_loss",
                    "min_epoch_before_trigger": 10,
                    "trigger_patience": 5,
                    "trigger_min_delta": 0.001,
                    "cooldown_epochs": 15,
                    "update_interval_epochs": 1,
                    "pair_targets": True,
                    "targets": {
                        "cell_prop": {
                            "range": [500.0, 100.0],
                            "step_size": 25.0,
                            "reverse_on_plateau": True,
                        },
                        "cell_type_sct_gep_weight": {
                            "range": [10.0, 30.0],
                            "step_size": 1.0,
                            "reverse_on_plateau": True,
                        },
                    },
                }
            },
            "model": {
                "predict_cell_prop": True,
            },
        }
    )

    schedule = loaded.training.adaptive_aux_loss_schedule
    assert schedule is not None
    assert schedule.enabled is True
    assert schedule.monitor == "val_loss"
    assert tuple(schedule.targets["cell_prop"].range) == (500.0, 100.0)
    assert tuple(schedule.targets["cell_type_sct_gep_weight"].range) == (10.0, 30.0)


def test_adaptive_aux_loss_schedule_rejects_overlap_with_linear_schedule():
    with pytest.raises(
        ValueError,
        match="adaptive_aux_loss_schedule conflicts with aux_loss_schedules",
    ):
        VAEDeconConfig.from_dict(
            {
                "data": {
                    "training_target_sets": {
                        "Train_set1": {
                            "training_set_file_path": "./datasets/train_bulk_a.h5ad",
                            "training_set_sample2cell_id_file_path": "./datasets/train_bulk_a_sample2cell.csv",
                            "training_sct_gep_file_path": "./datasets/train_sct_a.h5ad",
                        }
                    },
                },
                "training": {
                    "aux_loss_schedules": {
                        "cell_type_sct_gep_weight": {
                            "type": "linear",
                            "start_epoch": 0,
                            "end_epoch": 10,
                            "start_value": 10.0,
                            "end_value": 30.0,
                        }
                    },
                    "adaptive_aux_loss_schedule": {
                        "enabled": True,
                        "pair_targets": False,
                        "targets": {
                            "cell_type_sct_gep_weight": {
                                "range": [10.0, 30.0],
                                "step_size": 1.0,
                            },
                        },
                    },
                },
            }
        )


def test_adaptive_aux_loss_schedule_requires_predict_cell_prop_for_positive_cell_prop_range():
    with pytest.raises(
        ValueError,
        match="adaptive_aux_loss_schedule target 'cell_prop' requires model.predict_cell_prop=True",
    ):
        VAEDeconConfig.from_dict(
            {
                "training": {
                    "adaptive_aux_loss_schedule": {
                        "enabled": True,
                        "pair_targets": False,
                        "targets": {
                            "cell_prop": {
                                "range": [500.0, 100.0],
                                "step_size": 25.0,
                            },
                        },
                    },
                },
                "model": {
                    "predict_cell_prop": False,
                },
            }
        )


def test_adaptive_aux_loss_schedule_requires_training_target_sets_for_positive_sct_gep_range():
    with pytest.raises(
        ValueError,
        match="adaptive_aux_loss_schedule target 'cell_type_sct_gep_weight' requires data.training_target_sets",
    ):
        VAEDeconConfig.from_dict(
            {
                "training": {
                    "adaptive_aux_loss_schedule": {
                        "enabled": True,
                        "pair_targets": False,
                        "targets": {
                            "cell_type_sct_gep_weight": {
                                "range": [10.0, 30.0],
                                "step_size": 1.0,
                            },
                        },
                    },
                },
            }
        )


def test_adaptive_aux_loss_schedule_pair_targets_requires_both_targets():
    with pytest.raises(
        ValueError,
        match="adaptive_aux_loss_schedule with pair_targets=True must configure exactly",
    ):
        VAEDeconConfig.from_dict(
            {
                "training": {
                    "adaptive_aux_loss_schedule": {
                        "enabled": True,
                        "pair_targets": True,
                        "targets": {
                            "cell_prop": {
                                "range": [500.0, 100.0],
                                "step_size": 25.0,
                            },
                        },
                    },
                },
                "model": {
                    "predict_cell_prop": True,
                },
            }
        )
