"""
Default configuration for VAEDecon
"""
from dataclasses import dataclass, field, asdict, is_dataclass
from .base_config import BaseTrainerConfig, BaseModelConfig, BaseConfig
from typing import List, Dict, Optional, Tuple, Any, Union, Literal
from pathlib import Path
from pydantic import AliasChoices, Field, field_validator, model_validator, BaseModel


class LossCoefficient(BaseModel):
    beta: float = 2.0
    gamma: float = 0.005
    attractor_weight: float = 0.0
    kld_type: Literal["ave", "sep"] = "ave"   # your code uses 'sep', not 'sum'
    kld_p: float = 0.0
    cell_prop: float = 0.0
    weighting_gene_by_exp: bool = True
    weight_clamp_range: Tuple[float, float] = (0.2, 5.0)
    gene_mean_weight: float = 1.0
    gene_std_weight: float = 0.0
    gene_mean_std_weight: Optional[float] = None
    cross_sample_gene_var_weight: float = 0.0
    cell_type_sct_gep_weight: float = 0.0
    cell_type_existence_weight: float = 0.0
    z_score_reg_weight: float = 0.0
    z_score_kl_weight: float = 0.0  # Weight for KL divergence between empirical Z-score distribution and N(0,1)
    low_mean_std_weight: float = 1.0  # Weight for MSE regularization on low mean/std genes
    low_mean_threshold: float = 2.0
    low_std_threshold: float = 1.0
    hierarchical_code_weight: float = 0.0

    @field_validator(
        "beta",
        "gamma",
        "attractor_weight",
        "kld_p",
        "cell_prop",
        "gene_mean_weight",
        "gene_std_weight",
        "gene_mean_std_weight",
        "cross_sample_gene_var_weight",
        "cell_type_sct_gep_weight",
        "cell_type_existence_weight",
        "z_score_reg_weight",
        "z_score_kl_weight",
        "low_mean_std_weight",
        "low_mean_threshold",
        "low_std_threshold",
        "hierarchical_code_weight",
        mode="before"
    )
    @classmethod
    def non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in ("none", "null"):
                return None
            try:
                v = float(s)
            except ValueError as e:
                raise ValueError("must be a number") from e
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @model_validator(mode="after")
    def check_weight_range(self):
        mn, mx = self.weight_clamp_range
        if mn <= 0 or mn >= mx:
            raise ValueError("weight_clamp_range must satisfy 0 < min < max")
        return self

    @model_validator(mode="after")
    def reconcile_gene_stat_weights(self):
        if self.gene_mean_std_weight is not None:
            if self.gene_mean_weight == 1.0 and self.gene_std_weight == 1.0:
                self.gene_mean_weight = self.gene_mean_std_weight
                self.gene_std_weight = self.gene_mean_std_weight
        return self


class TestSetConfig(BaseModel):
    """Per-test-set file bundle used during inference and visualization."""

    test_set_file_path: str | Path = ''
    test_set_sample2cell_id_file_path: str | Path = ''
    sct_gep_file_path: str | Path = ''

    @model_validator(mode="after")
    def validate_required_test_file(self):
        if not self.test_set_file_path or str(self.test_set_file_path).strip() == "":
            raise ValueError("test_set_file_path must be set for each configured test set")
        return self


class TrainingSetSCTTargetConfig(BaseModel):
    """Per-training-set bundle for matched sctGEP supervision."""

    training_set_file_path: str | Path = ""
    training_set_sample2cell_id_file_path: str | Path = ""
    training_sct_gep_file_path: str | Path = ""

    @model_validator(mode="after")
    def validate_required_training_target_files(self):
        required_fields = {
            "training_set_file_path": self.training_set_file_path,
            "training_set_sample2cell_id_file_path": self.training_set_sample2cell_id_file_path,
            "training_sct_gep_file_path": self.training_sct_gep_file_path,
        }
        missing = [
            field_name for field_name, field_value in required_fields.items()
            if not field_value or str(field_value).strip() == ""
        ]
        if missing:
            raise ValueError(
                "training_target_sets entries must define "
                + ", ".join(missing)
            )
        return self


class EncoderOutputRoutingConfig(BaseModel):
    """Routing policy for multi-encoder outputs."""

    cell_prop_source: str = "fused"
    latent_posterior_source: str = "fused"
    decoder_context_source: str = "fused"

    @field_validator(
        "cell_prop_source",
        "latent_posterior_source",
        "decoder_context_source",
        mode="before",
    )
    @classmethod
    def validate_routing_source(cls, v: str) -> str:
        if v is None:
            return "fused"
        if not isinstance(v, str):
            raise ValueError("routing sources must be strings")
        source = v.strip()
        if not source:
            raise ValueError("routing sources cannot be empty")
        return source


class ScalarScheduleConfig(BaseModel):
    """Linear scalar schedule for training-time coefficient updates."""

    type: Literal["linear"] = "linear"
    start_epoch: int = Field(default=0, ge=0)
    end_epoch: int = Field(default=0, ge=0)
    start_value: float = 0.0
    end_value: float = 0.0

    @model_validator(mode="after")
    def validate_epoch_order(self):
        if self.end_epoch < self.start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")
        return self


class AdaptiveAuxLossTargetConfig(BaseModel):
    """Bounded adaptive schedule for one auxiliary loss target."""

    range: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description=(
            "Ordered two-point range. The first value is the initial weight and "
            "the second value defines the initial direction."
        ),
    )
    step_size: float = Field(
        default=1.0,
        gt=0.0,
        description="Absolute amount to move the target weight at each update event.",
    )
    reverse_on_plateau: bool = Field(
        default=True,
        description=(
            "Whether to reverse direction for this target when the adaptive "
            "schedule detects a plateau and cooldown has expired."
        ),
    )

    @model_validator(mode="after")
    def validate_range_values(self):
        start, end = self.range
        if start == end:
            raise ValueError("adaptive schedule range endpoints must differ to define a direction")
        return self


class AdaptiveAuxLossScheduleConfig(BaseModel):
    """Adaptive epoch-based schedule for selected multitask loss weights."""

    enabled: bool = Field(
        default=False,
        description="Enable adaptive epoch-based updates for configured auxiliary loss targets.",
    )
    monitor: str = Field(
        default="val_loss",
        description="Metric name used to detect plateaus for adaptive schedule reversals.",
    )
    min_epoch_before_trigger: int = Field(
        default=0,
        ge=0,
        description="Earliest epoch where plateau-triggered reversals are allowed.",
    )
    trigger_patience: int = Field(
        default=5,
        ge=1,
        description="Consecutive non-improving epochs required before a plateau trigger can fire.",
    )
    trigger_min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum decrease in the monitored metric that counts as an improvement.",
    )
    cooldown_epochs: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of epochs to wait after a reversal before another reversal "
            "is allowed."
        ),
    )
    update_interval_epochs: int = Field(
        default=1,
        ge=1,
        description="How often to update scheduled target weights, in epochs.",
    )
    pair_targets: bool = Field(
        default=True,
        description=(
            "If true, all configured targets are treated as one paired schedule "
            "and reverse together on plateau events."
        ),
    )
    targets: Dict[str, AdaptiveAuxLossTargetConfig] = Field(
        default_factory=dict,
        description="Per-target adaptive schedule configuration.",
    )

    @model_validator(mode="after")
    def validate_targets(self):
        supported_targets = {
            "cell_prop",
            "cell_type_sct_gep_weight",
        }
        invalid_targets = set(self.targets.keys()) - supported_targets
        if invalid_targets:
            raise ValueError(
                "adaptive_aux_loss_schedule contains unsupported targets: "
                + ", ".join(sorted(invalid_targets))
            )
        if self.enabled and not self.targets:
            raise ValueError(
                "adaptive_aux_loss_schedule.enabled=True requires at least one configured target"
            )
        if self.pair_targets and self.enabled and set(self.targets.keys()) != supported_targets:
            raise ValueError(
                "adaptive_aux_loss_schedule with pair_targets=True must configure exactly "
                "'cell_prop' and 'cell_type_sct_gep_weight'"
            )
        return self


class DebugOverfitConfig(BaseModel):
    """Config-gated overfit mode for memorization/debugging runs."""

    enabled: bool = False
    subset_size: int = Field(
        default=100,
        ge=1,
        description=(
            "Number of training samples to draw for the reproducible debug "
            "subset when debug overfit mode is enabled."
        ),
    )
    subset_seed: int = Field(
        default=123,
        description="Random seed used to sample the debug overfit subset.",
    )
    use_training_subset_as_eval: bool = Field(
        default=True,
        description=(
            "Whether to reuse the selected debug training subset as the eval/"
            "test subset for overfit verification."
        ),
    )
    num_epochs_override: int = Field(
        default=10000,
        ge=1,
        description=(
            "Effective max epoch budget used only when debug overfit mode is "
            "enabled."
        ),
    )
    n_early_stopping_patience_override: int = Field(
        default=10000,
        ge=1,
        description=(
            "Effective early-stopping patience used only when debug overfit "
            "mode is enabled."
        ),
    )
    disable_early_stopping: bool = Field(
        default=True,
        description=(
            "If true, skip adding the EarlyStopping callback during debug "
            "overfit runs."
        ),
    )
    save_post_train_predictions: bool = Field(
        default=True,
        description=(
            "If true, save a post-train prediction artifact on the selected "
            "debug subset."
        ),
    )


class StageEarlyStoppingConfig(BaseModel):
    """Stage-local early stopping settings for staged training."""

    monitor: str = Field(
        default="val_loss",
        description="Metric monitored by ModelCheckpoint and EarlyStopping during this stage.",
    )
    patience: int = Field(
        default=15,
        ge=1,
        description="Stage-local early-stopping patience.",
    )
    min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum improvement required to reset stage-local early stopping.",
    )

    @field_validator("monitor", mode="before")
    @classmethod
    def validate_monitor(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("early_stopping.monitor must be a string")
        monitor = v.strip()
        if not monitor:
            raise ValueError("early_stopping.monitor cannot be empty")
        return monitor


class StagedTrainingStageConfig(BaseModel):
    """Configuration for one named stage in the staged training workflow."""

    name: Literal[
        "cell_prop_predictor_pretrain",
        "reconstruction_training",
        "joint_finetune",
    ]
    max_epochs: int = Field(
        gt=0,
        description="Maximum epoch budget for this stage before stage-local early stopping.",
    )
    train_modules: List[str] = Field(
        default_factory=list,
        description="Logical module groups that should remain trainable during this stage.",
    )
    freeze_modules: List[str] = Field(
        default_factory=list,
        description="Logical module groups that should remain frozen during this stage.",
    )
    learning_rate_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Multiplicative scale applied to training.learning_rate for this stage.",
    )
    loss_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Stage-local overrides applied to supported loss-related config fields.",
    )
    early_stopping: StageEarlyStoppingConfig = Field(
        default_factory=StageEarlyStoppingConfig,
        description="Stage-local early stopping settings.",
    )

    @field_validator("train_modules", "freeze_modules")
    @classmethod
    def validate_module_names(cls, v: List[str]) -> List[str]:
        valid_modules = {"cell_prop_predictor", "encoders", "decoder"}
        normalized: List[str] = []
        for module_name in v:
            if not isinstance(module_name, str):
                raise ValueError("stage module names must be strings")
            clean_name = module_name.strip()
            if clean_name not in valid_modules:
                raise ValueError(
                    f"Unsupported stage module name: {module_name!r}. Valid names: {sorted(valid_modules)}"
                )
            normalized.append(clean_name)
        if len(set(normalized)) != len(normalized):
            raise ValueError("stage module names must be unique within one list")
        return normalized

    @model_validator(mode="after")
    def validate_stage_settings(self):
        if not self.train_modules:
            raise ValueError("Each staged training stage must define at least one train_modules entry")
        overlap = set(self.train_modules) & set(self.freeze_modules)
        if overlap:
            raise ValueError(
                "A staged training stage cannot both train and freeze the same modules: "
                + ", ".join(sorted(overlap))
            )

        supported_loss_overrides = set(LossCoefficient.model_fields.keys()) | {"cell_type_existence_shift_scale"}
        invalid_override_keys = set(self.loss_overrides.keys()) - supported_loss_overrides
        if invalid_override_keys:
            raise ValueError(
                "Unsupported staged-training loss_overrides keys: "
                + ", ".join(sorted(invalid_override_keys))
            )

        if self.name == "cell_prop_predictor_pretrain" and "cell_prop_predictor" not in set(self.train_modules):
            raise ValueError(
                "cell_prop_predictor_pretrain must include 'cell_prop_predictor' in train_modules"
            )
        if self.name == "cell_prop_predictor_pretrain":
            self.early_stopping.monitor = "val_cell_prop_loss"
        return self


class StagedTrainingConfig(BaseModel):
    """Multi-stage training workflow for the DeSide predictor integration."""

    enabled: bool = Field(
        default=False,
        description="Enable staged training instead of the legacy single-pass training loop.",
    )
    run_stages: List[str] = Field(
        default_factory=list,
        description="Optional subset of configured stage names to execute in this invocation.",
    )
    stage_init_checkpoints: Dict[str, str | Path] = Field(
        default_factory=dict,
        description=(
            "Optional stage-name -> checkpoint path mapping used when a selected stage does not "
            "follow its direct predecessor in the same run."
        ),
    )
    stages: List[StagedTrainingStageConfig] = Field(
        default_factory=list,
        description="Ordered stage definitions for the staged training workflow.",
    )

    @field_validator("run_stages")
    @classmethod
    def validate_run_stages(cls, v: List[str]) -> List[str]:
        normalized: List[str] = []
        for stage_name in v:
            if not isinstance(stage_name, str):
                raise ValueError("run_stages entries must be strings")
            clean_name = stage_name.strip()
            if not clean_name:
                raise ValueError("run_stages entries cannot be empty")
            normalized.append(clean_name)
        if len(set(normalized)) != len(normalized):
            raise ValueError("run_stages entries must be unique")
        return normalized

    @field_validator("stage_init_checkpoints", mode="before")
    @classmethod
    def normalize_stage_init_checkpoints(cls, v: Optional[Dict[str, str | Path]]) -> Dict[str, str | Path]:
        if not v:
            return {}
        normalized: Dict[str, str | Path] = {}
        for stage_name, checkpoint_path in dict(v).items():
            clean_name = str(stage_name).strip()
            if not clean_name:
                raise ValueError("stage_init_checkpoints keys must be non-empty")
            normalized[clean_name] = checkpoint_path
        return normalized

    @model_validator(mode="after")
    def validate_stage_plan(self):
        if self.enabled and not self.stages:
            raise ValueError("staged_training.enabled=True requires at least one configured stage")

        configured_stage_names = [stage.name for stage in self.stages]
        if len(set(configured_stage_names)) != len(configured_stage_names):
            raise ValueError("staged_training stage names must be unique")

        run_stage_names = self.run_stages or configured_stage_names
        unknown_run_stages = set(run_stage_names) - set(configured_stage_names)
        if unknown_run_stages:
            raise ValueError(
                "run_stages contains names that are not present in stages: "
                + ", ".join(sorted(unknown_run_stages))
            )
        if run_stage_names:
            expected_stage_slice = configured_stage_names[
                configured_stage_names.index(run_stage_names[0]): configured_stage_names.index(run_stage_names[-1]) + 1
            ]
            if run_stage_names != expected_stage_slice:
                raise ValueError(
                    "run_stages must follow the canonical contiguous stage order: "
                    "cell_prop_predictor_pretrain -> reconstruction_training -> joint_finetune"
                )

        unknown_init_ckpt_stages = set(self.stage_init_checkpoints.keys()) - set(configured_stage_names)
        if unknown_init_ckpt_stages:
            raise ValueError(
                "stage_init_checkpoints contains unknown stage names: "
                + ", ".join(sorted(unknown_init_ckpt_stages))
            )

        configured_index = {stage_name: idx for idx, stage_name in enumerate(configured_stage_names)}
        for stage_name in run_stage_names:
            stage_idx = configured_index[stage_name]
            if stage_idx == 0:
                continue
            direct_predecessor = configured_stage_names[stage_idx - 1]
            if direct_predecessor in run_stage_names:
                continue
            if stage_name not in self.stage_init_checkpoints:
                raise ValueError(
                    f"Selected stage {stage_name!r} requires an init checkpoint because its direct "
                    f"predecessor {direct_predecessor!r} is not selected in this run."
                )

        return self

# @dataclass
class DataConfig(BaseConfig):
    """dataset configuration"""
    data_dir: str | Path = './datasets/'

    # Training data
    sct_file_path: list[str | Path] = Field(default_factory=list)
    simu_bulk_file_path: list[str | Path] = Field(default_factory=list)

    # Test data
    test_set_file_path: str | Path = ''
    test_set_sample2cell_id_file_path: str | Path = ''
    sct_gep_file_path: str | Path = ''  # Used for query sampled sctGEPs in test set
    test_sets: Dict[str, TestSetConfig] = Field(default_factory=dict)
    training_target_sets: Dict[str, TrainingSetSCTTargetConfig] = Field(default_factory=dict)
    training_sct_gep_cell_prop_threshold: float = Field(
        default=0.005,
        ge=0.0,
        description=(
            "Mask matched-sctGEP supervision for cell types whose true training "
            "cell proportions are below this threshold."
        ),
    )

    gene_mean_std_source: Literal["sct_gep", "pooled_sc"] = Field(
        default="sct_gep",
        description="Source of the reference gene mean/std statistics used in training.",
    )
    gene_mean_std_sct_gep_file_path: str | Path = Field(
        default="",
        description=(
            "Optional dedicated SCT .h5ad used to compute gene mean/std when "
            "gene_mean_std_source='sct_gep'. Falls back to sct_gep_file_path when empty."
        ),
    )
    pooled_sc_h5ad_path: str | Path = Field(
        default="",
        description="Path to the pooled scRNA-seq .h5ad used when gene_mean_std_source='pooled_sc'.",
    )
    pooled_sc_cell_type_col: str = Field(
        default="cell_type",
        description="Column name in pooled scRNA-seq .h5ad .obs that stores string cell type labels.",
    )
    pooled_sc_cell_subtype_col: str = Field(
        default="cell_subtype",
        description="Optional column name in pooled scRNA-seq .h5ad .obs that stores string cell subtype labels.",
    )
    pooled_sc_sample_size: int = Field(
        default=1000,
        description="Maximum number of single cells to sample per cell type when computing gene mean/std from pooled scRNA-seq.",
    )
    pooled_sc_seed: int = Field(
        default=123,
        description="Random seed for pooled scRNA-seq cell sampling when computing gene mean/std.",
    )

    # Additional files
    pred_cell_prop_file_path: Optional[str] = None
    cell_type2ave_exp_file_path: Optional[str] = None
    # PPI and Pathway file paths
    ppi_file_path: Optional[Path] = Field(
        default=None,
        description="Path to protein-protein interaction network file"
    )
    pathway_file_path: Optional[list[Path]] = Field(
        default=[],
        description="Path to Pathway files in .gmt format. Can provide multiple files for different pathway databases (e.g., KEGG, Reactome)."
    )

    @field_validator('ppi_file_path')
    @classmethod
    def validate_ppi_file(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate PPI file exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"PPI file not found: {v}")
        return v

    @model_validator(mode="after")
    def validate_gene_mean_std_source(self):
        configured_fields = set(getattr(self, "model_fields_set", set()))
        if self.gene_mean_std_source == "pooled_sc":
            if not self.pooled_sc_h5ad_path or str(self.pooled_sc_h5ad_path).strip() == "":
                raise ValueError("pooled_sc_h5ad_path must be set when gene_mean_std_source='pooled_sc'")
            if self.pooled_sc_sample_size <= 0:
                raise ValueError("pooled_sc_sample_size must be > 0")
        else:
            # Keep model-only / minimal configs backward compatible: when the user
            # did not configure any gene-mean/std reference fields, defer this
            # requirement until the data path is actually used by training/inference.
            relevant_fields = {
                "gene_mean_std_source",
                "gene_mean_std_sct_gep_file_path",
                "sct_gep_file_path",
                "sct_file_path",
                "test_sets",
                "training_target_sets",
            }
            if not (configured_fields & relevant_fields):
                return self

            has_dedicated_sct = bool(
                self.gene_mean_std_sct_gep_file_path
                and str(self.gene_mean_std_sct_gep_file_path).strip() != ""
            )
            has_top_level_sct = bool(
                self.sct_gep_file_path
                and str(self.sct_gep_file_path).strip() != ""
            )
            has_training_sct = any(
                p is not None and str(p).strip() != ""
                for p in self.sct_file_path
            )
            has_test_set_sct = any(
                cfg.sct_gep_file_path and str(cfg.sct_gep_file_path).strip() != ""
                for cfg in self.test_sets.values()
            )
            has_training_target_sct = any(
                cfg.training_sct_gep_file_path and str(cfg.training_sct_gep_file_path).strip() != ""
                for cfg in self.training_target_sets.values()
            )
            if not (
                has_dedicated_sct
                or has_top_level_sct
                or has_training_sct
                or has_test_set_sct
                or has_training_target_sct
            ):
                  if "training_target_sets" in configured_fields:
                      raise ValueError(
                          "gene_mean_std_sct_gep_file_path, sct_gep_file_path, sct_file_path, "
                          "or training_target_sets[*].training_sct_gep_file_path "
                          "must be set when gene_mean_std_source='sct_gep'"
                      )
                  raise ValueError(
                      "gene_mean_std_sct_gep_file_path, sct_gep_file_path, or sct_file_path "
                      "must be set when gene_mean_std_source='sct_gep'"
                  )
        return self

    @model_validator(mode="after")
    def reconcile_test_sets(self):
        normalized_test_sets: Dict[str, TestSetConfig] = {}
        for name, cfg in self.test_sets.items():
            clean_name = str(name).strip()
            if not clean_name:
                raise ValueError("Configured test set names must be non-empty")
            normalized_test_sets[clean_name] = cfg

        if normalized_test_sets:
            first = next(iter(normalized_test_sets.values()))
            object.__setattr__(self, "test_sets", normalized_test_sets)
            if not self.test_set_file_path or str(self.test_set_file_path).strip() == "":
                object.__setattr__(self, "test_set_file_path", first.test_set_file_path)
            if not self.test_set_sample2cell_id_file_path or str(self.test_set_sample2cell_id_file_path).strip() == "":
                object.__setattr__(
                    self,
                    "test_set_sample2cell_id_file_path",
                    first.test_set_sample2cell_id_file_path,
                )
            if not self.sct_gep_file_path or str(self.sct_gep_file_path).strip() == "":
                object.__setattr__(self, "sct_gep_file_path", first.sct_gep_file_path)
            return self

        if self.test_set_file_path and str(self.test_set_file_path).strip() != "":
            legacy_name = Path(str(self.test_set_file_path)).stem.strip() or "test_set"
            object.__setattr__(
                self,
                "test_sets",
                {
                    legacy_name: TestSetConfig(
                        test_set_file_path=self.test_set_file_path,
                        test_set_sample2cell_id_file_path=self.test_set_sample2cell_id_file_path,
                        sct_gep_file_path=self.sct_gep_file_path,
                    )
                },
            )
        return self

    @model_validator(mode="after")
    def reconcile_training_target_sets(self):
        normalized_training_target_sets: Dict[str, TrainingSetSCTTargetConfig] = {}
        for name, cfg in self.training_target_sets.items():
            clean_name = str(name).strip()
            if not clean_name:
                raise ValueError("Configured training target set names must be non-empty")
            normalized_training_target_sets[clean_name] = cfg
        object.__setattr__(self, "training_target_sets", normalized_training_target_sets)
        return self

    # Processing options
    # Scale input GEP data by a constant factor after log transformation (range of the input data will be (0, 1)).
    # This can help stabilize training and improve performance.
    scaling_by_constant: bool = Field(
        default=True,
        description='Whether to scale input GEP data by a constant factor after log transformation. '
    )
    scaling_factor: float = Field(
        default=20.0,
        description='Constant factor to scale input GEP data after log transformation when scaling_by_constant=True. '
                    'This can help stabilize training by normalizing the input into (0, 1) range'
                    ' and improve performance.'
    )

    remove_low_var_genes: bool = True  # If True, perform low-variance gene filtering.
    min_var: float = 1.0  # Minimum variance threshold for gene filtering (if remove_low_var_genes is True).
    force_reprocess: bool = False  # If True, ignore cache and re-run preprocessing.

    use_memmap: bool = True  # Use np.memmap for .npy cache files to reduce RAM pressure.
    chunk_size: int = 10000  # Chunk size for chunked transform. Increase for speed, decrease for memory.
    # If True, use .npz compressed cache files (smaller, typically slower).
    # Note: compressed .npz does not support true memmap behavior.
    compress: bool = False


# @dataclass(frozen=True)
class GEPDatasetConfig(DataConfig):
    """
    Configuration for GEPDataset.

    Why use a config object?
    - Improves readability (fewer long argument lists)
    - Prevents accidental positional argument bugs
    - Easier to serialize/store with experiment artifacts

    Args:
        file_paths:
            List of input file paths. Supported: .h5ad, .csv
        processed_data_dir:
            Cache directory for processed arrays and metadata.
            Must be provided for this implementation.
        force_reprocess:
            If True, ignore cache and re-run preprocessing.
        scaling_by_constant:
            If True, use `scaling_factor`;
            if float, use that value directly;
            if False, do not scale.
        scaling_factor:
            Default scaling divisor when scaling_by_constant is True.
        gene_list_file:
            Optional gene list for gene alignment/filtering.
        remove_low_var_genes:
            If True, perform low-variance gene filtering.
        min_var:
            Minimum variance threshold for gene filtering.
        cell_cell2ave_exp_file_path:
            Optional reference expression file for additional gene filtering.
        use_memmap:
            Use np.memmap for .npy cache files to reduce RAM pressure.
        chunk_size:
            Chunk size for chunked transform. Increase for speed, decrease for memory.
        compress:
            If True, use .npz compressed cache files (smaller, typically slower).
            Note: compressed .npz does not support true memmap behavior.
    """
    file_paths: List[Union[str, Path]]
    processed_data_dir: Optional[Union[str, Path]]
    force_reprocess: bool = False

    scaling_by_constant: Union[bool, float] = True
    scaling_factor: float = 20.0

    gene_list_file: Optional[Union[str, Path]] = None
    remove_low_var_genes: bool = False
    min_var: float = 1.0
    cell_cell2ave_exp_file_path: Optional[Union[str, Path]] = None

    use_memmap: bool = True
    chunk_size: int = 10000
    compress: bool = False


# @dataclass
class TrainingConfig(BaseTrainerConfig):
    """training configuration"""
    # Basic settings
    output_dir: str | Path = Path('./output/vae')
    naming_postfix: str = 'default'

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 512
    num_epochs: int = 1000

    # Early stopping
    n_early_stopping_patience: int = 15
    saved_model_selection: Literal["best", "last"] = Field(
        default="best",
        description=(
            "Which checkpoint prediction and inference should load by default "
            "after training: the best monitored checkpoint or the last "
            "completed epoch checkpoint."
        ),
    )

    # Device settings
    devices: int = 1
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'

    # Optimizer
    optimizer_cls: str = 'AdamW'

    # Saving
    steps_saving: int = 0

    # Debug
    debug_model: bool = False
    debug_overfit: DebugOverfitConfig = Field(
        default_factory=DebugOverfitConfig,
        description=(
            "Optional overfit-only debug mode that trains on a reproducible "
            "small subset and can evaluate on that same subset."
        ),
    )

    # Data split
    train_split: float = 0.8
    val_split: float = 0.2

    # Scheduler
    scheduler_cls: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    warmup_epochs: int = 0
    gradient_clip_val: Optional[float] = 1.0
    gene_stat_weight_schedule: Literal["linear"] = 'linear'
    gene_stat_weight_schedule_steps: Optional[int] = None
    gene_stat_weight_schedule_epochs: Optional[int] = None
    gene_mean_weight_start: float = 1.0
    gene_mean_weight_end: float = 0.0
    gene_std_weight_start: float = 0.0
    gene_std_weight_end: float = 1.0
    prog_bar_metrics: List[str] = Field(
        default_factory=lambda: [
            "loss",
            "kld",
            "recon_loss_conv",
            "low_mean_std_gene_loss",
            "z_score_kl_loss",
            "repulsion_loss",
            "attractor_loss",
        ],
        description="Metric keys from model output to show in the progress bar/logging loop.",
    )
    aux_loss_schedules: Dict[str, ScalarScheduleConfig] = Field(
        default_factory=dict,
        description=(
            "Optional epoch-based linear schedules for selected auxiliary "
            "loss weights and related scalar controls."
        ),
    )
    adaptive_aux_loss_schedule: Optional[AdaptiveAuxLossScheduleConfig] = Field(
        default=None,
        description=(
            "Optional adaptive epoch-based schedule for selected auxiliary "
            "loss weights. When omitted or disabled, fixed loss coefficients "
            "behave exactly as before."
        ),
    )
    staged_training: Optional[StagedTrainingConfig] = Field(
        default=None,
        description=(
            "Optional multi-stage training workflow for DeSide predictor pretraining, "
            "reconstruction training, and low-LR joint fine-tuning."
        ),
    )

    @model_validator(mode="after")
    def validate_aux_schedule_targets(self):
        valid_targets = {
            "cell_type_sct_gep_weight",
            "hierarchical_code_weight",
            "cell_type_existence_weight",
            "cell_type_existence_shift_scale",
        }
        invalid_targets = set(self.aux_loss_schedules.keys()) - valid_targets
        if invalid_targets:
            raise ValueError(
                "aux_loss_schedules contains unsupported targets: "
                + ", ".join(sorted(invalid_targets))
            )
        adaptive_schedule = self.adaptive_aux_loss_schedule
        if adaptive_schedule is not None and adaptive_schedule.enabled:
            overlapping_targets = set(self.aux_loss_schedules.keys()) & set(adaptive_schedule.targets.keys())
            if overlapping_targets:
                raise ValueError(
                    "adaptive_aux_loss_schedule conflicts with aux_loss_schedules for targets: "
                    + ", ".join(sorted(overlapping_targets))
                )
        return self


class ModelConfig(BaseModelConfig):
    """Complete model configuration for VAE-based deconvolution.

    This configuration extends BaseModelConfig with settings for the
    hybrid encoder architecture, GNN components, and custom loss functions in the VAE model.

    Attributes:
        Architecture:
            input_dim: Input dimensions (channels, features).
            latent_dim: Latent space dimension per cell type.
            n_cell_types: Number of cell types to deconvolve.
            learn_gep_residual: Whether to learn GEP residuals compared to mean GEP of each cell type.

        Encoder/Decoder:
            encoder_hidden_dims: Hidden layer dimensions for encoder.
            decoder_hidden_dims: Hidden layer dimensions for decoder.
            encoder_dropout_rate: Dropout rates for encoder layers.
            decoder_dropout_rate: Dropout rates for decoder layers.
            encoders: List of encoder types (e.g., ['EncoderHybrid']).

        Fusion Layer:
            fusion_hidden_dims: Hidden dimensions for fusion layers.
            fusion_dropout_rate: Dropout rates for fusion layers.

        GNN Settings:
            gnn_n_genes: Number of genes in GNN.
            gnn_inter_col_dim: Intermediate dimension for cell embeddings.
            gnn_embd_col_dim: Final cell embedding dimension.
            gnn_lambda_cols: Weight for cell loss term.
            gnn_num_layers: Number of GNN layers.
            gnn_drop_p: Dropout probability in GNN.
            gene_hidden_dim: Hidden dimension for gene projection.

        Loss Configuration:
            loss_coefficient: Dictionary containing:
                - cell_prop: Weight for cell proportion loss
                - kld_p: Weight for Dirichlet KL regularization on cell proportions
                - beta: Weight for reconstruction loss
                - gamma: Weight for regularization
                - kld_type: Type of KLD computation ('ave' or 'sum')
                - weighting_gene_by_exp: Weight genes by expression level
                - weight_clamp_range: Range to clamp gene weights
                - gene_mean_weight: Weight for gene mean loss, same as gene_mean_weight_start at the beginning of training, and will be updated according to gene_stat_weight_schedule and gene_stat_weight_schedule_epochs
                - gene_std_weight: Weight for gene std loss, same as gene_std_weight_start at the beginning of training, and will be updated according to gene_stat_weight_schedule and gene_stat_weight_schedule_epochs

        File Paths:
            input_gene_list_fp: Path to input gene list.
            cell_type_fp: Path to cell type definitions.
            gene_mean_std_fp: Path to gene statistics.
            model_dir: Directory to save model checkpoints.

        Other Settings:
            predict_cell_prop: Whether to predict cell proportions.
            using_positional_encoding: Use positional encoding for cell types.
    """

    # ==================== Architecture ====================
    input_dim: Tuple[int, int] = Field(
        default=(1, 17834),
        description="Input dimensions (channels, features)"
    )
    latent_dim: int = Field(
        default=10,
        gt=0,
        description="Latent space dimension per cell type"
    )
    n_cell_types: int = Field(
        default=16,
        gt=0,
        description="Number of cell types to deconvolve"
    )

    # ==================== Encoder/Decoder ====================
    encoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [2048, 1024, 1024, 512],
        description="Hidden layer dimensions for encoder"
    )
    decoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [512, 1024, 1024, 2048],
        description="Hidden layer dimensions for decoder"
    )
    encoder_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.1, 0.0],
        description="Dropout rates for encoder layers"
    )
    decoder_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.1, 0.0],
        description="Dropout rates for decoder layers"
    )

    # ==================== Fusion Layer ====================
    fusion_hidden_dims: Tuple[int, ...] = Field(
        default=(512, 256),
        description="Hidden dimensions for fusion layers"
    )
    fusion_dropout_rate: Tuple[float, ...] = Field(
        default=(0.1, 0.0),
        description="Dropout rates for fusion layers"
    )

    # ==================== Encoder Types ====================
    encoders: List[str] = Field(
        default_factory=lambda: ['EncoderHybrid'],
        description="List of encoder types to use"
    )
    encoder_aliases: List[str] = Field(
        default_factory=list,
        description="Optional aliases for active encoders, aligned by position with `encoders`.",
    )
    encoder_output_routing: EncoderOutputRoutingConfig = Field(
        default_factory=EncoderOutputRoutingConfig,
        description="Routing policy for cell proportions, latent posteriors, and decoder context.",
    )
    cell_prop_predictor_cls: Optional[str] = Field(
        default=None,
        description=(
            "Optional dedicated predictor branch used only for cell proportions and bulk decoder context. "
            "Current supported value: 'DeSideCellPropPredictor'."
        ),
    )
    cell_prop_predictor_alias: str = Field(
        default="cell_prop_predictor",
        description="Alias used when routing cell proportions or decoder context from the dedicated predictor branch.",
    )
    deside_pathway_network: bool = Field(
        default=True,
        description="Enable the pathway branch inside the dedicated DeSide-style cell-proportion predictor.",
    )
    deside_hidden_dims: List[int] = Field(
        default_factory=lambda: [200, 2000, 2000, 2000, 50],
        description="Hidden dimensions for the DeSide-style bulk GEP branch.",
    )
    deside_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.05, 0.05, 0.05, 0.2, 0.0],
        description="Dropout rates for the DeSide-style bulk GEP branch.",
    )
    deside_pathway_hidden_dims: List[int] = Field(
        default_factory=lambda: [50, 500, 500, 500, 50],
        description="Hidden dimensions for the DeSide-style pathway branch.",
    )
    deside_pathway_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0],
        description="Dropout rates for the DeSide-style pathway branch.",
    )
    deside_input_gene_list: Literal["filtered_genes", "intersection_with_pathway_genes"] = Field(
        default="filtered_genes",
        description=(
            "Gene-selection mode used by the DeSide-style predictor GEP branch when the pathway "
            "branch is enabled. 'filtered_genes' keeps the full filtered input-gene list; "
            "'intersection_with_pathway_genes' matches standalone DeSide by restricting the GEP "
            "branch to genes that appear in the pathway mask."
        ),
    )
    deside_normalization: Optional[Literal["batch_normalization", "layer_normalization"]] = Field(
        default="layer_normalization",
        description="Normalization style used inside the DeSide-style predictor.",
    )
    deside_normalization_layer: List[int] = Field(
        default_factory=lambda: [0, 0, 1, 1, 1, 1],
        description=(
            "Per-layer normalization mask for the DeSide-style predictor. Length must equal "
            "len(deside_hidden_dims) + 1."
        ),
    )

    # ==================== Decoder Types ====================
    decoders: List[str] = Field(
        default_factory=lambda: ['DecoderMLP'],
        description="List of decoder types to use"
    )

    # ==================== Loss Coefficients ====================
    loss_coefficient: LossCoefficient = Field(
        default_factory=LossCoefficient,
        description="Coefficients for different loss components"
    )

    # ==================== GNN Settings ====================
    gnn_n_genes: int = Field(
        default=12596,
        gt=0,
        description="Number of genes in GNN"
    )
    gnn_inter_col_dim: int = Field(
        default=500,
        gt=0,
        description="Intermediate dimension for cell embeddings"
    )
    gnn_embd_col_dim: int = Field(
        default=30,
        gt=0,
        description="Final cell embedding dimension"
    )
    gnn_lambda_cols: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for cell loss term"
    )
    gnn_num_layers: int = Field(
        default=3,
        ge=1,
        description="Number of GNN layers"
    )
    gnn_drop_p: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Dropout probability in GNN"
    )
    gene_hidden_dim: int = Field(
        default=10,
        gt=0,
        description="Hidden dimension for gene projection"
    )
    gnn_topk_attention: int = Field(
        default=1024,
        gt=0,
        description="Number of top genes to keep in the cross-attention pooling layer of the GNN"
    )

    # ==================== Pathway DNN Settings ====================
    input_dim_pathway: Tuple[int, int] = Field(
        default=(1, 17834),
        description="Input dimensions (channels, features)"
    )
    encoder_hidden_dims_pathway: List[int] = Field(
        default_factory=lambda: [2048, 1024, 1024, 512],
        description="Hidden layer dimensions for encoder"
    )
    encoder_dropout_rate_pathway: List[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.1, 0.0],
        description="Dropout rates for encoder layers"
    )

    # ==================== File Paths ====================
    input_gene_list_fp: Optional[Path] = Field(
        default=None,
        description="Path to gene list file (after preprocessing) for GEP-level reconstruction"
    )
    cell_type_fp: Optional[Path] = Field(
        default=None,
        description="Path to cell type definitions file"
    )
    gene_mean_std_fp: Optional[Path] = Field(
        default=None,
        description="Path to gene mean/std statistics file"
    )
    training_sct_cross_sample_gene_var_fp: Optional[Path] = Field(
        default=None,
        description=(
            "Path to training-SCT cross-sample gene variance CSV used by "
            "loss_coefficient.cross_sample_gene_var_weight. Usually saved "
            "under model_dir/training_sct_cross_sample_gene_variances.csv."
        ),
    )
    model_dir: Path | str = Field(
        default=None,
        description="Directory to save model checkpoints and outputs"
    )

    # ==================== Other Settings ====================
    using_positional_encoding: bool = Field(
        default=False,
        description="Use positional encoding in latent space to distinguish cell types if True."
    )
    torch_compile: bool = Field(
        default=False,
        description="Use torch.compile() to compile the model for faster training. PyTorch >= 2.0 required. "
                    "Typically gives 30-50%% speedup with zero code changes."
    )

    # ==================== Cell Proportion Prediction ====================
    predict_cell_prop: bool = Field(
        default=False,
        description="Whether to predict cell type proportions"
    )
    cell_prop_activation_function: Literal["softplus", "sigmoid", "softmax", "sigmoid_all_norm"] = Field(
        default="softplus",
        description="Activation used for the cell proportion head. "
                    "'softplus' keeps the Dirichlet workflow; 'sigmoid' predicts only non-cancer "
                    "cell types and assigns the cancer proportion as the remainder; "
                    "'softmax' predicts all cell types directly and normalizes them to sum to 1; "
                    "'sigmoid_all_norm' applies sigmoid to all cell-type logits and then normalizes "
                    "them to sum to 1."
    )
    cancer_cell_type_name: Optional[str] = Field(
        default=None,
        description="Exact cell type label used for the cancer cell type when "
                    "cell_prop_activation_function='sigmoid'."
    )
    cell_type_existence_shift_scale: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Scale of the centered latent shift derived from predicted cell-type "
            "existence probabilities. Set > 0 to enable the existence-conditioned "
            "mu shift when predict_cell_prop=True."
        ),
    )
    cell_prop_fusion_strategy: Literal[
        "legacy_output_average",
        "shared_feature_mean",
        "shared_feature_gated",
    ] = Field(
        default="legacy_output_average",
        description=(
            "How to combine encoder information for cell-proportion prediction. "
            "'legacy_output_average' keeps the existing post-activation averaging behavior; "
            "'shared_feature_mean' and 'shared_feature_gated' use one shared head on fused encoder features."
        ),
    )
    cell_prop_fusion_dim: int = Field(
        default=256,
        gt=0,
        description="Shared fusion dimension used by the feature-based cell-proportion branch.",
    )
    cell_prop_head_hidden_dims: List[int] = Field(
        default_factory=lambda: [512, 256],
        description="Hidden dimensions for the shared cell-proportion head.",
    )
    cell_prop_head_dropout_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout rate used in the shared cell-proportion head and feature projectors.",
    )
    cell_prop_loss_type: Literal["mse", "l1_kl", "l1_rmse"] = Field(
        default="mse",
        description="Supervised cell-proportion loss family.",
    )
    cell_prop_loss_alpha_weight: float = Field(
        default=0.5,
        ge=0.0,
        validation_alias=AliasChoices("cell_prop_loss_alpha_weight", "cell_prop_loss_kl_weight"),
        serialization_alias="cell_prop_loss_alpha_weight",
        description=(
            "Alpha weight used by cell_prop_loss_type. For 'l1_kl' it is the KL multiplier; "
            "for 'l1_rmse' the loss is alpha * MAE + (1 - alpha) * RMSE."
        ),
    )
    cell_prop_loss_weighting: Literal["none", "low_prop_inverse"] = Field(
        default="none",
        description=(
            "Optional weighting mode for supervised cell-proportion loss. "
            "'low_prop_inverse' increases the relative contribution of low true proportions."
        ),
    )
    cell_prop_loss_low_prop_epsilon: float = Field(
        default=0.01,
        gt=0.0,
        description="Stabilizer used in low-proportion-aware loss weighting.",
    )
    cell_prop_loss_weight_clamp: Tuple[float, float] = Field(
        default=(1.0, 5.0),
        description="Clamp range for low-proportion-aware loss weights.",
    )


    # Mask fraction for input dropout
    mask_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of input features (genes) to mask for dropout"
    )

    # Whether to learn GEP residual compared to mean GEP of cell types instead of full GEP
    learn_gep_residual: bool = Field(
        default=False,
        description="Whether to learn GEP residuals compared to the mean GEP of each cell type (instead of learning the full GEP)"
    )
    learn_gep_residual_mode: Literal["zscore", "mean_centered"] = Field(
        default="zscore",
        description=(
            "Residual decoding mode used when learn_gep_residual=True. "
            "'zscore' keeps the current std-scaled residual path; "
            "'mean_centered' predicts scaled-log-space residuals relative to the "
            "cell-type-specific mean."
        ),
    )
    conditional_decoder_cell_type_emb_dim: int = Field(
        default=64,
        gt=0,
        description="Decoder-side cell-type embedding dimension used by conditioned decoders.",
    )
    conditional_decoder_context_dim: int = Field(
        default=256,
        gt=0,
        description="Projected bulk-context dimension used by conditioned decoders.",
    )
    conditional_decoder_dropout_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout rate used inside conditioned decoder FiLM blocks.",
    )

    # ==================== Validators ====================

    @field_validator('input_dim')
    @classmethod
    def validate_input_dim(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate input dimensions are positive."""
        if len(v) != 2:
            raise ValueError(f"input_dim must be a tuple of length 2, got {len(v)}")
        if any(dim <= 0 for dim in v):
            raise ValueError(f"All input dimensions must be positive, got {v}")
        return v

    @field_validator(
        'input_gene_list_fp', 'cell_type_fp', 'gene_mean_std_fp',
        'training_sct_cross_sample_gene_var_fp',
        check_fields=False,
        mode='before',
    )
    @classmethod
    def validate_file_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate file paths exist if provided."""
        if v is not None:
            # Convert string to Path if needed
            if isinstance(v, str):
                if v == '':  # Handle empty string
                    return None
                v = Path(v)

            # Check if file exists (only warn, don't fail)
            # This allows config creation before files exist
            if not v.exists():
                import warnings
                warnings.warn(f"File path does not exist yet: {v}")

        return v

    @field_validator('model_dir')
    @classmethod
    def validate_model_dir(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate and create model directory if needed."""
        if v is not None:
            if isinstance(v, str):
                if v == '':
                    return None
                v = Path(v)

            # Create directory if it doesn't exist
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)

        return v

    @field_validator(
        'encoder_hidden_dims',
        'decoder_hidden_dims',
        'cell_prop_head_hidden_dims',
        'deside_hidden_dims',
        'deside_pathway_hidden_dims',
    )
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        """Ensure all hidden dimensions are positive."""
        if not v:
            raise ValueError("Hidden dimensions list cannot be empty")
        if any(dim <= 0 for dim in v):
            raise ValueError(f"All hidden dimensions must be positive, got {v}")
        return v

    @field_validator(
        'encoder_dropout_rate',
        'decoder_dropout_rate',
        'deside_dropout_rate',
        'deside_pathway_dropout_rate',
    )
    @classmethod
    def validate_dropout_rates(cls, v: List[float]) -> List[float]:
        """Ensure dropout rates are in [0, 1]."""
        if not v:
            raise ValueError("Dropout rate list cannot be empty")
        if any(rate < 0 or rate > 1 for rate in v):
            raise ValueError(f"Dropout rates must be in [0, 1], got {v}")
        return v

    @field_validator('fusion_dropout_rate')
    @classmethod
    def validate_fusion_dropout(cls, v: Tuple[float, ...]) -> Tuple[float, ...]:
        """Ensure fusion dropout rates are in [0, 1]."""
        if any(rate < 0 or rate > 1 for rate in v):
            raise ValueError(f"Fusion dropout rates must be in [0, 1], got {v}")
        return v

    @field_validator('cell_prop_loss_weight_clamp')
    @classmethod
    def validate_cell_prop_loss_weight_clamp(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Ensure cell-proportion loss weight clamp bounds are valid."""
        mn, mx = v
        if mn <= 0 or mn > mx:
            raise ValueError(
                "cell_prop_loss_weight_clamp must satisfy 0 < min <= max"
            )
        return v

    @field_validator('encoders')
    @classmethod
    def validate_encoders(cls, v: List[str]) -> List[str]:
        """Validate encoder types."""
        if not v:
            raise ValueError("encoders list cannot be empty")

        valid_encoders = ['EncoderHybrid', 'EncoderMLP', 'EncoderSGNN', 'EncoderResMLP',
                          'GeneTransformerEncoder', 'EncoderPathNet']
        for encoder in v:
            if encoder not in valid_encoders:
                raise ValueError(
                    f"Unknown encoder type: {encoder}. "
                    f"Valid types: {valid_encoders}"
                )

        return v

    @field_validator('encoder_aliases')
    @classmethod
    def validate_encoder_aliases(cls, v: List[str]) -> List[str]:
        aliases: List[str] = []
        for alias in v:
            if not isinstance(alias, str):
                raise ValueError("encoder_aliases entries must be strings")
            normalized = alias.strip()
            if not normalized:
                raise ValueError("encoder_aliases entries cannot be empty")
            aliases.append(normalized)
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"encoder_aliases must be unique, got {aliases}")
        return aliases

    @field_validator('cell_prop_predictor_cls')
    @classmethod
    def validate_cell_prop_predictor_cls(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        normalized = v.strip()
        if not normalized:
            return None
        valid_predictors = {'DeSideCellPropPredictor'}
        if normalized not in valid_predictors:
            raise ValueError(
                f"Unknown cell_prop_predictor_cls: {normalized}. Valid types: {sorted(valid_predictors)}"
            )
        return normalized

    @field_validator('cell_prop_predictor_alias')
    @classmethod
    def validate_cell_prop_predictor_alias(cls, v: str) -> str:
        alias = v.strip()
        if not alias:
            raise ValueError("cell_prop_predictor_alias cannot be empty")
        return alias

    @model_validator(mode='after')
    def validate_architecture_consistency(self):
        """Ensure encoder/decoder architecture is consistent."""
        # Check encoder dimensions and dropout rates match
        if len(self.encoder_hidden_dims) != len(self.encoder_dropout_rate):
            raise ValueError(
                f"encoder_hidden_dims (len={len(self.encoder_hidden_dims)}) and "
                f"encoder_dropout_rate (len={len(self.encoder_dropout_rate)}) "
                f"must have the same length"
            )

        # Check decoder dimensions and dropout rates match
        if len(self.decoder_hidden_dims) != len(self.decoder_dropout_rate):
            raise ValueError(
                f"decoder_hidden_dims (len={len(self.decoder_hidden_dims)}) and "
                f"decoder_dropout_rate (len={len(self.decoder_dropout_rate)}) "
                f"must have the same length"
            )

        # Check fusion dimensions and dropout rates match
        if len(self.fusion_hidden_dims) != len(self.fusion_dropout_rate):
            raise ValueError(
                f"fusion_hidden_dims (len={len(self.fusion_hidden_dims)}) and "
                f"fusion_dropout_rate (len={len(self.fusion_dropout_rate)}) "
                f"must have the same length"
            )
        if len(self.deside_hidden_dims) != len(self.deside_dropout_rate):
            raise ValueError(
                f"deside_hidden_dims (len={len(self.deside_hidden_dims)}) and "
                f"deside_dropout_rate (len={len(self.deside_dropout_rate)}) "
                f"must have the same length"
            )
        if len(self.deside_pathway_hidden_dims) != len(self.deside_pathway_dropout_rate):
            raise ValueError(
                f"deside_pathway_hidden_dims (len={len(self.deside_pathway_hidden_dims)}) and "
                f"deside_pathway_dropout_rate (len={len(self.deside_pathway_dropout_rate)}) "
                f"must have the same length"
            )
        if self.deside_pathway_network and len(self.deside_pathway_hidden_dims) != len(self.deside_hidden_dims):
            raise ValueError(
                "deside_pathway_hidden_dims must match deside_hidden_dims in length when deside_pathway_network=True"
            )
        if len(self.deside_normalization_layer) != len(self.deside_hidden_dims) + 1:
            raise ValueError(
                "deside_normalization_layer length must equal len(deside_hidden_dims) + 1"
            )

        return self

    @model_validator(mode='after')
    def validate_encoder_output_routing(self):
        """Ensure encoder aliases and routing selections align with active encoders."""
        if not self.encoder_aliases:
            self.encoder_aliases = [f"encoder_{idx}" for idx in range(len(self.encoders))]

        if len(self.encoder_aliases) != len(self.encoders):
            raise ValueError(
                f"encoder_aliases (len={len(self.encoder_aliases)}) and "
                f"encoders (len={len(self.encoders)}) must have the same length"
            )
        if self.cell_prop_predictor_cls and self.cell_prop_predictor_alias in set(self.encoder_aliases):
            raise ValueError(
                "cell_prop_predictor_alias must not duplicate an encoder alias"
            )

        valid_sources = set(self.encoder_aliases) | {"fused"}
        if self.cell_prop_predictor_cls:
            valid_sources.add(self.cell_prop_predictor_alias)
        routing = self.encoder_output_routing
        cell_prop_context_routing = {
            "cell_prop_source": routing.cell_prop_source,
            "decoder_context_source": routing.decoder_context_source,
        }
        for field_name, source in cell_prop_context_routing.items():
            if source not in valid_sources:
                raise ValueError(
                    f"{field_name} must be one of {sorted(valid_sources)}, got {source!r}"
                )
        latent_valid_sources = set(self.encoder_aliases) | {"fused"}
        if routing.latent_posterior_source not in latent_valid_sources:
            raise ValueError(
                "latent_posterior_source must be one of "
                f"{sorted(latent_valid_sources)}, got {routing.latent_posterior_source!r}"
            )

        return self

    @model_validator(mode='after')
    def validate_cell_prop_consistency(self):
        """Ensure cell proportion prediction settings are consistent."""
        kld_p_weight = self.loss_coefficient.kld_p
        cell_prop_weight = self.loss_coefficient.cell_prop
        existence_weight = self.loss_coefficient.cell_type_existence_weight
        activation_function = self.cell_prop_activation_function

        if kld_p_weight < 0:
            raise ValueError(
                f"loss_coefficient['kld_p'] must be >= 0, got {kld_p_weight}."
            )

        if cell_prop_weight < 0:
            raise ValueError(
                f"loss_coefficient['cell_prop'] must be >= 0, got {cell_prop_weight}."
            )

        if kld_p_weight > 0 and not self.predict_cell_prop:
            raise ValueError(
                f"loss_coefficient['kld_p'] = {kld_p_weight} > 0 "
                f"but predict_cell_prop=False. "
                f"Either set predict_cell_prop=True or set kld_p to 0."
            )

        if cell_prop_weight > 0 and not self.predict_cell_prop:
            raise ValueError(
                f"loss_coefficient['cell_prop'] = {cell_prop_weight} > 0 "
                f"but predict_cell_prop=False. "
                f"Either set predict_cell_prop=True or set cell_prop to 0."
            )

        if existence_weight > 0 and not self.predict_cell_prop:
            raise ValueError(
                f"loss_coefficient['cell_type_existence_weight'] = {existence_weight} > 0 "
                f"but predict_cell_prop=False. "
                f"Either set predict_cell_prop=True or set cell_type_existence_weight to 0."
            )

        if not self.predict_cell_prop:
            return self

        if self.cell_prop_predictor_cls == "DeSideCellPropPredictor" and activation_function != "sigmoid":
            raise ValueError(
                "DeSideCellPropPredictor requires cell_prop_activation_function='sigmoid' "
                "to match standalone DeSide's non-cancer sigmoid workflow."
            )

        if activation_function == "sigmoid":
            if not self.cancer_cell_type_name or not self.cancer_cell_type_name.strip():
                raise ValueError(
                    "cancer_cell_type_name must be set when "
                    "cell_prop_activation_function='sigmoid'."
                )
            if kld_p_weight > 0:
                raise ValueError(
                    "loss_coefficient['kld_p'] must be 0 when "
                    "cell_prop_activation_function='sigmoid' because the sigmoid branch "
                    "does not define a Dirichlet posterior."
                )
        elif activation_function in {"softmax", "sigmoid_all_norm"}:
            if kld_p_weight > 0:
                raise ValueError(
                    "loss_coefficient['kld_p'] must be 0 when "
                    f"cell_prop_activation_function='{activation_function}' because this branch "
                    "does not define a Dirichlet posterior."
                )

        if cell_prop_weight == 0 and kld_p_weight == 0:
            import warnings
            warnings.warn(
                "Both loss_coefficient['cell_prop']=0 and loss_coefficient['kld_p']=0. "
                "If predict_cell_prop=True, the prediction head will not receive direct supervision "
                "or Dirichlet regularization during training."
            )

        return self

    @model_validator(mode='after')
    def validate_gnn_consistency(self):
        """Ensure GNN settings are consistent with input dimensions."""
        # Check if number of genes matches input dimension
        if self.input_dim[1] != self.gnn_n_genes:
            import warnings
            warnings.warn(
                f"input_dim[1]={self.input_dim[1]} does not match "
                f"gnn_n_genes={self.gnn_n_genes}. "
                f"This may cause dimension mismatch errors."
            )

        return self

    @model_validator(mode='after')
    def validate_residual_mode_consistency(self):
        """Ensure residual-mode-specific losses are only used with compatible branches."""
        if not self.learn_gep_residual:
            return self

        if self.learn_gep_residual_mode != "mean_centered":
            return self

        if self.loss_coefficient.z_score_kl_weight > 0:
            raise ValueError(
                "loss_coefficient['z_score_kl_weight'] must be 0 when "
                "learn_gep_residual_mode='mean_centered'."
            )

        if self.loss_coefficient.z_score_reg_weight > 0:
            raise ValueError(
                "loss_coefficient['z_score_reg_weight'] must be 0 when "
                "learn_gep_residual_mode='mean_centered'."
            )

        return self

    # ==================== Helper Methods ====================

    def get_encoder_architecture(self) -> List[Tuple[int, float]]:
        """Get encoder architecture as list of (dim, dropout) tuples."""
        return list(zip(self.encoder_hidden_dims, self.encoder_dropout_rate))

    def get_decoder_architecture(self) -> List[Tuple[int, float]]:
        """Get decoder architecture as list of (dim, dropout) tuples."""
        return list(zip(self.decoder_hidden_dims, self.decoder_dropout_rate))

    def get_fusion_architecture(self) -> List[Tuple[int, float]]:
        """Get fusion architecture as list of (dim, dropout) tuples."""
        return list(zip(self.fusion_hidden_dims, self.fusion_dropout_rate))

    def summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "model_type": "VAE-Deconvolution",
            "input_shape": self.input_dim,
            "latent_dim": self.latent_dim,
            "n_cell_types": self.n_cell_types,
            "encoder_layers": len(self.encoder_hidden_dims),
            "decoder_layers": len(self.decoder_hidden_dims),
            "fusion_layers": len(self.fusion_hidden_dims),
            "gnn_layers": self.gnn_num_layers,
            "total_params_estimate": self.estimate_total_params(),
            "encoder_types": self.encoders,
            "predict_cell_prop": self.predict_cell_prop,
            "cell_prop_activation_function": self.cell_prop_activation_function,
            "learn_gep_residual": self.learn_gep_residual,
            "learn_gep_residual_mode": self.learn_gep_residual_mode,
            "loss_settings": {
                "beta": self.loss_coefficient.beta,
                "gamma": self.loss_coefficient.gamma,
                "kld_p_weight": self.loss_coefficient.kld_p,
                "cell_prop_weight": self.loss_coefficient.cell_prop,
                "gene_mean_weight": self.loss_coefficient.gene_mean_weight,
                "gene_std_weight": self.loss_coefficient.gene_std_weight,
                "cross_sample_gene_var_weight": self.loss_coefficient.cross_sample_gene_var_weight,
                "cell_type_sct_gep_weight": self.loss_coefficient.cell_type_sct_gep_weight,
                "cell_type_existence_weight": self.loss_coefficient.cell_type_existence_weight,
                "z_score_kl_weight": self.loss_coefficient.z_score_kl_weight,
            }
        }

    def estimate_total_params(self) -> int:
        """Estimate total number of model parameters."""
        total = 0

        # Encoder parameters
        prev_dim = self.input_dim[1]
        for dim in self.encoder_hidden_dims:
            total += prev_dim * dim + dim  # weights + bias
            prev_dim = dim

        # Latent layer
        total += prev_dim * (self.latent_dim * self.n_cell_types) * 2  # mu and logvar

        # Decoder parameters
        prev_dim = self.latent_dim * self.n_cell_types
        for dim in self.decoder_hidden_dims:
            total += prev_dim * dim + dim
            prev_dim = dim

        # Output layer
        total += prev_dim * self.input_dim[1] + self.input_dim[1]

        return total

    def validate_paths_exist(self) -> Dict[str, bool]:
        """Check which file paths exist."""
        return {
            "ppi_file_path": self.ppi_file_path.exists() if self.ppi_file_path else False,
            "input_gene_list_fp": self.input_gene_list_fp.exists() if self.input_gene_list_fp else False,
            "cell_type_fp": self.cell_type_fp.exists() if self.cell_type_fp else False,
            "gene_mean_std_fp": self.gene_mean_std_fp.exists() if self.gene_mean_std_fp else False,
            "model_dir": self.model_dir.exists() if self.model_dir else False,
        }


@dataclass
class EvaluationConfig:
    """evaluation configuration"""
    n_samples: int = 3
    visualize_n_sample: int = 3
    cell_prop_threshold: float = 0.005
    plot_cell_proportions: bool = True
    plot_single_cell_gep: bool = True
    plot_bulk_gep: bool = True
    plot_latent_space: bool = True
    val_batch_size: int = 128
    remove_low_var_genes: bool = False

    # UMAP settings
    n_neighbors: int = 15
    min_dist: float = 0.1

    # Plotting
    figsize: Tuple[float, float] = (3.5, 3.5)
    rasterized: bool = True
    show_metrics: bool = True
    figure_format: str = 'png'  # 'png', 'svg', or 'pdf' etc.

    # Results
    save_reconstructed_gep: bool = True  # Whether to save reconstructed GEPs for all test samples
    save_cell_type_specific_gep_metrics: bool = True
    save_bulk_gep_input: bool = True  # Whether to save the input bulk GEPs
    save_recon_bulk_gep_conv: bool = True  # Whether to save the reconstructed bulk GEPs


def _to_plain_yaml_data(value: Any) -> Any:
    """Convert config objects into plain YAML-safe Python data."""
    if isinstance(value, BaseModel):
        return _to_plain_yaml_data(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_yaml_data(asdict(value))
    if isinstance(value, dict):
        return {
            key: _to_plain_yaml_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_plain_yaml_data(item) for item in value]
    return value


@dataclass
class VAEDeconConfig:
    """The complete configuration for VAEDecon"""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Creates configuration from a dictionary"""
        data = DataConfig(**config_dict.get('data', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        evaluation = EvaluationConfig(**config_dict.get('evaluation', {}))
        cls._validate_cross_section_config(
            data=data,
            training=training,
            model=model,
        )
        return cls(
            data=data,
            training=training,
            model=model,
            evaluation=evaluation
        )

    @staticmethod
    def _validate_cross_section_config(
        *,
        data: DataConfig,
        training: TrainingConfig,
        model: ModelConfig,
    ) -> None:
        staged_training = training.staged_training
        if staged_training is not None and staged_training.enabled:
            if not model.cell_prop_predictor_cls:
                raise ValueError(
                    "training.staged_training requires model.cell_prop_predictor_cls to be configured"
                )
            if not model.predict_cell_prop:
                raise ValueError(
                    "training.staged_training requires model.predict_cell_prop=True"
                )

            selected_stages = staged_training.run_stages or [stage.name for stage in staged_training.stages]
            if any(stage_name in {"reconstruction_training", "joint_finetune"} for stage_name in selected_stages):
                predictor_alias = model.cell_prop_predictor_alias
                if model.encoder_output_routing.cell_prop_source != predictor_alias:
                    raise ValueError(
                        "training.staged_training reconstruction/joint stages require "
                        "model.encoder_output_routing.cell_prop_source to use the cell_prop_predictor alias"
                    )
                if model.encoder_output_routing.decoder_context_source != predictor_alias:
                    raise ValueError(
                        "training.staged_training reconstruction/joint stages require "
                        "model.encoder_output_routing.decoder_context_source to use the cell_prop_predictor alias"
                    )

        adaptive_schedule = training.adaptive_aux_loss_schedule
        if adaptive_schedule is None or not adaptive_schedule.enabled:
            return

        cell_prop_schedule = adaptive_schedule.targets.get("cell_prop")
        if cell_prop_schedule is not None and max(cell_prop_schedule.range) > 0 and not model.predict_cell_prop:
            raise ValueError(
                "adaptive_aux_loss_schedule target 'cell_prop' requires model.predict_cell_prop=True "
                "when its configured range includes values > 0."
            )

        sct_gep_schedule = adaptive_schedule.targets.get("cell_type_sct_gep_weight")
        if sct_gep_schedule is not None and max(sct_gep_schedule.range) > 0:
            training_target_sets = dict(data.training_target_sets or {})
            if not training_target_sets:
                raise ValueError(
                    "adaptive_aux_loss_schedule target 'cell_type_sct_gep_weight' requires "
                    "data.training_target_sets when its configured range includes values > 0."
                )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path):
        """Loads configuration from a YAML file"""
        import yaml
        from yaml.constructor import ConstructorError
        from pathlib import Path as _Path
        yaml_path = _Path(yaml_path)

        class _NoDuplicateSafeLoader(yaml.SafeLoader):
            pass

        def _construct_mapping(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise ValueError(f"Duplicate key '{key}' in YAML: {yaml_path}")
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        _NoDuplicateSafeLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            _construct_mapping,
        )

        with open(yaml_path, 'r', encoding='utf-8') as f:
            try:
                config_dict = yaml.load(f, Loader=_NoDuplicateSafeLoader)
            except ConstructorError:
                f.seek(0)
                config_dict = _to_plain_yaml_data(
                    yaml.load(f, Loader=yaml.UnsafeLoader)
                )
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: str | Path):
        """Saves the configuration to a YAML file"""
        import yaml
        config_dict = _to_plain_yaml_data(
            {
                "data": self.data,
                "training": self.training,
                "model": self.model,
                "evaluation": self.evaluation,
            }
        )
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
