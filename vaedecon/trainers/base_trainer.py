import csv
import json
import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import logging
import platform
import shutil

from typing import Any, Dict, Optional, Union, Type

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from vaedecon.data.datasets import BaseDataset, collate_dataset_output
from vaedecon.models import BaseAE
from vaedecon.configs import TrainingConfig, DataConfig
from vaedecon.models.base import set_seed
from vaedecon.customexception import DatasetError
from .training_scheduler import build_scheduler, WarmupThenReduceOnPlateau

# Optional dependency
try:
    import anndata
    HAS_ANNDATA = True
except Exception:
    HAS_ANNDATA = False
    anndata = None  # type: ignore

logger = logging.getLogger(__name__)


def _unwrap_stage_model(model: BaseAE) -> BaseAE:
    """Return the underlying BaseAE when torch.compile wraps it."""
    return getattr(model, "_orig_mod", model)


def _resolve_stage_named_modules(model: BaseAE) -> Dict[str, Any]:
    """Resolve logical stage module groups on the current model."""
    base_model = _unwrap_stage_model(model)
    modules: Dict[str, Any] = {}
    encoders = getattr(base_model, "encoders", None)
    if encoders is not None:
        modules["encoders"] = encoders
    decoder = getattr(base_model, "decoder", None)
    if decoder is not None:
        modules["decoder"] = decoder
    cell_prop_predictor = getattr(base_model, "cell_prop_predictor", None)
    if cell_prop_predictor is not None:
        modules["cell_prop_predictor"] = cell_prop_predictor
    return modules


def _apply_stage_module_modes(model: BaseAE, module_mode_overrides: Optional[Dict[str, str]]) -> None:
    """Apply train/eval mode overrides to logical module groups."""
    if not module_mode_overrides:
        return
    modules = _resolve_stage_named_modules(model)
    for module_name, mode in module_mode_overrides.items():
        module = modules.get(module_name)
        if module is None:
            continue
        if str(mode).lower() == "eval":
            module.eval()
        else:
            module.train()


def _resolve_linear_schedule_value(schedule: Any, epoch: int) -> float:
    """Return the scheduled scalar value for a given epoch."""
    start_epoch = int(getattr(schedule, "start_epoch", 0))
    end_epoch = int(getattr(schedule, "end_epoch", start_epoch))
    start_value = float(getattr(schedule, "start_value", 0.0))
    end_value = float(getattr(schedule, "end_value", start_value))
    if epoch <= start_epoch:
        return start_value
    if epoch >= end_epoch:
        return end_value
    if end_epoch == start_epoch:
        return end_value
    progress = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    return start_value + progress * (end_value - start_value)


def _apply_aux_loss_schedules(model: BaseAE, training_config: TrainingConfig, epoch: int) -> None:
    """Apply configured epoch-based auxiliary schedules to the live model config."""
    schedules = getattr(training_config, "aux_loss_schedules", None) or {}
    if not schedules:
        return

    loss_coefficient = model.model_config.loss_coefficient
    target_map = {
        "cell_type_sct_gep_weight": loss_coefficient,
        "hierarchical_code_weight": loss_coefficient,
        "cell_type_existence_weight": loss_coefficient,
        "cell_type_existence_shift_scale": model.model_config,
    }
    for target_name, schedule in schedules.items():
        target_obj = target_map.get(target_name)
        if target_obj is None:
            continue
        setattr(target_obj, target_name, _resolve_linear_schedule_value(schedule, epoch))


@dataclass
class AdaptiveAuxLossTargetState:
    name: str
    current_value: float
    start_value: float
    end_value: float
    min_value: float
    max_value: float
    direction: int
    step_size: float
    reverse_on_plateau: bool


@dataclass
class AdaptiveAuxLossScheduleState:
    monitor: str
    min_epoch_before_trigger: int
    trigger_patience: int
    trigger_min_delta: float
    cooldown_epochs: int
    update_interval_epochs: int
    pair_targets: bool
    targets: Dict[str, AdaptiveAuxLossTargetState] = field(default_factory=dict)
    best_metric: Optional[float] = None
    epochs_since_improvement: int = 0
    last_trigger_epoch: Optional[int] = None
    trace_rows: list[Dict[str, Any]] = field(default_factory=list)


def _get_adaptive_aux_loss_target_map(model: BaseAE) -> Dict[str, tuple[Any, str]]:
    loss_coefficient = model.model_config.loss_coefficient
    return {
        "cell_prop": (loss_coefficient, "cell_prop"),
        "cell_type_sct_gep_weight": (loss_coefficient, "cell_type_sct_gep_weight"),
    }


def _set_adaptive_aux_loss_target_value(model: BaseAE, target_name: str, value: float) -> None:
    target_map = _get_adaptive_aux_loss_target_map(model)
    if target_name not in target_map:
        raise ValueError(f"Unsupported adaptive aux loss target: {target_name}")
    target_obj, attr_name = target_map[target_name]
    setattr(target_obj, attr_name, float(value))


def _adaptive_trace_weight_key(target_name: str) -> str:
    if target_name.endswith("_weight"):
        return target_name
    return f"{target_name}_weight"


def _adaptive_trace_next_weight_key(target_name: str) -> str:
    return f"next_{_adaptive_trace_weight_key(target_name)}"


def _adaptive_trace_direction_key(target_name: str) -> str:
    return f"{target_name}_direction"


def _adaptive_trace_next_direction_key(target_name: str) -> str:
    return f"next_{target_name}_direction"


def _initialize_adaptive_aux_loss_schedule(
    model: BaseAE,
    training_config: TrainingConfig,
) -> Optional[AdaptiveAuxLossScheduleState]:
    schedule_cfg = getattr(training_config, "adaptive_aux_loss_schedule", None)
    if schedule_cfg is None or not getattr(schedule_cfg, "enabled", False):
        return None

    target_states: Dict[str, AdaptiveAuxLossTargetState] = {}
    for target_name, target_cfg in schedule_cfg.targets.items():
        start_value = float(target_cfg.range[0])
        end_value = float(target_cfg.range[1])
        direction = 1 if end_value > start_value else -1
        state = AdaptiveAuxLossTargetState(
            name=target_name,
            current_value=start_value,
            start_value=start_value,
            end_value=end_value,
            min_value=min(start_value, end_value),
            max_value=max(start_value, end_value),
            direction=direction,
            step_size=float(target_cfg.step_size),
            reverse_on_plateau=bool(target_cfg.reverse_on_plateau),
        )
        target_states[target_name] = state
        _set_adaptive_aux_loss_target_value(model, target_name, state.current_value)

    return AdaptiveAuxLossScheduleState(
        monitor=str(schedule_cfg.monitor),
        min_epoch_before_trigger=int(schedule_cfg.min_epoch_before_trigger),
        trigger_patience=int(schedule_cfg.trigger_patience),
        trigger_min_delta=float(schedule_cfg.trigger_min_delta),
        cooldown_epochs=int(schedule_cfg.cooldown_epochs),
        update_interval_epochs=int(schedule_cfg.update_interval_epochs),
        pair_targets=bool(schedule_cfg.pair_targets),
        targets=target_states,
    )


def _update_adaptive_plateau_state(
    state: AdaptiveAuxLossScheduleState,
    monitored_metric: Optional[float],
) -> None:
    if monitored_metric is None:
        return
    if state.best_metric is None or monitored_metric < (state.best_metric - state.trigger_min_delta):
        state.best_metric = float(monitored_metric)
        state.epochs_since_improvement = 0
        return
    state.epochs_since_improvement += 1


def _adaptive_schedule_update_due(
    state: AdaptiveAuxLossScheduleState,
    epoch: int,
) -> bool:
    return ((epoch + 1) % max(1, state.update_interval_epochs)) == 0


def _adaptive_schedule_cooldown_active(
    state: AdaptiveAuxLossScheduleState,
    epoch: int,
) -> bool:
    if state.last_trigger_epoch is None:
        return False
    return (epoch - state.last_trigger_epoch) < state.cooldown_epochs


def _step_adaptive_aux_loss_schedule(
    model: BaseAE,
    state: Optional[AdaptiveAuxLossScheduleState],
    *,
    epoch: int,
    monitored_metric: Optional[float],
) -> Optional[Dict[str, Any]]:
    if state is None:
        return None

    weights_before = {
        target_name: float(target_state.current_value)
        for target_name, target_state in state.targets.items()
    }
    directions_before = {
        target_name: int(target_state.direction)
        for target_name, target_state in state.targets.items()
    }

    _update_adaptive_plateau_state(state, monitored_metric)
    update_due = _adaptive_schedule_update_due(state, epoch)
    cooldown_active = _adaptive_schedule_cooldown_active(state, epoch)
    plateau_reached = (
        epoch >= state.min_epoch_before_trigger
        and state.epochs_since_improvement >= state.trigger_patience
    )
    trigger_fired = False

    if update_due and plateau_reached and not cooldown_active:
        reversible_targets = [
            target_state
            for target_state in state.targets.values()
            if target_state.reverse_on_plateau
        ]
        if reversible_targets:
            for target_state in reversible_targets:
                target_state.direction *= -1
            state.last_trigger_epoch = epoch
            state.epochs_since_improvement = 0
            if monitored_metric is not None:
                state.best_metric = float(monitored_metric)
            trigger_fired = True

    if update_due:
        for target_state in state.targets.values():
            next_value = target_state.current_value + (target_state.direction * target_state.step_size)
            target_state.current_value = min(
                target_state.max_value,
                max(target_state.min_value, next_value),
            )
            _set_adaptive_aux_loss_target_value(model, target_state.name, target_state.current_value)

    weights_after = {
        target_name: float(target_state.current_value)
        for target_name, target_state in state.targets.items()
    }
    directions_after = {
        target_name: int(target_state.direction)
        for target_name, target_state in state.targets.items()
    }
    trace_row: Dict[str, Any] = {
        "epoch": int(epoch),
        "monitor": state.monitor,
        "monitored_metric": None if monitored_metric is None else float(monitored_metric),
        "update_due": int(update_due),
        "plateau_reached": int(plateau_reached),
        "cooldown_active": int(cooldown_active),
        "trigger_fired": int(trigger_fired),
        "epochs_since_improvement": int(state.epochs_since_improvement),
    }
    for target_name in sorted(state.targets.keys()):
        trace_row[_adaptive_trace_weight_key(target_name)] = weights_before[target_name]
        trace_row[_adaptive_trace_direction_key(target_name)] = directions_before[target_name]
        trace_row[_adaptive_trace_next_weight_key(target_name)] = weights_after[target_name]
        trace_row[_adaptive_trace_next_direction_key(target_name)] = directions_after[target_name]

    state.trace_rows.append(trace_row)
    return trace_row


def get_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,  # Default to None, decides based on num_workers
    collate_fn: Optional[callable] = None,
) -> DataLoader:
    """Creates a DataLoader for the given dataset with potentially optimized num_workers.

    Args:
        dataset: The dataset to load.
        batch_size: How many samples per batch to load.
        shuffle: Whether to shuffle the data at every epoch.
        num_workers: Number of subprocesses to use for data loading.
                     If None, a heuristic will be used. Defaults to None.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning them.
                    Useful when loading data to GPU.
        persistent_workers: If True and num_workers > 0, workers will not be shut down
                            after one epoch. Can speed up training.
        collate_fn: Function to merge a list of samples to form a mini-batch.
    """

    if num_workers is None:
        available_cpus = os.cpu_count()
        if available_cpus:
            # Heuristic: Use half the available CPUs, but cap it.
            # On Windows, multi-processing for DataLoader can sometimes be slow or problematic
            # if not handled carefully (e.g., in `if __name__ == '__main__':`).
            # For CPU-bound __getitem__ (not your case if GEPDataset is preprocessed), more workers help.
            # For IO-bound or already fast __getitem__, fewer workers or 0 can be better.
            if platform.system() == "Windows":
                # Often recommended to use 0 or 1 on Windows for stability/performance with PyTorch DataLoader
                # unless the __getitem__ is significantly heavy.
                num_workers = 0  # Safer default for Windows. Test if >0 helps your specific case.
                logger.info(f"OS is Windows, num_workers defaulted to {num_workers}. "
                            "Consider manual tuning if data loading is a bottleneck.")
            else:
                # A common heuristic: number of GPUs * 2 or 4, or num_cpus / 3 (multiple programs may be running).
                # Let's use a conservative approach:
                num_workers = max(1, int(available_cpus // 3)) if available_cpus > 3 else (1 if available_cpus > 0 else 0)
                num_workers = min(num_workers, 8)  # Cap at 8 to avoid excessive resource usage
                # num_workers = 0  # for multiple programs running, otherwise the first program will be killed by the new one

            # If dataset is small and __getitem__ is trivial (e.g., pre-loaded tensors),
            # num_workers > 0 might add overhead.
            # For GEPDataset (once preprocessed and cached), data is in memory, so __getitem__ is fast.
            # In such cases, num_workers=0 or 1 might be optimal.
            # This heuristic is a general starting point; empirical testing is best.
            logger.info(f"num_workers not specified, automatically set to {num_workers} "
                        f"(available CPUs: {available_cpus}).")
        else:
            num_workers = 0  # Fallback if os.cpu_count() is not available
            logger.info(f"Could not determine CPU count, defaulting num_workers to {num_workers}.")

    # Decide on persistent_workers default
    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False

    # pin_memory is only effective when using CUDA
    actual_pin_memory = pin_memory if torch.cuda.is_available() else False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=actual_pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


def _is_map_style_dataset(obj) -> bool:
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


# -----------------------------
# Data Adapter (thin + strict)
# -----------------------------
class DataAdapter:
    """Convert input data into Dataset/DataLoader-ready objects."""

    def process_array_like(
        self,
        data: Union[np.ndarray, torch.Tensor, anndata.AnnData],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        x = self._to_tensor(data, dtype=dtype)
        self._validate_tensor(x)
        return x

    @staticmethod
    def to_dataset(
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        if labels is None:
            labels = torch.ones(data.shape[0], dtype=torch.float32)
        return BaseDataset(data, labels)

    def prepare_for_training(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor, Dataset, DataLoader, anndata.AnnData, BaseDataset]],
        data_type: str = "train",
    ) -> Optional[Union[Dataset, DataLoader]]:
        if data is None:
            return None

        if isinstance(data, DataLoader):
            logger.info(f"Using provided {data_type} DataLoader.")
            return data

        if isinstance(data, BaseDataset):
            logger.info(f"Using provided {data_type} Dataset.")
            _check_dataset(data)
            return data

        if HAS_ANNDATA and isinstance(data, anndata.AnnData):
            logger.info(f"Processing {data_type} AnnData...")
            x = self.process_array_like(data)
            ds = self.to_dataset(x)
            _check_dataset(ds)
            return ds

        if isinstance(data, (np.ndarray, torch.Tensor)):
            logger.info(f"Processing {data_type} array/tensor...")
            x = self.process_array_like(data)
            ds = self.to_dataset(x)
            _check_dataset(ds)
            return ds

        if _is_map_style_dataset(data):
            logger.info(f"Using provided {data_type} map-style dataset: {type(data)}")
            _check_dataset(data)
            return data

        raise TypeError(f"Unsupported {data_type} data type: {type(data)}")

    @staticmethod
    def _to_tensor(
        data: Union[np.ndarray, torch.Tensor, anndata.AnnData],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if HAS_ANNDATA and isinstance(data, anndata.AnnData):
            x = data.X
            if hasattr(x, "toarray"):  # scipy sparse
                x = x.toarray()
            return torch.as_tensor(x, dtype=dtype)

        if torch.is_tensor(data):
            return data.to(dtype=dtype)

        if isinstance(data, np.ndarray):
            return torch.as_tensor(data, dtype=dtype)

        raise TypeError(f"Cannot convert type {type(data)} to tensor.")

    @staticmethod
    def _validate_tensor(x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [n_samples, n_features], got shape={tuple(x.shape)}")
        if x.numel() == 0:
            raise ValueError("Input data is empty.")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or Inf.")


class PLTrainer(L.LightningModule):
    """PyTorch Lightning-based trainer for BaseAE models."""

    def __init__(
        self,
        model: BaseAE,
        training_config: TrainingConfig,
        debug_model: Optional[bool] = False,
        monitor_metric: str = "val_loss",
        module_mode_overrides: Optional[Dict[str, str]] = None,
    ):
        """Initializes the PLTrainer.

        Args:
            model: The BaseAE model to train.
            training_config: The training configuration.
            debug_model: Whether to enable more detailed logging behavior.
            monitor_metric: Metric name monitored by scheduler/checkpoint if needed.
        """
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.model_name = model.model_name
        self.debug_model = debug_model
        self.monitor_metric = monitor_metric
        self.module_mode_overrides = dict(module_mode_overrides or {})

        cfg_metrics = getattr(training_config, "prog_bar_metrics", None)
        if cfg_metrics:
            self.prog_bar_metrics = set(cfg_metrics)
        else:
            self.prog_bar_metrics = {
                "loss",
                "kld",
                "recon_loss_conv",
                "low_mean_std_gene_loss",
                "z_score_kl_loss",
                "cross_sample_gene_var_loss",
                "per_sample_residual_var_loss",
                "inter_sample_similarity_loss",
                "repulsion_loss",
                "attractor_loss",
            }
        self.prog_bar_metrics.add("loss")
        self._adaptive_aux_schedule_state = _initialize_adaptive_aux_loss_schedule(
            model=self.model,
            training_config=self.training_config,
        )

    def on_fit_start(self) -> None:
        _apply_stage_module_modes(self.model, self.module_mode_overrides)

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Forward pass of the model."""
        return self.model(inputs, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        output = self(batch)
        self.log_learning_rate()

        # lo = self.model.model_config.loss_coefficient
        # self.log("w_gene_mean", float(lo.gene_mean_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log("w_gene_std", float(lo.gene_std_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.loss_monitor(
            step="train",
            output=output,
            loss_types=tuple(self.prog_bar_metrics),
        )

        return output.loss

    def on_train_batch_start(self, batch: Dict[str, Any], batch_idx: int) -> None:
        cfg = self.training_config
        schedule = getattr(cfg, "gene_stat_weight_schedule", None)
        if schedule != "linear":
            return

        steps = getattr(cfg, "gene_stat_weight_schedule_steps", None)
        if steps is None or steps <= 0:
            # gene_stat_weight_schedule_epochs controls how many epochs the linear schedule spans.
            # It is converted into a step budget via: steps = epochs * num_training_batches.
            # If unset/non-positive, it falls back to warmup_epochs; if that is also non-positive,
            # the schedule is disabled (no per-batch updates to gene_mean_weight/gene_std_weight).
            epochs = getattr(cfg, "gene_stat_weight_schedule_epochs", None)
            if epochs is None or epochs <= 0:
                epochs = getattr(cfg, "warmup_epochs", 0)
            num_batches = getattr(self.trainer, "num_training_batches", 0) or 0
            if epochs <= 0 or num_batches <= 0:
                return
            steps = int(epochs * num_batches)
            if steps <= 0:
                return

        progress = float(self.global_step) / float(max(1, steps))
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0

        lo = self.model.model_config.loss_coefficient
        mean_start = cfg.gene_mean_weight_start if hasattr(cfg, "gene_mean_weight_start") else 0.0
        std_start = cfg.gene_std_weight_start if hasattr(cfg, "gene_std_weight_start") else 0.0

        mean_end = cfg.gene_mean_weight_end if hasattr(cfg, "gene_mean_weight_end") else 0.0
        std_end = cfg.gene_std_weight_end if hasattr(cfg, "gene_std_weight_end") else 1.0

        lo.gene_mean_weight = max(0.0, mean_start + (mean_end - mean_start) * progress)
        lo.gene_std_weight = max(0.0, std_start + (std_end - std_start) * progress)

        # self.log("w_gene_mean", float(lo.gene_mean_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log("w_gene_std", float(lo.gene_std_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_train_epoch_start(self) -> None:
        _apply_stage_module_modes(self.model, self.module_mode_overrides)
        _apply_aux_loss_schedules(
            model=self.model,
            training_config=self.training_config,
            epoch=int(self.current_epoch),
        )

    def on_validation_epoch_start(self) -> None:
        _apply_stage_module_modes(self.model, self.module_mode_overrides)

    def _get_monitored_metric_value(self, metric_name: str) -> Optional[float]:
        monitored_value = self.trainer.callback_metrics.get(metric_name, None)
        if monitored_value is not None and torch.is_tensor(monitored_value):
            monitored_value = monitored_value.item()
        if monitored_value is None:
            return None
        return float(monitored_value)

    def _log_adaptive_schedule_epoch_metrics(
        self,
        *,
        trace_row: Dict[str, Any],
    ) -> None:
        for target_name in sorted(self._adaptive_aux_schedule_state.targets.keys()):
            weight_key = _adaptive_trace_weight_key(target_name)
            direction_key = _adaptive_trace_direction_key(target_name)
            next_weight_key = _adaptive_trace_next_weight_key(target_name)
            next_direction_key = _adaptive_trace_next_direction_key(target_name)
            self.log(
                f"schedule_{weight_key}",
                float(trace_row[next_weight_key]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"schedule_{direction_key}",
                float(trace_row[next_direction_key]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"schedule_previous_{weight_key}",
                float(trace_row[weight_key]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"schedule_previous_{direction_key}",
                float(trace_row[direction_key]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if trace_row["monitored_metric"] is not None:
            self.log(
                "schedule_monitored_metric",
                float(trace_row["monitored_metric"]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        self.log(
            "schedule_trigger_fired",
            float(trace_row["trigger_fired"]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single validation step."""
        output = self(batch)

        self.loss_monitor(
            step="val",
            output=output,
            loss_types=tuple(self.prog_bar_metrics),
        )

        return output.loss

    def on_train_epoch_end(self):
        """Optional debug info after each epoch."""
        adaptive_trace_row = _step_adaptive_aux_loss_schedule(
            model=self.model,
            state=self._adaptive_aux_schedule_state,
            epoch=int(self.current_epoch),
            monitored_metric=self._get_monitored_metric_value(
                getattr(
                    self._adaptive_aux_schedule_state,
                    "monitor",
                    self.monitor_metric,
                )
            ) if self._adaptive_aux_schedule_state is not None else None,
        )
        if adaptive_trace_row is not None:
            self._log_adaptive_schedule_epoch_metrics(trace_row=adaptive_trace_row)

        if self.debug_model:
            opt = self.optimizers()
            current_lr = None
            if opt is not None:
                current_lr = opt.param_groups[0]["lr"]

            monitored_value = self._get_monitored_metric_value(self.monitor_metric)

            if current_lr is not None:
                print(
                    f"Epoch {self.current_epoch}: "
                    f"lr={current_lr:.2e}, "
                    f"{self.monitor_metric}={monitored_value}"
                )

    def get_adaptive_aux_schedule_trace_rows(self) -> list[Dict[str, Any]]:
        if self._adaptive_aux_schedule_state is None:
            return []
        return list(self._adaptive_aux_schedule_state.trace_rows)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizer and learning rate scheduler.

        Supports an optional linear warmup phase prepended to any scheduler.
        Controlled by training_config.warmup_epochs (default 0 = no warmup).

        Scheduler combinations:
          - warmup_epochs=0 : original behaviour, unchanged
          - warmup_epochs>0 + scheduler_cls="ReduceLROnPlateau"
                            : LinearLR warmup → ReduceLROnPlateau
                              (via _WarmupReduceOnPlateauScheduler wrapper)
          - warmup_epochs>0 + scheduler_cls="CosineAnnealingLR"
                            : LinearLR warmup → CosineAnnealingLR
                              (via SequentialLR, natively supported by Lightning)
          - warmup_epochs>0 + scheduler_cls=None
                            : LinearLR warmup only, then constant lr
        """

        # ── 1. Build optimizer (unchanged from original) ─────────────────────────
        optimizer_cls = getattr(optim, self.training_config.optimizer_cls)
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters remain after staged-training module freezing.")
        if self.training_config.optimizer_params is not None:
            optimizer = optimizer_cls(
                trainable_parameters,
                lr=self.training_config.learning_rate,
                **self.training_config.optimizer_params,
            )
        else:
            optimizer = optimizer_cls(
                trainable_parameters,
                lr=self.training_config.learning_rate,
            )

        # ── 2. No scheduler ───────────────────────────────────────────────────────
        if self.training_config.scheduler_cls is None:
            return {"optimizer": optimizer}

        # ── 3. Build scheduler via build_scheduler ────────────────────────────────
        scheduler = build_scheduler(optimizer, self.training_config)

        # ── 4. Wrap into Lightning scheduler config ───────────────────────────────
        # WarmupThenReduceOnPlateau needs monitor + reduce_on_plateau flag.
        if isinstance(scheduler, WarmupThenReduceOnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": self.monitor_metric,
                    "strict": True,
                    "reduce_on_plateau": True,
                },
            }

        # SequentialLR (WarmupCosine) — no monitor needed.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def loss_monitor(
        self,
        loss_types: tuple = ("loss",),
        step: str = "train",
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Monitors and logs losses during training and validation.

        Args:
            loss_types: Tuple of loss names to log.
            step: Either 'train' or 'val'.
            output: Model output containing the relevant losses.
        """
        if output is None:
            return

        for loss_type in loss_types:
            if loss_type not in output.keys():
                continue

            value = output.get(loss_type)

            if value is None:
                continue

            # Naming convention
            if step == "train":
                loss_name = "train_loss" if loss_type == "loss" else f"train_{loss_type}"
            elif step == "val":
                loss_name = "val_loss" if loss_type == "loss" else f"val_{loss_type}"
            else:
                loss_name = loss_type

            # Logging strategy
            if step == "train":
                self.log(
                    loss_name,
                    value,
                    on_step=True,      # save each training step
                    on_epoch=True,     # also save epoch-aggregated version
                    prog_bar=(loss_type in self.prog_bar_metrics),  # show main losses in progress bar
                    logger=True,
                    batch_size=self.training_config.per_device_train_batch_size,
                )
            elif step == "val":
                self.log(
                    loss_name,
                    value,
                    on_step=False,     # epoch-level val logging
                    on_epoch=True,
                    prog_bar=(loss_type == "loss"),
                    logger=True,
                    batch_size=self.training_config.per_device_eval_batch_size,
                )

    def log_learning_rate(self):
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        current_lr = optimizer.param_groups[0]["lr"]

        self.log(
            "lr",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=self.training_config.per_device_train_batch_size,
        )


# -----------------------------
# Trainer (execution layer)
# -----------------------------
class BaseTrainerL:
    """PyTorch Lightning trainer wrapper."""

    def __init__(
        self,
        model: "BaseAE",
        result_dir: str,
        train_dataset: Union[DataLoader, Any],
        eval_dataset: Optional[Union[DataLoader, Any]] = None,
        training_config: Optional["TrainingConfig"] = None,
        data_config: Optional["DataConfig"] = None,
        n_early_stopping_patience: int = 10,
        debug_model: bool = False,
        monitor_metric: Optional[str] = None,
        early_stopping_min_delta: float = 0.001,
        max_epochs_override: Optional[int] = None,
        module_mode_overrides: Optional[Dict[str, str]] = None,
    ):
        if training_config is None:
            training_config = TrainingConfig()
        if data_config is None:
            data_config = DataConfig()
        if training_config.output_dir is None:
            training_config.output_dir = "dummy_output_dir"

        self.training_config = training_config
        self.data_config = data_config
        self.model_name = model.model_name
        self.n_early_stopping_patience = n_early_stopping_patience
        self.debug_model = debug_model
        self.model_dir = result_dir
        self.monitor_metric_override = monitor_metric
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.max_epochs_override = max_epochs_override
        self.module_mode_overrides = dict(module_mode_overrides or {})
        self.debug_overfit_config = getattr(self.training_config, "debug_overfit", None)
        self.train_dataset_obj = train_dataset
        self.eval_dataset_obj = eval_dataset

        self.train_loader = self._build_loader(train_dataset, is_train=True)
        self.eval_loader = self._build_loader(eval_dataset, is_train=False) if eval_dataset is not None else None

        if self.eval_loader is None:
            logger.info("No eval dataset provided -> keep_best_on_train=True")
            self.training_config.keep_best_on_train = True

        self.monitor_metric = (
            self.monitor_metric_override
            if self.monitor_metric_override is not None
            else ("val_loss" if self.eval_loader is not None else "train_loss")
        )

        self.pl_model = PLTrainer(
            model=model,
            training_config=training_config,
            debug_model=debug_model,
            monitor_metric=self.monitor_metric,
            module_mode_overrides=self.module_mode_overrides,
        )

    def _build_loader(self, ds_or_loader, is_train: bool) -> DataLoader:
        if isinstance(ds_or_loader, DataLoader):
            logger.warning("Using provided DataLoader; config batch_size/num_workers may be ignored.")
            return ds_or_loader

        if is_train:
            return get_dataloader(
                ds_or_loader,
                batch_size=self.training_config.per_device_train_batch_size,
                num_workers=self.training_config.train_dataloader_num_workers,
                shuffle=True,
                collate_fn=collate_dataset_output,
            )
        return get_dataloader(
            ds_or_loader,
            batch_size=self.training_config.per_device_eval_batch_size,
            num_workers=self.training_config.eval_dataloader_num_workers,
            shuffle=False,
            collate_fn=collate_dataset_output,
        )

    def _copy_metrics_file(self, csv_logger: CSVLogger) -> None:
        try:
            src = os.path.join(csv_logger.log_dir, "metrics.csv")
            dst = os.path.join(self.model_dir, "metrics.csv")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                logger.info(f"Copied metrics.csv -> {dst}")
            else:
                logger.warning(f"metrics.csv not found: {src}")
        except Exception as e:
            logger.warning(f"Failed to copy metrics.csv: {e}")

    def _save_checkpoint_selection_metadata(
        self,
        *,
        best_model_path: Optional[str],
        last_model_path: str,
    ) -> None:
        metadata = {
            "best_model_path": Path(best_model_path).name if best_model_path else "",
            "last_model_path": Path(last_model_path).name,
        }
        metadata_path = Path(self.model_dir) / "checkpoint_paths.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _save_adaptive_aux_schedule_trace(self) -> None:
        trace_rows = self.pl_model.get_adaptive_aux_schedule_trace_rows()
        if not trace_rows:
            return
        trace_path = Path(self.model_dir) / "adaptive_aux_loss_schedule_trace.csv"
        fieldnames = list(trace_rows[0].keys())
        with trace_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_rows)

    def train(self) -> str:
        set_seed(self.training_config.seed)
        os.makedirs(self.model_dir, exist_ok=True)
        debug_cfg = self.debug_overfit_config
        effective_max_epochs = (
            int(self.max_epochs_override)
            if self.max_epochs_override is not None
            else self.training_config.num_epochs
        )
        effective_patience = self.n_early_stopping_patience
        disable_early_stopping = False
        if debug_cfg is not None and getattr(debug_cfg, "enabled", False):
            effective_max_epochs = int(getattr(debug_cfg, "num_epochs_override", effective_max_epochs))
            effective_patience = int(
                getattr(debug_cfg, "n_early_stopping_patience_override", effective_patience)
            )
            disable_early_stopping = bool(getattr(debug_cfg, "disable_early_stopping", False))

        ckpt = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="best_model_epoch={epoch}",
            monitor=self.monitor_metric,
            mode="min",
            save_top_k=1,
        )
        early = EarlyStopping(
            monitor=self.monitor_metric,
            patience=effective_patience,
            mode="min",
            min_delta=self.early_stopping_min_delta,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        csv_logger = CSVLogger(save_dir=self.model_dir, name="training_logs")
        callbacks = [ckpt, lr_monitor]
        if disable_early_stopping:
            logger.info("Debug overfit mode: EarlyStopping callback disabled.")
        else:
            callbacks.append(early)

        trainer = L.Trainer(
            max_epochs=effective_max_epochs,
            accelerator="auto",
            devices=self.training_config.devices,
            callbacks=callbacks,
            logger=csv_logger,
            precision="16-mixed" if self.training_config.amp else 32,
            gradient_clip_val=getattr(self.training_config, 'gradient_clip_val', 1.0),
        )

        trainer.fit(
            model=self.pl_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.eval_loader,
        )

        last_model_path = str(Path(self.model_dir) / "last_model.ckpt")
        trainer.save_checkpoint(last_model_path)
        self._save_checkpoint_selection_metadata(
            best_model_path=getattr(ckpt, "best_model_path", ""),
            last_model_path=last_model_path,
        )

        self.pl_model.model.save(
            self.model_dir,
            training_config=self.training_config,
            data_config=self.data_config,
        )

        self._copy_metrics_file(csv_logger)
        self._save_adaptive_aux_schedule_trace()
        self._save_debug_overfit_artifacts()

        logger.info(f"Training done. Model saved to: {self.model_dir}")
        return self.model_dir

    def _get_dataset_sample_ids(self, dataset_obj) -> list[str]:
        """Extract sample IDs from a dataset or Subset when available."""
        if dataset_obj is None:
            return []
        if hasattr(dataset_obj, "dataset") and hasattr(dataset_obj, "indices"):
            base_dataset = dataset_obj.dataset
            base_sample_ids = (
                base_dataset.get_sample_ids() if hasattr(base_dataset, "get_sample_ids") else []
            )
            return [str(base_sample_ids[idx]) for idx in dataset_obj.indices if idx < len(base_sample_ids)]
        if hasattr(dataset_obj, "get_sample_ids"):
            return [str(sample_id) for sample_id in dataset_obj.get_sample_ids()]
        return []

    def _save_debug_overfit_artifacts(self) -> None:
        """Save post-train predictions on the debug overfit subset for inspection."""
        debug_cfg = self.debug_overfit_config
        if debug_cfg is None or not getattr(debug_cfg, "enabled", False):
            return
        # The full workflow can also run normal post-training prediction on
        # generated debug test sets. Keep this lightweight trainer artifact only
        # when explicitly requested.
        if not getattr(debug_cfg, "save_post_train_predictions", True):
            return

        target_dataset = self.eval_dataset_obj if self.eval_dataset_obj is not None else self.train_dataset_obj
        if target_dataset is None:
            return

        export_loader = self._build_loader(target_dataset, is_train=False)
        sample_ids = self._get_dataset_sample_ids(target_dataset)
        pred_cell_prop_batches = []
        recon_x_conv_batches = []
        recon_x_all_types_batches = []
        label_batches = []

        self.pl_model.eval()
        device = self.pl_model.device
        with torch.no_grad():
            for batch in export_loader:
                moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                output = self.pl_model(moved)
                if getattr(output, "pred_cell_prop", None) is not None:
                    pred_cell_prop_batches.append(output.pred_cell_prop.detach().cpu())
                if getattr(output, "recon_x_conv", None) is not None:
                    recon_x_conv_batches.append(output.recon_x_conv.detach().cpu())
                if getattr(output, "recon_x_all_types", None) is not None:
                    recon_x_all_types_batches.append(output.recon_x_all_types.detach().cpu())
                labels = batch.get("labels")
                if torch.is_tensor(labels) and labels.numel() > 0:
                    label_batches.append(labels.detach().cpu())

        artifact = {
            "sample_ids": sample_ids,
            "pred_cell_prop": torch.cat(pred_cell_prop_batches, dim=0) if pred_cell_prop_batches else None,
            "recon_x_conv": torch.cat(recon_x_conv_batches, dim=0) if recon_x_conv_batches else None,
            "recon_x_all_types": (
                torch.cat(recon_x_all_types_batches, dim=0) if recon_x_all_types_batches else None
            ),
            "labels": torch.cat(label_batches, dim=0) if label_batches else None,
        }

        artifact_path = Path(self.model_dir) / "debug_overfit_predictions.pt"
        torch.save(artifact, artifact_path)
        logger.info("Saved debug overfit prediction artifact to: %s", artifact_path)

    def predict(self) -> Dict[str, torch.Tensor]:
        if self.eval_loader is None:
            raise ValueError("eval_loader is None. Cannot run predict().")

        self.pl_model.eval()
        batch = next(iter(self.eval_loader))
        device = self.pl_model.device

        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = self.pl_model(moved)
        return out

    def __call__(self) -> str:
        return self.train()


# -----------------------------
# Base Pipeline
# -----------------------------
class Pipeline:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


# -----------------------------
# Training Pipeline (orchestration)
# -----------------------------
class TrainingPipeline(Pipeline):
    def __init__(
        self,
        model: "BaseAE",
        trainer_cls: Type[BaseTrainerL] = BaseTrainerL,
        training_config: Optional["TrainingConfig"] = None,
        data_config: Optional["DataConfig"] = None,
        result_dir: Optional[str | Path] = None,
        debug_model: bool = False,
        data_adapter: Optional[DataAdapter] = None,
    ):
        if model is None:
            raise ValueError("model must not be None.")
        self.model = model

        self.training_config = training_config or TrainingConfig(name="VAETrainerConfig")
        self.data_config = data_config or DataConfig(name="DataConfig")

        if not isinstance(self.training_config, TrainingConfig):
            raise TypeError("training_config must be TrainingConfig")

        self.trainer_cls = trainer_cls
        self.result_dir = result_dir or self.training_config.output_dir or "dummy_output_dir"
        self.debug_model = debug_model
        self.data_adapter = data_adapter or DataAdapter()
        self.n_early_stopping_patience = self.training_config.n_early_stopping_patience
        self.trainer: Optional[BaseTrainerL] = None

    def __call__(
        self,
        train_data: Optional[Union[np.ndarray, torch.Tensor, Dataset, DataLoader, "anndata.AnnData"]] = None,
        eval_data: Optional[Union[np.ndarray, torch.Tensor, Dataset, DataLoader, "anndata.AnnData"]] = None,
    ) -> str:
        train_dataset = self.data_adapter.prepare_for_training(train_data, data_type="train")
        eval_dataset = self.data_adapter.prepare_for_training(eval_data, data_type="eval")

        if train_dataset is None:
            raise ValueError("train_data cannot be None.")

        logger.info(f"Using trainer: {self.trainer_cls.__name__}")
        self.trainer = self.trainer_cls(
            model=self.model,
            result_dir=self.result_dir,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            data_config=self.data_config,
            n_early_stopping_patience=self.n_early_stopping_patience,
            debug_model=self.debug_model,
        )
        return self.trainer.train()


def _check_dataset(dataset: BaseDataset):
    """Checks if the dataset is valid."""
    try:
        dataset_output = dataset[0]
    except Exception as e:
        raise DatasetError(
            "Error when trying to collect data from the dataset. Check `__getitem__` method. "
            "The Dataset should output a dictionary with at least the key 'data'. "
            "Please check documentation.\n"
            f"Exception raised: {type(e)} with message: {e}"
        ) from e

    if not isinstance(dataset_output, dict) or "data" not in dataset_output.keys():
        raise DatasetError(
            "The Dataset should output a dictionary with at least the key 'data'."
        )
    try:
        len(dataset)
    except Exception as e:
        raise DatasetError(
            "Error when trying to get dataset len. Check `__len__` method. "
            "Please check documentation.\n"
            f"Exception raised: {type(e)} with message: {e}"
        ) from e

    # check if the dataset works with the data loader
    # from torch.utils.data import DataLoader
    try:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=min(len(dataset), 2),
            collate_fn=collate_dataset_output,
        )
        loader_out = next(iter(dataloader))
        assert loader_out.data.shape[0] == min(
            len(dataset), 2
        ), "Error when combining dataset with loader."
    except Exception as e:
        raise DatasetError(
            "Error when combining dataset with DataLoader. \n"
            f"Exception raised: {type(e)} with message: {e}"
        ) from e
