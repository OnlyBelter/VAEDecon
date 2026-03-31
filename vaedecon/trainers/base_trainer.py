import os
from pathlib import Path

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
        data: Union[np.ndarray, torch.Tensor, "anndata.AnnData"],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        x = self._to_tensor(data, dtype=dtype)
        self._validate_tensor(x)
        return x

    def to_dataset(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        if labels is None:
            labels = torch.ones(data.shape[0], dtype=torch.float32)
        return BaseDataset(data, labels)

    def prepare_for_training(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor, Dataset, DataLoader, "anndata.AnnData", BaseDataset]],
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
        data: Union[np.ndarray, torch.Tensor, "anndata.AnnData"],
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

        # self.prog_bar_metrics = {"loss", "kld", "recon_loss_conv", "cell_prop_loss"}
        self.prog_bar_metrics = {"loss", "kld", "recon_loss_conv", "z_score_reciprocal"}

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Forward pass of the model."""
        return self.model(inputs, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        output = self(batch)
        self.log_learning_rate()

        lo = self.model.model_config.loss_coefficient
        self.log("w_gene_mean", float(lo.gene_mean_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("w_gene_std", float(lo.gene_std_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.loss_monitor(
            step="train",
            output=output,
            loss_types=(
                "loss",
                "kld",
                "kld_p",
                "recon_loss_conv",
                "gene_mean_loss",
                "gene_std_loss",
                "repulsion_loss",
                "cell_prop_loss",
                "z_score_reciprocal",
            ),
        )

        return output.loss

    def on_train_batch_start(self, batch: Dict[str, Any], batch_idx: int) -> None:
        cfg = self.training_config
        schedule = getattr(cfg, "gene_stat_weight_schedule", None)
        if schedule != "linear":
            return

        steps = getattr(cfg, "gene_stat_weight_schedule_steps", None)
        if steps is None or steps <= 0:
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

        self.log("w_gene_mean", float(lo.gene_mean_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("w_gene_std", float(lo.gene_std_weight), on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single validation step."""
        output = self(batch)

        self.loss_monitor(
            step="val",
            output=output,
            loss_types=(
                "loss",
                "kld",
                "kld_p",
                "recon_loss_conv",
                "cell_prop_loss",
                "gene_mean_loss",
                "gene_std_loss",
                "repulsion_loss",
            ),
        )

        return output.loss

    def on_train_epoch_end(self):
        """Optional debug info after each epoch."""
        if self.debug_model:
            opt = self.optimizers()
            current_lr = None
            if opt is not None:
                current_lr = opt.param_groups[0]["lr"]

            monitored_value = self.trainer.callback_metrics.get(self.monitor_metric, None)
            if monitored_value is not None and torch.is_tensor(monitored_value):
                monitored_value = monitored_value.item()

            if current_lr is not None:
                print(
                    f"Epoch {self.current_epoch}: "
                    f"lr={current_lr:.2e}, "
                    f"{self.monitor_metric}={monitored_value}"
                )

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
        if self.training_config.optimizer_params is not None:
            optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                **self.training_config.optimizer_params,
            )
        else:
            optimizer = optimizer_cls(
                self.model.parameters(),
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

        self.train_loader = self._build_loader(train_dataset, is_train=True)
        self.eval_loader = self._build_loader(eval_dataset, is_train=False) if eval_dataset is not None else None

        if self.eval_loader is None:
            logger.info("No eval dataset provided -> keep_best_on_train=True")
            self.training_config.keep_best_on_train = True

        self.monitor_metric = "val_loss" if self.eval_loader is not None else "train_loss"

        self.pl_model = PLTrainer(
            model=model,
            training_config=training_config,
            debug_model=debug_model,
            monitor_metric=self.monitor_metric,
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

    def train(self) -> str:
        set_seed(self.training_config.seed)
        os.makedirs(self.model_dir, exist_ok=True)

        ckpt = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="best_model_epoch={epoch}",
            monitor=self.monitor_metric,
            mode="min",
            save_top_k=1,
        )
        early = EarlyStopping(
            monitor=self.monitor_metric,
            patience=self.n_early_stopping_patience,
            mode="min",
            min_delta=0.001,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        csv_logger = CSVLogger(save_dir=self.model_dir, name="training_logs")

        trainer = L.Trainer(
            max_epochs=self.training_config.num_epochs,
            accelerator="auto",
            devices=self.training_config.devices,
            callbacks=[ckpt, lr_monitor, early],
            logger=csv_logger,
            precision="16-mixed" if self.training_config.amp else 32,
        )

        trainer.fit(
            model=self.pl_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.eval_loader,
        )

        self.pl_model.model.save(
            self.model_dir,
            training_config=self.training_config,
            data_config=self.data_config,
        )

        self._copy_metrics_file(csv_logger)

        logger.info(f"Training done. Model saved to: {self.model_dir}")
        return self.model_dir

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
