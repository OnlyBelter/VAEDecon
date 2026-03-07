import os
import logging
import platform
import shutil

import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from ...data.datasets import BaseDataset, collate_dataset_output
from ...models import BaseAE
from ...models.base import ModelOutput, set_seed
from .base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


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


class PLTrainer(L.LightningModule):
    """PyTorch Lightning-based trainer for BaseAE models."""

    def __init__(
        self,
        model: BaseAE,
        training_config: BaseTrainerConfig,
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
        self.prog_bar_metrics = {"loss", "kld", "recon_loss_conv"}

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Forward pass of the model."""
        return self.model(inputs, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        output = self(batch)
        self.log_learning_rate()

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
            ),
        )

        return output.loss

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
        """Configures optimizer and optional learning rate scheduler."""
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

        if self.training_config.scheduler_cls is None:
            return {"optimizer": optimizer}

        scheduler_cls = getattr(lr_scheduler, self.training_config.scheduler_cls)

        if self.training_config.scheduler_params is not None:
            scheduler = scheduler_cls(
                optimizer, **self.training_config.scheduler_params
            )
        else:
            scheduler = scheduler_cls(optimizer)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        # For ReduceLROnPlateau, a monitored metric is required
        if self.training_config.scheduler_cls == "ReduceLROnPlateau":
            scheduler_config["monitor"] = self.monitor_metric
            scheduler_config["strict"] = True

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
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


class BaseTrainerL:
    """Trainer class that uses PyTorch Lightning."""

    def __init__(
        self,
        model: BaseAE,
        result_dir: str,
        train_dataset: Union[DataLoader, Any],
        eval_dataset: Optional[Union[DataLoader, Any]] = None,
        training_config: Optional[Any] = None,
        n_early_stopping_patience: int = 10,
        debug_model: Optional[bool] = False,
    ):
        """Initializes the Trainer.

        Args:
            model: The BaseAE model to train.
            result_dir: Directory to save checkpoints and logs.
            train_dataset: Training dataset or dataloader.
            eval_dataset: Evaluation dataset or dataloader.
            training_config: Training configuration.
            n_early_stopping_patience: Early stopping patience.
            debug_model: Whether to enable debug mode.
        """
        if training_config is None:
            training_config = BaseTrainerConfig()

        if training_config.output_dir is None:
            training_config.output_dir = "dummy_output_dir"

        self.training_config = training_config
        self.model_name = model.model_name
        self.n_early_stopping_patience = n_early_stopping_patience
        self.debug_model = debug_model

        if isinstance(train_dataset, DataLoader):
            train_loader = train_dataset
            logger.warning(
                "Using the provided train dataloader! Careful: this may overwrite "
                "some parameters provided in your training config."
            )
        else:
            train_loader = get_dataloader(
                train_dataset,
                batch_size=training_config.per_device_train_batch_size,
                num_workers=training_config.train_dataloader_num_workers,
                shuffle=True,
                collate_fn=collate_dataset_output,
            )

        if eval_dataset is not None:
            if isinstance(eval_dataset, DataLoader):
                eval_loader = eval_dataset
                logger.warning(
                    "Using the provided eval dataloader! Careful: this may overwrite "
                    "some parameters provided in your training config."
                )
            else:
                eval_loader = get_dataloader(
                    eval_dataset,
                    batch_size=training_config.per_device_eval_batch_size,
                    num_workers=training_config.eval_dataloader_num_workers,
                    shuffle=False,
                    collate_fn=collate_dataset_output,
                )
        else:
            logger.info("! No eval dataset provided ! -> keeping best model on train.\n")
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model_dir = result_dir

        # Dynamic monitor metric
        self.monitor_metric = "val_loss" if self.eval_loader is not None else "train_loss"

        self.pl_model = PLTrainer(
            model=model,
            training_config=training_config,
            debug_model=debug_model,
            monitor_metric=self.monitor_metric,
        )

    def _copy_metrics_file(self, csv_logger: CSVLogger) -> None:
        """Copy Lightning metrics.csv to self.model_dir."""
        try:
            src_metrics = os.path.join(csv_logger.log_dir, "metrics.csv")
            dst_metrics = os.path.join(self.model_dir, "metrics.csv")

            if os.path.exists(src_metrics):
                shutil.copy2(src_metrics, dst_metrics)
                logger.info(f"Copied metrics file to: {dst_metrics}")
            else:
                logger.warning(f"metrics.csv not found at: {src_metrics}")
        except Exception as e:
            logger.warning(f"Failed to copy metrics.csv to model_dir: {e}")

    def train(self) -> None:
        """Trains the model using PyTorch Lightning."""
        set_seed(self.training_config.seed)
        os.makedirs(self.model_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="best_model_epoch={epoch}",
            monitor=self.monitor_metric,
            mode="min",
            save_top_k=1,
        )

        early_stop_callback = EarlyStopping(
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
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            logger=csv_logger,
            precision="16-mixed" if self.training_config.amp else 32,
        )

        trainer.fit(
            model=self.pl_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.eval_loader,
        )

        # Save final model
        self.pl_model.model.save(self.model_dir, training_config=self.training_config)

        # Copy metrics.csv to self.model_dir
        self._copy_metrics_file(csv_logger)

        logger.info("Training ended!")
        logger.info(f"Saved final model in {self.model_dir}")
        logger.info(f"Lightning CSV logs saved in: {csv_logger.log_dir}")
        logger.info(f"Monitor metric used: {self.monitor_metric}")

    def predict(self) -> Dict[str, torch.Tensor]:
        """Generates predictions from the trained model."""
        if self.eval_loader is None:
            raise ValueError("eval_loader is None. Cannot run predict().")

        inputs = next(iter(self.eval_loader))
        self.pl_model.eval()

        # Move batch to device if needed
        device = self.pl_model.device
        moved_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                moved_inputs[k] = v.to(device)
            else:
                moved_inputs[k] = v

        with torch.no_grad():
            outputs = self.pl_model(moved_inputs)

        return outputs

    def __call__(self, *args, **kwargs):
        pass


def plot_loss(losses_df, result_dir, train_loss_col_name, val_loss_col_name):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(losses_df[train_loss_col_name], label='train_loss')
    ax.plot(losses_df[val_loss_col_name], label='eval_loss')
    ax.legend(loc='best')
    plt.savefig(os.path.join(result_dir, 'losses.png'), dpi=200)
    plt.close()
