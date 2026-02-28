import os
import logging
import platform

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
                num_workers = min(num_workers, 10)  # Cap at 8 to avoid excessive resource usage

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
    ):
        """Initializes the PLTrainer.

        Args:
            model: The BaseAE model to train.
            training_config: The training configuration.
        """
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.model_name = model.model_name
        self.debug_model = debug_model
        # self.save_hyperparameters(training_config)  # TODO

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Forward pass of the model."""
        return self.model(inputs, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        output = self(batch)
        # Log the learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=True)

        self.loss_monitor(step='train', output=output,
                          loss_types=('loss', 'kld', 'kld_p', 'recon_loss_conv',
                                      'gene_mean_loss', 'gene_std_loss', 'repulsion_loss', 'cell_prop_loss'
                                      ))
        return output.loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single validation step."""
        output = self(batch)
        self.loss_monitor(step='val', output=output,
                          loss_types=('loss', 'kld', 'kld_p', 'recon_loss_conv', 'cell_prop_loss'))
        return output.loss

    def on_train_epoch_end(self):  # Add here
        """Debug: Track LR changes after each training epoch."""
        sch = self.lr_schedulers()
        if sch is not None:
            current_lr = self.optimizers().param_groups[0]['lr']
            val_loss = self.trainer.callback_metrics.get('val_loss', None)
            print(f"Epoch {self.current_epoch}: LR={current_lr:.2e}, val_loss={val_loss}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the optimizer and learning rate scheduler."""
        optimizer_cls = getattr(optim, self.training_config.optimizer_cls)

        if self.training_config.optimizer_params is not None:
            optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                **self.training_config.optimizer_params,
            )
        else:
            optimizer = optimizer_cls(
                self.model.parameters(), lr=self.training_config.learning_rate
            )

        if self.training_config.scheduler_cls is not None:
            scheduler_cls = getattr(lr_scheduler, self.training_config.scheduler_cls)

            if self.training_config.scheduler_params is not None:
                scheduler = scheduler_cls(
                    optimizer, **self.training_config.scheduler_params
                )
            else:
                scheduler = scheduler_cls(optimizer)

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",  # 'epoch' or 'step'
                "frequency": 1,
            }

            # If the scheduler is ReduceLROnPlateau, we need to specify the metric to monitor
            if self.training_config.scheduler_cls == "ReduceLROnPlateau":
                scheduler_config["monitor"] = "val_loss"
                scheduler_config["strict"] = True  # If val_loss does not exist, it will report an error

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
            }
        else:
            return {"optimizer": optimizer}

    def loss_monitor(self, loss_types: tuple=('loss',), step: str='train', output: ModelOutput=None) -> None:
        """
        Monitors the loss during training and validation step.
        Args:
            loss_types (tuple): List of loss types to monitor.
            step (str): The step during which to monitor the loss ('train' or 'val').
            output (ModelOutput): The model output containing the loss values.
        """
        for loss_type in loss_types:
            if loss_type in output.keys():
                loss_name = loss_type
                if step == 'train' and loss_type == 'loss':
                    loss_name = 'train_loss'
                elif step == 'val' and loss_type == 'loss':
                    loss_name = 'val_loss'
                if self.debug_model:
                    self.log(loss_name, output.get(loss_type), on_step=True if step=='train' else False,
                             on_epoch=True if step=='val' else False,
                             prog_bar=True, logger=True)
                else:
                    self.log(loss_name, output.get(loss_type), on_step=False, on_epoch=True,
                             prog_bar=True, logger=True)


class BaseTrainerL:
    """Trainer class that uses PyTorch Lightning."""

    def __init__(
        self,
        model: BaseAE,
        result_dir: str,
        train_dataset: Union[BaseDataset, DataLoader],
        eval_dataset: Optional[Union[BaseDataset, DataLoader]] = None,
        training_config: Optional[BaseTrainerConfig] = None,
        n_early_stopping_patience: int = 10,
        debug_model: Optional[bool] = False,
    ):
        """Initializes the Trainer.

        Args:
            model: The BaseAE model to train.
            result_dir: The directory to save the model checkpoints and logs.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            training_config: The training configuration.
            n_early_stopping_patience: The number of early stopping patience.
        """
        if training_config is None:
            training_config = BaseTrainerConfig()

        if training_config.output_dir is None:
            output_dir = "dummy_output_dir"
            training_config.output_dir = output_dir

        self.training_config = training_config
        self.model_name = model.model_name
        self.n_early_stopping_patience = n_early_stopping_patience
        # self.rank = self.training_config.rank

        if isinstance(train_dataset, DataLoader):
            train_loader = train_dataset
            logger.warning(
                "Using the provided train dataloader! Carefull this may overwrite some "
                "parameters provided in your training config."
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
                    "Using the provided eval dataloader! Carefull this may overwrite some "
                    "parameters provided in your training config."
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
            logger.info(
                "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.pl_model = PLTrainer(model, training_config, debug_model=debug_model)

        self.model_dir = result_dir

    def train(self) -> None:
        """Trains the model using PyTorch Lightning."""
        set_seed(self.training_config.seed)
        # final_dir = self.model_dir
        # model_par_dir = os.path.dirname(self.model_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="best_model_epoch={epoch}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        csv_logger = CSVLogger(save_dir=self.model_dir, name="training_logs")
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.n_early_stopping_patience,
            mode="min",
            min_delta=0.001
        )

        trainer = L.Trainer(
            max_epochs=self.training_config.num_epochs,
            accelerator="auto",
            devices=self.training_config.devices,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            logger=csv_logger,
            precision=16 if self.training_config.amp else 32,
        )

        trainer.fit(
            model=self.pl_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.eval_loader,
        )
        self.pl_model.model.save(self.model_dir, training_config=self.training_config)

        logger.info("Training ended!")
        logger.info(f"Saved final model in {self.model_dir}")

    def predict(self) -> Dict[str, torch.Tensor]:
        """Generates predictions from the trained model."""
        inputs = next(iter(self.eval_loader))
        return self.pl_model.predict(inputs)

    # @property
    # def is_main_process(self):
    #     if self.rank == 0 or self.rank == -1:
    #         return True
    #     else:
    #         return False

    def __call__(self, *args, **kwargs):
        pass


def plot_loss(losses_df, result_dir, train_loss_col_name, val_loss_col_name):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(losses_df[train_loss_col_name], label='train_loss')
    ax.plot(losses_df[val_loss_col_name], label='eval_loss')
    ax.legend(loc='best')
    plt.savefig(os.path.join(result_dir, 'losses.png'), dpi=200)
    plt.close()
