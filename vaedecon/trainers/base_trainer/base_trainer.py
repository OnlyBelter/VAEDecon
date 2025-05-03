import datetime
import logging
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
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
from ...models.base import ModelOutput
from pythae.trainers.trainer_utils import set_seed
from .base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class PLTrainer(L.LightningModule):
    """PyTorch Lightning-based trainer for BaseAE models."""

    def __init__(
        self,
        model: BaseAE,
        training_config: BaseTrainerConfig,
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
        # self.save_hyperparameters(training_config)  # TODO

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Forward pass of the model."""
        return self.model(inputs, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        output = self(batch)
        self.loss_monitor(step='train', output=output,
                          loss_types=('loss', 'kld', 'recon_loss_conv', 'gene_mean_loss', 'gene_std_loss'))
        return output.loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single validation step."""
        output = self(batch)
        self.loss_monitor(step='val', output=output,
                          loss_types=('loss', 'kld', 'recon_loss_conv', 'gene_mean_loss', 'gene_std_loss'))
        return output.loss

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
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generates predictions from the model."""
        self.model.eval()
        with torch.no_grad():
            model_out = self(inputs)
            reconstructions = model_out.recon_x.cpu().detach()[
                : min(inputs["data"].shape[0], 10)
            ]
            z_enc = model_out.z[: min(inputs["data"].shape[0], 10)]
            z = torch.randn_like(z_enc)
            normal_generation = self.model.decoder(z).reconstruction.detach().cpu()
        return {
            "true_data": inputs["data"][: min(inputs["data"].shape[0], 10)],
            "reconstructions": reconstructions,
            "generations": normal_generation,
        }

    def loss_monitor(self, loss_types: tuple=('loss',), step: str='train', output: ModelOutput=None) -> None:
        """
        Monitors the loss during training and validation step.
        Args:
            loss_types (tuple): List of loss types to monitor.
            step (str): The step during which to monitor the loss ('train' or 'val').
            output (ModelOutput): The model output containing the loss values.
        """
        for loss_type in loss_types:
            loss_name = loss_type
            if step == 'train' and loss_type == 'loss':
                loss_name = 'train_loss'
            elif step == 'val' and loss_type == 'loss':
                loss_name = 'val_loss'
            self.log(loss_name, output.get(loss_type), on_step=True if step=='train' else False,
                     on_epoch=True, prog_bar=True, logger=True)


def get_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Creates a DataLoader for the given dataset."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_dataset_output,
    )


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
                )
        else:
            logger.info(
                "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.pl_model = PLTrainer(model, training_config)

        self.model_dir = result_dir

    def train(self) -> None:
        """Trains the model using PyTorch Lightning."""
        set_seed(self.training_config.seed)
        # final_dir = self.model_dir
        # model_par_dir = os.path.dirname(self.model_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="checkpoint_{epoch}",
            every_n_epochs=self.training_config.steps_saving
            if self.training_config.steps_saving
            else 1,
            save_top_k=0,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        csv_logger = CSVLogger(save_dir=self.model_dir, name="training_logs")
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=self.n_early_stopping_patience,
                                            mode="min", min_delta=0.001)

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
