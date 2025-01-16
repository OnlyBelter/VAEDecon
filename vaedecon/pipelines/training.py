import os
import logging
from typing import Optional, Union, Any, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..customexception import DatasetError
from ..data.datasets import collate_dataset_output, BaseDataset
from ..data.preprocessors import DataProcessor
from ..models import BaseAE
from ..trainers import BaseTrainerConfig, BaseTrainerL
from .base_pipeline import Pipeline
from ..utility import log_message

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


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


class TrainingPipeline(Pipeline):
    """
    This Pipeline provides an end to end way to train the VAE model.

    The trained model will be saved in ``output_dir`` stated in the
    :class:`trainers.BaseTrainerConfig`. A folder
    ``training_YYYY-MM-DD_hh-mm-ss`` is
    created where checkpoints and final model will be saved. Checkpoints are saved in
    ``checkpoint_epoch_{epoch}`` folder (optimizer and training config
    saved as well to resume training if needed)
    and the final model is saved in a ``final_model`` folder. If ``output_dir`` is
    None, data is saved in ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss`` is created.

    Parameters:
        model (BaseAE): An instance of :class:`models.BaseAE` you want to train.
            If None, a default :class:`models.VAE` model is used. Default: None.
        training_config (BaseTrainerConfig): An instance of
            :class:`trainers.BaseTrainerConfig` stating the training
            parameters. If None, a default configuration is used.
        trainer_cls: The trainer class to use.
            Defaults to BaseTrainerL.
        result_dir (str): The directory where the model will be saved.
    """

    def __init__(self,
                 model: BaseAE = None,
                 trainer_cls: Type[BaseTrainerL] = None,
                 training_config=None,
                 result_dir: str = None):
        super().__init__()
        if training_config is None:
            training_config = BaseTrainerConfig(name='VAETrainerConfig')

        if not isinstance(training_config, BaseTrainerConfig):
            raise AssertionError(
                "A 'BaseTrainerConfig' " "is expected for the pipeline"
            )

        self.data_processor = DataProcessor()
        self.model = model
        self.training_config = training_config
        self.trainer_cls = trainer_cls
        self.n_early_stopping_patience = training_config.n_early_stopping_patience
        self.result_dir = result_dir  # model directory
        # self.final_output_dir = os.path.join(self.result_dir, "final_model")

    def _prepare_data(
            self,
            data: Optional[
                Union[np.ndarray, torch.Tensor, Dataset, DataLoader, BaseDataset]
            ] = None,
            data_type: str = "train",
    ) -> Optional[Union[Dataset, DataLoader]]:
        """Prepares the data for training or evaluation."""
        if data is None:
            return None

        if isinstance(data, DataLoader):
            logger.info(f"Using provided {data_type} dataloader.")
            return data
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            logger.info(f"Preprocessing {data_type} data...")
            processed_data = self.data_processor.process_data(data)
            dataset = self.data_processor.to_dataset(processed_data)
            logger.info(f"Checking {data_type} dataset...")
            _check_dataset(dataset)
            return dataset
        elif isinstance(data, Dataset):
            logger.info(f"Checking {data_type} dataset...")
            _check_dataset(data)
            return data
        else:
            raise ValueError(
                f"Unsupported data type for {data_type} data: {type(data)}"
            )

    def __call__(
        self,
        train_data: Union[
            np.ndarray, torch.Tensor, Dataset, DataLoader,
        ] = None,
        eval_data: Union[
            np.ndarray, torch.Tensor, Dataset, DataLoader,
        ] = None,
    ) -> Any:
        """
        Launch the model training on the provided data.

        Args:
            train_data: The training data or DataLoader.
            eval_data: The evaluation data or DataLoader. If None, only uses train_data for training.

        Returns:
            str: The path to the final model directory.
        """

        # Initialize variables for datasets and dataloaders

        train_dataset = self._prepare_data(train_data, "train")
        eval_dataset = self._prepare_data(eval_data, "eval")
        logger.info(f"Using {self.trainer_cls.__name__} for training.")
        trainer = self.trainer_cls(
            model=self.model,
            result_dir=self.result_dir,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            n_early_stopping_patience=self.n_early_stopping_patience,
        )

        self.trainer = trainer
        # output_dir = trainer.set_output_dir()
        # self.final_output_dir =
        trainer.train()
        # self.final_output_dir = trainer.train()
