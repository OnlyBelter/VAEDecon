"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
from collections import OrderedDict
from typing import Any, Tuple

import torch
import numpy as np
import anndata
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class DatasetOutput(OrderedDict):
    """Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


def collate_dataset_output(batch):
    """Collate function that treats the `DatasetOutput` class correctly."""
    if isinstance(batch[0], DatasetOutput):
        # `default_collate` returns a dict for older versions of PyTorch.
        return DatasetOutput(**default_collate(batch))
    else:
        return default_collate(batch)


class BaseDataset(Dataset):
    """This class is the Base class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """

    def __init__(self, data, labels):
        self.labels = labels.type(torch.float)
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = self.data[index]
        y = self.labels[index]

        return DatasetOutput(data=X, labels=y)


class GEPDataset(Dataset):
    """This class is the Base class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """

    def __init__(self, file_path, scaling_by_constant=True):
        """
        Args:
            file_path (str): The path to the file containing the data
            scaling_by_constant (bool): If True, the data is scaled by a constant factor,
            so that the data is in the range [0, 1].
        """
        self.file_path = file_path
        self.gep_data = anndata.read_h5ad(file_path, backed='r')  # read the data in log2(TPM + 1) format
        cell_prop = self.gep_data.obs.values.copy()
        # cell_types = gep_data.obs.columns.to_list()
        self.labels = torch.tensor(cell_prop, dtype=torch.float32)
        self.scaling_by_constant = scaling_by_constant

        # self.labels = labels.type(torch.float)
        # self.data = data.type(torch.float)

    def __len__(self):
        return self.gep_data.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = self.gep_data.X[index]
        X = torch.tensor(X, dtype=torch.float32)
        y = self.labels[index]
        if self.scaling_by_constant:
            X = X / 20.0

        return DatasetOutput(data=X, labels=y)
