"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
import os
from collections import OrderedDict
from typing import Any, Tuple
from pathlib import Path

import logging
import torch
import numpy as np
import anndata
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from ..utility.read_file import ReadExp, ReadH5AD
from ..utility import non_log2log_cpm, check_dir

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class DatasetOutput(OrderedDict):
    """Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
    the ``ModelOutput`` class from huggingface transformers library"""

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


def load_gene_list(file_path: Path):
    with open(file_path, 'r') as f:
        gene_list = f.readlines()
    gene_list = [gene.strip() for gene in gene_list]
    return gene_list


class GEPDataset(Dataset):
    """This class is the Base class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """

    def __init__(self, file_path: list[str], scaling_by_constant=True, gene_list_file: Path = None):
        """
        Args:
            file_path (str): a list of file path containing the data

            scaling_by_constant (bool): If True, the data is scaled by a constant factor,
              so that the data is in the range [0, 1].
        """
        # self.file_path = file_path
        all_data = []
        all_cell_prop = []
        for path in file_path:
            logger.info(f"Reading data from {path}")
            h5ad_obj = ReadH5AD(path)  # read the data in log2(TPM + 1) format
            gep_data = h5ad_obj.get_df(convert_to_tpm=True)  # get the data in pandas DataFrame format in TPM
            logger.info(f"Data shape: {gep_data.shape}")
            cell_prop = h5ad_obj.get_cell_fraction()
            all_data.append(gep_data)
            all_cell_prop.append(cell_prop)
        # merge multiple datasets and rescale each dataset based on the intersection of genes
        self.gep_data = pd.concat(all_data, axis=0, join='inner')  # merge multiple datasets by rows
        if gene_list_file is not None:  # filter the data based on the gene list, for test sets
            gene_list = load_gene_list(gene_list_file)
            self.gep_data = self.gep_data.loc[:, gene_list]
        # rescaling the data to log2(CPM + 1) format after merging
        self.gep_data = non_log2log_cpm(self.gep_data, transpose=False)
        logger.info(f"Data shape after merging: {self.gep_data.shape}")
        # self.gep_data = anndata.concat(all_data)
        # self.gep_data = anndata.read_h5ad(file_path, backed='r')  # read the data in log2(TPM + 1) format
        self.data = self.gep_data.values.astype(np.float32)  # get the data in numpy format
        self.data = torch.from_numpy(self.data)
        if scaling_by_constant:
            self.data = self.data / 20.0
        self.cell_prop = pd.concat(all_cell_prop, axis=0)
        self.cell_types = self.cell_prop.columns.to_list()
        self.gene_list = self.gep_data.columns.to_list()
        self.labels = torch.tensor(self.cell_prop.values, dtype=torch.float32)

    def __len__(self):
        return self.gep_data.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        x = self.data[index]
        y = self.labels[index]

        return DatasetOutput(data=x, labels=y)

    def save_gene_list(self, file_path: Path):
        check_dir(Path(file_path).parent)
        with open(file_path, 'w') as f:
            for gene in self.gene_list:
                f.write(f"{gene}\n")
        logger.info(f"Gene list is saved to {file_path}")

    def save_cell_types(self, file_path: Path):
        check_dir(Path(file_path).parent)
        with open(file_path, 'w') as f:
            for cell_type in self.cell_types:
                f.write(f"{cell_type}\n")
        logger.info(f"Cell types are saved to {file_path}")

    def get_gene_list(self):
        return self.gene_list

    def get_cell_types(self):
        return self.cell_types

    def get_cell_prop(self) -> pd.DataFrame:
        return self.cell_prop
