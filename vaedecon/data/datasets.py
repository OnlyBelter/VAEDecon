"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
from collections import OrderedDict
from typing import Any, List, Union, Optional, Dict
from pathlib import Path

import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from ..utility.read_file import ReadExp, ReadH5AD
from ..utility import non_log2log_cpm, check_dir, log_message

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

    def to_tuple(self) -> tuple[Any, ...]:
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


def load_gene_list(file_path: Path) -> list[str]:
    with open(file_path, 'r') as f:
        gene_list = f.read().splitlines()
    # gene_list = [gene.strip() for gene in gene_list]
    return gene_list


# class GEPDataset(Dataset):
#     """
#     Dataset class for GEP data. Implements preprocessing and caching
#     to speed up data loading for repeated runs.
#
#     A ``__getitem__`` is redefined and outputs a python dictionary
#     with the keys corresponding to `data` and `labels`.
#     This Class should be used for any new data sets.
#     """
#
#     def __init__(self, file_path: List[str],
#                  processed_data_dir: Union[str, Path],
#                  scaling_by_constant: bool = True,
#                  gene_list_file: Optional[Union[str, Path]] = None,
#                  remove_low_var_genes: bool = False,
#                  min_var: float = 1.0,
#                  cell_cell2ave_exp_file_path: Optional[Union[str, Path]] = None):
#         """
#         Args:
#             file_path (str): a list of file path containing the data
#
#             scaling_by_constant (bool): If True, the data is scaled by a constant factor (20 by default),
#               so that the data is in the range [0, 1].
#
#             gene_list_file (str): a file path containing the gene list to filter the data
#
#             remove_low_var_genes (bool): If True, the low variance genes are removed from the dataset.
#
#             min_var (float): The minimum variance of the gene to be kept.
#
#             cell_cell2ave_exp_file_path (str): The file path to save the average expression of each cell type.
#                 - a table: genes x cell types, in TPM format
#         """
#         # self.file_path = file_path
#         all_data = []
#         all_cell_prop = []
#         for path in file_path:
#             log_message(f"Reading data from {path}")
#             h5ad_obj = ReadH5AD(path)  # read the data in log2(TPM + 1) format
#             gep_data = h5ad_obj.get_df(convert_to_tpm=True)  # get the data in pandas DataFrame format in TPM
#             log_message(f"Data shape: {gep_data.shape}")
#             cell_prop = h5ad_obj.get_cell_fraction()
#             all_data.append(gep_data)
#             all_cell_prop.append(cell_prop)
#         # merge multiple datasets and rescale each dataset based on the intersection of genes
#         self.gep_data = pd.concat(all_data, axis=0, join='inner')  # merge multiple datasets by rows (n_samples, n_genes)
#         self.cell_prop = pd.concat(all_cell_prop, axis=0, join='inner')  # merge multiple datasets by rows (n_samples, n_cell_types)
#         assert len(self.gep_data) == len(self.cell_prop)
#         assert np.all(self.gep_data.index == self.cell_prop.index)  # check the order of samples
#         if gene_list_file is not None:  # filter the data based on the gene list, for test sets
#             gene_list = load_gene_list(gene_list_file)
#             self.gep_data = self.gep_data.loc[:, self.gep_data.columns.isin(gene_list)]
#
#         # remove low variance genes here
#         if remove_low_var_genes:
#             self.remove_low_var_genes(min_var=min_var, cell_cell2ave_exp_file_path=cell_cell2ave_exp_file_path)
#
#         # rescaling the data to log2(CPM + 1) format after merging
#         self.gep_data = non_log2log_cpm(self.gep_data, transpose=False)
#         log_message(f"Data shape after merging: {self.gep_data.shape}")
#         # self.gep_data = anndata.concat(all_data)
#         # self.gep_data = anndata.read_h5ad(file_path, backed='r')  # read the data in log2(TPM + 1) format
#         self.data = self.gep_data.values.astype(np.float32)  # get the data in numpy format
#         self.data = torch.from_numpy(self.data)
#         if scaling_by_constant:
#             self.data = self.data / 20.0
#
#         self.cell_types = self.cell_prop.columns.to_list()
#         self.gene_list = self.gep_data.columns.to_list()
#         self.labels = torch.tensor(self.cell_prop.values, dtype=torch.float32)
#
#     def __len__(self):
#         return self.gep_data.shape[0]
#
#     def __getitem__(self, index):
#         """Generates one sample of data
#
#         Args:
#             index (int): The index of the data in the Dataset
#
#         Returns:
#             (dict): A dictionary with the keys 'data' and 'labels' and corresponding
#             torch.Tensor
#         """
#         # Select sample
#         x = self.data[index]
#         y = self.labels[index]
#         # sample_id = self.gep_data.index.to_list()[index]
#         # y = self.labels[index]
#
#         return DatasetOutput(data=x, labels=y)
#
#     def save_gene_list(self, file_path: Path):
#         check_dir(Path(file_path).parent)
#         with open(file_path, 'w') as f:
#             for gene in self.gene_list:
#                 f.write(f"{gene}\n")
#         logger.info(f"Gene list is saved to {file_path}")
#
#     def save_cell_types(self, file_path: Path):
#         check_dir(Path(file_path).parent)
#         with open(file_path, 'w') as f:
#             for cell_type in self.cell_types:
#                 f.write(f"{cell_type}\n")
#         logger.info(f"Cell types are saved to {file_path}")
#
#     def get_gene_list(self):
#         return self.gene_list
#
#     def get_cell_types(self):
#         return self.cell_types
#
#     def get_cell_prop(self) -> pd.DataFrame:
#         return self.cell_prop
#
#     def get_sample_ids(self):
#         return self.gep_data.index.to_list()
#
#     def remove_low_var_genes(self, min_var: float = 1, cell_cell2ave_exp_file_path: str = None):
#         """Remove low variance genes from the dataset.
#
#         Args:
#             min_var (float): The minimum variance of the gene to be kept.
#             cell_cell2ave_exp_file_path (str): The file path to save the average expression of each cell type.
#             - a table: genes x cell types
#         """
#         n_gene_before_filter = self.gep_data.shape[1]
#         var = self.gep_data.var(axis=0)
#         self.gep_data = self.gep_data.loc[:, var > min_var]
#         if cell_cell2ave_exp_file_path is not None:
#             cell_cell2ave_exp = pd.read_csv(cell_cell2ave_exp_file_path, index_col=0)
#             if 'var' not in cell_cell2ave_exp:
#                 cell_cell2ave_exp['var'] = cell_cell2ave_exp.var(axis=0)
#             cell_cell2ave_exp = cell_cell2ave_exp.loc[cell_cell2ave_exp['var'] > min_var, :]
#             self.gep_data = self.gep_data.loc[:, self.gep_data.columns.isin(cell_cell2ave_exp.index)]
#         n_gene_after_filter = self.gep_data.shape[1]
#         log_message(f"Number of genes before filter: {n_gene_before_filter}")
#         log_message(f"Number of genes after filter: {n_gene_after_filter}")
#         log_message(f"Number of genes removed: {n_gene_before_filter - n_gene_after_filter}")


class GEPDataset(Dataset):
    """
    Dataset class for GEP data. Implements preprocessing and caching
    to speed up data loading for repeated runs.
    """

    def __init__(self,
                 file_paths: List[str],
                 processed_data_dir: Union[str, Path],
                 scaling_by_constant: Union[bool, float] = True,
                 gene_list_file: Optional[Union[str, Path]] = None,
                 remove_low_var_genes: bool = False,
                 min_var: float = 1.0,
                 cell_cell2ave_exp_file_path: Optional[Union[str, Path]] = None,
                 force_reprocess: bool = False):
        """
        Args:
            file_paths: List of file paths containing the H5AD data, or .csv file in TPM format with shape (samples x genes)
            processed_data_dir: Directory to save/load preprocessed data.
            scaling_by_constant: If True, scales by 20.0. If float, scales by that value.
            gene_list_file: Path to a file containing the gene list for filtering.
            remove_low_var_genes: If True, remove low variance genes.
            min_var: Minimum variance for gene filtering.
            cell_cell2ave_exp_file_path: Path to a table (genes x cell types, TPM) for additional gene filtering.
            force_reprocess: If True, reprocesses data even if cached files exist.
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Define paths for cached processed files
        self.cached_data_path = self.processed_data_dir / "data.pt"
        self.cached_labels_path = self.processed_data_dir / "labels.pt"
        self.cached_gene_list_path = self.processed_data_dir / "gene_list.txt"
        self.cached_cell_types_path = self.processed_data_dir / "cell_types.txt"
        self.cached_sample_ids_path = self.processed_data_dir / "sample_ids.txt"

        if not force_reprocess and self._load_from_cache():
            log_message(f"Successfully loaded preprocessed data from {self.processed_data_dir}")
        else:
            log_message("Preprocessing data from scratch...")
            self._preprocess_and_cache(
                file_paths, scaling_by_constant, gene_list_file,
                remove_low_var_genes, min_var, cell_cell2ave_exp_file_path
            )

    def _load_from_cache(self) -> bool:
        files_to_check = [
            self.cached_data_path, self.cached_labels_path,
            self.cached_gene_list_path, self.cached_cell_types_path,
            self.cached_sample_ids_path
        ]
        if not all(f.exists() for f in files_to_check):
            return False

        self.data = torch.load(self.cached_data_path)
        self.labels = torch.load(self.cached_labels_path)
        self.gene_list = self._load_list_txt(self.cached_gene_list_path)
        self.cell_types = self._load_list_txt(self.cached_cell_types_path)
        self.sample_ids = self._load_list_txt(self.cached_sample_ids_path)
        return True

    def _save_list_txt(self, data_list: List[str], file_path: Path):
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")

    def _load_list_txt(self, file_path: Path) -> List[str]:
        with open(file_path, 'r') as f:
            return f.read().splitlines()

    def _preprocess_and_cache(self, file_paths, scaling_by_constant, gene_list_file,
                              remove_low_var_genes, min_var, cell_cell2ave_exp_file_path):
        all_data_dfs = []
        all_cell_prop_dfs = []
        for path_str in file_paths:
            log_message(f"Reading data from {path_str}")
            if path_str.endswith(".h5ad"):
                h5ad_obj = ReadH5AD(path_str)  # Assuming ReadH5AD takes Path
                gep_data_df = h5ad_obj.get_df(convert_to_tpm=True)
                cell_prop_df = h5ad_obj.get_cell_fraction()
            elif path_str.endswith(".csv"):
                gep_data_df = pd.read_csv(path_str, index_col=0)
                cell_prop_df = pd.DataFrame()  # No cellular proportion during evaluation, such as TCGA
            else:
                raise ValueError(f"Unrecognized file format: {path_str}")
            log_message(f"Raw data shape from {Path(path_str).name}: {gep_data_df.shape}")
            all_data_dfs.append(gep_data_df)
            all_cell_prop_dfs.append(cell_prop_df)

        if not all_data_dfs:
            raise ValueError("No data loaded. Please check file_paths.")

        # Merge datasets
        self.gep_data_df = pd.concat(all_data_dfs, axis=0, join='inner')
        self.cell_prop_df = pd.concat(all_cell_prop_dfs, axis=0, join='inner')
        del all_data_dfs, all_cell_prop_dfs  # Free memory
        if not self.cell_prop_df.empty:
            assert len(self.gep_data_df) == len(self.cell_prop_df), "Sample count mismatch post-concat"
            assert np.all(self.gep_data_df.index == self.cell_prop_df.index), "Sample ID/order mismatch post-concat"

        if gene_list_file is not None:
            target_genes = load_gene_list(Path(gene_list_file))
            # align with the loaded gene list
            gep_exp_obj = ReadExp(self.gep_data_df, exp_type='TPM')
            gep_exp_obj.align_with_gene_list(gene_list=target_genes, fill_not_exist=True)
            # Keep only genes present in both the data and the target list
            # common_genes = self.gep_data_df.columns.intersection(target_genes)
            self.gep_data_df = gep_exp_obj.get_exp()
            log_message(f"Data shape after common gene filtering: {self.gep_data_df.shape}")

        if remove_low_var_genes:
            self._apply_low_var_gene_removal(min_var=min_var, cell_cell2ave_exp_file_path=cell_cell2ave_exp_file_path)

        # Rescaling (assuming non_log2log_cpm operates on and returns a DataFrame)
        self.gep_data_df = non_log2log_cpm(self.gep_data_df, transpose=False)
        log_message(f"Data shape after transformations: {self.gep_data_df.shape}")

        self.data = torch.from_numpy(self.gep_data_df.values.astype(np.float32))

        scaling_value = 20.0  # Default if scaling_by_constant is True
        if isinstance(scaling_by_constant, float):
            scaling_value = scaling_by_constant

        if scaling_by_constant:  # True or a float value
            self.data = self.data / scaling_value

        # Ensure labels align with the potentially filtered/reordered gep_data_df
        if not self.cell_prop_df.empty:
            self.labels = torch.from_numpy(self.cell_prop_df.loc[self.gep_data_df.index].values.astype(np.float32))
            self.cell_types = self.cell_prop_df.columns.to_list()  # Cell types don't change by gene filtering
        else:
            self.labels = []
            self.cell_types = []

        self.gene_list = self.gep_data_df.columns.to_list()
        self.sample_ids = self.gep_data_df.index.to_list()

        # Save processed data to cache
        torch.save(self.data, self.cached_data_path)
        if self.labels is not None:
            torch.save(self.labels, self.cached_labels_path)
        self._save_list_txt(self.gene_list, self.cached_gene_list_path)
        if self.cell_types is not None:
            self._save_list_txt(self.cell_types, self.cached_cell_types_path)
        self._save_list_txt(self.sample_ids, self.cached_sample_ids_path)
        log_message(f"Finished processing and saved data to {self.processed_data_dir}")

        # Clean up large DataFrames if they are no longer needed as attributes
        del self.gep_data_df
        if hasattr(self, 'cell_prop_df'):  # It might be used by get_cell_prop if that returns DataFrame
            del self.cell_prop_df

    def _apply_low_var_gene_removal(self, min_var: float, cell_cell2ave_exp_file_path: Optional[Union[str, Path]]):
        """Modifies self.gep_data_df in place."""
        n_gene_before_filter = self.gep_data_df.shape[1]
        gene_variances = self.gep_data_df.var(axis=0)
        self.gep_data_df = self.gep_data_df.loc[:, gene_variances > min_var]

        if cell_cell2ave_exp_file_path is not None:
            try:
                cell_ave_exp_df = pd.read_csv(cell_cell2ave_exp_file_path, index_col=0)
                # Assuming cell_ave_exp_df.index contains gene symbols/IDs to keep.
                # The original logic for 'var' column in this ref seemed specific.
                # A common use is to ensure genes are also present in this reference.
                genes_in_ref = cell_ave_exp_df.index
                common_genes_after_var_and_ref = self.gep_data_df.columns.intersection(genes_in_ref)
                self.gep_data_df = self.gep_data_df[common_genes_after_var_and_ref]
            except FileNotFoundError:
                logger.warning(f"File not found: {cell_cell2ave_exp_file_path}. Skipping this part of gene filtering.")
            except Exception as e:  # Catch other potential pandas errors
                logger.error(
                    f"Error processing {cell_cell2ave_exp_file_path}: {e}. Skipping this part of gene filtering.")

        n_gene_after_filter = self.gep_data_df.shape[1]
        log_message(f"Gene filtering: {n_gene_before_filter} genes -> {n_gene_after_filter} genes. "
                    f"Removed {n_gene_before_filter - n_gene_after_filter} genes.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> dict:
        x = self.data[index]
        if self.labels:
            y = self.labels[index]
        else:
            y = []
        return DatasetOutput(data=x, labels=y)  # Or simply {'data': x, 'labels': y}

    def save_gene_list(self, file_path: Union[str, Path]):
        check_dir(Path(file_path).parent)
        self._save_list_txt(self.gene_list, Path(file_path))
        logger.info(f"Gene list saved to {file_path}")

    def save_cell_types(self, file_path: Union[str, Path]):
        check_dir(Path(file_path).parent)
        self._save_list_txt(self.cell_types, Path(file_path))
        logger.info(f"Cell types saved to {file_path}")

    def get_gene_list(self) -> List[str]:
        return self.gene_list

    def get_cell_types(self) -> List[str]:
        return self.cell_types

    def get_sample_ids(self) -> List[str]:
        return self.sample_ids

    def get_cell_prop(self) -> pd.DataFrame:
        """Reconstructs cell proportions DataFrame from loaded tensors."""
        if hasattr(self, 'labels') and hasattr(self, 'sample_ids') and hasattr(self, 'cell_types'):
            return pd.DataFrame(self.labels.cpu().numpy(), index=self.sample_ids, columns=self.cell_types)
        else:
            raise AttributeError("Dataset not fully loaded or preprocessed. Call __init__ first.")


def find_sct_gep_of_bulk_sample(
        sct_gep_dataset_file_path: Union[str, Path],
        sample2cell_id_file_path: Union[str, Path],
        bulk_dataset: GEPDataset,
        cell_types: List[str],
        result_dir: Optional[Union[str, Path]] = None,
        random_seed: Optional[int] = 42,
        n_samples: int = 3,
        selected_sample2cell_id_file_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Selects random bulk samples, finds corresponding single-cell IDs from a mapping file,
    loads their GEPs from a single-cell dataset, processes them,
    and either saves them grouped by cell type or returns them as a dictionary.
    """
    # Convert string paths to Path objects for consistent handling
    sct_gep_fp = Path(sct_gep_dataset_file_path)
    sample2cell_fp = Path(sample2cell_id_file_path)

    # --- 1. Input Validation and Setup ---
    if not sct_gep_fp.exists():
        logger.error(f"SCT GEP dataset file not found: {sct_gep_fp}")
        raise FileNotFoundError(f"SCT GEP dataset file not found: {sct_gep_fp}")
    if not sample2cell_fp.exists():
        logger.error(f"Sample to cell ID mapping file not found: {sample2cell_fp}")
        raise FileNotFoundError(f"Sample to cell ID mapping file not found: {sample2cell_fp}")

    all_bulk_sample_ids = bulk_dataset.get_sample_ids()
    target_gene_list = bulk_dataset.get_gene_list()  # Gene list from bulk dataset perspective

    if not all_bulk_sample_ids:
        logger.warning("No sample IDs found in bulk_dataset. Cannot select samples.")
        return {} if not result_dir else None  # Return empty dict or None based on mode

    # --- 2. Select Random Bulk Samples ---
    rng = np.random.default_rng(seed=random_seed)
    num_available_samples = len(all_bulk_sample_ids)
    n_to_select = min(n_samples, num_available_samples)

    if n_to_select < n_samples:
        logger.warning(
            f"Requested {n_samples} samples, but only {num_available_samples} available. Selecting {n_to_select}.")
    if n_to_select == 0:
        logger.warning("No samples to select.")
        return {} if not result_dir else None

    query_indices = rng.choice(num_available_samples, size=n_to_select, replace=False)
    query_bulk_ids = [all_bulk_sample_ids[i] for i in query_indices]
    logger.info(f"Selected bulk sample IDs: {query_bulk_ids}")

    # --- 3. Load Sample-to-Cell_ID Mapping and Filter ---
    try:
        sample2cell_ids_df_all = pd.read_csv(sample2cell_fp, index_col=0)
    except Exception as e:
        logger.error(f"Error reading sample to cell ID mapping file {sample2cell_fp}: {e}")
        raise

    # Filter mapping for selected query_bulk_ids that are actually in the mapping file
    mask = sample2cell_ids_df_all.index.isin(query_bulk_ids)
    filtered_mapping_df = sample2cell_ids_df_all.loc[mask, ['cell_type', 'selected_cell_id']].copy()
    # filtered_mapping_df = sample2cell_ids_df_all.reindex(query_bulk_ids).dropna(
    #     subset=['cell_type', 'selected_cell_id'])

    if filtered_mapping_df.empty:
        logger.warning(f"None of the selected query_ids {query_bulk_ids} found in mapping file or had valid entries.")
        return {} if not result_dir else None

    # Ensure 'selected_cell_id' is suitable for indexing (e.g. string type if H5AD obs_names are strings)
    filtered_mapping_df['selected_cell_id'] = filtered_mapping_df['selected_cell_id'].astype(str)

    if selected_sample2cell_id_file_path:
        selected_fp = Path(selected_sample2cell_id_file_path)
        try:
            selected_fp.parent.mkdir(parents=True, exist_ok=True)
            filtered_mapping_df.to_csv(selected_fp)
            logger.info(f"Saved mapping for selected samples to: {selected_fp}")
        except Exception as e:
            logger.error(f"Error saving selected sample to cell ID mapping: {e}")

    unique_sct_cell_ids_to_load = filtered_mapping_df['selected_cell_id'].unique().tolist()
    if not unique_sct_cell_ids_to_load:
        logger.warning("No unique single-cell IDs to load based on selected bulk samples.")
        return {} if not result_dir else None

    # --- 4. Load and Process Single-Cell GEPs ---
    logger.info(f"Loading GEPs for {len(unique_sct_cell_ids_to_load)} unique single-cell IDs...")
    try:
        # **CRITICAL EFFICIENCY POINT**:
        # Modify ReadH5AD().get_df() to accept a list of cell IDs (obs_names)
        # and load only those cells. This avoids loading the entire SCT GEP matrix.
        sct_gep_loader = ReadH5AD(sct_gep_fp)
        # This call assumes get_df can efficiently fetch specific cells
        sct_geps_df_raw = sct_gep_loader.get_df(obs_names=unique_sct_cell_ids_to_load)
    except Exception as e:
        logger.error(f"Error loading SCT GEPs for specified cell IDs: {e}")
        raise

    if sct_geps_df_raw.empty:
        logger.warning(f"No GEP data loaded for the selected cell IDs: {unique_sct_cell_ids_to_load}")
        return {} if not result_dir else None
    logger.info(f"Successfully loaded GEPs for {sct_geps_df_raw.shape[0]} single cells.")

    # Further processing (alignment, transformation)
    sct_exp_processor = ReadExp(sct_geps_df_raw, exp_type='log_space')  # Assuming input is log_space
    sct_exp_processor.align_with_gene_list(gene_list=target_gene_list, fill_not_exist=True)
    processed_sct_geps_df = sct_exp_processor.get_exp()  # DataFrame with cells as index, aligned genes as columns
    logger.info(f"Processed SCT GEPs, final shape: {processed_sct_geps_df.shape}")

    # --- 5. Group GEPs by Cell Type and Save/Return ---
    output_geps_by_celltype: Dict[str, pd.DataFrame] = {}

    result_dir_path = None
    if result_dir:
        result_dir_path = Path(result_dir)
        result_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving SCT GEPs by cell type to directory: {result_dir_path}")

    # Create a map from selected_cell_id to cell_type for efficient lookup
    cell_id_to_type_map = filtered_mapping_df.set_index('selected_cell_id')['cell_type']

    for cell_type_target in cell_types:  # Iterate through requested cell types
        # Find which of the loaded and processed single cells belong to this cell_type_target
        cell_ids_for_current_type = []
        for sct_cell_id in processed_sct_geps_df.index:
            if sct_cell_id in cell_id_to_type_map and cell_id_to_type_map[sct_cell_id] == cell_type_target:
                cell_ids_for_current_type.append(sct_cell_id)

        if not cell_ids_for_current_type:
            logger.warning(f"No SCT GEPs found for cell type '{cell_type_target}' among the loaded/processed cells.")
            # Create an empty DataFrame with correct gene columns for consistency if saving
            specific_cell_type_geps_df = pd.DataFrame(columns=processed_sct_geps_df.columns)
        else:
            specific_cell_type_geps_df = processed_sct_geps_df.loc[cell_ids_for_current_type, :]

        if result_dir:
            # Filename clarification: n_samples refers to bulk samples, not necessarily number of single cells
            result_file = result_dir_path / f"sct_gep_{cell_type_target}_from_{n_to_select}_bulksamples.csv"
            try:
                specific_cell_type_geps_df.T.to_csv(result_file)  # Transpose for genes x cells format
                logger.info(
                    f"Saved GEPs for {cell_type_target} ({specific_cell_type_geps_df.shape[0]} cells) to {result_file}")
            except Exception as e:
                logger.error(f"Error saving GEPs for {cell_type_target}: {e}")
        else:
            output_geps_by_celltype[cell_type_target] = specific_cell_type_geps_df

    if not result_dir:
        return output_geps_by_celltype
    return None  # Explicit return if saving to disk