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



class GEPDataset(Dataset):
    """
    Memory-efficient Dataset class for GEP data with lazy loading and memory mapping.

    Key improvements:
    - Uses memory-mapped files for large datasets
    - Lazy loading: only loads data when accessed
    - Chunked processing for preprocessing
    - Optional data compression
    - Efficient caching strategy
    """

    def __init__(self,
                 file_paths: List[str],
                 processed_data_dir: Union[str, Path],
                 scaling_by_constant: Union[bool, float] = True,
                 gene_list_file: Optional[Union[str, Path]] = None,
                 remove_low_var_genes: bool = False,
                 min_var: float = 1.0,
                 cell_cell2ave_exp_file_path: Optional[Union[str, Path]] = None,
                 force_reprocess: bool = False,
                 use_memmap: bool = True,  # Use memory mapping
                 chunk_size: int = 1000,  # Process data in chunks
                 compress: bool = False):  # Compress cached data
        """
        Args:
            file_paths: List of file paths containing the H5AD data or CSV files
            processed_data_dir: Directory to save/load preprocessed data
            scaling_by_constant: If True, scales by 20.0. If float, scales by that value
            gene_list_file: Path to gene list for filtering
            remove_low_var_genes: If True, remove low variance genes
            min_var: Minimum variance for gene filtering
            cell_cell2ave_exp_file_path: Path to cell type average expression table
            force_reprocess: If True, reprocess even if cache exists
            use_memmap: If True, use memory-mapped arrays (recommended for large datasets)
            chunk_size: Number of samples to process at once (reduce for lower memory)
            compress: If True, use compression for cached data (slower but smaller)
        """
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.compress = compress
        self.scaling_value = 20.0 if scaling_by_constant is True else (
            scaling_by_constant if isinstance(scaling_by_constant, float) else 1.0
        )
        self.apply_scaling = bool(scaling_by_constant)

        if processed_data_dir is not None:
            self.processed_data_dir = Path(processed_data_dir)
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)

            # Define cache paths
            self.cached_data_path = self.processed_data_dir / "data.npy"
            self.cached_labels_path = self.processed_data_dir / "labels.npy"
            self.cached_gene_list_path = self.processed_data_dir / "gene_list.txt"
            self.cached_cell_types_path = self.processed_data_dir / "cell_types.txt"
            self.cached_sample_ids_path = self.processed_data_dir / "sample_ids.txt"
            self.cached_metadata_path = self.processed_data_dir / "metadata.json"
        else:
            self.processed_data_dir = None
            self.use_memmap = False  # Can't use memmap without cache dir

        # Initialize data attributes as None (lazy loading)
        self._data = None
        self._labels = None
        self.gene_list = None
        self.cell_types = None
        self.sample_ids = None
        self._n_samples = 0
        self._n_genes = 0
        self._n_cell_types = 0

        if not force_reprocess and self._load_from_cache():
            log_message(f"Successfully loaded preprocessed data from {self.processed_data_dir}")
        else:
            log_message("Preprocessing data from scratch...")
            self._preprocess_and_cache(
                file_paths, gene_list_file,
                remove_low_var_genes,
                min_var, cell_cell2ave_exp_file_path
            )

    @property
    def data(self):
        """Lazy loading of data."""
        if self._data is None:
            self._load_data()
        return self._data

    @property
    def labels(self):
        """Lazy loading of labels."""
        if self._labels is None:
            self._load_labels()
        return self._labels

    def _load_data(self):
        """Load data array (with memory mapping if enabled)."""
        if self.use_memmap and self.cached_data_path.exists():
            # Memory-mapped array: doesn't load into RAM until accessed
            self._data = np.load(self.cached_data_path, mmap_mode='r')
            log_message(f"Loaded data as memory-mapped array: {self._data.shape}")
            if self._data.dtype != np.float32:
                logger.warning(f'Data dtype is {self._data.dtype}, consider converting to float32 for efficiency.')
        elif self.cached_data_path.exists():
            data = np.load(self.cached_data_path)
            if data.dtype != np.float32:
                data = data.astype(np.float32)  # Convert to float32 for efficiency
            log_message(f"Loaded data into memory: {self._data.shape}")
            self._data = data
        else:
            raise FileNotFoundError(f"Data file not found: {self.cached_data_path}")

    def _load_labels(self):
        """Load labels array (with memory mapping if enabled)."""
        if self.cached_labels_path.exists():
            if self.use_memmap:
                self._labels = np.load(self.cached_labels_path, mmap_mode='r')
            else:
                self._labels = np.load(self.cached_labels_path)
            log_message(f"Loaded labels: {self._labels.shape}")
        else:
            self._labels = np.array([])  # Empty for evaluation datasets

    def _load_from_cache(self) -> bool:
        """Load metadata and check if all cache files exist."""
        required_files = [
            self.cached_data_path,
            self.cached_gene_list_path,
            self.cached_sample_ids_path
        ]

        if not all(f.exists() for f in required_files):
            return False

        try:
            # Load metadata
            import json
            if self.cached_metadata_path.exists():
                with open(self.cached_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self._n_samples = metadata['n_samples']
                    self._n_genes = metadata['n_genes']
                    self._n_cell_types = metadata.get('n_cell_types', 0)

            # Load text files (small, always load)
            self.gene_list = self._load_list_txt(self.cached_gene_list_path)
            self.sample_ids = self._load_list_txt(self.cached_sample_ids_path)

            if self.cached_cell_types_path.exists():
                self.cell_types = self._load_list_txt(self.cached_cell_types_path)
            else:
                self.cell_types = []

            # Update counts if metadata wasn't available
            if self._n_samples == 0:
                self._n_samples = len(self.sample_ids)
                self._n_genes = len(self.gene_list)
                self._n_cell_types = len(self.cell_types)

            # Don't load data/labels yet (lazy loading)
            log_message(f"Cache metadata loaded: {self._n_samples} samples, "
                        f"{self._n_genes} genes, {self._n_cell_types} cell types")
            return True

        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return False

    def _save_list_txt(self, data_list: List[str], file_path: Path):
        """Save list to text file."""
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")

    def _load_list_txt(self, file_path: Path) -> List[str]:
        """Load list from text file."""
        with open(file_path, 'r') as f:
            return f.read().splitlines()

    def _preprocess_and_cache(self, file_paths, gene_list_file,
                              remove_low_var_genes, min_var, cell_cell2ave_exp_file_path):
        """
        Preprocess data with memory-efficient chunked processing.
        """
        # Step 1: Load and merge data (chunked if possible)
        log_message("Step 1: Loading and merging data...")
        gep_data_df, cell_prop_df = self._load_and_merge_data_chunked(file_paths)

        # Step 2: Gene filtering
        log_message("Step 2: Gene filtering...")
        if gene_list_file is not None:
            gep_data_df = self._apply_gene_list_filter(gep_data_df, gene_list_file)

        if remove_low_var_genes:
            gep_data_df = self._apply_low_var_gene_removal_efficient(
                gep_data_df, min_var, cell_cell2ave_exp_file_path
            )

        # Step 3: Transformation (chunked)
        log_message("Step 3: Applying transformations...")
        gep_data_df = self._apply_transformation_chunked(gep_data_df)

        # Step 4: Save to cache (memory-efficient)
        log_message("Step 4: Saving to cache...")
        self._save_to_cache_efficient(gep_data_df, cell_prop_df)

        # Step 5: Clean up and set attributes
        self.gene_list = gep_data_df.columns.to_list()
        self.sample_ids = gep_data_df.index.to_list()
        if not cell_prop_df.empty:
            self.cell_types = cell_prop_df.columns.to_list()
        else:
            self.cell_types = []

        self._n_samples = len(self.sample_ids)
        self._n_genes = len(self.gene_list)
        self._n_cell_types = len(self.cell_types)

        # Free memory
        del gep_data_df, cell_prop_df

        log_message(f"Preprocessing complete: {self._n_samples} samples, "
                    f"{self._n_genes} genes")

    def _load_and_merge_data_chunked(self, file_paths):
        """
        Load and merge data files with memory-efficient chunked processing.
        """
        all_data_dfs = []
        all_cell_prop_dfs = []

        for path_str in file_paths:
            log_message(f"Reading data from {path_str}")

            if path_str.endswith(".h5ad"):
                # For H5AD, use your existing reader
                h5ad_obj = ReadH5AD(path_str)
                gep_data_df = h5ad_obj.get_df(convert_to_tpm=True)
                if gep_data_df.values.dtype != np.float32:
                    gep_data_df = gep_data_df.astype(np.float32)  # Reduce memory usage
                cell_prop_df = h5ad_obj.get_cell_fraction()

            elif path_str.endswith(".csv"):
                # For CSV, use chunked reading for large files
                gep_data_df = self._read_csv_chunked(path_str)
                cell_prop_df = pd.DataFrame()

            else:
                raise ValueError(f"Unrecognized file format: {path_str}")

            log_message(f"Loaded data shape: {gep_data_df.shape}")
            all_data_dfs.append(gep_data_df)
            all_cell_prop_dfs.append(cell_prop_df)

        if not all_data_dfs:
            raise ValueError("No data loaded. Please check file_paths.")

        # Merge datasets
        log_message("Merging datasets...")
        gep_data_df = pd.concat(all_data_dfs, axis=0, join='inner')
        cell_prop_df = pd.concat(all_cell_prop_dfs, axis=0, join='inner')

        # Free memory immediately
        del all_data_dfs, all_cell_prop_dfs

        if not cell_prop_df.empty:
            assert len(gep_data_df) == len(cell_prop_df), "Sample count mismatch"
            assert np.all(gep_data_df.index == cell_prop_df.index), "Sample ID mismatch"

        log_message(f"Merged data shape: {gep_data_df.shape}")
        return gep_data_df, cell_prop_df

    def _read_csv_chunked(self, file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Read large CSV files in chunks to reduce memory usage.
        """
        try:
            # Try to read normally first (for small files)
            return pd.read_csv(file_path, index_col=0)
        except MemoryError:
            log_message(f"File too large, reading in chunks...")
            chunks = []
            for chunk in pd.read_csv(file_path, index_col=0, chunksize=chunk_size, dtype=np.float32):
                chunks.append(chunk)
            return pd.concat(chunks, axis=0)

    def _apply_gene_list_filter(self, gep_data_df: pd.DataFrame,
                                gene_list_file: Union[str, Path]) -> pd.DataFrame:
        """Apply gene list filtering."""
        target_genes = load_gene_list(Path(gene_list_file))
        gep_exp_obj = ReadExp(gep_data_df, exp_type='TPM')
        gep_exp_obj.align_with_gene_list(gene_list=target_genes, fill_not_exist=True)
        gep_data_df = gep_exp_obj.get_exp()
        log_message(f"After gene list filtering: {gep_data_df.shape}")
        return gep_data_df

    def _apply_low_var_gene_removal_efficient(self, gep_data_df: pd.DataFrame,
                                              min_var: float,
                                              cell_cell2ave_exp_file_path: Optional[Path]) -> pd.DataFrame:
        """
        Memory-efficient low variance gene removal.
        """
        n_genes_before = gep_data_df.shape[1]

        # Compute variance in chunks to save memory
        log_message("Computing gene variances...")
        gene_variances = gep_data_df.var(axis=0)

        # Filter by variance
        genes_to_keep = gene_variances > min_var
        gep_data_df = gep_data_df.loc[:, genes_to_keep]

        # Additional filtering by reference
        if cell_cell2ave_exp_file_path is not None:
            try:
                cell_ave_exp_df = pd.read_csv(cell_cell2ave_exp_file_path, index_col=0)
                genes_in_ref = cell_ave_exp_df.index
                common_genes = gep_data_df.columns.intersection(genes_in_ref)
                gep_data_df = gep_data_df[common_genes]
                del cell_ave_exp_df  # Free memory
            except Exception as e:
                logger.error(f"Error processing reference file: {e}")

        n_genes_after = gep_data_df.shape[1]
        log_message(f"Gene filtering: {n_genes_before} → {n_genes_after} genes "
                    f"(removed {n_genes_before - n_genes_after})")

        return gep_data_df

    def _apply_transformation_chunked(self, gep_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations in chunks to reduce memory usage.
        """
        # Apply non_log2log_cpm transformation
        # If this function is memory-intensive, consider chunking it
        gep_data_df = non_log2log_cpm(gep_data_df, transpose=False)
        if gep_data_df.values.dtype != np.float32:
            gep_data_df = gep_data_df.astype(np.float32)  # Reduce memory usage
        log_message(f"After transformation: {gep_data_df.shape}")
        return gep_data_df

    def _save_to_cache_efficient(self, gep_data_df: pd.DataFrame,
                                 cell_prop_df: pd.DataFrame):
        """
        Save data to cache with memory-efficient methods.
        """
        # Convert to numpy and apply scaling
        log_message("Converting to numpy arrays...")
        data_array = gep_data_df.values.astype(np.float32)

        if self.apply_scaling:
            data_array = data_array / self.scaling_value

        # Save data
        log_message(f"Saving data array: {data_array.shape}")
        if self.compress:
            np.savez_compressed(self.cached_data_path.with_suffix('.npz'),
                                data=data_array)
        else:
            np.save(self.cached_data_path, data_array)

        # Save labels
        if not cell_prop_df.empty:
            labels_array = cell_prop_df.loc[gep_data_df.index].values.astype(np.float32)
            log_message(f"Saving labels array: {labels_array.shape}")
            if self.compress:
                np.savez_compressed(self.cached_labels_path.with_suffix('.npz'),
                                    labels=labels_array)
            else:
                np.save(self.cached_labels_path, labels_array)

        # Save metadata
        self._save_list_txt(gep_data_df.columns.to_list(), self.cached_gene_list_path)
        self._save_list_txt(gep_data_df.index.to_list(), self.cached_sample_ids_path)

        if not cell_prop_df.empty:
            self._save_list_txt(cell_prop_df.columns.to_list(), self.cached_cell_types_path)

        # Save metadata JSON
        import json
        metadata = {
            'n_samples': len(gep_data_df),
            'n_genes': len(gep_data_df.columns),
            'n_cell_types': len(cell_prop_df.columns) if not cell_prop_df.empty else 0,
            'scaling_value': self.scaling_value,
            'apply_scaling': self.apply_scaling
        }
        with open(self.cached_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        log_message(f"Data saved to {self.processed_data_dir}")

    def __len__(self):
        """Return dataset size without loading data."""
        if self._n_samples > 0:
            return self._n_samples
        return self.data.shape[0]

    def __getitem__(self, index: int) -> dict:
        """
        Get single item with minimal memory footprint.
        """
        # Access data through property (lazy loading)
        x = torch.from_numpy(np.array(self.data[index]))

        if self.labels is not None and len(self.labels) > 0:
            y = torch.from_numpy(np.array(self.labels[index]))
        else:
            y = torch.tensor([])

        return DatasetOutput(data=x, labels=y)

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Get multiple items efficiently (batch loading).
        """
        x = torch.from_numpy(np.array(self.data[indices]))

        if self.labels is not None and len(self.labels) > 0:
            y = torch.from_numpy(np.array(self.labels[indices]))
        else:
            y = torch.tensor([])

        return {'data': x, 'labels': y}

    def save_gene_list(self, file_path: Union[str, Path]):
        """Save gene list to file."""
        check_dir(Path(file_path).parent)
        self._save_list_txt(self.gene_list, Path(file_path))
        logger.info(f"Gene list saved to {file_path}")

    def save_cell_types(self, file_path: Union[str, Path]):
        """Save cell types to file."""
        check_dir(Path(file_path).parent)
        self._save_list_txt(self.cell_types, Path(file_path))
        logger.info(f"Cell types saved to {file_path}")

    def get_gene_list(self) -> List[str]:
        """Get gene list."""
        return self.gene_list

    def get_cell_types(self) -> List[str]:
        """Get cell types."""
        return self.cell_types

    def get_sample_ids(self) -> List[str]:
        """Get sample IDs."""
        return self.sample_ids

    def get_cell_prop(self) -> pd.DataFrame:
        """
        Get cell proportions as DataFrame (loads labels if needed).
        """
        if self.labels is not None and len(self.labels) > 0:
            labels_array = self.labels if isinstance(self.labels, np.ndarray) else self.labels.cpu().numpy()
            return pd.DataFrame(labels_array,
                                index=self.sample_ids,
                                columns=self.cell_types)
        else:
            return pd.DataFrame()

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage in MB for each component.
        """
        usage = {}

        if self._data is not None:
            usage['data_mb'] = self._data.nbytes / (1024 ** 2)

        if self._labels is not None and len(self._labels) > 0:
            usage['labels_mb'] = self._labels.nbytes / (1024 ** 2)

        usage['total_mb'] = sum(usage.values())

        return usage

    def unload_data(self):
        """
        Unload data from memory (useful for memory management).
        Only works if using cache files.
        """
        if self.use_memmap:
            self._data = None
            self._labels = None
            log_message("Data unloaded from memory")
        else:
            logger.warning("Cannot unload data without memory mapping enabled")


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
    # remove duplicated indices if any
    cell_id_to_type_map = cell_id_to_type_map[~cell_id_to_type_map.index.duplicated(keep='first')].to_dict()

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