from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ..utility.read_file import ReadH5AD
from ..utility import non_log2log_cpm, check_dir, log_message
from ..configs import GEPDatasetConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

_DEFAULT_MAX_PARALLEL_SOURCE_FILE_LOADS = 2


class DatasetOutput(OrderedDict):
    """Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
    the ``ModelOutput`` class from huggingface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value
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


def _read_source_gene_names(file_path: Union[str, Path]) -> list[str]:
    """Read only gene names from one source file without materializing full data."""
    path = Path(file_path)
    path_str = str(path)
    if path_str.endswith(".h5ad"):
        h5ad_obj = ReadH5AD(path, backed="r")
        try:
            return [str(gene) for gene in h5ad_obj.get_var_names()]
        finally:
            h5ad_obj.close()
    if path_str.endswith(".csv"):
        header_df = pd.read_csv(path, nrows=0)
        return [str(col) for col in header_df.columns.tolist()[1:]]
    raise ValueError(f"Unrecognized file format: {path}")


def _ordered_intersection(base_order: Sequence[str], keep: set[str]) -> list[str]:
    """Return items from base_order that are present in keep, preserving order."""
    return [item for item in base_order if item in keep]


def _discover_final_target_gene_list(
    file_paths: Sequence[Union[str, Path]],
    gene_list_file: Optional[Union[str, Path]] = None,
) -> list[str]:
    """Compute the final ordered common gene list before reading full matrices."""
    if not file_paths:
        raise ValueError("No data loaded. Please check file_paths.")

    source_gene_lists = [_read_source_gene_names(path_like) for path_like in file_paths]
    common_gene_set = set(source_gene_lists[0])
    for gene_list in source_gene_lists[1:]:
        common_gene_set &= set(gene_list)

    if gene_list_file is not None:
        configured_gene_list = [str(gene) for gene in load_gene_list(Path(gene_list_file))]
        final_genes = _ordered_intersection(configured_gene_list, common_gene_set)
    else:
        final_genes = _ordered_intersection(source_gene_lists[0], common_gene_set)

    if not final_genes:
        raise ValueError("No common genes remain after source intersection and configured gene filtering.")

    log_message(f"Discovered {len(final_genes)} common target genes before full matrix loading.")
    return final_genes


def _load_or_create_cached_common_gene_list(
    file_paths: Sequence[Union[str, Path]],
    *,
    gene_list_file: Optional[Union[str, Path]] = None,
    cache_file_path: Optional[Union[str, Path]] = None,
) -> list[str]:
    """Load the cache-local common gene list when available, otherwise discover and save it."""
    cache_path = Path(cache_file_path) if cache_file_path is not None else None
    if cache_path is not None and cache_path.exists():
        cached_genes = [str(gene) for gene in load_gene_list(cache_path)]
        if cached_genes:
            log_message(f"Loaded {len(cached_genes)} common target genes from {cache_path}")
            return cached_genes
        logger.warning(f"Cached common gene list is empty at {cache_path}; rediscovering it.")

    final_genes = _discover_final_target_gene_list(
        file_paths=file_paths,
        gene_list_file=gene_list_file,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        GEPCacheManager.save_list_txt(final_genes, cache_path)
        log_message(f"Saved {len(final_genes)} common target genes to {cache_path}")
        return [str(gene) for gene in load_gene_list(cache_path)]
    return final_genes


def _reindex_expression_to_gene_list(exp_df: pd.DataFrame, target_gene_list: Sequence[str]) -> pd.DataFrame:
    """Reindex expression columns to the requested gene order, filling missing genes with zeros."""
    if exp_df.empty:
        return pd.DataFrame(columns=list(target_gene_list), index=exp_df.index)
    return exp_df.reindex(columns=list(target_gene_list), fill_value=0.0)


def _load_bulk_sample_ids_from_file(file_path: Union[str, Path]) -> list[str]:
    """Load bulk sample IDs from a training expression file without full preprocessing."""
    path = Path(file_path)
    if str(path).endswith(".h5ad"):
        h5ad_obj = ReadH5AD(path, backed="r")
        try:
            return h5ad_obj.get_h5ad().obs_names.to_list()
        finally:
            h5ad_obj.close()
    if str(path).endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        return df.index.astype(str).tolist()
    raise ValueError(f"Unrecognized training set file format: {path}")


def _namespace_sample_ids(sample_ids: Sequence[Union[str, Path]], namespace: Optional[str]) -> list[str]:
    """Add a stable namespace prefix to sample IDs for internal dataset alignment."""
    if namespace is None or str(namespace).strip() == "":
        return [str(sample_id) for sample_id in sample_ids]
    prefix = str(namespace).strip()
    return [f"{prefix}::{sample_id}" for sample_id in sample_ids]


def _load_cached_aligned_sct_geps(
    sct_gep_fp: Path,
    selected_cell_ids: list[str],
    target_gene_list: list[str],
    cache_sct_query_results: bool = True,
    sct_query_cache_file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load matched SCT cells and align them to the requested bulk gene order."""
    if not selected_cell_ids:
        return pd.DataFrame(columns=target_gene_list)

    cache_fp = Path(sct_query_cache_file_path) if sct_query_cache_file_path else Path(
        str(sct_gep_fp) + ".query_cache.pkl"
    )

    cached_df = pd.DataFrame()
    if cache_sct_query_results and cache_fp.exists():
        try:
            cache_obj = pd.read_pickle(cache_fp)
            if (
                isinstance(cache_obj, dict)
                and cache_obj.get("source_path") == str(sct_gep_fp)
                and cache_obj.get("source_mtime_ns") == sct_gep_fp.stat().st_mtime_ns
                and isinstance(cache_obj.get("df"), pd.DataFrame)
            ):
                cached_df = cache_obj["df"]
        except Exception as e:
            logger.warning(f"Failed to read SCT query cache {cache_fp}: {e}")

    cached_index = set(cached_df.index.tolist())
    missing_ids = [cid for cid in selected_cell_ids if cid not in cached_index]
    if missing_ids:
        sct_loader = ReadH5AD(sct_gep_fp, backed="r")
        fetched_df = sct_loader.get_df(obs_names=missing_ids)
        if not fetched_df.empty:
            cached_df = pd.concat([cached_df, fetched_df], axis=0)
            cached_df = cached_df[~cached_df.index.duplicated(keep="last")]

            if cache_sct_query_results:
                try:
                    cache_fp.parent.mkdir(parents=True, exist_ok=True)
                    pd.to_pickle(
                        {
                            "source_path": str(sct_gep_fp),
                            "source_mtime_ns": sct_gep_fp.stat().st_mtime_ns,
                            "df": cached_df,
                        },
                        cache_fp,
                    )
                except Exception as e:
                    logger.warning(f"Failed to write SCT query cache {cache_fp}: {e}")

    present_ids = [cid for cid in selected_cell_ids if cid in set(cached_df.index.tolist())]
    if cache_sct_query_results:
        sct_geps_df_raw = cached_df.loc[present_ids, :]
    else:
        sct_loader = ReadH5AD(sct_gep_fp, backed="r")
        try:
            sct_geps_df_raw = sct_loader.get_df(obs_names=present_ids)
        finally:
            sct_loader.close()

    if sct_geps_df_raw.empty:
        return pd.DataFrame(columns=target_gene_list)

    return _reindex_expression_to_gene_list(sct_geps_df_raw, target_gene_list)


def _filter_sample2cell_mapping(
    sample2cell_fp: Path,
    raw_bulk_sample_ids: list[str],
) -> pd.DataFrame:
    """Filter one sample-to-cell mapping file to the requested bulk samples."""
    sample2cell_df_all = pd.read_csv(sample2cell_fp, index_col=0)
    if sample2cell_df_all.empty:
        return pd.DataFrame(columns=["cell_type", "selected_cell_id"])

    filtered_mapping_df = sample2cell_df_all.loc[
        sample2cell_df_all.index.astype(str).isin(raw_bulk_sample_ids),
        ["cell_type", "selected_cell_id"],
    ].copy()
    filtered_mapping_df["selected_cell_id"] = filtered_mapping_df["selected_cell_id"].astype(str)
    filtered_mapping_df = (
        filtered_mapping_df
        .reset_index()
        .drop_duplicates(subset=["index", "cell_type"], keep="first")
        .rename(columns={"index": "sample_id"})
        .set_index("sample_id")
    )
    return filtered_mapping_df


def _materialize_true_sct_gep_targets(
    raw_bulk_sample_ids: list[str],
    target_gene_list: list[str],
    cell_types: list[str],
    filtered_mapping_df: pd.DataFrame,
    aligned_sct_geps_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense matched targets from a filtered mapping and aligned SCT rows."""
    n_samples = len(raw_bulk_sample_ids)
    n_genes = len(target_gene_list)
    n_cell_types = len(cell_types)
    true_sct_gep = np.zeros((n_samples, n_genes, n_cell_types), dtype=np.float32)
    true_sct_gep_present_mask = np.zeros((n_samples, n_cell_types), dtype=bool)
    cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}

    if filtered_mapping_df.empty or aligned_sct_geps_df.empty:
        return true_sct_gep, true_sct_gep_present_mask

    for sample_idx, sample_id in enumerate(raw_bulk_sample_ids):
        if sample_id not in filtered_mapping_df.index:
            continue
        sample_rows = filtered_mapping_df.loc[[sample_id], :]
        for _, row in sample_rows.iterrows():
            cell_type = str(row["cell_type"])
            selected_cell_id = str(row["selected_cell_id"])
            cell_type_idx = cell_type_to_idx.get(cell_type)
            if cell_type_idx is None or selected_cell_id not in aligned_sct_geps_df.index:
                continue
            true_sct_gep[sample_idx, :, cell_type_idx] = aligned_sct_geps_df.loc[selected_cell_id, :].to_numpy(
                dtype=np.float32,
                copy=False,
            )
            true_sct_gep_present_mask[sample_idx, cell_type_idx] = True

    return true_sct_gep, true_sct_gep_present_mask


def _write_true_sct_gep_targets_into(
    raw_bulk_sample_ids: list[str],
    row_positions: np.ndarray,
    cell_types: list[str],
    filtered_mapping_df: pd.DataFrame,
    aligned_sct_geps_df: pd.DataFrame,
    true_sct_gep_dest: np.ndarray,
    true_sct_gep_present_mask_dest: np.ndarray,
) -> None:
    """Write matched targets directly into destination arrays without a large temporary tensor."""
    if filtered_mapping_df.empty or aligned_sct_geps_df.empty:
        return

    cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
    for sample_idx, sample_id in enumerate(raw_bulk_sample_ids):
        if sample_id not in filtered_mapping_df.index:
            continue

        dataset_row_idx = int(row_positions[sample_idx])
        sample_rows = filtered_mapping_df.loc[[sample_id], :]
        for _, row in sample_rows.iterrows():
            cell_type = str(row["cell_type"])
            selected_cell_id = str(row["selected_cell_id"])
            cell_type_idx = cell_type_to_idx.get(cell_type)
            if cell_type_idx is None or selected_cell_id not in aligned_sct_geps_df.index:
                continue
            true_sct_gep_dest[dataset_row_idx, :, cell_type_idx] = aligned_sct_geps_df.loc[selected_cell_id, :].to_numpy(
                dtype=np.float32,
                copy=False,
            )
            true_sct_gep_present_mask_dest[dataset_row_idx, cell_type_idx] = True


def build_matched_sct_gep_training_targets(
    sct_gep_dataset_file_path: Union[str, Path],
    sample2cell_id_file_path: Union[str, Path],
    bulk_sample_ids: list[str],
    target_gene_list: list[str],
    cell_types: list[str],
    cache_sct_query_results: bool = True,
    sct_query_cache_file_path: Optional[Union[str, Path]] = None,
    sample_id_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """Build dense matched sctGEP targets aligned to bulk sample order."""
    sct_gep_fp = Path(sct_gep_dataset_file_path)
    sample2cell_fp = Path(sample2cell_id_file_path)
    raw_bulk_sample_ids = [str(sample_id) for sample_id in bulk_sample_ids]
    processed_bulk_sample_ids = _namespace_sample_ids(raw_bulk_sample_ids, sample_id_namespace)

    if not sct_gep_fp.exists():
        raise FileNotFoundError(f"SCT GEP dataset file not found: {sct_gep_fp}")
    if not sample2cell_fp.exists():
        raise FileNotFoundError(f"Sample to cell ID mapping file not found: {sample2cell_fp}")

    filtered_mapping_df = _filter_sample2cell_mapping(sample2cell_fp, raw_bulk_sample_ids)
    if filtered_mapping_df.empty:
        return {
            "sample_ids": processed_bulk_sample_ids,
            "gene_list": target_gene_list,
            "cell_types": cell_types,
            "true_sct_gep": np.zeros((len(processed_bulk_sample_ids), len(target_gene_list), len(cell_types)), dtype=np.float32),
            "true_sct_gep_present_mask": np.zeros((len(processed_bulk_sample_ids), len(cell_types)), dtype=bool),
            "selected_sample2cell_id": pd.DataFrame(columns=["cell_type", "selected_cell_id"]),
            "aligned_sct_geps_df": pd.DataFrame(columns=target_gene_list),
        }

    unique_sct_cell_ids = list(dict.fromkeys(filtered_mapping_df["selected_cell_id"].tolist()))
    aligned_sct_geps_df = _load_cached_aligned_sct_geps(
        sct_gep_fp=sct_gep_fp,
        selected_cell_ids=unique_sct_cell_ids,
        target_gene_list=target_gene_list,
        cache_sct_query_results=cache_sct_query_results,
        sct_query_cache_file_path=sct_query_cache_file_path,
    )

    true_sct_gep, true_sct_gep_present_mask = _materialize_true_sct_gep_targets(
        raw_bulk_sample_ids=raw_bulk_sample_ids,
        target_gene_list=target_gene_list,
        cell_types=cell_types,
        filtered_mapping_df=filtered_mapping_df,
        aligned_sct_geps_df=aligned_sct_geps_df,
    )

    return {
        "sample_ids": processed_bulk_sample_ids,
        "gene_list": target_gene_list,
        "cell_types": cell_types,
        "true_sct_gep": true_sct_gep,
        "true_sct_gep_present_mask": true_sct_gep_present_mask,
        "selected_sample2cell_id": filtered_mapping_df,
        "aligned_sct_geps_df": aligned_sct_geps_df,
    }


def _load_dataset_frames_from_path(
    path_like: Union[str, Path],
    *,
    namespace: Optional[str] = None,
    target_gene_list: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one source file into aligned expression and label DataFrames."""
    path_str = str(path_like)
    log_message(f"Reading data from {path_str}")

    if path_str.endswith(".h5ad"):
        h5ad_obj = ReadH5AD(path_str, backed="r" if target_gene_list is not None else None)
        try:
            # When we subset genes here, convert_to_tpm=True renormalizes the remaining
            # genes to TPM before later preprocessing steps recover log2(TPM + 1).
            gep_data_df = h5ad_obj.get_df(
                convert_to_tpm=True,
                var_names=list(target_gene_list) if target_gene_list is not None else None,
            )
            if gep_data_df.values.dtype != np.float32:
                gep_data_df = gep_data_df.astype(np.float32)
            cell_prop_df = h5ad_obj.get_cell_fraction()
        finally:
            h5ad_obj.close()
        if cell_prop_df is None:
            cell_prop_df = pd.DataFrame(index=gep_data_df.index)
    elif path_str.endswith(".csv"):
        if target_gene_list is not None:
            header_df = pd.read_csv(path_str, nrows=0)
            index_col_name = header_df.columns.tolist()[0]
            available_genes = set(str(col) for col in header_df.columns.tolist()[1:])
            selected_genes = [str(gene) for gene in target_gene_list if str(gene) in available_genes]
            gep_data_df = pd.read_csv(path_str, index_col=0, usecols=[index_col_name] + selected_genes)
            gep_data_df = gep_data_df.loc[:, selected_genes]
        else:
            gep_data_df = pd.read_csv(path_str, index_col=0)
        cell_prop_df = pd.DataFrame(index=gep_data_df.index)
    else:
        raise ValueError(f"Unrecognized file format: {path_str}")

    if namespace is not None:
        namespaced_index = pd.Index(
            _namespace_sample_ids(gep_data_df.index.astype(str).tolist(), namespace),
            dtype=object,
        )
        gep_data_df.index = namespaced_index
        cell_prop_df.index = namespaced_index

    log_message(f"Loaded data shape: {gep_data_df.shape}")
    return gep_data_df, cell_prop_df

# =============================================================================
# 1) Cache Manager
# =============================================================================

class GEPCacheManager:
    """
    Responsible only for cache file paths, read/write arrays, and metadata I/O.

    This class centralizes cache naming logic so we never mutate path attributes
    dynamically in different methods (a common source of subtle bugs).
    """

    def __init__(self, cache_dir: Path, compress: bool = False):
        self.cache_dir = cache_dir
        self.compress = compress
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define canonical cache file paths once
        self.data_path = self.cache_dir / ("data.npz" if self.compress else "data.npy")
        self.labels_path = self.cache_dir / ("labels.npz" if self.compress else "labels.npy")
        self.true_sct_gep_path = self.cache_dir / ("true_sct_gep.npz" if self.compress else "true_sct_gep.npy")
        self.true_sct_gep_present_mask_path = self.cache_dir / (
            "true_sct_gep_present_mask.npz" if self.compress else "true_sct_gep_present_mask.npy"
        )
        self.common_gene_list_path = self.cache_dir / "common_gene_list.txt"
        self.gene_list_path = self.cache_dir / "gene_list.txt"
        self.cell_types_path = self.cache_dir / "cell_types.txt"
        self.sample_ids_path = self.cache_dir / "sample_ids.txt"
        self.metadata_path = self.cache_dir / "metadata.json"

    # -------------------------------------------------------------------------
    # Text I/O helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def save_list_txt(data_list: Sequence[str], file_path: Path) -> None:
        """Save a list of strings to text file (one item per line)."""
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(f"{item}\n")

    @staticmethod
    def load_list_txt(file_path: Path) -> List[str]:
        """Load a text file as a list of lines."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()

    # -------------------------------------------------------------------------
    # Cache existence checks
    # -------------------------------------------------------------------------
    def has_required_cache(self) -> bool:
        """
        Check if minimum required cache files exist.

        Required:
        - processed data array
        - gene list
        - sample IDs

        Labels/cell types may not exist for unlabeled/eval datasets.
        """
        required = [self.data_path, self.gene_list_path, self.sample_ids_path]
        return all(p.exists() for p in required)

    # -------------------------------------------------------------------------
    # Array save/load
    # -------------------------------------------------------------------------
    def save_arrays(self, data_array: np.ndarray, labels_array: Optional[np.ndarray]) -> None:
        """
        Save data and optional labels arrays.

        - compress=False: .npy (supports memmap on load)
        - compress=True: .npz (smaller but no true memmap for inner arrays)
        """
        if self.compress:
            np.savez_compressed(self.data_path, data=data_array)
            if labels_array is not None:
                np.savez_compressed(self.labels_path, labels=labels_array)
        else:
            np.save(self.data_path, data_array)
            if labels_array is not None:
                np.save(self.labels_path, labels_array)

    def save_optional_array(self, array_path: Path, array: Optional[np.ndarray], key: str) -> None:
        """Save an optional array using the same compression policy as the main cache."""
        if array is None:
            return
        if self.compress:
            np.savez_compressed(array_path, **{key: array})
        else:
            np.save(array_path, array)

    def load_data(self, use_memmap: bool) -> np.ndarray:
        """
        Load data array.

        Important behavior:
        - For .npz, np.load returns NpzFile; inner array is not true memmap.
        - For .npy + use_memmap=True, return memory-mapped ndarray.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        if self.compress:
            arr = np.load(self.data_path, allow_pickle=False)["data"]
            return arr.astype(np.float32, copy=False)

        if use_memmap:
            return np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        return np.load(self.data_path, allow_pickle=False).astype(np.float32, copy=False)

    def load_labels(self, use_memmap: bool) -> np.ndarray:
        """Load labels array. Returns empty array if labels cache is absent."""
        if not self.labels_path.exists():
            return np.array([], dtype=np.float32)

        if self.compress:
            arr = np.load(self.labels_path, allow_pickle=False)["labels"]
            return arr.astype(np.float32, copy=False)

        if use_memmap:
            return np.load(self.labels_path, mmap_mode="r", allow_pickle=False)
        return np.load(self.labels_path, allow_pickle=False).astype(np.float32, copy=False)

    def load_optional_array(
        self,
        array_path: Path,
        key: str,
        use_memmap: bool,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Load an optional cached array or return an empty array when absent."""
        if not array_path.exists():
            return np.array([], dtype=dtype or np.float32)

        if self.compress:
            arr = np.load(array_path, allow_pickle=False)[key]
        elif use_memmap:
            arr = np.load(array_path, mmap_mode="r", allow_pickle=False)
        else:
            arr = np.load(array_path, allow_pickle=False)

        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr

    # -------------------------------------------------------------------------
    # Metadata save/load
    # -------------------------------------------------------------------------
    def save_metadata(
        self,
        gene_list: Sequence[str],
        sample_ids: Sequence[str],
        cell_types: Sequence[str],
        scaling_value: float,
        apply_scaling: bool,
    ) -> None:
        """Save text metadata + JSON metadata."""
        self.save_list_txt(gene_list, self.gene_list_path)
        self.save_list_txt(sample_ids, self.sample_ids_path)
        self.save_list_txt(cell_types, self.cell_types_path)

        metadata = {
            "n_samples": len(sample_ids),
            "n_genes": len(gene_list),
            "n_cell_types": len(cell_types),
            "scaling_value": scaling_value,
            "apply_scaling": apply_scaling,
            "compress": self.compress,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> Dict[str, Any]:
        """Load JSON metadata. Returns empty dict if not found."""
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_text_metadata(self) -> Tuple[List[str], List[str], List[str]]:
        """Load gene list, sample IDs, and optional cell types from cache."""
        gene_list = self.load_list_txt(self.gene_list_path)
        sample_ids = self.load_list_txt(self.sample_ids_path)
        if self.cell_types_path.exists():
            cell_types = self.load_list_txt(self.cell_types_path)
        else:
            cell_types = []
        return gene_list, sample_ids, cell_types


# =============================================================================
# 2) Preprocessor
# =============================================================================

class GEPPreprocessor:
    """
    Stateless preprocessor (except chunk_size) that handles data preparation.

    Responsibilities:
    - load and merge source files
    - optional gene list filtering
    - optional low variance gene removal
    - chunked transformation with memory-aware behavior
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        max_parallel_source_file_loads: int = _DEFAULT_MAX_PARALLEL_SOURCE_FILE_LOADS,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        self.chunk_size = chunk_size
        self.max_parallel_source_file_loads = max(1, int(max_parallel_source_file_loads))
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None

    def run(
        self,
        file_paths: Sequence[Union[str, Path]],
        gene_list_file: Optional[Union[str, Path]] = None,
        common_gene_list_path: Optional[Union[str, Path]] = None,
        remove_low_var_genes: bool = False,
        min_var: float = 1.0,
        cell_cell2ave_exp_file_path: Optional[Union[str, Path]] = None,
        sample_id_namespace_by_path: Optional[Dict[str, str]] = None,
        source_group_key_by_path: Optional[Dict[str, str]] = None,
        max_parallel_source_file_loads: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end preprocessing pipeline.
        Returns:
            dict containing final arrays and metadata
        """
        if max_parallel_source_file_loads is not None:
            self.max_parallel_source_file_loads = max(1, int(max_parallel_source_file_loads))

        log_message("Step 1: Discovering common genes...")
        initial_target_gene_list = _load_or_create_cached_common_gene_list(
            file_paths=file_paths,
            gene_list_file=gene_list_file,
            cache_file_path=common_gene_list_path,
        )

        log_message("Step 2: Loading groups and staging raw intermediates...")
        self._prepare_temp_dir()
        try:
            staged = self._stage_group_raw_intermediates(
                file_paths=file_paths,
                target_gene_list=initial_target_gene_list,
                sample_id_namespace_by_path=sample_id_namespace_by_path,
                source_group_key_by_path=source_group_key_by_path,
            )

            final_gene_list = initial_target_gene_list
            if remove_low_var_genes:
                log_message("Step 3: Applying low-variance gene filtering...")
                ref_path = Path(cell_cell2ave_exp_file_path) if cell_cell2ave_exp_file_path else None
                final_gene_list = self._apply_low_var_gene_removal(
                    gene_list=initial_target_gene_list,
                    gene_sums=staged["gene_sums"],
                    gene_squared_sums=staged["gene_squared_sums"],
                    n_samples=int(staged["n_samples"]),
                    min_var=min_var,
                    cell_cell2ave_exp_file_path=ref_path,
                )
            else:
                log_message("Step 3: Skipping low-variance gene filtering.")

            log_message("Step 4: Applying transformations and assembling final arrays...")
            return self._build_final_arrays_from_staged_groups(
                group_records=staged["group_records"],
                final_gene_list=final_gene_list,
                initial_target_gene_list=initial_target_gene_list,
                final_cell_types=staged["final_cell_types"],
                expected_sample_ids=staged["expected_sample_ids"],
            )
        finally:
            self._cleanup_temp_dir()

    def _prepare_temp_dir(self) -> None:
        """Create a fresh temporary preprocessing directory."""
        if self.temp_dir is None:
            return
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_temp_dir(self) -> None:
        """Remove temporary preprocessing files after the run finishes."""
        if self.temp_dir is not None and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _build_load_jobs(
        self,
        file_paths: Sequence[Union[str, Path]],
        sample_id_namespace_by_path: Optional[Dict[str, str]] = None,
        source_group_key_by_path: Optional[Dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        """Build file-loading jobs with stable ordering and group keys."""
        load_jobs: list[dict[str, Any]] = []
        for position, path_like in enumerate(file_paths):
            resolved_path = str(Path(path_like).expanduser().resolve())
            namespace = None
            if sample_id_namespace_by_path:
                namespace = sample_id_namespace_by_path.get(resolved_path)
            source_group_key = resolved_path
            if source_group_key_by_path:
                source_group_key = source_group_key_by_path.get(resolved_path, resolved_path)
            load_jobs.append(
                {
                    "position": position,
                    "path_like": path_like,
                    "resolved_path": resolved_path,
                    "namespace": namespace,
                    "source_group_key": str(source_group_key),
                }
            )

        if not load_jobs:
            raise ValueError("No data loaded. Please check file_paths.")
        return load_jobs

    @staticmethod
    def _group_load_jobs(load_jobs: Sequence[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
        """Group jobs while preserving the first-seen group order."""
        grouped_jobs: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for job in load_jobs:
            grouped_jobs.setdefault(job["source_group_key"], []).append(job)
        return grouped_jobs

    @staticmethod
    def _intersect_ordered_lists(ordered_lists: Sequence[Sequence[str]]) -> list[str]:
        """Intersect multiple ordered lists while preserving the first list's order."""
        if not ordered_lists:
            return []
        common_items = set(ordered_lists[0])
        for values in ordered_lists[1:]:
            common_items &= set(values)
        return [item for item in ordered_lists[0] if item in common_items]

    def _stage_group_raw_intermediates(
        self,
        file_paths: Sequence[Union[str, Path]],
        target_gene_list: Sequence[str],
        sample_id_namespace_by_path: Optional[Dict[str, str]] = None,
        source_group_key_by_path: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Load each preprocessing group, compute global stats, and persist raw group arrays."""
        load_jobs = self._build_load_jobs(
            file_paths=file_paths,
            sample_id_namespace_by_path=sample_id_namespace_by_path,
            source_group_key_by_path=source_group_key_by_path,
        )
        grouped_jobs = self._group_load_jobs(load_jobs)
        if len(grouped_jobs) > 1:
            log_message(
                f"Loading {len(load_jobs)} source files across {len(grouped_jobs)} preprocessing groups..."
            )

        gene_sums = np.zeros(len(target_gene_list), dtype=np.float64)
        gene_squared_sums = np.zeros(len(target_gene_list), dtype=np.float64)
        total_samples = 0
        sample_ids_by_position: dict[int, list[str]] = {}
        cell_type_lists: list[list[str]] = []
        group_records: list[dict[str, Any]] = []

        for group_idx, group_jobs in enumerate(grouped_jobs.values(), start=1):
            max_workers = min(self.max_parallel_source_file_loads, len(group_jobs))
            if max_workers > 1:
                log_message(
                    f"Loading preprocessing group {group_idx}/{len(grouped_jobs)} "
                    f"with {len(group_jobs)} files and {max_workers} worker threads..."
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    loaded_group = list(
                        executor.map(
                            lambda job: (
                                job["position"],
                                _load_dataset_frames_from_path(
                                    job["path_like"],
                                    namespace=job["namespace"],
                                    target_gene_list=target_gene_list,
                                ),
                            ),
                            group_jobs,
                        )
                    )
            else:
                loaded_group = [
                    (
                        group_jobs[0]["position"],
                        _load_dataset_frames_from_path(
                            group_jobs[0]["path_like"],
                            namespace=group_jobs[0]["namespace"],
                            target_gene_list=target_gene_list,
                        ),
                    )
                ]

            group_data_dfs: List[pd.DataFrame] = []
            group_cell_prop_dfs: List[pd.DataFrame] = []
            for position, (gep_data_df, cell_prop_df) in sorted(loaded_group, key=lambda item: item[0]):
                sample_ids_by_position[position] = gep_data_df.index.astype(str).tolist()
                group_data_dfs.append(gep_data_df)
                group_cell_prop_dfs.append(cell_prop_df)

            log_message(f"Merging preprocessing group {group_idx}/{len(grouped_jobs)}...")
            group_gep_data_df = pd.concat(group_data_dfs, axis=0, join="inner")
            group_cell_prop_df = pd.concat(group_cell_prop_dfs, axis=0, join="inner")
            group_gep_data_df = group_gep_data_df.loc[:, list(target_gene_list)]
            if not group_cell_prop_df.empty:
                group_cell_prop_df = group_cell_prop_df.loc[group_gep_data_df.index]

            group_values = group_gep_data_df.values.astype(np.float32, copy=False)
            gene_sums += group_values.sum(axis=0, dtype=np.float64)
            gene_squared_sums += np.square(group_values, dtype=np.float64).sum(axis=0, dtype=np.float64)
            total_samples += int(group_values.shape[0])

            raw_data_path = (self.temp_dir / f"group_{group_idx:04d}_raw_data.npy") if self.temp_dir else None
            if raw_data_path is not None:
                np.save(raw_data_path, group_values)

            group_record: dict[str, Any] = {
                "raw_data_path": raw_data_path,
                "sample_ids": group_gep_data_df.index.astype(str).tolist(),
                "cell_types": group_cell_prop_df.columns.astype(str).tolist(),
                "n_samples": int(group_values.shape[0]),
            }
            if not group_cell_prop_df.empty:
                raw_labels_path = (self.temp_dir / f"group_{group_idx:04d}_raw_labels.npy") if self.temp_dir else None
                group_labels = group_cell_prop_df.values.astype(np.float32, copy=False)
                if raw_labels_path is not None:
                    np.save(raw_labels_path, group_labels)
                group_record["raw_labels_path"] = raw_labels_path
            else:
                group_record["raw_labels_path"] = None
            group_records.append(group_record)
            cell_type_lists.append(group_record["cell_types"])

            del loaded_group, group_data_dfs, group_cell_prop_dfs, group_gep_data_df, group_cell_prop_df, group_values
            gc.collect()

        expected_sample_ids: list[str] = []
        for position in range(len(load_jobs)):
            expected_sample_ids.extend(sample_ids_by_position.get(position, []))

        return {
            "group_records": group_records,
            "gene_sums": gene_sums,
            "gene_squared_sums": gene_squared_sums,
            "n_samples": total_samples,
            "expected_sample_ids": expected_sample_ids,
            "final_cell_types": self._intersect_ordered_lists(cell_type_lists),
        }

    @staticmethod
    def _apply_low_var_gene_removal(
        gene_list: Sequence[str],
        gene_sums: np.ndarray,
        gene_squared_sums: np.ndarray,
        n_samples: int,
        min_var: float,
        cell_cell2ave_exp_file_path: Optional[Path],
    ) -> list[str]:
        """
        Remove low-variance genes from global sufficient statistics and optionally intersect with reference genes.
        """
        n_genes_before = len(gene_list)

        if n_samples <= 1:
            gene_variances = np.full(n_genes_before, np.nan, dtype=np.float64)
        else:
            numerator = gene_squared_sums - (np.square(gene_sums) / float(n_samples))
            gene_variances = numerator / float(n_samples - 1)
            gene_variances = np.maximum(gene_variances, 0.0)

        genes_to_keep = gene_variances > float(min_var)
        final_gene_list = [str(gene_list[idx]) for idx, keep in enumerate(genes_to_keep) if keep]

        if cell_cell2ave_exp_file_path is not None:
            try:
                cell_ave_exp_df = pd.read_csv(cell_cell2ave_exp_file_path, index_col=0)
                genes_in_ref = set(str(gene) for gene in cell_ave_exp_df.index.tolist())
                final_gene_list = [gene for gene in final_gene_list if gene in genes_in_ref]
                del cell_ave_exp_df
            except Exception as e:
                logger.error(f"Error processing reference file: {e}")

        n_genes_after = len(final_gene_list)
        if n_genes_after == 0:
            raise ValueError("Low-variance filtering removed all genes.")
        log_message(
            f"Gene filtering: {n_genes_before} → {n_genes_after} genes "
            f"(removed {n_genes_before - n_genes_after})"
        )
        return final_gene_list

    def _build_final_arrays_from_staged_groups(
        self,
        *,
        group_records: Sequence[dict[str, Any]],
        final_gene_list: Sequence[str],
        initial_target_gene_list: Sequence[str],
        final_cell_types: Sequence[str],
        expected_sample_ids: Sequence[str],
    ) -> Dict[str, Any]:
        """Transform staged raw group arrays and assemble one final dataset array."""
        gene_index_by_name = {str(gene): idx for idx, gene in enumerate(initial_target_gene_list)}
        final_gene_indices = [gene_index_by_name[str(gene)] for gene in final_gene_list]
        total_samples = sum(int(record["n_samples"]) for record in group_records)

        data_array = np.empty((total_samples, len(final_gene_list)), dtype=np.float32)
        labels_array: Optional[np.ndarray] = None
        if final_cell_types:
            labels_array = np.empty((total_samples, len(final_cell_types)), dtype=np.float32)

        aggregated_sample_ids: list[str] = []
        cursor = 0
        for record in group_records:
            raw_data = np.load(record["raw_data_path"], mmap_mode="r") if record["raw_data_path"] else None
            if raw_data is None:
                raise ValueError("Missing staged raw data for preprocessing group.")
            raw_subset = np.asarray(raw_data[:, final_gene_indices], dtype=np.float32)
            transformed_group_df = self._apply_transformation_chunked(
                pd.DataFrame(raw_subset, index=record["sample_ids"], columns=list(final_gene_list)),
                chunk_size=self.chunk_size,
            )
            group_values = transformed_group_df.values.astype(np.float32, copy=False)
            n_group_samples = int(record["n_samples"])
            data_array[cursor:cursor + n_group_samples] = group_values

            if labels_array is not None:
                raw_labels = np.load(record["raw_labels_path"], mmap_mode="r") if record["raw_labels_path"] else None
                if raw_labels is None:
                    raise ValueError("Expected staged label array for a labeled preprocessing group.")
                local_cell_types = list(record["cell_types"])
                col_indices = [local_cell_types.index(cell_type) for cell_type in final_cell_types]
                labels_array[cursor:cursor + n_group_samples] = np.asarray(raw_labels[:, col_indices], dtype=np.float32)
                del raw_labels

            aggregated_sample_ids.extend(record["sample_ids"])
            cursor += n_group_samples

            del raw_data, raw_subset, transformed_group_df, group_values
            gc.collect()

        if expected_sample_ids and len(set(aggregated_sample_ids)) == len(aggregated_sample_ids):
            sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(aggregated_sample_ids)}
            reorder_idx = np.asarray([sample_id_to_idx[sample_id] for sample_id in expected_sample_ids], dtype=np.int64)
            data_array = data_array[reorder_idx]
            if labels_array is not None:
                labels_array = labels_array[reorder_idx]
            final_sample_ids = list(expected_sample_ids)
        else:
            final_sample_ids = aggregated_sample_ids

        log_message(f"Final processed array shape: {data_array.shape}")
        return {
            "data_array": data_array,
            "labels_array": labels_array,
            "gene_list": list(final_gene_list),
            "sample_ids": final_sample_ids,
            "cell_types": list(final_cell_types),
        }

    @staticmethod
    def _apply_transformation_chunked(
        gep_data_df: pd.DataFrame,
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """
        Apply transformation in chunks to reduce peak RAM usage.

        Returns:
            float32 DataFrame with same index/columns
        """
        log_message(f"Applying transformation to {gep_data_df.shape} data...")
        log_message(f"Chunk size: {chunk_size}")

        n_samples, n_genes = gep_data_df.shape
        if n_samples <= chunk_size:
            transformed_df = non_log2log_cpm(gep_data_df, transpose=False)
            return transformed_df.astype(np.float32, copy=False)

        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        index = gep_data_df.index
        columns = gep_data_df.columns

        result_array = np.empty((n_samples, n_genes), dtype=np.float32)

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_samples)

            chunk_df = gep_data_df.iloc[start_idx:end_idx]
            try:
                transformed_chunk = non_log2log_cpm(chunk_df, transpose=False)
                result_array[start_idx:end_idx] = transformed_chunk.values.astype(np.float32, copy=False)
            except MemoryError:
                logger.error(f"Memory error in chunk {i + 1}/{n_chunks}. Reduce chunk_size.")
                raise
            finally:
                del chunk_df
                if "transformed_chunk" in locals():
                    del transformed_chunk

            if (i + 1) % 5 == 0:
                gc.collect()

        result = pd.DataFrame(result_array, index=index, columns=columns)
        del result_array
        gc.collect()

        log_message(f"Transformation complete: {result.shape}, dtype={result.values.dtype}")
        return result


# =============================================================================
# 3) Dataset
# =============================================================================

class GEPDataset(Dataset):
    """
    Memory-efficient Dataset class for GEP data with lazy loading and cache support.

    Key improvements:
    - Uses memory-mapped files for large datasets when possible (.npy)
    - Lazy loading: only loads data when accessed
    - Chunked preprocessing through GEPPreprocessor
    - Optional data compression (.npz)
    - Centralized cache strategy through GEPCacheManager
    """

    def __init__(self, config: GEPDatasetConfig):
        """
        Args:
            config:
                Typed configuration object. Replaces long constructor signature.
        """
        self.config = config

        # Scaling logic:
        # - True  => use config.scaling_factor
        # - float => use that float
        # - False => no scaling
        self.scaling_value = (
            config.scaling_factor if config.scaling_by_constant is True
            else config.scaling_by_constant if isinstance(config.scaling_by_constant, (float, int))
            else 1.0
        )
        self.apply_scaling = bool(config.scaling_by_constant)

        # Important: compressed npz does not provide true memmap arrays.
        self.compress = bool(config.compress)
        self.use_memmap = bool(config.use_memmap and not self.compress)
        if config.use_memmap and self.compress:
            warnings.warn(
                "compress=True disables true memmap behavior for arrays inside .npz. "
                "Proceeding with in-memory loading for compressed cache."
            )

        self.chunk_size = int(config.chunk_size)

        if config.processed_data_dir is None:
            raise ValueError("processed_data_dir must not be None in this implementation.")

        self.processed_data_dir = Path(config.processed_data_dir)
        self.cache = GEPCacheManager(self.processed_data_dir, compress=self.compress)

        # Lazy-loaded arrays
        self._data: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._true_sct_gep: Optional[np.ndarray] = None
        self._true_sct_gep_present_mask: Optional[np.ndarray] = None

        # Metadata
        self.gene_list: List[str] = []
        self.cell_types: List[str] = []
        self.sample_ids: List[str] = []
        self._n_samples = 0
        self._n_genes = 0
        self._n_cell_types = 0

        has_cache = self.cache.has_required_cache()

        # Try cache first unless force_reprocess=True
        if (not config.force_reprocess) and has_cache:
            if self._load_from_cache_metadata():
                log_message(f"Successfully loaded cache metadata from {self.processed_data_dir}")
            else:
                log_message("Cache metadata load failed. Reprocessing from scratch...")
                self._preprocess_and_cache()
                self._load_from_cache_metadata()
        else:
            if config.force_reprocess and has_cache:
                log_message(
                    f"force_reprocess=True, rebuilding processed dataset cache at {self.processed_data_dir}"
                )
            log_message("Preprocessing data from scratch...")
            self._preprocess_and_cache()
            self._load_from_cache_metadata()

    # -------------------------------------------------------------------------
    # Lazy-loading properties
    # -------------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """Lazy loading of data array."""
        if self._data is None:
            self._data = self.cache.load_data(use_memmap=self.use_memmap)
            log_message(f"Loaded data: shape={self._data.shape}, dtype={self._data.dtype}")
            if self._data.dtype != np.float32:
                logger.warning(
                    f"Data dtype is {self._data.dtype}, converting to float32 for efficiency."
                )
                self._data = self._data.astype(np.float32, copy=False)
        return self._data

    @property
    def labels(self) -> np.ndarray:
        """Lazy loading of labels array."""
        if self._labels is None:
            self._labels = self.cache.load_labels(use_memmap=self.use_memmap)
            log_message(f"Loaded labels: shape={self._labels.shape}, dtype={self._labels.dtype}")
            if self._labels.size > 0 and self._labels.dtype != np.float32:
                self._labels = self._labels.astype(np.float32, copy=False)
        return self._labels

    @property
    def true_sct_gep(self) -> np.ndarray:
        """Lazy loading of matched true sctGEP targets."""
        if self._true_sct_gep is None:
            self._true_sct_gep = self.cache.load_optional_array(
                array_path=self.cache.true_sct_gep_path,
                key="true_sct_gep",
                use_memmap=self.use_memmap,
                dtype=np.float32,
            )
        return self._true_sct_gep

    @property
    def true_sct_gep_present_mask(self) -> np.ndarray:
        """Lazy loading of matched true-sctGEP presence masks."""
        if self._true_sct_gep_present_mask is None:
            self._true_sct_gep_present_mask = self.cache.load_optional_array(
                array_path=self.cache.true_sct_gep_present_mask_path,
                key="true_sct_gep_present_mask",
                use_memmap=self.use_memmap,
                dtype=bool,
            )
        return self._true_sct_gep_present_mask

    # -------------------------------------------------------------------------
    # Internal cache metadata loader
    # -------------------------------------------------------------------------
    def _load_from_cache_metadata(self) -> bool:
        """Load metadata and small text files without loading large arrays."""
        try:
            metadata = self.cache.load_metadata()
            self.gene_list, self.sample_ids, self.cell_types = self.cache.load_text_metadata()

            self._n_samples = int(metadata.get("n_samples", len(self.sample_ids)))
            self._n_genes = int(metadata.get("n_genes", len(self.gene_list)))
            self._n_cell_types = int(metadata.get("n_cell_types", len(self.cell_types)))

            log_message(
                f"Cache metadata loaded: {self._n_samples} samples, "
                f"{self._n_genes} genes, {self._n_cell_types} cell types"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            return False

    def _build_sample_id_namespace_by_path(self) -> Dict[str, str]:
        """Build stable per-file sample-ID namespaces for training-time merged datasets."""
        training_target_sets = dict(self.config.training_target_sets or {})
        if not training_target_sets:
            return {}

        namespace_by_path: Dict[str, str] = {}
        for target_set_name, target_cfg in training_target_sets.items():
            resolved_path = str(Path(target_cfg.training_set_file_path).expanduser().resolve())
            namespace_by_path[resolved_path] = str(target_set_name)

        for path_like in self.config.file_paths:
            resolved_path = str(Path(path_like).expanduser().resolve())
            if resolved_path not in namespace_by_path:
                namespace_by_path[resolved_path] = Path(path_like).stem

        return namespace_by_path

    def _build_source_group_key_by_path(self) -> Dict[str, str]:
        """Group bulk source files by shared matched-target SCT reference."""
        source_group_key_by_path: Dict[str, str] = {}
        for target_cfg in dict(self.config.training_target_sets or {}).values():
            resolved_bulk_path = str(Path(target_cfg.training_set_file_path).expanduser().resolve())
            resolved_sct_gep_path = str(Path(target_cfg.training_sct_gep_file_path).expanduser().resolve())
            source_group_key_by_path[resolved_bulk_path] = resolved_sct_gep_path

        for path_like in self.config.file_paths:
            resolved_path = str(Path(path_like).expanduser().resolve())
            source_group_key_by_path.setdefault(resolved_path, f"source::{resolved_path}")

        return source_group_key_by_path

    # -------------------------------------------------------------------------
    # Preprocess + cache
    # -------------------------------------------------------------------------
    def _preprocess_and_cache(self) -> None:
        """Run preprocessing pipeline and save outputs to cache."""
        preprocessor = GEPPreprocessor(
            chunk_size=self.chunk_size,
            max_parallel_source_file_loads=self.config.max_parallel_source_file_loads,
            temp_dir=self.processed_data_dir / "_tmp_preprocess",
        )
        sample_id_namespace_by_path = self._build_sample_id_namespace_by_path()
        source_group_key_by_path = self._build_source_group_key_by_path()

        processed = preprocessor.run(
            file_paths=self.config.file_paths,
            gene_list_file=self.config.gene_list_file,
            common_gene_list_path=self.cache.common_gene_list_path,
            remove_low_var_genes=self.config.remove_low_var_genes,
            min_var=self.config.min_var,
            cell_cell2ave_exp_file_path=self.config.cell_cell2ave_exp_file_path,
            sample_id_namespace_by_path=sample_id_namespace_by_path,
            source_group_key_by_path=source_group_key_by_path,
            max_parallel_source_file_loads=self.config.max_parallel_source_file_loads,
        )
        data_array = processed["data_array"]
        labels_array = processed["labels_array"]
        gene_list = processed["gene_list"]
        sample_ids = processed["sample_ids"]
        cell_types = processed["cell_types"]

        # Arrays are already float32 at this point; only apply optional scaling.
        log_message("Applying final array scaling...")
        if self.apply_scaling:
            data_array = data_array / float(self.scaling_value)

        true_sct_gep_array: Optional[np.ndarray] = None
        true_sct_gep_present_mask_array: Optional[np.ndarray] = None
        sct_target_arrays_pre_saved = False
        temp_optional_cache_paths: list[Path] = []
        if self.config.training_target_sets:
            if labels_array is None or len(cell_types) == 0:
                raise ValueError(
                    "training_target_sets requires cell-fraction labels so matched "
                    "sctGEP targets can align with training cell types."
                )
            (
                true_sct_gep_array,
                true_sct_gep_present_mask_array,
                sct_target_arrays_pre_saved,
                temp_optional_cache_paths,
            ) = self._build_true_sct_gep_targets(
                sample_ids=[str(sample_id) for sample_id in sample_ids],
                gene_list=[str(gene) for gene in gene_list],
                cell_types=[str(cell_type) for cell_type in cell_types],
            )
            if self.apply_scaling and true_sct_gep_array is not None:
                if sct_target_arrays_pre_saved:
                    true_sct_gep_array[:] = true_sct_gep_array[:] / float(self.scaling_value)
                    if hasattr(true_sct_gep_array, "flush"):
                        true_sct_gep_array.flush()
                else:
                    true_sct_gep_array = true_sct_gep_array / float(self.scaling_value)

        # Save arrays + metadata
        self.cache.save_arrays(data_array, labels_array)
        if not sct_target_arrays_pre_saved:
            self.cache.save_optional_array(
                array_path=self.cache.true_sct_gep_path,
                array=true_sct_gep_array,
                key="true_sct_gep",
            )
            self.cache.save_optional_array(
                array_path=self.cache.true_sct_gep_present_mask_path,
                array=true_sct_gep_present_mask_array,
                key="true_sct_gep_present_mask",
            )
        self.cache.save_metadata(
            gene_list=gene_list,
            sample_ids=sample_ids,
            cell_types=cell_types,
            scaling_value=float(self.scaling_value),
            apply_scaling=self.apply_scaling,
        )

        # Set in-memory metadata immediately
        self.gene_list = list(gene_list)
        self.sample_ids = list(sample_ids)
        self.cell_types = list(cell_types)

        self._n_samples = len(self.sample_ids)
        self._n_genes = len(self.gene_list)
        self._n_cell_types = len(self.cell_types)

        # Cleanup
        del processed, data_array, labels_array
        if true_sct_gep_array is not None:
            del true_sct_gep_array
        if true_sct_gep_present_mask_array is not None:
            del true_sct_gep_present_mask_array
        for temp_path in temp_optional_cache_paths:
            if temp_path.exists():
                temp_path.unlink()
        gc.collect()

        log_message(
            f"Preprocessing complete: {self._n_samples} samples, "
            f"{self._n_genes} genes"
        )

    def _build_true_sct_gep_targets(
        self,
        sample_ids: list[str],
        gene_list: list[str],
        cell_types: list[str],
    ) -> tuple[np.ndarray, np.ndarray, bool, list[Path]]:
        """Build dataset-aligned matched sctGEP targets for configured training bulk sets."""
        n_samples = len(sample_ids)
        n_genes = len(gene_list)
        n_cell_types = len(cell_types)
        temp_paths: list[Path] = []
        arrays_pre_saved = not self.compress
        if self.compress:
            true_sct_gep_path = self.processed_data_dir / "_tmp_true_sct_gep.npy"
            true_sct_gep_present_mask_path = self.processed_data_dir / "_tmp_true_sct_gep_present_mask.npy"
            temp_paths.extend([true_sct_gep_path, true_sct_gep_present_mask_path])
        else:
            true_sct_gep_path = self.cache.true_sct_gep_path
            true_sct_gep_present_mask_path = self.cache.true_sct_gep_present_mask_path

        true_sct_gep = np.lib.format.open_memmap(
            true_sct_gep_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, n_genes, n_cell_types),
        )
        true_sct_gep[:] = 0.0
        true_sct_gep_present_mask = np.lib.format.open_memmap(
            true_sct_gep_present_mask_path,
            mode="w+",
            dtype=bool,
            shape=(n_samples, n_cell_types),
        )
        true_sct_gep_present_mask[:] = False

        sample_id_to_positions: Dict[str, list[int]] = {}
        for row_idx, sample_id in enumerate(sample_ids):
            sample_id_to_positions.setdefault(str(sample_id), []).append(row_idx)

        grouped_target_sets: dict[str, dict[str, Any]] = {}
        for target_set_name, target_cfg in self.config.training_target_sets.items():
            raw_bulk_sample_ids = _load_bulk_sample_ids_from_file(target_cfg.training_set_file_path)
            bulk_sample_ids = _namespace_sample_ids(
                raw_bulk_sample_ids,
                namespace=str(target_set_name),
            )
            if not raw_bulk_sample_ids:
                continue

            row_positions: list[int] = []
            missing_sample_ids: list[str] = []
            duplicate_sample_ids: list[str] = []
            for bulk_sample_id in bulk_sample_ids:
                positions = sample_id_to_positions.get(bulk_sample_id, [])
                if len(positions) == 1:
                    row_positions.append(positions[0])
                elif len(positions) == 0:
                    missing_sample_ids.append(bulk_sample_id)
                else:
                    duplicate_sample_ids.append(bulk_sample_id)

            if missing_sample_ids:
                preview = ", ".join(missing_sample_ids[:5])
                raise ValueError(
                    "Could not align matched sctGEP targets because some training bulk "
                    f"samples are missing from the processed dataset: {preview}"
                )
            if duplicate_sample_ids:
                preview = ", ".join(duplicate_sample_ids[:5])
                raise ValueError(
                    "Could not align matched sctGEP targets because some processed sample "
                    f"IDs are duplicated: {preview}"
                )

            sample2cell_fp = Path(target_cfg.training_set_sample2cell_id_file_path)
            filtered_mapping_df = _filter_sample2cell_mapping(sample2cell_fp, raw_bulk_sample_ids)
            sct_gep_fp = Path(target_cfg.training_sct_gep_file_path)
            resolved_sct_gep_fp = str(sct_gep_fp.expanduser().resolve())

            group_entry = grouped_target_sets.setdefault(
                resolved_sct_gep_fp,
                {
                    "sct_gep_fp": sct_gep_fp,
                    "items": [],
                    "selected_cell_ids": [],
                },
            )
            group_entry["items"].append(
                {
                    "raw_bulk_sample_ids": raw_bulk_sample_ids,
                    "row_positions": np.asarray(row_positions),
                    "filtered_mapping_df": filtered_mapping_df,
                }
            )
            if not filtered_mapping_df.empty:
                group_entry["selected_cell_ids"].extend(filtered_mapping_df["selected_cell_id"].tolist())

        for group_entry in grouped_target_sets.values():
            unique_sct_cell_ids = list(dict.fromkeys(group_entry["selected_cell_ids"]))
            aligned_sct_geps_df = _load_cached_aligned_sct_geps(
                sct_gep_fp=Path(group_entry["sct_gep_fp"]),
                selected_cell_ids=unique_sct_cell_ids,
                target_gene_list=gene_list,
            )
            for item in group_entry["items"]:
                _write_true_sct_gep_targets_into(
                    raw_bulk_sample_ids=item["raw_bulk_sample_ids"],
                    row_positions=item["row_positions"],
                    cell_types=cell_types,
                    filtered_mapping_df=item["filtered_mapping_df"],
                    aligned_sct_geps_df=aligned_sct_geps_df,
                    true_sct_gep_dest=true_sct_gep,
                    true_sct_gep_present_mask_dest=true_sct_gep_present_mask,
                )
            if hasattr(true_sct_gep, "flush"):
                true_sct_gep.flush()
            if hasattr(true_sct_gep_present_mask, "flush"):
                true_sct_gep_present_mask.flush()
            del aligned_sct_geps_df
            gc.collect()

        return true_sct_gep, true_sct_gep_present_mask, arrays_pre_saved, temp_paths

    # -------------------------------------------------------------------------
    # PyTorch Dataset interface
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Return dataset size without forcing full data load."""
        if self._n_samples > 0:
            return self._n_samples
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        """
        Get one sample.

        Using torch.as_tensor avoids an unnecessary extra copy in most cases.
        """
        x = torch.as_tensor(self.data[index], dtype=torch.float32)

        if self.labels.size > 0:
            y = torch.as_tensor(self.labels[index], dtype=torch.float32)
        else:
            y = torch.empty(0, dtype=torch.float32)
        if self.true_sct_gep.size > 0:
            true_sct_gep = torch.as_tensor(self.true_sct_gep[index], dtype=torch.float32)
            true_sct_gep_present_mask = torch.as_tensor(
                self.true_sct_gep_present_mask[index],
                dtype=torch.bool,
            )
        else:
            true_sct_gep = torch.empty(0, dtype=torch.float32)
            true_sct_gep_present_mask = torch.empty(0, dtype=torch.bool)

        return DatasetOutput(
            data=x,
            labels=y,
            true_sct_gep=true_sct_gep,
            true_sct_gep_present_mask=true_sct_gep_present_mask,
            sample_id=str(self.sample_ids[index]),
        )

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get multiple items efficiently (batch loading)."""
        x = torch.as_tensor(self.data[indices], dtype=torch.float32)
        if self.labels.size > 0:
            y = torch.as_tensor(self.labels[indices], dtype=torch.float32)
        else:
            y = torch.empty(0, dtype=torch.float32)
        if self.true_sct_gep.size > 0:
            true_sct_gep = torch.as_tensor(self.true_sct_gep[indices], dtype=torch.float32)
            true_sct_gep_present_mask = torch.as_tensor(
                self.true_sct_gep_present_mask[indices],
                dtype=torch.bool,
            )
        else:
            true_sct_gep = torch.empty(0, dtype=torch.float32)
            true_sct_gep_present_mask = torch.empty(0, dtype=torch.bool)
        return {
            "data": x,
            "labels": y,
            "true_sct_gep": true_sct_gep,
            "true_sct_gep_present_mask": true_sct_gep_present_mask,
            "sample_id": [str(self.sample_ids[idx]) for idx in indices],
        }

    def save_gene_list(self, file_path: Union[str, Path]) -> None:
        """Save gene list to file."""
        fp = Path(file_path)
        check_dir(fp.parent)
        GEPCacheManager.save_list_txt(self.gene_list, fp)
        logger.info(f"Gene list saved to {fp}")

    def save_cell_types(self, file_path: Union[str, Path]) -> None:
        """Save cell types to file."""
        fp = Path(file_path)
        check_dir(fp.parent)
        GEPCacheManager.save_list_txt(self.cell_types, fp)
        logger.info(f"Cell types saved to {fp}")

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
        Get cell proportions as DataFrame (loads labels lazily if needed).
        Returns empty DataFrame if labels do not exist.
        """
        if self.labels.size == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.labels, index=self.sample_ids, columns=self.cell_types)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics in MB for loaded arrays only.

        Notes:
        - For memmap arrays, nbytes reflects logical size, not necessarily resident RAM.
        """
        usage: Dict[str, float] = {}

        if self._data is not None:
            usage["data_mb"] = float(self._data.nbytes / (1024 ** 2))

        if self._labels is not None and self._labels.size > 0:
            usage["labels_mb"] = float(self._labels.nbytes / (1024 ** 2))
        if self._true_sct_gep is not None and self._true_sct_gep.size > 0:
            usage["true_sct_gep_mb"] = float(self._true_sct_gep.nbytes / (1024 ** 2))
        if self._true_sct_gep_present_mask is not None and self._true_sct_gep_present_mask.size > 0:
            usage["true_sct_gep_present_mask_mb"] = float(
                self._true_sct_gep_present_mask.nbytes / (1024 ** 2)
            )

        usage["total_mb"] = float(sum(usage.values()))
        return usage

    def unload_data(self) -> None:
        """
        Unload currently loaded arrays from memory references.

        Useful for manual memory management between pipeline stages.
        """
        self._data = None
        self._labels = None
        self._true_sct_gep = None
        self._true_sct_gep_present_mask = None
        gc.collect()
        log_message("Data references unloaded from memory.")


def find_sct_gep_of_bulk_sample(
    sct_gep_dataset_file_path: Union[str, Path],
    sample2cell_id_file_path: Union[str, Path],
    bulk_dataset: GEPDataset,
    cell_types: List[str],
    result_dir: Optional[Union[str, Path]] = None,
    random_seed: Optional[int] = 42,
    n_samples: int = 3,
    selected_sample2cell_id_file_path: Optional[Union[str, Path]] = None,
    cache_sct_query_results: bool = True,
    sct_query_cache_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Select random bulk samples and retrieve aligned SCT GEPs by cell type.

    Workflow:
    1) Validate input files
    2) Randomly select bulk samples
    3) Filter sample->cell mapping to selected samples
    4) Load only required SCT cells from .h5ad
    5) Align SCT genes to bulk gene list
    6) Group by target cell types
    7) Save to disk OR return as dict

    Returns:
        - dict[cell_type -> DataFrame] if result_dir is None
        - None if saving files to result_dir
    """
    all_bulk_sample_ids = bulk_dataset.get_sample_ids()
    target_gene_list = bulk_dataset.get_gene_list()  # bulk gene space

    if not all_bulk_sample_ids:
        logger.warning("No sample IDs found in bulk_dataset. Cannot select samples.")
        return {} if not result_dir else None

    # --- 2) Random bulk sample selection ---
    rng = np.random.default_rng(seed=random_seed)
    n_available = len(all_bulk_sample_ids)
    n_to_select = min(n_samples, n_available)

    if n_to_select < n_samples:
        logger.warning(
            f"Requested {n_samples} samples, but only {n_available} available. Selecting {n_to_select}."
        )
    if n_to_select == 0:
        logger.warning("No samples to select.")
        return {} if not result_dir else None

    selected_idx = rng.choice(n_available, size=n_to_select, replace=False)
    query_bulk_ids = [all_bulk_sample_ids[i] for i in selected_idx]
    logger.info(f"Selected bulk sample IDs: {query_bulk_ids}")

    matched_targets = build_matched_sct_gep_training_targets(
        sct_gep_dataset_file_path=sct_gep_dataset_file_path,
        sample2cell_id_file_path=sample2cell_id_file_path,
        bulk_sample_ids=query_bulk_ids,
        target_gene_list=target_gene_list,
        cell_types=cell_types,
        cache_sct_query_results=cache_sct_query_results,
        sct_query_cache_file_path=sct_query_cache_file_path,
    )
    filtered_mapping_df = matched_targets["selected_sample2cell_id"].copy()

    if filtered_mapping_df.empty:
        logger.warning("Selected bulk IDs have no valid mapping rows.")
        return {} if not result_dir else None

    # Ensure suitable type for obs_name indexing in H5AD
    filtered_mapping_df["selected_cell_id"] = filtered_mapping_df["selected_cell_id"].astype(str)

    if selected_sample2cell_id_file_path:
        selected_fp = Path(selected_sample2cell_id_file_path)
        try:
            selected_fp.parent.mkdir(parents=True, exist_ok=True)
            filtered_mapping_df.to_csv(selected_fp)
            logger.info(f"Saved selected mapping to: {selected_fp}")
        except Exception as e:
            logger.error(f"Error saving selected mapping: {e}")

    processed_sct_geps_df = matched_targets["aligned_sct_geps_df"].copy()
    if processed_sct_geps_df.empty:
        logger.warning("No SCT GEP data loaded for selected cell IDs.")
        return {} if not result_dir else None
    logger.info(f"Processed SCT GEP shape: {processed_sct_geps_df.shape}")

    # --- 5) Group by cell type and save/return ---
    output_geps_by_celltype: Dict[str, pd.DataFrame] = {}

    result_dir_path: Optional[Path] = None
    if result_dir:
        result_dir_path = Path(result_dir)
        result_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving SCT GEPs by cell type to: {result_dir_path}")

    # Build cell_id -> cell_type map and drop duplicated selected_cell_id
    cell_id_to_type = (
        filtered_mapping_df
        .drop_duplicates(subset=["selected_cell_id"], keep="first")
        .set_index("selected_cell_id")["cell_type"]
        .to_dict()
    )

    for cell_type_target in cell_types:
        # Find processed SCT cell IDs belonging to current target cell type
        matched_ids = [
            cid for cid in processed_sct_geps_df.index
            if cell_id_to_type.get(cid) == cell_type_target
        ]

        if not matched_ids:
            logger.warning(f"No SCT GEPs found for cell type '{cell_type_target}'.")
            specific_df = pd.DataFrame(columns=processed_sct_geps_df.columns)
        else:
            specific_df = processed_sct_geps_df.loc[matched_ids, :]

        if result_dir_path is not None:
            # Note: n_to_select is #bulk samples selected, not #single cells.
            out_file = result_dir_path / f"sct_gep_{cell_type_target}_from_{n_to_select}_bulksamples.csv"
            try:
                specific_df.T.to_csv(out_file, float_format='%g')  # genes x cells format
                logger.info(f"Saved {cell_type_target}: {specific_df.shape[0]} cells -> {out_file}")
            except Exception as e:
                logger.error(f"Error saving {cell_type_target} output: {e}")
        else:
            output_geps_by_celltype[cell_type_target] = specific_df

    if result_dir_path is not None:
        return None
    return output_geps_by_celltype
