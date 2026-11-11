"""Chunked, disk-backed datasets for proportion-only bulk training."""

import hashlib
import json
import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utility.read_file import read_gene_set
from .bulk_sources import group_cell_fractions, inspect_bulk_sources
from .datasets import DatasetOutput, load_gene_list

logger = logging.getLogger(__name__)


class BulkOnlyDataset(Dataset):
    """Preprocess bulk H5ADs without reading SCT inputs or constructing targets."""

    def __init__(self, config, model_config=None, preserve_existing=None, expected_fingerprint=None):
        self.config = config
        inspected = inspect_bulk_sources(config)
        self.cell_types = list(inspected.cell_types)
        self.source_cell_types = list(inspected.cell_types)
        genes = list(inspected.gene_names)
        if config.gene_list_file:
            available = set(genes)
            genes = [g for g in load_gene_list(Path(config.gene_list_file)) if g in available]
        if (
            model_config and model_config.deside_pathway_network
            and getattr(model_config, "deside_input_gene_list", None) == "intersection_with_pathway_genes"
        ):
            if not config.pathway_file_path:
                raise ValueError("Pathway gene intersection requires data.pathway_file_path")
            pathway_genes = set(read_gene_set(config.pathway_file_path).index)
            genes = [gene for gene in genes if gene in pathway_genes]
        if not genes or len(set(genes)) != len(genes):
            raise ValueError("The bulk input gene list must contain unique common genes")

        self.sources = inspected.sources
        self.exclusions = [
            {"source": sample_id.split("::", 1)[0], "sample_id": sample_id,
             "reason": "zero_label"}
            for sample_id in inspected.excluded_sample_ids
        ]
        self.excluded_sample_ids = list(inspected.excluded_sample_ids)
        self.sample_ids = [
            f"{source.name}::{source.sample_ids[i]}"
            for source in self.sources for i in source.valid_rows
        ]
        self.labels = np.concatenate([source.labels for source in self.sources], axis=0)
        groups = getattr(model_config, "deside_group_cell_types", {}) or {}
        selected = getattr(model_config, "deside_cell_types", None)
        if groups or selected is not None:
            grouped = group_cell_fractions(
                pd.DataFrame(self.labels, columns=self.cell_types), groups, selected
            )
            self.cell_types = grouped.columns.tolist()
            self.labels = grouped.to_numpy(dtype=np.float32)
        if model_config and model_config.cell_prop_activation_function == "sigmoid":
            if model_config.cancer_cell_type_name not in self.cell_types:
                raise ValueError("Sigmoid bulk training requires the cancer cell type in deside_cell_types")
        self._normalize_intersection = True
        reference = None
        if config.remove_low_var_genes and config.cell_cell2ave_exp_file_path:
            reference_path = Path(config.cell_cell2ave_exp_file_path).expanduser().resolve()
            reference_stat = reference_path.stat()
            reference = (str(reference_path), reference_stat.st_size, reference_stat.st_mtime_ns)
        payload = {
            "version": 4,
            "sources": [
                (source.name, str(source.path), source.path.stat().st_size,
                 source.path.stat().st_mtime_ns)
                for source in self.sources
            ],
            "genes": genes,
            "group_cell_types": groups,
            "cell_types": selected,
            "scale": float(config.scaling_factor) if config.scaling_by_constant else None,
            "normalize_intersection": self._normalize_intersection,
            "remove_low_var_genes": bool(config.remove_low_var_genes),
            "min_var": float(config.min_var),
            "reference": reference,
        }
        self.source_identity = payload["sources"]
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        self.fingerprint = fingerprint
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                "Stage-one bulk preprocessing or source files have changed; "
                "the saved predictor cannot be resumed with these inputs."
            )
        self.processed_data_dir = Path(config.data_dir) / "processed_training_sets" / f"bulk_only_{fingerprint}"
        metadata_path = self.processed_data_dir / "metadata.json"
        data_path = self.processed_data_dir / "data.npy"
        if expected_fingerprint is not None and not (metadata_path.is_file() and data_path.is_file()):
            raise ValueError("The saved stage-one bulk feature cache is missing; retrain stage one")
        if metadata_path.is_file() and data_path.is_file() and not config.force_reprocess:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.gene_list = metadata["gene_list"]
            valid_ids = metadata["sample_ids"]
            positions = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
            if len(valid_ids) != len(set(valid_ids)) or any(name not in positions for name in valid_ids):
                raise ValueError(f"Invalid bulk cache sample identities: {metadata_path}")
            self.labels = self.labels[[positions[name] for name in valid_ids]]
            self.sample_ids = valid_ids
            self.exclusions = metadata["exclusions"]
            self.excluded_sample_ids = [row["sample_id"] for row in self.exclusions]
            self.data = np.load(data_path, mmap_mode="r")
            if self.data.shape != (len(self.sample_ids), len(self.gene_list)):
                raise ValueError(f"Bulk cache shape does not match input metadata: {data_path}")
            return
        if self.processed_data_dir.exists() and any(self.processed_data_dir.iterdir()):
            if preserve_existing is None:
                raise FileExistsError(
                    f"Bulk training cache already contains files: {self.processed_data_dir}. "
                    "Use a fresh cache location."
                )
            preserve_existing(self.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.gene_list, common_valid = self._filter_genes(genes)
        all_ids = self.sample_ids
        total_rows = len(all_ids)
        matrix = np.lib.format.open_memmap(
            data_path, mode="w+", dtype=np.float32,
            shape=(total_rows, len(self.gene_list)),
        )
        cursor = written = 0
        kept = []
        for block, expression_valid in self._iter_expression(self.gene_list, with_valid=True):
            n_rows = len(block)
            common = common_valid[cursor:cursor + n_rows]
            valid = expression_valid & common
            matrix[written:written + int(valid.sum())] = block[valid]
            written += int(valid.sum())
            for offset in np.flatnonzero(~valid):
                sample_id = all_ids[cursor + offset]
                self.exclusions.append({
                    "source": sample_id.split("::", 1)[0],
                    "sample_id": sample_id,
                    "reason": "zero_common_expression" if not common[offset] or not config.remove_low_var_genes
                    else "zero_selected_expression",
                })
            kept.extend((cursor + np.flatnonzero(valid)).tolist())
            cursor += n_rows
            logger.info("Preprocessed %s/%s bulk samples", cursor, total_rows)
        matrix.flush()
        if not written:
            del matrix
            pd.DataFrame(self.exclusions, columns=["source", "sample_id", "reason"]).to_csv(
                self.processed_data_dir / "excluded_samples.csv", index=False
            )
            raise ValueError("No bulk samples have nonzero expression in the selected genes")
        if written != total_rows:
            compact_path = self.processed_data_dir / "data_compact.npy"
            compact = np.lib.format.open_memmap(
                compact_path, mode="w+", dtype=np.float32, shape=(written, len(self.gene_list))
            )
            for start in range(0, written, self.config.chunk_size):
                end = min(start + self.config.chunk_size, written)
                compact[start:end] = matrix[start:end]
            compact.flush()
            del compact, matrix
            compact_path.replace(data_path)
        else:
            del matrix
        self.labels = self.labels[kept]
        self.sample_ids = [all_ids[index] for index in kept]
        self.excluded_sample_ids = [row["sample_id"] for row in self.exclusions]
        metadata_path.write_text(json.dumps({
            "gene_list": self.gene_list, "sample_ids": self.sample_ids,
            "exclusions": self.exclusions,
        }, indent=2), encoding="utf-8")
        self.data = np.load(data_path, mmap_mode="r")

    def validate_source_identity(self):
        current = [
            (source.name, str(source.path), source.path.stat().st_size,
             source.path.stat().st_mtime_ns)
            for source in self.sources
        ]
        if current != self.source_identity:
            raise ValueError("Stage-one bulk source files changed after preprocessing")

    def _iter_expression(self, genes, *, linear=False, with_valid=False):
        linear_mode = linear
        for source in self.sources:
            gene_positions = {gene: i for i, gene in enumerate(source.gene_names)}
            positions = np.asarray([gene_positions[gene] for gene in genes])
            dataset = anndata.read_h5ad(source.path, backed="r")
            try:
                for start in range(0, len(source.valid_rows), self.config.chunk_size):
                    rows = source.valid_rows[start:start + self.config.chunk_size]
                    values = dataset.X[rows, :]
                    if hasattr(values, "toarray"):
                        values = values.toarray()
                    values = np.asarray(values[:, positions], dtype=np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Bulk source {source.name!r} contains non-finite expression")
                    if linear_mode or self._normalize_intersection:
                        linear_values = np.maximum(np.exp2(values) - 1.0, 0.0)
                        if not np.isfinite(linear_values).all():
                            raise ValueError(f"Bulk source {source.name!r} contains invalid linear expression")
                        totals = linear_values.sum(axis=1, keepdims=True)
                        if not np.isfinite(totals).all():
                            raise ValueError(f"Bulk source {source.name!r} has non-finite expression totals")
                        valid = totals[:, 0] > 0
                        normalized = linear_values * (1e6 / np.maximum(totals, 1e-12))
                        values = normalized if linear_mode else np.log2(normalized + 1.0)
                    if not linear_mode and self.config.scaling_by_constant:
                        values = values / float(self.config.scaling_factor)
                    yield (values, valid) if with_valid else values
            finally:
                dataset.file.close()

    def _filter_genes(self, genes):
        if not self.config.remove_low_var_genes:
            return genes, np.ones(len(self.sample_ids), dtype=bool)
        sums = np.zeros(len(genes), dtype=np.float64)
        squares = np.zeros(len(genes), dtype=np.float64)
        count = 0
        logger.info(
            "Filtering %s common genes by linear TPM variance > %s across %s bulk samples",
            len(genes), self.config.min_var, len(self.sample_ids),
        )
        valid_blocks = []
        for block, valid in self._iter_expression(genes, linear=True, with_valid=True):
            valid_blocks.append(valid)
            values = block[valid].astype(np.float64)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
            count += len(values)
            logger.info("Computed linear TPM variance for %s/%s bulk samples", count, len(self.sample_ids))
        if not count:
            self.exclusions.extend({
                "source": sample_id.split("::", 1)[0], "sample_id": sample_id,
                "reason": "zero_common_expression",
            } for sample_id in self.sample_ids)
            pd.DataFrame(self.exclusions, columns=["source", "sample_id", "reason"]).to_csv(
                self.processed_data_dir / "excluded_samples.csv", index=False
            )
            raise ValueError("No bulk samples have nonzero expression in the common genes")
        variance = (squares - sums * sums / count) / max(count - 1, 1)
        selected = [g for g, v in zip(genes, variance) if v > self.config.min_var]
        if self.config.cell_cell2ave_exp_file_path:
            reference = set(pd.read_csv(self.config.cell_cell2ave_exp_file_path, index_col=0).index.astype(str))
            selected = [g for g in selected if g in reference]
        if not selected:
            raise ValueError("Low-variance filtering removed all bulk genes")
        logger.info("Retained %s/%s genes after linear TPM variance filtering", len(selected), len(genes))
        return selected, np.concatenate(valid_blocks)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return DatasetOutput(
            data=torch.tensor(self.data[index], dtype=torch.float32),
            labels=torch.as_tensor(self.labels[index], dtype=torch.float32),
            sample_id=self.sample_ids[index],
        )

    def get_sample_ids(self):
        return list(self.sample_ids)

    def get_gene_list(self):
        return list(self.gene_list)

    def get_cell_types(self):
        return list(self.cell_types)

    def get_cell_prop(self):
        return pd.DataFrame(self.labels, index=self.sample_ids, columns=self.cell_types)

    def save_gene_list(self, path):
        Path(path).write_text("\n".join(self.gene_list) + "\n", encoding="utf-8")

    def save_cell_types(self, path):
        Path(path).write_text("\n".join(self.cell_types) + "\n", encoding="utf-8")


class BulkAlignedDataset(Dataset):
    """Use stage-one bulk features in a full dataset with SCT supervision."""

    stage_aware_target_loading = True

    def __init__(self, full_dataset, bulk_dataset):
        self.full_dataset = full_dataset
        self.bulk_dataset = bulk_dataset
        self.sample_ids = full_dataset.sample_ids
        self.bulk_indices = {
            sample_id: index for index, sample_id in enumerate(bulk_dataset.sample_ids)
        }

    def __len__(self):
        return len(self.full_dataset)

    def get_sample_ids(self):
        return list(self.sample_ids)

    def get_item(self, index, *, include_sct_gep, include_sct_mask):
        output = self.full_dataset.get_item(
            index, include_sct_gep=include_sct_gep, include_sct_mask=include_sct_mask
        )
        bulk_index = self.bulk_indices.get(self.sample_ids[index])
        if bulk_index is not None:
            bulk = self.bulk_dataset[bulk_index]
            output["data"] = bulk["data"]
            output["labels"] = bulk["labels"]
        return output

    def __getitem__(self, index):
        return self.get_item(index, include_sct_gep=True, include_sct_mask=True)
