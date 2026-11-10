"""Metadata validation for proportion-only bulk training inputs."""

from dataclasses import dataclass
from pathlib import Path

import anndata
import numpy as np

from ..configs import DataConfig


@dataclass(frozen=True)
class BulkSource:
    name: str
    path: Path
    sample_ids: tuple[str, ...]
    valid_rows: np.ndarray
    labels: np.ndarray
    gene_names: tuple[str, ...]


@dataclass(frozen=True)
class BulkSources:
    sources: tuple[BulkSource, ...]
    gene_names: tuple[str, ...]
    cell_types: tuple[str, ...]
    excluded_sample_ids: tuple[str, ...]


def group_cell_fractions(frame, groups, selected=None):
    """Apply DeSide-compatible cell-type grouping and optional selection."""
    frame = frame.copy()
    for group, subtypes in groups.items():
        subtypes = list(subtypes)
        if not subtypes:
            continue
        if len(subtypes) != len(set(subtypes)):
            raise ValueError(f"Duplicate subtypes in group {group!r}")
        missing = [name for name in subtypes if name not in frame.columns]
        if missing and group not in frame.columns:
            raise ValueError(f"Cell-type group {group!r} is missing subtypes: {missing}")
        present = [name for name in subtypes if name in frame.columns]
        if group in frame.columns and group not in present:
            present.insert(0, group)
        if not present:
            continue
        if len(present) == 1 and present[0] != group:
            frame = frame.rename(columns={present[0]: group})
        elif len(present) > 1:
            frame[group] = frame.loc[:, present].sum(axis=1)
            frame = frame.drop(columns=[name for name in present if name != group])
    if selected is not None:
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("deside_cell_types must contain unique cell types")
        missing = [name for name in selected if name not in frame.columns]
        if missing:
            raise ValueError(f"Selected cell types are missing: {missing}")
        frame = frame.loc[:, selected]
    if ((frame.to_numpy(dtype=float) < 0) | (frame.to_numpy(dtype=float) > 1)).any():
        raise ValueError("Grouped cell fractions must remain between zero and one")
    return frame


def resolve_bulk_source_paths(config: DataConfig) -> list[tuple[str, Path]]:
    """Resolve named bulk inputs and reject ambiguous source identities."""
    configured = [
        Path(path).expanduser().resolve()
        for path in config.simu_bulk_file_path
        if path is not None and str(path).strip()
    ]
    if len(configured) != len(set(configured)):
        raise ValueError("data.simu_bulk_file_path contains duplicate paths")

    sources = [
        (str(name), Path(bundle.training_set_file_path).expanduser().resolve())
        for name, bundle in config.training_target_sets.items()
    ]
    if sources:
        source_paths = [path for _, path in sources]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("data.training_target_sets contains duplicate bulk paths")
        if configured and set(configured) != set(source_paths):
            raise ValueError(
                "data.simu_bulk_file_path and data.training_target_sets must specify "
                "the same bulk files for cell_prop_predictor_pretrain"
            )
    else:
        sources = [(path.stem, path) for path in configured]

    if not sources:
        raise ValueError("cell_prop_predictor_pretrain requires bulk training files")
    names = [name for name, _ in sources]
    if any("::" in name for name in names) or len(names) != len(set(names)):
        raise ValueError("Bulk source names must be unique and must not contain '::'")
    return sources


def inspect_bulk_sources(config: DataConfig) -> BulkSources:
    """Read and validate H5AD metadata without loading expression matrices."""
    sources = []
    common_genes = None
    cell_types = None
    excluded = []
    for name, path in resolve_bulk_source_paths(config):
        if path.suffix.lower() != ".h5ad":
            raise ValueError(f"Bulk source {name!r} must be an H5AD file: {path}")
        dataset = anndata.read_h5ad(path, backed="r")
        try:
            sample_ids = tuple(str(value) for value in dataset.obs_names)
            genes = tuple(str(value) for value in dataset.var_names)
            columns = tuple(str(value) for value in dataset.obs.columns)
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Bulk source {name!r} has duplicate sample IDs")
            if len(genes) != len(set(genes)):
                raise ValueError(f"Bulk source {name!r} has duplicate genes")
            if not columns or len(columns) != len(set(columns)):
                raise ValueError(f"Bulk source {name!r} has missing or duplicate cell types")
            if cell_types is None:
                cell_types = columns
            elif set(columns) != set(cell_types):
                raise ValueError(f"Bulk source {name!r} has incompatible cell types")
            try:
                labels = dataset.obs.loc[:, list(cell_types)].to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Bulk source {name!r} has nonnumeric cell fractions") from exc
            if not np.isfinite(labels).all():
                raise ValueError(f"Bulk source {name!r} has missing or non-finite cell fractions")
            if ((labels < 0) | (labels > 1)).any():
                raise ValueError(f"Bulk source {name!r} has out-of-range cell fractions")
            valid_rows = np.flatnonzero(labels.sum(axis=1) > 0)
            excluded.extend(f"{name}::{sample_ids[i]}" for i in np.flatnonzero(labels.sum(axis=1) == 0))
            gene_set = set(genes)
            common_genes = list(genes) if common_genes is None else [
                gene for gene in common_genes if gene in gene_set
            ]
            sources.append(BulkSource(
                name=name,
                path=path,
                sample_ids=sample_ids,
                valid_rows=valid_rows,
                labels=labels[valid_rows].astype(np.float32),
                gene_names=genes,
            ))
        finally:
            dataset.file.close()
    if not common_genes:
        raise ValueError("Bulk training inputs have no common genes")
    if not any(len(source.valid_rows) for source in sources):
        raise ValueError("Bulk training inputs contain no nonzero-label samples")
    return BulkSources(tuple(sources), tuple(common_genes), cell_types, tuple(excluded))
