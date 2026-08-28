"""
Training pipeline for VAEDecon
"""
import copy
import hashlib
import json
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, random_split
from ..data import GEPDataset
# from ..models.vae import VAEConfig
from ..trainers import BaseTrainerL
from ..utility import set_output_dir, log_message, set_fig_style
from ..utility import load_or_compute_gene_mean_std, load_lightning_metrics, compute_gene_mean_std_from_pooled_sc_h5ad
from ..utility import compute_training_sct_cross_sample_gene_var
from ..utility import create_h5ad_dataset
from ..utility.read_file import ReadH5AD
from .workflow import create_model, train_model, save_metadata
from ..trainers.base_trainer import _unwrap_stage_model
from ..configs.default_config import (
    VAEDeconConfig,
    TrainingConfig,
    ModelConfig,
    GEPDatasetConfig,
    TestSetConfig,
)

logger = logging.getLogger(__name__)


def _resolve_path_for_fingerprint(path_like: Optional[str | Path]) -> Optional[str]:
    """Normalize paths for stable preprocessing-cache fingerprints."""
    if path_like is None or str(path_like).strip() == "":
        return None
    return str(Path(path_like).expanduser().resolve())

def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.tensor([0.0], device="cuda")
        (x + 1).sum().item()
        return True
    except Exception:
        return False


class VAEDeconTrainer:
    """VAEDecon Trainer"""

    def __init__(self, config: Optional[VAEDeconConfig] = None):
        """
        Initializes the trainer

        Parameters:
            config: VAEDeconConfig, using default VAEDeconConfig if None
        """
        self.config = config or VAEDeconConfig()
        self.model_dir: Path | str = ''
        # self._vae_config: Optional[ModelConfig] = None  # cached, built once in _prepare_data
        self._processed_training_set_dir: Optional[Path] = None  # exposed for cleanup after training

        self._setup_logging()
        self._setup_device()
        self._setup_directories()  # Set model_dir and result_dir, and update config.model paths
        set_fig_style(font_family='Arial', font_size=8)

    def _setup_logging(self):
        """Attach a console handler only if none exists yet (avoids duplicate lines)."""
        if not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers):
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            logger.addHandler(console)
        logger.setLevel(logging.INFO)

    def _setup_device(self):
        """Resolve and store the computing device"""
        if self.config.training.device == 'auto':
            if _cuda_usable():
                self.device = 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = self.config.training.device
            if self.device == "cuda" and not _cuda_usable():
                raise RuntimeError(
                    "Requested device 'cuda' but CUDA kernels cannot run on this machine. "
                    "This commonly indicates a GPU compute capability mismatch with the installed PyTorch CUDA build. "
                    "Use device='cpu' or install a compatible PyTorch CUDA build for your GPU."
                )
        logger.info(f"Using device: {self.device}")

    def _setup_directories(self):
        """Create and register output directories"""
        model_dir_cfg = self.config.model.model_dir
        model_dir_unset = not model_dir_cfg or str(model_dir_cfg).strip() == ''
        if model_dir_unset:
            self.result_dir = Path(
                set_output_dir(
                    output_dir=self.config.training.output_dir,
                    naming_postfix=self.config.training.naming_postfix
                )
            )
            self.model_dir = self.result_dir / 'final_model'
            self.config.model.model_dir = self.model_dir
        else:
            self.model_dir = Path(model_dir_cfg)
            self.result_dir = self.model_dir.parent

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Update file paths in model config
        self.config.model.input_gene_list_fp = self.model_dir / 'input_gene_list.txt'
        self.config.model.cell_type_fp = self.model_dir / 'cell_type_list.txt'

        logger.info(f"Results will be saved to: {self.result_dir}")

    def _prepare_data(self) -> Tuple[GEPDataset, any, any]:
        """Load, preprocess, and split the training data"""
        logger.info("Loading and preparing data...")

        debug_overfit_cfg = self.config.training.debug_overfit
        if not debug_overfit_cfg.enabled:
            total_split = self.config.training.train_split + self.config.training.val_split
            if not abs(total_split - 1.0) < 1e-6:
                raise ValueError(
                    f"train_split ({self.config.training.train_split}) + "
                    f"val_split ({self.config.training.val_split}) must sum to 1.0, got {total_split:.4f}."
                )

        # Load GEP dataset, PPI and Pathway data will be handled in each specified encoder class.
        gep_dataset_config = self._build_gepdataset_config()  # training file paths and preprocessing params
        dataset = GEPDataset(config=gep_dataset_config)

        logger.info(f"Dataset shape: {dataset.data.shape}")

        if debug_overfit_cfg.enabled:
            train_set, val_set = self._build_debug_overfit_subsets(dataset)
            logger.info(
                "Debug overfit mode enabled: train subset size=%s, eval subset size=%s",
                len(train_set),
                len(val_set) if val_set is not None else 0,
            )
        else:
            train_set, val_set = random_split(
                dataset,
                [self.config.training.train_split, self.config.training.val_split]
            )

            logger.info(f"Train set: {len(train_set)}, Val set: {len(val_set)}")

        # Update input_dim, gene_mean_std_fp, and build ModelConfig once
        # Build and cache ModelConfig here; _create_model reuses it
        self.config.model = self._build_vae_config(dataset=dataset)
        save_metadata(dataset=dataset, model_config=self.config.model)

        # TODO, only calculate gene mean/std when we need it, such as GNN or predict_gep_residual is true.
        # Calculate gene mean and std as features for GNN
        self._compute_gene_statistics(dataset, self.config.model.gene_mean_std_fp)

        # Export training SCT cross-sample gene variance CSV (only if new loss is enabled)
        cross_var_fp = self._prepare_and_save_training_sct_cross_sample_gene_var()
        if cross_var_fp is not None:
            self.config.model.training_sct_cross_sample_gene_var_fp = cross_var_fp
        per_sample_var_fp = self._prepare_and_save_training_sct_per_sample_residual_var(dataset)
        if per_sample_var_fp is not None:
            self.config.model.training_sct_per_sample_residual_var_fp = per_sample_var_fp

        return dataset, train_set, val_set

    def _build_debug_overfit_subsets(
        self,
        dataset: GEPDataset,
    ) -> Tuple[Subset, Optional[Subset]]:
        """Build reproducible debug overfit train/eval subsets from one dataset."""
        debug_cfg = self.config.training.debug_overfit
        n_samples = len(dataset)
        subset_size = int(debug_cfg.subset_size)
        if subset_size > n_samples:
            raise ValueError(
                f"debug_overfit.subset_size ({subset_size}) exceeds dataset size ({n_samples})."
            )

        generator = torch.Generator().manual_seed(int(debug_cfg.subset_seed))
        selected_indices = torch.randperm(n_samples, generator=generator)[:subset_size].tolist()
        selected_indices = sorted(int(idx) for idx in selected_indices)

        train_subset = Subset(dataset, selected_indices)
        eval_subset = train_subset if debug_cfg.use_training_subset_as_eval else None
        self._save_debug_overfit_subset_manifest(dataset, selected_indices)
        return train_subset, eval_subset

    def _save_debug_overfit_subset_manifest(
        self,
        dataset: GEPDataset,
        selected_indices: list[int],
    ) -> Path:
        """Save selected debug subset row indices and sample IDs for reproducibility."""
        sample_ids = dataset.get_sample_ids() if hasattr(dataset, "get_sample_ids") else []
        rows = []
        for idx in selected_indices:
            sample_id = sample_ids[idx] if idx < len(sample_ids) else str(idx)
            rows.append(
                {
                    "debug_subset_row_index": int(idx),
                    "sample_id": str(sample_id),
                }
            )

        out_fp = Path(self.model_dir) / "debug_overfit_subset.csv"
        pd.DataFrame(rows).to_csv(out_fp, index=False)
        logger.info("Saved debug overfit subset manifest to: %s", out_fp)
        return out_fp

    @staticmethod
    def _filter_rows_by_index_order(df: pd.DataFrame, ordered_index: list[str]) -> pd.DataFrame:
        """Return rows in the requested order while preserving duplicated indices."""
        selected_frames: list[pd.DataFrame] = []
        index_str = df.index.astype(str)
        for sample_id in ordered_index:
            current = df.loc[index_str == str(sample_id), :].copy()
            if not current.empty:
                selected_frames.append(current)
        if not selected_frames:
            return df.iloc[0:0, :].copy()
        return pd.concat(selected_frames, axis=0)

    def _build_debug_overfit_selected_samples_by_target_set(
        self,
        dataset: GEPDataset,
        train_set,
    ) -> Dict[str, List[tuple[str, str]]]:
        """Group selected debug subset samples by source training target set."""
        if not isinstance(train_set, Subset):
            raise ValueError(
                "debug overfit prediction workflow requires the training subset "
                "to be a torch.utils.data.Subset."
            )

        sample_ids = dataset.get_sample_ids()
        grouped: Dict[str, List[tuple[str, str]]] = {}
        for idx in train_set.indices:
            sample_id = str(sample_ids[int(idx)])
            if "::" not in sample_id:
                continue
            namespace, raw_sample_id = sample_id.split("::", 1)
            grouped.setdefault(namespace, []).append((sample_id, raw_sample_id))
        return grouped

    def _prepare_debug_overfit_test_sets(
        self,
        dataset: GEPDataset,
        train_set,
    ) -> None:
        """Materialize selected debug training samples as normal test-set bundles."""
        debug_cfg = self.config.training.debug_overfit
        if not debug_cfg.enabled or not debug_cfg.use_training_subset_as_eval:
            return
        training_target_sets = dict(self.config.data.training_target_sets or {})
        if not training_target_sets:
            logger.warning(
                "Debug overfit post-training prediction was requested, but "
                "data.training_target_sets is empty. Skipping debug test-set generation."
            )
            return

        grouped_samples = self._build_debug_overfit_selected_samples_by_target_set(
            dataset=dataset,
            train_set=train_set,
        )
        if not grouped_samples:
            logger.warning(
                "No namespaced debug subset samples could be mapped back to "
                "training_target_sets. Skipping debug test-set generation."
            )
            return

        debug_input_dir = Path(self.result_dir) / "debug_overfit_test_sets"
        debug_input_dir.mkdir(parents=True, exist_ok=True)
        debug_cell_prop = dataset.get_cell_prop()
        generated_test_sets: Dict[str, TestSetConfig] = {}
        manifest_rows: list[dict[str, str]] = []

        for target_set_name, selected_pairs in grouped_samples.items():
            target_cfg = training_target_sets.get(target_set_name)
            if target_cfg is None:
                logger.warning(
                    "Debug subset samples referenced namespace '%s', but no "
                    "matching training_target_sets entry was found.",
                    target_set_name,
                )
                continue

            namespaced_ids = [pair[0] for pair in selected_pairs]
            raw_sample_ids = [pair[1] for pair in selected_pairs]
            bulk_source_path = Path(target_cfg.training_set_file_path)
            subset_bulk_fp = debug_input_dir / f"{target_set_name}_debug_overfit_subset.h5ad"
            subset_mapping_fp = debug_input_dir / f"{target_set_name}_debug_overfit_sample2cell.csv"

            if bulk_source_path.suffix.lower() == ".h5ad":
                bulk_reader = ReadH5AD(bulk_source_path)
                bulk_adata = bulk_reader.get_h5ad()
                available_ids = [sample_id for sample_id in raw_sample_ids if sample_id in bulk_adata.obs_names]
                if not available_ids:
                    logger.warning(
                        "No selected debug samples from %s were found in %s.",
                        target_set_name,
                        bulk_source_path,
                    )
                    continue
                bulk_subset = bulk_adata[available_ids, :].copy()
                bulk_subset.write_h5ad(filename=subset_bulk_fp, compression="gzip")
            elif bulk_source_path.suffix.lower() == ".csv":
                bulk_exp_df = pd.read_csv(bulk_source_path, index_col=0)
                filtered_bulk_exp = self._filter_rows_by_index_order(
                    bulk_exp_df,
                    raw_sample_ids,
                )
                filtered_bulk_exp_fp = debug_input_dir / f"{target_set_name}_debug_overfit_subset_bulk.csv"
                filtered_bulk_exp.to_csv(filtered_bulk_exp_fp, float_format="%.6f")

                filtered_cell_prop = debug_cell_prop.loc[namespaced_ids, :].copy()
                filtered_cell_prop.index = raw_sample_ids
                filtered_cell_prop_fp = debug_input_dir / f"{target_set_name}_debug_overfit_subset_cell_prop.csv"
                filtered_cell_prop.to_csv(filtered_cell_prop_fp, float_format="%.6f")

                create_h5ad_dataset(
                    simulated_bulk_exp_file_path=str(filtered_bulk_exp_fp),
                    cell_fraction_file_path=str(filtered_cell_prop_fp),
                    dataset_info=f"debug overfit subset from {target_set_name}",
                    result_file_path=str(subset_bulk_fp),
                    gep_type="bulk",
                )
            else:
                logger.warning(
                    "Unsupported training_set_file_path suffix for debug test-set generation: %s",
                    bulk_source_path,
                )
                continue

            sample2cell_df = pd.read_csv(target_cfg.training_set_sample2cell_id_file_path, index_col=0)
            filtered_mapping_df = self._filter_rows_by_index_order(sample2cell_df, raw_sample_ids)
            filtered_mapping_df.to_csv(subset_mapping_fp)

            debug_test_name = f"Debug_overfit_{target_set_name}"
            generated_test_sets[debug_test_name] = TestSetConfig(
                test_set_file_path=subset_bulk_fp,
                test_set_sample2cell_id_file_path=subset_mapping_fp,
                sct_gep_file_path=target_cfg.training_sct_gep_file_path,
            )
            for namespaced_id, raw_sample_id in selected_pairs:
                manifest_rows.append(
                    {
                        "debug_test_name": debug_test_name,
                        "target_set_name": target_set_name,
                        "sample_id": namespaced_id,
                        "raw_sample_id": raw_sample_id,
                        "subset_bulk_file_path": str(subset_bulk_fp),
                        "subset_sample2cell_id_file_path": str(subset_mapping_fp),
                    }
                )

        if not generated_test_sets:
            logger.warning("No debug overfit test sets were generated.")
            return

        self.config.data.test_sets = generated_test_sets
        first_test = next(iter(generated_test_sets.values()))
        self.config.data.test_set_file_path = first_test.test_set_file_path
        self.config.data.test_set_sample2cell_id_file_path = first_test.test_set_sample2cell_id_file_path
        self.config.data.sct_gep_file_path = first_test.sct_gep_file_path

        manifest_fp = debug_input_dir / "debug_overfit_test_sets_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_fp, index=False)
        logger.info(
            "Prepared %s debug overfit test set(s) for normal post-training prediction. Manifest: %s",
            len(generated_test_sets),
            manifest_fp,
        )

    def _compute_gene_statistics(
        self,
        dataset: GEPDataset,
        gene_mean_std_fp: Path = None,
    ) -> None:
        """Compute or load per-gene mean/std statistics for GNN node features.
        Parameters:
            dataset: GEPDataset
            gene_mean_std_fp: Path to save or load the gene mean/std statistics.
                If the file exists, it will be loaded; otherwise, it will be computed and saved to this path.
        """
        logger.info("Computing gene statistics...")

        if self.config.data.gene_mean_std_source == "pooled_sc":
            compute_gene_mean_std_from_pooled_sc_h5ad(
                pooled_sc_h5ad_fp=str(self.config.data.pooled_sc_h5ad_path),
                result_fp=gene_mean_std_fp,
                gene_list_fp=self.config.model.input_gene_list_fp,
                cell_type_fp=self.config.model.cell_type_fp,
                cell_type_col=self.config.data.pooled_sc_cell_type_col,
                cell_subtype_col=self.config.data.pooled_sc_cell_subtype_col,
                sample_size=self.config.data.pooled_sc_sample_size,
                seed=self.config.data.pooled_sc_seed,
                scaling_by_constant=self.config.data.scaling_by_constant,
                scaling_factor=self.config.data.scaling_factor,
            )
        else:
            load_or_compute_gene_mean_std(
                sct_gep_fp=self._resolve_gene_mean_std_sct_gep_paths(),
                gene_list=dataset.gene_list,
                cell_type_fp=self.config.model.cell_type_fp,
                input_gene_list_fp=self.config.model.input_gene_list_fp,
                scaling_by_constant=self.config.data.scaling_by_constant,
                scaling_factor=self.config.data.scaling_factor,
                log_fn=log_message,
                out_fp=gene_mean_std_fp,
            )

    def _resolve_gene_mean_std_sct_gep_paths(self) -> list[Path]:
        dedicated_fp = self.config.data.gene_mean_std_sct_gep_file_path
        if dedicated_fp and str(dedicated_fp).strip() != "":
            return [Path(dedicated_fp)]
        training_sct_paths = [Path(fp) for fp in (self.config.data.sct_file_path or []) if fp and str(fp).strip() != ""]
        if training_sct_paths:
            return training_sct_paths
        training_target_sct_paths = [
            Path(cfg.training_sct_gep_file_path)
            for cfg in (self.config.data.training_target_sets or {}).values()
            if cfg.training_sct_gep_file_path and str(cfg.training_sct_gep_file_path).strip() != ""
        ]
        if training_target_sct_paths:
            deduped_paths: list[Path] = []
            seen: set[str] = set()
            for path in training_target_sct_paths:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                deduped_paths.append(path)
            return deduped_paths
        top_level_fp = self.config.data.sct_gep_file_path
        if top_level_fp and str(top_level_fp).strip() != "":
            return [Path(top_level_fp)]
        return []

    def _resolve_gene_mean_std_sct_gep_path(self) -> Path:
        paths = self._resolve_gene_mean_std_sct_gep_paths()
        if not paths:
            return Path("")
        return paths[0]

    def _build_gene_mean_std_output_path(self) -> Path:
        scaling_factor = self.config.data.scaling_factor
        model_dir = Path(self.config.model.model_dir)
        if self.config.data.scaling_by_constant:
            return model_dir / f"gene_mean_std_log2p1_scaled_by_{scaling_factor}.csv"
        return model_dir / "gene_mean_std_log2p1.csv"

    def _build_training_sct_cross_sample_gene_var_output_path(self) -> Path:
        model_dir = Path(self.config.model.model_dir)
        scaling_factor = self.config.data.scaling_factor
        if self.config.data.scaling_by_constant:
            return model_dir / f"training_sct_cross_sample_gene_variances_log2p1_scaled_by_{scaling_factor}.csv"
        return model_dir / "training_sct_cross_sample_gene_variances_log2p1.csv"

    def _build_training_sct_per_sample_residual_var_output_path(self) -> Path:
        model_dir = Path(self.config.model.model_dir)
        scaling_factor = self.config.data.scaling_factor
        if self.config.data.scaling_by_constant:
            return model_dir / f"training_sct_per_sample_residual_variance_log2p1_scaled_by_{scaling_factor}.csv"
        return model_dir / "training_sct_per_sample_residual_variance_log2p1.csv"

    def _prepare_and_save_training_sct_cross_sample_gene_var(self) -> Optional[Path]:
        """Compute and save SCT cross-sample gene variance CSV if the new loss is enabled."""
        weight = getattr(getattr(self.config.model, "loss_coefficient", None), "cross_sample_gene_var_weight", 0.0) or 0.0
        if weight <= 0:
            return None

        sct_gep_fps = [p for p in self._resolve_gene_mean_std_sct_gep_paths() if p is not None and Path(p).exists()]
        if not sct_gep_fps:
            raise FileNotFoundError(
                "loss_coefficient.cross_sample_gene_var_weight > 0 requires training SCT h5ad path(s), "
                "but none of gene_mean_std_sct_gep_file_path, sct_gep_file_path, or sct_file_path "
                "point to an existing file."
            )

        out_fp = self._build_training_sct_cross_sample_gene_var_output_path()
        if not Path(out_fp).exists():
            compute_training_sct_cross_sample_gene_var(
                sct_dataset_fp=sct_gep_fps,
                result_fp=out_fp,
                gene_list_fp=self.config.model.input_gene_list_fp,
                cell_type_fp=self.config.model.cell_type_fp,
                scaling_by_constant=self.config.data.scaling_by_constant,
                log2p1=True,
                scaling_factor=self.config.data.scaling_factor,
            )
        else:
            logger.info(f"Using existing training SCT cross-sample gene variance CSV at {out_fp}")
        return out_fp

    def _prepare_and_save_training_sct_per_sample_residual_var(
        self,
        dataset: GEPDataset,
    ) -> Optional[Path]:
        """Compute and save matched-SCT per-sample residual variance CSV if the new loss is enabled."""
        weight = float(
            getattr(getattr(self.config.model, "loss_coefficient", None), "per_sample_residual_var_weight", 0.0) or 0.0
        )
        if weight <= 0:
            return None

        if not self.config.model.learn_gep_residual or self.config.model.learn_gep_residual_mode != "mean_centered":
            raise ValueError(
                "loss_coefficient.per_sample_residual_var_weight > 0 requires "
                "learn_gep_residual=True and learn_gep_residual_mode='mean_centered'."
            )

        if dataset.true_sct_gep.size == 0 or dataset.true_sct_gep_present_mask.size == 0:
            raise ValueError(
                "loss_coefficient.per_sample_residual_var_weight > 0 requires dataset batches "
                "to include matched true_sct_gep targets. Configure data.training_target_sets."
            )

        out_fp = self._build_training_sct_per_sample_residual_var_output_path()
        if Path(out_fp).exists():
            logger.info(f"Using existing training SCT per-sample residual variance CSV at {out_fp}")
            return out_fp

        gene_stats_df = pd.read_csv(self.config.model.gene_mean_std_fp, index_col=0)
        avg_cols = [f"{ct}_avg" for ct in dataset.cell_types]
        if any(col not in gene_stats_df.columns for col in avg_cols):
            missing = [col for col in avg_cols if col not in gene_stats_df.columns]
            raise ValueError(
                "Could not compute per-sample residual variance targets because gene_mean_std "
                f"is missing expected columns: {missing}"
            )

        g_mean = gene_stats_df.loc[dataset.gene_list, avg_cols].to_numpy(dtype=np.float32, copy=False)
        true_sct_gep = np.asarray(dataset.true_sct_gep, dtype=np.float32)
        present_mask = np.asarray(dataset.true_sct_gep_present_mask, dtype=bool)

        residual = true_sct_gep - g_mean[np.newaxis, :, :]
        per_sample_var = np.var(residual, axis=1, ddof=0).astype(np.float32, copy=False)
        per_sample_var[~present_mask] = np.nan

        out_df = pd.DataFrame(
            per_sample_var,
            index=[str(sample_id) for sample_id in dataset.get_sample_ids()],
            columns=dataset.cell_types,
        )
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_fp, float_format='%g')
        logger.info("Saved training SCT per-sample residual variance to %s", out_fp)
        return out_fp

    def _create_model(self, model_config: ModelConfig):
        """Instantiate the VAE model using the cached ModelConfig."""
        logger.info("Creating model...")

        # Create VAE model by combining encoder and decoder classes specified in the config
        model = create_model(
            model_config=model_config,
            data_config=self.config.data,
            encoder_cls_name_list=model_config.encoders,
            decoder_cls=model_config.decoders,
            device=self.device,
        )

        return model

    def _build_vae_config(self, dataset: GEPDataset) -> ModelConfig:
        """
        Build a ModelConfig from self.config (model + data sections).

        """
        input_dim = dataset.data.shape[1]  # same as n_genes in each GEP
        n_genes = input_dim
        gene_mean_std_fp = self._build_gene_mean_std_output_path()
        per_sample_residual_var_fp = self.config.model.training_sct_per_sample_residual_var_fp
        if (
            float(getattr(self.config.model.loss_coefficient, "per_sample_residual_var_weight", 0.0) or 0.0) > 0
            and per_sample_residual_var_fp is None
        ):
            per_sample_residual_var_fp = self._build_training_sct_per_sample_residual_var_output_path()

        return ModelConfig(
            name='ModelConfig',
            input_dim=(1, n_genes),
            input_dim_pathway=self.config.model.input_dim_pathway,
            latent_dim=self.config.model.latent_dim,
            n_cell_types=self.config.model.n_cell_types,
            using_positional_encoding=self.config.model.using_positional_encoding,
            input_gene_list_fp=self.config.model.input_gene_list_fp,
            cell_type_fp=self.config.model.cell_type_fp,
            gene_mean_std_fp=gene_mean_std_fp,
            training_sct_cross_sample_gene_var_fp=self.config.model.training_sct_cross_sample_gene_var_fp,
            training_sct_per_sample_residual_var_fp=per_sample_residual_var_fp,
            # scaling_by_constant=self.config.data.scaling_by_constant,
            encoder_hidden_dims=self.config.model.encoder_hidden_dims,
            encoder_hidden_dims_pathway=self.config.model.encoder_hidden_dims_pathway,
            decoder_hidden_dims=self.config.model.decoder_hidden_dims,
            encoder_dropout_rate=self.config.model.encoder_dropout_rate,
            encoder_dropout_rate_pathway=self.config.model.encoder_dropout_rate_pathway,
            decoder_dropout_rate=self.config.model.decoder_dropout_rate,
            fusion_hidden_dims=self.config.model.fusion_hidden_dims,
            fusion_dropout_rate=self.config.model.fusion_dropout_rate,
            predict_cell_prop=self.config.model.predict_cell_prop,
            encoder_aliases=self.config.model.encoder_aliases,
            encoder_output_routing=self.config.model.encoder_output_routing,
            cell_prop_predictor_cls=self.config.model.cell_prop_predictor_cls,
            cell_prop_predictor_alias=self.config.model.cell_prop_predictor_alias,
            deside_pathway_network=self.config.model.deside_pathway_network,
            deside_hidden_dims=self.config.model.deside_hidden_dims,
            deside_dropout_rate=self.config.model.deside_dropout_rate,
            deside_pathway_hidden_dims=self.config.model.deside_pathway_hidden_dims,
            deside_pathway_dropout_rate=self.config.model.deside_pathway_dropout_rate,
            deside_input_gene_list=self.config.model.deside_input_gene_list,
            deside_normalization=self.config.model.deside_normalization,
            deside_normalization_layer=self.config.model.deside_normalization_layer,
            cell_prop_activation_function=self.config.model.cell_prop_activation_function,
            cancer_cell_type_name=self.config.model.cancer_cell_type_name,
            cell_type_existence_shift_scale=self.config.model.cell_type_existence_shift_scale,
            cell_prop_fusion_strategy=self.config.model.cell_prop_fusion_strategy,
            cell_prop_fusion_dim=self.config.model.cell_prop_fusion_dim,
            cell_prop_head_hidden_dims=self.config.model.cell_prop_head_hidden_dims,
            cell_prop_head_dropout_rate=self.config.model.cell_prop_head_dropout_rate,
            cell_prop_loss_type=self.config.model.cell_prop_loss_type,
            cell_prop_loss_alpha_weight=self.config.model.cell_prop_loss_alpha_weight,
            cell_prop_loss_weighting=self.config.model.cell_prop_loss_weighting,
            cell_prop_loss_low_prop_epsilon=self.config.model.cell_prop_loss_low_prop_epsilon,
            cell_prop_loss_weight_clamp=self.config.model.cell_prop_loss_weight_clamp,
            loss_coefficient=self.config.model.loss_coefficient,
            gnn_n_genes=self.config.model.gnn_n_genes,
            gnn_inter_col_dim=self.config.model.gnn_inter_col_dim,
            gnn_embd_col_dim=self.config.model.gnn_embd_col_dim,
            gnn_lambda_cols=self.config.model.gnn_lambda_cols,
            gnn_num_layers=self.config.model.gnn_num_layers,
            gnn_drop_p=self.config.model.gnn_drop_p,
            # ppi_file_path=self.config.model.ppi_file_path,
            # pathway_file_path=self.config.model.pathway_file_path,
            gene_hidden_dim=self.config.model.gene_hidden_dim,
            encoders=self.config.model.encoders,
            decoders=self.config.model.decoders,
            mask_ratio=self.config.model.mask_ratio,
            learn_gep_residual=self.config.model.learn_gep_residual,
            learn_gep_residual_mode=self.config.model.learn_gep_residual_mode,
            conditional_decoder_cell_type_emb_dim=self.config.model.conditional_decoder_cell_type_emb_dim,
            conditional_decoder_context_dim=self.config.model.conditional_decoder_context_dim,
            conditional_decoder_dropout_rate=self.config.model.conditional_decoder_dropout_rate,
            # SCALING_FACTOR=self.config.model.SCALING_FACTOR,
            model_dir=self.config.model.model_dir,
            torch_compile=self.config.model.torch_compile,
        )

    def _build_trainer_config(self) -> TrainingConfig:
        """Convert to TrainerConfig"""
        trainer_config = self.config.training.model_copy(deep=True)
        trainer_config.per_device_train_batch_size = self.config.training.batch_size
        trainer_config.per_device_eval_batch_size = self.config.training.batch_size
        return trainer_config

    def _resolve_training_dataset_inputs(self) -> tuple[list[str | Path], dict]:
        """Resolve training file paths and matched-target settings for the dataset."""
        simu_paths = [
            path for path in (self.config.data.simu_bulk_file_path or [])
            if path is not None and str(path).strip() != ""
        ]
        sct_paths  = self.config.data.sct_file_path or []
        training_file_paths = [p for p in simu_paths + sct_paths if p is not None]
        matched_sct_gep_weight = float(
            getattr(self.config.model.loss_coefficient, "cell_type_sct_gep_weight", 0.0) or 0.0
        )
        training_target_sets = {}
        if matched_sct_gep_weight > 0:
            training_target_sets = dict(self.config.data.training_target_sets or {})
            if not training_target_sets:
                raise ValueError(
                    "loss_coefficient.cell_type_sct_gep_weight > 0 requires "
                    "data.training_target_sets to define matched bulk/sample2cell/sct bundles."
                )

            target_bulk_paths: dict[str, str] = {}
            duplicate_target_bulk_paths: dict[str, list[str]] = {}
            derived_simu_paths: list[str | Path] = []
            for set_name, cfg in training_target_sets.items():
                raw_path = cfg.training_set_file_path
                resolved_path = str(Path(raw_path).expanduser().resolve())
                if resolved_path in target_bulk_paths:
                    duplicate_target_bulk_paths.setdefault(resolved_path, [target_bulk_paths[resolved_path]]).append(
                        set_name
                    )
                    continue
                target_bulk_paths[resolved_path] = set_name
                derived_simu_paths.append(raw_path)

            if duplicate_target_bulk_paths:
                preview = ", ".join(
                    f"{Path(path)} ({', '.join(set_names)})"
                    for path, set_names in list(sorted(duplicate_target_bulk_paths.items()))[:3]
                )
                raise ValueError(
                    "data.training_target_sets must not reuse the same training_set_file_path "
                    f"across multiple entries. Duplicates: {preview}"
                )

            if not simu_paths:
                simu_paths = derived_simu_paths
                training_file_paths = [p for p in simu_paths + sct_paths if p is not None]
            else:
                configured_bulk_paths = {
                    str(Path(path).expanduser().resolve()): str(path)
                    for path in simu_paths
                }
                missing_target_paths = set(configured_bulk_paths) - set(target_bulk_paths)
                if missing_target_paths:
                    preview = ", ".join(
                        configured_bulk_paths[path]
                        for path in sorted(missing_target_paths)[:3]
                    )
                    raise ValueError(
                        "Each simulated bulk training set needs a matched entry in "
                        f"data.training_target_sets when cell_type_sct_gep_weight > 0. Missing: {preview}"
                    )
                extra_target_paths = set(target_bulk_paths) - set(configured_bulk_paths)
                if extra_target_paths:
                    preview = ", ".join(
                        f"{target_bulk_paths[path]} -> {Path(path)}"
                        for path in sorted(extra_target_paths)[:3]
                    )
                    raise ValueError(
                        "data.training_target_sets contains bulk files that are not present in "
                        "data.simu_bulk_file_path when cell_type_sct_gep_weight > 0. "
                        f"Remove or disable the unused target set entries, or add their bulk "
                        f"files to data.simu_bulk_file_path. Extra: {preview}"
                    )

        return training_file_paths, training_target_sets

    def _build_training_dataset_cache_fingerprint(
        self,
        *,
        training_file_paths: list[str | Path],
        training_target_sets: dict,
    ) -> str:
        """Return a stable content fingerprint for the processed training dataset cache."""
        training_targets_payload = {
            str(set_name): {
                "training_set_file_path": _resolve_path_for_fingerprint(cfg.training_set_file_path),
                "training_set_sample2cell_id_file_path": _resolve_path_for_fingerprint(
                    cfg.training_set_sample2cell_id_file_path
                ),
                "training_sct_gep_file_path": _resolve_path_for_fingerprint(cfg.training_sct_gep_file_path),
            }
            for set_name, cfg in sorted(training_target_sets.items())
        }
        payload = {
            "version": 2,
            "file_paths": [
                _resolve_path_for_fingerprint(path)
                for path in training_file_paths
                if path is not None and str(path).strip() != ""
            ],
            "scaling_by_constant": self.config.data.scaling_by_constant,
            "scaling_factor": float(self.config.data.scaling_factor),
            "remove_low_var_genes": bool(self.config.data.remove_low_var_genes),
            "min_var": float(self.config.data.min_var),
            "gene_list_file": _resolve_path_for_fingerprint(getattr(self.config.data, "gene_list_file", None)),
            "cell_cell2ave_exp_file_path": _resolve_path_for_fingerprint(
                getattr(self.config.data, "cell_cell2ave_exp_file_path", None)
            ),
            "training_target_sets": training_targets_payload,
            "training_sct_gep_cell_prop_threshold": float(
                getattr(self.config.data, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
            ),
        }
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]

    def _resolve_processed_training_set_dir(
        self,
        *,
        training_file_paths: list[str | Path],
        training_target_sets: dict,
    ) -> Path:
        """Build the shared processed-training cache directory for the current dataset inputs."""
        fingerprint = self._build_training_dataset_cache_fingerprint(
            training_file_paths=training_file_paths,
            training_target_sets=training_target_sets,
        )
        return Path(self.config.data.data_dir) / "processed_training_sets" / fingerprint

    def _build_gepdataset_config(self) -> GEPDatasetConfig:
        """
        Build a config dict for GEPDataset from self.config.data
        """
        training_file_paths, training_target_sets = self._resolve_training_dataset_inputs()
        self._processed_training_set_dir = self._resolve_processed_training_set_dir(
            training_file_paths=training_file_paths,
            training_target_sets=training_target_sets,
        )

        return GEPDatasetConfig(
            file_paths=training_file_paths,
            scaling_by_constant=self.config.data.scaling_by_constant,
            remove_low_var_genes=self.config.data.remove_low_var_genes,
            force_reprocess=self.config.data.force_reprocess,
            max_parallel_source_file_loads=self.config.data.max_parallel_source_file_loads,
            use_memmap=self.config.data.use_memmap,
            chunk_size=self.config.data.chunk_size,
            min_var=self.config.data.min_var,
            scaling_factor=self.config.data.scaling_factor,
            gene_list_file=self.config.data.gene_list_file,
            cell_cell2ave_exp_file_path=self.config.data.cell_cell2ave_exp_file_path,
            gene_mean_std_source=self.config.data.gene_mean_std_source,
            gene_mean_std_sct_gep_file_path=self.config.data.gene_mean_std_sct_gep_file_path,
            sct_gep_file_path=self.config.data.sct_gep_file_path,
            pooled_sc_h5ad_path=self.config.data.pooled_sc_h5ad_path,
            pooled_sc_cell_type_col=self.config.data.pooled_sc_cell_type_col,
            pooled_sc_cell_subtype_col=self.config.data.pooled_sc_cell_subtype_col,
            pooled_sc_sample_size=self.config.data.pooled_sc_sample_size,
            pooled_sc_seed=self.config.data.pooled_sc_seed,
            training_target_sets=training_target_sets,
            training_sct_gep_cell_prop_threshold=self.config.data.training_sct_gep_cell_prop_threshold,
            processed_data_dir=self._processed_training_set_dir,
        )

    @staticmethod
    def _resolve_stage_named_modules(model) -> Dict[str, torch.nn.Module]:
        base_model = _unwrap_stage_model(model)
        modules: Dict[str, torch.nn.Module] = {}
        encoders = getattr(base_model, "encoders", None)
        if encoders is not None:
            modules["encoders"] = encoders
        decoder = getattr(base_model, "decoder", None)
        if decoder is not None:
            modules["decoder"] = decoder
        cell_prop_predictor = getattr(base_model, "cell_prop_predictor", None)
        if cell_prop_predictor is not None:
            modules["cell_prop_predictor"] = cell_prop_predictor
        return modules

    def _apply_stage_module_trainability(
        self,
        model,
        *,
        train_modules: list[str],
    ) -> Dict[str, str]:
        modules = self._resolve_stage_named_modules(model)
        module_mode_overrides: Dict[str, str] = {}
        for module_name, module in modules.items():
            is_trainable = module_name in set(train_modules)
            for parameter in module.parameters():
                parameter.requires_grad = is_trainable
            if is_trainable:
                module.train()
                module_mode_overrides[module_name] = "train"
            else:
                module.eval()
                module_mode_overrides[module_name] = "eval"
        return module_mode_overrides

    @staticmethod
    def _reset_stage_loss_overrides(model, base_model_config: ModelConfig) -> None:
        base_model = _unwrap_stage_model(model)
        base_model.model_config.loss_coefficient = base_model_config.loss_coefficient.model_copy(deep=True)
        base_model.model_config.cell_type_existence_shift_scale = base_model_config.cell_type_existence_shift_scale

    @staticmethod
    def _apply_stage_loss_overrides(model, loss_overrides: Dict[str, float]) -> None:
        if not loss_overrides:
            return
        base_model = _unwrap_stage_model(model)
        for attr_name, attr_value in loss_overrides.items():
            if hasattr(base_model.model_config.loss_coefficient, attr_name):
                setattr(base_model.model_config.loss_coefficient, attr_name, float(attr_value))
                continue
            if hasattr(base_model.model_config, attr_name):
                setattr(base_model.model_config, attr_name, float(attr_value))
                continue
            raise ValueError(f"Unsupported staged-training loss override: {attr_name}")

    @staticmethod
    def _extract_model_state_dict_from_checkpoint(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalized_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            while key.startswith("model._orig_mod."):
                key = key[len("model._orig_mod."):]
            if key.startswith("model."):
                key = key[len("model."):]
            while key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
            normalized_state_dict[key] = value
        return normalized_state_dict

    def _load_stage_checkpoint(self, model, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_model_state_dict_from_checkpoint(checkpoint)
        base_model = _unwrap_stage_model(model)
        base_model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _extract_cell_prop_predictor_state_dict_from_checkpoint(
        checkpoint: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        state_dict = checkpoint.get("state_dict", checkpoint)
        predictor_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key
            while normalized_key.startswith("model._orig_mod."):
                normalized_key = normalized_key[len("model._orig_mod."):]
            if normalized_key.startswith("model."):
                normalized_key = normalized_key[len("model."):]
            while normalized_key.startswith("_orig_mod."):
                normalized_key = normalized_key[len("_orig_mod."):]

            if normalized_key.startswith("cell_prop_predictor."):
                predictor_state_dict[normalized_key[len("cell_prop_predictor."):]] = value
            elif not any(
                normalized_key.startswith(prefix)
                for prefix in ("encoders.", "decoder.", "logits", "hierarchical_code_head")
            ):
                predictor_state_dict[normalized_key] = value
        return predictor_state_dict

    def _load_cell_prop_predictor_checkpoint(self, model, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        predictor_state_dict = self._extract_cell_prop_predictor_state_dict_from_checkpoint(checkpoint)
        if not predictor_state_dict:
            raise ValueError(
                f"No cell_prop_predictor weights could be resolved from staged-training checkpoint: {checkpoint_path}"
            )
        base_model = _unwrap_stage_model(model)
        predictor = getattr(base_model, "cell_prop_predictor", None)
        if predictor is None:
            raise ValueError("The current model does not expose cell_prop_predictor for predictor-only checkpoint import.")
        predictor.load_state_dict(predictor_state_dict, strict=True)

    def _export_cell_prop_predictor_checkpoint(self, model, output_path: Path) -> None:
        base_model = _unwrap_stage_model(model)
        predictor = getattr(base_model, "cell_prop_predictor", None)
        if predictor is None:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": predictor.state_dict()}, output_path)

    @staticmethod
    def _build_stage_training_config(
        base_training_config: TrainingConfig,
        *,
        stage_cfg,
    ) -> TrainingConfig:
        stage_training_config = base_training_config.model_copy(deep=True)
        stage_training_config.num_epochs = int(stage_cfg.max_epochs)
        stage_training_config.learning_rate = (
            float(base_training_config.learning_rate) * float(stage_cfg.learning_rate_scale)
        )
        stage_training_config.n_early_stopping_patience = int(stage_cfg.early_stopping.patience)
        stage_training_config.staged_training = None
        if stage_cfg.name == "cell_prop_predictor_pretrain":
            prog_bar_metrics = list(stage_training_config.prog_bar_metrics)
            if "cell_prop_loss" not in prog_bar_metrics:
                prog_bar_metrics.append("cell_prop_loss")
            stage_training_config.prog_bar_metrics = prog_bar_metrics
        return stage_training_config

    @staticmethod
    def _resolve_stage_checkpoint_paths(stage_dir: Path) -> tuple[Path, Path]:
        metadata_path = stage_dir / "checkpoint_paths.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing staged-training checkpoint metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        best_name = str(metadata.get("best_model_path", "")).strip()
        last_name = str(metadata.get("last_model_path", "")).strip()
        if not best_name:
            raise FileNotFoundError(f"No best checkpoint recorded in {metadata_path}")
        best_path = stage_dir / best_name
        last_path = stage_dir / last_name if last_name else stage_dir / "last_model.ckpt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best staged-training checkpoint not found: {best_path}")
        if not last_path.exists():
            raise FileNotFoundError(f"Last staged-training checkpoint not found: {last_path}")
        return best_path, last_path

    def _run_post_stage_test_set_prediction(
        self,
        *,
        stage_name: str,
        stage_dir: Path,
        predictor_model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        has_named_test_sets = bool(getattr(self.config.data, "test_sets", {}))
        legacy_test_set_path = getattr(self.config.data, "test_set_file_path", "")
        has_legacy_test_set = bool(legacy_test_set_path and str(legacy_test_set_path).strip())
        if not has_named_test_sets and not has_legacy_test_set:
            logger.info(
                "Skipping post-stage prediction for %s because no test sets are configured.",
                stage_name,
            )
            return None

        from .inference import VAEDeconPredictor

        inference_config = copy.deepcopy(self.config)
        effective_model_dir = predictor_model_dir or stage_dir
        inference_config.model.model_dir = effective_model_dir
        if stage_name == "cell_prop_predictor_pretrain":
            inference_config.evaluation.save_reconstructed_gep = False
            inference_config.evaluation.plot_single_cell_gep = False
            inference_config.evaluation.plot_bulk_gep = False
            inference_config.evaluation.plot_latent_space = False

        output_dir = output_dir or (stage_dir / "test_results")
        if stage_name == "cell_prop_predictor_pretrain":
            logger.info(
                "Running configured test-set cell proportion evaluation after %s in %s",
                stage_name,
                output_dir,
            )
        else:
            logger.info(
                "Running full configured test-set inference after %s in %s",
                stage_name,
                output_dir,
            )
        predictor = VAEDeconPredictor(
            model_dir=str(effective_model_dir),
            config=inference_config,
            device=self.device,
        )
        predictor.predict_configured_test_sets(
            output_dir=str(output_dir),
            dataset_type="test",
            visualize=True,
        )
        return output_dir

    @staticmethod
    def _should_run_stage_local_test_set_prediction(stage_name: str) -> bool:
        """Keep stage-local test results only for the earlier staged-training phases."""
        return stage_name != "joint_finetune"

    def _run_final_model_test_set_prediction(
        self,
        *,
        final_stage_name: str,
        final_stage_dir: Path,
    ) -> Optional[Path]:
        """Run the canonical final evaluation into final_model/test_results."""
        return self._run_post_stage_test_set_prediction(
            stage_name=final_stage_name,
            stage_dir=final_stage_dir,
            predictor_model_dir=self.model_dir,
            output_dir=self.model_dir / "test_results",
        )

    @staticmethod
    def _plot_stage_training_history(
        *,
        stage_name: str,
        stage_dir: Path,
    ) -> None:
        losses_path = stage_dir / "losses.csv"
        if not losses_path.exists():
            logger.info(
                "Skipping loss plot for %s because %s does not exist.",
                stage_name,
                losses_path,
            )
            return

        try:
            history_df = pd.read_csv(losses_path)
            from ..plot.plot_nn import plot_loss, plot_loss_panels

            metric_pairs = None
            if stage_name == "cell_prop_predictor_pretrain":
                metric_pairs = [
                    ("train_cell_prop_loss_epoch", "train loss"),
                    ("val_cell_prop_loss", "val loss"),
                ]
                plot_loss(
                    history_df=history_df,
                    output_dir=stage_dir,
                    metric_pairs=metric_pairs,
                )
            elif stage_name == "reconstruction_training":
                plot_loss_panels(
                    history_df=history_df,
                    output_dir=stage_dir,
                    panel_metric_pairs=[
                        {
                            "metric_pairs": [
                                ("train_loss_epoch", "train loss"),
                                ("val_loss", "val loss"),
                            ],
                            "title": "Total Loss",
                        },
                        {
                            "metric_pairs": [
                                ("train_cell_type_sct_gep_loss_epoch", "train sctGEP loss"),
                                ("val_cell_type_sct_gep_loss", "val sctGEP loss"),
                            ],
                            "title": "Cell-Type sctGEP Loss",
                        },
                    ],
                )
            else:
                plot_loss(
                    history_df=history_df,
                    output_dir=stage_dir,
                    metric_pairs=metric_pairs,
                )
            logger.info("Saved loss curve for %s to %s", stage_name, stage_dir / "loss.png")
        except Exception as exc:
            logger.warning("Could not plot loss curve for %s: %s", stage_name, exc)

    def _promote_stage_outputs_to_final_model(
        self,
        *,
        final_stage_dir: Path,
        model,
        base_training_config: TrainingConfig,
        stage_summary_rows: list[dict],
    ) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        removable_patterns = ["*.ckpt", "metrics.csv", "adaptive_aux_loss_schedule_trace.csv", "debug_overfit_predictions.pt"]
        for pattern in removable_patterns:
            for path in self.model_dir.glob(pattern):
                path.unlink()
        for path in [
            self.model_dir / "checkpoint_paths.json",
            self.model_dir / "training_logs",
            self.model_dir / "test_results",
        ]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

        skip_names = {
            "model_config.json",
            "training_config.json",
            "data_config.json",
            "environment.json",
            "test_results",
        }
        for item in final_stage_dir.iterdir():
            if item.name in skip_names:
                continue
            destination = self.model_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)

        summary_df = pd.DataFrame(stage_summary_rows)
        summary_df.to_csv(self.result_dir / "staged_training_summary.csv", index=False)
        summary_df.to_csv(self.model_dir / "staged_training_summary.csv", index=False)

        restored_model_config = self.config.model.model_copy(deep=True)
        base_model = _unwrap_stage_model(model)
        base_model.model_config = restored_model_config
        self.config.model = restored_model_config
        base_model.save(
            self.model_dir,
            training_config=base_training_config,
            data_config=self.config.data,
        )

    def _train_with_staged_training(
        self,
        *,
        model,
        train_set,
        val_set,
        trainer_config: TrainingConfig,
    ) -> None:
        staged_training_cfg = trainer_config.staged_training
        if staged_training_cfg is None or not staged_training_cfg.enabled:
            raise ValueError("Staged training was requested without an enabled training.staged_training config.")

        configured_stages = list(staged_training_cfg.stages)
        stage_by_name = {stage.name: stage for stage in configured_stages}
        configured_stage_names = [stage.name for stage in configured_stages]
        run_stage_names = staged_training_cfg.run_stages or configured_stage_names
        execution_stages = [stage_by_name[stage_name] for stage_name in run_stage_names]

        base_training_config = trainer_config.model_copy(deep=True)
        base_model_config = self.config.model.model_copy(deep=True)
        executed_stage_best_checkpoints: Dict[str, Path] = {}
        stage_summary_rows: list[dict] = []

        logger.info("Starting staged training with stages: %s", ", ".join(run_stage_names))
        for stage_index, stage_cfg in enumerate(execution_stages, start=1):
            configured_index = configured_stage_names.index(stage_cfg.name)
            direct_predecessor = configured_stage_names[configured_index - 1] if configured_index > 0 else None
            init_checkpoint_path: Optional[Path] = None
            init_source = "fresh_model"
            if direct_predecessor and direct_predecessor in executed_stage_best_checkpoints:
                init_checkpoint_path = executed_stage_best_checkpoints[direct_predecessor]
                init_source = f"previous_stage:{direct_predecessor}"
            elif direct_predecessor:
                raw_checkpoint = staged_training_cfg.stage_init_checkpoints.get(stage_cfg.name)
                if raw_checkpoint is not None:
                    init_checkpoint_path = Path(raw_checkpoint).expanduser()
                    init_source = f"stage_init_checkpoints:{stage_cfg.name}"

            if init_checkpoint_path is not None:
                if not init_checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Staged training init checkpoint for stage {stage_cfg.name!r} does not exist: "
                        f"{init_checkpoint_path}"
                    )
                logger.info(
                    "Loading staged-training init checkpoint for %s from %s",
                    stage_cfg.name,
                    init_checkpoint_path,
                )
                if stage_cfg.name == "reconstruction_training" and direct_predecessor not in executed_stage_best_checkpoints:
                    self._load_cell_prop_predictor_checkpoint(model, init_checkpoint_path)
                else:
                    self._load_stage_checkpoint(model, init_checkpoint_path)

            self._reset_stage_loss_overrides(model, base_model_config)
            self._apply_stage_loss_overrides(model, stage_cfg.loss_overrides)
            module_mode_overrides = self._apply_stage_module_trainability(
                model,
                train_modules=list(stage_cfg.train_modules),
            )

            stage_training_config = self._build_stage_training_config(
                base_training_config,
                stage_cfg=stage_cfg,
            )
            stage_dir = self.result_dir / f"stage_{stage_cfg.name}"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Starting stage %s/%s: %s (lr=%.3e, monitor=%s)",
                stage_index,
                len(execution_stages),
                stage_cfg.name,
                stage_training_config.learning_rate,
                stage_cfg.early_stopping.monitor,
            )
            stage_trainer = BaseTrainerL(
                model=model,
                result_dir=str(stage_dir),
                train_dataset=train_set,
                eval_dataset=val_set,
                training_config=stage_training_config,
                data_config=self.config.data,
                n_early_stopping_patience=stage_cfg.early_stopping.patience,
                debug_model=stage_training_config.debug_model,
                monitor_metric=stage_cfg.early_stopping.monitor,
                early_stopping_min_delta=stage_cfg.early_stopping.min_delta,
                max_epochs_override=stage_cfg.max_epochs,
                module_mode_overrides=module_mode_overrides,
            )
            stage_trainer.train()
            self._plot_stage_training_history(
                stage_name=stage_cfg.name,
                stage_dir=stage_dir,
            )
            if stage_cfg.name == "cell_prop_predictor_pretrain":
                self._export_cell_prop_predictor_checkpoint(
                    model=model,
                    output_path=stage_dir / "cell_prop_predictor_state_dict.pt",
                )

            best_ckpt_path, last_ckpt_path = self._resolve_stage_checkpoint_paths(stage_dir)
            executed_stage_best_checkpoints[stage_cfg.name] = best_ckpt_path
            post_stage_test_results_dir = None
            if self._should_run_stage_local_test_set_prediction(stage_cfg.name):
                post_stage_test_results_dir = self._run_post_stage_test_set_prediction(
                    stage_name=stage_cfg.name,
                    stage_dir=stage_dir,
                )
            stage_summary_rows.append(
                {
                    "stage_name": stage_cfg.name,
                    "stage_dir": str(stage_dir),
                    "init_source": init_source,
                    "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path is not None else "",
                    "best_checkpoint": str(best_ckpt_path),
                    "last_checkpoint": str(last_ckpt_path),
                    "learning_rate": float(stage_training_config.learning_rate),
                    "learning_rate_scale": float(stage_cfg.learning_rate_scale),
                    "monitor": stage_cfg.early_stopping.monitor,
                    "patience": int(stage_cfg.early_stopping.patience),
                    "min_delta": float(stage_cfg.early_stopping.min_delta),
                    "train_modules": ",".join(stage_cfg.train_modules),
                    "freeze_modules": ",".join(stage_cfg.freeze_modules),
                    "test_results_dir": (
                        str(post_stage_test_results_dir)
                        if post_stage_test_results_dir is not None
                        else ""
                    ),
                }
            )

        if not stage_summary_rows:
            raise ValueError("No staged-training stages were executed.")

        self._reset_stage_loss_overrides(model, base_model_config)
        final_stage_dir = Path(stage_summary_rows[-1]["stage_dir"])
        self._promote_stage_outputs_to_final_model(
            final_stage_dir=final_stage_dir,
            model=model,
            base_training_config=base_training_config,
            stage_summary_rows=stage_summary_rows,
        )
        self._run_final_model_test_set_prediction(
            final_stage_name=stage_summary_rows[-1]["stage_name"],
            final_stage_dir=final_stage_dir,
        )
        redundant_root_test_results_dir = self.result_dir / "test_results"
        if redundant_root_test_results_dir.exists():
            shutil.rmtree(redundant_root_test_results_dir, ignore_errors=True)

    def train(self) -> VAEDeconConfig:
        """
        Run the full training pipeline: data preparation, model creation, and training loop.

        Return:
            VAEDecon configuration (maybe updated during training)
        """
        dataset, train_set, val_set = self._prepare_data()  # self.config.model is updated
        model = self._create_model(model_config=self.config.model)
        trainer_config = self._build_trainer_config()

        # Train the model
        logger.info("Starting training...")
        staged_training_cfg = trainer_config.staged_training
        if staged_training_cfg is not None and staged_training_cfg.enabled:
            self._train_with_staged_training(
                model=model,
                train_set=train_set,
                val_set=val_set,
                trainer_config=trainer_config,
            )
        else:
            train_model(
                model=model,
                train_set=train_set,
                val_set=val_set,
                training_config=trainer_config,
                data_config=self.config.data,
                device=self.device,
                trainer_cls=BaseTrainerL,
                result_dir=self.model_dir
            )

        self._prepare_debug_overfit_test_sets(
            dataset=dataset,
            train_set=train_set,
        )

        logger.info(f"Training completed! Model saved to: {self.model_dir}")

        return self.config


def _save_config(
    config: VAEDeconConfig,
    model_dir: Path,
    config_file: Optional[str],
) -> None:
    """
    Persist configuration to model_dir.

    If config_file is given, the original YAML is copied (with original name,
    default name, and timestamped name).  Otherwise the config object is
    serialised directly.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if config_file is not None:
            src = Path(config_file)
            for dst in [
                model_dir / f"config_{src.stem}.yaml",
                model_dir / "config.yaml",
                model_dir / f"config_{timestamp}.yaml",
            ]:
                shutil.copy2(src, dst)
                logger.info(f"Saved config to {dst}")
        else:
            for dst in [
                model_dir / "config.yaml",
                model_dir / f"config_{timestamp}.yaml",
            ]:
                config.to_yaml(dst)
                logger.info(f"Saved config to {dst}")
    except Exception as exc:
        logger.warning(f"Could not save config: {exc}")


def _save_training_error_report(
    *,
    model_dir: Path,
    error: Exception,
) -> None:
    """Persist the latest training exception to the run directory."""
    error_text = (
        f"Training failed at {datetime.now().isoformat(timespec='seconds')}\n"
        f"Exception type: {type(error).__name__}\n"
        f"Message: {error}\n\n"
        "Traceback:\n"
        f"{traceback.format_exc()}"
    )

    candidate_paths = [
        model_dir / "training_error.txt",
        model_dir.parent / "training_error.txt",
    ]
    for path in candidate_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(error_text, encoding="utf-8")
            logger.error("Saved training error report to %s", path)
        except Exception as save_exc:
            logger.warning("Could not save training error report to %s: %s", path, save_exc)


def train_vaedecon(
        config: Optional[VAEDeconConfig] = None,
        config_file: Optional[str] = None
) -> Optional[VAEDeconConfig]:
    """
    Wrapper function to train VAEDecon model.

    This function serves as the main entry point for training a VAEDecon model.
    It handles configuration loading (from file or object), directory setup,
    model initialization, and the training loop.

    Parameters:
        config (Optional[VAEDeconConfig]):
            A VAEDecon configuration object. If provided, it overrides default settings.
            Either `config` or `config_file` should be provided.
        config_file (Optional[str]):
            Path to a YAML configuration file. If provided, the configuration is loaded from this file.
            If both `config` and `config_file` are provided, `config_file` takes precedence
            for initial loading, but any programmatic changes to `config` should be applied before calling.

    Returns:
        Optional[VAEDeconConfig]:
            The final configuration object used for training, which may contain updates
            (e.g., paths to saved models, calculated input dimensions).
            Returns None if training fails or is skipped.

    Examples:
        # 1. Use default configuration
        train_vaedecon()

        # 2. Use a YAML config file
        train_vaedecon(config_file='configs/example_config.yaml')

        # 3. Use a custom configuration object
        from vaedecon.configs import VAEDeconConfig
        config = VAEDeconConfig()
        config.training.num_epochs = 200
        train_vaedecon(config=config)
    """
    # ── Resolve config ────────────────────────────────────────────────────
    if config_file is not None:
        config = VAEDeconConfig.from_yaml(config_file)
    elif config is None:
        config = VAEDeconConfig()

    # ── Resolve model_dir ─────────────────────────────────────────────────
    if config.model.model_dir is not None:
        model_dir = Path(config.model.model_dir)
    else:
        model_dir = Path(config.training.output_dir) / config.training.naming_postfix / 'final_model'
        config.model.model_dir = model_dir

    # Check for existing checkpoint BEFORE creating directories
    if model_dir.exists():
        ckpt_files = [f for f in model_dir.iterdir() if f.suffix == '.ckpt']
        if ckpt_files:
            logger.info(f"Checkpoint found in {model_dir}. Skipping training.")
            saved_config_candidates = [
                model_dir / "config.yaml",
                model_dir / "config_final.yaml",
                model_dir / "used_config.yaml",
            ]
            for candidate in saved_config_candidates:
                if candidate.exists() and candidate.is_file():
                    try:
                        loaded_config = VAEDeconConfig.from_yaml(candidate)
                        for attr_name in (
                            "test_set_file_path",
                            "test_set_sample2cell_id_file_path",
                            "sct_gep_file_path",
                        ):
                            attr_value = getattr(loaded_config.data, attr_name, "")
                            if isinstance(attr_value, str) and attr_value.strip():
                                attr_path = Path(attr_value)
                                if attr_path.is_absolute():
                                    setattr(loaded_config.data, attr_name, attr_path)
                        loaded_config.model.cell_type_fp = model_dir / 'cell_type_list.txt'
                        loaded_config.model.input_gene_list_fp = model_dir / 'input_gene_list.txt'
                        loaded_config.model.model_dir = model_dir
                        logger.info(
                            "Loaded saved config from %s while reusing existing checkpoint.",
                            candidate,
                        )
                        return loaded_config
                    except Exception as exc:
                        logger.warning(
                            "Could not load saved config %s while reusing existing checkpoint. "
                            "Falling back to the input config. Details: %s",
                            candidate,
                            exc,
                        )
            config.model.cell_type_fp = model_dir / 'cell_type_list.txt'
            config.model.input_gene_list_fp = model_dir / 'input_gene_list.txt'
            return config
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config ───────────────────────────────────────────────────────
    _save_config(config, model_dir, config_file)

    # Train model
    trainer = VAEDeconTrainer(config)
    try:
        trained_config = trainer.train()
    except Exception as exc:
        _save_training_error_report(
            model_dir=model_dir,
            error=exc,
        )
        raise

    # ── Cleanup processed data ────────────────────────────────────────────
    # Keep the processed dataset cache so repeated ablations can reuse it.
    processed_dir = trainer._processed_training_set_dir
    if processed_dir:
        logger.info(f"Retaining processed training set directory for reuse: {processed_dir}")

    # Save final config after training (may have updates)
    try:
        final_config_path = model_dir / "config_final.yaml"
        trained_config.to_yaml(final_config_path)
        logger.info(f"Saved final config to {final_config_path}")
        trained_config.to_yaml(model_dir / "config.yaml")
        logger.info(f"Refreshed final config at {model_dir / 'config.yaml'}")
    except Exception as e:
        logger.warning(f"Could not save final config: {e}")

    # Plot loss curves
    log_file = model_dir / "metrics.csv"
    if log_file.exists():
        try:
            history_df = load_lightning_metrics(
                str(log_file),
                metric_cols=["train_loss_epoch", "val_loss", "lr-Adam"]
            )
            try:
                from ..plot.plot_nn import plot_loss
            except Exception as e:
                logger.warning(f"Could not import plotting utilities (skipping loss curve): {e}")
            else:
                plot_loss(history_df=history_df, output_dir=model_dir)
                logger.info(f"Saved loss curve to {model_dir}")
        except Exception as e:
            logger.warning(f"Could not plot loss curve: {e}")

    return trained_config
