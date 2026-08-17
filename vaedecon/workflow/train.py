"""
Training pipeline for VAEDecon
"""
import os
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import torch
from torch.utils.data import Subset, random_split
from ..data import GEPDataset
from ..models.nn import EncoderMLP, DecoderMLP
# from ..models.vae import VAEConfig
from ..trainers import BaseTrainerL
from ..utility import set_output_dir, log_message, set_fig_style
from ..utility import load_or_compute_gene_mean_std, load_lightning_metrics, compute_gene_mean_std_from_pooled_sc_h5ad
from ..utility import compute_training_sct_cross_sample_gene_var
from ..utility import create_h5ad_dataset
from ..utility.read_file import ReadH5AD
from .workflow import create_model, train_model, save_metadata
from ..configs.default_config import (
    VAEDeconConfig,
    TrainingConfig,
    ModelConfig,
    GEPDatasetConfig,
    TestSetConfig,
)

logger = logging.getLogger(__name__)

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

        self._processed_training_set_dir = Path(self.config.data.data_dir) / (
            f'processed_training_sets_{self.config.training.naming_postfix}'
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
        self.config.model = self._build_vae_config(
            training_file_paths=gep_dataset_config.file_paths,
            dataset=dataset
        )
        save_metadata(dataset=dataset, model_config=self.config.model)

        # TODO, only calculate gene mean/std when we need it, such as GNN or predict_gep_residual is true.
        # Calculate gene mean and std as features for GNN
        self._compute_gene_statistics(dataset, self.config.model.gene_mean_std_fp)

        # Export training SCT cross-sample gene variance CSV (only if new loss is enabled)
        cross_var_fp = self._prepare_and_save_training_sct_cross_sample_gene_var()
        if cross_var_fp is not None:
            self.config.model.training_sct_cross_sample_gene_var_fp = cross_var_fp

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

    def _build_vae_config(self, training_file_paths, dataset: GEPDataset) -> ModelConfig:
        """
        Build a ModelConfig from self.config (model + data sections).

        """
        input_dim = dataset.data.shape[1]  # same as n_genes in each GEP
        n_genes = input_dim
        gene_mean_std_fp = self._build_gene_mean_std_output_path()

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
            cell_prop_activation_function=self.config.model.cell_prop_activation_function,
            cancer_cell_type_name=self.config.model.cancer_cell_type_name,
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

    def _build_gepdataset_config(self) -> GEPDatasetConfig:
        """
        Build a config dict for GEPDataset from self.config.data
        """

        # All training set files
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

        return GEPDatasetConfig(
            file_paths=training_file_paths,
            scaling_by_constant=self.config.data.scaling_by_constant,
            remove_low_var_genes=self.config.data.remove_low_var_genes,
            force_reprocess=self.config.data.force_reprocess,
            use_memmap=self.config.data.use_memmap,
            chunk_size=self.config.data.chunk_size,
            min_var=self.config.data.min_var,
            scaling_factor=self.config.data.scaling_factor,
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
    trained_config = trainer.train()

    # ── Cleanup processed data ────────────────────────────────────────────
    # Reuse the path already computed inside the trainer
    processed_dir = trainer._processed_training_set_dir
    if processed_dir and processed_dir.exists():
        try:
            shutil.rmtree(processed_dir)
            logger.info(f"Deleted processed training set directory: {processed_dir}")
        except Exception as exc:
            logger.warning(f"Could not delete processed training set directory: {exc}")

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
