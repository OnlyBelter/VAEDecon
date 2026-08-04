"""
Training pipeline for VAEDecon
"""
import os
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import random_split
from ..data import GEPDataset
from ..models.nn import EncoderMLP, DecoderMLP
# from ..models.vae import VAEConfig
from ..trainers import BaseTrainerL
from ..utility import set_output_dir, log_message, set_fig_style
from ..utility import load_or_compute_gene_mean_std, load_lightning_metrics, compute_gene_mean_std_from_pooled_sc_h5ad
from .workflow import create_model, train_model, save_metadata
from ..configs.default_config import VAEDeconConfig, TrainingConfig, ModelConfig, GEPDatasetConfig

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
        self._precessed_training_set_dir: Optional[Path] = None  # exposed for cleanup after training

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
            self.result_dir = set_output_dir(
                output_dir=self.config.training.output_dir,
                naming_postfix=self.config.training.naming_postfix
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

        # validate split ratios
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

        # Split train/val sets
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

        return dataset, train_set, val_set

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
                sct_gep_fp=str(self._resolve_gene_mean_std_sct_gep_path()),
                gene_list=dataset.gene_list,
                cell_type_fp=self.config.model.cell_type_fp,
                input_gene_list_fp=self.config.model.input_gene_list_fp,
                scaling_by_constant=self.config.data.scaling_by_constant,
                scaling_factor=self.config.data.scaling_factor,
                log_fn=log_message,
                out_fp=gene_mean_std_fp,
            )

    def _resolve_gene_mean_std_sct_gep_path(self) -> Path:
        dedicated_fp = self.config.data.gene_mean_std_sct_gep_file_path
        if dedicated_fp and str(dedicated_fp).strip() != "":
            return Path(dedicated_fp)
        return Path(self.config.data.sct_gep_file_path)

    def _build_gene_mean_std_output_path(self) -> Path:
        scaling_factor = self.config.data.scaling_factor
        model_dir = Path(self.config.model.model_dir)
        if self.config.data.scaling_by_constant:
            return model_dir / f"gene_mean_std_log2p1_scaled_by_{scaling_factor}.csv"
        return model_dir / "gene_mean_std_log2p1.csv"

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
        return TrainingConfig(
            name='VAETrainerConfig',
            output_dir=self.config.training.output_dir,
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            train_dataloader_num_workers=self.config.training.train_dataloader_num_workers,
            eval_dataloader_num_workers=self.config.training.eval_dataloader_num_workers,
            steps_saving=self.config.training.steps_saving,
            num_epochs=self.config.training.num_epochs,
            optimizer_cls=self.config.training.optimizer_cls,
            n_early_stopping_patience=self.config.training.n_early_stopping_patience,
            devices=self.config.training.devices,
            debug_model=self.config.training.debug_model,
            scheduler_cls=self.config.training.scheduler_cls,
            scheduler_params=self.config.training.scheduler_params,
            warmup_epochs=self.config.training.warmup_epochs,
            gradient_clip_val=self.config.training.gradient_clip_val,
            gene_stat_weight_schedule=self.config.training.gene_stat_weight_schedule,
            gene_stat_weight_schedule_epochs=self.config.training.gene_stat_weight_schedule_epochs,
            gene_mean_weight_end=self.config.training.gene_mean_weight_end,
            gene_std_weight_end=self.config.training.gene_std_weight_end,
            gene_std_weight_start=self.config.training.gene_std_weight_start,
            gene_mean_weight_start=self.config.training.gene_mean_weight_start,
            prog_bar_metrics=self.config.training.prog_bar_metrics,
        )

    def _build_gepdataset_config(self) -> GEPDatasetConfig:
        """
        Build a config dict for GEPDataset from self.config.data
        """

        # All training set files
        simu_paths = self.config.data.simu_bulk_file_path or []
        sct_paths  = self.config.data.sct_file_path or []
        training_file_paths = [p for p in simu_paths + sct_paths if p is not None]

        return GEPDatasetConfig(
            file_paths=training_file_paths,
            scaling_by_constant=self.config.data.scaling_by_constant,
            remove_low_var_genes=self.config.data.remove_low_var_genes,
            force_reprocess=self.config.data.force_reprocess,
            use_memmap=self.config.data.use_memmap,
            chunk_size=self.config.data.chunk_size,
            min_var=self.config.data.min_var,
            scaling_factor=self.config.data.scaling_factor,
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
            config.model.cell_type_fp       = model_dir / 'cell_type_list.txt'
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
