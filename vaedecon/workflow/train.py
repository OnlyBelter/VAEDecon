"""
Training pipeline for VAEDecon
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import random_split
from ..data import GEPDataset
from ..models.nn import EncoderMLP, DecoderMLP
from ..models.vae import VAEConfig
from ..trainers import BaseTrainerConfig, BaseTrainerL
from ..utility import set_output_dir, log_message, set_fig_style
from ..utility.read_file import load_or_compute_gene_mean_std
from .workflow import create_model, train_model, save_metadata
from ..configs.default_config import VAEDeconConfig

logger = logging.getLogger(__name__)


class VAEDeconTrainer:
    """VAEDecon Trainer"""

    def __init__(self, config: Optional[VAEDeconConfig] = None):
        """
        Initializes the trainer

        Parameters:
            config: VAEDeconConfig, using default VAEDeconConfig if None
        """
        self.config = config or VAEDeconConfig()
        self._setup_logging()
        self._setup_device()
        self._setup_directories()

    def _setup_logging(self):
        """Setting up logging"""
        console = logging.StreamHandler()
        logger.addHandler(console)
        logger.setLevel(logging.INFO)

    def _setup_device(self):
        """Setup computing device"""
        if self.config.training.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.config.training.device
        logger.info(f"Using device: {self.device}")

    def _setup_directories(self):
        """Set up output directories"""
        set_fig_style(font_family='Arial', font_size=8)

        if self.config.model.model_dir == '':
            self.result_dir = set_output_dir(
                output_dir=self.config.training.output_dir,
                naming_postfix=self.config.training.naming_postfix
            )
            self.model_dir = self.result_dir / 'final_model'
            self.config.model.model_dir = self.model_dir
        else:
            self.model_dir = self.config.model.model_dir
            self.result_dir = os.path.dirname(self.model_dir)

        # # Update file paths in model config
        self.config.model.input_gene_list_fp = self.model_dir / 'input_gene_list.txt'
        self.config.model.cell_type_fp = self.model_dir / 'cell_type_list.txt'

        logger.info(f"Results will be saved to: {self.result_dir}")

    def _prepare_data(self) -> Tuple[GEPDataset, any, any]:
        """Prepare training data"""
        logger.info("Loading and preparing data...")

        # All training set files
        training_file_paths = [
            self.config.data.simu_bulk_file_path,
            self.config.data.sct_file_path
        ]

        processed_training_set_dir = os.path.join(
            self.config.data.data_dir,
            f'processed_{len(training_file_paths)}_training_sets'
        )

        # Load dataset
        dataset = GEPDataset(
            file_paths=training_file_paths,
            scaling_by_constant=self.config.data.scaling_by_constant,
            remove_low_var_genes=self.config.data.remove_low_var_genes,
            processed_data_dir=processed_training_set_dir,
        )

        logger.info(f"Dataset shape: {dataset.data.shape}")

        # Split train/val sets
        train_set, val_set = random_split(
            dataset,
            [self.config.training.train_split, self.config.training.val_split]
        )

        logger.info(f"Train set: {len(train_set)}, Val set: {len(val_set)}")

        # Update input_dim in model config
        input_dim = dataset.data.shape[1]
        self.config.model.input_dim = (1, input_dim)

        # Save metadata
        vae_config = self._convert_to_vae_config()
        save_metadata(dataset=dataset, model_config=vae_config)

        # Calculate gene mean and std as features for GNN
        self._compute_gene_statistics(dataset, training_file_paths)

        return dataset, train_set, val_set

    def _compute_gene_statistics(self, dataset: GEPDataset, training_file_paths: list):
        """Calculate gene mean and std for each cell type"""
        logger.info("Computing gene statistics...")

        self.config.model.gene_mean_std_fp = Path(self.config.data.sct_gep_file_path).parent / (
                f"gene_mean_std_log2p1_scaled_{len(training_file_paths)}training_files.csv"
                if self.config.model.scaling_by_constant
                else f"gene_mean_std_log2p1_{len(training_file_paths)}training_files.csv"
            )

        gene_mean_std_df = load_or_compute_gene_mean_std(
            sct_gep_fp=self.config.data.sct_gep_file_path,
            gene_list=dataset.gene_list,
            cell_type_fp=self.config.model.cell_type_fp,
            input_gene_list_fp=self.config.model.input_gene_list_fp,
            scaling_by_constant=self.config.model.scaling_by_constant,
            log_fn=log_message,
            out_fp=Path(self.config.model.gene_mean_std_fp),
        )

    def _create_model(self):
        """Create model"""
        logger.info("Creating model...")

        # 转换为VAEConfig
        vae_config = self._convert_to_vae_config()

        model = create_model(
            model_config=vae_config,
            encoder_cls_name_list=self.config.model.encoders,
            decoder_cls=DecoderMLP
        )

        return model, vae_config

    def _convert_to_vae_config(self) -> VAEConfig:
        """Convert to VAEConfig"""
        # Set default PPI file path if not provided
        if not self.config.model.ppi_file_path:
            self.config.model.ppi_file_path = Path(self.config.data.data_dir) / 'PPI' / 'format_h_sapiens.csv'

        return VAEConfig(
            name='VAEConfig',
            input_dim=self.config.model.input_dim,
            latent_dim=self.config.model.latent_dim,
            n_cell_types=self.config.model.n_cell_types,
            using_positional_encoding=self.config.model.using_positional_encoding,
            input_gene_list_fp=self.config.model.input_gene_list_fp,
            cell_type_fp=self.config.model.cell_type_fp,
            gene_mean_std_fp=self.config.model.gene_mean_std_fp,
            scaling_by_constant=self.config.model.scaling_by_constant,
            encoder_hidden_dims=self.config.model.encoder_hidden_dims,
            decoder_hidden_dims=self.config.model.decoder_hidden_dims,
            encoder_dropout_rate=self.config.model.encoder_dropout_rate,
            decoder_dropout_rate=self.config.model.decoder_dropout_rate,
            fusion_hidden_dims=self.config.model.fusion_hidden_dims,
            fusion_dropout_rate=self.config.model.fusion_dropout_rate,
            predict_cell_prop=self.config.model.predict_cell_prop,
            loss_coefficient=self.config.model.loss_coefficient,
            gnn_n_genes=self.config.model.gnn_n_genes,
            gnn_inter_col_dim=self.config.model.gnn_inter_col_dim,
            gnn_embd_col_dim=self.config.model.gnn_embd_col_dim,
            gnn_lambda_cols=self.config.model.gnn_lambda_cols,
            gnn_num_layers=self.config.model.gnn_num_layers,
            gnn_drop_p=self.config.model.gnn_drop_p,
            ppi_file_path=self.config.model.ppi_file_path,
            gene_hidden_dim=self.config.model.gene_hidden_dim,
            encoders=self.config.model.encoders,
        )

    def _convert_to_trainer_config(self) -> BaseTrainerConfig:
        """Convert to TrainerConfig"""
        return BaseTrainerConfig(
            name='VAETrainerConfig',
            output_dir=self.config.training.output_dir,
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            steps_saving=self.config.training.steps_saving,
            num_epochs=self.config.training.num_epochs,
            optimizer_cls=self.config.training.optimizer_cls,
            n_early_stopping_patience=self.config.training.n_early_stopping_patience,
            devices=self.config.training.devices,
            debug_model=self.config.training.debug_model,
        )

    def train(self) -> Optional[VAEDeconConfig]:
        """
        Run the training process

        Return:
            VAEDecon configuration (maybe updated during training)
        """
        # Prepare data
        dataset, train_set, val_set = self._prepare_data()

        # Create model
        model, vae_config = self._create_model()

        # Create trainer configuration
        trainer_config = self._convert_to_trainer_config()

        # Train the model
        logger.info("Starting training...")
        train_model(
            model=model,
            train_set=train_set,
            val_set=val_set,
            config=trainer_config,
            device=self.device,
            trainer_cls=BaseTrainerL,
            result_dir=self.model_dir
        )

        logger.info(f"Training completed! Model saved to: {self.model_dir}")

        return self.config


def train_vaedecon(
        config: Optional[VAEDeconConfig] = None,
        config_file: Optional[str] = None
) -> Optional[VAEDeconConfig]:
    """
    Wrapper function to train VAEDecon model

    Parameters:
        config: VAEDecon configuration object, either config or config_file should be provided
        config_file: Config file path in YAML format, either config or config_file should be provided

    Returns:
        VAEDecon configuration (maybe updated during training)

    Examples:
        # Use default configuration
        model_dir = train_vaedecon()

        # Use config file
        model_dir = train_vaedecon(config_file='config.yaml')

        # Use custom configuration
        config = VAEDeconConfig()
        config.training.num_epochs = 500
        model_dir = train_vaedecon(config=config)
    """
    if config_file is not None:
        config = VAEDeconConfig.from_yaml(config_file)
    elif config is None:
        config = VAEDeconConfig()
    if config.model.model_dir is not None:
        model_dir = Path(config.model.model_dir)
    else:
        model_dir = Path(config.training.output_dir) / config.training.naming_postfix / 'final_model'
        config.model.model_dir = model_dir
    # Check if there is a file ending with .ckpt in the model_dir
    if model_dir and os.path.exists(model_dir):
        ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
        if ckpt_files:
            logger.info(f"Model checkpoint found in {model_dir}. Skipping training.")
            config.model.cell_type_fp = model_dir / 'cell_type_list.txt'
            config.model.input_gene_list_fp = model_dir / 'input_gene_list.txt'
            return config

    trainer = VAEDeconTrainer(config)
    return trainer.train()