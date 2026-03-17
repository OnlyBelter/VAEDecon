"""
Default configuration for VAEDecon
"""
from dataclasses import dataclass, field
from .base_config import BaseTrainerConfig, BaseModelConfig, BaseConfig
from typing import List, Dict, Optional, Tuple, Any, Union, Literal
from pathlib import Path
from pydantic import Field, field_validator, model_validator, BaseModel


class LossCoefficient(BaseModel):
    beta: float = 2.0
    gamma: float = 0.005
    kld_type: Literal["ave", "sep"] = "ave"   # your code uses 'sep', not 'sum'
    cell_prop: float = 0.0
    weighting_gene_by_exp: bool = True
    weight_clamp_range: Tuple[float, float] = (0.2, 5.0)
    gene_mean_std_weight: float = 1.0
    z_score_reg_weight: float = 0.0

    @field_validator("beta", "gamma", "cell_prop", "gene_mean_std_weight", "z_score_reg_weight")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @model_validator(mode="after")
    def check_weight_range(self):
        mn, mx = self.weight_clamp_range
        if mn <= 0 or mn >= mx:
            raise ValueError("weight_clamp_range must satisfy 0 < min < max")
        return self

# @dataclass
class DataConfig(BaseConfig):
    """dataset configuration"""
    data_dir: str | Path = './datasets/'

    # Training data
    sct_file_path: list[str | Path] = Field(default_factory=list)
    simu_bulk_file_path: list[str | Path] = Field(default_factory=list)

    # Test data
    test_set_file_path: str | Path = ''
    test_set_sample2cell_id_file_path: str | Path = ''
    sct_gep_file_path: str | Path = ''  # Used for query sampled sctGEPs in test set

    # Additional files
    pred_cell_prop_file_path: Optional[str] = None
    cell_type2ave_exp_file_path: Optional[str] = None
    # PPI and Pathway file paths
    ppi_file_path: Optional[Path] = Field(
        default=None,
        description="Path to protein-protein interaction network file"
    )
    pathway_file_path: Optional[list[Path]] = Field(
        default=[],
        description="Path to Pathway files in .gmt format. Can provide multiple files for different pathway databases (e.g., KEGG, Reactome)."
    )

    @field_validator('ppi_file_path')
    @classmethod
    def validate_ppi_file(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate PPI file exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"PPI file not found: {v}")
        return v

    # Processing options
    # Scale input GEP data by a constant factor after log transformation (range of the input data will be (0, 1)).
    # This can help stabilize training and improve performance.
    scaling_by_constant: bool = Field(
        default=True,
        description='Whether to scale input GEP data by a constant factor after log transformation. '
    )
    scaling_factor: float = Field(
        default=20.0,
        description='Constant factor to scale input GEP data after log transformation when scaling_by_constant=True. '
                    'This can help stabilize training by normalizing the input into (0, 1) range'
                    ' and improve performance.'
    )

    remove_low_var_genes: bool = True  # If True, perform low-variance gene filtering.
    min_var: float = 1.0  # Minimum variance threshold for gene filtering (if remove_low_var_genes is True).
    force_reprocess: bool = False  # If True, ignore cache and re-run preprocessing.

    use_memmap: bool = True  # Use np.memmap for .npy cache files to reduce RAM pressure.
    chunk_size: int = 10000  # Chunk size for chunked transform. Increase for speed, decrease for memory.
    # If True, use .npz compressed cache files (smaller, typically slower).
    # Note: compressed .npz does not support true memmap behavior.
    compress: bool = False


# @dataclass(frozen=True)
class GEPDatasetConfig(DataConfig):
    """
    Configuration for GEPDataset.

    Why use a config object?
    - Improves readability (fewer long argument lists)
    - Prevents accidental positional argument bugs
    - Easier to serialize/store with experiment artifacts

    Args:
        file_paths:
            List of input file paths. Supported: .h5ad, .csv
        processed_data_dir:
            Cache directory for processed arrays and metadata.
            Must be provided for this implementation.
        force_reprocess:
            If True, ignore cache and re-run preprocessing.
        scaling_by_constant:
            If True, use `scaling_factor`;
            if float, use that value directly;
            if False, do not scale.
        scaling_factor:
            Default scaling divisor when scaling_by_constant is True.
        gene_list_file:
            Optional gene list for gene alignment/filtering.
        remove_low_var_genes:
            If True, perform low-variance gene filtering.
        min_var:
            Minimum variance threshold for gene filtering.
        cell_cell2ave_exp_file_path:
            Optional reference expression file for additional gene filtering.
        use_memmap:
            Use np.memmap for .npy cache files to reduce RAM pressure.
        chunk_size:
            Chunk size for chunked transform. Increase for speed, decrease for memory.
        compress:
            If True, use .npz compressed cache files (smaller, typically slower).
            Note: compressed .npz does not support true memmap behavior.
    """
    file_paths: List[Union[str, Path]]
    processed_data_dir: Optional[Union[str, Path]]
    force_reprocess: bool = False

    scaling_by_constant: Union[bool, float] = True
    scaling_factor: float = 20.0

    gene_list_file: Optional[Union[str, Path]] = None
    remove_low_var_genes: bool = False
    min_var: float = 1.0
    cell_cell2ave_exp_file_path: Optional[Union[str, Path]] = None

    use_memmap: bool = True
    chunk_size: int = 10000
    compress: bool = False


# @dataclass
class TrainingConfig(BaseTrainerConfig):
    """training configuration"""
    # Basic settings
    output_dir: str | Path = Path('./output/vae')
    naming_postfix: str = 'default'

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 512
    num_epochs: int = 1000

    # Early stopping
    n_early_stopping_patience: int = 15

    # Device settings
    devices: int = 1
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'

    # Optimizer
    optimizer_cls: str = 'Adam'

    # Saving
    steps_saving: int = 0

    # Debug
    debug_model: bool = False

    # Data split
    train_split: float = 0.8
    val_split: float = 0.2

    # Scheduler
    scheduler_cls: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None


class ModelConfig(BaseModelConfig):
    """Complete model configuration for VAE-based deconvolution.

    This configuration extends BaseModelConfig with settings for the
    hybrid encoder architecture, GNN components, and custom loss functions in the VAE model.

    Attributes:
        Architecture:
            input_dim: Input dimensions (channels, features).
            latent_dim: Latent space dimension per cell type.
            n_cell_types: Number of cell types to deconvolve.
            learn_gep_residual: Whether to learn GEP residuals compared to mean GEP of each cell type.

        Encoder/Decoder:
            encoder_hidden_dims: Hidden layer dimensions for encoder.
            decoder_hidden_dims: Hidden layer dimensions for decoder.
            encoder_dropout_rate: Dropout rates for encoder layers.
            decoder_dropout_rate: Dropout rates for decoder layers.
            encoders: List of encoder types (e.g., ['EncoderHybrid']).

        Fusion Layer:
            fusion_hidden_dims: Hidden dimensions for fusion layers.
            fusion_dropout_rate: Dropout rates for fusion layers.

        GNN Settings:
            gnn_n_genes: Number of genes in GNN.
            gnn_inter_col_dim: Intermediate dimension for cell embeddings.
            gnn_embd_col_dim: Final cell embedding dimension.
            gnn_lambda_cols: Weight for cell loss term.
            gnn_num_layers: Number of GNN layers.
            gnn_drop_p: Dropout probability in GNN.
            gene_hidden_dim: Hidden dimension for gene projection.

        Loss Configuration:
            loss_coefficient: Dictionary containing:
                - cell_prop: Weight for cell proportion loss
                - beta: Weight for reconstruction loss
                - gamma: Weight for regularization
                - kld_type: Type of KLD computation ('ave' or 'sum')
                - weighting_gene_by_exp: Weight genes by expression level
                - weight_clamp_range: Range to clamp gene weights
                - gene_mean_std_weight: Weight for gene mean/std loss

        File Paths:
            input_gene_list_fp: Path to input gene list.
            cell_type_fp: Path to cell type definitions.
            gene_mean_std_fp: Path to gene statistics.
            model_dir: Directory to save model checkpoints.

        Other Settings:
            predict_cell_prop: Whether to predict cell proportions.
            using_positional_encoding: Use positional encoding for cell types.
    """

    # ==================== Architecture ====================
    input_dim: Tuple[int, int] = Field(
        default=(1, 17834),
        description="Input dimensions (channels, features)"
    )
    latent_dim: int = Field(
        default=10,
        gt=0,
        description="Latent space dimension per cell type"
    )
    n_cell_types: int = Field(
        default=16,
        gt=0,
        description="Number of cell types to deconvolve"
    )

    # ==================== Encoder/Decoder ====================
    encoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [2048, 1024, 1024, 512],
        description="Hidden layer dimensions for encoder"
    )
    decoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [512, 1024, 1024, 2048],
        description="Hidden layer dimensions for decoder"
    )
    encoder_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.1, 0.0],
        description="Dropout rates for encoder layers"
    )
    decoder_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.1, 0.0],
        description="Dropout rates for decoder layers"
    )

    # ==================== Fusion Layer ====================
    fusion_hidden_dims: Tuple[int, ...] = Field(
        default=(512, 256),
        description="Hidden dimensions for fusion layers"
    )
    fusion_dropout_rate: Tuple[float, ...] = Field(
        default=(0.1, 0.0),
        description="Dropout rates for fusion layers"
    )

    # ==================== Encoder Types ====================
    encoders: List[str] = Field(
        default_factory=lambda: ['EncoderHybrid'],
        description="List of encoder types to use"
    )

    # ==================== Decoder Types ====================
    decoders: List[str] = Field(
        default_factory=lambda: ['DecoderMLP'],
        description="List of decoder types to use"
    )

    # ==================== Loss Coefficients ====================
    loss_coefficient: LossCoefficient = Field(
        default_factory=LossCoefficient,
        description="Coefficients for different loss components"
    )

    # ==================== GNN Settings ====================
    gnn_n_genes: int = Field(
        default=12596,
        gt=0,
        description="Number of genes in GNN"
    )
    gnn_inter_col_dim: int = Field(
        default=500,
        gt=0,
        description="Intermediate dimension for cell embeddings"
    )
    gnn_embd_col_dim: int = Field(
        default=30,
        gt=0,
        description="Final cell embedding dimension"
    )
    gnn_lambda_cols: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for cell loss term"
    )
    gnn_num_layers: int = Field(
        default=3,
        ge=1,
        description="Number of GNN layers"
    )
    gnn_drop_p: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Dropout probability in GNN"
    )
    gene_hidden_dim: int = Field(
        default=10,
        gt=0,
        description="Hidden dimension for gene projection"
    )

    # ==================== Pathway DNN Settings ====================
    input_dim_pathway: Tuple[int, int] = Field(
        default=(1, 17834),
        description="Input dimensions (channels, features)"
    )
    encoder_hidden_dims_pathway: List[int] = Field(
        default_factory=lambda: [2048, 1024, 1024, 512],
        description="Hidden layer dimensions for encoder"
    )
    encoder_dropout_rate_pathway: List[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.1, 0.0],
        description="Dropout rates for encoder layers"
    )

    # ==================== File Paths ====================
    input_gene_list_fp: Optional[Path] = Field(
        default=None,
        description="Path to gene list file (after preprocessing) for GEP-level reconstruction"
    )
    cell_type_fp: Optional[Path] = Field(
        default=None,
        description="Path to cell type definitions file"
    )
    gene_mean_std_fp: Optional[Path] = Field(
        default=None,
        description="Path to gene mean/std statistics file"
    )
    model_dir: Optional[Path] = Field(
        default=None,
        description="Directory to save model checkpoints and outputs"
    )

    # ==================== Other Settings ====================
    using_positional_encoding: bool = Field(
        default=False,
        description="Use positional encoding in latent space to distinguish cell types if True."
    )

    # ==================== Cell Proportion Prediction ====================
    predict_cell_prop: bool = Field(
        default=False,
        description="Whether to predict cell type proportions"
    )


    # Mask fraction for input dropout
    mask_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of input features (genes) to mask for dropout"
    )

    # Whether to learn GEP residual compared to mean GEP of cell types instead of full GEP
    learn_gep_residual: bool = Field(
        default=False,
        description="Whether to learn GEP residuals compared to the mean GEP of each cell type (instead of learning the full GEP)"
    )

    # ==================== Validators ====================

    @field_validator('input_dim')
    @classmethod
    def validate_input_dim(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate input dimensions are positive."""
        if len(v) != 2:
            raise ValueError(f"input_dim must be a tuple of length 2, got {len(v)}")
        if any(dim <= 0 for dim in v):
            raise ValueError(f"All input dimensions must be positive, got {v}")
        return v

    @field_validator('input_gene_list_fp', 'cell_type_fp', 'gene_mean_std_fp',
                     check_fields=False,
                     mode='before')
    @classmethod
    def validate_file_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate file paths exist if provided."""
        if v is not None:
            # Convert string to Path if needed
            if isinstance(v, str):
                if v == '':  # Handle empty string
                    return None
                v = Path(v)

            # Check if file exists (only warn, don't fail)
            # This allows config creation before files exist
            if not v.exists():
                import warnings
                warnings.warn(f"File path does not exist yet: {v}")

        return v

    @field_validator('model_dir')
    @classmethod
    def validate_model_dir(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate and create model directory if needed."""
        if v is not None:
            if isinstance(v, str):
                if v == '':
                    return None
                v = Path(v)

            # Create directory if it doesn't exist
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)

        return v

    @field_validator('encoder_hidden_dims', 'decoder_hidden_dims')
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        """Ensure all hidden dimensions are positive."""
        if not v:
            raise ValueError("Hidden dimensions list cannot be empty")
        if any(dim <= 0 for dim in v):
            raise ValueError(f"All hidden dimensions must be positive, got {v}")
        return v

    @field_validator('encoder_dropout_rate', 'decoder_dropout_rate')
    @classmethod
    def validate_dropout_rates(cls, v: List[float]) -> List[float]:
        """Ensure dropout rates are in [0, 1]."""
        if not v:
            raise ValueError("Dropout rate list cannot be empty")
        if any(rate < 0 or rate > 1 for rate in v):
            raise ValueError(f"Dropout rates must be in [0, 1], got {v}")
        return v

    @field_validator('fusion_dropout_rate')
    @classmethod
    def validate_fusion_dropout(cls, v: Tuple[float, ...]) -> Tuple[float, ...]:
        """Ensure fusion dropout rates are in [0, 1]."""
        if any(rate < 0 or rate > 1 for rate in v):
            raise ValueError(f"Fusion dropout rates must be in [0, 1], got {v}")
        return v

    @field_validator('encoders')
    @classmethod
    def validate_encoders(cls, v: List[str]) -> List[str]:
        """Validate encoder types."""
        if not v:
            raise ValueError("encoders list cannot be empty")

        valid_encoders = ['EncoderHybrid', 'EncoderMLP', 'EncoderSGNN', 'EncoderResMLP',
                          'GeneTransformerEncoder', 'EncoderPathNet']
        for encoder in v:
            if encoder not in valid_encoders:
                raise ValueError(
                    f"Unknown encoder type: {encoder}. "
                    f"Valid types: {valid_encoders}"
                )

        return v

    @model_validator(mode='after')
    def validate_architecture_consistency(self):
        """Ensure encoder/decoder architecture is consistent."""
        # Check encoder dimensions and dropout rates match
        if len(self.encoder_hidden_dims) != len(self.encoder_dropout_rate):
            raise ValueError(
                f"encoder_hidden_dims (len={len(self.encoder_hidden_dims)}) and "
                f"encoder_dropout_rate (len={len(self.encoder_dropout_rate)}) "
                f"must have the same length"
            )

        # Check decoder dimensions and dropout rates match
        if len(self.decoder_hidden_dims) != len(self.decoder_dropout_rate):
            raise ValueError(
                f"decoder_hidden_dims (len={len(self.decoder_hidden_dims)}) and "
                f"decoder_dropout_rate (len={len(self.decoder_dropout_rate)}) "
                f"must have the same length"
            )

        # Check fusion dimensions and dropout rates match
        if len(self.fusion_hidden_dims) != len(self.fusion_dropout_rate):
            raise ValueError(
                f"fusion_hidden_dims (len={len(self.fusion_hidden_dims)}) and "
                f"fusion_dropout_rate (len={len(self.fusion_dropout_rate)}) "
                f"must have the same length"
            )

        return self

    @model_validator(mode='after')
    def validate_cell_prop_consistency(self):
        """Ensure cell proportion prediction settings are consistent."""
        cell_prop_weight = self.loss_coefficient.cell_prop

        if cell_prop_weight > 0 and not self.predict_cell_prop:
            raise ValueError(
                f"loss_coefficient['cell_prop'] = {cell_prop_weight} > 0 "
                f"but predict_cell_prop=False. "
                f"Either set predict_cell_prop=True or set cell_prop to 0."
            )

        if self.predict_cell_prop and cell_prop_weight == 0:
            import warnings
            warnings.warn(
                "predict_cell_prop=True but loss_coefficient['cell_prop']=0. "
                "Cell proportion predictions will not affect training."
            )

        return self

    # @model_validator(mode='after')
    # def validate_gene_mean_std_requirement(self):
    #     """Ensure gene_mean_std_fp is provided when needed."""
    #     if self.loss_coefficient.get("gene_mean_std_weight", 0) > 0:
    #         if self.gene_mean_std_fp is None:
    #             raise ValueError(
    #                 "gene_mean_std_fp must be provided when "
    #                 "loss_coefficient['gene_mean_std_weight'] > 0"
    #             )
    #
    #     return self

    @model_validator(mode='after')
    def validate_gnn_consistency(self):
        """Ensure GNN settings are consistent with input dimensions."""
        # Check if number of genes matches input dimension
        if self.input_dim[1] != self.gnn_n_genes:
            import warnings
            warnings.warn(
                f"input_dim[1]={self.input_dim[1]} does not match "
                f"gnn_n_genes={self.gnn_n_genes}. "
                f"This may cause dimension mismatch errors."
            )

        return self

    # ==================== Helper Methods ====================

    def get_encoder_architecture(self) -> List[Tuple[int, float]]:
        """Get encoder architecture as list of (dim, dropout) tuples."""
        return list(zip(self.encoder_hidden_dims, self.encoder_dropout_rate))

    def get_decoder_architecture(self) -> List[Tuple[int, float]]:
        """Get decoder architecture as list of (dim, dropout) tuples."""
        return list(zip(self.decoder_hidden_dims, self.decoder_dropout_rate))

    def get_fusion_architecture(self) -> List[Tuple[int, float]]:
        """Get fusion architecture as list of (dim, dropout) tuples."""
        return list(zip(self.fusion_hidden_dims, self.fusion_dropout_rate))

    def summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "model_type": "VAE-Deconvolution",
            "input_shape": self.input_dim,
            "latent_dim": self.latent_dim,
            "n_cell_types": self.n_cell_types,
            "encoder_layers": len(self.encoder_hidden_dims),
            "decoder_layers": len(self.decoder_hidden_dims),
            "fusion_layers": len(self.fusion_hidden_dims),
            "gnn_layers": self.gnn_num_layers,
            "total_params_estimate": self.estimate_total_params(),
            "encoder_types": self.encoders,
            "predict_cell_prop": self.predict_cell_prop,
            "loss_settings": {
                "beta": self.loss_coefficient["beta"],
                "gamma": self.loss_coefficient["gamma"],
                "cell_prop_weight": self.loss_coefficient["cell_prop"],
            }
        }

    def estimate_total_params(self) -> int:
        """Estimate total number of model parameters."""
        total = 0

        # Encoder parameters
        prev_dim = self.input_dim[1]
        for dim in self.encoder_hidden_dims:
            total += prev_dim * dim + dim  # weights + bias
            prev_dim = dim

        # Latent layer
        total += prev_dim * (self.latent_dim * self.n_cell_types) * 2  # mu and logvar

        # Decoder parameters
        prev_dim = self.latent_dim * self.n_cell_types
        for dim in self.decoder_hidden_dims:
            total += prev_dim * dim + dim
            prev_dim = dim

        # Output layer
        total += prev_dim * self.input_dim[1] + self.input_dim[1]

        return total

    def validate_paths_exist(self) -> Dict[str, bool]:
        """Check which file paths exist."""
        return {
            "ppi_file_path": self.ppi_file_path.exists() if self.ppi_file_path else False,
            "input_gene_list_fp": self.input_gene_list_fp.exists() if self.input_gene_list_fp else False,
            "cell_type_fp": self.cell_type_fp.exists() if self.cell_type_fp else False,
            "gene_mean_std_fp": self.gene_mean_std_fp.exists() if self.gene_mean_std_fp else False,
            "model_dir": self.model_dir.exists() if self.model_dir else False,
        }


@dataclass
class EvaluationConfig:
    """evaluation configuration"""
    n_samples: int = 3
    plot_cell_proportions: bool = True
    plot_single_cell_gep: bool = True
    plot_bulk_gep: bool = True
    plot_latent_space: bool = True
    val_batch_size: int = 128
    remove_low_var_genes: bool = False

    # UMAP settings
    n_neighbors: int = 15
    min_dist: float = 0.1

    # Plotting
    figsize: Tuple[float, float] = (3.5, 3.5)
    rasterized: bool = True
    show_metrics: bool = True
    figure_format: str = 'png'  # 'png', 'svg', or 'pdf' etc.

    # Results
    save_reconstructed_gep: bool = True  # Whether to save reconstructed GEPs for all test samples


@dataclass
class VAEDeconConfig:
    """The complete configuration for VAEDecon"""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Creates configuration from a dictionary"""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {}))
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path):
        """Loads configuration from a YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: str | Path):
        """Saves the configuration to a YAML file"""
        import yaml
        from dataclasses import asdict
        config_dict = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)