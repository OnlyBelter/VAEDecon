from typing import List, Dict, Optional, Any
from pathlib import Path
from pydantic import Field, field_validator, model_validator

from ...models.base import BaseModelConfig


class VAEConfig(BaseModelConfig):
    """VAE configuration class.

    This configuration defines the architecture and hyperparameters for the
    Variational Autoencoder (VAE) model used in cell type deconvolution.

    Attributes:
        Basic Settings:
            input_dim: Input data dimensions (inherited from BaseModelConfig).
            latent_dim: Latent space dimension (inherited from BaseModelConfig).
            reconstruction_loss: Reconstruction loss type ['bce', 'mse'] (inherited).
            using_positional_encoding: Use positional encoding to distinguish cell types.
            scaling_by_constant: Scale input GEP data by constant factor (20 by default).
            predict_cell_prop: Whether to predict cell type proportions.

        File Paths:
            input_gene_list_fp: Path to gene list file (after preprocessing).
            cell_type_fp: Path to cell types file for reconstruction.
            gene_mean_std_fp: Path to mean/std of gene expression across cell types.

        Architecture:
            encoder_hidden_dims: Hidden layer dimensions for encoder.
            decoder_hidden_dims: Hidden layer dimensions for decoder.
            encoder_dropout_rate: Dropout rates for each encoder layer.
            decoder_dropout_rate: Dropout rates for each decoder layer.
            encoders: List of encoder types to use.

        Loss Coefficients:
            loss_coefficient: Weights for each loss term (recon_decoder,
                recon_convolution, kld, cell_prop). Must sum to 1.0.
    """

    # Basic settings
    using_positional_encoding: bool = Field(
        default=False,
        description="Use positional encoding in latent space to distinguish cell types"
    )
    scaling_by_constant: bool = Field(
        default=True,
        description="Scale input GEP data by constant factor (20 by default)"
    )
    predict_cell_prop: bool = Field(
        default=False,
        description="Predict cell type proportions"
    )

    # File paths
    input_gene_list_fp: Optional[Path] = Field(
        default=None,
        description="Path to gene list file (after preprocessing) for GEP-level reconstruction"
    )
    cell_type_fp: Optional[Path] = Field(
        default=None,
        description="Path to cell types file for reconstruction"
    )
    gene_mean_std_fp: Optional[Path] = Field(
        default=None,
        description="Path to mean/std of gene expression values across cell types in SCT dataset"
    )

    # Loss coefficients
    loss_coefficient: Dict[str, Any] = Field(
        default_factory=lambda: {
            "beta": 2,  # beta parameter for KLD loss, beta-VAE
            "gamma": 0.005,  # gamma parameter for the repulsion loss
            "kld_type": "ave",  # KL divergence loss
            "cell_prop": 0,  # Cell type proportion prediction loss
            "weighting_gene_by_exp": True,  # whether to weight the gene loss by the expression value across cell types
            'weight_clamp_range': (0.2, 5.0), # the range of the weights for the genes across cell types
            'gene_mean_std_weight': 1.0, # the weight for the gene mean and std loss
        },
        description="Coefficients for each term in total loss function"
    )

    # Encoder architecture
    encoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [1024, 512, 512],
        description="Hidden layer dimensions for encoder"
    )
    encoder_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.1, 0.1, 0.1],
        description="Dropout rates for each encoder layer"
    )

    # Decoder architecture
    decoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [512, 512, 1024],
        description="Hidden layer dimensions for decoder"
    )
    decoder_dropout_rate: List[float] = Field(
        default_factory=lambda: [0.1, 0.1, 0.1],
        description="Dropout rates for each decoder layer"
    )

    # Encoder types
    encoders: Optional[List[str]] = Field(
        default=None,
        description="List of encoder types to use (e.g., ['mlp', 'gnn'])"
    )

    # Decoder types
    decoders: Optional[List[str]] = Field(
        default=None,
        description="List of decoder types to use (e.g., ['mlp', 'res_mlp'])"
    )

    # ==================== Validators ====================

    @field_validator('input_gene_list_fp', 'cell_type_fp', 'gene_mean_std_fp', check_fields=False)
    @classmethod
    def validate_file_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate that file paths exist if provided."""
        if v is not None:
            if not v.exists():
                Warning(f"File not found: {v}")
            if not v.is_file():
                Warning(f"Path is not a file: {v}")
        return v

    @field_validator('encoder_hidden_dims', 'decoder_hidden_dims')
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        """Ensure all hidden dimensions are positive."""
        if not v:
            raise ValueError("Hidden dimensions list cannot be empty")
        if any(dim <= 0 for dim in v):
            raise ValueError("All hidden dimensions must be positive")
        return v

    @field_validator('encoder_dropout_rate', 'decoder_dropout_rate')
    @classmethod
    def validate_dropout_rates(cls, v: List[float]) -> List[float]:
        """Ensure dropout rates are in [0, 1]."""
        if not v:
            raise ValueError("Dropout rate list cannot be empty")
        if any(rate < 0 or rate > 1 for rate in v):
            raise ValueError("Dropout rates must be in [0, 1]")
        return v

    @field_validator('loss_coefficient')
    @classmethod
    def validate_loss_coefficient(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate loss coefficients."""
        required_keys = {"beta", "gamma", "kld_type", "cell_prop", "weighting_gene_by_exp",
                         "weight_clamp_range", "gene_mean_std_weight"}

        # Check required keys
        if set(v.keys()) != required_keys:
            raise ValueError(
                f"loss_coefficient must contain exactly these keys: {required_keys}, "
                f"got: {set(v.keys())}"
            )

        # Check all values are non-negative
        if any(coef < 0 for coef in v.values() if type(coef) in [int, float]):
            raise ValueError("All loss coefficients must be non-negative")

        # # Check sum is close to 1.0 (allow small floating point errors)
        # total = sum(v.values())
        # if abs(total - 1.0) > 1e-6:
        #     raise ValueError(
        #         f"Loss coefficients should sum to 1.0, got {total:.6f}. "
        #         f"Current values: {v}"
        #     )

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

        # Check if cell_prop loss is used but prediction is disabled
        if self.loss_coefficient.get("cell_prop", 0) > 0 and not self.predict_cell_prop:
            raise ValueError(
                "loss_coefficient['cell_prop'] > 0 but predict_cell_prop=False. "
                "Either set predict_cell_prop=True or set cell_prop coefficient to 0."
            )

        return self

    # @model_validator(mode='after')
    # def validate_gene_mean_std_requirement(self):
    #     """Ensure gene_mean_std_path is provided when scaling is enabled."""
    #     if self.scaling_by_constant and self.gene_mean_std_path is None:
    #         raise ValueError(
    #             "gene_mean_std_path must be provided when scaling_by_constant=True"
    #         )
    #     return self
