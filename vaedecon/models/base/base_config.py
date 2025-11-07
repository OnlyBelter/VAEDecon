from typing import Tuple, Optional, Literal
from pathlib import Path

from vaedecon.config import BaseConfig
from pydantic import Field, field_validator


class BaseModelConfig(BaseConfig):
    """Base configuration for VAE deconvolution models.

    This configuration defines the architecture and hyperparameters for the model,
    including encoder/decoder settings, GNN parameters, and loss functions.

    Attributes:
        input_dim: Input data dimensions (channels, x_dim, y_dim). None for auto-detection.
        latent_dim: Dimension of latent space for each cell type.
        n_cell_types: Number of cell types to deconvolve.
        uses_default_encoder: Whether to use the default encoder architecture.
        uses_default_decoder: Whether to use the default decoder architecture.
        reconstruction_loss: Type of reconstruction loss ('bce' or 'mse').

        GNN Parameters:
        gnn_n_genes: Number of genes in the graph.
        gnn_inter_col_dim: Intermediate layer dimension for cell embeddings.
        gnn_embd_col_dim: Final cell embedding dimension (before mu layer).
        gnn_lambda_cols: Weight for cell loss term.
        gnn_num_layers: Number of GNN layers.
        gnn_drop_p: Dropout probability in GNN layers.

        PPI and Gene Parameters:
        ppi_file_path: Path to protein-protein interaction network file.
        gene_hidden_dim: Dimension to project each gene expression value.
        biogrid_flag: If True, use only ["Source", "Target"] columns from PPI network.

        Fusion Model Parameters:
        fusion_hidden_dims: Hidden layer dimensions for fusion model (MLP + SGNN).
        fusion_dropout_rate: Dropout rates for each fusion layer.
    """

    # Basic architecture
    input_dim: Optional[Tuple[int, ...]] = Field(
        default=None,
        description="Input data dimensions (channels, x_dim, y_dim)"
    )
    latent_dim: int = Field(
        default=10,
        gt=0,
        description="Latent space dimension for each cell type"
    )
    n_cell_types: int = Field(
        default=10,
        gt=0,
        description="Number of cell types to deconvolve"
    )
    uses_default_encoder: bool = Field(
        default=True,
        description="Use default encoder architecture"
    )
    uses_default_decoder: bool = Field(
        default=True,
        description="Use default decoder architecture"
    )
    reconstruction_loss: Literal["bce", "mse"] = Field(
        default="mse",
        description="Reconstruction loss type"
    )

    # GNN parameters
    gnn_n_genes: int = Field(
        default=5000,
        gt=0,
        description="Number of genes in the graph"
    )
    gnn_inter_col_dim: int = Field(
        default=500,
        gt=0,
        description="Intermediate layer dimension for cell embeddings"
    )
    gnn_embd_col_dim: int = Field(
        default=50,
        gt=0,
        description="Final cell embedding dimension (before mu layer)"
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
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout probability in GNN"
    )

    # PPI and gene parameters
    ppi_file_path: Optional[Path] = Field(
        default=None,
        description="Path to PPI network file"
    )
    gene_hidden_dim: int = Field(
        default=10,
        gt=0,
        description="Dimension to project each gene expression value"
    )
    biogrid_flag: bool = Field(
        default=False,
        description="Use only Source/Target columns from PPI network"
    )

    # Fusion model parameters
    fusion_hidden_dims: Optional[Tuple[int, ...]] = Field(
        default=(1024,),
        description="Hidden dimensions for fusion model (MLP + SGNN)"
    )
    fusion_dropout_rate: Optional[Tuple[float, ...]] = Field(
        default=(0.1,),
        description="Dropout rates for fusion layers"
    )

    @field_validator('ppi_file_path')
    @classmethod
    def validate_ppi_file(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate PPI file exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"PPI file not found: {v}")
        return v

    @field_validator('fusion_hidden_dims')
    @classmethod
    def validate_positive_values(cls, v: Optional[Tuple]) -> Optional[Tuple]:
        """Ensure all values in tuples are positive."""
        if v is not None:
            if any(val <= 0 for val in v):
                raise ValueError("All values must be positive")
        return v

    @field_validator('fusion_dropout_rate')
    @classmethod
    def validate_dropout_range(cls, v: Optional[Tuple[float, ...]]) -> Optional[Tuple[float, ...]]:
        """Ensure dropout rates are in [0, 1]."""
        if v is not None:
            if any(rate < 0 or rate > 1 for rate in v):
                raise ValueError("Dropout rates must be in [0, 1]")
        return v

    @field_validator('fusion_hidden_dims', 'fusion_dropout_rate')
    @classmethod
    def validate_matching_lengths(cls, v, info):
        """Ensure fusion_hidden_dims and fusion_dropout_rate have matching lengths."""
        # This validator runs after both fields are set
        if 'fusion_hidden_dims' in info.data and 'fusion_dropout_rate' in info.data:
            dims = info.data['fusion_hidden_dims']
            rates = info.data['fusion_dropout_rate']
            if dims is not None and rates is not None:
                if len(dims) != len(rates):
                    raise ValueError(
                        f"fusion_hidden_dims (len={len(dims)}) and "
                        f"fusion_dropout_rate (len={len(rates)}) must have same length"
                    )
        return v


class EnvironmentConfig(BaseConfig):
    """Configuration for Python environment settings.

    Attributes:
        python_version: Required Python version for the project.
    """

    python_version: str = Field(
        default="3.8",
        pattern=r'^\d+\.\d+(\.\d+)?$',  # Validate version format
        description="Required Python version (e.g., '3.8', '3.10.5')"
    )
