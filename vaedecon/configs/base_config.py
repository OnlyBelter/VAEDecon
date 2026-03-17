import os
import json
import logging
import warnings
import torch.nn as nn
from typing import Tuple, Optional, Literal, Union, Any, Dict
from pathlib import Path

from pydantic import Field, field_validator, BaseModel, ConfigDict, ValidationError, field_serializer
# Configure logging
logger = logging.getLogger(__name__)


class BaseConfig(BaseModel):
    """
    Base configuration class providing JSON/dict serialisation, deserialisation,
    and Path-aware saving for all VAEDecon config subclasses.

    All subclasses automatically have their `name` field set to the class name.
    Extra fields are forbidden to catch typos early.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        arbitrary_types_allowed=True,
    )

    name: str = ""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.name = self.__class__.__name__

    # ── Deserialisation ───────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Create an instance from a plain Python dictionary.

        Args:
            config_dict: Dictionary of field names → values.

        Returns:
            A validated instance of this config class.

        Raises:
            ValidationError: If the dictionary contains invalid values.
            TypeError: If unexpected keyword arguments are passed.
        """
        try:
            return cls(**config_dict)
        except (ValidationError, TypeError) as e:
            logger.error(f"Failed to create {cls.__name__} from dict: {e}")
            raise  # Bare raise to preserve original traceback and error type

    @classmethod
    def _dict_from_json(cls, json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        """
        Load a plain dictionary from a JSON file

        Args:
            json_path: Path to the JSON file

        Returns:
            Dictionary containing the configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            TypeError: If file is not valid JSON
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise TypeError(
                f"File {json_path} is not valid JSON.\n"
                f"Catch Exception {type(e).__name__} with message: " + str(e)
            ) from e

    @classmethod
    def from_json_file(cls, json_path: Union[str, os.PathLike]) -> "BaseConfig":
        """
        Create an instance from a JSON config file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            A validated instance of this config class.
        """
        config_dict = cls._dict_from_json(json_path)

        config_name = config_dict.get("name")
        if config_name and cls.__name__ != config_name:
            warnings.warn(
        f"Loading a `{cls.__name__}` config from a file that was saved as `{config_name}`. Fields may not match.",
                UserWarning,
                stacklevel=2,
            )

        return cls.from_dict(config_dict)

    # ── Serialisation ─────────────────────────────────────────────────────

    @field_serializer('*', when_used='json')
    def serialize_paths(self, value: Any) -> Any:
        """Convert Path objects to strings when serializing to JSON."""
        if isinstance(value, Path):
            return str(value)
        return value

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-safe Python dictionary.
        Path objects are converted to strings automatically.

        Returns:
            A plain dictionary of all config fields.
        """
        return self.model_dump(mode='json')

    def to_json_string(self, indent: int = 4) -> str:
        """
        Serialise to a JSON-formatted string.

        Args:
            indent: Number of spaces for indentation (default 4).

        Returns:
            A JSON string representation of this config.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, dir_path: Union[str, os.PathLike], filename: str) -> None:
        """
        Save this configuration to a ``.json`` file.

        Args:
            dir_path: Directory in which to save the file.
            filename: File name (the ``.json`` extension is added if absent).

        Raises:
            OSError: If the file cannot be written.
        """
        dir_p = Path(dir_path)
        dir_p.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        file_path = dir_p / filename

        compact_json_string = self.to_json_string(indent=4)
        try:
            # Parse the compact JSON string back into a Python object
            python_obj = json.loads(compact_json_string)

            # Now dump this Python object to the file with indentation
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(python_obj, fp, indent=4, ensure_ascii=False)
            logger.info(f"Saved configuration to {file_path}")

        except OSError as e:
            # This can happen if self.to_json_string() doesn't return valid JSON
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving configuration to {file_path}: {e}")
            raise  # Re-raise if the method itself is missing/problematic


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

    # # Pathway network parameters (if using pathway-aware encoder)
    # input_dim_pathway: Optional[tuple[int, ...]] = Field(
    #     default=None,
    #     description="Input data dimensions for Pathway network (channels, x_dim, y_dim)"
    # )
    # pathway_file_path: list[Path] = Field(
    #     default=[],
    #     description="Path to Pathway (gene set) files in .gmt format"
    # )

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


class BaseTrainerConfig(BaseConfig):
    """
    BaseTrainer config class stating the main training arguments.

    Parameters:

        output_dir (str): The directory where model checkpoints, configs and final
            model will be stored. Default: None.
        per_device_train_batch_size (int): The number of training samples per batch and per device.
            Default 64
        per_device_eval_batch_size (int): The number of evaluation samples per batch and per device.
            Default 64
        num_epochs (int): The maximal number of epochs for training. Default: 100
        train_dataloader_num_workers (int): Number of subprocesses to use for train data loading.
            0 means that the data will be loaded in the main process. Default: 0
        eval_dataloader_num_workers (int): Number of subprocesses to use for evaluation data
            loading. 0 means that the data will be loaded in the main process. Default: 0
        optimizer_cls (str): The name of the `torch.optim.Optimizer` used for
            training. Default: :class:`~torch.optim.Adam`.
        optimizer_params (dict): A dict containing the parameters to use for the
            `torch.optim.Optimizer`. If None, uses the default parameters. Default: None.
        scheduler_cls (str): The name of the `torch.optim.lr_scheduler` used for
            training. If None, no scheduler is used. Default None.
        scheduler_params (dict): A dict containing the parameters to use for the
            `torch.optim.le_scheduler`. If None, uses the default parameters. Default: None.
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-4
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch.
            Default: None
        steps_predict (int): A prediction using the best model will be run every `steps_predict`
            epoch. Default: None
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False
        seed (int): The random seed for reproducibility
        amp (bool): Whether to use auto mixed precision in training. Default: False
    """

    output_dir: Union[str, Path, None] = None
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    num_epochs: int = 100
    train_dataloader_num_workers: Optional[int] = None  # None means it will be set depending on the system and cpus
    eval_dataloader_num_workers: Optional[int] = None
    optimizer_cls: str = "Adam"
    optimizer_params: Union[dict, None] = None
    scheduler_cls: Union[str, None] = None
    scheduler_params: Union[dict, None] = None
    learning_rate: float = 1e-4
    steps_saving: Union[int, None] = None
    steps_predict: Union[int, None] = None
    keep_best_on_train: bool = False
    seed: int = 8
    # no_cuda: bool = False
    # world_size: int = field(default=-1)
    # local_rank: int = field(default=-1)
    # rank: int = field(default=-1)
    # dist_backend: str = field(default="nccl")
    # master_addr: str = field(default="localhost")
    # master_port: str = field(default="12345")
    amp: bool = False
    # The number of epochs to wait before stopping the training if no improvement is observed.
    n_early_stopping_patience: int = 5
    devices: Union[int, str] = 1  # the number of gpus to use for training
    debug_model: bool = False  # if True, the model will be trained on a small subset of the data for debugging purposes

    def __post_init__(self):
        """Check compatibility and sets up distributed training"""
        super().__post_init__()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank == -1 and env_local_rank != -1:
            self.local_rank = env_local_rank

        env_world_size = int(os.environ.get("WORLD_SIZE", -1))
        if self.world_size == -1 and env_world_size != -1:
            self.world_size = env_world_size

        env_rank = int(os.environ.get("RANK", -1))
        if self.rank == -1 and env_rank != -1:
            self.rank = env_rank

        env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
        if self.master_addr == "localhost" and env_master_addr != "localhost":
            self.master_addr = env_master_addr
        os.environ["MASTER_ADDR"] = self.master_addr

        env_master_port = os.environ.get("MASTER_PORT", "12345")
        if self.master_port == "12345" and env_master_port != "12345":
            self.master_port = env_master_port
        os.environ["MASTER_PORT"] = self.master_port

        try:
            import torch.optim as optim

            optimizer_cls = getattr(optim, self.optimizer_cls)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to import `{self.optimizer_cls}` optimizer from 'torch.optim'. "
                "Check spelling and that it is part of 'torch.optim.Optimizers.'"
            )
        if self.optimizer_params is not None:
            try:
                optimizer = optimizer_cls(
                    nn.Linear(2, 2).parameters(),
                    lr=self.learning_rate,
                    **self.optimizer_params,
                )
            except TypeError as e:
                raise TypeError(
                    "Error in optimizer's parameters. Check that the provided dict contains only "
                    f"keys and values suitable for `{optimizer_cls}` optimizer. "
                    f"Got {self.optimizer_params} as parameters.\n"
                    f"Exception raised: {type(e)} with message: " + str(e)
                ) from e
        else:
            optimizer = optimizer_cls(
                nn.Linear(2, 2).parameters(), lr=self.learning_rate
            )

        if self.scheduler_cls is not None:
            try:
                import torch.optim.lr_scheduler as schedulers

                scheduder_cls = getattr(schedulers, self.scheduler_cls)
            except AttributeError as e:
                raise AttributeError(
                    f"Unable to import `{self.scheduler_cls}` scheduler from "
                    "'torch.optim.lr_scheduler'. Check spelling and that it is part of "
                    "'torch.optim.lr_scheduler.'"
                )

            if self.scheduler_params is not None:
                try:
                    scheduder_cls(optimizer, **self.scheduler_params)
                except TypeError as e:
                    raise TypeError(
                        "Error in scheduler's parameters. Check that the provided dict contains only "
                        f"keys and values suitable for `{scheduder_cls}` scheduler. "
                        f"Got {self.scheduler_params} as parameters.\n"
                        f"Exception raised: {type(e)} with message: " + str(e)
                    ) from e

        if self.no_cuda:
            self.amp = False
