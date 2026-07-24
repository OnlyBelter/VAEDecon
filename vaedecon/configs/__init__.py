"""
High-level configure classes for the VAEDecon model.
BaseModel (pydantic) -> BaseTrainerConfig -> TrainingConfig
BaseModel (pydantic) -> BaseModelConfig -> ModelConfig

VAEDeconConfig includes all other 4 configs: DataConfig, ModelConfig, TrainingConfig, EvaluationConfig
"""

from .default_config import VAEDeconConfig
from .default_config import DataConfig
from .default_config import ModelConfig
from .default_config import TrainingConfig
from .default_config import EvaluationConfig
from .default_config import GEPDatasetConfig
from .default_config import TestSetConfig
