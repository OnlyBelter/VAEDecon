r"""EMT Decode"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8 fallback
    raise ImportError("importlib.metadata is not available. Please use Python 3.8 or higher.")

try:
    __version__ = version("VAEDecon")
except PackageNotFoundError:
    __version__ = "0.3.3-dev"

from vaedecon.workflow import train_vaedecon, predict_vaedecon
from vaedecon.workflow import VAEDeconPredictor, VAEDeconTrainer
from .configs.default_config import (
    VAEDeconConfig,
    DataConfig,
    TrainingConfig,
    ModelConfig,
    EvaluationConfig
)

__all__ = [
    'train_vaedecon',
    'predict_vaedecon',
    'VAEDeconTrainer',
    'VAEDeconPredictor',
    'VAEDeconConfig',
    'DataConfig',
    'TrainingConfig',
    'ModelConfig',
    'EvaluationConfig',
]
