r"""EMT Decode"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8 fallback
    raise ImportError("importlib.metadata is not available. Please use Python 3.8 or higher.")

try:
    __version__ = version("VAEDecon")
except PackageNotFoundError:
    __version__ = "0.4.0-dev"

from .configs.default_config import (
    VAEDeconConfig,
    DataConfig,
    TrainingConfig,
    ModelConfig,
    EvaluationConfig
)

__all__ = [
    'VAEDeconConfig',
    'DataConfig',
    'TrainingConfig',
    'ModelConfig',
    'EvaluationConfig',
    'train_vaedecon',
    'predict_vaedecon',
    'VAEDeconTrainer',
    'VAEDeconPredictor',
]


def __getattr__(name: str):
    if name in {"train_vaedecon", "predict_vaedecon", "VAEDeconPredictor", "VAEDeconTrainer"}:
        from vaedecon.workflow import train_vaedecon, predict_vaedecon, VAEDeconPredictor, VAEDeconTrainer
        return {
            "train_vaedecon": train_vaedecon,
            "predict_vaedecon": predict_vaedecon,
            "VAEDeconPredictor": VAEDeconPredictor,
            "VAEDeconTrainer": VAEDeconTrainer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
