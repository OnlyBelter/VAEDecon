"""
In this module are stored the main Neural Networks Architectures.
"""


from .base_architectures import BaseDecoder, BaseDiscriminator, BaseEncoder, BaseMetric
from .mlp.mlp import Encoder_MLP, Decoder_MLP
from .positional_encoding import PositionalEncoding

__all__ = [
    "BaseDecoder",
    "BaseEncoder",
    "BaseMetric",
    "BaseDiscriminator",
    "Encoder_MLP",
    "Decoder_MLP",
    "PositionalEncoding",
]
