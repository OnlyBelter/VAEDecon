"""
In this module are stored the main Neural Networks Architectures.
"""


from .mlp import EncoderMLP, DecoderMLP
from .fused_mlp_gnn import EncoderHybrid
from .res_mlp import EncoderResMLP, DecoderResMLP
from .transformer import GeneTransformerEncoder
from vaedecon.models.gnn.positional_encoding import PositionalEncoding

__all__ = [
    "EncoderMLP",
    "DecoderMLP",
    "PositionalEncoding",
    "EncoderHybrid",
    "EncoderResMLP",
    "DecoderResMLP",
    "GeneTransformerEncoder",
]
