import logging
from typing import Optional

import numpy as np
import pandas as pd
from vaedecon.models import BaseModelConfig
from vaedecon.models.base import PositionalEncoding, ModelOutput, EPS
from vaedecon.models.nn import EncoderMLP
from vaedecon.utility.read_file import ReadExp, read_gene_set

logger = logging.getLogger(__name__)


class EncoderPathNet(EncoderMLP):
    """
    Pathway-aware MLP encoder. Identical to EncoderMLP but uses pathway-specific
    input dimensions, hidden layer sizes, and dropout rates from args.
    """

    def __init__(
        self,
        args: BaseModelConfig,
        position_encoding: Optional[PositionalEncoding] = None,
    ):
        # Override pathway-specific params BEFORE super().__init__()
        # so that _build_layers() and _build_heads() use them directly.
        # We temporarily patch the relevant attributes on args so the
        # parent __init__ picks them up cleanly.
        self._pathway_input_dim    = args.input_dim_pathway
        self._pathway_hidden_dims  = getattr(args, 'encoder_hidden_dims_pathway', [1024, 512, 512])
        self._pathway_dropout_rate = args.encoder_dropout_rate_pathway

        super().__init__(args, position_encoding)

    def _build_layers(self) -> None:
        # Swap in pathway-specific params for the build step only
        self.input_dim    = self._pathway_input_dim
        self.hidden_dims  = self._pathway_hidden_dims
        self.dropout_rate = self._pathway_dropout_rate
        super()._build_layers()

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg['class_name'] = self.__class__.__name__
        cfg['params']['input_dim_pathway']            = self._pathway_input_dim
        cfg['params']['encoder_hidden_dims_pathway']  = self._pathway_hidden_dims
        cfg['params']['encoder_dropout_rate_pathway'] = self._pathway_dropout_rate
        return cfg


def get_pathway_profiles(
    x_obj,
    pathway_mask: pd.DataFrame,
    method: str = 'add_to_end',
    filtered_gene_list=None,
):
    """
    Compute pathway profiles from a gene expression matrix.

    :param x_obj: An instance of ReadExp containing the input gene expression
                  matrix. If in log-space, it will be converted to TPM in-place
                  before computation.
    :param pathway_mask: A DataFrame of shape (n_genes, n_pathways) used as the
                         projection matrix. Genes not present in x_obj will be
                         treated as all-zero rows.
    :param method: How to return pathway profiles:
                   - 'convert'    : replace x with pathway profiles (m × p)
                   - 'add_to_end' : append pathway profiles to x   (m × (n+p))
    :param filtered_gene_list: Only used when method='add_to_end'. If provided,
                               x is aligned to this gene list (with zero-filling)
                               and re-normalised to TPM before appending pathway
                               profiles.
    :return: A new ReadExp instance in log2(TPM+1) space.
    :raises ValueError: If an unsupported method is given.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    supported_methods = {'convert', 'add_to_end'}
    if method not in supported_methods:
        raise ValueError(
            f"Unsupported method '{method}'. Choose from {supported_methods}."
        )

    # ── Convert to TPM if needed ──────────────────────────────────────────
    if x_obj.file_type == 'log_space':
        x_obj.to_tpm()
    x = x_obj.get_exp()  # shape: (m, n)

    # ── Align pathway_mask to x genes ────────────────────────────────────
    common_genes = list(set(x.columns) & set(pathway_mask.index))
    genes_only_in_x = list(set(x.columns) - set(pathway_mask.index))

    logger.info('Genes in common with pathway mask : %d', len(common_genes))
    if genes_only_in_x:
        logger.info(
            'Genes only in expression matrix (zero-padded): %d',
            len(genes_only_in_x),
        )
        zero_rows = pd.DataFrame(
            np.zeros((len(genes_only_in_x), pathway_mask.shape[1])),
            index=genes_only_in_x,
            columns=pathway_mask.columns,
        )
        pathway_mask = pd.concat([pathway_mask, zero_rows], axis=0)

    # Reindex to match x column order exactly; fill any remaining gaps with 0
    pathway_mask = pathway_mask.reindex(x.columns, fill_value=0.0)

    # ── Compute pathway profiles ──────────────────────────────────────────
    x_pathway = x @ pathway_mask  # (m, n) × (n, p) → (m, p)

    if method == 'convert':
        x_out = x_pathway  # replace x entirely

    else:  # method == 'add_to_end'
        if filtered_gene_list is not None:
            # Align x to the filtered gene list BEFORE concatenation so that
            # x_pathway (already computed) and x_filtered share the same row index
            x_obj.align_with_gene_list(
                gene_list=filtered_gene_list, fill_not_exist=True
            )
            x = x_obj.get_exp()  # re-fetch aligned x; x_pathway is unchanged

        x_out = pd.concat([x, x_pathway], axis=1)  # (m, n'+p)

    # ── Log2 transform and wrap ───────────────────────────────────────────
    x_out = np.log2(x_out + 1)
    logger.info('Output expression matrix shape: %s', x_out.shape)

    return ReadExp(x_out, exp_type='log_space')


def get_pathway_mask(pathway_file_path: list[str]):
    return read_gene_set(pathway_file_path)