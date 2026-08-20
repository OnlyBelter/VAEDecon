from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from ...configs import ModelConfig, DataConfig
from ...models.base import (
    BaseEncoder,
    EPS,
    ModelOutput,
    build_cell_prop_from_head_output,
    get_cell_prop_head_output_dim,
    has_usable_labels,
    resolve_cancer_cell_type_index_from_config,
)
from ...utility import log_exp2cpm_tensor
from ...utility.read_file import read_gene_set


def _build_norm(normalization: Optional[str], n_features: int) -> Optional[nn.Module]:
    if normalization == "batch_normalization":
        return nn.BatchNorm1d(n_features)
    if normalization == "layer_normalization":
        return nn.LayerNorm(n_features)
    return None


class _DenseNormBlock(nn.Module):
    """Linear layer with optional normalization and ReLU, following DeSide."""

    def __init__(self, in_features: int, out_features: int, normalization: Optional[str], use_norm: bool):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=not use_norm)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        self.norm = _build_norm(normalization, out_features) if use_norm else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.activation(x)


class _FeatureBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int],
        dropout_rates: list[float],
        normalization: Optional[str],
        normalization_layer: list[int],
    ):
        super().__init__()
        self.input_norm = None
        if normalization is not None and normalization_layer[0] == 1:
            self.input_norm = _build_norm(normalization, input_dim)

        blocks = []
        dropouts = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_units):
            use_norm = bool(normalization is not None and normalization_layer[i + 1] == 1)
            blocks.append(_DenseNormBlock(in_dim, out_dim, normalization, use_norm))
            dropouts.append(nn.Dropout(float(dropout_rates[i])) if float(dropout_rates[i]) > 0 else nn.Identity())
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        for block, dropout in zip(self.blocks, self.dropouts):
            x = block(x)
            x = dropout(x)
        return x


class DeSideCellPropPredictor(BaseEncoder):
    """
    DeSide-style predictor branch for high-accuracy cell proportions and bulk context.

    This branch intentionally does not emit latent posterior tensors; it is routed only
    for `cell_prop` and `bulk_context_feature`.
    """

    def __init__(
        self,
        args: ModelConfig,
        data_config: DataConfig | None = None,
        position_encoding=None,
    ):
        del position_encoding
        super().__init__()
        self.args = args
        self.data_config = data_config
        self.predict_cell_prop = args.predict_cell_prop
        self.n_cell_types = args.n_cell_types
        self.cell_prop_activation_function = args.cell_prop_activation_function
        self.cancer_cell_type_index = None
        if self.cell_prop_activation_function == "sigmoid":
            self.cancer_cell_type_index = resolve_cancer_cell_type_index_from_config(args)

        self.normalization = getattr(args, "deside_normalization", "layer_normalization")
        self.normalization_layer = list(getattr(args, "deside_normalization_layer", [0, 0, 1, 1, 1, 1]))
        self.pathway_enabled = bool(getattr(args, "deside_pathway_network", True))

        self.gep_hidden_dims = list(getattr(args, "deside_hidden_dims", [200, 2000, 2000, 2000, 50]))
        self.gep_dropout_rate = list(getattr(args, "deside_dropout_rate", [0.05, 0.05, 0.05, 0.2, 0.0]))
        self.pathway_hidden_dims = list(
            getattr(args, "deside_pathway_hidden_dims", [50, 500, 500, 500, 50])
        )
        self.pathway_dropout_rate = list(
            getattr(args, "deside_pathway_dropout_rate", [0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.gep_branch = _FeatureBranch(
            input_dim=int(args.input_dim[1]),
            hidden_units=self.gep_hidden_dims,
            dropout_rates=self.gep_dropout_rate,
            normalization=self.normalization,
            normalization_layer=self.normalization_layer,
        )

        self.pathway_branch = None
        self.merge_layer = None
        if self.pathway_enabled:
            if not data_config or not data_config.pathway_file_path:
                raise ValueError(
                    "DeSideCellPropPredictor requires data.pathway_file_path when deside_pathway_network=True."
                )
            pathway_mask = self._build_pathway_mask(
                pathway_file_path=data_config.pathway_file_path,
                gene_list_file_path=args.input_gene_list_fp,
            )
            self.register_buffer("pathway_mask", pathway_mask)
            self.pathway_branch = _FeatureBranch(
                input_dim=pathway_mask.shape[1],
                hidden_units=self.pathway_hidden_dims,
                dropout_rates=self.pathway_dropout_rate,
                normalization=self.normalization,
                normalization_layer=self.normalization_layer,
            )
            self.merge_layer = _DenseNormBlock(
                self.gep_hidden_dims[-1] + self.pathway_hidden_dims[-1],
                self.gep_hidden_dims[-1],
                normalization=None,
                use_norm=False,
            )

        head_output_dim = get_cell_prop_head_output_dim(
            n_cell_types=self.n_cell_types,
            activation_function=self.cell_prop_activation_function,
        )
        self.output_layer = nn.Linear(self.gep_hidden_dims[-1], head_output_dim)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity="relu")
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _build_pathway_mask(pathway_file_path, gene_list_file_path) -> torch.Tensor:
        if gene_list_file_path is None:
            raise ValueError(
                "DeSideCellPropPredictor requires model.input_gene_list_fp to build the pathway mask."
            )
        pathway_mask = read_gene_set(pathway_file_path)
        gene_list_order = pd.read_csv(gene_list_file_path, header=None)[0].tolist()
        pathway_mask = pathway_mask.reindex(gene_list_order, fill_value=0.0)
        return torch.as_tensor(pathway_mask.to_numpy(dtype="float32", copy=True))

    def _compute_pathway_profile(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_config.scaling_by_constant:
            x = x * self.data_config.scaling_factor
        x = log_exp2cpm_tensor(x)
        pathway_profile = x @ self.pathway_mask
        pathway_profile = torch.log2(pathway_profile + 1)
        if self.data_config.scaling_by_constant:
            pathway_profile = pathway_profile / self.data_config.scaling_factor
        return pathway_profile

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        output_layer_levels=None,
        eps: float = EPS,
    ) -> ModelOutput:
        del output_layer_levels
        features = self.gep_branch(x)
        if self.pathway_enabled:
            pathway_profile = self._compute_pathway_profile(x)
            pathway_features = self.pathway_branch(pathway_profile)
            features = torch.cat([features, pathway_features], dim=1)
            features = self.merge_layer(features)

        output = ModelOutput()
        output["bulk_context_feature"] = features
        if self.predict_cell_prop:
            cell_prop, dd_alpha = build_cell_prop_from_head_output(
                head_output=self.output_layer(features),
                activation_function=self.cell_prop_activation_function,
                n_cell_types=self.n_cell_types,
                eps=eps,
                cancer_cell_type_index=self.cancer_cell_type_index,
            )
            output["cell_prop"] = cell_prop
            output["dd_alpha"] = dd_alpha
            output["cell_type_existed"] = (cell_prop >= 0.01).float()
        elif has_usable_labels(y):
            output["cell_prop"] = y
            output["dd_alpha"] = None
        else:
            output["cell_prop"] = None
            output["dd_alpha"] = None
        return output
