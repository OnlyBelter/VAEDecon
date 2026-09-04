"""
Variational Autoencoder (VAE) implementation for cellular component deconvolution.
OnlyBelter (https://github.com/OnlyBelter, onlybelter@gmail.com)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Tuple, Literal, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from ...configs import DataConfig, ModelConfig
from ...data.datasets import DatasetOutput
from ...models.base import (
    BaseAE,
    reparameterize_gaussian,
    dirichlet_mean,
    ModelOutput,
    MLPBlock,
    StackedMLPHead,
    BaseDecoder,
    BaseEncoder,
    EPS,
    build_cell_prop_from_head_output,
    get_cell_prop_head_output_dim,
    has_usable_labels,
    remove_cancer_cell_type,
    resolve_cancer_cell_type_index,
)
from ...utility import log_exp2cpm_tensor, non_log2log_cpm_tensor, non_log2cpm_tensor
from ...utility.hierarchical_encoding import HIERARCHICAL_ENCODING

logger = logging.getLogger(__name__)

HIERARCHY_REPULSION_BASE_MARGIN = 0.0
HIERARCHY_REPULSION_MARGIN_ALPHA = 0.5


def _clamp_log2_expression_before_exp(
    x_log2: torch.Tensor,
    *,
    min_value: float = 0.0,
    max_value: float = 20.0,
) -> torch.Tensor:
    """Clamp log2-scale expression to a safe range before exponentiation."""
    if max_value < min_value:
        raise ValueError(f"max_value must be >= min_value, got {max_value} < {min_value}")
    return torch.clamp(x_log2, min=min_value, max=max_value)


@dataclass
class LossTerms:
    """
    Container for decomposed loss terms.

    Notes:
    - `total` is a scalar (already reduced by batch mean).
    - Most other fields are scalar means for logging.
    """
    total: torch.Tensor
    kld_z: torch.Tensor
    kld_p: torch.Tensor
    recon: torch.Tensor
    gm: torch.Tensor
    gs: torch.Tensor
    repulsion: torch.Tensor
    cell_prop: torch.Tensor
    attractor: torch.Tensor = torch.tensor(0.0)
    cell_type_sct_gep: torch.Tensor = torch.tensor(0.0)
    cell_type_existence: torch.Tensor = torch.tensor(0.0)
    z_score_reciprocal: torch.Tensor = torch.tensor(0.0)  # Optional term for std regularization
    z_score_kl_loss: torch.Tensor = torch.tensor(0.0)  # New KL term for empirical z-score to N(0,1)
    low_mean_std_gene_loss: torch.Tensor = torch.tensor(0.0)  # Optional term to prevent collapse of low-mean/std genes
    hierarchical_code: torch.Tensor = torch.tensor(0.0)
    cross_sample_gene_var_loss: torch.Tensor = torch.tensor(0.0)
    per_sample_residual_var_loss: torch.Tensor = torch.tensor(0.0)
    inter_sample_similarity_loss: torch.Tensor = torch.tensor(0.0)


class VAE(BaseAE):
    """
    Variational Autoencoder model for cellular component deconvolution.

    Design highlights:
    1) Supports 1 to 4 encoders.
    2) Uses posterior-level fusion (default PoE) when multiple encoders are provided.
    3) Maintains gene-statistic targets (mean/std) as registered buffers.
    4) Keeps reconstruction in biologically meaningful spaces (log/non-log/CPM) with explicit transforms.

    Why posterior-level fusion by default?
    - Fusing `mu/logvar` keeps uncertainty information from each encoder.
    - Product-of-Experts (PoE) has a clear probabilistic interpretation:
      precision-weighted combination.
    - In practice this is usually more stable/interpretable than feature fusion before latent heads.

    Config note:
    - `model_config.fusion_strategy` can be:
      - "poe_posterior" (recommended default)
      - "avg_posterior" (simple baseline)
      - "pre_latent" (placeholder: requires architecture-level support in encoders)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: Optional[DataConfig] = None,
        encoders: Optional[List[BaseEncoder]] = None,
        cell_prop_predictor: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        super().__init__(
            model_config=model_config,
            data_config=data_config,
            encoders=encoders,
            decoder=decoder,
        )

        # ---------------------------------------------------------------------
        # Basic sanity checks
        # ---------------------------------------------------------------------
        self.z_scores = None
        self.n_encoders = len(self.encoders)
        if self.n_encoders not in (1, 2, 3, 4):
            raise ValueError(f"Only 1, 2, 3, or 4 encoders are supported, got {self.n_encoders}.")
        self.encoder_aliases = list(getattr(model_config, "encoder_aliases", []) or [])
        if not self.encoder_aliases:
            self.encoder_aliases = [f"encoder_{idx}" for idx in range(self.n_encoders)]
        if len(self.encoder_aliases) != self.n_encoders:
            raise ValueError(
                f"encoder_aliases (len={len(self.encoder_aliases)}) must match "
                f"the number of encoders ({self.n_encoders})."
            )
        self.encoder_alias_to_index = {
            alias: idx for idx, alias in enumerate(self.encoder_aliases)
        }
        self.cell_prop_predictor = cell_prop_predictor
        self.cell_prop_predictor_alias = None
        if self.cell_prop_predictor is not None:
            self.cell_prop_predictor_alias = getattr(
                self.cell_prop_predictor,
                "encoder_alias",
                getattr(model_config, "cell_prop_predictor_alias", "cell_prop_predictor"),
            )
        routing = getattr(model_config, "encoder_output_routing", None)
        self.cell_prop_source = getattr(routing, "cell_prop_source", "fused")
        self.latent_posterior_source = getattr(routing, "latent_posterior_source", "fused")
        self.decoder_context_source = getattr(routing, "decoder_context_source", "fused")

        self.model_name = "VAE"
        self.scaling_factor = float(data_config.scaling_factor)
        if not self.data_config.scaling_by_constant:
            self.scaling_factor = 1.0  # No scaling if not specified in data config
        self.mask_ratio = float(getattr(model_config, "mask_ratio", 0.0))

        # Fusion strategy for multiple encoders
        # Recommended: poe_posterior
        self.fusion_strategy: Literal["poe_posterior", "avg_posterior", "pre_latent"] = getattr(
            model_config, "fusion_strategy", "poe_posterior"
        )
        if self.fusion_strategy not in ("poe_posterior", "avg_posterior", "pre_latent"):
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        latent_dim = int(model_config.latent_dim)
        n_cell_types = int(model_config.n_cell_types)

        # ---------------------------------------------------------------------
        # --- Anchor & Prior Setup ---
        # Initialize orthonormal anchors for structured latent space.
        # ---------------------------------------------------------------------
        anchor_vectors = self.make_orthonormal_anchors(latent_dim=latent_dim)
        self.register_buffer("anchors", anchor_vectors)

        # Logits to weight anchors (Learnable parameters to associate cell types with anchors)
        self.logits = nn.Parameter(torch.zeros(n_cell_types, latent_dim))
        self.hierarchical_code_head = nn.Linear(latent_dim, 8)

        # ---------------------------------------------------------------------
        # --- Gene Statistics Setup ---
        # TODO: only read when it is necessary (lazy init possible in future).
        # ---------------------------------------------------------------------
        gene_fp = model_config.gene_mean_std_fp
        if not os.path.exists(gene_fp):
            raise FileNotFoundError(f"Gene features file not found: {gene_fp}")

        # Optimization: Use usecols if file schema is fixed enough.
        gf_df = pd.read_csv(gene_fp, index_col=0)

        # log2(CPM + 1) / scaling_factor space, so we can directly compare with reconstructions without extra transformations.
        # Cell types are expected in the same order as "cell_type_list.txt".
        avg_cols = [c for c in gf_df.columns if c.endswith("avg")]
        std_cols = [c for c in gf_df.columns if c.endswith("std")]
        self.cell_types = [
            c[:-4] if c.endswith("_avg") else (c[:-4] if c.endswith(" avg") else c[:-3])
            for c in avg_cols
        ]
        g_mean_np = gf_df.loc[:, avg_cols].values
        g_std_np = gf_df.loc[:, std_cols].values

        g_mean = torch.tensor(g_mean_np, dtype=torch.float32)
        g_std = torch.tensor(g_std_np, dtype=torch.float32)

        # Convert mean/std GEP to non-log space for residual-mode reconstruction.
        g_mean_non_log = to_non_log_space(g_mean, self.scaling_factor)
        g_std_non_log = to_non_log_space(g_std, self.scaling_factor)

        # Register as buffers so they move to device automatically with the model.
        self.register_buffer("g_mean", g_mean)
        self.register_buffer("g_std", g_std)
        self.register_buffer("g_mean_non_log", g_mean_non_log)
        self.register_buffer("g_std_non_log", g_std_non_log)

        # --- Cross-sample gene-variance targets (for loss_coefficient.cross_sample_gene_var_weight) ---
        cross_var_fp = getattr(model_config, "training_sct_cross_sample_gene_var_fp", None)
        cross_var_weight = float(
            getattr(getattr(model_config, "loss_coefficient", None), "cross_sample_gene_var_weight", 0.0) or 0.0
        )
        if cross_var_weight > 0.0:
            if cross_var_fp is None or not os.path.exists(cross_var_fp):
                raise FileNotFoundError(
                    "loss_coefficient.cross_sample_gene_var_weight > 0 requires a valid "
                    f"training_sct_cross_sample_gene_var_fp; got {cross_var_fp}"
                )
            cv_df = pd.read_csv(cross_var_fp, index_col=0)
            var_cols = [c for c in cv_df.columns if c.endswith("_var")]
            expected_order = [f"{ct}_var" for ct in self.cell_types]
            if list(cv_df.index) != list(gf_df.index):
                cv_df = cv_df.reindex(gf_df.index)
                if cv_df.isna().any().any():
                    raise RuntimeError(
                        "training_sct_cross_sample_gene_var_fp gene index does not match "
                        "gene_mean_std_fp after reindex."
                    )
            if len(var_cols) != len(self.cell_types):
                raise RuntimeError(
                    f"training_sct_cross_sample_gene_var_fp has {len(var_cols)} *_var columns, "
                    f"expected {len(self.cell_types)} for cell types {self.cell_types}."
                )
            try:
                cv_df_sorted = cv_df.loc[:, expected_order]
            except KeyError as exc:
                raise RuntimeError(
                    "training_sct_cross_sample_gene_var_fp columns do not match cell types "
                    f"(expected {expected_order}, got {var_cols})"
                ) from exc
            g_cross_var_np = cv_df_sorted.values.astype(np.float32)
            if g_cross_var_np.shape != (g_mean_np.shape[0], len(self.cell_types)):
                raise RuntimeError(
                    "training_sct_cross_sample_gene_var_fp shape mismatch after alignment: "
                    f"{g_cross_var_np.shape} vs expected {(g_mean_np.shape[0], len(self.cell_types))}"
                )
            g_cross_var = torch.tensor(g_cross_var_np, dtype=torch.float32)
        else:
            g_cross_var = torch.zeros((g_mean_np.shape[0], len(self.cell_types)), dtype=torch.float32)
        self.register_buffer("g_cross_sample_gene_var", g_cross_var)

        per_sample_residual_var_fp = getattr(model_config, "training_sct_per_sample_residual_var_fp", None)
        per_sample_residual_var_weight = float(
            getattr(getattr(model_config, "loss_coefficient", None), "per_sample_residual_var_weight", 0.0) or 0.0
        )
        self.training_sct_per_sample_residual_var_by_sample: Dict[str, np.ndarray] = {}
        if per_sample_residual_var_weight > 0.0:
            if per_sample_residual_var_fp is None or not os.path.exists(per_sample_residual_var_fp):
                raise FileNotFoundError(
                    "loss_coefficient.per_sample_residual_var_weight > 0 requires a valid "
                    f"training_sct_per_sample_residual_var_fp; got {per_sample_residual_var_fp}"
                )
            per_sample_df = pd.read_csv(per_sample_residual_var_fp, index_col=0)
            if per_sample_df.index.has_duplicates:
                duplicate_ids = per_sample_df.index[per_sample_df.index.duplicated()].unique().tolist()
                raise RuntimeError(
                    "training_sct_per_sample_residual_var_fp contains duplicated sample IDs: "
                    f"{duplicate_ids[:5]}"
                )
            try:
                per_sample_df = per_sample_df.loc[:, self.cell_types]
            except KeyError as exc:
                raise RuntimeError(
                    "training_sct_per_sample_residual_var_fp columns do not match cell types "
                    f"(expected {self.cell_types}, got {per_sample_df.columns.tolist()})"
                ) from exc
            self.training_sct_per_sample_residual_var_by_sample = {
                str(sample_id): row.to_numpy(dtype=np.float32, copy=False)
                for sample_id, row in per_sample_df.iterrows()
            }

        hierarchical_targets = []
        missing = []
        for ct in self.cell_types:
            code = HIERARCHICAL_ENCODING.get(ct)
            if code is None:
                missing.append(ct)
                code = [0] * 8
            hierarchical_targets.append(code)
        if missing:
            raise ValueError(f"Missing hierarchical_encoding for cell types: {missing}")
        self.register_buffer(
            "hierarchical_code_targets",
            torch.tensor(hierarchical_targets, dtype=torch.float32),
        )
        self.register_buffer(
            "hierarchical_repulsion_margin",
            self._build_hierarchy_repulsion_margin(
                self.hierarchical_code_targets,
                base_margin=HIERARCHY_REPULSION_BASE_MARGIN,
                alpha=HIERARCHY_REPULSION_MARGIN_ALPHA,
            ),
        )
        self.cell_prop_activation_function = model_config.cell_prop_activation_function
        self.cancer_cell_type_index = None
        if self.cell_prop_activation_function == "sigmoid":
            self.cancer_cell_type_index = resolve_cancer_cell_type_index(
                cell_types=self.cell_types,
                cancer_cell_type_name=model_config.cancer_cell_type_name,
            )
        self.cell_prop_fusion_strategy: Literal[
            "legacy_output_average",
            "shared_feature_mean",
            "shared_feature_gated",
        ] = getattr(model_config, "cell_prop_fusion_strategy", "legacy_output_average")
        if self.cell_prop_fusion_strategy not in (
            "legacy_output_average",
            "shared_feature_mean",
            "shared_feature_gated",
        ):
            raise ValueError(
                f"Unsupported cell_prop_fusion_strategy: {self.cell_prop_fusion_strategy}"
            )
        self.cell_prop_fusion_dim = int(getattr(model_config, "cell_prop_fusion_dim", 256))
        self.cell_prop_head_dropout_rate = float(
            getattr(model_config, "cell_prop_head_dropout_rate", 0.1)
        )
        self.cell_prop_head_hidden_dims = list(
            getattr(model_config, "cell_prop_head_hidden_dims", [512, 256])
        )
        self._use_shared_cell_prop_head = (
            bool(model_config.predict_cell_prop)
            and self.cell_prop_fusion_strategy != "legacy_output_average"
        )
        self.cell_prop_feature_projectors = nn.ModuleList()
        self.cell_prop_fusion_gate = None
        self.cell_prop_head = None
        if self._use_shared_cell_prop_head:
            head_output_dim = get_cell_prop_head_output_dim(
                n_cell_types=n_cell_types,
                activation_function=self.cell_prop_activation_function,
            )
            self.cell_prop_feature_projectors = nn.ModuleList(
                [
                    MLPBlock(
                        in_dim=None,
                        out_dim=self.cell_prop_fusion_dim,
                        dropout=self.cell_prop_head_dropout_rate,
                        lazy=True,
                    )
                    for _ in range(self.n_encoders)
                ]
            )
            if self.cell_prop_fusion_strategy == "shared_feature_gated" and self.n_encoders > 1:
                gate_input_dim = self.cell_prop_fusion_dim * self.n_encoders
                self.cell_prop_fusion_gate = nn.Sequential(
                    nn.Linear(gate_input_dim, self.cell_prop_fusion_dim),
                    nn.LayerNorm(self.cell_prop_fusion_dim, eps=EPS),
                    nn.GELU(),
                    nn.Dropout(self.cell_prop_head_dropout_rate)
                    if self.cell_prop_head_dropout_rate > 0
                    else nn.Identity(),
                    nn.Linear(self.cell_prop_fusion_dim, self.n_encoders),
                )
            self.cell_prop_head = StackedMLPHead(
                in_dim=self.cell_prop_fusion_dim,
                hidden_dims=self.cell_prop_head_hidden_dims,
                out_dim=head_output_dim,
                dropout=self.cell_prop_head_dropout_rate,
            )
        self._use_decoder_conditioning = bool(getattr(self.decoder, "supports_conditioning", False))
        self.decoder_context_dim = int(
            getattr(model_config, "conditional_decoder_context_dim", 256)
        )
        self.decoder_context_feature_projectors = nn.ModuleList()
        self.cell_prop_predictor_context_projector = None
        if self._use_decoder_conditioning:
            decoder_context_dropout = float(
                getattr(model_config, "conditional_decoder_dropout_rate", 0.1)
            )
            self.decoder_context_feature_projectors = nn.ModuleList(
                [
                    MLPBlock(
                        in_dim=None,
                        out_dim=self.decoder_context_dim,
                        dropout=decoder_context_dropout,
                        lazy=True,
                    )
                    for _ in range(self.n_encoders)
                ]
            )
            if self.cell_prop_predictor is not None:
                self.cell_prop_predictor_context_projector = MLPBlock(
                    in_dim=None,
                    out_dim=self.decoder_context_dim,
                    dropout=decoder_context_dropout,
                    lazy=True,
                )

        # ---------------------------------------------------------------------
        # --- Gene Weights Calculation ---
        # Calculated once and fixed. If dynamic adjustment is needed, move to forward().
        # ---------------------------------------------------------------------
        w = torch.ones_like(self.g_mean)
        if self.model_config.loss_coefficient.weighting_gene_by_exp:
            w = self.compute_gene_weights(
                clamp_range=self.model_config.loss_coefficient.weight_clamp_range
            )
        self.register_buffer("w", w)

    def save(
        self,
        model_dir: str,
        training_config: Optional["TrainingConfig"] = None,
        data_config: Optional[DataConfig] = None,
    ):
        super().save(model_dir=model_dir, training_config=training_config, data_config=data_config)

        try:
            lo = self.model_config.loss_coefficient
            low_mean_threshold = getattr(lo, "low_mean_threshold", 2.0)
            low_std_threshold = getattr(lo, "low_std_threshold", 1.0)

            low_mean_mask = self.g_mean_non_log < low_mean_threshold
            low_std_mask = self.g_std_non_log < low_std_threshold
            low_mean_or_std_mask = low_mean_mask | low_std_mask

            low_mean_counts = low_mean_mask.sum(dim=0).to(torch.int64).cpu().tolist()
            low_std_counts = low_std_mask.sum(dim=0).to(torch.int64).cpu().tolist()
            low_mean_or_std_counts = low_mean_or_std_mask.sum(dim=0).to(torch.int64).cpu().tolist()

            n_genes = int(self.g_mean_non_log.shape[0])
            n_cell_types = int(self.g_mean_non_log.shape[1])
            cell_type_labels = (
                self.cell_types
                if isinstance(getattr(self, "cell_types", None), list) and len(self.cell_types) == n_cell_types
                else [f"cell_type_{i}" for i in range(n_cell_types)]
            )

            df = pd.DataFrame(
                {
                    "cell_type": cell_type_labels,
                    "low_mean_count": low_mean_counts,
                    "low_std_count": low_std_counts,
                    "low_mean_or_std_count": low_mean_or_std_counts,
                    "n_genes": n_genes,
                    "low_mean_threshold": low_mean_threshold,
                    "low_std_threshold": low_std_threshold,
                }
            )
            out_fp = os.path.join(model_dir, "low_mean_std_gene_counts.csv")
            df.to_csv(out_fp, index=False)
            logger.info("Saved low-mean/low-std gene counts to %s", out_fp)
        except Exception as e:
            logger.warning("Failed to save low-mean/low-std gene counts: %s", e)

    def _fuse_cell_prop_features(
        self,
        feature_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse encoder-side cell-proportion features before the shared head."""
        if not feature_list:
            raise ValueError("feature_list must contain at least one encoder feature tensor.")

        projected_features = [
            self.cell_prop_feature_projectors[idx](feature)
            for idx, feature in enumerate(feature_list)
            if feature is not None
        ]
        if not projected_features:
            raise ValueError("At least one encoder feature is required for shared feature fusion.")
        if len(projected_features) == 1:
            return projected_features[0], None

        stacked = torch.stack(projected_features, dim=1)
        if self.cell_prop_fusion_strategy == "shared_feature_gated":
            if self.cell_prop_fusion_gate is None:
                raise ValueError("shared_feature_gated requires cell_prop_fusion_gate to be initialized.")
            gate_input = torch.cat(projected_features, dim=-1)
            gate_logits = self.cell_prop_fusion_gate(gate_input)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            fused_feature = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
            return fused_feature, gate_weights

        return stacked.mean(dim=1), None

    def _predict_cell_prop_from_features(
        self,
        feature_list: List[Optional[torch.Tensor]],
        eps: float = EPS,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Predict cell proportions from one or more encoder features."""
        if not self.model_config.predict_cell_prop:
            return None, None

        available_features = [feature for feature in feature_list if feature is not None]
        if not available_features:
            raise ValueError(
                "cell_prop_fusion_strategy requires encoders to expose cell_prop_feature."
            )

        fused_feature, _ = self._fuse_cell_prop_features(available_features)
        head_output = self.cell_prop_head(fused_feature)
        return build_cell_prop_from_head_output(
            head_output=head_output,
            activation_function=self.cell_prop_activation_function,
            n_cell_types=self.model_config.n_cell_types,
            eps=eps,
            cancer_cell_type_index=self.cancer_cell_type_index,
        )

    def _build_decoder_bulk_context(
        self,
        feature_list: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Fuse encoder-side sample features into one per-sample decoder context."""
        projected_features = [
            self.decoder_context_feature_projectors[idx](feature)
            for idx, feature in enumerate(feature_list)
            if feature is not None
        ]
        if not projected_features:
            raise ValueError(
                "Conditioned decoders require encoders to expose cell_prop_feature."
            )
        if len(projected_features) == 1:
            return projected_features[0]
        return torch.stack(projected_features, dim=0).mean(dim=0)

    def _resolve_routing_index(self, source: str, *, field_name: str) -> Optional[int]:
        """Return the encoder index for a named routing source, or None for fused."""
        if source == "fused":
            return None
        if source not in self.encoder_alias_to_index:
            raise ValueError(
                f"{field_name}={source!r} does not match any active encoder alias "
                f"{self.encoder_aliases}."
            )
        return self.encoder_alias_to_index[source]

    def _is_cell_prop_predictor_source(self, source: str) -> bool:
        return self.cell_prop_predictor_alias is not None and source == self.cell_prop_predictor_alias

    def _route_latent_posterior(
        self,
        *,
        mu_list: List[torch.Tensor],
        logvar_list: List[torch.Tensor],
        mu_mean_list: List[torch.Tensor],
        logvar_mean_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one encoder posterior or fall back to the configured fusion path."""
        source_index = self._resolve_routing_index(
            self.latent_posterior_source,
            field_name="encoder_output_routing.latent_posterior_source",
        )
        if source_index is not None:
            return (
                mu_list[source_index],
                logvar_list[source_index],
                mu_mean_list[source_index],
                logvar_mean_list[source_index],
            )
        if self.n_encoders == 1:
            return mu_list[0], logvar_list[0], mu_mean_list[0], logvar_mean_list[0]
        return self._fuse_encoder_posteriors(
            mu_lists_celltype=mu_list,
            logvar_lists_celltype=logvar_list,
            mu_list_overall=mu_mean_list,
            logvar_list_overall=logvar_mean_list,
            strategy=self.fusion_strategy,
        )

    def _route_cell_prop_prediction(
        self,
        *,
        prop_list: List[Optional[torch.Tensor]],
        dd_alpha_list: List[Optional[torch.Tensor]],
        feature_list: List[Optional[torch.Tensor]],
        predictor_prop: Optional[torch.Tensor] = None,
        predictor_dd_alpha: Optional[torch.Tensor] = None,
        eps: float = EPS,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select one encoder cell-proportion branch or use the existing fusion path."""
        if not self.model_config.predict_cell_prop:
            return None, None

        if self._is_cell_prop_predictor_source(self.cell_prop_source):
            if predictor_prop is None:
                raise ValueError(
                    f"Cell-proportion predictor alias {self.cell_prop_predictor_alias!r} does not expose cell_prop."
                )
            return predictor_prop, predictor_dd_alpha

        source_index = self._resolve_routing_index(
            self.cell_prop_source,
            field_name="encoder_output_routing.cell_prop_source",
        )
        if source_index is not None:
            pred_cell_prop = prop_list[source_index]
            if pred_cell_prop is None:
                raise ValueError(
                    f"Encoder alias {self.encoder_aliases[source_index]!r} does not expose cell_prop."
                )
            return pred_cell_prop, dd_alpha_list[source_index]

        if self._use_shared_cell_prop_head:
            return self._predict_cell_prop_from_features(feature_list=feature_list, eps=eps)
        if self.n_encoders == 1:
            return prop_list[0], dd_alpha_list[0]

        available_props = [prop for prop in prop_list if prop is not None]
        if not available_props:
            raise ValueError("No encoder exposed cell_prop for fused cell-proportion routing.")
        pred_cell_prop = torch.mean(torch.stack(available_props, dim=0), dim=0)
        available_dd_alpha = [alpha for alpha in dd_alpha_list if alpha is not None]
        dd_alpha = (
            torch.mean(torch.stack(available_dd_alpha, dim=0), dim=0)
            if available_dd_alpha else None
        )
        return pred_cell_prop, dd_alpha

    def _route_decoder_bulk_context(
        self,
        *,
        feature_list: List[Optional[torch.Tensor]],
        predictor_feature: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Select one encoder decoder context feature or use the fused context path."""
        if not self._use_decoder_conditioning:
            return None

        if self._is_cell_prop_predictor_source(self.decoder_context_source):
            if predictor_feature is None:
                raise ValueError(
                    f"Cell-proportion predictor alias {self.cell_prop_predictor_alias!r} does not expose "
                    "a decoder context feature."
                )
            return self.cell_prop_predictor_context_projector(predictor_feature)

        source_index = self._resolve_routing_index(
            self.decoder_context_source,
            field_name="encoder_output_routing.decoder_context_source",
        )
        if source_index is None:
            return self._build_decoder_bulk_context(feature_list=feature_list)

        feature = feature_list[source_index]
        if feature is None:
            raise ValueError(
                f"Encoder alias {self.encoder_aliases[source_index]!r} does not expose "
                "a decoder context feature."
            )
        return self.decoder_context_feature_projectors[source_index](feature)

    def _resolve_effective_cell_prop(
        self,
        *,
        labels: Optional[torch.Tensor],
        predicted_cell_prop: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Resolve which cell proportions should be used downstream."""
        if self.model_config.predict_cell_prop:
            if predicted_cell_prop is None:
                raise ValueError(
                    "predict_cell_prop=True requires predicted cell proportions."
                )
            return predicted_cell_prop

        if has_usable_labels(labels):
            return labels

        raise ValueError(
            "predict_cell_prop=False requires ground-truth cell-fraction labels "
            "whenever cell proportions are needed, but no usable labels were provided."
        )

    # =========================================================================
    # Forward
    # =========================================================================
    def forward(self, inputs: "DatasetOutput", **kwargs) -> "ModelOutput":
        """
        Forward pass with optional random gene masking, encoder fusion, decoding, and loss computation.

        Input:
            inputs["data"]: (B, G)
            inputs.get("labels"): (B, C) or None
        """
        # 1. Get input and handle device automatically.
        # Do not use self.to(device) manually inside forward; rely on input tensor device.
        x = inputs["data"]
        device = x.device
        batch_size = x.shape[0]

        y = inputs.get("labels")
        if has_usable_labels(y):
            y = y.to(device)
        else:
            y = None
        true_sct_gep = inputs.get("true_sct_gep")
        if torch.is_tensor(true_sct_gep) and true_sct_gep.numel() > 0:
            true_sct_gep = true_sct_gep.to(device)
        else:
            true_sct_gep = None

        true_sct_gep_present_mask = inputs.get("true_sct_gep_present_mask")
        if torch.is_tensor(true_sct_gep_present_mask) and true_sct_gep_present_mask.numel() > 0:
            true_sct_gep_present_mask = true_sct_gep_present_mask.to(device=device, dtype=torch.bool)
        else:
            true_sct_gep_present_mask = None
        sample_ids = inputs.get("sample_id")
        if sample_ids is None:
            batch_sample_ids = None
        elif isinstance(sample_ids, str):
            batch_sample_ids = [sample_ids]
        else:
            batch_sample_ids = [str(sample_id) for sample_id in sample_ids]

        # ================== Random Gene Masking ==================
        # Only apply masking during training, not validation/testing.
        if self.training and self.mask_ratio > 0:
            # Create a random mask: 1 = keep, 0 = mask
            # Shape: (Batch, Genes)
            mask = torch.rand_like(x) > self.mask_ratio

            # Apply mask: Zero out masked genes
            x_masked = x * mask.float()

            # (Optional) Scale remaining values to preserve magnitude.
            # x_masked = x_masked / (1 - self.mask_ratio)

            # Use x_masked for the encoders.
            x_input = x_masked
        else:
            x_input = x

        # 2. Encoders Forward Pass
        mu_list, logvar_list = [], []
        mu_mean_list, logvar_mean_list = [], []
        prop_list, dd_alpha_list = [], []
        cell_prop_feature_list = []
        decoder_context_feature_list = []
        predictor_prop = None
        predictor_dd_alpha = None
        predictor_context_feature = None
        predictor_raw_non_cancer_prop = None

        for encoder in self.encoders:
            out = encoder(x=x_input, y=y)
            mu_list.append(out.mu_all_types)                # (B, L, C)
            logvar_list.append(out.logvar_all_types)        # (B, L, C)
            mu_mean_list.append(out.mu_mean)                # (B, L)
            logvar_mean_list.append(out.logvar_mean)        # (B, L)
            prop_list.append(out.cell_prop)                 # (B, C)
            cell_prop_feature = getattr(out, "cell_prop_feature", None)
            if cell_prop_feature is None:
                cell_prop_feature = getattr(out, "bulk_context_feature", None)
            cell_prop_feature_list.append(cell_prop_feature)
            decoder_context_feature = getattr(out, "bulk_context_feature", None)
            if decoder_context_feature is None:
                decoder_context_feature = getattr(out, "cell_prop_feature", None)
            decoder_context_feature_list.append(decoder_context_feature)
            dd_alpha_list.append(getattr(out, "dd_alpha", None))

        if self.cell_prop_predictor is not None:
            predictor_out = self.cell_prop_predictor(x=x_input, y=y)
            predictor_prop = getattr(predictor_out, "cell_prop", None)
            predictor_dd_alpha = getattr(predictor_out, "dd_alpha", None)
            predictor_context_feature = getattr(predictor_out, "bulk_context_feature", None)
            predictor_raw_non_cancer_prop = getattr(predictor_out, "raw_non_cancer_cell_prop", None)

        # 3. Fusion (Single or Multi-Encoder)
        mu_types, log_var_types, mu_mean, logvar_mean = self._route_latent_posterior(
            mu_list=mu_list,
            logvar_list=logvar_list,
            mu_mean_list=mu_mean_list,
            logvar_mean_list=logvar_mean_list,
        )

        pred_cell_prop, dd_alpha = self._route_cell_prop_prediction(
            prop_list=prop_list,
            dd_alpha_list=dd_alpha_list,
            feature_list=cell_prop_feature_list,
            predictor_prop=predictor_prop,
            predictor_dd_alpha=predictor_dd_alpha,
        )

        effective_cell_prop = self._resolve_effective_cell_prop(
            labels=y,
            predicted_cell_prop=pred_cell_prop,
        )

        n_cell_types = mu_types.shape[2]
        existence_logits, _, mu_types = self._apply_cell_type_existence_shift(
            mu_types=mu_types,
            cell_prop=effective_cell_prop,
            device=device,
        )
        mu_mean = mu_types.mean(dim=-1)
        decoder_bulk_context = self._route_decoder_bulk_context(
            feature_list=decoder_context_feature_list,
            predictor_feature=predictor_context_feature,
        )

        # 4. Reconstruction (Batch Decoding Optimization)
        # -------------------------------------------------------
        # Optimization: Merge Batch and CellType dimensions for one-time decoding
        # instead of looping through cell types.

        # Sample latent vectors: (B, L, C)
        # For each cell type k, sample z from diagonal Gaussian N(mu_k, diag(sigma_k^2))
        # via reparameterization trick.
        z_types = reparameterize_gaussian(mu_types, log_var_types)

        # Flatten for decoder: (B, L, C) -> (B, C, L) -> (B*C, L)
        z_types_flat = z_types.permute(0, 2, 1).reshape(-1, self.model_config.latent_dim)

        if self._use_decoder_conditioning:
            cell_type_indices = (
                torch.arange(n_cell_types, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
            )
            bulk_context_flat = (
                decoder_bulk_context.unsqueeze(1)
                .expand(-1, n_cell_types, -1)
                .reshape(-1, decoder_bulk_context.shape[-1])
            )
            recon_flat = self.decoder(
                z_types_flat,
                cell_type_indices=cell_type_indices,
                bulk_context=bulk_context_flat,
            )["reconstruction"]
        else:
            # One-time decoding: (B*C, L) -> (B*C, G)
            recon_flat = self.decoder(z_types_flat)["reconstruction"]

        # Restore shape: (B*C, G) -> (B, C, G) -> (B, G, C)
        # We need (B, G, C) for subsequent matrix multiplication.
        recon_x_all_types = recon_flat.view(batch_size, n_cell_types, -1).permute(0, 2, 1)
        # -------------------------------------------------------

        # 5. Scaling & Mixing

        residual_mode = getattr(self.model_config, "learn_gep_residual_mode", "zscore")
        recon_residual_log = None
        recon_x_all_types_log = None

        # Log -> CPM (Batch, Genes, C)
        if not self.model_config.learn_gep_residual:
            # Direct full-GEP mode predicts log2(TPM+1). Clamp to the same safe
            # exponentiation range used by the mean-centered residual path so
            # one extreme decoder output cannot poison CPM normalization.
            if self.data_config.scaling_by_constant:
                # Scale back up if input was scaled down.
                recon_x_all_types = recon_x_all_types * self.scaling_factor
            recon_x_all_types = _clamp_log2_expression_before_exp(
                recon_x_all_types,
                min_value=0.0,
                max_value=20.0,
            )
            recon_x_all_types_log = (
                recon_x_all_types / self.scaling_factor
                if self.data_config.scaling_by_constant
                else recon_x_all_types
            )
            # Decoder outputs full GEP in log space -> convert to CPM for mixing.
            recon_x_all_types_cpm = log_exp2cpm_tensor(recon_x_all_types, transpose=True)
            self.z_scores = None
        elif residual_mode == "mean_centered":
            # Decoder outputs a mean-centered residual in scaled log space.
            recon_residual_log = recon_x_all_types
            recon_x_all_types_log = recon_residual_log + self.g_mean.unsqueeze(0)
            # Clamp full reconstructed log2(TPM+1) to the normalized design range:
            # scaling_factor=20 means valid scaled expression stays within [0, 1].
            recon_x_all_types_unscaled_log = _clamp_log2_expression_before_exp(
                recon_x_all_types_log * self.scaling_factor,
                min_value=0.0,
                max_value=20.0,
            )
            recon_x_all_types_log = recon_x_all_types_unscaled_log / self.scaling_factor
            recon_x_all_types_cpm = log_exp2cpm_tensor(
                recon_x_all_types_unscaled_log,
                transpose=True,
            )
            self.z_scores = None
        else:
            # If learning residual, decoder outputs residual z-score in range (-6, 6).
            # We multiply by std and add mean GEP in non-log space, then normalize to CPM.

            # Redefine minimum z-score to guarantee all values >= 0 after adding mean GEP.
            mu = self.g_mean_non_log.unsqueeze(0)                           # (1, G, C)
            std = torch.clamp(self.g_std_non_log, min=EPS).unsqueeze(0)     # (1, G, C)
            z_min = torch.maximum(torch.full_like(std, -6.0), -mu / std)
            # Now recon_x_all_types is in range (z_min, 3).
            recon_x_all_types = z_min + (6.0 - z_min) * recon_x_all_types
            self.z_scores = recon_x_all_types  # For loss calculation: store the actual z-scores.

            # Convert residual z-score to residual in non-log space.
            recon_residual = recon_x_all_types * std                         # (B, G, C)

            # Add mean GEP in non-log space; transpose to (B, C, G) for non_log2cpm_tensor.
            recon_x_add_g_mean = torch.transpose(recon_residual + self.g_mean_non_log, 1, 2)

            # Normalize to CPM, then transpose back to (B, G, C).
            recon_x_all_types_cpm = non_log2cpm_tensor(recon_x_add_g_mean).transpose(1, 2)
            recon_x_all_types_log = to_log_space(recon_x_all_types_cpm, self.scaling_factor)

        # Prepare proportions for mixing.
        prop_matrix = effective_cell_prop.unsqueeze(-1)

        # Mixing: (B, G, C) @ (B, C, 1) -> (B, G, 1)
        recon_x_conv = torch.bmm(recon_x_all_types_cpm, prop_matrix).squeeze(-1)

        # Gene Stats for Loss calculation
        recon_gene_mean = recon_x_all_types_cpm.mean(dim=0)                 # (G, C)
        recon_gene_std = recon_x_all_types_cpm.std(dim=0)                   # (G, C)

        # Final Output Conversion
        recon_x_conv_log = non_log2log_cpm_tensor(recon_x_conv, transpose=False)
        if self.data_config.scaling_by_constant:
            recon_x_conv_log = recon_x_conv_log / self.scaling_factor

        # Ensure shape matches input x.
        recon_x_conv_log = recon_x_conv_log.reshape(x.shape)

        # 6. Prior Calculation
        # Calculate prior mean based on learned anchors.
        anchor_weights = F.softmax(self.logits, dim=-1)                     # (C, L)
        mu_prior = anchor_weights @ self.anchors                             # (C, L)

        # 7. Loss Calculation
        loss_terms = self.loss_function(
            x=x,
            y=y,
            recon_x_conv=recon_x_conv_log,
            mu_types=mu_types,
            logvar_types=log_var_types,
            pred_cell_prop=effective_cell_prop,
            raw_non_cancer_cell_prop=(
                predictor_raw_non_cancer_prop
                if self._is_cell_prop_predictor_source(self.cell_prop_source)
                else None
            ),
            existence_logits=existence_logits,
            dd_alpha=dd_alpha,
            mu_prior=mu_prior,
            recon_gene_mean=recon_gene_mean,
            recon_gene_std=recon_gene_std,
            logvar_mean=logvar_mean,
            mu_mean=mu_mean,
            device=device,
            recon_x_all_types_cpm=recon_x_all_types_cpm,
            recon_x_all_types_log=recon_x_all_types_log,
            recon_residual_log=recon_residual_log,
            true_sct_gep=true_sct_gep,
            true_sct_gep_present_mask=true_sct_gep_present_mask,
            sample_ids=batch_sample_ids,
        )

        return ModelOutput(
            loss=loss_terms.total,
            kld=loss_terms.kld_z,
            kld_p=loss_terms.kld_p,
            recon_loss_conv=loss_terms.recon,
            gene_mean_loss=loss_terms.gm,
            gene_std_loss=loss_terms.gs,
            repulsion_loss=loss_terms.repulsion,
            attractor_loss=loss_terms.attractor,
            hierarchical_code_loss=loss_terms.hierarchical_code,
            cell_prop_loss=loss_terms.cell_prop,
            z_score_kl_loss=loss_terms.z_score_kl_loss,
            low_mean_std_gene_loss=loss_terms.low_mean_std_gene_loss,
            cross_sample_gene_var_loss=loss_terms.cross_sample_gene_var_loss,
            per_sample_residual_var_loss=loss_terms.per_sample_residual_var_loss,
            inter_sample_similarity_loss=loss_terms.inter_sample_similarity_loss,
            cell_type_sct_gep_loss=loss_terms.cell_type_sct_gep,
            cell_type_existence_loss=loss_terms.cell_type_existence,
            mu=mu_mean,
            mu_deconv=mu_types,
            log_var=log_var_types,
            pred_cell_prop=effective_cell_prop,
            recon_x_conv=recon_x_conv_log,
            recon_x_all_types=recon_x_all_types_cpm,  # Usually return CPM format for analysis
            recon_x_all_types_log=recon_x_all_types_log,
            recon_residual_log=recon_residual_log,
            z_score_reciprocal=loss_terms.z_score_reciprocal,  # For monitoring potential z-score collapse when learning residuals
        )

    # =========================================================================
    # Loss Function
    # =========================================================================
    def loss_function(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        recon_x_conv: torch.Tensor,
        mu_types: torch.Tensor,
        logvar_types: torch.Tensor,
        pred_cell_prop: Optional[torch.Tensor],
        raw_non_cancer_cell_prop: Optional[torch.Tensor],
        existence_logits: Optional[torch.Tensor],
        dd_alpha: Optional[torch.Tensor],
        mu_prior: Optional[torch.Tensor],
        recon_gene_mean: torch.Tensor,
        recon_gene_std: torch.Tensor,
        logvar_mean: torch.Tensor,
        mu_mean: torch.Tensor,
        device: torch.device,
        recon_x_all_types_cpm: Optional[torch.Tensor] = None,
        recon_x_all_types_log: Optional[torch.Tensor] = None,
        recon_residual_log: Optional[torch.Tensor] = None,
        true_sct_gep: Optional[torch.Tensor] = None,
        true_sct_gep_present_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> LossTerms:
        """
        Compute all objective terms and aggregate total loss.
        Parameters:
            x: Input features (B, G, C), in log space after scaling by constant
            y: Optional target labels (B, C)
            recon_x_conv: Reconstructed bulk GEPs (B, G), in log space after scaling by constant
            mu_types: Latent space mean (B, L, C)
            logvar_types: Latent space log-variance (B, L, C)
            raw_non_cancer_cell_prop: Optional raw sigmoid outputs for the non-cancer cell types (B, C-1)
            mu_mean: Prior mean (B, L)
            logvar_mean: Prior log-variance (B, L)
            recon_gene_mean: Reconstructed gene mean per cell type across samples in a batch (G, C), in tpm space
            recon_gene_std: Reconstructed gene std per cell type across samples in a batch (G, C), in tpm space
            recon_x_all_types_cpm: GEPs of all cell types after deconvolution (B, G, C), in tpm space
            dd_alpha: Dirichlet distribution parameter (C,) for cell type proportions, if applicable.
            mu_prior: Prior mean (C, L) for latent space, if applicable.
            device: Device for tensor operations

        Shapes:
            x, recon_x_conv:      (B, G)
            mu_types/logvar_types:(B, L, C)
            mu_mean/logvar_mean:  (B, L)
            recon_gene_mean/std:  (G, C)
        """
        batch_size, n_gene = x.shape
        lo = self.model_config.loss_coefficient
        beta = lo.beta
        gamma = lo.gamma
        attractor_weight = lo.attractor_weight
        z_score_reg_weight = lo.z_score_reg_weight
        residual_mode = getattr(self.model_config, "learn_gep_residual_mode", "zscore")
        use_mean_centered_residual = bool(
            self.model_config.learn_gep_residual and residual_mode == "mean_centered"
        )
        labels_available = has_usable_labels(y)

        if use_mean_centered_residual:
            if lo.z_score_kl_weight > 0:
                raise ValueError(
                    "loss_coefficient['z_score_kl_weight'] must be 0 when "
                    "learn_gep_residual_mode='mean_centered'."
                )
            if z_score_reg_weight > 0:
                raise ValueError(
                    "loss_coefficient['z_score_reg_weight'] must be 0 when "
                    "learn_gep_residual_mode='mean_centered'."
                )

        if (
            self.training
            and self.model_config.predict_cell_prop
            and lo.cell_prop > 0
            and not labels_available
        ):
            raise ValueError(
                "predict_cell_prop=True with loss_coefficient.cell_prop > 0 "
                "requires cell-fraction labels during training."
            )

        # Optional regularization to prevent std from collapsing to zero.
        # Add 1 / mean_z_scores to the loss to encourage the model to keep larger z-scores (and thus std) from collapsing.
        if z_score_reg_weight > 0 and self.model_config.learn_gep_residual:
            mean_z_scores = self.z_scores.abs().mean(dim=(1, 2))  # (B,)
        else:
            mean_z_scores = torch.ones((batch_size,), device=device)
            # z_score_reg_weight = 0.0  # No regularization if not learning residuals.

        # --- 1. Reconstruction Loss ---
        recon_loss = self._reconstruction_loss(x=x, recon_x_conv=recon_x_conv)              # (B,)

        # --- 2. Gene Statistics Loss ---
        # If z-score KL regularization is enabled, ignore the original gene mean/std losses.
        if lo.z_score_kl_weight > 0:
            gm_loss = torch.tensor(0.0, device=device)
            gs_loss = torch.tensor(0.0, device=device)
        else:
            # Convert back to log space for consistency with input features.
            recon_gene_mean = to_log_space(recon_gene_mean, self.scaling_factor)
            recon_gene_std = to_log_space(recon_gene_std, self.scaling_factor)
            gm_loss, gs_loss = self._gene_statistics_loss(
                recon_gene_mean=recon_gene_mean,
                recon_gene_std=recon_gene_std,
                device=device,
            )                                                                               # scalar, scalar

        # --- 2.5 Z-score KL Loss ---
        if lo.z_score_kl_weight > 0:
            if recon_x_all_types_cpm is None:
                raise ValueError("recon_x_all_types_cpm is required when z_score_kl_weight > 0")
            z_score_kl_loss = self._z_score_kl_loss(recon_x_all_types_cpm)
        else:
            z_score_kl_loss = torch.tensor(0.0, device=device)

        # Identify genes with low mean or low std,
        # whose z-scores are unreliable for KL regularization.
        # Replace their predicted values with the reference mean plus small Gaussian noise.
        low_mean_std_weight = float(getattr(lo, "low_mean_std_weight", 0.0) or 0.0)
        if low_mean_std_weight > 0:
            low_mean_threshold = getattr(lo, "low_mean_threshold", 2.0)
            low_std_threshold = getattr(lo, "low_std_threshold", 1.0)
            mask = (self.g_mean_non_log < low_mean_threshold) | (self.g_std_non_log < low_std_threshold)  # (G, C)
            mask = mask.unsqueeze(0).expand(batch_size, n_gene, -1)  # (B, G, C)

            g_mean_expanded = self.g_mean_non_log.unsqueeze(0).expand(batch_size, n_gene, -1)  # (B, G, C)
            # g_std_expanded = self.g_std_non_log.unsqueeze(0).expand(batch_size, n_gene, -1)  # (B, G, C)
            g_mean_expanded_log = to_log_space(g_mean_expanded, self.scaling_factor)  # (B, G, C)
            if recon_x_all_types_log is None:
                if recon_x_all_types_cpm is None:
                    raise ValueError(
                        "Either recon_x_all_types_log or recon_x_all_types_cpm is required "
                        "for per-cell-type auxiliary losses."
                    )
                recon_x_all_types_log = to_log_space(recon_x_all_types_cpm, self.scaling_factor)
            mask_f = mask.to(dtype=recon_x_all_types_log.dtype)
            diff2 = (recon_x_all_types_log - g_mean_expanded_log).pow(2) * mask_f
            denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)
            low_mean_std_gene_loss_per_sample = diff2.sum(dim=(1, 2)) / denom  # (B,)
        else:
            low_mean_std_gene_loss_per_sample = torch.zeros((batch_size,), device=device)
        # --- 2.75 Cross-sample gene-variance matching loss ---
        cross_sample_gene_var_weight = float(getattr(lo, "cross_sample_gene_var_weight", 0.0) or 0.0)
        if cross_sample_gene_var_weight > 0 and batch_size >= 2:
            cross_var_loss_per_sample = self._cross_sample_gene_variance_loss(
                recon_x_all_types_log=recon_x_all_types_log,
                batch_size=batch_size,
            )
        else:
            cross_var_loss_per_sample = torch.zeros((batch_size,), device=device)

        per_sample_residual_var_weight = float(getattr(lo, "per_sample_residual_var_weight", 0.0) or 0.0)
        if per_sample_residual_var_weight > 0:
            if not use_mean_centered_residual or recon_residual_log is None:
                raise ValueError(
                    "per_sample_residual_var_weight > 0 requires recon_residual_log and "
                    "learn_gep_residual_mode='mean_centered'."
                )
            supervision_ready = (
                true_sct_gep_present_mask is not None
                and labels_available
                and sample_ids is not None
                and len(sample_ids) == batch_size
            )
            if supervision_ready:
                per_sample_residual_var_loss = self._per_sample_residual_variance_loss(
                    pred_residual_log=recon_residual_log,
                    sample_ids=sample_ids,
                    true_sct_gep_present_mask=true_sct_gep_present_mask,
                    true_cell_prop=y,
                    cell_prop_threshold=float(
                        getattr(self.data_config, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
                    ),
                )
            elif self.training:
                raise ValueError(
                    "Per-sample residual variance supervision requires sample_ids, "
                    "true_sct_gep_present_mask, and true cell-proportion labels during training."
                )
            else:
                per_sample_residual_var_loss = torch.zeros((batch_size,), device=device)
        else:
            per_sample_residual_var_loss = torch.zeros((batch_size,), device=device)

        inter_sample_similarity_weight = float(getattr(lo, "inter_sample_similarity_weight", 0.0) or 0.0)
        if inter_sample_similarity_weight > 0:
            supervision_ready = (
                true_sct_gep is not None
                and true_sct_gep_present_mask is not None
                and labels_available
            )
            if supervision_ready:
                cell_prop_threshold = float(
                    getattr(self.data_config, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
                )
                if use_mean_centered_residual:
                    if recon_residual_log is None:
                        raise ValueError(
                            "inter_sample_similarity_weight > 0 requires recon_residual_log "
                            "when learn_gep_residual_mode='mean_centered'."
                        )
                    inter_sample_similarity_loss = self._inter_sample_similarity_loss(
                        pred_residual_log=recon_residual_log,
                        true_sct_gep=true_sct_gep,
                        true_sct_gep_present_mask=true_sct_gep_present_mask,
                        true_cell_prop=y,
                        cell_prop_threshold=cell_prop_threshold,
                    )
                else:
                    if recon_x_all_types_log is None:
                        if recon_x_all_types_cpm is None:
                            raise ValueError(
                                "inter_sample_similarity_weight > 0 requires recon_x_all_types_log "
                                "or recon_x_all_types_cpm when residual learning is disabled."
                            )
                        recon_x_all_types_log = to_log_space(recon_x_all_types_cpm, self.scaling_factor)
                    inter_sample_similarity_loss = self._full_sct_gep_inter_sample_similarity_loss(
                        pred_sct_gep_log=recon_x_all_types_log,
                        true_sct_gep=true_sct_gep,
                        true_sct_gep_present_mask=true_sct_gep_present_mask,
                        true_cell_prop=y,
                        cell_prop_threshold=cell_prop_threshold,
                    )
            elif self.training:
                raise ValueError(
                    "Inter-sample similarity supervision requires true_sct_gep, "
                    "true_sct_gep_present_mask, and true cell-proportion labels during training."
                )
            else:
                inter_sample_similarity_loss = torch.zeros((batch_size,), device=device)
        else:
            inter_sample_similarity_loss = torch.zeros((batch_size,), device=device)

        cell_type_sct_gep_weight = float(getattr(lo, "cell_type_sct_gep_weight", 0.0) or 0.0)
        if cell_type_sct_gep_weight > 0:
            if use_mean_centered_residual:
                if recon_residual_log is None:
                    raise ValueError(
                        "recon_residual_log is required when cell_type_sct_gep_weight > 0 "
                        "and learn_gep_residual_mode='mean_centered'."
                    )
            elif recon_x_all_types_cpm is None:
                raise ValueError(
                    "recon_x_all_types_cpm is required when cell_type_sct_gep_weight > 0"
                )
            supervision_ready = (
                true_sct_gep is not None
                and true_sct_gep_present_mask is not None
                and labels_available
            )
            if supervision_ready:
                if use_mean_centered_residual:
                    cell_type_sct_gep_loss = self._matched_sct_gep_residual_supervision_loss(
                        pred_residual_log=recon_residual_log,
                        true_sct_gep=true_sct_gep,
                        true_sct_gep_present_mask=true_sct_gep_present_mask,
                        true_cell_prop=y,
                        cell_prop_threshold=float(
                            getattr(self.data_config, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
                        ),
                    )
                else:
                    cell_type_sct_gep_loss = self._matched_sct_gep_supervision_loss(
                        recon_x_all_types_cpm=recon_x_all_types_cpm,
                        true_sct_gep=true_sct_gep,
                        true_sct_gep_present_mask=true_sct_gep_present_mask,
                        true_cell_prop=y,
                        cell_prop_threshold=float(
                            getattr(self.data_config, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
                        ),
                    )
            elif self.training:
                if true_sct_gep is None or true_sct_gep_present_mask is None:
                    raise ValueError(
                        "Matched sctGEP supervision is enabled but the dataset batch does not "
                        "include true_sct_gep and true_sct_gep_present_mask."
                    )
                raise ValueError(
                    "Matched sctGEP supervision requires true cell-proportion labels during training."
                )
            else:
                # Inference/test batches do not carry training-only matched sctGEP targets.
                cell_type_sct_gep_loss = torch.zeros((batch_size,), device=device)
        else:
            cell_type_sct_gep_loss = torch.zeros((batch_size,), device=device)

        cell_type_existence_weight = float(getattr(lo, "cell_type_existence_weight", 0.0) or 0.0)
        if cell_type_existence_weight > 0:
            if not self.model_config.predict_cell_prop:
                raise ValueError(
                    "cell_type_existence_weight > 0 requires predict_cell_prop=True."
                )
            if existence_logits is None or pred_cell_prop is None:
                raise ValueError(
                    "Cell-type existence supervision requires predicted cell proportions."
                )
            if labels_available:
                cell_type_existence_loss = self._cell_type_existence_loss(
                    existence_logits=existence_logits,
                    true_cell_prop=y,
                    cell_prop_threshold=float(
                        getattr(self.data_config, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
                    ),
                )
            elif self.training:
                raise ValueError(
                    "Cell-type existence supervision requires true cell-proportion labels during training."
                )
            else:
                cell_type_existence_loss = torch.zeros((batch_size,), device=device)
        else:
            cell_type_existence_loss = torch.zeros((batch_size,), device=device)

        # --- 3. KL Divergence (Cell Proportions - Dirichlet) ---
        kld_p, cell_prop_loss = self._cell_prop_dirichlet_loss(
            y=y,
            dd_alpha=dd_alpha,
            pred_cell_prop=pred_cell_prop,
            raw_non_cancer_cell_prop=raw_non_cancer_cell_prop,
            batch_size=batch_size,
            device=device,
        )                                                                                   # (B,), (B,)

        # --- 4. KL Divergence (Latent - Gaussian) ---
        kld_z_types = self._latent_kld_loss(
            mu_types=mu_types,
            logvar_types=logvar_types,
            mu_prior=mu_prior,
            logvar_mean=logvar_mean,
            mu_mean=mu_mean,
            device=device,
        )                                                                                   # (B,)

        # --- 5. Repulsion Loss ---
        repulsion_loss = self._repulsion_loss(
            mu_types=mu_types,
            gamma=gamma,
        )                                                                                   # (B,)

        attractor_loss = self._attractor_loss(
            mu_types=mu_types,
            attractor_weight=attractor_weight,
        )                                                                                   # (B,)

        hierarchical_code_weight = lo.hierarchical_code_weight
        hierarchical_code_loss = self._hierarchical_code_loss(
            mu_types=mu_types,
            hierarchical_code_weight=hierarchical_code_weight,
        )                                                                                   # (B,)

        # --- Total Loss ---
        total_loss = (
            recon_loss
            + lo.low_mean_std_weight * low_mean_std_gene_loss_per_sample
            + beta * kld_z_types
            + lo.kld_p * kld_p
            + lo.cell_prop * cell_prop_loss
            + gamma * repulsion_loss
            + attractor_weight * attractor_loss
            + hierarchical_code_weight * hierarchical_code_loss
            # + lo.gene_mean_weight * gm_loss
            # + lo.gene_std_weight * gs_loss
            + lo.z_score_kl_weight * z_score_kl_loss
            + cross_sample_gene_var_weight * cross_var_loss_per_sample
            + per_sample_residual_var_weight * per_sample_residual_var_loss
            + inter_sample_similarity_weight * inter_sample_similarity_loss
            + cell_type_sct_gep_weight * cell_type_sct_gep_loss
            + cell_type_existence_weight * cell_type_existence_loss
            # + z_score_reg_weight * (1 / mean_z_scores)
        ).mean()

        return LossTerms(
            total=total_loss,
            kld_z=kld_z_types.mean(),
            kld_p=kld_p.mean(),
            recon=recon_loss.mean(),
            gm=gm_loss,
            gs=gs_loss,
            repulsion=repulsion_loss.mean(),
            cell_prop=cell_prop_loss.mean(),
            attractor=attractor_loss.mean(),
            cell_type_sct_gep=cell_type_sct_gep_loss.mean(),
            cell_type_existence=cell_type_existence_loss.mean(),
            hierarchical_code=hierarchical_code_loss.mean(),
            z_score_reciprocal=(1 / mean_z_scores).mean(),
            z_score_kl_loss=z_score_kl_loss.mean(),
            low_mean_std_gene_loss=low_mean_std_gene_loss_per_sample.mean(),
            cross_sample_gene_var_loss=cross_var_loss_per_sample.mean(),
            per_sample_residual_var_loss=per_sample_residual_var_loss.mean(),
            inter_sample_similarity_loss=inter_sample_similarity_loss.mean(),
        )

    # -------------------------------------------------------------------------
    # Loss Components
    # -------------------------------------------------------------------------
    def _reconstruction_loss(self, x: torch.Tensor, recon_x_conv: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction loss, shape (B,)."""
        if self.model_config.reconstruction_loss == "mse":
            return F.mse_loss(recon_x_conv, x, reduction="none").sum(dim=-1)
        # elif self.model_config.reconstruction_loss == "bce":
        #     return F.binary_cross_entropy(recon_x_conv, x, reduction="none").sum(dim=-1)
        raise ValueError(
            f"Unknown reconstruction loss: {self.model_config.reconstruction_loss}, only MSE is supported"
        )

    def _gene_statistics_loss(
        self,
        recon_gene_mean: torch.Tensor,
        recon_gene_std: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gene statistic losses (scalars):
        - GM: weighted MSE for mean
        - GS: MSE for std
        """
        lo = self.model_config.loss_coefficient
        if lo.gene_mean_weight == 0 and lo.gene_std_weight == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Note: self.g_mean and self.w are buffers, so they are on the correct device.
        gm_loss = F.mse_loss(recon_gene_mean, self.g_mean, reduction="none")
        gm_loss = (gm_loss * self.w).sum(dim=0).mean()  # Weighted sum over genes, mean over types
        gs_loss = F.mse_loss(recon_gene_std, self.g_std, reduction="none").sum(dim=0).mean()
        return gm_loss, gs_loss

    def _cross_sample_gene_variance_loss(
        self,
        recon_x_all_types_log: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Penalize mismatch between recon and training-SCT per-(gene, cell_type) cross-sample variance.

        Operates in scaled log-space (log2(CPM+1)/factor) so it directly matches
        the ``training_sct_cross_sample_gene_variances_*.csv`` targets and the
        per-(gene, ct) sample variance of ``recon_x_all_types_log`` over B.

        Shapes:
            recon_x_all_types_log: (B, G, C)
        Returns:
            Per-sample tensor of shape (B,); each row has the same scalar value
            (mean absolute difference per element) for consistency with other
            per-sample loss terms averaged at the end.
        """
        # Per-(gene, ct) sample variance over batch (ddof=1)
        # shape: (G, C)
        if batch_size < 2:
            B = recon_x_all_types_log.shape[0]
            return torch.zeros((B,), device=recon_x_all_types_log.device)

        mean_per_g = recon_x_all_types_log.mean(dim=0, keepdim=False)  # (G, C)
        sq_dev = (recon_x_all_types_log - mean_per_g.unsqueeze(0)).pow(2)  # (B, G, C)
        recon_var_per_g = sq_dev.sum(dim=0) / float(batch_size - 1)  # (G, C), unbiased

        target = self.g_cross_sample_gene_var.to(dtype=recon_var_per_g.dtype, device=recon_var_per_g.device)  # (G, C)
        abs_diff = (recon_var_per_g - target).abs()  # (G, C)
        denom = float(abs_diff.numel()) if abs_diff.numel() > 0 else 1.0
        scalar = abs_diff.sum() / denom  # mean absolute difference per (gene, ct) element

        B = recon_x_all_types_log.shape[0]
        return scalar.expand(B)

    def _lookup_per_sample_residual_var_targets(
        self,
        sample_ids: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return batch-aligned per-sample residual variance targets with shape (B, C)."""
        target_rows: list[np.ndarray] = []
        missing_sample_ids: list[str] = []
        for sample_id in sample_ids:
            target_row = self.training_sct_per_sample_residual_var_by_sample.get(str(sample_id))
            if target_row is None:
                missing_sample_ids.append(str(sample_id))
                target_rows.append(np.full((len(self.cell_types),), np.nan, dtype=np.float32))
            else:
                target_rows.append(target_row)
        if missing_sample_ids and self.training:
            raise ValueError(
                "Missing per-sample residual variance targets for batch sample IDs: "
                f"{missing_sample_ids[:5]}"
            )
        target_np = np.stack(target_rows, axis=0).astype(np.float32, copy=False)
        return torch.as_tensor(target_np, device=device, dtype=dtype)

    def _per_sample_residual_variance_loss(
        self,
        pred_residual_log: torch.Tensor,
        sample_ids: Sequence[str],
        true_sct_gep_present_mask: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Masked log-L1 loss on per-sample, per-cell-type residual variance across genes."""
        pred_var = pred_residual_log.var(dim=1, unbiased=False)
        target_var = self._lookup_per_sample_residual_var_targets(
            sample_ids,
            device=pred_var.device,
            dtype=pred_var.dtype,
        )
        active_mask = true_sct_gep_present_mask & (true_cell_prop >= cell_prop_threshold)
        valid_mask = active_mask & torch.isfinite(target_var)
        safe_target_var = torch.where(valid_mask, target_var, torch.ones_like(target_var))
        log_diff = (
            torch.log(pred_var.clamp_min(EPS))
            - torch.log(safe_target_var.clamp_min(EPS))
        ).abs()
        masked = log_diff * valid_mask.to(dtype=pred_var.dtype)
        denom = valid_mask.to(dtype=pred_var.dtype).sum(dim=1).clamp_min(1.0)
        return masked.sum(dim=1) / denom

    def _pairwise_cosine_similarity_matrix(
        self,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a differentiable sample-sample cosine similarity matrix.

        Args:
            residuals: Tensor of shape (B_active, G)

        Returns:
            Tensor of shape (B_active, B_active)
        """
        if residuals.shape[0] == 0:
            return torch.zeros((0, 0), device=residuals.device, dtype=residuals.dtype)

        normalized = F.normalize(residuals, p=2, dim=1, eps=EPS)
        cosine = normalized @ normalized.transpose(0, 1)
        eye = torch.eye(cosine.shape[0], device=cosine.device, dtype=cosine.dtype)
        return cosine * (1.0 - eye) + eye

    def _masked_inter_sample_similarity_loss(
        self,
        pred_all_types_log: torch.Tensor,
        true_all_types_log: torch.Tensor,
        true_sct_gep_present_mask: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Match batch-local cell-type-wise inter-sample cosine geometry in log space."""
        pairwise_cosine_fn = getattr(
            type(self),
            "_pairwise_cosine_similarity_matrix",
            VAE._pairwise_cosine_similarity_matrix,
        )
        active_mask = true_sct_gep_present_mask & (true_cell_prop >= cell_prop_threshold)  # (B, C)

        per_sample_loss = torch.zeros(
            (pred_all_types_log.shape[0],),
            device=pred_all_types_log.device,
            dtype=pred_all_types_log.dtype,
        )
        active_cell_type_count = 0

        for cell_type_idx in range(pred_all_types_log.shape[2]):
            sample_mask = active_mask[:, cell_type_idx]
            n_active = int(sample_mask.sum().item())
            if n_active < 2:
                continue

            pred_ct = pred_all_types_log[sample_mask, :, cell_type_idx]  # (B_active, G)
            true_ct = true_all_types_log[sample_mask, :, cell_type_idx]  # (B_active, G)

            pred_cosine = pairwise_cosine_fn(self, pred_ct)
            true_cosine = pairwise_cosine_fn(self, true_ct)

            off_diag_mask = ~torch.eye(n_active, device=pred_cosine.device, dtype=torch.bool)
            if not off_diag_mask.any():
                continue

            cell_type_loss = (pred_cosine - true_cosine).abs()[off_diag_mask].mean()
            per_sample_loss = per_sample_loss + cell_type_loss
            active_cell_type_count += 1

        if active_cell_type_count == 0:
            return per_sample_loss

        return per_sample_loss / float(active_cell_type_count)

    def _inter_sample_similarity_loss(
        self,
        pred_residual_log: torch.Tensor,
        true_sct_gep: torch.Tensor,
        true_sct_gep_present_mask: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Match batch-local cell-type-wise inter-sample cosine geometry in residual space."""
        true_residual_log = true_sct_gep - self.g_mean.unsqueeze(0)
        masked_similarity_fn = getattr(
            type(self),
            "_masked_inter_sample_similarity_loss",
            VAE._masked_inter_sample_similarity_loss,
        )
        return masked_similarity_fn(
            self,
            pred_all_types_log=pred_residual_log,
            true_all_types_log=true_residual_log,
            true_sct_gep_present_mask=true_sct_gep_present_mask,
            true_cell_prop=true_cell_prop,
            cell_prop_threshold=cell_prop_threshold,
        )

    def _full_sct_gep_inter_sample_similarity_loss(
        self,
        pred_sct_gep_log: torch.Tensor,
        true_sct_gep: torch.Tensor,
        true_sct_gep_present_mask: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Match batch-local cell-type-wise inter-sample cosine geometry for full sctGEPs."""
        masked_similarity_fn = getattr(
            type(self),
            "_masked_inter_sample_similarity_loss",
            VAE._masked_inter_sample_similarity_loss,
        )
        return masked_similarity_fn(
            self,
            pred_all_types_log=pred_sct_gep_log,
            true_all_types_log=true_sct_gep,
            true_sct_gep_present_mask=true_sct_gep_present_mask,
            true_cell_prop=true_cell_prop,
            cell_prop_threshold=cell_prop_threshold,
        )

    def _matched_sct_gep_supervision_loss(
        self,
        recon_x_all_types_cpm: torch.Tensor,
        true_sct_gep: torch.Tensor,
        true_sct_gep_present_mask: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Masked log-MSE between inferred cell-type GEPs and matched true sctGEPs."""
        recon_x_all_types_log = to_log_space(recon_x_all_types_cpm, self.scaling_factor)
        active_cell_type_mask = true_sct_gep_present_mask & (true_cell_prop >= cell_prop_threshold)
        active_gene_mask = active_cell_type_mask.unsqueeze(1).expand(-1, recon_x_all_types_log.shape[1], -1)
        active_gene_mask = active_gene_mask.to(dtype=recon_x_all_types_log.dtype)
        diff2 = (recon_x_all_types_log - true_sct_gep).pow(2) * active_gene_mask
        denom = active_gene_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return diff2.sum(dim=(1, 2)) / denom

    def _matched_sct_gep_residual_supervision_loss(
        self,
        pred_residual_log: torch.Tensor,
        true_sct_gep: torch.Tensor,
        true_sct_gep_present_mask: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Masked scaled-log MSE between predicted and true mean-centered SCT residuals."""
        true_residual_log = true_sct_gep - self.g_mean.unsqueeze(0)
        active_cell_type_mask = true_sct_gep_present_mask & (true_cell_prop >= cell_prop_threshold)
        active_gene_mask = active_cell_type_mask.unsqueeze(1).expand(-1, pred_residual_log.shape[1], -1)
        active_gene_mask = active_gene_mask.to(dtype=pred_residual_log.dtype)
        diff2 = (pred_residual_log - true_residual_log).pow(2) * active_gene_mask
        denom = active_gene_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return diff2.sum(dim=(1, 2)) / denom

    def _apply_cell_type_existence_shift(
        self,
        mu_types: torch.Tensor,
        cell_prop: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Shift cell-type latent means using a soft existence score from cell proportions."""
        if cell_prop is None:
            return None, None, mu_types

        threshold = float(
            getattr(self.data_config, "training_sct_gep_cell_prop_threshold", 0.0) or 0.0
        )
        denom = max(threshold, 1e-2)
        existence_logits = (cell_prop - threshold) / denom
        existence_probs = torch.sigmoid(existence_logits)

        shift_scale = float(getattr(self.model_config, "cell_type_existence_shift_scale", 0.0) or 0.0)
        if shift_scale == 0.0:
            return existence_logits, existence_probs, mu_types

        shift = shift_scale * (existence_probs - 0.5)
        mu_types_shifted = mu_types + shift.unsqueeze(1).to(device=device, dtype=mu_types.dtype)
        return existence_logits, existence_probs, mu_types_shifted

    def _cell_type_existence_loss(
        self,
        existence_logits: torch.Tensor,
        true_cell_prop: torch.Tensor,
        cell_prop_threshold: float,
    ) -> torch.Tensor:
        """Supervise thresholded cell-type existence using true cell proportions."""
        if true_cell_prop.ndim == 3:
            true_cell_prop = true_cell_prop.squeeze(-1)
        exist_target = (true_cell_prop >= cell_prop_threshold).to(dtype=existence_logits.dtype)
        loss_mat = F.binary_cross_entropy_with_logits(
            existence_logits,
            exist_target,
            reduction="none",
        )
        return loss_mat.mean(dim=-1)

    def _z_score_kl_loss(self, recon_x_all_types_cpm: torch.Tensor) -> torch.Tensor:
        """
        Regularize predicted per-cell-type GEPs by matching their empirical z-score distribution
        (computed across samples in the current batch) to a standard normal N(0, 1).

        recon_x_all_types_cpm: (B, G, C) in non-log CPM/TPM-like space.
        Z[b, g, c] = (X[b, g, c] - mean[g, c]) / std[g, c]

        Per-celltype overall normality:
        For each cell type c, we aggregate z-scores over all samples and genes in the batch
        and approximate the empirical distribution as N(mu_c, var_c). We then compute
        KL(N(mu_c, var_c) || N(0, 1)) and average over cell types.
        """
        if recon_x_all_types_cpm.ndim != 3:
            raise ValueError(
                f"Expected recon_x_all_types_cpm with shape (B, G, C), got {recon_x_all_types_cpm.shape}"
            )

        B, G, C = recon_x_all_types_cpm.shape
        assert self.g_mean_non_log.shape == (G, C), (
            f"g_mean_non_log shape mismatch: expected ({G}, {C}), "
            f"got {self.g_mean_non_log.shape}"
        )

        denom = torch.clamp(self.g_std_non_log, min=0.1).unsqueeze(0)  # (1, G, C)
        mu = self.g_mean_non_log.unsqueeze(0)                          # (1, G, C)
        z = (recon_x_all_types_cpm - mu) / denom                       # (B, G, C)
        z = torch.clamp(z, max=10, min=-10)  # Prevent extreme z-scores from destabilizing KL calculation.

        # Per-celltype: compute mu_z[C] and var_z[C] across (B, G)
        # (B, G, C) -> (C) after mean over B and G
        mu_z = z.mean(dim=(0, 1))                                       # (C,)
        var_z = z.var(dim=(0, 1), unbiased=False).clamp_min(EPS)        # (C,)

        # KL(N(mu, var) || N(0, 1)) = 0.5 * (var + mu^2 - 1 - log(var))
        kl_per_celltype = 0.5 * (var_z + mu_z.pow(2) - 1.0 - torch.log(var_z))  # (C,)
        kl_per_celltype = kl_per_celltype.clamp_min(0.0)
        kl = kl_per_celltype.mean()                                    # scalar

        # Expand back to batch dimension (B,) for consistency with existing code
        return kl.expand(B)

    def _cell_prop_supervision_loss(
        self,
        supervised_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the supervised cell-proportion loss per sample."""
        weighting_mode = getattr(self.model_config, "cell_prop_loss_weighting", "none")
        if weighting_mode == "none":
            weight = torch.ones_like(target)
        elif weighting_mode == "low_prop_inverse":
            low_prop_epsilon = float(
                getattr(self.model_config, "cell_prop_loss_low_prop_epsilon", 0.01)
            )
            weight_min, weight_max = getattr(
                self.model_config,
                "cell_prop_loss_weight_clamp",
                (1.0, 5.0),
            )
            weight = 1.0 / (target + low_prop_epsilon)
            weight = weight.clamp(min=weight_min, max=weight_max)
            weight = weight / weight.mean(dim=-1, keepdim=True).clamp_min(EPS)
        else:
            raise ValueError(
                f"Unsupported cell_prop_loss_weighting: {weighting_mode}"
            )

        loss_type = getattr(self.model_config, "cell_prop_loss_type", "mse")
        if loss_type == "mse":
            return (
                F.mse_loss(
                    supervised_pred,
                    target,
                    reduction="none",
                )
                * weight
            ).sum(dim=-1)

        alpha_weight = float(
            getattr(
                self.model_config,
                "cell_prop_loss_alpha_weight",
                getattr(self.model_config, "cell_prop_loss_kl_weight", 0.5),
            ) or 0.0
        )

        if loss_type == "l1_rmse":
            mae = (weight * torch.abs(supervised_pred - target)).sum(dim=-1)
            rmse = torch.sqrt(
                (weight * torch.square(supervised_pred - target)).sum(dim=-1).clamp_min(EPS)
            )
            return alpha_weight * mae + (1.0 - alpha_weight) * rmse

        if loss_type != "l1_kl":
            raise ValueError(f"Unsupported cell_prop_loss_type: {loss_type}")

        pred_safe = supervised_pred.clamp_min(EPS)
        target_safe = target.clamp_min(EPS)
        pred_safe = pred_safe / pred_safe.sum(dim=-1, keepdim=True).clamp_min(EPS)
        target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True).clamp_min(EPS)

        l1 = (weight * torch.abs(pred_safe - target_safe)).sum(dim=-1)
        kl = (
            weight
            * target_safe
            * (torch.log(target_safe) - torch.log(pred_safe))
        ).sum(dim=-1)
        return l1 + alpha_weight * kl

    def _cell_prop_dirichlet_loss(
        self,
        y: Optional[torch.Tensor],
        dd_alpha: Optional[torch.Tensor],
        pred_cell_prop: Optional[torch.Tensor],
        raw_non_cancer_cell_prop: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dirichlet KL + supervised cell proportion loss.
        Returns per-sample vectors: (kld_p, cell_prop_loss), both shape (B,).
        """
        kld_p = torch.zeros(batch_size, device=device)
        cell_prop_loss = torch.zeros(batch_size, device=device)

        if (
            self.model_config.predict_cell_prop
            and self.cell_prop_activation_function == "softplus"
            and dd_alpha is not None
        ):
            # KL(Posterior || Prior), prior is Uniform Dirichlet(alpha=1)
            prior_alpha = torch.ones_like(dd_alpha)
            prior_dist = Dirichlet(prior_alpha)
            posterior_dist = Dirichlet(dd_alpha)
            kld_p = kl_divergence(posterior_dist, prior_dist)

        if has_usable_labels(y) and self.model_config.predict_cell_prop:
            if self.cell_prop_activation_function == "softplus" and dd_alpha is not None:
                supervised_pred = dirichlet_mean(dd_alpha)
                target = y
            elif self.cell_prop_activation_function == "sigmoid" and pred_cell_prop is not None:
                supervised_pred = raw_non_cancer_cell_prop
                if supervised_pred is None:
                    supervised_pred = remove_cancer_cell_type(pred_cell_prop, self.cancer_cell_type_index)
                target = remove_cancer_cell_type(y, self.cancer_cell_type_index)
            elif (
                self.cell_prop_activation_function in {"softmax", "sigmoid_all_norm"}
                and pred_cell_prop is not None
            ):
                supervised_pred = pred_cell_prop
                target = y
            else:
                supervised_pred = None
                target = None

            if supervised_pred is not None and target is not None:
                cell_prop_loss = VAE._cell_prop_supervision_loss(
                    self,
                    supervised_pred=supervised_pred,
                    target=target,
                )

        return kld_p, cell_prop_loss

    def _latent_kld_loss(
        self,
        mu_types: torch.Tensor,
        logvar_types: torch.Tensor,
        mu_prior: Optional[torch.Tensor],
        logvar_mean: torch.Tensor,
        mu_mean: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Latent KL term.
        - kld_type == 'sep': KL computed per-celltype posterior (B, L, C)
        - kld_type == 'ave': KL computed from fused/averaged posterior (B, L)
        Returns per-sample vector, shape (B,).
        """
        lo = self.model_config.loss_coefficient
        kld_type = lo.kld_type
        batch_size = mu_types.shape[0]

        if kld_type == "sep":
            # Calculate KL for each Cell Type individually.
            # mu_types/logvar_types: (B, L, C)
            if mu_prior is not None:
                # Expand prior to match batch: (C, L) -> (1, L, C)
                mu_prior_expand = mu_prior.t().unsqueeze(0)

                # General KL for Gaussian:
                # KL = 0.5 * sum(exp(logvar) + (mu - mu_prior)^2 - 1 - logvar)
                # Assuming prior logvar is 0 (sigma=1).
                term1 = logvar_types.exp()                           # sigma_1^2
                term2 = (mu_types - mu_prior_expand).pow(2)         # (mu_1 - mu_2)^2
                return 0.5 * torch.sum(term1 + term2 - 1 - logvar_types, dim=1).sum(dim=-1)

            # Standard N(0,1) prior
            return -0.5 * torch.sum(
                1 + logvar_types - mu_types.pow(2) - logvar_types.exp(),
                dim=1
            ).sum(dim=-1)

        if kld_type == "ave":
            # Use fused/averaged mu_mean, logvar_mean against Standard Normal.
            return -0.5 * torch.sum(1 + logvar_mean - mu_mean.pow(2) - logvar_mean.exp(), dim=-1)

        return torch.zeros(batch_size, device=device)

    @staticmethod
    def _build_hierarchy_repulsion_margin(
        hierarchical_targets: torch.Tensor,
        *,
        base_margin: float = 0.0,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Build a pair-specific cosine margin from binary hierarchy codes.

        Related cell types receive a larger allowed cosine similarity, so the
        repulsion term stays soft for sibling subtypes while remaining strict
        for unrelated pairs.
        """
        if hierarchical_targets.ndim != 2:
            raise ValueError(
                "hierarchical_targets must have shape (n_cell_types, n_codes), "
                f"got {hierarchical_targets.shape}"
            )
        if base_margin < 0 or alpha < 0:
            raise ValueError(
                f"base_margin and alpha must be non-negative, got {base_margin}, {alpha}"
            )

        normalized_targets = F.normalize(
            hierarchical_targets.float(),
            p=2,
            dim=1,
            eps=EPS,
        )
        hierarchy_similarity = normalized_targets @ normalized_targets.transpose(0, 1)
        hierarchy_similarity = torch.clamp(hierarchy_similarity, min=0.0, max=1.0)
        margin = base_margin + alpha * hierarchy_similarity
        margin.fill_diagonal_(0.0)
        return margin

    def _repulsion_loss(
            self,
            mu_types: torch.Tensor,
            gamma: float,
            margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Repulsion between cell-type centroids in latent space.
        Uses a cosine-similarity hinge penalty to encourage separation between
        different cell-type embeddings.

        Loss per pair:  max(0, cos(i, j) - margin_ij)
        → zero gradient when cosine similarity is <= margin_ij.
        → linear penalty when embeddings become more similar (cosine increases).

        When available, `self.hierarchical_repulsion_margin` supplies a
        pair-specific margin matrix derived from hierarchy codes, allowing
        sibling subtypes to remain more similar than unrelated cell types.

        Args:
            mu_types : (B, L, C)  cell-type centroid means in latent space.
            gamma    : float       loss weight; if 0, returns zeros immediately.
            margin   : float       fallback scalar margin used only when a
                                  pair-specific hierarchy margin matrix is
                                  unavailable.

        Returns:
            repulsion_loss : (B,)  per-sample repulsion scalar.
        """
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device

        if gamma == 0 or n_cell_types < 2:
            return torch.zeros(batch_size, device=device)

        x = mu_types.permute(0, 2, 1)  # (B, C, L)
        x = F.normalize(x, p=2, dim=2, eps=EPS)
        cos_matrix = x @ x.transpose(1, 2)  # (B, C, C)

        margin_matrix = getattr(self, "hierarchical_repulsion_margin", None)
        if margin_matrix is None or tuple(margin_matrix.shape) != (n_cell_types, n_cell_types):
            margin_matrix = torch.full(
                (n_cell_types, n_cell_types),
                fill_value=margin,
                device=device,
                dtype=mu_types.dtype,
            )
            margin_matrix.fill_diagonal_(0.0)
        else:
            margin_matrix = margin_matrix.to(device=device, dtype=mu_types.dtype)

        # Upper triangle mask — count each pair (i, j) only once
        triu_mask = torch.triu(
            torch.ones(n_cell_types, n_cell_types, device=device), diagonal=1
        ).unsqueeze(0)  # (1, C, C)

        # Hinge penalty: penalize only pairs with cosine similarity above their allowed margin.
        hinge = torch.clamp(cos_matrix - margin_matrix.unsqueeze(0), min=0.0)  # (B, C, C)
        hinge = hinge * triu_mask  # upper triangle only

        # Normalize by number of pairs so loss scale is independent of C
        n_pairs = n_cell_types * (n_cell_types - 1) / 2
        repulsion_loss = hinge.sum(dim=(1, 2)) / n_pairs  # (B,)

        return repulsion_loss

    def _attractor_loss(
        self,
        mu_types: torch.Tensor,
        attractor_weight: float,
    ) -> torch.Tensor:
        """
        Attractor loss: L2-ball constraint around the origin for cell-type embeddings.
        → zero gradient when embeddings are inside the ball.
        → linear penalty when embeddings are outside the ball.
        Returns per-sample vector, shape (B,).
        
        Args:
            mu_types : (B, L, C)  cell-type centroid means in latent space.
            attractor_weight : float       loss weight; if 0, returns zeros immediately.
        """
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device

        if attractor_weight == 0 or n_cell_types < 2:
            return torch.zeros((batch_size,), device=device)

        x = mu_types.permute(0, 2, 1)  # (B, C, L)
        l2_norm = torch.linalg.vector_norm(x, ord=2, dim=2)  # (B, C)
        radius = 1.0
        violation = torch.clamp(l2_norm - radius, min=0.0)  # (B, C)
        loss = violation.mean(dim=1)  # (B,)
        return loss

    def _hierarchical_code_loss(
        self,
        mu_types: torch.Tensor,
        hierarchical_code_weight: float,
    ) -> torch.Tensor:
        batch_size, _, n_cell_types = mu_types.shape
        device = mu_types.device

        if hierarchical_code_weight == 0 or n_cell_types < 1:
            return torch.zeros((batch_size,), device=device)

        targets = getattr(self, "hierarchical_code_targets", None)
        if targets is None or targets.shape[0] != n_cell_types:
            raise ValueError("hierarchical_code_targets is missing or has wrong shape")

        x = mu_types.permute(0, 2, 1)  # (B, C, L)
        logits = self.hierarchical_code_head(x)  # (B, C, 8)
        y = targets.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (B, C, 8)
        loss_mat = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # (B, C, 8)
        return loss_mat.mean(dim=(1, 2))

    # =========================================================================
    # Gene Weighting
    # =========================================================================
    def compute_gene_weights(
        self,
        low_weight_coef: float = 1.0,
        clamp_range: Tuple[float, float] = (0.5, 2.0),
    ) -> torch.Tensor:
        """
        Compute per-gene weights, higher weights for lower-variance genes.

        Steps:
        1) Raw weight from inverse std.
        2) Normalize so average weight per cell type = 1.
        3) Clamp to avoid extremes.
        4) Optional exponent scaling.
        """
        # 1) Choose raw weight based on variance of gene expression in each cell type.
        w_g = 1.0 / (self.g_std + EPS)  # (G, C)

        # 2) Normalize so that average weight = 1.
        w_g = w_g / w_g.mean(dim=0, keepdim=True)

        # 3) Clamp to avoid extremes.
        w_g = torch.clamp(w_g, min=clamp_range[0], max=clamp_range[1])

        # 4) Optionally re-shape low-expression emphasis.
        if low_weight_coef != 1.0:
            w_g = w_g.pow(low_weight_coef)

        return w_g

    # =========================================================================
    # Anchors
    # =========================================================================
    @staticmethod
    def make_orthonormal_anchors(latent_dim: int, radius: float = 1.0) -> torch.Tensor:
        """
        Generate orthonormal anchors for latent space via QR decomposition.
        """
        random_weights = torch.randn(latent_dim, latent_dim)
        q, _ = torch.linalg.qr(random_weights)
        return radius * q.t()

    # =========================================================================
    # Encoder fusion
    # =========================================================================
    def _fuse_encoder_posteriors(
        self,
        mu_lists_celltype: List[torch.Tensor],
        logvar_lists_celltype: List[torch.Tensor],
        mu_list_overall: List[torch.Tensor],
        logvar_list_overall: List[torch.Tensor],
        strategy: Literal["poe_posterior", "avg_posterior", "pre_latent"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse multiple encoder outputs.

        Returns:
            fused_mu_celltype:      (B, L, C)
            fused_logvar_celltype:  (B, L, C)
            fused_mu_overall:       (B, L)
            fused_logvar_overall:   (B, L)
        """
        if strategy == "pre_latent":
            # Placeholder:
            # Pre-latent fusion requires explicit architecture support in encoders
            # (e.g., shared fusion block before mu/logvar heads).
            # We fallback to posterior PoE for safety.
            strategy = "poe_posterior"

        if strategy == "poe_posterior":
            fused_mu_celltype, fused_logvar_celltype = self._poe_fuse_core(
                mu_list=mu_lists_celltype,
                logvar_list=logvar_lists_celltype,
            )
            fused_mu_overall, fused_logvar_overall = self._poe_fuse_core(
                mu_list=mu_list_overall,
                logvar_list=logvar_list_overall,
            )
            return fused_mu_celltype, fused_logvar_celltype, fused_mu_overall, fused_logvar_overall

        if strategy == "avg_posterior":
            # Simple arithmetic averaging baseline.
            fused_mu_celltype = torch.mean(torch.stack(mu_lists_celltype, dim=0), dim=0)
            fused_logvar_celltype = torch.mean(torch.stack(logvar_lists_celltype, dim=0), dim=0)
            fused_mu_overall = torch.mean(torch.stack(mu_list_overall, dim=0), dim=0)
            fused_logvar_overall = torch.mean(torch.stack(logvar_list_overall, dim=0), dim=0)
            return fused_mu_celltype, fused_logvar_celltype, fused_mu_overall, fused_logvar_overall

        raise ValueError(f"Unknown fusion strategy: {strategy}")

    @staticmethod
    def _poe_fuse_core(
        mu_list: List[torch.Tensor],
        logvar_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core Product-of-Experts (PoE) fusion logic.

        For diagonal Gaussians:
            precision_i = exp(-logvar_i)
            precision_sum = sum_i precision_i
            mu_fused = sum_i(mu_i * precision_i) / precision_sum
            logvar_fused = -log(precision_sum)
        """
        if not mu_list or not logvar_list:
            raise ValueError("Input lists for PoE fusion cannot be empty.")
        if len(mu_list) != len(logvar_list):
            raise ValueError("Mismatch in number of experts for mus and logvars.")
        if len(mu_list) == 1:
            return mu_list[0], logvar_list[0]

        # Stack expert parameters. [n_encoder, batch_size, latent_dim, n_cell_type]
        mus_stacked = torch.stack(mu_list, dim=0)
        logvars_stacked = torch.stack(logvar_list, dim=0)

        # Calculate precisions: P_i = 1 / sigma_i^2 = exp(-logvar_i).
        precisions_stacked = torch.exp(-logvars_stacked)

        # Sum of precisions: P_poe = sum(P_i). [batch_size, latent_dim, n_cell_type]
        sum_of_precisions = torch.sum(precisions_stacked, dim=0)

        # Fused log variance: logvar_poe = -log(P_poe + eps).
        fused_logvar = -torch.log(sum_of_precisions + EPS)

        # Weighted sum of means: sum(mu_i * P_i).
        sum_of_weighted_mus = torch.sum(mus_stacked * precisions_stacked, dim=0)

        # Fused mean: mu_poe = sum(mu_i * P_i) / (P_poe + eps) = sum_of_weighted_mus * e^{fused_logvar}.
        fused_mu = sum_of_weighted_mus * torch.exp(fused_logvar)

        return fused_mu, fused_logvar


def to_non_log_space(
    x_log_scaled: torch.Tensor,
    scaling_factor: float,
    *,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert scaled-log2 features to non-log space.

    Expected transform pair:
        x_log_scaled = log2(x_non_log + 1) / scaling_factor
        x_non_log    = 2^(x_log_scaled * scaling_factor) - 1

    Args:
        x_log_scaled:
            Tensor in scaled log2 space.
        scaling_factor:
            Positive scaling factor used in preprocessing.
        clamp_min:
            Optional lower bound after inverse transform.
            Keep 0.0 for expression-like non-negative values.

    Returns:
        Tensor in non-log space.

    Notes:
        - Uses torch.pow for consistency with your current code.
        - Clamp avoids tiny negative values from numerical noise.
    """
    if scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be > 0, got {scaling_factor}")

    x_non_log = torch.pow(2.0, x_log_scaled * scaling_factor) - 1.0
    if clamp_min is not None:
        x_non_log = torch.clamp(x_non_log, min=clamp_min)
    return x_non_log


def to_log_space(
    x_non_log: torch.Tensor,
    scaling_factor: float,
    *,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert non-log features back to scaled-log2 space.

    Expected transform pair:
        x_log_scaled = log2(x_non_log + 1) / scaling_factor
        x_non_log    = 2^(x_log_scaled * scaling_factor) - 1

    Args:
        x_non_log:
            Tensor in non-log space (typically >= 0).
        scaling_factor:
            Positive scaling factor used in preprocessing.
        clamp_min:
            Clamp input before log to avoid invalid values (default 0).

    Returns:
        Tensor in scaled log2 space.
    """
    if scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be > 0, got {scaling_factor}")

    x_non_log = torch.clamp(x_non_log, min=clamp_min)
    x_log_scaled = torch.log2(x_non_log + 1.0) / scaling_factor
    return x_log_scaled
