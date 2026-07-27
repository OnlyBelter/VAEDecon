from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import logging
import warnings

from ...configs import ModelConfig
from ...models.base import (ModelOutput, build_cell_prop_from_head_output,
                            get_cell_prop_head_output_dim,
                            LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX, EPS, BaseEncoder)
from ...models.base import resolve_cancer_cell_type_index_from_config
from vaedecon.models.base.positional_encoding import PositionalEncoding
from .mlp import EncoderMLP
from ..gnn import EncoderSGNN

# Configure logging
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
# Assuming BaseEncoder, ModelOutput, PositionalEncoding, reparameterize_dirichlet,
# BaseModelConfig (or VAEConfig), EPS, LOGVAR_CLAMP_MIN, LOGVAR_CLAMP_MAX are defined
# Also assuming EncoderMLP and EncoderSGNN class definitions are available from your code.

class EncoderHybrid(BaseEncoder):
    """
    A hybrid encoder that fuses features from an MLP encoder and a GNN encoder.
    """

    def __init__(self,
                 args: ModelConfig,  # Overall config for this hybrid encoder's output
                 mlp_encoder: EncoderMLP,  # Pre-initialized MLP encoder instance
                 gnn_encoder: EncoderSGNN,  # Pre-initialized GNN encoder instance
                 position_encoding: Optional[PositionalEncoding] = None,
                 fusion_strategy: str = "concat",  # "concat", "average", "sum"
                 ):
        super().__init__()
        self.args = args
        self.mlp_encoder = mlp_encoder
        self.gnn_encoder = gnn_encoder
        self.fusion_strategy = fusion_strategy.lower()
        self.fusion_hidden_dims: Union[Tuple[int, ...], int] = args.fusion_hidden_dims  # MLP layers after fusion
        self.fusion_dropout_rate: Union[Tuple[float, ...], float] = self.args.fusion_dropout_rate  # Dropout rates for fusion MLP layers
        self.latent_dim = args.latent_dim
        self.n_cell_types = args.n_cell_types
        self.predict_cell_prop = args.predict_cell_prop
        self.using_positional_encoding = args.using_positional_encoding
        self.cell_prop_activation_function = args.cell_prop_activation_function
        self.cancer_cell_type_index = None

        self.device_param = nn.Parameter(torch.empty(0))  # For device inference

        # Get feature dimensions from child encoders
        # This assumes child encoders have these attributes accessible
        mlp_feature_dim: int = self.mlp_encoder.hidden_dims[-1]
        gnn_feature_dim = self.gnn_encoder.embd_col_dim  # from EncoderSGNN's self.cell_mlp output

        self.mlp_proj = nn.Identity()  # Default
        self.gnn_proj = nn.Identity()  # Default

        if self.fusion_strategy == "concat":
            self.fused_input_dim_to_mlp = mlp_feature_dim + gnn_feature_dim
        elif self.fusion_strategy in ["average", "sum", "add"]:
            # If dimensions differ, project one or both to a common dimension for averaging/summing
            # Here, projecting both to the smaller of the two, or a predefined fusion_projection_dim
            # For simplicity, let's assume they should be made equal if not.
            # A more robust way is to project both to a target `fusion_common_dim`.
            # For now, let's project GNN to MLP's dim if different.
            if mlp_feature_dim != gnn_feature_dim:
                logger.warning(
                    f"MLP ({mlp_feature_dim}) and GNN ({gnn_feature_dim}) feature dims differ for '{self.fusion_strategy}'."
                    f" GNN features will be projected to MLP feature dim.")
                self.gnn_proj = nn.Linear(gnn_feature_dim, mlp_feature_dim)
                self.fused_input_dim_to_mlp = mlp_feature_dim
            else:
                self.fused_input_dim_to_mlp = mlp_feature_dim
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Fusion MLP (processes the combined features)
        _fusion_hidden_dims = self.fusion_hidden_dims if self.fusion_hidden_dims is not None else []

        if isinstance(self.fusion_dropout_rate, (list, tuple)):
            if _fusion_hidden_dims and len(self.fusion_dropout_rate) != len(_fusion_hidden_dims):
                raise ValueError("Length of fusion_dropout_rate list must match fusion_hidden_dims")
            _fusion_dropout_rates = self.fusion_dropout_rate
        else:
            _fusion_dropout_rates = [float(self.fusion_dropout_rate)] * len(_fusion_hidden_dims)

        self.fusion_mlp_layers = nn.Sequential()
        current_fusion_size = self.fused_input_dim_to_mlp
        if not _fusion_hidden_dims:  # If no hidden dims, fc_mu_logvar takes fused_input_dim_to_mlp directly
            self.final_fused_embedding_dim = current_fusion_size
        else:
            for i, hidden_size in enumerate(_fusion_hidden_dims):
                self.fusion_mlp_layers.add_module(f"fusion_linear_{i}", nn.Linear(current_fusion_size, hidden_size))
                self.fusion_mlp_layers.add_module(f"fusion_norm_{i}", nn.LayerNorm(hidden_size, eps=EPS))
                self.fusion_mlp_layers.add_module(f"fusion_act_{i}", nn.GELU())
                if _fusion_dropout_rates and i < len(_fusion_dropout_rates) and _fusion_dropout_rates[i] > 0:
                    self.fusion_mlp_layers.add_module(f"fusion_dropout_{i}", nn.Dropout(p=_fusion_dropout_rates[i]))
                current_fusion_size = hidden_size
            self.final_fused_embedding_dim = current_fusion_size

        # Output heads (operate on the output of fusion_mlp_layers)
        self.fc_mu_logvar = nn.Linear(self.final_fused_embedding_dim, self.n_cell_types * self.latent_dim * 2)
        if self.predict_cell_prop:
            if self.cell_prop_activation_function == "sigmoid":
                self.cancer_cell_type_index = resolve_cancer_cell_type_index_from_config(args)
            self.fc_dd_alpha = nn.Linear(
                self.final_fused_embedding_dim,
                get_cell_prop_head_output_dim(
                    n_cell_types=self.n_cell_types,
                    activation_function=self.cell_prop_activation_function,
                ),
            )

        self.position_encoding_module: Optional[PositionalEncoding] = None
        if self.using_positional_encoding:
            if position_encoding is None:
                raise ValueError("positional_encoding_class must be provided if using_positional_encoding is True.")
            # Ensure PositionalEncoding class is instantiated with correct args if needed
            # e.g. positional_encoding_class(num_embeddings=self.n_cell_types, embedding_dim=self.latent_dim)
            self.position_encoding_module = position_encoding

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, eps: float = EPS) -> ModelOutput:
        current_device = self.device_param.device
        x = x.to(current_device)
        if y is not None and len(y) > 0:
            y = y.to(current_device)

        # 1. Extract features from child encoders
        mlp_features = self.mlp_encoder.extract_features(x)  # Shape: (batch_size, mlp_feature_dim)
        gnn_features = self.gnn_encoder.extract_features(x)  # Shape: (batch_size, gnn_feature_dim)

        # 2. Fuse features
        if self.fusion_strategy == "concat":
            fused_features = torch.cat([mlp_features, gnn_features], dim=-1)
        elif self.fusion_strategy in ["average", "sum", "add"]:
            mlp_f = self.mlp_proj(mlp_features)
            gnn_f = self.gnn_proj(gnn_features)  # gnn_proj might be nn.Identity or a Linear layer
            if self.fusion_strategy == "average":
                fused_features = (mlp_f + gnn_f) / 2.0
            else:  # sum or add
                fused_features = mlp_f + gnn_f
        else:
            # This case should ideally be caught in __init__
            raise ValueError(f"Internal error: Unknown fusion strategy: {self.fusion_strategy}")

        # 3. Process through fusion MLP
        final_embedding = self.fusion_mlp_layers(fused_features)

        # 4. Output Heads for mu, logvar
        mu_logvar_flat = self.fc_mu_logvar(final_embedding)
        mu_logvar_structured = mu_logvar_flat.view(x.size(0), self.n_cell_types, self.latent_dim * 2)
        mu_all_types_raw, logvar_all_types_raw = torch.chunk(mu_logvar_structured, 2, dim=-1)

        mu_all_types = mu_all_types_raw.permute(0, 2, 1).contiguous()  # (B, Latent, C)
        logvar_all_types = logvar_all_types_raw.permute(0, 2, 1).contiguous()  # (B, Latent, C)
        logvar_all_types = torch.clamp(logvar_all_types, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

        mu_mean = mu_all_types.mean(dim=2)  # (B, Latent) - average over cell types
        logvar_mean = logvar_all_types.mean(dim=2)  # (B, Latent)

        # 5. Cell Proportions
        cell_prop_final = None  # Initialize
        dd_alpha_final = None  # Initialize
        if self.predict_cell_prop:
            cell_prop_final, dd_alpha_final = build_cell_prop_from_head_output(
                head_output=self.fc_dd_alpha(final_embedding),
                activation_function=self.cell_prop_activation_function,
                n_cell_types=self.n_cell_types,
                eps=eps,
                cancer_cell_type_index=self.cancer_cell_type_index,
            )
        elif y is not None and len(y) > 0:
            cell_prop_final = y  # Use ground truth y directly
        else:
            cell_prop_final = None
            warnings.warn('If self.predict_cell_prop is False, '
                          'y (cell proportions of cell types) must be provided during training. '
                          'It can be predicted by DeSide.')
            # raise ValueError('If self.predict_cell_prop is False, y (ground truth cell proportions) must be provided.')

        # 6. Positional Encoding
        if self.using_positional_encoding and self.position_encoding_module is not None and cell_prop_final is not None:
            pe_matrix = self.position_encoding_module()  # Should be (n_cell_types, latent_dim)
            # Ensure pe_matrix is on the correct device (module should handle this ideally)
            if pe_matrix.device != current_device: pe_matrix = pe_matrix.to(current_device)

            current_cell_prop_for_pe = cell_prop_final
            if cell_prop_final.ndim == 3 and cell_prop_final.shape[-1] == 1:  # if (B, N, 1)
                current_cell_prop_for_pe = cell_prop_final.squeeze(-1)  # -> (B, N)

            exists = (current_cell_prop_for_pe >= 0.01).float()  # (B, n_cell_types)

            pe_to_add = pe_matrix.t().unsqueeze(0)  # (1, latent_dim, n_cell_types)
            exists_mask = exists.unsqueeze(1)  # (B, 1, n_cell_types)

            mu_all_types = mu_all_types + (pe_to_add * exists_mask)
            mu_mean = mu_all_types.mean(dim=2)  # Recalculate mu_mean

        # 7. Populate ModelOutput
        output = ModelOutput()
        output['mu_all_types'] = mu_all_types
        output['logvar_all_types'] = logvar_all_types
        output['mu_mean'] = mu_mean
        output['logvar_mean'] = logvar_mean
        output['dd_alpha'] = dd_alpha_final

        if cell_prop_final is not None:  # Ensure cell_prop_final is populated before accessing
            output_cell_prop = cell_prop_final
            if cell_prop_final.ndim == 3 and cell_prop_final.shape[-1] == 1:
                output_cell_prop = cell_prop_final.squeeze(-1)
            output['cell_prop'] = output_cell_prop
            output['cell_type_existed'] = (output_cell_prop >= 0.01).float()
        else:
            output['cell_prop'] = None

        return output

    def get_config(self):  # Basic config
        config_params = self.args.to_dict() if hasattr(self.args, 'to_dict') else vars(self.args)
        config_params['fusion_strategy'] = self.fusion_strategy
        config_params['fusion_hidden_dims'] = [m.out_features for m in self.fusion_mlp_layers if
                                               isinstance(m, nn.Linear)]  # Approx.
        return {
            "params": {"args": config_params},  # Needs more careful construction for full reproducibility
            "module_name": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }
