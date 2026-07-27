import importlib
import io
import logging
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

try:
    import pickle5 as pickle
except ImportError:
    import pickle
import torch
import torch.nn.functional as F
from torch.distributions import Gamma
import random
import numpy as np

EPS = 1e-8
# MAX_LOGSTD = 10
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 15
NETWORK_CUTOFF = 0.5

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from huggingface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            # self_dict = {k: v for (k, v) in self.items()}
            # return self_dict[k]
            return super().__getitem__(k)
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any, ...]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        # return tuple(self[k] for k in self.keys())
        return tuple(super().__getitem__(k) for k in self.keys())


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            if module.startswith('torch_geometric.nn.sequential_'):
                module = 'torch_geometric.nn'
                name = 'Sequential'
            return super().find_class(module, name)


def reparameterize_gaussian(mu, logvar):
    """Samples from a Gaussian distribution (N(0, I)) using the reparameterization trick."""
    std = torch.exp(0.5 * logvar)  # sigma
    eps = torch.randn_like(std)  # epsilon ~ N(0, 1)
    return mu + std * eps  # reparameterization trick, z = mu + sigma * epsilon


def reparameterize_dirichlet(alpha, device: torch.device):
    """
    Use the Gamma distribution reparameterization trick for Dirichlet.
    Args:
        alpha (torch.Tensor): The Dirichlet parameters. (batch_size, n_cell_types)
        device (torch.device): The device to use for the computation.
    Returns: samples from Dirichlet distribution.
    Requires PyTorch 1.8+ for Gamma.rsample().
    """
    try:
        gamma_dis = Gamma(concentration=alpha, rate=torch.tensor(1.0, device=device))
        gamma_samples = gamma_dis.rsample()  # shape: (batch_size, n_cell_types)
    except NotImplementedError:
        # Fallback to use cpu without MPS
        alpha_cup = alpha.cpu()
        gamma_dis = Gamma(concentration=alpha_cup, rate=torch.tensor(1.0, device="cpu"))
        gamma_samples = gamma_dis.rsample()
    if device.type != "cpu":
        gamma_samples = gamma_samples.to(device)
    # Normalize the samples to sum to 1 to get Dirichlet samples
    p = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
    return p


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    """Return the deterministic mean of a Dirichlet distribution."""
    return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _cell_type_from_avg_column(column_name: str) -> str:
    """Normalize `*_avg`/`* avg` column names to plain cell type names."""
    if column_name.endswith("_avg"):
        return column_name[:-4]
    if column_name.endswith(" avg"):
        return column_name[:-4]
    return column_name[:-3]


def read_cell_types_from_gene_mean_std(gene_mean_std_fp: str | Path) -> list[str]:
    """Read cell types from the header of the gene mean/std CSV."""
    gene_mean_std_fp = Path(gene_mean_std_fp)
    with gene_mean_std_fp.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    avg_columns = [column for column in header[1:] if column.endswith("avg")]
    if not avg_columns:
        raise ValueError(
            f"Could not find any '*avg' columns in gene mean/std file: {gene_mean_std_fp}"
        )
    return [_cell_type_from_avg_column(column) for column in avg_columns]


def read_cell_types_from_file(cell_type_fp: str | Path) -> list[str]:
    """Read cell type names from a plain-text file."""
    cell_type_fp = Path(cell_type_fp)
    return [
        line.strip()
        for line in cell_type_fp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_cancer_cell_type_index(
    cell_types: Sequence[str],
    cancer_cell_type_name: Optional[str],
) -> int:
    """Resolve the cancer cell type index from an ordered cell type list."""
    if not cancer_cell_type_name or not cancer_cell_type_name.strip():
        raise ValueError(
            "cancer_cell_type_name must be provided when "
            "cell_prop_activation_function='sigmoid'."
        )

    normalized_name = cancer_cell_type_name.strip()
    try:
        return list(cell_types).index(normalized_name)
    except ValueError as exc:
        raise ValueError(
            f"Could not find cancer cell type '{normalized_name}' in cell types: "
            f"{list(cell_types)}"
        ) from exc


def resolve_cancer_cell_type_index_from_config(model_config: Any) -> int:
    """Resolve the cancer cell type index using the available config file paths."""
    gene_mean_std_fp = getattr(model_config, "gene_mean_std_fp", None)
    if gene_mean_std_fp:
        cell_types = read_cell_types_from_gene_mean_std(gene_mean_std_fp)
    else:
        cell_type_fp = getattr(model_config, "cell_type_fp", None)
        if not cell_type_fp:
            raise ValueError(
                "Cannot resolve cancer_cell_type_name because neither gene_mean_std_fp "
                "nor cell_type_fp is set."
            )
        cell_types = read_cell_types_from_file(cell_type_fp)

    return resolve_cancer_cell_type_index(
        cell_types=cell_types,
        cancer_cell_type_name=getattr(model_config, "cancer_cell_type_name", None),
    )


def get_cell_prop_head_output_dim(
    n_cell_types: int,
    activation_function: str,
) -> int:
    """Return the output dimension required by the cell proportion head."""
    if activation_function == "sigmoid":
        if n_cell_types < 2:
            raise ValueError(
                "Sigmoid cell proportion prediction requires at least 2 cell types."
            )
        return n_cell_types - 1
    return n_cell_types


def remove_cancer_cell_type(
    cell_prop: torch.Tensor,
    cancer_cell_type_index: int,
) -> torch.Tensor:
    """Return the non-cancer subset of a full cell proportion tensor."""
    return torch.cat(
        (
            cell_prop[..., :cancer_cell_type_index],
            cell_prop[..., cancer_cell_type_index + 1:],
        ),
        dim=-1,
    )


def build_cell_prop_from_head_output(
    head_output: torch.Tensor,
    activation_function: str,
    n_cell_types: int,
    eps: float = EPS,
    cancer_cell_type_index: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convert raw head outputs into final cell proportions and optional Dirichlet alpha."""
    if activation_function == "softplus":
        dd_alpha = F.softplus(head_output) + eps
        return dirichlet_mean(dd_alpha), dd_alpha

    if activation_function != "sigmoid":
        raise ValueError(f"Unsupported cell proportion activation: {activation_function}")

    if cancer_cell_type_index is None:
        raise ValueError(
            "cancer_cell_type_index must be set when cell_prop_activation_function='sigmoid'."
        )

    if head_output.shape[-1] != n_cell_types - 1:
        raise ValueError(
            f"Sigmoid cell proportion head output must have size {n_cell_types - 1}, "
            f"got {head_output.shape[-1]}."
        )

    non_cancer_prop = torch.sigmoid(head_output)
    non_cancer_sum = non_cancer_prop.sum(dim=-1, keepdim=True)
    scale = torch.clamp(non_cancer_sum, min=1.0)
    non_cancer_prop = non_cancer_prop / scale

    cancer_prop = (1.0 - non_cancer_prop.sum(dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)
    full_cell_prop = torch.zeros(
        head_output.shape[0],
        n_cell_types,
        device=head_output.device,
        dtype=head_output.dtype,
    )
    full_cell_prop[..., :cancer_cell_type_index] = non_cancer_prop[..., :cancer_cell_type_index]
    full_cell_prop[..., cancer_cell_type_index] = cancer_prop.squeeze(-1)
    full_cell_prop[..., cancer_cell_type_index + 1:] = non_cancer_prop[..., cancer_cell_type_index:]
    full_cell_prop = full_cell_prop / full_cell_prop.sum(dim=-1, keepdim=True).clamp_min(eps)
    return full_cell_prop, None


def has_usable_labels(labels: Any) -> bool:
    """Return True when labels are present and contain at least one value."""
    if labels is None:
        return False
    if isinstance(labels, torch.Tensor):
        return labels.numel() > 0
    if isinstance(labels, np.ndarray):
        return labels.size > 0
    if isinstance(labels, (list, tuple)):
        return len(labels) > 0
    return False


def set_seed(seed: int):
    """
    Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``

    Args:

        seed (int): The seed to be applied
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
