import importlib
import io
import logging
from collections import OrderedDict
from typing import Any, Tuple

try:
    import pickle5 as pickle
except ImportError:
    import pickle
import torch
from torch.distributions import Gamma
import random
import numpy as np

EPS = 1e-6
# MAX_LOGSTD = 10
LOGVAR_CLAMP_MIN = -10
LOGVAR_CLAMP_MAX = 10
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
