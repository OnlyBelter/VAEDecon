import json
import logging
import os

import torch.nn as nn

# from ..base.base_utils import hf_hub_is_available

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AutoModel(nn.Module):
    "Utils class allowing to reload any :class:`pythae.models` automatically"

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def load_from_folder(cls, dir_path: str):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        with open(os.path.join(dir_path, "model_config.json")) as f:
            model_name = json.load(f)["name"]

        if model_name == "VAEConfig":
            from ..vae import VAE

            model = VAE.load_from_folder(dir_path=dir_path)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model
