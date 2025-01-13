import inspect
import logging
import os
import sys
from http.cookiejar import LoadError
from typing import Optional, Dict, Any

import cloudpickle
import torch
import lightning as L
from ...data.datasets import BaseDataset, DatasetOutput
from ...models.auto_model import AutoConfig
from ..nn import BaseDecoder, BaseEncoder, Encoder_MLP
from ..nn.default_architectures import Decoder_AE_MLP
from .base_config import BaseModelConfig, EnvironmentConfig
from ...customexception import BadInheritanceError
from ...models.base.base_utils import (
    CPU_Unpickler,
    ModelOutput,
    # check_decoder,
    # check_encoder,
)

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseAE(L.LightningModule):
    """Base class for Autoencoder based models."""

    def __init__(
        self,
        model_config: BaseModelConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        super().__init__()

        self.model_name = "BaseAE"
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.model_config = model_config

        if decoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "Input dimension ('input_dim') must be set in BaseModelConfig to build the decoder automatically."
                )
            decoder = Decoder_AE_MLP(model_config)
            self.model_config.uses_default_decoder = True
        else:
            self.model_config.uses_default_decoder = False
        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "Input dimension ('input_dim') must be set in BaseModelConfig to build the encoder automatically."
                )
            encoder = Encoder_MLP(model_config)
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False

        self.encoder = check_encoder(encoder)
        self.decoder = check_decoder(decoder)
        # self.device = None

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """Main forward pass. Must be implemented in child class."""
        raise NotImplementedError()

    def reconstruct(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstructions the given input data."""
        return self(DatasetOutput(data=inputs)).recon_x

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the embeddings of the input data.

        Args:
            inputs (torch.Tensor): The input data to be embedded, of shape [B x input_dim].

        Returns:
            torch.Tensor: A tensor of shape [B x latent_dim] containing the embeddings.
        """
        return self(DatasetOutput(data=inputs)).z

    def predict(self, inputs: torch.Tensor) -> ModelOutput:
        """Encodes and decodes input data without computing loss"""
        z = self.encoder(inputs).embedding
        recon_x = self.decoder(z)["reconstruction"]
        return ModelOutput(recon_x=recon_x, embedding=z)

    def interpolate(
        self,
        starting_inputs: torch.Tensor,
        ending_inputs: torch.Tensor,
        granularity: int = 10,
    ):
        """Performs linear interpolation in the latent space."""
        assert starting_inputs.shape[0] == ending_inputs.shape[0], (
            "The number of starting and ending inputs must be the same."
        )

        starting_z = self(DatasetOutput(data=starting_inputs)).z
        ending_z = self(DatasetOutput(data=ending_inputs)).z
        t = torch.linspace(0, 1, granularity).to(starting_inputs.device)
        intep_z = starting_z.unsqueeze(1) * (1 - t) + ending_z.unsqueeze(1) * t
        # intep_line = (
        #     torch.kron(
        #         starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)
        #     )
        #     + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
        # ).reshape((starting_z.shape[0] * t.shape[0],) + (starting_z.shape[1:]))

        decoded_line = self.decoder(intep_z.view(-1, self.latent_dim)).reconstruction.view(
            starting_inputs.shape[0], t.shape[0], *starting_inputs.shape[1:]
        )
        return decoded_line

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    def save(self, dir_path: str):
        """Saves the model and its configuration."""
        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
        model_dict = {"model_state_dict": self.state_dict()}
        os.makedirs(dir_path, exist_ok=True)

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
                cloudpickle.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                cloudpickle.dump(self.decoder, fp)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))

    @classmethod
    def _load_model_config_from_folder(cls, dir_path: str) -> BaseModelConfig:
        """Loads model config from a folder."""
        if "model_config.json" not in os.listdir(dir_path):
            raise FileNotFoundError(
                f"Missing 'model_config.json' in {dir_path}. Cannot load model."
            )
        config_path = os.path.join(dir_path, "model_config.json")
        model_config = AutoConfig.from_json_file(config_path)
        return model_config

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path: str) -> Dict[str, Any]:
        """Loads model weights from a folder."""
        if "model.pt" not in os.listdir(dir_path):
            raise FileNotFoundError(
                f"Missing 'model.pt' in {dir_path}. Cannot load model weights."
            )
        weights_path = os.path.join(dir_path, "model.pt")
        try:
            model_weights = torch.load(weights_path, map_location="cpu")
        except RuntimeError:
            raise RuntimeError(
                "Failed to load model weights. Ensure they are in '.pt' format."
            )
        if "model_state_dict" not in model_weights:
            raise KeyError(
                "'model_state_dict' not found in model weights file. Got keys:"
                f"{model_weights.keys()}"
            )
        return model_weights["model_state_dict"]

    @classmethod
    def _load_custom_encoder_from_folder(cls, dir_path: str) -> BaseEncoder:
        """Loads custom encoder from a folder."""
        cls._check_python_version_from_folder(dir_path=dir_path)
        if "encoder.pkl" not in os.listdir(dir_path):
            raise FileNotFoundError(
                f"Missing 'encoder.pkl' in{dir_path}. Cannot load encoder."
            )
        with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
            return CPU_Unpickler(fp).load()

    @classmethod
    def _load_custom_decoder_from_folder(cls, dir_path: str) -> BaseDecoder:
        """Loads custom decoder from a folder."""
        cls._check_python_version_from_folder(dir_path=dir_path)

        file_list = os.listdir(dir_path)
        if "decoder.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing 'decoder.pkl' in {dir_path}. Cannot load decoder."
            )
        with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
            return CPU_Unpickler(fp).load()

    @classmethod
    def load_from_folder(cls, dir_path: str) -> "BaseAE":
        """Loads the model from a specified folder."""
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        encoder = None
        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)
        decoder = None
        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)
        return model

    @classmethod
    def _check_python_version_from_folder(cls, dir_path: str) -> None:
        """Checks python version compatibility when loading a model."""
        if "environment.json" in os.listdir(dir_path):
            env_spec = EnvironmentConfig.from_json_file(
                os.path.join(dir_path, "environment.json")
            )
            python_version = env_spec.python_version
            python_version_minor = python_version.split(".")[1]

            if python_version_minor == "7" and sys.version_info[1] > 7:
                raise LoadError(
                    "Model saved with python 3.7, but trying to reload it with python 3.8+. "
                    "Use python 3.7 to reload this model."
                )

            elif int(python_version_minor) >= 8 and sys.version_info[1] == 7:
                raise LoadError(
                    "Model saved with python 3.8+, but trying to reload it with python 3.7. "
                    "Use python 3.8+ to reload this model."
                )


def check_encoder(encoder: BaseEncoder) -> BaseEncoder:
    """Checks if the encoder is compatible with this model."""
    if not issubclass(type(encoder), BaseEncoder):
        raise BadInheritanceError(
            "Encoder must inherit from BaseEncoder (...models.base_architectures.BaseEncoder)."
        )
    return encoder


def check_decoder(decoder: BaseDecoder) -> BaseDecoder:
    """Checks if the decoder is compatible with this model."""
    if not issubclass(type(decoder), BaseDecoder):
        raise BadInheritanceError(
            "Decoder must inherit from BaseDecoder (...models.base_architectures.BaseDecoder)."
        )
    return decoder
