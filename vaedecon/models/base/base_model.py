import inspect
import logging
import os
import sys
import importlib
from http.cookiejar import LoadError
from typing import Optional, Dict, Any, Type, Union, Sequence

import cloudpickle
import json
import torch
import pandas as pd
import lightning as L
from ...data.datasets import BaseDataset, DatasetOutput
from ...models.auto_model import AutoConfig
# from ...models.vae import VAEConfig
from ..nn import BaseDecoder, BaseEncoder, EncoderMLP
# from ..gnn import EncoderSGNN
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
        encoders: list[BaseEncoder] = None,  # one or two encoders
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
        if encoders is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "Input dimension ('input_dim') must be set in BaseModelConfig to build the encoder automatically."
                )
            encoders = [EncoderMLP(model_config)]
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False

        self.encoders = check_encoder(encoders)
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

    def _get_encoder_config(self, encoder: BaseEncoder) -> Dict[str, Any]:
        """Get encoder configuration for JSON serialization."""
        if hasattr(encoder, "get_config"):
            return encoder.get_config()
        else:
            # Create a basic configuration with class info
            return {
                "class_name": encoder.__class__.__name__,
                "module_name": encoder.__class__.__module__,
                "params": {
                    "input_dim": getattr(encoder, "input_dim", self.input_dim),
                    "latent_dim": getattr(encoder, "latent_dim", self.latent_dim),
                    # Add other common parameters that might be needed
                    "hidden_dims": getattr(encoder, "hidden_dims", []),
                }
            }

    def _get_decoder_config(self) -> Dict[str, Any]:
        """Get decoder configuration for JSON serialization."""
        if hasattr(self.decoder, "get_config"):
            return self.decoder.get_config()
        else:
            # Create a basic configuration with class info
            return {
                "class_name": self.decoder.__class__.__name__,
                "module_name": self.decoder.__class__.__module__,
                "params": {
                    "input_dim": getattr(self.decoder, "input_dim", self.latent_dim),
                    "output_dim": getattr(self.decoder, "output_dim", self.input_dim),
                    # Add other common parameters that might be needed
                    "hidden_dims": getattr(self.decoder, "hidden_dims", []),
                }
            }

    def save(self, model_dir: str, training_config: Optional[Any] = None):
        """Saves the model and its configuration.
        Args:
            model_dir (str): The directory path where the model will be saved.
            model_dir (str, optional): Optional directory path where training logs are stored. Defaults to None.
            training_config (Any, optional): The training configuration to be saved. Defaults to None.
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save environment information
        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
        env_spec.save_json(model_dir, "environment")

        # Save model configuration
        self.model_config.save_json(model_dir, "model_config")

        # # Save model architecture configuration (for compatibility with save_model)
        # model_config = self.get_config()
        # with open(os.path.join(model_dir, "model_architecture.json"), "w") as f:
        #     json.dump(model_config, f, indent=4)

        # # Save encoder and decoder configurations as JSON and pkl
        if hasattr(self, "encoders"):
            # json
            encoder_config = self._get_encoder_config(self.encoders[0])
            with open(os.path.join(model_dir, "encoder_config.json"), "w") as f:
                json.dump(encoder_config, f, indent=4)

            # Save encoder weights separately
            for idx, encoder in enumerate(self.encoders):
                encoder_weights = encoder.state_dict()
                if isinstance(encoder, BaseEncoder):
                    encoder_name = encoder.__class__.__name__.lower()
                else:
                    encoder_name = f"encoder_{idx}"
                torch.save(encoder_weights, os.path.join(model_dir, f"{encoder_name}_weights.pt"))

            # torch.save(self.encoder.state_dict(), os.path.join(model_dir, "encoder_weights.pt"))

            # # pkl
            # with open(os.path.join(model_dir, "encoder.pkl"), "wb") as fp:
            #     cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
            #     cloudpickle.dump(self.encoder, fp)

        if hasattr(self, "decoder"):
            # json
            decoder_config = self._get_decoder_config()
            with open(os.path.join(model_dir, "decoder_config.json"), "w") as f:
                json.dump(decoder_config, f, indent=4)

            # Save decoder weights separately
            torch.save(self.decoder.state_dict(), os.path.join(model_dir, "decoder_weights.pt"))

            # pkl
            with open(os.path.join(model_dir, "decoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                cloudpickle.dump(self.decoder, fp)
        # # Save model weights separately
        torch.save(self.state_dict(), os.path.join(model_dir, "model_weights.pt"))
        # model_dict = {"model_state_dict": self.state_dict()}
        # torch.save(model_dict, os.path.join(model_dir, "model.pt"))

        # Save training configuration if provided
        if training_config is not None and hasattr(training_config, "save_json"):
            training_config.save_json(model_dir, "training_config")
        elif hasattr(self, "training_config") and hasattr(self.training_config, "save_json"):
            self.training_config.save_json(model_dir, "training_config")

        # Copy training logs
        try:
            # Try to find the metrics.csv file
            metrics_paths = [
                os.path.join(model_dir, "training_logs", "metrics.csv"),
                os.path.join(model_dir, "training_logs", "version_0", "metrics.csv")
            ]

            for path in metrics_paths:
                if os.path.exists(path):
                    losses_df = pd.read_csv(path)
                    losses_df.to_csv(os.path.join(model_dir, 'losses.csv'))
                    logger.info(f"Copied training logs from {path} to {model_dir}")
                    break
            else:
                logger.warning("Could not find training logs to copy")
        except Exception as e:
            logger.warning(f"Failed to copy training logs: {e}")

        logger.info(f"Model successfully saved to {model_dir}")

    @classmethod
    def _instantiate_model_from_config(cls, config: Dict[str, Any], **kwargs):
        """Instantiate a model from its configuration."""
        class_name = config.get("class_name")
        module_name = config.get("module_name")

        if not class_name or not module_name:
            raise ValueError("Configuration must include 'class_name' and 'module_name'")

        # Import the module and get the class
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            input_module = importlib.import_module("vaedecon.models.vae")
            input_class = getattr(input_module, "VAEConfig")

        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import {class_name} from {module_name}: {e}")

        # Get parameters for instantiation
        params = config.get("params", {})
        # Update with any additional kwargs
        params.update(kwargs)

        # Create an instance of the model
        try:
            input_instance = input_class(**params["args"])
            model = model_class(input_instance)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate {class_name}: {e}")

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
        if "model_weights.pt" not in os.listdir(dir_path):
            raise FileNotFoundError(
                f"Missing 'model_weights.pt' in {dir_path}. Cannot load model weights."
            )
        weights_path = os.path.join(dir_path, "model_weights.pt")
        try:
            model_weights = torch.load(weights_path, map_location="cpu")
        except RuntimeError:
            raise RuntimeError(
                "Failed to load model weights. Ensure they are in '.pt' format."
            )
        # if "model_state_dict" not in model_weights:
        #     raise KeyError(
        #         "'model_state_dict' not found in model weights file. Got keys:"
        #         f"{model_weights.keys()}"
        #     )
        return model_weights

    @classmethod
    def _load_custom_encoder_from_folder(cls, dir_path: str, encoder_weights_fn: str) -> BaseEncoder:
        """Loads custom encoder from a folder."""
        cls._check_python_version_from_folder(dir_path=dir_path)

        # Try loading from JSON config first (new method)
        if "encoder_config.json" in os.listdir(dir_path) and encoder_weights_fn in os.listdir(dir_path):
            logger.info("Loading encoder from JSON configuration and weights")
            try:
                # Load encoder configuration
                with open(os.path.join(dir_path, "encoder_config.json"), "r") as f:
                    encoder_config = json.load(f)

                # Instantiate encoder from config
                encoder = cls._instantiate_model_from_config(encoder_config)

                # Load encoder weights
                encoder_weights = torch.load(os.path.join(dir_path, encoder_weights_fn), map_location="cpu")
                encoder.load_state_dict(encoder_weights)

                return encoder
            except Exception as e:
                logger.warning(f"Failed to load encoder from JSON config: {e}. Falling back to pickle.")

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

        # Try loading from JSON config first (new method)
        if "decoder_config.json" in os.listdir(dir_path) and "decoder_weights.pt" in os.listdir(dir_path):
            logger.info("Loading decoder from JSON configuration and weights")
            try:
                # Load decoder configuration
                with open(os.path.join(dir_path, "decoder_config.json"), "r") as f:
                    decoder_config = json.load(f)

                # Instantiate decoder from config
                decoder = cls._instantiate_model_from_config(decoder_config)

                # Load decoder weights
                decoder_weights = torch.load(os.path.join(dir_path, "decoder_weights.pt"), map_location="cpu")
                decoder.load_state_dict(decoder_weights)

                return decoder
            except Exception as e:
                logger.warning(f"Failed to load decoder from JSON config: {e}. Falling back to pickle.")

                file_list = os.listdir(dir_path)
                if "decoder.pkl" not in file_list:
                    raise FileNotFoundError(
                        f"Missing 'decoder.pkl' in {dir_path}. Cannot load decoder."
                    )
                with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
                    return CPU_Unpickler(fp).load()
        return None

    @classmethod
    def load_from_folder(cls, dir_path: str) -> "BaseAE":
        """Loads the model from a specified folder."""
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        encoders = []
        if not model_config.uses_default_encoder:
            file_list = [i.lower() for i in os.listdir(dir_path)]
            for encoder_type in ['EncoderMLP', 'EncoderSGNN']:  # Add other encoder types as needed
                encoder_weights_fn = f"{encoder_type.lower()}_weights.pt"
                if encoder_weights_fn in file_list:
                    encoders.append(cls._load_custom_encoder_from_folder(dir_path, encoder_weights_fn))
            # encoders = cls._load_custom_encoder_from_folder(dir_path)
        decoder = None
        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        model = cls(model_config, encoders=encoders, decoder=decoder)
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

    def get_config(self):
        """Returns the model's config."""
        return self.model_config


def check_encoder(encoders: list[BaseEncoder]) -> list[BaseEncoder]:
    """Checks if the encoder is compatible with this model."""
    assert isinstance(encoders, list), "Encoder must be a list instance of BaseEncoder."
    if len(encoders) == 0:
        raise ValueError("Encoder must be a non-empty list of BaseEncoder instances.")
    if len(encoders) > 2:
        raise ValueError("Encoder must be a list of 1 or 2 BaseEncoder instances.")

    for idx, encoder in enumerate(encoders):
        if not issubclass(type(encoder), BaseEncoder):
            raise BadInheritanceError(
                f"The {idx} encoder ({type(encoder)}) is not inherited from BaseEncoder (...models.base_architectures.BaseEncoder)."
            )
    return encoders


def check_decoder(decoder: BaseDecoder) -> BaseDecoder:
    """Checks if the decoder is compatible with this model."""
    if not issubclass(type(decoder), BaseDecoder):
        raise BadInheritanceError(
            "Decoder must inherit from BaseDecoder (...models.base_architectures.BaseDecoder)."
        )
    return decoder
