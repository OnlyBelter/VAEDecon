import json
import os
import logging
from pathlib import Path
import warnings
from dataclasses import asdict, field
from typing import Any, Dict, Union

from pydantic import ValidationError
from pydantic import BaseModel, ConfigDict

# Configure logging
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

class BaseConfig(BaseModel):
    """This is the BaseConfig class that defines all the useful loading and saving methods
    of the configs"""

    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Creates a: class:`~pythae.config.BaseConfig` instance from a dictionary

        Args:
            config_dict (dict): The Python dictionary containing all the parameters

        Returns:
            class:`BaseConfig`: The created instance
        """
        try:
            config = cls(**config_dict)
        except (ValidationError, TypeError) as e:
            raise e
        return config

    @classmethod
    def _dict_from_json(cls, json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        try:
            with open(json_path) as f:
                try:
                    config_dict = json.load(f)
                    return config_dict

                except (TypeError, json.JSONDecodeError) as e:
                    raise TypeError(
                        f"File {json_path} not loadable. Maybe not json ? \n"
                        f"Catch Exception {type(e)} with message: " + str(e)
                    ) from e

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )

    @classmethod
    def from_json_file(cls, json_path: str) -> "BaseConfig":
        """Creates a: class:`~pythae.config.BaseConfig` instance from a JSON config file

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            class:`BaseConfig`: The created instance
        """
        config_dict = cls._dict_from_json(json_path)

        config_name = config_dict["name"]

        if cls.__name__ != config_name:
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{config_name}` is given."
            )

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Transforms an object into a Python dictionary

        Returns:
            (dict): The dictionary containing all the parameters"""
        return self.model_dump()

    def to_json_string(self):
        """Transforms an object into a JSON string

        Returns:
            (str): The JSON str containing all the parameters"""
        return json.dumps(self.to_dict())

    def save_json(self, dir_path, filename):
        """Saves a ``.json`` file from the dataclass

        Args:
            dir_path (str): path to the folder
            filename (str): the name of the file

        """
        dir_p = Path(dir_path)
        dir_p.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        file_path = os.path.join(dir_p, f"{filename}.json")

        compact_json_string = self.to_json_string()

        try:
            # Parse the compact JSON string back into a Python object
            python_obj = json.loads(compact_json_string)

            # Now dump this Python object to the file with indentation
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(python_obj, fp, indent=4, ensure_ascii=False)
            # print(f"Successfully saved indented JSON to {file_path}") # Optional logging

        except json.JSONDecodeError as e:
            # This can happen if self.to_json_string() doesn't return valid JSON
            logger.error(f"Error: self.to_json_string() did not return a valid JSON string. {e}")
            logger.warning(
                f"Saving the original string from to_json_string() without indentation to {file_path} as a fallback.")
            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write(compact_json_string)  # Save the original string if parsing fails
        except AttributeError:
            logger.error("Error: self.to_json_string() method not found or failed internally.")
            raise  # Re-raise if the method itself is missing/problematic

        # with open(file_path, "w", encoding="utf-8") as fp:
        #     fp.write(self.to_json_string())
