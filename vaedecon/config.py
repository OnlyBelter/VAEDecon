import json
import os
import logging
from pathlib import Path
import warnings
from dataclasses import field, dataclass
from typing import Any, Dict, Union

from pydantic import ValidationError
from pydantic import BaseModel, ConfigDict, field_serializer

# Configure logging
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseConfig(BaseModel):
    """This is the BaseConfig class that defines all the useful loading and saving methods
    of the configs"""

    model_config = ConfigDict(validate_assignment=True)

    name: str = ""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
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
            logger.error(f"Failed to create {cls.__name__} from dict: {e}")
            raise e
        return config

    @classmethod
    def _dict_from_json(cls, json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        """Load dictionary from JSON file

        Args:
            json_path: Path to the JSON file

        Returns:
            Dictionary containing the configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            TypeError: If file is not valid JSON
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    config_dict = json.load(f)
                    return config_dict

                except (TypeError, json.JSONDecodeError) as e:
                    raise TypeError(
                        f"File {json_path} not loadable. Maybe not valid JSON?\n"
                        f"Catch Exception {type(e).__name__} with message: " + str(e)
                    ) from e

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )

    @classmethod
    def from_json_file(cls, json_path: Union[str, os.PathLike]) -> "BaseConfig":
        """Creates a: class:`~pythae.config.BaseConfig` instance from a JSON config file

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            class:`BaseConfig`: The created instance
        """
        config_dict = cls._dict_from_json(json_path)

        config_name = config_dict.get("name")

        if config_name and cls.__name__ != config_name:
            warnings.warn(
                f"You are trying to load a "
                f"`{cls.__name__}` while a "
                f"`{config_name}` is given.",
                UserWarning
            )

        return cls.from_dict(config_dict)

    @field_serializer('*', when_used='json')
    def serialize_paths(self, value: Any) -> Any:
        """Convert Path objects to strings when serializing to JSON."""
        if isinstance(value, Path):
            return str(value)
        return value

    def to_dict(self) -> dict:
        """Transforms an object into a Python dictionary

        Returns:
            (dict): The dictionary containing all the parameters"""
        return self.model_dump(mode='json')

    def to_json_string(self, indent: int = None) -> str:
        """Transforms an object into a JSON string

        Returns:
            (str): The JSON str containing all the parameters"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, dir_path: Union[str, os.PathLike], filename: str) -> None:
        """Saves a ``.json`` file from the dataclass

        Args:
            dir_path (str): path to the folder
            filename (str): the name of the file

        """
        dir_p = Path(dir_path)
        dir_p.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        file_path = dir_p / filename

        compact_json_string = self.to_json_string(indent=4)

        try:
            # Parse the compact JSON string back into a Python object
            python_obj = json.loads(compact_json_string)

            # Now dump this Python object to the file with indentation
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(python_obj, fp, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved configuration to {file_path}")
            # print(f"Successfully saved indented JSON to {file_path}") # Optional logging

        except (OSError, IOError) as e:
            # This can happen if self.to_json_string() doesn't return valid JSON
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
            # logger.error(f"Error: self.to_json_string() did not return a valid JSON string. {e}")
            # logger.warning(
            #     f"Saving the original string from to_json_string() without indentation to {file_path} as a fallback.")
            # with open(file_path, "w", encoding="utf-8") as fp:
            #     fp.write(compact_json_string)  # Save the original string if parsing fails
        except Exception as e:
            logger.error(f"Unexpected error while saving JSON to {file_path}: {e}")
            # logger.error("Error: self.to_json_string() method not found or failed internally.")
            raise  # Re-raise if the method itself is missing/problematic

        # with open(file_path, "w", encoding="utf-8") as fp:
        #     fp.write(self.to_json_string())
