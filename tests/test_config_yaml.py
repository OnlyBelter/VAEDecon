import importlib.resources as resources
from pathlib import Path

import pytest
import yaml

from vaedecon.configs import VAEDeconConfig


def test_example_config_resource_loads(tmp_path: Path):
    text = resources.files("vaedecon.configs").joinpath("example_config.yaml").read_text()
    cfg = yaml.safe_load(text)
    assert isinstance(cfg, dict)

    p = tmp_path / "example_config.yaml"
    p.write_text(text)
    loaded = VAEDeconConfig.from_yaml(p)
    assert loaded is not None


def test_duplicate_yaml_keys_raise(tmp_path: Path):
    p = tmp_path / "dup.yaml"
    p.write_text("evaluation:\n  val_batch_size: 128\n  val_batch_size: 64\n")
    with pytest.raises(ValueError):
        _ = VAEDeconConfig.from_yaml(p)
