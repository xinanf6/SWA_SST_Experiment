"""Training configuration and command validation without launching a model run."""

from dataclasses import replace
import json

import pytest

from sst_preprocessing.errors import ConfigurationError
from sst_training.__main__ import build_parser
from sst_training.config import TrainingConfig, load_training_config, validate_training_config


def test_training_relative_paths_ignore_terminal_cwd(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "training.json"
    path.write_text(json.dumps({"prepared_root": "../prepared", "output_root": "../training"}))
    monkeypatch.chdir(directory)
    config = load_training_config(path)
    assert config.prepared_root == tmp_path / "prepared"
    assert config.output_root == tmp_path / "training"


@pytest.mark.parametrize("name,value", [("batch_size", 0), ("workers", -1), ("seed", -1),
        ("accumulation_steps", 1.5), ("amp", "false"), ("learning_rate", float("nan")),
        ("max_epochs", True), ("weight_decay", -1), ("device", "invalid")])
def test_invalid_training_config_is_rejected(tmp_path, name, value):
    config = TrainingConfig(tmp_path / "prepared", tmp_path / "training")
    with pytest.raises(ConfigurationError):
        validate_training_config(replace(config, **{name: value}))


def test_training_cannot_write_into_prepared_root(tmp_path):
    with pytest.raises(ConfigurationError):
        validate_training_config(TrainingConfig(tmp_path / "prepared", tmp_path / "prepared/models"))


def test_prediction_requires_explicit_checkpoint_and_output():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict"])
    parsed = build_parser().parse_args(["predict", "--checkpoint", "best.pt", "--output", "predictions.nc"])
    assert parsed.split == "validation"
    assert not parsed.allow_smoke_checkpoint
