"""Offline C0 train/resume/predict/evaluate integration on synthetic source files."""

import contextlib
from dataclasses import replace
import io
import json

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from test_pipeline import create_pipeline_fixture
from sst_analysis.context import load_context
from sst_analysis.evaluation import evaluate_forecast_file
from sst_analysis.forecasts import ForecastFile
from sst_preprocessing.errors import DataValidationError
from sst_preprocessing.pipeline import run_preprocessing
from sst_training.config import TrainingConfig
from sst_training.data import TorchSSTDataset, build_loader
from sst_training.engine import run_training
from sst_training.predict import export_predictions


ARCHITECTURE = {"spatial_channels": 8, "global_hidden": 8, "pixel_hidden": 4,
                "fusion_channels": 8, "decoder_channels": 16, "decoder_hidden": 8}


@pytest.fixture
def fixture(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    prepared = create_pipeline_fixture(tmp_path / "raw")
    with contextlib.redirect_stdout(io.StringIO()):
        run_preprocessing(prepared)
    config = TrainingConfig(prepared.output_root, tmp_path / "training", architecture=ARCHITECTURE,
                            device="cpu", amp=False, batch_size=1, accumulation_steps=1, max_epochs=4)
    yield config
    torch.set_num_threads(previous)


def test_adapter_preserves_lag_dates_and_windows(fixture):
    dataset = TorchSSTDataset(fixture.prepared_root, "train")
    try:
        sample = dataset[0]
        original = dataset.prepared[0]
        for name in ("base", "thermal", "target", "last_sst"):
            np.testing.assert_array_equal(sample[name].numpy(), original[name])
        batch = next(iter(build_loader(dataset, fixture, {})))
        assert batch["base"].shape == (1, 14, 7, 7, 7)
        assert batch["origin_date"] == [original["origin_date"]]
        assert [day[0] for day in batch["target_dates"]] == sample["target_dates"]
    finally:
        dataset.close()
    assert dataset.prepared.arrays == {}


def test_epoch_boundary_resume_matches_uninterrupted_training(fixture):
    uninterrupted = replace(fixture, output_root=fixture.output_root.parent / "uninterrupted")
    run_training(uninterrupted)
    run_training(fixture, {"epochs": 2})
    result = run_training(fixture, {"resume": True})
    assert result["epochs_completed"] == 4
    first = torch.load(uninterrupted.output_root / "last.pt", map_location="cpu")
    second = torch.load(fixture.output_root / "last.pt", map_location="cpu")
    for name, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][name], rtol=0, atol=0)
    assert [row["validation"]["sst_mse_C2"] for row in first["history"]] == [
            row["validation"]["sst_mse_C2"] for row in second["history"]]


def test_trained_checkpoint_exports_full_forecast_and_scores_canonical_truth(fixture):
    run_training(fixture, {"epochs": 2})
    path = fixture.output_root / "predictions" / "forecast.nc"
    result = export_predictions(fixture, {"checkpoint": fixture.output_root / "best.pt",
                                          "output": path, "split": "validation"})
    assert result["origins"] == 1
    context = load_context(fixture.prepared_root)
    with ForecastFile(path, context, "validation") as forecasts:
        assert forecasts.read(slice(0, 1)).shape == (1, 3, 7, 7)
    report = evaluate_forecast_file(context, {"predictions": path, "split": "validation",
                   "output_root": path.parent / "metrics"})
    assert report["sample_count"] == 1
    assert all(np.isfinite(row["rmse"]) for row in report["metrics"])
    with pytest.raises(DataValidationError, match="new .nc"):
        export_predictions(fixture, {"checkpoint": fixture.output_root / "best.pt", "output": path})


def test_resume_rejects_changed_settings_and_data_identity(fixture):
    run_training(fixture, {"epochs": 1})
    changed = replace(fixture, learning_rate=.002)
    with pytest.raises(DataValidationError, match="config_fingerprint"):
        run_training(changed, {"resume": True})
    run_path = fixture.output_root / "run.json"
    manifest = json.loads(run_path.read_text())
    manifest["run_identity"] = "another-data-run"
    run_path.write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="run_identity"):
        run_training(fixture, {"resume": True})


def test_small_subset_training_reduces_training_loss(fixture):
    config = replace(fixture, max_epochs=12)
    result = run_training(config, {"kind": "smoke", "train_limit": 1, "validation_limit": 1})
    assert result["status"] == "smoke_complete"
    assert result["history"][-1]["training"]["sst_mse_C2"] < result["history"][0]["training"]["sst_mse_C2"]


def test_best_checkpoint_can_be_recovered_from_authoritative_last(fixture):
    run_training(fixture)
    original = torch.load(fixture.output_root / "best.pt", map_location="cpu")
    (fixture.output_root / "best.pt").unlink()
    run_training(fixture, {"resume": True})
    restored = torch.load(fixture.output_root / "best.pt", map_location="cpu")
    assert restored["epoch"] == original["epoch"]
    for name, value in original["model"].items():
        torch.testing.assert_close(value, restored["model"][name], rtol=0, atol=0)


def test_smoke_checkpoint_export_requires_explicit_debug_flag(fixture):
    run_training(fixture, {"kind": "smoke", "epochs": 1, "train_limit": 1, "validation_limit": 1})
    options = {"checkpoint": fixture.output_root / "best.pt",
               "output": fixture.output_root / "debug/predictions.nc"}
    with pytest.raises(DataValidationError, match="allow-smoke-checkpoint"):
        export_predictions(fixture, options)
    result = export_predictions(fixture, dict(options, allow_smoke_checkpoint=True))
    assert result["training_kind"] == "smoke"


def test_spawn_worker_reopens_memmaps(fixture):
    dataset = TorchSSTDataset(fixture.prepared_root, "train")
    try:
        config = replace(fixture, workers=1)
        loader = build_loader(dataset, config, {})
        batch = next(iter(loader))
        torch.testing.assert_close(batch["base"][0], dataset[0]["base"])
    finally:
        dataset.close()


def test_failed_prediction_keeps_existing_file_and_removes_temporary(fixture, monkeypatch):
    from sst_training.models.predictor import C0Predictor
    run_training(fixture, {"epochs": 1})
    original = C0Predictor.forward

    def nonfinite(self, batch):
        output = original(self, batch)
        output["predictions"][..., 2, 2] = float("nan")
        return output

    monkeypatch.setattr(C0Predictor, "forward", nonfinite)
    path = fixture.output_root / "existing.nc"
    path.write_bytes(b"existing-artifact")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        export_predictions(fixture, {"checkpoint": fixture.output_root / "best.pt",
                                      "output": path, "overwrite": True})
    assert path.read_bytes() == b"existing-artifact"
    assert not path.with_name(path.name + ".tmp").exists()
