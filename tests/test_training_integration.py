"""Offline C0 train/resume/predict/evaluate integration on synthetic source files."""

import contextlib
from dataclasses import replace
import io
import json

import numpy as np
import pytest
import xarray as xr
torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")

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


def test_c0_skips_unused_fields_and_reads_each_base_channel_at_its_own_clock(fixture):
    dataset = TorchSSTDataset(fixture.prepared_root, "train")
    try:
        prepared = dataset.prepared
        original = prepared[0]
        row = prepared.sample_index.iloc[0]
        raw = prepared.arrays["base"]
        previous_base = np.array(raw[row.sst_start_index:row.sst_end_index + 1])
        previous_base[:, 1:] = raw[row.environment_start_index:row.environment_end_index + 1, 1:]
        previous_base = np.where(prepared.ocean_mask, previous_base, 0.0).astype(np.float32)
        front = prepared.arrays.pop("front")
        try:
            sample = dataset[0]
        finally:
            prepared.arrays["front"] = front
        assert "front" not in sample and "heat_increment" not in sample
        np.testing.assert_array_equal(sample["base"].numpy(), previous_base)
        for name in ("thermal", "target", "last_sst"):
            np.testing.assert_array_equal(sample[name].numpy(), original[name])
        sample["base"].zero_()
        np.testing.assert_array_equal(dataset[0]["base"].numpy(), previous_base)
    finally:
        dataset.close()


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


def test_parallel8_train_resume_and_export_keep_model_identity(fixture):
    compact = replace(fixture, model_variant="parallel8", architecture={})
    assert compact.fingerprint() != fixture.fingerprint()
    result = run_training(compact, {"epochs": 1, "progress": False})
    assert result["model"] == "C0_parallel8"
    assert result["parameter_count"] == 28429
    assert run_training(compact, {"epochs": 1, "resume": True, "progress": False})["epochs_completed"] == 2
    forecast = compact.output_root / "compact.nc"
    options = {"checkpoint": compact.output_root / "best.pt", "output": forecast, "split": "validation"}
    with pytest.raises(DataValidationError, match="configuration"):
        export_predictions(fixture, options)
    assert export_predictions(compact, options)["origins"] == 1
    context = load_context(compact.prepared_root)
    with ForecastFile(forecast, context, "validation") as predictions:
        assert predictions.read(slice(0, 1)).shape == (1, 3, 7, 7)
    with xr.open_dataset(forecast) as dataset:
        assert dataset.attrs["model_name"] == "C0_parallel8"
        np.testing.assert_allclose(dataset.predictions.values,
                                   dataset.last_sst.values[:, None] + dataset.residual.values)


def test_dspg7_train_and_forecast_use_seven_channels_and_shared_evaluator(fixture):
    baseline = replace(fixture, model_variant="dspg7")
    dataset = TorchSSTDataset(baseline.prepared_root, "train", has_thermal_input=False)
    try:
        dropped = dataset.prepared.arrays.pop("thermal")
        try:
            sample = dataset[0]
        finally:
            dataset.prepared.arrays["thermal"] = dropped
        assert sample["base"].shape == (14, 7, 7, 7)
        assert "thermal" not in sample
    finally:
        dataset.close()
    assert run_training(baseline, {"epochs": 1, "progress": False})["model"] == "DSPG7"
    forecast = baseline.output_root / "dspg7.nc"
    options = {"checkpoint": baseline.output_root / "best.pt", "output": forecast}
    with pytest.raises(DataValidationError, match="configuration"):
        export_predictions(fixture, options)
    assert export_predictions(baseline, options)["origins"] == 1
    context = load_context(baseline.prepared_root)
    report = evaluate_forecast_file(context, {"predictions": forecast, "split": "validation",
                   "output_root": forecast.parent / "metrics", "model_name": "DSPG7"})
    assert report["sample_count"] == 1 and len(report["metrics"]) == 3
    with xr.open_dataset(forecast) as stored:
        assert stored.attrs["model_name"] == "DSPG7"
        assert stored.attrs["sst_head"] == "direct"


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


def test_progress_does_not_change_training_weights_or_rng(fixture):
    first = replace(fixture, output_root=fixture.output_root.parent / "visible")
    second = replace(fixture, output_root=fixture.output_root.parent / "hidden")
    run_training(first, {"progress": True})
    run_training(second, {"progress": False})
    visible = torch.load(first.output_root / "last.pt", map_location="cpu")
    hidden = torch.load(second.output_root / "last.pt", map_location="cpu")
    for name, value in visible["model"].items():
        torch.testing.assert_close(value, hidden["model"][name], rtol=0, atol=0)
    torch.testing.assert_close(visible["rng"]["torch"], hidden["rng"]["torch"], rtol=0, atol=0)


def test_progress_update_can_resume_old_checkpoints_but_rejects_unknown_code(fixture):
    legacy = "eb8e3f3d5c9a847be8bb61e8f5095916a8258f23c0fd08ac925dffa07278a31a"
    run_training(fixture, {"epochs": 2, "progress": False})
    run_path = fixture.output_root / "run.json"
    manifest = json.loads(run_path.read_text())
    manifest["training_code_sha256"] = legacy
    run_path.write_text(json.dumps(manifest))
    checkpoint = torch.load(fixture.output_root / "last.pt", map_location="cpu")
    checkpoint["metadata"]["training_code_sha256"] = legacy
    torch.save(checkpoint, fixture.output_root / "last.pt")
    result = run_training(fixture, {"resume": True})
    assert result["epochs_completed"] == 4
    run_training(fixture, {"resume": True})
    checkpoint = torch.load(fixture.output_root / "last.pt", map_location="cpu")
    checkpoint["metadata"]["training_code_sha256"] = "unknown-training-code"
    torch.save(checkpoint, fixture.output_root / "last.pt")
    with pytest.raises(DataValidationError, match="training_code_sha256"):
        run_training(fixture, {"resume": True})
