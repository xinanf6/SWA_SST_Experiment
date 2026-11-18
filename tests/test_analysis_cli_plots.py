"""Offline CLI, report and figure contracts, using tiny synthetic diagnostic maps."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import xarray as xr

import sst_analysis.__main__ as analysis_cli
from sst_analysis.config import load_analysis_config
from sst_analysis.context import AnalysisContext
from sst_analysis.diagnostics import METRIC_UNITS, SEASONS
from sst_analysis.plots import FIGURE_NAMES, render_diagnostic_figures
from sst_analysis.reports import validate_diagnostic_artifacts
from sst_preprocessing.artifacts import write_json_atomic
from sst_preprocessing.errors import ConfigurationError, DataValidationError


RUN_IDENTITY = "synthetic-cli-figures"
LOW_FIGURE_DPI = 72


def _write_config(path: Path, options: dict = None) -> Path:
    settings = {"prepared_root": "../prepared", "diagnostics_output": "../diagnostics",
                "figure_dpi": LOW_FIGURE_DPI, "batch_size": 1}
    settings.update(options or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, settings)
    return path


def _synthetic_context(root: Path) -> AnalysisContext:
    latitude, longitude = np.linspace(-44., -40., 5), np.linspace(-57., -52., 6)
    core = np.ones((5, 6), dtype=bool)
    ocean = core.copy()
    ocean[-1, -1] = False
    grid = {"latitude": latitude, "longitude": longitude, "core_mask": core,
            "ocean_mask": ocean, "gradient_mask": ocean.copy(),
            "area_weights": np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], core.shape).copy()}
    run = {"run_identity": RUN_IDENTITY,
           "config": {"splits": {"train": ["2011-01-01", "2011-12-31"]}}}
    root.mkdir()
    return AnalysisContext(root, run, {"front_threshold_p90": 0.08}, grid)


def _synthetic_maps(context: AnalysisContext) -> xr.Dataset:
    offsets = np.arange(5 * 5 * 6, dtype=np.float32).reshape(5, 5, 6) / 1000.
    variables = {}
    for offset, name in enumerate(METRIC_UNITS):
        values = np.float32(offset + 0.2) + offsets
        if name in ("front_frequency", "mean_gradient"):
            values = values / 20.
        values[:, ~context.grid["ocean_mask"]] = np.nan
        variables[name] = (("season", "latitude", "longitude"), values)
    for name in ("mean_uo", "mean_vo"):
        variables[name] = (("season", "latitude", "longitude"), np.full_like(offsets, 0.1))
    return xr.Dataset(variables, coords={"season": list(SEASONS),
                      "latitude": context.grid["latitude"], "longitude": context.grid["longitude"]},
                      attrs={"run_identity": context.run_identity})


def _write_synthetic_tables(root: Path) -> None:
    dates = pd.date_range("2011-01-01", "2011-12-31")
    daily = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    fields = [name for name in METRIC_UNITS if name not in ("std_sst", "front_frequency")]
    for offset, name in enumerate(fields):
        daily[name] = offset + 0.01 * np.arange(len(daily))
    daily["strong_front_area_fraction"] = 0.2
    daily.to_csv(root / "daily_regional.csv", index=False)
    monthly = daily.drop(columns="date").groupby(dates.month).mean()
    monthly.index.name = "month"
    monthly.to_csv(root / "monthly_cycle.csv")


@pytest.fixture
def diagnostic_artifacts(tmp_path):
    context = _synthetic_context(tmp_path / "prepared")
    root = tmp_path / "diagnostics"
    root.mkdir()
    _synthetic_maps(context).to_netcdf(root / "seasonal_maps.nc", engine="netcdf4")
    _write_synthetic_tables(root)
    artifacts = {"seasonal_maps": "seasonal_maps.nc", "daily_regional": "daily_regional.csv",
                 "monthly_cycle": "monthly_cycle.csv", "summary": "diagnostics_summary.json"}
    summary = {"status": "complete", "run_identity": RUN_IDENTITY,
               "train_period": list(context.train_period), "training_day_count": 365,
               "source_files": ["physical/2011-%02d.nc" % month for month in range(1, 13)],
               "artifacts": artifacts}
    write_json_atomic(root / "diagnostics_summary.json", summary)
    config = _write_config(tmp_path / "configs" / "analysis.json")
    return {"context": context, "root": root, "config": config, "summary": summary}


def test_config_paths_ignore_cwd_and_allow_explicit_prepared_override(tmp_path, monkeypatch):
    path = _write_config(tmp_path / "configs" / "analysis.json")
    terminal = tmp_path / "elsewhere"
    terminal.mkdir()
    monkeypatch.chdir(terminal)
    settings = load_analysis_config(path)
    assert settings.prepared_root == (tmp_path / "prepared").resolve()
    assert settings.diagnostics_output == (tmp_path / "diagnostics").resolve()
    overridden = load_analysis_config(path, tmp_path / "alternate_prepared")
    assert overridden.prepared_root == (tmp_path / "alternate_prepared").resolve()


def test_prepared_override_cannot_bypass_output_safety(tmp_path):
    path = _write_config(tmp_path / "configs" / "analysis.json",
                         {"diagnostics_output": "../overridden_prepared/diagnostics"})
    with pytest.raises(ConfigurationError, match="outside"):
        load_analysis_config(path, tmp_path / "overridden_prepared")


@pytest.mark.parametrize("arguments", [[], ["evaluate", "--output-root", "unused"],
                                      ["evaluate", "--output-root", "unused", "--predictions", "unused",
                                       "--split", "unknown"]])
def test_cli_rejects_missing_required_arguments_and_unknown_split(arguments, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sst_analysis"] + arguments)
    with pytest.raises(SystemExit) as caught:
        analysis_cli.main()
    assert caught.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_reports_missing_completed_preprocessing(tmp_path, monkeypatch, capsys):
    config = _write_config(tmp_path / "configs" / "analysis.json")
    monkeypatch.setattr(sys, "argv", ["sst_analysis", "diagnose", "--config", str(config), "--no-plots"])
    assert analysis_cli.main() == 2
    assert "Completed preprocessing artifact is missing" in capsys.readouterr().err


@pytest.mark.parametrize("relationship", ("same", "descendant", "ancestor"))
def test_cli_rejects_prepared_output_overlap(diagnostic_artifacts, monkeypatch, capsys, relationship):
    case = diagnostic_artifacts
    prepared = case["context"].root
    root = prepared if relationship == "same" else prepared / "diagnostics" if relationship == "descendant" else prepared.parent
    monkeypatch.setattr(analysis_cli, "load_context", lambda _: case["context"])
    monkeypatch.setattr(sys, "argv", ["sst_analysis", "diagnose", "--config", str(case["config"]),
                                     "--output-root", str(root), "--no-plots"])
    assert analysis_cli.main() == 2
    assert "Analysis failed:" in capsys.readouterr().err
    assert not (prepared / "diagnostics_summary.json").exists()


def test_cli_rejects_existing_nonempty_diagnostics_without_overwrite(diagnostic_artifacts, monkeypatch, capsys):
    case = diagnostic_artifacts
    monkeypatch.setattr(analysis_cli, "load_context", lambda _: case["context"])
    monkeypatch.setattr(sys, "argv", ["sst_analysis", "diagnose", "--config", str(case["config"]), "--no-plots"])
    assert analysis_cli.main() == 2
    assert "not empty" in capsys.readouterr().err


def test_plot_cli_writes_five_pairs_report_and_matching_manifest(diagnostic_artifacts, monkeypatch, capsys):
    case = diagnostic_artifacts
    monkeypatch.setattr(analysis_cli, "load_context", lambda _: case["context"])
    monkeypatch.setattr(sys, "argv", ["sst_analysis", "plot-diagnostics", "--config", str(case["config"])])
    assert analysis_cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    expected = ["figures/%s.%s" % (name, suffix) for name in FIGURE_NAMES for suffix in ("png", "pdf")]
    assert result["figure_files"] == expected
    assert result["training_days"] == 365 and result["training_months"] == 12
    assert result["output_root"] == str(case["root"].resolve())
    _assert_figure_files(case["root"], expected)
    _assert_report_manifest(case, expected)
    assert not plt.get_fignums()
    assert not list(case["root"].rglob("*.tmp"))


def _assert_figure_files(root: Path, relative_paths: list) -> None:
    assert sorted(path.name for path in (root / "figures").iterdir()) == sorted(Path(path).name for path in relative_paths)
    for relative in relative_paths:
        path = root / relative
        assert path.stat().st_size > 1000
        if path.suffix == ".png":
            with Image.open(path) as image:
                assert image.format == "PNG"
                assert min(image.size) >= 500
                image.verify()
        else:
            assert path.read_bytes().startswith(b"%PDF-")


def _assert_report_manifest(case: dict, figures: list) -> None:
    root, context = case["root"], case["context"]
    report = (root / "diagnostics_report.md").read_text(encoding="utf-8")
    manifest = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
    summary = validate_diagnostic_artifacts(context, root)
    assert "2011-01-01" in report and "2011-12-31" in report
    assert "365" in report and "DJF" in report
    for relative in figures:
        assert "](%s)" % relative in report
    assert manifest["status"] == summary["status"] == "complete"
    assert manifest["run_identity"] == summary["run_identity"] == context.run_identity
    assert manifest["kind"] == "training_regional_diagnostics"
    assert len(manifest["analysis_code_sha256"]) == 64
    assert manifest["settings"] == {"figure_dpi": LOW_FIGURE_DPI, "train_period": list(context.train_period)}
    assert manifest["artifacts"] == dict(summary["artifacts"], figures=figures, report="diagnostics_report.md")
    assert all((root / relative).is_file() for relative in summary["artifacts"].values())
    assert {"python", "numpy", "pandas", "xarray", "scipy", "matplotlib"} <= set(manifest["software"])


@pytest.mark.parametrize("case_name", ("identity", "status", "training_bounds"))
def test_existing_diagnostics_reject_mismatched_summary(diagnostic_artifacts, case_name):
    case = diagnostic_artifacts
    summary = dict(case["summary"])
    key = {"identity": "run_identity", "status": "status", "training_bounds": "train_period"}[case_name]
    summary[key] = ["2012-01-01", "2012-12-31"] if case_name == "training_bounds" else "incorrect"
    write_json_atomic(case["root"] / "diagnostics_summary.json", summary)
    with pytest.raises(DataValidationError):
        validate_diagnostic_artifacts(case["context"], case["root"])


def test_plotting_rejects_maps_from_another_run(diagnostic_artifacts):
    case = diagnostic_artifacts
    path = case["root"] / "seasonal_maps.nc"
    with xr.open_dataset(path) as source:
        maps = source.load()
    maps.attrs["run_identity"] = "another-run"
    maps.to_netcdf(path, engine="netcdf4")
    with pytest.raises(DataValidationError, match="different prepared run"):
        render_diagnostic_figures(case["context"], case["root"], {"dpi": LOW_FIGURE_DPI})
    assert not (case["root"] / "figures").exists()
