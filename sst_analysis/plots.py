"""Standalone publication-style figures; no basemap downloads or field smoothing."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import xarray as xr

from sst_preprocessing.errors import DataValidationError

from .context import AnalysisContext


FIGURE_NAMES = ("regional_overview", "front_structure", "thermal_environment",
                "seasonal_structure", "monthly_cycles")
FIELD_LABELS = {
    "mean_sst": ("平均海表温度", "Mean SST", "°C"),
    "std_sst": ("海表温度时间标准差", "Temporal SST standard deviation", "°C"),
    "mean_current_speed": ("平均海流速度", "Mean current speed", "m/s"),
    "mean_gradient": ("平均海温梯度", "Mean SST gradient", "°C/km"),
    "front_frequency": ("强锋面出现频率", "Strong-front frequency", "%"),
    "strong_front_area_fraction": ("强锋面面积比例", "Strong-front area fraction", "%"),
    "mean_net_heat_flux": ("平均净热通量", "Mean net surface heat flux", "W/m²"),
    "mean_mixed_layer_depth": ("平均混合层厚度", "Mean mixed-layer thickness", "m"),
    "mean_heat_increment": ("平均热力日倾向", "Mean thermal increment", "°C/day"),
    "mean_abs_heat_increment": ("平均热力倾向绝对值", "Mean absolute thermal increment", "°C/day"),
    "mean_abs_sst_change": ("平均海温日变化绝对值", "Mean absolute daily SST change", "°C/day"),
}
SEASON_LABELS = (("DJF", "夏季"), ("MAM", "秋季"), ("JJA", "冬季"), ("SON", "春季"))
SIGNED_FIELDS = {"mean_net_heat_flux", "mean_heat_increment"}
QUIVER_STRIDE = 14


def _font_settings() -> dict:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
                 if name in available), "DejaVu Sans")
    return {"font": font, "chinese": font != "DejaVu Sans"}


def _text(state: dict, chinese: str, english: str) -> str:
    return chinese if state["chinese"] else english


def _field_label(state: dict, name: str) -> str:
    chinese, english, units = FIELD_LABELS[name]
    return "%s (%s)" % (_text(state, chinese, english), units)


def _display_values(values: np.ndarray, name: str) -> np.ndarray:
    return values * 100.0 if name in {"front_frequency", "strong_front_area_fraction"} else values


def _limits(values: np.ndarray, name: str) -> tuple:
    finite = np.asarray(values)[np.isfinite(values)]
    if name == "front_frequency":
        return 0.0, 100.0
    if not finite.size:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if name in SIGNED_FIELDS:
        high = max(abs(low), abs(high), 1e-6)
        return -high, high
    if low == high:
        padding = max(abs(low) * 0.05, 1e-6)
        return low - padding, high + padding
    return low, high


def _configure_axis(axis, state: dict) -> None:
    grid = state["context"].grid
    core = grid["core_mask"]
    latitude = grid["latitude"][core.any(axis=1)]
    longitude = grid["longitude"][core.any(axis=0)]
    dy, dx = np.median(np.diff(grid["latitude"])), np.median(np.diff(grid["longitude"]))
    axis.set_xlim(longitude.min() - dx / 2, longitude.max() + dx / 2)
    axis.set_ylim(latitude.min() - dy / 2, latitude.max() + dy / 2)
    axis.set_aspect(1 / np.cos(np.deg2rad(latitude.mean())))
    axis.set_facecolor("#e5e7eb")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "%.0f°%s" % (abs(value), "W" if value < 0 else "E")))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "%.0f°%s" % (abs(value), "S" if value < 0 else "N")))
    axis.set_xlabel(_text(state, "经度", "Longitude"))
    axis.set_ylabel(_text(state, "纬度", "Latitude"))
    axis.tick_params(labelsize=9)


def _draw_map(axis, data: xr.DataArray, state: dict):
    name = data.name
    values = _display_values(data.values, name)
    limits = state.get("limits", _limits(values, name))
    colormap = "RdBu_r" if name in SIGNED_FIELDS else (
        "RdYlBu_r" if name == "mean_sst" else "viridis")
    image = axis.pcolormesh(data.longitude, data.latitude, np.ma.masked_invalid(values),
                            shading="auto", cmap=colormap, vmin=limits[0], vmax=limits[1],
                            rasterized=True)
    _configure_axis(axis, state)
    if not np.isfinite(values).any():
        axis.text(0.5, 0.5, _text(state, "无有效观测", "No observations"),
                  transform=axis.transAxes, ha="center")
    return image


def _map_panel(axis, data: xr.DataArray, state: dict) -> None:
    image = _draw_map(axis, data, state)
    axis.set_title(_field_label(state, data.name), fontsize=11)
    axis.figure.colorbar(image, ax=axis, shrink=0.85, pad=0.03, label=FIELD_LABELS[data.name][2])


def _title(figure, state: dict, chinese: str) -> None:
    period = state["context"].train_period
    title = _text(state, chinese, state["english_title"])
    figure.suptitle("%s (%s—%s)" % (title, period[0][:4], period[1][:4]), fontsize=15)


def _annual_overview(maps: xr.Dataset, state: dict):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 7.5), constrained_layout=True)
    annual = maps.sel(season="annual")
    for axis, name in zip(axes, ("mean_sst", "std_sst")):
        _map_panel(axis, annual[name], state)
    sample = annual.isel(latitude=slice(None, None, QUIVER_STRIDE),
                         longitude=slice(None, None, QUIVER_STRIDE))
    longitude, latitude = np.meshgrid(sample.longitude, sample.latitude)
    u = sample.mean_uo.values / np.cos(np.deg2rad(latitude))
    v = sample.mean_vo.values
    arrows = axes[0].quiver(longitude, latitude, u, v, color="#202020", angles="xy",
                            scale_units="xy", scale=1.0, width=0.003)
    key = axes[0].quiverkey(arrows, 0.12, 0.90, 0.5, "0.5 m/s",
                            angle=90, coordinates="axes", labelpos="E")
    key.text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    _title(figure, dict(state, english_title="Training SST and surface currents"),
           "训练期海表温度与表层海流")
    return figure


def _front_structure(maps: xr.Dataset, state: dict):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for axis, name in zip(axes, ("mean_gradient", "front_frequency")):
        _map_panel(axis, maps[name].sel(season="annual"), state)
    _title(figure, dict(state, english_title="Training front structure"),
           "训练期锋面结构（固定训练期P90阈值）")
    return figure


def _thermal_environment(maps: xr.Dataset, state: dict):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 12), constrained_layout=True)
    fields = ("mean_net_heat_flux", "mean_mixed_layer_depth",
              "mean_heat_increment", "mean_abs_heat_increment")
    for axis, name in zip(axes.flat, fields):
        _map_panel(axis, maps[name].sel(season="annual"), state)
    _title(figure, dict(state, english_title="Training thermal environment"),
           "训练期海气热交换与混合层热力环境")
    return figure


def _seasonal_structure(maps: xr.Dataset, state: dict):
    figure, axes = plt.subplots(2, 4, figsize=(16, 10.5), constrained_layout=True)
    for row, name in enumerate(("mean_gradient", "mean_heat_increment")):
        data = maps[name].sel(season=[item[0] for item in SEASON_LABELS])
        shared = dict(state, limits=_limits(_display_values(data.values, name), name))
        for column, (season, label) in enumerate(SEASON_LABELS):
            image = _draw_map(axes[row, column], maps[name].sel(season=season), shared)
            axes[row, column].set_title(_text(state, label + " (" + season + ")", season))
        figure.colorbar(image, ax=list(axes[row]), shrink=0.8, pad=0.02,
                        label=_field_label(state, name))
    _title(figure, dict(state, english_title="Austral seasonal front and heating structure"),
           "南半球季节锋面与热力结构")
    return figure


def _monthly_cycles(table: pd.DataFrame, state: dict):
    figure, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    fields = ("mean_sst", "mean_current_speed", "mean_gradient", "strong_front_area_fraction",
              "mean_net_heat_flux", "mean_mixed_layer_depth",
              "mean_heat_increment", "mean_abs_sst_change")
    for axis, name in zip(axes.flat, fields):
        axis.plot(table["month"], _display_values(table[name].to_numpy(), name),
                  marker="o", color="#2463a6", linewidth=1.6, markersize=4)
        axis.set_title(_field_label(state, name), fontsize=11)
        axis.set_xlabel(_text(state, "月份", "Month"))
        axis.set_xticks(np.arange(1, 13))
        axis.set_ylabel(FIELD_LABELS[name][2])
        axis.grid(alpha=0.25)
        if name in SIGNED_FIELDS:
            axis.axhline(0, color="#555555", linewidth=0.7)
    _title(figure, dict(state, english_title="Training monthly cycles; equal daily weights"),
           "训练期年内循环（日统计等权平均）")
    return figure


def render_diagnostic_figures(context: AnalysisContext, output_root: Path, options: dict) -> list:
    """Create five PNG/PDF figure pairs from existing training-only diagnostic tables."""
    output_root = Path(output_root)
    with xr.open_dataset(output_root / "seasonal_maps.nc") as opened:
        maps = opened.load()
    if maps.attrs.get("run_identity") != context.run_identity:
        raise DataValidationError("Diagnostic maps belong to a different prepared run")
    table = pd.read_csv(output_root / "monthly_cycle.csv")
    state = dict(_font_settings(), context=context)
    folder = output_root / "figures"
    folder.mkdir(exist_ok=True)
    functions = (_annual_overview, _front_structure, _thermal_environment, _seasonal_structure)
    artifacts = []
    settings = {"font.family": state["font"], "axes.unicode_minus": False,
                "font.size": 10, "savefig.facecolor": "white"}
    with plt.rc_context(settings):
        for name, function in zip(FIGURE_NAMES[:4], functions):
            artifacts.extend(_save_figure(function(maps, state), folder / name, options["dpi"]))
        artifacts.extend(_save_figure(_monthly_cycles(table, state),
                                      folder / FIGURE_NAMES[-1], options["dpi"]))
    return artifacts


def _save_figure(figure, stem: Path, dpi: int) -> list:
    paths = []
    try:
        for extension in ("png", "pdf"):
            path = stem.with_suffix("." + extension)
            figure.savefig(path, dpi=dpi)
            paths.append("figures/" + path.name)
    finally:
        plt.close(figure)
    return paths
