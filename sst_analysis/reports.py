"""Readable diagnostic outputs and reproducible analysis manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import xarray as xr

from sst_preprocessing.artifacts import read_json, write_json_atomic
from sst_preprocessing.errors import DataValidationError

from . import __version__
from .context import AnalysisContext
from .plots import FIELD_LABELS


def validate_diagnostic_artifacts(context: AnalysisContext, output_root: Path) -> dict:
    """Require matching training bounds/identity before re-rendering existing maps."""
    summary = read_json(Path(output_root) / "diagnostics_summary.json")
    if summary.get("status") != "complete" or summary.get("run_identity") != context.run_identity:
        raise DataValidationError("Diagnostics are incomplete or belong to another preprocessing run")
    if tuple(summary.get("train_period", ())) != context.train_period:
        raise DataValidationError("Diagnostic training bounds do not match the prepared run")
    return summary


def _regional_table(output_root: Path) -> list:
    daily = pd.read_csv(output_root / "daily_regional.csv")
    rows = []
    for name in ("mean_sst", "mean_current_speed", "mean_gradient", "strong_front_area_fraction",
                 "mean_net_heat_flux", "mean_mixed_layer_depth",
                 "mean_heat_increment", "mean_abs_sst_change"):
        values = daily[name].dropna()
        scale = 100.0 if name == "strong_front_area_fraction" else 1.0
        values = values * scale
        label, _, unit = FIELD_LABELS[name]
        if values.empty:
            rows.append("| %s | %s | 未定义 | 未定义 | 未定义 | 0 |" % (label, unit))
            continue
        rows.append("| %s | %s | %.6g | %.6g | %.6g | %d |" %
                    (label, unit, values.mean(), values.min(), values.max(), len(values)))
    return rows


def write_diagnostic_report(context: AnalysisContext, output_root: Path, options: dict) -> None:
    """Side effect: publish a Chinese report describing observations, never forecast skill."""
    output_root = Path(output_root)
    summary = validate_diagnostic_artifacts(context, output_root)
    lines = ["# 西南大西洋训练期区域诊断", "",
             "- 训练期：%s 至 %s。" % context.train_period,
             "- 读取训练日数：%d；物理月文件：%d。" %
             (summary["training_day_count"], len(summary["source_files"])),
             "- 固定锋面阈值：%.8f ℃/km，来自预处理训练期P90。" % context.front_threshold,
             "- 空间范围：核心区与固定海洋掩膜；梯度项另外使用有效差分模板。",
             "- 地图按格点作时间统计；区域逐日统计按余弦纬度面积权重归约。", "",
             "## 区域日均量的汇总", "",
             "下表极值指区域日均量的最小／最大值，不是逐格点极值。", "",
             "| 量 | 单位 | 日统计平均 | 日统计最小 | 日统计最大 | 有效日数 |",
             "| --- | --- | --- | --- | --- | --- |"]
    lines.extend(_regional_table(output_root))
    lines.extend(_interpretation_notes())
    lines.extend(["", "## 图表", ""])
    lines.extend("- [%s](%s)" % (path.split("/")[-1], path) for path in options["figures"])
    temporary = output_root / "diagnostics_report.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output_root / "diagnostics_report.md")


def _interpretation_notes() -> list:
    return ["", "## 统计口径与解释边界", "",
            "- 南半球夏、秋、冬、春分别为DJF、MAM、JJA、SON。",
            "- 季节图保留训练期所有对应月份；没有按完整跨年季节筛掉边界月份。",
            "- 强锋面频率采用固定阈值和有效梯度日数；不作空间平滑或连通域清理。",
            "- 海温日变化是相邻训练日的观测差，跨月连续，训练首日不计。",
            "- 热力倾向沿用净表面热量、3 m有效厚度下限；未加短波穿透或衰减。",
            "- 这些图表描述观测分布和季节变化，不能单独作为物理贡献比例或因果证据。",
            "- 本阶段未生成SST预测、无训练基线或模型性能结果。"]


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_analysis_manifest(context: AnalysisContext, output_root: Path, details: dict) -> None:
    """Side effect: record finished analysis identity, settings and software versions."""
    manifest = {"status": "complete", "kind": details["kind"],
                "run_identity": context.run_identity, "analysis_version": __version__,
                "analysis_code_sha256": _code_fingerprint(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "settings": details.get("settings", {}),
                "artifacts": details["artifacts"],
                "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "pandas": pd.__version__, "xarray": xr.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    write_json_atomic(Path(output_root) / "analysis_manifest.json", manifest)
