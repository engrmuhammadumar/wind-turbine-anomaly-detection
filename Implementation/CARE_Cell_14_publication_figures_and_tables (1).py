"""
CARE Cell 14 — publication figures, visual diagnostics, and metric tables.

Run after the locked Cell 13 final evaluation. This reporting-only cell:

* reads existing Cell 3–13 DataFrames from the notebook when available;
* otherwise discovers their persisted CSV files below CARE_OUTPUT_ROOT;
* produces 60+ publication-ready, scientifically defensible figures;
* exports every figure as PNG (600 dpi), vector PDF, and vector SVG;
* exports selected paper tables as CSV, LaTeX, PNG, and PDF;
* writes figure/table manifests, captions, a metric sheet, and a README; and
* never fits, recalibrates, rescores, or evaluates an alternative test policy.

The requested Windows output directory is used by default:
F:\\Umar-Wisal-Work\\Umar Wind-Turbine Work\\Implementation\\outputs\\paper results

Optional environment overrides for a separate Python process:
    CARE_SOURCE_ROOT         persisted CARE experiment directory
    CARE_PAPER_RESULTS_ROOT  publication-results output directory
    CARE_ALLOW_PARTIAL=1     allow a partial export when core tables are absent
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

try:
    import seaborn as sns
except Exception:  # pragma: no cover - matplotlib-only fallback
    sns = None


# =============================================================================
# 1. Immutable reporting configuration
# =============================================================================

PROJECT_ROOT = Path(
    r"F:\Umar-Wisal-Work\Umar Wind-Turbine Work\Implementation"
)
CARE_OUTPUT_ROOT = Path(
    os.environ.get(
        "CARE_SOURCE_ROOT",
        str(PROJECT_ROOT / "outputs" / "care_early_fault_detection"),
    )
)
PAPER_RESULTS_ROOT = Path(
    os.environ.get(
        "CARE_PAPER_RESULTS_ROOT",
        str(PROJECT_ROOT / "outputs" / "paper results"),
    )
)
ALLOW_PARTIAL = os.environ.get("CARE_ALLOW_PARTIAL", "0").strip() == "1"

PNG_DPI = 600
PREVIEW_DPI = 180
RASTER_FORMAT = "png"
VECTOR_FORMATS = ("pdf", "svg")
EXPECTED_MINIMUM_FIGURES = 60
FROZEN_CRITICALITY = 72
EXPECTED_STEP_MINUTES = 10
NOMINAL_ALPHA = 0.01
RANDOM_SEED = 20260812

FARM_ORDER = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
FARM_SHORT = {
    "Wind Farm A": "Farm A",
    "Wind Farm B": "Farm B",
    "Wind Farm C": "Farm C",
}
FARM_COLORS = {
    "Wind Farm A": "#0072B2",
    "Wind Farm B": "#E69F00",
    "Wind Farm C": "#009E73",
}
CLASS_COLORS = {False: "#4C78A8", True: "#D55E00"}
SPLIT_COLORS = {
    "train": "#4C78A8",
    "validation": "#F2B134",
    "test": "#D65F5F",
}
CARE_COLORS = {
    "coverage": "#0072B2",
    "accuracy": "#009E73",
    "reliability": "#D55E00",
    "earliness": "#CC79A7",
    "care_score": "#6A3D9A",
}

SECTION_DIRS = {
    "study_design": "01_study_design",
    "dataset": "02_dataset",
    "data_quality": "03_data_quality",
    "splits_windows": "04_splits_and_windows",
    "model_training": "05_model_and_training",
    "calibration": "06_score_calibration",
    "validation": "07_validation_selection",
    "test": "08_locked_test",
    "failure_analysis": "09_failure_analysis",
    "tables": "10_paper_tables",
    "catalog": "00_catalog",
}


def _configure_style() -> None:
    """Apply a consistent journal-style theme without changing any data."""

    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.titlesize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.45,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


_configure_style()
np.random.seed(RANDOM_SEED)


# =============================================================================
# 2. Source-table discovery with notebook-memory preference
# =============================================================================

TABLE_FILES: dict[str, str] = {
    # Dataset and quality
    "care_case_registry": "care_case_registry.csv",
    "care_event_label_summary": "care_event_label_summary.csv",
    "care_farm_schema_summary": "care_farm_schema_summary.csv",
    "all_case_quality_summary": "care_all_case_quality_summary.csv",
    "canonical_asset_summary": "care_canonical_asset_summary.csv",
    "all_case_category_distribution": "care_all_case_category_distribution.csv",
    "all_case_sampling_interval_distribution": "care_all_case_sampling_interval_distribution.csv",
    "all_case_signal_quality": "care_all_case_signal_quality.csv",
    "farm_signal_quality_summary": "care_farm_signal_quality_summary.csv",
    "farm_operational_summary": "care_farm_operational_summary.csv",
    "modeling_eligibility_registry": "care_modeling_eligibility_registry.csv",
    # Split, preprocessing, and windows
    "care_case_split_registry": "care_case_split_registry.csv",
    "care_asset_split_assignment": "care_asset_split_assignment.csv",
    "care_split_overall_summary": "care_split_overall_summary.csv",
    "care_split_farm_class_summary": "care_split_farm_class_summary.csv",
    "care_train_preprocessing_farm_summary": "care_train_preprocessing_farm_summary.csv",
    "care_train_preprocessing_signal_parameters": "care_train_preprocessing_signal_parameters.csv",
    "care_window_case_audit": "care_multiscale_window_case_audit.csv",
    "care_window_split_summary": "care_multiscale_window_split_summary.csv",
    "care_window_farm_split_summary": "care_multiscale_window_farm_split_summary.csv",
    "care_window_exclusion_summary": "care_multiscale_window_exclusion_summary.csv",
    "care_window_feature_farm_summary": "care_multiscale_window_feature_farm_summary.csv",
    "care_dataset_split_farm_summary": "care_dataset_split_farm_summary.csv",
    # Training
    "care_model_training_history": "care_model_training_history.csv",
    "care_model_training_summary": "care_model_training_summary.csv",
    "care_model_selection_audit": "care_model_selection_audit.csv",
    # Normal-control calibration
    "care_normal_score_calibration_components": "care_normal_score_calibration_components.csv",
    "care_normal_validation_score_components": "care_normal_validation_score_components.csv",
    "care_anomaly_score_calibration_summary": "care_anomaly_score_calibration_summary.csv",
    "care_operating_regime_calibration": "care_operating_regime_calibration.csv",
    # Validation-only selection
    "care_validation_anomaly_score_components": "care_validation_anomaly_score_components.csv",
    "care_validation_case_base_metrics": "care_validation_case_base_metrics.csv",
    "care_validation_policy_grid": "care_validation_policy_grid.csv",
    "care_validation_event_metrics": "care_validation_event_metrics.csv",
    "care_validation_care_summary": "care_validation_care_summary.csv",
    "care_selected_sequential_policy": "care_selected_sequential_policy.csv",
    # Locked final test
    "care_test_anomaly_score_components": "care_test_anomaly_score_components.csv",
    "care_test_case_base_metrics": "care_test_case_base_metrics.csv",
    "care_test_event_metrics": "care_test_event_metrics.csv",
    "care_test_farm_summary": "care_test_farm_summary.csv",
    "care_test_care_summary": "care_test_care_summary.csv",
    "care_validation_test_care_comparison": "care_validation_test_care_comparison.csv",
    "care_final_evaluation_audit": "care_final_evaluation_audit.csv",
    "care_final_evaluation_constraint_audit": "care_final_evaluation_constraint_audit.csv",
}


@dataclass
class LoadedTable:
    name: str
    frame: pd.DataFrame
    source: str


TABLES: dict[str, LoadedTable] = {}
LOAD_WARNINGS: list[str] = []


def _most_recent_csv(filename: str) -> Path | None:
    if not CARE_OUTPUT_ROOT.exists():
        return None
    candidates = [p for p in CARE_OUTPUT_ROOT.rglob(filename) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _load_table(name: str, filename: str) -> None:
    obj = globals().get(name)
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        TABLES[name] = LoadedTable(name, obj.copy(), f"notebook:{name}")
        return
    path = _most_recent_csv(filename)
    if path is None:
        LOAD_WARNINGS.append(f"Missing optional table: {name} ({filename})")
        return
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        LOAD_WARNINGS.append(f"Could not read {path}: {exc}")
        return
    TABLES[name] = LoadedTable(name, frame, str(path))


for _name, _filename in TABLE_FILES.items():
    _load_table(_name, _filename)


def table(name: str) -> pd.DataFrame | None:
    item = TABLES.get(name)
    return None if item is None else item.frame


def require_table(name: str, columns: Sequence[str] = ()) -> pd.DataFrame:
    frame = table(name)
    if frame is None:
        raise RuntimeError(
            f"Required reporting table {name!r} is unavailable. Run Cell 14 in "
            "the same notebook after Cell 13, or set CARE_SOURCE_ROOT to the "
            "persisted care_early_fault_detection directory."
        )
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{name!r} is missing required columns: {missing}")
    return frame


CORE_TABLES = (
    "care_model_training_history",
    "care_anomaly_score_calibration_summary",
    "care_validation_anomaly_score_components",
    "care_validation_policy_grid",
    "care_validation_event_metrics",
    "care_test_anomaly_score_components",
    "care_test_event_metrics",
    "care_validation_test_care_comparison",
)
missing_core = [name for name in CORE_TABLES if table(name) is None]
if missing_core and not ALLOW_PARTIAL:
    raise RuntimeError(
        "Cell 14 requires the completed Cell 10–13 reporting tables. Missing: "
        + ", ".join(missing_core)
        + ". No figures were generated."
    )


# =============================================================================
# 3. Output, manifests, captions, and reusable plot helpers
# =============================================================================

for _section_dir in SECTION_DIRS.values():
    (PAPER_RESULTS_ROOT / _section_dir).mkdir(parents=True, exist_ok=True)

FIGURE_ROWS: list[dict[str, Any]] = []
TABLE_ROWS: list[dict[str, Any]] = []
METRIC_ROWS: list[dict[str, Any]] = []
SKIPPED_ROWS: list[dict[str, Any]] = []
PLOT_AGGREGATION_ROWS: list[dict[str, Any]] = []


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "item"


def short_farm(value: Any) -> str:
    return FARM_SHORT.get(str(value), str(value).replace("Wind Farm ", "Farm "))


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _farm_numeric_values(
    frame: pd.DataFrame,
    column: str,
    source_table: str,
    aggregation: str = "median",
) -> pd.Series:
    """Return one numeric value per farm without assuming unique input rows.

    Identical duplicate values are collapsed without arithmetic. When duplicate
    rows differ, rates/fractions are averaged, additive quality counts can be
    summed explicitly, and summary-table values use the requested conservative
    aggregation. This affects plotting only; source tables remain unchanged.
    """

    if "farm" not in frame.columns or column not in frame.columns:
        return pd.Series(index=pd.Index(FARM_ORDER, name="farm"), dtype=float)

    work = frame.loc[:, ["farm", column]].copy()
    work["farm"] = work["farm"].astype(str).str.strip()
    work[column] = _numeric(work[column])
    work = work.loc[work["farm"].ne("") & work[column].notna()]
    if work.empty:
        return pd.Series(index=pd.Index(FARM_ORDER, name="farm"), dtype=float)

    column_key = column.lower()

    def reduce_group(group: pd.Series) -> float:
        values = group.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return float("nan")
        if len(values) == 1 or np.allclose(values, values[0], rtol=1e-12, atol=1e-12):
            return float(values[0])
        if aggregation == "auto_quality":
            if any(token in column_key for token in ("fraction", "rate", "percent", "ratio")):
                rule = "mean"
            elif any(token in column_key for token in ("count", "rows", "windows", "points", "values")):
                rule = "sum"
            else:
                rule = "median"
        else:
            rule = aggregation
        if rule == "sum":
            return float(np.sum(values))
        if rule == "mean":
            return float(np.mean(values))
        if rule == "first":
            return float(values[0])
        return float(np.median(values))

    duplicate_counts = work.groupby("farm", observed=False).size()
    duplicate_counts = duplicate_counts.loc[duplicate_counts.gt(1)]
    if not duplicate_counts.empty:
        applied_rule = aggregation
        if aggregation == "auto_quality":
            if any(token in column_key for token in ("fraction", "rate", "percent", "ratio")):
                applied_rule = "mean for differing values; identical duplicates collapsed"
            elif any(token in column_key for token in ("count", "rows", "windows", "points", "values")):
                applied_rule = "sum for differing values; identical duplicates collapsed"
            else:
                applied_rule = "median for differing values; identical duplicates collapsed"
        else:
            applied_rule = f"{aggregation} for differing values; identical duplicates collapsed"
        PLOT_AGGREGATION_ROWS.append(
            {
                "source_table": source_table,
                "column": column,
                "duplicate_farms": "; ".join(
                    f"{farm} ({int(count)} rows)" for farm, count in duplicate_counts.items()
                ),
                "plot_only_aggregation": applied_rule,
                "source_modified": False,
            }
        )

    values = work.groupby("farm", sort=False, observed=False)[column].apply(reduce_group)
    return values.reindex(FARM_ORDER)


def _farm_numeric_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    source_table: str,
    aggregation: str = "median",
) -> pd.DataFrame:
    """Build a farm-indexed numeric frame using duplicate-safe reductions."""

    return pd.concat(
        {
            column: _farm_numeric_values(frame, column, source_table, aggregation)
            for column in columns
        },
        axis=1,
    )


def _set_farm_ticklabels(ax: mpl.axes.Axes, farms: Sequence[Any] | None = None) -> None:
    """Relabel categorical farm ticks without Matplotlib FixedFormatter warnings."""

    ticks = np.asarray(ax.get_xticks())
    labels = (
        [short_farm(value) for value in farms]
        if farms is not None
        else [short_farm(label.get_text()) for label in ax.get_xticklabels()]
    )
    if len(ticks) == len(labels):
        ax.set_xticks(ticks, labels, rotation=0)


def _available(frame: pd.DataFrame | None, *columns: str) -> bool:
    return frame is not None and all(column in frame.columns for column in columns)


def _source_string(source_tables: Sequence[str]) -> str:
    values = []
    for source_name in source_tables:
        loaded = TABLES.get(source_name)
        values.append(loaded.source if loaded is not None else source_name)
    return " | ".join(values)


def save_figure(
    fig: mpl.figure.Figure,
    section: str,
    slug: str,
    title: str,
    caption: str,
    source_tables: Sequence[str],
    priority: str = "Supplementary",
    suggested_section: str = "Supplementary material",
) -> None:
    """Save one scientific figure in raster and vector formats."""

    figure_number = len(FIGURE_ROWS) + 1
    figure_id = f"F{figure_number:03d}"
    folder = PAPER_RESULTS_ROOT / SECTION_DIRS[section]
    base = folder / f"{figure_id}_{slugify(slug)}"
    fig.savefig(base.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    FIGURE_ROWS.append(
        {
            "figure_id": figure_id,
            "title": title,
            "caption": caption,
            "paper_priority": priority,
            "suggested_section": suggested_section,
            "png_path": str(base.with_suffix(".png")),
            "pdf_path": str(base.with_suffix(".pdf")),
            "svg_path": str(base.with_suffix(".svg")),
            "source_tables": _source_string(source_tables),
            "reporting_only": True,
            "test_policy_sweep": False,
        }
    )


def skip_plot(name: str, reason: str) -> None:
    SKIPPED_ROWS.append({"item": name, "reason": reason})


def add_metric(
    section: str,
    metric: str,
    value: Any,
    unit: str = "",
    scope: str = "",
    source_table: str = "",
) -> None:
    METRIC_ROWS.append(
        {
            "section": section,
            "scope": scope,
            "metric": metric,
            "value": value,
            "unit": unit,
            "source_table": source_table,
        }
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: mpl.axes.Axes, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()
        if not (np.isfinite(height) and np.isfinite(width)):
            continue
        if width >= height and abs(width) > 0:
            ax.text(
                patch.get_x() + width,
                patch.get_y() + patch.get_height() / 2,
                " " + fmt.format(width),
                va="center",
                ha="left",
                fontsize=7.5,
            )
        else:
            ax.text(
                patch.get_x() + width / 2,
                height,
                " " + fmt.format(height),
                va="bottom",
                ha="center",
                fontsize=7.5,
            )


def _grouped_bar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    hue: str,
    title: str,
    ylabel: str,
    palette: Mapping[Any, str] | None = None,
    percent: bool = False,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    if sns is not None:
        sns.barplot(
            data=frame,
            x=category,
            y=value,
            hue=hue,
            palette=palette,
            errorbar=None,
            ax=ax,
        )
    else:
        pivot = frame.pivot(index=category, columns=hue, values=value)
        pivot.plot(kind="bar", ax=ax, color=None if palette is None else list(palette.values()))
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, title="")
    clean_axis(ax)
    fig.tight_layout()
    return fig


def _simple_bar(
    labels: Sequence[Any],
    values: Sequence[float],
    title: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
    horizontal: bool = False,
    percent: bool = False,
    figsize: tuple[float, float] = (6.8, 3.8),
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    labels_text = [str(v) for v in labels]
    values_array = np.asarray(values, dtype=float)
    if horizontal:
        bars = ax.barh(labels_text, values_array, color=colors)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
        ax.invert_yaxis()
        for bar, value in zip(bars, values_array):
            if np.isfinite(value):
                ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3g}", va="center", fontsize=7.5)
        ax.grid(True, axis="x", alpha=0.22)
        ax.grid(False, axis="y")
    else:
        bars = ax.bar(labels_text, values_array, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        for bar, value in zip(bars, values_array):
            if np.isfinite(value):
                label = f"{value:.1%}" if percent else f"{value:.3g}"
                ax.text(bar.get_x() + bar.get_width() / 2, value, " " + label, ha="center", va="bottom", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.22)
        ax.grid(False, axis="x")
    if percent:
        (ax.xaxis if horizontal else ax.yaxis).set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title, loc="left", fontweight="bold")
    clean_axis(ax)
    fig.tight_layout()
    return fig


def _hist_overlay(
    frame: pd.DataFrame,
    value: str,
    group: str,
    title: str,
    xlabel: str,
    threshold: float | None = None,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for key, subset in frame.groupby(group, sort=True, observed=False):
        data = _numeric(subset[value]).dropna().to_numpy()
        if len(data) == 0:
            continue
        ax.hist(data, bins=45, density=True, histtype="step", linewidth=1.5, label=str(key))
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color="#C44E52", linestyle="--", linewidth=1.3, label="Threshold")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    clean_axis(ax)
    fig.tight_layout()
    return fig


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    y = np.arange(1, len(x) + 1, dtype=float) / max(1, len(x))
    return x, y


def _heatmap(
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    fmt: str = ".0f",
    cmap: str = "Blues",
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    if sns is not None:
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, cbar=False, square=True, ax=ax)
    else:
        ax.imshow(matrix, cmap=cmap)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center")
    ax.set_xticks(ax.get_xticks(), xlabels)
    ax.set_yticks(ax.get_yticks(), ylabels, rotation=0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    return fig


# =============================================================================
# 4. Study-design and method schematics (all manuscript-safe)
# =============================================================================

def _box(ax: mpl.axes.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.10),
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)


def _arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color=color))


def make_method_schematics() -> None:
    # F001: full experimental workflow
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    labels = [
        "Case/asset\nregistry",
        "Frozen\ntrain–validation–test",
        "Train-only\npreprocessing",
        "Farm-specific\nforecasting models",
        "Normal-control\nconformal calibration",
        "Validation-only\npolicy selection",
        "Single-use\nlocked test",
    ]
    colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    xs = np.linspace(0.015, 0.865, len(labels))
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        _box(ax, (x, 0.38), (0.115, 0.27), label, color)
        if i < len(labels) - 1:
            _arrow(ax, (x + 0.115, 0.515), (xs[i + 1], 0.515))
    ax.text(0.5, 0.88, "Leakage-controlled CARE experimental workflow", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.15, "No model, score, threshold, or persistence tuning after test labels were opened", ha="center", color="#A33A2B", fontsize=9.5, fontweight="bold")
    save_figure(fig, "study_design", "leakage_controlled_workflow", "Leakage-controlled CARE workflow", "Case/asset-disjoint experimental workflow from frozen assignment through the single-use final test.", [], "Main text", "Methods")

    # F002: leakage firewall
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.04, 0.63), (0.24, 0.22), "TRAIN\nfit model + preprocessing", "#4C78A8")
    _box(ax, (0.38, 0.63), (0.24, 0.22), "VALIDATION\ncalibrate + select policy", "#E69F00")
    _box(ax, (0.72, 0.63), (0.24, 0.22), "TEST\none final report", "#D55E00")
    _arrow(ax, (0.28, 0.74), (0.38, 0.74)); _arrow(ax, (0.62, 0.74), (0.72, 0.74))
    ax.add_patch(Rectangle((0.665, 0.08), 0.018, 0.82, facecolor="#A33A2B", alpha=0.85))
    ax.text(0.674, 0.49, "FROZEN BOUNDARY", rotation=90, color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.50, 0.30, "Allowed", color="#26734D", fontweight="bold", ha="center")
    ax.text(0.50, 0.21, "normal-control calibration; validation-only criticality choice", ha="center")
    ax.text(0.83, 0.30, "Forbidden", color="#A33A2B", fontweight="bold", ha="center")
    ax.text(0.83, 0.18, "test threshold grid\npost-test refit or reselection", ha="center")
    save_figure(fig, "study_design", "leakage_firewall", "Information-flow firewall", "Permitted information flow and the frozen boundary protecting the single-use test evaluation.", [], "Main text", "Methods")

    # F003: multiscale window geometry
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.set_xlim(-172, 8); ax.set_ylim(-0.1, 3.0)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.broken_barh([(-168, 168)], (1.9, 0.55), facecolors="#4C78A8", alpha=0.75)
    ax.broken_barh([(-24, 24)], (1.05, 0.55), facecolors="#009E73", alpha=0.82)
    ax.broken_barh([(0, 1)], (0.2, 0.55), facecolors="#D55E00", alpha=0.9)
    ax.text(-84, 2.18, "Long context: 168 hourly samples (7 days)", ha="center", va="center", color="white", fontweight="bold")
    ax.text(-12, 1.32, "Short context: 144 × 10 min (24 h)", ha="center", va="center", color="white", fontweight="bold")
    ax.text(0.5, 0.47, "1 h\nforecast", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xlabel("Hours relative to forecast origin")
    ax.set_yticks([])
    ax.set_title("Dual-scale forecasting window", loc="left", fontweight="bold")
    ax.set_xlim(-172, 5)
    clean_axis(ax); fig.tight_layout()
    save_figure(fig, "study_design", "multiscale_window_geometry", "Dual-scale input window", "The model uses a 24-hour 10-minute context and a seven-day hourly context to forecast primary signals one hour ahead.", [], "Main text", "Methods")

    # F004: architecture
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.03, 0.68), (0.18, 0.18), "24 h context\n144 × F", "#009E73")
    _box(ax, (0.03, 0.20), (0.18, 0.18), "7 d context\n168 × F", "#4C78A8")
    _box(ax, (0.29, 0.68), (0.18, 0.18), "Patch Conv1d\n6 steps → 64 d", "#009E73")
    _box(ax, (0.29, 0.20), (0.18, 0.18), "Patch Conv1d\n6 steps → 64 d", "#4C78A8")
    _box(ax, (0.55, 0.68), (0.18, 0.18), "2 Transformer layers\n4 heads; FFN 128", "#009E73")
    _box(ax, (0.55, 0.20), (0.18, 0.18), "2 Transformer layers\n4 heads; FFN 128", "#4C78A8")
    _box(ax, (0.78, 0.41), (0.18, 0.22), "Mean + last token\n× 2 branches\n→ MLP forecast", "#6A3D9A")
    for y in (0.77, 0.29):
        _arrow(ax, (0.21, y), (0.29, y)); _arrow(ax, (0.47, y), (0.55, y))
    _arrow(ax, (0.73, 0.77), (0.78, 0.56)); _arrow(ax, (0.73, 0.29), (0.78, 0.48))
    ax.text(0.5, 0.95, "Farm-specific multiscale patch-transformer forecaster", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.87, 0.30, "46 / 63 / 238\nprimary targets", ha="center", fontsize=8.5)
    save_figure(fig, "study_design", "model_architecture", "Forecasting-model architecture", "Dual-branch patch-transformer architecture used independently for Farms A, B, and C.", ["care_model_training_summary"], "Main text", "Methods")

    # F005: anomaly score
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.03, 0.58), (0.21, 0.20), "Top-5% sensor\nforecast residual", "#0072B2")
    _box(ax, (0.03, 0.20), (0.21, 0.20), "Latent distance to\noperating regime", "#CC79A7")
    _box(ax, (0.33, 0.39), (0.23, 0.22), "Reference-fitted\nrobust z scores\npositive part only", "#6A3D9A")
    _box(ax, (0.66, 0.39), (0.27, 0.22), "Weighted anomaly score\n0.70 residual + 0.30 latent", "#D55E00")
    _arrow(ax, (0.24, 0.68), (0.33, 0.53)); _arrow(ax, (0.24, 0.30), (0.33, 0.47)); _arrow(ax, (0.56, 0.50), (0.66, 0.50))
    ax.text(0.5, 0.90, "Frozen anomaly-score construction", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.08, "Thresholds are conformal order statistics from wholly normal validation controls", ha="center", fontsize=9)
    save_figure(fig, "study_design", "anomaly_score_construction", "Anomaly-score construction", "Forecast residual and latent-deviation components are robustly normalized on training references and combined before conformal thresholding.", ["care_anomaly_score_calibration_summary"], "Main text", "Methods")

    # F006: calibration split
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.04, 0.58), (0.25, 0.22), "Normal source-training cases\nfit regimes + score scaling", "#4C78A8")
    _box(ax, (0.38, 0.58), (0.25, 0.22), "Wholly normal validation cases\nconformal order statistic", "#E69F00")
    _box(ax, (0.72, 0.58), (0.24, 0.22), "Frozen farm/regime\nanomaly thresholds", "#D55E00")
    _arrow(ax, (0.29, 0.69), (0.38, 0.69)); _arrow(ax, (0.63, 0.69), (0.72, 0.69))
    ax.text(0.50, 0.31, "Case- and asset-disjoint reference → calibration handoff", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.50, 0.19, "Anomaly-validation and test tensors were not used", ha="center", color="#A33A2B", fontsize=9.5)
    save_figure(fig, "study_design", "normal_control_calibration", "Normal-control calibration protocol", "Training references define the score; disjoint normal validation cases define only the conformal threshold.", ["care_anomaly_score_calibration_summary"], "Main text", "Methods")

    # F007: sequential criticality rule
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.05, 0.55), (0.22, 0.23), "Actionable window\nscore > threshold", "#D55E00")
    _box(ax, (0.39, 0.55), (0.22, 0.23), "Criticality +1\n(non-flag: −1)", "#6A3D9A")
    _box(ax, (0.73, 0.55), (0.22, 0.23), "Alarm if criticality ≥72\n12 h net evidence", "#A33A2B")
    _arrow(ax, (0.27, 0.665), (0.39, 0.665)); _arrow(ax, (0.61, 0.665), (0.73, 0.665))
    ax.text(0.50, 0.31, "Non-actionable states pause the counter; segment gaps reset it", ha="center", fontsize=9.5)
    ax.text(0.50, 0.18, "The same 72-point rule was frozen on validation before final test inference", ha="center", fontsize=9.5, fontweight="bold")
    save_figure(fig, "study_design", "criticality_rule", "CARE sequential criticality rule", "Binary score exceedances update a bounded criticality counter; non-actionable states pause and temporal gaps reset the sequence.", ["care_selected_sequential_policy"], "Main text", "Methods")

    # F008: freeze timeline
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    steps = ["Split", "Preprocess", "Train", "Calibrate", "Select", "Test", "Lock"]
    xs = np.arange(len(steps))
    ax.plot(xs, np.zeros_like(xs), color="#555555", linewidth=1.4)
    ax.scatter(xs, np.zeros_like(xs), s=100, c=["#4C78A8"] * 3 + ["#E69F00"] * 2 + ["#D55E00"] * 2, zorder=3)
    for x, label in zip(xs, steps):
        ax.text(x, 0.10 if x % 2 == 0 else -0.13, label, ha="center", va="center", fontweight="bold")
    ax.axvspan(4.5, 6.4, color="#D55E00", alpha=0.08)
    ax.text(5.5, 0.25, "No tuning", color="#A33A2B", ha="center", fontweight="bold")
    ax.set_ylim(-0.35, 0.4); ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("Experiment freeze and single-use evaluation timeline", loc="left", fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "study_design", "experiment_freeze_timeline", "Experiment freeze timeline", "Ordered freeze points separating development from the single-use final evaluation.", [], "Main text", "Methods")


make_method_schematics()


# =============================================================================
# 5. Dataset, quality, split, preprocessing, and window figures
# =============================================================================

def make_dataset_figures() -> None:
    cases = table("care_case_registry")
    if cases is not None:
        work = cases.copy()
        if "is_anomaly" in work:
            work["case_class"] = _bool_series(work["is_anomaly"]).map({True: "Anomaly", False: "Normal"})
        if _available(work, "farm", "case_class"):
            counts = work.groupby(["farm", "case_class"], observed=False).size().rename("cases").reset_index()
            fig = _grouped_bar(counts, "farm", "cases", "case_class", "Dataset cases by farm and class", "Cases", {"Normal": "#4C78A8", "Anomaly": "#D55E00"})
            save_figure(fig, "dataset", "cases_by_farm_and_class", "Cases by farm and class", "Distribution of normal and anomaly cases across the three wind farms.", ["care_case_registry"], "Main text", "Dataset")
            totals = work.groupby("farm").size().reindex(FARM_ORDER)
            fig = _simple_bar([short_farm(x) for x in totals.index], totals.values, "Total cases by farm", "Cases", [FARM_COLORS.get(x, "#777777") for x in totals.index])
            save_figure(fig, "dataset", "total_cases_by_farm", "Total cases by farm", "Total number of available cases in each wind farm.", ["care_case_registry"], "Supplementary", "Dataset")
            anomaly_fraction = work.groupby("farm")["is_anomaly"].apply(lambda s: _bool_series(s).mean()).reindex(FARM_ORDER)
            fig = _simple_bar([short_farm(x) for x in anomaly_fraction.index], anomaly_fraction.values, "Anomaly-case fraction by farm", "Fraction", [FARM_COLORS.get(x, "#777777") for x in anomaly_fraction.index], percent=True)
            save_figure(fig, "dataset", "anomaly_fraction_by_farm", "Anomaly-case fraction", "Fraction of cases labeled as anomalous in each wind farm.", ["care_case_registry"], "Supplementary", "Dataset")
        if _available(work, "farm", "file_size_mb"):
            fig, ax = plt.subplots(figsize=(6.8, 3.9))
            order = [x for x in FARM_ORDER if x in set(work["farm"].astype(str))]
            if sns is not None:
                sns.boxplot(data=work, x="farm", y="file_size_mb", order=order, palette=FARM_COLORS, ax=ax)
                sns.stripplot(data=work, x="farm", y="file_size_mb", order=order, color="#333333", alpha=0.45, size=2.6, ax=ax)
            _set_farm_ticklabels(ax, order); ax.set_xlabel(""); ax.set_ylabel("File size (MB)")
            ax.set_title("Case-file size distribution", loc="left", fontweight="bold"); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "dataset", "case_file_size_distribution", "Case-file size distribution", "Distribution of source CSV sizes across farms.", ["care_case_registry"], "Supplementary", "Dataset")
        if _available(work, "farm", "event_start", "event_end"):
            start = pd.to_datetime(work["event_start"], errors="coerce")
            end = pd.to_datetime(work["event_end"], errors="coerce")
            work["event_duration_days"] = (end - start).dt.total_seconds() / 86400.0
            valid = work.loc[work["event_duration_days"].ge(0)].copy()
            if not valid.empty:
                fig, ax = plt.subplots(figsize=(7.0, 4.0))
                if sns is not None:
                    sns.boxplot(data=valid, x="farm", y="event_duration_days", hue="case_class" if "case_class" in valid else None, palette={"Normal": "#4C78A8", "Anomaly": "#D55E00"}, ax=ax)
                _set_farm_ticklabels(ax); ax.set_xlabel(""); ax.set_ylabel("Labeled interval (days)")
                ax.set_title("Labeled interval duration", loc="left", fontweight="bold"); clean_axis(ax); fig.tight_layout()
                save_figure(fig, "dataset", "labeled_interval_duration", "Labeled interval duration", "Distribution of metadata-defined event intervals by farm and class.", ["care_case_registry"], "Supplementary", "Dataset")
        if _available(work, "farm", "dataset_asset_id"):
            assets = work.groupby("farm")["dataset_asset_id"].nunique().reindex(FARM_ORDER)
            fig = _simple_bar([short_farm(x) for x in assets.index], assets.values, "Unique assets represented", "Unique assets", [FARM_COLORS.get(x, "#777777") for x in assets.index])
            save_figure(fig, "dataset", "unique_assets_by_farm", "Unique assets by farm", "Number of distinct turbine/asset identifiers represented per farm.", ["care_case_registry"], "Supplementary", "Dataset")

    schema = table("care_farm_schema_summary")
    if _available(schema, "farm", "all_signals", "primary_avg_signals"):
        long = schema.melt(id_vars="farm", value_vars=["all_signals", "primary_avg_signals"], var_name="signal_set", value_name="count")
        long["signal_set"] = long["signal_set"].map({"all_signals": "All signals", "primary_avg_signals": "Primary targets"})
        fig = _grouped_bar(long, "farm", "count", "signal_set", "Signal dimensionality by farm", "Signals", {"All signals": "#7F8C8D", "Primary targets": "#0072B2"})
        save_figure(fig, "dataset", "signal_dimensionality", "Signal dimensionality", "All available signals and primary forecasting targets for each farm.", ["care_farm_schema_summary"], "Main text", "Dataset")
        stat_cols = [c for c in ("primary_avg_signals", "min_signals", "max_signals", "std_signals", "unsuffixed_signals") if c in schema.columns]
        if stat_cols:
            plot = _farm_numeric_frame(schema, stat_cols, "care_farm_schema_summary")
            fig, ax = plt.subplots(figsize=(7.4, 4.1))
            plot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
            _set_farm_ticklabels(ax, plot.index); ax.set_xlabel(""); ax.set_ylabel("Signals")
            ax.set_title("Feature-statistic composition", loc="left", fontweight="bold"); ax.legend([x.replace("_signals", "").replace("_", " ").title() for x in stat_cols], frameon=False, ncol=3)
            clean_axis(ax); fig.tight_layout()
            save_figure(fig, "dataset", "feature_statistic_composition", "Feature-statistic composition", "Composition of primary/average, minimum, maximum, standard-deviation, and unsuffixed signals.", ["care_farm_schema_summary"], "Supplementary", "Dataset")


def make_quality_figures() -> None:
    farm_quality = table("farm_signal_quality_summary")
    if farm_quality is not None and "farm" in farm_quality.columns:
        numeric_cols = [c for c in farm_quality.columns if c != "farm" and pd.api.types.is_numeric_dtype(farm_quality[c])]
        preferred = [c for c in numeric_cols if any(token in c.lower() for token in ("missing", "zero", "constant", "invalid", "nonfinite"))]
        for column in preferred[:4]:
            values = _farm_numeric_values(
                farm_quality,
                column,
                "farm_signal_quality_summary",
                aggregation="auto_quality",
            )
            if values.notna().any():
                fig = _simple_bar([short_farm(x) for x in values.index], values.values, column.replace("_", " ").title(), column.replace("_", " "), [FARM_COLORS.get(x, "#777777") for x in values.index], percent="fraction" in column.lower() or "rate" in column.lower())
                save_figure(fig, "data_quality", column, column.replace("_", " ").title(), f"Farm-level data-quality metric: {column.replace('_', ' ')}.", ["farm_signal_quality_summary"], "Supplementary", "Data quality")

    sampling = table("all_case_sampling_interval_distribution")
    if sampling is not None:
        farm_col = next((c for c in ("farm", "farm_name") if c in sampling.columns), None)
        interval_col = next((c for c in sampling.columns if "interval" in c.lower() and pd.api.types.is_numeric_dtype(sampling[c])), None)
        count_col = next((c for c in sampling.columns if c.lower() in {"count", "rows", "frequency"}), None)
        if farm_col and interval_col:
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            if count_col:
                for farm, subset in sampling.groupby(farm_col, sort=True):
                    ax.plot(_numeric(subset[interval_col]), _numeric(subset[count_col]), marker="o", label=short_farm(farm), color=FARM_COLORS.get(str(farm)))
                ax.set_ylabel("Count")
            else:
                for farm, subset in sampling.groupby(farm_col, sort=True):
                    ax.hist(_numeric(subset[interval_col]).dropna(), bins=30, histtype="step", label=short_farm(farm), color=FARM_COLORS.get(str(farm)))
                ax.set_ylabel("Frequency")
            ax.set_xlabel(interval_col.replace("_", " ")); ax.set_title("Sampling-interval audit", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "data_quality", "sampling_interval_audit", "Sampling-interval audit", "Observed timestamp-step distribution used to verify the 10-minute source cadence.", ["all_case_sampling_interval_distribution"], "Supplementary", "Data quality")

    operational = table("farm_operational_summary")
    if operational is not None and "farm" in operational.columns:
        fraction_cols = [c for c in operational.columns if any(token in c.lower() for token in ("fraction", "rate", "percent")) and pd.api.types.is_numeric_dtype(operational[c])]
        for column in fraction_cols[:3]:
            values = _farm_numeric_values(
                operational,
                column,
                "farm_operational_summary",
                aggregation="mean",
            )
            fig = _simple_bar([short_farm(x) for x in values.index], values.values, column.replace("_", " ").title(), "Fraction", [FARM_COLORS.get(x, "#777777") for x in values.index], percent=True)
            save_figure(fig, "data_quality", column, column.replace("_", " ").title(), f"Farm-level operational-status metric: {column.replace('_', ' ')}.", ["farm_operational_summary"], "Supplementary", "Data quality")


def make_split_window_figures() -> None:
    split_registry = table("care_case_split_registry")
    if split_registry is not None and "farm" in split_registry.columns:
        split_col = next((c for c in ("model_split", "split") if c in split_registry.columns), None)
        anomaly_col = next((c for c in ("is_anomaly", "case_is_anomaly") if c in split_registry.columns), None)
        if split_col:
            counts = split_registry.groupby(["farm", split_col], observed=False).size().rename("cases").reset_index()
            fig = _grouped_bar(counts, "farm", "cases", split_col, "Case allocation by farm and split", "Cases", SPLIT_COLORS)
            save_figure(fig, "splits_windows", "case_allocation_by_farm_split", "Case allocation by farm and split", "Frozen case-level train, validation, and test allocation for each farm.", ["care_case_split_registry"], "Main text", "Experimental protocol")
            split_totals = split_registry.groupby(split_col).size().reindex(["train", "validation", "test"]).dropna()
            fig = _simple_bar([str(x).title() for x in split_totals.index], split_totals.values, "Overall case allocation", "Cases", [SPLIT_COLORS.get(str(x), "#777777") for x in split_totals.index])
            save_figure(fig, "splits_windows", "overall_case_allocation", "Overall case allocation", "Total number of cases assigned to each frozen experimental split.", ["care_case_split_registry"], "Supplementary", "Experimental protocol")
            if anomaly_col:
                tmp = split_registry.copy(); tmp["case_class"] = _bool_series(tmp[anomaly_col]).map({True: "Anomaly", False: "Normal"})
                counts = tmp.groupby([split_col, "case_class"], observed=False).size().rename("cases").reset_index()
                fig = _grouped_bar(counts, split_col, "cases", "case_class", "Class balance by split", "Cases", {"Normal": "#4C78A8", "Anomaly": "#D55E00"})
                save_figure(fig, "splits_windows", "class_balance_by_split", "Class balance by split", "Normal and anomaly case counts within the frozen train, validation, and test splits.", ["care_case_split_registry"], "Main text", "Experimental protocol")

    windows = table("care_window_farm_split_summary")
    if windows is None:
        windows = table("care_dataset_split_farm_summary")
    if windows is not None:
        split_col = next((c for c in ("model_split", "split") if c in windows.columns), None)
        window_col = next((c for c in ("window_count", "included_windows", "windows") if c in windows.columns), None)
        if _available(windows, "farm") and split_col and window_col:
            plot = windows[["farm", split_col, window_col]].copy()
            plot[window_col] = _numeric(plot[window_col])
            fig = _grouped_bar(plot, "farm", window_col, split_col, "Modeling windows by farm and split", "Windows", SPLIT_COLORS)
            ax = fig.axes[0]; ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"))
            save_figure(fig, "splits_windows", "modeling_windows_by_farm_split", "Modeling windows by farm and split", "Number of eligible multiscale windows in each farm/split combination.", ["care_window_farm_split_summary"], "Main text", "Experimental protocol")
        positive_fraction_col = next((c for c in windows.columns if "positive" in c.lower() and "fraction" in c.lower()), None)
        if _available(windows, "farm") and split_col and positive_fraction_col:
            fig = _grouped_bar(windows, "farm", positive_fraction_col, split_col, "Positive-window fraction", "Fraction", SPLIT_COLORS, percent=True)
            save_figure(fig, "splits_windows", "positive_window_fraction", "Positive-window fraction", "Fraction of event-positive target windows by farm and split.", ["care_window_farm_split_summary"], "Supplementary", "Experimental protocol")

    exclusions = table("care_window_exclusion_summary")
    if exclusions is not None and len(exclusions):
        reason_col = next((c for c in exclusions.columns if "reason" in c.lower() or "exclusion" in c.lower()), None)
        count_col = next((c for c in exclusions.columns if c.lower() in {"count", "windows", "excluded_windows"}), None)
        if reason_col and count_col:
            plot = exclusions.groupby(reason_col)[count_col].sum().sort_values(ascending=False).head(12)
            fig = _simple_bar(plot.index, plot.values, "Window exclusions by reason", "Excluded windows", horizontal=True, figsize=(8.0, 4.8))
            save_figure(fig, "splits_windows", "window_exclusions", "Window exclusions", "Leading reasons why structural candidate windows were excluded before modeling.", ["care_window_exclusion_summary"], "Supplementary", "Experimental protocol")

    feature_summary = table("care_window_feature_farm_summary")
    if feature_summary is not None and "farm" in feature_summary.columns:
        count_cols = [c for c in feature_summary.columns if pd.api.types.is_numeric_dtype(feature_summary[c]) and any(token in c.lower() for token in ("feature", "primary", "indicator"))]
        if count_cols:
            long = feature_summary.melt(id_vars="farm", value_vars=count_cols[:4], var_name="feature_set", value_name="count")
            long["feature_set"] = long["feature_set"].str.replace("_", " ").str.title()
            fig = _grouped_bar(long, "farm", "count", "feature_set", "Window feature dimensionality", "Features")
            save_figure(fig, "splits_windows", "window_feature_dimensionality", "Window feature dimensionality", "Primary and indicator feature widths used to construct multiscale windows.", ["care_window_feature_farm_summary"], "Supplementary", "Model inputs")

    prep = table("care_train_preprocessing_farm_summary")
    if prep is not None and "farm" in prep.columns:
        numeric_cols = [c for c in prep.columns if c != "farm" and pd.api.types.is_numeric_dtype(prep[c])]
        for column in [c for c in numeric_cols if any(token in c.lower() for token in ("signal", "feature", "case", "row", "window"))][:3]:
            values = _farm_numeric_values(
                prep,
                column,
                "care_train_preprocessing_farm_summary",
            )
            fig = _simple_bar([short_farm(x) for x in values.index], values.values, column.replace("_", " ").title(), column.replace("_", " "), [FARM_COLORS.get(x, "#777777") for x in values.index])
            save_figure(fig, "splits_windows", f"preprocessing_{column}", column.replace("_", " ").title(), f"Train-only preprocessing summary: {column.replace('_', ' ')}.", ["care_train_preprocessing_farm_summary"], "Supplementary", "Preprocessing")


make_dataset_figures()
make_quality_figures()
make_split_window_figures()


# =============================================================================
# 6. Training and calibration figures
# =============================================================================

def make_training_figures() -> None:
    history = table("care_model_training_history")
    summary = table("care_model_training_summary")
    if _available(history, "farm", "epoch", "train_loss", "normal_validation_loss"):
        work = history.copy()
        for column in ("epoch", "train_loss", "normal_validation_loss", "learning_rate"):
            if column in work: work[column] = _numeric(work[column])

        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharey=False)
        for ax, farm in zip(axes, FARM_ORDER):
            subset = work.loc[work["farm"].astype(str).eq(farm)].sort_values("epoch")
            ax.plot(subset["epoch"], subset["train_loss"], marker="o", markersize=3, label="Train", color="#4C78A8")
            ax.plot(subset["epoch"], subset["normal_validation_loss"], marker="s", markersize=3, label="Normal validation", color="#D55E00")
            ax.set_title(short_farm(farm), fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel("Masked Huber loss"); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); clean_axis(ax)
        axes[0].legend(frameon=False)
        fig.suptitle("Farm-specific training convergence", x=0.06, ha="left", fontweight="bold"); fig.tight_layout()
        save_figure(fig, "model_training", "training_convergence_all_farms", "Training convergence", "Training and normal-validation losses for the three farm-specific forecasters.", ["care_model_training_history"], "Main text", "Model training")

        for farm in FARM_ORDER:
            subset = work.loc[work["farm"].astype(str).eq(farm)].sort_values("epoch")
            if subset.empty: continue
            fig, ax = plt.subplots(figsize=(6.8, 3.9))
            ax.plot(subset["epoch"], subset["train_loss"], marker="o", label="Train", color="#4C78A8")
            ax.plot(subset["epoch"], subset["normal_validation_loss"], marker="s", label="Normal validation", color="#D55E00")
            best_i = subset["normal_validation_loss"].idxmin()
            ax.scatter([work.loc[best_i, "epoch"]], [work.loc[best_i, "normal_validation_loss"]], s=55, facecolor="white", edgecolor="#A33A2B", linewidth=1.4, zorder=4, label="Selected checkpoint")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Masked Huber loss"); ax.set_title(f"{short_farm(farm)} training history", loc="left", fontweight="bold"); ax.legend(frameon=False); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "model_training", f"{slugify(farm)}_training_history", f"{short_farm(farm)} training history", f"Training and normal-validation loss for {farm}; the selected checkpoint is marked.", ["care_model_training_history"], "Supplementary", "Model training")

        gap = work.copy(); gap["generalization_gap"] = gap["normal_validation_loss"] - gap["train_loss"]
        fig, ax = plt.subplots(figsize=(6.9, 3.9))
        for farm in FARM_ORDER:
            subset = gap.loc[gap["farm"].astype(str).eq(farm)].sort_values("epoch")
            ax.plot(subset["epoch"], subset["generalization_gap"], marker="o", markersize=3, label=short_farm(farm), color=FARM_COLORS[farm])
        ax.axhline(0, color="#555555", linewidth=0.8); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation − training loss"); ax.set_title("Generalization gap during training", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
        save_figure(fig, "model_training", "generalization_gap", "Generalization gap", "Difference between normal-validation and training loss across epochs.", ["care_model_training_history"], "Supplementary", "Model training")

        if "learning_rate" in work.columns:
            fig, ax = plt.subplots(figsize=(6.9, 3.8))
            for farm in FARM_ORDER:
                subset = work.loc[work["farm"].astype(str).eq(farm)].sort_values("epoch")
                ax.plot(subset["epoch"], subset["learning_rate"], marker="o", markersize=3, label=short_farm(farm), color=FARM_COLORS[farm])
            ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_title("Cosine learning-rate schedule", loc="left", fontweight="bold"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "model_training", "learning_rate_schedule", "Learning-rate schedule", "Learning-rate schedule used for all farm-specific models.", ["care_model_training_history"], "Supplementary", "Model training")

    if summary is not None and "farm" in summary.columns:
        for column, title, ylabel, priority in (
            ("parameter_count", "Model size by farm", "Trainable parameters", "Main text"),
            ("best_epoch", "Selected checkpoint epoch", "Epoch", "Supplementary"),
            ("best_validation_loss", "Best normal-validation loss", "Masked Huber loss", "Main text"),
            ("training_pool_windows", "Training-window pool", "Windows", "Supplementary"),
            ("normal_validation_control_windows", "Normal validation controls", "Windows", "Supplementary"),
        ):
            if column not in summary.columns: continue
            values = _farm_numeric_values(summary, column, "care_model_training_summary")
            fig = _simple_bar([short_farm(x) for x in values.index], values.values, title, ylabel, [FARM_COLORS.get(x, "#777777") for x in values.index])
            save_figure(fig, "model_training", column, title, f"{title} for each farm-specific forecaster.", ["care_model_training_summary"], priority, "Model training")


def make_calibration_figures() -> None:
    summary = table("care_anomaly_score_calibration_summary")
    regimes = table("care_operating_regime_calibration")
    components = table("care_normal_score_calibration_components")

    if summary is not None and "farm" in summary.columns:
        for column, title, ylabel, priority, percent in (
            ("global_anomaly_threshold", "Global conformal anomaly threshold", "Anomaly score", "Main text", False),
            ("calibration_exceedance_fraction_applied", "Realized normal-control exceedance rate", "Exceedance fraction", "Main text", True),
            ("reference_units", "Training reference controls", "Hourly controls", "Supplementary", False),
            ("calibration_units", "Validation calibration controls", "Hourly controls", "Supplementary", False),
            ("residual_location", "Residual robust location", "Residual location", "Supplementary", False),
            ("residual_scale", "Residual robust scale", "Residual scale", "Supplementary", False),
            ("latent_deviation_location", "Latent-deviation robust location", "Latent location", "Supplementary", False),
            ("latent_deviation_scale", "Latent-deviation robust scale", "Latent scale", "Supplementary", False),
        ):
            if column not in summary.columns: continue
            values = _farm_numeric_values(
                summary,
                column,
                "care_anomaly_score_calibration_summary",
            )
            fig = _simple_bar([short_farm(x) for x in values.index], values.values, title, ylabel, [FARM_COLORS.get(x, "#777777") for x in values.index], percent=percent)
            if column == "calibration_exceedance_fraction_applied":
                ax = fig.axes[0]; ax.axhline(NOMINAL_ALPHA, color="#A33A2B", linestyle="--", linewidth=1.2, label="Nominal 1%") ; ax.legend(frameon=False)
            save_figure(fig, "calibration", column, title, f"{title} estimated without anomaly-validation or test tensors.", ["care_anomaly_score_calibration_summary"], priority, "Score calibration")

    if regimes is not None and _available(regimes, "farm", "operating_regime_id"):
        if "anomaly_threshold" in regimes.columns:
            fig, ax = plt.subplots(figsize=(7.3, 4.0))
            for farm in FARM_ORDER:
                subset = regimes.loc[regimes["farm"].astype(str).eq(farm)].sort_values("operating_regime_id")
                ax.plot(_numeric(subset["operating_regime_id"]), _numeric(subset["anomaly_threshold"]), marker="o", label=short_farm(farm), color=FARM_COLORS[farm])
            ax.set_xlabel("Operating regime"); ax.set_ylabel("Anomaly threshold"); ax.set_title("Regime-specific conformal thresholds", loc="left", fontweight="bold"); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "calibration", "regime_specific_thresholds", "Regime-specific thresholds", "Conformal anomaly thresholds for four operating regimes within each farm.", ["care_operating_regime_calibration"], "Main text", "Score calibration")
        if "calibration_exceedance_fraction" in regimes.columns:
            fig, ax = plt.subplots(figsize=(7.3, 4.0))
            for farm in FARM_ORDER:
                subset = regimes.loc[regimes["farm"].astype(str).eq(farm)].sort_values("operating_regime_id")
                ax.plot(_numeric(subset["operating_regime_id"]), _numeric(subset["calibration_exceedance_fraction"]), marker="o", label=short_farm(farm), color=FARM_COLORS[farm])
            ax.axhline(NOMINAL_ALPHA, color="#A33A2B", linestyle="--", linewidth=1.2, label="Nominal 1%")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_xlabel("Operating regime"); ax.set_ylabel("Exceedance fraction"); ax.set_title("Calibration exceedance by regime", loc="left", fontweight="bold"); ax.legend(frameon=False, ncol=2); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "calibration", "regime_exceedance_fraction", "Regime exceedance fraction", "Observed conformal exceedance fractions within each operating regime.", ["care_operating_regime_calibration"], "Supplementary", "Score calibration")
        if "calibration_units" in regimes.columns:
            fig, ax = plt.subplots(figsize=(7.3, 4.0))
            for farm in FARM_ORDER:
                subset = regimes.loc[regimes["farm"].astype(str).eq(farm)].sort_values("operating_regime_id")
                ax.plot(_numeric(subset["operating_regime_id"]), _numeric(subset["calibration_units"]), marker="o", label=short_farm(farm), color=FARM_COLORS[farm])
            ax.set_xlabel("Operating regime"); ax.set_ylabel("Calibration controls"); ax.set_title("Calibration support per regime", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "calibration", "regime_calibration_support", "Calibration support per regime", "Number of wholly normal validation controls supporting each regime threshold.", ["care_operating_regime_calibration"], "Supplementary", "Score calibration")

    if components is not None and _available(components, "farm", "anomaly_score", "calibration_role"):
        for farm in FARM_ORDER:
            subset = components.loc[components["farm"].astype(str).eq(farm)].copy()
            if subset.empty: continue
            threshold = float(_numeric(subset["anomaly_threshold"]).median()) if "anomaly_threshold" in subset else None
            fig = _hist_overlay(subset, "anomaly_score", "calibration_role", f"{short_farm(farm)} normal-control score distribution", "Anomaly score", threshold)
            save_figure(fig, "calibration", f"{slugify(farm)}_normal_score_distribution", f"{short_farm(farm)} normal-control scores", "Training-reference and validation-calibration score distributions; the applied threshold is shown descriptively.", ["care_normal_score_calibration_components"], "Supplementary", "Score calibration")

            if _available(subset, "residual_robust_z", "latent_robust_z"):
                sample = subset.sample(min(len(subset), 6000), random_state=RANDOM_SEED)
                fig, ax = plt.subplots(figsize=(5.6, 4.6))
                groups = sample["calibration_role"].astype(str)
                for role, role_frame in sample.groupby(groups, sort=True):
                    ax.scatter(_numeric(role_frame["residual_robust_z"]), _numeric(role_frame["latent_robust_z"]), s=7, alpha=0.28, label=role.replace("_", " "))
                ax.axvline(0, color="#777777", linewidth=0.7); ax.axhline(0, color="#777777", linewidth=0.7)
                ax.set_xlabel("Residual robust z"); ax.set_ylabel("Latent-deviation robust z"); ax.set_title(f"{short_farm(farm)} score components", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
                save_figure(fig, "calibration", f"{slugify(farm)}_score_component_scatter", f"{short_farm(farm)} score components", "Relationship between robustly normalized residual and latent-deviation components in normal controls.", ["care_normal_score_calibration_components"], "Supplementary", "Score calibration")

            x, y = _ecdf(_numeric(subset["anomaly_score"]).dropna())
            fig, ax = plt.subplots(figsize=(6.4, 3.8)); ax.plot(x, y, color=FARM_COLORS[farm]);
            if threshold is not None: ax.axvline(threshold, color="#A33A2B", linestyle="--", label="Applied threshold")
            ax.set_xlabel("Anomaly score"); ax.set_ylabel("Empirical CDF"); ax.set_title(f"{short_farm(farm)} normal-score ECDF", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "calibration", f"{slugify(farm)}_normal_score_ecdf", f"{short_farm(farm)} normal-score ECDF", "Empirical cumulative distribution of normal-control anomaly scores.", ["care_normal_score_calibration_components"], "Supplementary", "Score calibration")

            if "operating_regime_id" in subset.columns:
                regime_counts = subset.groupby(["operating_regime_id", "calibration_role"], observed=False).size().rename("controls").reset_index()
                fig = _grouped_bar(regime_counts, "operating_regime_id", "controls", "calibration_role", f"{short_farm(farm)} operating-regime support", "Controls")
                save_figure(fig, "calibration", f"{slugify(farm)}_regime_support", f"{short_farm(farm)} regime support", "Reference and conformal-calibration control counts assigned to each operating regime.", ["care_normal_score_calibration_components"], "Supplementary", "Score calibration")


make_training_figures()
make_calibration_figures()


# =============================================================================
# 7. Validation/test case plots and frozen-policy aggregate figures
# =============================================================================

def _case_x_axis(case: pd.DataFrame) -> tuple[np.ndarray, str, bool]:
    is_anomaly = bool(_bool_series(case["case_is_anomaly"]).iloc[0]) if "case_is_anomaly" in case else False
    if is_anomaly and "hours_to_event_end" in case.columns:
        hours = _numeric(case["hours_to_event_end"])
        if hours.notna().sum() > 1:
            return -hours.to_numpy(dtype=float), "Hours relative to event end", True
    timestamps = pd.to_datetime(case["forecast_target_timestamp"], errors="coerce")
    if timestamps.notna().sum() > 1:
        elapsed = (timestamps - timestamps.min()).dt.total_seconds() / 3600.0
        return elapsed.to_numpy(dtype=float), "Elapsed hours", False
    return np.arange(len(case), dtype=float), "Window index", False


def _shade_boolean(ax: mpl.axes.Axes, x: np.ndarray, mask: np.ndarray, color: str, alpha: float, label: str | None = None) -> None:
    if len(x) != len(mask) or len(x) == 0: return
    mask = np.asarray(mask, dtype=bool)
    if not mask.any(): return
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    for i, (start, end) in enumerate(zip(starts, ends)):
        left = x[start]
        right = x[end]
        if right < left: left, right = right, left
        pad = max(1e-9, np.nanmedian(np.abs(np.diff(x))) / 2 if len(x) > 1 else 0.5)
        ax.axvspan(left - pad, right + pad, color=color, alpha=alpha, linewidth=0, label=label if i == 0 else None)


def make_case_trajectory(case: pd.DataFrame, split_label: str) -> mpl.figure.Figure:
    case = case.sort_values("forecast_target_timestamp", kind="stable").reset_index(drop=True)
    x, xlabel, relative_to_event = _case_x_axis(case)
    farm = str(case["farm"].iloc[0]); event_id = int(case["event_id"].iloc[0]); anomaly = bool(_bool_series(case["case_is_anomaly"]).iloc[0])
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.7), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0, 1.0]})

    score = _numeric(case["anomaly_score"]).to_numpy()
    threshold = _numeric(case["anomaly_threshold"]).to_numpy()
    axes[0].plot(x, score, color="#0072B2", label="Anomaly score")
    axes[0].plot(x, threshold, color="#A33A2B", linestyle="--", label="Frozen threshold")
    if "forecast_target_event_label" in case:
        _shade_boolean(axes[0], x, _bool_series(case["forecast_target_event_label"]).to_numpy(), "#D55E00", 0.10, "Fault interval")
    if "care_evaluation_normal_status" in case:
        _shade_boolean(axes[0], x, ~_bool_series(case["care_evaluation_normal_status"]).to_numpy(), "#777777", 0.13, "Non-actionable")
    axes[0].set_ylabel("Score"); axes[0].legend(frameon=False, ncol=4, loc="upper left"); clean_axis(axes[0])

    if "residual_robust_z" in case:
        axes[1].plot(x, _numeric(case["residual_robust_z"]), color="#0072B2", label="Residual z")
    if "latent_robust_z" in case:
        axes[1].plot(x, _numeric(case["latent_robust_z"]), color="#CC79A7", label="Latent z")
    axes[1].axhline(0, color="#777777", linewidth=0.7); axes[1].set_ylabel("Robust z"); axes[1].legend(frameon=False, ncol=2); clean_axis(axes[1])

    flags = _bool_series(case["raw_anomaly_prediction"]).to_numpy(dtype=float)
    axes[2].fill_between(x, 0, flags, step="mid", color="#D55E00", alpha=0.35, label="Raw flag")
    if "criticality" in case:
        axes[2].plot(x, _numeric(case["criticality"]), color="#6A3D9A", label="Criticality")
        threshold_col = "selected_criticality_threshold" if "selected_criticality_threshold" in case else "final_criticality_threshold"
        criticality_threshold = float(_numeric(case[threshold_col]).dropna().iloc[0]) if threshold_col in case and _numeric(case[threshold_col]).notna().any() else FROZEN_CRITICALITY
        axes[2].axhline(criticality_threshold, color="#A33A2B", linestyle="--", linewidth=1.1, label=f"Alarm threshold ({criticality_threshold:g})")
    axes[2].set_ylabel("Flag / criticality"); axes[2].set_xlabel(xlabel); axes[2].legend(frameon=False, ncol=3); clean_axis(axes[2])
    if relative_to_event:
        for ax in axes: ax.axvline(0, color="#222222", linewidth=1.0)
    label = "anomaly" if anomaly else "normal"
    fig.suptitle(f"{split_label.title()} — {short_farm(farm)} event {event_id} ({label})", x=0.06, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def make_case_component_scatter(case: pd.DataFrame, split_label: str) -> mpl.figure.Figure:
    farm = str(case["farm"].iloc[0]); event_id = int(case["event_id"].iloc[0]); anomaly = bool(_bool_series(case["case_is_anomaly"]).iloc[0])
    sample = case.sample(min(len(case), 10000), random_state=RANDOM_SEED)
    actionable = _bool_series(sample["care_evaluation_normal_status"]) if "care_evaluation_normal_status" in sample else pd.Series(True, index=sample.index)
    event_positive = _bool_series(sample["forecast_target_event_label"]) if "forecast_target_event_label" in sample else pd.Series(False, index=sample.index)
    state = np.select([~actionable, event_positive], ["Non-actionable", "Actionable fault"], default="Actionable non-fault")
    sample = sample.assign(plot_state=state)
    colors = {"Actionable non-fault": "#4C78A8", "Actionable fault": "#D55E00", "Non-actionable": "#7F8C8D"}
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    for state_name, subset in sample.groupby("plot_state", sort=False):
        ax.scatter(_numeric(subset["residual_robust_z"]), _numeric(subset["latent_robust_z"]), s=8, alpha=0.34, label=state_name, color=colors[state_name])
    ax.axvline(0, color="#777777", linewidth=0.7); ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel("Residual robust z"); ax.set_ylabel("Latent-deviation robust z")
    ax.set_title(f"{split_label.title()} {short_farm(farm)} event {event_id}\nscore-component geometry ({'anomaly' if anomaly else 'normal'})", loc="left", fontweight="bold")
    ax.legend(frameon=False, markerscale=1.6); clean_axis(ax); fig.tight_layout()
    return fig


def make_all_case_figures(split: str, components_name: str) -> None:
    components = table(components_name)
    required = ("farm", "event_id", "case_is_anomaly", "forecast_target_timestamp", "anomaly_score", "anomaly_threshold", "raw_anomaly_prediction", "residual_robust_z", "latent_robust_z")
    if not _available(components, *required):
        skip_plot(f"{split}_case_figures", "Required score-component columns unavailable")
        return
    for (farm, event_id), case in components.groupby(["farm", "event_id"], sort=True, observed=False):
        anomaly = bool(_bool_series(case["case_is_anomaly"]).iloc[0])
        role = "anomaly" if anomaly else "normal"
        title = f"{split.title()} {short_farm(farm)} event {int(event_id)} trajectory"
        fig = make_case_trajectory(case, split)
        save_figure(
            fig,
            "validation" if split == "validation" else "test",
            f"{split}_{slugify(farm)}_event_{int(event_id)}_{role}_trajectory",
            title,
            f"Frozen-score and criticality trajectory for the {split} {role} case {farm}, event {int(event_id)}. Shading distinguishes actionable fault intervals and non-actionable operation.",
            [components_name],
            "Supplementary",
            f"Per-case {split} diagnostics",
        )
        fig = make_case_component_scatter(case, split)
        save_figure(
            fig,
            "validation" if split == "validation" else "test",
            f"{split}_{slugify(farm)}_event_{int(event_id)}_{role}_components",
            f"{split.title()} {short_farm(farm)} event {int(event_id)} component geometry",
            f"Residual-versus-latent score geometry for the {split} {role} case {farm}, event {int(event_id)}. This is descriptive and does not alter thresholds.",
            [components_name],
            "Supplementary",
            f"Per-case {split} diagnostics",
        )


def _event_confusion(metrics: pd.DataFrame) -> np.ndarray:
    truth = _bool_series(metrics["case_is_anomaly"]).to_numpy()
    pred = _bool_series(metrics["event_alarm"]).to_numpy()
    tn = int((~truth & ~pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum()); tp = int((truth & pred).sum())
    return np.asarray([[tn, fp], [fn, tp]], dtype=int)


def make_validation_figures() -> None:
    grid = table("care_validation_policy_grid")
    metrics = table("care_validation_event_metrics")
    summary = table("care_validation_care_summary")
    if _available(grid, "criticality_threshold", "care_score"):
        fig, ax = plt.subplots(figsize=(7.0, 3.9))
        ax.plot(_numeric(grid["criticality_threshold"]), _numeric(grid["care_score"]), marker="o", color="#6A3D9A")
        ax.axvline(FROZEN_CRITICALITY, color="#A33A2B", linestyle="--", label="Selected/published 72")
        ax.set_xlabel("Criticality threshold"); ax.set_ylabel("Validation CARE"); ax.set_title("Validation-only sequential-policy grid", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
        save_figure(fig, "validation", "validation_policy_grid_care", "Validation policy grid", "Validation CARE over the prespecified criticality grid. Equal scores are retained; 72 is the published-policy tie-break.", ["care_validation_policy_grid"], "Main text", "Validation selection")
        if "event_reliability_f_beta" in grid.columns:
            fig, ax = plt.subplots(figsize=(7.0, 3.9))
            ax.plot(_numeric(grid["criticality_threshold"]), _numeric(grid["event_reliability_f_beta"]), marker="o", color="#D55E00", label="Event reliability")
            ax.plot(_numeric(grid["criticality_threshold"]), _numeric(grid["mean_normal_accuracy"]), marker="s", color="#009E73", label="Normal accuracy")
            ax.axvline(FROZEN_CRITICALITY, color="#A33A2B", linestyle="--")
            ax.set_xlabel("Criticality threshold"); ax.set_ylabel("Metric"); ax.set_ylim(-0.03, 1.03); ax.set_title("Validation policy components", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "validation", "validation_policy_components", "Validation policy components", "Event reliability and normal-case accuracy across validation-only candidate thresholds.", ["care_validation_policy_grid"], "Supplementary", "Validation selection")
        if "detected_anomaly_cases" in grid.columns:
            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            ax.step(_numeric(grid["criticality_threshold"]), _numeric(grid["detected_anomaly_cases"]), where="mid", color="#D55E00", label="Detected anomaly cases")
            ax.step(_numeric(grid["criticality_threshold"]), _numeric(grid["event_alarms"]), where="mid", color="#6A3D9A", label="All event alarms")
            ax.axvline(FROZEN_CRITICALITY, color="#A33A2B", linestyle="--"); ax.set_xlabel("Criticality threshold"); ax.set_ylabel("Cases"); ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.set_title("Validation alarm counts", loc="left", fontweight="bold"); ax.legend(frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "validation", "validation_alarm_counts", "Validation alarm counts", "Detected anomaly cases and total event alarms over the validation-only grid.", ["care_validation_policy_grid"], "Supplementary", "Validation selection")

    if metrics is not None and _available(metrics, "case_is_anomaly", "event_alarm"):
        matrix = _event_confusion(metrics)
        fig = _heatmap(matrix, ["Normal", "Alarm"], ["Normal", "Anomaly"], "Validation event confusion")
        save_figure(fig, "validation", "validation_event_confusion", "Validation event confusion", "Event-level confusion matrix under the selected 72-point validation policy.", ["care_validation_event_metrics"], "Main text", "Validation results")
        if "maximum_criticality" in metrics.columns:
            plot = metrics.copy(); plot["Class"] = _bool_series(plot["case_is_anomaly"]).map({True: "Anomaly", False: "Normal"}); plot["Case"] = plot["farm"].map(short_farm) + "-" + plot["event_id"].astype(str)
            plot = plot.sort_values("maximum_criticality", ascending=True)
            fig = _simple_bar(plot["Case"], _numeric(plot["maximum_criticality"]), "Validation maximum criticality by case", "Maximum criticality", ["#D55E00" if x == "Anomaly" else "#4C78A8" for x in plot["Class"]], horizontal=True, figsize=(7.4, 5.4))
            ax = fig.axes[0]; ax.axvline(FROZEN_CRITICALITY, color="#A33A2B", linestyle="--", label="Selected 72"); ax.legend(frameon=False)
            save_figure(fig, "validation", "validation_maximum_criticality", "Validation maximum criticality", "Maximum accumulated criticality for every validation case; colors distinguish anomaly and normal cases.", ["care_validation_event_metrics"], "Main text", "Validation results")

    if summary is not None and _available(summary, "coverage", "accuracy", "reliability", "earliness", "care_score"):
        selected = summary.loc[summary["policy"].astype(str).eq("validation_selected")].iloc[0] if "policy" in summary and summary["policy"].astype(str).eq("validation_selected").any() else summary.iloc[0]
        labels = ["Coverage", "Accuracy", "Reliability", "Earliness", "CARE"]
        values = [float(selected[c]) for c in ("coverage", "accuracy", "reliability", "earliness", "care_score")]
        fig = _simple_bar(labels, values, "Selected validation CARE components", "Score", [CARE_COLORS[c] for c in ("coverage", "accuracy", "reliability", "earliness", "care_score")], percent=True)
        save_figure(fig, "validation", "validation_care_components", "Validation CARE components", "Coverage, accuracy, reliability, earliness, and composite CARE under the selected validation policy.", ["care_validation_care_summary"], "Main text", "Validation results")


def make_test_figures() -> None:
    metrics = table("care_test_event_metrics")
    farm = table("care_test_farm_summary")
    comparison = table("care_validation_test_care_comparison")
    if metrics is not None and _available(metrics, "case_is_anomaly", "event_alarm"):
        matrix = _event_confusion(metrics)
        fig = _heatmap(matrix, ["Normal", "Alarm"], ["Normal", "Anomaly"], "Locked test event confusion")
        save_figure(fig, "test", "test_event_confusion", "Locked test event confusion", "Event-level confusion matrix under the frozen 72-point policy: no alternative test grid was evaluated.", ["care_test_event_metrics"], "Main text", "Final test results")

        if "maximum_criticality" in metrics.columns:
            plot = metrics.copy(); plot["Class"] = _bool_series(plot["case_is_anomaly"]).map({True: "Anomaly", False: "Normal"}); plot["Case"] = plot["farm"].map(short_farm) + "-" + plot["event_id"].astype(str)
            plot = plot.sort_values("maximum_criticality", ascending=True)
            fig = _simple_bar(plot["Case"], _numeric(plot["maximum_criticality"]), "Locked test maximum criticality by case", "Maximum criticality", ["#D55E00" if x == "Anomaly" else "#4C78A8" for x in plot["Class"]], horizontal=True, figsize=(7.4, 5.4))
            ax = fig.axes[0]; ax.axvline(FROZEN_CRITICALITY, color="#A33A2B", linestyle="--", label="Frozen 72"); ax.legend(frameon=False)
            save_figure(fig, "test", "test_maximum_criticality", "Locked test maximum criticality", "Maximum criticality reached by every test case under the frozen score and policy.", ["care_test_event_metrics"], "Main text", "Final test results")

        anomaly = metrics.loc[_bool_series(metrics["case_is_anomaly"])].copy()
        if not anomaly.empty and _available(anomaly, "true_positive", "positive_event_windows"):
            anomaly["point_detection_rate"] = _numeric(anomaly["true_positive"]) / _numeric(anomaly["positive_event_windows"]).replace(0, np.nan)
            anomaly["Case"] = anomaly["farm"].map(short_farm) + "-" + anomaly["event_id"].astype(str)
            fig = _simple_bar(anomaly["Case"], anomaly["point_detection_rate"], "Actionable fault-window detection rate", "Detected fraction", [FARM_COLORS.get(str(x), "#777777") for x in anomaly["farm"]], horizontal=True, percent=True, figsize=(7.2, 4.4))
            save_figure(fig, "test", "test_actionable_fault_detection_rate", "Actionable fault-window detection", "Fraction of actionable positive fault windows flagged in each locked test anomaly case.", ["care_test_event_metrics"], "Main text", "Failure analysis")

        normal = metrics.loc[~_bool_series(metrics["case_is_anomaly"])].copy()
        if not normal.empty and _available(normal, "false_positive", "care_evaluation_windows"):
            normal["false_positive_rate"] = _numeric(normal["false_positive"]) / _numeric(normal["care_evaluation_windows"]).replace(0, np.nan)
            normal["Case"] = normal["farm"].map(short_farm) + "-" + normal["event_id"].astype(str)
            fig = _simple_bar(normal["Case"], normal["false_positive_rate"], "Normal-case actionable false-positive rate", "False-positive fraction", [FARM_COLORS.get(str(x), "#777777") for x in normal["farm"]], horizontal=True, percent=True, figsize=(7.2, 4.4))
            save_figure(fig, "test", "test_normal_false_positive_rate", "Normal-case false-positive rate", "Actionable-window false-positive fraction in each locked normal test case.", ["care_test_event_metrics"], "Supplementary", "Final test results")

    if farm is not None and _available(farm, "scope", "coverage", "accuracy", "reliability", "earliness", "care_score"):
        long = farm.melt(id_vars="scope", value_vars=["coverage", "accuracy", "reliability", "earliness", "care_score"], var_name="metric", value_name="score")
        fig = _grouped_bar(long, "scope", "score", "metric", "Locked test CARE components by farm", "Score")
        ax = fig.axes[0]; ax.set_ylim(0, 1.05)
        save_figure(fig, "test", "test_care_components_by_farm", "Test CARE components by farm", "Farm-level CARE components under the frozen score and sequential policy.", ["care_test_farm_summary"], "Main text", "Final test results")

    if comparison is not None and _available(comparison, "split", "coverage", "accuracy", "reliability", "earliness", "care_score"):
        long = comparison.melt(id_vars="split", value_vars=["coverage", "accuracy", "reliability", "earliness", "care_score"], var_name="metric", value_name="score")
        fig = _grouped_bar(long, "metric", "score", "split", "Validation-to-test CARE comparison", "Score", {"validation": "#E69F00", "test": "#D55E00"})
        ax = fig.axes[0]; ax.set_ylim(0, 1.05)
        save_figure(fig, "test", "validation_test_care_comparison", "Validation-to-test CARE comparison", "CARE components for validation selection and the locked final test under the same 72-point policy.", ["care_validation_test_care_comparison"], "Main text", "Final test results")


make_all_case_figures("validation", "care_validation_anomaly_score_components")
make_validation_figures()
make_all_case_figures("test", "care_test_anomaly_score_components")
make_test_figures()


# =============================================================================
# 8. Failure-analysis figures — descriptive only, never a test-policy sweep
# =============================================================================

def make_failure_figures() -> None:
    components = table("care_test_anomaly_score_components")
    metrics = table("care_test_event_metrics")
    comparison = table("care_validation_test_care_comparison")
    if components is not None and _available(components, "farm", "event_id", "case_is_anomaly", "raw_anomaly_prediction", "care_evaluation_normal_status"):
        work = components.copy()
        work["is_flag"] = _bool_series(work["raw_anomaly_prediction"])
        work["actionable"] = _bool_series(work["care_evaluation_normal_status"])
        work["anomaly_case"] = _bool_series(work["case_is_anomaly"])
        work["event_positive"] = _bool_series(work["forecast_target_event_label"]) if "forecast_target_event_label" in work else False
        work["Case"] = work["farm"].map(short_farm) + "-" + work["event_id"].astype(str)

        anomaly = work.loc[work["anomaly_case"]].copy()
        records = []
        for (farm_name, event_id), case in anomaly.groupby(["farm", "event_id"], sort=True, observed=False):
            flags = case["is_flag"]
            actionable = case["actionable"]
            event_positive = case["event_positive"]
            records.append(
                {
                    "farm": farm_name,
                    "event_id": int(event_id),
                    "Case": f"{short_farm(farm_name)}-{int(event_id)}",
                    "actionable_fault_flags": int((flags & actionable & event_positive).sum()),
                    "actionable_nonfault_flags": int((flags & actionable & ~event_positive).sum()),
                    "masked_out_flags": int((flags & ~actionable).sum()),
                    "actionable_fault_windows": int((actionable & event_positive).sum()),
                    "masked_out_windows": int((~actionable).sum()),
                }
            )
        decomposition = pd.DataFrame(records)
        if len(decomposition):
            plot = decomposition.set_index("Case")[["actionable_fault_flags", "actionable_nonfault_flags", "masked_out_flags"]]
            fig, ax = plt.subplots(figsize=(8.2, 4.7)); plot.plot(kind="bar", stacked=True, color=["#D55E00", "#E69F00", "#7F8C8D"], ax=ax)
            ax.set_ylabel("Raw score exceedances"); ax.set_xlabel(""); ax.set_title("Where anomaly-case score exceedances occurred", loc="left", fontweight="bold"); ax.legend(["Actionable fault", "Actionable non-fault", "Masked out"], frameon=False); clean_axis(ax); fig.tight_layout()
            save_figure(fig, "failure_analysis", "test_flag_location_decomposition", "Location of test anomaly-case flags", "Raw anomaly-score exceedances decomposed into actionable fault, actionable non-fault, and CARE-masked windows.", ["care_test_anomaly_score_components"], "Main text", "Failure analysis")

            rates = decomposition.copy()
            rates["actionable_fault_rate"] = rates["actionable_fault_flags"] / rates["actionable_fault_windows"].replace(0, np.nan)
            rates["masked_out_rate"] = rates["masked_out_flags"] / rates["masked_out_windows"].replace(0, np.nan)
            long = rates.melt(id_vars="Case", value_vars=["actionable_fault_rate", "masked_out_rate"], var_name="window_type", value_name="flag_rate")
            long["window_type"] = long["window_type"].map({"actionable_fault_rate": "Actionable fault", "masked_out_rate": "Masked out"})
            fig = _grouped_bar(long, "Case", "flag_rate", "window_type", "Flag rate: actionable fault vs masked-out windows", "Flag fraction", {"Actionable fault": "#D55E00", "Masked out": "#7F8C8D"}, percent=True)
            save_figure(fig, "failure_analysis", "actionable_vs_masked_flag_rate", "Actionable versus masked flag rate", "For each anomaly test case, the score-exceedance rate inside actionable fault windows is contrasted with masked-out operation.", ["care_test_anomaly_score_components"], "Main text", "Failure analysis")

        # Farm/class/actionability rates
        farm_records = []
        for (farm_name, anomaly_case, actionable), subset in work.groupby(["farm", "anomaly_case", "actionable"], sort=True, observed=False):
            farm_records.append({"farm": farm_name, "case_class": "Anomaly" if anomaly_case else "Normal", "status": "Actionable" if actionable else "Masked out", "flag_rate": float(subset["is_flag"].mean()), "windows": int(len(subset))})
        farm_rates = pd.DataFrame(farm_records)
        for status in ("Actionable", "Masked out"):
            subset = farm_rates.loc[farm_rates["status"].eq(status)]
            if subset.empty: continue
            fig = _grouped_bar(subset, "farm", "flag_rate", "case_class", f"{status} test flag rate by farm and case class", "Flag fraction", {"Normal": "#4C78A8", "Anomaly": "#D55E00"}, percent=True)
            save_figure(fig, "failure_analysis", f"{slugify(status)}_farm_class_flag_rate", f"{status} flag rate", f"Raw anomaly-score exceedance rates in {status.lower()} windows, stratified by farm and case class.", ["care_test_anomaly_score_components"], "Main text" if status == "Actionable" else "Supplementary", "Failure analysis")

        # Score-to-threshold margin by status and farm
        if _available(work, "anomaly_score", "anomaly_threshold"):
            work["score_margin"] = _numeric(work["anomaly_score"]) - _numeric(work["anomaly_threshold"])
            sample = work.sample(min(len(work), 25000), random_state=RANDOM_SEED)
            sample["status_class"] = np.where(sample["actionable"], "Actionable", "Masked out") + " / " + np.where(sample["anomaly_case"], "anomaly case", "normal case")
            for farm_name in FARM_ORDER:
                subset = sample.loc[sample["farm"].astype(str).eq(farm_name)]
                if subset.empty: continue
                fig, ax = plt.subplots(figsize=(7.0, 4.0))
                if sns is not None:
                    sns.boxenplot(data=subset, x="status_class", y="score_margin", color=FARM_COLORS[farm_name], showfliers=False, ax=ax)
                ax.axhline(0, color="#A33A2B", linestyle="--", linewidth=1.0); ax.set_xlabel(""); ax.set_ylabel("Score − frozen threshold"); ax.set_title(f"{short_farm(farm_name)} score margin by state", loc="left", fontweight="bold"); ax.tick_params(axis="x", rotation=20); clean_axis(ax); fig.tight_layout()
                save_figure(fig, "failure_analysis", f"{slugify(farm_name)}_score_margin_by_state", f"{short_farm(farm_name)} score margin", "Distribution of score minus frozen threshold by actionability and case class; positive values are raw flags.", ["care_test_anomaly_score_components"], "Supplementary", "Failure analysis")

    if metrics is not None and _available(metrics, "farm", "event_id", "case_is_anomaly", "maximum_criticality"):
        plot = metrics.copy(); plot["Class"] = _bool_series(plot["case_is_anomaly"]).map({True: "Anomaly", False: "Normal"})
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if sns is not None:
            sns.stripplot(data=plot, x="Class", y="maximum_criticality", hue="farm", palette=FARM_COLORS, size=7, jitter=0.12, ax=ax)
        ax.axhline(FROZEN_CRITICALITY, color="#A33A2B", linestyle="--", label="Frozen 72"); ax.set_ylabel("Maximum criticality"); ax.set_xlabel(""); ax.set_title("Criticality lacks class separation", loc="left", fontweight="bold"); ax.legend(frameon=False, ncol=2); clean_axis(ax); fig.tight_layout()
        save_figure(fig, "failure_analysis", "criticality_class_overlap", "Criticality class overlap", "Maximum criticality in locked normal and anomaly test cases. Several normal cases exceed all anomaly cases, while none approach 72.", ["care_test_event_metrics"], "Main text", "Failure analysis")

        if _available(metrics, "true_positive", "false_positive", "false_negative", "true_negative"):
            point_matrix = np.asarray([[int(_numeric(metrics["true_negative"]).sum()), int(_numeric(metrics["false_positive"]).sum())], [int(_numeric(metrics["false_negative"]).sum()), int(_numeric(metrics["true_positive"]).sum())]])
            fig = _heatmap(point_matrix, ["Normal", "Flag"], ["Normal", "Fault"], "Locked test point confusion")
            save_figure(fig, "failure_analysis", "test_point_confusion", "Locked test point confusion", "Point-level confusion counts within the official CARE evaluation mask.", ["care_test_event_metrics"], "Supplementary", "Failure analysis")
            normalized = point_matrix / np.maximum(point_matrix.sum(axis=1, keepdims=True), 1)
            fig = _heatmap(normalized, ["Normal", "Flag"], ["Normal", "Fault"], "Locked test point confusion (row-normalized)", fmt=".3f", cmap="Oranges")
            save_figure(fig, "failure_analysis", "test_point_confusion_normalized", "Normalized test point confusion", "Row-normalized point-level confusion matrix within the CARE evaluation mask.", ["care_test_event_metrics"], "Main text", "Failure analysis")

    if comparison is not None and _available(comparison, "split", "coverage", "accuracy", "reliability", "earliness", "care_score"):
        indexed = comparison.set_index("split")
        if {"validation", "test"}.issubset(indexed.index):
            metric_names = ["coverage", "accuracy", "reliability", "earliness", "care_score"]
            change = indexed.loc["test", metric_names] - indexed.loc["validation", metric_names]
            colors = ["#009E73" if x >= 0 else "#D55E00" for x in change]
            fig = _simple_bar([x.title() for x in metric_names], change.values, "Absolute validation-to-test metric change", "Test − validation", colors)
            ax = fig.axes[0]; ax.axhline(0, color="#333333", linewidth=0.8)
            save_figure(fig, "failure_analysis", "validation_test_metric_change", "Validation-to-test change", "Absolute change in CARE components from validation selection to locked final test.", ["care_validation_test_care_comparison"], "Main text", "Failure analysis")


make_failure_figures()


# =============================================================================
# 9. Publication metric tables and table images
# =============================================================================

def _format_table_value(value: Any) -> str:
    if pd.isna(value): return "—"
    if isinstance(value, (bool, np.bool_)): return "Yes" if bool(value) else "No"
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) < 1e-12: return "0"
        if abs(float(value)) < 1: return f"{float(value):.4f}"
        return f"{float(value):,.3f}"
    if isinstance(value, (int, np.integer)): return f"{int(value):,}"
    return str(value)


def export_paper_table(
    frame: pd.DataFrame,
    slug: str,
    title: str,
    caption: str,
    source_tables: Sequence[str],
    priority: str,
    columns: Sequence[str] | None = None,
    max_rows: int = 30,
) -> None:
    if frame is None or frame.empty: return
    work = frame.copy()
    if columns is not None:
        selected = [c for c in columns if c in work.columns]
        if not selected: return
        work = work[selected]
    work = work.head(max_rows).reset_index(drop=True)
    table_number = len(TABLE_ROWS) + 1
    table_id = f"T{table_number:02d}"
    folder = PAPER_RESULTS_ROOT / SECTION_DIRS["tables"]
    base = folder / f"{table_id}_{slugify(slug)}"
    work.to_csv(base.with_suffix(".csv"), index=False)
    # Build a dependency-free LaTeX tabular. Recent pandas versions route
    # DataFrame.to_latex through Jinja2, which is optional and may not be
    # installed in the research notebook environment.
    def latex_escape(value: Any) -> str:
        text = _format_table_value(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in text)

    alignment = "l" + "r" * max(0, len(work.columns) - 1)
    latex_lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    latex_lines.append(" & ".join(latex_escape(c) for c in work.columns) + r" \\")
    latex_lines.append(r"\midrule")
    for row_values in work.itertuples(index=False, name=None):
        latex_lines.append(" & ".join(latex_escape(v) for v in row_values) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex = "\n".join(latex_lines) + "\n"
    base.with_suffix(".tex").write_text(latex, encoding="utf-8")

    display_frame = work.copy()
    display_frame.columns = [str(c).replace("_", " ").title() for c in display_frame.columns]
    display_values = [[_format_table_value(v) for v in row] for row in display_frame.to_numpy()]
    width = max(7.0, min(18.0, 1.20 * len(display_frame.columns)))
    height = max(2.3, min(13.0, 0.38 * (len(display_frame) + 3)))
    fig, ax = plt.subplots(figsize=(width, height)); ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    tab = ax.table(cellText=display_values, colLabels=list(display_frame.columns), loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(7.2); tab.scale(1.0, 1.25)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#D9D9D9"); cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#E9EFF5"); cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F8F8")
    fig.text(0.01, 0.01, caption, fontsize=7.2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(base.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    TABLE_ROWS.append({"table_id": table_id, "title": title, "caption": caption, "paper_priority": priority, "csv_path": str(base.with_suffix(".csv")), "latex_path": str(base.with_suffix(".tex")), "png_path": str(base.with_suffix(".png")), "pdf_path": str(base.with_suffix(".pdf")), "source_tables": _source_string(source_tables)})


cases = table("care_case_registry")
if cases is not None and _available(cases, "farm", "is_anomaly"):
    tmp = cases.copy(); tmp["is_anomaly"] = _bool_series(tmp["is_anomaly"])
    dataset_table = tmp.groupby("farm").agg(cases=("event_id", "size"), anomaly_cases=("is_anomaly", "sum"), assets=("dataset_asset_id", "nunique"), primary_signals=("number_of_primary_signals", "max")).reset_index()
    dataset_table["normal_cases"] = dataset_table["cases"] - dataset_table["anomaly_cases"]
    export_paper_table(dataset_table, "dataset_summary", "Dataset summary", "Case, class, asset, and primary-signal counts for each wind farm.", ["care_case_registry"], "Main text", ["farm", "cases", "normal_cases", "anomaly_cases", "assets", "primary_signals"])

for name, slug, title, caption, priority, cols, max_rows in (
    ("care_split_farm_class_summary", "frozen_split_summary", "Frozen split summary", "Farm- and class-stratified case allocation used by the experiment.", "Main text", None, 30),
    ("care_window_farm_split_summary", "window_summary", "Multiscale window summary", "Eligible multiscale windows by farm and frozen split.", "Supplementary", None, 30),
    ("care_model_training_summary", "model_training_summary", "Model training summary", "Selected checkpoint and model-size summary for each farm-specific forecaster.", "Main text", ["farm", "parameter_count", "epochs_completed", "best_epoch", "best_validation_loss", "training_pool_windows", "normal_validation_control_windows"], 10),
    ("care_anomaly_score_calibration_summary", "score_calibration_summary", "Normal-control score calibration", "Reference/calibration support, thresholds, and realized normal exceedance rates.", "Main text", ["farm", "reference_units", "calibration_units", "operating_regimes", "global_anomaly_threshold", "regime_specific_thresholds", "calibration_exceedance_fraction_applied", "nominal_alpha"], 10),
    ("care_operating_regime_calibration", "regime_calibration", "Operating-regime calibration", "Per-regime conformal support and frozen anomaly thresholds.", "Supplementary", ["farm", "operating_regime_id", "reference_units", "calibration_units", "threshold_source", "anomaly_threshold", "calibration_exceedance_fraction"], 30),
    ("care_validation_policy_grid", "validation_policy_grid", "Validation-only policy grid", "Prespecified validation policy grid; no analogous grid was evaluated on test.", "Supplementary", None, 30),
    ("care_validation_care_summary", "validation_care_summary", "Validation CARE summary", "Selected and published-policy validation CARE components.", "Main text", None, 10),
    ("care_validation_event_metrics", "validation_event_metrics", "Validation event results", "Per-case validation results under the selected 72-point policy.", "Supplementary", ["farm", "event_id", "event_type", "case_is_anomaly", "coverage_f_beta", "earliness_weighted_score", "maximum_criticality", "event_alarm"], 30),
    ("care_test_care_summary", "final_test_summary", "Locked final test summary", "Pooled final test results under the frozen score and 72-point policy.", "Main text", None, 10),
    ("care_test_farm_summary", "test_farm_summary", "Locked test results by farm", "Farm-level final test CARE components and confusion counts.", "Main text", ["scope", "case_count", "coverage", "accuracy", "reliability", "earliness", "care_score", "event_true_positive", "event_false_positive", "event_false_negative", "event_true_negative"], 10),
    ("care_test_event_metrics", "test_event_metrics", "Locked test event results", "Per-case results from the single-use final test evaluation.", "Main text", ["farm", "event_id", "event_type", "case_is_anomaly", "true_positive", "false_positive", "false_negative", "coverage_f_beta", "maximum_criticality", "event_alarm"], 30),
    ("care_validation_test_care_comparison", "validation_test_comparison", "Validation–test comparison", "Validation and locked-test CARE components under the same frozen sequential policy.", "Main text", None, 10),
    ("care_final_evaluation_constraint_audit", "final_constraint_audit", "Final evaluation constraints", "Audit evidence that the final test evaluation remained single-use and frozen.", "Supplementary", None, 50),
):
    frame = table(name)
    if frame is not None:
        export_paper_table(frame, slug, title, caption, [name], priority, cols, max_rows)


# =============================================================================
# 10. Metrics, catalogs, contact sheets, and completion guards
# =============================================================================

comparison = table("care_validation_test_care_comparison")
if comparison is not None:
    for row in comparison.itertuples(index=False):
        for metric in ("coverage", "accuracy", "reliability", "earliness", "care_score"):
            if hasattr(row, metric): add_metric("CARE", metric, float(getattr(row, metric)), "score", str(getattr(row, "split", "")), "care_validation_test_care_comparison")

test_events = table("care_test_event_metrics")
if test_events is not None:
    anomaly_mask = _bool_series(test_events["case_is_anomaly"])
    add_metric("Final test", "anomaly cases", int(anomaly_mask.sum()), "cases", "pooled", "care_test_event_metrics")
    add_metric("Final test", "normal cases", int((~anomaly_mask).sum()), "cases", "pooled", "care_test_event_metrics")
    add_metric("Final test", "detected anomaly cases", int((_bool_series(test_events["event_alarm"]) & anomaly_mask).sum()), "cases", "pooled", "care_test_event_metrics")
    if "true_positive" in test_events:
        add_metric("Final test", "actionable fault windows flagged", int(_numeric(test_events.loc[anomaly_mask, "true_positive"]).sum()), "windows", "pooled", "care_test_event_metrics")
    if "positive_event_windows" in test_events:
        add_metric("Final test", "actionable fault windows", int(_numeric(test_events.loc[anomaly_mask, "positive_event_windows"]).sum()), "windows", "pooled", "care_test_event_metrics")


figure_manifest = pd.DataFrame(FIGURE_ROWS)
table_manifest = pd.DataFrame(TABLE_ROWS)
metric_table = pd.DataFrame(METRIC_ROWS)
skipped_table = pd.DataFrame(SKIPPED_ROWS)

catalog_dir = PAPER_RESULTS_ROOT / SECTION_DIRS["catalog"]
figure_manifest.to_csv(catalog_dir / "figure_manifest.csv", index=False)
table_manifest.to_csv(catalog_dir / "table_manifest.csv", index=False)
metric_table.to_csv(catalog_dir / "paper_metrics_long.csv", index=False)
skipped_table.to_csv(catalog_dir / "skipped_optional_items.csv", index=False)

captions_text = "\n\n".join(
    f"{row['figure_id']} — {row['title']}\n{row['caption']}"
    for row in FIGURE_ROWS
)
(catalog_dir / "figure_captions.txt").write_text(captions_text, encoding="utf-8")


def make_contact_sheets() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    for section_key, directory_name in SECTION_DIRS.items():
        if section_key in {"catalog", "tables"}: continue
        rows = [row for row in FIGURE_ROWS if Path(row["png_path"]).parent.name == directory_name]
        if not rows: continue
        page_size = 12
        for page_i in range(math.ceil(len(rows) / page_size)):
            page_rows = rows[page_i * page_size:(page_i + 1) * page_size]
            canvas = Image.new("RGB", (1800, 2400), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((55, 30), f"{directory_name.replace('_', ' ').title()} — contact sheet {page_i + 1}", fill="#222222")
            for i, row in enumerate(page_rows):
                r, c = divmod(i, 3)
                x0, y0 = 40 + c * 590, 95 + r * 560
                try:
                    image = Image.open(row["png_path"]).convert("RGB")
                    image.thumbnail((540, 455))
                    canvas.paste(image, (x0, y0 + 34))
                    draw.text((x0, y0), f"{row['figure_id']} {row['title'][:62]}", fill="#222222")
                except Exception:
                    continue
            canvas.save(catalog_dir / f"contact_sheet_{directory_name}_{page_i + 1:02d}.jpg", quality=90)


make_contact_sheets()

source_inventory = pd.DataFrame(
    [
        {"table_name": name, "rows": len(item.frame), "columns": len(item.frame.columns), "source": item.source}
        for name, item in sorted(TABLES.items())
    ]
)
source_inventory.to_csv(catalog_dir / "loaded_source_inventory.csv", index=False)

plot_aggregation_notes = pd.DataFrame(PLOT_AGGREGATION_ROWS).drop_duplicates()
if plot_aggregation_notes.empty:
    plot_aggregation_notes = pd.DataFrame(
        columns=(
            "source_table",
            "column",
            "duplicate_farms",
            "plot_only_aggregation",
            "source_modified",
        )
    )
plot_aggregation_notes.to_csv(catalog_dir / "plot_aggregation_notes.csv", index=False)

result_summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "paper_results_root": str(PAPER_RESULTS_ROOT),
    "care_source_root": str(CARE_OUTPUT_ROOT),
    "figures": int(len(FIGURE_ROWS)),
    "paper_tables": int(len(TABLE_ROWS)),
    "loaded_source_tables": int(len(TABLES)),
    "missing_core_tables": missing_core,
    "optional_load_warnings": LOAD_WARNINGS,
    "skipped_optional_items": SKIPPED_ROWS,
    "plot_only_duplicate_aggregations": int(len(plot_aggregation_notes)),
    "frozen_criticality": FROZEN_CRITICALITY,
    "test_policy_grid_generated": False,
    "model_or_threshold_refit": False,
    "locked_test_result_modified": False,
}
(catalog_dir / "paper_results_manifest.json").write_text(json.dumps(result_summary, indent=2, default=str), encoding="utf-8")

readme = f"""# CARE Experiment 1 — paper figures and tables

Generated: {result_summary['generated_at_utc']}

## Contents

- Publication-ready figures: **{len(FIGURE_ROWS)}**
- Paper tables: **{len(TABLE_ROWS)}**
- Figure formats: PNG (600 dpi), PDF, SVG
- Table formats: CSV, LaTeX, PNG, PDF
- Figure captions and priority ranking: `00_catalog/figure_manifest.csv`
- Table catalog: `00_catalog/table_manifest.csv`
- Long-form metrics: `00_catalog/paper_metrics_long.csv`
- Duplicate-row plotting audit: `00_catalog/plot_aggregation_notes.csv`
- Thumbnail review sheets: `00_catalog/contact_sheet_*.jpg`

## Scientific boundary

These artifacts are reporting-only. They do not refit preprocessing or models,
recalibrate anomaly thresholds, select a new persistence rule, or construct an
alternative test-policy grid. The final test remains locked at a 72-point
criticality threshold. Descriptive plots of test scores, flags, and maximum
criticality must not be used as a new tuning round.

## Recommended manuscript figures

Filter `figure_manifest.csv` where `paper_priority == "Main text"`. Use
supplementary per-case trajectories and score-component plots to document the
negative baseline and operating-state confounding transparently.
"""
(PAPER_RESULTS_ROOT / "README.md").write_text(readme, encoding="utf-8")

if len(FIGURE_ROWS) < EXPECTED_MINIMUM_FIGURES and not ALLOW_PARTIAL:
    raise RuntimeError(
        f"Only {len(FIGURE_ROWS)} figures were generated; at least "
        f"{EXPECTED_MINIMUM_FIGURES} were expected. Check missing tables in "
        f"{catalog_dir / 'skipped_optional_items.csv'}. Existing figures were "
        "retained for inspection."
    )

print("\n" + "=" * 88)
print("CELL 14 COMPLETED SUCCESSFULLY — PUBLICATION RESULTS EXPORTED")
print("=" * 88)
print(f"Paper-results directory : {PAPER_RESULTS_ROOT}")
print(f"Figures generated       : {len(FIGURE_ROWS)}")
print(f"Paper tables generated  : {len(TABLE_ROWS)}")
print(f"Source tables loaded    : {len(TABLES)}")
print(f"PNG resolution          : {PNG_DPI} dpi")
print("Vector formats          : PDF + SVG")
print("Alternative test grid   : No")
print("Model / threshold refit : No / No")
print("Final test modified     : No — locked result preserved")
print("Review first            : 00_catalog\\figure_manifest.csv")
print("=" * 88)
