"""
Reviewer-safe plotting-only workflow for the causal PINDP-Net experiment.

The program reads the untouched prediction CSV files created by
``pindpnet_end_to_end.py``. It does not train a model, smooth predictions,
blend predictions with references, fabricate comparator curves, change metric
values, or hide uncertainty outside an axis range.

Outputs
-------
* Figure11_Causal_Wear_All_Six_Cutters.png
* Figure12_Causal_RUL_All_Six_Cutters.png
* Figure11a_Labelled_Wear_OOF.png
* Figure12a_Labelled_RUL_OOF.png
* Figure11b_Blind_Wear_Inference.png
* Figure12b_Blind_RUL_Inference.png
* twelve individual cutter/quantity PNG files
* plotted_values_labelled.csv
* plotted_values_blind.csv
* metrics_from_plotted_values.csv
* plotting_audit.json
* suggested_figure_captions.txt

All figures are saved as 600-dpi PNG files.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# 1. USER SETTINGS — EDIT ONLY IF YOUR OUTPUT FOLDER IS DIFFERENT
# =============================================================================

RESULTS_DIR = Path(r"F:\GIST work\Review Round 1\pindpnet_causal")
OOF_FILE = RESULTS_DIR / "pindpnet_oof_labelled.csv"
BLIND_FILE = RESULTS_DIR / "pindpnet_blind_c2_c3_c5.csv"
SAVE_DIR = RESULTS_DIR / "figures_reviewer_safe"

FAILURE_THRESHOLD_UM = 165.0
INTERVAL_Z = 1.96
DPI = 600

LABELLED_CUTTERS = ("c1", "c4", "c6")
BLIND_CUTTERS = ("c2", "c3", "c5")
ALL_CUTTERS = ("c1", "c2", "c3", "c4", "c5", "c6")
PANEL_LABELS = dict(zip(ALL_CUTTERS, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")))

RED = "#D7263D"
BLACK = "#202020"
THRESHOLD_RED = "#8B1A1A"
GRID_GREY = "#C9C9C9"


# =============================================================================
# 2. INPUT VALIDATION
# =============================================================================


OOF_REQUIRED = {
    "cutter",
    "cut_number",
    "wear_pred_um",
    "wear_std_um",
    "rul_pred_passes",
    "rul_std_passes",
    "wear_true_um",
}

BLIND_REQUIRED = {
    "cutter",
    "cut_number",
    "wear_pred_um",
    "wear_std_um",
    "rul_pred_passes",
    "rul_std_passes",
}

NUMERIC_COLUMNS = (
    "cut_number",
    "wear_pred_um",
    "wear_std_um",
    "rul_pred_passes",
    "rul_std_passes",
    "wear_true_um",
    "rul_true_passes",
    "failure_pass_true",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_prediction_file(
    path: Path,
    required_columns: set[str],
    expected_cutters: Sequence[str],
    name: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} was not found:\n{path}\n\n"
            "Change RESULTS_DIR in Section 1 if necessary."
        )

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {frame.columns.tolist()}"
        )

    frame["cutter"] = frame["cutter"].astype(str).str.strip().str.lower()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    found_cutters = set(frame["cutter"].dropna().unique())
    if found_cutters != set(expected_cutters):
        raise ValueError(
            f"{name} must contain exactly {tuple(c.upper() for c in expected_cutters)}; "
            f"found {tuple(c.upper() for c in sorted(found_cutters))}."
        )

    duplicate_mask = frame.duplicated(["cutter", "cut_number"], keep=False)
    if duplicate_mask.any():
        duplicated = frame.loc[duplicate_mask, ["cutter", "cut_number"]]
        raise ValueError(
            f"{name} contains duplicate cutter/pass rows:\n"
            f"{duplicated.head(20).to_string(index=False)}"
        )

    essential_numeric = sorted(required_columns.difference({"cutter"}))
    if frame[essential_numeric].isna().any().any():
        counts = frame[essential_numeric].isna().sum()
        counts = counts[counts > 0]
        raise ValueError(f"{name} contains missing/non-numeric values:\n{counts}")

    if (frame["wear_std_um"] < 0).any() or (frame["rul_std_passes"] < 0).any():
        raise ValueError(f"{name} contains negative predictive standard deviations.")
    if (frame["cut_number"] < 0).any():
        raise ValueError(f"{name} contains negative cutting-pass indices.")

    return frame.sort_values(["cutter", "cut_number"]).reset_index(drop=True)


def reject_blind_references(frame: pd.DataFrame) -> None:
    forbidden = ("wear_true_um", "rul_true_passes", "failure_pass_true")
    populated = [column for column in forbidden if column in frame and frame[column].notna().any()]
    if populated:
        raise ValueError(
            "Blind C2/C3/C5 predictions must not contain populated reference columns: "
            + ", ".join(populated)
        )


# =============================================================================
# 3. CONSISTENT REFERENCE-RUL RECONSTRUCTION
# =============================================================================


def first_crossing_linear(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    indices = np.flatnonzero(y >= threshold)
    if len(indices) == 0:
        return float("nan")
    index = int(indices[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if math.isclose(y0, y1):
        return x1
    fraction = float(np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def add_reference_rul(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    parts: list[pd.DataFrame] = []
    failure_times: dict[str, float] = {}

    for cutter in LABELLED_CUTTERS:
        subset = oof[oof["cutter"] == cutter].copy()
        failure_time = first_crossing_linear(
            subset["cut_number"].to_numpy(),
            subset["wear_true_um"].to_numpy(),
            FAILURE_THRESHOLD_UM,
        )
        if not np.isfinite(failure_time):
            raise ValueError(
                f"Measured wear for {cutter.upper()} does not reach "
                f"{FAILURE_THRESHOLD_UM:.2f} µm. Reference RUL cannot be constructed."
            )
        failure_times[cutter] = failure_time
        reconstructed = np.maximum(
            failure_time - subset["cut_number"].to_numpy(dtype=float), 0.0
        )

        if "failure_pass_true" in subset and subset["failure_pass_true"].notna().any():
            stored = subset["failure_pass_true"].dropna().to_numpy(dtype=float)
            if not np.allclose(stored, failure_time, rtol=0.0, atol=1.0e-3):
                raise ValueError(
                    f"Stored failure pass for {cutter.upper()} conflicts with the "
                    "165 µm first-crossing definition."
                )
        if "rul_true_passes" in subset and subset["rul_true_passes"].notna().any():
            stored_rul = subset["rul_true_passes"].to_numpy(dtype=float)
            if not np.allclose(stored_rul, reconstructed, rtol=0.0, atol=1.0e-3):
                raise ValueError(
                    f"Stored reference RUL for {cutter.upper()} conflicts with "
                    "max(T_fail - t, 0)."
                )

        subset["failure_pass_reference"] = failure_time
        subset["rul_reference_passes"] = reconstructed
        parts.append(subset)

    return (
        pd.concat(parts, ignore_index=True).sort_values(["cutter", "cut_number"]),
        failure_times,
    )


# =============================================================================
# 4. METRICS CALCULATED FROM THE EXACT PLOTTED VALUES
# =============================================================================


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    denominator = float(np.sum(np.square(target - np.mean(target))))
    r2 = 1.0 - float(np.sum(np.square(error))) / denominator if denominator > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "r2": r2,
    }


def uncertainty_metrics(
    target: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    mean = np.asarray(mean, dtype=float)
    standard_deviation = np.maximum(np.asarray(standard_deviation, dtype=float), 1.0e-12)
    lower = np.maximum(mean - INTERVAL_Z * standard_deviation, 0.0)
    upper = mean + INTERVAL_Z * standard_deviation
    covered = (target >= lower) & (target <= upper)
    nll = np.mean(
        np.log(standard_deviation)
        + 0.5 * np.square((target - mean) / standard_deviation)
        + 0.5 * np.log(2.0 * np.pi)
    )
    return {
        "picp_95": float(np.mean(covered)),
        "mpiw": float(np.mean(upper - lower)),
        "gaussian_nll": float(nll),
    }


def predicted_failure_pass(subset: pd.DataFrame) -> float:
    return first_crossing_linear(
        subset["cut_number"].to_numpy(),
        subset["wear_pred_um"].to_numpy(),
        FAILURE_THRESHOLD_UM,
    )


def calculate_metrics(
    labelled: pd.DataFrame,
    failure_times: Mapping[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [(cutter.upper(), labelled[labelled["cutter"] == cutter]) for cutter in LABELLED_CUTTERS]
    groups.append(("POOLED_OOF", labelled))

    for scope, subset in groups:
        wear = regression_metrics(subset["wear_true_um"], subset["wear_pred_um"])
        rul = regression_metrics(subset["rul_reference_passes"], subset["rul_pred_passes"])
        wear_u = uncertainty_metrics(
            subset["wear_true_um"], subset["wear_pred_um"], subset["wear_std_um"]
        )
        rul_u = uncertainty_metrics(
            subset["rul_reference_passes"],
            subset["rul_pred_passes"],
            subset["rul_std_passes"],
        )
        row: dict[str, Any] = {
            "scope": scope,
            "n_passes": len(subset),
            "wear_rmse_um": wear["rmse"],
            "wear_mae_um": wear["mae"],
            "wear_r2": wear["r2"],
            "rul_rmse_passes": rul["rmse"],
            "rul_mae_passes": rul["mae"],
            "rul_r2": rul["r2"],
            "wear_picp_95": wear_u["picp_95"],
            "wear_mpiw_um": wear_u["mpiw"],
            "wear_gaussian_nll": wear_u["gaussian_nll"],
            "rul_picp_95": rul_u["picp_95"],
            "rul_mpiw_passes": rul_u["mpiw"],
            "rul_gaussian_nll": rul_u["gaussian_nll"],
        }
        if scope != "POOLED_OOF":
            cutter = scope.lower()
            estimated_failure = predicted_failure_pass(subset)
            row["reference_failure_pass"] = failure_times[cutter]
            row["predicted_failure_pass"] = estimated_failure
            row["failure_pass_error"] = (
                estimated_failure - failure_times[cutter]
                if np.isfinite(estimated_failure)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 5. PUBLICATION STYLE AND PANEL DRAWING
# =============================================================================


def configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.fancybox": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def interval_bounds(mean: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(mean - INTERVAL_Z * sigma, 0.0)
    upper = mean + INTERVAL_Z * sigma
    return lower, upper


def panel_data(frame: pd.DataFrame, cutter: str) -> pd.DataFrame:
    subset = frame[frame["cutter"] == cutter].sort_values("cut_number").copy()
    if subset.empty:
        raise ValueError(f"No rows found for {cutter.upper()}.")
    return subset


def draw_panel(
    axis: plt.Axes,
    subset: pd.DataFrame,
    cutter: str,
    quantity: str,
    labelled: bool,
    show_panel_letter: bool,
) -> float:
    x = subset["cut_number"].to_numpy(dtype=float)
    if quantity == "wear":
        mean = subset["wear_pred_um"].to_numpy(dtype=float)
        sigma = subset["wear_std_um"].to_numpy(dtype=float)
        truth = subset["wear_true_um"].to_numpy(dtype=float) if labelled else None
    elif quantity == "rul":
        mean = subset["rul_pred_passes"].to_numpy(dtype=float)
        sigma = subset["rul_std_passes"].to_numpy(dtype=float)
        truth = subset["rul_reference_passes"].to_numpy(dtype=float) if labelled else None
    else:
        raise ValueError(quantity)

    lower, upper = interval_bounds(mean, sigma)
    axis.fill_between(x, lower, upper, color=RED, alpha=0.15, linewidth=0, zorder=1)
    axis.plot(x, mean, color=RED, linewidth=2.0, zorder=3)
    if labelled and truth is not None:
        axis.plot(
            x,
            truth,
            color=BLACK,
            linewidth=1.6,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markevery=max(1, len(x) // 11),
            zorder=4,
        )
    else:
        axis.text(
            0.025,
            0.055,
            "Blind inference—no reference labels",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#999999",
                "linewidth": 0.8,
            },
        )

    # The threshold is a reference-dependent evaluation marker here and is
    # deliberately omitted from the unlabelled C2/C3/C5 panels.
    if quantity == "wear" and labelled:
        axis.axhline(
            FAILURE_THRESHOLD_UM,
            color=THRESHOLD_RED,
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
    if quantity == "rul":
        axis.axhline(0.0, color=THRESHOLD_RED, linestyle=":", linewidth=0.9, zorder=2)

    if show_panel_letter:
        axis.text(
            0.025,
            0.94,
            PANEL_LABELS[cutter],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    axis.text(
        0.965,
        0.94,
        cutter.upper(),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.9,
        },
    )
    axis.set_xlim(0.0, 315.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Cutting pass")
    axis.grid(True, color=GRID_GREY, linestyle="--", linewidth=0.45, alpha=0.45)
    return float(np.nanmax(upper))


def legend_handles(quantity: str, include_reference: bool, include_threshold: bool) -> list[Any]:
    handles: list[Any] = []
    if include_reference:
        label = "Reference wear" if quantity == "wear" else "Reference RUL"
        handles.append(
            Line2D(
                [0], [0], color=BLACK, linewidth=1.7, marker="o",
                markerfacecolor="white", markersize=4.5, label=label
            )
        )
    handles.extend(
        [
            Line2D([0], [0], color=RED, linewidth=2.2, label="PINDP-Net"),
            Patch(facecolor=RED, alpha=0.15, label="PINDP-Net 95% predictive interval"),
        ]
    )
    if include_threshold:
        handles.append(
            Line2D(
                [0], [0], color=THRESHOLD_RED, linestyle="--", linewidth=1.1,
                label=f"Failure threshold ({FAILURE_THRESHOLD_UM:g} µm)"
            )
        )
    return handles


def y_axis_label(quantity: str) -> str:
    return "Mean flank wear (µm)" if quantity == "wear" else "Remaining useful life (cutting passes)"


def save_three_panel_figure(
    frame: pd.DataFrame,
    cutters: Sequence[str],
    quantity: str,
    labelled: bool,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.35), sharex=True, sharey=False)
    maxima = []
    for axis, cutter in zip(axes, cutters):
        maxima.append(
            draw_panel(axis, panel_data(frame, cutter), cutter, quantity, labelled, True)
        )
    # A common scale prevents visual exaggeration across cutters.
    common_upper = max(maxima) * 1.04
    if quantity == "wear":
        common_upper = max(common_upper, FAILURE_THRESHOLD_UM * 1.12)
    for axis in axes:
        axis.set_ylim(0.0, common_upper)
    axes[0].set_ylabel(y_axis_label(quantity))
    figure.legend(
        handles=legend_handles(
            quantity,
            include_reference=labelled,
            include_threshold=labelled and quantity == "wear",
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def save_six_panel_figure(
    labelled_frame: pd.DataFrame,
    blind_frame: pd.DataFrame,
    quantity: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), sharex=True, sharey=False)
    maxima = []
    for axis, cutter in zip(axes.flat, ALL_CUTTERS):
        is_labelled = cutter in LABELLED_CUTTERS
        source = labelled_frame if is_labelled else blind_frame
        maxima.append(
            draw_panel(
                axis,
                panel_data(source, cutter),
                cutter,
                quantity,
                is_labelled,
                True,
            )
        )
    common_upper = max(maxima) * 1.04
    if quantity == "wear":
        common_upper = max(common_upper, FAILURE_THRESHOLD_UM * 1.12)
    for index, axis in enumerate(axes.flat):
        axis.set_ylim(0.0, common_upper)
        if index % 3 == 0:
            axis.set_ylabel(y_axis_label(quantity))
    figure.legend(
        handles=legend_handles(
            quantity,
            include_reference=True,
            include_threshold=quantity == "wear",
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def save_individual_figure(
    frame: pd.DataFrame,
    cutter: str,
    quantity: str,
    labelled: bool,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.25, 4.65))
    maximum = draw_panel(
        axis, panel_data(frame, cutter), cutter, quantity, labelled, False
    )
    upper = maximum * 1.04
    if quantity == "wear":
        upper = max(upper, FAILURE_THRESHOLD_UM * 1.12)
    axis.set_ylim(0.0, upper)
    axis.set_ylabel(y_axis_label(quantity))
    axis.legend(
        handles=legend_handles(
            quantity,
            include_reference=labelled,
            include_threshold=labelled and quantity == "wear",
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=2,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


# =============================================================================
# 6. CAPTIONS AND COMPLETE EXECUTION
# =============================================================================


CAPTIONS = """Figure 11. Causal mean flank-wear predictions for the six PHM 2010 milling cutters. Panels (a), (d), and (f) show cutter-level out-of-fold predictions for the labelled cutters C1, C4, and C6, respectively; their black curves denote the experimentally measured mean of the three flute-wear measurements. Panels (b), (c), and (e) show blind inference for the unlabelled cutters C2, C3, and C5 without measured reference curves. The red curve and shaded region denote the PINDP-Net predictive mean and 95% predictive interval, respectively. The fixed 165 µm failure threshold is shown only in the labelled panels. C2, C3, and C5 are excluded from all reference-dependent metrics.

Figure 12. Causal remaining-useful-life predictions for the six PHM 2010 milling cutters. For C1, C4, and C6, reference RUL is defined in cutting passes as max(T_fail − t, 0), where T_fail is the first linearly interpolated crossing of the fixed 165 µm mean-wear threshold. The black curve is the resulting reference RUL, while the red curve and shaded region are the untouched PINDP-Net predictive mean and 95% predictive interval. C2, C3, and C5 are presented solely as blind inference because released wear annotations, reference failure passes, and reference RUL values are unavailable. Only C1, C4, and C6 contribute to quantitative RUL and uncertainty evaluation.
"""


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(dict(payload)), handle, indent=2, ensure_ascii=False)


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    labelled = load_prediction_file(
        OOF_FILE, OOF_REQUIRED, LABELLED_CUTTERS, "Labelled OOF prediction file"
    )
    blind = load_prediction_file(
        BLIND_FILE, BLIND_REQUIRED, BLIND_CUTTERS, "Blind prediction file"
    )
    reject_blind_references(blind)
    labelled, failure_times = add_reference_rul(labelled)

    # Immutable exports make it explicit which exact values enter every plot.
    labelled.to_csv(SAVE_DIR / "plotted_values_labelled.csv", index=False)
    blind.to_csv(SAVE_DIR / "plotted_values_blind.csv", index=False)
    metrics = calculate_metrics(labelled, failure_times)
    metrics.to_csv(SAVE_DIR / "metrics_from_plotted_values.csv", index=False)

    configure_style()
    save_six_panel_figure(
        labelled,
        blind,
        "wear",
        SAVE_DIR / "Figure11_Causal_Wear_All_Six_Cutters.png",
    )
    save_six_panel_figure(
        labelled,
        blind,
        "rul",
        SAVE_DIR / "Figure12_Causal_RUL_All_Six_Cutters.png",
    )
    save_three_panel_figure(
        labelled,
        LABELLED_CUTTERS,
        "wear",
        True,
        SAVE_DIR / "Figure11a_Labelled_Wear_OOF.png",
    )
    save_three_panel_figure(
        labelled,
        LABELLED_CUTTERS,
        "rul",
        True,
        SAVE_DIR / "Figure12a_Labelled_RUL_OOF.png",
    )
    save_three_panel_figure(
        blind,
        BLIND_CUTTERS,
        "wear",
        False,
        SAVE_DIR / "Figure11b_Blind_Wear_Inference.png",
    )
    save_three_panel_figure(
        blind,
        BLIND_CUTTERS,
        "rul",
        False,
        SAVE_DIR / "Figure12b_Blind_RUL_Inference.png",
    )

    for cutter in ALL_CUTTERS:
        is_labelled = cutter in LABELLED_CUTTERS
        source = labelled if is_labelled else blind
        suffix = "OOF_Validation" if is_labelled else "Blind_Inference"
        save_individual_figure(
            source,
            cutter,
            "wear",
            is_labelled,
            SAVE_DIR / f"{cutter.upper()}_Wear_{suffix}.png",
        )
        save_individual_figure(
            source,
            cutter,
            "rul",
            is_labelled,
            SAVE_DIR / f"{cutter.upper()}_RUL_{suffix}.png",
        )

    (SAVE_DIR / "suggested_figure_captions.txt").write_text(CAPTIONS, encoding="utf-8")
    audit = {
        "oof_input": OOF_FILE,
        "blind_input": BLIND_FILE,
        "oof_sha256": sha256(OOF_FILE),
        "blind_sha256": sha256(BLIND_FILE),
        "labelled_cutters": LABELLED_CUTTERS,
        "blind_cutters": BLIND_CUTTERS,
        "failure_threshold_um": FAILURE_THRESHOLD_UM,
        "uncertainty_interval": f"mean ± {INTERVAL_Z} predictive standard deviations",
        "reference_failure_passes": failure_times,
        "reference_rul_definition": "max(T_fail - cutting_pass, 0)",
        "failure_crossing": "first crossing with linear interpolation",
        "prediction_smoothing": False,
        "prediction_blending": False,
        "synthetic_curves": False,
        "manual_metric_values": False,
        "blind_reference_metrics": False,
        "lower_interval_truncated_at_physical_zero": True,
        "figure_dpi": DPI,
    }
    write_json(SAVE_DIR / "plotting_audit.json", audit)

    print("\n" + "=" * 96)
    print("REVIEWER-SAFE PLOTTING COMPLETED")
    print("=" * 96)
    print(f"Input OOF file:   {OOF_FILE}")
    print(f"Input blind file: {BLIND_FILE}")
    print(f"Output directory: {SAVE_DIR}")
    print("\nReference failure-pass audit:")
    for cutter, failure_time in failure_times.items():
        print(f"  {cutter.upper()}: {failure_time:.4f} cutting passes")
    print("\nMetrics calculated from the exact plotted values:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nNo training, smoothing, prediction correction, or synthetic data were used.")
    print("=" * 96)


if __name__ == "__main__":
    main()

