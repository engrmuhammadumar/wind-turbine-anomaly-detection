"""
Publication figures for a fixed-threshold, OFFLINE RETROSPECTIVE RUL analysis.

This program deliberately derives one failure time from each complete predicted
wear trajectory. It therefore must not be described as rolling-origin, online,
or causal RUL inference.

Only six PNG files are written. No prediction values are fabricated, smoothed,
blended, or extrapolated.
"""

from collections import OrderedDict
from pathlib import Path
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

PROJECT_DIR = Path(r"E:\4 Paper\New Implementation_final")
TRAJECTORY_DIR = PROJECT_DIR / "results_real" / "trajectories"

WEAR_VALIDATION_FILE = TRAJECTORY_DIR / "wear_trajectories_validation.csv"
WEAR_BLIND_FILE = TRAJECTORY_DIR / "wear_trajectories_test.csv"

OUTPUT_DIR = Path(r"F:\GIST work\Review Round 1\rul")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_THRESHOLD_UM = 165.0
INTERVAL_Z = 2.0
DPI = 600

LABELLED_CUTTERS = ("c1", "c4", "c6")
BLIND_CUTTERS = ("c2", "c3", "c5")
ALL_CUTTERS = ("c1", "c2", "c3", "c4", "c5", "c6")

# The displayed proposed method is stored in PINN_wear.
# Proposed_wear is the Deep State Space Model in the supplied files.
METHOD_COLUMNS = OrderedDict([
    ("TCN", "TCN_wear"),
    ("BiLSTM", "BiLSTM_wear"),
    ("Transformer", "Transformer_wear"),
    ("Neural ODE", "Neural ODE_wear"),
    ("Deep State Space Model", "Proposed_wear"),
    ("Proposed", "PINN_wear"),
])

PROPOSED_STD_COLUMN = "PINN_wear_std"

COLORS = {
    "Reference RUL": "#202020",
    "TCN": "#8172B3",
    "BiLSTM": "#3BA17C",
    "Transformer": "#3B84C3",
    "Neural ODE": "#E17C45",
    "Deep State Space Model": "#8A8A8A",
    "Proposed": "#D7263D",
}


# =============================================================================
# 2. DATA VALIDATION
# =============================================================================

def load_wear_file(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} file was not found:\n{path}\n\n"
            "Check PROJECT_DIR and the trajectories folder."
        )

    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]

    required = {"cutter", "cut_number", *METHOD_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} file is missing required columns:\n"
            + "\n".join(missing)
            + f"\n\nAvailable columns:\n{frame.columns.tolist()}"
        )

    frame["cutter"] = frame["cutter"].astype(str).str.strip().str.lower()
    frame["cut_number"] = pd.to_numeric(frame["cut_number"], errors="coerce")

    numeric_columns = list(METHOD_COLUMNS.values())
    if "wear_true" in frame.columns:
        numeric_columns.append("wear_true")
    if PROPOSED_STD_COLUMN in frame.columns:
        numeric_columns.append(PROPOSED_STD_COLUMN)

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["cut_number"].isna().any():
        raise ValueError(f"{dataset_name}: non-numeric cut_number values found.")

    duplicates = frame.duplicated(["cutter", "cut_number"], keep=False)
    if duplicates.any():
        duplicated_rows = frame.loc[
            duplicates, ["cutter", "cut_number"]
        ].sort_values(["cutter", "cut_number"])
        raise ValueError(
            f"{dataset_name}: duplicate cutter/pass rows found:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    return frame.sort_values(["cutter", "cut_number"]).reset_index(drop=True)


validation_df = load_wear_file(
    WEAR_VALIDATION_FILE, "Labelled wear-trajectory"
)
blind_df = load_wear_file(WEAR_BLIND_FILE, "Blind wear-trajectory")

if "wear_true" not in validation_df.columns:
    raise ValueError(
        "The validation trajectory file must contain the measured column "
        "'wear_true' for C1, C4 and C6."
    )

for cutter in LABELLED_CUTTERS:
    cutter_rows = validation_df[validation_df["cutter"] == cutter]
    if cutter_rows.empty:
        raise ValueError(f"Labelled cutter {cutter.upper()} is missing.")
    if cutter_rows["wear_true"].isna().any():
        raise ValueError(
            f"Measured wear contains missing values for {cutter.upper()}."
        )

for cutter in BLIND_CUTTERS:
    if blind_df[blind_df["cutter"] == cutter].empty:
        raise ValueError(f"Blind cutter {cutter.upper()} is missing.")

if PROPOSED_STD_COLUMN in validation_df.columns:
    if (validation_df[PROPOSED_STD_COLUMN].dropna() < 0).any():
        raise ValueError("Negative proposed wear standard deviations were found.")
if PROPOSED_STD_COLUMN in blind_df.columns:
    if (blind_df[PROPOSED_STD_COLUMN].dropna() < 0).any():
        raise ValueError("Negative proposed wear standard deviations were found.")


# =============================================================================
# 3. FIXED-THRESHOLD RUL CONSTRUCTION
# =============================================================================

def first_crossing_linear(x, y, threshold):
    """Return the first linearly interpolated threshold-crossing pass."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    crossing_indices = np.flatnonzero(y >= threshold)
    if len(crossing_indices) == 0:
        return np.nan

    k = int(crossing_indices[0])
    if k == 0:
        return float(x[0])

    x0, x1 = float(x[k - 1]), float(x[k])
    y0, y1 = float(y[k - 1]), float(y[k])

    if np.isclose(y1, y0):
        return x1

    fraction = (threshold - y0) / (y1 - y0)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def rul_from_failure_time(x, failure_time):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(failure_time):
        return np.full_like(x, np.nan, dtype=float)
    return np.maximum(float(failure_time) - x, 0.0)


def proposed_crossing_interval(frame):
    """
    Convert mean wear +/- 2 standard deviations into threshold-crossing times.

    The upper wear envelope normally gives the earlier failure time and the
    lower wear envelope the later failure time. Both must cross the threshold;
    otherwise the two-sided RUL interval is right-censored and is not drawn.
    """
    if PROPOSED_STD_COLUMN not in frame.columns:
        return np.nan, np.nan

    x = frame["cut_number"].to_numpy(dtype=float)
    mean = frame[METHOD_COLUMNS["Proposed"]].to_numpy(dtype=float)
    std = frame[PROPOSED_STD_COLUMN].to_numpy(dtype=float)

    if not np.all(np.isfinite(std)):
        return np.nan, np.nan

    early = first_crossing_linear(
        x, mean + INTERVAL_Z * std, FAILURE_THRESHOLD_UM
    )
    late = first_crossing_linear(
        x, mean - INTERVAL_Z * std, FAILURE_THRESHOLD_UM
    )

    if not (np.isfinite(early) and np.isfinite(late)):
        return np.nan, np.nan

    return min(early, late), max(early, late)


# =============================================================================
# 4. AUDIT ALL FAILURE TIMES BEFORE PLOTTING
# =============================================================================

reference_failure_times = {}
predicted_failure_times = {}
interval_failure_times = {}

print("\n" + "=" * 92)
print("FIXED-THRESHOLD OFFLINE RETROSPECTIVE RUL AUDIT")
print("=" * 92)
print(f"Common mean flank-wear threshold: {FAILURE_THRESHOLD_UM:.2f} µm")
print("Crossing rule: first crossing with linear interpolation")
print("Smoothing/extrapolation/blending: none")
print("=" * 92)

for cutter in ALL_CUTTERS:
    source = validation_df if cutter in LABELLED_CUTTERS else blind_df
    d = source[source["cutter"] == cutter].copy()
    x = d["cut_number"].to_numpy(dtype=float)

    if cutter in LABELLED_CUTTERS:
        t_reference = first_crossing_linear(
            x, d["wear_true"].to_numpy(dtype=float), FAILURE_THRESHOLD_UM
        )
        if not np.isfinite(t_reference):
            raise ValueError(
                f"Measured wear for {cutter.upper()} does not reach "
                f"{FAILURE_THRESHOLD_UM:.2f} µm. A reference RUL cannot be "
                "defined with the stated threshold."
            )
        reference_failure_times[cutter] = t_reference
        print(f"\n{cutter.upper()} — reference T_fail = {t_reference:.4f} passes")
    else:
        reference_failure_times[cutter] = np.nan
        print(f"\n{cutter.upper()} — reference T_fail unavailable (unlabelled)")

    predicted_failure_times[cutter] = {}
    for method, column in METHOD_COLUMNS.items():
        values = d[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            warnings.warn(
                f"{cutter.upper()} / {method}: missing predicted-wear values; "
                "the RUL curve will not be drawn."
            )
            crossing = np.nan
        else:
            crossing = first_crossing_linear(
                x, values, FAILURE_THRESHOLD_UM
            )

        predicted_failure_times[cutter][method] = crossing
        crossing_text = f"{crossing:.4f}" if np.isfinite(crossing) else "not reached"
        print(f"  {method:<29}: {crossing_text}")

    interval_failure_times[cutter] = proposed_crossing_interval(d)
    lo_t, hi_t = interval_failure_times[cutter]
    if np.isfinite(lo_t) and np.isfinite(hi_t):
        print(f"  Proposed crossing interval   : [{lo_t:.4f}, {hi_t:.4f}]")
    elif PROPOSED_STD_COLUMN in d.columns:
        print("  Proposed crossing interval   : right-censored/not drawable")


# =============================================================================
# 5. PUBLICATION STYLE
# =============================================================================

available_fonts = {font.name for font in fm.fontManager.ttflist}
serif_font = next(
    (
        name
        for name in ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
        if name in available_fonts
    ),
    "DejaVu Serif",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [serif_font],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.2,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "savefig.facecolor": "white",
})


# =============================================================================
# 6. SIX INDIVIDUAL FIGURES
# =============================================================================

def plot_one_cutter(cutter):
    labelled = cutter in LABELLED_CUTTERS
    source = validation_df if labelled else blind_df
    d = source[source["cutter"] == cutter].sort_values("cut_number").copy()

    x_full = d["cut_number"].to_numpy(dtype=float)
    if labelled:
        t_reference = reference_failure_times[cutter]
        # Display the physically meaningful pre-failure interval only.
        display_mask = x_full <= np.ceil(t_reference)
    else:
        t_reference = np.nan
        display_mask = np.ones(len(d), dtype=bool)

    x = x_full[display_mask]
    if len(x) == 0:
        raise ValueError(f"No display points remain for {cutter.upper()}.")

    fig, ax = plt.subplots(figsize=(7.25, 5.15), constrained_layout=False)

    handles = []
    censored_methods = []

    if labelled:
        reference_rul = rul_from_failure_time(x, t_reference)
        reference_line, = ax.plot(
            x,
            reference_rul,
            color=COLORS["Reference RUL"],
            linewidth=2.0,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markevery=max(1, len(x) // 11),
            zorder=8,
            label="Reference RUL",
        )
        handles.append(reference_line)

    # Draw uncertainty first so all curves remain visible above it.
    interval_low_t, interval_high_t = interval_failure_times[cutter]
    interval_drawn = np.isfinite(interval_low_t) and np.isfinite(interval_high_t)
    if interval_drawn:
        interval_low = rul_from_failure_time(x, interval_low_t)
        interval_high = rul_from_failure_time(x, interval_high_t)
        ax.fill_between(
            x,
            interval_low,
            interval_high,
            color=COLORS["Proposed"],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

    for method in METHOD_COLUMNS:
        t_predicted = predicted_failure_times[cutter][method]
        if not np.isfinite(t_predicted):
            censored_methods.append(method)
            continue

        predicted_rul = rul_from_failure_time(x, t_predicted)
        is_proposed = method == "Proposed"
        line, = ax.plot(
            x,
            predicted_rul,
            color=COLORS[method],
            linewidth=2.6 if is_proposed else 1.35,
            alpha=1.0 if is_proposed else 0.95,
            zorder=7 if is_proposed else 4,
            label=method,
        )
        handles.append(line)

    if interval_drawn:
        handles.append(
            Patch(
                facecolor=COLORS["Proposed"],
                edgecolor=COLORS["Proposed"],
                alpha=0.16,
                label="Proposed threshold-crossing interval (±2σ wear)",
            )
        )

    ax.axhline(0.0, color="#9A2F2F", linestyle=":", linewidth=1.0, zorder=2)

    if labelled:
        ax.axvline(
            t_reference,
            color="#9A2F2F",
            linestyle="--",
            linewidth=1.1,
            zorder=2,
        )
        ax.text(
            0.025,
            0.945,
            rf"$T_{{\mathrm{{fail}}}}={t_reference:.2f}$ passes",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#8F1D1D",
            fontsize=10.5,
        )
    else:
        ax.text(
            0.025,
            0.065,
            "Blind inference: no released wear labels or reference RUL",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#555555",
            fontsize=9.2,
            bbox={
                "boxstyle": "round,pad=0.27",
                "facecolor": "white",
                "edgecolor": "#888888",
                "alpha": 0.94,
            },
        )

    if censored_methods:
        ax.text(
            0.025,
            0.015 if labelled else 0.14,
            "Threshold not reached within observed horizon: "
            + ", ".join(censored_methods),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#666666",
            fontsize=7.8,
        )

    ax.text(
        0.965,
        0.94,
        cutter.upper(),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
        },
    )

    ax.set_xlabel("Cutting pass")
    ax.set_ylabel("Remaining useful life (cutting passes)")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.24)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_margin = max(1.0, 0.012 * max(x_max - x_min, 1.0))
    ax.set_xlim(x_min - x_margin, x_max + x_margin)

    visible_maxima = []
    if labelled:
        visible_maxima.append(float(np.nanmax(reference_rul)))
    for method in METHOD_COLUMNS:
        t_predicted = predicted_failure_times[cutter][method]
        if np.isfinite(t_predicted):
            visible_maxima.append(float(np.nanmax(rul_from_failure_time(x, t_predicted))))
    if interval_drawn:
        visible_maxima.append(float(np.nanmax(interval_high)))

    y_max = max(visible_maxima) if visible_maxima else 1.0
    ax.set_ylim(-0.02 * y_max, 1.10 * y_max)

    legend_columns = 3 if len(handles) >= 6 else 2
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=legend_columns,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        handlelength=2.6,
        columnspacing=1.25,
        borderpad=0.7,
    )
    legend.get_frame().set_linewidth(0.9)

    status = "Validation" if labelled else "Blind_Inference"
    output_path = OUTPUT_DIR / f"{cutter.upper()}_RUL_All_Methods_{status}.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[SAVED] {output_path}")


print("\nGenerating six individual PNG figures...")
for cutter in ALL_CUTTERS:
    plot_one_cutter(cutter)


# =============================================================================
# 7. FAIR LABELLED-CUTTER METRIC AUDIT (CONSOLE ONLY)
# =============================================================================

def deterministic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if denominator <= 0 else 1.0 - float(np.sum(error ** 2)) / denominator
    return rmse, mae, r2


metric_store = {method: {"truth": [], "prediction": [], "cutters": []}
                for method in METHOD_COLUMNS}

print("\n" + "=" * 92)
print("CUTTER-WISE RUL PERFORMANCE — LABELLED CUTTERS ONLY")
print("All metrics use the same fixed-threshold reference definition.")
print("=" * 92)

for cutter in LABELLED_CUTTERS:
    d = validation_df[validation_df["cutter"] == cutter].sort_values("cut_number")
    x_all = d["cut_number"].to_numpy(dtype=float)
    t_reference = reference_failure_times[cutter]
    use = x_all <= t_reference
    x = x_all[use]
    y_true = rul_from_failure_time(x, t_reference)

    print(f"\n{cutter.upper()} — reference T_fail={t_reference:.4f}")
    for method in METHOD_COLUMNS:
        t_predicted = predicted_failure_times[cutter][method]
        if not np.isfinite(t_predicted):
            print(f"  {method:<29}: not evaluated (threshold crossing censored)")
            continue

        y_pred = rul_from_failure_time(x, t_predicted)
        rmse, mae, r2 = deterministic_metrics(y_true, y_pred)
        fail_error = t_predicted - t_reference
        print(
            f"  {method:<29}: RMSE={rmse:8.3f}, MAE={mae:8.3f}, "
            f"R²={r2:8.4f}, T_fail error={fail_error:+8.3f}"
        )

        metric_store[method]["truth"].append(y_true)
        metric_store[method]["prediction"].append(y_pred)
        metric_store[method]["cutters"].append(cutter)

print("\n" + "=" * 92)
print("POOLED PERFORMANCE — REPORTED ONLY WHEN ALL THREE LABELLED CUTTERS EXIST")
print("=" * 92)

for method in METHOD_COLUMNS:
    used_cutters = metric_store[method]["cutters"]
    if set(used_cutters) != set(LABELLED_CUTTERS):
        missing_cutters = sorted(set(LABELLED_CUTTERS).difference(used_cutters))
        print(
            f"{method:<29}: not pooled; censored/missing on "
            + ", ".join(c.upper() for c in missing_cutters)
        )
        continue

    pooled_true = np.concatenate(metric_store[method]["truth"])
    pooled_prediction = np.concatenate(metric_store[method]["prediction"])
    rmse, mae, r2 = deterministic_metrics(pooled_true, pooled_prediction)
    print(f"{method:<29}: RMSE={rmse:8.3f}, MAE={mae:8.3f}, R²={r2:8.4f}")

print("\n" + "=" * 92)
print("Completed: six PNG files")
print(f"Output directory: {OUTPUT_DIR}")
print("IMPORTANT: These are offline retrospective threshold-crossing results.")
print("Do not describe them as online, causal, or rolling-origin predictions.")
print("=" * 92)

