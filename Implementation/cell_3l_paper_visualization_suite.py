"""Cell 3L — publication visualization suite after deterministic retention.

Run inside the notebook namespace immediately after Cell 3K:

    %run -i cell_3l_paper_visualization_suite.py

The cell writes publication-ready PNG and PDF figures, CSV/LaTeX tables,
and a machine-readable figure registry. It uses only post-policy manifest rows
and never changes ROW_LABEL_MANIFEST or assigns a data split.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Validate Cell 3K and prepare an immutable plotting frame
# -----------------------------------------------------------------------------

if "ROW_LABEL_MANIFEST" not in globals():
    raise NameError("Run Cells 3I–3K before Cell 3L.")

REQUIRED_3L = {
    "farm",
    "asset_id",
    "asset_key",
    "event_key",
    "source_row_index",
    "timestamp_utc",
    "final_label",
    "modeling_eligible",
    "split_assignment",
}

missing_3l = REQUIRED_3L - set(ROW_LABEL_MANIFEST.columns)
if missing_3l:
    raise ValueError(
        "ROW_LABEL_MANIFEST is missing columns required by Cell 3L: "
        f"{sorted(missing_3l)}"
    )

EXPECTED_ELIGIBLE_3L = 213_537
eligible_count_3l = int(ROW_LABEL_MANIFEST["modeling_eligible"].sum())
if eligible_count_3l != EXPECTED_ELIGIBLE_3L:
    raise ValueError(
        "Cell 3L expects the verified post-3K eligible count of 213,537, "
        f"but found {eligible_count_3l:,}."
    )

if ROW_LABEL_MANIFEST["split_assignment"].notna().any():
    raise ValueError("Cell 3L must run before train/validation/test assignment.")

optional_3l = [
    c
    for c in [
        "event_id",
        "raw_id",
        "source_event_label",
        "same_label_duplicate_action",
        "same_label_duplicate_canonical_event_key",
        "same_label_duplicate_reason",
    ]
    if c in ROW_LABEL_MANIFEST.columns
]

manifest_columns_3l = sorted(REQUIRED_3L | set(optional_3l))
manifest_3l = ROW_LABEL_MANIFEST.loc[:, manifest_columns_3l].copy()

manifest_3l["farm"] = manifest_3l["farm"].astype("string").str.strip()
manifest_3l["asset_key"] = manifest_3l["asset_key"].astype("string")
manifest_3l["event_key"] = manifest_3l["event_key"].astype("string")
manifest_3l["final_label"] = (
    manifest_3l["final_label"].astype("string").str.strip().str.lower()
)
manifest_3l["timestamp_utc"] = pd.to_datetime(
    manifest_3l["timestamp_utc"], errors="coerce", utc=True
)
manifest_3l["modeling_eligible"] = (
    manifest_3l["modeling_eligible"].fillna(False).astype(bool)
)

eligible_3l = manifest_3l.loc[
    manifest_3l["modeling_eligible"]
    & manifest_3l["final_label"].isin(["normal", "anomaly"])
    & manifest_3l["timestamp_utc"].notna()
].copy()

if len(eligible_3l) != EXPECTED_ELIGIBLE_3L:
    raise ValueError(
        "All 213,537 post-retention eligible rows must have a valid timestamp "
        "and a normalized normal/anomaly label."
    )

if eligible_3l[["event_key", "source_row_index"]].duplicated().any():
    raise ValueError("Eligible manifest source keys are not unique.")

FARMS_3L = sorted(eligible_3l["farm"].dropna().astype(str).unique())
LABELS_3L = ["normal", "anomaly"]

if FARMS_3L != ["A", "B", "C"]:
    raise ValueError(f"Expected farms A, B, and C; found {FARMS_3L}.")

eligible_3l["date_utc"] = eligible_3l["timestamp_utc"].dt.floor("D")
eligible_3l["month_utc"] = (
    eligible_3l["timestamp_utc"]
    .dt.tz_convert(None)
    .dt.to_period("M")
    .dt.to_timestamp()
)
eligible_3l["hour_utc"] = eligible_3l["timestamp_utc"].dt.hour
eligible_3l["weekday_number"] = eligible_3l["timestamp_utc"].dt.dayofweek
eligible_3l["weekday"] = eligible_3l["timestamp_utc"].dt.day_name().str[:3]


# -----------------------------------------------------------------------------
# 2. Journal-oriented style and artifact registry
# -----------------------------------------------------------------------------

OUTPUT_ROOT_3L = Path("paper_visuals_3l")
PNG_DIR_3L = OUTPUT_ROOT_3L / "figures_png"
PDF_DIR_3L = OUTPUT_ROOT_3L / "figures_pdf"
TABLE_DIR_3L = OUTPUT_ROOT_3L / "tables"

for directory_3l in [PNG_DIR_3L, PDF_DIR_3L, TABLE_DIR_3L]:
    directory_3l.mkdir(parents=True, exist_ok=True)

COLORS_3L = {
    "normal": "#0072B2",
    "anomaly": "#D55E00",
    "eligible": "#009E73",
    "exact_copy_excluded": "#CC79A7",
    "other_ineligible": "#999999",
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
}

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIGURE_REGISTRY_3L: list[dict] = []
TABLE_REGISTRY_3L: list[dict] = []


def slug_3l(value: object) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return value or "item"


def finish_axes_3l(ax, xlabel=None, ylabel=None, legend=True):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False)


def save_figure_3l(
    fig,
    figure_id,
    slug,
    title,
    tier,
    claim,
    source,
):
    filename = f"{figure_id}_{slug_3l(slug)}"
    png_path = PNG_DIR_3L / f"{filename}.png"
    pdf_path = PDF_DIR_3L / f"{filename}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURE_REGISTRY_3L.append(
        {
            "figure_id": figure_id,
            "title": title,
            "tier": tier,
            "claim_supported": claim,
            "source_table": source,
            "png_path": str(png_path),
            "pdf_path": str(pdf_path),
        }
    )


def save_table_3l(table_id, name, frame, purpose):
    clean = frame.copy()
    csv_path = TABLE_DIR_3L / f"{table_id}_{slug_3l(name)}.csv"
    tex_path = TABLE_DIR_3L / f"{table_id}_{slug_3l(name)}.tex"
    clean.to_csv(csv_path, index=False)

    # Use a dependency-free LaTeX writer so tables remain exportable even when
    # pandas' optional Jinja2 dependency is unavailable.
    def latex_escape_3l(value):
        text = str(value)
        replacements = [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def latex_value_3l(value):
        if pd.isna(value):
            return "--"
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return latex_escape_3l(pd.Timestamp(value).isoformat())
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return latex_escape_3l(value)

    alignment_3l = "".join(
        "r" if pd.api.types.is_numeric_dtype(clean[column]) else "l"
        for column in clean.columns
    )
    header_3l = " & ".join(
        latex_escape_3l(column.replace("_", " ")) for column in clean.columns
    )
    with tex_path.open("w", encoding="utf-8") as handle_3l:
        handle_3l.write(f"\\begin{{tabular}}{{{alignment_3l}}}\n")
        handle_3l.write("\\hline\n")
        handle_3l.write(header_3l + r" \\" + "\n")
        handle_3l.write("\\hline\n")
        for row_3l in clean.itertuples(index=False, name=None):
            handle_3l.write(
                " & ".join(latex_value_3l(value) for value in row_3l)
                + r" \\" + "\n"
            )
        handle_3l.write("\\hline\n")
        handle_3l.write("\\end{tabular}\n")
    TABLE_REGISTRY_3L.append(
        {
            "table_id": table_id,
            "name": name,
            "purpose": purpose,
            "rows": len(clean),
            "columns": len(clean.columns),
            "csv_path": str(csv_path),
            "latex_path": str(tex_path) if tex_path else pd.NA,
        }
    )


def add_bar_labels_3l(ax, fmt="{x:,.0f}", fontsize=7):
    for container in ax.containers:
        try:
            ax.bar_label(
                container,
                labels=[fmt.format(x=v) if np.isfinite(v) else "" for v in container.datavalues],
                padding=2,
                fontsize=fontsize,
            )
        except Exception:
            pass


def grouped_bar_3l(
    frame,
    index,
    columns,
    values,
    figure_id,
    slug,
    title,
    ylabel,
    tier,
    claim,
    source,
    normalize=False,
):
    pivot = frame.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    pivot = pivot.reindex(columns=[x for x in LABELS_3L if x in pivot.columns])
    if normalize:
        pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    pivot.plot(
        kind="bar",
        ax=ax,
        color=[COLORS_3L.get(str(c), "#777777") for c in pivot.columns],
        width=0.75,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)
    finish_axes_3l(ax, index.replace("_", " ").title(), ylabel)
    add_bar_labels_3l(ax, fmt="{x:.1f}" if normalize else "{x:,.0f}")
    fig.tight_layout()
    save_figure_3l(fig, figure_id, slug, title, tier, claim, source)


def horizontal_rank_3l(
    frame,
    label_column,
    value_column,
    figure_id,
    slug,
    title,
    xlabel,
    tier,
    claim,
    source,
    top_n=25,
    color="#0072B2",
):
    plot = frame.nlargest(top_n, value_column).sort_values(value_column)
    height = max(4.0, 0.24 * len(plot) + 1.2)
    fig, ax = plt.subplots(figsize=(7.4, height))
    ax.barh(plot[label_column].astype(str), plot[value_column], color=color)
    ax.set_title(title)
    finish_axes_3l(ax, xlabel, None, legend=False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    add_bar_labels_3l(ax)
    fig.tight_layout()
    save_figure_3l(fig, figure_id, slug, title, tier, claim, source)


def matrix_figure_3l(
    matrix,
    figure_id,
    slug,
    title,
    xlabel,
    ylabel,
    tier,
    claim,
    source,
    cmap="viridis",
):
    matrix = matrix.copy()
    fig_width = max(6.0, 0.24 * len(matrix.columns) + 2.2)
    fig_height = max(3.8, 0.24 * len(matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.ax.set_ylabel("Count", rotation=270, labelpad=12)
    fig.tight_layout()
    save_figure_3l(fig, figure_id, slug, title, tier, claim, source)


# -----------------------------------------------------------------------------
# 3. Reusable paper tables
# -----------------------------------------------------------------------------

eligible_label_summary_3l = (
    eligible_3l.groupby(["farm", "final_label"], as_index=False)
    .agg(
        eligible_rows=("event_key", "size"),
        eligible_event_files=("event_key", "nunique"),
        eligible_assets=("asset_key", "nunique"),
        first_timestamp_utc=("timestamp_utc", "min"),
        last_timestamp_utc=("timestamp_utc", "max"),
    )
    .sort_values(["farm", "final_label"])
    .reset_index(drop=True)
)

event_summary_3l = (
    eligible_3l.groupby(
        ["farm", "asset_id", "asset_key", "event_key", "final_label"],
        as_index=False,
        dropna=False,
    )
    .agg(
        eligible_rows=("source_row_index", "size"),
        unique_timestamps=("timestamp_utc", "nunique"),
        start_utc=("timestamp_utc", "min"),
        end_utc=("timestamp_utc", "max"),
    )
)
event_summary_3l["duration_hours"] = (
    event_summary_3l["end_utc"] - event_summary_3l["start_utc"]
).dt.total_seconds() / 3600.0

ordered_eligible_3l = eligible_3l.sort_values(["event_key", "timestamp_utc"])
intervals_3l = ordered_eligible_3l.loc[
    :,
    ["farm", "asset_key", "event_key", "final_label", "timestamp_utc"],
].copy()
intervals_3l["interval_minutes"] = (
    intervals_3l.groupby("event_key", sort=False)["timestamp_utc"]
    .diff()
    .dt.total_seconds()
    .div(60.0)
)
intervals_3l = intervals_3l.loc[
    intervals_3l["interval_minutes"].gt(0)
    & intervals_3l["interval_minutes"].le(24 * 60)
].copy()

event_cadence_3l = (
    intervals_3l.groupby("event_key", as_index=False)
    .agg(
        median_interval_minutes=("interval_minutes", "median"),
        p95_interval_minutes=("interval_minutes", lambda x: x.quantile(0.95)),
        maximum_interval_minutes=("interval_minutes", "max"),
    )
)
event_summary_3l = event_summary_3l.merge(
    event_cadence_3l, on="event_key", how="left", validate="one_to_one"
)

asset_summary_3l = (
    eligible_3l.groupby(["farm", "asset_id", "asset_key"], as_index=False)
    .agg(
        eligible_rows=("event_key", "size"),
        eligible_events=("event_key", "nunique"),
        labels=("final_label", "nunique"),
        first_timestamp_utc=("timestamp_utc", "min"),
        last_timestamp_utc=("timestamp_utc", "max"),
    )
)

asset_label_rows_3l = (
    eligible_3l.groupby(["farm", "asset_key", "final_label"], as_index=False)
    .size()
    .rename(columns={"size": "eligible_rows"})
)
asset_label_events_3l = (
    eligible_3l.groupby(["farm", "asset_key", "final_label"], as_index=False)
    .agg(eligible_events=("event_key", "nunique"))
)

row_pivot_3l = asset_label_rows_3l.pivot_table(
    index=["farm", "asset_key"],
    columns="final_label",
    values="eligible_rows",
    aggfunc="sum",
    fill_value=0,
).reset_index()
for label_3l in LABELS_3L:
    if label_3l not in row_pivot_3l:
        row_pivot_3l[label_3l] = 0
row_pivot_3l["anomaly_to_normal_row_ratio"] = (
    row_pivot_3l["anomaly"] / row_pivot_3l["normal"].replace(0, np.nan)
)

event_pivot_3l = asset_label_events_3l.pivot_table(
    index=["farm", "asset_key"],
    columns="final_label",
    values="eligible_events",
    aggfunc="sum",
    fill_value=0,
).reset_index()
for label_3l in LABELS_3L:
    if label_3l not in event_pivot_3l:
        event_pivot_3l[label_3l] = 0
event_pivot_3l["anomaly_to_normal_event_ratio"] = (
    event_pivot_3l["anomaly"] / event_pivot_3l["normal"].replace(0, np.nan)
)

manifest_status_3l = pd.Series(
    "other_ineligible", index=manifest_3l.index, dtype="string"
)
manifest_status_3l.loc[manifest_3l["modeling_eligible"]] = "eligible"
if "same_label_duplicate_action" in manifest_3l:
    manifest_status_3l.loc[
        manifest_3l["same_label_duplicate_action"]
        .astype("string")
        .eq("redundant_exact_copy_excluded")
    ] = "exact_copy_excluded"
manifest_3l["manifest_status"] = manifest_status_3l

status_summary_3l = (
    manifest_3l.groupby(["farm", "manifest_status"], as_index=False)
    .size()
    .rename(columns={"size": "manifest_rows"})
)

farm_temporal_summary_3l = (
    eligible_3l.groupby("farm", as_index=False)
    .agg(
        eligible_rows=("event_key", "size"),
        assets=("asset_key", "nunique"),
        events=("event_key", "nunique"),
        first_timestamp_utc=("timestamp_utc", "min"),
        last_timestamp_utc=("timestamp_utc", "max"),
        active_days=("date_utc", "nunique"),
    )
)
farm_temporal_summary_3l["calendar_span_days"] = (
    farm_temporal_summary_3l["last_timestamp_utc"]
    - farm_temporal_summary_3l["first_timestamp_utc"]
).dt.total_seconds() / 86400.0

tables_to_save_3l = [
    ("T001", "eligible_label_summary", eligible_label_summary_3l,
     "Post-deduplication class support by farm."),
    ("T002", "event_summary", event_summary_3l,
     "Event size, time coverage, duration, and sampling cadence."),
    ("T003", "asset_summary", asset_summary_3l,
     "Asset-level data support and temporal coverage."),
    ("T004", "asset_label_rows", asset_label_rows_3l,
     "Eligible normal/anomaly row counts by asset."),
    ("T005", "asset_label_events", asset_label_events_3l,
     "Eligible normal/anomaly event counts by asset."),
    ("T006", "manifest_status_summary", status_summary_3l,
     "Manifest disposition counts after canonical retention."),
    ("T007", "farm_temporal_summary", farm_temporal_summary_3l,
     "Temporal coverage and support by farm."),
]

for table_spec_3l in tables_to_save_3l:
    save_table_3l(*table_spec_3l)

optional_tables_3l = [
    ("T008", "canonical_retention_summary",
     "EXACT_SAME_LABEL_CANONICAL_RETENTION_SUMMARY",
     "Canonical source retention and redundant source exclusion."),
    ("T009", "affected_event_summary",
     "POST_DEDUPLICATION_AFFECTED_EVENT_SUMMARY",
     "Manifest effect for the two audited overlapping events."),
    ("T010", "collision_relationship_counts",
     "SAME_LABEL_COLLISION_RELATIONSHIP_COUNTS",
     "Exact versus ambiguous same-label collision relationships."),
    ("T011", "measurement_vector_reuse_summary",
     "MEASUREMENT_VECTOR_REUSE_SUMMARY",
     "Measurement vectors recurring at different timestamps."),
]

for table_id_3l, name_3l, object_name_3l, purpose_3l in optional_tables_3l:
    if object_name_3l in globals():
        save_table_3l(table_id_3l, name_3l, globals()[object_name_3l], purpose_3l)


# -----------------------------------------------------------------------------
# 4. Manuscript/overview candidates (C001–C035)
# -----------------------------------------------------------------------------

grouped_bar_3l(
    eligible_label_summary_3l,
    "farm", "final_label", "eligible_rows",
    "C001", "eligible_rows_by_farm_label",
    "Post-deduplication eligible observations",
    "Eligible observations", "core",
    "Shows class support after verified duplicate exclusion.",
    "eligible_label_summary_3l",
)

grouped_bar_3l(
    eligible_label_summary_3l,
    "farm", "final_label", "eligible_rows",
    "C002", "class_composition_by_farm",
    "Within-farm class composition",
    "Share of eligible observations (%)", "core",
    "Shows farm-specific class imbalance without conflating farm size.",
    "eligible_label_summary_3l", normalize=True,
)

grouped_bar_3l(
    eligible_label_summary_3l,
    "farm", "final_label", "eligible_event_files",
    "C003", "eligible_events_by_farm_label",
    "Eligible event files by farm and class",
    "Event files", "core",
    "Shows independent event support for each class.",
    "eligible_label_summary_3l",
)

grouped_bar_3l(
    eligible_label_summary_3l,
    "farm", "final_label", "eligible_assets",
    "C004", "eligible_assets_by_farm_label",
    "Eligible physical assets by farm and class",
    "Physical assets", "core",
    "Shows the asset-level support available for leakage-safe evaluation.",
    "eligible_label_summary_3l",
)

overall_label_3l = (
    eligible_3l.groupby("final_label", as_index=False)
    .size()
    .rename(columns={"size": "eligible_rows"})
)
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.bar(
    overall_label_3l["final_label"],
    overall_label_3l["eligible_rows"],
    color=[COLORS_3L.get(x, "#777777") for x in overall_label_3l["final_label"]],
)
ax.set_title("Overall post-deduplication class support")
finish_axes_3l(ax, "Class", "Eligible observations", legend=False)
add_bar_labels_3l(ax)
fig.tight_layout()
save_figure_3l(
    fig, "C005", "overall_class_support", "Overall class support", "core",
    "Summarizes the final row-level class balance.", "eligible_3l"
)

event_groups_3l = []
event_group_labels_3l = []
event_group_colors_3l = []
for farm_3l in FARMS_3L:
    for label_3l in LABELS_3L:
        values_3l = event_summary_3l.loc[
            event_summary_3l["farm"].eq(farm_3l)
            & event_summary_3l["final_label"].eq(label_3l),
            "eligible_rows",
        ].to_numpy()
        if len(values_3l):
            event_groups_3l.append(values_3l)
            event_group_labels_3l.append(f"{farm_3l}\n{label_3l}")
            event_group_colors_3l.append(COLORS_3L[label_3l])
fig, ax = plt.subplots(figsize=(7.6, 4.5))
box_3l = ax.boxplot(event_groups_3l, tick_labels=event_group_labels_3l,
                    showfliers=False, patch_artist=True)
for patch_3l, color_3l in zip(box_3l["boxes"], event_group_colors_3l):
    patch_3l.set_facecolor(color_3l)
    patch_3l.set_alpha(0.75)
ax.set_title("Eligible observations per event")
finish_axes_3l(ax, "Farm and class", "Observations per event", legend=False)
fig.tight_layout()
save_figure_3l(
    fig, "C006", "event_size_distribution", "Event size distribution", "core",
    "Shows event-size heterogeneity across farms and classes.", "event_summary_3l"
)

horizontal_rank_3l(
    asset_summary_3l, "asset_key", "eligible_rows",
    "C007", "asset_row_support", "Eligible observations by asset",
    "Eligible observations", "core",
    "Identifies concentration of row support in particular assets.",
    "asset_summary_3l", top_n=35,
)

horizontal_rank_3l(
    asset_summary_3l, "asset_key", "eligible_events",
    "C008", "asset_event_support", "Eligible event files by asset",
    "Eligible event files", "core",
    "Identifies the number of independent events contributed by each asset.",
    "asset_summary_3l", top_n=35, color="#E69F00",
)

ratio_rows_3l = row_pivot_3l.dropna(subset=["anomaly_to_normal_row_ratio"]).copy()
horizontal_rank_3l(
    ratio_rows_3l, "asset_key", "anomaly_to_normal_row_ratio",
    "C009", "asset_row_class_ratio", "Anomaly-to-normal row ratio by asset",
    "Anomaly / normal observations", "core",
    "Shows asset-level class imbalance among assets with normal support.",
    "row_pivot_3l", top_n=35, color="#D55E00",
)

ratio_events_3l = event_pivot_3l.dropna(subset=["anomaly_to_normal_event_ratio"]).copy()
horizontal_rank_3l(
    ratio_events_3l, "asset_key", "anomaly_to_normal_event_ratio",
    "C010", "asset_event_class_ratio", "Anomaly-to-normal event ratio by asset",
    "Anomaly / normal events", "core",
    "Shows imbalance in independent event support rather than row volume.",
    "event_pivot_3l", top_n=35, color="#CC79A7",
)

duration_groups_3l = []
duration_labels_3l = []
duration_colors_3l = []
for farm_3l in FARMS_3L:
    for label_3l in LABELS_3L:
        values_3l = event_summary_3l.loc[
            event_summary_3l["farm"].eq(farm_3l)
            & event_summary_3l["final_label"].eq(label_3l),
            "duration_hours",
        ].dropna().to_numpy()
        if len(values_3l):
            duration_groups_3l.append(values_3l)
            duration_labels_3l.append(f"{farm_3l}\n{label_3l}")
            duration_colors_3l.append(COLORS_3L[label_3l])
fig, ax = plt.subplots(figsize=(7.6, 4.5))
box_3l = ax.boxplot(duration_groups_3l, tick_labels=duration_labels_3l,
                    showfliers=False, patch_artist=True)
for patch_3l, color_3l in zip(box_3l["boxes"], duration_colors_3l):
    patch_3l.set_facecolor(color_3l)
    patch_3l.set_alpha(0.75)
ax.set_title("Event duration by farm and class")
finish_axes_3l(ax, "Farm and class", "Duration (hours)", legend=False)
fig.tight_layout()
save_figure_3l(
    fig, "C011", "event_duration_distribution", "Event duration distribution", "core",
    "Shows whether event windows differ materially by source and class.", "event_summary_3l"
)

fig, ax = plt.subplots(figsize=(6.4, 4.5))
for label_3l in LABELS_3L:
    subset_3l = event_summary_3l.loc[event_summary_3l["final_label"].eq(label_3l)]
    ax.scatter(
        subset_3l["duration_hours"], subset_3l["eligible_rows"],
        s=34, alpha=0.72, color=COLORS_3L[label_3l], label=label_3l.title(),
    )
ax.set_title("Event duration and eligible row support")
finish_axes_3l(ax, "Duration (hours)", "Eligible observations")
fig.tight_layout()
save_figure_3l(
    fig, "C012", "event_duration_vs_rows", "Event duration versus rows", "core",
    "Checks whether event size is primarily explained by temporal duration.", "event_summary_3l"
)

cadence_summary_3l = (
    event_summary_3l.groupby(["farm", "final_label"], as_index=False)
    .agg(median_interval_minutes=("median_interval_minutes", "median"))
)
grouped_bar_3l(
    cadence_summary_3l,
    "farm", "final_label", "median_interval_minutes",
    "C013", "median_sampling_interval", "Median within-event sampling interval",
    "Median interval (minutes)", "core",
    "Verifies sampling cadence comparability across farms and classes.",
    "event_summary_3l",
)

cadence_groups_3l = []
cadence_labels_3l = []
cadence_colors_3l = []
for farm_3l in FARMS_3L:
    for label_3l in LABELS_3L:
        values_3l = event_summary_3l.loc[
            event_summary_3l["farm"].eq(farm_3l)
            & event_summary_3l["final_label"].eq(label_3l),
            "p95_interval_minutes",
        ].dropna().to_numpy()
        if len(values_3l):
            cadence_groups_3l.append(values_3l)
            cadence_labels_3l.append(f"{farm_3l}\n{label_3l}")
            cadence_colors_3l.append(COLORS_3L[label_3l])
fig, ax = plt.subplots(figsize=(7.6, 4.5))
box_3l = ax.boxplot(cadence_groups_3l, tick_labels=cadence_labels_3l,
                    showfliers=False, patch_artist=True)
for patch_3l, color_3l in zip(box_3l["boxes"], cadence_colors_3l):
    patch_3l.set_facecolor(color_3l)
    patch_3l.set_alpha(0.75)
ax.set_title("Event-level 95th-percentile sampling gaps")
finish_axes_3l(ax, "Farm and class", "Gap (minutes)", legend=False)
fig.tight_layout()
save_figure_3l(
    fig, "C014", "sampling_gap_distribution", "Sampling gap distribution", "core",
    "Shows long within-event sampling gaps that may affect window construction.",
    "event_summary_3l"
)

monthly_farm_3l = (
    eligible_3l.groupby(["month_utc", "farm"], as_index=False)
    .size().rename(columns={"size": "eligible_rows"})
)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for farm_3l in FARMS_3L:
    subset_3l = monthly_farm_3l.loc[monthly_farm_3l["farm"].eq(farm_3l)]
    ax.plot(subset_3l["month_utc"], subset_3l["eligible_rows"], marker="o",
            linewidth=1.5, markersize=3.5, color=COLORS_3L[farm_3l], label=f"Farm {farm_3l}")
ax.set_title("Monthly eligible observations by farm")
finish_axes_3l(ax, "Month (UTC)", "Eligible observations")
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.tight_layout()
save_figure_3l(
    fig, "C015", "monthly_rows_by_farm", "Monthly rows by farm", "core",
    "Shows temporal coverage and source-specific acquisition periods.", "eligible_3l"
)

monthly_label_3l = (
    eligible_3l.groupby(["month_utc", "final_label"], as_index=False)
    .size().rename(columns={"size": "eligible_rows"})
)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for label_3l in LABELS_3L:
    subset_3l = monthly_label_3l.loc[monthly_label_3l["final_label"].eq(label_3l)]
    ax.plot(subset_3l["month_utc"], subset_3l["eligible_rows"], marker="o",
            linewidth=1.5, markersize=3.5, color=COLORS_3L[label_3l], label=label_3l.title())
ax.set_title("Monthly eligible observations by class")
finish_axes_3l(ax, "Month (UTC)", "Eligible observations")
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.tight_layout()
save_figure_3l(
    fig, "C016", "monthly_rows_by_label", "Monthly rows by class", "core",
    "Shows whether class support is temporally separated.", "eligible_3l"
)

daily_label_3l = (
    eligible_3l.groupby(["date_utc", "final_label"], as_index=False)
    .size().rename(columns={"size": "eligible_rows"})
    .sort_values("date_utc")
)
daily_label_3l["cumulative_rows"] = (
    daily_label_3l.groupby("final_label")["eligible_rows"].cumsum()
)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for label_3l in LABELS_3L:
    subset_3l = daily_label_3l.loc[daily_label_3l["final_label"].eq(label_3l)]
    ax.plot(subset_3l["date_utc"], subset_3l["cumulative_rows"],
            linewidth=1.6, color=COLORS_3L[label_3l], label=label_3l.title())
ax.set_title("Cumulative eligible observations through time")
finish_axes_3l(ax, "Date (UTC)", "Cumulative observations")
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.tight_layout()
save_figure_3l(
    fig, "C017", "cumulative_rows_by_label", "Cumulative class support", "core",
    "Shows how class evidence accumulates over the observation horizon.", "eligible_3l"
)

hour_profile_3l = (
    eligible_3l.groupby(["hour_utc", "final_label"], as_index=False)
    .size().rename(columns={"size": "eligible_rows"})
)
fig, ax = plt.subplots(figsize=(7.8, 4.2))
for label_3l in LABELS_3L:
    subset_3l = hour_profile_3l.loc[hour_profile_3l["final_label"].eq(label_3l)]
    ax.plot(subset_3l["hour_utc"], subset_3l["eligible_rows"], marker="o",
            color=COLORS_3L[label_3l], label=label_3l.title())
ax.set_xticks(range(0, 24, 2))
ax.set_title("UTC hour-of-day sampling profile")
finish_axes_3l(ax, "Hour (UTC)", "Eligible observations")
fig.tight_layout()
save_figure_3l(
    fig, "C018", "hourly_sampling_profile", "Hourly sampling profile", "core",
    "Checks for class-dependent hour-of-day acquisition bias.", "eligible_3l"
)

weekday_order_3l = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_profile_3l = (
    eligible_3l.groupby(["weekday", "final_label"], as_index=False)
    .size().rename(columns={"size": "eligible_rows"})
)
weekday_profile_3l["weekday"] = pd.Categorical(
    weekday_profile_3l["weekday"], categories=weekday_order_3l, ordered=True
)
weekday_profile_3l = weekday_profile_3l.sort_values("weekday")
grouped_bar_3l(
    weekday_profile_3l,
    "weekday", "final_label", "eligible_rows",
    "C019", "weekday_sampling_profile", "UTC weekday sampling profile",
    "Eligible observations", "core",
    "Checks for class-dependent day-of-week acquisition bias.",
    "eligible_3l",
)

for number_3l, label_3l in [(20, "anomaly"), (21, "normal")]:
    heat_3l = (
        eligible_3l.loc[eligible_3l["final_label"].eq(label_3l)]
        .groupby(["weekday_number", "hour_utc"]).size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    heat_3l.index = weekday_order_3l
    matrix_figure_3l(
        heat_3l, f"C{number_3l:03d}", f"{label_3l}_weekday_hour_heatmap",
        f"{label_3l.title()} observations by weekday and UTC hour",
        "Hour (UTC)", "Weekday", "core",
        f"Shows the temporal acquisition footprint of {label_3l} observations.",
        "eligible_3l", cmap="magma" if label_3l == "anomaly" else "Blues",
    )

status_pivot_3l = status_summary_3l.pivot_table(
    index="farm", columns="manifest_status", values="manifest_rows",
    aggfunc="sum", fill_value=0,
)
status_order_3l = [
    x for x in ["eligible", "exact_copy_excluded", "other_ineligible"]
    if x in status_pivot_3l.columns
]
status_pivot_3l = status_pivot_3l.reindex(columns=status_order_3l)

fig, ax = plt.subplots(figsize=(7.2, 4.3))
status_pivot_3l.plot(
    kind="bar", stacked=True, ax=ax,
    color=[COLORS_3L.get(x, "#777777") for x in status_pivot_3l.columns],
)
ax.set_title("Manifest row disposition by farm")
ax.tick_params(axis="x", rotation=0)
finish_axes_3l(ax, "Farm", "Manifest rows")
fig.tight_layout()
save_figure_3l(
    fig, "C022", "manifest_status_counts", "Manifest disposition counts", "core",
    "Shows eligible, exact-copy-excluded, and other non-modeling rows.", "status_summary_3l"
)

status_share_3l = status_pivot_3l.div(status_pivot_3l.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(7.2, 4.3))
status_share_3l.plot(
    kind="bar", stacked=True, ax=ax,
    color=[COLORS_3L.get(x, "#777777") for x in status_share_3l.columns],
)
ax.set_title("Within-farm manifest disposition")
ax.tick_params(axis="x", rotation=0)
finish_axes_3l(ax, "Farm", "Share of manifest rows (%)")
fig.tight_layout()
save_figure_3l(
    fig, "C023", "manifest_status_shares", "Manifest disposition shares", "core",
    "Normalizes manifest decisions for comparison across differently sized farms.",
    "status_summary_3l"
)

exact_excluded_3l = int(
    manifest_3l["manifest_status"].eq("exact_copy_excluded").sum()
)
dedup_overall_3l = pd.DataFrame(
    {"stage": ["Before retention", "After retention"],
     "eligible_rows": [len(eligible_3l) + exact_excluded_3l, len(eligible_3l)]}
)
fig, ax = plt.subplots(figsize=(5.6, 4.1))
ax.bar(dedup_overall_3l["stage"], dedup_overall_3l["eligible_rows"],
       color=["#999999", COLORS_3L["eligible"]])
ax.set_title("Effect of verified canonical retention")
finish_axes_3l(ax, "Audit stage", "Eligible observations", legend=False)
add_bar_labels_3l(ax)
fig.tight_layout()
save_figure_3l(
    fig, "C024", "dedup_before_after", "Canonical-retention effect", "core",
    "Shows the exact 1,009-row eligibility reduction without deleting data.",
    "manifest_3l"
)

dedup_farm_3l = (
    eligible_3l.groupby("farm").size().rename("after_retention").to_frame()
)
excluded_by_farm_3l = (
    manifest_3l.loc[manifest_3l["manifest_status"].eq("exact_copy_excluded")]
    .groupby("farm").size().rename("excluded_exact_copies")
)
dedup_farm_3l = dedup_farm_3l.join(excluded_by_farm_3l, how="left").fillna(0)
dedup_farm_3l["before_retention"] = (
    dedup_farm_3l["after_retention"] + dedup_farm_3l["excluded_exact_copies"]
)
fig, ax = plt.subplots(figsize=(7.0, 4.2))
dedup_farm_3l[["before_retention", "after_retention"]].plot(
    kind="bar", ax=ax, color=["#999999", COLORS_3L["eligible"]]
)
ax.set_title("Canonical-retention effect by farm")
ax.tick_params(axis="x", rotation=0)
finish_axes_3l(ax, "Farm", "Eligible observations")
add_bar_labels_3l(ax)
fig.tight_layout()
save_figure_3l(
    fig, "C025", "dedup_by_farm", "Canonical retention by farm", "core",
    "Localizes the exact-copy exclusion to its source farm.", "manifest_3l"
)

excluded_event_3l = (
    manifest_3l.loc[manifest_3l["manifest_status"].eq("exact_copy_excluded")]
    .groupby("event_key", as_index=False).size()
    .rename(columns={"size": "excluded_rows"})
)
horizontal_rank_3l(
    excluded_event_3l, "event_key", "excluded_rows",
    "C026", "excluded_rows_by_event", "Verified exact-copy exclusions by event",
    "Excluded rows", "core",
    "Identifies the redundant source event affected by canonical retention.",
    "manifest_3l", top_n=20, color=COLORS_3L["exact_copy_excluded"],
)

if "EXACT_SAME_LABEL_CANONICAL_RETENTION_AUDIT" in globals():
    retention_3l = EXACT_SAME_LABEL_CANONICAL_RETENTION_AUDIT.copy()
    retention_3l["timestamp_utc"] = pd.to_datetime(
        retention_3l["timestamp_utc"], errors="coerce", utc=True
    )
    retention_3l["date_utc"] = retention_3l["timestamp_utc"].dt.floor("D")
    retention_daily_3l = (
        retention_3l.groupby("date_utc", as_index=False).size()
        .rename(columns={"size": "duplicate_pairs"})
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(retention_daily_3l["date_utc"], retention_daily_3l["duplicate_pairs"],
            marker="o", color=COLORS_3L["exact_copy_excluded"])
    ax.set_title("Temporal extent of verified exact source copies")
    finish_axes_3l(ax, "Date (UTC)", "Duplicate timestamp pairs", legend=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    save_figure_3l(
        fig, "C027", "duplicate_overlap_timeline", "Duplicate overlap timeline", "core",
        "Shows the seven-day interval covered by verified exact copies.",
        "EXACT_SAME_LABEL_CANONICAL_RETENTION_AUDIT"
    )

    fig, ax = plt.subplots(figsize=(5.7, 5.0))
    ax.scatter(
        retention_3l["retained_source_row_index"],
        retention_3l["excluded_source_row_index"],
        s=10, alpha=0.55, color=COLORS_3L["exact_copy_excluded"],
    )
    ax.set_title("One-to-one retained and excluded source-row mapping")
    finish_axes_3l(ax, "Retained source-row index", "Excluded source-row index", legend=False)
    fig.tight_layout()
    save_figure_3l(
        fig, "C028", "retained_excluded_row_mapping", "Canonical row mapping", "core",
        "Demonstrates deterministic one-to-one source-row retention.",
        "EXACT_SAME_LABEL_CANONICAL_RETENTION_AUDIT"
    )

event_order_3l = event_summary_3l.sort_values(["farm", "start_utc"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.0, max(6.0, len(event_order_3l) * 0.11 + 1.8)))
for y_3l, row_3l in event_order_3l.iterrows():
    ax.plot([row_3l["start_utc"], row_3l["end_utc"]], [y_3l, y_3l],
            linewidth=2.2, color=COLORS_3L[row_3l["final_label"]])
ax.set_yticks(range(len(event_order_3l)))
ax.set_yticklabels(event_order_3l["event_key"], fontsize=5.5)
ax.set_title("Eligible event temporal coverage")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Event file")
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.tight_layout()
save_figure_3l(
    fig, "C029", "event_temporal_coverage", "Event temporal coverage", "core",
    "Shows separation and overlap among eligible event windows.", "event_summary_3l"
)

horizontal_rank_3l(
    event_summary_3l, "event_key", "duration_hours",
    "C030", "longest_events", "Longest eligible event windows",
    "Duration (hours)", "core",
    "Identifies events with the widest temporal support.", "event_summary_3l",
    top_n=20, color="#56B4E9",
)

horizontal_rank_3l(
    event_summary_3l, "event_key", "eligible_rows",
    "C031", "largest_events", "Largest eligible event files",
    "Eligible observations", "core",
    "Identifies events dominating the row-level dataset.", "event_summary_3l",
    top_n=25, color="#E69F00",
)

asset_row_matrix_3l = asset_label_rows_3l.pivot_table(
    index="asset_key", columns="final_label", values="eligible_rows",
    aggfunc="sum", fill_value=0,
).reindex(columns=LABELS_3L, fill_value=0)
matrix_figure_3l(
    asset_row_matrix_3l, "C032", "asset_label_row_heatmap",
    "Eligible observations by asset and class", "Class", "Asset",
    "core", "Shows asset-by-class row support and missing class combinations.",
    "asset_label_rows_3l", cmap="viridis",
)

asset_event_matrix_3l = asset_label_events_3l.pivot_table(
    index="asset_key", columns="final_label", values="eligible_events",
    aggfunc="sum", fill_value=0,
).reindex(columns=LABELS_3L, fill_value=0)
matrix_figure_3l(
    asset_event_matrix_3l, "C033", "asset_label_event_heatmap",
    "Eligible event files by asset and class", "Class", "Asset",
    "core", "Shows independent event support for every asset-class combination.",
    "asset_label_events_3l", cmap="cividis",
)

monthly_farm_matrix_3l = monthly_farm_3l.pivot_table(
    index="farm", columns="month_utc", values="eligible_rows",
    aggfunc="sum", fill_value=0,
)
monthly_farm_matrix_3l.columns = [x.strftime("%Y-%m") for x in monthly_farm_matrix_3l.columns]
matrix_figure_3l(
    monthly_farm_matrix_3l, "C034", "farm_month_activity_heatmap",
    "Monthly acquisition footprint by farm", "Month (UTC)", "Farm",
    "core", "Shows whether farms occupy distinct calendar periods.",
    "monthly_farm_3l", cmap="Blues",
)

fig, ax = plt.subplots(figsize=(8.0, 3.8))
for y_3l, row_3l in farm_temporal_summary_3l.sort_values("farm").reset_index(drop=True).iterrows():
    ax.plot([row_3l["first_timestamp_utc"], row_3l["last_timestamp_utc"]],
            [y_3l, y_3l], linewidth=8, solid_capstyle="round",
            color=COLORS_3L[str(row_3l["farm"])])
ax.set_yticks(range(len(farm_temporal_summary_3l)))
ax.set_yticklabels([f"Farm {x}" for x in sorted(farm_temporal_summary_3l["farm"])])
ax.set_title("Farm-level temporal coverage")
ax.set_xlabel("Time (UTC)")
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.tight_layout()
save_figure_3l(
    fig, "C035", "farm_temporal_coverage", "Farm temporal coverage", "core",
    "Summarizes the observation horizon of each farm.", "farm_temporal_summary_3l"
)


# -----------------------------------------------------------------------------
# 5. Farm-specific supplementary figures (three per farm)
# -----------------------------------------------------------------------------

for farm_number_3l, farm_3l in enumerate(FARMS_3L, start=1):
    farm_events_3l = event_summary_3l.loc[
        event_summary_3l["farm"].eq(farm_3l)
    ].sort_values("start_utc").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(farm_events_3l) * 0.18 + 1.5)))
    for y_3l, row_3l in farm_events_3l.iterrows():
        ax.plot([row_3l["start_utc"], row_3l["end_utc"]], [y_3l, y_3l],
                linewidth=3, color=COLORS_3L[row_3l["final_label"]])
    ax.set_yticks(range(len(farm_events_3l)))
    ax.set_yticklabels(farm_events_3l["event_key"], fontsize=6)
    ax.set_title(f"Farm {farm_3l}: eligible event timeline")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Event file")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    save_figure_3l(
        fig, f"F{farm_number_3l:02d}01", f"farm_{farm_3l}_event_timeline",
        f"Farm {farm_3l} event timeline", "supplement",
        "Shows within-farm event ordering, overlap, and class identity.", "event_summary_3l"
    )

    farm_monthly_3l = (
        eligible_3l.loc[eligible_3l["farm"].eq(farm_3l)]
        .groupby(["month_utc", "final_label"], as_index=False).size()
        .rename(columns={"size": "eligible_rows"})
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for label_3l in LABELS_3L:
        subset_3l = farm_monthly_3l.loc[farm_monthly_3l["final_label"].eq(label_3l)]
        ax.plot(subset_3l["month_utc"], subset_3l["eligible_rows"], marker="o",
                color=COLORS_3L[label_3l], label=label_3l.title())
    ax.set_title(f"Farm {farm_3l}: monthly class support")
    finish_axes_3l(ax, "Month (UTC)", "Eligible observations")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    save_figure_3l(
        fig, f"F{farm_number_3l:02d}02", f"farm_{farm_3l}_monthly_labels",
        f"Farm {farm_3l} monthly class support", "supplement",
        "Shows temporal separation of normal and anomaly evidence within the farm.",
        "eligible_3l"
    )

    farm_assets_3l = asset_summary_3l.loc[
        asset_summary_3l["farm"].eq(farm_3l)
    ]
    horizontal_rank_3l(
        farm_assets_3l, "asset_key", "eligible_rows",
        f"F{farm_number_3l:02d}03", f"farm_{farm_3l}_asset_ranking",
        f"Farm {farm_3l}: eligible observations by asset",
        "Eligible observations", "supplement",
        "Shows within-farm concentration of evidence by asset.",
        "asset_summary_3l", top_n=50, color=COLORS_3L[farm_3l],
    )


# -----------------------------------------------------------------------------
# 6. One supplementary activity timeline per physical asset
# -----------------------------------------------------------------------------

asset_keys_3l = sorted(eligible_3l["asset_key"].astype(str).unique())

for asset_number_3l, asset_key_3l in enumerate(asset_keys_3l, start=1):
    asset_daily_3l = (
        eligible_3l.loc[eligible_3l["asset_key"].eq(asset_key_3l)]
        .groupby(["date_utc", "final_label"], as_index=False).size()
        .rename(columns={"size": "eligible_rows"})
    )
    farm_3l = str(
        eligible_3l.loc[eligible_3l["asset_key"].eq(asset_key_3l), "farm"].iloc[0]
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for label_3l in LABELS_3L:
        subset_3l = asset_daily_3l.loc[asset_daily_3l["final_label"].eq(label_3l)]
        if len(subset_3l):
            ax.plot(
                subset_3l["date_utc"], subset_3l["eligible_rows"],
                marker="o", markersize=3, linewidth=1.3,
                color=COLORS_3L[label_3l], label=label_3l.title(),
            )
    ax.set_title(f"{asset_key_3l}: daily eligible observations")
    finish_axes_3l(ax, "Date (UTC)", "Eligible observations")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    save_figure_3l(
        fig, f"A{asset_number_3l:03d}", f"{asset_key_3l}_daily_activity",
        f"{asset_key_3l} daily activity", "supplement",
        "Shows asset-specific temporal and class coverage used for grouped splitting.",
        "eligible_3l"
    )


# If a future reduced dataset has fewer assets, add event diagnostics until the
# suite still contains at least 55 logical figures.
event_fallback_number_3l = 0
for event_key_3l in event_summary_3l["event_key"].astype(str):
    if len(FIGURE_REGISTRY_3L) >= 55:
        break
    event_fallback_number_3l += 1
    event_rows_3l = eligible_3l.loc[eligible_3l["event_key"].eq(event_key_3l)]
    event_daily_3l = (
        event_rows_3l.groupby("date_utc", as_index=False).size()
        .rename(columns={"size": "eligible_rows"})
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(event_daily_3l["date_utc"], event_daily_3l["eligible_rows"],
            marker="o", color="#56B4E9")
    ax.set_title(f"{event_key_3l}: daily eligible observations")
    finish_axes_3l(ax, "Date (UTC)", "Eligible observations", legend=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    save_figure_3l(
        fig, f"E{event_fallback_number_3l:03d}", f"{event_key_3l}_daily_activity",
        f"{event_key_3l} daily activity", "supplement",
        "Provides event-level temporal support when the asset count is small.",
        "eligible_3l"
    )


# -----------------------------------------------------------------------------
# 7. Registries, workbook, and non-destructive final checks
# -----------------------------------------------------------------------------

PAPER_FIGURE_REGISTRY_3L = pd.DataFrame(FIGURE_REGISTRY_3L)
PAPER_TABLE_REGISTRY_3L = pd.DataFrame(TABLE_REGISTRY_3L)

if PAPER_FIGURE_REGISTRY_3L["figure_id"].duplicated().any():
    duplicates_3l = PAPER_FIGURE_REGISTRY_3L.loc[
        PAPER_FIGURE_REGISTRY_3L["figure_id"].duplicated(keep=False),
        "figure_id",
    ].tolist()
    raise ValueError(f"Duplicate figure identifiers generated: {duplicates_3l}")

if len(PAPER_FIGURE_REGISTRY_3L) < 50:
    raise ValueError(
        f"Only {len(PAPER_FIGURE_REGISTRY_3L)} figures were generated; "
        "the Cell 3L minimum is 50."
    )

registry_csv_3l = OUTPUT_ROOT_3L / "paper_figure_registry.csv"
table_registry_csv_3l = OUTPUT_ROOT_3L / "paper_table_registry.csv"
PAPER_FIGURE_REGISTRY_3L.to_csv(registry_csv_3l, index=False)
PAPER_TABLE_REGISTRY_3L.to_csv(table_registry_csv_3l, index=False)

try:
    workbook_path_3l = OUTPUT_ROOT_3L / "paper_tables_3l.xlsx"

    def excel_safe_3l(frame):
        output_3l = frame.copy()
        for column_3l in output_3l.columns:
            if isinstance(output_3l[column_3l].dtype, pd.DatetimeTZDtype):
                output_3l[column_3l] = output_3l[column_3l].dt.tz_convert(None)
        return output_3l

    with pd.ExcelWriter(workbook_path_3l) as writer_3l:
        excel_safe_3l(eligible_label_summary_3l).to_excel(
            writer_3l, sheet_name="eligible_by_farm", index=False
        )
        excel_safe_3l(event_summary_3l).to_excel(
            writer_3l, sheet_name="event_summary", index=False
        )
        excel_safe_3l(asset_summary_3l).to_excel(
            writer_3l, sheet_name="asset_summary", index=False
        )
        excel_safe_3l(status_summary_3l).to_excel(
            writer_3l, sheet_name="manifest_status", index=False
        )
        excel_safe_3l(farm_temporal_summary_3l).to_excel(
            writer_3l, sheet_name="farm_temporal", index=False
        )
        PAPER_FIGURE_REGISTRY_3L.to_excel(
            writer_3l, sheet_name="figure_registry", index=False
        )
        PAPER_TABLE_REGISTRY_3L.to_excel(
            writer_3l, sheet_name="table_registry", index=False
        )
except Exception as exc:
    workbook_path_3l = None
    print(f"Excel workbook export skipped: {exc}")

if int(ROW_LABEL_MANIFEST["modeling_eligible"].sum()) != EXPECTED_ELIGIBLE_3L:
    raise ValueError("Cell 3L unexpectedly changed manifest eligibility.")
if ROW_LABEL_MANIFEST["split_assignment"].notna().any():
    raise ValueError("Cell 3L unexpectedly assigned a split.")

print("\nCell 3L completed successfully.")
print("Logical figures generated:", len(PAPER_FIGURE_REGISTRY_3L))
print("Core/overview figures:", int(PAPER_FIGURE_REGISTRY_3L["tier"].eq("core").sum()))
print("Supplementary figures:", int(PAPER_FIGURE_REGISTRY_3L["tier"].eq("supplement").sum()))
print("Physical assets with individual timelines:", len(asset_keys_3l))
print("Paper tables generated:", len(PAPER_TABLE_REGISTRY_3L))
print("Eligible rows visualized:", len(eligible_3l))
print("Figure registry:", registry_csv_3l)
print("Table registry:", table_registry_csv_3l)
if workbook_path_3l is not None:
    print("Combined table workbook:", workbook_path_3l)
print("Assigned train/validation/test rows:", int(ROW_LABEL_MANIFEST["split_assignment"].notna().sum()))
print("\nNo manifest row, label, eligibility decision, or split assignment was modified.")

print("\nRecommended manuscript candidates:")
print(
    PAPER_FIGURE_REGISTRY_3L.loc[
        PAPER_FIGURE_REGISTRY_3L["tier"].eq("core"),
        ["figure_id", "title", "claim_supported"],
    ].head(12).to_string(index=False)
)
