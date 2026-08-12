# ============================================================================
# CELL 8 — LEAKAGE-SAFE LAZY MULTISCALE WINDOW INDEX
#
# Prerequisites from the successful Cells 3–7:
#   - FARM_SCHEMAS
#   - OUTPUT_ROOT / TABLE_DIR
#   - iter_care_csv_chunks
#   - contiguous_segment_registry
#   - case_split_registry / asset_split_assignment
#   - cell_6_assignment_digest / assignment_digest_sha256
#   - train_preprocessing_signal_parameters
#   - train_preprocessing_farm_summary
#   - train_source_partition_audit
#   - cell_7_manifest
#
# This cell:
#   - verifies the frozen, asset-disjoint Cell 6 assignment
#   - verifies the final source-train-only Cell 7 preprocessing fit
#   - reads all 95 cases once in chunks without rewriting source data
#   - builds compact source-row window references instead of overlapping tensors
#   - uses a 24-hour 10-minute short context (144 points)
#   - uses a 7-day hourly long context (168 points)
#   - predicts a target exactly 1 hour (6 source rows) ahead
#   - keeps every context and target inside one Cell 5 gap-safe segment
#   - uses a 1-hour stride for training and a 10-minute stride for inference
#   - keeps status_type_id and event labels as audit metadata only
#   - provides a farm-specific lazy transform using Cell 7 parameters
#
# Training-index policy:
#   - model split must be train
#   - the complete 7-day source span and forecast target must be in CARE's
#     source-training partition
#   - the target must have a CARE-normal status (0 or 2)
#   - the context endpoint and target must be operationally usable
#   - the target must not fall inside a labeled anomaly interval
#
# Validation/test-index policy:
#   - model split must be validation or test
#   - the forecast target must be in CARE's source-prediction partition
#   - abnormal statuses are retained as metadata for later CARE-compatible
#     scoring; status values never become model inputs
#   - the context endpoint and target must be operationally usable
#
# This cell DOES NOT fit/refit preprocessing, interpolate across gaps, convert
# zeros to missing, clip signals, materialize model tensors, choose/tune a
# model or threshold, or evaluate the test split.
# ============================================================================


# ----------------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------------

from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# ----------------------------------------------------------------------------
# 2. Prerequisites, frozen protocol, and output paths
# ----------------------------------------------------------------------------

REQUIRED_CELL_8_OBJECTS = (
    "FARM_SCHEMAS",
    "OUTPUT_ROOT",
    "iter_care_csv_chunks",
    "contiguous_segment_registry",
    "case_split_registry",
    "asset_split_assignment",
    "cell_6_assignment_digest",
    "assignment_digest_sha256",
    "train_preprocessing_signal_parameters",
    "train_preprocessing_farm_summary",
    "train_source_partition_audit",
    "cell_7_manifest",
)

missing_cell_8_objects = [
    object_name
    for object_name in REQUIRED_CELL_8_OBJECTS
    if object_name not in globals()
]

if missing_cell_8_objects:
    raise RuntimeError(
        "Run the successful Cells 3–7 before Cell 8. Missing objects: "
        + ", ".join(missing_cell_8_objects)
    )

OUTPUT_ROOT = Path(OUTPUT_ROOT)

if "TABLE_DIR" not in globals():
    TABLE_DIR = OUTPUT_ROOT / "tables"
else:
    TABLE_DIR = Path(TABLE_DIR)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL_CASES = 95
EXPECTED_CANONICAL_ASSETS = 36
EXPECTED_CONTIGUOUS_SEGMENTS = 682
EXPECTED_PRIMARY_SIGNAL_PARAMETERS = 347
EXPECTED_KEPT_SIGNALS = 347
EXPECTED_MISSING_INDICATORS = 210
EXPECTED_SCALE_FALLBACK_SIGNALS = 9
EXPECTED_SOURCE_TRAIN_STATUS_EXCLUSIONS = 372_769
EXPECTED_SOURCE_PREDICTION_EXCLUSIONS = 193_952
EXPECTED_SPLIT_CASE_COUNTS = {
    "train": 67,
    "validation": 14,
    "test": 14,
}
EXPECTED_SPLIT_ASSET_COUNTS = {
    "train": 25,
    "validation": 6,
    "test": 5,
}

FROZEN_ASSIGNMENT_SHA256 = (
    "30cf8c5d10db81e2d730742908230c65f76654f1058e21d1f17989cc96ce9e27"
)

CELL_8_CHUNK_ROWS = 20_000
EXPECTED_SAMPLING_MINUTES = 10
EXPECTED_SAMPLING_INTERVAL = pd.Timedelta(
    minutes=EXPECTED_SAMPLING_MINUTES
)

SHORT_CONTEXT_STEPS = 144
SHORT_CONTEXT_HOURS = 24
LONG_CONTEXT_STEPS = 168
LONG_CONTEXT_SAMPLE_EVERY_SOURCE_ROWS = 6
LONG_CONTEXT_HOURS = 168
LONG_CONTEXT_BACKSTEPS = (
    (LONG_CONTEXT_STEPS - 1)
    * LONG_CONTEXT_SAMPLE_EVERY_SOURCE_ROWS
)
LONG_CONTEXT_SOURCE_SPAN_ROWS = LONG_CONTEXT_BACKSTEPS + 1
FORECAST_HORIZON_STEPS = 6
FORECAST_HORIZON_HOURS = 1
TRAINING_STRIDE_STEPS = 6
INFERENCE_STRIDE_STEPS = 1

NORMAL_STATUS_TYPE_IDS = frozenset({"0", "2"})
SOURCE_TRAIN_PARTITION_LABELS = frozenset(
    {"train", "training", "0"}
)
SOURCE_PREDICTION_PARTITION_LABELS = frozenset(
    {"test", "prediction", "predict", "1"}
)

CELL_8_POLICY = {
    "frozen_assignment_sha256": FROZEN_ASSIGNMENT_SHA256,
    "short_context": {
        "source_resolution_minutes": EXPECTED_SAMPLING_MINUTES,
        "steps": SHORT_CONTEXT_STEPS,
        "nominal_hours": SHORT_CONTEXT_HOURS,
        "inclusive_source_row_span": SHORT_CONTEXT_STEPS,
    },
    "long_context": {
        "sample_resolution_minutes": 60,
        "steps": LONG_CONTEXT_STEPS,
        "nominal_hours": LONG_CONTEXT_HOURS,
        "source_row_step": LONG_CONTEXT_SAMPLE_EVERY_SOURCE_ROWS,
        "inclusive_source_row_span": LONG_CONTEXT_SOURCE_SPAN_ROWS,
    },
    "forecast_horizon": {
        "source_steps": FORECAST_HORIZON_STEPS,
        "hours": FORECAST_HORIZON_HOURS,
        "target_semantics": (
            "Point target exactly one hour after the final observed context "
            "row; event label is determined at that target timestamp."
        ),
    },
    "stride": {
        "train_source_steps": TRAINING_STRIDE_STEPS,
        "train_minutes": 60,
        "validation_source_steps": INFERENCE_STRIDE_STEPS,
        "validation_minutes": 10,
        "test_source_steps": INFERENCE_STRIDE_STEPS,
        "test_minutes": 10,
    },
    "segment_rule": (
        "Long context, short context, context endpoint, and forecast target "
        "must all remain within one Cell 5 segment whose adjacent timestamps "
        "advance by exactly ten minutes."
    ),
    "training_window_rule": (
        "Use model-split training cases only. The complete seven-day source "
        "span and forecast target must be source-training rows; the target "
        "must have CARE-normal status 0 or 2, be operationally usable, and "
        "lie outside every labeled anomaly interval."
    ),
    "inference_window_rule": (
        "Use validation/test cases only and require the forecast target to "
        "belong to the source-prediction partition. Context may legitimately "
        "reach backward into the source-training partition but never outside "
        "the same gap-safe segment."
    ),
    "endpoint_operational_rule": (
        "Both context endpoint and forecast target require at least one "
        "finite primary signal and must not be synchronously all-zero across "
        "all finite primary signals. All-zero rows inside earlier context "
        "positions are retained and explicitly counted as quality metadata."
    ),
    "status_policy": (
        "status_type_id is retained only as target/audit metadata. It is not "
        "a model feature. Training targets require status 0 or 2; held-out "
        "abnormal-status targets remain indexed so a later scoring cell can "
        "apply the official CARE evaluation mask transparently."
    ),
    "event_label_policy": (
        "For anomaly cases, a target is positive exactly when its timestamp "
        "is within the inclusive [event_start, event_end] interval. Normal "
        "cases are always point-labeled zero. Event fields are metadata only."
    ),
    "preprocessing_policy": (
        "All model features are transformed lazily with the frozen farm-" 
        "specific Cell 7 training medians and scales. No window tensor is "
        "materialized by this cell."
    ),
    "individual_zero_policy": (
        "Finite individual zeros remain observed values and are never "
        "reclassified as missing."
    ),
    "source_values_modified": False,
}


# ----------------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------------

def cell_8_json_safe(value: Any) -> Any:
    """Recursively convert scientific-Python objects to strict JSON."""

    if isinstance(value, dict):
        return {
            str(key): cell_8_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [cell_8_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def save_cell_8_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write a strict, human-readable JSON manifest."""

    with Path(destination).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file_handle:
        json.dump(
            cell_8_json_safe(payload),
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file_handle.write("\n")


def cell_8_slug(value: Any) -> str:
    """Create a deterministic lowercase identifier component."""

    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")
    return slug or "unknown"


def cell_8_value_label(value: Any) -> str:
    """Normalize scalar categorical values without collapsing IDs."""

    if pd.isna(value):
        return "<missing>"

    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)

        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))

    text = str(value).strip()
    return text if text else "<missing>"


def cell_8_source_partition_label(value: Any) -> str:
    """Map raw train_test values to train, prediction, or unknown."""

    raw_label = cell_8_value_label(value)
    normalized = re.sub(
        r"[^a-z0-9]+",
        "",
        raw_label.lower(),
    )

    if normalized in SOURCE_TRAIN_PARTITION_LABELS:
        return "train"

    if normalized in SOURCE_PREDICTION_PARTITION_LABELS:
        return "prediction"

    return "unknown"


def cell_8_stable_dataframe_digest(
    dataframe: pd.DataFrame,
    columns: list[str],
    sort_columns: list[str],
) -> str:
    """Hash selected table content using a deterministic JSON encoding."""

    normalized = (
        dataframe[columns]
        .sort_values(sort_columns, kind="stable")
        .reset_index(drop=True)
        .copy()
    )
    payload = normalized.to_json(
        orient="records",
        date_format="iso",
        date_unit="ns",
        double_precision=15,
        force_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cell_8_contiguous_window_count(
    mask: np.ndarray,
    start_positions_1_based: np.ndarray,
    end_positions_1_based: np.ndarray,
) -> np.ndarray:
    """Count true values in inclusive, one-based contiguous row ranges."""

    integer_mask = np.asarray(mask, dtype=np.int64)
    prefix = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(integer_mask)]
    )
    starts = np.asarray(start_positions_1_based, dtype=np.int64)
    ends = np.asarray(end_positions_1_based, dtype=np.int64)
    return prefix[ends] - prefix[starts - 1]


def cell_8_sampled_window_count(
    mask: np.ndarray,
    anchor_positions_1_based: np.ndarray,
    sample_count: int = LONG_CONTEXT_STEPS,
    source_step: int = LONG_CONTEXT_SAMPLE_EVERY_SOURCE_ROWS,
) -> np.ndarray:
    """Count true values at equally spaced samples ending at each anchor."""

    mask = np.asarray(mask, dtype=bool)
    anchors = np.asarray(anchor_positions_1_based, dtype=np.int64)
    result = np.zeros(anchors.size, dtype=np.int64)

    for residue in range(source_step):
        candidate_indices = np.flatnonzero(
            ((anchors - 1) % source_step) == residue
        )

        if candidate_indices.size == 0:
            continue

        source_positions = np.arange(
            residue + 1,
            mask.size + 1,
            source_step,
            dtype=np.int64,
        )
        sampled_values = mask[source_positions - 1].astype(np.int64)
        prefix = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(sampled_values),
            ]
        )
        anchor_indices_within_residue = (
            (anchors[candidate_indices] - (residue + 1))
            // source_step
        )
        start_indices_within_residue = (
            anchor_indices_within_residue - (sample_count - 1)
        )

        if (start_indices_within_residue < 0).any():
            raise RuntimeError(
                "A long-context sampled range begins before the case."
            )

        result[candidate_indices] = (
            prefix[anchor_indices_within_residue + 1]
            - prefix[start_indices_within_residue]
        )

    return result


def cell_8_primary_exclusion_counts(
    reasons: np.ndarray,
) -> dict[str, int]:
    """Return deterministic primary-reason counts for one case."""

    unique_values, counts = np.unique(
        reasons.astype(str),
        return_counts=True,
    )
    return {
        str(value): int(count)
        for value, count in zip(unique_values, counts)
    }


# ----------------------------------------------------------------------------
# 4. Validate frozen split, segment registry, and final Cell 7 fit
# ----------------------------------------------------------------------------

cell_8_validation_errors: list[str] = []
cell_8_validation_warnings: list[str] = []

required_case_columns = {
    "farm",
    "event_id",
    "event_type",
    "is_anomaly",
    "canonical_asset_key",
    "model_split",
    "file_path",
    "event_start",
    "event_end",
}
missing_case_columns = sorted(
    required_case_columns - set(case_split_registry.columns)
)

if missing_case_columns:
    cell_8_validation_errors.append(
        "case_split_registry lacks required columns: "
        + ", ".join(missing_case_columns)
    )

required_segment_columns = {
    "segment_id",
    "farm",
    "event_id",
    "segment_number",
    "source_row_start_1_based",
    "source_row_end_1_based",
    "row_count",
    "timestamp_start",
    "timestamp_end",
    "exact_10_minute_internal_grid",
}
missing_segment_columns = sorted(
    required_segment_columns - set(contiguous_segment_registry.columns)
)

if missing_segment_columns:
    cell_8_validation_errors.append(
        "contiguous_segment_registry lacks required columns: "
        + ", ".join(missing_segment_columns)
    )

required_parameter_columns = {
    "farm",
    "feature_index",
    "signal_name",
    "imputation_value",
    "center_value",
    "scale_value",
    "scale_statistic",
    "keep_for_model",
    "add_missing_indicator",
    "missing_indicator_name",
}
missing_parameter_columns = sorted(
    required_parameter_columns
    - set(train_preprocessing_signal_parameters.columns)
)

if missing_parameter_columns:
    cell_8_validation_errors.append(
        "train_preprocessing_signal_parameters lacks required columns: "
        + ", ".join(missing_parameter_columns)
    )

if not cell_8_validation_errors:
    observed_assignment_sha256_cell_8 = cell_6_assignment_digest(
        asset_split_assignment
    )

    if observed_assignment_sha256_cell_8 != FROZEN_ASSIGNMENT_SHA256:
        cell_8_validation_errors.append(
            "The recomputed Cell 6 assignment SHA-256 differs from the "
            "frozen value."
        )

    if str(assignment_digest_sha256) != FROZEN_ASSIGNMENT_SHA256:
        cell_8_validation_errors.append(
            "The in-memory assignment_digest_sha256 is not frozen Cell 6."
        )

    if str(cell_7_manifest.get("assignment_digest_sha256")) != (
        FROZEN_ASSIGNMENT_SHA256
    ):
        cell_8_validation_errors.append(
            "Cell 7 was not fitted from the frozen Cell 6 assignment."
        )

    if len(case_split_registry) != EXPECTED_TOTAL_CASES:
        cell_8_validation_errors.append(
            f"case_split_registry contains {len(case_split_registry)} cases; "
            f"expected {EXPECTED_TOTAL_CASES}."
        )

    if case_split_registry.duplicated(
        subset=["farm", "event_id"]
    ).any():
        cell_8_validation_errors.append(
            "case_split_registry contains duplicate farm/event_id keys."
        )

    observed_split_case_counts = (
        case_split_registry["model_split"]
        .astype("string")
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in EXPECTED_SPLIT_CASE_COUNTS.items():
        if int(observed_split_case_counts.get(split_name, 0)) != expected_count:
            cell_8_validation_errors.append(
                f"Split {split_name!r} does not contain {expected_count} cases."
            )

    if len(asset_split_assignment) != EXPECTED_CANONICAL_ASSETS:
        cell_8_validation_errors.append(
            f"asset_split_assignment contains {len(asset_split_assignment)} "
            f"assets; expected {EXPECTED_CANONICAL_ASSETS}."
        )

    observed_split_asset_counts = (
        asset_split_assignment["model_split"]
        .astype("string")
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in EXPECTED_SPLIT_ASSET_COUNTS.items():
        if int(observed_split_asset_counts.get(split_name, 0)) != expected_count:
            cell_8_validation_errors.append(
                f"Split {split_name!r} does not contain {expected_count} assets."
            )

    asset_split_memberships = (
        case_split_registry
        .groupby("canonical_asset_key", observed=False)["model_split"]
        .nunique()
    )

    if not asset_split_memberships.eq(1).all():
        cell_8_validation_errors.append(
            "At least one canonical asset occurs in multiple model splits."
        )

    if len(contiguous_segment_registry) != EXPECTED_CONTIGUOUS_SEGMENTS:
        cell_8_validation_errors.append(
            "The Cell 5 segment registry is not the verified 682-segment "
            "registry."
        )

    if contiguous_segment_registry["segment_id"].duplicated().any():
        cell_8_validation_errors.append(
            "contiguous_segment_registry contains duplicate segment IDs."
        )

    if not contiguous_segment_registry[
        "exact_10_minute_internal_grid"
    ].eq(True).all():
        cell_8_validation_errors.append(
            "At least one Cell 5 segment is not marked as an exact 10-minute "
            "internal grid."
        )

    if len(train_preprocessing_signal_parameters) != (
        EXPECTED_PRIMARY_SIGNAL_PARAMETERS
    ):
        cell_8_validation_errors.append(
            "Cell 7 does not contain the expected 347 signal parameters."
        )

    kept_signal_count_cell_8 = int(
        train_preprocessing_signal_parameters["keep_for_model"].eq(True).sum()
    )
    missing_indicator_count_cell_8 = int(
        train_preprocessing_signal_parameters[
            "add_missing_indicator"
        ].eq(True).sum()
    )
    scale_fallback_count_cell_8 = int(
        (
            train_preprocessing_signal_parameters["keep_for_model"].eq(True)
            & ~train_preprocessing_signal_parameters[
                "scale_statistic"
            ].eq("interquartile_range")
        ).sum()
    )

    if kept_signal_count_cell_8 != EXPECTED_KEPT_SIGNALS:
        cell_8_validation_errors.append(
            f"Cell 7 keeps {kept_signal_count_cell_8} signals; expected 347."
        )

    if missing_indicator_count_cell_8 != EXPECTED_MISSING_INDICATORS:
        cell_8_validation_errors.append(
            "Cell 7 is not the final fit: expected 210 missing indicators, "
            f"observed {missing_indicator_count_cell_8}."
        )

    if scale_fallback_count_cell_8 != EXPECTED_SCALE_FALLBACK_SIGNALS:
        cell_8_validation_errors.append(
            "Cell 7 is not the final fit: expected 9 scale fallbacks, "
            f"observed {scale_fallback_count_cell_8}."
        )

    if int(
        cell_7_manifest.get(
            "source_training_status_excluded_row_count",
            -1,
        )
    ) != EXPECTED_SOURCE_TRAIN_STATUS_EXCLUSIONS:
        cell_8_validation_errors.append(
            "Cell 7 source-training status exclusions do not equal 372,769."
        )

    if int(
        cell_7_manifest.get(
            "source_prediction_excluded_row_count",
            -1,
        )
    ) != EXPECTED_SOURCE_PREDICTION_EXCLUSIONS:
        cell_8_validation_errors.append(
            "Cell 7 source-prediction exclusions do not equal 193,952."
        )

    if cell_7_manifest.get("sensor_body_read_counts") != {
        "train": 67,
        "validation": 0,
        "test": 0,
    }:
        cell_8_validation_errors.append(
            "Cell 7 sensor-body read counts do not preserve the required "
            "train-only fitting boundary."
        )

    kept_parameters_for_validation = (
        train_preprocessing_signal_parameters.loc[
            train_preprocessing_signal_parameters[
                "keep_for_model"
            ].eq(True)
        ]
    )

    for numeric_column in (
        "imputation_value",
        "center_value",
        "scale_value",
    ):
        numeric_values = pd.to_numeric(
            kept_parameters_for_validation[numeric_column],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(numeric_values).all():
            cell_8_validation_errors.append(
                f"Kept Cell 7 parameter {numeric_column!r} contains a "
                "nonfinite value."
            )

    if (
        pd.to_numeric(
            kept_parameters_for_validation["scale_value"],
            errors="coerce",
        )
        .le(0)
        .any()
    ):
        cell_8_validation_errors.append(
            "At least one kept Cell 7 scale_value is not positive."
        )

    anomaly_cases = case_split_registry.loc[
        case_split_registry["is_anomaly"].eq(True)
    ]

    if anomaly_cases[["event_start", "event_end"]].isna().any().any():
        cell_8_validation_errors.append(
            "At least one anomaly case lacks event_start or event_end."
        )

if cell_8_validation_errors:
    raise RuntimeError(
        "CELL 8 INPUT VALIDATION FAILED:\n  - "
        + "\n  - ".join(cell_8_validation_errors)
    )


# Freeze a deterministic digest of the numerical transform contract.
cell_8_parameter_digest_columns = [
    "farm",
    "feature_index",
    "signal_name",
    "imputation_value",
    "center_value",
    "scale_value",
    "scale_statistic",
    "keep_for_model",
    "add_missing_indicator",
    "missing_indicator_name",
]
preprocessing_parameter_digest_sha256 = cell_8_stable_dataframe_digest(
    train_preprocessing_signal_parameters,
    columns=cell_8_parameter_digest_columns,
    sort_columns=["farm", "feature_index", "signal_name"],
)


# ----------------------------------------------------------------------------
# 5. Build the farm-specific lazy transform and feature schema
# ----------------------------------------------------------------------------

window_feature_schema_records: list[dict[str, Any]] = []
care_model_feature_names: dict[str, tuple[str, ...]] = {}
care_kept_primary_signal_names: dict[str, tuple[str, ...]] = {}

for farm_name in sorted(FARM_SCHEMAS):
    farm_parameters = (
        train_preprocessing_signal_parameters.loc[
            train_preprocessing_signal_parameters["farm"]
            .astype(str)
            .eq(farm_name)
            & train_preprocessing_signal_parameters[
                "keep_for_model"
            ].eq(True)
        ]
        .sort_values("feature_index", kind="stable")
        .reset_index(drop=True)
    )
    expected_primary_signals = tuple(
        FARM_SCHEMAS[farm_name]["primary_signal_columns"]
    )
    observed_primary_signals = tuple(
        farm_parameters["signal_name"].astype(str)
    )

    if observed_primary_signals != expected_primary_signals:
        raise RuntimeError(
            f"Cell 7 kept-signal order for {farm_name} differs from the "
            "Cell 3 primary-signal schema."
        )

    care_kept_primary_signal_names[farm_name] = observed_primary_signals
    model_feature_index = 0
    farm_model_feature_names: list[str] = []

    for parameter_row in farm_parameters.itertuples(index=False):
        model_feature_name = f"scaled__{parameter_row.signal_name}"
        farm_model_feature_names.append(model_feature_name)
        window_feature_schema_records.append(
            {
                "farm": farm_name,
                "model_feature_index": model_feature_index,
                "model_feature_name": model_feature_name,
                "feature_kind": "scaled_primary_signal",
                "source_signal_name": str(parameter_row.signal_name),
                "source_feature_index": int(parameter_row.feature_index),
                "imputation_value": float(parameter_row.imputation_value),
                "center_value": float(parameter_row.center_value),
                "scale_value": float(parameter_row.scale_value),
                "scale_statistic": str(parameter_row.scale_statistic),
                "indicator_train_derived": False,
                "status_or_label_feature": False,
            }
        )
        model_feature_index += 1

    indicator_parameters = farm_parameters.loc[
        farm_parameters["add_missing_indicator"].eq(True)
    ]

    for parameter_row in indicator_parameters.itertuples(index=False):
        model_feature_name = str(parameter_row.missing_indicator_name)
        farm_model_feature_names.append(model_feature_name)
        window_feature_schema_records.append(
            {
                "farm": farm_name,
                "model_feature_index": model_feature_index,
                "model_feature_name": model_feature_name,
                "feature_kind": "missing_indicator",
                "source_signal_name": str(parameter_row.signal_name),
                "source_feature_index": int(parameter_row.feature_index),
                "imputation_value": np.nan,
                "center_value": np.nan,
                "scale_value": np.nan,
                "scale_statistic": "binary_indicator",
                "indicator_train_derived": True,
                "status_or_label_feature": False,
            }
        )
        model_feature_index += 1

    care_model_feature_names[farm_name] = tuple(farm_model_feature_names)

care_window_feature_schema = (
    pd.DataFrame(window_feature_schema_records)
    .sort_values(["farm", "model_feature_index"], kind="stable")
    .reset_index(drop=True)
)

care_window_feature_farm_summary = (
    care_window_feature_schema
    .groupby("farm", as_index=False, sort=False)
    .agg(
        primary_feature_count=(
            "feature_kind",
            lambda values: int(
                values.eq("scaled_primary_signal").sum()
            ),
        ),
        missing_indicator_count=(
            "feature_kind",
            lambda values: int(values.eq("missing_indicator").sum()),
        ),
        model_feature_count=("model_feature_name", "size"),
    )
)


def transform_care_primary_frame(
    farm_name: str,
    primary_frame: pd.DataFrame,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """
    Lazily apply the frozen Cell 7 transform to one raw primary-signal frame.

    The input frame is not mutated. status_type_id, train_test, event labels,
    timestamps, and identifiers are intentionally absent from the model array.
    """

    if farm_name not in care_kept_primary_signal_names:
        raise KeyError(f"Unknown CARE farm: {farm_name!r}")

    signal_names = list(care_kept_primary_signal_names[farm_name])
    missing_columns = [
        signal_name
        for signal_name in signal_names
        if signal_name not in primary_frame.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Raw frame for {farm_name} lacks primary signals: "
            + ", ".join(missing_columns)
        )

    farm_parameters = (
        train_preprocessing_signal_parameters.loc[
            train_preprocessing_signal_parameters["farm"]
            .astype(str)
            .eq(farm_name)
            & train_preprocessing_signal_parameters[
                "keep_for_model"
            ].eq(True)
        ]
        .sort_values("feature_index", kind="stable")
        .reset_index(drop=True)
    )
    numeric_frame = primary_frame[signal_names].apply(
        pd.to_numeric,
        errors="coerce",
    )
    raw_values = numeric_frame.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    nonfinite_mask = ~np.isfinite(raw_values)
    imputation_values = farm_parameters[
        "imputation_value"
    ].to_numpy(dtype=np.float64)
    center_values = farm_parameters[
        "center_value"
    ].to_numpy(dtype=np.float64)
    scale_values = farm_parameters[
        "scale_value"
    ].to_numpy(dtype=np.float64)
    imputed_values = np.where(
        nonfinite_mask,
        imputation_values[None, :],
        raw_values,
    )
    scaled_values = (
        (imputed_values - center_values[None, :])
        / scale_values[None, :]
    )
    output_parts = [scaled_values]
    indicator_signal_mask = farm_parameters[
        "add_missing_indicator"
    ].to_numpy(dtype=bool)

    if indicator_signal_mask.any():
        output_parts.append(
            nonfinite_mask[:, indicator_signal_mask].astype(np.float64)
        )

    transformed = np.concatenate(output_parts, axis=1)

    if not np.isfinite(transformed).all():
        raise RuntimeError(
            f"The lazy Cell 7 transform produced nonfinite values for "
            f"{farm_name}."
        )

    expected_feature_count = len(care_model_feature_names[farm_name])

    if transformed.shape[1] != expected_feature_count:
        raise RuntimeError(
            f"Lazy transform produced {transformed.shape[1]} features for "
            f"{farm_name}; expected {expected_feature_count}."
        )

    return transformed.astype(output_dtype, copy=False)


# ----------------------------------------------------------------------------
# 6. Stream all cases and construct compact window records
# ----------------------------------------------------------------------------

case_window_audit_records: list[dict[str, Any]] = []
window_index_frames: list[pd.DataFrame] = []
cell_8_sensor_body_read_counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

ordered_case_registry_cell_8 = (
    case_split_registry
    .assign(
        model_split_text=case_split_registry[
            "model_split"
        ].astype("string")
    )
    .sort_values(
        ["model_split_text", "farm", "event_id"],
        kind="stable",
    )
    .drop(columns=["model_split_text"])
    .reset_index(drop=True)
)

print("=" * 80)
print("BUILDING LEAKAGE-SAFE CARE MULTISCALE WINDOW INDEX")
print("=" * 80)
print(f"Frozen assignment SHA-256 : {FROZEN_ASSIGNMENT_SHA256}")
print(
    "Short context             : 144 x 10-minute points (24 hours)"
)
print(
    "Long context              : 168 x hourly points (7 days)"
)
print("Forecast horizon          : 6 x 10-minute steps (1 hour)")
print("Train / inference stride  : 1 hour / 10 minutes")
print(
    "Windows are indexed lazily; no overlapping model tensors are written."
)

for case_number, registry_row in enumerate(
    ordered_case_registry_cell_8.itertuples(index=False),
    start=1,
):
    farm_name = str(registry_row.farm)
    event_id = int(registry_row.event_id)
    event_type = str(registry_row.event_type)
    is_anomaly = bool(registry_row.is_anomaly)
    canonical_asset_key = str(registry_row.canonical_asset_key)
    model_split = str(registry_row.model_split)
    file_path = Path(registry_row.file_path)
    event_start = pd.to_datetime(
        registry_row.event_start,
        errors="coerce",
    )
    event_end = pd.to_datetime(
        registry_row.event_end,
        errors="coerce",
    )
    primary_signal_columns = list(
        FARM_SCHEMAS[farm_name]["primary_signal_columns"]
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"CARE event file does not exist: {file_path}"
        )

    case_segments = (
        contiguous_segment_registry.loc[
            contiguous_segment_registry["farm"].astype(str).eq(farm_name)
            & pd.to_numeric(
                contiguous_segment_registry["event_id"],
                errors="coerce",
            ).eq(event_id)
        ]
        .sort_values("segment_number", kind="stable")
        .reset_index(drop=True)
    )

    if case_segments.empty:
        raise RuntimeError(
            f"Cell 5 registered no gap-safe segment for {farm_name} "
            f"event {event_id}."
        )

    print(
        f"  [{case_number:02d}/{len(ordered_case_registry_cell_8):02d}] "
        f"{model_split:<10} | {farm_name:<11} | event {event_id:<3} | "
        f"{len(case_segments):>2} segments"
    )

    requested_columns = [
        "id",
        "time_stamp",
        "train_test",
        "status_type_id",
        *primary_signal_columns,
    ]
    case_row_ids: list[np.ndarray] = []
    case_timestamps: list[np.ndarray] = []
    case_partition_labels: list[np.ndarray] = []
    case_status_labels: list[np.ndarray] = []
    case_status_normal_masks: list[np.ndarray] = []
    case_any_finite_masks: list[np.ndarray] = []
    case_synchronous_zero_masks: list[np.ndarray] = []
    case_chunk_count = 0

    for chunk in iter_care_csv_chunks(
        file_path,
        usecols=requested_columns,
        chunksize=CELL_8_CHUNK_ROWS,
    ):
        case_chunk_count += 1

        if len(chunk) == 0:
            continue

        numeric_signal_frame = chunk[
            primary_signal_columns
        ].apply(pd.to_numeric, errors="coerce")
        signal_values = numeric_signal_frame.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        finite_mask = np.isfinite(signal_values)
        zero_mask = finite_mask & (signal_values == 0.0)
        any_finite_mask = finite_mask.any(axis=1)
        synchronous_zero_mask = (
            any_finite_mask
            & np.all(~finite_mask | zero_mask, axis=1)
        )
        status_labels = (
            chunk["status_type_id"]
            .map(cell_8_value_label)
            .to_numpy(dtype=object)
        )
        partition_labels = (
            chunk["train_test"]
            .map(cell_8_source_partition_label)
            .to_numpy(dtype=object)
        )
        timestamps = pd.to_datetime(
            chunk["time_stamp"],
            errors="coerce",
        ).to_numpy(dtype="datetime64[ns]")
        row_ids = pd.to_numeric(
            chunk["id"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        case_row_ids.append(row_ids)
        case_timestamps.append(timestamps)
        case_partition_labels.append(partition_labels)
        case_status_labels.append(status_labels)
        case_status_normal_masks.append(
            np.isin(status_labels, tuple(NORMAL_STATUS_TYPE_IDS))
        )
        case_any_finite_masks.append(any_finite_mask)
        case_synchronous_zero_masks.append(synchronous_zero_mask)

        del numeric_signal_frame, signal_values, finite_mask, zero_mask

    cell_8_sensor_body_read_counts[model_split] += 1

    if not case_timestamps:
        raise RuntimeError(
            f"{farm_name} event {event_id} contains no data rows."
        )

    row_ids = np.concatenate(case_row_ids)
    timestamps = np.concatenate(case_timestamps)
    partition_labels = np.concatenate(case_partition_labels)
    status_labels = np.concatenate(case_status_labels)
    status_normal_mask = np.concatenate(case_status_normal_masks)
    any_finite_mask = np.concatenate(case_any_finite_masks)
    synchronous_zero_mask = np.concatenate(
        case_synchronous_zero_masks
    )
    source_training_mask = partition_labels == "train"
    source_prediction_mask = partition_labels == "prediction"
    source_unknown_mask = ~(
        source_training_mask | source_prediction_mask
    )
    case_row_count = int(timestamps.size)

    if source_unknown_mask.any():
        cell_8_validation_errors.append(
            f"{farm_name} event {event_id} contains "
            f"{int(source_unknown_mask.sum())} unrecognized train_test rows."
        )

    case_primary_reason_counts: dict[str, int] = {}
    case_structural_candidates = 0
    case_included_windows = 0
    case_positive_windows = 0
    case_target_normal_status_windows = 0
    case_target_abnormal_status_windows = 0
    case_segments_long_enough = 0

    for segment_row in case_segments.itertuples(index=False):
        segment_id = str(segment_row.segment_id)
        segment_number = int(segment_row.segment_number)
        segment_start_1_based = int(
            segment_row.source_row_start_1_based
        )
        segment_end_1_based = int(
            segment_row.source_row_end_1_based
        )
        segment_row_count = int(segment_row.row_count)

        if not (
            1 <= segment_start_1_based <= segment_end_1_based <= case_row_count
        ):
            raise RuntimeError(
                f"Segment {segment_id} has invalid source-row bounds."
            )

        if (
            segment_end_1_based - segment_start_1_based + 1
            != segment_row_count
        ):
            raise RuntimeError(
                f"Segment {segment_id} source-row bounds do not conserve its "
                "registered row count."
            )

        segment_timestamp_values = timestamps[
            segment_start_1_based - 1:segment_end_1_based
        ]

        if np.isnat(segment_timestamp_values).any():
            raise RuntimeError(
                f"Segment {segment_id} contains an invalid timestamp."
            )

        if segment_timestamp_values.size > 1:
            timestamp_differences = np.diff(segment_timestamp_values)

            if not np.all(
                timestamp_differences
                == np.timedelta64(EXPECTED_SAMPLING_MINUTES, "m")
            ):
                raise RuntimeError(
                    f"Segment {segment_id} no longer has an exact 10-minute "
                    "source-order grid."
                )

        registered_start_timestamp = pd.to_datetime(
            segment_row.timestamp_start,
            errors="coerce",
        )
        registered_end_timestamp = pd.to_datetime(
            segment_row.timestamp_end,
            errors="coerce",
        )

        if (
            pd.Timestamp(segment_timestamp_values[0])
            != registered_start_timestamp
            or pd.Timestamp(segment_timestamp_values[-1])
            != registered_end_timestamp
        ):
            raise RuntimeError(
                f"Segment {segment_id} timestamp boundaries differ from "
                "Cell 5."
            )

        first_anchor_1_based = (
            segment_start_1_based + LONG_CONTEXT_BACKSTEPS
        )
        last_anchor_1_based = (
            segment_end_1_based - FORECAST_HORIZON_STEPS
        )

        if first_anchor_1_based > last_anchor_1_based:
            continue

        case_segments_long_enough += 1
        stride_steps = (
            TRAINING_STRIDE_STEPS
            if model_split == "train"
            else INFERENCE_STRIDE_STEPS
        )
        anchor_positions = np.arange(
            first_anchor_1_based,
            last_anchor_1_based + 1,
            stride_steps,
            dtype=np.int64,
        )
        target_positions = (
            anchor_positions + FORECAST_HORIZON_STEPS
        )
        short_start_positions = (
            anchor_positions - (SHORT_CONTEXT_STEPS - 1)
        )
        long_start_positions = (
            anchor_positions - LONG_CONTEXT_BACKSTEPS
        )
        anchor_indices = anchor_positions - 1
        target_indices = target_positions - 1
        target_timestamps = pd.to_datetime(
            timestamps[target_indices]
        )
        anchor_timestamps = pd.to_datetime(
            timestamps[anchor_indices]
        )

        if not (
            target_timestamps - anchor_timestamps
            == pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
        ).all():
            raise RuntimeError(
                f"Segment {segment_id} generated a non-one-hour target."
            )

        case_structural_candidates += int(anchor_positions.size)
        reasons = np.full(
            anchor_positions.size,
            "included",
            dtype=object,
        )

        def assign_primary_reason(
            mask: np.ndarray,
            reason: str,
        ) -> None:
            eligible_for_reason = reasons == "included"
            reasons[eligible_for_reason & np.asarray(mask, dtype=bool)] = reason

        assign_primary_reason(
            ~any_finite_mask[anchor_indices],
            "context_endpoint_without_finite_primary_signal",
        )
        assign_primary_reason(
            synchronous_zero_mask[anchor_indices],
            "context_endpoint_synchronous_all_zero",
        )
        assign_primary_reason(
            ~any_finite_mask[target_indices],
            "forecast_target_without_finite_primary_signal",
        )
        assign_primary_reason(
            synchronous_zero_mask[target_indices],
            "forecast_target_synchronous_all_zero",
        )

        context_source_training_rows = cell_8_contiguous_window_count(
            source_training_mask,
            long_start_positions,
            anchor_positions,
        )

        if model_split == "train":
            assign_primary_reason(
                partition_labels[target_indices] != "train",
                "forecast_target_not_source_training",
            )
            assign_primary_reason(
                context_source_training_rows
                != LONG_CONTEXT_SOURCE_SPAN_ROWS,
                "training_context_not_entirely_source_training",
            )
            assign_primary_reason(
                ~status_normal_mask[target_indices],
                "training_target_status_not_care_normal",
            )
        else:
            assign_primary_reason(
                partition_labels[target_indices] != "prediction",
                "forecast_target_not_source_prediction",
            )

        if is_anomaly:
            target_event_labels = np.asarray(
                (target_timestamps >= event_start)
                & (target_timestamps <= event_end),
                dtype=bool,
            )
        else:
            target_event_labels = np.zeros(
                anchor_positions.size,
                dtype=bool,
            )

        if model_split == "train":
            assign_primary_reason(
                target_event_labels,
                "training_target_inside_labeled_anomaly_event",
            )

        reason_counts = cell_8_primary_exclusion_counts(reasons)

        for reason_name, reason_count in reason_counts.items():
            case_primary_reason_counts[reason_name] = (
                case_primary_reason_counts.get(reason_name, 0)
                + reason_count
            )

        include_mask = reasons == "included"

        if not include_mask.any():
            continue

        included_anchor_positions = anchor_positions[include_mask]
        included_target_positions = target_positions[include_mask]
        included_short_start_positions = short_start_positions[include_mask]
        included_long_start_positions = long_start_positions[include_mask]
        included_anchor_indices = anchor_indices[include_mask]
        included_target_indices = target_indices[include_mask]
        included_target_timestamps = target_timestamps[include_mask]
        included_anchor_timestamps = anchor_timestamps[include_mask]
        included_target_event_labels = target_event_labels[include_mask]
        included_context_training_rows = context_source_training_rows[
            include_mask
        ]

        short_zero_rows = cell_8_contiguous_window_count(
            synchronous_zero_mask,
            included_short_start_positions,
            included_anchor_positions,
        )
        short_no_finite_rows = cell_8_contiguous_window_count(
            ~any_finite_mask,
            included_short_start_positions,
            included_anchor_positions,
        )
        short_normal_status_rows = cell_8_contiguous_window_count(
            status_normal_mask,
            included_short_start_positions,
            included_anchor_positions,
        )
        long_zero_points = cell_8_sampled_window_count(
            synchronous_zero_mask,
            included_anchor_positions,
        )
        long_no_finite_points = cell_8_sampled_window_count(
            ~any_finite_mask,
            included_anchor_positions,
        )
        long_normal_status_points = cell_8_sampled_window_count(
            status_normal_mask,
            included_anchor_positions,
        )

        if is_anomaly and pd.notna(event_start) and pd.notna(event_end):
            hours_to_event_start = (
                event_start - included_target_timestamps
            ).total_seconds() / 3_600.0
            hours_to_event_end = (
                event_end - included_target_timestamps
            ).total_seconds() / 3_600.0
            event_duration_seconds = (
                event_end - event_start
            ).total_seconds()
            relative_event_position = np.full(
                included_anchor_positions.size,
                np.nan,
                dtype=np.float64,
            )

            if event_duration_seconds > 0:
                positive_indices = np.flatnonzero(
                    included_target_event_labels
                )
                relative_event_position[positive_indices] = (
                    (
                        included_target_timestamps[positive_indices]
                        - event_start
                    ).total_seconds()
                    / event_duration_seconds
                )
        else:
            hours_to_event_start = np.full(
                included_anchor_positions.size,
                np.nan,
            )
            hours_to_event_end = np.full(
                included_anchor_positions.size,
                np.nan,
            )
            relative_event_position = np.full(
                included_anchor_positions.size,
                np.nan,
            )

        farm_slug = cell_8_slug(farm_name)
        window_ids = [
            (
                f"{model_split}__{farm_slug}__event_{event_id:03d}__"
                f"segment_{segment_number:04d}__anchor_{int(anchor):08d}"
            )
            for anchor in included_anchor_positions
        ]
        window_role = (
            "normal_behavior_training"
            if model_split == "train"
            else f"{model_split}_inference"
        )
        included_status_normal = status_normal_mask[
            included_target_indices
        ]
        included_frame = pd.DataFrame(
            {
                "window_id": window_ids,
                "window_role": window_role,
                "model_split": model_split,
                "farm": farm_name,
                "event_id": event_id,
                "event_type": event_type,
                "case_is_anomaly": is_anomaly,
                "canonical_asset_key": canonical_asset_key,
                "segment_id": segment_id,
                "segment_source_row_start_1_based": segment_start_1_based,
                "segment_source_row_end_1_based": segment_end_1_based,
                "long_context_source_row_start_1_based": (
                    included_long_start_positions
                ),
                "short_context_source_row_start_1_based": (
                    included_short_start_positions
                ),
                "context_end_source_row_1_based": (
                    included_anchor_positions
                ),
                "forecast_target_source_row_1_based": (
                    included_target_positions
                ),
                "context_end_row_id": row_ids[
                    included_anchor_indices
                ],
                "forecast_target_row_id": row_ids[
                    included_target_indices
                ],
                "long_context_timestamp_start": pd.to_datetime(
                    timestamps[included_long_start_positions - 1]
                ),
                "short_context_timestamp_start": pd.to_datetime(
                    timestamps[included_short_start_positions - 1]
                ),
                "context_end_timestamp": included_anchor_timestamps,
                "forecast_target_timestamp": included_target_timestamps,
                "window_stride_steps": stride_steps,
                "context_source_training_fraction": (
                    included_context_training_rows
                    / LONG_CONTEXT_SOURCE_SPAN_ROWS
                ),
                "context_end_source_partition": partition_labels[
                    included_anchor_indices
                ],
                "forecast_target_source_partition": partition_labels[
                    included_target_indices
                ],
                "forecast_target_status_type_id": status_labels[
                    included_target_indices
                ],
                "forecast_target_care_normal_status": (
                    included_status_normal
                ),
                "short_context_synchronous_all_zero_rows": short_zero_rows,
                "short_context_no_finite_signal_rows": short_no_finite_rows,
                "short_context_care_normal_status_fraction": (
                    short_normal_status_rows / SHORT_CONTEXT_STEPS
                ),
                "long_context_synchronous_all_zero_points": long_zero_points,
                "long_context_no_finite_signal_points": long_no_finite_points,
                "long_context_care_normal_status_fraction": (
                    long_normal_status_points / LONG_CONTEXT_STEPS
                ),
                "forecast_target_event_label": (
                    included_target_event_labels.astype(np.int8)
                ),
                "hours_to_event_start": hours_to_event_start,
                "hours_to_event_end": hours_to_event_end,
                "relative_event_position": relative_event_position,
            }
        )
        window_index_frames.append(included_frame)
        included_window_count = int(len(included_frame))
        case_included_windows += included_window_count
        case_positive_windows += int(
            included_frame["forecast_target_event_label"].sum()
        )
        case_target_normal_status_windows += int(
            included_status_normal.sum()
        )
        case_target_abnormal_status_windows += int(
            (~included_status_normal).sum()
        )

    case_audit_record = {
        "farm": farm_name,
        "event_id": event_id,
        "event_type": event_type,
        "is_anomaly": is_anomaly,
        "canonical_asset_key": canonical_asset_key,
        "model_split": model_split,
        "file_path": str(file_path),
        "source_row_count": case_row_count,
        "chunk_count": case_chunk_count,
        "source_training_rows": int(source_training_mask.sum()),
        "source_prediction_rows": int(source_prediction_mask.sum()),
        "source_unknown_partition_rows": int(source_unknown_mask.sum()),
        "care_normal_status_rows": int(status_normal_mask.sum()),
        "care_abnormal_or_missing_status_rows": int(
            (~status_normal_mask).sum()
        ),
        "rows_without_finite_primary_signal": int(
            (~any_finite_mask).sum()
        ),
        "synchronous_all_zero_rows": int(
            synchronous_zero_mask.sum()
        ),
        "segment_count": int(len(case_segments)),
        "segments_long_enough": case_segments_long_enough,
        "structural_candidate_windows": case_structural_candidates,
        "included_windows": case_included_windows,
        "excluded_windows": (
            case_structural_candidates - case_included_windows
        ),
        "positive_event_windows": case_positive_windows,
        "negative_event_windows": (
            case_included_windows - case_positive_windows
        ),
        "target_care_normal_status_windows": (
            case_target_normal_status_windows
        ),
        "target_abnormal_or_missing_status_windows": (
            case_target_abnormal_status_windows
        ),
        "source_values_modified": False,
    }

    all_reason_names = (
        "included",
        "context_endpoint_without_finite_primary_signal",
        "context_endpoint_synchronous_all_zero",
        "forecast_target_without_finite_primary_signal",
        "forecast_target_synchronous_all_zero",
        "forecast_target_not_source_training",
        "forecast_target_not_source_prediction",
        "training_context_not_entirely_source_training",
        "training_target_status_not_care_normal",
        "training_target_inside_labeled_anomaly_event",
    )

    for reason_name in all_reason_names:
        case_audit_record[f"reason__{reason_name}"] = int(
            case_primary_reason_counts.get(reason_name, 0)
        )

    case_window_audit_records.append(case_audit_record)

    del (
        row_ids,
        timestamps,
        partition_labels,
        status_labels,
        status_normal_mask,
        any_finite_mask,
        synchronous_zero_mask,
        source_training_mask,
        source_prediction_mask,
        source_unknown_mask,
    )
    gc.collect()


# ----------------------------------------------------------------------------
# 7. Materialize compact indexes and summaries
# ----------------------------------------------------------------------------

if not window_index_frames:
    raise RuntimeError("CELL 8 produced no eligible window records.")

care_window_index_registry = (
    pd.concat(window_index_frames, ignore_index=True)
    .sort_values(
        [
            "model_split",
            "farm",
            "event_id",
            "forecast_target_source_row_1_based",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)
care_window_case_audit = (
    pd.DataFrame(case_window_audit_records)
    .sort_values(["model_split", "farm", "event_id"], kind="stable")
    .reset_index(drop=True)
)

train_window_index = care_window_index_registry.loc[
    care_window_index_registry["model_split"].eq("train")
].reset_index(drop=True)
validation_window_index = care_window_index_registry.loc[
    care_window_index_registry["model_split"].eq("validation")
].reset_index(drop=True)
test_window_index = care_window_index_registry.loc[
    care_window_index_registry["model_split"].eq("test")
].reset_index(drop=True)

care_window_split_summary = (
    care_window_index_registry
    .groupby("model_split", as_index=False, sort=False)
    .agg(
        window_count=("window_id", "size"),
        case_count=("event_id", lambda values: 0),
        canonical_asset_count=("canonical_asset_key", "nunique"),
        positive_event_windows=("forecast_target_event_label", "sum"),
        care_normal_status_targets=(
            "forecast_target_care_normal_status",
            "sum",
        ),
        abnormal_or_missing_status_targets=(
            "forecast_target_care_normal_status",
            lambda values: int((~values.astype(bool)).sum()),
        ),
    )
)

case_counts_by_split = (
    care_window_case_audit.loc[
        care_window_case_audit["included_windows"].gt(0)
    ]
    .groupby("model_split", observed=False)
    .size()
    .to_dict()
)
care_window_split_summary["case_count"] = (
    care_window_split_summary["model_split"]
    .map(case_counts_by_split)
    .fillna(0)
    .astype(int)
)
care_window_split_summary["negative_event_windows"] = (
    care_window_split_summary["window_count"]
    - care_window_split_summary["positive_event_windows"]
)
care_window_split_summary["positive_event_fraction"] = (
    care_window_split_summary["positive_event_windows"]
    / care_window_split_summary["window_count"]
)

care_window_farm_split_summary = (
    care_window_index_registry
    .groupby(["model_split", "farm"], as_index=False, sort=False)
    .agg(
        window_count=("window_id", "size"),
        case_count=("event_id", "nunique"),
        canonical_asset_count=("canonical_asset_key", "nunique"),
        positive_event_windows=("forecast_target_event_label", "sum"),
        care_normal_status_targets=(
            "forecast_target_care_normal_status",
            "sum",
        ),
        mean_short_context_all_zero_rows=(
            "short_context_synchronous_all_zero_rows",
            "mean",
        ),
        mean_long_context_all_zero_points=(
            "long_context_synchronous_all_zero_points",
            "mean",
        ),
    )
)

care_window_exclusion_summary = (
    care_window_case_audit
    .groupby("model_split", as_index=False, sort=False)
    .agg(
        structural_candidate_windows=(
            "structural_candidate_windows",
            "sum",
        ),
        included_windows=("included_windows", "sum"),
        excluded_windows=("excluded_windows", "sum"),
        cases_without_windows=(
            "included_windows",
            lambda values: int(values.eq(0).sum()),
        ),
        context_endpoint_without_finite_primary_signal=(
            "reason__context_endpoint_without_finite_primary_signal",
            "sum",
        ),
        context_endpoint_synchronous_all_zero=(
            "reason__context_endpoint_synchronous_all_zero",
            "sum",
        ),
        forecast_target_without_finite_primary_signal=(
            "reason__forecast_target_without_finite_primary_signal",
            "sum",
        ),
        forecast_target_synchronous_all_zero=(
            "reason__forecast_target_synchronous_all_zero",
            "sum",
        ),
        forecast_target_not_source_training=(
            "reason__forecast_target_not_source_training",
            "sum",
        ),
        forecast_target_not_source_prediction=(
            "reason__forecast_target_not_source_prediction",
            "sum",
        ),
        training_context_not_entirely_source_training=(
            "reason__training_context_not_entirely_source_training",
            "sum",
        ),
        training_target_status_not_care_normal=(
            "reason__training_target_status_not_care_normal",
            "sum",
        ),
        training_target_inside_labeled_anomaly_event=(
            "reason__training_target_inside_labeled_anomaly_event",
            "sum",
        ),
    )
)


# ----------------------------------------------------------------------------
# 8. Leakage, temporal, label, and conservation checks
# ----------------------------------------------------------------------------

if care_window_index_registry["window_id"].duplicated().any():
    cell_8_validation_errors.append(
        "care_window_index_registry contains duplicate window IDs."
    )

if len(care_window_case_audit) != EXPECTED_TOTAL_CASES:
    cell_8_validation_errors.append(
        "care_window_case_audit does not contain all 95 cases."
    )

if not (
    care_window_case_audit["structural_candidate_windows"]
    == (
        care_window_case_audit["included_windows"]
        + care_window_case_audit["excluded_windows"]
    )
).all():
    cell_8_validation_errors.append(
        "Candidate-window conservation failed in the case audit."
    )

if not (
    care_window_index_registry["forecast_target_timestamp"]
    - care_window_index_registry["context_end_timestamp"]
    == pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
).all():
    cell_8_validation_errors.append(
        "At least one indexed forecast target is not exactly one hour ahead."
    )

if not (
    care_window_index_registry["context_end_source_row_1_based"]
    - care_window_index_registry[
        "short_context_source_row_start_1_based"
    ]
    + 1
    == SHORT_CONTEXT_STEPS
).all():
    cell_8_validation_errors.append(
        "At least one short context does not span 144 source rows."
    )

if not (
    (
        care_window_index_registry["context_end_source_row_1_based"]
        - care_window_index_registry[
            "long_context_source_row_start_1_based"
        ]
    )
    // LONG_CONTEXT_SAMPLE_EVERY_SOURCE_ROWS
    + 1
    == LONG_CONTEXT_STEPS
).all():
    cell_8_validation_errors.append(
        "At least one long context does not contain 168 hourly positions."
    )

inside_segment_mask = (
    care_window_index_registry[
        "long_context_source_row_start_1_based"
    ].ge(
        care_window_index_registry[
            "segment_source_row_start_1_based"
        ]
    )
    & care_window_index_registry[
        "forecast_target_source_row_1_based"
    ].le(
        care_window_index_registry[
            "segment_source_row_end_1_based"
        ]
    )
)

if not inside_segment_mask.all():
    cell_8_validation_errors.append(
        "At least one window crosses a Cell 5 segment boundary."
    )

if not train_window_index[
    "forecast_target_source_partition"
].eq("train").all():
    cell_8_validation_errors.append(
        "A training target occurs outside the source-training partition."
    )

if not train_window_index[
    "context_source_training_fraction"
].eq(1.0).all():
    cell_8_validation_errors.append(
        "A training context contains source-prediction rows."
    )

if not train_window_index[
    "forecast_target_care_normal_status"
].eq(True).all():
    cell_8_validation_errors.append(
        "A training target has an abnormal or missing CARE status."
    )

if not train_window_index[
    "forecast_target_event_label"
].eq(0).all():
    cell_8_validation_errors.append(
        "A training target lies inside a labeled anomaly interval."
    )

for held_out_split_name, held_out_index in (
    ("validation", validation_window_index),
    ("test", test_window_index),
):
    if not held_out_index[
        "forecast_target_source_partition"
    ].eq("prediction").all():
        cell_8_validation_errors.append(
            f"A {held_out_split_name} target occurs outside the source-"
            "prediction partition."
        )

if care_window_feature_schema[
    "status_or_label_feature"
].ne(False).any():
    cell_8_validation_errors.append(
        "The model feature schema contains status or event-label features."
    )

if care_window_case_audit["included_windows"].eq(0).any():
    zero_window_cases = care_window_case_audit.loc[
        care_window_case_audit["included_windows"].eq(0),
        ["farm", "event_id", "model_split"],
    ]
    cell_8_validation_errors.append(
        f"{len(zero_window_cases)} cases have no eligible windows."
    )

normal_case_positive_windows = care_window_case_audit.loc[
    care_window_case_audit["is_anomaly"].eq(False),
    "positive_event_windows",
]

if normal_case_positive_windows.ne(0).any():
    cell_8_validation_errors.append(
        "A normal case contains positive event windows."
    )

held_out_anomaly_cases = care_window_case_audit.loc[
    care_window_case_audit["model_split"].isin(["validation", "test"])
    & care_window_case_audit["is_anomaly"].eq(True)
]

if held_out_anomaly_cases["positive_event_windows"].le(0).any():
    cell_8_validation_errors.append(
        "At least one held-out anomaly case has no positive event window."
    )

observed_window_split_case_counts = (
    care_window_case_audit.loc[
        care_window_case_audit["included_windows"].gt(0)
    ]
    .groupby("model_split", observed=False)
    .size()
    .to_dict()
)

for split_name, expected_count in EXPECTED_SPLIT_CASE_COUNTS.items():
    if int(observed_window_split_case_counts.get(split_name, 0)) != expected_count:
        cell_8_validation_errors.append(
            f"Window index represents the wrong number of {split_name} cases."
        )

window_asset_memberships = (
    care_window_index_registry
    .groupby("canonical_asset_key", observed=False)["model_split"]
    .nunique()
)

if not window_asset_memberships.eq(1).all():
    cell_8_validation_errors.append(
        "The window index introduces canonical-asset overlap."
    )

if cell_8_sensor_body_read_counts != EXPECTED_SPLIT_CASE_COUNTS:
    cell_8_validation_errors.append(
        "Cell 8 did not read each of the 95 case bodies exactly once."
    )

cell_8_constraint_records = [
    {
        "constraint": "frozen_asset_assignment_verified",
        "passed": observed_assignment_sha256_cell_8
        == FROZEN_ASSIGNMENT_SHA256,
        "observed": observed_assignment_sha256_cell_8,
        "expected": FROZEN_ASSIGNMENT_SHA256,
    },
    {
        "constraint": "cell_7_final_fit_verified",
        "passed": bool(
            missing_indicator_count_cell_8 == EXPECTED_MISSING_INDICATORS
            and scale_fallback_count_cell_8
            == EXPECTED_SCALE_FALLBACK_SIGNALS
        ),
        "observed": (
            f"{missing_indicator_count_cell_8} indicators; "
            f"{scale_fallback_count_cell_8} scale fallbacks"
        ),
        "expected": "210 indicators; 9 scale fallbacks",
    },
    {
        "constraint": "all_95_case_bodies_read_once_for_indexing",
        "passed": cell_8_sensor_body_read_counts
        == EXPECTED_SPLIT_CASE_COUNTS,
        "observed": str(cell_8_sensor_body_read_counts),
        "expected": str(EXPECTED_SPLIT_CASE_COUNTS),
    },
    {
        "constraint": "all_cases_have_eligible_windows",
        "passed": bool(
            care_window_case_audit["included_windows"].gt(0).all()
        ),
        "observed": int(
            care_window_case_audit["included_windows"].gt(0).sum()
        ),
        "expected": EXPECTED_TOTAL_CASES,
    },
    {
        "constraint": "all_windows_remain_inside_one_gap_safe_segment",
        "passed": bool(inside_segment_mask.all()),
        "observed": int(inside_segment_mask.sum()),
        "expected": int(len(care_window_index_registry)),
    },
    {
        "constraint": "forecast_target_exactly_one_hour_ahead",
        "passed": bool(
            (
                care_window_index_registry["forecast_target_timestamp"]
                - care_window_index_registry["context_end_timestamp"]
                == pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
            ).all()
        ),
        "observed": FORECAST_HORIZON_HOURS,
        "expected": FORECAST_HORIZON_HOURS,
    },
    {
        "constraint": "training_context_and_target_source_training_only",
        "passed": bool(
            train_window_index[
                "context_source_training_fraction"
            ].eq(1.0).all()
            and train_window_index[
                "forecast_target_source_partition"
            ].eq("train").all()
        ),
        "observed": int(len(train_window_index)),
        "expected": int(len(train_window_index)),
    },
    {
        "constraint": "training_targets_normal_status_and_event_negative",
        "passed": bool(
            train_window_index[
                "forecast_target_care_normal_status"
            ].eq(True).all()
            and train_window_index[
                "forecast_target_event_label"
            ].eq(0).all()
        ),
        "observed": int(len(train_window_index)),
        "expected": int(len(train_window_index)),
    },
    {
        "constraint": "held_out_targets_source_prediction_only",
        "passed": bool(
            validation_window_index[
                "forecast_target_source_partition"
            ].eq("prediction").all()
            and test_window_index[
                "forecast_target_source_partition"
            ].eq("prediction").all()
        ),
        "observed": int(
            len(validation_window_index) + len(test_window_index)
        ),
        "expected": int(
            len(validation_window_index) + len(test_window_index)
        ),
    },
    {
        "constraint": "zero_canonical_asset_overlap",
        "passed": bool(window_asset_memberships.eq(1).all()),
        "observed": int((window_asset_memberships > 1).sum()),
        "expected": 0,
    },
    {
        "constraint": "status_and_labels_absent_from_model_features",
        "passed": bool(
            care_window_feature_schema[
                "status_or_label_feature"
            ].eq(False).all()
        ),
        "observed": int(
            care_window_feature_schema[
                "status_or_label_feature"
            ].ne(False).sum()
        ),
        "expected": 0,
    },
    {
        "constraint": "source_values_unmodified",
        "passed": True,
        "observed": False,
        "expected": False,
    },
]
care_window_constraint_audit = pd.DataFrame(
    cell_8_constraint_records
)

failed_constraints = care_window_constraint_audit.loc[
    ~care_window_constraint_audit["passed"].astype(bool)
]

if len(failed_constraints) > 0:
    cell_8_validation_errors.extend(
        "Constraint failed: " + str(constraint_name)
        for constraint_name in failed_constraints["constraint"]
    )

if cell_8_validation_errors:
    raise RuntimeError(
        "CELL 8 VALIDATION FAILED:\n  - "
        + "\n  - ".join(cell_8_validation_errors)
    )


# ----------------------------------------------------------------------------
# 9. Save compact window artifacts and manifest
# ----------------------------------------------------------------------------

care_window_index_registry.to_csv(
    TABLE_DIR / "care_multiscale_window_index.csv",
    index=False,
)
care_window_case_audit.to_csv(
    TABLE_DIR / "care_multiscale_window_case_audit.csv",
    index=False,
)
care_window_split_summary.to_csv(
    TABLE_DIR / "care_multiscale_window_split_summary.csv",
    index=False,
)
care_window_farm_split_summary.to_csv(
    TABLE_DIR / "care_multiscale_window_farm_split_summary.csv",
    index=False,
)
care_window_exclusion_summary.to_csv(
    TABLE_DIR / "care_multiscale_window_exclusion_summary.csv",
    index=False,
)
care_window_feature_schema.to_csv(
    TABLE_DIR / "care_multiscale_window_feature_schema.csv",
    index=False,
)
care_window_feature_farm_summary.to_csv(
    TABLE_DIR / "care_multiscale_window_feature_farm_summary.csv",
    index=False,
)
care_window_constraint_audit.to_csv(
    TABLE_DIR / "care_multiscale_window_constraint_audit.csv",
    index=False,
)

cell_8_manifest = {
    "cell": 8,
    "purpose": (
        "Build a leakage-safe lazy multiscale window index for CARE v6"
    ),
    "policy": CELL_8_POLICY,
    "assignment_digest_sha256": observed_assignment_sha256_cell_8,
    "preprocessing_parameter_digest_sha256": (
        preprocessing_parameter_digest_sha256
    ),
    "chunk_rows": CELL_8_CHUNK_ROWS,
    "sensor_body_read_counts": cell_8_sensor_body_read_counts,
    "case_count": int(len(care_window_case_audit)),
    "window_count": int(len(care_window_index_registry)),
    "train_window_count": int(len(train_window_index)),
    "validation_window_count": int(len(validation_window_index)),
    "test_window_count": int(len(test_window_index)),
    "feature_schema_row_count": int(len(care_window_feature_schema)),
    "feature_farm_summary": care_window_feature_farm_summary.to_dict(
        orient="records"
    ),
    "window_split_summary": care_window_split_summary.to_dict(
        orient="records"
    ),
    "window_exclusion_summary": care_window_exclusion_summary.to_dict(
        orient="records"
    ),
    "constraint_audit": care_window_constraint_audit.to_dict(
        orient="records"
    ),
    "validation_warnings": cell_8_validation_warnings,
    "validation_errors": cell_8_validation_errors,
    "overlapping_tensors_materialized": False,
    "preprocessing_refit": False,
    "status_or_event_label_used_as_model_feature": False,
    "source_data_modified": False,
}

save_cell_8_json(
    cell_8_manifest,
    OUTPUT_ROOT / "lazy_multiscale_window_index_manifest.json",
)


# ----------------------------------------------------------------------------
# 10. Display summaries and successful completion
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("MULTISCALE WINDOW SPLIT SUMMARY")
print("=" * 80)
display(care_window_split_summary)

print("\n" + "=" * 80)
print("WINDOW FEATURE SUMMARY")
print("=" * 80)
display(care_window_feature_farm_summary)

print("\n" + "=" * 80)
print("WINDOW EXCLUSION SUMMARY")
print("=" * 80)
display(care_window_exclusion_summary)

print("\n" + "=" * 80)
print("CELL 8 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Cases indexed             : {len(care_window_case_audit)}")
print(f"Train windows             : {len(train_window_index):,}")
print(f"Validation windows        : {len(validation_window_index):,}")
print(f"Test windows              : {len(test_window_index):,}")
print(
    "Window tensor storage     : None (lazy source-row references only)"
)
print(
    "Primary / indicator feats: "
    f"{EXPECTED_KEPT_SIGNALS} / {EXPECTED_MISSING_INDICATORS} "
    "across farm-specific schemas"
)
print(
    "Assignment SHA-256       : "
    f"{observed_assignment_sha256_cell_8}"
)
print(
    "Preprocessing SHA-256    : "
    f"{preprocessing_parameter_digest_sha256}"
)
print(f"Output directory          : {TABLE_DIR}")
print("Preprocessing refit       : No")
print("Source data modified      : No")
print("Reusable objects:")
print("  - care_window_index_registry")
print("  - train_window_index")
print("  - validation_window_index")
print("  - test_window_index")
print("  - care_window_case_audit")
print("  - care_window_split_summary")
print("  - care_window_farm_split_summary")
print("  - care_window_exclusion_summary")
print("  - care_window_feature_schema")
print("  - care_window_feature_farm_summary")
print("  - care_window_constraint_audit")
print("  - care_model_feature_names")
print("  - transform_care_primary_frame")
print("  - cell_8_manifest")
print("Created:")
print("  - care_multiscale_window_index.csv")
print("  - care_multiscale_window_case_audit.csv")
print("  - care_multiscale_window_split_summary.csv")
print("  - care_multiscale_window_farm_split_summary.csv")
print("  - care_multiscale_window_exclusion_summary.csv")
print("  - care_multiscale_window_feature_schema.csv")
print("  - care_multiscale_window_feature_farm_summary.csv")
print("  - care_multiscale_window_constraint_audit.csv")
print("  - lazy_multiscale_window_index_manifest.json")
print("=" * 80)
