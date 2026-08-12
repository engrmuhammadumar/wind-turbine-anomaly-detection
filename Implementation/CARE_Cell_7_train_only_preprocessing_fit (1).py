# ============================================================================
# CELL 7 — TRAIN-ONLY PREPROCESSING FIT AND OPERATIONAL-VALIDITY POLICY
#
# Prerequisites from the successful Cells 3–6:
#   - FARM_SCHEMAS
#   - OUTPUT_ROOT
#   - TABLE_DIR
#   - iter_care_csv_chunks
#   - asset_split_assignment
#   - case_split_registry
#   - cell_6_assignment_digest
#   - assignment_digest_sha256
#
# This cell:
#   - verifies the frozen Cell 6 asset-to-split assignment
#   - reads sensor bodies from the 67 training cases only
#   - leaves validation and test sensor bodies completely unread
#   - defines a transparent row-level operational-validity indicator
#   - fits farm-specific train-only signal statistics
#   - fits median imputation and robust scaling parameters
#   - marks signals for removal only when unusable or train-constant
#   - identifies train-derived missing-indicator columns
#   - audits status_type_id jointly with synchronous all-zero behavior
#
# This cell DOES NOT transform or rewrite raw values, read validation/test
# sensor bodies, resample, interpolate, clip, create windows, choose a model,
# tune a threshold, or train/evaluate a model.
# ============================================================================


# ----------------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------------

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# ----------------------------------------------------------------------------
# 2. Prerequisites, paths, frozen split, and fit policy
# ----------------------------------------------------------------------------

REQUIRED_CELL_7_OBJECTS = (
    "FARM_SCHEMAS",
    "OUTPUT_ROOT",
    "iter_care_csv_chunks",
    "asset_split_assignment",
    "case_split_registry",
    "cell_6_assignment_digest",
    "assignment_digest_sha256",
)

missing_cell_7_objects = [
    object_name
    for object_name in REQUIRED_CELL_7_OBJECTS
    if object_name not in globals()
]

if missing_cell_7_objects:
    raise RuntimeError(
        "Run the successful Cells 3–6 before Cell 7. Missing objects: "
        + ", ".join(missing_cell_7_objects)
    )

OUTPUT_ROOT = Path(OUTPUT_ROOT)

if "TABLE_DIR" not in globals():
    TABLE_DIR = OUTPUT_ROOT / "tables"
else:
    TABLE_DIR = Path(TABLE_DIR)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL_CASES = 95
EXPECTED_SPLIT_CASE_COUNTS = {
    "train": 67,
    "validation": 14,
    "test": 14,
}
EXPECTED_CANONICAL_ASSETS = 36
EXPECTED_SPLIT_ASSET_COUNTS = {
    "train": 25,
    "validation": 6,
    "test": 5,
}

FROZEN_ASSIGNMENT_SHA256 = (
    "30cf8c5d10db81e2d730742908230c65f76654f1058e21d1f17989cc96ce9e27"
)

# This affects only I/O/memory, not the fitted numerical policy.
CELL_7_CHUNK_ROWS = 20_000

# Robust quantiles are estimated from a deterministic uniform row reservoir
# for each farm. Exact counts, means, variances, minima, and maxima are still
# accumulated from every fit-eligible training observation.
CELL_7_QUANTILE_SAMPLE_ROWS = 50_000
CELL_7_RANDOM_SEED = 29

# CARE's official point-level labels define only status IDs 0 and 2 as normal
# turbine behavior. All other or missing status values are audited but cannot
# contribute to a normal-behavior preprocessing fit.
NORMAL_STATUS_TYPE_IDS = frozenset({"0", "2"})

CELL_7_POLICY = {
    "frozen_assignment_sha256": FROZEN_ASSIGNMENT_SHA256,
    "fit_split": "train",
    "expected_training_cases": EXPECTED_SPLIT_CASE_COUNTS["train"],
    "farm_specific_preprocessing": True,
    "validation_sensor_bodies_read": False,
    "test_sensor_bodies_read": False,
    "fit_row_rule": (
        "A training source row fits preprocessing statistics only when its "
        "status_type_id is CARE-normal (0 or 2), its timestamp is valid, at "
        "least one primary signal is finite, and the row is not synchronously "
        "all-zero across all finite primary signals."
    ),
    "synchronous_all_zero_rule": (
        "At least one primary signal is finite and every finite primary "
        "signal equals zero; non-finite primary signals are ignored only for "
        "this row-level operational-state flag."
    ),
    "individual_zero_policy": (
        "Individual finite zeros are retained as observed values. They are "
        "never converted to missing values."
    ),
    "imputation": (
        "Per-farm, per-signal training median. A rare signal absent from the "
        "quantile reservoir falls back to its exact training mean."
    ),
    "centering": "Per-farm, per-signal training median",
    "scaling_priority": [
        "interquartile_range",
        "1.4826_times_median_absolute_deviation",
        "exact_training_sample_standard_deviation",
        "unit_scale",
    ],
    "quantile_estimation": (
        "Deterministic uniform reservoir of fit-eligible training rows within "
        "each farm"
    ),
    "quantile_sample_rows_per_farm": CELL_7_QUANTILE_SAMPLE_ROWS,
    "constant_signal_rule": (
        "Drop later only when all finite fit-eligible training observations "
        "are exactly equal, or no finite fit observation exists."
    ),
    "missing_indicator_rule": (
        "Create later only for kept signals with at least one non-finite "
        "value in CARE-normal fit-eligible training rows."
    ),
    "status_type_id_policy": (
        "Use CARE status IDs 0 and 2 only to identify normal-behavior rows "
        "for preprocessing fit. Audit every status, exclude other or missing "
        "statuses from fit, and never include status_type_id as a model "
        "feature."
    ),
    "source_values_modified": False,
    "transformations_applied_to_source": [],
}


# ----------------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------------

def cell_7_json_safe(value: Any) -> Any:
    """Recursively convert scientific-Python objects to strict JSON."""

    if isinstance(value, dict):
        return {
            str(key): cell_7_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            cell_7_json_safe(item)
            for item in value
        ]

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


def save_cell_7_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write a strict, human-readable JSON manifest."""

    with Path(destination).open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            cell_7_json_safe(payload),
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def cell_7_value_label(value: Any) -> str:
    """Create a stable printable label for a categorical source value."""

    if pd.isna(value):
        return "<MISSING>"

    text_value = str(value).strip()

    try:
        numeric_value = float(text_value)
    except (TypeError, ValueError):
        return text_value

    if np.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))

    return text_value


def cell_7_new_moment_accumulator(
    signal_count: int,
) -> dict[str, np.ndarray]:
    """Create exact online moment/count arrays for one farm."""

    return {
        "fit_finite_count": np.zeros(signal_count, dtype=np.int64),
        "fit_mean": np.zeros(signal_count, dtype=np.float64),
        "fit_m2": np.zeros(signal_count, dtype=np.float64),
        "fit_min": np.full(signal_count, np.inf, dtype=np.float64),
        "fit_max": np.full(signal_count, -np.inf, dtype=np.float64),
        "raw_source_missing_count": np.zeros(
            signal_count,
            dtype=np.int64,
        ),
        "raw_non_numeric_count": np.zeros(
            signal_count,
            dtype=np.int64,
        ),
        "raw_infinite_count": np.zeros(
            signal_count,
            dtype=np.int64,
        ),
        "raw_nonfinite_count": np.zeros(
            signal_count,
            dtype=np.int64,
        ),
        "raw_finite_count": np.zeros(signal_count, dtype=np.int64),
        "raw_zero_count": np.zeros(signal_count, dtype=np.int64),
    }


def cell_7_update_exact_moments(
    accumulator: dict[str, np.ndarray],
    fit_values: np.ndarray,
) -> None:
    """Merge one finite-aware batch into exact per-signal Welford moments."""

    if fit_values.size == 0 or fit_values.shape[0] == 0:
        return

    finite_mask = np.isfinite(fit_values)
    batch_counts = finite_mask.sum(axis=0, dtype=np.int64)
    active_mask = batch_counts > 0

    if not active_mask.any():
        return

    safe_values = np.where(finite_mask, fit_values, 0.0)
    batch_sums = safe_values.sum(axis=0, dtype=np.float64)
    batch_means = np.zeros(fit_values.shape[1], dtype=np.float64)
    np.divide(
        batch_sums,
        batch_counts,
        out=batch_means,
        where=active_mask,
    )

    centered_values = np.where(
        finite_mask,
        fit_values - batch_means,
        0.0,
    )
    batch_m2 = np.square(centered_values).sum(
        axis=0,
        dtype=np.float64,
    )

    prior_counts = accumulator["fit_finite_count"].copy()
    prior_means = accumulator["fit_mean"].copy()
    combined_counts = prior_counts + batch_counts
    deltas = batch_means - prior_means

    accumulator["fit_mean"][active_mask] = (
        prior_means[active_mask]
        + deltas[active_mask]
        * batch_counts[active_mask]
        / combined_counts[active_mask]
    )
    accumulator["fit_m2"][active_mask] = (
        accumulator["fit_m2"][active_mask]
        + batch_m2[active_mask]
        + np.square(deltas[active_mask])
        * prior_counts[active_mask]
        * batch_counts[active_mask]
        / combined_counts[active_mask]
    )
    accumulator["fit_finite_count"] = combined_counts

    batch_minima = np.where(
        finite_mask,
        fit_values,
        np.inf,
    ).min(axis=0)
    batch_maxima = np.where(
        finite_mask,
        fit_values,
        -np.inf,
    ).max(axis=0)
    accumulator["fit_min"] = np.minimum(
        accumulator["fit_min"],
        batch_minima,
    )
    accumulator["fit_max"] = np.maximum(
        accumulator["fit_max"],
        batch_maxima,
    )


def cell_7_update_row_reservoir(
    current_values: np.ndarray,
    current_priorities: np.ndarray,
    new_values: np.ndarray,
    random_generator: np.random.Generator,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a deterministic uniform priority reservoir of training rows."""

    if new_values.size == 0 or new_values.shape[0] == 0:
        return current_values, current_priorities

    new_priorities = random_generator.random(new_values.shape[0])

    # Once the reservoir is full, only new rows with a smaller priority than
    # its current worst member can enter. Filtering them before concatenation
    # avoids repeatedly copying a full wide farm matrix for irrelevant rows.
    if current_values.shape[0] >= int(capacity):
        candidate_mask = new_priorities < current_priorities.max()

        if not candidate_mask.any():
            return current_values, current_priorities

        new_values = new_values[candidate_mask]
        new_priorities = new_priorities[candidate_mask]

    if current_values.shape[0] == 0:
        combined_values = new_values.copy()
        combined_priorities = new_priorities
    else:
        combined_values = np.concatenate(
            [current_values, new_values],
            axis=0,
        )
        combined_priorities = np.concatenate(
            [current_priorities, new_priorities]
        )

    if combined_values.shape[0] <= int(capacity):
        return combined_values, combined_priorities

    keep_indices = np.argpartition(
        combined_priorities,
        kth=int(capacity) - 1,
    )[:int(capacity)]
    keep_indices = keep_indices[
        np.argsort(
            combined_priorities[keep_indices],
            kind="stable",
        )
    ]

    return (
        combined_values[keep_indices],
        combined_priorities[keep_indices],
    )


def cell_7_scale_tolerance(center_value: float) -> float:
    """Return a small magnitude-aware positive-scale tolerance."""

    return np.finfo(np.float64).eps * 100.0 * max(
        1.0,
        abs(float(center_value)),
    )


# ----------------------------------------------------------------------------
# 4. Validate and freeze the Cell 6 assignment before any sensor read
# ----------------------------------------------------------------------------

cell_7_validation_errors: list[str] = []
cell_7_validation_warnings: list[str] = []

required_case_split_columns = {
    "farm",
    "event_id",
    "event_type",
    "is_anomaly",
    "canonical_asset_key",
    "model_split",
    "file_path",
}
missing_case_split_columns = sorted(
    required_case_split_columns - set(case_split_registry.columns)
)

if missing_case_split_columns:
    cell_7_validation_errors.append(
        "case_split_registry lacks required columns: "
        + ", ".join(missing_case_split_columns)
    )

required_asset_split_columns = {
    "canonical_asset_key",
    "model_split",
}
missing_asset_split_columns = sorted(
    required_asset_split_columns - set(asset_split_assignment.columns)
)

if missing_asset_split_columns:
    cell_7_validation_errors.append(
        "asset_split_assignment lacks required columns: "
        + ", ".join(missing_asset_split_columns)
    )

if not cell_7_validation_errors:
    observed_assignment_sha256 = cell_6_assignment_digest(
        asset_split_assignment
    )

    if observed_assignment_sha256 != FROZEN_ASSIGNMENT_SHA256:
        cell_7_validation_errors.append(
            "The recomputed Cell 6 assignment SHA-256 is not the frozen "
            f"value. Observed {observed_assignment_sha256}; expected "
            f"{FROZEN_ASSIGNMENT_SHA256}."
        )

    if str(assignment_digest_sha256) != FROZEN_ASSIGNMENT_SHA256:
        cell_7_validation_errors.append(
            "The in-memory assignment_digest_sha256 is not the frozen Cell 6 "
            "value."
        )

    if len(case_split_registry) != EXPECTED_TOTAL_CASES:
        cell_7_validation_errors.append(
            f"case_split_registry contains {len(case_split_registry)} cases; "
            f"expected {EXPECTED_TOTAL_CASES}."
        )

    if case_split_registry.duplicated(
        subset=["farm", "event_id"]
    ).any():
        cell_7_validation_errors.append(
            "case_split_registry contains duplicate farm/event_id keys."
        )

    observed_split_case_counts = (
        case_split_registry["model_split"]
        .astype("string")
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_SPLIT_CASE_COUNTS.items()
    ):
        observed_count = int(
            observed_split_case_counts.get(split_name, 0)
        )

        if observed_count != expected_count:
            cell_7_validation_errors.append(
                f"Split {split_name!r} contains {observed_count} cases; "
                f"expected {expected_count}."
            )

    if len(asset_split_assignment) != EXPECTED_CANONICAL_ASSETS:
        cell_7_validation_errors.append(
            f"asset_split_assignment contains {len(asset_split_assignment)} "
            f"assets; expected {EXPECTED_CANONICAL_ASSETS}."
        )

    observed_split_asset_counts = (
        asset_split_assignment["model_split"]
        .astype("string")
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_SPLIT_ASSET_COUNTS.items()
    ):
        observed_count = int(
            observed_split_asset_counts.get(split_name, 0)
        )

        if observed_count != expected_count:
            cell_7_validation_errors.append(
                f"Split {split_name!r} contains {observed_count} assets; "
                f"expected {expected_count}."
            )

    asset_membership_counts = (
        case_split_registry
        .groupby("canonical_asset_key", observed=False)["model_split"]
        .nunique()
    )

    if not asset_membership_counts.eq(1).all():
        cell_7_validation_errors.append(
            "At least one canonical asset occurs in multiple model splits."
        )

if cell_7_validation_errors:
    raise RuntimeError(
        "CELL 7 INPUT VALIDATION FAILED:\n  - "
        + "\n  - ".join(cell_7_validation_errors)
    )

train_case_registry_cell_7 = (
    case_split_registry.loc[
        case_split_registry["model_split"].astype("string").eq("train")
    ]
    .sort_values(["farm", "event_id"], kind="stable")
    .reset_index(drop=True)
)

training_farms = tuple(
    sorted(train_case_registry_cell_7["farm"].astype(str).unique())
)

missing_farm_schemas = [
    farm_name
    for farm_name in training_farms
    if farm_name not in FARM_SCHEMAS
]

if missing_farm_schemas:
    raise RuntimeError(
        "CELL 7 INPUT VALIDATION FAILED: missing FARM_SCHEMAS entries for "
        + ", ".join(missing_farm_schemas)
    )


# ----------------------------------------------------------------------------
# 5. Stream only training cases and fit farm-specific accumulators
# ----------------------------------------------------------------------------

train_preprocessing_fit_case_records: list[dict[str, Any]] = []
train_preprocessing_signal_parameter_records: list[dict[str, Any]] = []
train_status_operational_records: list[dict[str, Any]] = []

sensor_body_read_counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

print("=" * 80)
print("FITTING TRAIN-ONLY CARE PREPROCESSING PARAMETERS")
print("=" * 80)
print(f"Frozen assignment SHA-256: {FROZEN_ASSIGNMENT_SHA256}")
print(
    f"Training cases: {len(train_case_registry_cell_7)} | chunk size: "
    f"{CELL_7_CHUNK_ROWS:,} rows | quantile reservoir: "
    f"{CELL_7_QUANTILE_SAMPLE_ROWS:,} rows per farm"
)
print("Validation/test sensor bodies will not be read.")

for farm_index, farm_name in enumerate(training_farms):
    primary_signal_columns = list(
        FARM_SCHEMAS[farm_name]["primary_signal_columns"]
    )

    if not primary_signal_columns:
        raise RuntimeError(
            f"Cell 3 registered no primary signals for {farm_name}."
        )

    if len(primary_signal_columns) != len(set(primary_signal_columns)):
        raise RuntimeError(
            f"Cell 3 registered duplicate primary signals for {farm_name}."
        )

    farm_cases = (
        train_case_registry_cell_7.loc[
            train_case_registry_cell_7["farm"].astype(str).eq(farm_name)
        ]
        .sort_values("event_id", kind="stable")
        .reset_index(drop=True)
    )

    signal_count = len(primary_signal_columns)
    exact_accumulator = cell_7_new_moment_accumulator(signal_count)
    farm_random_generator = np.random.default_rng(
        CELL_7_RANDOM_SEED + farm_index
    )
    reservoir_values = np.empty(
        (0, signal_count),
        dtype=np.float64,
    )
    reservoir_priorities = np.empty(0, dtype=np.float64)

    farm_total_rows = 0
    farm_fit_rows = 0
    farm_valid_timestamp_rows = 0
    farm_any_finite_rows = 0
    farm_strict_all_zero_rows = 0
    farm_synchronous_all_zero_rows = 0
    farm_status_accumulators: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "row_count": 0,
            "valid_timestamp_rows": 0,
            "any_finite_rows": 0,
            "strict_all_zero_rows": 0,
            "synchronous_all_zero_rows": 0,
            "fit_eligible_rows": 0,
        }
    )

    print(
        f"\n{farm_name}: {len(farm_cases)} training cases | "
        f"{signal_count} primary signals"
    )

    for farm_case_number, registry_row in enumerate(
        farm_cases.itertuples(index=False),
        start=1,
    ):
        event_id = int(registry_row.event_id)
        event_type = str(registry_row.event_type)
        file_path = Path(registry_row.file_path)
        canonical_asset_key = str(registry_row.canonical_asset_key)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Training event file does not exist: {file_path}"
            )

        print(
            f"  [{farm_case_number:02d}/{len(farm_cases):02d}] "
            f"{event_type:<7} | event {event_id:<3} | "
            f"asset {canonical_asset_key}"
        )

        requested_columns = [
            "time_stamp",
            "status_type_id",
            *primary_signal_columns,
        ]
        case_row_count = 0
        case_chunk_count = 0
        case_valid_timestamp_rows = 0
        case_any_finite_rows = 0
        case_strict_all_zero_rows = 0
        case_synchronous_all_zero_rows = 0
        case_normal_status_rows = 0
        case_status_excluded_rows = 0
        case_fit_rows = 0
        case_source_missing_cells = 0
        case_non_numeric_cells = 0
        case_infinite_cells = 0

        for chunk in iter_care_csv_chunks(
            file_path,
            usecols=requested_columns,
            chunksize=CELL_7_CHUNK_ROWS,
        ):
            case_chunk_count += 1
            chunk_row_count = int(len(chunk))

            if chunk_row_count == 0:
                continue

            case_row_count += chunk_row_count
            farm_total_rows += chunk_row_count

            raw_signal_frame = chunk[primary_signal_columns]
            source_missing_mask = raw_signal_frame.isna().to_numpy(
                dtype=bool
            )
            numeric_signal_frame = raw_signal_frame.apply(
                pd.to_numeric,
                errors="coerce",
            )
            signal_values = numeric_signal_frame.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            finite_mask = np.isfinite(signal_values)
            infinite_mask = np.isinf(signal_values)
            non_numeric_mask = (
                ~source_missing_mask
                & np.isnan(signal_values)
            )
            zero_mask = finite_mask & (signal_values == 0.0)

            status_labels = (
                chunk["status_type_id"]
                .map(cell_7_value_label)
                .to_numpy(dtype=object)
            )
            normal_status_mask = np.isin(
                status_labels,
                tuple(NORMAL_STATUS_TYPE_IDS),
            )

            valid_timestamp_mask = (
                chunk["time_stamp"].notna().to_numpy(dtype=bool)
            )
            any_finite_mask = finite_mask.any(axis=1)
            strict_all_zero_mask = (
                finite_mask.all(axis=1)
                & zero_mask.all(axis=1)
            )
            synchronous_all_zero_mask = (
                any_finite_mask
                & np.all(~finite_mask | zero_mask, axis=1)
            )
            fit_eligible_mask = (
                normal_status_mask
                & valid_timestamp_mask
                & any_finite_mask
                & ~synchronous_all_zero_mask
            )

            valid_timestamp_rows = int(valid_timestamp_mask.sum())
            any_finite_rows = int(any_finite_mask.sum())
            strict_all_zero_rows = int(strict_all_zero_mask.sum())
            synchronous_all_zero_rows = int(
                synchronous_all_zero_mask.sum()
            )
            normal_status_rows = int(normal_status_mask.sum())
            status_excluded_rows = int((~normal_status_mask).sum())
            fit_rows = int(fit_eligible_mask.sum())

            case_valid_timestamp_rows += valid_timestamp_rows
            case_any_finite_rows += any_finite_rows
            case_strict_all_zero_rows += strict_all_zero_rows
            case_synchronous_all_zero_rows += synchronous_all_zero_rows
            case_normal_status_rows += normal_status_rows
            case_status_excluded_rows += status_excluded_rows
            case_fit_rows += fit_rows

            farm_valid_timestamp_rows += valid_timestamp_rows
            farm_any_finite_rows += any_finite_rows
            farm_strict_all_zero_rows += strict_all_zero_rows
            farm_synchronous_all_zero_rows += synchronous_all_zero_rows
            farm_fit_rows += fit_rows

            source_missing_counts = source_missing_mask.sum(
                axis=0,
                dtype=np.int64,
            )
            non_numeric_counts = non_numeric_mask.sum(
                axis=0,
                dtype=np.int64,
            )
            infinite_counts = infinite_mask.sum(
                axis=0,
                dtype=np.int64,
            )
            nonfinite_counts = (~finite_mask).sum(
                axis=0,
                dtype=np.int64,
            )

            exact_accumulator[
                "raw_source_missing_count"
            ] += source_missing_counts
            exact_accumulator[
                "raw_non_numeric_count"
            ] += non_numeric_counts
            exact_accumulator[
                "raw_infinite_count"
            ] += infinite_counts
            exact_accumulator[
                "raw_nonfinite_count"
            ] += nonfinite_counts
            exact_accumulator[
                "raw_finite_count"
            ] += finite_mask.sum(axis=0, dtype=np.int64)
            exact_accumulator[
                "raw_zero_count"
            ] += zero_mask.sum(axis=0, dtype=np.int64)

            case_source_missing_cells += int(source_missing_counts.sum())
            case_non_numeric_cells += int(non_numeric_counts.sum())
            case_infinite_cells += int(infinite_counts.sum())

            fit_values = signal_values[fit_eligible_mask]
            cell_7_update_exact_moments(
                exact_accumulator,
                fit_values,
            )
            (
                reservoir_values,
                reservoir_priorities,
            ) = cell_7_update_row_reservoir(
                reservoir_values,
                reservoir_priorities,
                fit_values,
                farm_random_generator,
                CELL_7_QUANTILE_SAMPLE_ROWS,
            )

            for status_label in pd.unique(status_labels):
                status_mask = status_labels == status_label
                status_accumulator = farm_status_accumulators[
                    str(status_label)
                ]
                status_accumulator["row_count"] += int(status_mask.sum())
                status_accumulator["valid_timestamp_rows"] += int(
                    (status_mask & valid_timestamp_mask).sum()
                )
                status_accumulator["any_finite_rows"] += int(
                    (status_mask & any_finite_mask).sum()
                )
                status_accumulator["strict_all_zero_rows"] += int(
                    (status_mask & strict_all_zero_mask).sum()
                )
                status_accumulator[
                    "synchronous_all_zero_rows"
                ] += int(
                    (status_mask & synchronous_all_zero_mask).sum()
                )
                status_accumulator["fit_eligible_rows"] += int(
                    (status_mask & fit_eligible_mask).sum()
                )

            del (
                raw_signal_frame,
                numeric_signal_frame,
                signal_values,
                finite_mask,
                fit_values,
            )

        sensor_body_read_counts["train"] += 1

        if case_row_count <= 0:
            cell_7_validation_errors.append(
                f"{farm_name} event {event_id} contains no data rows."
            )

        if case_fit_rows <= 0:
            cell_7_validation_warnings.append(
                f"{farm_name} event {event_id} contributes no "
                "fit-eligible rows."
            )

        train_preprocessing_fit_case_records.append(
            {
                "farm": farm_name,
                "event_id": event_id,
                "event_type": event_type,
                "is_anomaly": bool(registry_row.is_anomaly),
                "canonical_asset_key": canonical_asset_key,
                "model_split": "train",
                "chunk_count": case_chunk_count,
                "row_count": case_row_count,
                "valid_timestamp_rows": case_valid_timestamp_rows,
                "invalid_timestamp_rows": (
                    case_row_count - case_valid_timestamp_rows
                ),
                "rows_with_any_finite_primary_signal": (
                    case_any_finite_rows
                ),
                "rows_without_finite_primary_signal": (
                    case_row_count - case_any_finite_rows
                ),
                "strict_all_zero_rows": case_strict_all_zero_rows,
                "synchronous_all_zero_rows": (
                    case_synchronous_all_zero_rows
                ),
                "care_normal_status_rows": case_normal_status_rows,
                "status_excluded_rows": case_status_excluded_rows,
                "fit_eligible_rows": case_fit_rows,
                "fit_excluded_rows": case_row_count - case_fit_rows,
                "fit_eligible_fraction": (
                    case_fit_rows / case_row_count
                    if case_row_count > 0
                    else np.nan
                ),
                "source_missing_cells": case_source_missing_cells,
                "non_numeric_cells": case_non_numeric_cells,
                "infinite_cells": case_infinite_cells,
                "primary_signal_count": signal_count,
                "source_values_modified": False,
            }
        )

    # Finalize one farm's exact and reservoir-derived signal parameters.
    fit_counts = exact_accumulator["fit_finite_count"]
    fit_means = exact_accumulator["fit_mean"]
    fit_m2 = exact_accumulator["fit_m2"]
    fit_standard_deviations = np.full(signal_count, np.nan)
    standard_deviation_mask = fit_counts > 1
    fit_standard_deviations[standard_deviation_mask] = np.sqrt(
        fit_m2[standard_deviation_mask]
        / (fit_counts[standard_deviation_mask] - 1)
    )

    for feature_index, signal_name in enumerate(primary_signal_columns):
        finite_fit_count = int(fit_counts[feature_index])
        exact_mean = (
            float(fit_means[feature_index])
            if finite_fit_count > 0
            else np.nan
        )
        exact_standard_deviation = float(
            fit_standard_deviations[feature_index]
        )
        exact_minimum = (
            float(exact_accumulator["fit_min"][feature_index])
            if finite_fit_count > 0
            else np.nan
        )
        exact_maximum = (
            float(exact_accumulator["fit_max"][feature_index])
            if finite_fit_count > 0
            else np.nan
        )

        signal_sample = reservoir_values[:, feature_index]
        signal_sample = signal_sample[np.isfinite(signal_sample)]
        quantile_fallback_used = False

        if signal_sample.size > 0:
            quantile_25, median_value, quantile_75 = np.quantile(
                signal_sample,
                [0.25, 0.50, 0.75],
                method="linear",
            )
            median_absolute_deviation = float(
                np.median(np.abs(signal_sample - median_value))
            )
        elif finite_fit_count > 0:
            quantile_25 = exact_mean
            median_value = exact_mean
            quantile_75 = exact_mean
            median_absolute_deviation = 0.0
            quantile_fallback_used = True
        else:
            quantile_25 = np.nan
            median_value = np.nan
            quantile_75 = np.nan
            median_absolute_deviation = np.nan

        interquartile_range = float(quantile_75 - quantile_25)
        scaled_mad = float(1.4826 * median_absolute_deviation)
        train_constant = bool(
            finite_fit_count > 0
            and exact_minimum == exact_maximum
        )
        no_finite_fit_values = bool(finite_fit_count == 0)
        keep_for_model = bool(
            not no_finite_fit_values
            and not train_constant
        )
        scale_tolerance = (
            cell_7_scale_tolerance(median_value)
            if np.isfinite(median_value)
            else np.nan
        )

        if no_finite_fit_values or train_constant:
            scale_value = 1.0
            scale_statistic = "unit_scale_not_kept"
        elif (
            np.isfinite(interquartile_range)
            and interquartile_range > scale_tolerance
        ):
            scale_value = interquartile_range
            scale_statistic = "interquartile_range"
        elif np.isfinite(scaled_mad) and scaled_mad > scale_tolerance:
            scale_value = scaled_mad
            scale_statistic = "scaled_mad"
        elif (
            np.isfinite(exact_standard_deviation)
            and exact_standard_deviation > scale_tolerance
        ):
            scale_value = exact_standard_deviation
            scale_statistic = "exact_sample_standard_deviation"
        else:
            scale_value = 1.0
            scale_statistic = "unit_scale_numerical_fallback"

        if no_finite_fit_values:
            drop_reason = "no_finite_fit_eligible_training_values"
        elif train_constant:
            drop_reason = "constant_in_fit_eligible_training_values"
        else:
            drop_reason = ""

        raw_nonfinite_count = int(
            exact_accumulator["raw_nonfinite_count"][feature_index]
        )
        fit_nonfinite_count = int(farm_fit_rows - finite_fit_count)
        add_missing_indicator = bool(
            keep_for_model and fit_nonfinite_count > 0
        )

        train_preprocessing_signal_parameter_records.append(
            {
                "farm": farm_name,
                "feature_index": feature_index,
                "signal_name": signal_name,
                "training_case_count": int(len(farm_cases)),
                "training_source_row_count": farm_total_rows,
                "fit_eligible_row_count": farm_fit_rows,
                "raw_finite_count": int(
                    exact_accumulator["raw_finite_count"][feature_index]
                ),
                "raw_source_missing_count": int(
                    exact_accumulator[
                        "raw_source_missing_count"
                    ][feature_index]
                ),
                "raw_non_numeric_count": int(
                    exact_accumulator["raw_non_numeric_count"][feature_index]
                ),
                "raw_infinite_count": int(
                    exact_accumulator["raw_infinite_count"][feature_index]
                ),
                "raw_nonfinite_count": raw_nonfinite_count,
                "raw_zero_count": int(
                    exact_accumulator["raw_zero_count"][feature_index]
                ),
                "fit_finite_count": finite_fit_count,
                "fit_nonfinite_count": fit_nonfinite_count,
                "quantile_sample_finite_count": int(signal_sample.size),
                "quantile_fallback_used": quantile_fallback_used,
                "train_min": exact_minimum,
                "train_quantile_25": float(quantile_25),
                "train_median": float(median_value),
                "train_quantile_75": float(quantile_75),
                "train_max": exact_maximum,
                "train_mean": exact_mean,
                "train_sample_standard_deviation": (
                    exact_standard_deviation
                ),
                "train_interquartile_range": interquartile_range,
                "train_median_absolute_deviation": (
                    median_absolute_deviation
                ),
                "imputation_value": float(median_value),
                "imputation_statistic": "training_median",
                "center_value": float(median_value),
                "center_statistic": "training_median",
                "scale_value": float(scale_value),
                "scale_statistic": scale_statistic,
                "train_constant": train_constant,
                "no_finite_fit_values": no_finite_fit_values,
                "keep_for_model": keep_for_model,
                "drop_reason": drop_reason,
                "add_missing_indicator": add_missing_indicator,
                "missing_indicator_basis": (
                    "care_normal_fit_eligible_training_rows"
                ),
                "missing_indicator_name": (
                    f"missing__{signal_name}"
                    if add_missing_indicator
                    else ""
                ),
                "individual_zeros_retained": True,
            }
        )

    for status_label, status_accumulator in sorted(
        farm_status_accumulators.items(),
        key=lambda item: item[0],
    ):
        status_row_count = int(status_accumulator["row_count"])
        status_considered_normal = bool(
            status_label in NORMAL_STATUS_TYPE_IDS
        )
        train_status_operational_records.append(
            {
                "farm": farm_name,
                "status_type_id": status_label,
                "care_considered_normal": status_considered_normal,
                "row_count": status_row_count,
                "row_fraction_within_farm": (
                    status_row_count / farm_total_rows
                    if farm_total_rows > 0
                    else np.nan
                ),
                "valid_timestamp_rows": int(
                    status_accumulator["valid_timestamp_rows"]
                ),
                "any_finite_rows": int(
                    status_accumulator["any_finite_rows"]
                ),
                "strict_all_zero_rows": int(
                    status_accumulator["strict_all_zero_rows"]
                ),
                "synchronous_all_zero_rows": int(
                    status_accumulator["synchronous_all_zero_rows"]
                ),
                "synchronous_all_zero_fraction": (
                    status_accumulator["synchronous_all_zero_rows"]
                    / status_row_count
                    if status_row_count > 0
                    else np.nan
                ),
                "fit_eligible_rows": int(
                    status_accumulator["fit_eligible_rows"]
                ),
                "fit_eligible_fraction": (
                    status_accumulator["fit_eligible_rows"]
                    / status_row_count
                    if status_row_count > 0
                    else np.nan
                ),
                "status_used_as_exclusion_rule": bool(
                    not status_considered_normal
                ),
            }
        )

    del reservoir_values, reservoir_priorities, exact_accumulator
    gc.collect()


# ----------------------------------------------------------------------------
# 6. Materialize reusable tables and validate conservation/leakage boundaries
# ----------------------------------------------------------------------------

train_preprocessing_fit_case_audit = pd.DataFrame(
    train_preprocessing_fit_case_records
)
train_preprocessing_signal_parameters = pd.DataFrame(
    train_preprocessing_signal_parameter_records
)
train_status_operational_audit = pd.DataFrame(
    train_status_operational_records
)

if not train_preprocessing_signal_parameters.empty:
    train_preprocessing_signal_parameters = (
        train_preprocessing_signal_parameters
        .sort_values(["farm", "feature_index"], kind="stable")
        .reset_index(drop=True)
    )

if not train_preprocessing_fit_case_audit.empty:
    train_preprocessing_fit_case_audit = (
        train_preprocessing_fit_case_audit
        .sort_values(["farm", "event_id"], kind="stable")
        .reset_index(drop=True)
    )

if not train_status_operational_audit.empty:
    train_status_operational_audit = (
        train_status_operational_audit
        .sort_values(["farm", "status_type_id"], kind="stable")
        .reset_index(drop=True)
    )

train_preprocessing_farm_summary = (
    train_preprocessing_fit_case_audit
    .groupby("farm", as_index=False, sort=False)
    .agg(
        training_cases=("event_id", "size"),
        training_assets=("canonical_asset_key", "nunique"),
        training_source_rows=("row_count", "sum"),
        valid_timestamp_rows=("valid_timestamp_rows", "sum"),
        invalid_timestamp_rows=("invalid_timestamp_rows", "sum"),
        rows_with_any_finite_primary_signal=(
            "rows_with_any_finite_primary_signal",
            "sum",
        ),
        rows_without_finite_primary_signal=(
            "rows_without_finite_primary_signal",
            "sum",
        ),
        strict_all_zero_rows=("strict_all_zero_rows", "sum"),
        synchronous_all_zero_rows=("synchronous_all_zero_rows", "sum"),
        care_normal_status_rows=("care_normal_status_rows", "sum"),
        status_excluded_rows=("status_excluded_rows", "sum"),
        fit_eligible_rows=("fit_eligible_rows", "sum"),
        fit_excluded_rows=("fit_excluded_rows", "sum"),
    )
)

farm_signal_summary = (
    train_preprocessing_signal_parameters
    .groupby("farm", as_index=False, sort=False)
    .agg(
        primary_signal_count=("signal_name", "size"),
        kept_signal_count=("keep_for_model", "sum"),
        dropped_signal_count=(
            "keep_for_model",
            lambda values: int((~values.astype(bool)).sum()),
        ),
        train_constant_signal_count=("train_constant", "sum"),
        no_finite_signal_count=("no_finite_fit_values", "sum"),
        missing_indicator_count=("add_missing_indicator", "sum"),
        quantile_fallback_signal_count=("quantile_fallback_used", "sum"),
    )
)

train_preprocessing_farm_summary = (
    train_preprocessing_farm_summary
    .merge(
        farm_signal_summary,
        on="farm",
        how="left",
        validate="one_to_one",
    )
)
train_preprocessing_farm_summary["fit_eligible_fraction"] = (
    train_preprocessing_farm_summary["fit_eligible_rows"]
    / train_preprocessing_farm_summary["training_source_rows"]
)
train_preprocessing_farm_summary[
    "quantile_reservoir_capacity"
] = CELL_7_QUANTILE_SAMPLE_ROWS

expected_parameter_rows = int(
    sum(
        len(FARM_SCHEMAS[farm_name]["primary_signal_columns"])
        for farm_name in training_farms
    )
)

if sensor_body_read_counts != {
    "train": EXPECTED_SPLIT_CASE_COUNTS["train"],
    "validation": 0,
    "test": 0,
}:
    cell_7_validation_errors.append(
        "Sensor-body read counts violate the train-only boundary: "
        f"{sensor_body_read_counts}."
    )

if len(train_preprocessing_fit_case_audit) != (
    EXPECTED_SPLIT_CASE_COUNTS["train"]
):
    cell_7_validation_errors.append(
        "The fit-case audit does not contain exactly 67 training cases."
    )

if train_preprocessing_fit_case_audit.duplicated(
    subset=["farm", "event_id"]
).any():
    cell_7_validation_errors.append(
        "The fit-case audit contains duplicate training case keys."
    )

if len(train_preprocessing_signal_parameters) != expected_parameter_rows:
    cell_7_validation_errors.append(
        f"The parameter table contains "
        f"{len(train_preprocessing_signal_parameters)} rows; expected "
        f"{expected_parameter_rows}."
    )

if train_preprocessing_signal_parameters.duplicated(
    subset=["farm", "signal_name"]
).any():
    cell_7_validation_errors.append(
        "The parameter table contains duplicate farm/signal keys."
    )

kept_parameter_rows = train_preprocessing_signal_parameters.loc[
    train_preprocessing_signal_parameters["keep_for_model"].eq(True)
]

if kept_parameter_rows.empty:
    cell_7_validation_errors.append(
        "No primary signals remain eligible for future modeling."
    )

if not kept_parameter_rows.empty:
    invalid_imputation_mask = ~np.isfinite(
        kept_parameter_rows["imputation_value"].to_numpy(dtype=float)
    )
    invalid_center_mask = ~np.isfinite(
        kept_parameter_rows["center_value"].to_numpy(dtype=float)
    )
    invalid_scale_mask = (
        ~np.isfinite(
            kept_parameter_rows["scale_value"].to_numpy(dtype=float)
        )
        | kept_parameter_rows["scale_value"].le(0).to_numpy()
    )

    if invalid_imputation_mask.any():
        cell_7_validation_errors.append(
            f"{int(invalid_imputation_mask.sum())} kept signals have an "
            "invalid imputation value."
        )

    if invalid_center_mask.any():
        cell_7_validation_errors.append(
            f"{int(invalid_center_mask.sum())} kept signals have an invalid "
            "center value."
        )

    if invalid_scale_mask.any():
        cell_7_validation_errors.append(
            f"{int(invalid_scale_mask.sum())} kept signals have a nonfinite "
            "or nonpositive scale value."
        )

farm_without_kept_signals = (
    train_preprocessing_farm_summary.loc[
        train_preprocessing_farm_summary["kept_signal_count"].le(0),
        "farm",
    ]
    .astype(str)
    .tolist()
)

if farm_without_kept_signals:
    cell_7_validation_errors.append(
        "No signals remain for: " + ", ".join(farm_without_kept_signals)
    )

case_row_conservation_mask = (
    train_preprocessing_fit_case_audit["fit_eligible_rows"]
    + train_preprocessing_fit_case_audit["fit_excluded_rows"]
    != train_preprocessing_fit_case_audit["row_count"]
)

if case_row_conservation_mask.any():
    cell_7_validation_errors.append(
        f"{int(case_row_conservation_mask.sum())} training cases fail fit-row "
        "conservation."
    )

status_row_totals = (
    train_status_operational_audit
    .groupby("farm", as_index=False, sort=False)["row_count"]
    .sum()
    .rename(columns={"row_count": "status_row_count"})
)
status_conservation = train_preprocessing_farm_summary[
    ["farm", "training_source_rows"]
].merge(
    status_row_totals,
    on="farm",
    how="left",
    validate="one_to_one",
)

if not status_conservation["training_source_rows"].eq(
    status_conservation["status_row_count"]
).all():
    cell_7_validation_errors.append(
        "status_type_id row counts do not conserve all training source rows."
    )

case_status_conservation_mask = (
    train_preprocessing_fit_case_audit["care_normal_status_rows"]
    + train_preprocessing_fit_case_audit["status_excluded_rows"]
    != train_preprocessing_fit_case_audit["row_count"]
)

if case_status_conservation_mask.any():
    cell_7_validation_errors.append(
        f"{int(case_status_conservation_mask.sum())} training cases fail "
        "normal/abnormal-status row conservation."
    )

abnormal_status_fit_rows = int(
    train_status_operational_audit.loc[
        ~train_status_operational_audit[
            "care_considered_normal"
        ].eq(True),
        "fit_eligible_rows",
    ].sum()
)

if abnormal_status_fit_rows > 0:
    cell_7_validation_errors.append(
        f"{abnormal_status_fit_rows} abnormal- or missing-status rows "
        "incorrectly contributed to preprocessing fit."
    )

farms_without_normal_fit_rows = (
    train_preprocessing_farm_summary.loc[
        train_preprocessing_farm_summary["fit_eligible_rows"].le(0),
        "farm",
    ]
    .astype(str)
    .tolist()
)

if farms_without_normal_fit_rows:
    cell_7_validation_errors.append(
        "No CARE-normal fit rows remain for: "
        + ", ".join(farms_without_normal_fit_rows)
    )

unusable_signal_count = int(
    train_preprocessing_signal_parameters[
        "no_finite_fit_values"
    ].sum()
)
constant_signal_count = int(
    train_preprocessing_signal_parameters["train_constant"].sum()
)
missing_indicator_count = int(
    train_preprocessing_signal_parameters["add_missing_indicator"].sum()
)
non_iqr_scale_count = int(
    kept_parameter_rows["scale_statistic"]
    .ne("interquartile_range")
    .sum()
)
status_excluded_row_count = int(
    train_preprocessing_fit_case_audit["status_excluded_rows"].sum()
)

if unusable_signal_count > 0:
    cell_7_validation_warnings.append(
        f"{unusable_signal_count} signals have no finite fit-eligible "
        "training values and are marked for later removal."
    )

if constant_signal_count > 0:
    cell_7_validation_warnings.append(
        f"{constant_signal_count} signals are constant in fit-eligible "
        "training observations and are marked for later removal."
    )

if missing_indicator_count > 0:
    cell_7_validation_warnings.append(
        f"{missing_indicator_count} kept signals contain train-observed "
        "nonfinite values and will receive missing indicators later."
    )

if non_iqr_scale_count > 0:
    cell_7_validation_warnings.append(
        f"{non_iqr_scale_count} kept signals require a train-only scale "
        "fallback because their sampled IQR is numerically zero."
    )

if status_excluded_row_count > 0:
    cell_7_validation_warnings.append(
        f"{status_excluded_row_count} training rows have CARE-abnormal or "
        "missing status IDs and were excluded from normal-behavior "
        "preprocessing fit."
    )

cell_7_constraint_records = [
    {
        "constraint": "frozen_assignment_sha256_matches",
        "passed": observed_assignment_sha256 == FROZEN_ASSIGNMENT_SHA256,
        "observed": observed_assignment_sha256,
        "expected": FROZEN_ASSIGNMENT_SHA256,
    },
    {
        "constraint": "only_training_sensor_bodies_read",
        "passed": sensor_body_read_counts == {
            "train": 67,
            "validation": 0,
            "test": 0,
        },
        "observed": str(sensor_body_read_counts),
        "expected": "{'train': 67, 'validation': 0, 'test': 0}",
    },
    {
        "constraint": "only_care_normal_status_rows_fit",
        "passed": abnormal_status_fit_rows == 0,
        "observed": abnormal_status_fit_rows,
        "expected": 0,
    },
    {
        "constraint": "normal_status_ids_are_0_and_2",
        "passed": NORMAL_STATUS_TYPE_IDS == frozenset({"0", "2"}),
        "observed": " | ".join(sorted(NORMAL_STATUS_TYPE_IDS)),
        "expected": "0 | 2",
    },
    {
        "constraint": "all_67_training_cases_fitted_once",
        "passed": bool(
            len(train_preprocessing_fit_case_audit) == 67
            and not train_preprocessing_fit_case_audit.duplicated(
                subset=["farm", "event_id"]
            ).any()
        ),
        "observed": int(len(train_preprocessing_fit_case_audit)),
        "expected": 67,
    },
    {
        "constraint": "one_parameter_row_per_farm_primary_signal",
        "passed": len(train_preprocessing_signal_parameters)
        == expected_parameter_rows,
        "observed": int(len(train_preprocessing_signal_parameters)),
        "expected": expected_parameter_rows,
    },
    {
        "constraint": "positive_finite_scales_for_kept_signals",
        "passed": bool(
            not kept_parameter_rows.empty
            and np.isfinite(
                kept_parameter_rows["scale_value"].to_numpy(dtype=float)
            ).all()
            and kept_parameter_rows["scale_value"].gt(0).all()
        ),
        "observed": int(len(kept_parameter_rows)),
        "expected": "all kept signals",
    },
    {
        "constraint": "at_least_one_kept_signal_per_farm",
        "passed": not farm_without_kept_signals,
        "observed": int(
            train_preprocessing_farm_summary[
                "kept_signal_count"
            ].gt(0).sum()
        ),
        "expected": int(len(training_farms)),
    },
    {
        "constraint": "fit_row_conservation",
        "passed": not case_row_conservation_mask.any(),
        "observed": int(case_row_conservation_mask.sum()),
        "expected": 0,
    },
    {
        "constraint": "status_row_conservation",
        "passed": bool(
            status_conservation["training_source_rows"].eq(
                status_conservation["status_row_count"]
            ).all()
        ),
        "observed": int(
            status_conservation["training_source_rows"].sub(
                status_conservation["status_row_count"]
            ).abs().sum()
        ),
        "expected": 0,
    },
    {
        "constraint": "source_data_not_modified",
        "passed": True,
        "observed": False,
        "expected": False,
    },
]

train_preprocessing_constraint_audit = pd.DataFrame(
    cell_7_constraint_records
)

failed_cell_7_constraints = train_preprocessing_constraint_audit.loc[
    ~train_preprocessing_constraint_audit["passed"].eq(True)
]

if not failed_cell_7_constraints.empty:
    cell_7_validation_errors.extend(
        "Constraint failed: " + str(constraint_name)
        for constraint_name in failed_cell_7_constraints["constraint"]
    )

if cell_7_validation_errors:
    raise RuntimeError(
        "CELL 7 VALIDATION FAILED:\n  - "
        + "\n  - ".join(cell_7_validation_errors)
    )


# ----------------------------------------------------------------------------
# 7. Save train-only preprocessing outputs and manifest
# ----------------------------------------------------------------------------

train_preprocessing_fit_case_audit.to_csv(
    TABLE_DIR / "care_train_preprocessing_fit_case_audit.csv",
    index=False,
)

train_preprocessing_signal_parameters.to_csv(
    TABLE_DIR / "care_train_preprocessing_signal_parameters.csv",
    index=False,
)

train_preprocessing_farm_summary.to_csv(
    TABLE_DIR / "care_train_preprocessing_farm_summary.csv",
    index=False,
)

train_status_operational_audit.to_csv(
    TABLE_DIR / "care_train_status_operational_audit.csv",
    index=False,
)

train_preprocessing_constraint_audit.to_csv(
    TABLE_DIR / "care_train_preprocessing_constraint_audit.csv",
    index=False,
)

cell_7_manifest = {
    "cell": 7,
    "purpose": (
        "Fit farm-specific preprocessing statistics from training data only"
    ),
    "policy": CELL_7_POLICY,
    "assignment_digest_sha256": observed_assignment_sha256,
    "chunk_rows": CELL_7_CHUNK_ROWS,
    "quantile_sample_rows_per_farm": CELL_7_QUANTILE_SAMPLE_ROWS,
    "random_seed": CELL_7_RANDOM_SEED,
    "care_normal_status_type_ids": sorted(NORMAL_STATUS_TYPE_IDS),
    "sensor_body_read_counts": sensor_body_read_counts,
    "training_case_count": int(len(train_preprocessing_fit_case_audit)),
    "farm_count": int(len(training_farms)),
    "parameter_row_count": int(
        len(train_preprocessing_signal_parameters)
    ),
    "kept_signal_count": int(
        train_preprocessing_signal_parameters["keep_for_model"].sum()
    ),
    "dropped_signal_count": int(
        (~train_preprocessing_signal_parameters["keep_for_model"]).sum()
    ),
    "missing_indicator_count": missing_indicator_count,
    "status_excluded_row_count": status_excluded_row_count,
    "farm_summary": train_preprocessing_farm_summary.to_dict(
        orient="records"
    ),
    "constraint_audit": train_preprocessing_constraint_audit.to_dict(
        orient="records"
    ),
    "validation_warnings": cell_7_validation_warnings,
    "validation_errors": cell_7_validation_errors,
    "validation_sensor_bodies_read": False,
    "test_sensor_bodies_read": False,
    "source_data_modified": False,
}

save_cell_7_json(
    cell_7_manifest,
    OUTPUT_ROOT / "train_only_preprocessing_manifest.json",
)


# ----------------------------------------------------------------------------
# 8. Display summaries and successful completion
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAIN-ONLY PREPROCESSING FARM SUMMARY")
print("=" * 80)
display(train_preprocessing_farm_summary)

print("\n" + "=" * 80)
print("TRAIN-ONLY PREPROCESSING CONSTRAINT AUDIT")
print("=" * 80)
display(train_preprocessing_constraint_audit)

print("\n" + "=" * 80)
print("TRAIN STATUS / SYNCHRONOUS-ZERO AUDIT")
print("=" * 80)
display(train_status_operational_audit)

if cell_7_validation_warnings:
    print("\nTrain-only preprocessing findings")

    for warning_message in cell_7_validation_warnings:
        print(f"  - {warning_message}")

print("\n" + "=" * 80)
print("CELL 7 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Training cases read       : {sensor_body_read_counts['train']}")
print(f"Validation cases read     : {sensor_body_read_counts['validation']}")
print(f"Test cases read           : {sensor_body_read_counts['test']}")
print(
    "Primary signal parameters: "
    f"{len(train_preprocessing_signal_parameters)}"
)
print(
    "Signals kept / dropped   : "
    f"{int(train_preprocessing_signal_parameters['keep_for_model'].sum())} / "
    f"{int((~train_preprocessing_signal_parameters['keep_for_model']).sum())}"
)
print(f"Missing indicators       : {missing_indicator_count}")
print(f"Status-excluded fit rows : {status_excluded_row_count}")
print(f"Assignment SHA-256       : {observed_assignment_sha256}")
print(f"Output directory         : {TABLE_DIR}")
print("Source data modified     : No")
print("Reusable objects:")
print("  - train_case_registry_cell_7")
print("  - train_preprocessing_fit_case_audit")
print("  - train_preprocessing_signal_parameters")
print("  - train_preprocessing_farm_summary")
print("  - train_status_operational_audit")
print("  - train_preprocessing_constraint_audit")
print("  - cell_7_manifest")
print("Created:")
print("  - care_train_preprocessing_fit_case_audit.csv")
print("  - care_train_preprocessing_signal_parameters.csv")
print("  - care_train_preprocessing_farm_summary.csv")
print("  - care_train_status_operational_audit.csv")
print("  - care_train_preprocessing_constraint_audit.csv")
print("  - train_only_preprocessing_manifest.json")
print("=" * 80)
