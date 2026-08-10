"""Cell 3N — split-aware feature audit and train-only preprocessing.

Run once in the notebook namespace after the accepted Cell 3M split:

    %run -i cell_3n_train_only_preprocessing.py

The cell resolves raw sensor measurements by the immutable source key
(``event_key``, ``source_row_index``), fits every quality decision and every
preprocessing statistic on training assets only, and applies the frozen
transformation to validation and test rows. Validation is used only for
non-destructive shift diagnostics. Test distributions are not summarized,
ranked, plotted, or used in any decision.

Preferred explicit input (optional):

    FEATURE_SOURCE_3N = <DataFrame with event_key, source_row_index, sensors>

or:

    FEATURE_SOURCE_3N = {event_key: raw_event_dataframe, ...}

If ``FEATURE_SOURCE_3N`` is absent, the cell safely searches existing notebook
DataFrames and event-frame mappings. It refuses to continue unless all 213,537
eligible manifest keys resolve exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Lock and validate the accepted Cell 3M checkpoint
# -----------------------------------------------------------------------------

if "ROW_LABEL_MANIFEST" not in globals():
    raise NameError("Run Cells 3I–3M before Cell 3N.")

REQUIRED_MANIFEST_COLUMNS_3N = {
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

missing_manifest_columns_3n = (
    REQUIRED_MANIFEST_COLUMNS_3N - set(ROW_LABEL_MANIFEST.columns)
)
if missing_manifest_columns_3n:
    raise ValueError(
        "ROW_LABEL_MANIFEST is missing Cell 3N columns: "
        f"{sorted(missing_manifest_columns_3n)}"
    )

EXPECTED_ELIGIBLE_ROWS_3N = 213_537
EXPECTED_SPLIT_ROWS_3N = {
    "train": 152_236,
    "validation": 30_261,
    "test": 31_040,
}
EXPECTED_SPLIT_ASSETS_3N = {
    "train": 25,
    "validation": 5,
    "test": 6,
}
EXPECTED_SPLIT_EVENTS_3N = {
    "train": 64,
    "validation": 14,
    "test": 15,
}
SPLIT_ORDER_3N = ["train", "validation", "test"]
KEY_COLUMNS_3N = ["event_key", "source_row_index"]
RANDOM_SEED_3N = 20260810

# Training-only quality policy. Extreme values are audited but deliberately not
# clipped: genuine fault excursions may carry useful anomaly information.
MIN_TRAIN_NUMERIC_PARSE_FRACTION_3N = 0.98
MAX_TRAIN_MISSING_FRACTION_3N = 0.30
MAX_TRAIN_FARM_MISSING_FRACTION_3N = 0.50
MIN_TRAIN_FINITE_VALUES_3N = 100
NUMERICAL_EPSILON_3N = 1e-12
CORRELATION_ALERT_3N = 0.98
CORRELATION_SAMPLE_ROWS_3N = 50_000
PSI_EPSILON_3N = 1e-6

eligible_mask_3n = (
    ROW_LABEL_MANIFEST["modeling_eligible"].fillna(False).astype(bool)
)

if int(eligible_mask_3n.sum()) != EXPECTED_ELIGIBLE_ROWS_3N:
    raise ValueError(
        "Cell 3N expects 213,537 post-deduplication eligible rows, but found "
        f"{int(eligible_mask_3n.sum()):,}."
    )

if ROW_LABEL_MANIFEST.loc[
    eligible_mask_3n, "split_assignment"
].isna().any():
    raise ValueError("At least one eligible row lacks a Cell 3M split.")

if ROW_LABEL_MANIFEST.loc[
    ~eligible_mask_3n, "split_assignment"
].notna().any():
    raise ValueError("A modeling-ineligible row has a split assignment.")

manifest_length_before_3n = len(ROW_LABEL_MANIFEST)
manifest_index_before_3n = ROW_LABEL_MANIFEST.index.copy()
manifest_hash_before_3n = pd.util.hash_pandas_object(
    ROW_LABEL_MANIFEST,
    index=True,
).to_numpy(copy=True)

eligible_manifest_3n = ROW_LABEL_MANIFEST.loc[
    eligible_mask_3n,
    [
        "farm",
        "asset_id",
        "asset_key",
        "event_key",
        "source_row_index",
        "timestamp_utc",
        "final_label",
        "split_assignment",
    ],
].copy()

for column_3n in [
    "farm",
    "asset_id",
    "asset_key",
    "event_key",
    "final_label",
    "split_assignment",
]:
    eligible_manifest_3n[column_3n] = (
        eligible_manifest_3n[column_3n].astype("string").str.strip()
    )

eligible_manifest_3n["final_label"] = (
    eligible_manifest_3n["final_label"].str.lower()
)
eligible_manifest_3n["timestamp_utc"] = pd.to_datetime(
    eligible_manifest_3n["timestamp_utc"],
    errors="coerce",
    utc=True,
)
eligible_manifest_3n["source_row_index"] = pd.to_numeric(
    eligible_manifest_3n["source_row_index"],
    errors="coerce",
)

if eligible_manifest_3n[
    [
        "farm",
        "asset_id",
        "asset_key",
        "event_key",
        "source_row_index",
        "timestamp_utc",
        "final_label",
        "split_assignment",
    ]
].isna().any().any():
    raise ValueError("Eligible manifest rows contain missing critical values.")

if not np.isclose(
    eligible_manifest_3n["source_row_index"],
    np.floor(eligible_manifest_3n["source_row_index"]),
).all():
    raise ValueError("Manifest source_row_index contains non-integer values.")

eligible_manifest_3n["source_row_index"] = eligible_manifest_3n[
    "source_row_index"
].astype("int64")

if eligible_manifest_3n[KEY_COLUMNS_3N].duplicated().any():
    raise ValueError("Eligible manifest source keys are not unique.")

if sorted(eligible_manifest_3n["farm"].astype(str).unique()) != [
    "A",
    "B",
    "C",
]:
    raise ValueError("Cell 3N expects farms A, B, and C.")

if sorted(eligible_manifest_3n["final_label"].astype(str).unique()) != [
    "anomaly",
    "normal",
]:
    raise ValueError("Cell 3N expects labels normal and anomaly.")

observed_split_rows_3n = (
    eligible_manifest_3n["split_assignment"]
    .value_counts()
    .reindex(SPLIT_ORDER_3N, fill_value=0)
    .astype(int)
    .to_dict()
)
observed_split_assets_3n = (
    eligible_manifest_3n.groupby("split_assignment")["asset_key"]
    .nunique()
    .reindex(SPLIT_ORDER_3N, fill_value=0)
    .astype(int)
    .to_dict()
)
observed_split_events_3n = (
    eligible_manifest_3n.groupby("split_assignment")["event_key"]
    .nunique()
    .reindex(SPLIT_ORDER_3N, fill_value=0)
    .astype(int)
    .to_dict()
)

if observed_split_rows_3n != EXPECTED_SPLIT_ROWS_3N:
    raise ValueError(
        "Cell 3M row assignment differs from the accepted checkpoint: "
        f"{observed_split_rows_3n}."
    )
if observed_split_assets_3n != EXPECTED_SPLIT_ASSETS_3N:
    raise ValueError(
        "Cell 3M asset assignment differs from the accepted checkpoint: "
        f"{observed_split_assets_3n}."
    )
if observed_split_events_3n != EXPECTED_SPLIT_EVENTS_3N:
    raise ValueError(
        "Cell 3M event assignment differs from the accepted checkpoint: "
        f"{observed_split_events_3n}."
    )

assets_crossing_splits_3n = int(
    (
        eligible_manifest_3n.groupby("asset_key")["split_assignment"]
        .nunique()
        .gt(1)
    ).sum()
)
events_crossing_splits_3n = int(
    (
        eligible_manifest_3n.groupby("event_key")["split_assignment"]
        .nunique()
        .gt(1)
    ).sum()
)
if assets_crossing_splits_3n or events_crossing_splits_3n:
    raise ValueError("The accepted asset/event isolation no longer holds.")

eligible_key_index_3n = pd.MultiIndex.from_frame(
    eligible_manifest_3n[KEY_COLUMNS_3N]
)
eligible_event_keys_3n = frozenset(
    eligible_manifest_3n["event_key"].astype(str).unique()
)


# -----------------------------------------------------------------------------
# 2. Resolve a raw measurement source without guessing row identity
# -----------------------------------------------------------------------------

RESERVED_COLUMN_EXACT_3N = {
    "farm",
    "farm_id",
    "asset",
    "asset_id",
    "asset_key",
    "turbine",
    "turbine_id",
    "wtg",
    "event",
    "event_id",
    "event_key",
    "source_row_index",
    "raw_id",
    "row_id",
    "index",
    "timestamp",
    "timestamp_utc",
    "datetime",
    "date",
    "time",
    "label",
    "class",
    "target",
    "state",
    "status_label",
    "source_event_label",
    "final_label",
    "modeling_eligible",
    "split",
    "split_assignment",
    "measurement_schema_id",
    "measurement_fingerprint",
    "file",
    "filename",
    "file_path",
    "path",
}

RESERVED_COLUMN_PATTERN_3N = re.compile(
    r"(^|_)(label|class|target|split|eligible|event|asset|turbine|farm|"
    r"timestamp|datetime|date|time|index|row|raw|schema|fingerprint|"
    r"filename|filepath|path|audit|reason|action|fault_code|anomaly_flag)"
    r"($|_)",
    flags=re.IGNORECASE,
)


def slug_3n(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return value or "item"


def normalized_source_index_3n(series, source_name):
    numeric_3n = pd.to_numeric(series, errors="coerce")
    if numeric_3n.isna().any() or not np.isclose(
        numeric_3n, np.floor(numeric_3n)
    ).all():
        raise ValueError(
            f"{source_name} contains invalid source-row indices."
        )
    return numeric_3n.astype("int64")


def is_reserved_feature_name_3n(column):
    normalized_3n = slug_3n(column)
    return (
        normalized_3n in RESERVED_COLUMN_EXACT_3N
        or RESERVED_COLUMN_PATTERN_3N.search(normalized_3n) is not None
    )


def candidate_feature_columns_3n(frame):
    return [
        column_3n
        for column_3n in frame.columns
        if column_3n not in KEY_COLUMNS_3N
        and not is_reserved_feature_name_3n(column_3n)
    ]


def normalize_keyed_frame_3n(frame, source_name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{source_name} is not a pandas DataFrame.")
    if frame.columns.duplicated().any():
        duplicated_3n = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(
            f"{source_name} has duplicate columns: {duplicated_3n}."
        )
    missing_keys_3n = set(KEY_COLUMNS_3N) - set(frame.columns)
    if missing_keys_3n:
        raise ValueError(
            f"{source_name} lacks source keys {sorted(missing_keys_3n)}."
        )
    output_3n = frame.copy(deep=False)
    output_3n["event_key"] = output_3n["event_key"].astype("string").str.strip()
    output_3n["source_row_index"] = normalized_source_index_3n(
        output_3n["source_row_index"], source_name
    )
    if output_3n["event_key"].isna().any() or output_3n[
        "event_key"
    ].eq("").any():
        raise ValueError(f"{source_name} contains blank event keys.")
    return output_3n


def infer_leaf_event_key_3n(path_parts, frame):
    if "event_key" in frame.columns:
        values_3n = (
            frame["event_key"].dropna().astype(str).str.strip().unique().tolist()
        )
        if len(values_3n) == 1 and values_3n[0] in eligible_event_keys_3n:
            return values_3n[0]

    for part_3n in reversed(path_parts):
        part_text_3n = str(part_3n).strip()
        if part_text_3n in eligible_event_keys_3n:
            return part_text_3n

    if {"farm", "event_id"}.issubset(frame.columns):
        farm_values_3n = frame["farm"].dropna().astype(str).str.strip().unique()
        event_values_3n = frame["event_id"].dropna().astype(str).str.strip().unique()
        if len(farm_values_3n) == 1 and len(event_values_3n) == 1:
            farm_token_3n = farm_values_3n[0].lower().replace("farm_", "")
            event_token_3n = re.sub(r"\.0$", "", event_values_3n[0])
            candidate_3n = f"farm_{farm_token_3n}_event_{event_token_3n}"
            if candidate_3n in eligible_event_keys_3n:
                return candidate_3n

    path_text_3n = "::".join(str(part_3n) for part_3n in path_parts)
    farm_match_3n = re.search(r"(?:farm[_\s-]*)?([abc])", path_text_3n, re.I)
    event_match_3n = re.search(r"(?:event[_\s-]*)?(\d+)(?!.*\d)", path_text_3n, re.I)
    if farm_match_3n and event_match_3n:
        candidate_3n = (
            f"farm_{farm_match_3n.group(1).lower()}_event_{event_match_3n.group(1)}"
        )
        if candidate_3n in eligible_event_keys_3n:
            return candidate_3n
    return None


def iter_dataframe_leaves_3n(value, path=(), depth=0, seen=None):
    if seen is None:
        seen = set()
    object_id_3n = id(value)
    if object_id_3n in seen:
        return
    seen.add(object_id_3n)

    if isinstance(value, pd.DataFrame):
        yield path, value
        return
    if isinstance(value, Mapping) and depth < 4:
        for key_3n, child_3n in value.items():
            yield from iter_dataframe_leaves_3n(
                child_3n,
                path=path + (key_3n,),
                depth=depth + 1,
                seen=seen,
            )


def materialize_event_mapping_3n(mapping, source_name):
    pieces_3n = []
    resolved_events_3n = []
    unresolved_leaf_count_3n = 0

    for path_3n, frame_3n in iter_dataframe_leaves_3n(mapping):
        event_key_3n = infer_leaf_event_key_3n(path_3n, frame_3n)
        if event_key_3n is None:
            unresolved_leaf_count_3n += 1
            continue
        if event_key_3n not in eligible_event_keys_3n:
            continue

        piece_3n = frame_3n.copy(deep=False)
        if "event_key" not in piece_3n.columns:
            piece_3n = piece_3n.assign(event_key=event_key_3n)
        else:
            piece_3n = piece_3n.assign(event_key=event_key_3n)

        if "source_row_index" not in piece_3n.columns:
            if "raw_id" in piece_3n.columns:
                source_indices_3n = piece_3n["raw_id"]
            else:
                source_indices_3n = pd.Series(
                    piece_3n.index,
                    index=piece_3n.index,
                )
            piece_3n = piece_3n.assign(
                source_row_index=normalized_source_index_3n(
                    source_indices_3n,
                    f"{source_name}:{event_key_3n}",
                ).to_numpy()
            )

        pieces_3n.append(piece_3n)
        resolved_events_3n.append(event_key_3n)

    if not pieces_3n:
        raise ValueError(
            f"{source_name} contains no event DataFrames that can be mapped "
            "to eligible event keys."
        )

    output_3n = pd.concat(
        pieces_3n,
        axis=0,
        ignore_index=True,
        sort=False,
    )
    output_3n.attrs["resolved_event_count_3n"] = len(set(resolved_events_3n))
    output_3n.attrs["unresolved_leaf_count_3n"] = unresolved_leaf_count_3n
    return normalize_keyed_frame_3n(output_3n, source_name)


def inspect_source_candidate_3n(name, value, explicit=False):
    try:
        if isinstance(value, pd.DataFrame):
            keyed_3n = normalize_keyed_frame_3n(value, name)
            source_type_3n = "keyed_dataframe"
        elif isinstance(value, Mapping):
            keyed_3n = materialize_event_mapping_3n(value, name)
            source_type_3n = "event_dataframe_mapping"
        else:
            return None, None

        feature_candidates_3n = candidate_feature_columns_3n(keyed_3n)
        if not feature_candidates_3n:
            raise ValueError("no non-metadata feature columns were found")

        duplicated_keys_3n = int(keyed_3n[KEY_COLUMNS_3N].duplicated().sum())
        source_keys_3n = pd.MultiIndex.from_frame(keyed_3n[KEY_COLUMNS_3N])
        matched_keys_3n = int(source_keys_3n.isin(eligible_key_index_3n).sum())
        eligible_keys_matched_3n = int(eligible_key_index_3n.isin(source_keys_3n).sum())
        exact_coverage_3n = (
            eligible_keys_matched_3n == EXPECTED_ELIGIBLE_ROWS_3N
            and duplicated_keys_3n == 0
        )

        record_3n = {
            "source_name": str(name),
            "source_type": source_type_3n,
            "explicit": bool(explicit),
            "source_rows": int(len(keyed_3n)),
            "candidate_feature_columns": int(len(feature_candidates_3n)),
            "eligible_keys_matched": eligible_keys_matched_3n,
            "source_rows_matching_eligible": matched_keys_3n,
            "duplicated_source_keys": duplicated_keys_3n,
            "exact_eligible_coverage": bool(exact_coverage_3n),
            "inspection_error": pd.NA,
        }
        return record_3n, keyed_3n
    except Exception as error_3n:
        if explicit:
            raise
        record_3n = {
            "source_name": str(name),
            "source_type": type(value).__name__,
            "explicit": False,
            "source_rows": int(len(value)) if hasattr(value, "__len__") else pd.NA,
            "candidate_feature_columns": pd.NA,
            "eligible_keys_matched": 0,
            "source_rows_matching_eligible": 0,
            "duplicated_source_keys": pd.NA,
            "exact_eligible_coverage": False,
            "inspection_error": f"{type(error_3n).__name__}: {error_3n}",
        }
        return record_3n, None


source_candidate_records_3n = []
source_candidate_frames_3n = {}
source_resolution_mode_3n = "automatic"

if "FEATURE_SOURCE_3N" in globals():
    source_resolution_mode_3n = "explicit"
    source_record_3n, source_frame_3n = inspect_source_candidate_3n(
        "FEATURE_SOURCE_3N",
        globals()["FEATURE_SOURCE_3N"],
        explicit=True,
    )
    source_candidate_records_3n.append(source_record_3n)
    source_candidate_frames_3n["FEATURE_SOURCE_3N"] = source_frame_3n
else:
    globals_snapshot_3n = list(globals().items())
    excluded_source_names_3n = {
        "ROW_LABEL_MANIFEST",
        "eligible_manifest_3n",
        "ELIGIBLE_MEASUREMENT_INDEX",
        "EXACT_SAME_LABEL_CANONICAL_RETENTION_AUDIT",
    }

    for name_3n, value_3n in globals_snapshot_3n:
        if name_3n in excluded_source_names_3n or name_3n.startswith("_"):
            continue
        if isinstance(value_3n, pd.DataFrame) and set(KEY_COLUMNS_3N).issubset(
            value_3n.columns
        ):
            record_3n, frame_3n = inspect_source_candidate_3n(
                name_3n, value_3n
            )
            if record_3n is not None:
                source_candidate_records_3n.append(record_3n)
                if frame_3n is not None:
                    source_candidate_frames_3n[name_3n] = frame_3n

    mapping_name_pattern_3n = re.compile(
        r"event|data|frame|raw|measurement|source|farm", re.IGNORECASE
    )
    for name_3n, value_3n in globals_snapshot_3n:
        if (
            name_3n.startswith("_")
            or not isinstance(value_3n, Mapping)
            or not mapping_name_pattern_3n.search(name_3n)
        ):
            continue
        record_3n, frame_3n = inspect_source_candidate_3n(name_3n, value_3n)
        if record_3n is not None:
            source_candidate_records_3n.append(record_3n)
            if frame_3n is not None:
                source_candidate_frames_3n[name_3n] = frame_3n


FEATURE_SOURCE_CANDIDATES_3N = pd.DataFrame(source_candidate_records_3n)
if FEATURE_SOURCE_CANDIDATES_3N.empty:
    raise NameError(
        "Cell 3N could not find raw sensor measurements. Define "
        "FEATURE_SOURCE_3N as either a DataFrame containing event_key, "
        "source_row_index, and sensor columns, or a mapping from event_key "
        "to raw event DataFrames; then rerun Cell 3N."
    )

valid_sources_3n = FEATURE_SOURCE_CANDIDATES_3N.loc[
    FEATURE_SOURCE_CANDIDATES_3N["exact_eligible_coverage"].fillna(False)
].copy()
if valid_sources_3n.empty:
    compact_candidates_3n = FEATURE_SOURCE_CANDIDATES_3N.loc[
        :,
        [
            "source_name",
            "source_type",
            "source_rows",
            "candidate_feature_columns",
            "eligible_keys_matched",
            "duplicated_source_keys",
            "inspection_error",
        ],
    ]
    raise ValueError(
        "No measurement source covers every eligible manifest key exactly "
        "once. Inspect FEATURE_SOURCE_CANDIDATES_3N. Preferred repair:\n"
        "FEATURE_SOURCE_3N = <keyed measurement DataFrame or event mapping>\n\n"
        + compact_candidates_3n.to_string(index=False)
    )

valid_sources_3n = valid_sources_3n.sort_values(
    ["explicit", "candidate_feature_columns", "source_rows", "source_name"],
    ascending=[False, False, True, True],
).reset_index(drop=True)

selected_source_name_3n = str(valid_sources_3n.iloc[0]["source_name"])
selected_source_type_3n = str(valid_sources_3n.iloc[0]["source_type"])
MEASUREMENT_SOURCE_3N = source_candidate_frames_3n[selected_source_name_3n]

if MEASUREMENT_SOURCE_3N[KEY_COLUMNS_3N].duplicated().any():
    raise ValueError("The selected measurement source has duplicate source keys.")

if "FEATURE_COLUMNS_3N" in globals() and globals()["FEATURE_COLUMNS_3N"] is not None:
    requested_feature_columns_3n = list(globals()["FEATURE_COLUMNS_3N"])
    if len(requested_feature_columns_3n) != len(set(requested_feature_columns_3n)):
        raise ValueError("FEATURE_COLUMNS_3N contains duplicates.")
    missing_requested_features_3n = set(requested_feature_columns_3n) - set(
        MEASUREMENT_SOURCE_3N.columns
    )
    if missing_requested_features_3n:
        raise ValueError(
            "FEATURE_COLUMNS_3N is missing from the measurement source: "
            f"{sorted(missing_requested_features_3n)}"
        )
    leaking_requested_features_3n = [
        column_3n
        for column_3n in requested_feature_columns_3n
        if is_reserved_feature_name_3n(column_3n)
    ]
    if leaking_requested_features_3n:
        raise ValueError(
            "FEATURE_COLUMNS_3N includes identity/label/split-like columns: "
            f"{leaking_requested_features_3n}"
        )
    candidate_features_3n = requested_feature_columns_3n
    feature_column_mode_3n = "explicit"
else:
    candidate_features_3n = candidate_feature_columns_3n(MEASUREMENT_SOURCE_3N)
    feature_column_mode_3n = "automatic_metadata_exclusion"

if not candidate_features_3n:
    raise ValueError("No candidate sensor columns remain after metadata exclusion.")

source_subset_3n = MEASUREMENT_SOURCE_3N.loc[
    :,
    KEY_COLUMNS_3N + candidate_features_3n,
].copy()

RAW_ELIGIBLE_FEATURES_3N = eligible_manifest_3n.merge(
    source_subset_3n,
    on=KEY_COLUMNS_3N,
    how="left",
    validate="one_to_one",
    indicator="_measurement_match_3n",
)

unmatched_eligible_rows_3n = int(
    RAW_ELIGIBLE_FEATURES_3N["_measurement_match_3n"].ne("both").sum()
)
if unmatched_eligible_rows_3n:
    raise ValueError(
        f"{unmatched_eligible_rows_3n:,} eligible rows lack raw measurements."
    )
RAW_ELIGIBLE_FEATURES_3N = RAW_ELIGIBLE_FEATURES_3N.drop(
    columns="_measurement_match_3n"
)

if len(RAW_ELIGIBLE_FEATURES_3N) != EXPECTED_ELIGIBLE_ROWS_3N:
    raise ValueError("Measurement join did not conserve eligible rows.")


# -----------------------------------------------------------------------------
# 3. Fit feature-quality decisions using training assets only
# -----------------------------------------------------------------------------

train_mask_3n = RAW_ELIGIBLE_FEATURES_3N["split_assignment"].eq("train")
validation_mask_3n = RAW_ELIGIBLE_FEATURES_3N[
    "split_assignment"
].eq("validation")
test_mask_3n = RAW_ELIGIBLE_FEATURES_3N["split_assignment"].eq("test")

if int(train_mask_3n.sum()) != EXPECTED_SPLIT_ROWS_3N["train"]:
    raise ValueError("Training-row count changed during the measurement join.")

numeric_feature_data_3n = pd.DataFrame(
    index=RAW_ELIGIBLE_FEATURES_3N.index
)
feature_quality_records_3n = []

train_farms_3n = sorted(
    RAW_ELIGIBLE_FEATURES_3N.loc[train_mask_3n, "farm"].astype(str).unique()
)
if train_farms_3n != ["A", "B", "C"]:
    raise ValueError("Training assets do not cover all three farms.")

for feature_3n in candidate_features_3n:
    original_3n = RAW_ELIGIBLE_FEATURES_3N[feature_3n]
    numeric_3n = pd.to_numeric(original_3n, errors="coerce").astype("float64")
    numeric_3n = numeric_3n.mask(~np.isfinite(numeric_3n), np.nan)
    numeric_feature_data_3n[feature_3n] = numeric_3n

    train_original_3n = original_3n.loc[train_mask_3n]
    train_values_3n = numeric_3n.loc[train_mask_3n]
    original_nonmissing_3n = train_original_3n.notna()
    finite_train_3n = train_values_3n.dropna()
    parse_denominator_3n = int(original_nonmissing_3n.sum())
    parse_numerator_3n = int(
        (original_nonmissing_3n & train_values_3n.notna()).sum()
    )
    parse_fraction_3n = (
        parse_numerator_3n / parse_denominator_3n
        if parse_denominator_3n
        else 0.0
    )
    train_missing_fraction_3n = float(train_values_3n.isna().mean())

    farm_missing_fractions_3n = {}
    for farm_3n in train_farms_3n:
        farm_mask_3n = train_mask_3n & RAW_ELIGIBLE_FEATURES_3N[
            "farm"
        ].eq(farm_3n)
        farm_missing_fractions_3n[farm_3n] = float(
            numeric_3n.loc[farm_mask_3n].isna().mean()
        )

    finite_count_3n = int(len(finite_train_3n))
    unique_count_3n = int(finite_train_3n.nunique(dropna=True))

    if finite_count_3n:
        q1_3n = float(finite_train_3n.quantile(0.25))
        median_3n = float(finite_train_3n.median())
        q3_3n = float(finite_train_3n.quantile(0.75))
        iqr_3n = float(q3_3n - q1_3n)
        mean_3n = float(finite_train_3n.mean())
        std_3n = float(finite_train_3n.std(ddof=0))
        min_3n = float(finite_train_3n.min())
        max_3n = float(finite_train_3n.max())
        mad_3n = float((finite_train_3n - median_3n).abs().median())
    else:
        q1_3n = median_3n = q3_3n = iqr_3n = np.nan
        mean_3n = std_3n = min_3n = max_3n = mad_3n = np.nan

    decision_3n = "retain"
    reason_3n = "passed_training_only_quality_gates"
    scale_source_3n = "training_iqr"
    scale_3n = iqr_3n

    if parse_fraction_3n < MIN_TRAIN_NUMERIC_PARSE_FRACTION_3N:
        decision_3n = "exclude"
        reason_3n = "insufficient_numeric_parse_fraction_in_training"
    elif finite_count_3n < MIN_TRAIN_FINITE_VALUES_3N:
        decision_3n = "exclude"
        reason_3n = "insufficient_finite_training_values"
    elif train_missing_fraction_3n > MAX_TRAIN_MISSING_FRACTION_3N:
        decision_3n = "exclude"
        reason_3n = "excessive_training_missingness"
    elif max(farm_missing_fractions_3n.values()) > (
        MAX_TRAIN_FARM_MISSING_FRACTION_3N
    ):
        decision_3n = "exclude"
        reason_3n = "excessive_missingness_in_a_training_farm"
    elif unique_count_3n <= 1:
        decision_3n = "exclude"
        reason_3n = "constant_in_training"
    elif not np.isfinite(scale_3n) or abs(scale_3n) <= NUMERICAL_EPSILON_3N:
        scale_3n = std_3n
        scale_source_3n = "training_standard_deviation_fallback"
        if not np.isfinite(scale_3n) or abs(scale_3n) <= NUMERICAL_EPSILON_3N:
            decision_3n = "exclude"
            reason_3n = "near_constant_in_training"

    if decision_3n == "exclude":
        scale_source_3n = pd.NA
        scale_3n = np.nan

    feature_quality_records_3n.append(
        {
            "feature": str(feature_3n),
            "decision": decision_3n,
            "decision_reason": reason_3n,
            "fit_split": "train",
            "train_rows": int(train_mask_3n.sum()),
            "train_original_nonmissing": parse_denominator_3n,
            "train_numeric_parse_fraction": parse_fraction_3n,
            "train_finite_values": finite_count_3n,
            "train_missing_fraction": train_missing_fraction_3n,
            "train_farm_A_missing_fraction": farm_missing_fractions_3n["A"],
            "train_farm_B_missing_fraction": farm_missing_fractions_3n["B"],
            "train_farm_C_missing_fraction": farm_missing_fractions_3n["C"],
            "train_unique_values": unique_count_3n,
            "train_min": min_3n,
            "train_q1": q1_3n,
            "train_median": median_3n,
            "train_q3": q3_3n,
            "train_max": max_3n,
            "train_mean": mean_3n,
            "train_std": std_3n,
            "train_mad": mad_3n,
            "imputation_method": (
                "training_median" if decision_3n == "retain" else pd.NA
            ),
            "imputation_value": (
                median_3n if decision_3n == "retain" else np.nan
            ),
            "scaling_method": (
                "robust_center_and_scale" if decision_3n == "retain" else pd.NA
            ),
            "center_value": (
                median_3n if decision_3n == "retain" else np.nan
            ),
            "scale_value": scale_3n,
            "scale_source": scale_source_3n,
            "outlier_clipping": "none_fault_excursions_preserved",
        }
    )


FEATURE_QUALITY_AUDIT_3N = pd.DataFrame(feature_quality_records_3n).sort_values(
    ["decision", "feature"],
    ascending=[False, True],
).reset_index(drop=True)

FEATURE_NAMES_3N = FEATURE_QUALITY_AUDIT_3N.loc[
    FEATURE_QUALITY_AUDIT_3N["decision"].eq("retain"), "feature"
].astype(str).tolist()
EXCLUDED_FEATURES_3N = FEATURE_QUALITY_AUDIT_3N.loc[
    FEATURE_QUALITY_AUDIT_3N["decision"].eq("exclude"), "feature"
].astype(str).tolist()

if not FEATURE_NAMES_3N:
    raise ValueError(
        "No sensor feature passed the training-only quality gates. Inspect "
        "FEATURE_QUALITY_AUDIT_3N; do not relax gates using validation/test."
    )

PREPROCESSING_PARAMETERS_3N = FEATURE_QUALITY_AUDIT_3N.loc[
    FEATURE_QUALITY_AUDIT_3N["decision"].eq("retain"),
    [
        "feature",
        "fit_split",
        "imputation_method",
        "imputation_value",
        "scaling_method",
        "center_value",
        "scale_value",
        "scale_source",
        "outlier_clipping",
    ],
].reset_index(drop=True)

if not PREPROCESSING_PARAMETERS_3N["fit_split"].eq("train").all():
    raise ValueError("A preprocessing parameter was not fitted on training data.")
if not np.isfinite(
    PREPROCESSING_PARAMETERS_3N[
        ["imputation_value", "center_value", "scale_value"]
    ].to_numpy(dtype=float)
).all():
    raise ValueError("A retained feature has a non-finite fitted parameter.")
if PREPROCESSING_PARAMETERS_3N["scale_value"].abs().le(
    NUMERICAL_EPSILON_3N
).any():
    raise ValueError("A retained feature has a zero preprocessing scale.")


# -----------------------------------------------------------------------------
# 4. Freeze the training transform and apply without refitting
# -----------------------------------------------------------------------------

parameter_lookup_3n = PREPROCESSING_PARAMETERS_3N.set_index("feature")


def transform_features_3n(frame):
    """Apply the already-fitted Cell 3N transform; never refits parameters."""
    missing_features_3n = set(FEATURE_NAMES_3N) - set(frame.columns)
    if missing_features_3n:
        raise ValueError(
            "Transform input is missing retained features: "
            f"{sorted(missing_features_3n)}"
        )
    transformed_3n = pd.DataFrame(index=frame.index)
    for feature_3n in FEATURE_NAMES_3N:
        values_3n = pd.to_numeric(frame[feature_3n], errors="coerce").astype(
            "float64"
        )
        values_3n = values_3n.mask(~np.isfinite(values_3n), np.nan)
        params_3n = parameter_lookup_3n.loc[feature_3n]
        values_3n = values_3n.fillna(float(params_3n["imputation_value"]))
        values_3n = (
            values_3n - float(params_3n["center_value"])
        ) / float(params_3n["scale_value"])
        transformed_3n[feature_3n] = values_3n.astype("float32")
    return transformed_3n


transformed_features_3n = transform_features_3n(
    RAW_ELIGIBLE_FEATURES_3N[FEATURE_NAMES_3N]
)

if transformed_features_3n.isna().any().any():
    raise ValueError("Missing values remain after the frozen transform.")
if not np.isfinite(transformed_features_3n.to_numpy(dtype="float32")).all():
    raise ValueError("Non-finite values remain after the frozen transform.")

metadata_columns_3n = [
    "farm",
    "asset_id",
    "asset_key",
    "event_key",
    "source_row_index",
    "timestamp_utc",
    "final_label",
    "split_assignment",
]

PREPROCESSED_FEATURE_MATRIX_3N = pd.concat(
    [
        RAW_ELIGIBLE_FEATURES_3N[metadata_columns_3n].reset_index(drop=True),
        transformed_features_3n.reset_index(drop=True),
    ],
    axis=1,
)

label_map_3n = {"normal": 0, "anomaly": 1}


def split_outputs_3n(split_name):
    split_mask_3n = PREPROCESSED_FEATURE_MATRIX_3N[
        "split_assignment"
    ].eq(split_name)
    x_3n = PREPROCESSED_FEATURE_MATRIX_3N.loc[
        split_mask_3n, FEATURE_NAMES_3N
    ].reset_index(drop=True)
    y_3n = (
        PREPROCESSED_FEATURE_MATRIX_3N.loc[
            split_mask_3n, "final_label"
        ]
        .map(label_map_3n)
        .astype("int8")
        .reset_index(drop=True)
    )
    meta_3n = PREPROCESSED_FEATURE_MATRIX_3N.loc[
        split_mask_3n, metadata_columns_3n
    ].reset_index(drop=True)
    return x_3n, y_3n, meta_3n


X_TRAIN_3N, Y_TRAIN_3N, META_TRAIN_3N = split_outputs_3n("train")
X_VALIDATION_3N, Y_VALIDATION_3N, META_VALIDATION_3N = split_outputs_3n(
    "validation"
)
X_TEST_3N, Y_TEST_3N, META_TEST_3N = split_outputs_3n("test")

if [len(X_TRAIN_3N), len(X_VALIDATION_3N), len(X_TEST_3N)] != [
    EXPECTED_SPLIT_ROWS_3N[name_3n] for name_3n in SPLIT_ORDER_3N
]:
    raise ValueError("Preprocessed split outputs do not conserve rows.")

PREPROCESSOR_STATE_3N = {
    "cell": "3N",
    "version": 1,
    "fit_split": "train",
    "random_seed": RANDOM_SEED_3N,
    "source_name": selected_source_name_3n,
    "source_type": selected_source_type_3n,
    "feature_column_mode": feature_column_mode_3n,
    "candidate_features": [str(value_3n) for value_3n in candidate_features_3n],
    "retained_features": FEATURE_NAMES_3N,
    "excluded_features": EXCLUDED_FEATURES_3N,
    "label_map": label_map_3n,
    "imputation": "training median",
    "scaling": "training median and IQR; training std fallback",
    "outlier_clipping": "none",
    "test_policy": (
        "frozen transform only; no test feature selection, drift statistics, "
        "ranking, plotting, model selection, or threshold selection"
    ),
    "parameters": PREPROCESSING_PARAMETERS_3N.to_dict(orient="records"),
}


# -----------------------------------------------------------------------------
# 5. Train/validation diagnostics; keep test distributions sealed
# -----------------------------------------------------------------------------

SOURCE_RESOLUTION_AUDIT_3N = pd.DataFrame(
    [
        {
            "resolution_mode": source_resolution_mode_3n,
            "selected_source_name": selected_source_name_3n,
            "selected_source_type": selected_source_type_3n,
            "source_rows": int(len(MEASUREMENT_SOURCE_3N)),
            "eligible_keys_required": EXPECTED_ELIGIBLE_ROWS_3N,
            "eligible_keys_matched": EXPECTED_ELIGIBLE_ROWS_3N,
            "unmatched_eligible_rows": unmatched_eligible_rows_3n,
            "feature_column_mode": feature_column_mode_3n,
            "candidate_features": len(candidate_features_3n),
            "retained_features": len(FEATURE_NAMES_3N),
            "excluded_features": len(EXCLUDED_FEATURES_3N),
        }
    ]
)

transform_audit_records_3n = []
for split_3n in SPLIT_ORDER_3N:
    mask_3n = RAW_ELIGIBLE_FEATURES_3N["split_assignment"].eq(split_3n)
    transformed_split_3n = transformed_features_3n.loc[mask_3n]
    if split_3n == "test":
        raw_missing_cells_3n = pd.NA
        raw_missing_fraction_3n = pd.NA
        transformed_mean_abs_3n = pd.NA
        transformed_std_mean_3n = pd.NA
        distribution_reporting_3n = "sealed"
    else:
        raw_selected_3n = numeric_feature_data_3n.loc[
            mask_3n, FEATURE_NAMES_3N
        ]
        raw_missing_cells_3n = int(raw_selected_3n.isna().sum().sum())
        raw_missing_fraction_3n = float(raw_selected_3n.isna().to_numpy().mean())
        transformed_mean_abs_3n = float(
            transformed_split_3n.mean().abs().mean()
        )
        transformed_std_mean_3n = float(
            transformed_split_3n.std(ddof=0).mean()
        )
        distribution_reporting_3n = "reported"

    transform_audit_records_3n.append(
        {
            "split_assignment": split_3n,
            "rows": int(mask_3n.sum()),
            "features": len(FEATURE_NAMES_3N),
            "raw_missing_cells": raw_missing_cells_3n,
            "raw_missing_fraction": raw_missing_fraction_3n,
            "post_transform_missing_cells": int(
                transformed_split_3n.isna().sum().sum()
            ),
            "post_transform_nonfinite_cells": int(
                (~np.isfinite(transformed_split_3n.to_numpy())).sum()
            ),
            "transformed_mean_absolute_mean": transformed_mean_abs_3n,
            "transformed_mean_feature_std": transformed_std_mean_3n,
            "parameters_fitted_on": "train",
            "distribution_reporting": distribution_reporting_3n,
        }
    )

SPLIT_TRANSFORM_AUDIT_3N = pd.DataFrame(transform_audit_records_3n)

TRAIN_TRANSFORMED_FEATURE_SUMMARY_3N = pd.DataFrame(
    {
        "feature": FEATURE_NAMES_3N,
        "mean": X_TRAIN_3N.mean().reindex(FEATURE_NAMES_3N).to_numpy(),
        "std": X_TRAIN_3N.std(ddof=0).reindex(FEATURE_NAMES_3N).to_numpy(),
        "min": X_TRAIN_3N.min().reindex(FEATURE_NAMES_3N).to_numpy(),
        "q1": X_TRAIN_3N.quantile(0.25).reindex(FEATURE_NAMES_3N).to_numpy(),
        "median": X_TRAIN_3N.median().reindex(FEATURE_NAMES_3N).to_numpy(),
        "q3": X_TRAIN_3N.quantile(0.75).reindex(FEATURE_NAMES_3N).to_numpy(),
        "max": X_TRAIN_3N.max().reindex(FEATURE_NAMES_3N).to_numpy(),
    }
)


def population_stability_index_3n(train_values, comparison_values, bins=10):
    train_values_3n = np.asarray(train_values, dtype=float)
    comparison_values_3n = np.asarray(comparison_values, dtype=float)
    train_values_3n = train_values_3n[np.isfinite(train_values_3n)]
    comparison_values_3n = comparison_values_3n[
        np.isfinite(comparison_values_3n)
    ]
    if not len(train_values_3n) or not len(comparison_values_3n):
        return np.nan, 0

    edges_3n = np.unique(
        np.quantile(train_values_3n, np.linspace(0.0, 1.0, bins + 1))
    )
    if len(edges_3n) < 3:
        return 0.0, max(len(edges_3n) - 1, 1)
    edges_3n[0] = -np.inf
    edges_3n[-1] = np.inf
    train_counts_3n, _ = np.histogram(train_values_3n, bins=edges_3n)
    comparison_counts_3n, _ = np.histogram(
        comparison_values_3n, bins=edges_3n
    )
    train_props_3n = train_counts_3n / train_counts_3n.sum()
    comparison_props_3n = comparison_counts_3n / comparison_counts_3n.sum()
    train_props_3n = np.clip(train_props_3n, PSI_EPSILON_3N, None)
    comparison_props_3n = np.clip(
        comparison_props_3n, PSI_EPSILON_3N, None
    )
    psi_3n = float(
        np.sum(
            (comparison_props_3n - train_props_3n)
            * np.log(comparison_props_3n / train_props_3n)
        )
    )
    return psi_3n, len(edges_3n) - 1


validation_shift_records_3n = []
for feature_3n in FEATURE_NAMES_3N:
    psi_3n, bins_3n = population_stability_index_3n(
        X_TRAIN_3N[feature_3n],
        X_VALIDATION_3N[feature_3n],
    )
    if not np.isfinite(psi_3n):
        shift_band_3n = "not_estimable"
    elif psi_3n < 0.10:
        shift_band_3n = "small"
    elif psi_3n < 0.25:
        shift_band_3n = "moderate"
    else:
        shift_band_3n = "large"
    validation_shift_records_3n.append(
        {
            "feature": feature_3n,
            "comparison": "validation_vs_train",
            "train_defined_bins": bins_3n,
            "population_stability_index": psi_3n,
            "shift_band": shift_band_3n,
            "used_for_feature_exclusion": False,
            "test_distribution_accessed": False,
        }
    )

VALIDATION_SHIFT_AUDIT_3N = pd.DataFrame(
    validation_shift_records_3n
).sort_values(
    "population_stability_index",
    ascending=False,
    na_position="last",
).reset_index(drop=True)

correlation_sample_n_3n = min(CORRELATION_SAMPLE_ROWS_3N, len(X_TRAIN_3N))
if correlation_sample_n_3n < len(X_TRAIN_3N):
    correlation_sample_3n = X_TRAIN_3N.sample(
        n=correlation_sample_n_3n,
        random_state=RANDOM_SEED_3N,
        replace=False,
    )
else:
    correlation_sample_3n = X_TRAIN_3N

TRAIN_SPEARMAN_CORRELATION_3N = correlation_sample_3n.corr(method="spearman")
correlation_pair_records_3n = []
for index_a_3n, feature_a_3n in enumerate(FEATURE_NAMES_3N):
    for feature_b_3n in FEATURE_NAMES_3N[index_a_3n + 1 :]:
        correlation_3n = float(
            TRAIN_SPEARMAN_CORRELATION_3N.loc[feature_a_3n, feature_b_3n]
        )
        correlation_pair_records_3n.append(
            {
                "feature_a": feature_a_3n,
                "feature_b": feature_b_3n,
                "train_spearman_correlation": correlation_3n,
                "absolute_correlation": abs(correlation_3n),
                "high_correlation_alert": abs(correlation_3n)
                >= CORRELATION_ALERT_3N,
                "used_for_feature_exclusion": False,
                "sample_rows": correlation_sample_n_3n,
            }
        )

TRAIN_CORRELATION_PAIR_AUDIT_3N = pd.DataFrame(correlation_pair_records_3n)
if not TRAIN_CORRELATION_PAIR_AUDIT_3N.empty:
    TRAIN_CORRELATION_PAIR_AUDIT_3N = (
        TRAIN_CORRELATION_PAIR_AUDIT_3N.sort_values(
            "absolute_correlation", ascending=False
        ).reset_index(drop=True)
    )
else:
    TRAIN_CORRELATION_PAIR_AUDIT_3N = pd.DataFrame(
        columns=[
            "feature_a",
            "feature_b",
            "train_spearman_correlation",
            "absolute_correlation",
            "high_correlation_alert",
            "used_for_feature_exclusion",
            "sample_rows",
        ]
    )


# -----------------------------------------------------------------------------
# 6. Mutation boundary, leakage discipline, and conservation audit
# -----------------------------------------------------------------------------

manifest_hash_after_3n = pd.util.hash_pandas_object(
    ROW_LABEL_MANIFEST,
    index=True,
).to_numpy(copy=True)
manifest_values_changed_3n = int(
    np.count_nonzero(manifest_hash_before_3n != manifest_hash_after_3n)
)

if len(ROW_LABEL_MANIFEST) != manifest_length_before_3n:
    raise ValueError("Cell 3N changed the manifest row count.")
if not ROW_LABEL_MANIFEST.index.equals(manifest_index_before_3n):
    raise ValueError("Cell 3N changed the manifest index.")
if manifest_values_changed_3n:
    raise ValueError("Cell 3N changed ROW_LABEL_MANIFEST values.")

test_used_for_fitting_3n = int(
    PREPROCESSING_PARAMETERS_3N["fit_split"].ne("train").sum()
)
test_distribution_statistics_exported_3n = int(
    SPLIT_TRANSFORM_AUDIT_3N.loc[
        SPLIT_TRANSFORM_AUDIT_3N["split_assignment"].eq("test"),
        [
            "raw_missing_cells",
            "raw_missing_fraction",
            "transformed_mean_absolute_mean",
            "transformed_mean_feature_std",
        ],
    ].notna().sum().sum()
)

PREPROCESSING_LEAKAGE_AUDIT_3N = pd.DataFrame(
    [
        {
            "check": "eligible_row_conservation",
            "observed": len(PREPROCESSED_FEATURE_MATRIX_3N),
            "required": EXPECTED_ELIGIBLE_ROWS_3N,
        },
        {
            "check": "unmatched_eligible_measurement_keys",
            "observed": unmatched_eligible_rows_3n,
            "required": 0,
        },
        {
            "check": "post_transform_missing_cells",
            "observed": int(transformed_features_3n.isna().sum().sum()),
            "required": 0,
        },
        {
            "check": "post_transform_nonfinite_cells",
            "observed": int(
                (~np.isfinite(transformed_features_3n.to_numpy())).sum()
            ),
            "required": 0,
        },
        {
            "check": "parameters_not_fitted_on_train",
            "observed": test_used_for_fitting_3n,
            "required": 0,
        },
        {
            "check": "test_distribution_statistics_exported",
            "observed": test_distribution_statistics_exported_3n,
            "required": 0,
        },
        {
            "check": "assets_crossing_splits",
            "observed": assets_crossing_splits_3n,
            "required": 0,
        },
        {
            "check": "events_crossing_splits",
            "observed": events_crossing_splits_3n,
            "required": 0,
        },
        {
            "check": "manifest_rows_deleted",
            "observed": manifest_length_before_3n - len(ROW_LABEL_MANIFEST),
            "required": 0,
        },
        {
            "check": "manifest_values_changed",
            "observed": manifest_values_changed_3n,
            "required": 0,
        },
    ]
)
PREPROCESSING_LEAKAGE_AUDIT_3N["passed"] = (
    PREPROCESSING_LEAKAGE_AUDIT_3N["observed"]
    == PREPROCESSING_LEAKAGE_AUDIT_3N["required"]
)

if not PREPROCESSING_LEAKAGE_AUDIT_3N["passed"].all():
    failed_checks_3n = PREPROCESSING_LEAKAGE_AUDIT_3N.loc[
        ~PREPROCESSING_LEAKAGE_AUDIT_3N["passed"]
    ]
    raise ValueError(
        "Cell 3N leakage/conservation checks failed:\n"
        + failed_checks_3n.to_string(index=False)
    )


# -----------------------------------------------------------------------------
# 7. Export paper-ready tables, figures, registries, and frozen state
# -----------------------------------------------------------------------------

OUTPUT_ROOT_3N = Path(
    globals().get("OUTPUT_ROOT_3L", Path("paper_visuals_3l"))
) / "preprocessing_3n"
TABLE_DIR_3N = OUTPUT_ROOT_3N / "tables"
FIGURE_DIR_3N = OUTPUT_ROOT_3N / "figures"
for directory_3n in [OUTPUT_ROOT_3N, TABLE_DIR_3N, FIGURE_DIR_3N]:
    directory_3n.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SPLIT_COLORS_3N = {"train": "#2E6F9E", "validation": "#E69F00"}
DECISION_COLORS_3N = {"retain": "#2A9D8F", "exclude": "#C94C4C"}
figure_registry_records_3n = []


def save_figure_3n(figure, figure_id, title, claim_supported, source_objects):
    stem_3n = f"{figure_id}_{slug_3n(title)}"
    png_path_3n = FIGURE_DIR_3N / f"{stem_3n}.png"
    pdf_path_3n = FIGURE_DIR_3N / f"{stem_3n}.pdf"
    figure.savefig(png_path_3n, bbox_inches="tight", dpi=300)
    figure.savefig(pdf_path_3n, bbox_inches="tight")
    plt.close(figure)
    figure_registry_records_3n.append(
        {
            "figure_id": figure_id,
            "title": title,
            "claim_supported": claim_supported,
            "source_objects": source_objects,
            "fit_or_comparison_scope": "training and validation only; test sealed",
            "png_path": str(png_path_3n),
            "pdf_path": str(pdf_path_3n),
        }
    )


# F3N01 — feature decision counts.
decision_counts_3n = (
    FEATURE_QUALITY_AUDIT_3N["decision"]
    .value_counts()
    .reindex(["retain", "exclude"], fill_value=0)
)
figure_3n, axis_3n = plt.subplots(figsize=(5.8, 4.0))
bars_3n = axis_3n.bar(
    decision_counts_3n.index,
    decision_counts_3n.values,
    color=[DECISION_COLORS_3N[value_3n] for value_3n in decision_counts_3n.index],
)
axis_3n.bar_label(bars_3n, padding=3)
axis_3n.set_ylabel("Candidate sensor features")
axis_3n.set_title("Training-only feature-quality decisions")
axis_3n.grid(axis="y", alpha=0.2)
save_figure_3n(
    figure_3n,
    "F3N01",
    "Training-only feature quality decisions",
    "Documents how many candidate sensors passed predeclared training-only gates.",
    "FEATURE_QUALITY_AUDIT_3N",
)


# F3N02 — training missingness.
missing_plot_3n = FEATURE_QUALITY_AUDIT_3N.sort_values(
    "train_missing_fraction", ascending=True
).copy()
figure_height_3n = max(4.5, min(16.0, 0.28 * len(missing_plot_3n) + 1.8))
figure_3n, axis_3n = plt.subplots(figsize=(8.2, figure_height_3n))
axis_3n.barh(
    np.arange(len(missing_plot_3n)),
    100.0 * missing_plot_3n["train_missing_fraction"],
    color=[DECISION_COLORS_3N[value_3n] for value_3n in missing_plot_3n["decision"]],
)
axis_3n.axvline(
    100.0 * MAX_TRAIN_MISSING_FRACTION_3N,
    color="#333333",
    linestyle="--",
    linewidth=1.0,
    label="Global exclusion threshold",
)
axis_3n.set_yticks(np.arange(len(missing_plot_3n)))
axis_3n.set_yticklabels(missing_plot_3n["feature"])
axis_3n.set_xlabel("Missing training values (%)")
axis_3n.set_title("Training missingness by candidate sensor")
axis_3n.legend(loc="lower right")
axis_3n.grid(axis="x", alpha=0.2)
save_figure_3n(
    figure_3n,
    "F3N02",
    "Training missingness by candidate sensor",
    "Shows the missingness evidence used by the train-only feature gate.",
    "FEATURE_QUALITY_AUDIT_3N",
)


# F3N03 — farm-specific training missingness.
farm_missing_columns_3n = [
    "train_farm_A_missing_fraction",
    "train_farm_B_missing_fraction",
    "train_farm_C_missing_fraction",
]
farm_missing_matrix_3n = FEATURE_QUALITY_AUDIT_3N.set_index("feature").loc[
    :, farm_missing_columns_3n
]
figure_height_3n = max(4.5, min(16.0, 0.25 * len(farm_missing_matrix_3n) + 2.0))
figure_3n, axis_3n = plt.subplots(figsize=(6.8, figure_height_3n))
image_3n = axis_3n.imshow(
    100.0 * farm_missing_matrix_3n.to_numpy(dtype=float),
    aspect="auto",
    cmap="YlOrRd",
    vmin=0,
    vmax=max(
        1.0,
        min(100.0, 100.0 * float(farm_missing_matrix_3n.max().max())),
    ),
)
axis_3n.set_xticks([0, 1, 2])
axis_3n.set_xticklabels(["Farm A", "Farm B", "Farm C"])
axis_3n.set_yticks(np.arange(len(farm_missing_matrix_3n)))
axis_3n.set_yticklabels(farm_missing_matrix_3n.index)
axis_3n.set_title("Farm-specific training missingness")
colorbar_3n = figure_3n.colorbar(image_3n, ax=axis_3n, pad=0.02)
colorbar_3n.set_label("Missing values (%)")
save_figure_3n(
    figure_3n,
    "F3N03",
    "Farm-specific training missingness",
    "Verifies that retained features have usable training support in every farm.",
    "FEATURE_QUALITY_AUDIT_3N",
)


# F3N04 — validation PSI, never test PSI.
psi_plot_3n = VALIDATION_SHIFT_AUDIT_3N.sort_values(
    "population_stability_index", ascending=True
)
figure_height_3n = max(4.5, min(16.0, 0.28 * len(psi_plot_3n) + 1.8))
figure_3n, axis_3n = plt.subplots(figsize=(8.2, figure_height_3n))
axis_3n.barh(
    np.arange(len(psi_plot_3n)),
    psi_plot_3n["population_stability_index"],
    color="#6C5B7B",
)
axis_3n.axvline(0.10, color="#E69F00", linestyle="--", label="Moderate shift")
axis_3n.axvline(0.25, color="#C94C4C", linestyle="--", label="Large shift")
axis_3n.set_yticks(np.arange(len(psi_plot_3n)))
axis_3n.set_yticklabels(psi_plot_3n["feature"])
axis_3n.set_xlabel("Population stability index")
axis_3n.set_title("Validation-to-training feature shift")
axis_3n.legend(loc="lower right")
axis_3n.grid(axis="x", alpha=0.2)
save_figure_3n(
    figure_3n,
    "F3N04",
    "Validation to training feature shift",
    "Quantifies unseen-asset validation shift without consulting test distributions.",
    "VALIDATION_SHIFT_AUDIT_3N",
)


# F3N05 — training-only Spearman correlations.
max_heatmap_features_3n = min(30, len(FEATURE_NAMES_3N))
heatmap_features_3n = (
    TRAIN_TRANSFORMED_FEATURE_SUMMARY_3N.sort_values("std", ascending=False)
    .head(max_heatmap_features_3n)["feature"]
    .tolist()
)
heatmap_matrix_3n = TRAIN_SPEARMAN_CORRELATION_3N.loc[
    heatmap_features_3n, heatmap_features_3n
]
heatmap_size_3n = max(6.0, min(13.0, 0.42 * len(heatmap_features_3n) + 2.5))
figure_3n, axis_3n = plt.subplots(figsize=(heatmap_size_3n, heatmap_size_3n))
image_3n = axis_3n.imshow(
    heatmap_matrix_3n.to_numpy(dtype=float),
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
axis_3n.set_xticks(np.arange(len(heatmap_features_3n)))
axis_3n.set_xticklabels(heatmap_features_3n, rotation=90)
axis_3n.set_yticks(np.arange(len(heatmap_features_3n)))
axis_3n.set_yticklabels(heatmap_features_3n)
axis_3n.set_title("Training-only Spearman feature correlation")
colorbar_3n = figure_3n.colorbar(image_3n, ax=axis_3n, pad=0.02)
colorbar_3n.set_label("Spearman correlation")
save_figure_3n(
    figure_3n,
    "F3N05",
    "Training only Spearman feature correlation",
    "Documents multicollinearity without automatically deleting correlated sensors.",
    "TRAIN_SPEARMAN_CORRELATION_3N",
)


# F3N06 — train and validation transformed location/scale; test remains sealed.
comparison_summary_3n = pd.DataFrame(
    {
        "feature": FEATURE_NAMES_3N,
        "train_mean": X_TRAIN_3N.mean().reindex(FEATURE_NAMES_3N).to_numpy(),
        "validation_mean": X_VALIDATION_3N.mean().reindex(FEATURE_NAMES_3N).to_numpy(),
        "train_std": X_TRAIN_3N.std(ddof=0).reindex(FEATURE_NAMES_3N).to_numpy(),
        "validation_std": X_VALIDATION_3N.std(ddof=0).reindex(FEATURE_NAMES_3N).to_numpy(),
    }
)
plot_order_3n = comparison_summary_3n.assign(
    validation_mean_abs=lambda frame_3n: frame_3n["validation_mean"].abs()
).sort_values("validation_mean_abs", ascending=True)
figure_height_3n = max(5.0, min(16.0, 0.30 * len(plot_order_3n) + 2.2))
figure_3n, axes_3n = plt.subplots(
    1, 2, figsize=(11.0, figure_height_3n), sharey=True
)
y_positions_3n = np.arange(len(plot_order_3n))
for axis_3n, statistic_3n, title_3n in [
    (axes_3n[0], "mean", "Transformed mean"),
    (axes_3n[1], "std", "Transformed standard deviation"),
]:
    axis_3n.scatter(
        plot_order_3n[f"train_{statistic_3n}"],
        y_positions_3n,
        color=SPLIT_COLORS_3N["train"],
        marker="o",
        label="Train",
        s=25,
    )
    axis_3n.scatter(
        plot_order_3n[f"validation_{statistic_3n}"],
        y_positions_3n,
        color=SPLIT_COLORS_3N["validation"],
        marker="^",
        label="Validation",
        s=28,
    )
    axis_3n.set_xlabel(title_3n)
    axis_3n.grid(axis="x", alpha=0.2)
axes_3n[0].set_yticks(y_positions_3n)
axes_3n[0].set_yticklabels(plot_order_3n["feature"])
axes_3n[0].axvline(0, color="#555555", linewidth=0.8)
axes_3n[1].axvline(1, color="#555555", linewidth=0.8)
axes_3n[1].legend(loc="lower right")
figure_3n.suptitle("Frozen preprocessing across train and validation")
save_figure_3n(
    figure_3n,
    "F3N06",
    "Frozen preprocessing across train and validation",
    "Shows validation behavior under training-fitted preprocessing; test remains sealed.",
    "X_TRAIN_3N; X_VALIDATION_3N",
)


PREPROCESSING_FIGURE_REGISTRY_3N = pd.DataFrame(figure_registry_records_3n)

table_exports_3n = [
    (
        "T3N01",
        "source_resolution_audit",
        SOURCE_RESOLUTION_AUDIT_3N,
        "Documents exact source-key resolution before feature processing.",
    ),
    (
        "T3N02",
        "feature_quality_audit",
        FEATURE_QUALITY_AUDIT_3N,
        "Reports training-only feature decisions and fitted statistics.",
    ),
    (
        "T3N03",
        "preprocessing_parameters",
        PREPROCESSING_PARAMETERS_3N,
        "Provides the frozen median-imputation and robust-scaling parameters.",
    ),
    (
        "T3N04",
        "split_transform_audit",
        SPLIT_TRANSFORM_AUDIT_3N,
        "Verifies finite, row-conserving transformation while sealing test statistics.",
    ),
    (
        "T3N05",
        "validation_shift_audit",
        VALIDATION_SHIFT_AUDIT_3N,
        "Quantifies validation-to-training drift using training-defined bins.",
    ),
    (
        "T3N06",
        "training_correlation_pairs",
        TRAIN_CORRELATION_PAIR_AUDIT_3N,
        "Reports training-only correlated sensor pairs without automatic removal.",
    ),
    (
        "T3N07",
        "training_transformed_feature_summary",
        TRAIN_TRANSFORMED_FEATURE_SUMMARY_3N,
        "Summarizes the transformed training feature space.",
    ),
    (
        "T3N08",
        "preprocessing_leakage_audit",
        PREPROCESSING_LEAKAGE_AUDIT_3N,
        "Confirms train-only fitting, sealed test diagnostics, and manifest conservation.",
    ),
]

table_registry_records_3n = []
for table_id_3n, table_name_3n, table_frame_3n, purpose_3n in table_exports_3n:
    stem_3n = f"{table_id_3n}_{slug_3n(table_name_3n)}"
    csv_path_3n = TABLE_DIR_3N / f"{stem_3n}.csv"
    tex_path_3n = TABLE_DIR_3N / f"{stem_3n}.tex"
    table_frame_3n.to_csv(csv_path_3n, index=False)
    latex_status_3n = "exported"
    try:
        table_frame_3n.to_latex(tex_path_3n, index=False, escape=True)
        latex_path_value_3n = str(tex_path_3n)
    except Exception as error_3n:
        latex_status_3n = f"skipped: {type(error_3n).__name__}: {error_3n}"
        latex_path_value_3n = pd.NA
        print(f"LaTeX export skipped for {table_id_3n}: {error_3n}")

    table_registry_records_3n.append(
        {
            "table_id": table_id_3n,
            "name": table_name_3n,
            "purpose": purpose_3n,
            "rows": len(table_frame_3n),
            "columns": len(table_frame_3n.columns),
            "csv_path": str(csv_path_3n),
            "latex_path": latex_path_value_3n,
            "latex_status": latex_status_3n,
        }
    )

PREPROCESSING_TABLE_REGISTRY_3N = pd.DataFrame(table_registry_records_3n)

figure_registry_path_3n = OUTPUT_ROOT_3N / "preprocessing_figure_registry_3n.csv"
table_registry_path_3n = OUTPUT_ROOT_3N / "preprocessing_table_registry_3n.csv"
PREPROCESSING_FIGURE_REGISTRY_3N.to_csv(figure_registry_path_3n, index=False)
PREPROCESSING_TABLE_REGISTRY_3N.to_csv(table_registry_path_3n, index=False)

state_json_path_3n = OUTPUT_ROOT_3N / "preprocessor_state_3n.json"


def json_safe_3n(value):
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


with state_json_path_3n.open("w", encoding="utf-8") as state_file_3n:
    json.dump(
        PREPROCESSOR_STATE_3N,
        state_file_3n,
        indent=2,
        ensure_ascii=False,
        default=json_safe_3n,
    )

workbook_path_3n = OUTPUT_ROOT_3N / "preprocessing_diagnostics_3n.xlsx"
workbook_exported_3n = False
try:
    with pd.ExcelWriter(workbook_path_3n, engine="openpyxl") as writer_3n:
        for table_id_3n, table_name_3n, table_frame_3n, _ in table_exports_3n:
            sheet_name_3n = f"{table_id_3n}_{slug_3n(table_name_3n)}"[:31]
            workbook_frame_3n = table_frame_3n.copy()
            for column_3n in workbook_frame_3n.columns:
                if isinstance(workbook_frame_3n[column_3n].dtype, pd.DatetimeTZDtype):
                    workbook_frame_3n[column_3n] = workbook_frame_3n[
                        column_3n
                    ].dt.tz_convert("UTC").dt.tz_localize(None)
            workbook_frame_3n.to_excel(
                writer_3n,
                sheet_name=sheet_name_3n,
                index=False,
            )
        PREPROCESSING_FIGURE_REGISTRY_3N.to_excel(
            writer_3n, sheet_name="figure_registry", index=False
        )
        PREPROCESSING_TABLE_REGISTRY_3N.to_excel(
            writer_3n, sheet_name="table_registry", index=False
        )

        for worksheet_3n in writer_3n.book.worksheets:
            worksheet_3n.freeze_panes = "A2"
            worksheet_3n.auto_filter.ref = worksheet_3n.dimensions
            for cell_3n in worksheet_3n[1]:
                cell_3n.font = cell_3n.font.copy(bold=True, color="FFFFFF")
                cell_3n.fill = cell_3n.fill.copy(
                    fill_type="solid", fgColor="1F4E78"
                )
            for column_cells_3n in worksheet_3n.columns:
                maximum_length_3n = max(
                    len(str(cell_3n.value)) if cell_3n.value is not None else 0
                    for cell_3n in column_cells_3n
                )
                column_letter_3n = column_cells_3n[0].column_letter
                worksheet_3n.column_dimensions[column_letter_3n].width = min(
                    max(maximum_length_3n + 2, 10), 45
                )
    workbook_exported_3n = True
except Exception as error_3n:
    print(f"Excel workbook export skipped: {error_3n}")


# -----------------------------------------------------------------------------
# 8. Compact notebook report
# -----------------------------------------------------------------------------

decision_reason_summary_3n = (
    FEATURE_QUALITY_AUDIT_3N.groupby(
        ["decision", "decision_reason"], as_index=False
    )
    .agg(features=("feature", "size"))
    .sort_values(["decision", "features"], ascending=[False, False])
)

print("\nMeasurement-source resolution:")
print(SOURCE_RESOLUTION_AUDIT_3N.to_string(index=False))

print("\nTraining-only feature decisions:")
print(decision_reason_summary_3n.to_string(index=False))

print("\nFrozen split-transform audit:")
print(SPLIT_TRANSFORM_AUDIT_3N.to_string(index=False))

print("\nValidation-shift summary (test remains sealed):")
print(
    VALIDATION_SHIFT_AUDIT_3N.head(min(20, len(VALIDATION_SHIFT_AUDIT_3N)))
    .to_string(index=False)
)

print("\nPreprocessing leakage and conservation audit:")
print(PREPROCESSING_LEAKAGE_AUDIT_3N.to_string(index=False))

print("\nCell 3N completed successfully.")
print("Measurement source:", selected_source_name_3n)
print("Eligible rows resolved:", len(PREPROCESSED_FEATURE_MATRIX_3N))
print("Candidate sensor features:", len(candidate_features_3n))
print("Retained sensor features:", len(FEATURE_NAMES_3N))
print("Excluded sensor features:", len(EXCLUDED_FEATURES_3N))
print(
    "Train/validation/test rows:",
    f"{len(X_TRAIN_3N)}/{len(X_VALIDATION_3N)}/{len(X_TEST_3N)}",
)
print("Post-transform missing cells:", int(transformed_features_3n.isna().sum().sum()))
print(
    "Post-transform non-finite cells:",
    int((~np.isfinite(transformed_features_3n.to_numpy())).sum()),
)
print("Tables generated:", len(PREPROCESSING_TABLE_REGISTRY_3N))
print("Figures generated:", len(PREPROCESSING_FIGURE_REGISTRY_3N))
print("Figure registry:", figure_registry_path_3n)
print("Table registry:", table_registry_path_3n)
print("Frozen preprocessor state:", state_json_path_3n)
print("Excel workbook exported:", workbook_exported_3n)
print("Manifest values changed:", manifest_values_changed_3n)
print(
    "\nAll feature inclusion, imputation, and scaling parameters were fitted "
    "on training assets only. Validation was used only for shift diagnostics. "
    "Test received the frozen transform but its distributions remain sealed. "
    "No manifest row, label, eligibility decision, or split assignment changed."
)
