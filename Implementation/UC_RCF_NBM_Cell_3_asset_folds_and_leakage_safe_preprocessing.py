"""CELL 3 — asset folds and leakage-safe CARE preprocessing.

Paste this complete file into the third cell of the UC-RCF-NBM notebook and run
it only after Cells 1 and 2 have completed successfully.

This cell creates the 36 leave-one-asset-out outer folds, deterministic
farm-balanced inner asset folds, and label-free case caches. All fitted
preprocessing quantities are estimated from normal-status rows in each case's
source training partition. No event label, fault description, or event boundary
is loaded. Sensor eligibility is recorded per case; it is not selected globally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# =============================================================================
# 0. Bind this cell to Umar's completed Cell 1 and Cell 2 receipts
# =============================================================================

EXPECTED_CONTRACT_SHA256 = (
    "827641aecd8e807193ad193d64c274319756092faa53bdb5084f310f62041f49"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "62484bab1219888aa1d0788965ecd77db2b85f0bbb9b476cd3240f4143026f1f"
)
EXPECTED_OUTCOME_LOCKBOX_SHA256 = (
    "69c7c75ea8157e2e12cb61c596375168a90c73bef7ee283c618dab080a447e10"
)
EXPECTED_CELL2_AUDIT_SHA256 = (
    "83732caf4ad3e226b69a671287bb14c4ed72e1bfc3b7c26ca144310fce8e5990"
)

_required_objects = (
    "CONTRACT_SHA256",
    "DATASET_MANIFEST_SHA256",
    "OUTCOME_LOCKBOX_SHA256",
    "CELL2_AUDIT_SHA256",
    "DATASET",
    "QUALITY",
    "EVALUATION",
    "REPRODUCIBILITY",
    "CASE_REGISTRY",
    "CARE_FEATURE_REGISTRY",
    "FARM_SCHEMAS",
    "read_care_chunks",
    "normalized_partition",
    "save_csv_atomic",
    "save_json",
    "sha256_json",
    "utc_now",
    "CACHE_DIR",
    "INVENTORY_DIR",
    "QUALITY_DIR",
)
_missing_objects = [name for name in _required_objects if name not in globals()]
if _missing_objects:
    raise RuntimeError(
        "Run UC-RCF-NBM Cells 1 and 2 before Cell 3. Missing objects: "
        + ", ".join(_missing_objects)
    )

_observed_hashes = {
    "contract": CONTRACT_SHA256,
    "dataset_manifest": DATASET_MANIFEST_SHA256,
    "outcome_lockbox": OUTCOME_LOCKBOX_SHA256,
    "cell2_audit": CELL2_AUDIT_SHA256,
}
_expected_hashes = {
    "contract": EXPECTED_CONTRACT_SHA256,
    "dataset_manifest": EXPECTED_DATASET_MANIFEST_SHA256,
    "outcome_lockbox": EXPECTED_OUTCOME_LOCKBOX_SHA256,
    "cell2_audit": EXPECTED_CELL2_AUDIT_SHA256,
}
if _observed_hashes != _expected_hashes:
    raise RuntimeError(
        "Cell 3 is bound to the completed Cell 1/Cell 2 audit reported in this "
        f"experiment. Observed={_observed_hashes}, expected={_expected_hashes}."
    )

_safe_registry_expected = {
    "case_key",
    "farm",
    "asset_id",
    "source_asset_id",
    "event_id",
    "file_path",
    "relative_file_path",
    "size_bytes",
}
if set(CASE_REGISTRY.columns) != _safe_registry_expected:
    raise RuntimeError(
        "CASE_REGISTRY has changed or includes unsafe fields. "
        f"Observed={sorted(CASE_REGISTRY.columns)}"
    )

_forbidden_tokens = (
    "event_label",
    "is_anomaly",
    "event_start",
    "event_end",
    "event_description",
    "fault_type",
    "care_ground_truth",
)
if any(
    any(token in str(column).lower() for token in _forbidden_tokens)
    for column in CASE_REGISTRY.columns
):
    raise RuntimeError("An outcome field is present in CASE_REGISTRY.")

if EVALUATION.outer_strategy != "leave-one-asset-out":
    raise RuntimeError("Cell 3 requires the frozen leave-one-asset-out strategy.")
if EVALUATION.inner_group_column != "asset_id":
    raise RuntimeError("Inner resampling must remain grouped by asset_id.")

CELL3_VERSION = "1.0.0"
CELL3_CACHE_ROOT = Path(CACHE_DIR) / "cell3_case_preprocessing"
CELL3_FOLD_DIR = Path(INVENTORY_DIR) / "cell3_folds"
CELL3_QUALITY_DIR = Path(QUALITY_DIR) / "cell3_preprocessing"
for _directory in (CELL3_CACHE_ROOT, CELL3_FOLD_DIR, CELL3_QUALITY_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Frozen preprocessing policy derived from Cell 2 diagnostics
# =============================================================================

PREPROCESSING_POLICY = {
    "version": CELL3_VERSION,
    "input_statistics": tuple(QUALITY.primary_statistics),
    "source_fit_rows": "train partition AND normal status, independently per case",
    "time_order": "stable chronological order; no sorting quantity uses outcomes",
    "continuity": (
        f"new segment when delta is not exactly {DATASET.sampling_minutes} minutes; "
        "all temporal transformations restart"
    ),
    "numeric_missing": "non-finite values remain missing",
    "zero_handling": (
        "never convert sustained zero runs globally; convert zero to missing only "
        "when that case-channel is structurally zero across all finite fit rows"
    ),
    "availability": (
        f"case-channel usable only when fit availability >= "
        f"{QUALITY.minimum_sensor_availability:.2f}"
    ),
    "driver_imputation": (
        "case-local training-normal median; missingness indicators retained; "
        "imputation parameters frozen before prediction rows"
    ),
    "target_missing": "not imputed; masked from fitting, uncertainty calibration, and scoring",
    "angles": "degrees mapped to sine and cosine; reset-neutral and bounded",
    "counters": (
        "within-segment first difference; first row, negative resets, and "
        "missing-adjacent differences are missing"
    ),
    "global_sensor_selection": False,
    "event_outcomes_read": False,
    "outer_prediction_rows_used_to_fit_preprocessing": False,
    "cache_dtype": "float32 values, boolean masks, int32 segment identifiers",
}
PREPROCESSING_POLICY_SHA256 = sha256_json(PREPROCESSING_POLICY)


# =============================================================================
# 2. Deterministic leave-one-asset-out and farm-balanced inner folds
# =============================================================================

def stable_hash_int(text: str, seed: int) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


ASSET_REGISTRY = (
    CASE_REGISTRY.groupby(["farm", "asset_id", "source_asset_id"], as_index=False)
    .agg(cases=("case_key", "nunique"))
    .sort_values(["farm", "asset_id"], kind="stable")
    .reset_index(drop=True)
)

if len(ASSET_REGISTRY) != DATASET.expected_assets:
    raise RuntimeError(
        f"Found {len(ASSET_REGISTRY)} farm-qualified assets; "
        f"expected {DATASET.expected_assets}."
    )
if ASSET_REGISTRY.groupby("asset_id")["farm"].nunique().max() != 1:
    raise RuntimeError("A canonical asset_id appears in more than one farm.")

_outer_records: list[dict[str, Any]] = []
_inner_records: list[dict[str, Any]] = []

for _outer_index, _outer_asset in enumerate(ASSET_REGISTRY["asset_id"], start=1):
    _outer_farm = str(
        ASSET_REGISTRY.loc[ASSET_REGISTRY["asset_id"].eq(_outer_asset), "farm"].iloc[0]
    )
    _outer_cases = CASE_REGISTRY.loc[CASE_REGISTRY["asset_id"].eq(_outer_asset)]
    _development_assets = ASSET_REGISTRY.loc[
        ~ASSET_REGISTRY["asset_id"].eq(_outer_asset)
    ].copy()

    _outer_records.append(
        {
            "outer_fold": _outer_index,
            "outer_asset_id": _outer_asset,
            "outer_farm": _outer_farm,
            "outer_cases": int(len(_outer_cases)),
            "development_assets": int(len(_development_assets)),
            "development_cases": int(
                CASE_REGISTRY["asset_id"].isin(_development_assets["asset_id"]).sum()
            ),
        }
    )

    # Balance each farm's assets across the global five folds. The start offset
    # changes by outer fold, preventing one numbered fold from repeatedly holding
    # the same farm mix while keeping the assignment deterministic.
    _inner_assignment: dict[str, int] = {}
    for _farm_index, _farm in enumerate(DATASET.farms):
        _farm_assets = _development_assets.loc[
            _development_assets["farm"].eq(_farm), "asset_id"
        ].tolist()
        _farm_assets.sort(
            key=lambda asset: (stable_hash_int(asset, REPRODUCIBILITY.seed + _outer_index), asset)
        )
        _offset = (_outer_index + _farm_index) % EVALUATION.inner_splits
        for _position, _asset in enumerate(_farm_assets):
            _inner_assignment[_asset] = (
                (_position + _offset) % EVALUATION.inner_splits
            ) + 1

    if set(_inner_assignment) != set(_development_assets["asset_id"]):
        raise RuntimeError(f"Incomplete inner assignment in outer fold {_outer_index}.")

    _used_folds = set(_inner_assignment.values())
    if _used_folds != set(range(1, EVALUATION.inner_splits + 1)):
        raise RuntimeError(
            f"Outer fold {_outer_index} does not use all inner fold numbers: {_used_folds}"
        )

    for _asset, _inner_fold in sorted(_inner_assignment.items()):
        _asset_row = _development_assets.loc[
            _development_assets["asset_id"].eq(_asset)
        ].iloc[0]
        _inner_records.append(
            {
                "outer_fold": _outer_index,
                "outer_asset_id": _outer_asset,
                "inner_fold": _inner_fold,
                "asset_id": _asset,
                "farm": _asset_row["farm"],
                "cases": int(_asset_row["cases"]),
            }
        )

OUTER_FOLDS = pd.DataFrame(_outer_records)
INNER_ASSET_FOLDS = pd.DataFrame(_inner_records).sort_values(
    ["outer_fold", "inner_fold", "farm", "asset_id"], kind="stable"
).reset_index(drop=True)

if OUTER_FOLDS["outer_asset_id"].nunique() != DATASET.expected_assets:
    raise RuntimeError("Each asset must occur exactly once as the outer test asset.")
if len(INNER_ASSET_FOLDS) != DATASET.expected_assets * (DATASET.expected_assets - 1):
    raise RuntimeError("Unexpected number of outer-development asset assignments.")

for _outer_fold in OUTER_FOLDS["outer_fold"]:
    _outer_asset = OUTER_FOLDS.loc[
        OUTER_FOLDS["outer_fold"].eq(_outer_fold), "outer_asset_id"
    ].iloc[0]
    _assignments = INNER_ASSET_FOLDS.loc[
        INNER_ASSET_FOLDS["outer_fold"].eq(_outer_fold)
    ]
    if _outer_asset in set(_assignments["asset_id"]):
        raise RuntimeError(f"Outer asset leaked into inner folds for outer fold {_outer_fold}.")
    if _assignments["asset_id"].duplicated().any():
        raise RuntimeError(f"An inner asset is duplicated for outer fold {_outer_fold}.")
    if _assignments["asset_id"].nunique() != DATASET.expected_assets - 1:
        raise RuntimeError(f"Outer fold {_outer_fold} has the wrong development asset count.")


def iter_outer_folds() -> Iterator[dict[str, Any]]:
    """Yield safe case registries for one leave-one-asset-out fold at a time."""
    for row in OUTER_FOLDS.itertuples(index=False):
        test_mask = CASE_REGISTRY["asset_id"].eq(row.outer_asset_id)
        development = CASE_REGISTRY.loc[~test_mask].copy()
        test = CASE_REGISTRY.loc[test_mask].copy()
        if set(development["asset_id"]) & set(test["asset_id"]):
            raise RuntimeError(f"Asset leakage in outer fold {row.outer_fold}.")
        yield {
            "outer_fold": int(row.outer_fold),
            "outer_asset_id": row.outer_asset_id,
            "development_cases": development,
            "test_cases": test,
            "inner_asset_folds": INNER_ASSET_FOLDS.loc[
                INNER_ASSET_FOLDS["outer_fold"].eq(row.outer_fold)
            ].copy(),
        }


def iter_inner_folds(outer_fold: int) -> Iterator[dict[str, Any]]:
    """Yield development/validation asset partitions inside one outer fold."""
    assignments = INNER_ASSET_FOLDS.loc[
        INNER_ASSET_FOLDS["outer_fold"].eq(outer_fold)
    ]
    if assignments.empty:
        raise KeyError(f"Unknown outer fold: {outer_fold}")
    outer_asset = OUTER_FOLDS.loc[
        OUTER_FOLDS["outer_fold"].eq(outer_fold), "outer_asset_id"
    ].iloc[0]
    outer_development = CASE_REGISTRY.loc[
        ~CASE_REGISTRY["asset_id"].eq(outer_asset)
    ]
    for inner_fold in range(1, EVALUATION.inner_splits + 1):
        validation_assets = set(
            assignments.loc[assignments["inner_fold"].eq(inner_fold), "asset_id"]
        )
        validation = outer_development.loc[
            outer_development["asset_id"].isin(validation_assets)
        ].copy()
        development = outer_development.loc[
            ~outer_development["asset_id"].isin(validation_assets)
        ].copy()
        if validation.empty or development.empty:
            raise RuntimeError(
                f"Empty inner partition: outer={outer_fold}, inner={inner_fold}."
            )
        if set(validation["asset_id"]) & set(development["asset_id"]):
            raise RuntimeError(
                f"Asset leakage: outer={outer_fold}, inner={inner_fold}."
            )
        yield {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "development_cases": development,
            "validation_cases": validation,
        }


# =============================================================================
# 3. Case-local, training-normal preprocessing helpers
# =============================================================================

def _strict_segment_ids(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if len(timestamps) == 0:
        return np.empty(0, dtype=np.int32)
    values = timestamps.astype(np.int64)
    nat_ns = np.iinfo(np.int64).min
    expected_ns = int(DATASET.sampling_minutes * 60 * 1_000_000_000)
    new_segment = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        delta = values[1:] - values[:-1]
        new_segment[1:] = (
            (values[1:] == nat_ns)
            | (values[:-1] == nat_ns)
            | (delta != expected_ns)
        )
    return (np.cumsum(new_segment, dtype=np.int64) - 1).astype(np.int32)


def _counter_differences(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) <= 1 or values.shape[1] == 0:
        return output
    same_segment = segment_ids[1:] == segment_ids[:-1]
    differences = values[1:] - values[:-1]
    valid = (
        same_segment[:, None]
        & np.isfinite(values[1:])
        & np.isfinite(values[:-1])
        & np.isfinite(differences)
        & (differences >= 0.0)
    )
    output[1:] = np.where(valid, differences, np.nan)
    return output


def _safe_slug(text: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part)


def _atomic_save_npz(output_path: Path, **arrays: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output_path)


def _json_file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _first_pass_fit_statistics(case_row: Any, channels: list[str]) -> dict[str, Any]:
    count = np.zeros(len(channels), dtype=np.int64)
    zero_count = np.zeros(len(channels), dtype=np.int64)
    mean = np.zeros(len(channels), dtype=np.float64)
    m2 = np.zeros(len(channels), dtype=np.float64)
    total_rows = fit_rows = train_rows = prediction_rows = unknown_rows = 0
    asset_values: set[str] = set()

    required = [*DATASET.metadata_columns, *channels]
    for chunk in read_care_chunks(Path(case_row.file_path), required):
        total_rows += len(chunk)
        partition = normalized_partition(chunk["train_test"])
        is_train = partition.eq("train").to_numpy()
        is_prediction = partition.eq("prediction").to_numpy()
        unknown_rows += int(partition.eq("unknown").sum())
        train_rows += int(is_train.sum())
        prediction_rows += int(is_prediction.sum())
        status = pd.to_numeric(chunk["status_type_id"], errors="coerce")
        fit_mask = is_train & status.isin(DATASET.normal_status_ids).to_numpy()
        fit_rows += int(fit_mask.sum())
        asset_values.update(str(value).strip() for value in chunk["asset_id"].dropna().unique())

        values = chunk[channels].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        fit_values = values[fit_mask]
        finite = np.isfinite(fit_values)
        zero_count += (finite & (fit_values == 0.0)).sum(axis=0, dtype=np.int64)

        for column_index in range(len(channels)):
            observed = fit_values[finite[:, column_index], column_index]
            n_batch = len(observed)
            if n_batch == 0:
                continue
            batch_mean = float(observed.mean())
            batch_m2 = float(np.square(observed - batch_mean).sum())
            n_old = int(count[column_index])
            n_new = n_old + n_batch
            delta = batch_mean - mean[column_index]
            mean[column_index] += delta * n_batch / n_new
            m2[column_index] += batch_m2 + delta * delta * n_old * n_batch / n_new
            count[column_index] = n_new

    availability = count / max(fit_rows, 1)
    variance = np.divide(
        m2,
        np.maximum(count - 1, 1),
        out=np.zeros_like(m2),
        where=count > 1,
    )
    structurally_zero = (count > 0) & (zero_count == count)
    usable = (
        (availability >= QUALITY.minimum_sensor_availability)
        & (count > 1)
        & np.isfinite(variance)
        & (variance > 0.0)
        & ~structurally_zero
    )
    if fit_rows < QUALITY.minimum_training_rows:
        raise RuntimeError(
            f"{case_row.case_key}: {fit_rows} training-normal rows; "
            f"minimum is {QUALITY.minimum_training_rows}."
        )
    if unknown_rows:
        raise RuntimeError(f"{case_row.case_key}: {unknown_rows} unknown partition rows.")

    return {
        "total_rows": total_rows,
        "train_rows": train_rows,
        "prediction_rows": prediction_rows,
        "fit_rows": fit_rows,
        "asset_values": sorted(asset_values),
        "finite_count": count,
        "zero_count": zero_count,
        "mean": mean,
        "variance": variance,
        "availability": availability,
        "structurally_zero": structurally_zero,
        "usable": usable,
    }


def _second_pass_arrays(
    case_row: Any,
    channels: list[str],
    structurally_zero: np.ndarray,
) -> dict[str, np.ndarray]:
    timestamps_parts: list[np.ndarray] = []
    partition_parts: list[np.ndarray] = []
    normal_status_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []

    required = [*DATASET.metadata_columns, *channels]
    for chunk in read_care_chunks(Path(case_row.file_path), required):
        timestamps_parts.append(chunk["time_stamp"].to_numpy(dtype="datetime64[ns]"))
        partition = normalized_partition(chunk["train_test"])
        partition_code = np.full(len(chunk), -1, dtype=np.int8)
        partition_code[partition.eq("train").to_numpy()] = 0
        partition_code[partition.eq("prediction").to_numpy()] = 1
        partition_parts.append(partition_code)
        status = pd.to_numeric(chunk["status_type_id"], errors="coerce")
        normal_status_parts.append(
            status.isin(DATASET.normal_status_ids).to_numpy(dtype=bool)
        )
        values = chunk[channels].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        values[~np.isfinite(values)] = np.nan
        if structurally_zero.any():
            values[:, structurally_zero] = np.nan
        values_parts.append(values)

    timestamps = np.concatenate(timestamps_parts)
    partition_code = np.concatenate(partition_parts)
    normal_status = np.concatenate(normal_status_parts)
    values = np.concatenate(values_parts, axis=0)
    order = np.argsort(timestamps.astype(np.int64), kind="stable")
    timestamps = timestamps[order]
    partition_code = partition_code[order]
    normal_status = normal_status[order]
    values = values[order]

    segment_id = _strict_segment_ids(timestamps)
    fit_mask = (partition_code == 0) & normal_status
    prediction_mask = partition_code == 1
    return {
        "timestamp_ns": timestamps.astype(np.int64),
        "partition_code": partition_code,
        "normal_status": normal_status,
        "fit_mask": fit_mask,
        "prediction_mask": prediction_mask,
        "segment_id": segment_id,
        "values": values,
    }


def _fit_medians(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    medians = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        observed = values[fit_mask, index]
        observed = observed[np.isfinite(observed)]
        if len(observed):
            medians[index] = float(np.median(observed))
    return medians


def prepare_case_cache(case_row: Any) -> dict[str, Any]:
    farm = str(case_row.farm)
    feature_table = CARE_FEATURE_REGISTRY.loc[
        CARE_FEATURE_REGISTRY["farm"].eq(farm)
        & CARE_FEATURE_REGISTRY["primary_analysis"]
    ].copy()
    feature_table = feature_table.sort_values("column", kind="stable").reset_index(drop=True)
    channels = feature_table["column"].tolist()
    if not channels:
        raise RuntimeError(f"{farm} has no primary average channels.")

    first_pass = _first_pass_fit_statistics(case_row, channels)
    arrays = _second_pass_arrays(case_row, channels, first_pass["structurally_zero"])
    values = arrays.pop("values")
    if len(values) != first_pass["total_rows"]:
        raise RuntimeError(f"{case_row.case_key}: first/second pass row mismatch.")

    metadata_source_asset = str(case_row.source_asset_id)
    observed_source_assets = {
        str(int(float(value))) if str(value).replace(".", "", 1).isdigit() and float(value).is_integer()
        else str(value)
        for value in first_pass["asset_values"]
    }
    if observed_source_assets != {metadata_source_asset}:
        raise RuntimeError(
            f"{case_row.case_key}: asset mismatch, source={observed_source_assets}, "
            f"metadata={metadata_source_asset}."
        )

    is_angle = feature_table["is_angle"].to_numpy(dtype=bool)
    is_counter = feature_table["is_counter"].to_numpy(dtype=bool)
    role = feature_table["role"].astype(str).to_numpy()
    usable = first_pass["usable"].copy()

    counter_values = values[:, is_counter]
    counter_names = [f"{name}__rate" for name in np.asarray(channels)[is_counter]]
    counter_differences = _counter_differences(counter_values, arrays["segment_id"])
    counter_rate_usable = np.zeros(counter_differences.shape[1], dtype=bool)
    for counter_index in range(counter_differences.shape[1]):
        observed = counter_differences[arrays["fit_mask"], counter_index]
        observed = observed[np.isfinite(observed)]
        rate_availability = len(observed) / max(int(arrays["fit_mask"].sum()), 1)
        counter_rate_usable[counter_index] = (
            rate_availability >= QUALITY.minimum_sensor_availability
            and len(observed) > 1
            and float(np.var(observed, ddof=1)) > 0.0
        )
    usable[is_counter] = counter_rate_usable

    noncounter = ~is_counter
    identity_mask = noncounter & ~is_angle
    angle_mask = noncounter & is_angle

    driver_identity = usable & identity_mask & (role == "operating_driver_candidate")
    driver_angles = usable & angle_mask & (role == "operating_driver_candidate")
    driver_counters = usable & is_counter & (role == "operating_driver_candidate")
    target_identity = usable & identity_mask & (role != "operating_driver_candidate")
    target_angles = usable & angle_mask & (role != "operating_driver_candidate")
    target_counters = usable & is_counter & (role != "operating_driver_candidate")

    driver_blocks: list[np.ndarray] = []
    driver_names: list[str] = []
    driver_imputation_medians: list[float] = []
    driver_missing_indicator_names: list[str] = []

    if driver_identity.any():
        block = values[:, driver_identity]
        names = list(np.asarray(channels)[driver_identity])
        medians = _fit_medians(block, arrays["fit_mask"])
        missing = ~np.isfinite(block)
        block = np.where(missing, medians[None, :], block)
        driver_blocks.append(block)
        driver_names.extend(names)
        driver_imputation_medians.extend(medians.tolist())
        # Add indicators only for channels that are actually missing anywhere.
        for column_index, name in enumerate(names):
            if missing[:, column_index].any():
                driver_blocks.append(missing[:, [column_index]].astype(np.float64))
                driver_names.append(f"{name}__missing")
                driver_imputation_medians.append(0.0)
                driver_missing_indicator_names.append(f"{name}__missing")

    if driver_angles.any():
        block = values[:, driver_angles]
        names = list(np.asarray(channels)[driver_angles])
        radians = np.deg2rad(block)
        for column_index, name in enumerate(names):
            pair = np.column_stack((np.sin(radians[:, column_index]), np.cos(radians[:, column_index])))
            missing = ~np.isfinite(pair).all(axis=1)
            pair[missing] = 0.0
            driver_blocks.append(pair)
            driver_names.extend((f"{name}__sin", f"{name}__cos"))
            driver_imputation_medians.extend((0.0, 0.0))
            if missing.any():
                driver_blocks.append(missing[:, None].astype(np.float64))
                driver_names.append(f"{name}__missing")
                driver_imputation_medians.append(0.0)
                driver_missing_indicator_names.append(f"{name}__missing")

    if driver_counters.any():
        counter_selected = driver_counters[is_counter]
        block = counter_differences[:, counter_selected]
        names = list(np.asarray(counter_names)[counter_selected])
        medians = _fit_medians(block, arrays["fit_mask"])
        missing = ~np.isfinite(block)
        block = np.where(missing, medians[None, :], block)
        driver_blocks.append(block)
        driver_names.extend(names)
        driver_imputation_medians.extend(medians.tolist())
        for column_index, name in enumerate(names):
            if missing[:, column_index].any():
                driver_blocks.append(missing[:, [column_index]].astype(np.float64))
                driver_names.append(f"{name}__missing")
                driver_imputation_medians.append(0.0)
                driver_missing_indicator_names.append(f"{name}__missing")

    target_blocks: list[np.ndarray] = []
    target_names: list[str] = []
    target_temperature: list[bool] = []

    if target_identity.any():
        target_blocks.append(values[:, target_identity])
        identity_names = list(np.asarray(channels)[target_identity])
        target_names.extend(identity_names)
        target_temperature.extend(
            (role[target_identity] == "temperature_target_candidate").tolist()
        )

    if target_angles.any():
        block = values[:, target_angles]
        names = list(np.asarray(channels)[target_angles])
        temp = role[target_angles] == "temperature_target_candidate"
        radians = np.deg2rad(block)
        for column_index, name in enumerate(names):
            target_blocks.append(
                np.column_stack((np.sin(radians[:, column_index]), np.cos(radians[:, column_index])))
            )
            target_names.extend((f"{name}__sin", f"{name}__cos"))
            target_temperature.extend((bool(temp[column_index]), bool(temp[column_index])))

    if target_counters.any():
        counter_selected = target_counters[is_counter]
        target_blocks.append(counter_differences[:, counter_selected])
        selected_names = list(np.asarray(counter_names)[counter_selected])
        target_names.extend(selected_names)
        target_temperature.extend([False] * len(selected_names))

    if not driver_blocks:
        raise RuntimeError(f"{case_row.case_key}: no usable operating drivers.")
    if not target_blocks:
        raise RuntimeError(f"{case_row.case_key}: no usable monitoring targets.")

    driver_matrix = np.concatenate(driver_blocks, axis=1).astype(np.float32)
    target_matrix = np.concatenate(target_blocks, axis=1).astype(np.float32)
    target_observed = np.isfinite(target_matrix)

    if not np.isfinite(driver_matrix).all():
        raise RuntimeError(f"{case_row.case_key}: driver imputation left non-finite values.")
    if len(driver_names) != driver_matrix.shape[1]:
        raise RuntimeError(f"{case_row.case_key}: driver-name mismatch.")
    if len(target_names) != target_matrix.shape[1]:
        raise RuntimeError(f"{case_row.case_key}: target-name mismatch.")

    cache_relative = Path(_safe_slug(farm)) / f"event_{int(case_row.event_id):03d}.npz"
    metadata_relative = Path(_safe_slug(farm)) / f"event_{int(case_row.event_id):03d}.json"
    cache_path = CELL3_CACHE_ROOT / cache_relative
    metadata_path = CELL3_CACHE_ROOT / metadata_relative

    case_metadata = {
        "cell3_version": CELL3_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "cell2_audit_sha256": CELL2_AUDIT_SHA256,
        "preprocessing_policy_sha256": PREPROCESSING_POLICY_SHA256,
        "case_key": case_row.case_key,
        "farm": farm,
        "asset_id": case_row.asset_id,
        "source_asset_id": metadata_source_asset,
        "event_id": int(case_row.event_id),
        "rows": int(len(driver_matrix)),
        "training_normal_rows": int(arrays["fit_mask"].sum()),
        "prediction_rows": int(arrays["prediction_mask"].sum()),
        "segments": int(arrays["segment_id"].max() + 1),
        "raw_primary_channels": channels,
        "raw_roles": role.tolist(),
        "raw_availability": first_pass["availability"].tolist(),
        "raw_finite_count": first_pass["finite_count"].tolist(),
        "raw_zero_count": first_pass["zero_count"].tolist(),
        "raw_structurally_zero": first_pass["structurally_zero"].tolist(),
        "raw_usable": usable.tolist(),
        "driver_names": driver_names,
        "driver_imputation_medians": driver_imputation_medians,
        "driver_missing_indicator_names": driver_missing_indicator_names,
        "target_names": target_names,
        "target_temperature": target_temperature,
        "outcome_fields_present": False,
    }
    case_preprocessing_sha256 = sha256_json(case_metadata)
    case_metadata["case_preprocessing_sha256"] = case_preprocessing_sha256

    if metadata_path.exists() and cache_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("case_preprocessing_sha256") != case_preprocessing_sha256:
            raise RuntimeError(
                f"Preprocessing drift for {case_row.case_key}. Do not overwrite the cache."
            )
        cache_sha256 = _json_file_sha256(cache_path)
        if existing.get("cache_sha256") != cache_sha256:
            raise RuntimeError(
                f"Cached arrays failed their saved hash for {case_row.case_key}. "
                "Delete only that derived cache and rerun Cell 3."
            )
    else:
        _atomic_save_npz(
            cache_path,
            timestamp_ns=arrays["timestamp_ns"],
            partition_code=arrays["partition_code"],
            normal_status=arrays["normal_status"],
            fit_mask=arrays["fit_mask"],
            prediction_mask=arrays["prediction_mask"],
            segment_id=arrays["segment_id"],
            drivers=driver_matrix,
            targets=target_matrix,
            target_observed=target_observed,
            target_temperature=np.asarray(target_temperature, dtype=bool),
        )
        cache_sha256 = _json_file_sha256(cache_path)
        case_metadata["cache_sha256"] = cache_sha256
        save_json(case_metadata, metadata_path)

    return {
        "case_key": case_row.case_key,
        "farm": farm,
        "asset_id": case_row.asset_id,
        "event_id": int(case_row.event_id),
        "rows": int(len(driver_matrix)),
        "training_normal_rows": int(arrays["fit_mask"].sum()),
        "prediction_rows": int(arrays["prediction_mask"].sum()),
        "segments": int(arrays["segment_id"].max() + 1),
        "primary_channels": len(channels),
        "usable_driver_channels": int(
            driver_identity.sum() + driver_angles.sum() + driver_counters.sum()
        ),
        "usable_target_channels": int(
            target_identity.sum() + target_angles.sum() + target_counters.sum()
        ),
        "unusable_channels": int((~usable).sum()),
        "structurally_zero_channels": int(first_pass["structurally_zero"].sum()),
        "cache_relative_path": cache_relative.as_posix(),
        "metadata_relative_path": metadata_relative.as_posix(),
        "case_preprocessing_sha256": case_preprocessing_sha256,
        "cache_sha256": cache_sha256,
    }


def load_case_cache(case_key: str) -> dict[str, Any]:
    """Load a safe cached case by key; no outcomes are present in this cache."""
    match = CASE_CACHE_REGISTRY.loc[CASE_CACHE_REGISTRY["case_key"].eq(case_key)]
    if len(match) != 1:
        raise KeyError(f"Unknown or duplicate case_key: {case_key}")
    row = match.iloc[0]
    metadata_path = CELL3_CACHE_ROOT / row["metadata_relative_path"]
    cache_path = CELL3_CACHE_ROOT / row["cache_relative_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("outcome_fields_present") is not False:
        raise RuntimeError(f"Unsafe cache metadata for {case_key}.")
    if _json_file_sha256(cache_path) != row["cache_sha256"]:
        raise RuntimeError(f"Cache hash mismatch for {case_key}.")
    arrays = dict(np.load(cache_path, allow_pickle=False))
    arrays["metadata"] = metadata
    return arrays


# =============================================================================
# 4. Materialize all label-free case caches
# =============================================================================

_cache_receipts: list[dict[str, Any]] = []
for _farm in DATASET.farms:
    _farm_cases = CASE_REGISTRY.loc[CASE_REGISTRY["farm"].eq(_farm)]
    print(f"Preparing label-free case caches — {_farm}: {len(_farm_cases)} cases", flush=True)
    for _row in _farm_cases.itertuples(index=False):
        _cache_receipts.append(prepare_case_cache(_row))

CASE_CACHE_REGISTRY = pd.DataFrame(_cache_receipts).sort_values(
    ["farm", "event_id"], kind="stable"
).reset_index(drop=True)

if len(CASE_CACHE_REGISTRY) != DATASET.expected_total_cases:
    raise RuntimeError("Not all 95 cases produced a preprocessing cache.")
if CASE_CACHE_REGISTRY["case_key"].duplicated().any():
    raise RuntimeError("Duplicate case cache keys were produced.")

# A small sample is loaded immediately to verify cache schema and hashes.
for _case_key in CASE_CACHE_REGISTRY.groupby("farm", sort=False).head(1)["case_key"]:
    _sample = load_case_cache(_case_key)
    _required_arrays = {
        "timestamp_ns",
        "partition_code",
        "normal_status",
        "fit_mask",
        "prediction_mask",
        "segment_id",
        "drivers",
        "targets",
        "target_observed",
        "target_temperature",
        "metadata",
    }
    if set(_sample) != _required_arrays:
        raise RuntimeError(f"Unexpected cache members for {_case_key}: {set(_sample)}")
    if len(_sample["drivers"]) != len(_sample["targets"]):
        raise RuntimeError(f"Row mismatch inside cache {_case_key}.")
    del _sample


# =============================================================================
# 5. Freeze Cell 3 receipt and write auditable tables
# =============================================================================

def dataframe_sha256(frame: pd.DataFrame, sort_columns: Iterable[str]) -> str:
    ordered = frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.12g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_component_hashes = {
    "asset_registry_sha256": dataframe_sha256(ASSET_REGISTRY, ("farm", "asset_id")),
    "outer_folds_sha256": dataframe_sha256(OUTER_FOLDS, ("outer_fold",)),
    "inner_asset_folds_sha256": dataframe_sha256(
        INNER_ASSET_FOLDS, ("outer_fold", "inner_fold", "farm", "asset_id")
    ),
    "case_cache_registry_sha256": dataframe_sha256(
        CASE_CACHE_REGISTRY, ("farm", "event_id")
    ),
}
CELL3_RECEIPT_SHA256 = sha256_json(
    {
        "contract_sha256": CONTRACT_SHA256,
        "cell2_audit_sha256": CELL2_AUDIT_SHA256,
        "preprocessing_policy_sha256": PREPROCESSING_POLICY_SHA256,
        "component_hashes": _component_hashes,
    }
)

CELL3_RECEIPT_PATH = CELL3_FOLD_DIR / "cell3_fold_and_preprocessing_receipt.json"
if CELL3_RECEIPT_PATH.exists():
    _existing = json.loads(CELL3_RECEIPT_PATH.read_text(encoding="utf-8"))
    if _existing.get("cell3_receipt_sha256") != CELL3_RECEIPT_SHA256:
        raise RuntimeError(
            "A different Cell 3 receipt exists for this experiment. Do not overwrite it."
        )
    CELL3_STATE = "existing identical Cell 3 receipt verified"
    _write_receipt = False
else:
    CELL3_STATE = "new Cell 3 receipt frozen"
    _write_receipt = True

save_csv_atomic(ASSET_REGISTRY, CELL3_FOLD_DIR / "asset_registry.csv")
save_csv_atomic(OUTER_FOLDS, CELL3_FOLD_DIR / "outer_leave_one_asset_out_folds.csv")
save_csv_atomic(INNER_ASSET_FOLDS, CELL3_FOLD_DIR / "inner_grouped_asset_folds.csv")
save_csv_atomic(CASE_CACHE_REGISTRY, CELL3_QUALITY_DIR / "case_cache_registry.csv")
save_json(PREPROCESSING_POLICY, CELL3_QUALITY_DIR / "preprocessing_policy.json")

if _write_receipt:
    save_json(
        {
            "cell3_version": CELL3_VERSION,
            "created_at_utc": utc_now(),
            "contract_sha256": CONTRACT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "outcome_lockbox_sha256": OUTCOME_LOCKBOX_SHA256,
            "cell2_audit_sha256": CELL2_AUDIT_SHA256,
            "preprocessing_policy_sha256": PREPROCESSING_POLICY_SHA256,
            "component_hashes": _component_hashes,
            "cell3_receipt_sha256": CELL3_RECEIPT_SHA256,
            "outcomes_read": False,
            "global_sensor_selection": False,
        },
        CELL3_RECEIPT_PATH,
    )


# =============================================================================
# 6. Concise notebook report
# =============================================================================

_fold_balance = (
    INNER_ASSET_FOLDS.groupby(["outer_fold", "inner_fold"])
    .agg(validation_assets=("asset_id", "nunique"), validation_cases=("cases", "sum"))
    .reset_index()
)
INNER_FOLD_BALANCE_SUMMARY = (
    _fold_balance.groupby("inner_fold")
    .agg(
        min_validation_assets=("validation_assets", "min"),
        max_validation_assets=("validation_assets", "max"),
        min_validation_cases=("validation_cases", "min"),
        max_validation_cases=("validation_cases", "max"),
    )
    .reset_index()
)

CASE_CACHE_SUMMARY = (
    CASE_CACHE_REGISTRY.groupby("farm", sort=False)
    .agg(
        cases=("case_key", "size"),
        assets=("asset_id", "nunique"),
        rows=("rows", "sum"),
        median_drivers=("usable_driver_channels", "median"),
        median_targets=("usable_target_channels", "median"),
        median_unusable=("unusable_channels", "median"),
        structurally_zero_case_channels=("structurally_zero_channels", "sum"),
        median_segments=("segments", "median"),
    )
    .reset_index()
)

print("\n" + "=" * 92)
print("UC-RCF-NBM CELL 3 — ASSET FOLDS AND LEAKAGE-SAFE PREPROCESSING")
print("=" * 92)
print("\nASSET REGISTRY")
display(
    ASSET_REGISTRY.groupby("farm", sort=False)
    .agg(assets=("asset_id", "nunique"), cases=("cases", "sum"))
    .reset_index()
)
print("\nINNER-FOLD BALANCE ACROSS THE 36 OUTER FOLDS")
display(INNER_FOLD_BALANCE_SUMMARY)
print("\nLABEL-FREE CASE CACHE SUMMARY")
display(CASE_CACHE_SUMMARY)

print("\n" + "-" * 92)
print(f"Outer folds                     : {len(OUTER_FOLDS)}")
print(f"Outer test assets               : {OUTER_FOLDS['outer_asset_id'].nunique()}")
print(f"Inner folds per outer fold      : {EVALUATION.inner_splits}")
print(f"Cases cached                    : {len(CASE_CACHE_REGISTRY)}")
print(f"Preprocessing policy SHA-256    : {PREPROCESSING_POLICY_SHA256}")
print(f"Cell 3 receipt SHA-256          : {CELL3_RECEIPT_SHA256}")
print(f"Cell 3 state                    : {CELL3_STATE}")
print("Outcome metadata loaded         : No")
print("Case labels in cache            : No")
print("Global sensor selection applied : No")
print("Outer prediction used for fit   : No")
print("Fold leakage checks             : PASS")
print("=" * 92)
print("CELL 3 COMPLETED SUCCESSFULLY — ASSET FOLDS AND PREPROCESSING LOCKED")
