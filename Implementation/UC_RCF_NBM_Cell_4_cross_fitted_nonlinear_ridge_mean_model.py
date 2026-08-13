"""CELL 4 — cross-fitted nonlinear ridge normal-behaviour mean model.

Paste this complete file into the fourth cell of the UC-RCF-NBM notebook and
run it only after Cells 1–3 have completed successfully.

For every CARE case, this cell constructs embargoed temporal folds from the
normal source-training history, refits all preprocessing inside each fold,
selects the ridge penalty by masked out-of-fold error, produces honest OOF
residuals, and fits the final normal-behaviour mean model. It never reads event
labels, event boundaries, failure descriptions, or CARE outcome scores.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# =============================================================================
# 0. Bind Cell 4 to the completed experiment receipts
# =============================================================================

EXPECTED_CONTRACT_SHA256 = (
    "827641aecd8e807193ad193d64c274319756092faa53bdb5084f310f62041f49"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "62484bab1219888aa1d0788965ecd77db2b85f0bbb9b476cd3240f4143026f1f"
)
EXPECTED_CELL2_AUDIT_SHA256 = (
    "83732caf4ad3e226b69a671287bb14c4ed72e1bfc3b7c26ca144310fce8e5990"
)
EXPECTED_PREPROCESSING_POLICY_SHA256 = (
    "67ccb2442d0a44393ca7e3cc0bc8030f7cc3fc69458c1dce9f9b36c57577dcc9"
)
EXPECTED_CELL3_RECEIPT_SHA256 = (
    "aded107bea4397babbf24f4b9ab5740d9cfdd117a3783b3f03bd1cbcfbc55762"
)

_required_objects = (
    "CONTRACT_SHA256",
    "DATASET_MANIFEST_SHA256",
    "CELL2_AUDIT_SHA256",
    "PREPROCESSING_POLICY_SHA256",
    "CELL3_RECEIPT_SHA256",
    "DATASET",
    "MEAN_MODEL",
    "REPRODUCIBILITY",
    "CASE_REGISTRY",
    "CASE_CACHE_REGISTRY",
    "CARE_FEATURE_REGISTRY",
    "load_case_cache",
    "save_csv_atomic",
    "save_json",
    "sha256_json",
    "utc_now",
    "MODEL_DIR",
    "CACHE_DIR",
    "QUALITY_DIR",
)
_missing_objects = [name for name in _required_objects if name not in globals()]
if _missing_objects:
    raise RuntimeError(
        "Run UC-RCF-NBM Cells 1–3 before Cell 4. Missing objects: "
        + ", ".join(_missing_objects)
    )

_observed_receipts = {
    "contract": CONTRACT_SHA256,
    "dataset_manifest": DATASET_MANIFEST_SHA256,
    "cell2_audit": CELL2_AUDIT_SHA256,
    "preprocessing_policy": PREPROCESSING_POLICY_SHA256,
    "cell3_receipt": CELL3_RECEIPT_SHA256,
}
_expected_receipts = {
    "contract": EXPECTED_CONTRACT_SHA256,
    "dataset_manifest": EXPECTED_DATASET_MANIFEST_SHA256,
    "cell2_audit": EXPECTED_CELL2_AUDIT_SHA256,
    "preprocessing_policy": EXPECTED_PREPROCESSING_POLICY_SHA256,
    "cell3_receipt": EXPECTED_CELL3_RECEIPT_SHA256,
}
if _observed_receipts != _expected_receipts:
    raise RuntimeError(
        "Cell 4 is bound to the exact completed Cell 1–3 receipts. "
        f"Observed={_observed_receipts}, expected={_expected_receipts}."
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
    raise RuntimeError("Outcome information is present in the safe case registry.")

if MEAN_MODEL.crossfit_folds != 5:
    raise RuntimeError("Cell 4 requires the frozen five temporal cross-fitting folds.")
if MEAN_MODEL.temporal_block_steps <= MEAN_MODEL.embargo_steps:
    raise RuntimeError("Temporal blocks must be longer than the embargo.")
if tuple(MEAN_MODEL.ridge_grid) != tuple(sorted(MEAN_MODEL.ridge_grid)):
    raise RuntimeError("The frozen ridge grid must be ordered.")

CELL4_VERSION = "1.0.0"
CELL4_MODEL_ROOT = Path(MODEL_DIR) / "cell4_cross_fitted_mean_nbm"
CELL4_RESIDUAL_ROOT = Path(CACHE_DIR) / "cell4_cross_fitted_mean_residuals"
CELL4_QUALITY_ROOT = Path(QUALITY_DIR) / "cell4_cross_fitted_mean_nbm"
for _directory in (CELL4_MODEL_ROOT, CELL4_RESIDUAL_ROOT, CELL4_QUALITY_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Freeze implementation choices that do not use outcome information
# =============================================================================

MEAN_MODEL_IMPLEMENTATION = {
    "version": CELL4_VERSION,
    "model": MEAN_MODEL.model,
    "basis": MEAN_MODEL.basis,
    "temporal_blocks": {
        "folds": MEAN_MODEL.crossfit_folds,
        "block_steps": MEAN_MODEL.temporal_block_steps,
        "embargo_steps": MEAN_MODEL.embargo_steps,
        "assignment": "round-robin complete blocks within continuous segments",
        "embargo": "two-sided in original row steps within each segment",
    },
    "fold_local_preprocessing": {
        "driver_imputation": "training-fold median reconstructed from missingness indicators",
        "driver_scaling": "training-fold median and IQR; standard-deviation fallback",
        "basis_scaling": "training-fold median and IQR; standard-deviation fallback",
        "target_scaling": "target-wise training-fold median and IQR; standard-deviation fallback",
    },
    "basis_terms": {
        "linear": "all nonconstant processed drivers and missingness indicators",
        "quadratic": "all non-indicator drivers",
        "wind_cubic": "non-indicator wind-speed drivers only",
        "interactions": (
            f"all pairs among at most {MEAN_MODEL.maximum_core_interaction_drivers} "
            "deterministically prioritized physical drivers"
        ),
    },
    "ridge": {
        "penalties": tuple(MEAN_MODEL.ridge_grid),
        "intercept_penalized": False,
        "selection": MEAN_MODEL.penalty_objective,
        "selection_residual_units": "fold-local robust target scales",
        "selection_residual_cap": MEAN_MODEL.residual_cap,
        "tie_break": "largest penalty among numerically tied minima",
    },
    "masked_targets": {
        "grouping": "targets sharing identical training-observation masks",
        "minimum_observed_rows": 200,
        "minimum_rows_above_design_dimension": 10,
        "missing_targets_imputed": False,
    },
    "outputs": {
        "oof_residuals": "raw target units on observed training-normal rows only",
        "final_model": "coefficients plus deterministic transformation state",
        "final_predictions_cached": False,
        "outcomes_read": False,
    },
}
MEAN_MODEL_IMPLEMENTATION_SHA256 = sha256_json(MEAN_MODEL_IMPLEMENTATION)

NUMERICAL_EPSILON = 1.0e-12
MINIMUM_TARGET_ROWS = 200
RIDGE_PENALTIES = np.asarray(MEAN_MODEL.ridge_grid, dtype=np.float64)


# =============================================================================
# 2. General integrity and serialization helpers
# =============================================================================

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame, sort_columns: Iterable[str]) -> str:
    # JSON receipts are written with sorted keys. Canonical column ordering keeps
    # a first-run in-memory DataFrame identical to the same summaries reloaded
    # from those JSON files on a repeat run.
    ordered = (
        frame.loc[:, sorted(frame.columns)]
        .sort_values(list(sort_columns), kind="stable")
        .reset_index(drop=True)
    )
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.12g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_save_npz(output_path: Path, **arrays: np.ndarray) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output_path)


def safe_slug(text: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part)


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise median/IQR with standard-deviation fallback."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("robust_location_scale expects a two-dimensional matrix")
    location = np.median(values, axis=0)
    q25, q75 = np.percentile(values, (25.0, 75.0), axis=0)
    scale = q75 - q25
    standard_deviation = np.std(values, axis=0, ddof=0)
    use_standard_deviation = (~np.isfinite(scale)) | (scale <= NUMERICAL_EPSILON)
    scale = np.where(use_standard_deviation, standard_deviation, scale)
    valid = np.isfinite(location) & np.isfinite(scale) & (scale > NUMERICAL_EPSILON)
    safe_scale = np.where(valid, scale, 1.0)
    return location, safe_scale, valid


def robust_target_parameters(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust parameters for a fully observed target group."""
    return robust_location_scale(values)


# =============================================================================
# 3. Embargoed temporal block construction
# =============================================================================

def make_temporal_fold_ids(segment_id: np.ndarray) -> np.ndarray:
    """Assign every row to one complete temporal block and cross-fit fold."""
    segment_id = np.asarray(segment_id, dtype=np.int64)
    fold_id = np.full(len(segment_id), -1, dtype=np.int8)
    next_global_block = 0

    for segment in np.unique(segment_id):
        rows = np.flatnonzero(segment_id == segment)
        if len(rows) == 0:
            continue
        local_blocks = np.arange(len(rows), dtype=np.int64) // MEAN_MODEL.temporal_block_steps
        for local_block in np.unique(local_blocks):
            block_rows = rows[local_blocks == local_block]
            fold_id[block_rows] = next_global_block % MEAN_MODEL.crossfit_folds
            next_global_block += 1

    if (fold_id < 0).any():
        raise RuntimeError("At least one row was not assigned to a temporal fold.")
    return fold_id


def embargo_dilation(
    fold_id: np.ndarray,
    validation_fold: int,
    segment_id: np.ndarray,
) -> np.ndarray:
    """Dilate validation blocks by the frozen two-sided embargo within segments."""
    fold_id = np.asarray(fold_id)
    segment_id = np.asarray(segment_id)
    excluded = np.zeros(len(fold_id), dtype=bool)
    radius = int(MEAN_MODEL.embargo_steps)

    for segment in np.unique(segment_id):
        rows = np.flatnonzero(segment_id == segment)
        local_validation = fold_id[rows] == validation_fold
        if not local_validation.any():
            continue
        padded = np.concatenate(([False], local_validation, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        difference = np.zeros(len(rows) + 1, dtype=np.int64)
        for start, end in zip(starts, ends):
            left = max(0, int(start) - radius)
            right = min(len(rows), int(end) + radius)
            difference[left] += 1
            difference[right] -= 1
        excluded[rows] = np.cumsum(difference[:-1]) > 0
    return excluded


def iter_case_temporal_folds(
    fit_mask: np.ndarray,
    segment_id: np.ndarray,
) -> Iterator[dict[str, Any]]:
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fold_id = make_temporal_fold_ids(segment_id)

    for validation_fold in range(MEAN_MODEL.crossfit_folds):
        validation_mask = fit_mask & (fold_id == validation_fold)
        dilated = embargo_dilation(fold_id, validation_fold, segment_id)
        candidate_training = fit_mask & (fold_id != validation_fold)
        training_mask = candidate_training & ~dilated
        if not validation_mask.any():
            raise RuntimeError(f"Temporal validation fold {validation_fold + 1} is empty.")
        if training_mask.sum() < MINIMUM_TARGET_ROWS:
            raise RuntimeError(
                f"Temporal training fold {validation_fold + 1} contains only "
                f"{int(training_mask.sum())} eligible rows."
            )
        yield {
            "fold": validation_fold,
            "fold_number": validation_fold + 1,
            "fold_id": fold_id,
            "training_mask": training_mask,
            "validation_mask": validation_mask,
            "embargoed_fit_rows": int((candidate_training & dilated).sum()),
        }


# =============================================================================
# 4. Fold-local nonlinear driver transformation
# =============================================================================

_feature_description_lookup = {
    (str(row.farm), str(row.column)): str(row.description)
    for row in CARE_FEATURE_REGISTRY.itertuples(index=False)
}


def raw_driver_name(processed_name: str) -> str:
    name = str(processed_name)
    for _ in range(3):
        changed = False
        for suffix in ("__missing", "__sin", "__cos", "__rate"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
        if not changed:
            break
    return name


def semantic_text(farm: str, processed_name: str) -> str:
    raw = raw_driver_name(processed_name)
    description = _feature_description_lookup.get((farm, raw), "")
    text = f"{raw} {description}".lower().replace("windspeed", "wind speed")
    return re.sub(r"\s+", " ", text).strip()


def semantic_match(text: str, phrase: str) -> bool:
    phrase_pattern = re.escape(str(phrase).lower()).replace(r"\ ", r"\s+")
    return re.search(rf"\b{phrase_pattern}\b", text) is not None


def missing_indicator_name(processed_name: str) -> str:
    if processed_name.endswith("__sin") or processed_name.endswith("__cos"):
        return processed_name.rsplit("__", 1)[0] + "__missing"
    return processed_name + "__missing"


def choose_core_driver_indices(farm: str, driver_names: list[str]) -> list[int]:
    eligible = [
        index for index, name in enumerate(driver_names)
        if not name.endswith("__missing")
    ]
    selected: list[int] = []
    for pattern in MEAN_MODEL.driver_description_patterns:
        matches = [
            index for index in eligible
            if index not in selected and semantic_match(semantic_text(farm, driver_names[index]), pattern)
        ]
        matches.sort(key=lambda index: driver_names[index])
        for index in matches:
            selected.append(index)
            if len(selected) == MEAN_MODEL.maximum_core_interaction_drivers:
                return selected
    for index in sorted(eligible, key=lambda item: driver_names[item]):
        if index not in selected:
            selected.append(index)
        if len(selected) == MEAN_MODEL.maximum_core_interaction_drivers:
            break
    return selected


def fit_design_state(
    drivers: np.ndarray,
    driver_names: list[str],
    training_mask: np.ndarray,
    farm: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit all driver transformations using one temporal training fold only."""
    drivers = np.asarray(drivers, dtype=np.float64)
    training_mask = np.asarray(training_mask, dtype=bool)
    names = list(driver_names)
    if drivers.shape[1] != len(names):
        raise ValueError("Driver matrix and driver-name count differ.")

    indicator_lookup = {
        name: index for index, name in enumerate(names) if name.endswith("__missing")
    }
    is_indicator = np.asarray(
        [name.endswith("__missing") for name in names], dtype=bool
    )
    imputation = np.zeros(drivers.shape[1], dtype=np.float64)
    filled = drivers.copy()

    for index, name in enumerate(names):
        if is_indicator[index]:
            filled[:, index] = np.where(np.isfinite(filled[:, index]), filled[:, index], 0.0)
            continue
        indicator_index = indicator_lookup.get(missing_indicator_name(name))
        missing = ~np.isfinite(filled[:, index])
        if indicator_index is not None:
            missing |= filled[:, indicator_index] > 0.5
        observed_training = training_mask & ~missing & np.isfinite(filled[:, index])
        if not observed_training.any():
            raise RuntimeError(f"No fold-training observation for driver {name!r}.")
        imputation[index] = float(np.median(filled[observed_training, index]))
        filled[missing, index] = imputation[index]

    raw_location = np.zeros(drivers.shape[1], dtype=np.float64)
    raw_scale = np.ones(drivers.shape[1], dtype=np.float64)
    if (~is_indicator).any():
        location, scale, _ = robust_location_scale(
            filled[training_mask][:, ~is_indicator]
        )
        raw_location[~is_indicator] = location
        raw_scale[~is_indicator] = scale
    standardized = (filled - raw_location[None, :]) / raw_scale[None, :]

    term_kind: list[str] = []
    term_left: list[int] = []
    term_right: list[int] = []
    term_names: list[str] = []

    for index, name in enumerate(names):
        term_kind.append("linear")
        term_left.append(index)
        term_right.append(-1)
        term_names.append(f"linear::{name}")

    for index, name in enumerate(names):
        if is_indicator[index]:
            continue
        term_kind.append("quadratic")
        term_left.append(index)
        term_right.append(-1)
        term_names.append(f"quadratic::{name}")

    for index, name in enumerate(names):
        if is_indicator[index]:
            continue
        if semantic_match(semantic_text(farm, name), "wind speed"):
            term_kind.append("cubic")
            term_left.append(index)
            term_right.append(-1)
            term_names.append(f"wind_cubic::{name}")

    core_indices = choose_core_driver_indices(farm, names)
    for left, right in combinations(core_indices, 2):
        term_kind.append("interaction")
        term_left.append(left)
        term_right.append(right)
        term_names.append(f"interaction::{names[left]}*{names[right]}")

    basis = build_unscaled_basis(
        standardized,
        term_kind,
        np.asarray(term_left, dtype=np.int32),
        np.asarray(term_right, dtype=np.int32),
    )
    term_location, term_scale, active = robust_location_scale(basis[training_mask])
    if not active.any():
        raise RuntimeError("All nonlinear design terms are constant in a training fold.")
    design = (
        basis[:, active] - term_location[None, active]
    ) / term_scale[None, active]

    state = {
        "driver_names": names,
        "imputation": imputation,
        "raw_location": raw_location,
        "raw_scale": raw_scale,
        "is_indicator": is_indicator,
        "term_kind": term_kind,
        "term_left": np.asarray(term_left, dtype=np.int32),
        "term_right": np.asarray(term_right, dtype=np.int32),
        "term_names": term_names,
        "term_location": term_location,
        "term_scale": term_scale,
        "term_active": active,
        "core_driver_names": [names[index] for index in core_indices],
    }
    return state, design


def build_unscaled_basis(
    standardized_drivers: np.ndarray,
    term_kind: list[str],
    term_left: np.ndarray,
    term_right: np.ndarray,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for kind, left, right in zip(term_kind, term_left, term_right):
        if kind == "linear":
            columns.append(standardized_drivers[:, left])
        elif kind == "quadratic":
            columns.append(np.square(standardized_drivers[:, left]))
        elif kind == "cubic":
            columns.append(np.power(standardized_drivers[:, left], 3))
        elif kind == "interaction":
            columns.append(
                standardized_drivers[:, left] * standardized_drivers[:, right]
            )
        else:
            raise ValueError(f"Unknown design term kind: {kind}")
    return np.column_stack(columns)


def transform_design(drivers: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    drivers = np.asarray(drivers, dtype=np.float64)
    names = list(state["driver_names"])
    indicator_lookup = {
        name: index for index, name in enumerate(names) if name.endswith("__missing")
    }
    is_indicator = np.asarray(state["is_indicator"], dtype=bool)
    filled = drivers.copy()
    imputation = np.asarray(state["imputation"], dtype=np.float64)

    for index, name in enumerate(names):
        if is_indicator[index]:
            filled[:, index] = np.where(np.isfinite(filled[:, index]), filled[:, index], 0.0)
            continue
        indicator_index = indicator_lookup.get(missing_indicator_name(name))
        missing = ~np.isfinite(filled[:, index])
        if indicator_index is not None:
            missing |= filled[:, indicator_index] > 0.5
        filled[missing, index] = imputation[index]

    standardized = (
        filled - np.asarray(state["raw_location"])[None, :]
    ) / np.asarray(state["raw_scale"])[None, :]
    basis = build_unscaled_basis(
        standardized,
        list(state["term_kind"]),
        np.asarray(state["term_left"], dtype=np.int32),
        np.asarray(state["term_right"], dtype=np.int32),
    )
    active = np.asarray(state["term_active"], dtype=bool)
    return (
        basis[:, active] - np.asarray(state["term_location"])[None, active]
    ) / np.asarray(state["term_scale"])[None, active]


# =============================================================================
# 5. Mask-aware grouped multi-target ridge fitting
# =============================================================================

def group_targets_by_observation_mask(observed: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group targets with identical row-observation masks."""
    observed = np.asarray(observed, dtype=bool)
    packed = np.packbits(observed, axis=0)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for target_index in range(observed.shape[1]):
        groups[packed[:, target_index].tobytes()].append(target_index)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for target_indices in groups.values():
        indices = np.asarray(target_indices, dtype=np.int32)
        output.append((indices, observed[:, indices[0]]))
    return output


def augmented_design(design: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(design), dtype=np.float64), design))


def solve_ridge(
    gram: np.ndarray,
    cross_product: np.ndarray,
    penalty: float,
) -> np.ndarray:
    regularized = gram.copy()
    diagonal = np.arange(len(regularized))
    regularized[diagonal[1:], diagonal[1:]] += float(penalty)
    try:
        return np.linalg.solve(regularized, cross_product)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(regularized, cross_product, rcond=None)[0]


def score_penalties_for_fold(
    design: np.ndarray,
    targets: np.ndarray,
    training_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    training_rows = np.flatnonzero(training_mask)
    validation_rows = np.flatnonzero(validation_mask)
    x_train = augmented_design(design[training_rows])
    x_validation = augmented_design(design[validation_rows])
    y_train = np.asarray(targets[training_rows], dtype=np.float64)
    y_validation = np.asarray(targets[validation_rows], dtype=np.float64)
    observed_train = np.isfinite(y_train)

    squared_error = np.zeros(len(RIDGE_PENALTIES), dtype=np.float64)
    observation_count = np.zeros(len(RIDGE_PENALTIES), dtype=np.int64)
    fitted_targets = 0

    minimum_rows = max(MINIMUM_TARGET_ROWS, x_train.shape[1] + 10)
    for target_indices, group_observed in group_targets_by_observation_mask(observed_train):
        if int(group_observed.sum()) < minimum_rows:
            continue
        x_observed = x_train[group_observed]
        y_observed = y_train[group_observed][:, target_indices]
        location, scale, valid_target = robust_target_parameters(y_observed)
        if not valid_target.any():
            continue
        target_indices = target_indices[valid_target]
        location = location[valid_target]
        scale = scale[valid_target]
        y_standardized = (y_observed[:, valid_target] - location[None, :]) / scale[None, :]
        gram = x_observed.T @ x_observed
        cross_product = x_observed.T @ y_standardized
        validation_values = y_validation[:, target_indices]
        validation_observed = np.isfinite(validation_values)
        fitted_targets += len(target_indices)

        for penalty_index, penalty in enumerate(RIDGE_PENALTIES):
            coefficients = solve_ridge(gram, cross_product, float(penalty))
            prediction = (
                location[None, :] + scale[None, :] * (x_validation @ coefficients)
            )
            residual = (validation_values - prediction) / scale[None, :]
            valid = validation_observed & np.isfinite(residual)
            clipped = np.clip(
                residual[valid], -MEAN_MODEL.residual_cap, MEAN_MODEL.residual_cap
            )
            squared_error[penalty_index] += float(np.square(clipped).sum())
            observation_count[penalty_index] += int(len(clipped))

    return squared_error, observation_count, fitted_targets


def predict_one_fold(
    design: np.ndarray,
    targets: np.ndarray,
    training_mask: np.ndarray,
    validation_mask: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw predictions and fold-training mean baselines for validation rows."""
    training_rows = np.flatnonzero(training_mask)
    validation_rows = np.flatnonzero(validation_mask)
    x_train = augmented_design(design[training_rows])
    x_validation = augmented_design(design[validation_rows])
    y_train = np.asarray(targets[training_rows], dtype=np.float64)
    observed_train = np.isfinite(y_train)
    prediction = np.full((len(validation_rows), targets.shape[1]), np.nan, dtype=np.float64)
    baseline = np.full_like(prediction, np.nan)
    fitted = np.zeros(targets.shape[1], dtype=bool)
    minimum_rows = max(MINIMUM_TARGET_ROWS, x_train.shape[1] + 10)

    for target_indices, group_observed in group_targets_by_observation_mask(observed_train):
        if int(group_observed.sum()) < minimum_rows:
            continue
        x_observed = x_train[group_observed]
        y_observed = y_train[group_observed][:, target_indices]
        location, scale, valid_target = robust_target_parameters(y_observed)
        if not valid_target.any():
            continue
        target_indices = target_indices[valid_target]
        location = location[valid_target]
        scale = scale[valid_target]
        y_valid = y_observed[:, valid_target]
        y_standardized = (y_valid - location[None, :]) / scale[None, :]
        gram = x_observed.T @ x_observed
        cross_product = x_observed.T @ y_standardized
        coefficients = solve_ridge(gram, cross_product, penalty)
        prediction[:, target_indices] = (
            location[None, :] + scale[None, :] * (x_validation @ coefficients)
        )
        baseline[:, target_indices] = np.mean(y_valid, axis=0)[None, :]
        fitted[target_indices] = True
    return prediction, baseline, fitted


def fit_final_masked_ridge(
    design: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    fit_rows = np.flatnonzero(fit_mask)
    x_fit = augmented_design(design[fit_rows])
    y_fit = np.asarray(targets[fit_rows], dtype=np.float64)
    observed = np.isfinite(y_fit)
    coefficients = np.full(
        (x_fit.shape[1], targets.shape[1]), np.nan, dtype=np.float64
    )
    target_location = np.full(targets.shape[1], np.nan, dtype=np.float64)
    target_scale = np.full(targets.shape[1], np.nan, dtype=np.float64)
    target_count = observed.sum(axis=0, dtype=np.int64)
    modeled = np.zeros(targets.shape[1], dtype=bool)
    minimum_rows = max(MINIMUM_TARGET_ROWS, x_fit.shape[1] + 10)

    for target_indices, group_observed in group_targets_by_observation_mask(observed):
        if int(group_observed.sum()) < minimum_rows:
            continue
        x_observed = x_fit[group_observed]
        y_observed = y_fit[group_observed][:, target_indices]
        location, scale, valid_target = robust_target_parameters(y_observed)
        if not valid_target.any():
            continue
        target_indices = target_indices[valid_target]
        location = location[valid_target]
        scale = scale[valid_target]
        y_standardized = (
            y_observed[:, valid_target] - location[None, :]
        ) / scale[None, :]
        gram = x_observed.T @ x_observed
        cross_product = x_observed.T @ y_standardized
        coefficients[:, target_indices] = solve_ridge(gram, cross_product, penalty)
        target_location[target_indices] = location
        target_scale[target_indices] = scale
        modeled[target_indices] = True

    return {
        "coefficients": coefficients,
        "target_location": target_location,
        "target_scale": target_scale,
        "target_training_count": target_count,
        "target_modeled": modeled,
    }


# =============================================================================
# 6. Per-case cross-fitting and final model artifacts
# =============================================================================

def case_artifact_paths(farm: str, event_id: int) -> tuple[Path, Path]:
    relative = Path(safe_slug(farm)) / f"event_{int(event_id):03d}"
    return (
        CELL4_MODEL_ROOT / relative.with_suffix(".json"),
        CELL4_RESIDUAL_ROOT / relative.with_suffix(".npz"),
    )


def case_input_signature(case_row: Any) -> str:
    return sha256_json(
        {
            "cell3_receipt_sha256": CELL3_RECEIPT_SHA256,
            "case_key": str(case_row.case_key),
            "cell3_cache_sha256": str(case_row.cache_sha256),
            "cell3_case_preprocessing_sha256": str(case_row.case_preprocessing_sha256),
            "mean_model_implementation_sha256": MEAN_MODEL_IMPLEMENTATION_SHA256,
        }
    )


def serialize_design_state(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = {
        "driver_names": list(state["driver_names"]),
        "term_kind": list(state["term_kind"]),
        "term_names": list(state["term_names"]),
        "core_driver_names": list(state["core_driver_names"]),
    }
    arrays = {
        "design_imputation": np.asarray(state["imputation"], dtype=np.float64),
        "design_raw_location": np.asarray(state["raw_location"], dtype=np.float64),
        "design_raw_scale": np.asarray(state["raw_scale"], dtype=np.float64),
        "design_is_indicator": np.asarray(state["is_indicator"], dtype=bool),
        "design_term_left": np.asarray(state["term_left"], dtype=np.int32),
        "design_term_right": np.asarray(state["term_right"], dtype=np.int32),
        "design_term_location": np.asarray(state["term_location"], dtype=np.float64),
        "design_term_scale": np.asarray(state["term_scale"], dtype=np.float64),
        "design_term_active": np.asarray(state["term_active"], dtype=bool),
    }
    return metadata, arrays


def restore_design_state(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "driver_names": list(metadata["driver_names"]),
        "imputation": arrays["design_imputation"],
        "raw_location": arrays["design_raw_location"],
        "raw_scale": arrays["design_raw_scale"],
        "is_indicator": arrays["design_is_indicator"],
        "term_kind": list(metadata["term_kind"]),
        "term_left": arrays["design_term_left"],
        "term_right": arrays["design_term_right"],
        "term_names": list(metadata["term_names"]),
        "term_location": arrays["design_term_location"],
        "term_scale": arrays["design_term_scale"],
        "term_active": arrays["design_term_active"],
        "core_driver_names": list(metadata["core_driver_names"]),
    }


def existing_case_summary(
    metadata_path: Path,
    artifact_path: Path,
    expected_input_signature: str,
) -> dict[str, Any] | None:
    if not metadata_path.exists() or not artifact_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("case_input_signature") != expected_input_signature:
        raise RuntimeError(
            f"Mean-model input drift for {metadata.get('case_key', metadata_path.stem)}. "
            "Do not overwrite the existing artifact."
        )
    observed_hash = file_sha256(artifact_path)
    if metadata.get("artifact_sha256") != observed_hash:
        raise RuntimeError(f"Mean-model artifact hash failed: {artifact_path}")
    return dict(metadata["summary"])


def fit_one_case_mean_model(case_row: Any) -> dict[str, Any]:
    metadata_path, artifact_path = case_artifact_paths(case_row.farm, case_row.event_id)
    input_signature = case_input_signature(case_row)
    existing = existing_case_summary(metadata_path, artifact_path, input_signature)
    if existing is not None:
        return existing

    cache = load_case_cache(str(case_row.case_key))
    cache_metadata = cache["metadata"]
    if cache_metadata.get("outcome_fields_present") is not False:
        raise RuntimeError(f"Unsafe Cell 3 cache for {case_row.case_key}.")

    drivers = np.asarray(cache["drivers"], dtype=np.float64)
    targets = np.asarray(cache["targets"], dtype=np.float64)
    fit_mask = np.asarray(cache["fit_mask"], dtype=bool)
    segment_id = np.asarray(cache["segment_id"], dtype=np.int32)
    driver_names = list(cache_metadata["driver_names"])
    target_names = list(cache_metadata["target_names"])
    target_temperature = np.asarray(cache["target_temperature"], dtype=bool)

    if len(drivers) != len(targets) or len(fit_mask) != len(drivers):
        raise RuntimeError(f"Row mismatch in Cell 3 cache for {case_row.case_key}.")
    if drivers.shape[1] != len(driver_names) or targets.shape[1] != len(target_names):
        raise RuntimeError(f"Channel-name mismatch for {case_row.case_key}.")
    if not np.isfinite(drivers).all():
        raise RuntimeError(f"Non-finite driver values remain for {case_row.case_key}.")

    folds = list(iter_case_temporal_folds(fit_mask, segment_id))
    penalty_sse = np.zeros(len(RIDGE_PENALTIES), dtype=np.float64)
    penalty_count = np.zeros(len(RIDGE_PENALTIES), dtype=np.int64)
    fold_diagnostics: list[dict[str, Any]] = []

    # Pass 1: label-free ridge-penalty selection.
    for fold in folds:
        state, design = fit_design_state(
            drivers, driver_names, fold["training_mask"], str(case_row.farm)
        )
        fold_sse, fold_count, fitted_targets = score_penalties_for_fold(
            design,
            targets,
            fold["training_mask"],
            fold["validation_mask"],
        )
        penalty_sse += fold_sse
        penalty_count += fold_count
        fold_diagnostics.append(
            {
                "fold": int(fold["fold_number"]),
                "training_rows": int(fold["training_mask"].sum()),
                "validation_rows": int(fold["validation_mask"].sum()),
                "embargoed_fit_rows": int(fold["embargoed_fit_rows"]),
                "design_terms": int(design.shape[1]),
                "fitted_targets": int(fitted_targets),
            }
        )
        del state, design

    if (penalty_count == 0).any():
        raise RuntimeError(f"No OOF observations for one or more penalties in {case_row.case_key}.")
    penalty_mse = penalty_sse / penalty_count
    minimum_mse = float(np.min(penalty_mse))
    tied = np.flatnonzero(np.isclose(penalty_mse, minimum_mse, rtol=1e-12, atol=1e-12))
    selected_index = int(tied[-1])
    selected_penalty = float(RIDGE_PENALTIES[selected_index])

    # Pass 2: honest OOF predictions and residuals at the selected penalty.
    oof_prediction = np.full_like(targets, np.nan, dtype=np.float32)
    oof_baseline = np.full_like(targets, np.nan, dtype=np.float32)
    oof_fold_id = np.full(len(targets), -1, dtype=np.int8)
    fold_target_fit_count = np.zeros(targets.shape[1], dtype=np.int16)

    for fold in folds:
        state, design = fit_design_state(
            drivers, driver_names, fold["training_mask"], str(case_row.farm)
        )
        prediction, baseline, fitted_targets = predict_one_fold(
            design,
            targets,
            fold["training_mask"],
            fold["validation_mask"],
            selected_penalty,
        )
        validation_rows = np.flatnonzero(fold["validation_mask"])
        oof_prediction[validation_rows] = prediction.astype(np.float32)
        oof_baseline[validation_rows] = baseline.astype(np.float32)
        oof_fold_id[validation_rows] = int(fold["fold"])
        fold_target_fit_count += fitted_targets.astype(np.int16)
        del state, design, prediction, baseline

    oof_residual = targets - oof_prediction.astype(np.float64)
    observed_oof = fit_mask[:, None] & np.isfinite(targets) & np.isfinite(oof_prediction)
    baseline_valid = observed_oof & np.isfinite(oof_baseline)
    target_oof_count = observed_oof.sum(axis=0, dtype=np.int64)
    target_fit_observed = (fit_mask[:, None] & np.isfinite(targets)).sum(
        axis=0, dtype=np.int64
    )
    target_oof_coverage = target_oof_count / np.maximum(target_fit_observed, 1)
    target_crossfit_r2 = np.full(targets.shape[1], np.nan, dtype=np.float64)

    for target_index in range(targets.shape[1]):
        valid = observed_oof[:, target_index]
        valid_baseline = baseline_valid[:, target_index]
        if valid.sum() < MINIMUM_TARGET_ROWS or valid_baseline.sum() != valid.sum():
            continue
        model_sse = float(np.square(oof_residual[valid, target_index]).sum())
        baseline_error = (
            targets[valid, target_index]
            - oof_baseline[valid, target_index].astype(np.float64)
        )
        baseline_sse = float(np.square(baseline_error).sum())
        if baseline_sse > NUMERICAL_EPSILON:
            target_crossfit_r2[target_index] = 1.0 - model_sse / baseline_sse

    # Predictions and baselines have served their sole purpose. Releasing them
    # before the final fit keeps the Farm C peak memory bounded.
    del oof_prediction, oof_baseline

    # Final label-free model on every normal source-training row.
    final_state, final_design = fit_design_state(
        drivers, driver_names, fit_mask, str(case_row.farm)
    )
    final_model = fit_final_masked_ridge(
        final_design, targets, fit_mask, selected_penalty
    )
    state_metadata, state_arrays = serialize_design_state(final_state)

    fit_rows = np.flatnonzero(fit_mask).astype(np.int32)
    residual_fit = oof_residual[fit_mask].astype(np.float32)
    residual_fold_fit = oof_fold_id[fit_mask].astype(np.int8)
    if (residual_fold_fit < 0).any():
        raise RuntimeError(f"OOF fold assignment is incomplete for {case_row.case_key}.")

    artifact_arrays = {
        **state_arrays,
        "coefficients": final_model["coefficients"].astype(np.float32),
        "target_location": final_model["target_location"].astype(np.float64),
        "target_scale": final_model["target_scale"].astype(np.float64),
        "target_training_count": final_model["target_training_count"].astype(np.int64),
        "target_modeled": final_model["target_modeled"].astype(bool),
        "target_crossfit_r2": target_crossfit_r2.astype(np.float64),
        "target_oof_coverage": target_oof_coverage.astype(np.float64),
        "target_fold_fit_count": fold_target_fit_count.astype(np.int16),
        "target_temperature": target_temperature.astype(bool),
        "fit_row_indices": fit_rows,
        "oof_fold_fit": residual_fold_fit,
        "oof_residual_fit": residual_fit,
        "ridge_penalties": RIDGE_PENALTIES.astype(np.float64),
        "ridge_selection_mse": penalty_mse.astype(np.float64),
        "ridge_selection_count": penalty_count.astype(np.int64),
    }
    atomic_save_npz(artifact_path, **artifact_arrays)
    artifact_hash = file_sha256(artifact_path)

    finite_r2 = target_crossfit_r2[np.isfinite(target_crossfit_r2)]
    temperature_r2 = target_crossfit_r2[target_temperature & np.isfinite(target_crossfit_r2)]
    summary = {
        "case_key": str(case_row.case_key),
        "farm": str(case_row.farm),
        "asset_id": str(case_row.asset_id),
        "event_id": int(case_row.event_id),
        "rows": int(len(drivers)),
        "training_normal_rows": int(fit_mask.sum()),
        "drivers": int(drivers.shape[1]),
        "targets": int(targets.shape[1]),
        "final_design_terms": int(final_design.shape[1]),
        "selected_penalty": selected_penalty,
        "selection_mse": float(penalty_mse[selected_index]),
        "modeled_targets": int(final_model["target_modeled"].sum()),
        "complete_crossfit_targets": int((fold_target_fit_count == MEAN_MODEL.crossfit_folds).sum()),
        "median_crossfit_r2": float(np.median(finite_r2)) if len(finite_r2) else np.nan,
        "median_temperature_crossfit_r2": (
            float(np.median(temperature_r2)) if len(temperature_r2) else np.nan
        ),
        "fraction_crossfit_r2_positive": (
            float((finite_r2 > 0.0).mean()) if len(finite_r2) else np.nan
        ),
        "median_oof_coverage": float(np.median(target_oof_coverage)),
        "artifact_relative_path": artifact_path.relative_to(CELL4_RESIDUAL_ROOT).as_posix(),
        "metadata_relative_path": metadata_path.relative_to(CELL4_MODEL_ROOT).as_posix(),
        "case_input_signature": input_signature,
        "artifact_sha256": artifact_hash,
    }

    save_json(
        {
            "cell4_version": CELL4_VERSION,
            "created_at_utc": utc_now(),
            "contract_sha256": CONTRACT_SHA256,
            "cell3_receipt_sha256": CELL3_RECEIPT_SHA256,
            "mean_model_implementation_sha256": MEAN_MODEL_IMPLEMENTATION_SHA256,
            "case_input_signature": input_signature,
            "case_key": str(case_row.case_key),
            "farm": str(case_row.farm),
            "asset_id": str(case_row.asset_id),
            "event_id": int(case_row.event_id),
            "driver_names": driver_names,
            "target_names": target_names,
            "design_state": state_metadata,
            "fold_diagnostics": fold_diagnostics,
            "selected_penalty": selected_penalty,
            "ridge_penalties": RIDGE_PENALTIES.tolist(),
            "ridge_selection_mse": penalty_mse.tolist(),
            "artifact_sha256": artifact_hash,
            "artifact_relative_path": summary["artifact_relative_path"],
            "outcome_fields_present": False,
            "summary": summary,
        },
        metadata_path,
    )
    return summary


# =============================================================================
# 7. Safe model loading and reproducible mean prediction
# =============================================================================

def load_case_mean_model(case_key: str) -> dict[str, Any]:
    match = CASE_MEAN_MODEL_REGISTRY.loc[
        CASE_MEAN_MODEL_REGISTRY["case_key"].eq(case_key)
    ]
    if len(match) != 1:
        raise KeyError(f"Unknown or duplicate case_key: {case_key}")
    row = match.iloc[0]
    metadata_path = CELL4_MODEL_ROOT / row["metadata_relative_path"]
    artifact_path = CELL4_RESIDUAL_ROOT / row["artifact_relative_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("outcome_fields_present") is not False:
        raise RuntimeError(f"Unsafe mean-model metadata for {case_key}.")
    if file_sha256(artifact_path) != row["artifact_sha256"]:
        raise RuntimeError(f"Mean-model artifact hash mismatch for {case_key}.")
    arrays = dict(np.load(artifact_path, allow_pickle=False))
    arrays["metadata"] = metadata
    arrays["design_state"] = restore_design_state(metadata["design_state"], arrays)
    return arrays


def predict_case_mean(
    case_key: str,
    row_selector: slice | np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Reproduce the final normal-behaviour mean without outcome information."""
    cache = load_case_cache(case_key)
    model = load_case_mean_model(case_key)
    drivers = np.asarray(cache["drivers"], dtype=np.float64)
    if row_selector is not None:
        drivers = drivers[row_selector]
    design = transform_design(drivers, model["design_state"])
    x = augmented_design(design)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    target_location = np.asarray(model["target_location"], dtype=np.float64)
    target_scale = np.asarray(model["target_scale"], dtype=np.float64)
    prediction = target_location[None, :] + target_scale[None, :] * (x @ coefficients)
    prediction[:, ~np.asarray(model["target_modeled"], dtype=bool)] = np.nan
    return prediction.astype(np.float32)


# =============================================================================
# 8. Fit or verify all 95 cases
# =============================================================================

_model_summaries: list[dict[str, Any]] = []
for _farm in DATASET.farms:
    _farm_cases = CASE_CACHE_REGISTRY.loc[CASE_CACHE_REGISTRY["farm"].eq(_farm)]
    print(
        f"Cross-fitting nonlinear ridge mean models — {_farm}: {len(_farm_cases)} cases",
        flush=True,
    )
    for _case_row in _farm_cases.itertuples(index=False):
        _summary = fit_one_case_mean_model(_case_row)
        _model_summaries.append(_summary)
        print(
            f"  event {_case_row.event_id:>3}: λ={_summary['selected_penalty']:<8g} "
            f"targets={_summary['modeled_targets']:>3}/{_summary['targets']:<3} "
            f"median OOF R²={_summary['median_crossfit_r2']:.3f}",
            flush=True,
        )

CASE_MEAN_MODEL_REGISTRY = pd.DataFrame(_model_summaries).sort_values(
    ["farm", "event_id"], kind="stable"
).reset_index(drop=True)

if len(CASE_MEAN_MODEL_REGISTRY) != DATASET.expected_total_cases:
    raise RuntimeError("Not all 95 cases produced a mean-model artifact.")
if CASE_MEAN_MODEL_REGISTRY["case_key"].duplicated().any():
    raise RuntimeError("Duplicate case keys exist in the mean-model registry.")
if (CASE_MEAN_MODEL_REGISTRY["modeled_targets"] <= 0).any():
    raise RuntimeError("At least one case has no fitted monitoring target.")

# Load one artifact per farm and reproduce a small prediction slice.
for _case_key in CASE_MEAN_MODEL_REGISTRY.groupby("farm", sort=False).head(1)["case_key"]:
    _model = load_case_mean_model(_case_key)
    _prediction = predict_case_mean(_case_key, slice(0, 32))
    if _prediction.shape[0] != 32:
        raise RuntimeError(f"Prediction smoke test failed for {_case_key}.")
    if _prediction.shape[1] != len(_model["metadata"]["target_names"]):
        raise RuntimeError(f"Target shape mismatch for {_case_key}.")
    del _model, _prediction


# =============================================================================
# 9. Freeze Cell 4 receipt and report modelability diagnostics
# =============================================================================

MEAN_MODEL_FARM_SUMMARY = (
    CASE_MEAN_MODEL_REGISTRY.groupby("farm", sort=False)
    .agg(
        cases=("case_key", "size"),
        median_selected_penalty=("selected_penalty", "median"),
        median_design_terms=("final_design_terms", "median"),
        median_modeled_targets=("modeled_targets", "median"),
        median_crossfit_r2=("median_crossfit_r2", "median"),
        median_temperature_crossfit_r2=("median_temperature_crossfit_r2", "median"),
        median_positive_r2_fraction=("fraction_crossfit_r2_positive", "median"),
        median_oof_coverage=("median_oof_coverage", "median"),
    )
    .reset_index()
)

RIDGE_SELECTION_SUMMARY = (
    CASE_MEAN_MODEL_REGISTRY.groupby(["farm", "selected_penalty"])
    .size()
    .rename("cases")
    .reset_index()
    .sort_values(["farm", "selected_penalty"], kind="stable")
)

_component_hashes = {
    "case_mean_model_registry_sha256": dataframe_sha256(
        CASE_MEAN_MODEL_REGISTRY, ("farm", "event_id")
    ),
    "farm_summary_sha256": dataframe_sha256(
        MEAN_MODEL_FARM_SUMMARY, ("farm",)
    ),
    "ridge_selection_summary_sha256": dataframe_sha256(
        RIDGE_SELECTION_SUMMARY, ("farm", "selected_penalty")
    ),
}
CELL4_RECEIPT_SHA256 = sha256_json(
    {
        "contract_sha256": CONTRACT_SHA256,
        "cell3_receipt_sha256": CELL3_RECEIPT_SHA256,
        "mean_model_implementation_sha256": MEAN_MODEL_IMPLEMENTATION_SHA256,
        "component_hashes": _component_hashes,
    }
)

CELL4_RECEIPT_PATH = CELL4_QUALITY_ROOT / "cell4_cross_fitted_mean_model_receipt.json"
if CELL4_RECEIPT_PATH.exists():
    _existing_receipt = json.loads(CELL4_RECEIPT_PATH.read_text(encoding="utf-8"))
    if _existing_receipt.get("cell4_receipt_sha256") != CELL4_RECEIPT_SHA256:
        raise RuntimeError(
            "A different Cell 4 receipt exists for this experiment. Do not overwrite it."
        )
    CELL4_STATE = "existing identical Cell 4 receipt verified"
    _write_cell4_receipt = False
else:
    CELL4_STATE = "new Cell 4 receipt frozen"
    _write_cell4_receipt = True

save_csv_atomic(
    CASE_MEAN_MODEL_REGISTRY,
    CELL4_QUALITY_ROOT / "case_mean_model_registry.csv",
)
save_csv_atomic(
    MEAN_MODEL_FARM_SUMMARY,
    CELL4_QUALITY_ROOT / "mean_model_farm_summary.csv",
)
save_csv_atomic(
    RIDGE_SELECTION_SUMMARY,
    CELL4_QUALITY_ROOT / "ridge_selection_summary.csv",
)
save_json(
    MEAN_MODEL_IMPLEMENTATION,
    CELL4_QUALITY_ROOT / "mean_model_implementation.json",
)

if _write_cell4_receipt:
    save_json(
        {
            "cell4_version": CELL4_VERSION,
            "created_at_utc": utc_now(),
            "contract_sha256": CONTRACT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "cell2_audit_sha256": CELL2_AUDIT_SHA256,
            "cell3_receipt_sha256": CELL3_RECEIPT_SHA256,
            "preprocessing_policy_sha256": PREPROCESSING_POLICY_SHA256,
            "mean_model_implementation_sha256": MEAN_MODEL_IMPLEMENTATION_SHA256,
            "component_hashes": _component_hashes,
            "cell4_receipt_sha256": CELL4_RECEIPT_SHA256,
            "outcomes_read": False,
        },
        CELL4_RECEIPT_PATH,
    )


print("\n" + "=" * 92)
print("UC-RCF-NBM CELL 4 — CROSS-FITTED NONLINEAR RIDGE MEAN MODEL")
print("=" * 92)
print("\nMEAN-MODEL PERFORMANCE ON HONEST TRAINING-NORMAL OOF RESIDUALS")
display(MEAN_MODEL_FARM_SUMMARY)
print("\nRIDGE PENALTIES SELECTED BY MASKED OOF MSE")
display(RIDGE_SELECTION_SUMMARY)

print("\n" + "-" * 92)
print(f"Cases modeled                    : {len(CASE_MEAN_MODEL_REGISTRY)}")
print(f"Temporal folds per case          : {MEAN_MODEL.crossfit_folds}")
print(f"Temporal block / embargo         : {MEAN_MODEL.temporal_block_steps} / "
      f"{MEAN_MODEL.embargo_steps} steps")
print(f"Mean implementation SHA-256      : {MEAN_MODEL_IMPLEMENTATION_SHA256}")
print(f"Cell 4 receipt SHA-256           : {CELL4_RECEIPT_SHA256}")
print(f"Cell 4 state                     : {CELL4_STATE}")
print("Fold-local preprocessing         : Yes")
print("Masked targets imputed           : No")
print("Event outcomes accessed          : No")
print("Final prediction rows used in fit: No")
print("OOF/model artifact checks        : PASS")
print("=" * 92)
print("CELL 4 COMPLETED SUCCESSFULLY — CROSS-FITTED MEAN MODELS LOCKED")
