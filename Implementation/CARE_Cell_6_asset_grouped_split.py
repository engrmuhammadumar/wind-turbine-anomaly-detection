# ============================================================================
# CELL 6 — DETERMINISTIC FARM-AWARE, ASSET-GROUPED DATA SPLIT
#
# Prerequisites from the successful Cell 5:
#   - case_registry
#   - canonical_asset_registry
#   - canonical_asset_summary
#   - modeling_eligibility_registry
#   - OUTPUT_ROOT
#   - TABLE_DIR
#
# This cell:
#   - assigns each canonical asset wholly to train, validation, or test
#   - targets a 70% / 15% / 15% case split
#   - balances farms and anomaly/normal cases at the asset-group level
#   - requires every farm and both classes in every split
#   - proves that canonical assets do not overlap across splits
#   - creates reusable case and asset split registries
#
# This cell DOES NOT read sensor-file bodies, use the raw train_test field as
# the modeling split, impute, scale, clip, filter, select features, create
# windows, fit preprocessing, or train a model.
# ============================================================================


# ----------------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# ----------------------------------------------------------------------------
# 2. Prerequisites, paths, and fixed split policy
# ----------------------------------------------------------------------------

REQUIRED_CELL_6_OBJECTS = (
    "case_registry",
    "canonical_asset_registry",
    "canonical_asset_summary",
    "modeling_eligibility_registry",
    "OUTPUT_ROOT",
)

missing_cell_6_objects = [
    object_name
    for object_name in REQUIRED_CELL_6_OBJECTS
    if object_name not in globals()
]

if missing_cell_6_objects:
    raise RuntimeError(
        "Run the successful Cell 5 before Cell 6. Missing objects: "
        + ", ".join(missing_cell_6_objects)
    )

OUTPUT_ROOT = Path(OUTPUT_ROOT)

if "TABLE_DIR" not in globals():
    TABLE_DIR = OUTPUT_ROOT / "tables"
else:
    TABLE_DIR = Path(TABLE_DIR)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL_CASES = 95
EXPECTED_ANOMALY_CASES = 45
EXPECTED_NORMAL_CASES = 50
EXPECTED_CANONICAL_ASSETS = 36

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

# The optimizer is deterministic for this seed. Increasing the candidate count
# can improve the balance search without changing the split policy.
CELL_6_RANDOM_SEED = 13
CELL_6_SEARCH_CANDIDATES = 50_000

ELIGIBILITY_COLUMN = (
    "structurally_eligible_for_split_and_train_only_preprocessing"
)

CELL_6_POLICY = {
    "split_names": list(SPLIT_NAMES),
    "target_case_ratios": SPLIT_RATIOS,
    "grouping_unit": "farm-qualified canonical asset key",
    "asset_disjoint": True,
    "farm_aware": True,
    "both_classes_required_in_every_split": True,
    "both_classes_required_in_every_farm_split": True,
    "raw_train_test_field_used_for_assignment": False,
    "eligibility_required": ELIGIBILITY_COLUMN,
    "random_seed": CELL_6_RANDOM_SEED,
    "candidate_assignments_evaluated": CELL_6_SEARCH_CANDIDATES,
    "transformations_applied": [],
    "sensor_file_bodies_read": False,
    "source_data_modified": False,
}


# ----------------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------------

def cell_6_json_safe(value: Any) -> Any:
    """Recursively convert scientific-Python values for strict JSON."""

    if isinstance(value, dict):
        return {
            str(key): cell_6_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            cell_6_json_safe(item)
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


def save_cell_6_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write a strict, human-readable JSON manifest."""

    with Path(destination).open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            cell_6_json_safe(payload),
            file_handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def cell_6_integer_targets(
    total_count: int,
) -> np.ndarray:
    """Allocate an integer total by largest remainder in split order."""

    ratios = np.array(
        [SPLIT_RATIOS[name] for name in SPLIT_NAMES],
        dtype=float,
    )
    raw_targets = float(total_count) * ratios
    integer_targets = np.floor(raw_targets).astype(int)
    remaining = int(total_count - integer_targets.sum())

    if remaining > 0:
        remainder_order = np.argsort(
            -(raw_targets - integer_targets),
            kind="stable",
        )
        integer_targets[remainder_order[:remaining]] += 1

    return integer_targets


def cell_6_asset_quotas(
    asset_count: int,
) -> np.ndarray:
    """Allocate one farm's assets while guaranteeing all three splits."""

    if int(asset_count) < len(SPLIT_NAMES):
        raise RuntimeError(
            f"A farm has only {asset_count} canonical assets; at least "
            f"{len(SPLIT_NAMES)} are required for a three-way grouped split."
        )

    ratios = np.array(
        [SPLIT_RATIOS[name] for name in SPLIT_NAMES],
        dtype=float,
    )
    raw_quotas = float(asset_count) * ratios
    quotas = np.floor(raw_quotas).astype(int)
    quotas = np.maximum(quotas, 1)

    while int(quotas.sum()) > int(asset_count):
        removable = np.flatnonzero(quotas > 1)

        if removable.size == 0:
            raise RuntimeError(
                "Could not construct per-farm asset quotas."
            )

        surplus = quotas[removable] - raw_quotas[removable]
        remove_index = removable[
            int(np.argmax(surplus))
        ]
        quotas[remove_index] -= 1

    while int(quotas.sum()) < int(asset_count):
        deficits = raw_quotas - quotas
        add_index = int(np.argmax(deficits))
        quotas[add_index] += 1

    if (quotas < 1).any() or int(quotas.sum()) != int(asset_count):
        raise RuntimeError(
            "Per-farm asset-quota conservation failed."
        )

    return quotas


def cell_6_assignment_digest(
    assignment_table: pd.DataFrame,
) -> str:
    """Create a stable SHA-256 fingerprint of asset-to-split membership."""

    digest_lines = (
        assignment_table[
            [
                "canonical_asset_key",
                "model_split",
            ]
        ]
        .sort_values("canonical_asset_key")
        .astype("string")
        .agg("\t".join, axis=1)
        .tolist()
    )
    digest_payload = ("\n".join(digest_lines) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(digest_payload).hexdigest()


def cell_6_optimize_asset_assignment(
    asset_table: pd.DataFrame,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """
    Search deterministic farm-stratified asset assignments.

    Candidate quality is evaluated lexicographically:
      1. exact global integer case targets
      2. smallest maximum split-size deviation
      3. symmetric validation/test class counts
      4. global class-rate balance
      5. farm-level case and class balance
      6. stable assignment tie-break
    """

    farm_names = tuple(
        sorted(asset_table["farm"].astype(str).unique())
    )
    farm_to_code = {
        farm_name: farm_code
        for farm_code, farm_name in enumerate(farm_names)
    }
    farm_codes = (
        asset_table["farm"]
        .astype(str)
        .map(farm_to_code)
        .to_numpy(dtype=np.int16)
    )
    farm_indices = [
        np.flatnonzero(farm_codes == farm_code)
        for farm_code in range(len(farm_names))
    ]
    farm_quotas = [
        cell_6_asset_quotas(len(indices))
        for indices in farm_indices
    ]

    # Axis 2 contains case count, anomaly cases, and normal cases.
    asset_metrics = asset_table[
        [
            "case_count",
            "anomaly_cases",
            "normal_cases",
        ]
    ].to_numpy(dtype=np.int64)

    target_case_counts = cell_6_integer_targets(
        int(asset_metrics[:, 0].sum())
    )
    total_anomaly_cases = int(asset_metrics[:, 1].sum())
    total_normal_cases = int(asset_metrics[:, 2].sum())
    overall_anomaly_rate = (
        total_anomaly_cases
        / (total_anomaly_cases + total_normal_cases)
    )
    target_ratios = np.array(
        [SPLIT_RATIOS[name] for name in SPLIT_NAMES],
        dtype=float,
    )

    random_generator = np.random.default_rng(
        CELL_6_RANDOM_SEED
    )
    best_assignment: np.ndarray | None = None
    best_score: tuple[Any, ...] | None = None
    best_cube: np.ndarray | None = None
    valid_candidate_count = 0

    for _ in range(CELL_6_SEARCH_CANDIDATES):
        assignment = np.full(
            len(asset_table),
            -1,
            dtype=np.int8,
        )

        for indices, quotas in zip(
            farm_indices,
            farm_quotas,
        ):
            shuffled_indices = random_generator.permutation(
                indices
            )
            start_position = 0

            for split_code, quota in enumerate(quotas):
                stop_position = start_position + int(quota)
                assignment[
                    shuffled_indices[
                        start_position:stop_position
                    ]
                ] = split_code
                start_position = stop_position

        if (assignment < 0).any():
            raise RuntimeError(
                "The candidate generator left an asset unassigned."
            )

        farm_split_cube = np.zeros(
            (
                len(farm_names),
                len(SPLIT_NAMES),
                3,
            ),
            dtype=np.int64,
        )

        for farm_code, indices in enumerate(farm_indices):
            for split_code in range(len(SPLIT_NAMES)):
                selected_indices = indices[
                    assignment[indices] == split_code
                ]

                if selected_indices.size > 0:
                    farm_split_cube[
                        farm_code,
                        split_code,
                    ] = asset_metrics[selected_indices].sum(axis=0)

        # Every farm/split cell must contain anomaly and normal cases.
        if (
            (farm_split_cube[:, :, 1] < 1).any()
            or (farm_split_cube[:, :, 2] < 1).any()
        ):
            continue

        valid_candidate_count += 1
        global_split_metrics = farm_split_cube.sum(axis=0)
        split_case_counts = global_split_metrics[:, 0]
        case_deviations = np.abs(
            split_case_counts - target_case_counts
        )
        split_anomaly_rates = (
            global_split_metrics[:, 1]
            / split_case_counts
        )
        class_rate_score = float(
            np.square(
                split_anomaly_rates - overall_anomaly_rate
            ).sum()
        )

        validation_test_class_asymmetry = int(
            np.abs(
                global_split_metrics[1, 1:]
                - global_split_metrics[2, 1:]
            ).sum()
        )

        farm_balance_score = 0.0

        for farm_code in range(len(farm_names)):
            for metric_index in range(3):
                farm_metric_total = float(
                    farm_split_cube[
                        farm_code,
                        :,
                        metric_index,
                    ].sum()
                )
                expected_values = (
                    farm_metric_total * target_ratios
                )
                scale = np.maximum(expected_values, 1.0)
                deviations = (
                    farm_split_cube[
                        farm_code,
                        :,
                        metric_index,
                    ]
                    - expected_values
                ) / scale
                metric_weight = (
                    2.0 if metric_index == 0 else 1.0
                )
                farm_balance_score += metric_weight * float(
                    np.square(deviations).sum()
                )

        score = (
            int(case_deviations.sum()),
            int(case_deviations.max()),
            validation_test_class_asymmetry,
            round(class_rate_score, 14),
            round(farm_balance_score, 14),
            tuple(int(value) for value in assignment),
        )

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_cube = farm_split_cube.copy()

    if best_assignment is None or best_cube is None:
        raise RuntimeError(
            "No valid asset-grouped split was found. The required farm/class "
            "constraints could not be satisfied."
        )

    achieved_case_counts = best_cube.sum(axis=0)[:, 0]

    if not np.array_equal(
        achieved_case_counts,
        target_case_counts,
    ):
        raise RuntimeError(
            "The deterministic search did not reach the exact integer case "
            f"targets {target_case_counts.tolist()}; it reached "
            f"{achieved_case_counts.tolist()}. Increase "
            "CELL_6_SEARCH_CANDIDATES before accepting a split."
        )

    search_details = {
        "farm_names": farm_names,
        "farm_asset_quotas": {
            farm_name: {
                split_name: int(quota)
                for split_name, quota in zip(
                    SPLIT_NAMES,
                    quotas,
                )
            }
            for farm_name, quotas in zip(
                farm_names,
                farm_quotas,
            )
        },
        "target_case_counts": {
            split_name: int(target)
            for split_name, target in zip(
                SPLIT_NAMES,
                target_case_counts,
            )
        },
        "valid_candidate_count": int(valid_candidate_count),
        "best_score_components": {
            "total_absolute_case_target_deviation": int(
                best_score[0]
            ),
            "maximum_case_target_deviation": int(best_score[1]),
            "validation_test_class_asymmetry": int(best_score[2]),
            "global_class_rate_score": float(best_score[3]),
            "farm_balance_score": float(best_score[4]),
        },
    }

    return best_assignment, search_details


# ----------------------------------------------------------------------------
# 4. Validate Cell 5 tables and reconstruct the asset grouping table
# ----------------------------------------------------------------------------

cell_6_validation_errors: list[str] = []
cell_6_validation_warnings: list[str] = []

required_eligibility_columns = {
    "farm",
    "event_id",
    "event_type",
    "is_anomaly",
    "canonical_asset_id",
    "canonical_asset_key",
    "canonical_asset_source",
    "asset_grouping_eligible",
    ELIGIBILITY_COLUMN,
}

missing_eligibility_columns = sorted(
    required_eligibility_columns
    - set(modeling_eligibility_registry.columns)
)

if missing_eligibility_columns:
    raise RuntimeError(
        "Cell 5 modeling_eligibility_registry lacks required columns: "
        + ", ".join(missing_eligibility_columns)
    )

required_asset_registry_columns = {
    "farm",
    "event_id",
    "canonical_asset_id",
    "canonical_asset_key",
    "canonical_asset_source",
    "asset_grouping_eligible",
}

missing_asset_registry_columns = sorted(
    required_asset_registry_columns
    - set(canonical_asset_registry.columns)
)

if missing_asset_registry_columns:
    raise RuntimeError(
        "Cell 5 canonical_asset_registry lacks required columns: "
        + ", ".join(missing_asset_registry_columns)
    )

if len(modeling_eligibility_registry) != EXPECTED_TOTAL_CASES:
    cell_6_validation_errors.append(
        "modeling_eligibility_registry contains "
        f"{len(modeling_eligibility_registry)} cases; expected "
        f"{EXPECTED_TOTAL_CASES}."
    )

if modeling_eligibility_registry.duplicated(
    subset=[
        "farm",
        "event_id",
    ]
).any():
    cell_6_validation_errors.append(
        "modeling_eligibility_registry contains duplicate case keys."
    )

ineligible_case_count = int(
    (~modeling_eligibility_registry[ELIGIBILITY_COLUMN].eq(True)).sum()
)

if ineligible_case_count > 0:
    cell_6_validation_errors.append(
        f"{ineligible_case_count} cases are not structurally eligible."
    )

if modeling_eligibility_registry[
    "canonical_asset_key"
].isna().any():
    cell_6_validation_errors.append(
        "At least one eligible case lacks a canonical_asset_key."
    )

if (
    ~modeling_eligibility_registry[
        "asset_grouping_eligible"
    ].eq(True)
).any():
    cell_6_validation_errors.append(
        "At least one case is not eligible for canonical-asset grouping."
    )

if int(
    modeling_eligibility_registry["is_anomaly"].eq(True).sum()
) != EXPECTED_ANOMALY_CASES:
    cell_6_validation_errors.append(
        "The eligibility registry does not contain the expected 45 "
        "anomaly cases."
    )

if int(
    modeling_eligibility_registry["is_anomaly"].eq(False).sum()
) != EXPECTED_NORMAL_CASES:
    cell_6_validation_errors.append(
        "The eligibility registry does not contain the expected 50 "
        "normal cases."
    )

if cell_6_validation_errors:
    raise RuntimeError(
        "CELL 6 INPUT VALIDATION FAILED:\n  - "
        + "\n  - ".join(cell_6_validation_errors)
    )

case_split_base = (
    modeling_eligibility_registry[
        [
            "farm",
            "event_id",
            "event_type",
            "is_anomaly",
            "canonical_asset_id",
            "canonical_asset_key",
            "canonical_asset_source",
            "asset_grouping_eligible",
            ELIGIBILITY_COLUMN,
        ]
    ]
    .copy()
)

asset_group_table = (
    case_split_base
    .groupby(
        [
            "farm",
            "canonical_asset_id",
            "canonical_asset_key",
        ],
        as_index=False,
        sort=True,
        dropna=False,
    )
    .agg(
        case_count=("event_id", "size"),
        anomaly_cases=(
            "is_anomaly",
            lambda values: int(values.eq(True).sum()),
        ),
        normal_cases=(
            "is_anomaly",
            lambda values: int(values.eq(False).sum()),
        ),
        source_values=(
            "canonical_asset_source",
            lambda values: " | ".join(
                sorted(set(values.astype(str)))
            ),
        ),
    )
    .sort_values(
        [
            "farm",
            "canonical_asset_key",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)

if len(asset_group_table) != EXPECTED_CANONICAL_ASSETS:
    raise RuntimeError(
        f"Detected {len(asset_group_table)} canonical assets; CARE v6 "
        f"audit expects {EXPECTED_CANONICAL_ASSETS}."
    )

if not (
    asset_group_table["case_count"]
    == (
        asset_group_table["anomaly_cases"]
        + asset_group_table["normal_cases"]
    )
).all():
    raise RuntimeError(
        "Asset-level anomaly/normal counts do not conserve case counts."
    )

cell_5_asset_comparison = (
    canonical_asset_summary[
        [
            "farm",
            "canonical_asset_key",
            "case_count",
            "anomaly_cases",
            "normal_cases",
        ]
    ]
    .merge(
        asset_group_table[
            [
                "farm",
                "canonical_asset_key",
                "case_count",
                "anomaly_cases",
                "normal_cases",
            ]
        ],
        on=[
            "farm",
            "canonical_asset_key",
        ],
        how="outer",
        validate="one_to_one",
        suffixes=("_cell_5", "_cell_6"),
        indicator=True,
    )
)

asset_summary_mismatch = (
    cell_5_asset_comparison["_merge"].ne("both")
)

for metric_name in (
    "case_count",
    "anomaly_cases",
    "normal_cases",
):
    asset_summary_mismatch |= (
        cell_5_asset_comparison[
            f"{metric_name}_cell_5"
        ].ne(
            cell_5_asset_comparison[
                f"{metric_name}_cell_6"
            ]
        )
    )

if asset_summary_mismatch.any():
    raise RuntimeError(
        "The reconstructed asset grouping table disagrees with Cell 5's "
        "canonical_asset_summary."
    )


# ----------------------------------------------------------------------------
# 5. Optimize and materialize the asset-grouped split
# ----------------------------------------------------------------------------

print("=" * 80)
print("OPTIMIZING FARM-AWARE ASSET-GROUPED SPLIT")
print("=" * 80)
print(
    f"Assets: {len(asset_group_table)} | cases: "
    f"{int(asset_group_table['case_count'].sum())} | candidate "
    f"assignments: {CELL_6_SEARCH_CANDIDATES:,}"
)
print(
    "Sensor files are not read; source measurements and raw train_test "
    "values are untouched."
)

best_assignment_codes, cell_6_search_details = (
    cell_6_optimize_asset_assignment(asset_group_table)
)

asset_split_assignment = asset_group_table.copy()
asset_split_assignment["model_split"] = [
    SPLIT_NAMES[int(split_code)]
    for split_code in best_assignment_codes
]
asset_split_assignment["target_case_ratio"] = (
    asset_split_assignment["model_split"].map(
        SPLIT_RATIOS
    )
)
asset_split_assignment["assignment_seed"] = (
    CELL_6_RANDOM_SEED
)
asset_split_assignment["assignment_policy"] = (
    "farm_aware_asset_grouped_70_15_15"
)

assignment_digest_sha256 = cell_6_assignment_digest(
    asset_split_assignment
)

case_split_registry = case_split_base.merge(
    asset_split_assignment[
        [
            "farm",
            "canonical_asset_key",
            "model_split",
        ]
    ],
    on=[
        "farm",
        "canonical_asset_key",
    ],
    how="left",
    validate="many_to_one",
)

case_metadata_columns = [
    column_name
    for column_name in (
        "farm",
        "event_id",
        "file_name",
        "file_path",
        "size_bytes",
        "size_mb",
        "asset_id",
        "event_start",
        "event_end",
    )
    if column_name in case_registry.columns
]

case_split_registry = case_split_registry.merge(
    case_registry[case_metadata_columns],
    on=[
        "farm",
        "event_id",
    ],
    how="left",
    validate="one_to_one",
)

case_split_registry["is_train"] = (
    case_split_registry["model_split"].eq("train")
)
case_split_registry["is_validation"] = (
    case_split_registry["model_split"].eq("validation")
)
case_split_registry["is_test"] = (
    case_split_registry["model_split"].eq("test")
)

split_order_dtype = pd.CategoricalDtype(
    categories=list(SPLIT_NAMES),
    ordered=True,
)

asset_split_assignment["model_split"] = (
    asset_split_assignment["model_split"].astype(
        split_order_dtype
    )
)
case_split_registry["model_split"] = (
    case_split_registry["model_split"].astype(
        split_order_dtype
    )
)

asset_split_assignment = (
    asset_split_assignment
    .sort_values(
        [
            "model_split",
            "farm",
            "canonical_asset_key",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)

case_split_registry = (
    case_split_registry
    .sort_values(
        [
            "model_split",
            "farm",
            "event_id",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)

train_case_registry = (
    case_split_registry.loc[
        case_split_registry["model_split"].eq("train")
    ]
    .reset_index(drop=True)
)
validation_case_registry = (
    case_split_registry.loc[
        case_split_registry["model_split"].eq("validation")
    ]
    .reset_index(drop=True)
)
test_case_registry = (
    case_split_registry.loc[
        case_split_registry["model_split"].eq("test")
    ]
    .reset_index(drop=True)
)


# ----------------------------------------------------------------------------
# 6. Build split summaries
# ----------------------------------------------------------------------------

split_overall_summary = (
    case_split_registry
    .groupby(
        "model_split",
        observed=False,
        sort=False,
    )
    .agg(
        case_count=("event_id", "size"),
        anomaly_cases=(
            "is_anomaly",
            lambda values: int(values.eq(True).sum()),
        ),
        normal_cases=(
            "is_anomaly",
            lambda values: int(values.eq(False).sum()),
        ),
        farm_count=("farm", "nunique"),
        canonical_asset_count=(
            "canonical_asset_key",
            "nunique",
        ),
    )
    .reset_index()
)

split_overall_summary["target_case_ratio"] = (
    split_overall_summary["model_split"]
    .astype("object")
    .map(SPLIT_RATIOS)
)
split_overall_summary["achieved_case_ratio"] = (
    split_overall_summary["case_count"]
    / EXPECTED_TOTAL_CASES
)
split_overall_summary["anomaly_fraction"] = (
    split_overall_summary["anomaly_cases"]
    / split_overall_summary["case_count"]
)

split_farm_class_summary = (
    case_split_registry
    .groupby(
        [
            "model_split",
            "farm",
        ],
        observed=False,
        sort=False,
    )
    .agg(
        case_count=("event_id", "size"),
        anomaly_cases=(
            "is_anomaly",
            lambda values: int(values.eq(True).sum()),
        ),
        normal_cases=(
            "is_anomaly",
            lambda values: int(values.eq(False).sum()),
        ),
        canonical_asset_count=(
            "canonical_asset_key",
            "nunique",
        ),
    )
    .reset_index()
)

split_farm_class_summary["case_fraction_within_farm"] = (
    split_farm_class_summary["case_count"]
    / split_farm_class_summary.groupby(
        "farm",
        observed=False,
    )["case_count"].transform("sum")
)
split_farm_class_summary["anomaly_fraction"] = (
    split_farm_class_summary["anomaly_cases"]
    / split_farm_class_summary["case_count"]
)


# ----------------------------------------------------------------------------
# 7. Leakage and conservation validation
# ----------------------------------------------------------------------------

target_case_counts = cell_6_integer_targets(
    EXPECTED_TOTAL_CASES
)
observed_case_counts = (
    split_overall_summary
    .set_index(
        split_overall_summary["model_split"].astype(str)
    )["case_count"]
    .reindex(SPLIT_NAMES)
    .to_numpy(dtype=int)
)

asset_memberships_per_key = (
    asset_split_assignment
    .groupby("canonical_asset_key", observed=False)[
        "model_split"
    ]
    .nunique()
)

pairwise_overlap_records: list[dict[str, Any]] = []

for left_index, left_split in enumerate(SPLIT_NAMES):
    left_assets = set(
        asset_split_assignment.loc[
            asset_split_assignment["model_split"].eq(left_split),
            "canonical_asset_key",
        ].astype(str)
    )

    for right_split in SPLIT_NAMES[left_index + 1:]:
        right_assets = set(
            asset_split_assignment.loc[
                asset_split_assignment["model_split"].eq(
                    right_split
                ),
                "canonical_asset_key",
            ].astype(str)
        )
        overlapping_assets = sorted(
            left_assets & right_assets
        )
        pairwise_overlap_records.append(
            {
                "left_split": left_split,
                "right_split": right_split,
                "overlap_count": len(overlapping_assets),
                "overlapping_asset_keys": " | ".join(
                    overlapping_assets
                ),
            }
        )

split_asset_overlap_audit = pd.DataFrame(
    pairwise_overlap_records
)

expected_farm_split_rows = (
    len(SPLIT_NAMES)
    * case_split_registry["farm"].nunique()
)

all_farms_every_split = bool(
    len(split_farm_class_summary) == expected_farm_split_rows
    and split_farm_class_summary["case_count"].gt(0).all()
)

both_classes_every_farm_split = bool(
    split_farm_class_summary["anomaly_cases"].gt(0).all()
    and split_farm_class_summary["normal_cases"].gt(0).all()
)

both_classes_every_split = bool(
    split_overall_summary["anomaly_cases"].gt(0).all()
    and split_overall_summary["normal_cases"].gt(0).all()
)

constraint_records = [
    {
        "constraint": "all_95_cases_assigned_once",
        "passed": bool(
            len(case_split_registry) == EXPECTED_TOTAL_CASES
            and not case_split_registry.duplicated(
                subset=["farm", "event_id"]
            ).any()
        ),
        "observed": int(len(case_split_registry)),
        "expected": EXPECTED_TOTAL_CASES,
    },
    {
        "constraint": "all_36_assets_assigned_once",
        "passed": bool(
            len(asset_split_assignment) == EXPECTED_CANONICAL_ASSETS
            and asset_memberships_per_key.eq(1).all()
        ),
        "observed": int(len(asset_split_assignment)),
        "expected": EXPECTED_CANONICAL_ASSETS,
    },
    {
        "constraint": "zero_pairwise_asset_overlap",
        "passed": bool(
            split_asset_overlap_audit["overlap_count"].eq(0).all()
        ),
        "observed": int(
            split_asset_overlap_audit["overlap_count"].sum()
        ),
        "expected": 0,
    },
    {
        "constraint": "exact_integer_case_targets",
        "passed": bool(
            np.array_equal(
                observed_case_counts,
                target_case_counts,
            )
        ),
        "observed": " | ".join(
            f"{name}={count}"
            for name, count in zip(
                SPLIT_NAMES,
                observed_case_counts,
            )
        ),
        "expected": " | ".join(
            f"{name}={count}"
            for name, count in zip(
                SPLIT_NAMES,
                target_case_counts,
            )
        ),
    },
    {
        "constraint": "all_farms_present_in_every_split",
        "passed": all_farms_every_split,
        "observed": int(len(split_farm_class_summary)),
        "expected": int(expected_farm_split_rows),
    },
    {
        "constraint": "both_classes_present_in_every_split",
        "passed": both_classes_every_split,
        "observed": bool(both_classes_every_split),
        "expected": True,
    },
    {
        "constraint": "both_classes_present_in_every_farm_split",
        "passed": both_classes_every_farm_split,
        "observed": bool(both_classes_every_farm_split),
        "expected": True,
    },
    {
        "constraint": "all_cases_structurally_eligible",
        "passed": bool(
            case_split_registry[ELIGIBILITY_COLUMN].eq(True).all()
        ),
        "observed": int(
            case_split_registry[ELIGIBILITY_COLUMN].eq(True).sum()
        ),
        "expected": EXPECTED_TOTAL_CASES,
    },
    {
        "constraint": "anomaly_case_conservation",
        "passed": bool(
            case_split_registry["is_anomaly"].eq(True).sum()
            == EXPECTED_ANOMALY_CASES
        ),
        "observed": int(
            case_split_registry["is_anomaly"].eq(True).sum()
        ),
        "expected": EXPECTED_ANOMALY_CASES,
    },
    {
        "constraint": "normal_case_conservation",
        "passed": bool(
            case_split_registry["is_anomaly"].eq(False).sum()
            == EXPECTED_NORMAL_CASES
        ),
        "observed": int(
            case_split_registry["is_anomaly"].eq(False).sum()
        ),
        "expected": EXPECTED_NORMAL_CASES,
    },
    {
        "constraint": "source_data_not_modified",
        "passed": True,
        "observed": False,
        "expected": False,
    },
]

split_constraint_audit = pd.DataFrame(
    constraint_records
)

failed_constraints = split_constraint_audit.loc[
    ~split_constraint_audit["passed"].eq(True)
]

if not failed_constraints.empty:
    cell_6_validation_errors.extend(
        "Constraint failed: " + constraint_name
        for constraint_name in failed_constraints["constraint"]
    )

if cell_6_validation_errors:
    raise RuntimeError(
        "CELL 6 VALIDATION FAILED:\n  - "
        + "\n  - ".join(cell_6_validation_errors)
    )


# ----------------------------------------------------------------------------
# 8. Save Cell 6 outputs and manifest
# ----------------------------------------------------------------------------

asset_split_assignment.to_csv(
    TABLE_DIR / "care_asset_split_assignment.csv",
    index=False,
)

case_split_registry.to_csv(
    TABLE_DIR / "care_case_split_registry.csv",
    index=False,
)

split_overall_summary.to_csv(
    TABLE_DIR / "care_split_overall_summary.csv",
    index=False,
)

split_farm_class_summary.to_csv(
    TABLE_DIR / "care_split_farm_class_summary.csv",
    index=False,
)

split_asset_overlap_audit.to_csv(
    TABLE_DIR / "care_split_asset_overlap_audit.csv",
    index=False,
)

split_constraint_audit.to_csv(
    TABLE_DIR / "care_split_constraint_audit.csv",
    index=False,
)

cell_6_manifest = {
    "cell": 6,
    "purpose": (
        "Deterministic farm-aware, canonical-asset-grouped "
        "train/validation/test split"
    ),
    "policy": CELL_6_POLICY,
    "search_details": cell_6_search_details,
    "assignment_digest_sha256": assignment_digest_sha256,
    "case_count": int(len(case_split_registry)),
    "canonical_asset_count": int(len(asset_split_assignment)),
    "split_overall_summary": split_overall_summary.to_dict(
        orient="records"
    ),
    "split_farm_class_summary": split_farm_class_summary.to_dict(
        orient="records"
    ),
    "constraint_audit": split_constraint_audit.to_dict(
        orient="records"
    ),
    "validation_errors": cell_6_validation_errors,
    "validation_warnings": cell_6_validation_warnings,
    "sensor_file_bodies_read": False,
    "source_data_modified": False,
}

save_cell_6_json(
    cell_6_manifest,
    OUTPUT_ROOT / "asset_grouped_split_manifest.json",
)


# ----------------------------------------------------------------------------
# 9. Display summaries and successful completion
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ASSET-GROUPED SPLIT SUMMARY")
print("=" * 80)
display(split_overall_summary)

print("\n" + "=" * 80)
print("FARM AND CLASS DISTRIBUTION BY SPLIT")
print("=" * 80)
display(split_farm_class_summary)

print("\n" + "=" * 80)
print("ASSET-LEAKAGE AND CONSERVATION AUDIT")
print("=" * 80)
display(split_constraint_audit)

print("\n" + "=" * 80)
print("CELL 6 COMPLETED SUCCESSFULLY")
print("=" * 80)

for summary_row in split_overall_summary.itertuples(index=False):
    print(
        f"{str(summary_row.model_split):<10}: "
        f"{int(summary_row.case_count):>2} cases | "
        f"{int(summary_row.anomaly_cases):>2} anomaly | "
        f"{int(summary_row.normal_cases):>2} normal | "
        f"{int(summary_row.canonical_asset_count):>2} assets"
    )

print(f"Canonical asset overlap : 0")
print(f"Assignment SHA-256      : {assignment_digest_sha256}")
print(f"Output directory        : {TABLE_DIR}")
print("Sensor bodies read      : No")
print("Source data modified    : No")
print("Reusable objects:")
print("  - asset_split_assignment")
print("  - case_split_registry")
print("  - train_case_registry")
print("  - validation_case_registry")
print("  - test_case_registry")
print("  - split_overall_summary")
print("  - split_farm_class_summary")
print("  - split_asset_overlap_audit")
print("  - split_constraint_audit")
print("  - cell_6_manifest")
print("Created:")
print("  - care_asset_split_assignment.csv")
print("  - care_case_split_registry.csv")
print("  - care_split_overall_summary.csv")
print("  - care_split_farm_class_summary.csv")
print("  - care_split_asset_overlap_audit.csv")
print("  - care_split_constraint_audit.csv")
print("  - asset_grouped_split_manifest.json")
