"""
=============================================================================
MSSP UPGRADE
Cell 16: amended UORED v5 external-validation gate
=============================================================================

Run once after the successful Cells 15A and 15B. This cell applies the
unchanged Cell 13 unit-level inference rules to the frozen 54-revolution
features registered by the outcome-blind Cell 15A amendment.

Stage 1: leave-one-bearing-out healthy-reference conformal detection.
Stage 2: conditional BPFI/BPFO localization from unit-state median z scores.

No threshold, statistic, target class, alpha level or publication-gate
criterion is selected in this cell. The result is immutable after creation.
All figures are high-resolution PNG only.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import beta


# -------------------------------------------------------------------------
# 1. Frozen prerequisites and provenance checks
# -------------------------------------------------------------------------

required_cell16_names = [
    "MSSP_DIRS",
    "V5_REGISTRATION_HASH",
    "CELL13_REGISTRATION_PATH",
    "V5_RESOLUTION_AMENDMENT_HASH",
    "V5_RESOLUTION_AMENDMENT_PATH",
    "UORED_MANIFEST_HASH",
    "UORED_LOADER_SPEC_HASH",
    "CELL15_CONFIG_HASH",
    "UORED_SEGMENT_DESIGN_HASH",
    "CELL15_FEATURE_PATH",
    "CELL15_STATE_PATH",
]
missing_cell16_names = [
    name for name in required_cell16_names
    if name not in globals()
]
if missing_cell16_names:
    raise RuntimeError(
        "Run Cells 13, 14, 15A and 15B in the same notebook session. "
        "Missing: " + ", ".join(missing_cell16_names)
    )

EXPECTED_CELL13_HASH = (
    "82f23198d97b4c136096a8acbd8452fc5c53fdb58da79fbcd3590c9f74ed090c"
)
EXPECTED_AMENDMENT_HASH = (
    "00c6e38b6cfd06c2afda00e383a6e6d5398957b357b17133efb106cb92363e51"
)
EXPECTED_MANIFEST_HASH = (
    "41cb8a6870181556a2ae3b2636321ead9dd8f066def82dfc5bba98f6fec01018"
)
EXPECTED_LOADER_HASH = (
    "5782fe0afc5a824b252d46adde3338e2dffeaf44a55872ee13829186464ff55d"
)
EXPECTED_CELL15B_CONFIG_HASH = (
    "15ebacc76ad039fa737fbb15450b20bde7c0d7dda4bea2ff3d5addc5936d217f"
)
EXPECTED_CELL15B_DESIGN_HASH = (
    "b1070393ec02e410b6b649baa2dc3560022dee71270814d36e279de991632711"
)

observed_hashes = {
    "Cell 13 registration": str(V5_REGISTRATION_HASH),
    "Cell 15A amendment": str(V5_RESOLUTION_AMENDMENT_HASH),
    "Cell 14 manifest": str(UORED_MANIFEST_HASH),
    "Cell 14 loader": str(UORED_LOADER_SPEC_HASH),
    "Cell 15B configuration": str(CELL15_CONFIG_HASH),
    "Cell 15B segment design": str(UORED_SEGMENT_DESIGN_HASH),
}
expected_hashes = {
    "Cell 13 registration": EXPECTED_CELL13_HASH,
    "Cell 15A amendment": EXPECTED_AMENDMENT_HASH,
    "Cell 14 manifest": EXPECTED_MANIFEST_HASH,
    "Cell 14 loader": EXPECTED_LOADER_HASH,
    "Cell 15B configuration": EXPECTED_CELL15B_CONFIG_HASH,
    "Cell 15B segment design": EXPECTED_CELL15B_DESIGN_HASH,
}
for name, expected in expected_hashes.items():
    if observed_hashes[name] != expected:
        raise RuntimeError(
            f"{name} hash changed: expected {expected}, "
            f"found {observed_hashes[name]}"
        )


def canonical_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


registration_path = Path(CELL13_REGISTRATION_PATH)
amendment_path = Path(V5_RESOLUTION_AMENDMENT_PATH)
feature_path = Path(CELL15_FEATURE_PATH)
state_path = Path(CELL15_STATE_PATH)
for path in [registration_path, amendment_path, feature_path, state_path]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen Cell 16 input is missing: {path}")

with registration_path.open("r", encoding="utf-8") as handle:
    stored_registration_record = json.load(handle)
with amendment_path.open("r", encoding="utf-8") as handle:
    stored_amendment_record = json.load(handle)

if canonical_digest(stored_registration_record["registration"]) != (
    EXPECTED_CELL13_HASH
):
    raise RuntimeError("Stored Cell 13 registration failed verification")
if canonical_digest(stored_amendment_record["amendment"]) != (
    EXPECTED_AMENDMENT_HASH
):
    raise RuntimeError("Stored Cell 15A amendment failed verification")

registered_gate = stored_registration_record[
    "registration"
]["registered_publication_gate"]
expected_gate_values = {
    "healthy_unit_false_alarm_rate_at_most": 0.10,
    "faulty_minimum_class_detection_at_least": 0.80,
    "faulty_minimum_class_localization_at_least": 0.80,
    "faulty_minimum_class_correct_diagnosis_at_least": 0.80,
    "developing_minimum_class_detection_at_least": 0.60,
    "developing_minimum_class_localization_at_least": 0.60,
    "developing_minimum_class_correct_diagnosis_at_least": 0.60,
}
for key, expected in expected_gate_values.items():
    if not np.isclose(float(registered_gate[key]), expected):
        raise RuntimeError(f"Registered gate changed for {key}")

SOURCE_FEATURE_SHA256 = sha256_file(feature_path)
SOURCE_STATE_SHA256 = sha256_file(state_path)


# -------------------------------------------------------------------------
# 2. Lock the Cell 16 computation before deriving performance outcomes
# -------------------------------------------------------------------------

CELL16_CONFIG = {
    "method_version": "v5.0-uored-amended-external-validation",
    "parent_registration_sha256": EXPECTED_CELL13_HASH,
    "resolution_amendment_sha256": EXPECTED_AMENDMENT_HASH,
    "manifest_sha256": EXPECTED_MANIFEST_HASH,
    "loader_spec_sha256": EXPECTED_LOADER_HASH,
    "cell15b_config_sha256": EXPECTED_CELL15B_CONFIG_HASH,
    "cell15b_segment_design_sha256": EXPECTED_CELL15B_DESIGN_HASH,
    "segment_feature_file_sha256": SOURCE_FEATURE_SHA256,
    "unit_state_file_sha256": SOURCE_STATE_SHA256,
    "primary_units": {
        "IR": [1, 2, 3, 4, 5],
        "OR": [6, 7, 8, 9, 10],
    },
    "healthy_units": list(range(1, 21)),
    "states": ["healthy", "developing", "faulty"],
    "detection_score": "unit_detection_max_z_q90",
    "localization_scores": [
        "unit_BPFI_z_median",
        "unit_BPFO_z_median",
    ],
    "conformal_scheme": (
        "leave-one-physical-bearing-out against the other 19 healthy units"
    ),
    "conformal_probability": (
        "(1 + count(calibration score >= test score)) / 20"
    ),
    "tie_handling": "greater-than-or-equal counts against the test score",
    "alpha": 0.05,
    "alarm_rule": "p <= 0.05",
    "localization_rule": (
        "strict argmax of unit median BPFI_z versus BPFO_z; exact tie is "
        "indeterminate and incorrect"
    ),
    "interval": "two-sided 95% Clopper-Pearson exact binomial",
    "registered_gate": expected_gate_values,
    "all_gate_criteria_required": True,
    "secondary_states_excluded_from_gate": ["B", "C"],
    "figure_format": "PNG only",
    "figure_dpi": 600,
}
CELL16_CONFIG_HASH = canonical_digest(CELL16_CONFIG)

CELL16_ROOT = (
    Path(MSSP_DIRS["statistics"]) /
    "uored_v5_external_validation_amended"
)
CELL16_ROOT.mkdir(parents=True, exist_ok=True)
CELL16_CONFIG_PATH = CELL16_ROOT / "cell16_analysis_config.json"
CELL16_RESULT_PATH = CELL16_ROOT / "cell16_external_validation_result.json"

config_record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "analysis_config_sha256": CELL16_CONFIG_HASH,
    "config": CELL16_CONFIG,
}
if CELL16_CONFIG_PATH.exists():
    with CELL16_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        prior_config_record = json.load(handle)
    if prior_config_record.get("analysis_config_sha256") != (
        CELL16_CONFIG_HASH
    ):
        raise RuntimeError(
            "A conflicting Cell 16 configuration exists. Do not overwrite "
            f"{CELL16_CONFIG_PATH}."
        )
else:
    with CELL16_CONFIG_PATH.open("x", encoding="utf-8") as handle:
        json.dump(config_record, handle, indent=2, sort_keys=True)
        handle.write("\n")


# -------------------------------------------------------------------------
# 3. Frozen feature-integrity and registry audit
# -------------------------------------------------------------------------

segment_features = pd.read_csv(feature_path)
unit_states = pd.read_csv(state_path)

if len(segment_features) != 300:
    raise RuntimeError(
        f"Expected 300 frozen segment rows, found {len(segment_features)}"
    )
if segment_features.trial_id.duplicated().any():
    raise RuntimeError("Duplicate frozen segment trial IDs")
if segment_features.record_id.nunique() != 60:
    raise RuntimeError("Frozen features do not cover exactly 60 states")
if not (segment_features.groupby("record_id").size() == 5).all():
    raise RuntimeError("Every state must contain exactly five segments")
if set(segment_features.config_sha256.astype(str)) != {
    EXPECTED_CELL15B_CONFIG_HASH
}:
    raise RuntimeError("Segment feature configuration hash mismatch")
if set(segment_features.design_sha256.astype(str)) != {
    EXPECTED_CELL15B_DESIGN_HASH
}:
    raise RuntimeError("Segment feature design hash mismatch")

if len(unit_states) != 60:
    raise RuntimeError(
        f"Expected 60 unit-state rows, found {len(unit_states)}"
    )
if unit_states.record_id.duplicated().any():
    raise RuntimeError("Duplicate frozen unit-state record IDs")
if unit_states.unit.nunique() != 20:
    raise RuntimeError("Frozen state table does not contain 20 bearings")
if not (pd.to_numeric(unit_states.n_segments) == 5).all():
    raise RuntimeError("Unit-state aggregation does not use five segments")
if set(unit_states.aggregation_config_sha256.astype(str)) != {
    EXPECTED_CELL15B_CONFIG_HASH
}:
    raise RuntimeError("Unit-state configuration hash mismatch")
if set(unit_states.segment_design_sha256.astype(str)) != {
    EXPECTED_CELL15B_DESIGN_HASH
}:
    raise RuntimeError("Unit-state design hash mismatch")

primary_score_columns = [
    "unit_detection_max_z_q90",
    "unit_BPFI_z_median",
    "unit_BPFO_z_median",
]
for column in primary_score_columns:
    if column not in unit_states:
        raise RuntimeError(f"Missing frozen primary score: {column}")
    unit_states[column] = pd.to_numeric(unit_states[column], errors="coerce")
if not np.isfinite(unit_states[primary_score_columns].to_numpy()).all():
    raise RuntimeError("Non-finite frozen unit-state primary scores")

unit_states["unit_number"] = pd.to_numeric(
    unit_states.unit_number, errors="raise"
).astype(int)
unit_states["state"] = unit_states.state.astype(str).str.lower()
unit_states["fault_label"] = unit_states.fault_label.astype(str)

expected_primary_record_ids = {
    *(f"H-{unit}-0" for unit in range(1, 21)),
    *(f"I-{unit}-{state}" for unit in range(1, 6) for state in [1, 2]),
    *(f"O-{unit}-{state}" for unit in range(6, 11) for state in [1, 2]),
}
primary_mask = (
    (unit_states.state == "healthy") |
    (unit_states.fault_label.isin(["IR", "OR"]))
)
primary_states = unit_states.loc[primary_mask].copy()
actual_primary_record_ids = set(primary_states.record_id.astype(str))
missing_primary_ids = sorted(
    expected_primary_record_ids - actual_primary_record_ids
)
extra_primary_ids = sorted(
    actual_primary_record_ids - expected_primary_record_ids
)
if missing_primary_ids or extra_primary_ids:
    raise RuntimeError(
        "Primary registry mismatch: "
        f"missing={missing_primary_ids}, extra={extra_primary_ids}"
    )
if len(primary_states) != 40:
    raise RuntimeError("The registered primary evaluation must have 40 rows")

amended_error_path = feature_path.parent / "uored_v5_extraction_errors.csv"
amended_error_count = 0
if amended_error_path.exists():
    amended_error_count = len(pd.read_csv(amended_error_path))
if amended_error_count != 0:
    raise RuntimeError(
        f"The amended extraction contains {amended_error_count} errors"
    )


# -------------------------------------------------------------------------
# 4. Leave-one-bearing-out conformal evaluation
# -------------------------------------------------------------------------

healthy_states = primary_states[
    primary_states.state == "healthy"
].copy()
if len(healthy_states) != 20 or healthy_states.unit.nunique() != 20:
    raise RuntimeError("Healthy conformal bank must contain 20 bearings")

evaluation_rows = []
calibration_link_rows = []

for test in primary_states.sort_values(
    ["state_code", "fault_label", "unit_number"]
).itertuples(index=False):
    calibration = healthy_states[
        healthy_states.unit != test.unit
    ].sort_values("unit_number")
    if len(calibration) != 19:
        raise RuntimeError(
            f"Expected 19 calibration bearings for {test.record_id}"
        )
    if test.unit in set(calibration.unit):
        raise RuntimeError(f"Test-bearing leakage for {test.record_id}")

    test_score = float(test.unit_detection_max_z_q90)
    calibration_scores = calibration[
        "unit_detection_max_z_q90"
    ].to_numpy(dtype=float)
    exceedances = int(np.count_nonzero(calibration_scores >= test_score))
    conformal_p = float((1 + exceedances) / 20.0)
    alarm = bool(conformal_p <= 0.05)

    bpfi_score = float(test.unit_BPFI_z_median)
    bpfo_score = float(test.unit_BPFO_z_median)
    if bpfi_score > bpfo_score:
        predicted_order = "BPFI"
    elif bpfo_score > bpfi_score:
        predicted_order = "BPFO"
    else:
        predicted_order = "INDETERMINATE"

    target_order = (
        "BPFI" if test.fault_label == "IR" else
        "BPFO" if test.fault_label == "OR" else None
    )
    is_fault_state = test.fault_label in {"IR", "OR"}
    localization_correct = (
        bool(predicted_order == target_order)
        if is_fault_state else None
    )
    correct_diagnosis = (
        bool(alarm and localization_correct)
        if is_fault_state else None
    )

    evaluation_rows.append({
        "record_id": test.record_id,
        "unit": test.unit,
        "unit_number": int(test.unit_number),
        "state": test.state,
        "state_code": int(test.state_code),
        "fault_label": test.fault_label,
        "target_order": target_order,
        "n_segments": int(test.n_segments),
        "detection_score_q90_max_z": test_score,
        "BPFI_z_median": bpfi_score,
        "BPFO_z_median": bpfo_score,
        "order_score_difference_BPFI_minus_BPFO": (
            bpfi_score - bpfo_score
        ),
        "n_calibration_units": int(len(calibration)),
        "conformal_exceedances": exceedances,
        "conformal_p": conformal_p,
        "minimum_attainable_p": 0.05,
        "calibration_score_max": float(calibration_scores.max()),
        "score_minus_calibration_max": float(
            test_score - calibration_scores.max()
        ),
        "alarm": alarm,
        "predicted_order": predicted_order,
        "localization_correct": localization_correct,
        "correct_diagnosis": correct_diagnosis,
        "calibration_units": json.dumps(
            calibration.unit.astype(str).tolist()
        ),
    })

    for cal in calibration.itertuples(index=False):
        calibration_link_rows.append({
            "test_record_id": test.record_id,
            "test_unit": test.unit,
            "test_state": test.state,
            "test_fault_label": test.fault_label,
            "calibration_record_id": cal.record_id,
            "calibration_unit": cal.unit,
            "calibration_score": float(
                cal.unit_detection_max_z_q90
            ),
            "shared_physical_unit": bool(cal.unit == test.unit),
        })

V5_UNIT_REGISTRY = pd.DataFrame(evaluation_rows).sort_values(
    ["state_code", "fault_label", "unit_number"]
).reset_index(drop=True)
V5_CALIBRATION_LINKS = pd.DataFrame(calibration_link_rows)

if len(V5_UNIT_REGISTRY) != 40:
    raise RuntimeError("Cell 16 unit registry is not exactly 40 rows")
if len(V5_CALIBRATION_LINKS) != 40 * 19:
    raise RuntimeError("Cell 16 calibration-link registry is incomplete")
if V5_CALIBRATION_LINKS.shared_physical_unit.any():
    raise RuntimeError("Physical-bearing leakage detected")
if not set(V5_UNIT_REGISTRY.conformal_p.unique()).issubset(
    {index / 20.0 for index in range(1, 21)}
):
    raise RuntimeError("Non-attainable conformal probability detected")


# -------------------------------------------------------------------------
# 5. Exact intervals and registered estimands
# -------------------------------------------------------------------------

def clopper_pearson(successes, total, confidence=0.95):
    successes = int(successes)
    total = int(total)
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial count")
    alpha_tail = (1.0 - confidence) / 2.0
    lower = (
        0.0 if successes == 0 else
        float(beta.ppf(alpha_tail, successes, total - successes + 1))
    )
    upper = (
        1.0 if successes == total else
        float(beta.ppf(
            1.0 - alpha_tail,
            successes + 1,
            total - successes,
        ))
    )
    return lower, upper


def interval_row(estimand, severity, fault_label, values):
    values = np.asarray(values, dtype=bool)
    successes = int(values.sum())
    total = int(values.size)
    lower, upper = clopper_pearson(successes, total)
    return {
        "estimand": estimand,
        "severity": severity,
        "fault_label": fault_label,
        "successes": successes,
        "n": total,
        "rate": float(successes / total),
        "ci_low": lower,
        "ci_high": upper,
        "interval": "Clopper-Pearson exact 95%",
    }


interval_rows = []
healthy_eval = V5_UNIT_REGISTRY[V5_UNIT_REGISTRY.state == "healthy"]
interval_rows.append(interval_row(
    "false_alarm",
    "healthy",
    "N",
    healthy_eval.alarm,
))

for severity in ["developing", "faulty"]:
    for fault_label in ["IR", "OR"]:
        group = V5_UNIT_REGISTRY[
            (V5_UNIT_REGISTRY.state == severity) &
            (V5_UNIT_REGISTRY.fault_label == fault_label)
        ]
        if len(group) != 5:
            raise RuntimeError(
                f"Expected five {severity}/{fault_label} units"
            )
        interval_rows.append(interval_row(
            "detection", severity, fault_label, group.alarm
        ))
        interval_rows.append(interval_row(
            "localization",
            severity,
            fault_label,
            group.localization_correct,
        ))
        interval_rows.append(interval_row(
            "correct_diagnosis",
            severity,
            fault_label,
            group.correct_diagnosis,
        ))

V5_EXACT_INTERVALS = pd.DataFrame(interval_rows)


# -------------------------------------------------------------------------
# 6. Mandatory 2x2 detection tables and localization matrices
# -------------------------------------------------------------------------

healthy_fp = int(healthy_eval.alarm.sum())
healthy_tn = int(len(healthy_eval) - healthy_fp)
table_2x2_rows = []

for severity in ["developing", "faulty"]:
    for fault_label in ["IR", "OR"]:
        group = V5_UNIT_REGISTRY[
            (V5_UNIT_REGISTRY.state == severity) &
            (V5_UNIT_REGISTRY.fault_label == fault_label)
        ]
        tp = int(group.alarm.sum())
        fn = int(len(group) - tp)
        sensitivity_low, sensitivity_high = clopper_pearson(tp, len(group))
        specificity_low, specificity_high = clopper_pearson(
            healthy_tn,
            len(healthy_eval),
        )
        table_2x2_rows.append({
            "severity": severity,
            "fault_label": fault_label,
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": healthy_fp,
            "true_negative": healthy_tn,
            "sensitivity": float(tp / len(group)),
            "sensitivity_ci_low": sensitivity_low,
            "sensitivity_ci_high": sensitivity_high,
            "specificity": float(healthy_tn / len(healthy_eval)),
            "specificity_ci_low": specificity_low,
            "specificity_ci_high": specificity_high,
        })

V5_DETECTION_2X2 = pd.DataFrame(table_2x2_rows)

confusion_rows = []
for severity in ["developing", "faulty"]:
    fault_group = V5_UNIT_REGISTRY[
        V5_UNIT_REGISTRY.state == severity
    ]
    for target_order in ["BPFI", "BPFO"]:
        target_group = fault_group[
            fault_group.target_order == target_order
        ]
        for predicted_order in ["BPFI", "BPFO", "INDETERMINATE"]:
            confusion_rows.append({
                "severity": severity,
                "target_order": target_order,
                "predicted_order": predicted_order,
                "count": int(
                    (target_group.predicted_order == predicted_order).sum()
                ),
                "n_target": int(len(target_group)),
            })

V5_LOCALIZATION_CONFUSION = pd.DataFrame(confusion_rows)


# -------------------------------------------------------------------------
# 7. Frozen publication gate
# -------------------------------------------------------------------------

def registered_rate(estimand, severity, fault_label):
    row = V5_EXACT_INTERVALS[
        (V5_EXACT_INTERVALS.estimand == estimand) &
        (V5_EXACT_INTERVALS.severity == severity) &
        (V5_EXACT_INTERVALS.fault_label == fault_label)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Missing estimand: {estimand}/{severity}/{fault_label}"
        )
    return float(row.rate.iloc[0])


healthy_fwer = registered_rate("false_alarm", "healthy", "N")
faulty_detection_min = min(
    registered_rate("detection", "faulty", label)
    for label in ["IR", "OR"]
)
faulty_localization_min = min(
    registered_rate("localization", "faulty", label)
    for label in ["IR", "OR"]
)
faulty_diagnosis_min = min(
    registered_rate("correct_diagnosis", "faulty", label)
    for label in ["IR", "OR"]
)
developing_detection_min = min(
    registered_rate("detection", "developing", label)
    for label in ["IR", "OR"]
)
developing_localization_min = min(
    registered_rate("localization", "developing", label)
    for label in ["IR", "OR"]
)
developing_diagnosis_min = min(
    registered_rate("correct_diagnosis", "developing", label)
    for label in ["IR", "OR"]
)

gate_rows = [
    {
        "criterion": "complete_primary_registry",
        "comparison": "equal",
        "threshold": 1.0,
        "observed": float(
            actual_primary_record_ids == expected_primary_record_ids
        ),
        "passed": actual_primary_record_ids == expected_primary_record_ids,
    },
    {
        "criterion": "zero_unregistered_primary_exclusions",
        "comparison": "at_most",
        "threshold": 0.0,
        "observed": float(amended_error_count),
        "passed": amended_error_count == 0,
    },
    {
        "criterion": "healthy_unit_false_alarm_rate_at_most_0.10",
        "comparison": "at_most",
        "threshold": 0.10,
        "observed": healthy_fwer,
        "passed": healthy_fwer <= 0.10,
    },
    {
        "criterion": "faulty_minimum_class_detection_at_least_0.80",
        "comparison": "at_least",
        "threshold": 0.80,
        "observed": faulty_detection_min,
        "passed": faulty_detection_min >= 0.80,
    },
    {
        "criterion": "faulty_minimum_class_localization_at_least_0.80",
        "comparison": "at_least",
        "threshold": 0.80,
        "observed": faulty_localization_min,
        "passed": faulty_localization_min >= 0.80,
    },
    {
        "criterion": (
            "faulty_minimum_class_correct_diagnosis_at_least_0.80"
        ),
        "comparison": "at_least",
        "threshold": 0.80,
        "observed": faulty_diagnosis_min,
        "passed": faulty_diagnosis_min >= 0.80,
    },
    {
        "criterion": "developing_minimum_class_detection_at_least_0.60",
        "comparison": "at_least",
        "threshold": 0.60,
        "observed": developing_detection_min,
        "passed": developing_detection_min >= 0.60,
    },
    {
        "criterion": (
            "developing_minimum_class_localization_at_least_0.60"
        ),
        "comparison": "at_least",
        "threshold": 0.60,
        "observed": developing_localization_min,
        "passed": developing_localization_min >= 0.60,
    },
    {
        "criterion": (
            "developing_minimum_class_correct_diagnosis_at_least_0.60"
        ),
        "comparison": "at_least",
        "threshold": 0.60,
        "observed": developing_diagnosis_min,
        "passed": developing_diagnosis_min >= 0.60,
    },
]

V5_PUBLICATION_GATE = pd.DataFrame(gate_rows)
V5_PUBLICATION_GATE["passed"] = V5_PUBLICATION_GATE.passed.astype(bool)
CELL16_GATE_STATUS = (
    "PASS" if bool(V5_PUBLICATION_GATE.passed.all()) else "REVISE"
)


# -------------------------------------------------------------------------
# 8. Leakage and integrity audit
# -------------------------------------------------------------------------

V5_LEAKAGE_AUDIT = pd.DataFrame([
    {
        "check": "segment_feature_rows",
        "observed": len(segment_features),
        "expected": 300,
        "passed": len(segment_features) == 300,
    },
    {
        "check": "unit_state_rows",
        "observed": len(unit_states),
        "expected": 60,
        "passed": len(unit_states) == 60,
    },
    {
        "check": "primary_evaluation_rows",
        "observed": len(V5_UNIT_REGISTRY),
        "expected": 40,
        "passed": len(V5_UNIT_REGISTRY) == 40,
    },
    {
        "check": "calibration_links",
        "observed": len(V5_CALIBRATION_LINKS),
        "expected": 760,
        "passed": len(V5_CALIBRATION_LINKS) == 760,
    },
    {
        "check": "shared_test_calibration_physical_units",
        "observed": int(V5_CALIBRATION_LINKS.shared_physical_unit.sum()),
        "expected": 0,
        "passed": not V5_CALIBRATION_LINKS.shared_physical_unit.any(),
    },
    {
        "check": "amended_extraction_errors",
        "observed": amended_error_count,
        "expected": 0,
        "passed": amended_error_count == 0,
    },
    {
        "check": "duplicate_segment_trial_ids",
        "observed": int(segment_features.trial_id.duplicated().sum()),
        "expected": 0,
        "passed": not segment_features.trial_id.duplicated().any(),
    },
    {
        "check": "nonfinite_primary_unit_scores",
        "observed": int((~np.isfinite(
            unit_states[primary_score_columns].to_numpy()
        )).sum()),
        "expected": 0,
        "passed": np.isfinite(
            unit_states[primary_score_columns].to_numpy()
        ).all(),
    },
])
if not V5_LEAKAGE_AUDIT.passed.all():
    raise RuntimeError("Cell 16 integrity/leakage audit failed")


# -------------------------------------------------------------------------
# 9. Immutable tables and result record
# -------------------------------------------------------------------------

def atomic_csv_write(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


table_outputs = {
    "T_uored_v5_unit_registry.csv": V5_UNIT_REGISTRY,
    "T_uored_v5_calibration_links.csv": V5_CALIBRATION_LINKS,
    "T_uored_v5_exact_intervals.csv": V5_EXACT_INTERVALS,
    "T_uored_v5_detection_2x2.csv": V5_DETECTION_2X2,
    "T_uored_v5_localization_confusion.csv": V5_LOCALIZATION_CONFUSION,
    "T_uored_v5_publication_gate.csv": V5_PUBLICATION_GATE,
    "T_uored_v5_leakage_audit.csv": V5_LEAKAGE_AUDIT,
}

table_directory = Path(MSSP_DIRS["tables"])
table_directory.mkdir(parents=True, exist_ok=True)
for filename, dataframe in table_outputs.items():
    atomic_csv_write(dataframe, CELL16_ROOT / filename)
    atomic_csv_write(dataframe, table_directory / filename)


def frame_records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(
        orient="records"
    )


result_payload = {
    "analysis_config_sha256": CELL16_CONFIG_HASH,
    "gate_status": CELL16_GATE_STATUS,
    "unit_registry": frame_records(V5_UNIT_REGISTRY),
    "exact_intervals": frame_records(V5_EXACT_INTERVALS),
    "publication_gate": frame_records(V5_PUBLICATION_GATE),
    "leakage_audit": frame_records(V5_LEAKAGE_AUDIT),
}
CELL16_RESULT_HASH = canonical_digest(result_payload)
result_record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "analysis_config_sha256": CELL16_CONFIG_HASH,
    "result_sha256": CELL16_RESULT_HASH,
    "gate_status": CELL16_GATE_STATUS,
    "result": result_payload,
}

if CELL16_RESULT_PATH.exists():
    with CELL16_RESULT_PATH.open("r", encoding="utf-8") as handle:
        prior_result_record = json.load(handle)
    if prior_result_record.get("analysis_config_sha256") != (
        CELL16_CONFIG_HASH
    ):
        raise RuntimeError("Stored Cell 16 result uses another configuration")
    if prior_result_record.get("result_sha256") != CELL16_RESULT_HASH:
        raise RuntimeError(
            "Stored Cell 16 result conflicts with the current frozen result"
        )
else:
    with CELL16_RESULT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(result_record, handle, indent=2, sort_keys=True)
        handle.write("\n")


# -------------------------------------------------------------------------
# 10. High-resolution PNG-only figures S046-S056
# -------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

default_figure_directory = (
    Path(MSSP_DIRS["statistics"]).parent /
    "08_figures_supplementary"
)
CELL16_FIGURE_DIRECTORY = Path(
    MSSP_DIRS.get("figures_supplementary", default_figure_directory)
)
CELL16_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def save_cell16_png(figure, stem):
    path = CELL16_FIGURE_DIRECTORY / f"{stem}.png"
    figure.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(figure)
    return str(path)


COLORS = {
    "healthy": "#4C78A8",
    "developing": "#F2CF5B",
    "faulty": "#E45756",
    "IR": "#E45756",
    "OR": "#54A24B",
    "pass": "#2A9D8F",
    "fail": "#D1495B",
}
CELL16_FIGURES = []


def plot_rate_metric(estimand, title, stem):
    data = V5_EXACT_INTERVALS[
        V5_EXACT_INTERVALS.estimand == estimand
    ].copy()
    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    positions = {"developing": 0, "faulty": 1}
    offsets = {"IR": -0.11, "OR": 0.11}
    for fault_label in ["IR", "OR"]:
        group = data[data.fault_label == fault_label]
        x = np.array([positions[state] for state in group.severity]) + offsets[
            fault_label
        ]
        y = group.rate.to_numpy(dtype=float)
        yerr = np.vstack([
            y - group.ci_low.to_numpy(dtype=float),
            group.ci_high.to_numpy(dtype=float) - y,
        ])
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=2,
            color=COLORS[fault_label],
            label=fault_label,
        )
        for x_value, y_value, row in zip(x, y, group.itertuples()):
            axis.text(
                x_value,
                min(1.04, y_value + 0.045),
                f"{row.successes}/{row.n}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    thresholds = {"developing": 0.60, "faulty": 0.80}
    for state, x_value in positions.items():
        axis.hlines(
            thresholds[state],
            x_value - 0.32,
            x_value + 0.32,
            color="black",
            linestyle="--",
            linewidth=1.4,
        )
    axis.set_xticks([0, 1], ["Developing", "Faulty"])
    axis.set_ylim(-0.03, 1.10)
    axis.set_ylabel("Probability with exact 95% interval")
    axis.set_title(title, fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.20)
    return save_cell16_png(figure, stem)


# S046: unit detection score and its leave-one-out threshold.
figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                            constrained_layout=True)
for axis, fault_label in zip(axes, ["IR", "OR"]):
    group = V5_UNIT_REGISTRY[
        (V5_UNIT_REGISTRY.fault_label == fault_label) |
        (
            (V5_UNIT_REGISTRY.state == "healthy") &
            (V5_UNIT_REGISTRY.unit_number.isin(
                range(1, 6) if fault_label == "IR" else range(6, 11)
            ))
        )
    ]
    for state, marker in [("healthy", "o"), ("developing", "s"),
                          ("faulty", "^")]:
        state_group = group[group.state == state]
        axis.plot(
            state_group.unit_number,
            state_group.detection_score_q90_max_z,
            marker=marker,
            linewidth=1.5,
            markersize=7,
            color=COLORS[state],
            label=state.capitalize(),
        )
    fault_group = group[group.state != "healthy"]
    for state, linestyle in [("developing", "--"), ("faulty", ":")]:
        state_group = fault_group[fault_group.state == state]
        axis.plot(
            state_group.unit_number,
            state_group.calibration_score_max,
            color="black",
            linestyle=linestyle,
            linewidth=1.2,
            label=f"{state.capitalize()} calibration maximum",
        )
    axis.set_title(f"{fault_label} physical bearings", fontweight="bold")
    axis.set_xlabel("Physical bearing number")
    axis.grid(alpha=0.20)
axes[0].set_ylabel("Unit 90th-percentile max-z")
axes[1].legend(frameon=False, fontsize=7, loc="best")
figure.suptitle("Frozen unit detection scores versus conformal-bank maxima")
CELL16_FIGURES.append(save_cell16_png(
    figure, "S046_uored_unit_detection_scores"
))


# S047: conformal probabilities.
figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,
                            constrained_layout=True)
for axis, fault_label in zip(axes, ["IR", "OR"]):
    group = V5_UNIT_REGISTRY[
        (V5_UNIT_REGISTRY.fault_label == fault_label) &
        (V5_UNIT_REGISTRY.state != "healthy")
    ]
    for state, marker in [("developing", "s"), ("faulty", "^")]:
        state_group = group[group.state == state]
        axis.plot(
            state_group.unit_number,
            state_group.conformal_p,
            marker=marker,
            linewidth=1.5,
            markersize=7,
            color=COLORS[state],
            label=state.capitalize(),
        )
    axis.axhline(0.05, color="black", linestyle="--", label="Alarm α=0.05")
    axis.set_title(f"{fault_label} physical bearings", fontweight="bold")
    axis.set_xlabel("Physical bearing number")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.20)
axes[0].set_ylabel("Leave-one-bearing-out conformal probability")
axes[1].legend(frameon=False)
figure.suptitle("Registered unit-level conformal probabilities")
CELL16_FIGURES.append(save_cell16_png(
    figure, "S047_uored_conformal_probabilities"
))


CELL16_FIGURES.append(plot_rate_metric(
    "detection",
    "Registered fault-detection probability",
    "S048_uored_detection_exact_ci",
))
CELL16_FIGURES.append(plot_rate_metric(
    "localization",
    "Physics-guided fault-order localization",
    "S049_uored_localization_exact_ci",
))
CELL16_FIGURES.append(plot_rate_metric(
    "correct_diagnosis",
    "Detection with correct fault localization",
    "S050_uored_correct_diagnosis_exact_ci",
))


# S051: localization confusion matrices.
figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2),
                            constrained_layout=True)
for axis, severity in zip(axes, ["developing", "faulty"]):
    matrix = np.zeros((2, 3), dtype=int)
    for row_index, target in enumerate(["BPFI", "BPFO"]):
        for column_index, predicted in enumerate(
            ["BPFI", "BPFO", "INDETERMINATE"]
        ):
            selected = V5_LOCALIZATION_CONFUSION[
                (V5_LOCALIZATION_CONFUSION.severity == severity) &
                (V5_LOCALIZATION_CONFUSION.target_order == target) &
                (V5_LOCALIZATION_CONFUSION.predicted_order == predicted)
            ]
            matrix[row_index, column_index] = int(selected["count"].iloc[0])
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=5)
    for row_index in range(2):
        for column_index in range(3):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color=("white" if matrix[row_index, column_index] >= 3 else
                       "black"),
                fontweight="bold",
            )
    axis.set_xticks([0, 1, 2], ["BPFI", "BPFO", "Indeterminate"], rotation=20)
    axis.set_yticks([0, 1], ["BPFI", "BPFO"])
    axis.set_xlabel("Predicted physical order")
    axis.set_ylabel("Registered target order")
    axis.set_title(severity.capitalize(), fontweight="bold")
figure.colorbar(image, ax=axes, shrink=0.85, label="Bearing count")
figure.suptitle("Unit-level localization confusion matrices")
CELL16_FIGURES.append(save_cell16_png(
    figure, "S051_uored_localization_confusion"
))


# S052: signed margins for the seven numerical gate criteria.
rate_gate = V5_PUBLICATION_GATE.iloc[2:].copy()
signed_margin = np.where(
    rate_gate.comparison == "at_most",
    rate_gate.threshold - rate_gate.observed,
    rate_gate.observed - rate_gate.threshold,
)
short_gate_names = [
    "Healthy false alarm",
    "Faulty detection",
    "Faulty localization",
    "Faulty correct diagnosis",
    "Developing detection",
    "Developing localization",
    "Developing correct diagnosis",
]
figure, axis = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
y = np.arange(len(rate_gate))
colors = [COLORS["pass"] if passed else COLORS["fail"]
          for passed in rate_gate.passed]
axis.barh(y, signed_margin, color=colors, alpha=0.88)
axis.axvline(0, color="black", linewidth=1.5)
axis.set_yticks(y, short_gate_names)
axis.invert_yaxis()
axis.set_xlabel("Signed margin to frozen threshold (positive = pass)")
axis.set_title(
    f"Registered amended external gate: {CELL16_GATE_STATUS}",
    fontweight="bold",
)
axis.grid(axis="x", alpha=0.20)
CELL16_FIGURES.append(save_cell16_png(
    figure, "S052_uored_registered_gate"
))


# S053: healthy LOBO ranks.
healthy_plot = healthy_eval.sort_values(
    "detection_score_q90_max_z"
).reset_index(drop=True)
figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
bar_colors = [COLORS["fail"] if alarm else COLORS["healthy"]
              for alarm in healthy_plot.alarm]
axis.bar(
    np.arange(len(healthy_plot)),
    healthy_plot.detection_score_q90_max_z,
    color=bar_colors,
)
axis.set_xticks(
    np.arange(len(healthy_plot)),
    healthy_plot.unit.str.replace("UORED_", "U", regex=False),
    rotation=55,
    ha="right",
)
for index, row in healthy_plot.iterrows():
    axis.text(
        index,
        row.detection_score_q90_max_z,
        f"p={row.conformal_p:.2f}",
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=7,
    )
axis.set_ylabel("Healthy unit 90th-percentile max-z")
axis.set_title("Healthy leave-one-bearing-out conformal ranks",
               fontweight="bold")
axis.grid(axis="y", alpha=0.20)
CELL16_FIGURES.append(save_cell16_png(
    figure, "S053_uored_healthy_lobo_ranks"
))


# S054: two-stage decision surface.
fault_plot = V5_UNIT_REGISTRY[
    V5_UNIT_REGISTRY.fault_label.isin(["IR", "OR"])
].copy()
figure, axis = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
markers = {"developing": "s", "faulty": "^"}
for fault_label in ["IR", "OR"]:
    for severity in ["developing", "faulty"]:
        group = fault_plot[
            (fault_plot.fault_label == fault_label) &
            (fault_plot.state == severity)
        ]
        axis.scatter(
            group.score_minus_calibration_max,
            group.order_score_difference_BPFI_minus_BPFO,
            s=95,
            marker=markers[severity],
            color=COLORS[fault_label],
            edgecolor="black",
            linewidth=0.6,
            label=f"{fault_label} {severity}",
        )
        for row in group.itertuples():
            axis.annotate(
                str(row.unit_number),
                (row.score_minus_calibration_max,
                 row.order_score_difference_BPFI_minus_BPFO),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
axis.axvline(0, color="black", linestyle="--", label="Detection boundary")
axis.axhline(0, color="grey", linestyle=":", label="Order boundary")
axis.set_xlabel("Detection score minus leave-one-out healthy maximum")
axis.set_ylabel("Median BPFI z minus median BPFO z")
axis.set_title("Two-stage bearing-level decision surface", fontweight="bold")
axis.legend(frameon=False, fontsize=8, ncol=2)
axis.grid(alpha=0.18)
CELL16_FIGURES.append(save_cell16_png(
    figure, "S054_uored_two_stage_decision_surface"
))


# S055: conformal probability heatmap for the ten primary bearings.
heat = np.full((10, 3), np.nan)
for unit_number in range(1, 11):
    unit_rows = V5_UNIT_REGISTRY[
        V5_UNIT_REGISTRY.unit_number == unit_number
    ]
    for column_index, state in enumerate(
        ["healthy", "developing", "faulty"]
    ):
        selected = unit_rows[unit_rows.state == state]
        if len(selected) == 1:
            heat[unit_number - 1, column_index] = float(
                selected.conformal_p.iloc[0]
            )
figure, axis = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
image = axis.imshow(heat, cmap="viridis_r", vmin=0.05, vmax=1.0,
                    aspect="auto")
for row_index in range(10):
    for column_index in range(3):
        value = heat[row_index, column_index]
        axis.text(
            column_index,
            row_index,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=("white" if value < 0.35 else "black"),
            fontsize=8,
        )
axis.set_xticks([0, 1, 2], ["Healthy", "Developing", "Faulty"])
axis.set_yticks(range(10), [f"UORED {unit:02d}" for unit in range(1, 11)])
axis.set_xlabel("Bearing state")
axis.set_ylabel("Primary physical bearing")
axis.set_title("Unit-level conformal probability map", fontweight="bold")
figure.colorbar(image, ax=axis, label="Conformal probability")
CELL16_FIGURES.append(save_cell16_png(
    figure, "S055_uored_conformal_probability_map"
))


# S056: empirical cumulative score distributions.
figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True,
                            constrained_layout=True)
for axis, state in zip(axes, ["healthy", "developing", "faulty"]):
    state_group = V5_UNIT_REGISTRY[V5_UNIT_REGISTRY.state == state]
    labels = ["N"] if state == "healthy" else ["IR", "OR"]
    for label in labels:
        values = np.sort(
            state_group[
                state_group.fault_label == label
            ].detection_score_q90_max_z.to_numpy(dtype=float)
        )
        probabilities = np.arange(1, len(values) + 1) / len(values)
        axis.step(
            values,
            probabilities,
            where="post",
            linewidth=2,
            color=(COLORS["healthy"] if label == "N" else COLORS[label]),
            label=label,
        )
    axis.set_title(state.capitalize(), fontweight="bold")
    axis.set_xlabel("Unit 90th-percentile max-z")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)
axes[0].set_ylabel("Empirical cumulative probability")
figure.suptitle("Bearing-level detection-score distributions")
CELL16_FIGURES.append(save_cell16_png(
    figure, "S056_uored_detection_score_ecdf"
))


# -------------------------------------------------------------------------
# 11. Complete decision report
# -------------------------------------------------------------------------

print("\n" + "=" * 112)
print("CELL 16 AMENDED UORED V5 EXTERNAL-VALIDATION DECISION REPORT")
print("=" * 112)
print(f"Analysis configuration : {CELL16_CONFIG_HASH}")
print(f"Frozen result SHA-256  : {CELL16_RESULT_HASH}")
print(f"Parent registration    : {EXPECTED_CELL13_HASH}")
print(f"Resolution amendment   : {EXPECTED_AMENDMENT_HASH}")
print(f"Segment feature source : {SOURCE_FEATURE_SHA256}")
print(f"Unit-state source      : {SOURCE_STATE_SHA256}")

print("\nIntegrity and leakage audit:")
print(V5_LEAKAGE_AUDIT.to_string(index=False))

print("\nEvery registered unit-state result:")
print(V5_UNIT_REGISTRY[
    [
        "record_id",
        "unit",
        "state",
        "fault_label",
        "detection_score_q90_max_z",
        "conformal_p",
        "alarm",
        "BPFI_z_median",
        "BPFO_z_median",
        "predicted_order",
        "localization_correct",
        "correct_diagnosis",
    ]
].to_string(index=False))

print("\nExact 95% Clopper-Pearson estimands:")
print(V5_EXACT_INTERVALS.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nRegistered 2x2 detection tables:")
print(V5_DETECTION_2X2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nLocalization confusion counts:")
print(V5_LOCALIZATION_CONFUSION.to_string(index=False))

print("\nRegistered publication gate:")
print(V5_PUBLICATION_GATE.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nAMENDED EXTERNAL PUBLICATION GATE: {CELL16_GATE_STATUS}")

print("\nSaved immutable results:")
print(f"  {CELL16_ROOT.resolve()}")
print(f"  {CELL16_RESULT_PATH.resolve()}")
print(f"  {table_directory.resolve()}")

print("\nPNG figures S046-S056:")
for path in CELL16_FIGURES:
    print(f"  {path}")

print("\nCELL 16 COMPLETE")
print(
    "Send the complete report and all S046-S056 figures. Do not change "
    "the gate after seeing this result."
)
