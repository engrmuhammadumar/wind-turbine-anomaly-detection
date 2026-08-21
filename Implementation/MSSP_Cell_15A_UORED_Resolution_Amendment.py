"""
=============================================================================
MSSP UPGRADE
Cell 15A: outcome-blind UORED resolution-feasibility amendment
=============================================================================

Run after the failed original Cell 15 extraction.

Why this amendment is necessary
--------------------------------
The registered 25-revolution window gives only 1.2456 FFT-bin separation
between the UORED 6203 BPFO/BPFI orders and the nearest integer shaft
harmonics.  The already-frozen v4 collision rule requires at least 2 bins.
Consequently, BPFO, BPFI and BSF are all excluded and the old extractor
eventually calls np.concatenate on an empty set of fault-order nulls.

The failure occurred for every segment before any valid fault-order feature,
conformal probability, alarm, prediction, class-wise metric or gate was
produced.  This cell preserves that failure and registers a minimal
implementation-feasibility amendment.  It does not read raw signals or
compute spectra, features or outcomes.

The amendment keeps the v4 extractor and all collision/null parameters
unchanged.  It selects the smallest integer revolution count above 25 that:

1. remains exactly that many revolutions after fast-length selection;
2. passes the frozen collision rule for both BPFI and BPFO; and
3. leaves at least five complete non-overlapping segments in 420,000 samples.

For the registered UORED geometry and acquisition settings, this is 54
revolutions (77,760 samples), giving five segments per state.  BSF remains
excluded by the original resolution rule and ball/cage faults remain outside
the registered primary gate.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. Locked prerequisites and failed-run evidence
# -------------------------------------------------------------------------

required_cell15a_names = [
    "fast_len_leq",
    "MSSP_DIRS",
    "V5_REGISTRATION_HASH",
    "UORED_MANIFEST_HASH",
    "UORED_LOADER_SPEC_HASH",
    "UORED_MANIFEST",
]
missing_cell15a_names = [
    name for name in required_cell15a_names
    if name not in globals()
]
if missing_cell15a_names:
    raise RuntimeError(
        "Run Cells 13 and 14, then the failed original Cell 15. Missing: "
        + ", ".join(missing_cell15a_names)
    )

EXPECTED_CELL13_HASH = (
    "82f23198d97b4c136096a8acbd8452fc5c53fdb58da79fbcd3590c9f74ed090c"
)
EXPECTED_MANIFEST_HASH = (
    "41cb8a6870181556a2ae3b2636321ead9dd8f066def82dfc5bba98f6fec01018"
)
EXPECTED_LOADER_HASH = (
    "5782fe0afc5a824b252d46adde3338e2dffeaf44a55872ee13829186464ff55d"
)

if str(V5_REGISTRATION_HASH) != EXPECTED_CELL13_HASH:
    raise RuntimeError("The Cell 13 registration hash has changed")
if str(UORED_MANIFEST_HASH) != EXPECTED_MANIFEST_HASH:
    raise RuntimeError("The Cell 14 manifest hash has changed")
if str(UORED_LOADER_SPEC_HASH) != EXPECTED_LOADER_HASH:
    raise RuntimeError("The Cell 14 loader-spec hash has changed")
if len(UORED_MANIFEST) != 60:
    raise RuntimeError("The locked UORED manifest is not the expected 60 rows")

FAILED_CELL15_ROOT = Path(MSSP_DIRS["features"]) / "uored_v5_frozen"
FAILED_CELL15_ERROR_PATH = (
    FAILED_CELL15_ROOT / "uored_v5_extraction_errors.csv"
)
FAILED_CELL15_FEATURE_PATH = (
    FAILED_CELL15_ROOT / "uored_v5_segment_features.csv"
)

if not FAILED_CELL15_ERROR_PATH.exists():
    raise FileNotFoundError(
        "The original Cell 15 failure log is required for the amendment: "
        f"{FAILED_CELL15_ERROR_PATH}"
    )

failed_errors = pd.read_csv(FAILED_CELL15_ERROR_PATH)
required_error_columns = {"trial_id", "error_type", "error_message"}
if not required_error_columns.issubset(failed_errors.columns):
    raise RuntimeError("The original Cell 15 error log has changed schema")
if len(failed_errors) != 660:
    raise RuntimeError(
        f"Expected 660 failed registered segments, found {len(failed_errors)}"
    )
if failed_errors.trial_id.duplicated().any():
    raise RuntimeError("Duplicate trial IDs in the failed Cell 15 log")

failure_pairs = set(zip(
    failed_errors.error_type.astype(str),
    failed_errors.error_message.astype(str),
))
expected_failure = {
    ("ValueError", "need at least one array to concatenate")
}
if failure_pairs != expected_failure:
    raise RuntimeError(
        "The observed Cell 15 failure is not the registered empty-order "
        f"failure: {sorted(failure_pairs)}"
    )

if FAILED_CELL15_FEATURE_PATH.exists():
    old_features = pd.read_csv(FAILED_CELL15_FEATURE_PATH)
    if len(old_features) != 0:
        raise RuntimeError(
            "The old run contains valid feature rows. Stop: this amendment "
            "is permitted only before any valid UORED features/outcomes."
        )


# -------------------------------------------------------------------------
# 2. Deterministic resolution calculation
# -------------------------------------------------------------------------

def canonical_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FS = 42000.0
RPM = 1750.0
FR = RPM / 60.0
N_SAMPLES_PER_STATE = 420000
N_ROLLING_ELEMENTS = 8
BALL_DIAMETER_MM = 6.77
PITCH_DIAMETER_MM = 28.50
CONTACT_ANGLE_DEG = 0.0
ORDER_TOLERANCE = 0.03
MIN_COLLISION_BINS = 2.0
COLLISION_GUARD_BINS = 1.0

ratio = (
    BALL_DIAMETER_MM / PITCH_DIAMETER_MM *
    np.cos(np.deg2rad(CONTACT_ANGLE_DEG))
)
CHARACTERISTIC_ORDERS = {
    "FTF": 0.5 * (1.0 - ratio),
    "BPFO": 0.5 * N_ROLLING_ELEMENTS * (1.0 - ratio),
    "BPFI": 0.5 * N_ROLLING_ELEMENTS * (1.0 + ratio),
    "BSF": (
        PITCH_DIAMETER_MM / (2.0 * BALL_DIAMETER_MM) *
        (1.0 - ratio ** 2)
    ),
}


def resolution_row(nominal_revolutions):
    nominal_samples = int(nominal_revolutions * FS / FR)
    samples = int(fast_len_leq(nominal_samples))
    effective_revolutions = samples * FR / FS
    df_hz = FS / samples
    half_window_bins = ORDER_TOLERANCE * effective_revolutions
    required_bins = max(
        MIN_COLLISION_BINS,
        half_window_bins + COLLISION_GUARD_BINS,
    )
    result = {
        "nominal_revolutions": int(nominal_revolutions),
        "segment_samples": samples,
        "effective_revolutions": float(effective_revolutions),
        "df_hz": float(df_hz),
        "complete_segments_per_state": int(
            N_SAMPLES_PER_STATE // samples
        ),
        "required_collision_bins": float(required_bins),
    }
    for name, order in CHARACTERISTIC_ORDERS.items():
        nearest_integer = max(1, int(round(order)))
        separation_order = abs(order - nearest_integer)
        separation_bins = separation_order * effective_revolutions
        result[f"{name}_order"] = float(order)
        result[f"{name}_nearest_integer"] = nearest_integer
        result[f"{name}_separation_order"] = float(separation_order)
        result[f"{name}_separation_bins"] = float(separation_bins)
        result[f"{name}_passes"] = bool(
            name == "FTF" or separation_bins >= required_bins
        )
    return result


ORIGINAL_RESOLUTION = resolution_row(25)

eligible_amendments = []
for candidate_revolutions in range(26, 201):
    diagnostics = resolution_row(candidate_revolutions)
    exact_revolutions = np.isclose(
        diagnostics["effective_revolutions"],
        float(candidate_revolutions),
        rtol=0.0,
        atol=1e-12,
    )
    if (
        exact_revolutions and
        diagnostics["BPFI_passes"] and
        diagnostics["BPFO_passes"] and
        diagnostics["complete_segments_per_state"] >= 5
    ):
        eligible_amendments.append(diagnostics)

if not eligible_amendments:
    raise RuntimeError("No feasible outcome-blind resolution amendment found")

AMENDED_RESOLUTION = eligible_amendments[0]
if AMENDED_RESOLUTION["nominal_revolutions"] != 54:
    raise RuntimeError(
        "The deterministic amendment is no longer 54 revolutions; audit "
        "fast_len_leq and the frozen geometry before proceeding."
    )

if ORIGINAL_RESOLUTION["BPFI_passes"]:
    raise RuntimeError("Unexpectedly, BPFI passes at 25 revolutions")
if ORIGINAL_RESOLUTION["BPFO_passes"]:
    raise RuntimeError("Unexpectedly, BPFO passes at 25 revolutions")
if not AMENDED_RESOLUTION["BPFI_passes"]:
    raise RuntimeError("BPFI does not pass the amended resolution check")
if not AMENDED_RESOLUTION["BPFO_passes"]:
    raise RuntimeError("BPFO does not pass the amended resolution check")
if AMENDED_RESOLUTION["BSF_passes"]:
    raise RuntimeError("BSF unexpectedly passes the frozen collision rule")


# -------------------------------------------------------------------------
# 3. Locked amendment
# -------------------------------------------------------------------------

V5_RESOLUTION_AMENDMENT = {
    "amendment_id": "v5-resolution-amendment-001",
    "status": "locked_before_valid_uored_features_or_outcomes",
    "parent_registration_sha256": EXPECTED_CELL13_HASH,
    "uored_manifest_sha256": EXPECTED_MANIFEST_HASH,
    "uored_loader_spec_sha256": EXPECTED_LOADER_HASH,
    "trigger": {
        "stage": "implementation feasibility check",
        "failed_registered_segments": 660,
        "unique_error": (
            "ValueError: need at least one array to concatenate"
        ),
        "root_cause": (
            "At 25 revolutions the frozen resolution/collision rule drops "
            "BPFI, BPFO and BSF, leaving no primary fault order to pool."
        ),
        "valid_segment_features_produced": 0,
        "conformal_probabilities_computed": False,
        "alarms_or_predictions_computed": False,
        "classwise_metrics_computed": False,
        "publication_gate_computed": False,
    },
    "frozen_inputs": {
        "sampling_rate_hz": FS,
        "nominal_speed_rpm": RPM,
        "samples_per_state": N_SAMPLES_PER_STATE,
        "bearing_geometry": {
            "rolling_elements": N_ROLLING_ELEMENTS,
            "ball_diameter_mm": BALL_DIAMETER_MM,
            "pitch_diameter_mm": PITCH_DIAMETER_MM,
            "contact_angle_deg": CONTACT_ANGLE_DEG,
        },
        "order_tolerance": ORDER_TOLERANCE,
        "minimum_collision_bins": MIN_COLLISION_BINS,
        "collision_guard_bins": COLLISION_GUARD_BINS,
    },
    "original_resolution": ORIGINAL_RESOLUTION,
    "selection_rule": (
        "Choose the smallest integer revolution count above 25 whose "
        "fast-length segment has exactly the requested revolutions, retains "
        "BPFI and BPFO under the unchanged collision rule, and provides at "
        "least five complete non-overlapping segments per state."
    ),
    "amended_resolution": AMENDED_RESOLUTION,
    "protocol_change": {
        "segment_length_revolutions": {"from": 25, "to": 54},
        "segment_samples": {"from": 36000, "to": 77760},
        "minimum_complete_segments_per_state": {"from": 8, "to": 5},
        "expected_complete_segments_per_state": {"from": 11, "to": 5},
        "new_extraction_directory": (
            "uored_v5_frozen_resolution_amended"
        ),
        "preserve_failed_original_directory": True,
    },
    "unchanged": [
        "bearing-level inference unit",
        "non-overlapping consecutive segmentation",
        "frozen v4 feature extractor",
        "carrier-band search",
        "order tolerance and collision rule",
        "search- and collision-matched empirical null",
        "199 null orders per retained target",
        "90th-percentile max-z unit detection statistic",
        "median BPFI/BPFO z localization statistics",
        "leave-one-bearing-out healthy conformal calibration",
        "alpha=0.05",
        "all registered estimands and publication gates",
    ],
    "scope": {
        "primary_confirmatory_orders": ["BPFI", "BPFO"],
        "BSF_status": (
            "still excluded by the unchanged resolution/collision rule"
        ),
        "ball_and_cage_states": (
            "secondary exploratory only; excluded from the primary gate"
        ),
    },
    "reporting_requirement": (
        "Report the failed 25-revolution implementation and this amendment "
        "in the manuscript/protocol supplement. Do not describe the 54-"
        "revolution analysis as the unchanged original registration."
    ),
}

V5_RESOLUTION_AMENDMENT_HASH = canonical_digest(V5_RESOLUTION_AMENDMENT)

CELL15A_ROOT = (
    Path(MSSP_DIRS["statistics"]) /
    "v5_external_validation_registration"
)
CELL15A_ROOT.mkdir(parents=True, exist_ok=True)
V5_RESOLUTION_AMENDMENT_PATH = (
    CELL15A_ROOT / "v5_resolution_amendment_001.json"
)
V5_RESOLUTION_AMENDMENT_SHA_PATH = (
    CELL15A_ROOT / "v5_resolution_amendment_001.sha256"
)

amendment_record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "amendment_sha256": V5_RESOLUTION_AMENDMENT_HASH,
    "amendment": V5_RESOLUTION_AMENDMENT,
}

if V5_RESOLUTION_AMENDMENT_PATH.exists():
    with V5_RESOLUTION_AMENDMENT_PATH.open(
        "r", encoding="utf-8"
    ) as handle:
        stored_record = json.load(handle)
    if stored_record.get("amendment_sha256") != (
        V5_RESOLUTION_AMENDMENT_HASH
    ):
        raise RuntimeError(
            "A conflicting resolution amendment already exists. Do not "
            f"overwrite {V5_RESOLUTION_AMENDMENT_PATH}."
        )
else:
    with V5_RESOLUTION_AMENDMENT_PATH.open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(amendment_record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with V5_RESOLUTION_AMENDMENT_SHA_PATH.open(
        "x", encoding="utf-8"
    ) as handle:
        handle.write(
            f"{V5_RESOLUTION_AMENDMENT_HASH}  "
            f"{V5_RESOLUTION_AMENDMENT_PATH.name}\n"
        )

with V5_RESOLUTION_AMENDMENT_PATH.open(
    "r", encoding="utf-8"
) as handle:
    stored_record = json.load(handle)
if canonical_digest(stored_record["amendment"]) != (
    V5_RESOLUTION_AMENDMENT_HASH
):
    raise RuntimeError("Stored resolution amendment failed its hash check")


# -------------------------------------------------------------------------
# 4. Decision report
# -------------------------------------------------------------------------

print("\n" + "=" * 108)
print("CELL 15A UORED RESOLUTION-FEASIBILITY AMENDMENT")
print("=" * 108)
print("Status                  : LOCKED BEFORE VALID FEATURES/OUTCOMES")
print(f"Parent registration     : {EXPECTED_CELL13_HASH}")
print(f"Amendment SHA-256       : {V5_RESOLUTION_AMENDMENT_HASH}")
print(f"Saved amendment         : {V5_RESOLUTION_AMENDMENT_PATH.resolve()}")

print("\nOriginal 25-revolution feasibility:")
print(
    f"  segment samples       : "
    f"{ORIGINAL_RESOLUTION['segment_samples']}"
)
print(
    f"  frequency resolution  : "
    f"{ORIGINAL_RESOLUTION['df_hz']:.6f} Hz"
)
print(
    f"  BPFI/BPFO separation  : "
    f"{ORIGINAL_RESOLUTION['BPFI_separation_bins']:.6f} bins"
)
print(
    f"  required separation   : "
    f"{ORIGINAL_RESOLUTION['required_collision_bins']:.6f} bins"
)
print("  BPFI/BPFO usable      : False / False")
print("  valid feature rows    : 0")
print("  outcomes/gates        : NOT COMPUTED")

print("\nRegistered amendment:")
print(
    f"  nominal revolutions   : "
    f"{AMENDED_RESOLUTION['nominal_revolutions']}"
)
print(
    f"  effective revolutions : "
    f"{AMENDED_RESOLUTION['effective_revolutions']:.6f}"
)
print(
    f"  segment samples       : "
    f"{AMENDED_RESOLUTION['segment_samples']}"
)
print(
    f"  segments per state    : "
    f"{AMENDED_RESOLUTION['complete_segments_per_state']}"
)
print(
    f"  BPFI/BPFO separation  : "
    f"{AMENDED_RESOLUTION['BPFI_separation_bins']:.6f} bins"
)
print(
    f"  required separation   : "
    f"{AMENDED_RESOLUTION['required_collision_bins']:.6f} bins"
)
print("  BPFI/BPFO usable      : True / True")
print("  BSF usable            : False (unchanged resolution exclusion)")
print("  extractor/null/gates  : UNCHANGED")
print("  failed output retained: True")

print("\nCELL 15A COMPLETE")
print(
    "Run Cell 15B next. It will write to a new resolution-amended "
    "directory and will still compute no predictions or performance gate."
)
