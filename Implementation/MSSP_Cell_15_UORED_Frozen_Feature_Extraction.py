"""
=============================================================================
MSSP UPGRADE
Cell 15: frozen UORED-VAFCLS v5 segment-feature extraction
=============================================================================

Run after Cell 14 in the same notebook session. This cell uses only the
locked Cell 14 loader specification and the unchanged v4 feature extractor.

Registered analysis unit
------------------------
Each of the 60 bearing states is divided into all complete, consecutive,
non-overlapping 25-revolution segments. At 42 kHz and the registered 1750 RPM
fallback, this gives 11 segments per state and 660 segments in total.

This cell computes:

1. the frozen v4 search/collision-matched order features for every segment;
2. the registered state-level detection statistic: 90th percentile max-z;
3. the registered state-level BPFI and BPFO localization statistics: median z.

It does NOT compute conformal probabilities, alarms, predicted fault orders,
correctness, performance metrics, publication gates or class-wise figures.
Those operations remain sealed until Cell 16.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. Locked prerequisites
# -------------------------------------------------------------------------

required_cell15_names = [
    "extract_features_v3",
    "fast_len_leq",
    "BEARINGS",
    "MSSP_DIRS",
    "V5_REGISTRATION_HASH",
    "UORED_ROOT",
    "UORED_MANIFEST",
    "UORED_MANIFEST_HASH",
    "UORED_LOADER_SPEC_HASH",
    "UORED_LOADER_SPEC_PATH",
]
missing_cell15_names = [
    name for name in required_cell15_names
    if name not in globals()
]
if missing_cell15_names:
    raise RuntimeError(
        "Run Cells 13 and 14 after the v4 extractor cells. Missing: "
        + ", ".join(missing_cell15_names)
    )

EXPECTED_CELL13_HASH = (
    "82f23198d97b4c136096a8acbd8452fc5c53fdb58da79fbcd3590c9f74ed090c"
)
if str(V5_REGISTRATION_HASH) != EXPECTED_CELL13_HASH:
    raise RuntimeError("The frozen Cell 13 registration hash has changed")

loader_spec_path = Path(UORED_LOADER_SPEC_PATH)
if not loader_spec_path.exists():
    raise FileNotFoundError(
        f"Cell 14 loader specification is missing: {loader_spec_path}"
    )
with loader_spec_path.open("r", encoding="utf-8") as handle:
    stored_loader_spec = json.load(handle)
if stored_loader_spec.get("loader_spec_sha256") != str(
    UORED_LOADER_SPEC_HASH
):
    raise RuntimeError("Stored Cell 14 loader specification has changed")
if stored_loader_spec.get("manifest_sha256") != str(UORED_MANIFEST_HASH):
    raise RuntimeError("Stored Cell 14 manifest hash has changed")

if len(UORED_MANIFEST) != 60:
    raise RuntimeError(
        f"Expected the locked 60-row manifest, found {len(UORED_MANIFEST)}"
    )
if UORED_MANIFEST.record_id.duplicated().any():
    raise RuntimeError("Duplicate record IDs in the Cell 14 manifest")


# -------------------------------------------------------------------------
# 2. Correct UORED bearing geometry and frozen extraction configuration
# -------------------------------------------------------------------------

UORED_BEARING_KEY = "UORED_6203"
UORED_GEOMETRY = (
    8,
    6.77,
    28.50,
    0.0,
    "NSK 6203ZZ / FAFNIR 203KD - UORED-VAFCLS",
)

if UORED_BEARING_KEY in BEARINGS:
    existing = BEARINGS[UORED_BEARING_KEY]
    if not (
        int(existing[0]) == UORED_GEOMETRY[0] and
        np.isclose(float(existing[1]), UORED_GEOMETRY[1]) and
        np.isclose(float(existing[2]), UORED_GEOMETRY[2]) and
        np.isclose(float(existing[3]), UORED_GEOMETRY[3])
    ):
        raise RuntimeError(
            f"Conflicting {UORED_BEARING_KEY} geometry: {existing}"
        )
else:
    BEARINGS[UORED_BEARING_KEY] = UORED_GEOMETRY


def canonical_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
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


CELL15_CONFIG = {
    "method_version": "v5.0-uored-frozen-segment-extraction",
    "source_registration_sha256": EXPECTED_CELL13_HASH,
    "source_manifest_sha256": str(UORED_MANIFEST_HASH),
    "source_loader_spec_sha256": str(UORED_LOADER_SPEC_HASH),
    "seed": 20260820,
    "sampling_rate_hz": 42000.0,
    "nominal_rpm_fallback": 1750.0,
    "bearing_key": UORED_BEARING_KEY,
    "bearing_geometry": {
        "rolling_elements": 8,
        "ball_diameter_mm": 6.77,
        "pitch_diameter_mm": 28.50,
        "contact_angle_deg": 0.0,
    },
    "segment_revolutions": 25,
    "segment_overlap_fraction": 0.0,
    "segment_selection": "all complete consecutive segments",
    "partial_final_segment": "discard",
    "minimum_complete_segments_per_state": 8,
    "n_null": 199,
    "alpha": 0.05,
    "detection_aggregation": {
        "column": "max_z",
        "statistic": "quantile",
        "probability": 0.90,
        "numpy_method": "linear",
    },
    "localization_aggregation": {
        "columns": ["BPFI_z", "BPFO_z"],
        "statistic": "median",
    },
    "checkpoint_every_segments": 20,
    "prohibited_in_cell15": [
        "conformal probability",
        "alarm",
        "predicted order",
        "correctness",
        "class-wise performance",
        "publication gate",
    ],
}
CELL15_CONFIG_HASH = canonical_digest(CELL15_CONFIG)

CELL15_ROOT = Path(MSSP_DIRS["features"]) / "uored_v5_frozen"
CELL15_ROOT.mkdir(parents=True, exist_ok=True)

CELL15_CONFIG_PATH = CELL15_ROOT / "uored_v5_extraction_config.json"
CELL15_DESIGN_PATH = CELL15_ROOT / "uored_v5_segment_registry.csv"
CELL15_FEATURE_PATH = CELL15_ROOT / "uored_v5_segment_features.csv"
CELL15_STATE_PATH = CELL15_ROOT / "uored_v5_unit_state_statistics.csv"
CELL15_ERROR_PATH = CELL15_ROOT / "uored_v5_extraction_errors.csv"

config_record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "config_sha256": CELL15_CONFIG_HASH,
    "config": CELL15_CONFIG,
}
if CELL15_CONFIG_PATH.exists():
    with CELL15_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        prior_config = json.load(handle)
    if prior_config.get("config_sha256") != CELL15_CONFIG_HASH:
        raise RuntimeError(
            "A conflicting Cell 15 extraction configuration already exists. "
            f"Do not overwrite {CELL15_CONFIG_PATH}."
        )
else:
    with CELL15_CONFIG_PATH.open("x", encoding="utf-8") as handle:
        json.dump(config_record, handle, indent=2, sort_keys=True)
        handle.write("\n")


# -------------------------------------------------------------------------
# 3. Deterministic segment registry
# -------------------------------------------------------------------------

def record_rpm(manifest_row):
    if (
        manifest_row.rpm_source == "csv_hall_effect_rpm" and
        pd.notna(manifest_row.rpm_median)
    ):
        rpm = float(manifest_row.rpm_median)
    else:
        rpm = float(CELL15_CONFIG["nominal_rpm_fallback"])
    if not np.isfinite(rpm) or rpm <= 0:
        raise ValueError(
            f"Invalid RPM for {manifest_row.record_id}: {rpm}"
        )
    return rpm


def segment_length_samples(fs, fr, revolutions):
    nominal = int(float(revolutions) * float(fs) / float(fr))
    length = int(fast_len_leq(nominal))
    if length < 1024:
        raise ValueError(
            f"Registered segment is unexpectedly short: {length} samples"
        )
    return length


segment_rows = []
for manifest_row in UORED_MANIFEST.sort_values(
    ["unit_number", "state_code"]
).itertuples(index=False):
    fs = float(CELL15_CONFIG["sampling_rate_hz"])
    rpm = record_rpm(manifest_row)
    fr = rpm / 60.0
    length = segment_length_samples(
        fs,
        fr,
        CELL15_CONFIG["segment_revolutions"],
    )
    n_samples = int(manifest_row.row_count)
    starts = np.arange(
        0,
        n_samples - length + 1,
        length,
        dtype=int,
    )
    if len(starts) < CELL15_CONFIG["minimum_complete_segments_per_state"]:
        raise RuntimeError(
            f"{manifest_row.record_id} has only {len(starts)} complete "
            "registered segments"
        )

    for segment_index, start in enumerate(starts):
        stop = int(start + length)
        trial_key = (
            f"{manifest_row.record_id}::{segment_index}::{start}::{stop}::"
            f"{CELL15_CONFIG_HASH}"
        )
        trial_id = "UORED_SEG_" + hashlib.sha256(
            trial_key.encode("utf-8")
        ).hexdigest()[:24]
        segment_rows.append({
            "trial_id": trial_id,
            "config_sha256": CELL15_CONFIG_HASH,
            "record_id": manifest_row.record_id,
            "unit": manifest_row.unit,
            "unit_number": int(manifest_row.unit_number),
            "state": manifest_row.state,
            "state_code": int(manifest_row.state_code),
            "fault_label": manifest_row.fault_label,
            "planned_cohort": manifest_row.planned_cohort,
            "target_order": manifest_row.target_order,
            "primary_test_state": bool(manifest_row.primary_test_state),
            "healthy_calibration_eligible": bool(
                manifest_row.healthy_calibration_eligible
            ),
            "relative_path": manifest_row.relative_path,
            "source_file_sha256": manifest_row.file_sha256,
            "vibration_column": manifest_row.vibration_column,
            "csv_encoding": manifest_row.csv_encoding,
            "csv_delimiter": manifest_row.csv_delimiter,
            "rpm_source": manifest_row.rpm_source,
            "fs": fs,
            "rpm": rpm,
            "fr": fr,
            "segment_index": int(segment_index),
            "segment_start": int(start),
            "segment_stop": stop,
            "segment_samples": int(length),
            "actual_revolutions": float(length * fr / fs),
        })

UORED_SEGMENT_DESIGN = pd.DataFrame(segment_rows).sort_values(
    ["unit_number", "state_code", "segment_index"]
).reset_index(drop=True)

if UORED_SEGMENT_DESIGN.trial_id.duplicated().any():
    raise RuntimeError("Duplicate trial IDs in the UORED segment registry")

segment_counts = UORED_SEGMENT_DESIGN.groupby("record_id").size()
if len(segment_counts) != 60:
    raise RuntimeError(
        f"Segment registry covers {len(segment_counts)} states, expected 60"
    )
if int(segment_counts.min()) < 8:
    raise RuntimeError("A state has fewer than eight complete segments")

UORED_SEGMENT_DESIGN_HASH = canonical_digest(
    UORED_SEGMENT_DESIGN[
        [
            "trial_id",
            "record_id",
            "unit",
            "state",
            "relative_path",
            "segment_index",
            "segment_start",
            "segment_stop",
            "source_file_sha256",
        ]
    ].to_dict(orient="records")
)
UORED_SEGMENT_DESIGN["design_sha256"] = UORED_SEGMENT_DESIGN_HASH

if CELL15_DESIGN_PATH.exists():
    prior_design = pd.read_csv(CELL15_DESIGN_PATH)
    prior_hashes = set(prior_design.design_sha256.astype(str))
    if prior_hashes != {UORED_SEGMENT_DESIGN_HASH}:
        raise RuntimeError(
            "The stored Cell 15 segment registry conflicts with the current "
            "locked design."
        )
else:
    UORED_SEGMENT_DESIGN.to_csv(CELL15_DESIGN_PATH, index=False)


# -------------------------------------------------------------------------
# 4. Resumable, source-verified feature extraction
# -------------------------------------------------------------------------

def atomic_csv_write(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


if CELL15_FEATURE_PATH.exists():
    completed_features = pd.read_csv(CELL15_FEATURE_PATH)
    if completed_features.trial_id.duplicated().any():
        raise RuntimeError("Duplicate cached Cell 15 feature trial IDs")
    cached_hashes = set(completed_features.config_sha256.astype(str))
    if cached_hashes != {CELL15_CONFIG_HASH}:
        raise RuntimeError(
            "Cached UORED features were generated by another configuration"
        )
else:
    completed_features = pd.DataFrame()

completed_ids = (
    set(completed_features.trial_id.astype(str))
    if not completed_features.empty
    else set()
)
pending_design = UORED_SEGMENT_DESIGN[
    ~UORED_SEGMENT_DESIGN.trial_id.isin(completed_ids)
].copy()

print("Frozen UORED v5 feature extraction")
print(f"  configuration SHA-256 : {CELL15_CONFIG_HASH}")
print(f"  loader-spec SHA-256   : {UORED_LOADER_SPEC_HASH}")
print(f"  segment-design SHA-256: {UORED_SEGMENT_DESIGN_HASH}")
print(f"  bearing states        : {len(segment_counts)}")
print(
    f"  segments per state    : "
    f"{int(segment_counts.min())} - {int(segment_counts.max())}"
)
print(f"  registered segments   : {len(UORED_SEGMENT_DESIGN)}")
print(f"  cache contains        : {len(completed_ids)}")
print(f"  pending               : {len(pending_design)}")
print(f"  output                : {CELL15_ROOT.resolve()}")
print("  outcomes/gates        : NOT COMPUTED\n")

new_feature_rows = []
error_rows = []
run_start = time.time()
processed_pending = 0

pending_by_record = {
    record_id: group.sort_values("segment_index")
    for record_id, group in pending_design.groupby("record_id", sort=True)
}

for record_id, record_segments in pending_by_record.items():
    first = record_segments.iloc[0]
    source_path = Path(UORED_ROOT) / str(first.relative_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing locked source: {source_path}")
        current_hash = sha256_file(source_path)
        if current_hash != str(first.source_file_sha256):
            raise RuntimeError(
                f"Source hash changed for {record_id}: "
                f"expected {first.source_file_sha256}, got {current_hash}"
            )

        raw = pd.read_csv(
            source_path,
            sep=str(first.csv_delimiter),
            encoding=str(first.csv_encoding),
            usecols=[str(first.vibration_column)],
            low_memory=False,
        )
        signal = pd.to_numeric(
            raw[str(first.vibration_column)],
            errors="coerce",
        ).to_numpy(dtype=float)

        expected_rows = int(
            UORED_MANIFEST.loc[
                UORED_MANIFEST.record_id == record_id,
                "row_count",
            ].iloc[0]
        )
        if len(signal) != expected_rows:
            raise RuntimeError(
                f"Row count changed for {record_id}: "
                f"expected {expected_rows}, found {len(signal)}"
            )
        if not np.isfinite(signal).all():
            raise RuntimeError(f"Non-finite vibration signal: {record_id}")

        for trial in record_segments.itertuples(index=False):
            try:
                segment = signal[
                    int(trial.segment_start):int(trial.segment_stop)
                ]
                if len(segment) != int(trial.segment_samples):
                    raise RuntimeError(
                        f"Incomplete segment for {trial.trial_id}"
                    )

                extraction_start = time.time()
                features = extract_features_v3(
                    sig=segment,
                    fs=float(trial.fs),
                    fr=float(trial.fr),
                    bearing_key=UORED_BEARING_KEY,
                    seed_str=f"UORED-V5::{trial.trial_id}",
                    n_null=CELL15_CONFIG["n_null"],
                    alpha=CELL15_CONFIG["alpha"],
                    return_detail=False,
                )
                extraction_seconds = time.time() - extraction_start

                result = trial._asdict()
                result["bearing_key"] = UORED_BEARING_KEY
                result["extract_seconds"] = float(extraction_seconds)
                result.update(features)
                new_feature_rows.append(result)

            except Exception as exc:
                error_rows.append({
                    "trial_id": trial.trial_id,
                    "record_id": record_id,
                    "relative_path": trial.relative_path,
                    "segment_index": int(trial.segment_index),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

            processed_pending += 1
            if (
                processed_pending % CELL15_CONFIG[
                    "checkpoint_every_segments"
                ] == 0 or
                processed_pending == len(pending_design)
            ):
                elapsed = time.time() - run_start
                rate = processed_pending / max(elapsed, 1e-12)
                remaining = len(pending_design) - processed_pending
                eta_minutes = remaining / max(rate, 1e-12) / 60.0
                print(
                    f"  {processed_pending:4d}/{len(pending_design)} "
                    f"pending segments  elapsed={elapsed/60:7.2f} min  "
                    f"ETA={eta_minutes:7.2f} min  "
                    f"errors={len(error_rows)}"
                )

                frames = []
                if not completed_features.empty:
                    frames.append(completed_features)
                if new_feature_rows:
                    frames.append(pd.DataFrame(new_feature_rows))
                if frames:
                    checkpoint = pd.concat(
                        frames,
                        ignore_index=True,
                        sort=False,
                    ).drop_duplicates("trial_id", keep="last")
                    atomic_csv_write(checkpoint, CELL15_FEATURE_PATH)
                if error_rows:
                    atomic_csv_write(
                        pd.DataFrame(error_rows),
                        CELL15_ERROR_PATH,
                    )

    except Exception as exc:
        for trial in record_segments.itertuples(index=False):
            if trial.trial_id not in {
                row["trial_id"] for row in error_rows
            }:
                error_rows.append({
                    "trial_id": trial.trial_id,
                    "record_id": record_id,
                    "relative_path": trial.relative_path,
                    "segment_index": int(trial.segment_index),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
                processed_pending += 1

if error_rows:
    error_frame = pd.DataFrame(error_rows).drop_duplicates(
        "trial_id",
        keep="last",
    )
    atomic_csv_write(error_frame, CELL15_ERROR_PATH)
    print("\nUORED EXTRACTION ERRORS")
    print(error_frame.head(20).to_string(index=False))
    raise RuntimeError(
        f"{len(error_frame)} registered UORED segments failed. No source "
        "exclusion or imputation is permitted. Inspect {CELL15_ERROR_PATH}."
    )

if not CELL15_FEATURE_PATH.exists():
    raise RuntimeError("No UORED segment feature cache was produced")

UORED_SEGMENT_FEATURES = pd.read_csv(CELL15_FEATURE_PATH)
if UORED_SEGMENT_FEATURES.trial_id.duplicated().any():
    raise RuntimeError("Duplicate trial IDs in final UORED feature cache")

expected_trial_ids = set(UORED_SEGMENT_DESIGN.trial_id)
actual_trial_ids = set(UORED_SEGMENT_FEATURES.trial_id)
if actual_trial_ids != expected_trial_ids:
    missing = sorted(expected_trial_ids - actual_trial_ids)
    extra = sorted(actual_trial_ids - expected_trial_ids)
    raise RuntimeError(
        "Final UORED feature cache does not match the locked registry. "
        f"Missing={len(missing)}, extra={len(extra)}"
    )


# -------------------------------------------------------------------------
# 5. Registered unit-state aggregates, still without predictions
# -------------------------------------------------------------------------

required_feature_columns = ["max_z", "BPFI_z", "BPFO_z", "argmax_order"]
missing_feature_columns = [
    column for column in required_feature_columns
    if column not in UORED_SEGMENT_FEATURES.columns
]
if missing_feature_columns:
    raise RuntimeError(
        "Frozen extractor omitted required UORED columns: "
        + ", ".join(missing_feature_columns)
    )

for column in ["max_z", "BPFI_z", "BPFO_z"]:
    values = pd.to_numeric(
        UORED_SEGMENT_FEATURES[column],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite primary feature column: {column}")


def q90_linear(series):
    return float(np.quantile(
        np.asarray(series, dtype=float),
        0.90,
        method="linear",
    ))


state_rows = []
group_columns = [
    "record_id",
    "unit",
    "unit_number",
    "state",
    "state_code",
    "fault_label",
    "planned_cohort",
    "target_order",
    "primary_test_state",
    "healthy_calibration_eligible",
]
for keys, group in UORED_SEGMENT_FEATURES.groupby(
    group_columns,
    dropna=False,
    sort=True,
):
    metadata = dict(zip(group_columns, keys))
    state_rows.append({
        **metadata,
        "n_segments": int(len(group)),
        "unit_detection_max_z_q90": q90_linear(group.max_z),
        "unit_BPFI_z_median": float(np.median(group.BPFI_z)),
        "unit_BPFO_z_median": float(np.median(group.BPFO_z)),
        "aggregation_config_sha256": CELL15_CONFIG_HASH,
        "segment_design_sha256": UORED_SEGMENT_DESIGN_HASH,
    })

UORED_UNIT_STATE_STATISTICS = pd.DataFrame(state_rows).sort_values(
    ["unit_number", "state_code"]
).reset_index(drop=True)

if len(UORED_UNIT_STATE_STATISTICS) != 60:
    raise RuntimeError(
        "Expected exactly 60 registered unit-state aggregate rows, found "
        f"{len(UORED_UNIT_STATE_STATISTICS)}"
    )
if UORED_UNIT_STATE_STATISTICS.record_id.duplicated().any():
    raise RuntimeError("Duplicate unit-state aggregate record IDs")

atomic_csv_write(UORED_UNIT_STATE_STATISTICS, CELL15_STATE_PATH)


# -------------------------------------------------------------------------
# 6. Outcome-blind integrity report
# -------------------------------------------------------------------------

metadata_columns = set(UORED_SEGMENT_DESIGN.columns)
numeric_feature_columns = [
    column for column in UORED_SEGMENT_FEATURES.select_dtypes(
        include=[np.number]
    ).columns
    if column not in metadata_columns
]
all_nan_columns = [
    column for column in numeric_feature_columns
    if UORED_SEGMENT_FEATURES[column].isna().all()
]
unexpected_all_nan = [
    column for column in all_nan_columns
    if not column.startswith("BSF_")
]
if unexpected_all_nan:
    raise RuntimeError(
        "Unexpected all-NaN extracted feature columns: "
        + ", ".join(unexpected_all_nan)
    )

primary_nan_cells = int(
    UORED_SEGMENT_FEATURES[["max_z", "BPFI_z", "BPFO_z"]]
    .isna()
    .sum()
    .sum()
)
primary_inf_cells = int(np.isinf(
    UORED_SEGMENT_FEATURES[["max_z", "BPFI_z", "BPFO_z"]]
    .to_numpy(dtype=float)
).sum())

final_counts = UORED_SEGMENT_FEATURES.groupby("record_id").size()
elapsed_total = float(UORED_SEGMENT_FEATURES.extract_seconds.sum())

print("\n" + "=" * 104)
print("CELL 15 FROZEN UORED FEATURE-EXTRACTION REPORT")
print("=" * 104)
print(f"Configuration SHA-256  : {CELL15_CONFIG_HASH}")
print(f"Segment design SHA-256 : {UORED_SEGMENT_DESIGN_HASH}")
print(f"Bearing states         : {UORED_SEGMENT_FEATURES.record_id.nunique()}")
print(f"Physical bearings      : {UORED_SEGMENT_FEATURES.unit.nunique()}")
print(f"Extracted segments     : {len(UORED_SEGMENT_FEATURES)}")
print(
    f"Segments per state     : "
    f"{int(final_counts.min())} - {int(final_counts.max())}"
)
print(
    f"Segment samples        : "
    f"{int(UORED_SEGMENT_FEATURES.segment_samples.min())} - "
    f"{int(UORED_SEGMENT_FEATURES.segment_samples.max())}"
)
print(
    f"Actual revolutions     : "
    f"{UORED_SEGMENT_FEATURES.actual_revolutions.min():.6f} - "
    f"{UORED_SEGMENT_FEATURES.actual_revolutions.max():.6f}"
)
print(
    f"Median extraction time : "
    f"{UORED_SEGMENT_FEATURES.extract_seconds.median():.4f} s/segment"
)
print(f"Total extraction time  : {elapsed_total/60.0:.2f} min")
print(f"Primary NaN cells      : {primary_nan_cells}")
print(f"Primary Inf cells      : {primary_inf_cells}")
print(
    "Resolution exclusions : "
    + (", ".join(all_nan_columns) if all_nan_columns else "none")
)
print(f"Unit-state rows        : {len(UORED_UNIT_STATE_STATISTICS)}")
print("Conformal probabilities: NOT COMPUTED")
print("Predicted fault orders : NOT COMPUTED")
print("Performance/gates      : NOT COMPUTED")

print("\nSaved frozen features:")
print(f"  {CELL15_DESIGN_PATH.resolve()}")
print(f"  {CELL15_FEATURE_PATH.resolve()}")
print(f"  {CELL15_STATE_PATH.resolve()}")

print("\nCELL 15 COMPLETE")
print(
    "Send the complete report. Cell 16 will apply the locked leave-one-"
    "bearing-out conformal probabilities and evaluate the registered v5 "
    "external publication gate once."
)
