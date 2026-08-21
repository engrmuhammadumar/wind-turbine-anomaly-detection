"""
=============================================================================
MSSP UPGRADE
Cell 14: UORED-VAFCLS file manifest, schema audit and loader freeze
=============================================================================

Run after Cell 13 and after downloading/extracting the RAW CSV representation
of UORED-VAFCLS version 5 (DOI 10.17632/y2px5tg92h.5).

This cell is intentionally outcome-blind.  It:

1. constructs the expected 20-bearing x 3-state registry;
2. resolves exactly one raw CSV file for each of the 60 bearing states;
3. records file hashes, row counts and sensor-column mappings;
4. validates the vibration and optional RPM channels;
5. writes and hashes an immutable loader specification.

It does NOT calculate spectra, envelopes, order features, conformal scores,
predictions or performance metrics.  It creates no figures.

Before running, edit only UORED_ROOT below.  If automatic sensor-column
identification fails, use the two optional column overrides after inspecting
the schema error printed by this cell.  Do not inspect spectra or class-wise
signal summaries to choose the columns.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import hashlib
import json
import re

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. User path and optional schema-only overrides
# -------------------------------------------------------------------------

# EDIT THIS ONE PATH. It must contain the extracted raw CSV files.
UORED_ROOT = Path(globals().get(
    "UORED_ROOT",
    r"F:\Umar-Wisal-Work\Datasets\UORED-VAFCLS-v5",
))

# Leave these as None unless Cell 14 reports ambiguous/missing column names.
# Overrides must exactly match a CSV header and apply to all 60 raw files.
UORED_VIBRATION_COLUMN_OVERRIDE = globals().get(
    "UORED_VIBRATION_COLUMN_OVERRIDE",
    None,
)
UORED_RPM_COLUMN_OVERRIDE = globals().get(
    "UORED_RPM_COLUMN_OVERRIDE",
    None,
)


# -------------------------------------------------------------------------
# 2. Locked Cell 13 prerequisite
# -------------------------------------------------------------------------

required_cell14_names = [
    "V5_REGISTRATION_HASH",
    "CELL13_REGISTRATION_PATH",
    "CELL13_ROOT",
    "MSSP_DIRS",
]
missing_cell14_names = [
    name for name in required_cell14_names
    if name not in globals()
]
if missing_cell14_names:
    raise RuntimeError(
        "Run Cell 13 first. Missing: "
        + ", ".join(missing_cell14_names)
    )

EXPECTED_CELL13_HASH = (
    "82f23198d97b4c136096a8acbd8452fc5c53fdb58da79fbcd3590c9f74ed090c"
)
if str(V5_REGISTRATION_HASH) != EXPECTED_CELL13_HASH:
    raise RuntimeError(
        "Cell 13 registration hash does not match the frozen v5 protocol. "
        f"Expected {EXPECTED_CELL13_HASH}, got {V5_REGISTRATION_HASH}."
    )

cell13_path = Path(CELL13_REGISTRATION_PATH)
if not cell13_path.exists():
    raise FileNotFoundError(
        f"Stored Cell 13 registration is missing: {cell13_path}"
    )

with cell13_path.open("r", encoding="utf-8") as handle:
    cell13_record = json.load(handle)
if cell13_record.get("registration_sha256") != EXPECTED_CELL13_HASH:
    raise RuntimeError("Stored Cell 13 registration failed its lock check")

if not UORED_ROOT.exists():
    raise FileNotFoundError(
        "UORED_ROOT does not exist. Download UORED-VAFCLS version 5, "
        "extract the raw CSV archive, and edit UORED_ROOT.\n"
        f"Current path: {UORED_ROOT}"
    )


# -------------------------------------------------------------------------
# 3. Helpers
# -------------------------------------------------------------------------

def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_column_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def clean_relative_path(path):
    return Path(path).resolve().relative_to(UORED_ROOT.resolve()).as_posix()


def detect_text_encoding(path):
    raw = Path(path).read_bytes()[:65536]
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not determine text encoding: {path}")


def detect_delimiter(path, encoding):
    with Path(path).open("r", encoding=encoding, errors="strict") as handle:
        sample = handle.read(65536)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def header_looks_numeric(columns):
    numeric = 0
    for column in columns:
        try:
            float(str(column).strip())
            numeric += 1
        except ValueError:
            pass
    return numeric == len(columns)


def choose_named_column(columns, kind, override=None):
    columns = [str(column) for column in columns]
    if override is not None:
        if str(override) not in columns:
            raise RuntimeError(
                f"{kind} override {override!r} is not in columns {columns}"
            )
        return str(override)

    normalized = {
        column: normalized_column_name(column)
        for column in columns
    }

    if kind == "vibration":
        ranked_patterns = [
            r"accelerometer",
            r"acceleration",
            r"vibration",
            r"(^|_)acc($|_)",
            r"(^|_)vib($|_)",
        ]
        exclusions = [
            r"acoustic", r"microphone", r"sound", r"rpm", r"speed",
            r"load", r"temperature", r"thermo", r"time", r"index",
        ]
    elif kind == "rpm":
        ranked_patterns = [
            r"(^|_)rpm($|_)",
            r"rotational.*speed",
            r"shaft.*speed",
            r"speed.*rpm",
        ]
        exclusions = []
    else:
        raise ValueError(f"Unknown column kind: {kind}")

    for pattern in ranked_patterns:
        matches = []
        for column, name in normalized.items():
            if re.search(pattern, name) and not any(
                re.search(exclusion, name)
                for exclusion in exclusions
            ):
                matches.append(column)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous {kind} columns for pattern {pattern!r}: "
                f"{matches}"
            )

    if kind == "rpm":
        return None
    raise RuntimeError(
        f"No named {kind} column found in columns: {columns}"
    )


def inspect_csv_schema(path):
    encoding = detect_text_encoding(path)
    delimiter = detect_delimiter(path, encoding)
    sample = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        nrows=512,
        low_memory=False,
    )
    if sample.shape[0] < 100:
        raise RuntimeError(
            f"CSV has fewer than 100 sample rows: {path}"
        )
    if header_looks_numeric(sample.columns):
        raise RuntimeError(
            "CSV appears to have no textual header. A column-order guess is "
            f"not permitted by Cell 14: {path}"
        )

    columns = [str(column) for column in sample.columns]
    vibration_column = choose_named_column(
        columns,
        "vibration",
        UORED_VIBRATION_COLUMN_OVERRIDE,
    )
    rpm_column = choose_named_column(
        columns,
        "rpm",
        UORED_RPM_COLUMN_OVERRIDE,
    )

    vibration_sample = pd.to_numeric(
        sample[vibration_column],
        errors="coerce",
    )
    if vibration_sample.notna().mean() < 0.99:
        raise RuntimeError(
            f"Vibration column is not at least 99% numeric in {path}: "
            f"{vibration_column!r}"
        )

    if rpm_column is not None:
        rpm_sample = pd.to_numeric(sample[rpm_column], errors="coerce")
        if rpm_sample.notna().mean() < 0.80:
            rpm_column = None

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": columns,
        "vibration_column": vibration_column,
        "rpm_column": rpm_column,
    }


def audit_selected_csv(path, schema):
    usecols = [schema["vibration_column"]]
    if schema["rpm_column"] is not None:
        usecols.append(schema["rpm_column"])

    row_count = 0
    vibration_numeric = 0
    vibration_nonfinite = 0
    rpm_values = []

    reader = pd.read_csv(
        path,
        sep=schema["delimiter"],
        encoding=schema["encoding"],
        usecols=usecols,
        chunksize=100000,
        low_memory=False,
    )
    for chunk in reader:
        row_count += len(chunk)
        vibration = pd.to_numeric(
            chunk[schema["vibration_column"]],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite = np.isfinite(vibration)
        vibration_numeric += int(finite.sum())
        vibration_nonfinite += int((~finite).sum())

        if schema["rpm_column"] is not None:
            rpm = pd.to_numeric(
                chunk[schema["rpm_column"]],
                errors="coerce",
            ).to_numpy(dtype=float)
            rpm = rpm[np.isfinite(rpm)]
            if rpm.size:
                rpm_values.append(rpm)

    if row_count == 0:
        raise RuntimeError(f"Empty selected CSV: {path}")

    numeric_fraction = vibration_numeric / row_count
    if numeric_fraction < 0.999:
        raise RuntimeError(
            f"Vibration numeric/finite fraction is {numeric_fraction:.6f} "
            f"for {path}; expected at least 0.999."
        )

    rpm_median = np.nan
    rpm_valid_fraction = 0.0
    rpm_source = "registered_nominal_1750_rpm"
    if schema["rpm_column"] is not None and rpm_values:
        rpm_all = np.concatenate(rpm_values)
        rpm_valid_fraction = float(len(rpm_all) / row_count)
        rpm_median_candidate = float(np.median(rpm_all))
        if (
            rpm_valid_fraction >= 0.80 and
            1500.0 <= rpm_median_candidate <= 2000.0
        ):
            rpm_median = rpm_median_candidate
            rpm_source = "csv_hall_effect_rpm"

    return {
        "row_count": int(row_count),
        "vibration_numeric_fraction": float(numeric_fraction),
        "vibration_nonfinite_cells": int(vibration_nonfinite),
        "rpm_source": rpm_source,
        "rpm_valid_fraction": float(rpm_valid_fraction),
        "rpm_median": (
            None if not np.isfinite(rpm_median) else float(rpm_median)
        ),
    }


# -------------------------------------------------------------------------
# 4. Expected bearing-state registry
# -------------------------------------------------------------------------

FAULT_CODE = {
    "I": "IR",
    "O": "OR",
    "B": "B",
    "C": "C",
}
TARGET_ORDER = {
    "IR": "BPFI",
    "OR": "BPFO",
    "B": "BSF",
    "C": "FTF",
}


def planned_cohort(unit_number):
    if 1 <= unit_number <= 5:
        return "IR"
    if 6 <= unit_number <= 10:
        return "OR"
    if 11 <= unit_number <= 15:
        return "B"
    if 16 <= unit_number <= 20:
        return "C"
    raise ValueError(f"Unexpected UORED unit number: {unit_number}")


expected_rows = []
for unit_number in range(1, 21):
    cohort = planned_cohort(unit_number)
    expected_rows.append({
        "record_id": f"H-{unit_number}-0",
        "file_stem": f"H-{unit_number}-0",
        "unit": f"UORED_{unit_number:02d}",
        "unit_number": unit_number,
        "state": "healthy",
        "state_code": 0,
        "fault_label": "N",
        "planned_cohort": cohort,
        "target_order": None,
        "primary_test_state": False,
        "healthy_calibration_eligible": True,
    })

for code, label, units in [
    ("I", "IR", range(1, 6)),
    ("O", "OR", range(6, 11)),
    ("B", "B", range(11, 16)),
    ("C", "C", range(16, 21)),
]:
    for unit_number in units:
        for state_code, state in [(1, "developing"), (2, "faulty")]:
            expected_rows.append({
                "record_id": f"{code}-{unit_number}-{state_code}",
                "file_stem": f"{code}-{unit_number}-{state_code}",
                "unit": f"UORED_{unit_number:02d}",
                "unit_number": unit_number,
                "state": state,
                "state_code": state_code,
                "fault_label": label,
                "planned_cohort": label,
                "target_order": TARGET_ORDER[label],
                "primary_test_state": label in {"IR", "OR"},
                "healthy_calibration_eligible": False,
            })

UORED_EXPECTED_REGISTRY = pd.DataFrame(expected_rows).sort_values(
    ["unit_number", "state_code"]
).reset_index(drop=True)

if len(UORED_EXPECTED_REGISTRY) != 60:
    raise RuntimeError("Internal error: expected UORED registry is not 60 rows")
if UORED_EXPECTED_REGISTRY.record_id.duplicated().any():
    raise RuntimeError("Internal error: duplicate expected UORED record IDs")


# -------------------------------------------------------------------------
# 5. Discover exactly one raw CSV per expected state
# -------------------------------------------------------------------------

all_csv_paths = sorted(
    path for path in UORED_ROOT.rglob("*.csv")
    if path.is_file()
)

if not all_csv_paths:
    archives = sorted(
        path for path in UORED_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".zip", ".7z", ".rar"}
    )
    archive_note = "\n".join(f"  {path}" for path in archives[:20])
    raise FileNotFoundError(
        "No extracted CSV files were found below UORED_ROOT. Extract the "
        "raw CSV archive first; do not use the processed spectrogram PNGs."
        + (f"\nArchives found:\n{archive_note}" if archives else "")
    )

exact_stem_pattern = re.compile(
    r"^(?P<code>[HIOBC])[-_](?P<unit>\d{1,2})[-_](?P<state>[012])$",
    flags=re.IGNORECASE,
)

candidate_map = {}
ignored_csv_paths = []
for path in all_csv_paths:
    match = exact_stem_pattern.fullmatch(path.stem.strip())
    if match is None:
        ignored_csv_paths.append(path)
        continue
    record_id = (
        f"{match.group('code').upper()}-"
        f"{int(match.group('unit'))}-"
        f"{int(match.group('state'))}"
    )
    candidate_map.setdefault(record_id, []).append(path)

expected_ids = set(UORED_EXPECTED_REGISTRY.record_id)
unexpected_ids = sorted(set(candidate_map) - expected_ids)
missing_ids = sorted(expected_ids - set(candidate_map))

if unexpected_ids:
    raise RuntimeError(
        "Raw CSV filenames encode unexpected bearing/state combinations: "
        + ", ".join(unexpected_ids)
    )
if missing_ids:
    raise RuntimeError(
        f"Missing {len(missing_ids)} expected raw CSV states: "
        + ", ".join(missing_ids)
    )

selected_path = {}
duplicate_rows = []
for record_id in sorted(expected_ids):
    paths = candidate_map[record_id]
    if len(paths) == 1:
        selected_path[record_id] = paths[0]
        continue

    hash_groups = {}
    for path in paths:
        digest = sha256_file(path)
        hash_groups.setdefault(digest, []).append(path)
    if len(hash_groups) != 1:
        details = "\n".join(
            f"  {sha256_file(path)}  {path}"
            for path in paths
        )
        raise RuntimeError(
            f"Conflicting duplicate CSV files for {record_id}:\n{details}"
        )

    paths = sorted(
        paths,
        key=lambda path: (
            "raw" not in str(path.parent).lower(),
            len(path.parts),
            str(path).lower(),
        ),
    )
    selected_path[record_id] = paths[0]
    for duplicate in paths[1:]:
        duplicate_rows.append({
            "record_id": record_id,
            "selected_relative_path": clean_relative_path(paths[0]),
            "duplicate_relative_path": clean_relative_path(duplicate),
            "sha256": sha256_file(duplicate),
            "status": "byte_identical_duplicate_not_selected",
        })


# -------------------------------------------------------------------------
# 6. Schema and integrity audit (no signal features)
# -------------------------------------------------------------------------

print("Frozen UORED schema audit")
print(f"  source root          : {UORED_ROOT.resolve()}")
print(f"  expected raw states  : {len(UORED_EXPECTED_REGISTRY)}")
print(f"  discovered CSV files : {len(all_csv_paths)}")
print(f"  ignored CSV files    : {len(ignored_csv_paths)}")
print(f"  identical duplicates : {len(duplicate_rows)}")
print("  spectra/features     : NOT COMPUTED")

manifest_rows = []
schema_errors = []

for index, expected in enumerate(
    UORED_EXPECTED_REGISTRY.itertuples(index=False),
    start=1,
):
    path = selected_path[expected.record_id]
    try:
        schema = inspect_csv_schema(path)
        audit = audit_selected_csv(path, schema)
        manifest_rows.append({
            **expected._asdict(),
            "relative_path": clean_relative_path(path),
            "file_size_bytes": int(path.stat().st_size),
            "file_sha256": sha256_file(path),
            "csv_encoding": schema["encoding"],
            "csv_delimiter": schema["delimiter"],
            "csv_columns_json": json.dumps(schema["columns"]),
            "vibration_column": schema["vibration_column"],
            "rpm_column": schema["rpm_column"],
            **audit,
        })
    except Exception as exc:
        schema_errors.append({
            "record_id": expected.record_id,
            "relative_path": clean_relative_path(path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })

    if index % 10 == 0 or index == 60:
        print(
            f"  audited {index:2d}/60 raw states  "
            f"schema_errors={len(schema_errors)}"
        )

if schema_errors:
    error_frame = pd.DataFrame(schema_errors)
    print("\nSCHEMA ERRORS")
    print(error_frame.to_string(index=False))
    print("\nRepresentative CSV headers")
    for record_id in sorted(expected_ids)[:5]:
        path = selected_path[record_id]
        try:
            encoding = detect_text_encoding(path)
            delimiter = detect_delimiter(path, encoding)
            header = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                nrows=2,
                low_memory=False,
            ).columns.tolist()
            print(f"  {record_id}: {header}")
        except Exception as exc:
            print(f"  {record_id}: header read failed: {exc}")
    raise RuntimeError(
        "Cell 14 stopped before freezing the loader. Resolve the schema "
        "errors using exact header-name overrides; do not guess channel "
        "positions and do not compute spectra."
    )

UORED_MANIFEST = pd.DataFrame(manifest_rows).sort_values(
    ["unit_number", "state_code"]
).reset_index(drop=True)

if len(UORED_MANIFEST) != 60:
    raise RuntimeError(
        f"Expected 60 audited raw states, found {len(UORED_MANIFEST)}"
    )
if UORED_MANIFEST.record_id.duplicated().any():
    raise RuntimeError("Duplicate record IDs in final UORED manifest")

row_count_ok = UORED_MANIFEST.row_count.between(419000, 421000)
if not bool(row_count_ok.all()):
    bad = UORED_MANIFEST.loc[
        ~row_count_ok,
        ["record_id", "relative_path", "row_count"],
    ]
    print("\nUNEXPECTED RAW LENGTHS")
    print(bad.to_string(index=False))
    raise RuntimeError(
        "Raw record length is inconsistent with 42 kHz x 10 seconds. "
        "Stop before freezing the loader."
    )

if int(UORED_MANIFEST.vibration_nonfinite_cells.sum()) != 0:
    bad = UORED_MANIFEST[
        UORED_MANIFEST.vibration_nonfinite_cells > 0
    ][["record_id", "vibration_nonfinite_cells"]]
    print("\nNON-FINITE VIBRATION CELLS")
    print(bad.to_string(index=False))
    raise RuntimeError(
        "Non-finite vibration values are not allowed in the frozen source."
    )

if UORED_MANIFEST.file_sha256.duplicated().any():
    duplicates = UORED_MANIFEST[
        UORED_MANIFEST.file_sha256.duplicated(keep=False)
    ][["record_id", "relative_path", "file_sha256"]]
    print("\nCROSS-STATE BYTE-IDENTICAL FILES")
    print(duplicates.to_string(index=False))
    raise RuntimeError(
        "Different bearing states unexpectedly contain byte-identical raw "
        "files. Stop for a source-integrity audit."
    )


# -------------------------------------------------------------------------
# 7. Freeze manifest and exact loader specification
# -------------------------------------------------------------------------

CELL14_CONFIG = {
    "method_version": "v5.0-uored-manifest-loader-freeze",
    "source_registration_sha256": EXPECTED_CELL13_HASH,
    "dataset": "UORED-VAFCLS",
    "dataset_version": 5,
    "doi": "10.17632/y2px5tg92h.5",
    "canonical_raw_format": "CSV",
    "expected_raw_states": 60,
    "expected_samples_per_state": 420000,
    "accepted_sample_count_range": [419000, 421000],
    "sampling_rate_hz": 42000.0,
    "nominal_rpm_fallback": 1750.0,
    "vibration_column_override": UORED_VIBRATION_COLUMN_OVERRIDE,
    "rpm_column_override": UORED_RPM_COLUMN_OVERRIDE,
    "forbidden_operations": [
        "spectral calculation",
        "envelope calculation",
        "order-feature extraction",
        "class-wise signal summary",
        "prediction",
        "performance evaluation",
    ],
}
CELL14_CONFIG_HASH = canonical_digest(CELL14_CONFIG)

manifest_for_hash = UORED_MANIFEST.copy()
manifest_for_hash = manifest_for_hash.where(
    pd.notna(manifest_for_hash),
    None,
)
manifest_records = manifest_for_hash.to_dict(orient="records")
UORED_MANIFEST_HASH = canonical_digest(manifest_records)

UORED_LOADER_SPEC = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "config_sha256": CELL14_CONFIG_HASH,
    "manifest_sha256": UORED_MANIFEST_HASH,
    "external_outcomes_computed": False,
    "source_root_recorded_for_local_reproducibility": str(
        UORED_ROOT.resolve()
    ),
    "config": CELL14_CONFIG,
    "files": manifest_records,
}
UORED_LOADER_SPEC_HASH = canonical_digest({
    "config_sha256": CELL14_CONFIG_HASH,
    "manifest_sha256": UORED_MANIFEST_HASH,
    "files": manifest_records,
})
UORED_LOADER_SPEC["loader_spec_sha256"] = UORED_LOADER_SPEC_HASH

CELL14_ROOT = Path(MSSP_DIRS["audit"]) / "uored_v5_manifest"
CELL14_ROOT.mkdir(parents=True, exist_ok=True)

UORED_MANIFEST_PATH = CELL14_ROOT / "uored_v5_file_manifest.csv"
UORED_DUPLICATES_PATH = CELL14_ROOT / "uored_v5_identical_duplicates.csv"
UORED_LOADER_SPEC_PATH = CELL14_ROOT / "uored_v5_loader_spec.json"
UORED_LOADER_HASH_PATH = CELL14_ROOT / "uored_v5_loader_spec.sha256"

if UORED_LOADER_SPEC_PATH.exists():
    with UORED_LOADER_SPEC_PATH.open("r", encoding="utf-8") as handle:
        prior_spec = json.load(handle)
    if prior_spec.get("loader_spec_sha256") != UORED_LOADER_SPEC_HASH:
        raise RuntimeError(
            "A conflicting Cell 14 loader specification already exists at "
            f"{UORED_LOADER_SPEC_PATH}. Do not overwrite it."
        )
    UORED_LOADER_SPEC = prior_spec
else:
    UORED_MANIFEST.to_csv(UORED_MANIFEST_PATH, index=False)
    pd.DataFrame(duplicate_rows).to_csv(
        UORED_DUPLICATES_PATH,
        index=False,
    )
    with UORED_LOADER_SPEC_PATH.open("x", encoding="utf-8") as handle:
        json.dump(UORED_LOADER_SPEC, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with UORED_LOADER_HASH_PATH.open("x", encoding="utf-8") as handle:
        handle.write(
            f"{UORED_LOADER_SPEC_HASH}  "
            f"{UORED_LOADER_SPEC_PATH.name}\n"
        )

with UORED_LOADER_SPEC_PATH.open("r", encoding="utf-8") as handle:
    stored_loader_spec = json.load(handle)
if stored_loader_spec.get("loader_spec_sha256") != UORED_LOADER_SPEC_HASH:
    raise RuntimeError("Stored Cell 14 loader lock failed verification")


# -------------------------------------------------------------------------
# 8. Outcome-blind report
# -------------------------------------------------------------------------

schema_counts = (
    UORED_MANIFEST.groupby(
        [
            "vibration_column",
            "rpm_column",
            "rpm_source",
            "csv_delimiter",
        ],
        dropna=False,
    )
    .size()
    .rename("files")
    .reset_index()
)

registry_balance = (
    UORED_MANIFEST.groupby(
        ["state", "fault_label"],
        dropna=False,
    )
    .agg(
        files=("record_id", "size"),
        units=("unit", "nunique"),
    )
    .reset_index()
)

print("\n" + "=" * 104)
print("CELL 14 UORED MANIFEST AND LOADER-FREEZE REPORT")
print("=" * 104)
print(f"Dataset                 : UORED-VAFCLS version 5")
print(f"Source root             : {UORED_ROOT.resolve()}")
print(f"Expected/found states   : 60 / {len(UORED_MANIFEST)}")
print(f"Physical bearings       : {UORED_MANIFEST.unit.nunique()}")
print(f"Unique source hashes    : {UORED_MANIFEST.file_sha256.nunique()}")
print(f"Manifest SHA-256        : {UORED_MANIFEST_HASH}")
print(f"Loader-spec SHA-256     : {UORED_LOADER_SPEC_HASH}")
print(f"External outcomes       : NOT COMPUTED")

print("\nRegistry balance:")
print(registry_balance.to_string(index=False))

print("\nFrozen schema mappings:")
print(schema_counts.to_string(index=False))

print("\nRaw integrity:")
print(
    f"  row count range       : "
    f"{UORED_MANIFEST.row_count.min()} - "
    f"{UORED_MANIFEST.row_count.max()}"
)
print(
    f"  non-finite vibration  : "
    f"{int(UORED_MANIFEST.vibration_nonfinite_cells.sum())}"
)
print(
    f"  RPM sensor files      : "
    f"{int((UORED_MANIFEST.rpm_source == 'csv_hall_effect_rpm').sum())}/60"
)

print("\nSaved immutable audit:")
print(f"  {UORED_MANIFEST_PATH.resolve()}")
print(f"  {UORED_LOADER_SPEC_PATH.resolve()}")

print("\nCELL 14 COMPLETE")
print(
    "Send the complete report. Cell 15 will use only this locked loader "
    "specification to extract the frozen 25-revolution v4 segment features."
)
