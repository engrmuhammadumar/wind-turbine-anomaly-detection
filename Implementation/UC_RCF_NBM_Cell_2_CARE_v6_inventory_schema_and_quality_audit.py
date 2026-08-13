"""CELL 2 — CARE v6 inventory, schema, taxonomy, and leakage-safe quality audit.

Paste this complete file into the second cell of the UC-RCF-NBM notebook and
run it only after Cell 1 has printed its successful contract-lock banner.

This cell deliberately fits no model and chooses no hyperparameter.  It builds
a safe case registry, places outcome-only metadata in a separate lockbox,
constructs a metadata-grounded signal taxonomy, and audits signal quality using
only normal-status rows in each source training partition.  The audit is
diagnostic: channel eligibility is recomputed inside every outer fold later.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# =============================================================================
# 0. Cell-1 compatibility and leakage guards
# =============================================================================

EXPECTED_CELL1_CONTRACT_SHA256 = (
    "827641aecd8e807193ad193d64c274319756092faa53bdb5084f310f62041f49"
)

_required_cell1_names = (
    "CARE_DATA_ROOT",
    "OUTPUT_ROOT",
    "INVENTORY_DIR",
    "QUALITY_DIR",
    "CONTRACT_SHA256",
    "DATASET",
    "QUALITY",
    "MEAN_MODEL",
    "FORBIDDEN_PREDICTOR_FIELDS",
    "save_json",
    "sha256_json",
    "utc_now",
)
_missing_cell1_names = [name for name in _required_cell1_names if name not in globals()]
if _missing_cell1_names:
    raise RuntimeError(
        "Run UC-RCF-NBM Cell 1 before Cell 2. Missing notebook objects: "
        + ", ".join(_missing_cell1_names)
    )

if CONTRACT_SHA256 != EXPECTED_CELL1_CONTRACT_SHA256:
    raise RuntimeError(
        "Cell 1 does not match the frozen UC-RCF-NBM experiment contract. "
        f"Expected {EXPECTED_CELL1_CONTRACT_SHA256}, received {CONTRACT_SHA256}. "
        "Do not mix cells from different experiments."
    )

if not Path(CARE_DATA_ROOT).is_dir():
    raise FileNotFoundError(f"CARE v6 dataset root not found:\n{CARE_DATA_ROOT}")

for _directory in (Path(INVENTORY_DIR), Path(QUALITY_DIR)):
    _directory.mkdir(parents=True, exist_ok=True)


# Labels and event boundaries are allowed here only to validate the benchmark
# inventory and construct the outcome lockbox.  They never enter CASE_REGISTRY,
# CARE_FEATURE_REGISTRY, FARM_SCHEMAS, or the quality-audit functions.
SAFE_CASE_COLUMNS = (
    "case_key",
    "farm",
    "asset_id",          # Canonical farm-qualified asset identifier.
    "source_asset_id",   # Identifier written in the CARE source file.
    "event_id",          # Case/file identifier only; never a predictor.
    "file_path",
    "relative_file_path",
    "size_bytes",
)

OUTCOME_ONLY_COLUMNS = (
    "event_label_raw",
    "is_anomaly",
    "event_start",
    "event_start_id",
    "event_end",
    "event_end_id",
    "event_description",
)

if set(FORBIDDEN_PREDICTOR_FIELDS) - set(OUTCOME_ONLY_COLUMNS) - {
    "event_label",
    "care_ground_truth",
    "fault_type",
}:
    raise RuntimeError("Cell 1 and Cell 2 outcome-field definitions are inconsistent.")


# =============================================================================
# 1. Robust CARE readers and canonical names
# =============================================================================

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
CSV_SEPARATORS = (",", ";", "\t", "|")
EVENT_FILE_PATTERN = re.compile(r"^\d+$")
AUDIT_CHUNK_ROWS = 20_000

COLUMN_ALIASES = {
    "timestamp": "time_stamp",
    "time": "time_stamp",
    "datetime": "time_stamp",
    "asset": "asset_id",
    "assetid": "asset_id",
    "turbine_id": "asset_id",
    "wt_id": "asset_id",
    "status_id": "status_type_id",
    "status_type": "status_type_id",
    "train_or_test": "train_test",
    "split": "train_test",
    "eventid": "event_id",
    "event_start_index": "event_start_id",
    "event_end_index": "event_end_id",
    "start_id": "event_start_id",
    "end_id": "event_end_id",
    "event_start_time": "event_start",
    "event_end_time": "event_end",
    "start_time": "event_start",
    "end_time": "event_end",
    "sensor_name": "feature_name",
    "signal_name": "feature_name",
    "column_name": "feature_name",
    "featurename": "feature_name",
    "featuredescription": "feature_description",
    "description": "feature_description",
}


def standardize_column_name(raw: Any) -> str:
    """Convert CARE column aliases to stable snake_case names."""
    text = str(raw).strip().lstrip("\ufeff")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return COLUMN_ALIASES.get(text, text)


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        canonical = standardize_column_name(candidate)
        if canonical in frame.columns:
            return canonical
    return None


def normalize_identifier(value: Any) -> str:
    """Represent numeric and textual asset identifiers without float artifacts."""
    if pd.isna(value):
        raise ValueError("Missing asset identifier")
    text = str(value).strip()
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    if not text:
        raise ValueError("Empty asset identifier")
    return text


def parse_care_timestamps(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip().replace(
        {"": pd.NA, "None": pd.NA, "NULL": pd.NA, "nan": pd.NA, "NaT": pd.NA}
    )
    try:
        parsed = pd.to_datetime(
            cleaned,
            format="mixed",
            dayfirst=True,
            errors="coerce",
            utc=True,
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(cleaned, dayfirst=True, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


@lru_cache(maxsize=None)
def read_care_header(file_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    """Detect an event CSV's encoding and separator using its header only."""
    file_path = Path(file_path)
    best: tuple[int, int, list[str], list[str], str, str] | None = None
    metadata = set(DATASET.metadata_columns)
    for encoding in CSV_ENCODINGS:
        for separator in CSV_SEPARATORS:
            try:
                header = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=separator,
                    nrows=0,
                    engine="python",
                )
            except Exception:
                continue
            raw = [str(column).strip().lstrip("\ufeff") for column in header.columns]
            canonical = [standardize_column_name(column) for column in raw]
            if len(canonical) < 2 or len(canonical) != len(set(canonical)):
                continue
            candidate = (
                len(metadata & set(canonical)),
                len(canonical),
                raw,
                canonical,
                encoding,
                separator,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] < len(metadata):
        raise RuntimeError(
            "Could not identify all required CARE event columns in:\n"
            f"{file_path}\nRequired: {sorted(metadata)}"
        )
    return tuple(best[2]), tuple(best[3]), best[4], best[5]


def read_care_chunks(
    file_path: Path,
    usecols: Iterable[str],
    chunksize: int = AUDIT_CHUNK_ROWS,
) -> Iterable[pd.DataFrame]:
    """Stream selected CARE columns and return canonical column names."""
    raw, canonical, encoding, separator = read_care_header(Path(file_path))
    canonical_to_raw = dict(zip(canonical, raw))
    requested = [standardize_column_name(column) for column in usecols]
    missing = [column for column in requested if column not in canonical_to_raw]
    if missing:
        raise KeyError(f"{Path(file_path).name} lacks columns: {', '.join(missing)}")

    reader = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=separator,
        usecols=[canonical_to_raw[column] for column in requested],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk.columns = [standardize_column_name(column) for column in chunk.columns]
        if "time_stamp" in chunk.columns:
            chunk["time_stamp"] = parse_care_timestamps(chunk["time_stamp"])
        yield chunk


def read_metadata_csv(file_path: Path, required: set[str]) -> pd.DataFrame:
    """Parse a small CARE metadata CSV despite encoding or delimiter variation."""
    for encoding in CSV_ENCODINGS:
        for skiprows in range(0, 20):
            for separator in CSV_SEPARATORS:
                try:
                    frame = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=separator,
                        skiprows=skiprows,
                        engine="python",
                    )
                except Exception:
                    continue
                frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
                if frame.empty:
                    continue
                frame.columns = [standardize_column_name(c) for c in frame.columns]
                if len(frame.columns) != len(set(frame.columns)):
                    continue
                if required <= set(frame.columns):
                    return frame.reset_index(drop=True)
    raise RuntimeError(f"No table containing {sorted(required)} found in:\n{file_path}")


def select_unique_metadata_file(
    candidates: Iterable[Path],
    required: set[str],
    description: str,
) -> tuple[Path, pd.DataFrame]:
    parsed: list[tuple[Path, pd.DataFrame]] = []
    for path in candidates:
        try:
            parsed.append((Path(path), read_metadata_csv(Path(path), required)))
        except RuntimeError:
            continue
    if len(parsed) != 1:
        names = [str(path) for path, _ in parsed]
        raise RuntimeError(
            f"Expected exactly one parseable {description}; found {len(parsed)}: {names}"
        )
    return parsed[0]


def sha256_file(file_path: Path, maximum_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = maximum_bytes
    with Path(file_path).open("rb") as handle:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            block = handle.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def save_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(output_path)


def dataframe_sha256(frame: pd.DataFrame, sort_columns: Iterable[str]) -> str:
    ordered = frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.12g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# 2. File discovery and lightweight dataset manifest
# =============================================================================

def infer_farm(path: Path) -> str | None:
    compact = "".join(character.lower() for character in str(path.parent) if character.isalnum())
    for code in ("a", "b", "c"):
        if f"windfarm{code}" in compact or f"farm{code}" in compact:
            return f"Wind Farm {code.upper()}"
    return None


_all_csv_files = sorted(
    path for path in Path(CARE_DATA_ROOT).rglob("*")
    if path.is_file() and path.suffix.lower() == ".csv"
)

_event_records: list[dict[str, Any]] = []
_event_info_files: dict[str, list[Path]] = defaultdict(list)
_feature_description_files: dict[str, list[Path]] = defaultdict(list)

for _path in _all_csv_files:
    _farm = infer_farm(_path)
    _stem = _path.stem.lower()
    if EVENT_FILE_PATTERN.fullmatch(_path.stem):
        if _farm is None:
            raise RuntimeError(f"Numeric event CSV is outside a recognized farm: {_path}")
        _event_records.append(
            {
                "farm": _farm,
                "event_id": int(_path.stem),
                "file_path": _path,
                "relative_file_path": _path.relative_to(CARE_DATA_ROOT).as_posix(),
                "size_bytes": int(_path.stat().st_size),
            }
        )
    elif "feature" in _stem:
        _feature_description_files[_farm].append(_path)
    elif "event" in _stem:
        _event_info_files[_farm].append(_path)

EVENT_FILE_INVENTORY = (
    pd.DataFrame(_event_records)
    .sort_values(["farm", "event_id"], kind="stable")
    .reset_index(drop=True)
)

if len(EVENT_FILE_INVENTORY) != DATASET.expected_total_cases:
    raise RuntimeError(
        f"Found {len(EVENT_FILE_INVENTORY)} numeric event files; "
        f"expected {DATASET.expected_total_cases}."
    )

_expected_case_counts = dict(zip(DATASET.farms, DATASET.expected_cases_by_farm))
_observed_case_counts = EVENT_FILE_INVENTORY.groupby("farm").size().to_dict()
if _observed_case_counts != _expected_case_counts:
    raise RuntimeError(
        f"Per-farm event-file counts differ from CARE v6. "
        f"Observed={_observed_case_counts}, expected={_expected_case_counts}."
    )

_manifest_records: list[dict[str, Any]] = []
for _path in _all_csv_files:
    _is_event = bool(EVENT_FILE_PATTERN.fullmatch(_path.stem))
    _manifest_records.append(
        {
            "relative_file_path": _path.relative_to(CARE_DATA_ROOT).as_posix(),
            "farm": infer_farm(_path),
            "file_kind": (
                "event_data"
                if _is_event
                else "feature_metadata"
                if "feature" in _path.stem.lower()
                else "event_metadata"
                if "event" in _path.stem.lower()
                else "other_csv"
            ),
            "size_bytes": int(_path.stat().st_size),
            # Full hashes for small metadata; a clearly named 64-KiB head hash for
            # large event files. The manifest is not misrepresented as a full-data hash.
            "content_hash_scope": "first_65536_bytes" if _is_event else "full_file",
            "content_sha256": sha256_file(_path, 65_536 if _is_event else None),
        }
    )

DATASET_FILE_MANIFEST = pd.DataFrame(_manifest_records).sort_values(
    "relative_file_path", kind="stable"
).reset_index(drop=True)
DATASET_MANIFEST_SHA256 = dataframe_sha256(
    DATASET_FILE_MANIFEST, ("relative_file_path",)
)


# =============================================================================
# 3. Safe case registry and outcome lockbox
# =============================================================================

def normalize_event_label(value: Any) -> bool:
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    if text in {"1", "true", "yes"} or any(
        token in text for token in ("anomal", "abnormal", "fault", "failure")
    ):
        return True
    if text in {"0", "false", "no"} or any(
        token in text for token in ("normal", "healthy")
    ):
        return False
    raise ValueError(f"Unrecognized CARE event label: {value!r}")


def build_case_registry_and_lockbox() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata_frames: list[pd.DataFrame] = []

    for farm in DATASET.farms:
        _, frame = select_unique_metadata_file(
            _event_info_files.get(farm, []),
            {"event_id", "asset_id"},
            f"event-information file for {farm}",
        )
        label_column = find_column(frame, ("event_label", "label", "event_type", "class"))
        if label_column is None:
            raise RuntimeError(f"The event-information file for {farm} has no label column.")
        frame = frame.rename(columns={label_column: "event_label_raw"}).copy()
        frame["farm"] = farm
        frame["event_id"] = pd.to_numeric(frame["event_id"], errors="raise").astype(int)
        frame["source_asset_id"] = frame["asset_id"].map(normalize_identifier)
        frame["asset_id"] = farm + "::" + frame["source_asset_id"]
        frame["case_key"] = (
            farm + "::event_" + frame["event_id"].astype(str)
        )
        frame["is_anomaly"] = frame["event_label_raw"].map(normalize_event_label)

        for column in ("event_start", "event_end"):
            frame[column] = (
                parse_care_timestamps(frame[column])
                if column in frame.columns
                else pd.NaT
            )
        for column in ("event_start_id", "event_end_id"):
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce").astype("Int64")
                if column in frame.columns
                else pd.Series(pd.NA, index=frame.index, dtype="Int64")
            )
        if "event_description" not in frame.columns:
            frame["event_description"] = ""
        metadata_frames.append(frame)

    metadata = pd.concat(metadata_frames, ignore_index=True)
    if metadata.duplicated(["farm", "event_id"]).any():
        raise RuntimeError("Duplicate (farm, event_id) entries exist in event metadata.")

    merged = metadata.merge(
        EVENT_FILE_INVENTORY,
        on=["farm", "event_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        mismatch = merged.loc[
            ~merged["_merge"].eq("both"), ["farm", "event_id", "_merge"]
        ]
        raise RuntimeError(f"Event metadata/files do not match:\n{mismatch.to_string(index=False)}")
    merged = merged.drop(columns="_merge").sort_values(
        ["farm", "event_id"], kind="stable"
    ).reset_index(drop=True)

    anomaly_by_farm = (
        merged.groupby("farm", sort=False)["is_anomaly"].sum().astype(int).to_dict()
    )
    normal_by_farm = (
        merged.groupby("farm", sort=False)["is_anomaly"]
        .apply(lambda values: int((~values.astype(bool)).sum()))
        .to_dict()
    )
    expected_anomaly = dict(zip(DATASET.farms, DATASET.expected_anomaly_by_farm))
    expected_normal = dict(zip(DATASET.farms, DATASET.expected_normal_by_farm))
    if anomaly_by_farm != expected_anomaly or normal_by_farm != expected_normal:
        raise RuntimeError(
            "CARE outcome counts differ from the frozen v6 contract. "
            f"Anomaly={anomaly_by_farm}, normal={normal_by_farm}."
        )

    anomaly_boundaries = merged.loc[
        merged["is_anomaly"], ["event_start", "event_end"]
    ]
    if anomaly_boundaries.isna().any().any():
        raise RuntimeError("At least one anomaly case lacks a timestamp event boundary.")
    if (anomaly_boundaries["event_end"] < anomaly_boundaries["event_start"]).any():
        raise RuntimeError("At least one anomaly event ends before it starts.")

    observed_assets = int(merged["asset_id"].nunique())
    if observed_assets != DATASET.expected_assets:
        raw_unique = int(merged["source_asset_id"].nunique())
        raise RuntimeError(
            f"Found {observed_assets} farm-qualified assets; expected "
            f"{DATASET.expected_assets}. Raw unqualified identifiers={raw_unique}."
        )

    safe = merged.loc[:, SAFE_CASE_COLUMNS].copy()
    if set(safe.columns) & set(FORBIDDEN_PREDICTOR_FIELDS):
        raise RuntimeError("Outcome fields escaped into the safe CASE_REGISTRY.")

    lockbox_columns = (
        "case_key",
        "farm",
        "asset_id",
        "source_asset_id",
        "event_id",
        *OUTCOME_ONLY_COLUMNS,
    )
    lockbox = merged.loc[:, lockbox_columns].copy()
    counts = {
        "cases": int(len(merged)),
        "assets": observed_assets,
        "anomaly_cases": int(merged["is_anomaly"].sum()),
        "normal_cases": int((~merged["is_anomaly"]).sum()),
        "anomaly_by_farm": anomaly_by_farm,
        "normal_by_farm": normal_by_farm,
    }
    return safe, lockbox, counts


CASE_REGISTRY, _OUTCOME_LOCKBOX, OUTCOME_COUNT_CHECK = build_case_registry_and_lockbox()
OUTCOME_LOCKBOX_SHA256 = dataframe_sha256(
    _OUTCOME_LOCKBOX, ("farm", "event_id")
)

if tuple(CASE_REGISTRY.columns) != SAFE_CASE_COLUMNS:
    raise RuntimeError("CASE_REGISTRY does not have the exact safe-column contract.")


# =============================================================================
# 4. Per-farm schemas and metadata-grounded signal taxonomy
# =============================================================================

STATISTIC_SUFFIXES = {
    "_avg": "avg",
    "_std": "std",
    "_max": "max",
    "_min": "min",
}


def split_signal_statistic(column: str) -> tuple[str, str]:
    for suffix, statistic in STATISTIC_SUFFIXES.items():
        if column.endswith(suffix):
            return column[: -len(suffix)], statistic
    return column, "unspecified"


def parse_metadata_boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def semantic_role(
    description: str,
    is_angle: bool,
    is_counter: bool,
) -> tuple[str, str]:
    lowered = re.sub(r"\s+", " ", str(description).strip().lower())
    # CARE metadata uses both "wind speed" and "windspeed" spellings.
    lowered = re.sub(r"\bwindspeed\b", "wind speed", lowered)
    driver_matches = [
        pattern
        for pattern in MEAN_MODEL.driver_description_patterns
        if pattern in lowered
    ]
    temperature = bool(re.search(r"\btemp(?:erature)?\b", lowered))

    if driver_matches:
        role = "operating_driver_candidate"
        reason = "; ".join(driver_matches)
    elif temperature:
        role = "temperature_target_candidate"
        reason = "temperature semantics"
    elif is_angle:
        role = "angle_target_candidate"
        reason = "metadata is_angle"
    elif is_counter:
        role = "counter_target_candidate"
        reason = "metadata is_counter"
    elif lowered:
        role = "other_target_candidate"
        reason = "described non-driver signal"
    else:
        role = "unclassified_target_candidate"
        reason = "missing feature description"
    return role, reason


_feature_records: list[dict[str, Any]] = []
_schema_records: list[dict[str, Any]] = []
FARM_SCHEMAS: dict[str, dict[str, Any]] = {}

for _farm in DATASET.farms:
    _farm_cases = CASE_REGISTRY.loc[CASE_REGISTRY["farm"].eq(_farm)]
    _header_specs = [read_care_header(Path(path)) for path in _farm_cases["file_path"]]
    _schema_fingerprints = {spec[1] for spec in _header_specs}
    if len(_schema_fingerprints) != 1:
        raise RuntimeError(f"{_farm} contains {len(_schema_fingerprints)} event schemas.")

    _raw_columns, _columns, _encoding, _separator = _header_specs[0]
    _missing_metadata = set(DATASET.metadata_columns) - set(_columns)
    if _missing_metadata:
        raise RuntimeError(f"{_farm} schema lacks {sorted(_missing_metadata)}")
    _signal_columns = tuple(
        column for column in _columns if column not in set(DATASET.metadata_columns)
    )
    _forbidden_signals = set(_signal_columns) & set(FORBIDDEN_PREDICTOR_FIELDS)
    if _forbidden_signals:
        raise RuntimeError(
            f"Outcome fields appear inside {_farm}'s signal schema: {_forbidden_signals}"
        )

    _, _feature_metadata = select_unique_metadata_file(
        _feature_description_files.get(_farm, []),
        {"feature_name", "feature_description"},
        f"feature-description file for {_farm}",
    )
    _feature_metadata = _feature_metadata.copy()
    _feature_metadata["base_sensor"] = _feature_metadata["feature_name"].map(
        standardize_column_name
    )
    if _feature_metadata["base_sensor"].duplicated().any():
        duplicates = _feature_metadata.loc[
            _feature_metadata["base_sensor"].duplicated(False), "base_sensor"
        ].tolist()
        raise RuntimeError(f"Duplicate feature-description entries in {_farm}: {duplicates}")
    _metadata_lookup = _feature_metadata.set_index("base_sensor", drop=False)

    for _column in _signal_columns:
        _base, _statistic = split_signal_statistic(_column)
        _described = _base in _metadata_lookup.index
        if _described:
            _metadata_row = _metadata_lookup.loc[_base]
            _description = str(_metadata_row.get("feature_description", "")).strip()
            _unit = str(_metadata_row.get("unit", "")).strip()
            _statistics_declared = str(_metadata_row.get("statistics_type", "")).strip()
            _is_angle = parse_metadata_boolean(_metadata_row.get("is_angle", False))
            _is_counter = parse_metadata_boolean(_metadata_row.get("is_counter", False))
        else:
            _description = ""
            _unit = ""
            _statistics_declared = ""
            _is_angle = False
            _is_counter = False

        _role, _role_reason = semantic_role(_description, _is_angle, _is_counter)
        _preprocessing = (
            "within_segment_first_difference"
            if _is_counter
            else "sine_cosine_encoding"
            if _is_angle
            else "identity"
        )
        _feature_records.append(
            {
                "farm": _farm,
                "column": _column,
                "base_sensor": _base,
                "statistic": _statistic,
                "primary_analysis": _statistic in QUALITY.primary_statistics,
                "sensitivity_analysis": _statistic in QUALITY.sensitivity_statistics,
                "description": _description,
                "unit": _unit,
                "statistics_declared": _statistics_declared,
                "metadata_described": _described,
                "is_angle": _is_angle,
                "is_counter": _is_counter,
                "preprocessing": _preprocessing,
                "role": _role,
                "role_reason": _role_reason,
            }
        )

    _farm_features = pd.DataFrame(
        record for record in _feature_records if record["farm"] == _farm
    )
    _primary = tuple(
        _farm_features.loc[_farm_features["primary_analysis"], "column"]
    )
    _sensitivity = tuple(
        _farm_features.loc[_farm_features["sensitivity_analysis"], "column"]
    )
    _drivers = tuple(
        _farm_features.loc[
            _farm_features["primary_analysis"]
            & _farm_features["role"].eq("operating_driver_candidate"),
            "column",
        ]
    )
    _temperature_targets = tuple(
        _farm_features.loc[
            _farm_features["primary_analysis"]
            & _farm_features["role"].eq("temperature_target_candidate"),
            "column",
        ]
    )
    _other_targets = tuple(
        _farm_features.loc[
            _farm_features["primary_analysis"]
            & _farm_features["role"].isin(
                (
                    "other_target_candidate",
                    "angle_target_candidate",
                    "counter_target_candidate",
                    "unclassified_target_candidate",
                )
            ),
            "column",
        ]
    )
    _angles = tuple(
        _farm_features.loc[_farm_features["primary_analysis"] & _farm_features["is_angle"], "column"]
    )
    _counters = tuple(
        _farm_features.loc[_farm_features["primary_analysis"] & _farm_features["is_counter"], "column"]
    )

    FARM_SCHEMAS[_farm] = {
        "all_columns": tuple(_columns),
        "signal_columns": _signal_columns,
        "primary_columns": _primary,
        "sensitivity_columns": _sensitivity,
        "driver_candidates": _drivers,
        "temperature_target_candidates": _temperature_targets,
        "other_target_candidates": _other_targets,
        "angle_columns": _angles,
        "counter_columns": _counters,
        "encoding": _encoding,
        "separator": _separator,
    }
    _schema_records.append(
        {
            "farm": _farm,
            "cases": int(len(_farm_cases)),
            "total_columns": len(_columns),
            "signal_columns": len(_signal_columns),
            "primary_avg_columns": len(_primary),
            "sensitivity_avg_std_columns": len(_sensitivity),
            "driver_candidates": len(_drivers),
            "temperature_target_candidates": len(_temperature_targets),
            "other_target_candidates": len(_other_targets),
            "angle_columns": len(_angles),
            "counter_columns": len(_counters),
            "metadata_description_coverage": float(_farm_features["metadata_described"].mean()),
        }
    )

CARE_FEATURE_REGISTRY = pd.DataFrame(_feature_records).sort_values(
    ["farm", "column"], kind="stable"
).reset_index(drop=True)
FARM_SCHEMA_SUMMARY = pd.DataFrame(_schema_records)

if CARE_FEATURE_REGISTRY.loc[
    CARE_FEATURE_REGISTRY["statistic"].eq("std"), "primary_analysis"
].any():
    raise RuntimeError("A standard-deviation channel was incorrectly admitted to primary analysis.")

if not CARE_FEATURE_REGISTRY.loc[
    CARE_FEATURE_REGISTRY["primary_analysis"], "statistic"
].eq("avg").all():
    raise RuntimeError("Primary analysis contains a non-average channel.")


# =============================================================================
# 5. Streaming quality audit on source training-normal rows only
# =============================================================================

_train_tokens = {str(value).strip().lower() for value in DATASET.source_train_labels}
_prediction_tokens = {
    str(value).strip().lower() for value in DATASET.source_prediction_labels
}
_train_tokens |= {"0.0"} if "0" in _train_tokens else set()
_prediction_tokens |= {"1.0"} if "1" in _prediction_tokens else set()


def normalized_partition(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    result = pd.Series("unknown", index=values.index, dtype="string")
    result.loc[text.isin(_train_tokens)] = "train"
    result.loc[text.isin(_prediction_tokens)] = "prediction"
    return result


def update_zero_runs(
    zero_matrix: np.ndarray,
    break_before: np.ndarray,
    carry: np.ndarray,
    longest: np.ndarray,
) -> None:
    """Update per-channel zero runs, resetting at temporal discontinuities."""
    if zero_matrix.size == 0:
        return
    starts = [0] + [int(index) for index in np.flatnonzero(break_before) if index > 0]
    ends = starts[1:] + [len(zero_matrix)]
    for segment_number, (start, end) in enumerate(zip(starts, ends)):
        if segment_number > 0 or break_before[start]:
            carry[:] = 0
        segment = zero_matrix[start:end]
        if len(segment) == 0:
            continue
        for column_index in range(segment.shape[1]):
            mask = segment[:, column_index]
            false_indices = np.flatnonzero(~mask)
            if len(false_indices) == 0:
                carry[column_index] += len(mask)
                longest[column_index] = max(longest[column_index], carry[column_index])
                continue
            prefix = int(false_indices[0])
            suffix = int(len(mask) - false_indices[-1] - 1)
            run_lengths = np.diff(
                np.concatenate((np.array([-1]), false_indices, np.array([len(mask)])))
            ) - 1
            longest[column_index] = max(
                longest[column_index],
                carry[column_index] + prefix,
                int(run_lengths.max(initial=0)),
            )
            carry[column_index] = suffix


def audit_one_case(case_row: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    farm = str(case_row.farm)
    file_path = Path(case_row.file_path)
    channels = list(FARM_SCHEMAS[farm]["primary_columns"])
    required = [*DATASET.metadata_columns, *channels]

    finite_count = np.zeros(len(channels), dtype=np.int64)
    zero_count = np.zeros(len(channels), dtype=np.int64)
    longest_zero_run = np.zeros(len(channels), dtype=np.int64)
    zero_run_carry = np.zeros(len(channels), dtype=np.int64)

    total_rows = train_rows = prediction_rows = fit_rows = unknown_split_rows = 0
    missing_timestamps = duplicate_timestamps = nonmonotonic_timestamps = 0
    gap_count = 0
    maximum_gap_minutes = 0.0
    previous_timestamp_ns: int | None = None
    source_assets_seen: set[str] = set()
    status_values_seen: set[str] = set()

    nat_ns = np.iinfo(np.int64).min
    expected_delta_ns = int(DATASET.sampling_minutes * 60 * 1_000_000_000)

    for chunk in read_care_chunks(file_path, required):
        rows = len(chunk)
        total_rows += rows
        if rows == 0:
            continue

        partitions = normalized_partition(chunk["train_test"])
        is_train = partitions.eq("train").to_numpy()
        is_prediction = partitions.eq("prediction").to_numpy()
        unknown_split_rows += int(partitions.eq("unknown").sum())
        train_rows += int(is_train.sum())
        prediction_rows += int(is_prediction.sum())

        statuses = pd.to_numeric(chunk["status_type_id"], errors="coerce")
        is_normal_status = statuses.isin(DATASET.normal_status_ids).to_numpy()
        status_values_seen.update(
            str(int(value)) if float(value).is_integer() else str(float(value))
            for value in statuses.dropna().unique()
        )
        fit_mask = is_train & is_normal_status
        fit_rows += int(fit_mask.sum())

        for value in chunk["asset_id"].dropna().unique():
            source_assets_seen.add(normalize_identifier(value))

        timestamps_ns = chunk["time_stamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        missing_timestamps += int((timestamps_ns == nat_ns).sum())
        break_before = np.zeros(rows, dtype=bool)
        for index, current_ns in enumerate(timestamps_ns):
            if index == 0:
                prior_ns = previous_timestamp_ns
            else:
                prior_ns = int(timestamps_ns[index - 1])
            if prior_ns is None or prior_ns == nat_ns or current_ns == nat_ns:
                break_before[index] = prior_ns is not None
                continue
            delta_ns = int(current_ns) - int(prior_ns)
            if delta_ns == 0:
                duplicate_timestamps += 1
                break_before[index] = True
            elif delta_ns < 0:
                nonmonotonic_timestamps += 1
                break_before[index] = True
            elif delta_ns > expected_delta_ns:
                gap_count += 1
                break_before[index] = True
                maximum_gap_minutes = max(
                    maximum_gap_minutes, delta_ns / (60 * 1_000_000_000)
                )
        previous_timestamp_ns = int(timestamps_ns[-1])

        numeric = chunk[channels].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
        eligible_values = numeric[fit_mask]
        finite = np.isfinite(eligible_values)
        zeros = finite & (eligible_values == 0.0)
        finite_count += finite.sum(axis=0, dtype=np.int64)
        zero_count += zeros.sum(axis=0, dtype=np.int64)

        # Non-training or non-normal rows are False and therefore terminate a run.
        zero_timeline = np.zeros_like(numeric, dtype=bool)
        zero_timeline[fit_mask] = zeros
        update_zero_runs(
            zero_timeline,
            break_before,
            zero_run_carry,
            longest_zero_run,
        )

    expected_source_asset = str(case_row.source_asset_id)
    asset_match = source_assets_seen == {expected_source_asset}
    segments = int(gap_count + 1) if total_rows else 0
    case_record = {
        "case_key": case_row.case_key,
        "farm": farm,
        "asset_id": case_row.asset_id,
        "source_asset_id": expected_source_asset,
        "event_id": int(case_row.event_id),
        "rows": total_rows,
        "train_rows": train_rows,
        "prediction_rows": prediction_rows,
        "training_normal_rows": fit_rows,
        "unknown_split_rows": unknown_split_rows,
        "status_ids_seen": ";".join(sorted(status_values_seen)),
        "source_asset_ids_seen": ";".join(sorted(source_assets_seen)),
        "asset_id_matches_metadata": asset_match,
        "missing_timestamps": missing_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "nonmonotonic_timestamps": nonmonotonic_timestamps,
        "gap_count": gap_count,
        "continuous_segments": segments,
        "maximum_gap_minutes": maximum_gap_minutes,
    }

    channel_records: list[dict[str, Any]] = []
    for index, channel in enumerate(channels):
        availability = finite_count[index] / fit_rows if fit_rows else 0.0
        zero_fraction = zero_count[index] / finite_count[index] if finite_count[index] else np.nan
        all_missing = finite_count[index] == 0
        all_zero = finite_count[index] > 0 and zero_count[index] == finite_count[index]
        long_zero = longest_zero_run[index] >= QUALITY.constant_zero_run_min_steps
        low_availability = availability < QUALITY.minimum_sensor_availability

        if all_missing:
            review_flag = "all_missing_training_normal"
        elif all_zero:
            review_flag = "structurally_zero_training_normal"
        elif low_availability:
            review_flag = "low_availability_training_normal"
        elif long_zero:
            review_flag = "sustained_zero_run_review"
        else:
            review_flag = "pass"

        channel_records.append(
            {
                "case_key": case_row.case_key,
                "farm": farm,
                "asset_id": case_row.asset_id,
                "event_id": int(case_row.event_id),
                "column": channel,
                "training_normal_rows": fit_rows,
                "finite_rows": int(finite_count[index]),
                "missing_rows": int(fit_rows - finite_count[index]),
                "availability": float(availability),
                "zero_rows": int(zero_count[index]),
                "zero_fraction_of_finite": float(zero_fraction),
                "longest_zero_run_steps": int(longest_zero_run[index]),
                "all_missing": bool(all_missing),
                "all_zero": bool(all_zero),
                "low_availability": bool(low_availability),
                "sustained_zero_run": bool(long_zero),
                "review_flag": review_flag,
            }
        )
    return case_record, channel_records


_case_quality_records: list[dict[str, Any]] = []
_channel_quality_records: list[dict[str, Any]] = []

for _farm in DATASET.farms:
    _farm_registry = CASE_REGISTRY.loc[CASE_REGISTRY["farm"].eq(_farm)]
    print(
        f"Streaming training-normal quality audit — {_farm}: "
        f"{len(_farm_registry)} cases",
        flush=True,
    )
    for _case_row in _farm_registry.itertuples(index=False):
        _case_record, _channel_records = audit_one_case(_case_row)
        _case_quality_records.append(_case_record)
        _channel_quality_records.extend(_channel_records)

CASE_QUALITY_AUDIT = pd.DataFrame(_case_quality_records).sort_values(
    ["farm", "event_id"], kind="stable"
).reset_index(drop=True)
CHANNEL_QUALITY_AUDIT = pd.DataFrame(_channel_quality_records).sort_values(
    ["farm", "event_id", "column"], kind="stable"
).reset_index(drop=True)

_fatal_case_mask = (
    (CASE_QUALITY_AUDIT["rows"] <= 0)
    | (CASE_QUALITY_AUDIT["train_rows"] <= 0)
    | (CASE_QUALITY_AUDIT["prediction_rows"] <= 0)
    | (CASE_QUALITY_AUDIT["training_normal_rows"] < QUALITY.minimum_training_rows)
    | (CASE_QUALITY_AUDIT["unknown_split_rows"] > 0)
    | (~CASE_QUALITY_AUDIT["asset_id_matches_metadata"])
    | (CASE_QUALITY_AUDIT["missing_timestamps"] > 0)
    | (CASE_QUALITY_AUDIT["duplicate_timestamps"] > 0)
    | (CASE_QUALITY_AUDIT["nonmonotonic_timestamps"] > 0)
)
if _fatal_case_mask.any():
    _fatal_columns = [
        "case_key",
        "rows",
        "train_rows",
        "prediction_rows",
        "training_normal_rows",
        "unknown_split_rows",
        "asset_id_matches_metadata",
        "missing_timestamps",
        "duplicate_timestamps",
        "nonmonotonic_timestamps",
    ]
    raise RuntimeError(
        "Structural quality checks failed. No model may be fitted:\n"
        + CASE_QUALITY_AUDIT.loc[_fatal_case_mask, _fatal_columns].to_string(index=False)
    )

CHANNEL_QUALITY_SUMMARY = (
    CHANNEL_QUALITY_AUDIT.groupby(["farm", "column"], sort=False)
    .agg(
        cases=("case_key", "nunique"),
        assets=("asset_id", "nunique"),
        training_normal_rows=("training_normal_rows", "sum"),
        finite_rows=("finite_rows", "sum"),
        missing_rows=("missing_rows", "sum"),
        zero_rows=("zero_rows", "sum"),
        worst_case_availability=("availability", "min"),
        median_case_availability=("availability", "median"),
        longest_zero_run_steps=("longest_zero_run_steps", "max"),
        all_missing_cases=("all_missing", "sum"),
        all_zero_cases=("all_zero", "sum"),
        low_availability_cases=("low_availability", "sum"),
        sustained_zero_run_cases=("sustained_zero_run", "sum"),
    )
    .reset_index()
)
CHANNEL_QUALITY_SUMMARY["pooled_availability"] = (
    CHANNEL_QUALITY_SUMMARY["finite_rows"]
    / CHANNEL_QUALITY_SUMMARY["training_normal_rows"].clip(lower=1)
)
CHANNEL_QUALITY_SUMMARY["pooled_zero_fraction"] = (
    CHANNEL_QUALITY_SUMMARY["zero_rows"]
    / CHANNEL_QUALITY_SUMMARY["finite_rows"].replace(0, np.nan)
)

ZERO_AND_MISSINGNESS_REVIEW = CHANNEL_QUALITY_AUDIT.loc[
    ~CHANNEL_QUALITY_AUDIT["review_flag"].eq("pass")
].reset_index(drop=True)

QUALITY_FARM_SUMMARY = (
    CASE_QUALITY_AUDIT.groupby("farm", sort=False)
    .agg(
        cases=("case_key", "size"),
        assets=("asset_id", "nunique"),
        total_rows=("rows", "sum"),
        median_training_normal_rows=("training_normal_rows", "median"),
        median_prediction_rows=("prediction_rows", "median"),
        total_gaps=("gap_count", "sum"),
        median_segments=("continuous_segments", "median"),
        maximum_gap_minutes=("maximum_gap_minutes", "max"),
    )
    .reset_index()
)


# =============================================================================
# 6. Freeze the audit receipt, then write tables
# =============================================================================

_audit_component_hashes = {
    "safe_case_registry_sha256": dataframe_sha256(
        CASE_REGISTRY.assign(file_path=CASE_REGISTRY["file_path"].astype(str)),
        ("farm", "event_id"),
    ),
    "outcome_lockbox_sha256": OUTCOME_LOCKBOX_SHA256,
    "feature_registry_sha256": dataframe_sha256(
        CARE_FEATURE_REGISTRY, ("farm", "column")
    ),
    "case_quality_sha256": dataframe_sha256(
        CASE_QUALITY_AUDIT, ("farm", "event_id")
    ),
    "channel_quality_sha256": dataframe_sha256(
        CHANNEL_QUALITY_AUDIT, ("farm", "event_id", "column")
    ),
}
CELL2_AUDIT_SHA256 = sha256_json(
    {
        "contract_sha256": CONTRACT_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "component_hashes": _audit_component_hashes,
        "scope": "training-normal diagnostic audit; no global channel selection",
    }
)

CELL2_RECEIPT_PATH = Path(INVENTORY_DIR) / "cell2_inventory_quality_audit_receipt.json"
if CELL2_RECEIPT_PATH.exists():
    _existing_receipt = json.loads(CELL2_RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        _existing_receipt.get("contract_sha256") != CONTRACT_SHA256
        or _existing_receipt.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256
        or _existing_receipt.get("cell2_audit_sha256") != CELL2_AUDIT_SHA256
    ):
        raise RuntimeError(
            "A different Cell 2 audit already exists under this experiment ID. "
            "Do not overwrite it; investigate dataset or code drift and start a "
            "new experiment ID if the change is intentional."
        )
    CELL2_AUDIT_STATE = "existing identical audit verified"
    _write_new_receipt = False
else:
    CELL2_AUDIT_STATE = "new audit frozen"
    _write_new_receipt = True

_safe_registry_for_csv = CASE_REGISTRY.copy()
_safe_registry_for_csv["file_path"] = _safe_registry_for_csv["file_path"].astype(str)

save_csv_atomic(DATASET_FILE_MANIFEST, Path(INVENTORY_DIR) / "dataset_file_manifest.csv")
save_csv_atomic(_safe_registry_for_csv, Path(INVENTORY_DIR) / "care_case_registry_safe.csv")
save_csv_atomic(
    _OUTCOME_LOCKBOX,
    Path(INVENTORY_DIR) / "outcome_lockbox_do_not_load_before_outer_predictions.csv",
)
save_csv_atomic(FARM_SCHEMA_SUMMARY, Path(INVENTORY_DIR) / "farm_schema_summary.csv")
save_csv_atomic(CARE_FEATURE_REGISTRY, Path(INVENTORY_DIR) / "care_feature_registry.csv")
save_json(FARM_SCHEMAS, Path(INVENTORY_DIR) / "farm_schemas.json")
save_csv_atomic(CASE_QUALITY_AUDIT, Path(QUALITY_DIR) / "case_quality_audit.csv")
save_csv_atomic(CHANNEL_QUALITY_AUDIT, Path(QUALITY_DIR) / "channel_quality_audit.csv")
save_csv_atomic(CHANNEL_QUALITY_SUMMARY, Path(QUALITY_DIR) / "channel_quality_summary.csv")
save_csv_atomic(ZERO_AND_MISSINGNESS_REVIEW, Path(QUALITY_DIR) / "zero_missingness_review.csv")

if _write_new_receipt:
    save_json(
        {
            "contract_sha256": CONTRACT_SHA256,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "cell2_audit_sha256": CELL2_AUDIT_SHA256,
            "component_hashes": _audit_component_hashes,
            "outcome_count_check": OUTCOME_COUNT_CHECK,
            "quality_scope": (
                "Signal diagnostics use source training-partition rows with normal "
                "status only. They do not select globally retained channels."
            ),
            "safe_registry_columns": SAFE_CASE_COLUMNS,
            "outcome_lockbox_columns": (
                "case_key",
                "farm",
                "asset_id",
                "source_asset_id",
                "event_id",
                *OUTCOME_ONLY_COLUMNS,
            ),
            "created_at_utc": utc_now(),
        },
        CELL2_RECEIPT_PATH,
    )

# Remove the only global DataFrame containing outcome data. Later nested-scoring
# code must use a fold-gated accessor rather than an ambient notebook variable.
del _OUTCOME_LOCKBOX


# =============================================================================
# 7. Concise notebook report
# =============================================================================

print("\n" + "=" * 92)
print("UC-RCF-NBM CELL 2 — CARE V6 INVENTORY AND QUALITY AUDIT")
print("=" * 92)
display(FARM_SCHEMA_SUMMARY)

print("\nTRAINING-NORMAL QUALITY SUMMARY")
display(QUALITY_FARM_SUMMARY)

_review_counts = (
    ZERO_AND_MISSINGNESS_REVIEW.groupby(["farm", "review_flag"])
    .size()
    .rename("case_channel_pairs")
    .reset_index()
)
print("\nZERO/MISSINGNESS ITEMS FOR FOLD-LOCAL REVIEW")
if len(_review_counts):
    display(_review_counts)
else:
    print("No review flags were raised.")

print("\n" + "-" * 92)
print(f"Cases registered                 : {len(CASE_REGISTRY)}")
print(f"Farm-qualified assets           : {CASE_REGISTRY['asset_id'].nunique()}")
print(f"Anomaly/normal count check      : {OUTCOME_COUNT_CHECK['anomaly_cases']} / "
      f"{OUTCOME_COUNT_CHECK['normal_cases']}")
print(f"Event data size                 : {EVENT_FILE_INVENTORY['size_bytes'].sum() / 1024**3:.2f} GiB")
print(f"Dataset manifest SHA-256        : {DATASET_MANIFEST_SHA256}")
print(f"Outcome lockbox SHA-256         : {OUTCOME_LOCKBOX_SHA256}")
print(f"Cell 2 audit SHA-256            : {CELL2_AUDIT_SHA256}")
print(f"Audit state                     : {CELL2_AUDIT_STATE}")
print("Outcome fields in CASE_REGISTRY : No")
print("Global sensor selection applied : No — selection remains outer-fold local")
print("Model fitted                    : No")
print("Structural checks               : PASS")
print("=" * 92)
print("CELL 2 COMPLETED SUCCESSFULLY — CARE V6 INVENTORY AND QUALITY AUDIT LOCKED")
