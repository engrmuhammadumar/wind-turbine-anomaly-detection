"""Cell 3N adapter — rebuild the explicit raw sensor source from event CSVs.

Run after Cell 3M and immediately before cell_3n_train_only_preprocessing.py.
The adapter reads only events represented by modeling-eligible manifest rows,
detects the immutable source-row convention per event, and verifies timestamps.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


if "ROW_LABEL_MANIFEST" not in globals() or "FARMS" not in globals():
    raise NameError("Run Cells 3I–3M before the Cell 3N raw-file adapter.")

required_manifest_columns_3n_adapter = {
    "farm", "event_key", "source_row_index", "timestamp_utc",
    "modeling_eligible",
}
missing_manifest_columns_3n_adapter = (
    required_manifest_columns_3n_adapter - set(ROW_LABEL_MANIFEST.columns)
)
if missing_manifest_columns_3n_adapter:
    raise ValueError(
        "ROW_LABEL_MANIFEST lacks adapter columns: "
        f"{sorted(missing_manifest_columns_3n_adapter)}"
    )

eligible_keys_3n_adapter = ROW_LABEL_MANIFEST.loc[
    ROW_LABEL_MANIFEST["modeling_eligible"].fillna(False).astype(bool),
    ["farm", "event_key", "source_row_index", "timestamp_utc"],
].copy()
eligible_keys_3n_adapter["farm"] = (
    eligible_keys_3n_adapter["farm"].astype("string").str.strip().str.upper()
)
eligible_keys_3n_adapter["event_key"] = (
    eligible_keys_3n_adapter["event_key"].astype("string").str.strip()
)
eligible_keys_3n_adapter["source_row_index"] = pd.to_numeric(
    eligible_keys_3n_adapter["source_row_index"], errors="raise"
).astype("int64")
eligible_keys_3n_adapter["timestamp_utc"] = pd.to_datetime(
    eligible_keys_3n_adapter["timestamp_utc"], errors="raise", utc=True
)

if eligible_keys_3n_adapter[["event_key", "source_row_index"]].duplicated().any():
    raise ValueError("Eligible manifest source keys are not unique.")


def event_id_from_key_3n_adapter(event_key_3n_adapter):
    match_3n_adapter = re.fullmatch(
        r"farm_([abc])_event_(\d+)", str(event_key_3n_adapter), flags=re.I
    )
    if match_3n_adapter is None:
        raise ValueError(f"Unsupported event key: {event_key_3n_adapter!r}")
    return match_3n_adapter.group(1).upper(), int(match_3n_adapter.group(2))


def locate_event_file_3n_adapter(farm_3n_adapter, event_id_3n_adapter):
    farm_root_3n_adapter = Path(FARMS[farm_3n_adapter])
    direct_candidates_3n_adapter = [
        farm_root_3n_adapter / "datasets" / f"{event_id_3n_adapter}.csv",
        farm_root_3n_adapter / "dataset" / f"{event_id_3n_adapter}.csv",
        farm_root_3n_adapter / f"{event_id_3n_adapter}.csv",
    ]
    existing_3n_adapter = [
        path_3n_adapter
        for path_3n_adapter in direct_candidates_3n_adapter
        if path_3n_adapter.is_file()
    ]
    if len(existing_3n_adapter) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CSV for farm {farm_3n_adapter}, event "
            f"{event_id_3n_adapter}; found {existing_3n_adapter}."
        )
    return existing_3n_adapter[0]


def timestamp_match_count_3n_adapter(
    raw_frame_3n_adapter,
    manifest_event_3n_adapter,
    source_indices_3n_adapter,
):
    candidate_3n_adapter = pd.DataFrame({
        "source_row_index": pd.Series(source_indices_3n_adapter, dtype="int64"),
        "raw_timestamp_utc": raw_frame_3n_adapter["_timestamp_utc_3n"].to_numpy(),
    })
    if candidate_3n_adapter["source_row_index"].duplicated().any():
        return -1
    check_3n_adapter = manifest_event_3n_adapter.merge(
        candidate_3n_adapter,
        on="source_row_index",
        how="left",
        validate="one_to_one",
    )
    if check_3n_adapter["raw_timestamp_utc"].isna().any():
        return -1
    return int(
        check_3n_adapter["timestamp_utc"].eq(
            check_3n_adapter["raw_timestamp_utc"]
        ).sum()
    )


pieces_3n_adapter = []
audit_records_3n_adapter = []
sensor_schema_3n_adapter = None

for event_key_3n_adapter, manifest_event_3n_adapter in (
    eligible_keys_3n_adapter.groupby("event_key", sort=True, observed=True)
):
    farm_from_key_3n_adapter, event_id_3n_adapter = (
        event_id_from_key_3n_adapter(event_key_3n_adapter)
    )
    manifest_farms_3n_adapter = manifest_event_3n_adapter["farm"].unique().tolist()
    if manifest_farms_3n_adapter != [farm_from_key_3n_adapter]:
        raise ValueError(
            f"Farm mismatch for {event_key_3n_adapter}: "
            f"{manifest_farms_3n_adapter}."
        )

    event_path_3n_adapter = locate_event_file_3n_adapter(
        farm_from_key_3n_adapter, event_id_3n_adapter
    )
    raw_event_3n_adapter = pd.read_csv(
        event_path_3n_adapter,
        sep=";",
        low_memory=False,
    )
    if raw_event_3n_adapter.shape[1] <= 1:
        raise ValueError(
            f"Delimiter parsing failed for {event_path_3n_adapter}."
        )
    if "time_stamp" not in raw_event_3n_adapter.columns:
        raise ValueError(
            f"{event_path_3n_adapter} lacks the time_stamp column."
        )
    raw_event_3n_adapter["_timestamp_utc_3n"] = pd.to_datetime(
        raw_event_3n_adapter["time_stamp"], errors="coerce", utc=True
    )
    if raw_event_3n_adapter["_timestamp_utc_3n"].isna().any():
        raise ValueError(f"Invalid timestamps in {event_path_3n_adapter}.")

    sensor_columns_3n_adapter = [
        column_3n_adapter
        for column_3n_adapter in raw_event_3n_adapter.columns
        if re.search(r"_(avg|max|min|std)$", str(column_3n_adapter), flags=re.I)
    ]
    if not sensor_columns_3n_adapter:
        raise ValueError(f"No measurement columns in {event_path_3n_adapter}.")
    if sensor_schema_3n_adapter is None:
        sensor_schema_3n_adapter = sensor_columns_3n_adapter
    elif sensor_columns_3n_adapter != sensor_schema_3n_adapter:
        raise ValueError(
            f"Measurement schema mismatch in {event_path_3n_adapter}."
        )

    index_candidates_3n_adapter = {
        "zero_based_file_row": np.arange(len(raw_event_3n_adapter), dtype=np.int64),
        "one_based_file_row": np.arange(1, len(raw_event_3n_adapter) + 1, dtype=np.int64),
    }
    if "id" in raw_event_3n_adapter.columns:
        id_values_3n_adapter = pd.to_numeric(
            raw_event_3n_adapter["id"], errors="coerce"
        )
        if (
            id_values_3n_adapter.notna().all()
            and np.isclose(id_values_3n_adapter, np.floor(id_values_3n_adapter)).all()
        ):
            index_candidates_3n_adapter["id_column"] = (
                id_values_3n_adapter.astype("int64").to_numpy()
            )

    match_counts_3n_adapter = {
        name_3n_adapter: timestamp_match_count_3n_adapter(
            raw_event_3n_adapter,
            manifest_event_3n_adapter,
            values_3n_adapter,
        )
        for name_3n_adapter, values_3n_adapter in index_candidates_3n_adapter.items()
    }
    required_matches_3n_adapter = len(manifest_event_3n_adapter)
    exact_modes_3n_adapter = [
        name_3n_adapter
        for name_3n_adapter, count_3n_adapter in match_counts_3n_adapter.items()
        if count_3n_adapter == required_matches_3n_adapter
    ]
    if not exact_modes_3n_adapter:
        raise ValueError(
            f"No row-identity convention exactly matches timestamps for "
            f"{event_key_3n_adapter}: {match_counts_3n_adapter}."
        )

    # Equivalent conventions (commonly id == zero-based row) are harmless.
    chosen_mode_3n_adapter = exact_modes_3n_adapter[0]
    chosen_indices_3n_adapter = index_candidates_3n_adapter[
        chosen_mode_3n_adapter
    ]
    for other_mode_3n_adapter in exact_modes_3n_adapter[1:]:
        if not np.array_equal(
            chosen_indices_3n_adapter,
            index_candidates_3n_adapter[other_mode_3n_adapter],
        ):
            raise ValueError(
                f"Ambiguous non-equivalent row identities for "
                f"{event_key_3n_adapter}: {exact_modes_3n_adapter}."
            )

    keyed_event_3n_adapter = raw_event_3n_adapter.loc[
        :, sensor_schema_3n_adapter
    ].copy()
    keyed_event_3n_adapter.insert(0, "source_row_index", chosen_indices_3n_adapter)
    keyed_event_3n_adapter.insert(0, "event_key", str(event_key_3n_adapter))
    required_indices_3n_adapter = set(
        manifest_event_3n_adapter["source_row_index"].astype(int)
    )
    keyed_event_3n_adapter = keyed_event_3n_adapter.loc[
        keyed_event_3n_adapter["source_row_index"].isin(required_indices_3n_adapter)
    ].copy()
    if len(keyed_event_3n_adapter) != required_matches_3n_adapter:
        raise ValueError(f"Row conservation failed for {event_key_3n_adapter}.")
    pieces_3n_adapter.append(keyed_event_3n_adapter)
    audit_records_3n_adapter.append({
        "event_key": str(event_key_3n_adapter),
        "event_path": str(event_path_3n_adapter),
        "file_rows": int(len(raw_event_3n_adapter)),
        "eligible_rows": int(required_matches_3n_adapter),
        "row_identity_mode": chosen_mode_3n_adapter,
        "timestamp_matches": int(required_matches_3n_adapter),
        "measurement_columns": int(len(sensor_schema_3n_adapter)),
    })

FEATURE_SOURCE_3N = pd.concat(
    pieces_3n_adapter,
    axis=0,
    ignore_index=True,
    sort=False,
)
FEATURE_COLUMNS_3N = list(sensor_schema_3n_adapter)
RAW_FILE_ADAPTER_AUDIT_3N = pd.DataFrame(audit_records_3n_adapter)

if len(FEATURE_SOURCE_3N) != len(eligible_keys_3n_adapter):
    raise ValueError("The adapter did not conserve all eligible rows.")
if FEATURE_SOURCE_3N[["event_key", "source_row_index"]].duplicated().any():
    raise ValueError("The adapter produced duplicate immutable source keys.")
if not pd.MultiIndex.from_frame(eligible_keys_3n_adapter[[
    "event_key", "source_row_index"
]]).isin(pd.MultiIndex.from_frame(FEATURE_SOURCE_3N[[
    "event_key", "source_row_index"
]])).all():
    raise ValueError("The adapter failed to cover every eligible source key.")

print("Cell 3N raw-file adapter completed successfully.")
print("Events loaded:", RAW_FILE_ADAPTER_AUDIT_3N["event_key"].nunique())
print("Eligible rows reconstructed:", len(FEATURE_SOURCE_3N))
print("Physical measurement columns:", len(FEATURE_COLUMNS_3N))
print("Delimiter: semicolon")
print("All manifest-to-file timestamps matched exactly.")
