"""CELL 1 — UC-RCF-NBM configuration and frozen experiment contract.

Paste this complete file into the first cell of a new notebook.  The cell creates
an isolated output tree for Experiment 2 and freezes every choice that may affect
the subsequent nested CARE evaluation.  It does not read event labels or sensor
measurements.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Keep numerical libraries reproducible and avoid thread oversubscription.  These
# values are set before importing NumPy; setdefault preserves an explicit user
# choice made before this cell runs.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd


# =============================================================================
# 1. Paths and experiment identity
# =============================================================================

METHOD_NAME = "UC-RCF-NBM"
METHOD_LONG_NAME = (
    "Uncertainty-Calibrated Robust Cross-Fitted Normal-Behaviour Model"
)
METHOD_VERSION = "1.0.0-pre-outer-evaluation"
EXPERIMENT_ID = "uc_rcf_nbm_care_v6_exp01"

PROJECT_ROOT = Path.cwd().resolve()

# Environment overrides make the cell portable without changing the scientific
# contract.  On Umar's workstation the defaults point to the existing CARE data
# and the Implementation output tree.
CARE_DATA_ROOT = Path(
    os.environ.get(
        "UC_RCF_NBM_DATA_ROOT",
        r"F:\Umar-Wisal-Work\Wisal-Bearings-Work\CARE_To_Compare\CARE_To_Compare",
    )
).resolve()

OUTPUT_ROOT = Path(
    os.environ.get(
        "UC_RCF_NBM_OUTPUT_ROOT",
        str(PROJECT_ROOT / "outputs" / EXPERIMENT_ID),
    )
).resolve()

CONTRACT_DIR = OUTPUT_ROOT / "00_contract"
INVENTORY_DIR = OUTPUT_ROOT / "01_inventory"
QUALITY_DIR = OUTPUT_ROOT / "02_quality"
CACHE_DIR = OUTPUT_ROOT / "03_cache"
MODEL_DIR = OUTPUT_ROOT / "04_models"
PREDICTION_DIR = OUTPUT_ROOT / "05_outer_predictions"
TABLE_DIR = OUTPUT_ROOT / "06_tables"
FIGURE_DIR = OUTPUT_ROOT / "07_figures"
LOG_DIR = OUTPUT_ROOT / "08_logs"

OUTPUT_DIRECTORIES = (
    OUTPUT_ROOT,
    CONTRACT_DIR,
    INVENTORY_DIR,
    QUALITY_DIR,
    CACHE_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    LOG_DIR,
)


# =============================================================================
# 2. Frozen scientific configuration
# =============================================================================

@dataclass(frozen=True)
class DatasetContract:
    """Expected properties of the current CARE-to-Compare release."""

    name: str = "Wind Turbine SCADA Data For Early Fault Detection"
    release: str = "v6"
    zenodo_doi: str = "10.5281/zenodo.15846963"
    sampling_minutes: int = 10
    farms: tuple[str, ...] = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
    expected_cases_by_farm: tuple[int, ...] = (22, 15, 58)
    expected_anomaly_by_farm: tuple[int, ...] = (12, 6, 27)
    expected_normal_by_farm: tuple[int, ...] = (10, 9, 31)
    expected_total_cases: int = 95
    expected_anomaly_cases: int = 45
    expected_normal_cases: int = 50
    expected_assets: int = 36
    metadata_columns: tuple[str, ...] = (
        "id",
        "time_stamp",
        "asset_id",
        "train_test",
        "status_type_id",
    )
    source_train_labels: tuple[str, ...] = ("train", "training", "0")
    source_prediction_labels: tuple[str, ...] = (
        "test",
        "prediction",
        "predict",
        "1",
    )
    normal_status_ids: tuple[int, ...] = (0, 2)


@dataclass(frozen=True)
class QualityContract:
    """Primary signal-quality and temporal-continuity rules."""

    primary_statistics: tuple[str, ...] = ("avg",)
    sensitivity_statistics: tuple[str, ...] = ("avg", "std")
    standard_deviation_primary: bool = False
    maximum_gap_minutes: int = 10
    minimum_training_rows: int = 2_000
    minimum_sensor_availability: float = 0.70
    minimum_segment_steps: int = 144
    zero_coded_missingness_policy: str = (
        "metadata-guided plus training-only constant-zero-run audit"
    )
    constant_zero_run_min_steps: int = 144
    counter_policy: str = "within-segment first difference; negative resets missing"
    angle_policy: str = "sine-cosine encoding"
    driver_imputation: str = "training-normal median"
    target_imputation: str = "masked from fitting and scoring"
    temporal_operations_restart_at_gaps: bool = True


@dataclass(frozen=True)
class MeanModelContract:
    """Operating-conditioned mean response model."""

    model: str = "multi-output ridge normal-behaviour model"
    basis: str = "linear + quadratic + wind cubic + restricted physical interactions"
    driver_description_patterns: tuple[str, ...] = (
        "wind speed",
        "active power",
        "reactive power",
        "rotor speed",
        "generator speed",
        "generator acceleration",
        "torque",
        "ambient temperature",
    )
    maximum_core_interaction_drivers: int = 5
    ridge_grid: tuple[float, ...] = (
        1.0e-4,
        1.0e-2,
        1.0,
        1.0e2,
        1.0e4,
    )
    crossfit_folds: int = 5
    temporal_block_steps: int = 432
    embargo_steps: int = 144
    penalty_objective: str = "masked out-of-fold mean squared error"
    modelability_r2_grid: tuple[float, ...] = (0.0, 0.10, 0.30)
    fallback_minimum_r2: float = 0.0
    residual_cap: float = 10.0


@dataclass(frozen=True)
class UncertaintyContract:
    """Aleatoric, support, and blocked conformal uncertainty rules."""

    aleatoric_model: str = "multi-output ridge on log squared cross-fitted residual"
    scale_ridge_grid: tuple[float, ...] = (
        1.0e-4,
        1.0e-2,
        1.0,
        1.0e2,
        1.0e4,
    )
    scale_target_epsilon: float = 1.0e-6
    scale_crossfit_folds: int = 5
    support_uncertainty: str = "regularized nonlinear-design leverage"
    support_reference_scaling: str = "training cross-fitted median and IQR"
    support_nonnegative_for_total_scale: bool = True
    total_scale_rule: str = "aleatoric_scale * sqrt(1 + positive_support_uncertainty)"
    conformal_method: str = "embargoed blocked cross-conformal calibration"
    conformal_coverage_grid: tuple[float, ...] = (0.95, 0.98, 0.99)
    conformal_block_steps: int = 144
    minimum_conformal_blocks: int = 30
    finite_sample_quantile_correction: bool = True
    exact_exchangeability_claimed: bool = False
    primary_uncertainty_outputs: tuple[str, ...] = (
        "aleatoric_scale",
        "support_uncertainty",
        "total_predictive_scale",
        "conformal_interval",
        "interval_exceedance",
    )


@dataclass(frozen=True)
class AlarmContract:
    """Multichannel health evidence and sequential decision rule."""

    sensor_weighting: str = "cross-fitted modelability and calibration quality"
    health_indicator: str = "weighted mean of positive conformal interval excess"
    evidence_head_grid: tuple[str, ...] = (
        "all_modellable",
        "temperature_modellable",
    )
    smoothing_steps_grid: tuple[int, ...] = (36, 72, 144)
    smoothing: str = "causal rolling median within continuous segments"
    health_threshold_quantile_grid: tuple[float, ...] = (
        0.95,
        0.98,
        0.99,
        0.995,
    )
    timestamp_rule: str = "smoothed health indicator strictly exceeds frozen threshold"
    post_alarm_latching: bool = False
    criticality_feedback_into_predictions: bool = False


@dataclass(frozen=True)
class EvaluationContract:
    """Nested asset-grouped validation and official CARE scoring."""

    outer_strategy: str = "leave-one-asset-out"
    outer_group_column: str = "asset_id"
    inner_strategy: str = "grouped folds by asset on outer-development assets only"
    inner_group_column: str = "asset_id"
    inner_splits: int = 5
    selection_objective: str = "official CARE"
    deterministic_tie_break: tuple[str, ...] = (
        "higher normal accuracy",
        "higher event F0.5",
        "narrower mean normal interval",
        "lexicographic configuration",
    )
    outer_labels_available_during_selection: bool = False
    outer_predictions_generated_once: bool = True
    legacy_transformer_test_is_confirmation: bool = False
    primary_endpoint: str = "pooled outer-fold official CARE"
    secondary_endpoints: tuple[str, ...] = (
        "coverage F0.5",
        "normal accuracy",
        "event reliability F0.5",
        "earliness weighted score",
        "event sensitivity",
        "event false-alarm count",
        "median lead time",
        "normal prediction-interval coverage",
        "normal interval width",
        "coverage error",
        "interval score",
        "selective risk",
    )
    bootstrap_unit: str = "asset"
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 42
    multiple_comparison_adjustment: str = "Holm"


@dataclass(frozen=True)
class CareContract:
    """Published CARE settings retained for benchmark comparability."""

    beta: float = 0.5
    component_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0)
    criticality_threshold: int = 72
    criticality_update_state: str = "normal/actionable timestamps"
    raw_timestamp_predictions_only: bool = True
    farm_a_anomaly_prediction_status_policy: str = (
        "ignore status mask as prescribed by CARE v6 dataset notes"
    )
    farm_a_normal_prediction_status_policy: str = "normal-status timestamps"
    farms_b_c_prediction_status_policy: str = "normal-status timestamps"
    official_reference_unit_tests_required: bool = True


@dataclass(frozen=True)
class ReproducibilityContract:
    seed: int = 42
    deterministic_algorithms: bool = True
    float_dtype: str = "float64 for fitting; float32 permitted for caches"
    source_files_read_only: bool = True
    contract_change_requires_new_experiment_id: bool = True
    save_outer_predictions_before_label_scoring: bool = True


DATASET = DatasetContract()
QUALITY = QualityContract()
MEAN_MODEL = MeanModelContract()
UNCERTAINTY = UncertaintyContract()
ALARM = AlarmContract()
EVALUATION = EvaluationContract()
CARE = CareContract()
REPRODUCIBILITY = ReproducibilityContract()


# Predictor and calibration code may access sensor values, timestamps, partition
# identity, and asset identity.  The following fields are forbidden until an
# outer prediction has been frozen.  status_type_id is used only to select normal
# fitting rows and by the final CARE mask; it is never a predictor input.
FORBIDDEN_PREDICTOR_FIELDS = (
    "event_label",
    "event_label_raw",
    "is_anomaly",
    "event_start",
    "event_start_id",
    "event_end",
    "event_end_id",
    "event_description",
    "fault_type",
    "care_ground_truth",
)

PERMITTED_STATUS_USES = (
    "normal-training row eligibility",
    "final official CARE actionable mask",
)

LABEL_ACCESS_RULES = {
    "case_preprocessing": "event/status outcome fields forbidden as predictors",
    "mean_model_fit": "normal training partition only; no prediction event labels",
    "uncertainty_model_fit": "cross-fitted normal training residuals only",
    "conformal_calibration": "cross-fitted normal training nonconformity only",
    "inner_selection": "labels from inner-development assets only",
    "outer_inference": "outer event labels and boundaries unopened",
    "outer_scoring": "labels opened only after outer predictions are saved and hashed",
}


# =============================================================================
# 3. Validation, serialization, and reproducibility helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def save_json(data: Any, output_path: Path) -> None:
    """Atomically save human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            data,
            default=json_default,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if REPRODUCIBILITY.deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def validate_contract() -> None:
    if sum(DATASET.expected_cases_by_farm) != DATASET.expected_total_cases:
        raise ValueError("Farm case counts do not equal expected_total_cases.")
    if sum(DATASET.expected_anomaly_by_farm) != DATASET.expected_anomaly_cases:
        raise ValueError("Farm anomaly counts do not equal expected_anomaly_cases.")
    if sum(DATASET.expected_normal_by_farm) != DATASET.expected_normal_cases:
        raise ValueError("Farm normal counts do not equal expected_normal_cases.")
    if DATASET.expected_anomaly_cases + DATASET.expected_normal_cases != DATASET.expected_total_cases:
        raise ValueError("Anomaly and normal totals do not equal total cases.")
    if len(DATASET.farms) != len(DATASET.expected_cases_by_farm):
        raise ValueError("Farm names and expected counts have different lengths.")
    if QUALITY.primary_statistics != ("avg",):
        raise ValueError("Primary analysis must remain average-signal only.")
    if QUALITY.maximum_gap_minutes != DATASET.sampling_minutes:
        raise ValueError("A continuity gap must begin after one missing sample.")
    if not 0.0 < QUALITY.minimum_sensor_availability <= 1.0:
        raise ValueError("minimum_sensor_availability must lie in (0, 1].")
    if MEAN_MODEL.crossfit_folds < 3 or UNCERTAINTY.scale_crossfit_folds < 3:
        raise ValueError("At least three blocked cross-fitting folds are required.")
    if MEAN_MODEL.embargo_steps <= 0:
        raise ValueError("Temporal cross-fitting requires a positive embargo.")
    if any(not 0.0 < value < 1.0 for value in UNCERTAINTY.conformal_coverage_grid):
        raise ValueError("Conformal coverage levels must lie in (0, 1).")
    if any(not 0.0 < value < 1.0 for value in ALARM.health_threshold_quantile_grid):
        raise ValueError("Health threshold quantiles must lie in (0, 1).")
    if ALARM.post_alarm_latching or ALARM.criticality_feedback_into_predictions:
        raise ValueError("Post-alarm latching and CARE feedback are prohibited.")
    if not CARE.raw_timestamp_predictions_only:
        raise ValueError("Official CARE must receive raw timestamp predictions.")
    if EVALUATION.outer_group_column != "asset_id":
        raise ValueError("Outer evaluation must remain grouped by asset_id.")
    if EVALUATION.outer_labels_available_during_selection:
        raise ValueError("Outer labels cannot be available during selection.")
    if EVALUATION.legacy_transformer_test_is_confirmation:
        raise ValueError("The opened legacy Transformer test cannot be confirmatory.")


validate_contract()
set_global_seed(REPRODUCIBILITY.seed)

if not CARE_DATA_ROOT.is_dir():
    raise FileNotFoundError(
        "CARE v6 dataset root not found. Set UC_RCF_NBM_DATA_ROOT before running "
        f"Cell 1. Current path:\n{CARE_DATA_ROOT}"
    )

for _directory in OUTPUT_DIRECTORIES:
    _directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 4. Freeze the scientific contract
# =============================================================================

SCIENTIFIC_CONTRACT = {
    "method": {
        "name": METHOD_NAME,
        "long_name": METHOD_LONG_NAME,
        "version": METHOD_VERSION,
        "experiment_id": EXPERIMENT_ID,
    },
    "dataset": asdict(DATASET),
    "quality": asdict(QUALITY),
    "mean_model": asdict(MEAN_MODEL),
    "uncertainty": asdict(UNCERTAINTY),
    "alarm": asdict(ALARM),
    "evaluation": asdict(EVALUATION),
    "care": asdict(CARE),
    "reproducibility": asdict(REPRODUCIBILITY),
    "forbidden_predictor_fields": FORBIDDEN_PREDICTOR_FIELDS,
    "permitted_status_uses": PERMITTED_STATUS_USES,
    "label_access_rules": LABEL_ACCESS_RULES,
}

CONTRACT_SHA256 = sha256_json(SCIENTIFIC_CONTRACT)
CONTRACT_PATH = CONTRACT_DIR / "scientific_experiment_contract.json"

if CONTRACT_PATH.exists():
    existing_contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    existing_hash = str(existing_contract.get("contract_sha256", ""))
    if existing_hash != CONTRACT_SHA256:
        raise RuntimeError(
            "A different scientific contract already exists for this experiment ID. "
            "Do not overwrite it. Change EXPERIMENT_ID to start a genuinely new "
            f"experiment. Existing={existing_hash or '<missing>'}, "
            f"current={CONTRACT_SHA256}."
        )
    CONTRACT_STATE = "existing identical contract verified"
else:
    save_json(
        {
            "contract_sha256": CONTRACT_SHA256,
            "created_at_utc": utc_now(),
            "scientific_contract": SCIENTIFIC_CONTRACT,
        },
        CONTRACT_PATH,
    )
    CONTRACT_STATE = "new contract frozen"


ENVIRONMENT_SNAPSHOT = {
    "recorded_at_utc": utc_now(),
    "experiment_id": EXPERIMENT_ID,
    "contract_sha256": CONTRACT_SHA256,
    "project_root": str(PROJECT_ROOT),
    "dataset_root": str(CARE_DATA_ROOT),
    "output_root": str(OUTPUT_ROOT),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "packages": {
        name: package_version(name)
        for name in (
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "torch",
        )
    },
}
save_json(ENVIRONMENT_SNAPSHOT, CONTRACT_DIR / "environment_snapshot.json")


# =============================================================================
# 5. Cell summary
# =============================================================================

print("=" * 92)
print("UC-RCF-NBM EXPERIMENT INITIALIZED — SCIENTIFIC CONTRACT FROZEN")
print("=" * 92)
print(f"Method                         : {METHOD_LONG_NAME}")
print(f"Experiment ID                  : {EXPERIMENT_ID}")
print(f"CARE dataset release           : {DATASET.release} ({DATASET.zenodo_doi})")
print(
    "Expected cases                 : "
    f"{DATASET.expected_total_cases} "
    f"({DATASET.expected_anomaly_cases} anomaly / "
    f"{DATASET.expected_normal_cases} normal)"
)
print(f"Expected assets                : {DATASET.expected_assets}")
print(f"Primary feature statistics     : {QUALITY.primary_statistics}")
print(f"Mean model                     : {MEAN_MODEL.model}")
print(f"Temporal cross-fitting         : {MEAN_MODEL.crossfit_folds} folds, "
      f"{MEAN_MODEL.temporal_block_steps} steps/block, "
      f"{MEAN_MODEL.embargo_steps} steps embargo")
print(f"Aleatoric uncertainty          : {UNCERTAINTY.aleatoric_model}")
print(f"Support uncertainty            : {UNCERTAINTY.support_uncertainty}")
print(f"Conformal calibration          : {UNCERTAINTY.conformal_method}")
print(f"Outer evaluation               : {EVALUATION.outer_strategy}")
print(f"Primary endpoint               : {EVALUATION.primary_endpoint}")
print(f"Post-alarm latching            : {ALARM.post_alarm_latching}")
print(f"Legacy Transformer confirmation: {EVALUATION.legacy_transformer_test_is_confirmation}")
print(f"Contract state                 : {CONTRACT_STATE}")
print(f"Contract SHA-256               : {CONTRACT_SHA256}")
print(f"Dataset root                   : {CARE_DATA_ROOT}")
print(f"Output root                    : {OUTPUT_ROOT}")
print("Source data modified           : No — read-only contract")
print("=" * 92)
print("CELL 1 COMPLETED SUCCESSFULLY — UC-RCF-NBM EXPERIMENT CONTRACT LOCKED")

