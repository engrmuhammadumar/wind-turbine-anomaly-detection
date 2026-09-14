"""
PINDP-Net: end-to-end, leakage-safe PHM 2010 training and evaluation.

This script performs the complete reproducible workflow:

1. Discovers the six PHM 2010 cutter trajectories from raw CSV files.
2. Extracts pass-level features from the seven synchronized sensor channels.
3. Reads the three flute-wear measurements for C1, C4 and C6.
4. Defines mean flank wear and fixed-threshold RUL consistently.
5. Fits imputation, scaling and feature selection on training cutters only.
6. Selects hyperparameters with nested cutter-level validation.
7. Trains a strictly causal, unidirectional, physics-informed neural
   differential model with non-negative wear increments.
8. Produces untouched leave-one-cutter-out predictions for C1/C4/C6.
9. Fits the final model on C1/C4/C6 and performs blind inference on C2/C3/C5.
10. Calibrates uncertainty using development cutters only and exports metrics,
    audit files, checkpoints, CSV predictions and publication PNG figures.

Scientific-integrity safeguards
--------------------------------
* No random pass/window-level train/test split is used.
* No future machining pass enters a prediction at pass t.
* C2/C3/C5 never enter a supervised loss or reference-dependent metric.
* The fixed threshold is 165 micrometres for every labelled cutter.
* RUL_true(t) = max(T_fail - t, 0), in cutting passes.
* No prediction is smoothed, blended with the reference, or altered for plots.

Install once (Python 3.10+ recommended):
    pip install numpy pandas scipy PyWavelets scikit-learn matplotlib joblib torch

Example:
    python pindpnet_end_to_end.py \
        --data-root "E:/PHM_2010" \
        --output-dir "F:/GIST work/Review Round 1/pindpnet_causal"

Use --quick only to verify the pipeline. Do not report quick-mode results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import stft, welch
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import pywt
except ImportError:  # Allows --help to work and gives a focused runtime message.
    pywt = None


# =============================================================================
# 1. REPRODUCIBLE CONFIGURATION
# =============================================================================


LABELLED_CUTTERS = ("c1", "c4", "c6")
BLIND_CUTTERS = ("c2", "c3", "c5")
ALL_CUTTERS = ("c1", "c2", "c3", "c4", "c5", "c6")

SENSOR_CHANNELS = (
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "ae_rms",
)


@dataclass(frozen=True)
class StudyConfig:
    # Dataset and target definition
    sampling_hz: float = 50_000.0
    spindle_rpm: float = 10_400.0
    feed_mm_min: float = 1_555.0
    radial_depth_mm: float = 0.125
    axial_depth_mm: float = 0.2
    cutter_diameter_m: float = 0.006
    workpiece_hardness_hrc: float = 52.0
    failure_threshold_um: float = 165.0
    wear_file_scale_to_um: float = 1.0
    nominal_max_passes: float = 315.0

    # Feature extraction
    max_signal_points: int = 65_536
    welch_nperseg: int = 4_096
    wavelet: str = "db4"
    wavelet_level: int = 4
    local_segments: int = 8
    include_known_age: bool = True
    max_selected_features: int = 128

    # Optimization
    ensemble_members: int = 3
    max_epochs: int = 450
    patience: int = 70
    min_epochs: int = 80
    steps_per_epoch: int = 6
    weight_decay: float = 1.0e-4
    gradient_clip: float = 5.0
    uncertainty_z: float = 1.96
    num_workers: int = 0
    seed: int = 2026


@dataclass(frozen=True)
class HyperParameters:
    name: str
    hidden_dim: int
    conv_dim: int
    gru_layers: int
    dropout: float
    learning_rate: float
    physics_scale: float
    neural_rate_scale: float
    wear_huber_weight: float
    wear_nll_weight: float
    rul_huber_weight: float
    rul_nll_weight: float
    flute_weight: float
    threshold_consistency_weight: float
    rul_slope_weight: float
    rate_smoothness_weight: float
    residual_weight: float


def candidate_hyperparameters(quick: bool) -> list[HyperParameters]:
    candidates = [
        HyperParameters(
            name="balanced_96",
            hidden_dim=96,
            conv_dim=32,
            gru_layers=1,
            dropout=0.15,
            learning_rate=8.0e-4,
            physics_scale=0.012,
            neural_rate_scale=0.020,
            wear_huber_weight=1.00,
            wear_nll_weight=0.08,
            rul_huber_weight=0.65,
            rul_nll_weight=0.05,
            flute_weight=0.25,
            threshold_consistency_weight=0.12,
            rul_slope_weight=0.10,
            rate_smoothness_weight=0.08,
            residual_weight=0.02,
        ),
        HyperParameters(
            name="compact_64",
            hidden_dim=64,
            conv_dim=24,
            gru_layers=1,
            dropout=0.10,
            learning_rate=1.0e-3,
            physics_scale=0.015,
            neural_rate_scale=0.015,
            wear_huber_weight=1.00,
            wear_nll_weight=0.06,
            rul_huber_weight=0.75,
            rul_nll_weight=0.04,
            flute_weight=0.20,
            threshold_consistency_weight=0.15,
            rul_slope_weight=0.12,
            rate_smoothness_weight=0.10,
            residual_weight=0.03,
        ),
        HyperParameters(
            name="wide_128",
            hidden_dim=128,
            conv_dim=40,
            gru_layers=2,
            dropout=0.20,
            learning_rate=6.0e-4,
            physics_scale=0.010,
            neural_rate_scale=0.024,
            wear_huber_weight=1.00,
            wear_nll_weight=0.10,
            rul_huber_weight=0.60,
            rul_nll_weight=0.06,
            flute_weight=0.30,
            threshold_consistency_weight=0.10,
            rul_slope_weight=0.08,
            rate_smoothness_weight=0.06,
            residual_weight=0.015,
        ),
    ]
    return candidates[:1] if quick else candidates


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, indent=2, ensure_ascii=False)


# =============================================================================
# 2. RAW DATA DISCOVERY AND VALIDATION
# =============================================================================


_CUTTER_RE = re.compile(r"(?:^|[^a-z0-9])c[_\- ]?([1-6])(?:[^0-9]|$)", re.I)
_SIGNAL_STEM_RE = re.compile(
    r"c[_\- ]?([1-6])[_\- ]+(\d{1,4})(?:[^0-9]|$)", re.I
)


def infer_cutter(path: Path) -> str | None:
    searchable = " ".join([path.stem, *path.parts[-4:-1]])
    match = _CUTTER_RE.search(searchable)
    return f"c{match.group(1)}" if match else None


def infer_pass_number(path: Path, cutter: str) -> int | None:
    match = _SIGNAL_STEM_RE.search(path.stem)
    if match and f"c{match.group(1)}" == cutter:
        return int(match.group(2))

    numbers = [int(token) for token in re.findall(r"\d+", path.stem)]
    cutter_number = int(cutter[1:])
    candidates = [n for n in numbers if n != cutter_number and 1 <= n <= 10_000]
    return candidates[-1] if candidates else None


def discover_dataset(data_root: Path) -> tuple[dict[str, list[tuple[int, Path]]], dict[str, Path]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist:\n{data_root}")

    all_csv = sorted(p for p in data_root.rglob("*.csv") if p.is_file())
    if not all_csv:
        raise FileNotFoundError(f"No CSV files were found below:\n{data_root}")

    wear_files: dict[str, Path] = {}
    signal_records: dict[str, list[tuple[int, Path]]] = {c: [] for c in ALL_CUTTERS}

    for path in all_csv:
        cutter = infer_cutter(path)
        if cutter is None:
            continue
        is_wear = "wear" in path.stem.lower() or "wear" in path.parent.name.lower()
        if is_wear:
            if cutter in LABELLED_CUTTERS:
                if cutter in wear_files and wear_files[cutter] != path:
                    raise RuntimeError(
                        f"Multiple wear files were found for {cutter.upper()}:\n"
                        f"  {wear_files[cutter]}\n  {path}\n"
                        "Keep only the intended official wear file or reorganize DATA_ROOT."
                    )
                wear_files[cutter] = path
            continue

        pass_number = infer_pass_number(path, cutter)
        if pass_number is not None:
            signal_records[cutter].append((pass_number, path))

    for cutter in ALL_CUTTERS:
        rows = signal_records[cutter]
        if not rows:
            raise RuntimeError(
                f"No pass-level sensor CSV files were discovered for {cutter.upper()}. "
                "Expected names similar to c_1_001.csv within cutter directories."
            )
        counts = Counter(pass_no for pass_no, _ in rows)
        duplicates = sorted(k for k, v in counts.items() if v > 1)
        if duplicates:
            raise RuntimeError(
                f"Duplicate pass numbers for {cutter.upper()}: {duplicates[:20]}"
            )
        signal_records[cutter] = sorted(rows, key=lambda item: item[0])

    missing_wear = sorted(set(LABELLED_CUTTERS).difference(wear_files))
    if missing_wear:
        raise RuntimeError(
            "Wear files were not found for labelled cutters: "
            + ", ".join(c.upper() for c in missing_wear)
        )

    return signal_records, wear_files


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_sensor_csv(path: Path) -> np.ndarray:
    raw = pd.read_csv(path, header=None, low_memory=False)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    minimum_numeric = max(50, int(0.50 * len(numeric)))
    usable = [c for c in numeric.columns if numeric[c].notna().sum() >= minimum_numeric]
    if len(usable) < 7:
        raise ValueError(
            f"{path} contains only {len(usable)} usable numeric columns; seven are required."
        )
    values = numeric[usable[:7]].dropna(how="any").to_numpy(dtype=np.float64)
    if values.shape[0] < 128:
        raise ValueError(f"{path} contains only {values.shape[0]} complete rows.")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite sensor values remain in {path}.")
    return values


def read_wear_csv(path: Path, cutter: str, scale_to_um: float) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, low_memory=False)
    numeric = raw.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 3:
        raise ValueError(f"Wear file must contain at least three numeric columns: {path}")

    array = numeric.to_numpy(dtype=float)
    if array.shape[1] >= 4:
        first = array[:, 0]
        looks_like_pass = (
            np.isfinite(first).all()
            and len(np.unique(first)) == len(first)
            and np.all(np.diff(first) > 0)
            and np.nanmin(first) >= 0
        )
    else:
        looks_like_pass = False

    if looks_like_pass:
        cut_number = array[:, 0].astype(int)
        flutes = array[:, 1:4]
    else:
        cut_number = np.arange(1, len(array) + 1, dtype=int)
        flutes = array[:, -3:]

    flutes = flutes * float(scale_to_um)
    if not np.isfinite(flutes).all():
        raise ValueError(f"Wear file contains missing/non-finite flute values: {path}")
    if np.nanmin(flutes) < 0 or np.nanmax(flutes) > 2_000:
        raise ValueError(
            f"Implausible wear range after scaling in {path}: "
            f"[{np.nanmin(flutes):.3f}, {np.nanmax(flutes):.3f}] µm. "
            "Check --wear-scale-to-um."
        )

    frame = pd.DataFrame(
        {
            "cutter": cutter,
            "cut_number": cut_number,
            "flute_1_um": flutes[:, 0],
            "flute_2_um": flutes[:, 1],
            "flute_3_um": flutes[:, 2],
        }
    )
    frame["wear_true_um"] = frame[
        ["flute_1_um", "flute_2_um", "flute_3_um"]
    ].mean(axis=1)
    if frame.duplicated("cut_number").any():
        raise ValueError(f"Duplicate cut numbers in wear file: {path}")
    return frame.sort_values("cut_number").reset_index(drop=True)


# =============================================================================
# 3. PASS-LEVEL MULTI-DOMAIN FEATURE EXTRACTION
# =============================================================================


def safe_scalar(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def robust_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    return safe_scalar(np.polyfit(x, y, 1)[0])


def downsample_evenly(values: np.ndarray, max_points: int) -> np.ndarray:
    if len(values) <= max_points:
        return values
    indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return values[indices]


def channel_features(
    signal: np.ndarray,
    prefix: str,
    config: StudyConfig,
) -> dict[str, float]:
    if pywt is None:
        raise ImportError(
            "PyWavelets is required for the declared wavelet features. Install it "
            "with: pip install PyWavelets"
        )
    x = downsample_evenly(np.asarray(signal, dtype=float), config.max_signal_points)
    x = x[np.isfinite(x)]
    if len(x) < 128:
        raise ValueError(f"Insufficient valid samples in channel {prefix}.")

    mean = float(np.mean(x))
    centered = x - mean
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(np.square(x))))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    abs_mean = float(np.mean(abs_x)) + 1.0e-12
    peak = float(np.max(abs_x))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_x + 1.0e-12)))

    feats: dict[str, float] = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_mad": float(np.median(np.abs(x - np.median(x)))),
        f"{prefix}_minimum": float(np.min(x)),
        f"{prefix}_maximum": float(np.max(x)),
        f"{prefix}_ptp": float(np.ptp(x)),
        f"{prefix}_skewness": safe_scalar(stats.skew(x, bias=False)),
        f"{prefix}_kurtosis": safe_scalar(stats.kurtosis(x, fisher=False, bias=False)),
        f"{prefix}_crest_factor": peak / (rms + 1.0e-12),
        f"{prefix}_impulse_factor": peak / abs_mean,
        f"{prefix}_shape_factor": rms / abs_mean,
        f"{prefix}_clearance_factor": peak / (sqrt_abs_mean**2 + 1.0e-12),
        f"{prefix}_energy": float(np.mean(np.square(centered))),
        f"{prefix}_zero_crossing_rate": float(np.mean(centered[:-1] * centered[1:] < 0)),
    }

    nperseg = min(config.welch_nperseg, len(x))
    frequency, power = welch(x, fs=config.sampling_hz, nperseg=nperseg)
    power = np.maximum(power, 0.0)
    total_power = float(np.sum(power)) + 1.0e-18
    probability = power / total_power
    centroid = float(np.sum(frequency * probability))
    bandwidth = float(np.sqrt(np.sum(((frequency - centroid) ** 2) * probability)))
    feats.update(
        {
            f"{prefix}_spectral_centroid_hz": centroid,
            f"{prefix}_spectral_bandwidth_hz": bandwidth,
            f"{prefix}_spectral_entropy": float(
                -np.sum(probability * np.log(probability + 1.0e-18))
                / np.log(max(len(probability), 2))
            ),
            f"{prefix}_dominant_frequency_hz": float(frequency[int(np.argmax(power))]),
            f"{prefix}_spectral_power": total_power,
        }
    )
    nyquist = config.sampling_hz / 2.0
    band_edges = np.linspace(0.0, nyquist, 6)
    for index in range(5):
        if index == 4:
            mask = (frequency >= band_edges[index]) & (frequency <= band_edges[index + 1])
        else:
            mask = (frequency >= band_edges[index]) & (frequency < band_edges[index + 1])
        feats[f"{prefix}_bandpower_{index + 1}"] = float(np.sum(power[mask]) / total_power)

    maximum_level = pywt.dwt_max_level(len(x), pywt.Wavelet(config.wavelet).dec_len)
    level = min(config.wavelet_level, maximum_level)
    if level >= 1:
        coefficients = pywt.wavedec(x, config.wavelet, level=level)
        energies = np.array([np.sum(np.square(c)) for c in coefficients], dtype=float)
        energies /= np.sum(energies) + 1.0e-18
        for index, value in enumerate(energies):
            feats[f"{prefix}_wavelet_energy_{index}"] = float(value)
    for index in range(config.wavelet_level + 1):
        feats.setdefault(f"{prefix}_wavelet_energy_{index}", 0.0)

    # Short-time spectral descriptors quantify within-pass non-stationarity.
    stft_nperseg = min(2_048, len(x))
    stft_frequency, _, stft_values = stft(
        x,
        fs=config.sampling_hz,
        nperseg=stft_nperseg,
        noverlap=stft_nperseg // 2,
        boundary=None,
        padded=False,
    )
    stft_power = np.square(np.abs(stft_values))
    frame_power = np.sum(stft_power, axis=0) + 1.0e-18
    stft_probability = stft_power / frame_power[None, :]
    frame_centroid = np.sum(stft_frequency[:, None] * stft_probability, axis=0)
    high_frequency_mask = stft_frequency >= 0.60 * (config.sampling_hz / 2.0)
    high_frequency_ratio = (
        np.sum(stft_power[high_frequency_mask], axis=0) / frame_power
    )
    feats.update(
        {
            f"{prefix}_stft_energy_mean": float(np.mean(frame_power)),
            f"{prefix}_stft_energy_std": float(np.std(frame_power)),
            f"{prefix}_stft_centroid_mean_hz": float(np.mean(frame_centroid)),
            f"{prefix}_stft_centroid_std_hz": float(np.std(frame_centroid)),
            f"{prefix}_stft_centroid_slope": robust_slope(frame_centroid),
            f"{prefix}_stft_high_frequency_ratio_mean": float(
                np.mean(high_frequency_ratio)
            ),
            f"{prefix}_stft_high_frequency_ratio_std": float(
                np.std(high_frequency_ratio)
            ),
        }
    )

    segments = np.array_split(x, config.local_segments)
    local_rms = np.array([np.sqrt(np.mean(np.square(s))) for s in segments])
    local_std = np.array([np.std(s) for s in segments])
    local_kurtosis = np.array(
        [safe_scalar(stats.kurtosis(s, fisher=False, bias=False)) for s in segments]
    )
    for name, values in (
        ("local_rms", local_rms),
        ("local_std", local_std),
        ("local_kurtosis", local_kurtosis),
    ):
        feats[f"{prefix}_{name}_mean"] = float(np.mean(values))
        feats[f"{prefix}_{name}_std"] = float(np.std(values))
        feats[f"{prefix}_{name}_slope"] = robust_slope(values)

    return {key: safe_scalar(value) for key, value in feats.items()}


def extract_pass_features(
    values: np.ndarray,
    cutter: str,
    cut_number: int,
    config: StudyConfig,
) -> dict[str, Any]:
    if values.shape[1] < len(SENSOR_CHANNELS):
        raise ValueError("Expected seven synchronized sensor columns.")

    record: dict[str, Any] = {"cutter": cutter, "cut_number": int(cut_number)}
    for column, channel in enumerate(SENSOR_CHANNELS):
        record.update(channel_features(values[:, column], channel, config))

    force_rms = np.array([record[f"force_{axis}_rms"] for axis in "xyz"], dtype=float)
    vibration_rms = np.array(
        [record[f"vibration_{axis}_rms"] for axis in "xyz"], dtype=float
    )
    resultant_force = float(np.linalg.norm(force_rms))
    resultant_vibration = float(np.linalg.norm(vibration_rms))
    ae_energy = float(record["ae_rms_energy"])
    sliding_speed_m_s = (
        math.pi * config.cutter_diameter_m * config.spindle_rpm / 60.0
    )

    # F_n is a reproducible force-magnitude proxy from the three measured force
    # channels. The trainable positive wear coefficient absorbs unobserved
    # contact area/material constants. The scale below is normalized later using
    # training cutters only; its absolute unit is not interpreted as K itself.
    physics_exposure = (
        resultant_force
        * sliding_speed_m_s
        / max(config.workpiece_hardness_hrc, 1.0e-12)
    )
    record.update(
        {
            "force_resultant_rms": resultant_force,
            "vibration_resultant_rms": resultant_vibration,
            "force_xy_ratio": record["force_x_rms"] / (record["force_y_rms"] + 1.0e-12),
            "force_z_resultant_ratio": record["force_z_rms"] / (resultant_force + 1.0e-12),
            "vibration_force_ratio": resultant_vibration / (resultant_force + 1.0e-12),
            "ae_energy_index": ae_energy,
            "ae_force_ratio": ae_energy / (resultant_force + 1.0e-12),
            "cutting_power_proxy": resultant_force * sliding_speed_m_s,
            "sliding_speed_m_s": sliding_speed_m_s,
            "physics_exposure": physics_exposure,
            "known_age_pass": float(cut_number),
            "known_age_fraction": float(cut_number) / config.nominal_max_passes,
        }
    )
    if not config.include_known_age:
        record.pop("known_age_pass")
        record.pop("known_age_fraction")
    return record


def extract_or_load_features(
    signal_records: Mapping[str, Sequence[tuple[int, Path]]],
    cache_dir: Path,
    config: StudyConfig,
    rebuild: bool,
) -> dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    for cutter in ALL_CUTTERS:
        cache_path = cache_dir / f"{cutter}_pass_features.csv"
        if cache_path.is_file() and not rebuild:
            frame = pd.read_csv(cache_path)
            frame["cutter"] = frame["cutter"].astype(str).str.lower()
            outputs[cutter] = frame.sort_values("cut_number").reset_index(drop=True)
            print(f"[CACHE] {cutter.upper()}: {cache_path}")
            continue

        rows: list[dict[str, Any]] = []
        total = len(signal_records[cutter])
        print(f"[FEATURES] {cutter.upper()}: extracting {total} machining passes")
        for position, (cut_number, path) in enumerate(signal_records[cutter], start=1):
            values = read_sensor_csv(path)
            rows.append(extract_pass_features(values, cutter, cut_number, config))
            if position == 1 or position % 25 == 0 or position == total:
                print(f"           {position:>3}/{total}  {path.name}")
        frame = pd.DataFrame(rows).sort_values("cut_number").reset_index(drop=True)
        if frame.duplicated(["cutter", "cut_number"]).any():
            raise RuntimeError(f"Duplicate extracted rows for {cutter.upper()}.")
        frame.to_csv(cache_path, index=False)
        outputs[cutter] = frame
        print(f"[SAVED] {cache_path}")

    feature_sets = [set(f.columns) for f in outputs.values()]
    if any(columns != feature_sets[0] for columns in feature_sets[1:]):
        raise RuntimeError("Extracted feature columns differ between cutters.")
    return outputs


# =============================================================================
# 4. FIXED-THRESHOLD LABEL AND RUL CONSTRUCTION
# =============================================================================


def first_crossing_linear(
    cut_number: np.ndarray,
    wear_um: np.ndarray,
    threshold_um: float,
) -> float:
    x = np.asarray(cut_number, dtype=float)
    y = np.asarray(wear_um, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    crossings = np.flatnonzero(y >= threshold_um)
    if len(crossings) == 0:
        return float("nan")
    index = int(crossings[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if np.isclose(y1, y0):
        return x1
    fraction = float(np.clip((threshold_um - y0) / (y1 - y0), 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def attach_targets(
    feature_frames: Mapping[str, pd.DataFrame],
    wear_files: Mapping[str, Path],
    config: StudyConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    frames: dict[str, pd.DataFrame] = {}
    failure_times: dict[str, float] = {}

    for cutter in ALL_CUTTERS:
        features = feature_frames[cutter].copy()
        if cutter in LABELLED_CUTTERS:
            wear = read_wear_csv(
                wear_files[cutter], cutter, config.wear_file_scale_to_um
            )
            feature_passes = set(features["cut_number"].astype(int))
            wear_passes = set(wear["cut_number"].astype(int))
            if wear_passes != feature_passes:
                plus_one = set((wear["cut_number"] + 1).astype(int))
                minus_one = set((wear["cut_number"] - 1).astype(int))
                if plus_one == feature_passes:
                    wear["cut_number"] = wear["cut_number"] + 1
                elif minus_one == feature_passes:
                    wear["cut_number"] = wear["cut_number"] - 1
            merged = features.merge(
                wear,
                how="inner",
                on=["cutter", "cut_number"],
                validate="one_to_one",
            )
            if len(merged) != len(wear):
                missing = sorted(set(wear["cut_number"]) - set(merged["cut_number"]))
                raise RuntimeError(
                    f"{cutter.upper()}: {len(missing)} labelled passes could not be "
                    f"aligned with sensor files. First missing passes: {missing[:20]}"
                )
            failure_time = first_crossing_linear(
                merged["cut_number"].to_numpy(),
                merged["wear_true_um"].to_numpy(),
                config.failure_threshold_um,
            )
            if not np.isfinite(failure_time):
                raise RuntimeError(
                    f"{cutter.upper()} mean wear never reaches "
                    f"{config.failure_threshold_um:.2f} µm; reference RUL is undefined."
                )
            failure_times[cutter] = failure_time
            merged["failure_pass_true"] = failure_time
            merged["rul_true_passes"] = np.maximum(
                failure_time - merged["cut_number"].to_numpy(dtype=float), 0.0
            )
            # Retain the crossing observation but exclude later post-failure cuts.
            merged = merged[
                merged["cut_number"] <= int(math.ceil(failure_time))
            ].copy()
            frames[cutter] = merged.sort_values("cut_number").reset_index(drop=True)
        else:
            frames[cutter] = features.sort_values("cut_number").reset_index(drop=True)

    print("\nFixed-threshold target audit")
    print("-" * 72)
    print(f"Common threshold: {config.failure_threshold_um:.2f} µm")
    print("RUL definition: max(T_fail - current pass, 0), in cutting passes")
    print("Crossing: first crossing with linear interpolation; no smoothing")
    for cutter in LABELLED_CUTTERS:
        print(f"{cutter.upper()}: T_fail = {failure_times[cutter]:.4f} passes")
    print("C2/C3/C5: no wear labels, reference T_fail, or reference RUL")
    print("-" * 72)
    return frames, failure_times


# =============================================================================
# 5. TRAINING-ONLY PREPROCESSING
# =============================================================================


@dataclass
class FittedPreprocessor:
    feature_columns: list[str]
    imputer: SimpleImputer
    variance: VarianceThreshold
    scaler: StandardScaler
    selector: SelectKBest
    selected_feature_names: list[str]
    physics_median: float

    @classmethod
    def fit(
        cls,
        train_frames: Mapping[str, pd.DataFrame],
        max_features: int,
    ) -> "FittedPreprocessor":
        metadata = {
            "cutter",
            "cut_number",
            "flute_1_um",
            "flute_2_um",
            "flute_3_um",
            "wear_true_um",
            "rul_true_passes",
            "failure_pass_true",
        }
        first = next(iter(train_frames.values()))
        feature_columns = [c for c in first.columns if c not in metadata]
        if "physics_exposure" not in feature_columns:
            raise RuntimeError("Required physics_exposure feature is missing.")
        combined = pd.concat(
            [frame[feature_columns] for frame in train_frames.values()],
            ignore_index=True,
        )
        target = pd.concat(
            [frame["wear_true_um"] for frame in train_frames.values()],
            ignore_index=True,
        ).to_numpy(dtype=float)

        imputer = SimpleImputer(strategy="median")
        variance = VarianceThreshold(threshold=1.0e-12)
        scaler = StandardScaler()
        matrix = imputer.fit_transform(combined)
        matrix = variance.fit_transform(matrix)
        matrix = scaler.fit_transform(matrix)

        retained_names = np.asarray(feature_columns)[variance.get_support()].tolist()
        k = min(max_features, matrix.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(matrix, target)
        selected = np.asarray(retained_names)[selector.get_support()].tolist()

        physics_values = combined["physics_exposure"].to_numpy(dtype=float)
        physics_values = physics_values[np.isfinite(physics_values) & (physics_values > 0)]
        physics_median = float(np.median(physics_values)) if len(physics_values) else 1.0
        physics_median = max(physics_median, 1.0e-12)

        return cls(
            feature_columns=feature_columns,
            imputer=imputer,
            variance=variance,
            scaler=scaler,
            selector=selector,
            selected_feature_names=selected,
            physics_median=physics_median,
        )

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.imputer.transform(frame[self.feature_columns])
        matrix = self.variance.transform(matrix)
        matrix = self.scaler.transform(matrix)
        matrix = self.selector.transform(matrix).astype(np.float32)
        physics = (
            frame["physics_exposure"].to_numpy(dtype=np.float32) / self.physics_median
        )
        physics = np.clip(physics, 0.0, 20.0).reshape(-1, 1).astype(np.float32)
        if not np.isfinite(matrix).all() or not np.isfinite(physics).all():
            raise RuntimeError("Non-finite values remain after preprocessing.")
        return matrix, physics


@dataclass
class Trajectory:
    cutter: str
    cut_number: np.ndarray
    features: np.ndarray
    physics: np.ndarray
    wear: np.ndarray | None
    rul: np.ndarray | None
    flutes: np.ndarray | None


def transform_trajectories(
    frames: Mapping[str, pd.DataFrame],
    preprocessor: FittedPreprocessor,
    threshold_um: float,
    nominal_max_passes: float,
) -> dict[str, Trajectory]:
    output: dict[str, Trajectory] = {}
    for cutter, frame in frames.items():
        features, physics = preprocessor.transform(frame)
        labelled = "wear_true_um" in frame.columns
        output[cutter] = Trajectory(
            cutter=cutter,
            cut_number=frame["cut_number"].to_numpy(dtype=np.float32),
            features=features,
            physics=physics,
            wear=(
                frame["wear_true_um"].to_numpy(dtype=np.float32) / threshold_um
                if labelled
                else None
            ),
            rul=(
                frame["rul_true_passes"].to_numpy(dtype=np.float32)
                / nominal_max_passes
                if labelled
                else None
            ),
            flutes=(
                frame[["flute_1_um", "flute_2_um", "flute_3_um"]].to_numpy(
                    dtype=np.float32
                )
                / threshold_um
                if labelled
                else None
            ),
        )
    return output


# =============================================================================
# 6. FULL-TRAJECTORY DATASET (STRICTLY CAUSAL MODEL OUTPUTS)
# =============================================================================


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: Mapping[str, Trajectory], labelled: bool):
        self.items = [trajectories[c] for c in sorted(trajectories)]
        self.labelled = labelled
        if not self.items:
            raise ValueError("TrajectoryDataset received no trajectories.")
        for item in self.items:
            has_targets = item.wear is not None and item.rul is not None and item.flutes is not None
            if labelled != has_targets:
                raise ValueError(f"Label status mismatch for {item.cutter.upper()}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Trajectory:
        return self.items[index]


def collate_trajectories(batch: Sequence[Trajectory]) -> dict[str, Any]:
    lengths = torch.tensor([len(item.cut_number) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    feature_dim = batch[0].features.shape[1]
    size = len(batch)

    features = torch.zeros(size, maximum, feature_dim, dtype=torch.float32)
    physics = torch.zeros(size, maximum, 1, dtype=torch.float32)
    cut_number = torch.zeros(size, maximum, dtype=torch.float32)
    mask = torch.zeros(size, maximum, dtype=torch.bool)
    wear = torch.zeros(size, maximum, dtype=torch.float32)
    rul = torch.zeros(size, maximum, dtype=torch.float32)
    flutes = torch.zeros(size, maximum, 3, dtype=torch.float32)
    labelled = batch[0].wear is not None

    for row, item in enumerate(batch):
        length = len(item.cut_number)
        features[row, :length] = torch.from_numpy(item.features)
        physics[row, :length] = torch.from_numpy(item.physics)
        cut_number[row, :length] = torch.from_numpy(item.cut_number)
        mask[row, :length] = True
        if labelled:
            wear[row, :length] = torch.from_numpy(item.wear)
            rul[row, :length] = torch.from_numpy(item.rul)
            flutes[row, :length] = torch.from_numpy(item.flutes)

    return {
        "cutter": [item.cutter for item in batch],
        "features": features,
        "physics": physics,
        "cut_number": cut_number,
        "lengths": lengths,
        "mask": mask,
        "wear": wear if labelled else None,
        "rul": rul if labelled else None,
        "flutes": flutes if labelled else None,
    }


# =============================================================================
# 7. CAUSAL PHYSICS-INFORMED NEURAL DIFFERENTIAL MODEL
# =============================================================================


class CausalConv1d(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]. Only left padding is applied.
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x).transpose(1, 2)


class PINDPNet(nn.Module):
    """Causal UniGRU + positive physics branch + positive neural residual."""

    def __init__(
        self,
        input_dim: int,
        hp: HyperParameters,
        rul_normalization_passes: float = 315.0,
    ):
        super().__init__()
        self.hp = hp
        self.rul_normalization_passes = float(rul_normalization_passes)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hp.hidden_dim),
            nn.LayerNorm(hp.hidden_dim),
            nn.SiLU(),
            nn.Dropout(hp.dropout),
        )
        self.causal_branches = nn.ModuleList(
            [CausalConv1d(hp.hidden_dim, hp.conv_dim, kernel) for kernel in (3, 5, 9)]
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hp.conv_dim, hp.hidden_dim),
            nn.LayerNorm(hp.hidden_dim),
            nn.SiLU(),
            nn.Dropout(hp.dropout),
        )
        self.gru = nn.GRU(
            input_size=hp.hidden_dim,
            hidden_size=hp.hidden_dim,
            num_layers=hp.gru_layers,
            batch_first=True,
            dropout=hp.dropout if hp.gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.initial_wear = nn.Sequential(
            nn.Linear(hp.hidden_dim, hp.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hp.hidden_dim // 2, 1),
        )
        self.phase_head = nn.Linear(hp.hidden_dim + 1, 3)
        self.phase_raw_scales = nn.Parameter(torch.tensor([0.5, 0.0, 1.0]))
        self.raw_wear_coefficient = nn.Parameter(torch.tensor(-2.0))
        self.neural_residual = nn.Sequential(
            nn.Linear(hp.hidden_dim + 2, hp.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.hidden_dim // 2, 1),
        )

        decoder_dim = hp.hidden_dim + 3
        self.rul_mean_head = nn.Sequential(
            nn.Linear(decoder_dim, hp.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.hidden_dim // 2, 1),
        )
        self.rul_sigma_head = nn.Sequential(
            nn.Linear(decoder_dim, hp.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hp.hidden_dim // 2, 1),
        )
        self.wear_sigma_head = nn.Sequential(
            nn.Linear(hp.hidden_dim + 1, hp.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hp.hidden_dim // 2, 1),
        )
        self.flute_offsets = nn.Sequential(
            nn.Linear(hp.hidden_dim + 1, hp.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hp.hidden_dim // 2, 3),
        )

        # Stable physical starting scales. These are initial values only and
        # remain fully trainable.
        nn.init.normal_(self.neural_residual[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.neural_residual[-1].bias, -3.0)
        nn.init.constant_(self.wear_sigma_head[-1].bias, -2.5)
        nn.init.constant_(self.rul_sigma_head[-1].bias, -2.5)

    def _rate(
        self,
        wear: torch.Tensor,
        context: torch.Tensor,
        physics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_probability = torch.softmax(
            self.phase_head(torch.cat([context, wear], dim=-1)), dim=-1
        )
        phase_scales = F.softplus(self.phase_raw_scales) + 0.10
        phase_multiplier = torch.sum(phase_probability * phase_scales, dim=-1, keepdim=True)
        wear_coefficient = F.softplus(self.raw_wear_coefficient) + 1.0e-5
        physics_rate = (
            self.hp.physics_scale
            * wear_coefficient
            * torch.clamp(physics, min=0.0)
            * phase_multiplier
        )
        residual_input = torch.cat([context, wear, physics], dim=-1)
        residual_rate = self.hp.neural_rate_scale * F.softplus(
            self.neural_residual(residual_input)
        )
        total_rate = torch.clamp(physics_rate + residual_rate, min=1.0e-7, max=0.25)
        return total_rate, physics_rate, residual_rate, phase_probability

    def forward(
        self,
        features: torch.Tensor,
        physics: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(features)
        multiscale = torch.cat([branch(encoded) for branch in self.causal_branches], dim=-1)
        fused = self.fusion(multiscale)
        packed = pack_padded_sequence(
            fused, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_context, _ = self.gru(packed)
        context, _ = pad_packed_sequence(
            packed_context, batch_first=True, total_length=features.shape[1]
        )

        batch_size, maximum_time, _ = context.shape
        wear_states: list[torch.Tensor] = []
        rates: list[torch.Tensor] = []
        physics_rates: list[torch.Tensor] = []
        residual_rates: list[torch.Tensor] = []
        phases: list[torch.Tensor] = []

        wear = 0.35 * torch.sigmoid(self.initial_wear(context[:, 0]))
        for time_index in range(maximum_time):
            current_context = context[:, time_index]
            current_physics = physics[:, time_index]
            if time_index > 0:
                # RK4 integration over one cutting-pass interval. Context and
                # physics are held piecewise constant within the interval.
                k1, _, _, _ = self._rate(wear, current_context, current_physics)
                k2, _, _, _ = self._rate(
                    wear + 0.5 * k1, current_context, current_physics
                )
                k3, _, _, _ = self._rate(
                    wear + 0.5 * k2, current_context, current_physics
                )
                k4, _, _, _ = self._rate(wear + k3, current_context, current_physics)
                candidate = wear + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                active = (lengths > time_index).to(wear.dtype).unsqueeze(-1)
                wear = active * candidate + (1.0 - active) * wear

            rate, physics_rate, residual_rate, phase_probability = self._rate(
                wear, current_context, current_physics
            )
            wear_states.append(wear)
            rates.append(rate)
            physics_rates.append(physics_rate)
            residual_rates.append(residual_rate)
            phases.append(phase_probability)

        wear_mean = torch.stack(wear_states, dim=1).squeeze(-1)
        rate = torch.stack(rates, dim=1).squeeze(-1)
        physics_rate = torch.stack(physics_rates, dim=1).squeeze(-1)
        residual_rate = torch.stack(residual_rates, dim=1).squeeze(-1)
        phase_probability = torch.stack(phases, dim=1)

        wear_context = torch.cat([context, wear_mean.unsqueeze(-1)], dim=-1)
        wear_sigma = 0.003 + F.softplus(self.wear_sigma_head(wear_context)).squeeze(-1)
        offsets = 0.25 * torch.tanh(self.flute_offsets(wear_context))
        flute_mean = torch.clamp(wear_mean.unsqueeze(-1) + offsets, min=0.0)

        decoder = torch.cat(
            [context, wear_mean.unsqueeze(-1), rate.unsqueeze(-1), physics], dim=-1
        )
        rul_mean = 1.30 * torch.sigmoid(self.rul_mean_head(decoder)).squeeze(-1)
        rul_sigma = 0.005 + F.softplus(self.rul_sigma_head(decoder)).squeeze(-1)

        return {
            "wear_mean": wear_mean,
            "wear_sigma": wear_sigma,
            "rul_mean": rul_mean,
            "rul_sigma": rul_sigma,
            "flute_mean": flute_mean,
            "rate": rate,
            "physics_rate": physics_rate,
            "residual_rate": residual_rate,
            "phase_probability": phase_probability,
            "rul_normalization_passes": wear_mean.new_tensor(
                self.rul_normalization_passes
            ),
        }


# =============================================================================
# 8. COMPOSITE OBJECTIVE AND OPTIMIZATION
# =============================================================================


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1.0)


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=1.0e-4, max=2.0)
    loss = torch.log(sigma) + 0.5 * torch.square((target - mean) / sigma)
    return masked_mean(loss, mask)


def composite_loss(
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    hp: HyperParameters,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    wear_target = batch["wear"]
    rul_target = batch["rul"]
    flute_target = batch["flutes"]
    assert wear_target is not None and rul_target is not None and flute_target is not None

    wear_huber = masked_mean(
        F.smooth_l1_loss(prediction["wear_mean"], wear_target, reduction="none"), mask
    )
    wear_nll = gaussian_nll(
        wear_target, prediction["wear_mean"], prediction["wear_sigma"], mask
    )
    rul_huber = masked_mean(
        F.smooth_l1_loss(prediction["rul_mean"], rul_target, reduction="none"), mask
    )
    rul_nll = gaussian_nll(
        rul_target, prediction["rul_mean"], prediction["rul_sigma"], mask
    )
    flute_huber = masked_mean(
        F.smooth_l1_loss(
            prediction["flute_mean"], flute_target, reduction="none"
        ).mean(dim=-1),
        mask,
    )

    threshold_horizon = torch.clamp(
        (1.0 - prediction["wear_mean"])
        / (
            prediction["rate"] * prediction["rul_normalization_passes"]
            + 1.0e-5
        ),
        min=0.0,
        max=1.30,
    )
    threshold_consistency = masked_mean(
        F.smooth_l1_loss(
            prediction["rul_mean"], threshold_horizon.detach(), reduction="none"
        ),
        mask,
    )

    pair_mask = mask[:, 1:] & mask[:, :-1]
    if bool(pair_mask.any().item()):
        expected_normalized_decrement = 1.0 / prediction["rul_normalization_passes"]
        rul_slope_consistency = masked_mean(
            torch.square(
                prediction["rul_mean"][:, 1:]
                - prediction["rul_mean"][:, :-1]
                + expected_normalized_decrement
            ),
            pair_mask,
        )
        rate_smoothness = masked_mean(
            torch.square(prediction["rate"][:, 1:] - prediction["rate"][:, :-1]),
            pair_mask,
        )
    else:
        rul_slope_consistency = prediction["rul_mean"].sum() * 0.0
        rate_smoothness = prediction["rate"].sum() * 0.0

    residual_fraction = prediction["residual_rate"] / (
        prediction["rate"] + 1.0e-6
    )
    residual_regularization = masked_mean(torch.square(residual_fraction), mask)

    total = (
        hp.wear_huber_weight * wear_huber
        + hp.wear_nll_weight * wear_nll
        + hp.rul_huber_weight * rul_huber
        + hp.rul_nll_weight * rul_nll
        + hp.flute_weight * flute_huber
        + hp.threshold_consistency_weight * threshold_consistency
        + hp.rul_slope_weight * rul_slope_consistency
        + hp.rate_smoothness_weight * rate_smoothness
        + hp.residual_weight * residual_regularization
    )
    components = {
        "total": float(total.detach().cpu()),
        "wear_huber": float(wear_huber.detach().cpu()),
        "wear_nll": float(wear_nll.detach().cpu()),
        "rul_huber": float(rul_huber.detach().cpu()),
        "rul_nll": float(rul_nll.detach().cpu()),
        "flute_huber": float(flute_huber.detach().cpu()),
        "threshold_consistency": float(threshold_consistency.detach().cpu()),
        "rul_slope_consistency": float(rul_slope_consistency.detach().cpu()),
        "rate_smoothness": float(rate_smoothness.detach().cpu()),
        "residual_regularization": float(residual_regularization.detach().cpu()),
    }
    return total, components


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "features",
        "physics",
        "cut_number",
        "lengths",
        "mask",
        "wear",
        "rul",
        "flutes",
    ):
        if isinstance(moved.get(key), torch.Tensor):
            moved[key] = moved[key].to(device)
    return moved


@torch.no_grad()
def validation_objective(
    model: PINDPNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    wear_true, wear_pred, rul_true, rul_pred = [], [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch["features"], batch["physics"], batch["lengths"])
        mask = batch["mask"]
        wear_true.extend(batch["wear"][mask].detach().cpu().numpy())
        wear_pred.extend(output["wear_mean"][mask].detach().cpu().numpy())
        rul_true.extend(batch["rul"][mask].detach().cpu().numpy())
        rul_pred.extend(output["rul_mean"][mask].detach().cpu().numpy())
    wear_rmse = float(np.sqrt(mean_squared_error(wear_true, wear_pred)))
    rul_rmse = float(np.sqrt(mean_squared_error(rul_true, rul_pred)))
    score = wear_rmse + rul_rmse
    return score, {"wear_rmse_normalized": wear_rmse, "rul_rmse_normalized": rul_rmse}


def train_with_early_stopping(
    model: PINDPNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    hp: HyperParameters,
    config: StudyConfig,
    device: torch.device,
    seed: int,
) -> tuple[PINDPNet, int, list[dict[str, float]]]:
    set_deterministic(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(10, config.patience // 4), min_lr=1e-6
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        training_losses = []
        for _ in range(config.steps_per_epoch):
            for raw_batch in train_loader:
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["features"], batch["physics"], batch["lengths"])
                loss, _ = composite_loss(output, batch, hp)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss became non-finite.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                training_losses.append(float(loss.detach().cpu()))

        score, details = validation_objective(model, validation_loader, device)
        scheduler.step(score)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(training_losses)),
            "validation_score": score,
            **details,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        if score < best_score - 1.0e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"      epoch {epoch:>3}: train={row['train_loss']:.5f}, "
                f"val={score:.5f}, wear={details['wear_rmse_normalized']:.5f}, "
                f"RUL={details['rul_rmse_normalized']:.5f}"
            )
        if (
            epoch >= config.min_epochs
            and epochs_without_improvement >= config.patience
        ):
            break

    model.load_state_dict(best_state)
    return model, best_epoch, history


def train_fixed_epochs(
    model: PINDPNet,
    train_loader: DataLoader,
    hp: HyperParameters,
    config: StudyConfig,
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[PINDPNet, list[dict[str, float]]]:
    set_deterministic(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1.0e-6
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(config.steps_per_epoch):
            for raw_batch in train_loader:
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["features"], batch["physics"], batch["lengths"])
                loss, components = composite_loss(output, batch, hp)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss became non-finite.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            print(f"      epoch {epoch:>3}/{epochs}: loss={np.mean(losses):.5f}")
    return model, history


# =============================================================================
# 9. PREDICTION, DEEP-ENSEMBLE UNCERTAINTY AND CALIBRATION
# =============================================================================


@torch.no_grad()
def predict_single_model(
    model: PINDPNet,
    trajectories: Mapping[str, Trajectory],
    device: torch.device,
    config: StudyConfig,
) -> pd.DataFrame:
    labelled = all(item.wear is not None for item in trajectories.values())
    loader = DataLoader(
        TrajectoryDataset(trajectories, labelled=labelled),
        batch_size=len(trajectories),
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_trajectories,
    )
    model.eval()
    rows: list[pd.DataFrame] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch["features"], batch["physics"], batch["lengths"])
        for row_index, cutter in enumerate(raw_batch["cutter"]):
            length = int(raw_batch["lengths"][row_index])
            frame = pd.DataFrame(
                {
                    "cutter": cutter,
                    "cut_number": raw_batch["cut_number"][row_index, :length].numpy(),
                    "wear_mean_norm": output["wear_mean"][row_index, :length].cpu().numpy(),
                    "wear_sigma_norm": output["wear_sigma"][row_index, :length].cpu().numpy(),
                    "rul_mean_norm": output["rul_mean"][row_index, :length].cpu().numpy(),
                    "rul_sigma_norm": output["rul_sigma"][row_index, :length].cpu().numpy(),
                    "rate_norm_per_pass": output["rate"][row_index, :length].cpu().numpy(),
                    "physics_rate_norm_per_pass": output["physics_rate"][row_index, :length].cpu().numpy(),
                    "residual_rate_norm_per_pass": output["residual_rate"][row_index, :length].cpu().numpy(),
                }
            )
            phase = output["phase_probability"][row_index, :length].cpu().numpy()
            flutes = output["flute_mean"][row_index, :length].cpu().numpy()
            for phase_index, name in enumerate(("break_in", "steady", "accelerated")):
                frame[f"phase_probability_{name}"] = phase[:, phase_index]
            for flute_index in range(3):
                frame[f"flute_{flute_index + 1}_mean_norm"] = flutes[:, flute_index]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def combine_ensemble_predictions(
    predictions: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    if not predictions:
        raise ValueError("No ensemble predictions were supplied.")
    key_columns = ["cutter", "cut_number"]
    reference_keys = predictions[0][key_columns].reset_index(drop=True)
    for index, frame in enumerate(predictions[1:], start=2):
        if not reference_keys.equals(frame[key_columns].reset_index(drop=True)):
            raise RuntimeError(f"Prediction key mismatch in ensemble member {index}.")

    combined = reference_keys.copy()
    mean_columns = [
        "wear_mean_norm",
        "rul_mean_norm",
        "rate_norm_per_pass",
        "physics_rate_norm_per_pass",
        "residual_rate_norm_per_pass",
        "phase_probability_break_in",
        "phase_probability_steady",
        "phase_probability_accelerated",
        "flute_1_mean_norm",
        "flute_2_mean_norm",
        "flute_3_mean_norm",
    ]
    for column in mean_columns:
        stack = np.stack([frame[column].to_numpy(dtype=float) for frame in predictions])
        combined[column] = np.mean(stack, axis=0)

    for mean_column, sigma_column in (
        ("wear_mean_norm", "wear_sigma_norm"),
        ("rul_mean_norm", "rul_sigma_norm"),
    ):
        means = np.stack([frame[mean_column].to_numpy(dtype=float) for frame in predictions])
        sigmas = np.stack([frame[sigma_column].to_numpy(dtype=float) for frame in predictions])
        second_moment = np.mean(np.square(sigmas) + np.square(means), axis=0)
        variance = np.maximum(second_moment - np.square(np.mean(means, axis=0)), 1.0e-10)
        combined[sigma_column] = np.sqrt(variance)
    return combined


def attach_truth_to_predictions(
    predictions: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    truth_columns = [
        "cutter",
        "cut_number",
        "wear_true_um",
        "rul_true_passes",
        "failure_pass_true",
        "flute_1_um",
        "flute_2_um",
        "flute_3_um",
    ]
    truth = pd.concat([f[truth_columns] for f in frames.values()], ignore_index=True)
    return predictions.merge(
        truth, on=["cutter", "cut_number"], how="left", validate="one_to_one"
    )


def dimensionalize_predictions(
    frame: pd.DataFrame,
    config: StudyConfig,
    wear_scale: float,
    rul_scale: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["wear_pred_um"] = output["wear_mean_norm"] * config.failure_threshold_um
    output["wear_std_um"] = (
        output["wear_sigma_norm"] * config.failure_threshold_um * wear_scale
    )
    output["rul_pred_passes"] = output["rul_mean_norm"] * config.nominal_max_passes
    output["rul_std_passes"] = (
        output["rul_sigma_norm"] * config.nominal_max_passes * rul_scale
    )
    output["wear_rate_um_per_pass"] = (
        output["rate_norm_per_pass"] * config.failure_threshold_um
    )
    for flute_index in range(1, 4):
        output[f"flute_{flute_index}_pred_um"] = (
            output[f"flute_{flute_index}_mean_norm"] * config.failure_threshold_um
        )
    return output


def calibration_multiplier(
    target: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    z_value: float,
) -> float:
    target = np.asarray(target, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1.0e-6)
    ratio = np.abs(target - mean) / sigma
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) < 10:
        return 1.0
    scale = float(np.quantile(ratio, 0.95) / z_value)
    return float(np.clip(scale, 0.50, 5.00))


# =============================================================================
# 10. NESTED CUTTER-LEVEL MODEL SELECTION
# =============================================================================


def make_loader(
    trajectories: Mapping[str, Trajectory],
    config: StudyConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TrajectoryDataset(trajectories, labelled=True),
        batch_size=len(trajectories),
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_trajectories,
    )


def inner_model_selection(
    development_frames: Mapping[str, pd.DataFrame],
    candidates: Sequence[HyperParameters],
    config: StudyConfig,
    device: torch.device,
    audit_dir: Path,
    fold_name: str,
) -> tuple[HyperParameters, int, float, float, dict[str, Any]]:
    cutter_names = sorted(development_frames)
    if len(cutter_names) < 2:
        raise ValueError("Inner selection requires at least two development cutters.")

    candidate_audit: dict[str, Any] = {}
    best_hp: HyperParameters | None = None
    best_score = float("inf")
    best_epochs: list[int] = []
    best_predictions: list[pd.DataFrame] = []

    for candidate_index, hp in enumerate(candidates):
        print(f"    candidate: {hp.name}")
        fold_scores: list[float] = []
        fold_epochs: list[int] = []
        fold_predictions: list[pd.DataFrame] = []
        fold_details: list[dict[str, Any]] = []

        for holdout_index, holdout in enumerate(cutter_names):
            inner_train = {
                name: development_frames[name]
                for name in cutter_names
                if name != holdout
            }
            inner_validation = {holdout: development_frames[holdout]}
            assert set(inner_train).isdisjoint(inner_validation)
            print(
                f"      inner holdout {holdout.upper()}; train="
                f"{','.join(c.upper() for c in inner_train)}"
            )

            preprocessor = FittedPreprocessor.fit(
                inner_train, config.max_selected_features
            )
            train_trajectories = transform_trajectories(
                inner_train,
                preprocessor,
                config.failure_threshold_um,
                config.nominal_max_passes,
            )
            validation_trajectories = transform_trajectories(
                inner_validation,
                preprocessor,
                config.failure_threshold_um,
                config.nominal_max_passes,
            )
            train_loader = make_loader(train_trajectories, config, shuffle=True)
            validation_loader = make_loader(
                validation_trajectories, config, shuffle=False
            )
            seed = (
                config.seed
                + 10_000 * candidate_index
                + 1_000 * holdout_index
                + sum(ord(c) for c in fold_name)
            )
            set_deterministic(seed)
            model = PINDPNet(
                input_dim=train_trajectories[next(iter(train_trajectories))].features.shape[1],
                hp=hp,
                rul_normalization_passes=config.nominal_max_passes,
            )
            model, best_epoch, history = train_with_early_stopping(
                model,
                train_loader,
                validation_loader,
                hp,
                config,
                device,
                seed,
            )
            score, details = validation_objective(model, validation_loader, device)
            raw_prediction = predict_single_model(
                model, validation_trajectories, device, config
            )
            raw_prediction = attach_truth_to_predictions(
                raw_prediction, inner_validation
            )
            fold_predictions.append(raw_prediction)
            fold_scores.append(score)
            fold_epochs.append(best_epoch)
            fold_details.append(
                {
                    "holdout": holdout,
                    "training_cutters": sorted(inner_train),
                    "score": score,
                    "best_epoch": best_epoch,
                    **details,
                }
            )
            history_path = audit_dir / fold_name / hp.name / f"inner_{holdout}_history.csv"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(history_path, index=False)

        mean_score = float(np.mean(fold_scores))
        candidate_audit[hp.name] = {
            "mean_normalized_rmse_sum": mean_score,
            "folds": fold_details,
        }
        print(f"      mean selection score: {mean_score:.6f}")
        if mean_score < best_score:
            best_score = mean_score
            best_hp = hp
            best_epochs = fold_epochs
            best_predictions = fold_predictions

    if best_hp is None:
        raise RuntimeError("No hyperparameter candidate was selected.")

    calibration_frame = pd.concat(best_predictions, ignore_index=True)
    wear_scale = calibration_multiplier(
        calibration_frame["wear_true_um"].to_numpy() / config.failure_threshold_um,
        calibration_frame["wear_mean_norm"].to_numpy(),
        calibration_frame["wear_sigma_norm"].to_numpy(),
        config.uncertainty_z,
    )
    rul_scale = calibration_multiplier(
        calibration_frame["rul_true_passes"].to_numpy() / config.nominal_max_passes,
        calibration_frame["rul_mean_norm"].to_numpy(),
        calibration_frame["rul_sigma_norm"].to_numpy(),
        config.uncertainty_z,
    )
    chosen_epochs = max(
        config.min_epochs,
        int(round(float(np.median(best_epochs)) * 1.10)),
    )
    chosen_epochs = min(chosen_epochs, config.max_epochs)
    audit = {
        "fold": fold_name,
        "development_cutters": cutter_names,
        "selected_hyperparameters": asdict(best_hp),
        "selected_epochs": chosen_epochs,
        "wear_uncertainty_multiplier": wear_scale,
        "rul_uncertainty_multiplier": rul_scale,
        "candidate_results": candidate_audit,
    }
    write_json(audit_dir / fold_name / "inner_selection.json", audit)
    return best_hp, chosen_epochs, wear_scale, rul_scale, audit


def train_outer_ensemble(
    train_frames: Mapping[str, pd.DataFrame],
    evaluation_frames: Mapping[str, pd.DataFrame],
    hp: HyperParameters,
    epochs: int,
    wear_scale: float,
    rul_scale: float,
    config: StudyConfig,
    device: torch.device,
    fold_dir: Path,
    fold_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert set(train_frames).isdisjoint(evaluation_frames)
    fold_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = FittedPreprocessor.fit(train_frames, config.max_selected_features)
    joblib.dump(
        {
            "feature_columns": preprocessor.feature_columns,
            "imputer": preprocessor.imputer,
            "variance_filter": preprocessor.variance,
            "standard_scaler": preprocessor.scaler,
            "feature_selector": preprocessor.selector,
            "selected_feature_names": preprocessor.selected_feature_names,
            "physics_training_median": preprocessor.physics_median,
        },
        fold_dir / "preprocessor.joblib",
    )
    write_json(
        fold_dir / "selected_features.json",
        {
            "training_cutters": sorted(train_frames),
            "evaluation_cutters": sorted(evaluation_frames),
            "selected_features": preprocessor.selected_feature_names,
            "physics_training_median": preprocessor.physics_median,
        },
    )

    train_trajectories = transform_trajectories(
        train_frames,
        preprocessor,
        config.failure_threshold_um,
        config.nominal_max_passes,
    )
    evaluation_trajectories = transform_trajectories(
        evaluation_frames,
        preprocessor,
        config.failure_threshold_um,
        config.nominal_max_passes,
    )
    train_loader = make_loader(train_trajectories, config, shuffle=True)
    input_dim = train_trajectories[next(iter(train_trajectories))].features.shape[1]
    member_predictions: list[pd.DataFrame] = []
    member_seeds: list[int] = []
    learned_physics_parameters: list[dict[str, Any]] = []

    for member in range(config.ensemble_members):
        seed = fold_seed + member * 997
        member_seeds.append(seed)
        print(f"    ensemble member {member + 1}/{config.ensemble_members}; seed={seed}")
        set_deterministic(seed)
        model = PINDPNet(
            input_dim=input_dim,
            hp=hp,
            rul_normalization_passes=config.nominal_max_passes,
        )
        model, history = train_fixed_epochs(
            model,
            train_loader,
            hp,
            config,
            device,
            seed,
            epochs,
        )
        learned_physics_parameters.append(
            {
                "member": member + 1,
                "effective_positive_wear_coefficient": float(
                    (F.softplus(model.raw_wear_coefficient) + 1.0e-5)
                    .detach()
                    .cpu()
                ),
                "positive_phase_multipliers": (
                    F.softplus(model.phase_raw_scales) + 0.10
                )
                .detach()
                .cpu()
                .numpy()
                .tolist(),
            }
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hyperparameters": asdict(hp),
            "study_config": asdict(config),
            "training_cutters": sorted(train_frames),
            "evaluation_cutters": sorted(evaluation_frames),
            "seed": seed,
            "epochs": epochs,
        }
        torch.save(checkpoint, fold_dir / f"model_member_{member + 1}.pt")
        pd.DataFrame(history).to_csv(
            fold_dir / f"model_member_{member + 1}_history.csv", index=False
        )
        member_predictions.append(
            predict_single_model(model, evaluation_trajectories, device, config)
        )

    combined = combine_ensemble_predictions(member_predictions)
    combined = dimensionalize_predictions(
        combined, config, wear_scale=wear_scale, rul_scale=rul_scale
    )
    if all("wear_true_um" in frame.columns for frame in evaluation_frames.values()):
        combined = attach_truth_to_predictions(combined, evaluation_frames)

    audit = {
        "training_cutters": sorted(train_frames),
        "evaluation_cutters": sorted(evaluation_frames),
        "hyperparameters": asdict(hp),
        "epochs": epochs,
        "ensemble_member_seeds": member_seeds,
        "learned_physics_parameters": learned_physics_parameters,
        "wear_uncertainty_multiplier": wear_scale,
        "rul_uncertainty_multiplier": rul_scale,
        "causal": True,
        "bidirectional": False,
        "normal_force_source": "RMS resultant of measured Fx, Fy and Fz",
        "sliding_velocity_source": "pi * cutter_diameter * spindle_rpm / 60",
        "physics_constraint": "hard non-negative wear rate by positive parameterization",
    }
    write_json(fold_dir / "training_audit.json", audit)
    return combined, audit


# =============================================================================
# 11. METRICS AND REPRODUCIBILITY AUDITS
# =============================================================================


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    if len(target) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return {
        "n": int(len(target)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "mae": float(mean_absolute_error(target, prediction)),
        "r2": float(r2_score(target, prediction)) if len(target) >= 2 else float("nan"),
    }


def uncertainty_metrics(
    target: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    z_value: float,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(target) & np.isfinite(mean) & np.isfinite(sigma) & (sigma > 0)
    target, mean, sigma = target[valid], mean[valid], sigma[valid]
    if len(target) == 0:
        return {"n": 0, "picp": float("nan"), "mpiw": float("nan"), "nll": float("nan")}
    lower = mean - z_value * sigma
    upper = mean + z_value * sigma
    nll = np.mean(np.log(sigma) + 0.5 * np.square((target - mean) / sigma))
    return {
        "n": int(len(target)),
        "nominal_coverage": 0.95,
        "picp": float(np.mean((target >= lower) & (target <= upper))),
        "mpiw": float(np.mean(upper - lower)),
        "gaussian_nll_without_constant": float(nll),
    }


def compute_oof_metrics(
    oof: pd.DataFrame,
    config: StudyConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"per_cutter": {}, "pooled": {}}
    for cutter in LABELLED_CUTTERS:
        subset = oof[oof["cutter"] == cutter]
        wear = regression_metrics(subset["wear_true_um"], subset["wear_pred_um"])
        rul = regression_metrics(subset["rul_true_passes"], subset["rul_pred_passes"])
        wear_u = uncertainty_metrics(
            subset["wear_true_um"], subset["wear_pred_um"], subset["wear_std_um"], config.uncertainty_z
        )
        rul_u = uncertainty_metrics(
            subset["rul_true_passes"], subset["rul_pred_passes"], subset["rul_std_passes"], config.uncertainty_z
        )
        detail["per_cutter"][cutter] = {
            "wear": wear,
            "rul": rul,
            "wear_uncertainty": wear_u,
            "rul_uncertainty": rul_u,
        }
        rows.append(
            {
                "scope": cutter.upper(),
                "wear_rmse_um": wear["rmse"],
                "wear_mae_um": wear["mae"],
                "wear_r2": wear["r2"],
                "rul_rmse_passes": rul["rmse"],
                "rul_mae_passes": rul["mae"],
                "rul_r2": rul["r2"],
                "wear_picp_95": wear_u["picp"],
                "wear_mpiw_um": wear_u["mpiw"],
                "rul_picp_95": rul_u["picp"],
                "rul_mpiw_passes": rul_u["mpiw"],
            }
        )

    pooled_wear = regression_metrics(oof["wear_true_um"], oof["wear_pred_um"])
    pooled_rul = regression_metrics(oof["rul_true_passes"], oof["rul_pred_passes"])
    pooled_wear_u = uncertainty_metrics(
        oof["wear_true_um"], oof["wear_pred_um"], oof["wear_std_um"], config.uncertainty_z
    )
    pooled_rul_u = uncertainty_metrics(
        oof["rul_true_passes"], oof["rul_pred_passes"], oof["rul_std_passes"], config.uncertainty_z
    )
    detail["pooled"] = {
        "wear": pooled_wear,
        "rul": pooled_rul,
        "wear_uncertainty": pooled_wear_u,
        "rul_uncertainty": pooled_rul_u,
    }
    rows.append(
        {
            "scope": "POOLED_OOF",
            "wear_rmse_um": pooled_wear["rmse"],
            "wear_mae_um": pooled_wear["mae"],
            "wear_r2": pooled_wear["r2"],
            "rul_rmse_passes": pooled_rul["rmse"],
            "rul_mae_passes": pooled_rul["mae"],
            "rul_r2": pooled_rul["r2"],
            "wear_picp_95": pooled_wear_u["picp"],
            "wear_mpiw_um": pooled_wear_u["mpiw"],
            "rul_picp_95": pooled_rul_u["picp"],
            "rul_mpiw_passes": pooled_rul_u["mpiw"],
        }
    )
    return pd.DataFrame(rows), detail


def assert_causal_and_disjoint(audits: Sequence[Mapping[str, Any]]) -> None:
    for audit in audits:
        train = set(audit["training_cutters"])
        evaluate = set(audit["evaluation_cutters"])
        if train & evaluate:
            raise AssertionError(f"Cutter leakage detected: {sorted(train & evaluate)}")
        if not audit.get("causal", False) or audit.get("bidirectional", True):
            raise AssertionError("A non-causal or bidirectional configuration was detected.")


# =============================================================================
# 12. PUBLICATION FIGURES (DIRECT PREDICTIONS ONLY)
# =============================================================================


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def plot_single_quantity(
    frame: pd.DataFrame,
    cutter: str,
    quantity: str,
    output_path: Path,
    config: StudyConfig,
) -> None:
    labelled = cutter in LABELLED_CUTTERS
    if quantity == "wear":
        prediction_column = "wear_pred_um"
        sigma_column = "wear_std_um"
        truth_column = "wear_true_um"
        y_label = "Mean flank wear (µm)"
        title_word = "wear"
    elif quantity == "rul":
        prediction_column = "rul_pred_passes"
        sigma_column = "rul_std_passes"
        truth_column = "rul_true_passes"
        y_label = "Remaining useful life (cutting passes)"
        title_word = "RUL"
    else:
        raise ValueError(quantity)

    x = frame["cut_number"].to_numpy(dtype=float)
    mean = frame[prediction_column].to_numpy(dtype=float)
    sigma = frame[sigma_column].to_numpy(dtype=float)
    lower = np.maximum(mean - config.uncertainty_z * sigma, 0.0)
    upper = mean + config.uncertainty_z * sigma

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.fill_between(
        x, lower, upper, color="#D7263D", alpha=0.16, linewidth=0,
        label="PINDP-Net 95% predictive interval",
    )
    axis.plot(x, mean, color="#D7263D", linewidth=2.2, label="PINDP-Net")
    if labelled:
        axis.plot(
            x,
            frame[truth_column].to_numpy(dtype=float),
            color="#202020",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            markevery=max(1, len(frame) // 12),
            markerfacecolor="white",
            label=f"Reference {title_word}",
        )
    else:
        axis.text(
            0.025,
            0.06,
            "Blind inference—no released wear labels or reference RUL",
            transform=axis.transAxes,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#888888"},
        )

    if quantity == "wear":
        axis.axhline(
            config.failure_threshold_um,
            color="#8B1A1A",
            linestyle="--",
            linewidth=1.1,
            label=f"Failure threshold ({config.failure_threshold_um:g} µm)",
        )
    else:
        axis.axhline(0.0, color="#8B1A1A", linestyle=":", linewidth=1.0)

    axis.text(
        0.965,
        0.94,
        cutter.upper(),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "black"},
    )
    axis.set_xlabel("Cutting pass")
    axis.set_ylabel(y_label)
    axis.set_xlim(float(np.min(x)), float(np.max(x)))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True)
    figure.savefig(output_path, dpi=600, facecolor="white")
    plt.close(figure)


def plot_combined(
    all_predictions: Mapping[str, pd.DataFrame],
    quantity: str,
    output_path: Path,
    config: StudyConfig,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharey=False)
    for panel, cutter in enumerate(ALL_CUTTERS):
        axis = axes.flat[panel]
        frame = all_predictions[cutter]
        labelled = cutter in LABELLED_CUTTERS
        x = frame["cut_number"].to_numpy(dtype=float)
        if quantity == "wear":
            mean = frame["wear_pred_um"].to_numpy(dtype=float)
            sigma = frame["wear_std_um"].to_numpy(dtype=float)
            truth_column = "wear_true_um"
            y_label = "Mean flank wear (µm)"
        else:
            mean = frame["rul_pred_passes"].to_numpy(dtype=float)
            sigma = frame["rul_std_passes"].to_numpy(dtype=float)
            truth_column = "rul_true_passes"
            y_label = "RUL (cutting passes)"
        lower = np.maximum(mean - config.uncertainty_z * sigma, 0.0)
        upper = mean + config.uncertainty_z * sigma
        axis.fill_between(x, lower, upper, color="#D7263D", alpha=0.15, linewidth=0)
        axis.plot(x, mean, color="#D7263D", linewidth=1.8)
        if labelled:
            axis.plot(
                x,
                frame[truth_column],
                color="#202020",
                linewidth=1.5,
                marker="o",
                markersize=2.8,
                markevery=max(1, len(frame) // 10),
                markerfacecolor="white",
            )
        else:
            axis.text(
                0.03,
                0.06,
                "Blind inference",
                transform=axis.transAxes,
                fontsize=8,
                color="#555555",
            )
        if quantity == "wear":
            axis.axhline(config.failure_threshold_um, color="#8B1A1A", linestyle="--", linewidth=0.8)
        else:
            axis.axhline(0.0, color="#8B1A1A", linestyle=":", linewidth=0.8)
        axis.text(
            0.95,
            0.92,
            cutter.upper(),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "black"},
        )
        axis.set_xlabel("Cutting pass")
        if panel % 3 == 0:
            axis.set_ylabel(y_label)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, linestyle="--", linewidth=0.45, alpha=0.30)

    legend_handles = [
        Line2D([0], [0], color="#202020", marker="o", markerfacecolor="white", label="Reference (C1/C4/C6)"),
        Line2D([0], [0], color="#D7263D", linewidth=2.0, label="PINDP-Net"),
        Patch(facecolor="#D7263D", alpha=0.15, label="PINDP-Net 95% predictive interval"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=600, facecolor="white")
    plt.close(figure)


def create_publication_figures(
    oof: pd.DataFrame,
    blind: pd.DataFrame,
    figure_dir: Path,
    config: StudyConfig,
) -> None:
    configure_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[str, pd.DataFrame] = {}
    for cutter in ALL_CUTTERS:
        source = oof if cutter in LABELLED_CUTTERS else blind
        frame = source[source["cutter"] == cutter].sort_values("cut_number").copy()
        if frame.empty:
            raise RuntimeError(f"No predictions available for {cutter.upper()}.")
        all_predictions[cutter] = frame
        suffix = "OOF_Validation" if cutter in LABELLED_CUTTERS else "Blind_Inference"
        plot_single_quantity(
            frame,
            cutter,
            "wear",
            figure_dir / f"{cutter.upper()}_Wear_{suffix}.png",
            config,
        )
        plot_single_quantity(
            frame,
            cutter,
            "rul",
            figure_dir / f"{cutter.upper()}_RUL_{suffix}.png",
            config,
        )
    plot_combined(
        all_predictions,
        "wear",
        figure_dir / "Figure11_Causal_Wear_All_Six_Cutters.png",
        config,
    )
    plot_combined(
        all_predictions,
        "rul",
        figure_dir / "Figure12_Causal_RUL_All_Six_Cutters.png",
        config,
    )


# =============================================================================
# 13. COMPLETE PIPELINE
# =============================================================================


def run_pipeline(args: argparse.Namespace) -> None:
    start_time = time.time()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_cache"
    audit_dir = output_dir / "audit"
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"

    config = StudyConfig(
        wear_file_scale_to_um=args.wear_scale_to_um,
        failure_threshold_um=args.threshold_um,
        ensemble_members=1 if args.quick else args.ensemble_members,
        max_epochs=35 if args.quick else args.max_epochs,
        patience=10 if args.quick else args.patience,
        min_epochs=10 if args.quick else min(80, args.max_epochs),
        steps_per_epoch=2 if args.quick else args.steps_per_epoch,
        seed=args.seed,
    )
    candidates = candidate_hyperparameters(args.quick)
    device = choose_device(args.device)
    set_deterministic(config.seed)
    print(f"Device: {device}")
    print(f"Mode: {'QUICK PIPELINE CHECK—DO NOT REPORT' if args.quick else 'FULL REPRODUCIBLE RUN'}")

    signal_records, wear_files = discover_dataset(data_root)
    discovery_audit = {
        "data_root": data_root,
        "signal_file_counts": {c: len(v) for c, v in signal_records.items()},
        "wear_files": {c: str(p) for c, p in wear_files.items()},
        "source_hashes_sha256": {
            "wear_files": {c: sha256_file(p) for c, p in wear_files.items()},
            "first_and_last_sensor_files": {
                c: {
                    "first": sha256_file(records[0][1]),
                    "last": sha256_file(records[-1][1]),
                }
                for c, records in signal_records.items()
            },
        },
        "study_config": asdict(config),
    }
    write_json(audit_dir / "dataset_discovery.json", discovery_audit)

    feature_frames = extract_or_load_features(
        signal_records,
        cache_dir,
        config,
        rebuild=args.rebuild_features,
    )
    frames, failure_times = attach_targets(feature_frames, wear_files, config)

    # Save a compact target audit without copying raw sensor signals.
    write_json(
        audit_dir / "target_definition.json",
        {
            "labelled_cutters": LABELLED_CUTTERS,
            "blind_cutters": BLIND_CUTTERS,
            "failure_threshold_um": config.failure_threshold_um,
            "failure_times_passes": failure_times,
            "failure_crossing_rule": "first crossing with linear interpolation",
            "reference_rul": "max(T_fail - cut_number, 0)",
            "reference_rul_unit": "cutting passes",
            "smoothing": "none",
            "pseudo_labels_for_blind_cutters": False,
        },
    )

    # Outer leave-one-cutter-out evaluation. Every reported row is generated by
    # a model for which that entire cutter was absent from fitting and tuning.
    oof_parts: list[pd.DataFrame] = []
    outer_audits: list[dict[str, Any]] = []
    selected_hp_names: list[str] = []
    selected_epochs: list[int] = []
    selected_wear_scales: list[float] = []
    selected_rul_scales: list[float] = []

    print("\n" + "=" * 88)
    print("NESTED CUTTER-LEVEL LEAVE-ONE-CUTTER-OUT EVALUATION")
    print("=" * 88)
    for outer_index, holdout in enumerate(LABELLED_CUTTERS):
        train_frames = {c: frames[c] for c in LABELLED_CUTTERS if c != holdout}
        holdout_frames = {holdout: frames[holdout]}
        print(
            f"\n[OUTER {outer_index + 1}/3] holdout={holdout.upper()}, "
            f"training={','.join(c.upper() for c in train_frames)}"
        )
        hp, epochs, wear_scale, rul_scale, selection_audit = inner_model_selection(
            train_frames,
            candidates,
            config,
            device,
            audit_dir,
            fold_name=f"outer_{holdout}",
        )
        selected_hp_names.append(hp.name)
        selected_epochs.append(epochs)
        selected_wear_scales.append(wear_scale)
        selected_rul_scales.append(rul_scale)
        print(
            f"    selected {hp.name}; epochs={epochs}; "
            f"uncertainty scales wear={wear_scale:.3f}, RUL={rul_scale:.3f}"
        )
        prediction, training_audit = train_outer_ensemble(
            train_frames,
            holdout_frames,
            hp,
            epochs,
            wear_scale,
            rul_scale,
            config,
            device,
            checkpoint_dir / f"outer_{holdout}",
            fold_seed=config.seed + 100_000 * (outer_index + 1),
        )
        prediction["outer_holdout"] = holdout
        prediction["training_cutters"] = ",".join(sorted(train_frames))
        oof_parts.append(prediction)
        outer_audits.append(training_audit)

    assert_causal_and_disjoint(outer_audits)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["cutter", "cut_number"]
    )
    if set(oof["cutter"].unique()) != set(LABELLED_CUTTERS):
        raise AssertionError("OOF prediction file does not contain exactly C1/C4/C6.")
    oof.to_csv(output_dir / "pindpnet_oof_labelled.csv", index=False)

    metrics_table, metrics_detail = compute_oof_metrics(oof, config)
    metrics_table.to_csv(output_dir / "pindpnet_oof_metrics.csv", index=False)
    write_json(output_dir / "pindpnet_oof_metrics.json", metrics_detail)

    # Final blind model selection uses only outcomes from the nested labelled
    # analysis. It does not inspect any C2/C3/C5 label because none exists.
    hp_vote = Counter(selected_hp_names).most_common()
    winning_name = sorted(
        [name for name, count in hp_vote if count == hp_vote[0][1]]
    )[0]
    final_hp = next(hp for hp in candidates if hp.name == winning_name)
    final_epochs = int(round(float(np.median(selected_epochs))))
    final_wear_scale = float(np.median(selected_wear_scales))
    final_rul_scale = float(np.median(selected_rul_scales))
    labelled_frames = {c: frames[c] for c in LABELLED_CUTTERS}
    blind_frames = {c: frames[c] for c in BLIND_CUTTERS}
    print("\n" + "=" * 88)
    print("FINAL MODEL FOR BLIND C2/C3/C5 INFERENCE")
    print("=" * 88)
    print(
        f"Selected by outer-fold vote: {final_hp.name}; epochs={final_epochs}; "
        f"wear scale={final_wear_scale:.3f}; RUL scale={final_rul_scale:.3f}"
    )
    blind, blind_audit = train_outer_ensemble(
        labelled_frames,
        blind_frames,
        final_hp,
        final_epochs,
        final_wear_scale,
        final_rul_scale,
        config,
        device,
        checkpoint_dir / "final_blind_model",
        fold_seed=config.seed + 900_000,
    )
    assert_causal_and_disjoint([blind_audit])
    blind["reference_labels_available"] = False
    blind.to_csv(output_dir / "pindpnet_blind_c2_c3_c5.csv", index=False)

    create_publication_figures(oof, blind, figure_dir, config)

    complete_audit = {
        "runtime_seconds": time.time() - start_time,
        "device": device,
        "quick_mode": args.quick,
        "reported_results_permitted": not args.quick,
        "outer_folds": outer_audits,
        "final_blind_model": blind_audit,
        "selected_outer_hyperparameters": selected_hp_names,
        "final_hyperparameters": asdict(final_hp),
        "final_epochs": final_epochs,
        "failure_times": failure_times,
        "no_future_information": True,
        "cutter_level_separation": True,
        "training_only_preprocessing": True,
        "blind_cutters_used_in_supervised_loss": False,
        "prediction_postprocessing": "none",
    }
    write_json(output_dir / "complete_reproducibility_audit.json", complete_audit)

    print("\n" + "=" * 88)
    print("UNTOUCHED OUT-OF-FOLD RESULTS")
    print("=" * 88)
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFiles written:")
    print(f"  {output_dir / 'pindpnet_oof_labelled.csv'}")
    print(f"  {output_dir / 'pindpnet_blind_c2_c3_c5.csv'}")
    print(f"  {output_dir / 'pindpnet_oof_metrics.csv'}")
    print(f"  {figure_dir}")
    print(f"  {checkpoint_dir}")
    print(f"  {output_dir / 'complete_reproducibility_audit.json'}")
    if args.quick:
        print("\nWARNING: quick-mode results are only a software check and must not be reported.")
    print("=" * 88)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe, causal PINDP-Net pipeline for PHM 2010."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing the six raw cutter CSV trajectories and wear files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for caches, models, audit files, CSV outputs and PNG figures.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device (default: auto).",
    )
    parser.add_argument("--threshold-um", type=float, default=165.0)
    parser.add_argument(
        "--wear-scale-to-um",
        type=float,
        default=1.0,
        help="Multiply each flute value in the official wear CSV by this factor.",
    )
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=450)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--steps-per-epoch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore cached pass-level features and extract them again.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short software check only; generated metrics are not reportable.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.threshold_um <= 0:
        parser.error("--threshold-um must be positive.")
    if args.wear_scale_to_um <= 0:
        parser.error("--wear-scale-to-um must be positive.")
    if args.ensemble_members < 1:
        parser.error("--ensemble-members must be at least 1.")
    if args.max_epochs < 1 or args.patience < 1 or args.steps_per_epoch < 1:
        parser.error("Epoch, patience and step settings must be positive.")
    run_pipeline(args)


if __name__ == "__main__":
    main()
