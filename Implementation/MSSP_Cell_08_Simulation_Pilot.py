"""
=============================================================================
MSSP UPGRADE
Cell 8: synthetic null calibration and fault-detection power -- pilot
=============================================================================

Purpose
-------
This cell must be run after Cells 6 and 7. It does not rebuild CWRU or
Paderborn features. It tests whether the empirical probabilities produced by
extract_features_v3 have:

1. controlled false-positive rates for healthy signals;
2. useful detection power for BPFO, BPFI and BSF;
3. correct fault-order localization as SNR changes.

The pilot is deliberately modest. If it passes, a later cell will execute the
same registered design with publication-scale Monte Carlo replication.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve, lfilter
from scipy.stats import norm


# -------------------------------------------------------------------------
# 1. Prerequisite check
# -------------------------------------------------------------------------

required_cell8_names = [
    "extract_features_v3",
    "char_orders",
    "fast_len_leq",
    "MSSP_DIRS",
    "MSSP_COLORS",
    "save_publication_figure",
]

missing_cell8_names = [
    name for name in required_cell8_names
    if name not in globals()
]

if missing_cell8_names:
    raise RuntimeError(
        "Run Cells 6 and 7 first. Missing: "
        + ", ".join(missing_cell8_names)
    )


# -------------------------------------------------------------------------
# 2. Registered pilot configuration
# -------------------------------------------------------------------------

SIMULATION_MODE = "pilot"

SIM_CONFIG = {
    "mode": SIMULATION_MODE,
    "seed": 20260819,
    "bearing_scenarios": {
        "CWRU_DE_6205": {
            "fs": 12000.0,
            "revolutions": 40,
            "speeds_hz": [15.0, 25.0, 30.0],
            "fault_targets": ["BPFO", "BPFI", "BSF"],
        },
        "PADERBORN_6203": {
            "fs": 12000.0,
            "revolutions": 100,
            "speeds_hz": [15.0, 25.0],
            "fault_targets": ["BPFO", "BPFI"],
        },
    },
    "healthy_noise_profiles": [
        "white",
        "colored",
        "impulsive",
        "harmonic",
    ],
    "fault_noise_profiles": [
        "colored",
        "impulsive",
    ],
    "snr_db": [-12.0, -8.0, -4.0, 0.0],
    "n_repetitions_healthy": 5,
    "n_repetitions_fault": 5,
    "n_null": 199,
    "alpha": 0.05,
    "checkpoint_every": 20,
}

SIM_ROOT = Path(MSSP_DIRS["simulation"]) / SIMULATION_MODE
SIM_ROOT.mkdir(parents=True, exist_ok=True)

SIM_TRIAL_PATH = SIM_ROOT / f"simulation_trials_{SIMULATION_MODE}.csv"
SIM_CONFIG_PATH = SIM_ROOT / f"simulation_config_{SIMULATION_MODE}.json"
SIM_ERROR_PATH = SIM_ROOT / f"simulation_errors_{SIMULATION_MODE}.csv"


def canonical_json_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SIM_CONFIG_HASH = canonical_json_digest(SIM_CONFIG)

config_record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "config_sha256": SIM_CONFIG_HASH,
    "config": SIM_CONFIG,
}

if SIM_CONFIG_PATH.exists():
    with open(SIM_CONFIG_PATH, "r", encoding="utf-8") as file:
        existing_config_record = json.load(file)

    if existing_config_record.get("config_sha256") != SIM_CONFIG_HASH:
        raise RuntimeError(
            f"Configuration conflict at {SIM_CONFIG_PATH}. "
            "Rename the old pilot folder or restore the previous settings."
        )
else:
    with open(SIM_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config_record, file, indent=2)


# -------------------------------------------------------------------------
# 3. Physically motivated synthetic signal model
# -------------------------------------------------------------------------

def unit_rms(signal):
    signal = np.asarray(signal, dtype=float)
    rms = np.sqrt(np.mean(signal ** 2))
    return signal / (rms + 1e-12)


def colored_noise(rng, n, coefficient=0.92):
    white = rng.normal(size=n)
    colored = lfilter([1.0], [1.0, -coefficient], white)
    colored -= np.mean(colored)
    return unit_rms(colored)


def resonant_kernel(fs, resonance_hz, decay_rate, duration=0.035):
    n_kernel = max(32, int(duration * fs))
    time_axis = np.arange(n_kernel) / fs
    kernel = (
        np.exp(-decay_rate * time_axis) *
        np.sin(2 * np.pi * resonance_hz * time_axis)
    )
    return unit_rms(kernel)


def random_resonant_impulses(
    rng,
    n,
    fs,
    event_rate_hz,
    resonance_hz,
    decay_rate,
):
    duration = n / fs
    n_events = rng.poisson(event_rate_hz * duration)
    impulses = np.zeros(n, dtype=float)

    if n_events > 0:
        indices = rng.integers(0, n, size=n_events)
        amplitudes = rng.lognormal(mean=-0.2, sigma=0.55, size=n_events)
        amplitudes *= rng.choice([-1.0, 1.0], size=n_events)
        np.add.at(impulses, indices, amplitudes)

    kernel = resonant_kernel(
        fs=fs,
        resonance_hz=resonance_hz,
        decay_rate=decay_rate,
    )

    return fftconvolve(impulses, kernel, mode="full")[:n]


def periodic_fault_component(
    rng,
    n,
    fs,
    fr,
    fault_order,
    fault_name,
    bearing_key,
    resonance_hz,
    decay_rate,
    slip_std=0.004,
    timing_jitter_std=0.015,
):
    """
    Generate a quasi-periodic bearing-impact train.

    The recurrence interval contains small slip and timing variations.
    BPFI is load-zone modulated at shaft frequency. BSF is modulated by FTF.
    """
    duration = n / fs
    nominal_period = 1.0 / (fault_order * fr)
    impact_times = []
    current_time = float(rng.uniform(0, nominal_period))

    while current_time < duration:
        impact_times.append(current_time)
        local_period = nominal_period * (
            1.0 + rng.normal(0.0, slip_std)
        )
        timing_jitter = rng.normal(
            0.0,
            timing_jitter_std * nominal_period,
        )
        current_time += max(
            0.4 * nominal_period,
            local_period + timing_jitter,
        )

    impulses = np.zeros(n, dtype=float)
    phase = rng.uniform(0, 2 * np.pi)
    ftf_order = char_orders(bearing_key)["FTF"]

    for impact_time in impact_times:
        index = int(round(impact_time * fs))

        if not (0 <= index < n):
            continue

        if fault_name == "BPFI":
            modulation = 0.25 + 0.75 * (
                0.5 +
                0.5 * np.cos(2 * np.pi * fr * impact_time + phase)
            )
        elif fault_name == "BSF":
            modulation = 0.45 + 0.55 * (
                0.5 +
                0.5 * np.cos(
                    2 * np.pi * ftf_order * fr * impact_time + phase
                )
            )
        else:
            modulation = 0.9 + 0.1 * np.cos(
                2 * np.pi * fr * impact_time + phase
            )

        amplitude = modulation * rng.lognormal(
            mean=0.0,
            sigma=0.18,
        )
        impulses[index] += amplitude

    kernel = resonant_kernel(
        fs=fs,
        resonance_hz=resonance_hz,
        decay_rate=decay_rate,
    )

    component = fftconvolve(impulses, kernel, mode="full")[:n]
    component -= np.mean(component)
    return unit_rms(component)


def simulate_bearing_record(
    seed,
    bearing_key,
    fs,
    revolutions,
    fr,
    noise_profile,
    fault_name=None,
    snr_db=None,
):
    rng = np.random.default_rng(int(seed))
    fs = float(fs)
    revolutions = int(revolutions)
    n = fast_len_leq(int(revolutions * fs / fr))
    time_axis = np.arange(n) / fs

    # Trial-specific structural resonance prevents a fixed-band advantage.
    resonance_hz = float(rng.uniform(1500.0, 4600.0))
    decay_rate = float(rng.uniform(180.0, 650.0))

    if noise_profile == "white":
        stochastic = rng.normal(size=n)
    else:
        stochastic = colored_noise(rng, n, coefficient=0.90)

    stochastic = unit_rms(stochastic)

    # Deterministic shaft harmonics are present in all operating states.
    harmonic_strength = 0.18 if noise_profile != "harmonic" else 0.75
    shaft = np.zeros(n, dtype=float)

    for harmonic in range(1, 6):
        amplitude = harmonic_strength / (harmonic ** 0.8)
        phase = rng.uniform(0, 2 * np.pi)
        shaft += amplitude * np.sin(
            2 * np.pi * harmonic * fr * time_axis + phase
        )

    background = stochastic + shaft

    if noise_profile == "impulsive":
        nuisance = random_resonant_impulses(
            rng=rng,
            n=n,
            fs=fs,
            event_rate_hz=0.30 * fr,
            resonance_hz=resonance_hz,
            decay_rate=decay_rate,
        )
        background += 0.55 * unit_rms(nuisance)

    background -= np.mean(background)
    background_rms = np.sqrt(np.mean(background ** 2))

    fault_component = np.zeros(n, dtype=float)
    applied_scale = 0.0

    if fault_name is not None:
        if snr_db is None:
            raise ValueError("snr_db is required for a fault simulation")

        fault_order = char_orders(bearing_key)[fault_name]
        fault_component = periodic_fault_component(
            rng=rng,
            n=n,
            fs=fs,
            fr=fr,
            fault_order=fault_order,
            fault_name=fault_name,
            bearing_key=bearing_key,
            resonance_hz=resonance_hz,
            decay_rate=decay_rate,
        )

        requested_ratio = 10.0 ** (float(snr_db) / 20.0)
        fault_rms = np.sqrt(np.mean(fault_component ** 2))
        applied_scale = (
            requested_ratio * background_rms /
            (fault_rms + 1e-12)
        )

    signal = background + applied_scale * fault_component
    signal -= np.mean(signal)

    metadata = {
        "n_samples": int(n),
        "duration_seconds": float(n / fs),
        "resonance_hz": resonance_hz,
        "decay_rate": decay_rate,
        "background_rms": float(background_rms),
        "fault_scale": float(applied_scale),
    }

    return signal, metadata


# -------------------------------------------------------------------------
# 4. Deterministic Monte Carlo design
# -------------------------------------------------------------------------

def make_simulation_design(config):
    rows = []
    trial_counter = 0
    master_rng = np.random.default_rng(config["seed"])

    for bearing_key, scenario in config["bearing_scenarios"].items():
        for fr in scenario["speeds_hz"]:
            for profile in config["healthy_noise_profiles"]:
                for repetition in range(config["n_repetitions_healthy"]):
                    rows.append({
                        "trial_id": f"H0_{trial_counter:05d}",
                        "trial_seed": int(
                            master_rng.integers(0, 2 ** 32 - 1)
                        ),
                        "bearing_key": bearing_key,
                        "fs": float(scenario["fs"]),
                        "revolutions": int(scenario["revolutions"]),
                        "hypothesis": "H0",
                        "fault_name": "N",
                        "target_order": "NONE",
                        "fr": float(fr),
                        "rpm": float(60 * fr),
                        "noise_profile": profile,
                        "snr_db": np.nan,
                        "repetition": repetition,
                    })
                    trial_counter += 1

        for fault_name in scenario["fault_targets"]:
            for snr_db in config["snr_db"]:
                for fr in scenario["speeds_hz"]:
                    for profile in config["fault_noise_profiles"]:
                        for repetition in range(
                            config["n_repetitions_fault"]
                        ):
                            rows.append({
                                "trial_id": f"H1_{trial_counter:05d}",
                                "trial_seed": int(
                                    master_rng.integers(0, 2 ** 32 - 1)
                                ),
                                "bearing_key": bearing_key,
                                "fs": float(scenario["fs"]),
                                "revolutions": int(
                                    scenario["revolutions"]
                                ),
                                "hypothesis": "H1",
                                "fault_name": {
                                    "BPFO": "OR",
                                    "BPFI": "IR",
                                    "BSF": "B",
                                }[fault_name],
                                "target_order": fault_name,
                                "fr": float(fr),
                                "rpm": float(60 * fr),
                                "noise_profile": profile,
                                "snr_db": float(snr_db),
                                "repetition": repetition,
                            })
                            trial_counter += 1

    design = pd.DataFrame(rows)
    design["config_sha256"] = SIM_CONFIG_HASH
    return design


SIM_DESIGN = make_simulation_design(SIM_CONFIG)

print("Synthetic calibration pilot")
print(f"  registered trials : {len(SIM_DESIGN)}")
print(
    "  healthy trials    :",
    int((SIM_DESIGN.hypothesis == "H0").sum()),
)
print(
    "  faulty trials     :",
    int((SIM_DESIGN.hypothesis == "H1").sum()),
)
print(f"  output             : {SIM_ROOT.resolve()}\n")


# -------------------------------------------------------------------------
# 5. Resumable execution
# -------------------------------------------------------------------------

def atomic_csv_write(dataframe, path):
    path = Path(path)
    temporary = path.with_name(path.stem + "_temporary.csv")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


if SIM_TRIAL_PATH.exists():
    completed_frame = pd.read_csv(SIM_TRIAL_PATH)

    if not completed_frame.empty:
        old_hashes = set(completed_frame["config_sha256"].unique())

        if old_hashes != {SIM_CONFIG_HASH}:
            raise RuntimeError(
                "Existing simulation cache has a different configuration."
            )
else:
    completed_frame = pd.DataFrame()

completed_ids = (
    set(completed_frame["trial_id"])
    if not completed_frame.empty
    else set()
)

pending_design = SIM_DESIGN[
    ~SIM_DESIGN["trial_id"].isin(completed_ids)
].copy()

print(
    f"  cache contains     : {len(completed_ids)} completed trials\n"
    f"  pending            : {len(pending_design)} trials\n"
)

new_rows = []
error_rows = []
start_time = time.time()

for run_index, trial in enumerate(
    pending_design.itertuples(index=False),
    start=1,
):
    try:
        fault_target = (
            None if trial.hypothesis == "H0" else trial.target_order
        )
        trial_snr = (
            None if trial.hypothesis == "H0" else trial.snr_db
        )

        signal, simulation_metadata = simulate_bearing_record(
            seed=trial.trial_seed,
            bearing_key=trial.bearing_key,
            fs=trial.fs,
            revolutions=trial.revolutions,
            fr=trial.fr,
            noise_profile=trial.noise_profile,
            fault_name=fault_target,
            snr_db=trial_snr,
        )

        extraction_start = time.time()
        features = extract_features_v3(
            sig=signal,
            fs=trial.fs,
            fr=trial.fr,
            bearing_key=trial.bearing_key,
            seed_str=f"SIM::{trial.trial_id}",
            n_null=SIM_CONFIG["n_null"],
            alpha=SIM_CONFIG["alpha"],
            return_detail=False,
        )
        extraction_seconds = time.time() - extraction_start

        result = trial._asdict()
        result.update(simulation_metadata)
        result["extract_seconds"] = float(extraction_seconds)
        result["argmax_order"] = features["argmax_order"]
        result["argmax_raw_order"] = features["argmax_raw_order"]
        result["max_z"] = features["max_z"]
        result["margin"] = features["margin"]
        result["margin_raw"] = features["margin_raw"]
        result["min_p_emp"] = features["min_p_emp"]
        result["min_p_holm"] = features["min_p_holm"]
        result["n_sig_raw"] = features["n_sig_raw"]
        result["n_sig_holm"] = features["n_sig_holm"]
        result["n_sig_bh"] = features["n_sig_bh"]

        for order_name in ("BPFO", "BPFI", "BSF"):
            result[f"{order_name}_p2f"] = features.get(
                f"{order_name}_p2f",
                np.nan,
            )
            result[f"{order_name}_z"] = features.get(
                f"{order_name}_z",
                np.nan,
            )
            result[f"{order_name}_p_emp"] = features.get(
                f"{order_name}_p_emp",
                np.nan,
            )
            result[f"{order_name}_p_holm"] = features.get(
                f"{order_name}_p_holm",
                np.nan,
            )
            result[f"{order_name}_q_bh"] = features.get(
                f"{order_name}_q_bh",
                np.nan,
            )

        result["correct_localization"] = float(
            trial.hypothesis == "H1" and
            features["argmax_order"] == trial.target_order
        )
        result["correct_raw_localization"] = float(
            trial.hypothesis == "H1" and
            features["argmax_raw_order"] == trial.target_order
        )

        if trial.hypothesis == "H1":
            result["target_p_emp"] = features[
                f"{trial.target_order}_p_emp"
            ]
            result["target_p_holm"] = features[
                f"{trial.target_order}_p_holm"
            ]
            result["target_q_bh"] = features[
                f"{trial.target_order}_q_bh"
            ]
            result["target_z"] = features[f"{trial.target_order}_z"]
        else:
            result["target_p_emp"] = np.nan
            result["target_p_holm"] = np.nan
            result["target_q_bh"] = np.nan
            result["target_z"] = np.nan

        new_rows.append(result)

    except Exception as error:
        error_rows.append({
            "trial_id": trial.trial_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

    checkpoint_due = (
        run_index % SIM_CONFIG["checkpoint_every"] == 0 or
        run_index == len(pending_design)
    )

    if checkpoint_due:
        accumulated = pd.concat(
            [
                completed_frame,
                pd.DataFrame(new_rows),
            ],
            ignore_index=True,
        )

        if not accumulated.empty:
            accumulated = (
                accumulated
                .drop_duplicates("trial_id", keep="last")
                .sort_values("trial_id")
                .reset_index(drop=True)
            )
            atomic_csv_write(accumulated, SIM_TRIAL_PATH)

        if error_rows:
            pd.DataFrame(error_rows).to_csv(
                SIM_ERROR_PATH,
                index=False,
            )

        elapsed = time.time() - start_time
        rate = run_index / max(elapsed, 1e-12)
        remaining = len(pending_design) - run_index
        eta_seconds = remaining / max(rate, 1e-12)

        print(
            f"  {run_index:>4}/{len(pending_design)} pending trials  "
            f"elapsed={elapsed / 60:6.2f} min  "
            f"ETA={eta_seconds / 60:6.2f} min  "
            f"errors={len(error_rows)}"
        )


if SIM_TRIAL_PATH.exists():
    SIM_TRIALS = pd.read_csv(SIM_TRIAL_PATH)
else:
    raise RuntimeError("No simulation results were produced")

if error_rows:
    raise RuntimeError(
        f"{len(error_rows)} trials failed. Inspect {SIM_ERROR_PATH}."
    )

expected_ids = set(SIM_DESIGN["trial_id"])
actual_ids = set(SIM_TRIALS["trial_id"])

if actual_ids != expected_ids:
    missing_ids = sorted(expected_ids - actual_ids)
    raise RuntimeError(
        f"Simulation incomplete: {len(missing_ids)} trials missing."
    )


# -------------------------------------------------------------------------
# 6. Statistical summaries
# -------------------------------------------------------------------------

def wilson_interval(successes, total, confidence=0.95):
    if total <= 0:
        return np.nan, np.nan

    z_value = norm.ppf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z_value ** 2 / total
    centre = (
        proportion + z_value ** 2 / (2.0 * total)
    ) / denominator
    half_width = (
        z_value /
        denominator *
        np.sqrt(
            proportion * (1.0 - proportion) / total +
            z_value ** 2 / (4.0 * total ** 2)
        )
    )
    return centre - half_width, centre + half_width


def proportion_summary(dataframe, group_columns, success_column, name):
    rows = []

    grouped = dataframe.groupby(group_columns, dropna=False)

    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        total = len(group)
        successes = int(group[success_column].sum())
        estimate = successes / total
        lower, upper = wilson_interval(successes, total)

        row = dict(zip(group_columns, group_values))
        row.update({
            "metric": name,
            "successes": successes,
            "n": total,
            "estimate": estimate,
            "ci95_low": lower,
            "ci95_high": upper,
        })
        rows.append(row)

    return pd.DataFrame(rows)


alpha_grid = np.array([0.01, 0.025, 0.05, 0.10, 0.20])
healthy = SIM_TRIALS[SIM_TRIALS.hypothesis == "H0"].copy()
faulty = SIM_TRIALS[SIM_TRIALS.hypothesis == "H1"].copy()

healthy["family_false_positive"] = (
    healthy["min_p_holm"] <= SIM_CONFIG["alpha"]
).astype(int)

faulty["target_detected_holm"] = (
    faulty["target_p_holm"] <= SIM_CONFIG["alpha"]
).astype(int)
faulty["target_detected_raw"] = (
    faulty["target_p_emp"] <= SIM_CONFIG["alpha"]
).astype(int)

calibration_rows = []

for (bearing_key, profile), group in healthy.groupby(
    ["bearing_key", "noise_profile"]
):
    for nominal_alpha in alpha_grid:
        observed = float(
            np.mean(group["min_p_holm"] <= nominal_alpha)
        )
        calibration_rows.append({
            "bearing_key": bearing_key,
            "noise_profile": profile,
            "nominal_alpha": nominal_alpha,
            "observed_fwer": observed,
            "n": len(group),
        })

SIM_NULL_CALIBRATION = pd.DataFrame(calibration_rows)

SIM_FWER = proportion_summary(
    healthy,
    ["bearing_key", "noise_profile", "fr"],
    "family_false_positive",
    "FWER",
)

SIM_POWER = proportion_summary(
    faulty,
    [
        "bearing_key", "target_order", "noise_profile", "snr_db", "fr"
    ],
    "target_detected_holm",
    "Holm detection power",
)

SIM_LOCALIZATION = proportion_summary(
    faulty,
    [
        "bearing_key", "target_order", "noise_profile", "snr_db", "fr"
    ],
    "correct_localization",
    "Calibrated-order localization",
)

SIM_RAW_LOCALIZATION = proportion_summary(
    faulty,
    [
        "bearing_key", "target_order", "noise_profile", "snr_db", "fr"
    ],
    "correct_raw_localization",
    "Raw-order localization",
)

table_outputs = {
    f"T_sim_null_calibration_{SIMULATION_MODE}.csv": SIM_NULL_CALIBRATION,
    f"T_sim_fwer_{SIMULATION_MODE}.csv": SIM_FWER,
    f"T_sim_power_{SIMULATION_MODE}.csv": SIM_POWER,
    f"T_sim_localization_{SIMULATION_MODE}.csv": SIM_LOCALIZATION,
    f"T_sim_raw_localization_{SIMULATION_MODE}.csv": SIM_RAW_LOCALIZATION,
}

for filename, dataframe in table_outputs.items():
    dataframe.to_csv(
        Path(MSSP_DIRS["tables"]) / filename,
        index=False,
    )


# -------------------------------------------------------------------------
# 7. Publication diagnostics
# -------------------------------------------------------------------------

def short_bearing_name(bearing_key):
    name = bearing_key.replace("CWRU_DE_", "CWRU ")
    return name.replace("PADERBORN_", "PB ")


def plot_calibration_curve():
    bearing_keys = list(SIM_CONFIG["bearing_scenarios"])
    fig, axes = plt.subplots(
        1,
        len(bearing_keys),
        figsize=(10.4, 4.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    profile_colors = {
        "white": "#4C78A8",
        "colored": "#F58518",
        "impulsive": "#E45756",
        "harmonic": "#72B7B2",
    }

    for axis, bearing_key in zip(axes, bearing_keys):
        axis.plot(
            [0, 0.21],
            [0, 0.21],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Ideal calibration",
        )

        bearing_data = SIM_NULL_CALIBRATION[
            SIM_NULL_CALIBRATION.bearing_key == bearing_key
        ]

        for profile, group in bearing_data.groupby("noise_profile"):
            group = group.sort_values("nominal_alpha")
            axis.plot(
                group["nominal_alpha"],
                group["observed_fwer"],
                marker="o",
                color=profile_colors[profile],
                label=profile,
            )

        axis.axvline(0.05, color="0.75", linewidth=0.9)
        axis.axhline(0.05, color="0.75", linewidth=0.9)
        axis.set_xlim(0, 0.21)
        axis.set_ylim(0, 0.40)
        axis.set_xlabel("Nominal family-wise error rate")
        axis.set_title(short_bearing_name(bearing_key))
        axis.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("Observed false-positive rate")
    fig.suptitle("Healthy-signal empirical-null calibration", y=1.02)
    fig.tight_layout()
    return fig


def plot_fwer_by_condition():
    plot_data = SIM_FWER.sort_values(
        ["bearing_key", "noise_profile", "fr"]
    ).copy()
    labels = [
        f"{short_bearing_name(bearing)}\n{profile}\n{fr * 60:.0f} rpm"
        for bearing, profile, fr in zip(
            plot_data["bearing_key"],
            plot_data["noise_profile"],
            plot_data["fr"],
        )
    ]
    positions = np.arange(len(plot_data))
    estimates = plot_data["estimate"].to_numpy()
    lower = estimates - plot_data["ci95_low"].to_numpy()
    upper = plot_data["ci95_high"].to_numpy() - estimates

    fig, axis = plt.subplots(figsize=(13.2, 4.8))
    axis.bar(positions, estimates, color="#4C78A8", alpha=0.85)
    axis.errorbar(
        positions,
        estimates,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.0,
    )
    axis.axhline(
        SIM_CONFIG["alpha"],
        color="#E45756",
        linestyle="--",
        label="Nominal 0.05",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=50, ha="right", fontsize=7.5)
    axis.set_ylabel("Family-wise false-positive rate")
    axis.set_title("False alarms by geometry, speed and noise regime")
    axis.set_ylim(0, max(0.45, float(np.max(upper + estimates)) + 0.05))
    axis.legend()
    fig.tight_layout()
    return fig


def plot_fault_curves(summary, title, ylabel):
    curve_definitions = list(
        summary[["bearing_key", "target_order"]]
        .drop_duplicates()
        .sort_values(["bearing_key", "target_order"])
        .itertuples(index=False, name=None)
    )
    n_columns = 3
    n_rows = int(np.ceil(len(curve_definitions) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12.2, 3.7 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    colors = {"colored": "#4C78A8", "impulsive": "#E45756"}

    for axis, (bearing_key, target) in zip(flat_axes, curve_definitions):
        target_data = summary[
            (summary.bearing_key == bearing_key) &
            (summary.target_order == target)
        ]

        for profile, group in target_data.groupby("noise_profile"):
            aggregated = (
                group.groupby("snr_db")
                .apply(
                    lambda item: pd.Series({
                        "successes": item["successes"].sum(),
                        "n": item["n"].sum(),
                    }),
                    include_groups=False,
                )
                .reset_index()
            )
            aggregated["estimate"] = (
                aggregated["successes"] / aggregated["n"]
            )
            axis.plot(
                aggregated["snr_db"],
                aggregated["estimate"],
                marker="o",
                color=colors[profile],
                label=profile,
            )

        axis.set_title(f"{short_bearing_name(bearing_key)}: {target}")
        axis.set_xlabel("Fault-to-background SNR (dB)")
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(SIM_CONFIG["snr_db"])
        axis.legend(loc="lower right", fontsize=8)

    for axis in flat_axes[len(curve_definitions):]:
        axis.set_visible(False)

    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel(ylabel)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def plot_healthy_pvalue_histograms():
    bearing_keys = list(SIM_CONFIG["bearing_scenarios"])
    order_names = ("BPFO", "BPFI", "BSF")
    fig, axes = plt.subplots(
        len(bearing_keys),
        len(order_names),
        figsize=(11.4, 3.4 * len(bearing_keys)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_index, bearing_key in enumerate(bearing_keys):
        bearing_data = healthy[healthy.bearing_key == bearing_key]

        for column_index, order_name in enumerate(order_names):
            axis = axes[row_index, column_index]
            values = (
                bearing_data[f"{order_name}_p_emp"]
                .dropna()
                .to_numpy()
            )

            if values.size == 0:
                axis.text(
                    0.5,
                    0.5,
                    "Order excluded\nby resolution test",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_title(order_name)
                continue

            axis.hist(
                values,
                bins=np.linspace(0, 1, 11),
                density=True,
                color=MSSP_COLORS[order_name],
                alpha=0.8,
                edgecolor="white",
            )
            axis.axhline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Uniform reference",
            )
            axis.set_title(order_name)
            axis.set_xlim(0, 1)

        axes[row_index, 0].set_ylabel(
            f"{short_bearing_name(bearing_key)}\nProbability density"
        )

    for axis in axes[-1, :]:
        axis.set_xlabel("Raw empirical probability")

    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Healthy-signal empirical probability distributions", y=1.01)
    fig.tight_layout()
    return fig


figure_jobs = [
    (
        plot_calibration_curve(),
        "S002_simulation_null_calibration_pilot",
    ),
    (
        plot_fwer_by_condition(),
        "S003_simulation_fwer_by_condition_pilot",
    ),
    (
        plot_fault_curves(
            SIM_POWER,
            "Fault detection power after Holm correction",
            "Detection probability",
        ),
        "S004_simulation_detection_power_pilot",
    ),
    (
        plot_fault_curves(
            SIM_LOCALIZATION,
            "Calibrated fault-order localization",
            "Correct localization probability",
        ),
        "S005_simulation_calibrated_localization_pilot",
    ),
    (
        plot_fault_curves(
            SIM_RAW_LOCALIZATION,
            "Raw peak-to-floor fault-order localization",
            "Correct localization probability",
        ),
        "S006_simulation_raw_localization_pilot",
    ),
    (
        plot_healthy_pvalue_histograms(),
        "S007_simulation_healthy_pvalues_pilot",
    ),
]

CELL8_FIGURE_PATHS = {}

for figure, stem in figure_jobs:
    CELL8_FIGURE_PATHS[stem] = save_publication_figure(
        figure,
        stem,
        supplementary=True,
    )
    plt.show()
    plt.close(figure)


# -------------------------------------------------------------------------
# 8. Concise decision report
# -------------------------------------------------------------------------

overall_false_positives = int(healthy["family_false_positive"].sum())
overall_healthy_n = len(healthy)
overall_fwer = overall_false_positives / overall_healthy_n
overall_fwer_ci = wilson_interval(
    overall_false_positives,
    overall_healthy_n,
)

power_overall = (
    faulty.groupby(
        ["bearing_key", "target_order", "snr_db"]
    )["target_detected_holm"]
    .mean()
    .unstack("snr_db")
)

localization_overall = (
    faulty.groupby(
        ["bearing_key", "target_order", "snr_db"]
    )["correct_localization"]
    .mean()
    .unstack("snr_db")
)

raw_localization_overall = (
    faulty.groupby(["bearing_key", "target_order", "snr_db"])[
        "correct_raw_localization"
    ]
    .mean()
    .unstack("snr_db")
)

print("\n" + "=" * 88)
print("CELL 8 PILOT DECISION REPORT")
print("=" * 88)
print(
    f"Healthy family-wise false-positive rate: {overall_fwer:.4f} "
    f"({overall_false_positives}/{overall_healthy_n}), "
    f"Wilson 95% CI [{overall_fwer_ci[0]:.4f}, "
    f"{overall_fwer_ci[1]:.4f}]"
)

print("\nHolm-corrected detection power:")
print(power_overall.round(3).to_string())

print("\nCalibrated-order localization accuracy:")
print(localization_overall.round(3).to_string())

print("\nRaw-order localization accuracy:")
print(raw_localization_overall.round(3).to_string())

print("\nMedian extraction time:")
print(f"  {SIM_TRIALS['extract_seconds'].median():.3f} seconds/trial")

print("\nSaved simulation trials:")
print(f"  {SIM_TRIAL_PATH.resolve()}")

print("\nSaved tables:")
for filename in table_outputs:
    print(f"  {(Path(MSSP_DIRS['tables']) / filename).resolve()}")

print("\nCELL 8 COMPLETE")
print(
    "Send the complete decision report. We will use its false-positive "
    "rate and target-specific power to accept or revise the null design "
    "before running the full Monte Carlo experiment."
)
