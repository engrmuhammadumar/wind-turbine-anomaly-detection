"""
=============================================================================
MSSP UPGRADE
Cell 7: search-matched empirical-null feature extractor, version 3
=============================================================================
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, rfft, rfftfreq, fftfreq
from scipy.signal import get_window
from scipy.stats import kurtosis, skew


# -------------------------------------------------------------------------
# 1. Statistical utilities
# -------------------------------------------------------------------------

def empirical_upper_pvalue(observed, null_values):
    """Finite-sample, one-sided empirical p-value with +1 correction."""
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]

    if null_values.size == 0 or not np.isfinite(observed):
        return np.nan

    return float(
        (1 + np.count_nonzero(null_values >= observed)) /
        (null_values.size + 1)
    )


def holm_adjust(pvalues):
    """Holm family-wise-error adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(pvalues))

    if valid.size == 0:
        return adjusted

    p = pvalues[valid]
    sort_index = np.argsort(p)
    p_sorted = p[sort_index]
    m = len(p_sorted)

    adjusted_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * p_sorted
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    original_positions = valid[sort_index]
    adjusted[original_positions] = adjusted_sorted

    return adjusted


def benjamini_hochberg_adjust(pvalues):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(pvalues))

    if valid.size == 0:
        return adjusted

    p = pvalues[valid]
    sort_index = np.argsort(p)
    p_sorted = p[sort_index]
    m = len(p_sorted)
    ranks = np.arange(1, m + 1)

    adjusted_sorted = p_sorted * m / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    original_positions = valid[sort_index]
    adjusted[original_positions] = adjusted_sorted

    return adjusted


def robust_mad(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)) + 1e-12)


# -------------------------------------------------------------------------
# 2. Resolution-aware characteristic-order selection
# -------------------------------------------------------------------------

def nearest_positive_integer_distance(order, max_integer=20):
    integers = np.arange(1, max_integer + 1, dtype=float)
    index = int(np.argmin(np.abs(integers - order)))
    nearest = int(integers[index])
    return nearest, float(abs(order - nearest))


def usable_orders_v3(
    bearing_key,
    fr,
    df_resolution,
    order_tolerance=0.03,
    min_collision_bins=2.0,
    collision_guard_bins=1.0,
):
    """
    Keep a characteristic order only if its search window is separable
    from the nearest positive integer shaft harmonic.

    FTF is retained because it represents cage modulation rather than a
    competing race/ball fault class.
    """
    kept = {}
    dropped = {}

    for name, order in char_orders(bearing_key).items():
        nearest_integer, separation_order = (
            nearest_positive_integer_distance(order)
        )

        separation_bins = separation_order * fr / df_resolution
        half_window_bins = order_tolerance * fr / df_resolution
        required_bins = max(
            min_collision_bins,
            half_window_bins + collision_guard_bins,
        )

        diagnostics = {
            "order": float(order),
            "nearest_integer": nearest_integer,
            "separation_order": separation_order,
            "separation_bins": float(separation_bins),
            "required_bins": float(required_bins),
        }

        if name == "FTF" or separation_bins >= required_bins:
            kept[name] = float(order)
        else:
            dropped[name] = diagnostics

    return kept, dropped


# -------------------------------------------------------------------------
# 3. Search-matched null-order generator
# -------------------------------------------------------------------------

def physical_forbidden_order_centres(
    bearing_key,
    lo,
    hi,
    n_harm=3,
    include_first_sidebands=True,
):
    """
    Physical orders excluded from the empirical-null pool:
      - bearing characteristic fundamentals;
      - race/ball fault harmonics up to n_harm;
      - first shaft-order sidebands of those harmonics.

    Integer shaft orders are deliberately not excluded here. Each target's
    null orders are instead matched to the target's distance from its
    nearest integer harmonic. This makes the null fair for orders such as
    Paderborn BPFO/BPFI, which lie close to 3x/5x shaft speed.

    FTF harmonics are not recursively excluded because doing so can cover
    nearly the full order interval. Its fundamental is still excluded.
    """
    characteristic = char_orders(bearing_key)
    centres = set()

    for order in characteristic.values():
        if lo <= order <= hi:
            centres.add(float(order))

    for name in ("BPFO", "BPFI", "BSF"):
        base = characteristic[name]

        for harmonic in range(1, n_harm + 1):
            centre = harmonic * base

            if lo <= centre <= hi:
                centres.add(float(centre))

            if include_first_sidebands:
                for sign in (-1.0, 1.0):
                    sideband = centre + sign

                    if lo <= sideband <= hi:
                        centres.add(float(sideband))

    return np.array(sorted(centres), dtype=float)


def collision_distance(orders):
    """Distance of each order from the nearest positive shaft integer."""
    orders = np.asarray(orders, dtype=float)
    nearest = np.maximum(1.0, np.round(orders))
    return np.abs(orders - nearest)


def search_collision_matched_null_orders(
    seed_string,
    bearing_key,
    target_name,
    target_order,
    n_null=199,
    lo=1.25,
    hi=8.0,
    physical_guard=0.04,
    zeta_match_width=0.02,
    grid_size=50000,
    n_harm=3,
):
    """
    Draw deterministic null orders matched to the target's collision
    margin: distance from the nearest integer shaft harmonic.

    Each null probe therefore experiences approximately the same risk of
    leakage from shaft harmonics as its physical target. This is stricter
    than simply drawing nulls far away from every shaft harmonic.
    """
    if n_null < 19:
        raise ValueError(
            "n_null must be at least 19 to permit an empirical p <= 0.05"
        )

    combined_seed = f"{seed_string}::{bearing_key}::{target_name}"
    digest = hashlib.sha256(combined_seed.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2 ** 32)
    rng = np.random.default_rng(seed)

    centres = physical_forbidden_order_centres(
        bearing_key=bearing_key,
        lo=lo,
        hi=hi,
        n_harm=n_harm,
        include_first_sidebands=True,
    )

    grid = np.linspace(lo, hi, int(grid_size), dtype=float)
    allowed = np.ones(grid.size, dtype=bool)

    for centre in centres:
        allowed &= np.abs(grid - centre) > physical_guard

    target_zeta = nearest_positive_integer_distance(target_order)[1]
    grid_zeta = collision_distance(grid)
    allowed &= np.abs(grid_zeta - target_zeta) <= zeta_match_width

    pool = grid[allowed]

    if pool.size < n_null:
        raise RuntimeError(
            f"Only {pool.size} admissible null orders for "
            f"{bearing_key}/{target_name}; requested {n_null}. "
            "Increase grid_size or zeta_match_width."
        )

    selected = rng.choice(pool, size=n_null, replace=False)
    selected.sort()

    return selected, {
        "seed": int(seed),
        "seed_hash": digest,
        "n_pool": int(pool.size),
        "allowed_fraction": float(pool.size / grid.size),
        "target_name": str(target_name),
        "target_order": float(target_order),
        "target_zeta": float(target_zeta),
        "zeta_match_width": float(zeta_match_width),
        "physical_guard": float(physical_guard),
        "forbidden_centres": centres,
    }


# -------------------------------------------------------------------------
# 4. Fast frequency-window construction
# -------------------------------------------------------------------------

def frequency_slice(frequency_axis, lower, upper):
    start = int(np.searchsorted(frequency_axis, lower, side="left"))
    stop = int(np.searchsorted(frequency_axis, upper, side="right"))

    start = max(0, min(start, len(frequency_axis)))
    stop = max(start, min(stop, len(frequency_axis)))

    return slice(start, stop)


def make_order_windows(
    frequency_axis,
    order,
    fr,
    tolerance_hz,
    side_inner=3.0,
    side_outer=10.0,
):
    target = order * fr

    peak = frequency_slice(
        frequency_axis,
        target - tolerance_hz,
        target + tolerance_hz,
    )

    side_left = frequency_slice(
        frequency_axis,
        target - side_outer * tolerance_hz,
        target - side_inner * tolerance_hz,
    )

    side_right = frequency_slice(
        frequency_axis,
        target + side_inner * tolerance_hz,
        target + side_outer * tolerance_hz,
    )

    return {
        "target_hz": float(target),
        "peak": peak,
        "side_left": side_left,
        "side_right": side_right,
    }


def slice_length(item):
    return max(0, item.stop - item.start)


def band_analytic_v3(full_fft, full_frequency, f_low, f_high):
    transfer = np.zeros(full_fft.size, dtype=float)
    positive_band = (
        (full_frequency >= f_low) &
        (full_frequency <= f_high)
    )
    transfer[positive_band] = 2.0

    return ifft(full_fft * transfer)


# -------------------------------------------------------------------------
# 5. Calibrated order-targeted feature extractor
# -------------------------------------------------------------------------

def extract_features_v3(
    sig,
    fs,
    fr,
    bearing_key,
    seed_str="x",
    base_fmax=200.0,
    n_harm=3,
    n_null=199,
    null_lo=1.25,
    null_hi=8.0,
    null_physical_guard=0.04,
    null_zeta_width=None,
    order_tolerance=0.03,
    alpha=0.05,
    min_collision_bins=2.0,
    collision_guard_bins=1.0,
    return_detail=False,
):
    """
    Label-free, order-targeted squared-envelope descriptors with:
      1. adaptive frequency coverage for harmonics;
      2. resolution/collision-aware order selection;
      3. a band-search- and shaft-collision-matched empirical null;
      4. finite-sample empirical p-values;
      5. Holm FWER and Benjamini-Hochberg FDR correction.
    """
    if not np.isfinite(fr) or fr <= 0:
        raise ValueError(f"Invalid rotational frequency: {fr}")

    n = fast_len_leq(len(sig))
    x = np.asarray(sig[:n], dtype=float)

    if n < 1024 or not np.isfinite(x).all():
        raise ValueError(f"Invalid signal: n={n}")

    x_centered = x - np.mean(x)
    full_fft = fft(x_centered)
    full_frequency = fftfreq(n, d=1.0 / fs)

    envelope_frequency_full = rfftfreq(n, d=1.0 / fs)
    df_resolution = float(envelope_frequency_full[1])

    characteristic_all = char_orders(bearing_key)

    # Include target harmonics and their first shaft sidebands.
    largest_required_order = max(
        n_harm * order + 1.0
        for name, order in characteristic_all.items()
        if name in ("BPFO", "BPFI", "BSF")
    )

    analysis_fmax = max(
        float(base_fmax),
        float(largest_required_order * fr + 5 * df_resolution),
        float(
            (null_hi + 10 * order_tolerance) * fr +
            5 * df_resolution
        ),
    )
    analysis_fmax = min(analysis_fmax, fs / 2.0 - df_resolution)

    frequency_keep = envelope_frequency_full <= analysis_fmax
    envelope_frequency = envelope_frequency_full[frequency_keep]

    tolerance_hz = max(
        1.5 * df_resolution,
        order_tolerance * fr,
    )

    kept_orders, dropped_orders = usable_orders_v3(
        bearing_key=bearing_key,
        fr=fr,
        df_resolution=df_resolution,
        order_tolerance=order_tolerance,
        min_collision_bins=min_collision_bins,
        collision_guard_bins=collision_guard_bins,
    )

    fault_orders = [
        name for name in ("BPFO", "BPFI", "BSF")
        if name in kept_orders
    ]

    probes = dict(kept_orders)
    null_orders_by_target = {}
    null_metadata_by_target = {}
    null_keys_by_target = {}

    zeta_width = (
        max(0.015, df_resolution / fr)
        if null_zeta_width is None
        else float(null_zeta_width)
    )

    for target_name in fault_orders:
        target_null_orders, target_null_metadata = (
            search_collision_matched_null_orders(
                seed_string=seed_str,
                bearing_key=bearing_key,
                target_name=target_name,
                target_order=kept_orders[target_name],
                n_null=n_null,
                lo=null_lo,
                hi=null_hi,
                physical_guard=null_physical_guard,
                zeta_match_width=zeta_width,
                n_harm=n_harm,
            )
        )

        target_null_keys = []

        for index, order in enumerate(target_null_orders):
            key = f"NULL_{target_name}_{index:03d}"
            probes[key] = float(order)
            target_null_keys.append(key)

        null_orders_by_target[target_name] = target_null_orders
        null_metadata_by_target[target_name] = target_null_metadata
        null_keys_by_target[target_name] = target_null_keys

    windows = {
        name: make_order_windows(
            frequency_axis=envelope_frequency,
            order=order,
            fr=fr,
            tolerance_hz=tolerance_hz,
        )
        for name, order in probes.items()
    }

    valid_probes = {}

    for name, window in windows.items():
        enough_peak = slice_length(window["peak"]) >= 1
        enough_floor = (
            slice_length(window["side_left"]) +
            slice_length(window["side_right"])
        ) >= 4

        if enough_peak and enough_floor:
            valid_probes[name] = probes[name]

    missing_nulls = [
        name for name in probes
        if name.startswith("NULL_") and name not in valid_probes
    ]

    if missing_nulls:
        raise RuntimeError(
            f"{len(missing_nulls)} null probes lack valid spectral windows. "
            "Increase base_fmax or decrease null_hi."
        )

    best = {
        name: {
            "p2f": 0.0,
            "peak_norm": 0.0,
            "floor_norm": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "harm": 0.0,
            "side": 0.0,
        }
        for name in valid_probes
    }

    window_function = get_window("hann", n)
    window_amplitude_scale = 0.5 * np.sum(window_function)

    candidate_bands = band_list(fs)

    for f_low, f_high in candidate_bands:
        analytic = band_analytic_v3(
            full_fft,
            full_frequency,
            f_low,
            f_high,
        )

        squared_envelope = np.abs(analytic) ** 2
        squared_envelope -= np.mean(squared_envelope)

        spectrum = (
            np.abs(rfft(squared_envelope * window_function))[frequency_keep] /
            (window_amplitude_scale + 1e-20)
        )

        background_mask = envelope_frequency > 5.0

        if np.count_nonzero(background_mask) < 10:
            continue

        global_background = np.median(spectrum[background_mask]) + 1e-20
        normalized_spectrum = spectrum / global_background

        for name, order in valid_probes.items():
            window = windows[name]
            peak_values = normalized_spectrum[window["peak"]]

            floor_values = np.concatenate(
                [
                    normalized_spectrum[window["side_left"]],
                    normalized_spectrum[window["side_right"]],
                ]
            )

            if peak_values.size == 0 or floor_values.size < 4:
                continue

            peak_norm = float(np.max(peak_values))
            floor_norm = float(np.median(floor_values) + 1e-20)
            score = peak_norm / floor_norm

            if score <= best[name]["p2f"]:
                continue

            harmonic_sum = 0.0

            for harmonic in range(2, n_harm + 1):
                harmonic_frequency = harmonic * order * fr

                if harmonic_frequency >= envelope_frequency[-1]:
                    continue

                harmonic_window = frequency_slice(
                    envelope_frequency,
                    harmonic_frequency - tolerance_hz,
                    harmonic_frequency + tolerance_hz,
                )

                if slice_length(harmonic_window) > 0:
                    harmonic_sum += float(
                        np.max(normalized_spectrum[harmonic_window])
                    )

            sideband_sum = 0.0

            for sign in (-1.0, 1.0):
                sideband_frequency = (order + sign) * fr

                if not (0 < sideband_frequency < envelope_frequency[-1]):
                    continue

                sideband_window = frequency_slice(
                    envelope_frequency,
                    sideband_frequency - tolerance_hz,
                    sideband_frequency + tolerance_hz,
                )

                if slice_length(sideband_window) > 0:
                    sideband_sum += float(
                        np.max(normalized_spectrum[sideband_window])
                    )

            best[name] = {
                "p2f": float(score),
                "peak_norm": peak_norm,
                "floor_norm": floor_norm,
                "lo": float(f_low),
                "hi": float(f_high),
                "harm": float(harmonic_sum),
                "side": float(sideband_sum),
            }

    null_scores_by_target = {}
    null_summary_by_target = {}

    for target_name in fault_orders:
        target_scores = np.asarray(
            [
                best[key]["p2f"]
                for key in null_keys_by_target[target_name]
            ],
            dtype=float,
        )

        if target_scores.size != n_null or np.any(target_scores <= 0):
            raise RuntimeError(
                f"Invalid {target_name} null result: expected {n_null} "
                f"positive scores, received "
                f"{np.count_nonzero(target_scores > 0)}."
            )

        null_scores_by_target[target_name] = target_scores
        null_summary_by_target[target_name] = {
            "median": float(np.median(target_scores)),
            "q95": float(np.quantile(target_scores, 0.95)),
            "q99": float(np.quantile(target_scores, 0.99)),
            "std": float(np.std(target_scores, ddof=1) + 1e-12),
            "mad": robust_mad(target_scores),
        }

    pooled_null_scores = np.concatenate(
        [null_scores_by_target[name] for name in fault_orders]
    )

    pooled_null_median = float(np.median(pooled_null_scores))
    pooled_null_q95 = float(np.quantile(pooled_null_scores, 0.95))
    pooled_null_q99 = float(np.quantile(pooled_null_scores, 0.99))
    pooled_null_std = float(np.std(pooled_null_scores, ddof=1) + 1e-12)
    pooled_null_mad = robust_mad(pooled_null_scores)

    feature = {
        "fr": float(fr),
        "df_env": df_resolution,
        "analysis_fmax": float(analysis_fmax),
        "tol_order": float(order_tolerance),
        "tol_hz": float(tolerance_hz),
        # Pooled values are metadata only. Each tested fault order uses its
        # own collision-matched null distribution below.
        "null_med": pooled_null_median,
        "null_q95": pooled_null_q95,
        "null_q99": pooled_null_q99,
        "null_std": pooled_null_std,
        "null_mad": pooled_null_mad,
        "n_null_used": int(n_null),
        "n_null_total": int(pooled_null_scores.size),
        "null_zeta_width": float(zeta_width),
        "n_orders": int(len(fault_orders)),
        "alpha": float(alpha),
    }

    raw_pvalues = []

    for name in fault_orders:
        raw_pvalues.append(
            empirical_upper_pvalue(
                best[name]["p2f"],
                null_scores_by_target[name],
            )
        )

    holm_pvalues = holm_adjust(raw_pvalues)
    bh_qvalues = benjamini_hochberg_adjust(raw_pvalues)

    fault_statistics = {}

    for index, name in enumerate(fault_orders):
        summary = null_summary_by_target[name]
        fault_statistics[name] = {
            "p_emp": float(raw_pvalues[index]),
            "p_holm": float(holm_pvalues[index]),
            "q_bh": float(bh_qvalues[index]),
            "null_median": summary["median"],
            "null_q95": summary["q95"],
            "null_q99": summary["q99"],
            "null_std": summary["std"],
            "null_mad": summary["mad"],
        }

    # Retain the interpretable legacy names while adding calibrated values.
    for name in kept_orders:
        score = best[name]["p2f"]
        peak_norm = best[name]["peak_norm"]
        sideband_sum = best[name]["side"]

        feature[f"{name}_p2f"] = float(score)
        feature[f"{name}_band"] = float(
            0.5 * (best[name]["lo"] + best[name]["hi"]) / 1000.0
        )
        feature[f"{name}_bandwidth"] = float(
            (best[name]["hi"] - best[name]["lo"]) / 1000.0
        )
        feature[f"{name}_harm"] = float(
            np.log1p(best[name]["harm"])
        )
        feature[f"{name}_sb"] = float(
            sideband_sum /
            (sideband_sum + peak_norm + 1.0)
        )

        if name in fault_statistics:
            stats = fault_statistics[name]
            feature[f"{name}_null_med"] = stats["null_median"]
            feature[f"{name}_null_q95"] = stats["null_q95"]
            feature[f"{name}_null_q99"] = stats["null_q99"]
            feature[f"{name}_null_std"] = stats["null_std"]
            feature[f"{name}_null_mad"] = stats["null_mad"]
            feature[f"{name}_z"] = float(
                score / (stats["null_median"] + 1e-12)
            )
            feature[f"{name}_zs"] = float(
                (score - stats["null_median"]) /
                stats["null_std"]
            )
            feature[f"{name}_rz"] = float(
                (score - stats["null_median"]) /
                stats["null_mad"]
            )
            feature[f"{name}_effect_log"] = float(
                np.log(
                    (score + 1e-12) /
                    (stats["null_median"] + 1e-12)
                )
            )
            feature[f"{name}_p_emp"] = stats["p_emp"]
            feature[f"{name}_p_holm"] = stats["p_holm"]
            feature[f"{name}_q_bh"] = stats["q_bh"]
            feature[f"{name}_evidence"] = float(
                -np.log10(max(stats["p_emp"], 1e-12))
            )
            feature[f"{name}_evidence_holm"] = float(
                -np.log10(max(stats["p_holm"], 1e-12))
            )
            feature[f"{name}_evidence_bh"] = float(
                -np.log10(max(stats["q_bh"], 1e-12))
            )
            feature[f"{name}_sig_raw"] = float(
                stats["p_emp"] <= alpha
            )
            feature[f"{name}_sig_holm"] = float(
                stats["p_holm"] <= alpha
            )
            feature[f"{name}_sig_bh"] = float(
                stats["q_bh"] <= alpha
            )

            # Conservative compatibility feature.
            feature[f"{name}_sig"] = feature[f"{name}_sig_holm"]

    # Pairwise relative-order evidence.
    for index, first in enumerate(fault_orders):
        for second in fault_orders[index + 1:]:
            feature[f"LR_{first}_{second}"] = float(
                np.log(
                    (best[first]["p2f"] + 1e-12) /
                    (best[second]["p2f"] + 1e-12)
                )
            )
            feature[f"CLR_{first}_{second}"] = float(
                np.log(
                    (feature[f"{first}_z"] + 1e-12) /
                    (feature[f"{second}_z"] + 1e-12)
                )
            )

    if "FTF" in kept_orders:
        for name in fault_orders:
            feature[f"LR_{name}_FTF"] = float(
                np.log(
                    (best[name]["p2f"] + 1e-12) /
                    (best["FTF"]["p2f"] + 1e-12)
                )
            )

    if fault_orders:
        total_score = sum(best[name]["p2f"] for name in fault_orders)
        total_score += 1e-12

        for name in fault_orders:
            feature[f"R_{name}"] = float(
                best[name]["p2f"] / total_score
            )

        total_calibrated_score = sum(
            feature[f"{name}_z"] for name in fault_orders
        ) + 1e-12

        for name in fault_orders:
            feature[f"CR_{name}"] = float(
                feature[f"{name}_z"] / total_calibrated_score
            )

        ranked_raw = sorted(
            fault_orders,
            key=lambda name: best[name]["p2f"],
            reverse=True,
        )
        ranked = sorted(
            fault_orders,
            key=lambda name: feature[f"{name}_z"],
            reverse=True,
        )

        winner = ranked[0]
        feature["argmax_raw_order"] = ranked_raw[0]
        feature["argmax_order"] = winner
        feature["max_z"] = float(feature[f"{winner}_z"])
        feature["min_p_emp"] = float(
            min(fault_statistics[name]["p_emp"] for name in fault_orders)
        )
        feature["min_p_holm"] = float(
            min(fault_statistics[name]["p_holm"] for name in fault_orders)
        )
        feature["margin"] = (
            float(
                np.log(
                    (feature[f"{ranked[0]}_z"] + 1e-12) /
                    (feature[f"{ranked[1]}_z"] + 1e-12)
                )
            )
            if len(ranked) > 1
            else 0.0
        )
        feature["margin_raw"] = (
            float(
                np.log(
                    (best[ranked_raw[0]]["p2f"] + 1e-12) /
                    (best[ranked_raw[1]]["p2f"] + 1e-12)
                )
            )
            if len(ranked_raw) > 1
            else 0.0
        )
        feature["n_sig_raw"] = float(sum(
            fault_statistics[name]["p_emp"] <= alpha
            for name in fault_orders
        ))
        feature["n_sig_holm"] = float(sum(
            fault_statistics[name]["p_holm"] <= alpha
            for name in fault_orders
        ))
        feature["n_sig_bh"] = float(sum(
            fault_statistics[name]["q_bh"] <= alpha
            for name in fault_orders
        ))

    # Conventional descriptors retained for ablation experiments.
    feature["sig_kurt"] = float(kurtosis(x, fisher=True))
    feature["sig_skew"] = float(skew(x))
    feature["crest"] = float(
        np.max(np.abs(x)) /
        (np.sqrt(np.mean(x ** 2)) + 1e-12)
    )
    feature["rms_log"] = float(
        np.log(np.sqrt(np.mean(x ** 2)) + 1e-12)
    )

    detail = {
        "frequency": envelope_frequency,
        "orders_kept": kept_orders,
        "orders_dropped": dropped_orders,
        "fault_orders": fault_orders,
        "null_orders_by_target": null_orders_by_target,
        "null_scores_by_target": null_scores_by_target,
        "null_metadata_by_target": null_metadata_by_target,
        "null_summary_by_target": null_summary_by_target,
        "pooled_null_scores": pooled_null_scores,
        "best": best,
        "fault_statistics": fault_statistics,
        "candidate_bands": candidate_bands,
        "analysis_fmax": analysis_fmax,
        "df_resolution": df_resolution,
        "tolerance_hz": tolerance_hz,
    }

    if return_detail:
        return feature, detail

    return feature


# -------------------------------------------------------------------------
# 6. Exclude calibration metadata from trained model inputs
# -------------------------------------------------------------------------

META_COLS.update({
    "df_env",
    "analysis_fmax",
    "tol_order",
    "tol_hz",
    "null_q95",
    "null_q99",
    "null_mad",
    "n_null_total",
    "null_zeta_width",
    "alpha",
    "min_p_emp",
    "min_p_holm",
})

for _order_name in ("BPFO", "BPFI", "BSF"):
    for _suffix in (
        "null_med",
        "null_q95",
        "null_q99",
        "null_std",
        "null_mad",
    ):
        META_COLS.add(f"{_order_name}_{_suffix}")


# -------------------------------------------------------------------------
# 7. Smoke test: one Paderborn record per class
# -------------------------------------------------------------------------

required_names = [
    "char_orders",
    "fast_len_leq",
    "band_list",
    "pb_scan",
    "pb_records",
    "MSSP_DIRS",
    "MSSP_COLORS",
    "save_publication_figure",
]

missing_names = [
    name for name in required_names
    if name not in globals()
]

if missing_names:
    raise RuntimeError(
        "Cell 7 requires earlier notebook definitions: "
        + ", ".join(missing_names)
    )

print("Search- and collision-matched empirical-null smoke test")
print("n_null=199; empirical p_min=1/200=0.005\n")

smoke_manifest = pb_scan(conditions=("N15_M07_F10",))
smoke_rows = []
smoke_details = {}

for label in ("N", "IR", "OR"):
    selected = smoke_manifest[
        smoke_manifest["label"] == label
    ].head(1)

    records = pb_records(selected, verbose=False)

    if not records:
        raise RuntimeError(f"No smoke-test record found for {label}")

    record = records[0]
    segment_length = fast_len_leq(
        min(
            int(100 * record.fs / record.fr),
            len(record.sig),
        )
    )

    start_time = time.time()

    features_v3, details_v3 = extract_features_v3(
        sig=record.sig[:segment_length],
        fs=record.fs,
        fr=record.fr,
        bearing_key=record.meta["bearing_key"],
        seed_str=f"CELL7::{record.meta['file']}",
        n_null=199,
        return_detail=True,
    )

    elapsed = time.time() - start_time

    row = {
        "label": label,
        "unit": record.meta["unit"],
        "file": record.meta["file"],
        "seconds": round(elapsed, 3),
        "df_env": features_v3["df_env"],
        "analysis_fmax": features_v3["analysis_fmax"],
        "n_null": features_v3["n_null_used"],
        "null_med": features_v3["null_med"],
        "null_q95": features_v3["null_q95"],
        "argmax_order": features_v3["argmax_order"],
        "max_z": features_v3["max_z"],
        "n_sig_raw": features_v3["n_sig_raw"],
        "n_sig_holm": features_v3["n_sig_holm"],
    }

    for order_name in ("BPFO", "BPFI"):
        row[f"{order_name}_p2f"] = features_v3[f"{order_name}_p2f"]
        row[f"{order_name}_null_med"] = features_v3[
            f"{order_name}_null_med"
        ]
        row[f"{order_name}_z"] = features_v3[f"{order_name}_z"]
        row[f"{order_name}_p_emp"] = features_v3[f"{order_name}_p_emp"]
        row[f"{order_name}_p_holm"] = features_v3[
            f"{order_name}_p_holm"
        ]
        row[f"{order_name}_q_bh"] = features_v3[f"{order_name}_q_bh"]
        row[f"{order_name}_harm"] = features_v3[f"{order_name}_harm"]

    smoke_rows.append(row)
    smoke_details[label] = details_v3

    assert features_v3["n_null_used"] == 199
    assert 0.005 <= features_v3["BPFO_p_emp"] <= 1.0
    assert 0.005 <= features_v3["BPFI_p_emp"] <= 1.0
    assert "BSF" in details_v3["orders_dropped"]
    assert all(
        len(details_v3["null_scores_by_target"][name]) == 199
        for name in ("BPFO", "BPFI")
    )

    print(
        f"{label:<2}  unit={record.meta['unit']:<4}  "
        f"time={elapsed:5.2f}s  "
        f"kept={sorted(details_v3['orders_kept'])}  "
        f"dropped={sorted(details_v3['orders_dropped'])}"
    )

    for order_name in ("BPFO", "BPFI"):
        print(
            f"    {order_name}: "
            f"p2f={features_v3[order_name + '_p2f']:8.3f}  "
            f"z={features_v3[order_name + '_z']:6.3f}  "
            f"p_emp={features_v3[order_name + '_p_emp']:.3f}  "
            f"p_holm={features_v3[order_name + '_p_holm']:.3f}  "
            f"q_BH={features_v3[order_name + '_q_bh']:.3f}  "
            f"harm={features_v3[order_name + '_harm']:.3f}"
        )

    print(
        f"    winner={features_v3['argmax_order']}  "
        f"max_z={features_v3['max_z']:.3f}  "
        f"n_sig_raw={features_v3['n_sig_raw']:.0f}  "
        f"n_sig_holm={features_v3['n_sig_holm']:.0f}\n"
    )

MSSP_SMOKE_V3 = pd.DataFrame(smoke_rows)
MSSP_SMOKE_DETAILS_V3 = smoke_details

MSSP_SMOKE_V3.to_csv(
    Path(MSSP_DIRS["features"]) / "cell7_smoke_test_v3.csv",
    index=False,
)

# -------------------------------------------------------------------------
# 8. First supplementary diagnostic figure
# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True)

for axis, label in zip(axes, ("N", "IR", "OR")):
    details = smoke_details[label]

    for order_name in ("BPFO", "BPFI"):
        null_scores_sorted = np.sort(
            details["null_scores_by_target"][order_name]
        )
        empirical_cdf = (
            np.arange(1, len(null_scores_sorted) + 1) /
            len(null_scores_sorted)
        )
        score = details["best"][order_name]["p2f"]
        pvalue = details["fault_statistics"][order_name]["p_emp"]

        axis.step(
            null_scores_sorted,
            empirical_cdf,
            where="post",
            color=MSSP_COLORS[order_name],
            linewidth=1.4,
            alpha=0.55,
            label=f"{order_name}-matched null",
        )

        axis.axvline(
            score,
            color=MSSP_COLORS[order_name],
            linestyle="--",
            linewidth=1.6,
            label=f"{order_name}, p={pvalue:.3f}",
        )

    axis.set_xscale("log")
    axis.set_ylim(0, 1.02)
    axis.set_title(f"True class: {label}")
    axis.set_xlabel("Optimized peak-to-floor score")
    axis.legend(loc="lower right", fontsize=7.5)

axes[0].set_ylabel("Empirical cumulative probability")

fig.suptitle(
    "Search- and collision-matched null versus physical fault orders",
    y=1.03,
)
fig.tight_layout()

CELL7_FIGURE_PATHS = save_publication_figure(
    fig,
    "S001_search_collision_matched_null_smoke_test",
    supplementary=True,
)

plt.show()
plt.close(fig)

print("\nSmoke-test table:")
print(MSSP_SMOKE_V3.round(4).to_string(index=False))

print("\nSaved:")
print(
    Path(MSSP_DIRS["features"]) /
    "cell7_smoke_test_v3.csv"
)
print(CELL7_FIGURE_PATHS)

print("\nCELL 7 COMPLETE")
print(
    "Next: Cell 8 will validate false-positive calibration and detection "
    "power using controlled synthetic bearing-impact signals."
)
