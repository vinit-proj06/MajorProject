"""Patient-local, strictly causal time-series features.

This module never estimates population statistics. Missing values left here are
handled by the training-fitted transformation in ``preprocess.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


METADATA_COLUMNS = frozenset({"patient_id", "dataset", "SepsisLabel"})


def _predictors(schema: Sequence[str]) -> list[str]:
    return [column for column in schema if column not in METADATA_COLUMNS]


def _configured_columns(schema: Sequence[str], config: dict, key: str) -> list[str]:
    # Preserve configuration order, but do not generate a requested feature twice.
    return list(dict.fromkeys(column for column in config[key] if column in schema))


def _windows(config: dict) -> list[int]:
    windows = list(dict.fromkeys(config["rolling_windows"]))
    if not windows or any(isinstance(w, bool) or int(w) != w or w < 1 for w in windows):
        raise ValueError("rolling_windows must contain positive integer hours")
    return [int(w) for w in windows]


def feature_metadata(schema: Sequence[str], config: dict) -> dict[str, dict]:
    """Describe predictors and how their remaining NaNs must be filled.

    ``training_median`` always refers to the *raw training observations* of
    ``fallback_source``. ``zero`` is only a derived-change fallback. ``none``
    means the feature is already complete and must not be fitted or imputed.
    """
    result: dict[str, dict] = {}

    def add(name: str, description: str, kind: str, source: str,
            fallback: str, fallback_source: str | None = None) -> None:
        result[name] = {
            "description": description,
            "kind": kind,
            "source": source,
            "fallback": fallback,
            "fallback_source": fallback_source,
        }

    predictors = _predictors(schema)
    forward_fill = _configured_columns(predictors, config, "forward_fill_features")
    limit = config["forward_fill_limit_hours"]
    for column in predictors:
        detail = "Original hourly predictor."
        if column in forward_fill:
            detail += f" Forward-filled from this patient's past for at most {limit} hours."
        if column == "ICULOS":
            add(column, detail + " Required observed time coordinate.", "time", column, "none")
        else:
            add(column, detail, "current", column, "training_median", column)
            add(f"{column}__missing", "1 if the original value is missing, otherwise 0.",
                "missing_indicator", column, "none")

    for column in _configured_columns(predictors, config, "rolling_features"):
        add(f"{column}__previous", "Value at the preceding hour after bounded causal forward filling.",
            "previous", column, "training_median", column)
        add(f"{column}__delta", "Current minus preceding-hour value after bounded causal forward filling.",
            "delta", column, "zero")
        add(f"{column}__hours_since_observed", "Hours since last original observation; -1 if never observed.",
            "recency", column, "none")
        for window in _windows(config):
            for statistic in ("mean", "min", "max", "std", "slope"):
                detail = (f"{statistic} of original observations in (ICULOS-{window}, ICULOS]; "
                          "includes current hour, excludes future hours and forward-filled values.")
                if statistic == "std":
                    detail += " Population standard deviation; requires at least 2 observations."
                if statistic == "slope":
                    detail += " OLS slope per hour; requires at least 2 observations."
                fallback = "zero" if statistic in ("std", "slope") else "training_median"
                add(f"{column}__{statistic}_{window}h", detail, statistic, column,
                    fallback, column if fallback == "training_median" else None)
    return result


def _rolling_statistics(values: np.ndarray, hours: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Vectorized bounded historical windows; no Python loop over hourly rows.

    On hourly data the window view has at most ``window`` historical positions.
    The time mask also supports gaps and non-unit starting ICU hours. Views avoid
    copying the padded data; working arrays are bounded by one patient's length.
    """
    left = np.searchsorted(hours, hours - window, side="right")
    width = int(np.max(np.arange(len(hours)) + 1 - left))
    padded = np.pad(values, ((width - 1, 0), (0, 0)), constant_values=np.nan)
    history = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
    padded_hours = np.pad(hours, (width - 1, 0), constant_values=np.nan)
    historical_hours = np.lib.stride_tricks.sliding_window_view(padded_hours, width)
    x_raw = historical_hours - hours[:, None]
    observed = (~np.isnan(history)) & (x_raw[:, None, :] > -window)
    count = observed.sum(axis=2)
    y = np.where(observed, history, 0.0)
    sum_y = y.sum(axis=2)
    mean = np.full(values.shape, np.nan)
    np.divide(sum_y, count, out=mean, where=count > 0)
    minimum = np.where(observed, history, np.inf).min(axis=2)
    maximum = np.where(observed, history, -np.inf).max(axis=2)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    squared_residuals = np.where(observed, history - mean[:, :, None], 0.0) ** 2
    variance = np.full(values.shape, np.nan)
    np.divide(squared_residuals.sum(axis=2), count, out=variance, where=count >= 2)
    # Relative times in [-window, 0] avoid cancellation at large ICULOS values.
    x = np.where(observed, x_raw[:, None, :], 0.0)
    sum_x = x.sum(axis=2)
    sum_xx = (x * x).sum(axis=2)
    sum_xy = (x * y).sum(axis=2)
    denominator = count * sum_xx - sum_x * sum_x
    slope = np.full(values.shape, np.nan)
    np.divide(count * sum_xy - sum_x * sum_y, denominator, out=slope,
              where=(count >= 2) & (denominator > 0))
    return {"mean": mean, "min": minimum, "max": maximum,
            "std": np.sqrt(variance), "slope": slope}


def engineer_patient(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create features for exactly one patient, in increasing ICULOS order.

    The function leaves its input untouched, preserves row/index order, and is
    prefix-invariant: future measurements and targets cannot change past output.
    It also handles gaps using elapsed hours instead of treating every row as an
    hour; a missing preceding hour gives a missing previous/delta feature.
    """
    if frame.empty:
        raise ValueError("Cannot engineer an empty patient")
    if "ICULOS" not in frame:
        raise ValueError("ICULOS is required for hourly features")
    if "patient_id" in frame and frame["patient_id"].nunique(dropna=False) != 1:
        raise ValueError("engineer_patient accepts exactly one patient")
    predictors = _predictors(frame.columns)
    source_positions = {column: index for index, column in enumerate(frame.columns)}
    positions = {column: index for index, column in enumerate(predictors)}
    raw = frame.iloc[:, [source_positions[column] for column in predictors]].to_numpy(dtype=float)
    hours = raw[:, positions["ICULOS"]]
    if not np.isfinite(hours).all() or (np.diff(hours) <= 0).any():
        raise ValueError("ICULOS must be finite and strictly increasing")
    limit = config["forward_fill_limit_hours"]
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("forward_fill_limit_hours must be finite and nonnegative")

    result = frame.copy()
    additions: dict[str, np.ndarray] = {}
    missing = np.isnan(raw)
    indicator_positions = [index for index, column in enumerate(predictors) if column != "ICULOS"]
    indicators = pd.DataFrame(missing[:, indicator_positions].astype(np.uint8), index=frame.index,
                              columns=[f"{predictors[index]}__missing" for index in indicator_positions])

    rolling_features = _configured_columns(predictors, config, "rolling_features")
    forward_fill = _configured_columns(predictors, config, "forward_fill_features")
    combined_features = list(dict.fromkeys(forward_fill + rolling_features))
    if not combined_features:
        return pd.concat([result, indicators], axis=1)
    combined_positions = {column: index for index, column in enumerate(combined_features)}
    combined_source_positions = [positions[column] for column in combined_features]
    observed = ~missing[:, combined_source_positions]
    last_rows = np.maximum.accumulate(np.where(observed, np.arange(len(frame))[:, None], -1), axis=0)
    safe_rows = np.maximum(last_rows, 0)
    elapsed = np.where(last_rows >= 0, hours[:, None] - hours[safe_rows], -1.0)
    candidates = raw[safe_rows, np.asarray(combined_source_positions)[None, :]]
    candidates[(elapsed < 0) | (elapsed > limit)] = np.nan
    if forward_fill:
        result[forward_fill] = candidates[:, [combined_positions[column] for column in forward_fill]]

    consecutive = np.r_[False, np.isclose(np.diff(hours), 1.0)]
    for column in rolling_features:
        index = combined_positions[column]
        current = candidates[:, index] if column in forward_fill else raw[:, positions[column]]
        previous = np.r_[np.nan, current[:-1]]
        previous[~consecutive] = np.nan
        additions[f"{column}__previous"] = previous
        additions[f"{column}__delta"] = current - previous
        additions[f"{column}__hours_since_observed"] = elapsed[:, index]

    if rolling_features:
        values = raw[:, [positions[column] for column in rolling_features]]
        for window in _windows(config):
            stats = _rolling_statistics(values, hours, window)
            for index, column in enumerate(rolling_features):
                for statistic, values_array in stats.items():
                    additions[f"{column}__{statistic}_{window}h"] = values_array[:, index]

    parts = [result, indicators]
    if additions:
        parts.append(pd.DataFrame(np.column_stack(list(additions.values())),
                                  columns=list(additions), index=frame.index))
    return pd.concat(parts, axis=1)
