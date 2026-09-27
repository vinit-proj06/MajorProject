"""Exact native TreeSHAP explanations of the frozen XGBoost model.

Contributions are additive in raw log-odds, not probability percentage points.
XGBoost uses the trained trees' stored node-cover statistics for its path-based
expectation. No reference dataset, held-out fit, model update or SHAP dependency
is introduced here. Callers should pass bounded row batches to the core API.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xgboost as xgb


ADDITIVITY_ATOL = 5e-5
ADDITIVITY_RTOL = 5e-5
PROBABILITY_ATOL = 2e-7
PROBABILITY_RTOL = 2e-6


def _feature_names(feature_names):
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature_names must be a sequence of unique named predictors")
    try:
        names = list(feature_names)
    except TypeError as error:
        raise ValueError("feature_names must be a sequence of unique named predictors") from error
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("feature_names must contain nonempty unique strings")
    for name in names:
        lowered = name.lower()
        if (lowered in {"patient_id", "dataset", "split", "n_rows", "positive_rows", "positive_patient"}
                or any(token in lowered for token in ("sepsislabel", "time_to_sepsis", "onset"))
                or re.search(r"(^|_)(target|label|future|lead)(_|$)", lowered)):
            raise ValueError(f"Target, metadata, or future-derived feature is prohibited: {name}")
    return names


def _frozen_model_info(model, names):
    if not isinstance(model, xgb.Booster):
        raise TypeError("model must be a loaded xgboost.Booster")
    if model.feature_names is None or list(model.feature_names) != names:
        raise ValueError("feature_names must match the saved model feature order exactly")
    if model.num_features() != len(names):
        raise ValueError("Feature count differs from the saved model")
    try:
        learner = json.loads(model.save_config())["learner"]
        objective = learner["objective"]["name"]
        train_objective = learner["learner_train_param"]["objective"]
        tree_type = learner["gradient_booster"]["name"]
        parameters = learner["learner_model_param"]
        classes = int(parameters["num_class"])
        targets = int(parameters.get("num_target", 1))
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Cannot validate the saved XGBoost model configuration") from error
    if objective != "binary:logistic" or train_objective != "binary:logistic":
        raise ValueError("SHAP explanations require a binary:logistic model")
    if tree_type != "gbtree" or classes != 0 or targets != 1:
        raise ValueError("SHAP explanations require a single-target binary gbtree model")
    saved_best = model.attr("best_iteration")
    if not isinstance(saved_best, str) or not re.fullmatch(r"\d+", saved_best):
        raise ValueError("Saved model must contain a valid best_iteration; refusing an implicit tree range")
    best_iteration = int(saved_best)
    rounds = int(model.num_boosted_rounds())
    if rounds <= 0 or not 0 <= best_iteration < rounds:
        raise ValueError("Saved best_iteration is outside the available boosted rounds")
    return {"best_iteration": best_iteration, "num_boosted_rounds": rounds,
            "iteration_range": [0, best_iteration + 1]}


def _matrix(X, names):
    if isinstance(X, pd.DataFrame) and list(X.columns) != names:
        raise ValueError("DataFrame columns must match feature_names in exactly the same order")
    try:
        raw = np.asarray(X)
    except (TypeError, ValueError) as error:
        raise ValueError("X must be a numeric two-dimensional feature matrix") from error
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != len(names):
        raise ValueError(f"X must have nonzero rows and exactly {len(names)} feature columns")
    if np.iscomplexobj(raw):
        raise ValueError("Complex feature values are not supported")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Feature values must be numeric and finite") from error
    if not np.isfinite(values).all():
        raise ValueError("Feature values must be finite, without missing values or float32 overflow")
    return values


def _sigmoid(margins):
    values = np.asarray(margins, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def compute_contributions(model, X, feature_names) -> dict:
    """Explain a bounded batch with exact native TreeSHAP at the saved best round.

    ``X`` is an ordered two-dimensional numerical array or DataFrame; its column
    order must match both ``feature_names`` and the persisted model. Outputs are
    float64 arrays with one row per observation. The model is never modified.
    """
    names = _feature_names(feature_names)
    info = _frozen_model_info(model, names)
    values = _matrix(X, names)
    data = xgb.DMatrix(values, feature_names=names, nthread=1)
    iteration_range = tuple(info["iteration_range"])
    common = {"iteration_range": iteration_range, "strict_shape": True,
              "validate_features": True, "training": False}
    contributions = np.asarray(model.predict(
        data, pred_contribs=True, approx_contribs=False, **common), dtype=np.float64)
    margins = np.asarray(model.predict(data, output_margin=True, **common), dtype=np.float64)
    probabilities = np.asarray(model.predict(data, **common), dtype=np.float64)
    n_rows, n_features = values.shape
    if contributions.shape != (n_rows, 1, n_features + 1):
        raise ValueError(f"Unexpected binary SHAP output shape: {contributions.shape}")
    if margins.shape != (n_rows, 1) or probabilities.shape != (n_rows, 1):
        raise ValueError("Unexpected raw-margin or probability output shape")
    if not all(np.isfinite(array).all() for array in (contributions, margins, probabilities)):
        raise ValueError("XGBoost returned nonfinite SHAP values, margins or probabilities")
    margins, probabilities = margins[:, 0], probabilities[:, 0]
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("XGBoost returned probabilities outside [0, 1]")
    shap_values = contributions[:, 0, :-1]
    base_values = contributions[:, 0, -1]
    reconstruction = base_values + shap_values.sum(axis=1, dtype=np.float64)
    max_error = float(np.max(np.abs(reconstruction - margins)))
    if not np.allclose(reconstruction, margins, rtol=ADDITIVITY_RTOL, atol=ADDITIVITY_ATOL):
        raise ValueError(f"TreeSHAP additivity check failed: maximum raw-margin error={max_error:.8g}")
    expected_probabilities = _sigmoid(margins)
    probability_error = float(np.max(np.abs(expected_probabilities - probabilities)))
    if not np.allclose(expected_probabilities, probabilities,
                       rtol=PROBABILITY_RTOL, atol=PROBABILITY_ATOL):
        raise ValueError(f"Sigmoid(raw margin) disagrees with probability: maximum error={probability_error:.8g}")
    return {"values": shap_values, "base_values": base_values, "margins": margins,
            "probabilities": probabilities, "max_additivity_error": max_error,
            "max_probability_error": probability_error, "features": names,
            "output_space": "raw_margin_log_odds", "method": "native_exact_tree_shap",
            "additivity_tolerance": {"atol": ADDITIVITY_ATOL, "rtol": ADDITIVITY_RTOL},
            "probability_tolerance": {"atol": PROBABILITY_ATOL, "rtol": PROBABILITY_RTOL}, **info}


def _named_row(feature_row, names):
    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError("feature_row DataFrame must contain exactly one row")
        row = feature_row.iloc[0]
    elif isinstance(feature_row, pd.Series):
        row = feature_row
    elif isinstance(feature_row, Mapping):
        row = pd.Series(dict(feature_row))
    else:
        raise TypeError("feature_row must be a named Series, a one-row DataFrame, or a mapping")
    if row.index.has_duplicates:
        raise ValueError("feature_row contains duplicate feature names")
    provided = set(row.index)
    if provided != set(names):
        missing = sorted(set(names) - provided)
        extra = sorted((repr(name) for name in provided - set(names)))
        raise ValueError(f"feature_row names must match saved predictors exactly: missing={missing}, extra={extra}")
    # Local named input may arrive in any order; explicitly restore the model order.
    return _matrix(row.reindex(names).to_numpy().reshape(1, -1), names)[0]


def explain_patient_row(model, feature_row, feature_names, threshold, output_dir,
                        patient_id=None, hour=None) -> dict:
    """Save every local feature contribution and an additive raw-log-odds plot.

    Input values must already have undergone the saved preprocessing pipeline.
    Raw patient metadata and labels must not be included as feature columns.
    """
    names = _feature_names(feature_names)
    row = _named_row(feature_row, names)
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError("threshold must be finite and in [0, 1]")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be finite and in [0, 1]") from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if patient_id is not None and (not isinstance(patient_id, str) or not patient_id):
        raise ValueError("patient_id must be a nonempty string when provided")
    if hour is not None:
        try:
            hour = float(hour)
        except (TypeError, ValueError) as error:
            raise ValueError("hour must be finite when provided") from error
        if not np.isfinite(hour):
            raise ValueError("hour must be finite when provided")
        if "ICULOS" in names and not np.isclose(hour, row[names.index("ICULOS")], atol=1e-6, rtol=0):
            raise ValueError("hour does not match the feature row's ICULOS")
    explained = compute_contributions(model, row.reshape(1, -1), names)
    contributions = explained["values"][0]
    base = float(explained["base_values"][0])
    margin = float(explained["margins"][0])
    probability = float(explained["probabilities"][0])
    order = np.argsort(-np.abs(contributions), kind="stable")
    records = [{"rank": rank + 1, "feature": names[index], "feature_value": float(row[index]),
                "shap_value": float(contributions[index]),
                "absolute_shap_value": float(abs(contributions[index])),
                "direction": "positive" if contributions[index] > 0 else "negative" if contributions[index] < 0 else "zero"}
               for rank, index in enumerate(order)]
    positive_sum = float(contributions[contributions > 0].sum(dtype=np.float64))
    negative_sum = float(contributions[contributions < 0].sum(dtype=np.float64))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output / "local_contributions.csv", output / "local_explanation.json"
    plot_path = output / "local_waterfall.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Import lazily so native contributions do not require plotting dependencies.
    from src.shap_plots import plot_local_waterfall
    plot_local_waterfall(plot_path, names, row, contributions, base, margin,
                         probability, threshold, patient_id, hour)
    artifacts = {"contributions_csv": str(csv_path), "explanation_json": str(json_path),
                 "waterfall_png": str(plot_path)}
    if plot_path.with_suffix(".pdf").is_file():
        artifacts["waterfall_pdf"] = str(plot_path.with_suffix(".pdf"))
    result = {
        "patient_id": patient_id, "hour": hour, "feature_count": len(names),
        "feature_names": names, "feature_contributions": records,
        "method": explained["method"], "output_space": explained["output_space"],
        "objective": "binary:logistic", "best_iteration": explained["best_iteration"],
        "num_boosted_rounds": explained["num_boosted_rounds"],
        "iteration_range": explained["iteration_range"],
        "base_value": base, "base_probability": float(_sigmoid(np.array([base]))[0]),
        "margin": margin, "probability": probability, "threshold": threshold,
        "predicted_positive": bool(probability >= threshold),
        "positive_shap_sum": positive_sum, "negative_shap_sum": negative_sum,
        "sum_shap_values": float(contributions.sum(dtype=np.float64)),
        "zero_shap_count": int(np.count_nonzero(contributions == 0)),
        "reconstructed_margin": float(base + contributions.sum(dtype=np.float64)),
        "max_additivity_error": explained["max_additivity_error"],
        "max_probability_error": explained["max_probability_error"],
        "additivity_tolerance": explained["additivity_tolerance"],
        "probability_tolerance": explained["probability_tolerance"],
        "artifacts": artifacts,
        "definitions": {
            "base_value": "Native TreeSHAP expected raw output using the trained trees' stored node-cover statistics; not cohort prevalence.",
            "contribution": "Positive values raise the raw log-odds relative to the baseline; negative values lower them. Values are not probability percentage points.",
            "additivity": "base_value + sum(all feature SHAP values) = raw margin within floating-point tolerance; sigmoid(raw margin) = model probability.",
            "background": "Training-derived node cover stored in the existing model; no held-out reference dataset or refitting.",
            "feature_values": "Processed model inputs represented as float32; these may be imputed or engineered rather than raw measurements.",
            "interpretation": "Model associations under the path-based TreeSHAP expectation, not causal effects or evidence of clinical safety.",
            "probability": "The saved class-weighted model's uncalibrated score; no probability recalibration is performed.",
        },
    }
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
