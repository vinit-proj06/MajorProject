"""Shared, patient-isolated model training and evaluation orchestration."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, save_json
from src.model_data import build_cache, open_split
from src.model_evaluation import choose_threshold, write_evaluation

SPLITS = ("train", "validation", "test")


def load_model_config(path=None):
    path = Path(path or PROJECT_ROOT / "model_config.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("processed_dir", "cache_dir", "models_dir", "reports_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    if config["threshold_rule"] != "validation_max_f1":
        raise ValueError("Supported threshold_rule: validation_max_f1")
    if config["class_weight"] not in ("balanced", "none"):
        raise ValueError("class_weight must be balanced or none")
    for key in ("n_jobs", "batch_rows", "calibration_bins"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    destinations = [Path(config[k]) for k in ("cache_dir", "models_dir", "reports_dir")]
    source = Path(config["processed_dir"])
    for destination in destinations:
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("Model/cache/report directories must be separate from processed inputs")
    for left, destination in enumerate(destinations):
        for other in destinations[left + 1:]:
            if destination == other or destination in other.parents or other in destination.parents:
                raise ValueError("Model/cache/report directories must not overlap")
    forest, boosted = config["random_forest"], config["xgboost"]
    for key in ("n_estimators", "trees_per_batch", "max_depth", "max_leaf_nodes", "min_samples_leaf", "max_samples"):
        if not isinstance(forest[key], int) or forest[key] < 1:
            raise ValueError(f"random_forest.{key} must be a positive integer")
    for key in ("num_boost_round", "early_stopping_rounds", "max_depth", "max_bin", "log_every"):
        if not isinstance(boosted[key], int) or boosted[key] < 1:
            raise ValueError(f"xgboost.{key} must be a positive integer")
    if any(not 0 < boosted[k] <= 1 for k in ("learning_rate", "subsample", "colsample_bytree")):
        raise ValueError("XGBoost learning rate and sampling fractions must be in (0,1]")
    return config


def versions():
    values = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "not installed"
    return values


def close_arrays(split):
    for array in split.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def training_weight(y, config):
    positives = int(np.count_nonzero(y))
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training requires both positive and negative labels")
    ratio = negatives / positives if config["class_weight"] == "balanced" else 1.0
    return ratio, {"rows": len(y), "positive_rows": positives, "negative_rows": negatives,
                   "positive_class_weight": ratio, "fit_split": "train"}


def predict_in_batches(model, X, predictor, batch_rows):
    probabilities = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_rows):
        end = min(start + batch_rows, len(X))
        values = np.asarray(predictor(model, X[start:end]), dtype=np.float64)
        if values.shape != (end - start,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Estimator returned invalid binary probabilities")
        probabilities[start:end] = values
    return probabilities


def save_predictions(split, probabilities, threshold, path, batch_rows):
    import gzip
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for start in range(0, len(probabilities), batch_rows):
            end = min(start + batch_rows, len(probabilities))
            frame = pd.DataFrame({
                "patient_id": split["patient_id"][start:end], "dataset": split["dataset"][start:end],
                "ICULOS": split["ICULOS"][start:end], "SepsisLabel": split["y"][start:end],
                "probability": probabilities[start:end],
                "predicted_SepsisLabel": (probabilities[start:end] >= threshold).astype(np.uint8),
            })
            frame.to_csv(handle, index=False, header=start == 0)


def _prepare_destination(path, overwrite):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Results already exist: {path}. Use --overwrite to retain a backup and rebuild.")
        backup = path.with_name(path.name + ".backup-" + uuid.uuid4().hex[:10])
        path.rename(backup)
        logging.info("Previous results retained at %s", backup)
    path.mkdir(parents=True, exist_ok=True)


def run_training(model_name, trainer, predictor, saver, config, overwrite=False, rebuild_cache=False):
    """Train once; select threshold using validation; evaluate test only afterward.

    trainer(train, validation, features, config, model_dir) -> (estimator, details)
    predictor(estimator, X_batch) -> positive-class probabilities
    saver(estimator, model_dir) -> path of persisted estimator
    """
    started = time.perf_counter()
    report_dir = Path(config["reports_dir"]) / model_name
    model_dir = Path(config["models_dir"]) / model_name
    # Check both paths before moving either prior result.
    if not overwrite:
        for path in (report_dir, model_dir):
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"Existing {model_name} artifacts at {path}; use --overwrite")
    _prepare_destination(report_dir, overwrite)
    _prepare_destination(model_dir, overwrite)
    file_handler = logging.FileHandler(report_dir / "training.log", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), file_handler
    ], force=True)
    logging.info("Starting %s; seed=%s, threads=%s", model_name, config["seed"], config["n_jobs"])
    save_json(model_dir / "config_resolved.json", config)
    opened = []
    try:
        cache = build_cache(Path(config["processed_dir"]), Path(config["cache_dir"]), batch_rows=config["batch_rows"], rebuild=rebuild_cache)
        features = cache["features"]
        save_json(model_dir / "feature_columns.json", features)
        train = open_split(Path(config["cache_dir"]), "train")
        validation = open_split(Path(config["cache_dir"]), "validation")
        opened.extend([train, validation])
        _, weight_info = training_weight(train["y"], config)
        if set(np.unique(validation["y"])) != {0, 1}:
            raise ValueError("Validation requires both labels for threshold selection")
        logging.info("Fitting %s on all %s training rows, %s predictors", model_name, len(train["y"]), len(features))
        fitted_at = time.perf_counter()
        model, training_details = trainer(train, validation, features, config, model_dir)
        fit_seconds = time.perf_counter() - fitted_at
        model_path = saver(model, model_dir)
        # The estimators must not retain the input matrix. Release its mapping before scoring.
        close_arrays(train)
        opened.remove(train)
        del train
        gc.collect()
        logging.info("Predicting validation patients and selecting threshold")
        validation_probabilities = predict_in_batches(model, validation["X"], predictor, config["batch_rows"])
        threshold_info = choose_threshold(validation["y"], validation_probabilities)
        threshold = float(threshold_info["threshold"])
        save_json(model_dir / "threshold.json", threshold_info)
        logging.info("Frozen validation-selected threshold %.8f; validation F1=%.5f", threshold, threshold_info["validation_f1"])
        val_metrics = write_evaluation(validation["y"], validation_probabilities, threshold, report_dir, "validation", model_name, config["calibration_bins"])
        save_predictions(validation, validation_probabilities, threshold, report_dir / "validation_predictions.csv.gz", config["batch_rows"])
        close_arrays(validation)
        opened.remove(validation)
        del validation, validation_probabilities
        gc.collect()
        # No test scores or probabilities were supplied to fitting, stopping, or threshold selection.
        logging.info("Evaluating held-out test patients with frozen model and threshold")
        test = open_split(Path(config["cache_dir"]), "test")
        opened.append(test)
        test_probabilities = predict_in_batches(model, test["X"], predictor, config["batch_rows"])
        test_metrics = write_evaluation(test["y"], test_probabilities, threshold, report_dir, "test", model_name, config["calibration_bins"])
        save_predictions(test, test_probabilities, threshold, report_dir / "test_predictions.csv.gz", config["batch_rows"])
        result = {
            "status": "complete", "model": model_name, "model_file": str(model_path), "report_dir": str(report_dir),
            "seed": config["seed"], "feature_count": len(features), "cache_fingerprint": cache["fingerprint"],
            "feature_order_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
            "splits": cache["splits"], "training_class_weight": weight_info, "training_details": training_details,
            "threshold_selection": threshold_info, "fit_seconds": round(fit_seconds, 2),
            "elapsed_seconds": round(time.perf_counter() - started, 2), "versions": versions(),
            "validation": val_metrics, "test": test_metrics,
            "protocol": {
                "evaluation_unit": "hourly record; patients remain disjoint across train/validation/test",
                "test_role": "final evaluation only; never used for training, stopping or threshold selection",
                "validation_role": "threshold selection and XGBoost early stopping; development metrics are selection-biased",
                "calibration": "Assessment of raw model scores; no post-hoc probability calibrator was fitted",
                "class_imbalance": "Training-only class weights; original validation/test prevalence retained",
                "limitations": ["Hourly observations within each patient are correlated; these are point estimates, not independent-patient confidence intervals.",
                                "Hourly discrimination, including later positive hours, does not by itself measure early alert timing or clinical safety.",
                                "Class weighting can distort predicted probabilities; consult reliability curves alongside Brier/log loss and ECE.",
                                "This is a fixed research baseline, not an exhaustive hyperparameter search or external validation."],
            },
        }
        save_json(report_dir / "results.json", result)
        save_json(model_dir / "run_manifest.json", {k: v for k, v in result.items() if k not in {"validation", "test"}})
        display = lambda value: "N/A" if value is None else f"{value:.5f}"
        logging.info("COMPLETE %s: test AUROC=%s AUPRC(AP)=%s F1=%s elapsed=%.1fs", model_name, display(test_metrics["auroc"]), display(test_metrics["auprc"]), display(test_metrics["f1"]), result["elapsed_seconds"])
        return result
    except Exception as exc:
        save_json(report_dir / "RUN_FAILED.json", {"status": "failed", "model": model_name, "error": f"{type(exc).__name__}: {exc}"})
        logging.exception("%s training/evaluation failed", model_name)
        raise
    finally:
        for split in opened:
            close_arrays(split)
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def script_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Model configuration JSON (default: model_config.json)")
    parser.add_argument("--overwrite", action="store_true", help="Retain previous model/reports as backups and rerun")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild disk arrays if source data changed; retain prior cache")
    return parser.parse_args()
