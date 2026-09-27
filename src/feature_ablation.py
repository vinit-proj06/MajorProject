"""Train the authorized XGBoost feature ablations with the original protocol.

The existing cache, patient splits, preprocessing, baseline model and baseline
reports are read-only inputs. Only bounded row batches are copied when selecting
predictor columns. These follow-up comparisons are explicitly exploratory.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np

from src.common import save_json
from src.model_data import close_arrays, open_split
from src.model_evaluation import choose_threshold, write_evaluation
from src.model_training import predict_in_batches, save_predictions, training_weight, versions
from train_xgboost import fit_xgboost, predict_xgboost, save_xgboost


EXCLUSIONS = {
    "A": ["ICULOS"],
    "B": ["HospAdmTime", "HospAdmTime__missing"],
    "C": ["ICULOS", "HospAdmTime", "HospAdmTime__missing"],
}
LOGGER = logging.getLogger(__name__)


class ColumnSubset:
    """Lazy column selection, allocating only explicitly requested row slices.

    ``fit_xgboost`` and ``predict_in_batches`` already request bounded slices.
    Metadata remains outside this adapter, including ICULOS when excluded as a
    predictor. Implicit conversion of the complete matrix is prohibited.
    """

    def __init__(self, X, indices):
        if not hasattr(X, "shape") or len(X.shape) != 2:
            raise ValueError("ColumnSubset requires a two-dimensional source matrix")
        values = np.asarray(indices)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "iu":
            raise ValueError("indices must be a nonempty one-dimensional integer sequence")
        if (np.any(values < 0) or np.any(values >= X.shape[1])
                or len(np.unique(values)) != len(values)):
            raise ValueError("indices must be unique valid source-column indexes")
        self._source = X
        self.indices = values.astype(np.intp, copy=True)
        self.shape = (int(X.shape[0]), len(self.indices))
        self.ndim = 2
        self.dtype = np.dtype(np.float32)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row_key):
        if isinstance(row_key, (bool, np.bool_)) or not isinstance(row_key, (slice, int, np.integer)):
            raise TypeError("ColumnSubset supports integer or slice row selection only")
        rows = self._source[row_key]
        # Take columns only AFTER selecting the bounded rows. Reversing this
        # order would materialize the entire training matrix on every call.
        selected = np.take(rows, self.indices, axis=-1)
        return np.ascontiguousarray(selected, dtype=np.float32)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Do not materialize the full ColumnSubset; request bounded row slices")


def _validate_destinations(model_dir, report_dir, config):
    destinations = (model_dir, report_dir)
    if (model_dir == report_dir or model_dir in report_dir.parents
            or report_dir in model_dir.parents):
        raise ValueError("Ablation model and report directories must be separate")
    protected = [Path(config[key]).resolve() for key in ("processed_dir", "cache_dir")]
    for key in ("models_dir", "reports_dir"):
        parent = Path(config[key]).resolve()
        protected.extend(parent / name for name in ("xgboost", "random_forest"))
    for destination in destinations:
        if destination == destination.parent:
            raise ValueError("Ablation output cannot be a filesystem root")
        if destination.exists():
            raise FileExistsError(f"Ablation output already exists; refusing to overwrite: {destination}")
        for source in protected:
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError(f"Ablation output overlaps a protected input: {source}")


def _subset_split(raw, indices, expected_features, expected_rows):
    if raw["X"].shape != (expected_rows, expected_features) or len(raw["y"]) != expected_rows:
        raise ValueError("Source split shape differs from the supplied validated cache")
    result = dict(raw)
    result["X"] = ColumnSubset(raw["X"], indices)
    return result


def run_ablation_training(experiment, config, cache, model_dir: Path, report_dir: Path) -> dict:
    """Train A/B/C once using baseline settings and an independent validation threshold.

    ``config`` is the resolved original model configuration; ``cache`` is the
    original cache manifest already validated by the caller. Output directories
    must not exist. No source arrays, baseline artifacts or configuration files
    are rewritten, and test rows are opened only after validation selection.
    """
    started = time.perf_counter()
    if experiment not in EXCLUSIONS:
        raise ValueError(f"Unknown ablation {experiment!r}; expected A, B or C")
    if config.get("threshold_rule") != "validation_max_f1":
        raise ValueError("Ablation must retain the original validation_max_f1 threshold rule")
    original_features = list(cache["features"])
    if not original_features or len(original_features) != len(set(original_features)):
        raise ValueError("Validated cache must contain unique ordered feature names")
    excluded = list(EXCLUSIONS[experiment])
    if not set(excluded).issubset(original_features):
        raise ValueError(f"Required ablation features are absent from the cache: {excluded}")
    indices = [index for index, name in enumerate(original_features) if name not in excluded]
    features = [original_features[index] for index in indices]
    if not features:
        raise ValueError("Ablation cannot remove every predictor")
    model_dir, report_dir = Path(model_dir).resolve(), Path(report_dir).resolve()
    _validate_destinations(model_dir, report_dir, config)
    model_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(report_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    file_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    opened = {}
    try:
        save_json(model_dir / "config_resolved.json", config)
        save_json(model_dir / "feature_columns.json", features)
        save_json(model_dir / "ablation_specification.json", {
            "experiment": experiment, "excluded_features": excluded,
            "source_feature_count": len(original_features), "feature_count": len(features),
            "source_column_indices": indices, "cache_fingerprint": cache["fingerprint"],
            "exploratory": True,
            "metadata_preserved": ["patient_id", "dataset", "ICULOS", "SepsisLabel"],
            "configuration": "Original baseline training settings; only the stated predictor columns are excluded.",
        })
        LOGGER.info("Starting ablation %s: exclude=%s; predictors=%s; seed=%s; threads=%s",
                    experiment, excluded, len(features), config["seed"], config["n_jobs"])
        opened["train"] = open_split(Path(config["cache_dir"]), "train")
        opened["validation"] = open_split(Path(config["cache_dir"]), "validation")
        train = _subset_split(opened["train"], indices, len(original_features), cache["splits"]["train"]["rows"])
        validation = _subset_split(opened["validation"], indices, len(original_features), cache["splits"]["validation"]["rows"])
        _, weight_info = training_weight(train["y"], config)
        if set(np.unique(validation["y"])) != {0, 1}:
            raise ValueError("Validation requires both labels for threshold selection")
        fitted_at = time.perf_counter()
        model, training_details = fit_xgboost(train, validation, features, config, model_dir)
        fit_seconds = time.perf_counter() - fitted_at
        model_path = save_xgboost(model, model_dir)
        del train
        close_arrays(opened.pop("train"))
        gc.collect()

        LOGGER.info("Ablation %s: validation predictions and validation-only threshold selection", experiment)
        validation_probabilities = predict_in_batches(model, validation["X"], predict_xgboost, config["batch_rows"])
        threshold_info = choose_threshold(validation["y"], validation_probabilities)
        threshold = float(threshold_info["threshold"])
        save_json(model_dir / "threshold.json", threshold_info)
        LOGGER.info("Ablation %s: frozen threshold %.8f; validation F1=%.5f", experiment,
                    threshold, threshold_info["validation_f1"])
        val_metrics = write_evaluation(validation["y"], validation_probabilities, threshold,
                                       report_dir, "validation", "xgboost", config["calibration_bins"])
        save_predictions(validation, validation_probabilities, threshold,
                         report_dir / "validation_predictions.csv.gz", config["batch_rows"])
        del validation, validation_probabilities
        close_arrays(opened.pop("validation"))
        gc.collect()

        # The shared test cohort has already informed this ablation proposal.
        # It is used only for exploratory evaluation, never for fitting or selection.
        LOGGER.info("Ablation %s: exploratory test evaluation with frozen model and threshold", experiment)
        opened["test"] = open_split(Path(config["cache_dir"]), "test")
        test = _subset_split(opened["test"], indices, len(original_features), cache["splits"]["test"]["rows"])
        test_probabilities = predict_in_batches(model, test["X"], predict_xgboost, config["batch_rows"])
        test_metrics = write_evaluation(test["y"], test_probabilities, threshold,
                                        report_dir, "test", "xgboost", config["calibration_bins"])
        save_predictions(test, test_probabilities, threshold,
                         report_dir / "test_predictions.csv.gz", config["batch_rows"])
        result = {
            "status": "complete", "model": "xgboost", "experiment": experiment,
            "excluded_features": excluded, "exploratory": True,
            "model_file": str(model_path), "report_dir": str(report_dir),
            "seed": config["seed"], "feature_count": len(features),
            "source_feature_count": len(original_features), "cache_fingerprint": cache["fingerprint"],
            "feature_order_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
            "source_feature_order_sha256": hashlib.sha256(json.dumps(original_features).encode()).hexdigest(),
            "splits": cache["splits"], "training_class_weight": weight_info,
            "training_details": training_details, "threshold_selection": threshold_info,
            "fit_seconds": round(fit_seconds, 2),
            "elapsed_seconds": round(time.perf_counter() - started, 2), "versions": versions(),
            "validation": val_metrics, "test": test_metrics,
            "protocol": {
                "evaluation_unit": "Hourly record; original patients remain disjoint across train/validation/test.",
                "test_role": "Exploratory evaluation only; never used for training, early stopping or threshold selection. A fresh holdout is needed for final confirmation.",
                "validation_role": "Original validation patients select XGBoost early stopping and the experiment's maximum-F1 threshold; development metrics are selection-biased.",
                "feature_ablation": "Only the named predictor columns are omitted. Original row order, labels, patient IDs, dataset source and ICULOS metadata are preserved.",
                "training": "The original fit_xgboost implementation and unchanged baseline hyperparameters, seed, training-derived quantiles, class weights and early-stopping rule are reused.",
                "calibration": "Assessment of raw class-weighted model probabilities; no post-hoc calibrator is fitted.",
                "class_imbalance": "Training-only class weights; original validation/test prevalence retained.",
                "memory": "Lazy ColumnSubset selects columns after bounded row slicing; no complete feature-subset matrix is materialized.",
                "limitations": [
                    "The ablation was motivated by findings on an already examined test cohort; comparisons are exploratory, not fresh independent validation.",
                    "Correlated physiological or missingness features can retain timing/source information after explicit time predictors are removed.",
                    "Removing ICULOS as a predictor does not remove it as the historical time coordinate used by existing causal preprocessing and evaluation.",
                    "Hourly observations within patients are correlated; point estimates do not establish clinical benefit or safety.",
                ],
            },
        }
        save_json(report_dir / "results.json", result)
        save_json(model_dir / "run_manifest.json", {key: value for key, value in result.items()
                                                   if key not in {"validation", "test"}})
        LOGGER.info("COMPLETE ablation %s: features=%s; test AUROC=%s; AUPRC=%s; F1=%s; elapsed=%.1fs",
                    experiment, len(features), test_metrics["auroc"], test_metrics["auprc"],
                    test_metrics["f1"], result["elapsed_seconds"])
        return result
    except Exception as error:
        save_json(report_dir / "RUN_FAILED.json", {"status": "failed", "model": "xgboost",
                  "experiment": experiment, "error": f"{type(error).__name__}: {error}"})
        LOGGER.exception("Ablation %s failed; diagnostics retained at %s", experiment, report_dir)
        raise
    finally:
        for split in opened.values():
            close_arrays(split)
        root_logger.removeHandler(file_handler)
        file_handler.close()
