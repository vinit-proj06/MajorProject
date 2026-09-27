"""Exact, complete test-set TreeSHAP for the frozen timing ablation C model.

The original 187-column cache stays read-only. The saved model's 184 predictors
are selected in bounded batches and copied to a disk array for column-wise
plots. ICULOS remains in the row index, never in the model input. No estimator,
threshold, preprocessing statistic or explanatory background is fitted here.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import xgboost as xgb

from explain_xgboost import file_hash, verify_saved_batch, write_ranking
from src.common import save_json
from src.model_data import ARRAYS, close_arrays, open_split
from src.shap_explain import (
    ADDITIVITY_ATOL, ADDITIVITY_RTOL, compute_contributions,
)
from src.shap_plots import plot_elbow, plot_global_summary, plot_importance


LOGGER = logging.getLogger(__name__)
EXCLUDED_FEATURES = ("ICULOS", "HospAdmTime", "HospAdmTime__missing")
SOURCE_FEATURE_COUNT = 187
MODEL_FEATURE_COUNT = 184
SUMMARY_BINS = 160


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_destination(output, model_dir, report_dir, cache_dir):
    if output == output.parent:
        raise ValueError("SHAP output cannot be a filesystem root")
    for source in (model_dir, cache_dir):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("SHAP output must be separate from model and cache directories")
    if output == report_dir or output in report_dir.parents:
        raise ValueError("SHAP output cannot replace the model report directory or an ancestor")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"SHAP output must be new or empty: {output}")


def _validate_metadata(features, trained, threshold_info, cache):
    if (not isinstance(features, list) or len(features) != MODEL_FEATURE_COUNT
            or any(not isinstance(name, str) or not name for name in features)
            or len(set(features)) != len(features)):
        raise ValueError("Experiment C must retain exactly 184 unique named predictors")
    source_features = cache["features"]
    if (not isinstance(source_features, list) or len(source_features) != SOURCE_FEATURE_COUNT
            or any(not isinstance(name, str) or not name for name in source_features)
            or len(set(source_features)) != len(source_features)):
        raise ValueError("The original cache must contain exactly 187 unique named predictors")
    expected = [name for name in source_features if name not in EXCLUDED_FEATURES]
    if features != expected:
        raise ValueError("Saved features must preserve original cache order with exactly the three C exclusions")
    excluded = trained.get("excluded_features")
    if (trained.get("status") != "complete" or trained.get("model") != "xgboost"
            or trained.get("experiment") != "C" or trained.get("exploratory") is not True
            or not isinstance(excluded, list) or len(excluded) != len(EXCLUDED_FEATURES)
            or set(excluded) != set(EXCLUDED_FEATURES)):
        raise ValueError("Expected the completed exploratory XGBoost Experiment C run")
    if (trained.get("feature_count") != len(features)
            or trained.get("feature_order_sha256") != hashlib.sha256(json.dumps(features).encode()).hexdigest()):
        raise ValueError("Saved predictor count or order disagrees with training results")
    if (cache["fingerprint"] != trained.get("cache_fingerprint")
            or cache["splits"] != trained.get("splits")):
        raise ValueError("Saved model results and original patient-disjoint cache disagree")
    if threshold_info != trained.get("threshold_selection"):
        raise ValueError("Frozen threshold metadata differs from the saved training run")
    raw_threshold = threshold_info.get("threshold")
    if isinstance(raw_threshold, bool):
        raise ValueError("Frozen threshold must be finite and in [0, 1]")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen threshold must be finite and in [0, 1]") from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frozen threshold must be finite and in [0, 1]")
    return threshold, [source_features.index(name) for name in features]


def write_experiment_shap(model_dir, report_dir, cache_dir, output_dir,
                          batch_rows=4096, n_jobs=2) -> dict:
    """Explain every test row and all 184 predictors of saved Experiment C.

    Inputs follow the existing model/report/cache artifact contracts. Every
    saved test row must match the original cache's patient, dataset, hour and
    label in order; recomputed probabilities and threshold decisions must match
    the frozen prediction file. Exact native TreeSHAP checks every row's margin
    reconstruction. All source files are SHA256 checked before and after.

    Results are staged beside ``output_dir`` and published only on success.
    Existing nonempty output is rejected. Failed staging directories retain a
    RUN_FAILED.json diagnostic and are never silently published as complete.
    """
    started = time.perf_counter()
    for name, value in (("batch_rows", batch_rows), ("n_jobs", n_jobs)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    model_dir, report_dir, cache_dir, output = map(
        lambda path: Path(path).resolve(), (model_dir, report_dir, cache_dir, output_dir))
    _validate_destination(output, model_dir, report_dir, cache_dir)
    for source in (model_dir, report_dir, cache_dir):
        if not source.is_dir():
            raise FileNotFoundError(f"Missing SHAP input directory: {source}")
    model_path = model_dir / "xgboost.ubj"
    feature_path = model_dir / "feature_columns.json"
    threshold_path = model_dir / "threshold.json"
    result_path = report_dir / "results.json"
    prediction_path = report_dir / "test_predictions.csv.gz"
    cache_path = cache_dir / "cache_manifest.json"
    source_paths = [model_path, feature_path, threshold_path, result_path, prediction_path, cache_path]
    source_paths += [cache_dir / "test" / f"{name}.npy" for name in ARRAYS]
    # Hash before reading metadata/model so the final guard also catches changes
    # during the initial load, not merely during explanation batches.
    hashes = {str(path): file_hash(path) for path in source_paths}
    features, trained = _json(feature_path), _json(result_path)
    cache, threshold_info = _json(cache_path), _json(threshold_path)
    threshold, indexes = _validate_metadata(features, trained, threshold_info, cache)
    model = xgb.Booster()
    model.load_model(model_path)
    model.set_param({"nthread": n_jobs, "device": "cpu"})
    if model.feature_names != features or model.num_features() != len(features):
        raise ValueError("Loaded model feature order differs from the saved 184 predictors")
    details = trained["training_details"]
    if (model.num_boosted_rounds() != details["trees_saved"]
            or model.best_iteration != details["best_iteration"]):
        raise ValueError("Loaded tree count or best iteration differs from the saved run")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".inprogress-" + uuid.uuid4().hex[:10])
    stage.mkdir()
    arrays, mapped = {}, {}
    try:
        arrays = open_split(cache_dir, "test")
        n_rows = len(arrays["y"])
        n_features = len(features)
        if (n_rows <= 0 or arrays["X"].shape != (n_rows, SOURCE_FEATURE_COUNT)
                or n_rows != cache["splits"]["test"]["rows"]):
            raise ValueError("Original test cache shape or row count is invalid")
        save_json(stage / "feature_columns.json", features)
        shapes = {
            "test_feature_values": (n_rows, n_features), "shap_values": (n_rows, n_features),
            "base_values": (n_rows,), "raw_margins": (n_rows,), "probabilities": (n_rows,),
        }
        for key, shape in shapes.items():
            mapped[key] = np.lib.format.open_memmap(
                stage / f"{key}.npy", mode="w+", dtype=np.float32, shape=shape)
        abs_sum, signed_sum = np.zeros(n_features), np.zeros(n_features)
        offset = positive_rows = 0
        max_additivity_error = max_prediction_error = max_sigmoid_error = 0.0
        LOGGER.info("Explaining Experiment C: ALL %s test rows x %s features", n_rows, n_features)
        with pd.read_csv(prediction_path, chunksize=batch_rows,
                         dtype={"patient_id": str, "dataset": str},
                         float_precision="round_trip") as reader, gzip.open(
                             stage / "test_row_index.csv.gz", "wt", encoding="utf-8", newline="") as row_writer:
            for frame in reader:
                stop = offset + len(frame)
                if stop > n_rows:
                    raise ValueError("Saved test predictions contain extra rows")
                # Slice rows first: fancy column indexing must never copy the
                # full test matrix into RAM. Only this bounded batch is copied.
                values = np.ascontiguousarray(arrays["X"][offset:stop, :][:, indexes], dtype=np.float32)
                computed = compute_contributions(model, values, features)
                max_prediction_error = max(max_prediction_error, verify_saved_batch(
                    frame, arrays, offset, stop, computed["probabilities"], threshold))
                mapped["test_feature_values"][offset:stop] = values
                for key, source in (("shap_values", "values"), ("base_values", "base_values"),
                                    ("raw_margins", "margins"), ("probabilities", "probabilities")):
                    mapped[key][offset:stop] = computed[source]
                # Check the actual persisted precision as well as the core's
                # native contributions, not just a representative subset.
                reconstructed = np.asarray(mapped["base_values"][offset:stop], dtype=np.float64) + np.sum(
                    mapped["shap_values"][offset:stop], axis=1, dtype=np.float64)
                margins = np.asarray(mapped["raw_margins"][offset:stop], dtype=np.float64)
                error = float(np.max(np.abs(reconstructed - margins)))
                if not np.allclose(reconstructed, margins, atol=ADDITIVITY_ATOL, rtol=ADDITIVITY_RTOL):
                    raise ValueError("Persisted SHAP values fail raw-margin additivity")
                max_additivity_error = max(max_additivity_error, error, computed["max_additivity_error"])
                max_sigmoid_error = max(max_sigmoid_error, computed["max_probability_error"])
                abs_sum += np.abs(computed["values"]).sum(axis=0, dtype=np.float64)
                signed_sum += computed["values"].sum(axis=0, dtype=np.float64)
                labels = arrays["y"][offset:stop]
                if not np.isin(labels, (0, 1)).all():
                    raise ValueError("Cached test labels must be binary")
                positive_rows += int(labels.sum())
                pd.DataFrame({
                    "row_index": np.arange(offset, stop), "patient_id": arrays["patient_id"][offset:stop],
                    "dataset": arrays["dataset"][offset:stop], "ICULOS": arrays["ICULOS"][offset:stop],
                    "SepsisLabel": labels,
                }).to_csv(row_writer, index=False, header=offset == 0)
                offset = stop
                if offset % (batch_rows * 8) == 0 or offset == n_rows:
                    LOGGER.info("Experiment C SHAP: %s/%s rows (%.1f%%)", offset, n_rows, 100 * offset / n_rows)
        if offset != n_rows or positive_rows != cache["splits"]["test"]["positive_rows"]:
            raise ValueError("Incomplete test prediction coverage or target count mismatch")
        for array in mapped.values():
            array.flush()
        # Release source and dirty output mappings before rendering large plots.
        close_arrays(arrays)
        arrays = {}
        close_arrays(mapped)
        mapped = {}
        for key in ("test_feature_values", "shap_values"):
            mapped[key] = np.load(stage / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        mean_abs, mean_signed = abs_sum / n_rows, signed_sum / n_rows
        order = write_ranking(stage / "feature_importance_full.csv", features, mean_abs, mean_signed)
        LOGGER.info("Plotting all 184 features with all %s test rows; no cutoff", n_rows)
        importance = plot_importance(stage / "feature_importance_full.png", features, mean_abs, order)
        elbow = plot_elbow(stage / "importance_elbow.png", features, mean_abs, order)
        summary = plot_global_summary(stage / "shap_summary_all_features.png", features,
                                      mapped["test_feature_values"], mapped["shap_values"], order, bins=SUMMARY_BINS)
        if (summary["sample_count"] != n_rows or summary["feature_count"] != n_features
                or summary["represented_values_per_feature"] != [n_rows] * n_features
                or summary["feature_order"] != order.tolist()):
            raise ValueError("Global SHAP plot did not retain every test row and model feature")
        for metadata, stem in ((summary, "shap_summary_all_features"),
                               (importance, "feature_importance_full"), (elbow, "importance_elbow")):
            for suffix in ("png", "pdf"):
                artifact = stage / f"{stem}.{suffix}"
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"Missing SHAP plot artifact: {artifact}")
                metadata[suffix] = str(output / artifact.name)
        save_json(stage / "summary_plot_metadata.json", summary)
        save_json(stage / "ranking_plot_metadata.json", {"importance": importance, "elbow": elbow})
        for source, expected_hash in hashes.items():
            if file_hash(source) != expected_hash:
                raise ValueError(f"SHAP input changed during explanation: {source}")
        report = [
            "EXPERIMENT C XGBOOST SHAP — ALL TEST ROWS AND ALL 184 FEATURES", "",
            f"Rows: {n_rows}; patients: {cache['splits']['test']['patients']}; features: {n_features}",
            "Excluded model inputs: ICULOS, HospAdmTime, HospAdmTime__missing.",
            "ICULOS is retained only in test_row_index.csv.gz as the original hourly coordinate.",
            "Exact native TreeSHAP at the saved best iteration. Units: raw model log-odds.",
            "Every test hour has equal weight in mean absolute and mean signed SHAP.",
            "All features are ranked, including zeros; ties preserve saved feature order.",
            "The summary bins every observation for display; there is no row sample or feature cutoff.",
            "Exploratory timing ablation after baseline test review; not a new untouched test evaluation.",
            "SHAP explains model associations, not causal effects or clinical safety.", "",
            f"{'Rank':>4}  {'Feature':42} {'Mean |SHAP|':>14} {'Mean signed SHAP':>18}",
        ]
        report += [f"{rank + 1:4d}  {features[index]:42} {mean_abs[index]:14.9f} {mean_signed[index]:18.9f}"
                   for rank, index in enumerate(order)]
        (stage / "feature_ranking_full.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        config = {"model_dir": str(model_dir), "report_dir": str(report_dir), "cache_dir": str(cache_dir),
                  "output_dir": str(output), "batch_rows": batch_rows, "n_jobs": n_jobs, "summary_bins": SUMMARY_BINS}
        artifacts = {name: str(output / name) for name in (
            "feature_columns.json", "test_feature_values.npy", "shap_values.npy", "base_values.npy",
            "raw_margins.npy", "probabilities.npy", "test_row_index.csv.gz", "feature_importance_full.csv",
            "feature_ranking_full.txt", "summary_plot_metadata.json", "ranking_plot_metadata.json",
            "feature_importance_full.png", "feature_importance_full.pdf", "importance_elbow.png",
            "importance_elbow.pdf", "shap_summary_all_features.png", "shap_summary_all_features.pdf")}
        manifest = {
            "status": "complete", "experiment": "C", "exploratory": True,
            "backend": "XGBoost native exact TreeSHAP (pred_contribs=True, approx_contribs=False)",
            "output_units": "raw binary-logistic margin (log-odds)",
            "reference": "saved tree training node cover; no new background dataset fitted",
            "rows_explained": n_rows, "patients": cache["splits"]["test"]["patients"],
            "positive_rows": positive_rows, "features_ranked": n_features,
            "original_cache_feature_count": SOURCE_FEATURE_COUNT, "feature_names": features,
            "source_column_indexes": indexes, "excluded_features": list(EXCLUDED_FEATURES),
            "zero_importance_features": int(np.count_nonzero(mean_abs == 0)),
            "feature_cutoff": None, "row_subsampling": False, "model_retrained": False,
            "best_iteration": model.best_iteration, "trees_saved": model.num_boosted_rounds(),
            "iteration_range": [0, model.best_iteration + 1], "threshold": threshold,
            "model_sha256_before": hashes[str(model_path)], "model_sha256_after": file_hash(model_path),
            "max_additivity_error": max_additivity_error, "max_sigmoid_probability_error": max_sigmoid_error,
            "max_saved_probability_difference": max_prediction_error,
            "additivity_tolerance": {"atol": ADDITIVITY_ATOL, "rtol": ADDITIVITY_RTOL},
            "saved_probability_tolerance": {"atol": 1e-7, "rtol": 1e-6},
            "cache_fingerprint": cache["fingerprint"], "source_sha256": hashes,
            "config": config, "artifacts": artifacts,
            "versions": {name: importlib.metadata.version(name) for name in ("xgboost", "numpy", "pandas", "matplotlib")},
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "checks": ["all saved test row IDs/datasets/hours/labels match the original cache in order",
                       "all 184 saved model predictors preserve original cache order",
                       "all-row native and persisted SHAP additivity checked",
                       "all-row probabilities and threshold decisions match frozen test results",
                       "all source/model/test-cache SHA256 values unchanged",
                       "all test observations represented for every feature in summary density"],
            "limitations": ["Post-review exploratory ablation; the reused test set is not a fresh untouched holdout.",
                            "Mean importance weights patient-hours equally, so longer records contribute more observations.",
                            "Correlated and engineered predictors may share attribution; SHAP is not a causal effect.",
                            "Class-weighted model probabilities are uncalibrated; no recalibration was performed."],
        }
        save_json(stage / "run_manifest.json", manifest)
    except Exception as error:
        save_json(stage / "RUN_FAILED.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        LOGGER.exception("Experiment C SHAP failed; diagnostic staging directory: %s", stage)
        raise
    finally:
        close_arrays(mapped)
        close_arrays(arrays)
    # Only an empty destination can be removed. A concurrently created result
    # is never overwritten; publication uses one directory rename.
    _validate_destination(output, model_dir, report_dir, cache_dir)
    if output.exists():
        output.rmdir()
    stage.rename(output)
    LOGGER.info("Experiment C SHAP complete in %.1fs: %s", time.perf_counter() - started, output)
    return manifest
