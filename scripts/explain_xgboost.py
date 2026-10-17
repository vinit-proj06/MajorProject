"""Exact full-test SHAP explanations for the saved XGBoost model; never trains."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import sys
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.common import PROJECT_ROOT, save_json
from src.model_data import build_cache, open_split, close_arrays
from src.model_training import load_model_config
from src.shap_explain import compute_contributions, explain_patient_row
from src.shap_plots import plot_global_summary, plot_importance, plot_elbow

LOGGER = logging.getLogger(__name__)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_config(path=None):
    path = Path(path or PROJECT_ROOT / "shap_config.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_config", "output_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    for key in ("expected_features", "batch_rows", "n_jobs", "summary_bins"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["summary_bins"] < 2:
        raise ValueError("summary_bins must be at least 2")
    if config["split"] != "test":
        raise ValueError("This analysis is configured for the saved test set")
    if not isinstance(config["patient_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", config["patient_id"]):
        raise ValueError("patient_id must contain only letters, digits, underscores or hyphens")
    if isinstance(config["hour"], bool) or not isinstance(config["hour"], (float, int)) or not np.isfinite(config["hour"]) or config["hour"] <= 0:
        raise ValueError("hour must be a finite positive ICULOS value")
    return config


def write_ranking(path, features, mean_abs, mean_signed):
    """Keep every feature, including zero-importance features; ties use saved order."""
    if len(features) != len(set(features)) or len(features) == 0:
        raise ValueError("Feature names must be nonempty and unique")
    mean_abs, mean_signed = np.asarray(mean_abs), np.asarray(mean_signed)
    if mean_abs.shape != (len(features),) or mean_signed.shape != mean_abs.shape or not np.isfinite(mean_abs).all() or not np.isfinite(mean_signed).all() or (mean_abs < 0).any():
        raise ValueError("Invalid global importance arrays")
    order = np.argsort(-mean_abs, kind="stable")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature", "mean_abs_shap", "mean_signed_shap"])
        writer.writerows((rank + 1, features[index], float(mean_abs[index]), float(mean_signed[index])) for rank, index in enumerate(order))
    return order


def verify_saved_batch(frame, arrays, start, stop, probabilities, threshold):
    required = {"patient_id", "dataset", "ICULOS", "SepsisLabel", "probability", "predicted_SepsisLabel"}
    if set(frame.columns) != required or frame.columns.duplicated().any() or len(frame) != stop - start or frame.isna().any().any():
        raise ValueError("Saved prediction schema, coverage or values are invalid")
    for column, source in (("patient_id", "patient_id"), ("dataset", "dataset"), ("ICULOS", "ICULOS"), ("SepsisLabel", "y")):
        if not np.array_equal(frame[column].to_numpy(), arrays[source][start:stop]):
            raise ValueError(f"Saved {column} differs from cached test data at row {start}")
    old = frame["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(old).all() or np.any((old < 0) | (old > 1)):
        raise ValueError("Saved probabilities are invalid")
    if not np.allclose(old, probabilities, atol=1e-7, rtol=1e-6):
        raise ValueError("Loaded model predictions differ from saved test predictions")
    if not np.array_equal(frame["predicted_SepsisLabel"].to_numpy(), (old >= threshold).astype(np.uint8)):
        raise ValueError("Saved predicted labels do not match the frozen threshold")
    if not np.array_equal(old >= threshold, probabilities >= threshold):
        raise ValueError("Current predictions change a frozen threshold decision")
    return float(np.max(np.abs(old - probabilities)))


def run(config):
    started = time.perf_counter()
    model_config = load_model_config(config["model_config"])
    output = Path(config["output_dir"]).resolve()
    for key in ("processed_dir", "cache_dir", "models_dir", "reports_dir"):
        source = Path(model_config[key]).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"SHAP output must be separate from {key}")
    if output == output.parent:
        raise ValueError("Output cannot be a filesystem root")
    model_dir = Path(model_config["models_dir"]) / "xgboost"
    report_dir = Path(model_config["reports_dir"]) / "xgboost"
    model_path = model_dir / "xgboost.ubj"
    feature_path = model_dir / "feature_columns.json"
    result_path = report_dir / "results.json"
    threshold_path = model_dir / "threshold.json"
    prediction_path = report_dir / "test_predictions.csv.gz"
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    trained = json.loads(result_path.read_text(encoding="utf-8"))
    threshold_info = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = float(threshold_info["threshold"])
    if trained["status"] != "complete" or trained["model"] != "xgboost" or threshold_info != trained["threshold_selection"]:
        raise ValueError("Saved XGBoost run or threshold is inconsistent")
    if len(features) != config["expected_features"] or len(features) != trained["feature_count"]:
        raise ValueError("Unexpected predictor count; no features may be silently omitted")
    if hashlib.sha256(json.dumps(features).encode()).hexdigest() != trained["feature_order_sha256"]:
        raise ValueError("Saved predictor order changed")
    cache = build_cache(Path(model_config["processed_dir"]), Path(model_config["cache_dir"]), batch_rows=config["batch_rows"])
    if cache["fingerprint"] != trained["cache_fingerprint"] or features != cache["features"] or cache["splits"]["test"] != trained["splits"]["test"]:
        raise ValueError("Model and validated test cache differ")
    source_files = [model_path, feature_path, threshold_path, result_path, prediction_path,
                    Path(model_config["cache_dir"]) / "cache_manifest.json"]
    source_files += list((Path(model_config["cache_dir"]) / "test").glob("*.npy"))
    hashes = {str(path.resolve()): file_hash(path) for path in source_files}
    model = xgb.Booster()
    model.load_model(model_path)
    model.set_param({"nthread": config["n_jobs"], "device": "cpu"})
    if model.feature_names != features or model.num_features() != len(features):
        raise ValueError("Loaded model's named features do not match saved metadata")
    if model.num_boosted_rounds() != trained["training_details"]["trees_saved"] or model.best_iteration != trained["training_details"]["best_iteration"]:
        raise ValueError("Loaded tree count/early-stopping iteration differs from saved run")
    arrays, mapped = {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".inprogress-" + uuid.uuid4().hex[:10])
    stage.mkdir()
    try:
        arrays = open_split(Path(model_config["cache_dir"]), "test")
        n_rows, n_features = arrays["X"].shape
        matches = np.flatnonzero((arrays["patient_id"] == config["patient_id"]) & (arrays["ICULOS"] == config["hour"]))
        if matches.size != 1:
            raise ValueError("Patient/hour must identify exactly one saved test row")
        patient_index = int(matches[0])
        save_json(stage / "config_resolved.json", config)
        save_json(stage / "feature_columns.json", features)
        for key, shape in (("shap_values", (n_rows, n_features)), ("base_values", (n_rows,)), ("raw_margins", (n_rows,)), ("probabilities", (n_rows,))):
            mapped[key] = np.lib.format.open_memmap(stage / f"{key}.npy", mode="w+", dtype=np.float32, shape=shape)
        abs_sum = np.zeros(n_features, dtype=np.float64)
        signed_sum = np.zeros(n_features, dtype=np.float64)
        max_additivity_error = max_prediction_error = 0.0
        offset = positive_rows = 0
        LOGGER.info("Explaining ALL %s test rows × %s features using %s saved trees", n_rows, n_features, model.num_boosted_rounds())
        with pd.read_csv(prediction_path, chunksize=config["batch_rows"], dtype={"patient_id": str, "dataset": str}, float_precision="round_trip") as reader:
            for frame in reader:
                stop = offset + len(frame)
                if stop > n_rows:
                    raise ValueError("Saved predictions have extra rows")
                computed = compute_contributions(model, arrays["X"][offset:stop], features)
                max_prediction_error = max(max_prediction_error, verify_saved_batch(frame, arrays, offset, stop, computed["probabilities"], threshold))
                for key, source in (("shap_values", "values"), ("base_values", "base_values"), ("raw_margins", "margins"), ("probabilities", "probabilities")):
                    mapped[key][offset:stop] = computed[source]
                abs_sum += np.abs(computed["values"]).sum(axis=0, dtype=np.float64)
                signed_sum += computed["values"].sum(axis=0, dtype=np.float64)
                max_additivity_error = max(max_additivity_error, computed["max_additivity_error"])
                positive_rows += int(np.sum(arrays["y"][offset:stop]))
                offset = stop
                if offset % (config["batch_rows"] * 8) == 0 or offset == n_rows:
                    LOGGER.info("SHAP: %s/%s rows (%.1f%%)", offset, n_rows, 100 * offset / n_rows)
        if offset != n_rows or positive_rows != cache["splits"]["test"]["positive_rows"]:
            raise ValueError("Incomplete test coverage or target count mismatch")
        for array in mapped.values():
            array.flush()
        mean_abs, mean_signed = abs_sum / n_rows, signed_sum / n_rows
        order = write_ranking(stage / "feature_importance_full.csv", features, mean_abs, mean_signed)
        # A skinny row index preserves the exact mapping to the disk SHAP matrix.
        import gzip
        with gzip.open(stage / "test_row_index.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            for start in range(0, n_rows, config["batch_rows"]):
                stop = min(start + config["batch_rows"], n_rows)
                pd.DataFrame({"row_index": np.arange(start, stop), "patient_id": arrays["patient_id"][start:stop],
                              "dataset": arrays["dataset"][start:stop], "ICULOS": arrays["ICULOS"][start:stop],
                              "SepsisLabel": arrays["y"][start:stop]}).to_csv(handle, index=False, header=start == 0)
        LOGGER.info("Plotting complete ranking, uncut elbow curve and all-row SHAP distributions")
        plot_importance(stage / "feature_importance_full.png", features, mean_abs, order)
        plot_elbow(stage / "importance_elbow.png", features, mean_abs, order)
        summary_plot = plot_global_summary(stage / "shap_summary_all_features.png", features, arrays["X"], mapped["shap_values"], order, bins=config["summary_bins"])
        for extension in ("png", "pdf"):
            summary_plot[extension] = str(output / f"shap_summary_all_features.{extension}")
        save_json(stage / "summary_plot_metadata.json", summary_plot)
        patient_name = f"{config['patient_id']}_hour_{config['hour']:g}"
        patient_dir = stage / patient_name
        local = explain_patient_row(model, pd.Series(arrays["X"][patient_index].copy(), index=features), features,
                                    threshold, patient_dir, patient_id=config["patient_id"], hour=config["hour"])
        # The staging directory is renamed after validation; publish usable final paths.
        local["artifacts"] = {key: str(output / patient_name / Path(value).name) for key, value in local["artifacts"].items()}
        save_json(patient_dir / "local_explanation.json", local)
        # Local recomputation must explain the same row as the full-test matrix.
        local_check = compute_contributions(model, arrays["X"][patient_index:patient_index + 1], features)
        if not np.allclose(local_check["values"][0], mapped["shap_values"][patient_index], atol=1e-6, rtol=1e-5):
            raise ValueError("Individual and full-test SHAP explanations differ")
        save_json(patient_dir / "test_row_metadata.json", {"test_row_index": patient_index, "patient_id": config["patient_id"],
                  "ICULOS": config["hour"], "dataset": str(arrays["dataset"][patient_index]), "SepsisLabel": int(arrays["y"][patient_index]),
                  "feature_values_are": "causally processed/imputed model inputs, not necessarily raw measurements"})
        for source, digest in hashes.items():
            if file_hash(source) != digest:
                raise ValueError(f"Source changed during explanation: {source}")
        report = ["XGBOOST SHAP — COMPLETE TEST-SET FEATURE RANKING", "",
                  f"Rows: {n_rows}; patients: {cache['splits']['test']['patients']}; features: {n_features}; saved trees: {model.num_boosted_rounds()}",
                  "Mean absolute SHAP = sum of abs(feature contribution) over ALL test hours / number of test hours.",
                  "Units: raw model log-odds. Positive local values raise the score; negative values lower it.",
                  "Exact native XGBoost TreeSHAP; no approximate contributions, row sample, feature cutoff or retraining.",
                  "Distribution plot bins observations for display only; all rows and all features contribute.",
                  "Rank ties retain saved feature order; zero-importance features remain listed.", "",
                  f"{'Rank':>4}  {'Feature':42} {'Mean |SHAP|':>14} {'Mean signed SHAP':>18}"]
        report += [f"{rank + 1:4d}  {features[index]:42} {mean_abs[index]:14.9f} {mean_signed[index]:18.9f}" for rank, index in enumerate(order)]
        report += ["", f"Individual explanation: {patient_name}/local_waterfall.png and local_contributions.csv",
                   "SHAP explains this trained model; correlated measurements and their derived features can share attribution.",
                   "Test-set SHAP is descriptive. Feature selection or tuning based on it requires fresh held-out evaluation."]
        (stage / "feature_ranking_full.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        manifest = {"status": "complete", "backend": "XGBoost native exact TreeSHAP (pred_contribs=True, approx_contribs=False)",
                    "output_units": "raw binary-logistic margin (log-odds)", "reference": "saved tree training node cover; no new background dataset fitted",
                    "rows_explained": n_rows, "patients": cache["splits"]["test"]["patients"], "features_ranked": n_features,
                    "zero_importance_features": int(np.count_nonzero(mean_abs == 0)), "feature_cutoff": None, "row_subsampling": False,
                    "model_retrained": False, "model_sha256_before": hashes[str(model_path.resolve())], "model_sha256_after": file_hash(model_path),
                    "max_additivity_error": max_additivity_error, "max_saved_probability_difference": max_prediction_error,
                    "cache_fingerprint": cache["fingerprint"], "source_sha256": hashes, "config": config,
                    "versions": {name: importlib.metadata.version(name) for name in ("xgboost", "numpy", "pandas", "matplotlib")},
                    "local_patient": {"patient_id": config["patient_id"], "hour": config["hour"], "test_row_index": patient_index},
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "checks": ["all test prediction rows match patient/time/label source", "full feature order preserved", "all-row SHAP additivity checked",
                               "all-row probabilities match saved model results", "local/global SHAP agree", "source and saved model hashes unchanged"],
                    "references": ["https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict",
                                   "https://xgboost.readthedocs.io/en/stable/prediction.html"]}
        save_json(stage / "run_manifest.json", manifest)
    except Exception as error:
        save_json(stage / "RUN_FAILED.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        LOGGER.exception("SHAP failed; incomplete diagnostics retained at %s", stage)
        raise
    finally:
        close_arrays(mapped)
        close_arrays(arrays)
    backup = None
    if output.exists():
        backup = output.with_name(output.name + ".backup-" + uuid.uuid4().hex[:10])
        output.rename(backup)
    try:
        stage.rename(output)
    except Exception:
        if backup is not None and not output.exists():
            backup.rename(output)
        raise
    LOGGER.info("SHAP complete in %.1fs. Saved all %s features at %s", time.perf_counter() - started, len(features), output)
    print("\n".join(report))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="SHAP configuration JSON (default shap_config.json)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(parser.parse_args().config))
