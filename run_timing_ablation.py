"""Controlled exploratory XGBoost timing-feature ablations, with frozen splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from evaluate_early_detection import iter_prediction_patients, load_evaluation_config, write_csv
from src.common import PROJECT_ROOT, save_json
from src.early_detection import evaluate_patient, summarize_patients
from src.feature_ablation import EXCLUSIONS, run_ablation_training
from src.model_data import build_cache
from src.model_evaluation import choose_threshold, evaluate_binary
from src.model_training import load_model_config, versions

EXPLORATORY = (
    "Exploratory re-analysis: inspection of this test set motivated these ablations. "
    "Test rows are never used for fitting, early stopping or threshold selection, but "
    "this is no longer an untouched confirmatory evaluation. Validate findings on a "
    "genuinely untouched patient cohort, preferably from another hospital/time period."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def assert_hashes(hashes):
    for name, expected in hashes.items():
        if file_hash(name) != expected:
            raise ValueError(f"Protected baseline/input changed: {name}")


def preflight(config, early_config):
    """Reject changed settings/splits/versions instead of silently changing the experiment."""
    if load_model_config(early_config["model_config"]) != config:
        raise ValueError("Early-warning evaluation points to a different model configuration")
    baseline_dir = Path(config["models_dir"]) / "xgboost"
    report_dir = Path(config["reports_dir"]) / "xgboost"
    baseline = read_json(report_dir / "results.json")
    if baseline.get("status") != "complete" or baseline.get("model") != "xgboost":
        raise ValueError("A complete saved XGBoost baseline is required")
    if read_json(baseline_dir / "config_resolved.json") != config:
        raise ValueError("Current configuration differs from the original baseline")
    if baseline["versions"] != versions():
        raise ValueError("Runtime/package versions differ from the original baseline")
    cache = build_cache(Path(config["processed_dir"]), Path(config["cache_dir"]),
                        batch_rows=config["batch_rows"])
    if baseline["cache_fingerprint"] != cache["fingerprint"] or baseline["splits"] != cache["splits"]:
        raise ValueError("Baseline data or patient splits changed")
    if read_json(baseline_dir / "feature_columns.json") != cache["features"]:
        raise ValueError("Baseline feature order changed")
    if read_json(baseline_dir / "threshold.json") != baseline["threshold_selection"]:
        raise ValueError("Baseline threshold artifact differs from its results")
    for split in ("validation", "test"):
        original_early = read_json(Path(early_config["output_dir"]) / "xgboost" / f"{split}_metrics.json")
        if original_early["policy"] != early_config["policy"]:
            raise ValueError("Early-detection policy differs from the original evaluation")
    protected = [p for p in baseline_dir.iterdir() if p.is_file()]
    protected += [p for p in report_dir.iterdir() if p.is_file()]
    protected += list(Path(config["cache_dir"]).glob("*/*.npy"))
    protected += [Path(config["cache_dir"]) / "cache_manifest.json"]
    protected += list(Path(config["processed_dir"]).glob("*_patient_ids.csv"))
    protected += [PROJECT_ROOT / name for name in (
        "model_config.json", "early_detection_config.json", "train_xgboost.py",
        "src/model_training.py", "src/model_data.py", "src/model_evaluation.py",
        "src/early_detection.py", "src/feature_engineering.py")]
    logging.info("Hashing protected model, split IDs and disk arrays before the experiment")
    hashes = {str(p.resolve()): file_hash(p) for p in protected}
    return baseline, cache, hashes


def audit_saved_predictions(result, report_dir, config):
    """Recompute threshold from validation only, and requested metrics from saved scores."""
    for split in ("validation", "test"):
        frame = pd.read_csv(report_dir / f"{split}_predictions.csv.gz",
                            usecols=["SepsisLabel", "probability"], float_precision="round_trip")
        if split == "validation":
            selected = choose_threshold(frame.SepsisLabel.to_numpy(), frame.probability.to_numpy())
            if selected != result["threshold_selection"]:
                raise ValueError("Saved threshold is not the original validation-max-F1 rule")
        checked = evaluate_binary(frame.SepsisLabel.to_numpy(), frame.probability.to_numpy(),
                                  result["threshold_selection"]["threshold"], config["calibration_bins"])
        for key in ("n_samples", "n_positive", "auprc", "brier_score", "ece", "f1"):
            if not np.isclose(checked[key], result[split][key], rtol=0, atol=1e-12):
                raise ValueError(f"Saved {split} {key} differs from predictions")


def evaluate_early(name, result, prediction_dir, destination, config, early_config, cache):
    destination.mkdir(parents=True, exist_ok=True)
    summaries = {}
    threshold = result["threshold_selection"]["threshold"]
    for split in ("validation", "test"):
        logging.info("Auditing %s %s patient-level early warnings", name, split)
        records = [evaluate_patient(frame, threshold, early_config["policy"])
                   for frame in iter_prediction_patients(
                       prediction_dir / f"{split}_predictions.csv.gz", config["cache_dir"],
                       split, early_config["batch_rows"], early_config["max_patient_rows"])]
        summary = summarize_patients(records, early_config["policy"])
        expected = cache["splits"][split]
        if (summary["n_patients"] != expected["patients"]
                or summary["n_hourly_records"] != expected["rows"]
                or summary["n_septic_patients"] != expected["positive_patients"]):
            raise ValueError("Early-detection cohort changed")
        if name == "Baseline":
            old = read_json(Path(early_config["output_dir"]) / "xgboost" / f"{split}_metrics.json")
            if old["summary"] != summary:
                raise ValueError("Recomputed baseline early-detection metrics changed")
        summaries[split] = summary
        write_csv(destination / f"{split}_patients.csv", records)
        save_json(destination / f"{split}_metrics.json", {
            "experiment": name, "split": split, "threshold": threshold,
            "policy": early_config["policy"], "summary": summary, "interpretation": EXPLORATORY})
    return summaries


def comparison_rows(results, early):
    rows = []
    for split in ("validation", "test"):
        for name, result in results.items():
            metrics, timing = result[split], early[name][split]
            rows.append({
                "experiment": name, "split": split, "feature_count": result["feature_count"],
                "excluded_features": ";".join(EXCLUSIONS.get(name, [])),
                "trees_saved": result["training_details"]["trees_saved"],
                "threshold": result["threshold_selection"]["threshold"],
                "auprc_ap": metrics["auprc"], "brier_score": metrics["brier_score"], "ece": metrics["ece"],
                "auroc": metrics["auroc"], "f1": metrics["f1"],
                "early_detected_patients": timing["n_primary_detected"],
                "early_evaluable_patients": timing["n_primary_evaluable"],
                "early_detection_rate": timing["primary_sensitivity"],
                "left_censored_excluded": timing["n_left_censored_septic_patients"],
                "nonsepsis_alerted_patients": timing["n_nonseptic_patients_with_alert"],
                "nonsepsis_patients": timing["n_nonseptic_patients"],
                "nonsepsis_false_alert_rate": timing["nonseptic_false_alarm_rate"],
                "nonsepsis_alert_episodes_per_100_patient_hours": timing["nonseptic_alert_episodes_per_100_patient_hours"],
                "strict_early_detection_rate": timing["secondary_sensitivity"],
                "strict_early_detected": timing["n_secondary_detected"],
                "strict_early_evaluable": timing["n_secondary_evaluable"],
                "exploratory": True})
    return rows


def write_comparison(output, results, early, config):
    rows = comparison_rows(results, early)
    write_csv(output / "comparison.csv", rows)
    save_json(output / "comparison.json", {"interpretation": EXPLORATORY, "rows": rows})
    lines = ["# Controlled timing-feature removal — exploratory re-analysis", "", EXPLORATORY, "",
             "Only model input columns change. Original saved patient splits, feature values, seed, class weight, "
             "training implementation, hyperparameters, validation early-stopping rule and validation-max-F1 threshold rule are retained.", "",
             "C removes ICULOS, HospAdmTime AND HospAdmTime__missing (184 predictors). ICULOS remains unchanged in metadata.", "",
             "Trees and numerical thresholds may differ because the original validation-based selection is rerun for each new model. "
             "The baseline is reused without retraining. No probability calibrator is fitted.", ""]
    for split in ("test", "validation"):
        lines += [f"## {split.title()} comparison", "",
                  "| Model | Features | Trees | Threshold | AUPRC (AP) ↑ | Brier ↓ | ECE ↓ | Early detection | Non-sepsis patients alerted |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["split"] != split:
                continue
            lines.append(f"| {r['experiment']} | {r['feature_count']} | {r['trees_saved']} | {r['threshold']:.6f} | "
                         f"{r['auprc_ap']:.6f} | {r['brier_score']:.6f} | {r['ece']:.6f} | "
                         f"{100*r['early_detection_rate']:.2f}% ({r['early_detected_patients']}/{r['early_evaluable_patients']}) | "
                         f"{100*r['nonsepsis_false_alert_rate']:.2f}% ({r['nonsepsis_alerted_patients']}/{r['nonsepsis_patients']}) |")
        lines.append("")
    lines += ["## Definitions and limits", "",
              "- AUPRC is sklearn average precision on every original hourly row; repeated hours are correlated.",
              f"- Brier is mean squared probability error. ECE uses the original {config['calibration_bins']} fixed uniform bins; both assess raw, uncalibrated scores.",
              "- Early detection is any alert in the inclusive 6–12 hours before inferred onset; onset is first positive label hour + 6 after an observed 0-to-1 transition.",
              "- Patients whose records already begin positive are excluded from onset-based denominators, exactly as before. Partly observed windows remain eligible for the primary measure.",
              "- Non-sepsis false-alert rate is the fraction of never-positive patients with at least one alert at any recorded hour. It is not the hourly false-positive rate.",
              "- Thresholds maximize validation hourly F1 (ties choose the highest threshold). False-alert rates are NOT matched between models, so compare detection and alert burden together.",
              "- All other features, including units, missingness and measurement-recency predictors, remain unchanged. Removing explicit timing inputs does not guarantee removal of all timing or care-process proxies.",
              "- These are single-seed point estimates, not a statistical-significance claim or evidence of clinical safety.", "",
              "## Experiment C SHAP", "",
              "The complete ranking is in C/shap/feature_importance_full.csv. It includes every retained feature, including zero contributions, "
              "averaged over every test hour. No importance cutoff is selected.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def plot_comparison(output, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    colors = dict(zip(names, ["#4477aa", "#ee6677", "#228833", "#aa3377"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for name in names:
        bins = [r for r in results[name]["test"]["calibration_bins"] if r["count"]]
        axes[0].plot([r["mean_probability"] for r in bins], [r["observed_positive_fraction"] for r in bins],
                     marker="o", markersize=3, label=name, color=colors[name])
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Test reliability: original 10-bin method", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    rows = pd.read_csv(output / "comparison.csv")
    for name in names:
        color = colors[name]
        row = rows[(rows.experiment == name) & (rows.split == "test")].iloc[0]
        axes[1].scatter(100*row.nonsepsis_false_alert_rate, 100*row.early_detection_rate, color=color, s=55)
        axes[1].annotate(name, (100*row.nonsepsis_false_alert_rate, 100*row.early_detection_rate), xytext=(7, 6), textcoords="offset points")
    axes[1].set(xlabel="Non-sepsis patients with any alert (%)", ylabel="Eligible sepsis patients detected early (%)", title="Validation-selected operating points")
    axes[1].margins(.25)
    fig.suptitle("Timing-feature ablation | exploratory test comparison", fontsize=14)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def run(config, early_config, resume=False):
    started = time.perf_counter()
    output = PROJECT_ROOT / "reports" / "timing_ablation"
    model_root = PROJECT_ROOT / "models" / "timing_ablation"
    if (output.exists() or model_root.exists()) and not resume:
        raise FileExistsError("Ablation outputs already exist. Use --resume to verify and reuse completed experiments.")
    baseline, cache, hashes = preflight(config, early_config)
    output.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if resume and manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["protected_source_sha256"] != hashes or previous["config"] != config or previous["early_policy"] != early_config["policy"]:
            raise ValueError("Cannot resume with changed source artifacts, settings or evaluation policy")
    manifest = {"status": "running", "interpretation": EXPLORATORY, "config": config,
                "early_policy": early_config["policy"], "exclusions": EXCLUSIONS,
                "protected_source_sha256": hashes, "completed_experiments": [], "baseline_retrained": False}
    save_json(manifest_path, manifest)
    results, early = {"Baseline": baseline}, {}
    try:
        baseline_reports = Path(config["reports_dir"]) / "xgboost"
        audit_saved_predictions(baseline, baseline_reports, config)
        early["Baseline"] = evaluate_early("Baseline", baseline, baseline_reports,
                                          output / "Baseline" / "early_detection", config, early_config, cache)
        for name, excluded in EXCLUSIONS.items():
            model_dir, report_dir = model_root / name, output / name
            result_path = report_dir / "results.json"
            if resume and result_path.is_file():
                result = read_json(result_path)
                if result.get("status") != "complete" or read_json(model_dir / "config_resolved.json") != config:
                    raise ValueError(f"Incomplete or differently configured experiment: {name}")
                saved_hashes = read_json(report_dir / "artifact_hashes.json")
                assert_hashes(saved_hashes)
                logging.info("Reusing audited complete experiment %s", name)
            else:
                result = run_ablation_training(name, config, cache, model_dir, report_dir)
                paths = list(model_dir.glob("*.json")) + list(model_dir.glob("*.ubj"))
                paths += [result_path] + list(report_dir.glob("*_predictions.csv.gz"))
                save_json(report_dir / "artifact_hashes.json", {str(p.resolve()): file_hash(p) for p in paths})
            expected_features = [f for f in cache["features"] if f not in excluded]
            if (read_json(model_dir / "feature_columns.json") != expected_features
                    or result["feature_count"] != len(expected_features)
                    or result["cache_fingerprint"] != cache["fingerprint"]
                    or result["splits"] != baseline["splits"]
                    or result["training_class_weight"] != baseline["training_class_weight"]
                    or result["training_details"]["parameters"] != baseline["training_details"]["parameters"]):
                raise ValueError(f"Experiment {name} changed more than its feature subset")
            audit_saved_predictions(result, report_dir, config)
            results[name] = result
            early[name] = evaluate_early(name, result, report_dir, report_dir / "early_detection", config, early_config, cache)
            assert_hashes(hashes)
            manifest["completed_experiments"].append(name)
            save_json(manifest_path, manifest)
            write_comparison(output, results, early, config)
        from src.ablation_shap import write_experiment_shap
        shap_dir = output / "C" / "shap"
        if resume and (shap_dir / "run_manifest.json").is_file():
            shap_manifest = read_json(shap_dir / "run_manifest.json")
            if shap_manifest["status"] != "complete":
                raise ValueError("Incomplete C SHAP output")
            assert_hashes(shap_manifest["source_sha256"])
            if (shap_manifest["features_ranked"] != results["C"]["feature_count"]
                    or shap_manifest["rows_explained"] != cache["splits"]["test"]["rows"]
                    or shap_manifest["cache_fingerprint"] != cache["fingerprint"]):
                raise ValueError("Saved C SHAP does not cover the current model and test cohort")
            for path in shap_manifest["artifacts"].values():
                if not Path(path).is_file() or Path(path).stat().st_size == 0:
                    raise ValueError(f"Missing C SHAP artifact: {path}")
        else:
            shap_manifest = write_experiment_shap(model_root / "C", output / "C", Path(config["cache_dir"]), shap_dir,
                                                  batch_rows=4096, n_jobs=config["n_jobs"])
        assert_hashes(hashes)
        plot_comparison(output, results)
        manifest.update(status="complete", elapsed_seconds=round(time.perf_counter()-started, 2),
                        source_integrity_verified=True, shap=shap_manifest,
                        checks=["original baseline unchanged", "same patient memberships and row order",
                                "retained feature values unchanged; bounded column projection",
                                "original training parameters and class weights",
                                "validation-only early stopping and independently verified validation-max-F1 thresholds",
                                "all saved prediction IDs/times/labels checked against source cache",
                                "baseline early-detection metrics exactly reproduced", "all C test rows/features explained"])
        save_json(manifest_path, manifest)
        logging.info("COMPLETE: %s", output / "comparison.md")
        print((output / "comparison.md").read_text(encoding="utf-8"))
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    # Windows redirected consoles may default to cp1252, which cannot print
    # the directional arrows used in the UTF-8 comparison report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Original model_config.json; must match saved baseline")
    parser.add_argument("--early-config", default=None, help="Original early_detection_config.json")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed experiments; never overwrite models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_model_config(args.config), load_evaluation_config(args.early_config), args.resume)
