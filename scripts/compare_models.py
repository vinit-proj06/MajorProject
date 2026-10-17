"""Compare completed baselines without tuning anything on test results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import save_json
from src.model_training import load_model_config

METRICS = ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auroc", "auprc", "pr_auc_trapezoidal", "brier_score", "log_loss", "ece"]


def format_metric(value):
    return "N/A" if value is None else f"{value:.6f}"


def compare(config):
    directory = Path(config["reports_dir"])
    results = {}
    for name in ("xgboost", "random_forest"):
        path = directory / name / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Run scripts/train_{name}.py first; missing {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"{name} run is incomplete")
        results[name] = result
    first, second = results.values()
    for key in ("cache_fingerprint", "feature_order_sha256"):
        if first[key] != second[key]:
            raise ValueError("Cannot compare models trained/evaluated on different data or feature orders")
    rows = []
    for name, result in results.items():
        for split in ("validation", "test"):
            values = result[split]
            rows.append({"model": name, "split": split, "threshold": values["threshold"],
                         "tn": values["tn"], "fp": values["fp"], "fn": values["fn"], "tp": values["tp"],
                         **{key: values[key] for key in METRICS}})
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preferred = max(results, key=lambda name: results[name]["validation"]["auprc"])
    summary = {"models": list(results), "preferred_by_validation_auprc": preferred, "selection_metric": "validation average precision (AUPRC/AP)",
               "test_used_for_selection": False, "results": rows,
               "notes": ["Sensitivity equals recall for the positive class.", "Each model's decision threshold maximizes validation F1; test thresholds are frozen.",
                         "AUPRC is average precision; trapezoidal PR area is separately named.",
                         "Calibration is assessed on raw scores with reliability diagrams, Brier, log loss and ECE using the configured bins; no calibrator was fitted.",
                         "Validation results are development estimates; test results do not establish clinical safety or early-alert utility."]}
    save_json(directory / "comparison.json", summary)
    text = ["SEPSIS MODEL COMPARISON", "", "Patient-disjoint hourly evaluation; original holdout class distribution retained.",
            f"Preferred by validation AUPRC/AP: {preferred} (test not used for selection)", ""]
    for split in ("validation", "test"):
        text.extend([split.upper(), "Metric                       XGBoost        Random Forest"])
        for metric in ["threshold", *METRICS]:
            text.append(f"{metric:26s} {format_metric(results['xgboost'][split][metric]):>13s} {format_metric(results['random_forest'][split][metric]):>18s}")
        for name, result in results.items():
            m = result[split]
            text.append(f"{name} confusion matrix [[TN, FP], [FN, TP]] = [[{m['tn']}, {m['fp']}], [{m['fn']}, {m['tp']}]]")
        text.append("")
    text.extend(summary["notes"])
    text.extend(["", "Data and runtime:"])
    for name, result in results.items():
        text.append(f"{name}: {result['feature_count']} predictors; fit {result['fit_seconds']:.1f}s; total {result['elapsed_seconds']:.1f}s")
    (directory / "comparison.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    compare(load_model_config(parser.parse_args().config))
