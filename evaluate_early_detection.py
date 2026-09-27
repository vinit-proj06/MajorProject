"""Evaluate patient-level warning timing from saved, frozen model predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, save_json
from src.early_detection import evaluate_patient, summarize_patients
from src.model_data import build_cache, close_arrays
from src.model_training import load_model_config

LOGGER = logging.getLogger(__name__)
COLUMNS = ["patient_id", "dataset", "ICULOS", "SepsisLabel", "probability", "predicted_SepsisLabel"]


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_evaluation_config(path=None):
    path = Path(path or PROJECT_ROOT / "early_detection_config.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_config", "output_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    for key in ("batch_rows", "max_patient_rows"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, allowed in (("models", {"xgboost", "random_forest"}), ("splits", {"validation", "test"})):
        values = config[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)) or not set(values) <= allowed:
            raise ValueError(f"Invalid {key}: {values}")
    if config["policy"]["label_shift_hours"] != 6:
        raise ValueError("These PhysioNet labels require a six-hour onset offset")
    return config


def iter_prediction_patients(path, cache_dir, split, batch_rows, max_patient_rows):
    """Check every prediction row against validated source metadata; buffer one patient.

    Only four one-dimensional metadata arrays are mapped. No feature matrix or
    combined hourly dataset is loaded. Reappearing patient groups are rejected.
    """
    arrays, carry, seen, offset = {}, None, set(), 0
    try:
        for name in ("patient_id", "dataset", "ICULOS", "y"):
            arrays[name] = np.load(Path(cache_dir) / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        with pd.read_csv(path, chunksize=batch_rows, dtype={"patient_id": str, "dataset": str}, float_precision="round_trip") as reader:
            for chunk in reader:
                if list(chunk.columns) != COLUMNS or chunk.isna().any().any():
                    raise ValueError(f"Invalid prediction columns or missing values: {path}")
                end = offset + len(chunk)
                if end > len(arrays["y"]):
                    raise ValueError("Prediction file has extra rows")
                for column, source in (("patient_id", "patient_id"), ("dataset", "dataset"), ("ICULOS", "ICULOS"), ("SepsisLabel", "y")):
                    if not np.array_equal(chunk[column].to_numpy(), arrays[source][offset:end]):
                        raise ValueError(f"Prediction {column} differs from source {split} rows at offset {offset}")
                offset = end
                chunk = pd.concat([carry, chunk], ignore_index=True) if carry is not None else chunk.reset_index(drop=True)
                ids = chunk["patient_id"].to_numpy()
                boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(chunk)]
                for start, stop in zip(boundaries[:-2], boundaries[1:-1]):
                    patient = str(ids[start])
                    if patient in seen or stop - start > max_patient_rows:
                        raise ValueError(f"Reappearing or oversized patient group: {patient}")
                    seen.add(patient)
                    yield chunk.iloc[start:stop].copy()
                carry = chunk.iloc[boundaries[-2]:].copy()
                if len(carry) > max_patient_rows:
                    raise ValueError("Patient buffer exceeds configured max_patient_rows")
        if offset != len(arrays["y"]):
            raise ValueError("Prediction file is incomplete")
        if carry is not None and len(carry):
            if str(carry["patient_id"].iloc[0]) in seen:
                raise ValueError("Patient occurs in noncontiguous groups")
            yield carry
    finally:
        close_arrays(arrays)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty patient report")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def write_comparison(stage, results, records, policy):
    rows = [{"model": name, "split": split, "threshold": result["threshold"],
             **{key: value for key, value in result["summary"].items() if not isinstance(value, (dict, list))}}
            for (name, split), result in results.items()]
    write_csv(stage / "comparison.csv", rows)
    text = ["PATIENT-LEVEL EARLY SEPSIS DETECTION", "",
            f"Primary window: {policy['min_lead_hours']} to {policy['max_lead_hours']} hours before inferred onset (inclusive).",
            f"Alert policy: {policy['consecutive_positive_hours']} consecutive positive hour(s); emit at each qualifying hour, no cooldown.",
            "Thresholds are unchanged from validation selection; no test-set tuning or retraining.",
            "Onset = first positive label hour + 6 ONLY after an observed 0-to-1 transition.",
            "Records positive from their first hour have unknown onset and are excluded, not counted as misses.",
            "Primary denominator: known-transition sepsis patients with at least one observed hour in the warning window.",
            "Strict sensitivity: onset lies in the record and the entire warning window is observed.", ""]
    for (name, split), result in results.items():
        s = result["summary"]
        text.extend([f"{name.upper()} | {split.upper()} | threshold={result['threshold']:.8f}",
                     f"Patients: {s['n_patients']}; sepsis: {s['n_septic_patients']}; non-sepsis: {s['n_nonseptic_patients']}",
                     f"Early detection: {s['n_primary_detected']}/{s['n_primary_evaluable']} = {percentage(s['primary_sensitivity'])}",
                     f"Unknown onset (first row already positive): {s['n_left_censored_septic_patients']}",
                     f"Inferred onset after record end: {s['n_onset_after_record_end']}",
                     f"Strict sensitivity: {s['n_secondary_detected']}/{s['n_secondary_evaluable']} = {percentage(s['secondary_sensitivity'])}",
                     f"Median first useful warning lead (detected patients only): {s['median_first_useful_warning_lead_hours']} hours",
                     f"Non-sepsis patients with an alert: {s['n_nonseptic_patients_with_alert']}/{s['n_nonseptic_patients']} = {percentage(s['nonseptic_false_alarm_rate'])}",
                     f"Non-sepsis alert episodes per 100 patient-hours: {s['nonseptic_alert_episodes_per_100_patient_hours']}"])
        for lead, counts in s["secondary_sensitivity_by_min_lead_hours"].items():
            text.append(f"Strict {lead}-{policy['max_lead_hours']}h warning sensitivity: {counts['n_detected']}/{counts['n_evaluable']} = {percentage(counts['sensitivity'])}")
        text.append("")
    text.extend(["These are retrospective warning-timing measures, not proven clinical benefit or the official challenge utility score.",
                 "A very early alert alone does not count; a continuing/repeated positive output inside the window does.",
                 "Denominators can differ across lead times. Strict cohorts require full observation and onset within the record.",
                 "Validation is development data. Test results are descriptive; changing settings after viewing them needs a new holdout."])
    (stage / "comparison.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    save_json(stage / "comparison.json", {"policy": policy, "results": list(results.values())})
    for split in sorted({key[1] for key in results}):
        plot_comparison(stage / f"{split}_early_detection.png", split, results, records, policy)
    return "\n".join(text)


def plot_comparison(path, split, results, records, policy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name, result_split in results if result_split == split]
    labels = [name.replace("_", " ").title() for name in names]
    colors = ["#2675b5", "#e68a2e"][:len(names)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, metric, title in ((axes[0, 0], "primary_sensitivity", "Early detection among evaluable sepsis patients"),
                              (axes[0, 1], "nonseptic_false_alarm_rate", "Non-sepsis patients with at least one alert")):
        values = [results[(name, split)]["summary"][metric] for name in names]
        ax.bar(labels, [100 * value if value is not None else 0 for value in values], color=colors)
        ax.set(title=title, ylabel="Patients (%)", ylim=(0, 100))
        for index, name in enumerate(names):
            summary = results[(name, split)]["summary"]
            count = (f"{summary['n_primary_detected']}/{summary['n_primary_evaluable']}" if metric == "primary_sensitivity"
                     else f"{summary['n_nonseptic_patients_with_alert']}/{summary['n_nonseptic_patients']}")
            ax.text(index, 100 * (values[index] or 0) + 2, f"{percentage(values[index])}\n{count}", ha="center", va="bottom")
    leads = policy["secondary_min_lead_hours"]
    for index, name in enumerate(names):
        summaries = results[(name, split)]["summary"]["secondary_sensitivity_by_min_lead_hours"]
        width = 0.7 / len(names)
        lead_keys = [format(float(lead), "g") for lead in leads]
        heights = [summaries[key]["sensitivity"] for key in lead_keys]
        positions = np.arange(len(leads)) + (index - (len(names) - 1) / 2) * width
        axes[1, 0].bar(positions, [100 * x if x is not None else 0 for x in heights], width, label=labels[index], color=colors[index])
        for x, value, key in zip(positions, heights, lead_keys):
            axes[1, 0].text(x, 100 * (value or 0) + 1, f"n={summaries[key]['n_evaluable']}" if value is not None else "N/A", ha="center", fontsize=8)
        lead_values = [r["first_useful_alert_lead_hours"] for r in records[(name, split)] if r["primary_detected"]]
        if lead_values:
            axes[1, 1].hist(lead_values, bins=np.arange(policy["min_lead_hours"] - .5, policy["max_lead_hours"] + 1.5),
                            histtype="step", linewidth=2, label=labels[index], color=colors[index])
    axes[1, 0].set(xticks=np.arange(len(leads)), xticklabels=[f"{lead}–{policy['max_lead_hours']}h" for lead in leads],
                   ylim=(0, 100), ylabel="Patients detected (%)", title="Strict cohort sensitivity by warning window")
    axes[1, 0].legend()
    axes[1, 1].set(xlabel="Hours before inferred onset", ylabel="Detected sepsis patients", title="First useful warning lead time")
    if axes[1, 1].get_legend_handles_labels()[0]:
        axes[1, 1].legend()
    fig.suptitle(f"{split.title()} patient-level early detection | {policy['min_lead_hours']}–{policy['max_lead_hours']}h primary window", fontsize=15)
    fig.supxlabel("Frozen thresholds | Onset inferred from shifted labels | Unknown onset excluded | No proven clinical benefit", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config):
    started = time.perf_counter()
    model_config = load_model_config(config["model_config"])
    output = Path(config["output_dir"]).resolve()
    # A rerun can move only this dedicated output directory, never any input ancestor.
    for key in ("processed_dir", "cache_dir", "models_dir", "reports_dir"):
        source = Path(model_config[key]).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Early-detection output must be separate from {key}")
    if output == output.parent:
        raise ValueError("Cannot use a filesystem root as output")
    cache = build_cache(Path(model_config["processed_dir"]), Path(model_config["cache_dir"]), batch_rows=config["batch_rows"])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".inprogress-" + uuid.uuid4().hex[:10])
    stage.mkdir()
    results, patient_records, source_hashes = {}, {}, {}
    try:
        save_json(stage / "config_resolved.json", config)
        for name in config["models"]:
            report_dir = Path(model_config["reports_dir"]) / name
            result_path = report_dir / "results.json"
            model_result = json.loads(result_path.read_text(encoding="utf-8"))
            if model_result["status"] != "complete" or model_result["model"] != name or model_result["cache_fingerprint"] != cache["fingerprint"]:
                raise ValueError(f"{name} results do not match the validated processed data")
            threshold_path = Path(model_config["models_dir"]) / name / "threshold.json"
            threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
            if threshold != model_result["threshold_selection"]:
                raise ValueError(f"Saved threshold changed for {name}")
            for source in (result_path, threshold_path):
                source_hashes[str(source)] = file_hash(source)
            destination = stage / name
            destination.mkdir()
            for split in config["splits"]:
                LOGGER.info("Evaluating %s %s patients with frozen threshold %.8f", name, split, threshold["threshold"])
                if model_result["splits"][split] != cache["splits"][split]:
                    raise ValueError("Saved model cohort counts differ from validated data")
                path = report_dir / f"{split}_predictions.csv.gz"
                source_hashes[str(path)] = file_hash(path)
                records = []
                for frame in iter_prediction_patients(path, model_config["cache_dir"], split, config["batch_rows"], config["max_patient_rows"]):
                    records.append(evaluate_patient(frame, threshold["threshold"], config["policy"]))
                    if len(records) % 2000 == 0:
                        LOGGER.info("%s %s: evaluated %s patients", name, split, len(records))
                expected = cache["splits"][split]
                if len(records) != expected["patients"] or sum(r["n_hours"] for r in records) != expected["rows"] or sum(r["septic_patient"] for r in records) != expected["positive_patients"]:
                    raise ValueError("Patient-level summary counts disagree with the saved cohort")
                summary = summarize_patients(records, config["policy"])
                result = {"model": name, "split": split, "threshold": threshold["threshold"], "threshold_source": "existing validation-max-F1 threshold",
                          "policy": config["policy"], "summary": summary}
                results[(name, split)], patient_records[(name, split)] = result, records
                write_csv(destination / f"{split}_patients.csv", records)
                save_json(destination / f"{split}_metrics.json", result)
        for source, digest in source_hashes.items():
            if file_hash(source) != digest:
                raise ValueError(f"Source artifact changed during evaluation: {source}")
        report = write_comparison(stage, results, patient_records, config["policy"])
        save_json(stage / "run_manifest.json", {"status": "complete", "elapsed_seconds": round(time.perf_counter() - started, 2),
                  "cache_fingerprint": cache["fingerprint"], "source_sha256": source_hashes, "config": config,
                  "checks": ["source cache current", "patient splits disjoint", "every prediction label/time/ID matches source", "complete patient and row coverage",
                             "frozen predicted labels match saved threshold", "contiguous hourly time", "monotone sepsis labels", "source artifacts unchanged"],
                  "references": ["https://physionet.org/content/challenge-2019/1.0.0/",
                                 "https://physionet.org/files/challenge-2019/1.0.0/physionet_challenge_2019_ccm_manuscript.pdf"]})
    except Exception as error:
        save_json(stage / "RUN_FAILED.json", {"error": f"{type(error).__name__}: {error}"})
        LOGGER.exception("Evaluation failed; incomplete diagnostics retained at %s", stage)
        raise
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
    LOGGER.info("Completed in %.1fs; reports: %s", time.perf_counter() - started, output)
    print(report)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Early-detection config JSON; default early_detection_config.json")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_evaluation_config(parser.parse_args().config))
