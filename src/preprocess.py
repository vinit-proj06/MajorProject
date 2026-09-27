"""Run bounded ingestion, patient splitting, training-only fitting and export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import logging
from pathlib import Path
import platform
import sys
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import BoundedParquetWriter, METADATA, TARGET, load_config, read_patient, save_json, setup_logging
from src.dataset_statistics import DiskStatistics, fit_training_parameters, write_statistics_report
from src.feature_engineering import engineer_patient, feature_metadata
from src.inspect_dataset import inspect_dataset
from src.split_dataset import SPLITS, split_patients
from src.storage import extension, iter_batches, raw_chunk_paths, read_chunk, resolve_storage, write_chunk

LOG_FIELDS = ["patient_id", "dataset", "filename", "n_rows", "status", "error_message"]


def runtime_versions():
    result = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "pyarrow", "tqdm"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def ingest(records, schema, config, stage, inspection):
    reports = Path(config["reports_dir"])
    raw_dir = stage / "raw_processed"
    raw_dir.mkdir()
    duplicate_ids = set(inspection["discovery"]["duplicate_patient_ids"])
    manifest, frames = [], []
    batch_rows, chunk, failed = 0, 0, 0
    max_buffer_rows, max_buffer_patients = 0, 0
    duplicate_rows, row_count = 0, 0
    dataset_hash = hashlib.sha256()

    with DiskStatistics(schema, stage / ".scratch") as stats, (reports / "processing_log.csv").open("w", newline="", encoding="utf-8") as log_file, (reports / "failed_files.csv").open("w", newline="", encoding="utf-8") as failed_file:
        log_writer, failure_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS), csv.DictWriter(failed_file, fieldnames=LOG_FIELDS)
        log_writer.writeheader()
        failure_writer.writeheader()

        def flush():
            nonlocal batch_rows, chunk
            if not frames:
                return
            combined = pd.concat(frames, ignore_index=True)
            # I/O and statistics failures are systemic: propagate rather than silently losing a batch.
            stats.update(combined)
            chunk += 1
            write_chunk(combined, raw_dir / f"chunk_{chunk:05d}{extension(config['storage_format'])}", config)
            frames.clear()
            batch_rows = 0

        progress = tqdm(records, desc="Validating patients", unit="patient", disable=not sys.stderr.isatty())
        for index, record in enumerate(progress, 1):
            entry = {"patient_id": record.patient_id, "dataset": record.dataset, "filename": str(record.path), "n_rows": 0, "status": "success", "error_message": ""}
            try:
                if record.patient_id in duplicate_ids:
                    raise ValueError("Duplicate patient ID; all files sharing this ID are quarantined")
                frame, digest, duplicates = read_patient(record, schema, config)
            except Exception as exc:
                failed += 1
                entry.update(status="failed", error_message=f"{type(exc).__name__}: {exc}")
                failure_writer.writerow(entry)
                log_writer.writerow(entry)
                continue
            if frames and (len(frames) >= config["batch_size"] or batch_rows + len(frame) > config["max_batch_rows"]):
                flush()
            frames.append(frame)
            batch_rows += len(frame)
            max_buffer_rows, max_buffer_patients = max(max_buffer_rows, batch_rows), max(max_buffer_patients, len(frames))
            positives = int(frame[TARGET].sum())
            manifest.append({"patient_id": record.patient_id, "dataset": record.dataset, "filename": str(record.path), "n_rows": len(frame), "positive_rows": positives,
                             "positive_patient": positives > 0, "first_ICULOS": float(frame.ICULOS.iloc[0]), "last_ICULOS": float(frame.ICULOS.iloc[-1]),
                             "sha256": digest, "duplicate_rows": duplicates})
            dataset_hash.update(f"{record.dataset}/{record.patient_id}:{digest}\n".encode())
            duplicate_rows += duplicates
            row_count += len(frame)
            entry["n_rows"] = len(frame)
            log_writer.writerow(entry)
            if index % config["progress_every"] == 0:
                logging.info("Validated %s/%s: processed=%s failed=%s rows=%s", index, len(records), len(manifest), failed, row_count)
                log_file.flush()
                failed_file.flush()
        flush()
        feature_stats = stats.finalize()
    if not manifest:
        raise ValueError("No valid patients remain after ingestion; see failed_files.csv")
    manifest = pd.DataFrame(manifest)
    manifest.to_csv(reports / "patient_manifest.csv", index=False)
    manifest.to_csv(stage / "patient_manifest.csv", index=False)
    return manifest, feature_stats, {"failed_files": failed, "duplicate_rows": duplicate_rows, "raw_chunks": chunk,
                                    "input_content_sha256": dataset_hash.hexdigest(), "max_raw_buffer_rows": max_buffer_rows,
                                    "max_raw_buffer_patients": max_buffer_patients}


def apply_parameters(engineered, metadata, parameters):
    """Apply saved train-only fallbacks; never estimate anything from this patient."""
    dropped = set()
    fill_values = {}
    for column, entry in metadata.items():
        if entry["fallback"] == "training_median":
            value = parameters["values"][entry["fallback_source"]]
            if value is None:
                dropped.add(column)
            else:
                fill_values[column] = value
        elif entry["fallback"] == "zero":
            fill_values[column] = 0.0
    # One numeric matrix and broadcasted fill replace hundreds of per-column
    # pandas fillna/astype operations for every patient.
    columns = [column for column in engineered if column not in dropped]
    predictors = [column for column in columns if column not in METADATA]
    source_positions = {column: index for index, column in enumerate(engineered.columns)}
    numeric = engineered.iloc[:, [source_positions[column] for column in predictors]].to_numpy(dtype=np.float64, copy=True)
    fallbacks = np.array([fill_values.get(column, np.nan) for column in predictors], dtype=np.float64)
    np.copyto(numeric, fallbacks[None, :], where=np.isnan(numeric))
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite or unfilled engineered predictors")
    # Build dtype groups together, retaining metadata/target dtypes unchanged.
    groups = {"float64": [], "uint8": [], "float32": []}
    for index, column in enumerate(predictors):
        dtype = ("float64" if column in {"ICULOS", "HospAdmTime"}
                 else "uint8" if column.endswith("__missing") else "float32")
        groups[dtype].append(index)
    assembled_columns = [column for column in columns if column in METADATA]
    parts = [engineered.iloc[:, [source_positions[column] for column in assembled_columns]]]
    for dtype, indexes in groups.items():
        if not indexes:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            values = numeric[:, indexes].astype(dtype, copy=False)
        if dtype == "float32" and not np.isfinite(values).all():
            raise ValueError("Feature overflows float32")
        group_columns = [predictors[i] for i in indexes]
        parts.append(pd.DataFrame(values, columns=group_columns, index=engineered.index))
        assembled_columns.extend(group_columns)
    assembled_positions = {column: index for index, column in enumerate(assembled_columns)}
    return pd.concat(parts, axis=1).iloc[:, [assembled_positions[column] for column in columns]]


def check_causality(frame, config, metadata, parameters):
    """Prefix and future-mutation checks, including invariance to ALL target values."""
    if len(frame) < 2:
        return 0
    full = apply_parameters(engineer_patient(frame, config), metadata, parameters)
    checks = 0
    for cut in sorted({1, max(1, len(frame) // 2), len(frame) - 1}):
        prefix = apply_parameters(engineer_patient(frame.iloc[:cut], config), metadata, parameters)
        pd.testing.assert_frame_equal(full.iloc[:cut].reset_index(drop=True), prefix.reset_index(drop=True), check_exact=False, rtol=1e-5, atol=1e-5)
        changed = frame.copy()
        columns = [c for c in frame if c not in METADATA and c != "ICULOS"]
        changed.loc[changed.index[cut:], columns] = 999.0
        changed[TARGET] = 1 - changed[TARGET]
        mutated = apply_parameters(engineer_patient(changed, config), metadata, parameters)
        pd.testing.assert_frame_equal(full.drop(columns=TARGET).iloc[:cut].reset_index(drop=True), mutated.drop(columns=TARGET).iloc[:cut].reset_index(drop=True), check_exact=False, rtol=1e-5, atol=1e-5)
        checks += 2
    return checks


def export_patients(stage, config, mapping, metadata, parameters, manifest):
    writers = {split: BoundedParquetWriter(stage / f"{split}{extension(config['storage_format'])}", config) for split in SPLITS}
    total, causal_checks, failed_entries, seen = 0, 0, [], set()
    validated_sources = set()
    filenames = dict(zip(manifest.patient_id, manifest.filename))
    with (Path(config["reports_dir"]) / "output_processing_log.csv").open("w", newline="", encoding="utf-8") as handle:
        log_writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        log_writer.writeheader()
        try:
            for path in raw_chunk_paths(stage / "raw_processed"):
                # Raw chunks were bounded during ingestion and never split a patient.
                raw = read_chunk(path)
                for patient_id, patient in raw.groupby("patient_id", sort=False):
                    entry = {"patient_id": patient_id, "dataset": str(patient.dataset.iloc[0]), "filename": filenames[patient_id], "n_rows": len(patient), "status": "success", "error_message": ""}
                    try:
                        result = apply_parameters(engineer_patient(patient, config), metadata, parameters)
                    except Exception as exc:
                        entry.update(status="failed", error_message=f"Feature transform {type(exc).__name__}: {exc}")
                        failed_entries.append(entry)
                        log_writer.writerow(entry)
                        continue
                    source = entry["dataset"]
                    if source not in validated_sources:
                        causal_checks += check_causality(patient, config, metadata, parameters)
                        validated_sources.add(source)
                    # Write errors propagate: a disk error must not masquerade as a bad patient.
                    writers[mapping[patient_id]].add(result)
                    seen.add(patient_id)
                    log_writer.writerow(entry)
                    total += 1
                    if total % config["progress_every"] == 0:
                        logging.info("Engineered %s/%s patients", total, len(mapping))
        finally:
            for writer in writers.values():
                writer.close()
    if failed_entries:
        with (Path(config["reports_dir"]) / "failed_files.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS).writerows(failed_entries)
        log = pd.read_csv(Path(config["reports_dir"]) / "processing_log.csv", keep_default_na=False)
        for entry in failed_entries:
            row = log.patient_id == entry["patient_id"]
            log.loc[row, ["status", "error_message"]] = [entry["status"], entry["error_message"]]
        log.to_csv(Path(config["reports_dir"]) / "processing_log.csv", index=False)
    return seen, {"causality_checks_passed": causal_checks, "transform_failures": len(failed_entries),
                  "max_output_buffer_rows_per_split": max(w.max_buffer_rows for w in writers.values()),
                  "max_output_buffer_patients_per_split": max(w.max_buffer_patients for w in writers.values())}


def validate_outputs(stage, config, mapping, manifest, predictors):
    """Scan every final row in bounded batches and reconcile with patient manifests."""
    report, all_seen = {}, set()
    expected_by_patient = manifest.set_index("patient_id")
    for split in SPLITS:
        path = stage / f"{split}{extension(config['storage_format'])}"
        if not path.is_file():
            raise ValueError(f"No output for {split}: dataset is too small or all patients failed")
        patient_stats, rows, positives = {}, 0, 0
        for frame in iter_batches(path, batch_size=config["validation_batch_rows"]):
            if not {"patient_id", "dataset", "ICULOS", TARGET, *predictors}.issubset(frame.columns):
                raise ValueError(f"Missing output columns in {split}")
            if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
                raise ValueError("Invalid output target")
            if not np.isfinite(frame[predictors].to_numpy(dtype=np.float64)).all():
                raise ValueError("Output contains missing/infinite predictors")
            rows += len(frame)
            positives += int(frame[TARGET].sum())
            for patient_id, group in frame.groupby("patient_id", sort=False):
                if mapping.get(patient_id) != split:
                    raise ValueError("Patient is in an incorrect split")
                if patient_id not in patient_stats:
                    patient_stats[patient_id] = [0, 0, None, float(group.ICULOS.iloc[0])]
                state = patient_stats[patient_id]
                hours = group.ICULOS.to_numpy()
                if not np.equal(np.diff(hours), 1).all() or (state[2] is not None and hours[0] != state[2] + 1):
                    raise ValueError("Output hourly order/continuity changed")
                if set(group.dataset) != {expected_by_patient.loc[patient_id, "dataset"]}:
                    raise ValueError("Source metadata changed")
                state[0] += len(group)
                state[1] += int(group[TARGET].sum())
                state[2] = hours[-1]
        expected_ids = {p for p, s in mapping.items() if s == split}
        if set(patient_stats) != expected_ids or all_seen & set(patient_stats):
            raise ValueError("Patient coverage or split disjointness failure")
        for patient_id, (count, positive_count, last_hour, first_hour) in patient_stats.items():
            expected = expected_by_patient.loc[patient_id]
            if count != expected.n_rows or positive_count != expected.positive_rows or first_hour != expected.first_ICULOS or last_hour != expected.last_ICULOS:
                raise ValueError(f"Patient rows, labels or time changed: {patient_id}")
        all_seen.update(patient_stats)
        csv_ids = set(pd.read_csv(stage / f"{split}_patient_ids.csv").patient_id)
        if csv_ids != expected_ids:
            raise ValueError("Saved split IDs do not match output records")
        report[split] = {"patients": len(patient_stats), "rows": rows, "positive_rows": positives, "negative_rows": rows - positives,
                         "positive_patients": sum(s[1] > 0 for s in patient_stats.values()),
                         "negative_patients": sum(s[1] == 0 for s in patient_stats.values()), "bytes": path.stat().st_size}
    if all_seen != set(manifest.patient_id):
        raise ValueError("Final manifest and outputs differ")
    return {"passed": True, "split_overlap_counts": {"train_validation": 0, "train_test": 0, "validation_test": 0},
            "target_valid": True, "time_and_row_counts_preserved": True, "no_infinite_or_missing_predictors": True,
            "no_completely_empty_feature_columns": True, "predictor_count": len(predictors), "splits": report}


def run_pipeline(config, overwrite=False):
    started = time.perf_counter()
    output, reports = Path(config["output_dir"]), Path(config["reports_dir"])
    setup_logging(reports)
    config = dict(config)
    config["storage_format"], storage_warning = resolve_storage(config)
    if storage_warning:
        logging.warning(storage_warning)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} is not empty; use --overwrite to build a new run and retain the old directory as a backup")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f"{output.name}.inprogress-{uuid.uuid4().hex[:12]}"
    stage.mkdir()
    try:
        records, schema, inspection = inspect_dataset(config)
        metadata = feature_metadata(schema, config)
        save_json(stage / "config_resolved.json", config)
        save_json(stage / "schema.json", {"columns": schema, "derived_from": "modal representative header; enforced on every admitted PSV"})
        logging.info("Ingesting and validating all files with patient batches <=%s", config["batch_size"])
        manifest, feature_stats, ingestion = ingest(records, schema, config, stage, inspection)
        mapping, split_report = split_patients(manifest, config, stage)
        save_json(reports / "split_report.json", split_report)
        logging.info("Fitting imputation on %s training patients only", split_report["splits"]["train"]["patients"])
        parameters = fit_training_parameters(stage / "raw_processed", {p for p, s in mapping.items() if s == "train"}, schema, stage / ".scratch")
        parameters["training_patient_ids_sha256"] = hashlib.sha256("\n".join(sorted(p for p, s in mapping.items() if s == "train")).encode()).hexdigest()
        save_json(stage / "imputation_parameters.json", parameters)
        dropped = [c for c, entry in metadata.items() if entry["fallback"] == "training_median" and parameters["values"][entry["fallback_source"]] is None]
        predictors = [c for c in metadata if c not in dropped]
        save_json(stage / "feature_metadata.json", {"predictor_columns": predictors, "excluded_metadata_columns": sorted(METADATA), "dropped_empty_training_features": dropped, "features": metadata})
        seen, transformation = export_patients(stage, config, mapping, metadata, parameters, manifest)
        # A rare patient-local transform error is quarantined without resplitting other patients.
        final_manifest = manifest[manifest.patient_id.isin(seen)].copy()
        mapping = {p: s for p, s in mapping.items() if p in seen}
        for split in SPLITS:
            final_manifest[final_manifest.patient_id.map(mapping) == split][["patient_id", "dataset"]].sort_values("patient_id").to_csv(stage / f"{split}_patient_ids.csv", index=False)
        final_manifest.to_csv(stage / "patient_manifest.csv", index=False)
        # Preserve the raw manifest separately; final reports describe exported patients.
        manifest.to_csv(reports / "raw_patient_manifest.csv", index=False)
        final_manifest.to_csv(reports / "patient_manifest.csv", index=False)
        logging.info("Validating all final output rows and patient assignments...")
        validation = validate_outputs(stage, config, mapping, final_manifest, predictors)
        validation.update(transformation)
        save_json(reports / "validation_report.json", validation)
        warnings = inspection["discovery"]["warnings"] + split_report["warnings"]
        if storage_warning:
            warnings.append(storage_warning)
        split_report["splits"] = validation["splits"]
        split_report["scope"] = "Final exported patient sets; strata show initial stratified allocation before any transform failures."
        split_report["transform_failures"] = transformation["transform_failures"]
        save_json(reports / "split_report.json", split_report)
        if ingestion["failed_files"]:
            warnings.append(f"{ingestion['failed_files']} input files quarantined; see failed_files.csv")
        if transformation["transform_failures"]:
            warnings.append("Patient-local feature failures occurred. Final outputs exclude these patients; raw descriptive statistics and fitted training parameters retain their validated raw records.")
        if dropped:
            warnings.append(f"Dropped {len(dropped)} predictors with no training observations; raw missingness indicators retained.")
        raw_rows, final_rows = int(manifest.n_rows.sum()), int(final_manifest.n_rows.sum())
        positive_rows = int(final_manifest.positive_rows.sum())
        summary = {
            "status": "complete", "discovered_files": len(records), "discovered_by_dataset": inspection["discovery"]["patients_by_dataset"],
            "successfully_processed": len(final_manifest), "failed_files": ingestion["failed_files"] + transformation["transform_failures"],
            "total_patients": len(final_manifest), "total_hourly_rows": final_rows, "raw_validated_patients": len(manifest), "raw_validated_hourly_rows": raw_rows,
            "patients_by_dataset": {str(k): int(v) for k, v in final_manifest.dataset.value_counts().items()},
            "positive_patients": int(final_manifest.positive_patient.sum()), "negative_patients": int((~final_manifest.positive_patient).sum()),
            "positive_samples": positive_rows, "negative_samples": final_rows - positive_rows, "positive_sample_percentage": 100 * positive_rows / final_rows,
            "average_hours_per_patient": float(final_manifest.n_rows.mean()), "minimum_hours_per_patient": int(final_manifest.n_rows.min()), "maximum_hours_per_patient": int(final_manifest.n_rows.max()),
            "duplicate_patient_ids": len(inspection["discovery"]["duplicate_patient_ids"]), "duplicate_rows": ingestion["duplicate_rows"],
            "duplicate_rows_scope": "Exact full-row duplicates within patient, including ICULOS; repeated measurements at distinct hours are preserved, cross-patient identical rows are not assumed duplicates.",
            "schema_columns": schema, "all_admitted_schemas_consistent": True, "predictor_count": len(predictors), "engineered_feature_count": sum(c not in schema for c in predictors),
            "dropped_empty_training_features": dropped, "splits": validation["splits"], "validation_passed": True,
            "statistics_scope": "All validated raw patients, descriptive reporting only. Model fallbacks are independently fitted from training observations only.",
            "memory_strategy": "Bounded patient batches; exact medians via one disk-mapped column at a time; metadata-only patient manifests retained in memory.",
            "memory_bounds": {k: v for k, v in (ingestion | transformation).items() if k.startswith("max_")},
            "input_content_sha256": ingestion["input_content_sha256"], "seed": config["seed"],
            "runtime": runtime_versions(),
            "storage_format": config["storage_format"], "output_files": {s: str(output / f"{s}{extension(config['storage_format'])}") for s in SPLITS},
            "elapsed_seconds": round(time.perf_counter() - started, 2), "output_dir": str(output), "reports_dir": str(reports), "warnings": warnings}
        save_json(stage / "pipeline_manifest.json", summary)
        # Publish only verified outputs. Existing results are retained, never recursively deleted.
        if output.exists():
            if any(output.iterdir()):
                backup = output.parent / f"{output.name}.backup-{uuid.uuid4().hex[:12]}"
                output.rename(backup)
                logging.info("Previous results retained at %s", backup)
            else:
                output.rmdir()
        stage.rename(output)
        write_statistics_report(reports, summary, feature_stats)
        logging.info("COMPLETE: patients=%s failed=%s rows=%s elapsed=%.1fs", len(final_manifest), summary["failed_files"], final_rows, summary["elapsed_seconds"])
        return summary
    except Exception as exc:
        save_json(stage / "RUN_FAILED.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started})
        logging.exception("Pipeline failed; incomplete output retained at %s", stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild verified outputs; retain the previous output folder as a backup")
    args = parser.parse_args()
    run_pipeline(load_config(args.config), args.overwrite)


if __name__ == "__main__":
    main()
