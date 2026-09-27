"""Configuration, recursive discovery, strict reading, and bounded Parquet writes."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.storage import StreamingWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "SepsisLabel"
METADATA = {"patient_id", "dataset", TARGET}


def load_config(path=None):
    path = Path(path or PROJECT_ROOT / "config.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("output_dir", "reports_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    config["input_dirs"] = {k: str((path.parent / v).resolve()) for k, v in config["input_dirs"].items()}
    ratios = config["split_ratios"]
    if set(ratios) != {"train", "validation", "test"} or not np.isclose(sum(ratios.values()), 1) or min(ratios.values()) <= 0:
        raise ValueError("split_ratios must contain positive train/validation/test values summing to 1")
    for key in ("batch_size", "max_batch_rows", "max_patient_rows", "max_file_bytes", "progress_every", "validation_batch_rows", "inspection_samples_per_dataset"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_workers"] != 1:
        raise ValueError("This implementation uses one reliable, bounded-memory worker; set num_workers=1")
    if not config["rolling_windows"] or any(not isinstance(w, int) or w < 2 for w in config["rolling_windows"]):
        raise ValueError("rolling_windows must be integer hour windows >= 2")
    if len(set(config["rolling_windows"])) != len(config["rolling_windows"]):
        raise ValueError("rolling_windows must be unique")
    if config["forward_fill_limit_hours"] < 0:
        raise ValueError("forward_fill_limit_hours cannot be negative")
    for key in ("rolling_features", "forward_fill_features"):
        if len(config[key]) != len(set(config[key])) or set(config[key]) & (METADATA | {"ICULOS", "HospAdmTime"}):
            raise ValueError(f"{key} contains duplicate or prohibited target/metadata/time names")
    # Never write into an input directory (or an ancestor containing the inputs).
    output, reports = Path(config["output_dir"]), Path(config["reports_dir"])
    for destination in (output, reports):
        for source in map(Path, config["input_dirs"].values()):
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError("Output/report paths must be separate from source patient folders")
    if output == reports or output in reports.parents or reports in output.parents:
        raise ValueError("output_dir and reports_dir must be separate sibling locations")
    return config


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def setup_logging(reports_dir):
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(Path(reports_dir) / "pipeline.log", mode="a", encoding="utf-8")
    ], force=True)


@dataclass(frozen=True)
class PatientFile:
    patient_id: str
    dataset: str
    path: Path


def discover_files(config):
    records, counts = [], {}
    for dataset, root in sorted(config["input_dirs"].items()):
        folder = Path(root)
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected training folder does not exist: {folder}")
        paths = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".psv"), key=lambda p: str(p).casefold())
        counts[dataset] = len(paths)
        records.extend(PatientFile(p.stem, dataset, p) for p in paths)
    if not records or any(count == 0 for count in counts.values()):
        raise ValueError(f"No PSV files found in one or more input folders: {counts}")
    id_counts = Counter(r.patient_id for r in records)
    duplicates = {p for p, n in id_counts.items() if n > 1}
    warnings = []
    for dataset, actual in counts.items():
        expected = config.get("expected_counts", {}).get(dataset)
        if expected is not None and actual != expected:
            warnings.append(f"Dataset {dataset}: found {actual:,} files; expected {expected:,} (difference {actual - expected:+,}).")
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate patient IDs detected. All conflicting files will be quarantined.")
    return records, {"patients_by_dataset": counts, "total_files": len(records), "duplicate_patient_ids": sorted(duplicates), "warnings": warnings}


def read_header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f, delimiter="|"), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("Missing or duplicate header columns")
    if any(not name or name.strip() != name for name in header):
        raise ValueError("Empty or whitespace-padded header column")
    return header


def read_patient(record, schema, config):
    if record.path.stat().st_size > config["max_file_bytes"]:
        raise ValueError("File exceeds configured max_file_bytes")
    payload = record.path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    content = payload.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(content), delimiter="|")
    header = next(reader, [])
    if header != list(schema):
        raise ValueError(f"Schema mismatch: expected {list(schema)}, got {header}")
    row_count = 0
    for line_no, row in enumerate(reader, 2):
        if len(row) != len(schema):
            raise ValueError(f"Malformed row {line_no}: expected {len(schema)} fields, got {len(row)}")
        row_count += 1
        if row_count > config["max_patient_rows"]:
            raise ValueError("Patient exceeds configured max_patient_rows")
    if not row_count:
        raise ValueError("Empty patient file")
    frame = pd.read_csv(io.StringIO(content), sep="|", dtype=np.float64, on_bad_lines="error")
    if len(frame) != row_count:
        raise ValueError("Parsed row count differs from validated row count")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Infinite raw numerical value")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("SepsisLabel must be complete and binary (0/1)")
    hours = frame["ICULOS"].to_numpy()
    if np.isnan(hours).any() or (hours <= 0).any() or not np.equal(hours, np.floor(hours)).all() or not np.equal(np.diff(hours), 1).all():
        duplicates = int(frame.duplicated().sum())
        raise ValueError(f"ICULOS must be positive consecutive integer hours in original file order; exact duplicate rows={duplicates}")
    for col in ("Gender", "Unit1", "Unit2"):
        if col in frame and not frame[col].dropna().isin([0, 1]).all():
            raise ValueError(f"{col} has a value outside 0/1/NaN")
    # A strictly increasing ICULOS makes full-row duplicates impossible. Avoid
    # factorizing all 41 columns for every valid patient; inspect malformed time
    # records explicitly above instead.
    duplicate_rows = 0
    frame[TARGET] = frame[TARGET].astype("int8")
    frame.insert(0, "dataset", record.dataset)
    frame.insert(0, "patient_id", record.patient_id)
    return frame, digest, duplicate_rows


BoundedParquetWriter = StreamingWriter  # Backward-compatible name; format follows the actual suffix.
