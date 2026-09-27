"""Bounded, validated model inputs backed by NumPy files on disk.

The input guard hashes small manifests/ID lists and records source size and
nanosecond mtime. It is a change detector, not a cryptographic source-data hash.
Training matrices are float32 memory maps; a full dataset is never concatenated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import uuid

import numpy as np

from src.storage import iter_batches

SPLITS = ("train", "validation", "test")
ARRAYS = ("X", "y", "patient_id", "ICULOS", "dataset")
CACHE_VERSION = 1
LOGGER = logging.getLogger(__name__)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stat(path):
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _forbidden_feature(name):
    name = name.lower()
    return (name in {"patient_id", "dataset", "split", "n_rows", "positive_rows", "positive_patient"}
            or any(word in name for word in ("sepsislabel", "time_to_sepsis", "onset"))
            or bool(re.search(r"(^|_)(target|label|future|lead)(_|$)", name)))


def _load_inputs(processed_dir):
    feature_path = processed_dir / "feature_metadata.json"
    pipeline_path = processed_dir / "pipeline_manifest.json"
    feature_info, pipeline = _read_json(feature_path), _read_json(pipeline_path)
    features = feature_info["predictor_columns"]
    if (not isinstance(features, list) or not features
            or any(not isinstance(value, str) or not value for value in features)
            or len(set(features)) != len(features)):
        raise ValueError("predictor_columns must be a nonempty list of unique feature names")
    for name in features:
        details = feature_info.get("features", {}).get(name, {})
        sources = [name, details.get("source"), details.get("fallback_source")]
        if any(_forbidden_feature(source) for source in sources if isinstance(source, str)):
            raise ValueError(f"Target, metadata, or future-derived predictor is prohibited: {name}")
    if "ICULOS" not in features:
        raise ValueError("Predictor metadata must retain the ICULOS time coordinate")
    if pipeline.get("status") != "complete" or pipeline.get("validation_passed") is not True:
        raise ValueError("Preprocessing manifest must report a complete, validated dataset")

    small_files = [feature_path, pipeline_path]
    sources, memberships = {}, {}
    all_ids = set()
    for split in SPLITS:
        candidates = [processed_dir / f"{split}{suffix}" for suffix in (".parquet", ".csv.gz")]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise ValueError(f"Expected exactly one {split}.parquet or {split}.csv.gz; found {len(existing)}")
        sources[split] = existing[0]
        id_path = processed_dir / f"{split}_patient_ids.csv"
        small_files.append(id_path)
        members = {}
        with id_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"patient_id", "dataset"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing patient_id/dataset columns: {id_path}")
            for row in reader:
                patient, dataset = row["patient_id"], row["dataset"]
                if not patient or len(patient) > 64 or dataset not in {"A", "B"}:
                    raise ValueError(f"Invalid patient or dataset in {id_path}: {patient!r}")
                if patient in members or patient in all_ids:
                    raise ValueError(f"Duplicate or overlapping patient ID: {patient}")
                members[patient] = dataset
        expected = pipeline["splits"][split]
        if not members or len(members) != expected["patients"]:
            raise ValueError(f"{split} patient ID count disagrees with pipeline manifest")
        if expected["rows"] <= 0 or expected["positive_rows"] + expected["negative_rows"] != expected["rows"]:
            raise ValueError(f"Invalid expected row counts for {split}")
        memberships[split] = members
        all_ids.update(members)

    patient_details = {}
    patient_path = processed_dir / "patient_manifest.csv"
    if patient_path.is_file():
        small_files.append(patient_path)
        with patient_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                patient = row["patient_id"]
                if patient in patient_details:
                    raise ValueError(f"Duplicate patient manifest ID: {patient}")
                patient_details[patient] = row
        if set(patient_details) != all_ids:
            raise ValueError("Patient manifest and split patient IDs disagree")

    fingerprint_details = {
        "version": CACHE_VERSION,
        "source_dir": str(processed_dir),
        "manifests": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in small_files},
        "source_files": {split: {"path": str(path), **_stat(path)} for split, path in sources.items()},
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_details, sort_keys=True).encode()).hexdigest()
    return features, pipeline, sources, memberships, patient_details, fingerprint, fingerprint_details


def close_arrays(arrays):
    """Release disk mappings promptly (particularly important on Windows)."""
    for array in arrays.values():
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _check_cache(cache_dir, metadata):
    if metadata.get("version") != CACHE_VERSION:
        raise ValueError("Unsupported model cache version; rebuild the cache")
    for split in SPLITS:
        rows = metadata["splits"][split]["rows"]
        for name in ARRAYS:
            path = cache_dir / split / f"{name}.npy"
            expected = metadata["arrays"][split][name]
            if _stat(path) != expected["stat"]:
                raise ValueError(f"Changed cache array {path}; rebuild the cache")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                shape = (rows, len(metadata["features"])) if name == "X" else (rows,)
                if array.shape != shape or str(array.dtype) != expected["dtype"]:
                    raise ValueError(f"Invalid shape or dtype in cache array: {path}")
                if name == "X" and array.dtype != np.float32:
                    raise ValueError("Cached predictors must be float32")
                if name == "y" and array.dtype != np.uint8:
                    raise ValueError("Cached targets must be uint8")
                if name == "ICULOS" and array.dtype != np.float64:
                    raise ValueError("Cached time must be float64")
                if name in {"patient_id", "dataset"} and array.dtype.kind != "U":
                    raise ValueError("Cached identifiers must be Unicode arrays")
            finally:
                close_arrays({name: array})


def _build_split(split, path, stage, features, expected, members, patient_details, batch_rows):
    directory = stage / split
    directory.mkdir()
    rows_expected = int(expected["rows"])
    dtypes = {"X": np.float32, "y": np.uint8,
              "patient_id": f"U{max(map(len, members))}", "ICULOS": np.float64, "dataset": "U1"}
    for name, dtype in dtypes.items():
        shape = (rows_expected, len(features)) if name == "X" else (rows_expected,)
        array = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        close_arrays({name: array})
    count = positives = 0
    patient_stats = {}
    expected_columns = set(features) | {"patient_id", "dataset", "SepsisLabel"}
    for batch_number, frame in enumerate(iter_batches(path, batch_size=batch_rows), 1):
        if set(frame.columns) != expected_columns or frame.columns.duplicated().any():
            missing = sorted(expected_columns - set(frame.columns))
            extra = sorted(set(frame.columns) - expected_columns)
            raise ValueError(f"{split} feature schema mismatch: missing={missing}, extra={extra}")
        if count + len(frame) > rows_expected:
            raise ValueError(f"{split} has more rows than the pipeline manifest")
        labels = frame["SepsisLabel"]
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{split} has invalid SepsisLabel values")
        # Named selection enforces metadata order independently of the stored column order.
        with np.errstate(over="ignore", invalid="ignore"):
            values = frame[features].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{split} contains missing/infinite predictors or float32 overflow")
        hours = frame["ICULOS"].to_numpy(dtype=np.float64)
        if not np.isfinite(hours).all():
            raise ValueError(f"{split} has missing/infinite ICULOS")
        if frame["patient_id"].isna().any() or frame["dataset"].isna().any():
            raise ValueError(f"{split} has missing patient metadata")
        for patient, group in frame.groupby("patient_id", sort=False):
            if patient not in members:
                raise ValueError(f"Patient {patient} does not belong to {split}; possible split leakage")
            if set(group["dataset"]) != {members[patient]}:
                raise ValueError(f"Dataset source mismatch for {patient}")
            group_hours = group["ICULOS"].to_numpy(dtype=float)
            state = patient_stats.setdefault(patient, [0, 0, None, float(group_hours[0])])
            if (not np.equal(np.diff(group_hours), 1).all()
                    or (state[2] is not None and group_hours[0] != state[2] + 1)):
                raise ValueError(f"Hourly order/continuity changed for {patient}")
            state[0] += len(group)
            state[1] += int(group["SepsisLabel"].sum())
            state[2] = float(group_hours[-1])
        arrays = {}
        try:
            # Close/reopen per batch so resident mapped pages cannot accumulate across a split.
            arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r+", allow_pickle=False)
                      for name in ARRAYS}
            location = slice(count, count + len(frame))
            arrays["X"][location] = values
            arrays["y"][location] = labels.to_numpy(dtype=np.uint8)
            arrays["patient_id"][location] = frame["patient_id"].to_numpy()
            arrays["dataset"][location] = frame["dataset"].to_numpy()
            arrays["ICULOS"][location] = hours
            for array in arrays.values():
                array.flush()
        finally:
            close_arrays(arrays)
        count += len(frame)
        positives += int(labels.sum())
        if batch_number % 20 == 0:
            LOGGER.info("Building %s model cache: %s/%s rows", split, count, rows_expected)
    result = {"rows": count, "patients": len(patient_stats), "positive_rows": positives,
              "negative_rows": count - positives,
              "positive_patients": sum(state[1] > 0 for state in patient_stats.values()),
              "negative_patients": sum(state[1] == 0 for state in patient_stats.values())}
    if set(patient_stats) != set(members):
        raise ValueError(f"{split} actual patient coverage disagrees with split IDs")
    for key, value in result.items():
        if key in expected and value != expected[key]:
            raise ValueError(f"{split} {key} differs: actual={value}, manifest={expected[key]}")
    for patient, state in patient_stats.items():
        if patient in patient_details:
            record = patient_details[patient]
            if state[0] != int(record["n_rows"]) or state[1] != int(record["positive_rows"]):
                raise ValueError(f"Patient row/target counts changed: {patient}")
            if "first_ICULOS" in record and (state[3] != float(record["first_ICULOS"])
                                             or state[2] != float(record["last_ICULOS"])):
                raise ValueError(f"Patient time bounds changed: {patient}")
    array_metadata = {}
    for name in ARRAYS:
        array_metadata[name] = {"dtype": str(np.dtype(dtypes[name])),
                                "stat": _stat(directory / f"{name}.npy")}
    LOGGER.info("Validated %s model cache: %s patients, %s rows", split, result["patients"], count)
    return result, array_metadata


def build_cache(processed_dir: Path, cache_dir: Path, batch_rows=8192, rebuild=False) -> dict:
    """Validate and cache the three patient-disjoint splits using bounded batches.

    Stale/incomplete caches fail clearly unless ``rebuild=True``. Publication
    happens only after all splits pass validation. Old caches are retained under
    a uniquely named sibling backup; source datasets are never altered.
    """
    processed_dir, cache_dir = Path(processed_dir).resolve(), Path(cache_dir).resolve()
    if not isinstance(batch_rows, int) or isinstance(batch_rows, bool) or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    if cache_dir == processed_dir or cache_dir in processed_dir.parents:
        raise ValueError("Cache directory cannot replace the processed data directory or its ancestors")
    features, pipeline, sources, memberships, patients, fingerprint, details = _load_inputs(processed_dir)
    if cache_dir.exists() and not rebuild:
        try:
            metadata = _read_json(cache_dir / "cache_manifest.json")
            if metadata["fingerprint"] != fingerprint:
                raise ValueError("Source dataset/configuration fingerprint changed")
            _check_cache(cache_dir, metadata)
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"Model cache is stale or invalid ({error}); rerun with --rebuild-cache") from error
        LOGGER.info("Reusing validated model cache: %s", cache_dir)
        return metadata
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = cache_dir.with_name(f"{cache_dir.name}.building-{uuid.uuid4().hex}")
    stage.mkdir()
    metadata = {"version": CACHE_VERSION, "features": features, "splits": {}, "arrays": {},
                "fingerprint": fingerprint, "fingerprint_details": details,
                "source_dir": str(processed_dir), "batch_rows": batch_rows,
                "source_guard": "SHA256 of manifests/ID lists; source file path, byte size, nanosecond mtime",
                "memory_strategy": "Bounded pandas batches and float32 disk arrays; mappings closed per batch"}
    try:
        for split in SPLITS:
            metadata["splits"][split], metadata["arrays"][split] = _build_split(
                split, sources[split], stage, features, pipeline["splits"][split],
                memberships[split], patients, batch_rows)
        # Refuse publication if input manifests or source files changed during construction.
        if _load_inputs(processed_dir)[5] != fingerprint:
            raise ValueError("Processed inputs changed while building the model cache")
        (stage / "cache_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _check_cache(stage, metadata)
    except Exception:
        LOGGER.exception("Cache build failed; incomplete cache retained for inspection at %s", stage)
        raise
    backup = None
    if cache_dir.exists():
        backup = cache_dir.with_name(f"{cache_dir.name}.backup-{uuid.uuid4().hex}")
        cache_dir.rename(backup)
    try:
        stage.rename(cache_dir)
    except Exception:
        if backup is not None and not cache_dir.exists():
            backup.rename(cache_dir)
        raise
    LOGGER.info("Model cache ready: %s", cache_dir)
    return metadata


def open_split(cache_dir: Path, split: str) -> dict:
    """Return read-only memory maps; caller should use ``close_arrays`` on completion."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    cache_dir = Path(cache_dir)
    metadata = _read_json(cache_dir / "cache_manifest.json")
    _check_cache(cache_dir, metadata)
    result = {}
    try:
        for name in ARRAYS:
            result[name] = np.load(cache_dir / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    except Exception:
        close_arrays(result)
        raise
    return result
