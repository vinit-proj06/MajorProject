"""Streaming descriptive statistics and strictly training-only fallback fitting.

Finite observations are spooled by column to private temporary files. Exact
medians partition a single disk-backed array in place, never a dataset-sized
DataFrame. Descriptive statistics must never be reused as model parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import iter_batches, raw_chunk_paths

LOGGER = logging.getLogger(__name__)
STATISTIC_FIELDS = [
    "column", "total_values", "missing_values", "missing_percentage",
    "min", "max", "mean", "median",
]
CATEGORICAL_COLUMNS = ("Gender", "Unit1", "Unit2")


class DiskStatistics:
    """Accumulate exact raw numeric statistics with bounded explicit RAM.

    The operating system may cache mapped pages. The largest virtual mapping is
    one feature, and there is no full-dataset heap allocation. Scratch disk must
    accommodate eight bytes per observed numeric value until ``finalize``.
    A private temporary subdirectory prevents removal of unrelated files.
    """

    def __init__(self, columns: list[str], scratch_dir: Path):
        if len(columns) != len(set(columns)):
            raise ValueError("Statistics columns must be unique")
        self.columns = list(columns)
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self._workdir = Path(tempfile.mkdtemp(prefix="statistics_", dir=scratch_dir))
        self._states: dict[str, dict[str, Any]] = {}
        for index, column in enumerate(self.columns):
            path = self._workdir / f"column_{index:04d}.bin"
            self._states[column] = {
                "total": 0, "count": 0, "sum": 0.0,
                "min": math.inf, "max": -math.inf,
                "path": path, "handle": path.open("wb", buffering=65536),
            }
        self._result: list[dict[str, Any]] | None = None
        self._closed = False

    def update(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError("Cannot update finalized/closed statistics")
        missing = set(self.columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing statistics columns: {sorted(missing)}")
        # Check the whole incoming batch before changing accumulators.
        for column in self.columns:
            values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f"Infinite values in statistics column {column}")
        for column in self.columns:
            state = self._states[column]
            values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            valid = values[~np.isnan(values)]
            state["total"] += len(values)
            state["count"] += len(valid)
            if len(valid):
                state["sum"] += float(np.sum(valid, dtype=np.float64))
                state["min"] = min(state["min"], float(valid.min()))
                state["max"] = max(state["max"], float(valid.max()))
                state["handle"].write(valid.tobytes(order="C"))

    def finalize(self) -> list[dict[str, Any]]:
        if self._result is not None:
            return [dict(row) for row in self._result]
        if self._closed:
            raise RuntimeError("Statistics were closed without finalizing")
        result: list[dict[str, Any]] = []
        try:
            for column in self.columns:
                state = self._states[column]
                state["handle"].close()
                count, total = state["count"], state["total"]
                median = None
                if count:
                    mapped = np.memmap(state["path"], dtype=np.float64, mode="r+", shape=(count,))
                    try:
                        upper = count // 2
                        if count % 2:
                            mapped.partition(upper)
                            median = float(mapped[upper])
                        else:
                            mapped.partition((upper - 1, upper))
                            median = float(mapped[upper - 1]) / 2 + float(mapped[upper]) / 2
                    finally:
                        mapped.flush()
                        # Explicit close is needed before unlinking on Windows.
                        mapped._mmap.close()
                        del mapped
                result.append({
                    "column": column,
                    "total_values": total,
                    "missing_values": total - count,
                    "missing_percentage": (100.0 * (total - count) / total) if total else None,
                    "min": state["min"] if count else None,
                    "max": state["max"] if count else None,
                    "mean": state["sum"] / count if count else None,
                    "median": median,
                })
            self._result = result
        finally:
            self.close()
        return [dict(row) for row in result]

    def close(self) -> None:
        """Close handles and delete only this accumulator's scratch directory."""
        if self._closed:
            return
        for state in self._states.values():
            if not state["handle"].closed:
                state["handle"].close()
        # _workdir is an exclusively owned mkdtemp path, not user-derived input.
        shutil.rmtree(self._workdir)
        self._closed = True

    def __enter__(self) -> "DiskStatistics":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def iter_raw_batches(raw_dir: Path, columns: list[str] | None = None,
                     batch_size: int = 20_000) -> Iterator[pd.DataFrame]:
    """Read bounded batches from Parquet or compressed CSV raw chunks."""
    paths = raw_chunk_paths(Path(raw_dir))
    if not paths:
        raise FileNotFoundError(f"No raw Parquet/compressed CSV chunks in {raw_dir}")
    for path in paths:
        yield from iter_batches(path, batch_size=batch_size, columns=columns)


def fit_training_parameters(raw_dir: Path, train_ids: set[str], columns: list[str],
                            scratch_dir: Path) -> dict[str, Any]:
    """Fit fallback medians/binary modes from observed training values only.

    ``columns`` must be raw predictor columns. Even if a caller includes target
    or metadata, they are excluded defensively. Missing-only training columns
    are returned as ``None`` and explicitly listed so callers can drop them.
    """
    excluded = {"patient_id", "dataset", "SepsisLabel", "split"}
    predictors = [column for column in columns if column not in excluded]
    if not train_ids:
        raise ValueError("Training patient ID set is empty")
    mode_counts: dict[str, dict[float, int]] = {
        column: {} for column in CATEGORICAL_COLUMNS if column in predictors
    }
    seen_ids: set[str] = set()
    with DiskStatistics(predictors, scratch_dir) as accumulator:
        for frame in iter_raw_batches(raw_dir, ["patient_id", *predictors]):
            frame = frame.loc[frame["patient_id"].isin(train_ids)]
            if frame.empty:
                continue
            seen_ids.update(frame["patient_id"].unique().tolist())
            accumulator.update(frame)
            for column, counts in mode_counts.items():
                values, frequencies = np.unique(frame[column].dropna().to_numpy(dtype=np.float64),
                                                return_counts=True)
                for value, frequency in zip(values, frequencies):
                    counts[float(value)] = counts.get(float(value), 0) + int(frequency)
        missing_ids = train_ids.difference(seen_ids)
        if missing_ids:
            raise ValueError(f"{len(missing_ids)} training patients absent from raw chunks; "
                             f"examples: {sorted(missing_ids)[:5]}")
        statistics = accumulator.finalize()
    values = {row["column"]: row["median"] for row in statistics}
    for column, counts in mode_counts.items():
        if counts:
            values[column] = min(counts, key=lambda value: (-counts[value], value))
    return {
        "fit_split": "train",
        "training_patient_count": len(seen_ids),
        "method": "observed training median; observed training mode for Gender/Unit1/Unit2",
        "values": values,
        "all_missing_columns": [column for column, value in values.items() if value is None],
        "statistics": statistics,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_statistics_report(reports_dir: Path, summary: dict[str, Any],
                            feature_statistics: list[dict[str, Any]]) -> dict[str, Any]:
    """Write machine-readable and readable statistics plus separate CSV tables."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = _json_safe({**summary, "feature_statistics": feature_statistics})
    (reports_dir / "dataset_statistics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = ["Dataset statistics", "", "Summary", json.dumps(_json_safe(summary), indent=2),
            "", "Raw feature statistics (observed values; exact medians)"]
    if feature_statistics:
        text.append(pd.DataFrame(feature_statistics).to_string(index=False))
    text.append("\nDescriptive statistics include all valid patients and are not model-fitting parameters.")
    (reports_dir / "dataset_statistics.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    for filename, fields in (
        ("missingness.csv", STATISTIC_FIELDS[:4]),
        ("numerical_statistics.csv", STATISTIC_FIELDS),
    ):
        with (reports_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(feature_statistics)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.json")
    parser.add_argument("--regenerate", action="store_true",
                        help="Recompute raw statistics from streamed Parquet/compressed CSV chunks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Both documented direct invocation and python -m src.dataset_statistics work.
    from src.common import load_config
    config = load_config(args.config)
    reports_dir = Path(config["reports_dir"])
    saved_path = reports_dir / "dataset_statistics.json"
    if not saved_path.exists():
        raise SystemExit("Statistics are not available yet. Run python run_pipeline.py first.")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if args.regenerate:
        raw_dir = Path(config["output_dir"]) / "raw_processed"
        columns = [row["column"] for row in saved["feature_statistics"]]
        LOGGER.info("Regenerating exact descriptive statistics from %s", raw_dir)
        with DiskStatistics(columns, Path(config["output_dir"]) / ".statistics_scratch") as statistics:
            for frame in iter_raw_batches(raw_dir, columns):
                statistics.update(frame)
            feature_statistics = statistics.finalize()
        summary = {key: value for key, value in saved.items() if key != "feature_statistics"}
        write_statistics_report(reports_dir, summary, feature_statistics)
    LOGGER.info("%s", (reports_dir / "dataset_statistics.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
