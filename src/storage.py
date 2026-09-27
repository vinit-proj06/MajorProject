"""Bounded dataset I/O with a transparent compressed-CSV fallback.

Parquet support is imported lazily. If its native dependency is unavailable or
blocked by the host, ``auto`` mode uses ordinary gzip CSV without changing the
host's security configuration or pretending CSV files are Parquet files.
"""

from __future__ import annotations

import gzip
import importlib
import io
from pathlib import Path
from typing import Iterator

import pandas as pd


_PARQUET_CHECKED = False
_PARQUET_MODULE = None
_PARQUET_ERROR: ImportError | OSError | None = None


def _parquet():
    global _PARQUET_CHECKED, _PARQUET_MODULE, _PARQUET_ERROR
    if not _PARQUET_CHECKED:
        _PARQUET_CHECKED = True
        try:
            _PARQUET_MODULE = importlib.import_module("pyarrow.parquet")
        except (ImportError, OSError) as error:
            _PARQUET_ERROR = error
    if _PARQUET_MODULE is None:
        raise ImportError(f"Parquet support is unavailable: {_PARQUET_ERROR}") from _PARQUET_ERROR
    return _PARQUET_MODULE


def resolve_storage(config: dict) -> tuple[str, str | None]:
    """Choose actual storage format and report any auto-mode fallback."""
    requested = config.get("storage_format", "auto")
    if requested not in {"auto", "parquet", "csv_gzip"}:
        raise ValueError("storage_format must be auto, parquet, or csv_gzip")
    if requested == "csv_gzip":
        return "csv_gzip", None
    try:
        _parquet()
    except (ImportError, OSError) as error:
        if requested == "parquet":
            raise
        return "csv_gzip", (f"Parquet dependency unavailable ({error}). "
                            "Using bounded gzip-compressed CSV (.csv.gz) instead; "
                            "host application-control settings have not been changed.")
    return "parquet", None


def extension(format_name: str) -> str:
    if format_name == "parquet":
        return ".parquet"
    if format_name == "csv_gzip":
        return ".csv.gz"
    raise ValueError(f"Unknown storage format: {format_name}")


def _format(path: Path) -> str:
    if path.name.lower().endswith(".parquet"):
        return "parquet"
    if path.name.lower().endswith(".csv.gz"):
        return "csv_gzip"
    raise ValueError(f"Expected a .parquet or .csv.gz path: {path}")


def _csv_options(columns=None) -> dict:
    identifiers = ("patient_id", "dataset")
    return {
        "compression": "gzip", "float_precision": "round_trip",
        "dtype": {column: str for column in identifiers if columns is None or column in columns},
        "usecols": columns,
    }


def write_chunk(frame: pd.DataFrame, path, config: dict) -> None:
    """Write one bounded intermediate frame in the format named by its suffix."""
    writer = StreamingWriter(path, config)
    try:
        writer.add(frame)
        writer.close()
    finally:
        writer._close_handles()


def read_chunk(path) -> pd.DataFrame:
    """Read one already-bounded raw chunk; use iter_batches for final outputs."""
    path = Path(path)
    if _format(path) == "csv_gzip":
        return pd.read_csv(path, **_csv_options())
    return _parquet().ParquetFile(path).read().to_pandas()


def iter_batches(path, batch_size: int, columns=None) -> Iterator[pd.DataFrame]:
    """Read either output format without loading the entire dataset."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    path = Path(path)
    if columns is not None:
        columns = list(columns)
    if _format(path) == "csv_gzip":
        with pd.read_csv(path, chunksize=batch_size, **_csv_options(columns)) as reader:
            for frame in reader:
                # pandas usecols follows file order; make both formats consistent.
                yield frame if columns is None else frame[columns]
    else:
        parquet_file = _parquet().ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()


def raw_chunk_paths(raw_dir) -> list[Path]:
    """Return only supported chunks, ordered independently of filesystem order."""
    return sorted((path for path in Path(raw_dir).iterdir()
                   if path.is_file() and (path.name.lower().endswith(".parquet")
                                          or path.name.lower().endswith(".csv.gz"))),
                  key=lambda path: path.name)


class StreamingWriter:
    """Concatenate once per bounded batch and stream it to a single output file."""

    def __init__(self, path, config: dict):
        self.path = Path(path)
        self.config = config
        self.format_name = _format(self.path)
        self.batch_size = int(config.get("batch_size", 128))
        self.max_rows = int(config.get("max_batch_rows", 20000))
        if self.batch_size < 1 or self.max_rows < 1:
            raise ValueError("batch_size and max_batch_rows must be positive")
        self.frames: list[pd.DataFrame] = []
        self.rows = 0
        self.total_rows = 0
        self.max_buffer_rows = 0
        self.max_buffer_patients = 0
        self.writer = None
        self._raw_handle = None
        self._columns = None
        self._empty_template = None
        self._header_written = False
        self._closed = False

    def add(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise ValueError("Cannot add to a closed writer")
        columns = list(frame.columns)
        if self._columns is None:
            self._columns = columns
            self._empty_template = frame.iloc[:0].copy()
        elif columns != self._columns:
            raise ValueError("Column schema/order changed between output batches")
        if frame.empty:
            return
        # Very long patients are sliced only for storage. Feature computation has
        # already completed and patient identity/time order remain unchanged.
        for start in range(0, len(frame), self.max_rows):
            part = frame.iloc[start:start + self.max_rows]
            if self.frames and (len(self.frames) >= self.batch_size or self.rows + len(part) > self.max_rows):
                self.flush()
            self.frames.append(part)
            self.rows += len(part)
            self.max_buffer_rows = max(self.max_buffer_rows, self.rows)
            self.max_buffer_patients = max(self.max_buffer_patients, len(self.frames))
            if self.rows >= self.max_rows or len(self.frames) >= self.batch_size:
                self.flush()

    def _write(self, frame: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.format_name == "csv_gzip":
            if self.writer is None:
                self._raw_handle = self.path.open("wb")
                gzip_handle = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw_handle,
                                           compresslevel=int(self.config.get("gzip_compresslevel", 1)), mtime=0)
                self.writer = io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="")
            frame.to_csv(self.writer, index=False, header=not self._header_written, lineterminator="\n")
            self._header_written = True
        else:
            parquet = _parquet()
            arrow = importlib.import_module("pyarrow")
            table = arrow.Table.from_pandas(frame, preserve_index=False)
            if self.writer is None:
                self.writer = parquet.ParquetWriter(self.path, table.schema,
                                                   compression=self.config.get("compression", "zstd"))
            self.writer.write_table(table, row_group_size=self.max_rows)

    def flush(self) -> None:
        if not self.frames:
            return
        combined = pd.concat(self.frames, ignore_index=True)
        self._write(combined)
        self.total_rows += self.rows
        self.frames.clear()
        self.rows = 0

    def _close_handles(self) -> None:
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            if self._raw_handle is not None:
                self._raw_handle.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
            if self.writer is None and self._empty_template is not None:
                self._write(self._empty_template)
        finally:
            self._close_handles()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            self.close()
        else:
            self._close_handles()
            self._closed = True


BoundedParquetWriter = StreamingWriter
