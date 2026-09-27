"""Storage fallback, precision, and memory-bound invariants."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import storage


CONFIG = {"storage_format": "csv_gzip", "batch_size": 2, "max_batch_rows": 3}


class StorageTests(unittest.TestCase):
    def test_csv_streaming_round_trip_and_bounds(self):
        source = pd.DataFrame({"patient_id": ["000001"] * 7, "dataset": ["A"] * 7,
                               "ICULOS": np.arange(1, 8),
                               "HR": [np.nan, np.pi, -1.2345678901234567, 0., 1e-100, 1e100, 90.],
                               "SepsisLabel": [0, 0, 0, 1, 1, 1, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv.gz"
            writer = storage.StreamingWriter(path, CONFIG)
            writer.add(source.iloc[:2])
            writer.add(source.iloc[2:])
            writer.close()
            writer.close()
            self.assertEqual(writer.total_rows, 7)
            self.assertLessEqual(writer.max_buffer_rows, 3)
            self.assertLessEqual(writer.max_buffer_patients, 2)
            result = storage.read_chunk(path)
            self.assertEqual(result["patient_id"].tolist(), ["000001"] * 7)
            np.testing.assert_array_equal(result["HR"], source["HR"])
            batches = list(storage.iter_batches(path, 2, columns=["HR", "patient_id"]))
            self.assertTrue(all(len(batch) <= 2 for batch in batches))
            self.assertTrue(all(list(batch) == ["HR", "patient_id"] for batch in batches))
            self.assertEqual(sum(len(batch) for batch in batches), 7)

    def test_fallback_import_is_cached_and_explicit_parquet_fails(self):
        with patch.object(storage, "_PARQUET_CHECKED", False), \
             patch.object(storage, "_PARQUET_MODULE", None), \
             patch.object(storage, "_PARQUET_ERROR", None), \
             patch.object(storage.importlib, "import_module", side_effect=OSError("Host denied dependency")) as importer:
            name, warning = storage.resolve_storage({})
            self.assertEqual(name, "csv_gzip")
            self.assertIn("Host denied dependency", warning)
            storage.resolve_storage({})
            with self.assertRaises(ImportError):
                storage.resolve_storage({"storage_format": "parquet"})
            self.assertEqual(importer.call_count, 1)

    def test_explicit_csv_does_not_probe_parquet(self):
        with patch.object(storage.importlib, "import_module") as importer:
            self.assertEqual(storage.resolve_storage(CONFIG), ("csv_gzip", None))
            importer.assert_not_called()

    def test_truthful_extensions_and_chunk_discovery(self):
        self.assertEqual(storage.extension("csv_gzip"), ".csv.gz")
        self.assertEqual(storage.extension("parquet"), ".parquet")
        with tempfile.TemporaryDirectory() as directory:
            for name in ("chunk_002.csv.gz", "chunk_001.parquet", "notes.txt"):
                (Path(directory) / name).touch()
            self.assertEqual([path.name for path in storage.raw_chunk_paths(directory)],
                             ["chunk_001.parquet", "chunk_002.csv.gz"])

    def test_output_deterministic_and_empty_header_supported(self):
        frame = pd.DataFrame({"patient_id": ["p1"], "dataset": ["A"], "HR": [1.25]})
        with tempfile.TemporaryDirectory() as directory:
            first, second, empty = [Path(directory) / name for name in ("a.csv.gz", "b.csv.gz", "empty.csv.gz")]
            storage.write_chunk(frame, first, CONFIG)
            storage.write_chunk(frame, second, CONFIG)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            storage.write_chunk(frame.iloc[:0], empty, CONFIG)
            self.assertEqual(list(storage.read_chunk(empty)), list(frame))


if __name__ == "__main__":
    unittest.main()
