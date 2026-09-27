"""Meaningful checks for exact streaming statistics and train-only fitting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_statistics import DiskStatistics, fit_training_parameters
from src.storage import write_chunk


class StatisticsTests(unittest.TestCase):
    def test_exact_medians_and_cleanup_across_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "keep.txt"
            sentinel.write_text("unrelated")
            stats = DiskStatistics(["even", "odd", "empty"], root)
            stats.update(pd.DataFrame({"even": [9.0, 1.0], "odd": [5.0, np.nan], "empty": [np.nan, np.nan]}))
            stats.update(pd.DataFrame({"even": [4.0, 2.0], "odd": [3.0, 8.0], "empty": [np.nan, np.nan]}))
            result = {row["column"]: row for row in stats.finalize()}
            self.assertEqual(result["even"]["median"], 3.0)
            self.assertEqual(result["odd"]["median"], 5.0)
            self.assertEqual(result["odd"]["missing_percentage"], 25.0)
            self.assertIsNone(result["empty"]["median"])
            self.assertIsNone(result["empty"]["min"])
            self.assertEqual(list(root.iterdir()), [sentinel])
            self.assertEqual(len(stats.finalize()), 3)

    def test_infinity_rejected_without_partial_update(self):
        with tempfile.TemporaryDirectory() as directory:
            with DiskStatistics(["a", "b"], Path(directory)) as stats:
                with self.assertRaisesRegex(ValueError, "Infinite"):
                    stats.update(pd.DataFrame({"a": [1.0], "b": [np.inf]}))
                result = stats.finalize()
                self.assertEqual(result[0]["total_values"], 0)

    def test_fit_ignores_extreme_holdout_values_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            frame = pd.DataFrame({
                "patient_id": ["train", "heldout", "train", "heldout"],
                "HR": [60.0, 1e9, 80.0, 2e9],
                "Gender": [1.0, 1.0, 0.0, 1.0],
                "absent": [np.nan, 10.0, np.nan, 20.0],
                "SepsisLabel": [0, 1, 0, 1],
            })
            write_chunk(frame, raw / "chunk_001.csv.gz", {"storage_format": "csv_gzip"})
            fit = fit_training_parameters(raw, {"train"},
                                          ["HR", "Gender", "absent", "SepsisLabel"], root / "scratch")
            self.assertEqual(fit["values"]["HR"], 70.0)
            self.assertEqual(fit["values"]["Gender"], 0.0)
            self.assertEqual(fit["all_missing_columns"], ["absent"])
            self.assertNotIn("SepsisLabel", fit["values"])
            self.assertEqual(fit["training_patient_count"], 1)

    def test_missing_training_patient_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            frame = pd.DataFrame({"patient_id": ["other"], "HR": [70.0]})
            write_chunk(frame, raw / "chunk.csv.gz", {"storage_format": "csv_gzip"})
            with self.assertRaisesRegex(ValueError, "absent from raw"):
                fit_training_parameters(raw, {"train"}, ["HR"], root / "scratch")


if __name__ == "__main__":
    unittest.main()
