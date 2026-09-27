"""Model cache integrity checks with small, deliberately reordered input tables."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.model_data import build_cache, close_arrays, open_split


class ModelDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()
        self.cache = self.root / "cache"
        self.features = ["HR", "ICULOS", "HR__missing"]
        (self.processed / "feature_metadata.json").write_text(
            json.dumps({"predictor_columns": self.features}), encoding="utf-8")
        manifest = {"status": "complete", "validation_passed": True, "splits": {}}
        for split, patient in zip(("train", "validation", "test"), ("p001", "p002", "p003")):
            frame = pd.DataFrame({"patient_id": [patient] * 3, "dataset": ["A"] * 3,
                                  "ICULOS": [1, 2, 3], "SepsisLabel": [0, 0, 1],
                                  "HR__missing": [0, 1, 0], "HR": [70.0, 71.0, 72.0]})
            frame.to_csv(self.processed / f"{split}.csv.gz", index=False)
            pd.DataFrame({"patient_id": [patient], "dataset": ["A"]}).to_csv(
                self.processed / f"{split}_patient_ids.csv", index=False)
            manifest["splits"][split] = {"rows": 3, "patients": 1, "positive_rows": 1,
                                          "negative_rows": 2, "positive_patients": 1,
                                          "negative_patients": 0}
        (self.processed / "pipeline_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_bounded_cache_reorders_predictors_and_reuses(self):
        metadata = build_cache(self.processed, self.cache, batch_rows=2)
        self.assertEqual(metadata["features"], self.features)
        self.assertEqual(metadata["splits"]["train"]["positive_rows"], 1)
        arrays = open_split(self.cache, "train")
        try:
            self.assertIsInstance(arrays["X"], np.memmap)
            self.assertFalse(arrays["X"].flags.writeable)
            self.assertEqual(arrays["X"].dtype, np.float32)
            np.testing.assert_array_equal(arrays["X"], [[70, 1, 0], [71, 2, 1], [72, 3, 0]])
            np.testing.assert_array_equal(arrays["y"], [0, 0, 1])
            self.assertEqual(arrays["patient_id"].tolist(), ["p001"] * 3)
        finally:
            close_arrays(arrays)
        self.assertEqual(build_cache(self.processed, self.cache)["fingerprint"], metadata["fingerprint"])

    def test_overlap_is_rejected(self):
        (self.processed / "test_patient_ids.csv").write_text("patient_id,dataset\np001,A\n")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_cache(self.processed, self.cache)

    def test_target_or_future_predictors_rejected(self):
        for feature in ("SepsisLabel", "patient_id", "future_HR", "HR__lead_3h"):
            (self.processed / "feature_metadata.json").write_text(
                json.dumps({"predictor_columns": self.features + [feature]}))
            with self.assertRaisesRegex(ValueError, "prohibited"):
                build_cache(self.processed, self.cache)

    def test_missing_feature_is_rejected(self):
        path = self.processed / "train.csv.gz"
        pd.read_csv(path).drop(columns="HR").to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
            build_cache(self.processed, self.cache, batch_rows=1)
        self.assertFalse(self.cache.exists())

    def test_actual_membership_checked(self):
        path = self.processed / "test.csv.gz"
        frame = pd.read_csv(path)
        frame.loc[0, "patient_id"] = "p001"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            build_cache(self.processed, self.cache)

    def test_source_stat_change_requires_rebuild(self):
        first = build_cache(self.processed, self.cache)
        path = self.processed / "test.csv.gz"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(ValueError, "rebuild-cache"):
            build_cache(self.processed, self.cache)
        second = build_cache(self.processed, self.cache, rebuild=True)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(len(list(self.root.glob("cache.backup-*"))), 1)

    def test_manifest_row_counts_checked(self):
        path = self.processed / "train.csv.gz"
        frame = pd.read_csv(path)
        frame.loc[0, "SepsisLabel"] = 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "positive_rows differs"):
            build_cache(self.processed, self.cache)

    def test_infinite_feature_and_invalid_label_rejected(self):
        path = self.processed / "train.csv.gz"
        original = pd.read_csv(path)
        for column, value, error in (("HR", np.inf, "infinite"), ("SepsisLabel", 2, "invalid SepsisLabel")):
            frame = original.copy()
            frame.loc[0, column] = value
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, error):
                build_cache(self.processed, self.cache)


if __name__ == "__main__":
    unittest.main()
