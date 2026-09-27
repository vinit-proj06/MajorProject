"""Full-rank reporting and frozen-test prediction identity checks without fitting."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from explain_xgboost import verify_saved_batch, write_ranking


class ShapRunnerTests(unittest.TestCase):
    def test_ranking_contains_all_187_features_including_zeros_and_stable_ties(self):
        features = [f"feature_{index:03d}" for index in range(187)]
        # A large negative signed effect must still rank first by its magnitude.
        mean_abs = np.zeros(187, dtype=np.float64)
        mean_signed = np.zeros(187, dtype=np.float64)
        mean_abs[[0, 4, 8]] = [2, 3, 2]
        mean_signed[[0, 4, 8]] = [.1, -3, 2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.csv"
            order = write_ranking(path, features, mean_abs, mean_signed)
            with path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 187)
        self.assertEqual(len({row["feature"] for row in records}), 187)
        self.assertEqual([int(row["rank"]) for row in records], list(range(1, 188)))
        self.assertEqual(order[:3].tolist(), [4, 0, 8])
        self.assertEqual([row["feature"] for row in records[:3]], [features[4], features[0], features[8]])
        self.assertEqual(float(records[0]["mean_signed_shap"]), -3)
        self.assertEqual(sum(float(row["mean_abs_shap"]) == 0 for row in records), 184)
        expected_zero_order = [name for index, name in enumerate(features) if index not in (0, 4, 8)]
        self.assertEqual([row["feature"] for row in records[3:]], expected_zero_order)

    def test_ranking_rejects_invalid_arrays_and_feature_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            fixtures = [(["a", "b"], [1], [1, 0]),
                        (["a", "b"], [1, 0], [1]),
                        (["a", "b"], [-1, 0], [0, 0]),
                        (["a", "b"], [np.nan, 0], [0, 0]),
                        (["a", "b"], [1, 0], [np.inf, 0]),
                        (["a", "a"], [1, 0], [0, 0]),
                        ([], [], [])]
            for names, absolute, signed in fixtures:
                with self.subTest(names=names, absolute=absolute), self.assertRaises(ValueError):
                    write_ranking(path, names, absolute, signed)
            self.assertFalse(path.exists())

    def prediction_fixture(self):
        arrays = {
            "patient_id": np.asarray(["p_old", "p_one", "p_one", "p_two"]),
            "dataset": np.asarray(["A", "A", "A", "B"]),
            "ICULOS": np.asarray([1, 21, 22, 1]),
            "y": np.asarray([0, 0, 1, 0]),
        }
        frame = pd.DataFrame({"patient_id": ["p_one", "p_one"], "dataset": ["A", "A"],
                              "ICULOS": [21, 22], "SepsisLabel": [0, 1],
                              "probability": [.2, .8], "predicted_SepsisLabel": [0, 1]})
        return arrays, frame

    def test_saved_batch_matches_offset_metadata_scores_and_inclusive_threshold(self):
        arrays, frame = self.prediction_fixture()
        delta = verify_saved_batch(frame, arrays, 1, 3, np.asarray([.2 + 1e-8, .8]), .8)
        self.assertAlmostEqual(delta, 1e-8)

    def test_saved_batch_rejects_changed_metadata_missing_extra_and_invalid_probability(self):
        arrays, original = self.prediction_fixture()
        failures = [original.drop(columns="dataset"), original.assign(extra=0), original.iloc[:1]]
        for column, value in (("patient_id", "p_wrong"), ("dataset", "B"),
                              ("ICULOS", 999), ("SepsisLabel", 1),
                              ("probability", .3), ("probability", np.inf),
                              ("probability", np.nan), ("probability", 1.2),
                              ("predicted_SepsisLabel", 1)):
            frame = original.copy()
            frame.loc[0, column] = value
            failures.append(frame)
        for frame in failures:
            with self.subTest(columns=list(frame.columns)), self.assertRaises(ValueError):
                verify_saved_batch(frame, arrays, 1, 3, np.asarray([.2, .8]), .8)

    def test_even_tiny_probability_change_must_not_change_threshold_decision(self):
        arrays, frame = self.prediction_fixture()
        frame.loc[0, "probability"] = .5 - 1e-8
        frame["predicted_SepsisLabel"] = [0, 1]
        # Scores are within numerical tolerance but straddle the decision boundary.
        with self.assertRaisesRegex(ValueError, "threshold decision"):
            verify_saved_batch(frame, arrays, 1, 3, np.asarray([.5 + 1e-8, .8]), .5)


if __name__ == "__main__":
    unittest.main()
