"""Streaming prediction integrity tests independent of full model-cache construction."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from evaluate_early_detection import COLUMNS, iter_prediction_patients, plot_comparison
from src.early_detection import DEFAULT_POLICY, evaluate_patient, summarize_patients


class EarlyDetectionIOTests(unittest.TestCase):
    def test_plot_accepts_fractional_json_notation_for_integer_leads(self):
        policy = {**DEFAULT_POLICY, "secondary_min_lead_hours": [1.0, 3.0, 6.0]}
        records = [evaluate_patient(frame, .5, policy) for _, frame in self.frame.groupby("patient_id", sort=False)]
        summary = summarize_patients(records, policy)
        plot = self.root / "early_detection.png"
        plot_comparison(plot, "test", {("xgboost", "test"): {"summary": summary}}, {("xgboost", "test"): records}, policy)
        self.assertGreater(plot.stat().st_size, 1000)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        (self.cache / "test").mkdir(parents=True)
        self.path = self.root / "test_predictions.csv.gz"
        self.frame = pd.DataFrame({
            "patient_id": ["p001"] * 4 + ["p002"] * 3 + ["p003"] * 2,
            "dataset": ["A"] * 4 + ["B"] * 3 + ["A"] * 2,
            "ICULOS": [2., 3., 4., 5., 1., 2., 3., 10., 11.],
            "SepsisLabel": [0, 0, 1, 1, 0, 0, 0, 0, 1],
            "probability": [.1, .2, .8, .9, .3, .4, .6, .25, .75],
            "predicted_SepsisLabel": [0, 0, 1, 1, 0, 0, 1, 0, 1],
        })[COLUMNS]
        self.write_sources(self.frame)
        self.write_predictions(self.frame)

    def write_sources(self, frame):
        for name, column, dtype in (("patient_id", "patient_id", "U8"),
                                    ("dataset", "dataset", "U1"),
                                    ("ICULOS", "ICULOS", np.float64),
                                    ("y", "SepsisLabel", np.uint8)):
            np.save(self.cache / "test" / f"{name}.npy", frame[column].to_numpy(dtype=dtype))

    def write_predictions(self, frame):
        frame.to_csv(self.path, index=False)

    def read(self, batch_rows=2, max_patient_rows=100):
        return list(iter_prediction_patients(self.path, self.cache, "test", batch_rows, max_patient_rows))

    def test_patient_spanning_multiple_chunks_is_emitted_once_and_ordered(self):
        for batch_rows in (1, 2, 3, 4, 5, 50):
            with self.subTest(batch_rows=batch_rows):
                frames = self.read(batch_rows=batch_rows)
                self.assertEqual([frame.patient_id.iloc[0] for frame in frames], ["p001", "p002", "p003"])
                self.assertEqual([len(frame) for frame in frames], [4, 3, 2])
                reconstructed = pd.concat(frames, ignore_index=True)
                pd.testing.assert_frame_equal(reconstructed, self.frame, check_dtype=False)
                for frame in frames:
                    self.assertEqual(frame.patient_id.nunique(), 1)

    def test_final_patient_ends_on_exact_chunk_boundary(self):
        frames = self.read(batch_rows=3)
        self.assertEqual(frames[-1].patient_id.tolist(), ["p003", "p003"])
        self.assertEqual(frames[-1].ICULOS.tolist(), [10., 11.])

    def test_truncated_prediction_rows_rejected(self):
        self.write_predictions(self.frame.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.read()

    def test_header_only_file_rejected_as_incomplete(self):
        self.write_predictions(self.frame.iloc[:0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.read()

    def test_extra_prediction_rows_rejected(self):
        extra = pd.concat([self.frame, self.frame.iloc[[-1]]], ignore_index=True)
        self.write_predictions(extra)
        with self.assertRaisesRegex(ValueError, "extra rows"):
            self.read(batch_rows=2)

    def test_corrupted_source_identity_time_label_or_dataset_rejected(self):
        for column, value in (("patient_id", "p999"), ("dataset", "B"),
                              ("ICULOS", 3.5), ("SepsisLabel", 1)):
            with self.subTest(column=column):
                corrupted = self.frame.copy()
                corrupted.loc[0, column] = value
                self.write_predictions(corrupted)
                with self.assertRaisesRegex(ValueError, f"Prediction {column} differs from source"):
                    self.read(batch_rows=2)

    def test_reordered_rows_rejected_even_with_same_total_cohort(self):
        reordered = self.frame.iloc[[1, 0, *range(2, len(self.frame))]]
        self.write_predictions(reordered)
        with self.assertRaisesRegex(ValueError, "differs from source"):
            self.read()

    def test_reappearing_patient_rejected_even_when_cache_matches(self):
        for batch_rows in (1, 2, 20):
            with self.subTest(batch_rows=batch_rows):
                # Deliberately mirror a malformed cache so source equality cannot hide
                # p001's return after p002, including the final-carry branch.
                frame = self.frame.iloc[[0, 1, 4, 5, 2, 3]].reset_index(drop=True)
                self.write_sources(frame)
                self.write_predictions(frame)
                with self.assertRaisesRegex(ValueError, "Reappearing|noncontiguous"):
                    self.read(batch_rows=batch_rows)

    def test_reappearing_middle_group_rejected_before_final_carry(self):
        frame = self.frame.iloc[[0, 4, 1, 7]].reset_index(drop=True)
        self.write_sources(frame)
        self.write_predictions(frame)
        with self.assertRaisesRegex(ValueError, "Reappearing"):
            self.read(batch_rows=20)

    def test_max_patient_rows_limits_carried_and_complete_groups(self):
        for batch_rows in (1, 2, 5, 20):
            with self.subTest(batch_rows=batch_rows):
                with self.assertRaisesRegex(ValueError, "oversized|exceeds configured"):
                    self.read(batch_rows=batch_rows, max_patient_rows=3)
        self.assertEqual(len(self.read(batch_rows=2, max_patient_rows=4)), 3)

    def test_missing_prediction_value_rejected(self):
        frame = self.frame.copy()
        frame.loc[0, "probability"] = np.nan
        self.write_predictions(frame)
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.read()

    def test_schema_missing_extra_and_reordered_columns_rejected(self):
        bad_frames = [self.frame.drop(columns="probability"),
                      self.frame.assign(unexpected=1),
                      self.frame[list(reversed(COLUMNS))]]
        for frame in bad_frames:
            with self.subTest(columns=list(frame.columns)):
                self.write_predictions(frame)
                with self.assertRaisesRegex(ValueError, "Invalid prediction columns"):
                    self.read()

    def test_probability_threshold_boundaries_round_trip_exactly(self):
        threshold = np.float64(0.5)
        probabilities = np.array([0., np.nextafter(threshold, 0.), threshold,
                                  np.nextafter(threshold, 1.), 1., np.nextafter(1., 0.),
                                  np.nextafter(0., 1.), .12345678901234568, .9876543210987654])
        frame = self.frame.copy()
        frame["probability"] = probabilities
        frame["predicted_SepsisLabel"] = (probabilities >= threshold).astype(np.uint8)
        self.write_predictions(frame)
        recovered = pd.concat(self.read(batch_rows=1), ignore_index=True)
        np.testing.assert_array_equal(recovered.probability.to_numpy(), probabilities)
        np.testing.assert_array_equal(recovered.predicted_SepsisLabel.to_numpy(), probabilities >= threshold)


if __name__ == "__main__":
    unittest.main()
