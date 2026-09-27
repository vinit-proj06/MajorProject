"""Leakage and split-invariant regression tests; no external dataset required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_engineering import engineer_patient, feature_metadata
from src.split_dataset import split_patients


CONFIG = {
    "seed": 42,
    "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
    "rolling_windows": [3, 6, 12],
    "rolling_features": ["HR", "Temp"],
    "forward_fill_features": ["HR"],
    "forward_fill_limit_hours": 3,
}


def patient() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["p1"] * 7, "dataset": ["A"] * 7,
        "ICULOS": np.arange(2, 9), "HospAdmTime": [-8.0] * 7,
        "HR": [np.nan, 80., np.nan, 84., np.nan, np.nan, 88.],
        "Temp": [37., np.nan, 38., np.nan, 37.5, np.nan, 37.],
        "SepsisLabel": [0, 0, 0, 0, 1, 1, 1],
    })


class FeatureTests(unittest.TestCase):
    def test_prefix_invariance_and_labels_are_not_features(self):
        frame = patient()
        original = frame.copy(deep=True)
        full = engineer_patient(frame, CONFIG)
        prefix = engineer_patient(frame.iloc[:4], CONFIG)
        assert_frame_equal(prefix, full.iloc[:4])
        changed = frame.copy()
        changed.loc[4:, ["HR", "Temp"]] = 9999.0
        changed["SepsisLabel"] = 1 - changed["SepsisLabel"]
        transformed = engineer_patient(changed, CONFIG)
        assert_frame_equal(full.iloc[:4].drop(columns="SepsisLabel"),
                           transformed.iloc[:4].drop(columns="SepsisLabel"))
        self.assertFalse(any("SepsisLabel__" in column for column in full))
        assert_frame_equal(frame, original)

    def test_one_patient_only_and_no_cross_patient_state(self):
        first = engineer_patient(patient(), CONFIG)
        alternate = patient()
        alternate["patient_id"] = "p2"
        alternate["HR"] = np.nan
        second = engineer_patient(alternate, CONFIG)
        self.assertTrue(second["HR"].isna().all())
        self.assertTrue((second["HR__hours_since_observed"] == -1).all())
        self.assertTrue(np.isnan(first.loc[0, "HR"]))
        with self.assertRaisesRegex(ValueError, "exactly one patient"):
            engineer_patient(pd.concat([patient(), alternate], ignore_index=True), CONFIG)

    def test_forward_fill_is_bounded_and_indicators_remain_raw(self):
        frame = patient()
        frame["HR"] = [np.nan, 80., np.nan, np.nan, np.nan, np.nan, 88.]
        result = engineer_patient(frame, CONFIG)
        np.testing.assert_allclose(result["HR"], [np.nan, 80., 80., 80., 80., np.nan, 88.], equal_nan=True)
        np.testing.assert_array_equal(result["HR__missing"], [1, 0, 1, 1, 1, 1, 0])
        np.testing.assert_allclose(result["HR__hours_since_observed"], [-1, 0, 1, 2, 3, 4, 0])
        self.assertTrue(np.isnan(result.loc[1, "Temp"]))
        # No original HR measurement in the three-hour window, even though HR was filled.
        self.assertTrue(np.isnan(result.loc[4, "HR__mean_3h"]))

    def test_rolling_statistics_use_only_observed_values(self):
        result = engineer_patient(patient(), CONFIG)
        self.assertEqual(result.loc[3, "HR__mean_3h"], 82.0)
        self.assertEqual(result.loc[3, "HR__min_3h"], 80.0)
        self.assertEqual(result.loc[3, "HR__max_3h"], 84.0)
        self.assertEqual(result.loc[3, "HR__std_3h"], 2.0)
        self.assertEqual(result.loc[3, "HR__slope_3h"], 2.0)
        self.assertTrue(np.isnan(result.loc[1, "HR__std_3h"]))
        self.assertTrue(np.isnan(result.loc[1, "HR__slope_3h"]))

    def test_elapsed_hour_windows_and_forward_fill_across_gaps(self):
        frame = patient().iloc[:3].copy()
        frame["ICULOS"] = [10, 11, 20]
        frame["HR"] = [80., 82., np.nan]
        result = engineer_patient(frame, CONFIG)
        self.assertTrue(np.isnan(result.loc[2, "HR"]))
        self.assertTrue(np.isnan(result.loc[2, "HR__previous"]))
        self.assertTrue(np.isnan(result.loc[2, "HR__mean_6h"]))
        self.assertEqual(result.loc[2, "HR__hours_since_observed"], 9)

    def test_metadata_covers_exactly_predictors(self):
        frame = patient()
        output = engineer_patient(frame, CONFIG)
        metadata = feature_metadata(frame.columns, CONFIG)
        self.assertEqual(set(metadata), set(output) - {"patient_id", "dataset", "SepsisLabel"})
        self.assertEqual(metadata["HR__mean_3h"]["fallback_source"], "HR")
        self.assertEqual(metadata["HR__slope_3h"]["fallback"], "zero")
        self.assertEqual(metadata["HR__missing"]["fallback"], "none")

    def test_randomized_windows_match_independent_reference(self):
        rng = np.random.default_rng(14)
        hours = np.cumsum(rng.integers(1, 4, size=40)).astype(float) + 100
        values = rng.normal(80, 10, size=40)
        values[rng.random(40) < 0.6] = np.nan
        frame = pd.DataFrame({"patient_id": "p1", "ICULOS": hours,
                              "HR": values, "Temp": values / 2, "SepsisLabel": 0})
        output = engineer_patient(frame, CONFIG)
        reference = pd.Series(values, index=pd.to_timedelta(hours, unit="h"))
        for window in CONFIG["rolling_windows"]:
            rolling = reference.rolling(f"{window}h", min_periods=1)
            for statistic in ("mean", "min", "max"):
                expected = getattr(rolling, statistic)().to_numpy()
                np.testing.assert_allclose(output[f"HR__{statistic}_{window}h"], expected,
                                           atol=1e-10, equal_nan=True)
            expected_std = rolling.std(ddof=0).where(rolling.count() >= 2)
            np.testing.assert_allclose(output[f"HR__std_{window}h"], expected_std,
                                       atol=1e-10, equal_nan=True)
            for row, hour in enumerate(hours):
                selected = (hours > hour - window) & (hours <= hour) & np.isfinite(values)
                expected_slope = (np.polyfit(hours[selected] - hour, values[selected], 1)[0]
                                  if selected.sum() >= 2 else np.nan)
                np.testing.assert_allclose(output.loc[row, f"HR__slope_{window}h"], expected_slope,
                                           atol=1e-10, equal_nan=True)


class SplitTests(unittest.TestCase):
    @staticmethod
    def manifest() -> pd.DataFrame:
        return pd.DataFrame([
            {"patient_id": f"p{i:05d}", "dataset": "A" if i < 100 else "B",
             "positive_patient": i % 5 == 0, "n_rows": i + 1,
             "positive_rows": 1 if i % 5 == 0 else 0}
            for i in range(200)
        ])

    def test_splits_reproducible_disjoint_complete_and_patient_stratified(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            mapping, report = split_patients(manifest, CONFIG, Path(first))
            shuffled, _ = split_patients(manifest.sample(frac=1, random_state=91), CONFIG, Path(second))
            self.assertEqual(mapping, shuffled)
            self.assertEqual(set(mapping), set(manifest["patient_id"]))
            self.assertTrue(report["complete_coverage"])
            self.assertEqual(sum(report["overlap_counts"].values()), 0)
            self.assertEqual([report["splits"][name]["patients"] for name in ("train", "validation", "test")], [140, 30, 30])
            for name in ("train", "validation", "test"):
                path = Path(first) / f"{name}_patient_ids.csv"
                self.assertEqual(path.read_bytes(), (Path(second) / path.name).read_bytes())
                self.assertEqual(list(pd.read_csv(path).columns), ["patient_id", "dataset"])
                self.assertEqual(report["splits"][name]["positive_patients"] / report["splits"][name]["patients"], 0.2)

    def test_small_stratum_retains_training_representative_and_warns(self):
        manifest = self.manifest().iloc[[0, 1]].copy()
        with tempfile.TemporaryDirectory() as directory:
            mapping, report = split_patients(manifest, CONFIG, Path(directory))
            self.assertTrue(all(value == "train" for value in mapping.values()))
            self.assertEqual(len(report["warnings"]), 2)

    def test_duplicate_patient_rejected(self):
        manifest = self.manifest().iloc[:3].copy()
        manifest.loc[1, "patient_id"] = manifest.loc[0, "patient_id"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "globally unique"):
                split_patients(manifest, CONFIG, Path(directory))


if __name__ == "__main__":
    unittest.main()
