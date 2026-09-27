"""End-to-end regression tests with small, deliberately imperfect PSV inputs.

The fixture is synthetic; these tests do not touch the research dataset.
Run with ``python -m unittest discover -s tests -v``.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.common import load_config


SCHEMA = ["HR", "Temp", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]
SPLITS = ("train", "validation", "test")


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sepsis_pipeline_test_")
        self.root = Path(self.temporary.name)
        self.inputs = {source: self.root / f"training_set{source}" for source in ("A", "B")}
        self.raw = {}
        self.source_paths = {}
        for source, directory in self.inputs.items():
            nested = directory / "nested"
            nested.mkdir(parents=True)
            for number in range(12):
                patient_id = f"{source.lower()}{number:04d}"
                frame = pd.DataFrame({
                    "HR": [np.nan, 70.0 + number, np.nan, np.nan, np.nan, 80.0 + number],
                    "Temp": [np.nan, 36.0 + number / 20, np.nan, 37.0, np.nan, np.nan],
                    "Gender": [number % 2] * 6,
                    "Unit1": [number % 2] * 6,
                    "Unit2": [1 - number % 2] * 6,
                    "HospAdmTime": [-10.0 - number] * 6,
                    "ICULOS": np.arange(3, 9),
                    "SepsisLabel": [0, 0, 0, 0, number % 2, number % 2],
                }, columns=SCHEMA)
                path = nested / f"{patient_id}.psv"
                frame.to_csv(path, sep="|", index=False, na_rep="NaN")
                self.raw[patient_id] = frame
                self.source_paths[patient_id] = path

        # Corruption is spread across folders and names sorted after valid files.
        fixture = next(iter(self.raw.values())).copy()
        fixture.to_csv(self.inputs["A"] / "z_malformed.psv", sep="|", index=False, na_rep="NaN")
        with (self.inputs["A"] / "z_malformed.psv").open("a", encoding="utf-8") as stream:
            stream.write("1|2|3\n")
        invalid_target = fixture.copy()
        invalid_target.loc[2, "SepsisLabel"] = 2
        invalid_target.to_csv(self.inputs["A"] / "z_invalid_target.psv", sep="|", index=False, na_rep="NaN")
        infinite = fixture.copy()
        infinite.loc[2, "HR"] = np.inf
        infinite.to_csv(self.inputs["B"] / "z_infinite.psv", sep="|", index=False, na_rep="NaN")
        fixture.drop(columns="Temp").to_csv(self.inputs["B"] / "z_wrong_schema.psv", sep="|", index=False, na_rep="NaN")
        for directory in self.inputs.values():
            fixture.to_csv(directory / "z_duplicate.psv", sep="|", index=False, na_rep="NaN")

    def tearDown(self):
        # Release the pipeline FileHandler before removing files on Windows.
        for handler in logging.getLogger().handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logging.getLogger().removeHandler(handler)
        self.temporary.cleanup()

    def config(self, name):
        config = load_config()
        config.update({
            "input_dirs": {source: str(path) for source, path in self.inputs.items()},
            "output_dir": str(self.root / f"output_{name}"),
            "reports_dir": str(self.root / f"reports_{name}"),
            "expected_counts": {"A": 15, "B": 15},
            "rolling_features": ["HR", "Temp"],
            "forward_fill_features": ["HR", "Temp"],
            "batch_size": 2,
            "max_batch_rows": 12,
            "progress_every": 1000,
            "inspection_samples_per_dataset": 3,
            "storage_format": "csv_gzip",
        })
        return config

    def run_fixture(self, name):
        from src.preprocess import run_pipeline
        from src.storage import extension, read_chunk
        config = self.config(name)
        summary = run_pipeline(config)
        output = Path(config["output_dir"])
        frames = {split: read_chunk(output / f"{split}{extension(config['storage_format'])}") for split in SPLITS}
        identifiers = {split: set(pd.read_csv(output / f"{split}_patient_ids.csv")["patient_id"])
                       for split in SPLITS}
        return config, summary, frames, identifiers

    def test_quarantine_preservation_splits_and_reproducibility(self):
        hashes = {patient: hashlib.sha256(path.read_bytes()).hexdigest()
                  for patient, path in self.source_paths.items()}
        config, summary, frames, identifiers = self.run_fixture("first")
        self.assertFalse(identifiers["train"] & identifiers["validation"])
        self.assertFalse(identifiers["train"] & identifiers["test"])
        self.assertFalse(identifiers["validation"] & identifiers["test"])
        self.assertEqual(set.union(*identifiers.values()), set(self.raw))
        self.assertTrue(all(identifiers.values()), "All three splits should have patients")
        combined = pd.concat(list(frames.values()), ignore_index=True)
        self.assertEqual(len(combined), 24 * 6)
        self.assertEqual(int(combined.SepsisLabel.sum()), 24)
        self.assertEqual(combined.patient_id.nunique(), 24)
        self.assertNotIn("positive_patient", combined.columns)
        self.assertNotIn("positive_rows", combined.columns)
        self.assertFalse(any(column.startswith("SepsisLabel__") for column in combined.columns))
        for split, frame in frames.items():
            self.assertEqual(set(frame.patient_id), identifiers[split])
            self.assertFalse(frame.isna().any().any())
            self.assertTrue(np.isfinite(frame.select_dtypes(include="number").to_numpy()).all())
            self.assertTrue(frame.SepsisLabel.isin([0, 1]).all())
        self.assertLessEqual(summary["memory_bounds"]["max_raw_buffer_rows"], 12)
        self.assertLessEqual(summary["memory_bounds"]["max_raw_buffer_patients"], 2)
        self.assertLessEqual(summary["memory_bounds"]["max_output_buffer_rows_per_split"], 12)
        self.assertLessEqual(summary["memory_bounds"]["max_output_buffer_patients_per_split"], 2)
        from src.storage import raw_chunk_paths, read_chunk
        for chunk in raw_chunk_paths(Path(config["output_dir"]) / "raw_processed"):
            self.assertLessEqual(len(read_chunk(chunk)), 12)
        for patient, frame in combined.groupby("patient_id"):
            np.testing.assert_array_equal(frame.ICULOS, self.raw[patient].ICULOS)
            np.testing.assert_array_equal(frame.SepsisLabel, self.raw[patient].SepsisLabel)
            self.assertEqual(int(frame.iloc[0]["HR__missing"]), 1)
            self.assertEqual(float(frame.iloc[0]["HR__hours_since_observed"]), -1)
        failures = pd.read_csv(Path(config["reports_dir"]) / "failed_files.csv")
        self.assertEqual(len(failures), 6, failures.to_string())
        self.assertEqual(set(failures.patient_id), {
            "z_malformed", "z_invalid_target", "z_infinite", "z_wrong_schema", "z_duplicate"})
        for patient, path in self.source_paths.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashes[patient])

        _, _, repeated_frames, repeated_identifiers = self.run_fixture("repeat")
        self.assertEqual(identifiers, repeated_identifiers)
        for split in SPLITS:
            pd.testing.assert_frame_equal(frames[split], repeated_frames[split], check_exact=True)

    def test_held_out_extremes_do_not_change_training_transformations(self):
        _, _, initial_frames, initial_ids = self.run_fixture("baseline")
        # Change only measurements, preserving every held-out patient's outcome,
        # row count, filename, source and timestamps so stratification is stable.
        for patient in initial_ids["validation"] | initial_ids["test"]:
            frame = self.raw[patient].copy()
            for column in ("HR", "Temp"):
                frame[column] = frame[column].where(frame[column].isna(), frame[column] + 1_000_000)
            frame.to_csv(self.source_paths[patient], sep="|", index=False, na_rep="NaN")
        _, _, mutated_frames, mutated_ids = self.run_fixture("heldout_changed")
        self.assertEqual(initial_ids, mutated_ids)
        pd.testing.assert_frame_equal(initial_frames["train"], mutated_frames["train"], check_exact=True)
        training_raw_hr = pd.concat([self.raw[patient].HR for patient in initial_ids["train"]])
        expected_median = float(training_raw_hr.median())
        # The first hour has no preceding observation; every split must receive
        # the training-only median, never the patient's later HR or held-out HR.
        for frame in mutated_frames.values():
            first_hours = frame.groupby("patient_id", sort=False).head(1)
            np.testing.assert_allclose(first_hours.HR, expected_median, rtol=0, atol=0)

    def test_training_empty_source_drops_levels_but_retains_missingness(self):
        from src.feature_engineering import engineer_patient, feature_metadata
        from src.preprocess import apply_parameters
        config = self.config("empty_source")
        raw = next(iter(self.raw.values())).copy()
        raw["HR"] = np.nan
        raw.insert(0, "dataset", "A")
        raw.insert(0, "patient_id", "never_observed")
        metadata = feature_metadata(raw.columns, config)
        parameters = {"values": {
            "HR": None, "Temp": 36.5, "Gender": 0.0, "Unit1": 0.0,
            "Unit2": 1.0, "HospAdmTime": -10.0, "ICULOS": 5.5,
        }}
        result = apply_parameters(engineer_patient(raw, config), metadata, parameters)
        self.assertNotIn("HR", result)
        self.assertNotIn("HR__previous", result)
        for window in config["rolling_windows"]:
            for statistic in ("mean", "min", "max"):
                self.assertNotIn(f"HR__{statistic}_{window}h", result)
            for statistic in ("std", "slope"):
                self.assertTrue((result[f"HR__{statistic}_{window}h"] == 0).all())
        self.assertTrue((result["HR__missing"] == 1).all())
        self.assertTrue((result["HR__hours_since_observed"] == -1).all())
        self.assertTrue(np.isfinite(result.select_dtypes(include="number").to_numpy()).all())
        np.testing.assert_array_equal(result.SepsisLabel, raw.SepsisLabel)

    def test_invalid_times_labels_and_binary_covariates_are_rejected(self):
        from src.common import PatientFile, read_patient
        config = self.config("invalid_values")
        baseline = next(iter(self.raw.values())).copy()
        cases = [
            ("ICULOS", 2, 3.0, "consecutive"),  # backward and duplicate time
            ("ICULOS", 2, 9.0, "consecutive"),  # gap followed by backward time
            ("ICULOS", 2, 5.5, "integer"),
            ("ICULOS", 2, np.nan, "ICULOS"),
            ("SepsisLabel", 2, np.nan, "binary"),
            ("SepsisLabel", 2, 0.5, "binary"),
            ("Gender", 2, 0.5, "outside"),
            ("Unit1", 2, -1.0, "outside"),
        ]
        path = self.root / "invalid.psv"
        for column, row, value, expected_error in cases:
            with self.subTest(column=column, value=value):
                frame = baseline.astype(float).copy()
                frame.loc[row, column] = value
                frame.to_csv(path, sep="|", index=False, na_rep="NaN")
                with self.assertRaisesRegex(ValueError, expected_error):
                    read_patient(PatientFile("invalid", "A", path), SCHEMA, config)


if __name__ == "__main__":
    unittest.main()
