"""Replay safety contracts using real causal preprocessing and an unfitted fake model."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.dashboard_data import (
    ArtifactIntegrityError, DashboardData, DashboardInputError, PatientNotFoundError,
)
from src.feature_engineering import engineer_patient, feature_metadata
from src.preprocess import apply_parameters


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed_dir = self.root / "processed_data"
        self.cache_dir = self.root / "model_cache"
        self.raw_dir = self.root / "raw" / "A"
        self.model_dir = self.root / "models" / "xgboost"
        self.report_dir = self.root / "reports" / "models" / "xgboost"
        for folder in (self.processed_dir, self.cache_dir, self.raw_dir, self.model_dir, self.report_dir):
            folder.mkdir(parents=True)
        self.patient_id = "p000001"
        self.pipeline = {
            "input_dirs": {"A": str(self.raw_dir)}, "rolling_windows": [3],
            "rolling_features": ["HR", "SBP"], "forward_fill_features": ["HR", "SBP", "Temp"],
            "forward_fill_limit_hours": 3, "max_patient_rows": 100, "max_file_bytes": 1_000_000,
        }
        self.raw = pd.DataFrame({
            "HR": [80., 100., 90., 110.], "SBP": [np.nan, np.nan, 140., np.nan],
            "Temp": [37., np.nan, 38., np.nan], "ICULOS": [1., 2., 3., 4.],
            "Age": [65.] * 4, "Gender": [1.] * 4, "Lactate": [np.nan, np.nan, 2., np.nan],
            "HospAdmTime": [-4.] * 4, "SepsisLabel": [0, 0, 0, 1],
        })
        self.schema = list(self.raw)
        self.raw_path = self.raw_dir / f"{self.patient_id}.psv"
        self.raw.to_csv(self.raw_path, sep="|", index=False)
        self.raw.insert(0, "dataset", "A")
        self.raw.insert(0, "patient_id", self.patient_id)
        self.imputation = {"fit_split": "train", "training_patient_count": 5,
                           "values": {"HR": 83., "SBP": 121., "Temp": 37., "Age": 60., "Gender": 1.,
                                      "Lactate": 1.8, "HospAdmTime": -5.87, "ICULOS": 21.}}
        self.metadata = feature_metadata(self.schema, self.pipeline)
        self.features = list(self.metadata)
        prepared = apply_parameters(engineer_patient(self.raw, self.pipeline), self.metadata, self.imputation)
        self.arrays = {"X": prepared[self.features].to_numpy(dtype=np.float32),
                       "y": self.raw.SepsisLabel.to_numpy(dtype=np.uint8),
                       "patient_id": self.raw.patient_id.to_numpy(dtype=str),
                       "dataset": self.raw.dataset.to_numpy(dtype=str),
                       "ICULOS": self.raw.ICULOS.to_numpy(dtype=np.float64)}
        self.policy = {"label_shift_hours": 6, "min_lead_hours": 6, "max_lead_hours": 12,
                       "secondary_min_lead_hours": [1, 3, 6], "consecutive_positive_hours": 1}
        self.threshold = {"threshold": .8, "rule": "validation_max_f1"}
        self.results = {
            "status": "complete", "model": "xgboost", "cache_fingerprint": "fixture-cache",
            "feature_count": len(self.features), "threshold_selection": self.threshold,
            "feature_order_sha256": hashlib.sha256(json.dumps(self.features).encode()).hexdigest(),
            "training_details": {"best_iteration": 1, "trees_saved": 2},
        }
        fixtures = {
            self.root / "model_config.json": {"processed_dir": "processed_data", "cache_dir": "model_cache",
                                               "models_dir": "models", "reports_dir": "reports/models", "n_jobs": 1},
            self.root / "early_detection_config.json": {"policy": self.policy},
            self.processed_dir / "config_resolved.json": self.pipeline,
            self.processed_dir / "feature_metadata.json": {"predictor_columns": self.features, "features": self.metadata},
            self.processed_dir / "imputation_parameters.json": self.imputation,
            self.processed_dir / "schema.json": {"columns": self.schema},
            self.processed_dir / "pipeline_manifest.json": {"status": "complete", "validation_passed": True},
            self.cache_dir / "cache_manifest.json": {"features": self.features, "fingerprint": "fixture-cache"},
            self.model_dir / "feature_columns.json": self.features,
            self.model_dir / "threshold.json": self.threshold,
            self.report_dir / "results.json": self.results,
        }
        for path, value in fixtures.items():
            path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        (self.model_dir / "xgboost.ubj").write_bytes(b"unfitted-mock-for-tests-only")
        pd.DataFrame({"patient_id": [self.patient_id], "dataset": ["A"]}).to_csv(
            self.processed_dir / "test_patient_ids.csv", index=False)
        pd.DataFrame({"patient_id": [self.patient_id, "p000002"], "dataset": ["A", "A"],
                      "filename": [str(self.raw_path), str(self.raw_dir / "p000002.psv")],
                      "n_rows": [4, 4], "sha256": [hashlib.sha256(self.raw_path.read_bytes()).hexdigest(), "not-a-test-patient"]}).to_csv(
            self.processed_dir / "patient_manifest.csv", index=False)
        # The disk-cache implementation is independently tested. These tests use
        # its validated arrays directly and exercise the dashboard's own checks.
        self.open_patch = patch("src.dashboard_data.open_split", return_value=self.arrays)
        self.open_patch.start()
        self.addCleanup(self.open_patch.stop)
        self.booster = MagicMock()
        self.booster.feature_names = self.features.copy()
        self.booster.num_features.return_value = len(self.features)
        self.booster.best_iteration = 1
        self.booster.num_boosted_rounds.return_value = 2
        self.booster.inplace_predict.side_effect = self.predict
        self.booster_patch = patch("src.dashboard_data.xgb.Booster", return_value=self.booster)
        self.booster_patch.start()
        self.addCleanup(self.booster_patch.stop)
        self.shap_patch = patch("src.dashboard_data.compute_contributions", side_effect=self.explain)
        self.shap_patch.start()
        self.addCleanup(self.shap_patch.stop)
        self.data = DashboardData(self.root)
        self.addCleanup(self.data.close)

    def predict(self, values, iteration_range):
        self.assertEqual(iteration_range, (0, 2))
        self.assertEqual(values.shape[1], len(self.features))
        margin = values[:, self.features.index("HR")].astype(np.float64) * .1 - 7.
        return 1 / (1 + np.exp(-margin))

    def explain(self, booster, values, names):
        self.assertIs(booster, self.booster)
        self.assertEqual(names, self.features)
        shap = np.zeros(values.shape, dtype=np.float64)
        shap[:, names.index("HR")] = values[:, names.index("HR")].astype(np.float64) * .1
        base = np.full(len(values), -7.)
        margin = base + shap.sum(axis=1)
        return {"values": shap, "base_values": base, "margins": margin,
                "probabilities": 1 / (1 + np.exp(-margin)), "max_additivity_error": 0.}

    def assert_no_outcome_fields(self, value):
        if isinstance(value, dict):
            for key, nested in value.items():
                lower = key.lower()
                self.assertNotIn("onset", lower)
                self.assertNotIn("sepsislabel", lower)
                self.assertNotIn("target", lower)
                self.assertNotIn(key, {"label", "labels", "positive_patient", "first_positive_hour"})
                self.assert_no_outcome_fields(nested)
        elif isinstance(value, list):
            for nested in value:
                self.assert_no_outcome_fields(nested)

    def test_snapshot_prefix_unchanged_by_future_measurements_or_any_outcome_labels(self):
        original = self.data.snapshot(self.patient_id, 2)
        changed = self.raw.copy()
        changed.loc[changed.ICULOS > 2, ["HR", "SBP", "Temp", "Lactate"]] = 999.
        changed.SepsisLabel = 1 - changed.SepsisLabel
        seen_prefixes = []

        def engineering(frame, config):
            self.assertNotIn("SepsisLabel", frame)
            self.assertTrue((frame.ICULOS <= 2).all())
            seen_prefixes.append(frame.copy())
            return engineer_patient(frame, config)

        with patch.object(self.data, "_load_patient", return_value=changed), patch(
                "src.dashboard_data.engineer_patient", side_effect=engineering):
            after = self.data.snapshot(self.patient_id, 2)
        self.assertEqual(original, after)
        self.assertTrue(seen_prefixes)
        self.assertEqual([row["hour"] for row in after["history"]], [1., 2.])
        self.assertEqual([row["hour"] for row in after["score_history"]], [1., 2.])
        self.assert_no_outcome_fields(after)
        json.dumps(after, allow_nan=False)

    def test_missing_raw_measurements_remain_visible_and_saved_train_fallback_is_reused(self):
        captured_parameters = []

        def applying(frame, metadata, parameters):
            captured_parameters.append(parameters)
            self.assertEqual(parameters["fit_split"], "train")
            return apply_parameters(frame, metadata, parameters)

        with patch("src.dashboard_data.apply_parameters", side_effect=applying):
            snapshot = self.data.snapshot(self.patient_id, 2)
        self.assertIs(captured_parameters[0], self.data.imputation_parameters)
        measurements = {row["feature"]: row for row in snapshot["measurements"]}
        self.assertIsNone(measurements["SBP"]["raw_value"])
        self.assertEqual(measurements["SBP"]["model_value"], 121.)
        self.assertEqual(measurements["SBP"]["source"], "training_imputed")
        self.assertIsNone(measurements["SBP"]["last_observed_hour"])
        self.assertIsNone(snapshot["current_vitals"]["SBP"])
        self.assertIsNone(measurements["Temp"]["raw_value"])
        self.assertEqual(measurements["Temp"]["model_value"], 37.)
        self.assertEqual(measurements["Temp"]["source"], "forward_filled")
        self.assertEqual(measurements["Temp"]["last_observed_hour"], 1.)
        self.assertEqual(len(snapshot["shap"]["features"]), len(self.features))
        self.assertTrue(snapshot["alert"])
        self.assertEqual(snapshot["threshold"], .8)
        self.assertEqual([row["hour"] for row in snapshot["alerts"]], [2.])
        json.dumps(snapshot, allow_nan=False)

    def test_changed_imputation_parameters_or_cache_values_are_rejected(self):
        self.data.imputation_parameters["values"]["SBP"] = 999.
        with self.assertRaisesRegex(ArtifactIntegrityError, "prefix"):
            self.data.snapshot(self.patient_id, 2)
        self.booster.inplace_predict.assert_not_called()
        self.data.imputation_parameters["values"]["SBP"] = 121.
        self.arrays["X"][0, self.features.index("HR")] = 999.
        with self.assertRaisesRegex(ArtifactIntegrityError, "prefix"):
            self.data.snapshot(self.patient_id, 2)

    def test_test_membership_and_guarded_source_and_model_paths(self):
        self.assertEqual(self.data.catalog(), [{"patient_id": self.patient_id, "dataset": "A"}])
        for patient in ("p000002", "p999999", "../model_config.json", None):
            with self.subTest(patient=patient), self.assertRaises(PatientNotFoundError):
                self.data.patient_info(patient)
        with self.assertRaises(DashboardInputError):
            self.data.snapshot(self.patient_id, 2, model="../xgboost")
        # Even a whitelisted patient must resolve within the approved source root.
        self.data._patients[self.patient_id]["filename"] = str(self.root / "outside" / f"{self.patient_id}.psv")
        with self.assertRaisesRegex(ArtifactIntegrityError, "outside"):
            self.data.patient_info(self.patient_id)

    def test_saved_model_feature_alignment_and_invalid_scores_are_rejected(self):
        self.booster.feature_names = list(reversed(self.features))
        with self.assertRaisesRegex(ArtifactIntegrityError, "Loaded model"):
            self.data.snapshot(self.patient_id, 2)
        self.booster.inplace_predict.assert_not_called()
        self.booster.feature_names = self.features.copy()
        for output in (np.asarray([np.nan, .8]), np.asarray([.5, 1.2]), np.asarray([.5])):
            self.booster.inplace_predict.side_effect = None
            self.booster.inplace_predict.return_value = output
            with self.subTest(output=output.tolist()), self.assertRaisesRegex(ArtifactIntegrityError, "scores"):
                self.data.snapshot(self.patient_id, 2)

    def test_retrospective_outcomes_are_separate_and_do_not_pollute_live_payload(self):
        before = self.data.snapshot(self.patient_id, 2)
        review = self.data.retrospective(self.patient_id)
        self.assertEqual(review["mode"], "retrospective_review")
        self.assertEqual([row["label"] for row in review["timeline"]], [0, 0, 0, 1])
        self.assertEqual(review["summary"]["inferred_onset_hour"], 10.)
        after = self.data.snapshot(self.patient_id, 2)
        self.assertEqual(before, after)
        self.assert_no_outcome_fields(after)
        json.dumps(review, allow_nan=False)

    def test_exact_recorded_hours_and_closed_backend_are_enforced(self):
        for hour in (True, None, float("nan"), float("inf"), 2.5, 0, 5):
            with self.subTest(hour=hour), self.assertRaises(DashboardInputError):
                self.data.snapshot(self.patient_id, hour)
        self.data.close()
        self.assertTrue(self.data._closed)
        with self.assertRaises(RuntimeError):
            self.data.catalog()

    def test_constructor_rejects_imputation_fitted_outside_training(self):
        path = self.processed_dir / "imputation_parameters.json"
        altered = copy.deepcopy(self.imputation)
        altered["fit_split"] = "test"
        path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ArtifactIntegrityError, "training-fitted"):
            DashboardData(self.root)


if __name__ == "__main__":
    unittest.main()
