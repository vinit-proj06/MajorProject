"""Tiny real-estimator runs cover persistence and patient-isolated evaluation.

These tests train only synthetic data in temporary directories. They intentionally
use multiple input/prediction batches and shuffled source feature columns.
"""
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.model_data import close_arrays, open_split
from src.model_evaluation import choose_threshold
from src.model_training import load_model_config, run_training


def _close_file_handlers():
    """Release trainer log files before Windows removes temporary artifacts."""
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.flush()
            handler.close()


class ModelTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(_close_file_handlers)
        self.root = Path(temporary.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()
        self.features = ["HR", "ICULOS", "HR__missing", "Temp"]
        (self.processed / "feature_metadata.json").write_text(
            json.dumps({"predictor_columns": self.features}), encoding="utf-8")
        self.frames = {}
        manifest = {"status": "complete", "validation_passed": True, "splits": {}}
        # Distinct patient prefixes prevent overlap. Each split includes both
        # positive and negative patients and both hourly target values.
        for split_index, (split, patient_count) in enumerate(
                (("train", 4), ("validation", 3), ("test", 2))):
            records = []
            for patient_index in range(patient_count):
                patient = f"p{split_index}{patient_index:03d}"
                dataset = "A" if patient_index % 2 == 0 else "B"
                for hour in range(1, 13):
                    high_signal = patient_index > 0 and hour >= 6 + patient_index
                    records.append({
                        "Temp": 36.4 + 1.1 * high_signal + 0.02 * hour,
                        "patient_id": patient,
                        "dataset": dataset,
                        "ICULOS": hour,
                        "SepsisLabel": int(high_signal),
                        "HR__missing": int(hour % 5 == 0),
                        "HR": 67.0 + 28 * high_signal + 0.3 * hour + patient_index,
                    })
            frame = pd.DataFrame(records)
            self.frames[split] = frame
            frame.to_csv(self.processed / f"{split}.csv.gz", index=False)
            frame[["patient_id", "dataset"]].drop_duplicates().to_csv(
                self.processed / f"{split}_patient_ids.csv", index=False)
            positive_patients = int(frame.groupby("patient_id")["SepsisLabel"].max().sum())
            positive_rows = int(frame["SepsisLabel"].sum())
            manifest["splits"][split] = {
                "rows": len(frame), "patients": patient_count,
                "positive_rows": positive_rows, "negative_rows": len(frame) - positive_rows,
                "positive_patients": positive_patients,
                "negative_patients": patient_count - positive_patients,
            }
        (self.processed / "pipeline_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        config = {
            "processed_dir": "processed", "cache_dir": "cache",
            "models_dir": "models", "reports_dir": "reports", "seed": 42,
            "n_jobs": 1, "batch_rows": 7, "calibration_bins": 4,
            "threshold_rule": "validation_max_f1", "class_weight": "balanced",
            "random_forest": {
                "n_estimators": 4, "trees_per_batch": 2, "max_depth": 3,
                "max_leaf_nodes": 8, "min_samples_leaf": 1,
                "max_features": "sqrt", "max_samples": 20,
            },
            "xgboost": {
                "num_boost_round": 8, "early_stopping_rounds": 3,
                "learning_rate": 0.2, "max_depth": 2, "min_child_weight": 1,
                "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_lambda": 1.0, "max_bin": 16, "log_every": 8,
            },
        }
        path = self.root / "model_config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        self.config = load_model_config(path)

    def _verify_run(self, model_name, trainer, predictor, saver, loader):
        selected_inputs = []

        def select_from_validation(labels, probabilities):
            # Capture while the source memmaps are open; training closes them.
            selected_inputs.append((np.array(labels, copy=True), np.array(probabilities, copy=True)))
            return choose_threshold(labels, probabilities)

        with patch("src.model_training.choose_threshold", side_effect=select_from_validation) as selection:
            result = run_training(model_name, trainer, predictor, saver, self.config)
            self.assertEqual(selection.call_count, 1)
        np.testing.assert_array_equal(selected_inputs[0][0], self.frames["validation"]["SepsisLabel"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["feature_count"], len(self.features))
        self.assertEqual(result["training_class_weight"]["fit_split"], "train")
        train_labels = self.frames["train"]["SepsisLabel"]
        self.assertEqual(result["training_class_weight"]["positive_class_weight"],
                         int((train_labels == 0).sum()) / int(train_labels.sum()))
        report_dir = Path(result["report_dir"])
        model_dir = Path(result["model_file"]).parent
        self.assertTrue(Path(result["model_file"]).is_file())
        self.assertFalse((report_dir / "RUN_FAILED.json").exists())
        self.assertEqual(json.loads((model_dir / "feature_columns.json").read_text()), self.features)
        model = loader(result["model_file"])
        observed_patient_sets = {}
        predictions_by_split = {}
        for split in ("validation", "test"):
            expected = self.frames[split]
            predictions = pd.read_csv(report_dir / f"{split}_predictions.csv.gz")
            predictions_by_split[split] = predictions
            self.assertEqual(len(predictions), len(expected))
            pd.testing.assert_frame_equal(
                predictions[["patient_id", "dataset", "ICULOS", "SepsisLabel"]],
                expected[["patient_id", "dataset", "ICULOS", "SepsisLabel"]],
                check_dtype=False,
            )
            observed_patient_sets[split] = set(predictions["patient_id"])
            self.assertFalse(predictions.duplicated(["patient_id", "ICULOS"]).any())
            arrays = open_split(Path(self.config["cache_dir"]), split)
            try:
                np.testing.assert_array_equal(
                    arrays["X"], expected[self.features].to_numpy(dtype=np.float32))
                reloaded_probabilities = predictor(model, arrays["X"])
            finally:
                close_arrays(arrays)
            np.testing.assert_allclose(
                reloaded_probabilities, predictions["probability"], rtol=1e-7, atol=1e-8)
            threshold = result["threshold_selection"]["threshold"]
            np.testing.assert_array_equal(
                predictions["predicted_SepsisLabel"],
                (reloaded_probabilities >= threshold).astype(np.uint8))
            metrics = result[split]
            self.assertEqual(sum(metrics[key] for key in ("tn", "fp", "fn", "tp")), len(expected))
            self.assertEqual(metrics["n_samples"], len(expected))
            self.assertEqual(metrics["n_positive"], int(expected["SepsisLabel"].sum()))
            self.assertEqual(sum(row["count"] for row in metrics["calibration_bins"]), len(expected))
            self.assertTrue((report_dir / f"{split}_evaluation.png").is_file())
            self.assertTrue((report_dir / f"{split}_metrics.json").is_file())
        self.assertTrue(observed_patient_sets["validation"].isdisjoint(observed_patient_sets["test"]))
        self.assertTrue(set(self.frames["train"]["patient_id"]).isdisjoint(
            observed_patient_sets["validation"] | observed_patient_sets["test"]))
        validation_predictions = predictions_by_split["validation"]
        np.testing.assert_allclose(selected_inputs[0][1], validation_predictions["probability"], rtol=1e-7, atol=1e-8)
        chosen = choose_threshold(validation_predictions["SepsisLabel"], validation_predictions["probability"])
        self.assertAlmostEqual(result["threshold_selection"]["threshold"], chosen["threshold"], places=12)
        self.assertEqual(result["threshold_selection"]["validation_samples"], len(self.frames["validation"]))
        self.assertAlmostEqual(result["threshold_selection"]["validation_f1"], chosen["validation_f1"])
        persisted_threshold = json.loads((model_dir / "threshold.json").read_text())
        self.assertEqual(persisted_threshold, result["threshold_selection"])
        return result, model

    def test_random_forest_training_reload_and_frozen_threshold(self):
        import joblib
        from scripts.train_random_forest import fit_random_forest, predict_random_forest, save_random_forest
        result, model = self._verify_run(
            "random_forest", fit_random_forest, predict_random_forest,
            save_random_forest, joblib.load)
        self.assertEqual(len(model.estimators_), 4)
        self.assertEqual(result["training_details"]["trees"], 4)
        self.assertFalse(model.oob_score)

    def test_xgboost_external_memory_reload_and_frozen_threshold(self):
        import xgboost as xgb
        from scripts.train_xgboost import fit_xgboost, predict_xgboost, save_xgboost

        def load_booster(path):
            model = xgb.Booster()
            model.load_model(path)
            return model

        result, model = self._verify_run(
            "xgboost", fit_xgboost, predict_xgboost, save_xgboost, load_booster)
        self.assertGreater(model.num_boosted_rounds(), 0)
        self.assertLessEqual(model.num_boosted_rounds(), 8)
        self.assertEqual(model.num_boosted_rounds(), model.best_iteration + 1)
        self.assertEqual(result["training_details"]["trees_saved"], model.num_boosted_rounds())


if __name__ == "__main__":
    unittest.main()
