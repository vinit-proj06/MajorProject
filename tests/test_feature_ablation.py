"""Controlled timing-feature exclusions preserve data and validation-only selection."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.feature_ablation import ColumnSubset, EXCLUSIONS, run_ablation_training
from src.model_evaluation import choose_threshold, evaluate_binary
from src.model_training import predict_in_batches


class TrackingArray(np.ndarray):
    def __new__(cls, values):
        result = np.asarray(values).view(cls)
        result.reads = []
        return result

    def __array_finalize__(self, source):
        self.reads = getattr(source, "reads", [])

    def __getitem__(self, item):
        self.reads.append(item)
        result = super().__getitem__(item)
        return result.view(np.ndarray) if isinstance(result, np.ndarray) else result


class FeatureAblationTests(unittest.TestCase):
    def test_column_subset_is_lazy_exact_ordered_and_bounded_during_prediction(self):
        raw = np.arange(77, dtype=np.float32).reshape(11, 7)
        source = TrackingArray(raw)
        columns = [5, 1, 3]
        selected = ColumnSubset(source, columns)
        self.assertEqual(source.reads, [])
        self.assertEqual(selected.shape, (11, 3))
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected.dtype, np.dtype("float32"))
        self.assertEqual(selected.ndim, 2)
        expected = raw[:, columns]
        np.testing.assert_array_equal(selected[2:5], expected[2:5])
        np.testing.assert_array_equal(selected[4], expected[4])
        self.assertTrue(selected[2:5].flags.c_contiguous)
        with self.assertRaises((TypeError, ValueError)):
            np.asarray(selected)
        source.reads.clear()
        seen = []

        def predictor(model, values):
            seen.append(values.copy())
            self.assertLessEqual(len(values), 4)
            return np.full(len(values), .25)

        prediction = predict_in_batches(None, selected, predictor, 4)
        np.testing.assert_array_equal(np.concatenate(seen), expected)
        np.testing.assert_array_equal(prediction, np.full(11, .25))
        self.assertEqual(len(source.reads), 3)
        self.assertEqual([(item.start, item.stop) for item in source.reads], [(0, 4), (4, 8), (8, 11)])
        self.assertEqual(raw.tobytes(), np.asarray(source).tobytes())

    def test_exclusions_and_column_index_validation(self):
        self.assertEqual(EXCLUSIONS, {
            "A": ["ICULOS"],
            "B": ["HospAdmTime", "HospAdmTime__missing"],
            "C": ["ICULOS", "HospAdmTime", "HospAdmTime__missing"],
        })
        source = np.zeros((3, 5), dtype=np.float32)
        for indices in ([], [0, 0], [-1, 2], [0, 5], [0.5, 1], [True, False]):
            with self.subTest(indices=indices), self.assertRaises((ValueError, TypeError)):
                ColumnSubset(source, indices)

    def fixture(self, root):
        # Timing columns are deliberately interspersed among retained predictors.
        names = ["HR", "ICULOS", "HospAdmTime", "Temp", "HospAdmTime__missing"]
        names += [f"feature_{index:03d}" for index in range(182)]
        splits = {}
        summary = {}
        for index, (split, labels) in enumerate((
                ("train", [0, 0, 0, 0, 0, 0, 1, 1]),
                ("validation", [0, 0, 1, 1]), ("test", [1, 0, 1]))):
            rows = len(labels)
            X = np.arange(rows * len(names), dtype=np.float32).reshape(rows, len(names))
            X[:, 0] = np.arange(rows) + 10 * index
            splits[split] = {
                "X": X, "y": np.asarray(labels, dtype=np.uint8),
                "patient_id": np.asarray([f"p_{split}"] * rows),
                "dataset": np.asarray(["A"] * rows), "ICULOS": np.arange(1, rows + 1),
            }
            summary[split] = {"rows": rows, "patients": 1, "positive_rows": sum(labels),
                              "negative_rows": rows - sum(labels), "positive_patients": 1,
                              "negative_patients": 0}
        config = {
            "processed_dir": str(root / "processed"), "cache_dir": str(root / "cache"),
            "models_dir": str(root / "original_models"), "reports_dir": str(root / "original_reports"),
            "seed": 42, "n_jobs": 2, "batch_rows": 2, "calibration_bins": 4,
            "class_weight": "balanced", "threshold_rule": "validation_max_f1",
            "xgboost": {"num_boost_round": 400, "early_stopping_rounds": 30,
                        "learning_rate": .05, "max_depth": 5, "min_child_weight": 10,
                        "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5,
                        "max_bin": 128, "log_every": 20},
        }
        cache = {"features": names, "fingerprint": "unchanged-source", "splits": summary}
        return config, cache, splits

    def test_all_experiments_reuse_fit_settings_preserve_metadata_and_freeze_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for experiment in ("A", "B", "C"):
                with self.subTest(experiment=experiment):
                    config, cache, splits = self.fixture(root)
                    saved_config = copy.deepcopy(config)
                    model_dir = root / "ablation_models" / experiment
                    report_dir = root / "ablation_reports" / experiment
                    features = [name for name in cache["features"] if name not in EXCLUSIONS[experiment]]
                    indices = [cache["features"].index(name) for name in features]
                    events = []
                    selected = []
                    sentinel = object()

                    def opening(cache_dir, split):
                        events.append("open_" + split)
                        if split == "test":
                            self.assertIn("select_threshold", events)
                            self.assertTrue((model_dir / "threshold.json").is_file())
                        return splits[split]

                    def fitting(train, validation, supplied_features, supplied_config, destination):
                        events.append("fit")
                        self.assertNotIn("open_test", events)
                        self.assertEqual(supplied_config, saved_config)
                        self.assertEqual(supplied_features, features)
                        for name, data in (("train", train), ("validation", validation)):
                            self.assertIsInstance(data["X"], ColumnSubset)
                            self.assertEqual(data["X"][:].tobytes(), splits[name]["X"][:, indices].tobytes())
                            for metadata in ("y", "patient_id", "dataset", "ICULOS"):
                                self.assertIs(data[metadata], splits[name][metadata])
                        return sentinel, {"trees_saved": 2, "best_iteration": 1}

                    def saving(model, destination):
                        self.assertIs(model, sentinel)
                        path = Path(destination) / "xgboost.ubj"
                        path.write_bytes(b"mock-only-no-model-trained")
                        return path

                    def predicting(model, values):
                        self.assertIs(model, sentinel)
                        self.assertLessEqual(len(values), config["batch_rows"])
                        scores = {10: .1, 11: .4, 12: .6, 13: .8, 20: .5, 21: .7, 22: .9}
                        return np.asarray([scores[int(value)] for value in values[:, 0]])

                    def thresholding(labels, scores):
                        events.append("select_threshold")
                        self.assertNotIn("open_test", events)
                        selected.append((np.asarray(labels).copy(), np.asarray(scores).copy()))
                        return choose_threshold(labels, scores)

                    def evaluating(labels, scores, threshold, *args, **kwargs):
                        return evaluate_binary(labels, scores, threshold, n_bins=4)

                    with ExitStack() as stack:
                        mocks = {}
                        for name, function in (("open_split", opening), ("fit_xgboost", fitting),
                                               ("save_xgboost", saving), ("predict_xgboost", predicting),
                                               ("choose_threshold", thresholding), ("write_evaluation", evaluating)):
                            mocks[name] = stack.enter_context(patch("src.feature_ablation." + name, side_effect=function))
                        stack.enter_context(patch("src.feature_ablation.close_arrays"))
                        result = run_ablation_training(experiment, config, cache, model_dir, report_dir)
                    self.assertEqual(mocks["fit_xgboost"].call_count, 1)
                    self.assertEqual(mocks["choose_threshold"].call_count, 1)
                    np.testing.assert_array_equal(selected[0][0], splits["validation"]["y"])
                    np.testing.assert_array_equal(selected[0][1], [.1, .4, .6, .8])
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["experiment"], experiment)
                    self.assertTrue(result["exploratory"])
                    self.assertEqual(result["feature_count"], len(features))
                    self.assertEqual(result["excluded_features"], EXCLUSIONS[experiment])
                    self.assertEqual(result["threshold_selection"]["threshold"], .6)
                    self.assertEqual(result["training_class_weight"]["positive_class_weight"], 3)
                    self.assertEqual(config, saved_config)
                    self.assertEqual(json.loads((model_dir / "feature_columns.json").read_text()), features)
                    frame = pd.read_csv(report_dir / "test_predictions.csv.gz")
                    np.testing.assert_array_equal(frame["ICULOS"], splits["test"]["ICULOS"])
                    np.testing.assert_array_equal(frame["patient_id"], splits["test"]["patient_id"])
                    np.testing.assert_array_equal(frame["SepsisLabel"], splits["test"]["y"])
                    np.testing.assert_array_equal(frame["predicted_SepsisLabel"], [0, 1, 1])

    def test_invalid_experiment_or_missing_excluded_feature_fails_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, cache, splits = self.fixture(root)
            with patch("src.feature_ablation.fit_xgboost") as fit:
                with self.assertRaises(ValueError):
                    run_ablation_training("D", config, cache, root / "new_model", root / "new_report")
                cache["features"].remove("HospAdmTime")
                with self.assertRaises(ValueError):
                    run_ablation_training("B", config, cache, root / "new_model", root / "new_report")
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
