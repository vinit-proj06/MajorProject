"""Checks for frozen-model SHAP accounting, feature identity, and local exports."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xgboost as xgb

from src.shap_explain import compute_contributions, explain_patient_row


class ShapExplanationTests(unittest.TestCase):
    def setUp(self):
        self.features = ["HR", "Temp", "ICULOS", "HR__missing", "HR__mean_3h"]
        self.X = np.asarray([[100, 38, 22, 0, 98], [70, 36, 12, 1, 75]], dtype=np.float32)
        self.values = np.asarray([[1, -.2, 0, .05, .01], [-.5, .15, .1, -.02, 0]], dtype=np.float32)
        self.base = np.asarray([.25, .25], dtype=np.float32)
        self.model = MagicMock(spec=xgb.Booster)
        self.model.feature_names = self.features.copy()
        self.model.num_features.return_value = len(self.features)
        self.model.num_boosted_rounds.return_value = 4
        self.model.best_iteration = 2
        self.model.attr.side_effect = lambda name: "2" if name == "best_iteration" else None
        self.configuration = {
            "learner": {
                "objective": {"name": "binary:logistic"},
                "learner_train_param": {"objective": "binary:logistic"},
                "gradient_booster": {"name": "gbtree"},
                "learner_model_param": {"num_feature": "5", "num_target": "1", "num_class": "0"},
            }
        }
        self.model.save_config.side_effect = lambda: json.dumps(self.configuration)
        self.model.predict.side_effect = self.predict

    def predict(self, matrix, **kwargs):
        rows = matrix.num_row()
        contributions = np.column_stack((self.values[:rows], self.base[:rows]))
        margins = contributions.astype(np.float64).sum(axis=1)
        if kwargs.get("pred_contribs"):
            return contributions[:, None, :]
        if kwargs.get("output_margin"):
            return margins[:, None]
        return (1 / (1 + np.exp(-margins)))[:, None]

    def test_all_contributions_reconcile_with_margin_and_probability(self):
        result = compute_contributions(self.model, self.X, self.features)
        np.testing.assert_array_equal(result["values"], self.values)
        np.testing.assert_array_equal(result["base_values"], self.base)
        expected_margin = self.values.astype(np.float64).sum(axis=1) + self.base
        np.testing.assert_allclose(result["margins"], expected_margin)
        np.testing.assert_allclose(result["probabilities"], 1 / (1 + np.exp(-expected_margin)))
        self.assertLess(result["max_additivity_error"], 1e-6)
        calls = self.model.predict.call_args_list
        self.assertTrue(any(call.kwargs.get("pred_contribs") for call in calls))
        for call in calls:
            self.assertEqual(call.kwargs["iteration_range"], (0, 3))
            self.assertTrue(call.kwargs["strict_shape"])
            self.assertEqual(call.args[0].feature_names, self.features)
            if call.kwargs.get("pred_contribs"):
                self.assertIs(call.kwargs["approx_contribs"], False)

    def test_schema_rejects_duplicate_missing_reordered_and_target_features(self):
        for names in (self.features[:-1], self.features[:-1] + ["HR"],
                      ["SepsisLabel"] + self.features[1:],
                      ["patient_id"] + self.features[1:],
                      ["future_HR"] + self.features[1:]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                compute_contributions(self.model, self.X, names)
        with self.assertRaises(ValueError):
            compute_contributions(self.model, pd.DataFrame(self.X, columns=self.features).iloc[:, ::-1], self.features)
        self.model.feature_names = list(reversed(self.features))
        with self.assertRaises(ValueError):
            compute_contributions(self.model, self.X, self.features)

    def test_invalid_numeric_matrix_and_nonbinary_model_rejected(self):
        for matrix in (self.X[0], self.X[:, :-1], np.empty((0, 5)),
                       np.full((1, 5), np.nan), np.full((1, 5), np.inf)):
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                compute_contributions(self.model, matrix, self.features)
        self.configuration["learner"]["objective"]["name"] = "reg:squarederror"
        with self.assertRaises(ValueError):
            compute_contributions(self.model, self.X, self.features)

    def test_wrong_prediction_shape_and_broken_additivity_fail(self):
        native_predict = self.predict
        for problem in ("shape", "nonfinite", "margin", "probability"):
            def broken(matrix, _problem=problem, **kwargs):
                result = native_predict(matrix, **kwargs).copy()
                if _problem == "shape" and kwargs.get("pred_contribs"):
                    return result[:, :, :-1]
                if _problem == "nonfinite" and kwargs.get("pred_contribs"):
                    result[0, 0, 0] = np.inf
                if _problem == "margin" and kwargs.get("output_margin"):
                    result += 1
                if _problem == "probability" and not kwargs.get("output_margin") and not kwargs.get("pred_contribs"):
                    result[:] = .99
                return result
            self.model.predict.side_effect = broken
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                compute_contributions(self.model, self.X, self.features)

    def test_frozen_round_range_must_exist_and_be_within_model(self):
        self.model.attr.side_effect = None
        for saved_best in (None, "", "-1", "4", "2.5"):
            self.model.attr.return_value = saved_best
            with self.subTest(best_iteration=saved_best), self.assertRaises(ValueError):
                compute_contributions(self.model, self.X, self.features)
        self.model.attr.return_value = "2"
        self.configuration["learner"]["learner_model_param"]["num_target"] = "2"
        with self.assertRaises(ValueError):
            compute_contributions(self.model, self.X, self.features)

    def test_local_named_row_reordered_and_complete_signed_export(self):
        row = pd.Series(dict(zip(reversed(self.features), reversed(self.X[0]))))
        with tempfile.TemporaryDirectory() as directory:
            result = explain_patient_row(self.model, row, self.features, .7, directory,
                                         patient_id="p_example", hour=22)
            root = Path(directory)
            with (root / "local_contributions.csv").open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), len(self.features))
            self.assertEqual(set(record["feature"] for record in records), set(self.features))
            by_name = {record["feature"]: record for record in records}
            self.assertEqual(float(by_name["ICULOS"]["shap_value"]), 0)
            self.assertGreater(float(by_name["HR"]["shap_value"]), 0)
            self.assertLess(float(by_name["Temp"]["shap_value"]), 0)
            self.assertEqual(float(by_name["Temp"]["feature_value"]), 38)
            magnitudes = [float(record["absolute_shap_value"]) for record in records]
            self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
            loaded = json.loads((root / "local_explanation.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["feature_count"], len(self.features))
            self.assertEqual(len(loaded["feature_contributions"]), len(self.features))
            self.assertTrue(loaded["predicted_positive"])
            self.assertAlmostEqual(loaded["base_value"] + loaded["sum_shap_values"], loaded["margin"])
            self.assertEqual(loaded["feature_names"], self.features)
            json.dumps(result, allow_nan=False)
            self.assertGreater((root / "local_waterfall.png").stat().st_size, 1000)

    def test_local_requires_exact_named_feature_row_and_valid_threshold(self):
        valid = dict(zip(self.features, self.X[0]))
        invalid_rows = [self.X[0], {**valid, "SepsisLabel": 0}, {"HR": 80},
                        {**valid, "HR": np.inf}, pd.DataFrame(self.X, columns=self.features)]
        with tempfile.TemporaryDirectory() as directory:
            for row in invalid_rows:
                with self.subTest(row_type=type(row).__name__), self.assertRaises((ValueError, TypeError)):
                    explain_patient_row(self.model, row, self.features, .7, directory)
            for threshold in (np.nan, np.inf, -.1, 1.1):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    explain_patient_row(self.model, valid, self.features, threshold, directory)

    def test_local_feature_row_without_optional_patient_metadata(self):
        row = pd.Series(self.X[0], index=self.features)
        with tempfile.TemporaryDirectory() as directory:
            result = explain_patient_row(self.model, row, self.features, .7, directory)
            self.assertIsNone(result["patient_id"])
            self.assertIsNone(result["hour"])
            self.assertEqual(result["feature_count"], len(self.features))
            self.assertTrue((Path(directory) / "local_waterfall.png").is_file())


if __name__ == "__main__":
    unittest.main()
