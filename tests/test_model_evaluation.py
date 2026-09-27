"""Numerical contracts that prevent silent evaluation and threshold errors."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.model_evaluation import choose_threshold, evaluate_binary, write_evaluation


class ModelEvaluationTests(unittest.TestCase):
    def test_known_confusion_metrics_and_ranking(self):
        result = evaluate_binary([0, 0, 0, 1, 1, 1], [.1, .8, .2, .7, .4, .9], .5)
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 2]])
        for name in ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1"]:
            self.assertAlmostEqual(result[name], 2 / 3)
        self.assertAlmostEqual(result["auroc"], 7 / 9)
        self.assertAlmostEqual(result["auprc"], (1 + 2 / 3 + 3 / 4) / 3)
        self.assertEqual(result["all_negative_accuracy"], .5)
        json.dumps(result, allow_nan=False)

    def test_uniform_calibration_bins_keep_empty_bins_and_one_boundary(self):
        result = evaluate_binary([0, 1, 0, 1], [0, .1, .9, 1], .5, n_bins=10)
        bins = result["calibration_bins"]
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[9]["count"], 2)
        self.assertIsNone(bins[2]["mean_probability"])
        self.assertIsNone(bins[2]["observed_positive_fraction"])
        self.assertEqual(bins[2]["count"], 0)
        self.assertAlmostEqual(result["ece"], .45)
        self.assertAlmostEqual(result["brier_score"], .405)

    def test_threshold_ties_choose_highest_and_include_equal_scores(self):
        # F1 is 2/3 for threshold .9 (one correct) and .7 (two TP, two FP).
        y, p = [1, 0, 0, 1], [.9, .8, .8, .7]
        result = choose_threshold(y, p)
        self.assertEqual(result["threshold"], .9)
        self.assertAlmostEqual(result["validation_f1"], 2 / 3)
        self.assertEqual(evaluate_binary(y, p, result["threshold"])["tp"], 1)
        self.assertEqual(choose_threshold([0, 1, 1], [.2, .2, .2])["candidate_thresholds"], 1)

    def test_threshold_matches_exhaustive_candidates(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 100)
        p = np.round(rng.random(100), 2)
        chosen = choose_threshold(y, p)
        exhaustive = [(evaluate_binary(y, p, threshold)["f1"], threshold) for threshold in set(p)]
        expected_f1, expected_threshold = max(exhaustive)
        self.assertEqual(chosen["threshold"], expected_threshold)
        self.assertAlmostEqual(chosen["validation_f1"], expected_f1)

    def test_one_class_and_no_predicted_positive_are_explicit(self):
        result = evaluate_binary([0, 0], [.1, .2], .5)
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["auprc"])
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)
        self.assertTrue(result["warnings"])
        self.assertTrue(np.isfinite(result["log_loss"]))
        with self.assertRaisesRegex(ValueError, "both classes"):
            choose_threshold([1, 1], [.1, .2])

    def test_invalid_inputs_rejected(self):
        for labels, scores in [([0, 2], [.1, .2]), ([0, np.nan], [.1, .2]),
                               ([0, 1], [.1, np.inf]), ([0, 1], [.1, 1.1]),
                               ([0, 1], [.1]), ([], []), ([[0, 1]], [[.1, .2]])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                evaluate_binary(labels, scores, .5)
        with self.assertRaises(ValueError):
            evaluate_binary([0, 1], [.1, .9], .5, n_bins=0)
        with self.assertRaises(ValueError):
            evaluate_binary([0, 1], [.1, .9], float("nan"))

    def test_saved_metrics_curves_and_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_evaluation([0, 0, 1, 1], [.1, .4, .3, .9], .3,
                                      directory, "test", "Example")
            root = Path(directory)
            loaded = json.loads((root / "test_metrics.json").read_text())
            self.assertEqual(loaded["confusion_matrix"], [[1, 1], [0, 2]])
            self.assertEqual(loaded["at_threshold_0_5"]["confusion_matrix"], [[2, 0], [1, 1]])
            self.assertGreater((root / "test_evaluation.png").stat().st_size, 1000)
            with (root / "test_calibration.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)
            for artifact in result["artifacts"].values():
                self.assertTrue(Path(artifact).is_file())


if __name__ == "__main__":
    unittest.main()
