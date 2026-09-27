"""Ensure explanation plots retain every feature and represent every test row."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.shap_plots import (
    plot_elbow,
    plot_global_summary,
    plot_importance,
    plot_local_waterfall,
)


class ShapPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = ["HR", "Temp", "ICULOS", "HR__missing", "HR__mean_3h"]
        self.X = np.asarray([[100, 38, 22, 0, 98], [70, 36, 12, 1, 75],
                             [80, 37, 6, 0, 81], [90, 39, 10, 0, 86]], dtype=np.float32)
        self.shap = np.asarray([[1, -.2, 0, .05, .01], [-.5, .15, 0, -.02, 0],
                                [.25, .1, 0, .03, .02], [0, .25, 0, -.02, 0]], dtype=np.float32)
        self.mean_abs = np.abs(self.shap).mean(axis=0)
        self.order = np.argsort(-self.mean_abs, kind="stable")

    def assert_artifacts(self, result, stem):
        for suffix in ("png", "pdf"):
            path = self.root / f"{stem}.{suffix}"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(Path(result[suffix]), path)
        self.assertEqual(result["feature_count"], len(self.features))

    def test_global_density_keeps_all_rows_all_features_and_zero_column(self):
        result = plot_global_summary(self.root / "summary.png", self.features,
                                     self.X, self.shap, self.order, bins=8)
        self.assert_artifacts(result, "summary")
        self.assertEqual(result["feature_order"], self.order.tolist())
        self.assertEqual(result["feature_names"], [self.features[index] for index in self.order])
        self.assertEqual(result["sample_count"], len(self.X))
        self.assertEqual(result["represented_values_per_feature"], [len(self.X)] * len(self.features))
        self.assertFalse(result["row_subsampling"])
        self.assertTrue(result["binned_density"])
        zero_counts = dict(zip(result["feature_names"], result["zero_shap_values_per_feature"]))
        self.assertEqual(zero_counts["ICULOS"], len(self.X))
        self.assertEqual(zero_counts["HR"], 1)

    def test_importance_and_elbow_preserve_complete_given_order(self):
        for plotter, name in ((plot_importance, "importance"), (plot_elbow, "elbow")):
            with self.subTest(plotter=name):
                result = plotter(self.root / f"{name}.png", self.features, self.mean_abs, self.order)
                self.assert_artifacts(result, name)
                self.assertEqual(result["feature_order"], self.order.tolist())
                self.assertEqual(result["feature_names"][-1], "ICULOS")

    def test_partial_duplicate_or_wrong_feature_order_rejected(self):
        for order in (self.order[:-1], [0, 0, 2, 3, 4], [0, 1, 2, 3, 5]):
            with self.subTest(order=list(order)):
                for plotter in (plot_importance, plot_elbow):
                    with self.assertRaises(ValueError):
                        plotter(self.root / "bad.png", self.features, self.mean_abs, order)
                with self.assertRaises(ValueError):
                    plot_global_summary(self.root / "bad.png", self.features, self.X,
                                        self.shap, order, bins=8)
        self.assertFalse((self.root / "bad.png").exists())

    def test_global_rejects_malformed_and_nonfinite_data(self):
        for X, contributions in ((self.X[:-1], self.shap), (self.X, self.shap[:, :-1]),
                                  (self.X[:0], self.shap[:0]),
                                  (np.full_like(self.X, np.inf), self.shap),
                                  (self.X, np.full_like(self.shap, np.nan))):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                plot_global_summary(self.root / "bad.png", self.features, X,
                                    contributions, self.order, bins=8)
        for values in (self.mean_abs[:-1], [1, 2, -1, 0, .1], [1, 2, np.inf, 0, .1]):
            with self.subTest(values=list(values)), self.assertRaises(ValueError):
                plot_importance(self.root / "bad.png", self.features, values, self.order)

    def test_local_waterfall_reconciles_all_signed_contributions(self):
        contributions = self.shap[0]
        base = -.25
        margin = base + float(contributions.astype(np.float64).sum())
        probability = float(1 / (1 + np.exp(-margin)))
        result = plot_local_waterfall(self.root / "patient.png", self.features, self.X[0],
                                      contributions, base, margin, probability, .7, "p_example", 22)
        self.assert_artifacts(result, "patient")
        self.assertEqual(set(result["feature_names"]), set(self.features))
        self.assertAlmostEqual(result["displayed_contribution_sum"], float(contributions.sum()), places=6)
        self.assertAlmostEqual(result["reconstructed_margin"], margin, places=6)
        self.assertEqual(result["zero_contribution_count"], 1)
        with self.assertRaises(ValueError):
            plot_local_waterfall(self.root / "bad.png", self.features, self.X[0],
                                 contributions, base, margin + 1, probability, .7, "p_example", 22)
        with self.assertRaises(ValueError):
            plot_local_waterfall(self.root / "bad.png", self.features, self.X[0],
                                 contributions, base, margin, .99, .7, "p_example", 22)


if __name__ == "__main__":
    unittest.main()
