"""Ablation C explanation preserves all held-out rows and exactly 184 predictors."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.ablation_shap import write_experiment_shap


class AblationShapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_dir = self.root / "model_C"
        self.report_dir = self.root / "report_C"
        self.cache_dir = self.root / "cache"
        self.output = self.report_dir / "shap"
        for folder in (self.model_dir, self.report_dir, self.cache_dir / "test"):
            folder.mkdir(parents=True)
        self.features = [f"feature_{index:03d}" for index in range(184)]
        self.source_features = ["ICULOS", *self.features[:40], "HospAdmTime", *self.features[40:100],
                                "HospAdmTime__missing", *self.features[100:]]
        self.indices = [self.source_features.index(name) for name in self.features]
        self.X = np.arange(3 * 187, dtype=np.float32).reshape(3, 187) / 10
        self.hours = np.asarray([21, 22, 23], dtype=np.float32)
        self.X[:, self.source_features.index("ICULOS")] = self.hours
        self.X[:, self.source_features.index("HospAdmTime")] = -12
        self.X[:, self.source_features.index("HospAdmTime__missing")] = 0
        self.arrays = {"X": self.X, "y": np.asarray([0, 0, 1], dtype=np.uint8),
                       "patient_id": np.asarray(["p_example"] * 3), "dataset": np.asarray(["A"] * 3),
                       "ICULOS": self.hours}
        for name, array in self.arrays.items():
            np.save(self.cache_dir / "test" / f"{name}.npy", array)
        self.cache = {"features": self.source_features, "fingerprint": "fixed-cohort",
                      "splits": {"test": {"rows": 3, "patients": 1, "positive_rows": 1}}}
        self.threshold = {"threshold": .5, "rule": "validation_max_f1"}
        self.trained = {
            "status": "complete", "model": "xgboost", "experiment": "C", "exploratory": True,
            "excluded_features": ["ICULOS", "HospAdmTime", "HospAdmTime__missing"],
            "feature_count": 184, "feature_order_sha256": hashlib.sha256(json.dumps(self.features).encode()).hexdigest(),
            "cache_fingerprint": self.cache["fingerprint"], "splits": self.cache["splits"],
            "threshold_selection": self.threshold,
            "training_details": {"trees_saved": 2, "best_iteration": 1},
        }
        for path, data in ((self.cache_dir / "cache_manifest.json", self.cache),
                           (self.model_dir / "feature_columns.json", self.features),
                           (self.model_dir / "threshold.json", self.threshold),
                           (self.report_dir / "results.json", self.trained)):
            path.write_text(json.dumps(data), encoding="utf-8")
        (self.model_dir / "xgboost.ubj").write_bytes(b"mock-frozen-model-not-trained")
        computed = self.calculate(self.X[:, self.indices])
        self.predictions = pd.DataFrame({"patient_id": self.arrays["patient_id"], "dataset": self.arrays["dataset"],
                                         "ICULOS": self.hours, "SepsisLabel": self.arrays["y"],
                                         "probability": computed["probabilities"],
                                         "predicted_SepsisLabel": (computed["probabilities"] >= .5).astype(np.uint8)})
        self.predictions.to_csv(self.report_dir / "test_predictions.csv.gz", index=False)
        self.model = MagicMock()
        self.model.feature_names = self.features
        self.model.num_features.return_value = 184
        self.model.num_boosted_rounds.return_value = 2
        self.model.best_iteration = 1
        self.seen_batches = []

    @staticmethod
    def calculate(values):
        contributions = np.asarray(values, dtype=np.float64) / 1000
        contributions[:, 1::2] *= -1
        contributions[:, -5:] = 0
        base = np.full(len(values), .1)
        margins = base + contributions.sum(axis=1)
        return {"values": contributions, "base_values": base, "margins": margins,
                "probabilities": 1 / (1 + np.exp(-margins)),
                "max_additivity_error": 0., "max_probability_error": 0.}

    def compute(self, model, values, features):
        self.assertIs(model, self.model)
        self.assertEqual(features, self.features)
        self.assertEqual(values.shape[1], 184)
        self.assertLessEqual(len(values), 2)
        self.seen_batches.append(values.copy())
        return self.calculate(values)

    def plot(self, path, features, *args, **kwargs):
        self.assertEqual(features, self.features)
        path = Path(path)
        path.write_bytes(b"plot-test-stub")
        path.with_suffix(".pdf").write_bytes(b"plot-test-stub")
        order = np.asarray(args[-1])
        result = {"png": str(path), "pdf": str(path.with_suffix(".pdf")),
                  "feature_count": 184, "feature_order": order.tolist(),
                  "feature_names": [features[index] for index in order]}
        if len(args) == 3:
            # The global summary receives the persisted selected X and SHAP arrays.
            X, contributions, _ = args
            np.testing.assert_array_equal(X, self.X[:, self.indices])
            np.testing.assert_allclose(contributions, self.calculate(self.X[:, self.indices])["values"], atol=1e-8)
            result.update(sample_count=3, represented_values_per_feature=[3] * 184,
                          row_subsampling=False, binned_density=True)
        return result

    def patched(self, computation=None):
        stack = ExitStack()
        stack.enter_context(patch("src.ablation_shap.xgb.Booster", return_value=self.model))
        # Cache validation has its own tests. This isolates the exporter while
        # retaining real read-only mappings and its source-file integrity guard.
        stack.enter_context(patch("src.ablation_shap.open_split", side_effect=lambda root, split: {
            name: np.load(Path(root) / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in self.arrays
        }))
        compute = stack.enter_context(patch("src.ablation_shap.compute_contributions", side_effect=computation or self.compute))
        for name in ("plot_global_summary", "plot_importance", "plot_elbow"):
            stack.enter_context(patch("src.ablation_shap." + name, side_effect=self.plot))
        return stack, compute

    def run_export(self):
        return write_experiment_shap(self.model_dir, self.report_dir, self.cache_dir, self.output,
                                     batch_rows=2, n_jobs=1)

    def test_all_rows_184_features_exact_values_metadata_and_additivity(self):
        before = hashlib.sha256((self.model_dir / "xgboost.ubj").read_bytes()).hexdigest()
        stack, compute = self.patched()
        with stack:
            manifest = self.run_export()
        self.assertEqual(compute.call_count, 2)
        np.testing.assert_array_equal(np.concatenate(self.seen_batches), self.X[:, self.indices])
        self.assertEqual(manifest["rows_explained"], 3)
        self.assertEqual(manifest["features_ranked"], 184)
        self.assertEqual(manifest["original_cache_feature_count"], 187)
        self.assertEqual(manifest["source_column_indexes"], self.indices)
        self.assertIsNone(manifest["feature_cutoff"])
        self.assertFalse(manifest["row_subsampling"])
        self.assertFalse(manifest["model_retrained"])
        self.assertEqual(manifest["model_sha256_before"], before)
        self.assertEqual(manifest["model_sha256_after"], before)
        ranking = pd.read_csv(self.output / "feature_importance_full.csv")
        self.assertEqual(len(ranking), 184)
        self.assertEqual(set(ranking.feature), set(self.features))
        self.assertEqual(int((ranking.mean_abs_shap == 0).sum()), 5)
        self.assertTrue(ranking.mean_abs_shap.is_monotonic_decreasing)
        expected = np.abs(self.calculate(self.X[:, self.indices])["values"]).mean(axis=0)
        np.testing.assert_allclose(ranking.set_index("feature").loc[self.features, "mean_abs_shap"], expected)
        np.testing.assert_array_equal(np.load(self.output / "test_feature_values.npy"), self.X[:, self.indices])
        contributions = np.load(self.output / "shap_values.npy")
        base, margin = np.load(self.output / "base_values.npy"), np.load(self.output / "raw_margins.npy")
        np.testing.assert_allclose(base + contributions.sum(axis=1, dtype=np.float64), margin, atol=1e-7)
        row_index = pd.read_csv(self.output / "test_row_index.csv.gz")
        np.testing.assert_array_equal(row_index.ICULOS, self.hours)
        np.testing.assert_array_equal(row_index.SepsisLabel, self.arrays["y"])
        self.assertNotIn("ICULOS", json.loads((self.output / "feature_columns.json").read_text()))
        for artifact in manifest["artifacts"].values():
            self.assertTrue(Path(artifact).is_file())
        summary = json.loads((self.output / "summary_plot_metadata.json").read_text())
        self.assertEqual(summary["represented_values_per_feature"], [3] * 184)
        self.assertEqual(Path(summary["png"]), self.output / "shap_summary_all_features.png")

    def test_wrong_exclusion_or_predictor_order_rejected_before_explanation(self):
        original = (self.model_dir / "feature_columns.json").read_text()
        stack, compute = self.patched()
        with stack:
            for features in (list(reversed(self.features)), ["ICULOS", *self.features[1:]]):
                (self.model_dir / "feature_columns.json").write_text(json.dumps(features))
                with self.assertRaises(ValueError):
                    self.run_export()
            (self.model_dir / "feature_columns.json").write_text(original)
            changed = dict(self.trained, excluded_features=["ICULOS", "HospAdmTime"])
            (self.report_dir / "results.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.run_export()
        compute.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_bad_persisted_additivity_is_not_published(self):
        def broken(model, values, features):
            result = self.compute(model, values, features)
            result["margins"] = result["margins"] + .1
            return result
        stack, _ = self.patched(broken)
        with stack, self.assertRaisesRegex(ValueError, "additivity"):
            self.run_export()
        self.assertFalse(self.output.exists())
        staged = list(self.output.parent.glob("shap.inprogress-*"))
        self.assertEqual(len(staged), 1)
        self.assertTrue((staged[0] / "RUN_FAILED.json").is_file())

    def test_missing_saved_rows_is_not_silently_explained_as_full_test(self):
        self.predictions.iloc[:2].to_csv(self.report_dir / "test_predictions.csv.gz", index=False)
        stack, _ = self.patched()
        with stack, self.assertRaisesRegex(ValueError, "Incomplete"):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_changed_hour_or_target_rejected_and_existing_result_preserved(self):
        changed = self.predictions.copy()
        changed.loc[0, "ICULOS"] = 999
        changed.to_csv(self.report_dir / "test_predictions.csv.gz", index=False)
        stack, _ = self.patched()
        with stack, self.assertRaisesRegex(ValueError, "ICULOS"):
            self.run_export()
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        existing = self.output / "previous_result.txt"
        existing.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_export()
        self.assertEqual(existing.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
