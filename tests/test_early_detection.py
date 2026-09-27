"""Independent boundary and causality cases for patient-level warning metrics."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from src.early_detection import evaluate_patient, summarize_patients


def patient_frame(patient_id="p001", start=1, end=24, first_positive=14, warnings=()):
    """First label at hour 14 implies onset 20 and useful hours 8 through 14."""
    hours = np.arange(start, end + 1)
    warned = np.isin(hours, list(warnings))
    labels = np.zeros(len(hours), dtype=np.uint8)
    if first_positive is not None:
        labels = (hours >= first_positive).astype(np.uint8)
    return pd.DataFrame({
        "patient_id": patient_id,
        "dataset": "A",
        "ICULOS": hours,
        "SepsisLabel": labels,
        "probability": np.where(warned, 0.9, 0.1),
        "predicted_SepsisLabel": warned.astype(np.uint8),
    })


class EarlyDetectionTests(unittest.TestCase):
    def setUp(self):
        # Policy keys follow the public module contract, not private helpers.
        self.policy = {"label_shift_hours": 6, "min_lead_hours": 6, "max_lead_hours": 12,
                       "secondary_min_lead_hours": [1, 3, 6], "consecutive_positive_hours": 1}

    def evaluate(self, **kwargs):
        return evaluate_patient(patient_frame(**kwargs), 0.5, self.policy)

    def test_onset_is_first_observed_transition_plus_six(self):
        result = self.evaluate(warnings=(8,))
        self.assertEqual(result["onset_status"], "known_transition")
        self.assertEqual(result["first_label_positive_hour"], 14)
        self.assertEqual(result["inferred_onset_hour"], 20)
        self.assertEqual(result["warning_window_start"], 8)
        self.assertEqual(result["warning_window_end"], 14)
        self.assertEqual(result["observed_warning_hours"], 7)
        self.assertTrue(result["warning_window_complete"])
        self.assertTrue(result["onset_within_record"])
        self.assertFalse(result["onset_after_record_end"])

    def test_primary_window_boundaries_are_inclusive(self):
        for warning_hour, detected in ((7, False), (8, True), (14, True), (15, False), (20, False), (21, False)):
            with self.subTest(warning_hour=warning_hour):
                result = self.evaluate(warnings=(warning_hour,))
                self.assertTrue(result["primary_evaluable"])
                self.assertEqual(result["primary_detected"], detected)
                if detected:
                    self.assertEqual(result["first_useful_alert_hour"], warning_hour)
                    self.assertEqual(result["first_useful_alert_lead_hours"], 20 - warning_hour)
                else:
                    self.assertIsNone(result["first_useful_alert_hour"])

    def test_qualifying_warning_counts_even_after_an_earlier_warning(self):
        result = self.evaluate(warnings=(2, 3, 8, 9, 15))
        self.assertTrue(result["primary_detected"])
        self.assertEqual(result["first_alert_hour"], 2)
        self.assertEqual(result["first_alert_lead_hours"], 18)
        self.assertEqual(result["first_useful_alert_hour"], 8)
        self.assertEqual(result["first_useful_alert_lead_hours"], 12)
        self.assertEqual(result["alert_positive_hours"], 5)
        self.assertEqual(result["alert_episodes"], 3)

    def test_no_alert_is_a_missed_evaluable_patient(self):
        result = self.evaluate()
        self.assertTrue(result["primary_evaluable"])
        self.assertFalse(result["primary_detected"])
        self.assertFalse(result["any_alert"])
        self.assertIsNone(result["first_alert_hour"])
        self.assertIsNone(result["first_useful_alert_lead_hours"])
        self.assertEqual(result["alert_positive_hours"], 0)
        summary = summarize_patients([result], self.policy)
        self.assertEqual(summary["primary_sensitivity"], 0)
        self.assertIsNone(summary["mean_first_useful_warning_lead_hours"])
        self.assertIsNone(summary["median_first_useful_warning_lead_hours"])

    def test_first_row_positive_is_left_censored_not_an_inferred_onset(self):
        result = self.evaluate(start=14, end=24, first_positive=14, warnings=(14, 15))
        self.assertTrue(result["septic_patient"])
        self.assertEqual(result["onset_status"], "left_censored")
        self.assertIsNone(result["inferred_onset_hour"])
        self.assertFalse(result["primary_evaluable"])
        summary = summarize_patients([result], self.policy)
        self.assertEqual(summary["n_left_censored_septic_patients"], 1)
        self.assertEqual(summary["n_primary_evaluable"], 0)
        self.assertIsNone(summary["primary_sensitivity"])

    def test_partial_window_is_primary_eligible_but_not_fully_observed(self):
        result = self.evaluate(start=10, warnings=(10,))
        self.assertEqual(result["observed_warning_hours"], 5)
        self.assertFalse(result["warning_window_complete"])
        self.assertTrue(result["primary_evaluable"])
        self.assertTrue(result["primary_detected"])
        self.assertFalse(result["secondary_evaluable"])
        summary = summarize_patients([result], self.policy)
        self.assertEqual(summary["n_primary_evaluable_incomplete_window"], 1)
        self.assertEqual(summary["primary_sensitivity"], 1)
        self.assertIsNone(summary["secondary_sensitivity"])

    def test_onset_after_record_end_is_reported_separately(self):
        result = self.evaluate(first_positive=22, end=24, warnings=(16,))
        self.assertEqual(result["inferred_onset_hour"], 28)
        self.assertTrue(result["onset_after_record_end"])
        self.assertFalse(result["onset_within_record"])
        self.assertTrue(result["warning_window_complete"])
        self.assertTrue(result["primary_evaluable"])
        self.assertTrue(result["primary_detected"])
        self.assertFalse(result["secondary_evaluable"])
        summary = summarize_patients([result], self.policy)
        self.assertEqual(summary["n_onset_after_record_end"], 1)
        self.assertEqual(summary["n_primary_evaluable_onset_after_record_end"], 1)

    def test_negative_patient_false_alarm_hour_and_episode_burden(self):
        alerted = self.evaluate(patient_id="negative_alert", end=12, first_positive=None, warnings=(3, 4, 7))
        quiet = self.evaluate(patient_id="negative_quiet", end=12, first_positive=None)
        self.assertEqual(alerted["onset_status"], "never_positive")
        self.assertFalse(alerted["septic_patient"])
        self.assertFalse(alerted["primary_evaluable"])
        self.assertEqual(alerted["alert_positive_hours"], 3)
        self.assertEqual(alerted["alert_episodes"], 2)
        summary = summarize_patients([alerted, quiet], self.policy)
        self.assertEqual(summary["n_nonseptic_patients_with_alert"], 1)
        self.assertEqual(summary["nonseptic_patient_hours"], 24)
        self.assertEqual(summary["nonseptic_alert_positive_hours"], 3)
        self.assertEqual(summary["nonseptic_alert_episodes"], 2)
        self.assertEqual(summary["nonseptic_false_alarm_rate"], 0.5)
        self.assertAlmostEqual(summary["nonseptic_alert_positive_hours_per_100_patient_hours"], 12.5)
        self.assertAlmostEqual(summary["nonseptic_alert_episodes_per_100_patient_hours"], 100 * 2 / 24)
        self.assertIsNone(summary["primary_sensitivity"])

    def test_secondary_lead_windows_are_separate_from_primary(self):
        result = self.evaluate(warnings=(15,))  # lead 5, misses the requested >=6 h window.
        self.assertFalse(result["primary_detected"])
        summary = summarize_patients([result], self.policy)
        by_lead = summary["secondary_sensitivity_by_min_lead_hours"]
        self.assertEqual(by_lead["1"]["sensitivity"], 1)
        self.assertEqual(by_lead["3"]["sensitivity"], 1)
        self.assertEqual(by_lead["6"]["sensitivity"], 0)

    def test_empty_denominators_are_none(self):
        summary = summarize_patients([], self.policy)
        for key in ("primary_sensitivity", "secondary_sensitivity", "nonseptic_false_alarm_rate",
                    "mean_first_useful_warning_lead_hours", "median_first_useful_warning_lead_hours"):
            with self.subTest(key=key):
                self.assertIsNone(summary[key])
        self.assertEqual(summary["n_primary_evaluable"], 0)
        self.assertEqual(summary["n_primary_detected"], 0)

    def test_consecutive_confirmation_never_backdates_an_alert(self):
        policy = {**self.policy, "consecutive_positive_hours": 2}
        late = evaluate_patient(patient_frame(warnings=(14, 15)), 0.5, policy)
        self.assertEqual(late["first_alert_hour"], 15)
        self.assertEqual(late["alert_positive_hours"], 1)
        self.assertFalse(late["primary_detected"])
        at_boundary = evaluate_patient(patient_frame(warnings=(7, 8)), 0.5, policy)
        self.assertEqual(at_boundary["first_alert_hour"], 8)
        self.assertTrue(at_boundary["primary_detected"])
        self.assertEqual(at_boundary["first_useful_alert_lead_hours"], 12)

    def test_confirmation_is_causal_for_every_observed_prefix(self):
        policy = {**self.policy, "consecutive_positive_hours": 2}
        full = patient_frame(first_positive=None, end=12, warnings=(3, 4, 5, 9))
        # Raw scores are positive 3,4,5,9. Confirmed outputs are only 4,5;
        # neither the future hour 4 nor the future absence at 10 may backdate one.
        for prefix_end in range(1, 13):
            with self.subTest(prefix_end=prefix_end):
                result = evaluate_patient(full.iloc[:prefix_end].copy(), 0.5, policy)
                expected_alert_hours = int(prefix_end >= 4) + int(prefix_end >= 5)
                self.assertEqual(result["alert_positive_hours"], expected_alert_hours)
                self.assertEqual(result["alert_episodes"], int(prefix_end >= 4))
                self.assertEqual(result["first_alert_hour"], 4 if prefix_end >= 4 else None)

    def test_mixed_known_partial_and_unknown_onsets_use_distinct_denominators(self):
        full_missed = self.evaluate(patient_id="full_missed")
        partial_detected = self.evaluate(patient_id="partial_detected", start=10, warnings=(10,))
        beyond_detected = self.evaluate(patient_id="beyond_detected", first_positive=22, warnings=(16,))
        unknown = self.evaluate(patient_id="unknown", start=14, warnings=(14,))
        negative = self.evaluate(patient_id="negative", first_positive=None)
        summary = summarize_patients([full_missed, partial_detected, beyond_detected, unknown, negative], self.policy)
        self.assertEqual(summary["n_patients"], 5)
        self.assertEqual(summary["n_septic_patients"], 4)
        self.assertEqual(summary["n_nonseptic_patients"], 1)
        self.assertEqual(summary["n_primary_evaluable"], 3)
        self.assertEqual(summary["n_primary_detected"], 2)
        self.assertAlmostEqual(summary["primary_sensitivity"], 2 / 3)
        self.assertEqual(summary["n_secondary_evaluable"], 1)
        self.assertEqual(summary["n_secondary_detected"], 0)
        self.assertEqual(summary["secondary_sensitivity"], 0)
        self.assertEqual(summary["mean_first_useful_warning_lead_hours"], 11)
        self.assertEqual(summary["median_first_useful_warning_lead_hours"], 11)

    def test_summary_rejects_duplicate_patients_and_mixed_policies(self):
        first = self.evaluate(patient_id="first", warnings=(8, 9))
        with self.assertRaises(ValueError):
            summarize_patients([first, deepcopy(first)], self.policy)
        other_policy = {**self.policy, "consecutive_positive_hours": 2}
        second = evaluate_patient(patient_frame(patient_id="second", warnings=(8, 9)), 0.5, other_policy)
        with self.assertRaises(ValueError):
            summarize_patients([first, second], self.policy)
        other_threshold = evaluate_patient(patient_frame(patient_id="second", warnings=(8,)), 0.6, self.policy)
        with self.assertRaises(ValueError):
            summarize_patients([first, other_threshold], self.policy)

    def test_invalid_label_reversal_is_rejected(self):
        frame = patient_frame()
        frame.loc[frame["ICULOS"] == 15, "SepsisLabel"] = 0
        with self.assertRaises(ValueError):
            evaluate_patient(frame, 0.5, self.policy)

    def test_time_gaps_and_duplicate_hours_are_rejected(self):
        frame = patient_frame()
        variants = {
            "gap": frame.loc[frame["ICULOS"] != 12],
            "duplicate": pd.concat([frame.iloc[:8], frame.iloc[[7]], frame.iloc[8:]], ignore_index=True),
            "reverse": frame.iloc[::-1],
        }
        for name, variant in variants.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_patient(variant, 0.5, self.policy)

    def test_nonbinary_labels_or_predictions_are_rejected(self):
        for column in ("SepsisLabel", "predicted_SepsisLabel"):
            frame = patient_frame()
            frame.loc[0, column] = 2
            with self.subTest(column=column), self.assertRaises(ValueError):
                evaluate_patient(frame, 0.5, self.policy)

    def test_prediction_must_match_the_frozen_probability_threshold(self):
        frame = patient_frame()
        frame.loc[0, "predicted_SepsisLabel"] = 1  # probability remains 0.1.
        with self.assertRaises(ValueError):
            evaluate_patient(frame, 0.5, self.policy)

    def test_threshold_equality_is_a_positive_prediction(self):
        frame = patient_frame(warnings=(14,))
        frame.loc[frame["ICULOS"] == 14, "probability"] = 0.5
        result = evaluate_patient(frame, 0.5, self.policy)
        self.assertTrue(result["primary_detected"])


if __name__ == "__main__":
    unittest.main()
