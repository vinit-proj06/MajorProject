"""Patient-level early-warning evaluation of frozen hourly predictions.

Alerts depend only on current/past predictions. Labels define evaluation windows
afterward; they never trigger an alert. These metrics are not challenge utility.
"""
from __future__ import annotations

import numpy as np


DEFAULT_POLICY = {
    "label_shift_hours": 6,
    "min_lead_hours": 6,
    "max_lead_hours": 12,
    "secondary_min_lead_hours": [1, 3, 6],
    "consecutive_positive_hours": 1,
}


def _policy(policy):
    effective = {**DEFAULT_POLICY, **policy}
    for name in ("label_shift_hours", "min_lead_hours", "max_lead_hours"):
        try:
            value = float(effective[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and nonnegative") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        effective[name] = value
    if effective["min_lead_hours"] > effective["max_lead_hours"]:
        raise ValueError("min_lead_hours cannot exceed max_lead_hours")
    consecutive = effective["consecutive_positive_hours"]
    if isinstance(consecutive, bool) or not isinstance(consecutive, (int, np.integer)) or consecutive < 1:
        raise ValueError("consecutive_positive_hours must be a positive integer")
    effective["consecutive_positive_hours"] = int(consecutive)
    try:
        secondary = [float(value) for value in effective["secondary_min_lead_hours"]]
    except (TypeError, ValueError) as exc:
        raise ValueError("secondary_min_lead_hours must be a sequence of numeric hours") from exc
    if any(not np.isfinite(value) or value < 0 or value > effective["max_lead_hours"] for value in secondary):
        raise ValueError("Secondary lead hours must be finite and between 0 and max_lead_hours")
    if len(set(secondary)) != len(secondary):
        raise ValueError("secondary_min_lead_hours must not contain duplicates")
    effective["secondary_min_lead_hours"] = secondary
    return effective


def _lead_key(hours):
    return format(hours, "g")


def _lead_prefix(hours):
    return "secondary_lead_" + _lead_key(hours).replace(".", "p") + "h"


def _window(times, alerts, onset, minimum_lead, maximum_lead):
    if onset is None:
        return {"start": None, "end": None, "observed_hours": 0,
                "complete": False, "first_alert_hour": None, "first_alert_lead_hours": None,
                "alert_hours": 0}
    start, end = onset - maximum_lead, onset - minimum_lead
    observed = (times >= start) & (times <= end)
    useful = np.flatnonzero(observed & alerts)
    first = float(times[useful[0]]) if useful.size else None
    return {"start": float(start), "end": float(end),
            "observed_hours": int(observed.sum()),
            "complete": bool(times[0] <= start and times[-1] >= end),
            "first_alert_hour": first,
            "first_alert_lead_hours": float(onset - first) if first is not None else None,
            "alert_hours": int(useful.size)}


def evaluate_patient(frame, threshold, policy):
    """Evaluate one chronologically ordered patient's frozen predictions.

    A known 0-to-1 label transition gives inferred onset = first positive label
    hour + label_shift_hours. A positive first row is left-censored, not an
    observed transition. The primary cohort needs at least one observed hour in
    the inclusive warning window; stricter secondary cohorts require the whole
    window and onset to occur within the record.
    """
    policy = _policy(policy)
    required = {"patient_id", "dataset", "ICULOS", "SepsisLabel", "probability", "predicted_SepsisLabel"}
    if not hasattr(frame, "columns") or not required.issubset(frame.columns):
        raise ValueError(f"Patient frame must contain columns {sorted(required)}")
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Patient frame contains duplicate column names")
    if len(frame) == 0:
        raise ValueError("Patient frame must contain at least one hourly row")
    for name in ("patient_id", "dataset"):
        if frame[name].isna().any() or frame[name].nunique() != 1:
            raise ValueError(f"Patient frame must contain exactly one nonmissing {name}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be finite and in [0, 1]") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    try:
        times = np.asarray(frame["ICULOS"], dtype=np.float64)
        labels = np.asarray(frame["SepsisLabel"], dtype=np.float64)
        probabilities = np.asarray(frame["probability"], dtype=np.float64)
        predictions = np.asarray(frame["predicted_SepsisLabel"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Time, labels and predictions must be numeric") from exc
    if not np.isfinite(times).all() or (times.size > 1 and not np.allclose(np.diff(times), 1, rtol=0, atol=1e-8)):
        raise ValueError("ICULOS must be finite, strictly increasing and contiguous hourly observations")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all() or np.any(np.diff(labels) < 0):
        raise ValueError("SepsisLabel must be finite, binary and nondecreasing")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probability must be finite and in [0, 1]")
    if not np.isfinite(predictions).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("predicted_SepsisLabel must be finite and binary")
    if not np.array_equal(predictions.astype(bool), probabilities >= threshold):
        raise ValueError("Frozen predicted_SepsisLabel does not match probability >= threshold")

    # A warning is emitted at its confirmation hour, never backdated. There is
    # no cooldown: every subsequent qualifying positive hour emits a warning.
    alerts = np.zeros(len(frame), dtype=bool)
    streak = 0
    for index, predicted_positive in enumerate(predictions):
        streak = streak + 1 if predicted_positive else 0
        alerts[index] = streak >= policy["consecutive_positive_hours"]
    alert_indexes = np.flatnonzero(alerts)
    first_alert = float(times[alert_indexes[0]]) if alert_indexes.size else None
    episodes = int(np.count_nonzero(alerts & ~np.r_[False, alerts[:-1]]))
    positive_indexes = np.flatnonzero(labels == 1)
    septic = bool(positive_indexes.size)
    first_positive_hour = float(times[positive_indexes[0]]) if septic else None
    if not septic:
        status, onset = "never_positive", None
    elif labels[0] == 1:
        status, onset = "left_censored", None
    else:
        status = "known_transition"
        onset = float(first_positive_hour + policy["label_shift_hours"])
    onset_within = bool(onset is not None and times[0] <= onset <= times[-1])
    onset_after = bool(onset is not None and onset > times[-1])
    window = _window(times, alerts, onset, policy["min_lead_hours"], policy["max_lead_hours"])
    primary_evaluable = bool(onset is not None and window["observed_hours"] > 0)
    secondary_evaluable = bool(onset_within and window["complete"])
    if primary_evaluable:
        exclusion = None
    elif not septic:
        exclusion = "never_septic"
    elif status == "left_censored":
        exclusion = "onset_left_censored"
    else:
        exclusion = "no_observed_warning_window"

    record = {
        "patient_id": str(frame["patient_id"].iloc[0]),
        "dataset": str(frame["dataset"].iloc[0]),
        "n_hours": int(len(frame)),
        "first_record_hour": float(times[0]), "last_record_hour": float(times[-1]),
        "threshold": threshold,
        "label_shift_hours": policy["label_shift_hours"],
        "min_lead_hours": policy["min_lead_hours"],
        "max_lead_hours": policy["max_lead_hours"],
        "consecutive_positive_hours": policy["consecutive_positive_hours"],
        "septic_patient": septic,
        "positive_label_hours": int(labels.sum()),
        "onset_status": status,
        "first_label_positive_hour": first_positive_hour,
        "inferred_onset_hour": onset,
        "onset_within_record": onset_within,
        "onset_after_record_end": onset_after,
        "prediction_positive_hours": int(predictions.sum()),
        "alert_positive_hours": int(alerts.sum()),
        "alert_episodes": episodes,
        "any_alert": bool(alert_indexes.size),
        "first_alert_hour": first_alert,
        "first_alert_lead_hours": float(onset - first_alert) if onset is not None and first_alert is not None else None,
        "warning_window_start": window["start"],
        "warning_window_end": window["end"],
        "observed_warning_hours": window["observed_hours"],
        "warning_window_complete": window["complete"],
        "primary_evaluable": primary_evaluable,
        "primary_detected": bool(primary_evaluable and window["first_alert_hour"] is not None),
        "primary_exclusion_reason": exclusion,
        "first_useful_alert_hour": window["first_alert_hour"],
        "first_useful_alert_lead_hours": window["first_alert_lead_hours"],
        "useful_alert_positive_hours": window["alert_hours"],
        "secondary_evaluable": secondary_evaluable,
        "secondary_detected": bool(secondary_evaluable and window["first_alert_hour"] is not None),
    }
    for minimum in policy["secondary_min_lead_hours"]:
        prefix = _lead_prefix(minimum)
        alternate = _window(times, alerts, onset, minimum, policy["max_lead_hours"])
        eligible = bool(onset_within and alternate["complete"])
        record.update({
            f"{prefix}_evaluable": eligible,
            f"{prefix}_detected": bool(eligible and alternate["first_alert_hour"] is not None),
            f"{prefix}_first_alert_hour": alternate["first_alert_hour"],
            f"{prefix}_lead_hours": alternate["first_alert_lead_hours"],
            f"{prefix}_window_complete": alternate["complete"],
            f"{prefix}_observed_hours": alternate["observed_hours"],
        })
    return record


def summarize_patients(records, policy):
    """Aggregate patient detection and nonseptic alarm burden; empty rates=null."""
    policy = _policy(policy)
    records = list(records)
    ids = [record["patient_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Patient records contain duplicate patient IDs")
    if len({record["threshold"] for record in records}) > 1:
        raise ValueError("Patient records must share one frozen threshold")
    for record in records:
        for name in ("label_shift_hours", "min_lead_hours", "max_lead_hours", "consecutive_positive_hours"):
            if record[name] != policy[name]:
                raise ValueError(f"Patient record uses a different policy: {name}")
        expected_secondary = {_lead_prefix(value) + "_evaluable" for value in policy["secondary_min_lead_hours"]}
        actual_secondary = {key for key in record if key.startswith("secondary_lead_") and key.endswith("_evaluable")}
        if expected_secondary != actual_secondary:
            raise ValueError("Patient record uses different secondary lead windows")

    def count(predicate):
        return sum(bool(predicate(record)) for record in records)
    def fraction(numerator, denominator):
        return float(numerator / denominator) if denominator else None
    n_septic = count(lambda row: row["septic_patient"])
    nonseptic = [row for row in records if not row["septic_patient"]]
    n_primary = count(lambda row: row["primary_evaluable"])
    n_detected = count(lambda row: row["primary_detected"])
    n_secondary = count(lambda row: row["secondary_evaluable"])
    n_secondary_detected = count(lambda row: row["secondary_detected"])
    nonseptic_hours = sum(row["n_hours"] for row in nonseptic)
    nonseptic_alert_hours = sum(row["alert_positive_hours"] for row in nonseptic)
    nonseptic_episodes = sum(row["alert_episodes"] for row in nonseptic)
    false_alarm_patients = sum(bool(row["any_alert"]) for row in nonseptic)
    leads = [row["first_useful_alert_lead_hours"] for row in records if row["primary_detected"]]
    result = {
        "policy": policy,
        "threshold": records[0]["threshold"] if records else None,
        "n_patients": len(records),
        "n_hourly_records": sum(row["n_hours"] for row in records),
        "n_septic_patients": n_septic,
        "n_nonseptic_patients": len(nonseptic),
        "n_positive_label_hours": sum(row["positive_label_hours"] for row in records),
        "n_known_onset_patients": count(lambda row: row["onset_status"] == "known_transition"),
        "n_left_censored_septic_patients": count(lambda row: row["onset_status"] == "left_censored"),
        "n_onset_after_record_end": count(lambda row: row["onset_after_record_end"]),
        "n_onset_within_record": count(lambda row: row["onset_within_record"]),
        "n_primary_evaluable": n_primary,
        "n_primary_detected": n_detected,
        "primary_sensitivity": fraction(n_detected, n_primary),
        "n_primary_excluded_left_censored": count(lambda row: row["primary_exclusion_reason"] == "onset_left_censored"),
        "n_primary_excluded_no_observed_window": count(lambda row: row["primary_exclusion_reason"] == "no_observed_warning_window"),
        "n_primary_evaluable_incomplete_window": count(lambda row: row["primary_evaluable"] and not row["warning_window_complete"]),
        "n_primary_evaluable_complete_window": count(lambda row: row["primary_evaluable"] and row["warning_window_complete"]),
        "n_primary_evaluable_onset_after_record_end": count(lambda row: row["primary_evaluable"] and row["onset_after_record_end"]),
        "n_primary_detected_onset_after_record_end": count(lambda row: row["primary_detected"] and row["onset_after_record_end"]),
        "n_secondary_evaluable": n_secondary,
        "n_secondary_detected": n_secondary_detected,
        "secondary_sensitivity": fraction(n_secondary_detected, n_secondary),
        "all_positive_patient_detection_lower_bound": fraction(n_detected, n_septic),
        "mean_first_useful_warning_lead_hours": float(np.mean(leads)) if leads else None,
        "median_first_useful_warning_lead_hours": float(np.median(leads)) if leads else None,
        "n_patients_with_alert": count(lambda row: row["any_alert"]),
        "n_nonseptic_patients_with_alert": false_alarm_patients,
        "nonseptic_false_alarm_rate": fraction(false_alarm_patients, len(nonseptic)),
        "nonseptic_patient_hours": int(nonseptic_hours),
        "nonseptic_alert_positive_hours": int(nonseptic_alert_hours),
        "nonseptic_alert_episodes": int(nonseptic_episodes),
        "nonseptic_alert_positive_hours_per_100_patient_hours": fraction(100 * nonseptic_alert_hours, nonseptic_hours),
        "nonseptic_alert_episodes_per_100_patient_hours": fraction(100 * nonseptic_episodes, nonseptic_hours),
        "secondary_sensitivity_by_min_lead_hours": {},
        "definitions": {
            "onset": "first positive label hour + label_shift_hours only after an observed 0-to-1 transition",
            "left_censored": "positive at first observed row: onset is unknown, excluded from onset-based detection denominators",
            "primary": "any emitted alert in inclusive [onset-max_lead_hours, onset-min_lead_hours]; at least one window hour must be observed",
            "secondary": "same alert rule, restricted to onset within record and fully observed warning window",
            "alert_emission": "causal consecutive-positive confirmation; emit at confirming hour and each qualifying positive hour afterward; no cooldown",
            "alert_episode": "a rising edge in the emitted hourly alert sequence, including an alert at the first observed hour",
            "nonseptic": "no positive SepsisLabel in the observed record",
            "false_alarm_rate": "nonseptic patients with at least one emitted alert / all nonseptic patients",
            "burden_denominator": "one observed row equals one patient-hour; only never-positive-label patients contribute",
            "first_useful_lead": "onset minus the first alert inside the primary warning window, among detected primary-evaluable patients",
            "lower_bound": "known primary detections / all positive-label patients; a descriptive lower bound, not a point estimate for uncertain-onset patients",
            "utility": "these patient-level measures and alert episodes are not the official PhysioNet challenge utility score",
        },
    }
    for minimum in policy["secondary_min_lead_hours"]:
        prefix = _lead_prefix(minimum)
        evaluable = count(lambda row: row[f"{prefix}_evaluable"])
        detected = count(lambda row: row[f"{prefix}_detected"])
        result["secondary_sensitivity_by_min_lead_hours"][_lead_key(minimum)] = {
            "min_lead_hours": minimum,
            "max_lead_hours": policy["max_lead_hours"],
            "n_evaluable": evaluable,
            "n_detected": detected,
            "sensitivity": fraction(detected, evaluable),
        }
    return result
