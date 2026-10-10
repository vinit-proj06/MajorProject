"""Read-only structured ICU replay, causal scoring, and native SHAP for a demo.

Ordinary snapshots expose only the chosen history prefix. Outcome labels and
label-derived onset are returned exclusively by the explicit retrospective API.
No models, preprocessing parameters, patient files or project outputs are saved.
"""
from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import xgboost as xgb

from src.common import PROJECT_ROOT, PatientFile, read_patient
from src.early_detection import evaluate_patient
from src.feature_engineering import engineer_patient
from src.model_data import close_arrays, open_split
from src.preprocess import apply_parameters
from src.shap_explain import compute_contributions


class DashboardInputError(ValueError):
    """Invalid user-supplied replay selection (HTTP 400)."""


class PatientNotFoundError(LookupError):
    """Patient is not a member of the original test cohort (HTTP 404)."""


class ArtifactIntegrityError(RuntimeError):
    """A saved source/model artifact fails its integrity contract (HTTP 500)."""


VITALS = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2")
NON_LABS = set(VITALS) | {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"}
# Display units for common bedside measurements; other fields retain source names.
UNITS = {"HR": "beats/min", "O2Sat": "%", "Temp": "°C", "SBP": "mmHg", "MAP": "mmHg",
         "DBP": "mmHg", "Resp": "breaths/min", "EtCO2": "mmHg", "Age": "years",
         "HospAdmTime": "hours", "ICULOS": "hours"}
MODEL_SPECS = {
    "baseline": ("XGBoost baseline", [], "Original 187-predictor research baseline."),
    "A": ("A · without ICU time", ["ICULOS"], "Exploratory ablation of the ICULOS predictor."),
    "B": ("B · without admission time", ["HospAdmTime", "HospAdmTime__missing"], "Exploratory ablation of hospital-admission timing."),
    "C": ("C · without both time predictors", ["ICULOS", "HospAdmTime", "HospAdmTime__missing"], "Exploratory ablation of both explicit timing inputs."),
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value):
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    if not np.isfinite(converted):
        raise ArtifactIntegrityError("A nonfinite value reached the dashboard payload")
    return converted


def _stat(path):
    item = Path(path).stat()
    return item.st_size, item.st_mtime_ns


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class DashboardData:
    """Patient-bounded replay service; model objects and at most four raw cases cache in RAM."""

    def __init__(self, root=PROJECT_ROOT, patient_cache_size=4):
        if type(patient_cache_size) is not int or not 1 <= patient_cache_size <= 4:
            raise DashboardInputError("patient_cache_size must be an integer from 1 to 4")
        self.root = Path(root).resolve()
        self.patient_cache_size = patient_cache_size
        self._lock = threading.RLock()
        self._patient_cache = OrderedDict()
        self._models = {}
        self._closed = False
        self.arrays = {}
        self.model_config = _json(self.root / "model_config.json")
        for key in ("processed_dir", "cache_dir", "models_dir", "reports_dir"):
            self.model_config[key] = str((self.root / self.model_config[key]).resolve())
        self.processed_dir = Path(self.model_config["processed_dir"])
        self.cache_dir = Path(self.model_config["cache_dir"])
        self.pipeline_config = _json(self.processed_dir / "config_resolved.json")
        self.feature_info = _json(self.processed_dir / "feature_metadata.json")
        self.features = list(self.feature_info["predictor_columns"])
        self.imputation_parameters = _json(self.processed_dir / "imputation_parameters.json")
        self.schema = _json(self.processed_dir / "schema.json")["columns"]
        self.cache_manifest = _json(self.cache_dir / "cache_manifest.json")
        pipeline_manifest = _json(self.processed_dir / "pipeline_manifest.json")
        self.policy = _json(self.root / "early_detection_config.json")["policy"]
        if (pipeline_manifest.get("status") != "complete" or not pipeline_manifest.get("validation_passed")
                or self.imputation_parameters.get("fit_split") != "train"
                or self.features != self.cache_manifest["features"]
                or len(self.features) != len(set(self.features))):
            raise ArtifactIntegrityError("A complete, consistent, training-fitted preprocessing run is required")
        if self.policy.get("label_shift_hours") != 6:
            raise ArtifactIntegrityError("The saved early-detection policy must retain the six-hour label offset")
        if type(self.policy.get("consecutive_positive_hours")) is not int or self.policy["consecutive_positive_hours"] < 1:
            raise ArtifactIntegrityError("Invalid saved alert confirmation policy")

        with (self.processed_dir / "test_patient_ids.csv").open(newline="", encoding="utf-8") as handle:
            selected = list(csv.DictReader(handle))
        memberships = {row["patient_id"]: row["dataset"] for row in selected}
        if not memberships or len(memberships) != len(selected):
            raise ArtifactIntegrityError("Test patient IDs must be nonempty and unique")
        self._patients = {}
        with (self.processed_dir / "patient_manifest.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pid = row["patient_id"]
                if pid in memberships:
                    if pid in self._patients or memberships[pid] != row["dataset"]:
                        raise ArtifactIntegrityError("Duplicate or inconsistent test patient manifest entry")
                    self._patients[pid] = row
        if set(self._patients) != set(memberships):
            raise ArtifactIntegrityError("The patient manifest does not cover the test split")
        try:
            self.arrays = open_split(self.cache_dir, "test")
            ids = self.arrays["patient_id"]
            boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
            self._patient_bounds = {}
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                pid = str(ids[start])
                if pid in self._patient_bounds or pid not in self._patients:
                    raise ArtifactIntegrityError("Cached test patient order or membership is invalid")
                if stop - start != int(self._patients[pid]["n_rows"]):
                    raise ArtifactIntegrityError("Cached test patient row count differs from its manifest")
                self._patient_bounds[pid] = (int(start), int(stop))
            if set(self._patient_bounds) != set(self._patients):
                raise ArtifactIntegrityError("Test cache and saved test memberships differ")
        except Exception:
            close_arrays(self.arrays)
            raise
        models_root, reports_root = Path(self.model_config["models_dir"]), Path(self.model_config["reports_dir"])
        self._model_paths = {"baseline": (models_root / "xgboost", reports_root / "xgboost")}
        for name in ("A", "B", "C"):
            model_dir = self.root / "models" / "timing_ablation" / name
            report_dir = self.root / "reports" / "timing_ablation" / name
            if (model_dir / "xgboost.ubj").is_file() and (report_dir / "results.json").is_file():
                self._model_paths[name] = (model_dir, report_dir)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("DashboardData has been closed")

    def close(self):
        with self._lock:
            if not self._closed:
                close_arrays(self.arrays)
                self.arrays = {}
                self._patient_cache.clear()
                self._models.clear()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def catalog(self, query="", limit=50):
        self._ensure_open()
        if not isinstance(query, str) or len(query) > 64:
            raise DashboardInputError("query must be a string of at most 64 characters")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise DashboardInputError("limit must be an integer from 1 to 200")
        query = query.strip().casefold()
        return [{"patient_id": pid, "dataset": self._patients[pid]["dataset"]}
                for pid in sorted(self._patients) if query in pid.casefold()][:limit]

    def _load_patient(self, patient_id):
        self._ensure_open()
        if not isinstance(patient_id, str) or patient_id not in self._patients:
            raise PatientNotFoundError("Patient is not in the original test cohort")
        record = self._patients[patient_id]
        source = Path(record["filename"])
        source = (source if source.is_absolute() else self.root / source).resolve()
        allowed = Path(self.pipeline_config["input_dirs"][record["dataset"]])
        allowed = (allowed if allowed.is_absolute() else self.root / allowed).resolve()
        if allowed not in source.parents or source.stem != patient_id or source.suffix.lower() != ".psv":
            raise ArtifactIntegrityError("Patient manifest points outside its original source directory")
        stamp = _stat(source)
        if patient_id in self._patient_cache:
            frame, previous_stamp = self._patient_cache[patient_id]
            if previous_stamp == stamp:
                self._patient_cache.move_to_end(patient_id)
                return frame
            del self._patient_cache[patient_id]
        try:
            frame, digest, _ = read_patient(PatientFile(patient_id, record["dataset"], source), self.schema, self.pipeline_config)
        except (ValueError, UnicodeError) as error:
            raise ArtifactIntegrityError("Raw patient file fails the saved preprocessing validation") from error
        if digest != record["sha256"]:
            raise ArtifactIntegrityError("Raw patient file differs from the saved preprocessing SHA256")
        start, stop = self._patient_bounds[patient_id]
        if (len(frame) != stop - start
                or not np.array_equal(frame["ICULOS"].to_numpy(), self.arrays["ICULOS"][start:stop])
                or not np.array_equal(frame["SepsisLabel"].to_numpy(), self.arrays["y"][start:stop])
                or set(self.arrays["dataset"][start:stop]) != {record["dataset"]}):
            raise ArtifactIntegrityError("Raw patient time, labels or source differs from the test cache")
        self._patient_cache[patient_id] = (frame, stamp)
        self._patient_cache.move_to_end(patient_id)
        while len(self._patient_cache) > self.patient_cache_size:
            self._patient_cache.popitem(last=False)
        return frame

    @staticmethod
    def _hour(frame, hour):
        if isinstance(hour, (bool, np.bool_)):
            raise DashboardInputError("hour must identify an exact recorded integer ICU hour")
        try:
            value = float(hour)
        except (TypeError, ValueError) as error:
            raise DashboardInputError("hour must identify an exact recorded integer ICU hour") from error
        if not np.isfinite(value) or value != np.floor(value) or not (frame["ICULOS"] == value).any():
            raise DashboardInputError("hour must identify an exact recorded integer ICU hour")
        return value

    def patient_info(self, patient_id):
        with self._lock:
            frame = self._load_patient(patient_id)
            first = frame.iloc[0]
            return {"patient_id": patient_id, "dataset": str(first["dataset"]),
                    "first_hour": float(first["ICULOS"]), "last_hour": float(frame["ICULOS"].iloc[-1]),
                    "replay_rows": len(frame), "age": _number(first.get("Age")),
                    "gender": _number(first.get("Gender")), "mode": "retrospective_replay"}

    def _load_model(self, model):
        self._ensure_open()
        if not isinstance(model, str) or model not in self._model_paths:
            raise DashboardInputError("Unknown saved model; choose an available model ID")
        model_dir, report_dir = self._model_paths[model]
        paths = [model_dir / "xgboost.ubj", model_dir / "feature_columns.json",
                 model_dir / "threshold.json", report_dir / "results.json"]
        signature = tuple(_stat(path) for path in paths)
        if model in self._models:
            cached = self._models[model]
            if cached["signature"] != signature:
                raise ArtifactIntegrityError("Saved model artifacts changed during the dashboard session")
            return cached
        name, excluded, description = MODEL_SPECS[model]
        names, threshold_info, result = _json(paths[1]), _json(paths[2]), _json(paths[3])
        expected = [name for name in self.features if name not in excluded]
        if (names != expected or result.get("status") != "complete" or result.get("model") != "xgboost"
                or result.get("cache_fingerprint") != self.cache_manifest["fingerprint"]
                or result.get("feature_count") != len(names)
                or threshold_info != result.get("threshold_selection")
                or result.get("feature_order_sha256") != hashlib.sha256(json.dumps(names).encode()).hexdigest()):
            raise ArtifactIntegrityError("Saved model features, threshold or preprocessing identity differ")
        threshold = float(threshold_info["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ArtifactIntegrityError("Saved threshold is invalid")
        booster = xgb.Booster(params={"nthread": int(self.model_config["n_jobs"]), "device": "cpu"}, model_file=paths[0])
        if (booster.feature_names != names or booster.num_features() != len(names)
                or booster.best_iteration != result["training_details"]["best_iteration"]
                or booster.num_boosted_rounds() != result["training_details"]["trees_saved"]):
            raise ArtifactIntegrityError("Loaded model differs from the recorded fitted model")
        bundle = {"id": model, "name": name, "label": name, "description": description,
                  "booster": booster, "features": names, "feature_count": len(names),
                  "threshold": threshold, "best_iteration": int(booster.best_iteration),
                  "calibrated": False, "signature": signature, "model_sha256": _hash(paths[0])}
        self._models[model] = bundle
        return bundle

    def model_catalog(self):
        with self._lock:
            result = []
            for name in self._model_paths:
                bundle = self._load_model(name)
                result.append({key: bundle[key] for key in ("id", "name", "label", "description", "feature_count", "threshold", "calibrated")})
            return result

    def _processed_prefix(self, patient_id, hour):
        raw = self._load_patient(patient_id)
        hour = self._hour(raw, hour)
        # Remove the target and all future rows BEFORE any transform. No statistics
        # are estimated here; only the saved training parameters are applied.
        prefix = raw.loc[raw["ICULOS"] <= hour].drop(columns="SepsisLabel").copy()
        processed = apply_parameters(engineer_patient(prefix, self.pipeline_config),
                                     self.feature_info["features"], self.imputation_parameters)
        expected_columns = set(self.features) | {"patient_id", "dataset"}
        if set(processed.columns) != expected_columns:
            raise ArtifactIntegrityError("Dynamic prefix feature schema differs from the saved preprocessing run")
        values = processed[self.features].to_numpy(dtype=np.float32)
        start, _ = self._patient_bounds[patient_id]
        if not np.array_equal(values, self.arrays["X"][start:start + len(processed)]):
            raise ArtifactIntegrityError("Recomputed causal prefix differs from saved model inputs")
        return processed

    def _score_prefix(self, patient_id, hour, model):
        raw = self._load_patient(patient_id)
        hour = self._hour(raw, hour)
        prefix = raw.loc[raw["ICULOS"] <= hour].drop(columns="SepsisLabel").copy()
        processed = self._processed_prefix(patient_id, hour)
        bundle = self._load_model(model)
        values = processed[bundle["features"]].to_numpy(dtype=np.float32)
        probabilities = np.asarray(bundle["booster"].inplace_predict(
            values, iteration_range=(0, bundle["best_iteration"] + 1)), dtype=np.float64)
        if (probabilities.shape != (len(prefix),) or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))):
            raise ArtifactIntegrityError("Saved model returned invalid replay scores")
        return prefix, processed, bundle, probabilities

    def _alerts(self, probabilities, threshold):
        emitted = np.zeros(len(probabilities), dtype=bool)
        streak = 0
        for index, positive in enumerate(probabilities >= threshold):
            streak = streak + 1 if positive else 0
            emitted[index] = streak >= self.policy["consecutive_positive_hours"]
        return emitted

    def snapshot(self, patient_id, hour, model="baseline"):
        with self._lock:
            prefix, processed, bundle, probabilities = self._score_prefix(patient_id, hour, model)
            current, model_row = prefix.iloc[-1], processed.iloc[-1]
            hour = float(current["ICULOS"])
            threshold = bundle["threshold"]
            alerts = self._alerts(probabilities, threshold)
            values = processed[bundle["features"]].iloc[-1:].to_numpy(dtype=np.float32)
            explained = compute_contributions(bundle["booster"], values, bundle["features"])
            probability = float(explained["probabilities"][0])
            if (not np.isclose(probability, probabilities[-1], rtol=1e-6, atol=1e-7)
                    or (probability >= threshold) != (probabilities[-1] >= threshold)):
                raise ArtifactIntegrityError("Explanation and history disagree about the current score")
            shap_values = explained["values"][0]
            ordering = np.argsort(-np.abs(shap_values), kind="stable")
            contributions = [{"feature": bundle["features"][index], "feature_value": float(values[0, index]),
                              "shap_value": float(shap_values[index]),
                              "direction": "positive" if shap_values[index] > 0 else "negative" if shap_values[index] < 0 else "zero"}
                             for index in ordering]
            vitals = {name: _number(current.get(name)) for name in VITALS if name in prefix}
            labs = {name: _number(current[name]) for name in self.schema if name not in NON_LABS and name in prefix}
            measurements = []
            for name in [*vitals, *labs]:
                observed = prefix.loc[prefix[name].notna(), ["ICULOS", name]]
                last_hour = float(observed["ICULOS"].iloc[-1]) if len(observed) else None
                last_value = _number(observed[name].iloc[-1]) if len(observed) else None
                missing = bool(pd.isna(current[name]))
                elapsed = float(hour - last_hour) if last_hour is not None else None
                carried = (missing and name in self.pipeline_config["forward_fill_features"]
                           and elapsed is not None and elapsed <= self.pipeline_config["forward_fill_limit_hours"])
                source = "observed" if not missing else "forward_filled" if carried else "training_imputed"
                measurements.append({"feature": name, "unit": UNITS.get(name), "raw_value": _number(current[name]),
                                     "model_value": _number(model_row.get(name)), "missing": missing, "source": source,
                                     "last_observed_hour": last_hour, "last_observed_value": last_value,
                                     "hours_since_observed": elapsed, "included_in_model": name in bundle["features"]})
            history = [{"hour": float(row["ICULOS"]), **{name: _number(row[name]) for name in vitals}}
                       for _, row in prefix.iterrows()]
            score_history = [{"hour": float(t), "model_score": float(p), "threshold": threshold, "alert": bool(a)}
                             for t, p, a in zip(prefix["ICULOS"], probabilities, alerts)]
            alert_log = [{**score_history[index], "episode_start": bool(index == 0 or not alerts[index - 1])}
                         for index in np.flatnonzero(alerts)]
            result = {
                "patient_id": patient_id, "dataset": str(current["dataset"]), "hour": hour,
                "mode": "retrospective_replay", "prefix_rows": len(prefix),
                "model": {key: bundle[key] for key in ("id", "name", "feature_count", "best_iteration", "calibrated")},
                "model_score": probability, "probability": probability, "threshold": threshold,
                "alert": bool(alerts[-1]), "current_vitals": vitals, "current_labs": labs,
                "measurements": measurements, "history": history, "score_history": score_history, "alerts": alert_log,
                "shap": {"base_value": float(explained["base_values"][0]), "margin": float(explained["margins"][0]),
                         "probability": probability, "units": "log_odds", "max_additivity_error": explained["max_additivity_error"],
                         "features": contributions,
                         "positive": sorted((row for row in contributions if row["shap_value"] > 0), key=lambda row: -row["shap_value"]),
                         "negative": sorted((row for row in contributions if row["shap_value"] < 0), key=lambda row: row["shap_value"])},
                "freshness": {"observed_vitals": sum(value is not None for value in vitals.values()),
                              "missing_vitals": [name for name, value in vitals.items() if value is None]},
            }
            json.dumps(result, allow_nan=False)
            return result

    def retrospective(self, patient_id, model="baseline"):
        """Intentionally reveal full-record labels and outcome-derived timing for review."""
        with self._lock:
            raw = self._load_patient(patient_id)
            prefix, _, bundle, probabilities = self._score_prefix(patient_id, float(raw["ICULOS"].iloc[-1]), model)
            threshold = bundle["threshold"]
            predicted = (probabilities >= threshold).astype(np.uint8)
            frame = pd.DataFrame({"patient_id": patient_id, "dataset": raw["dataset"].to_numpy(),
                                  "ICULOS": raw["ICULOS"].to_numpy(), "SepsisLabel": raw["SepsisLabel"].to_numpy(),
                                  "probability": probabilities, "predicted_SepsisLabel": predicted})
            summary = evaluate_patient(frame, threshold, self.policy)
            alerts = self._alerts(probabilities, threshold)
            result = {"patient_id": patient_id, "dataset": str(raw["dataset"].iloc[0]), "model_id": bundle["id"],
                      "threshold": threshold, "mode": "retrospective_review", "policy": dict(self.policy),
                      "timeline": [{"hour": float(t), "label": int(y), "model_score": float(p), "alert": bool(a)}
                                   for t, y, p, a in zip(prefix["ICULOS"], raw["SepsisLabel"], probabilities, alerts)],
                      "summary": summary}
            json.dumps(result, allow_nan=False)
            return result
