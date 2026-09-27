"""Reproducible patient-level splitting, stratified by source and outcome."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")


def _allocate(size: int, ratios: np.ndarray) -> np.ndarray:
    """Largest-remainder allocation with a training representative per stratum."""
    expected = size * ratios
    counts = np.floor(expected).astype(int)
    # Stable tie-breaking follows train, validation, test order.
    remainder_order = np.argsort(-(expected - counts), kind="stable")
    for index in remainder_order[:size - counts.sum()]:
        counts[index] += 1
    if size and not counts[0]:
        donor = 1 + int(np.argmax(counts[1:]))
        counts[donor] -= 1
        counts[0] += 1
    return counts


def split_patients(manifest: pd.DataFrame, config: dict, output_dir: Path) -> tuple[dict, dict]:
    """Split each patient once and persist deterministic patient manifests.

    The manifest contains only successfully validated patients. Labels are used
    solely for patient-level stratification and never emitted as model features.
    """
    required = {"patient_id", "dataset", "positive_patient", "n_rows", "positive_rows"}
    absent = required.difference(manifest.columns)
    if absent:
        raise ValueError(f"Patient manifest is missing columns: {sorted(absent)}")
    if manifest.empty:
        raise ValueError("Cannot split an empty patient manifest")
    if manifest["patient_id"].isna().any() or manifest["patient_id"].duplicated().any():
        raise ValueError("Patient IDs must be nonmissing and globally unique")
    if manifest["dataset"].isna().any() or manifest["positive_patient"].isna().any():
        raise ValueError("Stratification columns must be nonmissing")
    if not manifest["positive_patient"].isin([True, False, 0, 1]).all():
        raise ValueError("positive_patient must be boolean or binary")
    ratio_config = config["split_ratios"]
    if set(ratio_config) != set(SPLITS):
        raise ValueError(f"split_ratios must define exactly {SPLITS}")
    ratios = np.array([ratio_config[name] for name in SPLITS], dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must be positive, finite, and sum to one")
    rng = np.random.default_rng(config["seed"])
    ordered = manifest.copy()
    ordered["patient_id"] = ordered["patient_id"].astype(str)
    # Check again after canonical string conversion, e.g. integer 1 and string '1'.
    if ordered["patient_id"].duplicated().any():
        raise ValueError("Patient IDs collide after string conversion")
    ordered = ordered.sort_values(["dataset", "positive_patient", "patient_id"], kind="stable")
    mapping: dict[str, str] = {}
    strata = []
    warnings = []
    for (dataset, positive), group in ordered.groupby(["dataset", "positive_patient"], sort=True):
        identifiers = group["patient_id"].to_numpy(copy=True)
        rng.shuffle(identifiers)
        counts = _allocate(len(identifiers), ratios)
        if (counts == 0).any():
            warnings.append(f"Stratum dataset={dataset}, positive_patient={bool(positive)} has "
                            f"{len(group)} patients; some splits receive no representative.")
        offset = 0
        for split, count in zip(SPLITS, counts):
            mapping.update((identifier, split) for identifier in identifiers[offset:offset + count])
            offset += int(count)
        strata.append({"dataset": str(dataset), "positive_patient": bool(positive),
                       "patients": len(group),
                       "split_counts": {split: int(count) for split, count in zip(SPLITS, counts)}})

    split_sets = {split: {patient for patient, value in mapping.items() if value == split}
                  for split in SPLITS}
    overlaps = {f"{left}_{right}": len(split_sets[left] & split_sets[right])
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    complete = set(mapping) == set(ordered["patient_id"])
    if any(overlaps.values()) or not complete:
        raise RuntimeError("Patient split invariant failed: overlap or incomplete coverage")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": int(config["seed"]), "ratios": dict(ratio_config), "strata": strata,
              "warnings": warnings, "overlap_counts": overlaps, "complete_coverage": complete,
              "splits": {}}
    for split in SPLITS:
        patients = ordered[ordered["patient_id"].isin(split_sets[split])].sort_values("patient_id", kind="stable")
        patients[["patient_id", "dataset"]].to_csv(output_dir / f"{split}_patient_ids.csv", index=False)
        rows = int(patients["n_rows"].sum())
        positives = int(patients["positive_rows"].sum())
        report["splits"][split] = {
            "patients": len(patients), "rows": rows,
            "positive_patients": int(patients["positive_patient"].sum()),
            "negative_patients": int((~patients["positive_patient"].astype(bool)).sum()),
            "positive_rows": positives, "negative_rows": rows - positives,
            "dataset_counts": {str(key): int(value) for key, value in patients["dataset"].value_counts().items()},
        }
    return mapping, report
