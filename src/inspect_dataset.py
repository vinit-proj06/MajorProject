"""Discover all patients and inspect representative files before any model fitting."""
from __future__ import annotations
import argparse
from collections import Counter
import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.common import discover_files, load_config, read_header, read_patient, save_json, setup_logging


def inspect_dataset(config):
    logging.info("Discovering files...")
    records, discovery = discover_files(config)
    for dataset, count in discovery["patients_by_dataset"].items():
        logging.info("Training Set %s: %s patients", dataset, f"{count:,}")
    logging.info("Total: %s patients", f"{len(records):,}")
    for warning in discovery["warnings"]:
        logging.warning(warning)
    selected, headers, samples = [], [], []
    rng = np.random.default_rng(config["seed"])
    for dataset in sorted(config["input_dirs"]):
        subset = [r for r in records if r.dataset == dataset]
        n = min(config["inspection_samples_per_dataset"], len(subset))
        indices = {0, len(subset) // 2, len(subset) - 1}
        indices = set(sorted(indices)[:n])
        while len(indices) < n:
            indices.add(int(rng.integers(len(subset))))
        selected.extend(subset[i] for i in sorted(indices))
    for record in selected:
        try:
            header = read_header(record.path)
            headers.append(tuple(header))
        except Exception as exc:
            samples.append({"patient_id": record.patient_id, "dataset": record.dataset, "error": str(exc)})
    if not headers:
        raise ValueError("No representative PSV has a readable schema")
    schema = list(Counter(headers).most_common(1)[0][0])
    if not {"SepsisLabel", "ICULOS"}.issubset(schema):
        raise ValueError("Observed schema must contain SepsisLabel and ICULOS")
    if set(schema) & {"patient_id", "dataset"}:
        raise ValueError("Source schema conflicts with pipeline metadata names")
    for record in selected:
        try:
            frame, _, _ = read_patient(record, schema, config)
            samples.append({"patient_id": record.patient_id, "dataset": record.dataset, "path": str(record.path), "rows": len(frame),
                            "first_ICULOS": float(frame.ICULOS.iloc[0]), "last_ICULOS": float(frame.ICULOS.iloc[-1]),
                            "positive_rows": int(frame.SepsisLabel.sum()), "missing_percentage": {c: float(frame[c].isna().mean() * 100) for c in schema}})
        except Exception as exc:
            samples.append({"patient_id": record.patient_id, "dataset": record.dataset, "path": str(record.path), "error": str(exc)})
    report = {"discovery": discovery, "columns": schema, "column_count": len(schema),
              "sampled_schemas_consistent": len(set(headers)) == 1, "sample_schema_variants": [list(h) for h in sorted(set(headers))],
              "scope": "Representative inspection; complete per-file schema validation occurs during ingestion.", "samples": samples}
    save_json(Path(config["reports_dir"]) / "inspection.json", report)
    lines = ["REPRESENTATIVE DATA INSPECTION", report["scope"], f"Discovered: {len(records):,}",
             f"Patients by source: {discovery['patients_by_dataset']}", f"Columns ({len(schema)}): {', '.join(schema)}",
             f"Sample schemas identical: {report['sampled_schemas_consistent']}"] + discovery["warnings"]
    for sample in samples:
        lines.append(f"{sample['dataset']} {sample['patient_id']}: " + (sample["error"] if "error" in sample else f"{sample['rows']} rows; ICULOS {sample['first_ICULOS']}-{sample['last_ICULOS']}; positive rows {sample['positive_rows']}"))
    Path(config["reports_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(config["reports_dir"]) / "inspection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Inspected %s representative files; %s columns; sampled schemas consistent=%s", len(selected), len(schema), report["sampled_schemas_consistent"])
    return records, schema, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config["reports_dir"])
    inspect_dataset(config)


if __name__ == "__main__":
    main()
