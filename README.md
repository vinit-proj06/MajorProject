# SepsiAlert: Sepsis Prediction and ICU Patient Replay

SepsiAlert is a research project that uses hourly ICU records to study sepsis prediction and early warning. It includes patient-wise preprocessing, XGBoost and Random Forest models, model evaluation, SHAP explanations, and a local dashboard for replaying recorded patients.

The saved dataset contains **40,336 patients**, **1,552,210 hourly records**, and **187 model features**. Patients are separated into training, validation, and test sets. The dashboard uses saved XGBoost models, including the timing-feature experiments.

## Run the dashboard

Open this project folder in VS Code and run this command in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B run_dashboard.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) and keep the terminal running. Stop the server with **Ctrl+C**. For a different port, add `--port 8766` to the command. Existing saved models are sufficient for the demonstration; training is a separate step.

The dashboard needs the saved `models/`, `processed_data/`, `model_cache/`, and model results in `reports/`, plus the original patient files. The current configuration and saved patient manifest refer to:

```text
archive/training_setA/
archive/training_setB/
```

These paths are relative to the project folder. Include the complete `archive/` folder when sharing a folder ZIP, preserving the nested training-set directories. The saved patient manifest uses the same relative paths. Raw data remains Git-ignored, so a Git-only archive will not include it. See the [dashboard guide](docs/DASHBOARD.md) for the presentation walkthrough and data requirements.

## Project structure

```text
MajorProjectCopy/
  archive/                     Raw patient records (training_setA and training_setB)
  dashboard/                   HTML, CSS and JavaScript interface
  src/                         Preprocessing, model and replay logic
  scripts/                     Training, evaluation and experiment commands
  models/                      Saved models, thresholds and feature lists
  processed_data/              Processed splits and dataset metadata
  model_cache/                 Disk-backed model inputs used by replay and training
  reports/                     Evaluation results, figures and validation records
  tests/                       Regression tests
  docs/                        Detailed methodology and usage guides
  requirements/                Dependency lists and recorded package versions
  config.json                  Dataset and preprocessing settings
  model_config.json            Training and evaluation settings
  early_detection_config.json  Early-warning evaluation settings
  shap_config.json             Explanation settings
  run_dashboard.py             Dashboard entry point
  run_pipeline.py              Preprocessing entry point
  requirements.txt             Full-project dependency entry point
  README.md                    Project overview
```

## Setup and other commands

The existing `.venv` can be used directly on this computer. For a fresh Python 3.11+ environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For preprocessing alone, install `requirements/base.txt`. Run the following commands from the project root when those stages are needed:

| Task | Command |
|---|---|
| Preprocess raw records | `.\.venv\Scripts\python.exe run_pipeline.py` |
| Train XGBoost | `.\.venv\Scripts\python.exe scripts/train_xgboost.py` |
| Train Random Forest | `.\.venv\Scripts\python.exe scripts/train_random_forest.py` |
| Compare saved models | `.\.venv\Scripts\python.exe scripts/compare_models.py` |
| Evaluate early warnings | `.\.venv\Scripts\python.exe scripts/evaluate_early_detection.py` |
| Generate SHAP reports | `.\.venv\Scripts\python.exe scripts/explain_xgboost.py` |
| View timing-experiment options | `.\.venv\Scripts\python.exe scripts/run_timing_ablation.py --help` |
| Run regression tests | `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` |

Existing outputs are protected by their respective commands. Read the relevant guide before rebuilding data or models.

The saved training cache and experiment manifests record the original workspace paths. Training or evaluation reruns in this copied folder may reject those historical fingerprints; timing-experiment resume also verifies source hashes. Moving scripts does not migrate saved experiment provenance. The dashboard reuses the saved artifacts without rebuilding them.

## Results and scope

The saved test reports show XGBoost AUROC **0.8241** and average precision **0.1142**; Random Forest AUROC **0.8083** and average precision **0.1029**. See the [model comparison](reports/models/comparison.txt) and [early-warning comparison](reports/early_detection/comparison.txt) for thresholds, precision, recall, timing definitions, and false alerts.

This is a replay of recorded structured ICU data. Model scores are uncalibrated, hourly labels are strongly imbalanced, and timing-feature comparisons are exploratory. The project does not establish clinical readiness or live hospital integration. A second data modality and independent prospective validation remain future work.

## Documentation

- [Data pipeline and feature engineering](docs/DATA_PIPELINE.md)
- [Dashboard demonstration](docs/DASHBOARD.md)
- [Model training and evaluation](docs/MODEL_TRAINING.md)
- [Patient-level early detection](docs/EARLY_DETECTION.md)
- [SHAP explainability](docs/SHAP_EXPLAINABILITY.md)
- [Timing-feature experiments](docs/ABLATION_EXPERIMENTS.md)
