# XGBoost timing-feature ablation experiments

Run commands from the project root. File paths below are relative to that folder.

These experiments test how the existing XGBoost model depends on `ICULOS` and `HospAdmTime`. Each variant is retrained using the original training patients and training settings, with only its specified predictor columns removed. The original baseline in `models/xgboost/` and its reports in `reports/models/xgboost/` are preserved and reused for comparison.

This is an **exploratory comparison**: inspection of the existing test data and SHAP results motivated the hypothesis. Test rows are not used for fitting, early stopping, or threshold selection, but the reused test cohort is no longer an untouched confirmatory evaluation. A future untouched patient cohort, preferably from another hospital or period, is needed to confirm any finding.

## Experiment definitions

| Experiment | Predictor columns removed | Predictors retained |
|---|---|---:|
| Baseline | None; use the existing saved model | 187 |
| A | `ICULOS` | 186 |
| B | `HospAdmTime`, `HospAdmTime__missing` | 185 |
| C | `ICULOS`, `HospAdmTime`, `HospAdmTime__missing` | 184 |

The hospital-admission missingness indicator is removed with `HospAdmTime` in B and C so that this indicator does not remain as a separate predictor from the removed variable. `ICULOS` has no corresponding missingness predictor in the original feature list.

All other retained feature values, feature order, row order, target labels, and patient split assignments stay unchanged. There is no new imputation, normalization, feature engineering, row sampling, or resplitting. The original hourly `ICULOS` remains available as **metadata** in prediction files and for temporal evaluation, including for A and C. Its previous use as the time coordinate in causal rolling features is unchanged.

`src/feature_ablation.py` reuses `fit_xgboost`, `predict_xgboost`, and `save_xgboost` from `scripts/train_xgboost.py`. Its `ColumnSubset` adapter selects columns only after taking a bounded row slice of the existing read-only cache. It does not create a complete in-memory matrix for an experiment.

## Run and resume

Use the existing project environment and the original configuration files. The model dependencies are described in [MODEL_TRAINING.md](MODEL_TRAINING.md).

From Windows PowerShell:

```powershell
cd "C:\Users\LENOVO\Desktop\MajorProjectCopy"
```

For the first run, when the timing-ablation output directories do not exist:

```powershell
.\.venv\Scripts\python.exe scripts/run_timing_ablation.py
```

For existing runs:

```powershell
.\.venv\Scripts\python.exe scripts/run_timing_ablation.py --resume
```

`--resume` verifies and reuses completed experiments instead of retraining or overwriting their models. It checks the original settings, evaluation policy, protected source hashes, and saved experiment artifacts, then refreshes comparisons. A partially trained experiment is not a completed checkpoint: existing incomplete model/report directories cause an explicit error rather than being overwritten. The command does not resume from an interrupted boosting iteration. Do not start a second process while a run is active.

Optional `--config path/to/model_config.json` and `--early-config path/to/early_detection_config.json` select configuration files, but their resolved settings must match the saved baseline and existing early-detection policy. There is no separate timing-ablation configuration file.

The runner trains A, B, and C sequentially, audits their saved predictions, evaluates the same early-warning policy, and generates full-test SHAP explanations for C. The baseline is not retrained. An in-progress comparison may contain only completed experiments; `reports/timing_ablation/run_manifest.json` must show `"status": "complete"` before treating the overall run as finished.

## Training and selection protocol

The original settings in `model_config.json` are retained:

| Setting | Value or rule |
|---|---|
| Random seed | 42 |
| CPU threads | 2 |
| Data batch size | 8,192 hourly rows |
| Objective and tree method | `binary:logistic`, CPU histogram trees |
| Maximum boosting rounds | 400 |
| Early-stopping patience | 30 rounds |
| Early-stopping data and metric | Validation patients only; maximize XGBoost `aucpr` |
| Learning rate | 0.05 |
| Maximum depth | 5 |
| Minimum child weight | 10 |
| Row and column subsampling | 0.8 each |
| L2 regularization | 5.0 |
| Histogram bins | 128 |
| Positive-class weight | Training negative-row count / training positive-row count |

The original `ExtMemQuantileDMatrix` workflow uses disk-backed training data. Validation quantiles reference the training quantile cuts. The class-weight calculation and original package versions are checked against the baseline.

Each experiment retains its own best validation iteration. Its alert threshold independently maximizes **hourly validation F1**, with ties choosing the **highest threshold**. Predictions are positive when `probability >= threshold`. Test patients are evaluated only after that threshold is saved. Different retained feature sets can therefore produce different tree counts and thresholds while using the same selection procedure.

No probability calibrator is fitted. Calibration metrics assess the raw, class-weighted model scores. XGBoost's early-stopping `aucpr` metric and the final scikit-learn average-precision metric are recorded separately; they should not be assumed numerically identical.

## Evaluation definitions

**AUPRC (AP)** is scikit-learn average precision over all original hourly records: precision weighted by increments in recall. It is not trapezoidal interpolation of the precision-recall curve. Higher values indicate better ranking on this cohort, whose original class prevalence is preserved.

**Brier score** is `mean((probability - SepsisLabel) ** 2)`. Lower is better; it measures overall probability error and is influenced by the severe class imbalance. It is not a pure calibration-only statistic.

**Expected calibration error (ECE)** uses the original **10 fixed uniform probability bins**. It averages the absolute difference between each nonempty bin's mean predicted probability and observed positive fraction, weighted by its row count. Bins are `[lower, upper)` except that the last includes 1. Lower is better under this binning rule; ECE depends on the chosen bins and can hide finer differences. Reliability plots and Brier scores accompany it.

Accuracy, precision, recall/sensitivity, specificity, F1, AUROC, confusion matrices, and the remaining original evaluation artifacts are also saved separately for each experiment. Their definitions are in [MODEL_TRAINING.md](MODEL_TRAINING.md).

**Patient-level early detection** retains the existing policy:

- After an observed `SepsisLabel` transition from 0 to 1, inferred onset is the first positive label's `ICULOS` plus 6 hours. This is a label-derived timing proxy, not a separately observed clinical onset measurement.
- The primary warning window is inclusive: `[inferred onset - 12, inferred onset - 6]`. A complete window contains seven hourly observation times.
- Any threshold-positive hour in that window counts as a detection. The current policy requires one positive hour, with no cooldown. A continuing or repeated alert inside the window can count even if the patient's first-ever alert occurred earlier.
- Patients with at least one observed warning-window hour are eligible; partially observed windows remain in the primary denominator. The existing stricter full-window metric is saved separately.
- Test timing denominators remain **375 eligible patients** and **64 excluded patients whose first observed label is already positive**. These 64 have unknown onset and are not counted as timing misses. They remain in the original hourly classification metrics.

The **nonseptic patient false-alert rate** is the fraction of the **5,611 test patients with no positive label during their observed record** who receive at least one alert at any recorded hour. It is not the hourly false-positive rate. Alert episodes per 100 patient-hours are also retained because recording length affects the opportunity to trigger an alert.

Thresholds optimize validation F1 independently; the variants are **not compared at a matched false-alert rate**. Interpret detection and alarm burden together. Further timing definitions are in [EARLY_DETECTION.md](EARLY_DETECTION.md).

## Output files

```text
models/
  xgboost/                         original baseline, preserved
  timing_ablation/
    A/                             186-predictor saved model
    B/                             185-predictor saved model
    C/                             184-predictor saved model
      xgboost.ubj
      feature_columns.json
      config_resolved.json
      ablation_specification.json
      learning_history.json
      threshold.json
      run_manifest.json
reports/
  models/xgboost/                   original baseline reports, preserved
  timing_ablation/
    comparison.md
    comparison.csv
    comparison.json
    comparison.png
    run_manifest.json
    Baseline/early_detection/       audited baseline timing results
    A/                             separate results/predictions and early_detection/
    B/                             separate results/predictions and early_detection/
    C/
      results.json
      artifact_hashes.json
      training.log
      validation_predictions.csv.gz
      test_predictions.csv.gz
      validation_metrics.json
      test_metrics.json
      early_detection/
        validation_patients.csv
        validation_metrics.json
        test_patients.csv
        test_metrics.json
      shap/
        feature_importance_full.csv
        feature_ranking_full.txt
        feature_importance_full.png / .pdf
        importance_elbow.png / .pdf
        shap_summary_all_features.png / .pdf
        feature_columns.json
        test_feature_values.npy
        shap_values.npy
        base_values.npy
        raw_margins.npy
        probabilities.npy
        test_row_index.csv.gz
        run_manifest.json
```

A and B contain the same training/evaluation artifact types shown for C, including their own thresholds, confusion matrices, calibration tables, and evaluation plots. Only C receives the additional full-test SHAP export in this workflow.

Open the main comparison after completion:

```powershell
notepad ".\reports\timing_ablation\comparison.md"
Invoke-Item ".\reports\timing_ablation\comparison.png"
explorer ".\reports\timing_ablation"
```

Experiment C's SHAP computation uses the saved best model and **all 231,884 test hours across all 184 retained features**. It performs no further training, row subsampling, or feature cutoff. Mean absolute importance weights each recorded hour equally, so longer records contribute more observations. The distribution plot bins observations for display while retaining all rows. Contributions are additive in raw model log-odds, and correlated or engineered predictors can share attribution. See [SHAP_EXPLAINABILITY.md](SHAP_EXPLAINABILITY.md) for the general interpretation.

## Interpreting the comparison

The generated comparison contains the actual measured results. This guide specifies the protocol and does not predeclare a winning experiment or invent results while training is in progress.

Removing explicit timing predictors does not remove all possible timing, hospital, measurement-frequency, or care-process signals. Other retained variables can proxy them. Conversely, a performance reduction does not by itself prove the removed feature was illegitimate: it shows that the fitted model used information associated with that feature under this dataset and protocol.

These are single-seed retrospective research comparisons with correlated hourly observations, not statistical-significance claims or evidence of clinical safety. Any model or feature choice based on these results needs confirmation using data untouched by the current analysis.
