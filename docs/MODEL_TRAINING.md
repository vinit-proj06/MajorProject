# Random Forest and XGBoost experiments

Run commands from the project root. File paths below are relative to that folder.

These are separate, reproducible research baselines using the already processed data. No raw patient files need to be processed again. Both scripts use the exact same 187 predictor columns and saved patient-level splits. The target is the existing hourly `SepsisLabel`, without any additional shift.

For patient-level warning timing, run `python scripts/evaluate_early_detection.py` after training. See [Early detection](EARLY_DETECTION.md) for the 6–12 hour detection rate, onset exclusions, lead times, and false-alarm burden.

For complete feature attribution without retraining, run `python scripts/explain_xgboost.py`. See [SHAP explainability](SHAP_EXPLAINABILITY.md) for all 187 global feature ranks, full-test distributions, an elbow chart, and the p006266 hour-22 explanation.

## Run on this computer

Open PowerShell:

```powershell
cd "C:\Users\LENOVO\Desktop\MajorProjectCopy"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/train_xgboost.py
.\.venv\Scripts\python.exe scripts/train_random_forest.py
.\.venv\Scripts\python.exe scripts/compare_models.py
```

Run the two training scripts sequentially on this laptop. Each script trains its own estimator, selects a decision threshold using validation data, evaluates both validation and test, and saves its own model and reports. `--config model_config.json` is supported by all three scripts. Existing results are protected; use `--overwrite` to rebuild while retaining prior model/report directories as backups.

The first run creates `model_cache/` from compressed CSV or Parquet in bounded chunks. The next model reuses the cache after checking input fingerprints and array integrity. The cache records predictor order and checks every patient's membership, time order, labels and counts. If inputs change, use `--rebuild-cache` with a training script. Old caches are retained as backups. The fingerprint hashes metadata and uses source file sizes/modification times; it is a change detector, not an adversarial cryptographic integrity guarantee for the large source files.

On another computer, create a Python 3.11+ environment and install `requirements.txt`. The existing `.venv` here uses a bundled Python runtime; use its explicit executable as shown. Do not disable security controls if a native dependency is blocked: use an environment where the installed libraries are permitted.

## Experiment design

| Split | Patients | Hourly rows | Positive rows |
|---|---:|---:|---:|
| Training | 28,235 | 1,092,243 | 19,541 |
| Validation | 6,051 | 228,083 | 4,177 |
| Test | 6,050 | 231,884 | 4,198 |

- Training uses the existing training patients only. Patient ID, source dataset and target are metadata, never predictors. No new imputation, scaling or feature selection is fitted on held-out data.
- Fixed seed: 42. Default CPU workers: 2. All settings are centralized in `model_config.json` and copied to each model's output directory.
- Class weighting uses the ratio of negative to positive **training** rows. No validation/test rows are resampled. All training rows remain eligible for fitting.
- XGBoost: CPU histogram boosting, up to 400 rounds, depth 5, learning rate 0.05, 128 bins, validation `aucpr` early stopping after 30 rounds without improvement. Only the best iteration is retained/used. Quantile cuts are learned from training and reused for validation. Final reported AUPRC uses scikit-learn average precision, which can differ from XGBoost's stopping metric.
- Random Forest: 160 trees, depth at most 14, at most 1,024 leaves per tree, at least 20 rows per leaf, square-root feature subsampling. Each tree draws up to 200,000 bootstrap rows from the complete training pool. This bounds individual tree computation; it is not a permanent subset of the training cohort. Trees are added in batches of 20 for progress reporting, using the same training data each time. OOB scoring is disabled because hourly rows from the same patient must not become a substitute holdout.
- Each model's operating threshold maximizes **validation F1**, with ties choosing the highest threshold. Predictions use `probability >= threshold`. The same frozen threshold is applied to the test set. Reports also contain results at 0.5.
- `scripts/compare_models.py` ranks the two fixed baselines by **validation average precision**, never by test scores. Validation scores are selection-set results and may be optimistic; test estimates remain separate. This is not an exhaustive hyperparameter search.

## Metrics and calibration

The confusion matrix is `[[TN, FP], [FN, TP]]`: rows are actual labels and columns predicted labels, each ordered 0 then 1. Reports include accuracy, precision, recall, sensitivity, specificity, F1, AUROC and AUPRC. Sensitivity is identical to recall for sepsis-positive rows. AUPRC is explicitly **average precision**; trapezoidal PR area is additionally reported under a separate name.

Calibration assessment includes a reliability diagram, bin counts, predicted probabilities versus observed positive fractions, Brier score, log loss and expected calibration error (ECE; 10 equal-width bins). ECE depends on binning. Brier/log loss measure more than calibration alone, so interpret them alongside the reliability curve. These are **raw class-weighted model scores**: a post-hoc calibrator has not been fitted, and the scores should not be assumed to be calibrated clinical risks. Probability calibration fitting would need its own properly separated development procedure.

All-negative accuracy and class prevalence are reported as baselines because only about 1.8% of hourly test labels are positive. High accuracy alone is insufficient. The same number of test rows and labels must appear in both models' confusion matrices.

## Memory and reproducibility

Input matrices are float32 NumPy memory maps, created one small batch at a time. The complete training predictor matrix is about 779 MiB on disk. XGBoost uses `ExtMemQuantileDMatrix` to store feature pages outside the main heap; labels, quantile sketches and training workspaces still use RAM. Random Forest is not an incremental/out-of-core estimator: a memmap avoids eagerly allocating the full input but does not cap physical memory usage, and tree training may page data or allocate temporary arrays. Tree size and CPU workers are bounded. Both models predict in batches.

Reports record seed, package versions, resolved settings, input-cache fingerprint, feature-order hash, timings, training class weights and threshold selection. `requirements/models-lock.txt` records the installed model dependency versions and includes the pinned preprocessing dependencies. The saved XGBoost model is UBJSON; Random Forest uses joblib. Only load joblib files from trusted sources. For inference, use the saved feature list in its exact order, the existing preprocessing parameters and the saved decision threshold.

## Files to open

```text
scripts/train_xgboost.py
scripts/train_random_forest.py
scripts/compare_models.py
model_config.json
model_cache/{train,validation,test}/     # disk-backed arrays
models/xgboost/
  xgboost.ubj
  threshold.json
  feature_columns.json
  learning_history.json
  run_manifest.json
models/random_forest/
  random_forest.joblib
  threshold.json
  feature_columns.json
  run_manifest.json
reports/models/
  comparison.txt / comparison.csv / comparison.json
  xgboost/
    results.json
    test_metrics.json
    test_evaluation.png
    test_calibration.csv
    test_confusion_matrix.csv
    test_predictions.csv.gz
    validation_*                       # corresponding development results
    training.log
  random_forest/                       # same report layout
```

View the comparison and charts after both runs:

```powershell
notepad ".\reports\models\comparison.txt"
Invoke-Item ".\reports\models\xgboost\test_evaluation.png"
Invoke-Item ".\reports\models\random_forest\test_evaluation.png"
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Interpretation limits

These are **hourly classification metrics**, including later positive hours. They do not measure time-to-first-alert, alert burden, or clinical utility, and do not alone demonstrate early detection. Observations within a patient are correlated; reported values are point estimates rather than independent-row confidence intervals. Evaluation uses the supplied label definition and the existing mixed-source patient holdout, not a prospective trial or independent-hospital validation. Use patient-level resampling if confidence intervals are added later.

Primary references: [XGBoost external memory](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html), [Random Forest parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
