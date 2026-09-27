# Sepsis research preprocessing

A reproducible, patient-wise preprocessing pipeline for PhysioNet/CinC 2019 hourly ICU records, with separate research training/evaluation scripts for XGBoost and Random Forest. See [MODEL_TRAINING.md](MODEL_TRAINING.md) for model commands, metrics and calibration reports. This project does not make a diagnosis, establish clinical safety, or implement the clinical-alerting stage in the supplied diagram.

Patient-level early warning evaluation is available through `evaluate_early_detection.py`. See [EARLY_DETECTION.md](EARLY_DETECTION.md) for the 6–12 hour detection rate, onset timing, false-alarm burden, and how to open the reports.

Full-test XGBoost feature explanations are available through `explain_xgboost.py`. See [SHAP_EXPLAINABILITY.md](SHAP_EXPLAINABILITY.md) for the complete 187-feature ranking and individual prediction explanations using the saved model.

Controlled timing-feature ablations are available through `run_timing_ablation.py`. See [ABLATION_EXPERIMENTS.md](ABLATION_EXPERIMENTS.md) for the A/B/C exclusions, first-run and resume commands, unchanged training protocol, comparison outputs, and full 184-feature SHAP ranking for experiment C. These follow-up comparisons are exploratory; the original baseline is preserved.

## Local data and inspection

`config.json` points to the existing folders on this computer:

```text
C:/Users/LENOVO/Desktop/MajorProject/archive/training_setA/
C:/Users/LENOVO/Desktop/MajorProject/archive/training_setB/
```

Discovery is recursive because these folders contain a second training-set directory. Source files are read only. Copying or renaming patients is unnecessary. On another machine, change `input_dirs`; all relative paths in the configuration resolve relative to the configuration file, not the terminal's working directory.

The initial local inspection found **20,336 A files + 20,000 B files = 40,336**, which is **307 fewer** than the requested expected count of 40,643. Filename gaps alone do not establish that this is an invalid dataset release. Expected counts are configurable and discrepancies are reported rather than silently accepted.

Fourteen representative files share **41 columns**, including 40 predictors and `SepsisLabel`. The saved inspection samples contain 19–54 hourly rows; `ICULOS` sometimes starts above 1 and is retained unchanged. Labs are frequently absent, and missingness differs by source. `reports/inspection.json` records the selected files, actual headers, row counts, labels and per-file missingness. Full ingestion validates every file against the observed modal representative schema; it does not silently union inconsistent headers.

## Installation and commands

The completed local run processed all **40,336 patients / 1,552,210 hourly rows**, with **zero failures**, in **32.25 minutes**. It retained 2,932 positive patients and 27,916 positive hourly labels. Patient counts are 28,235 train, 6,051 validation and 6,050 test. All final checks and 23 regression tests passed. Detailed statistics are in `reports/dataset_statistics.json`; actual output paths and buffer maxima are in `processed_data/pipeline_manifest.json`.

With a regular Python 3.11+ installation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/inspect_dataset.py
python run_pipeline.py
python src/dataset_statistics.py
python -m unittest discover -s tests -v
```

The Windows `python` command currently resolves to a Store placeholder on this computer. A working environment has been created in this project using the bundled Python runtime. Use these commands directly without activation:

```powershell
.\.venv\Scripts\python.exe src/inspect_dataset.py
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe src/dataset_statistics.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The inspection, preprocessing and statistics scripts accept `--config path/to/config.json`. `python src/preprocess.py` is an equivalent pipeline entry point. To rebuild completed outputs, use `python run_pipeline.py --overwrite`; the previous output directory is retained as a uniquely named backup. A failed run remains in an `inprogress-*` sibling directory with `RUN_FAILED.json`, and is never published as a completed output. There is no automatic resume: a rerun starts a fresh, reproducible run. Reports describe the latest attempt; retain them with a previous output if you need a full historical audit.

`python src/dataset_statistics.py --regenerate` recomputes raw descriptive statistics from saved chunks. Without that flag it displays the completed statistics report.

## Storage and memory

`storage_format: "auto"` prefers Parquet. On this computer, Windows Application Control blocks the installed PyArrow filesystem extension. Automatic selection therefore uses **gzip-compressed CSV**, with honest `.csv.gz` filenames. No security setting is changed. Use `"parquet"` to require a working Parquet runtime and fail explicitly if unavailable, or `"csv_gzip"` to select the supported fallback directly. Parquet is generated automatically in an environment with a functioning PyArrow installation. CSV does not retain the compact numerical dtypes; use the feature metadata when loading it for training.

The default is a single worker, at most **128 patients / 20,000 rows per raw batch**, and the same bound for each output split's writer. Individual patients are bounded by `max_patient_rows` and `max_file_bytes`; batches never split a patient during feature generation. Long or oversized files are quarantined with an explicit error. There is no repeatedly growing DataFrame and no loading of all patient files. Only small path/ID manifests are kept for all patients.

Exact numerical medians use temporary on-disk float64 arrays, partitioning one mapped feature at a time. The OS can cache mapped pages, but no complete dataset matrix is allocated. Statistics scratch space is released after fitting. Processing may hold one raw batch, one patient's engineered features, and up to three bounded output buffers simultaneously. `pipeline_manifest.json` records actual buffer maxima. Validation reads every final output in bounded batches.

## Processing stages

1. Discover files deterministically and inspect representatives from A and B.
2. Validate and ingest patient files into bounded raw chunks. Save raw missingness, exact descriptive numerical statistics, patient metadata, and source SHA-256 hashes.
3. Split **patients**, stratified jointly by source and whether the patient ever has a positive label. Labels are used for stratification only, never as predictors. The fixed seed is 42; largest-remainder allocation approximates 70/15/15. Tiny strata generate warnings and retain a training representative. Row fractions may differ because lengths of stay vary.
4. Fit raw-observation medians and binary modes **only on training patients**. Save parameters separately from all-patient descriptive statistics.
5. Engineer each patient's causal hourly features, apply the frozen training parameters, and write the patient's entire sequence to exactly one split.
6. Read all final outputs and verify patient coverage, split disjointness, row/label counts, original times, binary targets, finite complete predictors and schema. Runtime prefix/future-mutation checks cover representative patients from each source; unit tests exercise randomized missingness and time gaps.

Malformed/unreadable files, inconsistent schemas, invalid target values, non-numeric tokens, infinite values, and invalid/nonconsecutive ICU times are logged and excluded; processing continues. All copies of a duplicate filename-derived patient ID are quarantined, avoiding arbitrary selection. Every successfully admitted file has the same schema. Systemic errors such as a failed output write stop the run visibly; they are not mislabeled as patient corruption. Rare patient-specific feature failures are also quarantined; final manifests are reconciled with exported patients. Raw statistics retain the validated raw cohort, and any difference is explicitly reported.

## Missing values and temporal features

These are transparent research baselines, not validated physiological assumptions:

- Forward-fill **HR, O2Sat, Temp, SBP, MAP, DBP and Resp** from the same patient's past observation for at most **3 hours**. Do not backward-fill, interpolate using future values, or carry a vital indefinitely.
- Labs, binary unit indicators, demographics and time fields are **not forward-filled**. Remaining missing current measurements use the observed **training median**, or the observed training **mode** for Gender/Unit1/Unit2 (ties select 0). `ICULOS` must be observed and is never imputed.
- Before a patient's first measurement, there is no patient-derived fill. If a numeric fallback is needed, use only the population training value, with the raw missingness indicator still equal to 1. This is an explicit statistical prior, not a recovered observation. An entirely unobserved training variable and its median-dependent derived features are dropped; its missingness indicator is retained. Validation/test values never rescue a training-empty predictor.
- Create `__missing` indicators for all **39 non-ICULOS raw predictors**, from raw values before any filling.
- For **HR, O2Sat, Temp, SBP, MAP and Resp**, create previous-hour value, change from the previous hour, and hours since the last **raw** observation (`-1` means never observed).
- For those same six variables, create mean, minimum, maximum, population standard deviation and OLS slope per hour over **3, 6 and 12 hours**. The window at time `t` is `(t-window, t]`, including the current hour. Rolling statistics use **actual observations**, excluding synthetic forward-filled values. Standard deviation and slope require two observations.
- Missing previous/mean/min/max values use the raw source's training median. Undefined delta/std/slope values use a documented zero-change fallback; indicators and recency show missing measurement history. This fallback can bias sparse-history summaries and should be evaluated during model development.

With the observed complete schema, the default produces **187 predictors**: 40 retained current/time variables, 39 missing indicators, 18 previous/change/recency features and 90 rolling summaries. There are **147 generated features**, plus `patient_id`, `dataset`, and the target (190 columns total), unless training-empty features are dropped. `processed_data/feature_metadata.json` lists every feature, its definition, fallback and exact predictor list.

`patient_id`, `dataset`, and `SepsisLabel` are excluded from the predictor list. Dataset origin remains available for source-specific auditing; including it as a model input would require a separate generalization assessment. No patient length, eventual sepsis status, onset time, split ID, or future measurement is a feature. No scaling, feature selection or dimensionality reduction is performed. If added later, fit it on training data only, and repeat preprocessing within each training fold for cross-validation. Validation/test descriptive statistics are reporting artifacts and must not guide fitted transforms or tuning on the test set.

## Target and class imbalance

The target is the supplied `SepsisLabel`, unchanged at each hour. The official challenge labels are already shifted six hours before the defined onset; **do not shift them again**. Current-hour measurements are therefore allowed under the original challenge protocol. A different forecasting horizon or landmarking scheme requires an explicit, separately validated target design. See the [official dataset documentation](https://physionet.org/content/challenge-2019/1.0.0/).

All valid positive and negative samples are retained. Reports show positive/negative rows and patients overall and by split. There is no oversampling, undersampling or balancing of held-out data. Any training-time balancing or class weighting belongs inside the training workflow. A random patient holdout tests within-dataset generalization, not prospective or independent-hospital performance.

## Outputs

```text
src/
  common.py                 configuration, discovery, strict patient reader
  inspect_dataset.py        representative inspection
  preprocess.py             orchestration, transformation, validation
  feature_engineering.py    patient-local historical features
  split_dataset.py          reproducible stratified patient splits
  dataset_statistics.py     disk-backed statistics and train-only fitting
  storage.py                Parquet / gzip CSV, bounded reads and writes
processed_data/
  raw_processed/chunk_*.parquet  (or chunk_*.csv.gz)
  train.parquet                 (or train.csv.gz)
  validation.parquet            (or validation.csv.gz)
  test.parquet                  (or test.csv.gz)
  train_patient_ids.csv
  validation_patient_ids.csv
  test_patient_ids.csv
  patient_manifest.csv
  config_resolved.json
  schema.json
  feature_metadata.json
  imputation_parameters.json
  pipeline_manifest.json
reports/
  inspection.json / inspection.txt
  processing_log.csv
  output_processing_log.csv
  failed_files.csv
  raw_patient_manifest.csv / patient_manifest.csv
  missingness.csv
  numerical_statistics.csv
  dataset_statistics.json / dataset_statistics.txt
  split_report.json
  validation_report.json
  pipeline.log
```

The source SHA-256 manifest, resolved configuration, package versions and fixed seed support reproducibility. `requirements-lock.txt` records the versions used locally. Filesystem order is sorted before splitting. Run tests with the same package versions for exact numerical repeatability. Gzip writer timestamps are fixed; portable reproducibility is defined by patient assignments and numeric contents rather than archive/library implementation details.

Exact row duplicates are detected within patient using all raw columns including `ICULOS`. Identical measurements at different hours are valid repeated observations and are preserved. Identical rows across different IDs do not establish patient identity; the source filenames do not permit detecting the same real person under multiple IDs. Plausible but erroneous extreme values are reported, not silently clipped using undocumented clinical cutoffs. Nonconsecutive hours are rejected during ingestion rather than reordered or silently reindexed.
