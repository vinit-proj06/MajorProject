# Patient-level early detection

Run commands from the project root. File paths below are relative to that folder.

This evaluation answers: **Among sepsis patients with recoverable onset timing, how many had a model warning 6–12 hours before the inferred onset?** It uses the existing validation and test predictions from both trained models. It does not train models, change their thresholds, or alter hourly labels.

## Run and view results

In your IDE's PowerShell terminal:

```powershell
cd "C:\Users\LENOVO\Desktop\MajorProjectCopy"
.\.venv\Scripts\python.exe scripts/evaluate_early_detection.py
```

The command displays the results and saves:

```text
reports/early_detection/
  comparison.txt                    # readable results for both models
  comparison.csv                    # summary metrics
  comparison.json                   # complete metric definitions and results
  test_early_detection.png           # comparison chart
  validation_early_detection.png
  config_resolved.json
  run_manifest.json                 # source hashes and verification checks
  xgboost/
    test_patients.csv               # one auditable record per patient
    test_metrics.json
    validation_patients.csv
    validation_metrics.json
  random_forest/                    # same four reports
```

In the IDE Explorer, expand **reports → early_detection**, then open `comparison.txt` and `test_early_detection.png`. Successful reruns retain previous reports in a sibling `early_detection.backup-*` directory. An unsuccessful run keeps diagnostics in `early_detection.inprogress-*` and does not replace the last successful report.

## Definition

The primary early detection rate is:

```text
100 × number of evaluable sepsis patients with ≥1 warning 6–12 hours before onset
    / number of evaluable sepsis patients
```

Count each patient once, regardless of the number of positive hours. Both endpoints of the window are included. At hourly resolution, a complete 6–12 hour window contains seven observations. This is a research definition of timely warning, not proof that a warning is clinically useful. The 6–12 hour objective is described in the [original challenge manuscript](https://physionet.org/files/challenge-2019/1.0.0/physionet_challenge_2019_ccm_manuscript.pdf).

**The first positive `SepsisLabel` is not the clinical onset.** The [official dataset documentation](https://physionet.org/content/challenge-2019/1.0.0/) states that labels become positive six hours before the defined onset. Therefore, only when an observed zero is followed by the first one, use:

```text
inferred onset hour = first positive SepsisLabel ICULOS + 6
warning lead time   = inferred onset hour − alert ICULOS
```

For example, first positive label at hour 20 implies onset at hour 26. A warning at hour 19 has seven hours of lead time and qualifies; a warning at hour 24 has only two hours and does not satisfy the primary six-hour requirement. Evaluation uses ICULOS, not DataFrame row numbers. Onset remains an inference from hourly labels, not a separately available clinical timestamp.

## Patients included and excluded

- **Known onset transition:** A sepsis patient's record contains an observed 0→1 transition. Include in the primary denominator if at least one hour of the warning window is observed.
- **Positive from the first row:** True onset may have been earlier; adding six to the first row would invent its timing. Mark `onset_status=left_censored` and exclude from timing denominators. Report these patients separately, retaining their original hourly evaluation.
- **Partial warning window:** Include in the primary rate, flag incomplete observation, and also report a stricter sensitivity restricted to the complete window and onset within the recorded stay.
- **Inferred onset beyond the recorded stay:** The label transition can imply a future onset beyond follow-up. Include observable warning opportunities in the primary rate but flag/count these cases separately. Exclude from the strict cohort.
- **Never-positive labels:** Use for false-alarm burden; they are not in the sepsis detection denominator. Non-sepsis means no recorded positive label, not verified lifelong absence of sepsis.

The inspected test cohort has 439 positive-label patients: 375 observed transitions and 64 with unknown onset. Of the 375, 323 have a complete 6–12 hour warning window and 52 have a partial window. No test inferred onset lies beyond follow-up. Validation has 440 positive-label patients, 373 transitions, 67 unknown onsets, and four inferred onsets beyond follow-up. The program verifies counts on every run rather than hard-coding them.

`all_positive_patient_detection_lower_bound` divides known detections by all positive-label patients. This is a descriptive lower bound, not an estimated sensitivity for patients with unknown onset.

## Alert policy and leakage prevention

An alert is emitted when the saved probability meets the model's **existing validation-selected threshold**. By default, one positive hour is sufficient. The policy is continuous hourly output, with no cooldown or permanent suppression: a qualifying positive output in the warning window counts even if an earlier alert occurred. A very early alert alone does not count. `first_alert_hour` and `first_useful_alert_hour` are saved separately so earlier alarm activity remains visible.

`consecutive_positive_hours` can require consecutive positive scores. A warning then starts at the confirming hour, never at the beginning of the streak. Every subsequent qualifying hour emits an alert. The trigger uses only current/past model outputs; future labels are used solely afterward to evaluate timing. No onset-derived values become model features.

`early_detection_config.json` centralizes the window, confirmation requirement, model names, splits, and batch sizes. `--config <path>` selects another evaluation configuration. Defaults are fixed for this evaluation, not selected to maximize test results. Changing policy after inspecting test outcomes requires a new holdout for an unbiased comparison. The existing thresholds were optimized for hourly validation F1, not early detection sensitivity.

## Other reported measures

- Strict sensitivity for warnings **1–12**, **3–12**, and **6–12** hours before onset; every horizon requires a fully observed window and onset within the record, and reports its own numerator/denominator.
- Mean and median lead time of the **first qualifying warning**, only among primary detections. These conditional lead times do not describe missed patients.
- Non-sepsis patient false-alarm rate: patients with at least one alert divided by all never-positive-label patients.
- Alert-positive hours and alert episodes per 100 observed non-sepsis patient-hours. An episode begins on a false-to-true transition in emitted alerts; these are hourly simulated outputs, not observed clinical notifications.

Always inspect false-alarm burden alongside early detection. This patient-level rate differs from hourly recall and from the official challenge utility score. It does not measure actual treatment decisions, benefits, or clinical safety. Validation remains development data; test results are retrospective descriptive research estimates without confidence intervals or external validation.

## Integrity and memory

Predictions are read in chunks, with at most one unfinished patient carried between chunks. Only patient summaries are retained for reporting; no full predictor matrix is loaded. The existing model cache is validated/reused (or rebuilt from processed files if missing). Every prediction's patient ID, source dataset, time and target must match its cached source row. Threshold files, recorded predictions, cohort sizes, split membership, finite values, contiguous times, and monotone labels are checked. Corrupted evaluation input fails clearly instead of silently excluding patients from a denominator. Source SHA256 hashes are recorded and checked again before publishing reports.

Run all checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
