# Full-test XGBoost SHAP explanations

This analysis loads the existing `models/xgboost/xgboost.ubj` model and explains every hourly test record using all 187 saved predictors in their original order. It never retrains or saves over the model, selects a feature cutoff, or changes the decision threshold.

## Run in your IDE

```powershell
cd "C:\Users\LENOVO\Documents\ChatGPT\Major Project"
.\.venv\Scripts\python.exe explain_xgboost.py
```

The existing model dependencies are sufficient. XGBoost already implements exact TreeSHAP through `predict(pred_contribs=True, approx_contribs=False)`, so an additional `shap` package is not required. The explanation uses the same saved early-stopping tree range as the original predictions. See the [official XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict).

Settings are centralized in `shap_config.json`: batch size, CPU workers, display-bin resolution, output location, and the patient/hour for the local explanation. The defaults explain **p006266 at ICULOS 22**, the alert discussed previously. No top-N setting or feature-selection rule exists.

## Files to open

```text
reports/shap_xgboost/
  feature_importance_full.csv            # all 187 ranks, including zero values
  feature_ranking_full.txt               # full readable ranking, also printed in terminal
  feature_importance_full.png / .pdf     # all 187 feature-name labels and bars
  shap_summary_all_features.png / .pdf    # complete SHAP distribution summary
  summary_plot_metadata.json             # full feature order and per-feature row counts
  importance_elbow.png / .pdf            # every rank; no selected elbow/cutoff
  feature_columns.json                   # original matrix-column order
  shap_values.npy                        # all test rows × all 187 contributions
  base_values.npy
  raw_margins.npy
  probabilities.npy
  test_row_index.csv.gz                  # matrix row → patient/hour/source/label
  config_resolved.json
  run_manifest.json                     # integrity checks, hashes, versions, runtime
  p006266_hour_22/
    local_waterfall.png / .pdf            # all 187 signed contributions
    local_contributions.csv              # all 187 feature values and SHAP contributions
    local_explanation.json               # full additive accounting and prediction
    test_row_metadata.json
```

In the IDE Explorer, expand **reports → shap_xgboost**, then open the ranking CSV or an image. The full 187-row plots are deliberately tall; use their PDF companions to zoom without losing readable labels. Successful reruns retain previous results in `shap_xgboost.backup-*`. A failed run retains diagnostics in `shap_xgboost.inprogress-*` without replacing the last successful output.

## What the global ranking means

For feature j:

```text
mean_abs_shap[j] = sum(abs(SHAP[row, j]) for every test row) / number of test rows
```

The CSV columns are `rank`, `feature`, `mean_abs_shap`, and `mean_signed_shap`. Rank uses the **mean absolute** contribution, descending, so positive and negative contributions do not cancel. The signed mean is supplementary and does not determine rank. Ties retain saved feature order. Zero-contribution features remain in the list. The extra SHAP bias/base term is recorded separately and is not treated as a 188th feature.

All **231,884 hourly records from 6,050 test patients** contribute. Neither positive labels nor later hours are removed. This is an hourly-weighted model summary; patients with longer recordings contribute more rows. It is not a patient-balanced or pre-onset-only ranking.

The full ranking bar plot displays every feature name. The elbow plot places all ranks on the horizontal axis and their mean absolute contributions on the vertical axis. It does not identify or recommend an elbow, draw a selection boundary, or drop any feature. Use the rank CSV to map a point back to its feature.

## How to read the distribution summary

The summary orders every feature by the full-test mean absolute SHAP ranking. Horizontal position is signed SHAP contribution: positive raises the model's output and negative lowers it. Within each feature row, wider density indicates more hourly observations; color represents the mean within-feature value percentile in that SHAP bin (low to high).

**All observations are included.** To keep memory and file size bounded, each feature's distribution is binned for display instead of retaining more than 43 million individual scatter dots. This is a binned density SHAP summary, not an unbinned beeswarm or a row sample. Bin resolution affects the image only; the native SHAP values and the exact full-test mean-absolute ranking are unchanged. Zero-SHAP features remain visible.

## Individual explanation function

The reusable API lives in `src/shap_explain.py`:

```python
from src.shap_explain import explain_patient_row

explanation = explain_patient_row(
    model=loaded_xgboost_booster,
    feature_row=patient_feature_row,
    feature_names=saved_feature_names,
    threshold=saved_threshold,
    output_dir="reports/my_patient_explanation",
    patient_id="p006266",
    hour=22,
)
```

`patient_feature_row` must be a named pandas Series, one-row DataFrame, or dictionary containing exactly the **187 processed model inputs**. It is not a raw 41-column PSV row. The function restores the model's saved feature order and rejects missing/extra features, labels, metadata, invalid values, or mismatched time. The main script performs the complete loading and row selection automatically.

The waterfall includes **every feature**, ordered by absolute local contribution. Positive terms are red; negative terms are blue; zero terms remain listed. Labels show the processed input values, which may be imputed values or historical engineered features. They are not necessarily newly observed clinical measurements at that hour. The CSV provides exact signed values without any display cutoff.

## Units and interpretation

These native SHAP contributions are in **raw log-odds**, not probability percentage points:

```text
baseline log-odds + sum(all 187 SHAP values) = prediction log-odds
model probability = 1 / (1 + exp(-prediction log-odds))
```

The baseline uses training-derived node-cover statistics stored in the trained trees. No new background dataset is fitted using test rows. Its sigmoid is not the dataset's sepsis prevalence. These scores come from the existing class-weighted model and have not been probability-calibrated.

SHAP describes how this model allocates its prediction among its inputs. It does not establish a causal clinical effect. Correlated raw variables, rolling summaries, and missingness indicators can share importance; a small individual contribution does not establish that a variable is clinically irrelevant.

The requested full-test ranking is a descriptive explanation of the frozen model. If it informs feature selection or tuning, those decisions should be developed on training/validation data and assessed on a new untouched holdout. This script makes no such selection.

## Verification and memory use

The script uses the validated disk-backed test matrix and computes exact contributions in bounded batches. It writes the full contribution matrix to `.npy` and accumulates global sums in float64. Distribution plotting reads one feature column at a time. All features, all test rows, and zero contributions are retained.

It checks the saved model's feature names and tree range, matches every row against the original saved test predictions, verifies SHAP additivity and probability reconstruction, and cross-checks the individual explanation against the full-test matrix. Hashes of the model and source artifacts must remain unchanged before publication. This confirms the saved model was not overwritten during explanation. `run_manifest.json` records the maximum numerical errors and package versions.

Run checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
