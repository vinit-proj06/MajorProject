# ICU replay dashboard

Run commands from the project root. File paths below are relative to that folder.

This college demonstration replays patients from the existing **test split**, one recorded ICU hour at a time. It runs the already-saved XGBoost model and displays its score, threshold crossings, measurement trends and explanations. It does not retrain models or modify the dataset.

The first version uses structured ICU vitals, laboratory values and patient context. A genuinely distinct second modality, such as aligned clinical notes or waveforms, is a future extension; the current demonstration does not claim to have one.

## Start and open

Open this project folder in your IDE. In its PowerShell terminal, run:

```powershell
.\.venv\Scripts\python.exe -B run_dashboard.py
```

Keep that terminal running, then open **http://127.0.0.1:8765** in your browser. Stop it with **Ctrl+C** in the terminal. If that port is already occupied, use:

```powershell
.\.venv\Scripts\python.exe -B run_dashboard.py --port 8766
```

Then open http://127.0.0.1:8766. With another working Python environment, `python run_dashboard.py` is equivalent. Dependencies are the same as model evaluation (`requirements.txt`); no extra web framework, JavaScript installation, network service or SHAP package is needed.

## Demonstrate the system

1. Select a test patient. The default is **p006266**, the previously discussed example. Search lists IDs and source datasets without revealing outcomes.
2. Select a saved model. Baseline preserves the original model; A, B and C reproduce the existing timing-removal experiments. Each uses its own original validation-selected threshold. This selector is an exploratory comparison, not a new test-set model selection procedure.
3. Press **Play** to advance through recorded hours, or use the hour controls. Replay speed compresses time for the presentation; it does not change the model's hourly sampling or create new measurements.
4. Read the current raw measurements, score curve, threshold and alert history. Missing raw measurements remain visibly missing. The measurement details distinguish observed values, historical forward fills and training-derived imputation values used by the model.
5. Inspect the local SHAP explanation. Positive contributions increase the model's log-odds score; negative contributions decrease it. Expand the complete feature list or download the contribution table. These are explanations of model behavior, not causal explanations of illness.
6. Use **Reveal retrospective outcome** only when ready to compare the replay with the full recorded outcome. That separate review includes labels and inferred onset timing. The replay itself does not receive future labels or measurements in its responses.

For the baseline example, p006266 begins at ICU hour 4. At hour **22**, the saved model score is approximately **0.87293**, above its threshold **0.814708**. The first recorded positive label is at hour **25**. Under the existing challenge-label interpretation, inferred onset is hour **31**, so this alert is **9 hours before inferred onset**. This is a curated demonstration, not a new estimate of general model performance.

## What is computed

The backend reads original test membership and the recorded patient manifest. It validates the selected raw PSV file against the preprocessing manifest's SHA-256 fingerprint, and processes at most one patient's history per request with a small bounded patient cache. It does not read the entire processed CSV into memory.

At hour `t`, it selects only raw rows with `ICULOS <= t`, calls the original causal feature engineering, and applies the saved **training-only** imputation parameters. It selects the saved model's exact feature names and order, retains its saved best iteration, and applies its saved validation threshold. Past scores in the replay chart are generated from these causal features, without outcome labels as predictors. No model fitting, calibration, threshold optimization or feature selection happens in the dashboard.

The backend computes exact native XGBoost TreeSHAP contributions for the selected row. Their sum plus the base value equals the raw model margin within floating-point tolerance; applying the logistic function gives the displayed uncalibrated score. The complete contribution table includes every feature accepted by the selected model (187 baseline; 186 A; 185 B; 184 C).

The backend may validate and cache a complete source patient record, and patient metadata includes the available replay range. Only the selected prefix is passed to feature generation and inference; future measurements and outcomes are not returned by the replay endpoint. The separately requested review endpoint intentionally uses the complete record for retrospective evaluation.

## Outcome timing and alerts

The replay marks an alert at every observed hour whose score is **greater than or equal to** the frozen model threshold. Consecutive alert hours are grouped into episodes for the displayed history; they do not introduce a new alerting policy or change earlier evaluation results.

The retrospective review calls the existing early-detection evaluator. For an observed transition from label 0 to label 1 at hour `F`, inferred onset is `F + 6`. The existing primary useful-warning window is `[onset - 12, onset - 6]`, inclusive. Any qualifying alert in that window counts, even if an episode began earlier. First-ever alert time is a different quantity and is reported separately. Patients whose records begin with a positive label have unknown transition timing and are excluded from onset-based detection evaluation. Inferred onset is derived from the challenge labels, not an independently observed clinical timestamp.

## Files and local interface

- `run_dashboard.py`: read-only local HTTP server and launch command.
- `src/dashboard_data.py`: test-patient access, causal inference, provenance and explanations.
- `dashboard/index.html`, `dashboard/styles.css`, `dashboard/app.js`: browser interface and replay controls.
- `tests/test_dashboard_data.py`, `tests/test_dashboard_server.py`: causal inference and HTTP boundary tests.

Read-only API routes are `/api/health`, `/api/models`, `/api/patients`, `/api/patient`, `/api/snapshot`, `/api/shap.csv` (complete current-row contribution download), and the explicit `/api/review`. The server exposes only these routes and the three dashboard assets; it does not serve arbitrary project files. It binds to **127.0.0.1 only**, without third-party scripts or external data transmission. It is a local research demo, not a hospital server.

Saved inputs must remain accessible: `processed_data` metadata, `model_cache` metadata and test arrays, the original test patient PSV files referenced by the manifest, baseline saved model artifacts, and available timing-ablation models. Test arrays are opened as read-only memory maps; each recomputed prefix is compared with its original processed features before scoring. If original data or saved artifacts have moved, restore their intended paths or deliberately reconfigure the underlying project; the dashboard will not silently substitute another patient record or retrain a missing model.

## Interpretation and next extension

Scores are **uncalibrated model outputs**, not established probabilities of developing sepsis. Crossing a threshold means only that the saved research model fired. The interface provides no treatment recommendations and a below-threshold result does not establish safety.

The existing test set has already informed feature investigation, so further comparisons are exploratory. The replay demonstrates software behavior; it does not demonstrate clinical readiness, prospective performance, actual hospital integration or a useful clinical alert burden. An independent untouched cohort remains necessary for future performance validation.

The next practical extension can be a validated incoming-observation adapter behind the same patient/hour interface. A later multimodal version requires a distinct, correctly aligned modality and a separately evaluated fusion model. Real deployment would additionally require clinical collaboration, access controls, integration and prospective validation; none is simulated by claiming that recorded test data are live hospital observations.

## Verification

Run the regression checks from the project folder:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The completed implementation passed **113 tests**, including eight dashboard data tests and five local-server tests. Actual saved-model checks covered four patients (a known onset, a record beginning positive, and never-positive examples from both datasets) across all four models. Recomputed replay predictions matched the original saved test predictions within `1e-15`; earlier predictions were unchanged when later hours were included. Model-file SHA-256 fingerprints were unchanged. The results are saved in `reports/dashboard_validation.json` and the test log in `reports/dashboard_tests.log`. Browser checks covered replay, model/patient switching, unknown-onset review, the nine-hour example and a downloaded CSV containing all 187 baseline contributions. These checks validate software consistency, not clinical performance.
