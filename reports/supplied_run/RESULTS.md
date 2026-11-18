# Measured results: supplied five-model experiment

Steps 8 and 9 are complete for the supplied reduced-budget runs: one configuration per model. The 40-configuration experiment and legacy-feature model grid remain outside this completed result set.

## One-hour predictions

53,991 November-December 2013 flights; 13,946 delayed (25.83%). Each model flags exactly 5,399 flights under the fixed 10% ranking budget.

| Model | Raw PR-AUC | Calibrated PR-AUC | Calibrated ROC-AUC | Precision @10% | Recall @10% | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5061 | 0.4941 | 0.7203 | 63.57% | 24.61% | 0.1670 |
| XGBoost | 0.4915 | 0.4686 | 0.7099 | 61.59% | 23.84% | 0.1728 |
| Embedding MLP | 0.4314 | 0.4197 | 0.6896 | 53.62% | 20.76% | 0.1758 |
| Sequence GRU | 0.4358 | 0.4197 | 0.6877 | 52.27% | 20.24% | 0.1766 |
| Schedule only | 0.3459 | 0.3394 | 0.6163 | 38.03% | 14.72% | 0.1940 |

## Calibrated PR-AUC by prediction lead time

| Model | 24 hours | 6 hours | 2 hours | 1 hour |
|---|---:|---:|---:|---:|
| Schedule only | 0.3394 | 0.3394 | 0.3394 | 0.3394 |
| Embedding MLP | 0.3240 | 0.4144 | 0.4165 | 0.4197 |
| Logistic Regression | 0.3683 | 0.4601 | 0.4869 | 0.4941 |
| Sequence GRU | 0.3235 | 0.4162 | 0.4218 | 0.4197 |
| XGBoost | 0.3817 | 0.4645 | 0.4633 | 0.4686 |

## Findings

- The highest observed one-hour result is Logistic Regression: raw PR-AUC 0.5061, calibrated PR-AUC 0.4941, with a 95% day-bootstrap interval [0.4103, 0.5651]. This is a retrospective observation, not a new test-based model-selection decision.
- At its one-hour alert budget, it catches 3,432 delayed flights and raises 1,967 false alerts; 10,514 delayed flights are not flagged.
- Neither neural model beats matched XGBoost at any of the four lead times in this run. At 24 hours both neural models also trail the schedule-only model on PR-AUC.
- Raw and calibrated PR-AUC are both retained: isotonic calibration introduces ties and reduces AP here. Calibration was fixed using October validation labels only, with availability checked before the fit boundary.
- This is previously viewed November-December data. Predictions share flight IDs, labels and folds. The original Linux neural feature tables are absent; their different numeric CSV hashes remain an input-provenance qualification.

## Files

`numerical_results.csv` and `.json`: all 20 model/lead combinations, including raw/calibrated probability metrics and alert confusion counts. `evaluation.json`: complete monthly metrics, paired 1,000-resample confidence intervals, diagnostics and provenance. `subgroups.csv`: month, airport, carrier, hour and weather slices with denominators. `diagnostics.json`: validation-only permutation importance and refit ablations.
