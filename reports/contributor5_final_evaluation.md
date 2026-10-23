# Contributor 5 — Final Evaluation, Integration and Gap Analysis

**Evaluation status:** final held-out comparison completed once under the frozen validation plan.

## Shared chronological split

| Split | Dates | Rows | Positive | Negative | Prevalence |
|---|---|---:|---:|---:|---:|
| train | 2013-01-01 to 2013-08-31 | 217,727 | 57,423 | 160,304 | 0.2637 |
| validation | 2013-09-01 to 2013-10-31 | 55,628 | 8,731 | 46,897 | 0.1570 |
| test | 2013-11-01 to 2013-12-31 | 53,991 | 13,946 | 40,045 | 0.2583 |

The validation prevalence is materially lower than train and test; metrics are therefore interpreted as a chronological distribution shift, not balanced by changing dates.

## Frozen validation decisions

- Primary model: **logistic_regression**, selected by validation PR AUC.
- Operating threshold: maximum F1 on validation for each model; no stakeholder-approved precision floor exists.
- Calibration: no post-hoc calibrator fitted because no independent calibration fold was reserved. Raw probability calibration is reported with Brier score, ECE, and reliability plots.

## Validation comparison

| Model | PR AUC | ROC AUC | Brier | ECE | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.318863 | 0.681518 | 0.133287 | 0.094023 | 0.317684 | 0.278384 | 0.496507 | 0.356746 |
| xgboost | 0.294506 | 0.674452 | 0.124671 | 0.021922 | 0.170256 | 0.281861 | 0.482304 | 0.355794 |

## Final held-out test comparison

| Model | PR AUC | ROC AUC | Brier | ECE | Frozen threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.450485 | 0.690519 | 0.173514 | 0.023669 | 0.317684 | 0.463196 | 0.445361 | 0.454103 |
| xgboost | 0.485708 | 0.695974 | 0.182510 | 0.112193 | 0.170256 | 0.458053 | 0.487810 | 0.472463 |

XGBoost ranks better on the held-out test period even though Logistic Regression was selected on validation. The primary selection is not changed after observing test results; this reversal is evidence of temporal instability.

## Identify the Gap

1. **Performance gap:** ranking is useful but moderate; neither model is strong enough to support automated operational decisions without further validation.
2. **Seasonal gap:** validation and test prevalence differ sharply, exposing temporal instability that a single-year dataset cannot resolve.
3. **Feature gap:** weather is retrospective observation data rather than archived forecasts, and aircraft/rotation, gate, crew, network, and live congestion information are absent.
4. **Target gap:** missing arrival delays are unlabeled, and the source has no reliable cancellation/diversion target.
5. **Calibration gap:** probability quality varies by period. A future study should reserve a separate calibration window or use nested time-series validation before fitting a calibrator.
6. **Model gap:** the comparison covers the two required tabular baselines. More complex temporal or neural models require a separate contributor-owned experiment and cannot be claimed here.
7. **Generalization gap:** results cover only 2013 NYC-origin flights and do not establish performance for later years, other airports, or deployment conditions.

## Integration and handoff

The split, prediction schemas, model artifacts, metrics, figures, and report are reproducible by code. Test labels were not used for model or threshold selection. Subgroup results include denominators in `reports/evaluation/subgroup_analysis.csv`; they are diagnostic and not causal or fairness claims.
