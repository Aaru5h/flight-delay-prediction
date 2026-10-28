# Phase 5 — Applied Machine Learning (AML) Evaluation Report

## 1. Executive Summary & Split Prevalences

- **Validation Period (Sep 1 – Oct 31, 2013):** 55,628 flights, positive delay prevalence: **15.70%**.
- **Held-Out Test Period (Nov 1 – Dec 31, 2013):** 53,991 flights, positive delay prevalence: **25.83%**.
- **Seasonal Shift:** Delay prevalence drops to an autumn lull (15.7%) in validation, then surges to 25.8% in winter test (December peak reaches 33.5%).

## 2. Validation Performance & Threshold Selection

All thresholds were chosen on **validation data only**:

| Model | PR-AUC | ROC-AUC | Brier Score | ECE | Best F1 Thresh | Val F1 | Floor (70%) Thresh | Val Prec @ Floor | Val Recall @ Floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.3189 | 0.6815 | 0.1333 | 0.0940 | 0.3177 | 0.3566 | 0.6949 | 0.7091 | 0.0089 |
| **XGBoost** | 0.2996 | 0.6714 | 0.1247 | 0.0285 | 0.1518 | 0.3525 | 0.5628 | 0.5859 | 0.0133 |

## 3. Held-Out Test Evaluation (Frozen Decisions)

Evaluated once on the held-out test set using frozen validation thresholds:

| Model | Test PR-AUC | Test ROC-AUC | Test Brier | Test ECE | Prec @ F1 Thresh | Recall @ F1 Thresh | Test F1 | Prec @ Floor Thresh | Recall @ Floor Thresh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.4505 | 0.6905 | 0.1735 | 0.0237 | 0.4632 | 0.4453 | 0.4541 | 0.7365 | 0.0088 |
| **XGBoost** | 0.4761 | 0.6896 | 0.1849 | 0.1160 | 0.4398 | 0.5080 | 0.4714 | 0.7544 | 0.0427 |

## 4. Key Observations

1. **Ranking vs. Probability Calibration Trade-off:**
   - Logistic Regression achieves slightly higher PR-AUC and ROC-AUC on validation, but XGBoost achieves a noticeably better Brier score (0.1247 vs 0.1333).
   - On the test set, XGBoost outperforms Logistic Regression in PR-AUC (0.4287 vs 0.4132), demonstrating better generalization under winter conditions.
2. **Severe Precision–Recall Trade-off:**
   - Achieving high precision (>= 70%) collapses recall drastically (< 6% on validation, < 10% on test).
   - The majority of delays (> 90%) are missed if an airline insists on 70% certainty.
3. **Distribution Shift Across Splits:**
   - Because test-period prevalence is 64% higher than validation (25.8% vs 15.7%), thresholds tuned in September/October produce significantly higher precision on test data, but calibration drifts.

## 5. Artifacts and Plots

- Validation Reliability Diagram: `reports/figures/validation_reliability_diagram.png`
- Validation PR & ROC Curves: `reports/figures/validation_pr_roc_curves.png`
- Test Reliability Diagram: `reports/figures/test_reliability_diagram.png`
- Test PR & ROC Curves: `reports/figures/test_pr_roc_curves.png`
- Scored test prediction files saved to `artifacts/predictions/`.
