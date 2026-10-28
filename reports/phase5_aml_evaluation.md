# Phase 5 — AML evaluation

Run: `phase3-73f033b69a9f`; feature SHA-256: `fc1e2319664537b87e0947de4013ef53413e0320ecb8ace31a0d7ded8fcd0faf`.

The 35% precision floor is exploratory, with at least 5% validation recall and 100 alerts required. The test period was already viewed in earlier repository work, so this is a retrospective evaluation, not a pristine first holdout.

Validation uses Sep 1–24 for XGBoost tuning, Sep 25–Oct 12 to fit calibration, and Oct 13–31 to assess calibration and choose thresholds. Test outcomes do not affect these decisions.

| Model | Full val PR-AUC (raw) | Full val ROC-AUC (raw) | Assessment Brier raw | Assessment Brier isotonic | Selected probabilities | Validation precision | Validation recall | Test PR-AUC | Test ROC-AUC | Test Brier | Test precision | Test recall |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3235 | 0.6879 | 0.1518 | 0.1436 | isotonic | 0.3660 | 0.0504 | 0.4506 | 0.6945 | 0.1848 | 0.6347 | 0.1225 |
| XGBoost | 0.2914 | 0.6626 | 0.1450 | 0.1448 | raw | infeasible | infeasible | 0.4846 | 0.6946 | 0.1861 | infeasible | infeasible |

## Validation-selected F1 reference

| Model | Validation precision | Validation recall | Test precision | Test recall | Test F1 | Test false positives | Test false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.2178 | 0.7092 | 0.3398 | 0.7573 | 0.4691 | 20,523 | 3,384 |
| XGBoost | 0.2074 | 0.6815 | 0.3344 | 0.7693 | 0.4662 | 21,354 | 3,217 |

The test precision and recall columns use the validation-selected 35% floor threshold where feasible. Infeasible means no threshold met the floor plus the minimum recall and alert-count guards. See the JSON for F1 reference thresholds, confusion matrices, and exact decisions.

Figures: [validation reliability](figures/validation_reliability_diagram.png), [validation PR/ROC](figures/validation_pr_roc_curves.png), [test reliability](figures/test_reliability_diagram.png), [test PR/ROC](figures/test_pr_roc_curves.png).
