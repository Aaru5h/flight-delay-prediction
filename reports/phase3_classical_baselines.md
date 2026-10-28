# Phase 3 — Classical ML baselines

Run `phase3-73f033b69a9f` used 217,727 training, 55,628 validation, and 53,991 sealed test rows. Both models use the same feature allowlist and training-only preprocessing. XGBoost candidates were selected on Sep 1–24 validation PR-AUC; later dates are reserved for Phase 5.

| Model | Full validation PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| Logistic Regression | 0.3235 | 0.6879 | 0.1379 |
| XGBoost | 0.2914 | 0.6626 | 0.1260 |

These are validation diagnostics. See Phase 5 for calibrated selection and retrospective test evaluation.
