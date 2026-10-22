# Phase 3 — Classical ML Baselines

**Status:** Logistic Regression and XGBoost baselines trained; validation predictions and reloadable pipelines saved. The test set remains sealed for Phase 5.

## Shared data contract

| Split | Labelled rows | Use in Phase 3 |
|---|---:|---|
| Train (Jan–Aug) | 217,727 | Fit preprocessing and estimators |
| Validation (Sep–Oct) | 55,628 | Bounded XGBoost selection and diagnostic metrics |
| Test (Nov–Dec) | 53,991 | IDs assigned only; labels not used or scored |

Both models use the same 37-feature Phase 2 allowlist. Numeric imputation,
missing-value indicators, scaling where applicable, and categorical one-hot
encoding are contained in the saved pipelines and fitted on training rows only.
Unknown categories are ignored safely at inference.

## Validation diagnostics

These are Phase 3 diagnostics, not the official final comparison.

| Model | PR AUC | ROC AUC | Brier score |
|---|---:|---:|---:|
| Logistic Regression | 0.318863 | 0.681518 | 0.133287 |
| XGBoost | 0.294506 | 0.674452 | 0.124671 |

XGBoost was selected from two bounded CPU candidates using validation PR AUC.
No class weighting was added to the Logistic Regression baseline. Threshold
selection, calibration decisions, and final test comparison belong to Phase 5.

## Reproduction

```bash
python -m src.models.classical.train
```

Configuration is stored in `configs/classical_models.json`. Artifacts and
validation predictions are under `artifacts/classical/`; run metadata is also
mirrored to `reports/phase3_classical_baselines.json`.
