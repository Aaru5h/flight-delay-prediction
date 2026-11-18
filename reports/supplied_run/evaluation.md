# Step 9 model evaluation

**DEVELOPMENT ONLY: reduced search budget.**

Nov-Dec 2013 was previously viewed; retrospective evidence only

Ranking below uses calibrated probabilities; raw average precision is also retained in JSON to expose isotonic ties. Brier skill uses the corresponding training prevalence, not the evaluation prevalence. Alerts are the predeclared top 10% of each lead-time cohort, with deterministic ties.

| Feature set / model | Lead (min) | Flights | AP | Raw AP | ROC AUC | Brier skill | Precision @10% | Recall @10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| schedule_only/schedule_only | 60 | 53,991 | 0.3394 | 0.3459 | 0.6163 | -0.0111 | 0.3803 | 0.1472 |
| schedule_only/schedule_only | 120 | 53,991 | 0.3394 | 0.3459 | 0.6163 | -0.0111 | 0.3803 | 0.1472 |
| schedule_only/schedule_only | 360 | 53,991 | 0.3394 | 0.3459 | 0.6163 | -0.0111 | 0.3803 | 0.1472 |
| schedule_only/schedule_only | 1440 | 53,991 | 0.3394 | 0.3459 | 0.6163 | -0.0111 | 0.3803 | 0.1472 |
| multi_lead/embedding_mlp | 60 | 53,991 | 0.4197 | 0.4314 | 0.6896 | 0.0836 | 0.5362 | 0.2076 |
| multi_lead/embedding_mlp | 120 | 53,991 | 0.4165 | 0.4291 | 0.6863 | 0.0834 | 0.5225 | 0.2023 |
| multi_lead/embedding_mlp | 360 | 53,991 | 0.4144 | 0.4227 | 0.6766 | 0.0746 | 0.5173 | 0.2003 |
| multi_lead/embedding_mlp | 1440 | 53,991 | 0.3240 | 0.3284 | 0.6071 | 0.0061 | 0.3343 | 0.1294 |
| multi_lead/logistic_regression | 60 | 53,991 | 0.4941 | 0.5061 | 0.7203 | 0.1294 | 0.6357 | 0.2461 |
| multi_lead/logistic_regression | 120 | 53,991 | 0.4869 | 0.5017 | 0.7173 | 0.1196 | 0.6172 | 0.2389 |
| multi_lead/logistic_regression | 360 | 53,991 | 0.4601 | 0.4773 | 0.7024 | 0.0828 | 0.5897 | 0.2283 |
| multi_lead/logistic_regression | 1440 | 53,991 | 0.3683 | 0.3738 | 0.6537 | -0.0018 | 0.4266 | 0.1651 |
| multi_lead/sequence_model | 60 | 53,991 | 0.4197 | 0.4358 | 0.6877 | 0.0794 | 0.5227 | 0.2024 |
| multi_lead/sequence_model | 120 | 53,991 | 0.4218 | 0.4367 | 0.6866 | 0.0821 | 0.4729 | 0.1831 |
| multi_lead/sequence_model | 360 | 53,991 | 0.4162 | 0.4271 | 0.6761 | 0.0754 | 0.5177 | 0.2004 |
| multi_lead/sequence_model | 1440 | 53,991 | 0.3235 | 0.3293 | 0.6029 | -0.0407 | 0.3495 | 0.1353 |
| multi_lead/xgboost | 60 | 53,991 | 0.4686 | 0.4915 | 0.7099 | 0.0994 | 0.6159 | 0.2384 |
| multi_lead/xgboost | 120 | 53,991 | 0.4633 | 0.4907 | 0.7074 | 0.1014 | 0.6123 | 0.2371 |
| multi_lead/xgboost | 360 | 53,991 | 0.4645 | 0.4789 | 0.6969 | 0.1012 | 0.5999 | 0.2323 |
| multi_lead/xgboost | 1440 | 53,991 | 0.3817 | 0.3902 | 0.6543 | 0.0501 | 0.4721 | 0.1828 |

## Uncertainty and success gates

Paired bootstrap resamples whole local service days. JSON includes 95% intervals for AP, ROC AUC, Brier score and Brier skill, plus paired AP differences against matched XGBoost.

- multi_lead/embedding_mlp: measured performance gates not all passed; superiority claim not supported.
- multi_lead/sequence_model: measured performance gates not all passed; superiority claim not supported.

## Diagnostics and interpretation

`subgroups.csv` reports month, origin, carrier, scheduled-hour and weather-availability slices, including denominators. Their 10% budgets are local to each subgroup. `diagnostics.json` contains validation-only feature-group permutation importance and true refit ablations. Ablations reuse selected hyperparameters and are diagnostic, not separately tuned comparisons.

Validation calibration is prequential: each month uses available labels from an earlier OOF month, with identity fallback where history is insufficient. The final calibrators use only eligible October OOF labels. Model selection and OOF diagnostics reuse development folds; they are not nested unbiased estimates.

Observed weather is a persistence proxy, not archived forecasts. Reconstructed event availability uses an assumed buffer. The data covers one year of NYC departures; same-day departure inputs are omitted unless approved. No real cancellation/diversion target or causal rescheduling claim is made. Step 10 report/slides are outside this deliverable.

## Imported-run scope

These numbers use the supplied five-model archive (one configuration per model). No 40-configuration completion or legacy-feature model comparison is claimed. Neural numeric-table hashes differ from the Mac tables; the supplier attributes this to cross-platform rounding. The original Linux tables are absent, so that explanation has not been independently verified. Raw sources, feature definitions, fold assignments, prediction IDs and target labels agree. Original checkpoint inference is not required to evaluate the supplied saved probabilities.
