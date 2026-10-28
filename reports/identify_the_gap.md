# Identify the Gap — AML baselines

This analysis uses the frozen Phase 3 models and the Phase 5 metrics from the same feature-table hash. The existing repository had already exposed test results before this rerun; treat these as retrospective evidence, not a new pristine holdout.

On the Nov–Dec test period, delay prevalence is 25.8%. Logistic Regression PR-AUC is 0.4506 and XGBoost PR-AUC is 0.4846; ROC-AUC is 0.6945 and 0.6946, respectively. These results show limited discrimination, but one year and two models do not establish a performance ceiling.

On the threshold-selection slice, F1 reference precision/recall is 21.8%/70.9% for Logistic Regression and 20.7%/68.2% for XGBoost. On test, the frozen F1 points detect 75.7% and 76.9% of delays, with 20,523 and 21,354 false alerts. The 35% precision floor is exploratory; Logistic Regression: feasible, XGBoost: infeasible under the minimum recall and alert-count guards. See the [evaluation report](phase5_aml_evaluation.md) for each model's feasibility and test outcomes; an infeasible floor is never presented as achieved.

Calibration is chosen using a later validation slice than the fitting slice, then applied consistently to test probabilities when selected. Seasonal prevalence shifts from validation to test, so probability reliability and alert behavior may change. The supplied weather is retrospective observation data, not archived forecast vintages. Historical outcomes use a conservative availability bound rather than actual release timestamps. These data limits are plausible reasons for errors, not proof that a neural model will improve them.

Phase 4 remains the deep learning phase; rescheduling is outside this AML analysis.
