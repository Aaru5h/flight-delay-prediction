# Implementation Phase 3 Plan — Classical ML Baselines

**Lead:** Contributor 3  
**Goal:** Train comparable Logistic Regression and XGBoost baselines using the shared feature and split contracts, and export auditable probability predictions.

## 1. Entry conditions

- Phase 0's shared target, split, run and prediction contracts are frozen.
- Phase 1's labelled table is reproducible.
- Phase 2's feature matrix, manifest and leakage tests pass.
- The team can recreate the agreed software environment.

## 2. Implementation steps

### Step 1 — Set up the model module and configuration

Create a classical-model module with separate, clearly named functions or classes for pipeline creation, training, prediction, persistence and evaluation handoff. Read model settings from a configuration file or explicit arguments. Record seed, package versions, feature-manifest version and split-manifest version. Do not scatter constants across notebooks.

### Step 2 — Implement the Logistic Regression pipeline

- Use a scikit-learn `Pipeline`/`ColumnTransformer` or equivalent so preprocessing is learned from the training partition only.
- Impute numeric missingness using an explicit, training-fitted strategy and include missingness indicators where justified.
- Encode categoricals with an encoder that handles categories not seen during training.
- Scale numerical inputs where appropriate for optimization.
- Add class weighting only if chosen based on a documented validation experiment. Keep the default model as a clearly described baseline.
- Ensure identifiers, target columns and audit-only fields never enter the estimator.

The pipeline must be reloadable and must apply exactly the same transformations during validation, test and future what-if scoring.

### Step 3 — Implement the XGBoost pipeline

- Select the preprocessing strategy deliberately. Use supported categorical handling or a stable one-hot/encoded representation; document the choice and package version.
- Keep the feature and row contract identical to Logistic Regression. Model-specific transforms are acceptable; different target populations or holdout definitions are not.
- Configure a practical baseline first, then a small bounded hyperparameter search on training/validation only.
- Use validation data for early stopping/tuning only when implemented consistently and without consulting the test labels.
- Set random seed and thread/device options explicitly so a team member without a supported GPU can still run the required comparison.

### Step 4 — Standardize training and prediction output

For each model:

1. Load the exact approved feature version.
2. Apply the shared split manifest.
3. Fit model and preprocessing on training rows only.
4. Select settings using validation performance only.
5. Export validation predictions for comparison and model selection.
6. After the team freezes model choices and thresholds, generate test predictions once under the approved evaluation workflow.
7. Export one record per scored row with `flight_row_id`, true label, probability, split, model name and run ID.

If project workflow requires test predictions to be saved earlier for technical reasons, their contents must remain sealed from model selection until the final evaluation stage.

### Step 5 — Save artifacts and metadata

Persist:

- Trained pipeline/model and preprocessing state.
- Effective hyperparameters and training config.
- Seed, package versions, code commit and feature/split manifest versions.
- Validation metrics used for tuning and the tuning procedure.
- Predictions in the common output schema.

Never save only a model file without the preprocessing needed to reproduce predictions.

## 3. Test plan

- Both models can fit on a small synthetic/sample subset for smoke testing.
- Pipelines work with missing numeric values and unseen categorical values.
- Repeated runs with the same environment/configuration/seed are reproducible within documented numerical tolerance.
- Every output probability is finite and within `[0, 1]`.
- Prediction IDs are unique and match the expected split population exactly.
- No denied field or target leaks into the estimator input list.
- Model artifact can be reloaded in a fresh Python process and generates the same predictions within tolerance.
- Test labels are not consumed by tuning code.

## 4. Recommended bounded experiment sequence

1. Train a minimal Logistic Regression baseline.
2. Verify pipeline shape, prediction schema and validation metrics.
3. Train a minimal XGBoost baseline on the same data contract.
4. Add only a small documented set of hyperparameter experiments.
5. Compare validation PR AUC, ROC AUC, calibration signal and precision/recall threshold behavior without viewing test results.
6. Freeze each run's config and artifact reference for Phase 5.

This sequence prioritizes working, reproducible baselines over a large search that is difficult to audit.

## 5. Deliverables

- Logistic Regression training/prediction pipeline.
- XGBoost training/prediction pipeline.
- Configs and reproducibility metadata.
- Model artifacts including preprocessing.
- Validation prediction files using the shared schema.
- Test suite for training, inference, category handling, probability checks and ID alignment.
- Short baseline README with exact run commands.

## 6. Acceptance criteria

- [ ] Both models use Phase 2 features and the same split definition.
- [ ] Preprocessing is fit on training data only.
- [ ] Tuning and model selection do not use test labels.
- [ ] Predictions conform to the common schema with complete, aligned IDs.
- [ ] Model artifacts reload and score the same rows.
- [ ] Hyperparameters and run metadata are documented.
- [ ] No metric or model improvement is claimed without a saved run artifact.

## 7. Risks and mitigations

- **Preprocessing fitted globally:** wrap transformations in the training pipeline and test split boundaries.
- **Different populations between models:** compare IDs and class counts against the split manifest.
- **Large/unbounded tuning:** use a limited search and stop when the project time budget is threatened.
- **XGBoost installation/device differences:** pin the working package version and retain a CPU-compatible path.

## 8. Handoff to Phase 5

Supply frozen candidate artifacts, validation predictions, configuration, run metadata and exact reproducibility commands. Phase 5 owns the official metrics and final held-out comparison.
