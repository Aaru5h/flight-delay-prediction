# Implementation Phase 5 Plan — Evaluation, Calibration and Comparison

**Lead:** Contributor 5  
**Goal:** Produce one fair, repeatable comparison of the required models using the frozen chronological split, shared eligible population and a validation-only model/threshold/calibration process.

## 1. Entry conditions

- Phase 0's split, target, prediction schema and metric policy are documented.
- Phase 2's feature version is frozen for the primary comparison.
- Phase 3 and Phase 4 provide validation predictions and model artifacts with run metadata.
- Any threshold precision floor is explicitly agreed. If there is no agreed floor, report precision-recall behavior without pretending an arbitrary floor is an operational requirement.

## 2. Evaluation pipeline

### Step 1 — Validate split and prediction artifacts

For each prediction file:

- Validate required columns and datatypes.
- Check that `flight_row_id` is unique within the relevant split and that no required eligible rows are missing or duplicated.
- Join against the split manifest by stable ID, not row order.
- Verify target labels agree with the labelled dataset.
- Verify probabilities are finite and within `[0, 1]`.
- Record model, run, data, feature and split versions.
- Detect population differences across models and explain them before comparison. Do not compare metrics as though populations were identical if they are not.

### Step 2 — Report split-level dataset profile

For train, validation and test, report flight count, positive/negative count, positive prevalence and the scheduled-date window. Show missing-label exclusions separately. Discuss changes in prevalence across calendar periods as potential distribution shift rather than silently balancing away the difference.

### Step 3 — Compute ranking and threshold metrics

Required metrics:

- **PR AUC / average precision** with positive-class definition stated.
- **ROC AUC**.
- Precision, recall, F1 and confusion-matrix counts at the validation-selected operating threshold.
- **Recall at a stated precision floor**, if a floor has been approved. Select the threshold using validation predictions only.

State the exact threshold-selection rule, including what happens if no threshold reaches the requested precision. Do not calculate a threshold on the test labels. Make it clear that results can vary with class prevalence and the operational precision requirement.

### Step 4 — Evaluate calibration

- Compute Brier score for probabilistic predictions.
- Plot a reliability/calibration curve with bins and counts described.
- Optionally report expected calibration error (ECE), documenting the binning convention; do not substitute ECE for the Brier score.
- Fit a calibration method only using permitted training/validation data. If using a held-out validation calibration set, ensure threshold selection and calibration fitting are separated or their interaction is documented to avoid reusing the same validation evidence without acknowledging it.
- Freeze the chosen calibrator before final test evaluation.

### Step 5 — Conduct validation-only model selection

Use validation results to choose the primary model according to the predeclared priority (for example, PR AUC subject to an acceptable precision/recall and calibration profile). Document trade-offs instead of selecting the highest single number without context. Record the decision before producing or inspecting final test comparisons.

### Step 6 — Run the final test once

When model configuration, threshold, calibration and comparison rules are frozen:

- Generate/load held-out test predictions using the frozen artifacts.
- Compute the agreed metrics without retuning.
- Present test results for all comparable candidates, using the same test population whenever possible.
- If any candidate cannot be evaluated on the same rows, label the difference and provide a common-subset comparison where feasible.
- Do not return to feature engineering or model tuning because test scores are disappointing. Any later experiment must be identified as exploratory and requires a new holdout strategy to make a new final claim.

### Step 7 — Diagnose errors and subgroup performance

Review errors by a small, predeclared set of meaningful slices, such as carrier, origin, scheduled hour, month, route frequency and weather availability. Include group sample sizes and class prevalence. Treat small samples carefully and avoid unsupported claims about causality or protected-group fairness. Investigate whether errors concentrate where reference joins/history are missing.

## 3. Required report tables and plots

- Split population and target prevalence table.
- Model comparison table for validation and final test.
- Precision-recall curves and ROC curves.
- Reliability plot and Brier score.
- Confusion matrix at the selected validation threshold.
- A short error/subgroup analysis with denominators.
- Model-selection rationale and limitations.

Every chart/table must be generated from saved prediction artifacts, not manually typed numbers. Keep a script/notebook that rebuilds the report.

## 4. Tests and quality checks

- Metrics recompute from exported predictions.
- The evaluation code joins by `flight_row_id` and catches duplicate/missing rows.
- Threshold selection function accepts validation labels only or enforces split-specific inputs.
- Test labels do not appear in tuning/calibration call paths.
- Model table includes model/run IDs and data/feature/split versions.
- Plot generation works from a clean environment and fails clearly on malformed predictions.

## 5. Deliverables

- Evaluation module and tests.
- Reproducible metrics tables and plots.
- Validation-selected model/threshold/calibration decisions.
- Final held-out comparison, once the policy is frozen.
- Error analysis, target-rate distribution table and concise interpretation.
- An evaluation record that states what is exploratory versus final.

## 6. Acceptance criteria

- [ ] All scored files pass schema, probability, label and ID checks.
- [ ] Required models use the same target and chronological evaluation design.
- [ ] PR AUC, ROC AUC and calibration are reported.
- [ ] Operational threshold, if used, was selected on validation data only.
- [ ] Test data was not used for tuning/model selection.
- [ ] Final comparisons contain actual metrics generated by code and no placeholders.
- [ ] Any population mismatch or distribution shift is disclosed.

## 7. Risks and mitigations

- **ID misalignment:** merge only by stable identifier and assert exact counts.
- **Threshold overfitting:** choose it on validation once and write the rule down.
- **Misleading aggregate metric:** include prevalence and group denominators.
- **Repeated test peeking:** freeze the analysis plan before final test evaluation.
