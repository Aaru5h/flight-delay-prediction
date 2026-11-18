# Steps 8 and 9: calibration and evaluation

These steps consume the completed Step 5, 6 and 7 model handoffs. They preserve
protocol v2. November-December 2013 remains retrospective because earlier AML
work already inspected its outcomes.

## Step 8: freeze decisions

```bash
python -m src.evaluation.calibrate_step8 \
  --step5-dir artifacts/v2/step5 \
  --step6-dir artifacts/v2/step6 \
  --step7-dir artifacts/v2/step7 \
  --output-dir artifacts/v2/step8
```

The default dataset is `data/processed/step4/`. Both Step 5 and Step 6 must have
been run with the legacy feature/manifest arguments so the old/new comparison
grid contains LR, XGBoost and the embedding MLP at T-60. Step 7 can have a
documented no-go; the report records the omission explicitly.

Before fitting, the command recomputes availability-safe folds, checks every
validation ID and label, requires matching populations and data/protocol
fingerprints, and runs the cutoff/sequence tests in an isolated process.

Calibration uses the fixed method in `configs/step8_evaluation.json` (isotonic
by default). It fits separately per lead time on October out-of-fold
predictions whose labels were available by the final model fit cutoff. The
method is predeclared, not selected using retrospective outcomes. Too few
examples or only one class cause a clear failure.

Validation diagnostic probabilities use only an earlier out-of-fold month and
apply the fold's fit availability cutoff. The earliest month uses raw scores.
These development diagnostics reuse the folds used for hyperparameter
selection and are not nested unbiased validation estimates. Final October
calibrators are never assessed on their own fitting rows as if held out.

The operating rule is a fixed **10% ranking budget**, not a retrospectively
selected threshold: flag exactly `floor(0.10 * cohort_size)` flights, ordered
by calibrated probability descending and flight ID ascending. The cohort is
one lead-time reporting population. Monthly and subgroup tables apply their
own local budget; they do not imply a real-time dispatch policy.

`frozen_decisions.json` records configurations, model/run IDs, source hashes,
inference source-code hashes, calibration populations and tests.
`calibrators.joblib` persists every per-lead calibrator. Frozen artifacts and
their validation files are immutable; changed files are rejected by Step 9.

## Step 9: score and evaluate

```bash
python -m src.evaluation.evaluate_step9 \
  --step8-dir artifacts/v2/step8 \
  --output-dir artifacts/v2/step9 \
  --reports-dir reports/v2 \
  --legacy-features data/processed/phase2_features.csv
```

XGBoost and PyTorch inference run in separate OS processes to avoid their
conflicting OpenMP runtimes on Apple silicon. Each scorer reloads the frozen
model and preprocessing. Retrospective sequences are independently checked
for masks, ordering and event availability before scoring.

Outputs include:

- Common-key validation and retrospective comparisons for every feature set
  and model. Legacy inputs are compared only at T-60 on matched flights.
- Average precision, ROC AUC, Brier score/skill, equal-mass ECE and exact
  precision/recall at the alert budget, per lead and month.
- Brier references from the corresponding training population, never the
  evaluation labels' prevalence. Validation references vary by training fold.
- Paired 1,000-resample bootstrap confidence intervals over whole local
  scheduled service days for AP, ROC AUC, Brier and Brier skill, plus paired
  AP differences against matched XGBoost. The same days are sampled for every
  model at a lead time. All lead times of a flight stay together in training
  folds; bootstrap intervals are computed separately per lead.
- A lead-time curve, equal-mass reliability figures and subgroup tables by
  month, origin, carrier, scheduled hour and weather availability, with counts.
- October validation-only feature-group permutation importance and refit
  ablations for XGBoost. Removed-group models reuse selected hyperparameters
  and get their own chronological early stopping; they are not separately
  retuned. Schedule-only versus full models is an additional feature comparison.
- Explicit success gates. Non-overlapping AP intervals are required by the
  current protocol for a superiority claim; paired difference intervals are
  also reported. A negative result is a valid project result.

Ranking metrics are reported for calibrated scores, with raw average precision
alongside them because isotonic calibration can introduce ties. Calibration
plots and Brier scores use the same probabilities exported for inference.
The comparison with the historic 0.4846 AML result is contextual: the updated
feature set, training windows and availability rules differ.
Legacy models train at T-60 only, whereas the new models jointly train on four
lead times. Cross-feature-set differences therefore include the effect of
multi-horizon training; they are not a controlled estimate of feature changes
alone. Comparisons between models on the new feature set share that training
strategy and population.

## Verification and development mode

```bash
python -m src.checks
```

This runs the full suite in separate classical/evaluation and neural processes.
The integration test trains/reloads all model families, freezes calibration,
checks earlier-month diagnostics, scores the common retrospective population,
builds refit ablations/figures, and rejects a modified checkpoint.

`--allow-reduced-budget` on Step 8 supports explicitly labelled development
runs. It never permits a full-budget claim. `--skip-diagnostics` on Step 9
also marks diagnostics incomplete and prevents a superiority claim.

The one-command workflow is `python -m src.run_v2`. The configured full searches
can take substantially longer than a smoke test. Final reports and slides are
Step 10; the original non-causal rescheduling analysis is a separate phase.
