# Project status: protocol v2 through Step 9

The active milestone uses the ten **steps** in the AML audit and DL plan. These
are separate from the original Phase 0-7 roadmap. The requested delivery ends
at **Step 9 (evaluation)**; Step 10 presentation work and the original offline
rescheduling phase are outside that boundary.

**Completed delivery:** Steps 8-9 have now been run on the supplied five-model,
one-configuration-per-model archive. Measured numbers are in
[RESULTS.md](reports/supplied_run/RESULTS.md) and
[numerical_results.csv](reports/supplied_run/numerical_results.csv).
The full 40-configuration experiment and legacy/new feature comparison grid
are not claimed complete. Earlier local regeneration was superseded by this
supplied-run evaluation.

## Implementation and experiment status

| Step | Deliverable | Implementation | Experiment evidence |
|---|---|---|---|
| 1 | Protocol v2 | Present: `configs/project_contract_v2.yaml` | Frozen before v2 retrospective evaluation |
| 2 | Shared correctness fixes | Present, including direct alert counts and equal-mass ECE | Unit/contract checks in `tests/` |
| 3 | As-of feature builder | Present | Cutoff, availability and perturbation tests |
| 4 | Four-lead datasets and sequences | Present | Completed: 1,309,384 rows; see `reports/step4_multi_lead_quality.md` |
| 5 | Schedule-only, LR and XGBoost baselines | Present | Supplied one-configuration checkpoints and predictions evaluated |
| 6 | Embedding MLP | Present | Supplied one-configuration checkpoint and predictions evaluated; legacy run missing |
| 7 | Sequence gate and GRU model | Present | Supplied gate: go; weather-only GRU evaluated |
| 8 | Calibration and alert-budget policy | Completed for supplied runs | `artifacts/supplied_run/step8/frozen_decisions.json` |
| 9 | Comparison, intervals and diagnostics | Completed for supplied five-model scope | `reports/supplied_run/evaluation.json` |

Implementation is not the same as a completed full-budget experiment. A reduced
search is explicitly labelled **development only**. Do not copy indicative
handoff numbers into the final report. A model does not have to outperform
XGBoost to complete the experiment; a superiority claim requires the recorded
success gates to pass.

## Run and inspect progress

The implementation has passed **104 automated tests** (85 classical/evaluation
tests and 19 isolated neural tests), including a complete eight-model synthetic
run through Step 9. Additional archive checks validate imported runs. All 20
real model/lead combinations have been independently recomputed from exported
predictions to 1e-12 tolerance; see `reports/supplied_run/results_verification.json`.
Synthetic test scores are not project results.

To reproduce the supplied-run evaluation in fresh output directories:

```bash
python -m src.evaluation.calibrate_step8 --archive-path /path/to/predictions.zip \
  --allow-reduced-budget --output-dir artifacts/supplied_run/step8
python -m src.evaluation.evaluate_step9 --step8-dir artifacts/supplied_run/step8 \
  --output-dir artifacts/supplied_run/step9 --reports-dir reports/supplied_run
python -m src.evaluation.summarize_results --step8-dir artifacts/supplied_run/step8 \
  --step9-dir artifacts/supplied_run/step9 --reports-dir reports/supplied_run
```

The separate full-budget regeneration workflow remains available:

```bash
source .venv/bin/activate
python -m src.checks
python -m src.run_v2
```

The runner executes through Step 9, logs each stage under `artifacts/v2/`, and
writes `artifacts/v2/pipeline_state.json`. It resumes published stages after
checking their settings. Failed or interrupted training stages are rerun; a
partially written directory is never treated as a finished model.

For an explicitly reduced development run, use a separate root:

```bash
python -m src.run_v2 --output-root artifacts/v2-development --trial-limit 1
```

This still uses the full eligible flight population and all six folds. The
default neural epoch limits remain unchanged. It checks integration but does
not satisfy the 40-configuration protocol budget.

## Remaining external decisions

- Additional years and a clean unseen test period remain deferred.
- Same-day departure features and departure sequences remain disabled without
  instructor approval; weather sequences are the documented fallback.
- Forecast vintages and cancellation/diversion targets are deferred because the
  supplied files do not establish those inputs or labels.

## Result locations

- Supplied checkpoints and original predictions: `artifacts/supplied_run/supplied_predictions/`.
- Frozen calibration and alert policy: `artifacts/supplied_run/step8/`.
- Keyed retrospective predictions and complete evaluation: `artifacts/supplied_run/step9/`.
- Shareable numbers, tables and figures: `reports/supplied_run/`.

See [Step 8-9 handoff](docs/Engineering/step-8-9-evaluation-handoff.md) for the
statistical choices and command-level interfaces.
