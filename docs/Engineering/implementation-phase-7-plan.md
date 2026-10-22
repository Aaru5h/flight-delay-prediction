# Implementation Phase 7 Plan — Integration, Reproducibility and Final Handoff

**Lead:** Contributor 1 coordinates; every contributor owns review of their component  
**Goal:** Make the complete project runnable by a teammate from a clean checkout, verify that code/docs/artifacts agree, and deliver a defensible final report.

## 1. Entry conditions

- Phases 0–6 have passed their acceptance checks or have explicit documented limitations/no-go decisions.
- The selected model, saved predictions, calibration/threshold decisions and rescheduling artifacts are available.
- The team knows how raw CSVs are supplied and where generated artifacts live.

## 2. Integration steps

### Step 1 — Audit the end-to-end dependency chain

Trace the full workflow in order:

1. Validate source files and build the labelled dataset.
2. Build leakage-safe features and the feature manifest.
3. Create or load the frozen split manifest.
4. Train classical and neural model candidates.
5. Generate common-format validation predictions.
6. Run evaluation and freeze the model/threshold/calibration decisions.
7. Generate final test results under the frozen plan.
8. Run the offline rescheduling scenario analysis.
9. Generate final tables, plots and report.

Check that no component recreates labels, splits, encodings, historical features or evaluation rules independently when a canonical shared component exists.

### Step 2 — Create a single end-to-end run path

Provide one documented, ordered command sequence or a small orchestrator with explicit stages. Each stage must fail clearly when an upstream artifact is missing or has the wrong version. Do not hide errors, skip empty outputs silently, or default to stale artifacts without a visible warning.

The run should log configurations and artifact paths. Expensive stages may allow reuse of validated artifacts, but the user must be able to distinguish a fresh run from a cached run.

### Step 3 — Test from a clean environment

- Create a fresh virtual environment or clean container matching the documented Python version.
- Install dependencies only from the canonical file.
- Place the five source CSVs in the documented location or follow the documented download/supply procedure.
- Run formatting/lint checks if adopted, unit tests, data-contract tests, leakage tests and integration smoke tests.
- Run the pipeline far enough to regenerate data-quality outputs and a small end-to-end model run. Then run the required full experiments if the time/resources allow.
- Capture actual commands, environment notes, execution errors and resolved workarounds.

Do not write “reproducible” based only on a developer machine where hidden local files or packages may exist.

### Step 4 — Verify artifact consistency

- Compare source, feature, split and model-manifest versions.
- Confirm the final prediction files contain the expected IDs and labels.
- Recalculate metrics from saved predictions and ensure the final report matches the computed output.
- Regenerate plots from saved artifacts.
- Ensure the final report clearly distinguishes validation selection from the held-out test results.
- Check that no placeholder, sample metric or unrun temporal model is described as a final result.

### Step 5 — Finalize documentation

Update the root README with:

- Project purpose and target definition.
- Dataset files and where to put them.
- Environment setup and dependencies.
- Ordered commands for data checks, feature building, training, evaluation and rescheduling.
- Expected outputs and artifact locations.
- Actual model results with the evaluation protocol, once available.
- Limitations: one year/NYC-origin sample, missing labels, retrospective weather observations, incomplete lookups, and non-causal rescheduling analysis.

Update PRD/design/research/diagrams only where the implementation or reviewed decisions actually differ. Add a model card/data card or equivalent final section describing intended use, population, target, metrics, limitations and non-use cases. Keep initial architecture documents consistent with implemented components.

### Step 6 — Complete team contribution and delivery record

- Review each contributor's component against its plan and tests.
- Record important decisions, deferred work and known defects.
- Keep commits small enough to understand and avoid placing large generated model binaries into version control by default.
- Check license/data-source acknowledgement expectations before any external distribution.
- Agree how the final repository, report, figures and generated model artifacts are submitted.

## 3. Final tests

- Fresh-environment installation succeeds.
- Required tests run and their output/status is recorded.
- Full pipeline stages have clear preconditions and deterministic inputs/configuration.
- Raw files remain unchanged.
- Dataset counts and excluded rows reconcile.
- Feature leakage and chronology tests pass.
- All model prediction files and saved metrics are schema-valid and reproducible from artifacts.
- Rescheduling outputs pass zero-shift equivalence and cutoff tests.
- Final report and plots regenerate without manually editing metric values.
- README commands work on a teammate's machine or any deviations are precisely documented.

## 4. Final handoff package

The final repository should include:

- Root README and engineering/product/research docs.
- Source modules and tests.
- Dependency lock/canonical dependency file and configuration.
- Data-quality report and feature manifest.
- Split manifest, model configs and model/run metadata.
- Validation and final test predictions/metrics as project constraints allow.
- Final figures and report.
- Rescheduling assumptions and outputs.
- Known limitations and deferred experiments, including temporal-model no-go if applicable.

Large intermediate datasets and model checkpoints can remain local or be stored separately when repository size makes versioning impractical; provide an exact regeneration path.

## 5. Acceptance criteria

- [ ] A teammate can follow the README from a clean setup to the documented outputs.
- [ ] Tests and quality gates have recorded outcomes.
- [ ] Data, feature, split, model and report versions agree.
- [ ] Final metrics come from code-generated artifacts and match the report.
- [ ] Test set usage and validation-only decisions are clearly documented.
- [ ] Limitations and no-go decisions are honest and visible.
- [ ] No unsupported production, causal or generalization claims are made.

## 6. Final review checklist

- [ ] Re-run instructions are exact, not “run the notebook” without a specified order.
- [ ] No personal machine paths or secrets are present.
- [ ] Generated data/checkpoints are not accidentally committed unless intended.
- [ ] Every diagram still represents the actual system boundary.
- [ ] The results section contains no placeholder values.
- [ ] All five contributors approve their component handoff.