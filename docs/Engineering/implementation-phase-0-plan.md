# Implementation Phase 0 Plan — Repository and Experiment Contract

**Lead:** Contributor 1 coordinates; all contributors review  
**Goal:** Agree the contracts that prevent duplicated work, inconsistent model inputs and misleading evaluation before substantial implementation begins.

## 1. Entry conditions

The repository contains the initial project documents and five supplied CSV files: `nyc_flights.csv`, `nyc_airlines.csv`, `nyc_airports.csv`, `nyc_planes.csv`, and `nyc_weather.csv`. Existing code and folders may or may not already exist. Inspect the repository before creating anything; do not assume that a planned file is already implemented.

## 2. Work plan

### Step 1 — Inventory the repository and dataset

- List existing source, test, docs, data and configuration files.
- Confirm the exact locations and filenames of all five CSVs.
- Record file sizes and SHA-256 fingerprints so later runs can identify changed source data.
- Read the current PRD, design, research and diagram docs; log any real ambiguity for team resolution rather than silently rewriting the scope.
- Open each CSV header and verify that required fields match the documents. Full profiling and label counts belong to Phase 1.

**Output:** repository inventory and input manifest. No raw CSV is edited in place.

### Step 2 — Select and document the development environment

- Agree on a Python version that all members can use; if none is prescribed, Python 3.11 is a reasonable starting candidate.
- Choose one canonical dependency-management method: `requirements.txt` or `pyproject.toml`. Do not maintain competing dependency files without a reason.
- Include only dependencies required by the current scope, likely pandas, NumPy, scikit-learn, XGBoost, PyTorch, matplotlib/seaborn alternatives if explicitly chosen, and pytest. If Polars is selected instead of pandas, standardize on it rather than mixing data-frame libraries casually.
- Create a `.gitignore` for virtual environments, local caches, generated datasets, checkpoints, and large model artifacts. Decide explicitly whether the five raw data files are tracked by the repository or provided separately.
- Test a clean environment install before proceeding.

**Example commands (adapt paths and platform):**

```bash
python --version
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3 — Freeze the target and decision point

Document these decisions in a shared configuration or contract file:

- Binary target: 1 when `arr_delay >= 15` minutes; 0 when `arr_delay < 15` minutes.
- A row with missing `arr_delay` has no supervised target and is excluded from model fitting and metric calculation. It remains accounted for in the quality report.
- Prediction time is **T−60 minutes before scheduled departure**.
- `arr_delay` is used only to derive the label and audit the data, never as an estimator input.
- Exclude `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, and `air_time` from model features. Do not automatically use `tailnum`; justify availability at T−60 first.
- Weather freshness is three hours before the T−60 cutoff. Only observations timestamped at or before the cutoff are eligible.
- Historical delay aggregates use flights from calendar dates strictly earlier than the target flight date.

### Step 4 — Define the chronological split

The proposed split is:

| Split | Scheduled flight dates | Purpose |
|---|---|---|
| Train | Through 31 August 2013 | Fit model parameters and training-only preprocessing |
| Validation | 1 September–31 October 2013 | Tune hyperparameters, compare candidates, choose operating threshold and fit calibration as appropriate |
| Test | 1 November–31 December 2013 | One final held-out comparison after decisions are frozen |

Contributor 5 must inspect row counts and target prevalence across the windows before freezing the split. If a date boundary needs to change, record the reason, revised dates and decision date before model comparisons begin. Do not adjust dates in response to final test results.

### Step 5 — Standardize the data and artifact interfaces

Create and document these contracts:

1. `flight_row_id`: stable, unique identifier for a source record; it must survive cleaning, feature building, train/validation/test assignment and prediction export.
2. Labelled-data schema: schedule identifiers and timestamps, `flight_row_id`, target, and explicit handling for unknown labels.
3. Feature schema: the feature columns are separated from identifiers and target. Each feature has a manifest entry with source, availability time, transform, type and missing-value policy.
4. Prediction schema: `flight_row_id`, `y_true`, `y_probability`, `split`, `model_name`, and `run_id` at minimum.
5. Run configuration: data fingerprint, code commit, model settings, seed, feature-manifest version and split-manifest version.

Agree file format and paths as well. CSV is acceptable for inspectable reports; Parquet may be used for intermediate tables if everyone has the dependency and workflow. The choice matters less than one consistent contract.

### Step 6 — Set the metric and threshold policy

- Required ranking metrics: PR AUC and ROC AUC.
- Required operational metric: recall at a stated precision floor, with the threshold chosen on validation data only.
- Required calibration evidence: Brier score plus a reliability diagram or a documented alternative.
- The actual precision floor must be agreed before threshold analysis. A value such as 0.70 is only a candidate and must not be presented as a stakeholder requirement unless the team approves it.
- Test outcomes are for the final evaluation only and must not influence feature choices, hyperparameters, calibration, threshold or model selection.

### Step 7 — Set repository organization and quality gates

Use the existing organization if it is already coherent; otherwise agree on a minimal structure such as:

```text
data/raw/             # original CSV inputs; never edit
 data/processed/      # generated labelled data/features
src/data/             # loading, profiling, target
src/features/         # feature construction
src/models/classical/ # Logistic Regression, XGBoost
src/models/neural/    # embedding MLP, optional temporal experiment
src/evaluation/       # split, metrics, calibration
src/rescheduling/     # offline what-if analysis
tests/                # data, leakage, schema and model tests
reports/              # tables, plots and write-up
artifacts/            # local/generated model artifacts; normally gitignored
docs/Engineering/     # phase plans and engineering contracts
```

## 3. Deliverables

- Repository and source-data inventory.
- Canonical dependency/environment file and `.gitignore`.
- Written target/T−60, weather freshness, history, split and metric policies.
- Data/feature/prediction artifact schemas.
- Initial test command and baseline repository test result.
- Task allocation that matches the ownership table in `Implementation-Phases.md`.

## 4. Tests and review checklist

- [ ] The environment can be recreated from the documented dependency file.
- [ ] No raw CSV has been changed.
- [ ] The target definition and missing-label policy are unambiguous.
- [ ] The proposed date split has been checked against date-level counts and target prevalence.
- [ ] Every contributor understands the shared schemas and their component boundary.
- [ ] The repository has one canonical place for config, tests and generated artifacts.
- [ ] The metric/threshold policy protects the test set from tuning.

## 5. Acceptance criteria

Phase 0 is complete when all shared contracts are reviewed and written down, a clean environment is installable, file paths are unambiguous, and the team agrees on the first implementation tasks. This phase does not require a trained model.

## 6. Risks and stop conditions

- **Conflicting existing code:** audit and reuse suitable code before making parallel replacements.
- **Uncertain time-zone or timestamp semantics:** document the unresolved assumption and investigate before feature creation; do not guess silently.
- **Dataset version mismatch:** regenerate the source manifest and recalculate expected counts; never hard-code counts as runtime truth.
- **Unagreed precision floor:** continue reporting ranking/calibration curves, but defer claims about meeting an operational recall target.
