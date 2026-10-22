# Implementation Phase 6 Plan — Offline Rescheduling What-If Analysis

**Lead:** Contributor 5; Contributor 2 supports feature rebuilding, Contributors 3–4 support model loading/scoring  
**Goal:** Estimate how the selected model's predicted delay risk changes under explicit shifts to the scheduled departure time, while being transparent that this is a non-causal sensitivity analysis.

## 1. Entry conditions

- Phase 2 has a tested feature builder that can be called for a candidate schedule rather than only the original schedule.
- Phase 5 has selected the model, calibration procedure and any operating threshold using validation data only.
- The selected model and all preprocessing/calibration artifacts can be reloaded.
- The team accepts the scope limitation: supplied weather data is retrospective observations, not future forecast vintages; the analysis is not a full airline schedule optimizer.

## 2. Define the scenario contract

Before coding, document:

- Baseline schedule timestamp for each flight.
- Candidate shifts. Initial examples are −60, −30, −15, +15, +30 and +60 minutes, subject to declared operating-hour and date boundaries.
- Whether shifts that cross midnight are allowed, and how date-derived features change when they do.
- Which fields are recomputed for each candidate and which remain unchanged.
- How weather is selected for the shifted T−60 cutoff and how stale/missing observations are handled.
- The risk-change definition, such as `candidate_probability - baseline_probability`, and how candidates are flagged when inputs are unsupported.
- Whether the output is per-flight or aggregated by route/carrier/hour, and how low counts are handled.

These are hypothetical candidate times, not actual historical schedule changes.

## 3. Implementation steps

### Step 1 — Load the frozen scoring pipeline

Load the chosen model, preprocessing state, category mappings, feature manifest and calibrator (if used). Check the model and feature versions against the evaluation report. Do not retrain or recalibrate using test labels in this phase.

### Step 2 — Generate candidate schedules

For every eligible analysis row, create a baseline candidate and one row per candidate time shift. Keep a stable `scenario_id`, original `flight_row_id`, shift minutes, original timestamp and candidate timestamp. Validate that each candidate timestamp follows the declared operating-window rules. Do not silently roll a candidate into the next day unless the scenario contract permits it.

### Step 3 — Rebuild time-dependent inputs

- Recompute hour/day/month/cyclic schedule features from the candidate timestamp.
- Recompute any schedule-volume feature only if its definition supports the candidate scenario and the counterfactual assumptions are documented. Otherwise hold it fixed or mark it unavailable; do not implicitly simulate an entire airport schedule.
- Recalculate T−60 for the candidate time and perform the same origin-weather as-of join with the same three-hour freshness cap.
- Keep prior-date historical aggregates fixed for that flight date because the target flight's hypothetical time shift must not insert its own outcome into history.
- Rebuild any other time-sensitive feature through the shared feature builder rather than handwritten duplicate logic.
- Verify that the resulting feature columns, types and order match the selected model's manifest exactly.

Historical weather observations at the new cutoff do not equal a real forecast for a newly scheduled departure. This limitation must remain visible in outputs and final interpretation.

### Step 4 — Score each scenario

- Score the baseline and candidates with the same frozen model and preprocessing pipeline.
- Apply the same frozen calibrator, if applicable.
- Verify all probability values are finite and in `[0, 1]`.
- Calculate risk difference from baseline and, if helpful, relative change with safeguards for baseline probabilities near zero.
- Do not interpret the score difference as the causal impact of rescheduling.

### Step 5 — Summarize candidate comparisons

Report:

- Number of flights/scenarios scored and exclusions/invalid candidates by reason.
- Baseline and candidate mean/median predicted risk, distributions and risk-difference summaries.
- Results stratified carefully by relevant route/carrier/hour groups with denominators.
- Candidate shifts that appear to lower model scores, clearly labelled as model-predicted sensitivity.
- Sensitivity to time window, weather availability and other key assumptions where feasible.
- Explicit limitations from omitted aircraft rotations, crew legality, gates, airport capacity, network knock-on delays and passenger connections.

Avoid ranking specific shifts as recommended operational choices unless separate validated operational constraints and evidence exist.

## 4. Tests

- A zero-minute shift yields the same features and probability as the baseline (within tolerance).
- Shifted schedule features change consistently, including day/month boundaries where supported.
- Weather timestamps never exceed each candidate's T−60; stale observations follow the same rejection rule.
- Prior-date aggregates are unchanged by candidate shifts and do not include the target flight's label.
- Candidate records map back to original `flight_row_id` and have unique scenario IDs.
- Scoring uses the same model, preprocessing and calibrator for every candidate.
- Invalid/out-of-window candidates are rejected or flagged according to the contract.

## 5. Deliverables

- Scenario configuration and candidate generator.
- Reusable time-dependent feature rebuild path.
- Candidate-level predictions and risk deltas in CSV/Parquet.
- Aggregated summary tables and at least one explanatory plot.
- Report of assumptions, unsupported cases and non-causal limitations.
- Automated counterfactual consistency tests.

## 6. Acceptance criteria

- [ ] Candidate shifts and bounds are documented and reproducible.
- [ ] Time-derived and weather features are rebuilt from the candidate timestamp through the shared code path.
- [ ] Model, preprocessing and calibration are frozen from Phase 5.
- [ ] Baseline equivalence and cutoff tests pass.
- [ ] Every exclusion or invalid scenario is counted and explained.
- [ ] All outputs explicitly label results as non-causal, model-based what-if analysis.
- [ ] No claim is made that the exercise optimizes actual airline schedules or proves real delay reduction.

## 7. Risks and mitigations

- **False counterfactual precision:** state assumptions next to charts/tables, not only in an appendix.
- **Weather availability mismatch:** use the same cutoff rule and disclose that historical observations are not forecasts.
- **Schedule interactions ignored:** avoid claiming other flights, aircraft, crews or gates remain operationally feasible under a shift.
- **Duplicate feature logic:** call the Phase 2 builder to keep model inputs consistent.