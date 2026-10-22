# Implementation Phase 2 Plan — Leakage-Safe Feature Engineering

**Lead:** Contributor 2  
**Goal:** Create a documented feature matrix that represents information available at T−60 and can be reused by all three required model families.

## 1. Entry conditions

Phase 1's labelled table, timestamp convention, stable `flight_row_id`, source profile and exclusion policy are available. Phase 0's T−60 contract and chronological split are agreed. No feature should be added solely because a column exists in the CSV.

## 2. Feature groups and work plan

### Step 1 — Define feature availability before coding

Start a feature manifest that gives every candidate field:

- Feature name and type.
- Source column/table and join rule.
- When the information would be available operationally.
- Any time cutoff or freshness requirement.
- Transformation and missing-value policy.
- Whether it is approved, rejected or awaiting justification.

Create an explicit **allowlist** of approved predictors and a **denylist** of leakage-prone fields. At a minimum, do not pass `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, or `air_time` into `X`. `arr_delay` remains the label source only. Treat `tailnum` and aircraft-derived variables as excluded until the team documents why aircraft assignment is known at T−60.

### Step 2 — Build schedule and calendar features

Candidate features include:

- Scheduled departure hour, day of week, month and day-of-year where useful.
- Cyclical encodings for hour/day-of-week or month where they improve the selected model.
- Scheduled travel/route descriptors only where their meaning and availability are justified.
- Schedule-only departure volume/congestion counts calculated from schedules, not observed departure or arrival outcomes.

Use the scheduled timestamp as the canonical input. Keep identifiers and target outside the estimator feature set. Document any derived feature whose interpretation might be ambiguous.

### Step 3 — Join reference data without dropping flights

- Join airline labels using carrier code.
- Join origin/destination airport attributes using airport codes.
- Keep joins left-sided so unmatched airport lookup rows do not drop flight records.
- Record unmatched key counts separately for origin and destination.
- Treat aircraft lookup as optional and leave aircraft attributes out unless availability is justified in the feature manifest.
- Assert that each reference join does not unexpectedly multiply rows. If a lookup key is non-unique, resolve or block the join instead of duplicating flights.

### Step 4 — Join weather as of the prediction cutoff

For each flight:

1. Compute the prediction cutoff as scheduled departure minus 60 minutes.
2. Select observations for the **origin** airport only.
3. Choose the most recent weather observation with timestamp less than or equal to the cutoff.
4. Reject an observation older than three hours before the cutoff and record weather as unavailable for that row.
5. Record observation timestamp, age-at-cutoff and a missing/stale flag for auditing. Do not feed audit-only columns to the model unless deliberately approved.

Document the interpretation of timestamps/time zones. The supplied weather records are retrospective observations, not forecast snapshots, so the analysis must not claim the same availability as a real operational forecast. Do not select a weather observation simply because it shares the scheduled departure hour.

### Step 5 — Create historical delay and congestion features

- Start with simple, auditable aggregates such as prior-date carrier delay rate, prior-date route delay rate and prior-date origin departure/arrival congestion if appropriate.
- Build them incrementally by service date or through an equivalent strictly ordered approach. For a target flight on date D, use only outcomes from dates earlier than D.
- Exclude all outcomes from the target flight date, the target row and all later dates. Do not precompute full-dataset averages and then split.
- Specify minimum-history thresholds and fallback behavior. When history is insufficient, use a missing indicator/approved prior, not an unexamined zero.
- Consider smoothing for low-count carrier-route groups and document its formula and how its prior is estimated without future leakage.
- Ensure training rows do not receive statistics that include their own labels.

### Step 6 — Produce the feature matrix and manifest

- Keep one row per eligible labelled `flight_row_id`.
- Store target and identifier in separate fields, then explicitly build the estimator matrix from the approved allowlist.
- Save the feature matrix in the canonical format and generate a manifest/version identifier.
- Keep preprocessing objects that learn from data (imputers, scalers, encoders) inside model pipelines and fit them on training rows only. Do not globally scale or encode the full dataset before splitting.
- Document expected data types and how unknown categories are handled downstream.

## 3. Leakage and integrity tests

Required tests include:

- No denied/post-event column appears among model features.
- All weather timestamps are at or before their row's T−60 cutoff.
- Weather older than three hours is marked unavailable.
- Historical aggregate for date D does not change when outcomes on D or later dates are perturbed.
- Row count and unique `flight_row_id` remain stable through left joins.
- Unmatched reference keys are preserved and summarized.
- Feature and label row IDs align exactly; no silent inner join removes targets.
- Feature names, types and availability rules match the saved manifest.
- A synthetic example with a future weather record or future delay label proves that the relevant feature builder ignores it.

## 4. Deliverables

- Reusable schedule/calendar feature builder.
- Left-join reference feature builder with coverage report.
- Weather-as-of builder with freshness cap and audit fields.
- Prior-date historical aggregate builder.
- Versioned feature matrix, allowlist/denylist and feature manifest.
- Automated leakage, join-cardinality, row-alignment and missingness tests.
- A short limitations note on retrospective weather observations.

## 5. Acceptance criteria

- [ ] Every approved feature has a written T−60 availability rationale.
- [ ] No known post-departure outcome reaches `X`.
- [ ] Weather is joined by origin and as-of cutoff, with stale observations rejected.
- [ ] Historical aggregates use strictly earlier dates.
- [ ] Reference joins preserve the original flight population.
- [ ] Model-ready features, labels and IDs align and can be regenerated.
- [ ] The manifest and leakage tests pass before model training begins.

## 6. Risks and mitigations

- **Timezone ambiguity:** establish and test a documented conversion before comparing times. If it cannot be resolved, block the weather join rather than guess.
- **Same-day historical leakage:** build date-grouped unit tests that fail if same-day outcomes affect features.
- **Aircraft assignment leakage:** omit the field unless supported by evidence and a reviewed change.
- **Feature availability drift:** require any new feature to update the manifest and tests in the same change.

## 7. Handoff

Deliver one canonical feature table and a manifest used by Contributors 3 and 4. Include the command to regenerate it from Phase 1 outputs, the exact approved columns, and known missingness/coverage caveats.