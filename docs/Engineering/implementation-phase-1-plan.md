# Implementation Phase 1 Plan — Load, Profile, Clean and Label Data

**Lead:** Contributor 1  
**Goal:** Turn the five supplied source CSVs into a reproducible, validated labelled flight table and a transparent data-quality report. Do not build predictive features or train models in this phase.

## 1. Inputs and dependencies

- Phase 0 target, prediction-time and schema contracts.
- `nyc_flights.csv`, `nyc_airlines.csv`, `nyc_airports.csv`, `nyc_planes.csv`, and `nyc_weather.csv` in the agreed raw-data directory.
- Agreed Python/dependency setup and test command.

Keep the original CSVs read-only. Store outputs in the designated processed-data and reports directories.

## 2. Implementation steps

### Step 1 — Build a single input loader

- Define one reusable entry point for locating and reading the five files.
- Fail with an actionable error if a required file is absent or unreadable.
- Read data without altering source files. Avoid unnecessary type inference for codes with leading zeros; keep identifiers as strings where appropriate.
- Record row count, column count, column names, inferred/declared types, file size and SHA-256 fingerprint for each input.
- Make the input paths configurable rather than hard-coded to one developer's machine.

**Do not** load unrelated copies or silently choose a file based on filesystem order.

### Step 2 — Validate schemas and basic constraints

- Check required column presence for each file and report all missing required fields in one clear error.
- Check expected key fields, such as carrier codes, airport codes, tail number, origin/destination, dates and scheduled times.
- Report duplicate rows and duplicate lookup keys; do not drop them until the reason and expected behavior are clear.
- Profile missingness by column, distinct counts for categorical keys, numeric ranges and parse failures.
- Check that scheduled dates fall within the expected 2013 period, but treat this as a validation expectation rather than a reason to rewrite out-of-range data silently.
- Track duplicate flight-like records for review. Two similar records are not automatically duplicates unless the contract supports a safe key.

### Step 3 — Create the stable source row identifier

Create `flight_row_id` before exclusions or sorting. It must be unique and stable for a given source-file version. One viable implementation is an identifier based on the source file fingerprint and original row position; a simpler integer row index is acceptable only when the source file's identity/order is captured in the run manifest.

Validate uniqueness and preserve the identifier in all derived tables. Do not use a target or a post-event field to create the identifier.

### Step 4 — Parse schedule and date fields

- Parse the scheduled departure date and time fields under the documented source convention.
- Distinguish scheduled fields from realized times. Values encoded as HHMM-like integers need careful handling of nulls and edge cases; avoid treating them as ordinary decimal hours.
- Derive a normalized scheduled-flight date/timestamp needed by the split contract.
- Handle invalid and missing schedule values explicitly and report their counts. Do not fill unknown schedule times with arbitrary defaults.
- Record the time-zone assumption and any limitation; downstream feature and weather work depends on it.

### Step 5 — Construct the supervised target

Implement the exact policy:

- If `arr_delay >= 15`, assign `substantial_delay = 1`.
- If `arr_delay < 15`, assign `substantial_delay = 0`.
- If `arr_delay` is missing, keep target as missing/unknown and exclude the row from supervised train, validation and test populations.

Do not reinterpret missing arrival delay as on-time, and do not assert that every missing record is a cancellation: the source file does not provide a definitive cancellation/diversion flag.

### Step 6 — Produce a data-quality and label report

At minimum, report:

- Rows and columns read per file.
- Source fingerprints and schema validation outcome.
- Missing values per column, duplicate/unique-key checks and parse failures.
- Total flight rows, number with known label, number with unknown label, positive/negative counts and positive prevalence among labelled rows.
- Exclusion counts by reason, with overlapping reasons identified as overlapping if they are not mutually exclusive.
- Date coverage and daily/monthly row and target distributions.
- Join-key coverage for airline, airport, plane and weather tables as a profile only in this phase; actual modeling joins and weather as-of behavior are Phase 2.
- Warnings for unmatched airport codes and incomplete aircraft lookup coverage.

Use the counts measured at runtime. The numbers in README/research docs are checks for the current supplied version, not constants to place inside code. If measured results differ, investigate file version, parsing, filtering and counting rules before updating docs.

### Step 7 — Save deterministic outputs

- Save one labelled-data artifact with `flight_row_id`, needed raw/schedule columns, target and documented label-status fields.
- Save a machine-readable summary (JSON/CSV) and a readable report (Markdown or HTML) under `reports/`.
- Save the source manifest and a compact exclusion audit.
- Use deterministic sorting and formatting so reruns make meaningful diffs.

## 3. Unit and data tests

Add tests for:

- The threshold boundary: 14 minutes is class 0, 15 minutes is class 1, and negative delay is class 0.
- Missing `arr_delay` produces an unknown label and is not converted to zero.
- `flight_row_id` is non-null, unique and preserved after filtering.
- Missing required columns produce a clear schema-validation failure.
- Invalid HHMM-like time values are flagged instead of being silently normalized.
- Counts reconcile: labelled + unknown labels equals all source flight rows before any separately documented invalid-row exclusions.
- Raw-file fingerprints stay unchanged across the run.

Run the phase tests with the repository's agreed command, typically `python -m pytest`.

## 4. Expected deliverables

- Reusable loader and schema validation.
- Stable `flight_row_id` strategy.
- Reproducible date/schedule parsing.
- Target construction and exclusion handling.
- Machine-readable source/quality manifests and readable data-quality report.
- Unit/data tests and a short data contract update if observed schemas require clarification.

## 5. Acceptance criteria

- [ ] All five source files load or the program fails with a useful message.
- [ ] Required fields and types are checked rather than assumed.
- [ ] Target boundary and missing-label tests pass.
- [ ] The row-count reconciliation is explained and repeatable.
- [ ] The labelled dataset can be regenerated from the raw CSVs using one documented command.
- [ ] Data-quality outputs report actual measured results, including unmatched-key caveats.
- [ ] No engineered predictive features or model-training code is mixed into this phase.

## 6. Known risks

- **Changing dataset copy:** counts differ from documentation. Recompute from the current source and document the difference.
- **Missing arrival delay:** its reason may not be directly identifiable; report patterns, not unproven causes.
- **Time parsing defects:** can corrupt chronological splits; test edge cases before handing timestamps to Contributor 2.
- **Accidental row drop:** compare row identifiers before/after transforms and account for every exclusion.

## 7. Handoff to Phase 2

Provide the labelled table, stable identifier definition, schedule/date parsing behavior, source profile, exclusions and exact command to regenerate them. Contributor 2 should not have to re-read or reinterpret the raw source independently.