# Phase 2 — Leakage-Safe Feature Quality Report

**Status:** Implemented on the complete labelled population; leakage tests pass.

## Output

- Eligible labelled flights: **327,346**
- Approved estimator features: **37**
- Grain: one unique row per eligible `flight_row_id`
- Feature manifest: `data/processed/phase2_feature_manifest.json` (v2.0.0)
- Feature matrix: `data/processed/phase2_features.csv`

## T−60 controls

- Schedules use `America/New_York`; raw UTC fields confirm the convention (for example, 05:15 EST corresponds to 10:00Z).
- Weather is the latest origin observation at or before scheduled departure minus 60 minutes. Measurements older than 180 minutes are withheld, while timestamp and age remain audit-only.
- Historical carrier and route rates use calendar dates strictly earlier than the target date.
- `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`, and `tailnum` are denied estimator inputs.

## Measured coverage

- Fresh usable weather: **326,570 rows**
- Stale or missing weather: **776 rows**
- Unmatched airline/origin-airport rows: **0**
- Unmatched destination-airport rows: **7,537** (preserved)
- Missing initial carrier history: **834 rows**
- Missing initial route history: **894 rows**

## Limitations

Weather values are retrospective observations, not archived forecast vintages. Aircraft features remain excluded because assignment availability at T−60 is not established. Imputation, scaling, and encoding must be fitted inside model pipelines on training data only.
