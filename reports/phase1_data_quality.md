# Phase 1 — Data Quality and Label Report

## 1. Source files

| File | Rows | Columns | Size (bytes) | SHA-256 (first 8) |
|---|---:|---:|---:|---|
| nyc_flights.csv | 336776 | 19 | 31053850 | `563db8f1` |
| nyc_airlines.csv | 16 | 2 | 386 | `162551bd` |
| nyc_airports.csv | 1458 | 8 | 104302 | `d3b9145e` |
| nyc_planes.csv | 3322 | 9 | 247198 | `778962ed` |
| nyc_weather.csv | 26115 | 15 | 2294220 | `e70e506b` |

## 2. Schema validation

- `flights`: passed (336776 rows, 19 columns)
- `airlines`: passed (16 rows, 2 columns)
- `airports`: passed (1458 rows, 8 columns)
- `planes`: passed (3322 rows, 9 columns)
- `weather`: passed (26115 rows, 15 columns)

## 3. Flights missingness (columns with gaps)

| Column | Missing | % |
|---|---:|---:|
| dep_time | 8255 | 2.4512 |
| dep_delay | 8255 | 2.4512 |
| arr_time | 8713 | 2.5872 |
| arr_delay | 9430 | 2.8001 |
| tailnum | 2512 | 0.7459 |
| air_time | 9430 | 2.8001 |
| substantial_delay | 9430 | 2.8001 |

## 4. Duplicates and lookup keys

- Full duplicate flight rows: **0**

| Lookup | Key | Duplicate key rows |
|---|---|---:|
| airlines | carrier | 0 |
| airports | faa | 0 |
| planes | tailnum | 0 |
| weather | origin, time_hour | 0 |

## 5. Schedule parsing

- Invalid `sched_dep_time` values: **0**
- Invalid `sched_arr_time` values: **0**
- Rows with unusable scheduled departure timestamp: **0**

- Scheduled dates: 2013-01-01 to 2013-12-31 (365 distinct days)

## 6. Label summary and reconciliation

- Total flight rows: **336776**
- Known labels: **327346** (positive 80100, negative 247246, prevalence 0.244695)
- Unknown labels: **9430** (also missing dep_time: 8255; dep_time present: 1175)

Reconciliation: known + unknown = 336776 = total rows (336776): **OK**

### Comparison with the contract reference profile

| Count | Expected | Measured | Match |
|---|---:|---:|---|
| total_rows | 336776 | 336776 | yes |
| known_labels | 327346 | 327346 | yes |
| positive | 80100 | 80100 | yes |
| negative | 247246 | 247246 | yes |
| unknown_labels | 9430 | 9430 | yes |

## 7. Join-key coverage (profile only; joins are Phase 2)

- Unmatched destination airports: 7602 rows over ['BQN', 'PSE', 'SJU', 'STT']
- Unmatched origin airports: 0 rows
- Unmatched carriers: 0 rows
- Plane lookup: 284170 of 334264 non-missing tailnum rows matched (50094 unmatched)
- Same-hour weather observations available for 335220 of 336776 rows (profile only)

## 8. Monthly label distribution

| Month | Labelled | Positive | Negative | Prevalence |
|---:|---:|---:|---:|---:|
| 1 | 26398 | 6201 | 20197 | 0.234904 |
| 2 | 23611 | 5615 | 17996 | 0.237813 |
| 3 | 27902 | 6586 | 21316 | 0.23604 |
| 4 | 27564 | 7925 | 19639 | 0.287513 |
| 5 | 28128 | 6376 | 21752 | 0.226678 |
| 6 | 27075 | 8535 | 18540 | 0.315235 |
| 7 | 28293 | 9112 | 19181 | 0.322058 |
| 8 | 28756 | 7073 | 21683 | 0.245966 |
| 9 | 27010 | 3705 | 23305 | 0.137171 |
| 10 | 28618 | 5026 | 23592 | 0.175624 |
| 11 | 26971 | 4882 | 22089 | 0.181009 |
| 12 | 27020 | 9064 | 17956 | 0.335455 |

## 9. Notes

- Missing arr_delay is an unknown label, not a negative and not a confirmed cancellation (no cancellation/diversion flag exists).
- arr_delay is an audit column in the labelled dataset only; it must never enter a model's feature matrix.
- Weather coverage here is a same-hour profile; the T-60 as-of join is built in Phase 2.
- Unmatched destination airports (e.g. BQN, PSE, SJU, STT) and incomplete plane-lookup coverage are preserved and reported, never dropped.
