# Step 4 - Full multi-lead data quality

Eligible flights: 327,346. Four-lead rows: 1,309,384.

Unknown-label flights excluded: 9,430. Invalid scheduled departures: 0.
Eligible flights with unverified outcome availability: 7,538; retain for scoring, withhold from availability-constrained fitting.

| Lead (minutes) | Rows | Positive | Fresh weather | Weather sequences |
|---|---:|---:|---:|---:|
| 1440 | 327,346 | 80,100 | 326,447 | 326,447 |
| 360 | 327,346 | 80,100 | 326,473 | 326,473 |
| 120 | 327,346 | 80,100 | 326,582 | 326,582 |
| 60 | 327,346 | 80,100 | 326,570 | 326,570 |

Same-day departure inputs are disabled. Weather is an observation-based persistence proxy. Every observation and sequence event is checked against its own prediction cutoff. Arrival availability uses the declared reconstruction and buffer assumptions.

Feature manifest SHA-256: `0b8a3ee0244b14a0ee8fa34ac3751d51f22713f596f0dde73fb43237156a118d`.
