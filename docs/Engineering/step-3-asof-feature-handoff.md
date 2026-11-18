# Step 3 as-of feature builder

`AsOfFeatureBuilder` indexes the existing source tables once and takes a flight ID and timezone-aware cutoff for each query. It supports the four lead times in [protocol v2](../../configs/project_contract_v2.yaml).

```python
from datetime import timedelta

from src.data.config import RAW_DATA_DIR
from src.data.loading import load_all_sources
from src.features.asof import AsOfFeatureBuilder

builder = AsOfFeatureBuilder(load_all_sources(RAW_DATA_DIR))
flight_id = builder.flights.index[100_000]
departure_utc = builder.flights.loc[flight_id, "scheduled_departure_utc"]
cutoff = departure_utc - timedelta(hours=6)
row = builder.build(flight_id, cutoff, include_sequences=True)
feature_order = builder.feature_manifest()["allowlist"]
estimator_inputs = [row.features[name] for name in feature_order]
```

For Step 4, use `builder.flights` to enumerate stable IDs, local service dates, and scheduled UTC departures. Build each row with the requested cutoff and join labels by `flight_row_id` through the existing Phase 1 output. The separate `builder.outcome_available_utc` series provides the earliest estimated label-availability time for each fold. Treat missing timestamps as unavailable for fitting. Keep the cutoff, ID, observed-weather timestamp, sequence timestamps, and target in audit or label columns outside estimator inputs. The manifest records the exact feature allowlist and whether departure inputs were enabled.

## Timing and missing data

- Arrival history uses the seven- and 28-day elapsed windows ending at the cutoff. A completed flight enters the window only when reconstructed arrival plus a 15-minute buffer is at or before the cutoff. Arrival times are checked against the recorded destination-local clock, and departure times are checked against the recorded New York clock. Missing time zones and contradictory clocks withhold the event. The supplied data has no release timestamps, so the buffer is a retrospective availability assumption.
- Carrier, route, and origin rates shrink toward the global rate from that same available window with 20 prior observations. Each query excludes the target flight from both group and global history. Counts of zero pair with missing rates.
- Weather uses origin observations timestamped at or before the cutoff as a persistence proxy. A reading older than three hours yields missing weather inputs. Trends are hourly changes from the earliest valid observation in the three- or six-hour window to the latest fresh observation. A fresh endpoint also enables a sequence of up to 12 past readings.
- Set `allow_same_day_departures=True` only after the protocol's instructor approval. This adds the count and mean delay of earlier origin flights that departed on the target service date and cleared the 15-minute buffer by the cutoff. The optional sequence contains up to 32 available origin departures, including previous dates where needed.
- Sequence records are oldest to newest, left-padded with `None`, and paired with a boolean mask. Weather and departure records carry their UTC event time. Departure records also carry their estimated availability timestamp. Model preprocessing can use the audit timestamps to verify the cutoff and the mask to ignore padding.
- Calendar distance uses the observed U.S. federal holiday calendar. Schedule and reference inputs use the published schedule and static lookups as available at all four lead times. The CSVs provide one schedule snapshot, so revision histories cannot be verified.

Run `python -m pytest -q tests/test_step3_asof.py` for cutoff, missing-data, and perturbation checks.
