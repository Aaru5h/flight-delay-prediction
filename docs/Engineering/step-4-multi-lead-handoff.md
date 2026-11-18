# Step 4 multi-lead datasets

Use a checkout with protocol v2, the Step 2 schedule helpers, and the Step 3 builder. From the repository root, run:

```bash
python -m src.features.build_multi_lead
```

The command reads the raw tables and `data/processed/labelled_flights.csv`. Run `python -m src.data.prepare` if the label file is missing or the raw flights have changed. Use `--raw-dir`, `--labelled-path`, and `--output-dir` for a different input or output location.

## Output files

The default output directory is `data/processed/step4/`. The build writes in batches and publishes the completed directory after all partitions and reports succeed. Use a new output directory for a fresh run if the destination already contains files.

| File | Contents |
|---|---|
| `features_1440m.csv`, `features_360m.csv`, `features_120m.csv`, `features_60m.csv` | One row per eligible flight at each lead time. Join and compare by `flight_row_id` and `lead_time_minutes`. |
| `feature_manifest.json` | Ordered estimator allowlist, audit and label fields, lead-time partitions, and source and label-file fingerprints. |
| `data_quality.json` | Source and eligibility counts, label counts, weather coverage, sequence coverage when exported, and missing values for each feature by lead time. |
| `sequences_<lead>m.jsonl.gz` | Optional sequence records with the same key as the CSV, UTC event timestamps, and boolean padding masks. |

Use `--include-sequences` to write weather sequences without changing tabular features. If same-day departure inputs have instructor approval, also pass `--allow-same-day-departures` to export the departure features and, when sequences are requested, departure sequences.

The supplied data contains 327,346 known labels, or 1,309,384 rows across four lead times when all known-label flights have valid scheduled departures. Unknown labels and ambiguous or missing departure times are counted in the report and excluded. A known-label row can still have a blank `outcome_available_utc`. Keep that row for evaluation and exclude it from any training fit that requires verified label availability.

## Model handoff

Use the manifest to open a partition and select estimator columns. `lead_time_minutes` is both part of the row key and a model input. Assign folds from `sched_date` and join the resulting fold manifest by `flight_row_id` so all four rows stay together within each fold.

```python
import json
from pathlib import Path

import pandas as pd

root = Path("data/processed/step4")
manifest = json.loads((root / "feature_manifest.json").read_text())
partition = next(part for part in manifest["partitions"] if part["lead_time_minutes"] == 60)
frame = pd.read_csv(root / partition["features"])
X = frame[manifest["allowlist"]]
y = frame[manifest["label_column"]]
```
