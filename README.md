# Flight Delay Prediction and Rescheduling

**Project title:** Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning  
**Project type:** Applied machine-learning project and offline what-if analysis  
**Dataset:** NYC-origin flight records and associated airline, airport, aircraft, and weather CSV files for 2013  
**Primary task:** Predict whether a flight will arrive at least 15 minutes late, using information available before departure

---

## 1. Overview

Flight delays create uncertainty for passengers, airlines, airport operations, and connecting services. This project builds and compares machine-learning models that estimate whether a scheduled flight departing from one of New York City's three major airports—Newark Liberty (EWR), John F. Kennedy (JFK), or LaGuardia (LGA)—will arrive substantially late.

The project uses scheduled flight information, airline and route identifiers, calendar information, aircraft reference data where its availability can be justified, weather observations available before the prediction cutoff, and historical delay patterns constructed strictly from earlier flights. The models to compare are Logistic Regression, XGBoost, and a compact neural network with learned categorical embeddings. A temporal model is optional and will only be attempted if the ordering and availability of its history can be established correctly.

The final analytical component is an offline rescheduling what-if analysis. It compares predicted delay risk for explicitly defined alternative scheduled departure times. It is a sensitivity analysis, not a causal estimate and not a real-world dispatch recommendation.

## 2. Problem statement

The task is to estimate substantial-arrival-delay risk before departure. A model is not useful if it obtains attractive scores by using information that only becomes available after the flight departs or arrives. The project therefore treats prediction-time data availability and chronological evaluation as core correctness requirements, not optional refinements.

## 3. Dataset supplied with the project

The project currently includes five CSV files. The following profile was measured from the supplied files; the data pipeline should recheck schemas and counts whenever it runs.

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `nyc_flights.csv` | 336,776 | 19 | Flight schedule, route, carrier, and observed outcomes for 2013 |
| `nyc_airlines.csv` | 16 | 2 | Carrier-code to airline-name lookup |
| `nyc_airports.csv` | 1,458 | 8 | Airport names, locations, and reference attributes |
| `nyc_planes.csv` | 3,322 | 9 | Aircraft reference attributes keyed by tail number |
| `nyc_weather.csv` | 26,115 | 15 | Hourly weather observations at EWR, JFK, and LGA |

The flight records cover 1 January through 31 December 2013. `arr_delay` is present for 327,346 records and missing for 9,430. Of the records with a valid arrival delay, 80,100 meet the target threshold of at least 15 minutes (24.47%) and 247,246 do not (75.53%). Missing arrival delay must not be silently converted to zero.

The source files do not contain an explicit cancellation or diversion flag. Missing arrival-delay records must therefore be reported as unlabeled records; they must not all be described as confirmed cancellations. The airport reference file does not cover four destination codes present in the flights file (`BQN`, `PSE`, `SJU`, and `STT`), and aircraft lookup coverage is incomplete. These are known join limitations to preserve in data-quality reporting.

For field-level definitions and join considerations, see [Research](docs/Research/Research.md), [Data/Entity Relationships](docs/Architecture%20Diagrams/ER-Diagram.md), and [Product Requirements](docs/Product/PRD.md).

## 4. Target definition

For each flight with a known `arr_delay`:

- `substantial_delay = 1` when `arr_delay >= 15` minutes.
- `substantial_delay = 0` when `arr_delay < 15` minutes, including early arrivals and small delays.
- Flights with missing `arr_delay` are excluded from supervised classification and counted in the data-quality report.

The actual `arr_delay` is used to create the target only. It must never be included among model inputs.

## 5. Prediction-time contract

The baseline project design uses a prediction cutoff of **60 minutes before the scheduled departure time (T−60)**. Every model must use features that would be available by that cutoff. If the team later changes this decision time, the feature policy, evaluation protocol, and rescheduling analysis must all be updated together.

The following rules apply:

- Scheduled time, calendar, carrier, origin, and destination fields may be considered when their availability at T−60 is established.
- Actual departure time, departure delay, actual arrival time, arrival delay, and air time are post-departure outcomes and cannot be model inputs.
- Use the latest origin-airport weather observation timestamped no later than T−60. For the initial design, reject an observation if it is more than three hours older than the cutoff. Historical observations are not weather forecasts; that limitation must be acknowledged.
- Historical delay/congestion features may use only flights from dates strictly before the target flight's date. Same-day outcomes and the target flight itself are excluded.
- Treat `tailnum` as unavailable unless the team can justify that the assigned aircraft was known at T−60. Do not use it merely because it appears in the CSV.

## 6. Models and evaluation

The project compares:

1. **Logistic Regression** as the interpretable baseline.
2. **XGBoost** as the boosted-tree model for nonlinear relationships and feature interactions.
3. **Compact embedding MLP** using learned representations for airline, origin, and destination categories alongside numeric features.
4. **Optional temporal model**, only when records can be ordered and the history is constructed without future information.

Models must use the same chronological train/validation/test design and the same prediction-time rules. Report PR AUC, ROC AUC, recall at a precision threshold selected on validation data, and calibration diagnostics. Test results must not be used to tune the model or choose the operating threshold.

## 7. Offline rescheduling analysis

The rescheduling analysis evaluates candidate schedule shifts under declared assumptions. It may compare predicted risks for shifts such as −60, −30, −15, +15, +30, and +60 minutes, subject to reasonable operating-hour bounds. Time-derived features must be recalculated consistently for each candidate schedule.

The analysis does not establish that rescheduling will cause an actual reduction in delays. The supplied data does not model the full operational constraints of crews, aircraft rotations, gates, airport capacity, passenger connections, or network-wide knock-on effects. Any output must be described as model-based risk comparison, not an operationally validated schedule.

## 8. Team responsibilities

| Contributor | Primary responsibility | Main output |
|---|---|---|
| **1 — Data preparation** | Load and profile the five CSVs, clean data, define the delay label, audit exclusions and data quality | Reproducible labelled dataset and data-quality summary |
| **2 — Feature engineering** | Build schedule, calendar, route, safe weather, and lagged historical features; enforce feature availability | Versioned feature matrix and feature definitions |
| **3 — Classical models** | Implement and tune Logistic Regression and XGBoost | Reproducible classical baselines and predictions |
| **4 — Deep learning** | Implement the categorical-embedding MLP; evaluate whether a temporal model is justified | Neural-model code, configuration, and predictions |
| **5 — Evaluation and rescheduling** | Own the chronological split, metrics, threshold/calibration analysis, common comparison, offline rescheduling, and final report integration | Evaluation results and rescheduling report |

Contributors own their components; shared interfaces and data contracts must be agreed before integration. Contributor 5 owns common evaluation and rescheduling, not the implementation of everyone else's models.

## 9. Technology stack

The proposed stack is Python, pandas (or Polars if the team deliberately standardizes on it), scikit-learn, XGBoost, PyTorch, and matplotlib. The project is an offline analysis pipeline. A web service, database-backed application, desktop application, or live airline operations integration is not part of the current scope.

## 10. Initial repository layout

```text
.
├── README.md
├── data/
│   ├── raw/                 # original CSV files; never edit in place
│   └── processed/           # generated labelled data and feature matrices
├── src/
│   ├── data/                # loading, profiling, preparation, labels
│   ├── features/            # prediction-time-safe feature creation
│   ├── models/
│   │   ├── classical/       # Logistic Regression and XGBoost
│   │   └── neural/          # embedding MLP and optional temporal model
│   ├── evaluation/          # chronological splits, metrics, calibration
│   └── rescheduling/        # offline what-if analysis
├── tests/                   # data, feature, and model-contract checks
├── reports/                 # quality summaries, figures, evaluation results
├── artifacts/               # generated model artifacts; normally gitignored
└── docs/
    ├── Architecture Diagrams/
    ├── Product/
    └── Research/
```

This is the intended logical organization, not a claim that every directory or script already exists. Preserve the five raw CSV files and exclude large generated artifacts from version control unless the team has a deliberate reason to track them.

## 11. Getting started

The agreed interpreter is **Python 3.13** and the canonical dependency file is `requirements.txt` (frozen in `configs/project_contract.yaml`):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest          # baseline contract tests must pass
```

### Data preparation (Phase 1)

Regenerate the labelled dataset and data-quality report from the raw CSVs:

```bash
python -m src.data.prepare
```

Outputs: `data/processed/labelled_flights.csv` (one row per source flight with
`flight_row_id`, parsed schedule fields, the label and `label_status`;
`arr_delay` is kept for audit only), `data/processed/phase1_run_manifest.json`
(source fingerprints and code commit), and the JSON/Markdown quality report
under `reports/`. The measured quality report for the supplied files is
[reports/phase1_data_quality.md](reports/phase1_data_quality.md).

The raw input directory should contain the five supplied CSV files listed above. Before training models, run the data profile and label audit and review their counts. Do not treat the reference numbers in this README as values to hard-code into code; they are validation expectations for the supplied version of the files.

### Feature engineering (Phase 2)

Build the leakage-safe T−60 feature table after Phase 1:

```bash
python -m src.features.pipeline
```

This writes `data/processed/phase2_features.csv`, the versioned manifest at
`data/processed/phase2_feature_manifest.json`, and measured coverage at
`reports/phase2_feature_quality.json`. Only the manifest allowlist enters the
estimator; identifiers, target, cutoff, weather timestamp, and weather age are
audit fields outside `X`.

### Classical baselines (Phase 3)

Train the Logistic Regression and bounded-search XGBoost baselines:

```bash
python -m src.models.classical.train
```

The command fits preprocessing and models on the training window only, selects
the XGBoost candidate by validation PR AUC, and writes reloadable pipelines,
the split manifest, metadata, and validation prediction files under
`artifacts/classical/`. Test labels are deliberately not scored in Phase 3;
Phase 5 owns the final held-out evaluation.

### Shared evaluation and final comparison (Contributor 5)

After both baseline artifacts exist, run the frozen common evaluation:

```bash
python -m src.evaluation.run
```

This regenerates the canonical split manifest and summary, validates prediction
IDs and labels, freezes model and threshold decisions from validation data,
scores the held-out test set once, and generates ranking, calibration, and
confusion-matrix plots. Outputs are stored in `reports/evaluation/`; the final
interpretation and “Identify the Gap” section are in
`reports/contributor5_final_evaluation.md`.

No operational precision floor was stakeholder-approved. The current comparison
therefore uses maximum validation F1 as its documented operating-threshold rule
and does not present the contract's candidate 0.70 precision as a requirement.

## 12. Project documentation

- [Product Requirements Document](docs/Product/PRD.md)
- [Project and Modeling Design](docs/Product/Design.md)
- [Research and Dataset Notes](docs/Research/Research.md)
- [Use Case Diagram](docs/Architecture%20Diagrams/Use-Case-Diagram.md)
- [Entity Relationship Diagram](docs/Architecture%20Diagrams/ER-Diagram.md)
- [UML Class Diagram](docs/Architecture%20Diagrams/UML-Class-Diagram.md)
- [Project Contract (frozen Phase 0 decisions)](configs/project_contract.yaml)
- [Implementation Phases](docs/Engineering/Implementation-Phases.md)
- [Phase 0 Inventory and Environment Report](reports/phase0_repository_inventory.md)

## 13. Current limitations

- The dataset covers one calendar year and flights departing from three NYC-area airports; generalization to other years, airports, or airlines is not established.
- The target is observed arrival delay, not cancellation risk or departure-delay risk.
- Missing arrival delay is not a negative label.
- Weather data consists of historical observations, not forecast products.
- Some destination airport and aircraft references do not match the provided lookup files.
- Offline rescheduling results are not causal or operationally validated.
