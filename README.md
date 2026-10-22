# Flight Delay Prediction and Rescheduling

**Project title:** Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning  
**Project type:** Offline machine-learning research pipeline  
**Data:** NYC-origin flights and related reference/weather tables for calendar year 2013  
**Target:** Arrival delay of at least 15 minutes  
**Team:** Five contributors with explicit end-to-end ownership

## 1. Project overview

This project predicts whether a scheduled flight departing from New York City's three major airports—EWR, JFK, or LGA—will arrive at least 15 minutes late. It uses the supplied historical flight, airline, airport, aircraft, and hourly weather CSV files. The modeling comparison covers a Logistic Regression baseline, XGBoost, and a compact neural network with learned categorical embeddings. A sequence/temporal model is an additional experiment only if it can be evaluated with defensible time ordering and without future-information leakage.

After choosing a model using validation data, the team will perform an **offline rescheduling what-if analysis**. It compares predicted risk for a set of hypothetical schedule shifts under explicit constraints. It is not a causal estimate, operational flight plan, or claim that moving a real flight will reduce delays.

The project is a research and local-execution pipeline. It is not a production API, web app, desktop app, airline operations system, or live prediction service.

## 2. The actual supplied data

These file profiles are measured from the five CSVs supplied with the project. The pipeline must recheck them each time it runs; these are expected reference counts, not permission to hard-code row counts.

| File | Rows | Columns | Meaning |
|---|---:|---:|---|
| `nyc_flights.csv` | 336,776 | 19 | Scheduled flights from EWR, JFK, and LGA in 2013, including actual outcomes where present |
| `nyc_airlines.csv` | 16 | 2 | Carrier-code lookup |
| `nyc_airports.csv` | 1,458 | 8 | Airport reference attributes |
| `nyc_planes.csv` | 3,322 | 9 | Aircraft reference attributes by tail number |
| `nyc_weather.csv` | 26,115 | 15 | Hourly weather observations for the three origin airports |

The flight dates cover 2013-01-01 through 2013-12-31. Of the 336,776 flight records, 327,346 have a non-missing `arr_delay` and can be labelled. The remaining 9,430 have no valid arrival-delay label and must be excluded from supervised classification while their reasons are reported. Among labelled records, 80,100 are positive (`arr_delay >= 15`, 24.47%) and 247,246 are negative (75.53%).

**Do not call all 9,430 rows cancelled flights.** The supplied flight CSV has no explicit cancellation or diversion flag. There are 8,255 rows with both `dep_time` and `arr_delay` missing, and another 1,175 with `arr_delay` missing despite a non-missing `dep_time`. Report observable facts rather than assigning unsupported reasons.

See [Dataset Reference](docs/Data/Dataset-Reference.md) and [Data Dictionary](docs/Data/Data-Dictionary.md) for the full profile, missingness, joins and fields.

## 3. Prediction and evaluation contract

- **Decision point:** 60 minutes before scheduled departure (`T−60`). Contributors must use this same cutoff in features, model comparisons and the what-if analysis.
- **Label:** `substantial_delay = 1` when observed `arr_delay >= 15` minutes; otherwise `0`. Negative delay values and delays below 15 minutes are class `0`.
- **No label leakage:** actual `arr_delay` is used to construct `y` only. Actual `dep_time`, `dep_delay`, `arr_time`, `air_time`, and other post-decision outcomes cannot be model inputs.
- **Weather availability:** use the most recent origin-airport weather observation whose timestamp is no later than `T−60`; mark it missing if it is more than three hours older than the cutoff. The same-hour schedule join is useful for data profiling, but is not itself valid at `T−60`.
- **Historical outcomes:** delay aggregates must use completed prior calendar dates only; never use the target flight or same-day outcomes to construct historical features.
- **Split:** train on dates through 2013-08-31, validate on 2013-09-01–2013-10-31, and test on 2013-11-01–2013-12-31. Contributor 5 owns the frozen split manifest and must publish counts before model comparisons.
- **Test discipline:** threshold choice, hyperparameter tuning, calibration selection and model selection use training/validation data only. Test data is evaluated after decisions are locked.

The proposed split has a notable target-rate shift: the labelled training portion has 57,423 positives of 217,727 rows (26.37%); validation has 8,731 of 55,628 (15.70%); test has 13,946 of 53,991 (25.83%). This must be discussed rather than hidden. These figures describe the dataset split, not model performance.

## 4. Five-contributor ownership

| Contributor | Primary ownership | Main deliverables |
|---|---|---|
| **1 — Data pipeline and label** | Loading the five source files, schema/data-quality checks, target creation and exclusion audit | Reproducible labelled dataset, data profile, column guide, preparation tests |
| **2 — Feature engineering** | Schedule/calendar features, reference joins, safe weather as-of join, lagged historical features and leakage audit | Versioned feature matrix, feature manifest and feature tests |
| **3 — Classical ML** | Logistic Regression and XGBoost baselines using the shared features and split | Both trained pipelines, validation/test prediction artifacts, configs and model notes |
| **4 — Deep learning** | MLP with airline/origin/destination embeddings; optional temporal experiment if justified | Neural model, training code/config, predictions, model card and comparison notes |
| **5 — Evaluation and rescheduling** | Shared chronological split contract, common metrics, calibration/threshold analysis, fair comparison, what-if rescheduling and final report integration | Frozen split manifest, evaluation package, selected-model rationale, rescheduling report and reproduction checklist |

Contributor 5 owns the evaluation infrastructure but does **not** rewrite the other contributors' models. Contributors 3 and 4 each provide predictions in the agreed common format. Contributor 2 owns feature calculations; Contributor 5 audits their cutoff compliance. Contributor 1 owns target data quality but does not create model features. See [Team Ownership](docs/Team-Ownership.md).

## 5. Recommended repository layout

```text
.
├── README.md
├── data/
│   ├── raw/                       # supplied CSVs; read-only inputs
│   └── processed/                 # generated labelled/features data
├── configs/                       # split, features and model configs
├── src/
│   ├── data/                      # Contributor 1
│   ├── features/                  # Contributor 2
│   ├── models/
│   │   ├── classical/             # Contributor 3
│   │   └── neural/                # Contributor 4
│   ├── evaluation/                # Contributor 5
│   └── rescheduling/              # Contributor 5
├── tests/
├── artifacts/                     # generated model artifacts; usually gitignored
├── reports/                       # quality reports, metrics and figures
└── docs/
    ├── Architecture Diagrams/
    ├── Data/
    ├── Models/
    ├── Product/
    └── Research/
```

This is the agreed logical layout. OpenCode must inspect the actual repository before creating files. If equivalent modules already exist, extend them instead of building duplicate implementations.

## 6. Setup and local execution

Use the Python version already established in the repository. If no version is specified, the team should standardize on Python 3.11 and record it. Use one canonical dependency definition (`requirements.txt` or `pyproject.toml`). Do not create both without a deliberate reason.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

After dependencies are established, the expected workflow is:

```bash
# Proposed entry points; update these after implementation if the repo uses different commands.
python -m src.data.profile --data-dir data/raw --out reports/data_quality_summary.json
python -m src.data.prepare --data-dir data/raw --out data/processed/flights_labeled.parquet
python -m src.features.build --config configs/features.yaml
python -m src.evaluation.run --config configs/evaluation.yaml
python -m src.rescheduling.run --config configs/rescheduling.yaml
```

Do not claim these commands exist until the corresponding scripts have been implemented and tested. Never edit raw data in place. Do not hard-code the reference row counts into the pipeline as runtime assumptions.

## 7. Core deliverables

1. Data-quality report and reproducible labelled dataset.
2. Leakage-safe feature matrix and feature-availability manifest.
3. Logistic Regression, XGBoost and neural MLP model runs with saved configurations.
4. A fair chronological comparison reporting PR AUC, ROC AUC, recall at a preselected operational precision requirement, and calibration diagnostics.
5. An offline rescheduling what-if report with candidate-shift assumptions and non-causal limitations.
6. A final technical report, diagrams, model card and instructions to reproduce the results.

## 8. Documentation map

- [Product requirements](docs/Product/PRD.md)
- [Product/report design](docs/Product/Design.md)
- [Research](docs/Research/Research.md)
- [Team ownership and handoffs](docs/Team-Ownership.md)
- [Data profile](docs/Data/Dataset-Reference.md)
- [Data dictionary](docs/Data/Data-Dictionary.md)
- [Preparation and target contract](docs/Data/Data-Preparation.md)
- [Feature availability and leakage controls](docs/Data/Feature-Availability.md)
- [Data-quality report template](docs/Data/Data-Quality-Report.md)
- [Modeling and evaluation](docs/Models/Modeling-and-Evaluation.md)
- [Model card](docs/Models/Model-Card.md)
- [Rescheduling analysis](docs/Models/Rescheduling-Analysis.md)
- [Implementation phases](docs/Implementation-Phases.md)
- [OpenCode prompts](docs/OpenCode-Phase-Prompts.md)
- [Testing and security](docs/Testing-and-Security.md)
- [Module and CLI conventions](docs/API-Conventions.md)
- [Reproducibility and handoff](docs/Deployment-and-Launch.md)
- [Architecture diagrams index](docs/Architecture%20Diagrams/README.md)

## 9. Sources and provenance

The supplied schema and values match the `nycflights13` family of data. The package documentation describes flights departing EWR, JFK and LGA in 2013 and gives source-column definitions: [flights documentation](https://nycflights13.tidyverse.org/reference/flights.html), [weather documentation](https://nycflights13.tidyverse.org/reference/weather.html), and [package repository](https://github.com/tidyverse/nycflights13). The 15-minute arrival threshold is consistent with the on-time definition used in US aviation reporting; see [BTS Airline On-Time Performance](https://www.bts.gov/explore-topics-and-geography/topics/airline-time-performance-and-causes-flight-delays). Calibration background: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).

The dataset package is distributed as CC0/public-domain data, but record the actual source/version used. The project code still needs an explicit repository license decision.
