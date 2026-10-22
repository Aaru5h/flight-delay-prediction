# Flight Delay Prediction and Rescheduling

## Product Requirements Document (PRD)

**Document version:** 1.0  
**Status:** Initial project specification  
**Project type:** Applied machine-learning research pipeline  
**Dataset period:** Calendar year 2013  
**Primary prediction point:** 60 minutes before scheduled departure (T−60)  
**Target:** Arrival delay of at least 15 minutes

---

## 1. Project Overview

### 1.1 Description

This project develops a prediction pipeline for identifying flights that are likely to arrive substantially late. It uses historical flight schedules and outcomes, carrier and route information, calendar attributes, weather observations available before the prediction cutoff, and historical delay statistics derived only from earlier dates. The project compares Logistic Regression, XGBoost, and a compact neural network with learned embeddings for categorical variables. A temporal model is an optional experiment, not a mandatory dependency of the initial model comparison.

A second analytical component explores whether changing a scheduled departure time changes the model's predicted delay risk. This component is an offline what-if analysis whose assumptions and limitations must be reported.

### 1.2 Problem Statement

Flight delay is influenced by interacting factors such as scheduled departure time, route, carrier, weather, and prior operating conditions. Manually summarized route or congestion features may not represent every nonlinear interaction. The project will compare a clear baseline with tree-based and neural models while preventing post-departure information from entering pre-departure predictions.

### 1.3 Vision

Build a reproducible and fairly evaluated flight-delay risk modeling pipeline that demonstrates how operational tabular features and learned categorical/temporal patterns can be compared under a strict prediction-time contract.

### 1.4 Objectives

1. Prepare a reliable supervised dataset and document all target exclusions.
2. Establish feature definitions that respect the selected prediction time.
3. Build comparable Logistic Regression, XGBoost, and embedding-MLP models.
4. Evaluate ranking quality, operational recall/precision, and probability calibration on future dates.
5. Run a constrained, explicitly non-causal offline rescheduling analysis.
6. Produce repeatable code, plots, metrics, and project documentation.

## 2. Scope

### 2.1 In Scope

- Profiling and preparing the supplied flight, airline, airport, aircraft, and weather CSV files.
- Defining `substantial_delay` from observed arrival delay.
- Engineering scheduled-time, calendar, carrier, route, safe weather, and lagged historical features.
- Comparing Logistic Regression, XGBoost, and an embedding MLP.
- Considering a temporal model only if the sequence can be formed from correctly ordered past information.
- Chronological validation and testing.
- PR AUC, ROC AUC, recall at a validation-selected precision threshold, and calibration evaluation.
- Offline rescheduling what-if analysis and an explanation of its assumptions.
- A report documenting the dataset, model comparison, limitations, and findings.

### 2.2 Out of Scope

- Live predictions for active flights or production airline operations.
- Integrations with airline dispatch, airport systems, crew planning, gates, or reservation systems.
- A web application, mobile app, desktop app, or public inference API.
- Predicting cancellations as a separate target; the supplied files lack an explicit cancellation field.
- Claiming that model-selected schedule shifts cause delay reduction.
- Generalizing performance to airports, years, or carriers not represented by the supplied data without a separate external evaluation.

## 3. Dataset and Target Requirements

### 3.1 Source Files

| File | Rows | Main content |
|---|---:|---|
| `nyc_flights.csv` | 336,776 | Flight schedules, routes, carriers, observed flight outcomes |
| `nyc_airlines.csv` | 16 | Carrier lookup |
| `nyc_airports.csv` | 1,458 | Airport reference attributes |
| `nyc_planes.csv` | 3,322 | Aircraft reference attributes keyed by tail number |
| `nyc_weather.csv` | 26,115 | Hourly weather observations at NYC origin airports |

Flight dates cover 1 January through 31 December 2013. `nyc_flights.csv` has 327,346 observed `arr_delay` values and 9,430 missing values. The known-label class counts are 80,100 positive and 247,246 negative. These are reference profiles for the supplied files; the loader must compute rather than hard-code them.

### 3.2 Target Label

- `substantial_delay = 1` if `arr_delay >= 15` minutes.
- `substantial_delay = 0` if `arr_delay < 15` minutes.
- A missing `arr_delay` produces an unknown target and is excluded from supervised classification.

The target threshold follows the conventional 15-minute late-arrival definition used in US on-time performance reporting. Actual `arr_delay` is target-construction and audit data only; it is prohibited from the feature matrix.

### 3.3 Data Quality and Join Constraints

- The source files contain no explicit cancellation/diversion status field. Missing target records must not all be classified as cancellations.
- The airport lookup lacks destination rows for `BQN`, `PSE`, `SJU`, and `STT` in this supplied snapshot; downstream joins must preserve and report unmatched keys.
- The aircraft lookup is only a partial match for flight `tailnum` values. Aircraft identity is not an automatic feature: its availability at T−60 must be justified first.
- The weather table includes missing observations and missing values, especially `wind_gust` and `pressure`. Missingness must be measured and handled explicitly.
- Do not silently remove observations solely because an optional lookup value is missing.

## 4. Users and Stakeholders

This project is an analytical deliverable rather than an end-user application. Its primary stakeholders are:

- **Project team:** prepares data, builds models, and integrates results.
- **Faculty/reviewers:** evaluate technical correctness, reproducibility, experimental design, and interpretation.
- **Potential operations stakeholders:** can use the final report to understand the type of risk estimates the models produce, but must not treat this academic result as an operational tool.

## 5. Functional Requirements

### 5.1 Data Preparation

**FR-01 — Source loading:** The pipeline shall read the five CSV files without modifying raw inputs.

**FR-02 — Schema validation:** The pipeline shall verify required columns and report absent columns or incompatible types clearly.

**FR-03 — Data profile:** The pipeline shall report row counts, date range, duplicate checks, missingness, key coverage, and target balance.

**FR-04 — Target creation:** The pipeline shall construct the binary delay label using the agreed 15-minute threshold.

**FR-05 — Unlabelled records:** The pipeline shall exclude records with missing arrival delay from supervised classification and document the exclusion count and reason observable from the files.

### 5.2 Feature Engineering

**FR-06 — Prediction-time safety:** Every feature shall have a documented availability rule relative to T−60.

**FR-07 — Schedule and calendar features:** Features may describe scheduled departure/arrival times, date, seasonality, carrier, origin, destination, and route when known at the prediction cutoff.

**FR-08 — Weather as-of join:** Weather features shall use only origin-airport observations timestamped no later than T−60. The initial rule treats observations older than three hours as unavailable.

**FR-09 — Historical delay features:** Historical statistics shall be computed from prior calendar dates only, with no target-flight or same-day outcome leakage.

**FR-10 — Ambiguous aircraft features:** Aircraft identity shall be excluded unless the team documents evidence that it would be known at the selected prediction time.

### 5.3 Modeling

**FR-11 — Logistic Regression:** A reproducible, preprocessing-aware baseline shall be trained and evaluated.

**FR-12 — XGBoost:** A reproducible boosted-tree model shall use the common target, date split, and feature contract.

**FR-13 — Neural model:** A compact MLP shall support learned embeddings for airline, origin, and destination categorical features.

**FR-14 — Optional temporal experiment:** A temporal model may be attempted only where flight history is correctly ordered and all sequence inputs are available by the prediction time.

**FR-15 — Common predictions:** Every model shall emit prediction probabilities for the same evaluation examples with stable flight-record identifiers.

### 5.4 Evaluation and Rescheduling

**FR-16 — Chronological evaluation:** Training, validation, and test sets shall be split by date. A random row split is not acceptable as the main reported evaluation.

**FR-17 — Ranking metrics:** Report PR AUC and ROC AUC.

**FR-18 — Operational threshold:** Report recall at a stated precision requirement. Select the threshold using validation data only and state the required precision explicitly in the final report.

**FR-19 — Calibration:** Report a proper probability metric such as Brier score and at least one calibration diagnostic such as a reliability plot. Any calibration method must be fitted on training/validation data without using the test outcomes.

**FR-20 — Rescheduling analysis:** Compare baseline model risk with explicitly defined alternative scheduled times. Recalculate time-dependent inputs consistently and record constraints and assumptions.

**FR-21 — Limitations:** The final report shall state that the rescheduling analysis is non-causal and does not represent a complete operational schedule optimizer.

## 6. Non-Functional Requirements

- **Reproducibility:** Random seeds, split dates, feature definitions, model settings, and dependency versions must be recorded.
- **Leakage prevention:** Target and post-departure fields must not enter model inputs.
- **Chronology:** Historical aggregates and temporal sequences must obey event ordering.
- **Comparability:** Models must be evaluated on the same examples and with the same evaluation definitions.
- **Traceability:** Data exclusions, unmatched joins, experiment settings, and result artifacts must be documented.
- **Practicality:** The baseline pipeline should run on a typical modern laptop; GPU use is optional and primarily relevant to neural experiments.
- **Honest interpretation:** Predictive association must not be described as causal impact.

## 7. Experimental Design

The initial prediction timestamp is T−60. A proposed chronological partition is:

| Partition | Date range | Use |
|---|---|---|
| Train | 1 January–31 August 2013 | Fit preprocessing and models |
| Validation | 1 September–31 October 2013 | Tune models, choose threshold, and fit calibration where appropriate |
| Test | 1 November–31 December 2013 | One final estimate after decisions are frozen |

The team must inspect the date-level target distribution before locking this partition. If the partition changes, the change and its rationale must be documented before model comparison. Preprocessing, feature selection, threshold selection, and hyperparameter tuning must not use test outcomes.

## 8. Ownership and Deliverables

| Contributor | Ownership | Acceptance evidence |
|---|---|---|
| 1 — Data preparation | Loading, data profiling, delay label, exclusions, quality checks | Label counts and a reproducible cleaned dataset |
| 2 — Feature engineering | Schedule/calendar/route/weather/history features and leakage policy | Feature definitions, as-of joins, and leakage tests |
| 3 — Classical ML | Logistic Regression and XGBoost | Reproducible pipelines and aligned prediction outputs |
| 4 — Deep learning | Embedding MLP and optional temporal model assessment | Reproducible neural run and aligned predictions |
| 5 — Evaluation and rescheduling | Date split, metrics, calibration, model comparison, rescheduling report | Common comparison and documented what-if assumptions |

## 9. Acceptance Criteria

The project is ready for final review when:

1. The five source files load and are profiled by reproducible code.
2. The delay label and missing-label handling are tested and documented.
3. Every feature has a stated prediction-time availability rule.
4. The three required model families are compared on the same chronological holdout examples.
5. PR AUC, ROC AUC, recall at the chosen precision threshold, and calibration are reported.
6. Rescheduling scenarios, restrictions, risk differences, and non-causal limitations are stated.
7. The repository contains reproducible commands, fixed configuration artifacts, and project documentation.
