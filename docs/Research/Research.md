# Research Notes — Flight Delay Prediction and Rescheduling

## Executive Summary

This project studies whether flight schedules, carrier and route information, historical operating patterns, and pre-departure weather observations can predict substantial arrival delay for flights departing from EWR, JFK, and LGA during 2013. It compares a simple statistical baseline (Logistic Regression), a gradient-boosted model (XGBoost), and a compact neural network with learned categorical embeddings. A temporal method is optional and must be supported by correctly ordered, prediction-time-valid histories.

The primary methodological risk is leakage: flight datasets contain actual departure and arrival information, but those outcomes cannot be known at a pre-departure decision point. The project uses a baseline cutoff of 60 minutes before scheduled departure and requires all inputs, splits, and historical features to obey that cutoff. A separate offline rescheduling analysis compares model risk scores under candidate schedule shifts; the result is not a causal estimate of delay reduction.

## 1. Research Questions

1. How well can a pre-departure model rank flights by the probability of arriving at least 15 minutes late?
2. Does XGBoost improve over a Logistic Regression baseline using comparable inputs and a chronological split?
3. Does a compact embedding MLP provide useful gains over classical models for airline, origin, and destination categories?
4. Are the models' probabilities calibrated sufficiently for risk ranking and threshold-based decisions?
5. Under explicit hypothetical assumptions, how do predicted risk scores change when the scheduled departure time is shifted?

## 2. Dataset Description

The project uses five supplied CSV files consistent with the `nycflights13` dataset family. The current files cover 2013 and include flights departing from EWR, JFK, and LGA.

| File | Rows | Columns | Contents |
|---|---:|---:|---|
| `nyc_flights.csv` | 336,776 | 19 | Schedule, carrier, route, aircraft tail number, and observed flight outcomes |
| `nyc_airlines.csv` | 16 | 2 | Carrier code and airline name |
| `nyc_airports.csv` | 1,458 | 8 | Airport codes and geographic/reference attributes |
| `nyc_planes.csv` | 3,322 | 9 | Aircraft reference data by tail number |
| `nyc_weather.csv` | 26,115 | 15 | Hourly observations of temperature, dew point, humidity, wind, precipitation, pressure, and visibility |

### 2.1 Label availability and balance

Of 336,776 flight rows, 327,346 have a known `arr_delay` and 9,430 do not. Among rows with a known delay, 80,100 arrive at least 15 minutes late and 247,246 are below the threshold. The positive-class prevalence is 24.47% among labelled records.

There is no explicit cancellation or diversion flag in the supplied flights CSV. Of the 9,430 rows with missing `arr_delay`, 8,255 also have missing `dep_time`, while 1,175 have a non-missing `dep_time`. These are observable missingness patterns, not proof of a single cause. The pipeline must not convert these records into non-delayed examples.

### 2.2 Join coverage caveats

- All flight carrier codes in the current file have a corresponding entry in `nyc_airlines.csv`.
- All three origin airports have matching entries in `nyc_airports.csv`, but destination codes `BQN`, `PSE`, `SJU`, and `STT` are absent from that lookup. These account for 7,602 flight rows; retain the flights and record unmatched airport attributes.
- The flights file contains 334,264 non-missing `tailnum` values, but only 284,170 of those rows match the supplied aircraft lookup. The plane table is therefore optional and incomplete for this sample.
- A same-hour origin/time join can be found for 335,220 flight rows, but it is not automatically available at T−60. Use an as-of weather join, and only take observations at or before the cutoff.

## 3. Target Definition

The binary target is:

\[
\text{substantial\_delay} = \begin{cases}
1 & \text{if } arr\_delay \geq 15\text{ minutes} \\
0 & \text{if } arr\_delay < 15\text{ minutes}
\end{cases}
\]

Rows with missing `arr_delay` have unknown targets and are excluded from supervised classification. The 15-minute threshold is consistent with common US airline on-time reporting conventions. Arrival delay is used for target generation and auditing only.

## 4. Feature Availability and Leakage

The proposed prediction point is T−60 minutes before scheduled departure. Suitable candidates include scheduled time, calendar fields, carrier, origin, destination, route information, historical aggregates from earlier service dates, and recent origin-weather observations available by the cutoff.

The following fields are excluded from pre-departure model inputs because they represent observed outcomes or information unavailable at T−60: `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, and `air_time`. The `tailnum` field and derived aircraft characteristics require special scrutiny because the CSV records the aircraft associated with the realized flight, which may not have been assigned at the decision cutoff.

Historical delay statistics must be computed using prior calendar dates only. Aggregating delays across all rows before creating the split would leak future information even if the target column were later removed. Weather fields must be joined by the latest suitable observation at or before T−60; the observation at the scheduled departure hour can occur after the cutoff. The weather CSV contains observations, not weather forecast vintages, which limits the realism of any prospective claim.

## 5. Candidate Modeling Approaches

### 5.1 Logistic Regression

A transparent baseline that tests whether engineered operational features provide useful signal. It requires appropriate preprocessing for numeric and categorical fields. Its strengths are simplicity, speed, and easier interpretation; its limitations include a largely linear decision boundary unless nonlinear effects and interactions are engineered.

### 5.2 XGBoost

A gradient-boosted tree model that can learn nonlinear interactions between route, carrier, time-of-day, weather, and historical statistics. It often performs strongly on tabular data, but needs disciplined tuning, feature-availability checks, and validation-only model selection.

### 5.3 Embedding MLP

A compact neural network can learn embeddings for high-cardinality categorical concepts such as airline, origin, and destination, then combine these embeddings with numeric variables. The experiment should use clear regularization and early stopping because the dataset is a single year and has repeated structures by route and carrier.

### 5.4 Temporal Model (Optional)

A temporal model is justified only if histories are strictly time-ordered and built from records that were available before the prediction cutoff. If only aggregate historical features can be constructed reliably, those are preferable to a sequence model that accidentally includes future outcomes.

## 6. Experimental Design

The primary comparison must be chronological rather than a random row split. The initial proposed dates are train through 31 August 2013, validation from 1 September to 31 October, and test from 1 November to 31 December. Inspect target prevalence over these windows before freezing the boundaries; report any distribution shift rather than hiding it.

All models must use the same holdout rows and the same target definition. Report:

- **PR AUC** to focus on positive-class detection under class imbalance.
- **ROC AUC** as a complementary ranking measure.
- **Recall at a defined precision requirement**, with the decision threshold selected using validation data only.
- **Calibration diagnostics**, including Brier score and a reliability diagram or a clearly defined alternative.

A good ranking score does not guarantee calibrated probabilities. Calibration and threshold choices must not be fit against the test outcomes.

## 7. Calibration Background

Modern neural networks can be overconfident even when they rank cases well. Calibration analysis is therefore required for the neural model and useful for the other candidates. Guo et al. (2017) discuss modern neural-network calibration and temperature scaling; this is a relevant reference for deciding how to assess and, if needed, calibrate probabilities. The project should record the calibration method and fit it without accessing final test labels.

## 8. Offline Rescheduling Analysis

A selected model can estimate how its risk score changes when the scheduled departure is shifted by a limited offset. Candidate shifts should be explicit (for example, −60, −30, −15, +15, +30, and +60 minutes), and schedule-derived features must be recomputed consistently.

The counterfactual assumptions are strong: the supplied data does not represent crew/aircraft rotations, gate assignments, passenger connections, airport capacity, or all network propagation effects. Historical weather records also do not provide future forecasts for a newly shifted schedule. Therefore, results should be described as **model-based sensitivity analysis** and not as the delay reduction caused by a real rescheduling decision.

## 9. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Post-departure leakage | Unrealistically high performance | Explicit prediction cutoff, allowlist of features, automated leakage tests |
| Future data in historical aggregates | Inflated validation/test scores | Build aggregates from prior dates only and test boundary cases |
| Missing arrival delay treated as zero | Wrong target labels | Keep labels unknown and publish exclusion counts |
| One-year sample | Weak generalization to other years | State the scope limitation; recommend external temporal validation when available |
| Incomplete airport/aircraft joins | Missing or biased reference features | Preserve flights, log unmatched keys, and use fallback/missing handling |
| Weather observations mistaken for forecasts | Overstated pre-departure feasibility | Enforce timestamp cutoffs and identify the data as observed weather |
| Poor calibration | Misleading risk probabilities | Report calibration and fit any calibrator without test data |
| Rescheduling interpreted causally | Operational claims unsupported by data | Label the analysis non-causal and state omitted constraints |

## 10. Suggested Pre-Reads and References

1. **U.S. Bureau of Transportation Statistics (BTS)** — Airline On-Time Performance and reporting definitions. Use the official documentation when describing the 15-minute late-arrival threshold and interpreting operational fields.
2. **Wickham, H.** — `nycflights13` dataset documentation and related R package reference, for the meaning and origin of the flight, weather, airline, airport, and plane tables.
3. **Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017).** “On Calibration of Modern Neural Networks.” *Proceedings of the 34th International Conference on Machine Learning.* Relevant to calibration assessment and temperature scaling.

Dataset-specific counts and join observations in this document were measured from the five CSV files supplied for this project. Model scores and rescheduling conclusions are not yet available and must be reported only after the pipeline runs.
