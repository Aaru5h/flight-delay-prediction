# Identify the Gap: Applied Machine Learning Evaluation & Limitations

**Project:** Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning  
**Phase:** Phase 5 — AML Evaluation, Calibration, and Gap Analysis  
**Date:** October 2026  
**Evaluation Scope:** Baseline Logistic Regression vs. XGBoost on Chronological Split (Train: Jan–Aug 2013, Validation: Sep–Oct 2013, Held-Out Test: Nov–Dec 2013)

---

## 1. Executive Summary

This report synthesizes the empirical findings of the **Applied Machine Learning (AML)** stage (Part 1). Both models—**L2-Regularized Logistic Regression** and **Histogram Gradient Boosted Trees (XGBoost)**—were trained and evaluated under the strict **T−60 prediction contract** (all inputs known at least 60 minutes prior to scheduled departure, zero post-departure leakage, origin weather freshness <= 180 minutes, and strictly prior flight outcomes verified to have arrived before cutoff).

All operational thresholds and calibration decisions were selected exclusively on the **validation period** (55,628 flights) and frozen before running a single, un-tuned evaluation on the **held-out test set** (53,991 flights).

### Summary Evaluation Benchmark

| Split | Model | PR-AUC | ROC-AUC | Brier Score | ECE | Operating Thresh | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Validation** *(Prevalence: 15.70%)* | Logistic Regression | **0.3189** | **0.6815** | 0.1333 | 0.0526 | 0.6949 (Floor) | 70.91% | 0.89% | 0.0177 |
| | | | | | | 0.3177 (Best F1) | 27.83% | 49.63% | **0.3566** |
| | XGBoost | 0.2996 | 0.6714 | **0.1247** | 0.0381 | 0.5628 (Floor) | 58.59% | 1.33% | 0.0260 |
| | | | | | | 0.1518 (Best F1) | 27.98% | 47.62% | **0.3525** |
| **Held-Out Test** *(Prevalence: 25.83%)* | Logistic Regression | 0.4505 | **0.6905** | **0.1735** | 0.0714 | 0.6949 (Frozen Floor) | 73.65% | 0.88% | 0.0174 |
| | | | | | | 0.3177 (Frozen F1) | 46.32% | 44.53% | 0.4541 |
| | XGBoost | **0.4761** | 0.6896 | 0.1849 | 0.0829 | 0.5628 (Frozen Floor) | **75.44%** | 4.27% | 0.0809 |
| | | | | | | 0.1518 (Frozen F1) | 43.98% | **50.80%** | **0.4714** |

---

## 2. The Precision–Recall Dilemma and Missed Delays

### 2.1 The Operational Precision Floor
In real airline and airport operations, dispatchers and passenger-alert systems face severe costs from false alarms. Consequently, an operational requirement candidate was posited in the project contract: **achieve a precision floor of $\ge 70\%$**.

When evaluated on validation data:
- **Logistic Regression** reached 70.91% precision only at a high threshold of **0.6949**. At this threshold, the model achieved a **recall of only 0.89%**—catching just 78 out of 8,731 delayed flights and missing 99.11% of all delays.
- **XGBoost** could not reach 70% precision on validation data with any viable sample size ($\ge 1\%$ recall); its highest precision operating point was 58.59% at threshold 0.5628 (recall 1.33%).

### 2.2 Blindness to Delay Under Conservative Thresholds
When these frozen thresholds were evaluated on the held-out winter test set:
- **Logistic Regression (Threshold 0.6949):** Precision was 73.65%, but recall was **0.88%**. Out of **13,946 actual delays**, the model detected **123 flights** and missed **13,823 flights**.
- **XGBoost (Threshold 0.5628):** Precision was 75.44%, with a recall of **4.27%**. It detected **596 delays** and missed **13,350 delays**.

```text
Test Confusion Matrix at 70% Precision Floor (Logistic Regression):
                   Actual Delayed (1)    Actual On-Time (0)
Predicted Delayed:         123                    44           -> Precision: 73.65%
Predicted On-Time:      13,823                40,001           -> Recall:     0.88%
                        ^^^^^^
              (13,823 delayed flights missed!)
```

### 2.3 The F1 Operating Trade-Off
When optimizing for balanced F1 score on validation data:
- The optimal threshold drops to **0.3177** for Logistic Regression and **0.1518** for XGBoost.
- On the test set, this yields recall near 45–51% and precision around 44–46% (F1 = 0.45–0.47).
- While catching half of the delays, the false positive rate rises dramatically: for every delayed flight caught, approximately one on-time flight is falsely flagged as delayed.

---

## 3. Probability Calibration and Reliability Diagnostics

The reliability diagrams (saved to [`reports/figures/validation_reliability_diagram.png`](file:///Users/Deepak/Desktop/flight-delay-prediction/reports/figures/validation_reliability_diagram.png) and [`reports/figures/test_reliability_diagram.png`](file:///Users/Deepak/Desktop/flight-delay-prediction/reports/figures/test_reliability_diagram.png)) reveal clear structural calibration dynamics:

1. **Ranking vs. Probability Error:**
   - On the validation set, Logistic Regression attained a higher PR-AUC (0.3189 vs. 0.2996) and ROC-AUC (0.6815 vs. 0.6714).
   - Conversely, XGBoost achieved a superior Brier score (0.1247 vs. 0.1333) and lower ECE (0.0381 vs. 0.0526). XGBoost concentrates probability mass more effectively near 0.10–0.25 where the empirical validation density lies.
2. **Intermediate Bin Overconfidence:**
   - Both models exhibit calibration divergence in the predicted probability range between 0.30 and 0.60, where the models predict delay probabilities that exceed empirical rates by 8–15 percentage points.
3. **Post-Hoc Calibration Assessment:**
   - Fitting an isotonic probability calibrator on validation predictions marginally reduced validation Brier score (from 0.1333 to 0.1243 for Logistic Regression), but did not shift ranking ability (PR-AUC / ROC-AUC remain invariant under monotonic transforms).
   - Under seasonal shift, isotonic binning fitted on an autumn lull drifted when applied to winter conditions, confirming that calibration must be continually monitored across operational seasons.

---

## 4. Seasonal and Chronological Distribution Shift

A primary methodological vulnerability discovered in the tabular AML pipeline is its sensitivity to **temporal distribution drift**:

```text
Chronological Split Distribution:
  Train Period      (Jan 01 – Aug 31, 2013): 217,727 flights | Delay Prevalence: 26.4%
  Validation Period (Sep 01 – Oct 31, 2013):  55,628 flights | Delay Prevalence: 15.7%  (Autumn Lull)
  Test Period       (Nov 01 – Dec 31, 2013):  53,991 flights | Delay Prevalence: 25.8%  (Winter/Holiday Surge)
```

- **The Lull-to-Surge Distortion:** The validation period (September at 13.7% and October at 17.6%) represents the calmest period of the year in NYC airspace. In contrast, November and December bring severe winter storms, de-icing queues, and heavy holiday traffic (December delay prevalence surges to 33.55%).
- **Threshold Instability:** A model threshold tuned on a 15.7% prevalence dataset behaves drastically differently in a 25.8% prevalence environment. For example, the test PR-AUC numbers jump from ~0.30 to ~0.47 purely due to the higher baseline frequency of the positive class ($y=1$), while Brier score degrades from 0.125 to 0.185 because uncertainty increases.

---

## 5. Retrospective Weather Limitations

The dataset incorporates historical hourly METAR observations from Newark, JFK, and LaGuardia. Although our feature builder enforces strict T−60 cutoff and a 180-minute staleness cap, several prospective limitations remain:
1. **Retrospective Observations vs. Forecast Products:** In actual dispatch operations, decisions at T−60 are guided by Terminal Aerodrome Forecasts (TAF) and NOAA convective outlooks predicting future conditions over the next 2–6 hours, not past ground observations.
2. **Age of Observation:** An observation timestamped at T−60 can be up to 180 minutes old (mean age ~45 minutes). Rapidly moving cold fronts, squall lines, or sudden wind shifts occurring in the hour before departure cannot be detected.
3. **Destination and En-Route Omission:** Only origin airport weather is available in the dataset. Severe thunderstorms over destination hubs (e.g. Chicago O'Hare or Atlanta) or holding patterns along east coast corridors are entirely invisible to the feature matrix.

---

## 6. Identifying the Structural Gap: Why Classical AML Hits a Ceiling

Tabular machine learning models (Logistic Regression and XGBoost) have reached an empirical ceiling on this dataset ($\text{ROC-AUC} \approx 0.69$, $\text{PR-AUC} \approx 0.45\text{--}0.48$). The causes of this ceiling are structural:

### 1. The I.I.D. Independence Fallacy
Classical tabular models treat every flight record as an independent and identically distributed (i.i.d.) observation. In reality, commercial aviation is a **tightly coupled network**. A flight scheduled at 18:00 is heavily determined by:
- Whether the specific incoming aircraft (inbound leg) arrived on time at the gate.
- Gate occupancy conflicts and airport-level departure queue congestion.
- Cumulative crew flight-time limits.
Because tabular rows lack sequence representations of inbound aircraft rotations, classical models cannot track delay propagation along an aircraft's daily tail-number chain.

### 2. High-Cardinality Categorical Sparsity
The flight dataset contains 16 carriers, 3 origins, and 105 destinations (~220 active routes). One-hot encoding creates high-dimensional, orthogonal vectors with zero weight sharing between geographically or operationally similar routes (e.g. `JFK_LAX` and `EWR_SFO` are treated as completely unrelated).

---

## 7. Roadmap to Part 2: Deep Learning Architectures

The empirical limitations identified above directly motivate the deep learning architectures for Part 2:

1. **Entity Embedding MLP (Phase 4):**
   - Replace sparse one-hot matrices with compact, learned categorical embedding layers for `carrier`, `origin`, and `dest`.
   - Embeddings allow the neural network to project carriers and airports into continuous latent spaces where operational similarities (e.g. hub-and-spoke vs point-to-point airlines) can be discovered automatically.
2. **Sequential / Temporal Feasibility (Phase 4):**
   - Evaluate whether an ordered sequence model (LSTM, GRU, or Temporal Convolutional Network) tracking the preceding flight legs of an assigned aircraft or rolling airport departure queues can capture delay cascades without leaking future outcomes.
3. **Rescheduling Sensitivity (Phase 6):**
   - Use the calibrated model to evaluate what-if schedule shifts ($\pm 15, \pm 30, \pm 60$ minutes) while documenting the explicit non-causal boundaries of offline sensitivity analysis.
