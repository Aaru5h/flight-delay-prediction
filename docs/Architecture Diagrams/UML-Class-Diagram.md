# Flight Delay Prediction — UML Class Diagram

## 1. Overview

The class diagram describes the logical responsibilities of the analytical pipeline. It is a conceptual design, not a claim that these exact Python classes have already been implemented. The eventual module and class names can differ, provided that data contracts, ownership boundaries, and prediction-time rules remain consistent.

## 2. Class Diagram

```mermaid
classDiagram
    direction TB

    class FlightDataLoader {
        +load_flights(path)
        +load_airlines(path)
        +load_airports(path)
        +load_planes(path)
        +load_weather(path)
    }
    class DatasetProfiler {
        +profile_schema(data)
        +summarize_missingness(data)
        +check_keys(data)
        +summarize_label_balance(data)
    }
    class DelayLabeler {
        +threshold_minutes: int
        +create_label(flights)
        +separate_unlabelled(flights)
    }
    class DataQualityReport {
        +row_counts
        +date_range
        +missingness
        +unmatched_keys
        +label_balance
    }
    class FeatureBuilder {
        +prediction_cutoff_minutes: int
        +build_schedule_features(flights)
        +join_weather_asof(flights, weather)
        +build_historical_features(flights)
        +validate_feature_availability(features)
    }
    class FeatureManifest {
        +feature_name
        +source_fields
        +available_at
        +transformation
        +leakage_status
    }
    class ChronologicalSplitter {
        +train_start
        +validation_start
        +test_start
        +split_by_date(data)
        +write_split_manifest(data)
    }
    class ModelTrainer {
        <<abstract>>
        +fit(train_data)
        +predict_proba(data)
        +save_config()
    }
    class LogisticRegressionTrainer {
        +build_pipeline()
        +fit(train_data)
        +predict_proba(data)
    }
    class XGBoostTrainer {
        +fit(train_data)
        +predict_proba(data)
    }
    class EmbeddingMLPTrainer {
        +build_embeddings()
        +fit(train_data)
        +predict_proba(data)
    }
    class TemporalModelTrainer {
        +validate_sequence_order()
        +fit(train_data)
        +predict_proba(data)
    }
    class EvaluationRunner {
        +calculate_pr_auc()
        +calculate_roc_auc()
        +recall_at_precision()
        +evaluate_calibration()
        +compare_predictions()
    }
    class Calibrator {
        +fit(validation_predictions)
        +transform(probabilities)
    }
    class ReschedulingAnalyzer {
        +generate_candidate_shifts()
        +rebuild_shifted_features()
        +compare_risk_scores()
        +write_assumptions()
    }
    class ExperimentConfig {
        +prediction_cutoff
        +split_version
        +feature_version
        +random_seed
        +model_parameters
    }
    class PredictionArtifact {
        +flight_record_id
        +delay_probability
        +predicted_class
        +experiment_id
    }

    FlightDataLoader --> DatasetProfiler : provides raw tables
    DatasetProfiler --> DataQualityReport : creates
    FlightDataLoader --> DelayLabeler : provides flight rows
    DelayLabeler --> FeatureBuilder : provides labelled flight records
    FeatureBuilder --> FeatureManifest : documents features
    FeatureBuilder --> ChronologicalSplitter : provides ordered records
    ChronologicalSplitter --> ModelTrainer : defines data partitions
    ModelTrainer <|-- LogisticRegressionTrainer
    ModelTrainer <|-- XGBoostTrainer
    ModelTrainer <|-- EmbeddingMLPTrainer
    ModelTrainer <|-- TemporalModelTrainer
    ModelTrainer --> ExperimentConfig : configured by
    ModelTrainer --> PredictionArtifact : emits
    PredictionArtifact --> EvaluationRunner : evaluated by
    EvaluationRunner --> Calibrator : validation calibration
    EvaluationRunner --> ReschedulingAnalyzer : supplies selected model
```

## 3. Class Responsibilities

### Data Preparation — Contributor 1

- **`FlightDataLoader`** reads the five CSV sources and preserves raw inputs.
- **`DatasetProfiler`** computes schemas, date coverage, missingness, duplicate checks, key coverage, and label counts.
- **`DelayLabeler`** creates the binary target from `arr_delay` and separates records without a target.
- **`DataQualityReport`** stores measured counts and observed data issues for handoff.

### Feature Engineering — Contributor 2

- **`FeatureBuilder`** produces schedule, calendar, route, as-of weather, and lagged historical features.
- **`FeatureManifest`** records each feature's source, transformation, prediction-time availability, and leakage status.

### Classical Models — Contributor 3

- **`LogisticRegressionTrainer`** owns the baseline preprocessing and classifier pipeline.
- **`XGBoostTrainer`** owns the boosted-tree model and its configuration.

### Deep Learning — Contributor 4

- **`EmbeddingMLPTrainer`** owns the categorical-embedding MLP.
- **`TemporalModelTrainer`** is optional. It should be implemented only if a valid time-ordered sequence is available.

### Evaluation and Rescheduling — Contributor 5

- **`ChronologicalSplitter`** creates and documents the agreed date boundaries.
- **`EvaluationRunner`** calculates common metrics and compares predictions on the same held-out rows.
- **`Calibrator`** fits any calibration transformation using permitted data, without final test labels.
- **`ReschedulingAnalyzer`** creates candidate schedule shifts, rebuilds affected features, compares risk scores, and records assumptions.

## 4. Important Design Rules

- `DelayLabeler` must not provide `arr_delay` as a feature field to a trainer.
- `FeatureBuilder` must reject post-departure fields and validate the T−60 cutoff.
- Historical features must be based only on completed prior dates.
- The temporal trainer is optional; its inputs must be strictly ordered and available at prediction time.
- All trainers must emit probabilities aligned to the same stable flight identifiers for evaluation.
- Rescheduling operates on candidate feature rows and model probabilities; it does not claim to model the full airline operating network.
