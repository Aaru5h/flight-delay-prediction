"""Contributor 5 split, metric, and prediction-contract tests."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calibration_bins,
    compute_metrics,
    recall_at_precision_floor,
    select_max_f1_threshold,
)
from src.evaluation.split import build_split_manifest, validate_split_manifest
from src.evaluation.validation import validate_predictions


def _split_frame():
    return pd.DataFrame({
        "flight_row_id": ["a", "b", "c", "d", "e", "f"],
        "sched_date": ["2013-01-01", "2013-08-31", "2013-09-01", "2013-10-31", "2013-11-01", "2013-12-31"],
        "substantial_delay": [0, 1, 0, 1, 0, 1],
    })


def test_split_manifest_is_unique_nonoverlapping_and_chronological():
    manifest, summary = build_split_manifest(_split_frame())
    assert manifest["split"].tolist() == ["train", "train", "validation", "validation", "test", "test"]
    assert manifest["flight_row_id"].is_unique
    assert summary.set_index("split")["rows"].to_dict() == {"train": 2, "validation": 2, "test": 2}
    validate_split_manifest(manifest)


def test_split_rejects_duplicate_ids_and_date_overlap():
    duplicate = _split_frame()
    duplicate.loc[1, "flight_row_id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        build_split_manifest(duplicate)
    manifest, _ = build_split_manifest(_split_frame())
    manifest.loc[manifest["flight_row_id"].eq("c"), "sched_date"] = pd.Timestamp("2013-08-31")
    with pytest.raises(ValueError, match="more than one split"):
        validate_split_manifest(manifest)


def test_threshold_is_selected_from_validation_vectors_and_metrics_reuse_it():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    decision = select_max_f1_threshold(y, p)
    assert 0 <= decision["threshold"] <= 1
    metrics = compute_metrics(y, p, decision["threshold"])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_recall_at_precision_floor_reports_achievable_and_invalid_floor():
    result = recall_at_precision_floor([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.9)
    assert result["achievable"] and result["recall"] == 1.0
    with pytest.raises(ValueError, match="precision_floor"):
        recall_at_precision_floor([0, 1], [0.1, 0.9], 0)


def test_calibration_bins_reconcile_counts():
    bins = calibration_bins([0, 0, 1, 1], [0.05, 0.25, 0.75, 0.95], n_bins=4)
    assert bins["count"].sum() == 4
    assert bins["positive_rate"].between(0, 1).all()


def test_prediction_validation_joins_by_id_and_catches_population_or_label_errors():
    frame = _split_frame()
    manifest, _ = build_split_manifest(frame)
    val = manifest.loc[manifest["split"].eq("validation"), "flight_row_id"]
    predictions = pd.DataFrame({
        "flight_row_id": val.iloc[::-1].to_list(),
        "y_true": [1, 0],
        "y_probability": [0.8, 0.2],
        "split": "validation",
        "model_name": "m",
        "run_id": "r",
    })
    checked = validate_predictions(predictions, manifest, frame, "validation")
    assert len(checked) == 2
    with pytest.raises(ValueError, match="population mismatch"):
        validate_predictions(predictions.iloc[:1], manifest, frame, "validation")
    wrong = predictions.copy()
    wrong["y_true"] = 1 - wrong["y_true"]
    with pytest.raises(ValueError, match="labels do not agree"):
        validate_predictions(wrong, manifest, frame, "validation")
