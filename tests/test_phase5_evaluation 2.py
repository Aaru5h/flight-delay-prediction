"""Unit and contract tests for Phase 5 AML evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration import fit_probability_calibrator
from src.evaluation.metrics import compute_ece, evaluate_predictions
from src.evaluation.threshold import select_operating_thresholds


def test_threshold_selection_rejects_non_validation_split():
    """Ensure threshold selection raises ValueError if non-validation split is passed."""
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.2, 0.8, 0.3, 0.7])

    with pytest.raises(ValueError, match="prohibited on 'test' data"):
        select_operating_thresholds(y, probs, split_name="test")

    with pytest.raises(ValueError, match="prohibited on 'train' data"):
        select_operating_thresholds(y, probs, split_name="train")


def test_metrics_calculation_valid_bounds():
    """Test that metrics return finite values within legitimate statistical bounds."""
    y = np.array([0, 0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.7, 0.85, 0.4, 0.9])

    metrics = evaluate_predictions(y, probs, threshold=0.5)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["total_evaluated"] == 6


def test_ece_perfect_calibration_is_zero():
    """ECE should be near zero for perfectly calibrated predictions."""
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    ece = compute_ece(y, probs, n_bins=5)
    assert np.isclose(ece, 0.0, atol=1e-5)


def test_calibrator_fitting():
    """Test that validation probability calibrator fits and maps probabilities correctly."""
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])

    calibrator = fit_probability_calibrator(y, probs)
    cal_probs = calibrator.predict(probs)

    assert len(cal_probs) == len(probs)
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    # Monotonicity check
    assert np.all(np.diff(cal_probs) >= -1e-7)
